"""FastAPI REST 服务入口。

HTTP 层薄封装：
- 接收 URL，创建 job，返回 job_id
- /resolve 端点：解析抖音链接，返回无水印直链（供 iPhone Shortcuts 使用）
- 实际下载委托给 cli.main.download_url 的简化复用

fastapi/uvicorn 是**可选**依赖。若未安装，导入本模块会 ImportError。
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from auth import CookieManager
from config import ConfigLoader
from control import QueueManager, RateLimiter, RetryHandler
from core import DouyinAPIClient, URLParser, DownloaderFactory
from core.downloader_base import BaseDownloader
from server.jobs import JobManager
from storage import FileManager
from utils.logger import setup_logger
from utils.validators import is_short_url, normalize_short_url

logger = setup_logger("REST")

# Gallery aweme_type values (mirrors BaseDownloader._GALLERY_AWEME_TYPES)
_GALLERY_AWEME_TYPES = {2, 68, 150}

# Paths that never require API key authentication
_PUBLIC_PATHS = frozenset({"/api/v1/health", "/docs", "/openapi.json", "/redoc"})


class DownloadRequest(BaseModel):
    url: str


class JobResponse(BaseModel):
    job_id: str
    status: str
    url: str


class _ServerDeps:
    """跨请求复用的重量级依赖。

    REST 服务在进程生命周期内只需要一份 FileManager / RateLimiter / RetryHandler /
    QueueManager / CookieManager；每个请求重新构造既浪费又会触发文件系统 mkdir。
    DouyinAPIClient 由于持有 aiohttp.ClientSession，依旧按请求创建，避免跨请求泄漏
    连接状态或触发 "Session is closed" 错误。
    """

    def __init__(self, config: ConfigLoader):
        self.config = config
        self.cookie_manager = CookieManager()
        self.cookie_manager.set_cookies(config.get_cookies())
        self.file_manager = FileManager(config.get("path"))
        self.rate_limiter = RateLimiter(
            max_per_second=float(config.get("rate_limit", 2) or 2)
        )
        self.retry_handler = RetryHandler(
            max_retries=int(config.get("retry_times", 3) or 3)
        )
        self.queue_manager = QueueManager(
            max_workers=int(config.get("thread", 5) or 5)
        )


def _get_api_key(config: ConfigLoader) -> Optional[str]:
    """Resolve API key from config or environment."""
    server_cfg = config.get("server") or {}
    if isinstance(server_cfg, dict):
        key = server_cfg.get("api_key")
        if key:
            return str(key)
    return os.environ.get("DOUYIN_API_KEY") or None


def _detect_media_type(aweme_data: Dict[str, Any]) -> str:
    """Detect whether aweme is a video or gallery."""
    if (
        aweme_data.get("image_post_info")
        or aweme_data.get("images")
        or aweme_data.get("image_list")
    ):
        return "gallery"
    aweme_type = aweme_data.get("aweme_type")
    if isinstance(aweme_type, int) and aweme_type in _GALLERY_AWEME_TYPES:
        return "gallery"
    return "video"


def _extract_first_url(source: Any) -> Optional[str]:
    """Extract the first URL from various Douyin API response structures."""
    if isinstance(source, dict):
        url_list = source.get("url_list")
        if isinstance(url_list, list) and url_list:
            first_item = url_list[0]
            if isinstance(first_item, str) and first_item:
                return first_item
    elif isinstance(source, list) and source:
        first_item = source[0]
        if isinstance(first_item, str) and first_item:
            return first_item
    elif isinstance(source, str) and source:
        return source
    return None


def _pick_first_media_url(*sources: Any) -> Optional[str]:
    for source in sources:
        candidate = _extract_first_url(source)
        if candidate:
            return candidate
    return None


def _pick_highest_quality_play_addr(
    video: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Pick the highest bitrate play_addr from video.bit_rate list."""
    bit_rates = video.get("bit_rate") if isinstance(video, dict) else None
    if not isinstance(bit_rates, list) or not bit_rates:
        return None
    best = None  # type: Optional[Dict[str, Any]]
    best_score = -1
    for entry in bit_rates:
        if not isinstance(entry, dict):
            continue
        play_addr = entry.get("play_addr")
        if not isinstance(play_addr, dict):
            continue
        try:
            bit_rate = int(entry.get("bit_rate") or 0)
        except (TypeError, ValueError):
            bit_rate = 0
        width = int(play_addr.get("width") or entry.get("width") or 0)
        score = bit_rate * 10_000 + width
        if score > best_score:
            best_score = score
            best = play_addr
    return best


def _build_no_watermark_url(
    aweme_data: Dict[str, Any], api_client: DouyinAPIClient
) -> Optional[str]:
    """Extract the best no-watermark video URL from aweme data."""
    video = aweme_data.get("video", {})
    play_addr = _pick_highest_quality_play_addr(video) or video.get(
        "play_addr", {}
    )
    url_candidates = [c for c in (play_addr.get("url_list") or []) if c]
    url_candidates.sort(key=lambda u: 0 if "watermark=0" in u else 1)

    fallback_candidate = None  # type: Optional[str]

    for candidate in url_candidates:
        parsed = urlparse(candidate)
        if parsed.netloc.endswith("douyin.com"):
            if "X-Bogus=" not in candidate:
                signed_url, _ua = api_client.sign_url(candidate)
                return signed_url
            return candidate
        fallback_candidate = candidate

    if fallback_candidate:
        return fallback_candidate

    uri = (
        play_addr.get("uri")
        or video.get("vid")
        or video.get("download_addr", {}).get("uri")
    )
    if uri:
        params = {
            "video_id": uri,
            "ratio": "1080p",
            "line": "0",
            "is_play_url": "1",
            "watermark": "0",
            "source": "PackSourceEnum_PUBLISH",
        }
        signed_url, _ua = api_client.build_signed_path("/aweme/v1/play/", params)
        return signed_url

    return None


def _collect_image_urls(aweme_data: Dict[str, Any]) -> List[str]:
    """Collect gallery image download URLs from aweme data."""
    image_urls = []
    image_post = aweme_data.get("image_post_info")
    gallery_items = []  # type: List[Any]
    if isinstance(image_post, dict):
        for key in ("images", "image_list"):
            candidate = image_post.get(key)
            if isinstance(candidate, list) and candidate:
                gallery_items = candidate
                break
    if not gallery_items:
        images = aweme_data.get("images") or aweme_data.get("image_list") or []
        if isinstance(images, list):
            gallery_items = images

    for item in gallery_items:
        if not isinstance(item, dict):
            continue
        image_url = _pick_first_media_url(
            item.get("download_url"),
            item.get("download_addr"),
            item.get("download_url_list"),
            item,
            item.get("display_image"),
            item.get("owner_watermark_image"),
        )
        if image_url:
            image_urls.append(image_url)

    # Deduplicate
    seen = set()  # type: set
    deduped = []  # type: List[str]
    for url in image_urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


async def _resolve_media_urls(
    url: str, deps: _ServerDeps
) -> Dict[str, Any]:
    """Resolve a Douyin URL into direct no-watermark media URLs.

    Returns media type, description, author info, and direct CDN URLs
    without downloading any content.
    """
    async with DouyinAPIClient(deps.cookie_manager.get_cookies()) as api_client:
        if is_short_url(url):
            resolved = await api_client.resolve_short_url(normalize_short_url(url))
            if not resolved:
                raise RuntimeError("Failed to resolve short URL: %s" % url)
            url = resolved

        parsed = URLParser.parse(url)
        if not parsed:
            raise RuntimeError("Unsupported URL: %s" % url)

        url_type = parsed.get("type")
        aweme_id = parsed.get("aweme_id") or parsed.get("note_id")
        if not aweme_id:
            raise RuntimeError("Could not extract content ID from URL: %s" % url)

        aweme_data = await api_client.get_video_detail(aweme_id)
        if not aweme_data:
            raise RuntimeError("Failed to get content detail for: %s" % aweme_id)

        media_type = _detect_media_type(aweme_data)
        author = aweme_data.get("author", {})
        desc = (aweme_data.get("desc", "") or "").strip()

        result = {
            "media_type": media_type,
            "aweme_id": aweme_id,
            "description": desc,
            "author": author.get("nickname", ""),
            "urls": [],
        }  # type: Dict[str, Any]

        if media_type == "video":
            video_url = _build_no_watermark_url(aweme_data, api_client)
            if video_url:
                result["urls"] = [{"url": video_url, "type": "video"}]
            else:
                raise RuntimeError(
                    "No playable video URL found for: %s" % aweme_id
                )
        elif media_type == "gallery":
            image_urls = _collect_image_urls(aweme_data)
            result["urls"] = [
                {"url": u, "type": "image"} for u in image_urls
            ]
            if not image_urls:
                raise RuntimeError(
                    "No gallery images found for: %s" % aweme_id
                )

        # Include cover URL for convenience
        cover_url = _extract_first_url(
            aweme_data.get("video", {}).get("cover")
        )
        if cover_url:
            result["cover_url"] = cover_url

        # Include recommended headers for downloading from CDN
        result["download_headers"] = {
            "Referer": "https://www.douyin.com/",
            "User-Agent": (
                "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 "
                "Mobile/15E148 Safari/604.1"
            ),
        }

        return result


async def _execute_download(url: str, deps: _ServerDeps) -> Dict[str, int]:
    """Run download and return counts (used by async job endpoint)."""
    async with DouyinAPIClient(deps.cookie_manager.get_cookies()) as api_client:
        if is_short_url(url):
            resolved = await api_client.resolve_short_url(normalize_short_url(url))
            if not resolved:
                raise RuntimeError("Failed to resolve short URL: %s" % url)
            url = resolved

        parsed = URLParser.parse(url)
        if not parsed:
            raise RuntimeError("Unsupported URL: %s" % url)

        downloader = DownloaderFactory.create(
            parsed["type"],
            deps.config,
            api_client,
            deps.file_manager,
            deps.cookie_manager,
            None,
            deps.rate_limiter,
            deps.retry_handler,
            deps.queue_manager,
            progress_reporter=None,
        )
        if downloader is None:
            raise RuntimeError("No downloader for url_type=%s" % parsed["type"])

        result = await downloader.download(parsed)
        return {
            "total": result.total,
            "success": result.success,
            "failed": result.failed,
            "skipped": result.skipped,
        }


def build_app(config: ConfigLoader) -> FastAPI:
    deps = _ServerDeps(config)
    api_key = _get_api_key(config)

    async def executor(url: str) -> Dict[str, int]:
        return await _execute_download(url, deps)

    server_cfg = config.get("server") or {}
    if not isinstance(server_cfg, dict):
        server_cfg = {}
    manager = JobManager(
        executor=executor,
        max_concurrency=int(config.get("thread", 2) or 2),
        max_jobs=int(server_cfg.get("max_jobs") or JobManager.DEFAULT_MAX_JOBS),
        job_ttl_seconds=float(
            server_cfg.get("job_ttl_seconds") or JobManager.DEFAULT_JOB_TTL_SECONDS
        ),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        await manager.shutdown()

    app = FastAPI(
        title="Douyin Downloader API",
        version="2.0",
        description="REST API for resolving Douyin media URLs and dispatching download jobs.",
        lifespan=lifespan,
    )
    app.state.job_manager = manager
    app.state.deps = deps
    app.state.api_key = api_key

    # --- API key auth middleware ---
    @app.middleware("http")
    async def api_key_middleware(request: Request, call_next):
        if api_key is None:
            return await call_next(request)
        path = request.url.path
        if path in _PUBLIC_PATHS:
            return await call_next(request)
        auth_header = request.headers.get("Authorization", "")
        if auth_header == "Bearer %s" % api_key:
            return await call_next(request)
        return JSONResponse(
            status_code=401,
            content={"detail": "unauthorized"},
        )

    # --- Endpoints ---

    @app.get("/api/v1/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/v1/resolve")
    async def resolve(req: DownloadRequest) -> Dict[str, Any]:
        """Resolve a Douyin URL into direct no-watermark media URLs.

        Designed for iPhone Shortcuts: POST a Douyin URL, get back the
        direct CDN link(s) which can be saved to Photos directly.
        """
        if not req.url:
            raise HTTPException(status_code=400, detail="url is required")
        try:
            result = await _resolve_media_urls(req.url, deps)
            return {"status": "success", **result}
        except RuntimeError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/v1/download", response_model=JobResponse)
    async def create_job(req: DownloadRequest) -> JobResponse:
        if not req.url:
            raise HTTPException(status_code=400, detail="url is required")
        job = await manager.submit(req.url)
        return JobResponse(job_id=job.job_id, status=job.status, url=job.url)

    @app.get("/api/v1/jobs/{job_id}")
    async def get_job(job_id: str) -> Dict[str, Any]:
        job = await manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.to_dict()

    @app.get("/api/v1/jobs")
    async def list_jobs() -> Dict[str, List[Dict[str, Any]]]:
        jobs = await manager.list_jobs()
        return {"jobs": [j.to_dict() for j in jobs]}

    return app


async def run_server(config: ConfigLoader, *, host: str, port: int) -> None:
    import uvicorn

    app = build_app(config)
    uv_config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(uv_config)
    await server.serve()
