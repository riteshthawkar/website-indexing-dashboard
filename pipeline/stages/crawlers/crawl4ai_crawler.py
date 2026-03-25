"""
Crawl4AI crawler stage.

This stage provides the active production crawler for the modular pipeline.
It persists HTML, markdown, downloads, page image metadata, and a resumable
runtime checkpoint so failed crawls can continue from the last known frontier.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import inspect
import json
import logging
import mimetypes
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

from pipeline.core.base import CrawlerStage, StageContext, StageResult
from pipeline.core.media import dedupe_media_items
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe, safe_filename
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

try:
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
    from crawl4ai.deep_crawling.filters import FilterChain, URLFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
except ImportError:  # pragma: no cover - exercised in validate_config
    AsyncWebCrawler = None
    
    class BrowserConfig:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class CrawlerRunConfig:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class BFSDeepCrawlStrategy:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class DefaultMarkdownGenerator:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class PruningContentFilter:  # type: ignore[override]
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class URLFilter:  # type: ignore[override]
        def __init__(self, name: str | None = None):
            self.name = name or self.__class__.__name__

    class FilterChain:  # type: ignore[override]
        def __init__(self, filters: Sequence[URLFilter] | None = None):
            self.filters = list(filters or [])


DOWNLOADABLE_MIMETYPES = {
    "application/pdf": ".pdf",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-powerpoint": ".ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/vnd.oasis.opendocument.spreadsheet": ".ods",
    "application/vnd.oasis.opendocument.presentation": ".odp",
    "text/csv": ".csv",
    "text/plain": ".txt",
    "application/rtf": ".rtf",
}
DOWNLOADABLE_EXTENSIONS = set(DOWNLOADABLE_MIMETYPES.values())
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".avif",
    ".bmp",
}
VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".ogg",
    ".mov",
    ".m4v",
    ".m3u8",
}
TEXT_TRACK_EXTENSIONS = {
    ".vtt",
    ".srt",
    ".txt",
}
EXCLUDED_EXTENSIONS = {
    ".zip",
    ".rar",
    ".7z",
    ".tar",
    ".gz",
    ".exe",
    ".dmg",
    ".iso",
    ".bin",
    ".ico",
    ".svg",
}
IMAGE_SKIP_TOKENS = {
    "icon",
    "favicon",
    "logo",
    "logos",
    "sprite",
    "spacer",
    "pixel",
    "loader",
    "loading",
    "placeholder",
    "thumb",
    "thumbnail",
    "tracking",
    "analytics",
    "beacon",
    "counter",
}
IMAGE_SKIP_HOSTS = {
    "mc.yandex.ru",
    "www.google-analytics.com",
    "ssl.google-analytics.com",
    "stats.g.doubleclick.net",
    "googleads.g.doubleclick.net",
    "www.googletagmanager.com",
    "www.googleadservices.com",
    "pixel.facebook.com",
    "analytics.twitter.com",
    "bat.bing.com",
}
IMAGE_SRC_ATTRIBUTES = (
    "data-srcset",
    "srcset",
    "data-src",
    "data-original",
    "data-lazy-src",
    "src",
)
VIDEO_EMBED_HOSTS = {
    "youtube.com": "youtube",
    "www.youtube.com": "youtube",
    "youtu.be": "youtube",
    "player.vimeo.com": "vimeo",
    "vimeo.com": "vimeo",
    "fast.wistia.net": "wistia",
    "home.wistia.com": "wistia",
    "player.youku.com": "youku",
}
RETRYABLE_STATUSES = {408, 425, 429, 500, 502, 503, 504}
CRAWL_SKIP_EXTENSIONS = DOWNLOADABLE_EXTENSIONS | EXCLUDED_EXTENSIONS | IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | TEXT_TRACK_EXTENSIONS

CRAWL_STATE_FILENAME = "crawl_state.json"
MAPPINGS_FILENAME = "mappings.json"
PAGE_IMAGES_FILENAME = "page_images.json"
PAGE_VIDEOS_FILENAME = "page_videos.json"
PAGE_MEDIA_FILENAME = "page_media.json"
URL_TO_MD_FILENAME = "url_to_md_mapping.json"
RUNTIME_STATE_FILENAME = "crawler_checkpoint.json"
SITEMAP_STATE_FILENAME = "sitemap_discovery.json"


def _normalize_http_url(url: str | None, base_url: str | None = None) -> Optional[str]:
    """Normalize a URL for consistent storage and deduplication."""
    if not url:
        return None

    candidate = str(url).strip()
    if not candidate:
        return None
    if candidate.startswith(("javascript:", "mailto:", "tel:", "#", "data:")):
        return None

    if base_url:
        candidate = urljoin(base_url, candidate)

    def _extract_nested_absolute_url(value: str) -> Optional[str]:
        matches = list(re.finditer(r"https?://", value, flags=re.IGNORECASE))
        if len(matches) < 2:
            return None

        # Prefer the first later absolute URL that parses cleanly.
        for match in matches[1:]:
            nested = value[match.start():].strip()
            try:
                nested_parsed = urlparse(nested)
                nested_port = nested_parsed.port
            except ValueError:
                continue
            except Exception:
                continue

            if (
                nested_parsed.scheme in ("http", "https")
                and nested_parsed.hostname
            ):
                _ = nested_port
                return nested
        return None

    for _ in range(2):
        try:
            parsed = urlparse(candidate)
        except Exception:
            return None

        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            repaired = _extract_nested_absolute_url(candidate)
            if repaired and repaired != candidate:
                candidate = repaired
                continue
            return None

        scheme = parsed.scheme.lower()
        hostname = parsed.hostname.lower() if parsed.hostname else ""
        if not hostname:
            repaired = _extract_nested_absolute_url(candidate)
            if repaired and repaired != candidate:
                candidate = repaired
                continue
            return None

        try:
            port = parsed.port
        except ValueError:
            repaired = _extract_nested_absolute_url(candidate)
            if repaired and repaired != candidate:
                candidate = repaired
                continue
            return None

        if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
            netloc = f"{hostname}:{port}"
        else:
            netloc = hostname

        path = parsed.path or "/"
        path = re.sub(r"/{2,}", "/", path)
        if path == "/":
            path = ""
        elif path.endswith("/"):
            path = path[:-1]

        query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
        query = urlencode(sorted(query_pairs)) if query_pairs else ""
        return urlunparse((scheme, netloc, path, "", query, ""))

    return None


def _url_extension(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def _url_digest(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    match = re.search(r"\d+", str(value))
    if not match:
        return None
    try:
        return int(match.group(0))
    except ValueError:
        return None


def _tokenize_path(url: str) -> set[str]:
    tokens = set(re.split(r"[^a-z0-9]+", urlparse(url).path.lower()))
    tokens.discard("")
    return tokens


def _decode_sitemap_payload(payload: bytes, source_url: str = "", content_type: str = "") -> bytes:
    if not payload:
        return b""

    looks_gzipped = payload[:2] == b"\x1f\x8b"
    if source_url.endswith(".gz") or "gzip" in content_type.lower() or looks_gzipped:
        try:
            return gzip.decompress(payload)
        except OSError:
            return payload
    return payload


def _parse_sitemap_xml(payload: bytes, source_url: str = "", content_type: str = "") -> tuple[list[str], list[str]]:
    """Parse a sitemap payload and return child sitemap URLs and page URLs."""
    decoded = _decode_sitemap_payload(payload, source_url=source_url, content_type=content_type)
    if not decoded:
        return ([], [])

    root = ET.fromstring(decoded)
    root_name = _strip_namespace(root.tag)
    sitemap_urls: list[str] = []
    page_urls: list[str] = []

    if root_name == "sitemapindex":
        for sitemap_node in root:
            if _strip_namespace(sitemap_node.tag) != "sitemap":
                continue
            for child in sitemap_node:
                if _strip_namespace(child.tag) == "loc" and child.text:
                    sitemap_urls.append(child.text.strip())
    elif root_name == "urlset":
        for url_node in root:
            if _strip_namespace(url_node.tag) != "url":
                continue
            for child in url_node:
                if _strip_namespace(child.tag) == "loc" and child.text:
                    page_urls.append(child.text.strip())

    return (sitemap_urls, page_urls)


def _build_initial_crawl_state(
    start_url: str,
    discovered_urls: Sequence[str],
    max_pages: int,
) -> dict[str, Any]:
    """Seed the crawl frontier from the start URL and sitemap-discovered URLs."""
    normalized_start = _normalize_http_url(start_url)
    if not normalized_start:
        return {"visited": [], "pending": [], "depths": {}, "pages_crawled": 0}

    pending = [{"url": normalized_start, "parent_url": None}]
    depths = {normalized_start: 0}
    seen = {normalized_start}
    remaining = max(0, int(max_pages) - 1)

    for candidate in discovered_urls:
        normalized = _normalize_http_url(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        pending.append({"url": normalized, "parent_url": normalized_start})
        depths[normalized] = 1
        remaining -= 1
        if remaining <= 0:
            break

    return {
        "visited": [],
        "pending": pending,
        "depths": depths,
        "pages_crawled": 0,
    }


def _select_best_srcset(srcset: str) -> str:
    best_url = ""
    best_score = -1.0
    for item in srcset.split(","):
        parts = item.strip().split()
        if not parts:
            continue
        candidate = parts[0]
        descriptor = parts[1] if len(parts) > 1 else ""
        score = 1.0
        if descriptor.endswith("w"):
            try:
                score = float(descriptor[:-1])
            except ValueError:
                score = 1.0
        elif descriptor.endswith("x"):
            try:
                score = float(descriptor[:-1]) * 1000.0
            except ValueError:
                score = 1.0
        if score >= best_score:
            best_score = score
            best_url = candidate
    return best_url


def _image_src_from_tag(tag: Any) -> str:
    for attr in IMAGE_SRC_ATTRIBUTES:
        value = tag.get(attr)
        if not value:
            continue
        if "srcset" in attr:
            best = _select_best_srcset(str(value))
            if best:
                return best
        else:
            return str(value)
    return ""


def _extract_image_context(tag: Any) -> str:
    figure = tag.find_parent("figure")
    if figure:
        caption = figure.find("figcaption")
        if caption:
            text = caption.get_text(" ", strip=True)
            if text:
                return text[:240]

    parent = tag.parent
    if parent:
        text = parent.get_text(" ", strip=True)
        if text:
            return text[:240]
    return ""


def _extract_caption_text(tag: Any) -> str:
    figure = tag.find_parent("figure")
    if figure:
        caption = figure.find("figcaption")
        if caption:
            text = caption.get_text(" ", strip=True)
            if text:
                return text[:240]

    for attr in ("data-caption", "caption", "aria-label", "title"):
        value = tag.get(attr)
        if value:
            return str(value).strip()[:240]
    return ""


def _extract_media_context(tag: Any) -> str:
    context = _extract_caption_text(tag)
    if context:
        return context
    return _extract_image_context(tag)


def _extract_background_image_url(style: str, base_url: str) -> str:
    if not style:
        return ""
    match = re.search(r"background-image\s*:\s*url\((['\"]?)(.+?)\1\)", style, re.I)
    if not match:
        return ""
    return _normalize_http_url(match.group(2), base_url=base_url) or ""


def _video_provider(url: str) -> str:
    host = (urlparse(url).hostname or "").lower()
    if host in VIDEO_EMBED_HOSTS:
        return VIDEO_EMBED_HOSTS[host]
    for known_host, provider in VIDEO_EMBED_HOSTS.items():
        if host.endswith(f".{known_host}") or host == known_host:
            return provider
    return ""


def _looks_like_video_embed(url: str) -> bool:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if _url_extension(url) in VIDEO_EXTENSIONS:
        return True
    if _video_provider(url):
        return True
    return any(token in path for token in ("/video", "/videos", "/embed", "/watch"))


def _extract_video_src(tag: Any) -> str:
    src = tag.get("src")
    if src:
        return str(src)
    for source in tag.find_all("source"):
        source_src = source.get("src")
        if source_src:
            return str(source_src)
    return ""


def _extract_video_track_urls(tag: Any, base_url: str) -> List[str]:
    track_urls: List[str] = []
    for track in tag.find_all("track"):
        src = track.get("src")
        if not src:
            continue
        normalized = _normalize_http_url(src, base_url=base_url)
        if not normalized:
            continue
        ext = _url_extension(normalized)
        if ext in TEXT_TRACK_EXTENSIONS or ext == "":
            track_urls.append(normalized)
    return list(dict.fromkeys(track_urls))


def _extract_data_video_urls(tag: Any, base_url: str) -> List[str]:
    raw = tag.get("data-video-urls") or ""
    candidates = []
    for value in str(raw).split(","):
        normalized = _normalize_http_url(value.strip(), base_url=base_url)
        if normalized:
            candidates.append(normalized)
    return list(dict.fromkeys(candidates))


def _select_preferred_video_url(urls: List[str]) -> str:
    if not urls:
        return ""
    for preferred_ext in (".mp4", ".webm", ".m3u8"):
        for url in urls:
            if _url_extension(url) == preferred_ext:
                return url
    return urls[0]


def _strip_webvtt(text: str) -> str:
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if (
            not line
            or line == "WEBVTT"
            or "-->" in line
            or line.isdigit()
            or line.startswith(("NOTE", "STYLE", "REGION"))
        ):
            continue
        cleaned_lines.append(line)
    return re.sub(r"\s+", " ", " ".join(cleaned_lines)).strip()


def _json_ld_items(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        queue = parsed if isinstance(parsed, list) else [parsed]
        while queue:
            item = queue.pop(0)
            if isinstance(item, list):
                queue.extend(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("@type")
            if isinstance(item_type, list):
                if any(str(v).lower() == "videoobject" for v in item_type):
                    items.append(item)
            elif str(item_type).lower() == "videoobject":
                items.append(item)
            graph = item.get("@graph")
            if isinstance(graph, list):
                queue.extend(graph)
    return items


def _merge_media_item(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            if key == "track_urls":
                merged[key] = list(dict.fromkeys((merged.get(key) or []) + list(value)))
            elif key == "title" and merged.get(key) in {"", "Video", "Embedded video"}:
                merged[key] = value
            elif not merged.get(key):
                merged[key] = value
    return merged


def _extract_page_media(
    html: str,
    base_url: str,
    *,
    max_images: Optional[int] = None,
    max_videos: Optional[int] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Extract meaningful page images and videos in DOM order."""
    if not html or not html.strip():
        return {"images": [], "videos": [], "media": []}

    soup = BeautifulSoup(html, "html.parser")
    images: List[Dict[str, Any]] = []
    videos: List[Dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_videos: set[str] = set()
    image_count = 0
    video_count = 0

    for position, tag in enumerate(soup.find_all(["img", "video", "iframe"])):
        if tag.name == "img":
            if max_images and image_count >= max_images:
                continue
            role = (tag.get("role") or "").lower()
            if role in {"presentation", "none"}:
                continue
            if str(tag.get("aria-hidden") or "").lower() == "true":
                continue

            src = _image_src_from_tag(tag)
            image_url = _normalize_http_url(src, base_url=base_url)
            if not image_url or image_url in seen_images:
                continue

            alt = (tag.get("alt") or "").strip()
            width = tag.get("width") or tag.get("data-width")
            height = tag.get("height") or tag.get("data-height")
            caption = _extract_caption_text(tag)
            context = _extract_media_context(tag)
            if not _is_content_image(image_url, alt, width, height, caption=caption, context=context):
                continue

            images.append(
                {
                    "type": "image",
                    "url": image_url,
                    "alt": alt,
                    "title": alt,
                    "caption": caption,
                    "context": context,
                    "source_type": "html",
                    "embed_type": "img",
                    "local_path": "",
                    "position": position,
                }
            )
            seen_images.add(image_url)
            image_count += 1

        elif tag.name == "video":
            video_url = _normalize_http_url(_extract_video_src(tag), base_url=base_url)
            if not video_url or video_url in seen_videos:
                continue
            if max_videos and video_count >= max_videos:
                continue

            poster_url = (
                _normalize_http_url(tag.get("poster"), base_url=base_url)
                or _extract_background_image_url(str(tag.get("style") or ""), base_url)
            )
            title = (
                str(tag.get("title") or "").strip()
                or str(tag.get("aria-label") or "").strip()
                or _extract_caption_text(tag)
            )
            videos.append(
                {
                    "type": "video",
                    "url": video_url,
                    "title": title or "Video",
                    "caption": _extract_caption_text(tag),
                    "context": _extract_media_context(tag),
                    "poster_url": poster_url or "",
                    "provider": _video_provider(video_url),
                    "source_type": "html",
                    "embed_type": "video",
                    "track_urls": _extract_video_track_urls(tag, base_url),
                    "position": position,
                }
            )
            seen_videos.add(video_url)
            video_count += 1

        elif tag.name == "iframe":
            iframe_url = _normalize_http_url(tag.get("src"), base_url=base_url)
            if not iframe_url or iframe_url in seen_videos or not _looks_like_video_embed(iframe_url):
                continue
            if max_videos and video_count >= max_videos:
                continue

            title = (
                str(tag.get("title") or "").strip()
                or str(tag.get("aria-label") or "").strip()
                or _extract_caption_text(tag)
            )
            videos.append(
                {
                    "type": "video",
                    "url": iframe_url,
                    "title": title or "Embedded video",
                    "caption": _extract_caption_text(tag),
                    "context": _extract_media_context(tag),
                    "provider": _video_provider(iframe_url),
                    "source_type": "html",
                    "embed_type": "iframe",
                    "position": position,
                }
            )
            seen_videos.add(iframe_url)
            video_count += 1

    if not max_videos or video_count < max_videos:
        for position, tag in enumerate(soup.find_all(attrs={"data-video-urls": True}), start=len(images) + len(videos)):
            video_urls = _extract_data_video_urls(tag, base_url)
            video_url = _select_preferred_video_url(video_urls)
            if not video_url or video_url in seen_videos:
                continue
            if max_videos and video_count >= max_videos:
                break

            poster_url = (
                _normalize_http_url(tag.get("data-poster-url"), base_url=base_url)
                or _extract_background_image_url(str(tag.get("style") or ""), base_url)
            )
            title = (
                str(tag.get("title") or "").strip()
                or str(tag.get("aria-label") or "").strip()
                or _extract_caption_text(tag)
                or "Background video"
            )
            videos.append(
                {
                    "type": "video",
                    "url": video_url,
                    "title": title,
                    "caption": _extract_caption_text(tag),
                    "context": _extract_media_context(tag),
                    "poster_url": poster_url or "",
                    "provider": _video_provider(video_url),
                    "source_type": "html",
                    "embed_type": "background-video",
                    "position": position,
                }
            )
            seen_videos.add(video_url)
            video_count += 1

    for item in _json_ld_items(soup):
        url = (
            _normalize_http_url(item.get("contentUrl"), base_url=base_url)
            or _normalize_http_url(item.get("embedUrl"), base_url=base_url)
            or _normalize_http_url(item.get("url"), base_url=base_url)
        )
        if not url:
            continue
        extra = {
            "type": "video",
            "url": url,
            "title": str(item.get("name") or "").strip() or "Video",
            "caption": str(item.get("description") or "").strip(),
            "context": str(item.get("description") or "").strip(),
            "poster_url": _normalize_http_url(item.get("thumbnailUrl"), base_url=base_url) or "",
            "provider": _video_provider(url),
            "source_type": "html",
            "embed_type": "jsonld",
            "transcript": str(item.get("transcript") or "").strip(),
            "position": None,
        }
        merged = False
        for idx, existing in enumerate(videos):
            if existing.get("url") == url:
                videos[idx] = _merge_media_item(existing, extra)
                merged = True
                break
        if not merged and (not max_videos or len(videos) < max_videos):
            videos.append(extra)

    media = dedupe_media_items(images + videos)
    return {
        "images": [item for item in media if item.get("type") == "image"],
        "videos": [item for item in media if item.get("type") == "video"],
        "media": media,
    }


def _guess_extension(url: str, content_type: str = "") -> str:
    ext = _url_extension(url)
    if ext:
        return ext

    mime_type = content_type.split(";", 1)[0].strip().lower()
    if mime_type in DOWNLOADABLE_MIMETYPES:
        return DOWNLOADABLE_MIMETYPES[mime_type]

    guessed = mimetypes.guess_extension(mime_type) if mime_type else None
    return guessed or ".bin"


def _looks_like_markup_payload(payload_head: bytes) -> bool:
    if not payload_head:
        return False
    head = payload_head.lstrip()[:256].lower()
    return (
        head.startswith(b"<!doctype html")
        or head.startswith(b"<html")
        or head.startswith(b"<?xml")
        or b"<html" in head[:128]
    )


def _is_valid_downloaded_document_payload(payload_head: bytes, *, extension: str, content_type: str) -> bool:
    ext = (extension or "").lower()
    mime = (content_type or "").split(";", 1)[0].strip().lower()

    if mime in {"text/html", "application/xhtml+xml", "application/xml", "text/xml"}:
        return False
    if _looks_like_markup_payload(payload_head):
        return False

    if ext == ".pdf":
        return payload_head.lstrip().startswith(b"%PDF-")
    if ext in {".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp"}:
        return payload_head.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))
    return True


def _build_stable_output_path(destination_dir: Path, url: str, content_type: str = "", default_prefix: str = "asset") -> Path:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or parsed.hostname or default_prefix
    stem = safe_filename(stem) or default_prefix
    digest = _url_digest(url)[:12]
    ext = _guess_extension(url, content_type=content_type)
    return destination_dir / f"{stem[:80]}_{digest}{ext.lower()}"


def _is_content_image(url: str, alt: str, width: Any, height: Any, caption: str = "", context: str = "") -> bool:
    """Heuristic filter for meaningful content images."""
    if not url or str(url).startswith("data:"):
        return False

    host = (urlparse(url).hostname or "").lower()
    if host in IMAGE_SKIP_HOSTS:
        return False

    ext = _url_extension(url)
    if ext in {".ico", ".svg"}:
        return False
    if ext and ext not in IMAGE_EXTENSIONS:
        return False

    tokens = _tokenize_path(url)
    if tokens & IMAGE_SKIP_TOKENS:
        return False

    alt_lower = (alt or "").strip().lower()
    if alt_lower in IMAGE_SKIP_TOKENS:
        return False
    semantic_text = " ".join(part.strip().lower() for part in (alt, caption, context) if part).strip()
    if semantic_text in IMAGE_SKIP_TOKENS:
        return False

    w = _coerce_int(width)
    h = _coerce_int(height)
    if w is not None and h is not None:
        if w <= 2 or h <= 2:
            return False
        if w < 50 and h < 50:
            return False
    elif not ext and not semantic_text:
        return False

    return True


def _extract_page_images(html: str, base_url: str, max_images: Optional[int] = None) -> List[Dict[str, str]]:
    """Extract meaningful content images from page HTML."""
    extracted = _extract_page_media(html, base_url, max_images=max_images)["images"]
    return [
        {
            "url": item.get("url", ""),
            "alt": item.get("alt", ""),
            "context": item.get("context", ""),
            "source_type": item.get("source_type", "html"),
            "local_path": item.get("local_path", ""),
            "position": item.get("position"),
            "caption": item.get("caption", ""),
        }
        for item in extracted
    ]


def _extract_page_videos(html: str, base_url: str, max_videos: Optional[int] = None) -> List[Dict[str, Any]]:
    return _extract_page_media(html, base_url, max_videos=max_videos)["videos"]


def _host_matches_domain(host: str, domain: str) -> bool:
    host = host.lower().strip(".")
    domain = domain.lower().strip(".")
    return host == domain or host.endswith(f".{domain}")


def _merge_counter_dict(base: Dict[str, int], loaded: Dict[str, Any]) -> Dict[str, int]:
    merged = dict(base)
    for key, value in loaded.items():
        try:
            merged[key] = int(value)
        except (TypeError, ValueError):
            continue
    return merged


class AllowedDomainFilter(URLFilter):
    """Allow only configured domains while excluding known subdomains."""

    def __init__(self, allowed_domains: Iterable[str], excluded_subdomains: Iterable[str]):
        super().__init__(name="AllowedDomainFilter")
        self.allowed_domains = {domain.lower() for domain in allowed_domains if domain}
        self.excluded_subdomains = {domain.lower() for domain in excluded_subdomains if domain}

    def apply(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        if any(_host_matches_domain(host, excluded) for excluded in self.excluded_subdomains):
            return False
        return any(_host_matches_domain(host, allowed) for allowed in self.allowed_domains)


class SkipExtensionFilter(URLFilter):
    """Filter out direct file URLs from the page crawl frontier."""

    def __init__(self):
        super().__init__(name="SkipExtensionFilter")

    def apply(self, url: str) -> bool:
        ext = _url_extension(url)
        if not ext:
            return True
        return ext not in CRAWL_SKIP_EXTENSIONS


@register_stage
class Crawl4AICrawler(CrawlerStage):
    name = "crawl4ai"
    description = "Resumable Crawl4AI deep crawler with sitemap seeding and controlled downloads."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: list[str] = []
        crawler = config.get("crawler", {})

        start_url = crawler.get("start_url")
        if not start_url:
            errors.append("crawler.start_url is required")
        elif not _normalize_http_url(start_url):
            errors.append("crawler.start_url must be a valid http(s) URL")

        for key in (
            "max_pages",
            "fetch_concurrency",
            "download_concurrency",
            "timeout",
            "max_images_per_page",
            "max_videos_per_page",
            "max_video_transcript_kb",
        ):
            value = crawler.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
                if key in {"max_images_per_page", "max_videos_per_page"}:
                    if numeric < 0:
                        errors.append(f"crawler.{key} must be >= 0")
                elif numeric <= 0:
                    errors.append(f"crawler.{key} must be > 0")
            except (TypeError, ValueError):
                errors.append(f"crawler.{key} must be numeric")

        max_depth = crawler.get("max_depth")
        if max_depth is not None:
            try:
                if int(max_depth) < 0:
                    errors.append("crawler.max_depth must be >= 0")
            except (TypeError, ValueError):
                errors.append("crawler.max_depth must be an integer")

        if AsyncWebCrawler is None or BrowserConfig is None or CrawlerRunConfig is None:
            errors.append("crawl4ai is not installed. Run: pip install crawl4ai")

        try:
            import bs4  # noqa: F401
        except ImportError:
            errors.append("beautifulsoup4 is not installed. Run: pip install beautifulsoup4")

        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        self.ctx = ctx
        self.config = dict(ctx.crawler_config)
        self.start_url = _normalize_http_url(self.config.get("start_url"))
        if not self.start_url:
            return StageResult.failure("crawler.start_url is missing or invalid")

        if AsyncWebCrawler is None or BrowserConfig is None or CrawlerRunConfig is None:
            return StageResult.failure("crawl4ai is not installed")

        self.timeout = float(self.config.get("timeout", 30))
        self.fetch_concurrency = max(1, int(self.config.get("fetch_concurrency", 5)))
        self.download_concurrency = max(1, int(self.config.get("download_concurrency", 10)))
        self.max_pages = max(1, int(self.config.get("max_pages", 1000)))
        self.max_depth = max(0, int(self.config.get("max_depth", 10)))
        self.max_file_size_bytes = int(float(self.config.get("max_file_size_mb", 100)) * 1024 * 1024)
        self.max_image_size_bytes = int(float(self.config.get("max_image_size_mb", 10)) * 1024 * 1024)
        self.max_video_transcript_bytes = int(
            float(self.config.get("max_video_transcript_kb", 128)) * 1024
        )
        self.max_images_per_page = max(0, int(self.config.get("max_images_per_page", 20)))
        self.max_videos_per_page = max(0, int(self.config.get("max_videos_per_page", 10)))
        self.retry_attempts = max(1, int(self.config.get("download_retry_attempts", 3)))
        self.retry_backoff = max(0.1, float(self.config.get("download_retry_backoff_sec", 2.0)))
        self.checkpoint_flush_interval = max(
            0.0, float(self.config.get("checkpoint_flush_interval_sec", 5.0))
        )
        self.extract_images = bool(self.config.get("extract_images", True))
        self.extract_videos = bool(self.config.get("extract_videos", True))
        self.download_page_images = bool(self.config.get("download_page_images", True))
        self.fetch_video_transcripts = bool(self.config.get("fetch_video_transcripts", True))
        self.parse_source_html_for_media = bool(self.config.get("parse_source_html_for_media", True))
        self.headless = bool(self.config.get("headless", True))
        self.ignore_https_errors = bool(self.config.get("ignore_https_errors", True))
        self.enable_stealth = bool(self.config.get("enable_stealth", False))
        self.allowed_domains = self._resolve_allowed_domains()
        self.excluded_subdomains = {
            value.lower()
            for value in (self.config.get("excluded_subdomains") or [])
            if value
        }
        self.include_external = self._should_include_external_links()
        self.proxy = self.config.get("proxy")

        self.html_dir = ensure_dir(ctx.work_dir / "html")
        self.md_dir = ensure_dir(ctx.work_dir / "markdown")
        self.download_dir = ensure_dir(ctx.work_dir / "downloads")
        self.images_dir = ensure_dir(ctx.work_dir / "downloaded_page_images")

        self.mapping_file = ctx.work_dir / MAPPINGS_FILENAME
        self.page_images_file = ctx.work_dir / PAGE_IMAGES_FILENAME
        self.page_videos_file = ctx.work_dir / PAGE_VIDEOS_FILENAME
        self.page_media_file = ctx.work_dir / PAGE_MEDIA_FILENAME
        self.url_to_md_mapping_file = ctx.work_dir / URL_TO_MD_FILENAME
        self.crawl_state_file = ctx.work_dir / CRAWL_STATE_FILENAME
        self.runtime_state_file = ctx.work_dir / RUNTIME_STATE_FILENAME
        self.sitemap_state_file = ctx.work_dir / SITEMAP_STATE_FILENAME

        self.stats = {
            "pages_scraped": 0,
            "pages_failed": 0,
            "markdown_written": 0,
            "documents_downloaded": 0,
            "images_extracted": 0,
            "images_downloaded": 0,
            "videos_extracted": 0,
            "video_transcripts_fetched": 0,
            "bytes_downloaded": 0,
            "sitemap_urls_seeded": 0,
            "skipped_urls": 0,
        }
        self.url_mapping: Dict[str, str] = {}
        self.url_to_md_mapping: Dict[str, str] = {}
        self.page_images: Dict[str, List[Dict[str, str]]] = {}
        self.page_videos: Dict[str, List[Dict[str, Any]]] = {}
        self.page_media: Dict[str, List[Dict[str, Any]]] = {}
        self.downloaded_images: Dict[str, str] = {}
        self.crawl_state: Dict[str, Any] = {}
        self.discovered_sitemaps: Dict[str, Any] = {"sources": [], "urls": []}
        self._last_flush_at = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._download_semaphore = asyncio.Semaphore(self.download_concurrency)

        self._load_runtime_state(ctx.checkpoint)

        try:
            await self._open_http_session()

            if not self.crawl_state:
                sitemap_urls = []
                if self.config.get("sitemap_enabled", True):
                    sitemap_urls = await self._discover_sitemap_urls()
                    self.stats["sitemap_urls_seeded"] = len(sitemap_urls)
                self.crawl_state = _build_initial_crawl_state(self.start_url, sitemap_urls, self.max_pages)
                self._write_crawl_state_file()

            browser_config = self._build_browser_config()
            run_config = self._build_run_config()

            async with AsyncWebCrawler(config=browser_config) as crawler:
                results = crawler.arun(url=self.start_url, config=run_config)
                if asyncio.iscoroutine(results):
                    results = await results
                await self._consume_crawl_results(results)

            self._flush_runtime_state(force=True)

            return StageResult.success(
                outputs={
                    "html_dir": str(self.html_dir),
                    "md_dir": str(self.md_dir),
                    "download_dir": str(self.download_dir),
                    "mapping_file": str(self.mapping_file),
                    "page_images_file": str(self.page_images_file),
                    "page_videos_file": str(self.page_videos_file),
                    "page_media_file": str(self.page_media_file),
                    "images_dir": str(self.images_dir),
                    "md_mapping_file": str(self.url_to_md_mapping_file),
                },
                metrics=self._build_metrics(),
                checkpoint={"runtime_state_file": str(self.runtime_state_file)},
            )
        except Exception as exc:
            logger.exception("Crawler stage failed: %s", exc)
            self._flush_runtime_state(force=True)
            return StageResult.failure(
                str(exc),
                checkpoint={"runtime_state_file": str(self.runtime_state_file)},
            )
        finally:
            if self._session and not self._session.closed:
                await self._session.close()

    def _resolve_allowed_domains(self) -> set[str]:
        configured = {
            value.lower()
            for value in (self.config.get("allowed_domains") or [])
            if value
        }
        start_domain = (urlparse(self.start_url).hostname or "").lower()
        if start_domain:
            configured.add(start_domain)
        return configured

    def _should_include_external_links(self) -> bool:
        configured = bool(self.config.get("include_external", False))
        start_domain = (urlparse(self.start_url).hostname or "").lower()
        if not start_domain:
            return configured
        external_domains = {
            domain for domain in self.allowed_domains if not _host_matches_domain(domain, start_domain)
        }
        return configured or bool(external_domains)

    def _load_runtime_state(self, checkpoint: Optional[Dict[str, Any]]) -> None:
        runtime_path = None
        if isinstance(checkpoint, dict):
            runtime_path = checkpoint.get("runtime_state_file")

        state = None
        if runtime_path:
            state = load_json_safe(runtime_path)
        if not state:
            state = load_json_safe(self.runtime_state_file, {})

        state = state or {}
        self.url_mapping = load_json_safe(self.mapping_file, {}) or {}
        self.url_mapping.update(state.get("url_mapping") or {})

        self.url_to_md_mapping = load_json_safe(self.url_to_md_mapping_file, {}) or {}
        self.url_to_md_mapping.update(state.get("url_to_md_mapping") or {})

        self.page_images = load_json_safe(self.page_images_file, {}) or {}
        self.page_images.update(state.get("page_images") or {})

        self.page_videos = load_json_safe(self.page_videos_file, {}) or {}
        self.page_videos.update(state.get("page_videos") or {})

        self.page_media = load_json_safe(self.page_media_file, {}) or {}
        self.page_media.update(state.get("page_media") or {})
        if not self.page_media:
            for page_url in set(self.page_images) | set(self.page_videos):
                combined = []
                combined.extend(self.page_images.get(page_url, []))
                combined.extend(self.page_videos.get(page_url, []))
                if combined:
                    self.page_media[page_url] = dedupe_media_items(combined)

        self.downloaded_images = dict(state.get("downloaded_images") or {})
        loaded_stats = state.get("stats") or {}
        self.stats = _merge_counter_dict(self.stats, loaded_stats)

        crawl_state = state.get("crawl_state") or load_json_safe(self.crawl_state_file, {}) or {}
        pending = crawl_state.get("pending")
        if not pending and crawl_state.get("to_visit"):
            crawl_state["pending"] = [
                {"url": url, "parent_url": None}
                for url in crawl_state.get("to_visit", [])
            ]
        self.crawl_state = crawl_state

        sitemap_state = load_json_safe(self.sitemap_state_file, {}) or {}
        self.discovered_sitemaps = {
            "sources": sitemap_state.get("sources", []),
            "urls": sitemap_state.get("urls", []),
        }

    async def _open_http_session(self) -> None:
        headers = {}
        config_headers = self.config.get("headers")
        if isinstance(config_headers, dict):
            headers.update({str(k): str(v) for k, v in config_headers.items()})

        user_agent = self.config.get("user_agent")
        if user_agent and "User-Agent" not in headers:
            headers["User-Agent"] = str(user_agent)

        cookie_jar = aiohttp.CookieJar(unsafe=True)
        for cookie in self.config.get("cookies") or []:
            if not isinstance(cookie, dict):
                continue
            name = cookie.get("name")
            value = cookie.get("value")
            if name is not None and value is not None:
                cookie_jar.update_cookies({str(name): str(value)})

        connector = aiohttp.TCPConnector(
            limit=max(self.fetch_concurrency, self.download_concurrency),
            ssl=False if self.ignore_https_errors else True,
        )
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        self._session = aiohttp.ClientSession(
            headers=headers or None,
            timeout=timeout,
            connector=connector,
            cookie_jar=cookie_jar,
        )

    async def _discover_sitemap_urls(self) -> List[str]:
        if not self._session:
            return []

        start_parsed = urlparse(self.start_url)
        candidates = []
        robots_url = urljoin(self.start_url, "/robots.txt")

        try:
            async with self._session.get(robots_url, proxy=self.proxy) as response:
                if response.status == 200:
                    robots_text = await response.text()
                    for line in robots_text.splitlines():
                        if line.lower().startswith("sitemap:"):
                            sitemap_url = line.split(":", 1)[1].strip()
                            if sitemap_url:
                                candidates.append(sitemap_url)
        except Exception as exc:
            logger.debug("Could not read robots.txt for sitemap discovery: %s", exc)

        root_url = f"{start_parsed.scheme}://{start_parsed.netloc}"
        candidates.extend(
            [
                urljoin(root_url, "/sitemap.xml"),
                urljoin(root_url, "/sitemap_index.xml"),
            ]
        )

        seen_sitemaps = set()
        collected_urls: list[str] = []
        sitemap_limit = max(0, int(self.config.get("sitemap_seed_limit", 500)))

        async def _walk(sitemap_url: str) -> None:
            normalized = _normalize_http_url(sitemap_url)
            if not normalized or normalized in seen_sitemaps:
                return
            seen_sitemaps.add(normalized)

            try:
                async with self._session.get(normalized, proxy=self.proxy) as response:
                    if response.status != 200:
                        return
                    payload = await response.read()
                    content_type = response.headers.get("Content-Type", "")
            except Exception as exc:
                logger.debug("Failed to fetch sitemap %s: %s", normalized, exc)
                return

            child_sitemaps, page_urls = _parse_sitemap_xml(
                payload,
                source_url=normalized,
                content_type=content_type,
            )
            for page_url in page_urls:
                normalized_page = _normalize_http_url(page_url)
                if not normalized_page:
                    continue
                host = (urlparse(normalized_page).hostname or "").lower()
                if not any(_host_matches_domain(host, allowed) for allowed in self.allowed_domains):
                    continue
                collected_urls.append(normalized_page)
                if sitemap_limit and len(collected_urls) >= sitemap_limit:
                    break

            if sitemap_limit and len(collected_urls) >= sitemap_limit:
                return

            for child in child_sitemaps:
                await _walk(child)
                if sitemap_limit and len(collected_urls) >= sitemap_limit:
                    return

        for candidate in candidates:
            await _walk(candidate)
            if sitemap_limit and len(collected_urls) >= sitemap_limit:
                break

        deduped = list(dict.fromkeys(collected_urls))
        self.discovered_sitemaps = {
            "sources": sorted(seen_sitemaps),
            "urls": deduped,
        }
        atomic_write_json(self.sitemap_state_file, self.discovered_sitemaps)
        return deduped

    def _build_browser_config(self) -> BrowserConfig:
        browser_kwargs = {
            "headless": self.headless,
            "ignore_https_errors": self.ignore_https_errors,
            "user_agent": self.config.get("user_agent"),
            "headers": self.config.get("headers"),
            "cookies": self.config.get("cookies") or None,
            "proxy": self.proxy,
            "proxy_config": self.config.get("proxy_config"),
            "enable_stealth": self.enable_stealth,
            "java_script_enabled": bool(self.config.get("java_script_enabled", True)),
            "init_scripts": self.config.get("init_scripts") or None,
            "storage_state": self.config.get("storage_state"),
            "viewport": self.config.get("viewport"),
            "verbose": False,
        }
        browser_kwargs = {k: v for k, v in browser_kwargs.items() if v is not None}
        return BrowserConfig(**browser_kwargs)

    def _build_run_config(self) -> CrawlerRunConfig:
        content_filter_threshold = (
            self.ctx.config.get("converter", {}).get("content_filter_threshold", 0.48)
        )
        markdown_generator = None
        if DefaultMarkdownGenerator is not None and PruningContentFilter is not None:
            markdown_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=content_filter_threshold)
            )

        filter_chain = FilterChain(
            filters=[
                AllowedDomainFilter(self.allowed_domains, self.excluded_subdomains),
                SkipExtensionFilter(),
            ]
        )

        resume_state = self.crawl_state or None
        deep_crawl = BFSDeepCrawlStrategy(
            max_depth=self.max_depth,
            filter_chain=filter_chain,
            include_external=self.include_external,
            max_pages=self.max_pages,
            logger=logger,
            resume_state=resume_state,
            on_state_change=self._on_crawl_state_change,
        )

        run_kwargs = {
            "deep_crawl_strategy": deep_crawl,
            # Stream results as they are discovered so long crawls produce
            # incremental artifacts/checkpoints instead of buffering until the
            # entire traversal completes.
            "stream": True,
            "page_timeout": int(self.timeout * 1000),
            "wait_until": self.config.get("wait_until", "domcontentloaded"),
            "wait_for": self.config.get("wait_for"),
            "wait_for_timeout": self.config.get("wait_for_timeout_ms"),
            "delay_before_return_html": float(self.config.get("delay_before_return_html", 0.1)),
            "mean_delay": float(self.config.get("mean_delay", 0.1)),
            "max_range": float(self.config.get("max_range", 0.3)),
            "semaphore_count": self.fetch_concurrency,
            "process_iframes": bool(self.config.get("process_iframes", False)),
            "remove_overlay_elements": bool(self.config.get("remove_overlay_elements", True)),
            "simulate_user": bool(self.config.get("simulate_user", False)),
            "override_navigator": bool(self.config.get("override_navigator", False)),
            "magic": bool(self.config.get("magic", False)),
            "check_robots_txt": bool(self.config.get("respect_robots_txt", True)),
            "markdown_generator": markdown_generator,
            "verbose": False,
        }
        run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}
        return CrawlerRunConfig(**run_kwargs)

    async def _consume_crawl_results(self, results: Any) -> None:
        if inspect.isasyncgen(results):
            async for result in results:
                await self._process_result(result)
        elif isinstance(results, list):
            for result in results:
                await self._process_result(result)
        elif results is not None:
            await self._process_result(results)

    async def _process_result(self, result: Any) -> None:
        page_url = _normalize_http_url(getattr(result, "url", None))
        if not page_url:
            return

        status_code = getattr(result, "status_code", None)
        html = getattr(result, "html", None) or ""

        if not getattr(result, "success", False) or not html:
            self.stats["pages_failed"] += 1
            self.stats["skipped_urls"] += 1
            reason = "SKIPPED_ERROR"
            if status_code:
                reason = f"SKIPPED_HTTP_{status_code}"
            self.url_mapping[page_url] = reason
            self._flush_runtime_state()
            return

        html_path = self.html_dir / f"{_url_digest(page_url)}.html"
        html_path.write_text(html, encoding="utf-8")
        self.url_mapping[page_url] = str(html_path)
        self.stats["pages_scraped"] += 1
        self.stats["bytes_downloaded"] += len(html.encode("utf-8"))
        pages_scraped = self.stats["pages_scraped"]
        if pages_scraped <= 3 or pages_scraped % 25 == 0:
            logger.info(
                "Crawler progress: pages_scraped=%d documents_downloaded=%d images_extracted=%d videos_extracted=%d url=%s",
                pages_scraped,
                self.stats["documents_downloaded"],
                self.stats["images_extracted"],
                self.stats["videos_extracted"],
                page_url,
            )

        md_text = self._extract_markdown(result)
        if md_text:
            md_path = self.md_dir / f"{html_path.stem}.md"
            md_path.write_text(md_text, encoding="utf-8")
            self.url_to_md_mapping[page_url] = str(md_path)
            self.stats["markdown_written"] += 1

        if self.extract_images or self.extract_videos:
            try:
                extracted_media = _extract_page_media(
                    html,
                    page_url,
                    max_images=self.max_images_per_page,
                    max_videos=self.max_videos_per_page,
                )
                if self.parse_source_html_for_media and (
                    (self.extract_videos and not extracted_media["videos"])
                    or (self.extract_images and not extracted_media["images"])
                ):
                    raw_html = await self._fetch_raw_source_html(page_url)
                    if raw_html and raw_html != html:
                        raw_media = _extract_page_media(
                            raw_html,
                            page_url,
                            max_images=self.max_images_per_page,
                            max_videos=self.max_videos_per_page,
                        )
                        extracted_media = {
                            "images": dedupe_media_items(
                                [*extracted_media["images"], *raw_media["images"]]
                            ),
                            "videos": dedupe_media_items(
                                [*extracted_media["videos"], *raw_media["videos"]]
                            ),
                            "media": dedupe_media_items(
                                [*extracted_media["media"], *raw_media["media"]]
                            ),
                        }
                images = extracted_media["images"] if self.extract_images else []
                videos = extracted_media["videos"] if self.extract_videos else []

                if images and self.download_page_images:
                    await self._download_page_images(images)
                if videos:
                    await self._populate_video_transcripts(videos)

                if images:
                    self.page_images[page_url] = images
                    self.stats["images_extracted"] += len(images)
                if videos:
                    self.page_videos[page_url] = videos
                    self.stats["videos_extracted"] += len(videos)

                page_media = dedupe_media_items([*images, *videos])
                if page_media:
                    self.page_media[page_url] = page_media
            except Exception as exc:
                logger.warning("Skipping media extraction for %s: %s", page_url, exc)

        downloadable_urls = self._extract_downloadable_urls(result, html=html, base_url=page_url)
        if downloadable_urls:
            await asyncio.gather(
                *(self._download_document(url) for url in downloadable_urls),
                return_exceptions=True,
            )

        self._flush_runtime_state()

    def _extract_markdown(self, result: Any) -> str:
        markdown = getattr(result, "markdown", None)
        if markdown is None:
            return ""
        fit_markdown = getattr(markdown, "fit_markdown", "") or ""
        raw_markdown = getattr(markdown, "raw_markdown", "") or ""
        return fit_markdown.strip() or raw_markdown.strip()

    def _extract_downloadable_urls(self, result: Any, html: str, base_url: str) -> List[str]:
        candidates: list[str] = []

        links = getattr(result, "links", None) or {}
        for group in ("internal", "external"):
            for item in links.get(group, []) or []:
                href = item.get("href") if isinstance(item, dict) else None
                normalized = _normalize_http_url(href, base_url=base_url)
                if normalized and _url_extension(normalized) in DOWNLOADABLE_EXTENSIONS:
                    candidates.append(normalized)

        if html:
            soup = BeautifulSoup(html, "html.parser")
            for anchor in soup.find_all("a", href=True):
                normalized = _normalize_http_url(anchor.get("href"), base_url=base_url)
                if normalized and _url_extension(normalized) in DOWNLOADABLE_EXTENSIONS:
                    candidates.append(normalized)

        return list(dict.fromkeys(candidates))

    def _track_url_allowed(self, url: str) -> bool:
        host = (urlparse(url).hostname or "").lower()
        if not host:
            return False
        return any(_host_matches_domain(host, allowed) for allowed in self.allowed_domains)

    async def _populate_video_transcripts(self, videos: List[Dict[str, Any]]) -> None:
        if not self.fetch_video_transcripts or not self._session:
            return

        tasks = []
        for video in videos:
            if video.get("transcript"):
                continue
            track_urls = [
                url
                for url in (video.get("track_urls") or [])
                if isinstance(url, str) and self._track_url_allowed(url)
            ]
            if track_urls:
                tasks.append(self._fetch_video_transcript(video, track_urls[0]))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_video_transcript(self, video: Dict[str, Any], track_url: str) -> None:
        if not self._session:
            return

        normalized = _normalize_http_url(track_url)
        if not normalized:
            return

        try:
            async with self._download_semaphore:
                async with self._session.get(normalized, proxy=self.proxy) as response:
                    if response.status >= 400:
                        return
                    content_type = (response.headers.get("Content-Type") or "").lower()
                    ext = _url_extension(normalized)
                    if ext not in TEXT_TRACK_EXTENSIONS and "text" not in content_type and "vtt" not in content_type:
                        return

                    chunks = []
                    bytes_read = 0
                    async for chunk in response.content.iter_chunked(32 * 1024):
                        if not chunk:
                            continue
                        bytes_read += len(chunk)
                        if bytes_read > self.max_video_transcript_bytes:
                            return
                        chunks.append(chunk)

                    transcript = _strip_webvtt(b"".join(chunks).decode("utf-8", errors="replace"))
                    if transcript:
                        video["transcript"] = transcript[:4000]
                        video["transcript_url"] = normalized
                        self.stats["video_transcripts_fetched"] += 1
        except Exception as exc:
            logger.debug("Failed to fetch transcript %s: %s", normalized, exc)

    async def _fetch_raw_source_html(self, page_url: str) -> str:
        if not self._session:
            return ""
        try:
            async with self._session.get(page_url, proxy=self.proxy) as response:
                if response.status >= 400:
                    return ""
                return await response.text()
        except Exception as exc:
            logger.debug("Failed to fetch raw page source for %s: %s", page_url, exc)
            return ""

    async def _download_page_images(self, images: List[Dict[str, str]]) -> None:
        await asyncio.gather(
            *(self._download_single_image(image) for image in images),
            return_exceptions=True,
        )

    async def _download_single_image(self, image: Dict[str, str]) -> None:
        url = image.get("url", "")
        if not url:
            return
        existing = self.downloaded_images.get(url)
        if existing:
            image["local_path"] = existing
            return

        path = await self._download_binary(
            url=url,
            destination_dir=self.images_dir,
            max_bytes=self.max_image_size_bytes,
            expected_prefix="image/",
            status_on_failure=None,
        )
        if path:
            path_str = str(path)
            image["local_path"] = path_str
            self.downloaded_images[url] = path_str
            self.stats["images_downloaded"] += 1

    async def _download_document(self, url: str) -> None:
        if url in self.url_mapping and not str(self.url_mapping[url]).startswith("SKIPPED_"):
            return

        ext = _url_extension(url)
        if ext in EXCLUDED_EXTENSIONS:
            self.url_mapping[url] = "SKIPPED_EXCLUDED"
            self.stats["skipped_urls"] += 1
            return

        path = await self._download_binary(
            url=url,
            destination_dir=self.download_dir,
            max_bytes=self.max_file_size_bytes,
            expected_prefix=None,
            status_on_failure="SKIPPED_DOWNLOAD_FAILED",
            validate_document=True,
        )
        if path:
            self.url_mapping[url] = str(path)
            self.stats["documents_downloaded"] += 1

    async def _download_binary(
        self,
        url: str,
        destination_dir: Path,
        max_bytes: int,
        expected_prefix: Optional[str],
        status_on_failure: Optional[str],
        validate_document: bool = False,
    ) -> Optional[Path]:
        if not self._session:
            return None

        normalized = _normalize_http_url(url)
        if not normalized:
            return None

        for attempt in range(1, self.retry_attempts + 1):
            tmp_path = None
            try:
                async with self._download_semaphore:
                    async with self._session.get(
                        normalized,
                        allow_redirects=True,
                        proxy=self.proxy,
                    ) as response:
                        status = response.status
                        if status in RETRYABLE_STATUSES and attempt < self.retry_attempts:
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=status,
                                message=f"retryable status {status}",
                            )
                        if status >= 400:
                            if status_on_failure:
                                self.url_mapping[normalized] = f"SKIPPED_HTTP_{status}"
                                self.stats["skipped_urls"] += 1
                            return None

                        content_type = response.headers.get("Content-Type", "")
                        if expected_prefix and content_type and not content_type.lower().startswith(expected_prefix):
                            return None

                        target_path = _build_stable_output_path(
                            destination_dir,
                            normalized,
                            content_type=content_type,
                        )
                        tmp_path = target_path.with_suffix(target_path.suffix + ".part")
                        bytes_written = 0
                        payload_head = bytearray()

                        with open(tmp_path, "wb") as handle:
                            async for chunk in response.content.iter_chunked(64 * 1024):
                                if not chunk:
                                    continue
                                bytes_written += len(chunk)
                                if bytes_written > max_bytes:
                                    raise ValueError("download exceeds configured size limit")
                                if len(payload_head) < 1024:
                                    payload_head.extend(chunk[: 1024 - len(payload_head)])
                                handle.write(chunk)

                        if validate_document and not _is_valid_downloaded_document_payload(
                            bytes(payload_head),
                            extension=target_path.suffix.lower(),
                            content_type=content_type,
                        ):
                            if tmp_path.exists():
                                tmp_path.unlink(missing_ok=True)
                            if status_on_failure:
                                self.url_mapping[normalized] = "SKIPPED_INVALID_DOCUMENT"
                                self.stats["skipped_urls"] += 1
                            return None

                        tmp_path.replace(target_path)
                        self.stats["bytes_downloaded"] += bytes_written
                        return target_path
            except ValueError:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                if status_on_failure:
                    self.url_mapping[normalized] = "SKIPPED_TOO_LARGE"
                    self.stats["skipped_urls"] += 1
                return None
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if tmp_path and tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                if attempt >= self.retry_attempts:
                    logger.debug("Download failed for %s: %s", normalized, exc)
                    if status_on_failure:
                        self.url_mapping[normalized] = status_on_failure
                        self.stats["skipped_urls"] += 1
                    return None
                await asyncio.sleep(self.retry_backoff * attempt)

        return None

    async def _on_crawl_state_change(self, state: Dict[str, Any]) -> None:
        self.crawl_state = state
        self._write_crawl_state_file()
        self._flush_runtime_state()

    def _write_crawl_state_file(self) -> None:
        pending = self.crawl_state.get("pending", [])
        public_state = {
            "visited": self.crawl_state.get("visited", []),
            "pending": pending,
            "to_visit": [item.get("url") for item in pending if isinstance(item, dict)],
            "depths": self.crawl_state.get("depths", {}),
            "pages_crawled": self.crawl_state.get("pages_crawled", 0),
            "updated_at": time.time(),
        }
        atomic_write_json(self.crawl_state_file, public_state)

    def _serialize_runtime_state(self) -> Dict[str, Any]:
        return {
            "start_url": self.start_url,
            "crawl_state": self.crawl_state,
            "url_mapping": self.url_mapping,
            "url_to_md_mapping": self.url_to_md_mapping,
            "page_images": self.page_images,
            "page_videos": self.page_videos,
            "page_media": self.page_media,
            "downloaded_images": self.downloaded_images,
            "stats": self.stats,
            "updated_at": time.time(),
        }

    def _flush_runtime_state(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self._last_flush_at) < self.checkpoint_flush_interval:
            return

        self._write_crawl_state_file()
        atomic_write_json(self.mapping_file, self.url_mapping)
        atomic_write_json(self.url_to_md_mapping_file, self.url_to_md_mapping)
        atomic_write_json(self.page_images_file, self.page_images)
        atomic_write_json(self.page_videos_file, self.page_videos)
        atomic_write_json(self.page_media_file, self.page_media)
        atomic_write_json(self.runtime_state_file, self._serialize_runtime_state())
        if self.discovered_sitemaps["sources"] or self.discovered_sitemaps["urls"]:
            atomic_write_json(self.sitemap_state_file, self.discovered_sitemaps)
        self._last_flush_at = now

    def _build_metrics(self) -> Dict[str, int]:
        visited_count = len(self.crawl_state.get("visited", []))
        return {
            **self.stats,
            "visited_count": visited_count,
            "mapped_urls": len(self.url_mapping),
        }
