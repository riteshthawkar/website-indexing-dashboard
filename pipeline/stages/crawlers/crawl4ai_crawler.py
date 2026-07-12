"""
Crawl4AI crawler stage.

This stage provides the active production crawler for the modular pipeline.
It persists HTML, markdown, downloads, page image metadata, and a resumable
runtime checkpoint so failed crawls can continue from the last known frontier.
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import hashlib
import inspect
import ipaddress
import json
import logging
import mimetypes
import re
import socket
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
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
GENERIC_MBZUAI_PAGE_TITLES = {
    "mbzuai mohamed bin zayed university of artificial intelligence",
    "mbzuai - mohamed bin zayed university of artificial intelligence",
    "mohamed bin zayed university of artificial intelligence",
}
GENERIC_SITE_SHELL_PHRASES = {
    "thoughtcurators for ai creators",
    "explore our ai degrees",
    "staff login student login careers contact quick links",
}
COMMON_NAVIGATION_LABELS = {
    "about",
    "study",
    "research",
    "innovate",
    "sustainability",
    "careers",
    "contact",
    "quick links",
    "faculty directory",
    "fast facts",
    "student login",
    "staff login",
    "نبذة عن الجامعة",
    "الدراسة",
    "الأبحاث",
    "البحوث",
    "الابتكار",
    "الاستدامة",
    "الموارد الطلابية",
    "الأخبار والفعاليات",
    "اتصل بنا",
}
BOILERPLATE_ATTR_TOKENS = (
    "nav",
    "navigation",
    "navbar",
    "menu",
    "mega-menu",
    "header",
    "footer",
    "breadcrumb",
    "cookie",
    "modal",
    "popup",
    "overlay",
    "skip-link",
    "quick-link",
    "quicklink",
    "social",
    "newsletter",
)
CONTENT_ERROR_PATTERNS = (
    re.compile(r"\b403\s+forbidden\b", re.I),
    re.compile(r"\b404\s+not\s+found\b", re.I),
    re.compile(r"\b500\s+internal\s+server\s+error\b", re.I),
    re.compile(r"\baccess\s+denied\b", re.I),
    re.compile(r"\bcss\s+error\b", re.I),
    re.compile(r"\bsorry\s+to\s+interrupt\b", re.I),
    re.compile(r"\benable\s+javascript\s+and\s+cookies\b", re.I),
    re.compile(r"\bjust\s+a\s+moment\b", re.I),
)
URL_TOKEN_STOPWORDS = {
    "www",
    "http",
    "https",
    "html",
    "php",
    "asp",
    "aspx",
    "index",
    "page",
    "pages",
    "about",
    "study",
    "research",
    "news",
    "events",
    "resources",
    "resource",
    "student",
    "students",
    "faculty",
    "directory",
    "mbzuai",
    "ac",
    "ae",
    "the",
    "and",
    "for",
    "with",
    "from",
    "our",
    "your",
    "of",
}

CRAWL_STATE_FILENAME = "crawl_state.json"
MAPPINGS_FILENAME = "mappings.json"
PAGE_IMAGES_FILENAME = "page_images.json"
PAGE_VIDEOS_FILENAME = "page_videos.json"
PAGE_MEDIA_FILENAME = "page_media.json"
PAGE_METADATA_FILENAME = "page_metadata.json"
PAGE_LINK_GRAPH_FILENAME = "page_link_graph.json"
URL_TO_MD_FILENAME = "url_to_md_mapping.json"
RUNTIME_STATE_FILENAME = "crawler_checkpoint.json"
SITEMAP_STATE_FILENAME = "sitemap_discovery.json"


def _compact_failure_reason(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    return text[:max_chars]


def _is_recoverable_crawl_skip_reason(value: Any) -> bool:
    text = str(value or "").lower()
    if not text.startswith("skipped"):
        return False
    recoverable_tokens = (
        "browser has been closed",
        "target page, context or browser has been closed",
        "browsercontext.add_init_script",
        "page.goto: timeout",
        "net::err_internet_disconnected",
        "net::err_network_changed",
        "net::err_connection_reset",
        "net::err_connection_aborted",
        "net::err_connection_closed",
        "net::err_connection_timed_out",
        "skipped_no_result",
        "skipped_http_403:content_quality",
        "skipped_low_quality:content_quality:blocked_or_error_page",
    )
    return any(token in text for token in recoverable_tokens)


def _is_benign_browser_close_error(exc: BaseException) -> bool:
    text = str(exc or "").lower()
    return (
        "browser.close" in text
        and (
            "connection closed" in text
            or "target page, context or browser has been closed" in text
            or "browser has been closed" in text
        )
    )


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


def _is_private_or_reserved_address(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


@lru_cache(maxsize=2048)
def _host_resolves_to_private_or_reserved(host: str) -> bool:
    if not host:
        return True
    if host.lower().strip(".") == "localhost":
        return True
    if _is_private_or_reserved_address(host):
        return True
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        # DNS failures should not turn an outbound URL into an allowed URL.
        return True
    except Exception:
        logger.debug("Failed to resolve host for egress validation: %s", host, exc_info=True)
        return True
    return any(_is_private_or_reserved_address(info[4][0]) for info in infos if info and info[4])


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


def _semantic_tokens(text: str) -> set[str]:
    tokens = set()
    for token in re.split(r"[^a-z0-9]+", str(text or "").lower()):
        if len(token) < 3 or token in URL_TOKEN_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _url_semantic_tokens(url: str) -> set[str]:
    return _semantic_tokens(urlparse(url).path)


def _token_coverage(required: set[str], observed: set[str]) -> float:
    if not required:
        return 1.0
    return len(required & observed) / len(required)


def _extract_html_title(html: str) -> str:
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title:
            return " ".join(soup.title.get_text(" ", strip=True).split())
    except Exception:
        return ""
    return ""


def _visible_text_from_html(html: str) -> str:
    if not html:
        return ""
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "template", "noscript", "svg"]):
            tag.decompose()
        return " ".join(soup.get_text(" ", strip=True).split())
    except Exception:
        return " ".join(str(html or "").split())


def _contains_error_or_block_text(text: str) -> bool:
    if not text:
        return False
    return any(pattern.search(text) for pattern in CONTENT_ERROR_PATTERNS)


def _is_generic_mbzuai_title(title: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(title or "").strip().lower())
    return normalized in GENERIC_MBZUAI_PAGE_TITLES


def _normalize_path_prefix(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "/" + text.strip("/")


def _url_matches_path_prefix(url: str, prefixes: Iterable[str]) -> bool:
    normalized = _normalize_http_url(url)
    if not normalized:
        return False
    path = "/" + urlparse(normalized).path.strip("/")
    for prefix in prefixes:
        clean_prefix = _normalize_path_prefix(prefix)
        if not clean_prefix:
            continue
        if path == clean_prefix or path.startswith(f"{clean_prefix}/"):
            return True
    return False


def _html_quality_report(html: str, page_url: str, markdown: str = "") -> Dict[str, Any]:
    title = _extract_html_title(html)
    visible_text = _visible_text_from_html(html)
    combined_text = " ".join(part for part in (title, visible_text, markdown) if part)
    path_tokens = _url_semantic_tokens(page_url)
    combined_tokens = _semantic_tokens(combined_text)
    title_tokens = _semantic_tokens(title)
    token_coverage = _token_coverage(path_tokens, combined_tokens)
    title_token_coverage = _token_coverage(path_tokens, title_tokens)
    lower_text = combined_text.lower()

    reasons: List[str] = []
    critical = False
    if not html or not str(html).strip():
        reasons.append("empty_html")
        critical = True
    if _contains_error_or_block_text(combined_text):
        reasons.append("blocked_or_error_page")
        critical = True
    if path_tokens and _is_generic_mbzuai_title(title):
        reasons.append("generic_mbzuai_title")
    if path_tokens and any(phrase in lower_text for phrase in GENERIC_SITE_SHELL_PHRASES) and token_coverage < 0.5:
        reasons.append("generic_site_shell")
    if path_tokens and token_coverage < 0.34 and title_token_coverage < 0.34:
        reasons.append("url_token_mismatch")
    if path_tokens and 0 < len(visible_text) < 120:
        reasons.append("thin_html")
    if (
        path_tokens
        and "generic_mbzuai_title" in reasons
        and "url_token_mismatch" in reasons
        and ("thin_html" in reasons or len(visible_text) < 300)
    ):
        critical = True

    score = min(len(visible_text) / 100.0, 60.0)
    score += token_coverage * 30.0
    score += title_token_coverage * 10.0
    if "generic_mbzuai_title" in reasons:
        score -= 15.0
    if "generic_site_shell" in reasons:
        score -= 35.0
    if "url_token_mismatch" in reasons:
        score -= 20.0
    if critical:
        score -= 100.0

    return {
        "title": title,
        "visible_text_chars": len(visible_text),
        "path_tokens": sorted(path_tokens),
        "token_coverage": round(token_coverage, 3),
        "title_token_coverage": round(title_token_coverage, 3),
        "reasons": reasons,
        "score": round(score, 3),
        "usable": not critical,
    }


def _select_preferred_page_capture(
    page_url: str,
    rendered_html: str,
    raw_source_html: str = "",
    rendered_markdown: str = "",
) -> Tuple[str, str, Dict[str, Any]]:
    """Select the HTML capture that best represents the requested page."""
    rendered_report = _html_quality_report(rendered_html, page_url, markdown=rendered_markdown)
    raw_report = _html_quality_report(raw_source_html, page_url) if raw_source_html else {}
    selected_html = rendered_html
    selected_source = "rendered"
    selection_reason = "rendered_capture_accepted"

    if raw_source_html and raw_report.get("usable"):
        rendered_reasons = set(rendered_report.get("reasons") or [])
        raw_reasons = set(raw_report.get("reasons") or [])
        if not rendered_report.get("usable"):
            selection_reason = "rendered_capture_unusable"
            selected_html = raw_source_html
            selected_source = "raw_source"
        elif "generic_mbzuai_title" in rendered_reasons and "generic_mbzuai_title" not in raw_reasons:
            selection_reason = "raw_source_has_page_specific_title"
            selected_html = raw_source_html
            selected_source = "raw_source"
        elif "generic_site_shell" in rendered_reasons and "generic_site_shell" not in raw_reasons:
            selection_reason = "raw_source_avoids_generic_site_shell"
            selected_html = raw_source_html
            selected_source = "raw_source"
        elif "url_token_mismatch" in rendered_reasons and "url_token_mismatch" not in raw_reasons:
            selection_reason = "raw_source_matches_requested_url"
            selected_html = raw_source_html
            selected_source = "raw_source"
        elif float(raw_report.get("score") or 0.0) >= float(rendered_report.get("score") or 0.0) + 12.0:
            selection_reason = "raw_source_has_higher_content_quality"
            selected_html = raw_source_html
            selected_source = "raw_source"

    selected_report = raw_report if selected_source == "raw_source" else rendered_report
    return selected_html, selected_source, {
        "selected_source": selected_source,
        "selection_reason": selection_reason,
        "selected": selected_report,
        "rendered": rendered_report,
        "raw_source": raw_report,
    }


def _markdown_plain_text(markdown: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", str(markdown or ""))
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#*_`>|-]+", " ", text)
    return " ".join(text.split())


def _clean_page_title_for_markdown(title: str) -> str:
    title = " ".join(str(title or "").split()).strip()
    return re.sub(r"\s+-\s+MBZUAI$", "", title, flags=re.I).strip() or title


def _ensure_markdown_has_page_title(markdown: str, html: str) -> str:
    markdown = str(markdown or "").strip()
    if not markdown:
        return ""
    title = _clean_page_title_for_markdown(_extract_html_title(html))
    if not title:
        return markdown
    plain = _markdown_plain_text(markdown).lower()
    if title.lower() in plain[:600]:
        return markdown
    return f"# {title}\n\n{markdown}"


def _starts_with_generic_site_shell(markdown: str) -> bool:
    plain = _markdown_plain_text(markdown).lower()
    return plain.startswith("thought curators for ai creators") or plain.startswith("thoughtcurators for ai creators")


def _markdown_has_navigation_prefix(markdown: str) -> bool:
    lines = [line.strip() for line in str(markdown or "").splitlines() if line.strip()]
    if len(lines) < 5:
        return False

    first_lines = lines[:24]
    link_lines = [line for line in first_lines if re.search(r"\[[^\]]+\]\([^)]+\)", line)]
    ordered_link_lines = [
        line for line in link_lines if re.match(r"^\s*\d+[.)]\s+\[[^\]]+\]\([^)]+\)", line)
    ]
    section_label_hits = 0
    for line in first_lines:
        normalized = re.sub(r"[^a-z ]+", " ", line.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized in COMMON_NAVIGATION_LABELS:
            section_label_hits += 1

    return (
        len(ordered_link_lines) >= 6
        or (section_label_hits >= 2 and len(link_lines) >= 4)
        or (section_label_hits >= 1 and len(ordered_link_lines) >= 4)
    )


def _markdown_looks_navigation_heavy(markdown: str) -> bool:
    lines = [line.strip() for line in str(markdown or "").splitlines() if line.strip()]
    if not lines:
        return False
    if _markdown_has_navigation_prefix(markdown):
        return True
    link_lines = [line for line in lines if re.search(r"\[[^\]]+\]\([^)]+\)", line)]
    initial_lines = [
        re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().lower()
        for line in lines[:12]
    ]
    nav_label_hits = 0
    for line in initial_lines:
        compact = re.sub(r"\s+", " ", line)
        if compact in COMMON_NAVIGATION_LABELS:
            nav_label_hits += 1
        elif any(label in compact for label in COMMON_NAVIGATION_LABELS if " " in label):
            nav_label_hits += 1
    if len(lines) >= 5 and nav_label_hits >= 3:
        return True
    if len(lines) >= 8 and len(link_lines) / len(lines) > 0.55:
        return True
    plain = _markdown_plain_text(markdown)
    link_text = " ".join(re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line) for line in link_lines)
    return bool(plain) and len(link_text) / max(len(plain), 1) > 0.55 and len(link_lines) >= 10


def _markdown_looks_thin_boilerplate(markdown: str, page_url: str = "") -> bool:
    plain = _markdown_plain_text(markdown)
    if not plain:
        return False

    word_count = len(re.findall(r"\w+", plain))
    if word_count >= 80:
        return False

    lower_plain = plain.lower()
    has_cookie_banner = "we use cookies" in lower_plain and "necessary and analytics cookies" in lower_plain
    has_contact_cta = (
        "interested in working with our faculty" in lower_plain
        or "fill out the form below" in lower_plain
        or "مهتم بالعمل مع أعضاء هيئة التدريس لدينا" in plain
        or "قم بتعبئة النموذج أدناه" in plain
    )
    if not (has_cookie_banner and has_contact_cta):
        return False

    path = urlparse(page_url).path.lower()
    return "/study/faculty/" in path or "/ar/study/faculty/" in path


def _markdown_quality_reason(markdown: str, page_url: str, *, html: str = "") -> str:
    plain = _markdown_plain_text(markdown)
    if not plain:
        return "empty_markdown"
    if _contains_error_or_block_text(plain):
        return "blocked_or_error_page"
    if urlparse(page_url).path.strip("/") and _starts_with_generic_site_shell(markdown):
        return "generic_site_shell"
    if _markdown_looks_navigation_heavy(markdown):
        return "navigation_heavy"
    if _markdown_looks_thin_boilerplate(markdown, page_url):
        return "thin_boilerplate"

    path_tokens = _url_semantic_tokens(page_url)
    if not path_tokens:
        return ""

    markdown_tokens = _semantic_tokens(plain)
    markdown_coverage = _token_coverage(path_tokens, markdown_tokens)
    lower_plain = plain.lower()
    if _is_generic_mbzuai_title(plain) and markdown_coverage < 0.34:
        return "generic_mbzuai_title"
    if any(phrase in lower_plain for phrase in GENERIC_SITE_SHELL_PHRASES) and markdown_coverage < 0.5:
        return "generic_site_shell"

    if html:
        html_report = _html_quality_report(html, page_url)
        if (
            html_report.get("usable")
            and float(html_report.get("token_coverage") or 0.0) >= 0.67
            and markdown_coverage < 0.34
        ):
            return "url_token_mismatch"
    return ""


def _best_content_node(soup: BeautifulSoup) -> Any:
    candidates: List[Any] = []
    for selector in (
        "main",
        "article",
        "[role='main']",
        ".page-main-content",
        ".entry-content",
        ".page-content",
        ".post-content",
        ".content",
    ):
        candidates.extend(soup.select(selector))

    for node in soup.find_all(["section", "div"]):
        text = node.get_text(" ", strip=True)
        if len(text) < 300:
            continue
        paragraphs = node.find_all(["p", "h1", "h2", "h3", "li"])
        if len(paragraphs) < 2:
            continue
        candidates.append(node)

    if not candidates and soup.body:
        candidates.append(soup.body)
    if not candidates:
        candidates.append(soup)

    def score(node: Any) -> int:
        text = node.get_text(" ", strip=True) if node else ""
        link_text = " ".join(anchor.get_text(" ", strip=True) for anchor in node.find_all("a")) if node else ""
        paragraphs = len(node.find_all("p")) if node else 0
        headings = len(node.find_all(["h1", "h2", "h3"])) if node else 0
        return len(text) - int(len(link_text) * 1.5) + (paragraphs * 120) + (headings * 80)

    return max(candidates, key=score)


def _remove_boilerplate_nodes(soup: BeautifulSoup) -> None:
    for tag in soup(["script", "style", "template", "noscript", "svg", "nav", "header", "footer", "form"]):
        tag.decompose()

    for tag in list(soup.find_all(True)):
        if tag.parent is None or not isinstance(getattr(tag, "attrs", None), dict):
            continue
        if tag.name in {"html", "body", "main", "article"}:
            continue
        attr_parts = [
            str(tag.get("id") or ""),
            " ".join(str(value) for value in (tag.get("class") or [])),
            str(tag.get("role") or ""),
            str(tag.get("aria-label") or ""),
        ]
        attr_text = " ".join(attr_parts).lower()
        if not attr_text:
            continue
        if any(token in attr_text for token in BOILERPLATE_ATTR_TOKENS):
            text = tag.get_text(" ", strip=True)
            link_text = " ".join(anchor.get_text(" ", strip=True) for anchor in tag.find_all("a"))
            link_ratio = len(link_text) / max(len(text), 1)
            paragraph_count = len(tag.find_all("p"))
            if link_ratio >= 0.35 or paragraph_count <= 2 or len(text) < 2500:
                tag.decompose()


def _bs4_html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html or "", "html.parser")
    _remove_boilerplate_nodes(soup)

    node = _best_content_node(soup)
    lines: List[str] = []
    seen: set[str] = set()
    block_tags = {"h1", "h2", "h3", "h4", "p", "li", "blockquote", "figcaption", "td", "th"}
    for element in node.find_all(list(block_tags), recursive=True):
        if element.find_parent(block_tags):
            continue
        text = " ".join(element.get_text(" ", strip=True).split())
        if not text or text in seen:
            continue
        seen.add(text)
        if element.name == "h1":
            lines.append(f"# {text}")
        elif element.name == "h2":
            lines.append(f"## {text}")
        elif element.name in {"h3", "h4"}:
            lines.append(f"### {text}")
        elif element.name == "li":
            lines.append(f"- {text}")
        else:
            lines.append(text)

    if not lines:
        text = " ".join(node.get_text(" ", strip=True).split())
        if text:
            lines.append(text)
    return "\n\n".join(lines).strip()


def _html_to_markdown(html: str, page_url: str = "") -> str:
    if not html or not str(html).strip():
        return ""

    extracted = ""
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=page_url or None,
            output_format="markdown",
            include_tables=True,
            include_links=True,
            include_images=False,
            favor_recall=True,
        ) or ""
    except Exception:
        extracted = ""

    if extracted and not _markdown_looks_navigation_heavy(extracted):
        return _ensure_markdown_has_page_title(extracted, html)

    fallback = _bs4_html_to_markdown(html)
    if fallback:
        return _ensure_markdown_has_page_title(fallback, html)
    return _ensure_markdown_has_page_title(extracted, html)


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
    frontier_seed_limit: Optional[int] = None,
    priority_urls: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Seed the crawl frontier from the start URL and sitemap-discovered URLs."""
    normalized_start = _normalize_http_url(start_url)
    if not normalized_start:
        return {"visited": [], "pending": [], "depths": {}, "pages_crawled": 0}

    pending = [{"url": normalized_start, "parent_url": None}]
    depths = {normalized_start: 0}
    seen = {normalized_start}
    remaining = max(0, int(max_pages) - 1)
    if frontier_seed_limit is not None:
        remaining = min(remaining, max(0, int(frontier_seed_limit)))

    ordered_candidates = [*(priority_urls or []), *discovered_urls]
    for candidate in ordered_candidates:
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


def _trim_crawl_state_to_budget(crawl_state: Optional[Dict[str, Any]], max_pages: int) -> Dict[str, Any]:
    """Normalize and cap crawler frontier state to the configured page budget."""
    if not isinstance(crawl_state, dict):
        return {"visited": [], "pending": [], "depths": {}, "pages_crawled": 0}

    budget = max(1, int(max_pages or 1))

    visited: list[str] = []
    seen: set[str] = set()
    for value in crawl_state.get("visited") or []:
        normalized = _normalize_http_url(value)
        if not normalized or normalized in seen:
            continue
        visited.append(normalized)
        seen.add(normalized)
        if len(visited) >= budget:
            break

    pending: list[dict[str, Optional[str]]] = []
    for item in crawl_state.get("pending") or []:
        url = None
        parent_url = None
        if isinstance(item, dict):
            url = _normalize_http_url(item.get("url"))
            parent_url = _normalize_http_url(item.get("parent_url"))
        else:
            url = _normalize_http_url(item)
        if not url or url in seen:
            continue
        if len(visited) + len(pending) >= budget:
            break
        pending.append({"url": url, "parent_url": parent_url})
        seen.add(url)

    depths = crawl_state.get("depths") or {}
    if isinstance(depths, dict):
        depths = {
            _normalize_http_url(url): depth
            for url, depth in depths.items()
            if _normalize_http_url(url) in seen
        }
    else:
        depths = {}

    pages_crawled = crawl_state.get("pages_crawled", 0)
    try:
        pages_crawled = int(pages_crawled)
    except (TypeError, ValueError):
        pages_crawled = 0
    pages_crawled = min(max(pages_crawled, 0), len(visited), budget)

    crawl_state["visited"] = visited
    crawl_state["pending"] = pending
    crawl_state["depths"] = depths
    crawl_state["pages_crawled"] = pages_crawled
    return crawl_state


def _has_resumable_crawl_state(crawl_state: Optional[Dict[str, Any]]) -> bool:
    """Return True only when crawl state contains usable frontier or progress."""
    if not isinstance(crawl_state, dict):
        return False

    pending = crawl_state.get("pending") or []
    visited = crawl_state.get("visited") or []
    pages_crawled = crawl_state.get("pages_crawled", 0)
    try:
        pages_crawled = int(pages_crawled)
    except (TypeError, ValueError):
        pages_crawled = 0

    return bool(pending or visited or pages_crawled > 0)


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


def _compact_json_ld(value: Any, *, max_items: int = 20, max_string_len: int = 2000, depth: int = 0) -> Any:
    if depth > 5:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text[:max_string_len]
    if isinstance(value, list):
        compacted = [
            _compact_json_ld(item, max_items=max_items, max_string_len=max_string_len, depth=depth + 1)
            for item in value[:max_items]
        ]
        return [item for item in compacted if item not in (None, "", [], {})]
    if isinstance(value, dict):
        compacted = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= max_items:
                break
            compact = _compact_json_ld(item, max_items=max_items, max_string_len=max_string_len, depth=depth + 1)
            if compact not in (None, "", [], {}):
                compacted[str(key)] = compact
        return compacted
    return value


def _page_graph_node_id(url: str) -> str:
    return f"page:{_url_digest(url)[:24]}"


def _page_graph_edge_id(source_url: str, target_url: str) -> str:
    return f"edge:{_url_digest(f'{source_url}|{target_url}')[:24]}"


def _append_meta_value(target: Dict[str, Any], key: str, value: str) -> None:
    normalized_key = (key or "").strip().lower()
    normalized_value = " ".join(str(value or "").split()).strip()
    if not normalized_key or not normalized_value:
        return
    current = target.get(normalized_key)
    if current is None:
        target[normalized_key] = normalized_value
    elif isinstance(current, list):
        if normalized_value not in current:
            current.append(normalized_value)
    elif current != normalized_value:
        target[normalized_key] = [current, normalized_value]


def _extract_page_metadata(
    html: str,
    page_url: str,
    *,
    status_code: Any = None,
    html_path: str = "",
    markdown_path: str = "",
    depth: Any = None,
    images_count: int = 0,
    videos_count: int = 0,
    media_count: int = 0,
    capture_source: str = "",
    content_quality: Optional[Dict[str, Any]] = None,
    markdown_source: str = "",
    markdown_quality_reason: str = "",
) -> Dict[str, Any]:
    soup = BeautifulSoup(html or "", "html.parser")
    title = ""
    if soup.title:
        title = " ".join(soup.title.get_text(" ", strip=True).split())

    meta_tags: Dict[str, Any] = {}
    for tag in soup.find_all("meta"):
        key = tag.get("name") or tag.get("property") or tag.get("http-equiv") or tag.get("itemprop")
        content = tag.get("content")
        if key and content:
            _append_meta_value(meta_tags, str(key), str(content))

    canonical_url = ""
    canonical_tag = soup.find("link", rel=lambda value: value and "canonical" in str(value).lower())
    if canonical_tag:
        canonical_url = _normalize_http_url(canonical_tag.get("href"), base_url=page_url) or ""

    alternate_urls: List[Dict[str, str]] = []
    for link in soup.find_all("link"):
        rel_value = " ".join(link.get("rel") or []) if isinstance(link.get("rel"), list) else str(link.get("rel") or "")
        if "alternate" not in rel_value.lower():
            continue
        href = _normalize_http_url(link.get("href"), base_url=page_url)
        if not href:
            continue
        alternate_urls.append(
            {
                "url": href,
                "hreflang": str(link.get("hreflang") or ""),
                "type": str(link.get("type") or ""),
            }
        )

    headings = {
        tag_name: [
            " ".join(tag.get_text(" ", strip=True).split())[:240]
            for tag in soup.find_all(tag_name)
            if tag.get_text(" ", strip=True)
        ][:50]
        for tag_name in ("h1", "h2", "h3")
    }

    json_ld = []
    for item in _json_ld_items(soup):
        compact = _compact_json_ld(item)
        if compact:
            json_ld.append(compact)

    parsed = urlparse(page_url)
    return {
        "url": page_url,
        "status_code": status_code,
        "host": parsed.hostname or "",
        "path": parsed.path or "/",
        "depth": depth,
        "title": title,
        "description": meta_tags.get("description") or meta_tags.get("og:description") or "",
        "canonical_url": canonical_url,
        "language": str((soup.html or {}).get("lang") or ""),
        "robots": meta_tags.get("robots") or "",
        "meta_tags": meta_tags,
        "alternate_urls": alternate_urls,
        "headings": headings,
        "json_ld": json_ld[:20],
        "html_path": html_path,
        "markdown_path": markdown_path,
        "content_bytes": len((html or "").encode("utf-8")),
        "images_count": images_count,
        "videos_count": videos_count,
        "media_count": media_count,
        "capture_source": capture_source,
        "content_quality": content_quality or {},
        "markdown_source": markdown_source,
        "markdown_quality_reason": markdown_quality_reason,
    }


def _link_type_for_url(url: str, allowed_domains: set[str]) -> str:
    extension = _url_extension(url)
    if extension in DOWNLOADABLE_EXTENSIONS:
        return "document"
    host = (urlparse(url).hostname or "").lower()
    if any(_host_matches_domain(host, allowed) for allowed in allowed_domains):
        return "internal"
    return "external"


def _extract_page_links(
    result: Any,
    html: str,
    base_url: str,
    *,
    allowed_domains: set[str],
) -> List[Dict[str, Any]]:
    links_by_target: Dict[str, Dict[str, Any]] = {}

    def add_link(href: Any, *, anchor_text: str = "", rel: Any = None, source: str = "") -> None:
        target_url = _normalize_http_url(str(href or ""), base_url=base_url)
        if not target_url or target_url == base_url:
            return
        item = links_by_target.setdefault(
            target_url,
            {
                "source_url": base_url,
                "target_url": target_url,
                "link_type": _link_type_for_url(target_url, allowed_domains),
                "anchor_texts": [],
                "rels": [],
                "sources": [],
            },
        )
        text = " ".join(str(anchor_text or "").split()).strip()
        if text and text not in item["anchor_texts"]:
            item["anchor_texts"].append(text[:180])
        rel_values = rel if isinstance(rel, list) else [rel] if rel else []
        for value in rel_values:
            rel_text = str(value or "").strip()
            if rel_text and rel_text not in item["rels"]:
                item["rels"].append(rel_text[:80])
        if source and source not in item["sources"]:
            item["sources"].append(source)

    links = getattr(result, "links", None) or {}
    for group in ("internal", "external"):
        for item in links.get(group, []) or []:
            if isinstance(item, dict):
                add_link(
                    item.get("href"),
                    anchor_text=str(item.get("text") or item.get("title") or ""),
                    rel=item.get("rel"),
                    source=f"crawl4ai:{group}",
                )

    if html:
        soup = BeautifulSoup(html, "html.parser")
        for anchor in soup.find_all("a", href=True):
            add_link(
                anchor.get("href"),
                anchor_text=anchor.get_text(" ", strip=True),
                rel=anchor.get("rel"),
                source="html:a",
            )

    return list(links_by_target.values())


def _build_page_link_graph_payload(
    *,
    page_metadata: Dict[str, Dict[str, Any]],
    page_links: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    node_urls: set[str] = set(page_metadata.keys())
    for source_url, links in page_links.items():
        node_urls.add(source_url)
        for link in links or []:
            target_url = str(link.get("target_url") or "")
            if target_url:
                node_urls.add(target_url)

    nodes = []
    for url in sorted(node_urls):
        metadata = page_metadata.get(url) or {}
        nodes.append(
            {
                "id": _page_graph_node_id(url),
                "url": url,
                "node_type": "page" if url in page_metadata else "discovered_url",
                "label": metadata.get("title") or url,
                "properties": {
                    "status_code": metadata.get("status_code"),
                    "canonical_url": metadata.get("canonical_url"),
                    "depth": metadata.get("depth"),
                    "description": metadata.get("description"),
                    "path": metadata.get("path"),
                    "host": metadata.get("host"),
                },
            }
        )

    edges = []
    for source_url, links in sorted(page_links.items()):
        for link in links or []:
            target_url = str(link.get("target_url") or "")
            if not target_url:
                continue
            edges.append(
                {
                    "id": _page_graph_edge_id(source_url, target_url),
                    "edge_type": "LINKS_TO",
                    "source_id": _page_graph_node_id(source_url),
                    "target_id": _page_graph_node_id(target_url),
                    "source_url": source_url,
                    "target_url": target_url,
                    "properties": {
                        "link_type": link.get("link_type"),
                        "anchor_texts": list(link.get("anchor_texts") or [])[:5],
                        "rels": list(link.get("rels") or [])[:5],
                        "sources": list(link.get("sources") or [])[:5],
                    },
                }
            )

    link_type_counts: Dict[str, int] = {}
    for edge in edges:
        link_type = str((edge.get("properties") or {}).get("link_type") or "unknown")
        link_type_counts[link_type] = link_type_counts.get(link_type, 0) + 1

    return {
        "schema_version": 1,
        "graph_type": "website_page_link_graph",
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "crawled_page_count": len(page_metadata),
            "link_type_counts": link_type_counts,
        },
    }


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


def _raw_source_candidate_urls(page_url: str) -> List[str]:
    normalized = _normalize_http_url(page_url)
    if not normalized:
        return []
    candidates = [normalized]
    parsed = urlparse(normalized)
    if parsed.query:
        return candidates

    path = parsed.path or ""
    if path and not path.endswith("/") and not Path(path).suffix:
        candidates.append(urlunparse((parsed.scheme, parsed.netloc, f"{path}/", "", "", "")))
    elif path.endswith("/") and path != "/":
        candidates.append(urlunparse((parsed.scheme, parsed.netloc, path.rstrip("/"), "", "", "")))
    return list(dict.fromkeys(candidates))


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
        # Helper-level tests and resume recovery can construct a crawler from
        # persisted state before the full execute() initialization path runs.
        # Missing optional policy collections must default closed/safely.
        excluded_subdomains = getattr(self, "excluded_subdomains", set()) or set()
        if any(_host_matches_domain(host, excluded) for excluded in excluded_subdomains):
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


class SkipQueryFilter(URLFilter):
    """Reject crawl-frontier URLs with query params unless explicitly allowed."""

    def __init__(self, allow_query_urls: bool = False, allowed_query_param_names: Iterable[str] | None = None):
        super().__init__(name="SkipQueryFilter")
        self.allow_query_urls = bool(allow_query_urls)
        self.allowed_query_param_names = {
            str(value).strip().lower()
            for value in (allowed_query_param_names or [])
            if str(value).strip()
        }

    def apply(self, url: str) -> bool:
        normalized = _normalize_http_url(url)
        if not normalized:
            return False
        parsed = urlparse(normalized)
        if not parsed.query:
            return True
        if not self.allow_query_urls:
            return False
        if not self.allowed_query_param_names:
            return True
        query_names = {
            str(key).strip().lower()
            for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
            if str(key).strip()
        }
        return query_names.issubset(self.allowed_query_param_names)


class PathPrefixFilter(URLFilter):
    """Reject configured low-value path prefixes from the crawl frontier."""

    def __init__(self, excluded_path_prefixes: Iterable[str] | None = None):
        super().__init__(name="PathPrefixFilter")
        self.excluded_path_prefixes = {
            _normalize_path_prefix(value)
            for value in (excluded_path_prefixes or [])
            if _normalize_path_prefix(value)
        }

    def apply(self, url: str) -> bool:
        return not _url_matches_path_prefix(url, self.excluded_path_prefixes)


@register_stage
class Crawl4AICrawler(CrawlerStage):
    name = "crawl4ai"
    description = "Resumable Crawl4AI deep crawler with sitemap seeding and controlled downloads."

    def _has_crawl_output(self) -> bool:
        return any(
            [
                int(self.stats.get("pages_scraped") or 0) > 0,
                int(self.stats.get("markdown_written") or 0) > 0,
                int(self.stats.get("documents_downloaded") or 0) > 0,
                bool(self.url_mapping),
                bool(self.url_to_md_mapping),
            ]
        )

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
        self.respect_robots_txt = bool(self.config.get("respect_robots_txt", True))
        self.fetch_concurrency = max(1, int(self.config.get("fetch_concurrency", 5)))
        self.download_concurrency = max(1, int(self.config.get("download_concurrency", 10)))
        self.sitemap_batch_crawl = bool(self.config.get("sitemap_batch_crawl", False))
        self.sitemap_crawl_batch_size = max(
            1,
            int(
                self.config.get(
                    "sitemap_crawl_batch_size",
                    max(self.fetch_concurrency * 4, 20),
                )
            ),
        )
        self.sitemap_failed_url_retry_attempts = max(
            0, int(self.config.get("sitemap_failed_url_retry_attempts", 2))
        )
        self.sitemap_failed_retry_backoff = max(
            0.0, float(self.config.get("sitemap_failed_retry_backoff_sec", 2.0))
        )
        self.retry_recoverable_skipped_on_resume = bool(
            self.config.get("retry_recoverable_skipped_on_resume", True)
        )
        self.recoverable_skip_max_retries = max(
            0, int(self.config.get("recoverable_skip_max_retries", 2))
        )
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
        self.raw_source_retry_attempts = max(1, int(self.config.get("raw_source_retry_attempts", 2)))
        self.raw_source_retry_backoff = max(0.0, float(self.config.get("raw_source_retry_backoff_sec", 0.75)))
        self.checkpoint_flush_interval = max(
            0.0, float(self.config.get("checkpoint_flush_interval_sec", 5.0))
        )
        self.extract_images = bool(self.config.get("extract_images", True))
        self.extract_videos = bool(self.config.get("extract_videos", True))
        self.download_page_images = bool(self.config.get("download_page_images", True))
        self.fetch_video_transcripts = bool(self.config.get("fetch_video_transcripts", True))
        self.parse_source_html_for_media = bool(self.config.get("parse_source_html_for_media", True))
        self.validate_source_html = bool(self.config.get("validate_source_html", True))
        self.validate_source_html_mode = str(
            self.config.get("validate_source_html_mode", "on_quality_warning") or "on_quality_warning"
        ).strip().lower()
        self.headless = bool(self.config.get("headless", True))
        self.ignore_https_errors = bool(self.config.get("ignore_https_errors", False))
        self.require_https = bool(self.config.get("require_https", True))
        self.enable_stealth = bool(self.config.get("enable_stealth", False))
        self.allowed_domains = self._resolve_allowed_domains()
        self.excluded_subdomains = {
            value.lower()
            for value in (self.config.get("excluded_subdomains") or [])
            if value
        }
        self.include_external = self._should_include_external_links()
        self.proxy = self.config.get("proxy")
        self.allow_query_urls = bool(self.config.get("allow_query_urls", False))
        self.allowed_query_param_names = {
            str(value).strip().lower()
            for value in (self.config.get("allowed_query_param_names") or [])
            if str(value).strip()
        }
        self.excluded_path_prefixes = {
            _normalize_path_prefix(value)
            for value in (self.config.get("excluded_path_prefixes") or [])
            if _normalize_path_prefix(value)
        }
        self.frontier_empty_timeout = max(30, int(self.config.get("frontier_empty_timeout_sec", 180) or 180))
        self.crawl_stall_timeout = max(self.frontier_empty_timeout, int(self.config.get("crawl_stall_timeout_sec", 900) or 900))
        self.priority_seed_urls = []
        for value in self.config.get("priority_seed_urls") or self.config.get("required_seed_urls") or []:
            url = _normalize_http_url(value, self.start_url)
            host = (urlparse(url).hostname or "").lower() if url else ""
            if (
                url
                and self._allow_frontier_url(url)
                and any(_host_matches_domain(host, allowed) for allowed in self.allowed_domains)
            ):
                self.priority_seed_urls.append(url)

        self.html_dir = ensure_dir(ctx.work_dir / "html")
        self.md_dir = ensure_dir(ctx.work_dir / "markdown")
        self.download_dir = ensure_dir(ctx.work_dir / "downloads")
        self.images_dir = ensure_dir(ctx.work_dir / "downloaded_page_images")

        self.mapping_file = ctx.work_dir / MAPPINGS_FILENAME
        self.page_images_file = ctx.work_dir / PAGE_IMAGES_FILENAME
        self.page_videos_file = ctx.work_dir / PAGE_VIDEOS_FILENAME
        self.page_media_file = ctx.work_dir / PAGE_MEDIA_FILENAME
        self.page_metadata_file = ctx.work_dir / PAGE_METADATA_FILENAME
        self.page_link_graph_file = ctx.work_dir / PAGE_LINK_GRAPH_FILENAME
        self.url_to_md_mapping_file = ctx.work_dir / URL_TO_MD_FILENAME
        self.crawl_state_file = ctx.work_dir / CRAWL_STATE_FILENAME
        self.runtime_state_file = ctx.work_dir / RUNTIME_STATE_FILENAME
        self.sitemap_state_file = ctx.work_dir / SITEMAP_STATE_FILENAME

        self.stats = {
            "pages_scraped": 0,
            "pages_failed": 0,
            "markdown_written": 0,
            "documents_downloaded": 0,
            "http_fallback_pages": 0,
            "http_fallback_retries": 0,
            "source_html_validations": 0,
            "source_html_replacements": 0,
            "low_quality_markdown_suppressed": 0,
            "content_quality_warnings": 0,
            "images_extracted": 0,
            "images_downloaded": 0,
            "videos_extracted": 0,
            "video_transcripts_fetched": 0,
            "bytes_downloaded": 0,
            "sitemap_urls_seeded": 0,
            "sitemap_batches_completed": 0,
            "skipped_urls": 0,
            "excluded_frontier_urls": 0,
            "recoverable_skips_exhausted": 0,
        }
        self.url_mapping: Dict[str, str] = {}
        self.url_to_md_mapping: Dict[str, str] = {}
        self.page_images: Dict[str, List[Dict[str, str]]] = {}
        self.page_videos: Dict[str, List[Dict[str, Any]]] = {}
        self.page_media: Dict[str, List[Dict[str, Any]]] = {}
        self.page_metadata: Dict[str, Dict[str, Any]] = {}
        self.page_links: Dict[str, List[Dict[str, Any]]] = {}
        self.downloaded_images: Dict[str, str] = {}
        self.recoverable_skip_retries: Dict[str, int] = {}
        self.crawl_state: Dict[str, Any] = {}
        self.discovered_sitemaps: Dict[str, Any] = {"sources": [], "urls": []}
        self._last_flush_at = 0.0
        self._session: Optional[aiohttp.ClientSession] = None
        self._download_semaphore = asyncio.Semaphore(self.download_concurrency)
        self._last_crawl_state_update_at = time.time()
        self._last_result_seen_at = 0.0
        self._crawl_watchdog_error: Optional[str] = None

        self._load_runtime_state(ctx.checkpoint)
        if self.retry_recoverable_skipped_on_resume:
            self._requeue_recoverable_skipped_urls()

        try:
            await self._open_http_session()
            if not self._allow_frontier_url(self.start_url):
                raise ValueError("crawler.start_url is outside the allowed egress policy")

            if not _has_resumable_crawl_state(self.crawl_state):
                sitemap_urls = []
                if self.config.get("sitemap_enabled", True):
                    sitemap_urls = await self._discover_sitemap_urls()
                    self.stats["sitemap_urls_seeded"] = len(sitemap_urls)
                self.crawl_state = _build_initial_crawl_state(
                    self.start_url,
                    sitemap_urls,
                    self.max_pages,
                    frontier_seed_limit=self.config.get("sitemap_frontier_seed_limit"),
                    priority_urls=self.priority_seed_urls,
                )
                self._write_crawl_state_file()

            browser_config = self._build_browser_config()
            run_config = self._build_run_config()

            crawler_holder = {"crawler": AsyncWebCrawler(config=browser_config)}
            await crawler_holder["crawler"].__aenter__()
            try:
                if self._should_use_seed_batch_crawl():
                    await self._crawl_seed_frontier(crawler_holder, run_config, browser_config)
                else:
                    results = crawler_holder["crawler"].arun(url=self.start_url, config=run_config)
                    if asyncio.iscoroutine(results):
                        crawl_task = asyncio.create_task(results)
                        watchdog = asyncio.create_task(self._watch_crawl_health(crawl_task))
                        try:
                            results = await crawl_task
                        except asyncio.CancelledError:
                            if self._crawl_watchdog_error:
                                raise RuntimeError(self._crawl_watchdog_error) from None
                            raise
                        finally:
                            watchdog.cancel()
                            with contextlib.suppress(asyncio.CancelledError):
                                await watchdog
                    await self._consume_crawl_results(results)
            finally:
                try:
                    await crawler_holder["crawler"].__aexit__(None, None, None)
                except Exception as close_exc:
                    if _is_benign_browser_close_error(close_exc):
                        logger.warning("Ignoring benign browser shutdown error after crawler checkpoint flush: %s", close_exc)
                    else:
                        raise

            self._flush_runtime_state(force=True)

            if bool(self.config.get("fail_on_empty_result", True)) and not self._has_crawl_output():
                return StageResult.failure(
                    "Crawler produced no pages, markdown, or downloaded documents. Check start_url, allowed_domains, robots/auth requirements, or remote blocking.",
                    checkpoint={"runtime_state_file": str(self.runtime_state_file)},
                )

            return StageResult.success(
                outputs={
                    "html_dir": str(self.html_dir),
                    "md_dir": str(self.md_dir),
                    "download_dir": str(self.download_dir),
                    "mapping_file": str(self.mapping_file),
                    "page_images_file": str(self.page_images_file),
                    "page_videos_file": str(self.page_videos_file),
                    "page_media_file": str(self.page_media_file),
                    "page_metadata_file": str(self.page_metadata_file),
                    "page_link_graph_file": str(self.page_link_graph_file),
                    "runtime_state_file": str(self.runtime_state_file),
                    "crawler_runtime_state_file": str(self.runtime_state_file),
                    "images_dir": str(self.images_dir),
                    "md_mapping_file": str(self.url_to_md_mapping_file),
                },
                metrics=self._build_metrics(),
                checkpoint={"runtime_state_file": str(self.runtime_state_file)},
                artifacts=[
                    ctx.make_artifact(
                        self.page_metadata_file,
                        artifact_type="page_metadata",
                        role="website_page_metadata",
                        metadata={"records": len(self.page_metadata)},
                    ),
                    ctx.make_artifact(
                        self.page_link_graph_file,
                        artifact_type="page_link_graph",
                        role="website_page_connections",
                        metadata={
                            "nodes": len(self.page_metadata),
                            "edges": sum(len(links) for links in self.page_links.values()),
                        },
                    ),
                ],
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

        self.page_metadata = load_json_safe(self.page_metadata_file, {}) or {}
        self.page_metadata.update(state.get("page_metadata") or {})
        self.recoverable_skip_retries = {
            str(url): int(count)
            for url, count in (state.get("recoverable_skip_retries") or {}).items()
            if str(url)
        }

        graph_payload = load_json_safe(self.page_link_graph_file, {}) or {}
        graph_links: Dict[str, List[Dict[str, Any]]] = {}
        if isinstance(graph_payload, dict):
            for edge in graph_payload.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                source_url = str(edge.get("source_url") or "")
                target_url = str(edge.get("target_url") or "")
                if not source_url or not target_url:
                    continue
                properties = edge.get("properties") if isinstance(edge.get("properties"), dict) else {}
                graph_links.setdefault(source_url, []).append(
                    {
                        "source_url": source_url,
                        "target_url": target_url,
                        "link_type": properties.get("link_type") or _link_type_for_url(target_url, self.allowed_domains),
                        "anchor_texts": list(properties.get("anchor_texts") or []),
                        "rels": list(properties.get("rels") or []),
                        "sources": list(properties.get("sources") or []),
                    }
                )
        self.page_links = graph_links
        self.page_links.update(state.get("page_links") or {})

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
        normalized_crawl_state = _trim_crawl_state_to_budget(crawl_state, self.max_pages)
        self.crawl_state = (
            normalized_crawl_state
            if _has_resumable_crawl_state(normalized_crawl_state)
            else {}
        )
        if self.crawl_state:
            self.crawl_state["pending"] = self._filter_pending_items(
                self.crawl_state.get("pending") or []
            )

        sitemap_state = load_json_safe(self.sitemap_state_file, {}) or {}
        self.discovered_sitemaps = {
            "sources": sitemap_state.get("sources", []),
            "urls": sitemap_state.get("urls", []),
        }

    def _mark_url_excluded_from_frontier(self, page_url: str, reason: str = "path_prefix") -> None:
        normalized = _normalize_http_url(page_url)
        if not normalized:
            return
        existing = str(self.url_mapping.get(normalized) or "")
        if existing.startswith("SKIPPED_EXCLUDED_FRONTIER"):
            return
        already_skipped = existing.startswith("SKIPPED")
        self.url_mapping[normalized] = f"SKIPPED_EXCLUDED_FRONTIER:{reason}"
        if not already_skipped:
            self.stats["skipped_urls"] += 1
        self.stats["excluded_frontier_urls"] += 1

    def _filter_pending_items(
        self,
        items: Iterable[Dict[str, Any]],
    ) -> List[Dict[str, Optional[str]]]:
        filtered: List[Dict[str, Optional[str]]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            normalized = _normalize_http_url(item.get("url"))
            if not normalized or normalized in seen:
                continue
            if not self._allow_frontier_url(normalized):
                self._mark_url_excluded_from_frontier(normalized)
                continue
            filtered.append(
                {
                    "url": normalized,
                    "parent_url": _normalize_http_url(item.get("parent_url")),
                }
            )
            seen.add(normalized)
        return filtered

    def _requeue_recoverable_skipped_urls(self) -> List[str]:
        if not self.crawl_state or not self.url_mapping:
            return []

        pending = self._filter_pending_items(self.crawl_state.get("pending") or [])
        pending_urls = {
            _normalize_http_url(item.get("url"))
            for item in pending
            if isinstance(item, dict)
        }
        visited = [
            str(url)
            for url in (self.crawl_state.get("visited") or [])
            if isinstance(url, str) and url
        ]
        requeued: List[str] = []
        exhausted = 0
        for url, reason in list(self.url_mapping.items()):
            normalized = _normalize_http_url(url)
            if not normalized or not _is_recoverable_crawl_skip_reason(reason):
                continue
            if not self._allow_frontier_url(normalized):
                self._mark_url_excluded_from_frontier(normalized)
                continue
            attempts = int(self.recoverable_skip_retries.get(normalized) or 0)
            if attempts >= self.recoverable_skip_max_retries:
                exhausted += 1
                continue
            self.recoverable_skip_retries[normalized] = attempts + 1
            self.url_mapping.pop(url, None)
            self.url_mapping.pop(normalized, None)
            if normalized not in pending_urls:
                pending.insert(0, {"url": normalized, "parent_url": None})
                pending_urls.add(normalized)
            requeued.append(normalized)

        if exhausted:
            self.stats["recoverable_skips_exhausted"] += exhausted
            logger.warning(
                "Left %d recoverable skipped URL(s) in skipped state after reaching retry cap=%d.",
                exhausted,
                self.recoverable_skip_max_retries,
            )

        if not requeued:
            self.crawl_state["pending"] = pending
            return []

        requeued_set = set(requeued)
        self.crawl_state["visited"] = [url for url in visited if url not in requeued_set]
        self.crawl_state["pending"] = pending
        self.stats["pages_failed"] = max(0, int(self.stats.get("pages_failed", 0)) - len(requeued))
        self.stats["skipped_urls"] = max(0, int(self.stats.get("skipped_urls", 0)) - len(requeued))
        logger.info(
            "Requeued %d recoverable skipped URL(s) from checkpoint for retry.",
            len(requeued),
        )
        return requeued

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
        if self.respect_robots_txt:
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
            if not self._url_allowed_for_fetch(normalized):
                logger.debug("Skipping sitemap outside allowed egress policy: %s", normalized)
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
                if not self._allow_frontier_url(normalized_page):
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

    def _url_allowed_for_fetch(self, url: str) -> bool:
        normalized = _normalize_http_url(url)
        parsed = urlparse(normalized or "")
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or not host:
            return False
        if getattr(self, "require_https", True) and parsed.scheme != "https":
            return False
        excluded_subdomains = getattr(self, "excluded_subdomains", set()) or set()
        if any(_host_matches_domain(host, excluded) for excluded in excluded_subdomains):
            return False
        allowed_domains = getattr(self, "allowed_domains", None)
        if not allowed_domains:
            configured_host = (
                urlparse(str(getattr(self, "start_url", "") or "")).hostname or ""
            ).lower()
            allowed_domains = {configured_host} if configured_host else set()
        if not allowed_domains:
            return False
        if not any(_host_matches_domain(host, allowed) for allowed in allowed_domains):
            return False
        if _host_resolves_to_private_or_reserved(host):
            return False
        return True

    def _allow_frontier_url(self, url: str) -> bool:
        normalized = _normalize_http_url(url)
        parsed = urlparse(normalized or "")
        if not parsed.scheme or not parsed.netloc:
            return False
        if not self._url_allowed_for_fetch(normalized or ""):
            return False
        if _url_matches_path_prefix(normalized or "", self.excluded_path_prefixes):
            return False
        if parsed.query:
            if not self.allow_query_urls:
                return False
            if self.allowed_query_param_names:
                names = {
                    str(key).strip().lower()
                    for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
                    if str(key).strip()
                }
                if not names.issubset(self.allowed_query_param_names):
                    return False
        return True

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
                SkipQueryFilter(
                    allow_query_urls=self.allow_query_urls,
                    allowed_query_param_names=self.allowed_query_param_names,
                ),
                PathPrefixFilter(self.excluded_path_prefixes),
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
            # crawl4ai 0.8.0 returns empty async result streams for some valid
            # deep-crawl runs when stream=True. Default to the stable list mode.
            "stream": bool(self.config.get("stream_results", False)),
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
            "check_robots_txt": self.respect_robots_txt,
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

    async def _collect_crawl_results(self, results: Any) -> List[Any]:
        collected: List[Any] = []
        if inspect.isasyncgen(results):
            async for result in results:
                collected.append(result)
        elif isinstance(results, list):
            collected.extend(results)
        elif results is not None:
            collected.append(results)
        return collected

    def _should_use_seed_batch_crawl(self) -> bool:
        if not self.sitemap_batch_crawl:
            return False
        pending = self.crawl_state.get("pending") or []
        if len(pending) <= self.sitemap_crawl_batch_size:
            return False
        if self.stats.get("sitemap_urls_seeded", 0) > 0:
            return True
        return bool(self.config.get("sitemap_enabled", True)) and len(pending) > 1

    def _set_crawl_state(
        self,
        *,
        visited: List[str],
        pending: List[Dict[str, Optional[str]]],
        depths: Dict[str, int],
        pages_crawled: int,
    ) -> None:
        self.crawl_state = _trim_crawl_state_to_budget(
            {
                "visited": visited,
                "pending": self._filter_pending_items(pending),
                "depths": depths,
                "pages_crawled": pages_crawled,
            },
            self.max_pages,
        )
        self._last_crawl_state_update_at = time.time()

    async def _crawl_seed_frontier(self, crawler_holder: Dict[str, Any], run_config: CrawlerRunConfig, browser_config: Any) -> None:
        pending: List[Dict[str, Optional[str]]] = [
            {"url": item.get("url"), "parent_url": item.get("parent_url")}
            for item in (self.crawl_state.get("pending") or [])
            if isinstance(item, dict) and item.get("url")
        ]
        visited: List[str] = [
            str(url)
            for url in (self.crawl_state.get("visited") or [])
            if isinstance(url, str) and url
        ]
        visited_set = set(visited)
        depths = dict(self.crawl_state.get("depths") or {})
        pages_crawled = int(self.crawl_state.get("pages_crawled") or 0)
        last_recycle_page_count = pages_crawled
        batch_config = run_config.clone(deep_crawl_strategy=None, stream=False)

        logger.info(
            "Using bounded sitemap seed crawl: pending=%d batch_size=%d fetch_concurrency=%d",
            len(pending),
            self.sitemap_crawl_batch_size,
            self.fetch_concurrency,
        )

        while pending and pages_crawled < self.max_pages:
            batch = pending[: self.sitemap_crawl_batch_size]
            urls = [str(item["url"]) for item in batch if item.get("url")]
            if not urls:
                pending = pending[len(batch):]
                continue

            self._set_crawl_state(
                visited=visited,
                pending=pending,
                depths=depths,
                pages_crawled=pages_crawled,
            )
            requeued = self._requeue_recoverable_skipped_urls()
            if requeued:
                pending = [
                    item
                    for item in (self.crawl_state.get("pending") or [])
                    if isinstance(item, dict) and item.get("url")
                ]
                visited = [
                    str(url)
                    for url in (self.crawl_state.get("visited") or [])
                    if isinstance(url, str) and url
                ]
                visited_set = set(visited)
                pages_crawled = int(self.crawl_state.get("pages_crawled") or pages_crawled)
            self._flush_runtime_state(force=True)

            batch_results = await self._collect_crawl_results(
                await crawler_holder["crawler"].arun_many(urls=urls, config=batch_config)
            )
            returned_urls = {
                normalized
                for result in batch_results
                for normalized in [_normalize_http_url(getattr(result, "url", None))]
                if normalized
            }

            failed_batch_urls: Dict[str, Dict[str, Any]] = {}
            for result in batch_results:
                result_url = _normalize_http_url(getattr(result, "url", None))
                if result_url and result_url not in visited_set:
                    visited.append(result_url)
                    visited_set.add(result_url)
                if getattr(result, "success", False):
                    pages_crawled += 1
                processed = await self._process_result(result, mark_failure=False)
                if not processed and result_url:
                    failed_batch_urls[result_url] = {
                        "status_code": getattr(result, "status_code", None),
                        "error_message": getattr(result, "error_message", ""),
                    }

            for item in batch:
                url = _normalize_http_url(item.get("url"))
                if not url:
                    continue
                if url not in visited_set:
                    visited.append(url)
                    visited_set.add(url)
                if url not in returned_urls and url not in self.url_mapping:
                    failed_batch_urls[url] = {
                        "status_code": None,
                        "error_message": "SKIPPED_NO_RESULT",
                    }

            if failed_batch_urls:
                recovered = 0
                for url, failure in failed_batch_urls.items():
                    if await self._recover_url_with_http_retry(url):
                        recovered += 1
                    else:
                        self._mark_url_skipped(
                            url,
                            status_code=failure.get("status_code"),
                            error_message=failure.get("error_message"),
                            reason="SKIPPED_NO_RESULT" if failure.get("error_message") == "SKIPPED_NO_RESULT" else "SKIPPED_ERROR",
                        )
                if recovered:
                    logger.info(
                        "Recovered %d/%d failed sitemap batch URL(s) through HTTP retry.",
                        recovered,
                        len(failed_batch_urls),
                    )

            pending = pending[len(batch):]
            self.stats["sitemap_batches_completed"] = self.stats.get("sitemap_batches_completed", 0) + 1
            pages_crawled = max(pages_crawled, int(self.stats.get("pages_scraped", 0)))
            self._set_crawl_state(
                visited=visited,
                pending=pending,
                depths=depths,
                pages_crawled=pages_crawled,
            )
            self._flush_runtime_state(force=True)

            # Playwright headless browser recycling to prevent slow RAM memory leaks
            pages_since_last_recycle = pages_crawled - last_recycle_page_count
            if pages_since_last_recycle >= 80:
                logger.info(
                    "Recycling Playwright browser context after crawling %d pages (total: %d pages)...",
                    pages_since_last_recycle,
                    pages_crawled
                )
                try:
                    await crawler_holder["crawler"].__aexit__(None, None, None)
                except Exception as recycle_close_exc:
                    if _is_benign_browser_close_error(recycle_close_exc):
                        logger.warning("Ignoring benign browser shutdown error during recycling: %s", recycle_close_exc)
                    else:
                        logger.warning("Error closing crawler context during recycling (non-fatal): %s", recycle_close_exc)

                # Instantiate and context-enter a fresh crawler instance
                crawler_holder["crawler"] = AsyncWebCrawler(config=browser_config)
                await crawler_holder["crawler"].__aenter__()
                last_recycle_page_count = pages_crawled

            if self.stats["sitemap_batches_completed"] <= 3 or self.stats["sitemap_batches_completed"] % 10 == 0:
                logger.info(
                    "Sitemap seed crawl progress: batches=%d pages_scraped=%d pages_failed=%d pending=%d",
                    self.stats["sitemap_batches_completed"],
                    self.stats["pages_scraped"],
                    self.stats["pages_failed"],
                    len(pending),
                )

    def _should_validate_source_html(self, page_url: str) -> bool:
        if not self.validate_source_html:
            return False
        normalized = _normalize_http_url(page_url)
        if not normalized:
            return False
        if _url_extension(normalized) in CRAWL_SKIP_EXTENSIONS:
            return False
        host = (urlparse(normalized).hostname or "").lower()
        return any(_host_matches_domain(host, allowed) for allowed in self.allowed_domains)

    async def _process_result(self, result: Any, *, mark_failure: bool = True) -> bool:
        page_url = _normalize_http_url(getattr(result, "url", None))
        if not page_url:
            return False
        self._last_result_seen_at = time.time()

        status_code = getattr(result, "status_code", None)
        rendered_html = getattr(result, "html", None) or ""
        html = rendered_html

        if not getattr(result, "success", False) or not html:
            fallback_html, fallback_status = await self._fetch_raw_source_page(page_url)
            if fallback_html:
                html = fallback_html
                rendered_html = fallback_html
                status_code = fallback_status or status_code or 200
                self.stats["http_fallback_pages"] += 1
                logger.info("Recovered crawler failure through HTTP fallback: url=%s", page_url)
            else:
                if mark_failure:
                    self._mark_url_skipped(
                        page_url,
                        status_code=status_code,
                        error_message=getattr(result, "error_message", ""),
                    )
                    self._flush_runtime_state()
                return False

        raw_source_html = ""
        raw_source_status: Optional[int] = None
        rendered_markdown = self._extract_crawl4ai_markdown(result)
        rendered_quality_report = _html_quality_report(rendered_html, page_url, markdown=rendered_markdown)
        should_fetch_source = (
            self.validate_source_html_mode == "always"
            or (
                self.validate_source_html_mode != "never"
                and bool(rendered_quality_report.get("reasons"))
            )
        )
        if should_fetch_source and self._should_validate_source_html(page_url):
            raw_source_html, raw_source_status = await self._fetch_raw_source_page(page_url)
            if raw_source_html:
                self.stats["source_html_validations"] += 1

        html, capture_source, quality_report = _select_preferred_page_capture(
            page_url,
            rendered_html,
            raw_source_html=raw_source_html,
            rendered_markdown=rendered_markdown,
        )
        selected_quality_reasons = (quality_report.get("selected") or {}).get("reasons") or []
        if selected_quality_reasons:
            self.stats["content_quality_warnings"] += 1
            logger.warning(
                "Crawler content quality warning: url=%s source=%s reasons=%s",
                page_url,
                capture_source,
                ",".join(selected_quality_reasons),
            )
        if capture_source == "raw_source":
            self.stats["source_html_replacements"] += 1
            status_code = raw_source_status or status_code or 200
            logger.info(
                "Selected raw page source over rendered capture: url=%s reason=%s",
                page_url,
                quality_report.get("selection_reason"),
            )
        selected_quality_report = quality_report.get("selected") or {}
        if not selected_quality_report.get("usable", True):
            skip_status = None
            try:
                if status_code is not None and int(status_code) >= 400:
                    skip_status = status_code
            except (TypeError, ValueError):
                skip_status = None
            self._mark_url_skipped(
                page_url,
                status_code=skip_status,
                error_message=f"content_quality:{','.join(selected_quality_report.get('reasons') or [])}",
                reason="SKIPPED_LOW_QUALITY",
            )
            self._flush_runtime_state()
            return False

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

        md_path: Optional[Path] = None
        md_text, markdown_source, markdown_quality_reason = self._extract_markdown(
            result,
            html=html,
            page_url=page_url,
            rendered_markdown=rendered_markdown,
            capture_source=capture_source,
        )
        if md_text:
            md_path = self.md_dir / f"{html_path.stem}.md"
            md_path.write_text(md_text, encoding="utf-8")
            self.url_to_md_mapping[page_url] = str(md_path)
            self.stats["markdown_written"] += 1
        elif markdown_quality_reason:
            self.stats["low_quality_markdown_suppressed"] += 1

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
                    raw_html = raw_source_html or await self._fetch_raw_source_html(page_url)
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

        try:
            self.page_links[page_url] = _extract_page_links(
                result,
                html,
                page_url,
                allowed_domains=self.allowed_domains,
            )
            page_media = self.page_media.get(page_url, [])
            depth = (self.crawl_state.get("depths") or {}).get(page_url)
            self.page_metadata[page_url] = _extract_page_metadata(
                html,
                page_url,
                status_code=status_code,
                html_path=str(html_path),
                markdown_path=str(md_path) if md_path else "",
                depth=depth,
                images_count=len(self.page_images.get(page_url, [])),
                videos_count=len(self.page_videos.get(page_url, [])),
                media_count=len(page_media),
                capture_source=capture_source,
                content_quality=quality_report,
                markdown_source=markdown_source,
                markdown_quality_reason=markdown_quality_reason,
            )
        except Exception as exc:
            logger.warning("Skipping page graph/metadata extraction for %s: %s", page_url, exc)

        downloadable_urls = self._extract_downloadable_urls(result, html=html, base_url=page_url)
        if downloadable_urls:
            await asyncio.gather(
                *(self._download_document(url) for url in downloadable_urls),
                return_exceptions=True,
            )

        self._flush_runtime_state()
        return True

    def _mark_url_skipped(
        self,
        page_url: str,
        *,
        status_code: Any = None,
        error_message: Any = "",
        reason: str = "SKIPPED_ERROR",
    ) -> None:
        normalized = _normalize_http_url(page_url)
        if not normalized:
            return
        existing = str(self.url_mapping.get(normalized) or "")
        if existing.startswith("SKIPPED"):
            return
        if status_code:
            skip_reason = f"SKIPPED_HTTP_{status_code}"
        else:
            skip_reason = reason
        compact_error = _compact_failure_reason(error_message)
        if compact_error and compact_error != skip_reason:
            skip_reason = f"{skip_reason}:{compact_error}"
        self.url_mapping[normalized] = skip_reason
        self.stats["pages_failed"] += 1
        self.stats["skipped_urls"] += 1

    async def _recover_url_with_http_retry(self, page_url: str) -> bool:
        normalized = _normalize_http_url(page_url)
        if not normalized:
            return False
        attempts = max(1, self.sitemap_failed_url_retry_attempts)
        for attempt in range(1, attempts + 1):
            fallback_html, fallback_status = await self._fetch_raw_source_page(normalized)
            if fallback_html:
                self.stats["http_fallback_retries"] += 1
                self.stats["http_fallback_pages"] += 1
                result = SimpleNamespace(
                    url=normalized,
                    html=fallback_html,
                    success=True,
                    status_code=fallback_status or 200,
                    links={"internal": [], "external": []},
                    markdown=None,
                )
                return await self._process_result(result, mark_failure=True)
            if attempt < attempts and self.sitemap_failed_retry_backoff:
                await asyncio.sleep(self.sitemap_failed_retry_backoff)
        return False

    async def _watch_crawl_health(self, crawl_task: "asyncio.Task[Any]") -> None:
        while not crawl_task.done():
            await asyncio.sleep(5)
            if crawl_task.done():
                return

            pending = len((self.crawl_state or {}).get("pending", []))
            visited = len((self.crawl_state or {}).get("visited", []))
            idle_seconds = max(0.0, time.time() - float(self._last_crawl_state_update_at or time.time()))

            if pending == 0 and visited > 0 and not self._has_crawl_output() and idle_seconds >= self.frontier_empty_timeout:
                self._crawl_watchdog_error = (
                    f"Crawler exhausted its frontier without producing output "
                    f"(visited={visited}, pending=0, idle={int(idle_seconds)}s)."
                )
                crawl_task.cancel()
                return

            if visited > 0 and not self._has_crawl_output() and idle_seconds >= self.crawl_stall_timeout:
                self._crawl_watchdog_error = (
                    f"Crawler stalled without producing output "
                    f"(visited={visited}, pending={pending}, idle={int(idle_seconds)}s)."
                )
                crawl_task.cancel()
                return

    def _extract_crawl4ai_markdown(self, result: Any) -> str:
        markdown = getattr(result, "markdown", None)
        if markdown is None:
            return ""
        fit_markdown = getattr(markdown, "fit_markdown", "") or ""
        raw_markdown = getattr(markdown, "raw_markdown", "") or ""
        return fit_markdown.strip() or raw_markdown.strip()

    def _extract_markdown(
        self,
        result: Any,
        *,
        html: str,
        page_url: str,
        rendered_markdown: str = "",
        capture_source: str = "rendered",
    ) -> Tuple[str, str, str]:
        crawl_markdown = rendered_markdown if rendered_markdown else self._extract_crawl4ai_markdown(result)
        crawl_reason = _markdown_quality_reason(crawl_markdown, page_url, html=html) if crawl_markdown else "empty_markdown"
        if crawl_markdown and capture_source == "rendered" and not crawl_reason:
            return crawl_markdown, "crawl4ai", ""

        generated_markdown = _html_to_markdown(html, page_url)
        generated_reason = (
            _markdown_quality_reason(generated_markdown, page_url, html=html)
            if generated_markdown
            else "empty_markdown"
        )
        if generated_markdown and not generated_reason:
            source = "source_html" if capture_source == "raw_source" else "html_fallback"
            return generated_markdown, source, ""

        if crawl_markdown and not crawl_reason:
            return crawl_markdown, "crawl4ai", ""

        return "", "", generated_reason or crawl_reason or "low_quality_markdown"

    def _extract_downloadable_urls(self, result: Any, html: str, base_url: str) -> List[str]:
        candidates: list[str] = []

        links = getattr(result, "links", None) or {}
        for group in ("internal", "external"):
            for item in links.get(group, []) or []:
                href = item.get("href") if isinstance(item, dict) else None
                normalized = _normalize_http_url(href, base_url=base_url)
                if (
                    normalized
                    and _url_extension(normalized) in DOWNLOADABLE_EXTENSIONS
                    and self._url_allowed_for_fetch(normalized)
                ):
                    candidates.append(normalized)

        if html:
            soup = BeautifulSoup(html, "html.parser")
            for anchor in soup.find_all("a", href=True):
                normalized = _normalize_http_url(anchor.get("href"), base_url=base_url)
                if (
                    normalized
                    and _url_extension(normalized) in DOWNLOADABLE_EXTENSIONS
                    and self._url_allowed_for_fetch(normalized)
                ):
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

    async def _fetch_raw_source_page(self, page_url: str) -> Tuple[str, Optional[int]]:
        if not self._session:
            return "", None
        normalized_page = _normalize_http_url(page_url)
        if not normalized_page or not self._url_allowed_for_fetch(normalized_page):
            return "", None
        last_status: Optional[int] = None
        candidates = [
            candidate
            for candidate in (_raw_source_candidate_urls(normalized_page) or [normalized_page])
            if self._url_allowed_for_fetch(candidate)
        ]
        for attempt in range(1, self.raw_source_retry_attempts + 1):
            for candidate in candidates:
                try:
                    async with self._session.get(candidate, allow_redirects=True, proxy=self.proxy) as response:
                        last_status = response.status
                        final_url = _normalize_http_url(str(response.url))
                        if not final_url or not self._url_allowed_for_fetch(final_url):
                            logger.warning(
                                "Blocked raw page source redirect outside allowed egress policy: %s -> %s",
                                candidate,
                                final_url or response.url,
                            )
                            return "", last_status
                        if response.status >= 400:
                            logger.debug(
                                "Raw page source returned HTTP %s for %s candidate=%s attempt=%d/%d",
                                response.status,
                                page_url,
                                candidate,
                                attempt,
                                self.raw_source_retry_attempts,
                            )
                            continue
                        return await response.text(), response.status
                except Exception as exc:
                    logger.debug(
                        "Failed to fetch raw page source for %s candidate=%s attempt=%d/%d: %s",
                        page_url,
                        candidate,
                        attempt,
                        self.raw_source_retry_attempts,
                        exc,
                    )
            if attempt < self.raw_source_retry_attempts and self.raw_source_retry_backoff:
                await asyncio.sleep(self.raw_source_retry_backoff * attempt)
        return "", last_status

    async def _fetch_raw_source_html(self, page_url: str) -> str:
        html, _status = await self._fetch_raw_source_page(page_url)
        return html

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
            enforce_allowed_domain=True,
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
        enforce_allowed_domain: bool = False,
    ) -> Optional[Path]:
        if not self._session:
            return None

        normalized = _normalize_http_url(url)
        if not normalized:
            return None
        if enforce_allowed_domain and not self._url_allowed_for_fetch(normalized):
            if status_on_failure:
                self.url_mapping[normalized] = "SKIPPED_EGRESS_POLICY"
                self.stats["skipped_urls"] += 1
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
                        if enforce_allowed_domain:
                            final_url = _normalize_http_url(str(response.url))
                            if not final_url or not self._url_allowed_for_fetch(final_url):
                                if status_on_failure:
                                    self.url_mapping[normalized] = "SKIPPED_EGRESS_POLICY"
                                    self.stats["skipped_urls"] += 1
                                return None
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
        previous_visited = len((self.crawl_state or {}).get("visited", []))
        previous_pending = len((self.crawl_state or {}).get("pending", []))
        self.crawl_state = _trim_crawl_state_to_budget(state, self.max_pages)
        self._last_crawl_state_update_at = time.time()
        current_visited = len(self.crawl_state.get("visited", []))
        current_pending = len(self.crawl_state.get("pending", []))
        if current_visited != previous_visited or current_pending != previous_pending:
            logger.debug(
                "Normalized crawl state to budget: visited=%d pending=%d max_pages=%d",
                current_visited,
                current_pending,
                self.max_pages,
            )
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
            "page_metadata": self.page_metadata,
            "page_links": self.page_links,
            "downloaded_images": self.downloaded_images,
            "recoverable_skip_retries": self.recoverable_skip_retries,
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
        atomic_write_json(self.page_metadata_file, self.page_metadata)
        atomic_write_json(
            self.page_link_graph_file,
            _build_page_link_graph_payload(
                page_metadata=self.page_metadata,
                page_links=self.page_links,
            ),
        )
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
            "page_metadata_records": len(self.page_metadata),
            "page_link_edges": sum(len(links) for links in self.page_links.values()),
        }
