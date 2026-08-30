"""Acquire and validate visual assets before chunking/index formatting.

The crawler intentionally records image references without necessarily caching
their bytes.  This stage turns those references into immutable, validated local
assets and materializes figure crops from Docling layout boxes.  It is kept
separate from crawling so completed raw crawls remain immutable and media
policy can evolve independently.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import math
import os
import re
import socket
import ssl
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp
import certifi
from PIL import Image, ImageStat, UnidentifiedImageError

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe
from pipeline.core.media import build_media_manifest, normalize_media_item
from pipeline.core.registry import register_stage


logger = logging.getLogger(__name__)

_IMAGE_FORMAT_EXTENSIONS = {
    "AVIF": ".avif",
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "TIFF": ".tiff",
    "WEBP": ".webp",
}
_DECORATIVE_PATH_RE = re.compile(
    r"(?:^|[/_.-])(?:favicon|icon|icons|logo|logos|sprite|spinner|loader|loading|"
    r"chevron|caret|arrow|close|hamburger|search|social|facebook|instagram|linkedin|"
    r"youtube|twitter|x-logo|whatsapp)(?:[/_.-]|$)",
    re.IGNORECASE,
)
_WINDOW_STRUCTURED_RE = re.compile(r"\.docling\.pages_\d{4}_\d{4}\.json$", re.IGNORECASE)
_DOC_REF_RE = re.compile(r"^#/texts/(\d+)$")


def _clean_text(value: Any, *, max_chars: int = 1000) -> str:
    text = " ".join(str(value or "").split()).strip()
    return text[:max(0, int(max_chars))]


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _perceptual_dhash(image: Image.Image) -> str:
    gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(gray.get_flattened_data())
    bits = 0
    for row in range(8):
        for column in range(8):
            bits = (bits << 1) | int(pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return f"{bits:016x}"


def _hamming_distance(left: str, right: str) -> int:
    try:
        return (int(left, 16) ^ int(right, 16)).bit_count()
    except (TypeError, ValueError):
        return 64


def _image_inspection(
    payload: bytes,
    *,
    minimum_width: int,
    minimum_height: int,
    minimum_area: int,
    maximum_pixels: int,
    maximum_aspect_ratio: float,
    minimum_stddev: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        with Image.open(BytesIO(payload)) as source:
            source.load()
            width, height = source.size
            image_format = str(source.format or "").upper()
            if image_format not in _IMAGE_FORMAT_EXTENSIONS:
                return None, "unsupported_image_format"
            if width <= 0 or height <= 0 or width * height > maximum_pixels:
                return None, "invalid_image_dimensions"
            if width < minimum_width or height < minimum_height or width * height < minimum_area:
                return None, "too_small"
            aspect_ratio = max(width / max(height, 1), height / max(width, 1))
            if aspect_ratio > maximum_aspect_ratio:
                return None, "extreme_aspect_ratio"
            sample = source.convert("L")
            sample.thumbnail((256, 256), Image.Resampling.LANCZOS)
            visual_stddev = float(ImageStat.Stat(sample).stddev[0])
            if visual_stddev < minimum_stddev:
                return None, "near_blank"
            return {
                "width": int(width),
                "height": int(height),
                "mime_type": Image.MIME.get(image_format, f"image/{image_format.lower()}"),
                "extension": _IMAGE_FORMAT_EXTENSIONS[image_format],
                "content_hash": _sha256_bytes(payload),
                "perceptual_hash": _perceptual_dhash(source),
                "visual_stddev": round(visual_stddev, 4),
                "file_size_bytes": len(payload),
            }, ""
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return None, "decompression_bomb"
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "invalid_image_payload"


def _normalize_https_url(value: Any) -> str:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return ""
    host = (parsed.hostname or "").lower().strip(".")
    if parsed.scheme.lower() != "https" or not host or parsed.username or parsed.password:
        return ""
    try:
        port = parsed.port
    except ValueError:
        return ""
    if port not in (None, 443):
        return ""
    netloc = host if port is None else f"{host}:{port}"
    return urlunsplit(("https", netloc, parsed.path or "/", parsed.query, ""))


def _url_policy_reason(url: str, allowed_hosts: set[str]) -> str:
    normalized = _normalize_https_url(url)
    if not normalized:
        return "invalid_or_non_https_url"
    host = (urlsplit(normalized).hostname or "").lower()
    if host not in allowed_hosts:
        return "host_not_allowed"
    if _DECORATIVE_PATH_RE.search(urlsplit(normalized).path.lower()):
        return "decorative_filename"
    return ""


async def _host_resolves_publicly(host: str, cache: MutableMapping[str, bool]) -> bool:
    if host in cache:
        return cache[host]
    try:
        addresses = await asyncio.get_running_loop().getaddrinfo(
            host,
            443,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
        )
        ips = {str(address[4][0]).split("%", 1)[0] for address in addresses if address[4]}
        allowed = bool(ips) and all(ipaddress.ip_address(value).is_global for value in ips)
    except (OSError, ValueError):
        allowed = False
    cache[host] = allowed
    return allowed


def _peer_is_public(response: aiohttp.ClientResponse) -> bool:
    connection = response.connection
    transport = connection.transport if connection is not None else None
    peer = transport.get_extra_info("peername") if transport is not None else None
    if not peer:
        # aiohttp may release a keep-alive connection before this property is
        # inspected. The request host was already resolved above and every
        # returned address was required to be globally routable; absence of
        # optional peer metadata is therefore not evidence of a private hop.
        return True
    try:
        return ipaddress.ip_address(str(peer[0]).split("%", 1)[0]).is_global
    except ValueError:
        return False


async def _read_bounded(response: aiohttp.ClientResponse, maximum_bytes: int) -> bytes:
    declared = response.content_length
    if declared is not None and declared > maximum_bytes:
        raise ValueError("image_too_large")
    chunks: List[bytes] = []
    total = 0
    async for chunk in response.content.iter_chunked(64 * 1024):
        total += len(chunk)
        if total > maximum_bytes:
            raise ValueError("image_too_large")
        chunks.append(chunk)
    return b"".join(chunks)


async def _download_image(
    session: aiohttp.ClientSession,
    *,
    url: str,
    output_dir: Path,
    allowed_hosts: set[str],
    dns_cache: MutableMapping[str, bool],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized = _normalize_https_url(url)
    policy_reason = _url_policy_reason(normalized, allowed_hosts)
    if policy_reason:
        return {"url": url, "status": "rejected", "reason": policy_reason}

    attempts = max(1, int(config.get("download_retry_attempts", 3)))
    redirect_limit = max(0, int(config.get("redirect_limit", 5)))
    maximum_bytes = max(1024, int(float(config.get("max_image_size_mb", 10)) * 1024 * 1024))
    retryable_statuses = {408, 425, 429, 500, 502, 503, 504}
    last_reason = "download_failed"

    for attempt in range(1, attempts + 1):
        current = normalized
        try:
            for redirect_index in range(redirect_limit + 1):
                host = (urlsplit(current).hostname or "").lower()
                if _url_policy_reason(current, allowed_hosts):
                    return {"url": url, "status": "rejected", "reason": "redirect_policy_block"}
                if not await _host_resolves_publicly(host, dns_cache):
                    return {"url": url, "status": "rejected", "reason": "non_public_destination"}
                async with session.get(current, allow_redirects=False) as response:
                    if not _peer_is_public(response):
                        return {"url": url, "status": "rejected", "reason": "non_public_peer"}
                    if response.status in {301, 302, 303, 307, 308}:
                        location = response.headers.get("Location")
                        if not location or redirect_index >= redirect_limit:
                            return {"url": url, "status": "failed", "reason": "redirect_limit"}
                        redirected = _normalize_https_url(urljoin(current, location))
                        if not redirected:
                            return {"url": url, "status": "rejected", "reason": "redirect_policy_block"}
                        current = redirected
                        continue
                    if response.status >= 400:
                        last_reason = f"http_{response.status}"
                        if response.status not in retryable_statuses:
                            return {"url": url, "status": "failed", "reason": last_reason}
                        break
                    content_type = str(response.headers.get("Content-Type") or "").lower()
                    if content_type.startswith("image/svg"):
                        return {"url": url, "status": "rejected", "reason": "unsupported_image_format"}
                    payload = await _read_bounded(response, maximum_bytes)
                    inspection, reason = _image_inspection(
                        payload,
                        minimum_width=max(1, int(config.get("min_web_width", 120))),
                        minimum_height=max(1, int(config.get("min_web_height", 90))),
                        minimum_area=max(1, int(config.get("min_web_area", 30000))),
                        maximum_pixels=max(1, int(config.get("max_image_pixels", 50_000_000))),
                        maximum_aspect_ratio=max(1.0, float(config.get("max_aspect_ratio", 12.0))),
                        minimum_stddev=max(0.0, float(config.get("min_visual_stddev", 2.0))),
                    )
                    if inspection is None:
                        return {"url": url, "status": "rejected", "reason": reason}
                    filename = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24] + inspection["extension"]
                    path = output_dir / filename
                    temporary = path.with_suffix(path.suffix + ".part")
                    temporary.write_bytes(payload)
                    os.replace(temporary, path)
                    return {
                        "url": url,
                        "final_url": current,
                        "status": "accepted",
                        "reason": "",
                        "local_path": str(path.resolve()),
                        "asset_uri": path.resolve().as_uri(),
                        **inspection,
                    }
        except ValueError as exc:
            last_reason = str(exc) or "invalid_response"
            if last_reason == "image_too_large":
                return {"url": url, "status": "rejected", "reason": last_reason}
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
            last_reason = f"network_{type(exc).__name__.lower()}"
        if attempt < attempts:
            await asyncio.sleep(min(8.0, float(config.get("retry_backoff_sec", 1.0)) * attempt))
    return {"url": url, "status": "failed", "reason": last_reason}


async def _download_image_bounded(
    session: aiohttp.ClientSession,
    *,
    url: str,
    output_dir: Path,
    allowed_hosts: set[str],
    dns_cache: MutableMapping[str, bool],
    config: Mapping[str, Any],
    global_semaphore: asyncio.Semaphore,
    host_semaphores: MutableMapping[str, asyncio.Semaphore],
    per_host_concurrency: int,
) -> Dict[str, Any]:
    """Start the request timeout only after bounded queue admission.

    aiohttp's total timeout includes time spent waiting for a connector slot.
    Creating thousands of unbounded ``session.get`` calls therefore makes the
    queued requests expire before they ever reach the server.  The explicit
    gates keep queue wait outside the request timeout and retain separate
    global/per-host limits.
    """

    host = (urlsplit(_normalize_https_url(url)).hostname or "").lower()
    host_semaphore = host_semaphores.get(host)
    if host_semaphore is None:
        host_semaphore = asyncio.Semaphore(max(1, int(per_host_concurrency)))
        host_semaphores[host] = host_semaphore
    async with global_semaphore:
        async with host_semaphore:
            return await _download_image(
                session,
                url=url,
                output_dir=output_dir,
                allowed_hosts=allowed_hosts,
                dns_cache=dns_cache,
                config=config,
            )


def _reuse_download_result(result: Mapping[str, Any]) -> bool:
    return (
        result.get("status") == "accepted"
        and bool(result.get("local_path"))
        and Path(str(result["local_path"])).is_file()
    )


def _canonicalize_exact_web_duplicates(results: MutableMapping[str, Dict[str, Any]]) -> int:
    canonical_by_hash: Dict[str, Tuple[str, str]] = {}
    duplicates = 0
    for url in sorted(results):
        result = results[url]
        if result.get("status") != "accepted" or not result.get("content_hash"):
            continue
        digest = str(result["content_hash"])
        canonical = canonical_by_hash.get(digest)
        if canonical is None:
            canonical_by_hash[digest] = (url, str(result.get("local_path") or ""))
            continue
        canonical_url, canonical_path = canonical
        duplicate_path = Path(str(result.get("local_path") or ""))
        if duplicate_path.is_file() and canonical_path and duplicate_path.resolve() != Path(canonical_path).resolve():
            duplicate_path.unlink(missing_ok=True)
        result["local_path"] = canonical_path
        result["asset_uri"] = Path(canonical_path).resolve().as_uri() if canonical_path else ""
        result["duplicate_of"] = canonical_url
        duplicates += 1
    return duplicates


def _enrich_page_media(
    raw_page_media: Mapping[str, Any],
    download_results: Mapping[str, Mapping[str, Any]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    enriched: Dict[str, List[Dict[str, Any]]] = {}
    page_images: Dict[str, List[Dict[str, Any]]] = {}
    all_images: List[Dict[str, Any]] = []
    for source_url, raw_items in raw_page_media.items():
        if not isinstance(raw_items, list):
            continue
        items: List[Dict[str, Any]] = []
        images: List[Dict[str, Any]] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = normalize_media_item(raw_item)
            if item.get("type") != "image":
                items.append(item)
                continue
            result = download_results.get(str(item.get("url") or ""), {})
            if result.get("status") != "accepted":
                continue
            updated = {
                **item,
                **{
                    key: result.get(key)
                    for key in (
                        "asset_uri",
                        "content_hash",
                        "duplicate_of",
                        "file_size_bytes",
                        "final_url",
                        "height",
                        "local_path",
                        "mime_type",
                        "perceptual_hash",
                        "visual_stddev",
                        "width",
                    )
                    if result.get(key) not in (None, "")
                },
                "id": _stable_id("web-image", source_url, item.get("url")),
                "source_url": str(source_url),
                "provider": item.get("provider") or "web_download",
                "download_status": "accepted",
            }
            items.append(updated)
            images.append(updated)
            all_images.append(updated)
        if items:
            enriched[str(source_url)] = items
        if images:
            page_images[str(source_url)] = images
    return enriched, page_images, all_images


def _reverse_download_mapping(mapping_file: str | Path | None) -> Tuple[Dict[str, str], Dict[str, str]]:
    payload = load_json_safe(mapping_file, {}) if mapping_file else {}
    url_by_path: Dict[str, str] = {}
    path_by_name: Dict[str, str] = {}
    if not isinstance(payload, dict):
        return url_by_path, path_by_name
    for url, raw_path in payload.items():
        path = Path(str(raw_path or ""))
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        url_by_path[resolved] = str(url)
        path_by_name.setdefault(path.name, resolved)
    return url_by_path, path_by_name


def _resolve_text_reference(payload: Mapping[str, Any], reference: Any) -> str:
    ref = str(reference.get("$ref") if isinstance(reference, dict) else reference or "")
    match = _DOC_REF_RE.match(ref)
    if not match:
        return ""
    texts = payload.get("texts") or []
    index = int(match.group(1))
    if not isinstance(texts, list) or index >= len(texts) or not isinstance(texts[index], dict):
        return ""
    return _clean_text(texts[index].get("text"), max_chars=500)


def _picture_caption(payload: Mapping[str, Any], picture: Mapping[str, Any]) -> str:
    values = [_resolve_text_reference(payload, value) for value in picture.get("captions") or []]
    return _clean_text(" ".join(value for value in values if value), max_chars=500)


def _bbox_center(bbox: Mapping[str, Any]) -> Tuple[float, float]:
    left, right = float(bbox.get("l") or 0), float(bbox.get("r") or 0)
    top, bottom = float(bbox.get("t") or 0), float(bbox.get("b") or 0)
    return ((left + right) / 2.0, (top + bottom) / 2.0)


def _picture_context(
    payload: Mapping[str, Any],
    *,
    page_number: int,
    bbox: Mapping[str, Any],
    caption: str,
) -> str:
    target_x, target_y = _bbox_center(bbox)
    nearby: List[Tuple[float, str]] = []
    for text_item in payload.get("texts") or []:
        if not isinstance(text_item, dict):
            continue
        text = _clean_text(text_item.get("text"), max_chars=400)
        if not text or text == caption:
            continue
        for provenance in text_item.get("prov") or []:
            if not isinstance(provenance, dict) or int(provenance.get("page_no") or 0) != page_number:
                continue
            text_bbox = provenance.get("bbox") or {}
            x, y = _bbox_center(text_bbox)
            nearby.append((math.hypot(x - target_x, y - target_y), text))
            break
    selected: List[str] = []
    for _distance, text in sorted(nearby, key=lambda pair: pair[0]):
        if text not in selected:
            selected.append(text)
        if len(selected) >= 3:
            break
    return _clean_text(" ".join(selected), max_chars=800)


def _docling_box_to_fitz(
    bbox: Mapping[str, Any],
    *,
    page_width: float,
    page_height: float,
    padding_points: float,
) -> Tuple[float, float, float, float]:
    left = float(bbox.get("l") or 0)
    right = float(bbox.get("r") or 0)
    top = float(bbox.get("t") or 0)
    bottom = float(bbox.get("b") or 0)
    if str(bbox.get("coord_origin") or "BOTTOMLEFT").upper() == "BOTTOMLEFT":
        top, bottom = page_height - top, page_height - bottom
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    return (
        max(0.0, left - padding_points),
        max(0.0, top - padding_points),
        min(page_width, right + padding_points),
        min(page_height, bottom + padding_points),
    )


def _primary_markdown_by_source(ctx: StageContext) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for artifact in ctx.find_artifacts(artifact_type="markdown"):
        source_file = str((artifact.metadata or {}).get("source_file") or "")
        if source_file and artifact.local_path and Path(artifact.local_path).is_file():
            mapping[str(Path(source_file).resolve())] = str(Path(artifact.local_path).resolve())
    return mapping


def _layout_sources(ctx: StageContext) -> List[Dict[str, str]]:
    url_by_path, path_by_name = _reverse_download_mapping(ctx.previous_outputs.get("mapping_file"))
    markdown_by_source = _primary_markdown_by_source(ctx)
    records: List[Dict[str, str]] = []
    seen_structured: set[str] = set()

    for artifact in ctx.find_artifacts(artifact_type="structured_document"):
        if not artifact.local_path or not Path(artifact.local_path).is_file():
            continue
        metadata = dict(artifact.metadata or {})
        source_file = str(metadata.get("source_file") or "")
        if not source_file or not Path(source_file).is_file():
            continue
        structured_path = str(Path(artifact.local_path).resolve())
        seen_structured.add(structured_path)
        resolved_source = str(Path(source_file).resolve())
        records.append(
            {
                "structured_path": structured_path,
                "source_file": resolved_source,
                "source_markdown_path": str(
                    metadata.get("source_markdown_path")
                    or markdown_by_source.get(resolved_source)
                    or ""
                ),
                "source_url": str(metadata.get("source_url") or url_by_path.get(resolved_source) or ""),
            }
        )

    structured_dir = Path(str(ctx.previous_outputs.get("structured_documents_dir") or ""))
    fallback_candidates = list(structured_dir.glob("*.docling.json")) if structured_dir.is_dir() else []
    if bool((ctx.formatter_config.get("media_enrichment") or {}).get("include_unused_fallback_layouts", True)):
        quarantine_dir = Path(str(ctx.previous_outputs.get("quarantine_dir") or ""))
        unused_dir = quarantine_dir / "structured_documents" / "unused_after_fallback"
        if unused_dir.is_dir():
            fallback_candidates.extend(unused_dir.glob("*.docling.json"))

    for path in sorted(fallback_candidates):
        resolved_structured = str(path.resolve())
        if resolved_structured in seen_structured or _WINDOW_STRUCTURED_RE.search(path.name):
            continue
        payload = load_json_safe(path, {}) or {}
        filename = str((payload.get("origin") or {}).get("filename") or "") if isinstance(payload, dict) else ""
        source_file = path_by_name.get(filename, "")
        if not source_file:
            continue
        records.append(
            {
                "structured_path": resolved_structured,
                "source_file": source_file,
                "source_markdown_path": markdown_by_source.get(source_file, ""),
                "source_url": url_by_path.get(source_file, ""),
            }
        )
        seen_structured.add(resolved_structured)
    return records


def _extract_pdf_figures(
    ctx: StageContext,
    *,
    output_dir: Path,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - validated in deployed requirements
        raise RuntimeError("PyMuPDF is required for PDF figure extraction") from exc

    metrics: Counter[str] = Counter()
    extracted: List[Dict[str, Any]] = []
    scale = max(1.0, float(config.get("pdf_crop_scale", 2.0)))
    padding = max(0.0, float(config.get("pdf_crop_padding_points", 3.0)))
    minimum_area_ratio = max(0.0, float(config.get("min_pdf_bbox_area_ratio", 0.003)))
    maximum_area_ratio = min(1.0, float(config.get("max_pdf_bbox_area_ratio", 0.88)))
    max_per_page = max(1, int(config.get("max_pdf_crops_per_page", 8)))
    near_duplicate_distance = max(0, int(config.get("pdf_perceptual_duplicate_distance", 2)))

    for source in _layout_sources(ctx):
        structured_path = Path(source["structured_path"])
        payload = load_json_safe(structured_path, {}) or {}
        if not isinstance(payload, dict):
            metrics["layout_invalid"] += 1
            continue
        pictures = payload.get("pictures") or []
        if not isinstance(pictures, list) or not pictures:
            continue
        metrics["documents_with_layout_pictures"] += 1
        candidates_by_page: Dict[int, List[Tuple[float, Mapping[str, Any], Mapping[str, Any]]]] = defaultdict(list)
        pages = payload.get("pages") if isinstance(payload.get("pages"), dict) else {}
        for picture in pictures:
            if not isinstance(picture, dict):
                continue
            for provenance in picture.get("prov") or []:
                if not isinstance(provenance, dict):
                    continue
                page_number = int(provenance.get("page_no") or 0)
                bbox = provenance.get("bbox") or {}
                page_info = pages.get(str(page_number), {}) if isinstance(pages, dict) else {}
                size = page_info.get("size") or {}
                page_width = float(size.get("width") or 0)
                page_height = float(size.get("height") or 0)
                width = abs(float(bbox.get("r") or 0) - float(bbox.get("l") or 0))
                height = abs(float(bbox.get("t") or 0) - float(bbox.get("b") or 0))
                page_area = max(1.0, page_width * page_height)
                area_ratio = width * height / page_area
                metrics["layout_boxes_seen"] += 1
                if page_number <= 0 or not (minimum_area_ratio <= area_ratio <= maximum_area_ratio):
                    metrics["layout_boxes_size_filtered"] += 1
                    continue
                candidates_by_page[page_number].append((area_ratio, picture, provenance))

        source_file = Path(source["source_file"])
        seen_content_hashes: set[str] = set()
        seen_perceptual: List[Tuple[str, float]] = []
        try:
            with fitz.open(str(source_file)) as document:
                for page_number in sorted(candidates_by_page):
                    candidates = sorted(candidates_by_page[page_number], key=lambda row: row[0], reverse=True)
                    if len(candidates) > max_per_page:
                        metrics["layout_boxes_page_cap_filtered"] += len(candidates) - max_per_page
                    for area_ratio, picture, provenance in candidates[:max_per_page]:
                        if page_number > len(document):
                            metrics["layout_page_missing"] += 1
                            continue
                        page = document.load_page(page_number - 1)
                        box = _docling_box_to_fitz(
                            provenance.get("bbox") or {},
                            page_width=float(page.rect.width),
                            page_height=float(page.rect.height),
                            padding_points=padding,
                        )
                        if box[2] <= box[0] or box[3] <= box[1]:
                            metrics["layout_box_invalid"] += 1
                            continue
                        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), clip=fitz.Rect(*box), alpha=False)
                        image_bytes = pixmap.tobytes("png")
                        inspection, reason = _image_inspection(
                            image_bytes,
                            minimum_width=max(1, int(config.get("min_pdf_crop_width", 96))),
                            minimum_height=max(1, int(config.get("min_pdf_crop_height", 72))),
                            minimum_area=max(1, int(config.get("min_pdf_crop_area", 12000))),
                            maximum_pixels=max(1, int(config.get("max_image_pixels", 50_000_000))),
                            maximum_aspect_ratio=max(1.0, float(config.get("max_pdf_crop_aspect_ratio", 12.0))),
                            minimum_stddev=max(0.0, float(config.get("min_visual_stddev", 2.0))),
                        )
                        if inspection is None:
                            metrics[f"crop_rejected_{reason}"] += 1
                            continue
                        content_hash = str(inspection["content_hash"])
                        aspect = inspection["width"] / max(inspection["height"], 1)
                        if content_hash in seen_content_hashes:
                            metrics["crop_exact_duplicates"] += 1
                            continue
                        perceptual_hash = str(inspection["perceptual_hash"])
                        if any(
                            _hamming_distance(perceptual_hash, existing_hash) <= near_duplicate_distance
                            and abs(math.log(max(aspect, 1e-6) / max(existing_aspect, 1e-6))) < 0.08
                            for existing_hash, existing_aspect in seen_perceptual
                        ):
                            metrics["crop_perceptual_duplicates"] += 1
                            continue
                        seen_content_hashes.add(content_hash)
                        seen_perceptual.append((perceptual_hash, aspect))

                        caption = _picture_caption(payload, picture)
                        context = _picture_context(
                            payload,
                            page_number=page_number,
                            bbox=provenance.get("bbox") or {},
                            caption=caption,
                        )
                        document_id = str(payload.get("name") or source_file.stem)
                        image_id = _stable_id(
                            "pdf-figure",
                            str(source_file.resolve()),
                            page_number,
                            *(round(float(value), 3) for value in box),
                        )
                        path = output_dir / f"{image_id}.png"
                        temporary = path.with_suffix(".png.part")
                        temporary.write_bytes(image_bytes)
                        os.replace(temporary, path)
                        title = caption or f"Visual from {document_id} page {page_number}"
                        extracted.append(
                            {
                                "type": "image",
                                "id": image_id,
                                "url": path.resolve().as_uri(),
                                "asset_uri": path.resolve().as_uri(),
                                "local_path": str(path.resolve()),
                                "alt": title,
                                "title": title,
                                "caption": caption,
                                "description": "",
                                "context": context,
                                "source_type": "pdf",
                                "source_backend": "docling_layout_crop",
                                "source_file": str(source_file.resolve()),
                                "source_url": source.get("source_url", ""),
                                "source_document_path": source.get("source_markdown_path", ""),
                                "md_path": source.get("source_markdown_path", ""),
                                "document_id": document_id,
                                "page_number": page_number,
                                "bbox": {
                                    "l": round(box[0], 4),
                                    "t": round(box[1], 4),
                                    "r": round(box[2], 4),
                                    "b": round(box[3], 4),
                                    "coord_origin": "TOPLEFT",
                                },
                                "bbox_area_ratio": round(area_ratio, 6),
                                "crop_source": "docling_layout",
                                "download_status": "generated",
                                **inspection,
                            }
                        )
                        metrics["pdf_crops_accepted"] += 1
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("Could not extract PDF figures from %s: %s", source_file, exc)
            metrics["pdf_documents_failed"] += 1
    return extracted, metrics


@register_stage
class MediaEnrichmentFormatter(FormatterStage):
    name = "media_enrichment"
    description = "Downloads validated webpage images and crops PDF figures from layout boxes."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        formatter = config.get("formatter") if isinstance(config.get("formatter"), dict) else {}
        media_config = formatter.get("media_enrichment") if isinstance(formatter, dict) else {}
        if not isinstance(media_config, dict):
            return ["formatter.media_enrichment must be a mapping"]
        if bool(media_config.get("download_web_images", True)) and not media_config.get("allowed_media_hosts"):
            return ["formatter.media_enrichment.allowed_media_hosts is required when web downloads are enabled"]
        errors: List[str] = []
        for field in ("download_concurrency", "max_image_size_mb", "min_web_width", "min_web_height"):
            try:
                if float(media_config.get(field, 1)) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"formatter.media_enrichment.{field} must be positive")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.formatter_config.get("media_enrichment") or {}
        if not isinstance(config, dict):
            return StageResult.failure("formatter.media_enrichment must be a mapping")

        raw_page_media_path = ctx.previous_outputs.get("page_media_file") or ctx.previous_outputs.get("page_images_file")
        raw_page_media = load_json_safe(raw_page_media_path, {}) if raw_page_media_path else {}
        if not isinstance(raw_page_media, dict):
            raw_page_media = {}

        web_images_dir = ensure_dir(ctx.output_dir("web_images"))
        pdf_images_dir = ensure_dir(ctx.output_dir("pdf_figure_crops"))
        download_manifest_path = ctx.stage_work_dir / "web_image_download_manifest.json"
        existing_manifest = load_json_safe(download_manifest_path, {}) or {}
        existing_results = existing_manifest.get("results") if isinstance(existing_manifest, dict) else {}
        download_results: Dict[str, Dict[str, Any]] = {
            str(url): dict(result)
            for url, result in (existing_results or {}).items()
            if isinstance(result, dict)
        }

        image_urls = sorted(
            {
                str(item.get("url") or "")
                for items in raw_page_media.values()
                if isinstance(items, list)
                for item in items
                if isinstance(item, dict) and str(item.get("type") or "image").lower() == "image" and item.get("url")
            }
        )
        allowed_hosts = {
            str(value or "").lower().strip(".")
            for value in config.get("allowed_media_hosts") or []
            if str(value or "").strip()
        }
        reasons: Counter[str] = Counter()
        to_download: List[str] = []
        for url in image_urls:
            if _reuse_download_result(download_results.get(url, {})):
                reasons["resumed"] += 1
                continue
            policy_reason = _url_policy_reason(url, allowed_hosts)
            if policy_reason:
                download_results[url] = {"url": url, "status": "rejected", "reason": policy_reason}
                reasons[policy_reason] += 1
                continue
            to_download.append(url)

        if bool(config.get("download_web_images", True)) and to_download:
            timeout = aiohttp.ClientTimeout(total=max(1.0, float(config.get("request_timeout_sec", 45.0))))
            download_concurrency = max(1, int(config.get("download_concurrency", 8)))
            per_host_concurrency = max(1, int(config.get("per_host_concurrency", 4)))
            connector = aiohttp.TCPConnector(
                limit=download_concurrency,
                limit_per_host=per_host_concurrency,
                ttl_dns_cache=60,
                ssl=ssl.create_default_context(cafile=certifi.where()),
            )
            headers = {
                "User-Agent": str(
                    config.get("user_agent")
                    or ctx.crawler_config.get("user_agent")
                    or "MBZUAIKnowledgeIndexer/1.0"
                ),
                "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.5",
                "X-Crawler-Name": "MBZUAIKnowledgeIndexer",
                "X-Crawler-Purpose": "search-index-reference",
            }
            dns_cache: Dict[str, bool] = {}
            async with aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers=headers,
                trust_env=False,
            ) as session:
                global_semaphore = asyncio.Semaphore(download_concurrency)
                host_semaphores: Dict[str, asyncio.Semaphore] = {}
                tasks = [
                    asyncio.create_task(
                        _download_image_bounded(
                            session,
                            url=url,
                            output_dir=web_images_dir,
                            allowed_hosts=allowed_hosts,
                            dns_cache=dns_cache,
                            config=config,
                            global_semaphore=global_semaphore,
                            host_semaphores=host_semaphores,
                            per_host_concurrency=per_host_concurrency,
                        )
                    )
                    for url in to_download
                ]
                flush_every = max(1, int(config.get("manifest_flush_every", 25)))
                for completed_count, task in enumerate(asyncio.as_completed(tasks), start=1):
                    result = await task
                    download_results[str(result.get("url") or "")] = result
                    reasons[str(result.get("reason") or result.get("status") or "unknown")] += 1
                    if completed_count % flush_every == 0:
                        atomic_write_json(
                            download_manifest_path,
                            {"version": 1, "results": download_results},
                        )

        web_exact_duplicates = _canonicalize_exact_web_duplicates(download_results)
        status_counts = Counter(str(result.get("status") or "unknown") for result in download_results.values())
        reason_counts = Counter(
            str(result.get("reason") or "accepted") for result in download_results.values()
        )
        atomic_write_json(
            download_manifest_path,
            {
                "version": 1,
                "source_page_media_file": str(raw_page_media_path or ""),
                "unique_urls": len(image_urls),
                "status_counts": dict(sorted(status_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "exact_content_duplicates": web_exact_duplicates,
                "results": download_results,
            },
        )

        enriched_page_media, enriched_page_images, web_items = _enrich_page_media(
            raw_page_media,
            download_results,
        )
        enriched_page_media_path = ctx.stage_work_dir / "enriched_page_media.json"
        enriched_page_images_path = ctx.stage_work_dir / "enriched_page_images.json"
        atomic_write_json(enriched_page_media_path, enriched_page_media)
        atomic_write_json(enriched_page_images_path, enriched_page_images)

        pdf_items: List[Dict[str, Any]] = []
        pdf_metrics: Counter[str] = Counter()
        if bool(config.get("extract_pdf_figures", True)):
            pdf_items, pdf_metrics = await asyncio.to_thread(
                _extract_pdf_figures,
                ctx,
                output_dir=pdf_images_dir,
                config=config,
            )
        document_manifest_path = ctx.stage_work_dir / "extracted_images_index.json"
        atomic_write_json(document_manifest_path, build_media_manifest(pdf_items, kind="document_media"))

        combined_manifest_path = ctx.stage_work_dir / "multimodal_media_manifest.json"
        atomic_write_json(
            combined_manifest_path,
            build_media_manifest([*web_items, *pdf_items], kind="multimodal_media"),
        )

        accepted_unique_web = status_counts.get("accepted", 0)
        attempted_network = accepted_unique_web + status_counts.get("failed", 0)
        failure_ratio = status_counts.get("failed", 0) / max(1, attempted_network)
        minimum_assets = max(0, int(config.get("minimum_accepted_assets", 1)))
        total_assets = accepted_unique_web + len(pdf_items)
        maximum_failure_ratio = max(0.0, float(config.get("maximum_download_failure_ratio", 0.25)))

        report = {
            "version": 1,
            "source_page_media_file": str(raw_page_media_path or ""),
            "web": {
                "references": sum(
                    1
                    for items in raw_page_media.values()
                    if isinstance(items, list)
                    for item in items
                    if isinstance(item, dict) and str(item.get("type") or "image").lower() == "image"
                ),
                "unique_urls": len(image_urls),
                "accepted_unique": accepted_unique_web,
                "accepted_page_references": len(web_items),
                "status_counts": dict(sorted(status_counts.items())),
                "reason_counts": dict(sorted(reason_counts.items())),
                "exact_content_duplicates": web_exact_duplicates,
                "network_failure_ratio": round(failure_ratio, 6),
            },
            "pdf": dict(sorted(pdf_metrics.items())),
            "total_multimodal_assets": total_assets,
            "gates": {
                "minimum_accepted_assets": minimum_assets,
                "maximum_download_failure_ratio": maximum_failure_ratio,
                "accepted_assets_passed": total_assets >= minimum_assets,
                "download_failure_ratio_passed": failure_ratio <= maximum_failure_ratio,
            },
        }
        report_path = ctx.stage_work_dir / "media_enrichment_report.json"
        atomic_write_json(report_path, report)

        if total_assets < minimum_assets:
            return StageResult.failure(
                f"Media enrichment produced {total_assets} assets; minimum is {minimum_assets}",
                outputs={
                    "page_media_file": str(enriched_page_media_path),
                    "page_images_file": str(enriched_page_images_path),
                    "extracted_images_index_file": str(document_manifest_path),
                    "media_manifest_file": str(combined_manifest_path),
                    "media_enrichment_report_file": str(report_path),
                },
                metrics={"accepted_assets": total_assets, "download_failure_ratio": failure_ratio},
            )
        if failure_ratio > maximum_failure_ratio:
            return StageResult.failure(
                f"Web image network failure ratio {failure_ratio:.3f} exceeds {maximum_failure_ratio:.3f}",
                outputs={
                    "page_media_file": str(enriched_page_media_path),
                    "page_images_file": str(enriched_page_images_path),
                    "extracted_images_index_file": str(document_manifest_path),
                    "media_manifest_file": str(combined_manifest_path),
                    "media_enrichment_report_file": str(report_path),
                },
                metrics={"accepted_assets": total_assets, "download_failure_ratio": failure_ratio},
            )

        artifacts = [
            ctx.make_artifact(
                combined_manifest_path,
                artifact_type="media_manifest",
                role="multimodal_media",
                metadata={"web_items": len(web_items), "pdf_items": len(pdf_items)},
            ),
            ctx.make_artifact(
                report_path,
                artifact_type="media_enrichment_report",
                role="quality_report",
                metadata={"accepted_assets": total_assets, "download_failure_ratio": failure_ratio},
            ),
        ]
        for item in pdf_items:
            artifacts.append(
                ctx.make_artifact(
                    item["local_path"],
                    artifact_type="extracted_image",
                    role="document_figure",
                    metadata=item,
                )
            )

        unique_web_by_path: Dict[str, Dict[str, Any]] = {}
        pages_by_path: Dict[str, set[str]] = defaultdict(set)
        for item in web_items:
            path = str(item.get("local_path") or "")
            if not path:
                continue
            unique_web_by_path.setdefault(path, item)
            if item.get("source_url"):
                pages_by_path[path].add(str(item["source_url"]))
        for path, item in unique_web_by_path.items():
            artifacts.append(
                ctx.make_artifact(
                    path,
                    artifact_type="web_image",
                    role="web_content_image",
                    metadata={
                        **item,
                        "source_page_urls": sorted(pages_by_path[path]),
                    },
                )
            )

        return StageResult.success(
            outputs={
                "page_media_file": str(enriched_page_media_path),
                "page_images_file": str(enriched_page_images_path),
                "web_images_dir": str(web_images_dir),
                "images_dir": str(pdf_images_dir),
                "extracted_images_index_file": str(document_manifest_path),
                "extracted_images_count": len(pdf_items),
                "media_manifest_file": str(combined_manifest_path),
                "media_enrichment_report_file": str(report_path),
                "web_image_download_manifest_file": str(download_manifest_path),
            },
            metrics={
                "web_image_references": report["web"]["references"],
                "web_image_unique_urls": len(image_urls),
                "web_images_accepted": accepted_unique_web,
                "web_image_page_references_accepted": len(web_items),
                "web_images_rejected": status_counts.get("rejected", 0),
                "web_images_failed": status_counts.get("failed", 0),
                "web_image_exact_duplicates": web_exact_duplicates,
                "download_failure_ratio": round(failure_ratio, 6),
                "pdf_images_extracted": len(pdf_items),
                **dict(pdf_metrics),
            },
            artifacts=artifacts,
        )
