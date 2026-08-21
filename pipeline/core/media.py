"""
Shared media helpers for ingestion, indexing, and retrieval contracts.

The pipeline keeps media as structured metadata so downstream retrieval and
response generation can attach relevant images/videos without depending on raw
HTML tags surviving every stage.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DIRECT_VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".ogg",
    ".mov",
    ".m4v",
    ".m3u8",
}


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text.strip()


def _clean_multiline_text(value: Any) -> str:
    """Normalize exact text without destroying line and table boundaries."""
    if value is None:
        return ""
    lines = [" ".join(line.split()).strip() for line in str(value).splitlines()]
    output: List[str] = []
    for line in lines:
        if line:
            output.append(line)
        elif output and output[-1] != "":
            output.append("")
    while output and not output[-1]:
        output.pop()
    return "\n".join(output)


def _clean_url(value: Any) -> str:
    text = _clean_text(value)
    return text


def _coerce_position(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = _clean_text(value).lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def _clean_text_list(value: Any, *, max_items: int = 24, max_chars: int = 160) -> List[str]:
    if not isinstance(value, (list, tuple, set)):
        return []
    output: List[str] = []
    for item in value:
        text = _clean_text(item)[:max_chars]
        if text and text not in output:
            output.append(text)
        if len(output) >= max_items:
            break
    return output


def _normalize_bbox(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    bbox: Dict[str, Any] = {}
    for key in ("l", "t", "r", "b"):
        coordinate = _coerce_float(value.get(key))
        if coordinate is not None:
            bbox[key] = coordinate
    origin = _clean_text(value.get("coord_origin")).upper()
    if origin in {"TOPLEFT", "BOTTOMLEFT"}:
        bbox["coord_origin"] = origin
    return bbox if all(key in bbox for key in ("l", "t", "r", "b")) else {}


def normalize_media_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a media item to the canonical schema."""
    media_type = _clean_text(item.get("type") or item.get("media_type") or "image").lower()
    if media_type not in {"image", "video"}:
        media_type = "image"

    normalized = {
        "type": media_type,
        "url": _clean_url(item.get("url")),
        "alt": _clean_text(item.get("alt")),
        "title": _clean_text(item.get("title")),
        "caption": _clean_text(item.get("caption")),
        "description": _clean_text(item.get("description")),
        "context": _clean_text(item.get("context")),
        # Page/section context is occurrence-specific.  It must remain
        # separate from authored context and from the content-hash-level
        # visual annotation because one image may be reused on several pages.
        "context_reference_id": _clean_text(item.get("context_reference_id")),
        "context_source": _clean_text(item.get("context_source")),
        "context_association": _clean_text(item.get("context_association")),
        "context_confidence": _coerce_float(item.get("context_confidence")),
        "context_sha256": _clean_text(item.get("context_sha256")),
        "context_source_path": _clean_text(item.get("context_source_path")),
        "section_id": _clean_text(item.get("section_id")),
        "section_path": _clean_text_list(item.get("section_path"), max_items=8, max_chars=300),
        "section_heading": _clean_text(item.get("section_heading")),
        "surrounding_text_before": _clean_text(item.get("surrounding_text_before")),
        "surrounding_text_after": _clean_text(item.get("surrounding_text_after")),
        "nearby_text": _clean_text(item.get("nearby_text")),
        "page_title": _clean_text(item.get("page_title")),
        "poster_url": _clean_url(item.get("poster_url")),
        "provider": _clean_text(item.get("provider")),
        "transcript": _clean_text(item.get("transcript")),
        "local_path": _clean_text(item.get("local_path")),
        "asset_uri": _clean_url(item.get("asset_uri")),
        "source_type": _clean_text(item.get("source_type")),
        "source_file": _clean_text(item.get("source_file")),
        "source_url": _clean_url(item.get("source_url")),
        "source_document_path": _clean_text(item.get("source_document_path")),
        "embed_type": _clean_text(item.get("embed_type")),
        "document_id": _clean_text(item.get("document_id")),
        "md_path": _clean_text(item.get("md_path")),
        "id": _clean_text(item.get("id")),
        "position": _coerce_position(item.get("position")),
        "page_number": _coerce_position(item.get("page_number")),
        "width": _coerce_position(item.get("width")),
        "height": _coerce_position(item.get("height")),
        "mime_type": _clean_text(item.get("mime_type")),
        "content_hash": _clean_text(item.get("content_hash")),
        "perceptual_hash": _clean_text(item.get("perceptual_hash")),
        "duplicate_of": _clean_url(item.get("duplicate_of")),
        "final_url": _clean_url(item.get("final_url")),
        "download_status": _clean_text(item.get("download_status")),
        "source_backend": _clean_text(item.get("source_backend")),
        "crop_source": _clean_text(item.get("crop_source")),
        # OCR is an exact-text evidence channel. Keep it distinct from the
        # generative ``visible_text`` field and retain meaningful line breaks.
        "ocr_text": _clean_multiline_text(item.get("ocr_text")),
        "ocr_status": _clean_text(item.get("ocr_status")),
        "ocr_provider": _clean_text(item.get("ocr_provider")),
        "ocr_provider_revision": _clean_text(item.get("ocr_provider_revision")),
        "ocr_model": _clean_text(item.get("ocr_model")),
        "ocr_model_revision": _clean_text(item.get("ocr_model_revision")),
        "ocr_mode": _clean_text(item.get("ocr_mode")),
        "ocr_prompt_revision": _clean_text(item.get("ocr_prompt_revision")),
        "ocr_input_hash": _clean_text(item.get("ocr_input_hash")),
        "ocr_raw_output_sha256": _clean_text(item.get("ocr_raw_output_sha256")),
        "ocr_latency_ms": _coerce_float(item.get("ocr_latency_ms")),
        "ocr_attempts": _coerce_position(item.get("ocr_attempts")),
        "ocr_quality_score": _coerce_float(item.get("ocr_quality_score")),
        "ocr_quality_flags": _clean_text_list(item.get("ocr_quality_flags"), max_items=16),
        "ocr_error": _clean_text(item.get("ocr_error")),
        "ocr_completed_at": _clean_text(item.get("ocr_completed_at")),
        # Model-authored semantics are separate from website-authored
        # alt/caption/context so retrieval can preserve provenance and avoid
        # presenting contextual hints as literal visual observations.
        "semantic_caption": _clean_text(item.get("semantic_caption")),
        "contextual_caption": _clean_text(item.get("contextual_caption")),
        "visual_description": _clean_text(item.get("visual_description")),
        "visible_text": _clean_text(item.get("visible_text")),
        "image_kind": _clean_text(item.get("image_kind")),
        "semantic_tags": _clean_text_list(item.get("semantic_tags")),
        "semantic_relevance": _clean_text(item.get("semantic_relevance")),
        "annotation_status": _clean_text(item.get("annotation_status")),
        "annotation_provider": _clean_text(item.get("annotation_provider")),
        "annotation_model": _clean_text(item.get("annotation_model")),
        "annotation_model_revision": _clean_text(item.get("annotation_model_revision")),
        "annotation_prompt_revision": _clean_text(item.get("annotation_prompt_revision")),
        "annotation_input_hash": _clean_text(item.get("annotation_input_hash")),
        "annotation_confidence": _coerce_float(item.get("annotation_confidence")),
        "annotation_error": _clean_text(item.get("annotation_error")),
        "contextual_caption_scope": _clean_text(item.get("contextual_caption_scope")),
        "contextual_caption_reference_id": _clean_text(
            item.get("contextual_caption_reference_id")
        ),
        "contains_text": _coerce_bool(item.get("contains_text")),
        "needs_ocr": _coerce_bool(item.get("needs_ocr")),
        "needs_review": _coerce_bool(item.get("needs_review")),
        "uncertain_details": _clean_text_list(item.get("uncertain_details"), max_items=12),
        "bbox": _normalize_bbox(item.get("bbox")),
        "bbox_area_ratio": _coerce_float(item.get("bbox_area_ratio")),
        "visual_stddev": _coerce_float(item.get("visual_stddev")),
        "file_size_bytes": _coerce_position(item.get("file_size_bytes")),
        "track_urls": [
            _clean_url(track_url)
            for track_url in (item.get("track_urls") or [])
            if _clean_url(track_url)
        ],
        "transcript_url": _clean_url(item.get("transcript_url")),
    }

    if media_type == "image" and not normalized["title"]:
        normalized["title"] = (
            normalized["alt"]
            or normalized["caption"]
            or normalized["description"]
            or normalized["context"]
        )
    if media_type == "video" and not normalized["title"]:
        normalized["title"] = (
            normalized["caption"]
            or normalized["alt"]
            or normalized["description"]
            or normalized["context"]
            or "Video"
        )
    if normalized["local_path"] and not normalized["asset_uri"]:
        try:
            normalized["asset_uri"] = Path(normalized["local_path"]).resolve().as_uri()
        except Exception:
            pass

    return normalized


def dedupe_media_items(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate media by type + URL while preserving order."""
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        normalized = normalize_media_item(item)
        url = normalized.get("url")
        if not url:
            continue
        key = (normalized.get("type"), url)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)

    return sorted(
        deduped,
        key=lambda item: (
            item.get("position") is None,
            item.get("position") if item.get("position") is not None else 10**9,
        ),
    )


def media_items_by_type(items: Iterable[Dict[str, Any]], media_type: str) -> List[Dict[str, Any]]:
    return [item for item in dedupe_media_items(items) if item.get("type") == media_type]


def compact_media_for_metadata(
    items: Iterable[Dict[str, Any]],
    *,
    max_items: int = 8,
    max_text_len: int = 280,
    include_local_path: bool = False,
) -> List[Dict[str, Any]]:
    """Trim media metadata to a Pinecone-friendly shape."""
    compacted: List[Dict[str, Any]] = []
    for item in dedupe_media_items(items)[: max(0, int(max_items))]:
        compact = {
            "type": item.get("type", ""),
            "url": item.get("url", ""),
            "alt": item.get("alt", "")[:max_text_len],
            "title": item.get("title", "")[:max_text_len],
            "caption": item.get("caption", "")[:max_text_len],
            "description": item.get("description", "")[: max_text_len * 2],
            "context": item.get("context", "")[:max_text_len],
            "context_reference_id": item.get("context_reference_id", "")[:120],
            "context_source": item.get("context_source", "")[:80],
            "context_association": item.get("context_association", "")[:80],
            "context_confidence": item.get("context_confidence"),
            "context_sha256": item.get("context_sha256", "")[:80],
            "section_id": item.get("section_id", "")[:120],
            "section_path": list(item.get("section_path") or [])[:8],
            "section_heading": item.get("section_heading", "")[:max_text_len],
            "surrounding_text_before": item.get("surrounding_text_before", "")[: max_text_len * 2],
            "surrounding_text_after": item.get("surrounding_text_after", "")[: max_text_len * 2],
            "nearby_text": item.get("nearby_text", "")[: max_text_len * 2],
            "page_title": item.get("page_title", "")[:max_text_len],
            "poster_url": item.get("poster_url", ""),
            "asset_uri": item.get("asset_uri", ""),
            "provider": item.get("provider", "")[:80],
            "transcript": item.get("transcript", "")[: max_text_len * 2],
            "source_type": item.get("source_type", "")[:40],
            "source_file": item.get("source_file", "")[: max_text_len * 2],
            "source_url": item.get("source_url", ""),
            "document_id": item.get("document_id", "")[:120],
            "page_number": item.get("page_number"),
            "embed_type": item.get("embed_type", "")[:40],
            "position": item.get("position"),
            "transcript_url": item.get("transcript_url", ""),
            "mime_type": item.get("mime_type", "")[:80],
            "content_hash": item.get("content_hash", "")[:80],
            "perceptual_hash": item.get("perceptual_hash", "")[:32],
            "crop_source": item.get("crop_source", "")[:80],
            "ocr_text": item.get("ocr_text", "")[: max_text_len * 3],
            "ocr_status": item.get("ocr_status", "")[:40],
            "ocr_provider": item.get("ocr_provider", "")[:80],
            "ocr_provider_revision": item.get("ocr_provider_revision", "")[:120],
            "ocr_model": item.get("ocr_model", "")[:120],
            "ocr_model_revision": item.get("ocr_model_revision", "")[:120],
            "ocr_mode": item.get("ocr_mode", "")[:40],
            "ocr_prompt_revision": item.get("ocr_prompt_revision", "")[:120],
            "ocr_input_hash": item.get("ocr_input_hash", "")[:80],
            "ocr_raw_output_sha256": item.get("ocr_raw_output_sha256", "")[:80],
            "ocr_latency_ms": item.get("ocr_latency_ms"),
            "ocr_attempts": item.get("ocr_attempts"),
            "ocr_quality_score": item.get("ocr_quality_score"),
            "ocr_quality_flags": list(item.get("ocr_quality_flags") or [])[:12],
            "ocr_error": item.get("ocr_error", "")[:max_text_len],
            "semantic_caption": item.get("semantic_caption", "")[:max_text_len],
            "contextual_caption": item.get("contextual_caption", "")[: max_text_len * 2],
            "visual_description": item.get("visual_description", "")[: max_text_len * 2],
            "visible_text": item.get("visible_text", "")[: max_text_len * 2],
            "image_kind": item.get("image_kind", "")[:60],
            "semantic_tags": list(item.get("semantic_tags") or [])[:16],
            "semantic_relevance": item.get("semantic_relevance", "")[:40],
            "annotation_status": item.get("annotation_status", "")[:40],
            "annotation_provider": item.get("annotation_provider", "")[:80],
            "annotation_model": item.get("annotation_model", "")[:120],
            "annotation_model_revision": item.get("annotation_model_revision", "")[:120],
            "annotation_prompt_revision": item.get("annotation_prompt_revision", "")[:120],
            "annotation_input_hash": item.get("annotation_input_hash", "")[:80],
            "annotation_confidence": item.get("annotation_confidence"),
            "contextual_caption_scope": item.get("contextual_caption_scope", "")[:40],
            "contextual_caption_reference_id": item.get(
                "contextual_caption_reference_id", ""
            )[:120],
            "contains_text": item.get("contains_text"),
            "needs_ocr": item.get("needs_ocr"),
            "needs_review": item.get("needs_review"),
            "uncertain_details": list(item.get("uncertain_details") or [])[:8],
            "bbox": item.get("bbox") or {},
        }
        if include_local_path and item.get("local_path"):
            compact["local_path"] = item["local_path"]
        compacted.append({k: v for k, v in compact.items() if v not in ("", None, [])})
    return compacted


def build_media_embedding_text(
    items: Iterable[Dict[str, Any]],
    *,
    max_items: int = 5,
    max_transcript_chars: int = 500,
) -> str:
    """Build a compact text block so media semantics participate in retrieval."""
    lines: List[str] = []
    for item in compact_media_for_metadata(items, max_items=max_items, max_text_len=220):
        title = item.get("title") or item.get("alt") or item.get("caption") or item.get("type", "media")
        parts = [f"{item.get('type', 'media').upper()}: {title}"]
        if item.get("caption") and item["caption"] != title:
            parts.append(f"caption={item['caption']}")
        if item.get("description") and item["description"] != title:
            parts.append(f"description={item['description'][:max_transcript_chars]}")
        if item.get("semantic_caption"):
            parts.append(f"visual_caption={item['semantic_caption']}")
        if item.get("contextual_caption"):
            parts.append(f"contextual_caption={item['contextual_caption']}")
        if item.get("visual_description"):
            parts.append(f"visual_description={item['visual_description'][:max_transcript_chars]}")
        if item.get("visible_text"):
            parts.append(f"visible_text={item['visible_text'][:max_transcript_chars]}")
        if item.get("ocr_status") == "completed" and item.get("ocr_text"):
            parts.append(f"exact_ocr={item['ocr_text'][:max_transcript_chars]}")
        if item.get("semantic_tags"):
            parts.append(f"tags={', '.join(item['semantic_tags'])}")
        if item.get("context"):
            parts.append(f"context={item['context']}")
        if item.get("section_path"):
            parts.append(f"section={' > '.join(item['section_path'])}")
        if item.get("surrounding_text_before"):
            parts.append(
                f"surrounding_before={item['surrounding_text_before'][:max_transcript_chars]}"
            )
        if item.get("surrounding_text_after"):
            parts.append(
                f"surrounding_after={item['surrounding_text_after'][:max_transcript_chars]}"
            )
        if item.get("nearby_text") and item.get("nearby_text") != item.get("context"):
            parts.append(f"nearby_text={item['nearby_text'][:max_transcript_chars]}")
        if item.get("provider"):
            parts.append(f"provider={item['provider']}")
        if item.get("transcript"):
            parts.append(f"transcript={item['transcript'][:max_transcript_chars]}")
        lines.append(" - ".join(parts))
    return "\n".join(lines)


def build_media_markdown(
    items: Iterable[Dict[str, Any]],
    *,
    prefer_local_images: bool = True,
    response_mode: bool = False,
    allow_html_video: bool = True,
    relative_to: Optional[str | Path] = None,
    max_images: Optional[int] = None,
    max_videos: Optional[int] = None,
    skip_non_semantic_images: bool = False,
) -> str:
    """
    Render media into markdown.

    For indexing/conversion we keep video markup conservative and link-based.
    For response generation, direct video URLs can be rendered with HTML5 video.
    """
    blocks: List[str] = []
    image_count = 0
    video_count = 0
    for item in dedupe_media_items(items):
        if item.get("type") == "image":
            if max_images is not None and image_count >= max_images:
                continue
            if skip_non_semantic_images and not _has_semantic_signal(item):
                continue
            src = item.get("local_path") if prefer_local_images and item.get("local_path") else item.get("url")
            if not src:
                continue
            if relative_to and item.get("local_path"):
                src = _relative_media_path(item.get("local_path"), relative_to) or src
            alt = item.get("alt") or item.get("caption") or item.get("title") or item.get("context") or "Image"
            block_lines = [f"![{alt}]({src})"]
            if item.get("caption") and item["caption"] != alt:
                block_lines.append(f"Caption: {item['caption']}")
            if item.get("context"):
                block_lines.append(f"Context: {item['context']}")
            blocks.append("\n".join(block_lines))
            image_count += 1
            continue

        url = item.get("url", "")
        if not url:
            continue
        if max_videos is not None and video_count >= max_videos:
            continue

        title = item.get("title") or item.get("caption") or item.get("alt") or "Video"
        block_lines = [f"### Video: {title}"]

        if response_mode and allow_html_video and is_direct_video_url(url):
            block_lines.append(f'<video controls src="{url}"></video>')
        else:
            block_lines.append(f"[Watch video]({url})")

        if item.get("poster_url"):
            block_lines.append(f"![{title} poster]({item['poster_url']})")
        if item.get("caption") and item["caption"] != title:
            block_lines.append(f"Caption: {item['caption']}")
        if item.get("context"):
            block_lines.append(f"Context: {item['context']}")
        if item.get("transcript"):
            block_lines.append(f"Transcript: {item['transcript'][:800]}")
        blocks.append("\n".join(block_lines))
        video_count += 1

    if not blocks:
        return ""
    return "## Embedded Media\n\n" + "\n\n".join(blocks)


def _has_semantic_signal(item: Dict[str, Any]) -> bool:
    signal_values = [
        item.get("alt", ""),
        item.get("title", ""),
        item.get("caption", ""),
        item.get("context", ""),
    ]
    generic = {"image", "photo", "figure", "graphic", "img", ""}
    for value in signal_values:
        cleaned = _clean_text(value).lower()
        if cleaned and cleaned not in generic and not cleaned.startswith("figure "):
            return True
    return False


def _relative_media_path(path: str | Path, relative_to: str | Path) -> str:
    try:
        target = Path(path).resolve()
        base = Path(relative_to).resolve()
        if base.is_file():
            base = base.parent
        return os.path.relpath(str(target), str(base)).replace("\\", "/")
    except Exception:
        return ""


def build_media_manifest(
    items: Iterable[Dict[str, Any]],
    *,
    kind: str = "document_media",
) -> Dict[str, Any]:
    normalized_items = []
    documents: Dict[str, Dict[str, Any]] = {}

    for item in dedupe_media_items(items):
        normalized = dict(item)
        local_path = normalized.get("local_path")
        if local_path:
            normalized["asset_uri"] = Path(local_path).resolve().as_uri()

        document_key = (
            str(normalized.get("document_id") or "")
            or str(normalized.get("md_path") or "")
            or str(normalized.get("source_file") or "")
            or "unknown"
        )
        document = documents.setdefault(
            document_key,
            {
                "document_id": str(normalized.get("document_id") or ""),
                "md_path": str(normalized.get("md_path") or ""),
                "source_file": str(normalized.get("source_file") or ""),
                "item_ids": [],
                "image_count": 0,
                "video_count": 0,
            },
        )
        document["item_ids"].append(str(normalized.get("id") or ""))
        if normalized.get("type") == "video":
            document["video_count"] += 1
        else:
            document["image_count"] += 1
        normalized_items.append(normalized)

    return {
        "version": 2,
        "kind": kind,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "documents": list(documents.values()),
        "items": normalized_items,
    }


def load_media_manifest_items(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, list):
        return dedupe_media_items(value)
    if isinstance(value, dict):
        if isinstance(value.get("items"), list):
            return dedupe_media_items(value["items"])
    return []


def parse_media_field(value: Any) -> List[Dict[str, Any]]:
    """Parse a metadata field that may already be a list or a JSON string."""
    if isinstance(value, list):
        return dedupe_media_items(value)
    if isinstance(value, dict):
        return load_media_manifest_items(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return []
        return load_media_manifest_items(parsed)
    return []


def build_retrieval_documents(
    docs: Iterable[Dict[str, Any]],
    *,
    max_media_per_doc: int = 4,
    max_total_media: int = 8,
) -> List[Dict[str, Any]]:
    """
    Build a stable retrieval payload for a response generator.

    The caller can pass Pinecone match dictionaries or pipeline document dicts.
    """
    prepared: List[Dict[str, Any]] = []
    remaining = max_total_media

    for doc in docs:
        metadata = doc.get("metadata") or {}
        media = parse_media_field(metadata.get("media"))
        if not media:
            media = parse_media_field(metadata.get("images"))
            media.extend(
                {**item, "type": "video"}
                for item in parse_media_field(metadata.get("videos"))
            )
        media = compact_media_for_metadata(media, max_items=min(max_media_per_doc, remaining))
        remaining = max(0, remaining - len(media))

        prepared.append(
            {
                "id": doc.get("id") or metadata.get("document_source") or "",
                "text": doc.get("text", ""),
                "source_url": metadata.get("document_source") or metadata.get("page_source") or "",
                "document_title": metadata.get("document_title") or "",
                "document_summary": metadata.get("document_summary") or "",
                "media": media,
            }
        )
        if remaining <= 0:
            break

    return prepared


def response_agent_media_instructions() -> str:
    """Instructions for an answer agent consuming structured retrieval payloads."""
    return (
        "Use only the supplied media objects. "
        "Embed images with markdown image syntax. "
        "For videos, prefer an HTML5 <video> block only when the URL points to a direct video file; "
        "otherwise use a markdown watch link. "
        "Do not invent media URLs, captions, or transcripts. "
        "Only include media when it is directly relevant to the answer."
    )


def is_direct_video_url(url: str) -> bool:
    return Path(url.split("?", 1)[0]).suffix.lower() in DIRECT_VIDEO_EXTENSIONS
