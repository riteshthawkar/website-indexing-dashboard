"""
Helpers for deriving dashboard metrics and media views from pipeline artifacts.

The pipeline now emits stage-local outputs plus an artifact catalog. The
dashboard should read those contracts directly instead of guessing from legacy
root-level files.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.artifacts import ArtifactCatalog, load_artifact_catalog
from pipeline.core.io import load_json_safe
from pipeline.core.media import dedupe_media_items, load_media_manifest_items, normalize_media_item
from pipeline.core.state import load_state


PAGE_MEDIA_FILENAME = "page_media.json"
PAGE_IMAGES_FILENAME = "page_images.json"
PAGE_VIDEOS_FILENAME = "page_videos.json"


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, "", False):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def _count_total_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for child in path.iterdir():
        if child.is_file():
            total += child.stat().st_size
            continue
        if child.is_dir() and child.name not in ("__pycache__", ".venv", "env"):
            try:
                total += sum(item.stat().st_size for item in child.rglob("*") if item.is_file())
            except OSError:
                continue
    return total


def _stage_metric_sum(state: Any, metric_key: str, *, stage_type: str | None = None, plugin_name: str | None = None) -> Optional[int]:
    if not state:
        return None

    total = 0
    found = False
    for stage in state.stages:
        if stage_type and stage.stage_type != stage_type:
            continue
        if plugin_name and stage.name != plugin_name:
            continue
        metric_value = _coerce_int((stage.metrics or {}).get(metric_key))
        if metric_value is None:
            continue
        total += metric_value
        found = True
    return total if found else None


def _latest_artifact_path(catalog: ArtifactCatalog, artifact_type: str) -> Optional[Path]:
    for record in reversed(catalog.filter(artifact_type=artifact_type)):
        if not record.local_path:
            continue
        path = Path(record.local_path)
        if path.exists():
            return path
    return None


def load_run_media(work_dir: str | Path) -> List[Dict[str, Any]]:
    """Load unique media items for a run from crawler and converter outputs."""
    root = Path(work_dir)
    if not root.exists():
        return []

    media_items: List[Dict[str, Any]] = []

    page_media = load_json_safe(root / PAGE_MEDIA_FILENAME, {}) or {}
    if not page_media:
        page_images = load_json_safe(root / PAGE_IMAGES_FILENAME, {}) or {}
        page_videos = load_json_safe(root / PAGE_VIDEOS_FILENAME, {}) or {}
        for page_url in sorted(set(page_images) | set(page_videos)):
            combined = []
            combined.extend(page_images.get(page_url) or [])
            combined.extend(page_videos.get(page_url) or [])
            if combined:
                page_media[page_url] = combined

    if isinstance(page_media, dict):
        for page_url, items in page_media.items():
            for item in load_media_manifest_items(items):
                normalized = normalize_media_item(
                    {
                        **item,
                        "source_type": item.get("source_type") or "html",
                        "source_url": item.get("source_url") or page_url,
                    }
                )
                normalized["page_url"] = page_url
                media_items.append(normalized)

    for manifest_path in root.rglob("extracted_images_index.json"):
        manifest = load_json_safe(manifest_path, {})
        for item in load_media_manifest_items(manifest):
            media_items.append(
                normalize_media_item(
                    {
                        **item,
                        "source_type": item.get("source_type") or "pdf",
                    }
                )
            )

    extra_fields: Dict[tuple[str, str], Dict[str, Any]] = {}
    for item in media_items:
        normalized = normalize_media_item(item)
        key = (str(normalized.get("type") or ""), str(normalized.get("url") or ""))
        if not all(key):
            continue
        stored = extra_fields.setdefault(key, {})
        if item.get("page_url") and not stored.get("page_url"):
            stored["page_url"] = item["page_url"]

    deduped = dedupe_media_items(media_items)
    for item in deduped:
        key = (str(item.get("type") or ""), str(item.get("url") or ""))
        extras = extra_fields.get(key) or {}
        item.update(extras)

    return sorted(
        deduped,
        key=lambda item: (
            item.get("type", ""),
            item.get("source_type", ""),
            str(item.get("page_number") or ""),
            str(item.get("source_url") or item.get("source_file") or item.get("url") or ""),
            str(item.get("position") or 0),
        ),
    )


def collect_media_summary(work_dir: str | Path) -> Dict[str, Any]:
    items = load_run_media(work_dir)
    images = sum(1 for item in items if item.get("type") == "image")
    videos = sum(1 for item in items if item.get("type") == "video")
    source_counts = Counter(str(item.get("source_type") or "unknown") for item in items)
    provider_counts = Counter(str(item.get("provider") or "unknown") for item in items if item.get("type") == "video")
    if "unknown" in provider_counts:
        del provider_counts["unknown"]

    return {
        "total": len(items),
        "images": images,
        "videos": videos,
        "by_source": dict(sorted(source_counts.items())),
        "video_providers": dict(sorted(provider_counts.items())),
    }


def collect_artifact_summary(work_dir: str | Path) -> Dict[str, Any]:
    catalog = load_artifact_catalog(work_dir)
    counts = Counter(record.artifact_type for record in catalog.records)
    return {
        "total": len(catalog.records),
        "by_type": dict(sorted(counts.items())),
    }


def collect_metrics(work_dir: str | Path) -> Dict[str, Any]:
    """Collect dashboard-facing run metrics from current pipeline contracts."""
    root = Path(work_dir)
    if not root.exists():
        return {}

    state = load_state(root)
    catalog = load_artifact_catalog(root)
    artifact_summary = collect_artifact_summary(root)
    media_summary = collect_media_summary(root)

    chunk_index_payload: Dict[str, Any] = {}
    chunk_index_path = _latest_artifact_path(catalog, "chunk_index")
    if chunk_index_path:
        chunk_index_payload = load_json_safe(chunk_index_path, {}) or {}
    else:
        chunk_index_payload = load_json_safe(root / "stage_outputs" / "chunk_content" / "chunks" / "chunk_index.json", {}) or {}

    formatted_payload = []
    formatted_docs_path = _latest_artifact_path(catalog, "formatted_documents")
    if formatted_docs_path:
        formatted_payload = load_json_safe(formatted_docs_path, []) or []
    else:
        formatted_payload = load_json_safe(root / "stage_outputs" / "format_embeddings" / "formatted_for_embedding.json", []) or []

    artifact_counts = artifact_summary["by_type"]

    pages_scraped = _stage_metric_sum(state, "pages_scraped", stage_type="crawler", plugin_name="crawl4ai")
    if pages_scraped is None:
        pages_scraped = _count_files(root / "html", "*.html")

    documents_downloaded = _stage_metric_sum(state, "documents_downloaded", stage_type="crawler", plugin_name="crawl4ai")
    if documents_downloaded is None:
        documents_downloaded = _count_files(root / "downloads")

    pages_cleaned = artifact_counts.get("cleaned_html")
    if pages_cleaned is None:
        pages_cleaned = _stage_metric_sum(state, "cleaned", stage_type="cleaner")
    if pages_cleaned is None:
        pages_cleaned = _count_files(root / "cleaned_html", "*.html") + _count_files(root / "cleaned", "*.html")

    docs_converted = artifact_counts.get("markdown")
    if docs_converted is None:
        docs_converted = _stage_metric_sum(state, "converted", stage_type="converter")
    if docs_converted is None:
        docs_converted = _count_files(root / "markdown", "*.md")

    summaries_generated = artifact_counts.get("summary")
    if summaries_generated is None:
        summaries_generated = _stage_metric_sum(state, "processed", stage_type="summarizer")
    if summaries_generated is None:
        summaries_generated = _count_files(root / "summaries", "*.summary.json")

    structured_documents_created = artifact_counts.get("structured_document")
    if structured_documents_created is None:
        structured_documents_created = _count_files(root / "stage_outputs" / "convert_documents" / "structured_documents", "*.json")

    chunks_created = _coerce_int(chunk_index_payload.get("chunk_count"))
    if chunks_created is None:
        chunks_created = _stage_metric_sum(state, "chunks", stage_type="chunker")
    if chunks_created is None and isinstance(formatted_payload, list):
        chunks_created = len(formatted_payload)
    if chunks_created is None:
        chunks_created = 0

    embeddings_created = len(formatted_payload) if isinstance(formatted_payload, list) else None
    if embeddings_created is None:
        embeddings_created = _stage_metric_sum(state, "documents_total", stage_type="embedder")
    if embeddings_created is None:
        embeddings_created = _stage_metric_sum(state, "formatted_count", stage_type="formatter")
    if embeddings_created is None:
        embeddings_created = 0

    chunk_strategy = chunk_index_payload.get("strategy")
    if not chunk_strategy:
        for stage in reversed(state.stages if state else []):
            if stage.stage_type == "chunker":
                chunk_strategy = stage.name
                break

    return {
        "pages_scraped": int(pages_scraped or 0),
        "documents_downloaded": int(documents_downloaded or 0),
        "pages_cleaned": int(pages_cleaned or 0),
        "docs_converted": int(docs_converted or 0),
        "summaries_generated": int(summaries_generated or 0),
        "embeddings_created": int(embeddings_created or 0),
        "images_extracted": int(media_summary["images"]),
        "videos_extracted": int(media_summary["videos"]),
        "media_items_extracted": int(media_summary["total"]),
        "structured_documents_created": int(structured_documents_created or 0),
        "chunks_created": int(chunks_created or 0),
        "chunk_strategy": str(chunk_strategy or ""),
        "artifact_count": int(artifact_summary["total"]),
        "total_bytes": _count_total_bytes(root),
    }
