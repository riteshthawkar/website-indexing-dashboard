"""Complete PDF visual extraction without mutating an earlier corpus run.

The original media-enrichment pass handled ordinary Docling documents, but it
did not expand page-window wrappers and could not resolve quarantined Docling
layouts after a text-conversion fallback.  This stage consumes an immutable
semantically annotated corpus, repairs PDF URL lineage from raw crawl mappings,
recovers the missing visual regions, and publishes new manifests and assets.
"""

from __future__ import annotations

import hashlib
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import urlsplit

from pipeline.core.artifacts import load_artifact_catalog
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe, sha256_file
from pipeline.core.media import build_media_manifest, load_media_manifest_items, normalize_media_item
from pipeline.core.registry import register_stage
from pipeline.core.state import load_state
from pipeline.stages.formatters.media_enrichment_formatter import (
    _docling_box_to_fitz,
    _hamming_distance,
    _image_inspection,
    _picture_caption,
    _picture_context,
    _stable_id,
)


_WINDOW_SCHEMA = "mbzuai_docling_page_windows.v1"
_PAGE_WINDOW_MARKER = ".docling.pages_"


def _resolve_path(value: Any) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _http_url(value: Any) -> str:
    text = str(value or "").strip()
    try:
        parsed = urlsplit(text)
    except ValueError:
        return ""
    return text if parsed.scheme.lower() in {"http", "https"} and parsed.netloc else ""


def _conversion_outputs(run_dir: Path, stage_id: str) -> Dict[str, Any]:
    state = load_state(run_dir)
    if state is None:
        raise ValueError(f"Conversion run has no pipeline state: {run_dir}")
    if state.status not in {"paused", "completed"}:
        raise ValueError(
            f"Conversion run {state.run_id!r} must be paused or completed, found {state.status!r}"
        )
    for stage in state.stages:
        if str(stage.stage_id or "") == stage_id and stage.status == "completed":
            return dict(stage.outputs or {})
    raise ValueError(f"Conversion run {state.run_id!r} has no completed {stage_id!r} stage")


def _mapping_indexes(
    mapping_files: Iterable[Path],
) -> Tuple[Dict[str, str], Dict[str, str], List[Dict[str, Any]]]:
    url_by_path: Dict[str, str] = {}
    paths_by_name: MutableMapping[str, set[str]] = defaultdict(set)
    evidence: List[Dict[str, Any]] = []
    for mapping_path in sorted(set(mapping_files)):
        if not mapping_path.is_file():
            raise ValueError(f"Raw crawl mapping is missing: {mapping_path}")
        payload = load_json_safe(mapping_path, {}) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Raw crawl mapping is not an object: {mapping_path}")
        mapped_pdf_count = 0
        for raw_url, raw_path in payload.items():
            source_url = _http_url(raw_url)
            path = Path(str(raw_path or "")).expanduser()
            if not source_url or not path.is_file() or path.suffix.lower() != ".pdf":
                continue
            resolved = str(path.resolve())
            existing = url_by_path.get(resolved)
            if existing and existing != source_url:
                raise ValueError(f"Conflicting public URLs for PDF {resolved}: {existing}, {source_url}")
            url_by_path[resolved] = source_url
            paths_by_name[path.name].add(resolved)
            mapped_pdf_count += 1
        evidence.append(
            {
                "path": str(mapping_path),
                "sha256": sha256_file(mapping_path),
                "mapped_pdf_count": mapped_pdf_count,
            }
        )
    unique_path_by_name = {
        name: next(iter(paths)) for name, paths in paths_by_name.items() if len(paths) == 1
    }
    return url_by_path, unique_path_by_name, evidence


def _mapping_candidates_from_sources(source_files: Iterable[str]) -> set[Path]:
    candidates: set[Path] = set()
    for source_file in source_files:
        path = Path(str(source_file or ""))
        if path.parent.name == "downloads":
            candidate = path.parent.parent / "mappings.json"
            if candidate.is_file():
                candidates.add(candidate.resolve())
    return candidates


def _current_markdown_by_source(ctx: StageContext) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for artifact in ctx.find_artifacts(artifact_type="markdown"):
        metadata = dict(artifact.metadata or {})
        source_file = str(metadata.get("source_file") or "")
        if source_file and artifact.local_path and Path(artifact.local_path).is_file():
            mapping[str(Path(source_file).resolve())] = str(Path(artifact.local_path).resolve())
    return mapping


def _source_file_from_payload(payload: Mapping[str, Any], by_name: Mapping[str, str]) -> str:
    direct = str(payload.get("source_file") or "")
    if direct and Path(direct).is_file():
        return str(Path(direct).resolve())
    origin = payload.get("origin") if isinstance(payload.get("origin"), dict) else {}
    filename = str(origin.get("filename") or "")
    return str(by_name.get(filename) or "")


def _discover_layouts(
    *,
    conversion_run_dirs: Sequence[Path],
    conversion_stage_id: str,
    path_by_name: Mapping[str, str],
    url_by_path: Mapping[str, str],
    markdown_by_source: Mapping[str, str],
    include_unused_fallback_layouts: bool,
    include_quarantined_layouts: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    layouts: List[Dict[str, Any]] = []
    documents: Dict[str, Dict[str, Any]] = {}
    run_evidence: List[Dict[str, Any]] = []
    seen_layouts: set[Tuple[str, str]] = set()

    for run_dir in conversion_run_dirs:
        outputs = _conversion_outputs(run_dir, conversion_stage_id)
        catalog = load_artifact_catalog(run_dir)
        run_layout_count = 0
        wrapper_count = 0
        missing_parts: List[str] = []
        source_metadata: Dict[str, Dict[str, Any]] = {}
        for artifact in catalog.filter(artifact_type="structured_document"):
            metadata = dict(artifact.metadata or {})
            source_file = str(metadata.get("source_file") or "")
            if not source_file or not Path(source_file).is_file() or Path(source_file).suffix.lower() != ".pdf":
                continue
            source_file = str(Path(source_file).resolve())
            source_metadata[source_file] = metadata
            documents.setdefault(
                source_file,
                {
                    "source_file": source_file,
                    "source_url": str(url_by_path.get(source_file) or ""),
                    "source_markdown_path": str(
                        markdown_by_source.get(source_file)
                        or metadata.get("source_markdown_path")
                        or ""
                    ),
                    "layout_count": 0,
                    "picture_box_count": 0,
                    "from_window_wrapper": False,
                    "from_unused_fallback": False,
                    "from_quarantine": False,
                },
            )
            if not artifact.local_path or not Path(artifact.local_path).is_file():
                continue
            structured_path = Path(artifact.local_path).resolve()
            payload = load_json_safe(structured_path, {}) or {}
            if not isinstance(payload, dict):
                continue
            if str(payload.get("schema") or "") == _WINDOW_SCHEMA:
                wrapper_count += 1
                documents[source_file]["from_window_wrapper"] = True
                for part in payload.get("parts") or []:
                    if not isinstance(part, dict):
                        continue
                    part_path = Path(str(part.get("structured_document_path") or ""))
                    if not part_path.is_file():
                        missing_parts.append(str(part_path))
                        continue
                    key = (source_file, str(part_path.resolve()))
                    if key in seen_layouts:
                        continue
                    seen_layouts.add(key)
                    layouts.append(
                        {
                            "source_file": source_file,
                            "source_url": str(url_by_path.get(source_file) or ""),
                            "source_markdown_path": documents[source_file]["source_markdown_path"],
                            "structured_path": str(part_path.resolve()),
                            "layout_source": "docling_page_window",
                            "page_range": list(part.get("page_range") or []),
                            "full_page_fallback": False,
                        }
                    )
                    documents[source_file]["layout_count"] += 1
                    run_layout_count += 1
                continue
            key = (source_file, str(structured_path))
            if key not in seen_layouts:
                seen_layouts.add(key)
                layouts.append(
                    {
                        "source_file": source_file,
                        "source_url": str(url_by_path.get(source_file) or ""),
                        "source_markdown_path": documents[source_file]["source_markdown_path"],
                        "structured_path": str(structured_path),
                        "layout_source": "docling",
                        "page_range": [],
                        "full_page_fallback": False,
                    }
                )
                documents[source_file]["layout_count"] += 1
                run_layout_count += 1

        if missing_parts:
            raise ValueError(
                f"Page-window wrapper in {run_dir} has {len(missing_parts)} missing parts; "
                f"first={missing_parts[0]}"
            )

        fallback_count = 0
        if include_unused_fallback_layouts:
            quarantine = Path(str(outputs.get("quarantine_dir") or ""))
            unused_dir = quarantine / "structured_documents" / "unused_after_fallback"
            for structured_path in sorted(unused_dir.glob("*.docling.json")) if unused_dir.is_dir() else []:
                if _PAGE_WINDOW_MARKER in structured_path.name:
                    continue
                payload = load_json_safe(structured_path, {}) or {}
                if not isinstance(payload, dict):
                    continue
                source_file = _source_file_from_payload(payload, path_by_name)
                if not source_file or Path(source_file).suffix.lower() != ".pdf":
                    continue
                metadata = source_metadata.get(source_file, {})
                document = documents.setdefault(
                    source_file,
                    {
                        "source_file": source_file,
                        "source_url": str(url_by_path.get(source_file) or ""),
                        "source_markdown_path": str(
                            markdown_by_source.get(source_file)
                            or metadata.get("source_markdown_path")
                            or ""
                        ),
                        "layout_count": 0,
                        "picture_box_count": 0,
                        "from_window_wrapper": False,
                        "from_unused_fallback": False,
                        "from_quarantine": False,
                    },
                )
                key = (source_file, str(structured_path.resolve()))
                if key in seen_layouts:
                    continue
                seen_layouts.add(key)
                layouts.append(
                    {
                        "source_file": source_file,
                        "source_url": str(url_by_path.get(source_file) or ""),
                        "source_markdown_path": document["source_markdown_path"],
                        "structured_path": str(structured_path.resolve()),
                        "layout_source": "docling_unused_after_fallback",
                        "page_range": [],
                        "full_page_fallback": True,
                    }
                )
                document["layout_count"] += 1
                document["from_unused_fallback"] = True
                run_layout_count += 1
                fallback_count += 1

        quarantined_count = 0
        if include_quarantined_layouts:
            quarantine = Path(str(outputs.get("quarantine_dir") or ""))
            quarantined_dir = quarantine / "structured_documents"
            for structured_path in (
                sorted(quarantined_dir.glob("*.docling.json"))
                if quarantined_dir.is_dir()
                else []
            ):
                if _PAGE_WINDOW_MARKER in structured_path.name:
                    continue
                payload = load_json_safe(structured_path, {}) or {}
                if not isinstance(payload, dict):
                    continue
                source_file = _source_file_from_payload(payload, path_by_name)
                if not source_file or Path(source_file).suffix.lower() != ".pdf":
                    continue
                metadata = source_metadata.get(source_file, {})
                document = documents.setdefault(
                    source_file,
                    {
                        "source_file": source_file,
                        "source_url": str(url_by_path.get(source_file) or ""),
                        "source_markdown_path": str(
                            markdown_by_source.get(source_file)
                            or metadata.get("source_markdown_path")
                            or ""
                        ),
                        "layout_count": 0,
                        "picture_box_count": 0,
                        "from_window_wrapper": False,
                        "from_unused_fallback": False,
                        "from_quarantine": False,
                    },
                )
                key = (source_file, str(structured_path.resolve()))
                if key in seen_layouts:
                    continue
                seen_layouts.add(key)
                layouts.append(
                    {
                        "source_file": source_file,
                        "source_url": str(url_by_path.get(source_file) or ""),
                        "source_markdown_path": document["source_markdown_path"],
                        "structured_path": str(structured_path.resolve()),
                        "layout_source": "docling_quarantine",
                        "page_range": [],
                        # A quarantined text conversion is not trusted as
                        # corpus evidence. Full-page visual OCR is the safe
                        # recovery path for short scanned documents.
                        "full_page_fallback": True,
                    }
                )
                document["layout_count"] += 1
                document["from_quarantine"] = True
                run_layout_count += 1
                quarantined_count += 1

        run_evidence.append(
            {
                "run_dir": str(run_dir),
                "run_id": str((load_state(run_dir) or {}).run_id if load_state(run_dir) else ""),
                "layout_count": run_layout_count,
                "window_wrapper_count": wrapper_count,
                "unused_fallback_layout_count": fallback_count,
                "quarantined_layout_count": quarantined_count,
            }
        )
    return layouts, documents, run_evidence


def _semantic_richness(item: Mapping[str, Any]) -> int:
    return sum(
        len(str(item.get(key) or ""))
        for key in (
            "semantic_caption",
            "contextual_caption",
            "visual_description",
            "visible_text",
            "context",
            "source_url",
        )
    )


def _repair_pdf_items(
    items: Iterable[Mapping[str, Any]],
    *,
    url_by_path: Mapping[str, str],
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    repaired: List[Dict[str, Any]] = []
    repair_count = 0
    unresolved: List[str] = []
    for raw in items:
        item = normalize_media_item(dict(raw))
        if str(item.get("source_type") or "").lower() != "pdf":
            continue
        source_file = str(item.get("source_file") or "")
        resolved_source = str(Path(source_file).resolve()) if source_file and Path(source_file).is_file() else ""
        source_url = str(item.get("source_url") or url_by_path.get(resolved_source) or "")
        if not _http_url(source_url):
            unresolved.append(source_file or str(item.get("document_id") or "unknown"))
        elif source_url != item.get("source_url"):
            repair_count += 1
        item["source_url"] = source_url
        repaired.append(normalize_media_item(item))
    return repaired, repair_count, sorted(set(unresolved))


def _existing_indexes(
    items: Sequence[Mapping[str, Any]],
) -> Tuple[set[str], set[str], Dict[str, List[Tuple[str, float]]], Counter[str]]:
    content_hashes: set[str] = set()
    ids: set[str] = set()
    perceptual_by_source: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    metrics: Counter[str] = Counter()
    for item in items:
        path = Path(str(item.get("local_path") or ""))
        declared_hash = str(item.get("content_hash") or "").lower()
        if not path.is_file():
            raise ValueError(f"Existing PDF media asset is missing: {path}")
        actual_hash = sha256_file(path)
        if declared_hash and declared_hash != actual_hash:
            raise ValueError(f"Existing PDF media hash mismatch: {path}")
        content_hashes.add(actual_hash)
        if item.get("id"):
            ids.add(str(item["id"]))
        source_file = str(Path(str(item.get("source_file") or "")).resolve())
        perceptual_hash = str(item.get("perceptual_hash") or "")
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        if source_file and perceptual_hash and width > 0 and height > 0:
            perceptual_by_source[source_file].append((perceptual_hash, width / height))
        metrics["existing_pdf_items_validated"] += 1
    return content_hashes, ids, perceptual_by_source, metrics


def _write_generated_image(output_dir: Path, content_hash: str, payload: bytes) -> Path:
    path = output_dir / f"{content_hash}.png"
    if path.is_file():
        if sha256_file(path) != content_hash:
            raise ValueError(f"Generated-image path collision: {path}")
        return path
    temporary = path.with_suffix(".png.part")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return path


def _new_media_item(
    *,
    source: Mapping[str, Any],
    payload: Mapping[str, Any],
    page_number: int,
    box: Tuple[float, float, float, float],
    area_ratio: float,
    image_id: str,
    image_path: Path,
    inspection: Mapping[str, Any],
    caption: str,
    context: str,
    crop_source: str,
) -> Dict[str, Any]:
    source_file = str(Path(str(source["source_file"])).resolve())
    document_id = str(payload.get("name") or Path(source_file).stem)
    title = caption or f"Visual from {document_id} page {page_number}"
    return normalize_media_item(
        {
            "type": "image",
            "id": image_id,
            "url": image_path.resolve().as_uri(),
            "asset_uri": image_path.resolve().as_uri(),
            "local_path": str(image_path.resolve()),
            "alt": title,
            "title": title,
            "caption": caption,
            "description": "",
            "context": context,
            "source_type": "pdf",
            "source_backend": str(source.get("layout_source") or "docling_layout"),
            "source_file": source_file,
            "source_url": str(source.get("source_url") or ""),
            "source_document_path": str(source.get("source_markdown_path") or ""),
            "md_path": str(source.get("source_markdown_path") or ""),
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
            "crop_source": crop_source,
            "download_status": "generated",
            **inspection,
        }
    )


def _extract_recovered_media(
    *,
    layouts: Sequence[Mapping[str, Any]],
    documents: MutableMapping[str, Dict[str, Any]],
    existing_items: Sequence[Mapping[str, Any]],
    output_dir: Path,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError("PyMuPDF is required for PDF media completion") from exc

    content_hashes, existing_ids, perceptual_by_source, metrics = _existing_indexes(existing_items)
    new_items: List[Dict[str, Any]] = []
    scale = max(1.0, float(config.get("pdf_crop_scale", 2.0)))
    padding = max(0.0, float(config.get("pdf_crop_padding_points", 3.0)))
    minimum_area_ratio = max(0.0, float(config.get("min_pdf_bbox_area_ratio", 0.003)))
    maximum_area_ratio = min(1.0, float(config.get("max_pdf_bbox_area_ratio", 0.90)))
    near_duplicate_distance = max(0, int(config.get("pdf_perceptual_duplicate_distance", 2)))
    maximum_full_page_count = max(0, int(config.get("fallback_full_page_max_pages", 3)))
    full_page_only_unrepresented = bool(
        config.get("fallback_full_page_only_when_unrepresented", True)
    )
    represented_sources = {
        str(Path(str(item.get("source_file") or "")).resolve()) for item in existing_items
    }

    for source in layouts:
        structured_path = Path(str(source.get("structured_path") or ""))
        payload = load_json_safe(structured_path, {}) or {}
        if not isinstance(payload, dict):
            metrics["layout_invalid"] += 1
            continue
        source_file = str(Path(str(source["source_file"])).resolve())
        pictures = payload.get("pictures") if isinstance(payload.get("pictures"), list) else []
        pages = payload.get("pages") if isinstance(payload.get("pages"), dict) else {}
        candidates: List[Tuple[int, float, Mapping[str, Any], Mapping[str, Any]]] = []
        for picture in pictures:
            if not isinstance(picture, dict):
                continue
            for provenance in picture.get("prov") or []:
                if not isinstance(provenance, dict):
                    continue
                metrics["layout_boxes_seen"] += 1
                page_number = int(provenance.get("page_no") or 0)
                bbox = provenance.get("bbox") if isinstance(provenance.get("bbox"), dict) else {}
                page_info = pages.get(str(page_number), {}) if isinstance(pages, dict) else {}
                size = page_info.get("size") if isinstance(page_info, dict) else {}
                page_width = float((size or {}).get("width") or 0)
                page_height = float((size or {}).get("height") or 0)
                width = abs(float(bbox.get("r") or 0) - float(bbox.get("l") or 0))
                height = abs(float(bbox.get("t") or 0) - float(bbox.get("b") or 0))
                area_ratio = width * height / max(1.0, page_width * page_height)
                if page_number <= 0 or not (minimum_area_ratio <= area_ratio <= maximum_area_ratio):
                    metrics["layout_boxes_size_filtered"] += 1
                    continue
                candidates.append((page_number, area_ratio, picture, provenance))
                documents[source_file]["picture_box_count"] += 1

        try:
            with fitz.open(source_file) as document:
                for page_number, area_ratio, picture, provenance in candidates:
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
                    image_id = _stable_id(
                        "pdf-figure",
                        source_file,
                        page_number,
                        *(round(float(value), 3) for value in box),
                    )
                    if image_id in existing_ids:
                        metrics["layout_boxes_already_materialized"] += 1
                        continue
                    pixmap = page.get_pixmap(
                        matrix=fitz.Matrix(scale, scale), clip=fitz.Rect(*box), alpha=False
                    )
                    image_bytes = pixmap.tobytes("png")
                    inspection, reason = _image_inspection(
                        image_bytes,
                        minimum_width=max(1, int(config.get("min_pdf_crop_width", 96))),
                        minimum_height=max(1, int(config.get("min_pdf_crop_height", 72))),
                        minimum_area=max(1, int(config.get("min_pdf_crop_area", 12000))),
                        maximum_pixels=max(1, int(config.get("max_image_pixels", 50_000_000))),
                        maximum_aspect_ratio=max(
                            1.0, float(config.get("max_pdf_crop_aspect_ratio", 12.0))
                        ),
                        minimum_stddev=max(0.0, float(config.get("min_visual_stddev", 2.0))),
                    )
                    if inspection is None:
                        metrics[f"crop_rejected_{reason}"] += 1
                        continue
                    content_hash = str(inspection["content_hash"])
                    if content_hash in content_hashes:
                        metrics["crop_exact_duplicates"] += 1
                        continue
                    aspect = int(inspection["width"]) / max(1, int(inspection["height"]))
                    perceptual_hash = str(inspection["perceptual_hash"])
                    if any(
                        _hamming_distance(perceptual_hash, prior_hash) <= near_duplicate_distance
                        and abs(math.log(max(aspect, 1e-6) / max(prior_aspect, 1e-6))) < 0.08
                        for prior_hash, prior_aspect in perceptual_by_source[source_file]
                    ):
                        metrics["crop_perceptual_duplicates"] += 1
                        continue
                    path = _write_generated_image(output_dir, content_hash, image_bytes)
                    caption = _picture_caption(payload, picture)
                    context = _picture_context(
                        payload,
                        page_number=page_number,
                        bbox=provenance.get("bbox") or {},
                        caption=caption,
                    )
                    new_items.append(
                        _new_media_item(
                            source=source,
                            payload=payload,
                            page_number=page_number,
                            box=box,
                            area_ratio=area_ratio,
                            image_id=image_id,
                            image_path=path,
                            inspection=inspection,
                            caption=caption,
                            context=context,
                            crop_source="docling_layout_recovery",
                        )
                    )
                    existing_ids.add(image_id)
                    content_hashes.add(content_hash)
                    perceptual_by_source[source_file].append((perceptual_hash, aspect))
                    represented_sources.add(source_file)
                    metrics["pdf_crops_recovered"] += 1

                use_full_page = bool(source.get("full_page_fallback")) and len(document) <= maximum_full_page_count
                if full_page_only_unrepresented and source_file in represented_sources:
                    use_full_page = False
                if use_full_page:
                    for page_index in range(len(document)):
                        page_number = page_index + 1
                        page = document.load_page(page_index)
                        image_id = _stable_id("pdf-full-page", source_file, page_number)
                        if image_id in existing_ids:
                            continue
                        image_bytes = page.get_pixmap(
                            matrix=fitz.Matrix(scale, scale), alpha=False
                        ).tobytes("png")
                        inspection, reason = _image_inspection(
                            image_bytes,
                            minimum_width=max(1, int(config.get("min_pdf_crop_width", 96))),
                            minimum_height=max(1, int(config.get("min_pdf_crop_height", 72))),
                            minimum_area=max(1, int(config.get("min_pdf_crop_area", 12000))),
                            maximum_pixels=max(1, int(config.get("max_image_pixels", 50_000_000))),
                            maximum_aspect_ratio=max(
                                1.0, float(config.get("max_pdf_crop_aspect_ratio", 12.0))
                            ),
                            minimum_stddev=max(0.0, float(config.get("min_visual_stddev", 2.0))),
                        )
                        if inspection is None:
                            metrics[f"full_page_rejected_{reason}"] += 1
                            continue
                        content_hash = str(inspection["content_hash"])
                        if content_hash in content_hashes:
                            metrics["full_page_exact_duplicates"] += 1
                            continue
                        path = _write_generated_image(output_dir, content_hash, image_bytes)
                        box = (0.0, 0.0, float(page.rect.width), float(page.rect.height))
                        new_items.append(
                            _new_media_item(
                                source=source,
                                payload=payload,
                                page_number=page_number,
                                box=box,
                                area_ratio=1.0,
                                image_id=image_id,
                                image_path=path,
                                inspection=inspection,
                                caption="",
                                context="",
                                crop_source="pdf_full_page_fallback",
                            )
                        )
                        content_hashes.add(content_hash)
                        existing_ids.add(image_id)
                        represented_sources.add(source_file)
                        metrics["pdf_full_pages_recovered"] += 1
        except (OSError, RuntimeError, ValueError):
            metrics["pdf_documents_failed"] += 1
            documents[source_file]["failed"] = True

    # This recovery implementation has no per-page cap by construction.
    metrics["layout_boxes_page_cap_filtered"] = 0
    return new_items, metrics


def _replace_pdf_manifest_items(
    input_items: Iterable[Mapping[str, Any]], pdf_items: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    output = [
        normalize_media_item(dict(item))
        for item in input_items
        if str(item.get("source_type") or "").lower() != "pdf"
    ]
    output.extend(normalize_media_item(dict(item)) for item in pdf_items)
    return output


@register_stage
class PdfMediaCompletionFormatter(FormatterStage):
    name = "pdf_media_completion"
    description = "Recovers windowed/fallback PDF visuals and repairs public URL lineage."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        formatter = config.get("formatter") if isinstance(config.get("formatter"), dict) else {}
        stage_config = formatter.get("pdf_media_completion") if isinstance(formatter, dict) else {}
        if not isinstance(stage_config, dict):
            return ["formatter.pdf_media_completion must be a mapping"]
        errors: List[str] = []
        if not isinstance(stage_config.get("conversion_run_dirs"), list) or not stage_config.get(
            "conversion_run_dirs"
        ):
            errors.append("formatter.pdf_media_completion.conversion_run_dirs must be a non-empty list")
        if not isinstance(stage_config.get("raw_mapping_files"), list) or not stage_config.get(
            "raw_mapping_files"
        ):
            errors.append("formatter.pdf_media_completion.raw_mapping_files must be a non-empty list")
        for key in ("pdf_crop_scale", "min_pdf_crop_width", "min_pdf_crop_height"):
            try:
                if float(stage_config.get(key, 1)) <= 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"formatter.pdf_media_completion.{key} must be positive")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.formatter_config.get("pdf_media_completion") or {}
        if not isinstance(config, dict):
            return StageResult.failure("formatter.pdf_media_completion must be a mapping")
        try:
            conversion_run_dirs = [
                _resolve_path(value) for value in config.get("conversion_run_dirs") or []
            ]
            conversion_stage_id = str(config.get("conversion_stage_id") or "convert_documents")
            preliminary_source_files: List[str] = []
            for run_dir in conversion_run_dirs:
                catalog = load_artifact_catalog(run_dir)
                preliminary_source_files.extend(
                    str((record.metadata or {}).get("source_file") or "")
                    for record in catalog.filter(artifact_type="structured_document")
                )
            mapping_files = {
                _resolve_path(value) for value in config.get("raw_mapping_files") or []
            }
            mapping_files.update(_mapping_candidates_from_sources(preliminary_source_files))
            url_by_path, path_by_name, mapping_evidence = _mapping_indexes(mapping_files)
            markdown_by_source = _current_markdown_by_source(ctx)
            layouts, documents, conversion_evidence = _discover_layouts(
                conversion_run_dirs=conversion_run_dirs,
                conversion_stage_id=conversion_stage_id,
                path_by_name=path_by_name,
                url_by_path=url_by_path,
                markdown_by_source=markdown_by_source,
                include_unused_fallback_layouts=bool(
                    config.get("include_unused_fallback_layouts", True)
                ),
                include_quarantined_layouts=bool(
                    config.get("include_quarantined_layouts", False)
                ),
            )

            document_manifest = load_json_safe(
                ctx.previous_outputs.get("extracted_images_index_file"), {}
            ) or {}
            manifest = load_json_safe(ctx.previous_outputs.get("media_manifest_file"), {}) or {}
            input_document_items = load_media_manifest_items(document_manifest)
            input_manifest_items = load_media_manifest_items(manifest)
            repaired_items, source_url_repairs, unresolved_existing = _repair_pdf_items(
                input_document_items, url_by_path=url_by_path
            )
            if unresolved_existing:
                raise ValueError(
                    f"Could not resolve public URLs for {len(unresolved_existing)} existing PDF media sources; "
                    f"first={unresolved_existing[0]}"
                )
            output_dir = ensure_dir(ctx.output_dir("recovered_pdf_media"))
            new_items, extraction_metrics = _extract_recovered_media(
                layouts=layouts,
                documents=documents,
                existing_items=repaired_items,
                output_dir=output_dir,
                config=config,
            )
            completed_pdf_items = [*repaired_items, *new_items]
            unresolved_output = sorted(
                {
                    str(item.get("source_file") or item.get("document_id") or "unknown")
                    for item in completed_pdf_items
                    if not _http_url(item.get("source_url"))
                }
            )
            if unresolved_output:
                raise ValueError(
                    f"Completed PDF media still has {len(unresolved_output)} unresolved source URLs"
                )

            page_media_file = str(ctx.previous_outputs.get("page_media_file") or "")
            page_images_file = str(ctx.previous_outputs.get("page_images_file") or "")
            document_path = ctx.stage_work_dir / "completed_pdf_document_media.json"
            manifest_path = ctx.stage_work_dir / "completed_multimodal_media_manifest.json"
            completed_manifest_items = _replace_pdf_manifest_items(
                input_manifest_items, completed_pdf_items
            )
            atomic_write_json(
                document_path,
                build_media_manifest(completed_pdf_items, kind="completed_pdf_document_media"),
            )
            atomic_write_json(
                manifest_path,
                build_media_manifest(completed_manifest_items, kind="completed_multimodal_media"),
            )

            represented_counts = Counter(
                str(Path(str(item.get("source_file") or "")).resolve())
                for item in completed_pdf_items
            )
            for source_file, document in documents.items():
                document["source_url"] = str(url_by_path.get(source_file) or document.get("source_url") or "")
                document["media_count"] = represented_counts.get(source_file, 0)
                if document.get("failed"):
                    document["status"] = "failed"
                elif document["media_count"]:
                    document["status"] = "represented"
                elif document.get("picture_box_count"):
                    document["status"] = "visuals_filtered"
                else:
                    document["status"] = "no_visuals_detected"
            status_counts = Counter(str(value.get("status") or "unknown") for value in documents.values())
            failed_documents = status_counts.get("failed", 0)
            missing_document_urls = sorted(
                source_file for source_file, value in documents.items() if not _http_url(value.get("source_url"))
            )
            minimum_documents = max(0, int(config.get("minimum_pdf_documents", 1)))
            gates = {
                "minimum_pdf_documents": minimum_documents,
                "discovered_pdf_documents": len(documents),
                "minimum_pdf_documents_passed": len(documents) >= minimum_documents,
                "all_pdf_media_have_public_source_url": not unresolved_output,
                "all_discovered_pdfs_have_public_source_url": not missing_document_urls,
                "window_parts_resolved": True,
                "failed_document_count": failed_documents,
                "maximum_failed_documents": max(
                    0, int(config.get("maximum_failed_documents", 0))
                ),
                "failed_documents_passed": failed_documents
                <= max(0, int(config.get("maximum_failed_documents", 0))),
                "page_cap_filtered_count": int(
                    extraction_metrics.get("layout_boxes_page_cap_filtered", 0)
                ),
                "zero_page_cap_filtering_passed": int(
                    extraction_metrics.get("layout_boxes_page_cap_filtered", 0)
                )
                == 0,
            }
            gate_passed = all(
                bool(gates[key])
                for key in (
                    "minimum_pdf_documents_passed",
                    "all_pdf_media_have_public_source_url",
                    "all_discovered_pdfs_have_public_source_url",
                    "window_parts_resolved",
                    "failed_documents_passed",
                    "zero_page_cap_filtering_passed",
                )
            )
            report = {
                "version": 1,
                "kind": "pdf_media_completion_report",
                "input_document_media_file": str(
                    ctx.previous_outputs.get("extracted_images_index_file") or ""
                ),
                "input_document_media_sha256": sha256_file(
                    ctx.previous_outputs["extracted_images_index_file"]
                ),
                "raw_mapping_evidence": mapping_evidence,
                "conversion_run_evidence": conversion_evidence,
                "layout_count": len(layouts),
                "document_status_counts": dict(sorted(status_counts.items())),
                "documents": [documents[key] for key in sorted(documents)],
                "existing_pdf_media_count": len(repaired_items),
                "source_url_repair_count": source_url_repairs,
                "recovered_pdf_media_count": len(new_items),
                "completed_pdf_media_count": len(completed_pdf_items),
                "extraction_metrics": dict(sorted(extraction_metrics.items())),
                "gates": {**gates, "passed": gate_passed},
            }
            report_path = ctx.stage_work_dir / "pdf_media_completion_report.json"
            atomic_write_json(report_path, report)

            outputs = {
                "page_media_file": page_media_file,
                "page_images_file": page_images_file,
                "page_videos_file": str(ctx.previous_outputs.get("page_videos_file") or ""),
                "extracted_images_index_file": str(document_path),
                "extracted_images_count": len(completed_pdf_items),
                "media_manifest_file": str(manifest_path),
                "pdf_media_completion_report_file": str(report_path),
                "pdf_media_completion_passed": gate_passed,
            }
            current_extracted = ctx.find_artifacts(artifact_type="extracted_image")
            artifact_by_path = {
                str(Path(record.local_path).resolve()): record
                for record in current_extracted
                if record.local_path and Path(record.local_path).is_file()
            }
            richest_by_hash: Dict[str, Dict[str, Any]] = {}
            for item in completed_pdf_items:
                content_hash = str(item.get("content_hash") or "")
                existing = richest_by_hash.get(content_hash)
                if existing is None or _semantic_richness(item) > _semantic_richness(existing):
                    richest_by_hash[content_hash] = dict(item)
            artifacts: List[Any] = [
                ctx.make_artifact(
                    manifest_path,
                    artifact_type="media_manifest",
                    role="pdf_completed_multimodal_media",
                    metadata={
                        "pdf_media_count": len(completed_pdf_items),
                        "recovered_count": len(new_items),
                    },
                ),
                ctx.make_artifact(
                    report_path,
                    artifact_type="pdf_media_completion_report",
                    role="quality_report",
                    metadata={"passed": gate_passed, "recovered_count": len(new_items)},
                ),
            ]
            for content_hash, item in sorted(richest_by_hash.items()):
                local_path = str(item.get("local_path") or "")
                prior = artifact_by_path.get(str(Path(local_path).resolve())) if local_path else None
                artifacts.append(
                    ctx.make_artifact(
                        local_path,
                        artifact_type="extracted_image",
                        role="document_figure",
                        metadata=item,
                        source_artifact_ids=[prior.artifact_id] if prior else [],
                    )
                )
            if not gate_passed:
                return StageResult.failure(
                    "PDF media completion failed one or more production gates",
                    outputs=outputs,
                    metrics={
                        "existing_pdf_media": len(repaired_items),
                        "recovered_pdf_media": len(new_items),
                        "failed_documents": failed_documents,
                    },
                    artifacts=artifacts,
                )
            return StageResult.success(
                outputs=outputs,
                metrics={
                    "discovered_pdf_documents": len(documents),
                    "existing_pdf_media": len(repaired_items),
                    "source_url_repairs": source_url_repairs,
                    "recovered_pdf_media": len(new_items),
                    "completed_pdf_media": len(completed_pdf_items),
                    "failed_documents": failed_documents,
                },
                artifacts=artifacts,
                removed_artifact_ids=[record.artifact_id for record in current_extracted],
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            return StageResult.failure(f"PDF media completion failed: {exc}")
