"""
MBZUAI legacy vector-store formatter.

Builds the two payloads consumed by the current chatbot backend:
- summary index records for metadata-oriented retrieval
- text index records for chunk/detail retrieval

The output metadata intentionally preserves the backend contract:
page_source, source, context, document_summary, key_facts, keywords,
page_id, and chunk_id.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple
from urllib.parse import urlparse

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.chunking import load_chunk_index
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import compact_media_for_metadata, dedupe_media_items, media_items_by_type
from pipeline.core.mbzuai_indexing import (
    compact_citation_anchor,
    is_official_source_url,
    locale_family_url,
    normalize_url,
)
from pipeline.core.registry import register_stage
from pipeline.stages.formatters.pinecone_formatter import (
    _coerce_str_list,
    _extract_images_from_markdown,
    _stable_document_id,
)

logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _truncate(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip() or text[:max_chars].strip()
    return f"{trimmed}..."


def _stable_id(*parts: Any, prefix: str = "record") -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not raw:
        raw = prefix
    return f"{prefix}:{sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def _json_metadata(value: Any, max_chars: int = 12000) -> str:
    if value in (None, "", [], {}):
        return ""
    try:
        dumped = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except TypeError:
        dumped = json.dumps(str(value), ensure_ascii=False)
    return dumped[:max_chars]


def _summary_path_key(summary: Dict[str, Any]) -> str:
    source_file = str(summary.get("source_original_file") or "")
    if not source_file:
        return ""
    try:
        return str(Path(source_file).resolve())
    except Exception:
        return source_file


def _first_non_empty(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return text


def _same_registered_domain(left: str, right: str) -> bool:
    left_host = urlparse(left).netloc.lower()
    right_host = urlparse(right).netloc.lower()
    if not left_host or not right_host:
        return False
    left_parts = left_host.split(".")
    right_parts = right_host.split(".")
    return ".".join(left_parts[-2:]) == ".".join(right_parts[-2:])


def _effective_citation_source_url(source_url: str, page_meta: Dict[str, Any], document_type: str) -> str:
    source_url = _clean_text(source_url)
    canonical_url = _clean_text(page_meta.get("canonical_url")) if isinstance(page_meta, dict) else ""
    status_code = page_meta.get("status_code") if isinstance(page_meta, dict) else None
    try:
        status_int = int(status_code)
    except (TypeError, ValueError):
        status_int = 0
    if (
        source_url
        and canonical_url
        and 300 <= status_int < 400
        and _same_registered_domain(source_url, canonical_url)
        and _clean_text(document_type).lower() != "pdf"
    ):
        return canonical_url
    return source_url


def _load_download_url_by_file(ctx: StageContext) -> Dict[str, str]:
    mapping_file = ctx.work_dir / "mappings.json"
    payload = load_json_safe(mapping_file, {}) if mapping_file.exists() else {}
    if not isinstance(payload, dict):
        return {}
    by_file: Dict[str, str] = {}
    for source_url, local_path in payload.items():
        source = _clean_text(source_url)
        key = _path_key(local_path)
        if source and key:
            by_file[key] = source
    return by_file


_INTENT_MARKERS: Dict[str, Tuple[str, ...]] = {
    "admissions": ("admission", "admissions", "apply", "application", "deadline", "eligibility", "requirement", "requirements", "screening"),
    "programs": ("program", "programs", "degree", "degrees", "msc", "m.sc", "phd", "ph.d", "undergraduate", "specialization", "curriculum"),
    "scholarships": ("scholarship", "scholarships", "tuition", "fees", "funding", "stipend"),
    "faculty": ("faculty", "professor", "supervisor", "advisor", "profile"),
    "research": ("research", "lab", "labs", "publication", "project"),
    "campus": ("campus", "housing", "accommodation", "parking", "facility", "facilities", "shuttle", "gym", "pool", "library"),
    "contact": ("contact", "email", "phone", "telephone", "office"),
    "events": ("event", "events", "news", "workshop", "seminar", "conference"),
    "leadership": ("president", "leadership", "governance", "board", "trustee", "khaldoon", "eric xing"),
}


def _infer_authority(source_url: str, document_title: str, document_type: str) -> Tuple[str, float]:
    source = str(source_url or "").lower()
    title = _clean_text(document_title).lower()
    doc_type = _clean_text(document_type).lower()
    blob = f"{source} {title} {doc_type}"
    if "/news/" in source or "/the-node/" in source or "/event/" in source:
        return "time_bound_content", 0.36
    if source.endswith(".pdf") or "pdf" in doc_type:
        if any(marker in blob for marker in ("catalog", "catalogue", "handbook", "manual", "brochure")):
            return "official_pdf", 0.68
        return "official_document", 0.58
    if any(marker in blob for marker in ("admission", "apply", "requirement", "eligibility")):
        return "official_admissions", 0.95
    if any(marker in blob for marker in ("program", "degree", "msc", "m.sc", "phd", "ph.d", "undergraduate")):
        return "official_program", 0.92
    if any(marker in blob for marker in ("campus", "facilit", "student-resources", "housing", "accommodation")):
        return "official_student_life", 0.86
    if any(marker in blob for marker in ("president", "leadership", "governance", "board", "trustee")):
        return "official_leadership", 0.9
    if any(marker in blob for marker in ("contact", "faq", "fast-facts", "about")):
        return "official_reference", 0.82
    if "mbzuai.ac.ae" in source:
        return "official_webpage", 0.74
    return "external_or_unknown", 0.18


def _infer_intent_tags(source_url: str, document_title: str, document_summary: str, headings: Dict[str, Any]) -> List[str]:
    heading_values: List[str] = []
    for key in ("h1", "h2", "h3"):
        values = headings.get(key) if isinstance(headings, dict) else []
        heading_values.extend(_coerce_str_list(values if isinstance(values, list) else []))
    blob = " ".join(
        [
            str(source_url or ""),
            str(document_title or ""),
            str(document_summary or "")[:1200],
            " ".join(heading_values[:20]),
        ]
    ).lower()
    tags: Set[str] = set()
    for intent, markers in _INTENT_MARKERS.items():
        if any(marker in blob for marker in markers):
            tags.add(intent)
    return sorted(tags)


def _build_summary_text(
    *,
    title: str,
    source_url: str,
    document_type: str,
    document_date: str,
    summary: str,
    key_facts: Iterable[Any],
    keywords: Iterable[Any],
    headings: Dict[str, Any],
) -> str:
    lines = []
    if title:
        lines.append(f"TITLE: {title}")
    if source_url:
        lines.append(f"SOURCE_URL: {source_url}")
    if document_type:
        lines.append(f"TYPE: {document_type}")
    if document_date:
        lines.append(f"DATE: {document_date}")
    if summary:
        lines.extend(["", "SUMMARY:", summary])
    facts = _coerce_str_list(list(key_facts or []))
    if facts:
        lines.extend(["", "KEY FACTS:"])
        lines.extend(f"- {fact}" for fact in facts[:20])
    keyword_list = _coerce_str_list(list(keywords or []))
    if keyword_list:
        lines.extend(["", f"KEYWORDS: {', '.join(keyword_list[:30])}"])
    heading_values = []
    for key in ("h1", "h2", "h3"):
        values = headings.get(key) if isinstance(headings, dict) else []
        heading_values.extend(_coerce_str_list(values if isinstance(values, list) else []))
    if heading_values:
        lines.extend(["", "PAGE HEADINGS:"])
        lines.extend(f"- {heading}" for heading in heading_values[:30])
    return "\n".join(lines).strip()


def _build_text_embedding_text(
    *,
    title: str,
    source_url: str,
    document_type: str,
    document_summary: str,
    section_path: Iterable[Any],
    chunk_text: str,
) -> str:
    lines = []
    if title:
        lines.append(f"TITLE: {title}")
    if source_url:
        lines.append(f"SOURCE_URL: {source_url}")
    if document_type:
        lines.append(f"TYPE: {document_type}")
    sections = _coerce_str_list(list(section_path or []))
    if sections:
        lines.append(f"SECTION: {' > '.join(sections)}")
    if document_summary:
        lines.extend(["", "DOCUMENT SUMMARY:", _truncate(document_summary, 900)])
    lines.extend(["", "CONTENT:", chunk_text])
    return "\n".join(lines).strip()


def _load_summary_map(ctx: StageContext) -> Dict[str, Dict[str, Any]]:
    summary_artifacts = ctx.find_artifacts(artifact_type="summary")
    summaries_dir = ctx.previous_outputs.get("summaries_dir")
    summary_files: List[Path] = []
    if summary_artifacts:
        summary_files = [
            Path(record.local_path)
            for record in summary_artifacts
            if record.local_path and Path(record.local_path).is_file()
        ]
    elif summaries_dir:
        summary_files = list(Path(summaries_dir).glob("*.summary.json"))

    summary_by_path: Dict[str, Dict[str, Any]] = {}
    for path in summary_files:
        payload = load_json_safe(path, {}) or {}
        if not isinstance(payload, dict):
            continue
        key = _summary_path_key(payload)
        if key:
            summary_by_path[key] = payload
    return summary_by_path


def _load_chunk_records(ctx: StageContext) -> List[Dict[str, Any]]:
    chunk_artifacts = ctx.find_artifacts(artifact_type="chunk_index")
    if chunk_artifacts and chunk_artifacts[-1].local_path:
        payload = load_chunk_index(chunk_artifacts[-1].local_path)
        return list(payload.get("chunks") or [])
    chunks_file = ctx.previous_outputs.get("chunks_file")
    if chunks_file:
        payload = load_chunk_index(chunks_file)
        return list(payload.get("chunks") or [])
    return []


def _load_page_media(ctx: StageContext) -> Dict[str, List[Dict[str, Any]]]:
    page_media_file = ctx.previous_outputs.get("page_media_file")
    if page_media_file:
        payload = load_json_safe(page_media_file, {}) or {}
        if isinstance(payload, dict):
            return payload
    return {}


def _load_page_metadata(ctx: StageContext) -> Dict[str, Dict[str, Any]]:
    page_metadata_file = ctx.previous_outputs.get("page_metadata_file")
    if page_metadata_file:
        payload = load_json_safe(page_metadata_file, {}) or {}
        if isinstance(payload, dict):
            return payload
    return {}


def _allowed_source_suffixes(ctx: StageContext) -> List[str]:
    suffixes = ctx.formatter_config.get("official_source_host_suffixes")
    if not suffixes:
        suffixes = ["mbzuai.ac.ae", "staticcdn.mbzuai.ac.ae"]
    return [str(item).strip().lower().lstrip(".") for item in suffixes if str(item or "").strip()]


@register_stage
class MBZLegacyVectorStoreFormatter(FormatterStage):
    name = "mbzuai_legacy_vectorstores"
    description = "Builds chatbot-compatible MBZUAI summary/text vector-store payloads."

    async def execute(self, ctx: StageContext) -> StageResult:
        chunk_records = _load_chunk_records(ctx)
        if not chunk_records:
            return StageResult.failure("No chunk records available for legacy MBZUAI vector-store formatting")

        summary_by_path = _load_summary_map(ctx)
        page_media = _load_page_media(ctx)
        page_metadata = _load_page_metadata(ctx)
        download_url_by_file = _load_download_url_by_file(ctx)
        allowed_source_suffixes = _allowed_source_suffixes(ctx)
        exclude_non_official = bool(ctx.formatter_config.get("exclude_non_official_sources", True))
        skipped_non_official = 0
        skipped_non_indexable = 0

        max_media_per_doc = int(ctx.formatter_config.get("max_media_per_doc", 8))
        max_images_per_doc = int(ctx.formatter_config.get("max_images_per_doc", 5))
        max_videos_per_doc = int(ctx.formatter_config.get("max_videos_per_doc", 3))
        summary_context_max_chars = int(ctx.formatter_config.get("legacy_summary_context_max_chars", 10000))
        text_context_max_chars = int(ctx.formatter_config.get("legacy_text_context_max_chars", 12000))

        chunks_by_document: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in chunk_records:
            document_key = str(
                chunk.get("source_markdown_path")
                or chunk.get("source_url")
                or chunk.get("document_id")
                or chunk.get("document_title")
                or "document"
            )
            chunks_by_document[document_key].append(chunk)

        summary_docs: List[Dict[str, Any]] = []
        text_docs: List[Dict[str, Any]] = []

        for document_key, chunks in sorted(chunks_by_document.items()):
            chunks = sorted(chunks, key=lambda item: int(item.get("chunk_index") or 0))
            first_chunk = chunks[0]
            resolved_md_path = str(first_chunk.get("source_markdown_path") or "")
            resolved_key = ""
            if resolved_md_path:
                try:
                    resolved_key = str(Path(resolved_md_path).resolve())
                except Exception:
                    resolved_key = resolved_md_path

            summary_payload = dict(summary_by_path.get(resolved_key) or {})
            full_text = "\n\n".join(_clean_text(chunk.get("text")) for chunk in chunks if _clean_text(chunk.get("text")))
            source_file_url = download_url_by_file.get(_path_key(first_chunk.get("source_file")))
            source_url = _first_non_empty(
                first_chunk.get("source_url"),
                summary_payload.get("page_source"),
                summary_payload.get("source_url"),
                summary_payload.get("source"),
                summary_payload.get("document_source"),
                source_file_url,
            )
            document_title = _first_non_empty(
                summary_payload.get("document_title"),
                first_chunk.get("document_title"),
                Path(resolved_md_path).stem if resolved_md_path else "",
                source_url,
            )
            document_type = _first_non_empty(summary_payload.get("document_type"), first_chunk.get("document_type"), "webpage")
            document_date = _first_non_empty(summary_payload.get("document_date"), first_chunk.get("document_date"))
            document_summary = _first_non_empty(
                summary_payload.get("detailed_summary"),
                summary_payload.get("document_summary"),
                _truncate(full_text, 1800),
            )
            key_facts = summary_payload.get("key_facts") if isinstance(summary_payload.get("key_facts"), list) else []
            keywords = summary_payload.get("keywords") if isinstance(summary_payload.get("keywords"), list) else []
            entities = summary_payload.get("entities") if isinstance(summary_payload.get("entities"), dict) else {}
            document_id = _first_non_empty(
                first_chunk.get("document_id"),
                _stable_document_id(source_url, resolved_md_path, document_title),
            )
            normalized_source_url = normalize_url(source_url) or source_url
            page_meta = page_metadata.get(normalized_source_url) or page_metadata.get(source_url) or {}
            if page_meta and not bool(page_meta.get("indexable", True)):
                skipped_non_indexable += 1
                continue
            if exclude_non_official and not is_official_source_url(source_url, allowed_source_suffixes):
                skipped_non_official += 1
                continue
            media_lookup_url = source_url
            source_url = _effective_citation_source_url(source_url, page_meta, document_type)
            normalized_source_url = normalize_url(source_url) or source_url
            if exclude_non_official and not is_official_source_url(source_url, allowed_source_suffixes):
                skipped_non_official += 1
                continue
            headings = page_meta.get("headings") if isinstance(page_meta.get("headings"), dict) else {}
            authority_class, authority_score = _infer_authority(source_url, document_title, document_type)
            intent_tags = _infer_intent_tags(source_url, document_title, document_summary, headings)
            canonical_url = _first_non_empty(page_meta.get("canonical_url"), normalized_source_url, source_url)
            canonical_family_url = _first_non_empty(page_meta.get("canonical_family_url"), locale_family_url(canonical_url), canonical_url)
            language = _first_non_empty(page_meta.get("language"))
            content_hash = _first_non_empty(
                page_meta.get("content_hash"),
                sha1(full_text.encode("utf-8", errors="ignore")).hexdigest(),
            )

            markdown_images: List[Dict[str, Any]] = []
            if resolved_md_path and Path(resolved_md_path).exists():
                markdown_text = Path(resolved_md_path).read_text(encoding="utf-8", errors="replace")
                markdown_images = [
                    {
                        "type": "image",
                        "alt": image.get("alt", ""),
                        "title": image.get("alt", ""),
                        "url": image.get("url", ""),
                        "source_type": "markdown",
                    }
                    for image in _extract_images_from_markdown(markdown_text)
                ]

            doc_media = dedupe_media_items([*markdown_images, *(page_media.get(media_lookup_url) or page_media.get(source_url) or [])])
            doc_images = media_items_by_type(doc_media, "image")[:max_images_per_doc]
            doc_videos = media_items_by_type(doc_media, "video")[:max_videos_per_doc]
            doc_media = dedupe_media_items([*doc_images, *doc_videos])[:max_media_per_doc]

            page_metadata_json = _json_metadata(page_meta)
            media_metadata = compact_media_for_metadata(doc_media, max_items=max_media_per_doc, include_local_path=False) if doc_media else []
            image_metadata = compact_media_for_metadata(doc_images, max_items=max_images_per_doc, include_local_path=False) if doc_images else []
            video_metadata = compact_media_for_metadata(doc_videos, max_items=max_videos_per_doc, include_local_path=False) if doc_videos else []

            summary_text = _build_summary_text(
                title=document_title,
                source_url=source_url,
                document_type=document_type,
                document_date=document_date,
                summary=document_summary,
                key_facts=key_facts,
                keywords=keywords,
                headings=headings,
            )
            summary_context = _truncate(summary_text, summary_context_max_chars)
            summary_id = _stable_id("summary", document_id, source_url, prefix="summary")
            summary_citation_anchor = compact_citation_anchor(
                source_url=source_url,
                metadata={**page_meta, "title": document_title or page_meta.get("title")},
                section_path=[],
                chunk_index=0,
            )
            base_metadata = {
                "page_source": source_url,
                "source": source_url,
                "document_source": source_url,
                "canonical_url": canonical_url,
                "canonical_family_url": canonical_family_url,
                "normalized_url": normalized_source_url,
                "normalized_path": page_meta.get("normalized_path") or "",
                "language": language,
                "locale_variant_urls": page_meta.get("locale_variant_urls") or [],
                "page_type": page_meta.get("page_type") or "",
                "indexable": bool(page_meta.get("indexable", True)),
                "index_exclusion_reason": page_meta.get("index_exclusion_reason") or "",
                "content_hash": content_hash,
                "page_title": summary_citation_anchor["page_title"],
                "section_title": "",
                "heading_path": [],
                "breadcrumb": summary_citation_anchor["breadcrumb"],
                "citation_anchor": summary_citation_anchor,
                "document_title": document_title,
                "document_type": document_type,
                "document_date": document_date,
                "document_summary": document_summary,
                "key_facts": key_facts,
                "keywords": keywords,
                "entities": entities,
                "document_id": document_id,
                "source_file": resolved_md_path,
                "document_path": resolved_md_path,
                "page_metadata": page_metadata_json,
                "authority_class": authority_class,
                "authority_score": round(authority_score, 4),
                "intent_tags": intent_tags,
                "media": media_metadata,
                "images": image_metadata,
                "videos": video_metadata,
            }
            summary_docs.append(
                {
                    "id": summary_id,
                    "text": summary_text,
                    "metadata": {
                        **base_metadata,
                        "page_id": summary_id,
                        "chunk_id": summary_id,
                        "context": summary_context,
                        "index_role": "summary",
                        "chunk_index": 0,
                        "chunk_count": 1,
                    },
                }
            )

            chunk_count = len(chunks)
            for chunk in chunks:
                chunk_text = _clean_text(chunk.get("text"))
                if not chunk_text:
                    continue
                chunk_index = int(chunk.get("chunk_index") or 0)
                chunk_id = _first_non_empty(
                    chunk.get("chunk_id"),
                    f"{document_id}::chunk::{chunk_index + 1:03d}",
                )
                section_path = chunk.get("section_path") if isinstance(chunk.get("section_path"), list) else []
                citation_anchor = compact_citation_anchor(
                    source_url=source_url,
                    metadata={**page_meta, "title": document_title or page_meta.get("title")},
                    section_path=section_path,
                    chunk_index=chunk_index,
                )
                text_embedding_text = _build_text_embedding_text(
                    title=document_title,
                    source_url=source_url,
                    document_type=document_type,
                    document_summary=document_summary,
                    section_path=section_path,
                    chunk_text=chunk_text,
                )
                text_docs.append(
                    {
                        "id": chunk_id,
                        "text": text_embedding_text,
                        "metadata": {
                            **base_metadata,
                            "page_id": chunk_id,
                            "chunk_id": chunk_id,
                            "context": _truncate(chunk_text, text_context_max_chars),
                            "index_role": "text",
                            "chunk_index": chunk_index,
                            "chunk_count": int(chunk.get("chunk_count") or chunk_count),
                            "chunk_strategy": chunk.get("strategy"),
                            "section_path": section_path,
                            "page_title": citation_anchor["page_title"],
                            "section_title": citation_anchor["section_title"],
                            "heading_path": citation_anchor["heading_path"],
                            "breadcrumb": citation_anchor["breadcrumb"],
                            "citation_anchor": citation_anchor,
                            "page_numbers": chunk.get("page_numbers") or [],
                            "element_types": chunk.get("element_types") or [],
                            "content_sha1": sha1(chunk_text.encode("utf-8", errors="ignore")).hexdigest()[:20],
                        },
                    }
                )

        if not summary_docs or not text_docs:
            return StageResult.failure("Legacy formatter produced no summary or text vector records")

        summary_file = ctx.stage_work_dir / "legacy_summary_vectors.json"
        text_file = ctx.stage_work_dir / "legacy_text_vectors.json"
        manifest_file = ctx.stage_work_dir / "legacy_vectorstore_manifest.json"
        manifest = {
            "schema_version": 1,
            "vectorstore_contract": "mbzuai_chatbot_legacy_v1",
            "summary_records": len(summary_docs),
            "text_records": len(text_docs),
            "metadata_fields": [
                "page_source",
                "source",
                "context",
                "document_summary",
                "key_facts",
                "keywords",
                "authority_class",
                "authority_score",
                "intent_tags",
                "canonical_url",
                "canonical_family_url",
                "page_title",
                "section_title",
                "breadcrumb",
                "page_id",
                "chunk_id",
            ],
            "page_metadata_file": ctx.previous_outputs.get("page_metadata_file", ""),
            "page_link_graph_file": ctx.previous_outputs.get("page_link_graph_file", ""),
            "source_scope": {
                "exclude_non_official_sources": exclude_non_official,
                "official_source_host_suffixes": allowed_source_suffixes,
                "skipped_non_official_documents": skipped_non_official,
                "skipped_non_indexable_documents": skipped_non_indexable,
            },
        }
        atomic_write_json(summary_file, summary_docs)
        atomic_write_json(text_file, text_docs)
        atomic_write_json(manifest_file, manifest)

        logger.info(
            "MBZUAI legacy formatter: %d summary records, %d text records (skipped_non_official=%d skipped_non_indexable=%d)",
            len(summary_docs),
            len(text_docs),
            skipped_non_official,
            skipped_non_indexable,
        )

        return StageResult.success(
            outputs={
                "legacy_summary_formatted_file": str(summary_file),
                "legacy_text_formatted_file": str(text_file),
                "legacy_vectorstore_manifest_file": str(manifest_file),
                "legacy_summary_count": len(summary_docs),
                "legacy_text_count": len(text_docs),
            },
            metrics={
                "summary_records": len(summary_docs),
                "text_records": len(text_docs),
                "source_documents": len(chunks_by_document),
                "skipped_non_official_documents": skipped_non_official,
                "skipped_non_indexable_documents": skipped_non_indexable,
            },
            artifacts=[
                ctx.make_artifact(
                    summary_file,
                    artifact_type="legacy_vectorstore_payload",
                    role="summary_index_payload",
                    metadata={"records": len(summary_docs), "index_role": "summary"},
                ),
                ctx.make_artifact(
                    text_file,
                    artifact_type="legacy_vectorstore_payload",
                    role="text_index_payload",
                    metadata={"records": len(text_docs), "index_role": "text"},
                ),
                ctx.make_artifact(
                    manifest_file,
                    artifact_type="legacy_vectorstore_manifest",
                    role="vectorstore_contract",
                    metadata={"summary_records": len(summary_docs), "text_records": len(text_docs)},
                ),
            ],
        )
