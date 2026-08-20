"""
Gemini-oriented retrieval formatter.

Builds dense embedding corpora and a local retrieval bundle that supports:
- chunk-first retrieval for precision
- parent section/page expansion for broader questions
- media-aware retrieval with separate image/video records
- lexical sidecar retrieval using the same chunk/parent graph
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple
from urllib.parse import unquote, urlparse

from pipeline.core.answer_records import derive_answer_records_from_bundle
from pipeline.core.artifact_contracts import ArtifactContract, resolve_artifact_path
from pipeline.core.assertions import (
    build_answer_records_from_assertions,
    build_assertion_embedding_records,
    build_entity_records_from_assertions,
    merge_answer_records,
    merge_entity_records,
)
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.chunking import load_chunk_index
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import (
    build_media_embedding_text,
    compact_media_for_metadata,
    dedupe_media_items,
    load_media_manifest_items,
    media_items_by_type,
    normalize_media_item,
)
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

_LOW_SIGNAL_MEDIA_PHRASES = {
    "i'm sorry, but i cannot provide a description",
    "i cannot provide a description or answer",
    "doesn't contain any text or information",
    "does not contain any text or information",
}


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not raw:
        raw = "record"
    return sha1(raw.encode("utf-8")).hexdigest()[:24]


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _unique_clean_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values or []:
        text = _clean_text(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _assertions_from_promoted_graph(graph_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Convert promoted KG assertion nodes into the assertion sidecar shape."""

    nodes = [node for node in graph_payload.get("nodes") or [] if isinstance(node, Mapping)]
    edges = [edge for edge in graph_payload.get("edges") or [] if isinstance(edge, Mapping)]
    entities_by_id: Dict[str, Dict[str, Any]] = {}
    for node in nodes:
        if str(node.get("node_type") or "") != "entity":
            continue
        node_id = _clean_text(node.get("id"))
        if not node_id:
            continue
        properties = node.get("properties") if isinstance(node.get("properties"), Mapping) else {}
        entities_by_id[node_id] = {
            "id": node_id,
            "canonical_name": _clean_text(properties.get("canonical_name") or node.get("label")),
            "entity_type": _clean_text(properties.get("entity_type") or "other"),
            "aliases": _unique_clean_strings(properties.get("aliases") or []),
        }

    span_ids_by_assertion: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        if str(edge.get("edge_type") or "") != "ASSERTION_SUPPORTED_BY_SPAN":
            continue
        assertion_id = _clean_text(edge.get("source_id"))
        span_id = _clean_text(edge.get("target_id"))
        if assertion_id and span_id:
            span_ids_by_assertion[assertion_id].append(span_id)

    assertions: List[Dict[str, Any]] = []
    for node in nodes:
        if str(node.get("node_type") or "") != "relation_assertion":
            continue
        assertion_id = _clean_text(node.get("id"))
        properties = node.get("properties") if isinstance(node.get("properties"), Mapping) else {}
        validity_status = _clean_text(properties.get("validity_status") or "active").lower()
        if validity_status not in {"active", "valid"}:
            continue
        subject_id = _clean_text(properties.get("subject_entity_id"))
        object_id = _clean_text(properties.get("object_entity_id"))
        subject_entity = entities_by_id.get(subject_id, {})
        object_entity = entities_by_id.get(object_id, {})
        subject_name = _clean_text(properties.get("subject_name") or subject_entity.get("canonical_name"))
        object_name = _clean_text(
            properties.get("object_value")
            or properties.get("object_name")
            or object_entity.get("canonical_name")
            or properties.get("canonical_object")
        )
        predicate = _clean_text(properties.get("relation_type") or properties.get("canonical_predicate") or node.get("label"))
        if not assertion_id or not subject_name or not object_name or not predicate:
            continue
        source_span_ids = _unique_clean_strings(
            [
                *(properties.get("source_span_ids") or []),
                *(span_ids_by_assertion.get(assertion_id) or []),
            ]
        )
        assertion = {
            "id": assertion_id,
            "subject_name": subject_name,
            "subject_type": _clean_text(properties.get("subject_type") or subject_entity.get("entity_type") or "organization"),
            "subject_entity_id": subject_id,
            "predicate": predicate,
            "relation_type": predicate,
            "answer_type": predicate,
            "answer_subtype": _clean_text(properties.get("answer_subtype") or predicate),
            "object_name": object_name,
            "object_value": object_name,
            "object_type": _clean_text(properties.get("object_type") or object_entity.get("entity_type") or "other"),
            "object_entity_id": object_id,
            "qualifiers": _unique_clean_strings(properties.get("qualifiers") or []),
            "support_span": _clean_text(properties.get("evidence") or properties.get("text")),
            "evidence": _clean_text(properties.get("evidence") or properties.get("text")),
            "confidence": properties.get("confidence"),
            "validator_confidence": properties.get("confidence"),
            "validator_decision": "supported",
            "authority_class": _clean_text(properties.get("authority_class")),
            "authority_score": properties.get("authority_score"),
            "freshness_score": properties.get("freshness_score"),
            "source_doc_id": _clean_text(properties.get("source_id")),
            "source_chunk_ids": _unique_clean_strings(properties.get("source_chunk_ids") or []),
            "source_span_ids": source_span_ids,
            "linked_span_ids": source_span_ids,
            "source_parent_ids": _unique_clean_strings(properties.get("source_parent_ids") or []),
            "source_fact_ids": _unique_clean_strings(properties.get("source_fact_ids") or []),
            "source_url": _clean_text(properties.get("source_url")),
            "source_markdown_path": _clean_text(properties.get("source_markdown_path")),
            "document_title": _clean_text(properties.get("document_title")),
            "source_last_seen": _clean_text(properties.get("source_last_seen")),
            "canonical_subject": _clean_text(properties.get("canonical_subject") or subject_id),
            "canonical_predicate": _clean_text(properties.get("canonical_predicate") or predicate),
            "canonical_object": _clean_text(properties.get("canonical_object") or object_name).casefold(),
            "validity_status": validity_status,
            "text": _clean_text(properties.get("text")),
            "source_id": _clean_text(properties.get("source_id")),
            "source_kind": _clean_text(properties.get("source_kind")),
        }
        assertions.append(assertion)

    return assertions


def _canonical_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        path = unquote(parsed.path or "").rstrip("/")
        if path in {"", "/"}:
            path = ""
        return f"{scheme}://{netloc}{path}".rstrip("/")
    except Exception:
        return raw.lower().rstrip("/")


def _language_normalized_url(value: Any) -> str:
    canonical = _canonical_url(value)
    if not canonical:
        return ""
    try:
        parsed = urlparse(canonical)
        parts = [part for part in (parsed.path or "").split("/") if part]
        if parts and parts[0].lower() in {"ar", "en"}:
            path = "/" + "/".join(parts[1:])
        else:
            path = parsed.path or ""
        return f"{parsed.scheme}://{parsed.netloc}{path}".rstrip("/")
    except Exception:
        return canonical


def _sentence_spans(text: Any, *, max_sentences: int, max_chars: int) -> List[str]:
    normalized = _clean_text(text)
    if not normalized:
        return []
    sentences = [
        _clean_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", normalized)
        if _clean_text(sentence)
    ]
    if not sentences:
        sentences = [normalized]
    spans: List[str] = []
    i = 0
    while i < len(sentences):
        window: List[str] = []
        while i < len(sentences) and len(window) < max(1, max_sentences):
            candidate = _clean_text(" ".join([*window, sentences[i]]))
            if window and len(candidate) > max_chars:
                break
            window.append(sentences[i])
            i += 1
            if len(candidate) >= max_chars * 0.65:
                break
        span = _truncate_chars(" ".join(window), max_chars)
        if span:
            spans.append(span)
        if not window:
            i += 1
    return spans


def _span_signal_score(text: str, *, document_title: str, section_path: Iterable[Any], heading: str) -> float:
    lower = text.lower()
    score = 0.0
    if re.search(r"\b\d{4}\b|\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", lower):
        score += 1.0
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4}\b", text):
        score += 0.8
    if any(token in lower for token in ("deadline", "apply", "application", "requirement", "eligibility", "tuition", "scholarship", "fee", "cost")):
        score += 1.0
    if any(token in lower for token in ("ph.d", "phd", "master", "msc", "program", "degree", "bachelor", "undergraduate")):
        score += 0.9
    if any(token in lower for token in ("faculty", "professor", "research interest", "award", "recognition", "publication")):
        score += 0.9
    if any(token in lower for token in ("email", "phone", "contact", "location", "address", "campus", "masdar")):
        score += 0.9
    if any(token in lower for token in ("policy", "procedure", "guideline", "visa", "housing", "accommodation")):
        score += 0.7
    heading_text = " ".join([document_title, heading, " ".join(str(value) for value in section_path or [])]).lower()
    if heading_text and any(token in heading_text for token in ("admission", "program", "faculty", "research", "scholarship", "deadline")):
        score += 0.4
    return score


def _classify_span_type(text: str, *, document_title: str, section_path: Iterable[Any], heading: str) -> str:
    lower = " ".join([text, document_title, heading, " ".join(str(value) for value in section_path or [])]).lower()
    if any(token in lower for token in ("deadline", "date", "apply by", "applications close")):
        return "deadline"
    if any(token in lower for token in ("email", "phone", "contact", "address")):
        return "contact"
    if any(token in lower for token in ("requirement", "eligibility", "required", "admission criteria")):
        return "requirement"
    if any(token in lower for token in ("ph.d", "phd", "master", "msc", "bachelor", "program", "degree")):
        return "program"
    if any(token in lower for token in ("professor", "faculty", "research interests", "biography")):
        return "faculty_profile"
    if any(token in lower for token in ("award", "recognition", "prize", "honor")):
        return "award"
    if any(token in lower for token in ("policy", "procedure", "guideline")):
        return "policy"
    if any(token in lower for token in ("fact", "founded", "established", "located", "offers")):
        return "fact"
    return "general"


def _build_span_embedding_text(span: Dict[str, Any]) -> str:
    lines = []
    if span.get("document_title"):
        lines.append(f"TITLE: {span['document_title']}")
    if span.get("breadcrumb"):
        lines.append(f"BREADCRUMB: {span['breadcrumb']}")
    if span.get("section_heading"):
        lines.append(f"SECTION: {span['section_heading']}")
    if span.get("canonical_url"):
        lines.append(f"CANONICAL_URL: {span['canonical_url']}")
    if span.get("span_type"):
        lines.append(f"SPAN_TYPE: {span['span_type']}")
    lines.extend(["", str(span.get("text") or "")])
    return "\n".join(part for part in lines if part is not None).strip()


def _build_span_sparse_text(span: Dict[str, Any], *, max_chars: int) -> str:
    lines = []
    if span.get("document_title"):
        lines.append(f"TITLE: {span['document_title']}")
    if span.get("breadcrumb"):
        lines.append(f"BREADCRUMB: {span['breadcrumb']}")
    if span.get("section_heading"):
        lines.append(f"SECTION: {span['section_heading']}")
    if span.get("canonical_url"):
        lines.append(f"URL: {span['canonical_url']}")
    if span.get("span_type"):
        lines.append(f"TYPE: {span['span_type']}")
    lines.extend(["", str(span.get("text") or "")])
    return _truncate_chars("\n".join(lines).strip(), max_chars=max_chars)


def _is_generic_figure_label(text: str) -> bool:
    normalized = _clean_text(text).lower()
    if not normalized.startswith("figure"):
        return False
    suffix = normalized[len("figure") :].strip(" .:#-")
    return not suffix or suffix.isdigit()


def _is_low_signal_media_item(item: Dict[str, Any]) -> bool:
    title = _clean_text(item.get("title") or "")
    caption = _clean_text(item.get("caption") or "")
    description = _clean_text(item.get("description") or "")
    context = _clean_text(item.get("context") or "")
    transcript = _clean_text(item.get("transcript") or "")
    text = _clean_text(item.get("text") or "")
    combined = " ".join(
        part for part in (title, caption, description, context, transcript, text) if part
    ).lower()
    if any(phrase in combined for phrase in _LOW_SIGNAL_MEDIA_PHRASES):
        return True
    if _is_generic_figure_label(title) and not any((caption, description, context, transcript)):
        return True
    return False


def _truncate_chars(value: Any, max_chars: int) -> str:
    text = _clean_text(value)
    max_chars = max(0, int(max_chars or 0))
    if not text or max_chars <= 0 or len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rstrip()
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0].rstrip()
    return f"{trimmed}..."


def _normalize_page_numbers(values: Iterable[Any]) -> List[int]:
    page_numbers = []
    seen = set()
    for value in values or []:
        try:
            page = int(value)
        except (TypeError, ValueError):
            continue
        if page in seen:
            continue
        seen.add(page)
        page_numbers.append(page)
    return sorted(page_numbers)


def _page_key(source_markdown_path: str, source_url: str, page_numbers: List[int]) -> str:
    if page_numbers:
        token = ",".join(str(page) for page in page_numbers)
    else:
        token = "root"
    return _stable_id("page", source_markdown_path or source_url, token)


def _section_key(page_id: str, section_path: List[str]) -> str:
    joined = " > ".join(section_path) if section_path else "__root__"
    return _stable_id("section", page_id, joined)


def _relative_or_absolute(path: str | Path) -> str:
    try:
        return str(Path(path).resolve())
    except Exception:
        return str(path)


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return text


def _document_stem_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stem = Path(text).stem
    stem = re.sub(r"\.pages_\d+_\d+$", "", stem)
    return stem.casefold()


def _load_download_source_lookup(ctx: StageContext) -> Dict[str, str]:
    mapping_file = ctx.work_dir / "mappings.json"
    payload = load_json_safe(mapping_file, {}) if mapping_file.exists() else {}
    if not isinstance(payload, dict):
        return {}
    lookup: Dict[str, str] = {}
    for source_url, local_path in payload.items():
        source = _clean_text(source_url)
        if not source:
            continue
        for key in (
            _path_key(local_path),
            _document_stem_key(local_path),
            Path(str(local_path or "")).name.casefold(),
        ):
            if key:
                lookup.setdefault(key, source)
    return lookup


def _resolve_source_url(
    source_url: Any,
    *,
    source_file: Any = "",
    source_markdown_path: Any = "",
    document_title: Any = "",
    download_source_lookup: Mapping[str, str] | None = None,
) -> str:
    existing = _clean_text(source_url)
    if existing:
        return existing
    lookup = download_source_lookup or {}
    for key in (
        _path_key(source_file),
        _path_key(source_markdown_path),
        _document_stem_key(source_file),
        _document_stem_key(source_markdown_path),
        _document_stem_key(document_title),
        Path(str(source_file or "")).name.casefold(),
        Path(str(source_markdown_path or "")).name.casefold(),
    ):
        if key and key in lookup:
            return lookup[key]
    return ""


def _compact_section_path(values: Iterable[Any]) -> List[str]:
    return [str(value).strip() for value in (values or []) if str(value).strip()]


def _build_chunk_embedding_text(chunk: Dict[str, Any], media_items: List[Dict[str, Any]]) -> str:
    lines = []
    if chunk.get("document_title"):
        lines.append(f"TITLE: {chunk['document_title']}")
    if chunk.get("document_type"):
        lines.append(f"TYPE: {chunk['document_type']}")
    if chunk.get("section_path"):
        lines.append(f"SECTION: {' > '.join(chunk['section_path'])}")
    if chunk.get("page_numbers"):
        lines.append(f"PAGES: {', '.join(str(v) for v in chunk['page_numbers'])}")
    if chunk.get("source_url"):
        lines.append(f"SOURCE_URL: {chunk['source_url']}")
    lines.append("")
    lines.append(chunk.get("text", ""))
    media_block = build_media_embedding_text(media_items, max_items=3)
    if media_block:
        lines.extend(["", "MEDIA:", media_block])
    return "\n".join(part for part in lines if part is not None).strip()


def _build_parent_embedding_text(parent: Dict[str, Any], child_chunks: List[Dict[str, Any]], media_items: List[Dict[str, Any]]) -> str:
    lines = []
    if parent.get("document_title"):
        lines.append(f"TITLE: {parent['document_title']}")
    lines.append(f"PARENT_TYPE: {parent.get('parent_type', 'section')}")
    if parent.get("section_path"):
        lines.append(f"SECTION: {' > '.join(parent['section_path'])}")
    if parent.get("page_numbers"):
        lines.append(f"PAGES: {', '.join(str(v) for v in parent['page_numbers'])}")
    if parent.get("source_url"):
        lines.append(f"SOURCE_URL: {parent['source_url']}")
    lines.append("")
    lines.append("\n\n".join(chunk.get("text", "") for chunk in child_chunks if chunk.get("text")))
    media_block = build_media_embedding_text(media_items, max_items=4)
    if media_block:
        lines.extend(["", "MEDIA:", media_block])
    return "\n".join(part for part in lines if part is not None).strip()


def _build_chunk_sparse_text(chunk: Dict[str, Any], *, max_chars: int) -> str:
    lines = []
    if chunk.get("document_title"):
        lines.append(f"TITLE: {chunk['document_title']}")
    if chunk.get("document_type"):
        lines.append(f"TYPE: {chunk['document_type']}")
    if chunk.get("section_path"):
        lines.append(f"SECTION: {' > '.join(chunk['section_path'])}")
    if chunk.get("heading"):
        lines.append(f"HEADING: {chunk['heading']}")
    body = _truncate_chars(chunk.get("text", ""), max_chars=max_chars)
    if body:
        lines.extend(["", body])
    return _truncate_chars("\n".join(lines).strip(), max_chars=max_chars)


def _build_parent_sparse_text(
    parent: Dict[str, Any],
    child_chunks: List[Dict[str, Any]],
    media_items: List[Dict[str, Any]],
    *,
    max_chars: int,
    max_headings: int,
    max_child_snippets: int,
) -> str:
    lines = []
    if parent.get("document_title"):
        lines.append(f"TITLE: {parent['document_title']}")
    lines.append(f"PARENT_TYPE: {parent.get('parent_type', 'section')}")
    if parent.get("section_path"):
        lines.append(f"SECTION: {' > '.join(parent['section_path'])}")
    if parent.get("page_numbers"):
        lines.append(f"PAGES: {', '.join(str(v) for v in parent['page_numbers'])}")
    if parent.get("source_url"):
        lines.append(f"SOURCE_URL: {parent['source_url']}")

    headings: List[str] = []
    seen_headings = set()
    for chunk in child_chunks:
        heading = _clean_text(chunk.get("heading"))
        if not heading or heading in seen_headings:
            continue
        seen_headings.add(heading)
        headings.append(heading)
        if len(headings) >= max(1, int(max_headings)):
            break
    if headings:
        lines.extend(["", "HEADINGS:"] + [f"- {heading}" for heading in headings])

    snippets: List[str] = []
    for chunk in child_chunks:
        snippet = _truncate_chars(chunk.get("text", ""), 320)
        if not snippet:
            continue
        snippets.append(snippet)
        if len(snippets) >= max(1, int(max_child_snippets)):
            break
    if snippets:
        lines.extend(["", "CONTENT:"] + snippets)

    media_titles: List[str] = []
    for item in media_items[:3]:
        label = _clean_text(item.get("title") or item.get("caption") or item.get("alt") or item.get("type"))
        if label:
            media_titles.append(f"- {label}")
    if media_titles:
        lines.extend(["", "MEDIA:"] + media_titles)

    return _truncate_chars("\n".join(lines).strip(), max_chars=max_chars)


def _build_media_embedding_input(item: Dict[str, Any], *, document_title: str = "", section_path: List[str] | None = None) -> str:
    lines = []
    media_type = item.get("type") or "image"
    title = item.get("title") or item.get("caption") or item.get("alt") or media_type.title()
    lines.append(f"{media_type.upper()}: {title}")
    if document_title:
        lines.append(f"DOCUMENT: {document_title}")
    if section_path:
        lines.append(f"SECTION: {' > '.join(section_path)}")
    for key in (
        "caption",
        "description",
        "context",
        "semantic_caption",
        "contextual_caption",
        "visual_description",
        "visible_text",
        "ocr_text",
        "transcript",
    ):
        value = _clean_text(item.get(key))
        if value and value != title:
            lines.append(f"{key.upper()}: {value}")
    semantic_tags = [
        _clean_text(value) for value in item.get("semantic_tags") or [] if _clean_text(value)
    ]
    if semantic_tags:
        lines.append(f"SEMANTIC_TAGS: {', '.join(semantic_tags[:16])}")
    if item.get("image_kind"):
        lines.append(f"IMAGE_KIND: {_clean_text(item.get('image_kind'))}")
    if item.get("semantic_relevance"):
        lines.append(f"SEMANTIC_RELEVANCE: {_clean_text(item.get('semantic_relevance'))}")
    if item.get("source_url"):
        lines.append(f"SOURCE_URL: {item['source_url']}")
    return "\n".join(lines).strip()


def _build_media_sparse_text(item: Dict[str, Any], *, max_chars: int) -> str:
    return _truncate_chars(item.get("text") or "", max_chars=max_chars)


def _tokenize_for_bm25(text: str) -> List[str]:
    return [token for token in _clean_text(text).lower().split() if token]


def _split_sentences(text: str) -> List[str]:
    normalized = _clean_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
    output: List[str] = []
    for part in parts:
        candidate = _clean_text(part)
        if candidate:
            output.append(candidate)
    return output


def _build_extractive_summary(parent: Dict[str, Any], child_chunks: List[Dict[str, Any]], *, max_chars: int) -> str:
    heading_lines: List[str] = []
    for chunk in child_chunks:
        heading = _clean_text(chunk.get("heading"))
        if heading and heading not in heading_lines:
            heading_lines.append(heading)
        if len(heading_lines) >= 5:
            break

    sentences: List[str] = []
    for chunk in child_chunks:
        for sentence in _split_sentences(str(chunk.get("text") or "")):
            if len(sentence.split()) < 6:
                continue
            sentences.append(sentence)
            if len(sentences) >= 6:
                break
        if len(sentences) >= 6:
            break

    lines = []
    if parent.get("document_title"):
        lines.append(f"TITLE: {parent['document_title']}")
    if parent.get("section_path"):
        lines.append(f"SECTION: {' > '.join(parent['section_path'])}")
    if heading_lines:
        lines.extend(["HEADINGS:", *[f"- {heading}" for heading in heading_lines]])
    if sentences:
        lines.extend(["SUMMARY:", " ".join(sentences)])
    return _truncate_chars("\n".join(lines).strip(), max_chars=max_chars)


def _build_summary_embedding_text(summary_record: Dict[str, Any]) -> str:
    lines = []
    if summary_record.get("document_title"):
        lines.append(f"TITLE: {summary_record['document_title']}")
    if summary_record.get("summary_type"):
        lines.append(f"SUMMARY_TYPE: {summary_record['summary_type']}")
    if summary_record.get("section_path"):
        lines.append(f"SECTION: {' > '.join(summary_record['section_path'])}")
    if summary_record.get("source_url"):
        lines.append(f"SOURCE_URL: {summary_record['source_url']}")
    lines.extend(["", summary_record.get("text") or ""])
    return "\n".join(part for part in lines if part is not None).strip()


def _normalize_fact_candidate(value: str, *, min_chars: int, max_chars: int) -> str:
    candidate = _clean_text(value)
    if not candidate:
        return ""
    if len(candidate) <= max_chars:
        return candidate if len(candidate) >= min_chars else ""

    sentences = _split_sentences(candidate)
    if sentences:
        for length in range(min(3, len(sentences)), 0, -1):
            merged = _clean_text(" ".join(sentences[:length]))
            if min_chars <= len(merged) <= max_chars:
                return merged
        first = _truncate_chars(sentences[0], max_chars=max_chars)
        return first if len(first) >= min_chars else ""

    truncated = _truncate_chars(candidate, max_chars=max_chars)
    return truncated if len(truncated) >= min_chars else ""


def _extract_fact_snippets(
    chunk: Dict[str, Any],
    *,
    min_chars: int = 40,
    max_chars: int = 360,
    max_snippets: int = 5,
) -> List[str]:
    text = str(chunk.get("text") or "")
    sentences = _split_sentences(text)
    lines = [_clean_text(part) for part in re.split(r"[\r\n]+", text) if _clean_text(part)]
    heading = _clean_text(chunk.get("heading"))
    snippets: List[str] = []

    def _append(value: str) -> None:
        candidate = _normalize_fact_candidate(value, min_chars=min_chars, max_chars=max_chars)
        if not candidate:
            return
        snippets.append(candidate)

    def _strip_bullet_prefix(value: str) -> str:
        return _clean_text(re.sub(r"^[\-\*\u2022]+\s*", "", value))

    for idx, line in enumerate(lines):
        stripped = _strip_bullet_prefix(line)
        if not stripped:
            continue
        if stripped.endswith("?"):
            answer_parts: List[str] = []
            for follower in lines[idx + 1 :]:
                follower = _strip_bullet_prefix(follower)
                if not follower:
                    continue
                if follower.endswith("?"):
                    break
                answer_parts.append(follower)
                if len(" ".join(answer_parts)) >= max_chars:
                    break
            if answer_parts:
                answer = _normalize_fact_candidate(" ".join(answer_parts), min_chars=min_chars, max_chars=max_chars)
                if answer:
                    _append(f"{stripped} {answer}")
        elif re.match(r"^[\-\*\u2022]+\s*", line):
            _append(stripped)
            hours_match = re.search(r"(working hours[^.]*\.[^.]*|working hours.*)$", stripped, flags=re.IGNORECASE)
            if hours_match:
                _append(hours_match.group(1))

    for sentence in sentences:
        _append(sentence)
    for idx in range(len(sentences) - 1):
        _append(f"{sentences[idx]} {sentences[idx + 1]}")

    if heading.endswith("?"):
        for sentence in sentences[:2]:
            _append(f"{heading} {sentence}")

    deduped: List[str] = []
    seen = set()
    for snippet in snippets:
        key = snippet.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(snippet)
        if len(deduped) >= max_snippets:
            break
    return deduped


def _build_fact_embedding_text(
    fact_text: str,
    *,
    document_title: str,
    section_path: List[str],
    page_numbers: List[int],
    source_url: str,
    heading: str,
) -> str:
    lines = []
    if document_title:
        lines.append(f"TITLE: {document_title}")
    if heading:
        lines.append(f"HEADING: {heading}")
    if section_path:
        lines.append(f"SECTION: {' > '.join(section_path)}")
    if page_numbers:
        lines.append(f"PAGES: {', '.join(str(v) for v in page_numbers)}")
    if source_url:
        lines.append(f"SOURCE_URL: {source_url}")
    lines.extend(["", f"FACT: {fact_text}"])
    return "\n".join(lines).strip()


def _render_pdf_page_visual(
    pdf_path: str | Path,
    *,
    page_number: int,
    output_dir: str | Path,
    max_side: int = 1400,
) -> str:
    try:
        import fitz
    except Exception:
        return ""

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        return ""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{pdf_path.stem}_page_{page_number:04d}.png"
    if output_path.exists():
        return str(output_path)

    try:
        with fitz.open(str(pdf_path)) as doc:
            page_index = page_number - 1 if 1 <= page_number <= len(doc) else page_number
            if page_index < 0 or page_index >= len(doc):
                return ""
            page = doc.load_page(page_index)
            rect = page.rect
            largest_side = max(float(rect.width), float(rect.height), 1.0)
            scale = min(2.0, max(1.0, float(max_side) / largest_side))
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            pix.save(str(output_path))
    except Exception:
        return ""
    return str(output_path)


@register_stage
class GeminiRetrievalFormatter(FormatterStage):
    name = "gemini_retrieval"
    description = "Builds chunk, parent, and media corpora for Gemini multimodal retrieval."

    async def execute(self, ctx: StageContext) -> StageResult:
        chunk_index_resolution = resolve_artifact_path(
            ctx,
            ArtifactContract(
                artifact_type="chunk_index",
                role="retrieval_chunks",
                legacy_output_key="chunks_file",
                label="chunk index",
            ),
        )
        chunk_index_payload: Dict[str, Any] = {}
        if chunk_index_resolution:
            chunk_index_payload = load_chunk_index(chunk_index_resolution.path)

        chunk_records = list(chunk_index_payload.get("chunks") or [])
        if not chunk_records:
            return StageResult.failure("No chunk_index available for retrieval formatting")

        markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
        markdown_by_path = {
            str(Path(record.local_path).resolve()): record
            for record in markdown_artifacts
            if record.local_path and Path(record.local_path).is_file()
        }

        extracted_image_artifacts = ctx.find_artifacts(artifact_type="extracted_image")
        document_media_items: List[Dict[str, Any]] = []
        if extracted_image_artifacts:
            for record in extracted_image_artifacts:
                payload = normalize_media_item(
                    {
                        **dict(record.metadata or {}),
                        "local_path": record.local_path or record.metadata.get("local_path", ""),
                        "url": record.metadata.get("url") or record.uri,
                        "asset_uri": record.metadata.get("asset_uri") or record.uri,
                    }
                )
                document_media_items.append(payload)
        else:
            idx_file = ctx.previous_outputs.get("extracted_images_index_file")
            if idx_file:
                document_media_items = load_media_manifest_items(load_json_safe(idx_file, []))

        page_media_file = ctx.previous_outputs.get("page_media_file") or str(ctx.work_dir / "page_media.json")
        all_page_media = load_json_safe(page_media_file, {}) or {}
        if not isinstance(all_page_media, dict):
            all_page_media = {}

        chunk_map: Dict[str, Dict[str, Any]] = {}
        chunk_ids_by_page: Dict[str, List[str]] = defaultdict(list)
        chunk_ids_by_section: Dict[str, List[str]] = defaultdict(list)
        chunk_ids_by_url: Dict[str, List[str]] = defaultdict(list)
        chunk_ids_by_markdown_path: Dict[str, List[str]] = defaultdict(list)
        chunk_ids_by_doc_page: Dict[Tuple[str, int], List[str]] = defaultdict(list)
        page_record_map: Dict[str, Dict[str, Any]] = {}
        section_record_map: Dict[str, Dict[str, Any]] = {}
        download_source_lookup = _load_download_source_lookup(ctx)

        for chunk in chunk_records:
            source_markdown_path = _relative_or_absolute(chunk.get("source_markdown_path") or "")
            source_url = _resolve_source_url(
                chunk.get("source_url"),
                source_file=chunk.get("source_file"),
                source_markdown_path=source_markdown_path,
                document_title=chunk.get("document_title"),
                download_source_lookup=download_source_lookup,
            )
            canonical_url = _canonical_url(source_url)
            language_normalized_url = _language_normalized_url(source_url)
            page_numbers = _normalize_page_numbers(chunk.get("page_numbers") or [])
            section_path = _compact_section_path(chunk.get("section_path") or [])
            page_id = _page_key(source_markdown_path, source_url, page_numbers)
            section_id = _section_key(page_id, section_path)
            chunk_id = str(chunk.get("chunk_id") or _stable_id("chunk", source_markdown_path, chunk.get("chunk_index")))

            record = {
                "id": chunk_id,
                "record_type": "chunk",
                "text": str(chunk.get("text") or ""),
                "document_id": str(chunk.get("document_id") or ""),
                "document_title": str(chunk.get("document_title") or ""),
                "document_type": str(chunk.get("document_type") or ""),
                "source_backend": str(chunk.get("source_backend") or ""),
                "source_file": str(chunk.get("source_file") or ""),
                "source_markdown_path": source_markdown_path,
                "source_url": source_url,
                "canonical_url": canonical_url,
                "language_normalized_url": language_normalized_url,
                "chunk_index": int(chunk.get("chunk_index") or 0),
                "chunk_count": int(chunk.get("chunk_count") or 1),
                "section_path": section_path,
                "page_numbers": page_numbers,
                "heading": str(chunk.get("heading") or ""),
                "page_key": page_id,
                "section_key": section_id,
                "neighbor_ids": [],
                "media_ids": [],
            }
            chunk_map[chunk_id] = record
            chunk_ids_by_page[page_id].append(chunk_id)
            chunk_ids_by_section[section_id].append(chunk_id)
            if source_url:
                chunk_ids_by_url[source_url].append(chunk_id)
            if source_markdown_path:
                chunk_ids_by_markdown_path[source_markdown_path].append(chunk_id)
            if source_markdown_path and page_numbers:
                for page_number in page_numbers:
                    chunk_ids_by_doc_page[(source_markdown_path, page_number)].append(chunk_id)

            page_record = page_record_map.setdefault(
                page_id,
                {
                    "id": page_id,
                    "record_type": "parent",
                    "parent_type": "page",
                    "document_id": record["document_id"],
                    "document_title": record["document_title"],
                    "document_type": record["document_type"],
                    "source_markdown_path": source_markdown_path,
                    "source_url": source_url,
                    "canonical_url": canonical_url,
                    "language_normalized_url": language_normalized_url,
                    "page_numbers": page_numbers,
                    "section_path": [],
                    "child_chunk_ids": [],
                    "child_span_ids": [],
                    "media_ids": [],
                },
            )
            page_record["child_chunk_ids"].append(chunk_id)
            page_record["page_numbers"] = _normalize_page_numbers([*page_record["page_numbers"], *page_numbers])

            section_record = section_record_map.setdefault(
                section_id,
                {
                    "id": section_id,
                    "record_type": "parent",
                    "parent_type": "section",
                    "document_id": record["document_id"],
                    "document_title": record["document_title"],
                    "document_type": record["document_type"],
                    "source_markdown_path": source_markdown_path,
                    "source_url": source_url,
                    "canonical_url": canonical_url,
                    "language_normalized_url": language_normalized_url,
                    "page_numbers": page_numbers,
                    "section_path": section_path,
                    "page_key": page_id,
                    "child_chunk_ids": [],
                    "child_span_ids": [],
                    "media_ids": [],
                },
            )
            section_record["child_chunk_ids"].append(chunk_id)
            section_record["page_numbers"] = _normalize_page_numbers([*section_record["page_numbers"], *page_numbers])

        chunk_ids_by_document: Dict[str, List[str]] = defaultdict(list)
        for record in chunk_map.values():
            doc_key = record["source_markdown_path"] or record["document_id"] or record["source_url"]
            chunk_ids_by_document[doc_key].append(record["id"])
        for ids in chunk_ids_by_document.values():
            ids.sort(key=lambda item_id: chunk_map[item_id]["chunk_index"])
            for idx, chunk_id in enumerate(ids):
                neighbors = []
                if idx > 0:
                    neighbors.append(ids[idx - 1])
                if idx + 1 < len(ids):
                    neighbors.append(ids[idx + 1])
                chunk_map[chunk_id]["neighbor_ids"] = neighbors

        media_records: List[Dict[str, Any]] = []
        fact_records: List[Dict[str, Any]] = []
        media_counter = 0

        def _append_media(item: Dict[str, Any], *, source_url: str = "", source_markdown_path: str = "", page_number: int | None = None) -> None:
            nonlocal media_counter
            normalized = normalize_media_item(item)
            if _is_low_signal_media_item(normalized):
                return
            normalized["source_url"] = source_url or normalized.get("source_url", "")
            normalized["source_document_path"] = source_markdown_path or normalized.get("source_document_path", "")
            normalized["md_path"] = source_markdown_path or normalized.get("md_path", "")
            linked_chunk_ids: List[str] = []
            if source_markdown_path and page_number is not None:
                linked_chunk_ids.extend(chunk_ids_by_doc_page.get((source_markdown_path, int(page_number)), []))
            if not linked_chunk_ids and source_markdown_path:
                linked_chunk_ids.extend(chunk_ids_by_markdown_path.get(source_markdown_path, []))
            if not linked_chunk_ids and source_url:
                linked_chunk_ids.extend(chunk_ids_by_url.get(source_url, []))
            linked_chunk_ids = list(dict.fromkeys(linked_chunk_ids))
            if not linked_chunk_ids:
                return

            sample_chunk = chunk_map[linked_chunk_ids[0]]
            page_key = sample_chunk["page_key"]
            section_keys = list(dict.fromkeys(chunk_map[chunk_id]["section_key"] for chunk_id in linked_chunk_ids))
            media_id = normalized.get("id") or _stable_id(
                "media",
                normalized.get("type"),
                normalized.get("url"),
                normalized.get("local_path"),
                sample_chunk.get("document_id"),
                media_counter,
            )
            media_counter += 1
            document_title = sample_chunk.get("document_title", "")
            media_text = _build_media_embedding_input(
                normalized,
                document_title=document_title,
                section_path=sample_chunk.get("section_path") or [],
            )
            local_path = normalized.get("local_path") or ""
            record = {
                "id": media_id,
                "record_type": "media",
                "media_type": normalized.get("type", "image"),
                "text": media_text,
                "document_id": sample_chunk.get("document_id", ""),
                "document_title": document_title,
                "source_markdown_path": sample_chunk.get("source_markdown_path", ""),
                "source_url": source_url or sample_chunk.get("source_url", ""),
                "page_key": page_key,
                "section_keys": section_keys,
                "page_number": page_number,
                "linked_chunk_ids": linked_chunk_ids,
                "linked_parent_ids": list(dict.fromkeys([page_key, *section_keys])),
                "url": normalized.get("url", ""),
                "asset_uri": normalized.get("asset_uri", ""),
                "local_path": local_path,
                "title": normalized.get("title", ""),
                "caption": normalized.get("caption", ""),
                "description": normalized.get("description", ""),
                "context": normalized.get("context", ""),
                "transcript": normalized.get("transcript", ""),
                "provider": normalized.get("provider", ""),
                "content_hash": normalized.get("content_hash", ""),
                "perceptual_hash": normalized.get("perceptual_hash", ""),
                "source_backend": normalized.get("source_backend", ""),
                "crop_source": normalized.get("crop_source", ""),
                "ocr_text": normalized.get("ocr_text", ""),
                "ocr_model": normalized.get("ocr_model", ""),
                "ocr_model_revision": normalized.get("ocr_model_revision", ""),
                "semantic_caption": normalized.get("semantic_caption", ""),
                "contextual_caption": normalized.get("contextual_caption", ""),
                "visual_description": normalized.get("visual_description", ""),
                "visible_text": normalized.get("visible_text", ""),
                "image_kind": normalized.get("image_kind", ""),
                "semantic_tags": normalized.get("semantic_tags") or [],
                "semantic_relevance": normalized.get("semantic_relevance", ""),
                "annotation_status": normalized.get("annotation_status", ""),
                "annotation_provider": normalized.get("annotation_provider", ""),
                "annotation_model": normalized.get("annotation_model", ""),
                "annotation_model_revision": normalized.get("annotation_model_revision", ""),
                "annotation_prompt_revision": normalized.get("annotation_prompt_revision", ""),
                "annotation_confidence": normalized.get("annotation_confidence"),
                "contains_text": normalized.get("contains_text"),
                "needs_ocr": normalized.get("needs_ocr"),
                "needs_review": normalized.get("needs_review"),
                "bbox": normalized.get("bbox") or {},
                "can_embed_multimodal": bool(
                    local_path
                    and Path(local_path).is_file()
                    and normalized.get("type") in {"image", "page_visual"}
                ),
            }
            media_records.append(record)
            for chunk_id in linked_chunk_ids:
                chunk_map[chunk_id]["media_ids"].append(media_id)
            page_record_map[page_key]["media_ids"].append(media_id)
            for section_key in section_keys:
                section_record_map[section_key]["media_ids"].append(media_id)

        for item in document_media_items:
            source_markdown_path = _relative_or_absolute(item.get("source_document_path") or item.get("md_path") or "")
            page_number = item.get("page_number")
            try:
                page_number = int(page_number) if page_number not in (None, "") else None
            except (TypeError, ValueError):
                page_number = None
            _append_media(item, source_markdown_path=source_markdown_path, page_number=page_number)

        for source_url, items in all_page_media.items():
            if not isinstance(items, list):
                continue
            source_url = str(source_url)
            linked_chunks = chunk_ids_by_url.get(source_url, [])
            if not linked_chunks:
                continue
            source_markdown_path = chunk_map[linked_chunks[0]].get("source_markdown_path", "")
            for item in items:
                if isinstance(item, dict):
                    _append_media(item, source_url=source_url, source_markdown_path=source_markdown_path)

        page_visual_dir = ctx.output_dir("page_visuals")
        visual_page_keys = set()
        for record in chunk_map.values():
            source_file = str(record.get("source_file") or "")
            if not source_file.lower().endswith(".pdf"):
                continue
            for page_number in record.get("page_numbers") or []:
                key = (source_file, int(page_number))
                if key in visual_page_keys:
                    continue
                visual_page_keys.add(key)
                local_path = _render_pdf_page_visual(
                    source_file,
                    page_number=int(page_number),
                    output_dir=page_visual_dir,
                )
                if not local_path:
                    continue
                chunk_ids = list(chunk_ids_by_doc_page.get((_relative_or_absolute(record.get("source_markdown_path") or ""), int(page_number)), []))
                if not chunk_ids:
                    chunk_ids = list(chunk_ids_by_page.get(record["page_key"], []))
                if not chunk_ids:
                    continue
                sample_chunk = chunk_map[chunk_ids[0]]
                section_keys = list(dict.fromkeys(chunk_map[chunk_id]["section_key"] for chunk_id in chunk_ids))
                visual_id = _stable_id("page_visual", sample_chunk.get("source_markdown_path"), page_number)
                page_text = " ".join(
                    _clean_text(chunk_map[chunk_id].get("text"))
                    for chunk_id in chunk_ids[:3]
                    if chunk_map.get(chunk_id)
                ).strip()
                visual_record = {
                    "id": visual_id,
                    "record_type": "media",
                    "media_type": "page_visual",
                    "text": _build_media_embedding_input(
                        {
                            "type": "page_visual",
                            "title": f"{sample_chunk.get('document_title') or 'Document'} page {page_number}",
                            "description": page_text,
                            "context": page_text,
                            "source_url": sample_chunk.get("source_url", ""),
                        },
                        document_title=sample_chunk.get("document_title", ""),
                        section_path=sample_chunk.get("section_path") or [],
                    ),
                    "document_id": sample_chunk.get("document_id", ""),
                    "document_title": sample_chunk.get("document_title", ""),
                    "source_markdown_path": sample_chunk.get("source_markdown_path", ""),
                    "source_url": sample_chunk.get("source_url", ""),
                    "page_key": sample_chunk.get("page_key", ""),
                    "section_keys": section_keys,
                    "page_number": int(page_number),
                    "linked_chunk_ids": chunk_ids,
                    "linked_parent_ids": list(dict.fromkeys([sample_chunk.get("page_key", ""), *section_keys])),
                    "url": "",
                    "asset_uri": local_path,
                    "local_path": local_path,
                    "title": f"{sample_chunk.get('document_title') or 'Document'} page {page_number}",
                    "caption": "",
                    "description": page_text,
                    "context": page_text,
                    "transcript": "",
                    "provider": "rendered_pdf_page",
                    "can_embed_multimodal": True,
                }
                media_records.append(visual_record)
                for chunk_id in chunk_ids:
                    chunk_map[chunk_id]["media_ids"].append(visual_id)
                page_record_map[sample_chunk["page_key"]]["media_ids"].append(visual_id)
                for section_key in section_keys:
                    section_record_map[section_key]["media_ids"].append(visual_id)

        media_by_id = {record["id"]: record for record in media_records}
        sparse_chunk_max_chars = int(ctx.formatter_config.get("sparse_chunk_max_chars") or 6000)
        sparse_parent_max_chars = int(ctx.formatter_config.get("sparse_parent_max_chars") or 8000)
        sparse_media_max_chars = int(ctx.formatter_config.get("sparse_media_max_chars") or 4000)
        sparse_fact_max_chars = int(ctx.formatter_config.get("sparse_fact_max_chars") or 480)
        sparse_evidence_span_max_chars = int(ctx.formatter_config.get("sparse_evidence_span_max_chars") or 900)
        sparse_summary_max_chars = int(ctx.formatter_config.get("sparse_summary_max_chars") or 3000)
        summary_max_chars = int(ctx.formatter_config.get("summary_record_max_chars") or 1800)
        evidence_span_max_chars = int(ctx.formatter_config.get("evidence_span_max_chars") or 700)
        evidence_span_max_sentences = int(ctx.formatter_config.get("evidence_span_max_sentences") or 4)
        evidence_span_max_per_chunk = int(ctx.formatter_config.get("evidence_span_max_per_chunk") or 4)
        sparse_parent_max_headings = int(ctx.formatter_config.get("sparse_parent_max_headings") or 12)
        sparse_parent_max_snippets = int(ctx.formatter_config.get("sparse_parent_max_snippets") or 6)

        chunk_dense_records: List[Dict[str, Any]] = []
        evidence_span_records: List[Dict[str, Any]] = []
        evidence_span_ids_seen: set[str] = set()
        evidence_span_duplicate_count = 0
        lexical_records: List[Dict[str, Any]] = []
        for record in chunk_map.values():
            media_items = [media_by_id[mid] for mid in record["media_ids"][:3] if mid in media_by_id]
            dense_text = _build_chunk_embedding_text(record, media_items)
            sparse_text = _build_chunk_sparse_text(record, max_chars=sparse_chunk_max_chars)
            chunk_dense_records.append(
                {
                    **record,
                    "dense_text": dense_text,
                    "lexical_text": sparse_text,
                    "sparse_text": sparse_text,
                    "media": compact_media_for_metadata(media_items, max_items=3, include_local_path=False),
                }
            )
            lexical_records.append(
                {
                    "id": record["id"],
                    "record_type": "chunk",
                    "text": sparse_text,
                    "tokens": _tokenize_for_bm25(sparse_text),
                }
            )
            span_candidates = []
            for span_text in _sentence_spans(
                record.get("text", ""),
                max_sentences=evidence_span_max_sentences,
                max_chars=evidence_span_max_chars,
            ):
                if len(span_text) < int(ctx.formatter_config.get("evidence_span_min_chars") or 45):
                    continue
                score = _span_signal_score(
                    span_text,
                    document_title=record.get("document_title", ""),
                    section_path=record.get("section_path") or [],
                    heading=record.get("heading", ""),
                )
                span_candidates.append((span_text, score))
            if not span_candidates and record.get("text"):
                fallback = _truncate_chars(record.get("text", ""), evidence_span_max_chars)
                if len(fallback) >= int(ctx.formatter_config.get("evidence_span_min_chars") or 45):
                    span_candidates.append((fallback, 0.0))
            span_candidates.sort(key=lambda item: (-item[1], item[0]))
            for idx, (span_text, _score) in enumerate(span_candidates[: max(1, evidence_span_max_per_chunk)]):
                section_values = list(record.get("section_path") or [])
                breadcrumb = " > ".join(str(value) for value in section_values if str(value))
                section_heading = record.get("heading") or (section_values[-1] if section_values else "")
                span_id = _stable_id(
                    "evidence_span",
                    record.get("language_normalized_url") or record.get("canonical_url") or record.get("source_url"),
                    record.get("section_key"),
                    record["id"],
                    _clean_text(span_text).lower(),
                )
                if span_id in evidence_span_ids_seen:
                    evidence_span_duplicate_count += 1
                    continue
                evidence_span_ids_seen.add(span_id)
                span_record = {
                    "id": span_id,
                    "record_type": "evidence_span",
                    "text": span_text,
                    "document_id": record.get("document_id", ""),
                    "document_title": record.get("document_title", ""),
                    "document_type": record.get("document_type", ""),
                    "source_markdown_path": record.get("source_markdown_path", ""),
                    "source_url": record.get("source_url", ""),
                    "canonical_url": record.get("canonical_url", ""),
                    "language_normalized_url": record.get("language_normalized_url", ""),
                    "page_id": record.get("page_key", ""),
                    "section_id": record.get("section_key", ""),
                    "parent_id": record.get("section_key", "") or record.get("page_key", ""),
                    "chunk_id": record["id"],
                    "page_key": record.get("page_key", ""),
                    "section_key": record.get("section_key", ""),
                    "linked_chunk_ids": [record["id"]],
                    "linked_parent_ids": [record.get("section_key", ""), record.get("page_key", "")],
                    "section_path": list(record.get("section_path") or []),
                    "section_heading": section_heading,
                    "breadcrumb": breadcrumb,
                    "heading": record.get("heading", ""),
                    "page_numbers": list(record.get("page_numbers") or []),
                    "span_type": _classify_span_type(
                        span_text,
                        document_title=record.get("document_title", ""),
                        section_path=record.get("section_path") or [],
                        heading=record.get("heading", ""),
                    ),
                    "authority_class": "official" if "mbzuai.ac.ae" in str(record.get("source_url") or "") else "source",
                    "source_last_seen": record.get("source_last_seen") or "",
                    "validity_status": "active",
                    "span_index": idx,
                }
                span_record["dense_text"] = _build_span_embedding_text(span_record)
                span_record["embedding_text"] = span_record["dense_text"]
                span_record["lexical_text"] = _build_span_sparse_text(
                    span_record,
                    max_chars=sparse_evidence_span_max_chars,
                )
                span_record["sparse_text"] = span_record["lexical_text"]
                evidence_span_records.append(span_record)
                record.setdefault("evidence_span_ids", []).append(span_id)
                page_record_map.get(record.get("page_key", ""), {}).setdefault("child_span_ids", []).append(span_id)
                section_record_map.get(record.get("section_key", ""), {}).setdefault("child_span_ids", []).append(span_id)
                lexical_records.append(
                    {
                        "id": span_id,
                        "record_type": "evidence_span",
                        "span_type": span_record["span_type"],
                        "text": span_record["sparse_text"],
                        "tokens": _tokenize_for_bm25(span_record["sparse_text"]),
                    }
                )
            for idx, snippet in enumerate(
                _extract_fact_snippets(
                    record,
                    min_chars=int(ctx.formatter_config.get("fact_min_chars") or 40),
                    max_chars=int(ctx.formatter_config.get("fact_max_chars") or 360),
                    max_snippets=int(ctx.formatter_config.get("fact_max_snippets_per_chunk") or 5),
                )
            ):
                fact_id = _stable_id("fact", record["id"], idx)
                fact_dense_text = _build_fact_embedding_text(
                    snippet,
                    document_title=record.get("document_title", ""),
                    section_path=record.get("section_path") or [],
                    page_numbers=record.get("page_numbers") or [],
                    source_url=record.get("source_url", ""),
                    heading=record.get("heading", ""),
                )
                fact_record = {
                    "id": fact_id,
                    "record_type": "fact",
                    "text": snippet,
                    "dense_text": fact_dense_text,
                    "lexical_text": _truncate_chars(snippet, sparse_fact_max_chars),
                    "sparse_text": _truncate_chars(snippet, sparse_fact_max_chars),
                    "document_id": record.get("document_id", ""),
                    "document_title": record.get("document_title", ""),
                    "document_type": record.get("document_type", ""),
                    "source_markdown_path": record.get("source_markdown_path", ""),
                    "source_url": record.get("source_url", ""),
                    "page_key": record.get("page_key", ""),
                    "section_key": record.get("section_key", ""),
                    "page_numbers": list(record.get("page_numbers") or []),
                    "linked_chunk_ids": [record["id"]],
                    "linked_parent_ids": [record.get("section_key", ""), record.get("page_key", "")],
                    "linked_span_ids": list(record.get("evidence_span_ids") or []),
                    "heading": record.get("heading", ""),
                }
                fact_records.append(fact_record)
                lexical_records.append(
                    {
                        "id": fact_id,
                        "record_type": "fact",
                        "text": _truncate_chars(snippet, sparse_fact_max_chars),
                        "tokens": _tokenize_for_bm25(_truncate_chars(snippet, sparse_fact_max_chars)),
                    }
                )

        parent_records: List[Dict[str, Any]] = []
        for mapping in (section_record_map, page_record_map):
            for parent in mapping.values():
                child_chunks = [chunk_map[chunk_id] for chunk_id in parent["child_chunk_ids"] if chunk_id in chunk_map]
                media_items = [media_by_id[mid] for mid in list(dict.fromkeys(parent["media_ids"]))[:4] if mid in media_by_id]
                dense_text = _build_parent_embedding_text(parent, child_chunks, media_items)
                sparse_text = _build_parent_sparse_text(
                    parent,
                    child_chunks,
                    media_items,
                    max_chars=sparse_parent_max_chars,
                    max_headings=sparse_parent_max_headings,
                    max_child_snippets=sparse_parent_max_snippets,
                )
                parent_record = {
                    **parent,
                    "dense_text": dense_text,
                    "lexical_text": sparse_text,
                    "sparse_text": sparse_text,
                    "media": compact_media_for_metadata(media_items, max_items=4, include_local_path=False),
                }
                parent_records.append(parent_record)
                lexical_records.append(
                    {
                        "id": parent_record["id"],
                        "record_type": "parent",
                        "parent_type": parent_record["parent_type"],
                        "text": sparse_text,
                        "tokens": _tokenize_for_bm25(sparse_text),
                    }
                )

        for media_record in media_records:
            sparse_text = _build_media_sparse_text(media_record, max_chars=sparse_media_max_chars)
            media_record["sparse_text"] = sparse_text
            lexical_records.append(
                {
                    "id": media_record["id"],
                    "record_type": "media",
                    "text": sparse_text,
                    "tokens": _tokenize_for_bm25(sparse_text),
                }
            )

        summary_records: List[Dict[str, Any]] = []
        for parent in parent_records:
            child_chunks = [chunk_map[chunk_id] for chunk_id in parent.get("child_chunk_ids", []) if chunk_id in chunk_map]
            summary_text = _build_extractive_summary(parent, child_chunks, max_chars=summary_max_chars)
            if not summary_text:
                continue
            summary_id = _stable_id("summary", parent.get("id"), parent.get("document_id"), parent.get("source_url"))
            summary_record = {
                "id": summary_id,
                "record_type": "summary",
                "summary_type": parent.get("parent_type") or "section",
                "text": summary_text,
                "document_id": parent.get("document_id", ""),
                "document_title": parent.get("document_title", ""),
                "document_type": parent.get("document_type", ""),
                "source_markdown_path": parent.get("source_markdown_path", ""),
                "source_url": parent.get("source_url", ""),
                "page_key": parent.get("page_key") or (parent.get("id") if parent.get("parent_type") == "page" else ""),
                "section_key": parent.get("id") if parent.get("parent_type") == "section" else "",
                "section_path": list(parent.get("section_path") or []),
                "page_numbers": list(parent.get("page_numbers") or []),
                "linked_parent_ids": [parent.get("id")] if parent.get("id") else [],
                "linked_chunk_ids": list(parent.get("child_chunk_ids") or []),
                "linked_span_ids": list(parent.get("child_span_ids") or []),
            }
            summary_record["dense_text"] = _build_summary_embedding_text(summary_record)
            summary_record["lexical_text"] = _truncate_chars(summary_text, sparse_summary_max_chars)
            summary_record["sparse_text"] = summary_record["lexical_text"]
            summary_records.append(summary_record)
            lexical_records.append(
                {
                    "id": summary_id,
                    "record_type": "summary",
                    "summary_type": summary_record["summary_type"],
                    "text": summary_record["sparse_text"],
                    "tokens": _tokenize_for_bm25(summary_record["sparse_text"]),
                }
            )

        fallback_answer_records = derive_answer_records_from_bundle(
            {
                "chunk_records": chunk_dense_records,
                "fact_records": fact_records,
            }
        )

        promoted_entities_file = ctx.previous_outputs.get("promoted_entities_file")
        promoted_assertions_file = ctx.previous_outputs.get("promoted_assertions_file")
        promoted_knowledge_graph_file = ctx.previous_outputs.get("promoted_knowledge_graph_file")
        promoted_entities = load_json_safe(promoted_entities_file, []) if promoted_entities_file else []
        promoted_assertions = load_json_safe(promoted_assertions_file, []) if promoted_assertions_file else []
        if not isinstance(promoted_entities, list):
            promoted_entities = []
        if not isinstance(promoted_assertions, list):
            promoted_assertions = []
        promoted_graph_assertions: List[Dict[str, Any]] = []
        if promoted_knowledge_graph_file:
            promoted_graph = load_json_safe(promoted_knowledge_graph_file, {}) or {}
            if isinstance(promoted_graph, Mapping):
                promoted_graph_assertions = _assertions_from_promoted_graph(promoted_graph)
        if promoted_graph_assertions:
            assertion_by_id = {
                _clean_text(assertion.get("id")): assertion
                for assertion in promoted_assertions
                if isinstance(assertion, Mapping) and _clean_text(assertion.get("id"))
            }
            for assertion in promoted_graph_assertions:
                assertion_id = _clean_text(assertion.get("id"))
                if assertion_id and assertion_id not in assertion_by_id:
                    promoted_assertions.append(assertion)
                    assertion_by_id[assertion_id] = assertion

        derived_entity_records = build_entity_records_from_assertions(promoted_assertions)
        entity_records = merge_entity_records(promoted_entities, derived_entity_records)
        assertion_records = build_assertion_embedding_records(promoted_assertions)
        promoted_answer_records = build_answer_records_from_assertions(promoted_assertions)
        answer_records = merge_answer_records(promoted_answer_records, fallback_answer_records)

        span_ids_by_chunk: Dict[str, List[str]] = defaultdict(list)
        for span_record in evidence_span_records:
            for chunk_id in span_record.get("linked_chunk_ids") or []:
                if chunk_id:
                    span_ids_by_chunk[str(chunk_id)].append(str(span_record["id"]))

        for assertion_record in assertion_records:
            linked_chunk_ids = list(
                dict.fromkeys(
                    [
                        str(chunk_id)
                        for chunk_id in (
                            assertion_record.get("source_chunk_ids")
                            or assertion_record.get("linked_chunk_ids")
                            or []
                        )
                        if chunk_id
                    ]
                )
            )
            source_span_ids: List[str] = []
            for chunk_id in linked_chunk_ids:
                source_span_ids.extend(span_ids_by_chunk.get(chunk_id, []))
            source_span_ids = list(dict.fromkeys(source_span_ids))
            assertion_record["source_span_ids"] = source_span_ids
            assertion_record["linked_span_ids"] = source_span_ids

        span_ids_by_assertion = {
            str(assertion_record.get("id") or ""): list(assertion_record.get("source_span_ids") or [])
            for assertion_record in assertion_records
            if str(assertion_record.get("id") or "")
        }
        for answer_record in answer_records:
            assertion_id = str(answer_record.get("source_record_id") or answer_record.get("id") or "")
            source_span_ids = span_ids_by_assertion.get(assertion_id) or []
            if source_span_ids:
                answer_record["source_span_ids"] = source_span_ids
                answer_record["linked_span_ids"] = source_span_ids

        for assertion_record in assertion_records:
            lexical_text = assertion_record.get("lexical_text") or assertion_record.get("text") or ""
            lexical_records.append(
                {
                    "id": assertion_record["id"],
                    "record_type": "assertion",
                    "text": lexical_text,
                    "tokens": _tokenize_for_bm25(lexical_text),
                }
            )

        image_media_records = [
            record
            for record in media_records
            if record.get("media_type") in {"image", "page_visual"}
        ]
        multimodal_media_records = [
            record for record in image_media_records if record.get("can_embed_multimodal")
        ]
        text_only_image_records = len(image_media_records) - len(multimodal_media_records)
        image_text_only_ratio = text_only_image_records / max(1, len(image_media_records))
        if bool(ctx.formatter_config.get("require_multimodal_media", False)):
            minimum_multimodal = max(
                1,
                int(ctx.formatter_config.get("minimum_multimodal_media_records") or 1),
            )
            maximum_text_only_ratio = max(
                0.0,
                float(ctx.formatter_config.get("maximum_image_text_only_ratio") or 0.0),
            )
            if len(multimodal_media_records) < minimum_multimodal:
                return StageResult.failure(
                    "Retrieval formatting produced "
                    f"{len(multimodal_media_records)} multimodal image records; "
                    f"minimum is {minimum_multimodal}",
                    metrics={
                        "image_media_records": len(image_media_records),
                        "multimodal_media_records": len(multimodal_media_records),
                        "text_only_image_records": text_only_image_records,
                    },
                )
            if image_text_only_ratio > maximum_text_only_ratio:
                return StageResult.failure(
                    "Text-only image ratio "
                    f"{image_text_only_ratio:.6f} exceeds {maximum_text_only_ratio:.6f}",
                    metrics={
                        "image_media_records": len(image_media_records),
                        "multimodal_media_records": len(multimodal_media_records),
                        "text_only_image_records": text_only_image_records,
                        "image_text_only_ratio": image_text_only_ratio,
                    },
                )

        bundle = {
            "version": 5,
            "generated_at": ctx.run_id,
            "chunk_records": chunk_dense_records,
            "parent_records": parent_records,
            "media_records": media_records,
            "fact_records": fact_records,
            "evidence_span_records": evidence_span_records,
            "summary_records": summary_records,
            "entity_records": entity_records,
            "assertion_records": assertion_records,
            "answer_records": answer_records,
            "stats": {
                "chunk_count": len(chunk_dense_records),
                "parent_count": len(parent_records),
                "media_count": len(media_records),
                "image_media_count": len(image_media_records),
                "multimodal_media_count": len(multimodal_media_records),
                "text_only_image_count": text_only_image_records,
                "image_text_only_ratio": round(image_text_only_ratio, 6),
                "fact_count": len(fact_records),
                "evidence_span_count": len(evidence_span_records),
                "summary_count": len(summary_records),
                "entity_count": len(entity_records),
                "assertion_count": len(assertion_records),
                "graph_assertion_count": len(promoted_graph_assertions),
                "answer_count": len(answer_records),
                "lexical_count": len(lexical_records),
                "evidence_span_duplicate_count": evidence_span_duplicate_count,
            },
        }

        chunk_file = ctx.stage_work_dir / "chunk_dense_records.json"
        parent_file = ctx.stage_work_dir / "parent_dense_records.json"
        media_file = ctx.stage_work_dir / "media_dense_records.json"
        fact_file = ctx.stage_work_dir / "fact_dense_records.json"
        evidence_span_file = ctx.stage_work_dir / "evidence_span_dense_records.json"
        summary_file = ctx.stage_work_dir / "summary_dense_records.json"
        entity_file = ctx.stage_work_dir / "entity_records.json"
        assertion_file = ctx.stage_work_dir / "assertion_dense_records.json"
        answer_file = ctx.stage_work_dir / "answer_dense_records.json"
        lexical_file = ctx.stage_work_dir / "lexical_corpus.json"
        bundle_file = ctx.stage_work_dir / "retrieval_bundle.json"

        atomic_write_json(chunk_file, chunk_dense_records)
        atomic_write_json(parent_file, parent_records)
        atomic_write_json(media_file, media_records)
        atomic_write_json(fact_file, fact_records)
        atomic_write_json(evidence_span_file, evidence_span_records)
        atomic_write_json(summary_file, summary_records)
        atomic_write_json(entity_file, entity_records)
        atomic_write_json(assertion_file, assertion_records)
        atomic_write_json(answer_file, answer_records)
        atomic_write_json(lexical_file, lexical_records)
        atomic_write_json(bundle_file, bundle)

        logger.info(
            "Gemini retrieval formatter: %d chunks, %d parents, %d media records, %d fact records, %d evidence spans, %d summaries, %d assertions, %d answers, %d duplicate evidence spans skipped",
            len(chunk_dense_records),
            len(parent_records),
            len(media_records),
            len(fact_records),
            len(evidence_span_records),
            len(summary_records),
            len(assertion_records),
            len(answer_records),
            evidence_span_duplicate_count,
        )

        artifacts = [
            ctx.make_artifact(
                bundle_file,
                artifact_type="retrieval_bundle",
                role="retrieval_corpus",
                metadata=bundle["stats"],
            ),
            ctx.make_artifact(
                chunk_file,
                artifact_type="formatted_documents",
                role="embedding_payload_chunks",
                metadata={"records": len(chunk_dense_records), "kind": "chunks"},
            ),
            ctx.make_artifact(
                parent_file,
                artifact_type="formatted_documents",
                role="embedding_payload_parents",
                metadata={"records": len(parent_records), "kind": "parents"},
            ),
            ctx.make_artifact(
                media_file,
                artifact_type="formatted_documents",
                role="embedding_payload_media",
                metadata={"records": len(media_records), "kind": "media"},
            ),
            ctx.make_artifact(
                fact_file,
                artifact_type="formatted_documents",
                role="embedding_payload_facts",
                metadata={"records": len(fact_records), "kind": "facts"},
            ),
            ctx.make_artifact(
                evidence_span_file,
                artifact_type="formatted_documents",
                role="embedding_payload_evidence_spans",
                metadata={"records": len(evidence_span_records), "kind": "evidence_spans"},
            ),
            ctx.make_artifact(
                summary_file,
                artifact_type="formatted_documents",
                role="embedding_payload_summaries",
                metadata={"records": len(summary_records), "kind": "summaries"},
            ),
            ctx.make_artifact(
                entity_file,
                artifact_type="formatted_documents",
                role="entity_records",
                metadata={"records": len(entity_records), "kind": "entities"},
            ),
            ctx.make_artifact(
                assertion_file,
                artifact_type="formatted_documents",
                role="embedding_payload_assertions",
                metadata={"records": len(assertion_records), "kind": "assertions"},
            ),
            ctx.make_artifact(
                answer_file,
                artifact_type="formatted_documents",
                role="retrieval_answer_records",
                metadata={"records": len(answer_records), "kind": "answers"},
            ),
            ctx.make_artifact(
                lexical_file,
                artifact_type="lexical_corpus",
                role="lexical_retrieval",
                metadata={"records": len(lexical_records)},
            ),
        ]

        return StageResult.success(
            outputs={
                "retrieval_bundle_file": str(bundle_file),
                "chunk_embedding_file": str(chunk_file),
                "parent_embedding_file": str(parent_file),
                "media_embedding_file": str(media_file),
                "fact_embedding_file": str(fact_file),
                "evidence_span_embedding_file": str(evidence_span_file),
                "summary_embedding_file": str(summary_file),
                "entity_records_file": str(entity_file),
                "assertion_embedding_file": str(assertion_file),
                "answer_embedding_file": str(answer_file),
                "lexical_corpus_file": str(lexical_file),
                "formatted_file": str(chunk_file),
            },
            metrics={
                "chunk_records": len(chunk_dense_records),
                "parent_records": len(parent_records),
                "media_records": len(media_records),
                "image_media_records": len(image_media_records),
                "multimodal_media_records": len(multimodal_media_records),
                "text_only_image_records": text_only_image_records,
                "image_text_only_ratio": round(image_text_only_ratio, 6),
                "fact_records": len(fact_records),
                "evidence_span_records": len(evidence_span_records),
                "summary_records": len(summary_records),
                "entity_records": len(entity_records),
                "assertion_records": len(assertion_records),
                "graph_assertion_records": len(promoted_graph_assertions),
                "answer_records": len(answer_records),
                "lexical_records": len(lexical_records),
            },
            artifacts=artifacts,
        )
