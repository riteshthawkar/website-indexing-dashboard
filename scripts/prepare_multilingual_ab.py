from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.chunking import (
    estimate_token_count,
    token_counting_method,
    window_text_to_token_budget,
)
from pipeline.core.media import media_chunk_match
from pipeline.evaluation.dataset import load_eval_examples
from pipeline.stages.chunkers.common import docling_chunks_from_json, hybrid_markdown_chunks
from pipeline.stages.formatters.gemini_retrieval_formatter import (
    _build_chunk_embedding_text,
    _build_media_embedding_input,
)


EXPERIMENT_SCHEMA = "mbzuai.multilingual.controlled_ab.v1"
CANDIDATE_REPRESENTATION_REVISION = "multilingual-ab-grounded-media-v3"
PARENT_EMBEDDING_TOKEN_BUDGET = 4096
PARENT_SECTION_EMBEDDING_TOKEN_BUDGET = 4096
DEFAULT_REPRESENTATION = (
    PROJECT_ROOT
    / "runs/mbzuai_representation_v2/mbzuai-representation-v2-20260821-v2"
    / "stage_outputs/extract_page_cards_and_actions/representation_v2_bundle.json"
)
DEFAULT_MEDIA = (
    PROJECT_ROOT
    / "runs/mbzuai_corpus_preparation/mbzuai-corpus-preparation-20260821-v4"
    / "stage_outputs/prepare_corpus_boundary/prepared_media_manifest.json"
)
DEFAULT_CORPUS_RUN = (
    PROJECT_ROOT / "runs/mbzuai_corpus_preparation/mbzuai-corpus-preparation-20260821-v4"
)
DEFAULT_NAVIGATION_CATALOG = (
    PROJECT_ROOT
    / "runs/mbzuai_page_graph_bridge/mbzuai-page-graph-bridge-20260822-v4"
    / "stage_outputs/bridge_page_graph/page_graph_navigation_catalog.json"
)
DEFAULT_DATASET = PROJECT_ROOT / "eval/mbzuai_gold/mbzuai_multilingual_v2.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs/evaluation/mbzuai-multilingual-controlled-ab-v1"

CHUNK_CONFIGS: Dict[str, Dict[str, int]] = {
    "c300": {
        "target_tokens": 300,
        "max_tokens": 450,
        "overlap_tokens": 50,
        "min_chunk_tokens": 90,
    },
    "c450": {
        "target_tokens": 450,
        "max_tokens": 650,
        "overlap_tokens": 80,
        "min_chunk_tokens": 140,
    },
    "c650": {
        "target_tokens": 650,
        "max_tokens": 900,
        "overlap_tokens": 100,
        "min_chunk_tokens": 160,
    },
}
EMBEDDING_SPECS = {
    "gemini2_768": {
        "provider": "gemini",
        "model": "gemini-embedding-2",
        "dimensions": 768,
        "query_format": "task: search result | query: {query}",
        "document_format": "title: {title} | text: {text}",
        "media_input": "caption_text",
    },
    "gemini2_1536": {
        "provider": "gemini",
        "model": "gemini-embedding-2",
        "dimensions": 1536,
        "query_format": "task: search result | query: {query}",
        "document_format": "title: {title} | text: {text}",
        "media_input": "caption_text",
    },
    "gemini2_768_mm": {
        "provider": "gemini",
        "model": "gemini-embedding-2",
        "dimensions": 768,
        "query_format": "task: search result | query: {query}",
        "document_format": "title: {title} | text: {text}",
        "media_input": "image_and_caption_text",
        "base_text_embedding": "gemini2_768",
    },
    "gemini2_1536_mm": {
        "provider": "gemini",
        "model": "gemini-embedding-2",
        "dimensions": 1536,
        "query_format": "task: search result | query: {query}",
        "document_format": "title: {title} | text: {text}",
        "media_input": "image_and_caption_text",
        "base_text_embedding": "gemini2_1536",
    },
    "qwen3_06b_1024": {
        "provider": "qwen_local_gpu",
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "dimensions": 1024,
        "query_instruction": "Given a web search query, retrieve relevant passages that answer the query",
        "pooling": "last_token",
        "normalize": True,
        "media_input": "caption_text",
    },
}
INDEX_MODES = {
    "dense_core": {
        "record_kinds": ["chunk", "parent", "parent_section", "media"],
        "dense": True,
        "sparse": False,
        "rrf_k": None,
    },
    "hybrid_core": {
        "record_kinds": ["chunk", "parent", "parent_section", "media"],
        "dense": True,
        "sparse": True,
        "dense_top_k": 100,
        "sparse_top_k": 100,
        "rrf_k": 60,
    },
    "dense_graph": {
        "record_kinds": ["chunk", "parent", "parent_section", "media", "page_card", "action"],
        "dense": True,
        "sparse": False,
        "rrf_k": None,
    },
    "hybrid_graph": {
        "record_kinds": ["chunk", "parent", "parent_section", "media", "page_card", "action"],
        "dense": True,
        "sparse": True,
        "dense_top_k": 100,
        "sparse_top_k": 100,
        "rrf_k": 60,
    },
}
SELECTION_POLICY = {
    "selection_split": "selection",
    "regression_split": "regression",
    "sealed_holdout_split": "holdout",
    "selection_score_weights": {
        "evidence_ndcg_at_10": 0.27,
        "evidence_mrr_at_10": 0.18,
        "source_recall_at_10": 0.18,
        "arabic_evidence_hit_at_10": 0.09,
        "synthesis_source_recall_at_10": 0.09,
        "navigation_action_hit_at_5": 0.045,
        "media_hit_at_5": 0.045,
        "abstention_balanced_accuracy": 0.10,
    },
    "baseline_variant": "c450__gemini2_1536__hybrid_core",
    "finalist_count": 3,
    "paired_bootstrap_samples": 2000,
    "bootstrap_seed": 20260822,
    "regression_max_absolute_drop": 0.03,
    "holdout_tie_margin": 0.01,
    "tie_break_order": [
        "within_quality_margin",
        "monthly_cost",
        "p95_latency",
        "vector_bytes",
    ],
}

_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)


def _read_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _clean(value: Any) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFC", str(value or ""))).strip()


def _normalize_url(value: Any) -> str:
    raw = _clean(value)
    if not raw.startswith(("http://", "https://")):
        return ""
    parsed = urlsplit(raw)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _bounded_text(value: str, maximum_chars: int = 14000) -> str:
    value = str(value or "").strip()
    if len(value) <= maximum_chars:
        return value
    first = int(maximum_chars * 0.55)
    middle = int(maximum_chars * 0.20)
    last = maximum_chars - first - middle
    midpoint = max(0, len(value) // 2 - middle // 2)
    return (
        value[:first].rstrip()
        + "\n\n[...content window omitted...]\n\n"
        + value[midpoint : midpoint + middle].strip()
        + "\n\n[...content window omitted...]\n\n"
        + value[-last:].lstrip()
    )


def _record(
    *,
    record_id: str,
    kind: str,
    text: str,
    raw_text: str,
    title: str = "",
    language: str = "",
    source_url: str = "",
    document_revision_id: str = "",
    page_card_ids: Sequence[str] = (),
    section_ids: Sequence[str] = (),
    action_id: str = "",
    media_id: str = "",
    section_path: Sequence[str] = (),
    page_numbers: Sequence[int] = (),
    local_path: str = "",
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    normalized_text = str(text or "").strip()
    normalized_raw = str(raw_text or "").strip()
    return {
        "id": str(record_id),
        "kind": str(kind),
        "text": normalized_text,
        "raw_text": normalized_raw,
        "sparse_text": normalized_text,
        "text_sha256": _sha256_text(normalized_text),
        "title": _clean(title),
        "language": str(language or ""),
        "source_url": _normalize_url(source_url),
        "source_host": urlsplit(_normalize_url(source_url)).netloc.lower(),
        "document_revision_id": str(document_revision_id or ""),
        "page_card_ids": list(dict.fromkeys(str(value) for value in page_card_ids if str(value))),
        "section_ids": list(dict.fromkeys(str(value) for value in section_ids if str(value))),
        "action_id": str(action_id or ""),
        "media_id": str(media_id or ""),
        "section_path": [str(value) for value in section_path if str(value)],
        "page_numbers": [int(value) for value in page_numbers if str(value).strip()],
        "local_path": str(local_path or ""),
        "metadata": dict(metadata or {}),
    }


def _load_structured_by_markdown(corpus_run: Path) -> Dict[str, Path]:
    catalog_path = corpus_run / "artifact_catalog.json"
    payload = _read_json(catalog_path)
    output: Dict[str, Path] = {}
    for row in payload.get("records") or []:
        if not isinstance(row, Mapping) or row.get("artifact_type") != "structured_document":
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        markdown_path = str(metadata.get("source_markdown_path") or "")
        structured_path = Path(str(row.get("local_path") or ""))
        if markdown_path and structured_path.is_file():
            output[str(Path(markdown_path).expanduser().resolve())] = structured_path.resolve()
    return output


def _source_url_by_markdown(media_manifest: Mapping[str, Any]) -> Dict[str, str]:
    output: Dict[str, str] = {}
    for item in media_manifest.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        markdown = str(item.get("source_document_path") or item.get("md_path") or "")
        url = _normalize_url(item.get("source_url"))
        if markdown and url:
            output.setdefault(str(Path(markdown).expanduser().resolve()), url)
    return output


def _section_lookup(page_cards: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    output: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for page in page_cards:
        revision_id = str(page.get("document_revision_id") or "")
        if not revision_id:
            continue
        for section in page.get("sections") or []:
            if not isinstance(section, Mapping):
                continue
            heading = _clean(section.get("heading")).casefold()
            section_id = str(section.get("section_id") or "")
            if heading and section_id:
                output[revision_id][heading].append(section_id)
    return {key: dict(value) for key, value in output.items()}


def _matching_section_ids(
    revision_id: str,
    section_path: Sequence[str],
    heading: str,
    lookup: Mapping[str, Mapping[str, Sequence[str]]],
) -> List[str]:
    by_heading = lookup.get(revision_id) or {}
    output: List[str] = []
    for value in [heading, *reversed(list(section_path or []))]:
        normalized = _clean(value).casefold()
        if normalized:
            output.extend(str(item) for item in (by_heading.get(normalized) or []))
    return list(dict.fromkeys(output))


def _prepare_media_records(
    media_manifest: Mapping[str, Any],
    *,
    documents_by_markdown: Mapping[str, Mapping[str, Any]],
    pages_by_url: Mapping[str, Mapping[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, List[Mapping[str, Any]]]]:
    records: List[Dict[str, Any]] = []
    items_by_document: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    excluded_kinds = {"background", "decorative", "divider", "icon", "logo"}
    for item in media_manifest.get("items") or []:
        if not isinstance(item, Mapping):
            continue
        if item.get("annotation_status") != "completed" or item.get("semantic_relevance") != "substantive":
            continue
        if item.get("needs_review") or str(item.get("image_kind") or "") in excluded_kinds:
            continue
        if int(item.get("width") or 0) * int(item.get("height") or 0) < 90000:
            continue
        local_path = Path(str(item.get("local_path") or ""))
        if not local_path.is_file():
            continue
        content_hash = str(item.get("content_hash") or "")
        if content_hash and content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        source_url = _normalize_url(item.get("source_url"))
        markdown_path = str(item.get("source_document_path") or item.get("md_path") or "")
        resolved_markdown = str(Path(markdown_path).expanduser().resolve()) if markdown_path else ""
        document = documents_by_markdown.get(resolved_markdown)
        page = pages_by_url.get(source_url)
        revision_id = str((document or {}).get("document_revision_id") or (page or {}).get("document_revision_id") or "")
        page_card_ids = [str((page or {}).get("page_card_id") or "")] if page else []
        document_title = _clean((document or {}).get("title") or item.get("page_title") or item.get("title"))
        normalized_item = dict(item)
        normalized_item["source_url"] = source_url
        media_text = _build_media_embedding_input(
            normalized_item,
            document_title=document_title,
            section_path=list(item.get("section_path") or []),
        )
        media_id = str(item.get("id") or "")
        if not media_id or not media_text:
            continue
        record = _record(
            record_id=media_id,
            kind="media",
            text=media_text,
            raw_text=media_text,
            title=document_title,
            language=str((document or {}).get("language") or (page or {}).get("language") or ""),
            source_url=source_url,
            document_revision_id=revision_id,
            page_card_ids=page_card_ids,
            media_id=media_id,
            section_path=list(item.get("section_path") or []),
            page_numbers=[int(item["page_number"])] if item.get("page_number") not in (None, "") else [],
            local_path=str(local_path.resolve()),
            metadata={
                "image_kind": item.get("image_kind"),
                "semantic_relevance": item.get("semantic_relevance"),
                "content_hash": content_hash,
                "source_type": item.get("source_type"),
                "can_embed_multimodal": True,
            },
        )
        records.append(record)
        if revision_id:
            items_by_document[revision_id].append(normalized_item)
    records.sort(key=lambda row: row["id"])
    return records, items_by_document


def _prepare_graph_records(
    representation: Mapping[str, Any],
    navigation_catalog: Mapping[str, Any],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pages_by_id = {
        str(page.get("page_card_id")): page
        for page in representation.get("page_cards") or []
        if isinstance(page, Mapping) and page.get("page_card_id") and page.get("content_backed")
    }
    navigation_actions_by_id = {
        str(action.get("action_id")): action
        for action in navigation_catalog.get("actions") or []
        if isinstance(action, Mapping) and action.get("action_id")
    }
    page_records: List[Dict[str, Any]] = []
    for page_id, page in sorted(pages_by_id.items()):
        topics = [_clean(item.get("label")) for item in page.get("topics") or [] if isinstance(item, Mapping)]
        audiences = [_clean(item.get("label")) for item in page.get("audiences") or [] if isinstance(item, Mapping)]
        sections = [_clean(item.get("heading")) for item in page.get("sections") or [] if isinstance(item, Mapping)]
        raw_text = "\n".join(
            part
            for part in [
                _clean(page.get("title")),
                _clean(page.get("purpose_summary")),
                *topics[:24],
                *audiences[:12],
                *sections[:60],
            ]
            if part
        )
        dense_text = "\n".join(
            part
            for part in [
                f"TITLE: {_clean(page.get('title'))}",
                f"PAGE_TYPE: {_clean(page.get('page_type'))}",
                f"PURPOSE: {_clean(page.get('purpose_summary'))}",
                f"AUDIENCES: {', '.join(audiences)}" if audiences else "",
                f"TOPICS: {', '.join(topics)}" if topics else "",
                "SECTIONS:\n" + "\n".join(f"- {section}" for section in sections[:60]) if sections else "",
                f"SOURCE_URL: {_normalize_url(page.get('source_url'))}",
            ]
            if part
        )
        page_records.append(
            _record(
                record_id=page_id,
                kind="page_card",
                text=_bounded_text(dense_text),
                raw_text=raw_text,
                title=_clean(page.get("title")),
                language=str(page.get("language") or ""),
                source_url=str(page.get("source_url") or ""),
                document_revision_id=str(page.get("document_revision_id") or ""),
                page_card_ids=[page_id],
                section_ids=[
                    str(section.get("section_id"))
                    for section in page.get("sections") or []
                    if isinstance(section, Mapping) and section.get("section_id")
                ],
                metadata={"page_type": page.get("page_type")},
            )
        )

    action_records: List[Dict[str, Any]] = []
    for action in representation.get("actions") or []:
        action_id = str(action.get("action_id") or "") if isinstance(action, Mapping) else ""
        if (
            not isinstance(action, Mapping)
            or action_id not in navigation_actions_by_id
            or not action.get("retrieval_eligible")
            or action.get("is_template")
        ):
            continue
        catalog_action = navigation_actions_by_id[action_id]
        page_id = str(action.get("page_card_id") or "")
        page = pages_by_id.get(page_id)
        if not action_id or not page:
            continue
        raw_text = "\n".join(
            _clean(action.get(key))
            for key in ("label", "context_label", "action_type", "target_url", "source_section_heading")
            if _clean(action.get(key))
        )
        text = "\n".join(
            part
            for part in [
                f"ACTION: {_clean(action.get('label'))}",
                f"CONTEXT: {_clean(action.get('context_label'))}",
                f"ACTION_TYPE: {_clean(action.get('action_type'))}",
                f"TARGET: {_clean(action.get('target_url'))}",
                f"SOURCE_PAGE: {_clean(page.get('title'))}",
                f"SOURCE_SECTION: {_clean(action.get('source_section_heading'))}",
                f"SOURCE_URL: {_normalize_url(page.get('source_url'))}",
            ]
            if part
        )
        action_records.append(
            _record(
                record_id=action_id,
                kind="action",
                text=text,
                raw_text=raw_text,
                title=_clean(action.get("label")),
                language=str(page.get("language") or ""),
                source_url=str(page.get("source_url") or ""),
                document_revision_id=str(page.get("document_revision_id") or ""),
                page_card_ids=[page_id],
                section_ids=[str(action.get("source_section_id") or "")] if action.get("source_section_id") else [],
                action_id=action_id,
                metadata={
                    "action_type": action.get("action_type"),
                    "target_url": catalog_action.get("canonical_target_url")
                    or catalog_action.get("target_url"),
                    "official_target": bool(catalog_action.get("official_target")),
                },
            )
        )
    action_records.sort(key=lambda row: row["id"])
    return page_records, action_records


def _parent_records(
    chunk_records: Sequence[Dict[str, Any]],
    *,
    config_id: str,
    documents_by_revision: Mapping[str, Mapping[str, Any]],
    page_records_by_revision: Mapping[str, Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    by_document: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_section: Dict[tuple[str, tuple[str, ...], tuple[int, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for row in chunk_records:
        revision_id = str(row.get("document_revision_id") or "")
        if not revision_id:
            continue
        by_document[revision_id].append(row)
        section_path = tuple(str(value) for value in row.get("section_path") or [])
        page_numbers = tuple(int(value) for value in row.get("page_numbers") or [])
        if section_path or page_numbers:
            by_section[(revision_id, section_path, page_numbers)].append(row)

    output: List[Dict[str, Any]] = []

    def add_parent(kind: str, key: str, children: Sequence[Dict[str, Any]]) -> None:
        ordered = sorted(children, key=lambda row: int((row.get("metadata") or {}).get("chunk_index") or 0))
        first = ordered[0]
        section_path = list(first.get("section_path") or []) if kind == "parent_section" else []
        page_numbers = sorted({page for row in ordered for page in (row.get("page_numbers") or [])})
        header = [f"TITLE: {first.get('title') or ''}", f"PARENT_TYPE: {kind}"]
        if section_path:
            header.append(f"SECTION: {' > '.join(section_path)}")
        if page_numbers:
            header.append(f"PAGES: {', '.join(str(value) for value in page_numbers)}")
        revision_id = str(first.get("document_revision_id") or "")
        if kind == "parent":
            document = documents_by_revision.get(revision_id) or {}
            markdown_path = Path(str(document.get("markdown_path") or ""))
            complete_source_text = (
                markdown_path.read_text(encoding="utf-8", errors="replace")
                if markdown_path.is_file()
                else "\n\n".join(
                    str(row.get("raw_text") or "")
                    for row in ordered
                    if row.get("raw_text")
                )
            )
            page_synopsis = "\n\n".join(
                str(row.get("text") or "")
                for row in page_records_by_revision.get(revision_id) or []
                if row.get("text")
            )
            body = "\n\n".join(
                part
                for part in (
                    f"SOURCE_URL: {first.get('source_url') or ''}",
                    f"PAGE_SYNOPSIS:\n{page_synopsis}" if page_synopsis else "",
                    "CONTENT_WINDOWS:\n"
                    + window_text_to_token_budget(
                        complete_source_text,
                        max_tokens=PARENT_EMBEDDING_TOKEN_BUDGET,
                    ),
                )
                if part
            )
            complete_source_token_count = estimate_token_count(complete_source_text)
            budget = PARENT_EMBEDDING_TOKEN_BUDGET
        else:
            complete_source_text = "\n\n".join(
                str(row.get("raw_text") or "")
                for row in ordered
                if row.get("raw_text")
            )
            body = complete_source_text
            complete_source_token_count = estimate_token_count(complete_source_text)
            budget = PARENT_SECTION_EMBEDDING_TOKEN_BUDGET
        text = window_text_to_token_budget(
            "\n".join([*header, "", body]),
            max_tokens=budget,
        )
        if estimate_token_count(text) > budget:
            raise RuntimeError(f"{kind} representation exceeded {budget} tokens")
        output.append(
            _record(
                record_id=key,
                kind=kind,
                text=text,
                raw_text=text,
                title=str(first.get("title") or ""),
                language=str(first.get("language") or ""),
                source_url=str(first.get("source_url") or ""),
                document_revision_id=revision_id,
                page_card_ids=list(first.get("page_card_ids") or []),
                section_ids=list(dict.fromkeys(section for row in ordered for section in (row.get("section_ids") or []))),
                section_path=section_path,
                page_numbers=page_numbers,
                metadata={
                    "child_chunk_ids": [row["id"] for row in ordered],
                    "representation_revision": CANDIDATE_REPRESENTATION_REVISION,
                    "embedding_token_budget": budget,
                    "embedding_token_count": estimate_token_count(text),
                    "complete_source_token_count": complete_source_token_count,
                    "content_windowed": complete_source_token_count > budget,
                },
            )
        )

    for revision_id, children in sorted(by_document.items()):
        add_parent("parent", f"parent:{config_id}:{revision_id}:page", children)
    for (revision_id, section_path, page_numbers), children in sorted(by_section.items()):
        material = json.dumps([section_path, page_numbers], ensure_ascii=False, separators=(",", ":"))
        digest = hashlib.sha1(material.encode()).hexdigest()[:12]
        add_parent("parent_section", f"parent:{config_id}:{revision_id}:section:{digest}", children)
    return output


def _chunk_records_for_config(
    *,
    config_id: str,
    config: Mapping[str, int],
    documents: Sequence[Mapping[str, Any]],
    pages_by_revision: Mapping[str, Sequence[Mapping[str, Any]]],
    section_lookup: Mapping[str, Mapping[str, Sequence[str]]],
    structured_by_markdown: Mapping[str, Path],
    source_url_by_markdown: Mapping[str, str],
    media_items_by_document: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    structured_success = 0
    structured_fallback = 0
    chunks_with_media = 0
    media_chunk_association_count = 0
    media_match_method_counts: Counter[str] = Counter()
    linked_media_ids: set[str] = set()
    started = time.perf_counter()
    for document_index, document in enumerate(documents, start=1):
        markdown_path = Path(str(document.get("markdown_path") or "")).expanduser().resolve()
        if not markdown_path.is_file():
            raise FileNotFoundError(markdown_path)
        revision_id = str(document.get("document_revision_id") or "")
        pages = list(pages_by_revision.get(revision_id) or [])
        page_ids = [str(page.get("page_card_id")) for page in pages if page.get("page_card_id")]
        source_url = _normalize_url(document.get("source_url")) or source_url_by_markdown.get(str(markdown_path), "")
        source_info = {
            "path": markdown_path,
            "source_url": source_url,
            "source_file": str(document.get("source_file") or ""),
            "document_title": _clean(document.get("title")) or markdown_path.stem,
            "document_type": str(document.get("source_type") or ""),
            "source_backend": "docling" if str(markdown_path) in structured_by_markdown else "markdown",
        }
        chunks: List[Dict[str, Any]] = []
        structured_path = structured_by_markdown.get(str(markdown_path))
        if structured_path:
            try:
                chunks = docling_chunks_from_json(
                    structured_path,
                    strategy="hybrid",
                    source_info=source_info,
                    max_tokens=int(config["max_tokens"]),
                    overlap_tokens=int(config["overlap_tokens"]),
                    max_chunks_per_document=0,
                    always_emit_headings=False,
                )
                structured_success += 1
            except Exception as exc:
                structured_fallback += 1
                print(f"{config_id}: Docling chunk fallback for {markdown_path.name}: {exc}", flush=True)
        if not chunks:
            chunks = hybrid_markdown_chunks(
                markdown_path.read_text(encoding="utf-8", errors="replace"),
                source_info=source_info,
                target_tokens=int(config["target_tokens"]),
                max_tokens=int(config["max_tokens"]),
                overlap_tokens=int(config["overlap_tokens"]),
                min_chunk_tokens=int(config["min_chunk_tokens"]),
                max_chunks_per_document=0,
                include_section_headings=True,
            )
        media_items = list(media_items_by_document.get(revision_id) or [])
        for chunk in chunks:
            section_path = list(chunk.get("section_path") or [])
            heading = _clean(chunk.get("heading"))
            raw_text = str(chunk.get("text") or "").strip()
            page_numbers = list(chunk.get("page_numbers") or [])
            ranked_media: List[tuple[float, str, Mapping[str, Any], str]] = []
            for item in media_items:
                match = media_chunk_match(
                    item,
                    chunk_page_numbers=page_numbers,
                    chunk_section_path=section_path,
                    chunk_text=raw_text,
                )
                if not match["matched"]:
                    continue
                ranked_media.append(
                    (
                        float(match["score"]),
                        str(item.get("id") or item.get("content_hash") or ""),
                        item,
                        str(match["method"]),
                    )
                )
            ranked_media.sort(key=lambda row: (-row[0], row[1]))
            selected_media = ranked_media[:3]
            matched_media = [row[2] for row in selected_media]
            if matched_media:
                chunks_with_media += 1
                media_chunk_association_count += len(matched_media)
                for _score, media_id, _item, method in selected_media:
                    if media_id:
                        linked_media_ids.add(media_id)
                    media_match_method_counts[method] += 1
            chunk_payload = {
                **chunk,
                "document_title": source_info["document_title"],
                "document_type": source_info["document_type"],
                "source_url": source_url,
            }
            dense_text = _build_chunk_embedding_text(chunk_payload, matched_media)
            text_digest = hashlib.sha1(raw_text.encode("utf-8")).hexdigest()[:12]
            chunk_id = f"chunk:{config_id}:{revision_id}:{int(chunk.get('chunk_index') or 0):05d}:{text_digest}"
            output.append(
                _record(
                    record_id=chunk_id,
                    kind="chunk",
                    text=dense_text,
                    raw_text=raw_text,
                    title=source_info["document_title"],
                    language=str(document.get("language") or ""),
                    source_url=source_url,
                    document_revision_id=revision_id,
                    page_card_ids=page_ids,
                    section_ids=_matching_section_ids(revision_id, section_path, heading, section_lookup),
                    section_path=section_path,
                    page_numbers=page_numbers,
                    metadata={
                        "chunk_index": int(chunk.get("chunk_index") or 0),
                        "token_count": int(chunk.get("token_count") or estimate_token_count(raw_text)),
                        "source_backend": chunk.get("source_backend"),
                        "element_types": chunk.get("element_types") or [],
                        "media_ids": [row[1] for row in selected_media if row[1]],
                        "media_match_methods": [row[3] for row in selected_media],
                    },
                )
            )
        if document_index % 250 == 0:
            print(f"{config_id}: chunked {document_index}/{len(documents)} documents -> {len(output)} chunks", flush=True)
    metrics = {
        "document_count": len(documents),
        "chunk_count": len(output),
        "structured_document_success_count": structured_success,
        "structured_document_fallback_count": structured_fallback,
        "chunks_with_grounded_media_count": chunks_with_media,
        "media_chunk_association_count": media_chunk_association_count,
        "linked_media_count": len(linked_media_ids),
        "media_match_method_counts": dict(sorted(media_match_method_counts.items())),
        "media_attachment_policy": "exact PDF page; otherwise section hierarchy or strong occurrence-context match; unscoped media remains independent",
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    return output, metrics


def _write_candidate(
    *,
    output_dir: Path,
    config_id: str,
    config: Mapping[str, int],
    chunk_records: Sequence[Dict[str, Any]],
    media_records: Sequence[Dict[str, Any]],
    page_records: Sequence[Dict[str, Any]],
    action_records: Sequence[Dict[str, Any]],
    documents_by_revision: Mapping[str, Mapping[str, Any]],
    page_records_by_revision: Mapping[str, Sequence[Mapping[str, Any]]],
    chunk_metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    parent_records = _parent_records(
        chunk_records,
        config_id=config_id,
        documents_by_revision=documents_by_revision,
        page_records_by_revision=page_records_by_revision,
    )
    records = [*chunk_records, *parent_records, *media_records, *page_records, *action_records]
    seen: set[str] = set()
    for row in records:
        if row["id"] in seen:
            raise RuntimeError(f"Duplicate candidate record id: {row['id']}")
        seen.add(row["id"])
    candidate_dir = output_dir / "candidates" / config_id
    records_path = candidate_dir / "records.jsonl"
    _write_jsonl(records_path, records)
    kind_counts = Counter(row["kind"] for row in records)
    token_counts = [int((row.get("metadata") or {}).get("token_count") or 0) for row in chunk_records]
    record_token_counts = [estimate_token_count(str(row.get("text") or "")) for row in records]
    if any(value >= 8192 for value in record_token_counts):
        raise RuntimeError(
            f"{config_id} contains a record at or above Gemini's 8,192-token input limit"
        )
    stats = {
        "config_id": config_id,
        "candidate_representation_revision": CANDIDATE_REPRESENTATION_REVISION,
        "chunk_config": dict(config),
        "token_counting_method": token_counting_method(),
        "records_path": str(records_path.resolve()),
        "records_sha256": _sha256_file(records_path),
        "record_count": len(records),
        "record_kind_counts": dict(sorted(kind_counts.items())),
        "chunk_token_statistics": {
            "minimum": min(token_counts) if token_counts else 0,
            "maximum": max(token_counts) if token_counts else 0,
            "mean": round(sum(token_counts) / len(token_counts), 3) if token_counts else 0.0,
        },
        "record_text_bytes": sum(len(str(row.get("text") or "").encode("utf-8")) for row in records),
        "record_token_statistics": {
            "maximum": max(record_token_counts) if record_token_counts else 0,
            "at_or_above_gemini_8192_limit": sum(
                value >= 8192 for value in record_token_counts
            ),
        },
        "chunking": dict(chunk_metrics),
    }
    _write_json(candidate_dir / "manifest.json", stats)
    return stats


def prepare(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    representation_path = Path(args.representation).expanduser().resolve()
    media_path = Path(args.media_manifest).expanduser().resolve()
    navigation_catalog_path = Path(args.navigation_catalog).expanduser().resolve()
    corpus_run = Path(args.corpus_run).expanduser().resolve()
    dataset_path = Path(args.dataset).expanduser().resolve()
    representation = _read_json(representation_path)
    media_manifest = _read_json(media_path)
    navigation_catalog = _read_json(navigation_catalog_path)
    examples = load_eval_examples(dataset_path)

    documents = [row for row in representation.get("documents") or [] if isinstance(row, Mapping)]
    documents_by_revision = {
        str(row.get("document_revision_id") or ""): row
        for row in documents
        if str(row.get("document_revision_id") or "")
    }
    documents_by_markdown = {
        str(Path(str(row.get("markdown_path") or "")).expanduser().resolve()): row for row in documents
    }
    page_cards = [row for row in representation.get("page_cards") or [] if isinstance(row, Mapping)]
    pages_by_revision: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    pages_by_url: Dict[str, Mapping[str, Any]] = {}
    for page in page_cards:
        revision_id = str(page.get("document_revision_id") or "")
        if revision_id:
            pages_by_revision[revision_id].append(page)
        normalized_url = _normalize_url(page.get("source_url"))
        if normalized_url:
            pages_by_url[normalized_url] = page
    structured_by_markdown = _load_structured_by_markdown(corpus_run)
    source_url_by_markdown = _source_url_by_markdown(media_manifest)
    sections = _section_lookup(page_cards)
    media_records, media_items_by_document = _prepare_media_records(
        media_manifest,
        documents_by_markdown=documents_by_markdown,
        pages_by_url=pages_by_url,
    )
    page_records, action_records = _prepare_graph_records(
        representation, navigation_catalog
    )
    page_records_by_revision: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for page_record in page_records:
        revision_id = str(page_record.get("document_revision_id") or "")
        if revision_id:
            page_records_by_revision[revision_id].append(page_record)

    query_path = output_dir / "queries.jsonl"
    _write_jsonl(
        query_path,
        [
            {
                "id": row.id,
                "query": row.query,
                "language": row.language,
                "query_type": row.query_type,
                "source_type": row.source_type,
                "no_answer": row.no_answer,
                "split": str((row.metadata or {}).get("split") or ""),
                "benchmark_tags": list((row.metadata or {}).get("benchmark_tags") or []),
            }
            for row in examples
        ],
    )
    candidate_manifests: Dict[str, Dict[str, Any]] = {}
    selected_configs = list(CHUNK_CONFIGS)
    if args.config:
        selected_configs = [value for value in args.config if value in CHUNK_CONFIGS]
        unknown = set(args.config) - set(selected_configs)
        if unknown:
            raise ValueError(f"Unknown chunk config(s): {sorted(unknown)}")
    for config_id in selected_configs:
        existing_manifest_path = (
            output_dir / "candidates" / config_id / "manifest.json"
        )
        if existing_manifest_path.is_file() and not args.force_rebuild:
            existing_manifest = _read_json(existing_manifest_path)
            existing_records_path = Path(
                str(existing_manifest.get("records_path") or "")
            )
            if (
                existing_manifest.get("chunk_config") == CHUNK_CONFIGS[config_id]
                and existing_manifest.get("candidate_representation_revision")
                == CANDIDATE_REPRESENTATION_REVISION
                and existing_records_path.is_file()
                and existing_manifest.get("records_sha256")
                == _sha256_file(existing_records_path)
            ):
                candidate_manifests[config_id] = existing_manifest
                print(
                    f"Reusing hash-validated candidate {config_id}", flush=True
                )
                continue
        print(f"Preparing candidate {config_id}", flush=True)
        chunks, metrics = _chunk_records_for_config(
            config_id=config_id,
            config=CHUNK_CONFIGS[config_id],
            documents=documents,
            pages_by_revision=pages_by_revision,
            section_lookup=sections,
            structured_by_markdown=structured_by_markdown,
            source_url_by_markdown=source_url_by_markdown,
            media_items_by_document=media_items_by_document,
        )
        candidate_manifests[config_id] = _write_candidate(
            output_dir=output_dir,
            config_id=config_id,
            config=CHUNK_CONFIGS[config_id],
            chunk_records=chunks,
            media_records=media_records,
            page_records=page_records,
            action_records=action_records,
            documents_by_revision=documents_by_revision,
            page_records_by_revision=page_records_by_revision,
            chunk_metrics=metrics,
        )

    manifest = {
        "schema_version": EXPERIMENT_SCHEMA,
        "created_at_epoch": int(time.time()),
        "purpose": "Controlled multilingual selection of chunk, embedding, and index configuration",
        "production_mutation_performed": False,
        "preregistration": {
            "variant_results_inspected_before_lock": False,
            "holdout_results_opened_before_finalist_freeze": False,
            "pre_score_validity_amendments": [
                "keep source-connected examples in one evaluation split",
                "replace provider-truncated full-page parents with fixed 4,096-token explicit synopsis windows",
                "compare native Gemini image-plus-caption embeddings with caption-only media embeddings",
                "include frozen-threshold balanced abstention accuracy at 10 percent of the selection score",
                "remove cross-page and unscoped media leakage from chunk embedding text using fail-closed page/section/context matching",
            ],
        },
        "source_snapshot": {
            "representation_bundle": str(representation_path),
            "representation_sha256": _sha256_file(representation_path),
            "media_manifest": str(media_path),
            "media_manifest_sha256": _sha256_file(media_path),
            "navigation_catalog": str(navigation_catalog_path),
            "navigation_catalog_sha256": _sha256_file(navigation_catalog_path),
            "corpus_run": str(corpus_run),
            "document_count": len(documents),
        },
        "evaluation_dataset": {
            "path": str(dataset_path),
            "sha256": _sha256_file(dataset_path),
            "query_count": len(examples),
            "query_file": str(query_path),
            "query_file_sha256": _sha256_file(query_path),
            "holdout_sealed": True,
        },
        "controlled_variables": {
            "chunk_configs": CHUNK_CONFIGS,
            "embedding_specs": EMBEDDING_SPECS,
            "index_modes": INDEX_MODES,
            "fixed_parent_representation": {
                "revision": CANDIDATE_REPRESENTATION_REVISION,
                "page_parent_token_budget": PARENT_EMBEDDING_TOKEN_BUDGET,
                "section_parent_token_budget": PARENT_SECTION_EMBEDDING_TOKEN_BUDGET,
                "budget_tokenizer": token_counting_method(),
                "provider_limit_guard": "all record texts must remain below Gemini's 8,192-token limit",
                "omissions_are_explicitly_marked": True,
            },
            "fixed_media_filter": {
                "annotation_status": "completed",
                "semantic_relevance": "substantive",
                "excluded_image_kinds": ["background", "decorative", "divider", "icon", "logo"],
                "minimum_pixel_area": 90000,
                "deduplicate_by": "content_hash",
                "record_count": len(media_records),
            },
            "fixed_graph_records": {
                "page_cards": len(page_records),
                "retrieval_eligible_non_template_actions": len(action_records),
            },
            "selection_policy": SELECTION_POLICY,
        },
        "candidate_manifests": candidate_manifests,
    }
    manifest_path = output_dir / "experiment_manifest.json"
    _write_json(manifest_path, manifest)
    print(f"Wrote controlled A/B manifest to {manifest_path}", flush=True)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare isolated MBZUAI multilingual A/B candidate corpora")
    parser.add_argument("--representation", default=str(DEFAULT_REPRESENTATION))
    parser.add_argument("--media-manifest", default=str(DEFAULT_MEDIA))
    parser.add_argument("--corpus-run", default=str(DEFAULT_CORPUS_RUN))
    parser.add_argument("--navigation-catalog", default=str(DEFAULT_NAVIGATION_CATALOG))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--config", action="append", choices=sorted(CHUNK_CONFIGS))
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Recompute chunk candidates even when their manifests and hashes validate",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare(args)
    print(json.dumps({"candidate_manifests": manifest["candidate_manifests"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
