"""
Schema-aware semantic graph extraction using Gemini structured output.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from urllib.parse import urlparse

from pipeline.core.artifact_contracts import ArtifactContract, resolve_artifact_path
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.google_genai import import_genai, import_genai_types
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage
from pipeline.core.semantic_graph import clean_text, coerce_confidence, stable_semantic_id, unique_strings

logger = logging.getLogger(__name__)

LOW_SIGNAL_HEADING_TERMS = (
    "abstract",
    "references",
    "table of contents",
    "contents",
    "embedded media",
    "video: youtube video player",
    "related work",
    "introduction",
)

HIGH_SIGNAL_TEXT_TERMS = (
    "located",
    "based in",
    "emirate",
    "named after",
    "carry the name",
    "offers",
    "provides",
    "available",
    "available on campus",
    "housing",
    "accommodation",
    "parking",
    "shuttle",
    "transportation",
    "hours",
    "weekday",
    "weekend",
    "law",
    "decree",
    "policy",
    "regulation",
    "affiliated",
    "program",
    "specialization",
    "tuition",
    "scholarship",
    "admissions",
    "campus",
    "facilities",
    "support",
    "services",
    "library",
    "student life",
    "contact",
)

HIGH_SIGNAL_HEADING_TERMS = (
    "faq",
    "about",
    "contact",
    "admissions",
    "program",
    "campus",
    "facilities",
    "housing",
    "parking",
    "hours",
    "policy",
    "law",
    "regulation",
    "support",
    "services",
    "student",
    "transport",
    "shuttle",
    "map",
)


def _make_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for semantic graph extraction")
    genai = import_genai()
    return genai.Client(api_key=api_key)


def _json_schema(*, allowed_entity_types: Sequence[str], allowed_relation_types: Sequence[str]) -> Dict[str, Any]:
    entity_type_enum = list(dict.fromkeys([str(value) for value in allowed_entity_types if str(value)])) or ["Other"]
    relation_type_enum = list(dict.fromkeys([str(value) for value in allowed_relation_types if str(value)])) or ["RELATED_TO"]
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "entity_type": {"type": "string", "enum": entity_type_enum},
                                    "aliases": {"type": "array", "items": {"type": "string"}},
                                    "description": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["name", "entity_type", "confidence"],
                            },
                        },
                        "relations": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "subject_name": {"type": "string"},
                                    "subject_type": {"type": "string", "enum": entity_type_enum},
                                    "relation_type": {"type": "string", "enum": relation_type_enum},
                                    "object_name": {"type": "string"},
                                    "object_type": {"type": "string", "enum": entity_type_enum},
                                    "evidence": {"type": "string"},
                                    "confidence": {"type": "number"},
                                },
                                "required": ["subject_name", "subject_type", "relation_type", "object_name", "object_type", "confidence"],
                            },
                        },
                    },
                    "required": ["source_id", "entities", "relations"],
                },
            }
        },
        "required": ["items"],
    }


def _build_prompt(batch: Sequence[Dict[str, Any]], *, allowed_entity_types: Sequence[str], allowed_relation_types: Sequence[str]) -> str:
    entity_types = ", ".join(allowed_entity_types)
    relation_types = ", ".join(allowed_relation_types)
    payload = []
    for item in batch:
        obj = {
            "source_id": item["source_id"],
            "source_kind": item["source_kind"],
            "heading": item.get("heading", ""),
            "document_title": item.get("document_title", ""),
            "section_path": item.get("section_path", []),
            "text": item.get("text", ""),
        }
        if item.get("gliner_entities"):
            obj["pre_extracted_entities"] = [
                {"name": e["name"], "type": e["entity_type"]}
                for e in item["gliner_entities"]
            ]
        payload.append(obj)

    return (
        "Extract high-confidence entities and relations from the supplied document evidence.\n"
        "Only extract what is explicitly supported by the text.\n"
        "Return one object per source item.\n"
        "Use only these entity types when possible: "
        f"{entity_types}.\n"
        "Use only these relation types when possible: "
        f"{relation_types}.\n"
        "Some items have 'pre_extracted_entities' (from zero-shot models). Use them as hints, verify them, and extract complex relations between them and any other entities you find.\n"
        "If no supported entities or relations exist for an item, return empty arrays for that item.\n"
        "Keep confidence between 0 and 1.\n\n"
        f"INPUT:\n{json.dumps(payload, ensure_ascii=False)}"
    )


def _call_gemini_structured(
    *,
    client: Any,
    model: str,
    prompt: str,
    schema: Dict[str, Any],
    temperature: float,
) -> Dict[str, Any]:
    if client is None:
        client = _make_gemini_client()
    types = import_genai_types()

    config = types.GenerateContentConfig(
        temperature=temperature,
        responseMimeType="application/json",
        responseSchema=schema,
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )
    text = str(getattr(response, "text", "") or "").strip()
    if not text:
        return {"items": []}
    payload = json.loads(text)
    return payload if isinstance(payload, dict) else {"items": []}


def _is_retryable_extraction_exception(exc: Exception) -> bool:
    text = str(exc or "").lower()
    retryable_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "unavailable",
        "resource_exhausted",
        "deadline exceeded",
        "rate limit",
        "timeout",
        "temporarily unavailable",
        "connection reset",
    )
    return any(marker in text for marker in retryable_markers)


def _extract_batch_with_retry(
    *,
    client: Any,
    model: str,
    batch: Sequence[Dict[str, Any]],
    schema: Dict[str, Any],
    temperature: float,
    allowed_entity_types: Sequence[str],
    allowed_relation_types: Sequence[str],
    max_attempts: int,
    base_delay_sec: float,
    max_delay_sec: float,
    split_on_retryable_failure: bool,
    fail_open_after_retries: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    prompt = _build_prompt(
        batch,
        allowed_entity_types=allowed_entity_types,
        allowed_relation_types=allowed_relation_types,
    )
    delay = max(0.1, float(base_delay_sec))
    for attempt in range(1, max_attempts + 1):
        try:
            return (
                _call_gemini_structured(
                    client=client,
                    model=model,
                    prompt=prompt,
                    schema=schema,
                    temperature=temperature,
                ),
                [],
            )
        except Exception as exc:
            retryable = _is_retryable_extraction_exception(exc)
            if retryable and attempt < max_attempts:
                logger.warning(
                    "Semantic graph extraction batch failed with retryable error on attempt %d/%d: %s. Retrying in %.1fs",
                    attempt,
                    max_attempts,
                    exc,
                    min(delay, max_delay_sec),
                )
                time.sleep(min(delay, max_delay_sec))
                delay = min(delay * 2.0, max_delay_sec)
                continue
            if retryable and split_on_retryable_failure and len(batch) > 1:
                midpoint = max(1, len(batch) // 2)
                left_payload, left_errors = _extract_batch_with_retry(
                    client=client,
                    model=model,
                    batch=batch[:midpoint],
                    schema=schema,
                    temperature=temperature,
                    allowed_entity_types=allowed_entity_types,
                    allowed_relation_types=allowed_relation_types,
                    max_attempts=max_attempts,
                    base_delay_sec=base_delay_sec,
                    max_delay_sec=max_delay_sec,
                    split_on_retryable_failure=split_on_retryable_failure,
                    fail_open_after_retries=fail_open_after_retries,
                )
                right_payload, right_errors = _extract_batch_with_retry(
                    client=client,
                    model=model,
                    batch=batch[midpoint:],
                    schema=schema,
                    temperature=temperature,
                    allowed_entity_types=allowed_entity_types,
                    allowed_relation_types=allowed_relation_types,
                    max_attempts=max_attempts,
                    base_delay_sec=base_delay_sec,
                    max_delay_sec=max_delay_sec,
                    split_on_retryable_failure=split_on_retryable_failure,
                    fail_open_after_retries=fail_open_after_retries,
                )
                return (
                    {
                        "items": [
                            *(left_payload.get("items") or []),
                            *(right_payload.get("items") or []),
                        ]
                    },
                    [*left_errors, *right_errors],
                )
            if fail_open_after_retries:
                error_entry = {
                    "batch_source_ids": [str(item.get("source_id") or "") for item in batch if str(item.get("source_id") or "")],
                    "error": str(exc),
                    "retryable": retryable,
                }
                logger.warning(
                    "Semantic graph extraction failed open for %d source(s): %s",
                    len(error_entry["batch_source_ids"]),
                    exc,
                )
                return (
                    {
                        "items": [
                            {"source_id": str(item.get("source_id") or ""), "entities": [], "relations": []}
                            for item in batch
                        ]
                    },
                    [error_entry],
                )
            raise
    return {"items": []}, []


def _select_items_for_extraction(
    items: Sequence[Dict[str, Any]],
    *,
    extraction_max_items: int,
    extraction_max_items_per_document: int,
    extraction_min_score: float = 0.0,
    priority_host_suffixes: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    ordered_items = [item for item in items if item.get("source_id")]
    if extraction_max_items_per_document <= 0 and extraction_max_items <= 0:
        return ordered_items

    priority_host_suffixes = tuple(
        suffix.lower().lstrip(".")
        for suffix in (priority_host_suffixes or [])
        if str(suffix).strip()
    )

    def _group_key(item: Dict[str, Any]) -> str:
        return (
            str(item.get("document_id") or "").strip()
            or str(item.get("document_title") or "").strip()
            or str(item.get("source_url") or "").strip()
            or f"{item.get('source_kind','item')}:{item.get('source_id','')}"
        )

    def _url_host(url: str) -> str:
        try:
            return (urlparse(url).netloc or "").lower().strip()
        except Exception:
            return ""

    def _matches_priority_host(host: str) -> bool:
        if not host:
            return False
        normalized = host.lower().lstrip(".")
        return any(
            normalized == suffix or normalized.endswith(f".{suffix}")
            for suffix in priority_host_suffixes
        )

    def _selection_score(item: Dict[str, Any]) -> float:
        text = clean_text(item.get("text") or "").lower()
        heading = clean_text(item.get("heading") or "").lower()
        document_title = clean_text(item.get("document_title") or "").lower()
        section_path = " ".join(clean_text(part).lower() for part in item.get("section_path") or [])
        source_url = clean_text(item.get("source_url") or "")
        host = _url_host(source_url)
        path_terms = f"{source_url.lower()} {section_path}".strip()

        score = 0.0
        if item.get("source_kind") == "fact":
            score += 0.5
        if source_url:
            score += 0.5
        if _matches_priority_host(host):
            score += 1.5
        elif host:
            score += 0.5

        token_count = len(text.split())
        if 6 <= token_count <= 45:
            score += 1.5
        elif 46 <= token_count <= 80:
            score += 0.5
        elif token_count > 100:
            score -= 1.5

        if "?" in text:
            score += 2.0
        if ":" in text and token_count <= 32:
            score += 0.5

        combined_heading = f"{heading} {section_path} {document_title}".strip()
        heading_hits = sum(1 for term in HIGH_SIGNAL_HEADING_TERMS if term in combined_heading or term in path_terms)
        if heading_hits:
            score += min(1.5 + (heading_hits - 1) * 0.5, 3.0)
        if heading.startswith("page ") or document_title.startswith("page "):
            score -= 2.0
        if heading.startswith("video:") or "youtube" in heading:
            score -= 5.0
        if "xml sitemap" in combined_heading or "xml sitemap" in path_terms:
            score -= 5.0
        if any(term in heading for term in LOW_SIGNAL_HEADING_TERMS):
            score -= 3.0
        if any(term in document_title for term in LOW_SIGNAL_HEADING_TERMS):
            score -= 1.5

        high_signal_hits = sum(1 for term in HIGH_SIGNAL_TEXT_TERMS if term in text)
        score += min(high_signal_hits * 1.5, 6.0)
        if heading == "":
            score -= 1.0
        if heading == "" and section_path == "":
            score -= 1.0

        if text.startswith("abstract "):
            score -= 2.0
        if high_signal_hits == 0 and heading_hits == 0 and "?" not in text:
            score -= 3.0
        if source_url == "" and high_signal_hits == 0 and heading_hits == 0 and "faq" not in combined_heading:
            score -= 2.0

        return score

    grouped: Dict[str, deque[Dict[str, Any]]] = defaultdict(deque)
    group_order: List[str] = []
    seen_groups: set[str] = set()
    group_scores: Dict[str, float] = defaultdict(float)
    grouped_items: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in ordered_items:
        item["_selection_score"] = _selection_score(item)
        group_key = _group_key(item)
        if group_key not in seen_groups:
            seen_groups.add(group_key)
            group_order.append(group_key)
        grouped_items[group_key].append(item)
        group_scores[group_key] = max(group_scores[group_key], float(item["_selection_score"]))

    for group_key in group_order:
        prioritized = sorted(
            grouped_items[group_key],
            key=lambda value: (
                -float(value.get("_selection_score") or 0.0),
                str(value.get("source_id") or ""),
            ),
        )
        if extraction_min_score > 0:
            high_signal = [item for item in prioritized if float(item.get("_selection_score") or 0.0) >= extraction_min_score]
            prioritized = high_signal
        grouped[group_key].extend(prioritized)

    selected: List[Dict[str, Any]] = []
    per_group_counts: Dict[str, int] = defaultdict(int)
    group_order_index = {group_key: index for index, group_key in enumerate(group_order)}
    active_groups = deque(
        sorted(
            [group_key for group_key in group_order if grouped.get(group_key)],
            key=lambda group_key: (-group_scores.get(group_key, 0.0), group_order_index.get(group_key, 0)),
        )
    )
    max_total = extraction_max_items if extraction_max_items > 0 else None

    while active_groups:
        group_key = active_groups.popleft()
        queue = grouped.get(group_key)
        if not queue:
            continue
        if extraction_max_items_per_document > 0 and per_group_counts[group_key] >= extraction_max_items_per_document:
            continue
        selected.append(queue.popleft())
        per_group_counts[group_key] += 1
        if max_total is not None and len(selected) >= max_total:
            break
        if queue and (
            extraction_max_items_per_document <= 0
            or per_group_counts[group_key] < extraction_max_items_per_document
        ):
            active_groups.append(group_key)

    return selected


def _load_resume_cache(ctx: StageContext, cache_file: Path) -> Dict[str, Any]:
    cache_payload = load_json_safe(cache_file, {}) or {}
    if isinstance(cache_payload, dict) and cache_payload:
        return cache_payload
    stale_root = Path(ctx.work_dir) / "stage_outputs" / "_stale"
    if not stale_root.is_dir():
        return {}
    candidates = sorted(
        stale_root.glob("extract_semantic_graph_*/semantic_extraction_cache.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        payload = load_json_safe(candidate, {}) or {}
        if isinstance(payload, dict) and payload:
            logger.info(
                "Restoring semantic extraction cache from stale output: %s (%d entries)",
                candidate,
                len(payload),
            )
            atomic_write_json(cache_file, payload)
            return payload
    return {}


def _normalize_candidate_item(raw: Dict[str, Any], source_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    source_id = clean_text(raw.get("source_id"))
    source = source_lookup.get(source_id) or {}
    entities: List[Dict[str, Any]] = []
    relations: List[Dict[str, Any]] = []

    for entity in raw.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = clean_text(entity.get("name"))
        entity_type = clean_text(entity.get("entity_type")) or "Other"
        if not name:
            continue
        entities.append(
            {
                "id": stable_semantic_id("candidate_entity", source_id, entity_type, name),
                "source_id": source_id,
                "source_kind": source.get("source_kind", ""),
                "name": name,
                "entity_type": entity_type,
                "aliases": unique_strings(entity.get("aliases") or []),
                "description": clean_text(entity.get("description")),
                "confidence": coerce_confidence(entity.get("confidence"), default=0.5),
                "source_chunk_ids": list(source.get("source_chunk_ids") or []),
                "source_fact_ids": list(source.get("source_fact_ids") or []),
                "source_parent_ids": list(source.get("source_parent_ids") or []),
                "source_url": source.get("source_url", ""),
                "document_title": source.get("document_title", ""),
            }
        )

    for relation in raw.get("relations") or []:
        if not isinstance(relation, dict):
            continue
        subject_name = clean_text(relation.get("subject_name"))
        subject_type = clean_text(relation.get("subject_type")) or "Other"
        relation_type = clean_text(relation.get("relation_type")) or "OTHER"
        object_name = clean_text(relation.get("object_name"))
        object_type = clean_text(relation.get("object_type")) or "Other"
        if not subject_name or not object_name or not relation_type:
            continue
        relations.append(
            {
                "id": stable_semantic_id(
                    "candidate_relation",
                    source_id,
                    subject_type,
                    subject_name,
                    relation_type,
                    object_type,
                    object_name,
                ),
                "source_id": source_id,
                "source_kind": source.get("source_kind", ""),
                "subject_name": subject_name,
                "subject_type": subject_type,
                "relation_type": relation_type,
                "object_name": object_name,
                "object_type": object_type,
                "evidence": clean_text(relation.get("evidence")) or clean_text(source.get("text")),
                "confidence": coerce_confidence(relation.get("confidence"), default=0.5),
                "source_chunk_ids": list(source.get("source_chunk_ids") or []),
                "source_fact_ids": list(source.get("source_fact_ids") or []),
                "source_parent_ids": list(source.get("source_parent_ids") or []),
                "source_url": source.get("source_url", ""),
                "document_title": source.get("document_title", ""),
            }
        )
    return {
        "source_id": source_id,
        "entities": entities,
        "relations": relations,
    }


@register_stage
class SemanticGraphExtractFormatter(FormatterStage):
    name = "semantic_graph_extract"
    description = "Extracts candidate semantic entities and relations from retrieval evidence using Gemini."

    async def execute(self, ctx: StageContext) -> StageResult:
        retrieval_bundle = resolve_artifact_path(
            ctx,
            ArtifactContract(
                artifact_type="retrieval_bundle",
                role="retrieval_corpus",
                legacy_output_key="retrieval_bundle_file",
                label="retrieval bundle",
            ),
        )
        retrieval_bundle_file = retrieval_bundle.path if retrieval_bundle else ""
        if not retrieval_bundle_file:
            return StageResult.failure("No retrieval_bundle available for semantic graph extraction")

        bundle = load_json_safe(retrieval_bundle_file, {}) or {}
        if not isinstance(bundle, dict):
            return StageResult.failure("Invalid retrieval bundle payload for semantic graph extraction")

        graph_cfg = dict(ctx.graph_config or {})
        extraction_provider = str(graph_cfg.get("extraction_provider") or "gemini").strip().lower()
        if extraction_provider != "gemini":
            return StageResult.failure(f"Unsupported semantic graph extraction provider: {extraction_provider}")
        source_mode = str(graph_cfg.get("extraction_source") or "facts_first")
        batch_size = max(1, int(graph_cfg.get("extraction_batch_size") or 6))
        extraction_max_items = max(0, int(graph_cfg.get("extraction_max_items") or 0))
        extraction_max_items_per_document = max(
            0, int(graph_cfg.get("extraction_max_items_per_document") or 0)
        )
        extraction_parallelism = max(1, int(graph_cfg.get("extraction_parallelism") or 1))
        extraction_min_score = float(graph_cfg.get("extraction_min_score") or 0.0)
        priority_host_suffixes = list(graph_cfg.get("extraction_priority_host_suffixes") or [])
        min_confidence = float(graph_cfg.get("extraction_min_confidence") or 0.55)
        retry_attempts = max(1, int(graph_cfg.get("extraction_retry_attempts") or 5))
        retry_base_delay_sec = max(0.1, float(graph_cfg.get("extraction_retry_base_delay_sec") or 5.0))
        retry_max_delay_sec = max(retry_base_delay_sec, float(graph_cfg.get("extraction_retry_max_delay_sec") or 120.0))
        split_on_retryable_failure = bool(graph_cfg.get("extraction_split_on_retryable_failure", True))
        fail_open_after_retries = bool(graph_cfg.get("extraction_fail_open_after_retries", True))
        model = str(graph_cfg.get("extraction_model") or "gemini-2.5-flash")
        temperature = float(graph_cfg.get("extraction_temperature") or 0.0)
        allowed_entity_types = list(graph_cfg.get("allowed_entity_types") or ["Other"])
        allowed_relation_types = list(graph_cfg.get("allowed_relation_types") or ["RELATED_TO"])

        items: List[Dict[str, Any]] = []
        if source_mode in {"facts_first", "facts_only"}:
            for record in bundle.get("fact_records") or []:
                if not isinstance(record, dict):
                    continue
                text = clean_text(record.get("text") or record.get("dense_text"))
                if not text:
                    continue
                items.append(
                    {
                        "source_id": str(record.get("id") or ""),
                        "source_kind": "fact",
                        "text": text,
                        "heading": clean_text(record.get("heading")),
                        "document_id": clean_text(record.get("document_id")),
                        "document_title": clean_text(record.get("document_title")),
                        "section_path": list(record.get("section_path") or []),
                        "source_chunk_ids": list(record.get("linked_chunk_ids") or []),
                        "source_fact_ids": [str(record.get("id") or "")],
                        "source_parent_ids": list(record.get("linked_parent_ids") or []),
                        "source_url": record.get("source_url", ""),
                    }
                )
        if source_mode in {"chunks_only", "facts_and_chunks"} or (
            source_mode == "facts_first" and bool(graph_cfg.get("extraction_include_chunk_fallback", True))
        ):
            for record in bundle.get("chunk_records") or []:
                if not isinstance(record, dict):
                    continue
                text = clean_text(record.get("text"))
                if len(text) < 80:
                    continue
                items.append(
                    {
                        "source_id": str(record.get("id") or ""),
                        "source_kind": "chunk",
                        "text": text,
                        "heading": clean_text(record.get("heading")),
                        "document_id": clean_text(record.get("document_id")),
                        "document_title": clean_text(record.get("document_title")),
                        "section_path": list(record.get("section_path") or []),
                        "source_chunk_ids": [str(record.get("id") or "")],
                        "source_fact_ids": [],
                        "source_parent_ids": [str(record.get("section_key") or ""), str(record.get("page_key") or "")],
                        "source_url": record.get("source_url", ""),
                    }
                )
        gliner_entities_file = ctx.previous_outputs.get("gliner_entities_file")
        if not gliner_entities_file:
            gliner_artifacts = ctx.find_artifacts(artifact_type="gliner_entities")
            if gliner_artifacts and gliner_artifacts[-1].local_path:
                gliner_entities_file = gliner_artifacts[-1].local_path

        gliner_entities = {}
        if gliner_entities_file:
            gliner_entities = load_json_safe(gliner_entities_file, {}) or {}

        for item in items:
            item["gliner_entities"] = gliner_entities.get(item["source_id"]) or []


        source_lookup = {item["source_id"]: item for item in items if item.get("source_id")}
        ordered_items = _select_items_for_extraction(
            items,
            extraction_max_items=extraction_max_items,
            extraction_max_items_per_document=extraction_max_items_per_document,
            extraction_min_score=extraction_min_score,
            priority_host_suffixes=priority_host_suffixes,
        )

        cache_file = ctx.stage_work_dir / "semantic_extraction_cache.json"
        if bool(graph_cfg.get("extraction_resume_cache", True)):
            cache_payload = _load_resume_cache(ctx, cache_file)
        else:
            cache_payload = load_json_safe(cache_file, {}) or {}
        if not isinstance(cache_payload, dict):
            cache_payload = {}

        uncached: List[Dict[str, Any]] = []
        for item in ordered_items:
            cache_key = stable_semantic_id("semantic_extract", item["source_id"], item["text"])
            item["cache_key"] = cache_key
            cached = cache_payload.get(cache_key)
            if isinstance(cached, dict):
                continue
            uncached.append(item)

        schema = _json_schema(
            allowed_entity_types=allowed_entity_types,
            allowed_relation_types=allowed_relation_types,
        )
        extraction_errors: List[Dict[str, Any]] = []
        batches = [uncached[start : start + batch_size] for start in range(0, len(uncached), batch_size)]

        def _store_batch_results(batch: Sequence[Dict[str, Any]], payload: Dict[str, Any], batch_errors: List[Dict[str, Any]]) -> None:
            extraction_errors.extend(batch_errors)
            raw_items = payload.get("items") or []
            normalized_items = {
                item["source_id"]: _normalize_candidate_item(item, source_lookup)
                for item in raw_items
                if isinstance(item, dict) and clean_text(item.get("source_id"))
            }
            for item in batch:
                cache_payload[item["cache_key"]] = normalized_items.get(
                    item["source_id"],
                    {"source_id": item["source_id"], "entities": [], "relations": []},
                )
            atomic_write_json(cache_file, cache_payload)

        def _process_batch(batch: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
            client = _make_gemini_client()
            payload, batch_errors = _extract_batch_with_retry(
                client=client,
                model=model,
                batch=batch,
                schema=schema,
                temperature=temperature,
                allowed_entity_types=allowed_entity_types,
                allowed_relation_types=allowed_relation_types,
                max_attempts=retry_attempts,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
                split_on_retryable_failure=split_on_retryable_failure,
                fail_open_after_retries=fail_open_after_retries,
            )
            return list(batch), payload, batch_errors

        if extraction_parallelism <= 1 or len(batches) <= 1:
            for batch in batches:
                batch, payload, batch_errors = _process_batch(batch)
                _store_batch_results(batch, payload, batch_errors)
        else:
            with ThreadPoolExecutor(max_workers=extraction_parallelism) as executor:
                futures = [executor.submit(_process_batch, batch) for batch in batches]
                for future in as_completed(futures):
                    batch, payload, batch_errors = future.result()
                    _store_batch_results(batch, payload, batch_errors)

        candidate_entities: List[Dict[str, Any]] = []
        candidate_relations: List[Dict[str, Any]] = []
        extraction_records: List[Dict[str, Any]] = []

        seen_entity_ids = set()
        for item in ordered_items:
            extracted = cache_payload.get(item["cache_key"]) or {"source_id": item["source_id"], "entities": [], "relations": []}
            extraction_records.append(extracted)

            for entity in item.get("gliner_entities") or []:
                if entity["id"] not in seen_entity_ids:
                    seen_entity_ids.add(entity["id"])
                    candidate_entities.append(entity)

            for entity in extracted.get("entities") or []:
                if coerce_confidence(entity.get("confidence"), default=0.0) >= min_confidence:
                    if entity["id"] not in seen_entity_ids:
                        seen_entity_ids.add(entity["id"])
                        candidate_entities.append(entity)
            for relation in extracted.get("relations") or []:
                if coerce_confidence(relation.get("confidence"), default=0.0) >= min_confidence:
                    candidate_relations.append(relation)

        entities_file = ctx.stage_work_dir / "candidate_entities.json"
        relations_file = ctx.stage_work_dir / "candidate_relations.json"
        extraction_file = ctx.stage_work_dir / "semantic_extraction.json"
        errors_file = ctx.stage_work_dir / "semantic_extraction_errors.json"
        atomic_write_json(entities_file, candidate_entities)
        atomic_write_json(relations_file, candidate_relations)
        atomic_write_json(errors_file, extraction_errors)
        atomic_write_json(
            extraction_file,
            {
                "provider": extraction_provider,
                "model": model,
                "source_count": len(ordered_items),
                "entity_count": len(candidate_entities),
                "relation_count": len(candidate_relations),
                "error_count": len(extraction_errors),
                "records": extraction_records,
            },
        )

        artifacts = [
            ctx.make_artifact(
                extraction_file,
                artifact_type="semantic_graph_extraction",
                role="semantic_graph_candidates",
                metadata={
                    "sources": len(ordered_items),
                    "entities": len(candidate_entities),
                    "relations": len(candidate_relations),
                    "provider": extraction_provider,
                    "model": model,
                },
            ),
            ctx.make_artifact(
                entities_file,
                artifact_type="semantic_entity_candidates",
                role="semantic_graph_candidates",
                metadata={"records": len(candidate_entities)},
            ),
            ctx.make_artifact(
                relations_file,
                artifact_type="semantic_relation_candidates",
                role="semantic_graph_candidates",
                metadata={"records": len(candidate_relations)},
            ),
            ctx.make_artifact(
                errors_file,
                artifact_type="semantic_graph_extraction_errors",
                role="semantic_graph_candidates",
                metadata={"records": len(extraction_errors)},
            ),
        ]

        logger.info(
            "Semantic graph extraction: %d sources, %d candidate entities, %d candidate relations",
            len(ordered_items),
            len(candidate_entities),
            len(candidate_relations),
        )

        return StageResult.success(
            outputs={
                "semantic_extraction_file": str(extraction_file),
                "semantic_entity_candidates_file": str(entities_file),
                "semantic_relation_candidates_file": str(relations_file),
            },
            metrics={
                "semantic_sources": len(ordered_items),
                "semantic_candidate_entities": len(candidate_entities),
                "semantic_candidate_relations": len(candidate_relations),
                "semantic_cache_entries": len(cache_payload),
                "semantic_failed_sources": sum(len(item.get("batch_source_ids") or []) for item in extraction_errors),
                "semantic_extraction_errors": len(extraction_errors),
                "semantic_parallelism": extraction_parallelism,
            },
            artifacts=artifacts,
        )
