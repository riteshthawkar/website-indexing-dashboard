from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from pipeline.core.assertions import (
    authority_score,
    build_assertion_record,
    build_entity_record,
    clean_text,
    coerce_confidence,
    infer_authority_class,
    unique_strings,
)
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.openai_client import json_completion, make_openai_client
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

_EXTRACTION_JSON_SCHEMA = {
    "name": "assertion_extraction",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "name": {"type": "string"},
                        "entity_type": {"type": "string"},
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "description": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["name", "entity_type", "aliases", "description", "confidence"],
                },
            },
            "assertions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "subject": {"type": "string"},
                        "subject_type": {"type": "string"},
                        "predicate": {"type": "string"},
                        "answer_type": {"type": "string"},
                        "answer_subtype": {"type": "string"},
                        "object": {"type": "string"},
                        "object_type": {"type": "string"},
                        "qualifiers": {"type": "array", "items": {"type": "string"}},
                        "support_span": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": [
                        "subject",
                        "subject_type",
                        "predicate",
                        "answer_type",
                        "answer_subtype",
                        "object",
                        "object_type",
                        "qualifiers",
                        "support_span",
                        "confidence",
                    ],
                },
            },
            "quality_flags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["entities", "assertions", "quality_flags"],
    },
}


_EXTRACTION_SYSTEM_PROMPT = """
You extract grounded entities and machine-usable assertions from an institutional source slice.

Rules:
- Return a single JSON object only.
- Always return the keys "entities", "assertions", and "quality_flags".
- Never return {}.
- If the slice has no grounded items, return:
  {"entities": [], "assertions": [], "quality_flags": ["no_supported_assertions"]}
- Extract only claims explicitly supported by the provided source text.
- Prefer short, typed assertions over prose summaries.
- Extract stable factual information that is useful for retrieval and knowledge graphs.
- Ignore marketing language, testimonials, slogans, and vague claims of excellence.
- Preserve exact subject and object wording when possible.
- If the slice does not support a claim, do not invent one.
- support_span must be a short verbatim excerpt from the source.
- Always extract obvious people, organizations, locations, laws, and named programs into "entities".
- Prioritize assertions about:
  - role assignments
  - contact points
  - operating hours
  - dates and deadlines
  - locations
  - named-after relations
  - affiliations
  - legal basis
  - service availability
  - program areas
  - other explicit stable institutional facts from catalogues or brochures
- Use these predicates when applicable:
  - "role_holder"
  - "email"
  - "phone"
  - "website"
  - "hours"
  - "date"
  - "location"
  - "named_after"
  - "affiliation"
  - "legal_basis"
  - "service_availability"
  - "program_area"
- If a grounded stable fact does not fit the predicate list, use predicate "assertion" and make answer_subtype a short machine-readable label such as "duration", "format", "credential", "tuition", or "target_audience".

Example:
If the text says "Professor Eric Xing is the President of MBZUAI", extract:
{
  "subject": "MBZUAI",
  "subject_type": "organization",
  "predicate": "role_holder",
  "answer_type": "role_holder",
  "answer_subtype": "president",
  "object": "Professor Eric Xing",
  "object_type": "person",
  "qualifiers": ["current"],
  "support_span": "Professor Eric Xing is the President of MBZUAI",
  "confidence": 0.95
}

JSON schema:
{
  "entities": [
    {
      "name": "string",
      "entity_type": "person|organization|location|law|service|other",
      "aliases": ["string"],
      "description": "string or empty",
      "confidence": 0.0
    }
  ],
  "assertions": [
    {
      "subject": "string",
      "subject_type": "person|organization|location|law|service|other",
      "predicate": "string",
      "answer_type": "string",
      "answer_subtype": "string",
      "object": "string",
      "object_type": "person|organization|location|law|service|email|phone|website|hours|date|other",
      "qualifiers": ["string"],
      "support_span": "string",
      "confidence": 0.0
    }
  ],
  "quality_flags": ["string"]
}
"""


def _extract_prompt(slice_record: Dict[str, Any]) -> str:
    section_paths = [
        " > ".join(path)
        for path in (slice_record.get("section_paths") or [])
        if isinstance(path, list) and path
    ]
    metadata_lines = [
        f"TITLE: {clean_text(slice_record.get('document_title'))}",
        f"TYPE: {clean_text(slice_record.get('document_type'))}",
        f"SOURCE_URL: {clean_text(slice_record.get('source_url'))}",
        f"SECTION_ANCHOR: {' > '.join(slice_record.get('section_anchor') or [])}",
        f"SECTION_PATHS: {' || '.join(section_paths)}",
        f"PAGES: {', '.join(str(v) for v in (slice_record.get('page_numbers') or []))}",
        "",
        "SOURCE TEXT:",
        clean_text(slice_record.get("text")),
    ]
    return "\n".join(metadata_lines).strip()


def _normalize_entities(slice_record: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_chunk_ids = slice_record.get("linked_chunk_ids") or []
    source_parent_ids: List[str] = []
    source_urls = [slice_record.get("source_url")] if slice_record.get("source_url") else []
    document_titles = [slice_record.get("document_title")] if slice_record.get("document_title") else []
    entities: List[Dict[str, Any]] = []
    for entity in payload.get("entities") or []:
        if not isinstance(entity, dict):
            continue
        name = clean_text(entity.get("name"))
        if not name:
            continue
        entities.append(
            build_entity_record(
                canonical_name=name,
                entity_type=entity.get("entity_type") or "other",
                aliases=entity.get("aliases") or [],
                description=clean_text(entity.get("description")),
                confidence=entity.get("confidence"),
                source_chunk_ids=source_chunk_ids,
                source_parent_ids=source_parent_ids,
                source_urls=source_urls,
                document_titles=document_titles,
            )
        )
    return entities


def _normalize_assertions(slice_record: Dict[str, Any], payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    assertions: List[Dict[str, Any]] = []
    authority_class = clean_text(slice_record.get("authority_class")) or infer_authority_class(
        source_url=slice_record.get("source_url") or "",
        document_title=slice_record.get("document_title") or "",
        document_type=slice_record.get("document_type") or "",
        source_markdown_path=slice_record.get("source_markdown_path") or "",
    )
    authority_value = coerce_confidence(slice_record.get("authority_score"), default=authority_score(
        source_url=slice_record.get("source_url") or "",
        document_title=slice_record.get("document_title") or "",
        document_type=slice_record.get("document_type") or "",
        source_markdown_path=slice_record.get("source_markdown_path") or "",
    ))
    for item in payload.get("assertions") or []:
        if not isinstance(item, dict):
            continue
        subject = clean_text(item.get("subject"))
        obj = clean_text(item.get("object"))
        predicate = clean_text(item.get("predicate") or item.get("answer_type"))
        if not subject or not obj or not predicate:
            continue
        answer_type = clean_text(item.get("answer_type"))
        if not answer_type or answer_type.lower() in {"other", "string"}:
            answer_type = predicate
        assertions.append(
            build_assertion_record(
                subject_name=subject,
                predicate=predicate,
                answer_type=answer_type,
                answer_subtype=item.get("answer_subtype") or "",
                object_value=obj,
                subject_type=item.get("subject_type") or "organization",
                object_type=item.get("object_type") or "",
                qualifiers=unique_strings(item.get("qualifiers") or []),
                support_span=item.get("support_span") or "",
                confidence=item.get("confidence"),
                authority_class=authority_class,
                authority_score_value=authority_value,
                source_doc_id=slice_record.get("document_id") or "",
                source_chunk_ids=slice_record.get("linked_chunk_ids") or [],
                source_parent_ids=[],
                source_url=slice_record.get("source_url") or "",
                source_markdown_path=slice_record.get("source_markdown_path") or "",
                document_title=slice_record.get("document_title") or "",
                validator_decision="unvalidated",
            )
        )
    return assertions


def _extract_one(
    *,
    slice_record: Dict[str, Any],
    client: Any | None,
    model: str,
    reasoning_effort: str,
    temperature: float,
    max_completion_tokens: int,
    retries: int,
    retry_delay_sec: float,
    per_request_delay_sec: float,
) -> Dict[str, Any]:
    payload = json_completion(
        client=client,
        model=model,
        system_prompt=_EXTRACTION_SYSTEM_PROMPT,
        user_prompt=_extract_prompt(slice_record),
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        json_schema=_EXTRACTION_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
        retries=retries,
        retry_delay_sec=retry_delay_sec,
        per_request_delay_sec=per_request_delay_sec,
    )
    return {
        "slice_id": clean_text(slice_record.get("id")),
        "quality_flags": unique_strings(payload.get("quality_flags") or []),
        "entities": _normalize_entities(slice_record, payload),
        "assertions": _normalize_assertions(slice_record, payload),
        "raw_payload": payload,
    }


def _filter_slices(slices: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    allowed_authority_classes = {
        clean_text(value)
        for value in (cfg.get("allowed_authority_classes") or [])
        if clean_text(value)
    }
    excluded_authority_classes = {
        clean_text(value)
        for value in (cfg.get("excluded_authority_classes") or [])
        if clean_text(value)
    }
    allowed_document_types = {
        clean_text(value)
        for value in (cfg.get("allowed_document_types") or [])
        if clean_text(value)
    }

    filtered: List[Dict[str, Any]] = []
    for slice_record in slices:
        if not isinstance(slice_record, dict):
            continue
        authority_class = clean_text(slice_record.get("authority_class"))
        document_type = clean_text(slice_record.get("document_type"))
        if allowed_authority_classes and authority_class not in allowed_authority_classes:
            continue
        if excluded_authority_classes and authority_class in excluded_authority_classes:
            continue
        if allowed_document_types and document_type not in allowed_document_types:
            continue
        filtered.append(slice_record)
    return filtered


@register_stage
class OpenAIAssertionExtractFormatter(FormatterStage):
    name = "openai_assertion_extract"
    description = "Extracts candidate entities and assertions from section-aligned slices using OpenAI structured JSON."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        try:
            make_openai_client()
        except RuntimeError as exc:
            errors.append(str(exc))
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        cfg = dict(ctx.assertions_config or {})
        slices_file = ctx.previous_outputs.get("extraction_slices_file")
        if not slices_file:
            artifacts = ctx.find_artifacts(artifact_type="extraction_slices")
            if artifacts and artifacts[-1].local_path:
                slices_file = artifacts[-1].local_path
        if not slices_file:
            return StageResult.failure("No extraction_slices_file available for OpenAI assertion extraction")

        slices = load_json_safe(slices_file, []) or []
        if not isinstance(slices, list) or not slices:
            return StageResult.failure("Extraction slices payload is invalid or empty")

        slices = _filter_slices(slices, cfg)
        if not slices:
            return StageResult.failure("No extraction slices remain after assertion filtering")

        if int(cfg.get("max_slices") or 0) > 0:
            slices = slices[: int(cfg.get("max_slices"))]

        model = str(cfg.get("extract_model") or "gpt-5-mini")
        reasoning_effort = str(cfg.get("extract_reasoning_effort") or "minimal")
        temperature = float(cfg.get("extract_temperature") or 0.0)
        max_completion_tokens = int(cfg.get("extract_max_completion_tokens") or 4000)
        concurrency = max(1, int(cfg.get("extract_concurrency") or 4))
        retries = max(1, int(cfg.get("extract_retries") or 3))
        retry_delay_sec = float(cfg.get("extract_retry_delay_sec") or 3.0)
        per_request_delay_sec = float(cfg.get("extract_per_request_delay_sec") or 0.0)
        use_cache = bool(cfg.get("extract_use_cache", True))

        cache_file = ctx.stage_work_dir / "openai_assertion_extract_cache.json"
        cache_payload = load_json_safe(cache_file, {}) or {}
        if not isinstance(cache_payload, dict):
            cache_payload = {}

        results: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []

        for slice_record in slices:
            if not isinstance(slice_record, dict):
                continue
            slice_id = clean_text(slice_record.get("id"))
            cached = cache_payload.get(slice_id)
            if use_cache and isinstance(cached, dict):
                results.append(cached)
            else:
                pending.append(slice_record)

        if pending:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(
                        _extract_one,
                        slice_record=slice_record,
                        client=None,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                        retries=retries,
                        retry_delay_sec=retry_delay_sec,
                        per_request_delay_sec=per_request_delay_sec,
                    ): slice_record
                    for slice_record in pending
                }
                for future in as_completed(futures):
                    slice_record = futures[future]
                    result = future.result()
                    slice_id = clean_text(slice_record.get("id"))
                    cache_payload[slice_id] = result
                    atomic_write_json(cache_file, cache_payload)
                    results.append(result)

        results.sort(key=lambda item: clean_text(item.get("slice_id")))
        candidate_entities: List[Dict[str, Any]] = []
        candidate_assertions: List[Dict[str, Any]] = []
        quality_flags: Dict[str, List[str]] = {}
        for result in results:
            candidate_entities.extend(result.get("entities") or [])
            candidate_assertions.extend(result.get("assertions") or [])
            if result.get("quality_flags"):
                quality_flags[clean_text(result.get("slice_id"))] = unique_strings(result.get("quality_flags") or [])

        results_file = ctx.stage_work_dir / "openai_assertion_extract_results.json"
        entities_file = ctx.stage_work_dir / "candidate_entities.json"
        assertions_file = ctx.stage_work_dir / "candidate_assertions.json"
        flags_file = ctx.stage_work_dir / "assertion_extract_quality_flags.json"
        atomic_write_json(cache_file, cache_payload)
        atomic_write_json(results_file, results)
        atomic_write_json(entities_file, candidate_entities)
        atomic_write_json(assertions_file, candidate_assertions)
        atomic_write_json(flags_file, quality_flags)

        artifacts = [
            ctx.make_artifact(
                results_file,
                artifact_type="assertion_extraction_results",
                role="assertion_candidates",
                metadata={"slices": len(results), "model": model},
            ),
            ctx.make_artifact(
                entities_file,
                artifact_type="assertion_entities",
                role="assertion_candidates",
                metadata={"records": len(candidate_entities), "kind": "entities"},
            ),
            ctx.make_artifact(
                assertions_file,
                artifact_type="assertion_candidates",
                role="assertion_candidates",
                metadata={"records": len(candidate_assertions), "kind": "assertions"},
            ),
        ]

        return StageResult.success(
            outputs={
                "candidate_entities_file": str(entities_file),
                "candidate_assertions_file": str(assertions_file),
                "assertion_extract_results_file": str(results_file),
                "assertion_extract_quality_flags_file": str(flags_file),
            },
            metrics={
                "candidate_entities": len(candidate_entities),
                "candidate_assertions": len(candidate_assertions),
                "assertion_extraction_slices": len(results),
            },
            artifacts=artifacts,
        )
