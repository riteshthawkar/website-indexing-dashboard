from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from pipeline.core.assertions import clean_text, coerce_confidence, normalize_answer_subtype, normalize_predicate, unique_strings
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.openai_client import json_completion, make_openai_client
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

_VALIDATION_JSON_SCHEMA = {
    "name": "assertion_validation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "validations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "assertion_id": {"type": "string"},
                        "decision": {"type": "string"},
                        "confidence": {"type": "number"},
                        "normalized_subject": {"type": "string"},
                        "normalized_predicate": {"type": "string"},
                        "normalized_answer_subtype": {"type": "string"},
                        "normalized_object": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "assertion_id",
                        "decision",
                        "confidence",
                        "normalized_subject",
                        "normalized_predicate",
                        "normalized_answer_subtype",
                        "normalized_object",
                        "reason",
                    ],
                },
            }
        },
        "required": ["validations"],
    },
}


_VALIDATION_SYSTEM_PROMPT = """
You validate candidate assertions against source text.

Return JSON only using this schema:
{
  "validations": [
    {
      "assertion_id": "string",
      "decision": "supported|ambiguous|unsupported",
      "confidence": 0.0,
      "normalized_subject": "string",
      "normalized_predicate": "string",
      "normalized_answer_subtype": "string",
      "normalized_object": "string",
      "reason": "string"
    }
  ]
}

Rules:
- supported: fully grounded in the source.
- ambiguous: partially grounded, stale, or not explicit enough.
- unsupported: not grounded in the source.
- Do not invent new facts; only validate the provided candidates.
"""


def _validation_prompt(slice_record: Dict[str, Any], assertions: List[Dict[str, Any]]) -> str:
    lines = [
        f"TITLE: {clean_text(slice_record.get('document_title'))}",
        f"SOURCE_URL: {clean_text(slice_record.get('source_url'))}",
        "",
        "SOURCE TEXT:",
        clean_text(slice_record.get("text")),
        "",
        "CANDIDATE ASSERTIONS:",
    ]
    for assertion in assertions:
        lines.extend(
            [
                f"- assertion_id: {clean_text(assertion.get('id'))}",
                f"  subject: {clean_text(assertion.get('subject_name'))}",
                f"  predicate: {clean_text(assertion.get('predicate') or assertion.get('answer_type'))}",
                f"  answer_subtype: {clean_text(assertion.get('answer_subtype'))}",
                f"  object: {clean_text(assertion.get('object_value') or assertion.get('object_name'))}",
                f"  support_span: {clean_text(assertion.get('support_span') or assertion.get('evidence'))}",
            ]
        )
    return "\n".join(lines).strip()


def _validate_one(
    *,
    slice_record: Dict[str, Any],
    assertions: List[Dict[str, Any]],
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
        system_prompt=_VALIDATION_SYSTEM_PROMPT,
        user_prompt=_validation_prompt(slice_record, assertions),
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        json_schema=_VALIDATION_JSON_SCHEMA,
        reasoning_effort=reasoning_effort,
        retries=retries,
        retry_delay_sec=retry_delay_sec,
        per_request_delay_sec=per_request_delay_sec,
    )
    validations = payload.get("validations") or []
    by_id = {
        clean_text(item.get("assertion_id")): item
        for item in validations
        if isinstance(item, dict) and clean_text(item.get("assertion_id"))
    }

    validated: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for assertion in assertions:
        assertion_id = clean_text(assertion.get("id"))
        decision_payload = by_id.get(assertion_id, {})
        decision = clean_text(decision_payload.get("decision") or "ambiguous").lower()
        normalized_predicate = normalize_predicate(
            decision_payload.get("normalized_predicate") or assertion.get("predicate") or assertion.get("answer_type")
        )
        answer_type = normalize_predicate(assertion.get("answer_type") or normalized_predicate)
        normalized_object = clean_text(
            decision_payload.get("normalized_object") or assertion.get("object_value") or assertion.get("object_name")
        )
        assertion_copy = {
            **assertion,
            "predicate": normalized_predicate,
            "relation_type": normalized_predicate,
            "answer_type": answer_type,
            "answer_subtype": normalize_answer_subtype(
                answer_type,
                decision_payload.get("normalized_answer_subtype") or assertion.get("answer_subtype"),
                normalized_object,
            ),
            "subject_name": clean_text(decision_payload.get("normalized_subject") or assertion.get("subject_name")),
            "object_name": normalized_object,
            "object_value": normalized_object,
            "confidence": max(
                coerce_confidence(assertion.get("confidence"), default=0.0),
                coerce_confidence(decision_payload.get("confidence"), default=0.0),
            ),
            "validator_confidence": coerce_confidence(decision_payload.get("confidence"), default=0.0),
            "validator_decision": decision,
            "validator_reason": clean_text(decision_payload.get("reason")),
        }
        if decision == "supported":
            validated.append(assertion_copy)
        else:
            rejected.append(assertion_copy)

    return {
        "slice_id": clean_text(slice_record.get("id")),
        "validated_assertions": validated,
        "rejected_assertions": rejected,
        "raw_payload": payload,
    }


@register_stage
class OpenAIAssertionValidateFormatter(FormatterStage):
    name = "openai_assertion_validate"
    description = "Validates candidate assertions against the source slice using a cheaper OpenAI model."

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
        assertions_file = ctx.previous_outputs.get("candidate_assertions_file")
        if not slices_file or not assertions_file:
            return StageResult.failure("extraction_slices_file and candidate_assertions_file are required for validation")

        slices = load_json_safe(slices_file, []) or []
        assertions = load_json_safe(assertions_file, []) or []
        if not isinstance(slices, list) or not isinstance(assertions, list):
            return StageResult.failure("Assertion validation inputs are invalid")

        slices_by_id = {
            clean_text(item.get("id")): item
            for item in slices
            if isinstance(item, dict) and clean_text(item.get("id"))
        }
        assertions_by_slice: Dict[str, List[Dict[str, Any]]] = {}
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            source_chunk_ids = unique_strings(assertion.get("source_chunk_ids") or [])
            slice_id = ""
            if source_chunk_ids:
                for slice_record in slices:
                    if not isinstance(slice_record, dict):
                        continue
                    slice_chunk_ids = set(unique_strings(slice_record.get("linked_chunk_ids") or []))
                    if slice_chunk_ids & set(source_chunk_ids):
                        slice_id = clean_text(slice_record.get("id"))
                        break
            if not slice_id:
                slice_id = clean_text(assertion.get("source_slice_id"))
            if not slice_id:
                continue
            assertions_by_slice.setdefault(slice_id, []).append(assertion)

        model = str(cfg.get("validate_model") or "gpt-5-nano")
        reasoning_effort = str(cfg.get("validate_reasoning_effort") or "minimal")
        temperature = float(cfg.get("validate_temperature") or 0.0)
        max_completion_tokens = int(cfg.get("validate_max_completion_tokens") or 3000)
        concurrency = max(1, int(cfg.get("validate_concurrency") or 4))
        retries = max(1, int(cfg.get("validate_retries") or 3))
        retry_delay_sec = float(cfg.get("validate_retry_delay_sec") or 2.0)
        per_request_delay_sec = float(cfg.get("validate_per_request_delay_sec") or 0.0)
        use_cache = bool(cfg.get("validate_use_cache", True))

        cache_file = ctx.stage_work_dir / "openai_assertion_validate_cache.json"
        cache_payload = load_json_safe(cache_file, {}) or {}
        if not isinstance(cache_payload, dict):
            cache_payload = {}

        results: List[Dict[str, Any]] = []
        pending: List[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
        for slice_id, slice_assertions in assertions_by_slice.items():
            cached = cache_payload.get(slice_id)
            if use_cache and isinstance(cached, dict):
                results.append(cached)
                continue
            slice_record = slices_by_id.get(slice_id)
            if slice_record is None:
                continue
            pending.append((slice_record, slice_assertions))

        if pending:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(
                        _validate_one,
                        slice_record=slice_record,
                        assertions=slice_assertions,
                        client=None,
                        model=model,
                        reasoning_effort=reasoning_effort,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                        retries=retries,
                        retry_delay_sec=retry_delay_sec,
                        per_request_delay_sec=per_request_delay_sec,
                    ): clean_text(slice_record.get("id"))
                    for slice_record, slice_assertions in pending
                }
                for future in as_completed(futures):
                    slice_id = futures[future]
                    result = future.result()
                    cache_payload[slice_id] = result
                    atomic_write_json(cache_file, cache_payload)
                    results.append(result)

        results.sort(key=lambda item: clean_text(item.get("slice_id")))
        validated_assertions: List[Dict[str, Any]] = []
        rejected_assertions: List[Dict[str, Any]] = []
        for result in results:
            validated_assertions.extend(result.get("validated_assertions") or [])
            rejected_assertions.extend(result.get("rejected_assertions") or [])

        results_file = ctx.stage_work_dir / "openai_assertion_validate_results.json"
        validated_file = ctx.stage_work_dir / "validated_assertions.json"
        rejected_file = ctx.stage_work_dir / "rejected_assertions.json"
        atomic_write_json(cache_file, cache_payload)
        atomic_write_json(results_file, results)
        atomic_write_json(validated_file, validated_assertions)
        atomic_write_json(rejected_file, rejected_assertions)

        artifacts = [
            ctx.make_artifact(
                validated_file,
                artifact_type="validated_assertions",
                role="assertion_candidates",
                metadata={"records": len(validated_assertions), "model": model},
            ),
            ctx.make_artifact(
                rejected_file,
                artifact_type="rejected_assertions",
                role="assertion_candidates",
                metadata={"records": len(rejected_assertions), "model": model},
            ),
        ]

        return StageResult.success(
            outputs={
                "validated_assertions_file": str(validated_file),
                "rejected_assertions_file": str(rejected_file),
                "assertion_validate_results_file": str(results_file),
            },
            metrics={
                "validated_assertions": len(validated_assertions),
                "rejected_assertions": len(rejected_assertions),
            },
            artifacts=artifacts,
        )
