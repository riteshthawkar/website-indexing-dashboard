from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from pipeline.core.assertions import (
    aggregate_assertion_metrics,
    assertion_text,
    build_entity_records_from_assertions,
    clean_text,
    coerce_confidence,
    merge_entity_records,
    normalize_answer_subtype,
    normalize_entity_type,
    normalize_predicate,
    stable_assertion_id,
    stable_entity_id,
    unique_strings,
)
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage


def _entity_key(entity_type: str, canonical_name: str) -> Tuple[str, str]:
    return (normalize_entity_type(entity_type), clean_text(canonical_name).casefold())


def _provenance_values(
    assertion: Dict[str, Any],
    plural_field: str,
    scalar_field: str = "",
) -> List[str]:
    raw_values = assertion.get(plural_field) or []
    if isinstance(raw_values, (str, bytes)):
        values: List[Any] = [raw_values]
    elif isinstance(raw_values, (list, tuple, set)):
        values = list(raw_values)
    else:
        values = [raw_values]
    if scalar_field:
        values.append(assertion.get(scalar_field))
    return unique_strings(values)


_PROVENANCE_FIELDS = {
    "source_slice_ids": "source_slice_id",
    "source_doc_ids": "source_doc_id",
    "source_chunk_ids": "",
    "source_parent_ids": "",
    "source_fact_ids": "",
    "source_span_ids": "",
    "source_urls": "source_url",
    "source_markdown_paths": "source_markdown_path",
    "document_titles": "document_title",
}


@register_stage
class AssertionCanonicalizeFormatter(FormatterStage):
    name = "assertion_canonicalize"
    description = "Canonicalizes extracted entities and assertions into stable entity IDs and deduplicated assertions."

    async def execute(self, ctx: StageContext) -> StageResult:
        candidate_entities_file = ctx.previous_outputs.get("candidate_entities_file")
        validated_assertions_file = ctx.previous_outputs.get("validated_assertions_file")
        rejected_assertions_file = ctx.previous_outputs.get("rejected_assertions_file")
        if not validated_assertions_file:
            return StageResult.failure("validated_assertions_file is required for canonicalization")

        candidate_entities = load_json_safe(candidate_entities_file, []) if candidate_entities_file else []
        validated_assertions = load_json_safe(validated_assertions_file, []) or []
        rejected_assertions = load_json_safe(rejected_assertions_file, []) if rejected_assertions_file else []
        if not isinstance(candidate_entities, list) or not isinstance(validated_assertions, list):
            return StageResult.failure("Assertion canonicalization inputs are invalid")
        if not isinstance(rejected_assertions, list):
            rejected_assertions = []

        assertion_entities = build_entity_records_from_assertions(validated_assertions)
        merged_entities = merge_entity_records(assertion_entities, candidate_entities)

        canonical_entities: List[Dict[str, Any]] = []
        entity_alias_map: Dict[Tuple[str, str], str] = {}
        entity_id_map: Dict[str, str] = {}
        grouped_entities: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for entity in merged_entities:
            if not isinstance(entity, dict):
                continue
            key = _entity_key(entity.get("entity_type"), entity.get("canonical_name"))
            if not key[1]:
                continue
            grouped_entities[key].append(entity)

        for (entity_type, _), members in grouped_entities.items():
            canonical_name = clean_text(members[0].get("canonical_name"))
            aliases = unique_strings(alias for member in members for alias in (member.get("aliases") or []))
            source_chunk_ids = unique_strings(
                chunk_id for member in members for chunk_id in (member.get("source_chunk_ids") or [])
            )
            source_parent_ids = unique_strings(
                parent_id for member in members for parent_id in (member.get("source_parent_ids") or [])
            )
            source_urls = unique_strings(url for member in members for url in (member.get("source_urls") or []))
            document_titles = unique_strings(
                title for member in members for title in (member.get("document_titles") or [])
            )
            confidence = max(coerce_confidence(member.get("confidence"), default=0.0) for member in members)
            canonical_id = stable_entity_id(entity_type, canonical_name)
            canonical_entity = {
                "id": canonical_id,
                "entity_type": entity_type,
                "canonical_name": canonical_name,
                "aliases": aliases,
                "description": clean_text(next((member.get("description") for member in members if member.get("description")), "")),
                "confidence": confidence,
                "source_chunk_ids": source_chunk_ids,
                "source_parent_ids": source_parent_ids,
                "source_urls": source_urls,
                "document_titles": document_titles,
            }
            canonical_entities.append(canonical_entity)
            entity_alias_map[(entity_type, canonical_name.casefold())] = canonical_id
            for alias in aliases:
                entity_alias_map[(entity_type, alias.casefold())] = canonical_id
            for member in members:
                member_id = clean_text(member.get("id"))
                if member_id:
                    entity_id_map[member_id] = canonical_id

        def _canonical_entity_id(entity_id: str, entity_type: str, canonical_name: str) -> str:
            if entity_id and entity_id in entity_id_map:
                return entity_id_map[entity_id]
            alias_id = entity_alias_map.get(_entity_key(entity_type, canonical_name))
            if alias_id:
                return alias_id
            return stable_entity_id(entity_type, canonical_name)

        merged_assertions: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
        for assertion in validated_assertions:
            if not isinstance(assertion, dict):
                continue
            predicate = normalize_predicate(
                assertion.get("predicate")
                or assertion.get("relation_type")
                or assertion.get("answer_type")
            )
            answer_type = normalize_predicate(assertion.get("answer_type") or predicate)
            subject_name = clean_text(assertion.get("subject_name"))
            object_name = clean_text(assertion.get("object_value") or assertion.get("object_name"))
            if not subject_name or not object_name or not predicate:
                continue
            subtype = normalize_answer_subtype(
                answer_type,
                assertion.get("answer_subtype"),
                object_name,
            )
            subject_type = assertion.get("subject_type") or "organization"
            object_type = assertion.get("object_type") or "other"
            subject_entity_id = _canonical_entity_id(
                clean_text(assertion.get("subject_entity_id")),
                subject_type,
                subject_name,
            )
            object_entity_id = _canonical_entity_id(
                clean_text(assertion.get("object_entity_id")),
                object_type,
                object_name,
            )
            key = (subject_entity_id, predicate, subtype, object_entity_id)
            canonical_id = stable_assertion_id(
                "canonical-v2",
                subject_entity_id,
                predicate,
                subtype,
                object_entity_id,
            )
            payload = {
                **assertion,
                "id": canonical_id,
                "predicate": predicate,
                "relation_type": predicate,
                "answer_type": answer_type,
                "answer_subtype": subtype,
                "subject_name": subject_name,
                "object_name": object_name,
                "object_value": object_name,
                "subject_entity_id": subject_entity_id,
                "object_entity_id": object_entity_id,
                "canonical_subject": subject_entity_id,
                "canonical_predicate": predicate,
                "canonical_object": object_name.casefold(),
                "qualifiers": unique_strings(assertion.get("qualifiers") or []),
            }
            for plural_field, scalar_field in _PROVENANCE_FIELDS.items():
                payload[plural_field] = _provenance_values(
                    assertion,
                    plural_field,
                    scalar_field,
                )
            current = merged_assertions.get(key)
            if current is None:
                merged_assertions[key] = payload
                continue
            for plural_field in _PROVENANCE_FIELDS:
                current[plural_field] = unique_strings(
                    [
                        *(current.get(plural_field) or []),
                        *(payload.get(plural_field) or []),
                    ]
                )
            current["qualifiers"] = unique_strings([*(current.get("qualifiers") or []), *(payload.get("qualifiers") or [])])
            current["confidence"] = max(
                coerce_confidence(current.get("confidence"), default=0.0),
                coerce_confidence(payload.get("confidence"), default=0.0),
            )
            current["validator_confidence"] = max(
                coerce_confidence(current.get("validator_confidence"), default=0.0),
                coerce_confidence(payload.get("validator_confidence"), default=0.0),
            )
            current["authority_score"] = max(
                coerce_confidence(current.get("authority_score"), default=0.0),
                coerce_confidence(payload.get("authority_score"), default=0.0),
            )
            current["freshness_score"] = max(
                coerce_confidence(current.get("freshness_score"), default=0.0),
                coerce_confidence(payload.get("freshness_score"), default=0.0),
            )
            if not current.get("support_span") and payload.get("support_span"):
                current["support_span"] = payload["support_span"]
                current["evidence"] = payload.get("evidence")

        canonical_ids = [
            clean_text(assertion.get("id"))
            for assertion in merged_assertions.values()
        ]
        if len(canonical_ids) != len(set(canonical_ids)):
            return StageResult.failure(
                "Canonical assertion identity collision detected; semantic graph output is unsafe"
            )

        for assertion in merged_assertions.values():
            assertion["text"] = assertion_text(
                subject_name=assertion.get("subject_name"),
                predicate=assertion.get("predicate"),
                object_value=assertion.get("object_value"),
                answer_type=assertion.get("answer_type"),
                answer_subtype=assertion.get("answer_subtype"),
                qualifiers=assertion.get("qualifiers") or [],
            )

        canonical_assertions = sorted(
            merged_assertions.values(),
            key=lambda item: (
                item.get("answer_type", ""),
                item.get("answer_subtype", ""),
                item.get("subject_name", ""),
                item.get("object_name", ""),
            ),
        )

        entities_file = ctx.stage_work_dir / "canonical_entities.json"
        assertions_file = ctx.stage_work_dir / "canonical_assertions.json"
        rejected_file = ctx.stage_work_dir / "canonicalization_rejected_assertions.json"
        atomic_write_json(entities_file, canonical_entities)
        atomic_write_json(assertions_file, canonical_assertions)
        atomic_write_json(rejected_file, rejected_assertions)

        metrics = aggregate_assertion_metrics(canonical_assertions)

        artifacts = [
            ctx.make_artifact(
                entities_file,
                artifact_type="canonical_entities",
                role="assertion_canonical",
                metadata={"records": len(canonical_entities)},
            ),
            ctx.make_artifact(
                assertions_file,
                artifact_type="canonical_assertions",
                role="assertion_canonical",
                metadata=metrics,
            ),
        ]

        return StageResult.success(
            outputs={
                "canonical_entities_file": str(entities_file),
                "canonical_assertions_file": str(assertions_file),
                "canonicalization_rejected_assertions_file": str(rejected_file),
                "semantic_entities_file": str(entities_file),
                "semantic_assertions_file": str(assertions_file),
            },
            metrics={
                "canonical_entities": len(canonical_entities),
                **metrics,
            },
            artifacts=artifacts,
        )
