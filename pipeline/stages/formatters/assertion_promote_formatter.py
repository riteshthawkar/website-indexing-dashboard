from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

from pipeline.core.assertions import aggregate_assertion_metrics, clean_text, coerce_confidence, normalize_predicate
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage


def _canonicalized_assertion(assertion: Dict[str, Any]) -> Dict[str, Any]:
    predicate = normalize_predicate(
        assertion.get("answer_type")
        or assertion.get("predicate")
        or assertion.get("relation_type")
    )
    answer_subtype = clean_text(assertion.get("answer_subtype")).lower().replace(" ", "_")
    subject = clean_text(assertion.get("subject_entity_id")) or clean_text(assertion.get("subject_name")).casefold()
    obj = clean_text(assertion.get("object_value") or assertion.get("object_name")).casefold()
    source_last_seen = clean_text(
        assertion.get("source_last_seen")
        or assertion.get("last_seen")
        or assertion.get("crawled_at")
        or assertion.get("fetched_at")
    )
    return {
        **assertion,
        "canonical_subject": clean_text(assertion.get("canonical_subject") or subject),
        "canonical_predicate": clean_text(assertion.get("canonical_predicate") or predicate),
        "canonical_object": clean_text(assertion.get("canonical_object") or obj),
        "source_last_seen": source_last_seen,
        "validity_status": clean_text(assertion.get("validity_status") or "active").lower(),
        "answer_type": clean_text(assertion.get("answer_type") or predicate),
        "answer_subtype": answer_subtype,
    }


def _conflict_group_key(assertion: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        clean_text(assertion.get("canonical_subject")),
        clean_text(assertion.get("canonical_predicate")),
        clean_text(assertion.get("answer_subtype")),
    )


def _conflict_score(assertion: Dict[str, Any]) -> Tuple[float, float, float, int]:
    return (
        coerce_confidence(assertion.get("authority_score"), default=0.0),
        coerce_confidence(assertion.get("freshness_score"), default=0.0),
        coerce_confidence(assertion.get("confidence"), default=0.0),
        len(assertion.get("source_chunk_ids") or []),
    )


def _apply_conflict_governance(
    assertions: List[Dict[str, Any]],
    *,
    tie_tolerance: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    passthrough: List[Dict[str, Any]] = []
    for assertion in assertions:
        assertion = _canonicalized_assertion(assertion)
        key = _conflict_group_key(assertion)
        if not key[0] or not key[1]:
            passthrough.append(assertion)
            continue
        groups[key].append(assertion)

    governed: List[Dict[str, Any]] = []
    quarantined_conflicts: List[Dict[str, Any]] = []
    conflict_reports: List[Dict[str, Any]] = []
    governed.extend(passthrough)

    for key, records in groups.items():
        objects = {clean_text(record.get("canonical_object")) for record in records if clean_text(record.get("canonical_object"))}
        if len(objects) <= 1:
            governed.extend(records)
            continue

        ranked = sorted(records, key=_conflict_score, reverse=True)
        best = ranked[0]
        second = ranked[1]
        best_score = sum(_conflict_score(best)[:3])
        second_score = sum(_conflict_score(second)[:3])
        report = {
            "group_key": list(key),
            "objects": sorted(objects),
            "selected_assertion_id": "",
            "quarantined": False,
            "assertion_ids": [clean_text(record.get("id")) for record in ranked],
        }
        if abs(best_score - second_score) <= tie_tolerance:
            for record in ranked:
                quarantined_conflicts.append(
                    {
                        **record,
                        "validity_status": "quarantined_conflict",
                        "promotion_note": "equal_confidence_conflict",
                    }
                )
            report["quarantined"] = True
            conflict_reports.append(report)
            continue

        report["selected_assertion_id"] = clean_text(best.get("id"))
        governed.append({**best, "validity_status": "active"})
        for loser in ranked[1:]:
            governed.append(
                {
                    **loser,
                    "validity_status": "superseded",
                    "superseded_by": clean_text(best.get("id")),
                    "promotion_note": "superseded_by_authority_freshness_conflict_rule",
                }
            )
        conflict_reports.append(report)

    return governed, quarantined_conflicts, conflict_reports


@register_stage
class AssertionPromoteFormatter(FormatterStage):
    name = "assertion_promote"
    description = "Promotes validated canonical assertions into the governed assertion layer used by retrieval and graph storage."

    async def execute(self, ctx: StageContext) -> StageResult:
        cfg = dict(ctx.assertions_config or {})
        entities_file = ctx.previous_outputs.get("canonical_entities_file")
        assertions_file = ctx.previous_outputs.get("canonical_assertions_file")
        if not entities_file or not assertions_file:
            return StageResult.failure("canonical_entities_file and canonical_assertions_file are required for promotion")

        canonical_entities = load_json_safe(entities_file, []) or []
        canonical_assertions = load_json_safe(assertions_file, []) or []
        if not isinstance(canonical_entities, list) or not isinstance(canonical_assertions, list):
            return StageResult.failure("Assertion promotion inputs are invalid")

        min_confidence = float(cfg.get("promote_min_confidence") or 0.58)
        min_authority_score = float(cfg.get("promote_min_authority_score") or 0.40)
        allow_ambiguous = bool(cfg.get("promote_allow_ambiguous", False))
        conflict_tie_tolerance = float(cfg.get("conflict_tie_tolerance") or 0.03)

        promoted_assertions: List[Dict[str, Any]] = []
        quarantined_assertions: List[Dict[str, Any]] = []
        referenced_entity_ids: Set[str] = set()

        for assertion in canonical_assertions:
            if not isinstance(assertion, dict):
                continue
            decision = clean_text(assertion.get("validator_decision") or "supported").lower()
            confidence = coerce_confidence(assertion.get("confidence"), default=0.0)
            authority_score = coerce_confidence(assertion.get("authority_score"), default=0.0)
            if decision == "supported" and confidence >= min_confidence and authority_score >= min_authority_score:
                promoted_assertions.append(_canonicalized_assertion(assertion))
            elif allow_ambiguous and decision == "ambiguous" and confidence >= min_confidence and authority_score >= min_authority_score:
                promoted_assertions.append(_canonicalized_assertion({**assertion, "promotion_note": "promoted_ambiguous"}))
            else:
                quarantined_assertions.append(_canonicalized_assertion(assertion))

        promoted_assertions, conflict_quarantine, conflict_reports = _apply_conflict_governance(
            promoted_assertions,
            tie_tolerance=conflict_tie_tolerance,
        )
        quarantined_assertions.extend(conflict_quarantine)

        for assertion in promoted_assertions:
            if clean_text(assertion.get("validity_status") or "active").lower() not in {"active", "valid"}:
                continue
            subject_entity_id = clean_text(assertion.get("subject_entity_id"))
            object_entity_id = clean_text(assertion.get("object_entity_id"))
            if subject_entity_id:
                referenced_entity_ids.add(subject_entity_id)
            if object_entity_id:
                referenced_entity_ids.add(object_entity_id)

        promoted_entities = [
            entity
            for entity in canonical_entities
            if isinstance(entity, dict) and clean_text(entity.get("id")) in referenced_entity_ids
        ]

        promoted_entities_file = ctx.stage_work_dir / "promoted_entities.json"
        promoted_assertions_file = ctx.stage_work_dir / "promoted_assertions.json"
        quarantined_file = ctx.stage_work_dir / "quarantined_assertions.json"
        conflicts_file = ctx.stage_work_dir / "assertion_conflicts.json"
        atomic_write_json(promoted_entities_file, promoted_entities)
        atomic_write_json(promoted_assertions_file, promoted_assertions)
        atomic_write_json(quarantined_file, quarantined_assertions)
        atomic_write_json(conflicts_file, conflict_reports)

        active_assertions = [
            assertion
            for assertion in promoted_assertions
            if clean_text(assertion.get("validity_status") or "active").lower() == "active"
        ]
        metrics = aggregate_assertion_metrics(active_assertions)
        superseded_count = sum(
            1
            for assertion in promoted_assertions
            if clean_text(assertion.get("validity_status")).lower() == "superseded"
        )

        artifacts = [
            ctx.make_artifact(
                promoted_entities_file,
                artifact_type="promoted_entities",
                role="assertion_promoted",
                metadata={"records": len(promoted_entities)},
            ),
            ctx.make_artifact(
                promoted_assertions_file,
                artifact_type="promoted_assertions",
                role="assertion_promoted",
                metadata=metrics,
            ),
            ctx.make_artifact(
                conflicts_file,
                artifact_type="assertion_conflicts",
                role="assertion_governance",
                metadata={"records": len(conflict_reports)},
            ),
        ]

        return StageResult.success(
            outputs={
                "promoted_entities_file": str(promoted_entities_file),
                "promoted_assertions_file": str(promoted_assertions_file),
                "quarantined_assertions_file": str(quarantined_file),
                "assertion_conflicts_file": str(conflicts_file),
                "semantic_entities_file": str(promoted_entities_file),
                "semantic_assertions_file": str(promoted_assertions_file),
            },
            metrics={
                "promoted_entities": len(promoted_entities),
                **metrics,
                "assertion_conflicts_detected": len(conflict_reports),
                "assertions_superseded": superseded_count,
                "assertions_quarantined_by_conflict": len(conflict_quarantine),
                "quarantined_assertions": len(quarantined_assertions),
            },
            artifacts=artifacts,
        )
