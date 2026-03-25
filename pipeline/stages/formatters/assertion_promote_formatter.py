from __future__ import annotations

from typing import Any, Dict, List, Set

from pipeline.core.assertions import aggregate_assertion_metrics, clean_text, coerce_confidence
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage


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
                promoted_assertions.append(assertion)
            elif allow_ambiguous and decision == "ambiguous" and confidence >= min_confidence and authority_score >= min_authority_score:
                promoted_assertions.append({**assertion, "promotion_note": "promoted_ambiguous"})
            else:
                quarantined_assertions.append(assertion)

        for assertion in promoted_assertions:
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
        atomic_write_json(promoted_entities_file, promoted_entities)
        atomic_write_json(promoted_assertions_file, promoted_assertions)
        atomic_write_json(quarantined_file, quarantined_assertions)

        metrics = aggregate_assertion_metrics(promoted_assertions)

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
        ]

        return StageResult.success(
            outputs={
                "promoted_entities_file": str(promoted_entities_file),
                "promoted_assertions_file": str(promoted_assertions_file),
                "quarantined_assertions_file": str(quarantined_file),
                "semantic_entities_file": str(promoted_entities_file),
                "semantic_assertions_file": str(promoted_assertions_file),
            },
            metrics={
                "promoted_entities": len(promoted_entities),
                **metrics,
                "quarantined_assertions": len(quarantined_assertions),
            },
            artifacts=artifacts,
        )
