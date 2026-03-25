"""
Canonicalize semantic graph extraction candidates into governed entities and assertions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage
from pipeline.core.semantic_graph import (
    clean_text,
    coerce_confidence,
    entity_merge_key,
    normalize_entity_label,
    stable_semantic_id,
    unique_strings,
)

logger = logging.getLogger(__name__)


@register_stage
class SemanticGraphCanonicalizeFormatter(FormatterStage):
    name = "semantic_graph_canonicalize"
    description = "Canonicalizes semantic entity/relation candidates into stable entities and relation assertions."

    async def execute(self, ctx: StageContext) -> StageResult:
        entity_candidates_file = ctx.previous_outputs.get("semantic_entity_candidates_file")
        relation_candidates_file = ctx.previous_outputs.get("semantic_relation_candidates_file")
        if not entity_candidates_file or not relation_candidates_file:
            return StageResult.failure("Semantic candidate files are required for canonicalization")

        entity_candidates = load_json_safe(entity_candidates_file, []) or []
        relation_candidates = load_json_safe(relation_candidates_file, []) or []
        if not isinstance(entity_candidates, list) or not isinstance(relation_candidates, list):
            return StageResult.failure("Semantic candidate payloads are invalid")

        graph_cfg = dict(ctx.graph_config or {})
        promote_min_confidence = float(graph_cfg.get("promote_min_confidence") or 0.65)

        canonical_entities: Dict[str, Dict[str, Any]] = {}
        entity_id_by_key: Dict[str, str] = {}

        def _upsert_entity(*, entity_type: str, name: str, aliases: List[str], description: str, confidence: float, source_candidate: Dict[str, Any]) -> str:
            merge_key = entity_merge_key(entity_type, name)
            canonical_id = entity_id_by_key.get(merge_key)
            if not canonical_id:
                canonical_id = stable_semantic_id("entity", merge_key)
                entity_id_by_key[merge_key] = canonical_id
                canonical_entities[canonical_id] = {
                    "id": canonical_id,
                    "node_type": "entity",
                    "entity_type": clean_text(entity_type) or "Other",
                    "canonical_name": normalize_entity_label(name),
                    "aliases": [],
                    "description": "",
                    "confidence": 0.0,
                    "source_chunk_ids": [],
                    "source_fact_ids": [],
                    "source_parent_ids": [],
                    "source_urls": [],
                    "document_titles": [],
                }
            entity = canonical_entities[canonical_id]
            entity["aliases"] = unique_strings([entity["canonical_name"], *entity.get("aliases", []), *aliases, name])
            if description and (not entity.get("description") or len(description) > len(str(entity.get("description") or ""))):
                entity["description"] = description
            entity["confidence"] = max(float(entity.get("confidence") or 0.0), confidence)
            entity["source_chunk_ids"] = unique_strings([*entity.get("source_chunk_ids", []), *(source_candidate.get("source_chunk_ids") or [])])
            entity["source_fact_ids"] = unique_strings([*entity.get("source_fact_ids", []), *(source_candidate.get("source_fact_ids") or [])])
            entity["source_parent_ids"] = unique_strings([*entity.get("source_parent_ids", []), *(source_candidate.get("source_parent_ids") or [])])
            entity["source_urls"] = unique_strings([*entity.get("source_urls", []), source_candidate.get("source_url", "")])
            entity["document_titles"] = unique_strings([*entity.get("document_titles", []), source_candidate.get("document_title", "")])
            return canonical_id

        for candidate in entity_candidates:
            if not isinstance(candidate, dict):
                continue
            confidence = coerce_confidence(candidate.get("confidence"), default=0.0)
            if confidence < promote_min_confidence:
                continue
            _upsert_entity(
                entity_type=candidate.get("entity_type"),
                name=candidate.get("name"),
                aliases=list(candidate.get("aliases") or []),
                description=clean_text(candidate.get("description")),
                confidence=confidence,
                source_candidate=candidate,
            )

        assertions_by_id: Dict[str, Dict[str, Any]] = {}
        for candidate in relation_candidates:
            if not isinstance(candidate, dict):
                continue
            confidence = coerce_confidence(candidate.get("confidence"), default=0.0)
            if confidence < promote_min_confidence:
                continue
            subject_id = _upsert_entity(
                entity_type=candidate.get("subject_type"),
                name=candidate.get("subject_name"),
                aliases=[],
                description="",
                confidence=confidence,
                source_candidate=candidate,
            )
            object_id = _upsert_entity(
                entity_type=candidate.get("object_type"),
                name=candidate.get("object_name"),
                aliases=[],
                description="",
                confidence=confidence,
                source_candidate=candidate,
            )
            assertion_id = stable_semantic_id(
                "assertion",
                subject_id,
                clean_text(candidate.get("relation_type")).upper(),
                object_id,
                candidate.get("source_id"),
                clean_text(candidate.get("evidence")),
            )
            existing = assertions_by_id.get(assertion_id)
            if not existing:
                assertions_by_id[assertion_id] = {
                    "id": assertion_id,
                    "node_type": "relation_assertion",
                    "relation_type": clean_text(candidate.get("relation_type")).upper() or "OTHER",
                    "subject_entity_id": subject_id,
                    "object_entity_id": object_id,
                    "subject_name": normalize_entity_label(candidate.get("subject_name")),
                    "object_name": normalize_entity_label(candidate.get("object_name")),
                    "evidence": clean_text(candidate.get("evidence")),
                    "confidence": confidence,
                    "source_id": clean_text(candidate.get("source_id")),
                    "source_kind": clean_text(candidate.get("source_kind")),
                    "source_chunk_ids": unique_strings(candidate.get("source_chunk_ids") or []),
                    "source_fact_ids": unique_strings(candidate.get("source_fact_ids") or []),
                    "source_parent_ids": unique_strings(candidate.get("source_parent_ids") or []),
                    "source_url": clean_text(candidate.get("source_url")),
                    "document_title": clean_text(candidate.get("document_title")),
                }
                continue
            existing["confidence"] = max(float(existing.get("confidence") or 0.0), confidence)
            evidence = clean_text(candidate.get("evidence"))
            if evidence and len(evidence) > len(str(existing.get("evidence") or "")):
                existing["evidence"] = evidence
            existing["source_chunk_ids"] = unique_strings([*existing.get("source_chunk_ids", []), *(candidate.get("source_chunk_ids") or [])])
            existing["source_fact_ids"] = unique_strings([*existing.get("source_fact_ids", []), *(candidate.get("source_fact_ids") or [])])
            existing["source_parent_ids"] = unique_strings([*existing.get("source_parent_ids", []), *(candidate.get("source_parent_ids") or [])])
            if not existing.get("source_url"):
                existing["source_url"] = clean_text(candidate.get("source_url"))
            if not existing.get("document_title"):
                existing["document_title"] = clean_text(candidate.get("document_title"))

        assertions: List[Dict[str, Any]] = list(assertions_by_id.values())
        canonical_entities_list = list(canonical_entities.values())
        entities_file = ctx.stage_work_dir / "canonical_entities.json"
        assertions_file = ctx.stage_work_dir / "canonical_assertions.json"
        atomic_write_json(entities_file, canonical_entities_list)
        atomic_write_json(assertions_file, assertions)

        artifacts = [
            ctx.make_artifact(
                entities_file,
                artifact_type="semantic_entities",
                role="semantic_graph_canonical",
                metadata={"records": len(canonical_entities_list)},
            ),
            ctx.make_artifact(
                assertions_file,
                artifact_type="semantic_assertions",
                role="semantic_graph_canonical",
                metadata={"records": len(assertions)},
            ),
        ]

        logger.info(
            "Semantic graph canonicalization: %d entities, %d assertions",
            len(canonical_entities_list),
            len(assertions),
        )

        return StageResult.success(
            outputs={
                "semantic_entities_file": str(entities_file),
                "semantic_assertions_file": str(assertions_file),
            },
            metrics={
                "semantic_entities": len(canonical_entities_list),
                "semantic_assertions": len(assertions),
            },
            artifacts=artifacts,
        )
