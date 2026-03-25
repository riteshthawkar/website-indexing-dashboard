"""
Promote deterministic + semantic graph artifacts into a final graph bundle.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.knowledge_graph import (
    build_graph_bundle,
    build_graph_index,
    load_graph_bundle,
    make_graph_edge,
    make_graph_node,
    save_graph_bundle,
)
from pipeline.core.registry import register_stage
from pipeline.core.semantic_graph import semantic_assertion_text

logger = logging.getLogger(__name__)


@register_stage
class SemanticGraphPromoteFormatter(FormatterStage):
    name = "semantic_graph_promote"
    description = "Promotes deterministic and semantic graph artifacts into a final graph bundle."

    async def execute(self, ctx: StageContext) -> StageResult:
        base_graph_file = ctx.previous_outputs.get("knowledge_graph_file")
        entities_file = ctx.previous_outputs.get("semantic_entities_file")
        assertions_file = ctx.previous_outputs.get("semantic_assertions_file")
        if not base_graph_file or not entities_file or not assertions_file:
            return StageResult.failure("Base graph and canonical semantic graph files are required for promotion")

        base_graph = load_graph_bundle(base_graph_file)
        canonical_entities = load_json_safe(entities_file, []) or []
        canonical_assertions = load_json_safe(assertions_file, []) or []
        if not isinstance(base_graph, dict) or not isinstance(canonical_entities, list) or not isinstance(canonical_assertions, list):
            return StageResult.failure("Semantic graph promotion inputs are invalid")

        nodes = [
            make_graph_node(
                node_id=node.get("id"),
                node_type=node.get("node_type"),
                label=node.get("label", ""),
                properties=node.get("properties") or {},
            )
            for node in (base_graph.get("nodes") or [])
            if isinstance(node, dict) and node.get("id") and node.get("node_type")
        ]
        edges = [
            make_graph_edge(
                edge_type=edge.get("edge_type"),
                source_id=edge.get("source_id"),
                target_id=edge.get("target_id"),
                properties=edge.get("properties") or {},
                qualifier=edge.get("id", ""),
            )
            for edge in (base_graph.get("edges") or [])
            if isinstance(edge, dict) and edge.get("source_id") and edge.get("target_id") and edge.get("edge_type")
        ]
        node_ids: Set[str] = {node.id for node in nodes}

        for entity in canonical_entities:
            if not isinstance(entity, dict):
                continue
            entity_id = str(entity.get("id") or "")
            if not entity_id or entity_id in node_ids:
                continue
            nodes.append(
                make_graph_node(
                    node_id=entity_id,
                    node_type="entity",
                    label=str(entity.get("canonical_name") or ""),
                    properties={
                        "entity_type": entity.get("entity_type"),
                        "canonical_name": entity.get("canonical_name"),
                        "aliases": entity.get("aliases") or [],
                        "description": entity.get("description"),
                        "confidence": entity.get("confidence"),
                        "source_chunk_ids": entity.get("source_chunk_ids") or [],
                        "source_fact_ids": entity.get("source_fact_ids") or [],
                        "source_parent_ids": entity.get("source_parent_ids") or [],
                        "source_urls": entity.get("source_urls") or [],
                        "document_titles": entity.get("document_titles") or [],
                    },
                )
            )
            node_ids.add(entity_id)
            for fact_id in entity.get("source_fact_ids") or []:
                if fact_id in node_ids:
                    edges.append(
                        make_graph_edge(
                            edge_type="FACT_MENTIONS_ENTITY",
                            source_id=str(fact_id),
                            target_id=entity_id,
                            qualifier=f"{fact_id}:{entity_id}",
                        )
                    )
            for chunk_id in entity.get("source_chunk_ids") or []:
                if chunk_id in node_ids:
                    edges.append(
                        make_graph_edge(
                            edge_type="CHUNK_MENTIONS_ENTITY",
                            source_id=str(chunk_id),
                            target_id=entity_id,
                            qualifier=f"{chunk_id}:{entity_id}",
                        )
                    )

        for assertion in canonical_assertions:
            if not isinstance(assertion, dict):
                continue
            assertion_id = str(assertion.get("id") or "")
            subject_id = str(assertion.get("subject_entity_id") or "")
            object_id = str(assertion.get("object_entity_id") or "")
            if not assertion_id or not subject_id or not object_id:
                continue
            nodes.append(
                make_graph_node(
                    node_id=assertion_id,
                    node_type="relation_assertion",
                    label=str(assertion.get("relation_type") or ""),
                    properties={
                        "relation_type": assertion.get("relation_type"),
                        "subject_entity_id": subject_id,
                        "object_entity_id": object_id,
                        "subject_name": assertion.get("subject_name"),
                        "object_name": assertion.get("object_name"),
                        "confidence": assertion.get("confidence"),
                        "evidence": assertion.get("evidence"),
                        "text": semantic_assertion_text(
                            assertion.get("subject_name"),
                            assertion.get("relation_type"),
                            assertion.get("object_name"),
                            assertion.get("evidence"),
                        ),
                        "source_id": assertion.get("source_id"),
                        "source_kind": assertion.get("source_kind"),
                        "source_chunk_ids": assertion.get("source_chunk_ids") or [],
                        "source_fact_ids": assertion.get("source_fact_ids") or [],
                        "source_parent_ids": assertion.get("source_parent_ids") or [],
                        "source_url": assertion.get("source_url"),
                        "document_title": assertion.get("document_title"),
                    },
                )
            )
            edges.extend(
                [
                    make_graph_edge(
                        edge_type="ASSERTION_SUBJECT",
                        source_id=assertion_id,
                        target_id=subject_id,
                        qualifier=f"{assertion_id}:{subject_id}:subject",
                    ),
                    make_graph_edge(
                        edge_type="ASSERTION_OBJECT",
                        source_id=assertion_id,
                        target_id=object_id,
                        qualifier=f"{assertion_id}:{object_id}:object",
                    ),
                ]
            )
            for fact_id in assertion.get("source_fact_ids") or []:
                if fact_id in node_ids:
                    edges.append(
                        make_graph_edge(
                            edge_type="FACT_SUPPORTS_ASSERTION",
                            source_id=str(fact_id),
                            target_id=assertion_id,
                            qualifier=f"{fact_id}:{assertion_id}",
                        )
                    )
            for chunk_id in assertion.get("source_chunk_ids") or []:
                if chunk_id in node_ids:
                    edges.append(
                        make_graph_edge(
                            edge_type="CHUNK_SUPPORTS_ASSERTION",
                            source_id=str(chunk_id),
                            target_id=assertion_id,
                            qualifier=f"{chunk_id}:{assertion_id}",
                        )
                    )

        promoted_bundle = build_graph_bundle(
            nodes=nodes,
            edges=edges,
            schema_version=2,
            graph_type="promoted_semantic_graph",
        )
        promoted_index = build_graph_index(promoted_bundle)

        graph_file = ctx.stage_work_dir / "promoted_knowledge_graph.json"
        graph_index_file = ctx.stage_work_dir / "promoted_knowledge_graph_index.json"
        save_graph_bundle(promoted_bundle, graph_file)
        save_graph_bundle(promoted_index, graph_index_file)

        artifacts = [
            ctx.make_artifact(
                graph_file,
                artifact_type="knowledge_graph_bundle",
                role="knowledge_graph",
                metadata=promoted_bundle["stats"],
            ),
            ctx.make_artifact(
                graph_index_file,
                artifact_type="knowledge_graph_index",
                role="knowledge_graph_index",
                metadata={
                    "node_count": promoted_bundle["stats"]["node_count"],
                    "edge_count": promoted_bundle["stats"]["edge_count"],
                    "schema_version": promoted_bundle["schema_version"],
                },
            ),
        ]

        logger.info(
            "Semantic graph promotion: %d nodes, %d edges",
            promoted_bundle["stats"]["node_count"],
            promoted_bundle["stats"]["edge_count"],
        )

        return StageResult.success(
            outputs={
                "promoted_knowledge_graph_file": str(graph_file),
                "promoted_knowledge_graph_index_file": str(graph_index_file),
                "knowledge_graph_file": str(graph_file),
                "knowledge_graph_index_file": str(graph_index_file),
            },
            metrics={
                "promoted_graph_nodes": promoted_bundle["stats"]["node_count"],
                "promoted_graph_edges": promoted_bundle["stats"]["edge_count"],
                "semantic_entity_nodes": promoted_bundle["stats"]["node_type_counts"].get("entity", 0),
                "semantic_assertion_nodes": promoted_bundle["stats"]["node_type_counts"].get("relation_assertion", 0),
            },
            artifacts=artifacts,
        )
