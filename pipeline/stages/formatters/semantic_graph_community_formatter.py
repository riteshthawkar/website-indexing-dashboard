"""
Community detection stage for the knowledge graph.
Uses the Leiden algorithm to cluster entities into macro and micro communities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Set, Tuple

import igraph as ig
import leidenalg as la

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.knowledge_graph import (
    build_graph_bundle,
    load_graph_bundle,
    make_graph_edge,
    make_graph_node,
    save_graph_bundle_with_index,
)
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


@register_stage
class SemanticGraphCommunityFormatter(FormatterStage):
    name = "semantic_graph_community"
    description = "Clusters entities into communities using the Leiden algorithm."

    async def execute(self, ctx: StageContext) -> StageResult:
        graph_file = ctx.previous_outputs.get("knowledge_graph_file")
        if not graph_file:
            return StageResult.failure("Knowledge graph file is required for community detection")

        bundle = load_graph_bundle(graph_file)
        if not isinstance(bundle, dict):
            return StageResult.failure("Invalid graph bundle payload")

        nodes = bundle.get("nodes") or []
        edges = bundle.get("edges") or []

        # Build network mapping
        entity_ids = []
        entity_id_to_idx = {}
        for node in nodes:
            if isinstance(node, dict) and node.get("node_type") == "entity":
                entity_id = node.get("id")
                if entity_id:
                    entity_id_to_idx[entity_id] = len(entity_ids)
                    entity_ids.append(entity_id)

        # Collect assertion relations
        # An assertion connects a subject entity to an object entity
        subject_edges = {}  # assertion_id -> subject_entity_id
        object_edges = {}   # assertion_id -> object_entity_id

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            edge_type = edge.get("edge_type")
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            if edge_type == "ASSERTION_SUBJECT":
                subject_edges[source_id] = target_id
            elif edge_type == "ASSERTION_OBJECT":
                object_edges[source_id] = target_id

        # Build igraph
        g = ig.Graph(directed=False)
        g.add_vertices(len(entity_ids))

        # Add edges for entities connected by an assertion
        ig_edges = []
        for assertion_id in subject_edges:
            if assertion_id in object_edges:
                subj = subject_edges[assertion_id]
                obj = object_edges[assertion_id]
                if subj in entity_id_to_idx and obj in entity_id_to_idx:
                    ig_edges.append((entity_id_to_idx[subj], entity_id_to_idx[obj]))

        g.add_edges(ig_edges)

        # Community Detection (Leiden)
        # Using ModularityVertexPartition for simplicity; CPMVertexPartition could be used for resolution control
        try:
            partition = la.find_partition(g, la.ModularityVertexPartition)
        except Exception as e:
            logger.error(f"Leiden community detection failed: {e}")
            return StageResult.failure(f"Community detection failed: {e}")

        community_membership = partition.membership

        # Group entities by community
        communities_map: Dict[int, List[str]] = {}
        for idx, comm_id in enumerate(community_membership):
            communities_map.setdefault(comm_id, []).append(entity_ids[idx])

        # Append community nodes and edges
        new_nodes = list(nodes)
        new_edges = list(edges)

        valid_communities = 0
        for comm_id, members in communities_map.items():
            if len(members) < 2:
                continue # Skip tiny communities

            community_node_id = f"community:{comm_id}"

            # Create community node
            new_nodes.append(
                make_graph_node(
                    node_id=community_node_id,
                    node_type="community",
                    label=f"Community {comm_id}",
                    properties={
                        "community_id": comm_id,
                        "size": len(members),
                        "level": 0 # level 0 for base communities
                    }
                ).to_dict()
            )

            # Link members to community
            for entity_id in members:
                new_edges.append(
                    make_graph_edge(
                        edge_type="IN_COMMUNITY",
                        source_id=entity_id,
                        target_id=community_node_id,
                        qualifier=f"{entity_id}:{community_node_id}"
                    ).to_dict()
                )
            valid_communities += 1

        logger.info(f"Detected {valid_communities} communities from {len(entity_ids)} entities.")

        # Cleanly rebuild nodes and edges as GraphNode/GraphEdge objects to use build_graph_bundle safely.

        clean_nodes = []
        for n in new_nodes:
            clean_nodes.append(make_graph_node(
                node_id=n.get("id"),
                node_type=n.get("node_type"),
                label=n.get("label", ""),
                properties=n.get("properties", {})
            ))

        clean_edges = []
        for e in new_edges:
            clean_edges.append(make_graph_edge(
                edge_type=e.get("edge_type"),
                source_id=e.get("source_id"),
                target_id=e.get("target_id"),
                properties=e.get("properties", {}),
                qualifier=e.get("id", "")
            ))

        new_bundle = build_graph_bundle(
            nodes=clean_nodes,
            edges=clean_edges,
            schema_version=bundle.get("schema_version", 2),
            graph_type=bundle.get("graph_type", "promoted_semantic_graph")
        )

        out_graph_file = ctx.stage_work_dir / "community_knowledge_graph.json"
        out_index_file = ctx.stage_work_dir / "community_knowledge_graph_index.json"

        save_graph_bundle_with_index(new_bundle, out_graph_file, out_index_file)

        artifacts = [
            ctx.make_artifact(
                out_graph_file,
                artifact_type="knowledge_graph_bundle",
                role="knowledge_graph_with_communities",
                metadata=new_bundle["stats"],
            )
        ]

        return StageResult.success(
            outputs={
                "knowledge_graph_file": str(out_graph_file),
                "knowledge_graph_index_file": str(out_index_file),
                "community_graph_file": str(out_graph_file)
            },
            metrics={
                "detected_communities": valid_communities,
            },
            artifacts=artifacts
        )
