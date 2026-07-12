"""
Deterministic knowledge graph formatter.

Builds a production-safe structural graph from the retrieval bundle:
- documents
- pages
- sections
- chunks
- facts
- media

This stage deliberately avoids unconstrained entity/relation extraction.
"""

from __future__ import annotations

import logging
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline.core.artifact_contracts import ArtifactContract, resolve_artifact_path
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import load_json_safe
from pipeline.core.knowledge_graph import (
    build_graph_bundle,
    make_graph_edge,
    make_graph_node,
    save_graph_bundle_with_index,
)
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _stable_id(kind: str, *parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not raw:
        raw = kind
    return f"{kind}:{sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def _document_node_id(record: Dict[str, Any]) -> str:
    anchor = (
        record.get("source_markdown_path")
        or record.get("source_url")
        or record.get("document_id")
        or record.get("source_file")
        or record.get("document_title")
    )
    return _stable_id("document", anchor)


def _document_label(record: Dict[str, Any]) -> str:
    return (
        _clean_text(record.get("document_title"))
        or _clean_text(record.get("source_url"))
        or Path(str(record.get("source_markdown_path") or "")).stem
        or str(record.get("document_id") or "Document")
    )


@register_stage
class KnowledgeGraphFormatter(FormatterStage):
    name = "knowledge_graph"
    description = "Builds a deterministic knowledge graph bundle from retrieval artifacts."

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
            return StageResult.failure("No retrieval_bundle available for knowledge graph formatting")

        retrieval_bundle = load_json_safe(retrieval_bundle_file, {}) or {}
        if not isinstance(retrieval_bundle, dict):
            return StageResult.failure("retrieval_bundle payload is invalid")

        chunk_records = list(retrieval_bundle.get("chunk_records") or [])
        parent_records = list(retrieval_bundle.get("parent_records") or [])
        media_records = list(retrieval_bundle.get("media_records") or [])
        fact_records = list(retrieval_bundle.get("fact_records") or [])
        evidence_span_records = list(retrieval_bundle.get("evidence_span_records") or [])

        if not chunk_records:
            return StageResult.failure("retrieval_bundle contains no chunk_records")

        page_records = [record for record in parent_records if str(record.get("parent_type") or "") == "page"]
        section_records = [record for record in parent_records if str(record.get("parent_type") or "") == "section"]
        page_map = {str(record.get("id") or ""): record for record in page_records if record.get("id")}
        section_map = {str(record.get("id") or ""): record for record in section_records if record.get("id")}

        nodes_by_id: Dict[str, Any] = {}
        edges_by_id: Dict[str, Any] = {}

        def _add_node(node) -> None:
            nodes_by_id[node.id] = node

        def _add_edge(edge) -> None:
            edges_by_id[edge.id] = edge

        def _ensure_document_node(record: Dict[str, Any]) -> str:
            doc_id = _document_node_id(record)
            if doc_id not in nodes_by_id:
                _add_node(
                    make_graph_node(
                        node_id=doc_id,
                        node_type="document",
                        label=_document_label(record),
                        properties={
                            "document_id": record.get("document_id"),
                            "document_title": record.get("document_title"),
                            "document_type": record.get("document_type"),
                            "source_markdown_path": record.get("source_markdown_path"),
                            "source_url": record.get("source_url"),
                            "source_file": record.get("source_file"),
                        },
                    )
                )
            return doc_id

        for record in page_records:
            document_node_id = _ensure_document_node(record)
            page_id = str(record.get("id") or "")
            if not page_id:
                continue
            _add_node(
                make_graph_node(
                    node_id=page_id,
                    node_type="page",
                    label=f"{_document_label(record)} page",
                    properties={
                        "page_key": page_id,
                        "page_numbers": list(record.get("page_numbers") or []),
                        "source_markdown_path": record.get("source_markdown_path"),
                        "source_url": record.get("source_url"),
                        "document_node_id": document_node_id,
                    },
                )
            )
            _add_edge(
                make_graph_edge(
                    edge_type="HAS_PAGE",
                    source_id=document_node_id,
                    target_id=page_id,
                )
            )

        for record in section_records:
            document_node_id = _ensure_document_node(record)
            section_id = str(record.get("id") or "")
            if not section_id:
                continue
            _add_node(
                make_graph_node(
                    node_id=section_id,
                    node_type="section",
                    label=" > ".join(record.get("section_path") or []) or "Section",
                    properties={
                        "section_key": section_id,
                        "section_path": list(record.get("section_path") or []),
                        "page_key": record.get("page_key"),
                        "page_numbers": list(record.get("page_numbers") or []),
                        "source_markdown_path": record.get("source_markdown_path"),
                        "source_url": record.get("source_url"),
                        "document_node_id": document_node_id,
                    },
                )
            )
            _add_edge(
                make_graph_edge(
                    edge_type="HAS_SECTION",
                    source_id=document_node_id,
                    target_id=section_id,
                )
            )
            page_key = str(record.get("page_key") or "")
            if page_key and page_key in page_map:
                _add_edge(
                    make_graph_edge(
                        edge_type="PAGE_HAS_SECTION",
                        source_id=page_key,
                        target_id=section_id,
                    )
                )

        for record in chunk_records:
            chunk_id = str(record.get("id") or "")
            if not chunk_id:
                continue
            document_node_id = _ensure_document_node(record)
            page_key = str(record.get("page_key") or "")
            section_key = str(record.get("section_key") or "")
            _add_node(
                make_graph_node(
                    node_id=chunk_id,
                    node_type="chunk",
                    label=_clean_text(record.get("heading")) or f"Chunk {int(record.get('chunk_index') or 0) + 1}",
                    properties={
                        "chunk_index": int(record.get("chunk_index") or 0),
                        "chunk_count": int(record.get("chunk_count") or 1),
                        "heading": record.get("heading"),
                        "section_path": list(record.get("section_path") or []),
                        "page_numbers": list(record.get("page_numbers") or []),
                        "source_markdown_path": record.get("source_markdown_path"),
                        "source_url": record.get("source_url"),
                        "document_node_id": document_node_id,
                        "page_key": page_key,
                        "section_key": section_key,
                        "text_excerpt": _clean_text(record.get("text"))[:280],
                    },
                )
            )
            _add_edge(make_graph_edge(edge_type="DOCUMENT_HAS_CHUNK", source_id=document_node_id, target_id=chunk_id))
            if page_key and page_key in page_map:
                _add_edge(make_graph_edge(edge_type="PAGE_HAS_CHUNK", source_id=page_key, target_id=chunk_id))
            if section_key and section_key in section_map:
                _add_edge(make_graph_edge(edge_type="SECTION_HAS_CHUNK", source_id=section_key, target_id=chunk_id))
            for neighbor_id in list(record.get("neighbor_ids") or []):
                if neighbor_id:
                    _add_edge(
                        make_graph_edge(
                            edge_type="NEXT_CHUNK",
                            source_id=chunk_id,
                            target_id=str(neighbor_id),
                            qualifier=str(neighbor_id),
                        )
                    )

        for record in fact_records:
            fact_id = str(record.get("id") or "")
            if not fact_id:
                continue
            document_node_id = _ensure_document_node(record)
            _add_node(
                make_graph_node(
                    node_id=fact_id,
                    node_type="fact",
                    label=_clean_text(record.get("text"))[:120],
                    properties={
                        "text": record.get("text"),
                        "heading": record.get("heading"),
                        "page_numbers": list(record.get("page_numbers") or []),
                        "source_markdown_path": record.get("source_markdown_path"),
                        "source_url": record.get("source_url"),
                        "document_node_id": document_node_id,
                        "page_key": record.get("page_key"),
                        "section_key": record.get("section_key"),
                    },
                )
            )
            _add_edge(make_graph_edge(edge_type="DOCUMENT_HAS_FACT", source_id=document_node_id, target_id=fact_id))
            for chunk_id in list(record.get("linked_chunk_ids") or []):
                if chunk_id:
                    _add_edge(make_graph_edge(edge_type="CHUNK_HAS_FACT", source_id=str(chunk_id), target_id=fact_id))
            for parent_id in list(record.get("linked_parent_ids") or []):
                if not parent_id:
                    continue
                edge_type = "PAGE_HAS_FACT" if parent_id in page_map else "SECTION_HAS_FACT"
                _add_edge(make_graph_edge(edge_type=edge_type, source_id=str(parent_id), target_id=fact_id))

        for record in evidence_span_records:
            span_id = str(record.get("id") or "")
            if not span_id:
                continue
            document_node_id = _ensure_document_node(record)
            _add_node(
                make_graph_node(
                    node_id=span_id,
                    node_type="evidence_span",
                    label=_clean_text(record.get("text"))[:120],
                    properties={
                        "text": record.get("text"),
                        "span_type": record.get("span_type"),
                        "authority_class": record.get("authority_class"),
                        "source_last_seen": record.get("source_last_seen"),
                        "validity_status": record.get("validity_status") or "active",
                        "heading": record.get("heading"),
                        "section_heading": record.get("section_heading"),
                        "breadcrumb": record.get("breadcrumb"),
                        "page_numbers": list(record.get("page_numbers") or []),
                        "source_markdown_path": record.get("source_markdown_path"),
                        "source_url": record.get("source_url"),
                        "canonical_url": record.get("canonical_url"),
                        "language_normalized_url": record.get("language_normalized_url"),
                        "document_node_id": document_node_id,
                        "page_key": record.get("page_key") or record.get("page_id"),
                        "section_key": record.get("section_key") or record.get("section_id"),
                        "chunk_id": record.get("chunk_id"),
                    },
                )
            )
            _add_edge(make_graph_edge(edge_type="DOCUMENT_HAS_EVIDENCE_SPAN", source_id=document_node_id, target_id=span_id))
            for chunk_id in list(record.get("linked_chunk_ids") or [record.get("chunk_id")]):
                if chunk_id:
                    _add_edge(
                        make_graph_edge(
                            edge_type="CHUNK_HAS_EVIDENCE_SPAN",
                            source_id=str(chunk_id),
                            target_id=span_id,
                        )
                    )
            for parent_id in list(record.get("linked_parent_ids") or []):
                if not parent_id:
                    continue
                edge_type = "PAGE_HAS_EVIDENCE_SPAN" if parent_id in page_map else "SECTION_HAS_EVIDENCE_SPAN"
                _add_edge(make_graph_edge(edge_type=edge_type, source_id=str(parent_id), target_id=span_id))

        for record in media_records:
            media_id = str(record.get("id") or "")
            if not media_id:
                continue
            document_node_id = _ensure_document_node(record)
            _add_node(
                make_graph_node(
                    node_id=media_id,
                    node_type="media",
                    label=_clean_text(record.get("title")) or _clean_text(record.get("caption")) or str(record.get("media_type") or "media").title(),
                    properties={
                        "media_type": record.get("media_type"),
                        "title": record.get("title"),
                        "caption": record.get("caption"),
                        "description": record.get("description"),
                        "page_number": record.get("page_number"),
                        "source_markdown_path": record.get("source_markdown_path"),
                        "source_url": record.get("source_url"),
                        "asset_uri": record.get("asset_uri"),
                        "local_path": record.get("local_path"),
                        "provider": record.get("provider"),
                        "document_node_id": document_node_id,
                        "page_key": record.get("page_key"),
                        "section_keys": list(record.get("section_keys") or []),
                    },
                )
            )
            _add_edge(make_graph_edge(edge_type="DOCUMENT_HAS_MEDIA", source_id=document_node_id, target_id=media_id))
            for chunk_id in list(record.get("linked_chunk_ids") or []):
                if chunk_id:
                    _add_edge(make_graph_edge(edge_type="CHUNK_HAS_MEDIA", source_id=str(chunk_id), target_id=media_id))
            for parent_id in list(record.get("linked_parent_ids") or []):
                if not parent_id:
                    continue
                edge_type = "PAGE_HAS_MEDIA" if parent_id in page_map else "SECTION_HAS_MEDIA"
                _add_edge(make_graph_edge(edge_type=edge_type, source_id=str(parent_id), target_id=media_id))

        page_link_graph_file = (
            ctx.previous_outputs.get("canonical_page_link_graph_file")
            or ctx.previous_outputs.get("page_link_graph_file")
        )
        page_link_graph = load_json_safe(page_link_graph_file, {}) if page_link_graph_file else {}
        if isinstance(page_link_graph, dict):
            for node in page_link_graph.get("nodes") or []:
                if not isinstance(node, dict):
                    continue
                node_id = str(node.get("id") or "")
                if not node_id:
                    continue
                _add_node(
                    make_graph_node(
                        node_id=node_id,
                        node_type="website_page",
                        label=_clean_text(node.get("label")) or _clean_text(node.get("url")) or node_id,
                        properties={
                            **dict(node.get("properties") or {}),
                            "url": node.get("url"),
                            "canonical_url": node.get("canonical_url"),
                            "canonical_family_url": node.get("canonical_family_url"),
                            "source_graph_node_type": node.get("node_type"),
                        },
                    )
                )
            for edge in page_link_graph.get("edges") or []:
                if not isinstance(edge, dict):
                    continue
                source_id = str(edge.get("source_id") or "")
                target_id = str(edge.get("target_id") or "")
                if not source_id or not target_id:
                    continue
                _add_edge(
                    make_graph_edge(
                        edge_type="WEBSITE_LINKS_TO",
                        source_id=source_id,
                        target_id=target_id,
                        qualifier=str(edge.get("id") or ""),
                        properties={
                            **dict(edge.get("properties") or {}),
                            "source_url": edge.get("source_url"),
                            "target_url": edge.get("target_url"),
                            "source_family_url": edge.get("source_family_url"),
                            "target_family_url": edge.get("target_family_url"),
                        },
                    )
                )

        valid_node_ids = set(nodes_by_id.keys())
        invalid_edge_count = 0
        filtered_edges_by_id: Dict[str, Any] = {}
        for edge_id, edge in edges_by_id.items():
            if edge.source_id not in valid_node_ids or edge.target_id not in valid_node_ids:
                invalid_edge_count += 1
                continue
            filtered_edges_by_id[edge_id] = edge
        edges_by_id = filtered_edges_by_id

        graph_bundle = build_graph_bundle(
            nodes=list(nodes_by_id.values()),
            edges=list(edges_by_id.values()),
            schema_version=1,
            graph_type="deterministic_content_graph",
        )
        graph_file = ctx.stage_work_dir / "knowledge_graph.json"
        graph_index_file = ctx.stage_work_dir / "knowledge_graph_index.json"
        save_graph_bundle_with_index(graph_bundle, graph_file, graph_index_file)

        logger.info(
            "Knowledge graph formatter: %d nodes, %d edges",
            graph_bundle["stats"]["node_count"],
            graph_bundle["stats"]["edge_count"],
        )

        artifacts = [
            ctx.make_artifact(
                graph_file,
                artifact_type="knowledge_graph_bundle",
                role="knowledge_graph",
                metadata=graph_bundle["stats"],
            ),
            ctx.make_artifact(
                graph_index_file,
                artifact_type="knowledge_graph_index",
                role="knowledge_graph_index",
                metadata={
                    "node_count": graph_bundle["stats"]["node_count"],
                    "edge_count": graph_bundle["stats"]["edge_count"],
                },
            ),
        ]

        return StageResult.success(
            outputs={
                "knowledge_graph_file": str(graph_file),
                "knowledge_graph_index_file": str(graph_index_file),
            },
            metrics={
                "graph_nodes": graph_bundle["stats"]["node_count"],
                "graph_edges": graph_bundle["stats"]["edge_count"],
                "document_nodes": graph_bundle["stats"]["node_type_counts"].get("document", 0),
                "page_nodes": graph_bundle["stats"]["node_type_counts"].get("page", 0),
                "section_nodes": graph_bundle["stats"]["node_type_counts"].get("section", 0),
                "chunk_nodes": graph_bundle["stats"]["node_type_counts"].get("chunk", 0),
                "fact_nodes": graph_bundle["stats"]["node_type_counts"].get("fact", 0),
                "evidence_span_nodes": graph_bundle["stats"]["node_type_counts"].get("evidence_span", 0),
                "media_nodes": graph_bundle["stats"]["node_type_counts"].get("media", 0),
                "website_page_nodes": graph_bundle["stats"]["node_type_counts"].get("website_page", 0),
                "website_link_edges": graph_bundle["stats"]["edge_type_counts"].get("WEBSITE_LINKS_TO", 0),
                "graph_invalid_edges_dropped": invalid_edge_count,
            },
            artifacts=artifacts,
        )
