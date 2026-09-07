"""Synthetic release and external-store doubles shared by runtime smokes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.core.knowledge_graph import (
    build_graph_bundle,
    build_graph_index,
    make_graph_edge,
    make_graph_node,
)
from pipeline.vectorstores.pgvector_store import VectorMatch


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def write_synthetic_release(
    work_dir: Path,
    *,
    vector_store_provider: str = "pinecone",
) -> None:
    source_url = "https://mbzuai.ac.ae/study/admission-requirements"
    text = "MBZUAI graduate admissions require transcripts and recommendation letters."
    bundle = {
        "version": 5,
        "chunk_records": [
            {
                "id": "chunk-1",
                "text": text,
                "dense_text": text,
                "document_title": "Admissions Requirements",
                "document_id": "doc-1",
                "source_url": source_url,
                "section_key": "section-1",
                "page_key": "page-1",
                "fact_ids": ["fact-1"],
            }
        ],
        "parent_records": [
            {
                "id": "section-1",
                "parent_type": "section",
                "text": text,
                "dense_text": text,
                "child_chunk_ids": ["chunk-1"],
                "source_url": source_url,
                "document_title": "Admissions Requirements",
            },
            {
                "id": "page-1",
                "parent_type": "page",
                "text": text,
                "dense_text": text,
                "child_chunk_ids": ["chunk-1"],
                "source_url": source_url,
                "document_title": "Admissions Requirements",
            },
        ],
        "media_records": [],
        "fact_records": [
            {
                "id": "fact-1",
                "text": text,
                "dense_text": text,
                "linked_chunk_ids": ["chunk-1"],
                "linked_parent_ids": ["section-1", "page-1"],
                "source_url": source_url,
                "document_title": "Admissions Requirements",
            }
        ],
        "evidence_span_records": [
            {
                "id": "span-1",
                "text": text,
                "linked_chunk_ids": ["chunk-1"],
                "linked_parent_ids": ["section-1", "page-1"],
                "source_url": source_url,
                "document_title": "Admissions Requirements",
            }
        ],
        "summary_records": [
            {
                "id": "summary-1",
                "text": text,
                "linked_chunk_ids": ["chunk-1"],
                "linked_parent_ids": ["section-1", "page-1"],
                "source_url": source_url,
            }
        ],
        "assertion_records": [],
        "entity_records": [],
        "answer_records": [],
    }
    bundle_dir = work_dir / "stage_outputs" / "format_retrieval"
    _write_json(bundle_dir / "retrieval_bundle.json", bundle)

    lexical_records = []
    for record_type, key in (
        ("chunk", "chunk_records"),
        ("parent", "parent_records"),
        ("fact", "fact_records"),
        ("evidence_span", "evidence_span_records"),
        ("summary", "summary_records"),
    ):
        lexical_records.extend(
            {
                "id": record["id"],
                "record_type": record_type,
                "text": record["text"],
            }
            for record in bundle[key]
        )
    _write_json(bundle_dir / "lexical_corpus.json", lexical_records)

    graph = build_graph_bundle(
        nodes=(
            make_graph_node(node_id="chunk-1", node_type="chunk", label="Admissions"),
            make_graph_node(node_id="fact-1", node_type="fact", label=text),
        ),
        edges=(
            make_graph_edge(
                edge_type="CHUNK_HAS_FACT",
                source_id="chunk-1",
                target_id="fact-1",
            ),
        ),
    )
    graph_dir = work_dir / "stage_outputs" / "promote_graph"
    _write_json(graph_dir / "promoted_knowledge_graph.json", graph)
    _write_json(
        graph_dir / "promoted_knowledge_graph_index.json",
        build_graph_index(graph),
    )

    # This compact snapshot exercises the routed implementation used by the
    # production profile. Immutable production release contracts have their
    # own exhaustive fixtures.
    config = {
        "pipeline": {"production_profile": False},
        "embedder": {
            "pinecone_index": "synthetic-dense",
            "pinecone_sparse_index": "synthetic-sparse",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1536,
        },
        "graph": {"store_backend": "local_json"},
        "retrieval": {
            "retriever_backend": "routed_hybrid",
            "routed_graph_enabled": True,
            "routed_graph_required": True,
            "parallel_graph_enabled": True,
            "parallel_graph_augment_all_queries": True,
            "query_planner_enabled": False,
            "evidence_adjudicator_enabled": False,
            "hyde_enabled": False,
            "enable_sparse": False,
            "enable_rerank": False,
            "enable_local_bm25_fallback": True,
            "external_lanes_on_embedding_failure": False,
        },
    }
    if vector_store_provider == "pgvector":
        config["vector_store"] = {
            "provider": "pgvector",
            "release_id": work_dir.name,
            "dimensions": 1536,
            "schema_version": 2,
            "dsn_env": "PGVECTOR_DSN",
            "require_ssl": True,
            "require_active_release": False,
        }
    elif vector_store_provider != "pinecone":
        raise ValueError(f"Unsupported synthetic vector provider: {vector_store_provider}")
    _write_json(work_dir / "resolved_config.json", {"config": config})


class FakeDenseIndex:
    _IDS_BY_NAMESPACE = {
        "chunks": "chunk-1",
        "parents": "section-1",
        "facts": "fact-1",
        "evidence_spans": "span-1",
        "summaries": "summary-1",
    }

    def query(self, *, namespace: str, **_kwargs):
        record_id = self._IDS_BY_NAMESPACE.get(namespace)
        matches = [SimpleNamespace(id=record_id)] if record_id else []
        return SimpleNamespace(matches=matches)


class FakePgVectorStore:
    """In-memory substitute for only the PostgreSQL client boundary."""

    _IDS_BY_NAMESPACE = FakeDenseIndex._IDS_BY_NAMESPACE

    def __init__(self) -> None:
        self.queries = []

    def query(self, *, namespace: str, **kwargs):
        self.queries.append({"namespace": namespace, **kwargs})
        lane = str(namespace).split("--", 1)[0]
        record_id = self._IDS_BY_NAMESPACE.get(lane)
        return [VectorMatch(record_id=record_id, score=0.9)] if record_id else []

    def close(self) -> None:
        return None


__all__ = ["FakeDenseIndex", "FakePgVectorStore", "write_synthetic_release"]
