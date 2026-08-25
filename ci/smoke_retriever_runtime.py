#!/usr/bin/env python3
"""Exercise the retriever-only dependency closure without external services."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fastapi  # noqa: F401 - direct runtime dependency smoke
import openai  # noqa: F401 - direct runtime dependency smoke
import psycopg  # noqa: F401 - direct runtime dependency smoke
import psycopg_pool  # noqa: F401 - direct runtime dependency smoke
import pydantic  # noqa: F401 - direct runtime dependency smoke
import requests  # noqa: F401 - direct runtime dependency smoke
import uvicorn  # noqa: F401 - direct runtime dependency smoke
import yaml  # noqa: F401 - direct runtime dependency smoke
from google import genai  # noqa: F401 - direct runtime dependency smoke
from pinecone import Pinecone  # noqa: F401 - direct runtime dependency smoke
from pgvector.psycopg import register_vector  # noqa: F401 - direct runtime dependency smoke
from rank_bm25 import BM25Okapi  # noqa: F401 - direct runtime dependency smoke

from pipeline.core.knowledge_graph import (
    build_graph_bundle,
    build_graph_index,
    make_graph_edge,
    make_graph_node,
)
from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever
from pipeline.service.retrieval_api import create_retrieval_service_app


FORBIDDEN_RETRIEVER_PACKAGES = (
    "crawl4ai",
    "docling",
    "playwright",
    "torch",
    "transformers",
    "igraph",
    "leidenalg",
    # Canonical V3 uses a separate integrated-sparse Pinecone index.  This
    # package is reached only by the legacy same-index BM25 compatibility path.
    "pinecone_text",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_synthetic_release(work_dir: Path) -> None:
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
    _write_json(graph_dir / "promoted_knowledge_graph_index.json", build_graph_index(graph))

    # A non-production snapshot lets the smoke fixture stay compact while
    # exercising the same routed dense/local/graph implementation imported by
    # mbzuai_production. Production release invariants are tested separately.
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
    _write_json(work_dir / "resolved_config.json", {"config": config})


class _FakeDenseIndex:
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


async def _smoke_service_lifespan(work_dir: Path) -> None:
    app = create_retrieval_service_app(
        config_name="synthetic_retriever",
        work_dir=work_dir,
        max_concurrency=2,
        request_timeout_seconds=5,
        queue_timeout_seconds=0.25,
    )
    async with app.router.lifespan_context(app):
        if app.state.ready is not True:
            raise RuntimeError("retrieval service did not become ready")
        if not isinstance(app.state.retriever, RoutedHybridRetriever):
            raise TypeError(
                "retrieval service did not load the routed retriever: "
                f"{type(app.state.retriever).__name__}"
            )


def main() -> int:
    installed_forbidden = [
        package
        for package in FORBIDDEN_RETRIEVER_PACKAGES
        if importlib.util.find_spec(package) is not None
    ]
    if installed_forbidden:
        raise RuntimeError(f"retriever-only environment contains forbidden packages: {installed_forbidden}")

    with tempfile.TemporaryDirectory(prefix="mbzuai-retriever-smoke-") as root:
        work_dir = Path(root) / "synthetic-release"
        _write_synthetic_release(work_dir)
        retriever = AdaptiveHybridRetriever.from_config(
            config_name="synthetic_retriever",
            work_dir=work_dir,
        )
        if not isinstance(retriever, RoutedHybridRetriever):
            raise TypeError(f"expected RoutedHybridRetriever, got {type(retriever).__name__}")
        if retriever.graph is None:
            raise RuntimeError("synthetic routed retriever did not initialize its local knowledge graph")
        if retriever.vector.legacy_vectorstore_contract:
            raise RuntimeError("synthetic V3 retriever unexpectedly selected the legacy vector contract")

        # Mock only the external Pinecone network boundary. All routing, dense
        # lane handling, local BM25, graph loading, fusion, and evidence packing
        # execute through the production classes.
        retriever.vector._dense_index = _FakeDenseIndex()
        try:
            result = retriever.retrieve(
                "What documents are required for MBZUAI graduate admissions?",
                query_vector=[0.0] * 1536,
            )
        finally:
            retriever.close()

        if "chunk-1" not in result.get("selected_chunk_ids", []):
            raise RuntimeError(f"synthetic retrieval did not select its expected chunk: {result}")
        if result.get("abstained") is True:
            raise RuntimeError(f"synthetic retrieval unexpectedly abstained: {result}")
        if result.get("routing_graph_available") is not True:
            raise RuntimeError(f"synthetic retrieval did not expose graph availability: {result}")
        asyncio.run(_smoke_service_lifespan(work_dir))

    print("retriever-only dependency and routed retrieval smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
