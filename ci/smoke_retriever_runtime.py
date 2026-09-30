#!/usr/bin/env python3
"""Exercise the retriever-only dependency closure without external services."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# A source-tree execution needs the repository root. An installed-wheel smoke
# already has ``pipeline`` on its path and must not be redirected to the source
# checkout, otherwise CI could accidentally test the wrong copy.
if importlib.util.find_spec("pipeline") is None and str(PROJECT_ROOT) not in sys.path:
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
from pgvector.psycopg import register_vector  # noqa: F401 - dependency smoke
from rank_bm25 import BM25Okapi  # noqa: F401 - direct runtime dependency smoke

from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever
from pipeline.service.retrieval_api import create_retrieval_service_app

try:
    from ci.retriever_smoke_fixture import FakeDenseIndex, write_synthetic_release
except ModuleNotFoundError as exc:  # Docker mounts both smoke files into /tmp.
    if exc.name not in {"ci", "ci.retriever_smoke_fixture"}:
        raise
    from retriever_smoke_fixture import FakeDenseIndex, write_synthetic_release


FORBIDDEN_RETRIEVER_PACKAGES = (
    "crawl4ai",
    "docling",
    "playwright",
    "torch",
    "transformers",
    "igraph",
    "leidenalg",
    # Reached only by the legacy same-index BM25 compatibility path.
    "pinecone_text",
)


async def smoke_service_lifespan(work_dir: Path) -> None:
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
        raise RuntimeError(
            "retriever-only environment contains forbidden packages: "
            f"{installed_forbidden}"
        )

    with tempfile.TemporaryDirectory(prefix="mbzuai-retriever-smoke-") as root:
        work_dir = Path(root) / "synthetic-release"
        write_synthetic_release(work_dir)
        retriever = AdaptiveHybridRetriever.from_config(
            config_name="synthetic_retriever",
            work_dir=work_dir,
        )
        if not isinstance(retriever, RoutedHybridRetriever):
            raise TypeError(
                f"expected RoutedHybridRetriever, got {type(retriever).__name__}"
            )
        if retriever.graph is None:
            raise RuntimeError(
                "synthetic routed retriever did not initialize its local knowledge graph"
            )
        if retriever.vector.legacy_vectorstore_contract:
            raise RuntimeError(
                "synthetic V3 retriever unexpectedly selected the legacy vector contract"
            )

        # Mock only the external Pinecone network boundary. Routing, local
        # BM25, graph loading, fusion, and evidence packing remain real.
        retriever.vector._dense_index = FakeDenseIndex()
        try:
            result = retriever.retrieve(
                "What documents are required for MBZUAI graduate admissions?",
                query_vector=[0.0] * 1536,
            )
        finally:
            retriever.close()

        if "chunk-1" not in result.get("selected_chunk_ids", []):
            raise RuntimeError(
                f"synthetic retrieval did not select its expected chunk: {result}"
            )
        if result.get("abstained") is True:
            raise RuntimeError(f"synthetic retrieval unexpectedly abstained: {result}")
        if result.get("routing_graph_available") is not True:
            raise RuntimeError(
                f"synthetic retrieval did not expose graph availability: {result}"
            )
        asyncio.run(smoke_service_lifespan(work_dir))

    print("retriever-only dependency and routed retrieval smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
