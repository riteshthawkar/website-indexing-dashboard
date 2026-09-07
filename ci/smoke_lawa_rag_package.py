#!/usr/bin/env python3
"""Smoke an installed ``lawa-rag`` wheel through real deployment boundaries."""

from __future__ import annotations

import argparse
import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import lawa_rag
import pipeline
from lawa_rag import IndexingPipeline, load_pipeline_config
from lawa_rag.retrieval import (
    AdaptiveHybridRetriever,
    PgVectorStore,
    RoutedHybridRetriever,
)
from lawa_rag.service import create_retrieval_service_app

try:
    from ci.retriever_smoke_fixture import FakePgVectorStore, write_synthetic_release
except ModuleNotFoundError as exc:  # Direct execution uses the script directory.
    if exc.name not in {"ci", "ci.retriever_smoke_fixture"}:
        raise
    from retriever_smoke_fixture import FakePgVectorStore, write_synthetic_release


def _assert_import_root(module_file: str, expected_root: Path, label: str) -> None:
    resolved = Path(module_file).resolve()
    if not resolved.is_relative_to(expected_root):
        raise RuntimeError(
            f"{label} loaded from {resolved}, not installed wheel root {expected_root}"
        )


def _configure_offline_credentials() -> None:
    os.environ.setdefault("OPENAI_API_KEY", "installed-wheel-smoke-openai")
    os.environ.setdefault("GEMINI_API_KEY", "installed-wheel-smoke-gemini")
    os.environ.setdefault(
        "PGVECTOR_DSN",
        "postgresql://mbzuai_retriever:test@127.0.0.1:5432/mbzuai?sslmode=require",
    )
    os.environ.setdefault(
        "PGVECTOR_INGEST_DSN",
        "postgresql://mbzuai_indexer:test@127.0.0.1:5432/mbzuai?sslmode=require",
    )


def _smoke_production_plan(root: Path) -> int:
    config = load_pipeline_config("mbzuai_production")
    work_dir = root / "plan-only"
    runner = IndexingPipeline.from_config(
        config,
        run_id="installed-wheel-smoke",
        work_dir=work_dir,
    )
    plan = runner.plan_sync()
    report = runner.validate_sync()
    if not report.ok:
        raise RuntimeError(f"canonical production config is invalid: {report.to_dict()}")
    if plan[0].plugin != "selected_profile_guard":
        raise RuntimeError("production plan does not start with the selected-profile guard")
    if plan[-1].plugin != "gemini_pgvector":
        raise RuntimeError("production plan does not end with the pgvector upload")
    if config["vector_store"]["provider"] != "pgvector":
        raise RuntimeError("canonical production vector-store provider drifted")
    if work_dir.exists():
        raise RuntimeError("planning or validation unexpectedly wrote a run directory")
    return len(plan)


def _smoke_retrieval(root: Path) -> dict:
    work_dir = root / "synthetic-release"
    write_synthetic_release(work_dir, vector_store_provider="pgvector")
    store = FakePgVectorStore()
    with patch.object(PgVectorStore, "from_config", return_value=store):
        retriever = AdaptiveHybridRetriever.from_config(
            config_name="installed-wheel-smoke",
            work_dir=work_dir,
        )
    if not isinstance(retriever, RoutedHybridRetriever) or retriever.graph is None:
        raise RuntimeError("public facade did not construct the routed graph retriever")
    if retriever.vector.vector_store_provider != "pgvector":
        raise RuntimeError("public facade did not select the pgvector adapter")
    try:
        result = retriever.retrieve(
            "What documents are required for MBZUAI graduate admissions?",
            query_vector=[0.0] * 1536,
        )
    finally:
        retriever.close()
    if result.get("selected_chunk_ids") != ["chunk-1"]:
        raise RuntimeError(f"routed retrieval selected the wrong evidence: {result}")
    if result.get("routing_graph_available") is not True or result.get("abstained") is True:
        raise RuntimeError(f"routed graph retrieval contract failed: {result}")
    return result


async def _smoke_service(root: Path) -> None:
    with patch.object(
        PgVectorStore,
        "from_config",
        return_value=FakePgVectorStore(),
    ):
        app = create_retrieval_service_app(
            config_name="installed-wheel-smoke",
            work_dir=root / "synthetic-release",
            max_concurrency=2,
            request_timeout_seconds=5.0,
            queue_timeout_seconds=0.25,
        )
        ready_endpoint = next(
            route.endpoint for route in app.routes if route.path == "/readyz"
        )
        async with app.router.lifespan_context(app):
            if app.state.ready is not True:
                raise RuntimeError("retrieval service did not become ready")
            if not isinstance(app.state.retriever, RoutedHybridRetriever):
                raise RuntimeError("retrieval service loaded the wrong retriever implementation")
            ready = await ready_endpoint()
            if ready.get("service") != "retriever" or ready.get("ready") is not True:
                raise RuntimeError(f"retrieval readiness endpoint is invalid: {ready}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-package-root", type=Path, required=True)
    args = parser.parse_args()
    expected_root = args.expected_package_root.expanduser().resolve()
    _assert_import_root(lawa_rag.__file__, expected_root, "lawa_rag")
    _assert_import_root(pipeline.__file__, expected_root, "pipeline")
    _configure_offline_credentials()

    with tempfile.TemporaryDirectory(prefix="lawa-rag-wheel-smoke-") as value:
        root = Path(value)
        stage_count = _smoke_production_plan(root)
        result = _smoke_retrieval(root)
        asyncio.run(_smoke_service(root))

    print(
        "installed wheel production smoke passed: "
        f"{stage_count} stages, evidence={result['selected_chunk_ids']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
