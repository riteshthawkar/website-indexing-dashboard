"""Deployment-facing contracts for the public ``lawa_rag`` package."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ci.retriever_smoke_fixture import FakePgVectorStore, write_synthetic_release
from lawa_rag import IndexingPipeline, PipelineValidationError, load_pipeline_config
from lawa_rag.retrieval import (
    AdaptiveHybridRetriever,
    PgVectorStore,
    RoutedHybridRetriever,
)
from lawa_rag.service import create_retrieval_service_app


pytestmark = pytest.mark.production_contract


def _set_offline_deployment_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Satisfy config validation without contacting any external provider."""

    monkeypatch.setenv("OPENAI_API_KEY", "production-contract-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "production-contract-gemini")
    monkeypatch.setenv(
        "PGVECTOR_DSN",
        "postgresql://mbzuai_retriever:test@127.0.0.1:5432/mbzuai?sslmode=require",
    )
    monkeypatch.setenv(
        "PGVECTOR_INGEST_DSN",
        "postgresql://mbzuai_indexer:test@127.0.0.1:5432/mbzuai?sslmode=require",
    )


def test_canonical_config_resolves_real_plugins_and_validates_without_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_offline_deployment_credentials(monkeypatch)
    config = load_pipeline_config("mbzuai_production")
    work_dir = tmp_path / "must-remain-absent"
    pipeline = IndexingPipeline.from_config(
        config,
        run_id="package-production-contract",
        work_dir=work_dir,
    )

    plan = pipeline.plan_sync()
    report = pipeline.validate_sync()
    configured_stages = [
        (stage["id"], stage["type"], stage["plugin"])
        for stage in config["stages"]
        if stage.get("enabled", True)
    ]

    assert report.ok, report.to_dict()
    assert [(stage.id, stage.type, stage.plugin) for stage in plan] == configured_stages
    assert config["pipeline"]["production_profile"] is True
    assert config["selected_profile"]["variant_id"] == (
        "c650__gemini2_1536__dense_graph"
    )
    assert config["vector_store"]["provider"] == "pgvector"
    assert config["retrieval"]["retriever_backend"] == "routed_hybrid"
    assert config["retrieval"]["routed_graph_required"] is True
    assert plan[0].plugin == "selected_profile_guard"
    assert plan[-1].plugin == "gemini_pgvector"
    assert not work_dir.exists()


def test_canonical_config_fails_closed_before_writes_when_secrets_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "PGVECTOR_INGEST_DSN",
    ):
        monkeypatch.delenv(name, raising=False)
    work_dir = tmp_path / "must-remain-absent"
    pipeline = IndexingPipeline.from_config(
        "mbzuai_production",
        run_id="package-missing-secrets",
        work_dir=work_dir,
    )

    with pytest.raises(PipelineValidationError) as raised:
        pipeline.run_sync()

    messages = " ".join(
        message
        for errors in raised.value.report.errors.values()
        for message in errors
    )
    assert "OPENAI_API_KEY" in messages
    assert "GOOGLE_API_KEY or GEMINI_API_KEY" in messages
    assert "PGVECTOR_INGEST_DSN" in messages
    assert not work_dir.exists()


def test_public_retriever_runs_dense_local_graph_and_evidence_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_offline_deployment_credentials(monkeypatch)
    work_dir = tmp_path / "synthetic-release"
    write_synthetic_release(work_dir, vector_store_provider="pgvector")
    store = FakePgVectorStore()
    monkeypatch.setattr(
        PgVectorStore,
        "from_config",
        classmethod(lambda cls, config, **kwargs: store),
    )
    retriever = AdaptiveHybridRetriever.from_config(
        config_name="package-production-smoke",
        work_dir=work_dir,
    )

    assert isinstance(retriever, RoutedHybridRetriever)
    assert retriever.graph is not None
    assert retriever.vector.vector_store_provider == "pgvector"
    assert not retriever.vector.legacy_vectorstore_contract
    # Only the external PostgreSQL client boundary is replaced. Routing,
    # local retrieval, graph expansion, fusion, and evidence packing stay real.
    try:
        result = retriever.retrieve(
            "What documents are required for MBZUAI graduate admissions?",
            query_vector=[0.0] * 1536,
        )
    finally:
        retriever.close()

    assert result["selected_chunk_ids"] == ["chunk-1"]
    assert result["routing_parallel_vector_graph"] is True
    assert result["routing_graph_available"] is True
    assert result["graph_used"] is True
    assert result["abstained"] is False
    assert any(query["namespace"] == "chunks" for query in store.queries)
    assert result["evidence_pack"]["citations"] == [
        {
            "source_url": "https://mbzuai.ac.ae/study/admission-requirements",
            "document_title": "Admissions Requirements",
        }
    ]


def test_public_service_factory_reaches_ready_state_with_routed_retriever(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_offline_deployment_credentials(monkeypatch)
    work_dir = tmp_path / "synthetic-release"
    write_synthetic_release(work_dir, vector_store_provider="pgvector")
    monkeypatch.setattr(
        PgVectorStore,
        "from_config",
        classmethod(lambda cls, config, **kwargs: FakePgVectorStore()),
    )
    app = create_retrieval_service_app(
        config_name="package-production-smoke",
        work_dir=work_dir,
        max_concurrency=2,
        request_timeout_seconds=5.0,
        queue_timeout_seconds=0.25,
    )

    async def exercise_lifespan() -> None:
        ready_endpoint = next(
            route.endpoint for route in app.routes if route.path == "/readyz"
        )
        async with app.router.lifespan_context(app):
            assert app.state.ready is True
            assert isinstance(app.state.retriever, RoutedHybridRetriever)
            assert app.state.retriever.graph is not None
            ready = await ready_endpoint()
            assert ready["ok"] is True
            assert ready["service"] == "retriever"
            assert ready["ready"] is True
            assert 0.0 <= ready["uptime_seconds"] < 5.0

    asyncio.run(exercise_lifespan())
