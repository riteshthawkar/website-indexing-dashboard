from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


def _fact_payload() -> dict:
    return {
        "mode": "fact",
        "abstained": False,
        "answer_documents": [{"id": "answer-1", "text": "Supported answer"}],
        "fact_documents": [],
        "retrieval_documents": [{"id": "chunk-1", "text": "Supporting evidence"}],
    }


def test_retrieval_service_hides_internal_failure_details(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    class FailingRetriever:
        def retrieve(self, _query):
            raise RuntimeError("failed under /srv/private/releases/run-42 with secret-token")

    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: FailingRetriever(),
    )
    app = create_retrieval_service_app(config_name="cfg", work_dir=tmp_path)

    with TestClient(app) as client:
        response = client.post("/retrieve", json={"query": "test failure"})
        health = client.get("/healthz")

    assert response.status_code == 500
    assert response.json() == {"detail": "retrieval_failed"}
    assert "/srv/private" not in response.text
    assert "work_dir" not in health.json()


def test_retrieval_readiness_exposes_validated_release_identity(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    class FakeRetriever:
        def retrieve(self, _query):
            return {"abstained": True, "retrieval_documents": []}

    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: FakeRetriever(),
    )
    monkeypatch.setenv("RETRIEVAL_RELEASE_ID", "release-42")
    monkeypatch.setenv("RETRIEVAL_RELEASE_RUN_ID", "run-42")
    monkeypatch.setenv("RELEASE_COMMIT_SHA", "a" * 40)
    monkeypatch.setenv("RETRIEVAL_BUNDLE_SHA256", "b" * 64)
    token = "Rtrv-2026-StrongToken_A9z8Y7x6W5v4"
    monkeypatch.setenv("RETRIEVAL_SERVICE_TOKEN", token)

    app = create_retrieval_service_app(config_name="cfg", work_dir=tmp_path)
    with TestClient(app) as client:
        public_payload = client.get("/readyz").json()
        unauthorized = client.get("/attestationz")
        payload = client.get(
            "/attestationz",
            headers={"X-Retrieval-Service-Token": token},
        ).json()

    assert set(public_payload) == {"ok", "service", "ready", "uptime_seconds"}
    assert unauthorized.status_code == 401
    assert payload["release_id"] == "release-42"
    assert payload["run_id"] == "run-42"
    assert payload["commit_sha"] == "a" * 40
    assert payload["retrieval_bundle_sha256"] == "b" * 64
    assert "work_dir" not in payload


def test_startup_probe_checks_dense_sparse_embedding_and_full_retrieval(tmp_path):
    from pipeline.core.io import atomic_write_json
    from pipeline.service.retrieval_api import _run_startup_probe

    namespace = "chunks--run-42"
    manifest_path = tmp_path / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "namespaces": {"chunks": namespace},
            "uploaded": {"chunks": 2, "sparse_chunks": 2},
        },
    )

    class DenseIndex:
        def describe_index_stats(self, **_kwargs):
            return {"namespaces": {namespace: {"vector_count": 2}}}

        def query(self, **kwargs):
            assert kwargs["namespace"] == namespace
            assert kwargs["vector"] == [0.1, 0.2, 0.3]
            return SimpleNamespace(matches=[SimpleNamespace(id="chunk-1")])

    class SparseIndex:
        def describe_index_stats(self, **_kwargs):
            return {"namespaces": {namespace: {"vector_count": 2}}}

        def search(self, **kwargs):
            assert kwargs["namespace"] == namespace
            return {"result": {"hits": [{"_id": "chunk-1"}]}}

    class Vector:
        output_dimensionality = 3

        def _pinecone_index(self):
            return DenseIndex()

        def _pinecone_sparse_index(self):
            return SparseIndex()

        def embed_query(self, query):
            assert query == "Where is MBZUAI located?"
            return [0.1, 0.2, 0.3]

    class Retriever:
        vector = Vector()

        def retrieve(self, query, *, query_vector):
            assert query_vector == [0.1, 0.2, 0.3]
            return {
                "query_embedding_status": "ok",
                "abstained": False,
                "retrieval_documents": [{"id": "chunk-1"}],
            }

    report = _run_startup_probe(
        Retriever(),
        work_dir=tmp_path,
        query="Where is MBZUAI located?",
        operation_timeout_seconds=2.0,
    )

    assert report == {
        "dense_namespace_count": 1,
        "sparse_namespace_count": 1,
        "evidence_count": 1,
    }


def test_startup_probe_rejects_remote_namespace_count_drift(tmp_path):
    from pipeline.core.io import atomic_write_json
    from pipeline.service.retrieval_api import _run_startup_probe

    namespace = "chunks--run-42"
    atomic_write_json(
        tmp_path / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        {
            "namespaces": {"chunks": namespace},
            "uploaded": {"chunks": 2, "sparse_chunks": 2},
        },
    )

    class Index:
        def describe_index_stats(self, **_kwargs):
            return {"namespaces": {namespace: {"vector_count": 1}}}

    vector = SimpleNamespace(_pinecone_index=lambda: Index(), _pinecone_sparse_index=lambda: Index())

    with pytest.raises(RuntimeError, match="dense namespace count mismatch"):
        _run_startup_probe(
            SimpleNamespace(vector=vector),
            work_dir=tmp_path,
            query="Where is MBZUAI located?",
            operation_timeout_seconds=2.0,
        )


def test_retrieval_service_authenticates_requests_and_caps_unsafe_concurrency(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    class UnsafeRetriever:
        supports_shared_parallel_retrieval = False

        def retrieve(self, _query):
            return {"abstained": True, "retrieval_documents": []}

    token = "Rtrv-2026-StrongToken_A9z8Y7x6W5v4"
    monkeypatch.setenv("RETRIEVAL_SERVICE_TOKEN", token)
    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: UnsafeRetriever(),
    )
    app = create_retrieval_service_app(config_name="cfg", work_dir=tmp_path, max_concurrency=4)

    with TestClient(app) as client:
        health = client.get(
            "/attestationz",
            headers={"X-Retrieval-Service-Token": token},
        )
        unauthorized = client.post("/retrieve", json={"query": "Where is MBZUAI?"})
        authorized = client.post(
            "/retrieve",
            headers={"X-Retrieval-Service-Token": token},
            json={"query": "Where is MBZUAI?", "request_id": "request-1"},
        )
        invalid_request_id = client.post(
            "/retrieve",
            headers={"X-Retrieval-Service-Token": token},
            json={"query": "Where is MBZUAI?", "request_id": "bad\nlog"},
        )

    assert health.json()["max_concurrency"] == 1
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
    assert invalid_request_id.status_code == 422


@pytest.mark.parametrize(
    "token",
    [
        "t" * 32,
        "replace-with-retrieval-service-token",
        "CHANGE_ME_WITH_AT_LEAST_32_RANDOM_CHARACTERS",
        "change-me-change-me-change-me-change-me",
    ],
)
def test_retrieval_service_rejects_weak_shared_tokens(tmp_path, monkeypatch, token):
    from pipeline.service.retrieval_api import create_retrieval_service_app

    monkeypatch.setenv("RETRIEVAL_SERVICE_TOKEN", token)
    with pytest.raises(ValueError, match="RETRIEVAL_SERVICE_TOKEN"):
        create_retrieval_service_app(config_name="cfg", work_dir=tmp_path)


@pytest.mark.parametrize(
    "value",
    [
        "CHANGE_ME_IN_DIGITALOCEAN",
        "replace_me_with_provider_key",
        "example-key-value-1234567890",
    ],
)
def test_retrieval_service_rejects_provider_credential_placeholders(value):
    from pipeline.service.retrieval_api import _provider_credential_error

    assert _provider_credential_error(value) == "must not use a documented placeholder"


def test_retrieval_service_loads_env_before_capturing_auth_token(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import pipeline.service.retrieval_api as module

    token = "Rtrv-2026-StrongToken_A9z8Y7x6W5v4"
    monkeypatch.delenv("RETRIEVAL_SERVICE_TOKEN", raising=False)
    monkeypatch.setattr(module, "_load_env_files", lambda: monkeypatch.setenv("RETRIEVAL_SERVICE_TOKEN", token))
    monkeypatch.setattr(
        module.AdaptiveHybridRetriever,
        "from_config",
        lambda **_kwargs: SimpleNamespace(
            retrieve=lambda _query: {"abstained": True, "retrieval_documents": []}
        ),
    )
    app = module.create_retrieval_service_app(config_name="cfg", work_dir=tmp_path)

    with TestClient(app) as client:
        unauthorized = client.post("/retrieve", json={"query": "Where is MBZUAI?"})
        authorized = client.post(
            "/retrieve",
            headers={"X-Retrieval-Service-Token": token},
            json={"query": "Where is MBZUAI?"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200


def test_retrieval_service_drains_provider_work_before_closing_retriever(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    started = threading.Event()
    finish = threading.Event()
    request_done = threading.Event()
    events: list[str] = []

    class BlockingRetriever:
        def retrieve(self, _query):
            started.set()
            finish.wait(timeout=3.0)
            events.append("retrieve_done")
            return {"abstained": True, "retrieval_documents": []}

        def close(self):
            events.append("close")

    monkeypatch.setenv("RETRIEVER_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", "2")
    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: BlockingRetriever(),
    )
    app = create_retrieval_service_app(
        config_name="cfg",
        work_dir=tmp_path,
        request_timeout_seconds=3.0,
    )
    client = TestClient(app)
    client.__enter__()

    def _request() -> None:
        try:
            client.post("/retrieve", json={"query": "blocking query"})
        finally:
            request_done.set()

    request_thread = threading.Thread(target=_request, daemon=True)
    request_thread.start()
    assert started.wait(timeout=1.0)
    release_timer = threading.Timer(0.2, finish.set)
    release_timer.start()
    client.__exit__(None, None, None)
    request_thread.join(timeout=2.0)
    release_timer.cancel()

    assert request_done.is_set()
    assert events == ["retrieve_done", "close"]


def test_evidence_adjudicator_timeout_retains_bounded_capacity(monkeypatch):
    import pipeline.retrieval.routed_hybrid as module
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    started = threading.Event()
    finish = threading.Event()
    completed = threading.Event()
    calls = 0
    provider_timeouts: list[float] = []

    def blocking_adjudication(**kwargs):
        nonlocal calls
        calls += 1
        provider_timeouts.append(float(kwargs["provider_timeout_sec"]))
        started.set()
        try:
            finish.wait(timeout=5.0)
            return {
                "used": False,
                "method": "heuristic",
                "abstain": False,
                "selected_answer_ids": [],
                "selected_fact_ids": [],
                "selected_chunk_ids": [],
                "reason": "fallback",
                "confidence": 0.5,
            }
        finally:
            completed.set()

    monkeypatch.setattr(module, "adjudicate_factual_evidence", blocking_adjudication)
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.evidence_adjudicator_enabled = True
    retriever.selective_adjudication_enabled = False
    retriever.evidence_adjudicator_model = "gpt-5-nano"
    retriever.evidence_adjudicator_reasoning_effort = "minimal"
    retriever.evidence_adjudicator_min_confidence = 0.58
    retriever.evidence_adjudicator_max_completion_tokens = 100
    retriever.evidence_adjudicator_retries = 1
    retriever.evidence_adjudicator_retry_delay_sec = 0.0
    retriever.evidence_adjudicator_per_request_delay_sec = 0.0
    retriever.evidence_adjudicator_timeout_sec = 0.1
    retriever.evidence_adjudicator_max_workers = 1
    retriever.evidence_adjudicator_answer_limit = 1
    retriever.evidence_adjudicator_fact_limit = 1
    retriever.evidence_adjudicator_chunk_limit = 1

    try:
        timed_out = retriever._apply_evidence_adjudication("What is supported?", _fact_payload())
        assert started.is_set()
        assert timed_out["verification_status"] == "skipped_timeout"

        saturated = retriever._apply_evidence_adjudication("What else is supported?", _fact_payload())
        assert saturated["verification_status"] == "skipped_busy"
        assert saturated["adjudication_reason"] == "evidence_adjudicator_capacity_exhausted"
        assert calls == 1
        assert provider_timeouts == [0.1]
    finally:
        finish.set()
        assert completed.wait(timeout=1.0)
        retriever.close()


def test_openai_client_uses_explicit_provider_deadline(monkeypatch):
    import sys
    from types import SimpleNamespace

    import pipeline.core.openai_client as module

    captured: list[float] = []

    class FakeOpenAI:
        def __init__(self, *, api_key, timeout):
            assert api_key == "test-key"
            captured.append(float(timeout))

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    module._OPENAI_STATE.clients = {}
    if hasattr(module._OPENAI_STATE, "client"):
        del module._OPENAI_STATE.client

    first = module.make_openai_client(timeout_sec=3.5)
    second = module.make_openai_client(timeout_sec=3.5)

    assert first is second
    assert captured == [3.5]


def test_adaptive_retrieval_uses_request_local_timing_diagnostics():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.parallel_lane_workers = 1
    retriever.namespace_chunks = "chunks"
    retriever.namespace_assertions = "assertions"
    retriever.parent_map = {}
    retriever.summary_map = {}
    retriever.media_map = {}
    retriever.fact_map = {}
    retriever.evidence_span_map = {}
    retriever._dense_query_ids = lambda **_kwargs: ["dense-chunk"]
    retriever._sparse_query_ids = lambda **_kwargs: []
    retriever._local_chunk_query_ids = lambda **_kwargs: []
    retriever._local_answer_query_ids = lambda **_kwargs: []
    diagnostics: dict = {}

    lanes = retriever._run_query_lanes(
        query="campus facilities",
        query_vector=[0.1],
        mode=QueryMode.SCOPED,
        diagnostics=diagnostics,
        lane_top_ks={
            "chunk_dense": 1,
            "chunk_sparse": 0,
            "chunk_local": 0,
            "assertion_dense": 0,
            "assertion_sparse": 0,
            "answer_local": 0,
            "parent_dense": 0,
            "parent_sparse": 0,
            "parent_local": 0,
            "summary_dense": 0,
            "summary_sparse": 0,
            "media_dense": 0,
            "media_sparse": 0,
            "media_local": 0,
            "fact_dense": 0,
            "fact_sparse": 0,
            "fact_local": 0,
            "evidence_span_dense": 0,
            "evidence_span_sparse": 0,
            "evidence_span_local": 0,
        },
    )

    assert lanes["chunk_dense_ids"] == ["dense-chunk"]
    assert set(diagnostics["lane_latency_ms"]) == {"chunk_dense_ids"}
    assert not hasattr(retriever, "_last_lane_latency_ms")
