from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
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


def test_retrieval_service_runs_two_safe_requests_concurrently(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    class SafeRetriever:
        supports_shared_parallel_retrieval = True

        def __init__(self):
            self.barrier = threading.Barrier(2)
            self.lock = threading.Lock()
            self.active = 0
            self.max_active = 0

        def retrieve(self, query):
            with self.lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            try:
                self.barrier.wait(timeout=2.0)
                time.sleep(0.02)
                return {"query": query, "abstained": True, "retrieval_documents": []}
            finally:
                with self.lock:
                    self.active -= 1

    retriever = SafeRetriever()
    token = "Rtrv-2026-StrongToken_A9z8Y7x6W5v4"
    headers = {"X-Retrieval-Service-Token": token}
    monkeypatch.setenv("RETRIEVAL_SERVICE_TOKEN", token)
    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: retriever,
    )
    app = create_retrieval_service_app(config_name="cfg", work_dir=tmp_path, max_concurrency=2)

    with TestClient(app) as client:
        attestation = client.get("/attestationz", headers=headers)
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    client.post,
                    "/retrieve",
                    headers=headers,
                    json={"query": f"distinct query {index}", "request_id": f"parallel-{index}"},
                )
                for index in range(2)
            ]
            responses = [future.result(timeout=3.0) for future in futures]

    assert attestation.status_code == 200
    assert attestation.json()["max_concurrency"] == 2
    assert [response.status_code for response in responses] == [200, 200]
    assert retriever.max_active == 2


def test_retrieval_service_cache_separates_planner_handoff_mode(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    class PlannerAwareRetriever:
        supports_shared_parallel_retrieval = True

        def __init__(self):
            self.calls = []

        def retrieve(self, query, *, skip_query_planner=False):
            self.calls.append((query, skip_query_planner))
            return {
                "query": query,
                "abstained": True,
                "retrieval_documents": [],
                "planner_skipped": skip_query_planner,
            }

    retriever = PlannerAwareRetriever()
    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: retriever,
    )
    app = create_retrieval_service_app(config_name="cfg", work_dir=tmp_path)

    with TestClient(app) as client:
        planned_first = client.post("/retrieve", json={"query": "same query"})
        planned_cached = client.post("/retrieve", json={"query": "same query"})
        handoff_first = client.post(
            "/retrieve",
            json={"query": "same query", "skip_query_planner": True},
        )
        handoff_cached = client.post(
            "/retrieve",
            json={"query": "same query", "skip_query_planner": True},
        )

    assert retriever.calls == [("same query", False), ("same query", True)]
    assert planned_first.json()["service_cache_hit"] is False
    assert planned_cached.json()["service_cache_hit"] is True
    assert handoff_first.json()["service_cache_hit"] is False
    assert handoff_first.json()["service_query_planner_skipped"] is True
    assert handoff_cached.json()["service_cache_hit"] is True


def test_retrieval_service_forwards_original_query_and_separates_cache(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    class OriginalQueryAwareRetriever:
        supports_shared_parallel_retrieval = True

        def __init__(self):
            self.calls = []

        def retrieve(self, query, *, original_query=None):
            self.calls.append((query, original_query))
            return {
                "query": query,
                "original_query": original_query or query,
                "abstained": True,
                "retrieval_documents": [],
            }

    retriever = OriginalQueryAwareRetriever()
    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: retriever,
    )
    app = create_retrieval_service_app(config_name="cfg", work_dir=tmp_path)

    with TestClient(app) as client:
        first = client.post(
            "/retrieve",
            json={"query": "rewritten query", "original_query": "original Arabic query"},
        )
        cached = client.post(
            "/retrieve",
            json={"query": "rewritten query", "original_query": "original Arabic query"},
        )
        different_original = client.post(
            "/retrieve",
            json={"query": "rewritten query", "original_query": "different original query"},
        )

    assert retriever.calls == [
        ("rewritten query", "original Arabic query"),
        ("rewritten query", "different original query"),
    ]
    assert first.json()["service_original_query_forwarded"] is True
    assert cached.json()["service_cache_hit"] is True
    assert different_original.json()["service_cache_hit"] is False


def test_retrieval_service_forwards_navigation_context_and_separates_cache(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from pipeline.service.retrieval_api import create_retrieval_service_app

    class NavigationAwareRetriever:
        supports_shared_parallel_retrieval = True

        def __init__(self):
            self.calls = []

        def retrieve(self, query, *, navigation_context=None):
            self.calls.append((query, navigation_context))
            return {
                "query": query,
                "abstained": True,
                "retrieval_documents": [],
            }

    retriever = NavigationAwareRetriever()
    monkeypatch.setattr(
        "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: retriever,
    )
    app = create_retrieval_service_app(config_name="cfg", work_dir=tmp_path)
    apply_context = {
        "intent": "apply",
        "goal": "Apply to MBZUAI",
        "confidence": 0.94,
        "source": "backend_query_analysis",
    }
    contact_context = {
        "intent": "contact",
        "goal": "Contact MBZUAI",
        "confidence": 0.94,
        "source": "backend_query_analysis",
    }

    with TestClient(app) as client:
        first = client.post(
            "/retrieve",
            json={"query": "same query", "navigation_context": apply_context},
        )
        cached = client.post(
            "/retrieve",
            json={"query": "same query", "navigation_context": apply_context},
        )
        different_context = client.post(
            "/retrieve",
            json={"query": "same query", "navigation_context": contact_context},
        )

    assert retriever.calls == [
        ("same query", apply_context),
        ("same query", contact_context),
    ]
    assert first.status_code == 200
    assert first.json()["service_navigation_context_forwarded"] is True
    assert first.json()["service_cache_hit"] is False
    assert cached.json()["service_cache_hit"] is True
    assert different_context.json()["service_cache_hit"] is False


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


def test_premise_grounding_fallback_rejects_generic_scoped_evidence():
    from pipeline.core.evidence_adjudicator import (
        heuristic_adjudicate_factual_evidence,
    )

    result = heuristic_adjudicate_factual_evidence(
        query="ما رقم هاتف مكتب جامعة محمد بن زايد للذكاء الاصطناعي في سنغافورة؟",
        intent_summary={
            "answer_types": ["phone"],
            "requested_roles": [],
            "subject_tokens": [],
            "subject_phrases": [],
            "strict_answer_required": True,
        },
        answer_documents=[
            {
                "id": "generic-phone",
                "answer_type": "phone",
                "value": "+971 2 811 3333",
                "text": "The general MBZUAI phone number is +971 2 811 3333.",
            }
        ],
        fact_documents=[],
        retrieval_documents=[],
    )

    assert result["abstain"] is True
    assert result["reason"] == "presupposed_entity_or_scope_not_supported"


def test_premise_grounding_routes_non_fact_queries(monkeypatch):
    import pipeline.retrieval.routed_hybrid as module
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    calls = []

    def adjudicate(**kwargs):
        calls.append(kwargs)
        return {
            "used": True,
            "method": "openai",
            "abstain": True,
            "selected_answer_ids": [],
            "selected_fact_ids": [],
            "selected_chunk_ids": [],
            "reason": "offering_not_supported",
            "confidence": 0.94,
        }

    monkeypatch.setattr(module, "adjudicate_factual_evidence", adjudicate)
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.evidence_adjudicator_enabled = True
    retriever.selective_adjudication_enabled = True
    retriever.evidence_adjudicator_model = "gpt-5-nano"
    retriever.evidence_adjudicator_reasoning_effort = "minimal"
    retriever.evidence_adjudicator_min_confidence = 0.58
    retriever.evidence_adjudicator_max_completion_tokens = 100
    retriever.evidence_adjudicator_retries = 1
    retriever.evidence_adjudicator_retry_delay_sec = 0.0
    retriever.evidence_adjudicator_per_request_delay_sec = 0.0
    retriever.evidence_adjudicator_timeout_sec = 1.0
    retriever.evidence_adjudicator_provider_timeout_sec = 0.8
    retriever.evidence_adjudicator_max_workers = 1
    retriever.evidence_adjudicator_answer_limit = 2
    retriever.evidence_adjudicator_fact_limit = 2
    retriever.evidence_adjudicator_chunk_limit = 2

    try:
        result = retriever._apply_evidence_adjudication(
            "What are the admission requirements for MBZUAI's veterinary medicine degree?",
            {
                "mode": "scoped",
                "abstained": False,
                "retrieval_confidence": 0.95,
                "selected_chunk_ids": ["chunk-1"],
                "answer_documents": [],
                "fact_documents": [],
                "retrieval_documents": [
                    {"id": "chunk-1", "text": "General graduate admission requirements."}
                ],
            },
        )
    finally:
        retriever.close()

    assert len(calls) == 1
    assert result["premise_grounding_required"] is True
    assert result["abstained"] is True
    assert result["adjudication_reason"] == "offering_not_supported"


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


def test_navigation_catalog_rescue_skips_text_only_adjudication(monkeypatch):
    import pipeline.retrieval.routed_hybrid as module

    def fail_if_called(**_kwargs):
        raise AssertionError("navigation catalog evidence must not use text adjudication")

    monkeypatch.setattr(module, "adjudicate_factual_evidence", fail_if_called)
    retriever = module.RoutedHybridRetriever.__new__(module.RoutedHybridRetriever)
    retriever.evidence_adjudicator_enabled = True

    result = retriever._apply_evidence_adjudication(
        "Where can I apply for this position?",
        {
            "mode": "fact",
            "abstained": False,
            "navigation_evidence_rescued": True,
            "retrieval_documents": [{"id": "unrelated-text"}],
        },
    )

    assert result["premise_grounding_required"] is False
    assert result["abstained"] is False
    assert result["adjudication_used"] is False
    assert result["verification_status"] == "verified_navigation_catalog"
    assert result["adjudication_reason"] == "grounded_navigation_evidence"


def test_verified_media_evidence_skips_text_only_adjudication(monkeypatch):
    import pipeline.retrieval.routed_hybrid as module

    def fail_if_called(**_kwargs):
        raise AssertionError("verified media evidence must not use text adjudication")

    monkeypatch.setattr(module, "adjudicate_factual_evidence", fail_if_called)
    retriever = module.RoutedHybridRetriever.__new__(module.RoutedHybridRetriever)
    retriever.evidence_adjudicator_enabled = True
    retriever.vector = SimpleNamespace(
        _has_grounded_media_candidates=lambda **_kwargs: True,
    )

    result = retriever._apply_evidence_adjudication(
        "What does the image on page 10 show?",
        {
            "query_rewritten": "What does the image on page 10 show? visual figure",
            "mode": "fact",
            "abstained": False,
            "selected_media_ids": ["media-page-10"],
            "dense_media_ids": ["media-page-10"],
            "media": [{"id": "media-page-10"}],
            "retrieval_documents": [{"id": "weak-context", "text": "Program context."}],
        },
    )

    assert result["abstained"] is False
    assert result["adjudication_used"] is False
    assert result["media_evidence_verified"] is True
    assert result["verification_status"] == "verified_media_evidence"
    assert result["adjudication_reason"] == "grounded_media_evidence"


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


def test_navigation_candidates_can_rescue_text_abstention_only_when_grounded():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.page_card_map = {
        "page:profile": {"id": "page:profile"},
        "page:other": {"id": "page:other"},
    }
    retriever.action_map = {
        "action:email": {
            "id": "action:email",
            "page_card_ids": ["page:profile"],
            "title": "person@mbzuai.ac.ae",
            "raw_text": "person@mbzuai.ac.ae profile email",
            "metadata": {"action_type": "email"},
        },
        "action:download": {
            "id": "action:download",
            "page_card_id": "page:other",
            "action_type": "download",
            "label": "Download",
        },
    }

    assert retriever._has_grounded_navigation_candidates(
        query="What contact email is linked from the profile?",
        page_card_ids=["page:profile"],
        action_ids=["action:email"],
    )
    assert not retriever._has_grounded_navigation_candidates(
        query="What contact email is linked from the profile?",
        page_card_ids=["page:profile"],
        action_ids=["action:download"],
    )
    assert not retriever._has_grounded_navigation_candidates(
        query="What does the profile say?",
        page_card_ids=["page:profile"],
        action_ids=["action:email"],
    )


def test_local_fact_lane_bounds_expensive_rescoring_to_posting_limit():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    fact_ids = [f"fact:{index:04d}" for index in range(1000)]
    retriever.fact_map = {
        fact_id: {"id": fact_id, "text": "campus information"}
        for fact_id in fact_ids
    }
    retriever.fact_token_index = {"campus": fact_ids}
    retriever.fact_tokens_by_id = {
        fact_id: ["campus", "information"] for fact_id in fact_ids
    }
    retriever.local_index_max_postings_per_token = 64
    retriever._informative_query_tokens = lambda _query: ["campus"]
    retriever._source_query_bonus = lambda *_args, **_kwargs: 0.0
    scored: list[str] = []

    def fact_bonus(_query: str, fact_text: str) -> float:
        scored.append(fact_text)
        return 0.1

    retriever._fact_query_bonus = fact_bonus

    result = retriever._local_fact_query_ids("campus", top_k=8)

    assert len(result) == 8
    assert len(scored) == 64


def test_graph_assertion_lane_scores_only_indexed_candidates():
    from pipeline.retrieval.graph_rag import GraphRAGRetriever

    retriever = GraphRAGRetriever.__new__(GraphRAGRetriever)
    retriever.local_graph_available = True
    retriever._local_graph_loaded = True
    retriever.graph_relation_local_candidate_limit = 12
    retriever.assertion_map = {
        f"assertion:{index:04d}": {"id": f"assertion:{index:04d}"}
        for index in range(1000)
    }
    retriever.base = SimpleNamespace(
        namespace_assertions="assertions",
        _lexical_query_ids=lambda *_args, **_kwargs: ["assertion:0042"],
    )
    retriever._expanded_relation_query = lambda query, _plan: query
    scored: list[str] = []

    def score(_query, _plan, node):
        scored.append(node["id"])
        return 1.0

    retriever._score_relation_assertion_candidate = score

    result = retriever._local_relation_assertion_candidates(
        "Where is MBZUAI located?",
        SimpleNamespace(alias_tokens=()),
    )

    assert result == [("assertion:0042", 1.0)]
    assert scored == ["assertion:0042"]


def test_person_name_detection_does_not_treat_program_names_as_people():
    from pipeline.retrieval.adaptive_hybrid import (
        _person_name_tokens,
        _support_query_intents,
    )

    visitor_query = (
        "What does the MBZUAI Visitor Program say visitors can get hands-on access to?"
    )

    assert _person_name_tokens(visitor_query) == []
    assert "faculty_person" not in _support_query_intents(visitor_query)
    assert _person_name_tokens(
        "What is the email address linked to Mark Juan in the directory listing?"
    ) == ["mark", "juan"]
