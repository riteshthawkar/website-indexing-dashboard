#!/usr/bin/env python3
"""Fail closed when the committed DigitalOcean App Platform contract drifts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / ".do" / "app.yaml"


def _environment(component: dict[str, Any]) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for entry in component.get("envs") or []:
        assert isinstance(entry, dict) and entry.get("key"), "environment entry is invalid"
        key = str(entry["key"])
        assert key not in values, f"duplicate environment key: {key}"
        values[key] = entry
    return values


def _expect_value(environment: dict[str, dict[str, Any]], key: str, value: str) -> None:
    assert key in environment, f"missing environment key: {key}"
    assert str(environment[key].get("value")) == value, f"unsafe {key} value"


def main() -> int:
    assert SPEC_PATH.is_file(), "missing committed .do/app.yaml"
    spec = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
    assert isinstance(spec, dict), "App Platform spec must be a YAML object"
    services = {
        str(service.get("name")): service
        for service in spec.get("services") or []
        if isinstance(service, dict)
    }
    assert set(services) == {"backend", "retriever"}, "unexpected App Platform services"

    app_env = _environment(spec)
    assert set(app_env) == {"OPENAI_API_KEY", "RETRIEVAL_SERVICE_TOKEN"}, (
        "global environment must contain only intentionally shared secrets"
    )
    for key in ("OPENAI_API_KEY", "RETRIEVAL_SERVICE_TOKEN"):
        assert app_env.get(key, {}).get("type") == "SECRET", f"{key} must be encrypted"
        assert app_env.get(key, {}).get("scope") == "RUN_TIME", f"{key} must be runtime-only"
        assert str(app_env[key].get("value") or "").startswith("CHANGE_ME"), (
            f"{key} must remain a non-secret placeholder in Git"
        )

    backend = services["backend"]
    assert backend.get("github") == {
        "repo": "riteshthawkar/lawa-mbzuai-demo",
        "branch": "main",
        "deploy_on_push": False,
    }
    assert backend.get("dockerfile_path") == "Dockerfile"
    assert backend.get("http_port") == 8080
    assert backend.get("instance_count") == 1
    assert backend.get("instance_size_slug") == "apps-d-2vcpu-4gb"
    backend_env = _environment(backend)
    expected_backend_keys = {
        "SERVICE_ENVIRONMENT", "SERVICE_IDENTIFIER", "RELEASE_COMMIT_SHA", "PORT",
        "WEB_CONCURRENCY", "RETRIEVAL_SERVICE_URL", "RETRIEVAL_SERVICE_MODE",
        "RETRIEVAL_SERVICE_REQUIRE_READY", "RETRIEVAL_SERVICE_TIMEOUT_SECONDS",
        "RETRIEVAL_SERVICE_STARTUP_TIMEOUT_SECONDS", "RETRIEVAL_SERVICE_READINESS_TIMEOUT_SECONDS",
        "EXPECTED_RETRIEVAL_COMMIT_SHA", "EXPECTED_RETRIEVAL_RUN_ID",
        "EXPECTED_RETRIEVAL_BUNDLE_SHA256", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL_NAME",
        "EMBEDDING_OUTPUT_DIMENSIONALITY", "GENERATION_MODEL", "QUERY_REWRITE_MODEL",
        "RERANKER_MODEL", "GROUNDED_FINALIZER_MODEL", "FOLLOWUP_SUGGESTION_MODEL",
        "PRESENTATION_BLOCK_MODEL", "JWT_SECRET", "DATABASE_URL", "RATE_LIMIT_BACKEND",
        "RATE_LIMIT_REDIS_URL", "RATE_LIMIT_FAIL_OPEN", "RATE_LIMIT_ALLOW_MEMORY_IN_PRODUCTION",
        "ALLOWED_ORIGINS", "CORS_ALLOW_ORIGINS", "ALLOW_CREDENTIALS", "ALLOW_MISSING_WS_ORIGIN",
        "REQUIRE_WIDGET_KEY", "WIDGET_PUBLIC_KEYS", "OPERATIONS_API_TOKEN",
        "TELEGRAM_SHARED_SECRET", "STORE_CLIENT_IP_FOR_ANALYTICS", "LOG_QUERY_TEXT",
    }
    assert set(backend_env) == expected_backend_keys, "unexpected backend environment keys"
    for key, value in {
        "SERVICE_ENVIRONMENT": "production",
        "RELEASE_COMMIT_SHA": "${_self.COMMIT_HASH}",
        "RETRIEVAL_SERVICE_URL": "http://retriever:8060",
        "RETRIEVAL_SERVICE_MODE": "required",
        "RETRIEVAL_SERVICE_REQUIRE_READY": "true",
        "RETRIEVAL_SERVICE_TIMEOUT_SECONDS": "125",
        "RETRIEVAL_SERVICE_STARTUP_TIMEOUT_SECONDS": "1500",
        "RETRIEVAL_SERVICE_READINESS_TIMEOUT_SECONDS": "5",
        "EXPECTED_RETRIEVAL_COMMIT_SHA": "${retriever.COMMIT_HASH}",
        "EMBEDDING_PROVIDER": "google",
        "EMBEDDING_MODEL_NAME": "gemini-embedding-2",
        "EMBEDDING_OUTPUT_DIMENSIONALITY": "1536",
        "GENERATION_MODEL": "gpt-5.4-2026-03-05",
        "QUERY_REWRITE_MODEL": "gpt-5.4-mini-2026-03-17",
        "RERANKER_MODEL": "gpt-5.4-mini-2026-03-17",
        "GROUNDED_FINALIZER_MODEL": "gpt-5.4-mini-2026-03-17",
        "FOLLOWUP_SUGGESTION_MODEL": "gpt-5.4-mini-2026-03-17",
        "PRESENTATION_BLOCK_MODEL": "gpt-5.4-mini-2026-03-17",
        "RATE_LIMIT_BACKEND": "redis",
        "RATE_LIMIT_FAIL_OPEN": "false",
    }.items():
        _expect_value(backend_env, key, value)
    for key in (
        "EXPECTED_RETRIEVAL_RUN_ID",
        "EXPECTED_RETRIEVAL_BUNDLE_SHA256",
        "DATABASE_URL",
        "RATE_LIMIT_REDIS_URL",
        "JWT_SECRET",
        "OPERATIONS_API_TOKEN",
        "TELEGRAM_SHARED_SECRET",
        "WIDGET_PUBLIC_KEYS",
    ):
        assert key in backend_env, f"backend is missing {key}"
    for key in (
        "DATABASE_URL",
        "RATE_LIMIT_REDIS_URL",
        "JWT_SECRET",
        "OPERATIONS_API_TOKEN",
        "TELEGRAM_SHARED_SECRET",
        "WIDGET_PUBLIC_KEYS",
    ):
        assert backend_env[key].get("type") == "SECRET", f"{key} must be encrypted"
        assert backend_env[key].get("scope") == "RUN_TIME", f"{key} must be runtime-only"
        assert str(backend_env[key].get("value") or "").startswith("CHANGE_ME"), (
            f"{key} must remain a non-secret placeholder in Git"
        )
    assert backend.get("health_check", {}).get("http_path") == "/readyz"
    assert backend.get("liveness_health_check", {}).get("http_path") == "/livez"

    retriever = services["retriever"]
    assert retriever.get("github") == {
        "repo": "riteshthawkar/website-indexing-dashboard",
        "branch": "main",
        "deploy_on_push": False,
    }
    assert retriever.get("dockerfile_path") == "Dockerfile.retriever"
    assert "http_port" not in retriever, "retriever must not expose a public HTTP port"
    assert retriever.get("internal_ports") == [8060]
    assert retriever.get("instance_count") == 1
    assert retriever.get("instance_size_slug") == "apps-d-4vcpu-16gb"
    retriever_env = _environment(retriever)
    expected_retriever_keys = {
        "SERVICE_ENVIRONMENT", "PIPELINE_CONFIG", "GOOGLE_API_KEY", "PINECONE_API_KEY",
        "RELEASE_COMMIT_SHA", "RELEASE_STORAGE_MODE", "RELEASE_ARCHIVE_TARGET_ROOT",
        "ACTIVE_RELEASE_FILE", "RELEASE_RUNS_ROOT", "RELEASE_STORAGE_MARKER_FILE",
        "RELEASE_ARCHIVE_S3_URI", "RELEASE_ARCHIVE_S3_ENDPOINT_URL",
        "RELEASE_ARCHIVE_S3_REGION", "RELEASE_ARCHIVE_S3_ACCESS_KEY_ID",
        "RELEASE_ARCHIVE_S3_SECRET_ACCESS_KEY", "RELEASE_ARCHIVE_SHA256",
        "RELEASE_ARCHIVE_ALLOWED_HOSTS", "RELEASE_ARCHIVE_TIMEOUT_SECONDS",
        "RELEASE_ARCHIVE_VALIDATION_TIMEOUT_SECONDS", "RELEASE_ARCHIVE_MAX_BYTES",
        "RELEASE_ARCHIVE_MAX_EXTRACTED_BYTES", "RELEASE_ARCHIVE_MAX_FILES",
        "RELEASE_ARCHIVE_ALLOW_HTTP", "RELEASE_ARCHIVE_ALLOW_REPLACE",
        "RETRIEVER_REQUIRE_STORAGE_MARKER", "RETRIEVER_ALLOW_WAIVED_RELEASE",
        "RETRIEVER_REQUIRE_GRAPH", "RETRIEVER_EXPECTED_EMBEDDING_MODEL",
        "RETRIEVER_EXPECTED_EMBEDDING_DIMENSIONALITY", "RETRIEVER_HOST", "RETRIEVER_PORT",
        "RETRIEVER_MAX_CONCURRENCY", "RETRIEVER_REQUEST_TIMEOUT_SECONDS",
        "RETRIEVER_QUEUE_TIMEOUT_SECONDS", "RETRIEVAL_QUERY_EMBEDDING_RETRIES",
        "RETRIEVAL_QUERY_EMBEDDING_RETRY_DELAY_SECONDS", "RETRIEVER_STARTUP_PROBE_REQUIRED",
        "RETRIEVER_STARTUP_PROBE_QUERY", "RETRIEVER_STARTUP_PROBE_TIMEOUT_SECONDS",
    }
    assert set(retriever_env) == expected_retriever_keys, "unexpected retriever environment keys"
    for key in (
        "GOOGLE_API_KEY",
        "PINECONE_API_KEY",
        "RELEASE_ARCHIVE_S3_ACCESS_KEY_ID",
        "RELEASE_ARCHIVE_S3_SECRET_ACCESS_KEY",
    ):
        assert retriever_env.get(key, {}).get("type") == "SECRET", f"{key} must be encrypted"
        assert retriever_env.get(key, {}).get("scope") == "RUN_TIME", f"{key} must be runtime-only"
        assert str(retriever_env[key].get("value") or "").startswith("CHANGE_ME"), (
            f"{key} must remain a non-secret placeholder in Git"
        )
    for key, value in {
        "SERVICE_ENVIRONMENT": "production",
        "PIPELINE_CONFIG": "mbzuai_production",
        "RELEASE_COMMIT_SHA": "${_self.COMMIT_HASH}",
        "RELEASE_STORAGE_MODE": "hydrate",
        "RELEASE_ARCHIVE_TARGET_ROOT": "/data/releases/current",
        "ACTIVE_RELEASE_FILE": "/data/releases/current/mbzuai_main/active_release.json",
        "RELEASE_RUNS_ROOT": "/data/releases/current/runs/mbzuai_main",
        "RELEASE_STORAGE_MARKER_FILE": "/data/releases/current/.mbzuai-release-storage",
        "RELEASE_ARCHIVE_ALLOW_HTTP": "false",
        "RELEASE_ARCHIVE_ALLOW_REPLACE": "true",
        "RETRIEVER_REQUIRE_STORAGE_MARKER": "true",
        "RETRIEVER_ALLOW_WAIVED_RELEASE": "false",
        "RETRIEVER_REQUIRE_GRAPH": "true",
        "RETRIEVER_EXPECTED_EMBEDDING_MODEL": "gemini-embedding-2",
        "RETRIEVER_EXPECTED_EMBEDDING_DIMENSIONALITY": "1536",
        "RETRIEVER_MAX_CONCURRENCY": "1",
        "RETRIEVER_REQUEST_TIMEOUT_SECONDS": "110",
        "RETRIEVER_QUEUE_TIMEOUT_SECONDS": "10",
        "RETRIEVAL_QUERY_EMBEDDING_RETRIES": "1",
        "RETRIEVAL_QUERY_EMBEDDING_RETRY_DELAY_SECONDS": "0.75",
        "RETRIEVER_STARTUP_PROBE_REQUIRED": "true",
    }.items():
        _expect_value(retriever_env, key, value)
    backend_timeout = float(backend_env["RETRIEVAL_SERVICE_TIMEOUT_SECONDS"]["value"])
    retriever_request_timeout = float(retriever_env["RETRIEVER_REQUEST_TIMEOUT_SECONDS"]["value"])
    retriever_queue_timeout = float(retriever_env["RETRIEVER_QUEUE_TIMEOUT_SECONDS"]["value"])
    embedding_retries = int(retriever_env["RETRIEVAL_QUERY_EMBEDDING_RETRIES"]["value"])
    embedding_retry_delay = float(
        retriever_env["RETRIEVAL_QUERY_EMBEDDING_RETRY_DELAY_SECONDS"]["value"]
    )
    production_config = yaml.safe_load(
        (ROOT / "pipeline" / "configs" / "mbzuai_production.yaml").read_text(encoding="utf-8")
    )
    provider_timeout = float(production_config["embedder"]["gemini_request_timeout_ms"]) / 1000
    assert all(
        math.isfinite(value)
        for value in (
            backend_timeout,
            retriever_request_timeout,
            retriever_queue_timeout,
            embedding_retry_delay,
            provider_timeout,
        )
    ), "timeout values must be finite"
    assert backend_timeout > retriever_request_timeout + retriever_queue_timeout, (
        "backend retrieval timeout must exceed retriever request + queue timeouts"
    )
    provider_budget = (embedding_retries + 1) * provider_timeout + sum(
        embedding_retry_delay * (2**attempt) for attempt in range(embedding_retries)
    )
    assert provider_budget < retriever_request_timeout, (
        "Gemini query-embedding retry budget must fit inside the retriever timeout"
    )
    assert "RELEASE_ARCHIVE_URL" not in retriever_env, "production must not depend on an expiring URL"
    for key in (
        "RELEASE_ARCHIVE_S3_URI",
        "RELEASE_ARCHIVE_S3_ENDPOINT_URL",
        "RELEASE_ARCHIVE_S3_REGION",
        "RELEASE_ARCHIVE_SHA256",
        "RELEASE_ARCHIVE_ALLOWED_HOSTS",
    ):
        assert key in retriever_env, f"retriever is missing {key}"
        assert retriever_env[key].get("type") == "GENERAL", f"{key} must not be stored as a secret"
        assert str(retriever_env[key].get("value") or "").startswith(("CHANGE_ME", "https://CHANGE_ME")), (
            f"{key} must remain a non-secret placeholder in Git"
        )
    for check_name, expected_path in (
        ("health_check", "/readyz"),
        ("liveness_health_check", "/healthz"),
    ):
        check = retriever.get(check_name) or {}
        assert check.get("http_path") == expected_path
        assert check.get("port") == 8060
        assert int(check.get("initial_delay_seconds") or 0) >= 600
    hydration_budget = (
        float(retriever_env["RELEASE_ARCHIVE_TIMEOUT_SECONDS"]["value"])
        + float(retriever_env["RELEASE_ARCHIVE_VALIDATION_TIMEOUT_SECONDS"]["value"])
        + float(retriever_env["RETRIEVER_STARTUP_PROBE_TIMEOUT_SECONDS"]["value"])
    )
    for check_name in ("health_check", "liveness_health_check"):
        check = retriever.get(check_name) or {}
        failure_window = float(check["initial_delay_seconds"]) + (
            float(check["period_seconds"]) * float(check["failure_threshold"])
        )
        assert failure_window > hydration_budget, (
            f"retriever {check_name} window must exceed hydration + validation + startup probe budgets"
        )
    backend_startup_budget = float(
        backend_env["RETRIEVAL_SERVICE_STARTUP_TIMEOUT_SECONDS"]["value"]
    )
    assert backend_startup_budget > hydration_budget, (
        "backend retriever wait must exceed the complete retriever startup budget"
    )
    for check_name in ("health_check", "liveness_health_check"):
        check = backend.get(check_name) or {}
        failure_window = float(check["initial_delay_seconds"]) + (
            float(check["period_seconds"]) * float(check["failure_threshold"])
        )
        assert failure_window > backend_startup_budget, (
            f"backend {check_name} window must exceed the retriever startup wait"
        )
    termination = retriever.get("termination") or {}
    assert int(termination.get("drain_seconds") or 0) >= 30
    assert int(termination.get("grace_period_seconds") or 0) >= 120

    ingress_rules = (spec.get("ingress") or {}).get("rules") or []
    assert len(ingress_rules) == 1
    assert (ingress_rules[0].get("component") or {}).get("name") == "backend"
    print("DigitalOcean App Platform spec validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
