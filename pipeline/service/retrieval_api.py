from __future__ import annotations

import asyncio
import copy
import hmac
import inspect
import json
import logging
import os
import re
import time
import unicodedata
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Mapping
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pipeline.core.config import load_config, load_effective_config
from pipeline.retrieval import AdaptiveHybridRetriever
from pipeline.retrieval.adaptive_hybrid import apply_vector_upload_manifest_config


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SERVICE_TOKEN_PLACEHOLDER_MARKERS = (
    "change-me",
    "change_me",
    "changeme",
    "replace-with",
    "replace_me",
    "placeholder",
    "example-token",
    "retrieval-service-token",
)
_PROVIDER_CREDENTIAL_PLACEHOLDER_MARKERS = (
    "change-me",
    "change_me",
    "changeme",
    "replace-me",
    "replace_me",
    "placeholder",
    "example-key",
    "example_key",
    "your-key",
    "your_key",
)


class NavigationContextRequest(BaseModel):
    intent: str = Field(default="none", min_length=1, max_length=40)
    goal: str = Field(default="", max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    source: str = Field(default="upstream_query_planner", max_length=80)


class RetrieveRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="Natural-language query text.",
    )
    original_query: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
        description=(
            "Optional original user wording when query is an upstream rewrite. "
            "It is used for coverage, language, and navigation grounding, not as "
            "an instruction to bypass the frozen retrieval corpus."
        ),
    )
    request_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
        description="Optional caller-supplied request identifier.",
    )
    skip_query_planner: bool = Field(
        default=False,
        description=(
            "Skip the retriever LLM planner only when a trusted upstream "
            "query-analysis stage already rewrote the retrieval query."
        ),
    )
    navigation_context: NavigationContextRequest | None = Field(
        default=None,
        description=(
            "Optional trusted upstream navigation intent. It may influence "
            "selection, but URLs/actions are always resolved from the frozen "
            "Page Graph catalog."
        ),
    )
    context_page_url: str | None = Field(
        default=None,
        min_length=1,
        max_length=2048,
        description=(
            "Optional official page containing the widget. It is used only for "
            "deictic questions such as 'this page' and never as arbitrary web evidence."
        ),
    )


def _validated_context_page_url(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    host = (parsed.hostname or "").casefold().rstrip(".")
    official = bool(
        host == "mbzuai.ac.ae"
        or host.endswith(".mbzuai.ac.ae")
        or host == "ifm.ai"
        or host.endswith(".ifm.ai")
        or host == "mbzuai.gitbook.io"
    )
    if parsed.scheme.casefold() not in {"http", "https"} or not official or parsed.username or parsed.password:
        return None
    path = parsed.path or "/"
    return f"https://{host}{path}".rstrip("/")


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _service_token_error(token: str) -> str | None:
    value = str(token or "")
    lowered = value.casefold()
    if len(value) < 32:
        return "must contain at least 32 characters"
    if value != value.strip() or any(character.isspace() for character in value):
        return "must not contain whitespace"
    if any(marker in lowered for marker in _SERVICE_TOKEN_PLACEHOLDER_MARKERS):
        return "must not use a documented placeholder"
    if len(set(value)) < 10:
        return "must contain at least 10 distinct characters"
    if any(
        len(value) % width == 0 and value == value[:width] * (len(value) // width)
        for width in range(1, (len(value) // 2) + 1)
    ):
        return "must not be a repeated pattern"
    return None


def _provider_credential_error(value: str | None) -> str | None:
    candidate = str(value or "")
    normalized = candidate.casefold()
    if not candidate.strip():
        return "is required"
    if candidate != candidate.strip() or any(character.isspace() for character in candidate):
        return "must not contain whitespace"
    if any(marker in normalized for marker in _PROVIDER_CREDENTIAL_PLACEHOLDER_MARKERS):
        return "must not use a documented placeholder"
    if len(candidate) < 16:
        return "must contain at least 16 characters"
    return None


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _namespace_counts(stats: Any) -> Dict[str, int]:
    namespaces = _field(stats, "namespaces", {}) or {}
    if not hasattr(namespaces, "items"):
        return {}
    output: Dict[str, int] = {}
    for namespace, summary in namespaces.items():
        try:
            output[str(namespace)] = int(_field(summary, "vector_count", 0) or 0)
        except (TypeError, ValueError):
            output[str(namespace)] = 0
    return output


def _response_items(response: Any, *, nested: bool = False) -> list[Any]:
    source = _field(response, "result", None) if nested else response
    if source is None:
        source = response
    key = "hits" if nested else "matches"
    return list(_field(source, key, []) or [])


def _run_startup_probe(
    retriever: Any,
    *,
    work_dir: Path,
    query: str,
    operation_timeout_seconds: float,
) -> Dict[str, Any]:
    """Prove embedding and vector providers can serve the exact frozen release."""

    manifest_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError("startup probe could not read the vector upload manifest") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("startup probe vector upload manifest is invalid")

    vector = getattr(retriever, "vector", retriever)
    provider = str(
        manifest.get("provider")
        or getattr(vector, "vector_store_provider", "pinecone")
        or "pinecone"
    ).strip().lower()
    namespaces = manifest.get("namespaces") if isinstance(manifest.get("namespaces"), dict) else {}
    uploaded = manifest.get("uploaded") if isinstance(manifest.get("uploaded"), dict) else {}
    if not namespaces:
        raise RuntimeError("startup probe vector namespaces are missing")

    if provider == "pgvector":
        store = getattr(vector, "_pgvector_store", None)
        if store is None:
            raise RuntimeError("startup probe pgvector store is not initialized")
        release_id = str(
            manifest.get("namespace_release_id")
            or getattr(vector, "vector_release_id", "")
        ).strip()
        expected_counts = {
            str(namespace): int(uploaded.get(str(lane)) or 0)
            for lane, namespace in namespaces.items()
        }
        health = store.health_check(
            release_id=release_id,
            expected_namespaces=expected_counts,
            expected_model=str(manifest.get("model") or ""),
            expected_contract_sha256=str(
                manifest.get("production_indexing_contract_fingerprint") or ""
            ),
            expected_lane_counts={
                str(lane): int(uploaded.get(str(lane)) or 0)
                for lane in namespaces
            },
            expected_artifact_hashes={
                key: str(manifest.get(key) or "")
                for key in (
                    "retrieval_bundle_sha256",
                    "lexical_corpus_sha256",
                    "promoted_assertions_sha256",
                    "knowledge_graph_sha256",
                    "knowledge_graph_index_sha256",
                    "selected_release_assembly_sha256",
                    "selected_release_binding_sha256",
                    "page_graph_navigation_catalog_sha256",
                    "upload_input_sha256",
                )
                if str(manifest.get(key) or "")
            },
        )
        query_vector = list(vector.embed_query(query))
        expected_dimension = int(getattr(vector, "output_dimensionality", 0) or 0)
        if not query_vector or (expected_dimension and len(query_vector) != expected_dimension):
            raise RuntimeError("startup probe embedding dimension mismatch")
        chunk_namespace = str(namespaces.get("chunks") or "").strip()
        if not store.query_ids(
            release_id=release_id,
            namespace=chunk_namespace,
            vector=query_vector,
            top_k=1,
        ):
            raise RuntimeError("startup probe pgvector query returned no matches")
        dense_counts = dict(health.get("namespace_counts") or {})
        sparse_counts: Dict[str, int] = {}
    elif provider == "pinecone":
        dense_handle = vector._pinecone_index()
        sparse_enabled = bool(str(manifest.get("sparse_index_name") or "").strip())
        sparse_handle = vector._pinecone_sparse_index() if sparse_enabled else None
        dense_counts = _namespace_counts(
            dense_handle.describe_index_stats(timeout=operation_timeout_seconds)
        )
        sparse_counts = (
            _namespace_counts(
                sparse_handle.describe_index_stats(timeout=operation_timeout_seconds)
            )
            if sparse_handle is not None
            else {}
        )
        for lane, namespace_value in namespaces.items():
            namespace = str(namespace_value or "").strip()
            expected_dense = int(uploaded.get(str(lane)) or 0)
            expected_sparse = int(uploaded.get(f"sparse_{lane}") or 0)
            if not namespace or int(dense_counts.get(namespace) or 0) != expected_dense:
                raise RuntimeError(f"startup probe dense namespace count mismatch for {lane}")
            if sparse_enabled and (
                int(sparse_counts.get(namespace) or 0) != expected_sparse
            ):
                raise RuntimeError(f"startup probe sparse namespace count mismatch for {lane}")

        query_vector = list(vector.embed_query(query))
        expected_dimension = int(getattr(vector, "output_dimensionality", 0) or 0)
        if not query_vector or (expected_dimension and len(query_vector) != expected_dimension):
            raise RuntimeError("startup probe embedding dimension mismatch")
        chunk_namespace = str(namespaces.get("chunks") or "").strip()
        dense_response = dense_handle.query(
            vector=query_vector,
            top_k=1,
            namespace=chunk_namespace,
            include_metadata=False,
            include_values=False,
            timeout=operation_timeout_seconds,
        )
        if not _response_items(dense_response):
            raise RuntimeError("startup probe dense query returned no matches")
        if sparse_handle is not None:
            sparse_response = sparse_handle.search(
                namespace=chunk_namespace,
                top_k=1,
                inputs={"text": query},
                fields=[],
                timeout=operation_timeout_seconds,
            )
            if not _response_items(sparse_response, nested=True):
                raise RuntimeError("startup probe sparse query returned no hits")
    else:
        raise RuntimeError(f"startup probe vector provider is unsupported: {provider}")

    result = retriever.retrieve(query, query_vector=query_vector)
    if not isinstance(result, dict) or result.get("query_embedding_status") not in {None, "ok"}:
        raise RuntimeError("startup probe full retrieval reported an embedding failure")
    evidence = [
        *(result.get("answer_documents") or []),
        *(result.get("fact_documents") or []),
        *(result.get("retrieval_documents") or []),
    ]
    if result.get("abstained") is True or not evidence:
        raise RuntimeError("startup probe full retrieval returned no usable evidence")
    result_payload = {
        "dense_namespace_count": len(dense_counts),
        "sparse_namespace_count": len(sparse_counts),
        "evidence_count": len(evidence),
    }
    if provider != "pinecone":
        result_payload["provider"] = provider
    return result_payload


def _load_env_files() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _health_payload(app: FastAPI) -> Dict[str, Any]:
    started_at = float(getattr(app.state, "started_at_monotonic", time.monotonic()))
    timed_out_inflight = int(getattr(app.state, "timed_out_inflight", 0))
    detached_inflight = int(getattr(app.state, "detached_inflight", 0))
    max_concurrency = int(getattr(app.state, "max_concurrency", 0))
    ready = bool(getattr(app.state, "ready", False)) and (
        max_concurrency <= 0 or detached_inflight < max_concurrency
    )
    return {
        "ok": True,
        "service": "retriever",
        "ready": ready,
        "config_name": getattr(app.state, "config_name", None),
        "release_id": getattr(app.state, "release_id", None),
        "run_id": getattr(app.state, "release_run_id", None),
        "commit_sha": getattr(app.state, "release_commit_sha", None),
        "retrieval_bundle_sha256": getattr(app.state, "retrieval_bundle_sha256", None),
        "knowledge_graph_sha256": getattr(app.state, "knowledge_graph_sha256", None),
        "knowledge_graph_index_sha256": getattr(app.state, "knowledge_graph_index_sha256", None),
        "lexical_corpus_sha256": getattr(app.state, "lexical_corpus_sha256", None),
        "promoted_assertions_sha256": getattr(app.state, "promoted_assertions_sha256", None),
        "selected_release_assembly_sha256": getattr(
            app.state, "selected_release_assembly_sha256", None
        ),
        "selected_release_binding_sha256": getattr(
            app.state, "selected_release_binding_sha256", None
        ),
        "page_graph_navigation_catalog_sha256": getattr(
            app.state, "page_graph_navigation_catalog_sha256", None
        ),
        "answer_runtime_commit_sha": getattr(app.state, "answer_runtime_commit_sha", None),
        "indexing_build_commit_sha": getattr(app.state, "indexing_build_commit_sha", None),
        "startup_probe_required": bool(getattr(app.state, "startup_probe_required", False)),
        "startup_probe_passed": bool(getattr(app.state, "startup_probe_passed", False)),
        "request_count": int(getattr(app.state, "request_count", 0)),
        "error_count": int(getattr(app.state, "error_count", 0)),
        "max_concurrency": max_concurrency,
        "request_timeout_seconds": float(getattr(app.state, "request_timeout_seconds", 0.0)),
        "queue_timeout_seconds": float(getattr(app.state, "queue_timeout_seconds", 0.0)),
        "timed_out_inflight": timed_out_inflight,
        "detached_inflight": detached_inflight,
        "cancelled_request_count": int(getattr(app.state, "cancelled_request_count", 0)),
        "queue_rejection_count": int(getattr(app.state, "queue_rejection_count", 0)),
        "coalesced_request_count": int(getattr(app.state, "coalesced_request_count", 0)),
        "coalesced_inflight": len(getattr(app.state, "result_inflight", {}) or {}),
        "result_cache_size": len(getattr(app.state, "result_cache", {}) or {}),
        "result_cache_max_size": int(getattr(app.state, "result_cache_max_size", 0)),
        "uptime_seconds": round(max(0.0, time.monotonic() - started_at), 3),
    }


def _normalize_cache_query(
    query: str,
    *,
    original_query: str | None = None,
    skip_query_planner: bool = False,
    navigation_context: Mapping[str, Any] | None = None,
    context_page_url: str | None = None,
) -> str:
    def normalize_text(value: Any) -> str:
        normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
        # Retrieval is invariant to casing, surrounding punctuation, repeated
        # separators, and typographic quote variants. Preserve letters and
        # numbers from every language while normalizing those presentation-only
        # differences into one reusable cache key.
        return " ".join(re.findall(r"[^\W_]+", normalized, flags=re.UNICODE))

    normalized_query = normalize_text(query)
    if not normalized_query:
        return ""
    navigation = dict(navigation_context or {})
    navigation_key = json.dumps(
        {
            "intent": normalize_text(navigation.get("intent")),
            "goal": normalize_text(navigation.get("goal")),
            "confidence": round(float(navigation.get("confidence") or 0.0), 2),
            "source": normalize_text(navigation.get("source")),
        }
        if navigation
        else {},
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    original_key = normalize_text(original_query)
    return (
        f"planner-skip={int(bool(skip_query_planner))}:"
        f"navigation={navigation_key}:context-page={str(context_page_url or '').casefold()}:"
        f"original={original_key}:{normalized_query}"
    )


async def _claim_inflight_result(app: FastAPI, cache_key: str) -> tuple[asyncio.Future, bool]:
    """Coalesce equivalent cold requests before they consume retriever slots."""
    async with app.state.result_cache_lock:
        existing = app.state.result_inflight.get(cache_key)
        if existing is not None:
            app.state.coalesced_request_count += 1
            return existing, False
        future = asyncio.get_running_loop().create_future()
        app.state.result_inflight[cache_key] = future
        return future, True


async def _finish_inflight_result(
    app: FastAPI,
    cache_key: str,
    future: asyncio.Future,
    *,
    payload: Dict[str, Any] | None = None,
    status_code: int | None = None,
    detail: str | None = None,
) -> None:
    result = {
        "ok": status_code is None,
        "payload": copy.deepcopy(payload or {}),
        "status_code": int(status_code or 200),
        "detail": str(detail or "retrieval_failed"),
    }
    async with app.state.result_cache_lock:
        if app.state.result_inflight.get(cache_key) is future:
            app.state.result_inflight.pop(cache_key, None)
        if not future.done():
            future.set_result(result)


async def _get_cached_result(
    app: FastAPI,
    query: str,
    *,
    original_query: str | None = None,
    skip_query_planner: bool = False,
    navigation_context: Mapping[str, Any] | None = None,
    context_page_url: str | None = None,
) -> Dict[str, Any] | None:
    cache = getattr(app.state, "result_cache", None)
    if not cache:
        return None
    cache_key = _normalize_cache_query(
        query,
        original_query=original_query,
        skip_query_planner=skip_query_planner,
        navigation_context=navigation_context,
        context_page_url=context_page_url,
    )
    if not cache_key:
        return None
    async with app.state.result_cache_lock:
        cached = cache.get(cache_key)
        if not cached:
            return None
        cached_at, payload = cached
        ttl_seconds = float(getattr(app.state, "result_cache_ttl_seconds", 0.0) or 0.0)
        if ttl_seconds > 0 and time.monotonic() - float(cached_at) > ttl_seconds:
            cache.pop(cache_key, None)
            return None
        cache.move_to_end(cache_key)
        return copy.deepcopy(payload)


async def _cache_result(
    app: FastAPI,
    query: str,
    payload: Dict[str, Any],
    *,
    original_query: str | None = None,
    skip_query_planner: bool = False,
    navigation_context: Mapping[str, Any] | None = None,
    context_page_url: str | None = None,
) -> None:
    cache = getattr(app.state, "result_cache", None)
    max_size = int(getattr(app.state, "result_cache_max_size", 0) or 0)
    if cache is None or max_size <= 0:
        return
    cache_key = _normalize_cache_query(
        query,
        original_query=original_query,
        skip_query_planner=skip_query_planner,
        navigation_context=navigation_context,
        context_page_url=context_page_url,
    )
    if not cache_key:
        return
    async with app.state.result_cache_lock:
        cache[cache_key] = (time.monotonic(), copy.deepcopy(payload))
        cache.move_to_end(cache_key)
        while len(cache) > max_size:
            cache.popitem(last=False)


def create_retrieval_service_app(
    *,
    config_name: str,
    work_dir: str | Path,
    max_concurrency: int = 3,
    request_timeout_seconds: float = 90.0,
    queue_timeout_seconds: float = 3.0,
) -> FastAPI:
    # Load the optional local environment before reading service-local auth,
    # cache, probe, and shutdown settings. Injected production variables win.
    _load_env_files()
    config_name = str(config_name or "").strip()
    resolved_work_dir = Path(work_dir).resolve()
    if not config_name:
        raise ValueError("config_name is required")
    if not resolved_work_dir.exists():
        raise FileNotFoundError(f"Retrieval work directory does not exist: {resolved_work_dir}")
    bounded_concurrency = max(1, int(max_concurrency))
    bounded_timeout = max(1.0, float(request_timeout_seconds))
    bounded_queue_timeout = max(0.05, float(queue_timeout_seconds))
    try:
        config_preview = load_config(config_name)
    except FileNotFoundError:
        config_preview = {}
    preview_pipeline = (
        config_preview.get("pipeline")
        if isinstance(config_preview.get("pipeline"), Mapping)
        else {}
    )
    production_config = bool(preview_pipeline.get("production_profile", False))
    service_token = str(os.getenv("RETRIEVAL_SERVICE_TOKEN") or "")
    token_error = _service_token_error(service_token) if service_token else "is required"
    if production_config and token_error:
        raise ValueError(f"RETRIEVAL_SERVICE_TOKEN {token_error} in production")
    if service_token and token_error:
        raise ValueError(f"RETRIEVAL_SERVICE_TOKEN {token_error}")
    if production_config:
        runtime_config = load_effective_config(config_name, work_dir=resolved_work_dir)
        runtime_config = apply_vector_upload_manifest_config(
            runtime_config,
            resolved_work_dir,
        )
        vector_store_cfg = (
            runtime_config.get("vector_store")
            if isinstance(runtime_config.get("vector_store"), Mapping)
            else {}
        )
        vector_provider = str(vector_store_cfg.get("provider") or "pinecone").strip().lower()
        provider_credentials = [
            ("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
            (
                "GOOGLE_API_KEY or GEMINI_API_KEY",
                os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            ),
        ]
        if vector_provider == "pinecone":
            provider_credentials.append(("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY")))
        provider_errors = [
            f"{name} {error}"
            for name, value in provider_credentials
            if (error := _provider_credential_error(value))
        ]
        if provider_errors:
            raise ValueError(
                "Production retriever has invalid provider credentials: "
                + "; ".join(provider_errors)
            )
        if vector_provider == "pgvector":
            from pipeline.vectorstores.pgvector_store import PgVectorSettings

            PgVectorSettings.from_config(runtime_config, purpose="read")
    startup_probe_required = production_config or _env_bool(
        "RETRIEVER_STARTUP_PROBE_REQUIRED",
        default=False,
    )
    startup_probe_query = str(
        os.getenv("RETRIEVER_STARTUP_PROBE_QUERY") or "Where is MBZUAI located?"
    ).strip()
    if startup_probe_required and not startup_probe_query:
        raise ValueError("RETRIEVER_STARTUP_PROBE_QUERY must not be empty")
    startup_probe_timeout = max(
        1.0,
        float(os.getenv("RETRIEVER_STARTUP_PROBE_TIMEOUT_SECONDS", str(bounded_timeout))),
    )
    startup_probe_operation_timeout = max(
        1.0,
        min(
            startup_probe_timeout,
            float(
                os.getenv(
                    "RETRIEVER_STARTUP_PROBE_OPERATION_TIMEOUT_SECONDS",
                    str(min(15.0, startup_probe_timeout)),
                )
            ),
        ),
    )
    shutdown_drain_timeout = max(
        0.0,
        float(os.getenv("RETRIEVER_SHUTDOWN_DRAIN_TIMEOUT_SECONDS", "30")),
    )
    result_cache_size = max(
        0,
        int(os.getenv("RETRIEVAL_SERVICE_RESULT_CACHE_SIZE", "256") or "0"),
    )
    result_cache_ttl_seconds = max(
        0.0,
        float(os.getenv("RETRIEVAL_SERVICE_RESULT_CACHE_TTL_SECONDS", "300") or "0"),
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.started_at_monotonic = time.monotonic()
        app.state.ready = False
        app.state.request_count = 0
        app.state.error_count = 0
        app.state.timed_out_inflight = 0
        app.state.detached_inflight = 0
        app.state.cancelled_request_count = 0
        app.state.queue_rejection_count = 0
        app.state.coalesced_request_count = 0
        app.state.config_name = config_name
        # These values are populated only after the deployment wrapper has
        # validated the active release. They are intentionally non-secret and
        # let readiness checks prove which immutable code/artifact pair is live.
        app.state.release_id = os.getenv("RETRIEVAL_RELEASE_ID") or None
        app.state.release_run_id = os.getenv("RETRIEVAL_RELEASE_RUN_ID") or None
        app.state.release_commit_sha = os.getenv("RELEASE_COMMIT_SHA") or None
        app.state.retrieval_bundle_sha256 = os.getenv("RETRIEVAL_BUNDLE_SHA256") or None
        app.state.knowledge_graph_sha256 = os.getenv("RETRIEVAL_KNOWLEDGE_GRAPH_SHA256") or None
        app.state.knowledge_graph_index_sha256 = os.getenv("RETRIEVAL_KNOWLEDGE_GRAPH_INDEX_SHA256") or None
        app.state.lexical_corpus_sha256 = os.getenv("RETRIEVAL_LEXICAL_CORPUS_SHA256") or None
        app.state.promoted_assertions_sha256 = os.getenv("RETRIEVAL_PROMOTED_ASSERTIONS_SHA256") or None
        app.state.selected_release_assembly_sha256 = os.getenv(
            "RETRIEVAL_SELECTED_RELEASE_ASSEMBLY_SHA256"
        ) or None
        app.state.selected_release_binding_sha256 = os.getenv(
            "RETRIEVAL_SELECTED_RELEASE_BINDING_SHA256"
        ) or None
        app.state.page_graph_navigation_catalog_sha256 = os.getenv(
            "RETRIEVAL_PAGE_GRAPH_NAVIGATION_CATALOG_SHA256"
        ) or None
        app.state.answer_runtime_commit_sha = os.getenv("RETRIEVAL_ANSWER_RUNTIME_COMMIT_SHA") or None
        app.state.indexing_build_commit_sha = os.getenv("RETRIEVAL_INDEXING_BUILD_COMMIT_SHA") or None
        app.state.startup_probe_required = startup_probe_required
        app.state.startup_probe_passed = False
        app.state.work_dir = resolved_work_dir
        app.state.max_concurrency = bounded_concurrency
        app.state.request_timeout_seconds = bounded_timeout
        app.state.queue_timeout_seconds = bounded_queue_timeout
        app.state.semaphore = asyncio.Semaphore(bounded_concurrency)
        app.state.executor = ThreadPoolExecutor(
            max_workers=bounded_concurrency,
            thread_name_prefix="mbzuai-retriever",
        )
        app.state.result_cache = OrderedDict()
        app.state.result_cache_lock = asyncio.Lock()
        app.state.result_cache_max_size = result_cache_size
        app.state.result_cache_ttl_seconds = result_cache_ttl_seconds
        app.state.result_inflight = {}
        app.state.inflight_futures = set()
        logger.info(
            "Loading retrieval service: config=%s work_dir=%s max_concurrency=%s timeout=%ss",
            config_name,
            resolved_work_dir,
            bounded_concurrency,
            bounded_timeout,
        )
        app.state.retriever = None
        try:
            app.state.retriever = AdaptiveHybridRetriever.from_config(
                config_name=config_name,
                work_dir=resolved_work_dir,
            )
            if getattr(app.state.retriever, "supports_shared_parallel_retrieval", True) is False:
                if app.state.max_concurrency > 1:
                    logger.warning(
                        "Retriever does not support shared parallel calls; capping service concurrency from %s to 1",
                        app.state.max_concurrency,
                    )
                app.state.max_concurrency = 1
                app.state.semaphore = asyncio.Semaphore(1)
            if startup_probe_required:
                loop = asyncio.get_running_loop()
                probe_future = loop.run_in_executor(
                    app.state.executor,
                    lambda: _run_startup_probe(
                        app.state.retriever,
                        work_dir=resolved_work_dir,
                        query=startup_probe_query,
                        operation_timeout_seconds=startup_probe_operation_timeout,
                    ),
                )
                app.state.inflight_futures.add(probe_future)
                probe_future.add_done_callback(app.state.inflight_futures.discard)
                await asyncio.wait_for(
                    asyncio.shield(probe_future),
                    timeout=startup_probe_timeout,
                )
                app.state.startup_probe_passed = True
            app.state.ready = True
            logger.info("Retrieval service ready: config=%s work_dir=%s", config_name, resolved_work_dir)
            yield
        finally:
            app.state.ready = False
            pending = [
                future
                for future in list(getattr(app.state, "inflight_futures", set()))
                if not future.done()
            ]
            if pending and shutdown_drain_timeout > 0:
                _done, pending_set = await asyncio.wait(
                    pending,
                    timeout=shutdown_drain_timeout,
                )
                pending = list(pending_set)
            if pending:
                # Do not close shared clients while worker threads still use
                # them. The process supervisor owns forced termination after
                # its grace period; provider deadlines bound normal drains.
                logger.error(
                    "Retriever shutdown drain timed out with %d provider task(s) still active",
                    len(pending),
                )
            else:
                close_retriever = getattr(app.state.retriever, "close", None) if app.state.retriever else None
                if callable(close_retriever):
                    close_retriever()
            app.state.executor.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(
        title="MBZUAI Retrieval Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _public_health_payload() -> Dict[str, Any]:
        payload = _health_payload(app)
        return {
            "ok": True,
            "service": "retriever",
            "ready": payload["ready"],
            "uptime_seconds": payload["uptime_seconds"],
        }

    def _authorize(request: Request) -> None:
        if not service_token:
            return
        provided_token = request.headers.get("X-Retrieval-Service-Token", "")
        if not hmac.compare_digest(provided_token, service_token):
            raise HTTPException(status_code=401, detail="unauthorized")

    async def _health_response() -> Dict[str, Any]:
        return _public_health_payload()

    async def _ready_response() -> Dict[str, Any]:
        payload = _public_health_payload()
        if not payload["ready"]:
            raise HTTPException(status_code=503, detail="retriever_not_ready")
        return payload

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return await _health_response()

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        return await _health_response()

    @app.get("/readyz")
    async def readyz() -> Dict[str, Any]:
        return await _ready_response()

    @app.get("/ready")
    async def ready() -> Dict[str, Any]:
        return await _ready_response()

    @app.get("/attestationz")
    async def attestationz(request: Request) -> Any:
        _authorize(request)
        payload = _health_payload(app)
        if not payload["ready"]:
            return JSONResponse(status_code=503, content=payload)
        return payload

    @app.post("/retrieve")
    async def retrieve_endpoint(payload: RetrieveRequest, request: Request) -> Dict[str, Any]:
        _authorize(request)
        if not bool(getattr(app.state, "ready", False)):
            raise HTTPException(status_code=503, detail="retriever_not_ready")
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query must not be empty")
        original_query = (
            payload.original_query.strip()
            if payload.original_query is not None
            else None
        )
        if original_query == query:
            original_query = None

        request_id = payload.request_id or str(uuid.uuid4())
        started_at = time.perf_counter()
        retriever = app.state.retriever
        semaphore = app.state.semaphore
        executor = app.state.executor
        app.state.request_count += 1
        navigation_context = (
            payload.navigation_context.model_dump()
            if payload.navigation_context is not None
            else None
        )
        context_page_url = _validated_context_page_url(payload.context_page_url)
        if payload.context_page_url and not context_page_url:
            raise HTTPException(status_code=400, detail="context_page_url_not_official")
        cache_key = _normalize_cache_query(
            query,
            original_query=original_query,
            skip_query_planner=payload.skip_query_planner,
            navigation_context=navigation_context,
            context_page_url=context_page_url,
        )
        cached_result = await _get_cached_result(
            app,
            query,
            original_query=original_query,
            skip_query_planner=payload.skip_query_planner,
            navigation_context=navigation_context,
            context_page_url=context_page_url,
        )
        if cached_result is not None:
            output = dict(cached_result or {})
            output["service_request_id"] = request_id
            output["service_latency_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
            output["service_backend"] = "retrieval_service"
            output["service_config_name"] = app.state.config_name
            output["service_cache_hit"] = True
            output["service_coalesced"] = False
            output["service_query_planner_skipped"] = bool(payload.skip_query_planner)
            output["service_original_query_forwarded"] = bool(original_query)
            output["service_navigation_context_forwarded"] = bool(
                navigation_context
            )
            return output
        inflight_result, owns_inflight_result = await _claim_inflight_result(
            app,
            cache_key,
        )
        if not owns_inflight_result:
            try:
                shared = await asyncio.wait_for(
                    asyncio.shield(inflight_result),
                    timeout=(
                        app.state.queue_timeout_seconds
                        + app.state.request_timeout_seconds
                        + 1.0
                    ),
                )
            except asyncio.TimeoutError as exc:
                app.state.error_count += 1
                raise HTTPException(
                    status_code=504,
                    detail="retrieval_coalesced_timeout",
                ) from exc
            if not bool(shared.get("ok")):
                app.state.error_count += 1
                raise HTTPException(
                    status_code=int(shared.get("status_code") or 500),
                    detail=str(shared.get("detail") or "retrieval_failed"),
                )
            output = dict(copy.deepcopy(shared.get("payload") or {}))
            output["service_request_id"] = request_id
            output["service_latency_ms"] = round(
                (time.perf_counter() - started_at) * 1000.0,
                3,
            )
            output["service_backend"] = "retrieval_service"
            output["service_config_name"] = app.state.config_name
            output["service_cache_hit"] = False
            output["service_coalesced"] = True
            output["service_query_planner_skipped"] = bool(payload.skip_query_planner)
            output["service_original_query_forwarded"] = bool(original_query)
            output["service_navigation_context_forwarded"] = bool(
                navigation_context
            )
            output["service_context_page_forwarded"] = bool(context_page_url)
            return output
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=app.state.queue_timeout_seconds)
        except asyncio.TimeoutError as exc:
            app.state.error_count += 1
            app.state.queue_rejection_count += 1
            await _finish_inflight_result(
                app,
                cache_key,
                inflight_result,
                status_code=503,
                detail="retrieval_busy",
            )
            raise HTTPException(status_code=503, detail="retrieval_busy") from exc

        release_capacity_on_exit = True

        def _retain_capacity_until_done(future, *, timed_out: bool) -> None:
            app.state.detached_inflight += 1
            if timed_out:
                app.state.timed_out_inflight += 1

            def _release_detached_capacity(_future) -> None:
                app.state.detached_inflight = max(0, int(app.state.detached_inflight) - 1)
                if timed_out:
                    app.state.timed_out_inflight = max(0, int(app.state.timed_out_inflight) - 1)
                semaphore.release()

            future.add_done_callback(_release_detached_capacity)

        try:
            loop = asyncio.get_running_loop()
            retrieval_options: Dict[str, Any] = {}
            if payload.skip_query_planner:
                retrieval_options["skip_query_planner"] = True
            if navigation_context is not None:
                retrieval_options["navigation_context"] = navigation_context
            if context_page_url is not None:
                retrieval_options["context_page_url"] = context_page_url
            original_query_forwarded = False
            if original_query is not None:
                try:
                    retrieve_parameters = inspect.signature(retriever.retrieve).parameters
                    accepts_original_query = (
                        "original_query" in retrieve_parameters
                        or any(
                            parameter.kind == inspect.Parameter.VAR_KEYWORD
                            for parameter in retrieve_parameters.values()
                        )
                    )
                except (TypeError, ValueError):
                    accepts_original_query = False
                if accepts_original_query:
                    retrieval_options["original_query"] = original_query
                    original_query_forwarded = True
            retrieval_future = loop.run_in_executor(
                executor,
                lambda: retriever.retrieve(query, **retrieval_options),
            )
            app.state.inflight_futures.add(retrieval_future)
            retrieval_future.add_done_callback(app.state.inflight_futures.discard)
            result = await asyncio.wait_for(
                asyncio.shield(retrieval_future),
                timeout=app.state.request_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            app.state.error_count += 1
            release_capacity_on_exit = False
            _retain_capacity_until_done(retrieval_future, timed_out=True)
            await _finish_inflight_result(
                app,
                cache_key,
                inflight_result,
                status_code=504,
                detail="retrieval_timeout",
            )
            raise HTTPException(status_code=504, detail="retrieval_timeout") from exc
        except asyncio.CancelledError:
            # Client disconnects cancel the ASGI task, but Python cannot stop the
            # synchronous retriever thread. Keep its capacity reserved until the
            # worker genuinely exits so abandoned work cannot overrun the pool.
            app.state.cancelled_request_count += 1
            release_capacity_on_exit = False
            _retain_capacity_until_done(retrieval_future, timed_out=False)
            await _finish_inflight_result(
                app,
                cache_key,
                inflight_result,
                status_code=503,
                detail="retrieval_leader_cancelled",
            )
            raise
        except HTTPException as exc:
            app.state.error_count += 1
            await _finish_inflight_result(
                app,
                cache_key,
                inflight_result,
                status_code=exc.status_code,
                detail=str(exc.detail or "retrieval_failed"),
            )
            raise
        except Exception as exc:  # pragma: no cover - exercised in live validation
            app.state.error_count += 1
            logger.exception("Retrieval request failed: request_id=%s path=%s", request_id, request.url.path)
            await _finish_inflight_result(
                app,
                cache_key,
                inflight_result,
                status_code=500,
                detail="retrieval_failed",
            )
            raise HTTPException(status_code=500, detail="retrieval_failed") from exc
        finally:
            if release_capacity_on_exit:
                semaphore.release()

        output = dict(result or {})
        await _cache_result(
            app,
            query,
            output,
            original_query=original_query,
            skip_query_planner=payload.skip_query_planner,
            navigation_context=navigation_context,
            context_page_url=context_page_url,
        )
        await _finish_inflight_result(
            app,
            cache_key,
            inflight_result,
            payload=output,
        )
        output["service_request_id"] = request_id
        output["service_latency_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        output["service_backend"] = "retrieval_service"
        output["service_config_name"] = app.state.config_name
        output["service_cache_hit"] = False
        output["service_coalesced"] = False
        output["service_query_planner_skipped"] = bool(payload.skip_query_planner)
        output["service_original_query_forwarded"] = bool(original_query_forwarded)
        output["service_navigation_context_forwarded"] = bool(navigation_context)
        output["service_context_page_forwarded"] = bool(context_page_url)
        return output

    return app
