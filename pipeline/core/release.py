from __future__ import annotations

import asyncio
import errno
import fcntl
import hmac
import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Tuple

from pipeline.core.config import (
    indexing_implementation_hashes,
    load_config,
    load_effective_config,
    production_indexing_contract_fingerprint,
    production_serving_contract_fingerprint,
)
from pipeline.core.graph_artifacts import (
    GraphArtifactContractError,
    resolve_canonical_graph_artifacts,
)
from pipeline.core.io import (
    atomic_write_json,
    combine_sha256_digests,
    load_json_safe,
    sha256_file,
)
from pipeline.core.knowledge_graph import (
    community_summary_quality,
    load_graph_bundle,
    validate_graph_bundle,
    validate_graph_index_derivation,
)
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.core.preflight import assess_production_readiness
from pipeline.core.run_audit import audit_run
from pipeline.core.release_policy import (
    production_eval_manifest_metadata,
    validate_production_eval_inputs,
    validate_production_eval_manifest,
)
from pipeline.evaluation import evaluate_answer_readiness, evaluate_retrieval_dataset
from pipeline.evaluation.dataset_tools import validate_eval_examples
from pipeline.retrieval.adaptive_hybrid import apply_vector_upload_manifest_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_DATASET = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_multilingual_v2.jsonl"
DEFAULT_RELEASE_GATES = (
    PROJECT_ROOT / "eval" / "gates" / "retrieval_gate.multilingual_v2_release.json"
)
DEFAULT_RELEASE_ANSWER_DATASET = DEFAULT_RELEASE_DATASET
DEFAULT_RELEASE_ANSWER_GATES = (
    PROJECT_ROOT / "eval" / "gates" / "answer_readiness_gate.multilingual_v2_release.json"
)
LEGACY_ONLY_VALIDATION_PLUGINS = {"mbzuai_legacy_vectorstores", "mbzuai_legacy_pinecone"}
RELEASE_MUTATING_VALIDATION_PLUGINS = {"gemini_pgvector", "gemini_pinecone"}
ReleaseProgressCallback = Callable[[str, Mapping[str, Any]], None]
PROMOTION_ATTESTATION_SCHEMA_VERSION = 1
PROMOTION_ATTESTATION_MAX_AGE_SECONDS = 120.0
PROMOTION_ATTESTATION_MAX_FUTURE_SKEW_SECONDS = 5.0
_PROMOTION_ANSWER_MODEL_KEYS = (
    "generation_model",
    "query_rewrite_model",
    "reranker_model",
    "grounded_finalizer_model",
)
_PROMOTION_ATTESTATION_KEYS = {
    "schema_version",
    "attested_at",
    "release_manifest_sha256",
    "release_id",
    "config_name",
    "run_id",
    "backend_commit_sha",
    "retriever_commit_sha",
    "indexing_build_commit_sha",
    "retrieval_bundle_sha256",
    "knowledge_graph_sha256",
    "knowledge_graph_index_sha256",
    "lexical_corpus_sha256",
    "promoted_assertions_sha256",
    "answer_models",
    "signature_sha256",
}
_PROMOTION_SELECTED_ATTESTATION_KEYS = {
    "selected_release_assembly_sha256",
    "selected_release_binding_sha256",
    "page_graph_navigation_catalog_sha256",
}


@contextmanager
def _active_pointer_lock(active_release_file: str | Path):
    if os.getenv("RELEASE_POINTER_LOCK_HELD", "false").lower() == "true":
        yield
        return
    active_path = Path(active_release_file).expanduser().resolve()
    lock_path = active_path.with_name(active_path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o640)
    timeout_seconds = max(0.0, float(os.getenv("RELEASE_LOCK_TIMEOUT_SECONDS", "30")))
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for active release pointer lock: {lock_path}"
                    ) from exc
                time.sleep(0.2)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _serialize_active_pointer_update(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        active_release_file = kwargs.get("active_release_file")
        if active_release_file is None:
            raise TypeError("active_release_file must be passed by keyword")
        with _active_pointer_lock(active_release_file):
            return function(*args, **kwargs)

    return wrapped


def _emit_progress(
    callback: ReleaseProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        return


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _promotion_attestation_signing_key() -> bytes:
    values = (
        ("RETRIEVAL_SERVICE_TOKEN", os.getenv("RETRIEVAL_SERVICE_TOKEN", "")),
        (
            "CANDIDATE_BACKEND_OPERATIONS_TOKEN",
            os.getenv("CANDIDATE_BACKEND_OPERATIONS_TOKEN", ""),
        ),
    )
    material = bytearray(b"mbzuai-production-promotion-attestation-v1\0")
    for name, value in values:
        encoded = str(value or "").encode("utf-8")
        if len(encoded) < 32 or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in str(value or "")
        ):
            raise ValueError(f"{name} is required to authenticate promotion evidence")
        material.extend(len(encoded).to_bytes(4, "big"))
        material.extend(encoded)
    return hashlib.sha256(bytes(material)).digest()


def _promotion_attestation_signature(evidence: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in evidence.items() if key != "signature_sha256"}
    canonical = json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(
        _promotion_attestation_signing_key(),
        canonical,
        hashlib.sha256,
    ).hexdigest()


def _promotion_answer_models(config: Mapping[str, Any]) -> Dict[str, str]:
    serving = config.get("serving") if isinstance(config.get("serving"), Mapping) else {}
    return {
        key: str(serving.get(key) or "").strip()
        for key in _PROMOTION_ANSWER_MODEL_KEYS
    }


def _promotion_release_identity(
    manifest: Mapping[str, Any],
    *,
    retriever_commit_sha: str,
) -> Dict[str, str]:
    indexing_build = (
        manifest.get("indexing_build")
        if isinstance(manifest.get("indexing_build"), Mapping)
        else {}
    )
    vector_index = (
        manifest.get("vector_index")
        if isinstance(manifest.get("vector_index"), Mapping)
        else {}
    )
    retrieval_bundle = (
        manifest.get("retrieval_bundle")
        if isinstance(manifest.get("retrieval_bundle"), Mapping)
        else {}
    )
    identity = {
        "run_id": str(manifest.get("run_id") or "").strip(),
        "retriever_commit_sha": str(retriever_commit_sha or "").strip().lower(),
        "indexing_build_commit_sha": str(indexing_build.get("commit_sha") or "").strip().lower(),
        "retrieval_bundle_sha256": str(
            retrieval_bundle.get("retrieval_bundle_sha256")
            or vector_index.get("retrieval_bundle_sha256")
            or ""
        ).strip().lower(),
        "knowledge_graph_sha256": str(vector_index.get("knowledge_graph_sha256") or "").strip().lower(),
        "knowledge_graph_index_sha256": str(
            vector_index.get("knowledge_graph_index_sha256") or ""
        ).strip().lower(),
        "lexical_corpus_sha256": str(vector_index.get("lexical_corpus_sha256") or "").strip().lower(),
        "promoted_assertions_sha256": str(
            vector_index.get("promoted_assertions_sha256") or ""
        ).strip().lower(),
    }
    if str(vector_index.get("selected_release_assembly_sha256") or "").strip():
        identity.update(
            {
                key: str(vector_index.get(key) or "").strip().lower()
                for key in _PROMOTION_SELECTED_ATTESTATION_KEYS
            }
        )
    return identity


def build_promotion_attestation_evidence(
    *,
    manifest_path: str | Path,
    retriever_attestation: Mapping[str, Any],
    backend_health: Mapping[str, Any],
    expected_retriever_commit_sha: str,
    expected_backend_commit_sha: str,
    attested_at: str | None = None,
) -> Dict[str, Any]:
    """Create non-secret promotion evidence from freshly verified candidate health.

    The returned object is deliberately strict and contains only immutable
    identities. Candidate credentials, headers, URLs, and raw health payloads
    are never persisted in the release manifest.
    """
    resolved_manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = load_json_safe(resolved_manifest_path, None)
    if not isinstance(manifest, dict):
        raise ValueError("promotion release manifest is missing or invalid")
    if manifest.get("status") not in {"passed", "passed_with_waiver"}:
        raise ValueError("promotion release manifest has not passed its gates")
    if manifest.get("promoted") is True or manifest.get("promotion_attestation"):
        raise ValueError("promotion release manifest already contains promotion evidence")
    manifest_config_name = str(manifest.get("config_name") or "").strip()
    try:
        manifest_config = load_config(manifest_config_name)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError("promotion release manifest config is missing or invalid") from exc
    if not bool((manifest_config.get("pipeline") or {}).get("production_profile", False)):
        raise ValueError("promotion attestation evidence requires a production profile")

    retriever_commit = str(expected_retriever_commit_sha or "").strip().lower()
    backend_commit = str(expected_backend_commit_sha or "").strip().lower()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", retriever_commit):
        raise ValueError("expected retriever commit SHA is missing or invalid")
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", backend_commit):
        raise ValueError("expected backend commit SHA is missing or invalid")

    expected_identity = _promotion_release_identity(
        manifest,
        retriever_commit_sha=retriever_commit,
    )
    required_identity_values = {
        key: value for key, value in expected_identity.items() if key != "run_id"
    }
    if not expected_identity["run_id"]:
        raise ValueError("promotion release manifest is missing run_id")
    for key, value in required_identity_values.items():
        pattern = r"(?:[0-9a-f]{40}|[0-9a-f]{64})" if key in {
            "retriever_commit_sha",
            "indexing_build_commit_sha",
        } else r"[0-9a-f]{64}"
        if not re.fullmatch(pattern, value):
            raise ValueError(f"promotion release manifest has invalid {key}")

    if retriever_attestation.get("ready") is not True:
        raise ValueError("retriever-candidate is not ready")
    if str(retriever_attestation.get("config_name") or "") != manifest_config_name:
        raise ValueError(
            "retriever-candidate is not running the release manifest config"
        )
    retriever_field_map = {
        "run_id": "run_id",
        "retriever_commit_sha": "commit_sha",
        "indexing_build_commit_sha": "indexing_build_commit_sha",
        "retrieval_bundle_sha256": "retrieval_bundle_sha256",
        "knowledge_graph_sha256": "knowledge_graph_sha256",
        "knowledge_graph_index_sha256": "knowledge_graph_index_sha256",
        "lexical_corpus_sha256": "lexical_corpus_sha256",
        "promoted_assertions_sha256": "promoted_assertions_sha256",
    }
    for key in _PROMOTION_SELECTED_ATTESTATION_KEYS:
        if key in expected_identity:
            retriever_field_map[key] = key
    for identity_key, candidate_key in retriever_field_map.items():
        if str(retriever_attestation.get(candidate_key) or "").strip().lower() != expected_identity[identity_key]:
            raise ValueError(f"retriever-candidate identity mismatch for {candidate_key}")

    if backend_health.get("status") != "healthy":
        raise ValueError("backend-candidate is not healthy")
    actual_backend_commit = str(
        ((backend_health.get("release") or {}).get("commit_sha") if isinstance(backend_health.get("release"), Mapping) else "")
        or ""
    ).strip().lower()
    if actual_backend_commit != backend_commit:
        raise ValueError("backend-candidate code revision mismatch")
    answer_runtime = (
        manifest.get("answer_runtime")
        if isinstance(manifest.get("answer_runtime"), Mapping)
        else {}
    )
    if str(answer_runtime.get("commit_sha") or "").strip().lower() != backend_commit:
        raise ValueError("backend-candidate commit does not match the evaluated answer runtime")

    backend_retrieval = (
        (backend_health.get("checks") or {}).get("retrieval_service")
        if isinstance(backend_health.get("checks"), Mapping)
        else {}
    )
    if not isinstance(backend_retrieval, Mapping):
        backend_retrieval = {}
    if (
        backend_retrieval.get("status") != "healthy"
        or backend_retrieval.get("ready") is not True
        or backend_retrieval.get("mode") != "required"
    ):
        raise ValueError("backend-candidate is not using a healthy required retrieval service")
    for identity_key, candidate_key in retriever_field_map.items():
        if str(backend_retrieval.get(candidate_key) or "").strip().lower() != expected_identity[identity_key]:
            raise ValueError(f"backend-candidate retrieval identity mismatch for {candidate_key}")

    backend_openai = (
        (backend_health.get("checks") or {}).get("openai")
        if isinstance(backend_health.get("checks"), Mapping)
        else {}
    )
    if not isinstance(backend_openai, Mapping) or backend_openai.get("status") != "healthy":
        raise ValueError("backend-candidate OpenAI configuration is not healthy")
    observed_models = {
        key: str(backend_openai.get(key) or "").strip()
        for key in _PROMOTION_ANSWER_MODEL_KEYS
    }
    if any(not value for value in observed_models.values()):
        raise ValueError("backend-candidate answer model identity is incomplete")

    evidence = {
        "schema_version": (
            2
            if _PROMOTION_SELECTED_ATTESTATION_KEYS.intersection(expected_identity)
            else PROMOTION_ATTESTATION_SCHEMA_VERSION
        ),
        "attested_at": str(attested_at or _now_iso()),
        "release_manifest_sha256": _stable_json_sha256(manifest),
        "release_id": str(manifest.get("release_id") or ""),
        "config_name": str(manifest.get("config_name") or ""),
        **expected_identity,
        "backend_commit_sha": backend_commit,
        "answer_models": observed_models,
    }
    evidence["signature_sha256"] = _promotion_attestation_signature(evidence)
    return evidence


def _validate_promotion_attestation(
    manifest: Mapping[str, Any],
    evidence: Mapping[str, Any] | None,
    *,
    current_runtime_config: Mapping[str, Any],
    now: datetime | None = None,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(evidence, Mapping):
        return ["fresh promotion attestation evidence is required"]
    vector_index = (
        manifest.get("vector_index")
        if isinstance(manifest.get("vector_index"), Mapping)
        else {}
    )
    selected_attestation = bool(
        str(vector_index.get("selected_release_assembly_sha256") or "").strip()
    )
    expected_attestation_keys = set(_PROMOTION_ATTESTATION_KEYS)
    if selected_attestation:
        expected_attestation_keys.update(_PROMOTION_SELECTED_ATTESTATION_KEYS)
    evidence_keys = set(evidence)
    if evidence_keys != expected_attestation_keys:
        missing = sorted(expected_attestation_keys - evidence_keys)
        unexpected = sorted(evidence_keys - expected_attestation_keys)
        if missing:
            errors.append(f"promotion attestation is missing fields: {missing}")
        if unexpected:
            errors.append(f"promotion attestation has unexpected fields: {unexpected}")
    expected_schema_version = 2 if selected_attestation else PROMOTION_ATTESTATION_SCHEMA_VERSION
    if _count(evidence.get("schema_version")) != expected_schema_version:
        errors.append("promotion attestation schema_version is unsupported")
    signature = str(evidence.get("signature_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", signature):
        errors.append("promotion attestation signature is invalid")
    else:
        try:
            expected_signature = _promotion_attestation_signature(evidence)
        except ValueError as exc:
            errors.append(str(exc))
        else:
            if not hmac.compare_digest(signature, expected_signature):
                errors.append("promotion attestation signature verification failed")

    timestamp_text = str(evidence.get("attested_at") or "").strip()
    try:
        timestamp = datetime.fromisoformat(timestamp_text.replace("Z", "+00:00"))
    except ValueError:
        errors.append("promotion attestation timestamp is invalid")
    else:
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            errors.append("promotion attestation timestamp must include a timezone")
        else:
            observed_now = now or datetime.now(timezone.utc)
            age_seconds = (observed_now - timestamp.astimezone(timezone.utc)).total_seconds()
            if age_seconds < -PROMOTION_ATTESTATION_MAX_FUTURE_SKEW_SECONDS:
                errors.append("promotion attestation timestamp is in the future")
            elif age_seconds > PROMOTION_ATTESTATION_MAX_AGE_SECONDS:
                errors.append("promotion attestation evidence is stale")

    expected_manifest_digest = _stable_json_sha256(manifest)
    manifest_digest = str(evidence.get("release_manifest_sha256") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_digest):
        errors.append("promotion attestation manifest digest is invalid")
    elif manifest_digest != expected_manifest_digest:
        errors.append("promotion attestation does not match the exact pre-promotion release manifest")

    identity = _promotion_release_identity(
        manifest,
        retriever_commit_sha=str(evidence.get("retriever_commit_sha") or ""),
    )
    expected_fields = {
        "release_id": str(manifest.get("release_id") or ""),
        "config_name": str(manifest.get("config_name") or ""),
        **identity,
        "backend_commit_sha": str(
            ((manifest.get("answer_runtime") or {}).get("commit_sha") if isinstance(manifest.get("answer_runtime"), Mapping) else "")
            or ""
        ).strip().lower(),
    }
    for key, expected_value in expected_fields.items():
        actual_value = str(evidence.get(key) or "").strip()
        if key.endswith("_sha") or key.endswith("_sha256"):
            actual_value = actual_value.lower()
        if not expected_value or actual_value != expected_value:
            errors.append(f"promotion attestation identity mismatch for {key}")

    if not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
        str(evidence.get("retriever_commit_sha") or "").strip().lower(),
    ):
        errors.append("promotion attestation retriever commit SHA is invalid")
    answer_models = evidence.get("answer_models")
    if not isinstance(answer_models, Mapping):
        errors.append("promotion attestation answer_models is invalid")
    else:
        if set(answer_models) != set(_PROMOTION_ANSWER_MODEL_KEYS):
            errors.append("promotion attestation answer_models fields are incomplete or unexpected")
        expected_models = _promotion_answer_models(current_runtime_config)
        for key, expected_model in expected_models.items():
            if not expected_model or str(answer_models.get(key) or "") != expected_model:
                errors.append(f"promotion attestation answer model mismatch for {key}")
    return errors


def _validation_config_for_release(config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    """Return a config suitable for release validation of the uploaded contract."""
    payload = deepcopy(config or {})
    pipeline = dict(payload.get("pipeline") or {})
    pipeline["validation_purpose"] = "release"
    payload["pipeline"] = pipeline
    modern_manifest = load_json_safe(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        {},
    ) or {}
    modern_release = isinstance(modern_manifest, dict) and bool(
        str(modern_manifest.get("index_name") or "").strip()
    )
    skipped_plugins = set(RELEASE_MUTATING_VALIDATION_PLUGINS)
    if modern_release:
        skipped_plugins.update(LEGACY_ONLY_VALIDATION_PLUGINS)

    stages = []
    for stage in payload.get("stages") or []:
        if not isinstance(stage, dict):
            stages.append(stage)
            continue
        plugin = str(stage.get("plugin") or "")
        if plugin in skipped_plugins:
            continue
        stages.append(stage)
    payload["stages"] = stages
    return payload


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def default_active_release_path(config: Dict[str, Any], work_dir: str | Path | None = None) -> Path:
    if work_dir:
        return Path(work_dir).expanduser().resolve().parent / "active_release.json"
    configured_root = str(config.get("work_dir") or "").strip()
    if configured_root:
        project_name = str(config.get("project_name") or "").strip()
        root = _resolve_project_path(configured_root)
        return (root / project_name / "active_release.json").resolve() if project_name else (root / "active_release.json").resolve()
    return PROJECT_ROOT / "runs" / "active_release.json"


def release_manifest_path(work_dir: str | Path) -> Path:
    return Path(work_dir).expanduser().resolve() / "release" / "retrieval_release_manifest.json"


def _load_required_json(path: Path, label: str) -> Tuple[Dict[str, Any], List[str]]:
    payload = load_json_safe(path, None)
    if not path.exists():
        return {}, [f"{label} is missing: {path}"]
    if not isinstance(payload, dict):
        return {}, [f"{label} is invalid JSON object: {path}"]
    return payload, []


_REQUIRED_RETRIEVAL_BUNDLE_STATS = (
    "chunk_count",
    "parent_count",
    "media_count",
    "fact_count",
    "evidence_span_count",
    "summary_count",
    "assertion_count",
    "answer_count",
    "lexical_count",
)
_POSITIVE_RETRIEVAL_BUNDLE_STATS = (
    "chunk_count",
    "parent_count",
    "fact_count",
    "evidence_span_count",
    "summary_count",
    "assertion_count",
    "lexical_count",
)
_VECTOR_UPLOAD_COUNT_MAP = {
    "chunks": "chunk_count",
    "parents": "parent_count",
    "media": "media_count",
    "page_cards": "page_card_count",
    "actions": "action_count",
    "facts": "fact_count",
    "evidence_spans": "evidence_span_count",
    "summaries": "summary_count",
    "assertions": "assertion_count",
}
_MODERN_VECTOR_NAMESPACE_KEYS = (
    "chunks",
    "parents",
    "media",
    "page_cards",
    "actions",
    "facts",
    "evidence_spans",
    "summaries",
    "assertions",
    "entities",
    "communities",
)
_PRE_SELECTED_VECTOR_NAMESPACE_KEYS = tuple(
    key for key in _MODERN_VECTOR_NAMESPACE_KEYS if key not in {"page_cards", "actions"}
)
_LEGACY_VECTORSTORE_CONTRACT = "mbzuai_chatbot_legacy_v1"


def _is_selected_vector_manifest(manifest: Mapping[str, Any]) -> bool:
    selected = manifest.get("selected_profile")
    return bool(
        (isinstance(selected, Mapping) and str(selected.get("variant_id") or "").strip())
        or str(manifest.get("selected_release_assembly_sha256") or "").strip()
    )


def _vector_namespace_keys(manifest: Mapping[str, Any]) -> Tuple[str, ...]:
    if _is_selected_vector_manifest(manifest) or _count(manifest.get("schema_version")) >= 6:
        return _MODERN_VECTOR_NAMESPACE_KEYS
    return _PRE_SELECTED_VECTOR_NAMESPACE_KEYS


def _stage_plugins(config: Mapping[str, Any]) -> List[str]:
    plugins: List[str] = []
    for stage in config.get("stages") or []:
        if isinstance(stage, Mapping) and stage.get("plugin"):
            plugins.append(str(stage["plugin"]))
    return plugins


def _graph_store_backend(config: Mapping[str, Any], stage_plugins: Iterable[str] | None = None) -> str:
    graph_cfg = config.get("graph", {}) if isinstance(config.get("graph"), Mapping) else {}
    configured = str(
        graph_cfg.get("store_backend")
        or graph_cfg.get("graph_store_backend")
        or ""
    ).strip().lower()
    if configured in {"none", "off", "disabled"}:
        return "disabled"
    if configured in {"local", "local_json", "json", "file", "files"}:
        return "local_json"
    if configured == "neo4j":
        return "neo4j"
    plugins = set(stage_plugins or _stage_plugins(config))
    if bool(graph_cfg.get("require_neo4j_upload", False)) or "neo4j_graph_store" in plugins:
        return "neo4j"
    return "local_json"


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _verify_retrieval_bundle_stats(
    stats: Dict[str, Any],
    *,
    selected_profile: bool = False,
) -> List[str]:
    errors: List[str] = []
    if not stats.get("retrieval_bundle_file"):
        return ["Retrieval bundle is missing"]
    minimum_version = 6 if selected_profile else 5
    if _count(stats.get("retrieval_bundle_version")) < minimum_version:
        errors.append(f"Retrieval bundle version must be {minimum_version} or newer")
    required_keys = list(_REQUIRED_RETRIEVAL_BUNDLE_STATS)
    if selected_profile:
        required_keys.extend(("page_card_count", "action_count"))
    missing = [key for key in required_keys if key not in stats]
    if missing:
        errors.append(f"Retrieval bundle stats are missing required keys: {missing}")
    positive_keys = (
        (
            "chunk_count",
            "parent_count",
            "media_count",
            "page_card_count",
            "action_count",
            "lexical_count",
        )
        if selected_profile
        else _POSITIVE_RETRIEVAL_BUNDLE_STATS
    )
    for key in positive_keys:
        if key in stats and _count(stats.get(key)) <= 0:
            errors.append(f"Retrieval bundle {key} must be greater than zero")
    return errors


def _expected_vector_upload_counts(bundle_stats: Dict[str, Any], *, sparse_enabled: bool) -> Dict[str, int]:
    expected: Dict[str, int] = {}
    for upload_key, stat_key in _VECTOR_UPLOAD_COUNT_MAP.items():
        expected_count = _count(bundle_stats.get(stat_key))
        if expected_count > 0:
            expected[upload_key] = expected_count
            if sparse_enabled:
                expected[f"sparse_{upload_key}"] = expected_count
    return expected


def _sparse_record_skipped_count(manifest: Mapping[str, Any], upload_key: str) -> int:
    sparse_payload = manifest.get("sparse") if isinstance(manifest.get("sparse"), Mapping) else {}
    record_stats = sparse_payload.get("record_stats") if isinstance(sparse_payload.get("record_stats"), Mapping) else {}
    singular = {
        "chunks": "chunk",
        "parents": "parent",
        "media": "media",
        "page_cards": "page_card",
        "actions": "action",
        "facts": "fact",
        "evidence_spans": "evidence_span",
        "summaries": "summary",
        "assertions": "assertion",
    }.get(upload_key, upload_key.rstrip("s"))
    return _count(record_stats.get(f"{singular}_records_skipped"))


def _modern_vector_expected_counts_for_manifest(
    manifest: Mapping[str, Any],
    bundle_stats: Mapping[str, Any],
    embedder_config: Mapping[str, Any] | None = None,
) -> Dict[str, int]:
    """Derive upload cardinality from the canonical retrieval bundle.

    The upload manifest is an observation, not the source of truth.  Dense
    totals must exactly match the bundle.  Sparse totals may only differ by
    explicit per-record skip counters emitted by the sparse encoder.
    """
    expected: Dict[str, int] = {}
    embedder_config = embedder_config if isinstance(embedder_config, Mapping) else None
    sparse_enabled = bool(str(manifest.get("sparse_index_name") or "").strip())
    selected = _is_selected_vector_manifest(manifest)
    namespace_keys = set(_vector_namespace_keys(manifest))
    for upload_key, stat_key in _VECTOR_UPLOAD_COUNT_MAP.items():
        if upload_key not in namespace_keys:
            continue
        bundle_count = _count(bundle_stats.get(stat_key))
        dense_enabled = True
        if selected and upload_key not in {"chunks", "parents", "media", "page_cards", "actions"}:
            dense_enabled = False
        elif embedder_config is not None:
            if upload_key == "facts":
                dense_enabled = bool(embedder_config.get("enable_dense_facts", False))
            elif upload_key == "evidence_spans":
                dense_enabled = bool(embedder_config.get("enable_dense_evidence_spans", True))
            elif upload_key == "assertions":
                dense_enabled = bool(embedder_config.get("enable_dense_assertions", True))
            elif upload_key == "summaries":
                dense_enabled = bool(embedder_config.get("enable_dense_summaries", True))
        expected[upload_key] = bundle_count if dense_enabled else 0
        if sparse_enabled:
            skipped = _sparse_record_skipped_count(manifest, upload_key)
            sparse_lane_enabled = not (
                upload_key == "evidence_spans"
                and embedder_config is not None
                and not bool(embedder_config.get("enable_sparse_evidence_spans", True))
            )
            expected[f"sparse_{upload_key}"] = max(0, bundle_count - skipped) if sparse_lane_enabled else 0
    return expected


def _is_legacy_vector_manifest(manifest: Dict[str, Any]) -> bool:
    return (
        str(manifest.get("vectorstore_contract") or "").strip() == _LEGACY_VECTORSTORE_CONTRACT
        or "summary_vectors_uploaded" in manifest
        or "text_vectors_uploaded" in manifest
    )


def _legacy_uploaded_counts(manifest: Dict[str, Any]) -> Dict[str, int]:
    return {
        "summary": _count(manifest.get("summary_vectors_uploaded")),
        "text": _count(manifest.get("text_vectors_uploaded")),
    }


def _vector_uploaded_counts(manifest: Dict[str, Any]) -> Dict[str, int]:
    if _is_legacy_vector_manifest(manifest):
        return _legacy_uploaded_counts(manifest)
    uploaded = manifest.get("uploaded") if isinstance(manifest.get("uploaded"), dict) else {}
    return {str(key): _count(value) for key, value in uploaded.items()}


def _vector_expected_counts_for_manifest(
    manifest: Dict[str, Any],
    bundle_stats: Dict[str, Any],
    work_dir: Path,
    embedder_config: Mapping[str, Any] | None = None,
) -> Dict[str, int]:
    if not _is_legacy_vector_manifest(manifest):
        return _modern_vector_expected_counts_for_manifest(manifest, bundle_stats, embedder_config)
    formatted_manifest = load_json_safe(
        work_dir / "stage_outputs" / "format_legacy_vectorstores" / "legacy_vectorstore_manifest.json",
        {},
    )
    if isinstance(formatted_manifest, dict):
        return {
            "summary": _count(formatted_manifest.get("summary_records")),
            "text": _count(formatted_manifest.get("text_records")),
        }
    return {}


def _verify_legacy_vector_manifest(manifest: Dict[str, Any], work_dir: Path) -> List[str]:
    errors: List[str] = []
    if str(manifest.get("vectorstore_contract") or "").strip() != _LEGACY_VECTORSTORE_CONTRACT:
        errors.append("Legacy vectorstore manifest is missing the expected contract")
    uploaded = _legacy_uploaded_counts(manifest)
    for key in ("summary", "text"):
        if uploaded[key] <= 0:
            errors.append(f"Legacy vectorstore uploaded count for {key!r} must be greater than zero")

    formatted_manifest = load_json_safe(
        work_dir / "stage_outputs" / "format_legacy_vectorstores" / "legacy_vectorstore_manifest.json",
        {},
    )
    if isinstance(formatted_manifest, dict):
        expected_summary = _count(formatted_manifest.get("summary_records"))
        expected_text = _count(formatted_manifest.get("text_records"))
        if expected_summary and uploaded["summary"] != expected_summary:
            errors.append(
                "Legacy vectorstore summary count mismatch: "
                f"expected {expected_summary}, got {uploaded['summary']}"
            )
        if expected_text and uploaded["text"] != expected_text:
            errors.append(
                "Legacy vectorstore text count mismatch: "
                f"expected {expected_text}, got {uploaded['text']}"
            )

    progress = load_json_safe(
        work_dir / "stage_outputs" / "upload_legacy_vectorstores" / "legacy_upload_progress.json",
        {},
    )
    if isinstance(progress, dict):
        if str(progress.get("phase") or "").strip().lower() != "complete":
            errors.append("Legacy vectorstore upload progress is not complete")
        totals = progress.get("totals") if isinstance(progress.get("totals"), dict) else {}
        progress_uploaded = progress.get("uploaded") if isinstance(progress.get("uploaded"), dict) else {}
        for key in ("summary", "text"):
            if _count(totals.get(key)) and _count(progress_uploaded.get(key)) != _count(totals.get(key)):
                errors.append(
                    f"Legacy vectorstore upload progress mismatch for {key!r}: "
                    f"uploaded {_count(progress_uploaded.get(key))}, total {_count(totals.get(key))}"
                )

    if bool(manifest.get("use_sparse_embeddings")):
        bm25_file = str(manifest.get("bm25_model_file") or manifest.get("legacy_bm25_model_file") or "").strip()
        if not bm25_file:
            errors.append("Legacy vectorstore manifest has sparse embeddings enabled but no BM25 model file")
        elif not Path(bm25_file).expanduser().exists():
            errors.append(f"Legacy vectorstore BM25 model file is missing: {bm25_file}")
    return errors


def _verify_vector_manifest(
    manifest: Dict[str, Any],
    bundle_stats: Dict[str, Any],
    work_dir: Path,
    embedder_config: Mapping[str, Any] | None = None,
) -> List[str]:
    if _is_legacy_vector_manifest(manifest):
        return _verify_legacy_vector_manifest(manifest, work_dir)

    errors: List[str] = []
    provider = str(manifest.get("provider") or "pinecone").strip().lower()
    if provider not in {"pinecone", "pgvector"}:
        errors.append(f"Vector index manifest provider is unsupported: {provider}")
    if _count(manifest.get("schema_version")) < 2:
        errors.append("Vector index manifest schema_version must be 2 or newer")
    if provider == "pgvector" and _count(manifest.get("schema_version")) < 5:
        errors.append("pgvector manifest schema_version must be 5 or newer")
    selected = _is_selected_vector_manifest(manifest)
    if selected and _count(manifest.get("schema_version")) < 6:
        errors.append("selected-profile vector manifest schema_version must be 6 or newer")
    if not str(manifest.get("index_name") or "").strip():
        errors.append("Vector index manifest is missing index_name")

    namespace_strategy = str(manifest.get("namespace_strategy") or "").strip().lower()
    if namespace_strategy not in {"static", "release"}:
        errors.append("Vector index manifest is missing a valid namespace_strategy")
    if namespace_strategy == "release" and not str(manifest.get("namespace_release_id") or "").strip():
        errors.append("Release-scoped vector index manifest is missing namespace_release_id")

    namespaces = manifest.get("namespaces")
    if not isinstance(namespaces, dict):
        namespaces = {}
        errors.append("Vector index manifest is missing namespace mappings")
    namespace_keys = _vector_namespace_keys(manifest)
    missing_namespaces = [
        key for key in namespace_keys if not str(namespaces.get(key) or "").strip()
    ]
    if missing_namespaces:
        errors.append(f"Vector index manifest is missing namespaces: {missing_namespaces}")
    populated_namespaces = [str(namespaces.get(key) or "").strip() for key in namespace_keys]
    populated_namespaces = [namespace for namespace in populated_namespaces if namespace]
    if len(set(populated_namespaces)) != len(populated_namespaces):
        errors.append("Vector index manifest namespaces must be distinct per retrieval lane")

    uploaded = manifest.get("uploaded")
    if not isinstance(uploaded, dict):
        uploaded = {}
        errors.append("Vector index manifest is missing uploaded counts")
    expected_counts = _modern_vector_expected_counts_for_manifest(manifest, bundle_stats, embedder_config)
    missing_uploaded = [key for key in expected_counts if key not in uploaded]
    if missing_uploaded:
        errors.append(f"Vector index manifest is missing uploaded count keys: {missing_uploaded}")
    positive_lanes = (
        ("chunks", "parents", "media", "page_cards", "actions")
        if selected
        else ("chunks", "parents")
    )
    for key in positive_lanes:
        if _count(uploaded.get(key)) <= 0:
            errors.append(f"Vector index uploaded count for {key!r} must be greater than zero")
    for key, expected in expected_counts.items():
        actual = _count(uploaded.get(key))
        if actual != expected:
            errors.append(
                f"Vector index uploaded count mismatch for {key!r}: "
                f"expected {expected}, got {actual}"
            )

    planned = manifest.get("planned")
    if not isinstance(planned, dict):
        planned = {}
        errors.append("Vector index manifest is missing immutable planned counts")
    required_plan_keys = list(namespace_keys)
    if str(manifest.get("sparse_index_name") or "").strip():
        required_plan_keys.extend(f"sparse_{key}" for key in namespace_keys)
    missing_plan_keys = [key for key in required_plan_keys if key not in planned]
    if missing_plan_keys:
        errors.append(f"Vector index manifest is missing planned count keys: {missing_plan_keys}")
    missing_uploaded_lane_keys = [key for key in required_plan_keys if key not in uploaded]
    if missing_uploaded_lane_keys:
        errors.append(
            f"Vector index manifest is missing uploaded lane count keys: {missing_uploaded_lane_keys}"
        )
    for key in required_plan_keys:
        planned_count = _count(planned.get(key))
        uploaded_count = _count(uploaded.get(key))
        if planned_count != uploaded_count:
            errors.append(
                f"Vector upload plan mismatch for {key!r}: planned {planned_count}, uploaded {uploaded_count}"
            )
    for key, canonical_count in expected_counts.items():
        if key in planned and _count(planned.get(key)) != canonical_count:
            errors.append(
                f"Vector upload plan does not match canonical count for {key!r}: "
                f"expected {canonical_count}, planned {_count(planned.get(key))}"
            )

    sparse_payload = manifest.get("sparse") if isinstance(manifest.get("sparse"), Mapping) else {}
    sparse_stats = sparse_payload.get("record_stats") if isinstance(sparse_payload.get("record_stats"), Mapping) else {}
    for upload_key, stat_key in _VECTOR_UPLOAD_COUNT_MAP.items():
        if upload_key not in namespace_keys:
            continue
        singular = {
            "chunks": "chunk",
            "parents": "parent",
            "media": "media",
            "page_cards": "page_card",
            "actions": "action",
            "facts": "fact",
            "evidence_spans": "evidence_span",
            "summaries": "summary",
            "assertions": "assertion",
        }[upload_key]
        skip_key = f"{singular}_records_skipped"
        raw_skipped = sparse_stats.get(skip_key, 0)
        try:
            skipped = int(raw_skipped or 0)
        except (TypeError, ValueError):
            errors.append(f"Sparse skip count {skip_key!r} is not an integer")
            continue
        canonical_count = _count(bundle_stats.get(stat_key))
        if skipped < 0 or skipped > canonical_count:
            errors.append(
                f"Sparse skip count {skip_key!r} must be between 0 and {canonical_count}, got {skipped}"
            )

    verification = manifest.get("verification")
    if not isinstance(verification, dict) or not verification:
        errors.append("Vector index manifest is missing namespace verification reports")
        verification = {}
    labels = ["dense"]
    if str(manifest.get("sparse_index_name") or "").strip():
        labels.append("sparse")
    for label in labels:
        report = verification.get(label)
        if not isinstance(report, dict):
            errors.append(f"Vector index manifest is missing {label} namespace verification")
            continue
        expected_report = report.get("expected")
        actual_report = report.get("actual")
        if not isinstance(expected_report, dict) or not isinstance(actual_report, dict):
            errors.append(f"Vector {label} namespace verification must include expected and actual counts")
            continue
        if "failures" not in report or not isinstance(report.get("failures"), list):
            errors.append(f"Vector {label} namespace verification must include a failures list")
        elif report.get("failures"):
            errors.append(f"Vector {label} namespace verification has failures: {report.get('failures')}")

        prefix = "sparse_" if label == "sparse" else ""
        for upload_key in namespace_keys:
            count_key = f"{prefix}{upload_key}"
            expected_count = expected_counts.get(count_key, _count(planned.get(count_key)))
            if expected_count <= 0:
                continue
            namespace = str(namespaces.get(upload_key) or "").strip()
            if namespace not in expected_report or namespace not in actual_report:
                errors.append(
                    f"Vector {label} verification is missing counts for {upload_key!r} namespace {namespace!r}"
                )
                continue
            reported_expected = _count(expected_report.get(namespace))
            reported_actual = _count(actual_report.get(namespace))
            if reported_expected != expected_count or reported_actual != expected_count:
                errors.append(
                    f"Vector {label} verification count mismatch for {upload_key!r}: "
                    f"canonical={expected_count}, expected={reported_expected}, actual={reported_actual}"
                )
    if selected:
        for lane in (
            "facts",
            "evidence_spans",
            "summaries",
            "assertions",
            "entities",
            "communities",
        ):
            if _count(uploaded.get(lane)) != 0 or _count(planned.get(lane)) != 0:
                errors.append(f"Selected release contains unevaluated dense lane: {lane}")
    return errors


def _verify_vector_manifest_matches_config(
    manifest: Dict[str, Any],
    config: Dict[str, Any],
    *,
    expected_run_id: str = "",
) -> List[str]:
    """Ensure release evaluation uses the same vector targets that were uploaded."""
    errors: List[str] = []
    embedder_cfg = config.get("embedder") if isinstance(config.get("embedder"), dict) else {}
    retrieval_cfg = config.get("retrieval") if isinstance(config.get("retrieval"), dict) else {}
    vector_store_cfg = config.get("vector_store") if isinstance(config.get("vector_store"), dict) else {}
    if _is_legacy_vector_manifest(manifest):
        expected_summary = str(embedder_cfg.get("pinecone_summary_index") or "").strip()
        expected_text = str(embedder_cfg.get("pinecone_text_index") or embedder_cfg.get("pinecone_index") or "").strip()
        actual_summary = str(manifest.get("summary_index_name") or "").strip()
        actual_text = str(manifest.get("text_index_name") or "").strip()
        if expected_summary and actual_summary and expected_summary != actual_summary:
            errors.append(
                f"Configured summary Pinecone index {expected_summary!r} does not match uploaded index {actual_summary!r}"
            )
        if expected_text and actual_text and expected_text != actual_text:
            errors.append(
                f"Configured text Pinecone index {expected_text!r} does not match uploaded index {actual_text!r}"
            )
        expected_model = str(embedder_cfg.get("model") or "").strip()
        actual_model = str(manifest.get("model") or "").strip()
        if expected_model and actual_model and expected_model != actual_model:
            errors.append(f"Configured embedding model {expected_model!r} does not match uploaded model {actual_model!r}")
        expected_dim = _count(embedder_cfg.get("output_dimensionality"))
        actual_dim = _count(manifest.get("output_dimensionality") or manifest.get("dimension"))
        if expected_dim and actual_dim and expected_dim != actual_dim:
            errors.append(
                f"Configured embedding dimension {expected_dim} does not match uploaded dimension {actual_dim}"
            )
        expected_namespace = str(embedder_cfg.get("namespace") or "").strip()
        actual_namespace = str(manifest.get("namespace") or "").strip()
        if expected_namespace != actual_namespace:
            errors.append(
                f"Configured legacy Pinecone namespace {expected_namespace!r} does not match uploaded namespace {actual_namespace!r}"
            )
        return errors

    expected_provider = str(vector_store_cfg.get("provider") or "pinecone").strip().lower()
    actual_provider = str(manifest.get("provider") or "pinecone").strip().lower()
    if expected_provider != actual_provider:
        errors.append(
            f"Configured vector provider {expected_provider!r} does not match uploaded provider {actual_provider!r}"
        )
    if actual_provider == "pgvector":
        expected_contract = production_indexing_contract_fingerprint(config)
        actual_contract = str(
            manifest.get("production_indexing_contract_fingerprint") or ""
        ).strip()
        if actual_contract != expected_contract:
            errors.append("pgvector upload manifest indexing contract does not match configured production contract")
    if expected_provider == "pgvector":
        expected_dense = (
            f"{str(vector_store_cfg.get('schema') or 'mbzuai_retrieval')}."
            f"{str(vector_store_cfg.get('records_table') or 'embedding_records')}"
        )
        expected_sparse = ""
    else:
        expected_dense = str(embedder_cfg.get("pinecone_index") or "").strip()
        expected_sparse = str(embedder_cfg.get("pinecone_sparse_index") or retrieval_cfg.get("pinecone_sparse_index") or "").strip()
    actual_dense = str(manifest.get("index_name") or "").strip()
    actual_sparse = str(manifest.get("sparse_index_name") or "").strip()
    if expected_dense and expected_dense != actual_dense:
        errors.append(f"Configured vector index {expected_dense!r} does not match uploaded index {actual_dense!r}")
    if expected_sparse and expected_sparse != actual_sparse:
        errors.append(f"Configured sparse Pinecone index {expected_sparse!r} does not match uploaded sparse index {actual_sparse!r}")

    expected_namespace_strategy = str(embedder_cfg.get("namespace_strategy") or "static").strip().lower()
    actual_namespace_strategy = str(manifest.get("namespace_strategy") or "static").strip().lower()
    expected_resolved_namespaces: Dict[str, str] = {}
    if expected_namespace_strategy == "release":
        if actual_namespace_strategy != "release":
            errors.append("Production upload manifest is not release-scoped")
        namespace_release_id = str(manifest.get("namespace_release_id") or "").strip()
        if not namespace_release_id:
            errors.append("Release-scoped upload manifest is missing namespace_release_id")
        else:
            if expected_run_id and namespace_release_id != str(expected_run_id).strip():
                errors.append(
                    f"Release-scoped upload manifest belongs to run {namespace_release_id!r}, "
                    f"not {str(expected_run_id).strip()!r}"
                )
            from pipeline.stages.embedders.gemini_pinecone_embedder import _resolve_upload_namespaces

            try:
                expected_resolved_namespaces = _resolve_upload_namespaces(
                    embedder_cfg,
                    run_id=namespace_release_id,
                )
            except (KeyError, ValueError) as exc:
                errors.append(f"Could not resolve release-scoped namespaces from config: {exc}")

    manifest_namespaces = manifest.get("namespaces") if isinstance(manifest.get("namespaces"), dict) else {}
    namespace_checks = {
        "chunks": ("namespace_chunks", "namespace_chunks"),
        "parents": ("namespace_parents", "namespace_parents"),
        "media": ("namespace_media", "namespace_media"),
        "page_cards": ("namespace_page_cards", "namespace_page_cards"),
        "actions": ("namespace_actions", "namespace_actions"),
        "facts": ("namespace_facts", "namespace_facts"),
        "evidence_spans": ("namespace_evidence_spans", "namespace_evidence_spans"),
        "summaries": ("namespace_summaries", "namespace_summaries"),
        "assertions": ("namespace_assertions", "namespace_assertions"),
        "entities": ("namespace_entities", "namespace_entities"),
        "communities": ("namespace_communities", "namespace_communities"),
    }
    for manifest_key, (embedder_key, retrieval_key) in namespace_checks.items():
        expected = str(
            expected_resolved_namespaces.get(manifest_key)
            or embedder_cfg.get(embedder_key)
            or retrieval_cfg.get(retrieval_key)
            or ""
        ).strip()
        actual = str(manifest_namespaces.get(manifest_key) or "").strip()
        if expected and expected != actual:
            errors.append(
                f"Configured namespace {embedder_key}={expected!r} does not match uploaded {manifest_key} namespace {actual!r}"
            )
    return errors


def _verify_neo4j_manifest(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    node_count = int(manifest.get("node_count") or 0)
    if node_count <= 0:
        errors.append("Neo4j manifest node_count must be greater than zero")
    verification = manifest.get("verification") or {}
    if verification:
        expected_nodes = int(verification.get("expected_nodes") or 0)
        actual_nodes = int(verification.get("actual_nodes") or 0)
        expected_edges = int(verification.get("expected_edges") or 0)
        actual_edges = int(verification.get("actual_edges") or 0)
        if expected_nodes != actual_nodes or expected_edges != actual_edges:
            errors.append(
                "Neo4j verification mismatch: "
                f"expected nodes={expected_nodes} edges={expected_edges}, "
                f"actual nodes={actual_nodes} edges={actual_edges}"
            )
    return errors


def _load_local_graph_manifest(
    work_dir: Path,
    config: Mapping[str, Any] | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    pipeline_cfg = (
        config.get("pipeline")
        if isinstance(config, Mapping) and isinstance(config.get("pipeline"), Mapping)
        else {}
    )
    production = bool(pipeline_cfg.get("production_profile", False))
    try:
        selected = resolve_canonical_graph_artifacts(
            work_dir,
            required=True,
            require_index=True,
            validate_binding=production,
        )
    except GraphArtifactContractError as exc:
        return {}, [str(exc)]
    assert selected is not None
    selected_path = selected.graph_file
    payload: Any = load_json_safe(selected_path, None)
    if not isinstance(payload, dict):
        return {}, [f"Local graph artifact is invalid JSON object: {selected_path}"]

    nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else []
    edges = payload.get("edges") if isinstance(payload.get("edges"), list) else []
    node_count = _count(payload.get("node_count")) or len(nodes)
    edge_count = _count(payload.get("edge_count")) or len(edges)
    assertion_count = sum(
        1
        for node in nodes
        if isinstance(node, dict) and str(node.get("node_type") or node.get("type") or "").strip() == "relation_assertion"
    )
    errors: List[str] = []
    graph_issues = validate_graph_bundle(payload)
    errors.extend(
        f"Local graph bundle is invalid: {issue.get('message') or issue.get('code')}"
        for issue in graph_issues
    )
    if node_count <= 0:
        errors.append("Local graph node_count must be greater than zero")
    if edge_count <= 0:
        errors.append("Local graph edge_count must be greater than zero")
    if production and selected.index_file is not None:
        derivation_issues = validate_graph_index_derivation(
            selected.graph_file,
            selected.index_file,
        )
        errors.extend(
            "Local graph index derivation is invalid: "
            + str(issue.get("message") or issue.get("code"))
            for issue in derivation_issues[:1]
        )
    if production:
        graph_cfg = config.get("graph") if isinstance(config.get("graph"), Mapping) else {}
        minimum_summary_characters = int(
            graph_cfg.get("community_summary_min_characters", 40) or 40
        )
        minimum_summary_coverage = float(
            graph_cfg.get("community_summary_min_coverage_ratio", 1.0)
        )
        summary_quality = community_summary_quality(
            payload,
            min_characters=minimum_summary_characters,
        )
        if int(summary_quality["total_communities"]) <= 0:
            errors.append("Local production graph must contain at least one community")
        elif float(summary_quality["coverage_ratio"]) < minimum_summary_coverage:
            errors.append(
                "Local production graph community-summary coverage is below threshold: "
                f"actual={float(summary_quality['coverage_ratio']):.4f}, "
                f"required={minimum_summary_coverage:.4f}"
            )

    return {
        "manifest_file": str(selected_path),
        "index_file": str(selected.index_file or ""),
        "store_backend": "local_json",
        "graph_type": selected.kind,
        "knowledge_graph_sha256": selected.graph_sha256,
        "knowledge_graph_index_sha256": selected.index_sha256,
        "node_count": node_count,
        "edge_count": edge_count,
        "assertion_count": assertion_count,
        "verification": {
            "expected_nodes": node_count,
            "actual_nodes": node_count,
            "expected_edges": edge_count,
            "actual_edges": edge_count,
        },
    }, errors


def _load_graph_manifest_for_backend(
    *,
    work_dir: Path,
    config: Mapping[str, Any],
) -> Tuple[Dict[str, Any], List[str]]:
    backend = _graph_store_backend(config)
    if backend == "disabled":
        return {
            "manifest_file": "",
            "store_backend": "disabled",
            "graph_type": "",
            "node_count": 0,
            "edge_count": 0,
            "verification": {},
        }, []
    if backend == "neo4j":
        neo4j_manifest_path = work_dir / "stage_outputs" / "upload_graph" / "neo4j_upload_manifest.json"
        neo4j_manifest, graph_errors = _load_required_json(neo4j_manifest_path, "Neo4j upload manifest")
        graph_errors.extend(_verify_neo4j_manifest(neo4j_manifest))
        return {
            "manifest_file": str(neo4j_manifest_path),
            "store_backend": "neo4j",
            "neo4j_namespace": neo4j_manifest.get("neo4j_namespace", ""),
            "neo4j_database": neo4j_manifest.get("neo4j_database", ""),
            "graph_type": neo4j_manifest.get("graph_type", ""),
            "knowledge_graph_file": neo4j_manifest.get("knowledge_graph_file", ""),
            "knowledge_graph_sha256": neo4j_manifest.get("knowledge_graph_sha256", ""),
            "knowledge_graph_index_file": neo4j_manifest.get("knowledge_graph_index_file", ""),
            "knowledge_graph_index_sha256": neo4j_manifest.get("knowledge_graph_index_sha256", ""),
            "node_count": neo4j_manifest.get("node_count", 0),
            "edge_count": neo4j_manifest.get("edge_count", 0),
            "verification": neo4j_manifest.get("verification") or {},
        }, graph_errors
    return _load_local_graph_manifest(work_dir, config)


def _load_retrieval_bundle_stats(work_dir: Path) -> Dict[str, Any]:
    bundle_path = work_dir / "stage_outputs" / "finalize_retrieval_bundle" / "retrieval_bundle.json"
    if not bundle_path.exists():
        bundle_path = work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    if not bundle_path.exists():
        bundle_path = work_dir / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json"
    payload = load_json_safe(bundle_path, {}) or {}
    if isinstance(payload, dict):
        stats = payload.get("stats") if isinstance(payload.get("stats"), dict) else {}
        selected_contract = (
            payload.get("selected_release_contract")
            if isinstance(payload.get("selected_release_contract"), Mapping)
            else {}
        )
        lexical_corpus_path = bundle_path.with_name("lexical_corpus.json")
        promoted_assertions_path = (
            work_dir / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
        )
        return {
            **dict(stats or {}),
            "retrieval_bundle_version": payload.get("version", 0),
            "retrieval_bundle_file": str(bundle_path) if bundle_path.exists() else "",
            "retrieval_bundle_sha256": sha256_file(bundle_path) if bundle_path.exists() else "",
            "lexical_corpus_file": str(lexical_corpus_path) if lexical_corpus_path.exists() else "",
            "lexical_corpus_sha256": (
                sha256_file(lexical_corpus_path) if lexical_corpus_path.exists() else ""
            ),
            "promoted_assertions_file": (
                str(promoted_assertions_path) if promoted_assertions_path.exists() else ""
            ),
            "promoted_assertions_sha256": (
                sha256_file(promoted_assertions_path) if promoted_assertions_path.exists() else ""
            ),
            "selected_release_assembly_sha256": str(
                selected_contract.get("manifest_sha256") or ""
            ).strip().lower(),
            "selected_release_binding_sha256": str(
                selected_contract.get("assembly_sha256") or ""
            ).strip().lower(),
            "page_graph_navigation_catalog_sha256": str(
                selected_contract.get("navigation_catalog_sha256") or ""
            ).strip().lower(),
        }
    return {}


def _verify_upload_artifact_binding(
    *,
    vector_manifest: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
    retrieval_bundle_stats: Mapping[str, Any],
    require_current_contract: bool,
) -> List[str]:
    """Verify vectors, retrieval corpus, and GraphRAG share one release input."""

    if _is_legacy_vector_manifest(dict(vector_manifest)):
        return []

    errors: List[str] = []
    schema_version = _count(vector_manifest.get("schema_version"))
    if require_current_contract and schema_version < 4:
        errors.append("Canonical production vector manifest schema_version must be 4 or newer")

    expected_bundle_sha = str(vector_manifest.get("retrieval_bundle_sha256") or "").strip()
    actual_bundle_sha = str(retrieval_bundle_stats.get("retrieval_bundle_sha256") or "").strip()
    if require_current_contract and not expected_bundle_sha:
        errors.append("Vector manifest is missing retrieval_bundle_sha256")
    elif expected_bundle_sha and expected_bundle_sha != actual_bundle_sha:
        errors.append(
            "Uploaded retrieval bundle hash does not match the release bundle: "
            f"uploaded={expected_bundle_sha}, release={actual_bundle_sha}"
        )

    expected_lexical_sha = str(vector_manifest.get("lexical_corpus_sha256") or "").strip()
    actual_lexical_sha = str(retrieval_bundle_stats.get("lexical_corpus_sha256") or "").strip()
    if require_current_contract and not expected_lexical_sha:
        errors.append("Vector manifest is missing lexical_corpus_sha256")
    elif expected_lexical_sha and expected_lexical_sha != actual_lexical_sha:
        errors.append("Uploaded lexical corpus hash does not match the release sidecar")

    expected_promoted_assertions_sha = str(
        vector_manifest.get("promoted_assertions_sha256") or ""
    ).strip()
    actual_promoted_assertions_sha = str(
        retrieval_bundle_stats.get("promoted_assertions_sha256") or ""
    ).strip()
    if require_current_contract and not expected_promoted_assertions_sha:
        errors.append("Vector manifest is missing promoted_assertions_sha256")
    elif (
        expected_promoted_assertions_sha
        and expected_promoted_assertions_sha != actual_promoted_assertions_sha
    ):
        errors.append("Uploaded promoted assertions hash does not match the release sidecar")

    selected = _is_selected_vector_manifest(vector_manifest)
    selected_hashes: List[str] = []
    if selected:
        for key, label in (
            ("selected_release_assembly_sha256", "selected release assembly"),
            ("selected_release_binding_sha256", "selected release binding"),
            ("page_graph_navigation_catalog_sha256", "Page Graph navigation catalog"),
        ):
            expected = str(vector_manifest.get(key) or "").strip().lower()
            actual = str(retrieval_bundle_stats.get(key) or "").strip().lower()
            if not expected:
                errors.append(f"Vector manifest is missing {key}")
            elif expected != actual:
                errors.append(f"Uploaded {label} hash does not match the retrieval bundle contract")
            selected_hashes.append(expected)

    graph_backend = str(graph_manifest.get("store_backend") or "").strip()
    if graph_backend == "disabled":
        return errors

    expected_graph_sha = str(vector_manifest.get("knowledge_graph_sha256") or "").strip()
    actual_graph_sha = str(graph_manifest.get("knowledge_graph_sha256") or "").strip()
    if require_current_contract and not expected_graph_sha:
        errors.append("Vector manifest is missing knowledge_graph_sha256")
    elif expected_graph_sha and expected_graph_sha != actual_graph_sha:
        errors.append(
            "Uploaded vector graph hash does not match the runtime graph: "
            f"uploaded={expected_graph_sha}, runtime={actual_graph_sha}"
        )

    expected_graph_kind = str(vector_manifest.get("knowledge_graph_kind") or "").strip()
    actual_graph_kind = str(graph_manifest.get("graph_type") or "").strip()
    if require_current_contract and not expected_graph_kind:
        errors.append("Vector manifest is missing knowledge_graph_kind")
    elif (
        graph_backend == "local_json"
        and expected_graph_kind
        and actual_graph_kind
        and expected_graph_kind != actual_graph_kind
    ):
        errors.append(
            "Uploaded vector graph kind does not match the runtime graph: "
            f"uploaded={expected_graph_kind}, runtime={actual_graph_kind}"
        )

    expected_index_sha = str(vector_manifest.get("knowledge_graph_index_sha256") or "").strip()
    actual_index_sha = str(graph_manifest.get("knowledge_graph_index_sha256") or "").strip()
    if require_current_contract and graph_backend == "local_json" and not expected_index_sha:
        errors.append("Vector manifest is missing knowledge_graph_index_sha256")
    elif expected_index_sha and expected_index_sha != actual_index_sha:
        errors.append(
            "Uploaded vector graph-index hash does not match the runtime graph index: "
            f"uploaded={expected_index_sha}, runtime={actual_index_sha}"
        )

    expected_upload_input_sha = str(vector_manifest.get("upload_input_sha256") or "").strip()
    if expected_bundle_sha and expected_graph_sha:
        upload_digests = [
            expected_bundle_sha,
            expected_lexical_sha,
            expected_promoted_assertions_sha,
            expected_graph_sha,
            expected_index_sha,
        ]
        if selected:
            upload_digests.extend(selected_hashes)
        actual_upload_input_sha = combine_sha256_digests(*upload_digests)
        if require_current_contract and not expected_upload_input_sha:
            errors.append("Vector manifest is missing upload_input_sha256")
        elif expected_upload_input_sha and expected_upload_input_sha != actual_upload_input_sha:
            errors.append(
                "Vector manifest upload_input_sha256 does not bind its retrieval bundle and graph hashes"
            )
    return errors


def _find_vector_manifest_path(work_dir: Path) -> Path:
    candidates = (
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        work_dir / "stage_outputs" / "upload_legacy_vectorstores" / "legacy_pinecone_upload_manifest.json",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_release_manifest(
    *,
    config_name: str,
    work_dir: str | Path,
    dataset_path: str | Path | None = None,
    gates_path: str | Path | None = None,
    answer_dataset_path: str | Path | None = None,
    answer_gates_path: str | Path | None = None,
    answer_eval_mode: str = "websocket",
    answer_endpoint: str | None = None,
    answer_runtime_commit_sha: str | None = None,
    answer_auth_token: str | None = None,
    answer_widget_key: str | None = None,
    answer_model: str = "gemini-2.5-flash",
    answer_timeout_seconds: float = 120.0,
    answer_probe_mode: bool = False,
    answer_eval_request_mode: bool = True,
    answer_resume_predictions: bool = False,
    judge_enabled: bool = True,
    judge_model: str = "gemini-2.5-flash",
    judge_timeout_seconds: float = 120.0,
    skip_answer_readiness: bool = False,
    allow_answer_readiness_waiver: bool = False,
    answer_readiness_waiver_reason: str = "",
    query_cache_path: str | Path | None = None,
    retrieval_cache_path: str | Path | None = None,
    parallelism: int = 1,
    skip_stage_validation: bool = False,
    progress_callback: ReleaseProgressCallback | None = None,
) -> Tuple[Dict[str, Any], bool]:
    work_path = Path(work_dir).expanduser().resolve()
    _emit_progress(
        progress_callback,
        "release_check_start",
        work_dir=str(work_path),
        config_name=str(config_name),
    )
    config = load_effective_config(config_name, work_dir=work_path)
    retrieval_contract_config = apply_vector_upload_manifest_config(config, work_path)
    pipeline_config = config.get("pipeline") if isinstance(config.get("pipeline"), Mapping) else {}
    canonical_production = bool(pipeline_config.get("production_profile", False)) or Path(
        str(config_name or "")
    ).stem == "mbzuai_production"
    normalized_answer_runtime_commit = str(answer_runtime_commit_sha or "").strip().lower()
    resolved_snapshot = load_json_safe(work_path / "resolved_config.json", {}) or {}
    indexing_build = (
        dict(resolved_snapshot.get("indexing_build") or {})
        if isinstance(resolved_snapshot, Mapping)
        else {}
    )
    indexing_build_errors: List[str] = []
    if canonical_production:
        indexing_commit = str(indexing_build.get("commit_sha") or "").strip().lower()
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", indexing_commit):
            indexing_build_errors.append(
                "Canonical production indexing_build.commit_sha must be a full Git commit"
            )
        if indexing_build.get("dirty") is not False:
            indexing_build_errors.append(
                "Canonical production artifacts cannot be built from a dirty source tree"
            )
        if indexing_build.get("implementation_sha256") != indexing_implementation_hashes():
            indexing_build_errors.append(
                "Canonical production indexing implementation hashes do not match the release code"
            )

    validation_errors: Dict[str, List[str]] | None = None
    if not skip_stage_validation:
        _emit_progress(progress_callback, "release_check_stage_validation_start")
        validation_config = _validation_config_for_release(retrieval_contract_config, work_path)
        orchestrator = PipelineOrchestrator(validation_config)
        try:
            validation_errors = asyncio.run(orchestrator.validate())
        except (KeyError, ValueError) as exc:
            validation_errors = {"stage_registration": [str(exc)]}
        _emit_progress(
            progress_callback,
            "release_check_stage_validation_done",
            error_count=sum(len(errors) for errors in validation_errors.values()) if isinstance(validation_errors, dict) else 0,
        )

    _emit_progress(progress_callback, "release_check_preflight_start")
    preflight = assess_production_readiness(
        retrieval_contract_config,
        config_name=config_name,
        validation_errors=validation_errors,
        purpose="release",
    )
    _emit_progress(
        progress_callback,
        "release_check_preflight_done",
        ok=bool(preflight.get("ok")),
        error_count=int(preflight.get("error_count") or 0),
        warning_count=int(preflight.get("warning_count") or 0),
    )
    _emit_progress(progress_callback, "release_check_audit_start")
    audit = audit_run(work_path)
    _emit_progress(
        progress_callback,
        "release_check_audit_done",
        ok=bool(audit.ok),
        error_count=len(audit.errors),
        warning_count=len(audit.warnings),
    )

    vector_manifest_path = _find_vector_manifest_path(work_path)
    vector_manifest, vector_errors = _load_required_json(vector_manifest_path, "Vector index upload manifest")
    if canonical_production:
        if vector_manifest.get("indexing_build") != indexing_build:
            vector_errors.append(
                "Vector upload manifest indexing_build does not match resolved_config.json"
            )
        expected_indexing_build_sha = _stable_json_sha256(indexing_build) if indexing_build else ""
        if vector_manifest.get("indexing_build_sha256") != expected_indexing_build_sha:
            vector_errors.append(
                "Vector upload manifest indexing_build_sha256 does not match resolved_config.json"
            )
    graph_manifest, graph_errors = _load_graph_manifest_for_backend(
        work_dir=work_path,
        config=retrieval_contract_config,
    )
    retrieval_bundle_stats = _load_retrieval_bundle_stats(work_path)
    selected_profile = (
        config.get("selected_profile")
        if isinstance(config.get("selected_profile"), Mapping)
        else {}
    )
    bundle_errors = _verify_retrieval_bundle_stats(
        retrieval_bundle_stats,
        selected_profile=bool(str(selected_profile.get("variant_id") or "").strip()),
    )
    embedder_config = config.get("embedder") if isinstance(config.get("embedder"), Mapping) else {}
    vector_errors.extend(
        _verify_vector_manifest(
            vector_manifest,
            retrieval_bundle_stats,
            work_path,
            embedder_config,
        )
    )
    vector_errors.extend(
        _verify_upload_artifact_binding(
            vector_manifest=vector_manifest,
            graph_manifest=graph_manifest,
            retrieval_bundle_stats=retrieval_bundle_stats,
            require_current_contract=canonical_production,
        )
    )
    resolved_config_exists = (work_path / "resolved_config.json").exists()
    if resolved_config_exists:
        # Validate upload identity against the immutable run snapshot, not the
        # manifest-overlaid retrieval config (whose namespaces are already
        # resolved and would otherwise be release-suffixed a second time).
        vector_errors.extend(
            _verify_vector_manifest_matches_config(
                vector_manifest,
                config,
                expected_run_id=work_path.name,
            )
        )
    elif canonical_production and not _is_legacy_vector_manifest(vector_manifest):
        vector_errors.append(
            "Canonical production release is missing resolved_config.json; "
            "upload identity cannot be verified against an immutable run snapshot"
        )

    resolved_dataset = Path(dataset_path or DEFAULT_RELEASE_DATASET).expanduser().resolve()
    resolved_gates = Path(gates_path or DEFAULT_RELEASE_GATES).expanduser().resolve()
    resolved_answer_dataset = Path(
        answer_dataset_path or DEFAULT_RELEASE_ANSWER_DATASET
    ).expanduser().resolve()
    resolved_answer_gates = Path(
        answer_gates_path or DEFAULT_RELEASE_ANSWER_GATES
    ).expanduser().resolve()
    eval_policy_errors: List[str] = []
    if canonical_production:
        eval_policy_errors.extend(
            validate_production_eval_inputs(
                retrieval_dataset=resolved_dataset,
                retrieval_gates=resolved_gates,
                answer_dataset=resolved_answer_dataset,
                answer_gates=resolved_answer_gates,
            )
        )
    eval_output_path = work_path / "release" / "retrieval_eval_report.json"
    eval_report: Dict[str, Any] = {}
    eval_errors: List[str] = []
    if not resolved_dataset.exists():
        eval_errors.append(f"Release evaluation dataset is missing: {resolved_dataset}")
    if not resolved_gates.exists():
        eval_errors.append(f"Release evaluation gates file is missing: {resolved_gates}")
    if not eval_errors:
        try:
            dataset_validation = validate_eval_examples(resolved_dataset, work_dir=work_path)
            if not bool(dataset_validation.get("ok")):
                eval_report = {
                    "dataset_path": str(resolved_dataset),
                    "work_dir": str(work_path),
                    "query_count": int(((dataset_validation.get("summary") or {}).get("query_count") or 0)),
                    "dataset_validation": dataset_validation,
                    "gates": {
                        "path": str(resolved_gates),
                        "passed": False,
                        "failures": [
                            {
                                "section": "dataset_validation",
                                "slice": "overall",
                                "metric": "gold_id_validity",
                                "reason": "invalid_eval_dataset",
                                "error_count": len(dataset_validation.get("errors") or []),
                                "sample_errors": (dataset_validation.get("errors") or [])[:5],
                            }
                        ],
                    },
                }
                atomic_write_json(eval_output_path, eval_report)
                eval_errors.append("Release evaluation dataset does not match the indexed run")
                _emit_progress(
                    progress_callback,
                    "release_check_retrieval_eval_skipped_invalid_dataset",
                    dataset_path=str(resolved_dataset),
                    report_path=str(eval_output_path),
                    error_count=len(dataset_validation.get("errors") or []),
                )
            else:
                _emit_progress(
                    progress_callback,
                    "release_check_retrieval_eval_start",
                    dataset_path=str(resolved_dataset),
                    gates_path=str(resolved_gates),
                )
                eval_report = evaluate_retrieval_dataset(
                    config_name=config_name,
                    work_dir=str(work_path),
                    dataset_path=str(resolved_dataset),
                    gates_path=str(resolved_gates),
                    query_cache_path=str(query_cache_path) if query_cache_path else None,
                    retrieval_cache_path=str(retrieval_cache_path) if retrieval_cache_path else None,
                    parallelism=max(1, int(parallelism or 1)),
                    progress_callback=progress_callback,
                )
                atomic_write_json(eval_output_path, eval_report)
                if not bool((eval_report.get("gates") or {}).get("passed")):
                    eval_errors.append("Retrieval evaluation gates did not pass")
                if _count(eval_report.get("query_count")) <= 0:
                    eval_errors.append("Retrieval evaluation query_count must be greater than zero")
                _emit_progress(
                    progress_callback,
                    "release_check_retrieval_eval_done",
                    gates_passed=bool((eval_report.get("gates") or {}).get("passed")),
                    report_path=str(eval_output_path),
                )
        except Exception as exc:
            eval_errors.append(f"Retrieval evaluation failed: {exc}")
    answer_output_path = work_path / "release" / "answer_readiness_report.json"
    answer_predictions_path = work_path / "release" / "answer_readiness_predictions.jsonl"
    answer_report: Dict[str, Any] = {}
    answer_errors: List[str] = []
    answer_readiness_waived = False
    if skip_answer_readiness or str(answer_eval_mode or "").strip().lower() == "disabled":
        waiver_reason = str(answer_readiness_waiver_reason or "").strip()
        answer_readiness_waived = bool(allow_answer_readiness_waiver and waiver_reason)
        answer_report = {
            "skipped": True,
            "reason": "disabled",
            "waived": answer_readiness_waived,
            "waiver_reason": waiver_reason if answer_readiness_waived else "",
            "gates": {"passed": False, "failures": [], "path": str(resolved_answer_gates)},
        }
        if not answer_readiness_waived:
            answer_errors.append(
                "Answer readiness was skipped; production promotion requires the gate or an explicit waiver with a reason"
            )
    else:
        if not resolved_answer_dataset.exists():
            answer_errors.append(f"Answer readiness dataset is missing: {resolved_answer_dataset}")
        if not resolved_answer_gates.exists():
            answer_errors.append(f"Answer readiness gates file is missing: {resolved_answer_gates}")
        if not answer_errors:
            try:
                _emit_progress(
                    progress_callback,
                    "release_check_answer_eval_start",
                    dataset_path=str(resolved_answer_dataset),
                    gates_path=str(resolved_answer_gates),
                    mode=str(answer_eval_mode or "local"),
                )
                answer_report = evaluate_answer_readiness(
                    config_name=config_name,
                    work_dir=str(work_path),
                    dataset_path=str(resolved_answer_dataset),
                    gates_path=str(resolved_answer_gates),
                    output_path=str(answer_output_path),
                    predictions_path=str(answer_predictions_path),
                    mode=str(answer_eval_mode or "local"),
                    endpoint=answer_endpoint,
                    auth_token=answer_auth_token,
                    widget_key=answer_widget_key,
                    model=answer_model,
                    timeout_seconds=answer_timeout_seconds,
                    probe_mode=answer_probe_mode,
                    eval_request_mode=answer_eval_request_mode,
                    resume_predictions=answer_resume_predictions,
                    judge_enabled=judge_enabled,
                    judge_model=judge_model,
                    judge_timeout_seconds=judge_timeout_seconds,
                    allow_openai_judge_fallback=not canonical_production,
                    parallelism=max(1, int(parallelism or 1)),
                    progress_callback=progress_callback,
                )
                if not bool((answer_report.get("gates") or {}).get("passed")):
                    answer_errors.append("Answer readiness gates did not pass")
                if _count(answer_report.get("query_count")) <= 0:
                    answer_errors.append("Answer readiness query_count must be greater than zero")
                _emit_progress(
                    progress_callback,
                    "release_check_answer_eval_done",
                    gates_passed=bool((answer_report.get("gates") or {}).get("passed")),
                    report_path=str(answer_output_path),
                )
            except Exception as exc:
                answer_errors.append(f"Answer readiness evaluation failed: {exc}")
                _emit_progress(
                    progress_callback,
                    "release_check_answer_eval_failed",
                    error=str(exc),
                )

    validation_policy_errors = []
    validation_policy_errors.extend(indexing_build_errors)
    if skip_stage_validation and canonical_production:
        validation_policy_errors.append(
            "Canonical production releases cannot skip stage validation"
        )
    if canonical_production and not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
        normalized_answer_runtime_commit,
    ):
        validation_policy_errors.append(
            "Canonical production answer evaluation must identify the evaluated backend Git commit"
        )
    retrieval_eval_policy = {
        **production_eval_manifest_metadata(answer=False),
        "query_count": _count(eval_report.get("query_count")),
    }
    answer_eval_policy = {
        **production_eval_manifest_metadata(answer=True),
        "query_count": _count(answer_report.get("query_count")),
        "llm_judge": answer_report.get("llm_judge") or {},
    }
    if canonical_production:
        eval_policy_errors.extend(
            validate_production_eval_manifest(
                retrieval_eval_policy,
                answer_eval_policy,
                allow_answer_waiver=answer_readiness_waived,
            )
        )
    release_errors = [
        *[f"validation: {error}" for error in validation_policy_errors],
        *[f"preflight: {check.get('message')}" for check in preflight.get("checks", []) if check.get("status") == "error"],
        *[f"audit: {issue.get('message')}" for issue in audit.to_dict().get("errors", [])],
        *[f"bundle: {error}" for error in bundle_errors],
        *[f"vector: {error}" for error in vector_errors],
        *[f"graph: {error}" for error in graph_errors],
        *[f"eval: {error}" for error in eval_errors],
        *[f"answer_eval: {error}" for error in answer_errors],
        *[f"eval_policy: {error}" for error in eval_policy_errors],
    ]
    passed = not release_errors

    upload_counts = _vector_uploaded_counts(vector_manifest)
    vector_index_name = str(vector_manifest.get("index_name") or vector_manifest.get("text_index_name") or "")
    vector_summary_index_name = str(vector_manifest.get("summary_index_name") or "")
    manifest = {
        "schema_version": 2 if canonical_production else 1,
        "release_id": f"{work_path.name}-{_now_iso()}",
        "created_at": _now_iso(),
        "config_name": config_name,
        "config_fingerprint": _stable_json_sha256(retrieval_contract_config),
        "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(
            config
        ),
        "production_serving_contract_fingerprint": production_serving_contract_fingerprint(
            config
        ),
        "run_id": work_path.name,
        "answer_runtime": {
            "commit_sha": normalized_answer_runtime_commit,
            "pipeline_revision": str(
                ((config.get("serving") or {}).get("answer_pipeline_revision") or "")
            ),
        },
        "indexing_build": indexing_build,
        "work_dir": str(work_path),
        "status": ("passed_with_waiver" if passed and answer_readiness_waived else "passed") if passed else "failed",
        "promoted": False,
        "errors": release_errors,
        "preflight": {
            "ok": bool(preflight.get("ok")),
            "error_count": int(preflight.get("error_count") or 0),
            "warning_count": int(preflight.get("warning_count") or 0),
        },
        "audit": {
            "ok": bool(audit.ok),
            "error_count": len(audit.errors),
            "warning_count": len(audit.warnings),
        },
        "vector_index": {
            "provider": vector_manifest.get("provider", "pinecone"),
            "production_indexing_contract_fingerprint": vector_manifest.get(
                "production_indexing_contract_fingerprint", ""
            ),
            "manifest_file": str(vector_manifest_path),
            "manifest_schema_version": vector_manifest.get("schema_version", 0),
            "indexing_build": vector_manifest.get("indexing_build") or {},
            "indexing_build_sha256": vector_manifest.get("indexing_build_sha256", ""),
            "contract": vector_manifest.get("vectorstore_contract", ""),
            "index_name": vector_index_name,
            "summary_index_name": vector_summary_index_name,
            "text_index_name": vector_manifest.get("text_index_name", ""),
            "sparse_index_name": vector_manifest.get("sparse_index_name", ""),
            "namespace": vector_manifest.get("namespace", ""),
            "namespaces": vector_manifest.get("namespaces") or {},
            "namespace_strategy": vector_manifest.get("namespace_strategy", ""),
            "namespace_release_id": vector_manifest.get("namespace_release_id", ""),
            "retrieval_bundle_sha256": vector_manifest.get("retrieval_bundle_sha256", ""),
            "lexical_corpus_sha256": vector_manifest.get("lexical_corpus_sha256", ""),
            "promoted_assertions_sha256": vector_manifest.get("promoted_assertions_sha256", ""),
            "knowledge_graph_kind": vector_manifest.get("knowledge_graph_kind", ""),
            "knowledge_graph_sha256": vector_manifest.get("knowledge_graph_sha256", ""),
            "knowledge_graph_index_sha256": vector_manifest.get("knowledge_graph_index_sha256", ""),
            "selected_release_assembly_sha256": vector_manifest.get(
                "selected_release_assembly_sha256", ""
            ),
            "selected_release_binding_sha256": vector_manifest.get(
                "selected_release_binding_sha256", ""
            ),
            "page_graph_navigation_catalog_sha256": vector_manifest.get(
                "page_graph_navigation_catalog_sha256", ""
            ),
            "upload_input_sha256": vector_manifest.get("upload_input_sha256", ""),
            "expected_uploads": _vector_expected_counts_for_manifest(
                vector_manifest,
                retrieval_bundle_stats,
                work_path,
                embedder_config,
            ),
            "uploaded": upload_counts,
            "verification": vector_manifest.get("verification") or {},
        },
        "knowledge_graph": graph_manifest,
        "retrieval_bundle": retrieval_bundle_stats,
        "evaluation": {
            **retrieval_eval_policy,
            "dataset_path": str(resolved_dataset),
            "dataset_fingerprint": eval_report.get("dataset_fingerprint", ""),
            "gates_path": str(resolved_gates),
            "report_path": str(eval_output_path) if eval_report else "",
            "query_count": eval_report.get("query_count", 0),
            "overall": eval_report.get("overall") or {},
            "gates": eval_report.get("gates") or {},
        },
        "answer_evaluation": {
            **answer_eval_policy,
            "dataset_path": str(resolved_answer_dataset),
            "dataset_fingerprint": answer_report.get("dataset_fingerprint", ""),
            "gates_path": str(resolved_answer_gates),
            "mode": str(answer_eval_mode or "local"),
            "endpoint": str(answer_endpoint or ""),
            "probe_mode": bool(answer_probe_mode),
            "eval_request_mode": bool(answer_eval_request_mode),
            "report_path": str(answer_output_path) if answer_report and not answer_report.get("skipped") else "",
            "predictions_path": str(answer_predictions_path) if answer_report and not answer_report.get("skipped") else "",
            "query_count": answer_report.get("query_count", 0),
            "llm_judge": answer_report.get("llm_judge") or {},
            "overall": answer_report.get("overall") or {},
            "gates": answer_report.get("gates") or {},
            "skipped": bool(answer_report.get("skipped")),
            "waived": bool(answer_report.get("waived")),
            "waiver_reason": str(answer_report.get("waiver_reason") or ""),
        },
    }
    _emit_progress(
        progress_callback,
        "release_check_done",
        passed=bool(passed),
        error_count=len(release_errors),
    )
    return manifest, passed


def write_release_manifest(manifest: Dict[str, Any], work_dir: str | Path) -> Path:
    path = release_manifest_path(work_dir)
    atomic_write_json(path, manifest)
    return path


@_serialize_active_pointer_update
def promote_release_manifest(
    *,
    manifest_path: str | Path,
    active_release_file: str | Path,
    promotion_attestation: Mapping[str, Any] | str | Path | None = None,
) -> Path:
    manifest_path = Path(manifest_path).expanduser().resolve()
    active_path = Path(active_release_file).expanduser().resolve()
    payload = load_json_safe(manifest_path, {}) or {}
    if not isinstance(payload, dict) or payload.get("status") not in {"passed", "passed_with_waiver"}:
        raise ValueError(f"Only passed release manifests can be promoted: {manifest_path}")

    integrity_errors: List[str] = []
    current_runtime_config: Dict[str, Any] = {}
    release_errors = payload.get("errors")
    if not isinstance(release_errors, list) or release_errors:
        integrity_errors.append("errors must be an empty list")
    preflight_payload = payload.get("preflight") if isinstance(payload.get("preflight"), Mapping) else {}
    audit_payload = payload.get("audit") if isinstance(payload.get("audit"), Mapping) else {}
    evaluation_payload = payload.get("evaluation") if isinstance(payload.get("evaluation"), Mapping) else {}
    retrieval_gates = evaluation_payload.get("gates") if isinstance(evaluation_payload.get("gates"), Mapping) else {}
    if not bool(preflight_payload.get("ok")):
        integrity_errors.append("production preflight is not marked successful")
    if not bool(audit_payload.get("ok")):
        integrity_errors.append("run audit is not marked successful")
    if retrieval_gates.get("passed") is not True:
        integrity_errors.append("retrieval evaluation gates are not marked passed")
    if _count(evaluation_payload.get("query_count")) <= 0:
        integrity_errors.append("retrieval evaluation query_count must be greater than zero")

    answer_evaluation = payload.get("answer_evaluation")
    if not isinstance(answer_evaluation, dict):
        integrity_errors.append("answer evaluation is missing")
        answer_evaluation = {}
    if payload.get("status") == "passed_with_waiver":
        if answer_evaluation.get("waived") is not True:
            integrity_errors.append("passed_with_waiver requires answer_evaluation.waived=true")
        if answer_evaluation.get("skipped") is not True:
            integrity_errors.append("passed_with_waiver requires a skipped answer evaluation")
        if not str(answer_evaluation.get("waiver_reason") or "").strip():
            integrity_errors.append("passed_with_waiver requires a non-empty waiver reason")
    else:
        if answer_evaluation.get("waived") is True or answer_evaluation.get("skipped") is True:
            integrity_errors.append("passed releases cannot skip or waive answer evaluation")
        answer_gates = (
            answer_evaluation.get("gates")
            if isinstance(answer_evaluation.get("gates"), Mapping)
            else {}
        )
        if answer_gates.get("passed") is not True:
            integrity_errors.append("answer evaluation gates are not marked passed")
        if _count(answer_evaluation.get("query_count")) <= 0:
            integrity_errors.append("answer evaluation query_count must be greater than zero")

    current_contract = _count(payload.get("schema_version")) >= 2 or Path(
        str(payload.get("config_name") or "")
    ).stem == "mbzuai_production"
    attestation_payload: Mapping[str, Any] | None = None
    if isinstance(promotion_attestation, Mapping):
        attestation_payload = deepcopy(dict(promotion_attestation))
    elif promotion_attestation is not None:
        loaded_attestation = load_json_safe(
            Path(promotion_attestation).expanduser().resolve(),
            None,
        )
        if isinstance(loaded_attestation, dict):
            attestation_payload = loaded_attestation
        else:
            integrity_errors.append("promotion attestation evidence file is missing or invalid")
    attestation_binding_manifest: Mapping[str, Any] = payload
    attestation_validation_time: datetime | None = None
    attestation_replay = False
    if current_contract and payload.get("promoted") is True:
        stored_attestation = payload.get("promotion_attestation")
        stored_attestation_digest = str(
            payload.get("promotion_attestation_sha256") or ""
        ).strip().lower()
        if not isinstance(stored_attestation, Mapping):
            integrity_errors.append("promoted release is missing its recorded promotion attestation")
        elif attestation_payload != stored_attestation:
            integrity_errors.append("promotion retry evidence does not match the recorded attestation")
        elif stored_attestation_digest != _stable_json_sha256(stored_attestation):
            integrity_errors.append("recorded promotion attestation digest is invalid")
        else:
            pre_promotion_payload = deepcopy(payload)
            pre_promotion_payload.pop("promotion_attestation", None)
            pre_promotion_payload.pop("promotion_attestation_sha256", None)
            promoted_at_text = str(pre_promotion_payload.pop("promoted_at", "") or "")
            pre_promotion_payload["promoted"] = False
            attestation_binding_manifest = pre_promotion_payload
            try:
                parsed_promoted_at = datetime.fromisoformat(
                    promoted_at_text.replace("Z", "+00:00")
                )
            except ValueError:
                integrity_errors.append("recorded promotion timestamp is invalid")
            else:
                if parsed_promoted_at.tzinfo is None or parsed_promoted_at.utcoffset() is None:
                    integrity_errors.append("recorded promotion timestamp must include a timezone")
                else:
                    attestation_validation_time = parsed_promoted_at.astimezone(timezone.utc)
                    attestation_replay = True
    if current_contract:
        integrity_errors.extend(
            validate_production_eval_manifest(
                evaluation_payload,
                answer_evaluation,
                allow_answer_waiver=payload.get("status") == "passed_with_waiver",
            )
        )
        answer_runtime = (
            payload.get("answer_runtime")
            if isinstance(payload.get("answer_runtime"), Mapping)
            else {}
        )
        if not re.fullmatch(
            r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
            str(answer_runtime.get("commit_sha") or "").strip().lower(),
        ):
            integrity_errors.append("production answer runtime commit SHA is missing or invalid")
        if not str(answer_runtime.get("pipeline_revision") or "").strip():
            integrity_errors.append("production answer runtime pipeline revision is missing")
        production_contract_fingerprint = str(
            payload.get("production_indexing_contract_fingerprint") or ""
        ).strip()
        if not production_contract_fingerprint:
            integrity_errors.append("production indexing contract fingerprint is missing")
        serving_contract_fingerprint = str(
            payload.get("production_serving_contract_fingerprint") or ""
        ).strip()
        if not serving_contract_fingerprint:
            integrity_errors.append("production serving contract fingerprint is missing")
        else:
            try:
                current_runtime_config = load_effective_config(
                    str(payload.get("config_name") or "mbzuai_production"),
                    work_dir=str(payload.get("work_dir") or ""),
                )
            except Exception as exc:
                integrity_errors.append(
                    f"production serving contract config could not be loaded: {exc}"
                )
            else:
                current_serving_fingerprint = production_serving_contract_fingerprint(
                    current_runtime_config
                )
                if serving_contract_fingerprint != current_serving_fingerprint:
                    integrity_errors.append(
                        "production serving contract changed after answer-readiness evaluation"
                    )
        manifest_work_dir = Path(str(payload.get("work_dir") or "")).expanduser()
        resolved_snapshot = load_json_safe(manifest_work_dir / "resolved_config.json", None)
        resolved_snapshot_config = (
            resolved_snapshot.get("config")
            if isinstance(resolved_snapshot, Mapping)
            else None
        )
        if not isinstance(resolved_snapshot_config, dict):
            integrity_errors.append("production resolved_config.json is missing or invalid")
        else:
            snapshot_fingerprint = production_indexing_contract_fingerprint(
                resolved_snapshot_config
            )
            recorded_snapshot_fingerprint = str(
                resolved_snapshot.get("production_indexing_contract_fingerprint") or ""
            ).strip()
            if recorded_snapshot_fingerprint != snapshot_fingerprint:
                integrity_errors.append("production resolved config fingerprint is missing or invalid")
            if production_contract_fingerprint != snapshot_fingerprint:
                integrity_errors.append("release production indexing fingerprint does not match its run snapshot")
        release_indexing_build = (
            payload.get("indexing_build")
            if isinstance(payload.get("indexing_build"), Mapping)
            else {}
        )
        snapshot_indexing_build = (
            resolved_snapshot.get("indexing_build")
            if isinstance(resolved_snapshot, Mapping)
            and isinstance(resolved_snapshot.get("indexing_build"), Mapping)
            else {}
        )
        indexing_commit = str(release_indexing_build.get("commit_sha") or "").strip().lower()
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", indexing_commit):
            integrity_errors.append("production indexing build commit SHA is missing or abbreviated")
        if release_indexing_build.get("dirty") is not False:
            integrity_errors.append("production indexing build was created from a dirty source tree")
        if release_indexing_build.get("implementation_sha256") != indexing_implementation_hashes():
            integrity_errors.append("production indexing implementation hashes do not match release code")
        if release_indexing_build != snapshot_indexing_build:
            integrity_errors.append("release indexing build identity does not match resolved_config.json")
        run_id = str(payload.get("run_id") or "").strip()
        vector_payload = payload.get("vector_index") if isinstance(payload.get("vector_index"), Mapping) else {}
        if vector_payload.get("indexing_build") != release_indexing_build:
            integrity_errors.append("production vector manifest indexing build identity mismatch")
        if vector_payload.get("indexing_build_sha256") != _stable_json_sha256(
            release_indexing_build
        ):
            integrity_errors.append("production vector manifest indexing build digest mismatch")
        if _count(vector_payload.get("manifest_schema_version")) < 4:
            integrity_errors.append("production vector manifest schema_version must be 4 or newer")
        promotion_selected = _is_selected_vector_manifest(vector_payload)
        if promotion_selected and _count(vector_payload.get("manifest_schema_version")) < 6:
            integrity_errors.append(
                "selected production vector manifest schema_version must be 6 or newer"
            )
        if str(vector_payload.get("namespace_strategy") or "").strip().lower() != "release":
            integrity_errors.append("production vector namespaces are not release-scoped")
        if str(vector_payload.get("namespace_release_id") or "").strip() != run_id:
            integrity_errors.append("production vector namespace release ID does not match run_id")
        namespaces = vector_payload.get("namespaces") if isinstance(vector_payload.get("namespaces"), Mapping) else {}
        promotion_namespace_keys = _vector_namespace_keys(vector_payload)
        missing_namespaces = [
            key for key in promotion_namespace_keys if not str(namespaces.get(key) or "").strip()
        ]
        if missing_namespaces:
            integrity_errors.append(f"production vector namespaces are incomplete: {missing_namespaces}")

        vector_manifest_file = Path(str(vector_payload.get("manifest_file") or "")).expanduser()
        vector_manifest = load_json_safe(vector_manifest_file, None)
        if not isinstance(vector_manifest, dict):
            integrity_errors.append("production vector upload manifest file is missing or invalid")
        else:
            vector_field_checks = {
                "provider": vector_payload.get("provider"),
                "production_indexing_contract_fingerprint": vector_payload.get(
                    "production_indexing_contract_fingerprint"
                ),
                "index_name": vector_payload.get("index_name"),
                "sparse_index_name": vector_payload.get("sparse_index_name"),
                "namespace_strategy": vector_payload.get("namespace_strategy"),
                "namespace_release_id": vector_payload.get("namespace_release_id"),
                "retrieval_bundle_sha256": vector_payload.get("retrieval_bundle_sha256"),
                "lexical_corpus_sha256": vector_payload.get("lexical_corpus_sha256"),
                "promoted_assertions_sha256": vector_payload.get("promoted_assertions_sha256"),
                "knowledge_graph_kind": vector_payload.get("knowledge_graph_kind"),
                "knowledge_graph_sha256": vector_payload.get("knowledge_graph_sha256"),
                "knowledge_graph_index_sha256": vector_payload.get("knowledge_graph_index_sha256"),
                "selected_release_assembly_sha256": vector_payload.get(
                    "selected_release_assembly_sha256"
                ),
                "selected_release_binding_sha256": vector_payload.get(
                    "selected_release_binding_sha256"
                ),
                "page_graph_navigation_catalog_sha256": vector_payload.get(
                    "page_graph_navigation_catalog_sha256"
                ),
                "upload_input_sha256": vector_payload.get("upload_input_sha256"),
            }
            for key, release_value in vector_field_checks.items():
                if vector_manifest.get(key) != release_value:
                    integrity_errors.append(
                        f"release vector field {key!r} does not match its upload manifest"
                    )
            if dict(vector_manifest.get("namespaces") or {}) != dict(namespaces):
                integrity_errors.append("release vector namespaces do not match the upload manifest file")
            if dict(vector_manifest.get("uploaded") or {}) != dict(
                vector_payload.get("uploaded") or {}
            ):
                integrity_errors.append(
                    "release vector uploaded counts do not match the upload manifest file"
                )

        bundle_payload = payload.get("retrieval_bundle") if isinstance(payload.get("retrieval_bundle"), Mapping) else {}
        bundle_file = Path(str(bundle_payload.get("retrieval_bundle_file") or "")).expanduser()
        bundle_sha = str(bundle_payload.get("retrieval_bundle_sha256") or "").strip()
        if not bundle_file.is_file() or not bundle_sha:
            integrity_errors.append("production retrieval bundle file/hash is missing")
        elif sha256_file(bundle_file) != bundle_sha:
            integrity_errors.append("production retrieval bundle hash no longer matches its file")
        if str(vector_payload.get("retrieval_bundle_sha256") or "").strip() != bundle_sha:
            integrity_errors.append("production vector manifest is not bound to the release retrieval bundle")
        lexical_file = Path(str(bundle_payload.get("lexical_corpus_file") or "")).expanduser()
        lexical_sha = str(bundle_payload.get("lexical_corpus_sha256") or "").strip()
        if not lexical_file.is_file() or not lexical_sha:
            integrity_errors.append("production lexical corpus file/hash is missing")
        elif sha256_file(lexical_file) != lexical_sha:
            integrity_errors.append("production lexical corpus hash no longer matches its file")
        if str(vector_payload.get("lexical_corpus_sha256") or "").strip() != lexical_sha:
            integrity_errors.append("production vector manifest is not bound to the lexical corpus")

        promoted_assertions_file = Path(
            str(bundle_payload.get("promoted_assertions_file") or "")
        ).expanduser()
        promoted_assertions_sha = str(
            bundle_payload.get("promoted_assertions_sha256") or ""
        ).strip()
        if not promoted_assertions_file.is_file() or not promoted_assertions_sha:
            integrity_errors.append("production promoted assertions file/hash is missing")
        elif sha256_file(promoted_assertions_file) != promoted_assertions_sha:
            integrity_errors.append(
                "production promoted assertions hash no longer matches its file"
            )
        if (
            str(vector_payload.get("promoted_assertions_sha256") or "").strip()
            != promoted_assertions_sha
        ):
            integrity_errors.append(
                "production vector manifest is not bound to the promoted assertions"
            )

        selected_release_hashes: List[str] = []
        if promotion_selected:
            assembly_path = (
                manifest_work_dir
                / "stage_outputs"
                / "assemble_selected_release"
                / "selected_release_assembly.json"
            )
            assembly_payload = load_json_safe(assembly_path, None)
            if not isinstance(assembly_payload, Mapping):
                integrity_errors.append("production selected release assembly is missing or invalid")
            else:
                from pipeline.core.release_assembly import (
                    SELECTED_DENSE_RECORD_KINDS,
                    SELECTED_EMBEDDING_SPEC_KEYS,
                    SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
                    SELECTED_RELEASE_BINDING_ORDER,
                    SELECTED_RELEASE_SOURCE_HASH_KEYS,
                    SelectedReleaseAssemblyError,
                    selected_release_file_path,
                    validate_selected_release_embedding_spec,
                )

                assembly_manifest_sha = sha256_file(assembly_path)
                assembly_binding_sha = str(
                    assembly_payload.get("assembly_sha256") or ""
                ).strip().lower()
                if str(assembly_payload.get("schema_version") or "") != SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION:
                    integrity_errors.append("production selected release assembly schema is unsupported")
                if str(assembly_payload.get("status") or "") != "ready_for_embedding":
                    integrity_errors.append("production selected release assembly is not ready")
                selected_profile = (
                    current_runtime_config.get("selected_profile")
                    if isinstance(current_runtime_config.get("selected_profile"), Mapping)
                    else {}
                )
                if str(assembly_payload.get("variant_id") or "") != str(
                    selected_profile.get("variant_id") or ""
                ):
                    integrity_errors.append(
                        "production selected release assembly variant does not match config"
                    )
                if tuple(assembly_payload.get("record_kinds") or ()) != SELECTED_DENSE_RECORD_KINDS:
                    integrity_errors.append(
                        "production selected release assembly record kinds drifted"
                    )
                try:
                    embedding_spec = validate_selected_release_embedding_spec(
                        assembly_payload
                    )
                except SelectedReleaseAssemblyError as exc:
                    integrity_errors.append(str(exc))
                    embedding_spec = {}
                if embedding_spec:
                    embedder_config = (
                        current_runtime_config.get("embedder")
                        if isinstance(current_runtime_config.get("embedder"), Mapping)
                        else {}
                    )
                    configured_embedding = {
                        "provider": str(embedder_config.get("engine") or "").strip(),
                        "model": str(embedder_config.get("model") or "").strip(),
                        "dimensions": int(
                            embedder_config.get("output_dimensionality") or 0
                        ),
                        "query_format": str(
                            embedder_config.get("query_format") or ""
                        ).strip(),
                        "document_format": str(
                            embedder_config.get("document_format") or ""
                        ).strip(),
                        "media_input": str(
                            embedder_config.get("media_input") or ""
                        ).strip().casefold(),
                    }
                    for key in SELECTED_EMBEDDING_SPEC_KEYS:
                        if configured_embedding[key] != embedding_spec[key]:
                            integrity_errors.append(
                                f"production embedder.{key} differs from the selected release"
                            )
                    uploaded_profile = (
                        vector_payload.get("selected_profile")
                        if isinstance(vector_payload.get("selected_profile"), Mapping)
                        else {}
                    )
                    if str(vector_payload.get("media_input") or "") != str(
                        embedding_spec["media_input"]
                    ) or str(uploaded_profile.get("media_input") or "") != str(
                        embedding_spec["media_input"]
                    ):
                        integrity_errors.append(
                            "production vector upload media input differs from the selected release"
                        )
                    if uploaded_profile.get("embedding_spec") != embedding_spec:
                        integrity_errors.append(
                            "production vector upload embedding spec differs from the selected release"
                        )
                if tuple(assembly_payload.get("binding_order") or ()) != SELECTED_RELEASE_BINDING_ORDER:
                    integrity_errors.append(
                        "production selected release assembly binding order drifted"
                    )
                if assembly_payload.get("embedding_performed") is not False or assembly_payload.get(
                    "upload_performed"
                ) is not False:
                    integrity_errors.append(
                        "production selected release assembly is not an immutable pre-embedding artifact"
                    )

                assembly_files = (
                    assembly_payload.get("files")
                    if isinstance(assembly_payload.get("files"), Mapping)
                    else {}
                )
                if set(assembly_files) != set(SELECTED_RELEASE_BINDING_ORDER):
                    integrity_errors.append(
                        "production selected release assembly file set drifted"
                    )
                resolved_assembly_files: Dict[str, Path] = {}
                assembly_file_hashes: List[str] = []
                for file_key in SELECTED_RELEASE_BINDING_ORDER:
                    try:
                        resolved_file = selected_release_file_path(
                            assembly_payload,
                            assembly_path,
                            file_key,
                        )
                    except SelectedReleaseAssemblyError as exc:
                        integrity_errors.append(str(exc))
                    else:
                        resolved_assembly_files[file_key] = resolved_file
                        assembly_file_hashes.append(sha256_file(resolved_file))
                if len(set(resolved_assembly_files.values())) != len(
                    resolved_assembly_files
                ):
                    integrity_errors.append(
                        "production selected release assembly files do not resolve uniquely"
                    )

                source = (
                    assembly_payload.get("source")
                    if isinstance(assembly_payload.get("source"), Mapping)
                    else {}
                )
                source_hashes = [
                    str(source.get(key) or "").strip().lower()
                    for key in SELECTED_RELEASE_SOURCE_HASH_KEYS
                ]
                if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in source_hashes):
                    integrity_errors.append(
                        "production selected release assembly source hashes are incomplete"
                    )
                elif len(assembly_file_hashes) == len(SELECTED_RELEASE_BINDING_ORDER):
                    computed_binding_sha = combine_sha256_digests(
                        *source_hashes,
                        *assembly_file_hashes,
                    )
                    if computed_binding_sha != assembly_binding_sha:
                        integrity_errors.append(
                            "production selected release assembly binding digest is invalid"
                        )

                navigation_path = resolved_assembly_files.get("navigation_catalog")
                navigation_sha = sha256_file(navigation_path) if navigation_path else ""
                dense_counts = (
                    assembly_payload.get("dense_lane_counts")
                    if isinstance(assembly_payload.get("dense_lane_counts"), Mapping)
                    else {}
                )
                uploaded_counts = (
                    vector_payload.get("uploaded")
                    if isinstance(vector_payload.get("uploaded"), Mapping)
                    else {}
                )
                for lane in ("chunks", "parents", "media", "page_cards", "actions"):
                    expected_count = _count(dense_counts.get(lane))
                    if expected_count <= 0 or _count(uploaded_counts.get(lane)) != expected_count:
                        integrity_errors.append(
                            f"production selected release count differs for {lane}"
                        )
                for lane in (
                    "facts",
                    "evidence_spans",
                    "summaries",
                    "assertions",
                    "entities",
                    "communities",
                ):
                    if _count(uploaded_counts.get(lane)) != 0:
                        integrity_errors.append(
                            f"production selected release contains unevaluated lane {lane}"
                        )
                coverage = (
                    assembly_payload.get("coverage")
                    if isinstance(assembly_payload.get("coverage"), Mapping)
                    else {}
                )
                if coverage.get("all_candidate_chunks_mapped") is not True or coverage.get(
                    "all_navigation_chunks_remapped"
                ) is not True:
                    integrity_errors.append(
                        "production selected release assembly coverage is incomplete"
                    )
                selected_release_hashes = [
                    assembly_manifest_sha,
                    assembly_binding_sha,
                    navigation_sha,
                ]
                for key, actual in zip(
                    (
                        "selected_release_assembly_sha256",
                        "selected_release_binding_sha256",
                        "page_graph_navigation_catalog_sha256",
                    ),
                    selected_release_hashes,
                ):
                    if str(vector_payload.get(key) or "").strip().lower() != actual:
                        integrity_errors.append(
                            f"production vector manifest is not bound to runtime {key}"
                        )
                    if str(bundle_payload.get(key) or "").strip().lower() != actual:
                        integrity_errors.append(
                            f"production retrieval bundle contract is not bound to runtime {key}"
                        )

        graph_payload = payload.get("knowledge_graph") if isinstance(payload.get("knowledge_graph"), Mapping) else {}
        graph_backend = str(graph_payload.get("store_backend") or "").strip()
        graph_sha = str(graph_payload.get("knowledge_graph_sha256") or "").strip()
        graph_index_sha = ""
        if graph_backend != "disabled":
            graph_file_value = (
                graph_payload.get("manifest_file")
                if graph_backend == "local_json"
                else graph_payload.get("knowledge_graph_file")
            )
            graph_file = Path(str(graph_file_value or "")).expanduser()
            if not graph_file.is_file() or not graph_sha:
                integrity_errors.append("production knowledge graph file/hash is missing")
            elif sha256_file(graph_file) != graph_sha:
                integrity_errors.append("production knowledge graph hash no longer matches its file")
            if str(vector_payload.get("knowledge_graph_sha256") or "").strip() != graph_sha:
                integrity_errors.append("production vector manifest is not bound to the release knowledge graph")

            if graph_backend == "local_json":
                graph_index_file = Path(str(graph_payload.get("index_file") or "")).expanduser()
                graph_index_sha = str(graph_payload.get("knowledge_graph_index_sha256") or "").strip()
                if not graph_index_file.is_file() or not graph_index_sha:
                    integrity_errors.append("production knowledge graph index file/hash is missing")
                elif sha256_file(graph_index_file) != graph_index_sha:
                    integrity_errors.append("production knowledge graph index hash no longer matches its file")
                if str(vector_payload.get("knowledge_graph_index_sha256") or "").strip() != graph_index_sha:
                    integrity_errors.append(
                        "production vector manifest is not bound to the release knowledge graph index"
                    )
                if graph_file.is_file() and graph_index_file.is_file():
                    derivation_issues = validate_graph_index_derivation(
                        graph_file,
                        graph_index_file,
                    )
                    if derivation_issues:
                        integrity_errors.append(
                            "production knowledge graph index is not derived from its graph: "
                            + str(
                                derivation_issues[0].get("message")
                                or derivation_issues[0].get("code")
                            )
                        )
                    graph_config = (
                        current_runtime_config.get("graph")
                        if isinstance(current_runtime_config.get("graph"), Mapping)
                        else {}
                    )
                    minimum_summary_characters = int(
                        graph_config.get("community_summary_min_characters", 40) or 40
                    )
                    minimum_summary_coverage = float(
                        graph_config.get("community_summary_min_coverage_ratio", 1.0)
                    )
                    summary_quality = community_summary_quality(
                        load_graph_bundle(graph_file),
                        min_characters=minimum_summary_characters,
                    )
                    if int(summary_quality["total_communities"]) <= 0:
                        integrity_errors.append(
                            "production knowledge graph contains no community nodes"
                        )
                    elif float(summary_quality["coverage_ratio"]) < minimum_summary_coverage:
                        integrity_errors.append(
                            "production knowledge graph community-summary coverage is below threshold: "
                            f"actual={float(summary_quality['coverage_ratio']):.4f}, "
                            f"required={minimum_summary_coverage:.4f}"
                        )

        upload_digests = [
            bundle_sha,
            lexical_sha,
            promoted_assertions_sha,
            graph_sha,
            graph_index_sha,
        ]
        if promotion_selected:
            upload_digests.extend(selected_release_hashes)
        if all(upload_digests):
            expected_upload_input_sha = combine_sha256_digests(*upload_digests)
            if str(vector_payload.get("upload_input_sha256") or "").strip().lower() != expected_upload_input_sha:
                integrity_errors.append(
                    "production upload_input_sha256 does not bind the complete release assembly"
                )

        integrity_errors.extend(
            _validate_promotion_attestation(
                attestation_binding_manifest,
                attestation_payload,
                current_runtime_config=current_runtime_config,
                now=attestation_validation_time,
            )
        )

    if integrity_errors:
        raise ValueError(
            f"Release manifest failed promotion integrity checks: {manifest_path}: "
            + "; ".join(integrity_errors)
        )
    if current_contract and not attestation_replay:
        assert isinstance(attestation_payload, Mapping)
        payload["promotion_attestation"] = deepcopy(dict(attestation_payload))
        payload["promotion_attestation_sha256"] = _stable_json_sha256(attestation_payload)
    if not attestation_replay:
        payload["promoted"] = True
        payload["promoted_at"] = _now_iso()
        atomic_write_json(manifest_path, payload)
    atomic_write_json(
        active_path,
        {
            "schema_version": 1,
            "active_release_manifest": str(manifest_path),
            "release_id": payload.get("release_id"),
            "run_id": payload.get("run_id"),
            "production_indexing_contract_fingerprint": payload.get(
                "production_indexing_contract_fingerprint"
            ),
            "production_serving_contract_fingerprint": payload.get(
                "production_serving_contract_fingerprint"
            ),
            "answer_runtime_commit_sha": (
                (payload.get("answer_runtime") or {}).get("commit_sha")
                if isinstance(payload.get("answer_runtime"), Mapping)
                else ""
            ),
            "indexing_build_commit_sha": (
                (payload.get("indexing_build") or {}).get("commit_sha")
                if isinstance(payload.get("indexing_build"), Mapping)
                else ""
            ),
            "promotion_attestation_sha256": str(
                payload.get("promotion_attestation_sha256") or ""
            ),
            "backend_commit_sha": str(
                (attestation_payload or {}).get("backend_commit_sha") or ""
            ),
            "retriever_commit_sha": str(
                (attestation_payload or {}).get("retriever_commit_sha") or ""
            ),
            "promoted_at": payload["promoted_at"],
            "status": payload.get("status"),
        },
    )
    return active_path
