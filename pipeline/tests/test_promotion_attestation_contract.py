from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from pipeline.core.io import atomic_write_json
from pipeline.core.release import (
    _validate_promotion_attestation,
    build_promotion_attestation_evidence,
    promote_release_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVER_COMMIT = "d" * 40
BACKEND_COMMIT = "e" * 40
INDEXING_COMMIT = "f" * 40
ANSWER_MODELS = {
    "generation_model": "gpt-5.4-2026-03-05",
    "query_rewrite_model": "gpt-5.4-mini-2026-03-17",
    "reranker_model": "gpt-5.4-mini-2026-03-17",
    "grounded_finalizer_model": "gpt-5.4-mini-2026-03-17",
}


@pytest.fixture(autouse=True)
def _promotion_signing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_SERVICE_TOKEN",
        "Rtrv_7zQ9-aB3mN8.xK2pL6:sD4wF1cV5",
    )
    monkeypatch.setenv(
        "CANDIDATE_BACKEND_OPERATIONS_TOKEN",
        "Ops_7zQ9-aB3mN8.xK2pL6:sD4wF1cV5",
    )


def _candidate_contract(tmp_path: Path) -> tuple[Path, dict, dict, dict]:
    hashes = {
        "retrieval_bundle_sha256": "1" * 64,
        "knowledge_graph_sha256": "2" * 64,
        "knowledge_graph_index_sha256": "3" * 64,
        "lexical_corpus_sha256": "4" * 64,
        "promoted_assertions_sha256": "5" * 64,
    }
    manifest = {
        "schema_version": 2,
        "status": "passed",
        "promoted": False,
        "errors": [],
        "release_id": "run-1-release",
        "run_id": "run-1",
        "config_name": "mbzuai_production",
        "work_dir": str(tmp_path / "run-1"),
        "answer_runtime": {
            "commit_sha": BACKEND_COMMIT,
            "pipeline_revision": "mbzuai-agentic-grounded-v1",
        },
        "indexing_build": {"commit_sha": INDEXING_COMMIT},
        "vector_index": dict(hashes),
        "retrieval_bundle": {
            "retrieval_bundle_sha256": hashes["retrieval_bundle_sha256"],
        },
    }
    manifest_path = tmp_path / "run-1" / "release" / "retrieval_release_manifest.json"
    atomic_write_json(manifest_path, manifest)
    identity = {
        "run_id": "run-1",
        "commit_sha": RETRIEVER_COMMIT,
        "indexing_build_commit_sha": INDEXING_COMMIT,
        **hashes,
    }
    retriever = {
        "ready": True,
        "config_name": "mbzuai_production",
        **identity,
    }
    backend = {
        "status": "healthy",
        "release": {"commit_sha": BACKEND_COMMIT},
        "checks": {
            "openai": {"status": "healthy", **ANSWER_MODELS},
            "retrieval_service": {
                "status": "healthy",
                "ready": True,
                "mode": "required",
                **identity,
            },
        },
    }
    return manifest_path, manifest, retriever, backend


def test_promotion_evidence_is_fresh_non_secret_and_bound_to_exact_manifest(
    tmp_path: Path,
) -> None:
    manifest_path, manifest, retriever, backend = _candidate_contract(tmp_path)
    evidence = build_promotion_attestation_evidence(
        manifest_path=manifest_path,
        retriever_attestation=retriever,
        backend_health=backend,
        expected_retriever_commit_sha=RETRIEVER_COMMIT,
        expected_backend_commit_sha=BACKEND_COMMIT,
    )

    assert _validate_promotion_attestation(
        manifest,
        evidence,
        current_runtime_config={"serving": ANSWER_MODELS},
    ) == []
    assert "token" not in json.dumps(evidence).lower()
    assert "url" not in json.dumps(evidence).lower()

    changed_manifest = deepcopy(manifest)
    changed_manifest["release_id"] = "different-release"
    errors = _validate_promotion_attestation(
        changed_manifest,
        evidence,
        current_runtime_config={"serving": ANSWER_MODELS},
    )
    assert any("exact pre-promotion release manifest" in error for error in errors)
    assert any("release_id" in error for error in errors)


def test_promotion_evidence_rejects_stale_unknown_and_model_drift(tmp_path: Path) -> None:
    manifest_path, manifest, retriever, backend = _candidate_contract(tmp_path)
    evidence = build_promotion_attestation_evidence(
        manifest_path=manifest_path,
        retriever_attestation=retriever,
        backend_health=backend,
        expected_retriever_commit_sha=RETRIEVER_COMMIT,
        expected_backend_commit_sha=BACKEND_COMMIT,
    )
    evidence["attested_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=10)
    ).isoformat()
    evidence["api_key"] = "must-never-be-recorded"
    drifted_models = {**ANSWER_MODELS, "generation_model": "different-model"}

    errors = _validate_promotion_attestation(
        manifest,
        evidence,
        current_runtime_config={"serving": drifted_models},
    )

    assert any("stale" in error for error in errors)
    assert any("unexpected fields" in error for error in errors)
    assert any("generation_model" in error for error in errors)


def test_promotion_evidence_creation_rejects_candidate_identity_drift(tmp_path: Path) -> None:
    manifest_path, _, retriever, backend = _candidate_contract(tmp_path)
    backend["checks"]["retrieval_service"]["run_id"] = "old-run"

    with pytest.raises(ValueError, match="retrieval identity mismatch for run_id"):
        build_promotion_attestation_evidence(
            manifest_path=manifest_path,
            retriever_attestation=retriever,
            backend_health=backend,
            expected_retriever_commit_sha=RETRIEVER_COMMIT,
            expected_backend_commit_sha=BACKEND_COMMIT,
        )


def test_forged_unsigned_promotion_evidence_is_rejected(tmp_path: Path) -> None:
    manifest_path, manifest, retriever, backend = _candidate_contract(tmp_path)
    evidence = build_promotion_attestation_evidence(
        manifest_path=manifest_path,
        retriever_attestation=retriever,
        backend_health=backend,
        expected_retriever_commit_sha=RETRIEVER_COMMIT,
        expected_backend_commit_sha=BACKEND_COMMIT,
    )
    evidence["signature_sha256"] = "0" * 64

    errors = _validate_promotion_attestation(
        manifest,
        evidence,
        current_runtime_config={"serving": ANSWER_MODELS},
    )

    assert "promotion attestation signature verification failed" in errors


def test_production_core_promotion_requires_attestation_evidence(tmp_path: Path) -> None:
    manifest_path, _, _, _ = _candidate_contract(tmp_path)

    with pytest.raises(ValueError, match="fresh promotion attestation evidence is required"):
        promote_release_manifest(
            manifest_path=manifest_path,
            active_release_file=tmp_path / "active_release.json",
        )


def test_direct_production_release_check_promote_fails_before_evaluation(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline",
            "release-check",
            "--config",
            "mbzuai_production",
            "--work-dir",
            str(tmp_path / "does-not-exist"),
            "--promote",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Direct production release-check --promote is forbidden" in result.stderr
    assert not (tmp_path / "does-not-exist" / "release").exists()


def test_release_check_cli_declares_governed_split_option() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pipeline", "release-check", "--help"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--split" in result.stdout
    assert "selection" in result.stdout
    assert "holdout" in result.stdout
    assert "regression" in result.stdout
