from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from pipeline.tests.test_deploy_startup_safety import INDEXING_BUILD, _make_required_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMOTE_SCRIPT = PROJECT_ROOT / "scripts" / "deploy" / "release-check-promote.sh"
RUN_LOCK_SCRIPT = PROJECT_ROOT / "scripts" / "deploy" / "with-run-lock.py"


def _run_promotion_attestation(
    tmp_path: Path,
    *,
    backend_run_id: str = "candidate-1",
    backend_commit_sha: str = "e" * 40,
    backend_reranker_enabled: bool = False,
    drift_backend_run_after_initial_attestation: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate-1"
    _make_required_artifacts(work_dir)
    upload = json.loads(
        (work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    identity_hashes = {
        "indexing_build_commit_sha": INDEXING_BUILD["commit_sha"],
        "retrieval_bundle_sha256": upload["retrieval_bundle_sha256"],
        "knowledge_graph_sha256": upload["knowledge_graph_sha256"],
        "knowledge_graph_index_sha256": upload["knowledge_graph_index_sha256"],
        "lexical_corpus_sha256": upload["lexical_corpus_sha256"],
        "promoted_assertions_sha256": upload["promoted_assertions_sha256"],
    }
    release_manifest = {
        "schema_version": 2,
        "status": "passed",
        "promoted": False,
        "errors": [],
        "release_id": "candidate-1-release",
        "run_id": "candidate-1",
        "config_name": "mbzuai_production",
        "work_dir": str(work_dir),
        "answer_runtime": {
            "commit_sha": "e" * 40,
            "pipeline_revision": "mbzuai-agentic-grounded-v1",
        },
        "indexing_build": INDEXING_BUILD,
        "vector_index": {
            "retrieval_bundle_sha256": identity_hashes["retrieval_bundle_sha256"],
            "knowledge_graph_sha256": identity_hashes["knowledge_graph_sha256"],
            "knowledge_graph_index_sha256": identity_hashes[
                "knowledge_graph_index_sha256"
            ],
            "lexical_corpus_sha256": identity_hashes["lexical_corpus_sha256"],
            "promoted_assertions_sha256": identity_hashes[
                "promoted_assertions_sha256"
            ],
        },
        "retrieval_bundle": {
            "retrieval_bundle_sha256": identity_hashes["retrieval_bundle_sha256"],
        },
    }
    release_manifest_path = (
        work_dir / "release" / "retrieval_release_manifest.json"
    )
    release_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    release_manifest_path.write_text(json.dumps(release_manifest), encoding="utf-8")
    commit_sha = "d" * 40
    retriever_payload = {
        "ready": True,
        "config_name": "mbzuai_production",
        "run_id": "candidate-1",
        "commit_sha": commit_sha,
        **identity_hashes,
    }
    backend_payload = {
        "status": "healthy",
        "release": {"commit_sha": backend_commit_sha},
        "checks": {
            "openai": {
                "status": "healthy",
                "generation_model": "gpt-5.4-2026-03-05",
                "query_rewrite_model": "gpt-5.4-mini-2026-03-17",
                "reranker_model": "gpt-5.4-mini-2026-03-17",
                "grounded_finalizer_model": "gpt-5.4-mini-2026-03-17",
            },
            "reranker": {
                "status": "healthy",
                "enabled": backend_reranker_enabled,
                "mode": "enabled" if backend_reranker_enabled else "disabled",
            },
            "retrieval_service": {
                "status": "healthy",
                "ready": True,
                "mode": "required",
                "run_id": backend_run_id,
                "commit_sha": commit_sha,
                **identity_hashes,
            }
        },
    }

    backend_request_count = 0

    class CandidateHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            nonlocal backend_request_count
            if self.path == "/attestationz":
                if self.headers.get("X-Retrieval-Service-Token") != env_token:
                    self.send_response(401)
                    self.end_headers()
                    return
                payload = retriever_payload
            else:
                backend_request_count += 1
                payload = json.loads(json.dumps(backend_payload))
                if (
                    drift_backend_run_after_initial_attestation
                    and backend_request_count >= 2
                ):
                    payload["checks"]["retrieval_service"]["run_id"] = "drifted-run"
            body = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), CandidateHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    fake_python = tmp_path / "python-with-release-check-stub"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >>\"$PIPELINE_CALL_LOG\"\n"
        "if [[ \"${1:-}\" == \"-m\" && \"${2:-}\" == \"pipeline\" ]]; then\n"
        "  exit 0\n"
        "fi\n"
        "exec \"$REAL_PYTHON\" \"$@\"\n",
        encoding="utf-8",
    )
    fake_python.chmod(0o750)
    marker = tmp_path / ".mbzuai-release-storage"
    marker.write_text('{"mode":"persistent"}\n', encoding="utf-8")
    env_token = "prod-retrieval-token-7Y!k2Qx9M4vB"
    env = {
        **os.environ,
        "PYTHON": str(fake_python),
        "REAL_PYTHON": sys.executable,
        "RELEASE_WORK_DIR": str(work_dir),
        "RELEASE_RUNS_ROOT": str(runs_root),
        "RELEASE_STORAGE_MARKER_FILE": str(marker),
        "RELEASE_STORAGE_MODE": "persistent",
        "ACTIVE_RELEASE_FILE": str(tmp_path / "active" / "active_release.json"),
        "RELEASE_COMMIT_SHA": commit_sha,
        "ANSWER_EVAL_ENDPOINT": f"ws://127.0.0.1:{server.server_port}/chat",
        "CANDIDATE_RETRIEVER_ATTESTATION_URL": f"http://127.0.0.1:{server.server_port}/attestationz",
        "CANDIDATE_BACKEND_DETAILED_URL": f"http://127.0.0.1:{server.server_port}/health/detailed",
        "CANDIDATE_ATTESTATION_ALLOWED_HOSTS": "127.0.0.1",
        "CANDIDATE_BACKEND_OPERATIONS_TOKEN": "Ops_7zQ9-aB3mN8.xK2pL6:sD4wF1cV5",
        "CANDIDATE_BACKEND_COMMIT_SHA": "e" * 40,
        "RETRIEVAL_SERVICE_TOKEN": env_token,
        "PIPELINE_CALL_LOG": str(tmp_path / "pipeline-calls.log"),
        **(extra_env or {}),
    }
    try:
        return subprocess.run(
            ["bash", str(PROMOTE_SCRIPT)],
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_release_promotion_attests_candidate_backend_identity(tmp_path: Path) -> None:
    result = _run_promotion_attestation(tmp_path)

    assert result.returncode == 0, result.stderr
    assert "Candidate attestation passed" in result.stdout
    evidence_path = (
        tmp_path
        / "runs"
        / "candidate-1"
        / "release"
        / "promotion_attestation.json"
    )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["run_id"] == "candidate-1"
    assert evidence["backend_commit_sha"] == "e" * 40
    assert evidence["retriever_commit_sha"] == "d" * 40
    assert "token" not in json.dumps(evidence).lower()
    pipeline_calls = (tmp_path / "pipeline-calls.log").read_text(encoding="utf-8")
    assert "release-check" in pipeline_calls
    assert "release-check --config" in pipeline_calls
    assert "release-check --config" in pipeline_calls.split("promote-release", 1)[0]
    assert "--promote" not in pipeline_calls.split("promote-release", 1)[0]
    assert "--resume-answer-predictions" in pipeline_calls.split("promote-release", 1)[0]
    assert "promote-release --manifest" in pipeline_calls


def test_release_promotion_rejects_backend_using_another_run(tmp_path: Path) -> None:
    result = _run_promotion_attestation(tmp_path, backend_run_id="old-active-run")

    assert result.returncode != 0
    assert "backend-candidate retrieval identity mismatch for run_id" in result.stderr


def test_release_promotion_rejects_unexpected_backend_code_revision(tmp_path: Path) -> None:
    result = _run_promotion_attestation(tmp_path, backend_commit_sha="f" * 40)

    assert result.returncode != 0
    assert "backend-candidate code revision mismatch" in result.stderr


def test_release_promotion_rejects_duplicate_backend_llm_reranker(tmp_path: Path) -> None:
    result = _run_promotion_attestation(tmp_path, backend_reranker_enabled=True)

    assert result.returncode != 0
    assert "must disable the duplicate backend LLM reranker" in result.stderr


def test_release_promotion_rechecks_candidate_after_evaluation(tmp_path: Path) -> None:
    result = _run_promotion_attestation(
        tmp_path,
        drift_backend_run_after_initial_attestation=True,
    )

    assert result.returncode != 0
    assert "backend-candidate retrieval identity mismatch for run_id" in result.stderr
    pipeline_calls = (tmp_path / "pipeline-calls.log").read_text(encoding="utf-8")
    assert "release-check --config" in pipeline_calls
    assert "promote-release --manifest" not in pipeline_calls


def test_release_promotion_rejects_evaluation_path_override(tmp_path: Path) -> None:
    result = _run_promotion_attestation(
        tmp_path,
        extra_env={"RELEASE_GATES": str(tmp_path / "untrusted-gates.json")},
    )

    assert result.returncode != 0
    assert "RELEASE_GATES cannot override the canonical committed production evaluation contract" in result.stderr


def test_release_promotion_rejects_unallowlisted_candidate_before_curl(tmp_path: Path) -> None:
    result = _run_promotion_attestation(
        tmp_path,
        extra_env={
            "CANDIDATE_RETRIEVER_ATTESTATION_URL": "http://localhost:9/attestationz",
        },
    )

    assert result.returncode != 0
    assert "hostname is not exactly allowlisted" in result.stderr
    script = PROMOTE_SCRIPT.read_text(encoding="utf-8")
    assert script.index("hostname is not exactly allowlisted") < script.index("curl --fail")


def test_release_promotion_rejects_header_injection_tokens(tmp_path: Path) -> None:
    result = _run_promotion_attestation(
        tmp_path,
        extra_env={
            "CANDIDATE_BACKEND_OPERATIONS_TOKEN": (
                "Ops_7zQ9-aB3mN8.xK2pL6:sD4wF1cV5\nX-Injected: true"
            ),
        },
    )

    assert result.returncode != 0
    assert "non-whitespace characters" in result.stderr


def test_exclusive_run_lock_rejects_concurrent_mutation(tmp_path: Path) -> None:
    lock_file = tmp_path / "candidate" / ".run.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            str(RUN_LOCK_SCRIPT),
            "--lock-file",
            str(lock_file),
            "--timeout-seconds",
            "1",
            "--",
            sys.executable,
            "-c",
            "import time; print('locked', flush=True); time.sleep(10)",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        contender = subprocess.run(
            [
                sys.executable,
                str(RUN_LOCK_SCRIPT),
                "--lock-file",
                str(lock_file),
                "--timeout-seconds",
                "0.1",
                "--",
                sys.executable,
                "-c",
                "raise SystemExit(0)",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        assert contender.returncode == 75
        assert "Timed out waiting for exclusive run lock" in contender.stderr
    finally:
        holder.terminate()
        holder.wait(timeout=5)


def test_promotion_archive_and_rollback_share_active_pointer_lock() -> None:
    promotion = PROMOTE_SCRIPT.read_text(encoding="utf-8")
    rollback = (PROJECT_ROOT / "scripts" / "deploy" / "rollback-active-release.sh").read_text(
        encoding="utf-8"
    )
    archive = (PROJECT_ROOT / "scripts" / "deploy" / "build-runtime-release-archive.py").read_text(
        encoding="utf-8"
    )

    assert '${ACTIVE_RELEASE_FILE}.lock' in promotion
    assert '${ACTIVE_RELEASE_FILE}.lock' in rollback
    assert "active_release_file.name + \".lock\"" in archive
    assert ".rollback.lock" not in rollback
