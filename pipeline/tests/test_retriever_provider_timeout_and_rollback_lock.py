from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLLBACK_SCRIPT = PROJECT_ROOT / "scripts" / "deploy" / "rollback-active-release.sh"
RUN_LOCK_SCRIPT = PROJECT_ROOT / "scripts" / "deploy" / "with-run-lock.py"


def test_retriever_gemini_client_applies_and_caches_provider_timeout(monkeypatch) -> None:
    from pipeline.retrieval import adaptive_hybrid as module

    clients: list[dict[str, object]] = []

    class FakeHttpOptions:
        def __init__(self, *, timeout: int):
            self.timeout = timeout

    class FakeTypes:
        HttpOptions = FakeHttpOptions

    class FakeGenAI:
        class Client:
            def __init__(self, **kwargs):
                clients.append(kwargs)

    monkeypatch.setenv("GOOGLE_API_KEY", "unit-test-key")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(module, "import_genai", lambda: FakeGenAI)
    monkeypatch.setattr(module, "import_genai_types", lambda: FakeTypes)
    monkeypatch.setattr(module, "_GEMINI_CLIENT_STATE", threading.local())

    first = module._make_gemini_client(request_timeout_ms=12_345)
    repeated = module._make_gemini_client(request_timeout_ms=12_345)
    changed = module._make_gemini_client(request_timeout_ms=54_321)

    assert first is repeated
    assert changed is not first
    assert [entry["http_options"].timeout for entry in clients] == [12_345, 54_321]
    assert all(entry["api_key"] == "unit-test-key" for entry in clients)


def test_adaptive_retriever_passes_configured_timeout_to_embedding(monkeypatch) -> None:
    from pipeline.retrieval import adaptive_hybrid as module

    captured: dict[str, object] = {}

    def fake_embed_query(query: str, **kwargs):
        captured["query"] = query
        captured.update(kwargs)
        return [0.1, 0.2]

    monkeypatch.setattr(module, "_embed_query", fake_embed_query)
    retriever = object.__new__(module.AdaptiveHybridRetriever)
    retriever.model = "gemini-embedding-2"
    retriever.output_dimensionality = 1_536
    retriever.gemini_request_timeout_ms = 87_654

    assert retriever.embed_query("admissions") == [0.1, 0.2]
    assert captured["request_timeout_ms"] == 87_654


def test_stable_gemini_embedding_uses_documented_asymmetric_query_instruction(monkeypatch) -> None:
    from pipeline.retrieval import adaptive_hybrid as module

    captured: dict[str, object] = {}

    class FakeModels:
        def embed_content(self, **kwargs):
            captured.update(kwargs)
            return type(
                "Response",
                (),
                {"embeddings": [type("Embedding", (), {"values": [0.1, 0.2]})()]},
            )()

    monkeypatch.setattr(
        module,
        "_make_gemini_client",
        lambda **_kwargs: type("Client", (), {"models": FakeModels()})(),
    )
    monkeypatch.setattr(module, "_QUERY_EMBEDDING_RETRIES", 0)

    vector = module._embed_query(
        "MBZUAI admissions",
        model="gemini-embedding-2",
        output_dimensionality=1_536,
        request_timeout_ms=30_000,
    )

    assert vector == [0.1, 0.2]
    assert captured["contents"] == "task: search result | query: MBZUAI admissions"


def test_rollback_contends_on_target_run_lock_before_pointer_change(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    target_run = runs_root / "target-run"
    target_run.mkdir(parents=True)
    active_file = tmp_path / "active_release.json"
    initial_pointer = {"schema_version": 1, "run_id": "current-run"}
    active_file.write_text(json.dumps(initial_pointer), encoding="utf-8")
    lock_file = target_run / ".run.lock"

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
            "import time; print('locked', flush=True); time.sleep(30)",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "locked"
        result = subprocess.run(
            ["bash", str(ROLLBACK_SCRIPT)],
            cwd=PROJECT_ROOT,
            env={
                **os.environ,
                "ACTIVE_RELEASE_FILE": str(active_file),
                "RELEASE_RUNS_ROOT": str(runs_root),
                "RELEASE_STORAGE_MODE": "persistent",
                "RELEASE_LOCK_TIMEOUT_SECONDS": "0.1",
                "ROLLBACK_RUN_ID": "target-run",
            },
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    finally:
        holder.terminate()
        holder.wait(timeout=5)

    assert result.returncode == 75
    assert "Timed out waiting for exclusive run lock" in result.stderr
    assert json.loads(active_file.read_text(encoding="utf-8")) == initial_pointer


def test_rollback_lock_order_matches_promotion() -> None:
    promotion = (PROJECT_ROOT / "scripts" / "deploy" / "release-check-promote.sh").read_text(
        encoding="utf-8"
    )
    rollback = ROLLBACK_SCRIPT.read_text(encoding="utf-8")

    for script in (promotion, rollback):
        pointer_lock = script.index('${ACTIVE_RELEASE_FILE}.lock')
        run_lock = script.index("/.run.lock")
        assert pointer_lock < run_lock
