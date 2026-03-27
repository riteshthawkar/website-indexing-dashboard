from __future__ import annotations

import asyncio
from pathlib import Path
import signal
import sys

from fastapi.testclient import TestClient


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import RunManager, app
from database import Run, get_db, utcnow
from run_executor import create_run
from structured_logs import append_structured_log, make_structured_log_record
from worker_runtime import load_worker_state, pid_is_alive, save_worker_state


def _create_temp_run(tmp_path: Path, *, status: str = "pending") -> Run:
    db = get_db()
    try:
        run = Run(
            run_name="pytest-worker-isolation",
            run_type="full",
            config_name="default",
            status=status,
            work_dir=str(tmp_path),
            created_at=utcnow(),
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        db.expunge(run)
        return run
    finally:
        db.close()


def _delete_run(run_id: int) -> None:
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if run:
            db.delete(run)
            db.commit()
    finally:
        db.close()


def test_run_manager_start_spawns_detached_worker(monkeypatch, tmp_path: Path) -> None:
    run = _create_temp_run(tmp_path)
    manager = RunManager()
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 424242

    def _fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return _FakeProcess()

    monkeypatch.setattr("app.subprocess.Popen", _fake_popen)

    try:
        asyncio.run(manager.start(run.id))
        worker_state = load_worker_state(tmp_path)
        assert worker_state is not None
        assert worker_state["status"] == "starting"
        assert worker_state["pid"] == 424242
        assert "--run-id" in captured["command"]
        assert str(run.id) in captured["command"]
        assert captured["kwargs"]["start_new_session"] is True
        db = get_db()
        try:
            refreshed = db.get(Run, run.id)
            assert refreshed is not None
            assert refreshed.status == "running"
            assert refreshed.work_dir == str(tmp_path)
        finally:
            db.close()
    finally:
        _delete_run(run.id)


def test_create_run_assigns_unique_work_dir() -> None:
    payload = create_run("MBZUAI Latest Run", "default", "full", "https://mbzuai.ac.ae/")
    run_id = payload["id"]
    try:
        work_dir = Path(payload["work_dir"])
        assert work_dir.name != f"run_{run_id}"
        assert "mbzuai-latest-run" in work_dir.name
        assert work_dir.parent.name == "default"
    finally:
        _delete_run(run_id)


def test_run_logs_websocket_streams_records_from_file(tmp_path: Path) -> None:
    run = _create_temp_run(tmp_path, status="pending")
    append_structured_log(
        tmp_path,
        make_structured_log_record(
            sequence=1,
            run_id=run.id,
            pipeline_run_id=f"run_{run.id}",
            level="info",
            event_type="log",
            message="worker log line",
            stage="crawler/crawl4ai",
        ),
    )

    try:
        with TestClient(app) as client:
            with client.websocket_connect(f"/ws/runs/{run.id}/logs") as websocket:
                websocket.send_text("ping")
                pong = websocket.receive_json()
                assert pong["type"] == "pong"
                payload = websocket.receive_json()
                assert payload["type"] == "log"
                assert payload["record"]["message"] == "worker log line"
                assert payload["record"]["sequence"] == 1
    finally:
        _delete_run(run.id)


def test_run_manager_forces_stuck_worker_shutdown(monkeypatch, tmp_path: Path) -> None:
    run = _create_temp_run(tmp_path, status="running")
    manager = RunManager()
    save_worker_state(
        tmp_path,
        {
            "pid": 424243,
            "run_id": run.id,
            "status": "cancelling",
            "started_at": utcnow().isoformat(),
        },
    )
    kill_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("app.pid_is_alive", lambda pid: True)
    monkeypatch.setattr("app.os.killpg", lambda pid, sig: kill_calls.append((pid, sig)))

    try:
        asyncio.run(manager._wait_for_exit_and_finalize(run.id, tmp_path, 424243, grace_seconds=0.01))
        worker_state = load_worker_state(tmp_path)
        assert worker_state is not None
        assert worker_state["status"] == "stopped"
        assert worker_state["signal"] == "SIGKILL"
        assert worker_state["exit_code"] == -9
        assert kill_calls[-1] == (424243, signal.SIGKILL)
    finally:
        _delete_run(run.id)


def test_run_manager_fresh_start_does_not_reuse_existing_pipeline_state(monkeypatch, tmp_path: Path) -> None:
    run = _create_temp_run(tmp_path, status="pending")
    manager = RunManager()
    (tmp_path / "pipeline_state.json").write_text("{}", encoding="utf-8")

    class _FakeProcess:
        pid = 515151

    monkeypatch.setattr("app.subprocess.Popen", lambda *args, **kwargs: _FakeProcess())

    try:
        asyncio.run(manager.start(run.id, resume=False))
        db = get_db()
        try:
            refreshed = db.get(Run, run.id)
            assert refreshed is not None
            assert refreshed.work_dir != str(tmp_path)
            assert "__fresh_" in refreshed.work_dir
        finally:
            db.close()
    finally:
        _delete_run(run.id)


def test_pid_is_alive_treats_zombies_as_dead(monkeypatch) -> None:
    def _fake_exists(self) -> bool:
        return str(self) == "/proc/999999/stat"

    def _fake_read_text(self, encoding="utf-8") -> str:
        return "999999 (python) Z 1 1 1 0 -1 4228100 0 0 0 0 0 0 0 0 20 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

    monkeypatch.setattr("worker_runtime.Path.exists", _fake_exists)
    monkeypatch.setattr("worker_runtime.Path.read_text", _fake_read_text)
    monkeypatch.setattr("worker_runtime.os.kill", lambda pid, sig: None)

    assert pid_is_alive(999999) is False
