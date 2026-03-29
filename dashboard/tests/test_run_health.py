from __future__ import annotations

import json
from pathlib import Path
import sys
import time


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import _sync_run_health
from database import Run, get_db, utcnow
from run_health import detect_crawler_stall
from structured_logs import load_structured_logs
from worker_runtime import load_worker_state, save_worker_state


def _create_run(tmp_path: Path, *, status: str = "running") -> Run:
    db = get_db()
    try:
        run = Run(
            run_name="pytest-stall-run",
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


def _write_stalled_crawler_state(work_dir: Path, *, updated_at: float) -> None:
    (work_dir / "pipeline_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "current_stage_index": 0,
                "stages": [
                    {
                        "name": "crawl4ai",
                        "stage_type": "crawler",
                        "status": "running",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (work_dir / "crawl_state.json").write_text(
        json.dumps(
            {
                "visited": ["https://example.com", "https://example.com/a"],
                "pending": [
                    {"url": "https://example.com/b", "parent_url": "https://example.com"}
                ],
                "updated_at": updated_at,
            }
        ),
        encoding="utf-8",
    )


def test_detect_crawler_stall_reports_idle_run(tmp_path: Path) -> None:
    now = time.time()
    _write_stalled_crawler_state(tmp_path, updated_at=now - 7200)

    payload = detect_crawler_stall(tmp_path, now=now, timeout_seconds=1800)

    assert payload is not None
    assert payload["stage"] == "crawler/crawl4ai"
    assert payload["visited_count"] == 2
    assert payload["pending_count"] == 1
    assert "no crawl progress" in payload["reason"]


def test_detect_crawler_stall_uses_faster_threshold_when_frontier_is_empty(tmp_path: Path) -> None:
    now = time.time()
    _write_stalled_crawler_state(tmp_path, updated_at=now - 400)
    payload = json.loads((tmp_path / "crawl_state.json").read_text(encoding="utf-8"))
    payload["pending"] = []
    (tmp_path / "crawl_state.json").write_text(json.dumps(payload), encoding="utf-8")
    payload = detect_crawler_stall(tmp_path, now=now, timeout_seconds=1800)
    assert payload is not None
    assert payload["pending_count"] == 0


def test_sync_run_health_marks_stalled_worker_failed(monkeypatch, tmp_path: Path) -> None:
    run = _create_run(tmp_path, status="running")
    _write_stalled_crawler_state(tmp_path, updated_at=time.time() - 7200)
    save_worker_state(
        tmp_path,
        {
            "pid": 424242,
            "run_id": run.id,
            "status": "running",
            "started_at": utcnow().isoformat(),
        },
    )
    kill_calls: list[tuple[int, int]] = []

    monkeypatch.setattr("app.pid_is_alive", lambda pid: True)
    monkeypatch.setattr("app.is_worker_active", lambda work_dir: True)
    monkeypatch.setattr("app.os.killpg", lambda pid, sig: kill_calls.append((pid, sig)))

    try:
        refreshed = _sync_run_health(run)
        assert refreshed.status == "failed"
        assert "Crawler stalled" in (refreshed.error_message or "")
        assert kill_calls

        state_payload = json.loads((tmp_path / "pipeline_state.json").read_text(encoding="utf-8"))
        assert state_payload["status"] == "failed"
        assert state_payload["stages"][0]["status"] == "failed"
        assert "Crawler stalled" in (state_payload["stages"][0].get("error_message") or "")

        worker_state = load_worker_state(tmp_path)
        assert worker_state is not None
        assert worker_state["status"] == "stopped"
        assert worker_state["exit_code"] == -9

        structured = load_structured_logs(tmp_path, limit=10)
        assert structured["items"]
        assert structured["items"][-1]["event_type"] == "stage_stalled"
    finally:
        _delete_run(run.id)
