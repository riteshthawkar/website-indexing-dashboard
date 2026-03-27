from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from structured_logs import structured_log_path


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_dir(work_dir: str | Path) -> Path:
    return Path(work_dir) / "dashboard_runtime"


def runtime_state_path(work_dir: str | Path) -> Path:
    return runtime_dir(work_dir) / "worker_state.json"


def worker_stdout_path(work_dir: str | Path) -> Path:
    return runtime_dir(work_dir) / "worker_stdout.log"


def _archive_existing_file(path: Path, archive_dir: Path) -> Optional[Path]:
    if not path.exists():
        return None
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = archive_dir / f"{path.stem}_{stamp}{path.suffix}"
    counter = 1
    while archived.exists():
        archived = archive_dir / f"{path.stem}_{stamp}_{counter}{path.suffix}"
        counter += 1
    path.replace(archived)
    return archived


def rotate_attempt_logs(work_dir: str | Path) -> Dict[str, Optional[str]]:
    work_dir = Path(work_dir)
    archived_structured = _archive_existing_file(
        structured_log_path(work_dir),
        structured_log_path(work_dir).parent / "archive",
    )
    archived_stdout = _archive_existing_file(
        worker_stdout_path(work_dir),
        runtime_dir(work_dir) / "archive",
    )
    return {
        "structured_logs": str(archived_structured) if archived_structured else None,
        "worker_stdout": str(archived_stdout) if archived_stdout else None,
    }


def load_worker_state(work_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = runtime_state_path(work_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def save_worker_state(work_dir: str | Path, payload: Dict[str, Any]) -> Path:
    path = runtime_state_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def pid_is_alive(pid: int | None) -> bool:
    if not pid or int(pid) <= 0:
        return False
    stat_path = Path(f"/proc/{int(pid)}/stat")
    if stat_path.exists():
        try:
            stat = stat_path.read_text(encoding="utf-8")
            state = stat.rsplit(") ", 1)[1].split(" ", 1)[0]
            if state.upper() == "Z":
                return False
        except Exception:
            pass
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def is_worker_active(work_dir: str | Path) -> bool:
    payload = load_worker_state(work_dir)
    if not payload:
        return False
    status = str(payload.get("status") or "").lower()
    pid = payload.get("pid")
    return status in {"starting", "running", "cancelling"} and pid_is_alive(pid)
