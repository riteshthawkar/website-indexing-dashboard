from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_dir(work_dir: str | Path) -> Path:
    return Path(work_dir) / "dashboard_runtime"


def runtime_state_path(work_dir: str | Path) -> Path:
    return runtime_dir(work_dir) / "worker_state.json"


def worker_stdout_path(work_dir: str | Path) -> Path:
    return runtime_dir(work_dir) / "worker_stdout.log"


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
