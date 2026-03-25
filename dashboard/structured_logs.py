from __future__ import annotations

import json
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


_WRITE_LOCK = threading.Lock()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def structured_log_dir(work_dir: str | Path) -> Path:
    return Path(work_dir) / "dashboard_logs"


def structured_log_path(work_dir: str | Path) -> Path:
    return structured_log_dir(work_dir) / "structured_logs.jsonl"


def make_structured_log_record(
    *,
    sequence: int,
    run_id: int,
    pipeline_run_id: str,
    level: str,
    event_type: str,
    message: str,
    stage: str | None = None,
    data: Optional[Dict[str, Any]] = None,
    created_at: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "sequence": int(sequence),
        "run_id": int(run_id),
        "pipeline_run_id": str(pipeline_run_id),
        "created_at": created_at or utcnow_iso(),
        "level": str(level or "info"),
        "event_type": str(event_type or "log"),
        "stage": str(stage) if stage else None,
        "message": str(message or ""),
        "data": dict(data or {}),
    }


def append_structured_log(work_dir: str | Path, record: Dict[str, Any]) -> Path:
    path = structured_log_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
    return path


def _iter_structured_log_records(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def load_structured_logs(
    work_dir: str | Path,
    *,
    tail: int = 200,
    stage: str | None = None,
    event_type: str | None = None,
    level: str | None = None,
) -> list[Dict[str, Any]]:
    path = structured_log_path(work_dir)
    if not path.exists():
        return []

    window = deque(maxlen=max(int(tail or 200), 1))
    for record in _iter_structured_log_records(path):
        if stage and record.get("stage") != stage:
            continue
        if event_type and record.get("event_type") != event_type:
            continue
        if level and record.get("level") != level:
            continue
        window.append(record)
    return list(window)
