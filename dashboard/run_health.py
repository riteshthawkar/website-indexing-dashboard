from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_CRAWLER_STALL_TIMEOUT_SECONDS = int(
    os.environ.get("DASHBOARD_CRAWLER_STALL_TIMEOUT_SECONDS", "900")
)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def detect_crawler_stall(
    work_dir: str | Path,
    *,
    now: Optional[float] = None,
    timeout_seconds: int = DEFAULT_CRAWLER_STALL_TIMEOUT_SECONDS,
) -> Optional[Dict[str, Any]]:
    work_dir = Path(work_dir)
    pipeline_state = _load_json(work_dir / "pipeline_state.json")
    if not pipeline_state or str(pipeline_state.get("status")) != "running":
        return None

    current_stage_index = pipeline_state.get("current_stage_index")
    stages = pipeline_state.get("stages") or []
    try:
        current_stage = stages[int(current_stage_index)]
    except Exception:
        return None

    stage_type = str(current_stage.get("stage_type") or "")
    stage_name = str(current_stage.get("name") or "")
    if stage_type != "crawler" or stage_name != "crawl4ai":
        return None

    crawl_state_path = work_dir / "crawl_state.json"
    crawl_state = _load_json(crawl_state_path)
    if not crawl_state:
        return None

    updated_at = crawl_state.get("updated_at")
    if updated_at is None:
        try:
            updated_at = crawl_state_path.stat().st_mtime
        except FileNotFoundError:
            return None

    try:
        updated_at = float(updated_at)
    except (TypeError, ValueError):
        return None

    clock = float(now if now is not None else time.time())
    idle_seconds = max(0.0, clock - updated_at)

    visited = crawl_state.get("visited") or []
    pending = crawl_state.get("pending") or []
    pending_count = len(pending) if isinstance(pending, list) else 0
    visited_count = len(visited) if isinstance(visited, list) else 0
    effective_timeout = max(60, int(timeout_seconds))
    if pending_count == 0 and visited_count > 0:
        effective_timeout = min(effective_timeout, 300)
    if idle_seconds < effective_timeout:
        return None
    return {
        "stage": "crawler/crawl4ai",
        "reason": (
            f"Crawler stalled: no crawl progress for {int(idle_seconds)}s "
            f"(visited={visited_count}, pending={pending_count})"
        ),
        "idle_seconds": idle_seconds,
        "visited_count": visited_count,
        "pending_count": pending_count,
        "crawl_state_updated_at": updated_at,
    }
