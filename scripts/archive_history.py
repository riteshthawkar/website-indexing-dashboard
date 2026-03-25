#!/usr/bin/env python3
"""
Archive current run artifacts and reset dashboard tracking.

Moves:
- runs/* -> history/runs/<timestamp>/
- dashboard/dashboard.db* -> history/dashboard/<timestamp>/

Then recreates an empty runs/ directory and a fresh dashboard DB.
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

RUNS_DIR = ROOT_DIR / "runs"
HISTORY_DIR = ROOT_DIR / "history"
DASHBOARD_DIR = ROOT_DIR / "dashboard"
DB_BASENAME = "dashboard.db"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _move_children(source: Path, destination: Path) -> int:
    count = 0
    if not source.exists():
        return count

    destination.mkdir(parents=True, exist_ok=True)
    for child in sorted(source.iterdir()):
        target = destination / child.name
        shutil.move(str(child), str(target))
        count += 1
    return count


def _archive_dashboard_db(destination: Path) -> int:
    count = 0
    destination.mkdir(parents=True, exist_ok=True)
    for candidate in sorted(DASHBOARD_DIR.glob(f"{DB_BASENAME}*")):
        if candidate.is_file():
            shutil.move(str(candidate), str(destination / candidate.name))
            count += 1
    return count


def main() -> int:
    stamp = _timestamp()
    run_archive_dir = HISTORY_DIR / "runs" / stamp
    db_archive_dir = HISTORY_DIR / "dashboard" / stamp

    moved_runs = _move_children(RUNS_DIR, run_archive_dir)
    moved_db_files = _archive_dashboard_db(db_archive_dir)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    from dashboard.database import init_db

    init_db()

    print(f"Archived run directories: {moved_runs}")
    print(f"Archived dashboard DB files: {moved_db_files}")
    print(f"Run history: {run_archive_dir}")
    print(f"Dashboard history: {db_archive_dir}")
    print(f"Fresh runs directory: {RUNS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
