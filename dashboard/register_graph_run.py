"""Register an existing artifact folder in the local dashboard, without scanning it.

Usage: python dashboard/register_graph_run.py /absolute/path/to/completed/run
Only dashboard metadata is written. No pipeline state or source data is changed.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from database import Run, get_db, init_db
from lawa_rag.graphs import discover_graphs


def register(work_dir: Path) -> int:
    root = work_dir.expanduser().resolve()
    if not discover_graphs(root):
        raise ValueError("Folder has no supported graph snapshots")
    init_db()
    db = get_db()
    try:
        existing = db.query(Run).filter(Run.work_dir == str(root)).first()
        if existing:
            return existing.id
        run = Run(
            run_name=f"Graph snapshots · {root.name}",
            run_type="graph_preview",
            config_name="default",
            status="completed",
            is_imported=True,
            work_dir=str(root),
        )
        db.add(run)
        db.commit()
        return run.id
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    args = parser.parse_args()
    try:
        run_id = register(args.work_dir)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"Registered graph preview (not a pipeline execution): http://127.0.0.1:8050/projects/detail?id={run_id}&tab=graphs"
    )


if __name__ == "__main__":
    main()
