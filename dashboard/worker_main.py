from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database import Run, get_db
from run_executor import _resolve_work_dir, execute_pipeline
from worker_runtime import save_worker_state, utcnow_iso


def _resolve_run_work_dir(run_id: int) -> Path:
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise RuntimeError(f"Run {run_id} not found")
        if run.work_dir:
            return Path(run.work_dir).resolve()
        return _resolve_work_dir(run.config_name, f"run_{run_id}")
    finally:
        db.close()


async def _async_main(args: argparse.Namespace) -> int:
    run_id = int(args.run_id)
    work_dir = _resolve_run_work_dir(run_id)
    stop_signal = {"name": None}

    save_worker_state(
        work_dir,
        {
            "pid": os.getpid(),
            "run_id": run_id,
            "status": "running",
            "resume": bool(args.resume),
            "restart_from": args.restart_from,
            "started_at": utcnow_iso(),
        },
    )

    task = asyncio.create_task(
        execute_pipeline(
            run_id,
            resume=True if args.resume else None,
            restart_from=args.restart_from,
        )
    )

    def _request_stop(signame: str) -> None:
        stop_signal["name"] = signame
        save_worker_state(
            work_dir,
            {
                "pid": os.getpid(),
                "run_id": run_id,
                "status": "cancelling",
                "resume": bool(args.resume),
                "restart_from": args.restart_from,
                "started_at": utcnow_iso(),
                "signal": signame,
            },
        )
        task.cancel()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, _request_stop, sig.name)
        except NotImplementedError:
            signal.signal(sig, lambda *_args, name=sig.name: _request_stop(name))

    exit_code = 0
    try:
        await task
    except asyncio.CancelledError:
        exit_code = 130
    except Exception:
        exit_code = 1
        raise
    finally:
        save_worker_state(
            work_dir,
            {
                "pid": os.getpid(),
                "run_id": run_id,
                "status": "stopped",
                "resume": bool(args.resume),
                "restart_from": args.restart_from,
                "finished_at": utcnow_iso(),
                "signal": stop_signal["name"],
                "exit_code": exit_code,
            },
        )
    return exit_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Dashboard worker entrypoint for pipeline runs.")
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restart-from", default=None)
    args = parser.parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
