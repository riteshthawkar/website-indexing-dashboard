"""
Dashboard helpers for retrieval benchmark execution and report discovery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation import evaluate_retrieval_dataset


logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_project_path(path_value: str | Path, *, must_exist: bool = True) -> Path:
    raw = Path(path_value)
    resolved = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Path must stay within project root: {path_value}") from exc
    if must_exist and not resolved.exists():
        raise ValueError(f"Path not found: {resolved}")
    return resolved


def list_eval_assets() -> Dict[str, List[str]]:
    datasets_dir = PROJECT_ROOT / "eval" / "mbzuai_gold"
    gates_dir = PROJECT_ROOT / "eval" / "gates"
    datasets = []
    gates = []
    if datasets_dir.exists():
        datasets = sorted(str(path.relative_to(PROJECT_ROOT)) for path in datasets_dir.glob("*.jsonl"))
    if gates_dir.exists():
        gates = sorted(str(path.relative_to(PROJECT_ROOT)) for path in gates_dir.glob("*.json"))
    return {
        "datasets": datasets,
        "gates": gates,
    }


def _benchmark_dir(work_dir: str | Path) -> Path:
    target = Path(work_dir) / "dashboard_reports" / "retrieval_benchmarks"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _benchmark_manifest_path(work_dir: str | Path, job_id: str) -> Path:
    return _benchmark_dir(work_dir) / f"{job_id}.manifest.json"


def _benchmark_report_path(work_dir: str | Path, job_id: str) -> Path:
    return _benchmark_dir(work_dir) / f"{job_id}.report.json"


def _write_manifest(work_dir: str | Path, job_id: str, payload: Dict[str, Any]) -> None:
    path = _benchmark_manifest_path(work_dir, job_id)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def list_benchmark_jobs(work_dir: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in sorted(_benchmark_dir(work_dir).glob("*.manifest.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["manifest_path"] = str(path)
        items.append(payload)
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items


class BenchmarkJobManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}

    def start(
        self,
        *,
        run_id: int,
        config_name: str,
        work_dir: str | Path,
        dataset_path: str | Path,
        gates_path: str | Path | None,
        parallelism: int,
    ) -> Dict[str, Any]:
        resolved_work_dir = Path(work_dir).resolve()
        resolved_dataset = _resolve_project_path(dataset_path)
        resolved_gates = _resolve_project_path(gates_path) if gates_path else None
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"retrieval-{timestamp}"
        created_at = _utcnow_iso()
        manifest = {
            "job_id": job_id,
            "run_id": run_id,
            "job_type": "retrieval_benchmark",
            "status": "queued",
            "config_name": config_name,
            "work_dir": str(resolved_work_dir),
            "dataset_path": str(resolved_dataset),
            "gates_path": str(resolved_gates) if resolved_gates else None,
            "parallelism": int(parallelism),
            "created_at": created_at,
            "started_at": None,
            "finished_at": None,
            "report_path": str(_benchmark_report_path(resolved_work_dir, job_id)),
            "query_count": None,
            "overall": None,
            "gates": None,
            "error_message": None,
        }
        _write_manifest(resolved_work_dir, job_id, manifest)
        task = asyncio.create_task(
            self._run_job(
                job_id=job_id,
                run_id=run_id,
                config_name=config_name,
                work_dir=resolved_work_dir,
                dataset_path=resolved_dataset,
                gates_path=resolved_gates,
                parallelism=int(parallelism),
                created_at=created_at,
            )
        )
        self._tasks[job_id] = task
        return manifest

    async def _run_job(
        self,
        *,
        job_id: str,
        run_id: int,
        config_name: str,
        work_dir: Path,
        dataset_path: Path,
        gates_path: Optional[Path],
        parallelism: int,
        created_at: str,
    ) -> None:
        manifest = {
            "job_id": job_id,
            "run_id": run_id,
            "job_type": "retrieval_benchmark",
            "status": "running",
            "config_name": config_name,
            "work_dir": str(work_dir),
            "dataset_path": str(dataset_path),
            "gates_path": str(gates_path) if gates_path else None,
            "parallelism": parallelism,
            "created_at": created_at,
            "started_at": _utcnow_iso(),
            "finished_at": None,
            "report_path": str(_benchmark_report_path(work_dir, job_id)),
            "query_count": None,
            "overall": None,
            "gates": None,
            "error_message": None,
        }
        _write_manifest(work_dir, job_id, manifest)
        try:
            report = await asyncio.to_thread(
                evaluate_retrieval_dataset,
                config_name=config_name,
                work_dir=str(work_dir),
                dataset_path=str(dataset_path),
                gates_path=str(gates_path) if gates_path else None,
                parallelism=parallelism,
            )
            report_path = _benchmark_report_path(work_dir, job_id)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            manifest.update(
                {
                    "status": "completed",
                    "finished_at": _utcnow_iso(),
                    "query_count": report.get("query_count"),
                    "overall": report.get("overall"),
                    "gates": report.get("gates"),
                }
            )
        except Exception as exc:
            logger.exception("Dashboard retrieval benchmark job %s failed", job_id)
            manifest.update(
                {
                    "status": "failed",
                    "finished_at": _utcnow_iso(),
                    "error_message": str(exc),
                }
            )
        finally:
            _write_manifest(work_dir, job_id, manifest)
            self._tasks.pop(job_id, None)


benchmark_job_manager = BenchmarkJobManager()
