"""
Dashboard helpers for evaluation workflows and benchmark execution.

This module exposes:
- static asset discovery for eval datasets/gates/mappings
- immediate eval dataset utilities
- long-running evaluation jobs persisted under each run's dashboard reports
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation import (
    DEFAULT_RAGAS_METRICS,
    EVALUATION_PRESETS,
    evaluate_retrieval_dataset,
    evaluate_standard_rankings,
    export_hf_benchmark,
    export_ir_datasets_benchmark,
    generate_answer_predictions,
    load_eval_examples,
    mbzuai_eval_template,
    run_ragas_evaluation,
    run_standard_benchmark_retrieval,
    summarize_eval_examples,
    summarize_standard_benchmark,
    validate_eval_examples,
    write_eval_examples,
)


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


def _project_relative(path: str | Path | None) -> str | None:
    if not path:
        return None
    target = Path(path).resolve()
    try:
        return str(target.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(target)


def _find_benchmark_dirs(root: Path) -> List[str]:
    if not root.exists():
        return []
    results: List[str] = []
    for candidate in root.rglob("*"):
        if not candidate.is_dir():
            continue
        if all((candidate / name).exists() for name in ("corpus.jsonl", "queries.jsonl", "qrels.jsonl")):
            results.append(_project_relative(candidate) or str(candidate))
    return sorted(set(results))


def _find_matching_files(root: Path, pattern: str, *, name_fragments: Iterable[str] | None = None) -> List[str]:
    if not root.exists():
        return []
    fragments = [fragment.lower() for fragment in (name_fragments or []) if fragment]
    results: List[str] = []
    for path in root.rglob(pattern):
        if not path.is_file():
            continue
        if fragments and not any(fragment in path.name.lower() for fragment in fragments):
            continue
        results.append(_project_relative(path) or str(path))
    return sorted(set(results))


def list_eval_assets(work_dir: str | Path | None = None) -> Dict[str, List[str]]:
    datasets_dir = PROJECT_ROOT / "eval" / "mbzuai_gold"
    gates_dir = PROJECT_ROOT / "eval" / "gates"
    mappings_dir = PROJECT_ROOT / "eval" / "benchmarks"
    datasets = sorted(str(path.relative_to(PROJECT_ROOT)) for path in datasets_dir.glob("*.json*")) if datasets_dir.exists() else []
    gates = sorted(str(path.relative_to(PROJECT_ROOT)) for path in gates_dir.glob("*.json")) if gates_dir.exists() else []
    mapping_files = sorted(str(path.relative_to(PROJECT_ROOT)) for path in mappings_dir.glob("*.json")) if mappings_dir.exists() else []

    benchmark_dirs = _find_benchmark_dirs(PROJECT_ROOT / "eval")
    prediction_files: List[str] = []
    rankings_files: List[str] = []
    report_files: List[str] = []
    if work_dir:
        run_root = _resolve_project_path(work_dir)
        benchmark_dirs.extend(_find_benchmark_dirs(run_root / "dashboard_reports"))
        prediction_files.extend(_find_matching_files(run_root / "dashboard_reports", "*.jsonl", name_fragments=["prediction", "answer"]))
        rankings_files.extend(_find_matching_files(run_root / "dashboard_reports", "*.jsonl", name_fragments=["rankings"]))
        report_files.extend(_find_matching_files(run_root / "dashboard_reports", "*.json", name_fragments=["report", "ragas"]))

    return {
        "datasets": sorted(set(datasets)),
        "gates": sorted(set(gates)),
        "mapping_files": sorted(set(mapping_files)),
        "benchmark_dirs": sorted(set(benchmark_dirs)),
        "prediction_files": sorted(set(prediction_files)),
        "rankings_files": sorted(set(rankings_files)),
        "report_files": sorted(set(report_files)),
    }


def list_eval_presets() -> Dict[str, Dict[str, Any]]:
    return dict(EVALUATION_PRESETS)


def init_eval_set(output_path: str | Path, *, force: bool = False) -> Dict[str, Any]:
    resolved = _resolve_project_path(output_path, must_exist=False)
    if resolved.exists() and not force:
        raise ValueError(f"Refusing to overwrite existing file: {resolved}")
    examples = mbzuai_eval_template()
    write_eval_examples(resolved, examples)
    return {
        "output_path": str(resolved),
        "example_count": len(examples),
        "format": resolved.suffix.lower() or ".json",
    }


def summarize_eval_set(dataset_path: str | Path) -> Dict[str, Any]:
    resolved = _resolve_project_path(dataset_path)
    return summarize_eval_examples(load_eval_examples(resolved))


def validate_eval_set(dataset_path: str | Path, *, work_dir: str | Path | None = None) -> Dict[str, Any]:
    resolved_dataset = _resolve_project_path(dataset_path)
    resolved_work_dir = _resolve_project_path(work_dir) if work_dir else None
    return validate_eval_examples(resolved_dataset, work_dir=str(resolved_work_dir) if resolved_work_dir else None)


def summarize_benchmark_dataset(dataset_dir: str | Path) -> Dict[str, Any]:
    resolved = _resolve_project_path(dataset_dir)
    return summarize_standard_benchmark(resolved)


def _job_dir(work_dir: str | Path) -> Path:
    target = Path(work_dir) / "dashboard_reports" / "evaluation_jobs"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _job_manifest_path(work_dir: str | Path, job_id: str) -> Path:
    return _job_dir(work_dir) / f"{job_id}.manifest.json"


def _job_report_path(work_dir: str | Path, job_id: str) -> Path:
    return _job_dir(work_dir) / f"{job_id}.report.json"


def _write_manifest(work_dir: str | Path, job_id: str, payload: Dict[str, Any]) -> None:
    _job_manifest_path(work_dir, job_id).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def list_evaluation_jobs(work_dir: str | Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in sorted(_job_dir(work_dir).glob("*.manifest.json"), reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["manifest_path"] = str(path)
        items.append(payload)
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items


def list_benchmark_jobs(work_dir: str | Path) -> List[Dict[str, Any]]:
    return [item for item in list_evaluation_jobs(work_dir) if item.get("job_type") == "retrieval_benchmark"]


def cleanup_stale_evaluation_jobs() -> int:
    repaired = 0
    for path in PROJECT_ROOT.glob("runs/**/dashboard_reports/evaluation_jobs/*.manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") not in {"queued", "running"}:
            continue
        payload["status"] = "cancelled"
        payload["finished_at"] = _utcnow_iso()
        payload["error_message"] = payload.get("error_message") or "Dashboard backend restarted before the evaluation job completed."
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        repaired += 1
    return repaired


def _write_report(path: Path, payload: Dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(path)


class EvaluationJobManager:
    def __init__(self) -> None:
        self._tasks: dict[str, asyncio.Task] = {}

    def shutdown(self) -> None:
        for job_id, task in list(self._tasks.items()):
            if task.done():
                continue
            task.cancel()
            logger.info("Cancelled dashboard evaluation job %s during shutdown", job_id)

    def cancel(self, work_dir: str | Path, job_id: str) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        manifest_file = _job_manifest_path(resolved_work_dir, job_id)
        if not manifest_file.exists():
            raise FileNotFoundError(f"Unknown evaluation job: {job_id}")

        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        task = self._tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            manifest["status"] = "cancelled"
            manifest["finished_at"] = _utcnow_iso()
            manifest["error_message"] = "Dashboard evaluation job cancellation requested."
            _write_manifest(resolved_work_dir, job_id, manifest)
            return manifest

        if manifest.get("status") in {"queued", "running"}:
            manifest["status"] = "cancelled"
            manifest["finished_at"] = _utcnow_iso()
            manifest["error_message"] = manifest.get("error_message") or "Dashboard backend restarted before the evaluation job completed."
            _write_manifest(resolved_work_dir, job_id, manifest)
        return manifest

    def _start_job(
        self,
        *,
        work_dir: Path,
        job_type: str,
        run_id: int,
        config_name: str | None,
        payload: Dict[str, Any],
        runner: Callable[[], Dict[str, Any]],
        on_success: Callable[[Dict[str, Any], Dict[str, Any]], None] | None = None,
    ) -> Dict[str, Any]:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        job_id = f"{job_type}-{timestamp}"
        manifest: Dict[str, Any] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "queued",
            "run_id": run_id,
            "config_name": config_name,
            "work_dir": str(work_dir),
            "created_at": _utcnow_iso(),
            "started_at": None,
            "finished_at": None,
            "error_message": None,
            **payload,
        }
        _write_manifest(work_dir, job_id, manifest)
        self._tasks[job_id] = asyncio.create_task(
            self._run_job(
                job_id=job_id,
                work_dir=work_dir,
                manifest=manifest,
                runner=runner,
                on_success=on_success,
            )
        )
        return manifest

    async def _run_job(
        self,
        *,
        job_id: str,
        work_dir: Path,
        manifest: Dict[str, Any],
        runner: Callable[[], Dict[str, Any]],
        on_success: Callable[[Dict[str, Any], Dict[str, Any]], None] | None,
    ) -> None:
        manifest["status"] = "running"
        manifest["started_at"] = _utcnow_iso()
        _write_manifest(work_dir, job_id, manifest)
        try:
            result = await asyncio.to_thread(runner)
            if on_success:
                on_success(manifest, result)
            manifest["status"] = "completed"
            manifest["finished_at"] = _utcnow_iso()
        except asyncio.CancelledError:
            manifest["status"] = "cancelled"
            manifest["finished_at"] = _utcnow_iso()
            manifest["error_message"] = "Dashboard evaluation job was cancelled."
            raise
        except Exception as exc:
            logger.exception("Dashboard evaluation job %s failed", job_id)
            manifest["status"] = "failed"
            manifest["finished_at"] = _utcnow_iso()
            manifest["error_message"] = str(exc)
        finally:
            _write_manifest(work_dir, job_id, manifest)
            self._tasks.pop(job_id, None)

    def start_retrieval_benchmark(
        self,
        *,
        run_id: int,
        config_name: str,
        work_dir: str | Path,
        dataset_path: str | Path,
        gates_path: str | Path | None,
        parallelism: int,
    ) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        resolved_dataset = _resolve_project_path(dataset_path)
        resolved_gates = _resolve_project_path(gates_path) if gates_path else None
        report_path = _job_report_path(resolved_work_dir, f"retrieval-benchmark-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")

        def runner() -> Dict[str, Any]:
            return evaluate_retrieval_dataset(
                config_name=config_name,
                work_dir=str(resolved_work_dir),
                dataset_path=str(resolved_dataset),
                gates_path=str(resolved_gates) if resolved_gates else None,
                parallelism=int(parallelism),
            )

        def on_success(manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
            actual_report_path = _job_report_path(resolved_work_dir, manifest["job_id"])
            manifest["report_path"] = _write_report(actual_report_path, result)
            manifest["query_count"] = result.get("query_count")
            manifest["overall"] = result.get("overall")
            manifest["gates"] = result.get("gates")

        return self._start_job(
            work_dir=resolved_work_dir,
            job_type="retrieval_benchmark",
            run_id=run_id,
            config_name=config_name,
            payload={
                "dataset_path": str(resolved_dataset),
                "gates_path": str(resolved_gates) if resolved_gates else None,
                "parallelism": int(parallelism),
                "report_path": str(report_path),
            },
            runner=runner,
            on_success=on_success,
        )

    def start_answer_generation(
        self,
        *,
        run_id: int,
        config_name: str,
        work_dir: str | Path,
        dataset_path: str | Path,
        output_path: str | Path,
        model: str,
    ) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        resolved_dataset = _resolve_project_path(dataset_path)
        resolved_output = _resolve_project_path(output_path, must_exist=False)

        def runner() -> Dict[str, Any]:
            return generate_answer_predictions(
                config_name=config_name,
                work_dir=str(resolved_work_dir),
                dataset_path=str(resolved_dataset),
                output_path=str(resolved_output),
                model=model,
            )

        def on_success(manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
            manifest["output_path"] = result.get("output_path")
            manifest["row_count"] = result.get("row_count")
            manifest["model"] = result.get("model")

        return self._start_job(
            work_dir=resolved_work_dir,
            job_type="answer_generation",
            run_id=run_id,
            config_name=config_name,
            payload={
                "dataset_path": str(resolved_dataset),
                "output_path": str(resolved_output),
                "model": model,
            },
            runner=runner,
            on_success=on_success,
        )

    def start_ragas(
        self,
        *,
        run_id: int,
        work_dir: str | Path,
        predictions_path: str | Path,
        metric_names: Iterable[str] | None,
        llm_model: str,
        embedding_model: str,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        resolved_predictions = _resolve_project_path(predictions_path)
        resolved_output = _resolve_project_path(output_path, must_exist=False) if output_path else None
        metric_names_list = [str(name).strip() for name in (metric_names or DEFAULT_RAGAS_METRICS) if str(name).strip()]

        def runner() -> Dict[str, Any]:
            result = run_ragas_evaluation(
                predictions_path=str(resolved_predictions),
                metric_names=metric_names_list,
                llm_model=llm_model,
                embedding_model=embedding_model,
            )
            return result

        def on_success(manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
            report_path = resolved_output or _job_report_path(resolved_work_dir, manifest["job_id"])
            manifest["report_path"] = _write_report(Path(report_path), result)
            manifest["row_count"] = result.get("row_count")
            manifest["metrics"] = result.get("metrics")

        return self._start_job(
            work_dir=resolved_work_dir,
            job_type="ragas",
            run_id=run_id,
            config_name=None,
            payload={
                "predictions_path": str(resolved_predictions),
                "metric_names": metric_names_list,
                "llm_model": llm_model,
                "embedding_model": embedding_model,
                "report_path": str(resolved_output) if resolved_output else None,
            },
            runner=runner,
            on_success=on_success,
        )

    def start_export_ir_benchmark(
        self,
        *,
        run_id: int,
        work_dir: str | Path,
        dataset_id: str,
        output_dir: str | Path,
        max_queries: int | None,
        max_docs: int | None,
        full_corpus: bool,
    ) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        resolved_output = _resolve_project_path(output_dir, must_exist=False)

        def runner() -> Dict[str, Any]:
            return export_ir_datasets_benchmark(
                dataset_id=dataset_id,
                output_dir=str(resolved_output),
                max_queries=max_queries,
                max_docs=max_docs,
                full_corpus=full_corpus,
            )

        def on_success(manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
            manifest["output_dir"] = str(resolved_output)
            manifest["metadata"] = result

        return self._start_job(
            work_dir=resolved_work_dir,
            job_type="export_ir_benchmark",
            run_id=run_id,
            config_name=None,
            payload={
                "dataset_id": dataset_id,
                "output_dir": str(resolved_output),
                "max_queries": max_queries,
                "max_docs": max_docs,
                "full_corpus": bool(full_corpus),
            },
            runner=runner,
            on_success=on_success,
        )

    def start_export_hf_benchmark(
        self,
        *,
        run_id: int,
        work_dir: str | Path,
        mapping_path: str | Path,
        output_dir: str | Path,
        max_queries: int | None,
        max_docs: int | None,
        max_qrels: int | None,
    ) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        resolved_mapping = _resolve_project_path(mapping_path)
        resolved_output = _resolve_project_path(output_dir, must_exist=False)

        def runner() -> Dict[str, Any]:
            return export_hf_benchmark(
                mapping_path=str(resolved_mapping),
                output_dir=str(resolved_output),
                max_queries=max_queries,
                max_docs=max_docs,
                max_qrels=max_qrels,
            )

        def on_success(manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
            manifest["output_dir"] = str(resolved_output)
            manifest["metadata"] = result

        return self._start_job(
            work_dir=resolved_work_dir,
            job_type="export_hf_benchmark",
            run_id=run_id,
            config_name=None,
            payload={
                "mapping_path": str(resolved_mapping),
                "output_dir": str(resolved_output),
                "max_queries": max_queries,
                "max_docs": max_docs,
                "max_qrels": max_qrels,
            },
            runner=runner,
            on_success=on_success,
        )

    def start_standard_benchmark_retrieval(
        self,
        *,
        run_id: int,
        config_name: str,
        work_dir: str | Path,
        dataset_dir: str | Path,
        output_rankings_path: str | Path,
        top_k: int,
        dense_top_k: int,
        sparse_top_k: int,
        rrf_k: int,
        batch_size: int,
        doc_cache_path: str | Path | None,
        query_cache_path: str | Path | None,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        resolved_dataset = _resolve_project_path(dataset_dir)
        resolved_rankings = _resolve_project_path(output_rankings_path, must_exist=False)
        resolved_doc_cache = _resolve_project_path(doc_cache_path, must_exist=False) if doc_cache_path else None
        resolved_query_cache = _resolve_project_path(query_cache_path, must_exist=False) if query_cache_path else None
        resolved_output = _resolve_project_path(output_path, must_exist=False) if output_path else None

        def runner() -> Dict[str, Any]:
            return run_standard_benchmark_retrieval(
                config_name=config_name,
                dataset_dir=str(resolved_dataset),
                output_rankings_path=str(resolved_rankings),
                top_k=int(top_k),
                dense_top_k=int(dense_top_k),
                sparse_top_k=int(sparse_top_k),
                rrf_k=int(rrf_k),
                doc_cache_path=str(resolved_doc_cache) if resolved_doc_cache else None,
                query_cache_path=str(resolved_query_cache) if resolved_query_cache else None,
                batch_size=int(batch_size),
            )

        def on_success(manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
            report_path = resolved_output or _job_report_path(resolved_work_dir, manifest["job_id"])
            manifest["report_path"] = _write_report(Path(report_path), result)
            manifest["rankings_path"] = result.get("rankings_path")
            manifest["overall"] = result.get("overall")

        return self._start_job(
            work_dir=resolved_work_dir,
            job_type="standard_benchmark_retrieval",
            run_id=run_id,
            config_name=config_name,
            payload={
                "dataset_dir": str(resolved_dataset),
                "output_rankings_path": str(resolved_rankings),
                "report_path": str(resolved_output) if resolved_output else None,
                "top_k": int(top_k),
                "dense_top_k": int(dense_top_k),
                "sparse_top_k": int(sparse_top_k),
                "rrf_k": int(rrf_k),
                "batch_size": int(batch_size),
                "doc_cache_path": str(resolved_doc_cache) if resolved_doc_cache else None,
                "query_cache_path": str(resolved_query_cache) if resolved_query_cache else None,
            },
            runner=runner,
            on_success=on_success,
        )

    def start_benchmark_rankings_evaluation(
        self,
        *,
        run_id: int,
        work_dir: str | Path,
        dataset_dir: str | Path,
        rankings_path: str | Path,
        k: int,
        output_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        resolved_work_dir = _resolve_project_path(work_dir)
        resolved_dataset = _resolve_project_path(dataset_dir)
        resolved_rankings = _resolve_project_path(rankings_path)
        resolved_output = _resolve_project_path(output_path, must_exist=False) if output_path else None

        def runner() -> Dict[str, Any]:
            return evaluate_standard_rankings(
                dataset_dir=str(resolved_dataset),
                rankings_path=str(resolved_rankings),
                k=int(k),
            )

        def on_success(manifest: Dict[str, Any], result: Dict[str, Any]) -> None:
            report_path = resolved_output or _job_report_path(resolved_work_dir, manifest["job_id"])
            manifest["report_path"] = _write_report(Path(report_path), result)
            manifest["overall"] = result.get("overall")

        return self._start_job(
            work_dir=resolved_work_dir,
            job_type="benchmark_rankings_evaluation",
            run_id=run_id,
            config_name=None,
            payload={
                "dataset_dir": str(resolved_dataset),
                "rankings_path": str(resolved_rankings),
                "k": int(k),
                "report_path": str(resolved_output) if resolved_output else None,
            },
            runner=runner,
            on_success=on_success,
        )


evaluation_job_manager = EvaluationJobManager()


class BenchmarkJobManager:
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
        return evaluation_job_manager.start_retrieval_benchmark(
            run_id=run_id,
            config_name=config_name,
            work_dir=work_dir,
            dataset_path=dataset_path,
            gates_path=gates_path,
            parallelism=parallelism,
        )


benchmark_job_manager = BenchmarkJobManager()
