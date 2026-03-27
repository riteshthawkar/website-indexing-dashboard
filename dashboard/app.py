"""
MBZUAI Vectorstore Pipeline Dashboard — FastAPI Application.

Pure REST API + WebSocket for the Next.js frontend.
Stage details come from pipeline_state.json; the DB stores run metadata.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from artifact_ops import list_artifacts, list_run_files, read_run_file
from database import init_db, get_db, Run, RunLog
from run_data import collect_artifact_summary, collect_media_summary, load_run_media
from scanner import scan_and_import
from run_executor import (
    _load_env,
    create_run,
    execute_pipeline,
    cancel_run,
    get_run_logs,
    get_pipeline_state,
    get_available_configs,
    cleanup_stale_runs,
)
from config_manager import (
    load_config,
    save_config,
    get_config_schema,
    list_configs,
)
from control_ops import (
    audit_run_sync,
    dry_run_config_sync,
    list_eval_presets,
    validate_config_sync,
)
from pinecone_ops import (
    fetch_all_index_stats,
    fetch_index_stats,
    snapshot_indexes,
    get_snapshot_history,
)
from evaluation_ops import (
    benchmark_job_manager,
    cleanup_stale_evaluation_jobs,
    evaluation_job_manager,
    init_eval_set,
    list_benchmark_jobs,
    list_eval_assets,
    list_evaluation_jobs,
    summarize_benchmark_dataset,
    summarize_eval_set,
    validate_eval_set,
)
from knowledge_ops import browse_assertions, delete_vectors_for_source, get_run_knowledge_status
from retrieval_ops import run_retrieval_query
from retriever_service_ops import (
    get_retriever_service_status,
    start_retriever_service,
    stop_retriever_service,
)
from structured_logs import load_structured_logs, structured_log_path
from url_manager import (
    add_excluded_subdomain,
    add_target_url,
    get_all_urls_summary,
    get_indexed_urls_by_name,
    get_indexed_urls_for_run,
    get_target_urls,
    get_urls_detail,
    get_urls_detail_by_name,
    remove_excluded_subdomain,
    remove_target_domain,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


# ---------------------------------------------------------------------------
# Run Manager — tracks active tasks and WebSocket connections
# ---------------------------------------------------------------------------

class RunManager:
    def __init__(self):
        self._tasks: dict[int, asyncio.Task] = {}
        self._ws: dict[int, list[WebSocket]] = {}

    async def start(self, run_id: int, *, resume: Optional[bool] = None, restart_from: Optional[str] = None):
        if run_id in self._tasks and not self._tasks[run_id].done():
            raise HTTPException(status_code=400, detail="Pipeline is already running")

        task = asyncio.create_task(
            execute_pipeline(
                run_id,
                on_log=self._broadcast_log,
                on_stage_event=self._broadcast_stage,
                resume=resume,
                restart_from=restart_from,
            )
        )
        self._tasks[run_id] = task

    async def cancel(self, run_id: int):
        task = self._tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        cancel_run(run_id)

    def add_ws(self, run_id: int, ws: WebSocket):
        self._ws.setdefault(run_id, []).append(ws)

    def remove_ws(self, run_id: int, ws: WebSocket):
        if run_id in self._ws:
            self._ws[run_id] = [w for w in self._ws[run_id] if w != ws]

    async def _broadcast_log(self, run_id: int, level: str, stage: str, message: str, record: dict | None = None):
        """Broadcast log line to all WebSocket clients for this run."""
        payload = json.dumps({
            "type": "log",
            "level": level,
            "stage": stage,
            "message": message,
            "record": record,
        })
        await self._broadcast(run_id, payload)

    async def _broadcast_stage(self, run_id: int, event: str, stage_key: str, info: dict, record: dict | None = None):
        """Broadcast stage event to all WebSocket clients."""
        payload = json.dumps({
            "type": "stage",
            "event": event,
            "stage": stage_key,
            "info": info or {},
            "record": record,
        })
        await self._broadcast(run_id, payload)

    async def _broadcast(self, run_id: int, payload: str):
        ws_list = self._ws.get(run_id, [])
        dead = []
        for ws in ws_list:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            ws_list.remove(ws)

    def shutdown(self):
        for run_id, task in self._tasks.items():
            if not task.done():
                task.cancel()
                logger.info(f"Cancelled pipeline task for run {run_id}")


run_manager = RunManager()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_env()
    init_db()
    cleanup_stale_runs()
    repaired_eval_jobs = cleanup_stale_evaluation_jobs()
    if repaired_eval_jobs:
        logger.info("Repaired %s stale evaluation jobs on startup", repaired_eval_jobs)
    imported = scan_and_import()
    if imported:
        logger.info(f"Imported {imported} runs on startup")
    yield
    run_manager.shutdown()
    evaluation_job_manager.shutdown()


app = FastAPI(title="MBZUAI Pipeline Dashboard", lifespan=lifespan)

# CORS for dev mode (Next.js on :3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://0.0.0.0:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Next.js static export
UI_DIR = BASE_DIR.parent / "dashboard-ui" / "out"
if UI_DIR.is_dir():
    app.mount("/_next", StaticFiles(directory=str(UI_DIR / "_next")), name="nextjs")


# ---------------------------------------------------------------------------
# API: Runs
# ---------------------------------------------------------------------------

@app.get("/api/runs")
async def api_list_runs(status: Optional[str] = None, limit: int = 50):
    db = get_db()
    try:
        q = db.query(Run).order_by(Run.created_at.desc())
        if status:
            q = q.filter(Run.status == status)
        return [r.to_dict() for r in q.limit(limit).all()]
    finally:
        db.close()


@app.get("/api/runs/{run_id}")
async def api_get_run(run_id: int):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        result = run.to_dict(include_config_snapshot=True)
    finally:
        db.close()

    # Enrich with stage data from pipeline_state.json and artifact summaries.
    if result.get("work_dir"):
        result["artifact_summary"] = collect_artifact_summary(result["work_dir"])
        result["media_summary"] = collect_media_summary(result["work_dir"])
        result["structured_log_path"] = str(structured_log_path(result["work_dir"]))
        state = get_pipeline_state(result["work_dir"])
        if state:
            result["stages"] = state.get("stages", [])
            result["current_stage_index"] = state.get("current_stage_index", 0)
        else:
            result["stages"] = []
            result["current_stage_index"] = 0
    else:
        result["artifact_summary"] = {"total": 0, "by_type": {}}
        result["media_summary"] = {"total": 0, "images": 0, "videos": 0, "by_source": {}, "video_providers": {}}
        result["structured_log_path"] = None
        result["stages"] = []
        result["current_stage_index"] = 0

    return result


@app.post("/api/runs")
async def api_create_run(request: Request):
    data = await request.json()
    run_name = data.get("run_name", "").strip()
    config_name = (data.get("config_name") or data.get("pipeline_config") or "default").strip()
    run_type = data.get("run_type", "full")
    start_url = data.get("start_url", "")

    if not run_name:
        raise HTTPException(status_code=400, detail="run_name is required")
    if load_config(config_name) is None:
        raise HTTPException(status_code=400, detail=f"Unknown config '{config_name}'")

    return create_run(run_name, config_name, run_type, start_url)


@app.post("/api/runs/{run_id}/start")
async def api_start_run(run_id: int):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status not in ("pending", "failed"):
            raise HTTPException(status_code=400, detail=f"Cannot start run in '{run.status}' state")
    finally:
        db.close()

    await run_manager.start(run_id, resume=None)
    return {"status": "started", "run_id": run_id}


@app.post("/api/runs/{run_id}/resume")
async def api_resume_run(run_id: int):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status == "running":
            raise HTTPException(status_code=400, detail="Run is already running")
        if not run.work_dir:
            raise HTTPException(status_code=400, detail="Run has no existing work directory to resume")
    finally:
        db.close()

    await run_manager.start(run_id, resume=True)
    return {"status": "resumed", "run_id": run_id}


@app.post("/api/runs/{run_id}/restart")
async def api_restart_run(run_id: int, request: Request):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status == "running":
            raise HTTPException(status_code=400, detail="Run is already running")
        if not run.work_dir:
            raise HTTPException(status_code=400, detail="Run has no existing work directory to restart")
    finally:
        db.close()

    data = await request.json()
    restart_from = str(data.get("restart_from") or "").strip()
    if not restart_from:
        raise HTTPException(status_code=400, detail="restart_from is required")
    await run_manager.start(run_id, resume=False, restart_from=restart_from)
    return {"status": "restarted", "run_id": run_id, "restart_from": restart_from}


@app.post("/api/runs/{run_id}/stages/{stage_selector}/retry")
async def api_retry_stage(run_id: int, stage_selector: str):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status == "running":
            raise HTTPException(status_code=400, detail="Run is already running")
        if not run.work_dir:
            raise HTTPException(status_code=400, detail="Run has no existing work directory to restart")
    finally:
        db.close()

    await run_manager.start(run_id, resume=False, restart_from=stage_selector)
    return {"status": "stage_retry_started", "run_id": run_id, "restart_from": stage_selector}


@app.post("/api/runs/{run_id}/cancel")
async def api_cancel_run(run_id: int):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status != "running":
            raise HTTPException(status_code=400, detail=f"Cannot cancel run in '{run.status}' state")
    finally:
        db.close()

    await run_manager.cancel(run_id)
    return {"status": "cancelled", "run_id": run_id}


@app.delete("/api/runs/{run_id}")
async def api_delete_run(run_id: int):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status == "running":
            raise HTTPException(status_code=400, detail="Cannot delete a running pipeline")
        db.query(RunLog).filter(RunLog.run_id == run_id).delete(synchronize_session=False)
        db.delete(run)
        db.commit()
        return {"status": "deleted", "run_id": run_id}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# API: Run Logs & Stages
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/logs")
async def api_run_logs(run_id: int, tail: int = 200, stage: Optional[str] = None):
    return get_run_logs(run_id, tail=tail, stage=stage)


@app.get("/api/runs/{run_id}/structured-logs")
async def api_run_structured_logs(
    run_id: int,
    tail: Optional[int] = None,
    limit: int = 200,
    before_sequence: Optional[int] = None,
    stage: Optional[str] = None,
    event_type: Optional[str] = None,
    level: Optional[str] = None,
):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if not run.work_dir:
            return {
                "items": [],
                "path": None,
                "has_more": False,
                "next_before_sequence": None,
            }
        path = structured_log_path(run.work_dir)
        effective_limit = tail if tail is not None else limit
        payload = load_structured_logs(
            run.work_dir,
            limit=effective_limit,
            before_sequence=before_sequence,
            stage=stage,
            event_type=event_type,
            level=level,
        )
        return {
            **payload,
            "path": str(path),
        }
    finally:
        db.close()


@app.get("/api/runs/{run_id}/stages/{stage_name}/log")
async def api_stage_log(run_id: int, stage_name: str, tail: int = 200):
    """Backward-compatible stage log endpoint — returns logs filtered by stage."""
    logs = get_run_logs(run_id, tail=tail, stage=stage_name)
    return {"lines": [log["message"] for log in logs]}


@app.get("/api/runs/{run_id}/stages")
async def api_run_stages(run_id: int):
    """Get stage details from pipeline_state.json."""
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        work_dir = run.work_dir
    finally:
        db.close()

    if not work_dir:
        return []

    state = get_pipeline_state(work_dir)
    if not state:
        return []
    return state.get("stages", [])


# ---------------------------------------------------------------------------
# API: Retrieval / Evaluation / Knowledge Base
# ---------------------------------------------------------------------------


def _load_run_or_404(run_id: int) -> Run:
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        db.expunge(run)
        return run
    finally:
        db.close()


@app.post("/api/runs/{run_id}/retrieve")
async def api_run_retrieve(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    query = str(data.get("query") or "").strip()
    config_name = str(data.get("config_name") or run.config_name or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="query is required")
    if not config_name:
        raise HTTPException(status_code=400, detail="config_name is required")
    return await asyncio.to_thread(
        run_retrieval_query,
        config_name=config_name,
        work_dir=run.work_dir,
        query=query,
    )


@app.get("/api/evaluation/assets")
async def api_evaluation_assets():
    return await asyncio.to_thread(list_eval_assets)


@app.get("/api/runs/{run_id}/evaluation/assets")
async def api_run_evaluation_assets(run_id: int):
    run = _load_run_or_404(run_id)
    return await asyncio.to_thread(list_eval_assets, run.work_dir)


@app.get("/api/evaluation/presets")
async def api_evaluation_presets():
    return await asyncio.to_thread(list_eval_presets)


@app.post("/api/evaluation/datasets/init")
async def api_init_eval_dataset(request: Request):
    data = await request.json()
    output_path = str(data.get("output_path") or "").strip()
    if not output_path:
        raise HTTPException(status_code=400, detail="output_path is required")
    force = bool(data.get("force") or False)
    try:
        return await asyncio.to_thread(init_eval_set, output_path, force=force)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/evaluation/datasets/summarize")
async def api_summarize_eval_dataset(request: Request):
    data = await request.json()
    dataset_path = str(data.get("dataset_path") or "").strip()
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")
    try:
        return await asyncio.to_thread(summarize_eval_set, dataset_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/evaluation/datasets/validate")
async def api_validate_eval_dataset(request: Request):
    data = await request.json()
    dataset_path = str(data.get("dataset_path") or "").strip()
    work_dir = str(data.get("work_dir") or "").strip() or None
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")
    try:
        return await asyncio.to_thread(validate_eval_set, dataset_path, work_dir=work_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/evaluation/benchmarks/summarize")
async def api_summarize_benchmark_dataset(request: Request):
    data = await request.json()
    dataset_dir = str(data.get("dataset_dir") or "").strip()
    if not dataset_dir:
        raise HTTPException(status_code=400, detail="dataset_dir is required")
    try:
        return await asyncio.to_thread(summarize_benchmark_dataset, dataset_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/benchmarks/retrieval")
async def api_list_retrieval_benchmarks(run_id: int):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        return []
    return await asyncio.to_thread(list_benchmark_jobs, run.work_dir)


@app.get("/api/runs/{run_id}/evaluation/jobs")
async def api_list_evaluation_jobs(run_id: int):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        return []
    return await asyncio.to_thread(list_evaluation_jobs, run.work_dir)


@app.post("/api/runs/{run_id}/evaluation/jobs/{job_id}/cancel")
async def api_cancel_evaluation_job(run_id: int, job_id: str):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    try:
        return await asyncio.to_thread(evaluation_job_manager.cancel, run.work_dir, job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/benchmarks/retrieval")
async def api_start_retrieval_benchmark(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    dataset_path = data.get("dataset_path")
    gates_path = data.get("gates_path")
    config_name = str(data.get("config_name") or run.config_name or "").strip()
    parallelism = int(data.get("parallelism") or 4)
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")
    if not config_name:
        raise HTTPException(status_code=400, detail="config_name is required")
    try:
        job = benchmark_job_manager.start(
            run_id=run_id,
            config_name=config_name,
            work_dir=run.work_dir,
            dataset_path=dataset_path,
            gates_path=gates_path,
            parallelism=parallelism,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return job


@app.post("/api/runs/{run_id}/evaluation/answers")
async def api_start_answer_generation(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    dataset_path = str(data.get("dataset_path") or "").strip()
    output_path = str(data.get("output_path") or "").strip()
    config_name = str(data.get("config_name") or run.config_name or "").strip()
    model = str(data.get("model") or "gemini-2.5-flash").strip()
    if not dataset_path:
        raise HTTPException(status_code=400, detail="dataset_path is required")
    if not output_path:
        raise HTTPException(status_code=400, detail="output_path is required")
    if not config_name:
        raise HTTPException(status_code=400, detail="config_name is required")
    try:
        return evaluation_job_manager.start_answer_generation(
            run_id=run_id,
            config_name=config_name,
            work_dir=run.work_dir,
            dataset_path=dataset_path,
            output_path=output_path,
            model=model,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/evaluation/ragas")
async def api_start_ragas(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    predictions_path = str(data.get("predictions_path") or "").strip()
    if not predictions_path:
        raise HTTPException(status_code=400, detail="predictions_path is required")
    metric_names = data.get("metric_names") or []
    if isinstance(metric_names, str):
        metric_names = [value.strip() for value in metric_names.split(",") if value.strip()]
    output_path = str(data.get("output_path") or "").strip() or None
    llm_model = str(data.get("llm_model") or "gemini-2.5-flash").strip()
    embedding_model = str(data.get("embedding_model") or "gemini-embedding-2-preview").strip()
    try:
        return evaluation_job_manager.start_ragas(
            run_id=run_id,
            work_dir=run.work_dir,
            predictions_path=predictions_path,
            metric_names=metric_names,
            llm_model=llm_model,
            embedding_model=embedding_model,
            output_path=output_path,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/evaluation/benchmarks/run-standard-retrieval")
async def api_start_standard_benchmark_retrieval(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    dataset_dir = str(data.get("dataset_dir") or "").strip()
    output_rankings_path = str(data.get("output_rankings_path") or "").strip()
    config_name = str(data.get("config_name") or run.config_name or "").strip()
    if not dataset_dir:
        raise HTTPException(status_code=400, detail="dataset_dir is required")
    if not output_rankings_path:
        raise HTTPException(status_code=400, detail="output_rankings_path is required")
    try:
        return evaluation_job_manager.start_standard_benchmark_retrieval(
            run_id=run_id,
            config_name=config_name,
            work_dir=run.work_dir,
            dataset_dir=dataset_dir,
            output_rankings_path=output_rankings_path,
            top_k=int(data.get("top_k") or 10),
            dense_top_k=int(data.get("dense_top_k") or 100),
            sparse_top_k=int(data.get("sparse_top_k") or 100),
            rrf_k=int(data.get("rrf_k") or 60),
            batch_size=int(data.get("batch_size") or 32),
            doc_cache_path=str(data.get("doc_cache_path") or "").strip() or None,
            query_cache_path=str(data.get("query_cache_path") or "").strip() or None,
            output_path=str(data.get("output_path") or "").strip() or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/evaluation/benchmarks/evaluate-rankings")
async def api_start_benchmark_rankings_eval(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    dataset_dir = str(data.get("dataset_dir") or "").strip()
    rankings_path = str(data.get("rankings_path") or "").strip()
    if not dataset_dir:
        raise HTTPException(status_code=400, detail="dataset_dir is required")
    if not rankings_path:
        raise HTTPException(status_code=400, detail="rankings_path is required")
    try:
        return evaluation_job_manager.start_benchmark_rankings_evaluation(
            run_id=run_id,
            work_dir=run.work_dir,
            dataset_dir=dataset_dir,
            rankings_path=rankings_path,
            k=int(data.get("k") or 10),
            output_path=str(data.get("output_path") or "").strip() or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/evaluation/benchmarks/export-ir")
async def api_start_export_ir(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    dataset_id = str(data.get("dataset_id") or "").strip()
    output_dir = str(data.get("output_dir") or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="dataset_id is required")
    if not output_dir:
        raise HTTPException(status_code=400, detail="output_dir is required")
    try:
        return evaluation_job_manager.start_export_ir_benchmark(
            run_id=run_id,
            work_dir=run.work_dir,
            dataset_id=dataset_id,
            output_dir=output_dir,
            max_queries=int(data.get("max_queries")) if data.get("max_queries") not in (None, "", False) else None,
            max_docs=int(data.get("max_docs")) if data.get("max_docs") not in (None, "", False) else None,
            full_corpus=bool(data.get("full_corpus") or False),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/evaluation/benchmarks/export-hf")
async def api_start_export_hf(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    mapping_path = str(data.get("mapping_path") or "").strip()
    output_dir = str(data.get("output_dir") or "").strip()
    if not mapping_path:
        raise HTTPException(status_code=400, detail="mapping_path is required")
    if not output_dir:
        raise HTTPException(status_code=400, detail="output_dir is required")
    try:
        return evaluation_job_manager.start_export_hf_benchmark(
            run_id=run_id,
            work_dir=run.work_dir,
            mapping_path=mapping_path,
            output_dir=output_dir,
            max_queries=int(data.get("max_queries")) if data.get("max_queries") not in (None, "", False) else None,
            max_docs=int(data.get("max_docs")) if data.get("max_docs") not in (None, "", False) else None,
            max_qrels=int(data.get("max_qrels")) if data.get("max_qrels") not in (None, "", False) else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/knowledge-base")
async def api_run_knowledge_base(run_id: int):
    run = _load_run_or_404(run_id)
    return await asyncio.to_thread(get_run_knowledge_status, run)


@app.get("/api/runs/{run_id}/knowledge-base/assertions")
async def api_run_knowledge_assertions(
    run_id: int,
    source: str = "promoted",
    query: Optional[str] = None,
    answer_type: Optional[str] = None,
    authority_class: Optional[str] = None,
    limit: int = 100,
):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    try:
        return await asyncio.to_thread(
            browse_assertions,
            run.work_dir,
            source=source,
            query=query,
            answer_type=answer_type,
            authority_class=authority_class,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/audit")
async def api_run_audit(run_id: int, repair_state: bool = False):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    return await asyncio.to_thread(audit_run_sync, run.work_dir, repair_state=repair_state)


@app.get("/api/runs/{run_id}/retriever-service")
async def api_retriever_service_status(run_id: int):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    return await asyncio.to_thread(get_retriever_service_status, run.work_dir)


@app.post("/api/runs/{run_id}/retriever-service/start")
async def api_start_retriever_service(run_id: int, request: Request):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    data = await request.json()
    config_name = str(data.get("config_name") or run.config_name or "").strip()
    if not config_name:
        raise HTTPException(status_code=400, detail="config_name is required")
    host = str(data.get("host") or "127.0.0.1").strip() or "127.0.0.1"
    port = int(data.get("port") or 8600 + run_id)
    max_concurrency = int(data.get("max_concurrency") or 4)
    request_timeout_seconds = float(data.get("request_timeout_seconds") or 90.0)
    try:
        return await asyncio.to_thread(
            start_retriever_service,
            config_name=config_name,
            work_dir=run.work_dir,
            host=host,
            port=port,
            max_concurrency=max_concurrency,
            request_timeout_seconds=request_timeout_seconds,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/retriever-service/stop")
async def api_stop_retriever_service(run_id: int):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    return await asyncio.to_thread(stop_retriever_service, run.work_dir)


@app.post("/api/indexes/{index_name}/delete-by-source")
async def api_delete_vectors_by_source(index_name: str, request: Request):
    data = await request.json()
    source_url_prefix = str(data.get("source_url_prefix") or "").strip()
    if not source_url_prefix:
        raise HTTPException(status_code=400, detail="source_url_prefix is required")
    try:
        return await asyncio.to_thread(delete_vectors_for_source, index_name, source_url_prefix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# API: URLs
# ---------------------------------------------------------------------------

@app.get("/api/urls/summary")
async def api_urls_summary():
    return get_all_urls_summary()


@app.get("/api/runs/{run_id}/urls")
async def api_run_urls(run_id: int):
    data = get_urls_detail(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="URL data not found for run")
    return data


@app.get("/api/runs/{run_id}/indexed-urls")
async def api_run_indexed_urls(run_id: int):
    data = get_indexed_urls_for_run(run_id)
    if data is None:
        raise HTTPException(status_code=404, detail="Indexed URL data not found for run")
    return data


@app.get("/api/urls/runs/{run_name}")
async def api_urls_by_name(run_name: str):
    data = get_urls_detail_by_name(run_name)
    if data is None:
        raise HTTPException(status_code=404, detail="URL data not found for run")
    return data


@app.get("/api/urls/indexed/{run_name}")
async def api_indexed_urls_by_name(run_name: str):
    data = get_indexed_urls_by_name(run_name)
    if data is None:
        raise HTTPException(status_code=404, detail="Indexed URL data not found for run")
    return data


@app.get("/api/urls/targets")
async def api_target_urls():
    return get_target_urls()


@app.post("/api/urls/targets/add")
async def api_add_target_url(request: Request):
    data = await request.json()
    success, message = add_target_url(data.get("config_name", ""), data.get("url", ""))
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message}


@app.post("/api/urls/targets/remove")
async def api_remove_target_url(request: Request):
    data = await request.json()
    success, message = remove_target_domain(data.get("config_name", ""), data.get("domain", ""))
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message}


@app.post("/api/urls/excluded/add")
async def api_add_excluded_subdomain(request: Request):
    data = await request.json()
    success, message = add_excluded_subdomain(
        data.get("config_name", ""),
        data.get("subdomain", ""),
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message}


@app.post("/api/urls/excluded/remove")
async def api_remove_excluded_subdomain(request: Request):
    data = await request.json()
    success, message = remove_excluded_subdomain(
        data.get("config_name", ""),
        data.get("subdomain", ""),
    )
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "ok", "message": message}


# ---------------------------------------------------------------------------
# API: Images
# ---------------------------------------------------------------------------

@app.get("/api/runs/{run_id}/images")
async def api_run_images(run_id: int):
    media_response = await api_run_media(run_id)
    images = [item for item in media_response["items"] if item.get("type") == "image"]
    return {"images": images, "total": len(images)}


@app.get("/api/runs/{run_id}/media")
async def api_run_media(run_id: int):
    db = get_db()
    try:
        run = db.get(Run, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        work_dir = run.work_dir
    finally:
        db.close()

    if not work_dir:
        return {"items": [], "total": 0, "images": 0, "videos": 0, "by_source": {}, "video_providers": {}}

    items = load_run_media(work_dir)
    summary = collect_media_summary(work_dir)
    return {
        "items": items,
        **summary,
    }


@app.get("/api/runs/{run_id}/artifacts")
async def api_run_artifacts(
    run_id: int,
    artifact_type: Optional[str] = None,
    producer_stage: Optional[str] = None,
    role: Optional[str] = None,
    query: Optional[str] = None,
    limit: int = 500,
):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    try:
        return await asyncio.to_thread(
            list_artifacts,
            run.work_dir,
            artifact_type=artifact_type,
            producer_stage=producer_stage,
            role=role,
            query=query,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/files")
async def api_run_files(run_id: int, path: Optional[str] = None):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    try:
        return await asyncio.to_thread(list_run_files, run.work_dir, path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/runs/{run_id}/file-content")
async def api_run_file_content(run_id: int, path: str, max_bytes: int = 200000):
    run = _load_run_or_404(run_id)
    if not run.work_dir:
        raise HTTPException(status_code=400, detail="Run has no work directory yet")
    if not path:
        raise HTTPException(status_code=400, detail="path is required")
    try:
        return await asyncio.to_thread(read_run_file, run.work_dir, path, max_bytes=max_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/assets")
async def api_serve_asset(path: str):
    requested = Path(path)
    resolved = requested.resolve() if requested.is_absolute() else (PROJECT_ROOT / requested).resolve()
    asset_root = (PROJECT_ROOT / "runs").resolve()
    try:
        resolved.relative_to(asset_root)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Asset not found")
    return FileResponse(resolved)


@app.get("/api/images")
async def api_serve_image(path: str):
    return await api_serve_asset(path)


# ---------------------------------------------------------------------------
# API: Pipeline Configs
# ---------------------------------------------------------------------------

@app.get("/api/pipeline-configs")
async def api_pipeline_configs():
    return get_available_configs()


@app.get("/api/stages")
async def api_list_stages():
    """Return stage definitions from the default pipeline config."""
    try:
        config = load_config("default")
        if config and "stages" in config:
            return [
                {
                    "type": s.get("type", ""),
                    "plugin": s.get("plugin", ""),
                    "key": f"{s.get('type', '')}/{s.get('plugin', '')}",
                }
                for s in config["stages"]
            ]
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# API: Configuration
# ---------------------------------------------------------------------------

@app.get("/api/configs")
async def api_list_configs():
    return list_configs()


@app.get("/api/configs/schema")
async def api_config_schema():
    return get_config_schema(serializable=True)


@app.get("/api/configs/{config_name}")
async def api_get_config(config_name: str):
    config = load_config(config_name)
    if config is None:
        raise HTTPException(status_code=404, detail="Config not found")
    return config


@app.put("/api/configs/{config_name}")
async def api_save_config(config_name: str, request: Request):
    data = await request.json()
    success, message = save_config(config_name, data)
    if not success:
        raise HTTPException(status_code=400, detail=message)
    return {"status": "saved", "message": message}


@app.get("/api/configs/{config_name}/validate")
async def api_validate_config(config_name: str):
    try:
        return await asyncio.to_thread(validate_config_sync, config_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/configs/{config_name}/dry-run")
async def api_dry_run_config(config_name: str):
    try:
        return await asyncio.to_thread(dry_run_config_sync, config_name)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# API: Pinecone Indexes
# ---------------------------------------------------------------------------

@app.get("/api/indexes")
async def api_list_indexes():
    return await asyncio.to_thread(fetch_all_index_stats)


@app.get("/api/indexes/{index_name}")
async def api_get_index(index_name: str):
    stats = await asyncio.to_thread(fetch_index_stats, index_name)
    if stats is None:
        raise HTTPException(status_code=404, detail="Could not fetch index stats")
    return stats


@app.post("/api/indexes/snapshot")
async def api_snapshot_indexes():
    count = await asyncio.to_thread(snapshot_indexes)
    return {"status": "ok", "snapshots_saved": count}


@app.get("/api/indexes/{index_name}/history")
async def api_index_history(index_name: str, limit: int = 30):
    return get_snapshot_history(index_name, limit)


# ---------------------------------------------------------------------------
# API: Scanner
# ---------------------------------------------------------------------------

@app.post("/api/scan")
async def api_scan():
    count = await asyncio.to_thread(scan_and_import)
    return {"status": "ok", "imported": count}


@app.post("/api/scan/force")
async def api_force_scan():
    count = await asyncio.to_thread(scan_and_import, True)
    return {"status": "ok", "imported": count}


# ---------------------------------------------------------------------------
# WebSocket for live log streaming
# ---------------------------------------------------------------------------

@app.websocket("/ws/runs/{run_id}/logs")
async def ws_run_logs(websocket: WebSocket, run_id: int):
    await websocket.accept()
    run_manager.add_ws(run_id, websocket)

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        run_manager.remove_ws(run_id, websocket)


# ---------------------------------------------------------------------------
# Catch-all: serve Next.js static export
# ---------------------------------------------------------------------------

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    if UI_DIR.is_dir():
        file_path = UI_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        index_path = UI_DIR / full_path / "index.html"
        if index_path.is_file():
            return FileResponse(index_path)
        html_path = UI_DIR / f"{full_path}.html"
        if html_path.is_file():
            return FileResponse(html_path)
        root_index = UI_DIR / "index.html"
        if root_index.is_file():
            return FileResponse(root_index)
    raise HTTPException(status_code=404, detail="Not found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8050, reload=True)
