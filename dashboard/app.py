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

from database import init_db, get_db, Run, RunLog
from run_data import collect_artifact_summary, collect_media_summary, load_run_media
from scanner import scan_and_import
from run_executor import (
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
from pinecone_ops import (
    fetch_all_index_stats,
    fetch_index_stats,
    snapshot_indexes,
    get_snapshot_history,
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

    async def start(self, run_id: int):
        if run_id in self._tasks and not self._tasks[run_id].done():
            raise HTTPException(status_code=400, detail="Pipeline is already running")

        task = asyncio.create_task(
            execute_pipeline(
                run_id,
                on_log=self._broadcast_log,
                on_stage_event=self._broadcast_stage,
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
    init_db()
    cleanup_stale_runs()
    imported = scan_and_import()
    if imported:
        logger.info(f"Imported {imported} runs on startup")
    yield
    run_manager.shutdown()


app = FastAPI(title="MBZUAI Pipeline Dashboard", lifespan=lifespan)

# CORS for dev mode (Next.js on :3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
        run = db.query(Run).get(run_id)
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
        run = db.query(Run).get(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.status not in ("pending", "failed"):
            raise HTTPException(status_code=400, detail=f"Cannot start run in '{run.status}' state")
    finally:
        db.close()

    await run_manager.start(run_id)
    return {"status": "started", "run_id": run_id}


@app.post("/api/runs/{run_id}/cancel")
async def api_cancel_run(run_id: int):
    db = get_db()
    try:
        run = db.query(Run).get(run_id)
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
        run = db.query(Run).get(run_id)
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
        run = db.query(Run).get(run_id)
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
        run = db.query(Run).get(run_id)
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
        run = db.query(Run).get(run_id)
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
