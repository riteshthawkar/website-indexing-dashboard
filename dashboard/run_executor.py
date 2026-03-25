"""
Pipeline executor — directly imports and runs the modular pipeline orchestrator
with wired callbacks for real-time dashboard integration.

No subprocess, no duplicate state tracking.  The pipeline's own pipeline_state.json
is the single source of truth for stage progress.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from database import Run, RunLog, get_db, utcnow
from run_data import collect_metrics

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import load_config as load_pipeline_config, list_configs as list_pipeline_configs
from pipeline.core.orchestrator import PipelineOrchestrator
from pipeline.core.state import load_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env():
    """Load .env files from project root into os.environ."""
    for env_path in [PROJECT_ROOT / ".env", PROJECT_ROOT / "scrape_latest" / ".env"]:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key not in os.environ:
                            os.environ[key] = value


def _resolve_work_dir(config_name: str, run_id: str) -> Path:
    """Resolve the working directory for a pipeline run."""
    return PROJECT_ROOT / "runs" / config_name / run_id


def get_pipeline_state(work_dir: str) -> Optional[dict]:
    """Load pipeline_state.json from a run's work directory."""
    state = load_state(Path(work_dir))
    if state:
        return state.to_dict()
    return None


def get_available_configs() -> List[Dict[str, str]]:
    """Return available pipeline config names from the pipeline package."""
    return list_pipeline_configs()


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------

def create_run(run_name: str, config_name: str, run_type: str = "full", start_url: str = "") -> dict:
    """Create a new run record in the database."""
    config_name = (config_name or "default").strip()
    config_snapshot = None
    try:
        config_snapshot = load_pipeline_config(config_name)
    except Exception as e:
        logger.warning("Could not snapshot config %s for run %s: %s", config_name, run_name, e)

    db = get_db()
    try:
        run = Run(
            run_name=run_name,
            config_name=config_name,
            run_type=run_type,
            start_url=start_url or None,
            status="pending",
            created_at=utcnow(),
            config_snapshot_json=json.dumps(config_snapshot) if config_snapshot else None,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        return run.to_dict()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _update_run(run_id: int, **kwargs):
    """Update run fields in the database."""
    db = get_db()
    try:
        run = db.query(Run).get(run_id)
        if run:
            for k, v in kwargs.items():
                setattr(run, k, v)
            db.commit()
    finally:
        db.close()


def _add_log(run_id: int, message: str, level: str = "info", stage: str = None):
    """Insert a structured log entry."""
    db = get_db()
    try:
        log = RunLog(run_id=run_id, level=level, stage=stage, message=message)
        db.add(log)
        db.commit()
    finally:
        db.close()


async def execute_pipeline(
    run_id: int,
    on_log: Optional[Callable] = None,
    on_stage_event: Optional[Callable] = None,
):
    """
    Execute the pipeline by directly importing PipelineOrchestrator.

    Callbacks:
        on_log(run_id, level, stage, message) — for WebSocket streaming
        on_stage_event(run_id, event, stage_key, info) — for stage progress
    """
    _load_env()

    db = get_db()
    try:
        run = db.query(Run).get(run_id)
        if not run:
            logger.error(f"Run {run_id} not found")
            return
        config_name = run.config_name
        run_name = run.run_name
    finally:
        db.close()

    # Generate a unique run_id for the pipeline (use db id as string)
    pipeline_run_id = f"run_{run_id}"
    work_dir = _resolve_work_dir(config_name, pipeline_run_id)

    _update_run(run_id, status="running", started_at=utcnow(), work_dir=str(work_dir))

    async def _emit_log(level: str, message: str, stage: str = None):
        """Emit a log line to WebSocket and store in DB."""
        _add_log(run_id, message, level=level, stage=stage)
        if on_log:
            result = on_log(run_id, level, stage, message)
            if asyncio.iscoroutine(result):
                await result

    # Build pipeline callbacks
    current_stage = [None]  # mutable container for closure

    def _on_stage_start(stage_type: str, plugin_name: str, info: Optional[Dict]):
        stage_key = f"{stage_type}/{plugin_name}"
        current_stage[0] = stage_key
        asyncio.get_event_loop().create_task(
            _emit_log("info", f"Starting stage: {stage_key}", stage=stage_key)
        )
        if on_stage_event:
            result = on_stage_event(run_id, "start", stage_key, info)
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().create_task(result)

    def _on_stage_complete(stage_type: str, plugin_name: str, info: Optional[Dict]):
        stage_key = f"{stage_type}/{plugin_name}"
        status = info.get("status", "unknown") if info else "unknown"
        metrics_str = json.dumps(info.get("metrics", {})) if info else "{}"
        asyncio.get_event_loop().create_task(
            _emit_log("info", f"Stage {stage_key} {status}: {metrics_str}", stage=stage_key)
        )
        if on_stage_event:
            result = on_stage_event(run_id, "complete", stage_key, info)
            if asyncio.iscoroutine(result):
                asyncio.get_event_loop().create_task(result)

        # Update cached metrics after each stage
        try:
            m = collect_metrics(work_dir)
            if m:
                _update_run(run_id, **m)
        except Exception as e:
            logger.warning(f"Failed to collect metrics: {e}")

    def _on_pipeline_log(level: str, message: str):
        stage = current_stage[0]
        asyncio.get_event_loop().create_task(
            _emit_log(level, message, stage=stage)
        )

    try:
        await _emit_log("info", f"Loading config: {config_name}")

        # Load the pipeline config
        config = load_pipeline_config(config_name)

        # Apply overrides from the dashboard run
        db2 = get_db()
        try:
            run = db2.query(Run).get(run_id)
            if run and run.start_url:
                config["crawler"] = config.get("crawler", {})
                config["crawler"]["start_url"] = run.start_url
        finally:
            db2.close()

        # Create and run the orchestrator
        orchestrator = PipelineOrchestrator(
            config=config,
            work_dir=work_dir,
            run_id=pipeline_run_id,
            on_stage_start=_on_stage_start,
            on_stage_complete=_on_stage_complete,
            on_log=_on_pipeline_log,
        )

        await _emit_log("info", f"Pipeline starting with {len(config.get('stages', []))} stages")

        # Resume failed or interrupted runs when the pipeline state already exists.
        resume = (work_dir / "pipeline_state.json").exists()
        if resume:
            await _emit_log("info", f"Resuming pipeline state from {work_dir / 'pipeline_state.json'}")

        # Run the pipeline
        state = await orchestrator.run(resume=resume)

        # Final metrics update
        try:
            m = collect_metrics(work_dir)
            if m:
                _update_run(run_id, **m)
        except Exception:
            pass

        if state.status == "completed":
            _update_run(run_id, status="completed", completed_at=utcnow())
            await _emit_log("info", "Pipeline completed successfully!")
        else:
            error_msg = None
            for s in state.stages:
                if s.error_message:
                    error_msg = s.error_message
                    break
            _update_run(
                run_id,
                status="failed",
                completed_at=utcnow(),
                error_message=error_msg or "Pipeline failed",
            )
            await _emit_log("error", f"Pipeline failed: {error_msg or 'unknown error'}")

    except asyncio.CancelledError:
        logger.info(f"Pipeline cancelled for run {run_id}")
        _update_run(run_id, status="cancelled", completed_at=utcnow())
        await _emit_log("info", "Pipeline cancelled")

    except Exception as e:
        logger.error(f"Pipeline error for run {run_id}: {e}", exc_info=True)
        _update_run(
            run_id,
            status="failed",
            completed_at=utcnow(),
            error_message=str(e),
        )
        try:
            await _emit_log("error", f"Pipeline error: {e}")
        except Exception:
            pass


def cancel_run(run_id: int) -> bool:
    """Mark a run as cancelled."""
    db = get_db()
    try:
        run = db.query(Run).get(run_id)
        if not run or run.status != "running":
            return False
        run.status = "cancelled"
        run.completed_at = utcnow()
        db.commit()
        return True
    finally:
        db.close()


def get_run_logs(run_id: int, tail: int = 200, stage: str = None) -> list[dict]:
    """Get recent log entries for a run."""
    db = get_db()
    try:
        q = db.query(RunLog).filter(RunLog.run_id == run_id)
        if stage:
            q = q.filter(RunLog.stage == stage)
        q = q.order_by(RunLog.created_at.desc()).limit(tail)
        logs = q.all()
        return [log.to_dict() for log in reversed(logs)]
    finally:
        db.close()


def cleanup_stale_runs():
    """Mark any runs stuck in 'running' on startup as failed."""
    db = get_db()
    try:
        stale = db.query(Run).filter(Run.status == "running").all()
        for run in stale:
            run.status = "failed"
            run.error_message = "Interrupted (dashboard restart)"
            run.completed_at = utcnow()
        db.commit()
        if stale:
            logger.info(f"Cleaned up {len(stale)} stale runs")
    finally:
        db.close()
