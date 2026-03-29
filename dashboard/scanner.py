"""
Filesystem scanner — discovers pipeline runs by walking runs/*/*/pipeline_state.json
and imports them into the dashboard database.
"""

import json
import logging
from pathlib import Path

from database import Run, get_db, utcnow
from run_executor import collect_metrics, get_pipeline_state, load_pipeline_config, sync_terminal_pipeline_state

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def _parse_iso(iso_str: str):
    """Parse ISO-8601 datetime string."""
    from datetime import datetime, timezone
    if not iso_str:
        return None
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _state_error_message(state: dict) -> str | None:
    error_message = state.get("error_message")
    if error_message:
        return str(error_message)

    for stage in state.get("stages") or []:
        stage_error = stage.get("error_message")
        if stage_error:
            return str(stage_error)
    return None


def scan_and_import(force_rescan: bool = False) -> int:
    """
    Walk runs/*/*/pipeline_state.json and import any runs not already in the DB.
    Returns the number of newly imported runs.
    """
    if not RUNS_DIR.is_dir():
        return 0

    db = get_db()
    imported = 0

    try:
        existing_runs = db.query(Run).filter(Run.work_dir.isnot(None)).all()
        existing_by_dir = {run.work_dir: run for run in existing_runs if run.work_dir}

        # Walk runs/<config_name>/<run_id>/pipeline_state.json
        for state_file in sorted(RUNS_DIR.rglob("pipeline_state.json")):
            work_dir = state_file.parent
            work_dir_str = str(work_dir)

            # Load the pipeline state
            state = get_pipeline_state(work_dir_str)
            if not state:
                continue

            # Derive config name from directory structure: runs/<config_name>/<run_id>/
            parts = work_dir.relative_to(RUNS_DIR).parts
            if len(parts) < 2:
                continue
            config_name = parts[0]

            logger.info(f"Importing run from {work_dir}")

            # Collect metrics from the filesystem
            metrics = collect_metrics(work_dir)
            config_snapshot = None
            try:
                config_snapshot = load_pipeline_config(config_name)
            except Exception:
                pass

            existing = existing_by_dir.get(work_dir_str)
            if existing:
                imported_status = state.get("status", existing.status or "completed")
                imported_completed_at = _parse_iso(state.get("finished_at"))
                imported_error_message = _state_error_message(state)
                preserve_terminal_status = (
                    imported_status == "running"
                    and existing.status in {"failed", "cancelled", "completed"}
                )

                existing.run_name = state.get("project_name", existing.run_name or config_name)
                existing.config_name = config_name
                if not preserve_terminal_status:
                    existing.status = imported_status
                existing.started_at = _parse_iso(state.get("started_at"))
                if not preserve_terminal_status or existing.completed_at is None:
                    existing.completed_at = imported_completed_at
                if imported_error_message and not preserve_terminal_status:
                    existing.error_message = imported_error_message
                existing.pages_scraped = metrics.get("pages_scraped", 0)
                existing.documents_downloaded = metrics.get("documents_downloaded", 0)
                existing.pages_cleaned = metrics.get("pages_cleaned", 0)
                existing.docs_converted = metrics.get("docs_converted", 0)
                existing.summaries_generated = metrics.get("summaries_generated", 0)
                existing.embeddings_created = metrics.get("embeddings_created", 0)
                existing.images_extracted = metrics.get("images_extracted", 0)
                existing.videos_extracted = metrics.get("videos_extracted", 0)
                existing.media_items_extracted = metrics.get("media_items_extracted", 0)
                existing.structured_documents_created = metrics.get("structured_documents_created", 0)
                existing.chunks_created = metrics.get("chunks_created", 0)
                existing.artifact_count = metrics.get("artifact_count", 0)
                existing.chunk_strategy = metrics.get("chunk_strategy") or None
                existing.total_bytes = metrics.get("total_bytes", 0)
                if config_snapshot:
                    existing.config_snapshot_json = json.dumps(config_snapshot)
                if existing.status in {"completed", "failed", "cancelled"}:
                    sync_terminal_pipeline_state(
                        work_dir_str,
                        status=existing.status,
                        error_message=existing.error_message,
                        finished_at=existing.completed_at.isoformat() if existing.completed_at else None,
                    )
                continue

            run = Run(
                run_name=state.get("project_name", config_name),
                config_name=config_name,
                run_type="full",
                status=state.get("status", "completed"),
                work_dir=work_dir_str,
                created_at=_parse_iso(state.get("started_at")) or utcnow(),
                started_at=_parse_iso(state.get("started_at")),
                completed_at=_parse_iso(state.get("finished_at")),
                error_message=_state_error_message(state),
                pages_scraped=metrics.get("pages_scraped", 0),
                documents_downloaded=metrics.get("documents_downloaded", 0),
                pages_cleaned=metrics.get("pages_cleaned", 0),
                docs_converted=metrics.get("docs_converted", 0),
                summaries_generated=metrics.get("summaries_generated", 0),
                embeddings_created=metrics.get("embeddings_created", 0),
                images_extracted=metrics.get("images_extracted", 0),
                videos_extracted=metrics.get("videos_extracted", 0),
                media_items_extracted=metrics.get("media_items_extracted", 0),
                structured_documents_created=metrics.get("structured_documents_created", 0),
                chunks_created=metrics.get("chunks_created", 0),
                artifact_count=metrics.get("artifact_count", 0),
                chunk_strategy=metrics.get("chunk_strategy") or None,
                total_bytes=metrics.get("total_bytes", 0),
                is_imported=True,
                config_snapshot_json=json.dumps(config_snapshot) if config_snapshot else None,
            )
            db.add(run)
            if run.status in {"completed", "failed", "cancelled"}:
                sync_terminal_pipeline_state(
                    work_dir_str,
                    status=run.status,
                    error_message=run.error_message,
                    finished_at=run.completed_at.isoformat() if run.completed_at else None,
                )
            imported += 1

        db.commit()
        if imported:
            logger.info(f"Imported {imported} runs from filesystem")

    except Exception as e:
        db.rollback()
        logger.error(f"Error during scan: {e}", exc_info=True)
    finally:
        db.close()

    return imported
