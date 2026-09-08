"""
Pipeline orchestrator — runs stages sequentially with checkpoint/resume.

Usage:
    orchestrator = PipelineOrchestrator(config)
    result = await orchestrator.run()
    # or resume a crashed run:
    result = await orchestrator.run(resume=True)
"""

import fcntl
import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .artifacts import load_artifact_catalog, save_artifact_catalog
from .base import PipelineStage, StageContext, StageResult, StageStatus
from .config import (
    indexing_build_identity,
    production_indexing_contract_fingerprint,
    sanitized_config_snapshot,
)
from .io import atomic_write_json, ensure_dir, load_json_safe
from .registry import auto_discover, get_stage
from .run_audit import audit_run, reconcile_state_artifact_ids, save_run_audit
from .state import (
    PipelineState,
    StageState,
    load_state,
    now_iso,
    save_state,
)

logger = logging.getLogger(__name__)

_MAX_LOG_MAPPING_ITEMS = 20
_MAX_LOG_SEQUENCE_ITEMS = 20
_LOG_VALUE_SAMPLE_SIZE = 5

# Callback type aliases
StageCallback = Callable[[str, str, Optional[Dict]], None]  # (stage_type, name, info)
LogCallback = Callable[[str, str], None]  # (level, message)


def _metrics_for_log(value: Any) -> Any:
    """Bound verbose metric collections without changing persisted metrics."""

    if isinstance(value, dict):
        if len(value) > _MAX_LOG_MAPPING_ITEMS:
            sample = list(value.items())[:_LOG_VALUE_SAMPLE_SIZE]
            return {
                "_entry_count": len(value),
                "_sample": {
                    key: _metrics_for_log(item_value)
                    for key, item_value in sample
                },
            }
        return {key: _metrics_for_log(item_value) for key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_LOG_SEQUENCE_ITEMS:
            return {
                "_entry_count": len(value),
                "_sample": [
                    _metrics_for_log(item)
                    for item in value[:_LOG_VALUE_SAMPLE_SIZE]
                ],
            }
        return [_metrics_for_log(item) for item in value]
    return value


class RunLockError(RuntimeError):
    """Raised when another pipeline process already owns the run work directory."""


class RunConfigMismatchError(RuntimeError):
    """Raised when a resumed run does not match its immutable config snapshot."""


class RunAuditError(RuntimeError):
    """Raised when a run fails integrity validation after stage execution."""


class ActiveReleaseMutationError(RuntimeError):
    """Raised when a run would rewrite the active release's vector namespaces."""


class PipelineOrchestrator:
    """Runs a sequence of pipeline stages with checkpoint/resume support.

    Args:
        config: Fully merged configuration dictionary.
        work_dir: Optional override for the run working directory.
        run_id: Optional override for the run ID (auto-generated if omitted).
        on_stage_start: Callback fired before each stage.
        on_stage_complete: Callback fired after each stage.
        on_log: Callback for log messages (for terminal or service integrations).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        work_dir: Optional[Path] = None,
        run_id: Optional[str] = None,
        on_stage_start: Optional[StageCallback] = None,
        on_stage_complete: Optional[StageCallback] = None,
        on_log: Optional[LogCallback] = None,
    ):
        self.config = config
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.project_name = config.get("project_name", "default")

        # Resolve work directory
        base_work_dir = Path(config.get("work_dir", "./runs"))
        self.work_dir = (work_dir or (base_work_dir / self.project_name / self.run_id)).resolve()

        # Callbacks
        self._on_stage_start = on_stage_start
        self._on_stage_complete = on_stage_complete
        self._on_log = on_log

        # Stage instances (populated by _build_stages)
        self._stages: List[tuple[str, str, str, Dict[str, Any], PipelineStage]] = []
        self._run_lock_handle = None
        self._run_lock_path: Optional[Path] = None

    def _acquire_run_lock(self) -> None:
        ensure_dir(self.work_dir)
        lock_path = self.work_dir / ".run.lock"
        handle = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.seek(0)
            details = handle.read().strip()
            handle.close()
            raise RunLockError(
                f"Run directory is already locked by another process: {self.work_dir}"
                + (f" ({details})" if details else "")
            ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "run_id": self.run_id,
                    "project_name": self.project_name,
                    "locked_at": now_iso(),
                }
            )
        )
        handle.flush()
        os.fsync(handle.fileno())
        self._run_lock_handle = handle
        self._run_lock_path = lock_path

    def _release_run_lock(self) -> None:
        handle = self._run_lock_handle
        if handle is None:
            return
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            finally:
                handle.close()
        finally:
            # A flock belongs to an inode. Removing the file after unlocking
            # lets a waiter retain a lock on the old inode while a third process
            # creates and locks a new one. Keep one stable lock inode forever.
            self._run_lock_handle = None
            self._run_lock_path = None

    def _stage_output_dir(self, stage_id: str) -> Path:
        return self.work_dir / "stage_outputs" / stage_id

    def _active_release_pointer_path(self) -> Path:
        pipeline_cfg = self.config.get("pipeline") if isinstance(self.config.get("pipeline"), dict) else {}
        configured = str(
            pipeline_cfg.get("active_release_file")
            or os.environ.get("ACTIVE_RELEASE_FILE")
            or ""
        ).strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return (self.work_dir.parent / "active_release.json").resolve()

    def _is_active_release_run(self, run_id: str) -> bool:
        pointer = load_json_safe(self._active_release_pointer_path(), {}) or {}
        return isinstance(pointer, dict) and str(pointer.get("run_id") or "").strip() == str(run_id).strip()

    def _upload_stage_index(self) -> Optional[int]:
        for index, (_stage_type, plugin_name, stage_id, _stage_def, _instance) in enumerate(self._stages):
            if str(plugin_name) in {"gemini_pinecone", "gemini_pgvector"} or str(stage_id) == "upload_retrieval":
                return index
        return None

    def _raise_if_active_upload_would_run(
        self,
        *,
        state: Optional[PipelineState],
        restart_index: Optional[int] = None,
    ) -> None:
        upload_index = self._upload_stage_index()
        run_id = state.run_id if state is not None else self.run_id
        if upload_index is None or not self._is_active_release_run(run_id):
            return
        upload_would_run = state is None
        if restart_index is not None:
            upload_would_run = restart_index <= upload_index
        elif state is not None:
            upload_would_run = upload_index >= len(state.stages) or state.stages[upload_index].status != "completed"
        if upload_would_run:
            raise ActiveReleaseMutationError(
                f"Run {run_id!r} is the active release and cannot re-execute upload stage "
                f"{self._stages[upload_index][2]!r}; create a new candidate run ID instead"
            )

    def _archive_stage_output_dir(self, stage_id: str) -> Optional[Path]:
        source = self._stage_output_dir(stage_id)
        if not source.exists():
            return None

        stale_root = ensure_dir(self.work_dir / "stage_outputs" / "_stale")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = stale_root / f"{stage_id}_{stamp}"
        suffix = 1
        while destination.exists():
            destination = stale_root / f"{stage_id}_{stamp}_{suffix}"
            suffix += 1
        shutil.move(str(source), str(destination))
        return destination

    def _resolve_stage_index(self, selector: str) -> int:
        text = str(selector).strip()
        if not text:
            raise ValueError("Stage selector must not be empty")

        if text.isdigit():
            index = int(text)
            if 1 <= index <= len(self._stages):
                return index - 1
            if 0 <= index < len(self._stages):
                return index

        for index, (stage_type, plugin_name, stage_id, _stage_def, instance) in enumerate(self._stages):
            options = {
                str(stage_id),
                str(plugin_name),
                str(instance.name or ""),
                f"{stage_type}/{plugin_name}",
            }
            if text in options:
                return index

        raise ValueError(f"Unknown stage selector: {selector}")

    def resolve_stage_index(self, selector: str) -> int:
        """Resolve a public stage selector without starting or mutating a run."""
        if not self._stages:
            self._build_stages()
        return self._resolve_stage_index(selector)

    def _align_state_to_stages(self, state: PipelineState) -> None:
        configured_count = len(self._stages)
        existing_count = len(state.stages)
        shared = min(existing_count, configured_count)

        for index in range(shared):
            stage_type, plugin_name, stage_id, _stage_def, _instance = self._stages[index]
            stage_state = state.stages[index]
            expected = (stage_type, plugin_name, stage_id)
            actual = (stage_state.stage_type, stage_state.name, stage_state.stage_id or stage_id)
            if actual != expected:
                raise ValueError(
                    "Cannot resume run with a different stage layout at index "
                    f"{index}: expected {expected}, found {actual}"
                )

        if existing_count > configured_count:
            raise ValueError(
                "Cannot resume run: saved state has more stages than the current config"
            )

        if existing_count < configured_count:
            for index in range(existing_count, configured_count):
                stage_type, plugin_name, stage_id, _stage_def, _instance = self._stages[index]
                state.stages.append(
                    StageState(
                        name=plugin_name,
                        stage_type=stage_type,
                        stage_id=stage_id,
                        status="pending",
                    )
                )
            self._log(
                "info",
                "Extended saved state with %d new stage(s) from the current config",
                configured_count - existing_count,
            )

    def _reset_from_stage(
        self,
        state: PipelineState,
        artifact_catalog,
        start_index: int,
    ) -> None:
        for i in range(start_index, len(self._stages)):
            _stage_type, _plugin_name, stage_id, _stage_def, _instance = self._stages[i]
            archived_to = self._archive_stage_output_dir(stage_id)
            if archived_to:
                self._log("info", "Archived stale stage output %s -> %s", stage_id, archived_to)
            removed = artifact_catalog.remove_by_producer_stage(stage_id)
            if removed:
                self._log("info", "Removed %d stale artifact records for stage %s", len(removed), stage_id)
            state.stages[i].stage_id = stage_id
            state.stages[i].reset_for_rerun()

        state.status = "running"
        state.finished_at = None
        state.current_stage_index = start_index

    def _first_inconsistent_stage_index(self, state: PipelineState) -> Optional[int]:
        first_non_terminal: Optional[int] = None
        for index, stage in enumerate(state.stages):
            if not stage.is_terminal:
                if first_non_terminal is None:
                    first_non_terminal = index
                continue
            if first_non_terminal is not None:
                return first_non_terminal
        return None

    def _pipeline_config(self) -> Dict[str, Any]:
        cfg = self.config.get("pipeline", {})
        return cfg if isinstance(cfg, dict) else {}

    def _write_config_snapshot(self) -> None:
        atomic_write_json(
            self.work_dir / "resolved_config.json",
            {
                "run_id": self.run_id,
                "project_name": self.project_name,
                "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(
                    self.config
                ),
                "indexing_build": indexing_build_identity(),
                "config": sanitized_config_snapshot(self.config),
            },
        )

    def _validate_resume_config_snapshot(self) -> None:
        """Prove completed artifacts were built with the current index contract.

        This check must run before state alignment, stage skipping, or rewriting
        ``resolved_config.json``. Otherwise a changed production config could
        relabel old completed outputs as if they had been rebuilt.
        """

        snapshot_path = self.work_dir / "resolved_config.json"
        snapshot = load_json_safe(snapshot_path, None)
        production = bool(self._pipeline_config().get("production_profile", False))
        if not isinstance(snapshot, dict):
            if production:
                raise RunConfigMismatchError(
                    f"Cannot resume production run without a valid immutable config snapshot: {snapshot_path}"
                )
            return

        snapshot_config = snapshot.get("config")
        if not isinstance(snapshot_config, dict):
            raise RunConfigMismatchError(
                f"Cannot resume run because its config snapshot is invalid: {snapshot_path}"
            )
        snapshot_run_id = str(snapshot.get("run_id") or "").strip()
        if snapshot_run_id and snapshot_run_id != self.run_id:
            raise RunConfigMismatchError(
                "Cannot resume run because resolved_config.json belongs to a different run ID "
                f"({snapshot_run_id!r} != {self.run_id!r})"
            )

        actual_fingerprint = production_indexing_contract_fingerprint(snapshot_config)
        recorded_fingerprint = str(
            snapshot.get("production_indexing_contract_fingerprint") or ""
        ).strip()
        if production and not recorded_fingerprint:
            raise RunConfigMismatchError(
                "Cannot resume production run because its recorded indexing fingerprint is missing"
            )
        if recorded_fingerprint and recorded_fingerprint != actual_fingerprint:
            raise RunConfigMismatchError(
                "Cannot resume run because resolved_config.json no longer matches its recorded indexing fingerprint"
            )

        requested_fingerprint = production_indexing_contract_fingerprint(self.config)
        if requested_fingerprint != actual_fingerprint:
            raise RunConfigMismatchError(
                "Cannot resume run with a different immutable indexing configuration "
                f"(saved={actual_fingerprint}, requested={requested_fingerprint}); create a new run ID"
            )

    def _migrate_incomplete_config_snapshot(
        self,
        *,
        restart_from_index: Optional[int] = None,
    ) -> None:
        """Audit and migrate a non-production run to the current config.

        Production runs remain immutable. An unfinished development run may
        preserve its active-stage checkpoint. A completed non-production run is
        eligible only when the caller explicitly restarts from stage zero, which
        invalidates every completed stage while preserving resumable raw files.
        """

        snapshot_path = self.work_dir / "resolved_config.json"
        snapshot = load_json_safe(snapshot_path, None)
        state = load_state(self.work_dir)
        if not isinstance(snapshot, dict) or state is None:
            raise RunConfigMismatchError(
                "Cannot migrate incomplete run config without a valid snapshot and pipeline state"
            )
        snapshot_config = snapshot.get("config")
        if not isinstance(snapshot_config, dict):
            raise RunConfigMismatchError(
                "Cannot migrate incomplete run config because the saved config is invalid"
            )
        saved_pipeline_value = snapshot_config.get("pipeline")
        saved_pipeline = (
            saved_pipeline_value if isinstance(saved_pipeline_value, dict) else {}
        )
        current_pipeline = self._pipeline_config()
        if bool(saved_pipeline.get("production_profile", False)) or bool(
            current_pipeline.get("production_profile", False)
        ):
            raise RunConfigMismatchError(
                "Cannot migrate an incomplete production run; create a new run ID"
            )
        if str(snapshot.get("run_id") or self.run_id) != self.run_id:
            raise RunConfigMismatchError(
                "Cannot migrate incomplete run config for a different run ID"
            )
        has_completed_stages = any(
            stage.status == "completed" for stage in state.stages
        )
        if has_completed_stages and restart_from_index != 0:
            raise RunConfigMismatchError(
                "Cannot migrate run config after any stage has completed unless "
                "the non-production run is explicitly restarted from stage zero"
            )
        if len(state.stages) != len(self._stages):
            raise RunConfigMismatchError(
                "Cannot migrate incomplete run config with a different stage layout"
            )
        for index, (stage_type, plugin_name, stage_id, _definition, _instance) in enumerate(
            self._stages
        ):
            saved_stage = state.stages[index]
            actual = (
                saved_stage.stage_type,
                saved_stage.name,
                saved_stage.stage_id or stage_id,
            )
            expected = (stage_type, plugin_name, stage_id)
            if actual != expected:
                raise RunConfigMismatchError(
                    "Cannot migrate incomplete run config with a different stage layout "
                    f"at index {index}: saved={actual}, requested={expected}"
                )

        recorded_fingerprint = str(
            snapshot.get("production_indexing_contract_fingerprint") or ""
        ).strip()
        saved_build = snapshot.get("indexing_build")
        saved_hashes = (
            saved_build.get("implementation_sha256")
            if isinstance(saved_build, dict)
            else None
        )
        if recorded_fingerprint:
            if not isinstance(saved_hashes, dict) or not saved_hashes:
                raise RunConfigMismatchError(
                    "Cannot verify the saved run fingerprint because its implementation hashes are missing"
                )
            verified_saved_fingerprint = production_indexing_contract_fingerprint(
                snapshot_config,
                implementation_hashes=saved_hashes,
            )
            if verified_saved_fingerprint != recorded_fingerprint:
                raise RunConfigMismatchError(
                    "Cannot migrate incomplete run config because its saved snapshot failed integrity validation"
                )

        requested_fingerprint = production_indexing_contract_fingerprint(self.config)
        migrations_path = self.work_dir / "config_migrations.json"
        migrations = load_json_safe(migrations_path, [])
        if not isinstance(migrations, list):
            raise RunConfigMismatchError(
                "Cannot migrate incomplete run config because its migration audit is invalid"
            )
        migration_reason = (
            "explicit_full_restart_config_migration"
            if has_completed_stages
            else "explicit_incomplete_run_config_migration"
        )
        migrations.append(
            {
                "migrated_at": now_iso(),
                "run_id": self.run_id,
                "state_status": state.status,
                "current_stage_index": state.current_stage_index,
                "saved_fingerprint": recorded_fingerprint,
                "requested_fingerprint": requested_fingerprint,
                "restart_from_index": restart_from_index,
                "reason": migration_reason,
            }
        )
        atomic_write_json(migrations_path, migrations)
        self._log(
            "warning",
            "Migrating unfinished non-production run %s to a new audited config fingerprint",
            self.run_id,
        )

    def _validate_fresh_production_run_directory(self) -> None:
        if not bool(self._pipeline_config().get("production_profile", False)):
            return
        sentinels = (
            self.work_dir / "resolved_config.json",
            self.work_dir / "pipeline_state.json",
            self.work_dir / "artifact_catalog.json",
            self.work_dir / "stage_outputs",
        )
        existing = [path for path in sentinels if path.exists()]
        if existing:
            raise RunConfigMismatchError(
                "Production run directory is already initialized; use --resume with the matching config "
                "or create a new run ID. Existing paths: "
                + ", ".join(str(path) for path in existing)
            )

    def _run_integrity_audit(
        self,
        *,
        state: PipelineState,
        artifact_catalog,
        stage_id: Optional[str] = None,
    ) -> None:
        pipeline_config = self._pipeline_config()
        report = audit_run(self.work_dir, state=state, artifact_catalog=artifact_catalog)
        audit_path = save_run_audit(report, self.work_dir)

        if report.warnings:
            self._log(
                "warning",
                "Run audit completed with %d warning(s) after %s: %s",
                len(report.warnings),
                stage_id or "pipeline",
                audit_path,
            )

        if report.ok or not pipeline_config.get("fail_on_audit_error", True):
            return

        first_error = report.errors[0]
        raise RunAuditError(
            f"Integrity audit failed after {stage_id or 'pipeline'} with "
            f"{len(report.errors)} error(s): {first_error.code} - {first_error.message}"
        )

    def _log(self, level: str, msg: str, *args: Any) -> None:
        formatted = msg % args if args else msg
        getattr(logger, level)(formatted)
        if self._on_log:
            self._on_log(level, formatted)

    def _build_stages(self) -> None:
        """Instantiate stage objects from config."""
        auto_discover()

        stage_defs = self.config.get("stages", [])
        if not stage_defs:
            raise ValueError("No stages defined in config")

        self._stages = []
        for i, stage_def in enumerate(stage_defs):
            stage_type = stage_def.get("type")
            plugin_name = stage_def.get("plugin")
            if not stage_type or not plugin_name:
                raise ValueError(
                    f"Stage {i} must have 'type' and 'plugin' keys, got: {stage_def}"
                )
            stage_id = str(stage_def.get("id") or f"{stage_type}_{plugin_name}_{i}")
            cls = get_stage(stage_type, plugin_name)
            instance = cls()
            self._stages.append((stage_type, plugin_name, stage_id, stage_def, instance))
            self._log("debug", "Loaded stage %d: %s/%s (%s)", i, stage_type, plugin_name, stage_id)

    async def validate(self) -> Dict[str, List[str]]:
        """Validate config for all stages. Returns {stage_name: [errors]}."""
        self._build_stages()
        errors: Dict[str, List[str]] = {}
        for stage_type, plugin_name, stage_id, stage_def, instance in self._stages:
            key = f"{stage_type}/{plugin_name}"
            stage_errors = await instance.validate_config(self.config)
            if stage_errors:
                errors[key] = stage_errors
        return errors

    async def dry_run(self) -> List[Dict[str, str]]:
        """Return the planned execution order without running anything."""
        self._build_stages()
        return [
            {
                "index": i,
                "id": stage_id,
                "type": stage_type,
                "plugin": plugin_name,
                "description": instance.description or "",
            }
            for i, (stage_type, plugin_name, stage_id, stage_def, instance) in enumerate(self._stages)
        ]

    async def run(
        self,
        resume: bool = False,
        restart_from: Optional[str] = None,
        stop_after_stage: Optional[str] = None,
        migrate_incomplete_config: bool = False,
    ) -> PipelineState:
        """Execute the pipeline, optionally pausing after a selected stage.

        Args:
            resume: If True, load existing state and skip completed stages.
            restart_from: Reset the selected stage and every downstream stage before running.
            stop_after_stage: Execute through this stage (inclusive), then persist a resumable
                ``paused`` state without running the full-run completion audit.

        Returns:
            Final PipelineState with all stage results.
        """
        self._build_stages()
        # Resolve execution boundaries before creating a work directory, taking
        # a lock, loading artifacts, or doing any stage/provider work.
        restart_from_index = (
            self._resolve_stage_index(restart_from)
            if restart_from is not None
            else None
        )
        stop_after_index = (
            self._resolve_stage_index(stop_after_stage)
            if stop_after_stage is not None
            else None
        )
        if (
            restart_from_index is not None
            and stop_after_index is not None
            and restart_from_index > stop_after_index
        ):
            raise ValueError(
                "--stop-after-stage must select the same stage as, or a stage after, "
                "--restart-from-stage"
            )
        ensure_dir(self.work_dir)
        self._acquire_run_lock()

        try:
            if not resume and not restart_from:
                self._validate_fresh_production_run_directory()
                self._raise_if_active_upload_would_run(state=None)
            elif migrate_incomplete_config:
                self._migrate_incomplete_config_snapshot(
                    restart_from_index=restart_from_index,
                )
            else:
                self._validate_resume_config_snapshot()

            # Load or create state
            state = None
            artifact_catalog = load_artifact_catalog(self.work_dir)
            removed_missing_artifacts = artifact_catalog.prune_missing_local_paths()
            if removed_missing_artifacts:
                self._log(
                    "warning",
                    "Pruned %d missing artifact records from catalog",
                    len(removed_missing_artifacts),
                )

            if resume or restart_from:
                state = load_state(self.work_dir)
                if state:
                    self._align_state_to_stages(state)
                    self._log("info", "Resuming run %s from stage %d",
                              state.run_id, state.current_stage_index)
                    removed_stale_stage_artifact_ids = reconcile_state_artifact_ids(state, artifact_catalog)
                    if removed_stale_stage_artifact_ids:
                        self._log(
                            "info",
                            "Pruned %d stale stage artifact id references from state",
                            removed_stale_stage_artifact_ids,
                        )

                    inconsistent_index = self._first_inconsistent_stage_index(state)
                    if inconsistent_index is not None:
                        self._raise_if_active_upload_would_run(
                            state=state,
                            restart_index=inconsistent_index,
                        )
                        self._log(
                            "warning",
                            "State has completed stages after unfinished stage %d; resetting downstream",
                            inconsistent_index,
                        )
                        self._reset_from_stage(state, artifact_catalog, inconsistent_index)

                    if restart_from:
                        self._raise_if_active_upload_would_run(
                            state=state,
                            restart_index=restart_from_index,
                        )
                        self._log(
                            "info",
                            "Restarting run %s from stage %d",
                            state.run_id,
                            restart_from_index,
                        )
                        self._reset_from_stage(state, artifact_catalog, restart_from_index)
                    else:
                        self._raise_if_active_upload_would_run(state=state)

            if restart_from and not state:
                raise ValueError(
                    f"Cannot restart from stage {restart_from!r}: no existing state found in {self.work_dir}"
                )

            self._write_config_snapshot()
            if not state:
                state = PipelineState(
                    run_id=self.run_id,
                    project_name=self.project_name,
                    status="running",
                    started_at=now_iso(),
                    stages=[
                        StageState(name=name, stage_type=stype, stage_id=stage_id, status="pending")
                        for stype, name, stage_id, _stage_def, _instance in self._stages
                    ],
                )

            state.status = "running"
            state.finished_at = None
            save_state(state, self.work_dir)
            save_artifact_catalog(artifact_catalog, self.work_dir)

            # Collect outputs from all completed stages for context threading
            previous_outputs: Dict[str, Any] = {}
            for index, ss in enumerate(state.stages):
                if ss.status == "completed":
                    previous_outputs.update(ss.outputs)
                    completed_stage_id = ss.stage_id or f"{ss.stage_type}_{ss.name}_{index}"
                    previous_outputs.setdefault("stage_outputs", {})[completed_stage_id] = ss.outputs

            # Execute stages, bounded inclusively when an intentional checkpoint
            # was requested. If the target was already completed, the empty range
            # below simply re-persists the paused checkpoint without rerunning it.
            execution_end = (
                stop_after_index + 1
                if stop_after_index is not None
                else len(self._stages)
            )
            for i in range(state.current_stage_index, execution_end):
                stage_type, plugin_name, stage_id, stage_def, instance = self._stages[i]
                stage_state = state.stages[i]
                stage_key = f"{stage_type}/{plugin_name}"

                # Skip already completed stages (resume mode)
                if stage_state.status == "completed":
                    self._log("info", "Skipping completed stage %d: %s", i, stage_key)
                    previous_outputs.update(stage_state.outputs)
                    previous_outputs.setdefault("stage_outputs", {})[stage_id] = stage_state.outputs
                    continue

                self._log("info", "Running stage %d/%d: %s",
                          i + 1, len(self._stages), stage_key)

                if self._on_stage_start:
                    self._on_stage_start(stage_type, plugin_name, None)

                # Build context for this stage
                ctx = StageContext(
                    run_id=self.run_id,
                    project_name=self.project_name,
                    config=self.config,
                    work_dir=self.work_dir,
                    previous_outputs=previous_outputs.copy(),
                    checkpoint=stage_state.checkpoint,
                    stage_definition=stage_def,
                    stage_index=i,
                    stage_id=stage_id,
                    artifact_catalog=artifact_catalog,
                )

                # Execute
                stage_state.status = "running"
                stage_state.stage_id = stage_id
                stage_state.started_at = now_iso()
                stage_state.finished_at = None
                stage_state.error_message = None
                stage_state.metrics = {}
                stage_state.outputs = {}
                state.current_stage_index = i
                save_state(state, self.work_dir)

                try:
                    result = await instance.execute(ctx)
                except Exception as e:
                    self._log("error", "Stage %s failed with exception: %s", stage_key, e)
                    result = StageResult.failure(str(e))
                finally:
                    try:
                        await instance.cleanup(ctx)
                    except Exception as cleanup_err:
                        self._log("warning", "Cleanup failed for %s: %s",
                                  stage_key, cleanup_err)

                # Record result
                stage_state.status = result.status.value
                stage_state.finished_at = now_iso()
                stage_state.outputs = result.outputs
                stage_state.metrics = result.metrics
                stage_state.error_message = result.error_message
                stage_state.checkpoint = result.checkpoint
                stage_state.artifact_ids = []

                if result.removed_artifact_ids:
                    artifact_catalog.remove_many(result.removed_artifact_ids)
                    reconcile_state_artifact_ids(state, artifact_catalog)
                if result.artifacts:
                    added = artifact_catalog.extend(result.artifacts)
                    stage_state.artifact_ids = [record.artifact_id for record in added]

                save_state(state, self.work_dir)
                save_artifact_catalog(artifact_catalog, self.work_dir)

                if result.status == StageStatus.COMPLETED and self._pipeline_config().get("audit_on_stage_complete", True):
                    self._run_integrity_audit(
                        state=state,
                        artifact_catalog=artifact_catalog,
                        stage_id=stage_id,
                    )

                if self._on_stage_complete:
                    self._on_stage_complete(stage_type, plugin_name, {
                        "status": result.status.value,
                        "metrics": result.metrics,
                    })

                # Handle stage failure
                if result.status == StageStatus.FAILED:
                    self._log("error", "Pipeline stopped: stage %s failed — %s",
                              stage_key, result.error_message)
                    state.status = "failed"
                    state.finished_at = now_iso()
                    save_state(state, self.work_dir)
                    return state

                # Thread outputs forward
                if result.status == StageStatus.COMPLETED:
                    previous_outputs.update(result.outputs)
                    previous_outputs.setdefault("stage_outputs", {})[stage_id] = result.outputs

                self._log(
                    "info",
                    "Stage %s completed: %s",
                    stage_key,
                    _metrics_for_log(result.metrics),
                )

            if stop_after_index is not None:
                _stage_type, _plugin_name, stop_stage_id, _stage_def, _instance = self._stages[
                    stop_after_index
                ]
                stop_stage_state = state.stages[stop_after_index]
                if not stop_stage_state.is_terminal:
                    raise RuntimeError(
                        f"Cannot pause after unfinished stage {stop_stage_id!r} "
                        f"(status={stop_stage_state.status!r})"
                    )
                state.status = "paused"
                state.finished_at = None
                state.current_stage_index = min(stop_after_index + 1, len(self._stages))
                save_state(state, self.work_dir)
                self._log(
                    "info",
                    "Pipeline run %s intentionally paused after stage %s; resume from stage %d",
                    self.run_id,
                    stop_stage_id,
                    state.current_stage_index,
                )
                return state

            # All stages completed
            state.status = "completed"
            state.finished_at = now_iso()
            state.current_stage_index = len(self._stages)
            save_state(state, self.work_dir)
            if self._pipeline_config().get("audit_on_run_complete", True):
                self._run_integrity_audit(state=state, artifact_catalog=artifact_catalog)
            self._log("info", "Pipeline run %s completed successfully", self.run_id)
            return state
        except RunAuditError as exc:
            self._log("error", "%s", exc)
            state = load_state(self.work_dir) or state
            if state and state.stages:
                index = min(max(state.current_stage_index, 0), len(state.stages) - 1)
                stage_state = state.stages[index]
                stage_state.status = "failed"
                stage_state.finished_at = now_iso()
                stage_state.error_message = str(exc)
            if state:
                state.status = "failed"
                state.finished_at = now_iso()
                save_state(state, self.work_dir)
                save_artifact_catalog(artifact_catalog, self.work_dir)
                return state
            raise
        finally:
            self._release_run_lock()
