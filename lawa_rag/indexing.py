"""High-level, resumable indexing API.

This module deliberately delegates execution to the existing production
orchestrator. It provides a smaller, typed boundary without forking pipeline
behavior or changing the CLI and deployment imports.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar

from pipeline.core.artifacts import ArtifactCatalog, load_artifact_catalog
from pipeline.core.config import load_resolved_run_config
from pipeline.core.orchestrator import (
    LogCallback,
    PipelineOrchestrator,
    StageCallback,
)
from pipeline.core.state import PipelineState, StageState, load_state

from .config import ConfigSource, load_pipeline_config


_T = TypeVar("_T")


@dataclass(frozen=True)
class PipelinePlanStage:
    """One stage in the resolved execution plan."""

    index: int
    id: str
    type: str
    plugin: str
    description: str = ""

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelinePlanStage":
        return cls(
            index=int(value.get("index", 0)),
            id=str(value.get("id") or ""),
            type=str(value.get("type") or ""),
            plugin=str(value.get("plugin") or ""),
            description=str(value.get("description") or ""),
        )


@dataclass(frozen=True)
class PipelineValidationReport:
    """Detached validation result suitable for APIs and dashboards."""

    errors: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, list[str]]:
        return {name: list(messages) for name, messages in self.errors.items()}


class PipelineValidationError(ValueError):
    """Raised when execution is requested with an invalid stage config."""

    def __init__(self, report: PipelineValidationReport):
        self.report = report
        details = "; ".join(
            f"{stage}: {', '.join(messages)}"
            for stage, messages in report.errors.items()
        )
        super().__init__(f"Pipeline configuration is invalid: {details}")


@dataclass(frozen=True)
class StageRun:
    """Serializable snapshot of one stage execution."""

    id: str
    name: str
    type: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    outputs: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    error_message: str | None = None
    artifact_ids: tuple[str, ...] = ()

    @classmethod
    def from_state(cls, value: StageState) -> "StageRun":
        return cls(
            id=str(value.stage_id or ""),
            name=str(value.name),
            type=str(value.stage_type),
            status=str(value.status),
            started_at=value.started_at,
            finished_at=value.finished_at,
            outputs=deepcopy(value.outputs),
            metrics=deepcopy(value.metrics),
            error_message=value.error_message,
            artifact_ids=tuple(value.artifact_ids),
        )

    @property
    def succeeded(self) -> bool:
        return self.status in {"completed", "skipped"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outputs": deepcopy(dict(self.outputs)),
            "metrics": deepcopy(dict(self.metrics)),
            "error_message": self.error_message,
            "artifact_ids": list(self.artifact_ids),
        }


@dataclass(frozen=True)
class PipelineRun:
    """Detached result of a pipeline run or persisted checkpoint."""

    run_id: str
    project_name: str
    status: str
    work_dir: Path
    stages: tuple[StageRun, ...]
    current_stage_index: int = 0
    started_at: str | None = None
    finished_at: str | None = None

    @classmethod
    def from_state(cls, state: PipelineState, work_dir: str | Path) -> "PipelineRun":
        return cls(
            run_id=state.run_id,
            project_name=state.project_name,
            status=state.status,
            work_dir=Path(work_dir).expanduser().resolve(),
            stages=tuple(StageRun.from_state(stage) for stage in state.stages),
            current_stage_index=state.current_stage_index,
            started_at=state.started_at,
            finished_at=state.finished_at,
        )

    @classmethod
    def load(cls, work_dir: str | Path) -> "PipelineRun":
        """Load a previously persisted run without initializing providers."""

        resolved = Path(work_dir).expanduser().resolve()
        state = load_state(resolved)
        if state is None:
            raise FileNotFoundError(f"No pipeline state found in {resolved}")
        return cls.from_state(state, resolved)

    @property
    def succeeded(self) -> bool:
        return self.status == "completed"

    @property
    def paused(self) -> bool:
        return self.status == "paused"

    @property
    def failed_stage(self) -> StageRun | None:
        return next((stage for stage in self.stages if stage.status == "failed"), None)

    def artifacts(self) -> ArtifactCatalog:
        """Load the artifact catalog associated with this checkpoint."""

        return load_artifact_catalog(self.work_dir)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "project_name": self.project_name,
            "status": self.status,
            "work_dir": str(self.work_dir),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_stage_index": self.current_stage_index,
            "stages": [stage.to_dict() for stage in self.stages],
        }


def _run_synchronously(operation: Callable[[], Any]) -> _T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(operation())
    raise RuntimeError(
        "A Python event loop is already running; use the corresponding async method with await"
    )


class IndexingPipeline:
    """Configured pipeline runner with safe validation and resume support."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        work_dir: str | Path | None = None,
        run_id: str | None = None,
        on_stage_start: StageCallback | None = None,
        on_stage_complete: StageCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> None:
        self._config = deepcopy(dict(config))
        self._orchestrator = PipelineOrchestrator(
            self._config,
            work_dir=Path(work_dir).expanduser() if work_dir is not None else None,
            run_id=run_id,
            on_stage_start=on_stage_start,
            on_stage_complete=on_stage_complete,
            on_log=on_log,
        )

    @classmethod
    def from_config(
        cls,
        source: ConfigSource,
        *,
        search_dirs: list[str | Path] | None = None,
        overrides: Mapping[str, Any] | None = None,
        work_dir: str | Path | None = None,
        run_id: str | None = None,
        on_stage_start: StageCallback | None = None,
        on_stage_complete: StageCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> "IndexingPipeline":
        config = load_pipeline_config(
            source,
            search_dirs=search_dirs,
            overrides=overrides,
        )
        return cls(
            config,
            work_dir=work_dir,
            run_id=run_id,
            on_stage_start=on_stage_start,
            on_stage_complete=on_stage_complete,
            on_log=on_log,
        )

    @classmethod
    def from_run(
        cls,
        work_dir: str | Path,
        *,
        config: ConfigSource | None = None,
        search_dirs: list[str | Path] | None = None,
        overrides: Mapping[str, Any] | None = None,
        on_stage_start: StageCallback | None = None,
        on_stage_complete: StageCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> "IndexingPipeline":
        """Create a runner for an existing checkpoint.

        When ``config`` is omitted, the immutable resolved config saved in the
        run directory is used. A named production config can be supplied to
        enforce the current production contract during resume.
        """

        resolved = Path(work_dir).expanduser().resolve()
        state = load_state(resolved)
        if state is None:
            raise FileNotFoundError(f"No pipeline state found in {resolved}")
        if config is None:
            saved_config = load_resolved_run_config(resolved)
            if saved_config is None:
                raise FileNotFoundError(f"No resolved pipeline config found in {resolved}")
            source: ConfigSource = saved_config
        else:
            source = config
        return cls.from_config(
            source,
            search_dirs=search_dirs,
            overrides=overrides,
            work_dir=resolved,
            run_id=state.run_id,
            on_stage_start=on_stage_start,
            on_stage_complete=on_stage_complete,
            on_log=on_log,
        )

    @property
    def config(self) -> dict[str, Any]:
        return deepcopy(self._config)

    @property
    def run_id(self) -> str:
        return self._orchestrator.run_id

    @property
    def project_name(self) -> str:
        return self._orchestrator.project_name

    @property
    def work_dir(self) -> Path:
        return self._orchestrator.work_dir

    async def plan(self) -> tuple[PipelinePlanStage, ...]:
        stages = await self._orchestrator.dry_run()
        return tuple(PipelinePlanStage.from_dict(stage) for stage in stages)

    def plan_sync(self) -> tuple[PipelinePlanStage, ...]:
        return _run_synchronously(self.plan)

    async def validate(self) -> PipelineValidationReport:
        errors = await self._orchestrator.validate()
        return PipelineValidationReport(
            errors={name: tuple(messages) for name, messages in errors.items()}
        )

    def validate_sync(self) -> PipelineValidationReport:
        return _run_synchronously(self.validate)

    async def run(
        self,
        *,
        resume: bool = False,
        restart_from: str | None = None,
        stop_after_stage: str | None = None,
        migrate_incomplete_config: bool = False,
        validate: bool = True,
    ) -> PipelineRun:
        if validate:
            report = await self.validate()
            if not report.ok:
                raise PipelineValidationError(report)
        state = await self._orchestrator.run(
            resume=resume,
            restart_from=restart_from,
            stop_after_stage=stop_after_stage,
            migrate_incomplete_config=migrate_incomplete_config,
        )
        return PipelineRun.from_state(state, self.work_dir)

    def run_sync(
        self,
        *,
        resume: bool = False,
        restart_from: str | None = None,
        stop_after_stage: str | None = None,
        migrate_incomplete_config: bool = False,
        validate: bool = True,
    ) -> PipelineRun:
        return _run_synchronously(
            lambda: self.run(
                resume=resume,
                restart_from=restart_from,
                stop_after_stage=stop_after_stage,
                migrate_incomplete_config=migrate_incomplete_config,
                validate=validate,
            )
        )


__all__ = [
    "IndexingPipeline",
    "PipelinePlanStage",
    "PipelineRun",
    "PipelineValidationError",
    "PipelineValidationReport",
    "StageRun",
]
