"""Small, serializable result models returned by package runners."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from pipeline.core.artifacts import ArtifactCatalog, load_artifact_catalog
from pipeline.core.state import PipelineState, StageState, load_state


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


__all__ = [
    "PipelinePlanStage",
    "PipelineRun",
    "PipelineValidationReport",
    "StageRun",
]
