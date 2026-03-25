"""
Checkpoint / state persistence for pipeline runs.

After each stage completes, the orchestrator saves a snapshot so that
a crashed or interrupted run can resume from the last completed stage.
"""

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io import atomic_write_json, load_json_safe

logger = logging.getLogger(__name__)

STATE_FILENAME = "pipeline_state.json"


@dataclass
class StageState:
    """Persisted state for a single stage execution."""

    name: str
    stage_type: str
    status: str  # "pending" | "completed" | "failed" | "skipped"
    stage_id: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    checkpoint: Optional[Dict] = None
    artifact_ids: List[str] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "skipped"}

    def reset_for_rerun(self) -> "StageState":
        self.status = "pending"
        self.started_at = None
        self.finished_at = None
        self.outputs = {}
        self.metrics = {}
        self.error_message = None
        self.checkpoint = None
        self.artifact_ids = []
        return self

    def normalize(self) -> "StageState":
        if self.status in {"pending", "running"}:
            self.finished_at = None
        if self.status == "pending":
            self.error_message = None
        if self.status == "running":
            self.error_message = None
        return self


@dataclass
class PipelineState:
    """Full persisted state for a pipeline run."""

    run_id: str
    project_name: str
    status: str = "pending"  # "pending" | "running" | "completed" | "failed"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    stages: List[StageState] = field(default_factory=list)
    current_stage_index: int = 0

    def normalize(self) -> "PipelineState":
        self.stages = [stage.normalize() for stage in self.stages]

        if any(stage.status == "running" for stage in self.stages):
            self.status = "running"
            self.finished_at = None
        elif self.status in {"pending", "running"}:
            self.finished_at = None

        first_non_terminal = next(
            (index for index, stage in enumerate(self.stages) if not stage.is_terminal),
            None,
        )
        if first_non_terminal is not None:
            self.current_stage_index = first_non_terminal
            for stage in self.stages[first_non_terminal + 1 :]:
                if stage.is_terminal or stage.status in {"running", "failed"}:
                    stage.reset_for_rerun()

        if self.current_stage_index < 0:
            self.current_stage_index = 0
        if self.stages:
            self.current_stage_index = min(self.current_stage_index, len(self.stages))

        return self

    def to_dict(self) -> Dict[str, Any]:
        self.normalize()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineState":
        stages = [StageState(**s) for s in data.pop("stages", [])]
        return cls(stages=stages, **data)


def save_state(state: PipelineState, work_dir: Path) -> None:
    """Persist pipeline state to disk."""
    path = Path(work_dir) / STATE_FILENAME
    state.normalize()
    atomic_write_json(path, state.to_dict())
    logger.debug("Saved pipeline state to %s", path)


def load_state(work_dir: Path) -> Optional[PipelineState]:
    """Load pipeline state from disk, or None if no state file exists."""
    path = Path(work_dir) / STATE_FILENAME
    data = load_json_safe(path)
    if data is None:
        return None
    try:
        return PipelineState.from_dict(data).normalize()
    except (TypeError, KeyError) as e:
        logger.warning("Corrupt pipeline state at %s: %s", path, e)
        return None


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
