"""Validated, resumable execution facade for the production orchestrator."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Coroutine, Mapping, Sequence, TypeVar

from pipeline.core.config import load_resolved_run_config
from pipeline.core.orchestrator import LogCallback, PipelineOrchestrator, StageCallback
from pipeline.core.state import load_state

from .config import ConfigSource, load_pipeline_config
from .results import (
    PipelinePlanStage,
    PipelineRun,
    PipelineValidationReport,
)
from .website_config import build_website_config
from .websites import WebsiteProfile


_T = TypeVar("_T")


class PipelineValidationError(ValueError):
    """Raised when execution is requested with an invalid stage config."""

    def __init__(self, report: PipelineValidationReport):
        self.report = report
        details = "; ".join(
            f"{stage}: {', '.join(messages)}"
            for stage, messages in report.errors.items()
        )
        super().__init__(f"Pipeline configuration is invalid: {details}")


def _run_synchronously(
    operation: Callable[[], Coroutine[Any, Any, _T]],
) -> _T:
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
    def for_website(
        cls,
        profile: WebsiteProfile,
        *,
        overrides: Mapping[str, Any] | None = None,
        **runner_options: Any,
    ) -> "IndexingPipeline":
        """Create the canonical generic website pipeline for ``profile``."""

        config = build_website_config(profile, overrides=overrides)
        return cls(config, **runner_options)

    @classmethod
    def from_config(
        cls,
        source: ConfigSource,
        *,
        search_dirs: Sequence[str | Path] | None = None,
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
        search_dirs: Sequence[str | Path] | None = None,
        overrides: Mapping[str, Any] | None = None,
        on_stage_start: StageCallback | None = None,
        on_stage_complete: StageCallback | None = None,
        on_log: LogCallback | None = None,
    ) -> "IndexingPipeline":
        """Create a runner for an existing checkpoint."""

        resolved = Path(work_dir).expanduser().resolve()
        state = load_state(resolved)
        if state is None:
            raise FileNotFoundError(f"No pipeline state found in {resolved}")

        if config is None:
            source = load_resolved_run_config(resolved)
            if source is None:
                raise FileNotFoundError(f"No resolved pipeline config found in {resolved}")
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


__all__ = ["IndexingPipeline", "PipelineValidationError"]
