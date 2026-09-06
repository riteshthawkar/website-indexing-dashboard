from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List

import pytest

from lawa_rag import (
    IndexingPipeline,
    PipelineRun,
    PipelineStage,
    StageContext,
    StageResult,
    register_stage,
)


@register_stage
class PackageBehaviorStage(PipelineStage):
    name = "package_behavior_test"
    stage_type = "package_behavior"
    description = "Exercise package execution behavior and failure handling."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        value = config.get("package_behavior") or {}
        return [] if value.get("enabled") else ["package_behavior.enabled must be true"]

    async def execute(self, ctx: StageContext) -> StageResult:
        mode = str(ctx.stage_config.get("mode") or "success")
        if mode == "raise":
            raise RuntimeError("intentional provider exception")
        if mode == "failure":
            return StageResult.failure(
                "intentional stage failure",
                outputs={"failed_stage_id": ctx.stage_id},
                metrics={"attempted": 1},
            )
        if mode == "skip":
            return StageResult.skipped("intentionally skipped")

        prior = ctx.get_stage_outputs(str(ctx.stage_config.get("read_stage") or ""))
        output_file = ctx.output_dir("behavior") / "output.txt"
        output_file.write_text(ctx.stage_id, encoding="utf-8")
        return StageResult.success(
            outputs={
                "stage_id": ctx.stage_id,
                "output_file": str(output_file),
                "prior_stage_id": str(prior.get("stage_id") or ""),
            },
            metrics={"records": 1},
        )

    async def cleanup(self, ctx: StageContext) -> None:
        cleanup_file = ctx.work_dir / f"{ctx.stage_id}.cleanup"
        cleanup_file.write_text("clean", encoding="utf-8")


def _behavior_config(
    tmp_path: Path,
    *,
    stage_definitions: list[dict[str, Any]] | None = None,
    enabled: bool = True,
) -> dict[str, Any]:
    return {
        "project_name": "package_behavior",
        "work_dir": str(tmp_path / "runs"),
        "pipeline": {
            "audit_on_stage_complete": False,
            "audit_on_run_complete": False,
        },
        "package_behavior": {"enabled": enabled, "mode": "success"},
        "stages": stage_definitions
        or [
            {
                "id": "behavior",
                "type": "package_behavior",
                "plugin": "package_behavior_test",
            }
        ],
    }


def test_runner_properties_plan_callbacks_and_success_result(tmp_path: Path) -> None:
    starts: list[tuple[str, str, object]] = []
    completions: list[tuple[str, str, object]] = []
    logs: list[tuple[str, str]] = []
    pipeline = IndexingPipeline.from_config(
        _behavior_config(tmp_path),
        run_id="behavior-success",
        on_stage_start=lambda stage_type, name, info: starts.append((stage_type, name, info)),
        on_stage_complete=lambda stage_type, name, info: completions.append(
            (stage_type, name, info)
        ),
        on_log=lambda level, message: logs.append((level, message)),
    )

    assert pipeline.run_id == "behavior-success"
    assert pipeline.project_name == "package_behavior"
    assert pipeline.work_dir == (
        tmp_path / "runs" / "package_behavior" / "behavior-success"
    ).resolve()
    assert [stage.__dict__ for stage in pipeline.plan_sync()] == [
        {
            "index": 0,
            "id": "behavior",
            "type": "package_behavior",
            "plugin": "package_behavior_test",
            "description": "Exercise package execution behavior and failure handling.",
        }
    ]

    result = pipeline.run_sync()

    assert result.succeeded
    assert not result.paused
    assert result.failed_stage is None
    assert result.stages[0].outputs["stage_id"] == "behavior"
    assert result.stages[0].metrics == {"records": 1}
    assert Path(str(result.stages[0].outputs["output_file"])).read_text() == "behavior"
    assert (result.work_dir / "behavior.cleanup").read_text() == "clean"
    assert starts == [("package_behavior", "package_behavior_test", None)]
    assert completions[0][:2] == ("package_behavior", "package_behavior_test")
    assert completions[0][2] == {"status": "completed", "metrics": {"records": 1}}
    assert any("Running stage" in message for _level, message in logs)


def test_stage_outputs_flow_through_the_public_runner(tmp_path: Path) -> None:
    config = _behavior_config(
        tmp_path,
        stage_definitions=[
            {
                "id": "first",
                "type": "package_behavior",
                "plugin": "package_behavior_test",
            },
            {
                "id": "second",
                "type": "package_behavior",
                "plugin": "package_behavior_test",
                "config": {"read_stage": "first"},
            },
        ],
    )

    result = IndexingPipeline.from_config(config, run_id="flow").run_sync()

    assert result.succeeded
    assert result.stages[1].outputs["prior_stage_id"] == "first"


@pytest.mark.parametrize(
    ("mode", "expected_message"),
    [
        ("failure", "intentional stage failure"),
        ("raise", "intentional provider exception"),
    ],
)
def test_stage_failures_are_returned_as_inspectable_run_results(
    tmp_path: Path,
    mode: str,
    expected_message: str,
) -> None:
    config = _behavior_config(
        tmp_path,
        stage_definitions=[
            {
                "id": "will_fail",
                "type": "package_behavior",
                "plugin": "package_behavior_test",
                "config": {"mode": mode},
            },
            {
                "id": "must_not_run",
                "type": "package_behavior",
                "plugin": "package_behavior_test",
            },
        ],
    )

    result = IndexingPipeline.from_config(config, run_id=f"failure-{mode}").run_sync()

    assert result.status == "failed"
    assert not result.succeeded
    assert result.failed_stage is result.stages[0]
    assert result.failed_stage.error_message == expected_message
    assert result.stages[1].status == "pending"
    assert (result.work_dir / "will_fail.cleanup").read_text() == "clean"
    assert not (result.work_dir / "must_not_run.cleanup").exists()


def test_skipped_stage_is_successful_and_pipeline_continues(tmp_path: Path) -> None:
    config = _behavior_config(
        tmp_path,
        stage_definitions=[
            {
                "id": "optional",
                "type": "package_behavior",
                "plugin": "package_behavior_test",
                "config": {"mode": "skip"},
            },
            {
                "id": "required",
                "type": "package_behavior",
                "plugin": "package_behavior_test",
            },
        ],
    )

    result = IndexingPipeline.from_config(config, run_id="skip").run_sync()

    assert result.succeeded
    assert result.stages[0].status == "skipped"
    assert result.stages[0].succeeded
    assert result.stages[0].error_message == "intentionally skipped"
    assert result.stages[1].status == "completed"


def test_async_plan_validation_and_execution(tmp_path: Path) -> None:
    async def exercise() -> PipelineRun:
        pipeline = IndexingPipeline.from_config(
            _behavior_config(tmp_path),
            run_id="async",
        )
        assert (await pipeline.validate()).ok
        assert (await pipeline.plan())[0].id == "behavior"
        return await pipeline.run()

    assert asyncio.run(exercise()).succeeded


def test_sync_methods_fail_clearly_inside_an_event_loop(tmp_path: Path) -> None:
    async def exercise() -> None:
        pipeline = IndexingPipeline.from_config(
            _behavior_config(tmp_path),
            run_id="loop-guard",
        )
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.plan_sync()
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.validate_sync()
        with pytest.raises(RuntimeError, match="already running"):
            pipeline.run_sync()

    asyncio.run(exercise())


def test_validation_can_be_inspected_without_creating_a_run_directory(tmp_path: Path) -> None:
    pipeline = IndexingPipeline.from_config(
        _behavior_config(tmp_path, enabled=False),
        run_id="validation-only",
    )

    report = pipeline.validate_sync()

    assert not report.ok
    assert report.to_dict() == {
        "package_behavior/package_behavior_test": [
            "package_behavior.enabled must be true"
        ]
    }
    assert not pipeline.work_dir.exists()


def test_loading_missing_or_incomplete_checkpoints_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No pipeline state"):
        PipelineRun.load(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="No pipeline state"):
        IndexingPipeline.from_run(tmp_path / "missing")

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "pipeline_state.json").write_text(
        '{"run_id":"r1","project_name":"p1","stages":[]}',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="No resolved pipeline config"):
        IndexingPipeline.from_run(incomplete)


def test_run_and_stage_serialization_return_detached_data(tmp_path: Path) -> None:
    result = IndexingPipeline.from_config(
        _behavior_config(tmp_path),
        run_id="serialization",
    ).run_sync()

    serialized = result.to_dict()
    serialized["stages"][0]["outputs"]["stage_id"] = "mutated"
    serialized["stages"][0]["metrics"]["records"] = 999

    assert result.stages[0].outputs["stage_id"] == "behavior"
    assert result.stages[0].metrics["records"] == 1
