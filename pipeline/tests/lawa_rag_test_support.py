"""Shared, provider-free stages used by lawa-rag package tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from lawa_rag import PipelineStage, StageContext, StageResult, register_stage


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


def behavior_config(
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
