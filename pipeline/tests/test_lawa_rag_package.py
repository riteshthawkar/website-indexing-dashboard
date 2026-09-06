from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

import lawa_rag
from lawa_rag import (
    IndexingPipeline,
    PipelineRun,
    PipelineStage,
    PipelineValidationError,
    ReleaseContractError,
    StageContext,
    StageResult,
    load_active_release,
    load_pipeline_config,
    register_stage,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@register_stage
class PackageContractStage(PipelineStage):
    name = "package_contract_test"
    stage_type = "package_contract"
    description = "Exercise the stable package runner without external providers."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        stage_config = config.get("package_contract") or {}
        return [] if stage_config.get("message") else ["package_contract.message is required"]

    async def execute(self, ctx: StageContext) -> StageResult:
        output_file = ctx.output_dir("result") / "message.txt"
        output_file.write_text(str(ctx.stage_config["message"]), encoding="utf-8")
        artifact = ctx.make_artifact(
            output_file,
            artifact_type="package_contract_result",
            role="test_output",
        )
        return StageResult.success(
            outputs={"message_file": str(output_file)},
            metrics={"messages": 1},
            artifacts=[artifact],
        )


def _config(tmp_path: Path, *, stages: int = 1) -> Dict[str, Any]:
    return {
        "project_name": "package_contract",
        "work_dir": str(tmp_path / "runs"),
        "pipeline": {
            "audit_on_stage_complete": False,
            "audit_on_run_complete": False,
        },
        "package_contract": {"message": "hello from lawa_rag"},
        "stages": [
            {
                "id": f"contract_{index + 1}",
                "type": "package_contract",
                "plugin": "package_contract_test",
            }
            for index in range(stages)
        ],
    }


def _requirement_lines(path: Path) -> set[str]:
    values: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            values.update(_requirement_lines((path.parent / line[3:].strip()).resolve()))
            continue
        values.add(line.casefold())
    return values


def test_public_version_matches_distribution_metadata() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert lawa_rag.__version__ == metadata["project"]["version"]


def test_package_extras_match_canonical_runtime_requirements() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {value.casefold() for value in metadata["project"]["dependencies"]}
    extras = metadata["project"]["optional-dependencies"]

    assert base | {value.casefold() for value in extras["indexing"]} == _requirement_lines(
        PROJECT_ROOT / "requirements.txt"
    )
    assert base | {value.casefold() for value in extras["retrieval"]} == _requirement_lines(
        PROJECT_ROOT / "requirements-retriever.txt"
    )


def test_config_mapping_and_returned_config_are_defensive_copies(tmp_path: Path) -> None:
    source = _config(tmp_path)
    pipeline = IndexingPipeline.from_config(source, run_id="copy-test")
    source["package_contract"]["message"] = "mutated input"
    returned = pipeline.config
    returned["package_contract"]["message"] = "mutated output"

    assert pipeline.config["package_contract"]["message"] == "hello from lawa_rag"
    assert load_pipeline_config({"nested": {"one": 1}}, overrides={"nested": {"two": 2}}) == {
        "nested": {"one": 1, "two": 2}
    }


def test_plan_validate_run_pause_and_resume(tmp_path: Path) -> None:
    work_dir = tmp_path / "candidate"
    pipeline = IndexingPipeline.from_config(
        _config(tmp_path, stages=2),
        run_id="package-run",
        work_dir=work_dir,
    )

    assert pipeline.validate_sync().ok
    assert [stage.id for stage in pipeline.plan_sync()] == ["contract_1", "contract_2"]

    paused = pipeline.run_sync(stop_after_stage="contract_1")
    assert paused.paused
    assert paused.stages[0].succeeded
    assert paused.stages[1].status == "pending"
    assert len(paused.artifacts().records) == 1

    loaded = PipelineRun.load(work_dir)
    assert loaded.to_dict() == paused.to_dict()

    resumed_pipeline = IndexingPipeline.from_run(work_dir)
    completed = resumed_pipeline.run_sync(resume=True)
    assert completed.succeeded
    assert all(stage.succeeded for stage in completed.stages)
    assert len(completed.artifacts().records) == 2


def test_run_fails_closed_on_stage_validation_errors(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["package_contract"] = {}
    pipeline = IndexingPipeline.from_config(config, run_id="invalid")

    report = pipeline.validate_sync()
    assert not report.ok
    assert report.to_dict() == {
        "package_contract/package_contract_test": ["package_contract.message is required"]
    }
    with pytest.raises(PipelineValidationError, match="message is required"):
        pipeline.run_sync()
    assert not pipeline.work_dir.exists()


def test_active_release_loader_is_read_only_and_checks_identity(tmp_path: Path) -> None:
    work_dir = tmp_path / "runs" / "example" / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "release_id": "release-1",
                "config_name": "example",
                "status": "passed",
                "work_dir": str(work_dir),
            }
        ),
        encoding="utf-8",
    )
    pointer_path = tmp_path / "active_release.json"
    pointer_path.write_text(
        json.dumps(
            {
                "run_id": "run-1",
                "release_id": "release-1",
                "status": "passed",
                "active_release_manifest": str(manifest_path),
            }
        ),
        encoding="utf-8",
    )

    release = load_active_release(pointer_path)
    assert release.run_id == "run-1"
    assert release.release_id == "release-1"
    assert release.work_dir == work_dir.resolve()
    detached = release.manifest
    detached["run_id"] = "changed"
    assert release.run_id == "run-1"

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["run_id"] = "another-run"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="does not match"):
        load_active_release(pointer_path)
