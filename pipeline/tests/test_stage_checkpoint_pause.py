import asyncio
from types import SimpleNamespace

import pytest


def _register_checkpoint_stages(call_counts):
    from pipeline.core.base import PipelineStage, StageResult
    from pipeline.core.registry import register_stage

    @register_stage
    class CheckpointStageOne(PipelineStage):
        name = "checkpoint_one"
        stage_type = "test_checkpoint_one"

        async def execute(self, ctx):
            call_counts["one"] += 1
            return StageResult.success(outputs={"one": "ready"})

    @register_stage
    class CheckpointStageTwo(PipelineStage):
        name = "checkpoint_two"
        stage_type = "test_checkpoint_two"

        async def execute(self, ctx):
            assert ctx.previous_outputs["one"] == "ready"
            call_counts["two"] += 1
            return StageResult.success(outputs={"two": "ready"})

    @register_stage
    class CheckpointStageThree(PipelineStage):
        name = "checkpoint_three"
        stage_type = "test_checkpoint_three"

        async def execute(self, ctx):
            assert ctx.previous_outputs["two"] == "ready"
            call_counts["three"] += 1
            return StageResult.success(outputs={"three": "ready"})


def _checkpoint_config(tmp_path):
    return {
        "project_name": "checkpoint-test",
        "work_dir": str(tmp_path / "runs"),
        "pipeline": {
            "production_profile": True,
            "audit_on_stage_complete": False,
            "audit_on_run_complete": True,
        },
        "stages": [
            {
                "id": "stage_one",
                "type": "test_checkpoint_one",
                "plugin": "checkpoint_one",
            },
            {
                "id": "stage_two",
                "type": "test_checkpoint_two",
                "plugin": "checkpoint_two",
            },
            {
                "id": "stage_three",
                "type": "test_checkpoint_three",
                "plugin": "checkpoint_three",
            },
        ],
    }


@pytest.fixture
def checkpoint_stages():
    from pipeline.core.registry import _REGISTRY

    call_counts = {"one": 0, "two": 0, "three": 0}
    _register_checkpoint_stages(call_counts)
    try:
        yield call_counts
    finally:
        _REGISTRY.pop("test_checkpoint_one", None)
        _REGISTRY.pop("test_checkpoint_two", None)
        _REGISTRY.pop("test_checkpoint_three", None)


def test_stop_after_stage_persists_pause_and_resumes_without_reruns(
    tmp_path,
    monkeypatch,
    checkpoint_stages,
):
    from pipeline.core.orchestrator import PipelineOrchestrator
    from pipeline.core.state import load_state

    audit_calls = []

    def record_audit(self, *, state, artifact_catalog, stage_id=None):
        audit_calls.append((state.status, stage_id))

    monkeypatch.setattr(PipelineOrchestrator, "_run_integrity_audit", record_audit)
    config = _checkpoint_config(tmp_path)
    work_dir = tmp_path / "checkpoint-run"

    first = asyncio.run(
        PipelineOrchestrator(config, work_dir=work_dir, run_id="checkpoint-run").run(
            stop_after_stage="stage_one"
        )
    )
    assert first.status == "paused"
    assert first.finished_at is None
    assert first.current_stage_index == 1
    assert [stage.status for stage in first.stages] == ["completed", "pending", "pending"]
    assert checkpoint_stages == {"one": 1, "two": 0, "three": 0}
    assert audit_calls == []

    persisted = load_state(work_dir)
    assert persisted is not None
    assert persisted.status == "paused"
    assert persisted.current_stage_index == 1

    second = asyncio.run(
        PipelineOrchestrator(config, work_dir=work_dir, run_id="checkpoint-run").run(
            resume=True,
            stop_after_stage="stage_two",
        )
    )
    assert second.status == "paused"
    assert second.current_stage_index == 2
    assert [stage.status for stage in second.stages] == ["completed", "completed", "pending"]
    assert checkpoint_stages == {"one": 1, "two": 1, "three": 0}
    assert audit_calls == []

    final = asyncio.run(
        PipelineOrchestrator(config, work_dir=work_dir, run_id="checkpoint-run").run(
            resume=True
        )
    )
    assert final.status == "completed"
    assert final.current_stage_index == 3
    assert checkpoint_stages == {"one": 1, "two": 1, "three": 1}
    assert audit_calls == [("completed", None)]


def test_stop_after_final_stage_defers_completion_and_full_run_audit(
    tmp_path,
    monkeypatch,
    checkpoint_stages,
):
    from pipeline.core.orchestrator import PipelineOrchestrator

    audit_calls = []

    def record_audit(self, *, state, artifact_catalog, stage_id=None):
        audit_calls.append((state.status, stage_id))

    monkeypatch.setattr(PipelineOrchestrator, "_run_integrity_audit", record_audit)
    config = _checkpoint_config(tmp_path)
    work_dir = tmp_path / "final-stage-pause"

    paused = asyncio.run(
        PipelineOrchestrator(config, work_dir=work_dir, run_id="final-stage-pause").run(
            stop_after_stage="stage_three"
        )
    )
    assert paused.status == "paused"
    assert paused.finished_at is None
    assert paused.current_stage_index == 3
    assert all(stage.is_terminal for stage in paused.stages)
    assert audit_calls == []

    completed = asyncio.run(
        PipelineOrchestrator(config, work_dir=work_dir, run_id="final-stage-pause").run(
            resume=True
        )
    )
    assert completed.status == "completed"
    assert checkpoint_stages == {"one": 1, "two": 1, "three": 1}
    assert audit_calls == [("completed", None)]


def test_stage_audit_still_runs_before_pause_but_full_run_audit_does_not(
    tmp_path,
    monkeypatch,
    checkpoint_stages,
):
    from pipeline.core.orchestrator import PipelineOrchestrator

    audit_calls = []

    def record_audit(self, *, state, artifact_catalog, stage_id=None):
        audit_calls.append((state.status, stage_id))

    monkeypatch.setattr(PipelineOrchestrator, "_run_integrity_audit", record_audit)
    config = _checkpoint_config(tmp_path)
    config["pipeline"]["audit_on_stage_complete"] = True

    paused = asyncio.run(
        PipelineOrchestrator(
            config,
            work_dir=tmp_path / "stage-audit-pause",
            run_id="stage-audit-pause",
        ).run(stop_after_stage="stage_one")
    )

    assert paused.status == "paused"
    assert audit_calls == [("running", "stage_one")]


def test_invalid_stop_selector_fails_before_work_directory_or_stage_execution(
    tmp_path,
    checkpoint_stages,
):
    from pipeline.core.orchestrator import PipelineOrchestrator

    config = _checkpoint_config(tmp_path)
    work_dir = tmp_path / "must-not-exist"
    orchestrator = PipelineOrchestrator(config, work_dir=work_dir, run_id="invalid-selector")

    with pytest.raises(ValueError, match="Unknown stage selector"):
        asyncio.run(orchestrator.run(stop_after_stage="not-a-stage"))

    assert not work_dir.exists()
    assert checkpoint_stages == {"one": 0, "two": 0, "three": 0}


def test_stop_boundary_before_restart_boundary_is_rejected_before_mutation(
    tmp_path,
    checkpoint_stages,
):
    from pipeline.core.orchestrator import PipelineOrchestrator

    config = _checkpoint_config(tmp_path)
    work_dir = tmp_path / "invalid-stage-range"
    orchestrator = PipelineOrchestrator(config, work_dir=work_dir, run_id="invalid-stage-range")

    with pytest.raises(ValueError, match="same stage as, or a stage after"):
        asyncio.run(
            orchestrator.run(
                resume=True,
                restart_from="stage_three",
                stop_after_stage="stage_two",
            )
        )

    assert not work_dir.exists()
    assert checkpoint_stages == {"one": 0, "two": 0, "three": 0}


def test_cli_returns_success_for_intentional_pause_and_forwards_selector(
    monkeypatch,
    capsys,
):
    from pipeline import cli

    captured = {}

    class FakeOrchestrator:
        def __init__(self, config, run_id=None):
            captured["run_id"] = run_id

        def resolve_stage_index(self, selector):
            captured["validated_selector"] = selector
            return 1

        async def run(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return SimpleNamespace(
                status="paused",
                run_id="cli-pause",
                current_stage_index=2,
                stages=[],
            )

    monkeypatch.setattr(cli, "load_config", lambda _name: {"project_name": "test"})
    monkeypatch.setattr(cli, "PipelineOrchestrator", FakeOrchestrator)
    args = SimpleNamespace(
        config="test",
        run_id="cli-pause",
        resume=True,
        restart_from_stage=None,
        stop_after_stage="stage_two",
        preflight=False,
        skip_preflight=False,
    )

    assert cli.cmd_run(args) == 0
    assert captured["validated_selector"] == "stage_two"
    assert captured["run_kwargs"] == {
        "resume": True,
        "restart_from": None,
        "stop_after_stage": "stage_two",
    }
    assert "intentionally paused" in capsys.readouterr().out


def test_cli_rejects_invalid_selector_before_required_preflight(monkeypatch, capsys):
    from pipeline import cli

    class FakeOrchestrator:
        def __init__(self, config, run_id=None):
            pass

        def resolve_stage_index(self, selector):
            raise ValueError(f"Unknown stage selector: {selector}")

        async def validate(self):
            raise AssertionError("preflight validation must not run")

    monkeypatch.setattr(
        cli,
        "load_config",
        lambda _name: {
            "project_name": "test",
            "pipeline": {"require_production_preflight": True},
        },
    )
    monkeypatch.setattr(cli, "PipelineOrchestrator", FakeOrchestrator)
    args = SimpleNamespace(
        config="test",
        run_id="invalid-cli-selector",
        resume=False,
        restart_from_stage=None,
        stop_after_stage="missing",
        preflight=False,
        skip_preflight=False,
    )

    assert cli.cmd_run(args) == 1
    assert "Invalid stage selector" in capsys.readouterr().out
