"""Recovery contracts for persisted package runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from lawa_rag import IndexingPipeline, PipelineRun
from pipeline.tests.lawa_rag_test_support import behavior_config


def test_paused_run_resumes_from_persisted_checkpoint_and_artifact_catalog(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "candidate"
    config = behavior_config(
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
    pipeline = IndexingPipeline.from_config(
        config,
        run_id="checkpoint-resume",
        work_dir=work_dir,
    )

    paused = pipeline.run_sync(stop_after_stage="first")
    assert paused.paused
    assert paused.stages[1].status == "pending"
    assert len(paused.artifacts().records) == 1

    persisted = PipelineRun.load(work_dir)
    assert persisted.to_dict() == paused.to_dict()
    resumed = IndexingPipeline.from_run(work_dir).run_sync(resume=True)

    assert resumed.succeeded
    assert resumed.stages[1].outputs["prior_stage_id"] == "first"
    assert len(resumed.artifacts().records) == 2


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
