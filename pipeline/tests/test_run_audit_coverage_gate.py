from __future__ import annotations

from pipeline.core.artifacts import ArtifactCatalog
from pipeline.core.io import atomic_write_json
from pipeline.core.run_audit import audit_run
from pipeline.core.state import PipelineState


def _completed_state() -> PipelineState:
    return PipelineState(
        run_id="coverage-run",
        project_name="mbzuai_main",
        status="completed",
        stages=[],
    )


def test_run_audit_rejects_failed_site_coverage_gate(tmp_path):
    gate_path = (
        tmp_path
        / "stage_outputs"
        / "prepare_mbzuai_index"
        / "index_coverage_gate.json"
    )
    atomic_write_json(
        gate_path,
        {
            "schema_version": 1,
            "ok": False,
            "missing_critical_count": 0,
            "hard_failure_count": 5,
            "expected_site_inventory_count": 3_800,
            "minimum_inventory_coverage_ratio": 0.75,
            "inventory_coverage_ratio": 0.7003,
            "inventory_gap": True,
        },
    )

    report = audit_run(
        tmp_path,
        state=_completed_state(),
        artifact_catalog=ArtifactCatalog(),
    )

    assert report.ok is False
    issue = next(issue for issue in report.errors if issue.code == "index_coverage_gate_failed")
    assert issue.stage_id == "prepare_mbzuai_index"
    assert issue.metadata["inventory_coverage_ratio"] == 0.7003
    assert issue.metadata["hard_failure_count"] == 5


def test_run_audit_accepts_passing_site_coverage_gate(tmp_path):
    gate_path = (
        tmp_path
        / "stage_outputs"
        / "prepare_mbzuai_index"
        / "index_coverage_gate.json"
    )
    atomic_write_json(
        gate_path,
        {
            "schema_version": 1,
            "ok": True,
            "missing_critical_count": 0,
            "hard_failure_count": 0,
            "minimum_inventory_coverage_ratio": 0.75,
            "inventory_coverage_ratio": 0.91,
            "inventory_gap": False,
        },
    )

    report = audit_run(
        tmp_path,
        state=_completed_state(),
        artifact_catalog=ArtifactCatalog(),
    )

    assert report.ok is True
    assert not any(issue.code == "index_coverage_gate_failed" for issue in report.errors)
