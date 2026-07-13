from __future__ import annotations

import asyncio

import pytest

from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.config import load_config
from pipeline.core.io import atomic_write_json
from pipeline.core.knowledge_graph import validate_graph_bundle
from pipeline.core.run_audit import audit_run
from pipeline.core.state import PipelineState, StageState
from pipeline.stages.formatters import mbzuai_index_readiness_formatter as readiness_module
from pipeline.stages.formatters.mbzuai_index_readiness_formatter import (
    DEFAULT_CRITICAL_URL_PATTERNS,
    _coverage_gate,
)


def _valid_canonical_graph() -> dict:
    return {
        "schema_version": 2,
        "graph_type": "mbzuai_canonical_page_link_graph",
        "nodes": [
            {"id": "page:a", "node_type": "page", "url": "https://mbzuai.ac.ae/a"},
            {"id": "page:b", "node_type": "page", "url": "https://mbzuai.ac.ae/b"},
        ],
        "edges": [
            {
                "id": "edge:ab",
                "edge_type": "LINKS_TO",
                "source_id": "page:a",
                "target_id": "page:b",
                "properties": {"link_type": "internal"},
            }
        ],
        "stats": {
            "node_count": 2,
            "edge_count": 1,
            "link_type_counts": {"internal": 1},
        },
    }


def _invalid_canonical_graph() -> dict:
    graph = _valid_canonical_graph()
    graph["nodes"].append(
        {"id": "page:a", "node_type": "page", "url": "https://mbzuai.ac.ae/ar/a"}
    )
    graph["stats"] = {
        "node_count": 2,
        "edge_count": 2,
        "link_type_counts": {"internal": 2},
    }
    return graph


def test_graph_validator_accepts_exact_finalized_stats():
    assert validate_graph_bundle(_valid_canonical_graph(), require_stats=True) == []


def test_graph_validator_requires_link_type_counts_with_required_stats():
    graph = _valid_canonical_graph()
    graph["stats"].pop("link_type_counts")

    issues = validate_graph_bundle(graph, require_stats=True)

    assert {issue["code"] for issue in issues} == {
        "graph_stats_missing_link_type_counts"
    }


def test_graph_validator_rejects_duplicate_nodes_and_pre_dedup_stats():
    issues = validate_graph_bundle(_invalid_canonical_graph(), require_stats=True)
    codes = {issue["code"] for issue in issues}

    assert "duplicate_graph_node_id" in codes
    assert "graph_node_count_mismatch" in codes
    assert "graph_edge_count_mismatch" in codes
    assert "graph_link_type_counts_mismatch" in codes

    link_issue = next(
        issue for issue in issues if issue["code"] == "graph_link_type_counts_mismatch"
    )
    assert link_issue["expected"] == {"internal": 1}
    assert link_issue["actual"] == {"internal": 2}
    assert link_issue["expected_total"] == 1
    assert link_issue["actual_total"] == 2


def test_run_audit_validates_canonical_page_link_graph_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    graph_file = (
        run_dir
        / "stage_outputs"
        / "prepare_mbzuai_index"
        / "canonical_page_link_graph.json"
    )
    atomic_write_json(graph_file, _invalid_canonical_graph())

    graph_record = build_artifact_record(
        artifact_type="canonical_page_link_graph",
        role="page_link_graph",
        producer_stage="prepare_mbzuai_index",
        uri=graph_file.resolve().as_uri(),
        local_path=graph_file,
    )
    catalog = ArtifactCatalog(records=[graph_record])
    state = PipelineState(
        run_id="canonical-graph-audit",
        project_name="mbzuai_main",
        status="paused",
        current_stage_index=1,
        stages=[
            StageState(
                name="mbzuai_index_readiness",
                stage_type="formatter",
                stage_id="prepare_mbzuai_index",
                status="completed",
                outputs={"canonical_page_link_graph_file": str(graph_file)},
                artifact_ids=[graph_record.artifact_id],
            )
        ],
    )

    report = audit_run(run_dir, state=state, artifact_catalog=catalog)
    codes = {issue.code for issue in report.errors}

    assert report.ok is False
    assert "duplicate_graph_node_id" in codes
    assert "graph_node_count_mismatch" in codes
    assert "graph_edge_count_mismatch" in codes
    assert "graph_link_type_counts_mismatch" in codes


def test_stage2_fails_before_publishing_an_invalid_canonical_graph(
    tmp_path,
    monkeypatch,
):
    page_metadata_file = tmp_path / "page_metadata.json"
    atomic_write_json(
        page_metadata_file,
        {
            "https://mbzuai.ac.ae/a": {
                "url": "https://mbzuai.ac.ae/a",
                "title": "Page A",
                "status_code": 200,
            }
        },
    )
    monkeypatch.setattr(
        readiness_module,
        "canonicalize_link_graph",
        lambda *_args, **_kwargs: _invalid_canonical_graph(),
    )
    ctx = StageContext(
        run_id="stage2-fail-closed",
        project_name="mbzuai_main",
        config={"formatter": {}},
        work_dir=tmp_path,
        previous_outputs={"page_metadata_file": str(page_metadata_file)},
        stage_definition={"type": "formatter", "plugin": "mbzuai_index_readiness"},
        stage_id="prepare_mbzuai_index",
    )

    result = asyncio.run(readiness_module.MBZUAIIndexReadinessFormatter().execute(ctx))

    assert result.status is StageStatus.FAILED
    assert "Canonical page link graph validation failed" in str(result.error_message)
    assert "duplicate_graph_node_id=1" in str(result.error_message)
    assert not (tmp_path / "stage_outputs" / "prepare_mbzuai_index").exists()


def test_student_resources_critical_pattern_is_route_exact():
    pattern = DEFAULT_CRITICAL_URL_PATTERNS[-1]
    failure_manifest = {
        "hard_failure_count": 0,
        "expected_site_inventory_count": 0,
        "inventory_coverage_ratio": None,
        "cohort_evidence_errors": [],
        "failed_urls": [],
    }
    near_match = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources-archive": {"indexable": True}
        },
        failure_manifest=failure_manifest,
        formatter_config={"critical_url_patterns": [pattern]},
    )
    exact_match = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources/": {"indexable": True}
        },
        failure_manifest=failure_manifest,
        formatter_config={"critical_url_patterns": [pattern]},
    )

    assert pattern == r"/student-resources/?$"
    assert near_match["missing_critical_count"] == 1
    assert exact_match["missing_critical_count"] == 0


@pytest.mark.parametrize("config_name", ["default", "mbzuai_production"])
def test_resolved_student_resources_critical_pattern_is_route_exact(config_name):
    config = load_config(config_name)
    patterns = config["formatter"]["critical_url_patterns"]
    student_resources_patterns = [
        str(pattern) for pattern in patterns if "student-resources" in str(pattern)
    ]
    failure_manifest = {
        "hard_failure_count": 0,
        "expected_site_inventory_count": 0,
        "inventory_coverage_ratio": None,
        "cohort_evidence_errors": [],
        "failed_urls": [],
    }

    assert student_resources_patterns == [r"/student-resources/?$"]

    near_match = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources-archive": {"indexable": True}
        },
        failure_manifest=failure_manifest,
        formatter_config={"critical_url_patterns": student_resources_patterns},
    )
    assert near_match["missing_critical_count"] == 1
    assert [item["pattern"] for item in near_match["missing_critical_patterns"]] == [
        r"/student-resources/?$"
    ]

    for exact_url in (
        "https://mbzuai.ac.ae/student-resources",
        "https://mbzuai.ac.ae/student-resources/",
    ):
        exact_match = _coverage_gate(
            canonical_metadata={exact_url: {"indexable": True}},
            failure_manifest=failure_manifest,
            formatter_config={"critical_url_patterns": student_resources_patterns},
        )
        assert exact_match["missing_critical_count"] == 0
