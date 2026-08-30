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
from pipeline.stages.formatters.mbzuai_index_readiness_formatter import _coverage_gate


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


def test_coverage_gate_supports_route_exact_critical_pattern():
    pattern = r"/student-resources/?$"
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


def test_canonical_metadata_honors_robots_noindex_directives():
    from pipeline.core.mbzuai_indexing import canonicalize_page_metadata

    canonical = canonicalize_page_metadata(
        {
            "https://mbzuai.ac.ae/student-resources": {
                "url": "https://mbzuai.ac.ae/student-resources",
                "robots": "follow",
                "meta_tags": {"robots": ["index, follow", "NOINDEX, follow"]},
            }
        }
    )

    record = canonical["https://mbzuai.ac.ae/student-resources"]
    assert record["indexable"] is False
    assert record["index_exclusion_reason"] == "robots_noindex"
    assert record["robots_noindex"] is True
    assert "noindex" in record["robots_directives"]


def test_canonical_metadata_honors_top_level_x_robots_tag():
    from pipeline.core.mbzuai_indexing import canonicalize_page_metadata

    canonical = canonicalize_page_metadata(
        {
            "https://mbzuai.ac.ae/private-preview": {
                "url": "https://mbzuai.ac.ae/private-preview",
                "x-robots-tag": "noindex, nofollow",
            }
        }
    )

    record = canonical["https://mbzuai.ac.ae/private-preview"]
    assert record["indexable"] is False
    assert record["index_exclusion_reason"] == "robots_noindex"


def test_canonical_metadata_records_exact_host_noindex_authorization():
    from pipeline.core.mbzuai_indexing import canonicalize_page_metadata

    canonical = canonicalize_page_metadata(
        {
            "https://preprod.mbzuai.ac.ae/about-us": {
                "url": "https://preprod.mbzuai.ac.ae/about-us",
                "x-robots-tag": "noindex, nofollow",
                "indexable": False,
            },
            "https://other.mbzuai.ac.ae/private-preview": {
                "url": "https://other.mbzuai.ac.ae/private-preview",
                "x-robots-tag": "noindex, nofollow",
            },
        },
        authorized_noindex_hosts=["preprod.mbzuai.ac.ae"],
        noindex_override_reason="Explicit site-owner authorization for preproduction indexing.",
    )

    authorized = canonical["https://preprod.mbzuai.ac.ae/about-us"]
    assert authorized["indexable"] is True
    assert authorized["robots_noindex"] is True
    assert authorized["robots_noindex_authorized"] is True
    assert authorized["robots_noindex_overridden"] is True
    assert authorized["robots_noindex_authorization_reason"].startswith(
        "Explicit site-owner"
    )
    assert authorized["indexability_override_reason"].startswith("Explicit site-owner")

    unauthorized = canonical["https://other.mbzuai.ac.ae/private-preview"]
    assert unauthorized["indexable"] is False
    assert unauthorized["robots_noindex_authorized"] is False
    assert unauthorized["robots_noindex_overridden"] is False


def test_canonical_metadata_records_host_authorization_without_claiming_observation():
    from pipeline.core.mbzuai_indexing import canonicalize_page_metadata

    canonical = canonicalize_page_metadata(
        {
            "https://preprod.mbzuai.ac.ae/research": {
                "url": "https://preprod.mbzuai.ac.ae/research",
            }
        },
        authorized_noindex_hosts=["preprod.mbzuai.ac.ae"],
        noindex_override_reason="Explicit site-owner authorization for complete indexing.",
    )

    record = canonical["https://preprod.mbzuai.ac.ae/research"]
    assert record["indexable"] is True
    assert record["robots_noindex"] is False
    assert record["robots_noindex_authorized"] is True
    assert record["robots_noindex_overridden"] is False
    assert record["indexability_override_reason"] == ""
    assert record["robots_noindex_authorization_reason"].startswith(
        "Explicit site-owner"
    )


def test_coverage_gate_accepts_authorized_noindex_critical_route(tmp_path):
    markdown_path = tmp_path / "about-us.md"
    markdown_path.write_text(
        " ".join(["Official university history leadership and mission information."] * 30),
        encoding="utf-8",
    )
    gate = _coverage_gate(
        canonical_metadata={
            "https://preprod.mbzuai.ac.ae/about-us": {
                "indexable": True,
                "robots": "noindex, nofollow",
                "robots_noindex": True,
                "robots_noindex_overridden": True,
                "indexability_override_reason": "Explicit site-owner authorization.",
                "markdown_path": str(markdown_path),
            }
        },
        failure_manifest={
            "hard_failure_count": 0,
            "expected_site_inventory_count": 0,
            "inventory_coverage_ratio": None,
            "cohort_evidence_errors": [],
            "failed_urls": [],
        },
        formatter_config={"critical_url_patterns": [r"/about-us/?$"]},
    )

    assert gate["missing_critical_count"] == 0
    assert gate["unhealthy_critical_count"] == 0
    assert gate["ok"] is True


def test_coverage_gate_distinguishes_missing_from_unhealthy_critical_route(tmp_path):
    pattern = r"/student-resources/?$"
    failure_manifest = {
        "hard_failure_count": 0,
        "expected_site_inventory_count": 0,
        "inventory_coverage_ratio": None,
        "cohort_evidence_errors": [],
        "failed_urls": [],
    }

    missing = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources/campus-facilities": {
                "indexable": True,
            }
        },
        failure_manifest=failure_manifest,
        formatter_config={"critical_url_patterns": [pattern]},
    )
    unhealthy = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources": {
                "indexable": True,
                "markdown_path": str(tmp_path / "missing.md"),
            }
        },
        failure_manifest=failure_manifest,
        formatter_config={"critical_url_patterns": [pattern]},
    )

    assert missing["missing_critical_count"] == 1
    assert missing["unhealthy_critical_count"] == 0
    assert unhealthy["missing_critical_count"] == 0
    assert unhealthy["unhealthy_critical_count"] == 1
    evidence = unhealthy["unhealthy_critical_patterns"][0]["unhealthy_matches"][0]
    assert evidence["artifact_status"] == "missing"
    assert evidence["reasons"] == ["missing_markdown_artifact"]


def test_coverage_gate_rejects_noindex_critical_route_with_substantive_markdown(tmp_path):
    markdown_path = tmp_path / "student-resources.md"
    markdown_path.write_text(
        " ".join(["Official student resources and campus support information."] * 30),
        encoding="utf-8",
    )
    gate = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources": {
                "indexable": False,
                "index_exclusion_reason": "robots_noindex",
                "robots": "noindex, follow",
                "markdown_path": str(markdown_path),
            }
        },
        failure_manifest={
            "hard_failure_count": 0,
            "expected_site_inventory_count": 0,
            "inventory_coverage_ratio": None,
            "cohort_evidence_errors": [],
            "failed_urls": [],
        },
        formatter_config={"critical_url_patterns": [r"/student-resources/?$"]},
    )

    assert gate["missing_critical_count"] == 0
    assert gate["unhealthy_critical_count"] == 1
    evidence = gate["unhealthy_critical_patterns"][0]["unhealthy_matches"][0]
    assert evidence["reasons"] == ["robots_noindex"]
    assert gate["ok"] is False


def test_coverage_gate_rejects_failed_http_status_with_substantive_markdown(tmp_path):
    markdown_path = tmp_path / "contact.md"
    markdown_path.write_text(
        " ".join(["Official university contact and visitor information."] * 30),
        encoding="utf-8",
    )
    gate = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/about/contact": {
                "status_code": "404",
                "indexable": True,
                "markdown_path": str(markdown_path),
            }
        },
        failure_manifest={
            "hard_failure_count": 0,
            "expected_site_inventory_count": 0,
            "inventory_coverage_ratio": None,
            "cohort_evidence_errors": [],
            "failed_urls": [],
        },
        formatter_config={"critical_url_patterns": [r"/about/contact/?$"]},
    )

    evidence = gate["unhealthy_critical_patterns"][0]["unhealthy_matches"][0]
    assert evidence["status_code"] == 404
    assert evidence["reasons"] == ["http_status_404"]
    assert gate["ok"] is False


def test_coverage_gate_rejects_thin_navigation_heavy_markdown(tmp_path):
    markdown_path = tmp_path / "student-resources.md"
    links = " ".join(
        f"[Student service number {index}](https://mbzuai.ac.ae/service-{index})"
        for index in range(12)
    )
    markdown_path.write_text(
        f"# Student Resources\n\n{links}\n\nHome Student Resources cookies policy.",
        encoding="utf-8",
    )
    gate = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources": {
                "indexable": True,
                "markdown_path": str(markdown_path),
            }
        },
        failure_manifest={
            "hard_failure_count": 0,
            "expected_site_inventory_count": 0,
            "inventory_coverage_ratio": None,
            "cohort_evidence_errors": [],
            "failed_urls": [],
        },
        formatter_config={
            "critical_url_patterns": [r"/student-resources/?$"],
            "critical_url_min_markdown_words": 80,
            "critical_url_min_markdown_characters": 400,
            "critical_url_min_substantive_words": 40,
            "critical_url_navigation_min_links": 8,
            "critical_url_max_link_word_ratio": 0.55,
        },
    )

    evidence = gate["unhealthy_critical_patterns"][0]["unhealthy_matches"][0]
    assert evidence["reasons"] == ["thin_markdown", "navigation_heavy_markdown"]
    assert evidence["metrics"]["link_count"] == 12
    assert evidence["metrics"]["link_word_ratio"] > 0.55


def test_coverage_gate_accepts_substantive_indexable_critical_markdown(tmp_path):
    markdown_path = tmp_path / "student-resources.md"
    markdown_path.write_text(
        " ".join(["Official student resources and campus support information."] * 30),
        encoding="utf-8",
    )
    gate = _coverage_gate(
        canonical_metadata={
            "https://mbzuai.ac.ae/student-resources": {
                "indexable": True,
                "markdown_path": str(markdown_path),
            }
        },
        failure_manifest={
            "hard_failure_count": 0,
            "expected_site_inventory_count": 0,
            "inventory_coverage_ratio": None,
            "cohort_evidence_errors": [],
            "failed_urls": [],
        },
        formatter_config={"critical_url_patterns": [r"/student-resources/?$"]},
    )

    assert gate["missing_critical_count"] == 0
    assert gate["unhealthy_critical_count"] == 0
    assert gate["ok"] is True


def test_stage2_fails_closed_for_present_but_noindex_critical_route(tmp_path):
    markdown_path = tmp_path / "student-resources.md"
    markdown_path.write_text(
        " ".join(["Official student resources and campus support information."] * 30),
        encoding="utf-8",
    )
    page_metadata_file = tmp_path / "page_metadata.json"
    atomic_write_json(
        page_metadata_file,
        {
            "https://mbzuai.ac.ae/student-resources": {
                "url": "https://mbzuai.ac.ae/student-resources",
                "robots": "noindex, follow",
                "markdown_path": str(markdown_path),
            }
        },
    )
    ctx = StageContext(
        run_id="stage2-critical-health-fail-closed",
        project_name="mbzuai_main",
        config={
            "formatter": {
                "critical_url_patterns": [r"/student-resources/?$"],
                "fail_on_critical_coverage": True,
            }
        },
        work_dir=tmp_path,
        previous_outputs={"page_metadata_file": str(page_metadata_file)},
        stage_definition={"type": "formatter", "plugin": "mbzuai_index_readiness"},
        stage_id="prepare_mbzuai_index",
    )

    result = asyncio.run(readiness_module.MBZUAIIndexReadinessFormatter().execute(ctx))

    assert result.status is StageStatus.FAILED
    assert "missing_critical=0" in str(result.error_message)
    assert "unhealthy_critical=1" in str(result.error_message)
    gate = readiness_module.load_json_safe(
        tmp_path
        / "stage_outputs"
        / "prepare_mbzuai_index"
        / "index_coverage_gate.json"
    )
    assert gate["missing_critical_count"] == 0
    assert gate["unhealthy_critical_count"] == 1


@pytest.mark.parametrize("config_name", ["default", "mbzuai_production"])
def test_resolved_student_resources_hub_is_not_required_or_forced(config_name):
    config = load_config(config_name)
    patterns = config["formatter"]["critical_url_patterns"]
    student_resources_patterns = [
        str(pattern) for pattern in patterns if "student-resources" in str(pattern)
    ]
    priority_seed_urls = config["crawler"]["priority_seed_urls"]

    assert student_resources_patterns == []
    assert r"/student-resources/?$" not in (
        readiness_module.DEFAULT_CRITICAL_URL_PATTERNS
    )
    assert "https://mbzuai.ac.ae/student-resources" not in priority_seed_urls
    assert (
        "https://mbzuai.ac.ae/student-resources/campus-facilities"
        in priority_seed_urls
    )
    assert config["formatter"]["require_critical_url_markdown_evidence"] is True
    assert config["formatter"]["critical_url_min_markdown_words"] >= 40
    assert config["formatter"]["critical_url_min_substantive_words"] >= 20
    assert config["formatter"]["critical_url_max_link_word_ratio"] <= 0.60


def test_critical_markdown_threshold_config_is_validated_fail_closed():
    formatter = readiness_module.MBZUAIIndexReadinessFormatter()

    valid = asyncio.run(formatter.validate_config(load_config("mbzuai_production")))
    invalid = asyncio.run(
        formatter.validate_config(
            {
                "formatter": {
                    "critical_url_min_markdown_words": -1,
                    "critical_url_navigation_min_links": 0,
                    "critical_url_max_link_word_ratio": 1.1,
                }
            }
        )
    )

    assert valid == []
    assert "formatter.critical_url_min_markdown_words must be >= 0" in invalid
    assert "formatter.critical_url_navigation_min_links must be >= 1" in invalid
    assert (
        "formatter.critical_url_max_link_word_ratio must be between 0 and 1"
        in invalid
    )
