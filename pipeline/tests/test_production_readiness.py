"""
Production readiness tests for the modular pipeline.

These tests exercise real logic paths, boundary conditions, and edge cases.
They verify that the pipeline components handle bad input, real-world HTML,
and integration points correctly.
"""

import asyncio
import builtins
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


def run_async(coro):
    """Run an async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="pipeline_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_preflight_local_json_graph_does_not_require_neo4j(monkeypatch):
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    stage_plugins = {
        "crawl_web": "crawl4ai",
        "prepare_mbzuai_index": "mbzuai_index_readiness",
        "score_raw_content": "quality_scorer",
        "clean_html": "trafilatura",
        "convert_documents": "docling",
        "convert_html": "markitdown",
        "deduplicate_markdown": "dedup_filter",
        "chunk_content": "hybrid",
        "format_assertion_slices": "extraction_slices",
        "extract_assertions_openai": "openai_assertion_extract",
        "validate_assertions_openai": "openai_assertion_validate",
        "canonicalize_assertions": "assertion_canonicalize",
        "promote_assertions": "assertion_promote",
        "format_retrieval": "retrieval_bundle_v2",
        "format_graph": "knowledge_graph",
        "promote_graph": "semantic_graph_promote",
        "community_graph": "semantic_graph_community",
        "summarize_community_graph": "semantic_graph_summarize",
        "upload_retrieval": "gemini_pinecone",
    }
    config = {
        "pipeline": {
            "audit_on_stage_complete": True,
            "audit_on_run_complete": True,
            "fail_on_audit_error": True,
        },
        "stages": [{"id": stage_id, "plugin": plugin} for stage_id, plugin in stage_plugins.items()],
        "embedder": {
            "namespace_strategy": "release",
            "pinecone_index": "dense",
            "pinecone_sparse_index": "sparse",
            "namespace_chunks": "chunks",
            "namespace_parents": "parents",
            "namespace_media": "media",
            "namespace_facts": "facts",
            "namespace_assertions": "assertions",
            "verify_index_after_upload": True,
            "enable_sparse": True,
        },
        "graph": {
            "store_backend": "local_json",
            "extraction_fail_open_after_retries": False,
        },
    }

    report = assess_production_readiness(config, config_name="local-json", validation_errors={})

    assert report["ok"] is True
    names = {check["name"]: check for check in report["checks"]}
    assert names["graph_store"]["status"] == "ok"
    assert "neo4j_credentials" not in names


def test_preflight_rejects_release_template_without_identity(monkeypatch):
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    config = load_config("default")
    config["embedder"]["namespace_release_template"] = "{base}"

    report = assess_production_readiness(config, config_name="default", validation_errors={})

    check = next(item for item in report["checks"] if item["name"] == "pinecone_release_isolation")
    assert check["status"] == "error"
    assert "release_id" in check["details"]["error"]


def test_cli_runs_required_preflight_without_upload_graph(monkeypatch):
    from pipeline import cli

    called = {"preflight": 0, "run": 0}

    class FakeOrchestrator:
        def __init__(self, config, run_id=None):
            self.config = config

        async def validate(self):
            return {}

        async def run(self, **kwargs):
            called["run"] += 1
            raise AssertionError("blocked run must not execute")

    config = {
        "pipeline": {"require_production_preflight": True},
        "stages": [{"id": "upload_retrieval", "plugin": "gemini_pinecone"}],
    }
    monkeypatch.setattr(cli, "load_config", lambda name: config)
    monkeypatch.setattr(cli, "PipelineOrchestrator", FakeOrchestrator)

    def fake_preflight(*args, **kwargs):
        called["preflight"] += 1
        return {
            "ok": False,
            "error_count": 1,
            "warning_count": 0,
            "checks": [{"name": "contract", "status": "error", "message": "blocked"}],
        }

    monkeypatch.setattr(cli, "assess_production_readiness", fake_preflight)
    args = SimpleNamespace(
        config="default",
        run_id="candidate",
        resume=False,
        restart_from_stage=None,
        preflight=False,
        skip_preflight=False,
    )

    assert cli.cmd_run(args) == 1
    assert called == {"preflight": 1, "run": 0}

    args.skip_preflight = True
    assert cli.cmd_run(args) == 1
    assert called == {"preflight": 1, "run": 0}


def test_canonical_production_name_cannot_use_downgraded_effective_config(monkeypatch):
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    config = {
        "pipeline": {
            "audit_on_stage_complete": True,
            "audit_on_run_complete": True,
            "fail_on_audit_error": True,
        },
        "stages": [],
        "retrieval": {"retriever_backend": "adaptive_hybrid"},
    }

    report = assess_production_readiness(config, config_name="mbzuai_production")

    contract = next(check for check in report["checks"] if check["name"] == "canonical_production_contract")
    assert contract["status"] == "error"
    assert "pipeline.production_profile must be true" in contract["details"]["errors"]


def test_release_preflight_validates_pgvector_reader_without_writer(monkeypatch):
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv(
        "PGVECTOR_DSN",
        "postgresql://mbzuai_retriever:reader-password@pgvector.internal/vectors?sslmode=require",
    )
    monkeypatch.delenv("PGVECTOR_INGEST_DSN", raising=False)
    config = load_config("mbzuai_production")

    indexing_report = assess_production_readiness(
        config,
        config_name="mbzuai_production",
        validation_errors={},
    )
    release_report = assess_production_readiness(
        config,
        config_name="mbzuai_production",
        validation_errors={},
        purpose="release",
    )
    indexing_target = next(
        check for check in indexing_report["checks"] if check["name"] == "pgvector_target"
    )
    release_target = next(
        check for check in release_report["checks"] if check["name"] == "pgvector_target"
    )

    assert indexing_report["purpose"] == "indexing"
    assert indexing_target["status"] == "error"
    assert indexing_target["details"]["purpose"] == "write"
    assert release_report["purpose"] == "release"
    assert release_target["status"] == "ok"
    assert release_target["details"]["purpose"] == "read"


def test_canonical_production_rejects_downgraded_crawl_contract(monkeypatch):
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    config = load_config("mbzuai_production")
    config["crawler"].update(
        {
            "start_url": "https://untrusted.example",
            "allowed_domains": ["untrusted.example"],
            "respect_robots_txt": False,
            "minimum_sitemap_seed_count": 0,
            "sitemap_max_sources": 1000,
            "validate_source_html_mode": "on_quality_warning",
            "known_empty_sitemap_cohorts": [],
        }
    )
    config["formatter"].update(
        {
            "fail_on_inventory_gap": False,
            "minimum_inventory_coverage_ratio": 0.5,
            "critical_url_patterns": [".*"],
        }
    )

    report = assess_production_readiness(
        config,
        config_name="mbzuai_production",
        validation_errors={},
    )

    contract = next(
        check
        for check in report["checks"]
        if check["name"] == "canonical_production_contract"
    )
    assert contract["status"] == "error"
    errors = contract["details"]["errors"]
    assert "crawler.start_url must target https://mbzuai.ac.ae" in errors
    assert (
        "crawler.allowed_domains must contain only mbzuai.ac.ae and ifm.ai"
        in errors
    )
    assert "crawler.respect_robots_txt must be true" in errors
    assert "crawler.minimum_sitemap_seed_count must be >= 2100" in errors
    assert "crawler.sitemap_max_sources must be between 32 and 100" in errors
    assert "crawler.validate_source_html_mode must be always" in errors
    assert "crawler.known_empty_sitemap_cohorts must be configured" in errors
    assert "formatter.fail_on_inventory_gap must be true" in errors
    assert "formatter.minimum_inventory_coverage_ratio must be >= 0.90" in errors
    assert any(
        error.startswith(
            "formatter.critical_url_patterns is missing required exact MBZUAI routes:"
        )
        for error in errors
    )


def test_canonical_production_serves_the_attested_local_graph():
    from pipeline.core.config import load_config

    config = load_config("mbzuai_production")

    assert config["retrieval"]["routed_graph_required"] is True
    assert config["retrieval"]["graph_query_backend"] == "local"


def test_canonical_production_crawl_contract_is_satisfied(monkeypatch):
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    config = load_config("mbzuai_production")
    crawler = config["crawler"]
    buildit_seeds = {
        str(value).rstrip("/")
        for value in crawler["priority_seed_urls"]
        if str(value).startswith("https://buildit.mbzuai.ac.ae")
    }
    excluded_buildit_routes = {
        "https://buildit.mbzuai.ac.ae/about",
        "https://buildit.mbzuai.ac.ae/apply",
        "https://buildit.mbzuai.ac.ae/highlights",
        "https://buildit.mbzuai.ac.ae/benefits",
        "https://buildit.mbzuai.ac.ae/network",
        "https://buildit.mbzuai.ac.ae/faqs",
    }

    assert crawler["origin_inventory_revision"] == "2026-08-20-v3"
    assert buildit_seeds == {"https://buildit.mbzuai.ac.ae"}
    assert "buildit.mbzuai.ac.ae" not in crawler["link_discovery_hosts"]
    assert crawler["minimum_crawled_pages_by_host"]["buildit.mbzuai.ac.ae"] == 1
    assert excluded_buildit_routes.issubset(
        crawler["origin_inventory"]["intentional_content_exclusions"]
    )

    report = assess_production_readiness(
        config,
        config_name="mbzuai_production",
        validation_errors={},
    )

    contract = next(
        check
        for check in report["checks"]
        if check["name"] == "canonical_production_contract"
    )
    assert contract["status"] == "ok"


def test_canonical_production_rejects_intentionally_excluded_buildit_seed(monkeypatch):
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    config = load_config("mbzuai_production")
    config["crawler"]["priority_seed_urls"].append(
        "https://buildit.mbzuai.ac.ae/about/"
    )

    report = assess_production_readiness(
        config,
        config_name="mbzuai_production",
        validation_errors={},
    )

    contract = next(
        check
        for check in report["checks"]
        if check["name"] == "canonical_production_contract"
    )
    assert contract["status"] == "error"
    assert (
        "crawler.priority_seed_urls must not include intentional content exclusions: "
        "https://buildit.mbzuai.ac.ae/about"
        in contract["details"]["errors"]
    )


def test_canonical_production_rejects_downgraded_content_processing_contract(monkeypatch):
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    config = load_config("mbzuai_production")
    config["quality"].update(
        {
            "fail_on_empty_input": False,
            "minimum_retention_ratio": 0.1,
            "maximum_error_count": 2,
            "maximum_error_ratio": 0.1,
        }
    )
    config["cleaner"].update(
        {
            "min_content_length": 1,
            "min_content_words": 0,
            "fail_on_zero_output": False,
            "minimum_retention_ratio": 0.1,
            "minimum_host_retention_ratio": 0.1,
            "minimum_host_input_count": 50,
            "maximum_error_count": 2,
            "maximum_error_ratio": 0.1,
            "require_critical_url_survival": False,
        }
    )

    report = assess_production_readiness(
        config,
        config_name="mbzuai_production",
        validation_errors={},
    )

    contract = next(
        check
        for check in report["checks"]
        if check["name"] == "canonical_production_contract"
    )
    errors = contract["details"]["errors"]
    assert "quality.fail_on_empty_input must be true" in errors
    assert "quality.minimum_retention_ratio must be >= 0.70" in errors
    assert "quality.maximum_error_count must be 0" in errors
    assert "quality.maximum_error_ratio must be 0" in errors
    assert "cleaner.fail_on_zero_output must be true" in errors
    assert "cleaner.min_content_length must be >= 100" in errors
    assert "cleaner.min_content_words must be >= 5" in errors
    assert "cleaner.minimum_retention_ratio must be >= 0.75" in errors
    assert "cleaner.minimum_host_retention_ratio must be >= 0.50" in errors
    assert "cleaner.minimum_host_input_count must be <= 10" in errors
    assert "cleaner.maximum_error_count must be 0" in errors
    assert "cleaner.maximum_error_ratio must be 0" in errors
    assert "cleaner.require_critical_url_survival must be true" in errors


def test_preflight_neo4j_backend_requires_upload_stage_and_credentials(monkeypatch):
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    monkeypatch.delenv("NEO4J_URI", raising=False)
    monkeypatch.delenv("NEO4J_USERNAME", raising=False)
    monkeypatch.delenv("NEO4J_PASSWORD", raising=False)

    stage_plugins = {
        "crawl_web": "crawl4ai",
        "prepare_mbzuai_index": "mbzuai_index_readiness",
        "score_raw_content": "quality_scorer",
        "clean_html": "trafilatura",
        "convert_documents": "docling",
        "convert_html": "markitdown",
        "deduplicate_markdown": "dedup_filter",
        "chunk_content": "hybrid",
        "format_assertion_slices": "extraction_slices",
        "extract_assertions_openai": "openai_assertion_extract",
        "validate_assertions_openai": "openai_assertion_validate",
        "canonicalize_assertions": "assertion_canonicalize",
        "promote_assertions": "assertion_promote",
        "format_retrieval": "retrieval_bundle_v2",
        "format_graph": "knowledge_graph",
        "promote_graph": "semantic_graph_promote",
        "community_graph": "semantic_graph_community",
        "summarize_community_graph": "semantic_graph_summarize",
        "upload_retrieval": "gemini_pinecone",
    }
    config = {
        "pipeline": {
            "audit_on_stage_complete": True,
            "audit_on_run_complete": True,
            "fail_on_audit_error": True,
        },
        "stages": [{"id": stage_id, "plugin": plugin} for stage_id, plugin in stage_plugins.items()],
        "embedder": {
            "namespace_strategy": "release",
            "pinecone_index": "dense",
            "pinecone_sparse_index": "sparse",
            "namespace_chunks": "chunks",
            "namespace_parents": "parents",
            "namespace_media": "media",
            "namespace_facts": "facts",
            "namespace_assertions": "assertions",
            "verify_index_after_upload": True,
        },
        "graph": {
            "store_backend": "neo4j",
            "neo4j_verify_after_upload": True,
            "extraction_fail_open_after_retries": False,
        },
    }

    report = assess_production_readiness(config, config_name="neo4j", validation_errors={})
    messages = [check["message"] for check in report["checks"] if check["status"] == "error"]

    assert report["ok"] is False
    assert any("Production assertion-first stages are missing" in message for message in messages)
    assert any("Neo4j credentials are required" in message for message in messages)


def test_preflight_accepts_v5_semantic_graph_release_order(monkeypatch):
    from pipeline.core.preflight import assess_production_readiness

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone")
    stages = [
        ("crawl_web", "crawl4ai"),
        ("prepare_mbzuai_index", "mbzuai_index_readiness"),
        ("score_raw_content", "quality_scorer"),
        ("clean_html", "trafilatura"),
        ("convert_documents", "docling"),
        ("convert_html", "markitdown"),
        ("deduplicate_markdown", "dedup_filter"),
        ("chunk_content", "hybrid"),
        ("build_retrieval_bundle", "gemini_retrieval"),
        ("format_graph", "knowledge_graph"),
        ("extract_semantic_graph", "semantic_graph_extract"),
        ("canonicalize_semantic_graph", "semantic_graph_canonicalize"),
        ("promote_graph", "semantic_graph_promote"),
        ("community_graph", "semantic_graph_community"),
        ("summarize_community_graph", "semantic_graph_summarize"),
        ("finalize_retrieval_bundle", "gemini_retrieval"),
        ("upload_retrieval", "gemini_pinecone"),
    ]
    config = {
        "pipeline": {
            "audit_on_stage_complete": True,
            "audit_on_run_complete": True,
            "fail_on_audit_error": True,
        },
        "stages": [{"id": stage_id, "plugin": plugin} for stage_id, plugin in stages],
        "embedder": {
            "namespace_strategy": "release",
            "pinecone_index": "dense",
            "pinecone_sparse_index": "sparse",
            "namespace_chunks": "chunks",
            "namespace_parents": "parents",
            "namespace_media": "media",
            "namespace_facts": "facts",
            "namespace_evidence_spans": "evidence_spans",
            "namespace_summaries": "summaries",
            "namespace_assertions": "assertions",
            "namespace_entities": "entities",
            "namespace_communities": "communities",
            "verify_index_after_upload": True,
            "enable_sparse": True,
        },
        "graph": {
            "store_backend": "local_json",
            "extraction_fail_open_after_retries": False,
        },
    }

    report = assess_production_readiness(config, config_name="semantic-v5", validation_errors={})

    names = {check["name"]: check for check in report["checks"]}
    assert names["stage_order"]["status"] == "ok"
    assert "semantic-graph/v5" in names["stage_order"]["message"]
    assert names["pinecone_targets"]["status"] == "ok"
    assert not [check for check in report["checks"] if check["status"] == "error"]


def test_extractive_summarizer_runs_without_openai(monkeypatch, tmp_dir):
    from pipeline.core.base import StageContext
    from pipeline.stages.summarizers.openai_summarizer import OpenAISummarizer

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    md_dir = tmp_dir / "markdown"
    md_dir.mkdir()
    (md_dir / "admissions.md").write_text(
        "# Graduate Admissions\n\n"
        "MBZUAI offers MSc and PhD programs in artificial intelligence fields. "
        "Applicants must submit required documents before the published deadline. "
        "The university is located in Masdar City, Abu Dhabi, UAE. "
        "The 2026 application cycle includes online screening and admissions review.",
        encoding="utf-8",
    )

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "openai":
            raise AssertionError("extractive summarizer must not import openai")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    stage = OpenAISummarizer()
    config = {
        "summarizer": {
            "provider": "extractive",
            "concurrency": 1,
            "extractive_summary_max_chars": 800,
            "extractive_key_facts": 5,
        }
    }
    errors = run_async(stage.validate_config(config))
    assert errors == []

    ctx = StageContext(
        run_id="test-run",
        project_name="test",
        config=config,
        work_dir=tmp_dir,
        previous_outputs={"md_dir": str(md_dir)},
        stage_definition={"type": "summarizer", "plugin": "openai_summarizer"},
        stage_id="summarize_content",
    )

    result = run_async(stage.execute(ctx))

    assert result.status.value == "completed"
    assert result.metrics["processed"] == 1
    summary_files = list((tmp_dir / "stage_outputs" / "summarize_content" / "summaries").glob("*.summary.json"))
    assert len(summary_files) == 1
    payload = json.loads(summary_files[0].read_text(encoding="utf-8"))
    assert payload["summary_provider"] == "extractive"
    assert "MBZUAI offers MSc and PhD programs" in payload["detailed_summary"]


def test_gliner_stage_can_skip_without_loading_model(monkeypatch, tmp_dir):
    from pipeline.core.base import StageContext
    from pipeline.stages.formatters.gliner_extract_formatter import GLiNERExtractFormatter

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "gliner":
            raise AssertionError("disabled GLiNER stage must not import gliner")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    retrieval_bundle = tmp_dir / "retrieval_bundle.json"
    retrieval_bundle.write_text(
        json.dumps({"fact_records": [{"id": "fact-1", "text": "MBZUAI is in Abu Dhabi."}]}),
        encoding="utf-8",
    )
    ctx = StageContext(
        run_id="test-run",
        project_name="test",
        config={"graph": {"gliner_enabled": False}},
        work_dir=tmp_dir,
        previous_outputs={"retrieval_bundle_file": str(retrieval_bundle)},
        stage_definition={"type": "formatter", "plugin": "gliner_extract"},
        stage_id="gliner_extract_graph",
    )

    result = run_async(GLiNERExtractFormatter().execute(ctx))

    assert result.status.value == "completed"
    assert result.metrics["skipped"] is True
    output = tmp_dir / "stage_outputs" / "gliner_extract_graph" / "gliner_entities.json"
    assert json.loads(output.read_text(encoding="utf-8")) == {}


# ─────────────────────────────────────────────────────────────
# 1. IO Utilities — test real file operations and edge cases
# ─────────────────────────────────────────────────────────────

class TestIO:
    def test_atomic_write_json_roundtrip(self, tmp_dir):
        """Verify data survives write+read cycle, including unicode and nested structures."""
        from pipeline.core.io import atomic_write_json, load_json_safe
        data = {
            "unicode": "MBZUAI \u2014 \u0645\u0628\u0632\u0648\u0639\u064a",
            "nested": {"deep": {"list": [1, None, True, "string"]}},
            "empty": {},
        }
        path = tmp_dir / "test.json"
        atomic_write_json(path, data)
        loaded = load_json_safe(path)
        assert loaded == data

    def test_atomic_write_creates_parent_dirs(self, tmp_dir):
        """Atomic write should create intermediate directories."""
        from pipeline.core.io import atomic_write_json, load_json_safe
        path = tmp_dir / "a" / "b" / "c" / "test.json"
        atomic_write_json(path, [1, 2, 3])
        assert json.loads(path.read_text()) == [1, 2, 3]

    def test_load_json_safe_returns_default_on_missing_file(self, tmp_dir):
        from pipeline.core.io import load_json_safe
        assert load_json_safe(tmp_dir / "nope.json", default={"x": 1}) == {"x": 1}

    def test_load_json_safe_returns_default_on_corrupt_file(self, tmp_dir):
        from pipeline.core.io import load_json_safe
        path = tmp_dir / "bad.json"
        path.write_text("{broken json!!! [[[", encoding="utf-8")
        assert load_json_safe(path, default="fallback") == "fallback"

    def test_safe_filename_strips_dangerous_chars(self):
        from pipeline.core.io import safe_filename
        result = safe_filename('test<>:"/\\|?*file.txt')
        for c in '<>:"/\\|?*':
            assert c not in result

    def test_safe_filename_truncates_long_names(self):
        from pipeline.core.io import safe_filename
        result = safe_filename("a" * 500, max_length=100)
        assert len(result) <= 100


# ─────────────────────────────────────────────────────────────
# 2. Config — test inheritance, deep merge, and env overrides
# ─────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config_has_all_sections(self):
        from pipeline.core.config import load_config
        config = load_config("default")
        for section in ("crawler", "cleaner", "converter", "chunker", "quality", "summarizer", "embedder", "formatter", "graph", "retrieval"):
            assert section in config, f"Missing section: {section}"

    def test_inheritance_overrides_only_specified_keys(self):
        """mbzuai_main overrides start_url but inherits max_depth from default."""
        from pipeline.core.config import load_config
        config = load_config("mbzuai_main")
        assert config["crawler"]["start_url"] == "https://mbzuai.ac.ae"
        assert config["crawler"]["max_depth"] == 4  # inherited
        assert config["embedder"]["engine"] == "gemini"  # inherited
        assert config["embedder"]["model"] == "gemini-embedding-2"  # inherited
        assert config["embedder"]["output_dimensionality"] == 1536
        assert config["embedder"]["pinecone_summary_index"] == "mbzuai-summary-gemini-index-latest"
        assert config["embedder"]["pinecone_text_index"] == "mbzuai-text-gemini-index-latest"

    def test_deep_merge_preserves_base_keys(self):
        from pipeline.core.config import _deep_merge
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result["a"]["x"] == 1  # preserved from base
        assert result["a"]["y"] == 99  # overridden
        assert result["a"]["z"] == 100  # new key
        assert result["b"] == 3  # untouched

    def test_deep_merge_override_replaces_non_dict_with_dict(self):
        """When base has scalar and override has dict, override wins."""
        from pipeline.core.config import _deep_merge
        result = _deep_merge({"a": 1}, {"a": {"nested": True}})
        assert result["a"] == {"nested": True}

    def test_env_override_changes_nested_config(self):
        from pipeline.core.config import load_config
        with patch.dict(os.environ, {"PIPELINE_CRAWLER__MAX_PAGES": "42"}):
            config = load_config("default")
            assert config["crawler"]["max_pages"] == 42

    def test_deployment_control_environment_does_not_pollute_config(self):
        from pipeline.core.config import load_config

        with patch.dict(
            os.environ,
            {
                "PIPELINE_CONFIG": "mbzuai_production",
                "PIPELINE_PREFLIGHT": "true",
                "PIPELINE_RESUME": "false",
                "PIPELINE_RESTART_FROM_STAGE": "upload_retrieval",
            },
        ):
            config = load_config("default")

        assert "config" not in config
        assert "preflight" not in config
        assert "resume" not in config
        assert "restart_from_stage" not in config

    def test_env_override_type_parsing(self):
        from pipeline.core.config import _parse_env_value
        assert _parse_env_value("true") is True
        assert _parse_env_value("false") is False
        assert _parse_env_value("42") == 42
        assert _parse_env_value("3.14") == 3.14
        assert _parse_env_value("null") is None
        assert _parse_env_value("hello world") == "hello world"

    def test_circular_inheritance_raises(self, tmp_dir):
        from pipeline.core.config import load_config
        (tmp_dir / "a.yaml").write_text("_inherit: b.yaml\nname: a\n")
        (tmp_dir / "b.yaml").write_text("_inherit: a.yaml\nname: b\n")
        with pytest.raises(ValueError, match="[Cc]ircular"):
            load_config("a", search_dirs=[tmp_dir])

    def test_missing_config_raises_file_not_found(self):
        from pipeline.core.config import load_config
        with pytest.raises(FileNotFoundError):
            load_config("this_config_does_not_exist_xyz123")

    def test_repository_relative_config_path_is_accepted(self):
        from pipeline.core.config import load_config

        config = load_config("pipeline/configs/mbzuai_main_retrieval_bundle_refresh.yaml")
        assert config["project_name"] == "mbzuai_main"

    def test_canonical_production_config_keeps_required_capabilities(self):
        from pipeline.core.config import load_config
        from pipeline.stages.crawlers.crawl4ai_crawler import _url_matches_path_prefix

        config = load_config("mbzuai_production")
        plugins = [stage["plugin"] for stage in config["stages"]]
        assert config["pipeline"]["production_profile"] is True
        assert config["embedder"]["namespace_strategy"] == "release"
        excluded_prefixes = set(config["crawler"]["excluded_path_prefixes"])
        assert "/tag/" in excluded_prefixes
        assert "/ar/tag/" in excluded_prefixes
        assert "/moodle-service-interruption" in excluded_prefixes
        assert "/ar/moodle-service-interruption" in excluded_prefixes
        assert "/publication" not in excluded_prefixes
        assert _url_matches_path_prefix(
            "https://mbzuai.ac.ae/moodle-service-interruption",
            excluded_prefixes,
        )
        assert _url_matches_path_prefix(
            "https://mbzuai.ac.ae/ar/moodle-service-interruption/details",
            excluded_prefixes,
        )
        assert not _url_matches_path_prefix(
            "https://mbzuai.ac.ae/moodle",
            excluded_prefixes,
        )
        cohort_counts = {
            item["id"]: item["expected_member_count"]
            for item in config["crawler"]["known_empty_sitemap_cohorts"]
        }
        assert cohort_counts == {
            "legacy-publications-v1": 33,
            "empty-category-archives-v1": 55,
            "empty-news-hub-v1": 1,
            "empty-training-hubs-v1": 2,
            "empty-llm-vacancy-hub-v1": 1,
            "empty-course-type-archives-v1": 2,
            "empty-department-archives-v1": 26,
            "empty-job-category-archives-v1": 1,
            "empty-theme-archives-v1": 13,
            "empty-talk-category-archives-v1": 60,
            "empty-speaker-archives-v1": 263,
            "empty-podcast-category-archives-v1": 1,
            "empty-research-center-category-archives-v1": 1,
        }
        assert config["crawler"]["minimum_sitemap_seed_count"] == 2100
        assert config["crawler"]["cohort_probe_concurrency"] == 4
        assert config["crawler"]["cohort_probe_attempts"] == 3
        assert config["crawler"]["cohort_probe_backoff_sec"] == 2.0
        assert config["crawler"]["cohort_probe_min_interval_sec"] == 1.5
        assert config["formatter"]["expected_site_inventory_count"] == 2700
        assert config["retrieval"]["retriever_backend"] == "routed_hybrid"
        assert config["retrieval"]["query_planner_enabled"] is True
        assert config["retrieval"]["evidence_adjudicator_max_workers"] == 2
        assert (
            config["retrieval"]["evidence_adjudicator_provider_timeout_sec"]
            <= config["retrieval"]["evidence_adjudicator_timeout_sec"]
        )
        assert "openai_assertion_extract" in plugins
        assert "openai_assertion_validate" in plugins
        assert "assertion_promote" in plugins
        assert "retrieval_bundle_v2" in plugins

    def test_production_config_excludes_only_arabic_qirong_redirect_loop(self):
        from pipeline.core.config import load_config
        from pipeline.stages.crawlers.crawl4ai_crawler import _url_matches_path_prefix

        config = load_config("mbzuai_production")
        excluded_prefixes = set(config["crawler"]["excluded_path_prefixes"])

        assert "/ar/study/faculty/qirong-ho-ar" in excluded_prefixes
        assert "/ar/study/faculty/qirong-ho" in excluded_prefixes
        assert _url_matches_path_prefix(
            "https://mbzuai.ac.ae/ar/study/faculty/qirong-ho-ar",
            excluded_prefixes,
        )
        assert _url_matches_path_prefix(
            "https://mbzuai.ac.ae/ar/study/faculty/qirong-ho/",
            excluded_prefixes,
        )
        assert not _url_matches_path_prefix(
            "https://mbzuai.ac.ae/study/faculty/qirong-ho/",
            excluded_prefixes,
        )

    def test_default_config_passes_production_preflight_wiring(self):
        from pipeline.core.config import load_config
        from pipeline.core.preflight import assess_production_readiness

        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-openai",
                "GEMINI_API_KEY": "test-gemini",
                "PINECONE_API_KEY": "test-pinecone",
                "NEO4J_URI": "https://neo4j.example",
                "NEO4J_USERNAME": "neo4j",
                "NEO4J_PASSWORD": "secret",
            },
            clear=False,
        ):
            report = assess_production_readiness(load_config("default"), validation_errors={})

        assert report["ok"]
        assert report["error_count"] == 0
        assert any(check["name"] == "pinecone_verification" for check in report["checks"])


# ─────────────────────────────────────────────────────────────
# 3. Registry — verify real plugins are discovered
# ─────────────────────────────────────────────────────────────

class TestRegistry:
    def test_all_registered_plugins_available(self):
        """Verify every expected plugin is discovered and has correct stage_type."""
        from pipeline.core.registry import auto_discover, get_stage
        auto_discover()

        expected = {
            ("crawler", "crawl4ai"),
            ("cleaner", "bs4"),
            ("cleaner", "trafilatura"),
            ("converter", "markitdown"),
            ("converter", "pdf_converter"),
            ("converter", "docling"),
            ("converter", "crawl4ai_md"),
            ("chunker", "fixed_window"),
            ("chunker", "hierarchical"),
            ("chunker", "hybrid"),
            ("quality_gate", "quality_scorer"),
            ("quality_gate", "dedup_filter"),
            ("quality_gate", "language_detector"),
            ("summarizer", "openai_summarizer"),
            ("formatter", "pinecone_formatter"),
            ("formatter", "mbzuai_index_readiness"),
            ("formatter", "mbzuai_legacy_vectorstores"),
            ("formatter", "gemini_retrieval"),
            ("formatter", "knowledge_graph"),
            ("formatter", "semantic_graph_extract"),
            ("formatter", "semantic_graph_canonicalize"),
            ("formatter", "semantic_graph_promote"),
            ("embedder", "openai_embedder"),
            ("embedder", "mbzuai_legacy_pinecone"),
            ("embedder", "gemini_pinecone"),
            ("embedder", "neo4j_graph_store"),
        }
        for stage_type, name in expected:
            cls = get_stage(stage_type, name)
            assert cls.stage_type == stage_type
            assert cls.name == name
            assert cls.description, f"{name} has empty description"

    def test_get_unknown_type_raises(self):
        from pipeline.core.registry import get_stage
        with pytest.raises(KeyError, match="Unknown stage type"):
            get_stage("nonexistent", "foo")

    def test_get_unknown_plugin_raises(self):
        from pipeline.core.registry import auto_discover, get_stage
        auto_discover()
        with pytest.raises(KeyError, match="Unknown crawler plugin"):
            get_stage("crawler", "nonexistent_plugin")

    def test_list_stages_returns_metadata(self):
        from pipeline.core.registry import auto_discover, list_stages
        auto_discover()
        stages = list_stages()
        info = stages["converter"]["docling"]
        assert "class" in info
        assert "description" in info
        assert "Granite" in info["description"]


# ─────────────────────────────────────────────────────────────
# 4. State — test serialization roundtrips and corruption
# ─────────────────────────────────────────────────────────────

class TestState:
    def test_full_roundtrip(self, tmp_dir):
        from pipeline.core.state import PipelineState, StageState, save_state, load_state, now_iso
        state = PipelineState(
            run_id="rt1",
            project_name="test",
            status="running",
            started_at=now_iso(),
            stages=[
                StageState(name="s1", stage_type="crawler", status="completed",
                           outputs={"key": "val"}, metrics={"pages": 10}),
                StageState(name="s2", stage_type="cleaner", status="pending",
                           checkpoint={"resume_from": 42}),
            ],
            current_stage_index=1,
        )
        save_state(state, tmp_dir)
        loaded = load_state(tmp_dir)
        assert loaded.run_id == "rt1"
        assert loaded.stages[0].outputs == {"key": "val"}
        assert loaded.stages[0].metrics == {"pages": 10}
        assert loaded.stages[1].checkpoint == {"resume_from": 42}
        assert loaded.current_stage_index == 1

    def test_load_returns_none_for_missing(self, tmp_dir):
        from pipeline.core.state import load_state
        assert load_state(tmp_dir) is None

    def test_load_returns_none_for_corrupt(self, tmp_dir):
        from pipeline.core.state import load_state
        (tmp_dir / "pipeline_state.json").write_text("CORRUPT!!!")
        assert load_state(tmp_dir) is None

    def test_save_state_clears_finished_at_when_pipeline_is_running(self, tmp_dir):
        from pipeline.core.state import PipelineState, StageState, load_state, save_state

        state = PipelineState(
            run_id="rt2",
            project_name="test",
            status="running",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T01:00:00+00:00",
            stages=[
                StageState(
                    name="s1",
                    stage_type="converter",
                    status="running",
                    started_at="2026-01-01T00:10:00+00:00",
                    finished_at="2026-01-01T00:20:00+00:00",
                    error_message="old error",
                )
            ],
        )

        save_state(state, tmp_dir)
        loaded = load_state(tmp_dir)

        assert loaded.status == "running"
        assert loaded.finished_at is None
        assert loaded.stages[0].finished_at is None
        assert loaded.stages[0].error_message is None

    def test_load_state_demotes_completed_pipeline_if_any_stage_is_running(self, tmp_dir):
        from pipeline.core.state import load_state

        (tmp_dir / "pipeline_state.json").write_text(
            json.dumps(
                {
                    "run_id": "rt3",
                    "project_name": "test",
                    "status": "completed",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T01:00:00+00:00",
                    "current_stage_index": 0,
                    "stages": [
                        {
                            "name": "s1",
                            "stage_type": "converter",
                            "status": "running",
                            "started_at": "2026-01-01T00:10:00+00:00",
                            "finished_at": "2026-01-01T00:20:00+00:00",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = load_state(tmp_dir)
        assert loaded.status == "running"
        assert loaded.finished_at is None
        assert loaded.stages[0].finished_at is None

    def test_load_state_resets_downstream_terminal_stages_after_unfinished_stage(self, tmp_dir):
        from pipeline.core.state import load_state

        (tmp_dir / "pipeline_state.json").write_text(
            json.dumps(
                {
                    "run_id": "rt4",
                    "project_name": "test",
                    "status": "running",
                    "started_at": "2026-01-01T00:00:00+00:00",
                    "finished_at": "2026-01-01T01:00:00+00:00",
                    "current_stage_index": 2,
                    "stages": [
                        {
                            "name": "s1",
                            "stage_type": "converter",
                            "status": "running",
                            "started_at": "2026-01-01T00:10:00+00:00",
                        },
                        {
                            "name": "s2",
                            "stage_type": "chunker",
                            "status": "completed",
                            "finished_at": "2026-01-01T00:20:00+00:00",
                            "outputs": {"stale": True},
                            "metrics": {"count": 5},
                            "artifact_ids": ["dead"],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        loaded = load_state(tmp_dir)
        assert loaded.current_stage_index == 0
        assert loaded.stages[0].status == "running"
        assert loaded.stages[1].status == "pending"
        assert loaded.stages[1].outputs == {}
        assert loaded.stages[1].metrics == {}
        assert loaded.stages[1].artifact_ids == []


# ─────────────────────────────────────────────────────────────
# 5. Image Filtering — test real filtering logic with edge cases
# ─────────────────────────────────────────────────────────────

class TestImageFiltering:
    """Test _is_content_image against real-world edge cases."""

    def test_accepts_large_content_image(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("https://example.com/photo.jpg", "Campus", "800", "600") is True

    def test_rejects_tracking_pixel(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("https://example.com/pixel.gif", "", "1", "1") is False

    def test_rejects_small_icon_by_dimensions(self):
        """32x32 is below 50px threshold."""
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("https://example.com/unknown.png", "", "32", "32") is False

    def test_accepts_49x100_because_height_is_big_enough(self):
        """Both dimensions must be tiny to reject. w=49 h=100 should pass."""
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        result = _is_content_image("https://example.com/photo.jpg", "Test", "49", "100")
        assert result is True

    def test_rejects_favicon_ico(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("/favicon.ico", "", "", "") is False

    def test_rejects_data_uri(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("data:image/gif;base64,R0lGODlhAQABAIAAAA", "", "", "") is False

    def test_rejects_spacer_in_url(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("/images/spacer.png", "", "100", "100") is False

    def test_false_positive_iconic_in_url(self):
        """Tokenized matching should not reject 'iconic' as if it were 'icon'."""
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        result = _is_content_image("https://example.com/iconic-building.jpg", "Building", "800", "600")
        assert result is True

    def test_accepts_image_with_no_dimensions(self):
        """Unknown dimensions should not reject — we can't tell it's small."""
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("https://example.com/photo.jpg", "Photo", "", "") is True

    def test_rejects_svg_extension(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("https://example.com/diagram.svg", "Diagram", "800", "600") is False

    def test_handles_percentage_width_gracefully(self):
        """Width='100%' should not crash — parseInt fails, dimension check is skipped."""
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        # ValueError caught, falls through to accept
        result = _is_content_image("https://example.com/photo.jpg", "Photo", "100%", "auto")
        assert result is True

    def test_accepts_url_with_query_params(self):
        """Image URL with ?version=1 — splitext handles this via urlparse."""
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        # urlparse().path strips query, so splitext works correctly
        result = _is_content_image("https://example.com/photo.jpg?v=2&w=500", "Photo", "", "")
        assert result is True

    def test_rejects_non_image_extension(self):
        """A .pdf link should be rejected even if dimensions look fine."""
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image
        assert _is_content_image("https://example.com/report.pdf", "Report", "800", "600") is False


class TestImageExtraction:
    """Test _extract_page_images with real HTML structures."""

    @pytest.fixture
    def complex_html(self):
        return """
        <html><body>
            <img src="https://example.com/campus.jpg" alt="Campus" width="800" height="600">
            <img src="/icons/logo.svg" alt="Logo" width="32" height="32">
            <img src="https://example.com/pixel.gif" width="1" height="1">
            <img role="presentation" src="https://example.com/decorative.png" width="400" height="300">
            <img aria-hidden="true" src="https://example.com/hidden.jpg" width="400" height="300">
            <figure>
                <img src="/images/chart.png" alt="Research stats" width="600" height="400">
                <figcaption>AI Research Statistics 2025</figcaption>
            </figure>
            <img src="https://example.com/campus.jpg" alt="Duplicate" width="800" height="600">
        </body></html>
        """

    def test_extracts_content_images_only(self, complex_html):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        images = _extract_page_images(complex_html, "https://example.com/page")
        urls = [img["url"] for img in images]

        assert "https://example.com/campus.jpg" in urls
        assert "https://example.com/images/chart.png" in urls
        # NOT included:
        assert not any("logo.svg" in u for u in urls)
        assert not any("pixel.gif" in u for u in urls)
        assert not any("decorative.png" in u for u in urls)
        assert not any("hidden.jpg" in u for u in urls)

    def test_deduplicates_same_url(self, complex_html):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        images = _extract_page_images(complex_html, "https://example.com/page")
        urls = [img["url"] for img in images]
        assert urls.count("https://example.com/campus.jpg") == 1

    def test_extracts_figcaption_as_context(self, complex_html):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        images = _extract_page_images(complex_html, "https://example.com/page")
        chart = next(i for i in images if "chart" in i["url"])
        assert "AI Research Statistics" in chart["context"]

    def test_resolves_relative_urls(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        html = '<img src="/photos/building.jpg" alt="Building" width="500" height="300">'
        images = _extract_page_images(html, "https://mbzuai.ac.ae/about")
        assert images[0]["url"] == "https://mbzuai.ac.ae/photos/building.jpg"

    def test_returns_empty_for_no_images(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        html = "<html><body><p>No images</p></body></html>"
        assert _extract_page_images(html, "https://example.com") == []

    def test_returns_empty_for_empty_html(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        assert _extract_page_images("", "https://example.com") == []

    def test_protocol_relative_urls(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        html = '<img src="//cdn.example.com/photo.jpg" alt="Photo" width="500" height="300">'
        images = _extract_page_images(html, "https://example.com/page")
        assert images[0]["url"] == "https://cdn.example.com/photo.jpg"

    def test_extracts_lazy_loaded_images(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        html = '<img data-src="/media/campus.jpg" alt="Campus" width="600" height="400">'
        images = _extract_page_images(html, "https://example.com/about")
        assert images[0]["url"] == "https://example.com/media/campus.jpg"

    def test_prefers_largest_srcset_candidate(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images
        html = """
        <img
          src="/media/small.jpg"
          srcset="/media/small.jpg 320w, /media/medium.jpg 640w, /media/large.jpg 1280w"
          alt="Research"
          width="800"
          height="600"
        >
        """
        images = _extract_page_images(html, "https://example.com/page")
        assert images[0]["url"] == "https://example.com/media/large.jpg"

    def test_recovers_nested_absolute_url_from_malformed_media_src(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_images, _normalize_http_url

        malformed = (
            "https://test:8890https://sambzuaiwpprd01.blob.core.windows.net/"
            "mbzuaiwpprd01/2024/11/Radiology-healthcare-doctor-1.jpg"
        )
        recovered = (
            "https://sambzuaiwpprd01.blob.core.windows.net/"
            "mbzuaiwpprd01/2024/11/Radiology-healthcare-doctor-1.jpg"
        )
        assert _normalize_http_url(malformed) == recovered

        html = f'<img src="{malformed}" alt="Gallery Image" width="800" height="600">'
        images = _extract_page_images(html, "https://mbzuai.ac.ae/event/mbzuai-research-showcase")
        assert images[0]["url"] == recovered


class TestVideoExtraction:
    def test_extracts_direct_and_iframe_videos(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_videos

        html = """
        <html><body>
          <figure>
            <video src="/media/campus-tour.mp4" poster="/media/campus-tour.jpg" title="Campus Tour">
              <track kind="captions" src="/captions/campus-tour.vtt">
            </video>
            <figcaption>Guided tour of the MBZUAI campus.</figcaption>
          </figure>
          <iframe
            src="https://www.youtube.com/embed/abc123"
            title="MBZUAI overview video"
            allowfullscreen
          ></iframe>
        </body></html>
        """

        videos = _extract_page_videos(html, "https://mbzuai.ac.ae")
        urls = [video["url"] for video in videos]

        assert "https://mbzuai.ac.ae/media/campus-tour.mp4" in urls
        assert "https://www.youtube.com/embed/abc123" in urls

        direct = next(video for video in videos if video["url"].endswith("campus-tour.mp4"))
        assert direct["poster_url"] == "https://mbzuai.ac.ae/media/campus-tour.jpg"
        assert direct["caption"] == "Guided tour of the MBZUAI campus."
        assert direct["track_urls"] == ["https://mbzuai.ac.ae/captions/campus-tour.vtt"]

        iframe = next(video for video in videos if "youtube.com" in video["url"])
        assert iframe["provider"] == "youtube"
        assert iframe["embed_type"] == "iframe"

    def test_json_ld_enriches_video_metadata(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_videos

        html = """
        <html><body>
          <video src="/media/intro.mp4"></video>
          <script type="application/ld+json">
            {
              "@context": "https://schema.org",
              "@type": "VideoObject",
              "name": "MBZUAI introduction",
              "description": "Overview of MBZUAI programs and campus life.",
              "contentUrl": "https://example.com/media/intro.mp4",
              "thumbnailUrl": "https://example.com/media/intro.jpg",
              "transcript": "Welcome to MBZUAI."
            }
          </script>
        </body></html>
        """

        videos = _extract_page_videos(html, "https://example.com")
        assert len(videos) == 1
        assert videos[0]["title"] == "MBZUAI introduction"
        assert videos[0]["poster_url"] == "https://example.com/media/intro.jpg"
        assert videos[0]["transcript"] == "Welcome to MBZUAI."

    def test_extracts_background_video_from_data_attributes(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _extract_page_videos

        html = """
        <html><body>
          <section>
            <div
              class="w-background-video"
              data-poster-url="https://cdn.example.com/poster.jpg"
              data-video-urls="https://cdn.example.com/video.webm,https://cdn.example.com/video.mp4"
            ></div>
          </section>
        </body></html>
        """

        videos = _extract_page_videos(html, "https://example.com")
        assert len(videos) == 1
        assert videos[0]["url"] == "https://cdn.example.com/video.mp4"
        assert videos[0]["poster_url"] == "https://cdn.example.com/poster.jpg"
        assert videos[0]["embed_type"] == "background-video"

    def test_strips_webvtt_timing_lines(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _strip_webvtt

        transcript = _strip_webvtt(
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:02.000\nWelcome to MBZUAI.\n\n2\n00:00:02.100 --> 00:00:04.000\nCampus tour starts now."
        )
        assert transcript == "Welcome to MBZUAI. Campus tour starts now."


class TestCrawlerHelpers:
    def test_parse_sitemap_index(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _parse_sitemap_xml

        payload = b"""
        <sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
          <sitemap><loc>https://example.com/sitemap-pages.xml</loc></sitemap>
          <sitemap><loc>https://example.com/sitemap-docs.xml.gz</loc></sitemap>
        </sitemapindex>
        """

        child_sitemaps, page_urls = _parse_sitemap_xml(payload, source_url="https://example.com/sitemap.xml")
        assert child_sitemaps == [
            "https://example.com/sitemap-pages.xml",
            "https://example.com/sitemap-docs.xml.gz",
        ]
        assert page_urls == []

    def test_parse_gzipped_urlset(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _parse_sitemap_xml
        import gzip

        payload = gzip.compress(
            b"""
            <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
              <url><loc>https://example.com/a</loc></url>
              <url><loc>https://example.com/b</loc></url>
            </urlset>
            """
        )

        child_sitemaps, page_urls = _parse_sitemap_xml(
            payload,
            source_url="https://example.com/sitemap.xml.gz",
        )
        assert child_sitemaps == []
        assert page_urls == ["https://example.com/a", "https://example.com/b"]

    def test_parse_gzipped_sitemap_rejects_oversized_decoded_payload(self):
        import gzip

        from pipeline.stages.crawlers.crawl4ai_crawler import _parse_sitemap_xml

        payload = gzip.compress(
            b"<urlset>" + (b" " * 256) + b"</urlset>"
        )

        with pytest.raises(ValueError, match="decoded sitemap exceeds"):
            _parse_sitemap_xml(
                payload,
                source_url="https://example.com/sitemap.xml.gz",
                max_decoded_bytes=64,
            )

    def test_bounded_response_reader_rejects_stream_over_limit(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import (
            _read_bounded_response,
        )

        class FakeContent:
            async def iter_chunked(self, _size):
                yield b"12345"
                yield b"678901"

        response = SimpleNamespace(content=FakeContent())

        with pytest.raises(ValueError, match="response exceeds configured size limit"):
            run_async(_read_bounded_response(response, 10))

    def test_build_initial_crawl_state_seeds_unique_urls(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _build_initial_crawl_state

        state = _build_initial_crawl_state(
            "https://example.com",
            [
                "https://example.com/a",
                "https://example.com/a",
                "https://example.com/b",
            ],
            max_pages=3,
        )
        assert state["pending"] == [
            {"url": "https://example.com", "parent_url": None},
            {"url": "https://example.com/a", "parent_url": "https://example.com"},
            {"url": "https://example.com/b", "parent_url": "https://example.com"},
        ]
        assert state["depths"]["https://example.com"] == 0
        assert state["depths"]["https://example.com/a"] == 1

    def test_build_initial_crawl_state_respects_frontier_seed_limit(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _build_initial_crawl_state

        state = _build_initial_crawl_state(
            "https://example.com",
            [
                "https://example.com/a",
                "https://example.com/b",
                "https://example.com/c",
            ],
            max_pages=10,
            frontier_seed_limit=1,
        )

        assert state["pending"] == [
            {"url": "https://example.com", "parent_url": None},
            {"url": "https://example.com/a", "parent_url": "https://example.com"},
        ]
        assert set(state["depths"]) == {
            "https://example.com",
            "https://example.com/a",
        }

    def test_raw_source_candidate_urls_include_canonical_trailing_slash_variant(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _raw_source_candidate_urls

        assert _raw_source_candidate_urls("https://mbzuai.ac.ae/about/office-of-the-president") == [
            "https://mbzuai.ac.ae/about/office-of-the-president",
            "https://mbzuai.ac.ae/about/office-of-the-president/",
        ]
        assert _raw_source_candidate_urls("https://mbzuai.ac.ae/file.pdf") == [
            "https://mbzuai.ac.ae/file.pdf",
        ]

    def test_content_quality_403_skips_are_recoverable(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_recoverable_crawl_skip_reason

        assert _is_recoverable_crawl_skip_reason(
            "SKIPPED_HTTP_403:content_quality:blocked_or_error_page,url_token_mismatch,thin_html"
        )
        assert _is_recoverable_crawl_skip_reason(
            "SKIPPED_HTTP_403:Blocked by anti-bot protection while loading the page"
        )
        assert not _is_recoverable_crawl_skip_reason("SKIPPED_HTTP_403")
        assert not _is_recoverable_crawl_skip_reason("SKIPPED_HTTP_403:Forbidden")

    @pytest.mark.parametrize("status", [301, 302, 307, 308])
    def test_redirect_skips_are_recoverable(self, status):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_recoverable_crawl_skip_reason

        assert _is_recoverable_crawl_skip_reason(f"SKIPPED_HTTP_{status}")
        assert not _is_recoverable_crawl_skip_reason("SKIPPED_HTTP_404")
        assert not _is_recoverable_crawl_skip_reason(f"SKIPPED_HTTP_{status}0")

    def test_transient_network_navigation_errors_are_recoverable(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_recoverable_crawl_skip_reason

        assert _is_recoverable_crawl_skip_reason(
            "SKIPPED_ERROR:Unexpected error in _crawl_web: "
            "Page.goto: net::ERR_INTERNET_DISCONNECTED at https://mbzuai.ac.ae/news/example"
        )

    def test_browser_close_connection_error_is_benign_shutdown_error(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_benign_browser_close_error

        assert _is_benign_browser_close_error(
            Exception("Browser.close: Connection closed while reading from the driver")
        )
        assert not _is_benign_browser_close_error(Exception("Page.goto: Timeout 45000ms exceeded"))

    def test_path_prefix_filter_blocks_tag_archives(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import PathPrefixFilter

        flt = PathPrefixFilter(["/tag/", "/ar/tag/"])

        assert not flt.apply("https://mbzuai.ac.ae/tag/cohort6")
        assert not flt.apply("https://mbzuai.ac.ae/ar/tag/student-life")
        assert flt.apply("https://mbzuai.ac.ae/news/research-announcement")

    def test_requeue_recoverable_skipped_urls_honors_retry_cap(self, monkeypatch):
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        crawler = object.__new__(Crawl4AICrawler)
        crawler.crawl_state = {
            "pending": [],
            "visited": ["https://mbzuai.ac.ae/news/a"],
            "pages_crawled": 1,
            "depths": {},
        }
        crawler.url_mapping = {
            "https://mbzuai.ac.ae/news/a": "SKIPPED_ERROR:Page.goto: net::ERR_INTERNET_DISCONNECTED"
        }
        crawler.stats = {
            "pages_failed": 1,
            "skipped_urls": 1,
            "recoverable_skips_exhausted": 0,
            "excluded_frontier_urls": 0,
        }
        crawler.recoverable_skip_retries = {"https://mbzuai.ac.ae/news/a": 2}
        crawler.recoverable_skip_max_retries = 2
        crawler.excluded_path_prefixes = set()
        crawler.start_url = "https://mbzuai.ac.ae"
        crawler.allowed_domains = {"mbzuai.ac.ae"}
        crawler.excluded_subdomains = set()
        monkeypatch.setattr(
            crawler_module,
            "_host_resolves_to_private_or_reserved",
            lambda _host: False,
        )

        requeued = crawler._requeue_recoverable_skipped_urls()

        assert requeued == []
        assert crawler.url_mapping["https://mbzuai.ac.ae/news/a"].startswith("SKIPPED_ERROR")
        assert crawler.stats["recoverable_skips_exhausted"] == 1

        # Reconciliation runs before every bounded batch. Exhaustion metrics
        # must remain URL counts rather than growing once per later batch.
        assert crawler._requeue_recoverable_skipped_urls() == []
        assert crawler.stats["recoverable_skips_exhausted"] == 1

    def test_requeue_anti_bot_403_without_requeueing_generic_403(self, monkeypatch):
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        anti_bot_url = "https://mbzuai.ac.ae/news/anti-bot"
        pdf_url = "https://mbzuai.ac.ae/uploads/protected.pdf"
        crawler = object.__new__(Crawl4AICrawler)
        crawler.crawl_state = {
            "pending": [],
            "visited": [anti_bot_url, pdf_url],
            "pages_crawled": 0,
            "depths": {},
        }
        crawler.url_mapping = {
            anti_bot_url: "SKIPPED_HTTP_403:Blocked by anti-bot protection while loading the page",
            pdf_url: "SKIPPED_HTTP_403",
        }
        crawler.stats = {
            "pages_failed": 2,
            "skipped_urls": 2,
            "recoverable_skips_exhausted": 0,
            "excluded_frontier_urls": 0,
        }
        crawler.recoverable_skip_retries = {}
        crawler.recoverable_skip_exhausted_urls = set()
        crawler.recoverable_skip_max_retries = 2
        crawler.excluded_path_prefixes = set()
        crawler.start_url = "https://mbzuai.ac.ae"
        crawler.allowed_domains = {"mbzuai.ac.ae"}
        crawler.excluded_subdomains = set()
        monkeypatch.setattr(
            crawler_module,
            "_host_resolves_to_private_or_reserved",
            lambda _host: False,
        )

        requeued = crawler._requeue_recoverable_skipped_urls()

        assert requeued == [anti_bot_url]
        assert crawler.crawl_state["pending"] == [
            {"url": anti_bot_url, "parent_url": None}
        ]
        assert crawler.crawl_state["visited"] == [pdf_url]
        assert crawler.recoverable_skip_retries == {anti_bot_url: 1}
        assert anti_bot_url not in crawler.url_mapping
        assert crawler.url_mapping[pdf_url] == "SKIPPED_HTTP_403"
        assert crawler.stats["pages_failed"] == 1
        assert crawler.stats["skipped_urls"] == 1

    def test_seed_frontier_fetches_requeue_before_slicing_pending(self, monkeypatch):
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        retry_url = "https://mbzuai.ac.ae/news/retry"
        pending_urls = [
            "https://mbzuai.ac.ae/news/next-a",
            "https://mbzuai.ac.ae/news/next-b",
        ]
        fetched_batches = []

        class FakeCrawler:
            async def arun_many(self, *, urls, config):
                fetched_batches.append(list(urls))
                return [
                    SimpleNamespace(
                        url=url,
                        success=True,
                        status_code=200,
                        html="<html><body>ok</body></html>",
                    )
                    for url in urls
                ]

        crawler = object.__new__(Crawl4AICrawler)
        crawler.crawl_state = {
            "pending": [
                {"url": url, "parent_url": "https://mbzuai.ac.ae"}
                for url in pending_urls
            ],
            "visited": [retry_url],
            "pages_crawled": 0,
            "depths": {},
        }
        crawler.url_mapping = {retry_url: "SKIPPED_HTTP_301"}
        crawler.stats = {
            "pages_scraped": 0,
            "pages_failed": 1,
            "skipped_urls": 1,
            "recoverable_skips_exhausted": 0,
            "excluded_frontier_urls": 0,
            "sitemap_batches_completed": 0,
        }
        crawler.recoverable_skip_retries = {}
        crawler.recoverable_skip_exhausted_urls = set()
        crawler.recoverable_skip_max_retries = 2
        crawler.sitemap_crawl_batch_size = 2
        crawler.fetch_concurrency = 2
        crawler.max_pages = 3
        crawler.start_url = "https://mbzuai.ac.ae"
        crawler.allowed_domains = {"mbzuai.ac.ae"}
        crawler.excluded_subdomains = set()
        crawler.excluded_path_prefixes = set()
        crawler.allow_query_urls = False
        crawler.allowed_query_param_names = set()
        crawler.require_https = True
        monkeypatch.setattr(
            crawler_module,
            "_host_resolves_to_private_or_reserved",
            lambda _host: False,
        )

        run_config = SimpleNamespace(clone=lambda **_kwargs: SimpleNamespace())
        with patch.object(crawler, "_process_result", return_value=True), patch.object(
            crawler, "_flush_runtime_state"
        ):
            run_async(
                crawler._crawl_seed_frontier(
                    {"crawler": FakeCrawler()},
                    run_config,
                    SimpleNamespace(),
                )
            )

        assert fetched_batches == [[retry_url, pending_urls[0]], [pending_urls[1]]]
        assert crawler.recoverable_skip_retries == {retry_url: 1}
        assert retry_url not in crawler.url_mapping

    def test_seed_frontier_stops_after_redirect_retry_cap(self, monkeypatch):
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        retry_url = "https://mbzuai.ac.ae/news/always-redirects"
        fetched_batches = []

        class FakeCrawler:
            async def arun_many(self, *, urls, config):
                fetched_batches.append(list(urls))
                return [
                    SimpleNamespace(
                        url=url,
                        success=False,
                        status_code=301,
                        html="",
                        error_message="redirect response without a rendered page",
                    )
                    for url in urls
                ]

        crawler = object.__new__(Crawl4AICrawler)
        crawler.crawl_state = {
            "pending": [],
            "visited": [retry_url],
            "pages_crawled": 0,
            "depths": {},
        }
        crawler.url_mapping = {retry_url: "SKIPPED_HTTP_301"}
        crawler.stats = {
            "pages_scraped": 0,
            "pages_failed": 1,
            "skipped_urls": 1,
            "recoverable_skips_exhausted": 0,
            "excluded_frontier_urls": 0,
            "sitemap_batches_completed": 0,
        }
        crawler.recoverable_skip_retries = {}
        crawler.recoverable_skip_exhausted_urls = set()
        crawler.recoverable_skip_max_retries = 2
        crawler.sitemap_crawl_batch_size = 1
        crawler.fetch_concurrency = 1
        crawler.max_pages = 10
        crawler.start_url = "https://mbzuai.ac.ae"
        crawler.allowed_domains = {"mbzuai.ac.ae"}
        crawler.excluded_subdomains = set()
        crawler.excluded_path_prefixes = set()
        crawler.allow_query_urls = False
        crawler.allowed_query_param_names = set()
        crawler.require_https = True
        monkeypatch.setattr(
            crawler_module,
            "_host_resolves_to_private_or_reserved",
            lambda _host: False,
        )

        run_config = SimpleNamespace(clone=lambda **_kwargs: SimpleNamespace())
        with patch.object(crawler, "_process_result", return_value=False), patch.object(
            crawler, "_recover_url_with_http_retry", return_value=False
        ), patch.object(crawler, "_flush_runtime_state"):
            run_async(
                crawler._crawl_seed_frontier(
                    {"crawler": FakeCrawler()},
                    run_config,
                    SimpleNamespace(),
                )
            )

        assert fetched_batches == [[retry_url], [retry_url]]
        assert crawler.recoverable_skip_retries == {retry_url: 2}
        assert crawler.url_mapping[retry_url].startswith("SKIPPED_HTTP_301")
        assert crawler.stats["recoverable_skips_exhausted"] == 1

    def test_seed_frontier_reconciles_retry_at_exact_page_budget(self, monkeypatch):
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        retry_url = "https://mbzuai.ac.ae/news/retry-at-budget"
        fetched_batches = []

        class FakeCrawler:
            async def arun_many(self, *, urls, config):
                fetched_batches.append(list(urls))
                return [
                    SimpleNamespace(
                        url=url,
                        success=True,
                        status_code=200,
                        html="<html><body>recovered</body></html>",
                    )
                    for url in urls
                ]

        crawler = object.__new__(Crawl4AICrawler)
        crawler.crawl_state = {
            "pending": [],
            "visited": [retry_url],
            "pages_crawled": 1,
            "depths": {},
        }
        crawler.url_mapping = {retry_url: "SKIPPED_HTTP_301"}
        crawler.stats = {
            "pages_scraped": 0,
            "pages_failed": 1,
            "skipped_urls": 1,
            "recoverable_skips_exhausted": 0,
            "excluded_frontier_urls": 0,
            "sitemap_batches_completed": 0,
        }
        crawler.recoverable_skip_retries = {}
        crawler.recoverable_skip_exhausted_urls = set()
        crawler.recoverable_skip_max_retries = 2
        crawler.sitemap_crawl_batch_size = 1
        crawler.fetch_concurrency = 1
        crawler.max_pages = 1
        crawler.start_url = "https://mbzuai.ac.ae"
        crawler.allowed_domains = {"mbzuai.ac.ae"}
        crawler.excluded_subdomains = set()
        crawler.excluded_path_prefixes = set()
        crawler.allow_query_urls = False
        crawler.allowed_query_param_names = set()
        crawler.require_https = True
        monkeypatch.setattr(
            crawler_module,
            "_host_resolves_to_private_or_reserved",
            lambda _host: False,
        )

        run_config = SimpleNamespace(clone=lambda **_kwargs: SimpleNamespace())
        with patch.object(crawler, "_process_result", return_value=True), patch.object(
            crawler, "_flush_runtime_state"
        ):
            run_async(
                crawler._crawl_seed_frontier(
                    {"crawler": FakeCrawler()},
                    run_config,
                    SimpleNamespace(),
                )
            )

        assert fetched_batches == [[retry_url]]
        assert crawler.recoverable_skip_retries == {retry_url: 1}
        assert crawler.crawl_state["pages_crawled"] == 1
        assert crawler.crawl_state["pending"] == []

    def test_successful_http_retry_clears_existing_skip_counters_once(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        page_url = "https://mbzuai.ac.ae/news/raw-source-recovers"
        crawler = object.__new__(Crawl4AICrawler)
        crawler.url_mapping = {
            page_url: "SKIPPED_LOW_QUALITY:content_quality:blocked_or_error_page"
        }
        crawler.stats = {
            "pages_failed": 1,
            "skipped_urls": 1,
            "http_fallback_retries": 0,
            "http_fallback_pages": 0,
        }
        crawler.sitemap_failed_url_retry_attempts = 1
        crawler.sitemap_failed_retry_backoff = 0

        async def fetch_raw_source(_url):
            return "<html><body>recovered content</body></html>", 200

        async def process_result(result, *, mark_failure):
            assert mark_failure is True
            crawler.url_mapping[page_url] = "/tmp/recovered.html"
            return True

        with patch.object(
            crawler,
            "_fetch_raw_source_page",
            side_effect=fetch_raw_source,
        ), patch.object(crawler, "_process_result", side_effect=process_result):
            assert run_async(crawler._recover_url_with_http_retry(page_url)) is True
            assert run_async(crawler._recover_url_with_http_retry(page_url)) is True

        assert crawler.stats["pages_failed"] == 0
        assert crawler.stats["skipped_urls"] == 0

    def test_requeue_recoverable_skipped_urls_excludes_configured_path_prefix(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        crawler = object.__new__(Crawl4AICrawler)
        crawler.crawl_state = {
            "pending": [],
            "visited": ["https://mbzuai.ac.ae/tag/cohort6"],
            "pages_crawled": 1,
            "depths": {},
        }
        crawler.url_mapping = {
            "https://mbzuai.ac.ae/tag/cohort6": "SKIPPED_ERROR:Page.goto: net::ERR_INTERNET_DISCONNECTED"
        }
        crawler.stats = {
            "pages_failed": 1,
            "skipped_urls": 1,
            "recoverable_skips_exhausted": 0,
            "excluded_frontier_urls": 0,
        }
        crawler.recoverable_skip_retries = {}
        crawler.recoverable_skip_max_retries = 2
        crawler.excluded_path_prefixes = {"/tag"}

        requeued = crawler._requeue_recoverable_skipped_urls()

        assert requeued == []
        assert crawler.url_mapping["https://mbzuai.ac.ae/tag/cohort6"].startswith(
            "SKIPPED_EXCLUDED_FRONTIER"
        )
        assert crawler.stats["skipped_urls"] == 1
        assert crawler.stats["excluded_frontier_urls"] == 1

    def test_trim_crawl_state_to_budget_caps_pending_urls(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _trim_crawl_state_to_budget

        state = _trim_crawl_state_to_budget(
            {
                "visited": [
                    "https://example.com",
                    "https://example.com/a",
                    "https://example.com/a",
                ],
                "pending": [
                    {"url": "https://example.com/a", "parent_url": "https://example.com"},
                    {"url": "https://example.com/b", "parent_url": "https://example.com"},
                    {"url": "https://example.com/c", "parent_url": "https://example.com"},
                    {"url": "https://example.com/d", "parent_url": "https://example.com"},
                ],
                "depths": {
                    "https://example.com": 0,
                    "https://example.com/a": 1,
                    "https://example.com/b": 1,
                    "https://example.com/c": 1,
                    "https://example.com/d": 1,
                },
                "pages_crawled": 9,
            },
            max_pages=4,
        )

        assert state["visited"] == [
            "https://example.com",
            "https://example.com/a",
        ]
        assert state["pending"] == [
            {"url": "https://example.com/b", "parent_url": "https://example.com"},
            {"url": "https://example.com/c", "parent_url": "https://example.com"},
        ]
        assert state["pages_crawled"] == 2
        assert set(state["depths"]) == {
            "https://example.com",
            "https://example.com/a",
            "https://example.com/b",
            "https://example.com/c",
        }

    def test_selects_raw_source_when_rendered_capture_is_generic_mbzuai_shell(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import (
            _html_to_markdown,
            _select_preferred_page_capture,
        )

        page_url = "https://mbzuai.ac.ae/study/graduate-admission-process"
        rendered_html = """
        <html><head><title>MBZUAI - Mohamed bin Zayed University of Artificial Intelligence</title></head>
        <body><main><h1>Thoughtcurators for AI creators</h1><p>Explore our AI degrees.</p></main></body></html>
        """
        raw_source_html = """
        <html><head><title>Graduate admission process - MBZUAI</title></head>
        <body><main><h1>Graduate admission process</h1>
        <p>Applicants must submit transcripts, CV, recommendation letters, and a statement of purpose.</p>
        </main></body></html>
        """

        selected_html, selected_source, report = _select_preferred_page_capture(
            page_url,
            rendered_html,
            raw_source_html=raw_source_html,
            rendered_markdown="# Thoughtcurators for AI creators\n\nExplore our AI degrees.",
        )
        markdown = _html_to_markdown(selected_html, page_url)

        assert selected_source == "raw_source"
        assert report["selection_reason"] == "rendered_capture_unusable"
        assert "generic_site_shell" in report["rendered"]["reasons"]
        assert "Graduate admission process" in markdown
        assert "transcripts" in markdown
        assert "Thoughtcurators" not in markdown

    def test_selects_raw_source_and_links_when_rendered_nextjs_page_crashes(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import (
            Crawl4AICrawler,
            _extract_page_links,
            _select_preferred_page_capture,
        )

        page_url = "https://buildit.mbzuai.ac.ae/"
        rendered_html = """
        <html><body><h2>Application error: a client-side exception has occurred
        (see the browser console for more information).</h2></body></html>
        """
        raw_source_html = """
        <html><head><title>Build It</title></head><body>
        <main><h1>Build It Demo Days</h1>
        <a href="/about">About</a><a href="/benefits">Benefits</a>
        </main></body></html>
        """

        selected_html, selected_source, report = _select_preferred_page_capture(
            page_url,
            rendered_html,
            raw_source_html=raw_source_html,
            rendered_markdown="## Application error: a client-side exception has occurred",
        )
        links = _extract_page_links(
            SimpleNamespace(links={}),
            selected_html,
            page_url,
            allowed_domains={"mbzuai.ac.ae"},
            allowed_hosts={"buildit.mbzuai.ac.ae"},
        )

        assert selected_source == "raw_source"
        assert report["selection_reason"] == "rendered_capture_unusable"
        assert "blocked_or_error_page" in report["rendered"]["reasons"]
        assert [link["target_url"] for link in links] == [
            "https://buildit.mbzuai.ac.ae/about",
            "https://buildit.mbzuai.ac.ae/benefits",
        ]

        markdown, markdown_source, markdown_reason = Crawl4AICrawler()._extract_markdown(
            SimpleNamespace(markdown=None),
            html=selected_html,
            page_url=page_url,
            rendered_markdown="## Application error: a client-side exception has occurred",
            capture_source=selected_source,
        )

        assert "Application error" not in markdown
        assert "Build It Demo Days" in markdown
        assert markdown_source == "source_html"
        assert markdown_reason == ""

    def test_suppresses_rendered_error_markdown_when_raw_source_has_no_text(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler

        markdown, markdown_source, markdown_reason = Crawl4AICrawler()._extract_markdown(
            SimpleNamespace(markdown=None),
            html="<html><head><title>Build It</title></head><body></body></html>",
            page_url="https://buildit.mbzuai.ac.ae/apply",
            rendered_markdown="## Application error: a client-side exception has occurred",
            capture_source="raw_source",
        )

        assert markdown == ""
        assert markdown_source == ""
        assert markdown_reason == "empty_markdown"

    def test_markdown_quality_rejects_blocked_pages_before_indexing(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _markdown_quality_reason

        reason = _markdown_quality_reason(
            "# 403 Forbidden\n\nYou do not have permission to access this page.",
            "https://mbzuai.ac.ae/about/contact",
        )

        assert reason == "blocked_or_error_page"

    def test_markdown_quality_rejects_generic_title_only_redirect_shell(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _html_quality_report, _markdown_quality_reason

        page_url = (
            "https://mbzuai.ac.ae/news/"
            "h-h-sheikh-theyab-bin-zayed-al-nahyan-witnesses-mbzuai-inaugural-commencement"
        )
        html = """
        <html>
          <head><title>MBZUAI - Mohamed bin Zayed University of Artificial Intelligence</title></head>
          <body>MBZUAI - Mohamed bin Zayed University of Artificial Intelligence</body>
        </html>
        """

        report = _html_quality_report(html, page_url)

        assert report["usable"] is False
        assert "generic_mbzuai_title" in report["reasons"]
        assert _markdown_quality_reason(
            "MBZUAI - Mohamed bin Zayed University of Artificial Intelligence",
            page_url,
            html=html,
        ) == "generic_mbzuai_title"

    def test_html_markdown_adds_page_title_and_rejects_generic_shell_fallback(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _html_to_markdown, _markdown_quality_reason

        page_url = "https://mbzuai.ac.ae/news/working-together-to-serve-the-nation"
        html = """
        <html><head><title>Working together to serve the nation - MBZUAI</title></head>
        <body><main><p>MBZUAI hosted a delegation from Khalifa University to discuss collaboration
        and research opportunities for the UAE AI ecosystem.</p></main></body></html>
        """
        generic_markdown = (
            "# Thought curators for AI creators\n\n"
            "* [Commencement 2026](https://mbzuai.ac.ae/the-node/commencement-2026/)\n"
            "* [Research](https://research.mbzuai.ac.ae/)\n"
        )

        markdown = _html_to_markdown(html, page_url)

        assert markdown.startswith("# Working together to serve the nation")
        assert _markdown_quality_reason(markdown, page_url, html=html) == ""
        assert _markdown_quality_reason(generic_markdown, page_url, html=html) == "generic_site_shell"

    def test_markdown_quality_rejects_navigation_heavy_crawl_output(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _html_to_markdown, _markdown_quality_reason

        nav_links = "\n".join(
            f"{idx}. [Menu {idx}](https://mbzuai.ac.ae/ar/menu-{idx})"
            for idx in range(20)
        )
        markdown = nav_links + "\n\nريادة الأعمال في جامعة محمد بن زايد للذكاء الاصطناعي."
        reason = _markdown_quality_reason(markdown, "https://mbzuai.ac.ae/ar/innovate/entrepreneurship")

        html = """
        <html><body><nav><a>About</a><a>Study</a><a>Research</a></nav>
        <main><h1>ريادة الأعمال</h1><p>تدعم جامعة محمد بن زايد للذكاء الاصطناعي الشركات الناشئة وبرامج الابتكار.</p></main>
        </body></html>
        """

        assert reason == "navigation_heavy"
        regenerated = _html_to_markdown(html, "https://mbzuai.ac.ae/ar/innovate/entrepreneurship")
        assert "ريادة الأعمال" in regenerated
        assert "About" not in regenerated

    def test_markdown_quality_rejects_mega_menu_prefix_before_content(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _markdown_quality_reason

        markdown = """
        About
          1. [Leadership and Governance](https://mbzuai.ac.ae/about/leadership/)
          2. [Office of the President](https://mbzuai.ac.ae/about/office-of-the-president/)
          3. [Office of the Provost](https://mbzuai.ac.ae/office-of-the-provost/)

        Study
          1. [Graduate Admission Process](https://mbzuai.ac.ae/study/graduate-admission-process/)
          2. [Undergraduate Admission Process](https://mbzuai.ac.ae/study/ug-admission-process/)
          3. [Master's Programs](https://mbzuai.ac.ae/study/msc-programs/)

        Office of the President

        Professor Xing describes MBZUAI's research and education mission.
        """

        assert _markdown_quality_reason(
            markdown,
            "https://mbzuai.ac.ae/about/office-of-the-president",
        ) == "navigation_heavy"

    def test_html_to_markdown_fallback_removes_class_based_navigation(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _html_to_markdown, _markdown_quality_reason

        html = """
        <html><body>
          <div class="header-main"><a>About</a><a>Study</a><a>Research</a><a>Faculty Directory</a></div>
          <div class="container pt-24">
            <p class="blue-c">As a founding president, Professor Xing is keen to create a framework for AI research and education.</p>
            <p>There is a considerable need for more AI literate talent.</p>
          </div>
          <section class="chairs-message">
            <h2>Message from the President</h2>
            <p>Professor Eric Xing describes MBZUAI's mission and global research impact.</p>
          </section>
          <div class="footer-main"><a>Careers</a><a>Contact</a></div>
        </body></html>
        """

        markdown = _html_to_markdown(html, "https://mbzuai.ac.ae/about/office-of-the-president")

        assert "Professor Xing" in markdown
        assert "Professor Eric Xing" in markdown
        assert "Faculty Directory" not in markdown
        assert _markdown_quality_reason(markdown, "https://mbzuai.ac.ae/about/office-of-the-president", html=html) == ""

    def test_navigation_detector_does_not_reject_body_text_with_research_terms(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _markdown_quality_reason

        markdown = """
        # Office of the President

        As a founding president, Professor Xing is keen to create a framework that can facilitate research,
        education, and innovation to be better organized and more impactful.

        MBZUAI recruits exceptional faculty across a range of disciplines and develops AI literate talent.
        """

        assert _markdown_quality_reason(markdown, "https://mbzuai.ac.ae/about/office-of-the-president") == ""

    def test_markdown_quality_rejects_thin_faculty_cookie_boilerplate(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _markdown_quality_reason

        markdown = """
        # Yuxia Wang

        مهتم بالعمل مع أعضاء هيئة التدريس لديناقم بتعبئة النموذج أدناه وسنقوم بالرد عليك.

        We use cookies

        We use necessary and analytics cookies to operate our website, analyze traffic, and improve your experience.
        """

        assert _markdown_quality_reason(
            markdown,
            "https://mbzuai.ac.ae/ar/study/faculty/yuxia-wang-ar",
        ) == "thin_boilerplate"

    def test_markdown_quality_rejects_arabic_homepage_navigation_lead_in(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _markdown_quality_reason

        markdown = """
        الرئيسيه - MBZUAI
        MBZUAI
        نبذة عن الجامعة
        1. [أعضاء الهيئة التدريسية](https://mbzuai.ac.ae/ar/study/faculty-directory/)
        الدراسة
        1. [إجراءات القبول للبكالوريوس](https://mbzuai.ac.ae/ar/study/undergraduate-admission-process/)
        البحوث
        1. [قسم معالجة اللغات الطبيعية](https://mbzuai.ac.ae/ar/research-department/natural-language-processing-department/)
        الابتكار
        1. [علاقات الشراكة والتعاون](https://mbzuai.ac.ae/ar/innovate/partnership/)
        **اكتشف برامجنا الدراسية**
        """

        assert _markdown_quality_reason(markdown, "https://mbzuai.ac.ae/ar") == "navigation_heavy"


class TestCrawlerStage:
    def test_execute_writes_expected_outputs(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module

        class FakeAsyncCrawler:
            def __init__(self, config=None):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def arun(self, url, config):
                async def _gen():
                    yield SimpleNamespace(
                        url=url,
                        html="<html><body><h1>Test</h1></body></html>",
                        success=True,
                        status_code=200,
                        links={"internal": [], "external": []},
                        markdown=SimpleNamespace(
                            fit_markdown="# Test",
                            raw_markdown="# Test",
                        ),
                    )

                return _gen()

        monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeAsyncCrawler)

        ctx = StageContext(
            run_id="run_test",
            project_name="test",
            config={
                "crawler": {
                    "start_url": "https://example.com",
                    "max_pages": 1,
                    "max_depth": 0,
                    "fetch_concurrency": 1,
                    "download_concurrency": 1,
                    "timeout": 5,
                    "sitemap_enabled": False,
                    "extract_images": False,
                    "download_page_images": False,
                    "respect_robots_txt": False,
                    "validate_source_html": False,
                },
                "converter": {"content_filter_threshold": 0.48},
            },
            work_dir=tmp_dir,
        )

        result = run_async(crawler_module.Crawl4AICrawler().execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert Path(result.outputs["html_dir"]).exists()
        assert Path(result.outputs["md_dir"]).exists()

        mappings = json.loads((tmp_dir / "mappings.json").read_text())
        assert mappings["https://example.com"] == str(next((tmp_dir / "html").glob("*.html")))

        md_files = list((tmp_dir / "markdown").glob("*.md"))
        assert len(md_files) == 1
        assert md_files[0].read_text() == "# Test"

    def test_execute_replaces_generic_mbzuai_rendered_capture_with_valid_source_html(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import load_json_safe
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module

        page_url = "https://mbzuai.ac.ae/study/graduate-admission-process"
        rendered_html = """
        <html><head><title>MBZUAI - Mohamed bin Zayed University of Artificial Intelligence</title></head>
        <body><main><h1>Thoughtcurators for AI creators</h1><p>Explore our AI degrees.</p></main></body></html>
        """
        raw_source_html = """
        <html><head><title>Graduate admission process - MBZUAI</title></head>
        <body><main><h1>Graduate admission process</h1>
        <p>Applicants must submit official transcripts and other admission documents.</p>
        </main></body></html>
        """

        class FakeAsyncCrawler:
            def __init__(self, config=None):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def arun(self, url, config):
                return [
                    SimpleNamespace(
                        url=page_url,
                        html=rendered_html,
                        success=True,
                        status_code=200,
                        links={"internal": [], "external": []},
                        markdown=SimpleNamespace(
                            fit_markdown="# Thoughtcurators for AI creators\n\nExplore our AI degrees.",
                            raw_markdown="# Thoughtcurators for AI creators\n\nExplore our AI degrees.",
                        ),
                    )
                ]

        async def fake_fetch_raw_source_page(self, source_url):
            assert source_url == page_url
            return raw_source_html, 200

        monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeAsyncCrawler)
        monkeypatch.setattr(crawler_module.Crawl4AICrawler, "_fetch_raw_source_page", fake_fetch_raw_source_page)

        ctx = StageContext(
            run_id="run_test",
            project_name="test",
            config={
                "crawler": {
                    "start_url": page_url,
                    "max_pages": 1,
                    "max_depth": 0,
                    "fetch_concurrency": 1,
                    "download_concurrency": 1,
                    "timeout": 5,
                    "sitemap_enabled": False,
                    "extract_images": False,
                    "download_page_images": False,
                    "respect_robots_txt": False,
                    "validate_source_html": True,
                },
                "converter": {"content_filter_threshold": 0.48},
            },
            work_dir=tmp_dir,
        )

        result = run_async(crawler_module.Crawl4AICrawler().execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert result.metrics["source_html_replacements"] == 1
        html_file = next((tmp_dir / "html").glob("*.html"))
        markdown_file = next((tmp_dir / "markdown").glob("*.md"))
        metadata = load_json_safe(tmp_dir / "page_metadata.json")

        assert "Graduate admission process - MBZUAI" in html_file.read_text(encoding="utf-8")
        assert "official transcripts" in markdown_file.read_text(encoding="utf-8")
        assert "Thoughtcurators" not in markdown_file.read_text(encoding="utf-8")
        assert metadata[page_url]["capture_source"] == "raw_source"
        assert metadata[page_url]["markdown_source"] == "source_html"

    def test_execute_skips_blocked_rendered_capture_when_source_recovery_fails(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import load_json_safe
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module

        page_url = "https://mbzuai.ac.ae/about/contact"

        class FakeAsyncCrawler:
            def __init__(self, config=None):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def arun(self, url, config):
                return [
                    SimpleNamespace(
                        url=page_url,
                        html="<html><body><h1>403 Forbidden</h1><p>Access denied.</p></body></html>",
                        success=True,
                        status_code=200,
                        links={"internal": [], "external": []},
                        markdown=SimpleNamespace(
                            fit_markdown="# 403 Forbidden\n\nAccess denied.",
                            raw_markdown="# 403 Forbidden\n\nAccess denied.",
                        ),
                    )
                ]

        async def fake_fetch_raw_source_page(self, source_url):
            assert source_url == page_url
            return "", None

        monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeAsyncCrawler)
        monkeypatch.setattr(crawler_module.Crawl4AICrawler, "_fetch_raw_source_page", fake_fetch_raw_source_page)

        ctx = StageContext(
            run_id="run_test",
            project_name="test",
            config={
                "crawler": {
                    "start_url": page_url,
                    "max_pages": 1,
                    "max_depth": 0,
                    "fetch_concurrency": 1,
                    "download_concurrency": 1,
                    "timeout": 5,
                    "sitemap_enabled": False,
                    "extract_images": False,
                    "download_page_images": False,
                    "respect_robots_txt": False,
                    "validate_source_html": True,
                },
                "converter": {"content_filter_threshold": 0.48},
            },
            work_dir=tmp_dir,
        )

        result = run_async(crawler_module.Crawl4AICrawler().execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert result.metrics["pages_scraped"] == 0
        assert result.metrics["pages_failed"] == 1
        assert not list((tmp_dir / "html").glob("*.html"))
        mappings = load_json_safe(tmp_dir / "mappings.json")
        assert mappings[page_url].startswith("SKIPPED_LOW_QUALITY")

    def test_execute_seeds_initial_frontier_when_runtime_state_is_empty(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module

        class FakeAsyncCrawler:
            def __init__(self, config=None):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def arun(self, url, config):
                resume_state = getattr(config.deep_crawl_strategy, "_resume_state", {}) or {}
                assert resume_state.get("pending") == [
                    {"url": "https://example.com", "parent_url": None}
                ]
                return [
                    SimpleNamespace(
                        url=url,
                        html="<html><body><h1>Seeded</h1></body></html>",
                        success=True,
                        status_code=200,
                        links={"internal": [], "external": []},
                        markdown=SimpleNamespace(
                            fit_markdown="# Seeded",
                            raw_markdown="# Seeded",
                        ),
                    )
                ]

        monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeAsyncCrawler)

        ctx = StageContext(
            run_id="run_test",
            project_name="test",
            config={
                "crawler": {
                    "start_url": "https://example.com",
                    "max_pages": 1,
                    "max_depth": 0,
                    "fetch_concurrency": 1,
                    "download_concurrency": 1,
                    "timeout": 5,
                    "sitemap_enabled": False,
                    "extract_images": False,
                    "download_page_images": False,
                    "respect_robots_txt": False,
                    "stream_results": False,
                    "fail_on_empty_result": True,
                    "validate_source_html": False,
                },
                "converter": {"content_filter_threshold": 0.48},
            },
            work_dir=tmp_dir,
        )

        result = run_async(crawler_module.Crawl4AICrawler().execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert result.metrics["pages_scraped"] == 1

    def test_execute_crawls_large_sitemap_frontier_in_bounded_batches(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module

        class FakeAsyncCrawler:
            calls = []

            def __init__(self, config=None):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def arun(self, url, config):
                raise AssertionError("large sitemap frontier should use bounded arun_many batches")

            async def arun_many(self, urls, config):
                self.calls.append(list(urls))
                return [
                    SimpleNamespace(
                        url=url,
                        html=f"<html><head><title>{url}</title></head><body><h1>{url}</h1></body></html>",
                        success=True,
                        status_code=200,
                        links={"internal": [], "external": []},
                        markdown=SimpleNamespace(
                            fit_markdown=f"# {url}",
                            raw_markdown=f"# {url}",
                        ),
                    )
                    for url in urls
                ]

        async def fake_discover(self):
            return [f"https://example.com/page-{i}" for i in range(5)]

        monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeAsyncCrawler)
        monkeypatch.setattr(crawler_module.Crawl4AICrawler, "_discover_sitemap_urls", fake_discover)

        ctx = StageContext(
            run_id="run_test",
            project_name="test",
            config={
                "crawler": {
                    "start_url": "https://example.com",
                    "max_pages": 10,
                    "max_depth": 0,
                    "fetch_concurrency": 1,
                    "download_concurrency": 1,
                    "timeout": 5,
                    "sitemap_enabled": True,
                    "sitemap_batch_crawl": True,
                    "sitemap_crawl_batch_size": 2,
                    "sitemap_frontier_seed_limit": 10,
                    "extract_images": False,
                    "download_page_images": False,
                    "respect_robots_txt": False,
                    "fail_on_empty_result": True,
                    "validate_source_html": False,
                },
                "converter": {"content_filter_threshold": 0.48},
            },
            work_dir=tmp_dir,
        )

        result = run_async(crawler_module.Crawl4AICrawler().execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert result.metrics["pages_scraped"] == 6
        assert result.metrics["sitemap_batches_completed"] == 3
        assert [len(call) for call in FakeAsyncCrawler.calls] == [2, 2, 2]
        crawl_state = json.loads((tmp_dir / "crawl_state.json").read_text())
        assert crawl_state["pending"] == []
        assert crawl_state["pages_crawled"] == 6

    def test_execute_recovers_failed_sitemap_batch_url_with_http_retry(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module

        failed_url = "https://example.com/page-0"

        class FakeAsyncCrawler:
            def __init__(self, config=None):
                self.config = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

            async def arun(self, url, config):
                raise AssertionError("large sitemap frontier should use bounded arun_many batches")

            async def arun_many(self, urls, config):
                results = []
                for url in urls:
                    if url == failed_url:
                        results.append(
                            SimpleNamespace(
                                url=url,
                                html="",
                                success=False,
                                status_code=None,
                                links={"internal": [], "external": []},
                                markdown=None,
                                error_message="BrowserContext.add_init_script: Target page, context or browser has been closed",
                            )
                        )
                    else:
                        results.append(
                            SimpleNamespace(
                                url=url,
                                html=f"<html><body><h1>{url}</h1></body></html>",
                                success=True,
                                status_code=200,
                                links={"internal": [], "external": []},
                                markdown=SimpleNamespace(
                                    fit_markdown=f"# {url}",
                                    raw_markdown=f"# {url}",
                                ),
                            )
                        )
                return results

        async def fake_discover(self):
            return [f"https://example.com/page-{i}" for i in range(3)]

        fallback_calls = {"count": 0}

        async def fake_fetch_raw_source_page(self, page_url):
            if page_url == failed_url:
                fallback_calls["count"] += 1
                if fallback_calls["count"] == 1:
                    return "", None
                return "<html><body><h1>Recovered</h1></body></html>", 200
            return "", None

        monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeAsyncCrawler)
        monkeypatch.setattr(crawler_module.Crawl4AICrawler, "_discover_sitemap_urls", fake_discover)
        monkeypatch.setattr(crawler_module.Crawl4AICrawler, "_fetch_raw_source_page", fake_fetch_raw_source_page)

        ctx = StageContext(
            run_id="run_test",
            project_name="test",
            config={
                "crawler": {
                    "start_url": "https://example.com",
                    "max_pages": 10,
                    "max_depth": 0,
                    "fetch_concurrency": 1,
                    "download_concurrency": 1,
                    "timeout": 5,
                    "sitemap_enabled": True,
                    "sitemap_batch_crawl": True,
                    "sitemap_crawl_batch_size": 2,
                    "sitemap_failed_url_retry_attempts": 2,
                    "sitemap_frontier_seed_limit": 10,
                    "extract_images": False,
                    "download_page_images": False,
                    "respect_robots_txt": False,
                    "fail_on_empty_result": True,
                    "validate_source_html": False,
                },
                "converter": {"content_filter_threshold": 0.48},
            },
            work_dir=tmp_dir,
        )

        result = run_async(crawler_module.Crawl4AICrawler().execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert result.metrics["pages_scraped"] == 4
        assert result.metrics["pages_failed"] == 0
        assert result.metrics["skipped_urls"] == 0
        assert result.metrics["http_fallback_retries"] == 1
        mappings = json.loads((tmp_dir / "mappings.json").read_text())
        assert failed_url in mappings
        assert not str(mappings[failed_url]).startswith("SKIPPED")

    def test_skip_extension_filter_blocks_binary_and_media_assets(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import SkipExtensionFilter

        flt = SkipExtensionFilter()
        assert flt.apply("https://example.com/page")
        assert not flt.apply("https://example.com/photo.jpg")
        assert not flt.apply("https://example.com/video.mp4")
        assert not flt.apply("https://example.com/file.pdf")

    def test_skip_query_filter_blocks_query_urls_by_default(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import SkipQueryFilter

        flt = SkipQueryFilter()
        assert flt.apply("https://example.com/page")
        assert not flt.apply("https://example.com/news?tag=ai")

    def test_skip_query_filter_allows_whitelisted_query_params(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import SkipQueryFilter

        flt = SkipQueryFilter(allow_query_urls=True, allowed_query_param_names={"page"})
        assert flt.apply("https://example.com/archive?page=2")
        assert not flt.apply("https://example.com/archive?tag=ai")

    def test_document_payload_validator_rejects_html_disguised_as_pdf(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_valid_downloaded_document_payload

        assert _is_valid_downloaded_document_payload(
            b"%PDF-1.7\nfake",
            extension=".pdf",
            content_type="application/pdf",
        )
        assert not _is_valid_downloaded_document_payload(
            b"<html><body>blocked</body></html>",
            extension=".pdf",
            content_type="text/html",
        )

    def test_content_image_filter_rejects_tracking_and_extensionless_noise(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import _is_content_image

        assert not _is_content_image("https://mc.yandex.ru/watch/104010328", "", None, None)
        assert not _is_content_image("https://example.com/asset", "", None, None)
        assert _is_content_image("https://example.com/asset", "Campus hero", None, None)


# ─────────────────────────────────────────────────────────────
# 6. Quality Scorer — boundary conditions and pattern matching
# ─────────────────────────────────────────────────────────────

class TestQualityScorer:
    def test_validate_config_rejects_invalid_fail_closed_policy(self):
        from pipeline.stages.quality.quality_scorer import QualityScorer

        errors = run_async(
            QualityScorer().validate_config(
                {
                    "quality": {
                        "min_content_length": -1,
                        "minimum_retention_ratio": "bad",
                        "maximum_error_count": -1,
                        "fail_on_zero_output": "yes",
                    }
                }
            )
        )

        assert "quality.min_content_length must be a non-negative integer" in errors
        assert "quality.minimum_retention_ratio must be a number between 0 and 1" in errors
        assert "quality.maximum_error_count must be a non-negative integer" in errors
        assert "quality.fail_on_zero_output must be a boolean" in errors

    def test_boundary_exactly_at_min_length(self):
        """Text with exactly min_length chars should pass."""
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = "a" * 100
        assert _check_quality(text, 100, True) is None

    def test_boundary_one_below_min_length(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = "a" * 99
        assert _check_quality(text, 100, True) == "too_short"

    def test_whitespace_only_text_is_too_short(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        assert _check_quality("   \n\t\n   ", 1, True) == "too_short"

    def test_empty_string_is_too_short(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        assert _check_quality("", 0, True) is None  # 0 length should pass with min=0

    def test_detects_login_wall_case_insensitive(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = "PLEASE LOG IN to access this protected content."
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_403_forbidden(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = "403 Forbidden - You don't have permission."
        assert _check_quality(text, 10, True) == "login_wall"

    def test_public_instructions_can_say_please_log_in(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>Commencement information</h1>
        <p>{intro}</p><p>An account with the university gown supplier has been
        created on your behalf. Please log in and place your order, following
        the attached instructions.</p></main></body></html>""".format(
            intro="Public graduation guidance for students and families. " * 20
        )
        assert _check_quality(text, 100, True) is None

    def test_public_article_can_discuss_authentication_required(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>Authentication design research</h1>
        <p>{article}</p><p>The phrase authentication required is one response
        studied by the researchers.</p></main></body></html>""".format(
            article="Public research article content with evidence and analysis. " * 20
        )
        assert _check_quality(text, 100, True) is None

    @pytest.mark.parametrize(
        "wall_text",
        [
            "Please log in with your institutional credentials to continue.",
            "Please log in to your account.",
            "Login required to view this page.",
        ],
    )
    def test_detects_bounded_login_wall_responses(self, wall_text):
        from pipeline.stages.quality.quality_scorer import _check_quality

        assert _check_quality(wall_text, 10, True) == "login_wall"

    def test_detects_login_wall_heading_with_account_continuation(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = "<html><body><main><h1>Please log in to your account</h1></main></body></html>"
        assert _check_quality(text, 10, True) == "login_wall"

    def test_public_first_paragraph_login_instruction_is_not_a_wall(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main>
        <p>Please log in to access this protected resource.</p>
        <p>{content}</p></main></body></html>""".format(
            content="Public instructions for university users and visitors. " * 30
        )
        assert _check_quality(text, 100, True) is None

    def test_public_first_paragraph_authentication_article_is_not_a_wall(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main>
        <p>Authentication required: design patterns for public APIs.</p>
        <p>{content}</p></main></body></html>""".format(
            content="Public research analysis with examples and evidence. " * 30
        )
        assert _check_quality(text, 100, True) is None

    def test_public_parking_rule_is_not_an_access_wall(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>Campus parking</h1>
        <p>You are not authorized to park on campus without a permit.</p>
        <p>{content}</p></main></body></html>""".format(
            content="Public parking rules and visitor guidance. " * 20
        )
        assert _check_quality(text, 100, True) is None

    def test_detects_short_access_denied_page(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = "Access denied. Contact the administrator if you believe this is an error."
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_markdown_access_denied_heading(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        assert _check_quality("# Access denied", 10, True) == "login_wall"

    def test_detects_error_prefixed_access_denied_heading(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = "<html><body><h1>Error: Access denied</h1></body></html>"
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_status_prefixed_access_denied_heading(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        for heading in ("403 Access denied", "403 - Access denied", "HTTP 403: Access denied"):
            text = f"<html><body><h1>{heading}</h1></body></html>"
            assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_access_denied_alert_with_explanation(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><div role="alert">
        Access denied. Your request cannot be completed.
        </div></body></html>"""
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_access_denied_with_login_form_even_when_page_is_long(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body>
        <nav>{navigation}</nav>
        <main><h1>Account portal</h1>
        <form action="/login"><p>Unauthorized access</p>
        <input name="username"><input type="password"></form>
        </main></body></html>""".format(navigation="Navigation content. " * 200)
        assert _check_quality(text, 100, True) == "login_wall"

    def test_detects_access_message_adjacent_to_login_form(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><div class="auth-shell">
        <p>You are not authorized to view this resource.</p>
        <form action="/login"><input type="password"></form>
        </div></main></body></html>"""
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_long_access_denied_shell_without_login_form(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><nav>{navigation}</nav>
        <main><h1>Access denied</h1><p>Your request cannot be completed.</p></main>
        </body></html>""".format(navigation="Navigation content. " * 200)
        assert _check_quality(text, 100, True) == "login_wall"

    def test_detects_nested_access_denied_shell(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><div><div>
        Access denied. Contact your administrator.
        </div></div></main></body></html>"""
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_h3_access_denied_heading(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = "<html><body><main><h3>Access denied</h3></main></body></html>"
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_markdown_portal_heading_with_login_response(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = "# Account portal\n\nPlease sign in to continue."
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_primary_login_form_with_bare_heading(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><nav>{navigation}</nav><main>
        <h1>Sign in</h1><form action="/login"><input type="password"></form>
        </main></body></html>""".format(navigation="Navigation. " * 200)
        assert _check_quality(text, 100, True) == "login_wall"

    def test_detects_short_primary_login_form_without_marker(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>Member access</h1>
        <form><label>Username<input></label><label>Password
        <input type="password"></label><button>Continue</button></form>
        </main></body></html>"""
        assert _check_quality(text, 10, True) == "login_wall"

    @pytest.mark.parametrize("response", ["Access denied.", "Authentication required."])
    def test_detects_branded_short_access_shell(self, response):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = f"""<html><body><main><h1>University SSO</h1>
        <div><p>{response}</p></div></main></body></html>"""
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_nested_login_response_near_form(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>University SSO</h1><section>
        <div><p>Please sign in to continue.</p></div>
        <div><form action="/login"><input type="password"></form></div>
        </section></main></body></html>"""
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_waf_reference_response(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = "403 Forbidden. Reference ID 12345. Contact your administrator."
        assert _check_quality(text, 10, True) == "login_wall"

    def test_detects_branded_markdown_login_response(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = "# University SSO\n\nPlease sign in to continue."
        assert _check_quality(text, 10, True) == "login_wall"

    def test_global_navigation_login_widget_does_not_poison_public_page(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><nav><p>Please log in to your account.</p>
        <form action="/login"><input type="password"></form></nav>
        <main><h1>Public university information</h1><p>{content}</p></main>
        </body></html>""".format(content="Substantive public content. " * 50)
        assert _check_quality(text, 100, True) is None

    def test_custom_login_widget_does_not_poison_public_page(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><div class="account-widget">
        <h2>Sign in</h2><form action="/login"><input type="password"></form>
        </div><article><h1>Public information</h1><p>{content}</p></article>
        </main></body></html>""".format(content="Substantive public content. " * 50)
        assert _check_quality(text, 100, True) is None

    @pytest.mark.parametrize(
        "subheading",
        ["Sign in", "Access denied", "403 Forbidden", "404 Error"],
    )
    def test_long_public_subheading_is_not_a_wall_or_error(self, subheading):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>HTTP response handbook</h1>
        <p>{content}</p><h2>{subheading}</h2>
        <p>This section documents the response for developers.</p>
        </main></body></html>""".format(
            content="Substantive public documentation and examples. " * 50,
            subheading=subheading,
        )
        assert _check_quality(text, 100, True) is None

    def test_large_public_article_can_discuss_unauthorized_access(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>Responsible AI systems</h1>
        <p>Researchers study unauthorized model access and cases where access denied
        messages help protect sensitive systems.</p>
        <p>{article}</p></main></body></html>""".format(
            article="Public research article content with evidence and analysis. " * 80
        )
        assert _check_quality(text, 100, True) is None

    def test_script_access_markers_do_not_filter_public_html(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><head><script>
        const unauthorized = "access denied";
        </script></head><body><main><h1>Public page</h1>
        <p>{content}</p></main></body></html>""".format(
            content="Useful public university information. " * 20
        )
        assert _check_quality(text, 100, True) is None

    def test_short_public_security_article_is_not_a_login_wall(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>Securing AI services</h1>
        <p>Unauthorized requests and access denied responses are useful signals for
        defenders studying model security. This public article explains the research.</p>
        </main></body></html>"""
        assert _check_quality(text, 100, True) is None

    def test_public_article_title_starting_access_denied_is_not_a_login_wall(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main>
        <h1>Access Denied: A History of Authorization Systems</h1>
        <p>{article}</p></main></body></html>""".format(
            article="Public research article content with evidence and analysis. " * 20
        )
        assert _check_quality(text, 100, True) is None

    def test_hidden_access_marker_is_not_visible_wall_evidence(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><div style="display: none">Unauthorized</div>
        <main><h1>Public information</h1><p>{content}</p></main></body></html>""".format(
            content="Useful public university information. " * 10
        )
        assert _check_quality(text, 100, True) is None

    def test_article_cards_do_not_hide_substantive_body_content(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body>
        <article>Small card</article><article>Second card</article>
        <section><h1>Undergraduate research program</h1>
        <p>{content}</p></section>
        </body></html>""".format(content="Substantive program information. " * 20)

        assert _check_quality(text, 300, True) is None

    def test_content_bearing_page_header_survives_empty_spa_main(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><header class="page-hero">
        <nav>Information for Log in Apply About Research</nav>
        <section><h1>Dr. Example Researcher</h1>
        <p>{content}</p></section></header><main id="main-content"></main>
        <footer>University footer links</footer></body></html>""".format(
            content="Substantive public faculty biography and research information. " * 12
        )

        assert _check_quality(text, 100, True) is None

    def test_empty_spa_main_does_not_make_short_hero_substantive(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><header class="page-hero">
        <nav>Information for Log in Apply About Research</nav>
        <h1>Empty placeholder</h1></header><main id="main-content"></main>
        <footer>University footer links</footer></body></html>"""

        assert _check_quality(text, 100, True) == "too_short"

    def test_detects_error_page_500(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = "# 500 Internal Server Error\n\nPlease try again later."
        assert _check_quality(text, 10, True) == "error_page"

    def test_detects_plain_500_error_response(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        assert _check_quality("500 Internal Server Error occurred.", 10, True) == "error_page"

    @pytest.mark.parametrize(
        "error_html",
        [
            "<main><h1>Error 404</h1><p>The page does not exist.</p></main>",
            "<main><h1>Oops!</h1><p>Page not found.</p><p>The URL may be incorrect.</p></main>",
            "<main><h1>503 Service Unavailable</h1><p>Please try again later.</p></main>",
        ],
    )
    def test_detects_branded_structural_error_pages(self, error_html):
        from pipeline.stages.quality.quality_scorer import _check_quality

        assert _check_quality(error_html, 10, True) == "error_page"

    @pytest.mark.parametrize(
        "error_text",
        [
            "Page not found. The URL may be incorrect. Visit the homepage or contact support.",
            "500 Internal Server Error nginx",
            "# MBZUAI\n\n## Page not found\n\nThe URL may be incorrect.",
        ],
    )
    def test_detects_extended_plain_error_shells(self, error_text):
        from pipeline.stages.quality.quality_scorer import _check_quality

        assert _check_quality(error_text, 10, True) == "error_page"

    def test_error_pages_detected_even_when_login_detection_disabled(self):
        """Error patterns are always checked regardless of detect_login flag."""
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = "# Page not found\n\nThe URL may be incorrect."
        assert _check_quality(text, 10, False) == "error_page"

    def test_public_article_can_discuss_error_responses(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = """<html><body><main><h1>Reliable web services</h1>
        <p>An HTTP 404 error and service unavailable response have different causes.</p>
        <p>{content}</p></main></body></html>""".format(
            content="Public engineering article with analysis and examples. " * 30
        )
        assert _check_quality(text, 100, True) is None

    def test_plain_error_incident_sentence_is_not_an_error_page(self):
        from pipeline.stages.quality.quality_scorer import _check_quality

        text = (
            "A 500 Internal Server Error occurred during the incident. "
            "This engineering report explains the cause and remediation."
        )
        assert _check_quality(text, 50, True) is None

    def test_login_skipped_when_disabled(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = "Please sign in to continue. " * 5
        assert _check_quality(text, 10, False) is None

    def test_normal_content_passes(self):
        from pipeline.stages.quality.quality_scorer import _check_quality
        text = """MBZUAI is the world's first graduate-level, research-based AI university.
        Located in Abu Dhabi, it offers programs in machine learning, natural language processing,
        and computer vision. The university was established in 2019."""
        assert _check_quality(text, 100, True) is None

    def test_execute_filters_to_out_of_place_directory(self, tmp_dir):
        """Rejected inputs stay immutable while accepted content is materialized."""
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import load_json_safe
        from pipeline.stages.quality.quality_scorer import QualityScorer

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()

        # Good content — should survive
        (md_dir / "good.md").write_text("This is good content. " * 20)
        # Too short — should be deleted
        (md_dir / "short.md").write_text("tiny")
        # Login wall — should be deleted
        (md_dir / "login.md").write_text(
            "# Sign in\n\nPlease sign in to continue viewing this protected resource."
        )
        # 404 — should be deleted
        (md_dir / "error.md").write_text(
            "# Page not found\n\nThis page does not exist. Please check the URL."
        )

        ctx = StageContext(
            run_id="test", project_name="test",
            config={
                "quality": {
                    "min_content_length": 50,
                    "detect_login_walls": True,
                    "minimum_retention_ratio": 0.0,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
            stage_id="score_raw_content",
        )

        scorer = QualityScorer()
        result = run_async(scorer.execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert result.outputs["passed_count"] == 1
        assert result.outputs["filtered_count"] == 3
        accepted_dir = Path(result.outputs["md_dir"])
        assert (accepted_dir / "good.md").exists()
        assert not (accepted_dir / "short.md").exists()
        assert not (accepted_dir / "login.md").exists()
        assert not (accepted_dir / "error.md").exists()
        assert (md_dir / "good.md").exists()
        assert (md_dir / "short.md").exists()
        assert (md_dir / "login.md").exists()
        assert (md_dir / "error.md").exists()
        manifest = load_json_safe(result.outputs["quality_manifest_file"])
        assert manifest["gate"]["ok"] is True
        assert manifest["gate"]["reason_counts"] == {
            "accepted_quality": 1,
            "error_page": 1,
            "login_wall": 1,
            "too_short": 1,
        }

    def test_execute_fails_closed_on_empty_input(self, tmp_dir):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import load_json_safe
        from pipeline.stages.quality.quality_scorer import QualityScorer

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"quality": {"min_content_length": 50}},
            work_dir=tmp_dir,
            previous_outputs={"html_dir": str(html_dir)},
            stage_id="score_raw_content",
        )

        result = run_async(QualityScorer().execute(ctx))

        assert result.status == StageStatus.FAILED
        manifest = load_json_safe(result.outputs["quality_manifest_file"])
        assert manifest["gate"]["ok"] is False
        assert {item["code"] for item in manifest["gate"]["failures"]} == {
            "empty_input"
        }

    def test_execute_on_raw_html_projects_mapping_and_media_without_mutating_sources(self, tmp_dir):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.quality.quality_scorer import QualityScorer

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()

        good_html = html_dir / "good.html"
        good_html.write_text("<html><body><main><h1>Good</h1><p>" + ("Useful content " * 20) + "</p></main></body></html>")
        bad_html = html_dir / "login.html"
        bad_html.write_text("<html><body><h1>Login</h1><p>Please sign in to continue.</p></body></html>")

        good_md = md_dir / "good.md"
        good_md.write_text("Good content " * 20, encoding="utf-8")
        bad_md = md_dir / "login.md"
        bad_md.write_text("# Login\n\nLoading", encoding="utf-8")

        mapping_file = tmp_dir / "mappings.json"
        md_mapping_file = tmp_dir / "url_to_md_mapping.json"
        page_media_file = tmp_dir / "page_media.json"
        page_images_file = tmp_dir / "page_images.json"
        page_videos_file = tmp_dir / "page_videos.json"

        atomic_write_json(
            mapping_file,
            {
                "https://example.com/good": str(good_html),
                "https://example.com/login": str(bad_html),
                "https://example.com/verified-empty": "SKIPPED_VERIFIED_EMPTY_COHORT:test-policy",
                "https://example.com/not-found": "SKIPPED_HTTP_404",
            },
        )
        atomic_write_json(
            md_mapping_file,
            {
                "https://example.com/good": str(good_md),
                "https://example.com/login": str(bad_md),
            },
        )
        atomic_write_json(
            page_media_file,
            {
                "https://example.com/good": [{"type": "image", "url": "https://example.com/good.jpg"}],
                "https://example.com/login": [{"type": "image", "url": "https://example.com/login.jpg"}],
            },
        )
        atomic_write_json(
            page_images_file,
            {
                "https://example.com/good": [{"url": "https://example.com/good.jpg"}],
                "https://example.com/login": [{"url": "https://example.com/login.jpg"}],
            },
        )
        atomic_write_json(
            page_videos_file,
            {
                "https://example.com/good": [{"url": "https://example.com/good.mp4"}],
                "https://example.com/login": [{"url": "https://example.com/login.mp4"}],
            },
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={
                "quality": {
                    "min_content_length": 50,
                    "detect_login_walls": True,
                    "minimum_retention_ratio": 0.5,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "html_dir": str(html_dir),
                "md_dir": str(md_dir),
                "mapping_file": str(mapping_file),
                "md_mapping_file": str(md_mapping_file),
                "page_media_file": str(page_media_file),
                "page_images_file": str(page_images_file),
                "page_videos_file": str(page_videos_file),
            },
            stage_id="score_raw_content",
        )

        result = run_async(QualityScorer().execute(ctx))
        assert result.status == StageStatus.COMPLETED
        assert result.outputs["passed_count"] == 1
        assert result.outputs["filtered_count"] == 1
        assert good_html.exists()
        assert good_md.exists()
        assert bad_html.exists()
        assert bad_md.exists()
        assert load_json_safe(mapping_file) == {
            "https://example.com/good": str(good_html),
            "https://example.com/login": str(bad_html),
            "https://example.com/not-found": "SKIPPED_HTTP_404",
            "https://example.com/verified-empty": "SKIPPED_VERIFIED_EMPTY_COHORT:test-policy",
        }
        assert load_json_safe(md_mapping_file) == {
            "https://example.com/good": str(good_md),
            "https://example.com/login": str(bad_md),
        }
        assert "https://example.com/login" in load_json_safe(page_media_file)
        assert "https://example.com/login" in load_json_safe(page_images_file)
        assert "https://example.com/login" in load_json_safe(page_videos_file)

        accepted_html = Path(result.outputs["html_dir"]) / "good.html"
        assert accepted_html.exists()
        assert not (Path(result.outputs["html_dir"]) / "login.html").exists()
        assert load_json_safe(result.outputs["mapping_file"]) == {
            "https://example.com/good": str(accepted_html.resolve()),
            "https://example.com/not-found": "SKIPPED_HTTP_404",
            "https://example.com/verified-empty": "SKIPPED_VERIFIED_EMPTY_COHORT:test-policy",
        }
        assert load_json_safe(result.outputs["md_mapping_file"]) == {
            "https://example.com/good": str(good_md),
        }
        assert load_json_safe(result.outputs["page_media_file"]) == {
            "https://example.com/good": [
                {"type": "image", "url": "https://example.com/good.jpg"}
            ]
        }
        assert load_json_safe(result.outputs["page_images_file"]) == {
            "https://example.com/good": [{"url": "https://example.com/good.jpg"}]
        }
        assert load_json_safe(result.outputs["page_videos_file"]) == {
            "https://example.com/good": [{"url": "https://example.com/good.mp4"}]
        }


# ─────────────────────────────────────────────────────────────
# 7. Dedup Filter — test actual near-duplicate detection
# ─────────────────────────────────────────────────────────────

try:
    import datasketch  # noqa: F401
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False

try:
    import markitdown  # noqa: F401
    HAS_MARKITDOWN = True
except ImportError:
    HAS_MARKITDOWN = False


class TestDedupFilter:
    @staticmethod
    def _run_catalog_case(tmp_dir, specs, *, identity_records=None, force_lsh_collision=False):
        from datasketch import MinHashLSH
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.quality.dedup_filter import DedupFilter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir(parents=True)
        catalog = ArtifactCatalog()
        mapping = {}
        paths = {}
        for index, spec in enumerate(specs):
            path = md_dir / spec["name"]
            path.write_text(spec["content"], encoding="utf-8")
            paths[spec["name"]] = path
            metadata = dict(spec.get("metadata") or {})
            mapping_url = spec.get("mapping_url")
            if mapping_url:
                source_file = tmp_dir / "downloads" / spec.get("source_name", f"source-{index}.pdf")
                source_file.parent.mkdir(parents=True, exist_ok=True)
                if "source_bytes" in spec:
                    source_file.write_bytes(spec["source_bytes"])
                elif "source_pdf_text" in spec:
                    import fitz

                    document = fitz.open()
                    page = document.new_page()
                    page.insert_textbox(
                        fitz.Rect(36, 36, 560, 806),
                        spec["source_pdf_text"],
                        fontsize=10,
                    )
                    if spec.get("source_pdf_fill") is not None:
                        page.draw_rect(
                            fitz.Rect(420, 680, 540, 780),
                            color=spec["source_pdf_fill"],
                            fill=spec["source_pdf_fill"],
                        )
                    pdf_metadata = dict(spec.get("pdf_metadata") or {})
                    if pdf_metadata:
                        document.set_metadata(pdf_metadata)
                    document.save(source_file)
                    document.close()
                metadata["source_file"] = str(source_file)
                mapping[mapping_url] = str(source_file)
            catalog.add(
                build_artifact_record(
                    artifact_type="markdown",
                    role="content",
                    producer_stage=spec.get("producer_stage", "convert_html"),
                    uri=path.resolve().as_uri(),
                    local_path=path,
                    metadata=metadata,
                    artifact_id=f"md-{index}",
                )
            )

        previous_outputs = {"md_dir": str(md_dir)}
        if identity_records is not None:
            identity_file = tmp_dir / "canonical_url_identity_map.json"
            atomic_write_json(identity_file, {"schema_version": 1, "records": identity_records})
            previous_outputs["url_identity_map_file"] = str(identity_file)
        if mapping:
            mapping_file = tmp_dir / "mappings.json"
            atomic_write_json(mapping_file, mapping)
            previous_outputs["mapping_file"] = str(mapping_file)

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"quality": {"dedup_threshold": 0.85, "dedup_num_perm": 128, "dedup_ngram_size": 5}},
            work_dir=tmp_dir,
            previous_outputs=previous_outputs,
            stage_definition={"type": "quality_gate", "plugin": "dedup_filter"},
            stage_id="deduplicate_markdown",
            artifact_catalog=catalog,
        )
        if force_lsh_collision:
            all_keys = [str(path.resolve()) for path in paths.values()]
            with patch.object(MinHashLSH, "query", return_value=all_keys):
                result = run_async(DedupFilter().execute(ctx))
        else:
            result = run_async(DedupFilter().execute(ctx))
        manifest = load_json_safe(result.outputs["dedup_manifest_file"])
        return result, manifest, paths

    def test_word_ngrams_normal(self):
        from pipeline.stages.quality.dedup_filter import _word_ngrams
        ngrams = _word_ngrams("the quick brown fox jumps over the lazy dog", 3)
        assert "the quick brown" in ngrams
        assert "quick brown fox" in ngrams
        assert len(ngrams) == 7  # 9 words, 3-grams: 9-3+1 = 7

    def test_word_ngrams_fewer_words_than_n(self):
        from pipeline.stages.quality.dedup_filter import _word_ngrams
        ngrams = _word_ngrams("hello world", 5)
        assert ngrams == ["hello world"]

    def test_word_ngrams_empty_text(self):
        from pipeline.stages.quality.dedup_filter import _word_ngrams
        assert _word_ngrams("", 5) == []

    def test_word_ngrams_whitespace_only(self):
        from pipeline.stages.quality.dedup_filter import _word_ngrams
        assert _word_ngrams("   \n\t  ", 5) == []

    def test_strip_boilerplate_preserves_markdown_table_schedule_facts(self):
        from pipeline.stages.quality.dedup_filter import _strip_boilerplate

        schedule = """| Day | Start | End | Activity |
| --- | :---: | :---: | ---: |
| Monday | 08:30 | 10:00 | Women only |
| Tuesday | 14:00 | 16:30 | Open swim |"""
        cleaned = _strip_boilerplate(schedule)
        assert "| Monday | 08:30 | 10:00 | Women only |" in cleaned
        assert "| Tuesday | 14:00 | 16:30 | Open swim |" in cleaned
        assert "| --- | :---: | :---: | ---: |" not in cleaned

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_lsh_collision_below_exact_threshold_is_preserved(self, tmp_dir):
        first = " ".join([f"shared{i}" for i in range(10)] + [f"alpha{i}" for i in range(30)])
        second = " ".join([f"shared{i}" for i in range(10)] + [f"beta{i}" for i in range(30)])
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {"name": "first.md", "content": first},
                {"name": "second.md", "content": second},
            ],
            force_lsh_collision=True,
        )
        assert result.outputs["filtered_count"] == 0
        assert all(path.exists() for path in paths.values())
        assert manifest["exact_pairs_verified"] >= 1

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_winner_is_deterministic_across_artifact_order(self, tmp_dir):
        content = "Deterministic duplicate content for MBZUAI. " * 30
        for order in (("z.md", "a.md"), ("a.md", "z.md")):
            case_dir = tmp_dir / f"case-{order[0][0]}"
            result, manifest, paths = self._run_catalog_case(
                case_dir,
                [{"name": name, "content": content} for name in order],
            )
            assert result.outputs["filtered_count"] == 1
            assert paths["a.md"].exists()
            assert not paths["z.md"].exists()
            assert Path(manifest["decisions"][0]["winner"]).name == "a.md"

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_current_canonical_route_beats_stale_route(self, tmp_dir):
        content = "MBZUAI leadership and institutional governance information. " * 35
        current = "https://mbzuai.ac.ae/about/leadership"
        stale = "https://mbzuai.ac.ae/leadership-prev"
        identities = [
            {
                "source_url": current,
                "canonical_url": current,
                "canonical_family_url": current,
                "language": "en",
            },
            {
                "source_url": stale,
                "canonical_url": stale,
                "canonical_family_url": current,
                "language": "en",
            },
        ]
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {"name": "stale.md", "content": content, "metadata": {"source_url": stale, "source_type": "webpage"}},
                {"name": "current.md", "content": content, "metadata": {"source_url": current, "source_type": "webpage"}},
            ],
            identity_records=identities,
        )
        assert result.outputs["filtered_count"] == 1
        assert paths["current.md"].exists()
        assert not paths["stale.md"].exists()
        assert manifest["decisions"][0]["winner_source_url"] == current
        assert manifest["decisions"][0]["reason"] == "exact_markdown_bytes"

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_arabic_url_locale_overrides_incorrect_english_identity(self, tmp_dir):
        content = "MBZUAI privacy policy and data protection information. " * 35
        english = "https://mbzuai.ac.ae/privacy-policy"
        arabic = "https://mbzuai.ac.ae/ar/privacy-policy-2"
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {"name": "en.md", "content": content, "metadata": {"source_url": english, "source_type": "webpage"}},
                {"name": "ar.md", "content": content, "metadata": {"source_url": arabic, "source_type": "webpage"}},
            ],
            identity_records=[
                {"source_url": english, "canonical_url": english, "canonical_family_url": english, "language": "en"},
                {
                    "source_url": arabic,
                    "canonical_url": english,
                    "canonical_family_url": english,
                    "language": "en",
                },
            ],
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert all(path.exists() for path in paths.values())

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_distinct_faculty_streams_and_document_revisions_are_preserved(self, tmp_dir):
        shared_web = "Faculty profile research interests publications teaching biography " * 25
        shared_doc = "Pool access schedule Monday Tuesday Wednesday opening hours " * 30
        faculty_a = "https://mbzuai.ac.ae/research/faculty/alice"
        faculty_b = "https://mbzuai.ac.ae/research/faculty/bob"
        stream_en = "https://mbzuai.ac.ae/study/ai-stream"
        stream_ar = "https://mbzuai.ac.ae/ar/study/ai-stream"
        venue_a = "https://mbzuai.ac.ae/campus/auditorium"
        venue_b = "https://mbzuai.ac.ae/campus/knowledge-centre"
        identities = [
            {"source_url": faculty_a, "canonical_url": faculty_a, "canonical_family_url": faculty_a, "language": "en"},
            {"source_url": faculty_b, "canonical_url": faculty_b, "canonical_family_url": faculty_b, "language": "en"},
            {"source_url": stream_en, "canonical_url": stream_en, "canonical_family_url": stream_en, "language": "en"},
            {"source_url": stream_ar, "canonical_url": stream_ar, "canonical_family_url": stream_en, "language": "ar"},
            {"source_url": venue_a, "canonical_url": venue_a, "canonical_family_url": venue_a, "language": "en"},
            {"source_url": venue_b, "canonical_url": venue_b, "canonical_family_url": venue_b, "language": "en"},
        ]
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {"name": "alice.md", "content": shared_web + "alice", "metadata": {"source_url": faculty_a, "source_type": "webpage"}},
                {"name": "bob.md", "content": shared_web + "bob", "metadata": {"source_url": faculty_b, "source_type": "webpage"}},
                {"name": "stream-en.md", "content": shared_web + "english", "metadata": {"source_url": stream_en, "source_type": "webpage"}},
                {"name": "stream-ar.md", "content": shared_web + "arabic", "metadata": {"source_url": stream_ar, "source_type": "webpage"}},
                {"name": "auditorium.md", "content": shared_web + "auditorium", "metadata": {"source_url": venue_a, "source_type": "webpage"}},
                {"name": "knowledge-centre.md", "content": shared_web + "knowledge centre", "metadata": {"source_url": venue_b, "source_type": "webpage"}},
                {
                    "name": "pool-2022.md",
                    "content": shared_doc + "08:00 10:00",
                    "metadata": {"source_type": "pdf"},
                    "mapping_url": "https://static.example/2022/MBZUAI-Pool-Schedule.pdf",
                    "source_name": "MBZUAI-Pool-Schedule-2022.pdf",
                    "source_pdf_text": shared_doc + "08:00 10:00",
                },
                {
                    "name": "pool-2024.md",
                    "content": shared_doc + "09:00 11:00",
                    "metadata": {"source_type": "pdf"},
                    "mapping_url": "https://static.example/2024/MBZUAI-Pool-Schedule.pdf",
                    "source_name": "MBZUAI-Pool-Schedule-2024.pdf",
                    "source_pdf_text": shared_doc + "09:00 11:00",
                },
            ],
            identity_records=identities,
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert all(path.exists() for path in paths.values())

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_text_equal_catalogues_with_different_source_bytes_are_preserved(self, tmp_dir):
        body = " ".join(
            f"programme{i} policy{i} course{i} credits{i}"
            for i in range(80)
        )
        source_text = "MBZUAI University Catalogue 2023-2024\n" + body
        first_markdown = "# Catalogue 2023-2024\n" + body
        second_markdown = "# MBZUAI University Catalogue 2023-2024\n" + body
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {
                    "name": "dated.md",
                    "content": first_markdown,
                    "metadata": {"source_type": "pdf"},
                    "mapping_url": "https://static.example/2023/08/University-Catalogue-2023-24.pdf",
                    "source_name": "University-Catalogue-2023-24.pdf",
                    "source_pdf_text": source_text,
                    "pdf_metadata": {"modDate": "D:20230830162537+04'00'"},
                },
                {
                    "name": "canonical.md",
                    "content": second_markdown,
                    "metadata": {"source_type": "pdf"},
                    "mapping_url": "https://static.example/PDF/University_Catalogue.pdf",
                    "source_name": "University_Catalogue.pdf",
                    "source_pdf_text": source_text,
                    "pdf_metadata": {"modDate": "D:20230912065330Z"},
                },
            ],
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert paths["canonical.md"].exists()
        assert paths["dated.md"].exists()

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_text_equal_pdfs_with_different_graphics_are_preserved(self, tmp_dir):
        content = "Identical extracted text with a semantic graphic. " * 40
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {
                    "name": "red.md",
                    "content": content,
                    "metadata": {"source_type": "pdf"},
                    "mapping_url": "https://static.example/red.pdf",
                    "source_name": "red.pdf",
                    "source_pdf_text": content,
                    "source_pdf_fill": (1.0, 0.0, 0.0),
                },
                {
                    "name": "blue.md",
                    "content": content,
                    "metadata": {"source_type": "pdf"},
                    "mapping_url": "https://static.example/blue.pdf",
                    "source_name": "blue.pdf",
                    "source_pdf_text": content,
                    "source_pdf_fill": (0.0, 0.0, 1.0),
                },
            ],
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert all(path.exists() for path in paths.values())

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_exact_revised_catalogue_content_prefers_fresher_pdf_url(self, tmp_dir):
        content = "Identical normalized catalogue publication content. " * 40
        source_bytes = b"byte-identical source publication"
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {
                    "name": "catalogue-2023.md",
                    "content": content,
                    "metadata": {
                        "source_type": "pdf",
                        "source_url": "https://static.example/2023/08/University-Catalogue.pdf",
                        "quality_score": 1.0,
                    },
                    "mapping_url": "https://static.example/2023/08/University-Catalogue.pdf",
                    "source_name": "University-Catalogue-2023.pdf",
                    "source_bytes": source_bytes,
                },
                {
                    "name": "catalogue-2025.md",
                    "content": content,
                    "metadata": {
                        "source_type": "pdf",
                        "source_url": "https://static.example/2025/05/University-Catalogue.pdf",
                        "quality_score": 0.9,
                    },
                    "mapping_url": "https://static.example/2025/05/University-Catalogue.pdf",
                    "source_name": "University-Catalogue-2025.pdf",
                    "source_bytes": source_bytes,
                },
            ],
        )
        assert result.outputs["filtered_count"] == 1
        assert paths["catalogue-2025.md"].exists()
        assert not paths["catalogue-2023.md"].exists()
        assert manifest["decisions"][0]["winner_source_url"].startswith("https://static.example/2025/")
        assert manifest["decisions"][0]["reason"] == "exact_source_document_bytes"

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_execute_preserves_nonexact_near_duplicate_web_aliases(self, tmp_dir):
        """Canonical identity and similarity do not authorize non-exact deletion."""
        original = "MBZUAI is the world's first graduate-level research-based AI university. " * 20
        duplicate = "MBZUAI is the world's first graduate-level research-based AI university. " * 20
        duplicate += " Some tiny addition."
        different = "The weather in Abu Dhabi is typically hot and sunny throughout the year. " * 20
        canonical = "https://mbzuai.ac.ae/about"
        alias = "https://mbzuai.ac.ae/about-us"
        weather = "https://mbzuai.ac.ae/weather"
        result, _manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {"name": "original.md", "content": original, "metadata": {"source_url": canonical, "source_type": "webpage"}},
                {"name": "duplicate.md", "content": duplicate, "metadata": {"source_url": alias, "source_type": "webpage"}},
                {"name": "different.md", "content": different, "metadata": {"source_url": weather, "source_type": "webpage"}},
            ],
            identity_records=[
                {"source_url": canonical, "canonical_url": canonical, "canonical_family_url": canonical, "language": "en"},
                {"source_url": alias, "canonical_url": canonical, "canonical_family_url": canonical, "language": "en"},
                {"source_url": weather, "canonical_url": weather, "canonical_family_url": weather, "language": "en"},
            ],
        )
        assert result.outputs["filtered_count"] == 0
        assert result.outputs["passed_count"] == 3
        assert paths["different.md"].exists()
        assert paths["original.md"].exists()
        assert paths["duplicate.md"].exists()

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_generic_near_duplicates_without_identity_fail_closed(self, tmp_dir):
        shared = "Faculty biography research publications teaching awards and service. " * 30
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {"name": "faculty-a.md", "content": shared + "Alice leads vision research."},
                {"name": "faculty-b.md", "content": shared + "Bob leads language research."},
            ],
            force_lsh_collision=True,
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert all(path.exists() for path in paths.values())

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_distinct_resource_links_are_not_treated_as_exact_content(self, tmp_dir):
        shared = "Student resources, wellbeing, and campus support information. " * 30
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {
                    "name": "scholarships.md",
                    "content": shared + "\n- [Scholarship support](/students/scholarships)",
                },
                {
                    "name": "emergency.md",
                    "content": shared + "\n- [Emergency support](/students/emergency-support)",
                },
            ],
            force_lsh_collision=True,
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert all(path.exists() for path in paths.values())

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_same_family_case_sensitive_resource_links_fail_closed(self, tmp_dir):
        shared = "MBZUAI policy information for students and university staff. " * 30
        current = "https://mbzuai.ac.ae/policies/current"
        legacy = "https://mbzuai.ac.ae/policies/legacy"
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {
                    "name": "current.md",
                    "content": shared + "\n[Download policy](/documents/Policy.PDF)",
                    "metadata": {"source_url": current, "source_type": "webpage"},
                },
                {
                    "name": "legacy.md",
                    "content": shared + "\n[Download policy](/documents/policy.pdf)",
                    "metadata": {"source_url": legacy, "source_type": "webpage"},
                },
            ],
            identity_records=[
                {
                    "source_url": current,
                    "canonical_url": current,
                    "canonical_family_url": current,
                    "language": "en",
                },
                {
                    "source_url": legacy,
                    "canonical_url": legacy,
                    "canonical_family_url": current,
                    "language": "en",
                },
            ],
            force_lsh_collision=True,
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert all(path.exists() for path in paths.values())

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_web_page_is_not_removed_in_favor_of_document(self, tmp_dir):
        content = "President address on MBZUAI strategy, students, and research. " * 35
        web_url = "https://mbzuai.ac.ae/about/president-address"
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {
                    "name": "president-address.md",
                    "content": content,
                    "metadata": {"source_url": web_url, "source_type": "webpage"},
                },
                {
                    "name": "president-address-document.md",
                    "content": content,
                    "metadata": {"source_type": "pdf"},
                    "mapping_url": "https://static.example/president-address.pdf",
                    "source_name": "president-address.pdf",
                    "source_bytes": b"president address document",
                },
            ],
            identity_records=[
                {
                    "source_url": web_url,
                    "canonical_url": web_url,
                    "canonical_family_url": web_url,
                    "language": "en",
                }
            ],
            force_lsh_collision=True,
        )
        assert result.outputs["filtered_count"] == 0
        assert manifest["decisions"] == []
        assert all(path.exists() for path in paths.values())

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_similarity_chain_does_not_authorize_nonexact_deletion(self, tmp_dir):
        common = [f"common{i}" for i in range(200)]
        content_a = " ".join(common + [f"a{i}" for i in range(20)])
        content_b = " ".join(common + [f"a{i}" for i in range(10)] + [f"c{i}" for i in range(10, 20)])
        content_c = " ".join(common + [f"c{i}" for i in range(20)])
        family = "https://mbzuai.ac.ae/example-family"
        urls = [f"https://mbzuai.ac.ae/example-{suffix}" for suffix in "abc"]
        result, manifest, paths = self._run_catalog_case(
            tmp_dir,
            [
                {
                    "name": f"{suffix}.md",
                    "content": content,
                    "metadata": {"source_url": url, "source_type": "webpage"},
                }
                for suffix, content, url in zip("abc", (content_a, content_b, content_c), urls)
            ],
            identity_records=[
                {
                    "source_url": url,
                    "canonical_url": url,
                    "canonical_family_url": family,
                    "language": "en",
                }
                for url in urls
            ],
            force_lsh_collision=True,
        )
        assert result.outputs["filtered_count"] == 0
        assert paths["a.md"].exists()
        assert paths["b.md"].exists()
        assert paths["c.md"].exists()
        assert manifest["decisions"] == []

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_execute_keeps_all_unique_files(self, tmp_dir):
        """Completely different files should all survive dedup."""
        from pipeline.stages.quality.dedup_filter import DedupFilter
        from pipeline.core.base import StageContext

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()

        (md_dir / "a.md").write_text("Machine learning is a subset of AI. " * 20)
        (md_dir / "b.md").write_text("The university campus is located in Masdar City. " * 20)
        (md_dir / "c.md").write_text("Natural language processing involves understanding text. " * 20)

        ctx = StageContext(
            run_id="test", project_name="test",
            config={"quality": {"dedup_threshold": 0.85, "dedup_num_perm": 128, "dedup_ngram_size": 5}},
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
        )

        result = run_async(DedupFilter().execute(ctx))
        assert result.outputs["filtered_count"] == 0
        assert result.outputs["passed_count"] == 3

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_execute_skips_when_no_files(self, tmp_dir):
        from pipeline.stages.quality.dedup_filter import DedupFilter
        from pipeline.core.base import StageContext, StageStatus

        md_dir = tmp_dir / "empty_markdown"
        md_dir.mkdir()

        ctx = StageContext(
            run_id="test", project_name="test",
            config={"quality": {}},
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
        )

        result = run_async(DedupFilter().execute(ctx))
        assert result.status == StageStatus.SKIPPED

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_execute_rewrites_mapping_to_surviving_outputs(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.quality.dedup_filter import DedupFilter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        first = md_dir / "first.md"
        second = md_dir / "second.md"
        first.write_text("MBZUAI duplicate content " * 30, encoding="utf-8")
        second.write_text("MBZUAI duplicate content " * 30, encoding="utf-8")

        previous_mapping = tmp_dir / "convert_html_url_to_md_mapping.json"
        atomic_write_json(
            previous_mapping,
            {
                "https://example.com/first": str(first),
                "https://example.com/second": str(second),
            },
        )
        page_media_file = tmp_dir / "page_media.json"
        atomic_write_json(
            page_media_file,
            {
                "https://example.com/first": [{"type": "image", "url": "https://example.com/first.jpg"}],
                "https://example.com/second": [{"type": "image", "url": "https://example.com/second.jpg"}],
            },
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"quality": {"dedup_threshold": 0.85, "dedup_num_perm": 128, "dedup_ngram_size": 5}},
            work_dir=tmp_dir,
            previous_outputs={
                "md_dir": str(md_dir),
                "md_mapping_file": str(previous_mapping),
                "page_media_file": str(page_media_file),
            },
            stage_definition={"type": "quality_gate", "plugin": "dedup_filter"},
            stage_id="deduplicate_markdown",
        )

        result = run_async(DedupFilter().execute(ctx))
        final_mapping = load_json_safe(result.outputs["md_mapping_file"])
        previous_mapping_payload = load_json_safe(previous_mapping)
        root_mapping = load_json_safe(tmp_dir / "url_to_md_mapping.json")
        assert result.outputs["filtered_count"] == 1
        assert len(final_mapping) == 1
        assert final_mapping == previous_mapping_payload == root_mapping
        surviving_url = next(iter(final_mapping.keys()))
        surviving_path = Path(next(iter(final_mapping.values())))
        assert surviving_path.exists()
        assert load_json_safe(page_media_file) == {
            surviving_url: [{"type": "image", "url": f"https://example.com/{surviving_url.rsplit('/', 1)[-1]}.jpg"}]
        }

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_execute_prunes_dependent_document_artifacts_for_removed_markdown(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.quality.dedup_filter import DedupFilter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        kept_md = md_dir / "kept.md"
        dup_md = md_dir / "duplicate.md"
        kept_md.write_text("MBZUAI duplicate content " * 30, encoding="utf-8")
        dup_md.write_text("MBZUAI duplicate content " * 30, encoding="utf-8")
        source_document = tmp_dir / "downloads" / "same-source.pdf"
        source_document.parent.mkdir(parents=True)
        source_document.write_bytes(b"byte-identical source document")

        report = tmp_dir / "quality_reports" / "duplicate.validation.json"
        report.parent.mkdir(parents=True)
        atomic_write_json(
            report,
            {
                "source_file": str(source_document),
                "selected_backend": "docling",
                "selected_markdown_path": str(dup_md.resolve()),
                "quarantined": False,
                "selected_assessment": {"accepted": True, "score": 1.0, "reasons": [], "warnings": [], "metrics": {}},
            },
        )

        structured = tmp_dir / "structured_documents" / "duplicate.docling.json"
        structured.parent.mkdir(parents=True)
        structured.write_text('{"ok": true}', encoding="utf-8")

        image = tmp_dir / "extracted_images" / "duplicate" / "figure.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")

        catalog = ArtifactCatalog()
        catalog.extend(
            [
                build_artifact_record(
                    artifact_type="markdown",
                    role="content",
                    producer_stage="convert_documents",
                    uri=kept_md.resolve().as_uri(),
                    local_path=kept_md,
                    metadata={"source_file": str(source_document), "source_type": "pdf", "quality_score": 1.0},
                    artifact_id="md-kept",
                ),
                build_artifact_record(
                    artifact_type="markdown",
                    role="content",
                    producer_stage="convert_documents",
                    uri=dup_md.resolve().as_uri(),
                    local_path=dup_md,
                    metadata={"source_file": str(source_document), "source_type": "pdf", "quality_score": 0.9},
                    artifact_id="md-dup",
                ),
                build_artifact_record(
                    artifact_type="document_quality_report",
                    role="quality_report",
                    producer_stage="convert_documents",
                    uri=report.resolve().as_uri(),
                    local_path=report,
                    metadata={"source_file": str(source_document)},
                    artifact_id="report-dup",
                ),
                build_artifact_record(
                    artifact_type="structured_document",
                    role="docling_document",
                    producer_stage="convert_documents",
                    uri=structured.resolve().as_uri(),
                    local_path=structured,
                    metadata={"source_markdown_path": str(dup_md.resolve())},
                    artifact_id="structured-dup",
                ),
                build_artifact_record(
                    artifact_type="extracted_image",
                    role="document_media",
                    producer_stage="convert_documents",
                    uri=image.resolve().as_uri(),
                    local_path=image,
                    metadata={"source_document_path": str(dup_md.resolve())},
                    artifact_id="image-dup",
                ),
            ]
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"quality": {"dedup_threshold": 0.85, "dedup_num_perm": 128, "dedup_ngram_size": 5}},
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
            stage_definition={"type": "quality_gate", "plugin": "dedup_filter"},
            stage_id="deduplicate_markdown",
            artifact_catalog=catalog,
        )

        result = run_async(DedupFilter().execute(ctx))

        assert result.outputs["filtered_count"] == 1
        assert not dup_md.exists()
        assert not report.exists()
        assert not structured.exists()
        assert not image.exists()
        assert {"md-dup", "report-dup", "structured-dup", "image-dup"}.issubset(set(result.removed_artifact_ids))
        assert result.metrics["dependent_artifacts_removed"] == 3
        manifest = load_json_safe(result.outputs["dedup_manifest_file"])
        assert manifest["application_status"] == "applied"
        assert manifest["planned_removed_markdown_paths"] == [str(dup_md.resolve())]
        assert {item["artifact_id"] for item in manifest["planned_dependent_artifacts"]} == {
            "report-dup",
            "structured-dup",
            "image-dup",
        }
        assert manifest["dependent_artifacts_removed"] == manifest["planned_dependent_artifacts"]

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_dedup_does_not_prune_unrelated_orphan_artifacts(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.io import load_json_safe
        from pipeline.stages.quality.dedup_filter import DedupFilter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        unique_md = md_dir / "unique.md"
        unique_md.write_text("Unique MBZUAI research content. " * 30, encoding="utf-8")
        missing_md = md_dir / "already-missing.md"
        orphan = tmp_dir / "structured_documents" / "orphan.docling.json"
        orphan.parent.mkdir(parents=True)
        orphan.write_text('{"orphan": true}', encoding="utf-8")

        catalog = ArtifactCatalog()
        catalog.extend(
            [
                build_artifact_record(
                    artifact_type="markdown",
                    role="content",
                    producer_stage="convert_html",
                    uri=unique_md.resolve().as_uri(),
                    local_path=unique_md,
                    metadata={},
                    artifact_id="md-unique",
                ),
                build_artifact_record(
                    artifact_type="structured_document",
                    role="docling_document",
                    producer_stage="convert_documents",
                    uri=orphan.resolve().as_uri(),
                    local_path=orphan,
                    metadata={"source_markdown_path": str(missing_md.resolve())},
                    artifact_id="structured-orphan",
                ),
            ]
        )
        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"quality": {"dedup_threshold": 0.85, "dedup_num_perm": 128, "dedup_ngram_size": 5}},
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
            stage_definition={"type": "quality_gate", "plugin": "dedup_filter"},
            stage_id="deduplicate_markdown",
            artifact_catalog=catalog,
        )

        result = run_async(DedupFilter().execute(ctx))
        manifest = load_json_safe(result.outputs["dedup_manifest_file"])
        assert result.outputs["filtered_count"] == 0
        assert result.metrics["dependent_artifacts_removed"] == 0
        assert result.removed_artifact_ids == []
        assert orphan.exists()
        assert manifest["planned_dependent_artifacts"] == []
        assert manifest["dependent_artifacts_removed"] == []

    @pytest.mark.skipif(not HAS_DATASKETCH, reason="datasketch not installed")
    def test_dedup_manifest_is_written_before_markdown_unlink(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import load_json_safe
        from pipeline.stages.quality.dedup_filter import DedupFilter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        winner = md_dir / "a.md"
        loser = md_dir / "z.md"
        content = "Atomic deduplication decision evidence. " * 40
        winner.write_text(content, encoding="utf-8")
        loser.write_text(content, encoding="utf-8")
        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"quality": {"dedup_threshold": 0.85, "dedup_num_perm": 128, "dedup_ngram_size": 5}},
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
            stage_definition={"type": "quality_gate", "plugin": "dedup_filter"},
            stage_id="deduplicate_markdown",
        )
        manifest_path = ctx.stage_work_dir / "dedup_manifest.json"
        original_unlink = Path.unlink
        observed = []

        def guarded_unlink(path, *args, **kwargs):
            if path.resolve() == loser.resolve():
                payload = load_json_safe(manifest_path)
                observed.append(payload)
                assert payload["application_status"] == "planned"
                assert payload["planned_removed_markdown_paths"] == [str(loser.resolve())]
                raise RuntimeError("simulated interruption before unlink")
            return original_unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", guarded_unlink):
            with pytest.raises(RuntimeError, match="simulated interruption"):
                run_async(DedupFilter().execute(ctx))

        assert len(observed) == 1
        assert winner.exists()
        assert loser.exists()
        assert load_json_safe(manifest_path)["application_status"] == "planned"


# ─────────────────────────────────────────────────────────────
# 8. BS4 Cleaner — test real HTML cleaning
# ─────────────────────────────────────────────────────────────

class TestBS4Cleaner:
    def test_removes_script_nav_footer(self, tmp_dir):
        from pipeline.stages.cleaners.bs4_cleaner import clean_single_file
        html = """<html><head><title>Test</title>
        <script>var x = malicious_code();</script>
        <style>.hidden { display: none; }</style>
        </head><body>
        <nav><ul><li>Home</li><li>About</li></ul></nav>
        <main><h1>Real Content</h1><p>Important paragraph text.</p></main>
        <footer>Copyright 2024</footer>
        </body></html>"""

        fp = tmp_dir / "test.html"
        fp.write_text(html)
        status = clean_single_file(str(fp))
        assert status == "cleaned"

        cleaned = fp.read_text()
        assert "<script>" not in cleaned
        assert "<style>" not in cleaned
        assert "<nav>" not in cleaned
        assert "<footer>" not in cleaned
        assert "Real Content" in cleaned or "Important paragraph" in cleaned

    def test_removes_404_error_page(self, tmp_dir):
        from pipeline.stages.cleaners.bs4_cleaner import clean_single_file
        fp = tmp_dir / "404.html"
        fp.write_text("<html><body><h1>Page not found</h1></body></html>")
        status = clean_single_file(str(fp))
        assert status == "removed"
        assert not fp.exists()

    def test_removes_empty_page(self, tmp_dir):
        """Page with only scripts and no content should be removed."""
        from pipeline.stages.cleaners.bs4_cleaner import clean_single_file
        fp = tmp_dir / "empty.html"
        fp.write_text("<html><head><script>all js</script></head><body><script>more</script></body></html>")
        status = clean_single_file(str(fp))
        assert status == "removed_empty"
        assert not fp.exists()

    def test_clean_html_content_handles_tags_with_missing_attrs(self):
        from bs4 import BeautifulSoup
        from pipeline.stages.cleaners.bs4_cleaner import _remove_hidden, clean_html_content

        soup = BeautifulSoup('<html><body><div style="display:none">x</div></body></html>', "html.parser")
        broken = soup.find("div")
        broken.attrs = None

        _remove_hidden(soup)

        status, cleaned_html = clean_html_content("<html><body><div>Visible content</div></body></html>")
        assert status == "cleaned"
        assert "Visible content" in cleaned_html

    def test_hidden_content_with_spaced_css_is_removed_before_attributes(self):
        from pipeline.stages.cleaners.bs4_cleaner import clean_html_content

        raw = """<html><body><main>
        <div style="display: none">Hidden prompt injection text</div>
        <div style="visibility : hidden">Hidden navigation text</div>
        <p>Visible public university content.</p>
        </main></body></html>"""

        status, cleaned_html = clean_html_content(raw)

        assert status == "cleaned"
        assert "Visible public university content" in cleaned_html
        assert "Hidden prompt injection text" not in cleaned_html
        assert "Hidden navigation text" not in cleaned_html

    def test_page_not_found_phrase_inside_script_does_not_remove_public_page(self):
        from pipeline.stages.cleaners.bs4_cleaner import clean_html_content

        raw = """<html><head><script>const message = 'page not found';</script></head>
        <body><main><h1>Public page</h1><p>{content}</p></main></body></html>""".format(
            content="Useful public university information. " * 10
        )
        status, cleaned_html = clean_html_content(raw)

        assert status == "cleaned"
        assert "Public page" in cleaned_html

    def test_page_not_found_heading_inside_long_article_is_not_an_error_shell(self):
        from pipeline.stages.cleaners.bs4_cleaner import clean_html_content

        raw = """<html><body><main><h1>Web reliability handbook</h1>
        <p>{content}</p><h2>Page not found</h2>
        <p>This section explains how public 404 responses are designed.</p>
        </main></body></html>""".format(content="Substantive engineering guidance. " * 80)

        status, cleaned_html = clean_html_content(raw)

        assert status == "cleaned"
        assert "Web reliability handbook" in cleaned_html

    def test_main_element_is_preferred_over_earlier_article_teaser(self):
        from pipeline.stages.cleaners.bs4_cleaner import clean_html_content

        raw = """<html><body>
        <article><p>Unrelated teaser</p></article>
        <main><h1>Primary page</h1><p>{content}</p></main>
        </body></html>""".format(content="Substantive primary content. " * 20)

        status, cleaned_html = clean_html_content(raw)

        assert status == "cleaned"
        assert "Primary page" in cleaned_html
        assert "Unrelated teaser" not in cleaned_html

    def test_multiple_article_cards_do_not_discard_sibling_page_content(self):
        from pipeline.stages.cleaners.bs4_cleaner import clean_html_content

        raw = """<html><body>
        <article>Small card</article><article>Second card</article>
        <section><h1>Undergraduate research program</h1>
        <p>{content}</p></section>
        </body></html>""".format(content="Substantive program information. " * 20)

        status, cleaned_html = clean_html_content(raw)

        assert status == "cleaned"
        assert "Undergraduate research program" in cleaned_html
        assert "Substantive program information" in cleaned_html


# ─────────────────────────────────────────────────────────────
# 9. Formatter — test real formatting with image metadata
# ─────────────────────────────────────────────────────────────

class TestFormatterImages:
    def test_extract_images_deduplicates(self):
        """Same URL with different alt text — should only appear once."""
        from pipeline.stages.formatters.pinecone_formatter import _extract_images_from_markdown
        md = """![First](https://example.com/img.jpg)
![Second](https://example.com/img.jpg)
![Different](https://example.com/other.jpg)"""
        images = _extract_images_from_markdown(md)
        assert len(images) == 2
        assert images[0]["url"] == "https://example.com/img.jpg"
        assert images[0]["alt"] == "First"  # keeps first occurrence

    def test_extract_images_handles_no_images(self):
        from pipeline.stages.formatters.pinecone_formatter import _extract_images_from_markdown
        assert _extract_images_from_markdown("# Title\nJust text.") == []

    def test_format_document_includes_all_sections(self):
        from pipeline.stages.formatters.pinecone_formatter import format_document
        doc = {
            "document_title": "MBZUAI Overview",
            "document_type": "webpage",
            "document_date": "2024-01-15",
            "detailed_summary": "Summary of MBZUAI programs.",
            "key_facts": ["Founded 2019", "Located in Abu Dhabi"],
            "keywords": ["AI", "machine learning"],
            "entities": {
                "organizations": ["MBZUAI"],
                "locations": ["Abu Dhabi"],
            },
            "page_content": "Full page content here.",
        }
        text = format_document(doc, include_full_content=True, include_summary=True)
        assert "TITLE: MBZUAI Overview" in text
        assert "TYPE: webpage" in text
        assert "DATE: 2024-01-15" in text
        assert "Summary of MBZUAI programs" in text
        assert "Founded 2019" in text
        assert "AI, machine learning" in text
        assert "Organizations: MBZUAI" in text
        assert "FULL CONTENT:" in text
        assert "Full page content here." in text

    def test_format_document_without_summary(self):
        from pipeline.stages.formatters.pinecone_formatter import format_document
        doc = {
            "document_title": "Test",
            "detailed_summary": "Should not appear",
            "page_content": "Content only.",
        }
        text = format_document(doc, include_full_content=True, include_summary=False)
        assert "TITLE: Test" in text
        assert "SUMMARY:" not in text
        assert "Content only." in text

    def test_format_document_without_content(self):
        from pipeline.stages.formatters.pinecone_formatter import format_document
        doc = {
            "document_title": "Test",
            "detailed_summary": "Summary here",
            "page_content": "Should not appear",
        }
        text = format_document(doc, include_full_content=False, include_summary=True)
        assert "SUMMARY:" in text
        assert "FULL CONTENT:" not in text

    def test_execute_merges_markdown_and_crawler_images(self, tmp_dir):
        """Full execute: verify images from markdown + crawler's page_images are merged."""
        from pipeline.stages.formatters.pinecone_formatter import PineconeFormatter
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()

        # Create a markdown file with an inline image
        (md_dir / "page1.md").write_text(
            "# Title\n\n![Chart](https://example.com/chart.png)\n\nContent here."
        )
        # Create matching summary
        atomic_write_json(summaries_dir / "page1.summary.json", {
            "document_title": "Page 1",
            "document_type": "webpage",
            "detailed_summary": "A page about charts.",
        })

        # Crawler found additional images on the page
        page_images = {
            "https://example.com/page1": [
                {"url": "https://example.com/chart.png", "alt": "Chart"},  # duplicate
                {"url": "https://example.com/photo.jpg", "alt": "Photo", "local_path": "/img/photo.jpg", "context": ""},
            ]
        }

        # Create URL mapping: page URL -> md file
        url_to_md_mapping = {"https://example.com/page1": str(md_dir / "page1.md")}
        mapping_file = tmp_dir / "url_to_md.json"
        atomic_write_json(mapping_file, url_to_md_mapping)

        ctx = StageContext(
            run_id="test", project_name="test",
            config={"formatter": {"include_full_content": True, "include_summary": True, "max_images_per_doc": 5}},
            work_dir=tmp_dir,
            previous_outputs={
                "summaries_dir": str(summaries_dir),
                "md_dir": str(md_dir),
                "md_mapping_file": str(mapping_file),
                "page_images": page_images,
            },
        )

        formatter = PineconeFormatter()
        result = run_async(formatter.execute(ctx))

        assert result.outputs["formatted_count"] == 1

        # Read the output file and verify images
        from pipeline.core.io import load_json_safe
        docs = load_json_safe(tmp_dir / "formatted_for_embedding.json")
        assert len(docs) == 1
        metadata = docs[0]["metadata"]
        # Should have 2 images (chart from markdown + photo from crawler, chart deduped)
        assert "images" in metadata
        image_urls = [img["url"] for img in metadata["images"]]
        assert "https://example.com/chart.png" in image_urls
        assert "https://example.com/photo.jpg" in image_urls

    def test_execute_attaches_videos_and_unified_media(self, tmp_dir):
        from pipeline.stages.formatters.pinecone_formatter import PineconeFormatter
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()

        (md_dir / "page1.md").write_text("# Title\n\nContent here.")
        atomic_write_json(
            summaries_dir / "page1.summary.json",
            {
                "document_title": "Page 1",
                "document_type": "webpage",
                "detailed_summary": "A page with rich media.",
            },
        )

        page_media = {
            "https://example.com/page1": [
                {
                    "type": "image",
                    "url": "https://example.com/photo.jpg",
                    "alt": "Photo",
                    "caption": "Campus photo",
                },
                {
                    "type": "video",
                    "url": "https://example.com/intro.mp4",
                    "title": "Intro video",
                    "caption": "Watch the campus introduction",
                    "poster_url": "https://example.com/intro.jpg",
                    "transcript": "Welcome to MBZUAI.",
                },
            ]
        }

        mapping_file = tmp_dir / "url_to_md.json"
        atomic_write_json(mapping_file, {"https://example.com/page1": str(md_dir / "page1.md")})

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={
                "formatter": {
                    "include_full_content": True,
                    "include_summary": True,
                    "include_media_context": True,
                    "max_media_per_doc": 5,
                    "max_images_per_doc": 5,
                    "max_videos_per_doc": 2,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "summaries_dir": str(summaries_dir),
                "md_dir": str(md_dir),
                "md_mapping_file": str(mapping_file),
                "page_media": page_media,
            },
        )

        result = run_async(PineconeFormatter().execute(ctx))
        assert result.outputs["formatted_count"] == 1

        docs = load_json_safe(tmp_dir / "formatted_for_embedding.json")
        metadata = docs[0]["metadata"]
        assert "media" in metadata
        assert "videos" in metadata
        assert metadata["videos"][0]["url"] == "https://example.com/intro.mp4"
        assert "MEDIA CONTEXT:" in docs[0]["text"]
        assert "Welcome to MBZUAI." in docs[0]["text"]

    def test_execute_chunks_long_documents_and_loads_structured_media_manifest(self, tmp_dir):
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.core.media import build_media_manifest
        from pipeline.core.base import StageContext
        from pipeline.stages.formatters.pinecone_formatter import PineconeFormatter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()

        paragraphs = [
            f"Paragraph {idx}: " + ("MBZUAI research content " * 18).strip()
            for idx in range(1, 18)
        ]
        md_file = md_dir / "report.md"
        md_file.write_text("# Report\n\n" + "\n\n".join(paragraphs), encoding="utf-8")

        atomic_write_json(
            summaries_dir / "report.summary.json",
            {
                "document_title": "Report",
                "document_type": "pdf",
                "detailed_summary": "Structured PDF report",
                "source_original_file": str(md_file),
            },
        )

        image_path = tmp_dir / "images" / "figure.png"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake-image")

        manifest_path = tmp_dir / "extracted_images_index.json"
        atomic_write_json(
            manifest_path,
            build_media_manifest(
                [
                    {
                        "id": "img1",
                        "type": "image",
                        "url": image_path.resolve().as_uri(),
                        "asset_uri": image_path.resolve().as_uri(),
                        "local_path": str(image_path.resolve()),
                        "md_path": str(md_file),
                        "document_id": "doc-1",
                        "page_number": 1,
                        "alt": "Campus map",
                        "caption": "Campus map",
                        "description": "Annotated campus map",
                        "source_type": "pdf",
                    }
                ],
                kind="document_media",
            ),
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={
                "formatter": {
                    "include_full_content": True,
                    "include_summary": True,
                    "include_media_context": True,
                    "chunk_documents": True,
                    "chunk_size_chars": 500,
                    "chunk_overlap_chars": 80,
                    "max_chunks_per_document": 0,
                    "max_images_per_doc": 5,
                    "max_videos_per_doc": 2,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "summaries_dir": str(summaries_dir),
                "md_dir": str(md_dir),
                "extracted_images_index_file": str(manifest_path),
            },
        )

        result = run_async(PineconeFormatter().execute(ctx))
        docs = load_json_safe(result.outputs["formatted_file"])

        assert result.outputs["source_document_count"] == 1
        assert result.outputs["formatted_count"] > 1
        assert len(docs) == result.outputs["formatted_count"]
        assert all(doc["metadata"]["document_id"] == docs[0]["metadata"]["document_id"] for doc in docs)
        assert docs[0]["metadata"]["chunk_count"] == len(docs)
        assert docs[0]["metadata"]["images"][0]["asset_uri"].startswith("file://")
        assert docs[0]["metadata"]["images"][0]["page_number"] == 1
        assert docs[0]["metadata"]["images"][0]["description"] == "Annotated campus map"
        assert "CHUNK 1/" in docs[0]["text"]
        assert "Paragraph 17:" in "\n".join(doc["text"] for doc in docs)

    def test_execute_prefers_chunk_manifest_when_available(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.pinecone_formatter import PineconeFormatter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()

        md_file = md_dir / "page1.md"
        md_file.write_text("# Research\n\nSection A\n\nSection B", encoding="utf-8")
        atomic_write_json(
            summaries_dir / "page1.summary.json",
            {
                "document_title": "Research",
                "document_type": "webpage",
                "detailed_summary": "Research summary",
                "source_original_file": str(md_file),
            },
        )
        mapping_file = tmp_dir / "url_to_md.json"
        atomic_write_json(mapping_file, {"https://example.com/research": str(md_file)})
        chunks_file = tmp_dir / "chunks.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc1",
                        "chunk_index": 0,
                        "chunk_count": 2,
                        "strategy": "hybrid",
                        "text": "Section A chunk",
                        "section_path": ["Research", "A"],
                        "element_types": ["paragraph"],
                        "source_markdown_path": str(md_file),
                        "source_url": "https://example.com/research",
                        "document_title": "Research",
                        "document_type": "webpage",
                    },
                    {
                        "document_id": "doc1",
                        "chunk_index": 1,
                        "chunk_count": 2,
                        "strategy": "hybrid",
                        "text": "Section B chunk",
                        "section_path": ["Research", "B"],
                        "element_types": ["paragraph"],
                        "source_markdown_path": str(md_file),
                        "source_url": "https://example.com/research",
                        "document_title": "Research",
                        "document_type": "webpage",
                    },
                ],
                strategy="hybrid",
            ),
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"formatter": {"include_full_content": True, "include_summary": True}},
            work_dir=tmp_dir,
            previous_outputs={
                "summaries_dir": str(summaries_dir),
                "md_dir": str(md_dir),
                "md_mapping_file": str(mapping_file),
                "chunks_file": str(chunks_file),
            },
        )

        result = run_async(PineconeFormatter().execute(ctx))
        docs = load_json_safe(result.outputs["formatted_file"])

        assert result.outputs["formatted_count"] == 2
        assert docs[0]["metadata"]["chunk_strategy"] == "hybrid"
        assert docs[0]["metadata"]["section_path"] == ["Research", "A"]
        assert docs[1]["metadata"]["chunk_index"] == 1
        assert "Section A chunk" in docs[0]["text"]


# ─────────────────────────────────────────────────────────────
# 10. Orchestrator — test stage threading, failure, and resume
# ─────────────────────────────────────────────────────────────

class TestOrchestrator:
    def test_metric_logging_bounds_large_nested_collections(self):
        from pipeline.core.orchestrator import _metrics_for_log

        metrics = {
            "assertion_count": 42,
            "assertions_by_type": {f"type-{index}": index for index in range(100)},
            "small": {"supported": 40, "rejected": 2},
        }

        logged = _metrics_for_log(metrics)

        assert logged["assertion_count"] == 42
        assert logged["small"] == {"supported": 40, "rejected": 2}
        assert logged["assertions_by_type"]["_entry_count"] == 100
        assert len(logged["assertions_by_type"]["_sample"]) == 5
        assert len(metrics["assertions_by_type"]) == 100

    def _make_mock_stages(self):
        """Create mock stages for testing. Returns cleanup function."""
        from pipeline.core.base import PipelineStage, StageContext, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage

        @register_stage
        class MockCrawler(PipelineStage):
            name = "mock_crawler"
            stage_type = "test_crawler"
            description = "Mock crawler"

            async def execute(self, ctx):
                html_dir = ctx.output_dir("html")
                return StageResult.success(
                    outputs={"html_dir": str(html_dir), "data_from_crawler": "hello"},
                    metrics={"pages": 10},
                )

        @register_stage
        class MockProcessor(PipelineStage):
            name = "mock_processor"
            stage_type = "test_processor"
            description = "Mock processor"

            async def execute(self, ctx):
                # Verify we received outputs from previous stage
                assert ctx.previous_outputs.get("data_from_crawler") == "hello"
                return StageResult.success(
                    outputs={"processed": True},
                    metrics={"items": 5},
                )

        def cleanup():
            _REGISTRY.pop("test_crawler", None)
            _REGISTRY.pop("test_processor", None)

        return cleanup

    def test_outputs_thread_between_stages(self, tmp_dir):
        """Stage 2 should receive Stage 1's outputs via previous_outputs."""
        from pipeline.core.orchestrator import PipelineOrchestrator
        cleanup = self._make_mock_stages()
        try:
            config = {
                "project_name": "test",
                "stages": [
                    {"type": "test_crawler", "plugin": "mock_crawler"},
                    {"type": "test_processor", "plugin": "mock_processor"},
                ],
            }
            orch = PipelineOrchestrator(config, work_dir=tmp_dir / "run")
            state = run_async(orch.run())
            assert state.status == "completed"
            assert state.stages[0].metrics == {"pages": 10}
            assert state.stages[1].outputs == {"processed": True}
        finally:
            cleanup()

    def test_failure_stops_pipeline(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageContext, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.orchestrator import PipelineOrchestrator

        call_count = {"s2": 0}

        @register_stage
        class FailStage(PipelineStage):
            name = "fail_s"
            stage_type = "test_fail"
            description = "Fails"
            async def execute(self, ctx):
                return StageResult.failure("intentional failure")

        @register_stage
        class NeverRun(PipelineStage):
            name = "never_s"
            stage_type = "test_never"
            description = "Should never run"
            async def execute(self, ctx):
                call_count["s2"] += 1
                return StageResult.success()

        try:
            config = {
                "project_name": "test",
                "stages": [
                    {"type": "test_fail", "plugin": "fail_s"},
                    {"type": "test_never", "plugin": "never_s"},
                ],
            }
            state = run_async(PipelineOrchestrator(config, work_dir=tmp_dir / "run").run())
            assert state.status == "failed"
            assert state.stages[0].error_message == "intentional failure"
            assert call_count["s2"] == 0  # Second stage should NOT have run
        finally:
            _REGISTRY.pop("test_fail", None)
            _REGISTRY.pop("test_never", None)

    def test_exception_captured_as_failure(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageContext, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.orchestrator import PipelineOrchestrator

        @register_stage
        class BoomStage(PipelineStage):
            name = "boom_s"
            stage_type = "test_boom"
            description = "Throws"
            async def execute(self, ctx):
                raise RuntimeError("Kaboom!")

        try:
            config = {"project_name": "test", "stages": [{"type": "test_boom", "plugin": "boom_s"}]}
            state = run_async(PipelineOrchestrator(config, work_dir=tmp_dir / "run").run())
            assert state.status == "failed"
            assert "Kaboom!" in state.stages[0].error_message
        finally:
            _REGISTRY.pop("test_boom", None)

    def test_resume_skips_completed_stages(self, tmp_dir):
        """On resume, completed stages should NOT re-execute."""
        from pipeline.core.base import PipelineStage, StageContext, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.orchestrator import PipelineOrchestrator
        from pipeline.core.state import PipelineState, StageState, save_state

        call_count = {"s1": 0, "s2": 0}

        @register_stage
        class ResS1(PipelineStage):
            name = "res_s1"
            stage_type = "test_res1"
            description = "Stage 1"
            async def execute(self, ctx):
                call_count["s1"] += 1
                return StageResult.success(outputs={"from_s1": "data"})

        @register_stage
        class ResS2(PipelineStage):
            name = "res_s2"
            stage_type = "test_res2"
            description = "Stage 2"
            async def execute(self, ctx):
                call_count["s2"] += 1
                # Verify we got s1's outputs even though it was skipped
                assert ctx.previous_outputs.get("from_s1") == "data"
                return StageResult.success(outputs={"from_s2": "done"})

        work_dir = tmp_dir / "resume_run"
        work_dir.mkdir()

        # Pre-save state as if s1 already completed
        state = PipelineState(
            run_id="resume_test",
            project_name="test",
            status="running",
            stages=[
                StageState(name="res_s1", stage_type="test_res1", status="completed",
                           outputs={"from_s1": "data"}),
                StageState(name="res_s2", stage_type="test_res2", status="pending"),
            ],
            current_stage_index=1,
        )
        save_state(state, work_dir)

        try:
            config = {
                "project_name": "test",
                "stages": [
                    {"type": "test_res1", "plugin": "res_s1"},
                    {"type": "test_res2", "plugin": "res_s2"},
                ],
            }
            final = run_async(
                PipelineOrchestrator(config, work_dir=work_dir, run_id="resume_test")
                .run(resume=True)
            )
            assert final.status == "completed"
            assert call_count["s1"] == 0  # SKIPPED
            assert call_count["s2"] == 1  # RAN
        finally:
            _REGISTRY.pop("test_res1", None)
            _REGISTRY.pop("test_res2", None)

    def test_resume_clears_stale_stage_error_and_finished_at(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.orchestrator import PipelineOrchestrator
        from pipeline.core.state import PipelineState, StageState, save_state

        @register_stage
        class ResumeCleanStage(PipelineStage):
            name = "resume_clean"
            stage_type = "test_resume_clean"
            description = "Resumed stage"

            async def execute(self, ctx):
                return StageResult.success(outputs={"ok": True})

        work_dir = tmp_dir / "resume_clean_run"
        work_dir.mkdir()

        state = PipelineState(
            run_id="resume_clean_test",
            project_name="test",
            status="failed",
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T01:00:00+00:00",
            stages=[
                StageState(
                    name="resume_clean",
                    stage_type="test_resume_clean",
                    status="failed",
                    stage_id="resume_clean_stage",
                    started_at="2026-01-01T00:10:00+00:00",
                    finished_at="2026-01-01T00:20:00+00:00",
                    error_message="old failure",
                )
            ],
            current_stage_index=0,
        )
        save_state(state, work_dir)

        try:
            config = {
                "project_name": "test",
                "stages": [
                    {"id": "resume_clean_stage", "type": "test_resume_clean", "plugin": "resume_clean"},
                ],
            }
            final = run_async(
                PipelineOrchestrator(config, work_dir=work_dir, run_id="resume_clean_test").run(resume=True)
            )
            assert final.status == "completed"
            assert final.finished_at is not None
            assert final.stages[0].status == "completed"
            assert final.stages[0].error_message is None
            assert final.stages[0].outputs == {"ok": True}
        finally:
            _REGISTRY.pop("test_resume_clean", None)

    def test_resume_extends_state_with_new_tail_stages(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageResult
        from pipeline.core.orchestrator import PipelineOrchestrator
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.state import PipelineState, StageState, save_state

        call_count = {"index": 0}

        @register_stage
        class TailIndexStage(PipelineStage):
            name = "tail_index"
            stage_type = "test_tail_index"
            description = "Tail index stage"

            async def execute(self, ctx):
                call_count["index"] += 1
                return StageResult.success(outputs={"indexed": True})

        work_dir = tmp_dir / "extend_run"
        work_dir.mkdir()
        save_state(
            PipelineState(
                run_id="extend_test",
                project_name="test",
                status="completed",
                stages=[
                    StageState(
                        name="res_s1",
                        stage_type="test_res1",
                        stage_id="stage_one",
                        status="completed",
                        outputs={"from_s1": "data"},
                    )
                ],
                current_stage_index=1,
            ),
            work_dir,
        )

        from pipeline.core.registry import register_stage as _register_stage

        @_register_stage
        class HeadStage(PipelineStage):
            name = "res_s1"
            stage_type = "test_res1"
            description = "Head stage"

            async def execute(self, ctx):
                return StageResult.success(outputs={"from_s1": "data"})

        try:
            config = {
                "project_name": "test",
                "stages": [
                    {"id": "stage_one", "type": "test_res1", "plugin": "res_s1"},
                    {"id": "stage_two", "type": "test_tail_index", "plugin": "tail_index"},
                ],
            }
            final = run_async(
                PipelineOrchestrator(config, work_dir=work_dir, run_id="extend_test").run(resume=True)
            )
            assert final.status == "completed"
            assert len(final.stages) == 2
            assert final.stages[1].status == "completed"
            assert call_count["index"] == 1
        finally:
            _REGISTRY.pop("test_res1", None)
            _REGISTRY.pop("test_tail_index", None)

    def test_restart_from_stage_resets_downstream_outputs_and_artifacts(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageResult
        from pipeline.core.orchestrator import PipelineOrchestrator
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.artifacts import load_artifact_catalog
        from pipeline.core.io import ensure_dir

        call_count = {"s1": 0, "s2": 0}

        @register_stage
        class RestartStageOne(PipelineStage):
            name = "restart_one"
            stage_type = "test_restart_one"
            description = "Stage one"

            async def execute(self, ctx):
                call_count["s1"] += 1
                output = ensure_dir(ctx.stage_work_dir / "files") / "one.txt"
                output.write_text(f"one-{call_count['s1']}", encoding="utf-8")
                return StageResult.success(
                    outputs={"one": str(output)},
                    artifacts=[
                        ctx.make_artifact(
                            output,
                            artifact_type="text_file",
                            role="content",
                        )
                    ],
                )

        @register_stage
        class RestartStageTwo(PipelineStage):
            name = "restart_two"
            stage_type = "test_restart_two"
            description = "Stage two"

            async def execute(self, ctx):
                call_count["s2"] += 1
                output = ensure_dir(ctx.stage_work_dir / "files") / "two.txt"
                output.write_text(f"two-{call_count['s2']}", encoding="utf-8")
                return StageResult.success(
                    outputs={"two": str(output)},
                    artifacts=[
                        ctx.make_artifact(
                            output,
                            artifact_type="text_file",
                            role="content",
                        )
                    ],
                )

        try:
            config = {
                "project_name": "test",
                "stages": [
                    {"id": "stage_one", "type": "test_restart_one", "plugin": "restart_one"},
                    {"id": "stage_two", "type": "test_restart_two", "plugin": "restart_two"},
                ],
            }
            work_dir = tmp_dir / "restart_run"
            orch = PipelineOrchestrator(config, work_dir=work_dir, run_id="restart_test")
            first = run_async(orch.run())
            assert first.status == "completed"

            second = run_async(
                PipelineOrchestrator(config, work_dir=work_dir, run_id="restart_test").run(
                    resume=True,
                    restart_from="stage_one",
                )
            )
            assert second.status == "completed"
            assert call_count == {"s1": 2, "s2": 2}

            catalog = load_artifact_catalog(work_dir)
            assert len(catalog.records) == 2
            assert {record.producer_stage for record in catalog.records} == {"stage_one", "stage_two"}

            stale_root = work_dir / "stage_outputs" / "_stale"
            assert stale_root.exists()
            assert any(path.name.startswith("stage_one_") for path in stale_root.iterdir())
            assert any(path.name.startswith("stage_two_") for path in stale_root.iterdir())
        finally:
            _REGISTRY.pop("test_restart_one", None)
            _REGISTRY.pop("test_restart_two", None)

    def test_active_release_upload_cannot_be_restarted_but_candidate_resume_is_allowed(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageResult
        from pipeline.core.io import atomic_write_json
        from pipeline.core.orchestrator import ActiveReleaseMutationError, PipelineOrchestrator
        from pipeline.core.registry import _REGISTRY, register_stage

        calls = {"upload": 0}

        @register_stage
        class CandidateUploadStage(PipelineStage):
            name = "candidate_upload"
            stage_type = "test_active_upload"

            async def execute(self, ctx):
                calls["upload"] += 1
                return StageResult.success(outputs={"attempt": calls["upload"]})

        try:
            active_pointer = tmp_dir / "active_release.json"
            work_dir = tmp_dir / "candidate-immutable"
            config = {
                "project_name": "test",
                "pipeline": {"active_release_file": str(active_pointer)},
                "stages": [
                    {
                        "id": "upload_retrieval",
                        "type": "test_active_upload",
                        "plugin": "candidate_upload",
                    }
                ],
            }

            first = run_async(
                PipelineOrchestrator(config, work_dir=work_dir, run_id="candidate-immutable").run()
            )
            assert first.status == "completed"

            resumed_candidate = run_async(
                PipelineOrchestrator(config, work_dir=work_dir, run_id="candidate-immutable").run(
                    resume=True,
                    restart_from="upload_retrieval",
                )
            )
            assert resumed_candidate.status == "completed"
            assert calls["upload"] == 2

            atomic_write_json(
                active_pointer,
                {
                    "status": "passed",
                    "run_id": "candidate-immutable",
                    "active_release_manifest": str(work_dir / "release" / "retrieval_release_manifest.json"),
                },
            )
            snapshot_path = work_dir / "resolved_config.json"
            snapshot_before = snapshot_path.read_bytes()
            mutated_config = dict(config)
            mutated_config["forbidden_active_mutation"] = True

            with pytest.raises(ActiveReleaseMutationError, match="active release"):
                run_async(
                    PipelineOrchestrator(mutated_config, work_dir=work_dir, run_id="candidate-immutable").run(
                        resume=True,
                        restart_from="upload_retrieval",
                    )
                )
            with pytest.raises(ActiveReleaseMutationError, match="active release"):
                run_async(
                    PipelineOrchestrator(mutated_config, work_dir=work_dir, run_id="candidate-immutable").run()
                )
            assert calls["upload"] == 2
            assert snapshot_path.read_bytes() == snapshot_before
        finally:
            _REGISTRY.pop("test_active_upload", None)

    def test_run_fails_fast_when_work_dir_is_locked(self, tmp_dir, monkeypatch):
        from pipeline.core.base import PipelineStage, StageResult
        from pipeline.core.orchestrator import PipelineOrchestrator, RunLockError
        from pipeline.core.registry import _REGISTRY, register_stage

        @register_stage
        class LockedStage(PipelineStage):
            name = "locked_stage"
            stage_type = "test_locked"
            description = "Lock test"

            async def execute(self, ctx):
                return StageResult.success()

        try:
            import pipeline.core.orchestrator as orchestrator_module

            def _raise_blocking(_fd, _mode):
                raise BlockingIOError("busy")

            monkeypatch.setattr(orchestrator_module.fcntl, "flock", _raise_blocking)

            config = {
                "project_name": "test",
                "stages": [{"id": "locked_stage_1", "type": "test_locked", "plugin": "locked_stage"}],
            }
            with pytest.raises(RunLockError):
                run_async(PipelineOrchestrator(config, work_dir=tmp_dir / "locked_run").run())
        finally:
            _REGISTRY.pop("test_locked", None)

    def test_skipped_stage_does_not_thread_outputs(self, tmp_dir):
        """A stage returning SKIPPED should NOT propagate its outputs downstream.
        This tests the actual behavior on line 234 of orchestrator.py."""
        from pipeline.core.base import PipelineStage, StageContext, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.orchestrator import PipelineOrchestrator

        received_outputs = {}

        @register_stage
        class SkipStage(PipelineStage):
            name = "skip_s"
            stage_type = "test_skip"
            description = "Skips"
            async def execute(self, ctx):
                return StageResult.skipped("nothing to do")

        @register_stage
        class CheckStage(PipelineStage):
            name = "check_s"
            stage_type = "test_check"
            description = "Checks outputs"
            async def execute(self, ctx):
                received_outputs.update(ctx.previous_outputs)
                return StageResult.success()

        try:
            config = {
                "project_name": "test",
                "stages": [
                    {"type": "test_skip", "plugin": "skip_s"},
                    {"type": "test_check", "plugin": "check_s"},
                ],
            }
            state = run_async(PipelineOrchestrator(config, work_dir=tmp_dir / "run").run())
            assert state.status == "completed"
            # Skipped stage has empty outputs={}, so nothing should be in received
            assert received_outputs == {}
        finally:
            _REGISTRY.pop("test_skip", None)
            _REGISTRY.pop("test_check", None)

    def test_callbacks_are_called(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageContext, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.orchestrator import PipelineOrchestrator

        events = {"starts": [], "completes": []}

        @register_stage
        class CBStage(PipelineStage):
            name = "cb_s"
            stage_type = "test_cb"
            description = "CB"
            async def execute(self, ctx):
                return StageResult.success()

        def on_start(stype, name, info):
            events["starts"].append(name)

        def on_complete(stype, name, info):
            events["completes"].append((name, info["status"]))

        try:
            config = {"project_name": "test", "stages": [{"type": "test_cb", "plugin": "cb_s"}]}
            run_async(
                PipelineOrchestrator(
                    config, work_dir=tmp_dir / "run",
                    on_stage_start=on_start, on_stage_complete=on_complete,
                ).run()
            )
            assert events["starts"] == ["cb_s"]
            assert events["completes"] == [("cb_s", "completed")]
        finally:
            _REGISTRY.pop("test_cb", None)

    def test_empty_stages_raises(self, tmp_dir):
        from pipeline.core.orchestrator import PipelineOrchestrator
        config = {"project_name": "test", "stages": []}
        orch = PipelineOrchestrator(config, work_dir=tmp_dir / "run")
        with pytest.raises(ValueError, match="No stages"):
            run_async(orch.run())

    def test_state_persisted_after_each_stage(self, tmp_dir):
        """Verify pipeline_state.json is written after stage completion."""
        from pipeline.core.base import PipelineStage, StageContext, StageResult
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.orchestrator import PipelineOrchestrator
        from pipeline.core.state import load_state

        @register_stage
        class PersistStage(PipelineStage):
            name = "persist_s"
            stage_type = "test_persist"
            description = "Persist test"
            async def execute(self, ctx):
                return StageResult.success(outputs={"k": "v"}, metrics={"n": 1})

        try:
            work_dir = tmp_dir / "run"
            config = {"project_name": "test", "stages": [{"type": "test_persist", "plugin": "persist_s"}]}
            run_async(PipelineOrchestrator(config, work_dir=work_dir).run())

            state = load_state(work_dir)
            assert state is not None
            assert state.status == "completed"
            assert state.stages[0].outputs == {"k": "v"}
            assert state.stages[0].metrics == {"n": 1}
        finally:
            _REGISTRY.pop("test_persist", None)

    def test_orchestrator_resolves_work_dir_to_absolute_path(self, tmp_dir):
        from pipeline.core.orchestrator import PipelineOrchestrator

        config = {"project_name": "test", "stages": [{"type": "crawler", "plugin": "crawl4ai"}]}
        orch = PipelineOrchestrator(config, work_dir=Path("relative_run_dir"))
        assert orch.work_dir.is_absolute()


# ─────────────────────────────────────────────────────────────
# 11. Docling Converter — test registration and fallback logic
# ─────────────────────────────────────────────────────────────

class TestDoclingConverter:
    def test_registered_with_correct_metadata(self):
        from pipeline.core.registry import auto_discover, get_stage
        auto_discover()
        cls = get_stage("converter", "docling")
        assert cls.name == "docling"
        assert "Granite" in cls.description

    def test_fallback_returns_none_for_nonexistent_file(self, tmp_dir):
        from pipeline.stages.converters.docling_converter import _fallback_extract_text
        assert _fallback_extract_text(tmp_dir / "nonexistent.pdf") is None

    def test_fallback_returns_none_for_non_pdf(self, tmp_dir):
        """Fallback only works for PDFs — non-PDF files return None."""
        from pipeline.stages.converters.docling_converter import _fallback_extract_text
        docx_file = tmp_dir / "test.docx"
        docx_file.write_text("not a real docx")
        assert _fallback_extract_text(docx_file) is None

    def test_execute_skips_when_no_download_dir(self, tmp_dir):
        from pipeline.stages.converters.docling_converter import DoclingConverter
        from pipeline.core.base import StageContext, StageStatus

        ctx = StageContext(
            run_id="test", project_name="test",
            config={"converter": {}},
            work_dir=tmp_dir,
            previous_outputs={},  # no download_dir
        )
        result = run_async(DoclingConverter().execute(ctx))
        assert result.status == StageStatus.SKIPPED

    def test_execute_skips_when_no_supported_files(self, tmp_dir):
        from pipeline.stages.converters.docling_converter import DoclingConverter
        from pipeline.core.base import StageContext, StageStatus

        dl_dir = tmp_dir / "downloads"
        dl_dir.mkdir()
        (dl_dir / "unsupported.xyz").write_text("not a doc")

        ctx = StageContext(
            run_id="test", project_name="test",
            config={"converter": {}},
            work_dir=tmp_dir,
            previous_outputs={"download_dir": str(dl_dir)},
        )
        result = run_async(DoclingConverter().execute(ctx))
        assert result.status == StageStatus.SKIPPED

    def test_inject_image_descriptions_updates_markdown(self):
        from pipeline.stages.converters.docling_converter import _inject_image_descriptions

        markdown = "![Image](../img/a.png)\n\nText\n\n![Image](../img/b.png)"
        images = [
            {"alt": "Campus map", "description": "Campus map overview"},
            {"alt": "Logo", "description": "University logo"},
        ]

        rendered = _inject_image_descriptions(markdown, images)

        assert "![Campus map](../img/a.png)" in rendered
        assert "Image description: Campus map overview" in rendered
        assert "![Logo](../img/b.png)" in rendered
        assert "Image description: University logo" in rendered

    def test_inject_image_descriptions_leaves_tail_content(self):
        from pipeline.stages.converters.docling_converter import _inject_image_descriptions

        markdown = "Intro\n\n![Image](../img/a.png)\n\nConclusion"
        images = [{"alt": "Figure 1", "description": ""}]

        rendered = _inject_image_descriptions(markdown, images)

        assert rendered.endswith("Conclusion")

    def test_should_describe_with_vlm_skips_rich_existing_caption(self):
        from pipeline.stages.converters.docling_converter import _should_describe_with_vlm

        image = {
            "local_path": "/tmp/figure.png",
            "caption": (
                "Figure 2: Architecture overview showing encoder, retrieval pipeline, "
                "ranking stage, and grounded answer synthesis for production search."
            ),
            "alt": "Figure 2",
        }

        assert not _should_describe_with_vlm(
            image,
            {"vlm_strategy": "missing_only", "vlm_skip_caption_min_words": 10},
        )

    def test_build_granite_prompt_includes_document_context(self):
        from pipeline.stages.converters.docling_converter import _build_granite_prompt

        prompt = _build_granite_prompt(
            {
                "source_file": "/tmp/MBZUAI_Campus_Map.pdf",
                "caption": "",
                "context": "MASDAR CITY CENTRAL PARK",
                "page_number": 1,
            },
            {"vlm_max_description_words": 20},
        )

        assert "Document title: MBZUAI Campus Map" in prompt
        assert "Detected figure labels/context: MASDAR CITY CENTRAL PARK" in prompt
        assert "Keep the answer under 20 words." in prompt

    def test_sanitize_granite_description_trims_boilerplate(self):
        from pipeline.stages.converters.docling_converter import _sanitize_granite_description

        description = _sanitize_granite_description(
            (
                "The image shows a campus map with labeled buildings and roads. "
                "The image is intended for illustrative purposes only and does not contain "
                "any personal information or sensitive data."
            ),
            12,
        )

        assert description == "The image shows a campus map with labeled buildings and roads."

    def test_annotate_images_with_granite_stops_when_runtime_unavailable(self, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        calls = {"count": 0}

        def fake_generate(image, config):
            calls["count"] += 1
            raise module.GraniteUnavailableError("offline cache not available")

        monkeypatch.setattr(module, "_generate_granite_description", fake_generate)

        images = [
            {"local_path": "/tmp/a.png", "alt": "Figure 1"},
            {"local_path": "/tmp/b.png", "alt": "Figure 2"},
        ]

        described = module._annotate_images_with_granite(
            images,
            {"vlm_runtime": "transformers", "vlm_batch_size": 1},
        )

        assert described == 0
        assert calls["count"] == 1

    def test_convert_with_docling_skips_vlm_without_accelerator(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        class FakeExportDoc:
            def __init__(self):
                self.save_calls = []

            def export_to_markdown(self, image_mode=None):
                return "![Image](../extracted_images/test/image.png)\n\nText"

            def save_as_json(self, filename, artifacts_dir=None, image_mode=None, indent=2):
                self.save_calls.append({
                    "filename": str(filename),
                    "artifacts_dir": str(artifacts_dir) if artifacts_dir else None,
                    "image_mode": image_mode,
                    "indent": indent,
                })
                Path(filename).write_text('{"ok": true}', encoding="utf-8")

        class FakeDoc(FakeExportDoc):
            def _make_copy_with_refmode(self, image_artifacts_dir, image_mode, page_no=None, reference_path=None):
                return FakeExportDoc()

        class FakeConversion:
            def __init__(self):
                self.document = FakeDoc()

        class FakeDocumentConverter:
            def __init__(self, *args, **kwargs):
                pass

            def convert(self, *args, **kwargs):
                return FakeConversion()

        monkeypatch.setattr(module, "_extract_picture_metadata", lambda *args, **kwargs: [
            {
                "local_path": str(tmp_dir / "img.png"),
                "alt": "Image",
                "caption": "",
                "description": "",
            }
        ])
        monkeypatch.setattr(module, "_has_supported_vlm_accelerator", lambda: False)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Granite captioning should be skipped on CPU-only hosts")

        monkeypatch.setattr(module, "_annotate_images_with_granite", fail_if_called)

        import docling.document_converter as document_converter_mod

        monkeypatch.setattr(document_converter_mod, "DocumentConverter", FakeDocumentConverter)

        sample_pdf = tmp_dir / "sample.pdf"
        sample_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
        result = module._convert_with_docling(
            sample_pdf,
            tmp_dir / "out.md",
            tmp_dir / "images",
            tmp_dir / "out.docling.json",
            {"use_vlm": True, "vlm_require_accelerator": True, "generate_picture_images": True},
        )

        assert result is not None
        assert result["vlm_described"] == 0
        assert Path(result["md_path"]).exists()

    def test_convert_with_docling_saves_referenced_json_without_duplicate_artifact_export(
        self,
        tmp_dir,
        monkeypatch,
    ):
        from pipeline.stages.converters import docling_converter as module

        class FakeExportDoc:
            def __init__(self):
                self.save_calls = []

            def export_to_markdown(self, image_mode=None):
                return "![Image](../extracted_images/test/image.png)\n\nText"

            def save_as_json(self, filename, artifacts_dir=None, image_mode=None, indent=2):
                self.save_calls.append({
                    "filename": str(filename),
                    "artifacts_dir": str(artifacts_dir) if artifacts_dir else None,
                    "image_mode": image_mode,
                    "indent": indent,
                })
                Path(filename).write_text('{"ok": true}', encoding="utf-8")

        class FakeDoc:
            def __init__(self):
                self.export_doc = FakeExportDoc()
                self.save_calls = []

            def _make_copy_with_refmode(self, image_artifacts_dir, image_mode, page_no=None, reference_path=None):
                return self.export_doc

            def export_to_markdown(self, image_mode=None):
                return "fallback"

            def save_as_json(self, filename, artifacts_dir=None, image_mode=None, indent=2):
                self.save_calls.append({
                    "filename": str(filename),
                    "artifacts_dir": str(artifacts_dir) if artifacts_dir else None,
                    "image_mode": image_mode,
                    "indent": indent,
                })
                Path(filename).write_text('{"root": true}', encoding="utf-8")

        fake_doc = FakeDoc()

        class FakeConversion:
            def __init__(self, document):
                self.document = document

        class FakeDocumentConverter:
            def __init__(self, *args, **kwargs):
                pass

            def convert(self, *args, **kwargs):
                return FakeConversion(fake_doc)

        monkeypatch.setattr(module, "_extract_picture_metadata", lambda *args, **kwargs: [])

        import docling.document_converter as document_converter_mod

        monkeypatch.setattr(document_converter_mod, "DocumentConverter", FakeDocumentConverter)

        sample_pdf = tmp_dir / "sample.pdf"
        sample_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
        result = module._convert_with_docling(
            sample_pdf,
            tmp_dir / "out.md",
            tmp_dir / "images",
            tmp_dir / "out.docling.json",
            {"use_vlm": False, "generate_picture_images": True},
        )

        assert result is not None
        assert fake_doc.save_calls == []
        assert len(fake_doc.export_doc.save_calls) == 1
        assert fake_doc.export_doc.save_calls[0]["artifacts_dir"] is None
        assert str(fake_doc.export_doc.save_calls[0]["image_mode"]).endswith("PLACEHOLDER")

    def test_docling_perf_overrides_restore_settings(self):
        from docling.datamodel.settings import settings as docling_settings
        from pipeline.stages.converters.docling_converter import _docling_perf_overrides

        original = (
            docling_settings.perf.doc_batch_size,
            docling_settings.perf.doc_batch_concurrency,
            docling_settings.perf.page_batch_size,
            docling_settings.perf.page_batch_concurrency,
            docling_settings.perf.elements_batch_size,
        )

        with _docling_perf_overrides(
            {
                "docling_doc_batch_size": 7,
                "docling_doc_batch_concurrency": 3,
                "docling_page_batch_size": 9,
                "docling_page_batch_concurrency": 2,
                "docling_elements_batch_size": 21,
            }
        ):
            assert docling_settings.perf.doc_batch_size == 7
            assert docling_settings.perf.doc_batch_concurrency == 3
            assert docling_settings.perf.page_batch_size == 9
            assert docling_settings.perf.page_batch_concurrency == 2
            assert docling_settings.perf.elements_batch_size == 21

        assert (
            docling_settings.perf.doc_batch_size,
            docling_settings.perf.doc_batch_concurrency,
            docling_settings.perf.page_batch_size,
            docling_settings.perf.page_batch_concurrency,
            docling_settings.perf.elements_batch_size,
        ) == original

    def test_annotate_images_with_granite_batches_requests(self, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        images = [
            {"local_path": "/tmp/one.png", "alt": "Figure 1"},
            {"local_path": "/tmp/two.png", "alt": "Figure 2"},
        ]
        batch_calls = []

        monkeypatch.setattr(module, "_prepare_vlm_image", lambda path, max_side, max_pixels: (path, None))

        def fake_batch(requests, model_id, max_new_tokens, *, local_files_only, use_cache):
            batch_calls.append([str(path) for path, _prompt in requests])
            return ["chart of results", "system architecture diagram"]

        monkeypatch.setattr(module, "_run_granite_generation_batch", fake_batch)

        described = module._annotate_images_with_granite(
            images,
            {
                "vlm_runtime": "transformers",
                "vlm_batch_size": 2,
                "max_vlm_images_per_doc": 2,
                "vlm_model": "fake-model",
            },
        )

        assert described == 2
        assert batch_calls == [["/tmp/one.png", "/tmp/two.png"]]
        assert images[0]["description"] == "chart of results."
        assert images[1]["description"] == "system architecture diagram."

    def test_annotate_images_with_granite_falls_back_to_single_on_batch_failure(self, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        images = [
            {"local_path": "/tmp/one.png", "alt": "Figure 1"},
            {"local_path": "/tmp/two.png", "alt": "Figure 2"},
        ]
        single_calls = []

        monkeypatch.setattr(module, "_prepare_vlm_image", lambda path, max_side, max_pixels: (path, None))
        monkeypatch.setattr(module, "_run_granite_generation_batch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("batch failed")))

        def fake_single(image, config):
            single_calls.append(image["local_path"])
            return f"description for {Path(image['local_path']).stem}"

        monkeypatch.setattr(module, "_generate_granite_description", fake_single)

        described = module._annotate_images_with_granite(
            images,
            {
                "vlm_runtime": "transformers",
                "vlm_batch_size": 2,
                "max_vlm_images_per_doc": 2,
                "vlm_model": "fake-model",
            },
        )

        assert described == 2
        assert single_calls == ["/tmp/one.png", "/tmp/two.png"]
        assert images[0]["description"] == "description for one"
        assert images[1]["description"] == "description for two"

    def test_execute_uses_batch_pdf_conversion_for_multiple_pdfs(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.stages.converters import docling_converter as module

        download_dir = tmp_dir / "downloads"
        download_dir.mkdir()
        pdf_paths = []
        for name in ("one.pdf", "two.pdf"):
            path = download_dir / name
            path.write_bytes(b"%PDF-1.4\n%fake\n")
            pdf_paths.append(path)

        batch_calls = {"count": 0}

        def fake_build(config):
            return object(), object()

        def fake_batch(files, output_targets, config, *, converter=None, pdf_options=None):
            batch_calls["count"] += 1
            results = {}
            for file_path in files:
                target = output_targets[str(file_path.resolve())]
                target["md_path"].parent.mkdir(parents=True, exist_ok=True)
                target["structured_doc_path"].parent.mkdir(parents=True, exist_ok=True)
                target["md_path"].write_text(
                    f"# {file_path.stem}\n\n" + ("Content " * 40),
                    encoding="utf-8",
                )
                target["structured_doc_path"].write_text('{"doc": true}', encoding="utf-8")
                results[str(file_path.resolve())] = {
                    "md_path": str(target["md_path"]),
                    "images": [],
                    "source_file": str(file_path),
                    "engine": "docling",
                    "vlm_described": 0,
                    "structured_document_path": str(target["structured_doc_path"]),
                }
            return results

        def fail_single(*args, **kwargs):
            raise AssertionError("single-file Docling conversion should not be used for multi-PDF batch")

        monkeypatch.setattr(module, "_build_docling_pdf_converter", fake_build)
        monkeypatch.setattr(module, "_convert_pdf_batch_with_docling", fake_batch)
        monkeypatch.setattr(module, "_convert_with_docling", fail_single)

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"converter": {}},
            work_dir=tmp_dir,
            previous_outputs={"download_dir": str(download_dir)},
            stage_definition={"type": "converter", "plugin": "docling"},
            stage_id="convert_documents",
        )

        result = run_async(module.DoclingConverter().execute(ctx))

        assert result.metrics["converted"] == 2
        assert result.metrics["docling_converted"] == 2
        assert batch_calls["count"] == 1

    def test_convert_pdf_batch_matches_results_by_filename_when_docling_rewrites_input_path(
        self,
        tmp_dir,
        monkeypatch,
    ):
        from types import SimpleNamespace

        from docling.datamodel.base_models import ConversionStatus
        from pipeline.stages.converters import docling_converter as module

        pdf_path = tmp_dir / "downloads" / "sample_123.pdf"
        pdf_path.parent.mkdir(parents=True)
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

        class FakeExportDoc:
            def export_to_markdown(self, image_mode=None):
                return "![Image](../extracted_images/sample_123/image.png)\n\nText"

            def save_as_json(self, filename, artifacts_dir=None, image_mode=None, indent=2):
                Path(filename).write_text('{"ok": true}', encoding="utf-8")

        class FakeDoc:
            def _make_copy_with_refmode(self, image_artifacts_dir, image_mode, page_no=None, reference_path=None):
                return FakeExportDoc()

            def export_to_markdown(self, image_mode=None):
                return "Text"

        class FakeConversion:
            def __init__(self):
                self.input = SimpleNamespace(file=pdf_path.name)
                self.status = ConversionStatus.SUCCESS
                self.document = FakeDoc()
                self.errors = []

        class FakeConverter:
            def convert_all(self, source, raises_on_error=False):
                assert list(source) == [str(pdf_path)]
                yield FakeConversion()

        monkeypatch.setattr(module, "_extract_picture_metadata", lambda *args, **kwargs: [])

        md_path = tmp_dir / "out" / "sample_123.md"
        image_dir = tmp_dir / "out" / "images" / "sample_123"
        structured_doc_path = tmp_dir / "out" / "structured" / "sample_123.docling.json"
        output_targets = {
            str(pdf_path.resolve()): {
                "source_key": str(pdf_path.resolve()),
                "source_name": pdf_path.name,
                "md_path": md_path,
                "image_dir": image_dir,
                "structured_doc_path": structured_doc_path,
            }
        }

        results = module._convert_pdf_batch_with_docling(
            [pdf_path],
            output_targets,
            {"use_vlm": False, "generate_picture_images": True},
            converter=FakeConverter(),
            pdf_options=SimpleNamespace(generate_picture_images=True),
        )

        assert results is not None
        assert results[str(pdf_path.resolve())] is not None
        assert md_path.exists()
        assert structured_doc_path.exists()

    def test_convert_pdf_batch_submits_files_in_chunks(self, tmp_dir, monkeypatch):
        from types import SimpleNamespace

        from docling.datamodel.base_models import ConversionStatus
        from pipeline.stages.converters import docling_converter as module

        pdf_paths = []
        output_targets = {}
        for name in ("one.pdf", "two.pdf", "three.pdf"):
            pdf_path = tmp_dir / "downloads" / name
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
            pdf_paths.append(pdf_path)
            output_targets[str(pdf_path.resolve())] = {
                "source_key": str(pdf_path.resolve()),
                "source_name": pdf_path.name,
                "md_path": tmp_dir / "out" / f"{pdf_path.stem}.md",
                "image_dir": tmp_dir / "out" / "images" / pdf_path.stem,
                "structured_doc_path": tmp_dir / "out" / "structured" / f"{pdf_path.stem}.docling.json",
            }

        class FakeDoc:
            def export_to_markdown(self, image_mode=None):
                return "Text"

            def save_as_json(self, filename, artifacts_dir=None, image_mode=None, indent=2):
                Path(filename).write_text('{"ok": true}', encoding="utf-8")

        class FakeConversion:
            def __init__(self, path: Path):
                self.input = SimpleNamespace(file=str(path))
                self.status = ConversionStatus.SUCCESS
                self.document = FakeDoc()
                self.errors = []

        calls = []

        class FakeConverter:
            def convert_all(self, source, raises_on_error=False):
                materialized = list(source)
                calls.append(materialized)
                for item in materialized:
                    yield FakeConversion(Path(item))

        monkeypatch.setattr(module, "_extract_picture_metadata", lambda *args, **kwargs: [])

        results = module._convert_pdf_batch_with_docling(
            pdf_paths,
            output_targets,
            {
                "use_vlm": False,
                "generate_picture_images": True,
                "docling_submit_batch_size": 2,
            },
            converter=FakeConverter(),
            pdf_options=SimpleNamespace(generate_picture_images=True),
        )

        assert results is not None
        assert len(calls) == 2
        assert calls[0] == [str(pdf_paths[0]), str(pdf_paths[1])]
        assert calls[1] == [str(pdf_paths[2])]

    def test_large_pdf_defaults_to_docling_page_windows(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        pdf_path = tmp_dir / "catalogue.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
        monkeypatch.setattr(module, "_pdf_page_count", lambda path: 120)

        assert module._docling_pdf_conversion_mode(pdf_path, {"docling_max_pages": 80}) == (
            "page_windows",
            "page_count>80 (120)",
        )
        assert module._docling_pdf_conversion_mode(
            pdf_path,
            {"docling_max_pages": 80, "docling_large_pdf_strategy": "fallback"},
        ) == ("fallback", "page_count>80 (120)")

    def test_convert_large_pdf_with_docling_windows_preserves_docling_extraction(self, tmp_dir, monkeypatch):
        from docling.datamodel.base_models import ConversionStatus
        from pipeline.stages.converters import docling_converter as module

        pdf_path = tmp_dir / "catalogue.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
        monkeypatch.setattr(module, "_pdf_page_count", lambda path: 3)

        class FakeDoc:
            def __init__(self, text):
                self.text = text

            def export_to_markdown(self, image_mode=None):
                return self.text

            def save_as_json(self, filename, artifacts_dir=None, image_mode=None, indent=2):
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
                Path(filename).write_text(json.dumps({"text": self.text}), encoding="utf-8")

        class FakeConversion:
            def __init__(self, text):
                self.status = ConversionStatus.SUCCESS
                self.document = FakeDoc(text)
                self.errors = []

        calls = []

        class FakeConverter:
            def convert(self, source, raises_on_error=False, page_range=(1, 999999)):
                calls.append(page_range)
                return FakeConversion(f"Docling pages {page_range[0]}-{page_range[1]}")

        result = module._convert_large_pdf_with_docling_windows(
            pdf_path,
            tmp_dir / "out" / "catalogue.md",
            tmp_dir / "out" / "images" / "catalogue",
            tmp_dir / "out" / "structured" / "catalogue.docling.json",
            {
                "use_vlm": False,
                "generate_picture_images": False,
                "docling_large_pdf_window_pages": 2,
            },
            converter=FakeConverter(),
            pdf_options=SimpleNamespace(generate_picture_images=False),
        )

        assert result is not None
        assert result["engine"] == "docling_page_windows"
        assert calls == [(1, 2), (3, 3)]
        merged = Path(result["md_path"]).read_text(encoding="utf-8")
        assert "## Pages 1-2" in merged
        assert "Docling pages 1-2" in merged
        assert "## Pages 3-3" in merged
        structured = json.loads(Path(result["structured_document_path"]).read_text(encoding="utf-8"))
        assert structured["schema"] == "mbzuai_docling_page_windows.v1"
        assert len(structured["parts"]) == 2

    def test_rewrite_structured_doc_image_refs_removes_sidecar_artifacts(self, tmp_dir):
        from pipeline.stages.converters.docling_converter import _rewrite_structured_doc_image_refs

        structured_doc = tmp_dir / "structured_documents" / "sample.docling.json"
        structured_doc.parent.mkdir(parents=True)
        image_dir = tmp_dir / "extracted_images" / "sample"
        image_dir.mkdir(parents=True)
        canonical = image_dir / "image_000000_test.png"
        canonical.write_bytes(b"png")

        relative_structured_doc = Path("runs/test/stage_outputs/convert_documents/structured_documents/sample.docling.json")
        nested_sidecar = structured_doc.parent / relative_structured_doc.with_suffix("").with_suffix(".docling_artifacts")
        nested_sidecar.mkdir(parents=True)
        (nested_sidecar / "image_000000_test.png").write_bytes(b"dup")
        nested_extracted = structured_doc.parent / Path("runs/test/stage_outputs/convert_documents/extracted_images/sample")
        nested_extracted.mkdir(parents=True)
        (nested_extracted / "image_000000_test.png").write_bytes(b"dup2")

        structured_doc.write_text(
            json.dumps(
                {
                    "pictures": [
                        {
                            "image": {
                                "uri": str(Path("runs/test/stage_outputs/convert_documents/extracted_images/sample") / "image_000000_test.png"),
                            }
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        _rewrite_structured_doc_image_refs(structured_doc, image_dir)

        rewritten = json.loads(structured_doc.read_text(encoding="utf-8"))
        assert rewritten["pictures"][0]["image"]["uri"] == "../extracted_images/sample/image_000000_test.png"
        assert not nested_sidecar.exists()
        assert not nested_extracted.exists()

    def test_execute_prefers_fallback_when_docling_output_is_not_fit_for_indexing(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.stages.converters import docling_converter as module

        download_dir = tmp_dir / "downloads"
        download_dir.mkdir()
        pdf_path = download_dir / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

        def fake_convert(file_path, md_path, image_dir, structured_doc_path, config, **kwargs):
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(
                "# Paper\n\nThisMlCCAlpaperistheOpenAccessversionprovidedbytheSociety.",
                encoding="utf-8",
            )
            structured_doc_path.parent.mkdir(parents=True, exist_ok=True)
            structured_doc_path.write_text('{"doc": true}', encoding="utf-8")
            return {
                "md_path": str(md_path),
                "images": [],
                "source_file": str(file_path),
                "engine": "docling",
                "vlm_described": 0,
                "structured_document_path": str(structured_doc_path),
            }

        monkeypatch.setattr(module, "_convert_with_docling", fake_convert)
        monkeypatch.setattr(module, "_convert_pdf_batch_with_docling", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            module,
            "_fallback_extract_text",
            lambda fp: "# Paper\n\n## Page 1\n\nThis paper is the open access version provided by the society.",
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"converter": {"validation_min_informative_words": 10, "validation_min_text_chars": 40}},
            work_dir=tmp_dir,
            previous_outputs={"download_dir": str(download_dir)},
            stage_definition={"type": "converter", "plugin": "docling"},
            stage_id="convert_documents",
        )

        result = run_async(module.DoclingConverter().execute(ctx))

        assert result.metrics["converted"] == 1
        assert result.metrics["fallback_converted"] == 1
        assert result.metrics["fallback_selected"] == 1
        rendered = (Path(result.outputs["md_dir"]) / "paper.md").read_text(encoding="utf-8")
        assert "open access version provided by the society" in rendered
        assert not (Path(result.outputs["structured_documents_dir"]) / "paper.docling.json").exists()
        assert (Path(result.outputs["quarantine_dir"]) / "structured_documents" / "unused_after_fallback" / "paper.docling.json").exists()

    def test_execute_quarantines_document_when_all_candidates_fail_quality(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.stages.converters import docling_converter as module

        download_dir = tmp_dir / "downloads"
        download_dir.mkdir()
        pdf_path = download_dir / "bad.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

        def fake_convert(file_path, md_path, image_dir, structured_doc_path, config, **kwargs):
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text("# Bad\n\nShort", encoding="utf-8")
            return {
                "md_path": str(md_path),
                "images": [],
                "source_file": str(file_path),
                "engine": "docling",
                "vlm_described": 0,
                "structured_document_path": "",
            }

        monkeypatch.setattr(module, "_convert_with_docling", fake_convert)
        monkeypatch.setattr(module, "_convert_pdf_batch_with_docling", lambda *args, **kwargs: None)
        monkeypatch.setattr(module, "_fallback_extract_text", lambda fp: "# Bad\n\nShort")

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"converter": {}},
            work_dir=tmp_dir,
            previous_outputs={"download_dir": str(download_dir)},
            stage_definition={"type": "converter", "plugin": "docling"},
            stage_id="convert_documents",
        )

        result = run_async(module.DoclingConverter().execute(ctx))

        assert result.metrics["converted"] == 0
        assert result.metrics["failed"] == 1
        assert result.metrics["validation_failed"] == 1
        assert not (Path(result.outputs["md_dir"]) / "bad.md").exists()
        assert (Path(result.outputs["quarantine_dir"]) / "markdown" / "bad.md").exists()

    def test_execute_reuses_shared_docling_converter_for_multiple_pdfs(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.stages.converters import docling_converter as module

        download_dir = tmp_dir / "downloads"
        download_dir.mkdir()
        for name in ("one.pdf", "two.pdf"):
            (download_dir / name).write_bytes(b"%PDF-1.4\n%fake\n")

        build_calls = {"count": 0}
        shared_converter = object()
        shared_pdf_options = object()

        def fake_build(config):
            build_calls["count"] += 1
            return shared_converter, shared_pdf_options

        def fake_convert(file_path, md_path, image_dir, structured_doc_path, config, *, converter=None, pdf_options=None):
            assert converter is shared_converter
            assert pdf_options is shared_pdf_options
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(
                f"# {file_path.stem}\n\nThis document has enough content for indexing.\n\n"
                + ("Content " * 40),
                encoding="utf-8",
            )
            structured_doc_path.parent.mkdir(parents=True, exist_ok=True)
            structured_doc_path.write_text('{"doc": true}', encoding="utf-8")
            return {
                "md_path": str(md_path),
                "images": [],
                "source_file": str(file_path),
                "engine": "docling",
                "vlm_described": 0,
                "structured_document_path": str(structured_doc_path),
            }

        monkeypatch.setattr(module, "_build_docling_pdf_converter", fake_build)
        monkeypatch.setattr(module, "_convert_with_docling", fake_convert)

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"converter": {}},
            work_dir=tmp_dir,
            previous_outputs={"download_dir": str(download_dir)},
            stage_definition={"type": "converter", "plugin": "docling"},
            stage_id="convert_documents",
        )

        result = run_async(module.DoclingConverter().execute(ctx))

        assert result.metrics["converted"] == 2
        assert build_calls["count"] == 1

    def test_ensure_docling_artifacts_prefetches_local_cache(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        calls = []
        cache_dir = tmp_dir / ".cache" / "docling"

        def fake_download_models(output_dir, **kwargs):
            calls.append((Path(output_dir), kwargs))
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return Path(output_dir)

        monkeypatch.setitem(
            sys.modules,
            "docling.datamodel.settings",
            SimpleNamespace(settings=SimpleNamespace(cache_dir=cache_dir)),
        )
        monkeypatch.setitem(
            sys.modules,
            "docling.utils.model_downloader",
            SimpleNamespace(download_models=fake_download_models),
        )

        artifacts_path = module._ensure_docling_artifacts(
            {
                "do_table_structure": True,
                "do_ocr": True,
                "do_code_enrichment": False,
                "do_formula_enrichment": False,
                "do_picture_classification": False,
            }
        )

        assert artifacts_path == cache_dir / "models"
        assert calls
        output_dir, kwargs = calls[0]
        assert output_dir == cache_dir / "models"
        assert kwargs["with_layout"] is True
        assert kwargs["with_tableformer"] is True
        assert kwargs["with_easyocr"] is False
        assert kwargs["with_granite_vision"] is False

    def test_ensure_docling_artifacts_prefetches_easyocr_only_when_requested(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        calls = []
        cache_dir = tmp_dir / ".cache" / "docling"

        def fake_download_models(output_dir, **kwargs):
            calls.append(kwargs)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return Path(output_dir)

        monkeypatch.setitem(
            sys.modules,
            "docling.datamodel.settings",
            SimpleNamespace(settings=SimpleNamespace(cache_dir=cache_dir)),
        )
        monkeypatch.setitem(
            sys.modules,
            "docling.utils.model_downloader",
            SimpleNamespace(download_models=fake_download_models),
        )

        module._ensure_docling_artifacts(
            {
                "docling_ocr_kind": "easyocr",
                "do_ocr": True,
            }
        )

        assert calls
        assert calls[0]["with_easyocr"] is True

    def test_ensure_docling_artifacts_raises_when_local_artifacts_missing_and_prefetch_disabled(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        cache_dir = tmp_dir / ".cache" / "docling"
        monkeypatch.setitem(
            sys.modules,
            "docling.datamodel.settings",
            SimpleNamespace(settings=SimpleNamespace(cache_dir=cache_dir)),
        )
        monkeypatch.setitem(
            sys.modules,
            "docling.utils.model_downloader",
            SimpleNamespace(download_models=lambda **kwargs: None),
        )

        with pytest.raises(RuntimeError, match="Docling artifacts path does not exist"):
            module._ensure_docling_artifacts(
                {
                    "docling_prefetch_models": False,
                    "docling_artifacts_path": str(tmp_dir / "missing_models"),
                }
            )

    def test_build_docling_pdf_converter_sets_local_artifacts_and_disables_runtime_downloads(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        artifacts_path = tmp_dir / "docling_models"
        artifacts_path.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(module, "_ensure_docling_artifacts", lambda config: artifacts_path)

        class FakePdfPipelineOptions:
            def __init__(self, do_ocr, do_table_structure, force_backend_text):
                self.do_ocr = do_ocr
                self.do_table_structure = do_table_structure
                self.force_backend_text = force_backend_text
                self.images_scale = 1.0
                self.generate_picture_images = False
                self.generate_page_images = False
                self.document_timeout = None
                self.artifacts_path = None
                self.enable_remote_services = True
                self.ocr_options = SimpleNamespace(download_enabled=True)

        class FakePdfFormatOption:
            def __init__(self, pipeline_options):
                self.pipeline_options = pipeline_options

        class FakeDocumentConverter:
            def __init__(self, *, format_options):
                self.format_options = format_options

        fake_input_format = SimpleNamespace(PDF="pdf")

        monkeypatch.setitem(
            sys.modules,
            "docling.datamodel.base_models",
            SimpleNamespace(InputFormat=fake_input_format),
        )
        monkeypatch.setitem(
            sys.modules,
            "docling.datamodel.pipeline_options",
            SimpleNamespace(PdfPipelineOptions=FakePdfPipelineOptions),
        )
        monkeypatch.setitem(
            sys.modules,
            "docling.document_converter",
            SimpleNamespace(DocumentConverter=FakeDocumentConverter, PdfFormatOption=FakePdfFormatOption),
        )

        converter, pdf_options = module._build_docling_pdf_converter(
            {
                "do_ocr": True,
                "do_table_structure": True,
                "force_backend_text": True,
                "generate_picture_images": True,
            }
        )

        assert isinstance(converter, FakeDocumentConverter)
        assert pdf_options.artifacts_path == str(artifacts_path)
        assert pdf_options.enable_remote_services is False
        assert pdf_options.ocr_options.download_enabled is False

    def test_generate_granite_description_retries_multiple_downscales_after_oom(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        image_path = tmp_dir / "image.png"
        image_path.write_bytes(b"png")
        attempts = []

        monkeypatch.setattr(module, "_prepare_vlm_image", lambda path, side, pixels: (path, None))
        monkeypatch.setattr(module, "_clear_cuda_cache", lambda: None)

        def fake_run(*args, **kwargs):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("CUDA out of memory")
            return "A campus map is shown."

        monkeypatch.setattr(module, "_run_granite_generation", fake_run)

        description = module._generate_granite_description(
            {"local_path": str(image_path)},
            {
                "vlm_retry_attempts": 3,
                "vlm_retry_downscale_factor": 0.6,
            },
        )

        assert description == "A campus map is shown."
        assert len(attempts) == 3

    def test_annotate_images_with_granite_reduces_batch_size_when_cuda_memory_is_low(self, tmp_dir, monkeypatch):
        from pipeline.stages.converters import docling_converter as module

        image_one = tmp_dir / "one.png"
        image_two = tmp_dir / "two.png"
        image_one.write_bytes(b"png")
        image_two.write_bytes(b"png")

        monkeypatch.setattr(module, "_cuda_free_mb", lambda: 1024)
        monkeypatch.setattr(module, "_clear_cuda_cache", lambda: None)

        batch_calls = {"count": 0}

        def fake_batch(*args, **kwargs):
            batch_calls["count"] += 1
            return ["unused"]

        monkeypatch.setattr(module, "_run_granite_generation_batch", fake_batch)
        monkeypatch.setattr(module, "_generate_granite_description", lambda image, config: "A document figure.")

        images = [
            {"local_path": str(image_one)},
            {"local_path": str(image_two)},
        ]
        described = module._annotate_images_with_granite(
            images,
            {
                "vlm_batch_size": 2,
                "vlm_min_free_cuda_mb_for_batch": 4096,
                "vlm_use_cache": False,
            },
        )

        assert described == 2
        assert batch_calls["count"] == 0
        assert images[0]["description"] == "A document figure."
        assert images[1]["description"] == "A document figure."


class TestDocumentQuality:
    def test_assess_markdown_document_rejects_collapsed_spacing(self):
        from pipeline.core.document_quality import assess_markdown_document

        assessment = assess_markdown_document(
            "# Paper\n\nThisMlCCAlpaperistheOpenAccessversionprovidedbytheSociety.",
            source_ext=".pdf",
            config={},
            media_items=[],
        )

        assert not assessment["accepted"]
        assert "collapsed_spacing" in assessment["reasons"]

    def test_assess_markdown_document_allows_media_heavy_low_text(self):
        from pipeline.core.document_quality import assess_markdown_document

        assessment = assess_markdown_document(
            "# Map\n\nCampus map.",
            source_ext=".pdf",
            config={},
            media_items=[
                {
                    "description": "Campus map showing academic buildings, student housing, roads, and labeled entrances.",
                    "caption": "",
                    "context": "",
                    "alt": "Campus map",
                }
            ],
        )

        assert assessment["accepted"]
        assert "media_heavy_low_text" in assessment["warnings"]


# ─────────────────────────────────────────────────────────────
# 12. CLI Commands — test actual output and error conditions
# ─────────────────────────────────────────────────────────────

class TestCLI:
    def test_list_stages_returns_0(self):
        import argparse
        from pipeline.cli import cmd_list_stages
        assert cmd_list_stages(argparse.Namespace(type=None)) == 0

    def test_list_stages_with_type_filter(self):
        import argparse
        from pipeline.cli import cmd_list_stages
        # Should succeed even with a specific type filter
        assert cmd_list_stages(argparse.Namespace(type="crawler")) == 0

    def test_dry_run_shows_correct_stage_count(self, capsys):
        import argparse
        from pipeline.cli import cmd_dry_run
        result = cmd_dry_run(argparse.Namespace(config="mbzuai_main"))
        assert result == 0
        output = capsys.readouterr().out
        # Should show all 9 stages from default.yaml
        assert "crawl4ai" in output
        assert "docling" in output

    def test_validate_with_valid_config(self):
        import argparse
        from pipeline.cli import cmd_validate
        result = cmd_validate(argparse.Namespace(config="default"))
        # May return 0 or 1 depending on installed libraries — just verify it doesn't crash
        assert result in (0, 1)


# ─────────────────────────────────────────────────────────────
# 12b. Chunkers — verify pluggable chunk strategies
# ─────────────────────────────────────────────────────────────

class TestChunkers:
    def test_hierarchical_chunker_emits_section_metadata(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import load_chunk_index
        from pipeline.stages.chunkers.hierarchical_chunker import HierarchicalChunker

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        md_file = md_dir / "page.md"
        md_file.write_text(
            "# Admissions\n\nOverview paragraph.\n\n## Tuition\n\nFee details.\n\n## Scholarships\n\nScholarship details.",
            encoding="utf-8",
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"chunker": {"max_tokens": 120, "include_section_headings": True}},
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
            stage_definition={"type": "chunker", "plugin": "hierarchical"},
            stage_id="chunk_content",
        )

        result = run_async(HierarchicalChunker().execute(ctx))
        chunk_index = load_chunk_index(result.outputs["chunks_file"])

        assert result.outputs["chunk_count"] >= 2
        assert chunk_index["strategy"] == "hierarchical"
        assert chunk_index["chunks"][0]["section_path"][0] == "Admissions"
        assert "Admissions" in chunk_index["chunks"][0]["text"]

    def test_hybrid_chunker_splits_large_markdown_into_multiple_chunks(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import load_chunk_index
        from pipeline.stages.chunkers.hybrid_chunker import HybridChunker

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        md_file = md_dir / "report.md"
        md_file.write_text(
            "# Report\n\n" + "\n\n".join(
                f"## Section {idx}\n\n" + ("MBZUAI content " * 80).strip()
                for idx in range(1, 5)
            ),
            encoding="utf-8",
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={
                "chunker": {
                    "target_tokens": 180,
                    "max_tokens": 220,
                    "overlap_tokens": 40,
                    "min_chunk_tokens": 60,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={"md_dir": str(md_dir)},
            stage_definition={"type": "chunker", "plugin": "hybrid"},
            stage_id="chunk_content",
        )

        result = run_async(HybridChunker().execute(ctx))
        chunk_index = load_chunk_index(result.outputs["chunks_file"])

        assert result.outputs["chunk_count"] > 1
        assert chunk_index["strategy"] == "hybrid"
        assert all(chunk["chunk_count"] == result.outputs["chunk_count"] for chunk in chunk_index["chunks"])
        assert all(chunk["token_count"] > 0 for chunk in chunk_index["chunks"])

    def test_split_text_by_budget_splits_oversized_units_inside_multi_unit_text(self):
        from pipeline.core.chunking import estimate_token_count
        from pipeline.stages.chunkers.common import split_text_by_budget

        text = "\n\n".join(
            [
                "# Admissions",
                ("MBZUAI admissions requirements " * 500).strip(),
                "Short closing note.",
            ]
        )

        chunks = split_text_by_budget(text, max_tokens=120, overlap_tokens=20)

        assert len(chunks) > 3
        assert max(estimate_token_count(chunk) for chunk in chunks) <= 120

    def test_split_text_by_budget_drops_overlap_when_next_unit_would_exceed_budget(self):
        from pipeline.core.chunking import estimate_token_count
        from pipeline.stages.chunkers.common import split_text_by_budget

        text = " ".join(
            [
                ("Intro " * 150).strip() + ".",
                ("Program " * 150).strip() + ".",
                ("Research " * 150).strip() + ".",
                ("Country " * 450).strip() + ".",
            ]
        )

        chunks = split_text_by_budget(text, max_tokens=650, overlap_tokens=80)

        assert len(chunks) >= 2
        assert max(estimate_token_count(chunk) for chunk in chunks) <= 650

    def test_docling_hybrid_chunker_uses_configured_token_budget(self):
        from pipeline.stages.chunkers.common import _build_docling_hybrid_chunker

        chunker = _build_docling_hybrid_chunker(650, always_emit_headings=False)

        assert chunker.tokenizer.get_max_tokens() == 650
        assert chunker.tokenizer.count_tokens("<|fim_prefix|> MBZUAI") > 0


# ─────────────────────────────────────────────────────────────
# 13. Extension Sets — verify crawl filtering logic is consistent
# ─────────────────────────────────────────────────────────────

class TestExtensionSets:
    def test_content_images_not_excluded_from_crawl(self):
        """jpg/png/gif/webp should NOT be in EXCLUDED_EXTENSIONS."""
        from pipeline.stages.crawlers.crawl4ai_crawler import EXCLUDED_EXTENSIONS, IMAGE_EXTENSIONS
        for ext in IMAGE_EXTENSIONS:
            assert ext not in EXCLUDED_EXTENSIONS, f"{ext} should be crawlable"

    def test_ico_and_svg_are_excluded(self):
        from pipeline.stages.crawlers.crawl4ai_crawler import EXCLUDED_EXTENSIONS
        assert ".ico" in EXCLUDED_EXTENSIONS
        assert ".svg" in EXCLUDED_EXTENSIONS

    def test_downloadable_and_excluded_are_disjoint(self):
        """No extension should be in both DOWNLOADABLE and EXCLUDED."""
        from pipeline.stages.crawlers.crawl4ai_crawler import DOWNLOADABLE_EXTENSIONS, EXCLUDED_EXTENSIONS
        overlap = DOWNLOADABLE_EXTENSIONS & EXCLUDED_EXTENSIONS
        assert overlap == set(), f"Overlap: {overlap}"


# ─────────────────────────────────────────────────────────────
# 14. MarkItDown Converter — test image preservation
# ─────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_MARKITDOWN, reason="markitdown not installed")
class TestMarkItDownConverter:
    def test_convert_one_appends_images(self, tmp_dir):
        """Verify _convert_one appends image references to markdown output."""
        from pipeline.stages.converters.markitdown_converter import _convert_one

        # Create a minimal HTML file
        html_content = "<html><body><h1>Test</h1><p>Content here.</p></body></html>"
        html_path = tmp_dir / "test.html"
        html_path.write_text(html_content)

        md_dir = tmp_dir / "md_output"
        md_dir.mkdir()

        page_images = [
            {"url": "https://example.com/photo.jpg", "alt": "Photo", "local_path": "/images/photo.jpg", "context": ""},
            {"url": "https://example.com/chart.png", "alt": "", "local_path": "", "context": "Sales chart"},
        ]

        result = _convert_one(
            html_path,
            md_dir,
            page_images,
            append_only_semantic_images=False,
        )
        assert result is not None
        html_name, md_path = result

        md_content = Path(md_path).read_text()
        # First image uses local_path and may be relativized against the markdown output.
        assert "![Photo](" in md_content
        assert "photo.jpg" in md_content
        # Second image uses URL since no local_path
        assert "![Sales chart](https://example.com/chart.png)" in md_content

    def test_convert_one_without_images(self, tmp_dir):
        """Conversion should work fine without any images."""
        from pipeline.stages.converters.markitdown_converter import _convert_one

        html_path = tmp_dir / "plain.html"
        html_path.write_text("<html><body><h1>Title</h1><p>Text.</p></body></html>")

        md_dir = tmp_dir / "md"
        md_dir.mkdir()

        result = _convert_one(html_path, md_dir)
        assert result is not None
        _, md_path = result
        md_content = Path(md_path).read_text()
        assert "![" not in md_content  # no images added

    def test_convert_one_appends_video_block(self, tmp_dir):
        from pipeline.stages.converters.markitdown_converter import _convert_one

        html_path = tmp_dir / "video.html"
        html_path.write_text("<html><body><h1>Title</h1><p>Text.</p></body></html>")

        md_dir = tmp_dir / "md"
        md_dir.mkdir()

        page_media = [
            {
                "type": "video",
                "url": "https://example.com/intro.mp4",
                "title": "Intro video",
                "poster_url": "https://example.com/intro.jpg",
                "caption": "Campus introduction",
            }
        ]

        result = _convert_one(html_path, md_dir, page_media)
        assert result is not None
        _, md_path = result
        md_content = Path(md_path).read_text()
        assert "## Embedded Media" in md_content
        assert "[Watch video](https://example.com/intro.mp4)" in md_content
        assert "![Intro video poster](https://example.com/intro.jpg)" in md_content

    def test_convert_one_skips_non_semantic_images_when_enabled(self, tmp_dir):
        from pipeline.stages.converters.markitdown_converter import _convert_one

        html_path = tmp_dir / "images.html"
        html_path.write_text("<html><body><h1>Title</h1><p>Text.</p></body></html>", encoding="utf-8")

        md_dir = tmp_dir / "md"
        md_dir.mkdir()

        page_images = [
            {"type": "image", "url": "https://example.com/generic.jpg", "alt": "Image"},
            {"type": "image", "url": "https://example.com/chart.png", "caption": "Admissions growth chart"},
        ]

        result = _convert_one(
            html_path,
            md_dir,
            page_images,
            append_only_semantic_images=True,
        )
        assert result is not None
        _, md_path = result
        md_content = Path(md_path).read_text(encoding="utf-8")
        assert "generic.jpg" not in md_content
        assert "Admissions growth chart" in md_content

    def test_execute_uses_cleaned_html_artifact_source_url_for_media_attachment(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.converters.markitdown_converter import MarkItDownConverter

        cleaned_dir = tmp_dir / "cleaned_html"
        cleaned_dir.mkdir()
        html_path = cleaned_dir / "page.html"
        html_path.write_text("<html><body><h1>Title</h1><p>Content.</p></body></html>", encoding="utf-8")

        mapping_file = tmp_dir / "mapping.json"
        atomic_write_json(mapping_file, {"https://example.com/page": str(tmp_dir / "raw" / "page.html")})

        page_media_file = tmp_dir / "page_media.json"
        atomic_write_json(
            page_media_file,
            {
                "https://example.com/page": [
                    {
                        "type": "video",
                        "url": "https://example.com/intro.mp4",
                        "title": "Intro video",
                    }
                ]
            },
        )

        catalog = ArtifactCatalog.from_dict(
            {
                "records": [
                    {
                        "artifact_id": "clean1",
                        "artifact_type": "cleaned_html",
                        "role": "content",
                        "producer_stage": "clean_html",
                        "uri": html_path.resolve().as_uri(),
                        "local_path": str(html_path),
                        "metadata": {
                            "source_url": "https://example.com/page",
                            "relative_path": "page.html",
                        },
                    }
                ]
            }
        )

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"converter": {"overwrite": True, "max_workers": 1}},
            work_dir=tmp_dir,
            previous_outputs={
                "mapping_file": str(mapping_file),
                "page_media_file": str(page_media_file),
            },
            stage_definition={"type": "converter", "plugin": "markitdown"},
            stage_id="convert_html",
            artifact_catalog=catalog,
        )

        result = run_async(MarkItDownConverter().execute(ctx))
        md_file = Path(result.outputs["md_dir"]) / "page.md"
        md_content = md_file.read_text(encoding="utf-8")
        assert "## Embedded Media" in md_content
        assert "[Watch video](https://example.com/intro.mp4)" in md_content


class TestMediaHelpers:
    def test_build_retrieval_documents_parses_media_metadata(self):
        from pipeline.core.media import build_retrieval_documents

        docs = [
            {
                "id": "doc1",
                "text": "Text",
                "metadata": {
                    "document_source": "https://example.com/page",
                    "media": json.dumps(
                        [
                            {"type": "image", "url": "https://example.com/photo.jpg", "alt": "Photo"},
                            {"type": "video", "url": "https://example.com/intro.mp4", "title": "Intro"},
                        ]
                    ),
                },
            }
        ]

        retrieval_docs = build_retrieval_documents(docs, max_media_per_doc=5, max_total_media=5)
        assert retrieval_docs[0]["media"][0]["url"] == "https://example.com/photo.jpg"
        assert retrieval_docs[0]["media"][1]["type"] == "video"


class TestGeminiRetrievalFormatter:
    def test_extract_fact_snippets_prioritizes_faq_and_hours_bullets(self):
        from pipeline.stages.formatters.gemini_retrieval_formatter import _extract_fact_snippets

        faq_chunk = {
            "text": (
                "How do I get to campus?\n"
                "MBZUAI is located in Masdar City, Abu Dhabi.\n\n"
                "Does MBZUAI provide a shuttle bus service?\n"
                "Yes, a shuttle service connects students to key locations.\n\n"
                "Can I stay on campus between semesters?\n"
                "Yes, accommodation is available during the winter and spring breaks.\n\n"
                "Can my parents stay with me on campus?\n"
                "No, MBZUAI does not provide housing for parents. However, nearby hotels or Airbnbs can be recommended.\n"
            )
        }
        faq_snippets = _extract_fact_snippets(faq_chunk, min_chars=20, max_chars=220, max_snippets=5)
        assert any("Can my parents stay with me on campus?" in snippet for snippet in faq_snippets)
        assert any("housing for parents" in snippet.lower() for snippet in faq_snippets)

        bullet_chunk = {
            "text": (
                "Additional reminders\n"
                "- The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae for technical support. "
                "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays.\n"
            )
        }
        bullet_snippets = _extract_fact_snippets(bullet_chunk, min_chars=20, max_chars=220, max_snippets=5)
        assert any("working hours are at 8:00 am - 5:00 pm" in snippet.lower() for snippet in bullet_snippets)

    def test_execute_builds_retrieval_bundle_and_media_links(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.gemini_retrieval_formatter import GeminiRetrievalFormatter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        md_file = md_dir / "page.md"
        md_file.write_text("# Research\n\nChunk A\n\nChunk B", encoding="utf-8")

        chunks_file = tmp_dir / "chunk_index.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc1",
                        "chunk_id": "chunk-a",
                        "chunk_index": 0,
                        "chunk_count": 2,
                        "strategy": "hybrid",
                        "text": "Admissions overview for MBZUAI.",
                        "section_path": ["Admissions"],
                        "page_numbers": [],
                        "document_title": "MBZUAI",
                        "document_type": "webpage",
                        "source_backend": "markitdown",
                        "source_file": str(md_file),
                        "source_markdown_path": str(md_file),
                        "source_url": "https://example.com/page",
                    },
                    {
                        "document_id": "doc1",
                        "chunk_id": "chunk-b",
                        "chunk_index": 1,
                        "chunk_count": 2,
                        "strategy": "hybrid",
                        "text": "Research programs and labs.",
                        "section_path": ["Research"],
                        "page_numbers": [],
                        "document_title": "MBZUAI",
                        "document_type": "webpage",
                        "source_backend": "markitdown",
                        "source_file": str(md_file),
                        "source_markdown_path": str(md_file),
                        "source_url": "https://example.com/page",
                    },
                ],
                strategy="hybrid",
            ),
        )

        image_file = tmp_dir / "image.png"
        image_file.write_bytes(b"fake")
        catalog = ArtifactCatalog()
        catalog.add(
            build_artifact_record(
                artifact_type="chunk_index",
                role="retrieval_chunks",
                producer_stage="chunk_content",
                uri=chunks_file.resolve().as_uri(),
                local_path=chunks_file,
            )
        )
        catalog.add(
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_html",
                uri=md_file.resolve().as_uri(),
                local_path=md_file,
                metadata={"source_url": "https://example.com/page"},
            )
        )
        catalog.add(
            build_artifact_record(
                artifact_type="extracted_image",
                role="document_media",
                producer_stage="convert_documents",
                uri=image_file.resolve().as_uri(),
                local_path=image_file,
                metadata={
                    "id": "img1",
                    "type": "image",
                    "url": image_file.resolve().as_uri(),
                    "local_path": str(image_file),
                    "source_document_path": str(md_file),
                    "caption": "Campus map",
                },
            )
        )

        page_media_file = tmp_dir / "page_media.json"
        atomic_write_json(
            page_media_file,
            {
                "https://example.com/page": [
                    {
                        "type": "video",
                        "url": "https://example.com/intro.mp4",
                        "title": "Intro video",
                        "caption": "Welcome to MBZUAI",
                        "section_path": ["Research"],
                        "section_heading": "Research",
                        "surrounding_text_after": "Research programs and laboratory tours.",
                    }
                ]
            },
        )

        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"formatter": {"fact_min_chars": 20}},
            work_dir=tmp_dir,
            previous_outputs={"page_media_file": str(page_media_file)},
            stage_definition={"type": "formatter", "plugin": "gemini_retrieval"},
            stage_id="format_retrieval",
            artifact_catalog=catalog,
        )

        result = run_async(GeminiRetrievalFormatter().execute(ctx))
        bundle = load_json_safe(result.outputs["retrieval_bundle_file"])
        assert result.metrics["chunk_records"] == 2
        assert result.metrics["parent_records"] >= 2
        assert result.metrics["media_records"] == 2
        assert result.metrics["fact_records"] >= 2
        assert len(bundle["chunk_records"]) == 2
        assert len(bundle["fact_records"]) >= 2
        # Unscoped document-level images remain independently retrievable;
        # only media with page/section/context provenance attaches to chunks.
        assert bundle["media_records"][0]["linked_chunk_ids"] == []
        assert any(record["linked_chunk_ids"] for record in bundle["media_records"])
        contextual_video = next(
            record
            for record in bundle["media_records"]
            if record.get("url") == "https://example.com/intro.mp4"
        )
        assert contextual_video["linked_chunk_ids"] == ["chunk-b"]
        assert contextual_video["section_path"] == ["Research"]
        assert "SURROUNDING_TEXT_AFTER" in contextual_video["text"]
        assert bundle["parent_records"][0]["child_chunk_ids"]

    def test_execute_skips_low_signal_media_records(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.gemini_retrieval_formatter import GeminiRetrievalFormatter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        md_file = md_dir / "page.md"
        md_file.write_text("# Campus map\n\nMap content", encoding="utf-8")

        chunks_file = tmp_dir / "chunk_index.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc1",
                        "chunk_id": "chunk-a",
                        "chunk_index": 0,
                        "chunk_count": 1,
                        "strategy": "hybrid",
                        "text": "Campus map of MBZUAI.",
                        "section_path": ["Campus map"],
                        "document_title": "MBZUAI Map",
                        "document_type": "webpage",
                        "source_backend": "markitdown",
                        "source_file": str(md_file),
                        "source_markdown_path": str(md_file),
                        "source_url": "https://example.com/map",
                    },
                ],
                strategy="hybrid",
            ),
        )

        image_file = tmp_dir / "image.png"
        image_file.write_bytes(b"fake")

        catalog = ArtifactCatalog()
        catalog.add(
            build_artifact_record(
                artifact_type="chunk_index",
                role="retrieval_chunks",
                producer_stage="chunk_content",
                uri=chunks_file.resolve().as_uri(),
                local_path=chunks_file,
            )
        )
        catalog.add(
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_html",
                uri=md_file.resolve().as_uri(),
                local_path=md_file,
                metadata={"source_url": "https://example.com/map"},
            )
        )
        catalog.add(
            build_artifact_record(
                artifact_type="extracted_image",
                role="document_media",
                producer_stage="convert_documents",
                uri=image_file.resolve().as_uri(),
                local_path=image_file,
                metadata={
                    "id": "img-low-signal",
                    "type": "image",
                    "url": image_file.resolve().as_uri(),
                    "local_path": str(image_file),
                    "source_document_path": str(md_file),
                    "title": "Figure 2",
                    "caption": "",
                    "description": "I'm sorry, but I cannot provide a description or answer for the image you've mentioned as it doesn't contain any text or information that I can describe.",
                },
            )
        )

        ctx = StageContext(
            run_id="r-low-signal",
            project_name="p-low-signal",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={},
            stage_definition={"type": "formatter", "plugin": "gemini_retrieval"},
            stage_id="format_retrieval",
            artifact_catalog=catalog,
        )

        result = run_async(GeminiRetrievalFormatter().execute(ctx))
        bundle = load_json_safe(result.outputs["retrieval_bundle_file"])
        assert result.metrics["media_records"] == 0
        assert bundle["media_records"] == []

    def test_execute_bounds_sparse_text_for_large_parent_records(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.gemini_retrieval_formatter import GeminiRetrievalFormatter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        md_file = md_dir / "large.md"
        md_file.write_text("# Large\n\nBody", encoding="utf-8")

        huge_text = ("Important section detail " * 800).strip()
        chunks_file = tmp_dir / "chunk_index.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc-large",
                        "chunk_id": "chunk-a",
                        "chunk_index": 0,
                        "chunk_count": 2,
                        "strategy": "hybrid",
                        "text": huge_text,
                        "heading": "Section A",
                        "section_path": ["Programs", "A"],
                        "page_numbers": [1],
                        "document_title": "MBZUAI Catalogue",
                        "document_type": "pdf",
                        "source_backend": "docling",
                        "source_file": str(md_file),
                        "source_markdown_path": str(md_file),
                        "source_url": "https://example.com/catalogue",
                    },
                    {
                        "document_id": "doc-large",
                        "chunk_id": "chunk-b",
                        "chunk_index": 1,
                        "chunk_count": 2,
                        "strategy": "hybrid",
                        "text": huge_text,
                        "heading": "Section B",
                        "section_path": ["Programs", "A"],
                        "page_numbers": [1],
                        "document_title": "MBZUAI Catalogue",
                        "document_type": "pdf",
                        "source_backend": "docling",
                        "source_file": str(md_file),
                        "source_markdown_path": str(md_file),
                        "source_url": "https://example.com/catalogue",
                    },
                ],
                strategy="hybrid",
            ),
        )

        catalog = ArtifactCatalog()
        catalog.add(
            build_artifact_record(
                artifact_type="chunk_index",
                role="retrieval_chunks",
                producer_stage="chunk_content",
                uri=chunks_file.resolve().as_uri(),
                local_path=chunks_file,
            )
        )
        catalog.add(
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_documents",
                uri=md_file.resolve().as_uri(),
                local_path=md_file,
                metadata={"source_url": "https://example.com/catalogue"},
            )
        )

        ctx = StageContext(
            run_id="r2",
            project_name="p2",
            config={
                "formatter": {
                    "sparse_chunk_max_chars": 120,
                    "sparse_parent_max_chars": 180,
                    "sparse_parent_max_headings": 4,
                    "sparse_parent_max_snippets": 2,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={},
            stage_definition={"type": "formatter", "plugin": "gemini_retrieval"},
            stage_id="format_retrieval",
            artifact_catalog=catalog,
        )

        result = run_async(GeminiRetrievalFormatter().execute(ctx))
        chunk_records = load_json_safe(result.outputs["chunk_embedding_file"])
        parent_records = load_json_safe(result.outputs["parent_embedding_file"])
        lexical_records = load_json_safe(result.outputs["lexical_corpus_file"])

        assert max(len(record["sparse_text"]) for record in chunk_records) <= 123
        assert max(len(record["sparse_text"]) for record in parent_records) <= 183
        parent_lexical = [record for record in lexical_records if record["record_type"] == "parent"]
        assert parent_lexical
        assert max(len(record["text"]) for record in parent_lexical) <= 183


class TestKnowledgeGraphFormatter:
    def test_execute_builds_deterministic_graph_bundle(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.knowledge_graph_formatter import KnowledgeGraphFormatter

        retrieval_bundle = {
            "chunk_records": [
                {
                    "id": "chunk1",
                    "document_id": "doc1",
                    "document_title": "MBZUAI FAQ",
                    "document_type": "webpage",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "chunk_index": 0,
                    "chunk_count": 2,
                    "heading": "Location",
                    "section_path": ["FAQ"],
                    "page_numbers": [],
                    "page_key": "page1",
                    "section_key": "section1",
                    "neighbor_ids": ["chunk2"],
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                },
                {
                    "id": "chunk2",
                    "document_id": "doc1",
                    "document_title": "MBZUAI FAQ",
                    "document_type": "webpage",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "chunk_index": 1,
                    "chunk_count": 2,
                    "heading": "Transport",
                    "section_path": ["FAQ"],
                    "page_numbers": [],
                    "page_key": "page1",
                    "section_key": "section1",
                    "neighbor_ids": ["chunk1"],
                    "text": "A shuttle service is available.",
                },
            ],
            "parent_records": [
                {
                    "id": "page1",
                    "parent_type": "page",
                    "document_id": "doc1",
                    "document_title": "MBZUAI FAQ",
                    "document_type": "webpage",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "page_numbers": [],
                    "child_chunk_ids": ["chunk1", "chunk2"],
                },
                {
                    "id": "section1",
                    "parent_type": "section",
                    "document_id": "doc1",
                    "document_title": "MBZUAI FAQ",
                    "document_type": "webpage",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "page_numbers": [],
                    "page_key": "page1",
                    "section_path": ["FAQ"],
                    "child_chunk_ids": ["chunk1", "chunk2"],
                },
            ],
            "media_records": [
                {
                    "id": "media1",
                    "media_type": "image",
                    "title": "Campus map",
                    "description": "Campus map image.",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "page_key": "page1",
                    "section_keys": ["section1"],
                    "linked_chunk_ids": ["chunk1"],
                    "linked_parent_ids": ["page1", "section1"],
                }
            ],
            "fact_records": [
                {
                    "id": "fact1",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "heading": "Location",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "page_key": "page1",
                    "section_key": "section1",
                    "linked_chunk_ids": ["chunk1"],
                    "linked_parent_ids": ["section1", "page1"],
                }
            ],
        }
        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(bundle_file, retrieval_bundle)

        ctx = StageContext(
            run_id="r-graph",
            project_name="p-graph",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={"retrieval_bundle_file": str(bundle_file)},
            stage_definition={"type": "formatter", "plugin": "knowledge_graph"},
            stage_id="format_graph",
        )

        result = run_async(KnowledgeGraphFormatter().execute(ctx))
        graph = load_json_safe(result.outputs["knowledge_graph_file"], {})
        graph_index = load_json_safe(result.outputs["knowledge_graph_index_file"], {})

        assert result.metrics["document_nodes"] == 1
        assert result.metrics["page_nodes"] == 1
        assert result.metrics["section_nodes"] == 1
        assert result.metrics["chunk_nodes"] == 2
        assert result.metrics["fact_nodes"] == 1
        assert result.metrics["media_nodes"] == 1
        assert graph["stats"]["edge_type_counts"]["CHUNK_HAS_FACT"] == 1
        assert graph["stats"]["edge_type_counts"]["CHUNK_HAS_MEDIA"] == 1
        assert graph["stats"]["edge_type_counts"]["NEXT_CHUNK"] == 2
        assert "chunk1" in graph_index["outgoing_edge_ids"]

    def test_execute_drops_edges_with_missing_endpoints(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.knowledge_graph_formatter import KnowledgeGraphFormatter

        retrieval_bundle = {
            "chunk_records": [
                {
                    "id": "chunk1",
                    "document_id": "doc1",
                    "document_title": "MBZUAI FAQ",
                    "document_type": "webpage",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "chunk_index": 0,
                    "chunk_count": 1,
                    "heading": "Location",
                    "section_path": ["FAQ"],
                    "page_numbers": [],
                    "page_key": "page1",
                    "section_key": "section1",
                    "neighbor_ids": ["missing-chunk"],
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                }
            ],
            "parent_records": [
                {
                    "id": "section1",
                    "parent_type": "section",
                    "document_id": "doc1",
                    "document_title": "MBZUAI FAQ",
                    "document_type": "webpage",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "page_numbers": [],
                    "section_path": ["FAQ"],
                    "child_chunk_ids": ["chunk1"],
                },
            ],
            "media_records": [
                {
                    "id": "media1",
                    "media_type": "image",
                    "title": "Campus map",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "linked_chunk_ids": ["missing-chunk"],
                    "linked_parent_ids": ["missing-parent"],
                }
            ],
            "fact_records": [
                {
                    "id": "fact1",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "source_markdown_path": str(tmp_dir / "faq.md"),
                    "source_url": "https://example.com/faq",
                    "linked_chunk_ids": ["missing-chunk"],
                    "linked_parent_ids": ["missing-parent"],
                }
            ],
        }
        bundle_file = tmp_dir / "retrieval_bundle_missing_refs.json"
        atomic_write_json(bundle_file, retrieval_bundle)

        ctx = StageContext(
            run_id="r-graph-missing",
            project_name="p-graph-missing",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={"retrieval_bundle_file": str(bundle_file)},
            stage_definition={"type": "formatter", "plugin": "knowledge_graph"},
            stage_id="format_graph",
        )

        result = run_async(KnowledgeGraphFormatter().execute(ctx))
        graph = load_json_safe(result.outputs["knowledge_graph_file"], {})
        edge_types = [edge["edge_type"] for edge in graph["edges"]]

        assert result.metrics["graph_invalid_edges_dropped"] > 0
        assert "NEXT_CHUNK" not in edge_types
        assert "CHUNK_HAS_FACT" not in edge_types
        assert "CHUNK_HAS_MEDIA" not in edge_types


class TestSemanticGraphStages:
    def test_extract_stage_writes_candidate_entities_and_relations(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters import semantic_graph_extract_formatter as mod

        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(
            bundle_file,
            {
                "fact_records": [
                    {
                        "id": "fact1",
                        "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                        "heading": "Location",
                        "document_title": "FAQ",
                        "source_url": "https://example.com/faq",
                        "linked_chunk_ids": ["chunk1"],
                        "linked_parent_ids": ["page1", "section1"],
                    }
                ],
                "chunk_records": [],
            },
        )

        monkeypatch.setattr(
            mod,
            "_call_gemini_structured",
            lambda **kwargs: {
                "items": [
                    {
                        "source_id": "fact1",
                        "entities": [
                            {"name": "MBZUAI", "entity_type": "Organization", "aliases": ["Mohamed bin Zayed University of Artificial Intelligence"], "confidence": 0.94},
                            {"name": "Masdar City", "entity_type": "Location", "confidence": 0.90},
                        ],
                        "relations": [
                            {
                                "subject_name": "MBZUAI",
                                "subject_type": "Organization",
                                "relation_type": "LOCATED_IN",
                                "object_name": "Masdar City",
                                "object_type": "Location",
                                "evidence": "MBZUAI is located in Masdar City, Abu Dhabi.",
                                "confidence": 0.92,
                            }
                        ],
                    }
                ]
            },
        )

        ctx = StageContext(
            run_id="r-semantic",
            project_name="p-semantic",
            config={"graph": {"extraction_batch_size": 1}},
            work_dir=tmp_dir,
            previous_outputs={"retrieval_bundle_file": str(bundle_file)},
            stage_definition={"type": "formatter", "plugin": "semantic_graph_extract"},
            stage_id="extract_semantic_graph",
        )

        result = run_async(mod.SemanticGraphExtractFormatter().execute(ctx))
        entities = load_json_safe(result.outputs["semantic_entity_candidates_file"], [])
        relations = load_json_safe(result.outputs["semantic_relation_candidates_file"], [])
        extraction = load_json_safe(result.outputs["semantic_extraction_file"], {})

        assert result.metrics["semantic_sources"] == 1
        assert len(entities) == 2
        assert len(relations) == 1
        assert extraction["provider"] == "gemini"
        assert extraction["model"] == "gemini-2.5-flash"

    def test_extract_stage_fails_open_on_retryable_gemini_error(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters import semantic_graph_extract_formatter as mod

        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(
            bundle_file,
            {
                "fact_records": [
                    {
                        "id": "fact1",
                        "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                        "heading": "Location",
                        "document_title": "FAQ",
                        "source_url": "https://example.com/faq",
                        "linked_chunk_ids": ["chunk1"],
                        "linked_parent_ids": ["page1", "section1"],
                    }
                ],
                "chunk_records": [],
            },
        )

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(
            mod,
            "_call_gemini_structured",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("503 UNAVAILABLE")),
        )

        ctx = StageContext(
            run_id="r-semantic-open",
            project_name="p-semantic-open",
            config={
                "graph": {
                    "extraction_batch_size": 1,
                    "extraction_retry_attempts": 2,
                    "extraction_retry_base_delay_sec": 0.01,
                    "extraction_retry_max_delay_sec": 0.01,
                    "extraction_fail_open_after_retries": True,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={"retrieval_bundle_file": str(bundle_file)},
            stage_definition={"type": "formatter", "plugin": "semantic_graph_extract"},
            stage_id="extract_semantic_graph",
        )

        result = run_async(mod.SemanticGraphExtractFormatter().execute(ctx))
        entities = load_json_safe(result.outputs["semantic_entity_candidates_file"], [])
        relations = load_json_safe(result.outputs["semantic_relation_candidates_file"], [])
        extraction = load_json_safe(result.outputs["semantic_extraction_file"], {})
        errors = load_json_safe(ctx.stage_work_dir / "semantic_extraction_errors.json", [])

        assert result.status.value == "completed"
        assert entities == []
        assert relations == []
        assert extraction["error_count"] == 1
        assert result.metrics["semantic_failed_sources"] == 1
        assert result.metrics["semantic_extraction_errors"] == 1
        assert len(errors) == 1

    def test_extract_stage_restores_stale_cache_on_restart(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.core.semantic_graph import stable_semantic_id
        from pipeline.stages.formatters import semantic_graph_extract_formatter as mod

        bundle_file = tmp_dir / "retrieval_bundle.json"
        fact_text = "MBZUAI is located in Masdar City, Abu Dhabi."
        atomic_write_json(
            bundle_file,
            {
                "fact_records": [
                    {
                        "id": "fact1",
                        "text": fact_text,
                        "heading": "Location",
                        "document_title": "FAQ",
                        "source_url": "https://example.com/faq",
                        "linked_chunk_ids": ["chunk1"],
                        "linked_parent_ids": ["page1", "section1"],
                    }
                ],
                "chunk_records": [],
            },
        )
        stale_cache = {
            stable_semantic_id("semantic_extract", "fact1", fact_text): {
                "source_id": "fact1",
                "entities": [
                    {
                        "id": "candidate_entity:1",
                        "source_id": "fact1",
                        "source_kind": "fact",
                        "name": "MBZUAI",
                        "entity_type": "Organization",
                        "aliases": [],
                        "description": "",
                        "confidence": 0.9,
                        "source_chunk_ids": ["chunk1"],
                        "source_fact_ids": ["fact1"],
                        "source_parent_ids": ["page1", "section1"],
                        "source_url": "https://example.com/faq",
                        "document_title": "FAQ",
                    }
                ],
                "relations": [],
            }
        }
        stale_dir = tmp_dir / "stage_outputs" / "_stale" / "extract_semantic_graph_20260321T000000Z"
        stale_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(stale_dir / "semantic_extraction_cache.json", stale_cache)

        monkeypatch.setattr(
            mod,
            "_call_gemini_structured",
            lambda **kwargs: (_ for _ in ()).throw(AssertionError("fresh extraction should not run")),
        )

        ctx = StageContext(
            run_id="r-semantic-restore",
            project_name="p-semantic-restore",
            config={"graph": {"extraction_resume_cache": True}},
            work_dir=tmp_dir,
            previous_outputs={"retrieval_bundle_file": str(bundle_file)},
            stage_definition={"type": "formatter", "plugin": "semantic_graph_extract"},
            stage_id="extract_semantic_graph",
        )

        result = run_async(mod.SemanticGraphExtractFormatter().execute(ctx))
        entities = load_json_safe(result.outputs["semantic_entity_candidates_file"], [])
        assert result.status.value == "completed"
        assert len(entities) == 1
        assert entities[0]["name"] == "MBZUAI"

    def test_canonicalize_stage_merges_entities_and_builds_assertions(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.semantic_graph_canonicalize_formatter import SemanticGraphCanonicalizeFormatter

        entity_candidates_file = tmp_dir / "candidate_entities.json"
        relation_candidates_file = tmp_dir / "candidate_relations.json"
        atomic_write_json(
            entity_candidates_file,
            [
                {
                    "id": "cand-1",
                    "entity_type": "Organization",
                    "name": "MBZUAI",
                    "aliases": ["Mohamed bin Zayed University of Artificial Intelligence"],
                    "description": "University",
                    "confidence": 0.93,
                    "source_chunk_ids": ["chunk1"],
                    "source_fact_ids": ["fact1"],
                    "source_parent_ids": ["page1"],
                    "source_url": "https://example.com/faq",
                    "document_title": "FAQ",
                },
                {
                    "id": "cand-2",
                    "entity_type": "Organization",
                    "name": "MBZUAI",
                    "aliases": ["MBZUAI"],
                    "description": "",
                    "confidence": 0.88,
                    "source_chunk_ids": ["chunk2"],
                    "source_fact_ids": [],
                    "source_parent_ids": ["page1"],
                    "source_url": "https://example.com/faq",
                    "document_title": "FAQ",
                },
            ],
        )
        atomic_write_json(
            relation_candidates_file,
            [
                {
                    "id": "rel-1",
                    "source_id": "fact1",
                    "source_kind": "fact",
                    "subject_name": "MBZUAI",
                    "subject_type": "Organization",
                    "relation_type": "LOCATED_IN",
                    "object_name": "Masdar City",
                    "object_type": "Location",
                    "evidence": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "confidence": 0.91,
                    "source_chunk_ids": ["chunk1"],
                    "source_fact_ids": ["fact1"],
                    "source_parent_ids": ["page1"],
                    "source_url": "https://example.com/faq",
                    "document_title": "FAQ",
                }
            ],
        )

        ctx = StageContext(
            run_id="r-canonical",
            project_name="p-canonical",
            config={"graph": {"promote_min_confidence": 0.65}},
            work_dir=tmp_dir,
            previous_outputs={
                "semantic_entity_candidates_file": str(entity_candidates_file),
                "semantic_relation_candidates_file": str(relation_candidates_file),
            },
            stage_definition={"type": "formatter", "plugin": "semantic_graph_canonicalize"},
            stage_id="canonicalize_semantic_graph",
        )

        result = run_async(SemanticGraphCanonicalizeFormatter().execute(ctx))
        entities = load_json_safe(result.outputs["semantic_entities_file"], [])
        assertions = load_json_safe(result.outputs["semantic_assertions_file"], [])

        assert len(entities) == 2
        mbzuai = next(entity for entity in entities if entity["canonical_name"] == "MBZUAI")
        assert "Mohamed bin Zayed University of Artificial Intelligence" in mbzuai["aliases"]
        assert len(assertions) == 1
        assert assertions[0]["relation_type"] == "LOCATED_IN"

    def test_promote_stage_merges_semantic_entities_and_assertions(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.semantic_graph_promote_formatter import SemanticGraphPromoteFormatter

        base_graph_file = tmp_dir / "knowledge_graph.json"
        atomic_write_json(
            base_graph_file,
            {
                "schema_version": 1,
                "graph_type": "deterministic_content_graph",
                "nodes": [
                    {"id": "chunk1", "node_type": "chunk", "label": "Chunk"},
                    {"id": "fact1", "node_type": "fact", "label": "Fact"},
                ],
                "edges": [
                    {"id": "edge-1", "edge_type": "CHUNK_HAS_FACT", "source_id": "chunk1", "target_id": "fact1"},
                ],
                "stats": {"node_count": 2, "edge_count": 1, "node_type_counts": {"chunk": 1, "fact": 1}, "edge_type_counts": {"CHUNK_HAS_FACT": 1}},
            },
        )
        entities_file = tmp_dir / "canonical_entities.json"
        assertions_file = tmp_dir / "canonical_assertions.json"
        atomic_write_json(
            entities_file,
            [
                {
                    "id": "entity:mbzuai",
                    "node_type": "entity",
                    "entity_type": "Organization",
                    "canonical_name": "MBZUAI",
                    "aliases": ["MBZUAI"],
                    "description": "University",
                    "confidence": 0.93,
                    "source_chunk_ids": ["chunk1"],
                    "source_fact_ids": ["fact1"],
                    "source_parent_ids": ["page1"],
                    "source_urls": ["https://example.com/faq"],
                    "document_titles": ["FAQ"],
                },
                {
                    "id": "entity:masdar",
                    "node_type": "entity",
                    "entity_type": "Location",
                    "canonical_name": "Masdar City",
                    "aliases": ["Masdar City"],
                    "description": "",
                    "confidence": 0.90,
                    "source_chunk_ids": ["chunk1"],
                    "source_fact_ids": ["fact1"],
                    "source_parent_ids": ["page1"],
                    "source_urls": ["https://example.com/faq"],
                    "document_titles": ["FAQ"],
                },
            ],
        )
        atomic_write_json(
            assertions_file,
            [
                {
                    "id": "assertion:1",
                    "relation_type": "LOCATED_IN",
                    "subject_entity_id": "entity:mbzuai",
                    "object_entity_id": "entity:masdar",
                    "subject_name": "MBZUAI",
                    "object_name": "Masdar City",
                    "evidence": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "confidence": 0.91,
                    "source_chunk_ids": ["chunk1"],
                    "source_fact_ids": ["fact1"],
                    "source_parent_ids": ["page1"],
                    "source_url": "https://example.com/faq",
                    "document_title": "FAQ",
                }
            ],
        )

        ctx = StageContext(
            run_id="r-promote",
            project_name="p-promote",
            config={},
            work_dir=tmp_dir,
            previous_outputs={
                "knowledge_graph_file": str(base_graph_file),
                "semantic_entities_file": str(entities_file),
                "semantic_assertions_file": str(assertions_file),
            },
            stage_definition={"type": "formatter", "plugin": "semantic_graph_promote"},
            stage_id="promote_graph",
        )

        result = run_async(SemanticGraphPromoteFormatter().execute(ctx))
        promoted = load_json_safe(result.outputs["knowledge_graph_file"], {})
        edge_types = {edge["edge_type"] for edge in promoted["edges"]}
        node_types = {node["node_type"] for node in promoted["nodes"]}

        assert "entity" in node_types
        assert "relation_assertion" in node_types
        assert "FACT_SUPPORTS_ASSERTION" in edge_types
        assert "CHUNK_MENTIONS_ENTITY" in edge_types

    def test_canonicalize_stage_keeps_distinct_multilingual_entities(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.semantic_graph_canonicalize_formatter import SemanticGraphCanonicalizeFormatter

        entity_candidates_file = tmp_dir / "candidate_entities_multilingual.json"
        relation_candidates_file = tmp_dir / "candidate_relations_multilingual.json"
        atomic_write_json(entity_candidates_file, [])
        atomic_write_json(
            relation_candidates_file,
            [
                {
                    "id": "rel-ar-1",
                    "source_id": "fact-ar-1",
                    "source_kind": "fact",
                    "subject_name": "البروفيسورة دانييلا روس",
                    "subject_type": "Person",
                    "relation_type": "RESEARCHES",
                    "object_name": "الروبوتات",
                    "object_type": "Topic",
                    "evidence": "وتركز البروفيسورة اهتماماتها البحثية على مجال الروبوتات والذكاء الاصطناعي.",
                    "confidence": 0.9,
                },
                {
                    "id": "rel-ar-2",
                    "source_id": "fact-ar-1",
                    "source_kind": "fact",
                    "subject_name": "البروفيسورة دانييلا روس",
                    "subject_type": "Person",
                    "relation_type": "RESEARCHES",
                    "object_name": "الذكاء الاصطناعي",
                    "object_type": "Topic",
                    "evidence": "وتركز البروفيسورة اهتماماتها البحثية على مجال الروبوتات والذكاء الاصطناعي.",
                    "confidence": 0.9,
                },
            ],
        )

        ctx = StageContext(
            run_id="r-canonical-multi",
            project_name="p-canonical-multi",
            config={"graph": {"promote_min_confidence": 0.65}},
            work_dir=tmp_dir,
            previous_outputs={
                "semantic_entity_candidates_file": str(entity_candidates_file),
                "semantic_relation_candidates_file": str(relation_candidates_file),
            },
            stage_definition={"type": "formatter", "plugin": "semantic_graph_canonicalize"},
            stage_id="canonicalize_semantic_graph",
        )

        result = run_async(SemanticGraphCanonicalizeFormatter().execute(ctx))
        entities = load_json_safe(result.outputs["semantic_entities_file"], [])
        assertions = load_json_safe(result.outputs["semantic_assertions_file"], [])

        topic_names = {entity["canonical_name"] for entity in entities if entity["entity_type"] == "Topic"}
        assert "الروبوتات" in topic_names
        assert "الذكاء الاصطناعي" in topic_names
        assert len(assertions) == 2

    def test_canonicalize_stage_deduplicates_assertion_ids(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.semantic_graph_canonicalize_formatter import SemanticGraphCanonicalizeFormatter

        entity_candidates_file = tmp_dir / "candidate_entities_dupe.json"
        relation_candidates_file = tmp_dir / "candidate_relations_dupe.json"
        atomic_write_json(entity_candidates_file, [])
        atomic_write_json(
            relation_candidates_file,
            [
                {
                    "id": "rel-1",
                    "source_id": "fact1",
                    "source_kind": "fact",
                    "subject_name": "MBZUAI",
                    "subject_type": "Organization",
                    "relation_type": "LOCATED_IN",
                    "object_name": "Masdar City",
                    "object_type": "Location",
                    "evidence": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "confidence": 0.91,
                    "source_chunk_ids": ["chunk1"],
                    "source_fact_ids": ["fact1"],
                },
                {
                    "id": "rel-2",
                    "source_id": "fact1",
                    "source_kind": "fact",
                    "subject_name": "MBZUAI",
                    "subject_type": "Organization",
                    "relation_type": "LOCATED_IN",
                    "object_name": "Masdar City",
                    "object_type": "Location",
                    "evidence": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "confidence": 0.93,
                    "source_chunk_ids": ["chunk2"],
                    "source_fact_ids": ["fact2"],
                },
            ],
        )

        ctx = StageContext(
            run_id="r-canonical-dupe",
            project_name="p-canonical-dupe",
            config={"graph": {"promote_min_confidence": 0.65}},
            work_dir=tmp_dir,
            previous_outputs={
                "semantic_entity_candidates_file": str(entity_candidates_file),
                "semantic_relation_candidates_file": str(relation_candidates_file),
            },
            stage_definition={"type": "formatter", "plugin": "semantic_graph_canonicalize"},
            stage_id="canonicalize_semantic_graph",
        )

        result = run_async(SemanticGraphCanonicalizeFormatter().execute(ctx))
        assertions = load_json_safe(result.outputs["semantic_assertions_file"], [])

        assert len(assertions) == 1
        assert assertions[0]["confidence"] == 0.93
        assert set(assertions[0]["source_chunk_ids"]) == {"chunk1", "chunk2"}
        assert set(assertions[0]["source_fact_ids"]) == {"fact1", "fact2"}


class TestGeminiPineconeEmbedder:
    def test_upload_plan_rejects_partial_entity_or_sparse_lanes(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _assert_upload_plan_complete

        planned = {"chunks": 10, "entities": 2, "sparse_entities": 2, "communities": 1}
        _assert_upload_plan_complete(planned, dict(planned))

        with pytest.raises(RuntimeError, match="entities.*uploaded.*1"):
            _assert_upload_plan_complete(
                planned,
                {"chunks": 10, "entities": 1, "sparse_entities": 2, "communities": 1},
            )

    def test_release_namespaces_are_stable_and_isolated(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _resolve_upload_namespaces

        base = {
            "namespace_strategy": "release",
            "namespace_chunks": "mbzuai-chunks",
            "namespace_parents": "mbzuai-parents",
        }
        first = _resolve_upload_namespaces(base, run_id="release-2026-07-11")
        repeated = _resolve_upload_namespaces(base, run_id="release-2026-07-11")
        second = _resolve_upload_namespaces(base, run_id="release-2026-07-12")

        assert first == repeated
        assert first["chunks"].startswith("mbzuai-chunks--release-2026-07-11--")
        assert first["chunks"] != second["chunks"]
        assert len(set(first.values())) == len(first)

    def test_make_gemini_client_passes_http_timeout(self, monkeypatch):
        from pipeline.stages.embedders import gemini_pinecone_embedder as module

        captured = {}

        class FakeHttpOptions:
            def __init__(self, *, timeout=None):
                self.timeout = timeout

        class FakeTypes:
            HttpOptions = FakeHttpOptions

        class FakeGenAI:
            class Client:
                def __init__(self, **kwargs):
                    captured.update(kwargs)

        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
        monkeypatch.setenv("GEMINI_API_KEY", "fallback-key")
        monkeypatch.setattr(module, "import_genai", lambda: FakeGenAI)
        monkeypatch.setattr(module, "import_genai_types", lambda: FakeTypes)

        module._make_gemini_client(request_timeout_ms=12345)

        assert captured["api_key"] == "test-key"
        assert captured["http_options"].timeout == 12345

    def test_verify_namespace_counts_detects_mismatch(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _verify_namespace_counts

        class FakeIndex:
            def describe_index_stats(self):
                return {
                    "namespaces": {
                        "chunks": {"vector_count": 10},
                        "assertions": {"vector_count": 2},
                    }
                }

        report = _verify_namespace_counts(
            index=FakeIndex(),
            expected={"chunks": 10, "assertions": 3},
            min_count_only=False,
        )

        assert report["failures"] == [{"namespace": "assertions", "expected": 3, "actual": 2}]

    def test_embed_text_batch_sends_each_text_as_separate_content(self, monkeypatch):
        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        captured = {}

        class FakePart:
            @staticmethod
            def from_text(*, text):
                return ("text", text)

        class FakeContent:
            def __init__(self, *, role, parts):
                self.role = role
                self.parts = parts

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeModels:
            def embed_content(self, *, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return SimpleNamespace(
                    embeddings=[
                        SimpleNamespace(values=[float(index), float(index + 1)])
                        for index, _content in enumerate(contents)
                    ]
                )

        fake_types = SimpleNamespace(
            EmbedContentConfig=FakeConfig,
            Content=FakeContent,
            Part=FakePart,
        )
        monkeypatch.setattr(mod, "import_genai_types", lambda: fake_types)

        vectors = mod._embed_text_batch(
            SimpleNamespace(models=FakeModels()),
            model="gemini-embedding-2-preview",
            texts=["first", "second"],
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=1024,
        )

        assert vectors == [[0.0, 1.0], [1.0, 2.0]]
        assert captured["model"] == "gemini-embedding-2-preview"
        assert captured["config"].kwargs == {
            "task_type": "RETRIEVAL_DOCUMENT",
            "output_dimensionality": 1024,
        }
        assert [content.role for content in captured["contents"]] == ["user", "user"]
        assert [content.parts[0][1] for content in captured["contents"]] == ["first", "second"]

    def test_embed_text_batch_uses_prompt_instruction_for_gemini_embedding_2(self, monkeypatch):
        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        captured = {}

        class FakePart:
            @staticmethod
            def from_text(*, text):
                return ("text", text)

        class FakeContent:
            def __init__(self, *, role, parts):
                self.role = role
                self.parts = parts

        class FakeConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeModels:
            def embed_content(self, *, model, contents, config):
                captured["model"] = model
                captured["contents"] = contents
                captured["config"] = config
                return SimpleNamespace(embeddings=[SimpleNamespace(values=[1.0, 2.0])])

        fake_types = SimpleNamespace(
            EmbedContentConfig=FakeConfig,
            Content=FakeContent,
            Part=FakePart,
        )
        monkeypatch.setattr(mod, "import_genai_types", lambda: fake_types)

        vectors = mod._embed_text_batch(
            SimpleNamespace(models=FakeModels()),
            model="gemini-embedding-2",
            texts=["MBZUAI admissions"],
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=1024,
        )

        assert vectors == [[1.0, 2.0]]
        assert captured["config"].kwargs == {"output_dimensionality": 1024}
        assert captured["contents"][0].parts[0][1] == (
            "title: none | text: MBZUAI admissions"
        )

    def test_execute_uploads_chunk_parent_media_namespaces(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

        chunk_file = tmp_dir / "chunk_dense_records.json"
        parent_file = tmp_dir / "parent_dense_records.json"
        media_file = tmp_dir / "media_dense_records.json"
        bundle_file = tmp_dir / "retrieval_bundle.json"

        atomic_write_json(chunk_file, [{"id": "chunk1", "dense_text": "Chunk text", "document_id": "doc1"}])
        atomic_write_json(parent_file, [{"id": "parent1", "dense_text": "Parent text", "document_id": "doc1", "parent_type": "section"}])
        atomic_write_json(media_file, [{"id": "media1", "text": "Image caption", "media_type": "image", "can_embed_multimodal": False}])
        atomic_write_json(bundle_file, {"chunk_records": [], "parent_records": [], "media_records": []})

        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("PINECONE_API_KEY", "test")

        fake_calls = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                fake_calls.append((namespace, len(vectors)))

            def delete(self, *, delete_all, namespace):
                fake_calls.append(("delete", namespace, delete_all))

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def Index(self, name):
                return FakeIndex()

        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(mod, "_embed_text_batch", lambda *args, texts=None, **kwargs: [[0.1, 0.2] for _ in texts])
        monkeypatch.setattr(
            mod,
            "_embed_media_records",
            lambda *args, records=None, **kwargs: ([[0.3, 0.4] for _ in records], {"media_multimodal_records": 0, "media_text_only_records": len(records), "media_multimodal_fallbacks": 0}),
        )
        monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone, ServerlessSpec=lambda **kwargs: kwargs))

        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={
                "embedder": {
                    "model": "gemini-embedding-2-preview",
                    "pinecone_index": "test-index",
                    "output_dimensionality": 2,
                    "namespace_chunks": "chunks",
                    "namespace_parents": "parents",
                    "namespace_media": "media",
                    "enable_sparse": False,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "chunk_embedding_file": str(chunk_file),
                "parent_embedding_file": str(parent_file),
                "media_embedding_file": str(media_file),
                "retrieval_bundle_file": str(bundle_file),
            },
            stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
            stage_id="upload_retrieval",
        )

        result = run_async(GeminiPineconeEmbedder().execute(ctx))
        assert result.status.value == "completed"
        assert result.metrics["vectors_uploaded"] == 3
        assert ("chunks", 1) in fake_calls
        assert ("parents", 1) in fake_calls
        assert ("media", 1) in fake_calls

    def test_execute_resumes_from_progress_file(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

        chunk_file = tmp_dir / "chunk_dense_records.json"
        bundle_file = tmp_dir / "retrieval_bundle.json"

        atomic_write_json(
            chunk_file,
            [
                {"id": "chunk1", "dense_text": "Chunk text 1", "document_id": "doc1"},
                {"id": "chunk2", "dense_text": "Chunk text 2", "document_id": "doc2"},
            ],
        )
        atomic_write_json(bundle_file, {"chunk_records": [], "parent_records": [], "media_records": []})

        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("PINECONE_API_KEY", "test")

        fake_calls = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                fake_calls.append((namespace, [item["id"] for item in vectors]))

            def delete(self, *, delete_all, namespace):
                fake_calls.append(("delete", namespace, delete_all))

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def Index(self, name):
                return FakeIndex()

        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(mod, "_embed_text_batch", lambda *args, texts=None, **kwargs: [[0.1, 0.2] for _ in texts])
        monkeypatch.setattr(mod, "_ensure_sparse_index", lambda *args, **kwargs: True)
        monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone, ServerlessSpec=lambda **kwargs: kwargs))

        stage_dir = tmp_dir / "stage_outputs" / "upload_retrieval"
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            stage_dir / "index_upload_progress.json",
            {
                "index_name": "test-index",
                "model": "gemini-embedding-2-preview",
                "output_dimensionality": 2,
                "phase": "uploading_chunks",
                "totals": {
                    "chunks": 2,
                    "parents": 0,
                    "media": 0,
                    "facts": 0,
                    "sparse_chunks": 0,
                    "sparse_parents": 0,
                    "sparse_media": 0,
                    "sparse_facts": 0,
                },
                "uploaded": {
                    "chunks": 1,
                    "parents": 0,
                    "media": 0,
                    "facts": 0,
                    "sparse_chunks": 0,
                    "sparse_parents": 0,
                    "sparse_media": 0,
                    "sparse_facts": 0,
                },
                "media_metrics": {
                    "media_multimodal_records": 0,
                    "media_text_only_records": 0,
                    "media_multimodal_fallbacks": 0,
                },
            },
        )

        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={
                "embedder": {
                    "model": "gemini-embedding-2-preview",
                    "pinecone_index": "test-index",
                    "output_dimensionality": 2,
                    "namespace_chunks": "chunks",
                    "namespace_parents": "parents",
                    "namespace_media": "media",
                    "enable_sparse": False,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "chunk_embedding_file": str(chunk_file),
                "retrieval_bundle_file": str(bundle_file),
            },
            stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
            stage_id="upload_retrieval",
        )

        result = run_async(GeminiPineconeEmbedder().execute(ctx))
        assert result.status.value == "completed"
        assert result.metrics["chunk_vectors_uploaded"] == 2
        assert fake_calls == [("chunks", ["chunk2"])]

    def test_build_sparse_records_trims_payload_to_pinecone_safe_size(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _build_sparse_records

        records, stats = _build_sparse_records(
            [
                {
                    "id": "parent-1",
                    "sparse_text": "A" * 80000,
                    "lexical_text": "should not be used",
                }
            ],
            sparse_text_field="chunk_text",
            max_text_chars=50000,
            max_record_bytes=24000,
        )

        assert len(records) == 1
        payload = json.dumps(records[0], ensure_ascii=False).encode("utf-8")
        assert len(payload) <= 24000
        assert records[0]["chunk_text"].startswith("A")
        assert stats["trimmed"] >= 1
        assert stats["skipped"] == 0

    def test_build_sparse_records_skips_text_that_would_embed_to_empty_sparse_vector(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _build_sparse_records

        records, stats = _build_sparse_records(
            [
                {"id": "fact-1", "sparse_text": "the and of to in"},
                {"id": "fact-2", "sparse_text": "Masdar City Abu Dhabi"},
            ],
            sparse_text_field="chunk_text",
            max_text_chars=1000,
            max_record_bytes=24000,
        )

        assert [record["_id"] for record in records] == ["fact-2"]
        assert stats["skipped"] == 1

    def test_clear_namespace_ignores_missing_namespace_404(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _clear_namespace

        class MissingNamespaceIndex:
            def delete(self, *, delete_all, namespace):
                raise RuntimeError('(404) Namespace not found')

        _clear_namespace(MissingNamespaceIndex(), namespace="facts")

    def test_upsert_namespace_passes_request_timeout_when_supported(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _upsert_namespace

        captured = {}

        class FakeIndex:
            def upsert(self, *, vectors, namespace, _request_timeout=None):
                captured["namespace"] = namespace
                captured["count"] = len(vectors)
                captured["timeout"] = _request_timeout

        uploaded = _upsert_namespace(
            FakeIndex(),
            namespace="chunks",
            records=[{"id": "chunk1", "document_id": "doc1"}],
            vectors=[[0.1, 0.2]],
            kind="chunk",
            batch_size=10,
            request_timeout=(10.0, 120.0),
        )

        assert uploaded == 1
        assert captured == {"namespace": "chunks", "count": 1, "timeout": (10.0, 120.0)}

    def test_upsert_sparse_namespace_passes_request_timeout_when_supported(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _upsert_sparse_namespace

        captured = {}

        class FakeVectorApi:
            def upsert_records_namespace(self, **kwargs):
                captured.update(kwargs)

        class FakeIndex:
            _vector_api = FakeVectorApi()

        uploaded = _upsert_sparse_namespace(
            FakeIndex(),
            namespace="facts",
            records=[{"_id": "fact1", "chunk_text": "Masdar City Abu Dhabi"}],
            batch_size=10,
            request_timeout=(10.0, 120.0),
        )

        assert uploaded == 1
        assert captured["namespace"] == "facts"
        assert captured["_request_timeout"] == (10.0, 120.0)

    def test_load_progress_state_preserves_completed_namespaces_when_totals_expand(self, tmp_dir):
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.gemini_pinecone_embedder import _load_progress_state

        progress_path = tmp_dir / "index_upload_progress.json"
        atomic_write_json(
            progress_path,
            {
                "index_name": "test-index",
                "model": "gemini-embedding-2-preview",
                "output_dimensionality": 1536,
                "totals": {
                    "chunks": 10,
                    "parents": 5,
                    "media": 3,
                    "facts": 0,
                    "sparse_chunks": 10,
                    "sparse_parents": 5,
                    "sparse_media": 3,
                    "sparse_facts": 0,
                },
                "uploaded": {
                    "chunks": 10,
                    "parents": 5,
                    "media": 3,
                    "facts": 0,
                    "sparse_chunks": 10,
                    "sparse_parents": 5,
                    "sparse_media": 3,
                    "sparse_facts": 0,
                },
                "media_metrics": {
                    "media_multimodal_records": 3,
                    "media_text_only_records": 0,
                    "media_multimodal_fallbacks": 0,
                },
            },
        )

        uploaded, media_metrics = _load_progress_state(
            progress_path,
            index_name="test-index",
            model="gemini-embedding-2-preview",
            output_dimensionality=1536,
            totals={
                "chunks": 10,
                "parents": 5,
                "media": 3,
                "facts": 7,
                "sparse_chunks": 10,
                "sparse_parents": 5,
                "sparse_media": 3,
                "sparse_facts": 20,
            },
        )

        assert uploaded["chunks"] == 10
        assert uploaded["parents"] == 5
        assert uploaded["media"] == 3
        assert uploaded["facts"] == 0
        assert uploaded["sparse_facts"] == 0
        assert media_metrics["media_multimodal_records"] == 3

    def test_execute_resets_and_reuploads_when_bundle_fingerprint_changes(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

        chunk_file = tmp_dir / "chunk_dense_records.json"
        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(chunk_file, [{"id": "chunk1", "dense_text": "Chunk text", "lexical_text": "Chunk text"}])
        atomic_write_json(bundle_file, {"chunk_records": [{"id": "chunk1"}], "parent_records": [], "media_records": [], "fact_records": []})

        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("PINECONE_API_KEY", "test")

        calls = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                calls.append(("dense_upsert", namespace, len(vectors)))

            def upsert_records(self, *, namespace, records):
                calls.append(("sparse_upsert", namespace, len(records)))

            def delete(self, *, delete_all, namespace):
                calls.append(("delete", namespace, delete_all))

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def Index(self, name):
                return FakeIndex()

        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(mod, "_embed_text_batch", lambda *args, texts=None, **kwargs: [[0.1, 0.2] for _ in texts])
        monkeypatch.setattr(mod, "_ensure_sparse_index", lambda *args, **kwargs: True)
        monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone, ServerlessSpec=lambda **kwargs: kwargs))

        stage_dir = tmp_dir / "stage_outputs" / "upload_retrieval"
        stage_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            stage_dir / "index_upload_progress.json",
            {
                "index_name": "test-index",
                "model": "gemini-embedding-2-preview",
                "output_dimensionality": 2,
                "phase": "completed",
                "totals": {
                    "chunks": 1,
                    "parents": 0,
                    "media": 0,
                    "facts": 0,
                    "sparse_chunks": 1,
                    "sparse_parents": 0,
                    "sparse_media": 0,
                    "sparse_facts": 0,
                },
                "uploaded": {
                    "chunks": 1,
                    "parents": 0,
                    "media": 0,
                    "facts": 0,
                    "sparse_chunks": 1,
                    "sparse_parents": 0,
                    "sparse_media": 0,
                    "sparse_facts": 0,
                },
                "retrieval_bundle_sha256": "stale-sha",
                "media_metrics": {},
            },
        )

        ctx = StageContext(
            run_id="r-reset",
            project_name="p-reset",
            config={
                "embedder": {
                    "model": "gemini-embedding-2-preview",
                    "pinecone_index": "test-index",
                    "pinecone_sparse_index": "test-index-sparse",
                    "output_dimensionality": 2,
                    "namespace_chunks": "chunks",
                    "namespace_parents": "parents",
                    "namespace_media": "media",
                    "namespace_facts": "facts",
                    "enable_sparse": True,
                    "enable_dense_facts": False,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "chunk_embedding_file": str(chunk_file),
                "retrieval_bundle_file": str(bundle_file),
            },
            stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
            stage_id="upload_retrieval",
        )

        result = run_async(GeminiPineconeEmbedder().execute(ctx))
        assert result.status.value == "completed"
        assert ("delete", "chunks", True) in calls
        assert ("sparse_upsert", "chunks", 1) in calls
        manifest = load_json_safe(stage_dir / "index_upload_manifest.json")
        assert manifest["retrieval_bundle_sha256"]

    def test_audit_run_flags_retrieval_index_bundle_drift(self, tmp_dir):
        from pipeline.core.io import atomic_write_json
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, save_state

        stage_format = tmp_dir / "stage_outputs" / "format_retrieval"
        stage_upload = tmp_dir / "stage_outputs" / "upload_retrieval"
        stage_format.mkdir(parents=True)
        stage_upload.mkdir(parents=True)
        bundle_path = stage_format / "retrieval_bundle.json"
        atomic_write_json(bundle_path, {"chunk_records": [{"id": "c1"}], "parent_records": [], "media_records": [], "fact_records": []})
        atomic_write_json(
            stage_upload / "index_upload_manifest.json",
            {
                "index_name": "idx",
                "retrieval_bundle_file": str(bundle_path),
                "retrieval_bundle_sha256": "stale-sha",
            },
        )
        save_state(PipelineState(run_id="r1", project_name="p1", status="completed"), tmp_dir)

        report = audit_run(tmp_dir)
        assert any(issue.code == "index_manifest_bundle_mismatch" for issue in report.errors)

    def test_audit_run_uses_manifest_declared_finalized_bundle(self, tmp_dir):
        from pipeline.core.io import atomic_write_json, sha256_file
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, save_state

        stage_finalize = tmp_dir / "stage_outputs" / "finalize_retrieval_bundle"
        stage_build = tmp_dir / "stage_outputs" / "build_retrieval_bundle"
        stage_upload = tmp_dir / "stage_outputs" / "upload_retrieval"
        stage_finalize.mkdir(parents=True)
        stage_build.mkdir(parents=True)
        stage_upload.mkdir(parents=True)
        finalized_bundle = stage_finalize / "retrieval_bundle.json"
        stale_build_bundle = stage_build / "retrieval_bundle.json"
        atomic_write_json(finalized_bundle, {"bundle_version": 5, "chunk_records": [{"id": "final"}]})
        atomic_write_json(stale_build_bundle, {"bundle_version": 4, "chunk_records": [{"id": "stale"}]})
        atomic_write_json(
            stage_upload / "index_upload_manifest.json",
            {
                "index_name": "idx",
                "retrieval_bundle_file": str(finalized_bundle),
                "retrieval_bundle_sha256": sha256_file(finalized_bundle),
            },
        )
        save_state(PipelineState(run_id="r1", project_name="p1", status="completed"), tmp_dir)

        report = audit_run(tmp_dir)
        assert not any(issue.code == "index_manifest_bundle_mismatch" for issue in report.errors)

    def test_execute_clears_sparse_namespace_before_zero_progress_reupload(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

        chunk_file = tmp_dir / "chunk_dense_records.json"
        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(chunk_file, [{"id": "chunk1", "dense_text": "Admissions in Abu Dhabi", "lexical_text": "Admissions in Abu Dhabi"}])
        atomic_write_json(bundle_file, {"chunk_records": [], "parent_records": [], "media_records": []})

        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("PINECONE_API_KEY", "test")

        calls = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                calls.append(("dense_upsert", namespace, len(vectors)))

            def upsert_records(self, *, namespace, records):
                calls.append(("sparse_upsert", namespace, len(records)))

            def delete(self, *, delete_all, namespace):
                calls.append(("delete", namespace, delete_all))

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def Index(self, name):
                return FakeIndex()

        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(mod, "_embed_text_batch", lambda *args, texts=None, **kwargs: [[0.1, 0.2] for _ in texts])
        monkeypatch.setattr(mod, "_ensure_sparse_index", lambda *args, **kwargs: True)
        monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone, ServerlessSpec=lambda **kwargs: kwargs))

        ctx = StageContext(
            run_id="r-sparse",
            project_name="p-sparse",
            config={
                "embedder": {
                    "model": "gemini-embedding-2-preview",
                    "pinecone_index": "test-index",
                    "pinecone_sparse_index": "test-index-sparse",
                    "output_dimensionality": 2,
                    "namespace_chunks": "chunks",
                    "namespace_parents": "parents",
                    "namespace_media": "media",
                    "namespace_facts": "facts",
                    "enable_sparse": True,
                    "enable_dense_facts": False,
                    "clear_sparse_namespace_on_zero_progress": True,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "chunk_embedding_file": str(chunk_file),
                "retrieval_bundle_file": str(bundle_file),
            },
            stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
            stage_id="upload_retrieval",
        )

        result = run_async(GeminiPineconeEmbedder().execute(ctx))
        assert result.status.value == "completed"
        assert ("delete", "chunks", True) in calls
        assert ("sparse_upsert", "chunks", 1) in calls

    def test_execute_clears_all_sparse_namespaces_upfront_on_zero_progress_rebuild(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

        chunk_file = tmp_dir / "chunk_dense_records.json"
        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(chunk_file, [{"id": "chunk1", "dense_text": "Admissions in Abu Dhabi", "lexical_text": "Admissions in Abu Dhabi"}])
        atomic_write_json(bundle_file, {"chunk_records": [], "parent_records": [], "media_records": [], "fact_records": []})

        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("PINECONE_API_KEY", "test")

        calls = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                calls.append(("dense_upsert", namespace, len(vectors)))

            def upsert_records(self, *, namespace, records):
                calls.append(("sparse_upsert", namespace, len(records)))

            def delete(self, *, delete_all, namespace):
                calls.append(("delete", namespace, delete_all))

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def Index(self, name):
                return FakeIndex()

        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(mod, "_embed_text_batch", lambda *args, texts=None, **kwargs: [[0.1, 0.2] for _ in texts])
        monkeypatch.setattr(mod, "_ensure_sparse_index", lambda *args, **kwargs: True)
        monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone, ServerlessSpec=lambda **kwargs: kwargs))

        ctx = StageContext(
            run_id="r-sparse-upfront",
            project_name="p-sparse-upfront",
            config={
                "embedder": {
                    "model": "gemini-embedding-2-preview",
                    "pinecone_index": "test-index",
                    "pinecone_sparse_index": "test-index-sparse",
                    "output_dimensionality": 2,
                    "namespace_chunks": "chunks",
                    "namespace_parents": "parents",
                    "namespace_media": "media",
                    "namespace_facts": "facts",
                    "enable_sparse": True,
                    "enable_dense_facts": False,
                    "clear_sparse_namespace_on_zero_progress": True,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "chunk_embedding_file": str(chunk_file),
                "retrieval_bundle_file": str(bundle_file),
            },
            stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
            stage_id="upload_retrieval",
        )

        result = run_async(GeminiPineconeEmbedder().execute(ctx))
        assert result.status.value == "completed"
        for namespace in ("chunks", "parents", "media", "facts"):
            assert ("delete", namespace, True) in calls

    def test_execute_clears_dense_namespaces_before_zero_progress_rebuild(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

        chunk_file = tmp_dir / "chunk_dense_records.json"
        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(chunk_file, [{"id": "chunk1", "dense_text": "Admissions in Abu Dhabi"}])
        atomic_write_json(bundle_file, {"chunk_records": [{"id": "chunk1"}], "parent_records": [], "media_records": [], "fact_records": []})

        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("PINECONE_API_KEY", "test")

        calls = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                calls.append(("dense_upsert", namespace, len(vectors)))

            def delete(self, *, delete_all, namespace):
                calls.append(("delete", namespace, delete_all))

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def Index(self, name):
                return FakeIndex()

        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(mod, "_embed_text_batch", lambda *args, texts=None, **kwargs: [[0.1, 0.2] for _ in texts])
        monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone, ServerlessSpec=lambda **kwargs: kwargs))

        ctx = StageContext(
            run_id="r-dense",
            project_name="p-dense",
            config={
                "embedder": {
                    "model": "gemini-embedding-2-preview",
                    "pinecone_index": "test-index",
                    "output_dimensionality": 2,
                    "namespace_chunks": "chunks",
                    "namespace_parents": "parents",
                    "namespace_media": "media",
                    "namespace_facts": "facts",
                    "enable_sparse": False,
                    "clear_dense_namespace_on_zero_progress": True,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "chunk_embedding_file": str(chunk_file),
                "retrieval_bundle_file": str(bundle_file),
            },
            stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
            stage_id="upload_retrieval",
        )

        result = run_async(GeminiPineconeEmbedder().execute(ctx))
        assert result.status.value == "completed"
        assert ("delete", "chunks", True) in calls
        assert ("dense_upsert", "chunks", 1) in calls


class TestAdaptiveHybridRetriever:
    def _build_retrieval_run(self, tmp_dir):
        from pipeline.core.io import atomic_write_json

        run_dir = tmp_dir / "run"
        stage_dir = run_dir / "stage_outputs" / "format_retrieval"
        stage_dir.mkdir(parents=True)
        bundle = {
            "chunk_records": [
                {
                    "id": "chunk1",
                    "dense_text": "Admissions requirement details",
                    "text": "Admissions requirement details",
                    "document_title": "Admissions",
                    "document_id": "doc1",
                    "source_url": "https://example.com/admissions",
                    "source_markdown_path": "/tmp/admissions.md",
                    "section_key": "section-a",
                    "page_key": "page-a",
                    "neighbor_ids": ["chunk2"],
                    "media_ids": ["media1"],
                },
                {
                    "id": "chunk2",
                    "dense_text": "Application deadlines and fees",
                    "text": "Application deadlines and fees",
                    "document_title": "Admissions",
                    "document_id": "doc1",
                    "source_url": "https://example.com/admissions",
                    "source_markdown_path": "/tmp/admissions.md",
                    "section_key": "section-a",
                    "page_key": "page-a",
                    "neighbor_ids": ["chunk1", "chunk3"],
                    "media_ids": [],
                },
                {
                    "id": "chunk3",
                    "dense_text": "Faculty research and labs",
                    "text": "Faculty research and labs",
                    "document_title": "Research",
                    "document_id": "doc2",
                    "source_url": "https://example.com/research",
                    "source_markdown_path": "/tmp/research.md",
                    "section_key": "section-b",
                    "page_key": "page-b",
                    "neighbor_ids": ["chunk2"],
                    "media_ids": [],
                },
            ],
            "parent_records": [
                {
                    "id": "section-a",
                    "parent_type": "section",
                    "child_chunk_ids": ["chunk1", "chunk2"],
                },
                {
                    "id": "page-a",
                    "parent_type": "page",
                    "child_chunk_ids": ["chunk1", "chunk2"],
                },
            ],
            "media_records": [
                {
                    "id": "media1",
                    "media_type": "image",
                    "text": "Admissions chart with application requirements",
                    "linked_chunk_ids": ["chunk1"],
                    "linked_parent_ids": ["section-a", "page-a"],
                    "title": "Admissions chart",
                    "url": "https://example.com/chart.png",
                }
            ],
            "fact_records": [
                {
                    "id": "fact1",
                    "text": "Admissions requirements include transcripts and recommendation letters.",
                    "linked_chunk_ids": ["chunk1"],
                    "linked_parent_ids": ["section-a", "page-a"],
                }
            ],
        }
        lexical = [
            {"id": "chunk1", "record_type": "chunk", "text": "admissions requirement details", "tokens": ["admissions", "requirement", "details"]},
            {"id": "chunk2", "record_type": "chunk", "text": "application deadlines fees", "tokens": ["application", "deadlines", "fees"]},
            {"id": "section-a", "record_type": "parent", "text": "admissions requirement details application deadlines fees", "tokens": ["admissions", "requirement", "details", "application", "deadlines", "fees"]},
            {"id": "fact1", "record_type": "fact", "text": "Admissions requirements include transcripts and recommendation letters.", "tokens": ["admissions", "requirements", "include", "transcripts", "recommendation", "letters"]},
        ]
        atomic_write_json(stage_dir / "retrieval_bundle.json", bundle)
        atomic_write_json(stage_dir / "lexical_corpus.json", lexical)
        return run_dir

    def test_loads_current_mbzuai_retrieval_bundle_stage_id(self, tmp_dir):
        from pipeline.core.io import atomic_write_json
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = tmp_dir / "current_run"
        stage_dir = run_dir / "stage_outputs" / "build_retrieval_bundle"
        stage_dir.mkdir(parents=True)
        atomic_write_json(
            stage_dir / "retrieval_bundle.json",
            {
                "chunk_records": [
                    {
                        "id": "chunk-current",
                        "dense_text": "MBZUAI admission requirements",
                        "text": "MBZUAI admission requirements",
                        "document_title": "Admission Requirements",
                        "document_id": "doc-current",
                        "source_url": "https://mbzuai.ac.ae/study/admission-requirements",
                    }
                ],
                "parent_records": [],
                "media_records": [],
                "fact_records": [],
            },
        )
        atomic_write_json(
            stage_dir / "lexical_corpus.json",
            [
                {
                    "id": "chunk-current",
                    "record_type": "chunk",
                    "text": "MBZUAI admission requirements",
                    "tokens": ["mbzuai", "admission", "requirements"],
                }
            ],
        )

        retriever = AdaptiveHybridRetriever(
            config={"embedder": {"pinecone_index": "test-index"}, "retrieval": {}},
            work_dir=run_dir,
        )

        assert "chunk-current" in retriever.chunk_map

    def test_runtime_bridges_raw_chunk_hierarchy_keys_to_canonical_parent_ids(self, tmp_dir):
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        stage_dir = run_dir / "stage_outputs" / "format_retrieval"
        bundle_path = stage_dir / "retrieval_bundle.json"
        lexical_path = stage_dir / "lexical_corpus.json"
        bundle = load_json_safe(bundle_path, {})
        canonical_section_id = "parent:c650:document-revision:doc1:section:abc123"
        canonical_page_id = "parent:c650:document-revision:doc1:page"
        for parent in bundle["parent_records"]:
            if parent["id"] == "section-a":
                parent["id"] = canonical_section_id
            elif parent["id"] == "page-a":
                parent["id"] = canonical_page_id
        lexical = load_json_safe(lexical_path, [])
        for record in lexical:
            if record.get("id") == "section-a":
                record["id"] = canonical_section_id
        atomic_write_json(bundle_path, bundle)
        atomic_write_json(lexical_path, lexical)

        retriever = AdaptiveHybridRetriever(
            config={
                "embedder": {"pinecone_index": "idx"},
                "retrieval": {"enable_sparse": False, "enable_rerank": False},
            },
            work_dir=run_dir,
        )

        assert retriever._parent_ids_for_chunk("chunk1", parent_type="section") == [
            canonical_section_id
        ]
        assert retriever._parent_ids_for_chunk("chunk1", parent_type="page") == [
            canonical_page_id
        ]
        assert retriever._canonical_parent_ids(
            ["section-a", "page-a"],
            chunk_ids=["chunk1"],
        ) == [canonical_section_id, canonical_page_id]
        assert retriever._select_parent_ids(["chunk1"])[0:2] == [
            canonical_section_id,
            canonical_page_id,
        ]

    def _build_contact_lookup_run(self, tmp_dir):
        from pipeline.core.io import atomic_write_json

        run_dir = tmp_dir / "contact_run"
        stage_dir = run_dir / "stage_outputs" / "format_retrieval"
        stage_dir.mkdir(parents=True)
        bundle = {
            "chunk_records": [
                {
                    "id": "contact-chunk",
                    "dense_text": "Admissions contact directory. Admission admission@mbzuai.ac.ae Registrar registrar@mbzuai.ac.ae",
                    "text": "Admissions contact directory. Admission admission@mbzuai.ac.ae Registrar registrar@mbzuai.ac.ae",
                    "document_title": "Admissions contacts",
                    "document_id": "doc-contact",
                    "source_url": "https://example.com/contacts",
                    "source_markdown_path": "/tmp/contacts.md",
                    "section_key": "contact-section",
                    "page_key": "contact-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "policy-chunk",
                    "dense_text": "The MBZUAI Admission Committee reviews applications and can approve or reject transfer requests.",
                    "text": "The MBZUAI Admission Committee reviews applications and can approve or reject transfer requests.",
                    "document_title": "Admissions policy",
                    "document_id": "doc-policy",
                    "source_url": "https://example.com/policy",
                    "source_markdown_path": "/tmp/policy.md",
                    "section_key": "policy-section",
                    "page_key": "policy-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "ug-contact-chunk",
                    "dense_text": "Undergraduate admissions contact. Please send updated documents to ug.admission@mbzuai.ac.ae.",
                    "text": "Undergraduate admissions contact. Please send updated documents to ug.admission@mbzuai.ac.ae.",
                    "document_title": "Undergraduate admissions",
                    "document_id": "doc-ug",
                    "source_url": "https://example.com/undergraduate",
                    "source_markdown_path": "/tmp/undergraduate.md",
                    "section_key": "ug-section",
                    "page_key": "ug-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
            ],
            "parent_records": [
                {"id": "contact-section", "parent_type": "section", "child_chunk_ids": ["contact-chunk"]},
                {"id": "contact-page", "parent_type": "page", "child_chunk_ids": ["contact-chunk"]},
                {"id": "policy-section", "parent_type": "section", "child_chunk_ids": ["policy-chunk"]},
                {"id": "policy-page", "parent_type": "page", "child_chunk_ids": ["policy-chunk"]},
                {"id": "ug-section", "parent_type": "section", "child_chunk_ids": ["ug-contact-chunk"]},
                {"id": "ug-page", "parent_type": "page", "child_chunk_ids": ["ug-contact-chunk"]},
            ],
            "media_records": [],
            "fact_records": [
                {
                    "id": "fact-contact-general",
                    "text": "Admission-related questions may be sent to admission@mbzuai.ac.ae.",
                    "linked_chunk_ids": ["contact-chunk"],
                    "linked_parent_ids": ["contact-section", "contact-page"],
                },
                {
                    "id": "fact-contact-policy",
                    "text": "The MBZUAI Admission Committee reviews applications and transfer requests.",
                    "linked_chunk_ids": ["policy-chunk"],
                    "linked_parent_ids": ["policy-section", "policy-page"],
                },
                {
                    "id": "fact-contact-ug",
                    "text": "You can send updated documents to ug.admission@mbzuai.ac.ae with your full name in the subject line.",
                    "linked_chunk_ids": ["ug-contact-chunk"],
                    "linked_parent_ids": ["ug-section", "ug-page"],
                },
            ],
        }
        lexical = [
            {"id": "contact-chunk", "record_type": "chunk", "text": "admissions contact directory admission admission@mbzuai.ac.ae registrar registrar@mbzuai.ac.ae"},
            {"id": "policy-chunk", "record_type": "chunk", "text": "admissions committee reviews applications transfer requests"},
            {"id": "ug-contact-chunk", "record_type": "chunk", "text": "undergraduate admissions contact ug.admission@mbzuai.ac.ae updated documents"},
            {"id": "contact-section", "record_type": "parent", "text": "admissions contact directory admission admission@mbzuai.ac.ae"},
            {"id": "policy-section", "record_type": "parent", "text": "admissions committee policy applications transfer requests"},
            {"id": "ug-section", "record_type": "parent", "text": "undergraduate admissions contact ug.admission@mbzuai.ac.ae updated documents"},
            {"id": "fact-contact-general", "record_type": "fact", "text": "Admission-related questions may be sent to admission@mbzuai.ac.ae."},
            {"id": "fact-contact-policy", "record_type": "fact", "text": "The MBZUAI Admission Committee reviews applications and transfer requests."},
            {"id": "fact-contact-ug", "record_type": "fact", "text": "You can send updated documents to ug.admission@mbzuai.ac.ae with your full name in the subject line."},
        ]
        atomic_write_json(stage_dir / "retrieval_bundle.json", bundle)
        atomic_write_json(stage_dir / "lexical_corpus.json", lexical)
        return run_dir

    def _build_hours_lookup_run(self, tmp_dir):
        from pipeline.core.io import atomic_write_json

        run_dir = tmp_dir / "hours_run"
        stage_dir = run_dir / "stage_outputs" / "format_retrieval"
        stage_dir.mkdir(parents=True)
        bundle = {
            "chunk_records": [
                {
                    "id": "official-hours-chunk",
                    "dense_text": "FAQ. Our official workings hours are from 8:00 a.m. - 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday.",
                    "text": "FAQ. Our official workings hours are from 8:00 a.m. - 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday.",
                    "document_title": "Admissions FAQ",
                    "document_id": "doc-hours",
                    "source_url": "https://example.com/faq",
                    "source_markdown_path": "/tmp/faq.md",
                    "section_key": "faq-section",
                    "page_key": "faq-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "event-hours-chunk",
                    "dense_text": "Conference program. Registration opens at 8:00 AM, keynote at 9:00 AM, lunch at 12:00 PM, and the workshop continues until 6:00 PM.",
                    "text": "Conference program. Registration opens at 8:00 AM, keynote at 9:00 AM, lunch at 12:00 PM, and the workshop continues until 6:00 PM.",
                    "document_title": "Conference booklet",
                    "document_id": "doc-event",
                    "source_url": "https://example.com/booklet",
                    "source_markdown_path": "/tmp/booklet.md",
                    "section_key": "event-section",
                    "page_key": "event-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "support-hours-chunk",
                    "dense_text": "The MBZUAI IT team may be emailed for technical support. Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays.",
                    "text": "The MBZUAI IT team may be emailed for technical support. Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays.",
                    "document_title": "Screening exam instructions",
                    "document_id": "doc-support",
                    "source_url": "https://example.com/support",
                    "source_markdown_path": "/tmp/support.md",
                    "section_key": "support-section",
                    "page_key": "support-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
            ],
            "parent_records": [
                {"id": "faq-section", "parent_type": "section", "child_chunk_ids": ["official-hours-chunk"]},
                {"id": "faq-page", "parent_type": "page", "child_chunk_ids": ["official-hours-chunk"]},
                {"id": "event-section", "parent_type": "section", "child_chunk_ids": ["event-hours-chunk"]},
                {"id": "event-page", "parent_type": "page", "child_chunk_ids": ["event-hours-chunk"]},
                {"id": "support-section", "parent_type": "section", "child_chunk_ids": ["support-hours-chunk"]},
                {"id": "support-page", "parent_type": "page", "child_chunk_ids": ["support-hours-chunk"]},
            ],
            "media_records": [],
            "fact_records": [
                {
                    "id": "official-hours-fact",
                    "text": "Our official workings hours are from 8:00 a.m. - 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday.",
                    "linked_chunk_ids": ["official-hours-chunk"],
                    "linked_parent_ids": ["faq-section", "faq-page"],
                },
                {
                    "id": "event-hours-fact",
                    "text": "Registration opens at 8:00 AM and the workshop continues until 6:00 PM.",
                    "linked_chunk_ids": ["event-hours-chunk"],
                    "linked_parent_ids": ["event-section", "event-page"],
                },
                {
                    "id": "support-hours-fact",
                    "text": "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays.",
                    "linked_chunk_ids": ["support-hours-chunk"],
                    "linked_parent_ids": ["support-section", "support-page"],
                },
            ],
        }
        lexical = [
            {"id": "official-hours-chunk", "record_type": "chunk", "text": "official working hours monday thursday friday faq office hours"},
            {"id": "event-hours-chunk", "record_type": "chunk", "text": "conference program schedule registration keynote workshop lunch"},
            {"id": "support-hours-chunk", "record_type": "chunk", "text": "it support technical support working hours monday thursday friday screening exam"},
            {"id": "faq-section", "record_type": "parent", "text": "official working hours faq monday thursday friday"},
            {"id": "event-section", "record_type": "parent", "text": "conference schedule program registration keynote workshop"},
            {"id": "support-section", "record_type": "parent", "text": "it support technical support working hours screening exam"},
            {"id": "official-hours-fact", "record_type": "fact", "text": "Our official workings hours are from 8:00 a.m. - 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday."},
            {"id": "event-hours-fact", "record_type": "fact", "text": "Registration opens at 8:00 AM and the workshop continues until 6:00 PM."},
            {"id": "support-hours-fact", "record_type": "fact", "text": "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays."},
        ]
        atomic_write_json(stage_dir / "retrieval_bundle.json", bundle)
        atomic_write_json(stage_dir / "lexical_corpus.json", lexical)
        return run_dir

    def _build_named_relation_run(self, tmp_dir):
        from pipeline.core.io import atomic_write_json

        run_dir = tmp_dir / "named_relation_run"
        stage_dir = run_dir / "stage_outputs" / "format_retrieval"
        stage_dir.mkdir(parents=True)
        bundle = {
            "chunk_records": [
                {
                    "id": "mbzuai-name-chunk",
                    "dense_text": "The university is named after His Highness Sheikh Mohamed bin Zayed Al Nahyan, the President of the UAE.",
                    "text": "The university is named after His Highness Sheikh Mohamed bin Zayed Al Nahyan, the President of the UAE.",
                    "document_title": "MBZUAI FAQ",
                    "document_id": "doc-faq",
                    "source_url": "https://example.com/faq",
                    "source_markdown_path": "/tmp/faq.md",
                    "section_key": "faq-section",
                    "page_key": "faq-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "vicuna-name-chunk",
                    "dense_text": "Vicuna is named after a relative of the llama and was developed using Meta AI's LLaMA 2 model with MBZUAI contributions.",
                    "text": "Vicuna is named after a relative of the llama and was developed using Meta AI's LLaMA 2 model with MBZUAI contributions.",
                    "document_title": "MBZUAI magazine",
                    "document_id": "doc-mag",
                    "source_url": "https://example.com/magazine",
                    "source_markdown_path": "/tmp/magazine.md",
                    "section_key": "mag-section",
                    "page_key": "mag-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
            ],
            "parent_records": [
                {"id": "faq-section", "parent_type": "section", "child_chunk_ids": ["mbzuai-name-chunk"]},
                {"id": "faq-page", "parent_type": "page", "child_chunk_ids": ["mbzuai-name-chunk"]},
                {"id": "mag-section", "parent_type": "section", "child_chunk_ids": ["vicuna-name-chunk"]},
                {"id": "mag-page", "parent_type": "page", "child_chunk_ids": ["vicuna-name-chunk"]},
            ],
            "media_records": [],
            "fact_records": [
                {
                    "id": "mbzuai-name-fact",
                    "text": "The university is named after His Highness Sheikh Mohamed bin Zayed Al Nahyan, the President of the UAE.",
                    "linked_chunk_ids": ["mbzuai-name-chunk"],
                    "linked_parent_ids": ["faq-section", "faq-page"],
                },
                {
                    "id": "vicuna-name-fact",
                    "text": "Vicuna is named after a relative of the llama.",
                    "linked_chunk_ids": ["vicuna-name-chunk"],
                    "linked_parent_ids": ["mag-section", "mag-page"],
                },
            ],
        }
        lexical = [
            {"id": "mbzuai-name-chunk", "record_type": "chunk", "text": "mbzuai university named after sheikh mohamed bin zayed al nahyan"},
            {"id": "vicuna-name-chunk", "record_type": "chunk", "text": "vicuna named after relative llama mbzuai contributions"},
            {"id": "faq-section", "record_type": "parent", "text": "mbzuai faq named after sheikh mohamed bin zayed al nahyan"},
            {"id": "mag-section", "record_type": "parent", "text": "vicuna named after llama mbzuai contributions"},
            {"id": "mbzuai-name-fact", "record_type": "fact", "text": "The university is named after His Highness Sheikh Mohamed bin Zayed Al Nahyan, the President of the UAE."},
            {"id": "vicuna-name-fact", "record_type": "fact", "text": "Vicuna is named after a relative of the llama."},
        ]
        atomic_write_json(stage_dir / "retrieval_bundle.json", bundle)
        atomic_write_json(stage_dir / "lexical_corpus.json", lexical)
        return run_dir

    def _build_leadership_run(self, tmp_dir):
        from pipeline.core.io import atomic_write_json

        run_dir = tmp_dir / "leadership_run"
        stage_dir = run_dir / "stage_outputs" / "format_retrieval"
        stage_dir.mkdir(parents=True)
        bundle = {
            "chunk_records": [
                {
                    "id": "president-chunk",
                    "dense_text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                    "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                    "document_title": "Office of the President",
                    "document_id": "doc-president",
                    "source_url": "https://example.com/about/office-of-the-president",
                    "source_markdown_path": "/tmp/president.md",
                    "section_key": "leadership-section",
                    "page_key": "leadership-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "provost-chunk",
                    "dense_text": "Timothy Baldwin is the Provost of MBZUAI.",
                    "text": "Timothy Baldwin is the Provost of MBZUAI.",
                    "document_title": "Office of the Provost",
                    "document_id": "doc-provost",
                    "source_url": "https://example.com/office-of-the-provost",
                    "source_markdown_path": "/tmp/provost.md",
                    "section_key": "leadership-section",
                    "page_key": "leadership-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "associate-provost-chunk",
                    "dense_text": "Professor Timothy Baldwin Associate Provost for Academic and Student Affairs.",
                    "text": "Professor Timothy Baldwin Associate Provost for Academic and Student Affairs.",
                    "document_title": "NLP dream team",
                    "document_id": "doc-associate-provost",
                    "source_url": "https://example.com/news/nlp-dream-team",
                    "source_markdown_path": "/tmp/associate-provost.md",
                    "section_key": "leadership-section",
                    "page_key": "leadership-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "board-chair-current-chunk",
                    "dense_text": "Speaker His Excellency Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees.",
                    "text": "Speaker His Excellency Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees.",
                    "document_title": "MBZUAI leadership",
                    "document_id": "doc-chair-current",
                    "source_url": "https://example.com/about/leadership",
                    "source_markdown_path": "/tmp/leadership.md",
                    "section_key": "board-section",
                    "page_key": "board-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "board-chair-old-chunk",
                    "dense_text": "His Excellency Dr. Sultan Ahmed Al Jaber Chairman of the MBZUAI Board of Trustees.",
                    "text": "His Excellency Dr. Sultan Ahmed Al Jaber Chairman of the MBZUAI Board of Trustees.",
                    "document_title": "University Catalogue 2022",
                    "document_id": "doc-chair-old",
                    "source_url": "https://example.com/catalogue-2022",
                    "source_markdown_path": "/tmp/catalogue.md",
                    "section_key": "catalogue-section",
                    "page_key": "catalogue-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "chief-of-staff-chunk",
                    "dense_text": "Hong (Dekyi) Liang Vice President and Chief of Staff.",
                    "text": "Hong (Dekyi) Liang Vice President and Chief of Staff.",
                    "document_title": "MBZUAI leadership",
                    "document_id": "doc-chief-of-staff",
                    "source_url": "https://example.com/about/leadership",
                    "source_markdown_path": "/tmp/leadership.md",
                    "section_key": "leadership-section",
                    "page_key": "leadership-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
            ],
            "parent_records": [
                {"id": "leadership-section", "parent_type": "section", "child_chunk_ids": ["president-chunk", "provost-chunk", "associate-provost-chunk", "chief-of-staff-chunk"]},
                {"id": "leadership-page", "parent_type": "page", "child_chunk_ids": ["president-chunk", "provost-chunk", "associate-provost-chunk", "chief-of-staff-chunk"]},
                {"id": "board-section", "parent_type": "section", "child_chunk_ids": ["board-chair-current-chunk"]},
                {"id": "board-page", "parent_type": "page", "child_chunk_ids": ["board-chair-current-chunk"]},
                {"id": "catalogue-section", "parent_type": "section", "child_chunk_ids": ["board-chair-old-chunk"]},
                {"id": "catalogue-page", "parent_type": "page", "child_chunk_ids": ["board-chair-old-chunk"]},
            ],
            "media_records": [],
            "fact_records": [
                {
                    "id": "president-fact",
                    "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                    "linked_chunk_ids": ["president-chunk"],
                    "linked_parent_ids": ["leadership-section", "leadership-page"],
                    "document_title": "Office of the President",
                    "source_url": "https://example.com/about/office-of-the-president",
                },
                {
                    "id": "provost-fact",
                    "text": "Timothy Baldwin is the Provost of MBZUAI.",
                    "linked_chunk_ids": ["provost-chunk"],
                    "linked_parent_ids": ["leadership-section", "leadership-page"],
                    "document_title": "Office of the Provost",
                    "source_url": "https://example.com/office-of-the-provost",
                },
                {
                    "id": "associate-provost-fact",
                    "text": "Professor Timothy Baldwin Associate Provost for Academic and Student Affairs.",
                    "linked_chunk_ids": ["associate-provost-chunk"],
                    "linked_parent_ids": ["leadership-section", "leadership-page"],
                    "document_title": "NLP dream team",
                    "source_url": "https://example.com/news/nlp-dream-team",
                },
                {
                    "id": "board-chair-current-fact",
                    "text": "His Excellency Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees.",
                    "linked_chunk_ids": ["board-chair-current-chunk"],
                    "linked_parent_ids": ["board-section", "board-page"],
                    "document_title": "MBZUAI leadership",
                    "source_url": "https://example.com/about/leadership",
                },
                {
                    "id": "board-chair-old-fact",
                    "text": "His Excellency Dr. Sultan Ahmed Al Jaber Chairman of the MBZUAI Board of Trustees.",
                    "linked_chunk_ids": ["board-chair-old-chunk"],
                    "linked_parent_ids": ["catalogue-section", "catalogue-page"],
                    "document_title": "University Catalogue 2022",
                    "source_url": "https://example.com/catalogue-2022",
                },
                {
                    "id": "chief-of-staff-fact",
                    "text": "Hong (Dekyi) Liang Vice President and Chief of Staff.",
                    "linked_chunk_ids": ["chief-of-staff-chunk"],
                    "linked_parent_ids": ["leadership-section", "leadership-page"],
                    "document_title": "MBZUAI leadership",
                    "source_url": "https://example.com/about/leadership",
                },
            ],
        }
        lexical = [
            {"id": "president-chunk", "record_type": "chunk", "text": "eric xing president mbzuai office of the president leadership"},
            {"id": "provost-chunk", "record_type": "chunk", "text": "timothy baldwin provost mbzuai office of the provost leadership"},
            {"id": "associate-provost-chunk", "record_type": "chunk", "text": "timothy baldwin associate provost academic student affairs"},
            {"id": "board-chair-current-chunk", "record_type": "chunk", "text": "khaldoon khalifa al mubarak chairman mbzuai board of trustees leadership"},
            {"id": "board-chair-old-chunk", "record_type": "chunk", "text": "sultan ahmed al jaber chairman mbzuai board of trustees catalogue"},
            {"id": "chief-of-staff-chunk", "record_type": "chunk", "text": "hong dekyi liang vice president chief of staff mbzuai leadership"},
            {"id": "leadership-section", "record_type": "parent", "text": "mbzuai leadership president provost vice president chief of staff"},
            {"id": "board-section", "record_type": "parent", "text": "mbzuai board of trustees chairman khaldoon khalifa al mubarak"},
            {"id": "catalogue-section", "record_type": "parent", "text": "catalogue chairman mbzuai board of trustees sultan ahmed al jaber"},
            {"id": "president-fact", "record_type": "fact", "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence."},
            {"id": "provost-fact", "record_type": "fact", "text": "Timothy Baldwin is the Provost of MBZUAI."},
            {"id": "associate-provost-fact", "record_type": "fact", "text": "Professor Timothy Baldwin Associate Provost for Academic and Student Affairs."},
            {"id": "board-chair-current-fact", "record_type": "fact", "text": "His Excellency Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees."},
            {"id": "board-chair-old-fact", "record_type": "fact", "text": "His Excellency Dr. Sultan Ahmed Al Jaber Chairman of the MBZUAI Board of Trustees."},
            {"id": "chief-of-staff-fact", "record_type": "fact", "text": "Hong (Dekyi) Liang Vice President and Chief of Staff."},
        ]
        atomic_write_json(stage_dir / "retrieval_bundle.json", bundle)
        atomic_write_json(stage_dir / "lexical_corpus.json", lexical)
        return run_dir

    def _build_service_availability_run(self, tmp_dir):
        from pipeline.core.io import atomic_write_json

        run_dir = tmp_dir / "service_availability_run"
        stage_dir = run_dir / "stage_outputs" / "format_retrieval"
        stage_dir.mkdir(parents=True)
        bundle = {
            "chunk_records": [
                {
                    "id": "shuttle-chunk",
                    "dense_text": "Does MBZUAI provide a shuttle bus service? Yes, a shuttle service connects students to key locations, including shopping malls and other amenities.",
                    "text": "Does MBZUAI provide a shuttle bus service? Yes, a shuttle service connects students to key locations, including shopping malls and other amenities.",
                    "document_title": "MBZUAI FAQ",
                    "document_id": "doc-faq",
                    "source_url": "https://example.com/faq",
                    "source_markdown_path": "/tmp/faq.md",
                    "section_key": "faq-section",
                    "page_key": "faq-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "news-chunk",
                    "dense_text": "MBZUAI students were placed in internships as part of an internship programme.",
                    "text": "MBZUAI students were placed in internships as part of an internship programme.",
                    "document_title": "MBZUAI news",
                    "document_id": "doc-news",
                    "source_url": "https://example.com/news/internships",
                    "source_markdown_path": "/tmp/news.md",
                    "section_key": "news-section",
                    "page_key": "news-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "housing-chunk",
                    "dense_text": "MBZUAI provides student accommodation on a single occupancy basis with separate male and female student quarters.",
                    "text": "MBZUAI provides student accommodation on a single occupancy basis with separate male and female student quarters.",
                    "document_title": "MBZUAI FAQ",
                    "document_id": "doc-faq",
                    "source_url": "https://example.com/faq",
                    "source_markdown_path": "/tmp/faq.md",
                    "section_key": "housing-section",
                    "page_key": "housing-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
                {
                    "id": "parents-chunk",
                    "dense_text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. However, nearby hotels and Airbnbs are available for visiting parents.",
                    "text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. However, nearby hotels and Airbnbs are available for visiting parents.",
                    "document_title": "MBZUAI FAQ",
                    "document_id": "doc-faq",
                    "source_url": "https://example.com/faq",
                    "source_markdown_path": "/tmp/faq.md",
                    "section_key": "parents-section",
                    "page_key": "parents-page",
                    "neighbor_ids": [],
                    "media_ids": [],
                },
            ],
            "parent_records": [
                {"id": "faq-section", "parent_type": "section", "child_chunk_ids": ["shuttle-chunk"]},
                {"id": "faq-page", "parent_type": "page", "child_chunk_ids": ["shuttle-chunk"]},
                {"id": "news-section", "parent_type": "section", "child_chunk_ids": ["news-chunk"]},
                {"id": "news-page", "parent_type": "page", "child_chunk_ids": ["news-chunk"]},
                {"id": "housing-section", "parent_type": "section", "child_chunk_ids": ["housing-chunk"]},
                {"id": "housing-page", "parent_type": "page", "child_chunk_ids": ["housing-chunk"]},
                {"id": "parents-section", "parent_type": "section", "child_chunk_ids": ["parents-chunk"]},
                {"id": "parents-page", "parent_type": "page", "child_chunk_ids": ["parents-chunk"]},
            ],
            "media_records": [],
            "fact_records": [
                {
                    "id": "shuttle-fact",
                    "text": "Does MBZUAI provide a shuttle bus service? Yes, a shuttle service connects students to key locations, including shopping malls and other amenities.",
                    "linked_chunk_ids": ["shuttle-chunk"],
                    "linked_parent_ids": ["faq-section", "faq-page"],
                    "document_title": "MBZUAI FAQ",
                    "source_url": "https://example.com/faq",
                },
                {
                    "id": "news-fact",
                    "text": "MBZUAI students were placed in internships as part of an internship programme.",
                    "linked_chunk_ids": ["news-chunk"],
                    "linked_parent_ids": ["news-section", "news-page"],
                    "document_title": "MBZUAI news",
                    "source_url": "https://example.com/news/internships",
                },
                {
                    "id": "housing-fact",
                    "text": "MBZUAI provides student accommodation on a single occupancy basis with separate male and female student quarters.",
                    "linked_chunk_ids": ["housing-chunk"],
                    "linked_parent_ids": ["housing-section", "housing-page"],
                    "document_title": "MBZUAI FAQ",
                    "source_url": "https://example.com/faq",
                },
                {
                    "id": "parents-fact",
                    "text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. However, nearby hotels and Airbnbs are available for visiting parents.",
                    "linked_chunk_ids": ["parents-chunk"],
                    "linked_parent_ids": ["parents-section", "parents-page"],
                    "document_title": "MBZUAI FAQ",
                    "source_url": "https://example.com/faq",
                },
            ],
        }
        lexical = [
            {"id": "shuttle-chunk", "record_type": "chunk", "text": "mbzuai shuttle bus service students amenities"},
            {"id": "news-chunk", "record_type": "chunk", "text": "mbzuai students internships programme"},
            {"id": "housing-chunk", "record_type": "chunk", "text": "mbzuai student accommodation housing quarters"},
            {"id": "parents-chunk", "record_type": "chunk", "text": "parents stay campus housing parents hotels airbnbs"},
            {"id": "faq-section", "record_type": "parent", "text": "mbzuai faq shuttle bus service"},
            {"id": "news-section", "record_type": "parent", "text": "mbzuai news students internships"},
            {"id": "housing-section", "record_type": "parent", "text": "mbzuai faq student accommodation housing"},
            {"id": "parents-section", "record_type": "parent", "text": "mbzuai faq parents stay housing parents hotels"},
            {"id": "shuttle-fact", "record_type": "fact", "text": "Does MBZUAI provide a shuttle bus service? Yes, a shuttle service connects students to key locations, including shopping malls and other amenities."},
            {"id": "news-fact", "record_type": "fact", "text": "MBZUAI students were placed in internships as part of an internship programme."},
            {"id": "housing-fact", "record_type": "fact", "text": "MBZUAI provides student accommodation on a single occupancy basis with separate male and female student quarters."},
            {"id": "parents-fact", "record_type": "fact", "text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. However, nearby hotels and Airbnbs are available for visiting parents."},
        ]
        atomic_write_json(stage_dir / "retrieval_bundle.json", bundle)
        atomic_write_json(stage_dir / "lexical_corpus.json", lexical)
        return run_dir

    def test_fact_mode_uses_chunk_plus_neighbor_window(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 1, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["chunk1"] if kwargs["namespace"] == "chunks" else [])
        monkeypatch.setattr(
            retriever,
            "_sparse_query_ids",
            lambda **kwargs: ["chunk1"] if kwargs["namespace"] == "chunks" else ["fact1"] if kwargs["namespace"] == "facts" else [],
        )

        result = retriever.retrieve("Where are the admissions requirements?")
        assert result["mode"] == "fact"
        assert result["selected_chunk_ids"] == ["chunk1", "chunk2"]

    def test_fact_mode_prioritizes_fact_hits_in_candidate_selection(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 1, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else ["fact1"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What are the admissions requirements?")
        assert result["selected_chunk_ids"][0] == "chunk1"

    def test_fact_mode_uses_local_fact_candidates_when_remote_fact_lanes_are_empty(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 1, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What are the admissions requirements?")
        assert result["selected_chunk_ids"][0] == "chunk1"
        assert "fact1" in result["local_fact_ids"]

    def test_tokenize_normalizes_simple_plural_variants_for_fact_queries(self):
        from pipeline.retrieval.adaptive_hybrid import _named_query_tokens, _tokenize

        assert "working" in _tokenize("workings hours")
        assert "hour" in _tokenize("hours")
        assert "mbzuai" in _named_query_tokens("What are MBZUAI's official working hours?")
        assert "mbzuais" not in _named_query_tokens("What are MBZUAI's official working hours?")

    def test_informative_query_tokens_strip_role_noise_for_contact_lookup(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        tokens = retriever._informative_query_tokens("What is the admissions committee email address?")

        assert "committee" not in tokens
        assert "email" in tokens
        assert "admission" in tokens or "admissions" in tokens

    def test_fact_mode_prefers_direct_contact_chunk_for_email_lookup(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 0, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["policy-chunk"] if kwargs["namespace"] == "chunks" else ["fact-contact-general"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What is the admissions committee email address?")

        assert result["selected_chunk_ids"][0] == "contact-chunk"

    def test_derive_answer_records_extracts_operational_hours_and_event_schedule(self):
        from pipeline.core.answer_records import derive_answer_records_from_bundle

        bundle = {
            "chunk_records": [
                {
                    "id": "official-hours-chunk",
                    "text": "Our official workings hours are from 8:00 a.m. - 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday.",
                    "document_title": "FAQ",
                },
                {
                    "id": "event-hours-chunk",
                    "text": "Conference program. Registration opens at 8:00 AM, keynote at 9:00 AM, and the workshop continues until 6:00 PM.",
                    "document_title": "Booklet",
                },
            ],
            "fact_records": [],
        }

        records = derive_answer_records_from_bundle(bundle)
        hours_records = [record for record in records if record.get("answer_type") == "hours"]

        assert any(record.get("answer_subtype") == "operational_hours" for record in hours_records)
        assert any(record.get("answer_subtype") == "event_schedule" for record in hours_records)

    def test_answer_lane_prefers_official_hours_record_over_event_schedule(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_hours_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "fact_neighbor_window": 0, "max_context_chunks": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        answer_ids = retriever._local_answer_query_ids("What are MBZUAI's official working hours?", top_k=4)
        assert answer_ids
        top_answer = retriever.answer_map[answer_ids[0]]
        assert top_answer["answer_type"] == "hours"
        assert top_answer["answer_subtype"] == "operational_hours"

    def test_answer_lane_prefers_official_hours_over_support_hours_for_weekday_query(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_hours_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "fact_neighbor_window": 0, "max_context_chunks": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        answer_ids = retriever._local_answer_query_ids("What are MBZUAI's weekday operating hours?", top_k=4)
        ranked = retriever._rank_answer_ids("What are MBZUAI's weekday operating hours?", answer_ids, top_k=4)

        assert ranked
        top_answer = retriever.answer_map[ranked[0]]
        assert top_answer["answer_subtype"] == "operational_hours"
        assert "official workings hours" in top_answer["text"].lower()

    def test_named_after_answer_scoring_prefers_mbzuai_subject_over_other_named_entities(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_named_relation_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        answer_ids = retriever._local_answer_query_ids("Who is MBZUAI named after?", top_k=4)
        ranked = retriever._rank_answer_ids("Who is MBZUAI named after?", answer_ids, top_k=4)

        assert ranked
        top_answer = retriever.answer_map[ranked[0]]
        assert "sheikh mohamed bin zayed al nahyan" in top_answer["value"].lower()
        assert "vicuna" not in top_answer["text"].lower()

    def test_derive_answer_records_extract_role_holder_records(self):
        from pipeline.core.answer_records import derive_answer_records_from_bundle

        bundle = {
            "chunk_records": [
                {
                    "id": "leadership-chunk",
                    "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence. Timothy Baldwin is the Provost of MBZUAI.",
                    "document_title": "MBZUAI leadership",
                    "source_url": "https://example.com/about/leadership",
                },
                {
                    "id": "board-chunk",
                    "text": "Speaker His Excellency Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees.",
                    "document_title": "MBZUAI leadership",
                    "source_url": "https://example.com/about/leadership",
                },
            ],
            "fact_records": [],
        }

        records = derive_answer_records_from_bundle(bundle)
        role_records = [record for record in records if record.get("answer_type") == "role_holder"]

        assert role_records
        assert any(record.get("answer_subtype") == "president" and "eric xing" in str(record.get("value") or "").lower() for record in role_records)
        assert any(record.get("answer_subtype") == "provost" and "timothy baldwin" in str(record.get("value") or "").lower() for record in role_records)
        assert any(record.get("answer_subtype") == "board_chair" and "khaldoon" in str(record.get("value") or "").lower() for record in role_records)

    def test_derive_answer_records_does_not_promote_associate_provost_to_provost(self):
        from pipeline.core.answer_records import derive_answer_records_from_bundle

        bundle = {
            "chunk_records": [
                {
                    "id": "associate-provost-chunk",
                    "text": "Professor Timothy Baldwin Associate Provost for Academic and Student Affairs.",
                    "document_title": "NLP dream team",
                    "source_url": "https://example.com/news/nlp-dream-team",
                }
            ],
            "fact_records": [],
        }

        records = derive_answer_records_from_bundle(bundle)
        role_records = [record for record in records if record.get("answer_type") == "role_holder"]

        assert not any(record.get("answer_subtype") == "provost" for record in role_records)

    def test_derive_answer_records_rejects_organization_holder_for_president_role(self):
        from pipeline.core.answer_records import derive_answer_records_from_bundle

        bundle = {
            "chunk_records": [
                {
                    "id": "org-president-chunk",
                    "text": "MBZUAI President, Professor Eric Xing, addressed the delegation alongside US-UAE Business Council President Danny Sebright.",
                    "document_title": "Partnership news",
                    "source_url": "https://example.com/news/partnerships",
                }
            ],
            "fact_records": [],
        }

        records = derive_answer_records_from_bundle(bundle)
        president_values = [
            str(record.get("value") or "").lower()
            for record in records
            if record.get("answer_type") == "role_holder" and record.get("answer_subtype") == "president"
        ]

        assert "us-uae business council" not in president_values
        assert any("eric xing" in value for value in president_values)

    def test_derive_answer_records_extracts_heading_style_role_holder_records(self):
        from pipeline.core.answer_records import derive_answer_records_from_bundle

        bundle = {
            "chunk_records": [
                {
                    "id": "heading-role-chunk",
                    "text": "### Hong (Dekyi) Liang\n\n## Vice President and Chief of Staff\n\nHong (Dekyi) Liang advises the President of Mohamed bin Zayed University of Artificial Intelligence.",
                    "document_title": "Leadership",
                    "source_url": "https://mbzuai.ac.ae/about/leadership/hong-dekyi-liang",
                }
            ],
            "fact_records": [],
        }

        records = derive_answer_records_from_bundle(bundle)
        role_records = [
            record
            for record in records
            if record.get("answer_type") == "role_holder" and record.get("answer_subtype") == "vice_president_chief_of_staff"
        ]
        president_records = [
            record
            for record in records
            if record.get("answer_type") == "role_holder" and record.get("answer_subtype") == "president"
        ]

        assert role_records
        assert any("hong (dekyi) liang" in str(record.get("value") or "").lower() for record in role_records)
        assert not president_records

    def test_answer_record_validity_rejects_truncated_legal_basis(self):
        from pipeline.core.answer_records import answer_record_looks_valid

        assert answer_record_looks_valid(
            {
                "answer_type": "legal_basis",
                "answer_subtype": "establishment_law",
                "subject_text": "MBZUAI",
                "value": "Law No. 25 of 2019",
            }
        )
        assert not answer_record_looks_valid(
            {
                "answer_type": "legal_basis",
                "answer_subtype": "policy_relation",
                "subject_text": "| MBZUAI",
                "value": "established under Law No",
            }
        )

    def test_answer_record_validity_rejects_clause_fragment_affiliation_subject(self):
        from pipeline.core.answer_records import answer_record_looks_valid

        assert answer_record_looks_valid(
            {
                "answer_type": "affiliation",
                "answer_subtype": "government_affiliation",
                "subject_text": "MBZUAI",
                "value": "Abu Dhabi Executive Council",
            }
        )
        assert not answer_record_looks_valid(
            {
                "answer_type": "affiliation",
                "answer_subtype": "entity_relation",
                "subject_text": "MBZUAI is under and",
                "value": "the Abu Dhabi Executive Council",
            }
        )

    def test_answer_lane_prefers_current_board_chair_for_board_query(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_leadership_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        answer_ids = retriever._local_answer_query_ids("Who chairs MBZUAI's Board of Trustees?", top_k=6)
        ranked = retriever._rank_answer_ids("Who chairs MBZUAI's Board of Trustees?", answer_ids, top_k=6)

        assert ranked
        top_answer = retriever.answer_map[ranked[0]]
        assert top_answer["answer_type"] == "role_holder"
        assert top_answer["answer_subtype"] == "board_chair"
        assert "khaldoon khalifa al mubarak" in str(top_answer.get("value") or "").lower()

    def test_answer_lane_prefers_current_provost_over_catalogue_and_associate_role(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_leadership_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        answer_ids = retriever._local_answer_query_ids("Who is the provost of MBZUAI?", top_k=8)
        ranked = retriever._rank_answer_ids("Who is the provost of MBZUAI?", answer_ids, top_k=8)

        assert ranked
        top_answer = retriever.answer_map[ranked[0]]
        assert top_answer["answer_subtype"] == "provost"
        assert "timothy baldwin" in str(top_answer.get("value") or "").lower()
        assert "associate provost" not in str(top_answer.get("text") or "").lower()

    def test_retrieve_selects_multiple_role_answers_for_multi_slot_query(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_leadership_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "fact_neighbor_window": 0, "max_context_chunks": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: (
                ["board-chair-old-chunk"]
                if kwargs["namespace"] == "chunks"
                else ["board-chair-old-fact"]
                if kwargs["namespace"] == "facts"
                else []
            ),
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("Who are the president and provost of MBZUAI?")
        answer_texts = [str(doc.get("text") or "").lower() for doc in result.get("answer_documents") or []]

        assert any("eric xing" in text for text in answer_texts)
        assert any("timothy baldwin" in text for text in answer_texts)

    def test_retrieve_answer_documents_include_structured_metadata(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_leadership_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "fact_neighbor_window": 0, "max_context_chunks": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("Who is the provost of MBZUAI?")

        assert result["answer_documents"]
        top_answer = result["answer_documents"][0]
        assert top_answer["answer_type"] == "role_holder"
        assert top_answer["answer_subtype"] == "provost"
        assert "timothy baldwin" in str(top_answer["value"]).lower()
        assert top_answer["linked_chunk_ids"]

    def test_requested_role_subtypes_do_not_double_count_vice_president_query(self):
        from pipeline.retrieval.adaptive_hybrid import _requested_role_subtypes

        roles = _requested_role_subtypes("Who is MBZUAI's Vice President and Chief of Staff?")

        assert roles == ("vice_president_chief_of_staff",)

    def test_answer_generation_respects_retrieval_abstention(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.answer_generation import generate_answer_predictions

        run_dir = self._build_retrieval_run(tmp_dir)
        dataset_path = tmp_dir / "eval.jsonl"
        output_path = tmp_dir / "predictions.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "fact-001",
                    "query": "What is MBZUAI's Singapore office phone number?",
                    "query_type": "fact",
                    "source_type": "none",
                    "no_answer": True,
                    "reference_answer": "No grounded answer should be returned.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            @classmethod
            def from_config(cls, *, config_name, work_dir):
                return cls()

            def retrieve(self, query):
                return {
                    "abstained": True,
                    "retrieval_documents": [],
                    "answer_documents": [],
                    "fact_documents": [],
                    "selected_chunk_ids": [],
                    "seed_chunk_ids": [],
                    "dense_parent_ids": [],
                    "media": [],
                }

        class FakeClient:
            class models:
                @staticmethod
                def generate_content(*args, **kwargs):
                    raise AssertionError("generator should not be called when retrieval abstains")

        import pipeline.evaluation.answer_generation as mod

        monkeypatch.setattr(mod, "AdaptiveHybridRetriever", FakeRetriever)
        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: FakeClient())

        result = generate_answer_predictions(
            config_name="test",
            work_dir=run_dir,
            dataset_path=dataset_path,
            output_path=output_path,
            model="gemini-2.5-flash",
        )

        assert result["row_count"] == 1
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["response"] == "Insufficient evidence."

    def test_answer_generation_composes_multi_slot_role_answer_without_model(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.answer_generation import generate_answer_predictions

        run_dir = self._build_leadership_run(tmp_dir)
        dataset_path = tmp_dir / "role_eval.jsonl"
        output_path = tmp_dir / "role_predictions.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "fact-role-001",
                    "query": "Who are the president and provost of MBZUAI?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "reference_answer": "Professor Eric Xing is the president and Timothy Baldwin is the provost of MBZUAI.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            def __init__(self):
                self.answer_map = {
                    "president-id": {
                        "id": "president-id",
                        "answer_type": "role_holder",
                        "answer_subtype": "president",
                        "value": "Professor Eric Xing",
                        "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                        "confidence": 0.94,
                        "subject_text": "MBZUAI",
                        "source_record_type": "fact",
                    },
                    "provost-id": {
                        "id": "provost-id",
                        "answer_type": "role_holder",
                        "answer_subtype": "provost",
                        "value": "Timothy Baldwin",
                        "text": "Timothy Baldwin is the Provost of MBZUAI.",
                        "confidence": 0.95,
                        "subject_text": "MBZUAI",
                        "source_record_type": "fact",
                    },
                }

            @classmethod
            def from_config(cls, *, config_name, work_dir):
                return cls()

            def retrieve(self, query):
                return {
                    "abstained": False,
                    "mode": "fact",
                    "retrieval_documents": [],
                    "answer_documents": [
                        {
                            "id": "president-id",
                            "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                            "value": "Professor Eric Xing",
                            "answer_type": "role_holder",
                            "answer_subtype": "president",
                            "subject_text": "MBZUAI",
                        },
                        {
                            "id": "provost-id",
                            "text": "Timothy Baldwin is the Provost of MBZUAI.",
                            "value": "Timothy Baldwin",
                            "answer_type": "role_holder",
                            "answer_subtype": "provost",
                            "subject_text": "MBZUAI",
                        },
                    ],
                    "fact_documents": [],
                    "selected_answer_ids": ["president-id", "provost-id"],
                    "selected_chunk_ids": [],
                    "seed_chunk_ids": [],
                    "dense_parent_ids": [],
                    "media": [],
                }

        class FakeClient:
            class models:
                @staticmethod
                def generate_content(*args, **kwargs):
                    raise AssertionError("generator should not be called when structured answers fully cover the query")

        import pipeline.evaluation.answer_generation as mod

        monkeypatch.setattr(mod, "AdaptiveHybridRetriever", FakeRetriever)
        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: FakeClient())

        result = generate_answer_predictions(
            config_name="test",
            work_dir=run_dir,
            dataset_path=dataset_path,
            output_path=output_path,
            model="gemini-2.5-flash",
        )

        assert result["row_count"] == 1
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["response"] == "The president of MBZUAI is Professor Eric Xing. The provost of MBZUAI is Timothy Baldwin."

    def test_answer_generation_composes_board_chair_answer_without_model(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.answer_generation import generate_answer_predictions

        run_dir = self._build_leadership_run(tmp_dir)
        dataset_path = tmp_dir / "board_role_eval.jsonl"
        output_path = tmp_dir / "board_role_predictions.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "fact-role-003",
                    "query": "Who chairs MBZUAI's Board of Trustees?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "reference_answer": "Khaldoon Khalifa Al Mubarak chairs MBZUAI's Board of Trustees.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            def __init__(self):
                self.answer_map = {
                    "board-chair-id": {
                        "id": "board-chair-id",
                        "answer_type": "role_holder",
                        "answer_subtype": "board_chair",
                        "value": "Khaldoon Khalifa Al Mubarak",
                        "text": "Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees.",
                        "confidence": 0.95,
                        "subject_text": "MBZUAI",
                        "source_record_type": "fact",
                    }
                }

            @classmethod
            def from_config(cls, *, config_name, work_dir):
                return cls()

            def retrieve(self, query):
                return {
                    "abstained": False,
                    "mode": "fact",
                    "retrieval_documents": [],
                    "answer_documents": [
                        {
                            "id": "board-chair-id",
                            "text": "Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees.",
                            "value": "Khaldoon Khalifa Al Mubarak",
                            "answer_type": "role_holder",
                            "answer_subtype": "board_chair",
                            "subject_text": "MBZUAI",
                        }
                    ],
                    "fact_documents": [],
                    "selected_answer_ids": ["board-chair-id"],
                    "selected_chunk_ids": [],
                    "seed_chunk_ids": [],
                    "dense_parent_ids": [],
                    "media": [],
                }

        class FakeClient:
            class models:
                @staticmethod
                def generate_content(*args, **kwargs):
                    raise AssertionError("generator should not be called when structured board-chair evidence covers the query")

        import pipeline.evaluation.answer_generation as mod

        monkeypatch.setattr(mod, "AdaptiveHybridRetriever", FakeRetriever)
        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: FakeClient())

        result = generate_answer_predictions(
            config_name="test",
            work_dir=run_dir,
            dataset_path=dataset_path,
            output_path=output_path,
            model="gemini-2.5-flash",
        )

        assert result["row_count"] == 1
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["response"] == "The chair of MBZUAI's Board of Trustees is Khaldoon Khalifa Al Mubarak."

    def test_answer_generation_returns_insufficient_evidence_for_structured_subject_mismatch(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.answer_generation import generate_answer_predictions

        run_dir = self._build_leadership_run(tmp_dir)
        dataset_path = tmp_dir / "role_noanswer_eval.jsonl"
        output_path = tmp_dir / "role_noanswer_predictions.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "fact-role-002",
                    "query": "Who is the president of MBZUAI's New York campus?",
                    "query_type": "fact",
                    "source_type": "none",
                    "no_answer": True,
                    "reference_answer": "Insufficient evidence.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            def __init__(self):
                self.answer_map = {
                    "president-id": {
                        "id": "president-id",
                        "answer_type": "role_holder",
                        "answer_subtype": "president",
                        "value": "Professor Eric Xing",
                        "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                        "confidence": 0.94,
                        "subject_text": "MBZUAI",
                        "source_record_type": "fact",
                    }
                }

            @classmethod
            def from_config(cls, *, config_name, work_dir):
                return cls()

            def retrieve(self, query):
                return {
                    "abstained": False,
                    "mode": "fact",
                    "retrieval_documents": [],
                    "answer_documents": [
                        {
                            "id": "president-id",
                            "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                            "value": "Professor Eric Xing",
                            "answer_type": "role_holder",
                            "answer_subtype": "president",
                            "subject_text": "MBZUAI",
                        }
                    ],
                    "fact_documents": [],
                    "selected_answer_ids": ["president-id"],
                    "selected_chunk_ids": [],
                    "seed_chunk_ids": [],
                    "dense_parent_ids": [],
                    "media": [],
                }

        class FakeClient:
            class models:
                @staticmethod
                def generate_content(*args, **kwargs):
                    raise AssertionError("generator should not be called when structured evidence mismatches the requested subject")

        import pipeline.evaluation.answer_generation as mod

        monkeypatch.setattr(mod, "AdaptiveHybridRetriever", FakeRetriever)
        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: FakeClient())

        result = generate_answer_predictions(
            config_name="test",
            work_dir=run_dir,
            dataset_path=dataset_path,
            output_path=output_path,
            model="gemini-2.5-flash",
        )

        assert result["row_count"] == 1
        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert rows[0]["response"] == "Insufficient evidence."

    def test_compose_structured_answer_prefers_general_admissions_email_for_generic_query(self):
        from pipeline.evaluation.answer_generation import _compose_structured_answer

        class FakeRetriever:
            def __init__(self):
                self.answer_map = {
                    "admission": {
                        "id": "admission",
                        "answer_type": "email",
                        "answer_subtype": "contact",
                        "value": "admission@mbzuai.ac.ae",
                        "text": "Admission-related questions may be sent to admission@mbzuai.ac.ae.",
                        "subject_text": "",
                    },
                    "aiug": {
                        "id": "aiug",
                        "answer_type": "email",
                        "answer_subtype": "contact",
                        "value": "ai.ug@mbzuai.ac.ae",
                        "text": "Contact AI undergraduate admissions at ai.ug@mbzuai.ac.ae.",
                        "subject_text": "",
                    },
                }

            def _score_answer_record(self, query, answer):
                return {
                    "admission": 6.0,
                    "aiug": 3.0,
                }[answer["id"]]

        result = _compose_structured_answer(
            query="What is the admissions email address?",
            retriever=FakeRetriever(),
            retrieval_result={
                "selected_answer_ids": ["aiug", "admission"],
                "answer_documents": [],
            },
        )

        assert result == "The email address is admission@mbzuai.ac.ae."

    def test_compose_structured_answer_prefers_undergraduate_admissions_mailbox(self):
        from pipeline.evaluation.answer_generation import _compose_structured_answer

        class FakeRetriever:
            def __init__(self):
                self.answer_map = {
                    "ug-admission": {
                        "id": "ug-admission",
                        "answer_type": "email",
                        "answer_subtype": "contact_point",
                        "value": "ug.admission@mbzuai.ac.ae",
                        "text": "For undergraduate admissions contact ug.admission@mbzuai.ac.ae.",
                        "subject_text": "",
                    },
                    "aiug": {
                        "id": "aiug",
                        "answer_type": "email",
                        "answer_subtype": "contact",
                        "value": "ai.ug@mbzuai.ac.ae",
                        "text": "Contact AI undergraduate admissions at ai.ug@mbzuai.ac.ae.",
                        "subject_text": "",
                    },
                }

            def _score_answer_record(self, query, answer):
                return {
                    "ug-admission": 6.6,
                    "aiug": 6.1,
                }[answer["id"]]

        result = _compose_structured_answer(
            query="What is the undergraduate admissions email address?",
            retriever=FakeRetriever(),
            retrieval_result={
                "selected_answer_ids": ["aiug", "ug-admission"],
                "answer_documents": [],
            },
        )

        assert result == "The email address is ug.admission@mbzuai.ac.ae."

    def test_compose_structured_answer_allows_support_hours_lookup_without_subject_text(self):
        from pipeline.evaluation.answer_generation import _compose_structured_answer

        class FakeRetriever:
            def __init__(self):
                self.answer_map = {
                    "support-hours": {
                        "id": "support-hours",
                        "answer_type": "hours",
                        "answer_subtype": "support_hours",
                        "value": "8:00 AM - 5:00 PM",
                        "text": "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays.",
                        "subject_text": "",
                    },
                    "official-hours": {
                        "id": "official-hours",
                        "answer_type": "hours",
                        "answer_subtype": "operational_hours",
                        "value": "8:00 a.m. – 6:00 p.m.",
                        "text": "Our official workings hours are from 8:00 a.m. – 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday.",
                        "subject_text": "",
                    },
                }

            def _score_answer_record(self, query, answer):
                return {
                    "support-hours": 5.4,
                    "official-hours": 2.1,
                }[answer["id"]]

        result = _compose_structured_answer(
            query="What are the IT support working hours for the MBZUAI online screening exam?",
            retriever=FakeRetriever(),
            retrieval_result={
                "selected_answer_ids": ["official-hours", "support-hours"],
                "answer_documents": [],
            },
        )

        assert result == (
            "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays."
        )

    def test_source_query_bonus_penalizes_news_for_named_relation_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_named_relation_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        faq_bonus = retriever._source_query_bonus(
            "Who is MBZUAI named after?",
            source_url="https://example.com/faq",
            document_title="MBZUAI FAQ",
            text="The university is named after His Highness Sheikh Mohamed bin Zayed Al Nahyan.",
        )
        news_bonus = retriever._source_query_bonus(
            "Who is MBZUAI named after?",
            source_url="https://example.com/news/jais",
            document_title="MBZUAI magazine",
            text="Named after the highest peak in the UAE, Jais Climate is a collaboration with MBZUAI.",
        )

        assert faq_bonus > news_bonus

    def test_source_query_bonus_prefers_campus_sources_for_amenities_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        campus_bonus = retriever._source_query_bonus(
            "What campus amenities and support facilities can students use on site?",
            source_url="https://example.com/student-resources/campus-facilities",
            document_title="Campus Facilities",
            text="Students have access to support services and dedicated student facilities including health services, dining facilities, and student lounges.",
        )
        application_bonus = retriever._source_query_bonus(
            "What campus amenities and support facilities can students use on site?",
            source_url="https://example.com/study/undergraduate-application-submission",
            document_title="Undergraduate application submission",
            text="Your application has been successfully submitted.",
        )

        assert campus_bonus > application_bonus

    def test_service_availability_answer_extraction_captures_transport_faqs(self):
        from pipeline.core.answer_records import derive_answer_records_from_bundle

        bundle = {
            "chunk_records": [],
            "parent_records": [],
            "media_records": [],
            "fact_records": [
                {
                    "id": "shuttle-fact",
                    "text": "Does MBZUAI provide a shuttle bus service? Yes, a shuttle service connects students to key locations, including shopping malls and other amenities.",
                    "linked_chunk_ids": ["chunk-a"],
                    "linked_parent_ids": ["page-a"],
                    "document_title": "MBZUAI FAQ",
                    "source_url": "https://example.com/faq",
                }
            ],
        }

        records = derive_answer_records_from_bundle(bundle)
        transport_records = [
            record
            for record in records
            if record.get("answer_type") == "service_availability" and record.get("answer_subtype") == "transport"
        ]

        assert transport_records
        assert any("shuttle service connects students" in str(record.get("text") or "").lower() for record in transport_records)

    def test_service_availability_answer_lane_prefers_transport_faq_over_news(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, _requested_service_availability_subtypes

        run_dir = self._build_service_availability_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        query = "Do MBZUAI students have shuttle transportation? bus service"
        assert _requested_service_availability_subtypes(query) == {"transport"}

        answer_ids = retriever._local_answer_query_ids(query, top_k=4)
        ranked = retriever._rank_answer_ids(query, answer_ids, top_k=4)

        assert ranked
        top_answer = retriever.answer_map[ranked[0]]
        assert top_answer["answer_type"] == "service_availability"
        assert top_answer["answer_subtype"] == "transport"
        assert "shuttle service connects students" in top_answer["text"].lower()

    def test_service_availability_answer_lane_prefers_student_housing_over_parent_stay(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_service_availability_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        answer_ids = retriever._local_answer_query_ids("Is student housing available at MBZUAI?", top_k=6)
        ranked = retriever._rank_answer_ids("Is student housing available at MBZUAI?", answer_ids, top_k=6)

        assert ranked
        top_answer = retriever.answer_map[ranked[0]]
        assert top_answer["answer_type"] == "service_availability"
        assert top_answer["answer_subtype"] == "accommodation"
        assert "student accommodation" in top_answer["text"].lower()

    def test_service_availability_scoring_rejects_mismatched_housing_for_transport_query(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_service_availability_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        housing_answer = {
            "id": "housing-answer",
            "answer_type": "service_availability",
            "answer_subtype": "housing",
            "text": "Student accommodation on a single occupancy basis is provided.",
            "value": "Student accommodation on a single occupancy basis is provided.",
            "subject_text": "MBZUAI students",
            "qualifiers": [],
            "confidence": 0.8,
        }
        transport_answer = {
            "id": "transport-answer",
            "answer_type": "service_availability",
            "answer_subtype": "shuttle_option",
            "text": "Yes, a shuttle service connects students to key locations.",
            "value": "Yes, a shuttle service connects students to key locations.",
            "subject_text": "MBZUAI students",
            "qualifiers": [],
            "confidence": 0.8,
        }

        query = "Is there a shuttle service for MBZUAI students?"
        assert retriever._score_answer_record(query, housing_answer) < 0.0
        assert retriever._score_answer_record(query, transport_answer) > 0.0

    def test_retrieve_surfaces_answer_documents_for_official_hours_lookup(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_hours_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "fact_neighbor_window": 0, "max_context_chunks": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["event-hours-chunk"] if kwargs["namespace"] == "chunks" else ["event-hours-fact"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What are MBZUAI's official working hours?")

        assert result["selected_chunk_ids"][0] == "official-hours-chunk"
        assert result["selected_answer_ids"]
        assert result["answer_documents"][0]["id"] == result["selected_answer_ids"][0]
        assert "official workings hours" in result["answer_documents"][0]["text"].lower()

    def test_fact_mode_prefers_undergraduate_contact_chunk_for_undergrad_email_lookup(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 0, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["contact-chunk"] if kwargs["namespace"] == "chunks" else ["fact-contact-ug"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What is the undergraduate admissions email address?")

        assert result["selected_chunk_ids"][0] == "ug-contact-chunk"

    def test_generic_undergraduate_contact_score_beats_submitted_application_variant(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 0, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        generic_answer = {
            "id": "generic-undergrad-email",
            "answer_type": "email",
            "answer_subtype": "contact",
            "text": "Undergraduate admissions contact. Please email ug.admission@mbzuai.ac.ae.",
            "value": "ug.admission@mbzuai.ac.ae",
            "subject_text": "Undergraduate admissions",
            "qualifiers": [],
            "confidence": 0.9,
        }
        scoped_answer = {
            "id": "submitted-application-email",
            "answer_type": "email",
            "answer_subtype": "contact_for_submitted_application",
            "text": "For questions about your submitted application, email ug.admission@mbzuai.ac.ae.",
            "value": "ug.admission@mbzuai.ac.ae",
            "subject_text": "Undergraduate admissions inquiries about submitted application",
            "qualifiers": ["questions about your submitted application"],
            "confidence": 0.9,
        }

        query = "What is the undergraduate admissions email address?"
        assert retriever._score_answer_record(query, generic_answer) > retriever._score_answer_record(query, scoped_answer)

    def test_fact_mode_prefers_general_contact_over_undergraduate_contact_for_generic_email_lookup(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 0, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: (
                ["ug-contact-chunk", "contact-chunk"]
                if kwargs["namespace"] == "chunks"
                else ["fact-contact-ug", "fact-contact-general"]
                if kwargs["namespace"] == "facts"
                else []
            ),
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What is the admissions email address?")

        assert result["selected_chunk_ids"][0] == "contact-chunk"

    def test_generic_contact_query_prefers_general_contact_chunk(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 0, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: (
                ["policy-chunk", "contact-chunk"]
                if kwargs["namespace"] == "chunks"
                else ["fact-contact-policy", "fact-contact-general"]
                if kwargs["namespace"] == "facts"
                else []
            ),
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("How can I contact admissions?")

        assert result["selected_chunk_ids"][0] == "contact-chunk"

    def test_exact_lookup_payload_surfaces_fact_documents_first(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"fact_neighbor_window": 0, "max_context_chunks": 3, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["contact-chunk"] if kwargs["namespace"] == "chunks" else ["fact-contact-general"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What is the admissions email address?")

        assert result["selected_fact_ids"][0] == "fact-contact-general"
        assert result["fact_documents"][0]["id"] == "fact-contact-general"
        assert result["selected_answer_ids"]
        assert result["answer_documents"][0]["id"] == result["selected_answer_ids"][0]
        assert result["retrieval_documents"][0]["id"] == result["selected_answer_ids"][0]

    def test_support_hours_scoring_beats_official_hours_for_support_query(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_contact_lookup_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.answer_context_texts_by_id["support-hours"] = (
            "Additional reminders - The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae for technical support. "
            "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays."
        )
        support_answer = {
            "id": "support-hours",
            "answer_type": "hours",
            "answer_subtype": "support_hours",
            "value": "8:00 AM - 5:00 PM",
            "text": "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays.",
            "document_title": "MBZUAI-Online-Screening-Exam-Instructions",
            "source_record_type": "chunk",
        }
        official_answer = {
            "id": "official-hours",
            "answer_type": "hours",
            "answer_subtype": "office_hours",
            "value": "8:00 a.m. – 6:00 p.m., Monday to Thursday; 7:30 a.m. – 12:00 p.m., Friday",
            "text": "The hours for MBZUAI are 8:00 a.m. – 6:00 p.m., Monday to Thursday; 7:30 a.m. – 12:00 p.m., Friday. Qualifiers: official working hours.",
            "subject_text": "MBZUAI",
            "document_title": "Official contact page",
            "source_record_type": "assertion",
        }

        support_score = retriever._score_answer_record(
            "What are the IT support working hours for the MBZUAI online screening exam?",
            support_answer,
        )
        official_score = retriever._score_answer_record(
            "What are the IT support working hours for the MBZUAI online screening exam?",
            official_answer,
        )

        assert support_score > official_score

    def test_board_chair_scoring_penalizes_founding_chair_evidence(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_leadership_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        current_answer = {
            "id": "current-chair",
            "answer_type": "role_holder",
            "answer_subtype": "board_chair",
            "value": "Khaldoon Khalifa Al Mubarak",
            "text": "Khaldoon Khalifa Al Mubarak Chairman of MBZUAI's Board of Trustees.",
            "subject_text": "MBZUAI",
            "document_title": "Leadership and Governance",
            "source_url": "https://mbzuai.ac.ae/about/leadership/",
            "source_record_type": "assertion",
        }
        founding_answer = {
            "id": "founding-chair",
            "answer_type": "role_holder",
            "answer_subtype": "board_chair",
            "value": "Dr. Sultan Ahmed Al Jaber",
            "text": "Dr. Sultan Ahmed Al Jaber ... Founding Chair of the MBZUAI Board of Trustees.",
            "subject_text": "MBZUAI",
            "document_title": "Class of 2024 Program",
            "source_record_type": "assertion",
        }

        current_score = retriever._score_answer_record("Who chairs MBZUAI's Board of Trustees?", current_answer)
        founding_score = retriever._score_answer_record("Who chairs MBZUAI's Board of Trustees?", founding_answer)

        assert current_score > founding_score

    def test_media_query_prefers_scoped_mode_for_visual_questions(self):
        from pipeline.retrieval.adaptive_hybrid import QueryMode, classify_query_mode

        assert classify_query_mode("Which facilities are labeled on the MBZUAI campus map?") == QueryMode.SCOPED
        assert classify_query_mode("Is parking provided at MBZUAI?") == QueryMode.FACT
        assert classify_query_mode("Where is parking permitted at the Masdar City campus?") == QueryMode.FACT
        assert classify_query_mode("What are the IT support working hours for the MBZUAI online screening exam?") == QueryMode.FACT
        assert classify_query_mode("Whose name does MBZUAI carry?") == QueryMode.FACT
        assert classify_query_mode("In which city is MBZUAI based?") == QueryMode.FACT
        assert classify_query_mode(
            "Which MBZUAI careers page section lists vacancies for the Computing and Mathematical Sciences Division?"
        ) == QueryMode.FACT
        assert classify_query_mode(
            "Who is the upcoming MBZUAI Nexus Speaker Series talk by Xiang Meng hosted by?"
        ) == QueryMode.FACT
        assert classify_query_mode(
            'Who is the speaker for the MBZUAI Nexus talk titled "Applying Image Analysis & AI to Cancer & Metabolic Syndrome"?'
        ) == QueryMode.FACT
        assert classify_query_mode("What does the Digital Twin Lab create?") == QueryMode.FACT
        assert classify_query_mode("أين سيُعقد GITEX GLOBAL 2025 المذكور في الصفحة؟") == QueryMode.FACT
        assert classify_query_mode(
            "ما الفكرة الأساسية التي يشرحها ملخص فعالية «Physical AI and the Intelligence of Things»؟"
        ) == QueryMode.FACT
        assert classify_query_mode("أي نموذج مذكور في لوحة مشاريع البحث؟") == QueryMode.SCOPED
        assert classify_query_mode(
            "ما عنوان البريد الإلكتروني الذي ينبغي التواصل معه بشأن متطلبات إمكانية الوصول قبل زيارة جامعة محمد بن زايد للذكاء الاصطناعي؟"
        ) == QueryMode.FACT
        assert classify_query_mode("What five core AI specializations does MBZUAI offer in its M.Sc. and Ph.D. programs?") == QueryMode.SCOPED
        assert classify_query_mode("What everyday campus amenities can MBZUAI students use on site?") == QueryMode.SCOPED
        assert classify_query_mode("Explain MBZUAI's legal basis and institutional affiliation.") == QueryMode.SCOPED
        assert classify_query_mode("Summarize the law that created MBZUAI and the authority it is affiliated with.") == QueryMode.SCOPED
        assert classify_query_mode("Prepare a visitor briefing covering location, parking, transport, facilities, and working hours.") == QueryMode.SYNTHESIS

    def test_on_site_phrase_is_not_treated_as_website_lookup(self):
        from pipeline.retrieval.adaptive_hybrid import _lookup_query_profile

        profile = _lookup_query_profile("What everyday campus amenities can MBZUAI students use on site?")
        assert profile.answer_types == tuple()
        assert profile.is_exact_lookup is False

    def test_website_reference_in_synthesis_is_not_treated_as_url_lookup(self):
        from pipeline.retrieval.adaptive_hybrid import QueryMode, _lookup_query_profile, classify_query_mode

        query = (
            "What does the IFM website say it aims to build with partners, where is its "
            "headquarters located, and in which cities does it have research centers?"
        )
        profile = _lookup_query_profile(query)

        assert "website" not in profile.answer_types
        assert profile.is_exact_lookup is False
        assert classify_query_mode(query) == QueryMode.SYNTHESIS

    def test_explicit_website_url_question_remains_an_exact_lookup(self):
        from pipeline.retrieval.adaptive_hybrid import _lookup_query_profile

        profile = _lookup_query_profile("What is the official website for MBZUAI?")

        assert profile.answer_types == ("website",)
        assert profile.is_exact_lookup is True

    def test_email_address_lookup_does_not_also_request_location(self):
        from pipeline.retrieval.adaptive_hybrid import _lookup_query_profile

        profile = _lookup_query_profile("What is the admissions email address?")
        assert profile.answer_types == ("email",)

    def test_generic_contact_query_defaults_to_email_lookup(self):
        from pipeline.retrieval.adaptive_hybrid import _lookup_query_profile

        profile = _lookup_query_profile("How can I contact admissions?")
        assert profile.answer_types == ("email",)
        assert profile.is_exact_lookup is True

    def test_named_query_tokens_include_titlecase_entities_beyond_sentence_start(self):
        from pipeline.retrieval.adaptive_hybrid import _named_query_tokens

        tokens = _named_query_tokens("What is MBZUAI's Singapore office phone number?")
        assert "mbzuai" in tokens
        assert "singapore" in tokens

    def test_fact_query_bonus_prefers_concise_location_fact(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        concise = "MBZUAI is based in Abu Dhabi, in a single campus located in Masdar City."
        broad = "MBZUAI is located in the Emirate of Abu Dhabi and is officially licensed from March 10, 2020, to February 4, 2025, by the Ministry of Education."

        assert retriever._fact_query_bonus("Where is MBZUAI located?", concise) > retriever._fact_query_bonus("Where is MBZUAI located?", broad)

    def test_fact_query_bonus_prefers_parent_housing_fact_for_parent_stay_query(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        parent_fact = "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. Nearby hotels or Airbnbs can be recommended."
        noisy_fact = "Like many of the students who arrive at the MBZUAI campus, I was eager, hopeful, and a little daunted."

        assert retriever._fact_query_bonus(
            "Can parents stay with students on the MBZUAI campus?",
            parent_fact,
        ) > retriever._fact_query_bonus(
            "Can parents stay with students on the MBZUAI campus?",
            noisy_fact,
        )

    def test_fact_query_bonus_prefers_official_hours_over_support_hours(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        official_fact = "Our official workings hours are from 8:00 a.m. - 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday."
        support_fact = "The MBZUAI IT team may be emailed for technical support. Working hours are at 8:00 AM - 5:00 PM on Mondays to Thursdays and 8:00 AM - 12:30 PM on Fridays."

        assert retriever._fact_query_bonus(
            "What are MBZUAI's official working hours?",
            official_fact,
        ) > retriever._fact_query_bonus(
            "What are MBZUAI's official working hours?",
            support_fact,
        )

    def test_informative_query_tokens_add_semantic_aliases_for_fact_paraphrases(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        carry_tokens = retriever._informative_query_tokens("Whose name does MBZUAI carry?")
        visitor_tokens = retriever._informative_query_tokens("Is parking available for students and visitors at MBZUAI?")

        assert "named" in carry_tokens
        assert "after" in carry_tokens
        assert "guest" in visitor_tokens or "guests" in visitor_tokens

    def test_select_parent_ids_prefers_top_chunk_parents(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "parent_candidate_top_k": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        parent_ids = retriever._select_parent_ids(
            ["chunk1", "chunk3"],
            explicit_parent_ids=[],
        )

        assert "section-a" in parent_ids[:2] or "page-a" in parent_ids[:2]

    def test_promote_fact_supported_chunks_penalizes_generic_narrative_for_fact_query(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.chunk_map["chunk_parent_fact"] = {
            "id": "chunk_parent_fact",
            "text": "Campus housing information.",
            "dense_text": "Campus housing information.",
        }
        retriever.chunk_map["chunk_narrative"] = {
            "id": "chunk_narrative",
            "text": ("Like many of the students who arrive at the MBZUAI campus, I was eager, hopeful, and a little daunted. " * 4).strip(),
            "dense_text": ("Like many of the students who arrive at the MBZUAI campus, I was eager, hopeful, and a little daunted. " * 4).strip(),
        }
        retriever.fact_texts_by_chunk["chunk_parent_fact"] = [
            "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents.",
        ]

        support = {
            "chunk_parent_fact": {"score": 0.05, "sources": {"local_facts"}, "record_ids": {"fact-parent"}},
            "chunk_narrative": {"score": 0.20, "sources": {"dense_chunks"}, "record_ids": {"chunk-narrative"}},
        }

        promoted = retriever._promote_fact_supported_chunks(
            "Can parents stay with students on the MBZUAI campus?",
            [("chunk_narrative", 1.0), ("chunk_parent_fact", 0.8)],
            support,
        )

        assert promoted[0][0] == "chunk_parent_fact"

    def test_promote_fact_supported_chunks_preserves_fact_anchor_over_broader_parking_chunk(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.chunk_map["chunk_parking_fact"] = {
            "id": "chunk_parking_fact",
            "text": "Campus FAQ parking answer.",
            "dense_text": "Campus FAQ parking answer.",
        }
        retriever.chunk_map["chunk_parking_policy"] = {
            "id": "chunk_parking_policy",
            "text": "Parking rules and general campus policy for faculty, staff, and students.",
            "dense_text": "Parking rules and general campus policy for faculty, staff, and students.",
        }
        retriever.fact_texts_by_chunk["chunk_parking_fact"] = [
            "Car parking is provided for all registered students and available for guests.",
        ]

        support = {
            "chunk_parking_fact": {"score": 0.05, "sources": {"local_facts"}, "record_ids": {"fact-parking"}},
            "chunk_parking_policy": {"score": 0.30, "sources": {"dense_chunks"}, "record_ids": {"chunk-policy"}},
        }

        promoted = retriever._promote_fact_supported_chunks(
            "Is parking available for students and visitors at MBZUAI?",
            [("chunk_parking_policy", 1.2), ("chunk_parking_fact", 0.8)],
            support,
            anchored_chunk_ids=["chunk_parking_fact"],
        )

        assert promoted[0][0] == "chunk_parking_fact"

    def test_fact_mode_keeps_fact_anchor_chunk_ahead_of_rerank_output(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": True, "max_context_chunks": 3, "fact_neighbor_window": 0},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.fact_map["fact2"] = {
            "id": "fact2",
            "text": "Admissions requirements include transcripts and recommendation letters.",
            "dense_text": "Admissions requirements include transcripts and recommendation letters.",
            "linked_chunk_ids": ["chunk1"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else ["fact2"] if kwargs["namespace"] == "facts" else [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])
        monkeypatch.setattr(retriever, "_rerank_chunk_candidates", lambda *args, **kwargs: [("chunk3", 5.0), ("chunk1", 1.0)])

        result = retriever.retrieve("What are the admissions requirements?")
        assert result["selected_chunk_ids"][0] == "chunk1"

    def test_media_specificity_bonus_prefers_enumerated_facilities_media(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        generic_media = {
            "id": "media-generic",
            "media_type": "image",
            "text": "IMAGE: Map of MBZUAI campus showing location of buildings and facilities.",
            "title": "Campus map",
        }
        enumerated_media = {
            "id": "media-list",
            "media_type": "image",
            "text": "IMAGE: MBZUAI campus facilities: Knowledge Center, Recreation, Residential and retail, Park, Classrooms, Laboratory, Library, Swimming pool, Canteen, Medical Center, Gym.",
            "title": "Campus facilities",
        }

        assert retriever._score_media_relevance("Which facilities are labeled on the MBZUAI campus map?", enumerated_media) > retriever._score_media_relevance("Which facilities are labeled on the MBZUAI campus map?", generic_media)

    def test_rank_parent_candidates_prefers_scoped_legal_affiliation_parent(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.parent_map["parent-legal"] = {
            "id": "parent-legal",
            "parent_type": "section",
            "dense_text": "MBZUAI was established under Law No. 25 of 2019 and is affiliated to the Abu Dhabi Executive Council.",
            "child_chunk_ids": ["chunk1"],
        }
        retriever.parent_map["parent-generic"] = {
            "id": "parent-generic",
            "parent_type": "section",
            "dense_text": "MBZUAI supports research, innovation, students, and faculty across the UAE and internationally.",
            "child_chunk_ids": ["chunk3"],
        }

        ranked = retriever._rank_parent_candidates(
            "Explain MBZUAI's legal basis and institutional affiliation.",
            ["parent-generic", "parent-legal"],
        )

        assert ranked[0] == "parent-legal"

    def test_local_parent_query_ids_prefer_specialization_parent_over_generic_about_page(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.parent_map["parent-specializations"] = {
            "id": "parent-specializations",
            "parent_type": "section",
            "document_title": "University Catalogue",
            "section_path": ["MBZUAI currently offers"],
            "dense_text": "MBZUAI currently offers Ph.D. and M.Sc. programs in five AI specializations including machine learning, computer vision, natural language processing, robotics, and computer science.",
            "child_chunk_ids": ["chunk1"],
        }
        retriever.parent_map["parent-about"] = {
            "id": "parent-about",
            "parent_type": "section",
            "document_title": "Faculty Brochure",
            "section_path": ["About MBZUAI"],
            "dense_text": "About MBZUAI. The university supports academic offerings, students, and faculty in Abu Dhabi.",
            "child_chunk_ids": ["chunk3"],
        }
        for parent_id in ("parent-specializations", "parent-about"):
            parent_text = retriever.parent_map[parent_id]["dense_text"]
            parent_tokens = set(mod._tokenize(parent_text))
            retriever.lexical_map[parent_id] = {"id": parent_id, "text": parent_text}
            retriever._namespace_tokens_by_id.setdefault(retriever.namespace_parents, {})[
                parent_id
            ] = parent_tokens
            token_index = retriever._namespace_token_index.setdefault(retriever.namespace_parents, {})
            for token in parent_tokens:
                token_index.setdefault(token, []).append(parent_id)

        ids = retriever._local_parent_query_ids(
            "What five core AI specializations does MBZUAI offer in its M.Sc. and Ph.D. programs?",
            top_k=2,
        )

        assert ids[0] == "parent-specializations"

    def test_sparse_query_ids_merge_remote_and_local_lexical_hits(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "pinecone_sparse_index": "idx-sparse", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": True, "enable_rerank": False, "rrf_k": 60},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        class FakeSparseIndex:
            calls = []

            def search(self, **kwargs):
                self.calls.append(kwargs)
                return {"result": {"hits": [{"_id": "chunk3"}]}}

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        fake_sparse = FakeSparseIndex()
        retriever._sparse_index = fake_sparse

        ids = retriever._sparse_query_ids(namespace="chunks", query="admissions requirements", top_k=3)
        assert fake_sparse.calls == [
            {
                "namespace": "chunks",
                "top_k": 3,
                "inputs": {"text": "admissions requirements"},
                "fields": [],
                "timeout": 10.0,
            }
        ]
        assert "chunk1" in ids
        assert "chunk3" in ids

    def test_lane_top_ks_prune_parent_and_media_for_fact_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        plan = retriever._lane_top_ks(
            query="Whose name does MBZUAI carry?",
            mode=QueryMode.FACT,
            media_query=False,
        )

        assert plan["parent_dense"] == 0
        assert plan["parent_sparse"] == 0
        assert plan["media_dense"] == 0
        assert plan["media_sparse"] == 0
        assert plan["fact_dense"] > 0
        assert plan["fact_sparse"] > 0

    def test_lane_top_ks_prune_fact_lanes_for_synthesis_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        plan = retriever._lane_top_ks(
            query="Summarize the MBZUAI campus for a visitor.",
            mode=QueryMode.SYNTHESIS,
            media_query=False,
        )

        assert plan["fact_dense"] == 0
        assert plan["fact_sparse"] == 0
        assert plan["fact_local"] == 0
        assert plan["parent_dense"] > 0
        assert plan["media_dense"] > 0
        assert plan["media_sparse"] > 0

    def test_run_query_lanes_skips_zero_top_k_tasks(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "parallel_lane_workers": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        calls = []

        def dense(**kwargs):
            calls.append(("dense", kwargs["namespace"], kwargs["top_k"]))
            return [kwargs["namespace"]]

        def sparse(**kwargs):
            calls.append(("sparse", kwargs["namespace"], kwargs["top_k"]))
            return [kwargs["namespace"]]

        def local_chunk(query, *, top_k):
            calls.append(("local_chunk", top_k))
            return ["chunk1"]

        def local_parent(query, *, top_k):
            calls.append(("local_parent", top_k))
            return ["section-a"]

        def local_media(query, *, top_k):
            calls.append(("local_media", top_k))
            return ["media1"]

        def local_fact(query, *, top_k):
            calls.append(("local_fact", top_k))
            return ["fact1"]

        monkeypatch.setattr(retriever, "_dense_query_ids", dense)
        monkeypatch.setattr(retriever, "_sparse_query_ids", sparse)
        monkeypatch.setattr(retriever, "_local_chunk_query_ids", local_chunk)
        monkeypatch.setattr(retriever, "_local_parent_query_ids", local_parent)
        monkeypatch.setattr(retriever, "_local_media_query_ids", local_media)
        monkeypatch.setattr(retriever, "_local_fact_query_ids", local_fact)

        lane_top_ks = retriever._lane_top_ks(
            query="Whose name does MBZUAI carry?",
            mode=QueryMode.FACT,
            media_query=False,
        )
        results = retriever._run_query_lanes(
            query="Whose name does MBZUAI carry?",
            query_vector=[0.1, 0.2],
            lane_top_ks=lane_top_ks,
            mode=QueryMode.FACT,
        )

        assert ("dense", "parents", 0) not in calls
        assert ("dense", "media", 0) not in calls
        assert ("sparse", "parents", 0) not in calls
        assert ("sparse", "media", 0) not in calls
        assert not any(call[0] == "local_parent" for call in calls)
        assert not any(call[0] == "local_media" for call in calls)
        assert results["parent_dense_ids"] == []
        assert results["sparse_parent_ids"] == []
        assert results["media_dense_ids"] == []
        assert results["sparse_media_ids"] == []
        assert results["fact_dense_ids"] == ["facts"]

    def test_local_chunk_query_ids_promote_exact_chunk_matches(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        ids = retriever._local_chunk_query_ids("What are the admissions requirements?", top_k=2)
        assert ids[0] == "chunk1"

    def test_rerank_retries_with_smaller_documents_on_token_limit(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "enable_sparse": False,
                "enable_rerank": True,
                "rerank_doc_max_tokens": 24,
                "rerank_retry_doc_max_tokens": [8],
            },
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        calls = []

        class FakeInference:
            def rerank(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise RuntimeError("Request contains a query+document pair with 700 tokens, which exceeds the maximum token limit of 512")

                class Response:
                    data = [{"index": 0, "score": 1.0, "document": {"id": "chunk1"}}]

                return Response()

        class FakeClient:
            inference = FakeInference()

        retriever._pinecone_client = FakeClient()
        ranked = retriever._rerank_chunk_candidates(
            "What are the admissions requirements?",
            ["chunk1", "chunk2"],
            {"chunk1": {"score": 1.0, "sources": {"sparse_facts"}}, "chunk2": {"score": 0.5, "sources": {"dense_chunks"}}},
        )

        assert ranked[0][0] == "chunk1"
        assert len(calls) == 2
        first_len = len(calls[0]["documents"][0]["text"].split())
        second_len = len(calls[1]["documents"][0]["text"].split())
        assert second_len < first_len

    def test_fact_mode_keeps_fact_seed_candidates_when_reranker_returns_only_other_docs(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": True, "rerank_return_top_k": 1},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        class FakeInference:
            def rerank(self, **kwargs):
                class Response:
                    data = [{"index": 1, "score": 1.0, "document": {"id": "chunk3"}}]

                return Response()

        class FakeClient:
            inference = FakeInference()

        retriever._pinecone_client = FakeClient()
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else ["fact1"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What are the admissions requirements?")
        assert result["selected_chunk_ids"][0] == "chunk1"

    def test_fact_rerank_uses_smaller_fact_top_n(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "enable_sparse": False,
                "enable_rerank": True,
                "rerank_top_n": 8,
                "rerank_return_top_k": 4,
                "rerank_fact_top_n": 1,
                "rerank_fact_return_top_k": 1,
                "max_context_chunks": 1,
                "rerank_skip_high_confidence_fact": False,
            },
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        calls = []

        class FakeInference:
            def rerank(self, **kwargs):
                calls.append(kwargs)

                class Response:
                    data = [{"index": 0, "score": 1.0, "document": {"id": "chunk1"}}]

                return Response()

        class FakeClient:
            inference = FakeInference()

        retriever._pinecone_client = FakeClient()
        support = {
            "chunk1": {"score": 0.01, "sources": {"dense_chunks"}},
            "chunk2": {"score": 0.01, "sources": {"dense_chunks"}},
        }
        ranked = retriever._rerank_chunk_candidates(
            "Does MBZUAI provide student accommodation?",
            ["chunk1", "chunk2"],
            support,
            mode=QueryMode.FACT,
        )

        assert ranked[0][0] == "chunk1"
        assert len(calls) == 1
        assert len(calls[0]["documents"]) == 1
        assert calls[0]["top_n"] == 1

    def test_fact_rerank_skips_remote_call_for_high_confidence_fact_support(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "enable_sparse": False,
                "enable_rerank": True,
                "rerank_skip_high_confidence_fact": True,
                "rerank_skip_fact_overlap": 0.3,
                "rerank_skip_fact_support_score": 0.01,
            },
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)

        class FakeInference:
            def rerank(self, **kwargs):
                raise AssertionError("remote rerank should be skipped")

        class FakeClient:
            inference = FakeInference()

        retriever._pinecone_client = FakeClient()
        support = {
            "chunk1": {"score": 0.2, "sources": {"sparse_facts", "graph_relation_facts"}},
            "chunk3": {"score": 0.01, "sources": {"dense_chunks"}},
        }
        ranked = retriever._rerank_chunk_candidates(
            "Does MBZUAI provide student accommodation?",
            ["chunk1", "chunk3"],
            support,
            mode=QueryMode.FACT,
            graph_seed_chunk_ids=["chunk1"],
            graph_seed_fact_ids=["fact1"],
        )

        assert ranked[0][0] == "chunk1"

    def test_synthesis_mode_expands_parent_section(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"same_parent_expand_threshold": 1, "max_context_chunks": 5, "max_parent_chunks": 5, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["section-a"] if kwargs["namespace"] == "parents" else ["chunk1", "chunk2"] if kwargs["namespace"] == "chunks" else [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: ["section-a"] if kwargs["namespace"] == "parents" else ["chunk1"])

        result = retriever.retrieve("Explain admissions in detail and summarize the requirements and deadlines")
        assert result["mode"] == "synthesis"
        assert result["selected_chunk_ids"][:2] == ["chunk1", "chunk2"]
        assert result["selected_parent_ids"][0] == "section-a"
        assert result["selected_media_ids"][0] == "media1"
        assert result["retrieval_documents"][0]["media"][0]["url"] == "https://example.com/chart.png"

    def test_scoped_mode_can_expand_from_explicit_parent_hit(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"same_parent_expand_threshold": 2, "max_context_chunks": 5, "max_parent_chunks": 5, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["section-a"] if kwargs["namespace"] == "parents" else ["chunk1"] if kwargs["namespace"] == "chunks" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("Explain the admissions requirements")
        assert result["selected_chunk_ids"][:2] == ["chunk1", "chunk2"]

    def test_fact_mode_abstains_without_fact_support(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "fact_require_fact_support_overlap": 0.35},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What is MBZUAI's Singapore office phone number?")
        assert result["abstained"] is True
        assert result["selected_chunk_ids"] == []
        assert result["dense_parent_ids"] == []
        assert result["selected_parent_ids"] == []
        assert result["selected_media_ids"] == []
        assert result["media"] == []
        assert result["debug_candidates"]["dense_chunk_ids"] == ["chunk3"]

    def test_fact_mode_abstains_when_named_entity_is_missing_from_evidence(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["chunk1"] if kwargs["namespace"] == "chunks" else [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: ["fact1"] if kwargs["namespace"] == "facts" else [])

        result = retriever.retrieve("What is MBZUAI's Singapore office phone number?")
        assert result["abstained"] is True
        assert result["selected_chunk_ids"] == []

    def test_fact_mode_allows_strong_local_fact_support_to_bypass_chunk_only_overlap_gate(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "fact_abstain_min_token_overlap": 0.2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["chunk1"] if kwargs["namespace"] == "chunks" else [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What are the admissions requirements?")
        assert result["abstained"] is False
        assert result["selected_chunk_ids"][0] == "chunk1"

    def test_fact_mode_does_not_use_local_chunk_lane_without_any_fact_support(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What is MBZUAI's Singapore office phone number?")
        assert result["abstained"] is True
        assert result["local_chunk_ids"] == []
        assert result["selected_chunk_ids"] == []

    def test_fact_mode_abstains_on_long_contact_query_with_only_partial_fact_overlap(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.fact_texts_by_chunk["chunk1"] = ["Admissions office phone number: +1 555 0000"]
        monkeypatch.setattr(retriever, "_dense_query_ids", lambda **kwargs: ["chunk1"] if kwargs["namespace"] == "chunks" else ["fact1"] if kwargs["namespace"] == "facts" else [])
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])

        result = retriever.retrieve("What is MBZUAI's Singapore office phone number?")
        assert result["abstained"] is True
        assert result["selected_chunk_ids"] == []

    def test_fact_mode_abstains_on_foreign_campus_address_query_with_only_local_location_evidence(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.chunk_map["chunk1"]["dense_text"] = "MBZUAI is located in a newly developed area of Masdar City, Abu Dhabi."
        retriever.chunk_map["chunk1"]["text"] = "MBZUAI is located in a newly developed area of Masdar City, Abu Dhabi."
        retriever.fact_map["fact1"]["text"] = "MBZUAI is located in a newly developed area of Masdar City, Abu Dhabi."
        retriever.fact_texts_by_chunk["chunk1"] = ["MBZUAI is located in a newly developed area of Masdar City, Abu Dhabi."]
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["chunk1"] if kwargs["namespace"] == "chunks" else ["fact1"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: ["fact1"] if kwargs["namespace"] == "facts" else [])

        result = retriever.retrieve("What is the address of MBZUAI's New York campus?")
        assert result["abstained"] is True
        assert result["selected_chunk_ids"] == []

    def test_fact_mode_abstains_on_foreign_campus_station_query_with_only_local_location_evidence(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.chunk_map["chunk1"]["dense_text"] = "MBZUAI is located in Masdar City, Abu Dhabi."
        retriever.chunk_map["chunk1"]["text"] = "MBZUAI is located in Masdar City, Abu Dhabi."
        retriever.fact_map["fact1"]["text"] = "MBZUAI is located in Masdar City, Abu Dhabi."
        retriever.fact_texts_by_chunk["chunk1"] = ["MBZUAI is located in Masdar City, Abu Dhabi."]
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["chunk1"] if kwargs["namespace"] == "chunks" else ["fact1"] if kwargs["namespace"] == "facts" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: ["fact1"] if kwargs["namespace"] == "facts" else [])

        result = retriever.retrieve("What is the metro station for MBZUAI's Paris campus?")
        assert result["abstained"] is True
        assert result["selected_chunk_ids"] == []

    def test_attach_media_prefers_chunk_linked_media(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "max_media_results": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.media_map["media2"] = {
            "id": "media2",
            "media_type": "image",
            "title": "Generic campus image",
            "caption": "Generic campus image",
            "description": "Generic campus image",
            "linked_chunk_ids": [],
            "linked_parent_ids": ["section-a"],
        }

        result = retriever._attach_media(["chunk1"], ["media2"], "Show the admissions chart")
        assert result[0]["id"] == "media1"

    def test_local_media_query_ids_prioritize_page_visual_for_layout_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "max_media_results": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.media_map["media_visual"] = {
            "id": "media_visual",
            "media_type": "page_visual",
            "title": "Campus map page 1",
            "description": "Campus layout with buildings and facilities",
            "linked_chunk_ids": ["chunk1"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        retriever.media_texts_by_id["media_visual"] = "Campus map page 1 campus layout with buildings and facilities"
        media_text = retriever.media_texts_by_id["media_visual"]
        media_tokens = set(mod._tokenize(media_text))
        retriever.lexical_map["media_visual"] = {"id": "media_visual", "text": media_text}
        retriever._namespace_tokens_by_id.setdefault(retriever.namespace_media, {})[
            "media_visual"
        ] = media_tokens
        token_index = retriever._namespace_token_index.setdefault(retriever.namespace_media, {})
        for token in media_tokens:
            token_index.setdefault(token, []).append("media_visual")

        ids = retriever._local_media_query_ids("What does the campus map show about the building layout?", top_k=2)
        assert ids[0] == "media_visual"

    def test_local_media_query_ids_ignore_low_signal_media_records(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "max_media_results": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.media_map["bad_media"] = {
            "id": "bad_media",
            "media_type": "image",
            "title": "I'm sorry, but I cannot provide a description or answer for the image you've mentioned as it doesn't contain any text or information that I can describe.",
            "description": "",
            "linked_chunk_ids": ["chunk1"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        retriever.media_texts_by_id["bad_media"] = retriever.media_map["bad_media"]["title"]
        ids = retriever._local_media_query_ids("Show the admissions chart and application layout", top_k=3)
        assert "bad_media" not in ids

    def test_attach_media_skips_low_signal_media_even_when_linked(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "max_media_results": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.media_map["bad_media"] = {
            "id": "bad_media",
            "media_type": "image",
            "title": "I'm sorry, but I cannot provide a description or answer for the image you've mentioned as it doesn't contain any text or information that I can describe.",
            "description": "",
            "linked_chunk_ids": ["chunk1"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        attached = retriever._attach_media(["chunk1"], ["bad_media", "media1"], "Show the admissions chart")
        assert attached[0]["id"] == "media1"
        assert all(media["id"] != "bad_media" for media in attached)

    def test_attach_media_prefers_specific_image_over_generic_page_visual_for_map_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "max_media_results": 3},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.media_map["media_visual"] = {
            "id": "media_visual",
            "media_type": "page_visual",
            "title": "Campus brochure page",
            "description": "Parking, travel and tourism services near campus.",
            "linked_chunk_ids": ["chunk2"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        retriever.media_map["media_map"] = {
            "id": "media_map",
            "media_type": "image",
            "title": "Map of campus showing buildings and facilities.",
            "description": "Map of campus showing buildings, parking, and facilities.",
            "linked_chunk_ids": ["chunk1"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        retriever.media_texts_by_id["media_visual"] = "Parking, travel and tourism services near campus."
        retriever.media_texts_by_id["media_map"] = "Map of campus showing buildings, parking, and facilities."

        attached = retriever._attach_media(
            ["chunk1", "chunk2"],
            ["media_visual", "media_map"],
            "How does the campus map organize buildings, parking, and campus services?",
        )
        assert attached[0]["id"] == "media_map"

    def test_local_media_query_ids_skip_contents_page_visuals(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False, "max_media_results": 2},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.media_map["contents_visual"] = {
            "id": "contents_visual",
            "media_type": "page_visual",
            "title": "University catalogue page 3",
            "description": "Contents 07 08 10 12 13 Student services Parking Transportation Campus map",
            "linked_chunk_ids": ["chunk1"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        retriever.media_texts_by_id["contents_visual"] = retriever.media_map["contents_visual"]["description"]

        ids = retriever._local_media_query_ids("Which support services are visibly marked on the campus map?", top_k=3)
        assert "contents_visual" not in ids

    def test_media_query_uses_media_parent_hints_for_expansion(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"same_parent_expand_threshold": 2, "max_context_chunks": 4, "max_parent_chunks": 4, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        retriever.media_map["media_visual"] = {
            "id": "media_visual",
            "media_type": "page_visual",
            "title": "Campus map page 1",
            "description": "Campus map layout with admissions building and student services.",
            "linked_chunk_ids": ["chunk1"],
            "linked_parent_ids": ["section-a", "page-a"],
        }
        retriever.media_texts_by_id["media_visual"] = "Campus map layout with admissions building and student services."

        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else ["media_visual"] if kwargs["namespace"] == "media" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])
        monkeypatch.setattr(retriever, "_should_abstain", lambda **kwargs: False)

        result = retriever.retrieve("Show the campus map layout for admissions")
        assert result["selected_chunk_ids"][0] == "chunk1"
        assert "page-a" in result["selected_parent_ids"] or "section-a" in result["selected_parent_ids"]

    def test_scoped_query_prefers_explicit_parent_chunks_for_legal_relation_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"same_parent_expand_threshold": 2, "max_context_chunks": 4, "max_parent_chunks": 4, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        monkeypatch.setattr(
            retriever,
            "_dense_query_ids",
            lambda **kwargs: ["chunk3"] if kwargs["namespace"] == "chunks" else ["section-a", "page-a"] if kwargs["namespace"] == "parents" else [],
        )
        monkeypatch.setattr(retriever, "_sparse_query_ids", lambda **kwargs: [])
        monkeypatch.setattr(retriever, "_local_chunk_query_ids", lambda *args, **kwargs: [])
        monkeypatch.setattr(retriever, "_should_abstain", lambda **kwargs: False)

        result = retriever.retrieve("Under which law was MBZUAI established and to which entity is it affiliated?")
        assert result["selected_chunk_ids"][0] == "chunk1"
        assert "section-a" in result["selected_parent_ids"] or "page-a" in result["selected_parent_ids"]

    def test_local_parent_lane_includes_campus_services_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, classify_query_mode

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        query = "What student-facing campus services and facilities are available at MBZUAI?"
        mode = classify_query_mode(query)
        assert mode == mod.QueryMode.SCOPED
        assert retriever._should_use_local_parent_lane(query, mode=mode) is True

    def test_specialization_queries_do_not_force_explicit_parent_chunks_ahead_of_good_seeds(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

        run_dir = self._build_retrieval_run(tmp_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"same_parent_expand_threshold": 2, "max_context_chunks": 4, "max_parent_chunks": 4, "enable_sparse": False, "enable_rerank": False},
        }

        monkeypatch.setenv("PINECONE_API_KEY", "x")
        monkeypatch.setenv("GEMINI_API_KEY", "x")
        import pipeline.retrieval.adaptive_hybrid as mod
        monkeypatch.setattr(mod, "_embed_query", lambda *args, **kwargs: [0.1, 0.2])

        retriever = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        assert (
            retriever._should_prefer_explicit_parent_chunks(
                "Describe the main AI specialization areas covered by graduate programs.",
                mode=QueryMode.SCOPED,
                media_query=False,
            )
            is False
        )

    def test_embed_query_uses_gemini_compatible_config(self, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import _embed_query

        captured = {}

        class FakeEmbedResponse:
            embeddings = [type("Embedding", (), {"values": [0.1, 0.2]})()]

        class FakeModels:
            def embed_content(self, **kwargs):
                captured.update(kwargs)
                return FakeEmbedResponse()

        class FakeClient:
            models = FakeModels()

        monkeypatch.setattr(
            "pipeline.retrieval.adaptive_hybrid._make_gemini_client",
            lambda: FakeClient(),
        )

        vector = _embed_query("test query", model="gemini-embedding-2-preview", output_dimensionality=1536)

        assert vector == [0.1, 0.2]
        assert captured["model"] == "gemini-embedding-2-preview"
        assert getattr(captured["config"], "auto_truncate", None) is None


class TestEmbedderHelpers:
    def test_serialize_metadata_value_keeps_valid_json(self):
        from pipeline.stages.embedders.openai_embedder import _serialize_metadata_value

        value = [
            {
                "type": "video",
                "url": "https://example.com/intro.mp4",
                "transcript": "A" * 5000,
            }
            for _ in range(12)
        ]
        serialized = _serialize_metadata_value("media", value, max_len=800)
        parsed = json.loads(serialized)
        assert isinstance(parsed, list)
        assert parsed

    def test_gemini_embedder_retries_connectivity_errors(self):
        from pipeline.stages.embedders.gemini_pinecone_embedder import _is_retryable_exception

        assert _is_retryable_exception(RuntimeError("Failed to connect; did you specify the correct index name?"))


class TestRetrievalEvaluation:
    def test_retrieval_eval_accepts_alternate_gold_ids_from_metadata(self, tmp_dir):
        from pipeline.evaluation.dataset import EvalExample
        from pipeline.evaluation.retrieval_eval import _score_query

        example = EvalExample(
            id="q1",
            query="Is parking provided?",
            query_type="fact",
            source_type="webpage",
            gold_chunk_ids=["faq-chunk"],
            gold_parent_ids=["faq-parent"],
            metadata={
                "alternate_gold_chunk_ids": ["facilities-chunk"],
                "alternate_gold_parent_ids": ["facilities-parent"],
            },
        ).normalized()
        result = {
            "mode": "fact",
            "selected_chunk_ids": ["facilities-chunk"],
            "selected_parent_ids": ["facilities-parent"],
            "selected_media_ids": [],
            "dense_parent_ids": [],
            "dense_media_ids": [],
            "seed_chunk_ids": ["facilities-chunk"],
        }

        score = _score_query(example, result)
        assert score.chunk_hit_at_10 == 1.0
        assert score.parent_hit_at_5 == 1.0

    def test_retrieval_eval_excludes_no_answer_examples_from_relevance_averages(self):
        from pipeline.evaluation.retrieval_eval import QueryRetrievalScore, _aggregate_scores

        answerable = QueryRetrievalScore(
            id="q1",
            query="Where is MBZUAI located?",
            query_type="fact",
            source_type="webpage",
            mode="fact",
            no_answer=False,
            has_gold_chunks=True,
            has_gold_parents=True,
            has_gold_media=False,
            seed_chunk_hit_at_5=1.0,
            chunk_hit_at_5=1.0,
            chunk_hit_at_10=1.0,
            chunk_recall_at_10=1.0,
            chunk_mrr_at_10=1.0,
            chunk_ndcg_at_10=1.0,
            parent_hit_at_5=1.0,
            media_hit_at_1=0.0,
            media_hit_at_5=0.0,
            media_mrr_at_5=0.0,
            expansion_gain_hit=0.0,
            no_answer_violation=0.0,
            selected_chunk_ids=["chunk1"],
            selected_parent_ids=["parent1"],
            selected_media_ids=[],
            dense_parent_ids=[],
            dense_media_ids=[],
        )
        no_answer = QueryRetrievalScore(
            id="q2",
            query="What is MBZUAI's Singapore office phone number?",
            query_type="fact",
            source_type="none",
            mode="fact",
            no_answer=True,
            has_gold_chunks=False,
            has_gold_parents=False,
            has_gold_media=False,
            seed_chunk_hit_at_5=0.0,
            chunk_hit_at_5=0.0,
            chunk_hit_at_10=0.0,
            chunk_recall_at_10=0.0,
            chunk_mrr_at_10=0.0,
            chunk_ndcg_at_10=0.0,
            parent_hit_at_5=0.0,
            media_hit_at_1=0.0,
            media_hit_at_5=0.0,
            media_mrr_at_5=0.0,
            expansion_gain_hit=0.0,
            no_answer_violation=0.0,
            selected_chunk_ids=[],
            selected_parent_ids=[],
            selected_media_ids=[],
            dense_parent_ids=[],
            dense_media_ids=[],
        )

        aggregate = _aggregate_scores([answerable, no_answer])
        assert aggregate["chunk_mrr_at_10"] == 1.0
        assert aggregate["chunk_recall_at_10"] == 1.0
        assert aggregate["no_answer_violation_rate"] == 0.0

    def test_retrieval_eval_only_counts_media_metric_for_media_eligible_queries(self):
        from pipeline.evaluation.retrieval_eval import QueryRetrievalScore, _aggregate_scores

        chunk_only = QueryRetrievalScore(
            id="q1",
            query="Where is MBZUAI located?",
            query_type="fact",
            source_type="webpage",
            mode="fact",
            no_answer=False,
            has_gold_chunks=True,
            has_gold_parents=True,
            has_gold_media=False,
            seed_chunk_hit_at_5=1.0,
            chunk_hit_at_5=1.0,
            chunk_hit_at_10=1.0,
            chunk_recall_at_10=1.0,
            chunk_mrr_at_10=1.0,
            chunk_ndcg_at_10=1.0,
            parent_hit_at_5=1.0,
            media_hit_at_1=0.0,
            media_hit_at_5=0.0,
            media_mrr_at_5=0.0,
            expansion_gain_hit=0.0,
            no_answer_violation=0.0,
            selected_chunk_ids=["chunk1"],
            selected_parent_ids=["parent1"],
            selected_media_ids=[],
            dense_parent_ids=[],
            dense_media_ids=[],
        )
        media_query = QueryRetrievalScore(
            id="q2",
            query="What does the campus map show?",
            query_type="multimodal",
            source_type="pdf",
            mode="scoped",
            no_answer=False,
            has_gold_chunks=True,
            has_gold_parents=True,
            has_gold_media=True,
            seed_chunk_hit_at_5=1.0,
            chunk_hit_at_5=1.0,
            chunk_hit_at_10=1.0,
            chunk_recall_at_10=1.0,
            chunk_mrr_at_10=1.0,
            chunk_ndcg_at_10=1.0,
            parent_hit_at_5=1.0,
            media_hit_at_1=1.0,
            media_hit_at_5=1.0,
            media_mrr_at_5=1.0,
            expansion_gain_hit=0.0,
            no_answer_violation=0.0,
            selected_chunk_ids=["chunk2"],
            selected_parent_ids=["parent2"],
            selected_media_ids=["media1"],
            dense_parent_ids=[],
            dense_media_ids=[],
        )

        aggregate = _aggregate_scores([chunk_only, media_query])
        assert aggregate["eligible_media_query_count"] == 1.0
        assert aggregate["media_hit_at_1"] == 1.0
        assert aggregate["media_hit_at_5"] == 1.0
        assert aggregate["media_mrr_at_5"] == 1.0


class TestGraphRAGRetriever:
    def _write_graph_artifacts(self, run_dir):
        from pipeline.core.io import atomic_write_json

        graph_dir = run_dir / "stage_outputs" / "format_graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph = {
            "schema_version": 1,
            "graph_type": "deterministic_content_graph",
            "nodes": [
                {"id": "doc1", "node_type": "document", "label": "Admissions"},
                {"id": "page-a", "node_type": "page", "label": "Admissions page"},
                {"id": "section-a", "node_type": "section", "label": "Admissions section"},
                {"id": "chunk1", "node_type": "chunk", "label": "Admissions requirement details"},
                {"id": "fact1", "node_type": "fact", "label": "Admissions requirements include transcripts and recommendation letters."},
                {"id": "media1", "node_type": "media", "label": "Admissions chart"},
            ],
            "edges": [
                {"id": "edge-doc-page", "edge_type": "HAS_PAGE", "source_id": "doc1", "target_id": "page-a"},
                {"id": "edge-page-section", "edge_type": "PAGE_HAS_SECTION", "source_id": "page-a", "target_id": "section-a"},
                {"id": "edge-section-chunk", "edge_type": "SECTION_HAS_CHUNK", "source_id": "section-a", "target_id": "chunk1"},
                {"id": "edge-chunk-fact", "edge_type": "CHUNK_HAS_FACT", "source_id": "chunk1", "target_id": "fact1"},
                {"id": "edge-page-fact", "edge_type": "PAGE_HAS_FACT", "source_id": "page-a", "target_id": "fact1"},
                {"id": "edge-chunk-media", "edge_type": "CHUNK_HAS_MEDIA", "source_id": "chunk1", "target_id": "media1"},
            ],
        }
        graph_index = {
            "schema_version": 1,
            "graph_type": "deterministic_content_graph",
            "outgoing_edge_ids": {
                "doc1": ["edge-doc-page"],
                "page-a": ["edge-page-section", "edge-page-fact"],
                "section-a": ["edge-section-chunk"],
                "chunk1": ["edge-chunk-fact", "edge-chunk-media"],
            },
        }
        atomic_write_json(graph_dir / "knowledge_graph.json", graph)
        atomic_write_json(graph_dir / "knowledge_graph_index.json", graph_index)

    def _write_promoted_graph_artifacts(self, run_dir):
        from pipeline.core.io import atomic_write_json

        graph_dir = run_dir / "stage_outputs" / "promote_graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        graph = {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "nodes": [
                {"id": "chunk1", "node_type": "chunk", "label": "Admissions requirement details"},
                {"id": "fact1", "node_type": "fact", "label": "Admissions requirements include transcripts and recommendation letters."},
                {"id": "entity:admissions", "node_type": "entity", "label": "Admissions requirements"},
                {
                    "id": "assertion1",
                    "node_type": "relation_assertion",
                    "label": "REQUIRES",
                    "properties": {
                        "text": "Admissions requirements REQUIRES transcripts. Evidence: Admissions requirements include transcripts and recommendation letters.",
                        "source_chunk_ids": ["chunk1"],
                        "source_fact_ids": ["fact1"],
                        "source_parent_ids": ["page-a"],
                    },
                },
            ],
            "edges": [
                {"id": "edge-chunk-fact", "edge_type": "CHUNK_HAS_FACT", "source_id": "chunk1", "target_id": "fact1"},
                {"id": "edge-fact-assertion", "edge_type": "FACT_SUPPORTS_ASSERTION", "source_id": "fact1", "target_id": "assertion1"},
                {"id": "edge-assertion-entity", "edge_type": "ASSERTION_OBJECT", "source_id": "assertion1", "target_id": "entity:admissions"},
            ],
        }
        graph_index = {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "outgoing_edge_ids": {
                "chunk1": ["edge-chunk-fact"],
                "fact1": ["edge-fact-assertion"],
                "assertion1": ["edge-assertion-entity"],
            },
        }
        atomic_write_json(graph_dir / "promoted_knowledge_graph.json", graph)
        atomic_write_json(graph_dir / "promoted_knowledge_graph_index.json", graph_index)

    def test_from_config_returns_graph_retriever_when_enabled(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"retriever_backend": "graph_hybrid"},
        }

        import pipeline.retrieval.adaptive_hybrid as mod

        monkeypatch.setattr(mod, "load_config", lambda _name: config)
        retriever = AdaptiveHybridRetriever.from_config(config_name="graph", work_dir=run_dir)
        assert retriever.__class__.__name__ == "GraphRAGRetriever"

    def test_graph_retriever_augments_fact_evidence(self, tmp_dir):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_only": False,
                "graph_max_fact_results": 2,
                "graph_max_additional_media_results": 0,
            },
        }

        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None: {
            "query": query,
            "mode": "fact",
            "seed_chunk_ids": ["chunk1"],
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["section-a", "page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["section-a", "page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [
                {
                    "id": "chunk1",
                    "text": "Admissions requirement details",
                    "source_url": "https://example.com/admissions",
                    "document_title": "Admissions",
                    "document_summary": "",
                    "media": [],
                }
            ],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("What are the admissions requirements?")

        assert result["retriever_backend"] == "graph_hybrid"
        assert result["graph_used"] is True
        assert result["graph_fact_ids"] == ["fact1"]
        assert "fact1" in [doc["id"] for doc in result["retrieval_documents"]]

    def test_graph_retriever_can_augment_cached_vector_result(self, tmp_dir):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_only": False,
                "graph_max_fact_results": 2,
                "graph_max_additional_media_results": 0,
            },
        }

        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None: (_ for _ in ()).throw(AssertionError("base retrieve should not be called"))

        cached_result = {
            "query": "What are the admissions requirements?",
            "mode": "fact",
            "seed_chunk_ids": ["chunk1"],
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["section-a", "page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["section-a", "page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [
                {
                    "id": "chunk1",
                    "text": "Admissions requirement details",
                    "source_url": "https://example.com/admissions",
                    "document_title": "Admissions",
                    "document_summary": "",
                    "media": [],
                }
            ],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.augment_result("What are the admissions requirements?", cached_result)

        assert result["retriever_backend"] == "graph_hybrid"
        assert result["graph_used"] is True
        assert result["graph_fact_ids"] == ["fact1"]
        assert "fact1" in [doc["id"] for doc in result["retrieval_documents"]]

    def test_graph_retriever_keeps_base_chunk_order_while_adding_assertion_backed_chunk(self, tmp_dir):
        from pipeline.core.io import atomic_write_json
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        graph_dir = run_dir / "stage_outputs" / "promote_graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        promoted_graph = {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "nodes": [
                {"id": "chunk1", "node_type": "chunk", "label": "Admissions requirement details"},
                {"id": "chunk2", "node_type": "chunk", "label": "Application deadlines and fees"},
                {"id": "page-a", "node_type": "page", "label": "Admissions page"},
                {"id": "fact1", "node_type": "fact", "label": "Admissions requirements include transcripts and recommendation letters."},
                {
                    "id": "assertion1",
                    "node_type": "relation_assertion",
                    "label": "REQUIRES",
                    "properties": {
                        "text": "Admissions requirements REQUIRES transcripts. Evidence: official requirements are listed in the application deadlines and fees section.",
                        "source_chunk_ids": ["chunk2"],
                        "source_fact_ids": ["fact1"],
                        "source_parent_ids": ["page-a"],
                    },
                },
            ],
            "edges": [
                {"id": "edge-chunk1-fact", "edge_type": "CHUNK_HAS_FACT", "source_id": "chunk1", "target_id": "fact1"},
                {"id": "edge-fact-assertion", "edge_type": "FACT_SUPPORTS_ASSERTION", "source_id": "fact1", "target_id": "assertion1"},
                {"id": "edge-chunk2-assertion", "edge_type": "CHUNK_SUPPORTS_ASSERTION", "source_id": "chunk2", "target_id": "assertion1"},
            ],
        }
        promoted_index = {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "outgoing_edge_ids": {
                "chunk1": ["edge-chunk1-fact"],
                "fact1": ["edge-fact-assertion"],
                "chunk2": ["edge-chunk2-assertion"],
            },
        }
        atomic_write_json(graph_dir / "promoted_knowledge_graph.json", promoted_graph)
        atomic_write_json(graph_dir / "promoted_knowledge_graph_index.json", promoted_index)

        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_only": False,
                "graph_max_fact_results": 1,
                "graph_max_assertion_results": 1,
            },
        }
        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None: {
            "query": query,
            "mode": "fact",
            "seed_chunk_ids": ["chunk1", "chunk2"],
            "selected_chunk_ids": ["chunk1", "chunk2"],
            "selected_parent_ids": ["page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [
                {
                    "id": "chunk1",
                    "text": "Admissions requirement details",
                    "source_url": "https://example.com/admissions",
                    "document_title": "Admissions",
                    "document_summary": "",
                    "media": [],
                },
                {
                    "id": "chunk2",
                    "text": "Application deadlines and fees",
                    "source_url": "https://example.com/admissions",
                    "document_title": "Admissions",
                    "document_summary": "",
                    "media": [],
                },
            ],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("Which chunk contains the official admissions requirement evidence?")

        assert result["graph_assertion_ids"] == ["assertion1"]
        assert result["selected_chunk_ids"][:2] == ["chunk1", "chunk2"]

    def test_graph_retriever_prefers_promoted_graph_assertions(self, tmp_dir):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_only": False,
                "graph_max_fact_results": 1,
                "graph_max_assertion_results": 2,
            },
        }

        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None: {
            "query": query,
            "mode": "fact",
            "seed_chunk_ids": ["chunk1"],
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [
                {
                    "id": "chunk1",
                    "text": "Admissions requirement details",
                    "source_url": "https://example.com/admissions",
                    "document_title": "Admissions",
                    "document_summary": "",
                    "media": [],
                }
            ],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("Do admissions require transcripts?")

        assert result["graph_used"] is True
        assert result["graph_assertion_ids"] == ["assertion1"]
        assert "assertion1" in [doc["id"] for doc in result["retrieval_documents"]]

    def test_graph_retriever_can_use_neo4j_online_graph_results(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_query_backend": "neo4j",
                "graph_relation_only": False,
                "graph_max_fact_results": 2,
                "graph_max_additional_media_results": 0,
            },
            "graph": {
                "neo4j_uri": "https://neo4j.example",
                "neo4j_database": "neo4j",
                "neo4j_username": "neo4j",
                "neo4j_password": "secret",
                "neo4j_namespace": "mbzuai-test",
            },
        }

        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None: {
            "query": query,
            "mode": "fact",
            "seed_chunk_ids": ["chunk1"],
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["section-a", "page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["section-a", "page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [
                {
                    "id": "chunk1",
                    "text": "Admissions requirement details",
                    "source_url": "https://example.com/admissions",
                    "document_title": "Admissions",
                    "document_summary": "",
                    "media": [],
                }
            ],
            "media": [],
            "abstained": False,
        }

        monkeypatch.setattr(
            "pipeline.retrieval.graph_rag._post_query",
            lambda **kwargs: {
                "data": {
                    "fields": [
                        "target_id",
                        "node_type",
                        "label",
                        "text",
                        "evidence",
                        "source_url",
                        "document_title",
                        "title",
                        "caption",
                        "description",
                        "context",
                        "transcript",
                        "media_type",
                        "url",
                        "asset_uri",
                        "local_path",
                        "source_chunk_ids",
                        "source_fact_ids",
                        "source_parent_ids",
                        "edge_ids",
                    ],
                    "values": [
                        [
                            "fact1",
                            "fact",
                            "Admissions requirements include transcripts and recommendation letters.",
                            "Admissions requirements include transcripts and recommendation letters.",
                            None,
                            "https://example.com/admissions",
                            "Admissions",
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            None,
                            ["chunk1"],
                            [],
                            ["page-a", "section-a"],
                            ["edge-chunk-fact"],
                        ]
                    ],
                }
            },
        )

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("What are the admissions requirements?")

        assert result["graph_used"] is True
        assert result["graph_store_backend"] == "neo4j"
        assert result["graph_fact_ids"] == ["fact1"]
        assert result["graph_store_error"] is None

    def test_graph_retriever_falls_back_to_local_when_neo4j_query_fails(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_query_backend": "auto",
                "graph_relation_only": False,
                "graph_max_fact_results": 2,
                "graph_max_additional_media_results": 0,
            },
            "graph": {
                "neo4j_uri": "https://neo4j.example",
                "neo4j_database": "neo4j",
                "neo4j_username": "neo4j",
                "neo4j_password": "secret",
                "neo4j_namespace": "mbzuai-test",
            },
        }

        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None: {
            "query": query,
            "mode": "fact",
            "seed_chunk_ids": ["chunk1"],
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["section-a", "page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["section-a", "page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [
                {
                    "id": "chunk1",
                    "text": "Admissions requirement details",
                    "source_url": "https://example.com/admissions",
                    "document_title": "Admissions",
                    "document_summary": "",
                    "media": [],
                }
            ],
            "media": [],
            "abstained": False,
        }

        monkeypatch.setattr(
            "pipeline.retrieval.graph_rag._post_query",
            lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Neo4j unavailable")),
        )

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("What are the admissions requirements?")

        assert result["graph_used"] is True
        assert result["graph_store_backend"] == "local"
        assert result["graph_fact_ids"] == ["fact1"]
        assert result["graph_store_error"] == "Neo4j unavailable"

    def test_graph_retriever_relation_router_can_seed_base_retriever(self, tmp_dir):
        from pipeline.core.io import atomic_write_json
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever
        import pipeline.retrieval.adaptive_hybrid as mod

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        graph_dir = run_dir / "stage_outputs" / "promote_graph"
        graph_dir.mkdir(parents=True, exist_ok=True)
        promoted_graph = {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "nodes": [
                {"id": "chunk1", "node_type": "chunk", "label": "Campus location details"},
                {"id": "page-a", "node_type": "page", "label": "Campus page"},
                {"id": "fact1", "node_type": "fact", "label": "MBZUAI is located in Masdar City, Abu Dhabi."},
                {
                    "id": "assertion1",
                    "node_type": "relation_assertion",
                    "label": "LOCATED_IN",
                    "properties": {
                        "text": "MBZUAI LOCATED_IN Masdar City, Abu Dhabi.",
                        "relation_type": "LOCATED_IN",
                        "subject_name": "MBZUAI",
                        "object_name": "Masdar City, Abu Dhabi",
                        "confidence": 0.95,
                        "source_chunk_ids": ["chunk1"],
                        "source_fact_ids": ["fact1"],
                        "source_parent_ids": ["page-a"],
                    },
                },
            ],
            "edges": [
                {"id": "edge-fact-assertion", "edge_type": "FACT_SUPPORTS_ASSERTION", "source_id": "fact1", "target_id": "assertion1"},
                {"id": "edge-chunk-fact", "edge_type": "CHUNK_HAS_FACT", "source_id": "chunk1", "target_id": "fact1"},
            ],
        }
        promoted_index = {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "outgoing_edge_ids": {
                "fact1": ["edge-fact-assertion"],
                "chunk1": ["edge-chunk-fact"],
            },
        }
        atomic_write_json(graph_dir / "promoted_knowledge_graph.json", promoted_graph)
        atomic_write_json(graph_dir / "promoted_knowledge_graph_index.json", promoted_index)

        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_router_enabled": True,
                "graph_relation_route_min_confidence": 0.1,
                "graph_relation_graph_first_min_confidence": 0.1,
                "graph_relation_reorder_min_confidence": 0.9,
            },
        }
        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        assertion_text = "MBZUAI LOCATED_IN Masdar City, Abu Dhabi."
        assertion_tokens = set(
            mod._tokenize(
                "MBZUAI located location city Masdar Abu Dhabi campus address " + assertion_text
            )
        )
        base.lexical_map["assertion1"] = {"id": "assertion1", "text": assertion_text}
        base._namespace_tokens_by_id.setdefault(base.namespace_assertions, {})[
            "assertion1"
        ] = assertion_tokens
        token_index = base._namespace_token_index.setdefault(base.namespace_assertions, {})
        for token in assertion_tokens:
            token_index.setdefault(token, []).append("assertion1")
        seen = {}

        def _retrieve(query, *, query_vector=None, seed_overrides=None):
            seen["seed_overrides"] = dict(seed_overrides or {})
            return {
                "query": query,
                "mode": "fact",
                "seed_chunk_ids": ["chunk1"],
                "selected_chunk_ids": ["chunk1"],
                "selected_parent_ids": ["page-a"],
                "selected_media_ids": [],
                "dense_parent_ids": ["page-a"],
                "dense_media_ids": [],
                "retrieval_documents": [],
                "media": [],
                "abstained": False,
            }

        base.retrieve = _retrieve
        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("In which city is MBZUAI located?")

        assert result["graph_relation_family"] == "location"
        assert result["graph_relation_graph_first"] is True
        assert seen["seed_overrides"]["graph_relation_chunk_ids"] == ["chunk1"]
        assert "graph_relation_fact_ids" in seen["seed_overrides"]

    def test_graph_retriever_skips_graph_work_for_non_relation_query_when_relation_only(self, tmp_dir):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_only": True,
            },
        }
        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None, seed_overrides=None: {
            "query": query,
            "mode": "synthesis",
            "seed_chunk_ids": ["chunk1"],
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("Summarize the MBZUAI campus for a visitor.")

        assert result["graph_used"] is False
        assert result["graph_relation_family"] == ""
        assert result["graph_store_backend"] == "none"

    def test_graph_relation_router_does_not_use_graph_first_for_offering_queries(self, tmp_dir):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_router_enabled": True,
                "graph_relation_route_min_confidence": 0.1,
                "graph_relation_graph_first_min_confidence": 0.1,
            },
        }
        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        seen = {}

        def _retrieve(query, *, query_vector=None, seed_overrides=None):
            seen["seed_overrides"] = dict(seed_overrides or {})
            return {
                "query": query,
                "mode": "fact",
                "seed_chunk_ids": ["chunk1"],
                "selected_chunk_ids": ["chunk1"],
                "selected_parent_ids": ["page-a"],
                "selected_media_ids": [],
                "dense_parent_ids": ["page-a"],
                "dense_media_ids": [],
                "retrieval_documents": [],
                "media": [],
                "abstained": False,
            }

        base.retrieve = _retrieve
        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("Is student housing available at MBZUAI?")

        assert result["graph_relation_family"] == "accommodation"
        assert result["graph_relation_graph_first"] is False
        assert seen["seed_overrides"] == {}

    def test_graph_relation_router_classifies_where_is_parking_as_offering_not_location(self, tmp_dir):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        self._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_router_enabled": True,
                "graph_relation_route_min_confidence": 0.1,
                "graph_relation_graph_first_min_confidence": 0.1,
            },
        }
        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None, seed_overrides=None: {
            "query": query,
            "mode": "fact",
            "seed_chunk_ids": ["chunk1"],
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["page-a"],
            "selected_media_ids": [],
            "dense_parent_ids": ["page-a"],
            "dense_media_ids": [],
            "retrieval_documents": [],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("Where is parking permitted at the Masdar City campus?")

        assert result["graph_relation_family"] == "transport"
        assert result["graph_relation_graph_first"] is False

    def test_graph_relation_router_classifies_contact_email_queries(self, tmp_dir):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphRAGRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_contact_lookup_run(tmp_dir)
        self._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_router_enabled": True,
                "graph_relation_route_min_confidence": 0.1,
                "graph_relation_graph_first_min_confidence": 0.1,
            },
        }
        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        base.retrieve = lambda query, query_vector=None, seed_overrides=None: {
            "query": query,
            "mode": "fact",
            "seed_chunk_ids": ["contact-chunk"],
            "selected_chunk_ids": ["contact-chunk"],
            "selected_parent_ids": ["contact-page"],
            "selected_media_ids": [],
            "dense_parent_ids": ["contact-page"],
            "dense_media_ids": [],
            "retrieval_documents": [],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        result = retriever.retrieve("What is the admissions committee email address?")

        assert result["graph_relation_family"] == "contact"
        assert result["graph_relation_graph_first"] is False

    def test_graph_augment_does_not_reorder_chunks_for_non_graph_first_contact_queries(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
        from pipeline.retrieval.graph_rag import GraphQueryContext, GraphRAGRetriever, RelationCandidateSet, RelationQueryPlan

        run_dir = TestAdaptiveHybridRetriever()._build_contact_lookup_run(tmp_dir)
        self._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "graph_hybrid",
                "graph_relation_router_enabled": True,
                "graph_relation_route_min_confidence": 0.1,
                "graph_relation_graph_first_min_confidence": 0.95,
                "graph_relation_reorder_min_confidence": 0.1,
            },
        }
        base = AdaptiveHybridRetriever(config=config, work_dir=run_dir)
        cached_result = {
            "query": "How can I contact admissions?",
            "mode": "fact",
            "seed_chunk_ids": ["contact-chunk", "policy-chunk"],
            "selected_chunk_ids": ["contact-chunk", "policy-chunk"],
            "selected_parent_ids": ["contact-page"],
            "selected_media_ids": [],
            "dense_parent_ids": ["contact-page"],
            "dense_media_ids": [],
            "retrieval_documents": [],
            "media": [],
            "abstained": False,
        }

        retriever = GraphRAGRetriever(config=config, work_dir=run_dir, base_retriever=base)
        calls = {"reorder": 0}

        monkeypatch.setattr(retriever, "_targets_for_edge_types", lambda *args, **kwargs: ([], [], "local"))

        def _should_not_run(**kwargs):
            calls["reorder"] += 1
            return list(reversed(kwargs["selected_ids"]))

        monkeypatch.setattr(retriever, "_reorder_with_graph_support", _should_not_run)
        relation_plan = RelationQueryPlan(
            family="contact",
            confidence=0.9,
            primary_relation_types=tuple(),
            secondary_relation_types=tuple(),
            alias_tokens=tuple(),
            entity_tokens=("admissions",),
            graph_first=False,
        )
        result = retriever.augment_result(
            "How can I contact admissions?",
            cached_result,
            relation_plan=relation_plan,
            relation_candidates=RelationCandidateSet(confidence=0.9),
        )

        assert calls["reorder"] == 0
        assert result["selected_chunk_ids"][:2] == ["contact-chunk", "policy-chunk"]


class TestRoutedHybridRetriever:
    def test_from_config_returns_routed_retriever_when_enabled(self, tmp_dir, monkeypatch):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        TestGraphRAGRetriever()._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {"retriever_backend": "routed_hybrid"},
            "graph": {},
        }

        import pipeline.retrieval.adaptive_hybrid as mod

        monkeypatch.setattr(mod, "load_config", lambda _name: config)
        retriever = AdaptiveHybridRetriever.from_config(config_name="routed", work_dir=run_dir)
        assert retriever.__class__.__name__ == "RoutedHybridRetriever"

    def test_routed_retriever_runs_parallel_vector_and_graph_for_relation_queries(self, tmp_dir):
        from pipeline.retrieval.graph_rag import GraphQueryContext, RelationCandidateSet, RelationQueryPlan
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        TestGraphRAGRetriever()._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "routed_hybrid",
                "routed_graph_enabled": True,
                "routed_graph_query_types": ["fact"],
                "routed_graph_min_confidence": 0.2,
            },
            "graph": {},
        }
        retriever = RoutedHybridRetriever(config=config, work_dir=run_dir)
        retriever.vector.embed_query = lambda query: [0.1, 0.2]
        relation_plan = RelationQueryPlan(
            family="naming",
            confidence=0.95,
            primary_relation_types=("NAMED_AFTER",),
            secondary_relation_types=("ABOUT",),
            alias_tokens=("named", "after"),
            entity_tokens=("mbzuai",),
            graph_first=True,
        )
        retriever.graph._build_relation_query_plan = lambda query, mode, media_query: relation_plan
        vector_call = {}

        def retrieve_vector(query, query_vector=None, mode_override=None):
            vector_call["mode_override"] = mode_override
            return {
                "query": query,
                "mode": "fact",
                "selected_chunk_ids": ["chunk1"],
                "selected_parent_ids": ["page-a"],
                "selected_media_ids": [],
            }

        retriever.vector.retrieve = retrieve_vector
        retriever.graph.prepare_query_context = lambda query, **kwargs: GraphQueryContext(
            mode=kwargs.get("mode"),
            media_query=False,
            relation_plan=relation_plan,
            relation_candidates=RelationCandidateSet(
                family="naming",
                confidence=0.94,
                graph_first=True,
                chunk_ids=["chunk1"],
                parent_ids=["page-a"],
                fact_ids=["fact-1"],
            ),
            rewritten_query=f"{query} named after",
            rewrite_labels=("relation_alias_expansion",),
        )
        retriever.graph.augment_result = lambda query, result, relation_plan=None, relation_candidates=None: {
            **result,
            "graph_used": True,
            "graph_store_backend": "local",
        }

        result = retriever.retrieve("Whose name does MBZUAI carry?")

        assert result["routing_backend"] == "parallel_hybrid"
        assert result["routing_reason"] == "parallel_vector_graph"
        assert result["routing_relation_family"] == "naming"
        assert result["retriever_backend"] == "parallel_hybrid"
        assert result["graph_used"] is True
        assert result["routing_parallel_vector_graph"] is True
        assert "relation_alias_expansion" in result["query_rewrite_labels"]
        assert getattr(vector_call["mode_override"], "value", vector_call["mode_override"]) == "fact"

    def test_routed_retriever_runs_parallel_path_for_non_relation_queries(self, tmp_dir):
        from pipeline.retrieval.graph_rag import GraphQueryContext, RelationCandidateSet
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        TestGraphRAGRetriever()._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "routed_hybrid",
                "routed_graph_enabled": True,
                "routed_graph_query_types": ["fact"],
            },
            "graph": {},
        }
        retriever = RoutedHybridRetriever(config=config, work_dir=run_dir)
        retriever.vector.embed_query = lambda query: [0.1, 0.2]
        retriever.vector.retrieve = lambda query, query_vector=None, mode_override=None: {
            "query": query,
            "mode": "scoped",
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["page-a"],
            "selected_media_ids": [],
        }
        retriever.graph._build_relation_query_plan = lambda query, mode, media_query: None
        retriever.graph.prepare_query_context = lambda query, **kwargs: GraphQueryContext(
            mode=kwargs.get("mode"),
            media_query=False,
            relation_plan=None,
            relation_candidates=RelationCandidateSet(),
            rewritten_query=query,
            rewrite_labels=tuple(),
        )
        retriever.graph.augment_result = lambda query, result, relation_plan=None, relation_candidates=None: {
            **result,
            "graph_used": False,
            "graph_store_backend": "local",
        }

        result = retriever.retrieve("Summarize the MBZUAI campus for a visitor.")

        assert result["routing_backend"] == "parallel_hybrid"
        assert result["routing_reason"] == "parallel_vector_graph"
        assert result["retriever_backend"] == "parallel_hybrid"
        assert result["graph_used"] is False

    def test_routed_retriever_falls_back_to_vector_on_graph_error(self, tmp_dir):
        from pipeline.retrieval.graph_rag import RelationQueryPlan
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        TestGraphRAGRetriever()._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "routed_hybrid",
                "routed_graph_enabled": True,
                "routed_graph_query_types": ["fact"],
                "routed_fallback_to_vector": True,
            },
            "graph": {},
        }
        retriever = RoutedHybridRetriever(config=config, work_dir=run_dir)
        retriever.vector.embed_query = lambda query: [0.1, 0.2]
        retriever.vector.retrieve = lambda query, query_vector=None, mode_override=None: {
            "query": query,
            "mode": "fact",
            "selected_chunk_ids": ["chunk1"],
            "selected_parent_ids": ["page-a"],
            "selected_media_ids": [],
        }
        retriever.graph._build_relation_query_plan = lambda query, mode, media_query: RelationQueryPlan(
            family="location",
            confidence=0.9,
            primary_relation_types=("LOCATED_IN",),
            secondary_relation_types=("ABOUT",),
            alias_tokens=("located", "city"),
            entity_tokens=("mbzuai",),
            graph_first=True,
        )
        retriever.graph.prepare_query_context = lambda query, **kwargs: (_ for _ in ()).throw(
            RuntimeError("graph failure in /srv/private/releases/run-42 with secret-token")
        )

        result = retriever.retrieve("In which city is MBZUAI located?")

        assert result["routing_backend"] == "vector"
        assert result["routing_reason"] == "graph_error_fallback"
        assert result["routing_graph_error"] == "graph_context_failed"
        assert "/srv/private" not in str(result)

    def test_routed_retriever_routes_contact_lookup_queries_to_graph_backend(self, tmp_dir):
        from pipeline.retrieval.graph_rag import GraphQueryContext, RelationCandidateSet, RelationQueryPlan
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_contact_lookup_run(tmp_dir)
        TestGraphRAGRetriever()._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "routed_hybrid",
                "routed_graph_enabled": True,
                "routed_graph_query_types": ["fact"],
                "routed_graph_min_confidence": 0.1,
            },
            "graph": {},
        }
        retriever = RoutedHybridRetriever(config=config, work_dir=run_dir)
        retriever.vector.embed_query = lambda query: [0.1, 0.2]
        relation_plan = RelationQueryPlan(
            family="contact",
            confidence=0.91,
            primary_relation_types=("CONTACTS",),
            secondary_relation_types=("ABOUT",),
            alias_tokens=("contact", "email", "admissions"),
            entity_tokens=("admissions",),
            graph_first=False,
        )
        retriever.graph._build_relation_query_plan = lambda query, mode, media_query: relation_plan
        retriever.vector.retrieve = lambda query, query_vector=None, mode_override=None: {
            "query": query,
            "mode": "fact",
            "selected_chunk_ids": ["contact-chunk"],
            "selected_parent_ids": ["contact-page"],
            "selected_media_ids": [],
        }
        retriever.graph.prepare_query_context = lambda query, **kwargs: GraphQueryContext(
            mode=kwargs.get("mode"),
            media_query=False,
            relation_plan=relation_plan,
            relation_candidates=RelationCandidateSet(
                family="contact",
                confidence=0.88,
                graph_first=False,
                fact_ids=["fact-contact-general"],
            ),
            rewritten_query=f"{query} contact email admissions",
            rewrite_labels=("semantic_alias_expansion", "relation_alias_expansion"),
        )
        retriever.graph.augment_result = lambda query, result, relation_plan=None, relation_candidates=None: {
            **result,
            "graph_used": True,
            "graph_store_backend": "local",
        }

        result = retriever.retrieve("What is the admissions committee email address?")

        assert result["routing_backend"] == "parallel_hybrid"
        assert result["routing_reason"] == "parallel_vector_graph"
        assert result["routing_relation_family"] == "contact"
        assert result["routing_parallel_vector_graph"] is True
        assert "relation_alias_expansion" in result["query_rewrite_labels"]

    def test_routed_retriever_does_not_expand_generic_contact_queries(self, tmp_dir):
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_contact_lookup_run(tmp_dir)
        TestGraphRAGRetriever()._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "routed_hybrid",
                "routed_graph_enabled": True,
                "parallel_query_rewriting_enabled": True,
            },
            "graph": {},
        }
        retriever = RoutedHybridRetriever(config=config, work_dir=run_dir)

        bundle = retriever._build_query_rewrite_bundle(
            "How can I contact admissions?",
            relation_plan=None,
            query_mode="fact",
        )

        assert bundle.vector_query == "How can I contact admissions?"
        assert bundle.graph_query == "How can I contact admissions?"
        assert bundle.labels == tuple()

    def test_routed_retriever_uses_support_hours_rewrites_for_support_hours_queries(self, tmp_dir):
        from pipeline.retrieval.graph_rag import RelationQueryPlan
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

        run_dir = TestAdaptiveHybridRetriever()._build_contact_lookup_run(tmp_dir)
        TestGraphRAGRetriever()._write_promoted_graph_artifacts(run_dir)
        config = {
            "embedder": {"pinecone_index": "idx", "model": "gemini-embedding-2-preview", "output_dimensionality": 2},
            "retrieval": {
                "retriever_backend": "routed_hybrid",
                "routed_graph_enabled": True,
                "parallel_query_rewriting_enabled": True,
            },
            "graph": {},
        }
        retriever = RoutedHybridRetriever(config=config, work_dir=run_dir)
        relation_plan = RelationQueryPlan(
            family="hours",
            confidence=0.82,
            primary_relation_types=("OFFERS",),
            secondary_relation_types=("ABOUT",),
            alias_tokens=("working", "time", "times", "support", "technical", "it", "screening", "exam"),
            entity_tokens=("mbzuai", "it"),
            graph_first=False,
        )

        bundle = retriever._build_query_rewrite_bundle(
            "What are the IT support working hours for the MBZUAI online screening exam?",
            relation_plan=relation_plan,
            query_mode="fact",
        )

        assert "official" not in bundle.vector_query.lower()
        assert "official" not in bundle.graph_query.lower()
        assert "support" in bundle.vector_query.lower()
        assert "screening" in bundle.graph_query.lower()


class TestQueryAliasExpansion:
    def test_support_hours_queries_expand_to_support_not_official_terms(self):
        from pipeline.retrieval.adaptive_hybrid import _hours_query_alias_tokens, _semantic_query_alias_tokens

        query = "What are the IT support working hours for the MBZUAI online screening exam?"

        hour_aliases = _hours_query_alias_tokens(query)
        semantic_aliases = _semantic_query_alias_tokens(query)

        assert "support" in hour_aliases
        assert "technical" in hour_aliases
        assert "screening" in hour_aliases
        assert "official" not in hour_aliases
        assert "official" not in semantic_aliases


class TestNeo4jGraphStoreEmbedder:
    def test_query_count_supports_neo4j_query_api_values_shape(self):
        from pipeline.stages.embedders.neo4j_graph_store import _query_count

        assert _query_count({"data": {"fields": ["count"], "values": [[12]]}}) == 12

    def test_filter_graph_bundle_for_neo4j_keeps_compact_retrieval_subgraph(self):
        from pipeline.stages.embedders.neo4j_graph_store import _filter_graph_bundle_for_neo4j

        bundle = {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "nodes": [
                {"id": "doc1", "node_type": "document", "label": "Document"},
                {"id": "chunk1", "node_type": "chunk", "label": "Chunk"},
                {"id": "fact1", "node_type": "fact", "label": "Fact"},
                {"id": "entity1", "node_type": "entity", "label": "Entity"},
            ],
            "edges": [
                {"id": "e-doc", "edge_type": "DOCUMENT_HAS_CHUNK", "source_id": "doc1", "target_id": "chunk1"},
                {"id": "e-fact", "edge_type": "CHUNK_HAS_FACT", "source_id": "chunk1", "target_id": "fact1"},
                {"id": "e-entity", "edge_type": "FACT_MENTIONS_ENTITY", "source_id": "fact1", "target_id": "entity1"},
            ],
        }

        filtered = _filter_graph_bundle_for_neo4j(
            bundle,
            compact_upload=True,
            allowed_node_types=["chunk", "fact", "entity"],
            allowed_edge_types=["CHUNK_HAS_FACT", "FACT_MENTIONS_ENTITY"],
        )

        assert {node["id"] for node in filtered["nodes"]} == {"chunk1", "fact1", "entity1"}
        assert {edge["id"] for edge in filtered["edges"]} == {"e-fact", "e-entity"}

    def test_execute_syncs_graph_bundle_via_query_api(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.embedders.neo4j_graph_store import Neo4jGraphStoreEmbedder

        graph_file = tmp_dir / "promoted_knowledge_graph.json"
        atomic_write_json(
            graph_file,
            {
                "schema_version": 2,
                "graph_type": "promoted_semantic_graph",
                "nodes": [
                    {
                        "id": "chunk1",
                        "node_type": "chunk",
                        "label": "Chunk",
                        "properties": {"source_url": "https://example.com", "page_numbers": [1]},
                    },
                    {
                        "id": "entity:mbzuai",
                        "node_type": "entity",
                        "label": "MBZUAI",
                        "properties": {"entity_type": "Organization"},
                    },
                ],
                "edges": [
                    {
                        "id": "edge1",
                        "edge_type": "CHUNK_MENTIONS_ENTITY",
                        "source_id": "chunk1",
                        "target_id": "entity:mbzuai",
                        "properties": {"confidence": 0.9},
                    }
                ],
                "stats": {
                    "node_count": 2,
                    "edge_count": 1,
                    "node_type_counts": {"chunk": 1, "entity": 1},
                    "edge_type_counts": {"CHUNK_MENTIONS_ENTITY": 1},
                },
            },
        )

        calls = []

        class DummyResponse:
            def __init__(self):
                self.status_code = 202
                self.text = "{}"
                self.content = b"{}"

            def json(self):
                return {}

        def fake_post(url, auth=None, headers=None, json=None, timeout=None):
            calls.append(
                {
                    "url": url,
                    "auth": auth,
                    "headers": headers,
                    "json": json,
                    "timeout": timeout,
                }
            )
            return DummyResponse()

        monkeypatch.setattr("pipeline.stages.embedders.neo4j_graph_store.requests.post", fake_post)

        ctx = StageContext(
            run_id="r-neo4j",
            project_name="mbzuai",
            config={
                "graph": {
                    "neo4j_uri": "https://neo4j.example",
                    "neo4j_database": "neo4j",
                    "neo4j_username": "neo4j",
                    "neo4j_password": "secret",
                    "neo4j_namespace": "mbzuai-graph",
                    "neo4j_load_batch_size": 1,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={"knowledge_graph_file": str(graph_file)},
            stage_definition={"type": "embedder", "plugin": "neo4j_graph_store"},
            stage_id="upload_graph",
        )

        result = run_async(Neo4jGraphStoreEmbedder().execute(ctx))
        manifest = load_json_safe(result.outputs["neo4j_graph_manifest_file"], {})

        assert result.metrics["neo4j_nodes_synced"] == 2
        assert result.metrics["neo4j_edges_synced"] == 1
        assert manifest["neo4j_namespace"] == "mbzuai-graph"
        assert any("CREATE CONSTRAINT kg_node_key" in call["json"]["statement"] for call in calls)
        assert any("DETACH DELETE n" in call["json"]["statement"] for call in calls)
        assert any("MERGE (n:KGNode {key: row.key})" in call["json"]["statement"] for call in calls)
        assert any("MERGE (source)-[r:KG_EDGE {key: row.key}]->(target)" in call["json"]["statement"] for call in calls)

    def test_execute_defaults_namespace_to_project_and_run_id(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.embedders.neo4j_graph_store import Neo4jGraphStoreEmbedder

        graph_file = tmp_dir / "promoted_knowledge_graph.json"
        atomic_write_json(
            graph_file,
            {
                "schema_version": 2,
                "graph_type": "promoted_semantic_graph",
                "nodes": [
                    {
                        "id": "chunk1",
                        "node_type": "chunk",
                        "label": "Chunk",
                        "properties": {"source_url": "https://example.com"},
                    }
                ],
                "edges": [],
                "stats": {
                    "node_count": 1,
                    "edge_count": 0,
                    "node_type_counts": {"chunk": 1},
                    "edge_type_counts": {},
                },
            },
        )

        class DummyResponse:
            def __init__(self):
                self.status_code = 202
                self.text = "{}"
                self.content = b"{}"

            def json(self):
                return {}

        monkeypatch.setattr(
            "pipeline.stages.embedders.neo4j_graph_store.requests.post",
            lambda *args, **kwargs: DummyResponse(),
        )

        ctx = StageContext(
            run_id="run-123",
            project_name="mbzuai",
            config={
                "graph": {
                    "neo4j_uri": "https://neo4j.example",
                    "neo4j_database": "neo4j",
                    "neo4j_username": "neo4j",
                    "neo4j_password": "secret",
                }
            },
            work_dir=tmp_dir,
            previous_outputs={"knowledge_graph_file": str(graph_file)},
            stage_definition={"type": "embedder", "plugin": "neo4j_graph_store"},
            stage_id="upload_graph",
        )

        result = run_async(Neo4jGraphStoreEmbedder().execute(ctx))
        manifest = load_json_safe(result.outputs["neo4j_graph_manifest_file"], {})

        assert manifest["neo4j_namespace"] == "mbzuai:run-123"

    def test_execute_uses_env_database_when_config_does_not_set_one(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.embedders.neo4j_graph_store import Neo4jGraphStoreEmbedder

        graph_file = tmp_dir / "promoted_knowledge_graph.json"
        atomic_write_json(
            graph_file,
            {
                "schema_version": 2,
                "graph_type": "promoted_semantic_graph",
                "nodes": [
                    {"id": "chunk1", "node_type": "chunk", "label": "Chunk", "properties": {}},
                ],
                "edges": [],
                "stats": {
                    "node_count": 1,
                    "edge_count": 0,
                    "node_type_counts": {"chunk": 1},
                    "edge_type_counts": {},
                },
            },
        )

        calls = []

        class DummyResponse:
            def __init__(self):
                self.status_code = 202
                self.text = "{}"
                self.content = b"{}"

            def json(self):
                return {}

        def fake_post(url, auth=None, headers=None, json=None, timeout=None):
            calls.append(url)
            return DummyResponse()

        monkeypatch.setattr("pipeline.stages.embedders.neo4j_graph_store.requests.post", fake_post)
        monkeypatch.setenv("NEO4J_DATABASE", "aura-db")

        ctx = StageContext(
            run_id="run-neo4j-env-db",
            project_name="mbzuai",
            config={
                "graph": {
                    "neo4j_uri": "https://neo4j.example",
                    "neo4j_database": "",
                    "neo4j_username": "neo4j",
                    "neo4j_password": "secret",
                }
            },
            work_dir=tmp_dir,
            previous_outputs={"knowledge_graph_file": str(graph_file)},
            stage_definition={"type": "embedder", "plugin": "neo4j_graph_store"},
            stage_id="upload_graph",
        )

        result = run_async(Neo4jGraphStoreEmbedder().execute(ctx))
        manifest = load_json_safe(result.outputs["neo4j_graph_manifest_file"], {})

        assert any("/db/aura-db/query/v2" in url for url in calls)
        assert manifest["neo4j_database"] == "aura-db"

    def test_execute_can_clear_all_kg_nodes_before_upload(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.neo4j_graph_store import Neo4jGraphStoreEmbedder

        graph_file = tmp_dir / "promoted_knowledge_graph_clear.json"
        atomic_write_json(
            graph_file,
            {
                "schema_version": 2,
                "graph_type": "promoted_semantic_graph",
                "nodes": [{"id": "chunk1", "node_type": "chunk", "label": "Chunk", "properties": {}}],
                "edges": [],
                "stats": {
                    "node_count": 1,
                    "edge_count": 0,
                    "node_type_counts": {"chunk": 1},
                    "edge_type_counts": {},
                },
            },
        )

        statements = []

        class DummyResponse:
            def __init__(self):
                self.status_code = 202
                self.text = "{}"
                self.content = b"{}"

            def json(self):
                return {}

        def fake_post(url, auth=None, headers=None, json=None, timeout=None):
            statements.append(json["statement"])
            return DummyResponse()

        monkeypatch.setattr("pipeline.stages.embedders.neo4j_graph_store.requests.post", fake_post)

        ctx = StageContext(
            run_id="run-clear-all",
            project_name="mbzuai",
            config={
                "graph": {
                    "neo4j_uri": "https://neo4j.example",
                    "neo4j_database": "neo4j",
                    "neo4j_username": "neo4j",
                    "neo4j_password": "secret",
                    "neo4j_clear_all_kg_nodes_on_zero_progress": True,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={"knowledge_graph_file": str(graph_file)},
            stage_definition={"type": "embedder", "plugin": "neo4j_graph_store"},
            stage_id="upload_graph",
        )

        result = run_async(Neo4jGraphStoreEmbedder().execute(ctx))

        assert result.status.value == "completed"
        assert any(statement == "MATCH (n:KGNode) DETACH DELETE n" for statement in statements)


# ─────────────────────────────────────────────────────────────
# 15. Modular Artifact Contracts — stage-scoped config + artifacts
# ─────────────────────────────────────────────────────────────

class TestArtifactContracts:
    def test_artifact_catalog_roundtrip(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record, load_artifact_catalog, save_artifact_catalog

        catalog = ArtifactCatalog()
        catalog.add(
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_html",
                uri=(tmp_dir / "doc.md").resolve().as_uri(),
                local_path=tmp_dir / "doc.md",
                metadata={"source_url": "https://example.com"},
            )
        )

        save_artifact_catalog(catalog, tmp_dir)
        loaded = load_artifact_catalog(tmp_dir)
        assert len(loaded.records) == 1
        assert loaded.records[0].artifact_type == "markdown"
        assert loaded.records[0].metadata["source_url"] == "https://example.com"

    def test_artifact_catalog_prunes_missing_local_paths(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record

        existing = tmp_dir / "doc.md"
        existing.write_text("ok", encoding="utf-8")
        missing = tmp_dir / "missing.md"

        catalog = ArtifactCatalog()
        catalog.add(
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_html",
                uri=existing.resolve().as_uri(),
                local_path=existing,
            )
        )
        catalog.add(
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_html",
                uri=missing.resolve().as_uri(),
                local_path=missing,
            )
        )

        removed = catalog.prune_missing_local_paths()
        assert len(removed) == 1
        assert len(catalog.records) == 1
        assert catalog.records[0].local_path == str(existing.resolve())

    def test_stage_context_applies_stage_scoped_config_overlay(self, tmp_dir):
        from pipeline.core.base import StageContext

        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"converter": {"overwrite": False, "max_workers": 4}},
            work_dir=tmp_dir,
            stage_definition={
                "type": "converter",
                "plugin": "markitdown",
                "config": {"overwrite": True},
            },
        )

        assert ctx.converter_config["overwrite"] is True
        assert ctx.converter_config["max_workers"] == 4

    def test_orchestrator_persists_stage_artifacts(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageResult
        from pipeline.core.orchestrator import PipelineOrchestrator
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.artifacts import load_artifact_catalog

        artifact_file = tmp_dir / "artifact.txt"

        @register_stage
        class ArtifactStage(PipelineStage):
            name = "artifact_stage"
            stage_type = "test_artifact"
            description = "Publishes an artifact"

            async def execute(self, ctx):
                artifact_file.write_text("ok", encoding="utf-8")
                return StageResult.success(
                    artifacts=[
                        ctx.make_artifact(
                            artifact_file,
                            artifact_type="text_file",
                            role="output",
                            metadata={"kind": "test"},
                        )
                    ]
                )

        try:
            state = run_async(
                PipelineOrchestrator(
                    {
                        "project_name": "test",
                        "stages": [{"type": "test_artifact", "plugin": "artifact_stage", "id": "artifact_stage_1"}],
                    },
                    work_dir=tmp_dir / "run",
                ).run()
            )
            assert state.status == "completed"
            catalog = load_artifact_catalog(tmp_dir / "run")
            assert len(catalog.records) == 1
            assert catalog.records[0].artifact_type == "text_file"
            assert state.stages[0].artifact_ids
        finally:
            _REGISTRY.pop("test_artifact", None)


class TestRunAudit:
    def test_reconcile_state_artifact_ids_prunes_missing_ids(self):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.run_audit import reconcile_state_artifact_ids
        from pipeline.core.state import PipelineState, StageState

        catalog = ArtifactCatalog()
        existing = build_artifact_record(
            artifact_type="markdown",
            role="content",
            producer_stage="convert_html",
            uri="file:///tmp/doc.md",
            local_path="/tmp/doc.md",
        )
        catalog.add(existing)

        state = PipelineState(
            run_id="r1",
            project_name="p1",
            stages=[
                StageState(
                    name="markitdown",
                    stage_type="converter",
                    status="completed",
                    stage_id="convert_html",
                    artifact_ids=[existing.artifact_id, "missing-artifact-id"],
                )
            ],
        )

        removed = reconcile_state_artifact_ids(state, catalog)
        assert removed == 1
        assert state.stages[0].artifact_ids == [existing.artifact_id]

    def test_audit_allows_pruned_crawler_intermediates_after_downstream_stage(self, tmp_dir):
        from pipeline.core.io import atomic_write_json
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, StageState, save_state

        run_dir = tmp_dir / "run"
        run_dir.mkdir()
        cleaned_dir = run_dir / "stage_outputs" / "clean_html" / "cleaned_html"
        cleaned_dir.mkdir(parents=True)
        source_file = run_dir / "stage_outputs" / "score_raw_content" / "accepted_html" / "page.html"
        source_file.parent.mkdir(parents=True)
        source_file.write_text("<html><body>Source content</body></html>", encoding="utf-8")
        cleaned_file = cleaned_dir / "page.html"
        cleaned_file.write_text("<html><body>Cleaned content</body></html>", encoding="utf-8")
        manifest_file = run_dir / "stage_outputs" / "clean_html" / "cleaning_manifest.json"
        atomic_write_json(
            manifest_file,
            {
                "schema_version": 1,
                "application_status": "completed",
                "gate": {
                    "ok": True,
                    "input_count": 1,
                    "accepted_count": 1,
                    "filtered_count": 0,
                    "failed_count": 0,
                },
                "dispositions": [
                    {
                        "source_path": str(source_file),
                        "output_path": str(cleaned_file),
                        "status": "accepted",
                        "reason_code": "accepted_trafilatura",
                    }
                ],
            },
        )

        save_state(
            PipelineState(
                run_id="r1",
                project_name="p1",
                status="completed",
                stages=[
                    StageState(
                        name="crawl4ai",
                        stage_type="crawler",
                        stage_id="crawl_web",
                        status="completed",
                        outputs={
                            "html_dir": str(run_dir / "html"),
                            "images_dir": str(run_dir / "downloaded_page_images"),
                        },
                    ),
                    StageState(
                        name="trafilatura",
                        stage_type="cleaner",
                        stage_id="clean_html",
                        status="completed",
                        outputs={
                            "cleaned_dir": str(cleaned_dir),
                            "cleaned_count": 1,
                            "cleaning_manifest_file": str(manifest_file),
                        },
                    ),
                ],
                current_stage_index=2,
            ),
            run_dir,
        )

        report = audit_run(run_dir)
        assert report.ok
        assert any(issue.code == "pruned_intermediate_dir" for issue in report.warnings)

    def test_audit_requires_crawler_intermediates_before_downstream_stage(self, tmp_dir):
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, StageState, save_state

        run_dir = tmp_dir / "run"
        run_dir.mkdir()

        save_state(
            PipelineState(
                run_id="r1",
                project_name="p1",
                status="completed",
                stages=[
                    StageState(
                        name="crawl4ai",
                        stage_type="crawler",
                        stage_id="crawl_web",
                        status="completed",
                        outputs={"html_dir": str(run_dir / "html")},
                    )
                ],
                current_stage_index=1,
            ),
            run_dir,
        )

        report = audit_run(run_dir)
        assert not report.ok
        assert any(issue.code == "missing_output_dir" for issue in report.errors)

    def test_audit_rejects_completed_cleaner_without_disposition_manifest(self, tmp_dir):
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, StageState, save_state

        run_dir = tmp_dir / "run"
        cleaned_dir = run_dir / "stage_outputs" / "clean_html" / "cleaned_html"
        cleaned_dir.mkdir(parents=True)
        save_state(
            PipelineState(
                run_id="r1",
                project_name="p1",
                status="completed",
                current_stage_index=1,
                stages=[
                    StageState(
                        name="trafilatura",
                        stage_type="cleaner",
                        stage_id="clean_html",
                        status="completed",
                        outputs={"cleaned_dir": str(cleaned_dir), "cleaned_count": 0},
                    )
                ],
            ),
            run_dir,
        )

        report = audit_run(run_dir)

        assert not report.ok
        assert any(issue.code == "missing_cleaning_manifest" for issue in report.errors)

    def test_audit_detects_accepted_report_with_missing_markdown(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record, save_artifact_catalog
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, StageState, save_state

        run_dir = tmp_dir / "run"
        run_dir.mkdir()
        report_path = run_dir / "stage_outputs" / "convert_documents" / "quality_reports" / "doc.validation.json"
        report_path.parent.mkdir(parents=True)
        missing_markdown = run_dir / "stage_outputs" / "convert_documents" / "markdown" / "doc.md"
        report_path.write_text(
            json.dumps(
                {
                    "source_file": "/tmp/doc.pdf",
                    "selected_backend": "docling",
                    "selected_markdown_path": str(missing_markdown),
                    "quarantined": False,
                    "selected_assessment": {"accepted": True, "score": 0.95},
                }
            ),
            encoding="utf-8",
        )

        catalog = ArtifactCatalog()
        report_record = build_artifact_record(
            artifact_type="document_quality_report",
            role="quality_report",
            producer_stage="convert_documents",
            uri=report_path.resolve().as_uri(),
            local_path=report_path,
            metadata={
                "accepted": True,
                "selected_backend": "docling",
                "selected_markdown_path": str(missing_markdown),
            },
        )
        catalog.add(report_record)
        save_artifact_catalog(catalog, run_dir)

        save_state(
            PipelineState(
                run_id="r1",
                project_name="p1",
                status="completed",
                stages=[
                    StageState(
                        name="docling",
                        stage_type="converter",
                        stage_id="convert_documents",
                        status="completed",
                        artifact_ids=[report_record.artifact_id],
                        outputs={"quality_reports_dir": str(report_path.parent)},
                    )
                ],
                current_stage_index=1,
            ),
            run_dir,
        )

        report = audit_run(run_dir)
        assert not report.ok
        assert any(issue.code == "accepted_report_missing_markdown_target" for issue in report.errors)

    def test_audit_detects_chunk_manifest_with_missing_source_markdown(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record, save_artifact_catalog
        from pipeline.core.io import atomic_write_json
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, StageState, save_state

        run_dir = tmp_dir / "run"
        run_dir.mkdir()
        chunk_file = run_dir / "stage_outputs" / "chunk_content" / "chunks" / "chunk_index.json"
        chunk_file.parent.mkdir(parents=True)
        atomic_write_json(
            chunk_file,
            {
                "version": 1,
                "strategy": "hybrid",
                "document_count": 1,
                "chunk_count": 1,
                "documents": [
                    {
                        "document_id": "doc1",
                        "source_markdown_path": str(run_dir / "missing.md"),
                        "chunk_count": 1,
                    }
                ],
                "chunks": [
                    {
                        "document_id": "doc1",
                        "chunk_id": "doc1::chunk::001:test",
                        "chunk_index": 0,
                        "text": "content",
                        "source_markdown_path": str(run_dir / "missing.md"),
                    }
                ],
            },
        )

        catalog = ArtifactCatalog()
        chunk_record = build_artifact_record(
            artifact_type="chunk_index",
            role="retrieval_chunks",
            producer_stage="chunk_content",
            uri=chunk_file.resolve().as_uri(),
            local_path=chunk_file,
        )
        catalog.add(chunk_record)
        save_artifact_catalog(catalog, run_dir)

        save_state(
            PipelineState(
                run_id="r1",
                project_name="p1",
                status="completed",
                stages=[
                    StageState(
                        name="hybrid",
                        stage_type="chunker",
                        stage_id="chunk_content",
                        status="completed",
                        artifact_ids=[chunk_record.artifact_id],
                        outputs={"chunks_file": str(chunk_file)},
                    )
                ],
                current_stage_index=1,
            ),
            run_dir,
        )

        report = audit_run(run_dir)
        assert not report.ok
        assert any(issue.code == "chunk_missing_source_markdown_file" for issue in report.errors)

    def test_audit_detects_knowledge_graph_edge_with_missing_target_node(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record, save_artifact_catalog
        from pipeline.core.io import atomic_write_json
        from pipeline.core.run_audit import audit_run
        from pipeline.core.state import PipelineState, StageState, save_state

        run_dir = tmp_dir / "run"
        run_dir.mkdir()
        graph_file = run_dir / "stage_outputs" / "format_graph" / "knowledge_graph.json"
        graph_file.parent.mkdir(parents=True)
        atomic_write_json(
            graph_file,
            {
                "schema_version": 1,
                "graph_type": "deterministic_content_graph",
                "nodes": [
                    {"id": "doc1", "node_type": "document", "label": "Doc 1"},
                ],
                "edges": [
                    {
                        "id": "edge1",
                        "edge_type": "HAS_PAGE",
                        "source_id": "doc1",
                        "target_id": "missing-page",
                    }
                ],
            },
        )

        catalog = ArtifactCatalog()
        graph_record = build_artifact_record(
            artifact_type="knowledge_graph_bundle",
            role="knowledge_graph",
            producer_stage="format_graph",
            uri=graph_file.resolve().as_uri(),
            local_path=graph_file,
        )
        catalog.add(graph_record)
        save_artifact_catalog(catalog, run_dir)

        save_state(
            PipelineState(
                run_id="r1",
                project_name="p1",
                status="completed",
                stages=[
                    StageState(
                        name="knowledge_graph",
                        stage_type="formatter",
                        stage_id="format_graph",
                        status="completed",
                        artifact_ids=[graph_record.artifact_id],
                        outputs={"knowledge_graph_file": str(graph_file)},
                    )
                ],
                current_stage_index=1,
            ),
            run_dir,
        )

        report = audit_run(run_dir)
        assert not report.ok
        assert any(issue.code == "graph_edge_missing_target_node" for issue in report.errors)

    def test_orchestrator_fails_closed_when_audit_detects_bad_artifact(self, tmp_dir):
        from pipeline.core.base import PipelineStage, StageResult
        from pipeline.core.orchestrator import PipelineOrchestrator
        from pipeline.core.registry import _REGISTRY, register_stage
        from pipeline.core.run_audit import audit_run

        @register_stage
        class BadReportStage(PipelineStage):
            name = "bad_report_stage"
            stage_type = "test_audit"
            description = "Publishes an accepted report that points to missing markdown"

            async def execute(self, ctx):
                report_path = ctx.output_dir("quality_reports") / "bad.validation.json"
                report_path.write_text(
                    json.dumps(
                        {
                            "source_file": "/tmp/doc.pdf",
                            "selected_backend": "docling",
                            "selected_markdown_path": str(ctx.stage_work_dir / "markdown" / "missing.md"),
                            "quarantined": False,
                            "selected_assessment": {"accepted": True, "score": 0.99},
                        }
                    ),
                    encoding="utf-8",
                )
                return StageResult.success(
                    outputs={"quality_reports_dir": str(report_path.parent)},
                    artifacts=[
                        ctx.make_artifact(
                            report_path,
                            artifact_type="document_quality_report",
                            role="quality_report",
                            metadata={
                                "accepted": True,
                                "selected_backend": "docling",
                                "selected_markdown_path": str(ctx.stage_work_dir / "markdown" / "missing.md"),
                            },
                        )
                    ],
                )

        try:
            run_dir = tmp_dir / "run"
            state = run_async(
                PipelineOrchestrator(
                    {
                        "project_name": "test",
                        "pipeline": {
                            "audit_on_stage_complete": True,
                            "audit_on_run_complete": True,
                            "fail_on_audit_error": True,
                        },
                        "stages": [{"type": "test_audit", "plugin": "bad_report_stage", "id": "bad_stage"}],
                    },
                    work_dir=run_dir,
                ).run()
            )
            assert state.status == "failed"
            assert state.stages[0].status == "failed"
            assert "Integrity audit failed" in (state.stages[0].error_message or "")

            report = audit_run(run_dir)
            assert not report.ok
            assert any(issue.code == "accepted_report_missing_markdown_target" for issue in report.errors)
        finally:
            _REGISTRY.pop("test_audit", None)


class TestCleanerContracts:
    def test_trafilatura_falls_back_to_bs4_and_writes_out_of_place(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        source_file = html_dir / "page.html"
        source_file.write_text("<html><body><main><h1>Title</h1><p>" + ("Content " * 40) + "</p></main></body></html>")

        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(extract=lambda *args, **kwargs: ""),
        )

        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"cleaner": {"min_content_length": 10, "preserve_embedded_media": True}},
            work_dir=tmp_dir,
            previous_outputs={"html_dir": str(html_dir)},
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
        )

        result = run_async(TrafilaturaCleaner().execute(ctx))
        cleaned_file = Path(result.outputs["cleaned_dir"]) / "page.html"
        assert cleaned_file.exists()
        assert source_file.exists()
        assert cleaned_file != source_file
        assert result.metrics["fallback_cleaned"] == 1

    def test_trafilatura_falls_back_when_extraction_is_navigation_heavy(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        source_file = html_dir / "page.html"
        source_file.write_text(
            "<html><body><nav><ul><li>About</li><li>Study</li><li>Research</li></ul></nav>"
            "<main><h1>Real Content</h1><p>" + ("Important body text " * 40) + "</p></main></body></html>",
            encoding="utf-8",
        )

        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(
                extract=lambda *args, **kwargs: (
                    "<html><body><ul>"
                    + "".join(f"<li><a href='/item-{idx}'>Menu {idx}</a></li>" for idx in range(20))
                    + "</ul></body></html>"
                )
            ),
        )

        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"cleaner": {"min_content_length": 10, "preserve_embedded_media": True}},
            work_dir=tmp_dir,
            previous_outputs={"html_dir": str(html_dir)},
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
        )

        result = run_async(TrafilaturaCleaner().execute(ctx))
        cleaned_file = Path(result.outputs["cleaned_dir"]) / "page.html"
        cleaned = cleaned_file.read_text(encoding="utf-8")
        assert "Real Content" in cleaned
        assert "Menu 1" not in cleaned
        assert result.metrics["fallback_cleaned"] == 1

    def test_trafilatura_fails_closed_on_empty_input(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import load_json_safe
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(extract=lambda *args, **kwargs: ""),
        )
        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"cleaner": {"min_content_length": 10}},
            work_dir=tmp_dir,
            previous_outputs={"html_dir": str(html_dir)},
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
        )

        result = run_async(TrafilaturaCleaner().execute(ctx))

        assert result.status == StageStatus.FAILED
        manifest = load_json_safe(result.outputs["cleaning_manifest_file"])
        assert manifest["gate"]["ok"] is False
        assert {item["code"] for item in manifest["gate"]["failures"]} == {
            "empty_input"
        }

    def test_trafilatura_fails_closed_when_every_page_is_filtered(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import load_json_safe
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        (html_dir / "page.html").write_text(
            "<html><body><script>only javascript</script></body></html>",
            encoding="utf-8",
        )
        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(extract=lambda *args, **kwargs: ""),
        )
        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"cleaner": {"min_content_length": 100}},
            work_dir=tmp_dir,
            previous_outputs={"html_dir": str(html_dir)},
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
        )

        result = run_async(TrafilaturaCleaner().execute(ctx))

        assert result.status == StageStatus.FAILED
        assert result.outputs["cleaned_count"] == 0
        manifest = load_json_safe(result.outputs["cleaning_manifest_file"])
        assert manifest["gate"]["reason_counts"] == {"empty_content": 1}
        assert "zero_output" in {
            item["code"] for item in manifest["gate"]["failures"]
        }

    def test_trafilatura_threshold_uses_visible_text_not_markup(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import load_json_safe
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        (html_dir / "page.html").write_text("<html><body>x</body></html>", encoding="utf-8")
        markup_heavy = "<html><body>" + ("<span></span>" * 30) + "x</body></html>"
        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(extract=lambda *args, **kwargs: markup_heavy),
        )
        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"cleaner": {"min_content_length": 100}},
            work_dir=tmp_dir,
            previous_outputs={"html_dir": str(html_dir)},
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
        )

        result = run_async(TrafilaturaCleaner().execute(ctx))

        assert result.status == StageStatus.FAILED
        manifest = load_json_safe(result.outputs["cleaning_manifest_file"])
        disposition = manifest["dispositions"][0]
        assert disposition["status"] == "filtered"
        assert disposition["reason_code"] == "insufficient_visible_content"
        assert disposition["content_metrics"]["visible_characters"] == 1

    def test_trafilatura_resets_stale_output_before_retry(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        (html_dir / "page.html").write_text(
            "<html><body><main>" + ("Source content " * 20) + "</main></body></html>",
            encoding="utf-8",
        )
        stale_file = tmp_dir / "stage_outputs" / "clean_html" / "cleaned_html" / "stale.html"
        stale_file.parent.mkdir(parents=True)
        stale_file.write_text("stale", encoding="utf-8")
        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(
                extract=lambda *args, **kwargs: (
                    "<html><body><main>" + ("Accepted content " * 20) + "</main></body></html>"
                )
            ),
        )
        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"cleaner": {"min_content_length": 100}},
            work_dir=tmp_dir,
            previous_outputs={"html_dir": str(html_dir)},
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
        )

        result = run_async(TrafilaturaCleaner().execute(ctx))

        assert result.status == StageStatus.COMPLETED
        assert not stale_file.exists()
        assert (Path(result.outputs["cleaned_dir"]) / "page.html").exists()

    def test_trafilatura_validate_config_rejects_invalid_policy(self):
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        errors = run_async(
            TrafilaturaCleaner().validate_config(
                {
                    "cleaner": {
                        "min_content_length": "bad",
                        "minimum_retention_ratio": 1.1,
                        "fail_on_empty_input": "yes",
                    }
                }
            )
        )

        assert "cleaner.min_content_length must be a non-negative integer" in errors
        assert "cleaner.minimum_retention_ratio must be between 0 and 1" in errors
        assert "cleaner.fail_on_empty_input must be a boolean" in errors

    def test_trafilatura_requires_critical_url_to_survive_cleaning(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        source_file = html_dir / "page.html"
        source_file.write_text(
            "<html><body><main>" + ("Public content " * 20) + "</main></body></html>",
            encoding="utf-8",
        )
        mapping_file = tmp_dir / "mapping.json"
        atomic_write_json(mapping_file, {"https://example.com/other": str(source_file)})
        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(
                extract=lambda *args, **kwargs: (
                    "<html><body><main>" + ("Public content " * 20) + "</main></body></html>"
                )
            ),
        )
        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={
                "cleaner": {"min_content_length": 10},
                "formatter": {"critical_url_patterns": [r"/critical/?$"]},
            },
            work_dir=tmp_dir,
            previous_outputs={
                "html_dir": str(html_dir),
                "mapping_file": str(mapping_file),
            },
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
        )

        result = run_async(TrafilaturaCleaner().execute(ctx))

        assert result.status == StageStatus.FAILED
        manifest = load_json_safe(result.outputs["cleaning_manifest_file"])
        assert manifest["gate"]["missing_critical_url_patterns"] == [r"/critical/?$"]
        assert "missing_critical_url_after_cleaning" in {
            item["code"] for item in manifest["gate"]["failures"]
        }

    def test_quality_to_cleaner_artifact_lineage_is_preserved(self, tmp_dir, monkeypatch):
        from pipeline.core.artifacts import ArtifactCatalog
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.cleaners.trafilatura_cleaner import TrafilaturaCleaner
        from pipeline.stages.quality.quality_scorer import QualityScorer

        html_dir = tmp_dir / "html"
        html_dir.mkdir()
        source_file = html_dir / "page.html"
        source_file.write_text(
            "<html><body><main>" + ("Lineage content " * 20) + "</main></body></html>",
            encoding="utf-8",
        )
        mapping_file = tmp_dir / "mapping.json"
        atomic_write_json(mapping_file, {"https://example.com/page": str(source_file)})
        config = {
            "quality": {"min_content_length": 10, "minimum_retention_ratio": 1.0},
            "cleaner": {"min_content_length": 10},
        }
        catalog = ArtifactCatalog()
        quality_ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config=config,
            work_dir=tmp_dir,
            previous_outputs={
                "html_dir": str(html_dir),
                "mapping_file": str(mapping_file),
            },
            stage_definition={"type": "quality_gate", "plugin": "quality_scorer"},
            stage_id="score_raw_content",
            artifact_catalog=catalog,
        )
        quality_result = run_async(QualityScorer().execute(quality_ctx))
        catalog.extend(quality_result.artifacts)
        accepted_artifact = catalog.filter(artifact_type="quality_accepted_html")[0]

        monkeypatch.setitem(
            sys.modules,
            "trafilatura",
            SimpleNamespace(
                extract=lambda *args, **kwargs: (
                    "<html><body><main>" + ("Lineage content " * 20) + "</main></body></html>"
                )
            ),
        )
        cleaner_inputs = {
            "html_dir": str(html_dir),
            "mapping_file": str(mapping_file),
            **quality_result.outputs,
        }
        cleaner_ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config=config,
            work_dir=tmp_dir,
            previous_outputs=cleaner_inputs,
            stage_definition={"type": "cleaner", "plugin": "trafilatura"},
            stage_id="clean_html",
            artifact_catalog=catalog,
        )

        cleaner_result = run_async(TrafilaturaCleaner().execute(cleaner_ctx))

        assert quality_result.status == StageStatus.COMPLETED
        assert cleaner_result.status == StageStatus.COMPLETED
        cleaned_artifact = next(
            artifact
            for artifact in cleaner_result.artifacts
            if artifact.artifact_type == "cleaned_html"
        )
        assert cleaned_artifact.source_artifact_ids == [accepted_artifact.artifact_id]
        assert source_file.exists()


class TestFormatterContracts:
    def test_execute_attaches_pdf_images_by_exact_markdown_path(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.pinecone_formatter import PineconeFormatter

        md_dir = tmp_dir / "markdown"
        md_dir.mkdir()
        md_file = md_dir / "report.md"
        md_file.write_text("# Report\n\nPDF content here.", encoding="utf-8")

        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()
        atomic_write_json(
            summaries_dir / "report.summary.json",
            {
                "document_title": "Report",
                "document_type": "handbook",
                "detailed_summary": "Summary",
                "source_original_file": str(md_file),
            },
        )

        image_path = tmp_dir / "images" / "figure.png"
        image_path.parent.mkdir()
        image_path.write_bytes(b"fake")

        catalog = ArtifactCatalog()
        catalog.extend(
            [
                {
                    "artifact_id": "md1",
                    "artifact_type": "markdown",
                    "role": "content",
                    "producer_stage": "convert_documents",
                    "uri": md_file.resolve().as_uri(),
                    "local_path": str(md_file),
                    "metadata": {"source_file": "/tmp/report.pdf", "source_type": "pdf"},
                },
                {
                    "artifact_id": "img1",
                    "artifact_type": "extracted_image",
                    "role": "document_media",
                    "producer_stage": "convert_documents",
                    "uri": image_path.resolve().as_uri(),
                    "local_path": str(image_path),
                    "metadata": {
                        "md_path": str(md_file),
                        "alt": "Campus map",
                        "caption": "Campus map",
                        "source_type": "pdf",
                    },
                },
            ]
        )

        ctx = StageContext(
            run_id="r1",
            project_name="p1",
            config={"formatter": {"include_full_content": True, "include_summary": True}},
            work_dir=tmp_dir,
            previous_outputs={"summaries_dir": str(summaries_dir), "md_dir": str(md_dir)},
            stage_definition={"type": "formatter", "plugin": "pinecone_formatter"},
            stage_id="format_embeddings",
            artifact_catalog=catalog,
        )

        result = run_async(PineconeFormatter().execute(ctx))
        docs = load_json_safe(result.outputs["formatted_file"])
        assert result.outputs["formatted_count"] == 1
        assert docs[0]["metadata"]["images"][0]["alt"] == "Campus map"


# ─────────────────────────────────────────────────────────────
# 16. StageResult / StageContext — verify factory methods
# ─────────────────────────────────────────────────────────────

class TestBaseClasses:
    def test_success_has_correct_status(self):
        from pipeline.core.base import StageResult, StageStatus
        r = StageResult.success(outputs={"k": "v"}, metrics={"n": 1}, checkpoint={"p": 0})
        assert r.status == StageStatus.COMPLETED
        assert r.outputs == {"k": "v"}
        assert r.checkpoint == {"p": 0}
        assert r.error_message is None

    def test_failure_captures_message(self):
        from pipeline.core.base import StageResult, StageStatus
        r = StageResult.failure("disk full", checkpoint={"saved": True})
        assert r.status == StageStatus.FAILED
        assert r.error_message == "disk full"
        assert r.checkpoint == {"saved": True}

    def test_skipped_captures_reason(self):
        from pipeline.core.base import StageResult, StageStatus
        r = StageResult.skipped("no input files")
        assert r.status == StageStatus.SKIPPED
        assert r.error_message == "no input files"

    def test_context_config_properties_return_empty_dict_for_missing(self):
        """When config doesn't have a section, property should return {} not KeyError."""
        from pipeline.core.base import StageContext
        ctx = StageContext(run_id="t", project_name="t", config={}, work_dir=Path("/tmp"))
        assert ctx.crawler_config == {}
        assert ctx.cleaner_config == {}
        assert ctx.converter_config == {}
        assert ctx.summarizer_config == {}
        assert ctx.quality_config == {}
        assert ctx.embedder_config == {}


class TestDoclingOfficeFallback:
    def test_docling_converter_uses_office_fallback_for_xlsx(self, tmp_dir):
        from pipeline.core.base import StageContext, StageStatus
        from pipeline.stages.converters.docling_converter import DoclingConverter

        download_dir = tmp_dir / "downloads"
        download_dir.mkdir()

        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.append(["Name", "Value"])
        ws.append(["Rows", 2])
        xlsx_path = download_dir / "data.xlsx"
        wb.save(xlsx_path)

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"converter": {"use_vlm": False}},
            work_dir=tmp_dir,
            previous_outputs={"download_dir": str(download_dir)},
            stage_definition={"type": "converter", "plugin": "docling"},
            stage_id="convert_docs",
        )

        result = run_async(DoclingConverter().execute(ctx))
        assert result.status == StageStatus.COMPLETED
        assert result.metrics["office_fallback_converted"] == 1
        assert (Path(result.outputs["md_dir"]) / "data.md").exists()


class TestPDFConverterValidation:
    def test_pdf_converter_quarantines_low_quality_pdf_output(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.stages.converters import pdf_converter as module

        download_dir = tmp_dir / "downloads"
        download_dir.mkdir()
        pdf_path = download_dir / "paper.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")

        monkeypatch.setitem(module.EXTRACTORS, ".pdf", lambda path: "Short")

        ctx = StageContext(
            run_id="test",
            project_name="test",
            config={"converter": {}},
            work_dir=tmp_dir,
            previous_outputs={"download_dir": str(download_dir)},
            stage_definition={"type": "converter", "plugin": "pdf_converter"},
            stage_id="convert_documents",
        )

        result = run_async(module.PDFOfficeConverter().execute(ctx))

        assert result.metrics["converted"] == 0
        assert result.metrics["failed"] == 1
        assert result.metrics["validation_failed"] == 1
        assert not (Path(result.outputs["md_dir"]) / "paper.md").exists()
        assert (Path(result.outputs["quarantine_dir"]) / "markdown" / "paper.md").exists()


class TestEvaluationDataset:
    def test_mbzuai_eval_template_roundtrip(self, tmp_dir):
        from pipeline.evaluation.dataset import load_eval_examples, mbzuai_eval_template, write_eval_examples

        path = tmp_dir / "template.jsonl"
        write_eval_examples(path, mbzuai_eval_template())

        examples = load_eval_examples(path)
        assert len(examples) >= 5
        assert examples[0].query_type == "fact"
        assert examples[-1].no_answer is True


class TestRetrievalEvaluation:
    def test_evaluate_retrieval_dataset_scores_and_gates(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "q1",
                            "query": "Where is MBZUAI located?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-a"],
                            "gold_parent_ids": ["parent-a"],
                            "gold_media_ids": [],
                        }
                    ),
                    json.dumps(
                        {
                            "id": "q2",
                            "query": "Describe the campus map.",
                            "query_type": "multimodal",
                            "source_type": "pdf",
                            "gold_chunk_ids": ["chunk-b"],
                            "gold_parent_ids": ["parent-b"],
                            "gold_media_ids": ["media-b"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        gates_path = tmp_dir / "gates.json"
        gates_path.write_text(
            json.dumps(
                {
                    "overall": {
                        "chunk_hit_at_10": {"min": 1.0},
                        "parent_hit_at_5": {"min": 0.5},
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            def retrieve(self, query):
                if "located" in query:
                    return {
                        "mode": "fact",
                        "seed_chunk_ids": ["chunk-a"],
                        "selected_chunk_ids": ["chunk-a", "chunk-x"],
                        "selected_parent_ids": ["parent-a"],
                        "dense_parent_ids": ["parent-a"],
                        "selected_media_ids": [],
                        "dense_media_ids": [],
                        "media": [],
                    }
                return {
                    "mode": "multimodal",
                    "seed_chunk_ids": ["chunk-z"],
                    "selected_chunk_ids": ["chunk-z", "chunk-b"],
                    "selected_parent_ids": ["parent-b"],
                    "dense_parent_ids": ["parent-b"],
                    "selected_media_ids": ["media-b"],
                    "dense_media_ids": ["media-b"],
                    "media": [{"id": "media-b"}],
                }

        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: FakeRetriever(),
        )
        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.validate_eval_examples",
            lambda *args, **kwargs: {"ok": True, "errors": [], "warnings": [], "summary": {"query_count": 2}},
        )

        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            gates_path=gates_path,
        )

        assert report["query_count"] == 2
        assert report["execution"]["strict"] is True
        assert report["dataset_fingerprint"]
        assert report["config_fingerprint"]
        assert report["overall"]["chunk_hit_at_10"] == 1.0
        assert report["overall"]["parent_hit_at_5"] == 1.0
        assert report["overall"]["eligible_media_query_count"] == 1.0
        assert report["overall"]["media_hit_at_1"] == 1.0
        assert report["overall"]["media_hit_at_5"] == 1.0
        assert report["overall"]["media_mrr_at_5"] == 1.0
        assert report["gates"]["passed"] is True
        assert report["by_query_type"]["fact"]["chunk_mrr_at_10"] == 1.0

    def test_evaluate_retrieval_dataset_reports_benchmark_tag_slices(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "q1",
                            "query": "In which city is MBZUAI located?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-a"],
                            "gold_parent_ids": ["parent-a"],
                            "metadata": {"benchmark_tags": ["relation_heavy_fact", "relation_heavy"]},
                        }
                    ),
                    json.dumps(
                        {
                            "id": "q2",
                            "query": "Summarize the campus.",
                            "query_type": "scoped",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-b"],
                            "gold_parent_ids": ["parent-b"],
                            "metadata": {"benchmark_tags": ["non_relation"]},
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            def retrieve(self, query):
                if "city" in query:
                    return {
                        "mode": "fact",
                        "seed_chunk_ids": ["chunk-a"],
                        "selected_chunk_ids": ["chunk-a"],
                        "selected_parent_ids": ["parent-a"],
                        "dense_parent_ids": ["parent-a"],
                        "selected_media_ids": [],
                        "dense_media_ids": [],
                        "media": [],
                    }
                return {
                    "mode": "scoped",
                    "seed_chunk_ids": ["chunk-b"],
                    "selected_chunk_ids": ["chunk-b"],
                    "selected_parent_ids": ["parent-b"],
                    "dense_parent_ids": ["parent-b"],
                    "selected_media_ids": [],
                    "dense_media_ids": [],
                    "media": [],
                }

        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: FakeRetriever(),
        )

        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
        )

        assert report["by_benchmark_tag"]["relation_heavy_fact"]["query_count"] == 1.0
        assert report["by_benchmark_tag"]["relation_heavy_fact"]["chunk_hit_at_5"] == 1.0
        assert report["queries"][0]["benchmark_tags"]

    def test_evaluate_retrieval_dataset_reuses_query_embedding_cache(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "query": "Where is MBZUAI located?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "gold_chunk_ids": ["chunk-a"],
                    "gold_parent_ids": ["parent-a"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            model = "fake-embed-model"
            output_dimensionality = 8

            def __init__(self):
                self.embed_calls = 0

            def embed_query(self, query):
                self.embed_calls += 1
                return [0.1, 0.2, 0.3]

            def retrieve(self, query, *, query_vector=None):
                assert query_vector == [0.1, 0.2, 0.3]
                return {
                    "mode": "fact",
                    "seed_chunk_ids": ["chunk-a"],
                    "selected_chunk_ids": ["chunk-a"],
                    "selected_parent_ids": ["parent-a"],
                    "dense_parent_ids": ["parent-a"],
                    "selected_media_ids": [],
                    "dense_media_ids": [],
                    "media": [],
                }

        fake = FakeRetriever()
        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: fake,
        )

        cache_path = tmp_dir / "query_cache.json"
        first_retrieval_cache_path = tmp_dir / "retrieval_cache_first.json"
        second_retrieval_cache_path = tmp_dir / "retrieval_cache_second.json"
        first_report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            query_cache_path=cache_path,
            retrieval_cache_path=first_retrieval_cache_path,
        )
        second_report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            query_cache_path=cache_path,
            retrieval_cache_path=second_retrieval_cache_path,
        )

        assert fake.embed_calls == 1
        assert first_report["cache"]["miss_count"] == 1
        assert first_report["cache"]["hit_count"] == 0
        assert second_report["cache"]["miss_count"] == 0
        assert second_report["cache"]["hit_count"] == 1
        assert cache_path.exists()

    def test_evaluate_retrieval_dataset_reuses_retrieval_result_cache(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "query": "Where is MBZUAI located?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "gold_chunk_ids": ["chunk-a"],
                    "gold_parent_ids": ["parent-a"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            model = "fake-embed-model"
            output_dimensionality = 8

            def __init__(self):
                self.retrieve_calls = 0

            def retrieve(self, query, *, query_vector=None):
                self.retrieve_calls += 1
                return {
                    "mode": "fact",
                    "seed_chunk_ids": ["chunk-a"],
                    "selected_chunk_ids": ["chunk-a"],
                    "selected_parent_ids": ["parent-a"],
                    "dense_parent_ids": ["parent-a"],
                    "selected_media_ids": [],
                    "dense_media_ids": [],
                    "media": [],
                }

        fake = FakeRetriever()
        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: fake,
        )

        retrieval_cache_path = tmp_dir / "retrieval_cache.json"
        first_report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            retrieval_cache_path=retrieval_cache_path,
        )
        second_report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            retrieval_cache_path=retrieval_cache_path,
        )

        assert fake.retrieve_calls == 1
        assert first_report["retrieval_cache"]["miss_count"] == 1
        assert first_report["retrieval_cache"]["hit_count"] == 0
        assert second_report["retrieval_cache"]["miss_count"] == 0
        assert second_report["retrieval_cache"]["hit_count"] == 1
        assert retrieval_cache_path.exists()

    def test_evaluate_retrieval_dataset_uses_full_retrieval_cache_without_building_retriever(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import (
            _retrieval_cache_key,
            evaluate_retrieval_dataset,
        )

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "query": "Where is MBZUAI located?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "gold_chunk_ids": ["chunk-a"],
                    "gold_parent_ids": ["parent-a"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.load_config",
            lambda _config_name: {
                "embedder": {
                    "model": "fake-embed-model",
                    "output_dimensionality": 8,
                }
            },
        )

        retrieval_cache_path = tmp_dir / "retrieval_cache.json"
        retrieval_cache_path.write_text(
            json.dumps(
                {
                    _retrieval_cache_key(
                        config_name="unused",
                        work_dir=tmp_dir,
                        query="Where is MBZUAI located?",
                        model="fake-embed-model",
                        output_dimensionality=8,
                        config_payload={
                            "embedder": {
                                "model": "fake-embed-model",
                                "output_dimensionality": 8,
                            }
                        },
                    ): {
                        "mode": "fact",
                        "seed_chunk_ids": ["chunk-a"],
                        "selected_chunk_ids": ["chunk-a"],
                        "selected_parent_ids": ["parent-a"],
                        "dense_parent_ids": ["parent-a"],
                        "selected_media_ids": [],
                        "dense_media_ids": [],
                        "media": [],
                    }
                }
            )
            + "\n",
            encoding="utf-8",
        )

        def _fail(**kwargs):
            raise AssertionError("retriever should not be constructed when retrieval cache fully covers the dataset")

        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            _fail,
        )

        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            retrieval_cache_path=retrieval_cache_path,
        )

        assert report["query_count"] == 1
        assert report["retrieval_cache"]["hit_count"] == 1
        assert report["retrieval_cache"]["miss_count"] == 0
        assert report["overall"]["chunk_hit_at_10"] == 1.0

    def test_evaluate_retrieval_dataset_rejects_invalid_wrapped_retrieval_cache_entries(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.dataset import EvalExample
        from pipeline.evaluation.retrieval_eval import (
            _make_retrieval_cache_entry,
            _retrieval_cache_key,
            evaluate_retrieval_dataset,
        )

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "query": "Where is MBZUAI located?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "gold_chunk_ids": ["chunk-a"],
                    "gold_parent_ids": ["parent-a"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        config_payload = {
            "embedder": {
                "model": "fake-embed-model",
                "output_dimensionality": 8,
            }
        }
        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.load_config",
            lambda _config_name: config_payload,
        )

        retrieval_cache_path = tmp_dir / "retrieval_cache.json"
        cache_key = _retrieval_cache_key(
            config_name="unused",
            work_dir=tmp_dir,
            query="Where is MBZUAI located?",
            model="fake-embed-model",
            output_dimensionality=8,
            config_payload=config_payload,
        )
        wrapped_entry = _make_retrieval_cache_entry(
            example=EvalExample(
                id="q1",
                query="Where is MBZUAI located?",
                query_type="fact",
                source_type="webpage",
                gold_chunk_ids=["chunk-a"],
                gold_parent_ids=["parent-a"],
            ).normalized(),
            result={
                "mode": "fact",
                "seed_chunk_ids": ["chunk-a"],
                "selected_chunk_ids": ["chunk-a"],
                "selected_parent_ids": ["parent-a"],
                "dense_parent_ids": ["parent-a"],
                "selected_media_ids": [],
                "dense_media_ids": [],
                "media": [],
            },
            config_name="unused",
            work_dir=tmp_dir,
            model="fake-embed-model",
            output_dimensionality=8,
            config_payload=config_payload,
        )
        wrapped_entry["query"] = "Tampered query"
        retrieval_cache_path.write_text(
            json.dumps({cache_key: wrapped_entry}) + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            model = "fake-embed-model"
            output_dimensionality = 8

            def __init__(self):
                self.retrieve_calls = 0

            def retrieve(self, query, *, query_vector=None):
                self.retrieve_calls += 1
                return {
                    "mode": "fact",
                    "seed_chunk_ids": ["chunk-a"],
                    "selected_chunk_ids": ["chunk-a"],
                    "selected_parent_ids": ["parent-a"],
                    "dense_parent_ids": ["parent-a"],
                    "selected_media_ids": [],
                    "dense_media_ids": [],
                    "media": [],
                }

        fake = FakeRetriever()
        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: fake,
        )

        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            retrieval_cache_path=retrieval_cache_path,
        )

        assert fake.retrieve_calls == 1
        assert report["retrieval_cache"]["invalid_count"] == 1
        assert report["retrieval_cache"]["miss_count"] == 1
        assert report["retrieval_cache"]["hit_count"] == 0

    def test_evaluate_retrieval_dataset_batches_uncached_queries_when_supported(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "q1",
                            "query": "Where is MBZUAI located?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-a"],
                            "gold_parent_ids": ["parent-a"],
                        }
                    ),
                    json.dumps(
                        {
                            "id": "q2",
                            "query": "Does MBZUAI provide student accommodation?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-b"],
                            "gold_parent_ids": ["parent-b"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            model = "fake-embed-model"
            output_dimensionality = 8

            def __init__(self):
                self.batch_embed_calls = 0
                self.single_embed_calls = 0

            def embed_queries(self, queries):
                self.batch_embed_calls += 1
                return [[0.1, 0.2, float(idx)] for idx, _query in enumerate(queries, start=1)]

            def embed_query(self, query):
                self.single_embed_calls += 1
                raise AssertionError("single-query embedding should not be used when batch support exists")

            def retrieve(self, query, *, query_vector=None):
                if "located" in query:
                    assert query_vector == [0.1, 0.2, 1.0]
                    return {
                        "mode": "fact",
                        "seed_chunk_ids": ["chunk-a"],
                        "selected_chunk_ids": ["chunk-a"],
                        "selected_parent_ids": ["parent-a"],
                        "dense_parent_ids": ["parent-a"],
                        "selected_media_ids": [],
                        "dense_media_ids": [],
                        "media": [],
                    }
                assert query_vector == [0.1, 0.2, 2.0]
                return {
                    "mode": "fact",
                    "seed_chunk_ids": ["chunk-b"],
                    "selected_chunk_ids": ["chunk-b"],
                    "selected_parent_ids": ["parent-b"],
                    "dense_parent_ids": ["parent-b"],
                    "selected_media_ids": [],
                    "dense_media_ids": [],
                    "media": [],
                }

        fake = FakeRetriever()
        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: fake,
        )

        cache_path = tmp_dir / "query_cache.json"
        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            query_cache_path=cache_path,
        )

        assert fake.batch_embed_calls == 1
        assert fake.single_embed_calls == 0
        assert report["cache"]["miss_count"] == 2
        assert report["cache"]["hit_count"] == 0

    def test_evaluate_retrieval_dataset_caps_unshared_parallel_retrieval_workers(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "q1",
                            "query": "Where is MBZUAI located?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-a"],
                            "gold_parent_ids": ["parent-a"],
                        }
                    ),
                    json.dumps(
                        {
                            "id": "q2",
                            "query": "Does MBZUAI provide student accommodation?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-b"],
                            "gold_parent_ids": ["parent-b"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        created_instances = []

        class FakeRetriever:
            model = "fake-embed-model"
            output_dimensionality = 8

            def __init__(self):
                created_instances.append(self)

            def embed_queries(self, queries):
                return [[0.1, 0.2, float(idx)] for idx, _query in enumerate(queries, start=1)]

            def embed_query(self, query):
                return [0.1, 0.2, 9.0]

            def retrieve(self, query, *, query_vector=None):
                if "located" in query:
                    return {
                        "mode": "fact",
                        "seed_chunk_ids": ["chunk-a"],
                        "selected_chunk_ids": ["chunk-a"],
                        "selected_parent_ids": ["parent-a"],
                        "dense_parent_ids": ["parent-a"],
                        "selected_media_ids": [],
                        "dense_media_ids": [],
                        "media": [],
                    }
                return {
                    "mode": "fact",
                    "seed_chunk_ids": ["chunk-b"],
                    "selected_chunk_ids": ["chunk-b"],
                    "selected_parent_ids": ["parent-b"],
                    "dense_parent_ids": ["parent-b"],
                    "selected_media_ids": [],
                    "dense_media_ids": [],
                    "media": [],
                }

        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: FakeRetriever(),
        )

        events = []
        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            parallelism=2,
            progress_callback=lambda event, payload: events.append((event, dict(payload))),
        )

        assert report["query_count"] == 2
        assert report["overall"]["chunk_hit_at_10"] == 1.0
        assert report["overall"]["parent_hit_at_5"] == 1.0
        assert report["execution"]["parallelism_requested"] == 2
        assert report["execution"]["parallelism_effective"] == 1
        assert report["execution"]["shared_parallel_retriever"] is False
        assert len(created_instances) == 1
        assert any(event == "retrieval_eval_parallelism_capped" for event, _payload in events)

    def test_evaluate_retrieval_dataset_can_share_parallel_retriever_when_supported(self, tmp_dir, monkeypatch):
        import time

        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "q1",
                            "query": "Where is MBZUAI located?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-a"],
                            "gold_parent_ids": ["parent-a"],
                        }
                    ),
                    json.dumps(
                        {
                            "id": "q2",
                            "query": "Does MBZUAI provide student accommodation?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk-b"],
                            "gold_parent_ids": ["parent-b"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        created_instances = []

        class FakeRetriever:
            model = "fake-embed-model"
            output_dimensionality = 8
            supports_shared_parallel_retrieval = True

            def __init__(self):
                created_instances.append(self)

            def embed_queries(self, queries):
                return [[0.1, 0.2, float(idx)] for idx, _query in enumerate(queries, start=1)]

            def embed_query(self, query):
                return [0.1, 0.2, 9.0]

            def retrieve(self, query, *, query_vector=None):
                time.sleep(0.05)
                if "located" in query:
                    return {
                        "mode": "fact",
                        "seed_chunk_ids": ["chunk-a"],
                        "selected_chunk_ids": ["chunk-a"],
                        "selected_parent_ids": ["parent-a"],
                        "dense_parent_ids": ["parent-a"],
                        "selected_media_ids": [],
                        "dense_media_ids": [],
                        "media": [],
                    }
                return {
                    "mode": "fact",
                    "seed_chunk_ids": ["chunk-b"],
                    "selected_chunk_ids": ["chunk-b"],
                    "selected_parent_ids": ["parent-b"],
                    "dense_parent_ids": ["parent-b"],
                    "selected_media_ids": [],
                    "dense_media_ids": [],
                    "media": [],
                }

        fake = FakeRetriever()
        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: fake,
        )

        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            parallelism=2,
        )

        assert report["query_count"] == 2
        assert report["overall"]["chunk_hit_at_10"] == 1.0
        assert report["overall"]["parent_hit_at_5"] == 1.0
        assert report["execution"]["shared_parallel_retriever"] is True
        assert len(created_instances) == 1

    def test_evaluate_retrieval_dataset_records_query_errors_without_dropping_report(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

        dataset_path = tmp_dir / "gold.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "q1",
                    "query": "Where is MBZUAI located?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "gold_chunk_ids": ["chunk-a"],
                    "gold_parent_ids": ["parent-a"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        class FakeRetriever:
            model = "fake-embed-model"
            output_dimensionality = 8

            def retrieve(self, query, *, query_vector=None):
                raise RuntimeError("retriever exploded")

        monkeypatch.setattr(
            "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
            lambda **kwargs: FakeRetriever(),
        )
        events = []

        report = evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_dir,
            dataset_path=dataset_path,
            progress_callback=lambda event, payload: events.append((event, dict(payload))),
        )

        assert report["query_count"] == 1
        assert report["execution"]["retrieval_error_count"] == 1
        assert report["overall"]["retrieval_error_count"] == 1.0
        assert report["overall"]["successful_query_count"] == 0.0
        assert report["retrieval_errors"][0]["id"] == "q1"
        assert "retriever exploded" in report["retrieval_errors"][0]["error"]
        assert report["overall"]["chunk_hit_at_10"] == 0.0
        assert any(event == "retrieval_eval_query_done" and payload["error"] for event, payload in events)


class TestEvaluationDatasetTools:
    def test_validate_eval_set_accepts_current_mbzuai_retrieval_bundle_stage_id(self, tmp_dir):
        from pipeline.core.io import atomic_write_json
        from pipeline.evaluation.dataset_tools import validate_eval_examples

        run_dir = tmp_dir / "run"
        stage_dir = run_dir / "stage_outputs" / "build_retrieval_bundle"
        stage_dir.mkdir(parents=True)
        atomic_write_json(
            stage_dir / "retrieval_bundle.json",
            {
                "chunk_records": [{"id": "chunk-current"}],
                "parent_records": [{"id": "parent-current"}],
                "media_records": [],
            },
        )
        dataset_path = tmp_dir / "eval.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "id": "ok-current",
                    "query": "Admissions requirements?",
                    "query_type": "fact",
                    "source_type": "webpage",
                    "gold_chunk_ids": ["chunk-current"],
                    "gold_parent_ids": ["parent-current"],
                }
            )
            + "\n",
            encoding="utf-8",
        )

        report = validate_eval_examples(dataset_path, work_dir=run_dir)

        assert report["ok"] is True
        assert report["errors"] == []

    def test_validate_eval_set_checks_gold_ids_against_retrieval_bundle(self, tmp_dir):
        from pipeline.evaluation.dataset_tools import validate_eval_examples

        run_dir = TestAdaptiveHybridRetriever()._build_retrieval_run(tmp_dir)
        dataset_path = tmp_dir / "eval.jsonl"
        dataset_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "ok-1",
                            "query": "Admissions requirements?",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["chunk1"],
                            "gold_parent_ids": ["page-a"],
                        }
                    ),
                    json.dumps(
                        {
                            "id": "bad-1",
                            "query": "Broken id",
                            "query_type": "fact",
                            "source_type": "webpage",
                            "gold_chunk_ids": ["missing-chunk"],
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        report = validate_eval_examples(dataset_path, work_dir=run_dir)
        assert report["ok"] is False
        assert any(issue["id"] == "bad-1" and issue["field"] == "gold_chunk_ids" for issue in report["errors"])

    def test_summarize_eval_set_counts_query_slices(self):
        from pipeline.evaluation.dataset import EvalExample
        from pipeline.evaluation.dataset_tools import summarize_eval_examples

        examples = [
            EvalExample(id="q1", query="Where?", query_type="fact", source_type="webpage").normalized(),
            EvalExample(id="q2", query="Show map", query_type="multimodal", source_type="pdf").normalized(),
            EvalExample(id="q3", query="No answer", query_type="fact", source_type="none", no_answer=True).normalized(),
        ]

        report = summarize_eval_examples(examples)
        assert report["query_count"] == 3
        assert report["query_type_counts"] == {"fact": 2, "multimodal": 1}
        assert report["source_type_counts"] == {"none": 1, "pdf": 1, "webpage": 1}
        assert report["no_answer_count"] == 1


class TestMBZLegacyVectorStoreFormatterQualityMetadata:
    def test_authority_and_intent_metadata_for_official_admissions_page(self):
        from pipeline.stages.formatters.mbzuai_legacy_vectorstore_formatter import (
            _infer_authority,
            _infer_intent_tags,
        )

        authority_class, authority_score = _infer_authority(
            "https://mbzuai.ac.ae/study/admission-requirements",
            "Admission Requirements",
            "webpage",
        )
        intents = _infer_intent_tags(
            "https://mbzuai.ac.ae/study/admission-requirements",
            "Admission Requirements",
            "Official eligibility, application requirements, and deadlines.",
            {"h1": ["Admission Requirements"], "h2": ["Required documents"]},
        )

        assert authority_class == "official_admissions"
        assert authority_score >= 0.9
        assert "admissions" in intents

    def test_news_authority_is_lower_than_official_program_page(self):
        from pipeline.stages.formatters.mbzuai_legacy_vectorstore_formatter import _infer_authority

        news_class, news_score = _infer_authority(
            "https://mbzuai.ac.ae/news/artificial-intelligence-event",
            "AI Event",
            "webpage",
        )
        program_class, program_score = _infer_authority(
            "https://mbzuai.ac.ae/study/graduate-programs",
            "Graduate Programs",
            "webpage",
        )

        assert news_class == "time_bound_content"
        assert program_class == "official_program"
        assert program_score > news_score


class TestMBZUAIIndexReadiness:
    def test_validate_config_rejects_invalid_coverage_values_and_patterns(self):
        from pipeline.stages.formatters.mbzuai_index_readiness_formatter import (
            MBZUAIIndexReadinessFormatter,
        )

        errors = run_async(
            MBZUAIIndexReadinessFormatter().validate_config(
                {
                    "formatter": {
                        "expected_site_inventory_count": -1,
                        "maximum_hard_failure_count": -1,
                        "minimum_inventory_coverage_ratio": 1.1,
                        "critical_url_patterns": ["", "[invalid"],
                    }
                }
            )
        )

        assert "formatter.expected_site_inventory_count must be >= 0" in errors
        assert "formatter.maximum_hard_failure_count must be >= 0" in errors
        assert (
            "formatter.minimum_inventory_coverage_ratio must be between 0 and 1"
            in errors
        )
        assert "formatter.critical_url_patterns[0] must be non-empty" in errors
        assert any(
            error.startswith("formatter.critical_url_patterns[1] is invalid:")
            for error in errors
        )

    def test_inventory_coverage_excludes_intentional_url_exclusions(self, tmp_dir):
        from pipeline.stages.formatters.mbzuai_index_readiness_formatter import (
            _coverage_gate,
            _failure_manifest,
        )

        markdown_path = tmp_dir / "study.md"
        markdown_path.write_text(
            " ".join(["Official admissions requirements and program details."] * 20),
            encoding="utf-8",
        )

        runtime_state = {
            "stats": {
                "pages_scraped": 2504,
                "documents_downloaded": 120,
            },
            "url_mapping": {
                **{
                    f"https://mbzuai.ac.ae/tag/topic-{idx}": "SKIPPED_EXCLUDED_FRONTIER:path_prefix"
                    for idx in range(1004)
                },
                "https://mbzuai.ac.ae/missing": "SKIPPED_HTTP_404",
            },
            "crawl_state": {
                "visited": ["https://mbzuai.ac.ae"],
                "pending": [],
                "pages_crawled": 1,
            },
        }
        formatter_config = {
            "expected_site_inventory_count": 3800,
            "minimum_inventory_coverage_ratio": 0.75,
            "critical_url_patterns": [r"/study/"],
        }

        failure_manifest = _failure_manifest(runtime_state, formatter_config)
        coverage_gate = _coverage_gate(
            canonical_metadata={
                "https://mbzuai.ac.ae/study/graduate-admission-process": {
                    "indexable": True,
                    "markdown_path": str(markdown_path),
                }
            },
            failure_manifest=failure_manifest,
            formatter_config=formatter_config,
        )

        assert failure_manifest["intentional_excluded_count"] == 1004
        assert failure_manifest["hard_failure_count"] == 1
        assert failure_manifest["effective_expected_inventory_count"] == 2796
        assert failure_manifest["raw_inventory_coverage_ratio"] == 0.6905
        assert failure_manifest["inventory_coverage_ratio"] > 0.93
        assert coverage_gate["ok"] is True
        assert coverage_gate["inventory_gap"] is False

    def test_production_coverage_gate_rejects_excess_hard_failures(self):
        from pipeline.stages.formatters.mbzuai_index_readiness_formatter import _coverage_gate

        gate = _coverage_gate(
            canonical_metadata={
                "https://mbzuai.ac.ae/study/": {"indexable": True},
            },
            failure_manifest={
                "hard_failure_count": 26,
                "expected_site_inventory_count": 3800,
                "effective_expected_inventory_count": 3800,
                "inventory_coverage_ratio": 0.95,
                "failed_urls": [],
            },
            formatter_config={
                "expected_site_inventory_count": 3800,
                "minimum_inventory_coverage_ratio": 0.90,
                "maximum_hard_failure_count": 25,
                "critical_url_patterns": [r"/study/"],
            },
        )

        assert gate["ok"] is False
        assert gate["hard_failure_gap"] is True

    def test_canonical_stage_writes_url_identity_and_change_manifest(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.mbzuai_index_readiness_formatter import MBZUAIIndexReadinessFormatter

        md_path = tmp_dir / "admissions.md"
        md_path.write_text("Admission requirements for MBZUAI graduate applicants.", encoding="utf-8")
        page_metadata_file = tmp_dir / "page_metadata.json"
        page_link_graph_file = tmp_dir / "page_link_graph.json"
        atomic_write_json(
            page_metadata_file,
            {
                "https://mbzuai.ac.ae/en/study/admissions/?utm_source=test": {
                    "url": "https://mbzuai.ac.ae/en/study/admissions/?utm_source=test",
                    "title": "Admissions",
                    "language": "en-US",
                    "markdown_path": str(md_path),
                    "headings": {"h1": ["Admissions"]},
                },
                "https://mbzuai.ac.ae/ar/study/admissions/": {
                    "url": "https://mbzuai.ac.ae/ar/study/admissions/",
                    "title": "Admissions Arabic",
                    "language": "ar",
                    "markdown_path": str(md_path),
                    "headings": {"h1": ["Admissions"]},
                },
            },
        )
        atomic_write_json(
            page_link_graph_file,
            {
                "edges": [
                    {
                        "source_url": "https://mbzuai.ac.ae/en/study/admissions/",
                        "target_url": "https://mbzuai.ac.ae/en/study/scholarships/",
                        "properties": {"link_type": "internal", "anchor_texts": ["Scholarships"]},
                    }
                ]
            },
        )
        ctx = StageContext(
            run_id="r-index-ready",
            project_name="mbzuai",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={
                "page_metadata_file": str(page_metadata_file),
                "page_link_graph_file": str(page_link_graph_file),
            },
            stage_definition={"type": "formatter", "plugin": "mbzuai_index_readiness"},
            stage_id="prepare_mbzuai_index",
        )

        result = run_async(MBZUAIIndexReadinessFormatter().execute(ctx))

        assert result.status.value == "completed"
        identity = load_json_safe(result.outputs["url_identity_map_file"])
        changes = load_json_safe(result.outputs["page_change_manifest_file"])
        canonical_metadata = load_json_safe(result.outputs["canonical_page_metadata_file"])
        coverage_gate = load_json_safe(result.outputs["index_coverage_gate_file"])
        assert identity["duplicate_family_count"] == 1
        assert changes["new_page_count"] == 1
        assert coverage_gate["missing_critical_count"] >= 1
        assert "crawl_failure_manifest_file" in result.outputs
        assert any(item["language"] == "en" for item in identity["records"])
        first_record = next(iter(canonical_metadata.values()))
        assert first_record["canonical_family_url"] == "https://mbzuai.ac.ae/study/admissions"
        assert first_record["content_hash"]

    def test_canonical_metadata_marks_homepage_redirect_alias_non_indexable(self):
        from pipeline.core.mbzuai_indexing import canonicalize_page_metadata

        canonical = canonicalize_page_metadata(
            {
                "https://mbzuai.ac.ae/about/contact": {
                    "url": "https://mbzuai.ac.ae/about/contact",
                    "canonical_url": "https://mbzuai.ac.ae/",
                    "status_code": 301,
                    "title": "MBZUAI - Mohamed bin Zayed University of Artificial Intelligence",
                }
            }
        )

        record = canonical["https://mbzuai.ac.ae/about/contact"]
        assert record["page_type"] == "redirect_alias"
        assert record["indexable"] is False
        assert record["index_exclusion_reason"] == "homepage_redirect_alias"

    def test_legacy_formatter_preserves_backend_contract_and_adds_citation_metadata(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.mbzuai_legacy_vectorstore_formatter import MBZLegacyVectorStoreFormatter

        md_path = tmp_dir / "admissions.md"
        md_path.write_text("## Required documents\n\nApplicants must submit transcripts.", encoding="utf-8")
        chunks_file = tmp_dir / "chunks.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc-admissions",
                        "document_title": "Admission Requirements",
                        "document_type": "webpage",
                        "source_markdown_path": str(md_path),
                        "source_url": "https://mbzuai.ac.ae/en/study/admissions/",
                        "section_path": ["Study", "Admission Requirements", "Required documents"],
                        "text": "Applicants must submit transcripts and other required documents.",
                    }
                ],
                strategy="hybrid",
            ),
        )
        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()
        atomic_write_json(
            summaries_dir / "admissions.summary.json",
            {
                "source_original_file": str(md_path),
                "document_title": "Admission Requirements",
                "document_summary": "Official admission requirements.",
                "key_facts": ["Applicants submit transcripts."],
                "keywords": ["admissions"],
            },
        )
        page_metadata_file = tmp_dir / "canonical_page_metadata.json"
        atomic_write_json(
            page_metadata_file,
            {
                "https://mbzuai.ac.ae/en/study/admissions": {
                    "url": "https://mbzuai.ac.ae/en/study/admissions",
                    "canonical_url": "https://mbzuai.ac.ae/en/study/admissions",
                    "canonical_family_url": "https://mbzuai.ac.ae/study/admissions",
                    "normalized_path": "/study/admissions",
                    "language": "en",
                    "title": "Admission Requirements",
                    "content_hash": "abc123",
                    "locale_variant_urls": [
                        "https://mbzuai.ac.ae/en/study/admissions",
                        "https://mbzuai.ac.ae/ar/study/admissions",
                    ],
                    "headings": {"h1": ["Admission Requirements"], "h2": ["Required documents"]},
                }
            },
        )
        ctx = StageContext(
            run_id="r-legacy",
            project_name="mbzuai",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={
                "chunks_file": str(chunks_file),
                "summaries_dir": str(summaries_dir),
                "page_metadata_file": str(page_metadata_file),
            },
            stage_definition={"type": "formatter", "plugin": "mbzuai_legacy_vectorstores"},
            stage_id="format_legacy_vectorstores",
        )

        result = run_async(MBZLegacyVectorStoreFormatter().execute(ctx))

        assert result.status.value == "completed"
        text_docs = load_json_safe(result.outputs["legacy_text_formatted_file"])
        metadata = text_docs[0]["metadata"]
        assert metadata["page_source"] == "https://mbzuai.ac.ae/en/study/admissions/"
        assert metadata["context"]
        assert metadata["canonical_family_url"] == "https://mbzuai.ac.ae/study/admissions"
        assert metadata["page_title"] == "Admission Requirements"
        assert metadata["section_title"] == "Required documents"
        assert "Admission Requirements > Required documents" in metadata["breadcrumb"]
        assert metadata["citation_anchor"]["language"] == "en"

    def test_legacy_formatter_restores_public_pdf_source_url_from_download_mapping(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.mbzuai_legacy_vectorstore_formatter import MBZLegacyVectorStoreFormatter

        pdf_path = tmp_dir / "downloads" / "MAAIBrochure2025.pdf"
        pdf_path.parent.mkdir()
        pdf_path.write_text("pdf placeholder", encoding="utf-8")
        md_path = tmp_dir / "MAAIBrochure2025.md"
        md_path.write_text("## Admissions & Fees\n\nMBZUAI accepts applications from qualified candidates.", encoding="utf-8")
        atomic_write_json(
            tmp_dir / "mappings.json",
            {"https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/02/MAAIBrochure2025.pdf": str(pdf_path)},
        )
        chunks_file = tmp_dir / "chunks.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc-maai",
                        "document_title": "MAAI Brochure",
                        "document_type": "pdf",
                        "source_file": str(pdf_path),
                        "source_markdown_path": str(md_path),
                        "source_url": "",
                        "section_path": ["Admissions & Fees"],
                        "text": "MBZUAI accepts applications from qualified candidates.",
                    }
                ],
                strategy="hybrid",
            ),
        )
        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()
        atomic_write_json(
            summaries_dir / "maai.summary.json",
            {
                "source_original_file": str(md_path),
                "document_title": "MAAI Brochure",
                "document_summary": "Admissions and fees brochure.",
                "key_facts": [],
                "keywords": ["admissions"],
            },
        )
        ctx = StageContext(
            run_id="r-legacy-pdf",
            project_name="mbzuai",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={"chunks_file": str(chunks_file), "summaries_dir": str(summaries_dir)},
            stage_definition={"type": "formatter", "plugin": "mbzuai_legacy_vectorstores"},
            stage_id="format_legacy_vectorstores",
        )

        result = run_async(MBZLegacyVectorStoreFormatter().execute(ctx))

        assert result.status.value == "completed"
        text_docs = load_json_safe(result.outputs["legacy_text_formatted_file"])
        metadata = text_docs[0]["metadata"]
        assert metadata["page_source"] == "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/02/MAAIBrochure2025.pdf"
        assert metadata["source_file"] == str(md_path)

    def test_legacy_formatter_uses_canonical_url_for_redirected_html_citations(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.mbzuai_legacy_vectorstore_formatter import MBZLegacyVectorStoreFormatter

        md_path = tmp_dir / "redirected.md"
        md_path.write_text("## Explore our AI degrees\n\nOur degree programs.", encoding="utf-8")
        chunks_file = tmp_dir / "chunks.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc-home",
                        "document_title": "MBZUAI Degree Programs",
                        "document_type": "webpage",
                        "source_markdown_path": str(md_path),
                        "source_url": "https://mbzuai.ac.ae/about/leadership/hong-dekyi-liang",
                        "section_path": ["Explore our AI degrees"],
                        "text": "Our degree programs include master’s, doctoral, and undergraduate options.",
                    }
                ],
                strategy="hybrid",
            ),
        )
        summaries_dir = tmp_dir / "summaries"
        summaries_dir.mkdir()
        atomic_write_json(
            summaries_dir / "redirected.summary.json",
            {
                "source_original_file": str(md_path),
                "document_title": "MBZUAI Degree Programs",
                "document_summary": "Degree programs page.",
                "key_facts": [],
                "keywords": ["programs"],
            },
        )
        page_metadata_file = tmp_dir / "canonical_page_metadata.json"
        atomic_write_json(
            page_metadata_file,
            {
                "https://mbzuai.ac.ae/about/leadership/hong-dekyi-liang": {
                    "url": "https://mbzuai.ac.ae/about/leadership/hong-dekyi-liang",
                    "status_code": 301,
                    "canonical_url": "https://mbzuai.ac.ae/",
                    "canonical_family_url": "https://mbzuai.ac.ae/",
                    "title": "MBZUAI",
                    "headings": {"h2": ["Explore our AI degrees"]},
                }
            },
        )
        ctx = StageContext(
            run_id="r-legacy-redirect",
            project_name="mbzuai",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={
                "chunks_file": str(chunks_file),
                "summaries_dir": str(summaries_dir),
                "page_metadata_file": str(page_metadata_file),
            },
            stage_definition={"type": "formatter", "plugin": "mbzuai_legacy_vectorstores"},
            stage_id="format_legacy_vectorstores",
        )

        result = run_async(MBZLegacyVectorStoreFormatter().execute(ctx))

        assert result.status.value == "completed"
        text_docs = load_json_safe(result.outputs["legacy_text_formatted_file"])
        metadata = text_docs[0]["metadata"]
        assert metadata["page_source"] == "https://mbzuai.ac.ae/"
        assert metadata["canonical_url"] == "https://mbzuai.ac.ae/"

    def test_knowledge_graph_includes_website_link_graph_edges(self, tmp_dir):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.knowledge_graph_formatter import KnowledgeGraphFormatter

        retrieval_bundle_file = tmp_dir / "retrieval_bundle.json"
        page_graph_file = tmp_dir / "canonical_page_link_graph.json"
        atomic_write_json(
            retrieval_bundle_file,
            {
                "parent_records": [
                    {
                        "id": "page-admissions",
                        "parent_type": "page",
                        "document_id": "doc-admissions",
                        "document_title": "Admissions",
                        "source_url": "https://mbzuai.ac.ae/en/study/admissions",
                    }
                ],
                "chunk_records": [
                    {
                        "id": "chunk-admissions",
                        "document_id": "doc-admissions",
                        "document_title": "Admissions",
                        "source_url": "https://mbzuai.ac.ae/en/study/admissions",
                        "text": "Admissions requirements.",
                        "chunk_index": 0,
                        "chunk_count": 1,
                    }
                ],
                "media_records": [],
                "fact_records": [],
            },
        )
        atomic_write_json(
            page_graph_file,
            {
                "nodes": [
                    {"id": "page:a", "url": "https://mbzuai.ac.ae/en/study/admissions", "label": "Admissions"},
                    {"id": "page:b", "url": "https://mbzuai.ac.ae/en/study/scholarships", "label": "Scholarships"},
                ],
                "edges": [
                    {
                        "id": "edge:ab",
                        "source_id": "page:a",
                        "target_id": "page:b",
                        "source_url": "https://mbzuai.ac.ae/en/study/admissions",
                        "target_url": "https://mbzuai.ac.ae/en/study/scholarships",
                        "properties": {"link_type": "internal"},
                    }
                ],
            },
        )
        ctx = StageContext(
            run_id="r-graph",
            project_name="mbzuai",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={
                "retrieval_bundle_file": str(retrieval_bundle_file),
                "canonical_page_link_graph_file": str(page_graph_file),
            },
            stage_definition={"type": "formatter", "plugin": "knowledge_graph"},
            stage_id="format_graph",
        )

        result = run_async(KnowledgeGraphFormatter().execute(ctx))

        assert result.status.value == "completed"
        assert result.metrics["website_page_nodes"] == 2
        assert result.metrics["website_link_edges"] == 1
        graph = load_json_safe(result.outputs["knowledge_graph_file"])
        assert any(edge["edge_type"] == "WEBSITE_LINKS_TO" for edge in graph["edges"])


class TestBenchmarkIO:
    def test_evaluate_standard_rankings_scores_ranked_ids(self, tmp_dir):
        from pipeline.evaluation.benchmark_io import evaluate_standard_rankings

        dataset_dir = tmp_dir / "benchmark"
        dataset_dir.mkdir()
        (dataset_dir / "corpus.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "doc1", "text": "Alpha"}),
                    json.dumps({"id": "doc2", "text": "Beta"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "queries.jsonl").write_text(
            json.dumps({"id": "q1", "text": "Alpha question"}) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "qrels.jsonl").write_text(
            json.dumps({"query_id": "q1", "doc_id": "doc1", "relevance": 1}) + "\n",
            encoding="utf-8",
        )
        rankings_path = tmp_dir / "rankings.jsonl"
        rankings_path.write_text(
            json.dumps({"query_id": "q1", "ranked_ids": ["doc1", "doc2"]}) + "\n",
            encoding="utf-8",
        )

        report = evaluate_standard_rankings(dataset_dir=dataset_dir, rankings_path=rankings_path, k=10)
        assert report["overall"]["eligible_query_count"] == 1
        assert report["overall"]["hit_at_k"] == 1.0
        assert report["overall"]["mrr_at_k"] == 1.0
        assert report["overall"]["ndcg_at_k"] == 1.0

    def test_evaluate_standard_rankings_accepts_mapping_json(self, tmp_dir):
        from pipeline.evaluation.benchmark_io import evaluate_standard_rankings

        dataset_dir = tmp_dir / "benchmark"
        dataset_dir.mkdir()
        (dataset_dir / "corpus.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "doc1", "text": "Alpha"}),
                    json.dumps({"id": "doc2", "text": "Beta"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "queries.jsonl").write_text(
            json.dumps({"id": "q1", "text": "Alpha question"}) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "qrels.jsonl").write_text(
            json.dumps({"query_id": "q1", "doc_id": "doc1", "relevance": 1}) + "\n",
            encoding="utf-8",
        )
        rankings_path = tmp_dir / "rankings.json"
        rankings_path.write_text(
            json.dumps({"q1": [{"doc_id": "doc1", "score": 1.0}, {"doc_id": "doc2", "score": 0.5}]}) + "\n",
            encoding="utf-8",
        )

        report = evaluate_standard_rankings(dataset_dir=dataset_dir, rankings_path=rankings_path, k=10)
        assert report["overall"]["eligible_query_count"] == 1
        assert report["overall"]["hit_at_k"] == 1.0
        assert report["overall"]["mrr_at_k"] == 1.0
        assert report["overall"]["ndcg_at_k"] == 1.0


class TestBenchmarkRunner:
    def _write_standard_benchmark(self, dataset_dir):
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "corpus.jsonl").write_text(
            "\n".join(
                [
                    json.dumps({"id": "doc-a", "title": "Campus", "text": "MBZUAI campus facilities and accommodation"}),
                    json.dumps({"id": "doc-b", "title": "Research", "text": "Research labs and faculty"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "queries.jsonl").write_text(
            json.dumps({"id": "q1", "text": "campus facilities"}) + "\n",
            encoding="utf-8",
        )
        (dataset_dir / "qrels.jsonl").write_text(
            json.dumps({"query_id": "q1", "doc_id": "doc-a", "relevance": 1.0}) + "\n",
            encoding="utf-8",
        )

    def test_run_standard_benchmark_retrieval_writes_rankings_and_scores(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.benchmark_runner import run_standard_benchmark_retrieval

        dataset_dir = tmp_dir / "benchmark"
        self._write_standard_benchmark(dataset_dir)

        monkeypatch.setattr(
            "pipeline.evaluation.benchmark_runner.load_config",
            lambda _name: {"embedder": {"model": "fake-benchmark-model", "output_dimensionality": 3}},
        )
        monkeypatch.setattr(
            "pipeline.evaluation.benchmark_runner._make_gemini_client",
            lambda: object(),
        )

        def _fake_embed(_client, *, model, texts, task_type, output_dimensionality):
            assert model == "fake-benchmark-model"
            assert output_dimensionality == 3
            vectors = []
            for text in texts:
                lowered = str(text).lower()
                if "campus facilities" in lowered:
                    vectors.append([1.0, 0.0, 0.0])
                elif "research" in lowered:
                    vectors.append([0.0, 1.0, 0.0])
                else:
                    vectors.append([0.0, 0.0, 1.0])
            return vectors

        monkeypatch.setattr(
            "pipeline.evaluation.benchmark_runner._embed_text_batch",
            _fake_embed,
        )

        report = run_standard_benchmark_retrieval(
            config_name="unused",
            dataset_dir=dataset_dir,
            output_rankings_path=tmp_dir / "rankings.jsonl",
            top_k=2,
            dense_top_k=2,
            sparse_top_k=2,
            batch_size=8,
        )

        assert report["overall"]["hit_at_k"] == 1.0
        assert report["overall"]["mrr_at_k"] == 1.0
        rankings = (tmp_dir / "rankings.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert rankings
        first = json.loads(rankings[0])
        assert first["query_id"] == "q1"
        assert first["ranked_ids"][0] == "doc-a"

    def test_run_standard_benchmark_retrieval_reuses_embedding_caches(self, tmp_dir, monkeypatch):
        from pipeline.evaluation.benchmark_runner import run_standard_benchmark_retrieval

        dataset_dir = tmp_dir / "benchmark"
        self._write_standard_benchmark(dataset_dir)

        monkeypatch.setattr(
            "pipeline.evaluation.benchmark_runner.load_config",
            lambda _name: {"embedder": {"model": "fake-benchmark-model", "output_dimensionality": 3}},
        )
        monkeypatch.setattr(
            "pipeline.evaluation.benchmark_runner._make_gemini_client",
            lambda: object(),
        )

        state = {"calls": 0}

        def _fake_embed(_client, *, model, texts, task_type, output_dimensionality):
            state["calls"] += 1
            return [[1.0, 0.0, 0.0] if "campus" in str(text).lower() else [0.0, 1.0, 0.0] for text in texts]

        monkeypatch.setattr(
            "pipeline.evaluation.benchmark_runner._embed_text_batch",
            _fake_embed,
        )

        doc_cache = tmp_dir / "doc_cache.json"
        query_cache = tmp_dir / "query_cache.json"
        first = run_standard_benchmark_retrieval(
            config_name="unused",
            dataset_dir=dataset_dir,
            output_rankings_path=tmp_dir / "rankings_first.jsonl",
            doc_cache_path=doc_cache,
            query_cache_path=query_cache,
            batch_size=8,
        )
        second = run_standard_benchmark_retrieval(
            config_name="unused",
            dataset_dir=dataset_dir,
            output_rankings_path=tmp_dir / "rankings_second.jsonl",
            doc_cache_path=doc_cache,
            query_cache_path=query_cache,
            batch_size=8,
        )

        assert state["calls"] == 2
        assert first["document_cache"]["miss_count"] == 2
        assert first["query_cache"]["miss_count"] == 1
        assert second["document_cache"]["hit_count"] == 2
        assert second["query_cache"]["hit_count"] == 1


class TestRetrievalGateChecks:
    def test_gate_checks_allow_tiny_float_rounding_error(self):
        from pipeline.evaluation.retrieval_eval import check_metric_gates

        report = {
            "overall": {"chunk_recall_at_10": 0.7499999999999999},
            "by_query_type": {},
            "by_source_type": {},
            "by_benchmark_tag": {},
        }
        gates = {"overall": {"chunk_recall_at_10": 0.75}}
        assert check_metric_gates(report, gates) == []

    def test_gate_checks_support_benchmark_tag_slices(self):
        from pipeline.evaluation.retrieval_eval import check_metric_gates

        report = {
            "overall": {},
            "by_query_type": {},
            "by_source_type": {},
            "by_benchmark_tag": {
                "relation_heavy_fact": {
                    "chunk_hit_at_5": 0.92,
                    "chunk_mrr_at_10": 0.81,
                }
            },
        }
        gates = {
            "by_benchmark_tag": {
                "relation_heavy_fact": {
                    "chunk_hit_at_5": {"min": 0.90},
                    "chunk_mrr_at_10": {"min": 0.80},
                }
            }
        }
        assert check_metric_gates(report, gates) == []

    def test_retrieval_cache_key_ignores_unrelated_graph_config_for_vector_backend(self):
        from pipeline.evaluation.retrieval_eval import _retrieval_cache_key

        config_a = {
            "retrieval": {"retriever_backend": "vector", "top_k": 10},
            "embedder": {"model": "gemini-embedding-2-preview", "output_dimensionality": 1536},
            "graph": {"extraction_max_items": 4000, "neo4j_namespace": "a"},
        }
        config_b = {
            "retrieval": {"retriever_backend": "vector", "top_k": 10},
            "embedder": {"model": "gemini-embedding-2-preview", "output_dimensionality": 1536},
            "graph": {"extraction_max_items": 1000, "neo4j_namespace": "b"},
        }

        key_a = _retrieval_cache_key(
            config_name="cfg",
            work_dir="/tmp/run",
            query="Where is MBZUAI located?",
            model="gemini-embedding-2-preview",
            output_dimensionality=1536,
            config_payload=config_a,
        )
        key_b = _retrieval_cache_key(
            config_name="cfg",
            work_dir="/tmp/run",
            query="Where is MBZUAI located?",
            model="gemini-embedding-2-preview",
            output_dimensionality=1536,
            config_payload=config_b,
        )

        assert key_a == key_b


class TestRetrievalAblationComparison:
    def test_compare_reports_promotes_candidate_without_primary_metric_regressions(self):
        from pipeline.evaluation.ablation import compare_retrieval_reports

        baseline = {
            "query_count": 2,
            "overall": {
                "chunk_hit_at_5": 0.50,
                "chunk_recall_at_10": 0.50,
                "chunk_mrr_at_10": 0.50,
                "chunk_ndcg_at_10": 0.50,
                "parent_hit_at_5": 0.50,
                "no_answer_violation_rate": 0.10,
            },
        }
        candidate = {
            "query_count": 2,
            "overall": {
                "chunk_hit_at_5": 0.60,
                "chunk_recall_at_10": 0.55,
                "chunk_mrr_at_10": 0.50,
                "chunk_ndcg_at_10": 0.51,
                "parent_hit_at_5": 0.50,
                "no_answer_violation_rate": 0.05,
            },
        }

        report = compare_retrieval_reports(baseline=baseline, candidate=candidate)

        assert report["passed"] is True
        assert report["recommendation"] == "promote"
        assert report["regression_count"] == 0
        assert any(row["metric"] == "no_answer_violation_rate" and row["status"] == "improvement" for row in report["metrics"])

    def test_compare_reports_holds_candidate_on_regression(self):
        from pipeline.evaluation.ablation import compare_retrieval_reports

        baseline = {"overall": {"chunk_mrr_at_10": 0.70}}
        candidate = {"overall": {"chunk_mrr_at_10": 0.60}}

        report = compare_retrieval_reports(
            baseline=baseline,
            candidate=candidate,
            metrics=["chunk_mrr_at_10"],
            regression_tolerance=0.001,
        )

        assert report["passed"] is False
        assert report["recommendation"] == "hold"
        assert report["regressions"][0]["metric"] == "chunk_mrr_at_10"


class TestRagasEvaluation:
    def test_ragas_eval_requires_optional_dependency(self, tmp_dir, monkeypatch):
        from pipeline.evaluation import ragas_eval as module

        rows_path = tmp_dir / "predictions.jsonl"
        rows_path.write_text(
            json.dumps(
                {
                    "user_input": "Where is MBZUAI located?",
                    "response": "In Abu Dhabi.",
                    "retrieved_contexts": ["MBZUAI is in Masdar City, Abu Dhabi."],
                    "reference": "MBZUAI is located in Masdar City, Abu Dhabi.",
                }
            )
            + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(
            module,
            "_import_ragas_components",
            lambda: (_ for _ in ()).throw(
                RuntimeError("RAGAS evaluation requires optional dependencies. Install `ragas` in the active venv first.")
            ),
        )

        with pytest.raises(RuntimeError, match="RAGAS evaluation requires optional dependencies"):
            module.run_ragas_evaluation(predictions_path=rows_path)


class TestGoogleGenaiImports:
    def test_import_genai_falls_back_to_importlib(self, monkeypatch):
        from pipeline.core import google_genai as module

        fake_genai = SimpleNamespace(Client=object)
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "google" and "genai" in tuple(fromlist or ()):
                raise ImportError("simulated namespace failure")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(
            module.importlib,
            "import_module",
            lambda name: fake_genai if name == "google.genai" else original_import(name),
        )

        assert module.import_genai() is fake_genai

    def test_import_genai_types_falls_back_to_importlib(self, monkeypatch):
        from pipeline.core import google_genai as module

        fake_types = SimpleNamespace(EmbedContentConfig=object)
        original_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "google.genai" and "types" in tuple(fromlist or ()):
                raise ImportError("simulated namespace failure")
            return original_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        monkeypatch.setattr(
            module.importlib,
            "import_module",
            lambda name: fake_types if name == "google.genai.types" else original_import(name),
        )

        assert module.import_genai_types() is fake_types


class TestSemanticGraphExtractionSelection:
    def test_selection_balances_across_documents_before_global_cap(self):
        from pipeline.stages.formatters.semantic_graph_extract_formatter import _select_items_for_extraction

        items = [
            {"source_id": "a1", "document_id": "doc-a", "source_kind": "fact"},
            {"source_id": "a2", "document_id": "doc-a", "source_kind": "fact"},
            {"source_id": "a3", "document_id": "doc-a", "source_kind": "fact"},
            {"source_id": "b1", "document_id": "doc-b", "source_kind": "fact"},
            {"source_id": "b2", "document_id": "doc-b", "source_kind": "fact"},
            {"source_id": "c1", "document_id": "doc-c", "source_kind": "fact"},
        ]

        selected = _select_items_for_extraction(
            items,
            extraction_max_items=4,
            extraction_max_items_per_document=2,
        )

        assert [item["source_id"] for item in selected] == ["a1", "b1", "c1", "a2"]

    def test_selection_prioritizes_relation_rich_institutional_facts(self):
        from pipeline.stages.formatters.semantic_graph_extract_formatter import _select_items_for_extraction

        items = [
            {
                "source_id": "video-1",
                "source_kind": "fact",
                "document_id": "doc-video",
                "heading": "Video: YouTube video player",
                "text": "Watch the recorded talk on YouTube.",
                "source_url": "https://mbzuai.ac.ae/speaker-series/example",
            },
            {
                "source_id": "abstract-1",
                "source_kind": "fact",
                "document_id": "doc-paper",
                "heading": "Abstract",
                "text": "Abstract We present a novel model for multimodal representation learning.",
                "source_url": "",
            },
            {
                "source_id": "faq-1",
                "source_kind": "fact",
                "document_id": "doc-faq",
                "heading": "FAQ",
                "text": "Where is MBZUAI located? MBZUAI is located in Masdar City, Abu Dhabi.",
                "source_url": "https://mbzuai.ac.ae/faq/",
            },
        ]

        selected = _select_items_for_extraction(
            items,
            extraction_max_items=1,
            extraction_max_items_per_document=1,
            extraction_min_score=5.0,
            priority_host_suffixes=["mbzuai.ac.ae"],
        )

        assert [item["source_id"] for item in selected] == ["faq-1"]

    def test_selection_filters_low_signal_documents_when_min_score_is_set(self):
        from pipeline.stages.formatters.semantic_graph_extract_formatter import _select_items_for_extraction

        items = [
            {
                "source_id": "low-1",
                "source_kind": "fact",
                "document_id": "doc-low-1",
                "heading": "Abstract",
                "text": "Abstract We present a method.",
                "source_url": "",
            },
            {
                "source_id": "low-2",
                "source_kind": "fact",
                "document_id": "doc-low-2",
                "heading": "Video: Embedded video",
                "text": "Watch the video.",
                "source_url": "https://mbzuai.ac.ae/videos/example",
            },
            {
                "source_id": "high-1",
                "source_kind": "fact",
                "document_id": "doc-high",
                "heading": "Campus Facilities",
                "text": "The University provides a range of support services and student facilities on campus.",
                "source_url": "https://mbzuai.ac.ae/campus-facilities/",
            },
        ]

        selected = _select_items_for_extraction(
            items,
            extraction_max_items=5,
            extraction_max_items_per_document=2,
            extraction_min_score=5.0,
            priority_host_suffixes=["mbzuai.ac.ae"],
        )

        assert [item["source_id"] for item in selected] == ["high-1"]


class TestRetrievalService:
    def test_retrieval_service_loads_once_and_serves_requests(self, tmp_dir, monkeypatch):
        from fastapi.testclient import TestClient

        from pipeline.service.retrieval_api import create_retrieval_service_app

        class FakeRetriever:
            def __init__(self):
                self.calls = []

            def retrieve(self, query: str):
                self.calls.append(query)
                return {
                    "mode": "fact",
                    "selected_chunk_ids": ["chunk-1"],
                    "selected_parent_ids": ["parent-1"],
                    "selected_media_ids": [],
                    "graph_fact_ids": [],
                    "graph_assertion_ids": [],
                    "retrieval_documents": [{"id": "chunk-1", "document_title": "MBZUAI FAQ"}],
                    "abstained": False,
                }

        fake = FakeRetriever()

        def fake_from_config(*, config_name, work_dir):
            assert config_name == "cfg"
            assert Path(work_dir) == tmp_dir.resolve()
            return fake

        monkeypatch.setattr(
            "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
            fake_from_config,
        )

        app = create_retrieval_service_app(
            config_name="cfg",
            work_dir=tmp_dir,
            max_concurrency=2,
            request_timeout_seconds=5.0,
        )

        with TestClient(app) as client:
            health = client.get("/healthz")
            assert health.status_code == 200
            assert health.json()["ready"] is True
            assert "work_dir" not in health.json()

            health_alias = client.get("/health")
            assert health_alias.status_code == 200
            assert health_alias.json()["ready"] is True

            ready = client.get("/readyz")
            assert ready.status_code == 200
            assert ready.json()["service"] == "retriever"

            ready_alias = client.get("/ready")
            assert ready_alias.status_code == 200
            assert ready_alias.json()["service"] == "retriever"

            response = client.post("/retrieve", json={"query": "Where is MBZUAI located?"})
            assert response.status_code == 200
            payload = response.json()
            assert payload["service_backend"] == "retrieval_service"
            assert payload["service_config_name"] == "cfg"
            assert "service_work_dir" not in payload
            assert payload["selected_chunk_ids"] == ["chunk-1"]
            assert payload["retrieval_documents"][0]["document_title"] == "MBZUAI FAQ"
            assert payload["service_cache_hit"] is False
            assert fake.calls == ["Where is MBZUAI located?"]

            cached_response = client.post("/retrieve", json={"query": "  where   is mbzuai located?  "})
            assert cached_response.status_code == 200
            cached_payload = cached_response.json()
            assert cached_payload["service_cache_hit"] is True
            assert cached_payload["selected_chunk_ids"] == ["chunk-1"]
            assert fake.calls == ["Where is MBZUAI located?"]

    def test_retrieval_service_returns_timeout(self, tmp_dir, monkeypatch):
        import time

        from fastapi.testclient import TestClient

        from pipeline.service.retrieval_api import create_retrieval_service_app

        class SlowRetriever:
            def retrieve(self, query: str):
                time.sleep(1.5)
                return {"mode": "fact", "selected_chunk_ids": [], "retrieval_documents": [], "abstained": True}

        monkeypatch.setattr(
            "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
            lambda **_: SlowRetriever(),
        )

        app = create_retrieval_service_app(
            config_name="cfg",
            work_dir=tmp_dir,
            max_concurrency=1,
            request_timeout_seconds=1.0,
            queue_timeout_seconds=0.05,
        )

        with TestClient(app) as client:
            response = client.post("/retrieve", json={"query": "test timeout"})
            assert response.status_code == 504
            assert response.json()["detail"] == "retrieval_timeout"

            saturated = client.post("/retrieve", json={"query": "second request while worker is stuck"})
            assert saturated.status_code == 503
            assert saturated.json()["detail"] == "retrieval_busy"
            assert client.get("/readyz").status_code == 503

            time.sleep(0.6)
            ready = client.get("/readyz")
            assert ready.status_code == 200
            attestation = client.get("/attestationz").json()
            assert attestation["timed_out_inflight"] == 0
            assert attestation["detached_inflight"] == 0
            assert attestation["queue_rejection_count"] == 1

    def test_cancelled_request_keeps_worker_capacity_reserved(self, tmp_dir, monkeypatch):
        import threading

        import httpx

        from pipeline.service.retrieval_api import create_retrieval_service_app

        started = threading.Event()
        finish = threading.Event()

        class BlockingRetriever:
            def retrieve(self, query: str):
                started.set()
                finish.wait(timeout=5.0)
                return {"mode": "fact", "selected_chunk_ids": [], "retrieval_documents": [], "abstained": True}

        monkeypatch.setattr(
            "pipeline.service.retrieval_api.AdaptiveHybridRetriever.from_config",
            lambda **_: BlockingRetriever(),
        )
        app = create_retrieval_service_app(
            config_name="cfg",
            work_dir=tmp_dir,
            max_concurrency=1,
            request_timeout_seconds=4.0,
            queue_timeout_seconds=0.05,
        )

        async def exercise_disconnect():
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                    request_task = asyncio.create_task(client.post("/retrieve", json={"query": "cancel me"}))
                    assert await asyncio.to_thread(started.wait, 1.0)
                    request_task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await request_task

                    saturated = await client.post("/retrieve", json={"query": "must wait"})
                    assert saturated.status_code == 503
                    assert saturated.json()["detail"] == "retrieval_busy"
                    assert (await client.get("/readyz")).status_code == 503

                    finish.set()
                    for _ in range(50):
                        health = (await client.get("/attestationz")).json()
                        if health["detached_inflight"] == 0:
                            break
                        await asyncio.sleep(0.01)
                    assert health["detached_inflight"] == 0
                    assert health["cancelled_request_count"] == 1
                    assert (await client.get("/readyz")).status_code == 200

        try:
            asyncio.run(exercise_disconnect())
        finally:
            finish.set()


class TestIndexedLocalRetrieval:
    def test_configured_vector_namespaces_alias_all_local_indexes(self):
        from pipeline.retrieval.adaptive_hybrid import _register_configured_namespace_aliases

        chunk_token_index = {"mbzuai": ["chunk-1"]}
        chunk_tokens = {"chunk-1": {"mbzuai"}}
        assertion_token_index = {"founded": ["assertion-1"]}
        assertion_tokens = {"assertion-1": {"founded"}}
        chunk_bm25 = object()
        token_indexes = {
            "chunks": chunk_token_index,
            "assertions": assertion_token_index,
        }
        tokens_by_id = {
            "chunks": chunk_tokens,
            "assertions": assertion_tokens,
        }
        bm25_indexes = {"chunks": (["chunk-1"], chunk_bm25)}

        _register_configured_namespace_aliases(
            configured_to_canonical={
                "mbzuai_main-chunks": "chunks",
                "mbzuai_main-assertions": "assertions",
                "mbzuai_main-media": "media",
            },
            token_indexes=token_indexes,
            tokens_by_id=tokens_by_id,
            bm25_indexes=bm25_indexes,
        )

        assert token_indexes["mbzuai_main-chunks"] is chunk_token_index
        assert tokens_by_id["mbzuai_main-chunks"] is chunk_tokens
        assert bm25_indexes["mbzuai_main-chunks"] is bm25_indexes["chunks"]
        assert token_indexes["mbzuai_main-assertions"] is assertion_token_index
        assert tokens_by_id["mbzuai_main-assertions"] is assertion_tokens
        assert "mbzuai_main-media" not in token_indexes

    def test_lexical_query_ids_use_posting_index_without_bm25(self):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
        retriever.enable_local_bm25_fallback = False
        retriever.local_index_max_postings_per_token = 64
        retriever.lexical_map = {
            "chunk-1": {"text": "MBZUAI is located in Masdar City Abu Dhabi."},
            "chunk-2": {"text": "Parking is available for students and visitors."},
        }
        retriever._namespace_token_index = {
            "chunks": {
                "mbzuai": ["chunk-1"],
                "located": ["chunk-1"],
                "masdar": ["chunk-1"],
                "parking": ["chunk-2"],
            }
        }
        retriever._namespace_tokens_by_id = {
            "chunks": {
                "chunk-1": {"mbzuai", "located", "masdar", "city", "abu", "dhabi"},
                "chunk-2": {"parking", "available", "students", "visitors"},
            }
        }
        retriever._bm25_by_namespace = {}

        assert retriever._lexical_query_ids(
            "Where is MBZUAI located?",
            2,
            namespace="chunks",
        ) == ["chunk-1"]

    def test_local_media_query_ids_only_rescore_indexed_candidates(self):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
        retriever.enable_local_bm25_fallback = False
        retriever.local_index_max_postings_per_token = 64
        retriever.local_media_candidate_pool = 8
        retriever.namespace_media = "media"
        retriever.lexical_map = {
            "media-1": {"text": "Campus map showing parking and facilities."},
            "media-2": {"text": "Completely unrelated image."},
        }
        retriever._namespace_token_index = {
            "media": {
                "campus": ["media-1"],
                "map": ["media-1"],
                "parking": ["media-1"],
                "unrelated": ["media-2"],
            }
        }
        retriever._namespace_tokens_by_id = {
            "media": {
                "media-1": {"campus", "map", "parking", "facilities"},
                "media-2": {"completely", "unrelated", "image"},
            }
        }
        retriever._bm25_by_namespace = {}
        retriever.media_map = {
            "media-1": {"id": "media-1"},
            "media-2": {"id": "media-2"},
        }
        scored_ids = []

        def fake_score(query: str, media: Dict[str, Any]) -> float:
            scored_ids.append(str(media.get("id") or ""))
            return 1.0

        retriever._score_media_relevance = fake_score

        assert retriever._local_media_query_ids("Show the campus parking map", top_k=2) == ["media-1"]
        assert scored_ids == ["media-1"]


class TestParentSelectionRanking:
    def test_select_parent_ids_prioritizes_top_chunk_parents_over_explicit_noise(self):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
        retriever.parent_candidate_top_k = 3
        retriever.parent_map = {
            "section-gold": {"id": "section-gold"},
            "page-gold": {"id": "page-gold"},
            "section-noise": {"id": "section-noise"},
            "page-noise": {"id": "page-noise"},
            "section-noise-2": {"id": "section-noise-2"},
            "page-noise-2": {"id": "page-noise-2"},
        }
        retriever.chunk_map = {
            "chunk-gold": {"section_key": "section-gold", "page_key": "page-gold"},
            "chunk-noise": {"section_key": "section-noise", "page_key": "page-noise"},
            "chunk-noise-2": {"section_key": "section-noise-2", "page_key": "page-noise-2"},
        }
        retriever.parent_map["section-noise"]["child_chunk_ids"] = ["chunk-noise"]
        retriever.parent_map["page-noise"]["child_chunk_ids"] = ["chunk-noise"]
        retriever.parent_map["section-noise-2"]["child_chunk_ids"] = ["chunk-noise-2"]
        retriever.parent_map["page-noise-2"]["child_chunk_ids"] = ["chunk-noise-2"]

        selected = retriever._select_parent_ids(
            ["chunk-gold", "chunk-noise", "chunk-noise-2"],
            explicit_parent_ids=["section-noise", "page-noise", "section-noise-2", "page-noise-2"],
        )

        assert "section-gold" in selected[:3]
        assert "page-gold" in selected[:5]

    def test_select_parent_ids_keeps_second_chunk_parent_within_top_five(self):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
        retriever.parent_candidate_top_k = 3
        retriever.parent_map = {
            "section-top": {"id": "section-top"},
            "page-top": {"id": "page-top"},
            "section-second": {"id": "section-second"},
            "page-second": {"id": "page-second"},
            "explicit-1": {"id": "explicit-1", "child_chunk_ids": ["chunk-third"]},
            "explicit-2": {"id": "explicit-2", "child_chunk_ids": ["chunk-third"]},
            "explicit-3": {"id": "explicit-3", "child_chunk_ids": ["chunk-third"]},
        }
        retriever.chunk_map = {
            "chunk-top": {"section_key": "section-top", "page_key": "page-top"},
            "chunk-second": {"section_key": "section-second", "page_key": "page-second"},
            "chunk-third": {"section_key": "", "page_key": ""},
        }

        selected = retriever._select_parent_ids(
            ["chunk-top", "chunk-second", "chunk-third"],
            explicit_parent_ids=["explicit-1", "explicit-2", "explicit-3"],
        )

        assert "section-second" in selected[:5] or "page-second" in selected[:5]

    def test_select_parent_ids_keeps_strong_explicit_parent_when_chunk_parents_are_noisy(self):
        from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

        retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
        retriever.parent_candidate_top_k = 3
        retriever.parent_map = {
            "section-a": {"id": "section-a"},
            "page-a": {"id": "page-a"},
            "section-b": {"id": "section-b"},
            "page-b": {"id": "page-b"},
            "section-c": {"id": "section-c"},
            "page-c": {"id": "page-c"},
            "explicit-gold": {"id": "explicit-gold", "child_chunk_ids": []},
        }
        retriever.chunk_map = {
            "chunk-a": {"section_key": "section-a", "page_key": "page-a"},
            "chunk-b": {"section_key": "section-b", "page_key": "page-b"},
            "chunk-c": {"section_key": "section-c", "page_key": "page-c"},
        }

        selected = retriever._select_parent_ids(
            ["chunk-a", "chunk-b", "chunk-c"],
            explicit_parent_ids=["explicit-gold"],
            prefer_explicit_parents=True,
        )

        assert "explicit-gold" in selected[:5]


class TestOpenAIAssertionPipeline:
    def test_extraction_slice_formatter_builds_section_aligned_slices(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.extraction_slice_formatter import ExtractionSliceFormatter

        chunks_file = tmp_dir / "chunk_index.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc-1",
                        "chunk_id": "chunk-1",
                        "chunk_index": 0,
                        "strategy": "hybrid",
                        "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                        "section_path": ["Leadership"],
                        "document_title": "Leadership",
                        "document_type": "webpage",
                        "source_markdown_path": str(tmp_dir / "leadership.md"),
                        "source_url": "https://mbzuai.ac.ae/about/leadership/",
                    },
                    {
                        "document_id": "doc-1",
                        "chunk_id": "chunk-2",
                        "chunk_index": 1,
                        "strategy": "hybrid",
                        "text": "Timothy Baldwin is the Provost of MBZUAI.",
                        "section_path": ["Leadership"],
                        "document_title": "Leadership",
                        "document_type": "webpage",
                        "source_markdown_path": str(tmp_dir / "leadership.md"),
                        "source_url": "https://mbzuai.ac.ae/about/leadership/",
                    },
                ],
                strategy="hybrid",
            ),
        )

        catalog = ArtifactCatalog()
        catalog.add(
            build_artifact_record(
                artifact_type="chunk_index",
                role="retrieval_chunks",
                producer_stage="chunk_content",
                uri=chunks_file.resolve().as_uri(),
                local_path=chunks_file,
            )
        )

        ctx = StageContext(
            run_id="r-slices",
            project_name="p",
            config={"assertions": {"slice_target_tokens": 200, "slice_max_tokens": 300}},
            work_dir=tmp_dir,
            previous_outputs={},
            stage_definition={"type": "formatter", "plugin": "extraction_slices"},
            stage_id="format_assertion_slices",
            artifact_catalog=catalog,
        )

        result = run_async(ExtractionSliceFormatter().execute(ctx))
        assert result.status.value == "completed"
        payload = load_json_safe(result.outputs["extraction_slices_file"])
        assert len(payload) == 1
        assert payload[0]["linked_chunk_ids"] == ["chunk-1", "chunk-2"]
        assert payload[0]["authority_class"] == "canonical_page"

    def test_openai_assertion_extract_validate_and_promote_pipeline(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.core.openai_client import json_completion
        from pipeline.stages.formatters.assertion_canonicalize_formatter import AssertionCanonicalizeFormatter
        from pipeline.stages.formatters.assertion_promote_formatter import AssertionPromoteFormatter
        from pipeline.stages.formatters.openai_assertion_extract_formatter import OpenAIAssertionExtractFormatter
        from pipeline.stages.formatters.openai_assertion_validate_formatter import OpenAIAssertionValidateFormatter

        slices_file = tmp_dir / "extraction_slices.json"
        atomic_write_json(
            slices_file,
            [
                {
                    "id": "slice-1",
                    "document_id": "doc-1",
                    "document_title": "Leadership",
                    "document_type": "webpage",
                    "source_url": "https://mbzuai.ac.ae/about/leadership/",
                    "source_markdown_path": str(tmp_dir / "leadership.md"),
                    "linked_chunk_ids": ["chunk-1"],
                    "section_anchor": ["Leadership"],
                    "section_paths": [["Leadership"]],
                    "page_numbers": [],
                    "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                    "authority_class": "canonical_page",
                    "authority_score": 1.0,
                }
            ],
        )

        monkeypatch.setenv("OPENAI_API_KEY", "test")

        import pipeline.stages.formatters.openai_assertion_extract_formatter as extract_mod
        import pipeline.stages.formatters.openai_assertion_validate_formatter as validate_mod

        monkeypatch.setattr(extract_mod, "make_openai_client", lambda: object())
        monkeypatch.setattr(
            extract_mod,
            "json_completion",
            lambda **kwargs: {
                "entities": [
                    {"name": "MBZUAI", "entity_type": "organization", "aliases": ["Mohamed bin Zayed University of Artificial Intelligence"], "confidence": 0.95},
                    {"name": "Professor Eric Xing", "entity_type": "person", "aliases": ["Eric Xing"], "confidence": 0.95},
                ],
                "assertions": [
                    {
                        "subject": "MBZUAI",
                        "subject_type": "organization",
                        "predicate": "role_holder",
                        "answer_type": "role_holder",
                        "answer_subtype": "president",
                        "object": "Professor Eric Xing",
                        "object_type": "person",
                        "qualifiers": ["current"],
                        "support_span": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                        "confidence": 0.96,
                    }
                ],
                "quality_flags": [],
            },
        )

        extract_ctx = StageContext(
            run_id="r-extract",
            project_name="p",
            config={"assertions": {"extract_model": "gpt-5-mini"}},
            work_dir=tmp_dir,
            previous_outputs={"extraction_slices_file": str(slices_file)},
            stage_definition={"type": "formatter", "plugin": "openai_assertion_extract"},
            stage_id="extract_assertions_openai",
        )
        extract_result = run_async(OpenAIAssertionExtractFormatter().execute(extract_ctx))
        assert extract_result.status.value == "completed"
        extracted_assertions = load_json_safe(
            extract_result.outputs["candidate_assertions_file"]
        )
        assert extracted_assertions[0]["source_slice_id"] == "slice-1"

        monkeypatch.setattr(validate_mod, "make_openai_client", lambda: object())
        monkeypatch.setattr(
            validate_mod,
            "json_completion",
            lambda **kwargs: {
                "validations": [
                    {
                        "assertion_id": load_json_safe(extract_result.outputs["candidate_assertions_file"])[0]["id"],
                        "decision": "supported",
                        "confidence": 0.98,
                        "normalized_subject": "MBZUAI",
                        "normalized_predicate": "role_holder",
                        "normalized_answer_subtype": "president",
                        "normalized_object": "Professor Eric Xing",
                        "reason": "Explicitly stated in the source.",
                    }
                ]
            },
        )

        validate_ctx = StageContext(
            run_id="r-validate",
            project_name="p",
            config={"assertions": {"validate_model": "gpt-5-nano"}},
            work_dir=tmp_dir,
            previous_outputs={
                "extraction_slices_file": str(slices_file),
                "candidate_assertions_file": extract_result.outputs["candidate_assertions_file"],
            },
            stage_definition={"type": "formatter", "plugin": "openai_assertion_validate"},
            stage_id="validate_assertions_openai",
        )
        validate_result = run_async(OpenAIAssertionValidateFormatter().execute(validate_ctx))
        assert validate_result.status.value == "completed"

        canonicalize_ctx = StageContext(
            run_id="r-canonicalize",
            project_name="p",
            config={},
            work_dir=tmp_dir,
            previous_outputs={
                "candidate_entities_file": extract_result.outputs["candidate_entities_file"],
                "validated_assertions_file": validate_result.outputs["validated_assertions_file"],
                "rejected_assertions_file": validate_result.outputs["rejected_assertions_file"],
            },
            stage_definition={"type": "formatter", "plugin": "assertion_canonicalize"},
            stage_id="canonicalize_assertions",
        )
        canonicalize_result = run_async(AssertionCanonicalizeFormatter().execute(canonicalize_ctx))
        assert canonicalize_result.status.value == "completed"

        promote_ctx = StageContext(
            run_id="r-promote",
            project_name="p",
            config={"assertions": {"promote_min_confidence": 0.5, "promote_min_authority_score": 0.2}},
            work_dir=tmp_dir,
            previous_outputs={
                "canonical_entities_file": canonicalize_result.outputs["canonical_entities_file"],
                "canonical_assertions_file": canonicalize_result.outputs["canonical_assertions_file"],
            },
            stage_definition={"type": "formatter", "plugin": "assertion_promote"},
            stage_id="promote_assertions",
        )
        promote_result = run_async(AssertionPromoteFormatter().execute(promote_ctx))
        assert promote_result.status.value == "completed"

        promoted_assertions = load_json_safe(promote_result.outputs["promoted_assertions_file"])
        assert len(promoted_assertions) == 1
        assert promoted_assertions[0]["answer_type"] == "role_holder"
        assert promoted_assertions[0]["answer_subtype"] == "president"

        captured_kwargs = {}

        class _FakeChatCompletions:
            def create(self, **kwargs):
                captured_kwargs.update(kwargs)
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
                )

        fake_client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeChatCompletions()))
        payload = json_completion(
            client=fake_client,
            model="gpt-5-mini",
            system_prompt="system",
            user_prompt="user",
            temperature=0.0,
            max_completion_tokens=64,
            json_schema={"name": "demo", "strict": True, "schema": {"type": "object", "properties": {}, "required": []}},
        )
        assert payload["ok"] is True
        assert "temperature" not in captured_kwargs
        assert captured_kwargs["response_format"]["type"] == "json_schema"
        assert "reasoning_effort" not in captured_kwargs

        captured_kwargs.clear()
        payload = json_completion(
            client=fake_client,
            model="gpt-4.1-mini",
            system_prompt="system",
            user_prompt="user",
            temperature=0.0,
            max_completion_tokens=64,
            reasoning_effort="minimal",
        )
        assert payload["ok"] is True
        assert captured_kwargs["temperature"] == 0.0
        assert captured_kwargs["response_format"]["type"] == "json_object"
        assert captured_kwargs["reasoning_effort"] == "minimal"

    def test_openai_assertion_extract_filters_by_authority_class(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.openai_assertion_extract_formatter import OpenAIAssertionExtractFormatter

        slices_file = tmp_dir / "extraction_slices.json"
        atomic_write_json(
            slices_file,
            [
                {
                    "id": "slice-news",
                    "document_title": "News",
                    "document_type": "webpage",
                    "source_url": "https://mbzuai.ac.ae/news/example/",
                    "source_markdown_path": str(tmp_dir / "news.md"),
                    "linked_chunk_ids": ["chunk-news"],
                    "text": "A news article.",
                    "authority_class": "news_page",
                    "authority_score": 0.3,
                },
                {
                    "id": "slice-leadership",
                    "document_title": "Leadership",
                    "document_type": "webpage",
                    "source_url": "https://mbzuai.ac.ae/about/leadership/",
                    "source_markdown_path": str(tmp_dir / "leadership.md"),
                    "linked_chunk_ids": ["chunk-leadership"],
                    "text": "Professor Eric Xing is the President of MBZUAI.",
                    "authority_class": "canonical_page",
                    "authority_score": 1.0,
                },
            ],
        )

        monkeypatch.setenv("OPENAI_API_KEY", "test")
        import pipeline.stages.formatters.openai_assertion_extract_formatter as extract_mod

        calls = []
        monkeypatch.setattr(extract_mod, "make_openai_client", lambda: object())

        def _fake_json_completion(**kwargs):
            calls.append(kwargs["user_prompt"])
            return {"entities": [], "assertions": [], "quality_flags": []}

        monkeypatch.setattr(extract_mod, "json_completion", _fake_json_completion)

        ctx = StageContext(
            run_id="r-filter",
            project_name="p",
            config={"assertions": {"allowed_authority_classes": ["canonical_page"]}},
            work_dir=tmp_dir,
            previous_outputs={"extraction_slices_file": str(slices_file)},
            stage_definition={"type": "formatter", "plugin": "openai_assertion_extract"},
            stage_id="extract_assertions_openai",
        )

        result = run_async(OpenAIAssertionExtractFormatter().execute(ctx))
        assert result.status.value == "completed"
        assert len(calls) == 1
        assert "Leadership" in calls[0]
        payload = load_json_safe(result.outputs["assertion_extract_results_file"])
        assert len(payload) == 1
        assert payload[0]["slice_id"] == "slice-leadership"

    def test_retrieval_formatter_uses_promoted_assertions(self, tmp_dir):
        from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
        from pipeline.core.base import StageContext
        from pipeline.core.chunking import build_chunk_index
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.formatters.retrieval_bundle_v2_formatter import RetrievalBundleV2Formatter

        md_file = tmp_dir / "leadership.md"
        md_file.write_text("# Leadership\n\nProfessor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.", encoding="utf-8")
        chunks_file = tmp_dir / "chunk_index.json"
        atomic_write_json(
            chunks_file,
            build_chunk_index(
                [
                    {
                        "document_id": "doc-1",
                        "chunk_id": "leadership-chunk",
                        "chunk_index": 0,
                        "strategy": "hybrid",
                        "text": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                        "section_path": ["Leadership"],
                        "document_title": "Leadership",
                        "document_type": "webpage",
                        "source_backend": "markitdown",
                        "source_file": str(md_file),
                        "source_markdown_path": str(md_file),
                        "source_url": "https://mbzuai.ac.ae/about/leadership/",
                    }
                ],
                strategy="hybrid",
            ),
        )
        promoted_entities_file = tmp_dir / "promoted_entities.json"
        promoted_assertions_file = tmp_dir / "promoted_assertions.json"
        atomic_write_json(
            promoted_entities_file,
            [
                {
                    "id": "entity:mbzuai",
                    "entity_type": "organization",
                    "canonical_name": "MBZUAI",
                    "aliases": ["Mohamed bin Zayed University of Artificial Intelligence"],
                    "confidence": 0.9,
                    "source_chunk_ids": ["leadership-chunk"],
                    "source_parent_ids": [],
                    "source_urls": ["https://mbzuai.ac.ae/about/leadership/"],
                    "document_titles": ["Leadership"],
                },
                {
                    "id": "entity:eric-xing",
                    "entity_type": "person",
                    "canonical_name": "Professor Eric Xing",
                    "aliases": ["Eric Xing"],
                    "confidence": 0.9,
                    "source_chunk_ids": ["leadership-chunk"],
                    "source_parent_ids": [],
                    "source_urls": ["https://mbzuai.ac.ae/about/leadership/"],
                    "document_titles": ["Leadership"],
                },
            ],
        )
        atomic_write_json(
            promoted_assertions_file,
            [
                {
                    "id": "assertion:president",
                    "subject_name": "MBZUAI",
                    "subject_type": "organization",
                    "subject_entity_id": "entity:mbzuai",
                    "predicate": "role_holder",
                    "relation_type": "role_holder",
                    "answer_type": "role_holder",
                    "answer_subtype": "president",
                    "object_name": "Professor Eric Xing",
                    "object_value": "Professor Eric Xing",
                    "object_type": "person",
                    "object_entity_id": "entity:eric-xing",
                    "qualifiers": ["current"],
                    "support_span": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                    "evidence": "Professor Eric Xing is the President of Mohamed bin Zayed University of Artificial Intelligence.",
                    "confidence": 0.97,
                    "authority_class": "canonical_page",
                    "authority_score": 1.0,
                    "freshness_score": 1.0,
                    "source_doc_id": "doc-1",
                    "source_chunk_ids": ["leadership-chunk"],
                    "source_parent_ids": [],
                    "source_fact_ids": [],
                    "source_url": "https://mbzuai.ac.ae/about/leadership/",
                    "source_markdown_path": str(md_file),
                    "document_title": "Leadership",
                    "validator_decision": "supported",
                    "text": "Professor Eric Xing is the president of MBZUAI.",
                }
            ],
        )

        catalog = ArtifactCatalog()
        catalog.add(
            build_artifact_record(
                artifact_type="chunk_index",
                role="retrieval_chunks",
                producer_stage="chunk_content",
                uri=chunks_file.resolve().as_uri(),
                local_path=chunks_file,
            )
        )
        catalog.add(
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_html",
                uri=md_file.resolve().as_uri(),
                local_path=md_file,
                metadata={"source_url": "https://mbzuai.ac.ae/about/leadership/"},
            )
        )

        ctx = StageContext(
            run_id="r-bundle-v2",
            project_name="p",
            config={"formatter": {}},
            work_dir=tmp_dir,
            previous_outputs={
                "promoted_entities_file": str(promoted_entities_file),
                "promoted_assertions_file": str(promoted_assertions_file),
            },
            stage_definition={"type": "formatter", "plugin": "retrieval_bundle_v2"},
            stage_id="format_retrieval",
            artifact_catalog=catalog,
        )

        result = run_async(RetrievalBundleV2Formatter().execute(ctx))
        bundle = load_json_safe(result.outputs["retrieval_bundle_file"])
        assert result.status.value == "completed"
        assert bundle["assertion_records"]
        assert bundle["entity_records"]
        assert bundle["answer_records"]
        assert bundle["answer_records"][0]["answer_type"] == "role_holder"
        assert bundle["answer_records"][0]["answer_subtype"] == "president"

    def test_build_answer_records_from_assertions_normalizes_inverted_role_holder(self):
        from pipeline.core.assertions import build_answer_records_from_assertions

        answers = build_answer_records_from_assertions(
            [
                {
                    "id": "assertion:president-inverted",
                    "subject_name": "Professor Eric Xing",
                    "subject_type": "person",
                    "predicate": "role_holder",
                    "answer_type": "role_holder",
                    "answer_subtype": "president",
                    "object_name": "MBZUAI",
                    "object_value": "MBZUAI",
                    "object_type": "organization",
                    "confidence": 0.98,
                    "authority_score": 1.0,
                    "freshness_score": 1.0,
                    "support_span": "MBZUAI President, Professor Eric Xing",
                    "text": "MBZUAI is the president of Professor Eric Xing.",
                }
            ]
        )

        assert len(answers) == 1
        assert answers[0]["answer_type"] == "role_holder"
        assert answers[0]["answer_subtype"] == "president"
        assert answers[0]["subject_text"] == "MBZUAI"
        assert answers[0]["value"] == "Professor Eric Xing"

    def test_build_answer_records_from_assertions_preserves_establishment_law(self):
        from pipeline.core.assertions import build_answer_records_from_assertions

        answers = build_answer_records_from_assertions(
            [
                {
                    "id": "assertion:legal-basis",
                    "subject_name": "Mohamed bin Zayed University of Artificial Intelligence",
                    "subject_type": "organization",
                    "predicate": "legal_basis",
                    "answer_type": "legal_basis",
                    "answer_subtype": "establishment_law",
                    "object_name": "Law No. 25 of 2019",
                    "object_value": "Law No. 25 of 2019",
                    "object_type": "law",
                    "confidence": 0.97,
                    "authority_score": 0.94,
                    "freshness_score": 1.0,
                    "text": "Mohamed bin Zayed University of Artificial Intelligence was established under Law No. 25 of 2019.",
                }
            ]
        )

        assert len(answers) == 1
        assert answers[0]["answer_type"] == "legal_basis"
        assert answers[0]["answer_subtype"] == "establishment_law"
        assert answers[0]["value"] == "Law No. 25 of 2019"

    def test_gemini_embedder_uploads_assertion_namespace(self, tmp_dir, monkeypatch):
        from pipeline.core.base import StageContext
        from pipeline.core.io import atomic_write_json
        from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

        chunk_file = tmp_dir / "chunk_dense_records.json"
        assertion_file = tmp_dir / "assertion_dense_records.json"
        bundle_file = tmp_dir / "retrieval_bundle.json"
        atomic_write_json(chunk_file, [{"id": "chunk1", "dense_text": "Chunk text", "document_id": "doc1"}])
        atomic_write_json(
            assertion_file,
            [
                {
                    "id": "assertion:president",
                    "dense_text": "SUBJECT: MBZUAI\nPREDICATE: role_holder\nOBJECT: Professor Eric Xing\nProfessor Eric Xing is the President of MBZUAI.",
                    "text": "Professor Eric Xing is the President of MBZUAI.",
                    "answer_type": "role_holder",
                    "answer_subtype": "president",
                }
            ],
        )
        atomic_write_json(bundle_file, {"chunk_records": [], "assertion_records": []})

        monkeypatch.setenv("GEMINI_API_KEY", "test")
        monkeypatch.setenv("PINECONE_API_KEY", "test")

        fake_calls = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                fake_calls.append((namespace, len(vectors)))

            def delete(self, *, delete_all, namespace):
                fake_calls.append(("delete", namespace, delete_all))

        class FakePinecone:
            def __init__(self, api_key):
                self.api_key = api_key

            def has_index(self, name):
                return True

            def Index(self, name):
                return FakeIndex()

        import pipeline.stages.embedders.gemini_pinecone_embedder as mod

        monkeypatch.setattr(mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(mod, "_embed_text_batch", lambda *args, texts=None, **kwargs: [[0.1, 0.2] for _ in texts])
        monkeypatch.setitem(sys.modules, "pinecone", SimpleNamespace(Pinecone=FakePinecone, ServerlessSpec=lambda **kwargs: kwargs))

        ctx = StageContext(
            run_id="r-upload-assertions",
            project_name="p",
            config={
                "embedder": {
                    "model": "gemini-embedding-2-preview",
                    "pinecone_index": "test-index",
                    "output_dimensionality": 2,
                    "namespace_chunks": "chunks",
                    "namespace_assertions": "assertions",
                    "enable_sparse": False,
                    "enable_dense_assertions": True,
                }
            },
            work_dir=tmp_dir,
            previous_outputs={
                "chunk_embedding_file": str(chunk_file),
                "assertion_embedding_file": str(assertion_file),
                "retrieval_bundle_file": str(bundle_file),
            },
            stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
            stage_id="upload_retrieval",
        )

        result = run_async(GeminiPineconeEmbedder().execute(ctx))
        assert result.status.value == "completed"
        assert ("chunks", 1) in fake_calls
        assert ("assertions", 1) in fake_calls
        assert result.metrics["assertion_vectors_uploaded"] == 1

    def test_routed_rewrite_bundle_uses_query_planner_when_enabled(self):
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

        retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
        retriever.query_planner_enabled = True
        retriever.query_planner_model = "gpt-5-nano"
        retriever.query_planner_min_confidence = 0.5
        retriever.parallel_query_rewriting_enabled = False

        import pipeline.retrieval.routed_hybrid as mod

        original_planner = mod.plan_query
        try:
            mod.plan_query = lambda **kwargs: {
                "query_type": "fact",
                "vector_query": "MBZUAI president",
                "graph_query": "MBZUAI role_holder president",
                "answer_types": ["role_holder"],
                "entity_hints": ["MBZUAI"],
                "confidence": 0.92,
            }
            bundle = retriever._build_query_rewrite_bundle("Who is the president of MBZUAI?", relation_plan=None, query_mode="fact")
        finally:
            mod.plan_query = original_planner

        assert bundle.vector_query == "MBZUAI president"
        assert bundle.graph_query == "MBZUAI role_holder president"
        assert "openai_vector_plan" in bundle.labels

    def test_evidence_adjudicator_heuristic_prefers_support_hours(self, monkeypatch):
        from pipeline.core.evidence_adjudicator import adjudicate_factual_evidence

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = adjudicate_factual_evidence(
            query="What are the IT support working hours for the MBZUAI online screening exam?",
            intent_summary={
                "answer_types": ["hours"],
                "requested_roles": [],
                "subject_tokens": ["mbzuai", "online", "screening", "exam", "it", "support"],
                "subject_phrases": ["mbzuai online screening exam"],
                "strict_answer_required": True,
            },
            answer_documents=[
                {
                    "id": "official-hours",
                    "answer_type": "hours",
                    "answer_subtype": "official_working_hours",
                    "subject_text": "MBZUAI",
                    "text": "The hours for MBZUAI are 8:00 a.m. – 6:00 p.m., Monday to Thursday.",
                },
                {
                    "id": "support-hours",
                    "answer_type": "hours",
                    "answer_subtype": "support_hours",
                    "subject_text": "MBZUAI online screening exam",
                    "text": "IT support hours are 8:00 AM - 5:00 PM (UAE time) Monday to Thursday.",
                },
            ],
            fact_documents=[],
            retrieval_documents=[],
            model="gpt-5-nano",
        )

        assert result["abstain"] is False
        assert result["selected_answer_ids"] == ["support-hours"]
        assert result["method"] == "heuristic"

    def test_evidence_adjudicator_heuristic_prefers_current_board_chair(self, monkeypatch):
        from pipeline.core.evidence_adjudicator import adjudicate_factual_evidence

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = adjudicate_factual_evidence(
            query="Who chairs MBZUAI's Board of Trustees?",
            intent_summary={
                "answer_types": ["role_holder"],
                "requested_roles": ["board_chair"],
                "subject_tokens": ["mbzuai", "board", "trustees"],
                "subject_phrases": ["board of trustees"],
                "strict_answer_required": True,
            },
            answer_documents=[
                {
                    "id": "founding-chair",
                    "answer_type": "role_holder",
                    "answer_subtype": "board_chair",
                    "subject_text": "MBZUAI",
                    "text": "Dr. Someone was the founding chairman of MBZUAI.",
                },
                {
                    "id": "current-chair",
                    "answer_type": "role_holder",
                    "answer_subtype": "board_chair",
                    "subject_text": "MBZUAI",
                    "text": "The chair of MBZUAI's Board of Trustees is Khaldoon Khalifa Al Mubarak.",
                },
            ],
            fact_documents=[],
            retrieval_documents=[],
            model="gpt-5-nano",
        )

        assert result["abstain"] is False
        assert result["selected_answer_ids"] == ["current-chair"]

    def test_evidence_adjudicator_heuristic_abstains_when_subject_not_supported(self, monkeypatch):
        from pipeline.core.evidence_adjudicator import adjudicate_factual_evidence

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        result = adjudicate_factual_evidence(
            query="Who is the president of MBZUAI's New York campus?",
            intent_summary={
                "answer_types": ["role_holder"],
                "requested_roles": ["president"],
                "subject_tokens": ["mbzuai", "new", "york", "campus"],
                "subject_phrases": ["new york campus"],
                "strict_answer_required": True,
            },
            answer_documents=[
                {
                    "id": "mbzuai-president",
                    "answer_type": "role_holder",
                    "answer_subtype": "president",
                    "subject_text": "MBZUAI",
                    "text": "The president of MBZUAI is Professor Eric Xing.",
                }
            ],
            fact_documents=[],
            retrieval_documents=[
                {
                    "id": "chunk-1",
                    "text": "MBZUAI is located in Abu Dhabi.",
                    "source_url": "https://mbzuai.ac.ae/",
                }
            ],
            model="gpt-5-nano",
        )

        assert result["abstain"] is True
        assert result["selected_answer_ids"] == []

    def test_routed_retriever_applies_evidence_adjudicator_selection(self):
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever
        import pipeline.retrieval.routed_hybrid as mod

        retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
        retriever.evidence_adjudicator_enabled = True
        retriever.evidence_adjudicator_model = "gpt-5-nano"
        retriever.evidence_adjudicator_reasoning_effort = "minimal"
        retriever.evidence_adjudicator_min_confidence = 0.58
        retriever.evidence_adjudicator_max_completion_tokens = 800
        retriever.evidence_adjudicator_retries = 2
        retriever.evidence_adjudicator_retry_delay_sec = 1.0
        retriever.evidence_adjudicator_per_request_delay_sec = 0.0
        retriever.evidence_adjudicator_answer_limit = 4
        retriever.evidence_adjudicator_fact_limit = 4
        retriever.evidence_adjudicator_chunk_limit = 6

        original = mod.adjudicate_factual_evidence
        try:
            mod.adjudicate_factual_evidence = lambda **kwargs: {
                "used": True,
                "method": "openai",
                "abstain": False,
                "selected_answer_ids": ["support-hours"],
                "selected_fact_ids": [],
                "selected_chunk_ids": [],
                "reason": "qualifier_match",
                "confidence": 0.91,
            }
            payload = retriever._apply_evidence_adjudication(
                "What are the IT support working hours for the MBZUAI online screening exam?",
                {
                    "mode": "fact",
                    "abstained": False,
                    "selected_answer_ids": ["official-hours", "support-hours"],
                    "answer_documents": [
                        {
                            "id": "official-hours",
                            "answer_type": "hours",
                            "answer_subtype": "official_working_hours",
                            "text": "Official hours",
                        },
                        {
                            "id": "support-hours",
                            "answer_type": "hours",
                            "answer_subtype": "support_hours",
                            "text": "Support hours",
                        },
                    ],
                    "fact_documents": [],
                    "retrieval_documents": [
                        {"id": "official-hours", "text": "Official hours", "answer_type": "hours"},
                        {"id": "support-hours", "text": "Support hours", "answer_type": "hours"},
                        {"id": "chunk-1", "text": "Supporting chunk"},
                    ],
                },
            )
        finally:
            mod.adjudicate_factual_evidence = original

        assert payload["adjudication_used"] is True
        assert payload["selected_answer_ids"] == ["support-hours"]
        assert payload["answer_documents"][0]["id"] == "support-hours"
        assert payload["retrieval_documents"][0]["id"] == "support-hours"

    def test_routed_retriever_applies_evidence_adjudicator_abstention(self):
        from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever
        import pipeline.retrieval.routed_hybrid as mod

        retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
        retriever.evidence_adjudicator_enabled = True
        retriever.evidence_adjudicator_model = "gpt-5-nano"
        retriever.evidence_adjudicator_reasoning_effort = "minimal"
        retriever.evidence_adjudicator_min_confidence = 0.58
        retriever.evidence_adjudicator_max_completion_tokens = 800
        retriever.evidence_adjudicator_retries = 2
        retriever.evidence_adjudicator_retry_delay_sec = 1.0
        retriever.evidence_adjudicator_per_request_delay_sec = 0.0
        retriever.evidence_adjudicator_answer_limit = 4
        retriever.evidence_adjudicator_fact_limit = 4
        retriever.evidence_adjudicator_chunk_limit = 6

        original = mod.adjudicate_factual_evidence
        try:
            mod.adjudicate_factual_evidence = lambda **kwargs: {
                "used": True,
                "method": "openai",
                "abstain": True,
                "selected_answer_ids": [],
                "selected_fact_ids": [],
                "selected_chunk_ids": [],
                "reason": "subject_not_supported",
                "confidence": 0.93,
            }
            payload = retriever._apply_evidence_adjudication(
                "Who is the president of MBZUAI's New York campus?",
                {
                    "mode": "fact",
                    "abstained": False,
                    "selected_answer_ids": ["president-id"],
                    "selected_fact_ids": ["fact-1"],
                    "selected_chunk_ids": ["chunk-1"],
                    "selected_parent_ids": ["parent-1"],
                    "selected_media_ids": ["media-1"],
                    "answer_documents": [{"id": "president-id", "text": "President"}],
                    "fact_documents": [{"id": "fact-1", "text": "Fact"}],
                    "retrieval_documents": [{"id": "chunk-1", "text": "Chunk"}],
                    "media": [{"id": "media-1"}],
                },
            )
        finally:
            mod.adjudicate_factual_evidence = original

        assert payload["abstained"] is True
        assert payload["selected_answer_ids"] == []
        assert payload["selected_fact_ids"] == []
        assert payload["selected_chunk_ids"] == []
        assert payload["retrieval_documents"] == []


class TestMBZLegacyPineconeMetadata:
    def test_legacy_metadata_compaction_preserves_answer_contract_under_byte_limit(self):
        from pipeline.stages.embedders.mbzuai_legacy_pinecone_embedder import (
            _metadata_size_bytes,
            _serialize_legacy_metadata,
        )

        raw_metadata = {
            "page_source": "https://mbzuai.ac.ae/study/admissions/",
            "source": "https://mbzuai.ac.ae/study/admissions/",
            "canonical_url": "https://mbzuai.ac.ae/study/admissions/",
            "canonical_family_url": "https://mbzuai.ac.ae/study/admissions",
            "page_title": "Admission Requirements",
            "section_title": "Required documents",
            "breadcrumb": "Admissions > Required documents",
            "authority_class": "official",
            "authority_score": 1.0,
            "intent_tags": ["admissions", "requirements"],
            "context": "شروط القبول في جامعة محمد بن زايد للذكاء الاصطناعي. " * 900,
            "document_summary": "Admission summary. " * 900,
            "key_facts": ["Applicants must submit official documents. " * 30 for _ in range(24)],
            "keywords": ["admissions", "requirements", "scholarship"] * 20,
            "page_metadata": {"raw_html_snapshot": "x" * 50000, "title": "Admission Requirements"},
            "media": [{"url": "https://example.com/image.jpg", "description": "x" * 10000}],
            "images": [{"url": "https://example.com/image.jpg", "description": "x" * 10000}],
            "citation_anchor": {"source_url": "https://mbzuai.ac.ae/study/admissions/", "text": "x" * 10000},
        }

        metadata, stats = _serialize_legacy_metadata(raw_metadata, max_bytes=12000)

        assert _metadata_size_bytes(metadata) <= 12000
        assert metadata["page_source"] == raw_metadata["page_source"]
        assert metadata["canonical_family_url"] == raw_metadata["canonical_family_url"]
        assert metadata["page_title"] == "Admission Requirements"
        assert metadata["context"]
        assert "page_metadata" not in metadata
        assert "media" not in metadata
        assert stats["compacted"] is True
        assert stats["removed_fields"]

    def test_upload_records_compacts_metadata_before_upsert(self, tmp_dir):
        from pipeline.stages.embedders.mbzuai_legacy_pinecone_embedder import (
            _metadata_size_bytes,
            _upload_records,
        )

        upserts = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                upserts.append((namespace, vectors))

        docs = [
            {
                "id": "doc-1",
                "metadata": {
                    "page_source": "https://mbzuai.ac.ae/study/",
                    "source": "https://mbzuai.ac.ae/study/",
                    "context": "MBZUAI admissions context. " * 1000,
                    "document_summary": "Summary. " * 1500,
                    "page_metadata": {"raw": "x" * 40000},
                    "media": [{"description": "x" * 20000}],
                },
            }
        ]

        uploaded, stats = _upload_records(
            index=FakeIndex(),
            docs=docs,
            dense_vectors=[[0.1, 0.2]],
            sparse_vectors=[{"indices": [1], "values": [0.5]}],
            namespace="test-run",
            upsert_batch_size=1,
            progress_path=tmp_dir / "progress.json",
            progress_phase="summary",
            progress_totals={"summary": 1},
            progress_uploaded={"summary": 0},
            metadata_max_bytes=9000,
        )

        assert uploaded == 1
        assert upserts[0][0] == "test-run"
        vector = upserts[0][1][0]
        assert vector["sparse_values"]["indices"] == [1]
        assert _metadata_size_bytes(vector["metadata"]) <= 9000
        assert vector["metadata"]["page_source"] == "https://mbzuai.ac.ae/study/"
        assert "context" in vector["metadata"]
        assert stats["metadata_compacted_vectors"] == 1

    def test_upload_records_splits_large_upsert_payloads(self, tmp_dir):
        from pipeline.stages.embedders.mbzuai_legacy_pinecone_embedder import _json_size_bytes, _upload_records

        upserts = []

        class FakeIndex:
            def upsert(self, *, vectors, namespace):
                assert _json_size_bytes({"vectors": vectors, "namespace": namespace}) <= 50000
                upserts.append(vectors)

        docs = [
            {
                "id": f"doc-{idx}",
                "metadata": {
                    "page_source": "https://mbzuai.ac.ae/study/",
                    "context": "Admission context. " * 200,
                },
            }
            for idx in range(6)
        ]
        dense_vectors = [[0.123456789 for _ in range(1024)] for _ in docs]

        uploaded, stats = _upload_records(
            index=FakeIndex(),
            docs=docs,
            dense_vectors=dense_vectors,
            sparse_vectors=None,
            namespace="test-run",
            upsert_batch_size=6,
            progress_path=tmp_dir / "progress.json",
            progress_phase="text",
            progress_totals={"text": 6},
            progress_uploaded={"text": 0},
            metadata_max_bytes=35000,
            upsert_payload_max_bytes=50000,
        )

        assert uploaded == 6
        assert len(upserts) > 1
        assert stats["upsert_requests"] == len(upserts)

    def test_legacy_gemini_embedding_cache_is_bound_to_text_digest(self, tmp_dir, monkeypatch):
        from pipeline.core.io import atomic_write_json, load_json_safe
        from pipeline.stages.embedders.mbzuai_legacy_pinecone_embedder import (
            _generate_dense_embeddings,
            _texts_digest,
        )
        import pipeline.stages.embedders.gemini_pinecone_embedder as gemini_mod

        cache_path = tmp_dir / "legacy_dense_embedding_cache.json"
        atomic_write_json(
            cache_path,
            {
                "schema_version": 1,
                "engine": "gemini",
                "model": "gemini-embedding-2-preview",
                "output_dimensionality": 2,
                "text_count": 1,
                "text_digest": _texts_digest(["old text"]),
                "embeddings": [[9.0, 9.0]],
            },
        )

        monkeypatch.setattr(gemini_mod, "_make_gemini_client", lambda **_: object())
        monkeypatch.setattr(gemini_mod, "_call_with_retry", lambda _name, fn, **_kwargs: fn())
        monkeypatch.setattr(
            gemini_mod,
            "_embed_text_batch",
            lambda _client, *, model, texts, task_type, output_dimensionality: [[0.1, 0.2] for _ in texts],
        )

        embeddings = _generate_dense_embeddings(
            ["new text"],
            engine="gemini",
            model="gemini-embedding-2-preview",
            batch_size=1,
            output_dimensionality=2,
            cache_path=cache_path,
        )

        assert embeddings == [[0.1, 0.2]]
        cache_payload = load_json_safe(cache_path)
        assert cache_payload["text_digest"] == _texts_digest(["new text"])
