from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

import lawa_rag
from lawa_rag import available_pipeline_configs, load_pipeline_config
from lawa_rag.representation import (
    REPRESENTATION_V2_KIND,
    REPRESENTATION_V2_SCHEMA_VERSION,
    attach_page_links_to_documents,
    representation_v2_json_schema,
    stable_representation_id,
    validate_representation_v2,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_PUBLIC_API = {
    "ActiveRelease",
    "ArtifactCatalog",
    "ArtifactContract",
    "ArtifactRecord",
    "IndexingPipeline",
    "PipelinePlanStage",
    "PipelineRun",
    "PipelineStage",
    "PipelineValidationError",
    "PipelineValidationReport",
    "REPRESENTATION_V2_KIND",
    "REPRESENTATION_V2_SCHEMA_VERSION",
    "ReleaseContractError",
    "StageContext",
    "StageResult",
    "StageRun",
    "StageStatus",
    "__version__",
    "available_pipeline_configs",
    "load_active_release",
    "load_pipeline_config",
    "register_stage",
    "representation_v2_json_schema",
    "stable_representation_id",
    "validate_representation_v2",
}


def test_top_level_public_api_is_an_explicit_compatibility_contract() -> None:
    assert set(lawa_rag.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(lawa_rag, name) for name in EXPECTED_PUBLIC_API)


def test_importing_core_and_lazy_facades_does_not_load_provider_implementations() -> None:
    script = """
import sys
import lawa_rag
import lawa_rag.evaluation
import lawa_rag.retrieval
assert 'pipeline.retrieval.adaptive_hybrid' not in sys.modules
assert 'pipeline.evaluation.answer_generation' not in sys.modules
assert 'google.genai' not in sys.modules
assert 'pinecone' not in sys.modules
print('lazy imports preserved')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lazy imports preserved"


def test_retrieval_and_evaluation_facades_resolve_real_implementations() -> None:
    import lawa_rag.evaluation as evaluation
    import lawa_rag.retrieval as retrieval
    from pipeline.evaluation.dataset import EvalExample
    from pipeline.retrieval.adaptive_hybrid import QueryMode

    assert retrieval.QueryMode is QueryMode
    assert evaluation.EvalExample is EvalExample
    assert "RoutedHybridRetriever" in dir(retrieval)
    assert "evaluate_retrieval_dataset" in dir(evaluation)
    with pytest.raises(AttributeError):
        getattr(retrieval, "UnknownRetriever")
    with pytest.raises(AttributeError):
        getattr(evaluation, "unknown_evaluator")


def test_service_factory_forwards_every_runtime_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipeline.service.retrieval_api as implementation
    from lawa_rag.service import create_retrieval_service_app

    captured: dict[str, Any] = {}
    sentinel = object()

    def fake_factory(**kwargs: Any) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(implementation, "create_retrieval_service_app", fake_factory)
    result = create_retrieval_service_app(
        config_name="production",
        work_dir="/tmp/release",
        max_concurrency=7,
        request_timeout_seconds=12.5,
        queue_timeout_seconds=0.75,
    )

    assert result is sentinel
    assert captured == {
        "config_name": "production",
        "work_dir": "/tmp/release",
        "max_concurrency": 7,
        "request_timeout_seconds": 12.5,
        "queue_timeout_seconds": 0.75,
    }


def test_yaml_config_loading_inheritance_environment_and_explicit_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "base.yaml").write_text(
        "project_name: base\nfeature:\n  enabled: false\n  limit: 1\n",
        encoding="utf-8",
    )
    (config_dir / "child.yaml").write_text(
        "_inherit: base.yaml\nproject_name: child\nfeature:\n  label: inherited\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PIPELINE_FEATURE__ENABLED", "true")
    monkeypatch.setenv("PIPELINE_FEATURE__LIMIT", "5")

    loaded = load_pipeline_config(
        "child",
        search_dirs=[config_dir],
        overrides={"feature": {"limit": 9}},
    )

    assert loaded == {
        "project_name": "child",
        "feature": {"enabled": True, "limit": 9, "label": "inherited"},
    }
    assert available_pipeline_configs([config_dir]) == [
        {"file": str(config_dir / "base.yaml"), "name": "base", "project_name": "base"},
        {"file": str(config_dir / "child.yaml"), "name": "child", "project_name": "child"},
    ]


def test_representation_facade_executes_schema_id_link_and_validation_code() -> None:
    schema = representation_v2_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["properties"]["schema_version"]["const"] == REPRESENTATION_V2_SCHEMA_VERSION
    assert stable_representation_id("doc", "same") == stable_representation_id("doc", "same")
    assert stable_representation_id("doc", "same") != stable_representation_id("doc", "other")

    documents = [
        {
            "document_revision_id": "revision-1",
            "page_card_ids": [],
            "title": "0a1b2c3d4e5f6789",
            "source_url": "https://example.test/faculty",
        }
    ]
    attach_page_links_to_documents(
        documents,
        [
            {
                "document_revision_id": "revision-1",
                "page_card_id": "page-1",
                "title": "Faculty Directory",
            }
        ],
    )
    assert documents[0]["page_card_ids"] == ["page-1"]
    assert documents[0]["title"] == "Faculty Directory"

    report = validate_representation_v2(
        {
            "schema_version": "wrong-version",
            "kind": REPRESENTATION_V2_KIND,
            "documents": [],
            "page_cards": [],
            "actions": [],
        }
    )
    assert report["passed"] is False
    assert report["counts"]["issue_codes"] == {"schema_version": 1}
