from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import load_json_safe
from pipeline.core.knowledge_graph import (
    save_graph_bundle_with_index,
    validate_graph_index_derivation,
)
from pipeline.stages.formatters.semantic_graph_summarize_formatter import (
    SemanticGraphSummarizeFormatter,
    _bounded_context_lines,
)


def _community_graph(
    tmp_path: Path,
    *,
    community_count: int = 2,
    members_per_community: int = 1,
) -> tuple[Path, Path]:
    nodes = []
    edges = []
    for index in range(community_count):
        community_id = f"community:{index}"
        nodes.append(
            {
                "id": community_id,
                "node_type": "community",
                "label": f"Community {index}",
                "properties": {
                    "community_id": index,
                    "size": members_per_community,
                    "level": 0,
                },
            }
        )
        entity_ids = []
        for member_index in range(members_per_community):
            entity_id = f"entity:{index}:{member_index}"
            entity_ids.append(entity_id)
            nodes.append(
                {
                    "id": entity_id,
                    "node_type": "entity",
                    "label": f"Entity {index}-{member_index}",
                    "properties": {
                        "canonical_name": f"Entity {index}-{member_index}",
                        "description": "A source-backed entity in the MBZUAI knowledge graph.",
                    },
                }
            )
            edges.append(
                {
                    "id": f"membership:{index}:{member_index}",
                    "edge_type": "IN_COMMUNITY",
                    "source_id": entity_id,
                    "target_id": community_id,
                }
            )
        if len(entity_ids) >= 2:
            assertion_id = f"assertion:{index}"
            nodes.append(
                {
                    "id": assertion_id,
                    "node_type": "relation_assertion",
                    "label": "related_to",
                    "properties": {
                        "text": (
                            f"Entity {index}-0 is related to Entity {index}-1 "
                            "in the indexed MBZUAI corpus."
                        )
                    },
                }
            )
            edges.extend(
                [
                    {
                        "id": f"subject:{index}",
                        "edge_type": "ASSERTION_SUBJECT",
                        "source_id": assertion_id,
                        "target_id": entity_ids[0],
                    },
                    {
                        "id": f"object:{index}",
                        "edge_type": "ASSERTION_OBJECT",
                        "source_id": assertion_id,
                        "target_id": entity_ids[1],
                    },
                ]
            )
    directory = tmp_path / "stage_outputs" / "community_graph"
    graph_file = directory / "community_knowledge_graph.json"
    index_file = directory / "community_knowledge_graph_index.json"
    save_graph_bundle_with_index(
        {
            "schema_version": 2,
            "graph_type": "promoted_semantic_graph",
            "nodes": nodes,
            "edges": edges,
        },
        graph_file,
        index_file,
    )
    return graph_file, index_file


def _context(
    tmp_path: Path,
    graph_file: Path,
    index_file: Path,
    *,
    graph_overrides: dict | None = None,
) -> StageContext:
    graph_config = {
        "community_summary_min_coverage_ratio": 1.0,
        "community_summary_min_characters": 40,
        "community_summary_max_provider_failures": 0,
        "community_summary_reuse_existing": True,
    }
    graph_config.update(graph_overrides or {})
    return StageContext(
        run_id=tmp_path.name,
        project_name="mbzuai_main",
        config={
            "pipeline": {"production_profile": True},
            "graph": graph_config,
        },
        work_dir=tmp_path,
        previous_outputs={
            "community_graph_file": str(graph_file),
            "knowledge_graph_index_file": str(index_file),
        },
        stage_definition={"type": "formatter", "plugin": "semantic_graph_summarize"},
        stage_id="summarize_community_graph",
    )


class _FakeModels:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls = 0
        self.configs = []

    def generate_content(self, **kwargs):
        self.calls += 1
        self.configs.append(kwargs.get("config"))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response)


def test_production_summary_stage_rebuilds_bound_index(monkeypatch, tmp_path: Path) -> None:
    graph_file, index_file = _community_graph(tmp_path)
    fake_client = SimpleNamespace(
        models=_FakeModels(
            [
                "This community covers source-backed university programs and related academic facts.",
                "This community describes MBZUAI research entities and their verified relationships.",
            ]
        )
    )
    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        lambda: fake_client,
    )

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(_context(tmp_path, graph_file, index_file))
    )

    assert result.status is StageStatus.COMPLETED
    assert result.metrics["valid_community_summaries"] == 2
    assert result.metrics["community_summary_coverage_ratio"] == 1.0
    output_graph = Path(result.outputs["knowledge_graph_file"])
    output_index = Path(result.outputs["knowledge_graph_index_file"])
    assert output_graph != graph_file
    assert output_index != index_file
    assert validate_graph_index_derivation(output_graph, output_index) == []
    quality = load_json_safe(result.outputs["community_summary_quality_file"], {})
    assert quality["passed"] is True
    assert quality["provider_failures"] == 0
    assert fake_client.models.configs
    assert all(
        config.thinking_config.thinking_budget == 0
        for config in fake_client.models.configs
    )


def test_production_summary_stage_fails_closed_without_placeholder(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph_file, index_file = _community_graph(tmp_path)
    fake_client = SimpleNamespace(
        models=_FakeModels(
            [
                "This community contains a complete source-backed summary for production retrieval.",
                RuntimeError("provider unavailable"),
            ]
        )
    )
    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        lambda: fake_client,
    )

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(_context(tmp_path, graph_file, index_file))
    )

    assert result.status is StageStatus.FAILED
    assert "quality gate failed" in str(result.error_message)
    output_graph = Path(result.outputs["knowledge_graph_file"])
    output_index = Path(result.outputs["knowledge_graph_index_file"])
    graph = load_json_safe(output_graph, {})
    summaries = [
        str((node.get("properties") or {}).get("summary") or "")
        for node in graph["nodes"]
        if node.get("node_type") == "community"
    ]
    assert "Summary generation failed." not in summaries
    assert sum(bool(summary) for summary in summaries) == 1
    assert validate_graph_index_derivation(output_graph, output_index) == []
    quality = load_json_safe(
        tmp_path
        / "stage_outputs"
        / "summarize_community_graph"
        / "community_summary_quality.json",
        {},
    )
    assert quality["passed"] is False
    assert quality["provider_failures"] == 1
    assert quality["coverage_ratio"] == 0.5


def test_small_communities_use_grounded_summaries_without_provider_calls(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph_file, index_file = _community_graph(
        tmp_path,
        community_count=3,
        members_per_community=2,
    )
    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        lambda: (_ for _ in ()).throw(
            AssertionError("small communities must not call Gemini")
        ),
    )

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(
            _context(
                tmp_path,
                graph_file,
                index_file,
                graph_overrides={
                    "community_summary_llm_min_size": 10,
                    "community_summary_max_provider_requests": 1,
                },
            )
        )
    )

    assert result.status is StageStatus.COMPLETED
    assert result.metrics["deterministic_summarized_communities"] == 3
    assert result.metrics["community_summary_provider_requests"] == 0
    graph = load_json_safe(result.outputs["knowledge_graph_file"], {})
    communities = [
        node for node in graph["nodes"] if node.get("node_type") == "community"
    ]
    assert all(
        (node.get("properties") or {}).get("summary_method")
        == "grounded_extractive"
        for node in communities
    )
    assert all(
        "source-backed relationships" in (node.get("properties") or {}).get("summary", "")
        for node in communities
    )


def test_provider_request_budget_fails_before_any_calls(monkeypatch, tmp_path: Path) -> None:
    graph_file, index_file = _community_graph(tmp_path, community_count=3)
    called = False

    def _unexpected_client():
        nonlocal called
        called = True
        raise AssertionError("provider must not initialize above the request budget")

    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        _unexpected_client,
    )

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(
            _context(
                tmp_path,
                graph_file,
                index_file,
                graph_overrides={
                    "community_summary_llm_min_size": 0,
                    "community_summary_max_provider_requests": 2,
                },
            )
        )
    )

    assert result.status is StageStatus.FAILED
    assert "budget exceeded before any calls" in str(result.error_message)
    assert called is False
    quality = load_json_safe(result.outputs["community_summary_quality_file"], {})
    assert quality["provider_requests_required"] == 3
    assert quality["provider_requests"] == 0


def test_provider_cache_resumes_only_missing_communities(monkeypatch, tmp_path: Path) -> None:
    graph_file, index_file = _community_graph(tmp_path)
    first_models = _FakeModels(
        [
            "This first source-backed community summary is complete and reusable.",
            RuntimeError("provider unavailable"),
        ]
    )
    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        lambda: SimpleNamespace(models=first_models),
    )
    context = _context(
        tmp_path,
        graph_file,
        index_file,
        graph_overrides={"community_summary_concurrency": 1},
    )

    first_result = asyncio.run(SemanticGraphSummarizeFormatter().execute(context))

    assert first_result.status is StageStatus.FAILED
    assert first_models.calls == 2
    second_models = _FakeModels(
        ["This second source-backed community summary completes the resumed stage."]
    )
    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        lambda: SimpleNamespace(models=second_models),
    )

    second_result = asyncio.run(SemanticGraphSummarizeFormatter().execute(context))

    assert second_result.status is StageStatus.COMPLETED
    assert second_models.calls == 1
    assert second_result.metrics["cached_provider_community_summaries"] == 1
    assert second_result.metrics["community_summary_provider_requests"] == 1


def test_retry_requests_never_exceed_global_provider_budget(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph_file, index_file = _community_graph(tmp_path, community_count=3)
    fake_models = _FakeModels(
        [RuntimeError("provider unavailable") for _ in range(4)]
    )
    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        lambda: SimpleNamespace(models=fake_models),
    )

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(
            _context(
                tmp_path,
                graph_file,
                index_file,
                graph_overrides={
                    "community_summary_concurrency": 1,
                    "community_summary_retry_attempts": 4,
                    "community_summary_retry_base_delay_sec": 0,
                    "community_summary_max_provider_requests": 4,
                },
            )
        )
    )

    assert result.status is StageStatus.FAILED
    assert fake_models.calls == 4
    quality = load_json_safe(result.outputs["community_summary_quality_file"], {})
    assert quality["provider_requests"] == 4
    assert quality["provider_communities_submitted"] == 3
    assert quality["provider_retries"] == 1


def test_provider_context_budget_retains_entities_and_facts() -> None:
    lines = _bounded_context_lines(
        [f"Entity: entity-{index} description" for index in range(100)],
        [f"Fact: fact-{index} evidence" for index in range(100)],
        maximum_characters=1000,
    )

    assert any(line.startswith("Entity: ") for line in lines)
    assert any(line.startswith("Fact: ") for line in lines)
    assert len("\n".join(lines)) <= 1000


def test_membership_size_mismatch_fails_closed_without_provider(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph_file, index_file = _community_graph(
        tmp_path,
        community_count=1,
        members_per_community=2,
    )
    graph = load_json_safe(graph_file, {})
    community = next(
        node for node in graph["nodes"] if node.get("node_type") == "community"
    )
    community["properties"]["size"] = 3
    save_graph_bundle_with_index(graph, graph_file, index_file)
    monkeypatch.setattr(
        "pipeline.stages.formatters.semantic_graph_summarize_formatter._make_gemini_client",
        lambda: (_ for _ in ()).throw(
            AssertionError("invalid communities must not call Gemini")
        ),
    )

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(
            _context(
                tmp_path,
                graph_file,
                index_file,
                graph_overrides={"community_summary_llm_min_size": 10},
            )
        )
    )

    assert result.status is StageStatus.FAILED
    quality = load_json_safe(result.outputs["community_summary_quality_file"], {})
    assert quality["coverage_ratio"] == 0.0
    assert quality["failures"] == [
        {
            "community_id": "community:0",
            "reason": "community_membership_size_mismatch",
        }
    ]


def test_production_summary_stage_rejects_zero_community_graph(tmp_path: Path) -> None:
    graph_file, index_file = _community_graph(tmp_path, community_count=0)

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(_context(tmp_path, graph_file, index_file))
    )

    assert result.status is StageStatus.FAILED
    assert "no communities" in str(result.error_message)
