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
)


def _community_graph(tmp_path: Path, *, community_count: int = 2) -> tuple[Path, Path]:
    nodes = []
    edges = []
    for index in range(community_count):
        community_id = f"community:{index}"
        entity_id = f"entity:{index}"
        nodes.extend(
            [
                {
                    "id": entity_id,
                    "node_type": "entity",
                    "label": f"Entity {index}",
                    "properties": {
                        "canonical_name": f"Entity {index}",
                        "description": "A source-backed entity in the MBZUAI knowledge graph.",
                    },
                },
                {
                    "id": community_id,
                    "node_type": "community",
                    "label": f"Community {index}",
                    "properties": {"community_id": index, "size": 1, "level": 0},
                },
            ]
        )
        edges.append(
            {
                "id": f"membership:{index}",
                "edge_type": "IN_COMMUNITY",
                "source_id": entity_id,
                "target_id": community_id,
            }
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


def _context(tmp_path: Path, graph_file: Path, index_file: Path) -> StageContext:
    return StageContext(
        run_id=tmp_path.name,
        project_name="mbzuai_main",
        config={
            "pipeline": {"production_profile": True},
            "graph": {
                "community_summary_min_coverage_ratio": 1.0,
                "community_summary_min_characters": 40,
                "community_summary_max_provider_failures": 0,
                "community_summary_reuse_existing": True,
            },
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

    def generate_content(self, **_kwargs):
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
    assert validate_graph_index_derivation(graph_file, index_file) == []
    quality = load_json_safe(result.outputs["community_summary_quality_file"], {})
    assert quality["passed"] is True
    assert quality["provider_failures"] == 0


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
    graph = load_json_safe(graph_file, {})
    summaries = [
        str((node.get("properties") or {}).get("summary") or "")
        for node in graph["nodes"]
        if node.get("node_type") == "community"
    ]
    assert "Summary generation failed." not in summaries
    assert sum(bool(summary) for summary in summaries) == 1
    assert validate_graph_index_derivation(graph_file, index_file) == []
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


def test_production_summary_stage_rejects_zero_community_graph(tmp_path: Path) -> None:
    graph_file, index_file = _community_graph(tmp_path, community_count=0)

    result = asyncio.run(
        SemanticGraphSummarizeFormatter().execute(_context(tmp_path, graph_file, index_file))
    )

    assert result.status is StageStatus.FAILED
    assert "no communities" in str(result.error_message)
