import asyncio
from pathlib import Path

from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import load_json_safe
from pipeline.core.knowledge_graph import save_graph_bundle_with_index
from pipeline.stages.formatters import semantic_graph_community_formatter as mod


def _source_graph(tmp_path: Path) -> Path:
    nodes = [
        {
            "id": f"entity:{index}",
            "node_type": "entity",
            "label": f"Entity {index}",
            "properties": {"canonical_name": f"Entity {index}"},
        }
        for index in range(8)
    ]
    edges = []
    assertion_index = 0
    for subject, obj in (
        (0, 1),
        (1, 2),
        (0, 2),
        (3, 4),
        (4, 5),
        (3, 5),
        (2, 3),
        (6, 7),
    ):
        assertion_id = f"assertion:{assertion_index}"
        assertion_index += 1
        nodes.append(
            {
                "id": assertion_id,
                "node_type": "relation_assertion",
                "label": "related_to",
                "properties": {},
            }
        )
        edges.extend(
            [
                {
                    "id": f"{assertion_id}:subject",
                    "edge_type": "ASSERTION_SUBJECT",
                    "source_id": assertion_id,
                    "target_id": f"entity:{subject}",
                },
                {
                    "id": f"{assertion_id}:object",
                    "edge_type": "ASSERTION_OBJECT",
                    "source_id": assertion_id,
                    "target_id": f"entity:{obj}",
                },
            ]
        )
    graph_file = tmp_path / "source" / "promoted_knowledge_graph.json"
    index_file = tmp_path / "source" / "promoted_knowledge_graph_index.json"
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
    return graph_file


def _context(work_dir: Path, graph_file: Path) -> StageContext:
    return StageContext(
        run_id=work_dir.name,
        project_name="mbzuai",
        config={
            "graph": {
                "community_detection_seed": 17,
                "community_detection_iterations": 4,
            }
        },
        work_dir=work_dir,
        previous_outputs={"knowledge_graph_file": str(graph_file)},
        stage_definition={"type": "formatter", "plugin": "semantic_graph_community"},
        stage_id="community_graph",
    )


def _memberships(graph_file: str) -> dict[str, str]:
    graph = load_json_safe(graph_file, {})
    return {
        str(edge["source_id"]): str(edge["target_id"])
        for edge in graph.get("edges", [])
        if edge.get("edge_type") == "IN_COMMUNITY"
    }


def test_community_detection_passes_seed_and_reproduces_memberships(
    monkeypatch,
    tmp_path: Path,
) -> None:
    graph_file = _source_graph(tmp_path)
    calls = []
    original = mod.la.find_partition

    def _recording_find_partition(*args, **kwargs):
        calls.append(dict(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(mod.la, "find_partition", _recording_find_partition)
    first = asyncio.run(
        mod.SemanticGraphCommunityFormatter().execute(
            _context(tmp_path / "run-1", graph_file)
        )
    )
    second = asyncio.run(
        mod.SemanticGraphCommunityFormatter().execute(
            _context(tmp_path / "run-2", graph_file)
        )
    )

    assert first.status is StageStatus.COMPLETED
    assert second.status is StageStatus.COMPLETED
    assert calls == [
        {"n_iterations": 4, "seed": 17},
        {"n_iterations": 4, "seed": 17},
    ]
    assert _memberships(first.outputs["community_graph_file"]) == _memberships(
        second.outputs["community_graph_file"]
    )
    assert first.metrics["community_detection_seed"] == 17
    assert first.metrics["community_detection_iterations"] == 4


def test_community_detection_config_rejects_invalid_seed_and_iterations() -> None:
    errors = asyncio.run(
        mod.SemanticGraphCommunityFormatter().validate_config(
            {
                "graph": {
                    "community_detection_seed": -1,
                    "community_detection_iterations": 0,
                }
            }
        )
    )

    assert "graph.community_detection_seed must be between 0 and 2147483647" in errors
    assert "graph.community_detection_iterations must be between 1 and 100" in errors
