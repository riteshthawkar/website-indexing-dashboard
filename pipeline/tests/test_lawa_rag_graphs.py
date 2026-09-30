"""Production contracts for the optional, offline graph explorer package API."""

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("ijson")

from lawa_rag.graphs import GraphIndex, discover_graphs, resolve_graph


def write_graph(tmp_path, payload, name="page_link_graph.json"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def sample():
    return {
        "nodes": [
            {
                "id": "a",
                "node_type": "page",
                "label": "Admissions 100%",
                "url": "https://example.org/admissions",
                "properties": {"text": "A" * 7000},
            },
            {"id": "b", "node_type": "entity", "label": "القبول"},
            {"id": "c", "node_type": "chunk", "label": "Entry_requirements"},
            {"id": "isolated", "node_type": "discovered_url", "label": "Not fetched"},
        ],
        "edges": [
            {
                "id": "ab",
                "edge_type": "LINKS_TO",
                "source_id": "a",
                "target_id": "b",
                "properties": {"anchor_texts": ["Apply"]},
            },
            {"id": "bc", "edge_type": "SUPPORTS", "source_id": "b", "target_id": "c"},
            {"id": "ca", "edge_type": "LINKS_TO", "source_id": "c", "target_id": "a"},
        ],
    }


def test_streamed_build_preserves_sources_and_multilingual_search(tmp_path, monkeypatch):
    source = write_graph(tmp_path, sample())
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    # Full-document text reads are forbidden: real artifacts can be ~500 MB.
    monkeypatch.setattr(Path, "read_text", lambda *_args, **_kwargs: pytest.fail("Full-file read"))
    index = GraphIndex.build(source, tmp_path / "cache")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    assert index.summary()["node_count"] == 4
    assert index.search("القبول")["items"][0]["id"] == "b"
    assert index.search("ADMISSIONS")["items"][0]["id"] == "a"
    assert len(index.search("%")["items"]) == 1
    assert len(index.search("_")["items"]) == 1
    assert index.search("", "page")["items"][0]["id"] == "a"
    assert index.search("", limit=1)["has_more"]
    assert index.node("a")["preview_truncated"] is True
    assert len(index.node("a")["properties"]["text"]) == 6000
    assert index.node("a")["incoming"] == 1


def test_direction_hops_filters_and_cycles(tmp_path):
    index = GraphIndex.build(write_graph(tmp_path, sample()), tmp_path / "cache")
    outgoing = index.neighborhood("a", direction="out")
    assert [edge["id"] for edge in outgoing["edges"]] == ["ab"]
    assert [edge["id"] for edge in index.neighborhood("a", direction="in")["edges"]] == ["ca"]
    two = index.neighborhood("a", direction="out", hops=2)
    assert {n["id"]: n["depth"] for n in two["nodes"]} == {"a": 0, "b": 1, "c": 2}
    assert {e["id"] for e in two["edges"]} == {"ab", "bc"}
    assert len(index.neighborhood("a", hops=2)["edges"]) == 3
    assert len(index.neighborhood("a", edge_type="SUPPORTS")["nodes"]) == 1
    assert not index.neighborhood("isolated")["edges"]
    with pytest.raises(KeyError):
        index.node("unknown")


def test_hubs_are_bounded_and_truncation_is_explicit(tmp_path):
    graph = sample()
    for number in range(2100):
        graph["nodes"].append({"id": f"hub{number}", "label": "Hub"})
        graph["edges"].append(
            {"source_id": "a", "target_id": f"hub{number}", "edge_type": "LINKS_TO"}
        )
    index = GraphIndex.build(write_graph(tmp_path, graph), tmp_path / "cache")
    result = index.neighborhood("a", max_nodes=8, max_edges=5)
    assert result["truncated"]
    assert len(result["nodes"]) <= 8 and len(result["edges"]) <= 5
    ids = {node["id"] for node in result["nodes"]}
    assert all(edge["source_id"] in ids and edge["target_id"] in ids for edge in result["edges"])
    with pytest.raises(ValueError):
        index.neighborhood("a", hops=9)


def test_bridge_collections_and_dangling_diagnostics(tmp_path):
    graph = {
        "documents": [{"document_revision_id": "doc", "title": "Handbook"}],
        "pages": [{"page_card_id": "page", "title": "Apply", "source_url": "https://example.org"}],
        "sections": [{"section_id": "section", "heading": "Requirements"}],
        "chunks": [{"chunk_id": "chunk"}],
        "actions": [{"action_id": "apply"}],
        "edges": [
            {
                "edge_id": "e",
                "source_id": "page",
                "target_id": "doc",
                "edge_type": "PAGE_REPRESENTS_DOCUMENT",
            },
            {"id": "missing", "source_id": "page", "target_id": "absent"},
        ],
    }
    index = GraphIndex.build(
        write_graph(tmp_path, graph, "page_graph_bridge.json"), tmp_path / "cache"
    )
    assert index.summary()["node_count"] == 5
    assert index.summary()["dangling_edges_omitted"] == 1
    assert index.summary()["edge_count"] == 1
    assert index.node("section")["label"] == "Requirements"


@pytest.mark.parametrize(
    "payload", [{"nodes": {}}, {"unrelated": []}, {"nodes": [{"id": "a"}, {"id": "a"}]}]
)
def test_invalid_artifact_never_publishes_partial_cache(tmp_path, payload):
    with pytest.raises(ValueError):
        GraphIndex.build(write_graph(tmp_path, payload), tmp_path / "cache")
    assert not list((tmp_path / "cache").iterdir())


def test_cache_reuse_and_changed_source(tmp_path):
    source = write_graph(tmp_path, sample())
    index = GraphIndex.build(source, tmp_path / "cache")
    assert GraphIndex.build(source, tmp_path / "cache").path == index.path
    graph = sample()
    graph["nodes"][0]["label"] = "Updated admissions"
    write_graph(tmp_path, graph)
    changed = GraphIndex.build(source, tmp_path / "cache")
    assert changed.path != index.path
    assert changed.node("a")["label"] == "Updated admissions"


def test_catalog_limits_paths_and_does_not_follow_external_symlinks(tmp_path):
    run = tmp_path / "run"
    source = write_graph(run, sample())
    write_graph(run, sample(), "stage_outputs/promote_graph/promoted_knowledge_graph.json")
    outside = write_graph(tmp_path, sample(), "outside.json")
    (run / "knowledge_graph.json").symlink_to(outside)
    catalog = discover_graphs(run)
    assert len(catalog) == 2
    assert {item["kind"] for item in catalog} == {"website", "knowledge"}
    assert resolve_graph(run, catalog[0]["id"]) == source
    for malicious in ("../outside.json", str(outside), "page_link_graph.json"):
        with pytest.raises(FileNotFoundError):
            resolve_graph(run, malicious)


def test_source_changed_mid_build_is_not_published(tmp_path, monkeypatch):
    import lawa_rag.graphs.index as module

    source = write_graph(tmp_path, sample())
    original = module.stream_records

    def changing(path):
        yield from original(path)
        source.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(module, "stream_records", changing)
    with pytest.raises(ValueError, match="changed"):
        GraphIndex.build(source, tmp_path / "cache")
    assert not list((tmp_path / "cache").iterdir())
