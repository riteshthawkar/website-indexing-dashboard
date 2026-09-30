"""Whole-site coverage is independent of edge simplification and hop limits."""

import hashlib
import json

import pytest

pytest.importorskip("ijson")

from lawa_rag.graphs import GraphIndex


@pytest.fixture
def website(tmp_path):
    # A long directed chain, returns, a same-level edge, parallel link and islands.
    nodes = [{"id": str(i), "node_type": "page", "label": f"Page {i}",
              "url": "https://example.org/" + (f"p/{i}" if i else "")} for i in range(10)]
    edges = [(str(i), str(i + 1)) for i in range(6)]
    edges += [("0", "7"), ("7", "1"), ("1", "0"), ("2", "2"), ("8", "9"), ("9", "8"), ("0", "1")]
    payload = {"nodes": [*nodes, {"id": "external", "node_type": "discovered_url"}],
               "edges": [{"id": f"e{i}", "source_id": a, "target_id": b, "edge_type": "LINKS_TO"}
                         for i, (a, b) in enumerate(edges)]}
    payload["edges"].append({"id": "external-link", "source_id": "0", "target_id": "external", "edge_type": "LINKS_TO"})
    payload["edges"].append({"id": "not-a-link", "source_id": "0", "target_id": "6", "edge_type": "RELATED_TO"})
    source = tmp_path / "page_link_graph.json"
    source.write_text(json.dumps(payload))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    yield GraphIndex.build(source, tmp_path / "cache")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before


def test_whole_site_retains_every_page_even_beyond_two_hops_and_in_islands(website):
    tree = website.website()
    assert tree == website.website()  # reproducible root, tree and ordering
    assert tree["root"]["id"] == "0"
    assert {n["id"] for n in tree["nodes"]} == {str(i) for i in range(10)}
    assert tree["overview"] == {
        "total_pages": 10, "total_page_links": 13, "excluded_nodes": 1,
        "excluded_relationships": 2, "roots": ["0", "8"],
        "outside_primary_root": 2, "max_depth": 6, "edge_limit_omitted": 0,
    }
    assert len(tree["edges"]) == 8 and tree["hidden_edges"] == 5
    assert tree["limits"]["hops"] is None and not tree["truncated"]
    depths = {n["id"]: n["depth"] for n in tree["nodes"]}
    targets = [e["target_id"] for e in tree["edges"]]
    assert len(targets) == len(set(targets))
    assert all(depths[e["target_id"]] == depths[e["source_id"]] + 1 for e in tree["edges"])


def test_forward_mode_keeps_cross_parent_forward_links_not_returns_or_siblings(website):
    tree, forward = website.website(), website.website(edge_mode="forward")
    assert tree["nodes"] == forward["nodes"]
    assert len(forward["edges"]) == 9  # includes the recorded parallel forward link
    assert forward["hidden_edges"] == 4
    depths = {n["id"]: n["depth"] for n in forward["nodes"]}
    assert all(depths[e["target_id"]] > depths[e["source_id"]] for e in forward["edges"])


def test_neighborhood_eight_hops_and_edge_modes_keep_node_coverage(website):
    views = [website.neighborhood("0", hops=8, direction="out", edge_mode=mode) for mode in ("all", "tree", "forward")]
    assert all(view["nodes"] == views[0]["nodes"] for view in views)
    assert {n["id"] for n in views[0]["nodes"]} == {"0", "1", "2", "3", "4", "5", "6", "7", "external"}
    assert len(views[1]["edges"]) == len(views[1]["nodes"]) - 1
    assert len(views[0]["edges"]) > len(views[2]["edges"]) >= len(views[1]["edges"])
    assert views[2]["hidden_edges"] > 0
    incoming = website.neighborhood("6", hops=8, direction="in", edge_mode="forward")
    depths = {n["id"]: n["depth"] for n in incoming["nodes"]}
    assert all(depths[e["source_id"]] > depths[e["target_id"]] for e in incoming["edges"])


def test_budget_overflow_is_explicit_never_a_silent_partial_site(website, monkeypatch):
    import lawa_rag.graphs.website as module

    monkeypatch.setattr(module, "MAX_PAGES", 9)
    with pytest.raises(ValueError, match="page overview limit"):
        website.website()
    monkeypatch.setattr(module, "MAX_PAGES", 10)
    monkeypatch.setattr(module, "MAX_PAGE_LINKS", 12)
    with pytest.raises(ValueError, match="link overview limit"):
        website.website()
    monkeypatch.setattr(module, "MAX_PAGE_LINKS", 13)
    monkeypatch.setattr(module, "MAX_DISPLAY_LINKS", 8)
    result = website.website(edge_mode="forward")
    assert len(result["nodes"]) == 10 and result["truncated"]
    assert result["overview"]["edge_limit_omitted"] == 1
    assert {e["id"] for e in result["edges"]} == {e["id"] for e in website.website()["edges"]}


def test_bridge_hyperlinks_supported_without_document_relations(tmp_path):
    payload = {"pages": [{"page_card_id": "a"}, {"page_card_id": "b"}],
               "documents": [{"document_revision_id": "doc"}],
               "edges": [{"source_id": "a", "target_id": "b", "edge_type": "PAGE_LINKS_TO_PAGE"},
                         {"source_id": "a", "target_id": "doc", "edge_type": "PAGE_REPRESENTS_DOCUMENT"}]}
    source = tmp_path / "page_graph_bridge.json"
    source.write_text(json.dumps(payload))
    result = GraphIndex.build(source, tmp_path / "cache").website()
    assert len(result["nodes"]) == 2 and len(result["edges"]) == 1
    assert result["overview"]["excluded_nodes"] == 1


def test_empty_or_non_page_graph_fails_with_actionable_message(tmp_path):
    source = tmp_path / "knowledge_graph.json"
    source.write_text('{"nodes":[{"id":"entity","node_type":"entity"}],"edges":[]}')
    index = GraphIndex.build(source, tmp_path / "cache")
    with pytest.raises(ValueError, match="no saved page nodes"):
        index.website()
    with pytest.raises(ValueError, match="tree or forward"):
        index.website(edge_mode="all")
