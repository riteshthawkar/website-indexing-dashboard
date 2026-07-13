from __future__ import annotations

from pipeline.core.knowledge_graph import validate_graph_bundle
from pipeline.core.mbzuai_indexing import (
    canonicalize_link_graph,
    normalize_url,
    sha1_text,
)


def _node_id(url: str) -> str:
    return f"page:{sha1_text(normalize_url(url), 24)}"


def test_locale_variants_remain_distinct_page_nodes_and_edges() -> None:
    english_url = "https://mbzuai.ac.ae/en/study/admissions/"
    arabic_url = "https://mbzuai.ac.ae/ar/study/admissions/"
    english_target = "https://mbzuai.ac.ae/en/study/scholarships/"
    arabic_target = "https://mbzuai.ac.ae/ar/study/scholarships/"
    family_url = "https://mbzuai.ac.ae/study/admissions"

    graph = canonicalize_link_graph(
        {
            "edges": [
                {
                    "source_url": english_url,
                    "target_url": english_target,
                    "properties": {"link_type": "internal", "anchor_texts": ["Scholarships"]},
                },
                {
                    "source_url": arabic_url,
                    "target_url": arabic_target,
                    "properties": {"link_type": "internal", "anchor_texts": ["المنح الدراسية"]},
                },
            ]
        },
        {
            english_url: {
                "url": english_url,
                "canonical_url": english_url,
                "canonical_family_url": family_url,
                "title": "Admissions",
                "language": "en",
                "content_hash": "english-content",
            },
            arabic_url: {
                "url": arabic_url,
                "canonical_url": arabic_url,
                "canonical_family_url": family_url,
                "title": "القبول",
                "language": "ar",
                "content_hash": "arabic-content",
            },
        },
    )

    nodes_by_url = {node["url"]: node for node in graph["nodes"]}
    assert nodes_by_url[normalize_url(english_url)]["id"] == _node_id(english_url)
    assert nodes_by_url[normalize_url(arabic_url)]["id"] == _node_id(arabic_url)
    assert _node_id(english_url) != _node_id(arabic_url)
    assert nodes_by_url[normalize_url(english_url)]["canonical_family_url"] == family_url
    assert nodes_by_url[normalize_url(arabic_url)]["canonical_family_url"] == family_url
    assert {
        (edge["source_id"], edge["target_id"])
        for edge in graph["edges"]
    } == {
        (_node_id(english_url), _node_id(english_target)),
        (_node_id(arabic_url), _node_id(arabic_target)),
    }
    assert graph["schema_version"] == 3
    assert graph["node_identity"] == "normalized_url_v1"
    assert graph["stats"] == {
        "node_count": 4,
        "edge_count": 2,
        "link_type_counts": {"internal": 2},
    }
    assert validate_graph_bundle(graph) == []


def test_duplicate_normalized_edges_merge_evidence_and_stats_exactly() -> None:
    source_url = "https://mbzuai.ac.ae/news/agentic-ai/"
    tracked_target = "https://example.org/report/?utm_source=mbzuai"
    canonical_target = "https://example.org/report"
    edges = [
        {
            "source_url": source_url,
            "target_url": tracked_target,
            "properties": {
                "link_type": "external",
                "anchor_texts": ["Read report", "Details"],
                "rels": ["noopener"],
                "sources": ["html:a"],
            },
        },
        {
            "source_url": source_url,
            "target_url": canonical_target,
            "properties": {
                "link_type": "external",
                "anchor_texts": ["Details", "Full report"],
                "rels": ["nofollow"],
                "sources": ["crawl4ai:external"],
            },
        },
    ]
    metadata = {
        source_url: {
            "url": source_url,
            "canonical_url": source_url,
            "canonical_family_url": normalize_url(source_url),
            "title": "Agentic AI",
            "language": "en",
        }
    }

    forward = canonicalize_link_graph({"edges": edges}, metadata)
    reverse = canonicalize_link_graph({"edges": list(reversed(edges))}, metadata)

    assert forward == reverse
    assert len(forward["nodes"]) == len({node["id"] for node in forward["nodes"]}) == 2
    assert len(forward["edges"]) == 1
    edge = forward["edges"][0]
    assert edge["target_url"] == normalize_url(canonical_target)
    assert edge["properties"] == {
        "anchor_texts": ["Details", "Full report", "Read report"],
        "link_type": "external",
        "rels": ["nofollow", "noopener"],
        "sources": ["crawl4ai:external", "html:a"],
    }
    assert forward["stats"] == {
        "node_count": 2,
        "edge_count": 1,
        "link_type_counts": {"external": 1},
    }
    assert sum(forward["stats"]["link_type_counts"].values()) == len(forward["edges"])
    assert validate_graph_bundle(forward) == []
