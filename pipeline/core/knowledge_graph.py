"""
Deterministic knowledge graph helpers for pipeline retrieval artifacts.

The first production-safe graph slice models document structure and retrieval
provenance only. It does not attempt open-ended entity/relation extraction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List

from .io import atomic_write_json, load_json_safe, sha256_file


def _stable_graph_id(kind: str, *parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not raw:
        raw = kind
    return f"{kind}:{sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def _compact_property_value(value: Any) -> Any:
    if value in (None, "", [], {}):
        return None
    if isinstance(value, list):
        compacted = [_compact_property_value(item) for item in value]
        return [item for item in compacted if item not in (None, "", [], {})]
    if isinstance(value, dict):
        return {
            str(key): compact
            for key, raw in value.items()
            if (compact := _compact_property_value(raw)) not in (None, "", [], {})
        }
    return value


@dataclass
class GraphNode:
    id: str
    node_type: str
    label: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["properties"] = _compact_property_value(data.get("properties") or {}) or {}
        return {key: value for key, value in data.items() if value not in ("", [], {}, None)}


@dataclass
class GraphEdge:
    id: str
    edge_type: str
    source_id: str
    target_id: str
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["properties"] = _compact_property_value(data.get("properties") or {}) or {}
        return {key: value for key, value in data.items() if value not in ("", [], {}, None)}


def make_graph_node(*, node_id: str, node_type: str, label: str = "", properties: Dict[str, Any] | None = None) -> GraphNode:
    return GraphNode(
        id=str(node_id),
        node_type=str(node_type),
        label=str(label or ""),
        properties=dict(properties or {}),
    )


def make_graph_edge(
    *,
    edge_type: str,
    source_id: str,
    target_id: str,
    properties: Dict[str, Any] | None = None,
    qualifier: str = "",
) -> GraphEdge:
    return GraphEdge(
        id=_stable_graph_id("edge", edge_type, source_id, target_id, qualifier),
        edge_type=str(edge_type),
        source_id=str(source_id),
        target_id=str(target_id),
        properties=dict(properties or {}),
    )


def build_graph_bundle(
    *,
    nodes: Iterable[GraphNode],
    edges: Iterable[GraphEdge],
    schema_version: int = 1,
    graph_type: str = "deterministic_content_graph",
) -> Dict[str, Any]:
    node_list = [node.to_dict() for node in nodes]
    edge_list = [edge.to_dict() for edge in edges]
    node_type_counts: Dict[str, int] = {}
    edge_type_counts: Dict[str, int] = {}
    for node in node_list:
        node_type = str(node.get("node_type") or "")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    for edge in edge_list:
        edge_type = str(edge.get("edge_type") or "")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    return {
        "schema_version": int(schema_version),
        "graph_type": graph_type,
        "nodes": node_list,
        "edges": edge_list,
        "stats": {
            "node_count": len(node_list),
            "edge_count": len(edge_list),
            "node_type_counts": node_type_counts,
            "edge_type_counts": edge_type_counts,
        },
    }


def build_graph_index(
    graph_bundle: Dict[str, Any],
    *,
    source_graph_sha256: str = "",
) -> Dict[str, Any]:
    """Build the deterministic adjacency index for ``graph_bundle``.

    ``source_graph_sha256`` binds the derived index to the exact serialized
    graph file it indexes.  Callers writing a production artifact pair should
    use :func:`save_graph_bundle_with_index`, which writes the graph first and
    supplies its file digest automatically.
    """
    outgoing: Dict[str, List[str]] = {}
    incoming: Dict[str, List[str]] = {}
    node_type_by_id: Dict[str, str] = {}
    edge_type_by_id: Dict[str, str] = {}

    for node in graph_bundle.get("nodes") or []:
        node_id = str(node.get("id") or "")
        if not node_id:
            continue
        node_type_by_id[node_id] = str(node.get("node_type") or "")
        outgoing.setdefault(node_id, [])
        incoming.setdefault(node_id, [])

    for edge in graph_bundle.get("edges") or []:
        edge_id = str(edge.get("id") or "")
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        if not edge_id or not source_id or not target_id:
            continue
        edge_type_by_id[edge_id] = str(edge.get("edge_type") or "")
        outgoing.setdefault(source_id, []).append(edge_id)
        incoming.setdefault(target_id, []).append(edge_id)

    payload: Dict[str, Any] = {}
    if source_graph_sha256:
        # Keep the binding in the small JSON prefix so runtime validation can
        # verify it without materializing an ~86 MB adjacency index.
        payload["source_graph_sha256"] = str(source_graph_sha256).strip().lower()
    payload.update({
        "index_schema_version": 1,
        "schema_version": int(graph_bundle.get("schema_version") or 1),
        "graph_type": str(graph_bundle.get("graph_type") or "deterministic_content_graph"),
        "node_type_by_id": node_type_by_id,
        "edge_type_by_id": edge_type_by_id,
        "outgoing_edge_ids": outgoing,
        "incoming_edge_ids": incoming,
    })
    return payload


def save_graph_bundle(graph_bundle: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    atomic_write_json(path, graph_bundle)
    return path


def save_graph_bundle_with_index(
    graph_bundle: Dict[str, Any],
    graph_path: str | Path,
    index_path: str | Path,
) -> tuple[Path, Path]:
    """Atomically write a graph and a cryptographically bound derived index."""

    graph_file = save_graph_bundle(graph_bundle, graph_path)
    graph_sha256 = sha256_file(graph_file)
    index_file = save_graph_bundle(
        build_graph_index(graph_bundle, source_graph_sha256=graph_sha256),
        index_path,
    )
    return graph_file, index_file


def validate_graph_index_derivation(
    graph_path: str | Path,
    index_path: str | Path,
) -> List[Dict[str, Any]]:
    """Verify an index is a complete deterministic derivation of its graph.

    Independent file hashes only prove that two files have not changed.  This
    check additionally proves that the index names the exact graph digest and
    that every derived mapping equals a fresh deterministic rebuild.
    """

    graph_file = Path(graph_path)
    index_file = Path(index_path)
    issues: List[Dict[str, Any]] = []
    graph_bundle = load_json_safe(graph_file, None)
    graph_index = load_json_safe(index_file, None)
    if not isinstance(graph_bundle, dict):
        return [{
            "code": "invalid_graph_bundle",
            "message": f"Knowledge-graph bundle is missing or invalid: {graph_file}",
        }]
    if not isinstance(graph_index, dict):
        return [{
            "code": "invalid_graph_index",
            "message": f"Knowledge-graph index is missing or invalid: {index_file}",
        }]

    source_graph_sha256 = sha256_file(graph_file)
    recorded_source_sha256 = str(graph_index.get("source_graph_sha256") or "").strip().lower()
    if not recorded_source_sha256:
        issues.append({
            "code": "graph_index_missing_source_hash",
            "message": "Knowledge-graph index is missing source_graph_sha256",
        })
    elif recorded_source_sha256 != source_graph_sha256:
        issues.append({
            "code": "graph_index_source_hash_mismatch",
            "message": "Knowledge-graph index source_graph_sha256 does not match its graph file",
            "expected": source_graph_sha256,
            "actual": recorded_source_sha256,
        })

    expected_index = build_graph_index(
        graph_bundle,
        source_graph_sha256=source_graph_sha256,
    )
    if graph_index != expected_index:
        issues.append({
            "code": "graph_index_derivation_mismatch",
            "message": "Knowledge-graph index is not the deterministic derivation of its graph file",
        })
    return issues


def validate_graph_index_binding(
    graph_path: str | Path,
    index_path: str | Path,
) -> List[Dict[str, Any]]:
    """Cheaply verify the index names the exact graph file it accompanies.

    Producers place ``source_graph_sha256`` in the index JSON prefix. Runtime
    uses this streaming-friendly binding plus the independently recorded index
    file hash. Release construction/promotion additionally calls
    :func:`validate_graph_index_derivation` once for a full deterministic
    rebuild comparison.
    """

    graph_file = Path(graph_path)
    index_file = Path(index_path)
    try:
        with index_file.open("rb") as handle:
            prefix = handle.read(4096)
    except OSError:
        prefix = b""
    match = re.search(
        rb'"source_graph_sha256"\s*:\s*"([0-9a-fA-F]{64})"',
        prefix,
    )
    if match is None:
        return [{
            "code": "graph_index_missing_source_hash",
            "message": "Knowledge-graph index JSON prefix is missing source_graph_sha256",
        }]
    recorded_source_sha256 = match.group(1).decode("ascii").lower()
    actual_source_sha256 = sha256_file(graph_file)
    if recorded_source_sha256 != actual_source_sha256:
        return [{
            "code": "graph_index_source_hash_mismatch",
            "message": "Knowledge-graph index source_graph_sha256 does not match its graph file",
            "expected": actual_source_sha256,
            "actual": recorded_source_sha256,
        }]
    return []


def load_graph_bundle(path: str | Path) -> Dict[str, Any]:
    payload = load_json_safe(path, {}) or {}
    return payload if isinstance(payload, dict) else {}


def validate_graph_bundle(graph_bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues: List[Dict[str, Any]] = []
    nodes = graph_bundle.get("nodes")
    edges = graph_bundle.get("edges")

    if not isinstance(nodes, list):
        issues.append({"code": "invalid_graph_nodes", "message": "Graph bundle nodes payload is not a list"})
        nodes = []
    if not isinstance(edges, list):
        issues.append({"code": "invalid_graph_edges", "message": "Graph bundle edges payload is not a list"})
        edges = []

    node_ids = set()
    for node in nodes:
        if not isinstance(node, dict):
            issues.append({"code": "invalid_graph_node", "message": "Graph node is not an object"})
            continue
        node_id = str(node.get("id") or "")
        node_type = str(node.get("node_type") or "")
        if not node_id:
            issues.append({"code": "graph_node_missing_id", "message": "Graph node is missing id"})
            continue
        if not node_type:
            issues.append({"code": "graph_node_missing_type", "message": "Graph node is missing node_type", "node_id": node_id})
        if node_id in node_ids:
            issues.append({"code": "duplicate_graph_node_id", "message": "Graph bundle contains duplicate node ids", "node_id": node_id})
            continue
        node_ids.add(node_id)

    edge_ids = set()
    for edge in edges:
        if not isinstance(edge, dict):
            issues.append({"code": "invalid_graph_edge", "message": "Graph edge is not an object"})
            continue
        edge_id = str(edge.get("id") or "")
        edge_type = str(edge.get("edge_type") or "")
        source_id = str(edge.get("source_id") or "")
        target_id = str(edge.get("target_id") or "")
        if not edge_id:
            issues.append({"code": "graph_edge_missing_id", "message": "Graph edge is missing id"})
            continue
        if edge_id in edge_ids:
            issues.append({"code": "duplicate_graph_edge_id", "message": "Graph bundle contains duplicate edge ids", "edge_id": edge_id})
            continue
        edge_ids.add(edge_id)
        if not edge_type:
            issues.append({"code": "graph_edge_missing_type", "message": "Graph edge is missing edge_type", "edge_id": edge_id})
        if not source_id or not target_id:
            issues.append({"code": "graph_edge_missing_endpoint", "message": "Graph edge is missing source or target id", "edge_id": edge_id})
            continue
        if source_id not in node_ids:
            issues.append({"code": "graph_edge_missing_source_node", "message": "Graph edge references a missing source node", "edge_id": edge_id, "node_id": source_id})
        if target_id not in node_ids:
            issues.append({"code": "graph_edge_missing_target_node", "message": "Graph edge references a missing target node", "edge_id": edge_id, "node_id": target_id})

    return issues


def community_summary_quality(
    graph_bundle: Dict[str, Any],
    *,
    min_characters: int = 40,
) -> Dict[str, Any]:
    """Return deterministic completeness metrics for community summaries."""

    communities = [
        node
        for node in (graph_bundle.get("nodes") or [])
        if isinstance(node, dict) and str(node.get("node_type") or "") == "community"
    ]
    invalid_ids: List[str] = []
    for node in communities:
        properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
        summary = " ".join(str(properties.get("summary") or "").split()).strip()
        if (
            len(summary) < max(1, int(min_characters))
            or summary.casefold() in {
                "summary generation failed.",
                "no entity details available to summarize.",
            }
        ):
            invalid_ids.append(str(node.get("id") or "<missing>"))
    total = len(communities)
    valid = total - len(invalid_ids)
    return {
        "total_communities": total,
        "valid_summaries": valid,
        "invalid_summary_community_ids": invalid_ids,
        "coverage_ratio": (valid / total) if total else 0.0,
        "minimum_summary_characters": max(1, int(min_characters)),
    }
