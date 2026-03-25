"""
Deterministic knowledge graph helpers for pipeline retrieval artifacts.

The first production-safe graph slice models document structure and retrieval
provenance only. It does not attempt open-ended entity/relation extraction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import atomic_write_json, load_json_safe


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


def build_graph_index(graph_bundle: Dict[str, Any]) -> Dict[str, Any]:
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

    return {
        "schema_version": int(graph_bundle.get("schema_version") or 1),
        "graph_type": str(graph_bundle.get("graph_type") or "deterministic_content_graph"),
        "node_type_by_id": node_type_by_id,
        "edge_type_by_id": edge_type_by_id,
        "outgoing_edge_ids": outgoing,
        "incoming_edge_ids": incoming,
    }


def save_graph_bundle(graph_bundle: Dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    atomic_write_json(path, graph_bundle)
    return path


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
