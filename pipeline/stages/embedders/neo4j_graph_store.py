"""
Neo4j graph-store sync stage for promoted knowledge graph bundles.

This keeps the vector index as the primary retrieval path while optionally
loading the promoted graph into Neo4j for graph-aware workflows.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import urlparse, urlunparse

import requests

from pipeline.core.base import EmbedderStage, StageContext, StageResult
from pipeline.core.graph_artifacts import resolve_canonical_graph_artifacts
from pipeline.core.io import atomic_write_json, load_json_safe, sha256_file
from pipeline.core.knowledge_graph import load_graph_bundle, validate_graph_bundle
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

DEFAULT_NEO4J_NODE_TYPES = (
    "chunk",
    "page",
    "section",
    "fact",
    "evidence_span",
    "media",
    "entity",
    "relation_assertion",
    "community",
    "community_summary",
)

DEFAULT_NEO4J_EDGE_TYPES = (
    "CHUNK_HAS_FACT",
    "PAGE_HAS_FACT",
    "SECTION_HAS_FACT",
    "DOCUMENT_HAS_EVIDENCE_SPAN",
    "CHUNK_HAS_EVIDENCE_SPAN",
    "PAGE_HAS_EVIDENCE_SPAN",
    "SECTION_HAS_EVIDENCE_SPAN",
    "CHUNK_HAS_MEDIA",
    "PAGE_HAS_MEDIA",
    "SECTION_HAS_MEDIA",
    "FACT_MENTIONS_ENTITY",
    "CHUNK_MENTIONS_ENTITY",
    "ASSERTION_SUBJECT",
    "ASSERTION_OBJECT",
    "FACT_SUPPORTS_ASSERTION",
    "CHUNK_SUPPORTS_ASSERTION",
    "ASSERTION_SUPPORTED_BY_SPAN",
    "ENTITY_MENTIONED_IN_SPAN",
    "IN_COMMUNITY",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _env_or_config(config: Dict[str, Any], key: str, env_name: str) -> str:
    value = str(config.get(key) or "").strip()
    if value:
        return value
    return str(os.environ.get(env_name) or "").strip()


def _coerce_http_base(uri: str) -> str:
    raw = str(uri or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc
    path = parsed.path or ""
    if not netloc and parsed.path:
        netloc = parsed.path
        path = ""

    if scheme in {"http", "https"}:
        return raw.rstrip("/")

    host = netloc
    if scheme in {"neo4j+s", "neo4j+ssc", "bolt+s", "bolt+ssc"}:
        if host.endswith(":7687"):
            host = host[:-5]
        return urlunparse(("https", host, path.rstrip("/"), "", "", "")).rstrip("/")
    if scheme in {"neo4j", "bolt"}:
        if host.endswith(":7687"):
            host = host[:-5] + ":7474"
        elif ":" not in host:
            host = f"{host}:7474"
        return urlunparse(("http", host, path.rstrip("/"), "", "", "")).rstrip("/")
    raise ValueError("Neo4j URI must use http(s), neo4j, or bolt scheme")


def _neo4j_endpoint(base_uri: str, database: str) -> str:
    return f"{_coerce_http_base(base_uri)}/db/{database}/query/v2"


def _neo4j_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        output: List[str] = []
        seen = set()
        for item in value:
            scalar = _neo4j_scalar(item)
            if scalar in (None, ""):
                continue
            text = str(scalar)
            if text in seen:
                continue
            seen.add(text)
            output.append(text)
        return output or None
    return str(value)


def _flatten_props(props: Dict[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in (props or {}).items():
        clean_key = str(key or "").strip()
        if not clean_key:
            continue
        scalar = _neo4j_scalar(value)
        if scalar in (None, "", [], {}):
            continue
        output[clean_key] = scalar
    return output


def _post_query(
    *,
    endpoint: str,
    username: str,
    password: str,
    statement: str,
    parameters: Dict[str, Any] | None,
    timeout_sec: int,
) -> Dict[str, Any]:
    response = requests.post(
        endpoint,
        auth=(username, password),
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json={
            "statement": statement,
            "parameters": parameters or {},
        },
        timeout=timeout_sec,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"Neo4j Query API request failed with status {response.status_code}: "
            f"{response.text[:500]}"
        )
    if not response.content:
        return {}
    payload = response.json()
    if isinstance(payload, dict) and payload.get("errors"):
        raise RuntimeError(f"Neo4j Query API returned errors: {payload['errors']}")
    return payload if isinstance(payload, dict) else {}


def _first_query_scalar(payload: Dict[str, Any]) -> Any:
    data = payload.get("data") if isinstance(payload, dict) else None
    if isinstance(data, dict):
        values = data.get("values")
        if isinstance(values, list) and values and isinstance(values[0], list) and values[0]:
            return values[0][0]
        records = data.get("records")
        if isinstance(records, list) and records:
            first = records[0]
            if isinstance(first, list) and first:
                return first[0]
            if isinstance(first, dict):
                for value in first.values():
                    return value

    records = payload.get("records") if isinstance(payload, dict) else None
    if isinstance(records, list) and records:
        first = records[0]
        if isinstance(first, list) and first:
            return first[0]
        if isinstance(first, dict):
            for value in first.values():
                return value

    results = payload.get("results") if isinstance(payload, dict) else None
    if isinstance(results, list) and results:
        first_result = results[0]
        if isinstance(first_result, dict):
            data_rows = first_result.get("data")
            if isinstance(data_rows, list) and data_rows:
                row = data_rows[0]
                if isinstance(row, dict):
                    row_values = row.get("row")
                    if isinstance(row_values, list) and row_values:
                        return row_values[0]
    return None


def _query_count(payload: Dict[str, Any]) -> int:
    value = _first_query_scalar(payload)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _node_row(node: Dict[str, Any], *, namespace: str, graph_type: str) -> Dict[str, Any]:
    node_id = str(node.get("id") or "")
    return {
        "key": f"{namespace}:{node_id}",
        "namespace": namespace,
        "id": node_id,
        "node_type": str(node.get("node_type") or ""),
        "label": str(node.get("label") or ""),
        "graph_type": graph_type,
        "props": _flatten_props(node.get("properties") or {}),
    }


def _edge_row(edge: Dict[str, Any], *, namespace: str, graph_type: str) -> Dict[str, Any]:
    edge_id = str(edge.get("id") or "")
    source_id = str(edge.get("source_id") or "")
    target_id = str(edge.get("target_id") or "")
    return {
        "key": f"{namespace}:{edge_id}",
        "namespace": namespace,
        "id": edge_id,
        "source_key": f"{namespace}:{source_id}",
        "target_key": f"{namespace}:{target_id}",
        "source_id": source_id,
        "target_id": target_id,
        "edge_type": str(edge.get("edge_type") or ""),
        "graph_type": graph_type,
        "props": _flatten_props(edge.get("properties") or {}),
    }


def _filter_graph_bundle_for_neo4j(
    graph_bundle: Dict[str, Any],
    *,
    compact_upload: bool,
    allowed_node_types: Iterable[str],
    allowed_edge_types: Iterable[str],
) -> Dict[str, Any]:
    if not compact_upload:
        return graph_bundle

    allowed_node_types = {str(value).strip() for value in allowed_node_types if str(value).strip()}
    allowed_edge_types = {str(value).strip() for value in allowed_edge_types if str(value).strip()}

    raw_nodes = [node for node in (graph_bundle.get("nodes") or []) if isinstance(node, dict)]
    raw_edges = [edge for edge in (graph_bundle.get("edges") or []) if isinstance(edge, dict)]

    filtered_nodes = [
        node
        for node in raw_nodes
        if str(node.get("node_type") or "") in allowed_node_types
    ]
    node_ids = {str(node.get("id") or "") for node in filtered_nodes if str(node.get("id") or "")}

    filtered_edges = [
        edge
        for edge in raw_edges
        if str(edge.get("edge_type") or "") in allowed_edge_types
        and str(edge.get("source_id") or "") in node_ids
        and str(edge.get("target_id") or "") in node_ids
    ]

    return {
        "schema_version": graph_bundle.get("schema_version"),
        "graph_type": graph_bundle.get("graph_type"),
        "nodes": filtered_nodes,
        "edges": filtered_edges,
    }


@register_stage
class Neo4jGraphStoreEmbedder(EmbedderStage):
    name = "neo4j_graph_store"
    description = "Loads the promoted knowledge graph bundle into Neo4j using the Query API."

    async def execute(self, ctx: StageContext) -> StageResult:
        canonical_graph = resolve_canonical_graph_artifacts(
            ctx.work_dir,
            required=False,
            require_index=True,
            validate_binding=bool(
                (ctx.config.get("pipeline") or {}).get("production_profile", False)
            ),
        )
        graph_file = str(canonical_graph.graph_file) if canonical_graph is not None else ""
        if not graph_file:
            graph_file = ctx.previous_outputs.get("knowledge_graph_file")
        if not graph_file:
            graph_artifacts = ctx.find_artifacts(artifact_type="knowledge_graph_bundle")
            if graph_artifacts and graph_artifacts[-1].local_path:
                graph_file = graph_artifacts[-1].local_path
        if not graph_file:
            return StageResult.failure("No knowledge graph bundle available for Neo4j sync")

        graph_bundle = load_graph_bundle(graph_file)
        issues = validate_graph_bundle(graph_bundle)
        if issues:
            return StageResult.failure(f"Knowledge graph bundle is invalid: {issues[0].get('message')}")

        graph_cfg = dict(ctx.graph_config or {})
        compact_upload = bool(graph_cfg.get("neo4j_compact_upload", True))
        allowed_node_types = list(graph_cfg.get("neo4j_allowed_node_types") or DEFAULT_NEO4J_NODE_TYPES)
        allowed_edge_types = list(graph_cfg.get("neo4j_allowed_edge_types") or DEFAULT_NEO4J_EDGE_TYPES)
        neo4j_uri = _env_or_config(graph_cfg, "neo4j_uri", "NEO4J_URI")
        neo4j_database = _env_or_config(graph_cfg, "neo4j_database", "NEO4J_DATABASE") or "neo4j"
        neo4j_username = _env_or_config(graph_cfg, "neo4j_username", "NEO4J_USERNAME")
        neo4j_password = _env_or_config(graph_cfg, "neo4j_password", "NEO4J_PASSWORD")
        neo4j_namespace = _env_or_config(graph_cfg, "neo4j_namespace", "NEO4J_NAMESPACE") or f"{ctx.project_name}:{ctx.run_id}"
        timeout_sec = int(graph_cfg.get("neo4j_http_timeout_sec") or 60)
        batch_size = max(1, int(graph_cfg.get("neo4j_load_batch_size") or 250))
        verify_after_upload = bool(graph_cfg.get("neo4j_verify_after_upload", False))
        clear_on_zero = bool(graph_cfg.get("neo4j_clear_namespace_on_zero_progress", True))
        clear_all_kg = bool(graph_cfg.get("neo4j_clear_all_kg_nodes_on_zero_progress", False))
        clear_namespaces = [
            str(value).strip()
            for value in (graph_cfg.get("neo4j_clear_namespaces_on_zero_progress") or [])
            if str(value).strip()
        ]
        if not neo4j_uri or not neo4j_username or not neo4j_password:
            return StageResult.failure(
                "Neo4j sync requires graph.neo4j_uri, graph.neo4j_username, and graph.neo4j_password "
                "(or NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD)"
            )

        upload_bundle = _filter_graph_bundle_for_neo4j(
            graph_bundle,
            compact_upload=compact_upload,
            allowed_node_types=allowed_node_types,
            allowed_edge_types=allowed_edge_types,
        )
        issues = validate_graph_bundle(upload_bundle)
        if issues:
            return StageResult.failure(f"Filtered Neo4j graph bundle is invalid: {issues[0].get('message')}")

        endpoint = _neo4j_endpoint(neo4j_uri, neo4j_database)
        graph_type = str(upload_bundle.get("graph_type") or "knowledge_graph")
        progress_file = ctx.stage_work_dir / "neo4j_upload_progress.json"
        manifest_file = ctx.stage_work_dir / "neo4j_upload_manifest.json"
        progress = load_json_safe(progress_file, {}) or {}
        if not isinstance(progress, dict):
            progress = {}

        node_offset = max(0, int(progress.get("node_offset") or 0))
        edge_offset = max(0, int(progress.get("edge_offset") or 0))
        nodes = list(upload_bundle.get("nodes") or [])
        edges = list(upload_bundle.get("edges") or [])

        def run_query(statement: str, parameters: Dict[str, Any] | None = None) -> Dict[str, Any]:
            return _post_query(
                endpoint=endpoint,
                username=neo4j_username,
                password=neo4j_password,
                statement=statement,
                parameters=parameters,
                timeout_sec=timeout_sec,
            )

        run_query("CREATE CONSTRAINT kg_node_key IF NOT EXISTS FOR (n:KGNode) REQUIRE n.key IS UNIQUE")
        run_query("CREATE INDEX kg_node_namespace IF NOT EXISTS FOR (n:KGNode) ON (n.namespace)")
        run_query("CREATE INDEX kg_edge_namespace IF NOT EXISTS FOR ()-[r:KG_EDGE]-() ON (r.namespace)")

        if clear_on_zero and node_offset == 0 and edge_offset == 0:
            if clear_all_kg:
                run_query("MATCH (n:KGNode) DETACH DELETE n")
            else:
                if clear_namespaces:
                    run_query(
                        "MATCH (n:KGNode) WHERE n.namespace IN $namespaces DETACH DELETE n",
                        {"namespaces": clear_namespaces},
                    )
                run_query(
                    "MATCH (n:KGNode {namespace: $namespace}) DETACH DELETE n",
                    {"namespace": neo4j_namespace},
                )

        node_statement = (
            "UNWIND $rows AS row "
            "MERGE (n:KGNode {key: row.key}) "
            "SET n.namespace = row.namespace, "
            "    n.id = row.id, "
            "    n.node_type = row.node_type, "
            "    n.label = row.label, "
            "    n.graph_type = row.graph_type, "
            "    n.updated_at = $updated_at "
            "SET n += row.props"
        )
        edge_statement = (
            "UNWIND $rows AS row "
            "MATCH (source:KGNode {key: row.source_key}) "
            "MATCH (target:KGNode {key: row.target_key}) "
            "MERGE (source)-[r:KG_EDGE {key: row.key}]->(target) "
            "SET r.namespace = row.namespace, "
            "    r.id = row.id, "
            "    r.source_id = row.source_id, "
            "    r.target_id = row.target_id, "
            "    r.edge_type = row.edge_type, "
            "    r.graph_type = row.graph_type, "
            "    r.updated_at = $updated_at "
            "SET r += row.props"
        )

        updated_at = _now_iso()
        for start in range(node_offset, len(nodes), batch_size):
            batch = [_node_row(node, namespace=neo4j_namespace, graph_type=graph_type) for node in nodes[start : start + batch_size]]
            run_query(node_statement, {"rows": batch, "updated_at": updated_at})
            node_offset = start + len(batch)
            atomic_write_json(
                progress_file,
                {
                    "namespace": neo4j_namespace,
                    "graph_type": graph_type,
                    "node_offset": node_offset,
                    "edge_offset": edge_offset,
                    "node_total": len(nodes),
                    "edge_total": len(edges),
                    "updated_at": updated_at,
                },
            )

        for start in range(edge_offset, len(edges), batch_size):
            batch = [_edge_row(edge, namespace=neo4j_namespace, graph_type=graph_type) for edge in edges[start : start + batch_size]]
            run_query(edge_statement, {"rows": batch, "updated_at": updated_at})
            edge_offset = start + len(batch)
            atomic_write_json(
                progress_file,
                {
                    "namespace": neo4j_namespace,
                    "graph_type": graph_type,
                    "node_offset": node_offset,
                    "edge_offset": edge_offset,
                    "node_total": len(nodes),
                    "edge_total": len(edges),
                    "updated_at": updated_at,
                },
            )

        verification: Dict[str, Any] = {}
        if verify_after_upload:
            actual_nodes = _query_count(
                run_query(
                    "MATCH (n:KGNode {namespace: $namespace}) RETURN count(n) AS count",
                    {"namespace": neo4j_namespace},
                )
            )
            actual_edges = _query_count(
                run_query(
                    "MATCH ()-[r:KG_EDGE {namespace: $namespace}]->() RETURN count(r) AS count",
                    {"namespace": neo4j_namespace},
                )
            )
            verification = {
                "expected_nodes": len(nodes),
                "actual_nodes": actual_nodes,
                "expected_edges": len(edges),
                "actual_edges": actual_edges,
            }
            if actual_nodes != len(nodes) or actual_edges != len(edges):
                return StageResult.failure(
                    "Neo4j post-upload verification failed: "
                    f"expected nodes={len(nodes)} edges={len(edges)}, "
                    f"found nodes={actual_nodes} edges={actual_edges}"
                )

        manifest = {
            "schema_version": 2,
            "neo4j_endpoint": endpoint,
            "neo4j_database": neo4j_database,
            "neo4j_namespace": neo4j_namespace,
            "graph_type": graph_type,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "compact_upload": compact_upload,
            "allowed_node_types": allowed_node_types,
            "allowed_edge_types": allowed_edge_types,
            "clear_all_kg": clear_all_kg,
            "knowledge_graph_file": str(Path(graph_file).expanduser().resolve()),
            "knowledge_graph_kind": canonical_graph.kind if canonical_graph is not None else "legacy_external_graph",
            "knowledge_graph_sha256": (
                canonical_graph.graph_sha256
                if canonical_graph is not None
                else sha256_file(graph_file)
            ),
            "knowledge_graph_index_file": str(canonical_graph.index_file or "") if canonical_graph is not None else "",
            "knowledge_graph_index_sha256": canonical_graph.index_sha256 if canonical_graph is not None else "",
            "verification": verification,
            "synced_at": _now_iso(),
        }
        atomic_write_json(manifest_file, manifest)

        artifacts = [
            ctx.make_artifact(
                manifest_file,
                artifact_type="neo4j_graph_manifest",
                role="graph_store_manifest",
                metadata={
                    "database": neo4j_database,
                    "namespace": neo4j_namespace,
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                },
            ),
        ]

        logger.info(
            "Neo4j graph sync complete: namespace=%s nodes=%d edges=%d",
            neo4j_namespace,
            len(nodes),
            len(edges),
        )

        return StageResult.success(
            outputs={
                "neo4j_graph_manifest_file": str(manifest_file),
                "neo4j_graph_progress_file": str(progress_file),
                "neo4j_namespace": neo4j_namespace,
            },
            metrics={
                "neo4j_nodes_synced": len(nodes),
                "neo4j_edges_synced": len(edges),
            },
            checkpoint={
                "namespace": neo4j_namespace,
                "node_offset": len(nodes),
                "edge_offset": len(edges),
            },
            artifacts=artifacts,
        )
