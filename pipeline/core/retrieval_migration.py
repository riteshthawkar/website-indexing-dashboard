"""Production-safe retrieval artifact migration helpers.

This module upgrades an existing MBZUAI retrieval run into the current v2
artifact contract without re-scraping. It is intentionally conservative:
source chunks, parents, media, facts, answers, and graph edges must not shrink,
and generated v2-only records are derived from already-promoted local artifacts.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record, load_artifact_catalog, save_artifact_catalog
from pipeline.core.assertions import (
    build_answer_records_from_assertions,
    build_assertion_embedding_records,
    build_entity_record,
    build_entity_records_from_assertions,
    clean_text,
    merge_entity_records,
    normalize_predicate,
)
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.config import production_indexing_contract_fingerprint
from pipeline.core.graph_artifacts import resolve_canonical_graph_artifacts
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe, sha256_file
from pipeline.core.knowledge_graph import load_graph_bundle, save_graph_bundle_with_index
from pipeline.core.run_audit import audit_run, save_run_audit
from pipeline.core.state import PipelineState, StageState, load_state, now_iso, save_state
from pipeline.stages.formatters.gemini_retrieval_formatter import (
    _build_extractive_summary,
    _build_summary_embedding_text,
    _stable_id,
    _tokenize_for_bm25,
    _truncate_chars,
)


class RetrievalMigrationError(RuntimeError):
    """Raised when a migration would be lossy or internally inconsistent."""


@dataclass
class RetrievalMigrationOptions:
    source_work_dir: Path
    target_work_dir: Path
    config: Dict[str, Any]
    config_name: str = "default"
    target_run_id: str = ""
    force: bool = False
    dry_run: bool = False
    min_summary_coverage: float = 0.75
    min_assertion_records: int = 1
    summary_max_chars: int = 1800
    sparse_summary_max_chars: int = 1200


@dataclass
class PreparedMigration:
    manifest: Dict[str, Any]
    outputs: Dict[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        validation = self.manifest.get("validation") if isinstance(self.manifest, dict) else {}
        return bool((validation or {}).get("passed"))


_RECORD_FILES = {
    "chunk_records": ("chunk_dense_records.json", "chunk_count"),
    "parent_records": ("parent_dense_records.json", "parent_count"),
    "media_records": ("media_dense_records.json", "media_count"),
    "fact_records": ("fact_dense_records.json", "fact_count"),
    "summary_records": ("summary_dense_records.json", "summary_count"),
    "entity_records": ("entity_records.json", "entity_count"),
    "assertion_records": ("assertion_dense_records.json", "assertion_count"),
    "answer_records": ("answer_dense_records.json", "answer_count"),
}


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _read_json(path: Path, default: Any) -> Any:
    payload = load_json_safe(path, default)
    if payload is default:
        return default
    return payload


def _require_json(path: Path, expected_type: type, label: str) -> Any:
    if not path.is_file():
        raise RetrievalMigrationError(f"Missing {label}: {path}")
    payload = load_json_safe(path, None)
    if not isinstance(payload, expected_type):
        raise RetrievalMigrationError(f"Invalid {label}; expected {expected_type.__name__}: {path}")
    return payload


def _candidate_bundle_dirs(source_work_dir: Path) -> List[Path]:
    return [
        source_work_dir / "stage_outputs" / "format_retrieval",
        source_work_dir / "stage_outputs" / "build_retrieval_bundle",
    ]


def _locate_bundle_dir(source_work_dir: Path) -> Tuple[Path, Path]:
    for directory in _candidate_bundle_dirs(source_work_dir):
        bundle_path = directory / "retrieval_bundle.json"
        if bundle_path.is_file():
            return directory, bundle_path
    raise RetrievalMigrationError(
        "No retrieval_bundle.json found under stage_outputs/format_retrieval "
        f"or stage_outputs/build_retrieval_bundle in {source_work_dir}"
    )


def _locate_promoted_graph(source_work_dir: Path) -> Path:
    selected = resolve_canonical_graph_artifacts(
        source_work_dir,
        required=False,
        require_index=False,
        validate_binding=False,
    )
    if selected is None:
        raise RetrievalMigrationError(f"No local knowledge graph artifact found in {source_work_dir}")
    return selected.graph_file


def _load_records(
    *,
    bundle_dir: Path,
    bundle: Mapping[str, Any],
    bundle_key: str,
) -> List[Dict[str, Any]]:
    filename, _stat_key = _RECORD_FILES[bundle_key]
    path = bundle_dir / filename
    payload: Any = []
    if path.is_file():
        payload = _read_json(path, [])
    elif isinstance(bundle.get(bundle_key), list):
        payload = bundle.get(bundle_key) or []
    if not isinstance(payload, list):
        raise RetrievalMigrationError(f"Invalid {filename}; expected a JSON list")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _load_lexical_records(bundle_dir: Path, bundle: Mapping[str, Any]) -> List[Dict[str, Any]]:
    lexical_file = bundle_dir / "lexical_corpus.json"
    payload = _read_json(lexical_file, []) if lexical_file.is_file() else bundle.get("lexical_corpus", [])
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _graph_nodes(graph: Mapping[str, Any], node_type: str) -> List[Dict[str, Any]]:
    return [
        dict(node)
        for node in graph.get("nodes") or []
        if isinstance(node, Mapping) and node.get("node_type") == node_type
    ]


def _graph_stats(graph: Mapping[str, Any]) -> Dict[str, Any]:
    stats = graph.get("stats") if isinstance(graph.get("stats"), Mapping) else {}
    if stats:
        return dict(stats)
    node_type_counts: Dict[str, int] = {}
    for node in graph.get("nodes") or []:
        if not isinstance(node, Mapping):
            continue
        node_type = str(node.get("node_type") or "unknown")
        node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
    edge_type_counts: Dict[str, int] = {}
    for edge in graph.get("edges") or []:
        if not isinstance(edge, Mapping):
            continue
        edge_type = str(edge.get("edge_type") or "unknown")
        edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
    return {
        "node_count": len(graph.get("nodes") or []),
        "edge_count": len(graph.get("edges") or []),
        "node_type_counts": node_type_counts,
        "edge_type_counts": edge_type_counts,
    }


def _assertions_from_graph(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    assertions: List[Dict[str, Any]] = []
    for node in _graph_nodes(graph, "relation_assertion"):
        props = node.get("properties") if isinstance(node.get("properties"), Mapping) else {}
        assertion = dict(props or {})
        assertion_id = clean_text(node.get("id") or assertion.get("id"))
        if not assertion_id:
            continue
        relation_type = assertion.get("relation_type") or assertion.get("predicate") or node.get("label")
        predicate = normalize_predicate(relation_type)
        object_name = assertion.get("object_value") or assertion.get("object_name")
        assertion.update(
            {
                "id": assertion_id,
                "relation_type": relation_type,
                "predicate": predicate,
                "answer_type": assertion.get("answer_type") or predicate,
                "object_value": object_name,
                "validity_status": assertion.get("validity_status") or "active",
                "text": assertion.get("text") or assertion.get("evidence"),
                "canonical_subject": assertion.get("canonical_subject") or assertion.get("subject_entity_id"),
                "canonical_predicate": assertion.get("canonical_predicate") or predicate,
                "canonical_object": assertion.get("canonical_object") or object_name,
            }
        )
        assertions.append(assertion)
    return assertions


def _entities_from_graph(graph: Mapping[str, Any]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for node in _graph_nodes(graph, "entity"):
        props = node.get("properties") if isinstance(node.get("properties"), Mapping) else {}
        canonical_name = clean_text(props.get("canonical_name") or node.get("label"))
        if not canonical_name:
            continue
        records.append(
            build_entity_record(
                canonical_name=canonical_name,
                entity_type=props.get("entity_type") or "other",
                aliases=props.get("aliases") or [canonical_name],
                description=props.get("description") or "",
                confidence=props.get("confidence") or 0.0,
                source_chunk_ids=props.get("source_chunk_ids") or [],
                source_parent_ids=props.get("source_parent_ids") or [],
                source_urls=props.get("source_urls") or [],
                document_titles=props.get("document_titles") or [],
                entity_id=clean_text(node.get("id")),
            )
        )
    return records


def _build_summary_records(
    *,
    parent_records: Sequence[Mapping[str, Any]],
    chunk_records: Sequence[Mapping[str, Any]],
    summary_max_chars: int,
    sparse_summary_max_chars: int,
) -> List[Dict[str, Any]]:
    chunk_map = {str(chunk.get("id") or ""): dict(chunk) for chunk in chunk_records if chunk.get("id")}
    summaries: List[Dict[str, Any]] = []
    for parent in parent_records:
        child_chunks = [
            chunk_map[str(chunk_id)]
            for chunk_id in parent.get("child_chunk_ids") or []
            if str(chunk_id) in chunk_map
        ]
        summary_text = _build_extractive_summary(dict(parent), child_chunks, max_chars=summary_max_chars)
        if not summary_text:
            fallback = parent.get("dense_text") or parent.get("text") or parent.get("sparse_text")
            summary_text = _truncate_chars(fallback, summary_max_chars)
        if not summary_text:
            continue
        summary_id = _stable_id("summary", parent.get("id"), parent.get("document_id"), parent.get("source_url"))
        summary_record = {
            "id": summary_id,
            "record_type": "summary",
            "summary_type": parent.get("parent_type") or "section",
            "text": summary_text,
            "document_id": parent.get("document_id", ""),
            "document_title": parent.get("document_title", ""),
            "document_type": parent.get("document_type", ""),
            "source_markdown_path": parent.get("source_markdown_path", ""),
            "source_url": parent.get("source_url", ""),
            "page_key": parent.get("page_key") or (parent.get("id") if parent.get("parent_type") == "page" else ""),
            "section_key": parent.get("id") if parent.get("parent_type") == "section" else "",
            "section_path": list(parent.get("section_path") or []),
            "page_numbers": list(parent.get("page_numbers") or []),
            "linked_parent_ids": [parent.get("id")] if parent.get("id") else [],
            "linked_chunk_ids": list(parent.get("child_chunk_ids") or []),
        }
        summary_record["dense_text"] = _build_summary_embedding_text(summary_record)
        summary_record["lexical_text"] = _truncate_chars(summary_text, sparse_summary_max_chars)
        summary_record["sparse_text"] = summary_record["lexical_text"]
        summaries.append(summary_record)
    return summaries


def _record_key(record: Mapping[str, Any]) -> Tuple[str, str]:
    return (clean_text(record.get("record_type") or ""), clean_text(record.get("id") or ""))


def _append_lexical_record(
    lexical_records: List[Dict[str, Any]],
    seen: set[Tuple[str, str]],
    *,
    record_id: str,
    record_type: str,
    text: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    key = (record_type, record_id)
    if not record_id or key in seen:
        return
    payload = {
        "id": record_id,
        "record_type": record_type,
        "text": text,
        "tokens": _tokenize_for_bm25(text),
    }
    if extra:
        payload.update(extra)
    lexical_records.append(payload)
    seen.add(key)


def _ensure_v2_lexical_records(
    lexical_records: Sequence[Mapping[str, Any]],
    *,
    summary_records: Sequence[Mapping[str, Any]],
    assertion_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output = [dict(item) for item in lexical_records if isinstance(item, Mapping)]
    seen = {_record_key(item) for item in output if _record_key(item)[1]}
    for summary in summary_records:
        _append_lexical_record(
            output,
            seen,
            record_id=clean_text(summary.get("id")),
            record_type="summary",
            text=clean_text(summary.get("sparse_text") or summary.get("lexical_text") or summary.get("text")),
            extra={"summary_type": summary.get("summary_type")},
        )
    for assertion in assertion_records:
        _append_lexical_record(
            output,
            seen,
            record_id=clean_text(assertion.get("id")),
            record_type="assertion",
            text=clean_text(assertion.get("lexical_text") or assertion.get("text")),
        )
    return output


def _answer_semantic_key(record: Mapping[str, Any]) -> Tuple[str, str, str, str]:
    return (
        clean_text(record.get("answer_type")).lower(),
        clean_text(record.get("answer_subtype")).lower(),
        clean_text(record.get("subject_text")).lower(),
        clean_text(record.get("value") or record.get("text")).lower(),
    )


def _merge_answer_records_lossless(
    existing_records: Sequence[Mapping[str, Any]],
    promoted_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output = [dict(record) for record in existing_records if isinstance(record, Mapping)]
    seen_ids = {clean_text(record.get("id")) for record in output if clean_text(record.get("id"))}
    seen_keys = {_answer_semantic_key(record) for record in output}
    for record in promoted_records or []:
        if not isinstance(record, Mapping):
            continue
        record_id = clean_text(record.get("id"))
        key = _answer_semantic_key(record)
        if (record_id and record_id in seen_ids) or key in seen_keys:
            continue
        output.append(dict(record))
        if record_id:
            seen_ids.add(record_id)
        seen_keys.add(key)
    return output


def _count_records(records_by_key: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for bundle_key, records in records_by_key.items():
        _filename, stat_key = _RECORD_FILES[bundle_key]
        counts[stat_key] = len(records or [])
    return counts


def _validate_counts(
    *,
    source_counts: Mapping[str, int],
    target_counts: Mapping[str, int],
    graph_stats: Mapping[str, Any],
    relation_assertion_count: int,
    options: RetrievalMigrationOptions,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for key in ("chunk_count", "parent_count", "media_count", "fact_count"):
        if int(target_counts.get(key) or 0) != int(source_counts.get(key) or 0):
            errors.append(
                f"{key} changed during migration: source={source_counts.get(key, 0)} "
                f"target={target_counts.get(key, 0)}"
            )

    if int(target_counts.get("answer_count") or 0) < int(source_counts.get("answer_count") or 0):
        errors.append(
            f"answer_count regressed: source={source_counts.get('answer_count', 0)} "
            f"target={target_counts.get('answer_count', 0)}"
        )

    parent_count = max(1, int(target_counts.get("parent_count") or 0))
    summary_count = int(target_counts.get("summary_count") or 0)
    summary_coverage = summary_count / float(parent_count)
    if summary_coverage < float(options.min_summary_coverage):
        errors.append(
            f"summary coverage is below production threshold: "
            f"{summary_count}/{parent_count} ({summary_coverage:.3f}) < {options.min_summary_coverage:.3f}"
        )

    assertion_count = int(target_counts.get("assertion_count") or 0)
    if assertion_count < int(options.min_assertion_records):
        errors.append(
            f"assertion_count below required minimum: {assertion_count} < {options.min_assertion_records}"
        )
    if relation_assertion_count and assertion_count < relation_assertion_count:
        errors.append(
            f"assertion vector records do not cover promoted graph assertions: "
            f"{assertion_count} < {relation_assertion_count}"
        )

    node_count = int(graph_stats.get("node_count") or 0)
    edge_count = int(graph_stats.get("edge_count") or 0)
    if node_count <= 0 or edge_count <= 0:
        errors.append(f"knowledge graph is empty or invalid: nodes={node_count}, edges={edge_count}")
    if relation_assertion_count <= 0:
        errors.append("promoted knowledge graph has no relation_assertion nodes")

    if int(target_counts.get("entity_count") or 0) <= 0:
        warnings.append("migrated retrieval bundle has no entity records")

    return errors, warnings


def _write_record_files(
    *,
    target_format_dir: Path,
    records_by_key: Mapping[str, Sequence[Mapping[str, Any]]],
    lexical_records: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for bundle_key, records in records_by_key.items():
        filename, output_key = _RECORD_FILES[bundle_key]
        path = target_format_dir / filename
        atomic_write_json(path, list(records))
        outputs[f"{output_key.replace('_count', '')}_embedding_file"] = str(path)
    lexical_file = target_format_dir / "lexical_corpus.json"
    bundle_file = target_format_dir / "retrieval_bundle.json"
    atomic_write_json(lexical_file, list(lexical_records))
    atomic_write_json(bundle_file, dict(bundle))
    outputs["lexical_corpus_file"] = str(lexical_file)
    outputs["retrieval_bundle_file"] = str(bundle_file)
    return outputs


def _register_artifacts(
    *,
    target_work_dir: Path,
    outputs: Mapping[str, str],
    graph_file: Path,
    migration_manifest_file: Path,
    stats: Mapping[str, Any],
    graph_stats: Mapping[str, Any],
) -> ArtifactCatalog:
    catalog = ArtifactCatalog()
    catalog.add(
        build_artifact_record(
            artifact_type="retrieval_migration_manifest",
            role="migration_manifest",
            producer_stage="migrate_retrieval",
            uri=str(migration_manifest_file),
            local_path=migration_manifest_file,
            metadata={"schema_version": 1},
        )
    )
    catalog.add(
        build_artifact_record(
            artifact_type="retrieval_bundle",
            role="retrieval_corpus",
            producer_stage="format_retrieval",
            uri=str(outputs["retrieval_bundle_file"]),
            local_path=outputs["retrieval_bundle_file"],
            metadata=dict(stats),
        )
    )
    catalog.add(
        build_artifact_record(
            artifact_type="knowledge_graph_bundle",
            role="knowledge_graph",
            producer_stage="promote_graph",
            uri=str(graph_file),
            local_path=graph_file,
            metadata=dict(graph_stats),
        )
    )
    promoted_assertions_file = outputs.get("promoted_assertions_file")
    if promoted_assertions_file:
        catalog.add(
            build_artifact_record(
                artifact_type="promoted_assertions",
                role="assertion_promoted",
                producer_stage="format_retrieval",
                uri=str(promoted_assertions_file),
                local_path=promoted_assertions_file,
                metadata={"records": int(stats.get("assertion_count") or 0)},
            )
        )
    for output_key, path_value in outputs.items():
        if not output_key.endswith("_embedding_file") and output_key != "lexical_corpus_file":
            continue
        path = Path(path_value)
        record_type = path.stem.replace("_dense_records", "").replace("_records", "")
        catalog.add(
            build_artifact_record(
                artifact_type="retrieval_records",
                role=record_type,
                producer_stage="format_retrieval",
                uri=str(path),
                local_path=path,
                metadata={"record_type": record_type},
            )
        )
    save_artifact_catalog(catalog, target_work_dir)
    return catalog


def _write_state(
    *,
    target_work_dir: Path,
    run_id: str,
    project_name: str,
    migration_manifest_file: Path,
    format_outputs: Mapping[str, str],
    graph_outputs: Mapping[str, str],
    stats: Mapping[str, Any],
    graph_stats: Mapping[str, Any],
    catalog: ArtifactCatalog,
) -> PipelineState:
    now = now_iso()

    def _artifact_ids_for_stage(stage_id: str) -> List[str]:
        return [record.artifact_id for record in catalog.records if record.producer_stage == stage_id]

    state = PipelineState(
        run_id=run_id,
        project_name=project_name,
        status="completed",
        started_at=now,
        finished_at=now,
        current_stage_index=3,
        stages=[
            StageState(
                name="retrieval_v2_migration",
                stage_type="formatter",
                stage_id="migrate_retrieval",
                status="completed",
                started_at=now,
                finished_at=now,
                outputs={"migration_manifest_file": str(migration_manifest_file)},
                metrics={"schema_version": 1},
                artifact_ids=_artifact_ids_for_stage("migrate_retrieval"),
            ),
            StageState(
                name="promote_graph",
                stage_type="formatter",
                stage_id="promote_graph",
                status="completed",
                started_at=now,
                finished_at=now,
                outputs=dict(graph_outputs),
                metrics=dict(graph_stats),
                artifact_ids=_artifact_ids_for_stage("promote_graph"),
            ),
            StageState(
                name="format_retrieval",
                stage_type="formatter",
                stage_id="format_retrieval",
                status="completed",
                started_at=now,
                finished_at=now,
                outputs=dict(format_outputs),
                metrics=dict(stats),
                artifact_ids=_artifact_ids_for_stage("format_retrieval"),
            ),
        ],
    )
    save_state(state, target_work_dir)
    return state


def prepare_retrieval_v2_migration(options: RetrievalMigrationOptions) -> PreparedMigration:
    source_work_dir = _as_path(options.source_work_dir)
    target_work_dir = _as_path(options.target_work_dir)
    if not source_work_dir.is_dir():
        raise RetrievalMigrationError(f"Source run directory does not exist: {source_work_dir}")
    if source_work_dir == target_work_dir:
        raise RetrievalMigrationError("Source and target run directories must be different")
    if target_work_dir.exists() and any(target_work_dir.iterdir()) and not options.force and not options.dry_run:
        raise RetrievalMigrationError(f"Target work dir is not empty; pass --force to overwrite: {target_work_dir}")

    bundle_dir, bundle_path = _locate_bundle_dir(source_work_dir)
    graph_path = _locate_promoted_graph(source_work_dir)
    bundle = _require_json(bundle_path, dict, "retrieval bundle")
    graph = _require_json(graph_path, dict, "knowledge graph")

    records_by_key: Dict[str, List[Dict[str, Any]]] = {
        key: _load_records(bundle_dir=bundle_dir, bundle=bundle, bundle_key=key)
        for key in _RECORD_FILES
    }
    lexical_records = _load_lexical_records(bundle_dir, bundle)

    source_counts = _count_records(records_by_key)
    for required_key in ("chunk_count", "parent_count", "fact_count"):
        if int(source_counts.get(required_key) or 0) <= 0:
            raise RetrievalMigrationError(f"Source retrieval bundle has no {required_key.replace('_count', '')} records")

    graph_stats = _graph_stats(graph)
    graph_assertions = _assertions_from_graph(graph)
    graph_entities = _entities_from_graph(graph)

    generated_summary_records = False
    if not records_by_key["summary_records"]:
        records_by_key["summary_records"] = _build_summary_records(
            parent_records=records_by_key["parent_records"],
            chunk_records=records_by_key["chunk_records"],
            summary_max_chars=options.summary_max_chars,
            sparse_summary_max_chars=options.sparse_summary_max_chars,
        )
        generated_summary_records = True

    generated_assertion_records = False
    if not records_by_key["assertion_records"]:
        records_by_key["assertion_records"] = build_assertion_embedding_records(graph_assertions)
        generated_assertion_records = True

    graph_derived_entities = merge_entity_records(graph_entities, build_entity_records_from_assertions(graph_assertions))
    records_by_key["entity_records"] = merge_entity_records(records_by_key["entity_records"], graph_derived_entities)

    promoted_answers = build_answer_records_from_assertions(graph_assertions)
    records_by_key["answer_records"] = _merge_answer_records_lossless(records_by_key["answer_records"], promoted_answers)

    lexical_records = _ensure_v2_lexical_records(
        lexical_records,
        summary_records=records_by_key["summary_records"],
        assertion_records=records_by_key["assertion_records"],
    )

    target_counts = _count_records(records_by_key)
    target_counts["lexical_count"] = len(lexical_records)

    validation_errors, validation_warnings = _validate_counts(
        source_counts=source_counts,
        target_counts=target_counts,
        graph_stats=graph_stats,
        relation_assertion_count=len(graph_assertions),
        options=options,
    )

    run_id = options.target_run_id or target_work_dir.name
    project_name = str(options.config.get("project_name") or "default")
    migrated_bundle = dict(bundle)
    migrated_bundle.update(
        {
            "version": max(5, int(bundle.get("version") or 0)),
            "generated_at": run_id,
            "migration": {
                "schema_version": 1,
                "source_work_dir": str(source_work_dir),
                "source_bundle_file": str(bundle_path),
                "source_graph_file": str(graph_path),
                "migrated_at": now_iso(),
            },
            "stats": {
                **dict(bundle.get("stats") if isinstance(bundle.get("stats"), Mapping) else {}),
                **target_counts,
            },
        }
    )
    for key, records in records_by_key.items():
        migrated_bundle[key] = records

    manifest = {
        "schema_version": 1,
        "migration_type": "retrieval_v2_from_existing_artifacts",
        "status": "dry_run_ready" if options.dry_run and not validation_errors else ("ready" if not validation_errors else "failed"),
        "config_name": options.config_name,
        "project_name": project_name,
        "source_work_dir": str(source_work_dir),
        "target_work_dir": str(target_work_dir),
        "source_run_id": source_work_dir.name,
        "target_run_id": run_id,
        "source_artifacts": {
            "retrieval_bundle_file": str(bundle_path),
            "retrieval_bundle_sha256": sha256_file(bundle_path),
            "knowledge_graph_file": str(graph_path),
            "knowledge_graph_sha256": sha256_file(graph_path),
        },
        "source_counts": dict(source_counts),
        "target_counts": dict(target_counts),
        "graph": {
            "stats": graph_stats,
            "relation_assertion_nodes": len(graph_assertions),
            "entity_nodes": len(graph_entities),
        },
        "generated": {
            "summary_records": generated_summary_records,
            "assertion_records": generated_assertion_records,
            "promoted_answer_records_available": len(promoted_answers),
        },
        "validation": {
            "passed": not validation_errors,
            "errors": validation_errors,
            "warnings": validation_warnings,
            "min_summary_coverage": options.min_summary_coverage,
            "min_assertion_records": options.min_assertion_records,
        },
    }

    if validation_errors:
        if not options.dry_run:
            manifest_dir = ensure_dir(target_work_dir / "stage_outputs" / "migrate_retrieval")
            manifest_path = manifest_dir / "migration_manifest.json"
            atomic_write_json(manifest_path, manifest)
            manifest["manifest_file"] = str(manifest_path)
        return PreparedMigration(manifest=manifest)

    if options.dry_run:
        return PreparedMigration(manifest=manifest)

    if target_work_dir.exists() and options.force:
        shutil.rmtree(target_work_dir)
    ensure_dir(target_work_dir)
    ensure_dir(target_work_dir / "stage_outputs")
    target_format_dir = ensure_dir(target_work_dir / "stage_outputs" / "format_retrieval")
    target_graph_dir = ensure_dir(target_work_dir / "stage_outputs" / "promote_graph")
    target_assertions_dir = ensure_dir(
        target_work_dir / "stage_outputs" / "promote_assertions"
    )
    target_migration_dir = ensure_dir(target_work_dir / "stage_outputs" / "migrate_retrieval")

    graph_target = target_graph_dir / "promoted_knowledge_graph.json"
    shutil.copy2(graph_path, graph_target)
    graph_index_target = target_graph_dir / "promoted_knowledge_graph_index.json"
    migrated_graph = load_graph_bundle(graph_target)
    save_graph_bundle_with_index(migrated_graph, graph_target, graph_index_target)

    format_outputs = _write_record_files(
        target_format_dir=target_format_dir,
        records_by_key=records_by_key,
        lexical_records=lexical_records,
        bundle=migrated_bundle,
    )
    promoted_assertions_file = target_assertions_dir / "promoted_assertions.json"
    atomic_write_json(promoted_assertions_file, graph_assertions)
    format_outputs["promoted_assertions_file"] = str(promoted_assertions_file)
    graph_outputs = {
        "promoted_knowledge_graph_file": str(graph_target),
        "promoted_knowledge_graph_index_file": str(graph_index_target),
        "knowledge_graph_file": str(graph_target),
        "knowledge_graph_index_file": str(graph_index_target),
    }

    manifest["outputs"] = {**format_outputs, **graph_outputs}
    manifest["target_artifacts"] = {
        "retrieval_bundle_file": format_outputs["retrieval_bundle_file"],
        "retrieval_bundle_sha256": sha256_file(format_outputs["retrieval_bundle_file"]),
        "lexical_corpus_file": format_outputs["lexical_corpus_file"],
        "lexical_corpus_sha256": sha256_file(format_outputs["lexical_corpus_file"]),
        "promoted_assertions_file": str(promoted_assertions_file),
        "promoted_assertions_sha256": sha256_file(promoted_assertions_file),
        "knowledge_graph_file": str(graph_target),
        "knowledge_graph_sha256": sha256_file(graph_target),
        "knowledge_graph_index_file": str(graph_index_target),
        "knowledge_graph_index_sha256": sha256_file(graph_index_target),
    }
    manifest_path = target_migration_dir / "migration_manifest.json"
    atomic_write_json(manifest_path, manifest)
    manifest["manifest_file"] = str(manifest_path)
    atomic_write_json(manifest_path, manifest)

    atomic_write_json(
        target_work_dir / "resolved_config.json",
        {
            "run_id": run_id,
            "project_name": project_name,
            "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(
                options.config
            ),
            "config": options.config,
            "migration": {
                "source_work_dir": str(source_work_dir),
                "manifest_file": str(manifest_path),
            },
        },
    )

    catalog = _register_artifacts(
        target_work_dir=target_work_dir,
        outputs=format_outputs,
        graph_file=graph_target,
        migration_manifest_file=manifest_path,
        stats=target_counts,
        graph_stats=graph_stats,
    )
    state = _write_state(
        target_work_dir=target_work_dir,
        run_id=run_id,
        project_name=project_name,
        migration_manifest_file=manifest_path,
        format_outputs=format_outputs,
        graph_outputs=graph_outputs,
        stats=target_counts,
        graph_stats=graph_stats,
        catalog=catalog,
    )
    audit = audit_run(target_work_dir, state=state, artifact_catalog=catalog)
    save_run_audit(audit, target_work_dir)
    if not audit.ok:
        manifest["status"] = "failed"
        manifest["validation"]["passed"] = False
        manifest["validation"]["errors"].extend(
            f"run audit failed: {issue.code} - {issue.message}" for issue in audit.errors
        )
        atomic_write_json(manifest_path, manifest)
        return PreparedMigration(manifest=manifest, outputs=format_outputs)

    return PreparedMigration(manifest=manifest, outputs=format_outputs)


def upload_migrated_retrieval(
    *,
    config: Dict[str, Any],
    target_work_dir: str | Path,
) -> Dict[str, Any]:
    """Run the production Pinecone uploader against a migrated v2 run."""

    from pipeline.stages.embedders.gemini_pinecone_embedder import GeminiPineconeEmbedder

    work_dir = _as_path(target_work_dir)
    migration_manifest_path = work_dir / "stage_outputs" / "migrate_retrieval" / "migration_manifest.json"
    manifest = _require_json(migration_manifest_path, dict, "migration manifest")
    if not (manifest.get("validation") or {}).get("passed"):
        raise RetrievalMigrationError("Migration manifest did not pass validation; refusing upload")

    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), Mapping) else {}
    required_keys = [
        "chunk_embedding_file",
        "parent_embedding_file",
        "media_embedding_file",
        "fact_embedding_file",
        "summary_embedding_file",
        "assertion_embedding_file",
        "retrieval_bundle_file",
        "promoted_knowledge_graph_file",
    ]
    missing = [key for key in required_keys if not outputs.get(key) or not Path(str(outputs[key])).is_file()]
    if missing:
        raise RetrievalMigrationError(f"Migration outputs are incomplete; missing: {', '.join(missing)}")

    catalog = load_artifact_catalog(work_dir)
    ctx = StageContext(
        run_id=str(manifest.get("target_run_id") or work_dir.name),
        project_name=str(manifest.get("project_name") or config.get("project_name") or "default"),
        config=config,
        work_dir=work_dir,
        previous_outputs=dict(outputs),
        stage_definition={"id": "upload_retrieval", "type": "embedder", "plugin": "gemini_pinecone"},
        stage_index=3,
        stage_id="upload_retrieval",
        artifact_catalog=catalog,
    )

    result = asyncio.run(GeminiPineconeEmbedder().execute(ctx))
    if result.status != StageStatus.COMPLETED:
        raise RetrievalMigrationError(result.error_message or "Gemini Pinecone upload failed")

    catalog.extend(result.artifacts)
    save_artifact_catalog(catalog, work_dir)

    state = load_state(work_dir)
    if state is None:
        raise RetrievalMigrationError("Migrated run is missing pipeline_state.json after migration")
    now = now_iso()
    upload_stage = StageState(
        name="gemini_pinecone",
        stage_type="embedder",
        stage_id="upload_retrieval",
        status="completed",
        started_at=now,
        finished_at=now,
        outputs=dict(result.outputs),
        metrics=dict(result.metrics),
        artifact_ids=[record.artifact_id for record in result.artifacts if hasattr(record, "artifact_id")],
    )
    state.stages = [stage for stage in state.stages if stage.stage_id != "upload_retrieval"]
    state.stages.append(upload_stage)
    state.current_stage_index = len(state.stages)
    state.status = "completed"
    state.finished_at = now
    save_state(state, work_dir)

    manifest["status"] = "uploaded"
    manifest["upload"] = {
        "uploaded_at": now,
        "outputs": dict(result.outputs),
        "metrics": dict(result.metrics),
    }
    atomic_write_json(migration_manifest_path, manifest)

    audit = audit_run(work_dir, state=state, artifact_catalog=catalog)
    save_run_audit(audit, work_dir)
    if not audit.ok:
        raise RetrievalMigrationError(
            "Upload completed but run audit failed: "
            + "; ".join(f"{issue.code} - {issue.message}" for issue in audit.errors[:5])
        )
    return {
        "manifest_file": str(migration_manifest_path),
        "upload_outputs": dict(result.outputs),
        "upload_metrics": dict(result.metrics),
        "audit": audit.to_dict(),
    }
