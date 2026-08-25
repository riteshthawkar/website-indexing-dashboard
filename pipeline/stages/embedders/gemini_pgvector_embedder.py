"""Gemini dense embedding upload stage for the production pgvector store."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from pipeline.core.base import EmbedderStage, StageContext, StageResult
from pipeline.core.config import production_indexing_contract_fingerprint
from pipeline.core.graph_artifacts import resolve_canonical_graph_artifacts
from pipeline.core.io import atomic_write_json, combine_sha256_digests, load_json_safe, sha256_file
from pipeline.core.release_assembly import (
    SELECTED_DENSE_RECORD_KINDS,
    SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
    selected_release_file_path,
)
from pipeline.core.registry import register_stage
from pipeline.stages.embedders.gemini_pinecone_embedder import (
    _call_with_retry,
    _dedupe_records_by_id,
    _embed_multimodal_batch,
    _embed_text_batch,
    _graph_entity_record,
    _iter_batches,
    _load_records,
    _make_gemini_client,
    _normalise_entity_record,
    _record_embedding_text,
    _record_metadata,
    _record_text_fingerprint,
    _resolve_indexing_input_paths,
    _resolve_upload_namespaces,
)
from pipeline.vectorstores.pgvector_store import PgVectorStore


logger = logging.getLogger(__name__)
_LANE_KINDS = {
    "chunks": "chunk",
    "parents": "parent",
    "media": "media",
    "page_cards": "page_card",
    "actions": "action",
    "facts": "fact",
    "evidence_spans": "evidence_span",
    "summaries": "summary",
    "assertions": "assertion",
    "entities": "entity",
    "communities": "community",
}


def _graph_fallback_records(
    graph_bundle_file: str,
    *,
    entities: List[Dict[str, Any]],
    communities: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not graph_bundle_file or (entities and communities):
        return entities, communities
    payload = load_json_safe(graph_bundle_file, {}) or {}
    load_entities = not entities
    load_communities = not communities
    for node in payload.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("node_type") or "")
        if node_type == "entity" and load_entities:
            entities.append(_graph_entity_record(node))
        elif node_type in {"community", "community_summary"} and load_communities:
            properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
            title = properties.get("title") or node.get("title") or node.get("label") or ""
            summary = properties.get("summary") or node.get("summary") or ""
            community_id = (
                properties.get("community_id")
                or node.get("community_id")
                or node.get("id")
            )
            communities.append(
                {
                    "id": community_id,
                    "text": f"{title}\n{summary}".strip(),
                    "title": title,
                    "summary": summary,
                    "community_id": community_id,
                    "size": properties.get("size"),
                    "level": properties.get("level"),
                }
            )
    return entities, communities


def _load_lane_records(
    paths: Mapping[str, str],
    *,
    enable_dense_facts: bool,
    enable_dense_evidence_spans: bool,
    enable_dense_assertions: bool,
    enable_dense_summaries: bool = True,
    enable_dense_entities: bool = True,
    enable_dense_communities: bool = True,
    selected_profile: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    entities = [
        _normalise_entity_record(record)
        for record in (_load_records(paths["entities"]) if paths.get("entities") else [])
    ]
    communities = _load_records(paths["communities"]) if paths.get("communities") else []
    entities, communities = _graph_fallback_records(
        paths.get("graph_bundle") or "",
        entities=entities,
        communities=communities,
    )
    lanes = {
        "chunks": _load_records(paths["chunks"]) if paths.get("chunks") else [],
        "parents": _load_records(paths["parents"]) if paths.get("parents") else [],
        "media": _load_records(paths["media"]) if paths.get("media") else [],
        "page_cards": _load_records(paths["page_cards"]) if paths.get("page_cards") else [],
        "actions": _load_records(paths["actions"]) if paths.get("actions") else [],
        "facts": _load_records(paths["facts"]) if paths.get("facts") and enable_dense_facts else [],
        "evidence_spans": (
            _load_records(paths["evidence_spans"])
            if paths.get("evidence_spans") and enable_dense_evidence_spans
            else []
        ),
        "summaries": (
            _load_records(paths["summaries"])
            if paths.get("summaries") and enable_dense_summaries
            else []
        ),
        "assertions": (
            _load_records(paths["assertions"])
            if paths.get("assertions") and enable_dense_assertions
            else []
        ),
        "entities": entities if enable_dense_entities else [],
        "communities": communities if enable_dense_communities else [],
    }
    if selected_profile:
        for lane in (
            "facts",
            "evidence_spans",
            "summaries",
            "assertions",
            "entities",
            "communities",
        ):
            lanes[lane] = []
        _add_selected_release_links(lanes)
    return {
        lane: _dedupe_records_by_id(records, record_kind=_LANE_KINDS[lane])
        for lane, records in lanes.items()
    }


def _add_selected_release_links(lanes: Dict[str, List[Dict[str, Any]]]) -> None:
    """Add runtime-only graph links without changing evaluated embedding text."""

    chunks_by_page: Dict[str, List[str]] = {}
    chunks_by_media: Dict[str, List[str]] = {}
    for record in lanes.get("chunks") or []:
        record["record_type"] = "chunk"
        record["dense_text"] = str(record.get("text") or "")
        chunk_id = str(record.get("id") or "")
        for page_card_id in record.get("page_card_ids") or []:
            chunks_by_page.setdefault(str(page_card_id), []).append(chunk_id)
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        for media_id in metadata.get("media_ids") or []:
            chunks_by_media.setdefault(str(media_id), []).append(chunk_id)

    for record in lanes.get("parents") or []:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        record["record_type"] = str(record.get("kind") or "parent")
        record["parent_type"] = (
            "section" if str(record.get("kind") or "") == "parent_section" else "page"
        )
        record["child_chunk_ids"] = list(metadata.get("child_chunk_ids") or [])
        record["linked_chunk_ids"] = list(metadata.get("child_chunk_ids") or [])
        record["dense_text"] = str(record.get("text") or "")

    for record in lanes.get("media") or []:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        media_id = str(record.get("media_id") or record.get("id") or "")
        record["record_type"] = "media"
        record["media_type"] = "image"
        record["linked_chunk_ids"] = list(dict.fromkeys(chunks_by_media.get(media_id, [])))
        record["can_embed_multimodal"] = bool(
            metadata.get("can_embed_multimodal") and str(record.get("local_path") or "")
        )
        record["dense_text"] = str(record.get("text") or "")

    for record in lanes.get("page_cards") or []:
        page_card_id = str(record.get("id") or "")
        record["record_type"] = "page_card"
        record["linked_chunk_ids"] = list(dict.fromkeys(chunks_by_page.get(page_card_id, [])))
        record["dense_text"] = str(record.get("text") or "")

    for record in lanes.get("actions") or []:
        page_ids = [str(value) for value in record.get("page_card_ids") or [] if value]
        linked: List[str] = []
        for page_card_id in page_ids:
            linked.extend(chunks_by_page.get(page_card_id, []))
        metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
        record["record_type"] = "action"
        record["linked_chunk_ids"] = list(dict.fromkeys(linked))
        record["action_type"] = metadata.get("action_type")
        record["target_url"] = metadata.get("target_url")
        record["official_target"] = metadata.get("official_target")
        record["dense_text"] = str(record.get("text") or "")


def _validate_selected_lanes(
    *,
    paths: Mapping[str, str],
    lanes: Mapping[str, List[Dict[str, Any]]],
    variant_id: str,
) -> Mapping[str, Any]:
    assembly_file = Path(str(paths.get("release_assembly") or "")).expanduser().resolve()
    payload = load_json_safe(assembly_file, None)
    if not isinstance(payload, Mapping):
        raise ValueError("Selected-profile upload requires a valid release assembly manifest")
    if str(payload.get("schema_version") or "") != SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION:
        raise ValueError("Selected release assembly schema is unsupported")
    if str(payload.get("status") or "") != "ready_for_embedding":
        raise ValueError("Selected release assembly is not ready for embedding")
    if str(payload.get("variant_id") or "") != variant_id:
        raise ValueError("Selected release assembly variant does not match the runtime profile")
    if tuple(payload.get("record_kinds") or []) != SELECTED_DENSE_RECORD_KINDS:
        raise ValueError("Selected release assembly record kinds differ from the evaluated profile")

    lane_file_keys = {
        "chunks": "chunks",
        "parents": "parents",
        "media": "media",
        "page_cards": "page_cards",
        "actions": "actions",
    }
    expected_counts = payload.get("dense_lane_counts") if isinstance(payload.get("dense_lane_counts"), Mapping) else {}
    for lane, file_key in lane_file_keys.items():
        source_file = selected_release_file_path(payload, assembly_file, file_key)
        source_records = _load_records(source_file)
        actual_records = lanes.get(lane) or []
        expected_count = int(expected_counts.get(lane) or 0)
        if expected_count <= 0 or len(source_records) != expected_count or len(actual_records) != expected_count:
            raise ValueError(
                f"Selected release lane {lane} count mismatch: expected {expected_count}, "
                f"source {len(source_records)}, upload {len(actual_records)}"
            )
        if _record_text_fingerprint(source_records) != _record_text_fingerprint(actual_records):
            raise ValueError(f"Selected release lane {lane} ID/text fingerprint drifted")
    for lane in ("facts", "evidence_spans", "summaries", "assertions", "entities", "communities"):
        if lanes.get(lane):
            raise ValueError(f"Selected release contains unevaluated dense lane: {lane}")
    return payload


def _artifact_identity(ctx: StageContext, paths: Mapping[str, str]) -> Dict[str, Any]:
    production = bool((ctx.config.get("pipeline") or {}).get("production_profile", False))
    selected_profile = ctx.config.get("selected_profile") if isinstance(ctx.config.get("selected_profile"), Mapping) else {}
    selected = bool(str(selected_profile.get("variant_id") or ""))
    bundle = str(paths.get("bundle") or "")
    lexical = str(paths.get("lexical_corpus") or "")
    assertions = str(paths.get("promoted_assertions") or "")
    graph = str(paths.get("graph_bundle") or "")
    release_assembly = str(paths.get("release_assembly") or "")
    navigation_catalog = str(paths.get("navigation_catalog") or "")
    if not bundle:
        raise ValueError("retrieval formatter outputs are required before Gemini indexing")
    if production and not lexical:
        raise ValueError("Production indexing requires the lexical retrieval corpus sidecar")
    if production and not assertions:
        raise ValueError("Production indexing requires the promoted assertions sidecar")
    if selected and (not release_assembly or not navigation_catalog):
        raise ValueError(
            "Selected-profile indexing requires the release assembly and navigation catalog"
        )

    canonical_graph = resolve_canonical_graph_artifacts(
        ctx.work_dir,
        required=False,
        require_index=True,
        validate_binding=production,
    )
    graph_kind = ""
    graph_index_file = ""
    graph_index_sha = ""
    if canonical_graph is not None and graph:
        if canonical_graph.graph_file != Path(graph).expanduser().resolve():
            raise ValueError("Resolved graph input does not match the canonical graph artifact")
        graph_kind = canonical_graph.kind
        graph_index_file = str(canonical_graph.index_file or "")
        graph_index_sha = canonical_graph.index_sha256
    elif graph:
        graph_kind = "legacy_external_graph"

    hashes = {
        "retrieval_bundle_sha256": sha256_file(bundle),
        "lexical_corpus_sha256": sha256_file(lexical) if lexical else "",
        "promoted_assertions_sha256": sha256_file(assertions) if assertions else "",
        "knowledge_graph_sha256": sha256_file(graph) if graph else "",
        "knowledge_graph_index_sha256": graph_index_sha,
        "selected_release_assembly_sha256": (
            sha256_file(release_assembly) if release_assembly else ""
        ),
        "page_graph_navigation_catalog_sha256": (
            sha256_file(navigation_catalog) if navigation_catalog else ""
        ),
    }
    assembly_binding_sha = ""
    if release_assembly:
        assembly_payload = load_json_safe(release_assembly, None)
        if not isinstance(assembly_payload, Mapping):
            raise ValueError("Selected release assembly manifest is invalid")
        assembly_binding_sha = str(assembly_payload.get("assembly_sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", assembly_binding_sha):
            raise ValueError("Selected release assembly binding digest is invalid")
        declared_navigation = (
            (assembly_payload.get("files") or {}).get("navigation_catalog")
            if isinstance(assembly_payload.get("files"), Mapping)
            else {}
        )
        if not isinstance(declared_navigation, Mapping) or str(
            declared_navigation.get("sha256") or ""
        ).strip().lower() != hashes["page_graph_navigation_catalog_sha256"]:
            raise ValueError("Runtime navigation catalog differs from the selected release assembly")
    hashes["selected_release_binding_sha256"] = assembly_binding_sha
    if selected:
        upload_input_sha = combine_sha256_digests(
            hashes["retrieval_bundle_sha256"],
            hashes["lexical_corpus_sha256"],
            hashes["promoted_assertions_sha256"],
            hashes["knowledge_graph_sha256"],
            hashes["knowledge_graph_index_sha256"],
            hashes["selected_release_assembly_sha256"],
            hashes["selected_release_binding_sha256"],
            hashes["page_graph_navigation_catalog_sha256"],
        )
    elif hashes["lexical_corpus_sha256"] and hashes["promoted_assertions_sha256"]:
        upload_input_sha = combine_sha256_digests(
            hashes["retrieval_bundle_sha256"],
            hashes["lexical_corpus_sha256"],
            hashes["promoted_assertions_sha256"],
            hashes["knowledge_graph_sha256"],
            hashes["knowledge_graph_index_sha256"],
        )
    elif hashes["knowledge_graph_sha256"]:
        upload_input_sha = combine_sha256_digests(
            hashes["retrieval_bundle_sha256"],
            hashes["knowledge_graph_sha256"],
        )
    else:
        upload_input_sha = hashes["retrieval_bundle_sha256"]
    return {
        **hashes,
        "upload_input_sha256": upload_input_sha,
        "knowledge_graph_kind": graph_kind,
        "knowledge_graph_index_file": graph_index_file,
    }


def _indexing_build_identity(ctx: StageContext) -> tuple[Dict[str, Any], str]:
    snapshot = load_json_safe(ctx.work_dir / "resolved_config.json", {}) or {}
    build = dict(snapshot.get("indexing_build") or {}) if isinstance(snapshot, dict) else {}
    build_sha = (
        hashlib.sha256(
            json.dumps(build, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if build
        else ""
    )
    production = bool((ctx.config.get("pipeline") or {}).get("production_profile", False))
    commit = str(build.get("commit_sha") or "").strip().lower()
    if production and (
        not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit)
        or build.get("dirty") is not False
        or not isinstance(build.get("implementation_sha256"), dict)
        or not build.get("implementation_sha256")
    ):
        raise ValueError(
            "Production upload requires a clean, full-commit indexing_build identity "
            "in resolved_config.json"
        )
    return build, build_sha


def _database_records(
    records: Sequence[Mapping[str, Any]],
    vectors: Sequence[Sequence[float]],
    *,
    kind: str,
) -> List[Dict[str, Any]]:
    if len(records) != len(vectors):
        raise RuntimeError("Gemini embedding cardinality does not match the upload batch")
    output: List[Dict[str, Any]] = []
    for record, vector in zip(records, vectors):
        text = _record_embedding_text(dict(record))
        record_id = str(record.get("id") or "").strip()
        if not record_id or not text:
            raise ValueError(f"{kind} embedding records require non-empty id and text")
        metadata = _record_metadata(dict(record), kind=kind)
        # Pinecone's metadata adapter JSON-encodes containers. pgvector stores
        # native JSONB, so retain bounded arrays/objects for real filtering.
        for key in list(metadata):
            original = record.get(key)
            if isinstance(original, (list, dict)):
                encoded = json.dumps(original, ensure_ascii=False, separators=(",", ":"))
                if len(encoded.encode("utf-8")) <= 32_000:
                    metadata[key] = original
        output.append(
            {
                "record_id": record_id,
                "retrieval_text": text,
                "source_url": record.get("source_url") or record.get("canonical_url") or record.get("url"),
                "language": record.get("language") or record.get("language_code"),
                "content_sha256": record.get("content_sha256") or record.get("content_hash"),
                "metadata": metadata,
                "embedding": list(vector),
            }
        )
    return output


@register_stage
class GeminiPgVectorEmbedder(EmbedderStage):
    name = "gemini_pgvector"
    description = "Embeds the release-scoped retrieval lanes with Gemini and writes them to pgvector."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        embedder = config.get("embedder") if isinstance(config.get("embedder"), dict) else {}
        vector_store = config.get("vector_store") if isinstance(config.get("vector_store"), dict) else {}
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            errors.append("GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini embeddings")
        if not os.getenv(str(vector_store.get("ingest_dsn_env") or "PGVECTOR_INGEST_DSN")):
            errors.append("PGVECTOR_INGEST_DSN is required for pgvector indexing")
        if str(vector_store.get("provider") or "").strip().lower() != "pgvector":
            errors.append("vector_store.provider must be pgvector")
        if bool(embedder.get("enable_sparse", False)):
            errors.append("gemini_pgvector currently supports the selected dense-graph profile only")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.embedder_config
        selected_profile = (
            ctx.config.get("selected_profile")
            if isinstance(ctx.config.get("selected_profile"), Mapping)
            else {}
        )
        selected = bool(str(selected_profile.get("variant_id") or ""))
        if selected and tuple(selected_profile.get("record_kinds") or []) != SELECTED_DENSE_RECORD_KINDS:
            return StageResult.failure(
                "selected_profile.record_kinds differs from the evaluated dense-graph contract"
            )
        if bool(config.get("enable_sparse", False)):
            return StageResult.failure(
                "gemini_pgvector requires embedder.enable_sparse=false for the selected dense-graph profile"
            )
        paths = _resolve_indexing_input_paths(ctx)
        try:
            identity = _artifact_identity(ctx, paths)
            indexing_build, indexing_build_sha = _indexing_build_identity(ctx)
            lanes = _load_lane_records(
                paths,
                enable_dense_facts=bool(config.get("enable_dense_facts", False)),
                enable_dense_evidence_spans=bool(config.get("enable_dense_evidence_spans", True)),
                enable_dense_assertions=bool(config.get("enable_dense_assertions", True)),
                enable_dense_summaries=bool(config.get("enable_dense_summaries", True)),
                enable_dense_entities=bool(config.get("enable_dense_entities", True)),
                enable_dense_communities=bool(config.get("enable_dense_communities", True)),
                selected_profile=selected,
            )
            assembly = (
                _validate_selected_lanes(
                    paths=paths,
                    lanes=lanes,
                    variant_id=str(selected_profile.get("variant_id") or ""),
                )
                if selected
                else {}
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return StageResult.failure(str(exc))
        if not lanes["chunks"]:
            return StageResult.failure("No chunk embedding records found")

        model = str(config.get("model") or "gemini-embedding-2")
        dimensions = int(config.get("output_dimensionality") or 1536)
        vector_dimensions = int(ctx.vector_store_config.get("dimensions") or dimensions)
        if dimensions != vector_dimensions:
            return StageResult.failure(
                "embedder.output_dimensionality must match vector_store.dimensions"
            )
        namespaces = _resolve_upload_namespaces(config, run_id=ctx.run_id)
        totals = {lane: len(records) for lane, records in lanes.items()}
        expected_by_namespace = {
            namespaces[lane]: count for lane, count in totals.items()
        }
        contract_sha = production_indexing_contract_fingerprint(ctx.config)
        artifact_hashes = {
            key: value
            for key, value in identity.items()
            if key.endswith("_sha256") and value
        }
        progress_path = ctx.stage_work_dir / "index_upload_progress.json"
        text_batch_size = max(1, int(config.get("batch_size") or 32))
        media_text_batch_size = max(1, int(config.get("media_text_batch_size") or text_batch_size))
        media_multimodal_batch_size = max(1, int(config.get("media_multimodal_batch_size") or 4))
        allow_media_fallback = bool(config.get("media_multimodal_fallback_enabled", False))
        max_retries = max(1, int(config.get("max_retries") or 6))
        retry_base = float(config.get("retry_base_delay_sec") or 5.0)
        retry_max = float(config.get("retry_max_delay_sec") or 120.0)
        request_timeout = int(config.get("gemini_request_timeout_ms") or 120000)
        task_type = str(config.get("task_type_document") or "RETRIEVAL_DOCUMENT")
        media_metrics = {
            "media_multimodal_records": sum(
                1 for record in lanes["media"] if record.get("can_embed_multimodal")
            ),
            "media_text_only_records": sum(
                1 for record in lanes["media"] if not record.get("can_embed_multimodal")
            ),
            "media_multimodal_fallbacks": 0,
        }

        # A deterministic cohort order lets a failed run resume from exact
        # committed row counts without paying to re-embed completed batches.
        lanes["media"] = [
            *[record for record in lanes["media"] if record.get("can_embed_multimodal")],
            *[record for record in lanes["media"] if not record.get("can_embed_multimodal")],
        ]

        store: PgVectorStore | None = None
        try:
            store = PgVectorStore.from_config(ctx.config, purpose="write")
            store.check_schema()
            store.begin_release(
                release_id=ctx.run_id,
                project_name=ctx.project_name,
                embedding_model=model,
                contract_sha256=contract_sha,
                expected_counts=totals,
                artifact_hashes=artifact_hashes,
            )
            uploaded_by_namespace = store.namespace_counts(ctx.run_id)
            client = _make_gemini_client(request_timeout_ms=request_timeout)
            uploaded = {
                lane: int(uploaded_by_namespace.get(namespaces[lane], 0))
                for lane in lanes
            }
            for lane, records in lanes.items():
                already_uploaded = uploaded[lane]
                if already_uploaded > len(records):
                    raise RuntimeError(
                        f"pgvector contains more {lane} rows than the immutable upload plan"
                    )
                remaining = records[already_uploaded:]
                if not remaining:
                    continue
                if lane == "media":
                    groups = (
                        (
                            [record for record in remaining if record.get("can_embed_multimodal")],
                            media_multimodal_batch_size,
                            True,
                        ),
                        (
                            [record for record in remaining if not record.get("can_embed_multimodal")],
                            media_text_batch_size,
                            False,
                        ),
                    )
                else:
                    groups = ((remaining, text_batch_size, False),)

                for group, batch_size, multimodal in groups:
                    for batch in _iter_batches(group, batch_size):
                        batch_records = list(batch)
                        if multimodal:
                            try:
                                vectors = _call_with_retry(
                                    "embed_pgvector_media_multimodal_batch",
                                    lambda batch_records=batch_records: _embed_multimodal_batch(
                                        client,
                                        model=model,
                                        items=batch_records,
                                        task_type=task_type,
                                        output_dimensionality=dimensions,
                                    ),
                                    max_attempts=max_retries,
                                    base_delay_sec=retry_base,
                                    max_delay_sec=retry_max,
                                )
                            except Exception:
                                if not allow_media_fallback:
                                    raise
                                media_metrics["media_multimodal_fallbacks"] += len(batch_records)
                                vectors = _call_with_retry(
                                    "embed_pgvector_media_text_fallback_batch",
                                    lambda batch_records=batch_records: _embed_text_batch(
                                        client,
                                        model=model,
                                        texts=[_record_embedding_text(record) for record in batch_records],
                                        task_type=task_type,
                                        output_dimensionality=dimensions,
                                    ),
                                    max_attempts=max_retries,
                                    base_delay_sec=retry_base,
                                    max_delay_sec=retry_max,
                                )
                        else:
                            vectors = _call_with_retry(
                                f"embed_pgvector_{lane}_batch",
                                lambda batch_records=batch_records: _embed_text_batch(
                                    client,
                                    model=model,
                                    texts=[_record_embedding_text(record) for record in batch_records],
                                    task_type=task_type,
                                    output_dimensionality=dimensions,
                                ),
                                max_attempts=max_retries,
                                base_delay_sec=retry_base,
                                max_delay_sec=retry_max,
                            )
                        database_records = _database_records(
                            batch_records,
                            vectors,
                            kind=_LANE_KINDS[lane],
                        )
                        uploaded[lane] += store.upsert_records(
                            release_id=ctx.run_id,
                            lane=lane,
                            namespace=namespaces[lane],
                            records=database_records,
                        )
                        atomic_write_json(
                            progress_path,
                            {
                                "schema_version": 1,
                                "provider": "pgvector",
                                "release_id": ctx.run_id,
                                "phase": f"uploading_{lane}",
                                "model": model,
                                "output_dimensionality": dimensions,
                                "upload_input_sha256": identity["upload_input_sha256"],
                                "planned": totals,
                                "uploaded": uploaded,
                                "media_metrics": media_metrics,
                            },
                        )
                        logger.info(
                            "pgvector upload progress: %s %d/%d",
                            lane,
                            uploaded[lane],
                            totals[lane],
                        )

            verification = store.mark_ready(
                release_id=ctx.run_id,
                expected_by_namespace=expected_by_namespace,
            )
            health = store.health_check(
                release_id=ctx.run_id,
                expected_namespaces=expected_by_namespace,
                expected_model=model,
                expected_contract_sha256=contract_sha,
                expected_lane_counts=totals,
                expected_artifact_hashes=artifact_hashes,
            )
        except Exception as exc:
            logger.exception("Gemini pgvector upload failed")
            return StageResult.failure(str(exc))
        finally:
            if store is not None:
                store.close()

        index_name = (
            f"{str(ctx.vector_store_config.get('schema') or 'mbzuai_retrieval')}."
            f"{str(ctx.vector_store_config.get('records_table') or 'embedding_records')}"
        )
        manifest = {
            "schema_version": 6,
            "provider": "pgvector",
            "pgvector_schema_version": int(health["schema_version"]),
            "pgvector_extension_version": str(health["vector_version"]),
            "release_status": "ready",
            "activation_required": True,
            "production_indexing_contract_fingerprint": contract_sha,
            "indexing_build": indexing_build,
            "indexing_build_sha256": indexing_build_sha,
            "index_name": index_name,
            "sparse_index_name": "",
            "model": model,
            "output_dimensionality": dimensions,
            "namespaces": namespaces,
            "namespace_strategy": str(config.get("namespace_strategy") or "static").strip().lower(),
            "namespace_release_id": (
                ctx.run_id
                if str(config.get("namespace_strategy") or "static").strip().lower() == "release"
                else ""
            ),
            "planned": totals,
            "uploaded": totals,
            "bundle_version": int(
                (load_json_safe(paths["bundle"], {}) or {}).get("version") or 0
            ),
            "evidence_span_count": totals["evidence_spans"],
            "sparse_evidence_span_count": 0,
            "retrieval_bundle_file": paths["bundle"],
            "lexical_corpus_file": paths.get("lexical_corpus") or "",
            "promoted_assertions_file": paths.get("promoted_assertions") or "",
            "knowledge_graph_file": paths.get("graph_bundle") or "",
            "knowledge_graph_index_file": identity["knowledge_graph_index_file"],
            "selected_release_assembly_file": paths.get("release_assembly") or "",
            "page_graph_navigation_catalog_file": paths.get("navigation_catalog") or "",
            "selected_profile": {
                "variant_id": str(selected_profile.get("variant_id") or ""),
                "record_kinds": list(selected_profile.get("record_kinds") or []),
                "assembly_sha256": str(assembly.get("assembly_sha256") or ""),
            }
            if selected
            else {},
            **identity,
            "record_fingerprints": {
                "entities": _record_text_fingerprint(lanes["entities"]),
                "communities": _record_text_fingerprint(lanes["communities"]),
            },
            "retrieval_bundle_stats": totals,
            "media_metrics": media_metrics,
            "sparse": {"enabled": False, "index_name": "", "model": "", "record_stats": {}},
            "verification": {
                "dense": {
                    "expected": verification["expected"],
                    "actual": verification["actual"],
                    "failures": [],
                }
            },
        }
        manifest_path = ctx.stage_work_dir / "index_upload_manifest.json"
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            progress_path,
            {
                "schema_version": 1,
                "provider": "pgvector",
                "release_id": ctx.run_id,
                "phase": "completed",
                "model": model,
                "output_dimensionality": dimensions,
                "upload_input_sha256": identity["upload_input_sha256"],
                "planned": totals,
                "uploaded": totals,
                "media_metrics": media_metrics,
            },
        )
        vectors_uploaded = sum(totals.values())
        return StageResult.success(
            outputs={
                "vectors_uploaded": vectors_uploaded,
                "index_name": index_name,
                "sparse_index_name": "",
                "index_manifest_file": str(manifest_path),
                "release_status": "ready",
            },
            metrics={
                "vectors_uploaded": vectors_uploaded,
                **{
                    f"{_LANE_KINDS[lane]}_vectors_uploaded": count
                    for lane, count in totals.items()
                },
                **media_metrics,
                "engine": "gemini",
                "model": model,
                "vector_store": "pgvector",
            },
            artifacts=[
                ctx.make_artifact(
                    manifest_path,
                    artifact_type="index_manifest",
                    role="vector_index_upload",
                    metadata={
                        "provider": "pgvector",
                        "index_name": index_name,
                        "release_id": ctx.run_id,
                        "release_status": "ready",
                        "vectors_uploaded": vectors_uploaded,
                    },
                )
            ],
        )
