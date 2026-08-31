"""Fail-closed Pinecone upload for the evaluated Representation V2 release."""

from __future__ import annotations

import logging
from typing import Any, Dict, Mapping

from pipeline.core.base import StageContext, StageResult
from pipeline.core.config import production_indexing_contract_fingerprint
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.release_assembly import SELECTED_DENSE_RECORD_KINDS
from pipeline.stages.embedders.gemini_pgvector_embedder import (
    _LANE_KINDS,
    _apply_selected_media_input_contract,
    _artifact_identity,
    _indexing_build_identity,
    _load_lane_records,
    _validate_selected_lanes,
)
from pipeline.stages.embedders.gemini_pinecone_embedder import (
    _assert_upload_plan_complete,
    _call_with_retry,
    _clear_namespace,
    _embed_text_batch,
    _ensure_index,
    _iter_batches,
    _make_gemini_client,
    _make_index_handle,
    _make_pinecone_client,
    _record_embedding_text,
    _record_text_fingerprint,
    _resolve_upload_namespaces,
    _upsert_namespace,
    _verify_namespace_counts,
    _write_progress,
)


logger = logging.getLogger(__name__)


def _zero_progress(totals: Mapping[str, int]) -> Dict[str, int]:
    return {lane: 0 for lane in totals}


def _load_matching_progress(
    path: Any,
    *,
    index_name: str,
    model: str,
    dimensions: int,
    totals: Mapping[str, int],
    upload_input_sha256: str,
) -> tuple[Dict[str, int], bool]:
    payload = load_json_safe(path, {}) or {}
    matches = bool(
        isinstance(payload, Mapping)
        and payload.get("index_name") == index_name
        and payload.get("model") == model
        and int(payload.get("output_dimensionality") or 0) == dimensions
        and dict(payload.get("totals") or {}) == dict(totals)
        and str(payload.get("upload_input_sha256") or "") == upload_input_sha256
    )
    if not matches:
        return _zero_progress(totals), False
    uploaded = payload.get("uploaded") if isinstance(payload.get("uploaded"), Mapping) else {}
    return (
        {
            lane: max(0, min(int(uploaded.get(lane) or 0), int(total)))
            for lane, total in totals.items()
        },
        True,
    )


def execute_selected_profile_pinecone(
    ctx: StageContext,
    config: Dict[str, Any],
    paths: Mapping[str, str],
) -> StageResult:
    """Embed and upload all six selected record kinds to release namespaces.

    This path intentionally shares the selected-release validation helpers with
    pgvector. Switching stores therefore cannot change embedding text, omit
    Page Cards/actions, or admit an unevaluated lane.
    """

    selected_profile = (
        ctx.config.get("selected_profile")
        if isinstance(ctx.config.get("selected_profile"), Mapping)
        else {}
    )
    variant_id = str(selected_profile.get("variant_id") or "").strip()
    if not variant_id:
        return StageResult.failure("Selected Pinecone upload requires selected_profile.variant_id")
    if tuple(selected_profile.get("record_kinds") or []) != SELECTED_DENSE_RECORD_KINDS:
        return StageResult.failure(
            "selected_profile.record_kinds differs from the evaluated dense-graph contract"
        )
    if bool(config.get("enable_sparse", False)):
        return StageResult.failure("Selected Pinecone upload requires embedder.enable_sparse=false")
    namespace_strategy = str(config.get("namespace_strategy") or "static").strip().lower()
    if namespace_strategy != "release":
        return StageResult.failure(
            "Selected Pinecone upload requires embedder.namespace_strategy=release"
        )

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
            selected_profile=True,
        )
        assembly = _validate_selected_lanes(
            paths=paths,
            lanes=lanes,
            variant_id=variant_id,
            embedder_config=config,
        )
        media_input = _apply_selected_media_input_contract(lanes["media"], assembly)
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
    index_name = str(config.get("pinecone_index") or "").strip()
    if not index_name:
        return StageResult.failure("embedder.pinecone_index is required")

    namespaces = _resolve_upload_namespaces(config, run_id=ctx.run_id)
    totals = {lane: len(records) for lane, records in lanes.items()}
    expected_by_namespace = {
        namespaces[lane]: count for lane, count in totals.items() if count > 0
    }
    progress_path = ctx.stage_work_dir / "index_upload_progress.json"
    uploaded, resumable = _load_matching_progress(
        progress_path,
        index_name=index_name,
        model=model,
        dimensions=dimensions,
        totals=totals,
        upload_input_sha256=identity["upload_input_sha256"],
    )
    media_metrics = {
        "media_multimodal_records": 0,
        "media_text_only_records": totals["media"],
        "media_multimodal_fallbacks": 0,
    }
    max_retries = max(1, int(config.get("max_retries") or 6))
    retry_base = float(config.get("retry_base_delay_sec") or 5.0)
    retry_max = float(config.get("retry_max_delay_sec") or 120.0)
    text_batch_size = max(1, int(config.get("batch_size") or 32))
    upsert_batch_size = max(1, int(config.get("upsert_batch_size") or 100))
    request_timeout = (
        float(config.get("pinecone_connect_timeout_sec") or 10.0),
        float(config.get("pinecone_read_timeout_sec") or 120.0),
    )
    task_type = str(config.get("task_type_document") or "RETRIEVAL_DOCUMENT")

    try:
        pc = _make_pinecone_client()
        index_existed = True
        if bool(config.get("create_index_if_missing", True)):
            index_existed = _call_with_retry(
                "ensure_selected_pinecone_index",
                lambda: _ensure_index(
                    pc,
                    index_name=index_name,
                    dimension=dimensions,
                    cloud=str(config.get("pinecone_cloud") or "aws"),
                    region=str(config.get("pinecone_region") or "us-east-1"),
                    metric=str(config.get("metric") or "cosine"),
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base,
                max_delay_sec=retry_max,
            )
        index = _make_index_handle(index_name)
        if not index_existed or not resumable:
            uploaded = _zero_progress(totals)
            for lane, namespace in namespaces.items():
                _call_with_retry(
                    f"clear_selected_pinecone_namespace_{lane}",
                    lambda namespace=namespace: _clear_namespace(
                        index,
                        namespace=namespace,
                        request_timeout=request_timeout,
                    ),
                    max_attempts=max_retries,
                    base_delay_sec=retry_base,
                    max_delay_sec=retry_max,
                )

        gemini = _make_gemini_client(
            request_timeout_ms=int(config.get("gemini_request_timeout_ms") or 120000)
        )
        for lane, records in lanes.items():
            remaining = records[uploaded[lane] :]
            for batch in _iter_batches(remaining, text_batch_size):
                batch_records = list(batch)
                vectors = _call_with_retry(
                    f"embed_selected_pinecone_{lane}_batch",
                    lambda batch_records=batch_records: _embed_text_batch(
                        gemini,
                        model=model,
                        texts=[_record_embedding_text(record) for record in batch_records],
                        task_type=task_type,
                        output_dimensionality=dimensions,
                    ),
                    max_attempts=max_retries,
                    base_delay_sec=retry_base,
                    max_delay_sec=retry_max,
                )
                uploaded[lane] += _call_with_retry(
                    f"upsert_selected_pinecone_{lane}_batch",
                    lambda batch_records=batch_records, vectors=vectors, lane=lane: _upsert_namespace(
                        index,
                        namespace=namespaces[lane],
                        records=batch_records,
                        vectors=vectors,
                        kind=_LANE_KINDS[lane],
                        batch_size=upsert_batch_size,
                        request_timeout=request_timeout,
                    ),
                    max_attempts=max_retries,
                    base_delay_sec=retry_base,
                    max_delay_sec=retry_max,
                )
                _write_progress(
                    progress_path,
                    index_name=index_name,
                    model=model,
                    output_dimensionality=dimensions,
                    totals=totals,
                    uploaded=uploaded,
                    phase=f"uploading_{lane}",
                    retrieval_bundle_file=paths["bundle"],
                    retrieval_bundle_sha256=identity["retrieval_bundle_sha256"],
                    knowledge_graph_sha256=identity["knowledge_graph_sha256"],
                    upload_input_sha256=identity["upload_input_sha256"],
                    retrieval_bundle_stats=totals,
                    media_metrics=media_metrics,
                )
                logger.info(
                    "Selected Pinecone upload progress: %s %d/%d",
                    lane,
                    uploaded[lane],
                    totals[lane],
                )

        _assert_upload_plan_complete(totals, uploaded)

        def verify_counts() -> Dict[str, Any]:
            report = _verify_namespace_counts(
                index=index,
                expected=expected_by_namespace,
                min_count_only=False,
            )
            if report.get("failures"):
                raise RuntimeError(
                    "Selected Pinecone namespace verification is pending: "
                    f"{report['failures']}"
                )
            return report

        verification = _call_with_retry(
            "verify_selected_pinecone_index_counts",
            verify_counts,
            max_attempts=max_retries,
            base_delay_sec=retry_base,
            max_delay_sec=retry_max,
        )
    except Exception as exc:
        logger.exception("Selected Gemini Pinecone upload failed")
        return StageResult.failure(str(exc))

    contract_sha = production_indexing_contract_fingerprint(ctx.config)
    manifest = {
        "schema_version": 6,
        "provider": "pinecone",
        "release_status": "ready",
        "activation_required": True,
        "production_indexing_contract_fingerprint": contract_sha,
        "indexing_build": indexing_build,
        "indexing_build_sha256": indexing_build_sha,
        "index_name": index_name,
        "sparse_index_name": "",
        "model": model,
        "output_dimensionality": dimensions,
        "media_input": media_input,
        "namespaces": namespaces,
        "namespace_strategy": namespace_strategy,
        "namespace_release_id": ctx.run_id,
        "planned": totals,
        "uploaded": totals,
        "bundle_version": int((load_json_safe(paths["bundle"], {}) or {}).get("version") or 0),
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
            "variant_id": variant_id,
            "record_kinds": list(selected_profile.get("record_kinds") or []),
            "assembly_sha256": str(assembly.get("assembly_sha256") or ""),
            "embedding_spec": dict(assembly.get("embedding_spec") or {}),
            "media_input": media_input,
        },
        **identity,
        "record_fingerprints": {
            lane: _record_text_fingerprint(records) for lane, records in lanes.items()
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
    _write_progress(
        progress_path,
        index_name=index_name,
        model=model,
        output_dimensionality=dimensions,
        totals=totals,
        uploaded=totals,
        phase="completed",
        retrieval_bundle_file=paths["bundle"],
        retrieval_bundle_sha256=identity["retrieval_bundle_sha256"],
        knowledge_graph_sha256=identity["knowledge_graph_sha256"],
        upload_input_sha256=identity["upload_input_sha256"],
        retrieval_bundle_stats=totals,
        media_metrics=media_metrics,
        record_fingerprints=manifest["record_fingerprints"],
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
            "media_input": media_input,
            "engine": "gemini",
            "model": model,
            "vector_store": "pinecone",
        },
        artifacts=[
            ctx.make_artifact(
                manifest_path,
                artifact_type="index_manifest",
                role="vector_index_upload",
                metadata={
                    "provider": "pinecone",
                    "index_name": index_name,
                    "release_id": ctx.run_id,
                    "release_status": "ready",
                    "vectors_uploaded": vectors_uploaded,
                },
            )
        ],
    )
