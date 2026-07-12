"""
MBZUAI legacy Pinecone uploader.

Uploads the current chatbot-compatible two-index contract:
- summary index: metadata/query-oriented records
- text index: chunk/detail records

Each index receives dense vectors and optional BM25 sparse values in the same
Pinecone record shape expected by LangChain's PineconeHybridSearchRetriever.
"""

from __future__ import annotations

import json
import logging
import os
import time
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline.core.base import EmbedderStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage
from pipeline.stages.embedders.openai_embedder import (
    _generate_openai_embeddings,
    _generate_st_embeddings,
    _replace_md_links,
    _serialize_metadata_value,
)

logger = logging.getLogger(__name__)

PINECONE_METADATA_LIMIT_BYTES = 40960
PINECONE_METADATA_TARGET_BYTES = 35000
PINECONE_UPSERT_REQUEST_LIMIT_BYTES = 2_000_000
PINECONE_UPSERT_REQUEST_TARGET_BYTES = 1_750_000

_OPTIONAL_METADATA_DROP_ORDER = (
    "page_metadata",
    "media",
    "images",
    "videos",
    "locale_variant_urls",
    "citation_anchor",
    "entities",
)

_VERBOSE_METADATA_SHRINK_ORDER = (
    "document_summary",
    "key_facts",
    "keywords",
    "context",
)


def _metadata_size_bytes(metadata: Dict[str, Any]) -> int:
    return len(json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _json_size_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _split_upsert_batches(
    vectors: List[Dict[str, Any]],
    *,
    max_payload_bytes: int = PINECONE_UPSERT_REQUEST_TARGET_BYTES,
) -> List[List[Dict[str, Any]]]:
    max_payload_bytes = min(
        max(50_000, int(max_payload_bytes)),
        PINECONE_UPSERT_REQUEST_LIMIT_BYTES,
    )
    batches: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_size = _json_size_bytes({"vectors": []})

    for vector in vectors:
        vector_size = _json_size_bytes(vector) + 2
        if current and current_size + vector_size > max_payload_bytes:
            batches.append(current)
            current = []
            current_size = _json_size_bytes({"vectors": []})
        current.append(vector)
        current_size += vector_size

    if current:
        batches.append(current)
    return batches


def _truncate_utf8(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    suffix = "..."
    suffix_bytes = suffix.encode("utf-8")
    payload_budget = max(0, max_bytes - len(suffix_bytes))
    return encoded[:payload_budget].decode("utf-8", errors="ignore").rstrip() + suffix


def _shrink_string_field_to_fit(metadata: Dict[str, Any], key: str, *, max_bytes: int) -> None:
    value = metadata.get(key)
    if not isinstance(value, str):
        return
    for _ in range(8):
        current_size = _metadata_size_bytes(metadata)
        if current_size <= max_bytes:
            return
        current_bytes = len(value.encode("utf-8"))
        if current_bytes <= 256:
            return
        overflow = current_size - max_bytes
        next_budget = max(256, current_bytes - overflow - 512)
        if next_budget >= current_bytes:
            next_budget = max(256, current_bytes // 2)
        value = _truncate_utf8(value, next_budget)
        metadata[key] = value


def _serialize_legacy_metadata(
    raw_metadata: Dict[str, Any],
    *,
    max_bytes: int = PINECONE_METADATA_TARGET_BYTES,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Serialize metadata while enforcing Pinecone's per-vector metadata cap.

    The legacy chatbot depends on flat fields such as context, source URLs,
    authority metadata, summaries, and key facts. Rich crawler metadata is useful
    for audits but can exceed Pinecone's 40 KB limit, so this boundary keeps the
    answer-critical fields and removes/trims lower-value auxiliary fields only
    when a vector would otherwise be rejected.
    """

    metadata = {
        key: _serialize_metadata_value(key, value)
        for key, value in raw_metadata.items()
        if value is not None
    }
    original_bytes = _metadata_size_bytes(metadata)
    removed_fields: List[str] = []
    truncated_fields: List[str] = []

    if original_bytes > max_bytes:
        for key in _OPTIONAL_METADATA_DROP_ORDER:
            if _metadata_size_bytes(metadata) <= max_bytes:
                break
            if key in metadata:
                metadata.pop(key, None)
                removed_fields.append(key)

    if _metadata_size_bytes(metadata) > max_bytes:
        for key in _VERBOSE_METADATA_SHRINK_ORDER:
            before = metadata.get(key)
            _shrink_string_field_to_fit(metadata, key, max_bytes=max_bytes)
            if metadata.get(key) != before:
                truncated_fields.append(key)
            if _metadata_size_bytes(metadata) <= max_bytes:
                break

    if _metadata_size_bytes(metadata) > PINECONE_METADATA_LIMIT_BYTES:
        for key in reversed(_VERBOSE_METADATA_SHRINK_ORDER):
            before = metadata.get(key)
            _shrink_string_field_to_fit(metadata, key, max_bytes=PINECONE_METADATA_LIMIT_BYTES - 512)
            if metadata.get(key) != before and key not in truncated_fields:
                truncated_fields.append(key)
            if _metadata_size_bytes(metadata) <= PINECONE_METADATA_LIMIT_BYTES - 512:
                break

    final_bytes = _metadata_size_bytes(metadata)
    if final_bytes > PINECONE_METADATA_LIMIT_BYTES:
        raise RuntimeError(
            f"Serialized Pinecone metadata is {final_bytes} bytes after compaction; "
            f"limit is {PINECONE_METADATA_LIMIT_BYTES} bytes"
        )

    return metadata, {
        "original_bytes": original_bytes,
        "final_bytes": final_bytes,
        "compacted": final_bytes != original_bytes,
        "removed_fields": removed_fields,
        "truncated_fields": truncated_fields,
    }


def _resolve_optional_path(path_value: Any, *, base_dir: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        candidate = (base_dir / path).resolve()
        if candidate.exists() or candidate.parent.exists():
            return candidate
        path = Path.cwd() / path
    return path.resolve()


def _index_names(response: Any) -> set[str]:
    if response is None:
        return set()
    if isinstance(response, dict):
        indexes = response.get("indexes") or response.get("index_list") or []
        names = set()
        for item in indexes:
            if isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
            elif isinstance(item, str):
                names.add(item)
        return names
    names = set()
    try:
        for item in response:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
            elif getattr(item, "name", None):
                names.add(str(item.name))
    except TypeError:
        indexes = getattr(response, "indexes", None) or []
        for item in indexes:
            if getattr(item, "name", None):
                names.add(str(item.name))
            elif isinstance(item, dict) and item.get("name"):
                names.add(str(item["name"]))
    return names


def _index_attr(description: Any, key: str) -> Any:
    if description is None:
        return None
    if isinstance(description, dict):
        return description.get(key)
    return getattr(description, key, None)


def _texts_digest(texts: List[str]) -> str:
    payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":"))
    return sha256(payload.encode("utf-8")).hexdigest()


def _ensure_index(
    pc: Any,
    *,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
    create_if_missing: bool,
    wait_timeout_sec: int,
) -> None:
    existing = _index_names(pc.list_indexes())
    if index_name in existing:
        try:
            description = pc.describe_index(index_name)
        except Exception as exc:  # pragma: no cover - network/client dependent
            logger.warning("Could not describe existing Pinecone index %s: %s", index_name, exc)
            return
        existing_dimension = _index_attr(description, "dimension")
        existing_metric = str(_index_attr(description, "metric") or "").lower()
        if existing_dimension and int(existing_dimension) != int(dimension):
            raise RuntimeError(
                f"Pinecone index {index_name!r} has dimension {existing_dimension}, expected {dimension}"
            )
        if existing_metric and existing_metric != metric.lower():
            raise RuntimeError(
                f"Pinecone index {index_name!r} uses metric {existing_metric!r}, expected {metric!r}. "
                "Create a new index or rebuild the existing index with the correct metric."
            )
        return
    if not create_if_missing:
        raise RuntimeError(f"Pinecone index {index_name!r} does not exist")

    try:
        from pinecone import ServerlessSpec
    except Exception as exc:  # pragma: no cover - depends on installed Pinecone client
        raise RuntimeError("Pinecone ServerlessSpec is unavailable; create the indexes manually or upgrade pinecone") from exc

    logger.info("Creating Pinecone index %s (dimension=%d, metric=%s)", index_name, dimension, metric)
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    deadline = time.time() + max(1, wait_timeout_sec)
    while time.time() < deadline:
        if index_name in _index_names(pc.list_indexes()):
            return
        time.sleep(2)
    raise RuntimeError(f"Timed out waiting for Pinecone index {index_name!r} to become available")


def _generate_dense_embeddings(
    texts: List[str],
    *,
    engine: str,
    model: str,
    batch_size: int,
    output_dimensionality: int | None = None,
    cache_path: Path | None = None,
    max_retries: int = 5,
    retry_base_delay_sec: float = 5.0,
    retry_max_delay_sec: float = 120.0,
    batch_delay_sec: float = 0.0,
) -> List[List[float]]:
    if engine == "openai":
        return _generate_openai_embeddings(texts, model, batch_size)
    if engine in {"sentence_transformer", "sentence-transformer", "huggingface"}:
        return _generate_st_embeddings(texts, model, batch_size)
    if engine == "gemini":
        from pipeline.stages.embedders.gemini_pinecone_embedder import (
            _call_with_retry,
            _embed_text_batch,
            _make_gemini_client,
        )

        client = _make_gemini_client()
        embeddings: List[List[float]] = []
        text_digest = _texts_digest(texts)
        if cache_path and cache_path.exists():
            payload = load_json_safe(cache_path, {}) or {}
            cached_embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
            if (
                payload.get("schema_version") == 1
                and payload.get("engine") == engine
                and payload.get("model") == model
                and payload.get("output_dimensionality") == output_dimensionality
                and payload.get("text_count") == len(texts)
                and payload.get("text_digest") == text_digest
                and isinstance(cached_embeddings, list)
                and len(cached_embeddings) <= len(texts)
            ):
                embeddings = [list(vector) for vector in cached_embeddings if isinstance(vector, list)]
                logger.info("Loaded %d cached Gemini legacy embeddings from %s", len(embeddings), cache_path)
            elif payload.get("text_digest") != text_digest:
                logger.info("Ignoring stale Gemini legacy embedding cache at %s because source text changed", cache_path)

        for start in range(len(embeddings), len(texts), batch_size):
            batch = texts[start : start + batch_size]
            batch_embeddings = _call_with_retry(
                "embed_legacy_gemini_batch",
                lambda batch=batch: _embed_text_batch(
                    client,
                    model=model,
                    texts=batch,
                    task_type="RETRIEVAL_DOCUMENT",
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    f"Gemini embedding batch returned {len(batch_embeddings)} vectors for {len(batch)} texts"
                )
            embeddings.extend(batch_embeddings)
            if cache_path:
                atomic_write_json(
                    cache_path,
                    {
                        "schema_version": 1,
                        "engine": engine,
                        "model": model,
                        "output_dimensionality": output_dimensionality,
                        "text_count": len(texts),
                        "text_digest": text_digest,
                        "embeddings": embeddings,
                    },
                )
            logger.info("Gemini embedded legacy batch %d-%d/%d", start, start + len(batch), len(texts))
            if batch_delay_sec > 0 and start + len(batch) < len(texts):
                time.sleep(batch_delay_sec)
        return embeddings
    raise ValueError(f"Unsupported legacy MBZUAI embedding engine: {engine}")


def _fit_or_load_bm25(
    *,
    texts: List[str],
    configured_model_path: Path | None,
    output_path: Path,
    refit: bool,
) -> Tuple[Any, Path]:
    from pinecone_text.sparse import BM25Encoder

    if configured_model_path and configured_model_path.exists() and not refit:
        logger.info("Loading existing BM25 encoder from %s", configured_model_path)
        return BM25Encoder().load(str(configured_model_path)), configured_model_path

    logger.info("Fitting BM25 encoder on %d legacy vector-store records", len(texts))
    bm25 = BM25Encoder()
    bm25.fit(texts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bm25.dump(str(output_path))
    return bm25, output_path


def _encode_sparse_documents(bm25: Any, texts: List[str]) -> List[Dict[str, Any]]:
    sparse_values: List[Dict[str, Any]] = []
    for index in range(0, len(texts), 100):
        sparse_values.extend(bm25.encode_documents(texts[index : index + 100]))
    return sparse_values


def _upload_records(
    *,
    index: Any,
    docs: List[Dict[str, Any]],
    dense_vectors: List[List[float]],
    sparse_vectors: List[Dict[str, Any]] | None,
    namespace: str,
    upsert_batch_size: int,
    progress_path: Path | None = None,
    progress_phase: str = "",
    progress_totals: Dict[str, int] | None = None,
    progress_uploaded: Dict[str, int] | None = None,
    metadata_max_bytes: int = PINECONE_METADATA_TARGET_BYTES,
    upsert_payload_max_bytes: int = PINECONE_UPSERT_REQUEST_TARGET_BYTES,
) -> Tuple[int, Dict[str, Any]]:
    uploaded = 0
    metadata_stats: Dict[str, Any] = {
        "metadata_original_bytes_max": 0,
        "metadata_uploaded_bytes_max": 0,
        "metadata_compacted_vectors": 0,
        "metadata_removed_fields": {},
        "metadata_truncated_fields": {},
        "upsert_requests": 0,
        "upsert_request_bytes_max": 0,
    }
    if progress_totals is None:
        progress_totals = {}
    if progress_uploaded is None:
        progress_uploaded = {}
    for start in range(0, len(docs), upsert_batch_size):
        batch_docs = docs[start : start + upsert_batch_size]
        batch_dense = dense_vectors[start : start + upsert_batch_size]
        batch_sparse = sparse_vectors[start : start + upsert_batch_size] if sparse_vectors else [None] * len(batch_docs)
        vectors = []
        for doc, dense, sparse in zip(batch_docs, batch_dense, batch_sparse):
            metadata, compact_stats = _serialize_legacy_metadata(
                doc.get("metadata") or {},
                max_bytes=metadata_max_bytes,
            )
            metadata_stats["metadata_original_bytes_max"] = max(
                int(metadata_stats["metadata_original_bytes_max"]),
                int(compact_stats["original_bytes"]),
            )
            metadata_stats["metadata_uploaded_bytes_max"] = max(
                int(metadata_stats["metadata_uploaded_bytes_max"]),
                int(compact_stats["final_bytes"]),
            )
            if compact_stats["compacted"]:
                metadata_stats["metadata_compacted_vectors"] += 1
                for key in compact_stats["removed_fields"]:
                    removed = metadata_stats["metadata_removed_fields"]
                    removed[key] = int(removed.get(key, 0)) + 1
                for key in compact_stats["truncated_fields"]:
                    truncated = metadata_stats["metadata_truncated_fields"]
                    truncated[key] = int(truncated.get(key, 0)) + 1
            vector = {
                "id": str(doc["id"]),
                "values": dense,
                "metadata": metadata,
            }
            if sparse:
                vector["sparse_values"] = sparse
            vectors.append(vector)
        for payload_vectors in _split_upsert_batches(vectors, max_payload_bytes=upsert_payload_max_bytes):
            request_bytes = _json_size_bytes({"vectors": payload_vectors, "namespace": namespace})
            metadata_stats["upsert_requests"] += 1
            metadata_stats["upsert_request_bytes_max"] = max(
                int(metadata_stats["upsert_request_bytes_max"]),
                request_bytes,
            )
            index.upsert(vectors=payload_vectors, namespace=namespace)
            uploaded += len(payload_vectors)
            if progress_path and progress_phase:
                progress_uploaded[progress_phase] = uploaded
                atomic_write_json(
                    progress_path,
                    {
                        "kind": "upload_legacy_vectorstores",
                        "phase": f"uploading_{progress_phase}",
                        "uploaded": progress_uploaded,
                        "totals": progress_totals,
                    },
                )
            logger.info(
                "Uploaded legacy vectors %d/%d",
                uploaded,
                len(docs),
            )
    return uploaded, metadata_stats


@register_stage
class MBZLegacyPineconeEmbedder(EmbedderStage):
    name = "mbzuai_legacy_pinecone"
    description = "Uploads chatbot-compatible MBZUAI summary/text hybrid Pinecone indexes."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors: List[str] = []
        emb_cfg = config.get("embedder", {})
        if not os.getenv("PINECONE_API_KEY"):
            errors.append("PINECONE_API_KEY is required")
        engine = emb_cfg.get("engine", "gemini")
        if engine == "openai" and not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY is required for OpenAI embeddings")
        if engine == "gemini" and not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            errors.append("GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini embeddings")
        if not (emb_cfg.get("pinecone_summary_index") or emb_cfg.get("summary_index")):
            errors.append("embedder.pinecone_summary_index is required")
        if not (emb_cfg.get("pinecone_text_index") or emb_cfg.get("text_index") or emb_cfg.get("pinecone_index")):
            errors.append("embedder.pinecone_text_index is required")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.embedder_config
        summary_file = ctx.previous_outputs.get("legacy_summary_formatted_file")
        text_file = ctx.previous_outputs.get("legacy_text_formatted_file")
        if not summary_file or not text_file:
            return StageResult.failure("Legacy summary/text formatted payloads are required before upload")

        summary_docs = load_json_safe(summary_file, []) or []
        text_docs = load_json_safe(text_file, []) or []
        if not summary_docs or not text_docs:
            return StageResult.failure("Legacy summary/text payloads are empty")

        summary_index_name = str(config.get("pinecone_summary_index") or config.get("summary_index") or "").strip()
        text_index_name = str(config.get("pinecone_text_index") or config.get("text_index") or config.get("pinecone_index") or "").strip()
        if not summary_index_name or not text_index_name:
            return StageResult.failure("Both summary and text Pinecone index names are required")

        engine = str(config.get("engine") or "gemini")
        model = str(config.get("model") or "gemini-embedding-2")
        batch_size = int(config.get("batch_size") or 32)
        upsert_batch_size = int(config.get("upsert_batch_size") or 100)
        namespace = str(config.get("namespace") or "")
        use_sparse = bool(config.get("use_sparse_embeddings", True))
        max_retries = max(1, int(config.get("max_retries") or config.get("retry_attempts") or 8))
        retry_base_delay_sec = max(0.1, float(config.get("retry_base_delay_sec") or 10.0))
        retry_max_delay_sec = max(retry_base_delay_sec, float(config.get("retry_max_delay_sec") or 180.0))
        batch_delay_sec = max(0.0, float(config.get("embedding_batch_delay_sec") or config.get("request_delay_sec") or 0.0))
        metadata_max_bytes = int(config.get("metadata_max_bytes") or PINECONE_METADATA_TARGET_BYTES)
        metadata_max_bytes = min(max(1024, metadata_max_bytes), PINECONE_METADATA_LIMIT_BYTES)
        upsert_payload_max_bytes = int(config.get("upsert_payload_max_bytes") or PINECONE_UPSERT_REQUEST_TARGET_BYTES)
        upsert_payload_max_bytes = min(
            max(50_000, upsert_payload_max_bytes),
            PINECONE_UPSERT_REQUEST_LIMIT_BYTES,
        )
        output_dimensionality = config.get("output_dimensionality")
        if output_dimensionality not in (None, ""):
            try:
                output_dimensionality = int(output_dimensionality)
            except (TypeError, ValueError):
                return StageResult.failure("embedder.output_dimensionality must be an integer when provided")
        else:
            output_dimensionality = None

        summary_texts = [_replace_md_links(str(doc.get("text") or "")) for doc in summary_docs]
        text_texts = [_replace_md_links(str(doc.get("text") or "")) for doc in text_docs]
        all_texts = [*summary_texts, *text_texts]

        logger.info(
            "Generating legacy MBZUAI dense embeddings: summary=%d text=%d engine=%s model=%s",
            len(summary_docs),
            len(text_docs),
            engine,
            model,
        )
        dense_all = _generate_dense_embeddings(
            all_texts,
            engine=engine,
            model=model,
            batch_size=batch_size,
            output_dimensionality=output_dimensionality,
            cache_path=ctx.stage_work_dir / "legacy_dense_embedding_cache.json",
            max_retries=max_retries,
            retry_base_delay_sec=retry_base_delay_sec,
            retry_max_delay_sec=retry_max_delay_sec,
            batch_delay_sec=batch_delay_sec,
        )
        if not dense_all:
            return StageResult.failure("Dense embedding generation returned no vectors")
        if len(dense_all) != len(all_texts):
            return StageResult.failure(
                f"Dense embedding count mismatch: expected {len(all_texts)}, got {len(dense_all)}"
            )

        dimension = len(dense_all[0])
        summary_dense = dense_all[: len(summary_docs)]
        text_dense = dense_all[len(summary_docs) :]

        summary_sparse = None
        text_sparse = None
        bm25_model_file = ""
        if use_sparse:
            configured_bm25_path = _resolve_optional_path(config.get("bm25_model_path"), base_dir=ctx.work_dir)
            output_path = _resolve_optional_path(config.get("bm25_output_path"), base_dir=ctx.stage_work_dir)
            if output_path is None:
                output_path = ctx.stage_work_dir / "MBZUAI_BM25_ENCODER.json"
            bm25, resolved_bm25_path = _fit_or_load_bm25(
                texts=all_texts,
                configured_model_path=configured_bm25_path,
                output_path=output_path,
                refit=bool(config.get("refit_bm25", True)),
            )
            sparse_all = _encode_sparse_documents(bm25, all_texts)
            if len(sparse_all) != len(all_texts):
                return StageResult.failure(
                    f"Sparse embedding count mismatch: expected {len(all_texts)}, got {len(sparse_all)}"
                )
            summary_sparse = sparse_all[: len(summary_docs)]
            text_sparse = sparse_all[len(summary_docs) :]
            bm25_model_file = str(resolved_bm25_path)

        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        create_if_missing = bool(config.get("create_index_if_missing", False))
        metric = str(config.get("metric") or ("dotproduct" if use_sparse else "cosine")).lower()
        if use_sparse and metric != "dotproduct":
            return StageResult.failure(
                "Hybrid dense+sparse legacy uploads require embedder.metric='dotproduct'. "
                f"Received {metric!r}."
            )
        cloud = str(config.get("pinecone_cloud") or "aws")
        region = str(config.get("pinecone_region") or "us-east-1")
        wait_timeout_sec = int(config.get("create_index_wait_timeout_sec") or 180)

        for index_name in (summary_index_name, text_index_name):
            _ensure_index(
                pc,
                index_name=index_name,
                dimension=dimension,
                metric=metric,
                cloud=cloud,
                region=region,
                create_if_missing=create_if_missing,
                wait_timeout_sec=wait_timeout_sec,
            )

        progress_path = ctx.stage_work_dir / "legacy_upload_progress.json"
        progress_totals = {"summary": len(summary_docs), "text": len(text_docs)}
        progress_uploaded = {"summary": 0, "text": 0}
        atomic_write_json(
            progress_path,
            {
                "kind": "upload_legacy_vectorstores",
                "phase": "uploading_summary",
                "uploaded": progress_uploaded,
                "totals": progress_totals,
            },
        )

        summary_uploaded, summary_metadata_stats = _upload_records(
            index=pc.Index(summary_index_name),
            docs=summary_docs,
            dense_vectors=summary_dense,
            sparse_vectors=summary_sparse,
            namespace=namespace,
            upsert_batch_size=upsert_batch_size,
            progress_path=progress_path,
            progress_phase="summary",
            progress_totals=progress_totals,
            progress_uploaded=progress_uploaded,
            metadata_max_bytes=metadata_max_bytes,
            upsert_payload_max_bytes=upsert_payload_max_bytes,
        )
        progress_uploaded["summary"] = summary_uploaded
        text_uploaded, text_metadata_stats = _upload_records(
            index=pc.Index(text_index_name),
            docs=text_docs,
            dense_vectors=text_dense,
            sparse_vectors=text_sparse,
            namespace=namespace,
            upsert_batch_size=upsert_batch_size,
            progress_path=progress_path,
            progress_phase="text",
            progress_totals=progress_totals,
            progress_uploaded=progress_uploaded,
            metadata_max_bytes=metadata_max_bytes,
            upsert_payload_max_bytes=upsert_payload_max_bytes,
        )
        progress_uploaded["text"] = text_uploaded
        atomic_write_json(
            progress_path,
            {
                "kind": "upload_legacy_vectorstores",
                "phase": "complete",
                "uploaded": progress_uploaded,
                "totals": progress_totals,
            },
        )

        manifest_file = ctx.stage_work_dir / "legacy_pinecone_upload_manifest.json"
        manifest = {
            "schema_version": 1,
            "vectorstore_contract": "mbzuai_chatbot_legacy_v1",
            "summary_index_name": summary_index_name,
            "text_index_name": text_index_name,
            "namespace": namespace,
            "engine": engine,
            "model": model,
            "dimension": dimension,
            "metric": metric,
            "output_dimensionality": output_dimensionality,
            "use_sparse_embeddings": use_sparse,
            "bm25_model_file": bm25_model_file,
            "summary_vectors_uploaded": summary_uploaded,
            "text_vectors_uploaded": text_uploaded,
            "metadata_max_bytes": metadata_max_bytes,
            "upsert_payload_max_bytes": upsert_payload_max_bytes,
            "summary_metadata_stats": summary_metadata_stats,
            "text_metadata_stats": text_metadata_stats,
        }
        atomic_write_json(manifest_file, manifest)

        return StageResult.success(
            outputs={
                "legacy_summary_index_name": summary_index_name,
                "legacy_text_index_name": text_index_name,
                "legacy_bm25_model_file": bm25_model_file,
                "legacy_upload_manifest_file": str(manifest_file),
                "legacy_upload_progress_file": str(progress_path),
                "vectors_uploaded": summary_uploaded + text_uploaded,
            },
            metrics={
                "summary_vectors_uploaded": summary_uploaded,
                "text_vectors_uploaded": text_uploaded,
                "vectors_uploaded": summary_uploaded + text_uploaded,
                "dimension": dimension,
                "metric": metric,
                "sparse_enabled": use_sparse,
                "metadata_compacted_vectors": int(summary_metadata_stats["metadata_compacted_vectors"])
                + int(text_metadata_stats["metadata_compacted_vectors"]),
                "metadata_uploaded_bytes_max": max(
                    int(summary_metadata_stats["metadata_uploaded_bytes_max"]),
                    int(text_metadata_stats["metadata_uploaded_bytes_max"]),
                ),
                "upsert_request_bytes_max": max(
                    int(summary_metadata_stats["upsert_request_bytes_max"]),
                    int(text_metadata_stats["upsert_request_bytes_max"]),
                ),
            },
            artifacts=[
                ctx.make_artifact(
                    manifest_file,
                    artifact_type="legacy_pinecone_upload_manifest",
                    role="pinecone_upload_manifest",
                    metadata={
                        "summary_index_name": summary_index_name,
                        "text_index_name": text_index_name,
                        "vectors_uploaded": summary_uploaded + text_uploaded,
                        "metric": metric,
                    },
                )
            ],
        )
