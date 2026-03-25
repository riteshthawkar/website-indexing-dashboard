"""
Gemini multimodal embeddings + Pinecone upload stage.

Uploads chunk, parent, and media vectors into separate namespaces so retrieval
can blend precise chunk recall with parent expansion and media search.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pipeline.core.base import EmbedderStage, StageContext, StageResult
from pipeline.core.google_genai import import_genai, import_genai_types
from pipeline.core.io import atomic_write_json, load_json_safe, sha256_file
from pipeline.core.registry import register_stage
from pipeline.stages.embedders.openai_embedder import _serialize_metadata_value

logger = logging.getLogger(__name__)
_SPARSE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
}


def _load_records(path: str | Path) -> List[Dict[str, Any]]:
    payload = load_json_safe(path, []) or []
    return payload if isinstance(payload, list) else []


def _make_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    genai = import_genai()
    return genai.Client(api_key=api_key)


def _pinecone_cache_token() -> str:
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required")
    return f"{api_key}:{id(Pinecone)}"


@lru_cache(maxsize=4)
def _make_pinecone_client_cached(api_key: str, pinecone_identity: int):
    from pinecone import Pinecone
    return Pinecone(api_key=api_key)


def _make_pinecone_client():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required")
    token = _pinecone_cache_token()
    _api_key, identity = token.split(":", 1)
    return _make_pinecone_client_cached(api_key, int(identity))


def _embed_text_batch(
    client: Any,
    *,
    model: str,
    texts: Sequence[str],
    task_type: str,
    output_dimensionality: int | None,
) -> List[List[float]]:
    types = import_genai_types()

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    )
    response = client.models.embed_content(
        model=model,
        contents=list(texts),
        config=config,
    )
    return [list(embedding.values) for embedding in response.embeddings]


def _load_image(path: str | Path):
    from PIL import Image

    with Image.open(path) as image:
        return image.convert("RGB").copy()


def _embed_multimodal_batch(
    client: Any,
    *,
    model: str,
    items: Sequence[Dict[str, Any]],
    task_type: str,
    output_dimensionality: int | None,
) -> List[List[float]]:
    types = import_genai_types()

    contents: List[Any] = []
    for item in items:
        text = str(item.get("text") or "")
        local_path = str(item.get("local_path") or "")
        if local_path and Path(local_path).is_file():
            image = _load_image(local_path)
            contents.append([text, image] if text else [image])
        else:
            contents.append(text)

    config = types.EmbedContentConfig(
        task_type=task_type,
        output_dimensionality=output_dimensionality,
    )
    response = client.models.embed_content(
        model=model,
        contents=contents,
        config=config,
    )
    return [list(embedding.values) for embedding in response.embeddings]


def _record_metadata(record: Dict[str, Any], *, kind: str) -> Dict[str, Any]:
    metadata = {
        "record_type": kind,
        "document_id": record.get("document_id"),
        "document_title": record.get("document_title"),
        "document_type": record.get("document_type"),
        "source_url": record.get("source_url"),
        "source_markdown_path": record.get("source_markdown_path"),
        "source_file": record.get("source_file"),
        "page_key": record.get("page_key"),
        "section_key": record.get("section_key"),
        "section_path": record.get("section_path"),
        "page_numbers": record.get("page_numbers"),
        "parent_type": record.get("parent_type"),
        "media_type": record.get("media_type"),
        "asset_uri": record.get("asset_uri"),
        "url": record.get("url"),
        "title": record.get("title"),
        "caption": record.get("caption"),
        "provider": record.get("provider"),
        "predicate": record.get("predicate"),
        "answer_type": record.get("answer_type"),
        "answer_subtype": record.get("answer_subtype"),
        "subject_name": record.get("subject_name"),
        "object_value": record.get("object_value"),
        "linked_chunk_ids": record.get("linked_chunk_ids"),
        "linked_parent_ids": record.get("linked_parent_ids"),
        "source_chunk_ids": record.get("source_chunk_ids"),
        "source_parent_ids": record.get("source_parent_ids"),
        "source_fact_ids": record.get("source_fact_ids"),
    }
    return {
        key: _serialize_metadata_value(key, value)
        for key, value in metadata.items()
        if value not in (None, "", [], {})
    }


def _truncate_utf8_bytes(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    truncated = encoded[:max_bytes]
    return truncated.decode("utf-8", errors="ignore").rstrip()


def _fit_sparse_record_text(
    *,
    record_id: str,
    sparse_text_field: str,
    text: str,
    max_record_bytes: int,
) -> str:
    payload = {"_id": record_id, sparse_text_field: text}
    if len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) <= max_record_bytes:
        return text

    low = 0
    high = len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = _truncate_utf8_bytes(text, len(text[:mid].encode("utf-8")))
        payload = {"_id": record_id, sparse_text_field: candidate}
        size = len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        if size <= max_record_bytes:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best.rstrip()


def _has_sparse_embed_content(text: str) -> bool:
    tokens = re.findall(r"[\w'-]+", text.lower(), flags=re.UNICODE)
    if not tokens:
        return False
    for token in tokens:
        letters = "".join(ch for ch in token if ch.isalpha())
        digits = "".join(ch for ch in token if ch.isdigit())
        if len(letters) >= 2 and letters not in _SPARSE_STOPWORDS:
            return True
        if letters and digits and len(token) >= 3:
            return True
    return False


def _is_retryable_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "429",
            "resource_exhausted",
            "rate limit",
            "quota",
            "too many requests",
            "503",
            "500",
            "temporarily unavailable",
            "unavailable",
            "deadline exceeded",
            "timeout",
            "failed to connect",
            "connection error",
            "connection reset",
            "connection refused",
            "max retries exceeded",
            "remote disconnected",
            "broken pipe",
            "temporarily failed",
        )
    )


def _call_with_retry(
    label: str,
    func,
    *,
    max_attempts: int,
    base_delay_sec: float,
    max_delay_sec: float,
):
    attempt = 1
    while True:
        try:
            return func()
        except Exception as exc:
            if attempt >= max_attempts or not _is_retryable_exception(exc):
                raise
            delay = min(max_delay_sec, base_delay_sec * (2 ** (attempt - 1)))
            logger.warning(
                "%s failed with a retryable error on attempt %d/%d: %s. Retrying in %.1fs",
                label,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            time.sleep(delay)
            attempt += 1


def _wait_for_index_ready(pc: Any, index_name: str, *, timeout_sec: int = 300) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        description = pc.describe_index(index_name)
        status = getattr(description, "status", None)
        if isinstance(status, dict) and status.get("ready"):
            return
        if hasattr(status, "ready") and getattr(status, "ready"):
            return
        time.sleep(2.0)
    raise TimeoutError(f"Pinecone index {index_name} did not become ready within {timeout_sec}s")


def _ensure_index(pc: Any, *, index_name: str, dimension: int, cloud: str, region: str, metric: str = "cosine") -> bool:
    from pinecone import ServerlessSpec

    if pc.has_index(index_name):
        return True
    pc.create_index(
        name=index_name,
        spec=ServerlessSpec(cloud=cloud, region=region),
        dimension=dimension,
        metric=metric,
    )
    _wait_for_index_ready(pc, index_name)
    return False


def _ensure_sparse_index(
    pc: Any,
    *,
    index_name: str,
    cloud: str,
    region: str,
    sparse_model: str,
    sparse_text_field: str,
    metric: str = "dotproduct",
) -> bool:
    from pinecone.models.index_embed import IndexEmbed

    if pc.has_index(index_name):
        return True
    pc.create_index_for_model(
        name=index_name,
        cloud=cloud,
        region=region,
        embed=IndexEmbed(
            model=sparse_model,
            field_map={"text": sparse_text_field},
            metric=metric,
        ),
    )
    _wait_for_index_ready(pc, index_name)
    return False


def _iter_batches(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _optional_pinecone_timeout_kwargs(request_timeout: float | Tuple[float, float] | None) -> Dict[str, Any]:
    return {"_request_timeout": request_timeout} if request_timeout else {}


def _call_with_optional_pinecone_timeout(method, *args, request_timeout: float | Tuple[float, float] | None = None, **kwargs):
    if not request_timeout:
        return method(*args, **kwargs)
    try:
        return method(*args, **kwargs, **_optional_pinecone_timeout_kwargs(request_timeout))
    except TypeError as exc:
        if "_request_timeout" not in str(exc):
            raise
        return method(*args, **kwargs)


@lru_cache(maxsize=64)
def _resolve_index_host(index_name: str, cache_token: str) -> str:
    client = _make_pinecone_client()
    describe_index = getattr(client, "describe_index", None)
    if not callable(describe_index):
        return ""
    description = describe_index(index_name)
    if isinstance(description, dict):
        host = description.get("host")
    else:
        host = getattr(description, "host", None)
    return str(host or "")


@lru_cache(maxsize=64)
def _make_index_handle_cached(index_name: str, cache_token: str):
    # Use the explicit data-plane host. In this environment the SDK path that
    # opens an index by name can hang or fail to connect during upserts even
    # when the control-plane index is ready.
    client = _make_pinecone_client()
    host = _resolve_index_host(index_name, cache_token)
    if host:
        try:
            return client.Index(host=host)
        except TypeError:
            pass
    return client.Index(index_name)


def _make_index_handle(index_name: str):
    return _make_index_handle_cached(index_name, _pinecone_cache_token())


def _embed_media_records(
    client: Any,
    *,
    model: str,
    records: List[Dict[str, Any]],
    task_type: str,
    output_dimensionality: int | None,
    multimodal_batch_size: int,
    text_batch_size: int,
) -> Tuple[List[List[float]], Dict[str, int]]:
    embeddings: List[List[float]] = []
    multimodal_records = [record for record in records if record.get("can_embed_multimodal")]
    text_only_records = [record for record in records if not record.get("can_embed_multimodal")]
    metrics = {
        "media_multimodal_records": len(multimodal_records),
        "media_text_only_records": len(text_only_records),
        "media_multimodal_fallbacks": 0,
    }

    record_to_embedding: Dict[str, List[float]] = {}
    for batch in _iter_batches(multimodal_records, multimodal_batch_size):
        try:
            batch_embeddings = _embed_multimodal_batch(
                client,
                model=model,
                items=batch,
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            )
        except Exception:
            metrics["media_multimodal_fallbacks"] += len(batch)
            batch_embeddings = _embed_text_batch(
                client,
                model=model,
                texts=[str(item.get("text") or "") for item in batch],
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            )
        for record, vector in zip(batch, batch_embeddings):
            record_to_embedding[str(record["id"])] = vector

    for batch in _iter_batches(text_only_records, text_batch_size):
        batch_embeddings = _embed_text_batch(
            client,
            model=model,
            texts=[str(item.get("text") or "") for item in batch],
            task_type=task_type,
            output_dimensionality=output_dimensionality,
        )
        for record, vector in zip(batch, batch_embeddings):
            record_to_embedding[str(record["id"])] = vector

    for record in records:
        embeddings.append(record_to_embedding[str(record["id"])])
    return embeddings, metrics


def _upsert_namespace(
    index: Any,
    *,
    namespace: str,
    records: List[Dict[str, Any]],
    vectors: List[List[float]],
    kind: str,
    batch_size: int,
    request_timeout: float | Tuple[float, float] | None = None,
) -> int:
    uploaded = 0
    for batch_records, batch_vectors in zip(_iter_batches(records, batch_size), _iter_batches(vectors, batch_size)):
        payload = []
        for record, vector in zip(batch_records, batch_vectors):
            payload.append(
                {
                    "id": str(record["id"]),
                    "values": vector,
                    "metadata": _record_metadata(record, kind=kind),
                }
            )
        _call_with_optional_pinecone_timeout(
            index.upsert,
            vectors=payload,
            namespace=namespace,
            request_timeout=request_timeout,
        )
        uploaded += len(payload)
    return uploaded


def _write_progress(
    path: str | Path,
    *,
    index_name: str,
    model: str,
    output_dimensionality: int,
    totals: Dict[str, int],
    uploaded: Dict[str, int],
    phase: str,
    retrieval_bundle_file: str = "",
    retrieval_bundle_sha256: str = "",
    retrieval_bundle_stats: Dict[str, int] | None = None,
    media_metrics: Dict[str, int] | None = None,
) -> None:
    atomic_write_json(
        path,
        {
            "index_name": index_name,
            "model": model,
            "output_dimensionality": output_dimensionality,
            "phase": phase,
            "totals": totals,
            "uploaded": uploaded,
            "retrieval_bundle_file": retrieval_bundle_file,
            "retrieval_bundle_sha256": retrieval_bundle_sha256,
            "retrieval_bundle_stats": retrieval_bundle_stats or {},
            "media_metrics": media_metrics or {},
        },
    )


def _load_progress_state(
    path: str | Path,
    *,
    index_name: str,
    model: str,
    output_dimensionality: int,
    totals: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    payload = load_json_safe(path, {}) or {}
    if not isinstance(payload, dict):
        return {key: 0 for key in totals}, {
            "media_multimodal_records": 0,
            "media_text_only_records": 0,
            "media_multimodal_fallbacks": 0,
        }
    if (
        payload.get("index_name") != index_name
        or payload.get("model") != model
        or int(payload.get("output_dimensionality") or 0) != int(output_dimensionality)
    ):
        return {key: 0 for key in totals}, {
            "media_multimodal_records": 0,
            "media_text_only_records": 0,
            "media_multimodal_fallbacks": 0,
        }
    previous_totals = payload.get("totals") or {}
    uploaded = payload.get("uploaded") or {}
    media_metrics = payload.get("media_metrics") or {}
    return (
        {
            key: max(
                0,
                min(
                    int(uploaded.get(key) or 0),
                    int(totals[key]),
                ),
            )
            for key in totals
            if key in totals and (
                key in uploaded
                or key in previous_totals
            )
        }
        | {
            key: 0
            for key in totals
            if key not in uploaded and key not in previous_totals
        },
        {
            "media_multimodal_records": int(media_metrics.get("media_multimodal_records") or 0),
            "media_text_only_records": int(media_metrics.get("media_text_only_records") or 0),
            "media_multimodal_fallbacks": int(media_metrics.get("media_multimodal_fallbacks") or 0),
        },
    )


def _build_sparse_records(
    records: Sequence[Dict[str, Any]],
    *,
    sparse_text_field: str,
    max_text_chars: int,
    max_record_bytes: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    sparse_records: List[Dict[str, Any]] = []
    trimmed_count = 0
    skipped_count = 0
    for record in records:
        text = str(
            record.get("sparse_text")
            or record.get("lexical_text")
            or record.get("text")
            or record.get("dense_text")
            or ""
        ).strip()
        if not text:
            continue
        if max_text_chars > 0 and len(text) > max_text_chars:
            text = text[:max_text_chars].rstrip()
            trimmed_count += 1
        fitted = _fit_sparse_record_text(
            record_id=str(record["id"]),
            sparse_text_field=sparse_text_field,
            text=text,
            max_record_bytes=max_record_bytes,
        )
        if fitted != text:
            trimmed_count += 1
        if not fitted:
            skipped_count += 1
            logger.warning("Skipping sparse record %s because no Pinecone-safe payload could be built", record.get("id"))
            continue
        if not _has_sparse_embed_content(fitted):
            skipped_count += 1
            logger.warning(
                "Skipping sparse record %s because its text would produce an empty sparse vector",
                record.get("id"),
            )
            continue
        sparse_records.append(
            {
                "_id": str(record["id"]),
                sparse_text_field: fitted,
            }
        )
    return sparse_records, {"trimmed": trimmed_count, "skipped": skipped_count}


def _upsert_sparse_namespace(
    index: Any,
    *,
    namespace: str,
    records: List[Dict[str, Any]],
    batch_size: int,
    request_timeout: float | Tuple[float, float] | None = None,
) -> int:
    from pinecone.data.request_factory import IndexRequestFactory

    uploaded = 0
    for batch in _iter_batches(records, batch_size):
        payload = list(batch)
        if not payload:
            continue
        if hasattr(index, "_vector_api"):
            args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=payload)
            _call_with_optional_pinecone_timeout(
                index._vector_api.upsert_records_namespace,
                request_timeout=request_timeout,
                **args,
            )
        else:
            _call_with_optional_pinecone_timeout(
                index.upsert_records,
                namespace=namespace,
                records=payload,
                request_timeout=request_timeout,
            )
        uploaded += len(payload)
    return uploaded


def _clear_namespace(index: Any, *, namespace: str, request_timeout: float | Tuple[float, float] | None = None) -> None:
    try:
        _call_with_optional_pinecone_timeout(
            index.delete,
            delete_all=True,
            namespace=namespace,
            request_timeout=request_timeout,
        )
    except Exception as exc:
        message = str(exc).lower()
        if "namespace not found" in message or ('(404)' in message and 'namespace' in message):
            logger.info("Namespace %s does not exist yet; skipping clear", namespace)
            return
        raise


@register_stage
class GeminiPineconeEmbedder(EmbedderStage):
    name = "gemini_pinecone"
    description = "Embeds chunks, parents, and media with Gemini and uploads them to Pinecone."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        emb_cfg = config.get("embedder", {})
        if not os.getenv("GEMINI_API_KEY"):
            errors.append("GEMINI_API_KEY is required for Gemini embeddings")
        if not os.getenv("PINECONE_API_KEY"):
            errors.append("PINECONE_API_KEY is required")
        if not emb_cfg.get("pinecone_index"):
            errors.append("embedder.pinecone_index is required")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.embedder_config
        chunk_file = ctx.previous_outputs.get("chunk_embedding_file")
        parent_file = ctx.previous_outputs.get("parent_embedding_file")
        media_file = ctx.previous_outputs.get("media_embedding_file")
        fact_file = ctx.previous_outputs.get("fact_embedding_file")
        assertion_file = ctx.previous_outputs.get("assertion_embedding_file")
        bundle_file = ctx.previous_outputs.get("retrieval_bundle_file")

        if not chunk_file or not bundle_file:
            return StageResult.failure("retrieval formatter outputs are required before Gemini indexing")

        chunk_records = _load_records(chunk_file)
        parent_records = _load_records(parent_file) if parent_file else []
        media_records = _load_records(media_file) if media_file else []
        fact_records = _load_records(fact_file) if fact_file else []
        assertion_records = _load_records(assertion_file) if assertion_file else []
        if not chunk_records:
            return StageResult.failure("No chunk embedding records found")
        bundle_sha256 = sha256_file(bundle_file)
        bundle_stats = {
            "chunks": len(chunk_records),
            "parents": len(parent_records),
            "media": len(media_records),
            "facts": len(fact_records),
            "assertions": len(assertion_records),
        }

        model = str(config.get("model") or "gemini-embedding-2-preview")
        output_dimensionality = int(config.get("output_dimensionality") or 1536)
        task_type_document = str(config.get("task_type_document") or "RETRIEVAL_DOCUMENT")
        pinecone_index = str(config.get("pinecone_index") or "").strip()
        if not pinecone_index:
            return StageResult.failure("embedder.pinecone_index is required")

        text_batch_size = int(config.get("batch_size") or 32)
        media_text_batch_size = int(config.get("media_text_batch_size") or text_batch_size)
        media_multimodal_batch_size = int(config.get("media_multimodal_batch_size") or 4)
        upsert_batch_size = int(config.get("upsert_batch_size") or 100)
        max_retries = max(1, int(config.get("max_retries") or 6))
        retry_base_delay_sec = float(config.get("retry_base_delay_sec") or 5.0)
        retry_max_delay_sec = float(config.get("retry_max_delay_sec") or 120.0)
        pinecone_connect_timeout_sec = float(config.get("pinecone_connect_timeout_sec") or 10.0)
        pinecone_read_timeout_sec = float(config.get("pinecone_read_timeout_sec") or 120.0)
        pinecone_request_timeout: Tuple[float, float] = (
            pinecone_connect_timeout_sec,
            pinecone_read_timeout_sec,
        )
        progress_flush_every_batches = max(1, int(config.get("progress_flush_every_batches") or 1))
        namespace_chunks = str(config.get("namespace_chunks") or "chunks")
        namespace_parents = str(config.get("namespace_parents") or "parents")
        namespace_media = str(config.get("namespace_media") or "media")
        namespace_facts = str(config.get("namespace_facts") or "facts")
        namespace_assertions = str(config.get("namespace_assertions") or "assertions")
        enable_dense_facts = bool(config.get("enable_dense_facts", False))
        dense_fact_records = fact_records if enable_dense_facts else []
        enable_dense_assertions = bool(config.get("enable_dense_assertions", True))
        dense_assertion_records = assertion_records if enable_dense_assertions else []
        sparse_enabled = bool(config.get("enable_sparse", True))
        sparse_index_name = str(config.get("pinecone_sparse_index") or f"{pinecone_index}-sparse").strip()
        sparse_model = str(config.get("sparse_model") or "pinecone-sparse-english-v0")
        sparse_text_field = str(config.get("sparse_text_field") or "chunk_text")
        sparse_upsert_batch_size = int(config.get("sparse_upsert_batch_size") or upsert_batch_size)
        sparse_max_text_chars = int(config.get("sparse_max_text_chars") or 12000)
        sparse_max_record_bytes = int(config.get("sparse_max_record_bytes") or 24000)
        clear_dense_namespace_on_zero_progress = bool(config.get("clear_dense_namespace_on_zero_progress", True))
        clear_sparse_namespace_on_zero_progress = bool(config.get("clear_sparse_namespace_on_zero_progress", True))
        progress_path = ctx.stage_work_dir / "index_upload_progress.json"
        dense_namespace_reset_required = False
        sparse_namespace_reset_required = False
        totals = {
            "chunks": len(chunk_records),
            "parents": len(parent_records),
            "media": len(media_records),
            "facts": len(dense_fact_records),
            "assertions": len(dense_assertion_records),
            "sparse_chunks": len(chunk_records) if sparse_enabled else 0,
            "sparse_parents": len(parent_records) if sparse_enabled else 0,
            "sparse_media": len(media_records) if sparse_enabled else 0,
            "sparse_facts": len(fact_records) if sparse_enabled else 0,
            "sparse_assertions": len(assertion_records) if sparse_enabled else 0,
        }
        previous_progress = load_json_safe(progress_path, {}) or {}
        bundle_changed = (
            isinstance(previous_progress, dict)
            and bool(previous_progress.get("retrieval_bundle_sha256"))
            and str(previous_progress.get("retrieval_bundle_sha256")) != bundle_sha256
        )
        uploaded, media_metrics = _load_progress_state(
            progress_path,
            index_name=pinecone_index,
            model=model,
            output_dimensionality=output_dimensionality,
            totals=totals,
        )
        if bundle_changed:
            logger.warning(
                "Gemini embedder detected retrieval bundle drift for %s; resetting progress and rebuilding namespaces",
                pinecone_index,
            )
            uploaded = {key: 0 for key in totals}
            media_metrics = {
                "media_multimodal_records": 0,
                "media_text_only_records": 0,
                "media_multimodal_fallbacks": 0,
            }
        sparse_stats = {
            "chunk_records_trimmed": 0,
            "chunk_records_skipped": 0,
            "parent_records_trimmed": 0,
            "parent_records_skipped": 0,
            "media_records_trimmed": 0,
            "media_records_skipped": 0,
            "fact_records_trimmed": 0,
            "fact_records_skipped": 0,
            "assertion_records_trimmed": 0,
            "assertion_records_skipped": 0,
        }

        client = _make_gemini_client()

        logger.info(
            "Gemini embedder: chunks=%d parents=%d media=%d facts=%d assertions=%d model=%s dims=%d",
            len(chunk_records),
            len(parent_records),
            len(media_records),
            len(dense_fact_records),
            len(dense_assertion_records),
            model,
            output_dimensionality,
        )

        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_existed = True
        sparse_index_existed = True
        if bool(config.get("create_index_if_missing", True)):
            index_existed = _call_with_retry(
                "ensure_pinecone_index",
                lambda: _ensure_index(
                    pc,
                    index_name=pinecone_index,
                    dimension=output_dimensionality,
                    cloud=str(config.get("pinecone_cloud") or "aws"),
                    region=str(config.get("pinecone_region") or "us-east-1"),
                    metric=str(config.get("metric") or "cosine"),
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
        if not index_existed:
            uploaded = {key: 0 for key in totals}
            media_metrics = {
                "media_multimodal_records": 0,
                "media_text_only_records": 0,
                "media_multimodal_fallbacks": 0,
            }
        dense_namespace_reset_required = bundle_changed or (
            clear_dense_namespace_on_zero_progress
            and uploaded["chunks"] == 0
            and uploaded["parents"] == 0
            and uploaded["media"] == 0
            and uploaded["facts"] == 0
            and uploaded["assertions"] == 0
                )
        if dense_namespace_reset_required:
            for namespace in (namespace_chunks, namespace_parents, namespace_media, namespace_facts, namespace_assertions):
                _call_with_retry(
                    f"clear_dense_namespace_{namespace}",
                    lambda namespace=namespace: _clear_namespace(
                        _make_index_handle(pinecone_index),
                        namespace=namespace,
                        request_timeout=pinecone_request_timeout,
                    ),
                    max_attempts=max_retries,
                    base_delay_sec=retry_base_delay_sec,
                    max_delay_sec=retry_max_delay_sec,
                )
        if sparse_enabled:
            sparse_index_existed = _call_with_retry(
                "ensure_sparse_pinecone_index",
                lambda: _ensure_sparse_index(
                    pc,
                    index_name=sparse_index_name,
                    cloud=str(config.get("pinecone_cloud") or "aws"),
                    region=str(config.get("pinecone_region") or "us-east-1"),
                    sparse_model=sparse_model,
                    sparse_text_field=sparse_text_field,
                    metric=str(config.get("sparse_metric") or "dotproduct"),
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            if not sparse_index_existed:
                for key in ("sparse_chunks", "sparse_parents", "sparse_media", "sparse_facts"):
                    uploaded[key] = 0
            sparse_namespace_reset_required = bundle_changed or (
                clear_sparse_namespace_on_zero_progress
                and uploaded["sparse_chunks"] == 0
                and uploaded["sparse_parents"] == 0
                and uploaded["sparse_media"] == 0
                and uploaded["sparse_facts"] == 0
                and uploaded["sparse_assertions"] == 0
            )
            if sparse_namespace_reset_required:
                for namespace in (namespace_chunks, namespace_parents, namespace_media, namespace_facts, namespace_assertions):
                    _call_with_retry(
                        f"clear_sparse_namespace_{namespace}",
                        lambda namespace=namespace: _clear_namespace(
                            _make_index_handle(sparse_index_name),
                            namespace=namespace,
                            request_timeout=pinecone_request_timeout,
                        ),
                        max_attempts=max_retries,
                        base_delay_sec=retry_base_delay_sec,
                        max_delay_sec=retry_max_delay_sec,
                    )
        _write_progress(
            progress_path,
            index_name=pinecone_index,
            model=model,
            output_dimensionality=output_dimensionality,
            totals=totals,
            uploaded=uploaded,
            phase="index_ready",
            retrieval_bundle_file=str(bundle_file),
            retrieval_bundle_sha256=bundle_sha256,
            retrieval_bundle_stats=bundle_stats,
            media_metrics=media_metrics,
        )

        remaining_chunk_records = chunk_records[uploaded["chunks"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_chunk_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_chunk_batch",
                lambda batch_list=batch_list: _embed_text_batch(
                    client,
                    model=model,
                    texts=[str(record.get("dense_text") or record.get("text") or "") for record in batch_list],
                    task_type=task_type_document,
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            uploaded["chunks"] += _call_with_retry(
                "upsert_chunk_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_chunks,
                    records=batch_list,
                    vectors=vectors,
                    kind="chunk",
                    batch_size=upsert_batch_size,
                    request_timeout=pinecone_request_timeout,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            _write_progress(
                progress_path,
                index_name=pinecone_index,
                model=model,
                output_dimensionality=output_dimensionality,
                totals=totals,
                uploaded=uploaded,
                phase="uploading_chunks",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=bundle_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["chunks"] == totals["chunks"]:
                logger.info("Gemini embedder progress: uploaded %d/%d chunk vectors", uploaded["chunks"], totals["chunks"])

        remaining_parent_records = parent_records[uploaded["parents"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_parent_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_parent_batch",
                lambda batch_list=batch_list: _embed_text_batch(
                    client,
                    model=model,
                    texts=[str(record.get("dense_text") or record.get("text") or "") for record in batch_list],
                    task_type=task_type_document,
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            uploaded["parents"] += _call_with_retry(
                "upsert_parent_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_parents,
                    records=batch_list,
                    vectors=vectors,
                    kind="parent",
                    batch_size=upsert_batch_size,
                    request_timeout=pinecone_request_timeout,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            _write_progress(
                progress_path,
                index_name=pinecone_index,
                model=model,
                output_dimensionality=output_dimensionality,
                totals=totals,
                uploaded=uploaded,
                phase="uploading_parents",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=bundle_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["parents"] == totals["parents"]:
                logger.info("Gemini embedder progress: uploaded %d/%d parent vectors", uploaded["parents"], totals["parents"])

        remaining_fact_records = dense_fact_records[uploaded["facts"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_fact_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_fact_batch",
                lambda batch_list=batch_list: _embed_text_batch(
                    client,
                    model=model,
                    texts=[str(record.get("dense_text") or record.get("text") or "") for record in batch_list],
                    task_type=task_type_document,
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            uploaded["facts"] += _call_with_retry(
                "upsert_fact_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_facts,
                    records=batch_list,
                    vectors=vectors,
                    kind="fact",
                    batch_size=upsert_batch_size,
                    request_timeout=pinecone_request_timeout,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            _write_progress(
                progress_path,
                index_name=pinecone_index,
                model=model,
                output_dimensionality=output_dimensionality,
                totals=totals,
                uploaded=uploaded,
                phase="uploading_facts",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=bundle_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["facts"] == totals["facts"]:
                logger.info("Gemini embedder progress: uploaded %d/%d fact vectors", uploaded["facts"], totals["facts"])

        remaining_assertion_records = dense_assertion_records[uploaded["assertions"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_assertion_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_assertion_batch",
                lambda batch_list=batch_list: _embed_text_batch(
                    client,
                    model=model,
                    texts=[str(record.get("dense_text") or record.get("text") or "") for record in batch_list],
                    task_type=task_type_document,
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            uploaded["assertions"] += _call_with_retry(
                "upsert_assertion_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_assertions,
                    records=batch_list,
                    vectors=vectors,
                    kind="assertion",
                    batch_size=upsert_batch_size,
                    request_timeout=pinecone_request_timeout,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            _write_progress(
                progress_path,
                index_name=pinecone_index,
                model=model,
                output_dimensionality=output_dimensionality,
                totals=totals,
                uploaded=uploaded,
                phase="uploading_assertions",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=bundle_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["assertions"] == totals["assertions"]:
                logger.info(
                    "Gemini embedder progress: uploaded %d/%d assertion vectors",
                    uploaded["assertions"],
                    totals["assertions"],
                )

        remaining_media_records = media_records[uploaded["media"] :]
        media_metrics["media_multimodal_records"] = len(
            [record for record in media_records if record.get("can_embed_multimodal")]
        )
        media_metrics["media_text_only_records"] = len(media_records) - media_metrics["media_multimodal_records"]
        multimodal_buffer: List[Dict[str, Any]] = []
        text_buffer: List[Dict[str, Any]] = []

        def flush_media_batch(batch_records: List[Dict[str, Any]], *, multimodal: bool) -> None:
            nonlocal uploaded, media_metrics
            if not batch_records:
                return
            label = "embed_media_multimodal_batch" if multimodal else "embed_media_text_batch"
            if multimodal:
                try:
                    vectors = _call_with_retry(
                        label,
                        lambda batch_records=batch_records: _embed_multimodal_batch(
                            client,
                            model=model,
                            items=batch_records,
                            task_type=task_type_document,
                            output_dimensionality=output_dimensionality,
                        ),
                        max_attempts=max_retries,
                        base_delay_sec=retry_base_delay_sec,
                        max_delay_sec=retry_max_delay_sec,
                    )
                except Exception:
                    media_metrics["media_multimodal_fallbacks"] += len(batch_records)
                    vectors = _call_with_retry(
                        "embed_media_fallback_text_batch",
                        lambda batch_records=batch_records: _embed_text_batch(
                            client,
                            model=model,
                            texts=[str(item.get("text") or "") for item in batch_records],
                            task_type=task_type_document,
                            output_dimensionality=output_dimensionality,
                        ),
                        max_attempts=max_retries,
                        base_delay_sec=retry_base_delay_sec,
                        max_delay_sec=retry_max_delay_sec,
                    )
            else:
                vectors = _call_with_retry(
                    label,
                    lambda batch_records=batch_records: _embed_text_batch(
                        client,
                        model=model,
                        texts=[str(item.get("text") or "") for item in batch_records],
                        task_type=task_type_document,
                        output_dimensionality=output_dimensionality,
                    ),
                    max_attempts=max_retries,
                    base_delay_sec=retry_base_delay_sec,
                    max_delay_sec=retry_max_delay_sec,
                )
            uploaded["media"] += _call_with_retry(
                "upsert_media_batch",
                lambda batch_records=batch_records, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_media,
                    records=batch_records,
                    vectors=vectors,
                    kind="media",
                    batch_size=upsert_batch_size,
                    request_timeout=pinecone_request_timeout,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            _write_progress(
                progress_path,
                index_name=pinecone_index,
                model=model,
                output_dimensionality=output_dimensionality,
                totals=totals,
                uploaded=uploaded,
                phase="uploading_media",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=bundle_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )

        for record in remaining_media_records:
            if record.get("can_embed_multimodal"):
                if text_buffer:
                    flush_media_batch(text_buffer, multimodal=False)
                    text_buffer = []
                multimodal_buffer.append(record)
                if len(multimodal_buffer) >= media_multimodal_batch_size:
                    flush_media_batch(multimodal_buffer, multimodal=True)
                    multimodal_buffer = []
            else:
                if multimodal_buffer:
                    flush_media_batch(multimodal_buffer, multimodal=True)
                    multimodal_buffer = []
                text_buffer.append(record)
                if len(text_buffer) >= media_text_batch_size:
                    flush_media_batch(text_buffer, multimodal=False)
                    text_buffer = []

        flush_media_batch(multimodal_buffer, multimodal=True)
        flush_media_batch(text_buffer, multimodal=False)
        if media_records:
            logger.info("Gemini embedder progress: uploaded %d/%d media vectors", uploaded["media"], totals["media"])

        sparse_manifest = {
            "enabled": sparse_enabled,
            "index_name": sparse_index_name if sparse_enabled else "",
            "model": sparse_model if sparse_enabled else "",
            "record_stats": sparse_stats,
        }
        if sparse_enabled:
            sparse_chunks, chunk_sparse_stats = _build_sparse_records(
                chunk_records,
                sparse_text_field=sparse_text_field,
                max_text_chars=sparse_max_text_chars,
                max_record_bytes=sparse_max_record_bytes,
            )
            sparse_parents, parent_sparse_stats = _build_sparse_records(
                parent_records,
                sparse_text_field=sparse_text_field,
                max_text_chars=sparse_max_text_chars,
                max_record_bytes=sparse_max_record_bytes,
            )
            sparse_media, media_sparse_stats = _build_sparse_records(
                media_records,
                sparse_text_field=sparse_text_field,
                max_text_chars=sparse_max_text_chars,
                max_record_bytes=sparse_max_record_bytes,
            )
            sparse_facts, fact_sparse_stats = _build_sparse_records(
                fact_records,
                sparse_text_field=sparse_text_field,
                max_text_chars=sparse_max_text_chars,
                max_record_bytes=sparse_max_record_bytes,
            )
            sparse_assertions, assertion_sparse_stats = _build_sparse_records(
                assertion_records,
                sparse_text_field=sparse_text_field,
                max_text_chars=sparse_max_text_chars,
                max_record_bytes=sparse_max_record_bytes,
            )
            sparse_stats = {
                "chunk_records_trimmed": chunk_sparse_stats["trimmed"],
                "chunk_records_skipped": chunk_sparse_stats["skipped"],
                "parent_records_trimmed": parent_sparse_stats["trimmed"],
                "parent_records_skipped": parent_sparse_stats["skipped"],
                "media_records_trimmed": media_sparse_stats["trimmed"],
                "media_records_skipped": media_sparse_stats["skipped"],
                "fact_records_trimmed": fact_sparse_stats["trimmed"],
                "fact_records_skipped": fact_sparse_stats["skipped"],
                "assertion_records_trimmed": assertion_sparse_stats["trimmed"],
                "assertion_records_skipped": assertion_sparse_stats["skipped"],
            }

            sparse_plan = [
                ("sparse_chunks", namespace_chunks, sparse_chunks),
                ("sparse_parents", namespace_parents, sparse_parents),
                ("sparse_media", namespace_media, sparse_media),
                ("sparse_facts", namespace_facts, sparse_facts),
                ("sparse_assertions", namespace_assertions, sparse_assertions),
            ]
            for progress_key, namespace, records in sparse_plan:
                if not records:
                    continue
                if uploaded[progress_key] == 0 and clear_sparse_namespace_on_zero_progress and not sparse_namespace_reset_required:
                    _call_with_retry(
                        f"clear_{progress_key}_namespace",
                        lambda namespace=namespace: _clear_namespace(
                            _make_index_handle(sparse_index_name),
                            namespace=namespace,
                            request_timeout=pinecone_request_timeout,
                        ),
                        max_attempts=max_retries,
                        base_delay_sec=retry_base_delay_sec,
                        max_delay_sec=retry_max_delay_sec,
                    )
                remaining_records = records[uploaded[progress_key] :]
                if not remaining_records:
                    continue
                uploaded[progress_key] += _call_with_retry(
                    f"upsert_{progress_key}",
                    lambda records=remaining_records, namespace=namespace: _upsert_sparse_namespace(
                        _make_index_handle(sparse_index_name),
                        namespace=namespace,
                        records=records,
                        batch_size=sparse_upsert_batch_size,
                        request_timeout=pinecone_request_timeout,
                    ),
                    max_attempts=max_retries,
                    base_delay_sec=retry_base_delay_sec,
                    max_delay_sec=retry_max_delay_sec,
                )
                logger.info(
                    "Gemini embedder progress: uploaded %d/%d sparse records for %s",
                    uploaded[progress_key],
                    totals[progress_key],
                    namespace,
                )
                _write_progress(
                    progress_path,
                    index_name=pinecone_index,
                    model=model,
                    output_dimensionality=output_dimensionality,
                    totals=totals,
                    uploaded=uploaded,
                    phase=f"uploading_{progress_key}",
                    retrieval_bundle_file=str(bundle_file),
                    retrieval_bundle_sha256=bundle_sha256,
                    retrieval_bundle_stats=bundle_stats,
                    media_metrics=media_metrics,
                )
            sparse_manifest["record_stats"] = sparse_stats

        upload_manifest = {
            "index_name": pinecone_index,
            "sparse_index_name": sparse_index_name if sparse_enabled else "",
            "model": model,
            "output_dimensionality": output_dimensionality,
            "namespaces": {
                "chunks": namespace_chunks,
                "parents": namespace_parents,
                "media": namespace_media,
                "facts": namespace_facts,
                "assertions": namespace_assertions,
            },
            "uploaded": {
                "chunks": uploaded["chunks"],
                "parents": uploaded["parents"],
                "media": uploaded["media"],
                "facts": uploaded["facts"],
                "assertions": uploaded["assertions"],
                "sparse_chunks": uploaded["sparse_chunks"],
                "sparse_parents": uploaded["sparse_parents"],
                "sparse_media": uploaded["sparse_media"],
                "sparse_facts": uploaded["sparse_facts"],
                "sparse_assertions": uploaded["sparse_assertions"],
            },
            "retrieval_bundle_file": bundle_file,
            "retrieval_bundle_sha256": bundle_sha256,
            "retrieval_bundle_stats": bundle_stats,
            "media_metrics": media_metrics,
            "sparse": sparse_manifest,
        }
        manifest_path = ctx.stage_work_dir / "index_upload_manifest.json"
        atomic_write_json(manifest_path, upload_manifest)
        _write_progress(
            progress_path,
            index_name=pinecone_index,
            model=model,
            output_dimensionality=output_dimensionality,
            totals=totals,
            uploaded=uploaded,
            phase="completed",
            retrieval_bundle_file=str(bundle_file),
            retrieval_bundle_sha256=bundle_sha256,
            retrieval_bundle_stats=bundle_stats,
            media_metrics=media_metrics,
        )

        return StageResult.success(
            outputs={
                "vectors_uploaded": (
                    uploaded["chunks"]
                    + uploaded["parents"]
                    + uploaded["media"]
                    + uploaded["facts"]
                    + uploaded["assertions"]
                ),
                "index_name": pinecone_index,
                "sparse_index_name": sparse_index_name if sparse_enabled else "",
                "index_manifest_file": str(manifest_path),
            },
            metrics={
                "vectors_uploaded": (
                    uploaded["chunks"]
                    + uploaded["parents"]
                    + uploaded["media"]
                    + uploaded["facts"]
                    + uploaded["assertions"]
                ),
                "chunk_vectors_uploaded": uploaded["chunks"],
                "parent_vectors_uploaded": uploaded["parents"],
                "media_vectors_uploaded": uploaded["media"],
                "fact_vectors_uploaded": uploaded["facts"],
                "assertion_vectors_uploaded": uploaded["assertions"],
                "sparse_chunk_records_uploaded": uploaded["sparse_chunks"],
                "sparse_parent_records_uploaded": uploaded["sparse_parents"],
                "sparse_media_records_uploaded": uploaded["sparse_media"],
                "sparse_fact_records_uploaded": uploaded["sparse_facts"],
                "sparse_assertion_records_uploaded": uploaded["sparse_assertions"],
                "sparse_records_trimmed": sum(value for key, value in sparse_stats.items() if key.endswith("_trimmed")),
                "sparse_records_skipped": sum(value for key, value in sparse_stats.items() if key.endswith("_skipped")),
                **media_metrics,
                "documents_total": len(chunk_records),
                "engine": "gemini",
                "model": model,
            },
            artifacts=[
                ctx.make_artifact(
                    manifest_path,
                    artifact_type="index_manifest",
                    role="vector_index_upload",
                    metadata={
                        "index_name": pinecone_index,
                        "sparse_index_name": sparse_index_name if sparse_enabled else "",
                        "vectors_uploaded": (
                            uploaded["chunks"]
                            + uploaded["parents"]
                            + uploaded["media"]
                            + uploaded["facts"]
                            + uploaded["assertions"]
                        ),
                    },
                )
            ],
        )
