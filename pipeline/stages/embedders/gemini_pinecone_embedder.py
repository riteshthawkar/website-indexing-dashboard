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
import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from pipeline.core.artifact_contracts import ArtifactContract, resolve_first_artifact_path
from pipeline.core.base import EmbedderStage, StageContext, StageResult
from pipeline.core.google_genai import import_genai, import_genai_types
from pipeline.core.graph_artifacts import resolve_canonical_graph_artifacts
from pipeline.core.io import (
    atomic_write_json,
    combine_sha256_digests,
    load_json_safe,
    sha256_file,
)
from pipeline.core.registry import register_stage
from pipeline.stages.embedders.openai_embedder import _serialize_metadata_value

logger = logging.getLogger(__name__)
_SPARSE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "that", "the", "their", "this", "to", "was", "were", "with",
}


def _release_namespace_token(run_id: str, *, max_length: int = 48) -> str:
    """Return a stable Pinecone-safe token for one immutable indexing run."""
    raw = str(run_id or "").strip()
    if not raw:
        raise ValueError("run_id is required when embedder.namespace_strategy=release")
    token = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw).strip("-_").lower()
    if not token:
        token = "release"
    # The readable form is not injective (for example ``candidate/a`` and
    # ``candidate-a`` normalize to the same value).  Always bind the namespace
    # token to the exact raw run id, even when the readable portion is short.
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    readable_length = max(1, max_length - len(digest) - 2)
    return f"{token[:readable_length]}--{digest}"


def _resolve_upload_namespaces(config: Dict[str, Any], *, run_id: str) -> Dict[str, str]:
    """Resolve lane namespaces, optionally isolating every release.

    Static namespaces remain supported for legacy runs. Production configs use
    ``namespace_strategy: release`` so a candidate upload cannot clear or mutate
    the namespaces used by the active retriever.
    """
    lane_defaults = {
        "chunks": "chunks",
        "parents": "parents",
        "media": "media",
        "facts": "facts",
        "evidence_spans": "evidence_spans",
        "summaries": "summaries",
        "assertions": "assertions",
        "entities": "entities",
        "communities": "communities",
    }
    strategy = str(config.get("namespace_strategy") or "static").strip().lower()
    if strategy not in {"static", "release"}:
        raise ValueError("embedder.namespace_strategy must be 'static' or 'release'")

    release_token = _release_namespace_token(run_id) if strategy == "release" else ""
    namespaces: Dict[str, str] = {}
    for lane, default in lane_defaults.items():
        configured = str(config.get(f"namespace_{lane}") or default).strip()
        if strategy == "release":
            template = str(config.get("namespace_release_template") or "{base}--{release_id}")
            if "{release_id}" not in template:
                raise ValueError(
                    "embedder.namespace_release_template must contain the exact "
                    "{release_id} placeholder"
                )
            configured = template.format(base=configured, lane=lane, release_id=release_token).strip()
            if release_token not in configured:
                raise ValueError(
                    "embedder.namespace_release_template must preserve the complete release identity"
                )
        if not configured:
            raise ValueError(f"Resolved Pinecone namespace for {lane} is empty")
        namespaces[lane] = configured
    if len(set(namespaces.values())) != len(namespaces):
        raise ValueError("Each retrieval lane must resolve to a distinct Pinecone namespace")
    return namespaces


def _assert_upload_plan_complete(
    planned: Mapping[str, Any],
    uploaded: Mapping[str, Any],
) -> None:
    """Fail before manifest emission when any dense or sparse lane is partial."""
    count_mismatches = {
        key: {"planned": int(total), "uploaded": int(uploaded.get(key) or 0)}
        for key, total in planned.items()
        if int(uploaded.get(key) or 0) != int(total)
    }
    if count_mismatches:
        raise RuntimeError(
            "Vector upload cardinality did not reach the immutable upload plan: "
            f"{count_mismatches}"
        )


def _load_records(path: str | Path) -> List[Dict[str, Any]]:
    payload = load_json_safe(path, []) or []
    return payload if isinstance(payload, list) else []


def _resolve_input_path(
    ctx: StageContext,
    *contracts: ArtifactContract,
) -> str:
    resolved = resolve_first_artifact_path(ctx, contracts)
    return resolved.path if resolved else ""


def _resolve_indexing_input_paths(ctx: StageContext) -> Dict[str, str]:
    """Resolve upload inputs from artifact catalog first, then legacy outputs."""
    canonical_graph = resolve_canonical_graph_artifacts(
        ctx.work_dir,
        required=False,
        require_index=True,
        validate_binding=bool(
            (ctx.config.get("pipeline") or {}).get("production_profile", False)
        ),
    )
    graph_bundle = str(canonical_graph.graph_file) if canonical_graph is not None else _resolve_input_path(
        ctx,
        ArtifactContract(
            artifact_type="knowledge_graph_bundle",
            role="knowledge_graph_with_communities",
            legacy_output_key="summarized_community_graph_file",
            label="community knowledge graph bundle",
        ),
        ArtifactContract(
            artifact_type="knowledge_graph_bundle",
            role="knowledge_graph_with_communities",
            legacy_output_key="community_graph_file",
            label="community graph bundle",
        ),
        ArtifactContract(
            artifact_type="knowledge_graph_bundle",
            role="knowledge_graph",
            legacy_output_key="promoted_knowledge_graph_file",
            label="promoted knowledge graph bundle",
        ),
        ArtifactContract(
            artifact_type="knowledge_graph_bundle",
            role="knowledge_graph",
            legacy_output_key="knowledge_graph_file",
            label="knowledge graph bundle",
        ),
        ArtifactContract(
            artifact_type="knowledge_graph_bundle",
            role="knowledge_graph",
            legacy_output_key="graph_bundle_file",
            label="graph bundle",
        ),
    )
    return {
        "bundle": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="retrieval_bundle",
                role="retrieval_corpus",
                legacy_output_key="retrieval_bundle_file",
                label="retrieval bundle",
            ),
        ),
        "lexical_corpus": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="lexical_corpus",
                role="lexical_retrieval",
                legacy_output_key="lexical_corpus_file",
                label="lexical retrieval corpus",
            ),
        ),
        "promoted_assertions": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="promoted_assertions",
                role="assertion_promoted",
                legacy_output_key="promoted_assertions_file",
                label="promoted assertions",
            ),
        ),
        "chunks": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_chunks",
                legacy_output_key="chunk_embedding_file",
                label="chunk embedding payload",
            ),
        ),
        "parents": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_parents",
                legacy_output_key="parent_embedding_file",
                label="parent embedding payload",
            ),
        ),
        "media": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_media",
                legacy_output_key="media_embedding_file",
                label="media embedding payload",
            ),
        ),
        "facts": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_facts",
                legacy_output_key="fact_embedding_file",
                label="fact embedding payload",
            ),
        ),
        "evidence_spans": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_evidence_spans",
                legacy_output_key="evidence_span_embedding_file",
                label="evidence-span embedding payload",
            ),
        ),
        "summaries": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_summaries",
                legacy_output_key="summary_embedding_file",
                label="summary embedding payload",
            ),
        ),
        "assertions": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_assertions",
                legacy_output_key="assertion_embedding_file",
                label="assertion embedding payload",
            ),
        ),
        "entities": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="entity_records",
                legacy_output_key="entity_records_file",
                label="entity records",
            ),
            ArtifactContract(
                artifact_type="formatted_documents",
                role="entity_records",
                legacy_output_key="entity_embedding_file",
                label="entity records",
            ),
        ),
        "communities": _resolve_input_path(
            ctx,
            ArtifactContract(
                artifact_type="formatted_documents",
                role="embedding_payload_communities",
                legacy_output_key="community_embedding_file",
                label="community embedding payload",
            ),
        ),
        "graph_bundle": graph_bundle,
    }


def _make_gemini_client(*, request_timeout_ms: int | None = None):
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
    genai = import_genai()
    if request_timeout_ms and request_timeout_ms > 0:
        types = import_genai_types()
        return genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(request_timeout_ms)),
        )
    return genai.Client(api_key=api_key)


def _pinecone_cache_token() -> str:
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required")
    transport = str(os.environ.get("PINECONE_TRANSPORT") or "rest").strip().lower()
    pinecone_cls = _pinecone_client_class(transport)
    return f"{api_key}:{transport}:{id(pinecone_cls)}"


@lru_cache(maxsize=4)
def _make_pinecone_client_cached(api_key: str, transport: str, pinecone_identity: int):
    pinecone_cls = _pinecone_client_class(transport)
    return pinecone_cls(api_key=api_key)


def _pinecone_client_class(transport: str):
    if str(transport or "").strip().lower() == "grpc":
        try:
            from pinecone.grpc import PineconeGRPC

            return PineconeGRPC
        except Exception:
            logger.warning("PINECONE_TRANSPORT=grpc requested but PineconeGRPC is unavailable; falling back to REST")
    from pinecone import Pinecone

    return Pinecone


def _make_pinecone_client():
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY is required")
    token = _pinecone_cache_token()
    _api_key, transport, identity = token.split(":", 2)
    return _make_pinecone_client_cached(api_key, transport, int(identity))


def _uses_prompt_task_instruction(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized == "gemini-embedding-2"


def _format_embedding_text(text: Any, *, task_type: str, model: str) -> str:
    body = str(text or " ").strip() or " "
    if not _uses_prompt_task_instruction(model):
        return body
    task = str(task_type or "").upper()
    if task == "RETRIEVAL_QUERY":
        return f"task: search result | query: {body}"
    title = "none"
    for line in body.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip().casefold() == "title" and value.strip():
            title = value.strip()
            break
    return f"title: {title} | text: {body}"


def _embed_text_batch(
    client: Any,
    *,
    model: str,
    texts: Sequence[str],
    task_type: str,
    output_dimensionality: int | None,
) -> List[List[float]]:
    types = import_genai_types()
    config_kwargs = {"output_dimensionality": output_dimensionality}
    if not _uses_prompt_task_instruction(model):
        config_kwargs["task_type"] = task_type
    config = types.EmbedContentConfig(**config_kwargs)
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=_format_embedding_text(text, task_type=task_type, model=model))],
        )
        for text in texts
    ]
    response = client.models.embed_content(
        model=model,
        contents=contents,
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
        text = _format_embedding_text(item.get("text") or "", task_type=task_type, model=model)
        local_path = str(item.get("local_path") or "")
        if local_path and Path(local_path).is_file():
            image = _load_image(local_path)
            contents.append([text, image] if text else [image])
        else:
            contents.append(text)

    config_kwargs = {"output_dimensionality": output_dimensionality}
    if not _uses_prompt_task_instruction(model):
        config_kwargs["task_type"] = task_type
    config = types.EmbedContentConfig(**config_kwargs)
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
        "summary_type": record.get("summary_type"),
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
        "canonical_name": record.get("canonical_name"),
        "aliases": record.get("aliases"),
        "entity_type": record.get("entity_type"),
        "description": record.get("description"),
        "confidence": record.get("confidence"),
        "validity_status": record.get("validity_status"),
        "source_last_seen": record.get("source_last_seen"),
        "canonical_subject": record.get("canonical_subject"),
        "canonical_predicate": record.get("canonical_predicate"),
        "canonical_object": record.get("canonical_object"),
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


def _dedupe_text_parts(parts: Iterable[Any]) -> List[str]:
    unique: List[str] = []
    seen = set()
    for part in parts:
        text = " ".join(str(part or "").split()).strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(text)
    return unique


def _record_identity_text(record: Dict[str, Any]) -> str:
    aliases = record.get("aliases")
    alias_parts = aliases if isinstance(aliases, list) else []
    return " ".join(
        _dedupe_text_parts(
            [
                record.get("canonical_name"),
                *alias_parts,
                record.get("description"),
            ]
        )
    ).strip()


def _add_type_context_for_weak_text(record: Dict[str, Any], text: str) -> str:
    if not text or _has_sparse_embed_content(text):
        return text
    entity_type = str(record.get("entity_type") or "").replace("_", " ").strip()
    if entity_type and entity_type.casefold() not in {"other", "unknown", "misc"}:
        return " ".join(_dedupe_text_parts([text, entity_type])).strip()
    return text


def _record_embedding_text(record: Dict[str, Any]) -> str:
    text = str(
        record.get("dense_text")
        or record.get("text")
        or record.get("sparse_text")
        or record.get("lexical_text")
        or ""
    ).strip()
    if not text:
        text = _record_identity_text(record)
    return _add_type_context_for_weak_text(record, text)


def _dedupe_records_by_id(records: Sequence[Dict[str, Any]], *, record_kind: str) -> List[Dict[str, Any]]:
    """Collapse exact duplicate vector IDs before count verification.

    Pinecone stores one vector per ID. A duplicate ID in the source records
    therefore makes uploaded-record counts impossible to verify. Exact
    duplicates are safe to collapse; conflicting duplicates indicate an ID
    generation bug and must fail before upload.
    """

    deduped: List[Dict[str, Any]] = []
    seen: Dict[str, Dict[str, Any]] = {}
    duplicate_count = 0
    for record in records:
        record_id = str(record.get("id") or "").strip()
        if not record_id:
            deduped.append(record)
            continue
        existing = seen.get(record_id)
        if existing is None:
            seen[record_id] = record
            deduped.append(record)
            continue
        duplicate_count += 1
        existing_text = _record_embedding_text(existing)
        incoming_text = _record_embedding_text(record)
        existing_sparse = str(existing.get("sparse_text") or existing.get("lexical_text") or "").strip()
        incoming_sparse = str(record.get("sparse_text") or record.get("lexical_text") or "").strip()
        existing_source = str(existing.get("source_url") or existing.get("canonical_url") or "").strip()
        incoming_source = str(record.get("source_url") or record.get("canonical_url") or "").strip()
        existing_chunk = str(existing.get("chunk_id") or "").strip()
        incoming_chunk = str(record.get("chunk_id") or "").strip()
        if (
            existing_text != incoming_text
            or existing_sparse != incoming_sparse
            or existing_source != incoming_source
            or existing_chunk != incoming_chunk
        ):
            raise ValueError(f"{record_kind} contains conflicting duplicate vector id {record_id}")
    if duplicate_count:
        logger.warning(
            "Deduplicated %d duplicate %s records by vector id before upload",
            duplicate_count,
            record_kind,
        )
    return deduped


def _normalise_entity_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalised = dict(record)
    text = _record_embedding_text(normalised)
    if text:
        normalised.setdefault("text", text)
        normalised.setdefault("dense_text", text)
    return normalised


def _graph_entity_record(node: Dict[str, Any]) -> Dict[str, Any]:
    properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    canonical_name = (
        node.get("canonical_name")
        or properties.get("canonical_name")
        or node.get("label")
        or properties.get("label")
        or properties.get("name")
        or node.get("id")
    )
    aliases = node.get("aliases") or properties.get("aliases") or []
    if aliases and not isinstance(aliases, list):
        aliases = [aliases]
    record = {
        "id": node.get("id"),
        "canonical_name": canonical_name,
        "aliases": aliases,
        "description": node.get("description") or properties.get("description") or "",
        "entity_type": node.get("entity_type") or properties.get("entity_type") or "Other",
        "confidence": node.get("confidence") or properties.get("confidence"),
    }
    return _normalise_entity_record(record)


def _record_text_fingerprint(records: Sequence[Dict[str, Any]]) -> str:
    if not records:
        return ""
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record.get("id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(_record_embedding_text(record).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


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
            "timed out",
            "write operation timed out",
            "failed to connect",
            "connection error",
            "connection reset",
            "connection refused",
            "max retries exceeded",
            "remote disconnected",
            "server disconnected without sending a response",
            "remoteprotocolerror",
            "remote protocol error",
            "broken pipe",
            "unexpected_eof_while_reading",
            "eof occurred in violation of protocol",
            "nodename nor servname",
            "name or service not known",
            "temporary failure in name resolution",
            "getaddrinfo",
            "dns",
            "temporarily failed",
            "verification pending",
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
    if pc.has_index(index_name):
        return True
    index_embed_cls = None
    for module_name in (
        "pinecone.inference.models.index_embed",
        "pinecone.models.index_embed",
    ):
        try:
            module = __import__(module_name, fromlist=["IndexEmbed"])
            index_embed_cls = getattr(module, "IndexEmbed")
            break
        except (ImportError, AttributeError):
            continue
    embed_config = (
        index_embed_cls(
            model=sparse_model,
            field_map={"text": sparse_text_field},
            metric=metric,
        )
        if index_embed_cls is not None
        else {
            "model": sparse_model,
            "field_map": {"text": sparse_text_field},
            "metric": metric,
        }
    )
    pc.create_index_for_model(
        name=index_name,
        cloud=cloud,
        region=region,
        embed=embed_config,
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
    import time
    import random

    max_retries = 5
    backoff = 1.0

    for attempt in range(max_retries + 1):
        try:
            if not request_timeout:
                return method(*args, **kwargs)
            if str(os.environ.get("PINECONE_TRANSPORT") or "").strip().lower() == "grpc":
                timeout_value = request_timeout[1] if isinstance(request_timeout, tuple) else request_timeout
                return method(*args, **kwargs, timeout=timeout_value)
            try:
                return method(*args, **kwargs, **_optional_pinecone_timeout_kwargs(request_timeout))
            except TypeError as exc:
                if "_request_timeout" not in str(exc):
                    raise
                timeout_value = request_timeout[1] if isinstance(request_timeout, tuple) else request_timeout
                try:
                    return method(*args, **kwargs, timeout=timeout_value)
                except TypeError as timeout_exc:
                    if "timeout" not in str(timeout_exc):
                        raise
                    return method(*args, **kwargs)
        except Exception as e:
            e_str = str(e).lower()
            is_transient = any(
                k in e_str
                for k in [
                    "429",
                    "502",
                    "503",
                    "504",
                    "rate limit",
                    "timeout",
                    "timed out",
                    "write operation timed out",
                    "connection refused",
                    "temporary",
                    "busy",
                    "nodename nor servname",
                    "name or service not known",
                    "temporary failure in name resolution",
                    "getaddrinfo",
                    "dns",
                ]
            )

            if not is_transient or attempt >= max_retries:
                logger.error("Pinecone API call failed permanently: %s", e)
                raise e

            jitter = random.uniform(0.5, 1.5)
            sleep_time = backoff * jitter
            logger.warning(
                "Transient Pinecone failure: %s. Retrying in %.2fs (attempt %d/%d)...",
                e, sleep_time, attempt + 1, max_retries
            )
            time.sleep(sleep_time)
            backoff *= 2.0


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


def _namespace_vector_counts(index: Any) -> Dict[str, int]:
    describe = getattr(index, "describe_index_stats", None)
    if not callable(describe):
        raise RuntimeError("Pinecone index handle does not support describe_index_stats")
    stats = describe()
    namespaces = stats.get("namespaces") if isinstance(stats, dict) else getattr(stats, "namespaces", None)
    counts: Dict[str, int] = {}
    if not isinstance(namespaces, dict):
        return counts
    for namespace, payload in namespaces.items():
        if isinstance(payload, dict):
            count = payload.get("vector_count") or payload.get("record_count") or 0
        else:
            count = getattr(payload, "vector_count", None) or getattr(payload, "record_count", None) or 0
        try:
            counts[str(namespace)] = int(count)
        except (TypeError, ValueError):
            counts[str(namespace)] = 0
    return counts


def _verify_namespace_counts(
    *,
    index: Any,
    expected: Dict[str, int],
    min_count_only: bool,
) -> Dict[str, Any]:
    actual = _namespace_vector_counts(index)
    failures: List[Dict[str, Any]] = []
    for namespace, expected_count in expected.items():
        expected_int = int(expected_count or 0)
        actual_int = int(actual.get(namespace) or 0)
        if expected_int == 0:
            continue
        mismatch = actual_int < expected_int if min_count_only else actual_int != expected_int
        if mismatch:
            failures.append(
                {
                    "namespace": namespace,
                    "expected": expected_int,
                    "actual": actual_int,
                }
            )
    return {"actual": actual, "expected": expected, "failures": failures}


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
        extra_kwargs: Dict[str, Any] = {}
        if str(os.environ.get("PINECONE_TRANSPORT") or "").strip().lower() == "grpc":
            grpc_batch_size = int(os.environ.get("PINECONE_GRPC_UPSERT_BATCH_SIZE") or min(32, max(1, batch_size)))
            grpc_max_concurrency = int(os.environ.get("PINECONE_GRPC_MAX_CONCURRENCY") or 4)
            extra_kwargs = {
                "batch_size": max(1, grpc_batch_size),
                "max_concurrency": max(1, grpc_max_concurrency),
                "show_progress": False,
            }
        _call_with_optional_pinecone_timeout(
            index.upsert,
            vectors=payload,
            namespace=namespace,
            request_timeout=request_timeout,
            **extra_kwargs,
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
    knowledge_graph_sha256: str = "",
    upload_input_sha256: str = "",
    retrieval_bundle_stats: Dict[str, int] | None = None,
    media_metrics: Dict[str, int] | None = None,
    record_fingerprints: Dict[str, str] | None = None,
) -> None:
    if record_fingerprints is None:
        previous = load_json_safe(path, {}) or {}
        if isinstance(previous, dict) and isinstance(previous.get("record_fingerprints"), dict):
            record_fingerprints = dict(previous.get("record_fingerprints") or {})
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
            "knowledge_graph_sha256": knowledge_graph_sha256,
            "upload_input_sha256": upload_input_sha256 or retrieval_bundle_sha256,
            "record_fingerprints": record_fingerprints or {},
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
            text = _record_identity_text(record)
        if not text:
            continue
        text = _add_type_context_for_weak_text(record, text)
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
    try:
        from pinecone.data.request_factory import IndexRequestFactory
    except ModuleNotFoundError:  # pragma: no cover - depends on Pinecone SDK version
        IndexRequestFactory = None

    uploaded = 0
    for batch in _iter_batches(records, batch_size):
        payload = list(batch)
        if not payload:
            continue
        if hasattr(index, "upsert_records"):
            _call_with_optional_pinecone_timeout(
                index.upsert_records,
                namespace=namespace,
                records=payload,
                request_timeout=request_timeout,
            )
        elif hasattr(index, "_vector_api"):
            if IndexRequestFactory is not None:
                args = IndexRequestFactory.upsert_records_args(namespace=namespace, records=payload)
            else:
                args = {"namespace": namespace, "records": payload}
            _call_with_optional_pinecone_timeout(
                index._vector_api.upsert_records_namespace,
                request_timeout=request_timeout,
                **args,
            )
        else:
            raise RuntimeError("Pinecone sparse index handle does not expose upsert_records")
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
        if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
            errors.append("GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini embeddings")
        if not os.getenv("PINECONE_API_KEY"):
            errors.append("PINECONE_API_KEY is required")
        if not emb_cfg.get("pinecone_index"):
            errors.append("embedder.pinecone_index is required")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.embedder_config
        input_paths = _resolve_indexing_input_paths(ctx)
        chunk_file = input_paths["chunks"]
        parent_file = input_paths["parents"]
        media_file = input_paths["media"]
        fact_file = input_paths["facts"]
        evidence_span_file = input_paths["evidence_spans"]
        summary_file = input_paths["summaries"]
        assertion_file = input_paths["assertions"]
        entity_file = input_paths["entities"]
        community_file = input_paths["communities"]
        bundle_file = input_paths["bundle"]
        lexical_corpus_file = input_paths["lexical_corpus"]
        promoted_assertions_file = input_paths["promoted_assertions"]

        if not chunk_file or not bundle_file:
            return StageResult.failure("retrieval formatter outputs are required before Gemini indexing")
        production = bool((ctx.config.get("pipeline") or {}).get("production_profile", False))
        if production and not lexical_corpus_file:
            return StageResult.failure("Production indexing requires the lexical retrieval corpus sidecar")
        if production and not promoted_assertions_file:
            return StageResult.failure("Production indexing requires the promoted assertions sidecar")

        chunk_records = _load_records(chunk_file)
        parent_records = _load_records(parent_file) if parent_file else []
        media_records = _load_records(media_file) if media_file else []
        fact_records = _load_records(fact_file) if fact_file else []
        evidence_span_records = _load_records(evidence_span_file) if evidence_span_file else []
        summary_records = _load_records(summary_file) if summary_file else []
        assertion_records = _load_records(assertion_file) if assertion_file else []
        entity_records = [_normalise_entity_record(record) for record in _load_records(entity_file)] if entity_file else []
        community_records = _load_records(community_file) if community_file else []

        # Load graph bundle if available to embed entities and communities.
        # Prefer formatter-produced entity records because they carry canonical
        # names, aliases, descriptions, and provenance in a retrieval-safe shape.
        graph_bundle_file = input_paths["graph_bundle"]
        load_entities_from_graph = not entity_records
        load_communities_from_graph = not community_records
        if graph_bundle_file and (load_entities_from_graph or load_communities_from_graph):
            graph_bundle = load_json_safe(graph_bundle_file, {}) or {}
            nodes = graph_bundle.get("nodes") or []
            for node in nodes:
                node_type = node.get("node_type")
                if node_type == "entity" and load_entities_from_graph:
                    entity_records.append(_graph_entity_record(node))
                elif node_type in {"community", "community_summary"} and load_communities_from_graph:
                    properties = node.get("properties") if isinstance(node.get("properties"), dict) else {}
                    title = properties.get("title") or node.get("title") or node.get("label") or ""
                    summary = properties.get("summary") or node.get("summary") or ""
                    community_id = properties.get("community_id")
                    if community_id in (None, ""):
                        community_id = node.get("community_id")
                    if community_id in (None, ""):
                        community_id = node.get("id")
                    community_records.append({
                        "id": community_id,
                        "text": f"{title}\n{summary}".strip(),
                        "title": title,
                        "summary": summary,
                        "community_id": community_id,
                        "size": properties.get("size"),
                        "level": properties.get("level"),
                    })

        chunk_records = _dedupe_records_by_id(chunk_records, record_kind="chunk")
        parent_records = _dedupe_records_by_id(parent_records, record_kind="parent")
        media_records = _dedupe_records_by_id(media_records, record_kind="media")
        fact_records = _dedupe_records_by_id(fact_records, record_kind="fact")
        evidence_span_records = _dedupe_records_by_id(evidence_span_records, record_kind="evidence_span")
        summary_records = _dedupe_records_by_id(summary_records, record_kind="summary")
        assertion_records = _dedupe_records_by_id(assertion_records, record_kind="assertion")
        entity_records = _dedupe_records_by_id(entity_records, record_kind="entity")
        community_records = _dedupe_records_by_id(community_records, record_kind="community")

        if not chunk_records:
            return StageResult.failure("No chunk embedding records found")
        retrieval_bundle_sha256 = sha256_file(bundle_file)
        lexical_corpus_sha256 = sha256_file(lexical_corpus_file) if lexical_corpus_file else ""
        promoted_assertions_sha256 = (
            sha256_file(promoted_assertions_file) if promoted_assertions_file else ""
        )
        knowledge_graph_sha256 = sha256_file(graph_bundle_file) if graph_bundle_file else ""
        canonical_graph = resolve_canonical_graph_artifacts(
            ctx.work_dir,
            required=False,
            require_index=True,
            validate_binding=production,
        )
        knowledge_graph_kind = ""
        knowledge_graph_index_file = ""
        knowledge_graph_index_sha256 = ""
        if canonical_graph is not None and graph_bundle_file:
            if canonical_graph.graph_file != Path(graph_bundle_file).expanduser().resolve():
                raise ValueError(
                    "Resolved graph upload input does not match the canonical graph artifact: "
                    f"input={graph_bundle_file}, canonical={canonical_graph.graph_file}"
                )
            knowledge_graph_kind = canonical_graph.kind
            knowledge_graph_index_file = str(canonical_graph.index_file or "")
            knowledge_graph_index_sha256 = canonical_graph.index_sha256
        elif graph_bundle_file:
            knowledge_graph_kind = "legacy_external_graph"
        current_upload_contract = bool(
            lexical_corpus_sha256 and promoted_assertions_sha256
        )
        resolved_snapshot = load_json_safe(ctx.work_dir / "resolved_config.json", {}) or {}
        indexing_build = (
            dict(resolved_snapshot.get("indexing_build") or {})
            if isinstance(resolved_snapshot, dict)
            else {}
        )
        indexing_commit = str(indexing_build.get("commit_sha") or "").strip().lower()
        indexing_build_sha256 = hashlib.sha256(
            json.dumps(indexing_build, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest() if indexing_build else ""
        if production and (
            not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", indexing_commit)
            or indexing_build.get("dirty") is not False
            or not isinstance(indexing_build.get("implementation_sha256"), dict)
            or not indexing_build.get("implementation_sha256")
        ):
            return StageResult.failure(
                "Production upload requires a clean, full-commit indexing_build identity "
                "in resolved_config.json"
            )
        if current_upload_contract:
            upload_input_sha256 = combine_sha256_digests(
                retrieval_bundle_sha256,
                lexical_corpus_sha256,
                promoted_assertions_sha256,
                knowledge_graph_sha256,
                knowledge_graph_index_sha256,
            )
        elif graph_bundle_file:
            upload_input_sha256 = combine_sha256_digests(
                retrieval_bundle_sha256,
                knowledge_graph_sha256,
            )
        else:
            upload_input_sha256 = retrieval_bundle_sha256

        bundle_stats = {
            "chunks": len(chunk_records),
            "parents": len(parent_records),
            "media": len(media_records),
            "facts": len(fact_records),
            "evidence_spans": len(evidence_span_records),
            "summaries": len(summary_records),
            "assertions": len(assertion_records),
            "entities": len(entity_records),
            "communities": len(community_records),
        }
        record_fingerprints = {
            "entities": _record_text_fingerprint(entity_records),
            "communities": _record_text_fingerprint(community_records),
        }

        model = str(config.get("model") or "gemini-embedding-2")
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
        gemini_request_timeout_ms = int(config.get("gemini_request_timeout_ms") or 120000)
        pinecone_connect_timeout_sec = float(config.get("pinecone_connect_timeout_sec") or 10.0)
        pinecone_read_timeout_sec = float(config.get("pinecone_read_timeout_sec") or 120.0)
        pinecone_request_timeout: Tuple[float, float] = (
            pinecone_connect_timeout_sec,
            pinecone_read_timeout_sec,
        )
        progress_flush_every_batches = max(1, int(config.get("progress_flush_every_batches") or 1))
        resolved_namespaces = _resolve_upload_namespaces(config, run_id=ctx.run_id)
        namespace_chunks = resolved_namespaces["chunks"]
        namespace_parents = resolved_namespaces["parents"]
        namespace_media = resolved_namespaces["media"]
        namespace_facts = resolved_namespaces["facts"]
        namespace_evidence_spans = resolved_namespaces["evidence_spans"]
        namespace_summaries = resolved_namespaces["summaries"]
        namespace_assertions = resolved_namespaces["assertions"]
        namespace_entities = resolved_namespaces["entities"]
        namespace_communities = resolved_namespaces["communities"]
        enable_dense_facts = bool(config.get("enable_dense_facts", False))
        dense_fact_records = fact_records if enable_dense_facts else []
        enable_dense_evidence_spans = bool(config.get("enable_dense_evidence_spans", True))
        dense_evidence_span_records = evidence_span_records if enable_dense_evidence_spans else []
        enable_dense_assertions = bool(config.get("enable_dense_assertions", True))
        dense_assertion_records = assertion_records if enable_dense_assertions else []
        sparse_enabled = bool(config.get("enable_sparse", True))
        enable_sparse_evidence_spans = bool(config.get("enable_sparse_evidence_spans", True))
        sparse_index_name = str(config.get("pinecone_sparse_index") or f"{pinecone_index}-sparse").strip()
        sparse_model = str(config.get("sparse_model") or "pinecone-sparse-english-v0")
        sparse_text_field = str(config.get("sparse_text_field") or "chunk_text")
        sparse_upsert_batch_size = int(config.get("sparse_upsert_batch_size") or upsert_batch_size)
        sparse_max_text_chars = int(config.get("sparse_max_text_chars") or 12000)
        sparse_max_record_bytes = int(config.get("sparse_max_record_bytes") or 24000)
        clear_dense_namespace_on_zero_progress = bool(config.get("clear_dense_namespace_on_zero_progress", True))
        clear_sparse_namespace_on_zero_progress = bool(config.get("clear_sparse_namespace_on_zero_progress", True))
        verify_index_after_upload = bool(config.get("verify_index_after_upload", False))
        verify_index_min_count_only = bool(config.get("verify_index_min_count_only", False))
        progress_path = ctx.stage_work_dir / "index_upload_progress.json"
        dense_namespace_reset_required = False
        sparse_namespace_reset_required = False
        totals = {
            "chunks": len(chunk_records),
            "parents": len(parent_records),
            "media": len(media_records),
            "facts": len(dense_fact_records),
            "evidence_spans": len(dense_evidence_span_records),
            "summaries": len(summary_records),
            "assertions": len(dense_assertion_records),
            "entities": len(entity_records),
            "communities": len(community_records),
            "sparse_chunks": len(chunk_records) if sparse_enabled else 0,
            "sparse_parents": len(parent_records) if sparse_enabled else 0,
            "sparse_media": len(media_records) if sparse_enabled else 0,
            "sparse_facts": len(fact_records) if sparse_enabled else 0,
            "sparse_evidence_spans": len(evidence_span_records) if sparse_enabled and enable_sparse_evidence_spans else 0,
            "sparse_summaries": len(summary_records) if sparse_enabled else 0,
            "sparse_assertions": len(assertion_records) if sparse_enabled else 0,
            "sparse_entities": len(entity_records) if sparse_enabled else 0,
            "sparse_communities": len(community_records) if sparse_enabled else 0,
        }
        previous_progress = load_json_safe(progress_path, {}) or {}
        previous_input_sha256 = ""
        if isinstance(previous_progress, dict):
            previous_input_sha256 = str(
                previous_progress.get("upload_input_sha256")
                or previous_progress.get("retrieval_bundle_sha256")
                or ""
            )
        bundle_changed = (
            isinstance(previous_progress, dict)
            and bool(previous_input_sha256)
            and previous_input_sha256 != upload_input_sha256
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
        previous_fingerprints = (
            previous_progress.get("record_fingerprints")
            if isinstance(previous_progress, dict) and isinstance(previous_progress.get("record_fingerprints"), dict)
            else {}
        )
        if (
            not bundle_changed
            and record_fingerprints["entities"]
            and uploaded.get("entities")
            and previous_fingerprints.get("entities") != record_fingerprints["entities"]
        ):
            logger.warning("Gemini embedder detected entity text fingerprint drift; rebuilding entity namespaces")
            uploaded["entities"] = 0
            uploaded["sparse_entities"] = 0
        sparse_stats = {
            "chunk_records_trimmed": 0,
            "chunk_records_skipped": 0,
            "parent_records_trimmed": 0,
            "parent_records_skipped": 0,
            "media_records_trimmed": 0,
            "media_records_skipped": 0,
            "fact_records_trimmed": 0,
            "fact_records_skipped": 0,
            "evidence_span_records_trimmed": 0,
            "evidence_span_records_skipped": 0,
            "summary_records_trimmed": 0,
            "summary_records_skipped": 0,
            "assertion_records_trimmed": 0,
            "assertion_records_skipped": 0,
            "entity_records_trimmed": 0,
            "entity_records_skipped": 0,
            "community_records_trimmed": 0,
            "community_records_skipped": 0,
        }

        client = _make_gemini_client(request_timeout_ms=gemini_request_timeout_ms)

        logger.info(
            "Gemini embedder: chunks=%d parents=%d media=%d facts=%d evidence_spans=%d summaries=%d assertions=%d entities=%d model=%s dims=%d",
            len(chunk_records),
            len(parent_records),
            len(media_records),
            len(dense_fact_records),
            len(dense_evidence_span_records),
            len(summary_records),
            len(dense_assertion_records),
            len(entity_records),
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
            and uploaded["evidence_spans"] == 0
            and uploaded["summaries"] == 0
            and uploaded["assertions"] == 0
            and uploaded["entities"] == 0
            and uploaded["communities"] == 0
                )
        if dense_namespace_reset_required:
            for namespace in (namespace_chunks, namespace_parents, namespace_media, namespace_facts, namespace_evidence_spans, namespace_summaries, namespace_assertions, namespace_entities, namespace_communities):
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
                for key in ("sparse_chunks", "sparse_parents", "sparse_media", "sparse_facts", "sparse_evidence_spans", "sparse_summaries", "sparse_assertions", "sparse_entities", "sparse_communities"):
                    uploaded[key] = 0
            sparse_namespace_reset_required = bundle_changed or (
                clear_sparse_namespace_on_zero_progress
                and uploaded["sparse_chunks"] == 0
                and uploaded["sparse_parents"] == 0
                and uploaded["sparse_media"] == 0
                and uploaded["sparse_facts"] == 0
                and uploaded["sparse_evidence_spans"] == 0
                and uploaded["sparse_summaries"] == 0
                and uploaded["sparse_assertions"] == 0
                and uploaded["sparse_entities"] == 0
                and uploaded["sparse_communities"] == 0
            )
            if sparse_namespace_reset_required:
                for namespace in (namespace_chunks, namespace_parents, namespace_media, namespace_facts, namespace_evidence_spans, namespace_summaries, namespace_assertions, namespace_entities, namespace_communities):
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
            retrieval_bundle_sha256=retrieval_bundle_sha256,
            knowledge_graph_sha256=knowledge_graph_sha256,
            upload_input_sha256=upload_input_sha256,
            record_fingerprints=record_fingerprints,
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
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
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
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
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
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["facts"] == totals["facts"]:
                logger.info("Gemini embedder progress: uploaded %d/%d fact vectors", uploaded["facts"], totals["facts"])

        remaining_evidence_span_records = dense_evidence_span_records[uploaded["evidence_spans"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_evidence_span_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_evidence_span_batch",
                lambda batch_list=batch_list: _embed_text_batch(
                    client,
                    model=model,
                    texts=[
                        str(
                            record.get("dense_text")
                            or record.get("embedding_text")
                            or record.get("text")
                            or ""
                        )
                        for record in batch_list
                    ],
                    task_type=task_type_document,
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            uploaded["evidence_spans"] += _call_with_retry(
                "upsert_evidence_span_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_evidence_spans,
                    records=batch_list,
                    vectors=vectors,
                    kind="evidence_span",
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
                phase="uploading_evidence_spans",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["evidence_spans"] == totals["evidence_spans"]:
                logger.info(
                    "Gemini embedder progress: uploaded %d/%d evidence span vectors",
                    uploaded["evidence_spans"],
                    totals["evidence_spans"],
                )

        remaining_summary_records = summary_records[uploaded["summaries"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_summary_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_summary_batch",
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
            uploaded["summaries"] += _call_with_retry(
                "upsert_summary_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_summaries,
                    records=batch_list,
                    vectors=vectors,
                    kind="summary",
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
                phase="uploading_summaries",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["summaries"] == totals["summaries"]:
                logger.info(
                    "Gemini embedder progress: uploaded %d/%d summary vectors",
                    uploaded["summaries"],
                    totals["summaries"],
                )

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
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
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
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
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

        # Entities
        remaining_entity_records = entity_records[uploaded["entities"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_entity_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_entity_batch",
                lambda batch_list=batch_list: _embed_text_batch(
                    client,
                    model=model,
                    texts=[_record_embedding_text(record) for record in batch_list],
                    task_type=task_type_document,
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            uploaded["entities"] += _call_with_retry(
                "upsert_entity_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_entities,
                    records=batch_list,
                    vectors=vectors,
                    kind="entity",
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
                phase="uploading_entities",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["entities"] == totals["entities"]:
                logger.info(
                    "Gemini embedder progress: uploaded %d/%d entity vectors",
                    uploaded["entities"],
                    totals["entities"],
                )

        # Communities
        remaining_community_records = community_records[uploaded["communities"] :]
        for batch_number, batch in enumerate(_iter_batches(remaining_community_records, text_batch_size), start=1):
            batch_list = list(batch)
            vectors = _call_with_retry(
                "embed_community_batch",
                lambda batch_list=batch_list: _embed_text_batch(
                    client,
                    model=model,
                    texts=[str(record.get("text") or "") for record in batch_list],
                    task_type=task_type_document,
                    output_dimensionality=output_dimensionality,
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            uploaded["communities"] += _call_with_retry(
                "upsert_community_batch",
                lambda batch_list=batch_list, vectors=vectors: _upsert_namespace(
                    _make_index_handle(pinecone_index),
                    namespace=namespace_communities,
                    records=batch_list,
                    vectors=vectors,
                    kind="community",
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
                phase="uploading_communities",
                retrieval_bundle_file=str(bundle_file),
                retrieval_bundle_sha256=retrieval_bundle_sha256,
                knowledge_graph_sha256=knowledge_graph_sha256,
                upload_input_sha256=upload_input_sha256,
                retrieval_bundle_stats=bundle_stats,
                media_metrics=media_metrics,
            )
            if batch_number == 1 or batch_number % progress_flush_every_batches == 0 or uploaded["communities"] == totals["communities"]:
                logger.info(
                    "Gemini embedder progress: uploaded %d/%d community vectors",
                    uploaded["communities"],
                    totals["communities"],
                )

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
            if enable_sparse_evidence_spans:
                sparse_evidence_spans, evidence_span_sparse_stats = _build_sparse_records(
                    evidence_span_records,
                    sparse_text_field=sparse_text_field,
                    max_text_chars=sparse_max_text_chars,
                    max_record_bytes=sparse_max_record_bytes,
                )
            else:
                sparse_evidence_spans = []
                evidence_span_sparse_stats = {"trimmed": 0, "skipped": 0}
            sparse_summaries, summary_sparse_stats = _build_sparse_records(
                summary_records,
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
            sparse_entities, entity_sparse_stats = _build_sparse_records(
                entity_records,
                sparse_text_field=sparse_text_field,
                max_text_chars=sparse_max_text_chars,
                max_record_bytes=sparse_max_record_bytes,
            )
            sparse_communities, community_sparse_stats = _build_sparse_records(
                community_records,
                sparse_text_field=sparse_text_field,
                max_text_chars=sparse_max_text_chars,
                max_record_bytes=sparse_max_record_bytes,
            )
            totals.update(
                {
                    "sparse_chunks": len(sparse_chunks),
                    "sparse_parents": len(sparse_parents),
                    "sparse_media": len(sparse_media),
                    "sparse_facts": len(sparse_facts),
                    "sparse_evidence_spans": len(sparse_evidence_spans),
                    "sparse_summaries": len(sparse_summaries),
                    "sparse_assertions": len(sparse_assertions),
                    "sparse_entities": len(sparse_entities),
                    "sparse_communities": len(sparse_communities),
                }
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
                "evidence_span_records_trimmed": evidence_span_sparse_stats["trimmed"],
                "evidence_span_records_skipped": evidence_span_sparse_stats["skipped"],
                "summary_records_trimmed": summary_sparse_stats["trimmed"],
                "summary_records_skipped": summary_sparse_stats["skipped"],
                "assertion_records_trimmed": assertion_sparse_stats["trimmed"],
                "assertion_records_skipped": assertion_sparse_stats["skipped"],
                "entity_records_trimmed": entity_sparse_stats["trimmed"],
                "entity_records_skipped": entity_sparse_stats["skipped"],
                "community_records_trimmed": community_sparse_stats["trimmed"],
                "community_records_skipped": community_sparse_stats["skipped"],
            }

            sparse_plan = [
                ("sparse_chunks", namespace_chunks, sparse_chunks),
                ("sparse_parents", namespace_parents, sparse_parents),
                ("sparse_media", namespace_media, sparse_media),
                ("sparse_facts", namespace_facts, sparse_facts),
                ("sparse_evidence_spans", namespace_evidence_spans, sparse_evidence_spans),
                ("sparse_summaries", namespace_summaries, sparse_summaries),
                ("sparse_assertions", namespace_assertions, sparse_assertions),
                ("sparse_entities", namespace_entities, sparse_entities),
                ("sparse_communities", namespace_communities, sparse_communities),
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
                    retrieval_bundle_sha256=retrieval_bundle_sha256,
                    knowledge_graph_sha256=knowledge_graph_sha256,
                    upload_input_sha256=upload_input_sha256,
                    retrieval_bundle_stats=bundle_stats,
                    media_metrics=media_metrics,
                )
            sparse_manifest["record_stats"] = sparse_stats

        _assert_upload_plan_complete(totals, uploaded)

        verification: Dict[str, Any] = {}
        if verify_index_after_upload:
            def _verified_report(label: str, report: Dict[str, Any]) -> Dict[str, Any]:
                failures = list(report.get("failures") or [])
                if failures:
                    raise RuntimeError(f"{label} namespace count verification pending: {failures}")
                return report

            dense_expected = {
                namespace_chunks: uploaded["chunks"],
                namespace_parents: uploaded["parents"],
                namespace_media: uploaded["media"],
                namespace_facts: uploaded["facts"],
                namespace_evidence_spans: uploaded["evidence_spans"],
                namespace_summaries: uploaded["summaries"],
                namespace_assertions: uploaded["assertions"],
                namespace_entities: uploaded["entities"],
                namespace_communities: uploaded["communities"],
            }
            dense_expected = {namespace: count for namespace, count in dense_expected.items() if count}
            verification["dense"] = _call_with_retry(
                "verify_dense_index_counts",
                lambda: _verified_report(
                    "dense",
                    _verify_namespace_counts(
                        index=_make_index_handle(pinecone_index),
                        expected=dense_expected,
                        min_count_only=verify_index_min_count_only,
                    ),
                ),
                max_attempts=max_retries,
                base_delay_sec=retry_base_delay_sec,
                max_delay_sec=retry_max_delay_sec,
            )
            if sparse_enabled:
                sparse_expected = {
                    namespace_chunks: uploaded["sparse_chunks"],
                    namespace_parents: uploaded["sparse_parents"],
                    namespace_media: uploaded["sparse_media"],
                    namespace_facts: uploaded["sparse_facts"],
                    namespace_evidence_spans: uploaded["sparse_evidence_spans"],
                    namespace_summaries: uploaded["sparse_summaries"],
                    namespace_assertions: uploaded["sparse_assertions"],
                    namespace_entities: uploaded["sparse_entities"],
                    namespace_communities: uploaded["sparse_communities"],
                }
                sparse_expected = {namespace: count for namespace, count in sparse_expected.items() if count}
                verification["sparse"] = _call_with_retry(
                    "verify_sparse_index_counts",
                    lambda: _verified_report(
                        "sparse",
                        _verify_namespace_counts(
                            index=_make_index_handle(sparse_index_name),
                            expected=sparse_expected,
                            min_count_only=verify_index_min_count_only,
                        ),
                    ),
                    max_attempts=max_retries,
                    base_delay_sec=retry_base_delay_sec,
                    max_delay_sec=retry_max_delay_sec,
                )

        upload_manifest = {
            "schema_version": 4 if current_upload_contract else 3,
            "indexing_build": indexing_build,
            "indexing_build_sha256": indexing_build_sha256,
            "index_name": pinecone_index,
            "sparse_index_name": sparse_index_name if sparse_enabled else "",
            "model": model,
            "output_dimensionality": output_dimensionality,
            "namespaces": {
                "chunks": namespace_chunks,
                "parents": namespace_parents,
                "media": namespace_media,
                "facts": namespace_facts,
                "evidence_spans": namespace_evidence_spans,
                "summaries": namespace_summaries,
                "assertions": namespace_assertions,
                "entities": namespace_entities,
                "communities": namespace_communities,
            },
            "namespace_strategy": str(config.get("namespace_strategy") or "static").strip().lower(),
            "namespace_release_id": ctx.run_id if str(config.get("namespace_strategy") or "static").strip().lower() == "release" else "",
            "planned": {key: int(value) for key, value in totals.items()},
            "uploaded": {
                "chunks": uploaded["chunks"],
                "parents": uploaded["parents"],
                "media": uploaded["media"],
                "facts": uploaded["facts"],
                "evidence_spans": uploaded["evidence_spans"],
                "summaries": uploaded["summaries"],
                "assertions": uploaded["assertions"],
                "entities": uploaded["entities"],
                "communities": uploaded["communities"],
                "sparse_chunks": uploaded["sparse_chunks"],
                "sparse_parents": uploaded["sparse_parents"],
                "sparse_media": uploaded["sparse_media"],
                "sparse_facts": uploaded["sparse_facts"],
                "sparse_evidence_spans": uploaded["sparse_evidence_spans"],
                "sparse_summaries": uploaded["sparse_summaries"],
                "sparse_assertions": uploaded["sparse_assertions"],
                "sparse_entities": uploaded["sparse_entities"],
                "sparse_communities": uploaded["sparse_communities"],
            },
            "bundle_version": 5,
            "evidence_span_count": uploaded["evidence_spans"],
            "sparse_evidence_span_count": uploaded["sparse_evidence_spans"],
            "retrieval_bundle_file": bundle_file,
            "retrieval_bundle_sha256": retrieval_bundle_sha256,
            "lexical_corpus_file": lexical_corpus_file,
            "lexical_corpus_sha256": lexical_corpus_sha256,
            "promoted_assertions_file": promoted_assertions_file,
            "promoted_assertions_sha256": promoted_assertions_sha256,
            "knowledge_graph_file": str(Path(graph_bundle_file).expanduser().resolve()) if graph_bundle_file else "",
            "knowledge_graph_kind": knowledge_graph_kind,
            "knowledge_graph_sha256": knowledge_graph_sha256,
            "knowledge_graph_index_file": knowledge_graph_index_file,
            "knowledge_graph_index_sha256": knowledge_graph_index_sha256,
            "upload_input_sha256": upload_input_sha256,
            "record_fingerprints": record_fingerprints,
            "retrieval_bundle_stats": bundle_stats,
            "media_metrics": media_metrics,
            "sparse": sparse_manifest,
            "verification": verification,
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
            retrieval_bundle_sha256=retrieval_bundle_sha256,
            knowledge_graph_sha256=knowledge_graph_sha256,
            upload_input_sha256=upload_input_sha256,
            retrieval_bundle_stats=bundle_stats,
            media_metrics=media_metrics,
        )

        dense_vectors_uploaded = (
            uploaded["chunks"]
            + uploaded["parents"]
            + uploaded["media"]
            + uploaded["facts"]
            + uploaded["evidence_spans"]
            + uploaded["summaries"]
            + uploaded["assertions"]
            + uploaded["entities"]
            + uploaded["communities"]
        )

        return StageResult.success(
            outputs={
                "vectors_uploaded": dense_vectors_uploaded,
                "index_name": pinecone_index,
                "sparse_index_name": sparse_index_name if sparse_enabled else "",
                "index_manifest_file": str(manifest_path),
            },
            metrics={
                "vectors_uploaded": dense_vectors_uploaded,
                "chunk_vectors_uploaded": uploaded["chunks"],
                "parent_vectors_uploaded": uploaded["parents"],
                "media_vectors_uploaded": uploaded["media"],
                "fact_vectors_uploaded": uploaded["facts"],
                "evidence_span_vectors_uploaded": uploaded["evidence_spans"],
                "summary_vectors_uploaded": uploaded["summaries"],
                "assertion_vectors_uploaded": uploaded["assertions"],
                "entity_vectors_uploaded": uploaded["entities"],
                "community_vectors_uploaded": uploaded["communities"],
                "sparse_chunk_records_uploaded": uploaded["sparse_chunks"],
                "sparse_parent_records_uploaded": uploaded["sparse_parents"],
                "sparse_media_records_uploaded": uploaded["sparse_media"],
                "sparse_fact_records_uploaded": uploaded["sparse_facts"],
                "sparse_evidence_span_records_uploaded": uploaded["sparse_evidence_spans"],
                "sparse_summary_records_uploaded": uploaded["sparse_summaries"],
                "sparse_assertion_records_uploaded": uploaded["sparse_assertions"],
                "sparse_entity_records_uploaded": uploaded["sparse_entities"],
                "sparse_community_records_uploaded": uploaded["sparse_communities"],
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
                        "vectors_uploaded": dense_vectors_uploaded,
                        "evidence_span_count": uploaded["evidence_spans"],
                        "sparse_evidence_span_count": uploaded["sparse_evidence_spans"],
                    },
                )
            ],
        )
