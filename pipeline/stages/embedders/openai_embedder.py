"""
OpenAI embeddings + Pinecone upload stage.

Generates dense embeddings using OpenAI's text-embedding model (or a
SentenceTransformer) and optionally sparse BM25 embeddings, then upserts
vectors to a Pinecone index.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.core.base import EmbedderStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


def _replace_md_links(text: str) -> str:
    """Replace markdown link URLs with placeholder to avoid noisy embeddings."""
    return re.sub(r"(\[.*?\])\((.*?)\)", r"\1(<URL>)", text)


def _compact_json_like(value: Any, *, max_items: int, max_string_len: int, depth: int = 0) -> Any:
    if depth > 4:
        return None
    if isinstance(value, str):
        return value[:max_string_len]
    if isinstance(value, list):
        return [
            _compact_json_like(item, max_items=max_items, max_string_len=max_string_len, depth=depth + 1)
            for item in value[:max_items]
        ]
    if isinstance(value, dict):
        compacted = {}
        for idx, (key, item) in enumerate(value.items()):
            if idx >= max_items:
                break
            compacted[str(key)] = _compact_json_like(
                item,
                max_items=max_items,
                max_string_len=max_string_len,
                depth=depth + 1,
            )
        return compacted
    return value


def _serialize_metadata_value(key: str, value: Any, max_len: int = 10000) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        return value[:max_len]
    if not isinstance(value, (dict, list)):
        return value

    max_items = 8 if key in {"media", "images", "videos"} else 20
    max_string_len = 320 if key in {"media", "images", "videos"} else 1000

    for _ in range(6):
        compact = _compact_json_like(
            value,
            max_items=max_items,
            max_string_len=max_string_len,
        )
        dumped = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        if len(dumped) <= max_len:
            return dumped
        max_items = max(1, max_items // 2)
        max_string_len = max(80, max_string_len // 2)

    fallback = [] if isinstance(value, list) else {}
    return json.dumps(fallback, separators=(",", ":"))


def _generate_openai_embeddings(
    texts: List[str], model: str, batch_size: int = 100
) -> List[List[float]]:
    """Generate embeddings using OpenAI API."""
    import openai

    client = openai.OpenAI()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
        logger.info("Embedded batch %d-%d/%d", i, i + len(batch), len(texts))

    return all_embeddings


def _generate_st_embeddings(
    texts: List[str], model_name: str, batch_size: int = 32
) -> List[List[float]]:
    """Generate embeddings using SentenceTransformer."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, trust_remote_code=True)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings = model.encode(batch, batch_size=batch_size)
        all_embeddings.extend(embeddings.tolist())
        logger.info("ST embedded batch %d-%d/%d", i, i + len(batch), len(texts))

    return all_embeddings


def _generate_bm25_sparse(texts: List[str], bm25_path: Optional[str] = None) -> List[Dict]:
    """Generate BM25 sparse embeddings."""
    from pinecone_text.sparse import BM25Encoder

    bm25 = BM25Encoder()
    if bm25_path and os.path.exists(bm25_path):
        bm25 = BM25Encoder().load(bm25_path)
    else:
        bm25.fit(texts)

    sparse = []
    for i in range(0, len(texts), 100):
        batch = bm25.encode_documents(texts[i : i + 100])
        sparse.extend(batch)
    return sparse


@register_stage
class OpenAIEmbedder(EmbedderStage):
    name = "openai_embedder"
    description = "Generates embeddings and uploads to Pinecone."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        emb_cfg = config.get("embedder", {})

        engine = emb_cfg.get("engine", "openai")
        if engine == "openai" and not os.getenv("OPENAI_API_KEY"):
            errors.append("OPENAI_API_KEY is required for OpenAI embeddings")

        if not os.getenv("PINECONE_API_KEY"):
            errors.append("PINECONE_API_KEY is required")

        if not emb_cfg.get("pinecone_index"):
            errors.append("embedder.pinecone_index is required")

        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.embedder_config

        formatted_file = ctx.previous_outputs.get("formatted_file")
        if not formatted_file:
            return StageResult.failure("No formatted_file in previous outputs")

        docs = load_json_safe(formatted_file)
        if not docs:
            return StageResult.failure(f"Cannot load formatted data from {formatted_file}")

        index_name = config.get("pinecone_index")
        if not index_name:
            return StageResult.failure("embedder.pinecone_index is required")

        engine = config.get("engine", "openai")
        model = config.get("model", "text-embedding-3-small")
        batch_size = config.get("batch_size", 100)
        use_sparse = config.get("use_sparse_embeddings", False)
        namespace = config.get("namespace", "")

        # Extract texts
        texts = [doc["text"] for doc in docs]
        cleaned_texts = [_replace_md_links(t) for t in texts]

        logger.info("Generating embeddings for %d documents (engine=%s, model=%s)",
                     len(texts), engine, model)

        # Dense embeddings
        if engine == "openai":
            dense = _generate_openai_embeddings(cleaned_texts, model, batch_size)
        elif engine == "sentence_transformer":
            dense = _generate_st_embeddings(cleaned_texts, model, batch_size)
        else:
            return StageResult.failure(f"Unknown embedding engine: {engine}")

        # Sparse embeddings (optional)
        sparse = None
        if use_sparse:
            bm25_path = config.get("bm25_model_path")
            sparse = _generate_bm25_sparse(cleaned_texts, bm25_path)

            # Save BM25 model for future use
            if not bm25_path:
                bm25_save_path = str(ctx.work_dir / "bm25_model.json")
                from pinecone_text.sparse import BM25Encoder
                bm25 = BM25Encoder()
                bm25.fit(cleaned_texts)
                bm25.dump(bm25_save_path)

        # Upload to Pinecone
        from pinecone import Pinecone

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index = pc.Index(index_name)

        vectors_uploaded = 0
        upsert_batch_size = config.get("upsert_batch_size", 100)

        for i in range(0, len(docs), upsert_batch_size):
            batch_docs = docs[i : i + upsert_batch_size]
            batch_dense = dense[i : i + upsert_batch_size]
            batch_sparse = sparse[i : i + upsert_batch_size] if sparse else [None] * len(batch_docs)

            vectors = []
            for doc, d_emb, s_emb in zip(batch_docs, batch_dense, batch_sparse):
                vec = {
                    "id": doc["id"],
                    "values": d_emb,
                    "metadata": {
                        k: _serialize_metadata_value(k, v)
                        for k, v in (doc.get("metadata") or {}).items()
                        if v is not None
                    },
                }

                if s_emb:
                    vec["sparse_values"] = s_emb

                vectors.append(vec)

            import time
            import random

            max_retries = 5
            backoff = 1.0

            for attempt in range(max_retries + 1):
                try:
                    index.upsert(vectors=vectors, namespace=namespace)
                    vectors_uploaded += len(vectors)
                    logger.info("Uploaded batch %d-%d/%d", i, i + len(vectors), len(docs))
                    break
                except Exception as e:
                    e_str = str(e).lower()
                    is_transient = any(
                        k in e_str
                        for k in ["429", "502", "503", "504", "rate limit", "timeout", "connection refused", "temporary", "busy"]
                    )

                    if not is_transient or attempt >= max_retries:
                        logger.error("Pinecone upsert failed permanently for batch %d: %s", i, e)
                        raise e

                    jitter = random.uniform(0.5, 1.5)
                    sleep_time = backoff * jitter
                    logger.warning(
                        "Transient Pinecone failure during batch %d: %s. Retrying in %.2fs (attempt %d/%d)...",
                        i, e, sleep_time, attempt + 1, max_retries
                    )
                    time.sleep(sleep_time)
                    backoff *= 2.0

        logger.info("Embedder done: %d vectors uploaded to %s", vectors_uploaded, index_name)

        return StageResult.success(
            outputs={
                "vectors_uploaded": vectors_uploaded,
                "index_name": index_name,
            },
            metrics={
                "vectors_uploaded": vectors_uploaded,
                "documents_total": len(docs),
                "engine": engine,
                "model": model,
            },
        )
