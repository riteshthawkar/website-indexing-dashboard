from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pipeline.core.config import load_effective_config
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.evaluation.benchmark_io import (
    _load_jsonl,
    evaluate_standard_rankings,
)
from pipeline.stages.embedders.gemini_pinecone_embedder import (
    _call_with_retry,
    _embed_text_batch,
    _make_gemini_client,
)

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover
    class BM25Okapi:  # type: ignore[override]
        def __init__(self, corpus: Sequence[Sequence[str]]):
            self.corpus = [list(doc or []) for doc in corpus]

        def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
            query_terms = set(query_tokens or [])
            if not query_terms:
                return [0.0 for _ in self.corpus]
            scores: List[float] = []
            for doc in self.corpus:
                doc_terms = set(doc)
                scores.append(len(query_terms & doc_terms) / float(len(query_terms)))
            return scores


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _tokenize(value: str) -> List[str]:
    return [token for token in _normalize_text(value).lower().split() if token]


def _compose_document_text(row: Dict[str, Any]) -> str:
    title = _normalize_text(row.get("title"))
    text = _normalize_text(row.get("text"))
    if title and text:
        return f"{title}\n\n{text}"
    return title or text


def _cosine_similarity(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if not lhs or not rhs or len(lhs) != len(rhs):
        return 0.0
    dot = 0.0
    lhs_norm = 0.0
    rhs_norm = 0.0
    for left, right in zip(lhs, rhs):
        left_f = float(left)
        right_f = float(right)
        dot += left_f * right_f
        lhs_norm += left_f * left_f
        rhs_norm += right_f * right_f
    if lhs_norm <= 0.0 or rhs_norm <= 0.0:
        return 0.0
    return dot / math.sqrt(lhs_norm * rhs_norm)


def _rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    rrf_k: int,
    top_k: int,
) -> List[str]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / float(rrf_k + rank))
    return [
        doc_id
        for doc_id, _score in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    ]


def _default_embedding_cache_path(dataset_dir: str | Path, name: str) -> Path:
    return Path(dataset_dir).resolve() / "cache" / f"{name}.json"


def _load_embedding_cache(path: str | Path | None) -> Tuple[Path | None, Dict[str, Dict[str, Any]]]:
    if not path:
        return None, {}
    cache_path = Path(path).resolve()
    payload = load_json_safe(cache_path, default={})
    if not isinstance(payload, dict):
        return cache_path, {}
    cache: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            cache[key] = dict(value)
    return cache_path, cache


def _text_digest(text: str) -> str:
    return hashlib.sha1(_normalize_text(text).encode("utf-8")).hexdigest()


def _cache_lookup(cache: Dict[str, Dict[str, Any]], key: str, text: str) -> List[float] | None:
    payload = cache.get(key)
    if not isinstance(payload, dict):
        return None
    if str(payload.get("text_sha1") or "") != _text_digest(text):
        return None
    values = payload.get("values")
    if not isinstance(values, list):
        return None
    try:
        return [float(value) for value in values]
    except Exception:
        return None


def _cache_store(
    cache: Dict[str, Dict[str, Any]],
    *,
    key: str,
    text: str,
    values: Sequence[float],
) -> None:
    cache[key] = {
        "text_sha1": _text_digest(text),
        "values": [float(value) for value in values],
    }


def _embed_with_cache(
    *,
    cache_file: Path | None,
    cache: Dict[str, Dict[str, Any]],
    rows: Sequence[Tuple[str, str]],
    model: str,
    output_dimensionality: int | None,
    task_type: str,
    batch_size: int,
) -> Tuple[Dict[str, List[float]], Dict[str, int]]:
    client = _make_gemini_client()
    vectors: Dict[str, List[float]] = {}
    hit_count = 0
    miss_count = 0
    missing: List[Tuple[str, str]] = []
    for item_id, text in rows:
        cached = _cache_lookup(cache, item_id, text)
        if cached is not None:
            vectors[item_id] = cached
            hit_count += 1
            continue
        missing.append((item_id, text))
    for start in range(0, len(missing), max(1, int(batch_size))):
        batch = missing[start : start + max(1, int(batch_size))]
        texts = [text for _item_id, text in batch]
        embeddings = _call_with_retry(
            "benchmark_embed",
            lambda texts=texts: _embed_text_batch(
                client,
                model=model,
                texts=texts,
                task_type=task_type,
                output_dimensionality=output_dimensionality,
            ),
            max_attempts=5,
            base_delay_sec=2.0,
            max_delay_sec=20.0,
        )
        for (item_id, text), embedding in zip(batch, embeddings):
            vectors[item_id] = list(embedding)
            _cache_store(cache, key=item_id, text=text, values=embedding)
            miss_count += 1
        if cache_file is not None:
            atomic_write_json(cache_file, cache)
    return vectors, {"hit_count": hit_count, "miss_count": miss_count, "size": len(cache)}


def run_standard_benchmark_retrieval(
    *,
    config_name: str,
    work_dir: str | Path | None = None,
    dataset_dir: str | Path,
    output_rankings_path: str | Path,
    top_k: int = 10,
    dense_top_k: int = 100,
    sparse_top_k: int = 100,
    rrf_k: int = 60,
    doc_cache_path: str | Path | None = None,
    query_cache_path: str | Path | None = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir).resolve()
    corpus_rows = _load_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = _load_jsonl(dataset_dir / "queries.jsonl")

    config = load_effective_config(config_name, work_dir=work_dir)
    embed_cfg = dict(config.get("embedder") or {})
    model = str(embed_cfg.get("model") or "gemini-embedding-2-preview")
    output_dimensionality = int(embed_cfg.get("output_dimensionality") or 1536)

    doc_cache_file, doc_cache = _load_embedding_cache(
        doc_cache_path or _default_embedding_cache_path(dataset_dir, "document_embeddings")
    )
    query_cache_file, query_cache = _load_embedding_cache(
        query_cache_path or _default_embedding_cache_path(dataset_dir, "query_embeddings")
    )

    document_rows = [
        (str(row.get("id") or "").strip(), _compose_document_text(row))
        for row in corpus_rows
        if str(row.get("id") or "").strip() and _compose_document_text(row)
    ]
    query_text_rows = [
        (str(row.get("id") or "").strip(), _normalize_text(row.get("text")))
        for row in query_rows
        if str(row.get("id") or "").strip() and _normalize_text(row.get("text"))
    ]

    document_vectors, doc_cache_stats = _embed_with_cache(
        cache_file=doc_cache_file,
        cache=doc_cache,
        rows=document_rows,
        model=model,
        output_dimensionality=output_dimensionality,
        task_type="RETRIEVAL_DOCUMENT",
        batch_size=batch_size,
    )
    query_vectors, query_cache_stats = _embed_with_cache(
        cache_file=query_cache_file,
        cache=query_cache,
        rows=query_text_rows,
        model=model,
        output_dimensionality=output_dimensionality,
        task_type="RETRIEVAL_QUERY",
        batch_size=batch_size,
    )

    doc_text_by_id = {doc_id: text for doc_id, text in document_rows}
    doc_ids = [doc_id for doc_id, _text in document_rows]
    bm25 = BM25Okapi([_tokenize(doc_text_by_id[doc_id]) for doc_id in doc_ids])

    rankings_rows: List[Dict[str, Any]] = []
    for query_id, query_text in query_text_rows:
        query_vector = query_vectors.get(query_id) or []
        dense_scored = sorted(
            (
                (_cosine_similarity(query_vector, document_vectors.get(doc_id) or []), doc_id)
                for doc_id in doc_ids
            ),
            key=lambda item: (-item[0], item[1]),
        )
        dense_ranked = [doc_id for _score, doc_id in dense_scored[: max(top_k, dense_top_k)] if _score > 0.0]

        sparse_scores = bm25.get_scores(_tokenize(query_text))
        sparse_ranked = [
            doc_id
            for _score, doc_id in sorted(
                zip(sparse_scores, doc_ids),
                key=lambda item: (-float(item[0]), item[1]),
            )[: max(top_k, sparse_top_k)]
            if float(_score) > 0.0
        ]

        ranked_ids = _rrf_fuse(
            [dense_ranked[:dense_top_k], sparse_ranked[:sparse_top_k]],
            rrf_k=int(rrf_k),
            top_k=int(top_k),
        )
        rankings_rows.append(
            {
                "query_id": query_id,
                "ranked_ids": ranked_ids,
                "dense_ranked_ids": dense_ranked[:dense_top_k],
                "sparse_ranked_ids": sparse_ranked[:sparse_top_k],
            }
        )

    output_rankings_path = Path(output_rankings_path)
    output_rankings_path.parent.mkdir(parents=True, exist_ok=True)
    output_rankings_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rankings_rows) + "\n",
        encoding="utf-8",
    )

    ranking_report = evaluate_standard_rankings(
        dataset_dir=dataset_dir,
        rankings_path=output_rankings_path,
        k=top_k,
    )
    return {
        "dataset_dir": str(dataset_dir),
        "rankings_path": str(output_rankings_path.resolve()),
        "top_k": int(top_k),
        "dense_top_k": int(dense_top_k),
        "sparse_top_k": int(sparse_top_k),
        "rrf_k": int(rrf_k),
        "embedding_model": model,
        "output_dimensionality": output_dimensionality,
        "document_cache": {
            "path": str(doc_cache_file) if doc_cache_file else "",
            **doc_cache_stats,
        },
        "query_cache": {
            "path": str(query_cache_file) if query_cache_file else "",
            **query_cache_stats,
        },
        "overall": ranking_report["overall"],
        "queries": ranking_report["queries"],
    }
