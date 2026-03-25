from __future__ import annotations

import hashlib
import json
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pipeline.core.config import load_config
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.retrieval import AdaptiveHybridRetriever

_RETRIEVAL_RESULT_CACHE_VERSION = 1
_RETRIEVAL_RESULT_CACHE_ENTRY_VERSION = 1


def _fingerprint_payload(payload: Any) -> str:
    return hashlib.sha1(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _hit_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return 1.0 if any(item in gold_set for item in retrieved[:k]) else 0.0


def _recall_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    return len(set(retrieved[:k]) & gold_set) / float(len(gold_set))


def _mrr_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        if item in gold_set:
            return 1.0 / float(rank)
    return 0.0


def _dcg_at_k(retrieved: Sequence[str], gains: Dict[str, float], k: int) -> float:
    score = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        gain = float(gains.get(item, 0.0))
        if gain <= 0.0:
            continue
        score += gain / __import__("math").log2(rank + 1.0)
    return score


def _ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    gains = {item: 1.0 for item in gold}
    ideal = _dcg_at_k(list(gains.keys()), gains, k)
    if ideal <= 0.0:
        return 0.0
    return _dcg_at_k(retrieved, gains, k) / ideal


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _unique_in_order(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _gold_ids(example: EvalExample, field_name: str) -> List[str]:
    base_values = list(getattr(example, field_name) or [])
    metadata = dict(example.metadata or {})
    alternate_key = f"alternate_{field_name}"
    alternate_values = metadata.get(alternate_key) or []
    return _unique_in_order([*base_values, *[str(value) for value in alternate_values if str(value)]])


def _example_fingerprint(example: EvalExample) -> str:
    return _fingerprint_payload(example.to_dict())


def _dataset_fingerprint(examples: Sequence[EvalExample]) -> str:
    return _fingerprint_payload([example.to_dict() for example in examples])


def _config_fingerprint(config_payload: Dict[str, Any] | None) -> str:
    return _fingerprint_payload(config_payload or {})


def _retrieval_cache_config_payload(config_payload: Dict[str, Any] | None) -> Dict[str, Any]:
    payload = dict(config_payload or {})
    retrieval_cfg = dict(payload.get("retrieval") or {})
    embedder_cfg = dict(payload.get("embedder") or {})
    relevant_embedder = {
        key: embedder_cfg.get(key)
        for key in (
            "model",
            "output_dimensionality",
            "index_name",
            "sparse_index_name",
            "namespace",
            "parent_namespace",
            "media_namespace",
            "fact_namespace",
            "sparse_namespace",
            "sparse_parent_namespace",
            "sparse_media_namespace",
            "sparse_fact_namespace",
            "enable_dense_facts",
        )
        if key in embedder_cfg
    }
    normalized: Dict[str, Any] = {
        "retrieval": retrieval_cfg,
        "embedder": relevant_embedder,
    }
    retriever_backend = str(retrieval_cfg.get("retriever_backend") or "vector").strip().lower()
    if retriever_backend != "vector":
        normalized["graph"] = dict(payload.get("graph") or {})
    return normalized


@dataclass
class QueryRetrievalScore:
    id: str
    query: str
    query_type: str
    source_type: str
    benchmark_tags: List[str]
    mode: str
    no_answer: bool
    has_gold_chunks: bool
    has_gold_parents: bool
    has_gold_media: bool
    seed_chunk_hit_at_5: float
    chunk_hit_at_5: float
    chunk_hit_at_10: float
    chunk_recall_at_10: float
    chunk_mrr_at_10: float
    chunk_ndcg_at_10: float
    parent_hit_at_5: float
    media_hit_at_1: float
    media_hit_at_5: float
    media_mrr_at_5: float
    expansion_gain_hit: float
    no_answer_violation: float
    selected_chunk_ids: List[str]
    selected_parent_ids: List[str]
    selected_media_ids: List[str]
    dense_parent_ids: List[str]
    dense_media_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_query(example: EvalExample, result: Dict[str, Any]) -> QueryRetrievalScore:
    gold_chunk_ids = _gold_ids(example, "gold_chunk_ids")
    gold_parent_ids = _gold_ids(example, "gold_parent_ids")
    gold_media_ids = _gold_ids(example, "gold_media_ids")
    seed_chunk_ids = _unique_in_order(result.get("seed_chunk_ids") or [])
    selected_chunk_ids = _unique_in_order(result.get("selected_chunk_ids") or [])
    dense_parent_ids = _unique_in_order(result.get("dense_parent_ids") or [])
    selected_parent_ids = _unique_in_order(result.get("selected_parent_ids") or dense_parent_ids)
    dense_media_ids = _unique_in_order(result.get("dense_media_ids") or [])
    selected_media_ids = _unique_in_order(
        list(result.get("selected_media_ids") or [])
        + [str(item.get("id") or "") for item in (result.get("media") or []) if isinstance(item, dict)]
        + dense_media_ids
    )

    seed_hit = _hit_at_k(seed_chunk_ids, gold_chunk_ids, 5)
    chunk_hit_10 = _hit_at_k(selected_chunk_ids, gold_chunk_ids, 10)
    return QueryRetrievalScore(
        id=example.id,
        query=example.query,
        query_type=example.query_type,
        source_type=example.source_type,
        benchmark_tags=[
            str(tag)
            for tag in (dict(example.metadata or {}).get("benchmark_tags") or [])
            if str(tag)
        ],
        mode=str(result.get("mode") or ""),
        no_answer=example.no_answer,
        has_gold_chunks=bool(gold_chunk_ids),
        has_gold_parents=bool(gold_parent_ids),
        has_gold_media=bool(gold_media_ids),
        seed_chunk_hit_at_5=seed_hit,
        chunk_hit_at_5=_hit_at_k(selected_chunk_ids, gold_chunk_ids, 5),
        chunk_hit_at_10=chunk_hit_10,
        chunk_recall_at_10=_recall_at_k(selected_chunk_ids, gold_chunk_ids, 10),
        chunk_mrr_at_10=_mrr_at_k(selected_chunk_ids, gold_chunk_ids, 10),
        chunk_ndcg_at_10=_ndcg_at_k(selected_chunk_ids, gold_chunk_ids, 10),
        parent_hit_at_5=_hit_at_k(selected_parent_ids, gold_parent_ids, 5),
        media_hit_at_1=_hit_at_k(selected_media_ids, gold_media_ids, 1),
        media_hit_at_5=_hit_at_k(selected_media_ids, gold_media_ids, 5),
        media_mrr_at_5=_mrr_at_k(selected_media_ids, gold_media_ids, 5),
        expansion_gain_hit=1.0 if (chunk_hit_10 > 0.0 and seed_hit <= 0.0) else 0.0,
        no_answer_violation=1.0 if example.no_answer and bool(selected_chunk_ids or selected_parent_ids or selected_media_ids) else 0.0,
        selected_chunk_ids=selected_chunk_ids,
        selected_parent_ids=selected_parent_ids,
        selected_media_ids=selected_media_ids,
        dense_parent_ids=dense_parent_ids,
        dense_media_ids=dense_media_ids,
    )


def _aggregate_scores(scores: Sequence[QueryRetrievalScore]) -> Dict[str, float]:
    answerable_scores = [score for score in scores if not score.no_answer] or list(scores)
    chunk_scores = [score for score in answerable_scores if score.has_gold_chunks] or answerable_scores
    parent_scores = [score for score in answerable_scores if score.has_gold_parents] or answerable_scores
    media_scores = [score for score in answerable_scores if score.has_gold_media]
    return {
        "query_count": float(len(scores)),
        "eligible_chunk_query_count": float(len(chunk_scores)),
        "eligible_parent_query_count": float(len(parent_scores)),
        "eligible_media_query_count": float(len(media_scores)),
        "seed_chunk_hit_at_5": _mean(score.seed_chunk_hit_at_5 for score in chunk_scores),
        "chunk_hit_at_5": _mean(score.chunk_hit_at_5 for score in chunk_scores),
        "chunk_hit_at_10": _mean(score.chunk_hit_at_10 for score in chunk_scores),
        "chunk_recall_at_10": _mean(score.chunk_recall_at_10 for score in chunk_scores),
        "chunk_mrr_at_10": _mean(score.chunk_mrr_at_10 for score in chunk_scores),
        "chunk_ndcg_at_10": _mean(score.chunk_ndcg_at_10 for score in chunk_scores),
        "parent_hit_at_5": _mean(score.parent_hit_at_5 for score in parent_scores),
        "media_hit_at_1": _mean(score.media_hit_at_1 for score in media_scores),
        "media_hit_at_5": _mean(score.media_hit_at_5 for score in media_scores),
        "media_mrr_at_5": _mean(score.media_mrr_at_5 for score in media_scores),
        "expansion_gain_hit_rate": _mean(score.expansion_gain_hit for score in chunk_scores),
        "no_answer_violation_rate": _mean(score.no_answer_violation for score in scores),
    }


def _slice_scores(scores: Sequence[QueryRetrievalScore], attribute: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[QueryRetrievalScore]] = defaultdict(list)
    for score in scores:
        grouped[str(getattr(score, attribute))].append(score)
    return {key: _aggregate_scores(items) for key, items in sorted(grouped.items())}


def _slice_scores_by_benchmark_tag(scores: Sequence[QueryRetrievalScore]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[QueryRetrievalScore]] = defaultdict(list)
    for score in scores:
        tags = [str(tag) for tag in (score.benchmark_tags or []) if str(tag)]
        if not tags:
            continue
        for tag in dict.fromkeys(tags):
            grouped[tag].append(score)
    return {key: _aggregate_scores(items) for key, items in sorted(grouped.items())}


def load_eval_gates(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def check_metric_gates(report: Dict[str, Any], gates: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    if not gates:
        return failures
    epsilon = 1e-9
    sections = {
        "overall": report.get("overall", {}),
        "by_query_type": report.get("by_query_type", {}),
        "by_source_type": report.get("by_source_type", {}),
        "by_benchmark_tag": report.get("by_benchmark_tag", {}),
    }
    for section_name, section_gates in gates.items():
        if section_name not in sections or not isinstance(section_gates, dict):
            continue
        section_report = sections[section_name]
        if section_name == "overall":
            targets = {"overall": section_gates}
        else:
            targets = section_gates
        for slice_name, metric_rules in targets.items():
            metrics = section_report if section_name == "overall" else section_report.get(slice_name, {})
            if not isinstance(metrics, dict):
                failures.append(
                    {
                        "section": section_name,
                        "slice": slice_name,
                        "metric": None,
                        "reason": "missing_slice",
                    }
                )
                continue
            for metric_name, rule in metric_rules.items():
                actual = metrics.get(metric_name)
                if actual is None:
                    failures.append(
                        {
                            "section": section_name,
                            "slice": slice_name,
                            "metric": metric_name,
                            "reason": "missing_metric",
                        }
                    )
                    continue
                if isinstance(rule, (int, float)):
                    minimum = float(rule)
                    if float(actual) + epsilon < minimum:
                        failures.append(
                            {
                                "section": section_name,
                                "slice": slice_name,
                                "metric": metric_name,
                                "expected_min": minimum,
                                "actual": float(actual),
                            }
                        )
                    continue
                if isinstance(rule, dict):
                    minimum = rule.get("min")
                    maximum = rule.get("max")
                    if minimum is not None and float(actual) + epsilon < float(minimum):
                        failures.append(
                            {
                                "section": section_name,
                                "slice": slice_name,
                                "metric": metric_name,
                                "expected_min": float(minimum),
                                "actual": float(actual),
                            }
                        )
                    if maximum is not None and float(actual) - epsilon > float(maximum):
                        failures.append(
                            {
                                "section": section_name,
                                "slice": slice_name,
                                "metric": metric_name,
                                "expected_max": float(maximum),
                                "actual": float(actual),
                            }
                        )
    return failures


def _default_query_cache_path(*, work_dir: str | Path, dataset_path: str | Path) -> Path:
    dataset_stem = Path(dataset_path).stem or "eval"
    return (
        Path(work_dir).resolve()
        / "eval_cache"
        / f"{dataset_stem}.query_embeddings.json"
    )


def _query_cache_key(
    *,
    query: str,
    model: str,
    output_dimensionality: int | None,
) -> str:
    payload = json.dumps(
        {
            "query": str(query),
            "model": str(model),
            "output_dimensionality": int(output_dimensionality) if output_dimensionality is not None else None,
            "task_type": "RETRIEVAL_QUERY",
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_query_embedding_cache(path: str | Path | None) -> Tuple[Path | None, Dict[str, List[float]]]:
    if not path:
        return None, {}
    cache_path = Path(path).resolve()
    payload = load_json_safe(cache_path, default={})
    if not isinstance(payload, dict):
        return cache_path, {}
    cache: Dict[str, List[float]] = {}
    for key, values in payload.items():
        if not isinstance(key, str) or not isinstance(values, list):
            continue
        try:
            cache[key] = [float(value) for value in values]
        except Exception:
            continue
    return cache_path, cache


def _save_query_embedding_cache(path: Path | None, cache: Dict[str, List[float]]) -> None:
    if path is None:
        return
    atomic_write_json(path, cache)


def _default_retrieval_cache_path(*, work_dir: str | Path, dataset_path: str | Path) -> Path:
    dataset_stem = Path(dataset_path).stem or "eval"
    return (
        Path(work_dir).resolve()
        / "eval_cache"
        / f"{dataset_stem}.retrieval_results.json"
    )


def _retrieval_cache_key(
    *,
    config_name: str,
    work_dir: str | Path,
    query: str,
    model: str,
    output_dimensionality: int | None,
    config_payload: Dict[str, Any] | None,
) -> str:
    config_fingerprint = _config_fingerprint(_retrieval_cache_config_payload(config_payload))
    payload = json.dumps(
        {
            "version": _RETRIEVAL_RESULT_CACHE_VERSION,
            "config_name": str(config_name),
            "work_dir": str(Path(work_dir).resolve()),
            "query": str(query),
            "model": str(model),
            "output_dimensionality": int(output_dimensionality) if output_dimensionality is not None else None,
            "config_fingerprint": config_fingerprint,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _load_retrieval_result_cache(path: str | Path | None) -> Tuple[Path | None, Dict[str, Dict[str, Any]]]:
    if not path:
        return None, {}
    cache_path = Path(path).resolve()
    payload = load_json_safe(cache_path, default={})
    if not isinstance(payload, dict):
        return cache_path, {}
    cache: Dict[str, Dict[str, Any]] = {}
    for key, value in payload.items():
        if isinstance(key, str) and isinstance(value, dict):
            cache[key] = value
    return cache_path, cache


def _save_retrieval_result_cache(path: Path | None, cache: Dict[str, Dict[str, Any]]) -> None:
    if path is None:
        return
    atomic_write_json(path, cache)


def _make_retrieval_cache_entry(
    *,
    example: EvalExample,
    result: Dict[str, Any],
    config_name: str,
    work_dir: str | Path,
    model: str,
    output_dimensionality: int | None,
    config_payload: Dict[str, Any] | None,
) -> Dict[str, Any]:
    return {
        "_cache_entry_version": _RETRIEVAL_RESULT_CACHE_ENTRY_VERSION,
        "query_id": example.id,
        "query": example.query,
        "query_fingerprint": _fingerprint_payload(example.query),
        "example_fingerprint": _example_fingerprint(example),
        "config_name": str(config_name),
        "config_fingerprint": _config_fingerprint(_retrieval_cache_config_payload(config_payload)),
        "work_dir": str(Path(work_dir).resolve()),
        "model": str(model),
        "output_dimensionality": int(output_dimensionality) if output_dimensionality is not None else None,
        "result": result,
    }


def _get_cached_retrieval_result(
    cached_value: Dict[str, Any] | None,
    *,
    example: EvalExample,
    config_name: str,
    work_dir: str | Path,
    config_payload: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    if not isinstance(cached_value, dict):
        return None
    if "result" not in cached_value:
        return cached_value
    if int(cached_value.get("_cache_entry_version") or 0) != _RETRIEVAL_RESULT_CACHE_ENTRY_VERSION:
        return None
    if str(cached_value.get("query") or "") != example.query:
        return None
    if str(cached_value.get("query_fingerprint") or "") != _fingerprint_payload(example.query):
        return None
    if str(cached_value.get("config_name") or "") != str(config_name):
        return None
    if str(cached_value.get("config_fingerprint") or "") != _config_fingerprint(
        _retrieval_cache_config_payload(config_payload)
    ):
        return None
    if str(cached_value.get("work_dir") or "") != str(Path(work_dir).resolve()):
        return None
    result = cached_value.get("result")
    return result if isinstance(result, dict) else None


def evaluate_retrieval_dataset(
    *,
    config_name: str,
    work_dir: str | Path,
    dataset_path: str | Path,
    gates_path: str | Path | None = None,
    query_cache_path: str | Path | None = None,
    retrieval_cache_path: str | Path | None = None,
    parallelism: int = 1,
) -> Dict[str, Any]:
    examples = load_eval_examples(dataset_path)
    dataset_fingerprint = _dataset_fingerprint(examples)
    try:
        config_payload = dict(load_config(config_name) or {})
    except FileNotFoundError:
        config_payload = {}
    config_fingerprint = _config_fingerprint(config_payload)
    embed_cfg = dict(config_payload.get("embedder") or {})
    cache_model = str(embed_cfg.get("model") or "gemini-embedding-2-preview")
    cache_output_dimensionality = embed_cfg.get("output_dimensionality")
    cache_path = (
        Path(query_cache_path).resolve()
        if query_cache_path
        else _default_query_cache_path(work_dir=work_dir, dataset_path=dataset_path)
    )
    cache_file, query_cache = _load_query_embedding_cache(cache_path)
    initial_cache_keys = set(query_cache)
    retrieval_cache_target = (
        Path(retrieval_cache_path).resolve()
        if retrieval_cache_path
        else _default_retrieval_cache_path(work_dir=work_dir, dataset_path=dataset_path)
    )
    retrieval_cache_file, retrieval_cache = _load_retrieval_result_cache(retrieval_cache_target)
    scores: List[QueryRetrievalScore] = []
    cache_hits = 0
    cache_misses = 0
    cache_dirty = False
    retrieval_cache_hits = 0
    retrieval_cache_misses = 0
    retrieval_cache_dirty = False
    retrieval_cache_invalid = 0
    legacy_retrieval_cache_hits = 0
    cached_scores: Dict[int, QueryRetrievalScore] = {}
    ordered_scores: List[QueryRetrievalScore | None] = [None] * len(examples)
    uncached_examples: List[Tuple[int, EvalExample, List[float] | None, str]] = []
    for index, example in enumerate(examples):
        retrieval_key = _retrieval_cache_key(
            config_name=config_name,
            work_dir=work_dir,
            query=example.query,
            model=cache_model,
            output_dimensionality=cache_output_dimensionality,
            config_payload=config_payload,
        )
        cached_value = retrieval_cache.get(retrieval_key)
        cached_result = _get_cached_retrieval_result(
            cached_value,
            example=example,
            config_name=config_name,
            work_dir=work_dir,
            config_payload=config_payload,
        )
        if cached_result is not None:
            score = _score_query(example, cached_result)
            cached_scores[index] = score
            ordered_scores[index] = score
            retrieval_cache_hits += 1
            if isinstance(cached_value, dict) and "result" not in cached_value:
                legacy_retrieval_cache_hits += 1
            continue
        if cached_value is not None:
            retrieval_cache_invalid += 1
        retrieval_cache_misses += 1
        uncached_examples.append((index, example, None, retrieval_key))

    retriever = None
    supports_query_cache = False
    if uncached_examples:
        retriever = AdaptiveHybridRetriever.from_config(config_name=config_name, work_dir=work_dir)
        supports_query_cache = hasattr(retriever, "embed_query") and callable(getattr(retriever, "embed_query"))
        missing_examples: List[Tuple[EvalExample, str]] = []
        for _index, example, _query_vector, _retrieval_key in uncached_examples:
            cache_key = _query_cache_key(
                query=example.query,
                model=getattr(retriever, "model", cache_model),
                output_dimensionality=getattr(retriever, "output_dimensionality", cache_output_dimensionality),
            )
            if supports_query_cache and cache_key not in query_cache:
                missing_examples.append((example, cache_key))

        if missing_examples:
            miss_queries = [example.query for example, _cache_key in missing_examples]
            if hasattr(retriever, "embed_queries") and callable(getattr(retriever, "embed_queries")):
                embedded_vectors = retriever.embed_queries(miss_queries)
            else:
                embedded_vectors = [retriever.embed_query(query) for query in miss_queries]
            for (_example, cache_key), vector in zip(missing_examples, embedded_vectors):
                query_cache[cache_key] = [float(value) for value in vector]
                cache_misses += 1
                cache_dirty = True
            _save_query_embedding_cache(cache_file, query_cache)

        indexed_uncached_examples: List[Tuple[int, EvalExample, List[float] | None, str]] = []
        for index, example, _query_vector, retrieval_key in uncached_examples:
            cache_key = _query_cache_key(
                query=example.query,
                model=getattr(retriever, "model", cache_model),
                output_dimensionality=getattr(retriever, "output_dimensionality", cache_output_dimensionality),
            )
            query_vector = query_cache.get(cache_key)
            if query_vector is None and supports_query_cache:
                query_vector = retriever.embed_query(example.query)
                query_cache[cache_key] = list(query_vector)
                cache_misses += 1
                cache_dirty = True
                _save_query_embedding_cache(cache_file, query_cache)
            elif query_vector is not None and cache_key in initial_cache_keys:
                cache_hits += 1
            indexed_uncached_examples.append((index, example, query_vector, retrieval_key))
        uncached_examples = indexed_uncached_examples

    max_workers = max(1, int(parallelism or 1))
    if max_workers == 1 or len(uncached_examples) <= 1:
        for index, example, query_vector, retrieval_key in uncached_examples:
            try:
                result = retriever.retrieve(example.query, query_vector=query_vector)
            except TypeError:
                result = retriever.retrieve(example.query)
            retrieval_cache[retrieval_key] = _make_retrieval_cache_entry(
                example=example,
                result=result,
                config_name=config_name,
                work_dir=work_dir,
                model=getattr(retriever, "model", cache_model),
                output_dimensionality=getattr(retriever, "output_dimensionality", cache_output_dimensionality),
                config_payload=config_payload,
            )
            retrieval_cache_dirty = True
            _save_retrieval_result_cache(retrieval_cache_file, retrieval_cache)
            score = _score_query(example, result)
            cached_scores[index] = score
            ordered_scores[index] = score
    else:
        thread_state = threading.local()
        shared_parallel_retriever = bool(
            retriever is not None and getattr(retriever, "supports_shared_parallel_retrieval", False)
        )

        def _worker_retriever() -> AdaptiveHybridRetriever:
            if shared_parallel_retriever:
                return retriever
            worker = getattr(thread_state, "retriever", None)
            if worker is None:
                worker = AdaptiveHybridRetriever.from_config(config_name=config_name, work_dir=work_dir)
                thread_state.retriever = worker
            return worker

        def _retrieve_indexed(item: Tuple[int, EvalExample, List[float] | None, str]) -> Tuple[int, str, Dict[str, Any], QueryRetrievalScore]:
            index, example, query_vector, retrieval_key = item
            worker_retriever = _worker_retriever()
            try:
                result = worker_retriever.retrieve(example.query, query_vector=query_vector)
            except TypeError:
                result = worker_retriever.retrieve(example.query)
            return index, retrieval_key, result, _score_query(example, result)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_retrieve_indexed, item) for item in uncached_examples]
            for future in as_completed(futures):
                index, retrieval_key, result, score = future.result()
                example = examples[index]
                retrieval_cache[retrieval_key] = _make_retrieval_cache_entry(
                    example=example,
                    result=result,
                    config_name=config_name,
                    work_dir=work_dir,
                    model=getattr(retriever, "model", cache_model),
                    output_dimensionality=getattr(retriever, "output_dimensionality", cache_output_dimensionality),
                    config_payload=config_payload,
                )
                retrieval_cache_dirty = True
                _save_retrieval_result_cache(retrieval_cache_file, retrieval_cache)
                ordered_scores[index] = score

    if any(score is None for score in ordered_scores):
        missing_ids = [examples[index].id for index, score in enumerate(ordered_scores) if score is None]
        raise RuntimeError(f"Strict evaluation failed to score all queries. Missing ids: {missing_ids}")
    scores = [score for score in ordered_scores if score is not None]

    if cache_dirty:
        _save_query_embedding_cache(cache_file, query_cache)
    if retrieval_cache_dirty:
        _save_retrieval_result_cache(retrieval_cache_file, retrieval_cache)

    report = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "work_dir": str(Path(work_dir).resolve()),
        "config_name": str(config_name),
        "config_fingerprint": config_fingerprint,
        "query_cache_path": str(cache_file) if cache_file else "",
        "retrieval_cache_path": str(retrieval_cache_file) if retrieval_cache_file else "",
        "query_count": len(scores),
        "execution": {
            "strict": True,
            "parallelism_requested": int(parallelism or 1),
            "parallelism_effective": 1 if len(uncached_examples) <= 1 else max_workers,
            "shared_parallel_retriever": bool(
                max_workers > 1 and retriever is not None and getattr(retriever, "supports_shared_parallel_retrieval", False)
            ),
        },
        "cache": {
            "path": str(cache_file) if cache_file else "",
            "hit_count": cache_hits,
            "miss_count": cache_misses,
            "size": len(query_cache),
        },
        "retrieval_cache": {
            "path": str(retrieval_cache_file) if retrieval_cache_file else "",
            "hit_count": retrieval_cache_hits,
            "miss_count": retrieval_cache_misses,
            "invalid_count": retrieval_cache_invalid,
            "legacy_hit_count": legacy_retrieval_cache_hits,
            "size": len(retrieval_cache),
        },
        "overall": _aggregate_scores(scores),
        "by_query_type": _slice_scores(scores, "query_type"),
        "by_source_type": _slice_scores(scores, "source_type"),
        "by_benchmark_tag": _slice_scores_by_benchmark_tag(scores),
        "queries": [score.to_dict() for score in scores],
    }
    gates = load_eval_gates(gates_path)
    failures = check_metric_gates(report, gates)
    report["gates"] = {
        "path": str(Path(gates_path).resolve()) if gates_path else "",
        "passed": not failures,
        "failures": failures,
    }
    return report
