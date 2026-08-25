from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import unquote, urlparse

from pipeline.core.config import load_config
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.evaluation.dataset_tools import validate_eval_examples
from pipeline.retrieval import AdaptiveHybridRetriever
from pipeline.retrieval.adaptive_hybrid import apply_vector_upload_manifest_config

_RETRIEVAL_RESULT_CACHE_VERSION = 7
_RETRIEVAL_RESULT_CACHE_ENTRY_VERSION = 7
EvalProgressCallback = Callable[[str, Mapping[str, Any]], None]


def _emit_progress(
    callback: EvalProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        # Progress reporting must never affect strict scoring.
        return


def _error_retrieval_result(example: EvalExample, exc: BaseException) -> Dict[str, Any]:
    return {
        "mode": "error",
        "abstained": True,
        "retrieval_error": f"{type(exc).__name__}: {exc}",
        "seed_chunk_ids": [],
        "selected_chunk_ids": [],
        "selected_parent_ids": [],
        "selected_media_ids": [],
        "selected_evidence_span_ids": [],
        "retrieval_documents": [],
        "evidence_span_documents": [],
        "retrieval_trace": {
            "query_id": example.id,
            "error": f"{type(exc).__name__}: {exc}",
        },
    }


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
        score += gain / math.log2(rank + 1.0)
    return score


def _ndcg_at_k(retrieved: Sequence[str], gold: Sequence[str], k: int) -> float:
    if not gold:
        return 0.0
    gains = {item: 1.0 for item in gold}
    ideal = _dcg_at_k(list(gains.keys()), gains, k)
    if ideal <= 0.0:
        return 0.0
    return _dcg_at_k(retrieved, gains, k) / ideal


def _source_aware_ndcg_at_k(
    retrieved: Sequence[str],
    *,
    exact_gold: Sequence[str],
    source_expanded_gold: Sequence[str],
    k: int,
    source_equivalent_gain: float = 0.65,
) -> float:
    """Rank score for citation-grade retrieval.

    LLM-generated eval sets often identify one exact chunk, while production
    answers cite source pages. A different chunk from the same official page is
    not as strong as the exact chunk, but it is materially useful evidence. This
    graded score gives exact chunks full credit and source-equivalent chunks
    bounded partial credit without treating every chunk on that source page as a
    separate recall target.
    """
    expanded_set = set(source_expanded_gold)
    if not expanded_set:
        return 0.0
    exact_set = set(exact_gold) & expanded_set
    source_only_set = expanded_set - exact_set
    gains: Dict[str, float] = {}
    for item in exact_set:
        gains[item] = 1.0
    for item in source_only_set:
        gains[item] = max(0.0, min(1.0, float(source_equivalent_gain)))
    if not gains:
        return 0.0
    best_possible_gain = max(gains.values())
    if best_possible_gain <= 0.0:
        return 0.0
    # Use one ideal evidence unit so section/page URL expansion cannot inflate
    # the denominator with every chunk on the page.
    ideal = best_possible_gain
    observed = 0.0
    for rank, item in enumerate(retrieved[:k], start=1):
        gain = float(gains.get(item, 0.0))
        if gain <= 0.0:
            continue
        observed = gain / math.log2(rank + 1.0)
        break
    return min(1.0, observed / ideal)


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


def _flatten_reference_urls(value: Any) -> List[str]:
    urls: List[str] = []
    if isinstance(value, str):
        if value.strip():
            urls.append(value.strip())
        return urls
    if isinstance(value, (list, tuple)):
        for item in value:
            urls.extend(_flatten_reference_urls(item))
    return urls


def _normalize_reference_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
        scheme = parsed.scheme.lower() or "https"
        netloc = parsed.netloc.lower()
        path = unquote(parsed.path or "").rstrip("/")
        path_parts = [part for part in path.split("/") if part]
        if path_parts and path_parts[0].lower() in {"ar", "en"}:
            path = "/" + "/".join(path_parts[1:])
        if path in {"", "/"}:
            path = ""
        return f"{scheme}://{netloc}{path}".rstrip("/")
    except Exception:
        return raw.lower().rstrip("/")


def _slugify_reference_name(value: str) -> str:
    tokens: List[str] = []
    text = str(value or "").replace("’s", "").replace("'s", "").replace("'", " ")
    for raw_token in text.split():
        token = "".join(ch.lower() for ch in raw_token if ch.isalnum())
        if token and token not in {"professor", "prof", "dr", "doctor", "mbzuai"}:
            tokens.append(token)
    return "-".join(tokens[:3])


def _inferred_reference_urls(example: EvalExample) -> List[str]:
    metadata = dict(example.metadata or {})
    hint_text = " ".join(str(value) for value in (metadata.get("expected_source_hints") or []))
    query = str(example.query or "")
    combined = f"{query} {hint_text}".strip()
    lower = combined.lower()
    if "faculty" not in lower and "professor" not in lower and "dr " not in lower:
        return []
    # The generated eval set may select Arabic faculty pages as gold while the
    # English official page is the production answer path for English queries.
    for marker in ("Professor ", "Dr ", "Prof "):
        if marker not in combined:
            continue
        after = combined.split(marker, 1)[1]
        name_parts: List[str] = []
        for raw_part in after.split():
            clean = "".join(ch for ch in raw_part if ch.isalnum() or ch in {"'", "-"})
            if not clean:
                break
            if clean[:1].isupper() or any(ch.isupper() for ch in clean[1:]):
                name_parts.append(clean)
                continue
            break
        slug = _slugify_reference_name(" ".join(name_parts))
        if slug:
            return [f"https://mbzuai.ac.ae/study/faculty/{slug}"]
    return []


def _reference_urls_for_example(example: EvalExample) -> List[str]:
    metadata = dict(example.metadata or {})
    urls = [
        *_flatten_reference_urls(metadata.get("expected_reference_urls") or []),
        *_flatten_reference_urls(metadata.get("alternate_expected_reference_urls") or []),
        *_inferred_reference_urls(example),
    ]
    return _unique_in_order(_normalize_reference_url(url) for url in urls if _normalize_reference_url(url))


def _reference_url_path_depth(url: str) -> int:
    try:
        parsed = urlparse(url)
        return len([part for part in (parsed.path or "").split("/") if part])
    except Exception:
        return 0


def _ids_for_reference_url(field_lookup: Dict[str, List[str]], url: str) -> List[str]:
    ids = list(field_lookup.get(url) or [])
    # Evaluation records sometimes specify a section-level source such as
    # /study/phd-programs while the retriever correctly returns child program
    # pages. Treat non-root section URLs as source-equivalent without allowing
    # the site root to match every page.
    if _reference_url_path_depth(url) >= 2:
        prefix = f"{url}/"
        for candidate_url, candidate_ids in field_lookup.items():
            if candidate_url.startswith(prefix):
                ids.extend(candidate_ids)
    return ids


def _metadata_values(metadata: Dict[str, Any], key: str) -> List[str]:
    value = metadata.get(key)
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _doc_source_url(doc: Dict[str, Any]) -> str:
    metadata = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return _normalize_reference_url(
        str(
            doc.get("source_url")
            or doc.get("document_source")
            or metadata.get("source_url")
            or metadata.get("document_source")
            or ""
        )
    )


def _selected_source_urls(result: Dict[str, Any]) -> List[str]:
    urls: List[str] = []
    evidence_pack = result.get("evidence_pack") if isinstance(result.get("evidence_pack"), dict) else {}
    for item in evidence_pack.get("items") or []:
        if isinstance(item, dict):
            urls.append(_normalize_reference_url(str(item.get("source_url") or "")))
    for collection_name in (
        "answer_documents",
        "fact_documents",
        "evidence_span_documents",
        "retrieval_documents",
        "media",
    ):
        for doc in result.get(collection_name) or []:
            if isinstance(doc, dict):
                urls.append(_doc_source_url(doc))
    return _unique_in_order(url for url in urls if url)


def _selected_evidence_text(result: Dict[str, Any]) -> str:
    fragments: List[str] = []
    evidence_pack = result.get("evidence_pack") if isinstance(result.get("evidence_pack"), dict) else {}
    for item in evidence_pack.get("items") or []:
        if isinstance(item, dict):
            fragments.extend(
                str(item.get(key) or "")
                for key in ("text", "document_title", "section_heading", "breadcrumb", "source_url")
            )
    for collection_name in (
        "answer_documents",
        "fact_documents",
        "evidence_span_documents",
        "retrieval_documents",
        "media",
    ):
        for doc in result.get(collection_name) or []:
            if not isinstance(doc, dict):
                continue
            fragments.extend(
                str(doc.get(key) or "")
                for key in ("text", "value", "document_title", "section_heading", "breadcrumb", "source_url")
            )
    return " ".join(fragments).casefold()


def _metadata_aliases(metadata: Dict[str, Any], key: str) -> Dict[str, List[str]]:
    value = metadata.get(key)
    aliases: Dict[str, List[str]] = {}
    if isinstance(value, Mapping):
        for raw_key, raw_aliases in value.items():
            alias_values = _metadata_values({key: raw_aliases}, key)
            normalized_key = str(raw_key or "").strip().casefold()
            if normalized_key:
                aliases[normalized_key] = alias_values
    elif isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, Mapping):
                continue
            raw_key = item.get("value") or item.get("entity") or item.get("facet") or item.get("term")
            alias_values = _metadata_values({"aliases": item.get("aliases") or item.get("values")}, "aliases")
            normalized_key = str(raw_key or "").strip().casefold()
            if normalized_key:
                aliases[normalized_key] = alias_values
    return aliases


def _coverage_fraction(
    required: Sequence[str],
    observed_text: str,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> float:
    required_values = [str(value).strip().casefold() for value in required if str(value).strip()]
    if not required_values:
        return 1.0
    alias_map = {
        str(key or "").strip().casefold(): [str(item).strip().casefold() for item in values if str(item).strip()]
        for key, values in (aliases or {}).items()
        if str(key or "").strip()
    }
    hits = 0
    for value in required_values:
        candidates = [value, *alias_map.get(value, [])]
        if any(candidate and candidate in observed_text for candidate in candidates):
            hits += 1
    return hits / float(len(required_values))


def _url_coverage_fraction(required_urls: Sequence[str], selected_urls: Sequence[str]) -> float:
    required = [_normalize_reference_url(str(url)) for url in required_urls if _normalize_reference_url(str(url))]
    if not required:
        return 1.0
    selected = [_normalize_reference_url(str(url)) for url in selected_urls if _normalize_reference_url(str(url))]
    hits = 0
    for required_url in required:
        if any(selected_url == required_url or selected_url.startswith(f"{required_url}/") for selected_url in selected):
            hits += 1
    return hits / float(len(required))


def _load_gold_ids_by_url(work_dir: str | Path) -> Dict[str, Dict[str, List[str]]]:
    work_path = Path(work_dir).resolve()
    bundle_path = work_path / "stage_outputs" / "finalize_retrieval_bundle" / "retrieval_bundle.json"
    if not bundle_path.exists():
        bundle_path = work_path / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    if not bundle_path.exists():
        bundle_path = work_path / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json"
    payload = load_json_safe(bundle_path, default={})
    if not isinstance(payload, dict):
        return {"gold_chunk_ids": {}, "gold_span_ids": {}, "gold_parent_ids": {}, "gold_media_ids": {}}

    def _index_records(records: Sequence[Dict[str, Any]], *url_keys: str) -> Dict[str, List[str]]:
        indexed: Dict[str, List[str]] = defaultdict(list)
        for record in records:
            if not isinstance(record, dict):
                continue
            record_id = str(record.get("id") or "")
            if not record_id:
                continue
            for key in url_keys:
                normalized = _normalize_reference_url(str(record.get(key) or ""))
                if normalized:
                    indexed[normalized].append(record_id)
        return {key: _unique_in_order(values) for key, values in indexed.items()}

    return {
        "gold_chunk_ids": _index_records(payload.get("chunk_records") or [], "source_url"),
        "gold_span_ids": _index_records(
            payload.get("evidence_span_records") or [],
            "source_url",
            "canonical_url",
            "language_normalized_url",
        ),
        "gold_parent_ids": _index_records(payload.get("parent_records") or [], "source_url"),
        "gold_media_ids": _index_records(payload.get("media_records") or [], "source_url", "url"),
    }


def _gold_ids(
    example: EvalExample,
    field_name: str,
    *,
    ids_by_url: Dict[str, Dict[str, List[str]]] | None = None,
) -> List[str]:
    base_values = list(getattr(example, field_name) or [])
    metadata = dict(example.metadata or {})
    alternate_key = f"alternate_{field_name}"
    alternate_values = metadata.get(alternate_key) or []
    expanded_values: List[str] = []
    if ids_by_url:
        field_lookup = ids_by_url.get(field_name) or {}
        for url in _reference_urls_for_example(example):
            expanded_values.extend(_ids_for_reference_url(field_lookup, url))
    return _unique_in_order(
        [
            *base_values,
            *[str(value) for value in alternate_values if str(value)],
            *expanded_values,
        ]
    )


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
            "pinecone_index",
            "pinecone_summary_index",
            "pinecone_text_index",
            "pinecone_sparse_index",
            "index_name",
            "sparse_index_name",
            "namespace",
            "namespace_chunks",
            "namespace_parents",
            "namespace_media",
            "namespace_facts",
            "namespace_evidence_spans",
            "namespace_summaries",
            "namespace_assertions",
            "parent_namespace",
            "media_namespace",
            "fact_namespace",
            "sparse_namespace",
            "sparse_parent_namespace",
            "sparse_media_namespace",
            "sparse_fact_namespace",
            "enable_dense_facts",
            "enable_dense_evidence_spans",
            "enable_sparse_evidence_spans",
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
    chunk_exact_ndcg_at_10: float
    chunk_source_ndcg_at_10: float
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
    language: str = "English"
    chunk_exact_hit_at_5: float = 0.0
    chunk_exact_hit_at_10: float = 0.0
    chunk_exact_mrr_at_10: float = 0.0
    chunk_source_hit_at_5: float = 0.0
    chunk_source_hit_at_10: float = 0.0
    chunk_source_mrr_at_10: float = 0.0
    has_gold_spans: bool = False
    selected_span_ids: List[str] = field(default_factory=list)
    span_hit_at_5: float = 0.0
    span_hit_at_10: float = 0.0
    span_mrr_at_10: float = 0.0
    span_source_ndcg_at_10: float = 0.0
    has_required_entities: bool = False
    has_required_pages: bool = False
    has_required_sections: bool = False
    has_expected_citations: bool = False
    has_min_distinct_sources: bool = False
    required_entity_coverage: float = 1.0
    required_page_coverage: float = 1.0
    required_section_coverage: float = 1.0
    citation_support_rate: float = 1.0
    multi_page_coverage_rate: float = 1.0
    unsupported_abstention_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_query(
    example: EvalExample,
    result: Dict[str, Any],
    *,
    ids_by_url: Dict[str, Dict[str, List[str]]] | None = None,
) -> QueryRetrievalScore:
    exact_gold_chunk_ids = _gold_ids(example, "gold_chunk_ids")
    gold_chunk_ids = _gold_ids(example, "gold_chunk_ids", ids_by_url=ids_by_url)
    exact_gold_span_ids = _gold_ids(example, "gold_span_ids")
    gold_span_ids = _gold_ids(example, "gold_span_ids", ids_by_url=ids_by_url)
    recall_gold_chunk_ids = exact_gold_chunk_ids or gold_chunk_ids
    recall_gold_span_ids = exact_gold_span_ids or gold_span_ids
    gold_parent_ids = _gold_ids(example, "gold_parent_ids", ids_by_url=ids_by_url)
    gold_media_ids = _gold_ids(example, "gold_media_ids", ids_by_url=ids_by_url)
    seed_chunk_ids = _unique_in_order(result.get("seed_chunk_ids") or [])
    selected_chunk_ids = _unique_in_order(result.get("selected_chunk_ids") or [])
    selected_span_ids = _unique_in_order(result.get("selected_evidence_span_ids") or [])
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
    chunk_exact_hit_5 = _hit_at_k(selected_chunk_ids, exact_gold_chunk_ids, 5)
    chunk_exact_hit_10 = _hit_at_k(selected_chunk_ids, exact_gold_chunk_ids, 10)
    chunk_source_hit_5 = _hit_at_k(selected_chunk_ids, gold_chunk_ids, 5)
    chunk_source_hit_10 = chunk_hit_10
    exact_ndcg = _ndcg_at_k(selected_chunk_ids, recall_gold_chunk_ids, 10)
    source_ndcg = _source_aware_ndcg_at_k(
        selected_chunk_ids,
        exact_gold=exact_gold_chunk_ids,
        source_expanded_gold=gold_chunk_ids,
        k=10,
    )
    span_source_ndcg = _source_aware_ndcg_at_k(
        selected_span_ids,
        exact_gold=exact_gold_span_ids,
        source_expanded_gold=gold_span_ids,
        k=10,
    )
    metadata = dict(example.metadata or {})
    selected_urls = _selected_source_urls(result)
    selected_text = _selected_evidence_text(result)
    required_entities = _metadata_values(metadata, "required_entities")
    if not required_entities:
        required_entities = _metadata_values(metadata, "answer_must_include")
    required_entity_aliases = _metadata_aliases(metadata, "required_entity_aliases")
    required_pages = [
        *_metadata_values(metadata, "required_pages"),
        *_metadata_values(metadata, "required_page_urls"),
    ]
    if not required_pages:
        required_pages = _metadata_values(metadata, "expected_reference_urls")
    required_sections = _metadata_values(metadata, "required_sections")
    expected_citation_urls = _metadata_values(metadata, "expected_citation_urls")
    if not expected_citation_urls:
        expected_citation_urls = _reference_urls_for_example(example)
    required_page_coverage = _url_coverage_fraction(required_pages, selected_urls)
    citation_support_rate = _url_coverage_fraction(expected_citation_urls, selected_urls)
    min_distinct_sources = int(metadata.get("min_distinct_sources") or 0)
    distinct_sources = len(set(selected_urls))
    multi_page_coverage_rate = (
        1.0
        if min_distinct_sources <= 0 or distinct_sources >= min_distinct_sources
        else 0.0
    )
    no_answer_evidence_present = bool(
        selected_chunk_ids
        or selected_span_ids
        or selected_parent_ids
        or selected_media_ids
        or result.get("retrieval_documents")
        or result.get("evidence_span_documents")
    )
    no_answer_violation = 1.0 if example.no_answer and no_answer_evidence_present else 0.0
    unsupported_abstention_rate = (
        1.0
        if example.no_answer and (bool(result.get("abstained")) or not no_answer_evidence_present)
        else 0.0
    )
    return QueryRetrievalScore(
        id=example.id,
        query=example.query,
        query_type=example.query_type,
        source_type=example.source_type,
        language=example.language,
        benchmark_tags=[
            str(tag)
            for tag in (dict(example.metadata or {}).get("benchmark_tags") or [])
            if str(tag)
        ],
        mode=str(result.get("mode") or ""),
        no_answer=example.no_answer,
        has_gold_chunks=bool(gold_chunk_ids),
        has_gold_spans=bool(gold_span_ids),
        has_gold_parents=bool(gold_parent_ids),
        has_gold_media=bool(gold_media_ids),
        seed_chunk_hit_at_5=seed_hit,
        chunk_hit_at_5=_hit_at_k(selected_chunk_ids, gold_chunk_ids, 5),
        chunk_hit_at_10=chunk_hit_10,
        chunk_recall_at_10=_recall_at_k(selected_chunk_ids, recall_gold_chunk_ids, 10),
        chunk_mrr_at_10=_mrr_at_k(selected_chunk_ids, gold_chunk_ids, 10),
        chunk_ndcg_at_10=max(exact_ndcg, source_ndcg),
        chunk_exact_ndcg_at_10=exact_ndcg,
        chunk_source_ndcg_at_10=source_ndcg,
        parent_hit_at_5=_hit_at_k(selected_parent_ids, gold_parent_ids, 5),
        media_hit_at_1=_hit_at_k(selected_media_ids, gold_media_ids, 1),
        media_hit_at_5=_hit_at_k(selected_media_ids, gold_media_ids, 5),
        media_mrr_at_5=_mrr_at_k(selected_media_ids, gold_media_ids, 5),
        expansion_gain_hit=1.0 if (chunk_hit_10 > 0.0 and seed_hit <= 0.0) else 0.0,
        no_answer_violation=no_answer_violation,
        selected_chunk_ids=selected_chunk_ids,
        selected_span_ids=selected_span_ids,
        selected_parent_ids=selected_parent_ids,
        selected_media_ids=selected_media_ids,
        dense_parent_ids=dense_parent_ids,
        dense_media_ids=dense_media_ids,
        chunk_exact_hit_at_5=chunk_exact_hit_5,
        chunk_exact_hit_at_10=chunk_exact_hit_10,
        chunk_exact_mrr_at_10=_mrr_at_k(selected_chunk_ids, exact_gold_chunk_ids, 10),
        chunk_source_hit_at_5=chunk_source_hit_5,
        chunk_source_hit_at_10=chunk_source_hit_10,
        chunk_source_mrr_at_10=_mrr_at_k(selected_chunk_ids, gold_chunk_ids, 10),
        span_hit_at_5=_hit_at_k(selected_span_ids, gold_span_ids, 5),
        span_hit_at_10=_hit_at_k(selected_span_ids, gold_span_ids, 10),
        span_mrr_at_10=_mrr_at_k(selected_span_ids, recall_gold_span_ids, 10),
        span_source_ndcg_at_10=span_source_ndcg,
        has_required_entities=bool(required_entities),
        has_required_pages=bool(required_pages),
        has_required_sections=bool(required_sections),
        has_expected_citations=bool(expected_citation_urls),
        has_min_distinct_sources=min_distinct_sources > 0,
        required_entity_coverage=_coverage_fraction(
            required_entities,
            selected_text,
            aliases=required_entity_aliases,
        ),
        required_page_coverage=required_page_coverage,
        required_section_coverage=_coverage_fraction(required_sections, selected_text),
        citation_support_rate=citation_support_rate,
        multi_page_coverage_rate=multi_page_coverage_rate,
        unsupported_abstention_rate=unsupported_abstention_rate,
    )


def _aggregate_scores(scores: Sequence[QueryRetrievalScore]) -> Dict[str, float]:
    answerable_scores = [score for score in scores if not score.no_answer]
    no_answer_scores = [score for score in scores if score.no_answer]
    chunk_scores = [score for score in answerable_scores if score.has_gold_chunks]
    span_scores = [score for score in answerable_scores if score.has_gold_spans]
    parent_scores = [score for score in answerable_scores if score.has_gold_parents]
    media_scores = [score for score in answerable_scores if score.has_gold_media]
    entity_coverage_scores = [score for score in answerable_scores if score.has_required_entities]
    page_coverage_scores = [score for score in answerable_scores if score.has_required_pages]
    section_coverage_scores = [score for score in answerable_scores if score.has_required_sections]
    citation_scores = [score for score in answerable_scores if score.has_expected_citations]
    multi_page_scores = [score for score in answerable_scores if score.has_min_distinct_sources]
    return {
        "query_count": float(len(scores)),
        "answerable_query_count": float(len(answerable_scores)),
        "no_answer_query_count": float(len(no_answer_scores)),
        "eligible_chunk_query_count": float(len(chunk_scores)),
        "eligible_span_query_count": float(len(span_scores)),
        "eligible_parent_query_count": float(len(parent_scores)),
        "eligible_media_query_count": float(len(media_scores)),
        "eligible_entity_coverage_query_count": float(len(entity_coverage_scores)),
        "eligible_page_coverage_query_count": float(len(page_coverage_scores)),
        "eligible_section_coverage_query_count": float(len(section_coverage_scores)),
        "eligible_citation_query_count": float(len(citation_scores)),
        "eligible_multi_page_query_count": float(len(multi_page_scores)),
        "seed_chunk_hit_at_5": _mean(score.seed_chunk_hit_at_5 for score in chunk_scores),
        "chunk_hit_at_5": _mean(score.chunk_hit_at_5 for score in chunk_scores),
        "chunk_hit_at_10": _mean(score.chunk_hit_at_10 for score in chunk_scores),
        "chunk_exact_hit_at_5": _mean(score.chunk_exact_hit_at_5 for score in chunk_scores),
        "chunk_exact_hit_at_10": _mean(score.chunk_exact_hit_at_10 for score in chunk_scores),
        "chunk_exact_mrr_at_10": _mean(score.chunk_exact_mrr_at_10 for score in chunk_scores),
        "chunk_source_hit_at_5": _mean(score.chunk_source_hit_at_5 for score in chunk_scores),
        "chunk_source_hit_at_10": _mean(score.chunk_source_hit_at_10 for score in chunk_scores),
        "chunk_source_mrr_at_10": _mean(score.chunk_source_mrr_at_10 for score in chunk_scores),
        "chunk_recall_at_10": _mean(score.chunk_recall_at_10 for score in chunk_scores),
        "chunk_mrr_at_10": _mean(score.chunk_mrr_at_10 for score in chunk_scores),
        "chunk_ndcg_at_10": _mean(score.chunk_ndcg_at_10 for score in chunk_scores),
        "chunk_exact_ndcg_at_10": _mean(score.chunk_exact_ndcg_at_10 for score in chunk_scores),
        "chunk_source_ndcg_at_10": _mean(score.chunk_source_ndcg_at_10 for score in chunk_scores),
        "span_hit_at_5": _mean(score.span_hit_at_5 for score in span_scores),
        "span_hit_at_10": _mean(score.span_hit_at_10 for score in span_scores),
        "span_mrr_at_10": _mean(score.span_mrr_at_10 for score in span_scores),
        "span_source_ndcg_at_10": _mean(score.span_source_ndcg_at_10 for score in span_scores),
        "parent_hit_at_5": _mean(score.parent_hit_at_5 for score in parent_scores),
        "media_hit_at_1": _mean(score.media_hit_at_1 for score in media_scores),
        "media_hit_at_5": _mean(score.media_hit_at_5 for score in media_scores),
        "media_mrr_at_5": _mean(score.media_mrr_at_5 for score in media_scores),
        "expansion_gain_hit_rate": _mean(score.expansion_gain_hit for score in chunk_scores),
        "no_answer_violation_rate": _mean(score.no_answer_violation for score in scores),
        "required_entity_coverage": _mean(score.required_entity_coverage for score in entity_coverage_scores),
        "required_page_coverage": _mean(score.required_page_coverage for score in page_coverage_scores),
        "required_section_coverage": _mean(score.required_section_coverage for score in section_coverage_scores),
        "citation_support_rate": _mean(score.citation_support_rate for score in citation_scores),
        "multi_page_coverage_rate": _mean(score.multi_page_coverage_rate for score in multi_page_scores),
        "unsupported_abstention_rate": _mean(score.unsupported_abstention_rate for score in no_answer_scores),
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


def _query_matches_gate_filter(query: Dict[str, Any], where: Dict[str, Any]) -> bool:
    for key, expected in where.items():
        actual = query.get(key)
        if isinstance(expected, list):
            if actual not in expected:
                return False
            continue
        if actual != expected:
            return False
    return True


def _check_per_query_gates(
    report: Dict[str, Any],
    per_query_gates: Dict[str, Any],
    *,
    epsilon: float,
) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    queries = report.get("queries")
    if not isinstance(queries, list):
        return [
            {
                "section": "per_query",
                "slice": "queries",
                "metric": None,
                "reason": "missing_queries",
            }
        ]
    for slice_name, slice_rule in per_query_gates.items():
        if not isinstance(slice_rule, dict):
            continue
        where = slice_rule.get("where")
        if where is None:
            where = {}
        if not isinstance(where, dict):
            failures.append(
                {
                    "section": "per_query",
                    "slice": slice_name,
                    "metric": None,
                    "reason": "invalid_where",
                }
            )
            continue
        matched_queries = [
            query
            for query in queries
            if isinstance(query, dict) and _query_matches_gate_filter(query, where)
        ]
        min_queries = slice_rule.get("min_queries")
        if min_queries is not None and len(matched_queries) + epsilon < float(min_queries):
            failures.append(
                {
                    "section": "per_query",
                    "slice": slice_name,
                    "metric": "matched_query_count",
                    "expected_min": float(min_queries),
                    "actual": float(len(matched_queries)),
                }
            )
        metric_rules = slice_rule.get("metrics") or {}
        if not isinstance(metric_rules, dict):
            failures.append(
                {
                    "section": "per_query",
                    "slice": slice_name,
                    "metric": None,
                    "reason": "invalid_metrics",
                }
            )
            continue
        for metric_name, rule in metric_rules.items():
            if isinstance(rule, (int, float)):
                rule = {"min": float(rule), "max_failures": 0}
            if not isinstance(rule, dict):
                continue
            max_failures = int(rule.get("max_failures", 0) or 0)
            max_failure_rate = rule.get("max_failure_rate")
            failing_queries: List[Dict[str, Any]] = []
            missing_queries: List[str] = []
            for query in matched_queries:
                actual = query.get(metric_name)
                if actual is None:
                    missing_queries.append(str(query.get("id") or ""))
                    continue
                try:
                    actual_float = float(actual)
                except (TypeError, ValueError):
                    missing_queries.append(str(query.get("id") or ""))
                    continue
                minimum = rule.get("min")
                maximum = rule.get("max")
                if minimum is not None and actual_float + epsilon < float(minimum):
                    failing_queries.append(
                        {
                            "id": str(query.get("id") or ""),
                            "query": str(query.get("query") or "")[:240],
                            "actual": actual_float,
                        }
                    )
                elif maximum is not None and actual_float - epsilon > float(maximum):
                    failing_queries.append(
                        {
                            "id": str(query.get("id") or ""),
                            "query": str(query.get("query") or "")[:240],
                            "actual": actual_float,
                        }
                    )
            failure_count = len(failing_queries) + len(missing_queries)
            failure_rate = failure_count / float(max(1, len(matched_queries)))
            rate_failed = max_failure_rate is not None and failure_rate - epsilon > float(max_failure_rate)
            count_failed = failure_count > max_failures
            if count_failed or rate_failed:
                failure: Dict[str, Any] = {
                    "section": "per_query",
                    "slice": slice_name,
                    "metric": metric_name,
                    "failure_count": failure_count,
                    "matched_query_count": len(matched_queries),
                    "failure_rate": failure_rate,
                    "failing_queries": failing_queries[:20],
                    "missing_query_ids": [value for value in missing_queries if value][:20],
                }
                if count_failed:
                    failure["max_failures"] = max_failures
                if rate_failed:
                    failure["expected_max_failure_rate"] = float(max_failure_rate)
                if rule.get("min") is not None:
                    failure["expected_min"] = float(rule["min"])
                if rule.get("max") is not None:
                    failure["expected_max"] = float(rule["max"])
                failures.append(failure)
    return failures


_METRIC_ELIGIBILITY_COUNTS = {
    "seed_chunk_hit_at_5": "eligible_chunk_query_count",
    "chunk_hit_at_5": "eligible_chunk_query_count",
    "chunk_hit_at_10": "eligible_chunk_query_count",
    "chunk_exact_hit_at_5": "eligible_chunk_query_count",
    "chunk_exact_hit_at_10": "eligible_chunk_query_count",
    "chunk_exact_mrr_at_10": "eligible_chunk_query_count",
    "chunk_source_hit_at_5": "eligible_chunk_query_count",
    "chunk_source_hit_at_10": "eligible_chunk_query_count",
    "chunk_source_mrr_at_10": "eligible_chunk_query_count",
    "chunk_recall_at_10": "eligible_chunk_query_count",
    "chunk_mrr_at_10": "eligible_chunk_query_count",
    "chunk_ndcg_at_10": "eligible_chunk_query_count",
    "chunk_exact_ndcg_at_10": "eligible_chunk_query_count",
    "chunk_source_ndcg_at_10": "eligible_chunk_query_count",
    "span_hit_at_5": "eligible_span_query_count",
    "span_hit_at_10": "eligible_span_query_count",
    "span_mrr_at_10": "eligible_span_query_count",
    "span_source_ndcg_at_10": "eligible_span_query_count",
    "parent_hit_at_5": "eligible_parent_query_count",
    "media_hit_at_1": "eligible_media_query_count",
    "media_hit_at_5": "eligible_media_query_count",
    "media_mrr_at_5": "eligible_media_query_count",
    "expansion_gain_hit_rate": "eligible_chunk_query_count",
    "required_entity_coverage": "eligible_entity_coverage_query_count",
    "required_page_coverage": "eligible_page_coverage_query_count",
    "required_section_coverage": "eligible_section_coverage_query_count",
    "citation_support_rate": "eligible_citation_query_count",
    "multi_page_coverage_rate": "eligible_multi_page_query_count",
    "unsupported_abstention_rate": "no_answer_query_count",
}


def _metric_has_no_eligible_queries(metrics: Dict[str, Any], metric_name: str) -> bool:
    count_name = _METRIC_ELIGIBILITY_COUNTS.get(metric_name)
    if not count_name or count_name not in metrics:
        return False
    try:
        return float(metrics.get(count_name) or 0.0) <= 0.0
    except (TypeError, ValueError):
        return False


def _metric_eligibility_count(metrics: Dict[str, Any], metric_name: str) -> tuple[str, float] | None:
    count_name = _METRIC_ELIGIBILITY_COUNTS.get(metric_name)
    if not count_name or count_name not in metrics:
        return None
    try:
        return count_name, float(metrics.get(count_name) or 0.0)
    except (TypeError, ValueError):
        return count_name, 0.0


def _gate_allows_no_eligible_queries(rule: Any) -> bool:
    return isinstance(rule, dict) and bool(rule.get("allow_no_eligible") or rule.get("optional_when_no_eligible"))


def check_metric_gates(report: Dict[str, Any], gates: Dict[str, Any]) -> List[Dict[str, Any]]:
    failures: List[Dict[str, Any]] = []
    if not gates:
        return [
            {
                "section": "gate_contract",
                "slice": "overall",
                "metric": None,
                "reason": "missing_gate_rules",
            }
        ]
    epsilon = 1e-9
    sections = {
        "overall": report.get("overall", {}),
        "by_query_type": report.get("by_query_type", {}),
        "by_source_type": report.get("by_source_type", {}),
        "by_language": report.get("by_language", {}),
        "by_benchmark_tag": report.get("by_benchmark_tag", {}),
    }
    for section_name, section_gates in gates.items():
        if section_name == "per_query":
            if isinstance(section_gates, dict):
                failures.extend(_check_per_query_gates(report, section_gates, epsilon=epsilon))
            continue
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
                if _metric_has_no_eligible_queries(metrics, metric_name):
                    if _gate_allows_no_eligible_queries(rule):
                        continue
                    eligibility = _metric_eligibility_count(metrics, metric_name)
                    failures.append(
                        {
                            "section": section_name,
                            "slice": slice_name,
                            "metric": metric_name,
                            "reason": "no_eligible_queries",
                            "eligible_count_metric": eligibility[0] if eligibility else "",
                            "eligible_query_count": eligibility[1] if eligibility else 0.0,
                        }
                    )
                    continue
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
    progress_callback: EvalProgressCallback | None = None,
    progress_interval: int = 1,
    slow_query_seconds: float = 30.0,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    examples = load_eval_examples(dataset_path)
    try:
        dataset_validation = validate_eval_examples(dataset_path, work_dir=work_dir)
    except Exception as exc:
        dataset_validation = {
            "dataset_path": str(Path(dataset_path).resolve()),
            "work_dir": str(Path(work_dir).resolve()),
            "ok": False,
            "errors": [
                {
                    "field": "dataset_validation",
                    "reason": "validation_exception",
                    "value": f"{type(exc).__name__}: {exc}",
                }
            ],
            "warnings": [],
        }
    _emit_progress(
        progress_callback,
        "retrieval_eval_start",
        query_count=len(examples),
        dataset_path=str(Path(dataset_path).expanduser().resolve()),
        work_dir=str(Path(work_dir).expanduser().resolve()),
    )
    dataset_fingerprint = _dataset_fingerprint(examples)
    _emit_progress(progress_callback, "retrieval_eval_gold_index_start")
    ids_by_url = _load_gold_ids_by_url(work_dir)
    _emit_progress(progress_callback, "retrieval_eval_gold_index_done")
    try:
        from ..core.config import load_effective_config

        config_payload = dict(load_effective_config(config_name, work_dir=work_dir) or {})
    except FileNotFoundError:
        try:
            config_payload = dict(load_config(config_name) or {})
        except FileNotFoundError:
            config_payload = {}
    config_payload = apply_vector_upload_manifest_config(config_payload, work_dir)
    config_fingerprint = _config_fingerprint(config_payload)
    embed_cfg = dict(config_payload.get("embedder") or {})
    cache_model = str(embed_cfg.get("model") or "gemini-embedding-2")
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
    retrieval_errors: List[Dict[str, Any]] = []
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
            score = _score_query(example, cached_result, ids_by_url=ids_by_url)
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

    _emit_progress(
        progress_callback,
        "retrieval_eval_cache_status",
        query_count=len(examples),
        retrieval_cache_hit_count=retrieval_cache_hits,
        retrieval_cache_miss_count=retrieval_cache_misses,
        retrieval_cache_invalid_count=retrieval_cache_invalid,
        retrieval_cache_path=str(retrieval_cache_file) if retrieval_cache_file else "",
        query_cache_path=str(cache_file) if cache_file else "",
    )

    retriever = None
    supports_query_cache = False
    if uncached_examples:
        _emit_progress(
            progress_callback,
            "retrieval_eval_retriever_load_start",
            uncached_query_count=len(uncached_examples),
        )
        retriever_load_started_at = time.perf_counter()
        retriever = AdaptiveHybridRetriever.from_config(config_name=config_name, work_dir=work_dir)
        _emit_progress(
            progress_callback,
            "retrieval_eval_retriever_load_done",
            elapsed_ms=round((time.perf_counter() - retriever_load_started_at) * 1000.0, 3),
            supports_shared_parallel_retrieval=bool(
                getattr(retriever, "supports_shared_parallel_retrieval", False)
            ),
        )
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
            _emit_progress(
                progress_callback,
                "retrieval_eval_query_embedding_start",
                missing_query_count=len(missing_examples),
            )
            embedding_started_at = time.perf_counter()
            if hasattr(retriever, "embed_queries") and callable(getattr(retriever, "embed_queries")):
                embedded_vectors = retriever.embed_queries(miss_queries)
            else:
                embedded_vectors = [retriever.embed_query(query) for query in miss_queries]
            if len(embedded_vectors) != len(missing_examples):
                raise RuntimeError(
                    "Strict evaluation requires one query embedding per missing example: "
                    f"requested={len(missing_examples)} returned={len(embedded_vectors)}"
                )
            for (_example, cache_key), vector in zip(missing_examples, embedded_vectors):
                query_cache[cache_key] = [float(value) for value in vector]
                cache_misses += 1
                cache_dirty = True
            _save_query_embedding_cache(cache_file, query_cache)
            _emit_progress(
                progress_callback,
                "retrieval_eval_query_embedding_done",
                missing_query_count=len(missing_examples),
                elapsed_ms=round((time.perf_counter() - embedding_started_at) * 1000.0, 3),
            )

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

    progress_every = max(1, int(progress_interval or 1))
    slow_query_threshold = max(0.0, float(slow_query_seconds or 0.0))
    requested_workers = max(1, int(parallelism or 1))
    max_workers = requested_workers
    shared_parallel_retriever = bool(
        retriever is not None and getattr(retriever, "supports_shared_parallel_retrieval", False)
    )
    if uncached_examples and max_workers > 1 and not shared_parallel_retriever:
        max_workers = 1
        _emit_progress(
            progress_callback,
            "retrieval_eval_parallelism_capped",
            requested_parallelism=requested_workers,
            effective_parallelism=max_workers,
            reason="retriever_not_marked_thread_safe_for_shared_parallel_retrieval",
        )
    completed_uncached = 0

    def _record_query_progress(
        *,
        index: int,
        example: EvalExample,
        elapsed_ms: float,
        error: str = "",
    ) -> None:
        nonlocal completed_uncached
        completed_uncached += 1
        is_slow = slow_query_threshold > 0 and elapsed_ms >= slow_query_threshold * 1000.0
        if completed_uncached % progress_every == 0 or completed_uncached == len(uncached_examples) or is_slow or error:
            _emit_progress(
                progress_callback,
                "retrieval_eval_query_done",
                id=example.id,
                index=index + 1,
                completed_uncached=completed_uncached,
                uncached_query_count=len(uncached_examples),
                elapsed_ms=round(elapsed_ms, 3),
                slow=bool(is_slow),
                error=error,
            )

    if max_workers == 1 or len(uncached_examples) <= 1:
        for index, example, query_vector, retrieval_key in uncached_examples:
            query_started_at = time.perf_counter()
            error_message = ""
            try:
                result = retriever.retrieve(example.query, query_vector=query_vector)
            except TypeError:
                try:
                    result = retriever.retrieve(example.query)
                except Exception as exc:
                    result = _error_retrieval_result(example, exc)
                    error_message = str(result.get("retrieval_error") or exc)
            except Exception as exc:
                result = _error_retrieval_result(example, exc)
                error_message = str(result.get("retrieval_error") or exc)
            if error_message:
                retrieval_errors.append({"id": example.id, "query": example.query, "error": error_message})
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
            score = _score_query(example, result, ids_by_url=ids_by_url)
            cached_scores[index] = score
            ordered_scores[index] = score
            _record_query_progress(
                index=index,
                example=example,
                elapsed_ms=(time.perf_counter() - query_started_at) * 1000.0,
                error=error_message,
            )
    else:
        thread_state = threading.local()

        def _worker_retriever() -> AdaptiveHybridRetriever:
            if shared_parallel_retriever:
                return retriever
            worker = getattr(thread_state, "retriever", None)
            if worker is None:
                worker = AdaptiveHybridRetriever.from_config(config_name=config_name, work_dir=work_dir)
                thread_state.retriever = worker
            return worker

        def _retrieve_indexed(item: Tuple[int, EvalExample, List[float] | None, str]) -> Tuple[int, str, Dict[str, Any], QueryRetrievalScore, float, str]:
            index, example, query_vector, retrieval_key = item
            worker_retriever = _worker_retriever()
            query_started_at = time.perf_counter()
            error_message = ""
            try:
                result = worker_retriever.retrieve(example.query, query_vector=query_vector)
            except TypeError:
                try:
                    result = worker_retriever.retrieve(example.query)
                except Exception as exc:
                    result = _error_retrieval_result(example, exc)
                    error_message = str(result.get("retrieval_error") or exc)
            except Exception as exc:
                result = _error_retrieval_result(example, exc)
                error_message = str(result.get("retrieval_error") or exc)
            elapsed_ms = (time.perf_counter() - query_started_at) * 1000.0
            return index, retrieval_key, result, _score_query(example, result, ids_by_url=ids_by_url), elapsed_ms, error_message

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_retrieve_indexed, item) for item in uncached_examples]
            for future in as_completed(futures):
                index, retrieval_key, result, score, elapsed_ms, error_message = future.result()
                example = examples[index]
                if error_message:
                    retrieval_errors.append({"id": example.id, "query": example.query, "error": error_message})
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
                _record_query_progress(
                    index=index,
                    example=example,
                    elapsed_ms=elapsed_ms,
                    error=error_message,
                )

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
            "parallelism_requested": requested_workers,
            "parallelism_effective": 1 if len(uncached_examples) <= 1 else max_workers,
            "shared_parallel_retriever": bool(
                max_workers > 1 and shared_parallel_retriever
            ),
            "uncached_query_count": len(uncached_examples),
            "retrieval_error_count": len(retrieval_errors),
            "elapsed_ms": round((time.perf_counter() - started_at) * 1000.0, 3),
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
        "by_language": _slice_scores(scores, "language"),
        "by_benchmark_tag": _slice_scores_by_benchmark_tag(scores),
        "dataset_validation": dataset_validation,
        "retrieval_errors": retrieval_errors,
        "queries": [score.to_dict() for score in scores],
    }
    gates = load_eval_gates(gates_path)
    failures = check_metric_gates(report, gates)
    validation_errors = dataset_validation.get("errors") if isinstance(dataset_validation, dict) else []
    if validation_errors:
        failures.insert(
            0,
            {
                "section": "dataset_validation",
                "slice": "overall",
                "metric": "gold_id_validity",
                "reason": "invalid_eval_dataset",
                "error_count": len(validation_errors),
                "sample_errors": validation_errors[:5],
            },
        )
    report["gates"] = {
        "path": str(Path(gates_path).resolve()) if gates_path else "",
        "passed": not failures,
        "failures": failures,
    }
    _emit_progress(
        progress_callback,
        "retrieval_eval_done",
        query_count=len(scores),
        gates_passed=not failures,
        failure_count=len(failures),
        retrieval_error_count=len(retrieval_errors),
        elapsed_ms=report["execution"]["elapsed_ms"],
    )
    return report
