from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from pipeline.evaluation.dataset import EvalExample
from pipeline.evaluation.multilingual_v2 import normalize_evidence_text


_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0600-\u06ff]+", flags=re.IGNORECASE)
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")
_ARABIC_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
    }
)


def _light_arabic_stem(token: str) -> str:
    value = token
    for prefix in ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ك", "ل"):
        if value.startswith(prefix) and len(value) - len(prefix) >= 3:
            value = value[len(prefix) :]
            break
    for suffix in ("يات", "ات", "ون", "ين", "يه", "ها", "هم", "هن", "كم", "نا"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 3:
            value = value[: -len(suffix)]
            break
    return value


def tokenize_multilingual(value: Any) -> List[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = _ARABIC_DIACRITICS_RE.sub("", text).replace("ـ", "").translate(_ARABIC_TRANSLATION)
    output: List[str] = []
    for match in _TOKEN_RE.finditer(text):
        token = match.group(0)
        if not token:
            continue
        output.append(token)
        if _ARABIC_RE.search(token):
            stem = _light_arabic_stem(token)
            if stem != token:
                output.append(f"ar:{stem}")
    return output


def record_matches_source(record: Mapping[str, Any], source_key: str) -> bool:
    source_key = str(source_key or "")
    if not source_key:
        return False
    if source_key == str(record.get("id") or ""):
        return True
    if source_key.startswith("document-revision:"):
        return source_key == str(record.get("document_revision_id") or "")
    if source_key.startswith("page-card:"):
        return source_key in set(str(value) for value in record.get("page_card_ids") or [])
    if source_key.startswith("page-section:"):
        return source_key in set(str(value) for value in record.get("section_ids") or [])
    if source_key.startswith("page-action:"):
        return source_key == str(record.get("action_id") or "")
    if source_key.startswith("media:"):
        media_id = str(record.get("media_id") or "")
        return source_key == media_id or source_key.removeprefix("media:") == media_id
    return False


def _source_keys(example: EvalExample) -> List[str]:
    metadata_keys = [str(value) for value in (example.metadata or {}).get("source_keys") or [] if str(value)]
    if metadata_keys:
        return list(dict.fromkeys(metadata_keys))
    return list(
        dict.fromkeys(
            [
                *example.gold_document_revision_ids,
                *example.gold_page_card_ids,
                *example.gold_section_ids,
                *example.gold_media_ids,
            ]
        )
    )


def _record_evidence_text(record: Mapping[str, Any]) -> str:
    return normalize_evidence_text(
        f"{record.get('raw_text') or ''}\n{record.get('text') or ''}"
    )


def score_ranking(
    example: EvalExample,
    ranked: Sequence[tuple[Mapping[str, Any], float]],
    *,
    evidence_k: int = 10,
    action_k: int = 5,
    media_k: int = 5,
) -> Dict[str, Any]:
    top_score = float(ranked[0][1]) if ranked else float("-inf")
    base = {
        "id": example.id,
        "language": example.language,
        "query_type": example.query_type,
        "source_type": example.source_type,
        "no_answer": example.no_answer,
        "top_score": top_score,
        "evidence_ndcg_at_10": 0.0,
        "evidence_mrr_at_10": 0.0,
        "evidence_hit_at_10": 0.0,
        "source_recall_at_10": 0.0,
        "navigation_action_hit_at_5": 0.0,
        "media_hit_at_5": 0.0,
        "ranked_ids_at_10": [str(record.get("id") or "") for record, _ in ranked[:10]],
    }
    if example.no_answer:
        return base

    normalized_text_by_id: Dict[str, str] = {}
    evidence_units = []
    for unit in (example.metadata or {}).get("evidence_quotes") or []:
        if not isinstance(unit, Mapping):
            continue
        source_key = str(unit.get("source_key") or "")
        quote = normalize_evidence_text(unit.get("quote"))
        if source_key and quote:
            evidence_units.append((source_key, quote))
    first_ranks: List[int | None] = []
    for source_key, quote in evidence_units:
        first_rank = None
        for rank, (record, _score) in enumerate(ranked[:evidence_k], start=1):
            if not record_matches_source(record, source_key):
                continue
            record_id = str(record.get("id") or "")
            evidence_text = normalized_text_by_id.setdefault(
                record_id, _record_evidence_text(record)
            )
            structured_gold_match = (
                str(record.get("action_id") or "") in set(example.gold_action_ids)
                or str(record.get("media_id") or "") in set(example.gold_media_ids)
            )
            if quote in evidence_text or structured_gold_match:
                first_rank = rank
                break
        first_ranks.append(first_rank)
    if first_ranks:
        discounted = [
            1.0 / math.log2(rank + 1.0) if rank is not None else 0.0
            for rank in first_ranks
        ]
        base["evidence_ndcg_at_10"] = float(sum(discounted) / len(discounted))
        matched_ranks = [rank for rank in first_ranks if rank is not None]
        if matched_ranks:
            base["evidence_mrr_at_10"] = 1.0 / float(min(matched_ranks))
            base["evidence_hit_at_10"] = 1.0

    sources = _source_keys(example)
    if sources:
        covered = sum(
            1
            for source_key in sources
            if any(
                record_matches_source(record, source_key)
                for record, _score in ranked[:evidence_k]
            )
        )
        base["source_recall_at_10"] = covered / float(len(sources))

    gold_actions = set(example.gold_action_ids)
    if gold_actions:
        base["navigation_action_hit_at_5"] = float(
            any(str(record.get("action_id") or "") in gold_actions for record, _ in ranked[:action_k])
        )
    gold_media = set(example.gold_media_ids)
    if gold_media:
        base["media_hit_at_5"] = float(
            any(str(record.get("media_id") or "") in gold_media for record, _ in ranked[:media_k])
        )
    return base


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key) or 0.0) for row in rows) / len(rows)) if rows else 0.0


def aggregate_quality(
    rows: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    answerable = [row for row in rows if not row.get("no_answer")]
    arabic = [row for row in answerable if row.get("language") == "Arabic"]
    synthesis = [row for row in answerable if row.get("query_type") == "synthesis"]
    navigation = [
        row
        for row in answerable
        if float(row.get("navigation_action_hit_at_5") or 0.0) > 0.0
        or bool(row.get("has_gold_actions"))
    ]
    multimodal = [
        row
        for row in answerable
        if float(row.get("media_hit_at_5") or 0.0) > 0.0
        or bool(row.get("has_gold_media"))
    ]
    unsupported = [row for row in rows if row.get("no_answer")]
    answerable_accept_accuracy = _mean(answerable, "abstention_correct")
    unsupported_abstain_accuracy = _mean(unsupported, "abstention_correct")
    if answerable and unsupported:
        abstention_balanced_accuracy = (
            answerable_accept_accuracy + unsupported_abstain_accuracy
        ) / 2.0
    elif answerable:
        abstention_balanced_accuracy = answerable_accept_accuracy
    else:
        abstention_balanced_accuracy = unsupported_abstain_accuracy
    components = {
        "evidence_ndcg_at_10": _mean(answerable, "evidence_ndcg_at_10"),
        "evidence_mrr_at_10": _mean(answerable, "evidence_mrr_at_10"),
        "source_recall_at_10": _mean(answerable, "source_recall_at_10"),
        "arabic_evidence_hit_at_10": _mean(arabic, "evidence_hit_at_10"),
        "synthesis_source_recall_at_10": _mean(synthesis, "source_recall_at_10"),
        "navigation_action_hit_at_5": _mean(navigation, "navigation_action_hit_at_5"),
        "media_hit_at_5": _mean(multimodal, "media_hit_at_5"),
        "abstention_balanced_accuracy": abstention_balanced_accuracy,
    }
    selection_score = sum(
        float(weights.get(key) or 0.0) * value for key, value in components.items()
    )
    return {
        **components,
        "selection_score": float(selection_score),
        "query_count": len(rows),
        "answerable_count": len(answerable),
        "no_answer_count": len(rows) - len(answerable),
        "arabic_answerable_count": len(arabic),
        "synthesis_count": len(synthesis),
        "navigation_count": len(navigation),
        "multimodal_count": len(multimodal),
    }


def annotate_abstention_decisions(
    rows: Sequence[Dict[str, Any]], threshold: float
) -> None:
    """Attach frozen-threshold decisions used by aggregate quality/bootstrap."""

    for row in rows:
        accepted = float(row.get("top_score", float("-inf"))) >= threshold
        row["accepted_at_frozen_threshold"] = bool(accepted)
        row["abstention_correct"] = float(
            (not accepted) if row.get("no_answer") else accepted
        )


def annotate_gold_presence(
    row: Dict[str, Any], example: EvalExample
) -> Dict[str, Any]:
    row["has_gold_actions"] = bool(example.gold_action_ids)
    row["has_gold_media"] = bool(example.gold_media_ids)
    return row


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    top_k_per_ranking: int = 100,
    rrf_k: int = 60,
    output_k: int = 200,
) -> List[tuple[int, float]]:
    scores: Counter[int] = Counter()
    for ranking in rankings:
        for rank, record_index in enumerate(ranking[:top_k_per_ranking], start=1):
            scores[int(record_index)] += 1.0 / float(rrf_k + rank)
    return sorted(scores.items(), key=lambda item: (-float(item[1]), int(item[0])))[:output_k]


def choose_abstention_threshold(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    finite_rows = [row for row in rows if math.isfinite(float(row.get("top_score", float("-inf"))))]
    if not finite_rows:
        return {"threshold": 0.0, "balanced_accuracy": 0.0}
    values = sorted(set(float(row["top_score"]) for row in finite_rows))
    candidates = [values[0] - 1e-9, *[(left + right) / 2.0 for left, right in zip(values, values[1:])], values[-1] + 1e-9]
    best: Dict[str, Any] | None = None
    for threshold in candidates:
        answerable = [row for row in finite_rows if not row.get("no_answer")]
        unsupported = [row for row in finite_rows if row.get("no_answer")]
        true_positive_rate = _mean(
            [{"value": float(float(row["top_score"]) >= threshold)} for row in answerable],
            "value",
        )
        true_negative_rate = _mean(
            [{"value": float(float(row["top_score"]) < threshold)} for row in unsupported],
            "value",
        )
        balanced = (true_positive_rate + true_negative_rate) / 2.0
        candidate = {
            "threshold": float(threshold),
            "balanced_accuracy": balanced,
            "answerable_accept_rate": true_positive_rate,
            "no_answer_abstain_rate": true_negative_rate,
        }
        if best is None or (
            candidate["balanced_accuracy"],
            candidate["no_answer_abstain_rate"],
            candidate["answerable_accept_rate"],
        ) > (
            best["balanced_accuracy"],
            best["no_answer_abstain_rate"],
            best["answerable_accept_rate"],
        ):
            best = candidate
    return best or {"threshold": 0.0, "balanced_accuracy": 0.0}


def evaluate_abstention_threshold(
    rows: Sequence[Mapping[str, Any]], threshold: float
) -> Dict[str, Any]:
    answerable = [row for row in rows if not row.get("no_answer")]
    unsupported = [row for row in rows if row.get("no_answer")]
    accept = (
        sum(float(float(row.get("top_score", float("-inf"))) >= threshold) for row in answerable)
        / len(answerable)
        if answerable
        else 0.0
    )
    abstain = (
        sum(float(float(row.get("top_score", float("-inf"))) < threshold) for row in unsupported)
        / len(unsupported)
        if unsupported
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "balanced_accuracy": (accept + abstain) / 2.0,
        "answerable_accept_rate": accept,
        "no_answer_abstain_rate": abstain,
    }


def paired_bootstrap_delta(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    *,
    weights: Mapping[str, float],
    samples: int,
    seed: int,
) -> Dict[str, float]:
    candidate_by_id = {str(row["id"]): row for row in candidate_rows}
    baseline_by_id = {str(row["id"]): row for row in baseline_rows}
    ids = sorted(set(candidate_by_id) & set(baseline_by_id))
    if not ids:
        return {"mean_delta": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = np.random.default_rng(seed)
    deltas = np.empty(max(1, samples), dtype=np.float64)
    for sample_index in range(len(deltas)):
        selected = rng.integers(0, len(ids), size=len(ids))
        candidate_sample = [candidate_by_id[ids[index]] for index in selected]
        baseline_sample = [baseline_by_id[ids[index]] for index in selected]
        deltas[sample_index] = (
            aggregate_quality(candidate_sample, weights=weights)["selection_score"]
            - aggregate_quality(baseline_sample, weights=weights)["selection_score"]
        )
    return {
        "mean_delta": float(np.mean(deltas)),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
    }
