"""Deterministic routing and benchmark sampling for hybrid image OCR."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


UNLIMITED_OCR_ROUTE = "unlimited_ocr"
SCENE_OCR_ROUTE = "paddleocr_scene"

# The production route is evidence-driven: fast scene OCR runs first for every
# asset. Unlimited-OCR is reserved for difficult document-like assets rejected
# by the first pass.
_UNLIMITED_ESCALATION_KINDS = {"document_fragment", "screenshot"}

BENCHMARK_BUCKETS: Tuple[Tuple[str, Tuple[str, ...], int], ...] = (
    ("document_fragment", ("document_fragment",), 4),
    ("screenshot", ("screenshot",), 4),
    ("map", ("map",), 3),
    ("structured_visual", ("chart", "diagram", "infographic"), 2),
    ("photo", ("photo",), 3),
    ("portrait", ("portrait",), 2),
    ("logo", ("logo",), 1),
    ("illustration", ("illustration",), 1),
)

_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")


def choose_ocr_route(item: Mapping[str, Any]) -> str:
    """Choose the production primary engine from stable semantic metadata."""

    del item
    return SCENE_OCR_ROUTE


def should_escalate_to_unlimited(
    item: Mapping[str, Any], scene_result: Mapping[str, Any]
) -> bool:
    """Escalate only rejected/failed document-like first-pass results."""

    image_kind = str(item.get("image_kind") or "").strip().lower()
    if image_kind not in _UNLIMITED_ESCALATION_KINDS:
        return False
    status = str(scene_result.get("status") or "failed")
    if status in {"failed", "rejected_low_quality"}:
        return True
    if status != "no_readable_text":
        return False
    reference_count = sum(_word_counts(str(item.get("visible_text_reference") or "")).values())
    return reference_count >= 12


def _unique_candidates(items: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_hash: Dict[str, Dict[str, Any]] = {}
    for raw in items:
        item = dict(raw)
        if item.get("type") != "image" or item.get("needs_ocr") is not True:
            continue
        content_hash = str(item.get("content_hash") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            continue
        current = by_hash.get(content_hash)
        if current is None or len(str(item.get("visible_text") or "")) > len(
            str(current.get("visible_text") or "")
        ):
            item["content_hash"] = content_hash
            by_hash[content_hash] = item
    return [by_hash[key] for key in sorted(by_hash)]


def _candidate_key(item: Mapping[str, Any]) -> Tuple[int, int, str]:
    visible_text = str(item.get("visible_text") or "")
    # Ensure the small audit set includes multilingual content, then prefer
    # text-bearing examples which can be independently compared across engines.
    return (
        0 if _ARABIC_RE.search(visible_text) else 1,
        -min(4000, len(visible_text)),
        str(item.get("content_hash") or ""),
    )


def _balanced_take(candidates: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in sorted(candidates, key=_candidate_key):
        source_type = str(item.get("source_type") or "unknown").strip().lower()
        by_source[source_type].append(item)

    selected: List[Dict[str, Any]] = []
    source_order = [source for source in ("pdf", "html") if by_source.get(source)]
    source_order.extend(sorted(source for source in by_source if source not in source_order))
    while len(selected) < count and source_order:
        next_order: List[str] = []
        for source in source_order:
            values = by_source[source]
            if values and len(selected) < count:
                selected.append(values.pop(0))
            if values:
                next_order.append(source)
        source_order = next_order
    return selected


def build_stratified_benchmark(
    items: Iterable[Mapping[str, Any]],
    *,
    buckets: Sequence[Tuple[str, Sequence[str], int]] = BENCHMARK_BUCKETS,
) -> List[Dict[str, Any]]:
    """Return a deterministic, source-balanced 20-image OCR audit set."""

    candidates = _unique_candidates(items)
    selected_hashes = set()
    selected: List[Dict[str, Any]] = []
    for bucket_name, kinds, quota in buckets:
        accepted_kinds = {str(kind).lower() for kind in kinds}
        available = [
            item
            for item in candidates
            if item["content_hash"] not in selected_hashes
            and str(item.get("image_kind") or "").lower() in accepted_kinds
        ]
        chosen = _balanced_take(available, max(0, int(quota)))
        if len(chosen) != max(0, int(quota)):
            raise ValueError(
                f"Benchmark bucket {bucket_name!r} requires {quota} images; "
                f"only {len(chosen)} were available"
            )
        for item in chosen:
            content_hash = str(item["content_hash"])
            selected_hashes.add(content_hash)
            selected.append(
                {
                    **item,
                    "benchmark_bucket": bucket_name,
                    "ocr_route": choose_ocr_route(item),
                }
            )
    return selected


def batch_contract_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    """Hash the immutable remote-work contract independently of local paths."""

    contract = [
        {
            "content_hash": str(item.get("content_hash") or ""),
            "image_file": str(item.get("image_file") or ""),
            "image_kind": str(item.get("image_kind") or ""),
            "ocr_route": str(item.get("ocr_route") or choose_ocr_route(item)),
        }
        for item in items
    ]
    return hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_local_assets(items: Iterable[Mapping[str, Any]]) -> None:
    """Fail closed when a selected image is missing or no longer hash-identical."""

    for item in items:
        path = Path(str(item.get("local_path") or "")).resolve()
        if not path.is_file():
            raise ValueError(f"Selected OCR asset is missing: {path}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = str(item.get("content_hash") or "").lower()
        if actual != expected:
            raise ValueError(f"Selected OCR asset hash mismatch: {path}")


def assess_scene_ocr_lines(
    lines: Iterable[Mapping[str, Any]],
    *,
    reference_text: str = "",
    minimum_line_confidence: float = 0.55,
    minimum_mean_confidence: float = 0.75,
    minimum_alphanumeric_chars: int = 2,
    minimum_reference_token_count: int = 12,
    minimum_reference_recall: float = 0.45,
    minimum_candidate_reference_size_ratio: float = 0.60,
) -> Dict[str, Any]:
    """Gate PaddleOCR text using its recognition scores without inventing text."""

    normalized: List[Dict[str, Any]] = []
    for raw in lines:
        text = " ".join(str(raw.get("text") or "").split()).strip()
        if not text:
            continue
        try:
            confidence = min(1.0, max(0.0, float(raw.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        normalized.append(
            {
                "text": text,
                "confidence": confidence,
                "box": list(raw.get("box") or []),
            }
        )
    accepted = [
        line for line in normalized if line["confidence"] >= minimum_line_confidence
    ]
    candidate_text = "\n".join(line["text"] for line in accepted).strip()
    alphanumeric_chars = sum(character.isalnum() for character in candidate_text)
    weight = sum(max(1, sum(character.isalnum() for character in line["text"])) for line in accepted)
    weighted_confidence = (
        sum(
            line["confidence"]
            * max(1, sum(character.isalnum() for character in line["text"]))
            for line in accepted
        )
        / max(1, weight)
    )

    flags: List[str] = []
    if alphanumeric_chars < max(0, int(minimum_alphanumeric_chars)):
        if normalized:
            flags.append("no_confident_readable_text")
        else:
            flags.append("no_readable_text")
    if accepted and weighted_confidence < minimum_mean_confidence:
        flags.append("mean_confidence_below_threshold")
    corroboration = assess_reference_coverage(
        reference_text,
        candidate_text,
        minimum_reference_token_count=minimum_reference_token_count,
        minimum_reference_recall=minimum_reference_recall,
        minimum_candidate_reference_size_ratio=minimum_candidate_reference_size_ratio,
    )
    if candidate_text and corroboration["required"] and not corroboration["passed"]:
        flags.append("insufficient_visible_text_corroboration")

    if "no_readable_text" in flags:
        status = "no_readable_text"
    elif flags:
        status = "rejected_low_quality"
    else:
        status = "completed"
    return {
        "status": status,
        "text": candidate_text if status == "completed" else "",
        "candidate_text": candidate_text,
        "quality_score": round(weighted_confidence, 6) if accepted else 0.0,
        "quality_flags": flags,
        "quality_metrics": {
            "detected_line_count": len(normalized),
            "accepted_line_count": len(accepted),
            "rejected_line_count": len(normalized) - len(accepted),
            "alphanumeric_character_count": alphanumeric_chars,
            "minimum_line_confidence": minimum_line_confidence,
            "weighted_mean_confidence": round(weighted_confidence, 6)
            if accepted
            else 0.0,
            "reference_corroboration": corroboration,
        },
        "accepted_lines": accepted,
    }


def _word_counts(value: Any) -> Counter[str]:
    return Counter(re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE))


def assess_reference_coverage(
    reference_text: str,
    candidate_text: str,
    *,
    minimum_reference_token_count: int = 12,
    minimum_reference_recall: float = 0.45,
    minimum_candidate_reference_size_ratio: float = 0.60,
) -> Dict[str, Any]:
    """Use existing VLM-visible text only as a conservative OCR cross-check.

    Additional valid OCR text is allowed. A candidate is blocked only when it
    is both substantially shorter than an informative reference and fails to
    recover enough of that reference. This catches confident gibberish while
    avoiding a claim that the VLM transcription is human ground truth.
    """

    reference = _word_counts(reference_text)
    candidate = _word_counts(candidate_text)
    reference_count = sum(reference.values())
    candidate_count = sum(candidate.values())
    overlap = sum((reference & candidate).values())
    recall = overlap / max(1, reference_count)
    size_ratio = candidate_count / max(1, reference_count)
    required = reference_count >= max(1, int(minimum_reference_token_count))
    passed = (
        not required
        or size_ratio >= minimum_candidate_reference_size_ratio
        or recall >= minimum_reference_recall
    )
    return {
        "required": required,
        "passed": passed,
        "reference_token_count": reference_count,
        "candidate_token_count": candidate_count,
        "overlap_token_count": overlap,
        "reference_recall": round(recall, 6),
        "candidate_reference_size_ratio": round(size_ratio, 6),
        "reference_is_advisory_not_ground_truth": True,
    }
