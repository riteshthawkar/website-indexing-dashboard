#!/usr/bin/env python3
"""Merge raw scene/escalation batches into one conservative exact-OCR result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.hybrid_ocr import assess_reference_coverage, assess_scene_ocr_lines
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.unlimited_ocr import assess_ocr_quality


QUALITY_REVISION = "hybrid-exact-ocr-quality-v1"
MINIMUM_COMPLETED_RATIO = 0.55
MAXIMUM_REJECTED_RATIO = 0.25


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--scene-results", type=Path, required=True)
    parser.add_argument("--unlimited-batch", type=Path)
    parser.add_argument("--unlimited-results", type=Path)
    parser.add_argument("--unlimited-retry-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _failed(raw: Mapping[str, Any] | None, error: str) -> Dict[str, Any]:
    return {
        "status": "failed",
        "text": "",
        "candidate_text": "",
        "quality_score": 0.0,
        "quality_flags": ["provider_failure"],
        "quality_metrics": {},
        "error": str((raw or {}).get("error") or error),
    }


def _scene_result(raw: Mapping[str, Any] | None, reference: str) -> Dict[str, Any]:
    if not raw or raw.get("status") != "raw_completed":
        return _failed(raw, "missing_or_failed_scene_result")
    quality = assess_scene_ocr_lines(raw.get("raw_lines") or [], reference_text=reference)
    return {
        **quality,
        "provider": str(raw.get("provider") or "paddleocr"),
        "provider_revision": str(raw.get("provider_revision") or ""),
        "model": str(raw.get("model") or ""),
        "model_revision": str(raw.get("model_revision") or ""),
        "mode": "scene_text",
        "prompt_revision": "",
        "raw_output_sha256": str(raw.get("raw_output_sha256") or ""),
        "latency_ms": raw.get("latency_ms"),
        "completed_at": str(raw.get("completed_at") or ""),
        "raw_evidence": {"lines": list(raw.get("raw_lines") or [])},
    }


def _unlimited_result(raw: Mapping[str, Any] | None, reference: str) -> Dict[str, Any]:
    if not raw or raw.get("status") != "raw_completed":
        return _failed(raw, "missing_or_failed_unlimited_result")
    quality = assess_ocr_quality(str(raw.get("raw_output") or ""))
    corroboration = assess_reference_coverage(reference, str(quality.get("candidate_text") or ""))
    if (
        quality.get("status") == "completed"
        and corroboration["required"]
        and not corroboration["passed"]
    ):
        quality["status"] = "rejected_low_quality"
        quality["text"] = ""
        quality["quality_flags"] = list(quality.get("quality_flags") or []) + [
            "insufficient_visible_text_corroboration"
        ]
    quality.setdefault("quality_metrics", {})["reference_corroboration"] = corroboration
    return {
        **quality,
        "provider": str(raw.get("provider") or "transformers_direct"),
        "provider_revision": str(raw.get("provider_revision") or ""),
        "model": str(raw.get("model") or "baidu/Unlimited-OCR"),
        "model_revision": str(raw.get("model_revision") or ""),
        "mode": str(raw.get("mode") or "gundam"),
        "prompt_revision": str(raw.get("prompt_revision") or "unlimited-ocr-document-v1"),
        "raw_output_sha256": str(raw.get("raw_output_sha256") or ""),
        "latency_ms": raw.get("latency_ms"),
        "completed_at": str(raw.get("completed_at") or ""),
        "raw_evidence": {"raw_output": str(raw.get("raw_output") or "")},
    }


def _select_unlimited_result(candidates: list[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not candidates:
        return None
    status_rank = {
        "completed": 3,
        "no_readable_text": 2,
        "rejected_low_quality": 1,
        "failed": 0,
    }
    selected = max(
        candidates,
        key=lambda value: (
            status_rank.get(str(value.get("status") or ""), -1),
            float(value.get("quality_score") or 0.0),
            -float(value.get("latency_ms") or 0.0),
        ),
    )
    selected["attempt_results"] = [
        {
            "mode": str(value.get("mode") or ""),
            "status": str(value.get("status") or "failed"),
            "quality_score": value.get("quality_score"),
            "quality_flags": list(value.get("quality_flags") or []),
            "latency_ms": value.get("latency_ms"),
        }
        for value in candidates
    ]
    return selected


def _latency_summary(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(statistics.median(ordered), 3),
        "p95": round(ordered[p95_index], 3),
        "max": round(ordered[-1], 3),
    }


def main() -> int:
    args = _args()
    batch = load_json_safe(args.batch, {}) or {}
    scene_payload = load_json_safe(args.scene_results, {}) or {}
    full_contract = str(batch.get("batch_contract_sha256") or "")
    if str(scene_payload.get("batch_contract_sha256") or "") != full_contract:
        raise ValueError("Scene result contract does not match the full batch")
    unlimited_batch = load_json_safe(args.unlimited_batch, {}) or {} if args.unlimited_batch else {}
    unlimited_payload = (
        load_json_safe(args.unlimited_results, {}) or {} if args.unlimited_results else {}
    )
    unlimited_retry_payload = (
        load_json_safe(args.unlimited_retry_results, {}) or {}
        if args.unlimited_retry_results
        else {}
    )
    if bool(unlimited_batch) != bool(unlimited_payload):
        raise ValueError("Unlimited escalation batch and results must be supplied together")
    if unlimited_payload and str(unlimited_payload.get("batch_contract_sha256") or "") != str(
        unlimited_batch.get("batch_contract_sha256") or ""
    ):
        raise ValueError("Unlimited result contract does not match its escalation batch")
    if unlimited_retry_payload and str(
        unlimited_retry_payload.get("batch_contract_sha256") or ""
    ) != str(unlimited_batch.get("batch_contract_sha256") or ""):
        raise ValueError("Unlimited retry result contract does not match its escalation batch")
    if unlimited_batch and str(unlimited_batch.get("parent_batch_contract_sha256") or "") != full_contract:
        raise ValueError("Unlimited escalation does not belong to the full batch")

    scene_results = scene_payload.get("results") or {}
    unlimited_results = unlimited_payload.get("results") or {}
    unlimited_retry_results = unlimited_retry_payload.get("results") or {}
    results: Dict[str, Dict[str, Any]] = {}
    for item in batch.get("items") or []:
        content_hash = str(item["content_hash"])
        reference = str(item.get("visible_text_reference") or "")
        scene = _scene_result(scene_results.get(content_hash), reference)
        unlimited_candidates = [
            _unlimited_result(raw, reference)
            for raw in (
                unlimited_results.get(content_hash),
                unlimited_retry_results.get(content_hash),
            )
            if isinstance(raw, dict)
        ]
        unlimited = _select_unlimited_result(unlimited_candidates)
        if scene.get("status") == "completed":
            selected = scene
            selection_reason = "scene_completed"
        elif unlimited and unlimited.get("status") == "completed":
            selected = unlimited
            selection_reason = "unlimited_recovered_scene_rejection"
        elif scene.get("status") in {"no_readable_text", "rejected_low_quality"}:
            selected = scene
            selection_reason = "scene_terminal_no_usable_escalation"
        elif unlimited:
            selected = unlimited
            selection_reason = "scene_failed_unlimited_terminal"
        else:
            selected = scene
            selection_reason = "scene_failed_without_escalation"
        results[content_hash] = {
            **selected,
            "content_hash": content_hash,
            "quality_revision": QUALITY_REVISION,
            "selection_reason": selection_reason,
            "attempts": 1 + len(unlimited_candidates),
            "scene_status": str(scene.get("status") or "failed"),
            "unlimited_status": str((unlimited or {}).get("status") or "not_requested"),
            "unlimited_attempt_results": list(
                (unlimited or {}).get("attempt_results") or []
            ),
            "ocr_input_hash": hashlib.sha256(
                f"{full_contract}:{content_hash}:{QUALITY_REVISION}".encode("utf-8")
            ).hexdigest(),
        }

    status_counts = Counter(str(value.get("status") or "failed") for value in results.values())
    selected_provider_counts = Counter(
        str(value.get("provider") or "") for value in results.values()
    )
    selection_reason_counts = Counter(
        str(value.get("selection_reason") or "") for value in results.values()
    )
    quality_flag_counts = Counter(
        str(flag)
        for value in results.values()
        for flag in value.get("quality_flags") or []
    )
    status_by_image_kind: Dict[str, Counter[str]] = {}
    for item in batch.get("items") or []:
        content_hash = str(item["content_hash"])
        image_kind = str(item.get("image_kind") or "unknown")
        status_by_image_kind.setdefault(image_kind, Counter()).update(
            [str(results[content_hash].get("status") or "failed")]
        )
    scene_latencies = [
        float(value.get("latency_ms") or 0.0)
        for value in scene_results.values()
        if isinstance(value, dict)
    ]
    unlimited_latencies = [
        float(value.get("latency_ms") or 0.0)
        for payload_results in (unlimited_results, unlimited_retry_results)
        for value in payload_results.values()
        if isinstance(value, dict)
    ]
    selected_count = len(results)
    completed_ratio = status_counts.get("completed", 0) / max(1, selected_count)
    rejected_ratio = status_counts.get("rejected_low_quality", 0) / max(1, selected_count)
    gates = {
        "all_requests_adjudicated": status_counts.get("failed", 0) == 0,
        "minimum_completed_ratio": MINIMUM_COMPLETED_RATIO,
        "completed_ratio": round(completed_ratio, 6),
        "completed_ratio_passed": completed_ratio >= MINIMUM_COMPLETED_RATIO,
        "maximum_rejected_ratio": MAXIMUM_REJECTED_RATIO,
        "rejected_ratio": round(rejected_ratio, 6),
        "rejected_ratio_passed": rejected_ratio <= MAXIMUM_REJECTED_RATIO,
    }
    gates["passed"] = all(
        gates[key]
        for key in (
            "all_requests_adjudicated",
            "completed_ratio_passed",
            "rejected_ratio_passed",
        )
    )
    output = {
        "version": 1,
        "kind": "hybrid_ocr_adjudicated_results",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_contract_sha256": full_contract,
        "quality_revision": QUALITY_REVISION,
        "selected_count": len(results),
        "status_counts": dict(sorted(status_counts.items())),
        "selected_provider_counts": dict(sorted(selected_provider_counts.items())),
        "selection_reason_counts": dict(sorted(selection_reason_counts.items())),
        "quality_flag_counts": dict(sorted(quality_flag_counts.items())),
        "status_by_image_kind": {
            image_kind: dict(sorted(counts.items()))
            for image_kind, counts in sorted(status_by_image_kind.items())
        },
        "exact_text_character_count": sum(
            len(str(value.get("text") or ""))
            for value in results.values()
            if value.get("status") == "completed"
        ),
        "scene_preprocessed_conversion_count": sum(
            bool(value.get("preprocessing", {}).get("converted"))
            for value in scene_results.values()
            if isinstance(value, dict)
        ),
        "latency_ms": {
            "scene": _latency_summary(scene_latencies),
            "unlimited_all_attempts": _latency_summary(unlimited_latencies),
        },
        "all_requests_adjudicated": status_counts.get("failed", 0) == 0,
        "gates": gates,
        "results": results,
    }
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "selected_count": len(results),
                "status_counts": output["status_counts"],
                "selected_provider_counts": output["selected_provider_counts"],
                "all_requests_adjudicated": output["all_requests_adjudicated"],
                "gates": gates,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
