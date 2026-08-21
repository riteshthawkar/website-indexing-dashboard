#!/usr/bin/env python3
"""Adjudicate hybrid OCR benchmark results with reproducible quality metrics."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.hybrid_ocr import assess_reference_coverage, assess_scene_ocr_lines
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.unlimited_ocr import assess_ocr_quality


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--scene-results", type=Path, required=True)
    parser.add_argument("--unlimited-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _tokens(value: Any) -> List[str]:
    return re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE)


def _agreement(reference: str, candidate: str) -> Dict[str, Any]:
    reference_counts = Counter(_tokens(reference))
    candidate_counts = Counter(_tokens(candidate))
    overlap = sum((reference_counts & candidate_counts).values())
    reference_count = sum(reference_counts.values())
    candidate_count = sum(candidate_counts.values())
    precision = overlap / max(1, candidate_count)
    recall = overlap / max(1, reference_count)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "reference_token_count": reference_count,
        "candidate_token_count": candidate_count,
        "overlap_token_count": overlap,
        "token_precision": round(precision, 6),
        "token_recall": round(recall, 6),
        "token_f1": round(f1, 6),
        "advisory_only": True,
        "reference_source": "Gemini visible_text; not human ground truth",
    }


def _percentile(values: List[float], proportion: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * proportion) - 1))
    return ordered[index]


def _summary(rows: Iterable[Mapping[str, Any]], engine: str) -> Dict[str, Any]:
    values = [dict(row) for row in rows if isinstance(row.get(engine), Mapping)]
    latencies = [float(row[engine].get("latency_ms") or 0.0) for row in values]
    quality_statuses = Counter(str(row[engine].get("status") or "missing") for row in values)
    f1_values = [
        float(row[engine].get("reference_agreement", {}).get("token_f1") or 0.0)
        for row in values
    ]
    return {
        "item_count": len(values),
        "quality_status_counts": dict(sorted(quality_statuses.items())),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "advisory_reference_token_f1": {
            "mean": round(statistics.fmean(f1_values), 6) if f1_values else 0.0,
            "median": round(statistics.median(f1_values), 6) if f1_values else 0.0,
        },
    }


def main() -> int:
    args = _args()
    benchmark = load_json_safe(args.benchmark, {}) or {}
    scene_payload = load_json_safe(args.scene_results, {}) or {}
    unlimited_payload = (
        load_json_safe(args.unlimited_results, {}) or {}
        if args.unlimited_results
        else {}
    )
    contract_hash = str(benchmark.get("batch_contract_sha256") or "")
    for label, payload in (("scene", scene_payload), ("unlimited", unlimited_payload)):
        if payload and str(payload.get("batch_contract_sha256") or "") != contract_hash:
            raise ValueError(f"{label} result contract does not match the benchmark")
    scene_results = scene_payload.get("results") or {}
    unlimited_results = unlimited_payload.get("results") or {}

    rows: List[Dict[str, Any]] = []
    for item in benchmark.get("items") or []:
        content_hash = str(item["content_hash"])
        reference = str(item.get("visible_text_reference") or "")
        row: Dict[str, Any] = {
            "content_hash": content_hash,
            "benchmark_bucket": str(item.get("benchmark_bucket") or ""),
            "image_kind": str(item.get("image_kind") or ""),
            "source_type": str(item.get("source_type") or ""),
            "source_url": str(item.get("source_url") or ""),
            "ocr_route": str(item.get("ocr_route") or ""),
            "image_file": str(item.get("image_file") or ""),
            "visible_text_reference": reference,
        }
        raw_scene = scene_results.get(content_hash)
        if isinstance(raw_scene, dict):
            if raw_scene.get("status") == "raw_completed":
                quality = assess_scene_ocr_lines(
                    raw_scene.get("raw_lines") or [], reference_text=reference
                )
            else:
                quality = {
                    "status": "failed",
                    "text": "",
                    "candidate_text": "",
                    "quality_score": 0.0,
                    "quality_flags": ["provider_failure"],
                    "quality_metrics": {},
                }
            candidate = str(quality.get("candidate_text") or "")
            row["scene"] = {
                **quality,
                "latency_ms": raw_scene.get("latency_ms"),
                "error": str(raw_scene.get("error") or ""),
                "reference_agreement": _agreement(reference, candidate),
            }
        raw_unlimited = unlimited_results.get(content_hash)
        if isinstance(raw_unlimited, dict):
            if raw_unlimited.get("status") == "raw_completed":
                quality = assess_ocr_quality(str(raw_unlimited.get("raw_output") or ""))
                corroboration = assess_reference_coverage(
                    reference, str(quality.get("candidate_text") or "")
                )
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
                quality.setdefault("quality_metrics", {})[
                    "reference_corroboration"
                ] = corroboration
            else:
                quality = {
                    "status": "failed",
                    "text": "",
                    "candidate_text": "",
                    "quality_score": 0.0,
                    "quality_flags": ["provider_failure"],
                    "quality_metrics": {},
                }
            candidate = str(quality.get("candidate_text") or "")
            row["unlimited"] = {
                **quality,
                "latency_ms": raw_unlimited.get("latency_ms"),
                "peak_allocated_gib": raw_unlimited.get("peak_allocated_gib"),
                "error": str(raw_unlimited.get("error") or ""),
                "reference_agreement": _agreement(reference, candidate),
            }
        rows.append(row)

    by_bucket: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bucket[row["benchmark_bucket"]].append(row)
    report = {
        "version": 1,
        "kind": "hybrid_ocr_benchmark_report",
        "batch_contract_sha256": contract_hash,
        "reference_warning": (
            "Reference agreement uses Gemini visible_text as corroborating evidence only; "
            "route approval also requires status gates and visual spot-checking."
        ),
        "summary": {
            "scene": _summary(rows, "scene"),
            "unlimited": _summary(rows, "unlimited"),
            "by_bucket": {
                bucket: {
                    "scene": _summary(values, "scene"),
                    "unlimited": _summary(values, "unlimited"),
                }
                for bucket, values in sorted(by_bucket.items())
            },
        },
        "items": rows,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
