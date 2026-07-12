"""Ablation comparison helpers for retrieval evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


DEFAULT_PRIMARY_METRICS = (
    "chunk_hit_at_5",
    "chunk_recall_at_10",
    "chunk_mrr_at_10",
    "chunk_ndcg_at_10",
    "parent_hit_at_5",
    "no_answer_violation_rate",
)
LOWER_IS_BETTER_METRICS = {"no_answer_violation_rate"}


def load_eval_report(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in eval report: {path}")
    return payload


def _metric_value(report: Mapping[str, Any], metric: str) -> float | None:
    overall = report.get("overall") if isinstance(report.get("overall"), Mapping) else {}
    value = overall.get(metric)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_retrieval_reports(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    candidate_label: str = "candidate",
    baseline_label: str = "baseline",
    metrics: Iterable[str] | None = DEFAULT_PRIMARY_METRICS,
    regression_tolerance: float = 0.0001,
) -> Dict[str, Any]:
    metric_rows = []
    regressions = []
    improvements = []
    missing = []

    selected_metrics = tuple(metrics or DEFAULT_PRIMARY_METRICS)
    for metric in selected_metrics:
        baseline_value = _metric_value(baseline, metric)
        candidate_value = _metric_value(candidate, metric)
        if baseline_value is None or candidate_value is None:
            missing.append(metric)
            continue
        raw_delta = candidate_value - baseline_value
        direction = "lower_is_better" if metric in LOWER_IS_BETTER_METRICS else "higher_is_better"
        effective_delta = -raw_delta if direction == "lower_is_better" else raw_delta
        row = {
            "metric": metric,
            "baseline": baseline_value,
            "candidate": candidate_value,
            "delta": raw_delta,
            "effective_delta": effective_delta,
            "direction": direction,
            "status": "unchanged",
        }
        if effective_delta < -abs(regression_tolerance):
            row["status"] = "regression"
            regressions.append(row)
        elif effective_delta > abs(regression_tolerance):
            row["status"] = "improvement"
            improvements.append(row)
        metric_rows.append(row)

    return {
        "schema_version": 1,
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "baseline_query_count": baseline.get("query_count"),
        "candidate_query_count": candidate.get("query_count"),
        "metrics": metric_rows,
        "missing_metrics": missing,
        "regressions": regressions,
        "improvements": improvements,
        "regression_count": len(regressions),
        "improvement_count": len(improvements),
        "passed": not regressions and not missing,
        "recommendation": "promote" if not regressions and not missing else "hold",
    }


def compare_retrieval_report_files(
    *,
    baseline_path: str | Path,
    candidate_path: str | Path,
    candidate_label: str = "candidate",
    baseline_label: str = "baseline",
    metrics: Iterable[str] | None = DEFAULT_PRIMARY_METRICS,
    regression_tolerance: float = 0.0001,
) -> Dict[str, Any]:
    return compare_retrieval_reports(
        baseline=load_eval_report(baseline_path),
        candidate=load_eval_report(candidate_path),
        candidate_label=candidate_label,
        baseline_label=baseline_label,
        metrics=metrics,
        regression_tolerance=regression_tolerance,
    )
