"""Shared production contracts for HTML cleaner stages."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

from bs4 import BeautifulSoup


@dataclass(frozen=True)
class VisibleContentMetrics:
    visible_characters: int
    visible_words: int

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class CleaningPolicy:
    min_content_length: int = 100
    min_content_words: int = 5
    minimum_retention_ratio: float = 0.75
    minimum_host_retention_ratio: float = 0.50
    minimum_host_input_count: int = 5
    maximum_error_count: int = 0
    maximum_error_ratio: float = 0.0
    fail_on_empty_input: bool = True
    fail_on_zero_output: bool = True
    require_critical_url_survival: bool = True

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "CleaningPolicy":
        errors = validate_cleaner_policy_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        return cls(
            min_content_length=int(config.get("min_content_length", 100)),
            min_content_words=int(config.get("min_content_words", 5)),
            minimum_retention_ratio=float(config.get("minimum_retention_ratio", 0.75)),
            minimum_host_retention_ratio=float(
                config.get("minimum_host_retention_ratio", 0.50)
            ),
            minimum_host_input_count=int(config.get("minimum_host_input_count", 5)),
            maximum_error_count=int(config.get("maximum_error_count", 0)),
            maximum_error_ratio=float(config.get("maximum_error_ratio", 0.0)),
            fail_on_empty_input=bool(config.get("fail_on_empty_input", True)),
            fail_on_zero_output=bool(config.get("fail_on_zero_output", True)),
            require_critical_url_survival=bool(
                config.get("require_critical_url_survival", True)
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _validate_non_negative_integer(
    config: Mapping[str, Any],
    key: str,
    default: int,
) -> List[str]:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return [f"cleaner.{key} must be a non-negative integer"]
    return []


def _validate_ratio(
    config: Mapping[str, Any],
    key: str,
    default: float,
) -> List[str]:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return [f"cleaner.{key} must be a number between 0 and 1"]
    if not 0.0 <= float(value) <= 1.0:
        return [f"cleaner.{key} must be between 0 and 1"]
    return []


def validate_cleaner_policy_config(config: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for key, default in (
        ("min_content_length", 100),
        ("min_content_words", 5),
        ("minimum_host_input_count", 5),
        ("maximum_error_count", 0),
    ):
        errors.extend(_validate_non_negative_integer(config, key, default))
    for key, default in (
        ("minimum_retention_ratio", 0.75),
        ("minimum_host_retention_ratio", 0.50),
        ("maximum_error_ratio", 0.0),
    ):
        errors.extend(_validate_ratio(config, key, default))
    for key, default in (
        ("fail_on_empty_input", True),
        ("fail_on_zero_output", True),
        ("require_critical_url_survival", True),
        ("include_tables", True),
        ("include_links", True),
        ("preserve_embedded_media", True),
        ("recursive", True),
    ):
        if not isinstance(config.get(key, default), bool):
            errors.append(f"cleaner.{key} must be a boolean")
    return errors


def normalized_visible_text(html: str) -> str:
    """Return normalized text that a user could reasonably see."""

    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup.find_all(
        ["script", "style", "noscript", "template", "svg", "canvas"]
    ):
        element.decompose()
    for element in list(soup.find_all(True)):
        attrs = getattr(element, "attrs", None)
        if not isinstance(attrs, dict):
            continue
        style = re.sub(r"\s+", "", str(attrs.get("style", "")).lower())
        if (
            "hidden" in attrs
            or str(attrs.get("aria-hidden", "")).lower() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
        ):
            element.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()


def visible_content_metrics(html: str) -> VisibleContentMetrics:
    text = normalized_visible_text(html)
    return VisibleContentMetrics(
        visible_characters=len(text),
        visible_words=len(text.split()),
    )


def content_meets_policy(metrics: VisibleContentMetrics, policy: CleaningPolicy) -> bool:
    return (
        metrics.visible_characters >= policy.min_content_length
        and metrics.visible_words >= policy.min_content_words
    )


def urls_by_source_path(mapping: Mapping[str, Any]) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = defaultdict(list)
    for url, raw_path in mapping.items():
        if not isinstance(url, str) or not isinstance(raw_path, str):
            continue
        if raw_path.startswith("SKIPPED"):
            continue
        try:
            resolved = str(Path(raw_path).resolve())
        except OSError:
            continue
        result[resolved].append(url)
    return {
        path: sorted(set(urls))
        for path, urls in result.items()
    }


def _host_counts(
    dispositions: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, int | float]]:
    input_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    for item in dispositions:
        hosts = {
            (urlparse(str(url)).hostname or "").lower()
            for url in item.get("source_urls", [])
            if str(url).strip()
        }
        hosts.discard("")
        for host in hosts:
            input_counts[host] += 1
            if item.get("status") == "accepted":
                accepted_counts[host] += 1
    return {
        host: {
            "input_count": input_count,
            "accepted_count": accepted_counts[host],
            "retention_ratio": round(accepted_counts[host] / input_count, 6),
        }
        for host, input_count in sorted(input_counts.items())
    }


def evaluate_cleaning_gate(
    dispositions: Sequence[Mapping[str, Any]],
    *,
    policy: CleaningPolicy,
    critical_url_patterns: Iterable[str] = (),
) -> Dict[str, Any]:
    counts = Counter(str(item.get("status") or "failed") for item in dispositions)
    reasons = Counter(str(item.get("reason_code") or "unknown") for item in dispositions)
    input_count = len(dispositions)
    accepted_count = counts["accepted"]
    filtered_count = counts["filtered"]
    failed_count = counts["failed"]
    retention_ratio = accepted_count / input_count if input_count else 0.0
    error_ratio = failed_count / input_count if input_count else 0.0
    failures: List[Dict[str, Any]] = []

    if policy.fail_on_empty_input and input_count == 0:
        failures.append({"code": "empty_input", "message": "Cleaner received no HTML files"})
    if policy.fail_on_zero_output and input_count > 0 and accepted_count == 0:
        failures.append({"code": "zero_output", "message": "Cleaner accepted no HTML files"})
    if input_count and retention_ratio < policy.minimum_retention_ratio:
        failures.append(
            {
                "code": "retention_ratio_below_minimum",
                "actual": round(retention_ratio, 6),
                "minimum": policy.minimum_retention_ratio,
            }
        )
    if failed_count > policy.maximum_error_count:
        failures.append(
            {
                "code": "error_count_exceeded",
                "actual": failed_count,
                "maximum": policy.maximum_error_count,
            }
        )
    if input_count and error_ratio > policy.maximum_error_ratio:
        failures.append(
            {
                "code": "error_ratio_exceeded",
                "actual": round(error_ratio, 6),
                "maximum": policy.maximum_error_ratio,
            }
        )

    host_counts = _host_counts(dispositions)
    unhealthy_hosts = []
    for host, host_metrics in host_counts.items():
        if int(host_metrics["input_count"]) < policy.minimum_host_input_count:
            continue
        if float(host_metrics["retention_ratio"]) < policy.minimum_host_retention_ratio:
            unhealthy_hosts.append({"host": host, **host_metrics})
    if unhealthy_hosts:
        failures.append(
            {
                "code": "host_retention_ratio_below_minimum",
                "minimum": policy.minimum_host_retention_ratio,
                "hosts": unhealthy_hosts,
            }
        )

    patterns = [str(value).strip() for value in critical_url_patterns if str(value).strip()]
    accepted_urls = {
        str(url)
        for item in dispositions
        if item.get("status") == "accepted"
        for url in item.get("source_urls", [])
        if str(url).strip()
    }
    missing_critical_patterns: List[str] = []
    invalid_critical_patterns: List[str] = []
    if policy.require_critical_url_survival:
        for pattern in patterns:
            try:
                matched = any(re.search(pattern, url, re.IGNORECASE) for url in accepted_urls)
            except re.error:
                invalid_critical_patterns.append(pattern)
                continue
            if not matched:
                missing_critical_patterns.append(pattern)
        if invalid_critical_patterns:
            failures.append(
                {
                    "code": "invalid_critical_url_pattern",
                    "patterns": invalid_critical_patterns,
                }
            )
        if missing_critical_patterns:
            failures.append(
                {
                    "code": "missing_critical_url_after_cleaning",
                    "patterns": missing_critical_patterns,
                }
            )

    return {
        "ok": not failures,
        "input_count": input_count,
        "accepted_count": accepted_count,
        "filtered_count": filtered_count,
        "failed_count": failed_count,
        "retention_ratio": round(retention_ratio, 6),
        "error_ratio": round(error_ratio, 6),
        "reason_counts": dict(sorted(reasons.items())),
        "host_counts": host_counts,
        "missing_critical_url_patterns": missing_critical_patterns,
        "failures": failures,
    }
