"""Integrity helpers for run-scoped verified-empty sitemap cohorts."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, Tuple
from urllib.parse import urlparse


SCHEMA_VERSION = 1
VERIFIED_EMPTY_REASON_PREFIX = "SKIPPED_VERIFIED_EMPTY_COHORT:"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def member_urls_sha256(urls: Iterable[str]) -> str:
    normalized = sorted({str(url).strip() for url in urls if str(url).strip()})
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def evidence_sha256(payload: Dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("evidence_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def finalize_evidence(payload: Dict[str, Any]) -> Dict[str, Any]:
    finalized = dict(payload)
    finalized["schema_version"] = SCHEMA_VERSION
    finalized["evidence_sha256"] = evidence_sha256(finalized)
    return finalized


def _strict_int(value: Any) -> int | None:
    """Parse JSON integer fields without accepting booleans or raising."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def validate_evidence(
    payload: Any,
    *,
    expected_policies: Any = None,
    sitemap_snapshot: Any = None,
) -> Tuple[Dict[str, str], list[str]]:
    """Return URL -> policy ID for proven empty records plus validation errors."""
    if payload is None:
        if expected_policies:
            return {}, ["configured sitemap cohort evidence is missing"]
        return {}, []
    if not isinstance(payload, dict):
        return {}, ["sitemap cohort evidence must be an object"]
    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported sitemap cohort evidence schema_version")
    expected_digest = str(payload.get("evidence_sha256") or "")
    if not expected_digest or expected_digest != evidence_sha256(payload):
        errors.append("sitemap cohort evidence digest mismatch")

    policies = payload.get("policies")
    if not isinstance(policies, list):
        return {}, [*errors, "sitemap cohort evidence policies must be a list"]

    verified: Dict[str, str] = {}
    seen_policy_ids: set[str] = set()
    observed_policies: Dict[str, Dict[str, Any]] = {}
    for policy in policies:
        if not isinstance(policy, dict):
            errors.append("sitemap cohort policy evidence must be an object")
            continue
        policy_id = str(policy.get("id") or "").strip()
        source_url = str(policy.get("source_url") or "").strip()
        if not policy_id or policy_id in seen_policy_ids:
            errors.append(f"invalid or duplicate sitemap cohort policy id: {policy_id!r}")
            continue
        seen_policy_ids.add(policy_id)
        observed_policies[policy_id] = policy
        if not source_url:
            errors.append(f"sitemap cohort policy {policy_id!r} is missing source_url")
        source_payload_sha256 = str(policy.get("source_payload_sha256") or "")
        if not re.fullmatch(r"[0-9a-f]{64}", source_payload_sha256):
            errors.append(
                f"sitemap cohort policy {policy_id!r} has an invalid source payload digest"
            )

        records = policy.get("records")
        if not isinstance(records, list):
            errors.append(f"sitemap cohort policy {policy_id!r} records must be a list")
            continue
        member_urls = [
            str(record.get("url") or "").strip()
            for record in records
            if isinstance(record, dict)
        ]
        if _strict_int(policy.get("member_count")) != len(records):
            errors.append(f"sitemap cohort policy {policy_id!r} member_count mismatch")
        if str(policy.get("member_urls_sha256") or "") != member_urls_sha256(member_urls):
            errors.append(f"sitemap cohort policy {policy_id!r} member URL digest mismatch")
        expected_count = policy.get("expected_member_count")
        if expected_count is not None and _strict_int(expected_count) != len(records):
            errors.append(f"sitemap cohort policy {policy_id!r} expected member count mismatch")

        seen_urls: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                errors.append(f"sitemap cohort policy {policy_id!r} has a non-object record")
                continue
            url = str(record.get("url") or "").strip()
            record_policy = str(record.get("policy_id") or "").strip()
            record_source = str(record.get("source_url") or "").strip()
            if not url or url in seen_urls:
                errors.append(f"sitemap cohort policy {policy_id!r} has an invalid or duplicate URL")
                continue
            seen_urls.add(url)
            if record_policy != policy_id or record_source != source_url:
                errors.append(f"sitemap cohort record identity mismatch for {url}")
                continue

            if record.get("classification") != "verified_empty":
                continue
            if (
                record.get("final_status") != 200
                or _strict_int(record.get("body_bytes")) != 0
                or record.get("reached_eof") is not True
                or not str(record.get("final_url") or "").strip()
            ):
                errors.append(f"invalid verified-empty proof for {url}")
                continue
            if url in verified:
                errors.append(f"verified-empty URL appears in multiple policies: {url}")
                continue
            verified[url] = policy_id

    expected_by_id: Dict[str, Dict[str, Any]] = {}
    if expected_policies is not None:
        if not isinstance(expected_policies, list):
            errors.append("configured sitemap cohort policies must be a list")
        else:
            for policy in expected_policies:
                if not isinstance(policy, dict):
                    errors.append("configured sitemap cohort policy must be an object")
                    continue
                policy_id = str(policy.get("id") or "").strip()
                if not policy_id or policy_id in expected_by_id:
                    errors.append(
                        f"invalid or duplicate configured sitemap cohort policy id: {policy_id!r}"
                    )
                    continue
                expected_by_id[policy_id] = policy

            if set(observed_policies) != set(expected_by_id):
                errors.append("sitemap cohort evidence policy set does not match configuration")
            for policy_id in sorted(set(observed_policies) & set(expected_by_id)):
                observed = observed_policies[policy_id]
                expected = expected_by_id[policy_id]
                for field in ("source_url", "expected_member_count", "max_members"):
                    if observed.get(field) != expected.get(field):
                        errors.append(
                            f"sitemap cohort policy {policy_id!r} {field} does not match configuration"
                        )
                for field in ("allowed_path_prefixes", "exact_urls"):
                    observed_values = sorted(str(item) for item in (observed.get(field) or []))
                    expected_values = sorted(str(item) for item in (expected.get(field) or []))
                    if observed_values != expected_values:
                        errors.append(
                            f"sitemap cohort policy {policy_id!r} {field} does not match configuration"
                        )

    if expected_policies and sitemap_snapshot is None:
        errors.append("configured sitemap cohort discovery snapshot is missing")
    elif sitemap_snapshot is not None:
        if not isinstance(sitemap_snapshot, dict):
            errors.append("sitemap discovery snapshot is missing or invalid")
        else:
            source_fetches = sitemap_snapshot.get("source_fetches")
            url_sources = sitemap_snapshot.get("url_sources")
            raw_urls = sitemap_snapshot.get("raw_urls")
            if not isinstance(source_fetches, dict):
                errors.append("sitemap discovery source fetches are missing or invalid")
                source_fetches = {}
            if not isinstance(url_sources, dict):
                errors.append("sitemap discovery URL provenance is missing or invalid")
                url_sources = {}
            if not isinstance(raw_urls, list):
                errors.append("sitemap discovery raw URL inventory is missing or invalid")
                raw_urls = []
            raw_url_set = {str(url) for url in raw_urls}
            for policy_id, policy in observed_policies.items():
                source_url = str(policy.get("source_url") or "")
                source_record = source_fetches.get(source_url)
                if not isinstance(source_record, dict) or source_record.get("status") != 200:
                    errors.append(
                        f"sitemap cohort policy {policy_id!r} has no successful source fetch"
                    )
                elif str(source_record.get("payload_sha256") or "") != str(
                    policy.get("source_payload_sha256") or ""
                ):
                    errors.append(
                        f"sitemap cohort policy {policy_id!r} source payload digest mismatch"
                    )
                for record in policy.get("records") or []:
                    if not isinstance(record, dict):
                        continue
                    url = str(record.get("url") or "")
                    sources = url_sources.get(url)
                    if url not in raw_url_set or not isinstance(sources, list) or source_url not in sources:
                        errors.append(
                            f"sitemap cohort record lacks sitemap provenance for {url}"
                        )
                configured_policy = expected_by_id.get(policy_id)
                if configured_policy is not None:
                    exact_urls = {
                        str(url).strip()
                        for url in (configured_policy.get("exact_urls") or [])
                        if str(url).strip()
                    }
                    prefixes = {
                        "/" + str(prefix).strip().strip("/")
                        for prefix in (
                            configured_policy.get("allowed_path_prefixes") or []
                        )
                        if str(prefix).strip().strip("/")
                    }

                    def _selected(url: str) -> bool:
                        if url in exact_urls:
                            return True
                        try:
                            path = "/" + urlparse(url).path.strip("/")
                        except (TypeError, ValueError):
                            return False
                        return any(
                            path == prefix or path.startswith(f"{prefix}/")
                            for prefix in prefixes
                        )

                    selected_members = {
                        url
                        for url in raw_url_set
                        if source_url in (url_sources.get(url) or []) and _selected(url)
                    }
                    record_members = {
                        str(record.get("url") or "")
                        for record in (policy.get("records") or [])
                        if isinstance(record, dict)
                    }
                    if selected_members != record_members:
                        errors.append(
                            f"sitemap cohort policy {policy_id!r} records do not match "
                            "selector-derived sitemap members"
                        )

    return ({} if errors else verified), errors


def verified_empty_reason(policy_id: str) -> str:
    return f"{VERIFIED_EMPTY_REASON_PREFIX}{str(policy_id).strip()}"


def parse_verified_empty_reason(reason: Any) -> str:
    text = str(reason or "")
    if not text.startswith(VERIFIED_EMPTY_REASON_PREFIX):
        return ""
    return text[len(VERIFIED_EMPTY_REASON_PREFIX):].strip()
