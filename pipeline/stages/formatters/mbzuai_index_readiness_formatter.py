"""MBZUAI index-readiness formatter.

This stage enriches crawler outputs before vector formatting. It is intentionally
additive: existing downstream keys such as page_metadata_file and
page_link_graph_file are preserved, but now point to canonicalized variants.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.mbzuai_indexing import (
    build_url_identity_map,
    canonicalize_link_graph,
    canonicalize_page_metadata,
    compare_page_hashes,
)
from pipeline.core.registry import register_stage
from pipeline.core.sitemap_cohorts import (
    parse_verified_empty_reason,
    validate_evidence,
)
from pipeline.core.state import now_iso

logger = logging.getLogger(__name__)


def _resolve_optional_path(path_value: Any, *, base_dir: Path) -> Path | None:
    text = str(path_value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


DEFAULT_CRITICAL_URL_PATTERNS = [
    r"/about/office-of-the-president/?$",
    r"/about/leadership",
    r"/about/contact/?$",
    r"/study/",
    r"/student-resources",
]


def _load_crawler_runtime_state(ctx: StageContext) -> Dict[str, Any]:
    candidates = [
        ctx.previous_outputs.get("runtime_state_file"),
        ctx.previous_outputs.get("crawler_runtime_state_file"),
        ctx.work_dir / "crawler_checkpoint.json",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.is_file():
            payload = load_json_safe(path, {}) or {}
            if isinstance(payload, dict):
                return payload
    return {}


def _failure_manifest(
    runtime_state: Dict[str, Any],
    formatter_config: Dict[str, Any],
    crawler_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    url_mapping = runtime_state.get("url_mapping") if isinstance(runtime_state.get("url_mapping"), dict) else {}
    crawl_state = runtime_state.get("crawl_state") if isinstance(runtime_state.get("crawl_state"), dict) else {}
    stats = runtime_state.get("stats") if isinstance(runtime_state.get("stats"), dict) else {}
    failures: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    route_counts: Dict[str, int] = {}
    intentional_excluded_count = 0
    hard_failure_count = 0
    verified_empty_urls, cohort_evidence_errors = validate_evidence(
        runtime_state.get("sitemap_cohort_verification"),
        expected_policies=(crawler_config or {}).get("known_empty_sitemap_cohorts"),
        sitemap_snapshot=runtime_state.get("discovered_sitemaps"),
    )
    matched_verified_empty_urls: set[str] = set()

    for url, value in sorted(url_mapping.items()):
        reason = str(value or "")
        if not reason.startswith("SKIPPED"):
            continue
        route = "/" + "/".join([part for part in str(url).split("//", 1)[-1].split("/", 1)[-1].split("/")[:2] if part])
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        route_counts[route] = route_counts.get(route, 0) + 1
        verified_empty_policy = parse_verified_empty_reason(reason)
        if verified_empty_policy:
            if verified_empty_urls.get(str(url)) == verified_empty_policy:
                intentional_excluded_count += 1
                matched_verified_empty_urls.add(str(url))
            else:
                hard_failure_count += 1
        elif reason.startswith("SKIPPED_EXCLUDED"):
            intentional_excluded_count += 1
        else:
            hard_failure_count += 1
        failures.append({"url": url, "reason": reason, "route_group": route})

    missing_verified_mappings = sorted(
        set(verified_empty_urls) - matched_verified_empty_urls
    )
    if missing_verified_mappings:
        cohort_evidence_errors.append(
            "verified-empty evidence is missing matching URL mappings: "
            + ", ".join(missing_verified_mappings[:10])
        )

    visited = crawl_state.get("visited") if isinstance(crawl_state.get("visited"), list) else []
    pending = crawl_state.get("pending") if isinstance(crawl_state.get("pending"), list) else []
    expected_inventory = int(formatter_config.get("expected_site_inventory_count") or 0)
    scraped_or_downloaded = int(stats.get("pages_scraped") or 0) + int(stats.get("documents_downloaded") or 0)
    effective_expected_inventory = max(0, expected_inventory - intentional_excluded_count)
    coverage_denominator = effective_expected_inventory or expected_inventory
    inventory_coverage_ratio = round(scraped_or_downloaded / coverage_denominator, 4) if coverage_denominator > 0 else None
    raw_inventory_coverage_ratio = round(scraped_or_downloaded / expected_inventory, 4) if expected_inventory > 0 else None
    return {
        "schema_version": 1,
        "stats": stats,
        "failure_count": len(failures),
        "hard_failure_count": hard_failure_count,
        "intentional_excluded_count": intentional_excluded_count,
        "verified_empty_count": len(verified_empty_urls),
        "cohort_evidence_errors": cohort_evidence_errors,
        "reason_counts": reason_counts,
        "route_counts": route_counts,
        "failed_urls": failures,
        "crawl_state": {
            "visited_count": len(visited),
            "pending_count": len(pending),
            "pages_crawled": crawl_state.get("pages_crawled"),
        },
        "expected_site_inventory_count": expected_inventory,
        "effective_expected_inventory_count": effective_expected_inventory,
        "scraped_or_downloaded_count": scraped_or_downloaded,
        "raw_inventory_coverage_ratio": raw_inventory_coverage_ratio,
        "inventory_coverage_ratio": inventory_coverage_ratio,
    }


def _coverage_gate(
    *,
    canonical_metadata: Dict[str, Dict[str, Any]],
    failure_manifest: Dict[str, Any],
    formatter_config: Dict[str, Any],
) -> Dict[str, Any]:
    patterns = formatter_config.get("critical_url_patterns") or DEFAULT_CRITICAL_URL_PATTERNS
    compiled_patterns = [
        (str(pattern), re.compile(str(pattern), re.IGNORECASE))
        for pattern in patterns
        if str(pattern or "").strip()
    ]
    indexed_urls = [url for url, meta in canonical_metadata.items() if bool(meta.get("indexable", True))]
    missing_critical = []
    for pattern, regex in compiled_patterns:
        if not any(regex.search(url) for url in indexed_urls):
            failed_matches = [
                item for item in failure_manifest.get("failed_urls", [])
                if regex.search(str(item.get("url") or ""))
            ]
            missing_critical.append(
                {
                    "pattern": str(pattern),
                    "failed_matches": failed_matches[:10],
                    "failed_match_count": len(failed_matches),
                }
            )

    expected_inventory = int(formatter_config.get("expected_site_inventory_count") or 0)
    minimum_ratio = float(formatter_config.get("minimum_inventory_coverage_ratio") or 0.0)
    coverage_ratio = failure_manifest.get("inventory_coverage_ratio")
    hard_failure_count = int(
        failure_manifest.get("hard_failure_count", failure_manifest.get("failure_count", 0)) or 0
    )
    maximum_hard_failures_raw = formatter_config.get("maximum_hard_failure_count")
    maximum_hard_failures = (
        int(maximum_hard_failures_raw)
        if maximum_hard_failures_raw is not None
        else None
    )
    hard_failure_gap = (
        maximum_hard_failures is not None
        and maximum_hard_failures >= 0
        and hard_failure_count > maximum_hard_failures
    )
    inventory_gap = (
        expected_inventory > 0
        and minimum_ratio > 0
        and coverage_ratio is not None
        and float(coverage_ratio) < minimum_ratio
    )
    cohort_evidence_errors = list(failure_manifest.get("cohort_evidence_errors") or [])
    return {
        "schema_version": 1,
        "critical_url_patterns": [str(pattern) for pattern in patterns],
        "missing_critical_patterns": missing_critical,
        "missing_critical_count": len(missing_critical),
        "expected_site_inventory_count": expected_inventory,
        "effective_expected_inventory_count": failure_manifest.get("effective_expected_inventory_count", expected_inventory),
        "minimum_inventory_coverage_ratio": minimum_ratio,
        "hard_failure_count": hard_failure_count,
        "maximum_hard_failure_count": maximum_hard_failures,
        "hard_failure_gap": hard_failure_gap,
        "intentional_excluded_count": failure_manifest.get("intentional_excluded_count", 0),
        "verified_empty_count": failure_manifest.get("verified_empty_count", 0),
        "cohort_evidence_errors": cohort_evidence_errors,
        "cohort_evidence_error_count": len(cohort_evidence_errors),
        "raw_inventory_coverage_ratio": failure_manifest.get("raw_inventory_coverage_ratio"),
        "inventory_coverage_ratio": coverage_ratio,
        "inventory_gap": inventory_gap,
        "ok": (
            not missing_critical
            and not inventory_gap
            and not hard_failure_gap
            and not cohort_evidence_errors
        ),
    }


@register_stage
class MBZUAIIndexReadinessFormatter(FormatterStage):
    name = "mbzuai_index_readiness"
    description = "Canonicalizes MBZUAI page metadata and writes index manifests before vector formatting."

    async def execute(self, ctx: StageContext) -> StageResult:
        page_metadata_file = ctx.previous_outputs.get("page_metadata_file")
        page_link_graph_file = ctx.previous_outputs.get("page_link_graph_file")
        if not page_metadata_file:
            return StageResult.failure("page_metadata_file is required before MBZUAI index readiness")

        raw_page_metadata = load_json_safe(page_metadata_file, {}) or {}
        if not isinstance(raw_page_metadata, dict) or not raw_page_metadata:
            return StageResult.failure("page_metadata_file is empty or invalid")

        canonical_metadata = canonicalize_page_metadata(raw_page_metadata)
        if not canonical_metadata:
            return StageResult.failure("Canonical page metadata is empty")

        raw_link_graph = load_json_safe(page_link_graph_file, {}) if page_link_graph_file else {}
        raw_link_graph = raw_link_graph if isinstance(raw_link_graph, dict) else {}
        canonical_graph = canonicalize_link_graph(raw_link_graph, canonical_metadata)
        url_identity_map = build_url_identity_map(canonical_metadata)

        previous_metadata: Dict[str, Any] = {}
        previous_metadata_path = _resolve_optional_path(
            ctx.formatter_config.get("previous_canonical_page_metadata_file"),
            base_dir=ctx.work_dir,
        )
        if previous_metadata_path and previous_metadata_path.is_file():
            payload = load_json_safe(previous_metadata_path, {}) or {}
            if isinstance(payload, dict):
                previous_metadata = payload
        change_manifest = compare_page_hashes(canonical_metadata, previous_metadata)

        canonical_metadata_file = ctx.stage_work_dir / "canonical_page_metadata.json"
        url_identity_file = ctx.stage_work_dir / "canonical_url_identity_map.json"
        canonical_graph_file = ctx.stage_work_dir / "canonical_page_link_graph.json"
        change_manifest_file = ctx.stage_work_dir / "page_change_manifest.json"
        failure_manifest_file = ctx.stage_work_dir / "crawl_failure_manifest.json"
        coverage_gate_file = ctx.stage_work_dir / "index_coverage_gate.json"
        manifest_file = ctx.stage_work_dir / "index_readiness_manifest.json"

        runtime_state = _load_crawler_runtime_state(ctx)
        crawl_failure_manifest = _failure_manifest(
            runtime_state,
            ctx.formatter_config,
            ctx.crawler_config,
        )
        coverage_gate = _coverage_gate(
            canonical_metadata=canonical_metadata,
            failure_manifest=crawl_failure_manifest,
            formatter_config=ctx.formatter_config,
        )

        manifest = {
            "schema_version": 1,
            "created_at": now_iso(),
            "project_name": ctx.project_name,
            "run_id": ctx.run_id,
            "stage_id": ctx.stage_id,
            "page_metadata_file": str(canonical_metadata_file),
            "page_link_graph_file": str(canonical_graph_file),
            "url_identity_map_file": str(url_identity_file),
            "page_change_manifest_file": str(change_manifest_file),
            "crawl_failure_manifest_file": str(failure_manifest_file),
            "index_coverage_gate_file": str(coverage_gate_file),
            "source_page_metadata_file": str(page_metadata_file),
            "source_page_link_graph_file": str(page_link_graph_file or ""),
            "page_count": len(canonical_metadata),
            "indexable_page_count": sum(1 for item in canonical_metadata.values() if bool(item.get("indexable", True))),
            "excluded_page_count": sum(1 for item in canonical_metadata.values() if not bool(item.get("indexable", True))),
            "canonical_family_count": url_identity_map["canonical_family_count"],
            "duplicate_family_count": url_identity_map["duplicate_family_count"],
            "link_graph_nodes": canonical_graph["stats"]["node_count"],
            "link_graph_edges": canonical_graph["stats"]["edge_count"],
            "crawl_failures": {
                "failure_count": crawl_failure_manifest["failure_count"],
                "hard_failure_count": crawl_failure_manifest["hard_failure_count"],
                "intentional_excluded_count": crawl_failure_manifest["intentional_excluded_count"],
                "verified_empty_count": crawl_failure_manifest["verified_empty_count"],
                "cohort_evidence_errors": crawl_failure_manifest["cohort_evidence_errors"],
                "reason_counts": crawl_failure_manifest["reason_counts"],
                "expected_site_inventory_count": crawl_failure_manifest["expected_site_inventory_count"],
                "effective_expected_inventory_count": crawl_failure_manifest["effective_expected_inventory_count"],
                "raw_inventory_coverage_ratio": crawl_failure_manifest["raw_inventory_coverage_ratio"],
                "inventory_coverage_ratio": crawl_failure_manifest["inventory_coverage_ratio"],
            },
            "coverage_gate": {
                "ok": coverage_gate["ok"],
                "missing_critical_count": coverage_gate["missing_critical_count"],
                "inventory_gap": coverage_gate["inventory_gap"],
                "cohort_evidence_error_count": coverage_gate["cohort_evidence_error_count"],
            },
            "change_detection": {
                "previous_metadata_file": str(previous_metadata_path or ""),
                "new_page_count": change_manifest["new_page_count"],
                "changed_page_count": change_manifest["changed_page_count"],
                "unchanged_page_count": change_manifest["unchanged_page_count"],
                "removed_page_count": change_manifest["removed_page_count"],
            },
        }

        atomic_write_json(canonical_metadata_file, canonical_metadata)
        atomic_write_json(url_identity_file, url_identity_map)
        atomic_write_json(canonical_graph_file, canonical_graph)
        atomic_write_json(change_manifest_file, change_manifest)
        atomic_write_json(failure_manifest_file, crawl_failure_manifest)
        atomic_write_json(coverage_gate_file, coverage_gate)
        atomic_write_json(manifest_file, manifest)

        fail_on_critical = bool(ctx.formatter_config.get("fail_on_critical_coverage", False))
        fail_on_inventory_gap = bool(ctx.formatter_config.get("fail_on_inventory_gap", False))
        fail_on_hard_failures = bool(ctx.formatter_config.get("fail_on_hard_failure_gap", False))
        if (
            (fail_on_critical and coverage_gate["missing_critical_count"])
            or (fail_on_inventory_gap and coverage_gate["inventory_gap"])
            or (fail_on_hard_failures and coverage_gate["hard_failure_gap"])
            or coverage_gate["cohort_evidence_error_count"]
        ):
            return StageResult.failure(
                "MBZUAI index coverage gate failed: "
                f"missing_critical={coverage_gate['missing_critical_count']} "
                f"inventory_gap={coverage_gate['inventory_gap']} "
                f"hard_failure_gap={coverage_gate['hard_failure_gap']} "
                f"cohort_evidence_errors={coverage_gate['cohort_evidence_error_count']}"
            )

        logger.info(
            "MBZUAI index readiness: pages=%d families=%d duplicate_families=%d graph_edges=%d",
            len(canonical_metadata),
            url_identity_map["canonical_family_count"],
            url_identity_map["duplicate_family_count"],
            canonical_graph["stats"]["edge_count"],
        )

        return StageResult.success(
            outputs={
                "page_metadata_file": str(canonical_metadata_file),
                "canonical_page_metadata_file": str(canonical_metadata_file),
                "url_identity_map_file": str(url_identity_file),
                "page_link_graph_file": str(canonical_graph_file),
                "canonical_page_link_graph_file": str(canonical_graph_file),
                "page_change_manifest_file": str(change_manifest_file),
                "crawl_failure_manifest_file": str(failure_manifest_file),
                "index_coverage_gate_file": str(coverage_gate_file),
                "index_readiness_manifest_file": str(manifest_file),
            },
            metrics={
                "canonical_pages": len(canonical_metadata),
                "indexable_pages": manifest["indexable_page_count"],
                "excluded_pages": manifest["excluded_page_count"],
                "canonical_families": url_identity_map["canonical_family_count"],
                "duplicate_families": url_identity_map["duplicate_family_count"],
                "page_link_graph_edges": canonical_graph["stats"]["edge_count"],
                "crawl_failures": crawl_failure_manifest["failure_count"],
                "coverage_missing_critical": coverage_gate["missing_critical_count"],
                "coverage_inventory_gap": int(bool(coverage_gate["inventory_gap"])),
                "new_pages": change_manifest["new_page_count"],
                "changed_pages": change_manifest["changed_page_count"],
                "unchanged_pages": change_manifest["unchanged_page_count"],
                "removed_pages": change_manifest["removed_page_count"],
            },
            artifacts=[
                ctx.make_artifact(
                    canonical_metadata_file,
                    artifact_type="canonical_page_metadata",
                    role="page_metadata",
                    metadata={"records": len(canonical_metadata)},
                ),
                ctx.make_artifact(
                    canonical_graph_file,
                    artifact_type="canonical_page_link_graph",
                    role="page_link_graph",
                    metadata=canonical_graph["stats"],
                ),
                ctx.make_artifact(
                    manifest_file,
                    artifact_type="index_readiness_manifest",
                    role="run_manifest",
                    metadata={
                        "page_count": len(canonical_metadata),
                        "duplicate_family_count": url_identity_map["duplicate_family_count"],
                        "coverage_ok": coverage_gate["ok"],
                    },
                ),
            ],
        )
