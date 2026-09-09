"""Build Representation V2 Page Cards and actions from frozen crawl HTML."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe, sha256_file
from pipeline.core.page_cards import (
    build_document_revisions,
    extract_page_draft,
    finalize_page_drafts,
    link_page_cards_to_revisions,
)
from pipeline.core.registry import register_stage
from pipeline.core.representation_v2 import (
    REPRESENTATION_V2_KIND,
    REPRESENTATION_V2_SCHEMA_VERSION,
    attach_page_links_to_documents,
    now_iso,
    representation_v2_json_schema,
    validate_representation_v2,
)


logger = logging.getLogger(__name__)


def _path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _resolve_inputs(
    ctx: StageContext,
    config: Mapping[str, Any],
) -> Tuple[Path, Path, Path | None]:
    inventory_path = _path(
        config.get("inventory_file")
        or ctx.previous_outputs.get("prepared_corpus_inventory_file")
    )
    page_metadata_path = _path(
        config.get("page_metadata_file")
        or ctx.previous_outputs.get("canonical_page_metadata_file")
        or ctx.previous_outputs.get("page_metadata_file")
    )
    source_run_dir = _path(config.get("source_corpus_run_dir"))
    if source_run_dir:
        boundary_dir = source_run_dir / "stage_outputs" / "prepare_corpus_boundary"
        inventory_path = inventory_path or boundary_dir / "prepared_corpus_inventory.json"
        page_metadata_path = (
            page_metadata_path or boundary_dir / "prepared_canonical_page_metadata.json"
        )
    if inventory_path is None:
        raise ValueError(
            "Representation V2 requires prepared_corpus_inventory_file or "
            "formatter.representation_v2.inventory_file/source_corpus_run_dir"
        )
    if page_metadata_path is None:
        raise ValueError(
            "Representation V2 requires canonical_page_metadata_file or "
            "formatter.representation_v2.page_metadata_file/source_corpus_run_dir"
        )
    for label, path in (
        ("corpus inventory", inventory_path),
        ("canonical page metadata", page_metadata_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Representation V2 {label} is missing: {path}")
    return inventory_path, page_metadata_path, source_run_dir


def _json_schema_issues(
    bundle: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    maximum_samples: int = 200,
) -> Tuple[int, List[Dict[str, Any]]]:
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(bundle),
        key=lambda error: tuple(str(value) for value in error.absolute_path),
    )
    samples = []
    for error in errors[: max(0, int(maximum_samples))]:
        path = ".".join(str(value) for value in error.absolute_path) or "$"
        samples.append(
            {
                "code": "json_schema",
                "path": path,
                "message": error.message,
            }
        )
    return len(errors), samples


def _combined_html_sha256(page_cards: List[Mapping[str, Any]]) -> str:
    digest = sha256()
    for page in sorted(page_cards, key=lambda value: str(value.get("source_url") or "")):
        source_html = page.get("source_html") if isinstance(page.get("source_html"), Mapping) else {}
        digest.update(str(page.get("source_url") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(source_html.get("sha256") or "").encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _configured_count_gate(
    coverage: Dict[str, Any],
    *,
    label: str,
    expected: int,
    actual: int,
) -> None:
    gate_name = f"expected_{label}_count"
    passed = expected <= 0 or actual == expected
    coverage["gates"][gate_name] = passed
    coverage["counts"][f"configured_{label}_count"] = expected
    if not passed:
        coverage["counts"]["issues"] = int(coverage["counts"].get("issues") or 0) + 1
        issue_codes = coverage["counts"].setdefault("issue_codes", {})
        issue_codes[gate_name] = int(issue_codes.get(gate_name) or 0) + 1
        coverage.setdefault("issue_samples", []).append(
            {
                "code": gate_name,
                "message": f"Expected exactly {expected} {label} records, found {actual}",
            }
        )
    coverage["passed"] = bool(coverage.get("passed")) and passed
    coverage["gates"]["passed"] = bool(coverage["passed"])


@register_stage
class RepresentationV2Formatter(FormatterStage):
    name = "representation_v2"
    description = (
        "Creates evidence-backed Representation V2 document revisions, Page Cards, "
        "and typed webpage actions without chunking, embedding, or indexing."
    )

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        formatter = config.get("formatter") if isinstance(config.get("formatter"), dict) else {}
        stage_config = formatter.get("representation_v2") if isinstance(formatter, dict) else {}
        if not isinstance(stage_config, dict):
            return ["formatter.representation_v2 must be a mapping"]
        errors: List[str] = []
        for key in (
            "expected_document_count",
            "expected_page_count",
            "expected_web_document_count",
            "maximum_topics_per_page",
            "template_minimum_page_count",
        ):
            try:
                if int(stage_config.get(key, 0)) < 0:
                    raise ValueError
            except (TypeError, ValueError):
                errors.append(f"formatter.representation_v2.{key} must be non-negative")
        try:
            workers = int(stage_config.get("maximum_workers", 4))
            if not 1 <= workers <= 32:
                raise ValueError
        except (TypeError, ValueError):
            errors.append("formatter.representation_v2.maximum_workers must be between 1 and 32")
        executor_type = str(stage_config.get("executor_type") or "thread").lower()
        if executor_type not in {"thread", "process"}:
            errors.append(
                "formatter.representation_v2.executor_type must be thread or process"
            )
        try:
            ratio = float(stage_config.get("template_minimum_page_ratio", 0.03))
            if not 0 <= ratio <= 1:
                raise ValueError
        except (TypeError, ValueError):
            errors.append(
                "formatter.representation_v2.template_minimum_page_ratio must be between 0 and 1"
            )
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.formatter_config.get("representation_v2") or {}
        if not isinstance(config, dict):
            return StageResult.failure("formatter.representation_v2 must be a mapping")

        started = time.monotonic()
        try:
            inventory_path, page_metadata_path, source_run_dir = _resolve_inputs(ctx, config)
            inventory = load_json_safe(inventory_path, None)
            page_metadata = load_json_safe(page_metadata_path, None)
            if not isinstance(inventory, dict) or not isinstance(inventory.get("documents"), list):
                raise ValueError("Prepared corpus inventory is not a valid document inventory")
            if not isinstance(page_metadata, dict) or not page_metadata:
                raise ValueError("Canonical page metadata is not a non-empty mapping")
            inventory_documents = [
                dict(value)
                for value in inventory.get("documents") or []
                if isinstance(value, Mapping)
            ]
            if len(inventory_documents) != len(inventory.get("documents") or []):
                raise ValueError("Prepared corpus inventory contains non-object documents")
            if int(inventory.get("document_count") or 0) != len(inventory_documents):
                raise ValueError("Prepared corpus inventory document_count does not match documents")
            invalid_page_records = [
                str(url) for url, value in page_metadata.items() if not isinstance(value, Mapping)
            ]
            if invalid_page_records:
                raise ValueError(
                    "Canonical page metadata contains non-object records: "
                    + ", ".join(invalid_page_records[:5])
                )

            documents, url_index, expected_web_revision_ids = build_document_revisions(
                inventory_documents
            )
            official_hosts = {
                (urlsplit(str(url)).hostname or "").lower()
                for url in page_metadata
                if str(url).strip()
            }
            official_hosts.update(
                str(value.get("host") or "").strip().lower()
                for value in page_metadata.values()
                if isinstance(value, Mapping) and value.get("host")
            )
            official_hosts.discard("")

            maximum_workers = int(config.get("maximum_workers", 4))
            executor_type = str(config.get("executor_type") or "thread").lower()
            executor_class = (
                ProcessPoolExecutor
                if executor_type == "process"
                else ThreadPoolExecutor
            )
            maximum_topics = max(1, int(config.get("maximum_topics_per_page", 24)))
            drafts: List[Dict[str, Any]] = []
            parse_failures: List[Dict[str, str]] = []
            ordered_pages = sorted(
                (str(url), dict(value)) for url, value in page_metadata.items()
            )
            with executor_class(max_workers=maximum_workers) as executor:
                future_urls = {
                    executor.submit(
                        extract_page_draft,
                        url,
                        metadata,
                        official_hosts=official_hosts,
                        maximum_topics=maximum_topics,
                    ): url
                    for url, metadata in ordered_pages
                }
                for completed, future in enumerate(as_completed(future_urls), start=1):
                    url = future_urls[future]
                    try:
                        drafts.append(future.result())
                    except Exception as exc:  # fail closed after collecting diagnostics
                        parse_failures.append(
                            {"source_url": url, "error": f"{type(exc).__name__}: {exc}"}
                        )
                    if completed % 100 == 0 or completed == len(future_urls):
                        logger.info(
                            "Representation V2 HTML extraction: %d/%d pages (%d failures)",
                            completed,
                            len(future_urls),
                            len(parse_failures),
                        )
            if parse_failures:
                diagnostics_path = ctx.stage_work_dir / "representation_v2_parse_failures.json"
                atomic_write_json(
                    diagnostics_path,
                    {
                        "schema_version": REPRESENTATION_V2_SCHEMA_VERSION,
                        "kind": "representation_v2_parse_failures",
                        "created_at": now_iso(),
                        "failure_count": len(parse_failures),
                        "failures": parse_failures,
                    },
                )
                return StageResult.failure(
                    f"Representation V2 could not parse {len(parse_failures)} raw HTML artifacts",
                    outputs={"representation_parse_failures_file": str(diagnostics_path)},
                    metrics={"html_parse_failures": len(parse_failures)},
                    artifacts=[
                        ctx.make_artifact(
                            diagnostics_path,
                            artifact_type="representation_v2_parse_failures",
                            role="quality_diagnostics",
                            metadata={"failure_count": len(parse_failures)},
                        )
                    ],
                )

            page_cards, actions, action_stats = finalize_page_drafts(
                drafts,
                template_minimum_page_count=max(
                    1, int(config.get("template_minimum_page_count", 20))
                ),
                template_minimum_page_ratio=float(
                    config.get("template_minimum_page_ratio", 0.03)
                ),
            )
            linkage_stats = link_page_cards_to_revisions(page_cards, url_index)
            attach_page_links_to_documents(documents, page_cards)
            raw_html_paths = {
                str((page.get("source_html") or {}).get("path") or "")
                for page in page_cards
            }
            raw_html_paths.discard("")
            html_combined_sha256 = _combined_html_sha256(page_cards)
            stats = {
                "documents": len(documents),
                "web_document_revisions": len(expected_web_revision_ids),
                "page_cards": len(page_cards),
                "content_backed_page_cards": sum(
                    bool(page.get("content_backed")) for page in page_cards
                ),
                "raw_html_files": len(raw_html_paths),
                "raw_html_bytes": sum(
                    int((page.get("source_html") or {}).get("byte_count") or 0)
                    for page in page_cards
                ),
                "sections": sum(len(page.get("sections") or []) for page in page_cards),
                "topics": sum(len(page.get("topics") or []) for page in page_cards),
                "audience_labels": sum(
                    len(page.get("audiences") or []) for page in page_cards
                ),
                "actions": len(actions),
                "action_extraction": action_stats,
                "document_linkage": linkage_stats,
                "chunking_performed": False,
                "embedding_performed": False,
                "indexing_performed": False,
            }
            bundle: Dict[str, Any] = {
                "schema_version": REPRESENTATION_V2_SCHEMA_VERSION,
                "kind": REPRESENTATION_V2_KIND,
                "generated_at": now_iso(),
                "source_snapshot": {
                    "source_corpus_run_dir": str(source_run_dir or ""),
                    "inventory_file": str(inventory_path),
                    "inventory_sha256": sha256_file(inventory_path),
                    "page_metadata_file": str(page_metadata_path),
                    "page_metadata_sha256": sha256_file(page_metadata_path),
                    "raw_html_file_count": len(raw_html_paths),
                    "raw_html_combined_sha256": html_combined_sha256,
                    "extractor": "deterministic_html_page_cards_v2",
                    "extractor_config": {
                        "maximum_topics_per_page": maximum_topics,
                        "executor_type": executor_type,
                        "maximum_workers": maximum_workers,
                        "template_minimum_page_count": max(
                            1, int(config.get("template_minimum_page_count", 20))
                        ),
                        "template_minimum_page_ratio": float(
                            config.get("template_minimum_page_ratio", 0.03)
                        ),
                    },
                },
                "documents": documents,
                "page_cards": page_cards,
                "actions": actions,
                "stats": stats,
            }
            expected_corpus_ids = {
                str(value.get("record_id") or "") for value in inventory_documents
            }
            coverage = validate_representation_v2(
                bundle,
                expected_corpus_record_ids=expected_corpus_ids,
                expected_page_urls=set(str(value) for value in page_metadata),
                expected_web_revision_ids=expected_web_revision_ids,
            )
            schema = representation_v2_json_schema()
            schema_issue_count, schema_issue_samples = _json_schema_issues(bundle, schema)
            coverage["gates"]["json_schema_valid"] = schema_issue_count == 0
            coverage["counts"]["json_schema_issues"] = schema_issue_count
            if schema_issue_count:
                coverage["counts"]["issues"] = int(coverage["counts"].get("issues") or 0) + schema_issue_count
                issue_codes = coverage["counts"].setdefault("issue_codes", {})
                issue_codes["json_schema"] = schema_issue_count
                coverage["issue_samples"] = (
                    list(coverage.get("issue_samples") or []) + schema_issue_samples
                )[:200]
                coverage["passed"] = False
                coverage["gates"]["passed"] = False

            unique_html_passed = len(raw_html_paths) == len(page_metadata)
            coverage["gates"]["one_unique_html_artifact_per_page"] = unique_html_passed
            if not unique_html_passed:
                coverage["passed"] = False
                coverage["gates"]["passed"] = False
                coverage["counts"]["issues"] = int(coverage["counts"].get("issues") or 0) + 1
                issue_codes = coverage["counts"].setdefault("issue_codes", {})
                issue_codes["html_artifact_coverage"] = 1
                coverage.setdefault("issue_samples", []).append(
                    {
                        "code": "html_artifact_coverage",
                        "message": (
                            f"Expected one unique HTML file per page: "
                            f"pages={len(page_metadata)}, files={len(raw_html_paths)}"
                        ),
                    }
                )

            for label, configured, actual in (
                (
                    "document",
                    int(config.get("expected_document_count", 0)),
                    len(documents),
                ),
                (
                    "page",
                    int(config.get("expected_page_count", 0)),
                    len(page_cards),
                ),
                (
                    "web_document",
                    int(config.get("expected_web_document_count", 0)),
                    len(expected_web_revision_ids),
                ),
            ):
                _configured_count_gate(
                    coverage,
                    label=label,
                    expected=configured,
                    actual=actual,
                )
            coverage["gates"]["passed"] = bool(coverage.get("passed")) and all(
                bool(value)
                for key, value in coverage["gates"].items()
                if key != "passed"
            )
            coverage["passed"] = bool(coverage["gates"]["passed"])
            coverage["source_snapshot"] = dict(bundle["source_snapshot"])
            coverage["duration_seconds"] = round(time.monotonic() - started, 3)

            schema_path = ctx.stage_work_dir / "representation_v2.schema.json"
            bundle_path = ctx.stage_work_dir / "representation_v2_bundle.json"
            coverage_path = ctx.stage_work_dir / "representation_v2_coverage_report.json"
            atomic_write_json(schema_path, schema)
            atomic_write_json(bundle_path, bundle)
            atomic_write_json(coverage_path, coverage)
            outputs = {
                "representation_v2_schema_file": str(schema_path),
                "representation_v2_bundle_file": str(bundle_path),
                "representation_v2_coverage_report_file": str(coverage_path),
                "representation_v2_complete": bool(coverage["passed"]),
                "representation_status": "v2_page_cards_and_actions",
                "chunking_performed": False,
                "embedding_performed": False,
                "indexing_performed": False,
            }
            artifacts = [
                ctx.make_artifact(
                    schema_path,
                    artifact_type="representation_v2_schema",
                    role="representation_contract",
                    metadata={"schema_version": REPRESENTATION_V2_SCHEMA_VERSION},
                ),
                ctx.make_artifact(
                    bundle_path,
                    artifact_type="representation_v2_bundle",
                    role="pre_chunking_representation",
                    metadata={
                        "documents": len(documents),
                        "page_cards": len(page_cards),
                        "actions": len(actions),
                    },
                ),
                ctx.make_artifact(
                    coverage_path,
                    artifact_type="representation_v2_coverage_report",
                    role="quality_report",
                    metadata={"passed": bool(coverage["passed"])},
                ),
            ]
            metrics = {
                "documents": len(documents),
                "page_cards": len(page_cards),
                "actions": len(actions),
                "content_backed_page_cards": stats["content_backed_page_cards"],
                "coverage_issues": int(coverage["counts"].get("issues") or 0),
                "duration_seconds": coverage["duration_seconds"],
            }
            if not coverage["passed"]:
                first_issue = next(iter(coverage.get("issue_samples") or []), {})
                return StageResult.failure(
                    "Representation V2 failed coverage/schema gates"
                    + (f": {first_issue.get('message')}" if first_issue else ""),
                    outputs=outputs,
                    metrics=metrics,
                    artifacts=artifacts,
                )
            return StageResult.success(
                outputs=outputs,
                metrics=metrics,
                artifacts=artifacts,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return StageResult.failure(f"Representation V2 extraction failed: {exc}")
