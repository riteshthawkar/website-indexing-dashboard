"""
Trafilatura-based content cleaner stage.

Uses Trafilatura for high-quality boilerplate removal and main content
extraction. Falls back to BS4 cleaning when Trafilatura produces no output.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from bs4 import BeautifulSoup

from pipeline.core.base import CleanerStage, StageContext, StageResult
from pipeline.core.io import (
    atomic_write_json,
    atomic_write_text,
    load_json_safe,
    reset_stage_output_directory,
)
from pipeline.core.registry import register_stage
from pipeline.stages.cleaners.bs4_cleaner import clean_html_content
from pipeline.stages.cleaners.common import (
    CleaningPolicy,
    content_meets_policy,
    evaluate_cleaning_gate,
    urls_by_source_path,
    validate_cleaner_policy_config,
    visible_content_metrics,
)
from pipeline.stages.cleaners.route_scoping import scope_route_specific_html

logger = logging.getLogger(__name__)


def _looks_navigation_heavy(cleaned_html: str) -> bool:
    if not cleaned_html or not cleaned_html.strip():
        return False

    soup = BeautifulSoup(cleaned_html, "html.parser")
    body = soup.body or soup
    total_text = body.get_text(" ", strip=True)
    if not total_text:
        return True

    paragraphs = [
        node.get_text(" ", strip=True)
        for node in body.find_all("p")
        if node.get_text(" ", strip=True)
    ]
    links = [
        node.get_text(" ", strip=True)
        for node in body.find_all("a")
        if node.get_text(" ", strip=True)
    ]
    list_items = [
        node.get_text(" ", strip=True)
        for node in body.find_all("li")
        if node.get_text(" ", strip=True)
    ]

    link_text_len = sum(len(text) for text in links)
    total_text_len = len(total_text)

    if len(paragraphs) <= 1 and len(list_items) >= 10:
        return True
    if total_text_len and link_text_len / total_text_len > 0.6 and len(paragraphs) < 3 and len(links) >= 12:
        return True
    return False


@register_stage
class TrafilaturaCleaner(CleanerStage):
    name = "trafilatura"
    description = "Trafilatura boilerplate removal with optional BS4 fallback."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        cleaner_config = config.get("cleaner", {})
        if not isinstance(cleaner_config, dict):
            return ["cleaner must be a mapping"]
        errors = validate_cleaner_policy_config(cleaner_config)
        try:
            import trafilatura  # noqa: F401
        except ImportError:
            errors.append("trafilatura is not installed. Run: pip install trafilatura")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        import trafilatura

        config = ctx.cleaner_config
        try:
            policy = CleaningPolicy.from_config(config)
        except ValueError as exc:
            return StageResult.failure(f"Invalid cleaner configuration: {exc}")

        html_dir = ctx.previous_outputs.get("html_dir")
        if not html_dir:
            return StageResult.failure("No html_dir in previous outputs")

        html_dir = Path(html_dir)
        if not html_dir.is_dir():
            return StageResult.failure(f"html_dir does not exist: {html_dir}")

        cleaned_dir = reset_stage_output_directory(
            ctx.stage_work_dir / "cleaned_html",
            ctx.stage_work_dir,
        )

        include_tables = config.get("include_tables", True)
        include_links = config.get("include_links", True)
        include_images = config.get(
            "preserve_embedded_media",
            ctx.crawler_config.get("extract_images", True),
        )
        recursive = config.get("recursive", True)

        accepted_html_artifacts = ctx.find_artifacts(artifact_type="quality_accepted_html")
        pattern = "**/*.html" if recursive else "*.html"
        files = (
            [Path(record.local_path) for record in accepted_html_artifacts if record.local_path]
            if accepted_html_artifacts
            else list(html_dir.glob(pattern))
        )
        files = sorted(set(files), key=lambda path: str(path))
        logger.info("Trafilatura cleaner found %d HTML files", len(files))

        fallback_cleaned = 0
        route_scoped = 0
        content_artifacts = []
        dispositions: List[Dict[str, Any]] = []

        url_mapping = {}
        mapping_file = ctx.previous_outputs.get("mapping_file")
        if mapping_file:
            url_mapping = load_json_safe(mapping_file, {}) or {}
        urls_by_path = urls_by_source_path(url_mapping if isinstance(url_mapping, dict) else {})
        artifact_by_path = {
            str(Path(record.local_path).resolve()): record
            for record in accepted_html_artifacts
            if record.local_path
        }

        for i, fp in enumerate(files, 1):
            resolved_source = str(fp.resolve())
            source_artifact = artifact_by_path.get(resolved_source)
            source_urls = set(urls_by_path.get(resolved_source, []))
            if source_artifact:
                artifact_urls = source_artifact.metadata.get("source_urls") or []
                if isinstance(artifact_urls, list):
                    source_urls.update(str(url) for url in artifact_urls if str(url).strip())
                source_url = str(source_artifact.metadata.get("source_url") or "")
                if source_url:
                    source_urls.add(source_url)
            relative_value = (
                source_artifact.metadata.get("relative_path")
                if source_artifact
                else None
            )
            try:
                relative = Path(str(relative_value)) if relative_value else fp.relative_to(html_dir)
            except ValueError:
                relative = Path(fp.name)
            disposition: Dict[str, Any] = {
                "source_path": resolved_source,
                "source_urls": sorted(source_urls),
                "relative_path": relative.as_posix(),
            }
            try:
                raw = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning("Read error %s: %s", fp, e)
                disposition.update(
                    {
                        "status": "failed",
                        "reason_code": "read_error",
                        "error_type": type(e).__name__,
                    }
                )
                dispositions.append(disposition)
                continue

            scope_result = scope_route_specific_html(raw, sorted(source_urls))
            extraction_input = scope_result.html
            if scope_result.applied:
                route_scoped += 1
                disposition["route_scope_method"] = scope_result.method

            extraction_error = None
            try:
                text = trafilatura.extract(
                    extraction_input,
                    include_tables=include_tables,
                    include_images=include_images,
                    include_links=include_links,
                    output_format="html",
                    favor_recall=True,
                )
            except Exception as exc:
                logger.warning("Trafilatura extraction error %s: %s", fp, exc)
                extraction_error = exc
                text = ""

            navigation_heavy = bool(text and _looks_navigation_heavy(text))
            if navigation_heavy:
                text = ""

            selected_backend = "trafilatura"
            selected_metrics = visible_content_metrics(text or "")
            if not text or not content_meets_policy(selected_metrics, policy):
                try:
                    fallback_status, fallback_html = clean_html_content(
                        extraction_input,
                        preserve_media=config.get(
                            "preserve_embedded_media",
                            ctx.crawler_config.get("extract_images", True)
                            or ctx.crawler_config.get("extract_videos", True),
                        ),
                    )
                except Exception as exc:
                    logger.warning("BS4 fallback error %s: %s", fp, exc)
                    disposition.update(
                        {
                            "status": "failed",
                            "reason_code": "fallback_cleaning_error",
                            "error_type": type(exc).__name__,
                            "content_metrics": selected_metrics.to_dict(),
                        }
                    )
                    dispositions.append(disposition)
                    continue
                fallback_metrics = visible_content_metrics(fallback_html or "")
                disposition["fallback_content_metrics"] = fallback_metrics.to_dict()
                if (
                    fallback_status == "cleaned"
                    and fallback_html
                    and content_meets_policy(fallback_metrics, policy)
                ):
                    text = fallback_html
                    selected_backend = "bs4_fallback"
                    selected_metrics = fallback_metrics
                    fallback_cleaned += 1
                else:
                    if extraction_error is not None:
                        disposition.update(
                            {
                                "status": "failed",
                                "reason_code": "extraction_error",
                                "error_type": type(extraction_error).__name__,
                            }
                        )
                    else:
                        reason_code = "insufficient_visible_content"
                        if fallback_status == "removed":
                            reason_code = "structural_error_page"
                        elif fallback_status == "removed_empty":
                            reason_code = "empty_content"
                        elif navigation_heavy:
                            reason_code = "navigation_heavy_content"
                        disposition.update(
                            {
                                "status": "filtered",
                                "reason_code": reason_code,
                            }
                        )
                    disposition["content_metrics"] = selected_metrics.to_dict()
                    dispositions.append(disposition)
                    continue

            out_path = cleaned_dir / relative
            try:
                atomic_write_text(out_path, text)
            except Exception as exc:
                logger.error("Write error %s: %s", out_path, exc)
                disposition.update(
                    {
                        "status": "failed",
                        "reason_code": "write_error",
                        "error_type": type(exc).__name__,
                        "content_metrics": selected_metrics.to_dict(),
                    }
                )
                dispositions.append(disposition)
                continue

            reason_code = (
                "accepted_trafilatura"
                if selected_backend == "trafilatura"
                else "accepted_bs4_fallback"
            )
            disposition.update(
                {
                    "status": "accepted",
                    "reason_code": reason_code,
                    "selected_backend": selected_backend,
                    "content_metrics": selected_metrics.to_dict(),
                    "output_path": str(out_path.resolve()),
                }
            )
            if extraction_error is not None:
                disposition["warnings"] = ["trafilatura_extraction_error_recovered"]
            dispositions.append(disposition)
            source_url = sorted(source_urls)[0] if source_urls else ""
            content_artifacts.append(
                ctx.make_artifact(
                    out_path,
                    artifact_type="cleaned_html",
                    role="content",
                    metadata={
                        "source_path": resolved_source,
                        "source_url": source_url,
                        "source_urls": sorted(source_urls),
                        "relative_path": relative.as_posix(),
                        "selected_backend": selected_backend,
                        "document_title": scope_result.document_title,
                        "route_scope_method": scope_result.method,
                        "content_metrics": selected_metrics.to_dict(),
                    },
                    source_artifact_ids=(
                        [source_artifact.artifact_id] if source_artifact else None
                    ),
                )
            )

            if i % 100 == 0:
                logger.info("Progress: %d/%d", i, len(files))

        critical_patterns = ctx.formatter_config.get("critical_url_patterns") or []
        gate = evaluate_cleaning_gate(
            dispositions,
            policy=policy,
            critical_url_patterns=critical_patterns,
        )
        manifest_path = ctx.stage_work_dir / "cleaning_manifest.json"
        manifest = {
            "schema_version": 1,
            "stage_id": ctx.stage_id or "clean_html",
            "engine": self.name,
            "application_status": "completed" if gate["ok"] else "failed",
            "policy": policy.to_dict(),
            "gate": gate,
            "dispositions": dispositions,
        }
        atomic_write_json(manifest_path, manifest)
        manifest_artifact = ctx.make_artifact(
            manifest_path,
            artifact_type="cleaning_manifest",
            role="cleaning_dispositions",
            metadata={
                "input_count": gate["input_count"],
                "accepted_count": gate["accepted_count"],
                "filtered_count": gate["filtered_count"],
                "failed_count": gate["failed_count"],
                "gate_ok": gate["ok"],
            },
        )

        logger.info(
            "Trafilatura cleaner done: accepted=%d filtered=%d failed=%d retention=%.3f",
            gate["accepted_count"],
            gate["filtered_count"],
            gate["failed_count"],
            gate["retention_ratio"],
        )

        outputs = {
            "cleaned_dir": str(cleaned_dir),
            "cleaned_count": gate["accepted_count"],
            "removed_count": gate["filtered_count"],
            "failed_count": gate["failed_count"],
            "cleaning_manifest_file": str(manifest_path),
        }
        metrics = {
            "cleaned": gate["accepted_count"],
            "removed": gate["filtered_count"],
            "errors": gate["failed_count"],
            "fallback_cleaned": fallback_cleaned,
            "route_scoped": route_scoped,
            "retention_ratio": gate["retention_ratio"],
        }
        if not gate["ok"]:
            failure_codes = ", ".join(item["code"] for item in gate["failures"])
            return StageResult.failure(
                f"Cleaning quality gate failed: {failure_codes}",
                checkpoint={"cleaning_manifest_file": str(manifest_path)},
                outputs=outputs,
                metrics=metrics,
                artifacts=[manifest_artifact],
            )

        return StageResult.success(
            outputs=outputs,
            metrics=metrics,
            checkpoint={"cleaning_manifest_file": str(manifest_path)},
            artifacts=[*content_artifacts, manifest_artifact],
        )
