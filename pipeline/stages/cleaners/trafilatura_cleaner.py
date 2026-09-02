"""
Trafilatura-based content cleaner stage.

Uses Trafilatura for high-quality boilerplate removal and main content
extraction. Falls back to BS4 cleaning when Trafilatura produces no output.
"""

import logging
import re
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


def _heading_signatures(html: str) -> set[str]:
    """Return normalized, substantive heading labels from an HTML candidate."""

    if not html:
        return set()
    soup = BeautifulSoup(html, "html.parser")
    signatures: set[str] = set()
    for heading in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        value = re.sub(r"\s+", " ", heading.get_text(" ", strip=True)).strip().casefold()
        # Tiny glyph-like labels and very long accidental wrappers are not useful
        # structural evidence.
        if 2 <= len(value) <= 200:
            signatures.add(value)
    return signatures


def _structured_fallback_comparison(
    primary_html: str,
    fallback_html: str,
    *,
    max_primary_words: int = 2000,
    min_word_gain: int = 80,
    min_word_ratio: float = 1.35,
    min_missing_headings: int = 3,
) -> Dict[str, Any]:
    """Assess whether BS4 recovered material structure omitted by Trafilatura.

    Word gain alone is deliberately insufficient: the fallback must also recover
    multiple headings absent from the primary candidate. This prevents ordinary
    boilerplate growth from displacing a good article extraction.
    """

    primary_metrics = visible_content_metrics(primary_html)
    fallback_metrics = visible_content_metrics(fallback_html)
    primary_headings = _heading_signatures(primary_html)
    fallback_headings = _heading_signatures(fallback_html)
    missing_headings = sorted(fallback_headings - primary_headings)
    word_gain = fallback_metrics.visible_words - primary_metrics.visible_words
    word_ratio = fallback_metrics.visible_words / max(primary_metrics.visible_words, 1)
    prefer_fallback = (
        primary_metrics.visible_words <= max_primary_words
        and word_gain >= min_word_gain
        and word_ratio >= min_word_ratio
        and len(missing_headings) >= min_missing_headings
    )
    return {
        "prefer_fallback": prefer_fallback,
        "primary_words": primary_metrics.visible_words,
        "fallback_words": fallback_metrics.visible_words,
        "word_gain": word_gain,
        "word_ratio": round(word_ratio, 6),
        "primary_heading_count": len(primary_headings),
        "fallback_heading_count": len(fallback_headings),
        "missing_heading_count": len(missing_headings),
        "missing_headings": missing_headings[:25],
    }


def _validate_structured_fallback_config(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    bool_value = config.get("compare_bs4_when_structurally_richer", True)
    if not isinstance(bool_value, bool):
        errors.append("cleaner.compare_bs4_when_structurally_richer must be a boolean")
    for key, default in (
        ("structured_fallback_max_primary_words", 2000),
        ("structured_fallback_min_word_gain", 80),
        ("structured_fallback_min_missing_headings", 3),
    ):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"cleaner.{key} must be a non-negative integer")
    ratio = config.get("structured_fallback_min_word_ratio", 1.35)
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or float(ratio) < 1.0:
        errors.append("cleaner.structured_fallback_min_word_ratio must be a number >= 1")
    return errors


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
        errors.extend(_validate_structured_fallback_config(cleaner_config))
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
        compare_structured_fallback = config.get(
            "compare_bs4_when_structurally_richer", True
        )
        structured_fallback_options = {
            "max_primary_words": int(
                config.get("structured_fallback_max_primary_words", 2000)
            ),
            "min_word_gain": int(
                config.get("structured_fallback_min_word_gain", 80)
            ),
            "min_word_ratio": float(
                config.get("structured_fallback_min_word_ratio", 1.35)
            ),
            "min_missing_headings": int(
                config.get("structured_fallback_min_missing_headings", 3)
            ),
        }

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
        fallback_compared = 0
        structurally_richer_fallbacks = 0
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
            primary_usable = bool(text) and content_meets_policy(selected_metrics, policy)
            needs_fallback = not primary_usable
            compare_fallback = bool(compare_structured_fallback and primary_usable)
            fallback_status = None
            fallback_html = None
            fallback_metrics = None
            cleaner_warnings: List[str] = []

            if needs_fallback or compare_fallback:
                if compare_fallback:
                    fallback_compared += 1
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
                    if needs_fallback:
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
                    cleaner_warnings.append("bs4_structured_comparison_error")

            fallback_usable = False
            if fallback_metrics is None and fallback_status is not None:
                fallback_metrics = visible_content_metrics(fallback_html or "")
                disposition["fallback_content_metrics"] = fallback_metrics.to_dict()
                fallback_usable = bool(
                    fallback_status == "cleaned"
                    and fallback_html
                    and content_meets_policy(fallback_metrics, policy)
                )

            if needs_fallback:
                if fallback_usable:
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
            elif compare_fallback and fallback_usable:
                comparison = _structured_fallback_comparison(
                    text or "",
                    fallback_html or "",
                    **structured_fallback_options,
                )
                disposition["structured_fallback_comparison"] = comparison
                if comparison["prefer_fallback"]:
                    text = fallback_html
                    selected_backend = "bs4_structured_fallback"
                    selected_metrics = fallback_metrics
                    fallback_cleaned += 1
                    structurally_richer_fallbacks += 1

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

            reason_code = {
                "trafilatura": "accepted_trafilatura",
                "bs4_fallback": "accepted_bs4_fallback",
                "bs4_structured_fallback": "accepted_bs4_structured_fallback",
            }[selected_backend]
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
                cleaner_warnings.append("trafilatura_extraction_error_recovered")
            if cleaner_warnings:
                disposition["warnings"] = cleaner_warnings
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
            "fallback_compared": fallback_compared,
            "structurally_richer_fallbacks": structurally_richer_fallbacks,
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
