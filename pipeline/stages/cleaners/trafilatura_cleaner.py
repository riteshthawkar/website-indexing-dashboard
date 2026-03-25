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
from pipeline.core.registry import register_stage
from pipeline.stages.cleaners.bs4_cleaner import clean_html_content

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
        errors = []
        try:
            import trafilatura  # noqa: F401
        except ImportError:
            errors.append("trafilatura is not installed. Run: pip install trafilatura")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        import trafilatura

        config = ctx.cleaner_config
        html_dir = ctx.previous_outputs.get("html_dir")
        if not html_dir:
            return StageResult.failure("No html_dir in previous outputs")

        html_dir = Path(html_dir)
        if not html_dir.is_dir():
            return StageResult.failure(f"html_dir does not exist: {html_dir}")

        cleaned_dir = ctx.output_dir("cleaned_html")

        include_tables = config.get("include_tables", True)
        include_links = config.get("include_links", True)
        include_images = config.get(
            "preserve_embedded_media",
            ctx.crawler_config.get("extract_images", True),
        )
        min_length = config.get("min_content_length", 100)

        files = list(html_dir.glob("**/*.html"))
        logger.info("Trafilatura cleaner found %d HTML files", len(files))

        cleaned = 0
        removed = 0
        errors = 0
        fallback_cleaned = 0
        artifacts = []

        url_mapping = {}
        mapping_file = ctx.previous_outputs.get("mapping_file")
        if mapping_file:
            from pipeline.core.io import load_json_safe

            url_mapping = load_json_safe(mapping_file, {}) or {}

        for i, fp in enumerate(files, 1):
            try:
                raw = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                logger.warning("Read error %s: %s", fp, e)
                errors += 1
                continue

            text = trafilatura.extract(
                raw,
                include_tables=include_tables,
                include_images=include_images,
                include_links=include_links,
                output_format="html",
                favor_recall=True,
            )

            if text and _looks_navigation_heavy(text):
                text = ""

            if not text or len(text.strip()) < min_length:
                fallback_status, fallback_html = clean_html_content(
                    raw,
                    preserve_media=config.get(
                        "preserve_embedded_media",
                        ctx.crawler_config.get("extract_images", True) or ctx.crawler_config.get("extract_videos", True),
                    ),
                )
                if fallback_status == "cleaned" and fallback_html and len(fallback_html.strip()) >= min_length:
                    text = fallback_html
                    fallback_cleaned += 1
                else:
                    removed += 1
                    continue

            relative = fp.relative_to(html_dir)
            out_path = cleaned_dir / relative
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text, encoding="utf-8")
            cleaned += 1
            source_url = next(
                (
                    url
                    for url, html_path in url_mapping.items()
                    if Path(html_path).name == fp.name
                ),
                "",
            )
            artifacts.append(
                ctx.make_artifact(
                    out_path,
                    artifact_type="cleaned_html",
                    role="content",
                    metadata={
                        "source_path": str(fp),
                        "source_url": source_url,
                        "relative_path": relative.as_posix(),
                    },
                )
            )

            if i % 100 == 0:
                logger.info("Progress: %d/%d", i, len(files))

        logger.info(
            "Trafilatura cleaner done: cleaned=%d removed=%d errors=%d",
            cleaned, removed, errors,
        )

        return StageResult.success(
            outputs={
                "cleaned_dir": str(cleaned_dir),
                "cleaned_count": cleaned,
                "removed_count": removed,
            },
            metrics={
                "cleaned": cleaned,
                "removed": removed,
                "errors": errors,
                "fallback_cleaned": fallback_cleaned,
            },
            artifacts=artifacts,
        )
