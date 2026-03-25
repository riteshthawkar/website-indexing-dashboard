"""
Crawl4AI fit_markdown converter stage.

If the Crawl4AI crawler already produced markdown via fit_markdown,
this stage is a no-op pass-through. Otherwise, it re-processes HTML
files through Crawl4AI's markdown generator for high-quality output.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import ConverterStage, StageContext, StageResult
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


@register_stage
class Crawl4AIMdConverter(ConverterStage):
    name = "crawl4ai_md"
    description = "Uses Crawl4AI fit_markdown for main-content-only Markdown."

    async def execute(self, ctx: StageContext) -> StageResult:
        md_dir = ctx.previous_outputs.get("md_dir")

        # If crawl4ai already produced markdown, just pass through
        if md_dir:
            md_path = Path(md_dir)
            if md_path.is_dir():
                count = len(list(md_path.glob("*.md")))
                if count > 0:
                    logger.info("Crawl4AI markdown already available: %d files", count)
                    return StageResult.success(
                        outputs={"md_dir": str(md_path), "md_count": count},
                        metrics={"pass_through": True, "count": count},
                    )

        # Otherwise, re-process HTML through Crawl4AI markdown generator
        html_dir = ctx.previous_outputs.get("cleaned_dir") or ctx.previous_outputs.get("html_dir")
        if not html_dir:
            return StageResult.failure("No html_dir or cleaned_dir available")

        html_dir = Path(html_dir)
        if not html_dir.is_dir():
            return StageResult.failure(f"HTML dir does not exist: {html_dir}")

        try:
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
            from crawl4ai.content_filter_strategy import PruningContentFilter
        except ImportError:
            return StageResult.failure("crawl4ai is not installed")

        md_dir = ctx.output_dir("markdown")

        threshold = ctx.converter_config.get("content_filter_threshold", 0.48)
        content_filter = PruningContentFilter(threshold=threshold)
        md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

        files = list(html_dir.glob("**/*.html"))
        logger.info("Crawl4AI MD converter: %d HTML files", len(files))

        converted = 0
        failed = 0
        artifacts = []

        for fp in files:
            try:
                html = fp.read_text(encoding="utf-8", errors="replace")
                result = md_generator.generate_markdown(
                    cleaned_html=html,
                    base_url="",
                )
                # result may be a MarkdownGenerationResult or similar
                md_text = getattr(result, "fit_markdown", None) or str(result)

                if md_text and md_text.strip():
                    relative = fp.relative_to(html_dir)
                    out = md_dir / relative.with_suffix(".md")
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(md_text, encoding="utf-8")
                    artifacts.append(
                        ctx.make_artifact(
                            out,
                            artifact_type="markdown",
                            role="content",
                            metadata={
                                "source_html_path": str(fp),
                                "source_type": "webpage",
                                "backend": "crawl4ai_md",
                            },
                        )
                    )
                    converted += 1
                else:
                    failed += 1
            except Exception as e:
                logger.debug("Crawl4AI md conversion failed for %s: %s", fp.name, e)
                failed += 1

        logger.info("Crawl4AI MD converter: converted=%d failed=%d", converted, failed)

        return StageResult.success(
            outputs={"md_dir": str(md_dir), "md_count": converted},
            metrics={"converted": converted, "failed": failed},
            artifacts=artifacts,
        )
