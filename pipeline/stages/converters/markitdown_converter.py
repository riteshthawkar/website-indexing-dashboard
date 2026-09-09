"""
MarkItDown HTML-to-Markdown converter stage.

Converts cleaned HTML files to Markdown using the MarkItDown library.
Preserves structured image/video references in the markdown output.
Supports concurrent conversion via ThreadPoolExecutor.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pipeline.core.base import ConverterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe
from pipeline.core.media import build_media_markdown, dedupe_media_items
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)


def _unique_urls(values: Iterable[Any]) -> List[str]:
    """Return non-empty URL strings in stable first-seen order."""
    output: List[str] = []
    seen = set()
    for value in values:
        url = str(value or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(url)
    return output


def _preferred_source_url(
    urls: Iterable[Any],
    canonical_page_metadata: Mapping[str, Any],
) -> str:
    """Prefer a URL recognized by the canonical page inventory."""
    candidates = _unique_urls(urls)
    for url in candidates:
        if isinstance(canonical_page_metadata.get(url), dict):
            return url
    return candidates[0] if candidates else ""


def _convert_one(
    html_path: Path,
    md_target: Path,
    page_media: Optional[List[Dict[str, Any]]] = None,
    *,
    media_relative_to: Optional[Path] = None,
    max_images_in_markdown: int = 4,
    max_videos_in_markdown: int = 3,
    append_only_semantic_images: bool = True,
) -> tuple[str, str] | None:
    """Convert a single HTML file to Markdown. Returns (html_name, md_path) or None."""
    try:
        from markitdown import MarkItDown

        md = MarkItDown()
        md_path = md_target if md_target.suffix.lower() == ".md" else md_target / (html_path.stem + ".md")
        result = md.convert(str(html_path))
        text = result.text_content or ""

        if page_media:
            existing_urls = {
                media_url
                for media in dedupe_media_items(page_media)
                for media_url in (media.get("url"), media.get("local_path"), media.get("poster_url"))
                if media_url and media_url in text
            }
            media_to_append = [
                media
                for media in dedupe_media_items(page_media)
                if media.get("url") not in existing_urls and media.get("local_path") not in existing_urls
            ]
            media_block = build_media_markdown(
                media_to_append,
                prefer_local_images=True,
                response_mode=False,
                allow_html_video=False,
                relative_to=media_relative_to or md_path,
                max_images=max_images_in_markdown,
                max_videos=max_videos_in_markdown,
                skip_non_semantic_images=append_only_semantic_images,
            )
            if media_block:
                text += "\n\n" + media_block

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(text, encoding="utf-8")
        return (str(html_path.resolve()), str(md_path))
    except Exception as e:
        logger.error("MarkItDown conversion failed for %s: %s", html_path.name, e)
        return None


@register_stage
class MarkItDownConverter(ConverterStage):
    name = "markitdown"
    description = "Converts HTML to Markdown using MarkItDown with thread pool."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        errors = []
        try:
            from markitdown import MarkItDown  # noqa: F401
        except ImportError:
            errors.append("markitdown is not installed. Run: pip install markitdown")
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.converter_config

        html_artifacts = ctx.find_artifacts(artifact_type="cleaned_html") or ctx.find_artifacts(artifact_type="raw_html")
        input_dir = None
        if not html_artifacts:
            input_dir = ctx.previous_outputs.get("cleaned_dir") or ctx.previous_outputs.get("html_dir")
            if not input_dir:
                return StageResult.failure("No cleaned_dir or html_dir in previous outputs")

            input_dir = Path(input_dir)
            if not input_dir.is_dir():
                return StageResult.failure(f"Input dir does not exist: {input_dir}")

        md_dir = ensure_dir(ctx.output_dir("markdown"))
        workers = config.get("max_workers", 10)
        overwrite = config.get("overwrite", False)
        max_images_in_markdown = int(config.get("max_images_in_markdown", 4))
        max_videos_in_markdown = int(config.get("max_videos_in_markdown", 3))
        append_only_semantic_images = bool(config.get("append_only_semantic_images", True))

        if html_artifacts:
            files = [
                Path(record.local_path)
                for record in html_artifacts
                if record.local_path and Path(record.local_path).is_file()
            ]
        else:
            files = list(input_dir.glob("**/*.html"))
        logger.info("MarkItDown converter: %d HTML files, %d workers", len(files), workers)

        # Track URL→MD mapping
        url_mapping = ctx.previous_outputs.get("url_mapping", {})
        if not url_mapping:
            mapping_file = ctx.previous_outputs.get("mapping_file")
            if mapping_file:
                url_mapping = load_json_safe(mapping_file, {}) or {}

        all_page_media = ctx.previous_outputs.get("page_media", {})
        if not all_page_media:
            page_media_file = ctx.previous_outputs.get("page_media_file")
            if page_media_file:
                all_page_media = load_json_safe(page_media_file, {}) or {}

        # Backward compatibility with image-only runs.
        if not all_page_media:
            all_page_images = ctx.previous_outputs.get("page_images", {})
            if not all_page_images:
                page_images_file = ctx.previous_outputs.get("page_images_file")
                if page_images_file:
                    all_page_images = load_json_safe(page_images_file, {}) or {}
            all_page_media = all_page_images

        canonical_page_metadata: Dict[str, Any] = {}
        canonical_page_metadata_file = (
            ctx.previous_outputs.get("canonical_page_metadata_file")
            or ctx.previous_outputs.get("page_metadata_file")
        )
        if canonical_page_metadata_file:
            loaded_page_metadata = load_json_safe(canonical_page_metadata_file, {}) or {}
            if isinstance(loaded_page_metadata, dict):
                canonical_page_metadata = loaded_page_metadata

        # Build html_path -> page_media lookup. Keep every explicitly supplied
        # URL identity: cleaned artifacts can have a legacy request URL plus a
        # canonical redirect target, and both must resolve to the same Markdown.
        html_to_media: Dict[str, List[Dict[str, Any]]] = {}
        html_to_urls: Dict[str, List[str]] = {}
        html_to_artifact_ids: Dict[str, List[str]] = {}
        html_relative_paths: Dict[str, Path] = {}
        for record in html_artifacts:
            if not record.local_path:
                continue
            html_key = str(Path(record.local_path).resolve())
            html_to_artifact_ids[html_key] = [record.artifact_id]
            metadata_source_urls = record.metadata.get("source_urls") or []
            if not isinstance(metadata_source_urls, (list, tuple, set)):
                metadata_source_urls = [metadata_source_urls]
            record_source_urls = _unique_urls(
                [record.metadata.get("source_url"), *metadata_source_urls]
            )
            if record_source_urls:
                html_to_urls[html_key] = record_source_urls
                media_items = [
                    item
                    for source_url in record_source_urls
                    for item in (all_page_media.get(source_url) or [])
                    if isinstance(item, dict)
                ]
                if media_items:
                    html_to_media[html_key] = dedupe_media_items(media_items)
            relative_path = record.metadata.get("relative_path")
            if relative_path:
                html_relative_paths[html_key] = Path(str(relative_path))

        for page_url, media_items in all_page_media.items():
            html_path_str = url_mapping.get(page_url, "")
            if html_path_str:
                html_key = str(Path(html_path_str).resolve())
                html_to_urls[html_key] = _unique_urls(
                    [*(html_to_urls.get(html_key) or []), page_url]
                )
                html_to_media[html_key] = dedupe_media_items(
                    [
                        *(html_to_media.get(html_key) or []),
                        *(media_items or []),
                    ]
                )

        md_mapping: Dict[str, str] = {}
        lock = Lock()
        artifacts = []

        converted = 0
        failed = 0

        # Filter already-done files
        to_process: List[tuple[Path, Path]] = []
        for fp in files:
            html_key = str(fp.resolve())
            relative = html_relative_paths.get(html_key)
            if relative is None:
                relative = fp.relative_to(input_dir) if input_dir else Path(fp.name)
            md_path = md_dir / relative.with_suffix(".md")
            if md_path.exists() and not overwrite:
                md_mapping[html_key] = str(md_path)
                converted += 1
                continue
            to_process.append((fp, md_path))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _convert_one, fp, md_path,
                    html_to_media.get(str(fp.resolve())),
                    media_relative_to=md_path,
                    max_images_in_markdown=max_images_in_markdown,
                    max_videos_in_markdown=max_videos_in_markdown,
                    append_only_semantic_images=append_only_semantic_images,
                ): fp
                for fp, md_path in to_process
            }
            for future in as_completed(futures):
                result = future.result()
                if result:
                    html_name, md_path_str = result
                    with lock:
                        md_mapping[html_name] = md_path_str
                    converted += 1
                else:
                    failed += 1

        # Build URL→MD mapping from URL→HTML mapping
        url_to_md: Dict[str, str] = {}
        for html_key, md_path_str in md_mapping.items():
            source_urls = html_to_urls.get(html_key, [])
            source_url = _preferred_source_url(
                source_urls,
                canonical_page_metadata,
            )
            for alias_url in source_urls:
                url_to_md[alias_url] = md_path_str
            artifacts.append(
                ctx.make_artifact(
                    md_path_str,
                    artifact_type="markdown",
                    role="content",
                    metadata={
                        "source_url": source_url,
                        "source_urls": source_urls,
                        "source_html_path": html_key,
                        "source_type": "webpage",
                        "backend": "markitdown",
                    },
                    source_artifact_ids=html_to_artifact_ids.get(html_key),
                )
            )

        # Save mapping
        mapping_file = ctx.stage_work_dir / "url_to_md_mapping.json"
        atomic_write_json(mapping_file, url_to_md)
        artifacts.append(
            ctx.make_artifact(
                mapping_file,
                artifact_type="mapping",
                role="url_to_markdown",
                metadata={"entries": len(url_to_md)},
            )
        )

        logger.info("MarkItDown converter done: converted=%d failed=%d", converted, failed)

        return StageResult.success(
            outputs={
                "md_dir": str(md_dir),
                "md_mapping_file": str(mapping_file),
                "md_count": converted,
            },
            metrics={"converted": converted, "failed": failed},
            artifacts=artifacts,
        )
