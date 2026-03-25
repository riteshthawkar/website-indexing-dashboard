"""
BeautifulSoup-based HTML cleaner stage.

Strips boilerplate tags, removes noise elements, extracts main content,
and cleans attributes. Deletes 404 / empty pages.
"""

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Comment, Doctype

from pipeline.core.base import CleanerStage, StageContext, StageResult
from pipeline.core.io import load_json_safe
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

# Default tags to strip entirely
REMOVE_TAGS = [
    "script", "noscript", "style", "link", "php",
    "nav", "navbar", "header", "footer",
]
BODY_CLEAN_TAGS = [
    "meta", "link", "script", "noscript", "nav", "navbar",
    "header", "footer", "aside", "form", "iframe",
    "button", "input", "svg", "canvas", "template",
]
KEEP_META_NAMES = ["description", "keywords"]
NOISE_KEYS = [
    "cookie", "consent", "subscribe", "newsletter", "advert", "ad-",
    "promo", "banner", "share", "social", "breadcrumb", "sidebar",
    "toc", "related", "comments", "rating",
]


# ── helpers ──────────────────────────────────────────────────────────

def _remove_unwanted_tags(soup, tags):
    for tag_name in tags:
        for tag in soup.find_all(tag_name):
            tag.decompose()


def _remove_comments(soup):
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()


def _safe_tag_attr(tag, name: str, default=None):
    attrs = getattr(tag, "attrs", None)
    if not isinstance(attrs, dict):
        return default
    value = attrs.get(name, default)
    return default if value is None else value


def _process_head(soup) -> Tuple[Optional[str], Dict[str, str]]:
    title_text = None
    meta_contents: Dict[str, str] = {}
    if soup.head:
        if soup.head.title and soup.head.title.string:
            title_text = soup.head.title.string.strip()
            soup.head.title.decompose()
        for meta in soup.head.find_all("meta"):
            name = str(_safe_tag_attr(meta, "name", "")).lower()
            content = _safe_tag_attr(meta, "content")
            if name in KEEP_META_NAMES and content:
                meta_contents[name] = content.strip()
            meta.decompose()
    return title_text, meta_contents


def _inject_metadata(soup, title_text):
    if not soup.body:
        return
    if title_text and not soup.body.find(["h1", "h2"]):
        h1 = soup.new_tag("h1")
        h1.string = title_text
        soup.body.insert(0, h1)


def _strip_attributes(soup):
    strip = {"class", "style", "id", "role"}
    for tag in soup.find_all(True):
        attrs = getattr(tag, "attrs", None)
        if not isinstance(attrs, dict):
            continue
        for attr in list(attrs):
            if attr in strip or attr.startswith(("data-", "aria-", "on")):
                del tag.attrs[attr]


def _remove_hidden(soup):
    for el in soup.find_all(True):
        attrs = getattr(el, "attrs", None)
        if not isinstance(attrs, dict):
            continue

        if "hidden" in attrs or str(attrs.get("aria-hidden", "")).lower() == "true":
            el.decompose()
            continue

        style = str(attrs.get("style", "")).lower()
        if "display:none" in style or "visibility:hidden" in style:
            el.decompose()


def _replace_images_with_alt(soup):
    for img in soup.find_all("img"):
        alt = str(_safe_tag_attr(img, "alt", "")).strip()
        if alt:
            img.replace_with(soup.new_string(alt))
        else:
            img.decompose()


def _remove_non_video_iframes(soup):
    for iframe in soup.find_all("iframe"):
        src = str(_safe_tag_attr(iframe, "src", "")).lower()
        if not any(token in src for token in ("youtube", "vimeo", "wistia", "/embed/", "/video")):
            iframe.decompose()


def _remove_noise(soup):
    def is_noise(tag):
        classes_value = _safe_tag_attr(tag, "class", [])
        if isinstance(classes_value, str):
            classes_value = [classes_value]
        classes = " ".join(classes_value).lower()
        id_ = str(_safe_tag_attr(tag, "id", "")).lower()
        return any(k in classes or k in id_ for k in NOISE_KEYS)
    for tag in soup.find_all(is_noise):
        tag.decompose()


def _keep_main_content(soup):
    main = soup.find(["main", "article"]) or soup.find(attrs={"role": "main"})
    if main:
        return BeautifulSoup(f"<html><body>{main}</body></html>", "html.parser")
    return soup


def _collapse_blanks(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", text)


def clean_html_content(raw: str, preserve_media: bool = False) -> Tuple[str, Optional[str]]:
    """Clean HTML content and return (status, cleaned_html)."""
    if re.search(r"(?i)page not found", raw):
        return "removed", None

    soup = BeautifulSoup(raw, "html.parser")

    _remove_unwanted_tags(soup, REMOVE_TAGS)
    _remove_comments(soup)
    title, _ = _process_head(soup)
    soup = _keep_main_content(soup)
    _remove_hidden(soup)
    _remove_noise(soup)

    if not soup.body:
        body_tag = soup.new_tag("body")
        html_tag = soup.find("html")
        if html_tag:
            html_tag.append(body_tag)
        else:
            soup.append(body_tag)

    if soup.body:
        _inject_metadata(soup, title)
        body_clean_tags = BODY_CLEAN_TAGS
        if preserve_media:
            body_clean_tags = [tag for tag in BODY_CLEAN_TAGS if tag != "iframe"]
        _remove_unwanted_tags(soup.body, body_clean_tags)
        if preserve_media:
            _remove_non_video_iframes(soup)
        else:
            _replace_images_with_alt(soup)
        _strip_attributes(soup)

    for item in list(soup.contents):
        if isinstance(item, Doctype):
            item.extract()

    text = soup.get_text(separator=" ", strip=True)
    if not text:
        return "removed_empty", None

    return "cleaned", _collapse_blanks(str(soup))


def clean_single_file(file_path: str, preserve_media: bool = False) -> str:
    """Clean one HTML file in-place. Returns status string."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        logger.error("Read error %s: %s", file_path, e)
        return "error"

    status, cleaned_html = clean_html_content(raw, preserve_media=preserve_media)
    if status in {"removed", "removed_empty"}:
        os.remove(file_path)
        return status
    if status != "cleaned" or cleaned_html is None:
        return status

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(cleaned_html)
    return status


# ── Stage ────────────────────────────────────────────────────────────

@register_stage
class BS4Cleaner(CleanerStage):
    name = "bs4"
    description = "BeautifulSoup-based HTML cleaner — strips boilerplate, noise, and empty pages."

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.cleaner_config
        html_dir = ctx.previous_outputs.get("html_dir")
        if not html_dir:
            return StageResult.failure("No html_dir in previous outputs")

        html_dir = Path(html_dir)
        if not html_dir.is_dir():
            return StageResult.failure(f"html_dir does not exist: {html_dir}")

        recursive = config.get("recursive", True)
        preserve_media = config.get(
            "preserve_embedded_media",
            ctx.crawler_config.get("extract_images", True) or ctx.crawler_config.get("extract_videos", True),
        )
        cleaned_dir = ctx.output_dir("cleaned_html")
        pattern = "**/*.html" if recursive else "*.html"
        files = list(html_dir.glob(pattern))
        logger.info("BS4 cleaner found %d HTML files in %s", len(files), html_dir)

        counts = {"cleaned": 0, "removed": 0, "removed_empty": 0, "error": 0}
        artifacts = []
        url_mapping = load_json_safe(ctx.previous_outputs.get("mapping_file"), {}) or {}
        for i, fp in enumerate(files, 1):
            try:
                raw = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                logger.error("Read error %s: %s", fp, exc)
                counts["error"] = counts.get("error", 0) + 1
                continue

            status, cleaned_html = clean_html_content(raw, preserve_media=preserve_media)
            counts[status] = counts.get(status, 0) + 1
            if status == "cleaned" and cleaned_html is not None:
                relative = fp.relative_to(html_dir)
                out_path = cleaned_dir / relative
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(cleaned_html, encoding="utf-8")

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
            "BS4 cleaner done: cleaned=%d removed=%d removed_empty=%d errors=%d",
            counts["cleaned"], counts["removed"], counts["removed_empty"], counts["error"],
        )

        return StageResult.success(
            outputs={
                "cleaned_dir": str(cleaned_dir),
                "cleaned_count": counts["cleaned"],
                "removed_count": counts["removed"] + counts["removed_empty"],
            },
            metrics=counts,
            artifacts=artifacts,
        )
