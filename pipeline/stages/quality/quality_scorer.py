"""
Quality scoring / filtering gate.

Filters out low-quality pages: login walls, error pages, too-short content,
and pages older than a configurable cutoff.
"""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Set

from bs4 import BeautifulSoup

from pipeline.core.base import QualityGate, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

# Strong patterns indicating a login wall or access-denied response. These are
# specific enough to reject whenever they appear in visible page text.
LOGIN_PATTERNS = [
    r"(?i)sign\s*in\s+to\s+continue",
    r"(?i)log\s*in\s+required",
    r"(?i)please\s+log\s*in",
    r"(?i)authentication\s+required",
    r"(?i)403\s+forbidden",
]

# These phrases also occur legitimately in articles and JavaScript bundles.
# Treat them as a wall only in a structural wall context (a leading response
# line, heading/alert, or authentication form), rather than rejecting a full
# public page on one token.
GENERIC_ACCESS_PATTERNS = [
    r"(?i)access\s+denied",
    r"(?i)unauthorized",
    r"(?i)you\s+are\s+not\s+authorized",
]

_ACCESS_PREFIX = r"(?:error\s*[:\-]\s*|(?:http\s*)?(?:401|403)\s*[:\-]?\s*)?"
_ACCESS_MARKER = (
    r"(?:access\s+denied|unauthorized(?:\s+access)?|you\s+are\s+not\s+authorized)"
)

GENERIC_ACCESS_HEADING = re.compile(
    rf"(?i)^\s*(?:#{{1,6}}\s*)?{_ACCESS_PREFIX}{_ACCESS_MARKER}\s*[.!]?\s*$"
)

GENERIC_ACCESS_RESPONSE = re.compile(
    rf"(?i)^\s*{_ACCESS_PREFIX}(?:"
    r"(?:access\s+denied|unauthorized(?:\s+access)?)"
    r"(?:\s*[.!:;\-]\s*[^\n]{0,160})?"
    r"|you\s+are\s+not\s+authorized"
    r"(?:(?:\s+(?:to|for)\b[^\n]{0,160})|(?:\s*[.!:;\-]\s*[^\n]{0,160}))?"
    r")\s*$"
)

ERROR_PATTERNS = [
    r"(?i)page\s+not\s+found",
    r"(?i)404\s+error",
    r"(?i)500\s+internal\s+server\s+error",
    r"(?i)service\s+unavailable",
    r"(?i)502\s+bad\s+gateway",
]


def _has_generic_access_marker(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in GENERIC_ACCESS_PATTERNS)


def _login_form_has_access_marker(form: Any) -> bool:
    """Require a marker inside, adjacent to, or tightly wrapping a real login form."""
    if _has_generic_access_marker(form.get_text(" ", strip=True)):
        return True

    for sibling in (form.find_previous_sibling(), form.find_next_sibling()):
        if sibling is None:
            continue
        sibling_text = sibling.get_text(" ", strip=True)
        if GENERIC_ACCESS_HEADING.fullmatch(sibling_text) or GENERIC_ACCESS_RESPONSE.fullmatch(
            sibling_text
        ):
            return True

    parent = form.parent
    if getattr(parent, "name", None) in {"main", "section", "div", "dialog"}:
        parent_text = parent.get_text(" ", strip=True)
        if len(parent_text) <= 500 and GENERIC_ACCESS_RESPONSE.fullmatch(parent_text):
            return True
    return False


def _visible_text_and_access_signals(text: str) -> tuple[str, bool, bool]:
    """Return visible text and contextual access-wall evidence."""
    if not re.search(r"<\s*(?:!doctype|html|head|body|main|article|form|script)\b", text, re.IGNORECASE):
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        first_line_pattern = (
            GENERIC_ACCESS_HEADING
            if re.match(r"^\s*#{1,6}\s", first_line)
            else GENERIC_ACCESS_RESPONSE
        )
        return text, False, bool(first_line_pattern.fullmatch(first_line))

    soup = BeautifulSoup(text, "lxml")
    for element in soup.select("script, style, noscript, template, svg"):
        element.decompose()
    for element in soup.select(
        "[hidden], [aria-hidden='true' i], "
        "[style*='display:none' i], [style*='display: none' i], "
        "[style*='visibility:hidden' i], [style*='visibility: hidden' i]"
    ):
        element.decompose()

    login_forms = soup.select(
        "form:has(input[type='password']), form:has(input[name*='password' i]), "
        "form[action*='login' i], form[action*='signin' i], "
        "form[id*='login' i], form[class*='login' i]"
    )
    form_has_access_marker = any(_login_form_has_access_marker(form) for form in login_forms)
    heading_has_access_marker = any(
        GENERIC_ACCESS_HEADING.fullmatch(element.get_text(" ", strip=True) or "")
        for element in soup.select("title, h1, h2")
    ) or any(
        GENERIC_ACCESS_RESPONSE.fullmatch(element.get_text(" ", strip=True) or "")
        for element in soup.select("[role='alert' i]")
    )
    return soup.get_text(" ", strip=True), form_has_access_marker, heading_has_access_marker


def _check_quality(text: str, min_length: int, detect_login: bool) -> str | None:
    """Return a rejection reason string, or None if the content passes."""
    visible_text, form_has_access_marker, heading_has_access_marker = (
        _visible_text_and_access_signals(text)
    )
    stripped_text = visible_text.strip()

    if len(stripped_text) < min_length:
        return "too_short"

    if detect_login:
        for pattern in LOGIN_PATTERNS:
            if re.search(pattern, stripped_text):
                return "login_wall"
        if form_has_access_marker or heading_has_access_marker:
            return "login_wall"

    for pattern in ERROR_PATTERNS:
        if re.search(pattern, stripped_text):
            return "error_page"

    return None


def _load_mapping(path: Any) -> Dict[str, str]:
    if not path:
        return {}
    data = load_json_safe(path, {}) or {}
    if not isinstance(data, dict):
        return {}
    return {
        str(url): str(value)
        for url, value in data.items()
        if isinstance(url, str) and isinstance(value, str)
    }


def _remove_file(path_str: str, removed_artifact_ids: List[str], artifact_ids_by_path: Dict[str, List[str]]) -> None:
    path = Path(path_str)
    path.unlink(missing_ok=True)
    removed_artifact_ids.extend(artifact_ids_by_path.get(str(path.resolve()), []))


def _mapping_value_is_available(value: str) -> bool:
    """Keep durable crawler evidence sentinels as well as existing files."""
    return value.startswith("SKIPPED") or Path(value).exists()


@register_stage
class QualityScorer(QualityGate):
    name = "quality_scorer"
    description = "Filters login walls, error pages, and short content."

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.quality_config

        artifact_ids_by_path: Dict[str, List[str]] = {}
        files: List[Path] = []
        input_mode = "unknown"

        markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
        cleaned_artifacts = ctx.find_artifacts(artifact_type="cleaned_html")
        raw_html_dir = ctx.previous_outputs.get("html_dir")
        if cleaned_artifacts:
            input_mode = "cleaned_html"
            for record in cleaned_artifacts:
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                if path.is_file():
                    files.append(path)
                    artifact_ids_by_path.setdefault(str(path.resolve()), []).append(record.artifact_id)
        elif raw_html_dir:
            html_dir = Path(raw_html_dir)
            if html_dir.is_dir():
                input_mode = "raw_html"
                files.extend(html_dir.rglob("*.html"))
        elif markdown_artifacts:
            input_mode = "markdown"
            for record in markdown_artifacts:
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                if path.is_file():
                    files.append(path)
                    artifact_ids_by_path.setdefault(str(path.resolve()), []).append(record.artifact_id)
        else:
            input_dir = (
                ctx.previous_outputs.get("cleaned_dir")
                or ctx.previous_outputs.get("html_dir")
                or ctx.previous_outputs.get("md_dir")
            )
            if not input_dir:
                return StageResult.skipped("No content directory in previous outputs")

            input_dir = Path(input_dir)
            if not input_dir.is_dir():
                return StageResult.skipped(f"Directory does not exist: {input_dir}")

            extensions = ("*.md", "*.html", "*.txt")
            for ext in extensions:
                files.extend(input_dir.rglob(ext))

        mapping_file = ctx.previous_outputs.get("mapping_file")
        md_mapping_file = ctx.previous_outputs.get("md_mapping_file")
        page_media_file = ctx.previous_outputs.get("page_media_file")
        page_images_file = ctx.previous_outputs.get("page_images_file")
        page_videos_file = ctx.previous_outputs.get("page_videos_file")

        url_to_html = _load_mapping(mapping_file)
        url_to_md = _load_mapping(md_mapping_file)
        raw_page_media = load_json_safe(page_media_file, {}) if page_media_file else {}
        raw_page_images = load_json_safe(page_images_file, {}) if page_images_file else {}
        raw_page_videos = load_json_safe(page_videos_file, {}) if page_videos_file else {}

        urls_by_path: Dict[str, Set[str]] = {}
        for mapping in (url_to_html, url_to_md):
            for url, path_str in mapping.items():
                try:
                    resolved = str(Path(path_str).resolve())
                except OSError:
                    continue
                urls_by_path.setdefault(resolved, set()).add(url)

        min_length = config.get("min_content_length", 100)
        detect_login = config.get("detect_login_walls", True)

        logger.info("Quality gate: scanning %d files from %s", len(files), input_mode)

        passed = 0
        filtered_items: List[str] = []
        reasons: Dict[str, int] = {}
        removed_artifact_ids: List[str] = []

        for fp in files:
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            reason = _check_quality(text, min_length, detect_login)
            if reason:
                filtered_items.append(str(fp))
                reasons[reason] = reasons.get(reason, 0) + 1
                resolved = str(fp.resolve())
                related_urls = set(urls_by_path.get(resolved, set()))
                if not related_urls:
                    _remove_file(resolved, removed_artifact_ids, artifact_ids_by_path)
                    continue

                for url in related_urls:
                    html_path = url_to_html.pop(url, "")
                    md_path = url_to_md.pop(url, "")
                    if html_path:
                        _remove_file(str(Path(html_path).resolve()), removed_artifact_ids, artifact_ids_by_path)
                    if md_path:
                        _remove_file(str(Path(md_path).resolve()), removed_artifact_ids, artifact_ids_by_path)
                    if isinstance(raw_page_media, dict):
                        raw_page_media.pop(url, None)
                    if isinstance(raw_page_images, dict):
                        raw_page_images.pop(url, None)
                    if isinstance(raw_page_videos, dict):
                        raw_page_videos.pop(url, None)
            else:
                passed += 1

        url_to_html = {
            url: path for url, path in url_to_html.items() if _mapping_value_is_available(path)
        }
        url_to_md = {
            url: path for url, path in url_to_md.items() if _mapping_value_is_available(path)
        }

        outputs = {
            "passed_count": passed,
            "filtered_count": len(filtered_items),
            "filtered_items": filtered_items,
        }

        if mapping_file:
            atomic_write_json(Path(mapping_file), url_to_html)
            outputs["mapping_file"] = str(mapping_file)
        if md_mapping_file:
            atomic_write_json(Path(md_mapping_file), url_to_md)
            outputs["md_mapping_file"] = str(md_mapping_file)
        if page_media_file and isinstance(raw_page_media, dict):
            atomic_write_json(Path(page_media_file), raw_page_media)
            outputs["page_media_file"] = str(page_media_file)
        if page_images_file and isinstance(raw_page_images, dict):
            atomic_write_json(Path(page_images_file), raw_page_images)
            outputs["page_images_file"] = str(page_images_file)
        if page_videos_file and isinstance(raw_page_videos, dict):
            atomic_write_json(Path(page_videos_file), raw_page_videos)
            outputs["page_videos_file"] = str(page_videos_file)

        logger.info(
            "Quality gate: passed=%d filtered=%d reasons=%s",
            passed, len(filtered_items), reasons,
        )

        return StageResult.success(
            outputs=outputs,
            metrics={"passed": passed, "filtered": len(filtered_items), **reasons},
            removed_artifact_ids=removed_artifact_ids,
        )
