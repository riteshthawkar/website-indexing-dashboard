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

# Login phrases can also appear in legitimate public instructions or articles.
# Evaluate them only in a structural wall context, never as page-wide tokens.
LOGIN_PATTERNS = [
    r"(?i)sign\s*in\s+to\s+continue",
    r"(?i)log\s*in\s+required",
    r"(?i)please\s+log\s*in",
    r"(?i)authentication\s+required",
    r"(?i)403\s+forbidden",
]

_AUTH_OBJECT = (
    r"(?:this|the|your)?\s*(?:protected|private|restricted|institutional)?\s*"
    r"(?:account|application|content|page|portal|resource|site)"
)
_AUTH_CONTINUATION = (
    rf"(?:"
    rf"to\s+continue(?:\s+(?:to\s+)?(?:access|accessing|use|using|view|viewing)"
    rf"(?:\s+{_AUTH_OBJECT})?)?"
    rf"|to\s+(?:access|view|open|use)(?:\s+{_AUTH_OBJECT})?"
    r"|to\s+your\s+account"
    r"|(?:with|using)\s+(?:your\s+)?(?:institutional\s+)?(?:account|credentials)"
    r"(?:\s+to\s+continue)?"
    r")"
)
_LOGIN_WALL_MARKER = (
    rf"(?:"
    rf"(?:please\s+)?(?:log|sign)\s*in(?:\s+{_AUTH_CONTINUATION})?"
    rf"|login\s+required(?:\s+to\s+(?:continue|access|view|open|use)(?:\s+{_AUTH_OBJECT})?)?"
    r"|authentication\s+required(?:\s+to\s+(?:continue|access|view|open|use))?"
    r"|(?:http\s*)?403\s*[:\-]?\s*forbidden"
    r")"
)

LOGIN_WALL_HEADING = re.compile(
    rf"(?i)^\s*(?:#{{1,6}}\s*)?{_LOGIN_WALL_MARKER}\s*[.!]?\s*$"
)

LOGIN_WALL_RESPONSE = re.compile(
    rf"(?i)^\s*{_LOGIN_WALL_MARKER}"
    r"(?:\s*[.!]\s*|\s*[.!;\-]\s*(?:you\s+(?:do\s+not|don't)\s+have\s+permission|"
    r"contact\s+(?:the|your)\s+administrator|use\s+your\s+(?:account|credentials)|"
    r"(?:reference|request)\s+(?:id|number)\s*[:#\-]?\s*[a-z0-9\-]+"
    r"(?:\s*[.!;\-]\s*contact\s+(?:the|your)\s+administrator)?"
    r")"
    r"(?:\s+[^\n]{0,100})?\s*[.!]?)?\s*$"
)

AUTH_PORTAL_HEADING = re.compile(
    r"(?i)^\s*(?:#{1,6}\s*)?(?:account\s+portal|login|log\s*in|sign\s*in)\s*[.!]?\s*$"
)

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
    r"(?:\s*[.!;\-]\s*(?:contact\s+(?:the|your)\s+administrator|"
    r"you\s+(?:do\s+not|don't)\s+have\s+(?:access|permission)|"
    r"your\s+request\s+(?:cannot|could\s+not)\s+be\s+completed|"
    r"please\s+(?:log|sign)\s*in)"
    r"(?:\s+[^\n]{0,100})?\s*[.!]?)?"
    r"|you\s+are\s+not\s+authorized"
    r"(?:\s+to\s+(?:access|continue|open|use|view)\b[^\n]{0,120})?"
    r")\s*$"
)

ERROR_PATTERNS = [
    r"(?i)page\s+not\s+found",
    r"(?i)404\s+error",
    r"(?i)500\s+internal\s+server\s+error",
    r"(?i)service\s+unavailable",
    r"(?i)502\s+bad\s+gateway",
]

_ERROR_MARKER = (
    r"(?:page\s+not\s+found|error\s+404|"
    r"(?:http\s*)?404(?:\s+error|\s*[:\-]?\s*not\s+found)?|"
    r"(?:http\s*)?500\s+internal\s+server\s+error|service\s+unavailable|"
    r"(?:http\s*)?502\s+bad\s+gateway|(?:http\s*)?503\s+service\s+unavailable)"
)

ERROR_PAGE_HEADING = re.compile(
    rf"(?i)^\s*(?:#{{1,6}}\s*)?{_ERROR_MARKER}\s*[.!]?\s*$"
)

ERROR_PAGE_RESPONSE = re.compile(
    rf"(?i)^\s*{_ERROR_MARKER}"
    r"(?:\s+(?:occurred|nginx|apache|cloudflare)|"
    r"\s*[.!;\-]\s*(?:(?:the|this)\s+(?:page|url)\s+(?:does\s+not|doesn't)\s+exist|"
    r"(?:the\s+)?url\s+may\s+be\s+incorrect|please\s+try\s+again\s+later|"
    r"the\s+server\s+is\s+temporarily\s+unavailable|(?:an?\s+)?error\s+occurred|"
    r"occurred)"
    r"(?:\s*[.!]\s*(?:visit\s+(?:the\s+)?homepage|contact\s+support)"
    r"(?:\s+or\s+(?:visit\s+(?:the\s+)?homepage|contact\s+support))?)?"
    r")?\s*[.!]?\s*$"
)

_WALL_DOMINANT_MAX_CHARS = 1200
_NON_CONTENT_CONTAINERS = {"aside", "footer", "header", "nav"}


def _has_generic_access_marker(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in GENERIC_ACCESS_PATTERNS)


def _has_login_wall_marker(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in LOGIN_PATTERNS)


def _matches_wall_heading(text: str) -> bool:
    return bool(
        GENERIC_ACCESS_HEADING.fullmatch(text) or LOGIN_WALL_HEADING.fullmatch(text)
    )


def _matches_wall_response(text: str) -> bool:
    return bool(
        GENERIC_ACCESS_RESPONSE.fullmatch(text) or LOGIN_WALL_RESPONSE.fullmatch(text)
    )


def _matches_error_heading(text: str) -> bool:
    return bool(ERROR_PAGE_HEADING.fullmatch(text))


def _matches_error_response(text: str) -> bool:
    return bool(ERROR_PAGE_RESPONSE.fullmatch(text))


def _normalized_text(element: Any) -> str:
    return re.sub(r"\s+", " ", element.get_text(" ", strip=True) or "").strip()


def _has_widget_ancestor(form: Any, primary: Any) -> bool:
    for ancestor in form.parents:
        if ancestor is primary:
            break
        attributes = " ".join(
            [
                str(ancestor.get("id") or ""),
                " ".join(str(item) for item in (ancestor.get("class") or [])),
                str(ancestor.get("role") or ""),
            ]
        )
        if re.search(r"(?i)\b(?:account-)?widget\b|\bsidebar\b|\bmenu\b", attributes):
            return True
    return False


def _login_form_has_access_marker(
    form: Any,
    *,
    primary: Any,
    primary_text_length: int,
) -> bool:
    """Identify a primary-content login form without trusting global widgets."""
    short_shell = primary_text_length <= _WALL_DOMINANT_MAX_CHARS
    if short_shell:
        return True
    if _has_widget_ancestor(form, primary):
        return False

    form_text = _normalized_text(form)
    if (
        _matches_wall_response(form_text)
        or _has_generic_access_marker(form_text)
        or _has_login_wall_marker(form_text)
    ):
        return True

    for sibling in (form.find_previous_sibling(), form.find_next_sibling()):
        if sibling is None:
            continue
        sibling_text = _normalized_text(sibling)
        if (
            _matches_wall_heading(sibling_text)
            or _matches_wall_response(sibling_text)
            or AUTH_PORTAL_HEADING.fullmatch(sibling_text)
        ):
            return True

    for container in form.parents:
        if getattr(container, "name", None) not in {"main", "section", "div", "dialog"}:
            continue
        container_text = _normalized_text(container)
        if len(container_text) <= 600 and any(
            _matches_wall_heading(_normalized_text(element))
            or _matches_wall_response(_normalized_text(element))
            or AUTH_PORTAL_HEADING.fullmatch(_normalized_text(element))
            for element in container.select(
                "h1, h2, h3, h4, h5, h6, p, [role='alert' i]"
            )
        ):
            return True
        if container is primary:
            break
    return False


def _plain_text_quality_signals(text: str) -> tuple[bool, bool]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False, False

    first_line = lines[0]
    short_shell = len(text.strip()) <= _WALL_DOMINANT_MAX_CHARS
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not short_shell:
        return False, False

    candidate_lines = lines[:5]
    access_wall = _matches_wall_response(normalized_text) or any(
        _matches_wall_heading(line) or _matches_wall_response(line)
        for line in candidate_lines
    )
    error_page = _matches_error_response(normalized_text) or any(
        _matches_error_heading(line) or _matches_error_response(line)
        for line in candidate_lines
    )

    if short_shell and AUTH_PORTAL_HEADING.fullmatch(first_line):
        access_wall = len(lines) == 1 or any(
            _matches_wall_response(line) for line in lines[1:3]
        )
    return access_wall, error_page


def _primary_content(soup: BeautifulSoup) -> Any:
    return (
        soup.find("main")
        or soup.find(attrs={"role": re.compile(r"^main$", re.IGNORECASE)})
        or soup.find("article")
        or soup.body
        or soup
    )


def _visible_text_and_quality_signals(text: str) -> tuple[str, bool, bool]:
    """Return visible text plus structural access-wall and error-page evidence."""
    if not re.search(
        r"<\s*(?:!doctype|html|head|body|main|article|section|div|p|h[1-6]|form|script)\b",
        text,
        re.IGNORECASE,
    ):
        access_wall, error_page = _plain_text_quality_signals(text)
        return text, access_wall, error_page

    soup = BeautifulSoup(text, "lxml")
    for element in soup.select("script, style, noscript, template, svg"):
        element.decompose()
    for element in soup.select(
        "[hidden], [aria-hidden='true' i], "
        "[style*='display:none' i], [style*='display: none' i], "
        "[style*='visibility:hidden' i], [style*='visibility: hidden' i]"
    ):
        element.decompose()

    visible_text = soup.get_text(" ", strip=True)
    for element in soup.find_all(_NON_CONTENT_CONTAINERS):
        element.decompose()
    primary = _primary_content(soup)
    primary_text = _normalized_text(primary)
    short_shell = len(primary_text) <= _WALL_DOMINANT_MAX_CHARS

    login_forms = primary.select(
        "form:has(input[type='password']), form:has(input[name*='password' i]), "
        "form[action*='login' i], form[action*='signin' i], "
        "form[id*='login' i], form[class*='login' i]"
    )
    headings = primary.select("h1, h2, h3, h4, h5, h6")
    title_text = _normalized_text(soup.title) if soup.title else ""
    heading_has_access_marker = short_shell and any(
        _matches_wall_heading(_normalized_text(element)) for element in headings
    )
    title_has_access_marker = short_shell and bool(
        title_text and _matches_wall_heading(title_text)
    )
    alert_has_access_marker = any(
        _matches_wall_response(_normalized_text(element))
        for element in primary.select("[role='alert' i]")
    )
    form_has_access_marker = any(
        _login_form_has_access_marker(
            form,
            primary=primary,
            primary_text_length=len(primary_text),
        )
        for form in login_forms
    )

    block_texts = [
        _normalized_text(element)
        for element in primary.select("h1, h2, h3, h4, h5, h6, p, [role='alert' i], div")
        if _normalized_text(element)
    ]
    short_shell_has_access_response = short_shell and any(
        _matches_wall_heading(block_text) or _matches_wall_response(block_text)
        for block_text in block_texts[:12]
    )
    access_wall = (
        heading_has_access_marker
        or title_has_access_marker
        or alert_has_access_marker
        or form_has_access_marker
        or short_shell_has_access_response
    )

    heading_has_error_marker = short_shell and any(
        _matches_error_heading(_normalized_text(element)) for element in headings
    )
    title_has_error_marker = short_shell and bool(
        title_text and _matches_error_heading(title_text)
    )
    alert_has_error_marker = any(
        _matches_error_response(_normalized_text(element))
        for element in primary.select("[role='alert' i]")
    )
    short_shell_has_error_response = short_shell and any(
        _matches_error_heading(block_text) or _matches_error_response(block_text)
        for block_text in block_texts[:12]
    )
    error_page = (
        heading_has_error_marker
        or title_has_error_marker
        or alert_has_error_marker
        or short_shell_has_error_response
    )

    return visible_text, access_wall, error_page


def _check_quality(text: str, min_length: int, detect_login: bool) -> str | None:
    """Return a rejection reason string, or None if the content passes."""
    visible_text, access_wall, error_page = _visible_text_and_quality_signals(text)
    stripped_text = visible_text.strip()

    if len(stripped_text) < min_length:
        return "too_short"

    if detect_login:
        if access_wall:
            return "login_wall"

    if error_page:
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
