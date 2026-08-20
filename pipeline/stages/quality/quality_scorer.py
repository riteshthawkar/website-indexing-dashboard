"""
Quality scoring / filtering gate.

Filters out low-quality pages: login walls, error pages, too-short content,
and pages older than a configurable cutoff.
"""

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set

from bs4 import BeautifulSoup

from pipeline.core.base import QualityGate, StageContext, StageResult
from pipeline.core.io import (
    atomic_copy_file,
    atomic_write_json,
    load_json_safe,
    reset_stage_output_directory,
)
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

    return primary_text, access_wall, error_page


def _quality_assessment(
    text: str,
    min_length: int,
    detect_login: bool,
) -> tuple[str | None, str]:
    """Return a rejection reason and the normalized visible text."""

    visible_text, access_wall, error_page = _visible_text_and_quality_signals(text)
    stripped_text = visible_text.strip()

    if detect_login:
        if access_wall:
            return "login_wall", stripped_text

    if error_page:
        return "error_page", stripped_text

    if len(stripped_text) < min_length:
        return "too_short", stripped_text

    return None, stripped_text


def _check_quality(text: str, min_length: int, detect_login: bool) -> str | None:
    """Return a rejection reason string, or None if the content passes."""

    reason, _visible_text = _quality_assessment(text, min_length, detect_login)
    return reason


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


def _validate_quality_config(config: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    for key, default in (
        ("min_content_length", 100),
        ("maximum_error_count", 0),
    ):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"quality.{key} must be a non-negative integer")
    for key, default in (
        ("minimum_retention_ratio", 0.70),
        ("maximum_error_ratio", 0.0),
    ):
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"quality.{key} must be a number between 0 and 1")
        elif not 0.0 <= float(value) <= 1.0:
            errors.append(f"quality.{key} must be between 0 and 1")
    for key, default in (
        ("detect_login_walls", True),
        ("fail_on_empty_input", True),
        ("fail_on_zero_output", True),
    ):
        if not isinstance(config.get(key, default), bool):
            errors.append(f"quality.{key} must be a boolean")
    return errors


def _quality_policy(config: Mapping[str, Any]) -> Dict[str, Any]:
    errors = _validate_quality_config(config)
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "min_content_length": int(config.get("min_content_length", 100)),
        "detect_login_walls": bool(config.get("detect_login_walls", True)),
        "minimum_retention_ratio": float(config.get("minimum_retention_ratio", 0.70)),
        "maximum_error_count": int(config.get("maximum_error_count", 0)),
        "maximum_error_ratio": float(config.get("maximum_error_ratio", 0.0)),
        "fail_on_empty_input": bool(config.get("fail_on_empty_input", True)),
        "fail_on_zero_output": bool(config.get("fail_on_zero_output", True)),
    }


def _evaluate_quality_gate(
    dispositions: List[Dict[str, Any]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    counts = Counter(str(item.get("status") or "failed") for item in dispositions)
    reasons = Counter(str(item.get("reason_code") or "unknown") for item in dispositions)
    input_count = len(dispositions)
    accepted_count = counts["accepted"]
    filtered_count = counts["filtered"]
    failed_count = counts["failed"]
    retention_ratio = accepted_count / input_count if input_count else 0.0
    error_ratio = failed_count / input_count if input_count else 0.0
    failures: List[Dict[str, Any]] = []

    if policy["fail_on_empty_input"] and input_count == 0:
        failures.append({"code": "empty_input", "message": "Quality scorer received no files"})
    if policy["fail_on_zero_output"] and input_count > 0 and accepted_count == 0:
        failures.append({"code": "zero_output", "message": "Quality scorer accepted no files"})
    if input_count and retention_ratio < policy["minimum_retention_ratio"]:
        failures.append(
            {
                "code": "retention_ratio_below_minimum",
                "actual": round(retention_ratio, 6),
                "minimum": policy["minimum_retention_ratio"],
            }
        )
    if failed_count > policy["maximum_error_count"]:
        failures.append(
            {
                "code": "error_count_exceeded",
                "actual": failed_count,
                "maximum": policy["maximum_error_count"],
            }
        )
    if input_count and error_ratio > policy["maximum_error_ratio"]:
        failures.append(
            {
                "code": "error_ratio_exceeded",
                "actual": round(error_ratio, 6),
                "maximum": policy["maximum_error_ratio"],
            }
        )

    return {
        "ok": not failures,
        "input_count": input_count,
        "accepted_count": accepted_count,
        "filtered_count": filtered_count,
        "failed_count": failed_count,
        "retention_ratio": round(retention_ratio, 6),
        "error_ratio": round(error_ratio, 6),
        "reason_counts": dict(sorted(reasons.items())),
        "failures": failures,
    }


@register_stage
class QualityScorer(QualityGate):
    name = "quality_scorer"
    description = "Filters login walls, error pages, and short content."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        quality_config = config.get("quality", {})
        if not isinstance(quality_config, dict):
            return ["quality must be a mapping"]
        return _validate_quality_config(quality_config)

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.quality_config
        try:
            policy = _quality_policy(config)
        except ValueError as exc:
            return StageResult.failure(f"Invalid quality configuration: {exc}")

        files: List[Path] = []
        input_mode = "unknown"
        input_root: Path | None = None
        selected_artifacts = []

        markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
        cleaned_artifacts = ctx.find_artifacts(artifact_type="cleaned_html")
        raw_html_dir = ctx.previous_outputs.get("html_dir")
        if cleaned_artifacts:
            input_mode = "cleaned_html"
            selected_artifacts = cleaned_artifacts
            for record in cleaned_artifacts:
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                files.append(path)
        elif raw_html_dir:
            html_dir = Path(raw_html_dir)
            if html_dir.is_dir():
                input_mode = "raw_html"
                input_root = html_dir
                files.extend(html_dir.rglob("*.html"))
        elif markdown_artifacts:
            input_mode = "markdown"
            selected_artifacts = markdown_artifacts
            for record in markdown_artifacts:
                if not record.local_path:
                    continue
                path = Path(record.local_path)
                files.append(path)
        else:
            cleaned_dir_value = ctx.previous_outputs.get("cleaned_dir")
            html_dir_value = ctx.previous_outputs.get("html_dir")
            md_dir_value = ctx.previous_outputs.get("md_dir")
            input_dir = cleaned_dir_value or html_dir_value or md_dir_value
            if not input_dir:
                return StageResult.failure("No content directory in previous outputs")

            input_dir = Path(input_dir)
            if not input_dir.is_dir():
                return StageResult.failure(f"Directory does not exist: {input_dir}")

            input_root = input_dir
            if cleaned_dir_value:
                input_mode = "cleaned_html"
            elif html_dir_value:
                input_mode = "raw_html"
            elif md_dir_value:
                input_mode = "markdown"
            extensions = ("*.md", "*.html", "*.txt")
            for ext in extensions:
                files.extend(input_dir.rglob(ext))

        files = sorted(set(files), key=lambda path: str(path))
        output_name = {
            "raw_html": "accepted_html",
            "cleaned_html": "accepted_cleaned_html",
            "markdown": "accepted_markdown",
        }.get(input_mode, "accepted_content")
        accepted_dir = reset_stage_output_directory(
            ctx.stage_work_dir / output_name,
            ctx.stage_work_dir,
        )

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

        logger.info("Quality gate: scanning %d files from %s", len(files), input_mode)

        artifact_by_path = {
            str(Path(record.local_path).resolve()): record
            for record in selected_artifacts
            if record.local_path
        }
        accepted_path_by_source: Dict[str, str] = {}
        accepted_urls: Set[str] = set()
        dispositions: List[Dict[str, Any]] = []
        content_artifacts = []
        filtered_items: List[str] = []
        used_relative_paths: Set[str] = set()

        for fp in files:
            resolved = str(fp.resolve())
            source_artifact = artifact_by_path.get(resolved)
            source_urls = set(urls_by_path.get(resolved, set()))
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
                relative = (
                    Path(str(relative_value))
                    if relative_value
                    else fp.relative_to(input_root)
                    if input_root
                    else Path(fp.name)
                )
            except ValueError:
                relative = Path(fp.name)
            if relative.as_posix() in used_relative_paths:
                suffix = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:10]
                relative = relative.with_name(f"{relative.stem}-{suffix}{relative.suffix}")
            used_relative_paths.add(relative.as_posix())

            disposition: Dict[str, Any] = {
                "source_path": resolved,
                "source_urls": sorted(source_urls),
                "relative_path": relative.as_posix(),
            }
            try:
                text = fp.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                disposition.update(
                    {
                        "status": "failed",
                        "reason_code": "read_error",
                        "error_type": type(exc).__name__,
                    }
                )
                dispositions.append(disposition)
                continue

            try:
                reason, visible_text = _quality_assessment(
                    text,
                    policy["min_content_length"],
                    policy["detect_login_walls"],
                )
            except Exception as exc:
                disposition.update(
                    {
                        "status": "failed",
                        "reason_code": "assessment_error",
                        "error_type": type(exc).__name__,
                    }
                )
                dispositions.append(disposition)
                continue

            disposition["content_metrics"] = {
                "visible_characters": len(visible_text),
                "visible_words": len(visible_text.split()),
            }
            if reason:
                filtered_items.append(resolved)
                disposition.update({"status": "filtered", "reason_code": reason})
                dispositions.append(disposition)
                continue

            destination = accepted_dir / relative
            try:
                atomic_copy_file(fp, destination)
            except Exception as exc:
                disposition.update(
                    {
                        "status": "failed",
                        "reason_code": "copy_error",
                        "error_type": type(exc).__name__,
                    }
                )
                dispositions.append(disposition)
                continue

            accepted_path_by_source[resolved] = str(destination.resolve())
            accepted_urls.update(source_urls)
            disposition.update(
                {
                    "status": "accepted",
                    "reason_code": "accepted_quality",
                    "output_path": str(destination.resolve()),
                }
            )
            dispositions.append(disposition)
            artifact_type = {
                "raw_html": "quality_accepted_html",
                "cleaned_html": "quality_accepted_cleaned_html",
                "markdown": "quality_accepted_markdown",
            }.get(input_mode, "quality_accepted_content")
            content_artifacts.append(
                ctx.make_artifact(
                    destination,
                    artifact_type=artifact_type,
                    role="quality_accepted_content",
                    metadata={
                        "source_path": resolved,
                        "source_url": sorted(source_urls)[0] if source_urls else "",
                        "source_urls": sorted(source_urls),
                        "relative_path": relative.as_posix(),
                        "input_mode": input_mode,
                        "content_metrics": disposition["content_metrics"],
                    },
                    source_artifact_ids=(
                        [source_artifact.artifact_id] if source_artifact else None
                    ),
                )
            )

        gate = _evaluate_quality_gate(dispositions, policy)
        manifest_path = ctx.stage_work_dir / "quality_manifest.json"

        def project_mapping(mapping: Mapping[str, str]) -> Dict[str, str]:
            projected: Dict[str, str] = {}
            for url, raw_path in sorted(mapping.items()):
                if raw_path.startswith("SKIPPED"):
                    projected[url] = raw_path
                    continue
                resolved_path = str(Path(raw_path).resolve())
                accepted_path = accepted_path_by_source.get(resolved_path)
                if accepted_path:
                    projected[url] = accepted_path
                elif url in accepted_urls and Path(raw_path).is_file():
                    projected[url] = raw_path
            return projected

        outputs = {
            "accepted_dir": str(accepted_dir),
            "passed_count": gate["accepted_count"],
            "filtered_count": gate["filtered_count"],
            "failed_count": gate["failed_count"],
            "filtered_items": filtered_items,
            "quality_manifest_file": str(manifest_path),
        }
        if input_mode == "raw_html":
            outputs.update({"html_dir": str(accepted_dir), "accepted_html_dir": str(accepted_dir)})
        elif input_mode == "cleaned_html":
            outputs.update(
                {
                    "cleaned_dir": str(accepted_dir),
                    "accepted_cleaned_html_dir": str(accepted_dir),
                }
            )
        elif input_mode == "markdown":
            outputs.update(
                {"md_dir": str(accepted_dir), "accepted_markdown_dir": str(accepted_dir)}
            )

        try:
            if mapping_file:
                projected_html_mapping = project_mapping(url_to_html)
                accepted_mapping_file = (
                    ctx.stage_work_dir / "accepted_url_to_html_mapping.json"
                )
                atomic_write_json(accepted_mapping_file, projected_html_mapping)
                outputs["mapping_file"] = str(accepted_mapping_file)
            if md_mapping_file:
                projected_md_mapping = project_mapping(url_to_md)
                accepted_md_mapping_file = (
                    ctx.stage_work_dir / "accepted_url_to_md_mapping.json"
                )
                atomic_write_json(accepted_md_mapping_file, projected_md_mapping)
                outputs["md_mapping_file"] = str(accepted_md_mapping_file)

            for output_key, source_path, payload, filename in (
                (
                    "page_media_file",
                    page_media_file,
                    raw_page_media,
                    "accepted_page_media.json",
                ),
                (
                    "page_images_file",
                    page_images_file,
                    raw_page_images,
                    "accepted_page_images.json",
                ),
                (
                    "page_videos_file",
                    page_videos_file,
                    raw_page_videos,
                    "accepted_page_videos.json",
                ),
            ):
                if not source_path or not isinstance(payload, dict):
                    continue
                projected_payload = {
                    str(url): items
                    for url, items in sorted(payload.items(), key=lambda item: str(item[0]))
                    if str(url) in accepted_urls
                }
                projected_path = ctx.stage_work_dir / filename
                atomic_write_json(projected_path, projected_payload)
                outputs[output_key] = str(projected_path)
        except Exception as exc:
            gate["ok"] = False
            gate["failures"].append(
                {
                    "code": "output_materialization_error",
                    "error_type": type(exc).__name__,
                }
            )

        manifest = {
            "schema_version": 1,
            "stage_id": ctx.stage_id or "score_raw_content",
            "application_status": "completed" if gate["ok"] else "failed",
            "input_mode": input_mode,
            "source_directory": str(input_root.resolve()) if input_root else "",
            "accepted_directory": str(accepted_dir),
            "policy": policy,
            "gate": gate,
            "dispositions": dispositions,
        }
        atomic_write_json(manifest_path, manifest)
        manifest_artifact = ctx.make_artifact(
            manifest_path,
            artifact_type="quality_manifest",
            role="quality_dispositions",
            metadata={
                "input_count": gate["input_count"],
                "accepted_count": gate["accepted_count"],
                "filtered_count": gate["filtered_count"],
                "failed_count": gate["failed_count"],
                "gate_ok": gate["ok"],
            },
        )

        logger.info(
            "Quality gate: accepted=%d filtered=%d failed=%d retention=%.3f",
            gate["accepted_count"],
            gate["filtered_count"],
            gate["failed_count"],
            gate["retention_ratio"],
        )

        metrics = {
            "passed": gate["accepted_count"],
            "filtered": gate["filtered_count"],
            "failed": gate["failed_count"],
            "retention_ratio": gate["retention_ratio"],
            **gate["reason_counts"],
        }
        if not gate["ok"]:
            failure_codes = ", ".join(item["code"] for item in gate["failures"])
            return StageResult.failure(
                f"Raw-content quality gate failed: {failure_codes}",
                checkpoint={"quality_manifest_file": str(manifest_path)},
                outputs=outputs,
                metrics=metrics,
                artifacts=[manifest_artifact],
            )

        return StageResult.success(
            outputs=outputs,
            metrics=metrics,
            checkpoint={"quality_manifest_file": str(manifest_path)},
            artifacts=[*content_artifacts, manifest_artifact],
        )
