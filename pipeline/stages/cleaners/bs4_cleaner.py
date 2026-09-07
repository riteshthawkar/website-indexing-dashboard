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
from pipeline.core.io import (
    atomic_write_json,
    atomic_write_text,
    load_json_safe,
    reset_stage_output_directory,
)
from pipeline.core.registry import register_stage
from pipeline.stages.cleaners.common import (
    CleaningPolicy,
    content_meets_policy,
    evaluate_cleaning_gate,
    normalized_visible_text,
    urls_by_source_path,
    validate_cleaner_policy_config,
    visible_content_metrics,
)

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
NOISE_TOKENS = {
    "ad",
    "ads",
    "advert",
    "advertisement",
    "banner",
    "breadcrumb",
    "comments",
    "consent",
    "cookie",
    "newsletter",
    "promo",
    "rating",
    "related",
    "share",
    "sidebar",
    "social",
    "subscribe",
    "toc",
}
ERROR_PAGE_HEADING = re.compile(
    r"(?i)^\s*(?:(?:error|http)\s*[:\-]?\s*)?"
    r"(?:404(?:\s+(?:error|not\s+found))?|page\s+not\s+found)\s*[.!]?\s*$"
)

DISCLOSURE_CONTAINER_NAMES = {
    "accordioncontent",
    "accordion-content",
    "accordioncollapse",
    "accordion-collapse",
    "disclosurecontent",
    "disclosure-content",
}


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


def _attribute_tokens(tag, name: str) -> set[str]:
    value = _safe_tag_attr(tag, name, [])
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        return set()
    return {
        str(token).strip().casefold()
        for token in value
        if str(token).strip()
    }


def _strip_hidden_style(style: Any) -> str:
    """Remove only CSS declarations that collapse disclosure text."""

    retained: List[str] = []
    for declaration in str(style or "").split(";"):
        declaration = declaration.strip()
        if not declaration:
            continue
        property_name, separator, value = declaration.partition(":")
        normalized_property = property_name.strip().casefold()
        normalized_value = re.sub(r"\s+", "", value.casefold())
        if separator and (
            (normalized_property == "display" and normalized_value == "none")
            or (
                normalized_property == "visibility"
                and normalized_value in {"hidden", "collapse"}
            )
        ):
            continue
        retained.append(declaration)
    return "; ".join(retained)


def _disclosure_region_is_semantic(tag, controlled_ids: set[str]) -> bool:
    attrs = getattr(tag, "attrs", None)
    if not isinstance(attrs, dict):
        return False
    if tag.find_parent(["nav", "header", "footer", "aside"]):
        return False
    text = tag.get_text(" ", strip=True)
    if len(text) < 8 or len(text.split()) < 2:
        return False

    element_id = str(attrs.get("id") or "").strip()
    pc_name = str(attrs.get("data-pc-name") or "").strip().casefold()
    class_tokens = _attribute_tokens(tag, "class")
    inside_accordion_panel = tag.find_parent(
        attrs={"data-pc-name": "accordionpanel"}
    ) is not None
    has_label_contract = bool(
        str(attrs.get("aria-labelledby") or "").strip()
        and str(attrs.get("role") or "").strip().casefold() == "region"
    )
    return bool(
        (element_id and element_id in controlled_ids)
        or pc_name in DISCLOSURE_CONTAINER_NAMES
        or class_tokens.intersection(DISCLOSURE_CONTAINER_NAMES)
        or (inside_accordion_panel and has_label_contract)
    )


def _find_disclosure_trigger(soup, region):
    panel = region.find_parent(attrs={"data-pc-name": "accordionpanel"})
    if panel is None:
        panel = region.find_parent(
            class_=lambda value: value
            and "accordion" in " ".join(
                value if isinstance(value, list) else [value]
            ).casefold()
        )
    if panel is not None:
        trigger = panel.find(attrs={"data-pc-name": "accordionheader"})
        if trigger is None:
            trigger = panel.find(["button", "summary"])
        if trigger is not None:
            return trigger

    region_id = str(_safe_tag_attr(region, "id", "")).strip()
    if region_id:
        for candidate in soup.find_all(attrs={"aria-controls": True}):
            controls = str(_safe_tag_attr(candidate, "aria-controls", "")).split()
            if region_id in controls:
                return candidate
    labelled_by = str(_safe_tag_attr(region, "aria-labelledby", "")).strip()
    if labelled_by:
        return soup.find(id=labelled_by)
    return None


def preserve_accessible_disclosures(soup) -> int:
    """Expose public collapsed panels while leaving arbitrary hidden DOM out."""

    controlled_ids = {
        control_id
        for trigger in soup.find_all(attrs={"aria-controls": True})
        for control_id in str(_safe_tag_attr(trigger, "aria-controls", "")).split()
        if control_id
    }
    preserved = 0
    converted_triggers: set[int] = set()
    for region in list(soup.find_all(True)):
        if not _disclosure_region_is_semantic(region, controlled_ids):
            continue
        attrs = getattr(region, "attrs", None)
        if not isinstance(attrs, dict):
            continue
        attrs.pop("hidden", None)
        if str(attrs.get("aria-hidden", "")).strip().casefold() == "true":
            attrs.pop("aria-hidden", None)
        cleaned_style = _strip_hidden_style(attrs.get("style", ""))
        if cleaned_style:
            attrs["style"] = cleaned_style
        else:
            attrs.pop("style", None)
        attrs["data-indexer-visible-disclosure"] = "true"

        trigger = _find_disclosure_trigger(soup, region)
        if trigger is not None and id(trigger) not in converted_triggers:
            trigger_text = trigger.get_text(" ", strip=True)
            if trigger_text and trigger.name in {"button", "summary"}:
                trigger.name = "h2"
                trigger.attrs.pop("type", None)
                converted_triggers.add(id(trigger))
        preserved += 1
    return preserved


def prepare_html_for_content_extraction(raw: str) -> Tuple[str, int]:
    """Return HTML with semantically public disclosure bodies index-visible."""

    soup = BeautifulSoup(raw, "html.parser")
    preserved = preserve_accessible_disclosures(soup)
    return str(soup), preserved


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

        if str(attrs.get("data-indexer-visible-disclosure", "")).lower() == "true":
            attrs.pop("hidden", None)
            attrs.pop("aria-hidden", None)
            cleaned_style = _strip_hidden_style(attrs.get("style", ""))
            if cleaned_style:
                attrs["style"] = cleaned_style
            else:
                attrs.pop("style", None)
            continue

        if "hidden" in attrs or str(attrs.get("aria-hidden", "")).lower() == "true":
            el.decompose()
            continue

        style = re.sub(r"\s+", "", str(attrs.get("style", "")).lower())
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
        tokens = {
            token
            for token in re.split(r"[^a-z0-9]+", f"{classes} {id_}")
            if token
        }
        return bool(tokens.intersection(NOISE_TOKENS))
    for tag in soup.find_all(is_noise):
        tag.decompose()


def _keep_main_content(soup):
    main = (
        soup.find("main")
        or soup.find(attrs={"role": "main"})
    )
    if main is None:
        articles = soup.find_all("article")
        # A single article is normally the page body. Multiple articles are
        # commonly cards in a listing or program page; selecting the first one
        # would discard substantive sibling content.
        if len(articles) == 1:
            main = articles[0]
    if main:
        return BeautifulSoup(f"<html><body>{main}</body></html>", "html.parser")
    return soup


def _collapse_blanks(text: str) -> str:
    return re.sub(r"\n\s*\n+", "\n", text)


def _is_structural_error_page(soup: BeautifulSoup) -> bool:
    """Detect an error shell from visible structural elements only."""

    probe = BeautifulSoup(str(soup), "html.parser")
    _remove_unwanted_tags(probe, ["script", "style", "noscript", "template"])
    _remove_unwanted_tags(probe, ["nav", "header", "footer", "aside"])
    _remove_hidden(probe)
    primary = (
        probe.find("main")
        or probe.find(attrs={"role": "main"})
        or probe.find("article")
        or probe.body
        or probe
    )
    visible_text = normalized_visible_text(str(primary))
    if len(visible_text) > 1200:
        return False

    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if title and ERROR_PAGE_HEADING.fullmatch(title):
        return True
    for heading in primary.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if ERROR_PAGE_HEADING.fullmatch(heading.get_text(" ", strip=True)):
            return True
    return len(visible_text) <= 240 and bool(ERROR_PAGE_HEADING.fullmatch(visible_text))


def clean_html_content(raw: str, preserve_media: bool = False) -> Tuple[str, Optional[str]]:
    """Clean HTML content and return (status, cleaned_html)."""
    soup = BeautifulSoup(raw, "html.parser")
    preserve_accessible_disclosures(soup)
    if _is_structural_error_page(soup):
        return "removed", None

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

    atomic_write_text(file_path, cleaned_html)
    return status


# ── Stage ────────────────────────────────────────────────────────────

@register_stage
class BS4Cleaner(CleanerStage):
    name = "bs4"
    description = "BeautifulSoup-based HTML cleaner — strips boilerplate, noise, and empty pages."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        cleaner_config = config.get("cleaner", {})
        if not isinstance(cleaner_config, dict):
            return ["cleaner must be a mapping"]
        return validate_cleaner_policy_config(cleaner_config)

    async def execute(self, ctx: StageContext) -> StageResult:
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

        recursive = config.get("recursive", True)
        preserve_media = config.get(
            "preserve_embedded_media",
            ctx.crawler_config.get("extract_images", True) or ctx.crawler_config.get("extract_videos", True),
        )
        cleaned_dir = reset_stage_output_directory(
            ctx.stage_work_dir / "cleaned_html",
            ctx.stage_work_dir,
        )
        pattern = "**/*.html" if recursive else "*.html"
        accepted_html_artifacts = ctx.find_artifacts(artifact_type="quality_accepted_html")
        files = (
            [Path(record.local_path) for record in accepted_html_artifacts if record.local_path]
            if accepted_html_artifacts
            else list(html_dir.glob(pattern))
        )
        files = sorted(set(files), key=lambda path: str(path))
        logger.info("BS4 cleaner found %d HTML files in %s", len(files), html_dir)

        content_artifacts = []
        url_mapping = load_json_safe(ctx.previous_outputs.get("mapping_file"), {}) or {}
        urls_by_path = urls_by_source_path(url_mapping if isinstance(url_mapping, dict) else {})
        artifact_by_path = {
            str(Path(record.local_path).resolve()): record
            for record in accepted_html_artifacts
            if record.local_path
        }
        dispositions: List[Dict[str, Any]] = []

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
            except Exception as exc:
                logger.error("Read error %s: %s", fp, exc)
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
                status, cleaned_html = clean_html_content(raw, preserve_media=preserve_media)
            except Exception as exc:
                logger.error("Cleaning error %s: %s", fp, exc)
                disposition.update(
                    {
                        "status": "failed",
                        "reason_code": "cleaning_error",
                        "error_type": type(exc).__name__,
                    }
                )
                dispositions.append(disposition)
                continue

            metrics = visible_content_metrics(cleaned_html or "")
            disposition["content_metrics"] = metrics.to_dict()
            if status != "cleaned" or cleaned_html is None:
                disposition.update(
                    {
                        "status": "filtered",
                        "reason_code": (
                            "structural_error_page" if status == "removed" else "empty_content"
                        ),
                    }
                )
                dispositions.append(disposition)
                continue
            if not content_meets_policy(metrics, policy):
                disposition.update(
                    {
                        "status": "filtered",
                        "reason_code": "insufficient_visible_content",
                    }
                )
                dispositions.append(disposition)
                continue

            out_path = cleaned_dir / relative
            try:
                atomic_write_text(out_path, cleaned_html)
            except Exception as exc:
                logger.error("Write error %s: %s", out_path, exc)
                disposition.update(
                    {
                        "status": "failed",
                        "reason_code": "write_error",
                        "error_type": type(exc).__name__,
                    }
                )
                dispositions.append(disposition)
                continue

            disposition.update(
                {
                    "status": "accepted",
                    "reason_code": "accepted_bs4",
                    "output_path": str(out_path.resolve()),
                }
            )
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
                        "content_metrics": metrics.to_dict(),
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
            "BS4 cleaner done: accepted=%d filtered=%d failed=%d retention=%.3f",
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
