"""Deterministic, user-facing titles for corpus and retrieval records."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit


_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_OPAQUE_TITLE_RE = re.compile(r"^[a-f0-9]{16,64}$", flags=re.IGNORECASE)
_TRAILING_CRAWL_DIGEST_RE = re.compile(
    r"[-_][a-f0-9]{12,64}$", flags=re.IGNORECASE
)
_DOCUMENT_EXTENSION_RE = re.compile(
    r"\.(?:html?|pdf|docx?|xlsx?|pptx?|csv|tsv|rtf)$", flags=re.IGNORECASE
)
_ACRONYM_DISPLAY = {
    "ad": "AD",
    "ai": "AI",
    "aiq": "AIQ",
    "en": "EN",
    "faq": "FAQ",
    "hci": "HCI",
    "hpp": "HPP",
    "ifm": "IFM",
    "llm": "LLM",
    "mbzuai": "MBZUAI",
    "msc": "MSc",
    "nlp": "NLP",
    "ocr": "OCR",
    "phd": "PhD",
    "toefl": "TOEFL",
    "uae": "UAE",
    "ugrip": "UGRIP",
}


def clean_title_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFC", str(value or ""))).strip()


def looks_like_opaque_title(value: Any) -> bool:
    """Return whether a title is a truncated/full hexadecimal artifact ID."""

    return bool(_OPAQUE_TITLE_RE.fullmatch(clean_title_text(value)))


def _humanize_identifier(value: Any) -> str:
    raw = clean_title_text(value)
    raw = _DOCUMENT_EXTENSION_RE.sub("", raw)
    raw = _TRAILING_CRAWL_DIGEST_RE.sub("", raw).strip("-_. ")
    raw = re.sub(r"[-_]+", " ", raw)
    raw = _SPACE_RE.sub(" ", raw).strip()
    if not raw or looks_like_opaque_title(raw):
        return ""

    words: list[str] = []
    for word in raw.split():
        normalized = word.casefold().strip(".")
        if normalized in _ACRONYM_DISPLAY:
            words.append(_ACRONYM_DISPLAY[normalized])
        elif re.fullmatch(r"v\d+", normalized):
            words.append(word.upper())
        elif word.isupper():
            words.append(word.capitalize())
        else:
            words.append(word[:1].upper() + word[1:])
    return " ".join(words).strip()


def title_from_source_url(value: Any) -> str:
    raw = clean_title_text(value)
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except (TypeError, ValueError):
        return ""
    host = (parsed.hostname or "").casefold().rstrip(".")
    if not host:
        return ""
    parts = [
        unquote(part).strip()
        for part in (parsed.path or "").split("/")
        if unquote(part).strip()
        and unquote(part).strip().casefold() not in {"en", "ar"}
        and not re.fullmatch(r"20\d{2}|\d{1,2}", unquote(part).strip())
    ]
    if parts:
        title = _humanize_identifier(parts[-1])
        if title:
            return title

    if host == "ifm.ai" or host.endswith(".ifm.ai"):
        subdomain = "" if host == "ifm.ai" else host.removesuffix(".ifm.ai").strip(".")
        suffix = _humanize_identifier(subdomain)
        return f"IFM {suffix}".strip()
    if host == "mbzuai.gitbook.io":
        return "MBZUAI Documentation"
    if host == "mbzuai.ac.ae" or host.endswith(".mbzuai.ac.ae"):
        subdomain = (
            ""
            if host == "mbzuai.ac.ae"
            else host.removesuffix(".mbzuai.ac.ae").strip(".")
        )
        if subdomain in {"", "www", "preprod"}:
            return "MBZUAI"
        return f"MBZUAI {_humanize_identifier(subdomain)}".strip()
    return _humanize_identifier(host.split(".", 1)[0])


def title_from_source_file(value: Any) -> str:
    raw = clean_title_text(value)
    if not raw:
        return ""
    return _humanize_identifier(unquote(Path(raw).name))


def resolve_document_title(
    title: Any = "",
    *,
    page_titles: Iterable[Any] = (),
    source_url: Any = "",
    source_file: Any = "",
    source_locator: Mapping[str, Any] | None = None,
    fallback: str = "MBZUAI document",
) -> str:
    """Resolve a stable readable title without exposing internal filenames."""

    direct = clean_title_text(title)
    if direct and not looks_like_opaque_title(direct):
        return direct
    for page_title in page_titles:
        candidate = clean_title_text(page_title)
        if candidate and not looks_like_opaque_title(candidate):
            return candidate
    from_url = title_from_source_url(source_url)
    if from_url:
        return from_url
    from_file = title_from_source_file(source_file)
    if from_file:
        return from_file
    locator = source_locator if isinstance(source_locator, Mapping) else {}
    locator_kind = clean_title_text(locator.get("kind")).casefold()
    locator_value = locator.get("value")
    from_locator = (
        title_from_source_url(locator_value)
        if locator_kind == "url"
        else title_from_source_file(locator_value)
    )
    if from_locator:
        return from_locator
    return clean_title_text(fallback) or "MBZUAI document"
