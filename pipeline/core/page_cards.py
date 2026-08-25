"""Deterministic, evidence-backed webpage representation extraction.

This module deliberately does not call an LLM.  It turns the frozen crawl
artifacts into concise Page Cards and typed actions while retaining a locator
and an extractive evidence excerpt for every semantic field.  A later stage
may enrich these records, but it must not replace their source evidence.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, Tag

from pipeline.core.representation_v2 import (
    REPRESENTATION_V2_SCHEMA_VERSION,
    stable_representation_id,
)


_SPACE_RE = re.compile(r"\s+", flags=re.UNICODE)
_MULTISLASH_RE = re.compile(r"/{2,}")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_DOCUMENT_EXTENSION_RE = re.compile(
    r"\.(?:pdf|docx?|xlsx?|pptx?|csv|tsv|zip|rtf)(?:$|[?#])",
    flags=re.IGNORECASE,
)
_ASSET_EXTENSION_RE = re.compile(
    r"\.(?:avif|bmp|css|eot|gif|ico|jpe?g|js|json|mjs|mp3|mp4|ogg|png|svg|"
    r"ttf|webm|webp|woff2?)(?:$|[?#])",
    flags=re.IGNORECASE,
)
_BOILERPLATE_TOKENS = {
    "breadcrumb",
    "cookie",
    "footer",
    "header",
    "language-switcher",
    "mega-menu",
    "megamenu",
    "menu-item",
    "mobile-menu",
    "modal",
    "navbar",
    "navigation",
    "newsletter-popup",
    "offcanvas",
    "popup",
    "search-modal",
    "sidebar",
    "site-footer",
    "site-header",
    "social-links",
    "wpml-ls",
}
_SOCIAL_HOSTS = {
    "facebook.com",
    "instagram.com",
    "linkedin.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "youtube.com",
    "youtu.be",
}
_GENERIC_ACTION_LABELS = {
    "arrow",
    "click here",
    "discover",
    "discover more",
    "explore",
    "here",
    "learn more",
    "more",
    "read more",
    "view",
    "view all",
    "view more",
    "اعرف المزيد",
    "اكتشف المزيد",
    "المزيد",
    "اقرأ المزيد",
}
_LANGUAGE_LABELS = {
    "ar",
    "arabic",
    "en",
    "english",
    "العربية",
    "الإنجليزية",
}
_GENERIC_DESCRIPTION_MARKERS = (
    "leading ai education and research institution",
    "graduate research university dedicated to advancing ai as a global force",
)

_AUDIENCE_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (
        "prospective_students",
        (
            "prospective student",
            "future student",
            "study at mbzuai",
            "الطلاب المحتملين",
            "الطلبة المحتملين",
        ),
    ),
    (
        "applicants",
        (
            "applicant",
            "application",
            "apply now",
            "admission",
            "القبول",
            "التقديم",
            "مقدم الطلب",
        ),
    ),
    (
        "students",
        (
            "student life",
            "student services",
            "students",
            "student",
            "الحياة الطلابية",
            "الطلاب",
            "الطلبة",
            "الطالب",
        ),
    ),
    (
        "faculty",
        (
            "faculty",
            "professor",
            "academic staff",
            "أعضاء هيئة التدريس",
            "هيئة التدريس",
            "أستاذ",
        ),
    ),
    (
        "researchers",
        (
            "researcher",
            "research scientist",
            "research community",
            "الباحثين",
            "الباحث",
            "مجتمع البحث",
        ),
    ),
    ("alumni", ("alumni", "graduate community", "الخريجين", "الخريجون")),
    (
        "industry_partners",
        (
            "industry partner",
            "partnership",
            "collaborate with",
            "الشركاء الصناعيين",
            "الشراكة",
            "تعاون معنا",
        ),
    ),
    (
        "job_candidates",
        (
            "careers",
            "job opening",
            "join our team",
            "vacancy",
            "الوظائف",
            "الشواغر",
            "انضم إلى فريق",
        ),
    ),
)


def clean_text(value: Any, *, maximum_chars: int = 0) -> str:
    """Normalize Unicode and whitespace without paraphrasing source text."""

    normalized = unicodedata.normalize("NFC", str(value or ""))
    normalized = _SPACE_RE.sub(" ", normalized).strip()
    if maximum_chars > 0 and len(normalized) > maximum_chars:
        return normalized[: max(1, maximum_chars - 1)].rstrip() + "…"
    return normalized


def _bound_text(value: Any, maximum_chars: int) -> str:
    normalized = clean_text(value)
    if len(normalized) <= maximum_chars:
        return normalized
    return normalized[: max(1, maximum_chars - 1)].rstrip() + "…"


def normalize_url(
    value: Any,
    *,
    base_url: str = "",
    keep_fragment: bool = False,
    sort_query: bool = True,
) -> str:
    """Resolve and deterministically normalize a web/action URL."""

    raw = clean_text(value)
    if not raw:
        return ""
    lowered = raw.lower()
    if lowered.startswith(("mailto:", "tel:")):
        scheme, remainder = raw.split(":", 1)
        return f"{scheme.lower()}:{remainder.strip()}"
    if raw.startswith("#"):
        return raw if keep_fragment else ""
    try:
        resolved = urljoin(base_url, raw) if base_url else raw
        parsed = urlsplit(resolved)
    except (TypeError, ValueError):
        return ""
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    host = (parsed.hostname or "").lower()
    if not host:
        return ""
    port = parsed.port
    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = _MULTISLASH_RE.sub("/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    query = parsed.query
    if query and sort_query:
        query = urlencode(sorted(parse_qsl(query, keep_blank_values=True)), doseq=True)
    fragment = parsed.fragment if keep_fragment else ""
    return urlunsplit((scheme, netloc, path, query, fragment))


def _normalized_comparison_url(value: Any) -> str:
    return normalize_url(value, keep_fragment=False)


def _tag_text(tag: Any, *, maximum_chars: int = 0) -> str:
    if not isinstance(tag, Tag):
        return ""
    value = tag.get_text(" ", strip=True)
    return _bound_text(value, maximum_chars) if maximum_chars else clean_text(value)


def _element_tokens(tag: Tag) -> set[str]:
    values: List[str] = [str(tag.get("id") or "")]
    classes = tag.get("class") or []
    if isinstance(classes, str):
        values.append(classes)
    else:
        values.extend(str(value) for value in classes)
    tokens: set[str] = set()
    for value in values:
        tokens.update(
            token
            for token in re.split(r"[^a-z0-9_-]+", value.lower())
            if token
        )
    return tokens


def _has_boilerplate_ancestor(tag: Tag) -> bool:
    for ancestor in [tag, *list(tag.parents)]:
        if not isinstance(ancestor, Tag):
            continue
        if ancestor.name in {"header", "footer", "nav", "aside"}:
            return True
        tokens = _element_tokens(ancestor)
        if tokens & _BOILERPLATE_TOKENS:
            return True
        joined = " ".join(tokens)
        if any(marker in joined for marker in ("cookie", "popup", "offcanvas", "mega-menu")):
            return True
    return False


def _is_hidden(tag: Tag) -> bool:
    for ancestor in [tag, *list(tag.parents)]:
        if not isinstance(ancestor, Tag):
            continue
        if ancestor.has_attr("hidden") or str(ancestor.get("aria-hidden") or "").lower() == "true":
            return True
        style = str(ancestor.get("style") or "").replace(" ", "").lower()
        if "display:none" in style or "visibility:hidden" in style:
            return True
    return False


def html_locator(tag: Tag) -> str:
    """Build a stable, human-auditable CSS-like structural locator."""

    parts: List[str] = []
    current: Any = tag
    while isinstance(current, Tag) and current.name != "[document]":
        name = str(current.name or "node").lower()
        element_id = clean_text(current.get("id"))
        if element_id:
            safe_id = re.sub(r"[^\w:-]+", "-", element_id, flags=re.UNICODE)
            parts.append(f"{name}#{safe_id}")
            break
        parent = current.parent
        if isinstance(parent, Tag):
            siblings = [child for child in parent.find_all(name, recursive=False)]
            if len(siblings) > 1:
                try:
                    position = siblings.index(current) + 1
                except ValueError:
                    position = 1
                name = f"{name}:nth-of-type({position})"
        parts.append(name)
        current = parent
        if len(parts) >= 10:
            break
    return " > ".join(reversed(parts)) or str(tag.name or "node")


def _dom_region(tag: Tag) -> str:
    for ancestor in [tag, *list(tag.parents)]:
        if not isinstance(ancestor, Tag):
            continue
        name = str(ancestor.name or "").lower()
        if name == "main" or str(ancestor.get("role") or "").lower() == "main":
            return "main"
        if name == "article":
            return "article"
        if name == "nav" or str(ancestor.get("role") or "").lower() == "navigation":
            return "navigation"
        if name in {"header", "footer", "aside"}:
            return name
    return "body"


def _nearest_heading(tag: Tag) -> Tag | None:
    for ancestor in [tag, *list(tag.parents)]:
        if not isinstance(ancestor, Tag):
            continue
        heading = ancestor.find(re.compile(r"^h[1-6]$"))
        if isinstance(heading, Tag) and heading is not tag and _tag_text(heading):
            return heading
        previous = ancestor.find_previous(re.compile(r"^h[1-6]$"))
        if isinstance(previous, Tag) and not _has_boilerplate_ancestor(previous):
            return previous
        if ancestor.name in {"main", "article", "body"}:
            break
    return None


def _card_context(tag: Tag) -> str:
    for ancestor in [tag, *list(tag.parents)]:
        if not isinstance(ancestor, Tag):
            continue
        if ancestor.name not in {"a", "li", "article", "div", "section", "td"}:
            continue
        heading = ancestor.find(re.compile(r"^h[1-6]$"))
        heading_text = _tag_text(heading, maximum_chars=180)
        if heading_text:
            return heading_text
        for selector in ("[aria-label]", "[data-title]", ".title", ".card-title"):
            candidate = ancestor.select_one(selector)
            if isinstance(candidate, Tag):
                value = clean_text(
                    candidate.get("aria-label")
                    or candidate.get("data-title")
                    or _tag_text(candidate),
                    maximum_chars=180,
                )
                if value:
                    return value
        if ancestor.name in {"article", "section"}:
            break
    return ""


def _metadata_scalar(metadata: Mapping[str, Any], key: str) -> str:
    value = metadata.get(key)
    if isinstance(value, list):
        return next((clean_text(item) for item in value if clean_text(item)), "")
    return clean_text(value)


def _metadata_description(metadata: Mapping[str, Any]) -> str:
    raw = metadata.get("description")
    candidates = list(raw) if isinstance(raw, list) else [raw]
    meta_tags = metadata.get("meta_tags") if isinstance(metadata.get("meta_tags"), Mapping) else {}
    candidates.extend(
        [meta_tags.get("description"), meta_tags.get("og:description")]
    )
    flattened: List[Any] = []
    for value in candidates:
        if isinstance(value, list):
            flattened.extend(value)
        else:
            flattened.append(value)
    for value in flattened:
        text = clean_text(value)
        if len(text) >= 35:
            return _bound_text(text, 600)
    return ""


def _is_generic_description(text: str) -> bool:
    lowered = text.casefold()
    return any(marker in lowered for marker in _GENERIC_DESCRIPTION_MARKERS)


def _primary_content_root(soup: BeautifulSoup) -> Tag | None:
    for selector in ("main", "[role='main']", "article"):
        node = soup.select_one(selector)
        if isinstance(node, Tag):
            return node
    return soup.body if isinstance(soup.body, Tag) else None


def _first_content_paragraph(soup: BeautifulSoup) -> Tuple[str, Tag | None]:
    root = _primary_content_root(soup)
    if not isinstance(root, Tag):
        return "", None
    # Prefer actual paragraphs across the primary region.  A visually styled
    # div near the top of a page is often a breadcrumb/menu flattened into one
    # misleading sentence (especially on the careers subdomain).
    for tag_name in ("p", "div"):
        for tag in root.find_all(tag_name):
            if not isinstance(tag, Tag) or _is_hidden(tag) or _has_boilerplate_ancestor(tag):
                continue
            # A div that contains nested content blocks is a container, not a
            # self-contained extractive purpose statement.
            if tag.name == "div" and tag.find(
                ["p", "section", "article", "ul", "ol", "table", "nav"]
            ):
                continue
            text = _tag_text(tag)
            if 45 <= len(text) <= 1200:
                return _bound_text(text, 600), tag
    return "", None


class _EvidenceBuilder:
    def __init__(self, page_card_id: str) -> None:
        self.page_card_id = page_card_id
        self.items: List[Dict[str, Any]] = []
        self._by_key: Dict[Tuple[str, str, str, str], str] = {}

    def add(
        self,
        *,
        source_kind: str,
        locator: str,
        excerpt: Any,
        attribute: str = "",
        extraction_method: str = "",
    ) -> str:
        text = _bound_text(excerpt, 800)
        locator = clean_text(locator)
        if not text or not locator:
            raise ValueError("Representation evidence requires a locator and non-empty excerpt")
        key = (source_kind, locator, attribute, text)
        existing = self._by_key.get(key)
        if existing:
            return existing
        evidence_id = stable_representation_id(
            "page-evidence", self.page_card_id, source_kind, locator, attribute, text
        )
        item: Dict[str, Any] = {
            "evidence_id": evidence_id,
            "source_kind": source_kind,
            "locator": locator,
            "excerpt": text,
        }
        if attribute:
            item["attribute"] = attribute
        if extraction_method:
            item["extraction_method"] = extraction_method
        self._by_key[key] = evidence_id
        self.items.append(item)
        return evidence_id


def _contains_pattern(text: str, pattern: str) -> bool:
    folded_text = text.casefold()
    folded_pattern = pattern.casefold()
    if any("\u0600" <= char <= "\u06ff" for char in folded_pattern):
        normalized_text = _ARABIC_DIACRITICS_RE.sub("", folded_text)
        normalized_pattern = _ARABIC_DIACRITICS_RE.sub("", folded_pattern)
        if normalized_pattern in normalized_text:
            return True
        pattern_core = normalized_pattern.removeprefix("ال")
        for token in re.findall(r"[\u0600-\u06ff]+", normalized_text):
            token_core = token
            # Arabic conjunction/preposition clitics attach directly to words;
            # the definite article's alef can be elided after lam (للطلاب).
            for _ in range(3):
                if len(token_core) > 3 and token_core[0] in "وفبكل":
                    token_core = token_core[1:]
                else:
                    break
            token_core = token_core.removeprefix("ال")
            if token_core == pattern_core:
                return True
        return False
    return bool(
        re.search(
            rf"(?<![a-z0-9]){re.escape(folded_pattern)}(?![a-z0-9])",
            folded_text,
        )
    )


def _semantic_audiences(
    evidence_texts: Sequence[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    audiences: List[Dict[str, Any]] = []
    for label, patterns in _AUDIENCE_PATTERNS:
        evidence_ids: List[str] = []
        for text, evidence_id in evidence_texts:
            if any(_contains_pattern(text, pattern) for pattern in patterns):
                evidence_ids.append(evidence_id)
        if evidence_ids:
            audiences.append(
                {
                    "label": label,
                    "method": "explicit_keyword_match",
                    "evidence_ids": sorted(set(evidence_ids)),
                }
            )
    return audiences


def _label_for_anchor(tag: Tag) -> Tuple[str, str, str]:
    aria = clean_text(tag.get("aria-label"))
    title = clean_text(tag.get("title"))
    image = tag.find("img")
    image_alt = clean_text(image.get("alt")) if isinstance(image, Tag) else ""
    visible = _tag_text(tag)
    if visible and len(visible) <= 180:
        label, source = visible, "visible_text"
    elif aria:
        label, source = aria, "aria-label"
    elif title:
        label, source = title, "title"
    elif image_alt:
        label, source = image_alt, "image_alt"
    elif visible:
        heading = tag.find(re.compile(r"^h[1-6]$"))
        heading_text = _tag_text(heading, maximum_chars=180)
        label, source = (heading_text or _bound_text(visible, 180)), "contained_text"
    else:
        label, source = "", ""
    context = _card_context(tag)
    if clean_text(label).casefold() in _GENERIC_ACTION_LABELS and context:
        return label, context, source
    return label, context if context != label else "", source


def _action_target(raw_target: Any, source_url: str) -> Tuple[str, str]:
    raw = clean_text(raw_target)
    if not raw:
        return "", ""
    lowered = raw.casefold()
    if lowered.startswith(("javascript:", "data:", "vbscript:")):
        return "", ""
    if raw.startswith("#"):
        return raw, raw
    if lowered.startswith(("mailto:", "tel:")):
        normalized = normalize_url(raw)
        return raw, normalized
    absolute = normalize_url(raw, base_url=source_url, keep_fragment=True)
    if not absolute:
        return "", ""
    canonical = normalize_url(absolute, keep_fragment=True)
    return absolute, canonical


def _host_is_official(host: str, official_hosts: set[str]) -> bool:
    host = host.lower().removeprefix("www.")
    normalized = {value.lower().removeprefix("www.") for value in official_hosts}
    return (
        host in normalized
        or host == "mbzuai.ac.ae"
        or host.endswith(".mbzuai.ac.ae")
        or host == "ifm.ai"
        or host.endswith(".ifm.ai")
    )


def _is_social_target(target_url: str) -> bool:
    try:
        host = (urlsplit(target_url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return False
    return any(host == value or host.endswith(f".{value}") for value in _SOCIAL_HOSTS)


def _classify_action(
    *,
    label: str,
    context_label: str,
    target_url: str,
    is_form: bool,
    has_search_input: bool = False,
    download_attribute: bool = False,
) -> str:
    target_lower = target_url.casefold()
    material = f"{label} {context_label} {target_lower}".casefold()
    if target_lower.startswith("mailto:"):
        return "email"
    if target_lower.startswith("tel:"):
        return "telephone"
    if target_url.startswith("#"):
        return "fragment_navigation"
    if download_attribute or _DOCUMENT_EXTENSION_RE.search(target_url):
        return "download"
    if is_form:
        if has_search_input or re.search(r"(?:\bsearch\b|بحث)", material):
            return "search"
        if re.search(r"(?:\blog[ -]?in\b|\bsign[ -]?in\b|تسجيل الدخول)", material):
            return "login"
        if re.search(r"(?:\bapply\b|\bapplication\b|قد[ّ]?م|التقديم)", material):
            return "apply"
        if re.search(r"(?:\bregister\b|\bregistration\b|التسجيل)", material):
            return "register"
        if re.search(r"(?:\bcontact\b|\bget in touch\b|\bcollaborate\b|تواصل|اتصل)", material):
            return "contact"
        return "submit_form"
    if re.search(r"(?:\blog[ -]?in\b|\bsign[ -]?in\b|/login(?:/|$)|/signin(?:/|$)|تسجيل الدخول)", material):
        return "login"
    if re.search(r"(?:\bapply now\b|\bsubmit (?:an? )?application\b|/apply(?:/|$)|application[-_/ ]form|قد[ّ]?م الآن|التقديم الآن)", material):
        return "apply"
    if re.search(r"(?:\bregister(?: now)?\b|\bevent registration\b|/register(?:/|$)|التسجيل)", material):
        return "register"
    if re.search(r"(?:\bcontact(?: us)?\b|\bget in touch\b|\bcollaborate(?: with us)?\b|/contact(?:/|$)|تواصل معنا|اتصل بنا)", material):
        return "contact"
    return "navigate"


def _target_kind(
    *,
    target_url: str,
    action_type: str,
    official_hosts: set[str],
    is_form: bool,
) -> Tuple[str, bool]:
    lowered = target_url.casefold()
    if lowered.startswith("mailto:"):
        return "email", False
    if lowered.startswith("tel:"):
        return "telephone", False
    if target_url.startswith("#"):
        return "fragment", True
    if action_type == "download":
        try:
            host = (urlsplit(target_url).hostname or "").lower()
        except ValueError:
            host = ""
        return "download", _host_is_official(host, official_hosts)
    try:
        host = (urlsplit(target_url).hostname or "").lower()
    except ValueError:
        host = ""
    official = _host_is_official(host, official_hosts)
    if is_form:
        return "form_endpoint", official
    return ("official_page" if official else "external_page"), official


def _action_evidence(
    *,
    page_card_id: str,
    locator: str,
    label: str,
    label_source: str,
    target_evidence: str,
    target_attribute: str,
    context_label: str,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    label_excerpt = label or context_label
    label_id = stable_representation_id(
        "action-evidence", page_card_id, locator, label_source, label_excerpt
    )
    items.append(
        {
            "evidence_id": label_id,
            "source_kind": (
                "html_text"
                if label_source in {"visible_text", "contained_text", "nearest_context"}
                else "html_attribute"
            ),
            "locator": locator,
            "excerpt": _bound_text(label_excerpt, 400),
            **(
                {"attribute": label_source}
                if label_source
                not in {"visible_text", "contained_text", "nearest_context"}
                else {}
            ),
            "extraction_method": "extractive_action_label",
        }
    )
    target_id = stable_representation_id(
        "action-evidence", page_card_id, locator, target_attribute, target_evidence
    )
    items.append(
        {
            "evidence_id": target_id,
            "source_kind": (
                "url" if target_attribute == "form_default_current_url" else "html_attribute"
            ),
            "locator": locator,
            "excerpt": _bound_text(target_evidence, 800),
            **(
                {"attribute": target_attribute}
                if target_attribute != "form_default_current_url"
                else {}
            ),
            "extraction_method": "resolved_url_attribute",
        }
    )
    if context_label and context_label != label:
        context_id = stable_representation_id(
            "action-evidence", page_card_id, locator, "context", context_label
        )
        items.append(
            {
                "evidence_id": context_id,
                "source_kind": "html_text",
                "locator": locator,
                "excerpt": _bound_text(context_label, 400),
                "extraction_method": "nearest_card_or_section_heading",
            }
        )
    return items


def _extract_anchor_candidate(
    tag: Tag,
    *,
    page_card_id: str,
    source_url: str,
    official_hosts: set[str],
    section_by_locator: Mapping[str, Mapping[str, Any]],
    order: int,
) -> Dict[str, Any] | None:
    raw_target = tag.get("href")
    target_url, canonical_target = _action_target(raw_target, source_url)
    if not target_url or not canonical_target:
        return None
    if _ASSET_EXTENSION_RE.search(target_url) and not _DOCUMENT_EXTENSION_RE.search(target_url):
        return None
    if _is_social_target(target_url):
        return None
    label, context_label, label_source = _label_for_anchor(tag)
    label = _bound_text(label, 180)
    context_label = _bound_text(context_label, 220)
    if not label:
        label = context_label
        context_label = ""
        label_source = "nearest_context"
    if not label:
        return None
    if label.casefold() in _LANGUAGE_LABELS:
        return None

    locator = html_locator(tag)
    region = _dom_region(tag)
    heading = _nearest_heading(tag)
    heading_locator = html_locator(heading) if isinstance(heading, Tag) else ""
    section = section_by_locator.get(heading_locator, {})
    source_section_heading = clean_text(section.get("heading")) or _tag_text(
        heading, maximum_chars=220
    )
    source_section_id = clean_text(section.get("section_id")) or None
    action_type = _classify_action(
        label=label,
        context_label=context_label,
        target_url=target_url,
        is_form=False,
        download_attribute=tag.has_attr("download"),
    )
    target_kind, official = _target_kind(
        target_url=target_url,
        action_type=action_type,
        official_hosts=official_hosts,
        is_form=False,
    )
    element_role = clean_text(tag.get("role")) or "link"
    signature_context = (
        context_label if clean_text(label).casefold() in _GENERIC_ACTION_LABELS else ""
    )
    signature = "\0".join(
        (
            action_type,
            canonical_target.casefold(),
            clean_text(label).casefold(),
            clean_text(signature_context).casefold(),
        )
    )
    return {
        "_order": order,
        "_signature": signature,
        "label": label,
        "context_label": context_label,
        "label_source": label_source,
        "action_type": action_type,
        "target_url": target_url,
        "canonical_target_url": canonical_target,
        "target_kind": target_kind,
        "official_target": official,
        "element_role": element_role,
        "dom_region": region,
        "html_locator": locator,
        "source_section_id": source_section_id,
        "source_section_heading": source_section_heading,
        "opens_new_window": clean_text(tag.get("target")).casefold() == "_blank",
        "authentication_requirement": (
            "explicit" if action_type == "login" else "not_indicated"
        ),
        "evidence": _action_evidence(
            page_card_id=page_card_id,
            locator=locator,
            label=label,
            label_source=label_source,
            target_evidence=clean_text(raw_target),
            target_attribute="href",
            context_label=context_label,
        ),
    }


def _extract_form_candidate(
    tag: Tag,
    *,
    page_card_id: str,
    source_url: str,
    official_hosts: set[str],
    section_by_locator: Mapping[str, Mapping[str, Any]],
    order: int,
) -> Dict[str, Any] | None:
    explicit_action = clean_text(tag.get("action"))
    raw_action = explicit_action or source_url
    target_url, canonical_target = _action_target(raw_action, source_url)
    if not target_url or not canonical_target:
        return None
    locator = html_locator(tag)
    heading = _nearest_heading(tag)
    heading_locator = html_locator(heading) if isinstance(heading, Tag) else ""
    section = section_by_locator.get(heading_locator, {})
    source_section_heading = clean_text(section.get("heading")) or _tag_text(
        heading, maximum_chars=220
    )
    source_section_id = clean_text(section.get("section_id")) or None
    search_input = tag.find(
        "input", attrs={"type": re.compile(r"^(?:search)$", re.IGNORECASE)}
    )
    if not isinstance(search_input, Tag):
        search_input = tag.find(
            "input", attrs={"name": re.compile(r"(?:^s$|search|query)", re.IGNORECASE)}
        )
    submit = tag.find(["button", "input"], attrs={"type": re.compile(r"^(?:submit|search)$", re.IGNORECASE)})
    label = ""
    label_source = ""
    if isinstance(submit, Tag):
        visible_submit = _tag_text(submit)
        if visible_submit:
            label, label_source = visible_submit, "visible_text"
        elif clean_text(submit.get("value")):
            label, label_source = clean_text(submit.get("value")), "value"
        elif clean_text(submit.get("aria-label")):
            label, label_source = clean_text(submit.get("aria-label")), "aria-label"
    context_label = _card_context(tag) or source_section_heading
    if not label:
        for element, attribute in (
            (tag, "aria-label"),
            (tag, "title"),
            (search_input, "aria-label"),
            (search_input, "placeholder"),
            (search_input, "name"),
            (search_input, "type"),
        ):
            if isinstance(element, Tag) and clean_text(element.get(attribute)):
                label = clean_text(element.get(attribute))
                label_source = attribute
                break
    if not label and context_label:
        label, label_source = context_label, "nearest_context"
        context_label = ""
    if not label:
        return None
    action_type = _classify_action(
        label=label,
        context_label=context_label,
        target_url=target_url,
        is_form=True,
        has_search_input=isinstance(search_input, Tag),
    )
    target_kind, official = _target_kind(
        target_url=target_url,
        action_type=action_type,
        official_hosts=official_hosts,
        is_form=True,
    )
    method = clean_text(tag.get("method") or "get").upper()
    signature = "\0".join(
        (
            action_type,
            canonical_target.casefold(),
            clean_text(label).casefold(),
            clean_text(context_label).casefold(),
            method,
        )
    )
    return {
        "_order": order,
        "_signature": signature,
        "label": _bound_text(label, 180),
        "context_label": _bound_text(context_label, 220),
        "label_source": label_source,
        "action_type": action_type,
        "target_url": target_url,
        "canonical_target_url": canonical_target,
        "target_kind": target_kind,
        "official_target": official,
        "element_role": "form",
        "dom_region": _dom_region(tag),
        "html_locator": locator,
        "source_section_id": source_section_id,
        "source_section_heading": source_section_heading,
        "form_method": method,
        "opens_new_window": clean_text(tag.get("target")).casefold() == "_blank",
        "authentication_requirement": "not_indicated",
        "evidence": _action_evidence(
            page_card_id=page_card_id,
            locator=locator,
            label=label,
            label_source=label_source,
            target_evidence=raw_action,
            target_attribute=("action" if explicit_action else "form_default_current_url"),
            context_label=context_label,
        ),
    }


def _crawl_projection(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "status_code": metadata.get("status_code"),
        "depth": metadata.get("depth"),
        "capture_source": clean_text(metadata.get("capture_source")),
        "indexable": metadata.get("indexable"),
        "index_exclusion_reason": clean_text(metadata.get("index_exclusion_reason")),
        "robots_noindex": metadata.get("robots_noindex"),
        "corpus_source_run_id": clean_text(metadata.get("corpus_source_run_id")),
    }


def extract_page_draft(
    source_url: str,
    metadata: Mapping[str, Any],
    *,
    official_hosts: Iterable[str],
    maximum_topics: int = 24,
) -> Dict[str, Any]:
    """Extract one Page Card draft and unfiltered action candidates."""

    source_url = clean_text(source_url)
    html_path = Path(clean_text(metadata.get("html_path"))).resolve()
    if not source_url:
        raise ValueError("Page metadata key/source URL is empty")
    if not html_path.is_file():
        raise FileNotFoundError(f"Raw HTML artifact is missing for {source_url}: {html_path}")
    raw_bytes = html_path.read_bytes()
    if not raw_bytes:
        raise ValueError(f"Raw HTML artifact is empty for {source_url}: {html_path}")
    html_sha256 = sha256(raw_bytes).hexdigest()
    soup = BeautifulSoup(raw_bytes, "lxml")

    page_card_id = stable_representation_id("page-card", source_url, html_sha256)
    evidence = _EvidenceBuilder(page_card_id)
    source_evidence = evidence.add(
        source_kind="url",
        locator="page_metadata.key",
        excerpt=source_url,
        extraction_method="crawl_inventory_key",
    )

    canonical_link = soup.find("link", attrs={"rel": lambda value: value and "canonical" in value})
    canonical_from_html = clean_text(canonical_link.get("href")) if isinstance(canonical_link, Tag) else ""
    canonical_raw = _metadata_scalar(metadata, "canonical_url") or canonical_from_html or source_url
    canonical_url = normalize_url(canonical_raw, base_url=source_url) or source_url
    if _metadata_scalar(metadata, "canonical_url"):
        canonical_evidence = evidence.add(
            source_kind="crawl_metadata",
            locator="page_metadata.canonical_url",
            excerpt=canonical_raw,
            extraction_method="crawl_metadata_projection",
        )
    elif canonical_from_html and isinstance(canonical_link, Tag):
        canonical_evidence = evidence.add(
            source_kind="html_attribute",
            locator=html_locator(canonical_link),
            excerpt=canonical_from_html,
            attribute="href",
            extraction_method="canonical_link",
        )
    else:
        canonical_evidence = source_evidence

    family_raw = _metadata_scalar(metadata, "canonical_family_url") or canonical_url
    canonical_family_url = normalize_url(family_raw, base_url=source_url) or canonical_url
    family_evidence = evidence.add(
        source_kind="crawl_metadata" if _metadata_scalar(metadata, "canonical_family_url") else "url",
        locator=(
            "page_metadata.canonical_family_url"
            if _metadata_scalar(metadata, "canonical_family_url")
            else "derived_from_canonical_url"
        ),
        excerpt=family_raw,
        extraction_method="canonical_family_projection",
    )

    parsed_source = urlsplit(source_url)
    host = clean_text(metadata.get("host")) or (parsed_source.hostname or "").lower()
    path = clean_text(metadata.get("path")) or parsed_source.path or "/"

    html_tag = soup.find("html")
    html_language = clean_text(html_tag.get("lang")) if isinstance(html_tag, Tag) else ""
    language = _metadata_scalar(metadata, "language") or html_language
    if not language:
        language = "ar" if parsed_source.path.startswith("/ar/") else "und"
    language_source_kind = "crawl_metadata" if _metadata_scalar(metadata, "language") else "html_attribute" if html_language else "url"
    language_locator = "page_metadata.language" if _metadata_scalar(metadata, "language") else html_locator(html_tag) if html_language and isinstance(html_tag, Tag) else "page_url.path"
    language_evidence = evidence.add(
        source_kind=language_source_kind,
        locator=language_locator,
        excerpt=language,
        attribute="lang" if language_source_kind == "html_attribute" else "",
        extraction_method="language_projection",
    )
    meta_tags = metadata.get("meta_tags") if isinstance(metadata.get("meta_tags"), Mapping) else {}
    locale = clean_text(meta_tags.get("og:locale"))
    if isinstance(meta_tags.get("og:locale"), list):
        locale = next((clean_text(value) for value in meta_tags.get("og:locale") if clean_text(value)), "")
    locale = locale or language
    if meta_tags.get("og:locale"):
        locale_evidence = evidence.add(
            source_kind="meta_tag",
            locator="page_metadata.meta_tags.og:locale",
            excerpt=locale,
            attribute="og:locale",
            extraction_method="open_graph_locale_projection",
        )
    else:
        locale_evidence = language_evidence

    page_type = _metadata_scalar(metadata, "page_type") or "content"
    page_type_evidence = evidence.add(
        source_kind="crawl_metadata",
        locator="page_metadata.page_type" if metadata.get("page_type") else "page_metadata.default",
        excerpt=page_type,
        extraction_method="crawler_page_type_classifier" if metadata.get("page_type") else "default_content_type",
    )

    title = _metadata_scalar(metadata, "title")
    title_evidence = ""
    if title:
        title_evidence = evidence.add(
            source_kind="crawl_metadata",
            locator="page_metadata.title",
            excerpt=title,
            extraction_method="crawl_metadata_projection",
        )
    if not title:
        h1 = soup.find("h1")
        title_tag = soup.find("title")
        candidate = h1 if isinstance(h1, Tag) and _tag_text(h1) else title_tag
        title = _tag_text(candidate, maximum_chars=300)
        if title and isinstance(candidate, Tag):
            title_evidence = evidence.add(
                source_kind="html_text",
                locator=html_locator(candidate),
                excerpt=title,
                extraction_method="h1_or_title_fallback",
            )
    if not title:
        slug = parsed_source.path.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
        title = clean_text(slug) or host
        title_evidence = evidence.add(
            source_kind="url",
            locator="page_url.path",
            excerpt=title,
            extraction_method="url_slug_fallback",
        )

    metadata_description = _metadata_description(metadata)
    paragraph, paragraph_tag = _first_content_paragraph(soup)
    if metadata_description and not _is_generic_description(metadata_description):
        purpose_summary = metadata_description
        purpose_method = "meta_description_extract"
        purpose_evidence = evidence.add(
            source_kind="meta_tag",
            locator="page_metadata.description",
            excerpt=metadata_description,
            attribute="description",
            extraction_method=purpose_method,
        )
    elif paragraph and isinstance(paragraph_tag, Tag):
        purpose_summary = paragraph
        purpose_method = "first_primary_content_paragraph"
        purpose_evidence = evidence.add(
            source_kind="html_text",
            locator=html_locator(paragraph_tag),
            excerpt=paragraph,
            extraction_method=purpose_method,
        )
    elif metadata_description:
        purpose_summary = metadata_description
        purpose_method = "generic_meta_description_fallback"
        purpose_evidence = evidence.add(
            source_kind="meta_tag",
            locator="page_metadata.description",
            excerpt=metadata_description,
            attribute="description",
            extraction_method=purpose_method,
        )
    else:
        purpose_summary = title
        purpose_method = "title_fallback"
        purpose_evidence = title_evidence

    sections: List[Dict[str, Any]] = []
    section_by_locator: Dict[str, Dict[str, Any]] = {}
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        if not isinstance(heading, Tag) or _is_hidden(heading) or _has_boilerplate_ancestor(heading):
            continue
        heading_text = _tag_text(heading, maximum_chars=300)
        if not heading_text:
            continue
        locator = html_locator(heading)
        level = int(str(heading.name)[1])
        heading_evidence = evidence.add(
            source_kind="html_text",
            locator=locator,
            excerpt=heading_text,
            extraction_method="semantic_heading_extract",
        )
        section = {
            "section_id": stable_representation_id(
                "page-section", page_card_id, locator, heading_text
            ),
            "level": level,
            "heading": heading_text,
            "html_locator": locator,
            "evidence_id": heading_evidence,
        }
        sections.append(section)
        section_by_locator[locator] = section

    topic_sources: List[Tuple[str, str]] = []
    seen_topics: set[str] = set()
    for section in sections:
        normalized_topic = clean_text(section["heading"]).casefold()
        if normalized_topic in seen_topics:
            continue
        seen_topics.add(normalized_topic)
        topic_sources.append((section["heading"], section["evidence_id"]))
        if len(topic_sources) >= max(1, int(maximum_topics)):
            break
    if not topic_sources:
        topic_sources.append((title, title_evidence))
    topics = [
        {
            "label": label,
            "method": "explicit_section_heading" if evidence_id != title_evidence else "explicit_page_title",
            "evidence_ids": [evidence_id],
        }
        for label, evidence_id in topic_sources
    ]
    audience_sources = [(title, title_evidence), (purpose_summary, purpose_evidence)]
    audience_sources.extend(topic_sources)
    audiences = _semantic_audiences(audience_sources)

    official_host_set = {clean_text(value).lower() for value in official_hosts if clean_text(value)}
    candidates: List[Dict[str, Any]] = []
    order = 0
    for tag in soup.find_all(["a", "form"]):
        if not isinstance(tag, Tag) or _is_hidden(tag):
            continue
        if tag.name == "a":
            candidate = _extract_anchor_candidate(
                tag,
                page_card_id=page_card_id,
                source_url=source_url,
                official_hosts=official_host_set,
                section_by_locator=section_by_locator,
                order=order,
            )
        else:
            candidate = _extract_form_candidate(
                tag,
                page_card_id=page_card_id,
                source_url=source_url,
                official_hosts=official_host_set,
                section_by_locator=section_by_locator,
                order=order,
            )
        order += 1
        if candidate is not None:
            candidates.append(candidate)

    page_card: Dict[str, Any] = {
        "page_card_id": page_card_id,
        "schema_version": REPRESENTATION_V2_SCHEMA_VERSION,
        "source_url": source_url,
        "canonical_url": canonical_url,
        "canonical_family_url": canonical_family_url,
        "document_revision_id": None,
        "content_backed": False,
        "host": host,
        "path": path,
        "language": language,
        "locale": locale,
        "page_type": page_type,
        "title": _bound_text(title, 300),
        "purpose_summary": _bound_text(purpose_summary, 600),
        "purpose_method": purpose_method,
        "audiences": audiences,
        "topics": topics,
        "sections": sections,
        "action_ids": [],
        "retrieval_action_ids": [],
        "source_html": {
            "path": str(html_path),
            "sha256": html_sha256,
            "byte_count": len(raw_bytes),
        },
        "crawl": _crawl_projection(metadata),
        "field_evidence": {
            "source_url": [source_evidence],
            "canonical_url": [canonical_evidence],
            "canonical_family_url": [family_evidence],
            "host": [source_evidence],
            "path": [source_evidence],
            "language": [language_evidence],
            "locale": [locale_evidence],
            "page_type": [page_type_evidence],
            "title": [title_evidence],
            "purpose_summary": [purpose_evidence],
        },
        "evidence": evidence.items,
    }
    return {
        "source_url": source_url,
        "page_card": page_card,
        "action_candidates": candidates,
        "extraction_stats": {
            "raw_action_candidates": len(candidates),
            "sections": len(sections),
            "html_bytes": len(raw_bytes),
        },
    }


def _candidate_preference(candidate: Mapping[str, Any]) -> Tuple[int, int, int]:
    region_rank = {
        "main": 0,
        "article": 1,
        "body": 2,
        "aside": 3,
        "navigation": 4,
        "header": 5,
        "footer": 6,
    }
    return (
        region_rank.get(clean_text(candidate.get("dom_region")), 9),
        0 if clean_text(candidate.get("context_label")) else 1,
        int(candidate.get("_order") or 0),
    )


def _keep_action(candidate: Mapping[str, Any], *, is_template: bool) -> bool:
    action_type = clean_text(candidate.get("action_type"))
    region = clean_text(candidate.get("dom_region"))
    semantic_types = {
        "apply",
        "register",
        "login",
        "download",
        "email",
        "telephone",
        "contact",
        "search",
        "submit_form",
    }
    if action_type in semantic_types:
        return True
    if region not in {"main", "article", "body"}:
        return False
    if is_template:
        return False
    if action_type == "fragment_navigation":
        return True
    source_target = _normalized_comparison_url(candidate.get("target_url"))
    return bool(source_target)


def finalize_page_drafts(
    drafts: Sequence[MutableMapping[str, Any]],
    *,
    template_minimum_page_count: int = 20,
    template_minimum_page_ratio: float = 0.03,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Deduplicate candidates, detect templates, and produce final actions."""

    if template_minimum_page_count < 1:
        raise ValueError("template_minimum_page_count must be at least 1")
    if not 0 <= template_minimum_page_ratio <= 1:
        raise ValueError("template_minimum_page_ratio must be between 0 and 1")
    page_count = len(drafts)
    frequency_threshold = max(
        int(template_minimum_page_count),
        int(math.ceil(page_count * template_minimum_page_ratio)),
    )
    signature_pages: Dict[str, set[str]] = defaultdict(set)
    for draft in drafts:
        page_id = clean_text((draft.get("page_card") or {}).get("page_card_id"))
        for candidate in draft.get("action_candidates") or []:
            signature = clean_text(candidate.get("_signature"))
            if page_id and signature:
                signature_pages[signature].add(page_id)

    page_cards: List[Dict[str, Any]] = []
    actions: List[Dict[str, Any]] = []
    raw_candidates = 0
    duplicate_candidates = 0
    template_candidates = 0
    discarded_candidates = 0
    for draft in sorted(drafts, key=lambda value: clean_text(value.get("source_url"))):
        page = dict(draft.get("page_card") or {})
        page_id = clean_text(page.get("page_card_id"))
        candidates = [dict(value) for value in draft.get("action_candidates") or []]
        raw_candidates += len(candidates)
        best_by_signature: Dict[str, Dict[str, Any]] = {}
        for candidate in candidates:
            signature = clean_text(candidate.get("_signature"))
            existing = best_by_signature.get(signature)
            if existing is None or _candidate_preference(candidate) < _candidate_preference(existing):
                if existing is not None:
                    duplicate_candidates += 1
                best_by_signature[signature] = candidate
            else:
                duplicate_candidates += 1

        page_actions: List[Dict[str, Any]] = []
        for signature, candidate in sorted(
            best_by_signature.items(), key=lambda item: int(item[1].get("_order") or 0)
        ):
            template_page_count = len(signature_pages.get(signature, ())) or 1
            template_ratio = template_page_count / page_count if page_count else 0.0
            region_template = clean_text(candidate.get("dom_region")) in {
                "header",
                "footer",
                "navigation",
                "aside",
            }
            is_template = region_template or template_page_count >= frequency_threshold
            if is_template:
                template_candidates += 1
            if not _keep_action(candidate, is_template=is_template):
                discarded_candidates += 1
                continue
            action_id = stable_representation_id(
                "page-action",
                page_id,
                candidate.get("action_type"),
                candidate.get("canonical_target_url"),
                candidate.get("label"),
                candidate.get("context_label"),
            )
            action = {
                key: value
                for key, value in candidate.items()
                if not key.startswith("_")
            }
            action.update(
                {
                    "action_id": action_id,
                    "page_card_id": page_id,
                    "template_page_count": template_page_count,
                    "template_page_ratio": round(template_ratio, 8),
                    "is_template": is_template,
                    # Keep high-value global actions in the page's navigation
                    # inventory, but do not let repeated header/footer CTAs
                    # dominate a later action-retrieval index.
                    "retrieval_eligible": not is_template,
                }
            )
            page_actions.append(action)
        page_actions.sort(key=lambda value: str(value.get("action_id") or ""))
        page["action_ids"] = [str(value["action_id"]) for value in page_actions]
        page["retrieval_action_ids"] = [
            str(value["action_id"])
            for value in page_actions
            if bool(value.get("retrieval_eligible"))
        ]
        actions.extend(page_actions)
        page_cards.append(page)

    actions.sort(key=lambda value: (str(value.get("page_card_id") or ""), str(value.get("action_id") or "")))
    page_cards.sort(key=lambda value: str(value.get("source_url") or ""))
    stats = {
        "page_count": page_count,
        "template_frequency_threshold": frequency_threshold,
        "raw_action_candidates": raw_candidates,
        "duplicate_action_candidates": duplicate_candidates,
        "template_action_candidates": template_candidates,
        "discarded_action_candidates": discarded_candidates,
        "retained_actions": len(actions),
        "retrieval_eligible_actions": sum(
            bool(value.get("retrieval_eligible")) for value in actions
        ),
        "retained_template_actions": sum(
            bool(value.get("is_template")) for value in actions
        ),
        "retained_actions_by_type": dict(
            sorted(Counter(str(value.get("action_type") or "") for value in actions).items())
        ),
    }
    return page_cards, actions, stats


def build_document_revisions(
    inventory_documents: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, str], set[str]]:
    """Create stable revisions and a deterministic URL-to-revision index."""

    documents: List[Dict[str, Any]] = []
    primary_url_index: Dict[str, str] = {}
    fallback_url_revisions: Dict[str, set[str]] = defaultdict(set)
    web_revision_ids: set[str] = set()
    for source in sorted(
        inventory_documents,
        key=lambda value: str(value.get("record_id") or ""),
    ):
        corpus_record_id = clean_text(source.get("record_id"))
        source_locator = source.get("source_locator") if isinstance(source.get("source_locator"), Mapping) else {}
        locator_kind = clean_text(source_locator.get("kind"))
        locator_value = clean_text(source_locator.get("value"))
        markdown_sha256 = clean_text(source.get("markdown_sha256")).lower()
        if not corpus_record_id or not locator_kind or not locator_value:
            raise ValueError("Corpus inventory document lacks stable record/source provenance")
        if len(markdown_sha256) != 64:
            raise ValueError(f"Corpus inventory document has invalid Markdown digest: {corpus_record_id}")
        document_id = stable_representation_id("document", locator_kind, locator_value)
        revision_id = stable_representation_id(
            "document-revision", document_id, markdown_sha256
        )
        document: Dict[str, Any] = {
            "document_id": document_id,
            "document_revision_id": revision_id,
            "corpus_record_id": corpus_record_id,
            "revision_status": "current",
            "source_type": clean_text(source.get("source_type")),
            "language": clean_text(source.get("language")) or "und",
            "title": clean_text(source.get("title")),
            "source_locator": {"kind": locator_kind, "value": locator_value},
            "source_url": clean_text(source.get("source_url")),
            "canonical_url": clean_text(source.get("canonical_url")),
            "canonical_family_url": clean_text(source.get("canonical_family_url")),
            "source_file": clean_text(source.get("source_file")),
            "markdown_path": str(Path(clean_text(source.get("markdown_path"))).resolve()),
            "markdown_sha256": markdown_sha256,
            "content_statistics": dict(source.get("content_statistics") or {}),
            "page_card_ids": [],
            "media_references": [
                dict(value)
                for value in source.get("media_references") or []
                if isinstance(value, Mapping)
            ],
        }
        documents.append(document)
        primary_url = _normalized_comparison_url(source.get("source_url"))
        if primary_url:
            web_revision_ids.add(revision_id)
            existing = primary_url_index.get(primary_url)
            if existing and existing != revision_id:
                raise ValueError(
                    f"A primary corpus URL maps to multiple revisions: {primary_url}"
                )
            primary_url_index[primary_url] = revision_id
        # Canonical and alias URLs are only safe as fallbacks when exactly one
        # revision claims them.  This prevents generic/malformed canonicals
        # (for example a site root) from binding unrelated Page Cards.
        raw_aliases = source.get("source_alias_urls")
        aliases = raw_aliases if isinstance(raw_aliases, list) else []
        for value in [source.get("canonical_url"), *aliases]:
            normalized = _normalized_comparison_url(value)
            if normalized:
                fallback_url_revisions[normalized].add(revision_id)
    url_index = dict(primary_url_index)
    for normalized, revision_ids in sorted(fallback_url_revisions.items()):
        if normalized not in url_index and len(revision_ids) == 1:
            url_index[normalized] = next(iter(revision_ids))
    return documents, url_index, web_revision_ids


def link_page_cards_to_revisions(
    page_cards: Sequence[MutableMapping[str, Any]],
    url_index: Mapping[str, str],
) -> Dict[str, Any]:
    linked = 0
    unlinked_urls: List[str] = []
    for page in page_cards:
        candidates = (
            page.get("source_url"),
            page.get("canonical_url"),
            page.get("canonical_family_url"),
        )
        revision_id = next(
            (
                url_index[normalized]
                for normalized in (
                    _normalized_comparison_url(value) for value in candidates
                )
                if normalized and normalized in url_index
            ),
            None,
        )
        page["document_revision_id"] = revision_id
        page["content_backed"] = revision_id is not None
        if revision_id:
            linked += 1
        else:
            unlinked_urls.append(clean_text(page.get("source_url")))
    return {
        "linked_page_cards": linked,
        "unlinked_page_cards": len(unlinked_urls),
        "unlinked_page_url_samples": unlinked_urls[:100],
    }
