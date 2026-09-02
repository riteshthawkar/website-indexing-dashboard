"""Scope rendered SPA HTML to the route-specific content before extraction.

Rendered application snapshots can legitimately contain content for many routes
at once (accordions, drawers, carousels, or pre-rendered panels).  Indexing the
whole DOM under every route produces duplicate documents and false citations.
This module uses route identity already present in the HTML rather than any
site-specific answer or topic knowledge.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import escape
from typing import Iterable
from urllib.parse import unquote, urlparse

from bs4 import BeautifulSoup, Tag


@dataclass(frozen=True)
class RouteScopeResult:
    html: str
    method: str = ""
    document_title: str = ""

    @property
    def applied(self) -> bool:
        return bool(self.method)


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split()).strip()


def _normalized_identity(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_text(value).casefold()).strip()


def _normalized_route_id(value: object) -> str:
    return re.sub(r"-\d+$", "", _clean_text(value).casefold())


def _route_identity_similarity(left: object, right: object) -> float:
    left_tokens = set(_normalized_identity(_normalized_route_id(left)).split())
    right_tokens = set(_normalized_identity(_normalized_route_id(right)).split())
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return 0.0
    overlap = left_tokens & right_tokens
    return len(overlap) / float(min(len(left_tokens), len(right_tokens)))


def _route_slugs(source_urls: Iterable[str]) -> list[str]:
    values: list[str] = []
    for source_url in source_urls:
        try:
            path = unquote(urlparse(str(source_url or "")).path or "")
        except Exception:
            continue
        parts = [part.strip() for part in path.rstrip("/").split("/") if part.strip()]
        if parts:
            values.append(parts[-1])
    return list(dict.fromkeys(values))


def _is_faq_route(source_urls: Iterable[str]) -> bool:
    for source_url in source_urls:
        try:
            parts = {
                part.casefold()
                for part in (urlparse(str(source_url or "")).path or "").split("/")
                if part
            }
        except Exception:
            continue
        if "faq" in parts or "faqs" in parts:
            return True
    return False


def document_title_from_html(raw: str) -> str:
    """Extract a readable page title while removing a declared site suffix."""

    soup = BeautifulSoup(raw or "", "html.parser")
    site_name_node = soup.find(
        "meta",
        attrs={"property": re.compile(r"^og:site_name$", re.IGNORECASE)},
    )
    site_name = _clean_text(site_name_node.get("content")) if site_name_node else ""
    title_node = soup.find(
        "meta",
        attrs={"property": re.compile(r"^og:title$", re.IGNORECASE)},
    )
    title = _clean_text(title_node.get("content")) if title_node else ""
    if not title and soup.title:
        title = _clean_text(soup.title.get_text(" ", strip=True))
    if not title:
        heading = soup.find("h1")
        title = _clean_text(heading.get_text(" ", strip=True)) if heading else ""
    if site_name and title:
        title = re.sub(
            rf"\s*(?:\||-|–|—|:)\s*{re.escape(site_name)}\s*$",
            "",
            title,
            flags=re.IGNORECASE,
        ).strip()
    return title


def _has_substantive_content(node: Tag, *, minimum_words: int = 12) -> bool:
    text = _clean_text(node.get_text(" ", strip=True))
    words = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    return len(words) >= minimum_words and bool(
        node.find(["p", "li", "table", "h1", "h2", "h3", "h4"])
    )


def _faq_panel(soup: BeautifulSoup, slugs: list[str]) -> Tag | None:
    panels = [
        node
        for node in soup.find_all(attrs={"data-pc-name": "accordionpanel"})
        if isinstance(node, Tag)
    ]
    if not panels:
        return None
    slug_set = {slug.casefold() for slug in slugs}
    exact = [
        panel
        for panel in panels
        if _clean_text(panel.get("id")).casefold() in slug_set
    ]
    route_family = {_normalized_route_id(slug) for slug in slugs}
    route_variants = [
        panel
        for panel in panels
        if _normalized_route_id(panel.get("id")) in route_family
    ]
    fuzzy_variants = sorted(
        (
            (
                max(
                    (_route_identity_similarity(panel.get("id"), slug) for slug in slugs),
                    default=0.0,
                ),
                panel,
            )
            for panel in panels
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    fuzzy_variants = [panel for score, panel in fuzzy_variants if score >= 0.8]
    active = [
        panel
        for panel in panels
        if _clean_text(panel.get("data-p-active")).casefold() == "true"
    ]
    for panel in [*exact, *active, *route_variants, *fuzzy_variants]:
        if _has_substantive_content(panel, minimum_words=5):
            return panel
    return None


def _faq_panel_title(panel: Tag) -> str:
    question = panel.find(["button", "h1", "h2", "h3", "h4", "h5"])
    return _clean_text(question.get_text(" ", strip=True)) if question else ""


def _route_id_panel(soup: BeautifulSoup, slugs: list[str]) -> Tag | None:
    for slug in slugs:
        candidates = soup.find_all(id=re.compile(rf"^{re.escape(slug)}$", re.IGNORECASE))
        for candidate in candidates:
            if not isinstance(candidate, Tag) or candidate.name in {
                "a",
                "button",
                "h1",
                "h2",
                "h3",
                "h4",
            }:
                continue
            class_tokens = {
                str(value).casefold()
                for value in (candidate.get("class") or [])
            }
            route_container = bool(
                class_tokens
                & {
                    "panel-content",
                    "modal-content",
                    "drawer-content",
                    "tab-content",
                    "accordion-panel",
                    "p-accordionpanel",
                }
            )
            if route_container and _has_substantive_content(candidate, minimum_words=8):
                return candidate
    return None


def _title_matched_record(soup: BeautifulSoup, title: str) -> Tag | None:
    identity = _normalized_identity(title)
    if not identity:
        return None
    for attribute in ("data-title", "aria-label"):
        for candidate in soup.find_all(attrs={attribute: True}):
            if not isinstance(candidate, Tag):
                continue
            if _normalized_identity(candidate.get(attribute)) != identity:
                continue
            if candidate.name not in {"article", "section", "div"}:
                continue
            if _has_substantive_content(candidate, minimum_words=8):
                return candidate
    return None


def _wrap_scoped_node(node: Tag, *, title: str, method: str) -> str:
    heading = ""
    if title and not node.find(["h1", "h2", "h3"]):
        heading = f"<h1>{escape(title)}</h1>"
    return (
        "<html><head>"
        f"<title>{escape(title)}</title>"
        "</head><body>"
        f'<main data-route-scope="{escape(method)}">{heading}{str(node)}</main>'
        "</body></html>"
    )


def scope_route_specific_html(raw: str, source_urls: Iterable[str]) -> RouteScopeResult:
    """Return route-scoped HTML when the rendered DOM exposes a safe identity.

    If no high-confidence route container is found, the original HTML is
    returned unchanged.  This makes the scoper safe for conventional pages.
    """

    urls = [str(value) for value in source_urls if str(value).strip()]
    title = document_title_from_html(raw)
    slugs = _route_slugs(urls)
    if not raw or not slugs:
        return RouteScopeResult(html=raw, document_title=title)

    soup = BeautifulSoup(raw, "html.parser")
    candidate: Tag | None = None
    method = ""
    if _is_faq_route(urls):
        candidate = _faq_panel(soup, slugs)
        if candidate is not None:
            method = "faq_active_panel"
            title = _faq_panel_title(candidate) or title
    if candidate is None:
        candidate = _route_id_panel(soup, slugs)
        if candidate is not None:
            method = "route_id_panel"
    if candidate is None:
        candidate = _title_matched_record(soup, title)
        if candidate is not None:
            method = "title_matched_record"

    if candidate is None:
        return RouteScopeResult(html=raw, document_title=title)
    return RouteScopeResult(
        html=_wrap_scoped_node(candidate, title=title, method=method),
        method=method,
        document_title=title,
    )
