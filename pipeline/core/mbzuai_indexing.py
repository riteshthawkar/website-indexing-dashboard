"""MBZUAI indexing helpers shared by pipeline stages.

These utilities keep canonicalization deterministic and side-effect free so
formatters, graph builders, and eval tooling use the same URL/page identity
rules.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


LOCALE_PATH_SEGMENTS = {"ar", "en"}
DEFAULT_OFFICIAL_HOST_SUFFIXES = (
    "mbzuai.ac.ae",
    "staticcdn.mbzuai.ac.ae",
)
GENERIC_HOME_TITLES = {
    "mbzuai - mohamed bin zayed university of artificial intelligence",
    "mohamed bin zayed university of artificial intelligence",
    "mbzuai",
}
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "gbraid",
    "wbraid",
    "mc_cid",
    "mc_eid",
}
ROBOTS_META_KEYS = {
    "robots",
    "googlebot",
    "googlebot-news",
    "bingbot",
    "x-robots-tag",
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _text_values(value: Any) -> List[str]:
    if isinstance(value, (list, tuple, set)):
        output: List[str] = []
        for item in value:
            output.extend(_text_values(item))
        return output
    text = clean_text(value)
    return [text] if text else []


def robots_directives(metadata: Mapping[str, Any] | None) -> List[str]:
    """Return normalized robots directives declared by page metadata."""
    metadata = metadata or {}
    values: List[str] = []
    for key, value in metadata.items():
        if clean_text(key).casefold() in ROBOTS_META_KEYS:
            values.extend(_text_values(value))
    meta_tags = metadata.get("meta_tags")
    if isinstance(meta_tags, Mapping):
        for key, value in meta_tags.items():
            if clean_text(key).casefold() in ROBOTS_META_KEYS:
                values.extend(_text_values(value))

    directives: set[str] = set()
    for value in values:
        directives.update(
            token
            for token in re.split(r"[\s,;]+", value.casefold())
            if token
        )
    return sorted(directives)


def has_robots_noindex(metadata: Mapping[str, Any] | None) -> bool:
    """Return whether page metadata declares the standard noindex behavior."""
    directives = set(robots_directives(metadata))
    # The standard ``none`` directive is equivalent to ``noindex, nofollow``.
    return bool({"noindex", "none"} & directives)


def sha1_text(value: str, length: int | None = 40) -> str:
    digest = hashlib.sha1(str(value or "").encode("utf-8", errors="ignore")).hexdigest()
    return digest if length is None else digest[:length]


def normalize_url(url: Any, *, strip_fragment: bool = True) -> str:
    """Normalize an HTTP(S) URL without changing its language path."""
    text = str(url or "").strip()
    if not text:
        return ""
    try:
        parsed = urlparse(text)
    except Exception:
        return text
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return text

    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if not host:
        return text
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"
    else:
        netloc = host

    path = parsed.path or "/"
    if path != "/":
        path = "/" + path.strip("/")
    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=False):
        lower_key = key.lower()
        if lower_key in TRACKING_QUERY_KEYS or any(lower_key.startswith(prefix) for prefix in TRACKING_QUERY_PREFIXES):
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items), doseq=True)
    fragment = "" if strip_fragment else parsed.fragment
    return urlunparse((scheme, netloc, path, "", query, fragment))


def strip_locale_from_path(path: Any) -> str:
    text = str(path or "/").strip() or "/"
    if not text.startswith("/"):
        text = f"/{text}"
    parts = [part for part in text.split("/") if part]
    if parts and parts[0].lower() in LOCALE_PATH_SEGMENTS:
        parts = parts[1:]
    stripped = "/" + "/".join(parts)
    return stripped if stripped != "/" else "/"


def locale_family_url(url: Any) -> str:
    normalized = normalize_url(url)
    if not normalized:
        return ""
    try:
        parsed = urlparse(normalized)
    except Exception:
        return normalized
    path = strip_locale_from_path(parsed.path)
    return urlunparse((parsed.scheme, parsed.netloc, path, "", parsed.query, ""))


def url_host(url: Any) -> str:
    try:
        return (urlparse(str(url or "")).hostname or "").lower()
    except Exception:
        return ""


def is_official_source_url(url: Any, allowed_host_suffixes: Sequence[str] | None = None) -> bool:
    host = url_host(url)
    if not host:
        return False
    suffixes = tuple(str(item or "").strip().lower().lstrip(".") for item in (allowed_host_suffixes or DEFAULT_OFFICIAL_HOST_SUFFIXES))
    return any(host == suffix or host.endswith(f".{suffix}") for suffix in suffixes if suffix)


def is_homepage_redirect_alias(source_url: Any, canonical_url: Any, title: Any = "", status_code: Any = None) -> bool:
    source = normalize_url(source_url)
    canonical = normalize_url(canonical_url)
    if not source or not canonical or source == canonical:
        return False
    try:
        source_path = urlparse(source).path or "/"
        canonical_path = urlparse(canonical).path or "/"
    except Exception:
        return False
    if canonical_path != "/" or source_path == "/":
        return False
    title_text = clean_text(title).casefold()
    try:
        status_int = int(status_code)
    except (TypeError, ValueError):
        status_int = 0
    return status_int in {0, 301, 302, 303, 307, 308, 404} or title_text in GENERIC_HOME_TITLES


def classify_page_type(source_url: Any, metadata: Mapping[str, Any] | None = None) -> str:
    metadata = metadata or {}
    source = normalize_url(source_url or metadata.get("url") or metadata.get("source_url"))
    canonical = normalize_url(metadata.get("canonical_url") or source)
    title = metadata.get("title") or metadata.get("page_title") or ""
    status_code = metadata.get("status_code")
    if is_homepage_redirect_alias(source, canonical, title, status_code):
        return "redirect_alias"
    try:
        path = (urlparse(source).path or "/").lower()
    except Exception:
        path = ""
    doc_type = clean_text(metadata.get("document_type")).lower()
    if path.endswith(".pdf") or "pdf" in doc_type:
        return "pdf"
    if "/news/" in path or "/event/" in path or "/the-node/" in path:
        return "news_or_event"
    if "/leadership" in path or "office-of-the-president" in path:
        return "leadership"
    if "/study/" in path or "admission" in path or "program" in path:
        return "admissions_or_program"
    if "/student-resources" in path or "campus" in path:
        return "student_life"
    if "/about/contact" in path or path.endswith("/contact"):
        return "contact"
    return "content"


def infer_language(url: Any, metadata: Mapping[str, Any] | None = None) -> str:
    metadata = metadata or {}
    language = clean_text(metadata.get("language")).lower()
    if language:
        return language.split("-", 1)[0]
    try:
        path_parts = [part for part in urlparse(str(url or "")).path.split("/") if part]
    except Exception:
        path_parts = []
    if path_parts and path_parts[0].lower() in LOCALE_PATH_SEGMENTS:
        return path_parts[0].lower()
    return ""


def page_content_hash(metadata: Mapping[str, Any] | None, *, max_bytes: int = 2_000_000) -> str:
    metadata = metadata or {}
    for key in ("markdown_path", "html_path"):
        path_text = clean_text(metadata.get(key))
        if not path_text:
            continue
        path = Path(path_text)
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()[:max_bytes]
        except OSError:
            continue
        if data:
            return hashlib.sha1(data).hexdigest()
    fallback_parts = [
        metadata.get("url"),
        metadata.get("title"),
        metadata.get("description"),
        metadata.get("content_bytes"),
        metadata.get("headings"),
    ]
    return sha1_text("|".join(str(part or "") for part in fallback_parts), length=None)


def canonicalize_page_metadata(
    page_metadata: Mapping[str, Any] | None,
    *,
    authorized_noindex_hosts: Sequence[str] | None = None,
    noindex_override_reason: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Return metadata keyed by normalized source URL with canonical identity fields.

    ``noindex`` remains fail-closed by default.  A caller may override it only
    for exact, explicitly authorized hosts and only with a recorded reason.
    The source directive is retained in the resulting evidence either way.
    """

    authorized_hosts = {
        str(value or "").strip().lower().strip(".")
        for value in (authorized_noindex_hosts or [])
        if str(value or "").strip()
    }
    override_reason = clean_text(noindex_override_reason)
    output: Dict[str, Dict[str, Any]] = {}
    family_to_variants: Dict[str, List[str]] = {}

    for raw_url, raw_metadata in sorted((page_metadata or {}).items()):
        if not isinstance(raw_metadata, Mapping):
            continue
        source_url = normalize_url(raw_metadata.get("url") or raw_url)
        if not source_url:
            continue
        canonical_url = normalize_url(raw_metadata.get("canonical_url") or source_url) or source_url
        family_url = locale_family_url(canonical_url or source_url) or canonical_url or source_url
        language = infer_language(source_url, raw_metadata)
        parsed = urlparse(source_url)
        normalized_path = strip_locale_from_path(parsed.path or "/")
        alternates = []
        for item in raw_metadata.get("alternate_urls") or []:
            if not isinstance(item, Mapping):
                continue
            href = normalize_url(item.get("url"))
            if not href:
                continue
            alternates.append(
                {
                    "url": href,
                    "hreflang": clean_text(item.get("hreflang")).lower(),
                    "type": clean_text(item.get("type")),
                }
            )

        enriched = dict(raw_metadata)
        page_type = classify_page_type(source_url, {**raw_metadata, "canonical_url": canonical_url})
        declared_robots_directives = robots_directives(raw_metadata)
        robots_noindex = has_robots_noindex(raw_metadata)
        source_marked_non_indexable = raw_metadata.get("indexable") is False
        noindex_overridden = bool(
            robots_noindex
            and override_reason
            and url_host(source_url) in authorized_hosts
        )
        if page_type == "redirect_alias":
            indexable = False
            exclusion_reason = "homepage_redirect_alias"
        elif robots_noindex and not noindex_overridden:
            indexable = False
            exclusion_reason = "robots_noindex"
        elif source_marked_non_indexable and not noindex_overridden:
            indexable = False
            exclusion_reason = clean_text(raw_metadata.get("index_exclusion_reason")) or "source_marked_non_indexable"
        else:
            indexable = True
            exclusion_reason = ""
        enriched.update(
            {
                "url": source_url,
                "source_url": source_url,
                "normalized_url": source_url,
                "canonical_url": canonical_url,
                "canonical_family_url": family_url,
                "normalized_path": normalized_path,
                "language": language,
                "alternate_urls": alternates,
                "content_hash": page_content_hash({**raw_metadata, "url": source_url}),
                "page_type": page_type,
                "indexable": indexable,
                "index_exclusion_reason": exclusion_reason,
                "robots_directives": declared_robots_directives,
                "robots_noindex": robots_noindex,
                "robots_noindex_overridden": noindex_overridden,
                "indexability_override_reason": (
                    override_reason if noindex_overridden else ""
                ),
                "url_identity_version": 1,
            }
        )
        output[source_url] = enriched
        family_to_variants.setdefault(family_url, []).append(source_url)
        for alternate in alternates:
            alternate_url = alternate.get("url")
            if alternate_url:
                family_to_variants.setdefault(locale_family_url(alternate_url) or family_url, []).append(alternate_url)

    for source_url, metadata in output.items():
        variants = sorted(dict.fromkeys(family_to_variants.get(metadata["canonical_family_url"], [])))
        metadata["locale_variant_urls"] = variants
        metadata["has_locale_variants"] = len(variants) > 1
    return output


def build_url_identity_map(page_metadata: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    records = []
    by_family: Dict[str, List[str]] = {}
    for source_url, metadata in sorted(page_metadata.items()):
        family_url = clean_text(metadata.get("canonical_family_url")) or locale_family_url(source_url)
        by_family.setdefault(family_url, []).append(source_url)
        records.append(
            {
                "source_url": source_url,
                "canonical_url": metadata.get("canonical_url") or source_url,
                "canonical_family_url": family_url,
                "normalized_path": metadata.get("normalized_path") or strip_locale_from_path(urlparse(source_url).path),
                "language": metadata.get("language") or "",
                "title": metadata.get("title") or "",
                "content_hash": metadata.get("content_hash") or "",
                "page_type": metadata.get("page_type") or "",
                "indexable": bool(metadata.get("indexable", True)),
                "index_exclusion_reason": metadata.get("index_exclusion_reason") or "",
                "robots_noindex": bool(metadata.get("robots_noindex", False)),
                "robots_noindex_overridden": bool(
                    metadata.get("robots_noindex_overridden", False)
                ),
                "indexability_override_reason": metadata.get(
                    "indexability_override_reason"
                )
                or "",
                "locale_variant_urls": metadata.get("locale_variant_urls") or [],
            }
        )
    duplicate_families = {family: sorted(urls) for family, urls in by_family.items() if len(set(urls)) > 1}
    return {
        "schema_version": 1,
        "record_count": len(records),
        "canonical_family_count": len(by_family),
        "duplicate_family_count": len(duplicate_families),
        "records": records,
        "duplicate_families": duplicate_families,
    }


def compare_page_hashes(
    current_metadata: Mapping[str, Mapping[str, Any]],
    previous_metadata: Mapping[str, Mapping[str, Any]] | None,
) -> Dict[str, Any]:
    previous_metadata = previous_metadata or {}
    current_by_family = {
        clean_text(item.get("canonical_family_url")) or locale_family_url(url): item
        for url, item in current_metadata.items()
    }
    previous_by_family = {
        clean_text(item.get("canonical_family_url")) or locale_family_url(url): item
        for url, item in previous_metadata.items()
    }
    current_keys = set(current_by_family)
    previous_keys = set(previous_by_family)

    new_pages = sorted(current_keys - previous_keys)
    removed_pages = sorted(previous_keys - current_keys)
    changed_pages = []
    unchanged_pages = []
    for key in sorted(current_keys & previous_keys):
        current_hash = clean_text(current_by_family[key].get("content_hash"))
        previous_hash = clean_text(previous_by_family[key].get("content_hash"))
        if current_hash and previous_hash and current_hash == previous_hash:
            unchanged_pages.append(key)
        else:
            changed_pages.append(key)

    if not previous_metadata:
        new_pages = sorted(current_keys)
        changed_pages = []
        unchanged_pages = []
        removed_pages = []

    return {
        "schema_version": 1,
        "current_page_count": len(current_keys),
        "previous_page_count": len(previous_keys),
        "new_pages": new_pages,
        "changed_pages": changed_pages,
        "unchanged_pages": unchanged_pages,
        "removed_pages": removed_pages,
        "new_page_count": len(new_pages),
        "changed_page_count": len(changed_pages),
        "unchanged_page_count": len(unchanged_pages),
        "removed_page_count": len(removed_pages),
    }


def canonicalize_link_graph(
    graph: Mapping[str, Any] | None,
    page_metadata: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    graph = graph or {}
    nodes_by_id: Dict[str, Dict[str, Any]] = {}
    edges_by_id: Dict[str, Dict[str, Any]] = {}

    def node_id_for(url: str) -> str:
        # A locale family is a useful grouping key, but it is not page
        # identity. English and Arabic variants can have different content,
        # titles, and link topology, so they must remain separate graph nodes.
        return f"page:{sha1_text(normalize_url(url), 24)}"

    def stable_value_key(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def normalized_list(value: Any) -> List[Any]:
        values = value if isinstance(value, (list, tuple, set)) else [value]
        by_key: Dict[str, Any] = {}
        for item in values:
            if item in (None, "", [], {}):
                continue
            by_key[stable_value_key(item)] = item
        return [by_key[key] for key in sorted(by_key)]

    def merge_edge_properties(
        current: Mapping[str, Any] | None,
        incoming: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        """Merge duplicate normalized-link evidence without order dependence."""

        current = current or {}
        incoming = incoming or {}
        merged: Dict[str, Any] = {}
        list_evidence_keys = {"anchor_texts", "rels", "sources"}
        link_types = sorted(
            {
                clean_text(value)
                for value in (
                    current.get("link_type"),
                    incoming.get("link_type"),
                    *normalized_list(current.get("link_type_variants")),
                    *normalized_list(incoming.get("link_type_variants")),
                )
                if clean_text(value)
            }
        )
        if link_types:
            merged["link_type"] = link_types[0]
            if len(link_types) > 1:
                merged["link_type_variants"] = link_types

        for key in sorted(set(current) | set(incoming)):
            if key in {"link_type", "link_type_variants"}:
                continue
            left = current.get(key)
            right = incoming.get(key)
            if key in list_evidence_keys:
                values = normalized_list(left) + normalized_list(right)
                merged[key] = normalized_list(values)
                continue
            if isinstance(left, (list, tuple, set)) or isinstance(right, (list, tuple, set)):
                merged[key] = normalized_list(normalized_list(left) + normalized_list(right))
                continue
            candidates = [value for value in (left, right) if value not in (None, "", [], {})]
            if candidates:
                # Unknown scalar properties are not part of the link contract.
                # Select them by serialized value so duplicate input order can
                # never change the canonical graph.
                merged[key] = min(candidates, key=stable_value_key)
        return merged

    for raw_source_url, metadata in sorted(page_metadata.items()):
        source_url = normalize_url(metadata.get("url") or raw_source_url)
        if not source_url:
            continue
        node_id = node_id_for(source_url)
        nodes_by_id[node_id] = {
            "id": node_id,
            "url": source_url,
            "canonical_url": normalize_url(metadata.get("canonical_url")) or source_url,
            "canonical_family_url": metadata.get("canonical_family_url") or locale_family_url(source_url),
            "node_type": "page",
            "label": metadata.get("title") or source_url,
            "properties": {
                "title": metadata.get("title") or "",
                "description": metadata.get("description") or "",
                "language": metadata.get("language") or "",
                "normalized_path": metadata.get("normalized_path") or "",
                "content_hash": metadata.get("content_hash") or "",
                "status_code": metadata.get("status_code"),
                "depth": metadata.get("depth"),
            },
        }

    for raw_edge in graph.get("edges") or []:
        if not isinstance(raw_edge, Mapping):
            continue
        source_url = normalize_url(raw_edge.get("source_url"))
        target_url = normalize_url(raw_edge.get("target_url"))
        if not source_url or not target_url:
            continue
        source_id = node_id_for(source_url)
        target_id = node_id_for(target_url)
        if source_id not in nodes_by_id:
            nodes_by_id[source_id] = {
                "id": source_id,
                "url": source_url,
                "canonical_family_url": locale_family_url(source_url),
                "node_type": "discovered_url",
                "label": source_url,
                "properties": {},
            }
        if target_id not in nodes_by_id:
            nodes_by_id[target_id] = {
                "id": target_id,
                "url": target_url,
                "canonical_family_url": locale_family_url(target_url),
                "node_type": "discovered_url",
                "label": target_url,
                "properties": {},
            }
        edge_id = f"edge:{sha1_text(f'{source_id}|{target_id}', 24)}"
        existing_edge = edges_by_id.get(edge_id)
        edges_by_id[edge_id] = {
            "id": edge_id,
            "edge_type": "LINKS_TO",
            "source_id": source_id,
            "target_id": target_id,
            "source_url": source_url,
            "target_url": target_url,
            "source_family_url": locale_family_url(source_url),
            "target_family_url": locale_family_url(target_url),
            "properties": merge_edge_properties(
                (existing_edge or {}).get("properties"),
                raw_edge.get("properties") if isinstance(raw_edge.get("properties"), Mapping) else {},
            ),
        }

    nodes = sorted(nodes_by_id.values(), key=lambda item: item["id"])
    edges = sorted(edges_by_id.values(), key=lambda item: item["id"])

    link_type_counts: Dict[str, int] = {}
    for edge in edges:
        link_type = clean_text((edge.get("properties") or {}).get("link_type")) or "unknown"
        link_type_counts[link_type] = link_type_counts.get(link_type, 0) + 1

    return {
        "schema_version": 3,
        "graph_type": "mbzuai_canonical_page_link_graph",
        "node_identity": "normalized_url_v1",
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "link_type_counts": link_type_counts,
        },
    }


def compact_citation_anchor(
    *,
    source_url: str,
    metadata: Mapping[str, Any] | None,
    section_path: Iterable[Any] | None = None,
    chunk_index: int | None = None,
) -> Dict[str, Any]:
    metadata = metadata or {}
    sections = [clean_text(item) for item in (section_path or []) if clean_text(item)]
    section_title = sections[-1] if sections else ""
    breadcrumb_parts = [clean_text(metadata.get("title")), *sections]
    return {
        "source_url": source_url,
        "canonical_url": metadata.get("canonical_url") or normalize_url(source_url),
        "canonical_family_url": metadata.get("canonical_family_url") or locale_family_url(source_url),
        "page_title": metadata.get("title") or "",
        "section_title": section_title,
        "heading_path": sections,
        "breadcrumb": " > ".join(part for part in breadcrumb_parts if part),
        "language": metadata.get("language") or "",
        "normalized_path": metadata.get("normalized_path") or "",
        "content_hash": metadata.get("content_hash") or "",
        "chunk_index": chunk_index,
    }
