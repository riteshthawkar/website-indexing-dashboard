"""Clean the crawl page graph and bridge it to Representation V2 evidence IDs.

The crawl graph is intentionally broad: it contains every discovered URL,
including external destinations.  Retrieval needs a smaller, auditable graph
whose page nodes are backed by Page Cards and whose hierarchy reaches document
revisions, sections, and (once available) chunks.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence
from urllib.parse import urlsplit

from pipeline.core.page_cards import normalize_url
from pipeline.core.representation_v2 import now_iso, stable_representation_id


PAGE_GRAPH_BRIDGE_SCHEMA_VERSION = "mbzuai.page_graph_bridge.v2"
PAGE_GRAPH_BRIDGE_KIND = "representation_v2_page_graph_bridge"
NAVIGATION_CATALOG_SCHEMA_VERSION = "mbzuai.navigation_catalog.v2"
NAVIGATION_ACTION_TYPES = frozenset(
    {
        "apply",
        "register",
        "contact",
        "download",
        "email",
        "telephone",
        "login",
        "search",
        "submit_form",
    }
)


def _clean_text(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split()).strip()


def _normalized_url(value: Any) -> str:
    return normalize_url(value, keep_fragment=False)


def _normalized_heading(value: Any) -> str:
    text = _clean_text(value).casefold()
    return re.sub(r"[^\w\u0600-\u06ff]+", " ", text, flags=re.UNICODE).strip()


def _safe_navigation_target(value: Any) -> bool:
    raw = _clean_text(value)
    if not raw or any(character in raw for character in ("\r", "\n", "\x00")):
        return False
    lowered = raw.casefold()
    if lowered.startswith("mailto:"):
        address = raw.split(":", 1)[1].strip()
        return bool(address and "@" in address and not any(ch.isspace() for ch in address))
    if lowered.startswith("tel:"):
        number = raw.split(":", 1)[1].strip()
        return bool(number and re.fullmatch(r"[+()0-9 .-]+", number))
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return False
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.hostname)


def _official_navigation_host(value: Any) -> bool:
    host = _clean_text(value).casefold().removeprefix("www.")
    return bool(
        host == "mbzuai.ac.ae"
        or host.endswith(".mbzuai.ac.ae")
        or host == "ifm.ai"
        or host.endswith(".ifm.ai")
    )


def _navigation_official_target(action: Mapping[str, Any]) -> tuple[bool, str]:
    if bool(action.get("official_target")):
        return True, "representation_v2"
    target = _clean_text(
        action.get("canonical_target_url") or action.get("target_url")
    )
    lowered = target.casefold()
    if lowered.startswith("mailto:"):
        address = target.split(":", 1)[1].strip()
        domain = address.rsplit("@", 1)[-1] if "@" in address else ""
        if _safe_navigation_target(target) and _official_navigation_host(domain):
            return True, "official_email_domain"
    if lowered.startswith("tel:") and _safe_navigation_target(target):
        # The number is extracted from a frozen, content-backed official Page
        # Card. Telephone URIs have no host to independently classify.
        return True, "official_page_published_telephone"
    return False, "representation_v2"


def _path_key(value: Any) -> str:
    raw = _clean_text(value)
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return raw


def _unique_index(claims: Mapping[str, set[str]]) -> Dict[str, str]:
    return {
        key: next(iter(values))
        for key, values in claims.items()
        if key and len(values) == 1
    }


def _append_edge(
    edges: MutableMapping[tuple[str, str, str], Dict[str, Any]],
    *,
    edge_type: str,
    source_id: str,
    target_id: str,
    properties: Mapping[str, Any] | None = None,
) -> None:
    if not source_id or not target_id or source_id == target_id:
        return
    key = (edge_type, source_id, target_id)
    incoming = dict(properties or {})
    existing = edges.get(key)
    if existing is None:
        edges[key] = {
            "edge_id": stable_representation_id(
                "page-graph-edge", edge_type, source_id, target_id
            ),
            "edge_type": edge_type,
            "source_id": source_id,
            "target_id": target_id,
            "properties": incoming,
        }
        return

    existing_properties = existing.setdefault("properties", {})
    for property_name in ("anchor_texts", "rels", "sources"):
        values = [
            _clean_text(value)
            for value in [
                *(existing_properties.get(property_name) or []),
                *(incoming.get(property_name) or []),
            ]
            if _clean_text(value)
        ]
        if values:
            existing_properties[property_name] = sorted(set(values))[:100]
    for property_name, value in incoming.items():
        if property_name not in {"anchor_texts", "rels", "sources"}:
            existing_properties.setdefault(property_name, value)


def _page_indexes(
    page_cards: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, str], Dict[str, str]]:
    primary: Dict[str, str] = {}
    fallback_claims: Dict[str, set[str]] = defaultdict(set)
    for page in page_cards:
        page_id = _clean_text(page.get("page_card_id"))
        source_url = _normalized_url(page.get("source_url"))
        if page_id and source_url:
            existing = primary.get(source_url)
            if existing and existing != page_id:
                raise ValueError(f"Multiple Page Cards claim source URL {source_url}")
            primary[source_url] = page_id
        for value in (
            page.get("canonical_url"),
            page.get("canonical_family_url"),
        ):
            normalized = _normalized_url(value)
            if normalized and page_id:
                fallback_claims[normalized].add(page_id)
    return primary, _unique_index(fallback_claims)


def _resolve_page_id(
    value: Any,
    *,
    primary: Mapping[str, str],
    fallback: Mapping[str, str],
) -> str:
    normalized = _normalized_url(value)
    return primary.get(normalized) or fallback.get(normalized) or ""


def _document_indexes(
    documents: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, Mapping[str, Any]], Dict[str, str], Dict[str, str]]:
    by_revision: Dict[str, Mapping[str, Any]] = {}
    path_claims: Dict[str, set[str]] = defaultdict(set)
    url_claims: Dict[str, set[str]] = defaultdict(set)
    for document in documents:
        revision_id = _clean_text(document.get("document_revision_id"))
        if not revision_id:
            continue
        if revision_id in by_revision:
            raise ValueError(f"Duplicate document revision ID {revision_id}")
        by_revision[revision_id] = document
        markdown_path = _path_key(document.get("markdown_path"))
        if markdown_path:
            path_claims[markdown_path].add(revision_id)
        for value in (
            document.get("source_url"),
            document.get("canonical_url"),
            document.get("canonical_family_url"),
        ):
            normalized = _normalized_url(value)
            if normalized:
                url_claims[normalized].add(revision_id)
    return by_revision, _unique_index(path_claims), _unique_index(url_claims)


def _resolve_document_revision_id(
    chunk: Mapping[str, Any],
    *,
    documents_by_revision: Mapping[str, Mapping[str, Any]],
    document_revision_by_path: Mapping[str, str],
    document_revision_by_url: Mapping[str, str],
) -> str:
    explicit = _clean_text(chunk.get("document_revision_id"))
    if explicit:
        return explicit if explicit in documents_by_revision else ""
    for key in ("source_markdown_path", "markdown_path"):
        path = _path_key(chunk.get(key))
        if path and path in document_revision_by_path:
            return document_revision_by_path[path]
    normalized_url = _normalized_url(chunk.get("source_url"))
    if normalized_url:
        return document_revision_by_url.get(normalized_url, "")
    return ""


def _resolve_page_section_ids(
    chunk: Mapping[str, Any],
    *,
    page_id: str,
    sections_by_id: Mapping[str, Mapping[str, Any]],
    section_ids_by_page_heading: Mapping[tuple[str, str], Sequence[str]],
) -> List[str]:
    resolved: List[str] = []
    explicit_values = [
        *(chunk.get("page_section_ids") or []),
        *(chunk.get("section_ids") or []),
        chunk.get("section_id"),
    ]
    for value in explicit_values:
        explicit = _clean_text(value)
        section = sections_by_id.get(explicit)
        if explicit and section and section.get("page_card_id") == page_id:
            resolved.append(explicit)
    section_path = [
        _clean_text(value)
        for value in (chunk.get("section_path") or [])
        if _clean_text(value)
    ]
    for heading in [chunk.get("heading"), *reversed(section_path)]:
        normalized_heading = _normalized_heading(heading)
        if page_id and normalized_heading:
            resolved.extend(
                section_ids_by_page_heading.get((page_id, normalized_heading)) or []
            )
    return list(dict.fromkeys(resolved))


def _resolve_section_id(
    chunk: Mapping[str, Any],
    *,
    page_id: str,
    sections_by_id: Mapping[str, Mapping[str, Any]],
    section_ids_by_page_heading: Mapping[tuple[str, str], Sequence[str]],
) -> str:
    """Return a raw Page Card section only when the match is unambiguous."""

    candidates = _resolve_page_section_ids(
        chunk,
        page_id=page_id,
        sections_by_id=sections_by_id,
        section_ids_by_page_heading=section_ids_by_page_heading,
    )
    return candidates[0] if len(candidates) == 1 else ""


def _chunk_records(chunk_index: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if chunk_index is None:
        return []
    chunks = chunk_index.get("chunks")
    if not isinstance(chunks, list):
        raise ValueError("Chunk index must contain a chunks list")
    records = [dict(value) for value in chunks if isinstance(value, Mapping)]
    if len(records) != len(chunks):
        raise ValueError("Chunk index contains non-object chunk records")
    return records


def build_page_graph_bridge(
    *,
    representation_bundle: Mapping[str, Any],
    crawl_graph: Mapping[str, Any],
    chunk_index: Mapping[str, Any] | None = None,
    require_chunk_index: bool = False,
    derive_document_sections: bool = False,
) -> Dict[str, Any]:
    """Build a deterministic, retrieval-safe Page Card graph bridge."""

    documents = [
        dict(value)
        for value in representation_bundle.get("documents") or []
        if isinstance(value, Mapping)
    ]
    page_cards = [
        dict(value)
        for value in representation_bundle.get("page_cards") or []
        if isinstance(value, Mapping)
    ]
    actions = [
        dict(value)
        for value in representation_bundle.get("actions") or []
        if isinstance(value, Mapping)
    ]
    raw_edges = crawl_graph.get("edges")
    if not documents or not page_cards:
        raise ValueError("Representation V2 bundle must contain documents and Page Cards")
    if not isinstance(raw_edges, list):
        raise ValueError("Crawl graph must contain an edges list")

    page_by_id = {
        _clean_text(page.get("page_card_id")): page
        for page in page_cards
        if _clean_text(page.get("page_card_id"))
    }
    action_by_id = {
        _clean_text(action.get("action_id")): action
        for action in actions
        if _clean_text(action.get("action_id"))
    }
    if len(page_by_id) != len(page_cards):
        raise ValueError("Representation V2 Page Card IDs are missing or duplicated")
    if len(action_by_id) != len(actions):
        raise ValueError("Representation V2 action IDs are missing or duplicated")

    page_primary, page_fallback = _page_indexes(page_cards)
    (
        documents_by_revision,
        document_revision_by_path,
        document_revision_by_url,
    ) = _document_indexes(documents)

    sections: List[Dict[str, Any]] = []
    sections_by_id: Dict[str, Dict[str, Any]] = {}
    section_ids_by_page_heading: Dict[tuple[str, str], List[str]] = defaultdict(list)
    duplicate_section_instances_removed = 0
    for page in page_cards:
        page_id = _clean_text(page.get("page_card_id"))
        revision_id = _clean_text(page.get("document_revision_id"))
        for section in page.get("sections") or []:
            if not isinstance(section, Mapping):
                continue
            section_id = _clean_text(section.get("section_id"))
            if not section_id:
                raise ValueError("Page section ID is missing")
            record = {
                "section_id": section_id,
                "section_kind": "page_heading",
                "page_card_id": page_id,
                "document_revision_id": revision_id,
                "heading": _clean_text(section.get("heading")),
                "level": int(section.get("level") or 1),
                "html_locator": _clean_text(section.get("html_locator")),
                "evidence_id": _clean_text(section.get("evidence_id")),
                "section_path": [_clean_text(section.get("heading"))],
                "page_numbers": [],
                "representation_section_ids": [section_id],
                "chunk_ids": [],
            }
            existing_section = sections_by_id.get(section_id)
            if existing_section is not None:
                comparable_fields = (
                    "page_card_id",
                    "document_revision_id",
                    "heading",
                    "level",
                    "html_locator",
                    "evidence_id",
                )
                if all(
                    existing_section.get(field) == record.get(field)
                    for field in comparable_fields
                ):
                    # Some source pages repeat the same invalid HTML id/heading
                    # block. Representation V2 preserves both observations;
                    # the graph bridge must expose one unambiguous section node.
                    duplicate_section_instances_removed += 1
                    continue
                raise ValueError(
                    f"Section ID {section_id} resolves to different Page Card sections"
                )
            sections.append(record)
            sections_by_id[section_id] = record
            normalized_heading = _normalized_heading(record["heading"])
            if normalized_heading:
                section_ids_by_page_heading[(page_id, normalized_heading)].append(
                    section_id
                )

    edges: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    page_chunk_ids: Dict[str, List[str]] = defaultdict(list)
    document_chunk_ids: Dict[str, List[str]] = defaultdict(list)
    document_section_ids: Dict[str, List[str]] = defaultdict(list)
    page_document_section_ids: Dict[str, List[str]] = defaultdict(list)
    chunk_bridge_records: List[Dict[str, Any]] = []
    chunk_issues: List[Dict[str, Any]] = []
    derived_document_section_count = 0

    for document in documents:
        revision_id = _clean_text(document.get("document_revision_id"))
        for page_id in document.get("page_card_ids") or []:
            page_id = _clean_text(page_id)
            if page_id in page_by_id:
                _append_edge(
                    edges,
                    edge_type="PAGE_REPRESENTS_DOCUMENT",
                    source_id=page_id,
                    target_id=revision_id,
                )

    for section in sections:
        _append_edge(
            edges,
            edge_type="PAGE_HAS_SECTION",
            source_id=section["page_card_id"],
            target_id=section["section_id"],
        )
        _append_edge(
            edges,
            edge_type="DOCUMENT_HAS_SECTION",
            source_id=section["document_revision_id"],
            target_id=section["section_id"],
        )

    for action in actions:
        action_id = _clean_text(action.get("action_id"))
        page_id = _clean_text(action.get("page_card_id"))
        if page_id not in page_by_id:
            raise ValueError(f"Action {action_id} references unknown Page Card {page_id}")
        _append_edge(
            edges,
            edge_type="PAGE_HAS_ACTION",
            source_id=page_id,
            target_id=action_id,
            properties={"retrieval_eligible": bool(action.get("retrieval_eligible"))},
        )
        target_page_id = _resolve_page_id(
            action.get("canonical_target_url") or action.get("target_url"),
            primary=page_primary,
            fallback=page_fallback,
        )
        if target_page_id:
            _append_edge(
                edges,
                edge_type="ACTION_TARGETS_PAGE",
                source_id=action_id,
                target_id=target_page_id,
            )

    dropped_graph_edges = Counter()
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, Mapping):
            dropped_graph_edges["non_object"] += 1
            continue
        source_id = _resolve_page_id(
            raw_edge.get("source_url"), primary=page_primary, fallback=page_fallback
        )
        target_id = _resolve_page_id(
            raw_edge.get("target_url"), primary=page_primary, fallback=page_fallback
        )
        if not source_id:
            dropped_graph_edges["source_not_page_card"] += 1
            continue
        if not target_id:
            dropped_graph_edges["target_not_page_card"] += 1
            continue
        if source_id == target_id:
            dropped_graph_edges["self_loop"] += 1
            continue
        properties = raw_edge.get("properties") if isinstance(raw_edge.get("properties"), Mapping) else {}
        _append_edge(
            edges,
            edge_type="PAGE_LINKS_TO_PAGE",
            source_id=source_id,
            target_id=target_id,
            properties={
                "link_type": _clean_text(properties.get("link_type")) or "internal",
                "anchor_texts": list(properties.get("anchor_texts") or []),
                "rels": list(properties.get("rels") or []),
                "sources": list(properties.get("sources") or []),
            },
        )

    chunks = _chunk_records(chunk_index)
    if require_chunk_index and not chunks:
        raise ValueError("A non-empty chunk index is required for the production graph bridge")
    for position, chunk in enumerate(chunks):
        chunk_id = _clean_text(chunk.get("chunk_id") or chunk.get("id"))
        if not chunk_id:
            chunk_issues.append(
                {"code": "missing_chunk_id", "position": position}
            )
            continue
        revision_id = _resolve_document_revision_id(
            chunk,
            documents_by_revision=documents_by_revision,
            document_revision_by_path=document_revision_by_path,
            document_revision_by_url=document_revision_by_url,
        )
        document = documents_by_revision.get(revision_id, {})
        explicit_page_id = _clean_text(chunk.get("page_card_id"))
        if explicit_page_id and explicit_page_id not in page_by_id:
            explicit_page_id = ""
        page_candidates = [
            page_id
            for page_id in (document.get("page_card_ids") or [])
            if _clean_text(page_id) in page_by_id
        ]
        page_id = explicit_page_id
        if not page_id:
            page_id = _resolve_page_id(
                chunk.get("source_url"), primary=page_primary, fallback=page_fallback
            )
        if not page_id and len(page_candidates) == 1:
            page_id = _clean_text(page_candidates[0])
        page_section_ids = _resolve_page_section_ids(
            chunk,
            page_id=page_id,
            sections_by_id=sections_by_id,
            section_ids_by_page_heading=section_ids_by_page_heading,
        )
        section_path = [
            _clean_text(value)
            for value in (chunk.get("section_path") or [])
            if _clean_text(value)
        ]
        page_numbers = sorted(
            {
                int(value)
                for value in (chunk.get("page_numbers") or [])
                if value not in (None, "")
            }
        )
        section_id = ""
        if derive_document_sections and revision_id and (section_path or page_numbers):
            section_id = stable_representation_id(
                "document-section",
                revision_id,
                json.dumps(
                    [section_path, page_numbers],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            section_record = sections_by_id.get(section_id)
            if section_record is None:
                derived_document_section_count += 1
                heading = section_path[-1] if section_path else (
                    f"Pages {page_numbers[0]}-{page_numbers[-1]}"
                    if len(page_numbers) > 1
                    else f"Page {page_numbers[0]}"
                )
                section_record = {
                    "section_id": section_id,
                    "section_kind": "document_section",
                    "page_card_id": page_id,
                    "document_revision_id": revision_id,
                    "heading": heading,
                    "level": max(1, min(len(section_path), 6)),
                    "html_locator": "",
                    "evidence_id": "",
                    "section_path": section_path,
                    "page_numbers": page_numbers,
                    "representation_section_ids": list(page_section_ids),
                    "chunk_ids": [],
                }
                sections.append(section_record)
                sections_by_id[section_id] = section_record
                document_section_ids[revision_id].append(section_id)
                _append_edge(
                    edges,
                    edge_type="DOCUMENT_HAS_SECTION",
                    source_id=revision_id,
                    target_id=section_id,
                )
                if page_id:
                    page_document_section_ids[page_id].append(section_id)
                    _append_edge(
                        edges,
                        edge_type="PAGE_HAS_DOCUMENT_SECTION",
                        source_id=page_id,
                        target_id=section_id,
                    )
                for page_section_id in page_section_ids:
                    _append_edge(
                        edges,
                        edge_type="DOCUMENT_SECTION_ALIGNS_PAGE_SECTION",
                        source_id=section_id,
                        target_id=page_section_id,
                    )
        elif page_section_ids:
            section_id = page_section_ids[0] if len(page_section_ids) == 1 else ""
        record = {
            "chunk_id": chunk_id,
            "document_revision_id": revision_id,
            "page_card_id": page_id,
            "section_id": section_id,
            "page_section_ids": list(page_section_ids),
            "source_url": _clean_text(chunk.get("source_url")),
            "source_markdown_path": _clean_text(
                chunk.get("source_markdown_path") or chunk.get("markdown_path")
            ),
            "section_path": section_path,
            "page_numbers": page_numbers,
        }
        chunk_bridge_records.append(record)
        if not revision_id:
            chunk_issues.append(
                {"code": "chunk_document_unresolved", "chunk_id": chunk_id}
            )
            continue
        document_chunk_ids[revision_id].append(chunk_id)
        _append_edge(
            edges,
            edge_type="DOCUMENT_HAS_CHUNK",
            source_id=revision_id,
            target_id=chunk_id,
        )
        if page_id:
            page_chunk_ids[page_id].append(chunk_id)
            _append_edge(
                edges,
                edge_type="PAGE_HAS_CHUNK",
                source_id=page_id,
                target_id=chunk_id,
            )
        if section_id:
            sections_by_id[section_id]["chunk_ids"].append(chunk_id)
            _append_edge(
                edges,
                edge_type="SECTION_HAS_CHUNK",
                source_id=section_id,
                target_id=chunk_id,
            )
        elif section_path or page_numbers:
            chunk_issues.append(
                {"code": "chunk_section_unresolved", "chunk_id": chunk_id}
            )
        if str(document.get("source_type") or "").casefold() == "webpage" and not page_id:
            chunk_issues.append(
                {"code": "web_chunk_page_unresolved", "chunk_id": chunk_id}
            )

    outgoing_page_ids_by_page: Dict[str, set[str]] = defaultdict(set)
    for edge in edges.values():
        if edge["edge_type"] == "PAGE_LINKS_TO_PAGE":
            outgoing_page_ids_by_page[edge["source_id"]].add(edge["target_id"])

    page_records: List[Dict[str, Any]] = []
    for page in sorted(page_cards, key=lambda value: _clean_text(value.get("page_card_id"))):
        page_id = _clean_text(page.get("page_card_id"))
        outgoing_page_ids = sorted(outgoing_page_ids_by_page.get(page_id, set()))
        page_heading_section_ids = list(
            dict.fromkeys(
                _clean_text(value.get("section_id"))
                for value in page.get("sections") or []
                if isinstance(value, Mapping) and _clean_text(value.get("section_id"))
            )
        )
        derived_section_ids = sorted(
            set(page_document_section_ids.get(page_id, []))
        )
        page_records.append(
            {
                "page_card_id": page_id,
                "document_revision_id": _clean_text(page.get("document_revision_id")),
                "source_url": _clean_text(page.get("source_url")),
                "canonical_url": _clean_text(page.get("canonical_url")),
                "canonical_family_url": _clean_text(
                    page.get("canonical_family_url")
                ),
                "canonical_family_url": _clean_text(page.get("canonical_family_url")),
                "title": _clean_text(page.get("title")),
                "purpose_summary": _clean_text(page.get("purpose_summary")),
                "language": _clean_text(page.get("language")),
                "page_type": _clean_text(page.get("page_type")),
                "content_backed": bool(page.get("content_backed")),
                "topic_labels": [
                    _clean_text(value.get("label"))
                    for value in page.get("topics") or []
                    if isinstance(value, Mapping) and _clean_text(value.get("label"))
                ],
                "audience_labels": [
                    _clean_text(value.get("label"))
                    for value in page.get("audiences") or []
                    if isinstance(value, Mapping) and _clean_text(value.get("label"))
                ],
                "section_ids": [*page_heading_section_ids, *derived_section_ids],
                "page_heading_section_ids": page_heading_section_ids,
                "document_section_ids": derived_section_ids,
                "chunk_ids": sorted(set(page_chunk_ids.get(page_id, []))),
                "action_ids": [
                    _clean_text(value) for value in page.get("action_ids") or [] if _clean_text(value)
                ],
                "retrieval_action_ids": [
                    _clean_text(value)
                    for value in page.get("retrieval_action_ids") or []
                    if _clean_text(value)
                ],
                "outgoing_page_card_ids": outgoing_page_ids,
            }
        )

    all_section_ids_by_document: Dict[str, List[str]] = defaultdict(list)
    for section in sections:
        revision_id = _clean_text(section.get("document_revision_id"))
        section_id = _clean_text(section.get("section_id"))
        if revision_id and section_id:
            all_section_ids_by_document[revision_id].append(section_id)

    document_records = [
        {
            "document_id": _clean_text(document.get("document_id")),
            "document_revision_id": _clean_text(document.get("document_revision_id")),
            "corpus_record_id": _clean_text(document.get("corpus_record_id")),
            "source_type": _clean_text(document.get("source_type")),
            "language": _clean_text(document.get("language")),
            "title": _clean_text(document.get("title")),
            "source_url": _clean_text(document.get("source_url")),
            "markdown_path": _clean_text(document.get("markdown_path")),
            "page_card_ids": sorted(
                _clean_text(value)
                for value in document.get("page_card_ids") or []
                if _clean_text(value)
            ),
            "section_ids": sorted(
                set(
                    all_section_ids_by_document.get(
                        _clean_text(document.get("document_revision_id")), []
                    )
                )
            ),
            "document_section_ids": sorted(
                set(
                    document_section_ids.get(
                        _clean_text(document.get("document_revision_id")), []
                    )
                )
            ),
            "chunk_ids": sorted(
                set(
                    document_chunk_ids.get(
                        _clean_text(document.get("document_revision_id")), []
                    )
                )
            ),
        }
        for document in sorted(
            documents, key=lambda value: _clean_text(value.get("document_revision_id"))
        )
    ]

    action_records = []
    for action in sorted(actions, key=lambda value: _clean_text(value.get("action_id"))):
        official_target, official_target_method = _navigation_official_target(action)
        target_page_id = _resolve_page_id(
            action.get("canonical_target_url") or action.get("target_url"),
            primary=page_primary,
            fallback=page_fallback,
        )
        action_records.append(
            {
                "action_id": _clean_text(action.get("action_id")),
                "page_card_id": _clean_text(action.get("page_card_id")),
                "source_section_id": _clean_text(action.get("source_section_id")),
                "source_section_heading": _clean_text(
                    action.get("source_section_heading")
                ),
                "label": _clean_text(action.get("label")),
                "context_label": _clean_text(action.get("context_label")),
                "action_type": _clean_text(action.get("action_type")),
                "target_url": _clean_text(action.get("target_url")),
                "canonical_target_url": _clean_text(
                    action.get("canonical_target_url")
                ),
                "target_kind": _clean_text(action.get("target_kind")),
                "target_page_card_id": target_page_id,
                "official_target": official_target,
                "official_target_method": official_target_method,
                "retrieval_eligible": bool(action.get("retrieval_eligible")),
                "opens_new_window": bool(action.get("opens_new_window")),
                "authentication_requirement": _clean_text(
                    action.get("authentication_requirement")
                ),
                "evidence_ids": [
                    _clean_text(value.get("evidence_id"))
                    for value in action.get("evidence") or []
                    if isinstance(value, Mapping) and _clean_text(value.get("evidence_id"))
                ],
            }
        )

    chunk_issue_counts = Counter(value["code"] for value in chunk_issues)
    chunk_ids = [_clean_text(value.get("chunk_id")) for value in chunk_bridge_records]
    duplicate_chunk_ids = [
        value for value, count in Counter(chunk_ids).items() if value and count > 1
    ]
    if duplicate_chunk_ids:
        chunk_issue_counts["duplicate_chunk_id"] += len(duplicate_chunk_ids)
        chunk_issues.extend(
            {"code": "duplicate_chunk_id", "chunk_id": value}
            for value in duplicate_chunk_ids[:100]
        )
    chunk_bridge_ready = bool(chunks) and not chunk_issue_counts
    core_gates = {
        "representation_version_supported": representation_bundle.get("schema_version")
        == "mbzuai.representation.v2",
        "page_ids_unique": len(page_by_id) == len(page_cards),
        "action_ids_unique": len(action_by_id) == len(actions),
        "all_sections_linked_to_pages": all(
            value.get("section_kind") != "page_heading"
            or value.get("page_card_id") in page_by_id
            for value in sections
        ),
        "all_document_sections_linked_to_documents": all(
            value.get("section_kind") != "document_section"
            or value.get("document_revision_id") in documents_by_revision
            for value in sections
        ),
        "all_content_backed_page_sections_linked_to_documents": all(
            value.get("section_kind") != "page_heading"
            or not bool(
                page_by_id.get(_clean_text(value.get("page_card_id")), {}).get(
                    "content_backed"
                )
            )
            or value.get("document_revision_id") in documents_by_revision
            for value in sections
        ),
        "all_actions_linked_to_pages": all(
            value.get("page_card_id") in page_by_id for value in action_records
        ),
        "no_external_discovered_page_nodes": True,
        "no_page_self_loops": all(
            not (
                value["edge_type"] == "PAGE_LINKS_TO_PAGE"
                and value["source_id"] == value["target_id"]
            )
            for value in edges.values()
        ),
    }
    chunk_gates = {
        "chunk_index_supplied": bool(chunks),
        "chunk_ids_unique": not duplicate_chunk_ids,
        "all_chunks_linked_to_documents": bool(chunks)
        and not chunk_issue_counts.get("chunk_document_unresolved"),
        "all_web_chunks_linked_to_pages": bool(chunks)
        and not chunk_issue_counts.get("web_chunk_page_unresolved"),
        "all_section_scoped_chunks_linked_to_sections": bool(chunks)
        and not chunk_issue_counts.get("chunk_section_unresolved"),
        "chunk_bridge_ready": chunk_bridge_ready,
    }
    passed = all(core_gates.values()) and (
        chunk_bridge_ready if chunks or require_chunk_index else True
    )
    graph_edges = sorted(
        edges.values(),
        key=lambda value: (
            value["edge_type"], value["source_id"], value["target_id"]
        ),
    )
    stats = {
        "documents": len(document_records),
        "pages": len(page_records),
        "sections": len(sections),
        "page_heading_sections": len(sections) - derived_document_section_count,
        "document_sections": derived_document_section_count,
        "discovery_only_page_heading_sections": sum(
            value.get("section_kind") == "page_heading"
            and not _clean_text(value.get("document_revision_id"))
            for value in sections
        ),
        "chunks": len(chunk_bridge_records),
        "actions": len(action_records),
        "retrieval_eligible_actions": sum(
            bool(value.get("retrieval_eligible")) for value in action_records
        ),
        "input_crawl_nodes": len(crawl_graph.get("nodes") or []),
        "input_crawl_edges": len(raw_edges),
        "clean_page_link_edges": sum(
            value["edge_type"] == "PAGE_LINKS_TO_PAGE" for value in graph_edges
        ),
        "bridge_edges": len(graph_edges),
        "dropped_crawl_edges": sum(dropped_graph_edges.values()),
        "dropped_crawl_edges_by_reason": dict(sorted(dropped_graph_edges.items())),
        "chunk_issue_counts": dict(sorted(chunk_issue_counts.items())),
        "duplicate_section_instances_removed": duplicate_section_instances_removed,
    }
    return {
        "schema_version": PAGE_GRAPH_BRIDGE_SCHEMA_VERSION,
        "kind": PAGE_GRAPH_BRIDGE_KIND,
        "generated_at": now_iso(),
        "source_snapshot": {
            "representation_schema_version": representation_bundle.get(
                "schema_version"
            ),
            "representation_generated_at": representation_bundle.get("generated_at"),
            "crawl_graph_schema_version": crawl_graph.get("schema_version"),
            "crawl_graph_type": crawl_graph.get("graph_type"),
            "chunk_index_version": (
                chunk_index.get("version") if isinstance(chunk_index, Mapping) else None
            ),
            "derive_document_sections": bool(derive_document_sections),
        },
        "documents": document_records,
        "pages": page_records,
        "sections": sorted(sections, key=lambda value: value["section_id"]),
        "chunks": sorted(chunk_bridge_records, key=lambda value: value["chunk_id"]),
        "actions": action_records,
        "edges": graph_edges,
        "stats": stats,
        "coverage": {
            "passed": passed,
            "status": (
                "ready"
                if passed
                else "awaiting_chunk_index"
                if not chunks
                else "failed"
            ),
            "require_chunk_index": bool(require_chunk_index),
            "core_gates": core_gates,
            "chunk_gates": chunk_gates,
            "chunk_issue_samples": chunk_issues[:200],
        },
    }


def validate_page_graph_bridge(
    bridge: Mapping[str, Any], *, require_chunk_index: bool = False
) -> Dict[str, Any]:
    """Validate ID integrity independently of the builder implementation."""

    issues: List[Dict[str, str]] = []
    if bridge.get("schema_version") != PAGE_GRAPH_BRIDGE_SCHEMA_VERSION:
        issues.append({"code": "schema_version", "message": "Unexpected schema version"})
    if bridge.get("kind") != PAGE_GRAPH_BRIDGE_KIND:
        issues.append({"code": "kind", "message": "Unexpected bridge kind"})

    collections = {
        "document": (bridge.get("documents") or [], "document_revision_id"),
        "page": (bridge.get("pages") or [], "page_card_id"),
        "section": (bridge.get("sections") or [], "section_id"),
        "chunk": (bridge.get("chunks") or [], "chunk_id"),
        "action": (bridge.get("actions") or [], "action_id"),
    }
    node_ids: set[str] = set()
    ids_by_kind: Dict[str, set[str]] = {}
    for kind, (values, id_field) in collections.items():
        ids = [_clean_text(value.get(id_field)) for value in values if isinstance(value, Mapping)]
        ids_by_kind[kind] = set(ids)
        if any(not value for value in ids):
            issues.append({"code": "missing_id", "message": f"Missing {kind} ID"})
        duplicates = [value for value, count in Counter(ids).items() if value and count > 1]
        if duplicates:
            issues.append(
                {
                    "code": "duplicate_id",
                    "message": f"Duplicate {kind} IDs: {duplicates[:5]}",
                }
            )
        overlap = node_ids & set(ids)
        if overlap:
            issues.append(
                {
                    "code": "cross_type_id_collision",
                    "message": f"Cross-type IDs collide: {sorted(overlap)[:5]}",
                }
            )
        node_ids.update(ids)

    for edge in bridge.get("edges") or []:
        if not isinstance(edge, Mapping):
            issues.append({"code": "edge_schema", "message": "Non-object edge"})
            continue
        source_id = _clean_text(edge.get("source_id"))
        target_id = _clean_text(edge.get("target_id"))
        if source_id not in node_ids or target_id not in node_ids:
            issues.append(
                {
                    "code": "orphan_edge",
                    "message": f"Unresolved edge {source_id} -> {target_id}",
                }
            )

    for page in bridge.get("pages") or []:
        if not isinstance(page, Mapping):
            continue
        revision_id = _clean_text(page.get("document_revision_id"))
        if bool(page.get("content_backed")) and revision_id not in ids_by_kind["document"]:
            issues.append(
                {
                    "code": "page_document_link",
                    "message": f"Page references unknown revision {revision_id}",
                }
            )
        for section_id in page.get("section_ids") or []:
            if _clean_text(section_id) not in ids_by_kind["section"]:
                issues.append(
                    {
                        "code": "page_section_link",
                        "message": f"Page references unknown section {section_id}",
                    }
                )
    for section in bridge.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        revision_id = _clean_text(section.get("document_revision_id"))
        page_id = _clean_text(section.get("page_card_id"))
        section_kind = _clean_text(section.get("section_kind"))
        if (
            section_kind == "document_section"
            and revision_id not in ids_by_kind["document"]
        ) or (
            section_kind != "document_section"
            and revision_id
            and revision_id not in ids_by_kind["document"]
        ):
            issues.append(
                {
                    "code": "section_document_link",
                    "message": f"Section references unknown revision {revision_id}",
                }
            )
        if page_id and page_id not in ids_by_kind["page"]:
            issues.append(
                {
                    "code": "section_page_link",
                    "message": f"Section references unknown page {page_id}",
                }
            )
    for chunk in bridge.get("chunks") or []:
        if not isinstance(chunk, Mapping):
            continue
        chunk_id = _clean_text(chunk.get("chunk_id"))
        revision_id = _clean_text(chunk.get("document_revision_id"))
        page_id = _clean_text(chunk.get("page_card_id"))
        section_id = _clean_text(chunk.get("section_id"))
        if revision_id not in ids_by_kind["document"]:
            issues.append(
                {
                    "code": "chunk_document_link",
                    "message": f"Chunk {chunk_id} references unknown revision {revision_id}",
                }
            )
        if page_id and page_id not in ids_by_kind["page"]:
            issues.append(
                {
                    "code": "chunk_page_link",
                    "message": f"Chunk {chunk_id} references unknown page {page_id}",
                }
            )
        if section_id and section_id not in ids_by_kind["section"]:
            issues.append(
                {
                    "code": "chunk_section_link",
                    "message": f"Chunk {chunk_id} references unknown section {section_id}",
                }
            )
    if require_chunk_index and not ids_by_kind["chunk"]:
        issues.append(
            {"code": "chunk_index_required", "message": "Bridge contains no chunks"}
        )
    return {
        "passed": not issues,
        "issue_count": len(issues),
        "issue_counts": dict(sorted(Counter(value["code"] for value in issues).items())),
        "issue_samples": issues[:200],
        "counts": {kind: len(ids) for kind, ids in ids_by_kind.items()},
    }


def build_navigation_catalog(bridge: Mapping[str, Any]) -> Dict[str, Any]:
    """Project the full audit graph into a compact read-only runtime catalog."""

    sections_by_page: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for section in bridge.get("sections") or []:
        if not isinstance(section, Mapping):
            continue
        page_id = _clean_text(section.get("page_card_id"))
        if not page_id:
            continue
        sections_by_page[page_id].append(
            {
                "section_id": _clean_text(section.get("section_id")),
                "section_kind": _clean_text(section.get("section_kind")),
                "heading": _clean_text(section.get("heading")),
                "section_path": list(section.get("section_path") or []),
                "page_numbers": list(section.get("page_numbers") or []),
                "representation_section_ids": list(
                    section.get("representation_section_ids") or []
                ),
                "chunk_ids": [
                    _clean_text(value)
                    for value in section.get("chunk_ids") or []
                    if _clean_text(value)
                ],
            }
        )

    operational_actions = [
        dict(action)
        for action in bridge.get("actions") or []
        if isinstance(action, Mapping)
        and bool(action.get("retrieval_eligible"))
        and bool(action.get("official_target"))
        and _clean_text(action.get("action_type")) in NAVIGATION_ACTION_TYPES
        and _safe_navigation_target(
            action.get("canonical_target_url") or action.get("target_url")
        )
    ]
    operational_action_ids = {
        _clean_text(value.get("action_id")) for value in operational_actions
    }
    pages = []
    for page in bridge.get("pages") or []:
        if not isinstance(page, Mapping):
            continue
        page_id = _clean_text(page.get("page_card_id"))
        page_sections = sorted(
            sections_by_page.get(page_id, []), key=lambda value: value["section_id"]
        )
        pages.append(
            {
                "page_card_id": page_id,
                "document_revision_id": _clean_text(
                    page.get("document_revision_id")
                ),
                "source_url": _clean_text(page.get("source_url")),
                "canonical_url": _clean_text(page.get("canonical_url")),
                "title": _clean_text(page.get("title")),
                "purpose_summary": _clean_text(page.get("purpose_summary")),
                "language": _clean_text(page.get("language")),
                "page_type": _clean_text(page.get("page_type")),
                "topic_labels": list(page.get("topic_labels") or []),
                "audience_labels": list(page.get("audience_labels") or []),
                "sections": page_sections,
                "chunk_ids": list(page.get("chunk_ids") or []),
                "outgoing_page_card_ids": list(
                    page.get("outgoing_page_card_ids") or []
                ),
                "action_ids": [
                    _clean_text(value)
                    for value in page.get("retrieval_action_ids") or []
                    if _clean_text(value) in operational_action_ids
                ],
            }
        )

    chunks = [
        {
            "chunk_id": _clean_text(chunk.get("chunk_id")),
            "document_revision_id": _clean_text(
                chunk.get("document_revision_id")
            ),
            "page_card_id": _clean_text(chunk.get("page_card_id")),
            "section_id": _clean_text(chunk.get("section_id")),
            "page_section_ids": [
                _clean_text(value)
                for value in chunk.get("page_section_ids") or []
                if _clean_text(value)
            ],
        }
        for chunk in bridge.get("chunks") or []
        if isinstance(chunk, Mapping) and _clean_text(chunk.get("chunk_id"))
    ]
    actions = [
        {
            key: action.get(key)
            for key in (
                "action_id",
                "page_card_id",
                "source_section_id",
                "source_section_heading",
                "label",
                "context_label",
                "action_type",
                "target_url",
                "canonical_target_url",
                "target_kind",
                "target_page_card_id",
                "official_target",
                "opens_new_window",
                "authentication_requirement",
                "evidence_ids",
            )
        }
        for action in operational_actions
    ]
    return {
        "schema_version": NAVIGATION_CATALOG_SCHEMA_VERSION,
        "kind": "grounded_navigation_catalog",
        "generated_at": now_iso(),
        "source_bridge_schema_version": bridge.get("schema_version"),
        "source_bridge_status": (bridge.get("coverage") or {}).get("status"),
        "source_bridge_coverage_passed": bool(
            (bridge.get("coverage") or {}).get("passed")
        ),
        "pages": sorted(pages, key=lambda value: value["page_card_id"]),
        "chunks": sorted(chunks, key=lambda value: value["chunk_id"]),
        "actions": sorted(actions, key=lambda value: value["action_id"]),
        "stats": {
            "pages": len(pages),
            "sections": sum(len(value["sections"]) for value in pages),
            "chunks": len(chunks),
            "actions": len(actions),
            "action_types": dict(
                sorted(Counter(value["action_type"] for value in actions).items())
            ),
        },
    }
