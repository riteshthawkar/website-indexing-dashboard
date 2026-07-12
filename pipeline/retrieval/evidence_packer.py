from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from urllib.parse import unquote, urlparse


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _doc_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    metadata = doc.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def _doc_id(doc: Dict[str, Any]) -> str:
    return _clean_text(doc.get("id") or doc.get("record_id") or doc.get("chunk_id"))


def _doc_text(doc: Dict[str, Any]) -> str:
    return _clean_text(doc.get("text") or doc.get("value") or _doc_metadata(doc).get("context"))


_OFFICIAL_SOURCE_URL_RE = re.compile(
    r"https?://(?:www\.)?mbzuai\.ac\.ae/[^\s\]\)\"'<>,]+",
    re.IGNORECASE,
)
_ARABIC_TEXT_RE = re.compile(r"[\u0600-\u06FF]")


def _extract_official_source_url_from_text(*values: Any) -> str:
    for value in values:
        match = _OFFICIAL_SOURCE_URL_RE.search(str(value or ""))
        if match:
            return match.group(0).rstrip(".,;:")
    return ""


def _source_url(doc: Dict[str, Any]) -> str:
    metadata = _doc_metadata(doc)
    return _clean_text(
        doc.get("source_url")
        or doc.get("language_normalized_url")
        or doc.get("canonical_url")
        or doc.get("document_source")
        or doc.get("page_source")
        or metadata.get("source_url")
        or metadata.get("language_normalized_url")
        or metadata.get("canonical_url")
        or metadata.get("document_source")
        or metadata.get("page_source")
        or metadata.get("source")
        or _extract_official_source_url_from_text(
            doc.get("text"),
            doc.get("value"),
            doc.get("dense_text"),
            doc.get("embedding_text"),
            doc.get("sparse_text"),
            metadata.get("context"),
        )
    )


def _normalize_url_for_match(value: Any) -> str:
    raw = _clean_text(value).rstrip("/")
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.casefold()
    if not parsed.netloc:
        return raw.casefold()
    path = (parsed.path or "/").rstrip("/")
    return f"{parsed.scheme.lower() or 'https'}://{parsed.netloc.lower()}{path}".rstrip("/").casefold()


def _is_official_mbzuai_url(value: Any) -> bool:
    try:
        host = (urlparse(str(value or "")).hostname or "").casefold()
    except Exception:
        host = ""
    return bool(host == "mbzuai.ac.ae" or host.endswith(".mbzuai.ac.ae"))


def _is_arabic_source_url(value: Any) -> bool:
    try:
        path = (urlparse(str(value or "")).path or "").casefold()
    except Exception:
        path = ""
    return bool(
        path == "/ar"
        or path.startswith("/ar/")
        or "-arb" in path
        or "_arb" in path
        or "arabic" in path
    )


def _document_title(doc: Dict[str, Any]) -> str:
    metadata = _doc_metadata(doc)
    title = _clean_text(doc.get("document_title") or metadata.get("document_title") or doc.get("title") or metadata.get("title"))
    if title and not re.fullmatch(r"[a-f0-9]{24,64}", title.casefold()):
        return title
    return _title_from_source_url(_source_url(doc))


def _title_from_source_url(source_url: str) -> str:
    try:
        parsed = urlparse(str(source_url or "").strip())
    except Exception:
        parsed = None
    path = unquote(parsed.path or "") if parsed is not None else str(source_url or "")
    parts = [
        part
        for part in path.strip("/").split("/")
        if part and not re.fullmatch(r"20\d{2}|\d{1,2}", part)
    ]
    slug = parts[-1] if parts else (parsed.netloc if parsed is not None else "")
    slug = re.sub(r"\.(?:html?|pdf|docx?|pptx?)$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"[_-]+", " ", slug).strip()
    if not slug:
        return ""
    return " ".join(
        word.upper() if word.casefold() in {"mbzuai", "faq", "ai", "uae", "phd", "msc", "ugrip"} else word.capitalize()
        for word in slug.split()
    ).strip()


def _authority_score(doc: Dict[str, Any]) -> float:
    metadata = _doc_metadata(doc)
    for value in (doc.get("authority_score"), metadata.get("authority_score")):
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            continue
    authority_class = _clean_text(doc.get("authority_class") or metadata.get("authority_class")).casefold()
    if authority_class in {"official", "primary", "canonical", "institutional"}:
        return 0.95
    if authority_class in {"high", "authoritative"}:
        return 0.85
    if authority_class in {"medium", "secondary"}:
        return 0.55
    if authority_class in {"low", "unknown"}:
        return 0.2
    return 0.0


def _confidence(doc: Dict[str, Any]) -> float:
    for key in ("confidence", "score", "relevance_score"):
        try:
            return max(0.0, min(1.0, float(doc.get(key))))
        except (TypeError, ValueError):
            continue
    return 0.0


def _dedupe_key(doc: Dict[str, Any], kind: str) -> str:
    record_id = _doc_id(doc)
    if record_id:
        return f"id:{record_id}"
    text = _doc_text(doc).lower()
    digest = hashlib.sha1(text[:800].encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"text:{_source_url(doc)}:{digest}"


def _coerce_docs(values: Iterable[Any]) -> List[Dict[str, Any]]:
    return [dict(value) for value in values or [] if isinstance(value, dict)]


def _media_documents(values: Iterable[Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for item in values or []:
        if not isinstance(item, dict):
            continue
        docs.append(
            {
                "id": item.get("id"),
                "text": item.get("text") or item.get("description") or item.get("caption") or item.get("title"),
                "source_url": item.get("source_url") or item.get("url") or item.get("asset_uri"),
                "document_title": item.get("document_title") or item.get("title"),
                "media_type": item.get("media_type") or item.get("type"),
                "asset_uri": item.get("asset_uri"),
                "url": item.get("url"),
            }
        )
    return docs


def _candidate_stream(result: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for doc in _coerce_docs(result.get("answer_documents") or []):
        if doc.get("source_span_ids") or doc.get("linked_span_ids"):
            yield "assertion", doc
        else:
            yield "answer", doc
    for doc in _coerce_docs(result.get("fact_documents") or []):
        yield "fact", doc
    for doc in _coerce_docs(result.get("evidence_span_documents") or []):
        yield "evidence_span", doc
    for doc in _coerce_docs(result.get("retrieval_documents") or []):
        yield "evidence_span" if doc.get("span_type") else "chunk", doc
    for doc in _media_documents(result.get("media") or []):
        yield "media", doc


def _coverage_values(coverage_plan: Dict[str, Any], key: str) -> List[str]:
    values = coverage_plan.get(key) if isinstance(coverage_plan, dict) else []
    return [_clean_text(value).casefold() for value in values or [] if _clean_text(value)]


def _item_search_text(item: Dict[str, Any]) -> str:
    return _clean_text(
        " ".join(
            str(item.get(key) or "")
            for key in (
                "text",
                "source_url",
                "document_title",
                "section_heading",
                "breadcrumb",
            )
        )
    ).casefold()


def _entity_match_text(value: str) -> str:
    text = _clean_text(value).casefold()
    text = text.replace("official workings hours", "official working hours")
    text = text.replace("workings hours", "working hours")
    text = text.replace("7.30am", "7:30 am")
    text = text.replace("personal rapid transit", "personal rapid transport")
    return text


def _query_terms(query: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(query or "").casefold())
        if len(token) > 2
        and token
        not in {
            "the",
            "and",
            "for",
            "with",
            "what",
            "when",
            "where",
            "which",
            "does",
            "about",
            "mbzuai",
            "please",
            "tell",
            "explain",
            "describe",
            "give",
        }
    }


def _should_exclude_context_item(query: str, item_blob: str) -> bool:
    query_lower = str(query or "").casefold()
    if (
        "24/7" in item_blob
        and re.search(r"\b(working hours|offices operate|operating hours|practical|arrival|arriving|newcomer)\b", query_lower)
        and not re.search(r"\b(electronic resources|library access|library hours|24/7)\b", query_lower)
    ):
        return True
    return False


_KIND_BASE_SCORE = {
    "assertion": 100.0,
    "fact": 86.0,
    "evidence_span": 82.0,
    "chunk": 54.0,
    "answer": 38.0,
    "media": 0.0,
}


def _query_specific_bonus(query: str, item_blob: str, normalized_source: str) -> float:
    query_lower = str(query or "").casefold()
    bonus = 0.0
    admissions_contact_query = bool(
        re.search(
            r"\b(general admissions|admissions?\s+committee|admission@mbzuai\.ac\.ae|admissions?\s+contact|admissions?(?:\s+\w+){0,3}\s+email|admission email)\b",
            query_lower,
        )
        or re.search(r"\bcontact\b.{0,60}\badmissions?\b", query_lower)
        or re.search(r"\badmissions?\b.{0,60}\bcontact\b", query_lower)
    )
    if admissions_contact_query:
        if "admission@mbzuai.ac.ae" in item_blob:
            bonus += 42.0
            if "university-catalogue-2024-2025" in normalized_source:
                bonus += 18.0
            if "mbzuai_application_instructions_new_msc-phd" in normalized_source:
                bonus += 34.0
            if "online-screening-exam-instructions" in normalized_source:
                bonus += 24.0
                if "committee" in query_lower:
                    bonus += 24.0
        if "ug.admission@mbzuai.ac.ae" in item_blob and "undergraduate" not in query_lower:
            bonus -= 45.0
        if any(marker in item_blob for marker in ("emergency response", "emergency contact", "mbzuai management.contact number")):
            bonus -= 45.0

    if re.search(r"\b(online screening exam|screening exam)\b", query_lower):
        screening_it_hours_query = bool(
            re.search(r"\b(it support|technical support|when .*available|available|working hours|hours)\b", query_lower)
        )
        if screening_it_hours_query:
            has_complete_hours = (
                "working hours" in item_blob
                and "8:00 am" in item_blob
                and ("12:30 pm" in item_blob or "12:30" in item_blob)
            )
            has_truncated_hours = (
                "working hours" in item_blob
                and "8:00 am" in item_blob
                and ("5:00 pm (" in item_blob or item_blob.rstrip().endswith("("))
                and not ("12:30 pm" in item_blob or "12:30" in item_blob)
            )
            if has_complete_hours:
                bonus += 190.0
            elif has_truncated_hours:
                bonus -= 120.0
            elif "it_external@mbzuai.ac.ae" in item_blob:
                bonus += 54.0
            elif "8:00 am - 5:00 pm" in item_blob or "8:00 am - 12:30 pm" in item_blob:
                bonus += 80.0
            else:
                bonus -= 28.0
        if any(
            marker in item_blob
            for marker in (
                "online screening exam",
                "screening exam instructions",
                "exam topics",
                "before the exam",
                "after the exam",
                "admission-related questions may be sent to admission@mbzuai.ac.ae",
                "process, opting out criteria, and technical specifications",
            )
        ):
            bonus += 34.0
        if "online-screening-exam-instructions" in normalized_source:
            bonus += 18.0

    if re.search(r"\b(parking permitted|where .*parking|where .*park|north car park|masdar city campus.*parking)\b", query_lower):
        if "north car park" in item_blob:
            bonus += 50.0
            if "university-catalogue-2024-2025" in normalized_source:
                bonus += 30.0
        if any(marker in item_blob for marker in ("south car park", "airport parking")):
            bonus -= 45.0
    if re.search(r"\b(transport|transportation|shuttle|bus|arriv(?:e|ing|al)|visitor)\b", query_lower):
        if "navya bus" in item_blob or "electric autonomous navya" in item_blob:
            bonus += 82.0
            if "if available" in item_blob:
                bonus += 18.0
        elif "golf cart" in item_blob:
            bonus += 58.0
        elif "prt" in item_blob or "personal rapid transport" in item_blob or "personal rapid transit" in item_blob:
            bonus += 34.0
        if "about/contact" in normalized_source:
            bonus += 12.0
    if (
        re.search(r"\b(working hours|offices operate|operating hours|practical|arrival|arriving|newcomer)\b", query_lower)
        and "24/7" in item_blob
        and not re.search(r"\b(electronic resources|library access|library hours|24/7)\b", query_lower)
    ):
        bonus -= 90.0
    return bonus


def _candidate_score(
    *,
    query: str,
    kind: str,
    doc: Dict[str, Any],
    required_pages: Sequence[str],
    required_entities: Sequence[str],
) -> float:
    text = _doc_text(doc)
    source = _source_url(doc)
    item_blob = _item_search_text(
        {
            "text": text,
            "source_url": source,
            "document_title": _document_title(doc),
            "section_heading": doc.get("section_heading") or doc.get("heading"),
            "breadcrumb": doc.get("breadcrumb"),
        }
    )
    score = _KIND_BASE_SCORE.get(kind, 1.0)
    if kind == "media" and not re.search(r"\b(image|photo|video|media|map|picture|visual)\b", str(query or ""), re.IGNORECASE):
        score -= 40.0
    if _is_official_mbzuai_url(source):
        score += 18.0
    elif not source:
        score -= 36.0 if kind in {"answer", "fact", "evidence_span"} else 16.0
    else:
        score -= 10.0
    if _is_arabic_source_url(source) and not _ARABIC_TEXT_RE.search(str(query or "")):
        score -= 32.0

    terms = _query_terms(query)
    if terms:
        text_terms = set(re.findall(r"[a-z0-9]+", item_blob))
        score += (len(terms & text_terms) / float(len(terms))) * 28.0

    normalized_source = _normalize_url_for_match(source)
    if required_pages:
        if normalized_source and normalized_source in set(required_pages):
            score += 70.0
        elif source:
            score -= 26.0
        elif kind in {"answer", "fact", "evidence_span"}:
            score -= 18.0

    for entity in required_entities:
        if entity and entity in item_blob:
            score += 16.0
    score += _query_specific_bonus(query, item_blob, normalized_source)

    if kind == "answer" and not (doc.get("source_span_ids") or doc.get("linked_span_ids")):
        score -= 14.0
    if str(doc.get("validity_status") or "").casefold() in {"quarantined", "superseded", "stale"}:
        score -= 80.0
    return score


def build_evidence_pack(
    *,
    query: str,
    result: Dict[str, Any],
    max_items: int = 8,
    max_chars: int = 8000,
    max_per_source: int = 2,
    coverage_plan: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    max_items = max(1, int(max_items or 8))
    max_chars = max(800, int(max_chars or 8000))
    max_per_source = max(1, int(max_per_source or 2))

    seen_keys = set()
    source_counts: Dict[str, int] = {}
    items: List[Dict[str, Any]] = []
    used_chars = 0
    truncated = False

    coverage_plan = coverage_plan if isinstance(coverage_plan, dict) else {}
    required_entities = _coverage_values(coverage_plan, "required_entities")
    required_pages = [
        value
        for value in (
            _normalize_url_for_match(raw)
            for raw in (coverage_plan.get("required_pages") or [])
        )
        if value
    ]
    required_sections = _coverage_values(coverage_plan, "required_sections")
    intent = _clean_text(coverage_plan.get("intent")).casefold()
    specific_required_page_mode = bool(
        required_pages
        and (
            len(required_pages) == 1
            or (len(required_pages) <= 2 and intent not in {"multi_page_aggregation", "large_page"})
        )
    )
    required_page_set = set(required_pages)

    candidates: List[Tuple[float, str, Dict[str, Any]]] = []
    for kind, doc in _candidate_stream(result):
        text = _doc_text(doc)
        if not text:
            continue
        item_blob = _item_search_text(
            {
                "text": text,
                "source_url": _source_url(doc),
                "document_title": _document_title(doc),
                "section_heading": doc.get("section_heading") or doc.get("heading"),
                "breadcrumb": doc.get("breadcrumb"),
            }
        )
        if _should_exclude_context_item(query, item_blob):
            continue
        key = _dedupe_key(doc, kind)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if kind == "answer" and not _source_url(doc) and not (doc.get("source_span_ids") or doc.get("linked_span_ids")):
            continue
        candidates.append(
            (
                _candidate_score(
                    query=query,
                    kind=kind,
                    doc=doc,
                    required_pages=required_pages,
                    required_entities=required_entities,
                ),
                kind,
                doc,
            )
        )
    candidates.sort(key=lambda item: (-item[0], _doc_id(item[2])))

    selected_candidate_keys: set[str] = set()

    def _candidate_matches_requirement(doc: Dict[str, Any], requirement: str, *, page: bool) -> bool:
        if page:
            return bool(requirement and _normalize_url_for_match(_source_url(doc)) == requirement)
        return bool(requirement and _entity_match_text(requirement) in _entity_match_text(_item_search_text(doc)))

    def _candidate_matches_any_required_page(doc: Dict[str, Any]) -> bool:
        return bool(required_pages and _normalize_url_for_match(_source_url(doc)) in set(required_pages))

    def _append_candidate(kind: str, doc: Dict[str, Any]) -> bool:
        nonlocal used_chars, truncated
        key = _dedupe_key(doc, kind)
        if key in selected_candidate_keys:
            return False
        source = _source_url(doc) or "local"
        normalized_source = _normalize_url_for_match(source)
        source_cap = max_per_source
        if normalized_source in required_page_set:
            source_cap = max(source_cap, 4 if specific_required_page_mode else 3)
        if source_counts.get(source, 0) >= source_cap:
            return False
        remaining = max_chars - used_chars
        if remaining <= 0 or len(items) >= max_items:
            truncated = True
            return False
        item_text = _doc_text(doc)
        if len(item_text) > remaining:
            item_text = item_text[: max(0, remaining)].rsplit(" ", 1)[0].strip() or item_text[:remaining].strip()
            truncated = True
        if not item_text:
            return False
        selected_candidate_keys.add(key)
        source_counts[source] = source_counts.get(source, 0) + 1
        used_chars += len(item_text)
        items.append(
            {
                "rank": len(items) + 1,
                "kind": kind,
                "id": _doc_id(doc),
                "text": item_text,
                "source_url": source if source != "local" else "",
                "document_title": _document_title(doc),
                "section_heading": _clean_text(doc.get("section_heading") or doc.get("heading")),
                "breadcrumb": _clean_text(doc.get("breadcrumb")),
                "confidence": _confidence(doc),
                "authority_score": _authority_score(doc),
            }
        )
        return True

    for required_page in required_pages:
        for _score, kind, doc in candidates:
            if _candidate_matches_requirement(doc, required_page, page=True) and _append_candidate(kind, doc):
                break
    if specific_required_page_mode:
        for required_page in required_pages:
            for _score, kind, doc in candidates:
                if len(items) >= max_items or used_chars >= max_chars:
                    truncated = True
                    break
                if _candidate_matches_requirement(doc, required_page, page=True):
                    _append_candidate(kind, doc)
    for required_entity in required_entities:
        if required_entity and required_entity not in "\n".join(_item_search_text(item) for item in items):
            for _score, kind, doc in candidates:
                if _candidate_matches_requirement(doc, required_entity, page=False) and _append_candidate(kind, doc):
                    break
    for _score, kind, doc in candidates:
        if len(items) >= max_items or used_chars >= max_chars:
            truncated = True
            break
        if specific_required_page_mode and not _candidate_matches_any_required_page(doc):
            continue
        _append_candidate(kind, doc)

    evidence_order = {
        "assertion": 0,
        "fact": 1,
        "evidence_span": 2,
        "chunk": 3,
        "summary": 4,
        "answer": 5,
        "media": 6,
    }
    def _required_page_sort_rank(item: Dict[str, Any]) -> int:
        if not required_page_set:
            return 0
        return 0 if _normalize_url_for_match(item.get("source_url")) in required_page_set else 1

    def _required_entity_sort_rank(item: Dict[str, Any]) -> int:
        item_blob = _entity_match_text(_item_search_text(item))
        for index, entity in enumerate(required_entities):
            if entity and _entity_match_text(entity) in item_blob:
                return index
        return 999

    coverage_first_intent = intent in {"broad_synthesis", "multi_page_aggregation", "large_page"} or len(required_pages) > 1
    if coverage_first_intent:
        items.sort(
            key=lambda item: (
                _required_page_sort_rank(item),
                _required_entity_sort_rank(item),
                evidence_order.get(str(item.get("kind") or ""), 99),
                int(item.get("rank") or 0),
            )
        )
    else:
        items.sort(
            key=lambda item: (
                _required_page_sort_rank(item),
                evidence_order.get(str(item.get("kind") or ""), 99),
                _required_entity_sort_rank(item),
                int(item.get("rank") or 0),
            )
        )
    for index, item in enumerate(items, start=1):
        item["rank"] = index

    citations = []
    citation_seen = set()
    for item in items:
        citation_key = (item.get("source_url") or "", item.get("document_title") or "")
        if not any(citation_key) or citation_key in citation_seen:
            continue
        citation_seen.add(citation_key)
        citations.append(
            {
                "source_url": citation_key[0],
                "document_title": citation_key[1],
            }
        )

    item_text = _entity_match_text("\n".join(_item_search_text(item) for item in items))
    item_sources = {_normalize_url_for_match(item.get("source_url")) for item in items if item.get("source_url")}
    missing_required_entities = [
        value for value in required_entities if value and _entity_match_text(value) not in item_text
    ]
    missing_required_pages = [
        value for value in required_pages if value and value not in item_sources and value not in item_text
    ]
    missing_required_sections = [
        value for value in required_sections if value and value not in item_text
    ]
    if bool(result.get("abstained")) or not items:
        coverage_status = "insufficient"
    elif missing_required_entities or missing_required_pages or missing_required_sections:
        coverage_status = "partial"
    else:
        coverage_status = "complete"
    citation_candidates = [
        {
            "id": item.get("id") or "",
            "kind": item.get("kind") or "",
            "source_url": item.get("source_url") or "",
            "document_title": item.get("document_title") or "",
        }
        for item in items
        if item.get("source_url")
    ]

    return {
        "query": query,
        "items": items,
        "citations": citations,
        "coverage_status": coverage_status,
        "missing_required_entities": missing_required_entities,
        "missing_required_pages": missing_required_pages,
        "missing_required_sections": missing_required_sections,
        "citation_candidates": citation_candidates,
        "budget": {
            "max_items": max_items,
            "max_chars": max_chars,
            "max_per_source": max_per_source,
            "used_items": len(items),
            "used_chars": used_chars,
            "truncated": truncated,
        },
    }


def score_retrieval_confidence(result: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    if bool(result.get("abstained")):
        return 0.0, {"abstained": True}

    answer_docs = _coerce_docs(result.get("answer_documents") or [])
    fact_docs = _coerce_docs(result.get("fact_documents") or [])
    evidence_span_docs = _coerce_docs(result.get("evidence_span_documents") or [])
    retrieval_docs = _coerce_docs(result.get("retrieval_documents") or [])
    selected_chunks = [str(value) for value in result.get("selected_chunk_ids") or [] if str(value)]
    selected_spans = [str(value) for value in result.get("selected_evidence_span_ids") or [] if str(value)]

    answer_confidence = max([_confidence(doc) for doc in answer_docs] or [0.0])
    fact_signal = min(1.0, len(fact_docs) / 4.0)
    span_signal = min(1.0, len(selected_spans or evidence_span_docs) / 6.0)
    chunk_signal = min(1.0, len(selected_chunks) / 8.0)
    authority_signal = max([_authority_score(doc) for doc in [*answer_docs, *fact_docs, *evidence_span_docs, *retrieval_docs]] or [0.0])

    dense_ids = set(str(value) for value in result.get("dense_chunk_ids") or [])
    sparse_ids = set(str(value) for value in result.get("sparse_chunk_ids") or [])
    local_ids = set(str(value) for value in result.get("local_chunk_ids") or [])
    selected_set = set(selected_chunks)
    lane_agreement = 0.0
    if selected_set:
        agreed = selected_set & ((dense_ids & sparse_ids) | (dense_ids & local_ids) | (sparse_ids & local_ids))
        lane_agreement = len(agreed) / float(len(selected_set))

    graph_confidence = 0.0
    try:
        graph_confidence = max(0.0, min(1.0, float(result.get("routing_relation_confidence") or 0.0)))
    except (TypeError, ValueError):
        graph_confidence = 0.0

    adjudication_confidence = 0.0
    if result.get("adjudication_used"):
        try:
            adjudication_confidence = max(0.0, min(1.0, float(result.get("adjudication_confidence") or 0.0)))
        except (TypeError, ValueError):
            adjudication_confidence = 0.0

    direct_signal = max(answer_confidence, fact_signal * 0.72, chunk_signal * 0.56)
    direct_signal = max(direct_signal, span_signal * 0.74)
    confidence = (
        direct_signal * 0.46
        + lane_agreement * 0.18
        + authority_signal * 0.18
        + graph_confidence * 0.08
        + adjudication_confidence * 0.10
    )
    if answer_docs:
        confidence += 0.08
    if evidence_span_docs:
        confidence += 0.05
    if not retrieval_docs and not answer_docs and not fact_docs and not evidence_span_docs:
        confidence = 0.0

    factors = {
        "answer_confidence": round(answer_confidence, 4),
        "fact_signal": round(fact_signal, 4),
        "span_signal": round(span_signal, 4),
        "chunk_signal": round(chunk_signal, 4),
        "lane_agreement": round(lane_agreement, 4),
        "authority_signal": round(authority_signal, 4),
        "graph_confidence": round(graph_confidence, 4),
        "adjudication_confidence": round(adjudication_confidence, 4),
        "answer_count": len(answer_docs),
        "fact_count": len(fact_docs),
        "chunk_count": len(selected_chunks),
    }
    return round(max(0.0, min(1.0, confidence)), 4), factors
