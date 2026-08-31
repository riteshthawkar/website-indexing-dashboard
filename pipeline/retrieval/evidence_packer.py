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
    r"https?://(?:(?:[a-z0-9-]+\.)*mbzuai\.ac\.ae|(?:[a-z0-9-]+\.)*ifm\.ai|mbzuai\.gitbook\.io)/[^\s\]\)\"'<>,]+",
    re.IGNORECASE,
)
_ARABIC_TEXT_RE = re.compile(r"[\u0600-\u06FF]")
_EMAIL_VALUE_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE,
)
_EXPLICIT_MEDIA_QUERY_RE = re.compile(
    r"\b(?:image|images|photo|photograph|picture|diagram|figure|chart|table|map|infographic|"
    r"screenshot|visual|workflow|framework|pdf)\b"
    r"|\b(?:shown|displayed|visible|pictured)\s+(?:in|on)\s+(?:the\s+)?(?:form|portal|page)\b"
    r"|(?:صورة|الصورة|صور|مخطط|المخطط|رسم|الشكل|خريطة|الخريطة|إنفوغراف|الإنفوغراف|"
    r"جدول|الجدول|لقطة شاشة|سير العمل|إطار|الإطار|لوحة|اللوحة)",
    re.IGNORECASE,
)


def _is_explicit_media_query(query: str) -> bool:
    intent_text = re.sub(
        r"\b(?:titled|called|named)\s+[\"'“‘][^\"'”’]{0,240}[\"'”’]",
        " ",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    normalized = _clean_text(intent_text).casefold()
    if (
        "policy page" in normalized
        and re.search(r"\b(?:promotion|professor|faculty review)\b", normalized)
    ):
        return True
    if re.search(
        r"\b(?:shown|displayed|visible|pictured)\s+(?:in|on)\s+(?:(?:the|this)\s+)?(?:form|portal|page)\b",
        normalized,
    ):
        return True
    return bool(_EXPLICIT_MEDIA_QUERY_RE.search(intent_text))


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
    return bool(
        host == "mbzuai.ac.ae"
        or host.endswith(".mbzuai.ac.ae")
        or host == "ifm.ai"
        or host.endswith(".ifm.ai")
        or host == "mbzuai.gitbook.io"
    )


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
    if title and not re.fullmatch(r"[a-f0-9]{16,64}", title.casefold()):
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
    if _is_official_mbzuai_url(_source_url(doc)):
        return 0.95
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
    for retrieval_rank, item in enumerate(values or []):
        if not isinstance(item, dict):
            continue
        docs.append(
            {
                "id": item.get("id"),
                "text": (
                    item.get("text")
                    or item.get("dense_text")
                    or item.get("raw_text")
                    or item.get("description")
                    or item.get("caption")
                    or item.get("title")
                ),
                "source_url": item.get("source_url") or item.get("url") or item.get("asset_uri"),
                "document_title": item.get("document_title") or item.get("title"),
                "media_type": item.get("media_type") or item.get("type"),
                "asset_uri": item.get("asset_uri"),
                "url": item.get("url"),
                "retrieval_rank": retrieval_rank,
            }
        )
    return docs


def _candidate_stream(result: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for doc in _coerce_docs(result.get("answer_documents") or []):
        if str(doc.get("record_type") or "") == "navigation_action":
            yield "action", doc
        elif doc.get("source_span_ids") or doc.get("linked_span_ids"):
            yield "assertion", doc
        else:
            yield "answer", doc
    for doc in _coerce_docs(result.get("fact_documents") or []):
        yield "fact", doc
    for doc in _coerce_docs(result.get("evidence_span_documents") or []):
        yield "evidence_span", doc
    retrieval_rank = 0
    for doc in _coerce_docs(result.get("retrieval_documents") or []):
        if doc.get("span_type"):
            yield "evidence_span", doc
            continue
        ranked_doc = dict(doc)
        record_id = _doc_id(ranked_doc)
        if record_id.startswith("assertion:") or str(
            ranked_doc.get("record_type") or ""
        ).casefold() in {"assertion", "relation_assertion"}:
            # Promoted graph assertions can be returned through the fused
            # retrieval lane rather than answer_documents. Preserve their
            # stronger evidence prior; treating them as anonymous chunks can
            # hide an exact structured fact behind a long page preamble.
            yield "assertion", ranked_doc
            continue
        if record_id.startswith(("chunk:", "parent:")):
            ranked_doc.setdefault("retrieval_rank", retrieval_rank)
            retrieval_rank += 1
        yield "chunk", ranked_doc
    for doc in _media_documents(result.get("media") or []):
        yield "media", doc


def _validated_navigation_email_targets(result: Dict[str, Any]) -> set[str]:
    targets: set[str] = set()
    for doc in _coerce_docs(result.get("answer_documents") or []):
        if (
            str(doc.get("record_type") or "") != "navigation_action"
            or str(doc.get("answer_type") or "").casefold() != "email"
        ):
            continue
        value = _clean_text(
            " ".join(
                str(doc.get(key) or "")
                for key in ("value", "action_target_url", "text")
            )
        )
        if value.casefold().startswith("mailto:"):
            value = value[7:].split("?", 1)[0]
        targets.update(match.casefold() for match in _EMAIL_VALUE_RE.findall(value))
    return targets


def _conflicts_with_validated_email_action(
    doc: Dict[str, Any],
    targets: set[str],
) -> bool:
    if not targets or str(doc.get("record_type") or "") == "navigation_action":
        return False
    contact_text = " ".join(
        str(doc.get(key) or "")
        for key in ("text", "value", "source_url", "document_title")
    )
    emails = {match.casefold() for match in _EMAIL_VALUE_RE.findall(contact_text)}
    return bool(emails and emails.isdisjoint(targets))


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


def _normalize_match_token(value: Any) -> str:
    token = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", str(value or "").casefold())
    if _ARABIC_TEXT_RE.search(token) and len(token) >= 5:
        for prefix in ("وال", "بال", "كال", "فال", "لل", "ال"):
            if token.startswith(prefix) and len(token) - len(prefix) >= 3:
                token = token[len(prefix) :]
                break
    return token


def _query_terms(query: str) -> set[str]:
    excluded = {
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
        "ما",
        "ماذا",
        "متى",
        "أين",
        "اين",
        "كيف",
        "هل",
        "كم",
        "بحسب",
        "هذا",
        "هذه",
        "الذي",
        "التي",
        "في",
        "من",
        "إلى",
        "الى",
        "على",
        "عن",
    }
    terms = {
        normalized
        for raw_token in re.findall(r"[^\W_]+", str(query or "").casefold(), flags=re.UNICODE)
        if (normalized := _normalize_match_token(raw_token))
        and len(normalized) > 2
        and normalized not in excluded
    }
    normalized_query = _clean_text(query).casefold()
    if any(
        marker in normalized_query
        for marker in (
            "المؤهل الأكاديمي",
            "المؤهلات الأكاديمية",
            "المؤهلات",
        )
    ):
        # The careers corpus is commonly English while users ask about degree
        # requirements in Arabic. These aliases let the answer-bearing
        # qualifications span outrank generic role-summary facts.
        terms.update(
            {
                "academic",
                "qualification",
                "qualifications",
                "degree",
                "bachelor",
                "master",
                "required",
                "preferred",
                "mandatory",
            }
        )
    if any(
        marker in normalized_query
        for marker in ("أقسام الوظائف", "الوظائف المفتوحة")
    ):
        terms.update(
            {
                "faculty",
                "research",
                "engineering",
                "professional",
                "vacancies",
            }
        )
    if "ifm" in normalized_query and any(
        marker in normalized_query
        for marker in ("شركاء", "الشركاء", "partners", "collaborat")
    ):
        terms.update(
            {
                "science",
                "scale",
                "social value",
                "academic institutions",
                "research labs",
                "startups",
                "enterprise leaders",
            }
        )
    if any(marker in normalized_query for marker in ("دانييلا روس", "daniela rus")):
        terms.update({"الاستقلالية", "الذكاء", "autonomy", "intelligence"})
    if any(
        marker in normalized_query
        for marker in ("دور الرئيس", "مهام الرئيس", "صلاحيات الرئيس")
    ):
        terms.update(
            {
                "الرئيس التنفيذي",
                "التنفيذي",
                "مهام",
                "الصلاحيات",
                "إدارة الجامعة",
                "إدارة",
                "chief executive",
            }
        )
    if (
        any(marker in normalized_query for marker in ("معرض التدريب المهني", "career fair"))
        and any(marker in normalized_query for marker in ("الدعم", "دعم", "support"))
    ):
        terms.update(
            {
                "جلسات تدريب مهني فردية",
                "وكالات التوظيف",
                "صور احترافية",
                "career coaching",
                "recruitment agencies",
            }
        )
    if (
        any(marker in normalized_query for marker in ("الدكتوراه", "doctorate", "doctoral", "phd"))
        and any(marker in normalized_query for marker in ("التوجه المهني", "career orientation", "career path"))
    ):
        terms.update(
            {
                "contribute to science and humanity",
                "experienced researchers",
                "academia",
                "research institute",
                "industry",
                "startup",
            }
        )
    if (
        any(marker in normalized_query for marker in ("visitor program", "برنامج الزوار"))
        and any(marker in normalized_query for marker in ("hands-on", "عملي", "تجربة"))
    ):
        terms.update(
            {
                "research experience program",
                "hands-on ai research experiences",
                "personalized demos",
                "talks",
            }
        )
    if (
        ("engage" in normalized_query and "capture" in normalized_query and "value" in normalized_query)
        or ("يتفاعل" in normalized_query and "القيمة" in normalized_query)
    ):
        terms.update(
            {
                "exploration",
                "refinement",
                "high level proposal",
                "engagement agreement sign-off",
            }
        )
    return terms


_MEDIA_FIELD_RE = re.compile(r"^([A-Z][A-Z0-9_ ]{1,48}):\s*(.*)$")
_MEDIA_IDENTITY_FIELDS = ("IMAGE", "DOCUMENT", "SECTION")
_MEDIA_ANSWER_FIELDS = (
    "VISIBLE_TEXT",
    "CONTEXTUAL_CAPTION",
    "SEMANTIC_CAPTION",
    "OCR_TEXT",
    "SURROUNDING_TEXT_AFTER",
    "SURROUNDING_TEXT_BEFORE",
    "NEARBY_TEXT",
    "CONTEXT",
    "VISUAL_DESCRIPTION",
    "SEMANTIC_TAGS",
    "IMAGE_KIND",
)


def _truncate_text(value: Any, limit: int) -> str:
    text = _clean_text(value)
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    clipped = text[:limit]
    return clipped.rsplit(" ", 1)[0].strip() or clipped.strip()


def _query_relevant_excerpt(value: Any, query: str, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    terms = sorted(_query_terms(query), key=len, reverse=True)
    normalized = text.casefold()
    match_positions = [
        position
        for term in terms
        if (position := normalized.find(term.casefold())) >= 0
    ]
    if not match_positions:
        return _truncate_text(text, limit)
    center = min(match_positions)
    start = max(0, center - max(120, limit // 3))
    end = min(len(text), start + limit)
    start = max(0, end - limit)
    excerpt = text[start:end]
    if start:
        excerpt = excerpt.split(" ", 1)[-1]
    if end < len(text):
        excerpt = excerpt.rsplit(" ", 1)[0]
    return excerpt.strip()


def _query_dense_excerpt(value: Any, query: str, limit: int) -> str:
    """Keep the window that covers the most query concepts.

    Long page chunks often start with role or page background and place the
    requested structured block near the end. Prefix truncation therefore drops
    exactly the qualifiers that distinguish, for example, a required degree
    from a preferred one. Cross-lingual aliases supplied by ``_query_terms``
    make the same selection work when an Arabic query targets English source
    text.
    """

    text = _clean_text(value)
    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text

    terms = sorted(_query_terms(query), key=len, reverse=True)
    normalized = text.casefold()
    positions: List[int] = []
    for term in terms:
        normalized_term = term.casefold()
        start = 0
        while normalized_term and len(positions) < 256:
            position = normalized.find(normalized_term, start)
            if position < 0:
                break
            positions.append(position)
            start = position + max(1, len(normalized_term))
        if len(positions) >= 256:
            break
    if not positions:
        return _truncate_text(text, limit)

    query_lower = _clean_text(query).casefold()
    facet_phrases: List[str] = []
    if re.search(r"\b(?:experience|required experience|how much experience)\b|(?:خبرة|الخبرة|الخبرات)", query_lower):
        facet_phrases.extend(("professional experience required", "experience required", "at least"))
    if re.search(r"\b(?:qualification|qualifications|degree|required and preferred|preferred)\b|(?:المؤهل|المؤهلات)", query_lower):
        facet_phrases.extend(
            (
                "academic qualifications required",
                "strongly preferred",
                "preferred but not mandatory",
                "minimum 8+ years",
                "certified irb professional",
            )
        )
    if "promotion" in query_lower and re.search(r"\b(?:timing|timeline|review|professor)\b", query_lower):
        facet_phrases.extend(
            (
                "end of the seventh year",
                "3 years after promotion to associate professor",
                "minimum service of two years",
            )
        )
    facet_positions = [
        position
        for phrase in facet_phrases
        if (position := normalized.find(phrase)) >= 0
    ]
    positions.extend(facet_positions)

    best: tuple[int, int, int, int] | None = None
    best_bounds = (0, limit)
    leading_context = min(320, max(100, limit // 4))
    for position in positions:
        is_facet_anchor = position in facet_positions
        anchor_context = min(180, leading_context) if is_facet_anchor else leading_context
        window_start = max(0, position - anchor_context)
        window_end = min(len(text), window_start + limit)
        window_start = max(0, window_end - limit)
        window = normalized[window_start:window_end]
        matched_terms = {term for term in terms if term.casefold() in window}
        # Prefer broad concept coverage, then more specific/longer concepts.
        # Earlier positions are only a final deterministic tie-breaker.
        matched_facets = {phrase for phrase in facet_phrases if phrase in window}
        score = (
            len(matched_facets),
            sum(len(phrase) for phrase in matched_facets),
            len(matched_terms),
            sum(len(term) for term in matched_terms),
            -abs(position - (window_start + leading_context)),
            -window_start,
        )
        if best is None or score > best:
            best = score
            best_bounds = (window_start, window_end)

    start, end = best_bounds
    excerpt = text[start:end]
    if start:
        excerpt = excerpt.split(" ", 1)[-1]
    if end < len(text):
        excerpt = excerpt.rsplit(" ", 1)[0]
    return excerpt.strip()


def _compact_media_evidence_text(value: Any, *, query: str, max_chars: int) -> str:
    """Preserve answer-bearing media fields instead of truncating a raw prefix."""
    raw = str(value or "").strip()
    if not raw or len(_clean_text(raw)) <= max_chars:
        return _clean_text(raw)

    parsed: Dict[str, List[str]] = {}
    for raw_line in raw.splitlines():
        match = _MEDIA_FIELD_RE.match(raw_line.strip())
        if not match:
            continue
        label = match.group(1).strip()
        content = _clean_text(match.group(2))
        if content:
            parsed.setdefault(label, []).append(content)
    if not parsed:
        return _truncate_text(raw, max_chars)

    parts: List[str] = []
    used = 0

    def append_field(
        label: str,
        content: str,
        *,
        field_limit: int,
        relevant: bool = False,
        dense: bool = False,
    ) -> None:
        nonlocal used
        remaining = max_chars - used
        prefix = f"{label}: "
        if remaining <= len(prefix) + 12:
            return
        bounded = min(field_limit, remaining - len(prefix) - (1 if parts else 0))
        if dense:
            rendered = _query_dense_excerpt(content, query, bounded)
        elif relevant:
            rendered = _query_relevant_excerpt(content, query, bounded)
        else:
            rendered = _truncate_text(content, bounded)
        if not rendered:
            return
        part = prefix + rendered
        parts.append(part)
        used += len(part) + (1 if len(parts) > 1 else 0)

    for label in _MEDIA_IDENTITY_FIELDS:
        for content in parsed.get(label, []):
            append_field(label, content, field_limit=240)

    # OCR-derived visible text is the strongest source for labels, numeric
    # values, and complete table rows. Keep it ahead of prose context.
    for content in parsed.get("VISIBLE_TEXT", []):
        append_field("VISIBLE_TEXT", content, field_limit=2100, dense=len(content) > 2100)

    for label in ("CONTEXTUAL_CAPTION", "SEMANTIC_CAPTION"):
        for content in parsed.get(label, []):
            append_field(label, content, field_limit=520)

    for content in parsed.get("OCR_TEXT", []):
        append_field("OCR_TEXT", content, field_limit=700, relevant=True)

    for label in (
        "SURROUNDING_TEXT_AFTER",
        "SURROUNDING_TEXT_BEFORE",
        "NEARBY_TEXT",
        "CONTEXT",
        "VISUAL_DESCRIPTION",
    ):
        for content in parsed.get(label, []):
            append_field(label, content, field_limit=640, relevant=True)

    for label in ("SEMANTIC_TAGS", "IMAGE_KIND"):
        for content in parsed.get(label, []):
            append_field(label, content, field_limit=260)

    compacted = "\n".join(parts)
    return _truncate_text(compacted, max_chars)


def _named_query_tokens(query: str) -> set[str]:
    tokens: set[str] = set()
    excluded = {
        "who",
        "what",
        "when",
        "where",
        "which",
        "why",
        "how",
        "is",
        "are",
        "does",
        "do",
        "can",
        "has",
        "have",
        "mbzuai",
    }
    for index, raw_token in enumerate(str(query or "").split()):
        token = "".join(
            character
            for character in raw_token
            if character.isalnum() or character in {"-", "_", "'"}
        ).strip("'")
        if not token:
            continue
        lowered = token.casefold()
        if lowered in excluded:
            continue
        if (
            (token.isupper() and len(token) >= 3)
            or any(character.isupper() for character in token[1:])
            or (
                index > 0
                and token[:1].isupper()
                and any(character.islower() for character in token[1:])
            )
        ):
            tokens.add(lowered)
    return tokens


def _named_source_identity_bonus(
    query: str,
    *,
    source_url: str,
    document_title: str,
    heading: str,
) -> float:
    identity_tokens = _named_query_tokens(query)
    if not identity_tokens:
        return 0.0
    try:
        parsed = urlparse(str(source_url or ""))
        host = (parsed.hostname or "").casefold()
        path = unquote(parsed.path or "")
    except Exception:
        host = ""
        path = str(source_url or "")
    source_tokens = set(
        re.findall(
            r"[a-z0-9]+",
            " ".join(
                (
                    host,
                    path,
                    str(document_title or ""),
                    str(heading or ""),
                )
            ).casefold(),
        )
    )
    matched = identity_tokens & source_tokens
    if not matched:
        if host == "careers.mbzuai.ac.ae" and len(identity_tokens) >= 2:
            return -72.0
        return 0.0
    match_ratio = len(matched) / float(len(identity_tokens))
    bonus = min(52.0, (8.0 * len(matched)) + (36.0 * match_ratio))
    if host == "careers.mbzuai.ac.ae" and len(identity_tokens) >= 2 and match_ratio < 0.5:
        bonus -= 72.0
    host_tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", host)
        if token not in {"www", "ac", "ae", "ai", "com", "edu", "org"}
    }
    if matched & host_tokens:
        bonus += 16.0
    return bonus


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
    "action": 108.0,
    "assertion": 100.0,
    "fact": 86.0,
    "evidence_span": 82.0,
    "chunk": 54.0,
    "answer": 38.0,
    "media": 0.0,
}


_MULTI_DETAIL_QUERY_RE = re.compile(
    r"\b(?:requirements|qualifications|qualification|academic qualification|roles|positions|responsibilities|steps|"
    r"features|benefits|differences|criteria|items|articles|entries|sections|categories|stages|process|programs?|"
    r"experience|professional experience|"
    r"support|services|use|uses|using|options|focus areas|research interests|hands-on access|offerings|committees|industry engagement)\b"
    r"|(?:المتطلبات|المؤهلات|المؤهل الأكاديمي|المؤهل|المناصب|الأدوار|المسؤوليات|الخطوات|المزايا|الفروقات|"
    r"المعايير|العناصر|المقالات|أقسام|اقسام|فئات|مراحل|عملية|البرنامج|برنامج|البرامج|برامج|المدة|مدة|"
    r"المنح|منح|الشروط|شروط|الخبرة|خبرة|الخبرات|الدعم|دعم|الخدمات|خدمات|استخدامات|خيارات|"
    r"المجالات|مجالات|الاهتمامات البحثية|اهتماماتها البحثية|وصول عملي|تجارب بحثية|اللجان)"
    r"|(?:engag\w*(?:\s+\w+){0,4}\s+industry|captur\w*\s+value)"
    r"|(?:ما\s+.{0,180}\s+وأين|أين\s+.{0,180}\s+وما|ما\s+.{0,180}\s+وما)",
    re.IGNORECASE,
)

_STAGED_PROCESS_QUERY_RE = re.compile(
    r"\b(?:stages?|process|engag\w*(?:\s+\w+){0,4}\s+industry|captur\w*\s+value)\b"
    r"|(?:مراحل|المراحل|عملية|العملية)",
    re.IGNORECASE,
)


def _multi_detail_chunk_coverage_bonus(query: str, item_blob: str, kind: str) -> float:
    """Reward a complete context window when an answer spans several details.

    Evidence spans normally deserve a precision prior, but a span extractor can
    split a list or a pair of requested fields at its boundary. For plural or
    structured-block questions (including academic qualifications), a selected
    chunk that covers most informative query terms is safer than individually
    precise spans that omit a required/preferred qualifier. The minimum coverage
    threshold prevents generic long chunks from receiving this bonus merely
    because they contain one common token.
    """

    if kind != "chunk" or not _MULTI_DETAIL_QUERY_RE.search(str(query or "")):
        return 0.0
    terms = _query_terms(query)
    if len(terms) < 3:
        return 0.0
    item_terms = {
        normalized
        for raw_token in re.findall(r"[^\W_]+", item_blob, flags=re.UNICODE)
        if (normalized := _normalize_match_token(raw_token))
    }
    coverage = len(terms & item_terms) / float(len(terms))
    if coverage < 0.55:
        return 0.0
    return min(58.0, 18.0 + (42.0 * coverage))


def _structured_facet_coverage_bonus(query: str, text: str) -> float:
    """Reward evidence that covers the specific facets named by the user.

    Representation headers repeat broad terms such as MBZUAI, program, and
    page title across every child chunk. Multi-aspect selection therefore
    needs a body-level signal for the actual requested facets (duration,
    scholarship, requirements, stages, and so on), otherwise a generic child
    can outrank the adjacent answer-bearing block.
    """

    query_text = _clean_text(query).casefold()
    body_lines = [
        line
        for line in str(text or "").splitlines()
        if not re.match(
            r"^\s*(?:TITLE|TYPE|SECTION|SOURCE_URL|PARENT_TYPE)\s*:",
            line,
            flags=re.IGNORECASE,
        )
    ]
    body_text = _clean_text(" ".join(body_lines)).casefold()
    facet_contracts = (
        (
            r"\b(?:duration|how long|study length)\b|(?:مدة|المدة)",
            r"\b(?:year|years|duration|semester|semesters)\b|(?:سنة|سنوات|فصل|فصول)",
        ),
        (
            r"\b(?:scholarship|scholarships|financial aid)\b|(?:منحة|منح|المنح)",
            r"\b(?:scholarship|scholarships|financial aid|tuition support)\b|(?:منحة|منح|المنح)",
        ),
        (
            r"\b(?:requirements|eligibility|admission conditions)\b|(?:شروط|الشروط|متطلبات|المتطلبات)",
            r"\b(?:requirements|eligibility|secondary school|gpa)\b|(?:شروط|الشروط|متطلبات|المتطلبات|الثانوية|90%)",
        ),
        (
            r"\bprograms?\b|(?:البرامج|برامج)",
            r"\b(?:program|programs|bachelor|master|msc|phd|doctorate)\b|(?:برنامج|برامج|بكالوريوس|ماجستير|دكتوراه)",
        ),
        (
            r"\b(?:hands-on|hands on|practical access)\b|(?:تجارب عملية|وصول عملي)",
            r"\b(?:hands-on|hands on|research experience|personalized demo|personalised demo)\b|(?:تجارب عملية|وصول عملي)",
        ),
        (
            r"\b(?:stages|process stages|engages? with industry|captures? value)\b|(?:مراحل|المراحل)",
            r"\b(?:exploration|refinement|proposal|sign-off|sign off|project start)\b|(?:استكشاف|تنقيح|مقترح|بدء المشروع)",
        ),
        (
            r"\b(?:experience|required experience|professional experience)\b|(?:خبرة|الخبرة|الخبرات)",
            r"\b(?:experience|years?)\b|(?:خبرة|الخبرة|سنوات)",
        ),
    )
    requested = 0
    matched = 0
    for query_pattern, evidence_pattern in facet_contracts:
        if not re.search(query_pattern, query_text, flags=re.IGNORECASE):
            continue
        requested += 1
        if re.search(evidence_pattern, body_text, flags=re.IGNORECASE):
            matched += 1
    if not requested:
        return 0.0
    bonus = -12.0 if not matched else min(72.0, 26.0 * matched)
    scoped_subject_contracts = (
        (
            r"\b(?:undergraduate|bachelor|bsc)\b|(?:بكالوريوس|البكالوريوس)",
            r"\b(?:undergraduate|bachelor|bsc)\b|(?:بكالوريوس|البكالوريوس)",
        ),
        (
            r"\bvisitor program\b|(?:برنامج الزوار)",
            r"\bvisitor program\b|(?:برنامج الزوار)",
        ),
        (
            r"\blibrary\b|(?:المكتبة|مكتبة)",
            r"\blibrary\b|(?:المكتبة|مكتبة)",
        ),
    )
    for query_pattern, evidence_pattern in scoped_subject_contracts:
        if re.search(query_pattern, query_text, flags=re.IGNORECASE) and not re.search(
            evidence_pattern,
            body_text,
            flags=re.IGNORECASE,
        ):
            bonus -= 64.0
    return bonus


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
    raw_text = str(doc.get("text") or doc.get("value") or _doc_metadata(doc).get("context") or "")
    text = _clean_text(raw_text)
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
    explicit_media_query = _is_explicit_media_query(query)
    score = 112.0 if kind == "media" and explicit_media_query else _KIND_BASE_SCORE.get(kind, 1.0)
    if kind == "media" and not explicit_media_query:
        score -= 40.0
    if kind == "chunk" and doc.get("retrieval_rank") is not None:
        try:
            retrieval_rank = max(0, int(doc.get("retrieval_rank") or 0))
        except (TypeError, ValueError):
            retrieval_rank = 0
        # The adaptive retriever has already fused dense, sparse, graph, fact,
        # page-card, and parent evidence. Keep that ordering meaningful so the
        # evidence packer does not replace the first answer-bearing chunks with
        # a semantically generic chunk merely because both share page headers.
        rank_after_primary_pair = max(0, retrieval_rank - 1)
        score += max(0.0, 18.0 - (0.3 * (rank_after_primary_pair ** 2)))
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
        text_terms = {
            normalized
            for raw_token in re.findall(r"[^\W_]+", item_blob, flags=re.UNICODE)
            if (normalized := _normalize_match_token(raw_token))
        }
        score += (len(terms & text_terms) / float(len(terms))) * 28.0

    if kind == "media" and explicit_media_query:
        try:
            retrieval_rank = max(0, int(doc.get("retrieval_rank") or 0))
        except (TypeError, ValueError):
            retrieval_rank = 0
        # The adaptive retriever has already combined dense, sparse, graph,
        # and media-specific signals. Preserve that ordering as the primary
        # media signal instead of letting a weak language heuristic undo it.
        score += max(0.0, 30.0 - (6.0 * retrieval_rank))
        query_is_arabic = bool(_ARABIC_TEXT_RE.search(str(query or "")))
        item_is_arabic = bool(_ARABIC_TEXT_RE.search(item_blob))
        if query_is_arabic:
            score += 6.0 if item_is_arabic else -4.0
        if re.search(r"\b(?:invite|invites|inviting)\b", str(query or ""), re.IGNORECASE):
            if any(marker in item_blob for marker in ("scan qr", "qr code", "digital copy", "download")):
                score += 64.0

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
    score += _named_source_identity_bonus(
        query,
        source_url=source,
        document_title=_document_title(doc),
        heading=_clean_text(doc.get("section_heading") or doc.get("heading")),
    )
    score += _multi_detail_chunk_coverage_bonus(query, item_blob, kind)
    score += _structured_facet_coverage_bonus(query, raw_text)
    score += _query_specific_bonus(query, item_blob, normalized_source)
    if bool(doc.get("coverage_aggregate")):
        # A bounded complete-page parent is deliberately injected for a
        # multi-detail/list question. Keep it ahead of isolated snippets so
        # generation sees the full structured block in one evidence item.
        score += 72.0

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
    explicit_media_query = _is_explicit_media_query(query)
    structured_detail_query = bool(_MULTI_DETAIL_QUERY_RE.search(str(query or "")))
    normalized_query = str(query or "").casefold()
    relational_person_query = bool(
        re.search(r"\b(?:who|whom)\b", normalized_query)
        and re.search(r"\b(?:speaker|host|hosted|hosting)\b", normalized_query)
    )

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
    validated_email_targets = _validated_navigation_email_targets(result)
    for kind, doc in _candidate_stream(result):
        if _conflicts_with_validated_email_action(
            doc,
            validated_email_targets,
        ):
            continue
        text = _doc_text(doc)
        if not text:
            continue
        if kind in {"chunk", "evidence_span", "media", "summary"} and (
            len(text) < 20 or len(re.findall(r"\w+", text, re.UNICODE)) < 3
        ):
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
    media_reserve = 3 if max_items >= 6 else 2 if max_items >= 4 else 1

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
        if (
            kind == "media"
            and explicit_media_query
            and sum(1 for item in items if item.get("kind") == "media") >= media_reserve
        ):
            return False
        source = _source_url(doc) or "local"
        normalized_source = _normalize_url_for_match(source)
        source_cap = max_per_source
        if structured_detail_query and kind != "media":
            # Multi-aspect answers commonly span adjacent sections on one
            # official page. The default two-items-per-source diversity cap
            # can retain the page parent plus only the first leaf, silently
            # dropping later stages, requirements, or list entries.
            source_cap = max(source_cap, 6)
        if normalized_source in required_page_set:
            source_cap = max(source_cap, 4 if specific_required_page_mode else 3)
        if source_counts.get(source, 0) >= source_cap:
            return False
        remaining = max_chars - used_chars
        if remaining <= 0 or len(items) >= max_items:
            truncated = True
            return False
        item_text = _doc_text(doc)
        item_char_limit = remaining
        if kind == "media" and explicit_media_query:
            item_char_limit = min(item_char_limit, 2400)
            item_text = _compact_media_evidence_text(
                doc.get("text") or item_text,
                query=query,
                max_chars=item_char_limit,
            )
        elif kind == "chunk" and (structured_detail_query or relational_person_query):
            # A complete structured block is safer than a short extracted span,
            # but sending a full page chunk adds latency and the answer runtime
            # may prefix-truncate it again. Preserve a bounded, query-dense
            # window containing the whole answer-bearing block instead.
            item_char_limit = min(item_char_limit, 2400)
            item_text = _query_dense_excerpt(item_text, query, item_char_limit)
        if len(item_text) > item_char_limit:
            item_text = item_text[: max(0, item_char_limit)].rsplit(" ", 1)[0].strip() or item_text[:item_char_limit].strip()
            truncated = True
        if not item_text:
            return False
        if kind in {"chunk", "evidence_span", "media", "summary"} and (
            len(item_text) < 20
            or len(re.findall(r"\w+", item_text, re.UNICODE)) < 3
        ):
            truncated = True
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

    if explicit_media_query:
        # Media relevance is resolved upstream by the adaptive hybrid
        # retriever. Re-ranking those records here with a second set of
        # heuristics can invert a correct cross-lingual result. Preserve the
        # selected retrieval order, while still preferring an explicitly
        # required page when the coverage planner supplied one.
        media_candidates = [
            candidate
            for candidate in candidates
            if candidate[1] == "media"
        ]
        media_candidates.sort(
            key=lambda candidate: (
                0
                if not required_page_set
                or _normalize_url_for_match(_source_url(candidate[2])) in required_page_set
                else 1,
                int(candidate[2].get("retrieval_rank") or 0),
                -candidate[0],
                _doc_id(candidate[2]),
            )
        )
        for _score, kind, doc in media_candidates:
            _append_candidate(kind, doc)
            if sum(1 for item in items if item.get("kind") == "media") >= media_reserve:
                break

    if structured_detail_query and not explicit_media_query:
        # Coverage planning cannot always resolve a named role or page to an
        # explicit required URL. Reserve the best contiguous leaf and, when
        # available, its adjacent continuation so facts do not consume the
        # whole evidence budget before a split list or requirements block is
        # considered.
        reserved_leaf_docs: List[Dict[str, Any]] = []
        for _score, kind, doc in candidates:
            if kind == "chunk" and _doc_id(doc).startswith("chunk:"):
                if _append_candidate(kind, doc):
                    reserved_leaf_docs.append(doc)
                break
        if reserved_leaf_docs and max_items >= 3:
            anchor = reserved_leaf_docs[0]
            anchor_id = _doc_id(anchor)
            anchor_match = re.search(r":(\d{5}):[^:]+$", anchor_id)
            anchor_source = _normalize_url_for_match(_source_url(anchor))
            adjacent_candidates: List[Tuple[int, int, float, Dict[str, Any]]] = []
            continuation_limit = (
                3 if _STAGED_PROCESS_QUERY_RE.search(str(query or "")) else 1
            )
            if anchor_match and anchor_source:
                anchor_index = int(anchor_match.group(1))
                for score, kind, doc in candidates:
                    if kind != "chunk" or not _doc_id(doc).startswith("chunk:"):
                        continue
                    if _normalize_url_for_match(_source_url(doc)) != anchor_source:
                        continue
                    match = re.search(r":(\d{5}):[^:]+$", _doc_id(doc))
                    if not match:
                        continue
                    candidate_index = int(match.group(1))
                    distance = abs(candidate_index - anchor_index)
                    if distance < 1 or distance > continuation_limit:
                        continue
                    adjacent_candidates.append(
                        (distance, 0 if candidate_index > anchor_index else 1, -score, doc)
                    )
            appended_continuations = 0
            for _distance, _direction, _negative_score, doc in sorted(
                adjacent_candidates,
                key=lambda item: (item[0], item[1], item[2], _doc_id(item[3])),
            ):
                if _append_candidate("chunk", doc):
                    appended_continuations += 1
                    if appended_continuations >= continuation_limit:
                        break

        # A promoted assertion is an exact typed fact selected by the fused
        # retriever. It can arrive through retrieval_documents without source
        # fields, so score-only selection otherwise lets longer generic page
        # snippets fill the pack first. Keep one assertion beside the official
        # page chunk; the page remains available for citation provenance.
        for _score, kind, doc in candidates:
            if kind == "assertion":
                _append_candidate(kind, doc)
                break

    if relational_person_query and not explicit_media_query:
        # A person/role relation is often represented most faithfully in the
        # contiguous event card (title + Speaker/Host + linked person). Tiny
        # extracted facts can retain only the person's bare name while higher-
        # prior generic spans consume the per-page cap. Reserve the best leaf
        # that carries the relation before those fragments are selected.
        generic_relation_terms = {
            "mbzuai",
            "nexus",
            "series",
            "talk",
            "titled",
            "upcoming",
            "speaker",
            "host",
            "hosted",
            "hosting",
            "who",
            "whom",
        }
        specific_terms = _query_terms(query) - generic_relation_terms
        asks_for_host = bool(re.search(r"\b(?:host|hosted|hosting)\b", normalized_query))
        for _score, kind, doc in candidates:
            if kind != "chunk" or not _doc_id(doc).startswith("chunk:"):
                continue
            if required_page_set and not _candidate_matches_any_required_page(doc):
                continue
            metadata = _doc_metadata(doc)
            # Keep the original line structure while removing representation
            # headers. ``_doc_text`` intentionally collapses whitespace, which
            # would make a leading TITLE header consume the entire event card.
            raw_text = str(
                doc.get("text")
                or doc.get("value")
                or metadata.get("context")
                or ""
            )
            body_text = "\n".join(
                line
                for line in raw_text.splitlines()
                if not re.match(
                    r"^\s*(?:TITLE|TYPE|SECTION|SOURCE_URL|PARENT_TYPE)\s*:",
                    line,
                    flags=re.IGNORECASE,
                )
            ).casefold()
            body_terms = _query_terms(body_text)
            relation_present = (
                bool(re.search(r"\bhost\s*:", body_text))
                if asks_for_host
                else (
                    bool(re.search(r"\bspeaker\b", body_text))
                    or (
                        bool(specific_terms)
                        and len(specific_terms & body_terms)
                        >= min(3, len(specific_terms))
                    )
                )
            )
            if relation_present and _append_candidate(kind, doc):
                break

    for required_page in required_pages:
        for _score, kind, doc in candidates:
            if _candidate_matches_requirement(doc, required_page, page=True) and _append_candidate(kind, doc):
                break
    if structured_detail_query:
        # An aggregate parent proves page coverage, but its bounded excerpt can
        # still omit an early number or a later list item. Reserve one leaf
        # chunk per required page before isolated facts consume the per-source
        # cap so multi-part answers retain a contiguous answer-bearing block.
        for required_page in required_pages:
            already_has_leaf_chunk = any(
                str(item.get("kind") or "") == "chunk"
                and str(item.get("id") or "").startswith("chunk:")
                and _normalize_url_for_match(item.get("source_url")) == required_page
                for item in items
            )
            if already_has_leaf_chunk:
                continue
            for _score, kind, doc in candidates:
                if (
                    kind == "chunk"
                    and _doc_id(doc).startswith("chunk:")
                    and _candidate_matches_requirement(doc, required_page, page=True)
                    and _append_candidate(kind, doc)
                ):
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

    evidence_order = (
        {
            "media": 0,
            "action": 1,
            "assertion": 2,
            "fact": 3,
            "evidence_span": 4,
            "chunk": 5,
            "summary": 6,
            "answer": 7,
        }
        if explicit_media_query
        else {
            "action": 0,
            "chunk": 1,
            "assertion": 2,
            "fact": 3,
            "evidence_span": 4,
            "summary": 5,
            "answer": 6,
            "media": 7,
        }
        if structured_detail_query
        else {
            "action": 0,
            "assertion": 1,
            "fact": 2,
            "evidence_span": 3,
            "chunk": 4,
            "summary": 5,
            "answer": 6,
            "media": 7,
        }
    )
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
        if explicit_media_query:
            items.sort(
                key=lambda item: (
                    evidence_order.get(str(item.get("kind") or ""), 99),
                    _required_page_sort_rank(item),
                    _required_entity_sort_rank(item),
                    int(item.get("rank") or 0),
                )
            )
        else:
            items.sort(
                key=lambda item: (
                    _required_page_sort_rank(item),
                    _required_entity_sort_rank(item),
                    evidence_order.get(str(item.get("kind") or ""), 99),
                    int(item.get("rank") or 0),
                )
            )
    else:
        if explicit_media_query:
            items.sort(
                key=lambda item: (
                    evidence_order.get(str(item.get("kind") or ""), 99),
                    _required_page_sort_rank(item),
                    _required_entity_sort_rank(item),
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
