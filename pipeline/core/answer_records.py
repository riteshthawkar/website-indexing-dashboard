from __future__ import annotations

import re
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_EMAIL_RE = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?){2,}\d{3,4})")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:ac\.ae|edu|com|org|net)\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE)
_TIME_RANGE_RE = re.compile(
    r"\b\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\s*(?:-|to|–|—)\s*\d{1,2}[:.]\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
_DATE_HINT_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)

_CONTACT_QUALIFIER_TOKENS = {
    "admission",
    "admissions",
    "undergraduate",
    "graduate",
    "ug",
    "pg",
    "registrar",
    "student",
    "students",
    "campus",
    "facilities",
    "library",
    "medical",
    "it",
    "support",
    "security",
    "office",
    "official",
    "visitor",
    "visitors",
}

_EVENT_SCHEDULE_TOKENS = {
    "agenda",
    "arrival",
    "booklet",
    "conference",
    "details",
    "keynote",
    "panel",
    "program",
    "registration",
    "schedule",
    "session",
    "speaker",
    "summit",
    "talk",
    "welcome",
    "workshop",
}

_FACILITY_HOURS_TOKENS = {
    "canteen",
    "gym",
    "library",
    "medical",
    "pool",
    "security",
    "space",
    "support",
}

_LOCATION_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    (
        "location",
        re.compile(
            r"\b(?:is|was|are|were)?\s*(?:fully\s+integrated\s+with\s+the\s+city\s+of\s+)?"
            r"(?:located|based|situated)\s+(?:in|at)\s+([^.;]+)",
            re.IGNORECASE,
        ),
    ),
    (
        "emirate",
        re.compile(r"\bin\s+the\s+emirate\s+of\s+([^.;]+)", re.IGNORECASE),
    ),
)
_NAMING_PATTERN = re.compile(r"\bnamed\s+after\s+([^.;]+)", re.IGNORECASE)
_AFFILIATION_PATTERN = re.compile(r"\baffiliated\s+(?:to|with)\s+([^.;]+)", re.IGNORECASE)
_LEGAL_BASIS_PATTERN = re.compile(
    r"\b(?:established|created|formed)\b[^.]{0,120}?\b(?:law\s+no\.?\s*[^.;]+|by\s+law\s+[^.;]+|under\s+law\s+[^.;]+)",
    re.IGNORECASE,
)
_SERVICE_QA_PATTERN = re.compile(
    r"((?:does|do|is|are|can)\b[^?]{0,240}\?)\s*([^?]{1,320})",
    re.IGNORECASE,
)
_ROLE_HOLDER_PATTERN = (
    r"(?:(?i:speaker)\s+)?"
    r"(?:(?i:his excellency|h\.e\.|prof(?:essor)?\.?|dr\.?|mr\.?|ms\.?|mrs\.?)\s+)?"
    r"[A-Z][A-Za-z'().-]+(?:\s+\(?[A-Z][A-Za-z'().-]+\)?){0,7}"
)
_ROLE_ENTITY_PATTERNS: Sequence[Tuple[str, Sequence[str], Sequence[re.Pattern[str]]]] = (
    (
        "vice_president_chief_of_staff",
        ("vice president and chief of staff", "vp and chief of staff"),
        (
            re.compile(
                rf"(?P<holder>{_ROLE_HOLDER_PATTERN})\s+"
                rf"(?P<role>(?i:vice\s+president\s+and\s+chief\s+of\s+staff))\b",
            ),
        ),
    ),
    (
        "board_chair",
        (
            "chairman of mbzuai's board of trustees",
            "chair of mbzuai's board of trustees",
            "chairman of the mbzuai board of trustees",
            "chair of the mbzuai board of trustees",
            "chairman of the board of trustees",
            "chair of the board of trustees",
        ),
        (
            re.compile(
                rf"(?P<holder>{_ROLE_HOLDER_PATTERN})\s+"
                rf"(?P<role>(?i:chair(?:man)?\s+of\s+(?:mbzuai'?s\s+)?board\s+of\s+trustees))\b",
            ),
            re.compile(
                rf"(?i:under\s+the\s+chair(?:manship)?\s+of)\s+(?P<holder>{_ROLE_HOLDER_PATTERN})",
            ),
        ),
    ),
    (
        "provost",
        ("provost", "acting provost"),
        (
            re.compile(
                rf"(?P<role>(?<!Associate\s)(?<!Assistant\s)(?<!Deputy\s)(?i:(?:acting\s+)?provost))\s*,\s*(?P<holder>{_ROLE_HOLDER_PATTERN})",
            ),
            re.compile(
                rf"(?P<holder>{_ROLE_HOLDER_PATTERN})\s+"
                rf"(?P<role>(?<!Associate\s)(?<!Assistant\s)(?<!Deputy\s)(?i:(?:acting\s+)?provost))\b",
            ),
            re.compile(
                rf"(?P<holder>{_ROLE_HOLDER_PATTERN})\s+(?i:is\s+the)\s+"
                rf"(?P<role>(?<!Associate\s)(?<!Assistant\s)(?<!Deputy\s)(?i:(?:acting\s+)?provost))(?:\s+(?i:of)\s+(?P<subject>[^.;]+))?",
            ),
            re.compile(
                rf"(?P<role>(?<!Associate\s)(?<!Assistant\s)(?<!Deputy\s)(?i:(?:acting\s+)?provost))\s*[:\-]\s*(?P<holder>{_ROLE_HOLDER_PATTERN})",
            ),
        ),
    ),
    (
        "president",
        ("president", "interim president"),
        (
            re.compile(
                rf"(?P<role>(?<!Vice\s)(?i:(?:interim\s+)?president))\s*,\s*(?P<holder>{_ROLE_HOLDER_PATTERN})",
            ),
            re.compile(
                rf"(?P<holder>{_ROLE_HOLDER_PATTERN})\s+"
                rf"(?P<role>(?<!Vice\s)(?i:(?:interim\s+)?president(?:\s+and\s+university\s+professor)?))\b",
            ),
            re.compile(
                rf"(?P<holder>{_ROLE_HOLDER_PATTERN})\s+(?i:is\s+the)\s+"
                rf"(?P<role>(?<!Vice\s)(?i:(?:interim\s+)?president))\s+(?i:of)\s+(?P<subject>[^.;]+)",
            ),
            re.compile(
                rf"(?P<role>(?<!Vice\s)(?i:(?:interim\s+)?president))\s*[:\-]\s*(?P<holder>{_ROLE_HOLDER_PATTERN})",
            ),
        ),
    ),
)
_ROLE_GENERIC_PREFIX_RE = re.compile(r"^(?:speaker|guest speaker)\s+", re.IGNORECASE)
_ROLE_DISQUALIFYING_MODIFIERS = ("associate", "assistant", "deputy")
_ROLE_HOLDER_INVALID_TOKENS = {
    "is",
    "the",
    "of",
    "and",
    "acting",
    "vice",
    "president",
    "provost",
    "chair",
    "chairman",
    "chief",
    "staff",
    "board",
    "trustees",
    "mbzuai",
    "authority",
    "business",
    "center",
    "centre",
    "committee",
    "council",
    "department",
    "group",
    "institute",
    "language",
    "ministry",
    "natural",
    "office",
    "processing",
    "school",
    "university",
    "fast",
    "facts",
    "context",
    "image",
    "video",
}
_ROLE_HOLDER_PREFIX_TOKENS = {
    "speaker",
    "his",
    "excellency",
    "h.e",
    "h.e.",
    "prof",
    "prof.",
    "professor",
    "dr",
    "dr.",
    "mr",
    "mr.",
    "ms",
    "ms.",
    "mrs",
    "mrs.",
}
_ROLE_HOLDER_NAME_CONNECTORS = {
    "al",
    "bin",
    "da",
    "de",
    "del",
    "der",
    "di",
    "el",
    "la",
    "le",
    "van",
    "von",
}
_ROLE_SUBJECT_OF_RE = re.compile(
    r"\b(?:president|provost|vice\s+president\s+and\s+chief\s+of\s+staff|chair(?:man)?\s+of\s+(?:mbzuai'?s\s+)?board\s+of\s+trustees)\b\s+(?:of|at)\s+([^.;,]+)",
    re.IGNORECASE,
)
_ROLE_MBZUAI_MARKERS = (
    "mbzuai",
    "mohamed bin zayed university of artificial intelligence",
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _stable_id(*parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    return sha1((raw or "answer").encode("utf-8")).hexdigest()[:24]


def _token_variants(token: str) -> List[str]:
    value = "".join(ch for ch in str(token or "").lower() if ch.isalnum() or ch in {"-", "_", "'"}).strip("'")
    if not value:
        return []
    variants = [value]
    if value.endswith("'s"):
        variants.append(value[:-2])
    if value.endswith("ies") and len(value) > 4:
        variants.append(value[:-3] + "y")
    if value.endswith("s") and len(value) > 4 and not value.endswith("ss"):
        variants.append(value[:-1])
    return list(dict.fromkeys(item for item in variants if item))


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _clean_text(text).split():
        tokens.extend(_token_variants(token))
    return [token for token in tokens if token]


def _sentence_fragments(text: str) -> List[str]:
    clean_text = _clean_text(text)
    if not clean_text:
        return []
    normalized = (
        clean_text
        .replace("a.m.", "am")
        .replace("p.m.", "pm")
        .replace("A.M.", "AM")
        .replace("P.M.", "PM")
    )
    parts = re.split(r"(?<=[.!?])\s+|[\r\n]+", normalized)
    return [_clean_text(part) for part in parts if _clean_text(part)]


def _window_around(text: str, match: re.Match[str], *, width: int = 64) -> str:
    start = max(0, match.start() - width)
    end = min(len(text), match.end() + width)
    return _clean_text(text[start:end])


def _support_contact_window(text: str, match: re.Match[str], *, default_width: int = 64) -> str:
    window = _window_around(text, match, width=default_width)
    expanded = _window_around(text, match, width=220)
    expanded_lower = expanded.lower()
    if (
        any(marker in expanded_lower for marker in ("working hours", "technical support", "it team", "support"))
        and "8:00" in expanded_lower
        and ("12:30" in expanded_lower or "friday" in expanded_lower)
    ):
        fragments = _sentence_fragments(expanded)
        for index, fragment in enumerate(fragments):
            fragment_lower = fragment.lower()
            if match.group(0).lower() in fragment_lower and (
                "working hours" in fragment_lower or "technical support" in fragment_lower or "it team" in fragment_lower
            ):
                if "12:30" not in fragment_lower and index + 1 < len(fragments):
                    next_fragment = fragments[index + 1]
                    next_lower = next_fragment.lower()
                    if "working hours" in next_lower and ("12:30" in next_lower or "friday" in next_lower):
                        return _clean_text(f"{fragment} {next_fragment}")
                return fragment
        return expanded
    return window


def _extract_qualifiers(*values: str) -> List[str]:
    qualifiers = set()
    for value in values:
        qualifiers.update(set(_tokenize(value)) & _CONTACT_QUALIFIER_TOKENS)
    return sorted(qualifiers)


def _clean_role_holder(value: str) -> str:
    candidate = _clean_text(value)
    if not candidate:
        return ""
    candidate = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", candidate)
    candidate = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", candidate)
    candidate = re.sub(r"https?://\S+|www\.\S+", " ", candidate)
    candidate = candidate.replace("###", " ").replace("Context:", " ")
    candidate = _ROLE_GENERIC_PREFIX_RE.sub("", candidate)
    candidate = candidate.strip(" ,.:;")
    return _clean_text(candidate)


def _role_match_is_disqualified(
    *,
    canonical_role: str,
    fragment: str,
    match: re.Match[str],
    holder: str,
) -> bool:
    role_span = match.span("role") if "role" in match.groupdict() and match.groupdict().get("role") else (-1, -1)
    prefix = fragment[max(0, role_span[0] - 24): role_span[0]].lower() if role_span[0] >= 0 else ""
    holder_tail = (_clean_text(holder).split() or [""])[-1].lower()
    if canonical_role in {"president", "provost"}:
        if any(prefix.rstrip().endswith(f"{modifier} ") or holder_tail == modifier for modifier in _ROLE_DISQUALIFYING_MODIFIERS):
            return True
        if re.search(r"\b(?:associate|assistant|deputy)\s+$", prefix):
            return True
    return False


def _role_holder_looks_valid(holder: str) -> bool:
    candidate = _clean_text(holder)
    if not candidate:
        return False
    lower_candidate = candidate.lower()
    if any(
        marker in lower_candidate
        for marker in ("![", "](", "http://", "https://", ".jpg", ".jpeg", ".png", ".webp", "downloaded_page_images")
    ):
        return False
    words = [token for token in re.findall(r"[A-Za-z][A-Za-z'().-]*", candidate)]
    significant = [token for token in words if token.lower() not in _ROLE_HOLDER_PREFIX_TOKENS]
    if len(significant) < 2:
        return False
    lower_tokens = {token.lower() for token in significant}
    if lower_tokens & _ROLE_HOLDER_INVALID_TOKENS:
        return False
    name_tokens = [token for token in significant if token.lower() not in _ROLE_HOLDER_NAME_CONNECTORS]
    if len(name_tokens) < 2:
        return False
    capitalized = [
        token
        for token in name_tokens
        if token[:1].isupper() or token.startswith("(")
    ]
    if len(capitalized) < 2:
        return False
    if len(capitalized) / float(len(name_tokens)) < 0.75:
        return False
    if any(token.isupper() and len(token) >= 2 for token in significant):
        return False
    return True


def _relation_subject_looks_valid(subject: str) -> bool:
    candidate = _clean_text(subject)
    if not candidate:
        return True
    lower_candidate = candidate.lower()
    if "|" in candidate:
        return False
    if lower_candidate.startswith(("is ", "are ", "was ", "were ", "under ", "with ")):
        return False
    if lower_candidate.endswith((" and", " or", " with", " under", " by")):
        return False
    if re.search(r"\b(?:is|are|was|were|being|been)\b", lower_candidate):
        return False
    return True


def _legal_basis_value_looks_valid(value: str) -> bool:
    candidate = _clean_text(value)
    if not candidate:
        return False
    lower_candidate = candidate.lower()
    if "|" in candidate:
        return False
    if "law no" in lower_candidate and not re.search(r"\d", candidate):
        return False
    if lower_candidate in {"established under law no", "established by law no", "law no"}:
        return False
    return True


def answer_record_looks_valid(record: Dict[str, Any]) -> bool:
    if not isinstance(record, dict):
        return False
    answer_type = str(record.get("answer_type") or "")
    if answer_type == "legal_basis":
        return _relation_subject_looks_valid(str(record.get("subject_text") or "")) and _legal_basis_value_looks_valid(
            str(record.get("value") or record.get("text") or "")
        )
    if answer_type in {"affiliation", "named_after", "location"}:
        return _relation_subject_looks_valid(str(record.get("subject_text") or ""))
    if answer_type != "role_holder":
        return True
    holder = _clean_role_holder(str(record.get("value") or record.get("text") or ""))
    if not _role_holder_looks_valid(holder):
        return False
    subtype = str(record.get("answer_subtype") or "")
    lower_text = _clean_text(record.get("text") or "").lower()
    if subtype == "president" and "vice president" in lower_text:
        return False
    if subtype == "provost" and any(term in lower_text for term in ("associate provost", "assistant provost", "deputy provost")):
        return False
    return True


def _canonical_role_subtype(value: str) -> str:
    lower_value = _clean_text(value).lower()
    if "chief of staff" in lower_value:
        return "vice_president_chief_of_staff"
    if "board of trustees" in lower_value and "chair" in lower_value:
        return "board_chair"
    if "provost" in lower_value:
        return "provost"
    if "president" in lower_value:
        return "president"
    return ""


def _normalize_role_subject(value: str) -> str:
    candidate = _clean_text(value).strip(" .,:;")
    lower_candidate = candidate.lower()
    if any(marker in lower_candidate for marker in _ROLE_MBZUAI_MARKERS):
        return "MBZUAI"
    if lower_candidate in {"the university", "university"}:
        return "MBZUAI"
    if candidate.upper() == "UAE":
        return "UAE"
    return candidate


def _role_subject_text(
    fragment: str,
    role_text: str,
    source: Dict[str, Any],
    *,
    explicit_subject: str = "",
) -> str:
    if explicit_subject:
        return _normalize_role_subject(explicit_subject)
    lower_fragment = _clean_text(fragment).lower()
    if any(marker in lower_fragment for marker in _ROLE_MBZUAI_MARKERS):
        return "MBZUAI"
    canonical_role = _canonical_role_subtype(role_text)
    if canonical_role == "board_chair":
        return "MBZUAI"
    subject_match = _ROLE_SUBJECT_OF_RE.search(fragment)
    if subject_match:
        subject = _normalize_role_subject(subject_match.group(1))
        if subject:
            return subject
    for raw_value in (
        source.get("heading"),
        " ".join(source.get("section_path") or []),
        source.get("document_title"),
        source.get("source_url"),
    ):
        if any(marker in _clean_text(raw_value).lower() for marker in _ROLE_MBZUAI_MARKERS):
            return "MBZUAI"
    return _normalize_role_subject(_source_subject_fallback(source))


def _extract_relation_subject(fragment: str, *, match_start: int) -> str:
    prefix = _clean_text(fragment[: max(0, int(match_start))])
    if not prefix:
        return ""
    prefix = re.sub(r"^[\-\u2022*\d\.\)\s]+", "", prefix)
    words = prefix.split()
    if not words:
        return ""
    while words and words[-1].lower() in {
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "am",
        "has",
        "have",
        "had",
        "shall",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
    }:
        words.pop()
    if not words:
        return ""
    if len(words) > 8:
        words = words[-8:]
    return _clean_text(" ".join(words))


def _source_subject_fallback(source: Dict[str, Any]) -> str:
    for raw_value in (
        source.get("heading"),
        source.get("document_title"),
    ):
        candidate = _clean_text(raw_value)
        if not candidate:
            continue
        candidate = candidate.split(" > ")[0].strip()
        candidate = candidate.split(":")[0].strip() if ":" in candidate and len(candidate.split()) > 6 else candidate
        lower_candidate = candidate.lower()
        if lower_candidate in {
            "faq",
            "our history",
            "history",
            "parking",
            "campus facilities",
            "available services for students on campus",
            "video",
            "video: embedded video",
            "video: youtube video player",
        }:
            continue
        if any(
            marker in lower_candidate
            for marker in (
                "embedded media",
                "youtube video player",
                "watch video",
                "available services",
                "campus facilities",
            )
        ):
            continue
        if len(candidate.split()) <= 12:
            return candidate
    return ""


def _service_availability_subtype(text: str) -> str:
    lower_text = _clean_text(text).lower()
    if any(token in lower_text for token in ("shuttle", "transportation", "transport", "bus service", "navya bus", "rapid transport")):
        return "transport"
    if any(token in lower_text for token in ("student accommodation", "housing", "residences", "housing for parents", "visiting parents", "apartments")):
        return "accommodation"
    if any(token in lower_text for token in ("parking", "car park", "car parking", "visitor parking", "north car park")):
        return "parking"
    if any(
        token in lower_text
        for token in (
            "support services",
            "student facilities",
            "campus facilities",
            "student lounges",
            "health services",
            "dining facilities",
            "prayer rooms",
            "advising",
            "counseling",
        )
    ):
        return "amenities"
    return ""


def _service_availability_direct_signal(subtype: str, text: str) -> bool:
    lower_text = _clean_text(text).lower()
    if subtype == "transport":
        return any(
            token in lower_text
            for token in (
                "shuttle service",
                "shuttle bus service",
                "bus service",
                "connects students",
                "navya bus",
                "rapid transport",
            )
        )
    if subtype == "accommodation":
        return any(
            token in lower_text
            for token in (
                "student accommodation",
                "housing accommodation",
                "provided accommodation",
                "university accommodation",
                "residences at mbzuai",
                "housing for parents",
                "visiting parents",
                "apartments are equipped",
            )
        )
    if subtype == "parking":
        return "parking" in lower_text and any(
            token in lower_text
            for token in (
                "permitted",
                "available",
                "visitor parking",
                "car park",
                "car parking",
                "north parking lot",
            )
        )
    if subtype == "amenities":
        return any(
            token in lower_text
            for token in (
                "support services",
                "student facilities",
                "students have access to",
                "range of other services",
                "dining facilities",
                "student lounges",
                "health services",
                "prayer rooms",
            )
        )
    return False


def _base_record(source: Dict[str, Any], *, source_kind: str) -> Dict[str, Any]:
    linked_chunk_ids = [str(value) for value in (source.get("linked_chunk_ids") or []) if str(value)]
    if source_kind == "chunk" and not linked_chunk_ids and source.get("id"):
        linked_chunk_ids = [str(source["id"])]
    linked_parent_ids = [str(value) for value in (source.get("linked_parent_ids") or []) if str(value)]
    if source_kind == "chunk":
        for value in (source.get("section_key"), source.get("page_key")):
            if value:
                linked_parent_ids.append(str(value))
    linked_fact_ids = [str(source.get("id"))] if source_kind == "fact" and source.get("id") else []
    return {
        "document_id": str(source.get("document_id") or ""),
        "document_title": str(source.get("document_title") or ""),
        "document_type": str(source.get("document_type") or ""),
        "source_markdown_path": str(source.get("source_markdown_path") or ""),
        "source_url": str(source.get("source_url") or ""),
        "page_key": str(source.get("page_key") or ""),
        "section_key": str(source.get("section_key") or ""),
        "page_numbers": list(source.get("page_numbers") or []),
        "heading": str(source.get("heading") or ""),
        "linked_chunk_ids": list(dict.fromkeys(linked_chunk_ids)),
        "linked_parent_ids": list(dict.fromkeys(linked_parent_ids)),
        "linked_fact_ids": linked_fact_ids,
        "source_record_type": source_kind,
    }


def _make_answer_record(
    *,
    answer_type: str,
    answer_subtype: str,
    value: str,
    text: str,
    confidence: float,
    qualifiers: Iterable[str],
    subject_text: str,
    source: Dict[str, Any],
    source_kind: str,
) -> Dict[str, Any]:
    base = _base_record(source, source_kind=source_kind)
    clean_value = _clean_text(value)
    clean_text = _clean_text(text)
    qualifier_list = list(dict.fromkeys(str(value) for value in qualifiers if str(value)))
    return {
        "id": _stable_id(
            "answer",
            answer_type,
            answer_subtype,
            clean_value,
            tuple(base["linked_chunk_ids"]),
            tuple(base["linked_fact_ids"]),
            tuple(base["linked_parent_ids"]),
        ),
        "record_type": "answer",
        "answer_type": answer_type,
        "answer_subtype": answer_subtype,
        "value": clean_value,
        "text": clean_text,
        "dense_text": clean_text,
        "lexical_text": clean_text,
        "sparse_text": clean_text,
        "confidence": float(confidence),
        "qualifiers": qualifier_list,
        "subject_text": _clean_text(subject_text),
        **base,
    }


def _extract_contact_records(source: Dict[str, Any], *, source_kind: str, text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context = " ".join(
        part for part in (source.get("heading"), " ".join(source.get("section_path") or []), source.get("document_title")) if part
    )
    for regex, answer_type in ((_EMAIL_RE, "email"), (_PHONE_RE, "phone"), (_URL_RE, "website"), (_DOMAIN_RE, "website")):
        for match in regex.finditer(text):
            window = _support_contact_window(text, match) if answer_type == "email" else _window_around(text, match)
            lower_window = window.lower()
            confidence = 0.60
            if any(token in lower_window for token in ("contact", "email", "phone", "website", "directory", "reach", "mail")):
                confidence += 0.22
            if any(token in lower_window for token in ("admission", "registrar", "office", "support")):
                confidence += 0.10
            qualifiers = _extract_qualifiers(window, context)
            records.append(
                _make_answer_record(
                    answer_type=answer_type,
                    answer_subtype="contact",
                    value=match.group(0),
                    text=window,
                    confidence=confidence,
                    qualifiers=qualifiers,
                    subject_text="",
                    source=source,
                    source_kind=source_kind,
                )
            )
    return records


def _extract_hour_records(source: Dict[str, Any], *, source_kind: str, text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context = " ".join(
        part for part in (source.get("heading"), " ".join(source.get("section_path") or []), source.get("document_title")) if part
    )
    source_context = f"{context} {text}".lower()
    for fragment in _sentence_fragments(text):
        lower_fragment = fragment.lower()
        if not (_TIME_RE.search(fragment) or "working hours" in lower_fragment or "operating hours" in lower_fragment or "hours of operation" in lower_fragment):
            continue
        if len(fragment) > 320:
            continue
        subtype = "generic_hours"
        confidence = 0.48
        if any(token in source_context for token in ("technical support", "helpdesk", "it team", "screening exam", "host organization")):
            subtype = "support_hours"
            confidence = 0.42
        elif any(token in lower_fragment for token in ("official working hours", "official workings hours", "working hours", "operating hours", "office hours")):
            subtype = "operational_hours"
            confidence = 0.88
        elif any(token in lower_fragment for token in _EVENT_SCHEDULE_TOKENS):
            subtype = "event_schedule"
            confidence = 0.20
        elif any(token in lower_fragment for token in _FACILITY_HOURS_TOKENS) or "hours of operation" in lower_fragment:
            subtype = "facility_hours"
            confidence = 0.68
        value_match = _TIME_RANGE_RE.search(fragment)
        value = value_match.group(0) if value_match else fragment
        qualifiers = _extract_qualifiers(fragment, context)
        records.append(
            _make_answer_record(
                answer_type="hours",
                answer_subtype=subtype,
                value=value,
                text=fragment,
                confidence=confidence,
                qualifiers=qualifiers,
                subject_text="",
                source=source,
                source_kind=source_kind,
            )
        )
    return records


def _extract_date_records(source: Dict[str, Any], *, source_kind: str, text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context = " ".join(
        part for part in (source.get("heading"), " ".join(source.get("section_path") or []), source.get("document_title")) if part
    )
    for fragment in _sentence_fragments(text):
        lower_fragment = fragment.lower()
        if not (_DATE_HINT_RE.search(fragment) or "deadline" in lower_fragment):
            continue
        subtype = "generic_date"
        confidence = 0.50
        if "deadline" in lower_fragment:
            subtype = "deadline"
            confidence = 0.82
        elif any(token in lower_fragment for token in ("start date", "semester start", "orientation")):
            subtype = "start_date"
            confidence = 0.76
        elif any(token in lower_fragment for token in _EVENT_SCHEDULE_TOKENS):
            subtype = "event_date"
            confidence = 0.38
        records.append(
            _make_answer_record(
                answer_type="date",
                answer_subtype=subtype,
                value=fragment,
                text=fragment,
                confidence=confidence,
                qualifiers=_extract_qualifiers(fragment, context),
                subject_text="",
                source=source,
                source_kind=source_kind,
            )
        )
    return records


def _extract_location_records(source: Dict[str, Any], *, source_kind: str, text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for fragment in _sentence_fragments(text):
        for subtype, pattern in _LOCATION_PATTERNS:
            for match in pattern.finditer(fragment):
                value = _clean_text(match.group(1)).strip(" .,:;")
                if not value:
                    continue
                confidence = 0.72 if subtype == "location" else 0.82
                if "abu dhabi" in value.lower() or "masdar city" in value.lower():
                    confidence += 0.12
                records.append(
                    _make_answer_record(
                        answer_type="location",
                        answer_subtype=subtype,
                        value=value,
                        text=fragment,
                        confidence=confidence,
                        qualifiers=_extract_qualifiers(fragment, source.get("document_title", "")),
                        subject_text=_extract_relation_subject(fragment, match_start=match.start()),
                        source=source,
                        source_kind=source_kind,
                    )
                )
    return records


def _extract_relation_records(source: Dict[str, Any], *, source_kind: str, text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for fragment in _sentence_fragments(text):
        naming = _NAMING_PATTERN.search(fragment)
        if naming:
            subject_text = _extract_relation_subject(fragment, match_start=naming.start()) or _source_subject_fallback(source)
            records.append(
                _make_answer_record(
                    answer_type="named_after",
                    answer_subtype="entity_relation",
                    value=_clean_text(naming.group(1)).strip(" .,:;"),
                    text=fragment,
                    confidence=0.84,
                    qualifiers=_extract_qualifiers(fragment, source.get("document_title", "")),
                    subject_text=subject_text,
                    source=source,
                    source_kind=source_kind,
                )
            )
        affiliation = _AFFILIATION_PATTERN.search(fragment)
        if affiliation:
            subject_text = _extract_relation_subject(fragment, match_start=affiliation.start()) or _source_subject_fallback(source)
            records.append(
                _make_answer_record(
                    answer_type="affiliation",
                    answer_subtype="entity_relation",
                    value=_clean_text(affiliation.group(1)).strip(" .,:;"),
                    text=fragment,
                    confidence=0.84,
                    qualifiers=_extract_qualifiers(fragment, source.get("document_title", "")),
                    subject_text=subject_text,
                    source=source,
                    source_kind=source_kind,
                )
            )
        legal_basis = _LEGAL_BASIS_PATTERN.search(fragment)
        if legal_basis:
            subject_text = _extract_relation_subject(fragment, match_start=legal_basis.start()) or _source_subject_fallback(source)
            records.append(
                _make_answer_record(
                    answer_type="legal_basis",
                    answer_subtype="policy_relation",
                    value=_clean_text(legal_basis.group(0)),
                    text=fragment,
                    confidence=0.80,
                    qualifiers=_extract_qualifiers(fragment, source.get("document_title", "")),
                    subject_text=subject_text,
                    source=source,
                    source_kind=source_kind,
                )
            )
    return records


def _extract_role_records(source: Dict[str, Any], *, source_kind: str, text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    context = " ".join(
        part for part in (source.get("heading"), " ".join(source.get("section_path") or []), source.get("document_title")) if part
    )
    fragments = _sentence_fragments(text)
    for fragment in fragments:
        lower_fragment = fragment.lower()
        if not any(term in lower_fragment for term in ("president", "provost", "chief of staff", "board of trustees", "chairman", "chair of")):
            continue
        if len(fragment) > 360:
            continue
        for canonical_role, role_aliases, patterns in _ROLE_ENTITY_PATTERNS:
            if not any(alias in lower_fragment for alias in role_aliases):
                continue
            for pattern in patterns:
                for match in pattern.finditer(fragment):
                    holder = _clean_role_holder(match.groupdict().get("holder") or "")
                    if not holder:
                        continue
                    if not _role_holder_looks_valid(holder):
                        continue
                    if _role_match_is_disqualified(
                        canonical_role=canonical_role,
                        fragment=fragment,
                        match=match,
                        holder=holder,
                    ):
                        continue
                    role_text = _clean_text(match.groupdict().get("role") or canonical_role)
                    subject_text = _role_subject_text(
                        fragment,
                        role_text,
                        source,
                        explicit_subject=_clean_text(match.groupdict().get("subject") or ""),
                    )
                    if not subject_text:
                        continue
                    confidence = 0.78
                    if subject_text == "MBZUAI":
                        confidence += 0.12
                    if source_kind == "fact":
                        confidence += 0.04
                    records.append(
                        _make_answer_record(
                            answer_type="role_holder",
                            answer_subtype=canonical_role,
                            value=holder,
                            text=fragment,
                            confidence=confidence,
                            qualifiers=_extract_qualifiers(fragment, context, subject_text),
                            subject_text=subject_text,
                            source=source,
                            source_kind=source_kind,
                        )
                    )
    raw_lines = [
        _clean_text(re.sub(r"^[#>\-*.\d)\s]+", "", line))
        for line in re.split(r"[\r\n]+", str(text or ""))
        if _clean_text(re.sub(r"^[#>\-*.\d)\s]+", "", line))
    ]
    for index in range(len(raw_lines) - 1):
        holder_fragment = raw_lines[index]
        role_fragment = raw_lines[index + 1]
        if not holder_fragment or not role_fragment:
            continue
        holder = _clean_role_holder(holder_fragment)
        if not _role_holder_looks_valid(holder):
            continue
        lower_role_fragment = role_fragment.lower()
        for canonical_role, role_aliases, _patterns in _ROLE_ENTITY_PATTERNS:
            if not any(alias in lower_role_fragment for alias in role_aliases):
                continue
            if canonical_role == "president" and "vice president" in lower_role_fragment:
                continue
            if canonical_role == "provost" and any(term in lower_role_fragment for term in ("associate provost", "assistant provost", "deputy provost")):
                continue
            trailing_fragment = raw_lines[index + 2] if index + 2 < len(raw_lines) else ""
            subject_text = _role_subject_text(f"{holder} {role_fragment} {trailing_fragment}", role_fragment, source)
            if not subject_text:
                continue
            confidence = 0.82
            if subject_text == "MBZUAI":
                confidence += 0.08
            if source_kind == "fact":
                confidence += 0.04
            records.append(
                _make_answer_record(
                    answer_type="role_holder",
                    answer_subtype=canonical_role,
                    value=holder,
                    text=f"{holder_fragment} {role_fragment}",
                    confidence=confidence,
                    qualifiers=_extract_qualifiers(holder_fragment, role_fragment, context, subject_text),
                    subject_text=subject_text,
                    source=source,
                    source_kind=source_kind,
                )
            )
    return records


def _extract_service_availability_records(source: Dict[str, Any], *, source_kind: str, text: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    clean_text = _clean_text(text)
    context = " ".join(
        part for part in (source.get("heading"), " ".join(source.get("section_path") or []), source.get("document_title")) if part
    )

    qa_match = _SERVICE_QA_PATTERN.search(clean_text)
    if qa_match:
        question = _clean_text(qa_match.group(1))
        answer_text = _clean_text(qa_match.group(2))
        subtype = _service_availability_subtype(f"{question} {answer_text}")
        if subtype:
            value = answer_text.split(".")[0].strip()
            if value:
                records.append(
                    _make_answer_record(
                        answer_type="service_availability",
                        answer_subtype=subtype,
                        value=value,
                        text=f"{question} {answer_text}",
                        confidence=0.84,
                        qualifiers=_extract_qualifiers(question, answer_text, context),
                        subject_text="",
                        source=source,
                        source_kind=source_kind,
                    )
                )

    for fragment in _sentence_fragments(text):
        if len(fragment) > 360 or len(fragment.split()) > 48:
            continue
        subtype = _service_availability_subtype(fragment)
        if not subtype:
            continue
        if not _service_availability_direct_signal(subtype, fragment):
            continue
        lower_fragment = fragment.lower()
        confidence = 0.66
        if any(token in lower_fragment for token in ("yes", "no", "provided", "available", "permitted", "connects", "access to")):
            confidence += 0.12
        records.append(
            _make_answer_record(
                answer_type="service_availability",
                answer_subtype=subtype,
                value=fragment,
                text=fragment,
                confidence=confidence,
                qualifiers=_extract_qualifiers(fragment, context),
                subject_text="",
                source=source,
                source_kind=source_kind,
            )
        )
    return records


def _dedupe_answer_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for record in records:
        if not answer_record_looks_valid(record):
            continue
        key = (
            record.get("answer_type"),
            record.get("answer_subtype"),
            _clean_text(record.get("value")).lower(),
            tuple(record.get("linked_chunk_ids") or []),
            tuple(record.get("linked_fact_ids") or []),
        )
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = record
            continue
        existing_score = (
            float(existing.get("confidence") or 0.0),
            1 if existing.get("source_record_type") == "fact" else 0,
            -len(_clean_text(existing.get("text"))),
        )
        record_score = (
            float(record.get("confidence") or 0.0),
            1 if record.get("source_record_type") == "fact" else 0,
            -len(_clean_text(record.get("text"))),
        )
        if record_score > existing_score:
            best_by_key[key] = record
    return sorted(best_by_key.values(), key=lambda item: str(item.get("id") or ""))


def derive_answer_records_from_bundle(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not isinstance(bundle, dict):
        return []
    extracted: List[Dict[str, Any]] = []
    for source_kind, key in (("fact", "fact_records"), ("chunk", "chunk_records")):
        for source in bundle.get(key) or []:
            if not isinstance(source, dict):
                continue
            raw_text = str(source.get("text") or "")
            text = _clean_text(raw_text)
            if not text:
                continue
            extracted.extend(_extract_contact_records(source, source_kind=source_kind, text=text))
            extracted.extend(_extract_hour_records(source, source_kind=source_kind, text=text))
            extracted.extend(_extract_date_records(source, source_kind=source_kind, text=text))
            extracted.extend(_extract_location_records(source, source_kind=source_kind, text=text))
            extracted.extend(_extract_relation_records(source, source_kind=source_kind, text=text))
            extracted.extend(_extract_role_records(source, source_kind=source_kind, text=raw_text or text))
            extracted.extend(_extract_service_availability_records(source, source_kind=source_kind, text=text))
    return _dedupe_answer_records(extracted)
