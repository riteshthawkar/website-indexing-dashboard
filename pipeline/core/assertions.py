from __future__ import annotations

import re
from collections import defaultdict
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


_EMAIL_RE = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?){2,}\d{3,4})")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
_DATE_TOKEN_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b",
    re.IGNORECASE,
)
_TIME_TOKEN_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?(?:\s*(?:-|to|–|—)\s*\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)?)?\b",
    re.IGNORECASE,
)

_PREDICATE_ALIASES = {
    "role": "role_holder",
    "role_holder": "role_holder",
    "contact_email": "email",
    "email": "email",
    "contact_phone": "phone",
    "phone": "phone",
    "website": "website",
    "operating_hours": "hours",
    "hours": "hours",
    "date": "date",
    "location": "location",
    "located_in": "location",
    "named_after": "named_after",
    "affiliation": "affiliation",
    "affiliated_with": "affiliation",
    "legal_basis": "legal_basis",
    "established_under": "legal_basis",
    "service_availability": "service_availability",
    "program_area": "program_area",
}

_ANSWER_SUBTYPE_NORMALIZERS = {
    "email": lambda value: "email",
    "phone": lambda value: "phone",
    "website": lambda value: "website",
    "hours": lambda value: "hours",
    "date": lambda value: "date",
    "location": lambda value: "location",
    "named_after": lambda value: "named_after",
    "affiliation": lambda value: "affiliation",
    "legal_basis": lambda value: "legal_basis",
    "program_area": lambda value: "program_area",
}

_AUTHORITY_SCORES = {
    "canonical_page": 1.0,
    "policy_page": 0.96,
    "official_catalog": 0.94,
    "official_brochure": 0.86,
    "faq_page": 0.84,
    "news_page": 0.48,
    "event_page": 0.42,
    "external_or_embedded": 0.30,
}

_ROLE_QUERY_MARKERS = (
    "president",
    "provost",
    "vice president",
    "chief of staff",
    "board of trustees",
    "chairman",
    "chair of",
)
_ROLE_ORG_MARKERS = (
    "mbzuai",
    "mohamed bin zayed university of artificial intelligence",
    "board of trustees",
    "office of the president",
    "office of the provost",
    "university",
)
_ROLE_NAME_CONNECTORS = {
    "al",
    "bin",
    "da",
    "de",
    "del",
    "der",
    "di",
    "el",
    "ibn",
    "la",
    "le",
    "van",
    "von",
}


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values or []:
        text = clean_text(value)
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def stable_assertion_id(*parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if clean_text(part))
    if not raw:
        raw = "assertion"
    return f"assertion:{sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def stable_entity_id(*parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if clean_text(part))
    if not raw:
        raw = "entity"
    return f"entity:{sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def coerce_confidence(value: Any, *, default: float = 0.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return float(default)
    return max(0.0, min(1.0, score))


def normalize_predicate(value: Any) -> str:
    predicate = clean_text(value).lower().replace(" ", "_")
    return _PREDICATE_ALIASES.get(predicate, predicate or "assertion")


def normalize_entity_type(value: Any) -> str:
    entity_type = clean_text(value).lower().replace(" ", "_")
    if not entity_type:
        return "other"
    if entity_type in {"organization", "organisation", "org"}:
        return "organization"
    if entity_type in {"person", "people"}:
        return "person"
    if entity_type in {"place", "city", "country"}:
        return "location"
    return entity_type


def normalize_answer_subtype(answer_type: str, subtype: Any, value: Any = "") -> str:
    normalized = clean_text(subtype).lower().replace(" ", "_")
    if normalized:
        return normalized
    normalizer = _ANSWER_SUBTYPE_NORMALIZERS.get(answer_type)
    if normalizer is not None:
        return normalizer(value)
    return answer_type


def normalize_value(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if _EMAIL_RE.search(text):
        return _EMAIL_RE.search(text).group(0).lower()
    if _URL_RE.search(text):
        matched = _URL_RE.search(text).group(0)
        return matched.rstrip(".,);")
    if _PHONE_RE.search(text):
        return clean_text(_PHONE_RE.search(text).group(0))
    return text


def _looks_person_name(value: Any) -> bool:
    text = clean_text(value)
    if not text:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z'().-]*", text)
    if len(words) < 2:
        return False
    significant = [token for token in words if token.lower() not in _ROLE_NAME_CONNECTORS]
    if len(significant) < 2:
        return False
    capitalized = [token for token in significant if token[:1].isupper() or token.startswith("(")]
    if len(capitalized) < 2:
        return False
    if len(capitalized) / float(len(significant)) < 0.7:
        return False
    lower_text = text.lower()
    if any(marker in lower_text for marker in _ROLE_QUERY_MARKERS):
        return False
    return True


def _looks_role_subject_reference(value: Any) -> bool:
    text = clean_text(value).lower()
    if not text:
        return False
    return any(marker in text for marker in _ROLE_ORG_MARKERS)


def _normalize_role_subject_text(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    lower_text = text.lower()
    if "mbzuai" in lower_text or "mohamed bin zayed university of artificial intelligence" in lower_text:
        return "MBZUAI"
    match = re.search(
        r"(?:president|provost|vice\s+president\s+and\s+chief\s+of\s+staff|chair(?:man)?\s+of\s+(?:the\s+)?board\s+of\s+trustees)\s+(?:of|at)\s+(.+)",
        text,
        re.IGNORECASE,
    )
    if match:
        extracted = clean_text(match.group(1))
        if "mbzuai" in extracted.lower() or "mohamed bin zayed university of artificial intelligence" in extracted.lower():
            return "MBZUAI"
        return extracted
    return text


def _role_answer_subject_and_value(assertion: Mapping[str, Any]) -> Tuple[str, str]:
    subject_name = clean_text(assertion.get("subject_name"))
    subject_type = normalize_entity_type(assertion.get("subject_type"))
    object_name = clean_text(assertion.get("object_value") or assertion.get("object_name"))
    object_type = normalize_entity_type(assertion.get("object_type"))

    subject_is_person = subject_type == "person" or _looks_person_name(subject_name)
    object_is_person = object_type == "person" or _looks_person_name(object_name)
    subject_is_role_target = _looks_role_subject_reference(subject_name)
    object_is_role_target = _looks_role_subject_reference(object_name)

    if subject_is_person and not object_is_person:
        return _normalize_role_subject_text(object_name), subject_name
    if object_is_person and not subject_is_person:
        return _normalize_role_subject_text(subject_name), object_name
    if subject_is_role_target and not object_is_role_target:
        return _normalize_role_subject_text(subject_name), object_name
    if object_is_role_target and not subject_is_role_target:
        return _normalize_role_subject_text(object_name), subject_name
    return _normalize_role_subject_text(subject_name), object_name


def infer_authority_class(
    *,
    source_url: str = "",
    document_title: str = "",
    document_type: str = "",
    source_markdown_path: str = "",
) -> str:
    url = clean_text(source_url).lower()
    title = clean_text(document_title).lower()
    doc_type = clean_text(document_type).lower()
    path = clean_text(source_markdown_path).lower()
    combined = " ".join(part for part in (url, title, doc_type, path) if part)

    if any(token in combined for token in ("leadership", "office-of-the-president", "office-of-the-provost")):
        return "canonical_page"
    if any(token in combined for token in ("policy", "regulation", "law", "governance")):
        return "policy_page"
    if any(token in combined for token in ("catalogue", "catalog", "handbook", "university_catalogue")):
        return "official_catalog"
    if any(token in combined for token in ("brochure", "prospectus")):
        return "official_brochure"
    if "faq" in combined:
        return "faq_page"
    if any(token in combined for token in ("news", "press-release", "press release")):
        return "news_page"
    if any(token in combined for token in ("event", "conference", "summit", "workshop", "booklet", "program")):
        return "event_page"
    if "mbzuai.ac.ae" in combined:
        return "canonical_page"
    return "external_or_embedded"


def authority_score(
    *,
    source_url: str = "",
    document_title: str = "",
    document_type: str = "",
    source_markdown_path: str = "",
) -> float:
    authority_class = infer_authority_class(
        source_url=source_url,
        document_title=document_title,
        document_type=document_type,
        source_markdown_path=source_markdown_path,
    )
    return float(_AUTHORITY_SCORES.get(authority_class, 0.0))


def freshness_score(*, source_url: str = "", document_title: str = "", qualifiers: Sequence[Any] | None = None) -> float:
    text = " ".join(
        clean_text(part).lower()
        for part in (source_url, document_title, *list(qualifiers or []))
        if clean_text(part)
    )
    if any(token in text for token in ("former", "previous", "past", "historic", "historical", "2022", "2023")):
        return 0.35
    if any(token in text for token in ("2024", "2025", "2026", "current", "present")):
        return 1.0
    return 0.75


def assertion_text(
    *,
    subject_name: str,
    predicate: str,
    object_value: str,
    answer_type: str = "",
    answer_subtype: str = "",
    qualifiers: Sequence[Any] | None = None,
) -> str:
    subject = clean_text(subject_name) or "This entity"
    value = clean_text(object_value)
    subtype = clean_text(answer_subtype).replace("_", " ")
    predicate = normalize_predicate(predicate)
    qualifiers_text = ", ".join(unique_strings(qualifiers or []))

    if predicate == "role_holder":
        text = f"{value} is the {subtype or 'role holder'} of {subject}."
    elif predicate == "email":
        text = f"The email address for {subject} is {value}."
    elif predicate == "phone":
        text = f"The phone number for {subject} is {value}."
    elif predicate == "website":
        text = f"The website for {subject} is {value}."
    elif predicate == "hours":
        text = f"The hours for {subject} are {value}."
    elif predicate == "date":
        text = f"The date for {subject} is {value}."
    elif predicate == "location":
        text = f"{subject} is located in {value}."
    elif predicate == "named_after":
        text = f"{subject} is named after {value}."
    elif predicate == "affiliation":
        text = f"{subject} is affiliated with {value}."
    elif predicate == "legal_basis":
        text = f"{subject} was established under {value}."
    elif predicate == "service_availability":
        label = subtype or "service"
        text = f"{subject}: {label} - {value}."
    elif predicate == "program_area":
        text = f"{subject} covers the program area {value}."
    else:
        text = f"{subject} {predicate.replace('_', ' ')} {value}."

    if qualifiers_text:
        return f"{text} Qualifiers: {qualifiers_text}."
    return text


def _value_entity_type(answer_type: str, object_type: str) -> str:
    object_type = normalize_entity_type(object_type)
    if object_type and object_type != "other":
        return object_type
    if answer_type in {"email", "phone", "website", "hours", "date"}:
        return answer_type
    if answer_type in {"named_after", "role_holder"}:
        return "person"
    if answer_type in {"location"}:
        return "location"
    if answer_type in {"affiliation"}:
        return "organization"
    if answer_type in {"legal_basis"}:
        return "law"
    return "other"


def build_entity_record(
    *,
    canonical_name: str,
    entity_type: str,
    aliases: Sequence[Any] | None = None,
    description: str = "",
    confidence: float = 0.0,
    source_chunk_ids: Sequence[Any] | None = None,
    source_parent_ids: Sequence[Any] | None = None,
    source_urls: Sequence[Any] | None = None,
    document_titles: Sequence[Any] | None = None,
    entity_id: str | None = None,
) -> Dict[str, Any]:
    canonical_name = clean_text(canonical_name)
    entity_type = normalize_entity_type(entity_type)
    aliases_list = unique_strings([canonical_name, *(aliases or [])])
    return {
        "id": entity_id or stable_entity_id(entity_type, canonical_name),
        "entity_type": entity_type,
        "canonical_name": canonical_name,
        "aliases": aliases_list,
        "description": clean_text(description),
        "confidence": coerce_confidence(confidence, default=0.0),
        "source_chunk_ids": unique_strings(source_chunk_ids or []),
        "source_parent_ids": unique_strings(source_parent_ids or []),
        "source_urls": unique_strings(source_urls or []),
        "document_titles": unique_strings(document_titles or []),
    }


def build_assertion_record(
    *,
    subject_name: str,
    predicate: str,
    object_value: str,
    answer_type: str = "",
    answer_subtype: str = "",
    subject_type: str = "organization",
    object_type: str = "",
    qualifiers: Sequence[Any] | None = None,
    support_span: str = "",
    confidence: float = 0.0,
    validator_confidence: float = 0.0,
    authority_class: str = "",
    authority_score_value: float | None = None,
    freshness_score_value: float | None = None,
    source_doc_id: str = "",
    source_slice_id: str = "",
    source_chunk_ids: Sequence[Any] | None = None,
    source_parent_ids: Sequence[Any] | None = None,
    source_fact_ids: Sequence[Any] | None = None,
    source_url: str = "",
    source_markdown_path: str = "",
    document_title: str = "",
    source_last_seen: str = "",
    validator_decision: str = "supported",
    assertion_id: str | None = None,
) -> Dict[str, Any]:
    predicate = normalize_predicate(predicate or answer_type)
    answer_type = normalize_predicate(answer_type or predicate)
    answer_subtype = normalize_answer_subtype(answer_type, answer_subtype, object_value)
    subject_name = clean_text(subject_name)
    object_value = normalize_value(object_value)
    qualifiers_list = unique_strings(qualifiers or [])
    authority_class = clean_text(authority_class) or infer_authority_class(
        source_url=source_url,
        document_title=document_title,
        source_markdown_path=source_markdown_path,
    )
    if authority_score_value is None:
        authority_score_value = authority_score(
            source_url=source_url,
            document_title=document_title,
            source_markdown_path=source_markdown_path,
        )
    if freshness_score_value is None:
        freshness_score_value = freshness_score(
            source_url=source_url,
            document_title=document_title,
            qualifiers=qualifiers_list,
        )
    subject_entity_id = stable_entity_id(normalize_entity_type(subject_type), subject_name)
    object_entity_id = stable_entity_id(_value_entity_type(answer_type, object_type), object_value)
    return {
        "id": assertion_id
        or stable_assertion_id(subject_name, predicate, answer_subtype, object_value, source_url, source_doc_id),
        "subject_name": subject_name,
        "subject_type": normalize_entity_type(subject_type),
        "subject_entity_id": subject_entity_id,
        "predicate": predicate,
        "relation_type": predicate,
        "answer_type": answer_type,
        "answer_subtype": answer_subtype,
        "object_name": object_value,
        "object_value": object_value,
        "object_type": _value_entity_type(answer_type, object_type),
        "object_entity_id": object_entity_id,
        "qualifiers": qualifiers_list,
        "support_span": clean_text(support_span),
        "evidence": clean_text(support_span),
        "confidence": max(
            coerce_confidence(confidence, default=0.0),
            coerce_confidence(validator_confidence, default=0.0),
        ),
        "validator_confidence": coerce_confidence(validator_confidence, default=0.0),
        "validator_decision": clean_text(validator_decision or "supported").lower(),
        "authority_class": authority_class,
        "authority_score": coerce_confidence(authority_score_value, default=0.0),
        "freshness_score": coerce_confidence(freshness_score_value, default=0.0),
        "source_doc_id": clean_text(source_doc_id),
        "source_slice_id": clean_text(source_slice_id),
        "source_chunk_ids": unique_strings(source_chunk_ids or []),
        "source_parent_ids": unique_strings(source_parent_ids or []),
        "source_fact_ids": unique_strings(source_fact_ids or []),
        "source_url": clean_text(source_url),
        "source_markdown_path": clean_text(source_markdown_path),
        "document_title": clean_text(document_title),
        "source_last_seen": clean_text(source_last_seen),
        "canonical_subject": subject_entity_id,
        "canonical_predicate": predicate,
        "canonical_object": object_value.casefold(),
        "validity_status": "active",
        "text": assertion_text(
            subject_name=subject_name,
            predicate=predicate,
            object_value=object_value,
            answer_type=answer_type,
            answer_subtype=answer_subtype,
            qualifiers=qualifiers_list,
        ),
        "source_id": clean_text(source_chunk_ids[0] if source_chunk_ids else source_doc_id),
        "source_kind": "chunk" if source_chunk_ids else "document",
    }


def build_entity_records_from_assertions(assertions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    def _merge(entity_id: str, payload: Dict[str, Any]) -> None:
        current = merged.get(entity_id)
        if current is None:
            merged[entity_id] = payload
            return
        current["aliases"] = unique_strings([*(current.get("aliases") or []), *(payload.get("aliases") or [])])
        current["source_chunk_ids"] = unique_strings(
            [*(current.get("source_chunk_ids") or []), *(payload.get("source_chunk_ids") or [])]
        )
        current["source_parent_ids"] = unique_strings(
            [*(current.get("source_parent_ids") or []), *(payload.get("source_parent_ids") or [])]
        )
        current["source_urls"] = unique_strings([*(current.get("source_urls") or []), *(payload.get("source_urls") or [])])
        current["document_titles"] = unique_strings(
            [*(current.get("document_titles") or []), *(payload.get("document_titles") or [])]
        )
        current["confidence"] = max(
            coerce_confidence(current.get("confidence"), default=0.0),
            coerce_confidence(payload.get("confidence"), default=0.0),
        )
        if not current.get("description") and payload.get("description"):
            current["description"] = payload["description"]

    for assertion in assertions or []:
        if not isinstance(assertion, Mapping):
            continue
        source_chunk_ids = assertion.get("source_chunk_ids") or []
        source_parent_ids = assertion.get("source_parent_ids") or []
        source_urls = [assertion.get("source_url")] if assertion.get("source_url") else []
        document_titles = [assertion.get("document_title")] if assertion.get("document_title") else []

        subject_name = clean_text(assertion.get("subject_name"))
        if subject_name:
            subject_record = build_entity_record(
                canonical_name=subject_name,
                entity_type=assertion.get("subject_type") or "organization",
                aliases=[subject_name],
                confidence=assertion.get("confidence"),
                source_chunk_ids=source_chunk_ids,
                source_parent_ids=source_parent_ids,
                source_urls=source_urls,
                document_titles=document_titles,
                entity_id=assertion.get("subject_entity_id") or stable_entity_id(assertion.get("subject_type"), subject_name),
            )
            _merge(subject_record["id"], subject_record)

        object_value = clean_text(assertion.get("object_value") or assertion.get("object_name"))
        if object_value:
            object_record = build_entity_record(
                canonical_name=object_value,
                entity_type=assertion.get("object_type") or _value_entity_type(
                    normalize_predicate(assertion.get("answer_type") or assertion.get("predicate")),
                    assertion.get("object_type") or "",
                ),
                aliases=[object_value],
                confidence=assertion.get("confidence"),
                source_chunk_ids=source_chunk_ids,
                source_parent_ids=source_parent_ids,
                source_urls=source_urls,
                document_titles=document_titles,
                entity_id=assertion.get("object_entity_id") or stable_entity_id(assertion.get("object_type"), object_value),
            )
            _merge(object_record["id"], object_record)

    return sorted(merged.values(), key=lambda item: (item.get("entity_type", ""), item.get("canonical_name", "")))


def build_answer_records_from_assertions(assertions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    best_by_key: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    for assertion in assertions or []:
        if not isinstance(assertion, Mapping):
            continue
        if clean_text(assertion.get("validity_status") or "active").lower() not in {"active", "valid"}:
            continue
        answer_type = normalize_predicate(assertion.get("answer_type") or assertion.get("predicate"))
        answer_subtype = normalize_answer_subtype(
            answer_type,
            assertion.get("answer_subtype"),
            assertion.get("object_value") or assertion.get("object_name"),
        )
        if answer_type == "role_holder":
            subject, value = _role_answer_subject_and_value(assertion)
        else:
            subject = clean_text(assertion.get("subject_name"))
            value = clean_text(assertion.get("object_value") or assertion.get("object_name"))
        if not answer_type or not subject or not value:
            continue
        key = (answer_type, answer_subtype, subject.casefold(), value.casefold())
        score = (
            coerce_confidence(assertion.get("confidence"), default=0.0),
            coerce_confidence(assertion.get("authority_score"), default=0.0),
            coerce_confidence(assertion.get("freshness_score"), default=0.0),
            len(assertion.get("source_chunk_ids") or []),
        )
        current = best_by_key.get(key)
        if current is not None:
            current_score = (
                coerce_confidence(current.get("confidence"), default=0.0),
                coerce_confidence(current.get("authority_score"), default=0.0),
                coerce_confidence(current.get("freshness_score"), default=0.0),
                len(current.get("linked_chunk_ids") or []),
            )
            if score <= current_score:
                continue

        text = assertion_text(
            subject_name=subject,
            predicate=answer_type,
            object_value=value,
            answer_type=answer_type,
            answer_subtype=answer_subtype,
            qualifiers=assertion.get("qualifiers") or [],
        )
        best_by_key[key] = {
            "id": clean_text(assertion.get("id")) or stable_assertion_id(subject, answer_type, answer_subtype, value),
            "answer_type": answer_type,
            "answer_subtype": answer_subtype,
            "subject_text": subject,
            "value": value,
            "text": text,
            "qualifiers": unique_strings(assertion.get("qualifiers") or []),
            "confidence": coerce_confidence(assertion.get("confidence"), default=0.0),
            "authority_class": clean_text(assertion.get("authority_class")),
            "authority_score": coerce_confidence(assertion.get("authority_score"), default=0.0),
            "freshness_score": coerce_confidence(assertion.get("freshness_score"), default=0.0),
            "source_last_seen": clean_text(assertion.get("source_last_seen")),
            "validity_status": clean_text(assertion.get("validity_status") or "active").lower(),
            "source_record_type": "assertion",
            "source_record_id": clean_text(assertion.get("id")),
            "linked_chunk_ids": unique_strings(assertion.get("source_chunk_ids") or []),
            "linked_span_ids": unique_strings(assertion.get("source_span_ids") or assertion.get("linked_span_ids") or []),
            "source_span_ids": unique_strings(assertion.get("source_span_ids") or assertion.get("linked_span_ids") or []),
            "linked_parent_ids": unique_strings(assertion.get("source_parent_ids") or []),
            "linked_fact_ids": unique_strings(assertion.get("source_fact_ids") or []),
            "document_title": clean_text(assertion.get("document_title")),
            "source_url": clean_text(assertion.get("source_url")),
            "source_markdown_path": clean_text(assertion.get("source_markdown_path")),
            "support_span": clean_text(assertion.get("support_span") or assertion.get("evidence")),
        }

    answers = sorted(
        best_by_key.values(),
        key=lambda item: (
            -coerce_confidence(item.get("confidence"), default=0.0),
            -coerce_confidence(item.get("authority_score"), default=0.0),
            -coerce_confidence(item.get("freshness_score"), default=0.0),
            item.get("answer_type", ""),
            item.get("answer_subtype", ""),
            item.get("value", ""),
        ),
    )
    return answers


def build_assertion_embedding_records(assertions: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for assertion in assertions or []:
        if not isinstance(assertion, Mapping):
            continue
        if clean_text(assertion.get("validity_status") or "active").lower() not in {"active", "valid"}:
            continue
        record_id = clean_text(assertion.get("id"))
        text = clean_text(assertion.get("text") or assertion.get("evidence"))
        if not record_id or not text:
            continue
        answer_type = normalize_predicate(assertion.get("answer_type") or assertion.get("predicate"))
        answer_subtype = normalize_answer_subtype(
            answer_type,
            assertion.get("answer_subtype"),
            assertion.get("object_value") or assertion.get("object_name"),
        )
        dense_text = "\n".join(
            part
            for part in (
                f"SUBJECT: {clean_text(assertion.get('subject_name'))}",
                f"PREDICATE: {answer_type}",
                f"SUBTYPE: {answer_subtype}" if answer_subtype else "",
                f"OBJECT: {clean_text(assertion.get('object_value') or assertion.get('object_name'))}",
                f"DOCUMENT: {clean_text(assertion.get('document_title'))}" if assertion.get("document_title") else "",
                f"SOURCE_URL: {clean_text(assertion.get('source_url'))}" if assertion.get("source_url") else "",
                "",
                text,
            )
            if part is not None and part != ""
        )
        records.append(
            {
                "id": record_id,
                "record_type": "assertion",
                "text": text,
                "dense_text": dense_text,
                "lexical_text": text,
                "sparse_text": text,
                "subject_name": clean_text(assertion.get("subject_name")),
                "subject_type": clean_text(assertion.get("subject_type")),
                "subject_entity_id": clean_text(assertion.get("subject_entity_id")),
                "object_value": clean_text(assertion.get("object_value") or assertion.get("object_name")),
                "object_type": clean_text(assertion.get("object_type")),
                "object_entity_id": clean_text(assertion.get("object_entity_id")),
                "predicate": answer_type,
                "answer_type": answer_type,
                "answer_subtype": answer_subtype,
                "qualifiers": unique_strings(assertion.get("qualifiers") or []),
                "confidence": coerce_confidence(assertion.get("confidence"), default=0.0),
                "authority_class": clean_text(assertion.get("authority_class")),
                "authority_score": coerce_confidence(assertion.get("authority_score"), default=0.0),
                "freshness_score": coerce_confidence(assertion.get("freshness_score"), default=0.0),
                "source_last_seen": clean_text(assertion.get("source_last_seen")),
                "validity_status": clean_text(assertion.get("validity_status") or "active").lower(),
                "canonical_subject": clean_text(assertion.get("canonical_subject") or assertion.get("subject_entity_id")),
                "canonical_predicate": clean_text(assertion.get("canonical_predicate") or answer_type),
                "canonical_object": clean_text(assertion.get("canonical_object") or assertion.get("object_value") or assertion.get("object_name")).casefold(),
                "source_chunk_ids": unique_strings(assertion.get("source_chunk_ids") or []),
                "source_span_ids": unique_strings(assertion.get("source_span_ids") or assertion.get("linked_span_ids") or []),
                "linked_span_ids": unique_strings(assertion.get("source_span_ids") or assertion.get("linked_span_ids") or []),
                "source_parent_ids": unique_strings(assertion.get("source_parent_ids") or []),
                "source_fact_ids": unique_strings(assertion.get("source_fact_ids") or []),
                "document_title": clean_text(assertion.get("document_title")),
                "source_url": clean_text(assertion.get("source_url")),
                "source_markdown_path": clean_text(assertion.get("source_markdown_path")),
            }
        )
    return records


def merge_answer_records(
    primary: Sequence[Mapping[str, Any]],
    fallback: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    def _ingest(records: Sequence[Mapping[str, Any]]) -> None:
        for record in records or []:
            if not isinstance(record, Mapping):
                continue
            answer_type = clean_text(record.get("answer_type")).lower()
            answer_subtype = clean_text(record.get("answer_subtype")).lower()
            subject = clean_text(record.get("subject_text")).lower()
            value = clean_text(record.get("value") or record.get("text")).lower()
            if not answer_type or not value:
                continue
            key = (answer_type, answer_subtype, subject, value)
            if key not in merged:
                merged[key] = dict(record)

    _ingest(primary)
    _ingest(fallback)
    return list(merged.values())


def merge_entity_records(
    primary: Sequence[Mapping[str, Any]],
    fallback: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for record in [*(primary or []), *(fallback or [])]:
        if not isinstance(record, Mapping):
            continue
        record_id = clean_text(record.get("id"))
        if not record_id:
            continue
        current = by_id.get(record_id)
        if current is None:
            by_id[record_id] = dict(record)
            continue
        current["aliases"] = unique_strings([*(current.get("aliases") or []), *(record.get("aliases") or [])])
        current["source_chunk_ids"] = unique_strings(
            [*(current.get("source_chunk_ids") or []), *(record.get("source_chunk_ids") or [])]
        )
        current["source_parent_ids"] = unique_strings(
            [*(current.get("source_parent_ids") or []), *(record.get("source_parent_ids") or [])]
        )
        current["source_urls"] = unique_strings([*(current.get("source_urls") or []), *(record.get("source_urls") or [])])
        current["document_titles"] = unique_strings(
            [*(current.get("document_titles") or []), *(record.get("document_titles") or [])]
        )
        current["confidence"] = max(
            coerce_confidence(current.get("confidence"), default=0.0),
            coerce_confidence(record.get("confidence"), default=0.0),
        )
    return list(by_id.values())


def keyword_anchors_from_query(query: str) -> List[str]:
    query_text = clean_text(query)
    anchors: List[str] = []
    for regex in (_EMAIL_RE, _PHONE_RE, _URL_RE, _DATE_TOKEN_RE, _TIME_TOKEN_RE):
        for match in regex.findall(query_text):
            token = clean_text(match)
            if token and token not in anchors:
                anchors.append(token)
    return anchors


def aggregate_assertion_metrics(assertions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, int] = defaultdict(int)
    by_source: Dict[str, int] = defaultdict(int)
    for assertion in assertions or []:
        if not isinstance(assertion, Mapping):
            continue
        by_type[normalize_predicate(assertion.get("answer_type") or assertion.get("predicate"))] += 1
        by_source[clean_text(assertion.get("authority_class")) or "unknown"] += 1
    return {
        "assertion_count": sum(by_type.values()),
        "assertions_by_type": dict(sorted(by_type.items())),
        "assertions_by_authority": dict(sorted(by_source.items())),
    }
