from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from pipeline.core.openai_client import json_completion, make_openai_client

_TOKEN_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_PREMISE_STOPWORDS = {
    "a", "about", "all", "an", "and", "are", "at", "be", "by", "can",
    "do", "does", "every", "for", "from", "have", "how", "i", "if", "in",
    "is", "it", "many", "me", "my", "of", "on", "or", "should", "the",
    "their", "this", "to", "use", "uses", "using", "was", "were", "what",
    "when", "where", "which", "who", "why", "will", "with", "you", "your",
    "mbzuai", "mbzuais", "s", "mohamed", "bin", "zayed", "university", "artificial",
    "intelligence", "page", "official", "exact", "number", "address",
    "requirements", "required", "admission", "applicants", "applicant",
    "program", "programme", "programs", "degree", "phd", "master", "masters",
    "doctorate", "office", "campus", "center", "centre", "research", "housing",
    "airport", "wallet", "fee", "fees", "year", "academic", "opening", "hours",
    "ما", "ماذا", "من", "متى", "أين", "اين", "كيف", "كم", "هل", "في", "على",
    "إلى", "الى", "عن", "أن", "ان", "التي", "الذي", "هذه", "هذا", "هو", "هي",
    "جامعة", "الجامعة", "جامعه", "محمد", "بن", "زايد", "للذكاء", "الاصطناعي",
    "الصفحة", "صفحة", "بحسب", "اذكر", "جميع", "كل", "دقيق", "الدقيق", "الدقيقة",
    "ينبغي", "يجب", "رقم", "عنوان", "متطلبات", "القبول", "برنامج", "برامج",
    "دكتوراه", "الماجستير", "ماجستير", "بكالوريوس", "مكتب", "حرم", "الحرم",
    "مركز", "أبحاث", "ابحاث", "بحثي", "البحثي", "سكن", "مطار", "المطار",
    "رسوم", "الرسوم", "الدراسية", "العام", "الأكاديمي", "الاكاديمي",
}

_EVIDENCE_ADJUDICATION_JSON_SCHEMA = {
    "name": "evidence_adjudication",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "abstain": {"type": "boolean"},
            "selected_answer_ids": {"type": "array", "items": {"type": "string"}},
            "selected_fact_ids": {"type": "array", "items": {"type": "string"}},
            "selected_chunk_ids": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"},
            "confidence": {"type": "number"},
        },
        "required": [
            "abstain",
            "selected_answer_ids",
            "selected_fact_ids",
            "selected_chunk_ids",
            "reason",
            "confidence",
        ],
    },
}

_EVIDENCE_ADJUDICATION_SYSTEM_PROMPT = """
You adjudicate a bounded set of retrieved evidence candidates for a factual question.

Return JSON only with this schema:
{
  "abstain": true|false,
  "selected_answer_ids": ["id"],
  "selected_fact_ids": ["id"],
  "selected_chunk_ids": ["id"],
  "reason": "string",
  "confidence": 0.0
}

Rules:
- Select only candidates that directly support the user query.
- First verify every presupposed entity, program, degree, campus, office, center, location, year, and requested attribute. A related MBZUAI page is not proof that the presupposed thing exists.
- A generic MBZUAI phone, address, fee, program, campus, or service must not answer a question scoped to a different location, discipline, office, year, or subtype.
- Require the premise and requested value to be supported by the same candidate or by an explicit, unambiguous evidence chain. Do not combine unrelated fragments into an inferred answer.
- Never infer future values, speakers, winners, schedules, fees, or outcomes from current or historical material.
- Pay close attention to qualifiers, scope, subtype, department, role, and currentness.
- Prefer current authoritative evidence over historical, founding, event, or incidental mentions.
- If the query asks for a specific subtype such as support hours, department contact, or current leadership role, do not choose a broader generic candidate unless it is the only direct supported answer.
- If the provided candidates do not directly support the answer, abstain.
- Only return ids that appear in the candidate lists.
- Keep selections small and precise.
"""


def _tokenize(text: str) -> List[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = _ARABIC_DIACRITICS_RE.sub("", normalized)
    normalized = normalized.replace("ـ", "")
    return [token for token in _TOKEN_RE.findall(normalized) if token]


def _clean_requirement(
    value: str,
    *,
    preserve: Iterable[str] = (),
) -> str:
    preserved = set(_tokenize(" ".join(str(item or "") for item in preserve)))
    tokens = [
        token
        for token in _tokenize(value)
        if len(token) > 1
        and (token not in _PREMISE_STOPWORDS or token in preserved)
        and not token.isdigit()
    ]
    return " ".join(dict.fromkeys(tokens))


def extract_premise_requirements(
    query: str,
    intent_summary: Mapping[str, Any] | None = None,
) -> List[str]:
    """Extract closed-world premises that retrieved evidence must support.

    The rules describe shapes of claims (named entity, scoped location,
    academic offering, compound institutional asset, or guarantee) rather than
    enumerating benchmark entities.  They therefore apply to new questions and
    both supported and unsupported premises.
    """

    text = " ".join(str(query or "").split())
    if not text:
        return []
    requirements: List[str] = []
    # Academic-offering premises: "PhD in marine biology", "veterinary
    # medicine degree", and their Arabic equivalents.
    for match in re.finditer(
        r"\b(ph\.?d\.?|doctorate|master(?:'s)?|bachelor(?:'s)?)\s+in\s+([a-z][a-z\- ]{1,60}?)(?=\s+(?:at|from|within|offered|require)|[?.,]|$)",
        text.casefold(),
        flags=re.IGNORECASE,
    ):
        kind = "phd" if match.group(1).startswith("ph") else match.group(1)
        cleaned = _clean_requirement(
            f"{match.group(2)} {kind}",
            preserve={"phd", "doctorate", "master", "bachelor"},
        )
        if cleaned:
            requirements.append(cleaned)
    for match in re.finditer(
        r"\bmbzuai(?:'s|’s)?\s+([a-z][a-z\- ]{1,45}?)\s+(degree|program(?:me)?)\b",
        text.casefold(),
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_requirement(
            f"{match.group(1)} {match.group(2)}",
            preserve={"degree", "program", "programme"},
        )
        if cleaned:
            requirements.append(cleaned)
    for match in re.finditer(
        r"(?:^|\s)ل?(دكتوراه|ال?ماجستير|ال?بكالوريوس)\s+(?!في\s+جامعة|بجامعة|بالجامعة)(.+?)(?=\s+(?:في\s+جامعة|بجامعة|بالجامعة)|[؟?،,]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_requirement(
            f"{match.group(2)} {match.group(1)}",
            preserve={"دكتوراه", "ماجستير", "الماجستير", "بكالوريوس", "البكالوريوس"},
        )
        if cleaned:
            requirements.append(cleaned)
    for match in re.finditer(
        r"(برنامج)\s+(.+?)(?=\s+(?:في\s+جامعة|بجامعة|بالجامعة)|[؟?،,]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_requirement(
            f"{match.group(2)} {match.group(1)}",
            preserve={"برنامج"},
        )
        if cleaned and not set(_tokenize(cleaned)) <= {
            "برنامج", "دكتوراه", "ماجستير", "الماجستير", "بكالوريوس", "البكالوريوس"
        }:
            requirements.append(cleaned)

    # Location-scoped institutional units.  Require the location and unit to
    # co-occur in evidence instead of allowing a generic contact/campus result.
    location_matches: List[tuple[str, str, bool]] = []
    for match in re.finditer(
        r"\bmbzuai(?:'s|’s)?\s+([a-z][a-z\-]{1,20}(?:\s+[a-z][a-z\-]{1,20}){0,2})\s+(office|campus|research\s+center|research\s+centre)\b",
        text,
        flags=re.IGNORECASE,
    ):
        location_matches.append((match.group(1), match.group(2), False))
    for match in re.finditer(
        r"\b(office|campus|center|centre|housing)\s+(?:in|at|on)\s+([a-z][a-z\- ]{1,35}?)(?=[?.,]|$)",
        text,
        flags=re.IGNORECASE,
    ):
        location_matches.append((match.group(2), match.group(1), False))
    for match in re.finditer(
        r"(مكتب|حرم|الحرم|مركز|سكن|المطار|مطار).*?\s(?:في|على)\s+([^؟?،,]{2,55})",
        text,
        flags=re.IGNORECASE,
    ):
        location_matches.append((match.group(2), match.group(1), True))
    requested_types = {
        str(value or "").strip()
        for value in (
            ((intent_summary or {}).get("answer_types") or [])
            if isinstance(intent_summary, Mapping)
            else []
        )
        if str(value or "").strip()
    }
    answer_type_labels = {
        "phone": ("phone", "هاتف"),
        "hours": ("hours", "ساعات"),
        "location": ("location", "موقع"),
    }
    for location, unit, arabic in location_matches:
        suffixes = [
            labels[1 if arabic else 0]
            for answer_type, labels in answer_type_labels.items()
            if answer_type in requested_types
        ]
        preserve = {
            "office", "campus", "center", "centre", "research", "housing",
            "مكتب", "حرم", "الحرم", "مركز", "سكن", "هاتف", "ساعات", "موقع",
        }
        cleaned = _clean_requirement(
            " ".join([location, unit, *suffixes]),
            preserve=preserve,
        )
        if cleaned:
            requirements.append(cleaned)

    # Compound assets and unsupported-method premises should be verified as a
    # phrase, not by the generic noun alone.
    for match in re.finditer(
        r"\b([a-z][a-z\-]{2,25})\s+(wallet|airport)\b",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_requirement(
            f"{match.group(1)} {match.group(2)}",
            preserve={"wallet", "airport"},
        )
        if cleaned:
            requirements.append(cleaned)
    for match in re.finditer(
        r"محفظة\s+([^\s؟?،,]+(?:\s+[^\s؟?،,]+)?)",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_requirement(
            f"محفظة {match.group(1)}", preserve={"محفظة"}
        )
        if cleaned:
            requirements.append(cleaned)
    for match in re.finditer(
        r"(المطار|مطار)\s+([^؟?،,\s]{2,20})",
        text,
        flags=re.IGNORECASE,
    ):
        cleaned = _clean_requirement(
            f"{match.group(1)} {match.group(2)}", preserve={"المطار", "مطار"}
        )
        if cleaned:
            requirements.append(cleaned)

    guarantee_markers: List[str] = []
    if (
        re.search(r"\bguarante(?:e|es|ed|eing)\b", text.casefold())
        and re.search(r"\b(?:gpa|cgpa|admission)\b", text.casefold())
    ):
        guarantee_markers.append("gpa guarantee admission")
    if (
        any(marker in text.casefold() for marker in ("يضمن", "ضمان", "مضمون"))
        and any(marker in text for marker in ("المعدل", "التراكمي"))
        and "القبول" in text
    ):
        guarantee_markers.append("المعدل يضمن القبول")
    requirements.extend(
        _clean_requirement(
            value,
            preserve={"admission", "المعدل", "القبول"},
        )
        for value in guarantee_markers
    )
    return list(dict.fromkeys(value for value in requirements if value))


def query_requires_premise_grounding(
    query: str,
    intent_summary: Mapping[str, Any] | None = None,
) -> bool:
    return bool(extract_premise_requirements(query, intent_summary))


def _truncate_words(text: str, limit: int) -> str:
    words = str(text or "").split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit]).strip()


def _candidate_text(candidate: Mapping[str, Any]) -> str:
    return " ".join(
        str(part or "")
        for part in (
            candidate.get("subject_text"),
            candidate.get("value"),
            candidate.get("text"),
            " ".join(candidate.get("qualifiers") or []),
            candidate.get("document_title"),
            candidate.get("source_url"),
        )
        if part
    ).strip()


def _normalize_ids(values: Iterable[Any], allowed: set[str], *, limit: int) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values or []:
        item = str(value or "").strip()
        if not item or item not in allowed or item in seen:
            continue
        seen.add(item)
        output.append(item)
        if len(output) >= max(1, int(limit or 1)):
            break
    return output


def _support_hours_query(query: str) -> bool:
    query_tokens = set(_tokenize(query))
    return bool(query_tokens & {"support", "technical", "it", "helpdesk", "screening", "exam"})


def _subject_supported(intent_summary: Mapping[str, Any], texts: Sequence[str]) -> bool:
    haystack = " ".join(str(value or "").lower() for value in texts)
    subject_phrases = [str(value or "").strip().lower() for value in (intent_summary.get("subject_phrases") or []) if str(value or "").strip()]
    if subject_phrases and any(phrase in haystack for phrase in subject_phrases):
        return True
    subject_tokens = {
        str(value or "").strip().lower()
        for value in (intent_summary.get("subject_tokens") or [])
        if str(value or "").strip()
    }
    subject_tokens.discard("mbzuai")
    if not subject_tokens:
        return True
    haystack_tokens = set(_tokenize(haystack))
    return bool(subject_tokens & haystack_tokens)


def _token_supported(token: str, candidate_tokens: set[str]) -> bool:
    if token in candidate_tokens:
        return True
    if not token.isascii() or len(token) < 5:
        return False
    variants = {token}
    if token.endswith("ies") and len(token) > 5:
        variants.add(f"{token[:-3]}y")
    for suffix in ("s", "ed", "ing"):
        if token.endswith(suffix) and len(token) > len(suffix) + 3:
            variants.add(token[: -len(suffix)])
    for candidate in candidate_tokens:
        if candidate in variants:
            return True
        if candidate.endswith("s") and candidate[:-1] in variants:
            return True
    return False


def _premises_supported(requirements: Sequence[str], texts: Sequence[str]) -> bool:
    if not requirements:
        return True
    candidate_token_sets = [set(_tokenize(text)) for text in texts if str(text or "").strip()]
    if not candidate_token_sets:
        return False
    for requirement in requirements:
        required_tokens = list(dict.fromkeys(_tokenize(requirement)))
        if not required_tokens:
            continue
        if not any(
            all(_token_supported(token, candidate_tokens) for token in required_tokens)
            for candidate_tokens in candidate_token_sets
        ):
            return False
    return True


def _premise_explicitly_refuted(query: str, texts: Sequence[str]) -> bool:
    query_text = " ".join(str(query or "").split()).casefold()
    guarantee_query = bool(
        re.search(r"\bguarante(?:e|es|ed|eing)\b", query_text)
        or any(marker in query_text for marker in ("يضمن", "ضمان", "مضمون"))
    )
    if not guarantee_query:
        return False
    for value in texts:
        candidate = " ".join(str(value or "").split()).casefold()
        if re.search(
            r"\b(?:does|do|will|can|is|are)?\s*not\s+guarante(?:e|es|ed)\b|"
            r"\bno\b.{0,40}\bguarantee\b|"
            r"لا\s+يضمن|لا.{0,40}ضمان|ليس.{0,40}مضمون",
            candidate,
            flags=re.IGNORECASE,
        ):
            return True
    return False


def _fallback_heuristic_adjudication(
    *,
    query: str,
    intent_summary: Mapping[str, Any],
    answer_documents: Sequence[Mapping[str, Any]],
    fact_documents: Sequence[Mapping[str, Any]],
    retrieval_documents: Sequence[Mapping[str, Any]],
    max_answer_ids: int,
    max_fact_ids: int,
    max_chunk_ids: int,
) -> Dict[str, Any]:
    requested_types = {
        str(value or "").strip()
        for value in (intent_summary.get("answer_types") or [])
        if str(value or "").strip()
    }
    requested_roles = {
        str(value or "").strip()
        for value in (intent_summary.get("requested_roles") or [])
        if str(value or "").strip()
    }
    strict_answer_required = bool(intent_summary.get("strict_answer_required", False))
    query_tokens = set(_tokenize(query))
    support_hours_query = _support_hours_query(query)
    premise_requirements = extract_premise_requirements(query, intent_summary)

    candidate_answers: List[Mapping[str, Any]] = [
        candidate
        for candidate in (answer_documents or [])
        if isinstance(candidate, Mapping) and str(candidate.get("id") or "").strip()
    ]
    if requested_types:
        typed_answers = [
            candidate
            for candidate in candidate_answers
            if str(candidate.get("answer_type") or "").strip() in requested_types
        ]
        if typed_answers:
            candidate_answers = typed_answers
    if requested_roles:
        role_answers = [
            candidate
            for candidate in candidate_answers
            if str(candidate.get("answer_subtype") or "").strip() in requested_roles
        ]
        if role_answers:
            candidate_answers = role_answers
    if support_hours_query:
        preferred = [
            candidate
            for candidate in candidate_answers
            if str(candidate.get("answer_subtype") or "").strip() == "support_hours"
        ]
        if preferred:
            candidate_answers = preferred
    if {"board", "trustee", "trustees"} & query_tokens and {"chair", "chairs", "chairman"} & query_tokens:
        board_answers = [
            candidate
            for candidate in candidate_answers
            if str(candidate.get("answer_subtype") or "").strip() == "board_chair"
        ]
        if board_answers:
            non_founding = [
                candidate
                for candidate in board_answers
                if "founding" not in _candidate_text(candidate).lower()
            ]
            candidate_answers = non_founding or board_answers

    selected_answer_ids = _normalize_ids(
        [candidate.get("id") for candidate in candidate_answers],
        {str(candidate.get("id") or "") for candidate in candidate_answers},
        limit=max_answer_ids,
    )
    selected_fact_ids = _normalize_ids(
        [candidate.get("id") for candidate in (fact_documents or []) if isinstance(candidate, Mapping)],
        {str(candidate.get("id") or "") for candidate in (fact_documents or []) if isinstance(candidate, Mapping)},
        limit=max_fact_ids,
    )
    selected_chunk_ids = _normalize_ids(
        [
            candidate.get("id")
            for candidate in (retrieval_documents or [])
            if isinstance(candidate, Mapping)
            and not str(candidate.get("answer_type") or "").strip()
        ],
        {
            str(candidate.get("id") or "")
            for candidate in (retrieval_documents or [])
            if isinstance(candidate, Mapping)
            and not str(candidate.get("answer_type") or "").strip()
        },
        limit=max_chunk_ids,
    )

    candidate_texts = [
        *[_candidate_text(candidate) for candidate in candidate_answers],
        *[_candidate_text(candidate) for candidate in (fact_documents or []) if isinstance(candidate, Mapping)],
        *[_candidate_text(candidate) for candidate in (retrieval_documents or []) if isinstance(candidate, Mapping)],
    ]
    subject_supported = _subject_supported(intent_summary, candidate_texts)
    premise_supported = _premises_supported(premise_requirements, candidate_texts)

    if premise_requirements and _premise_explicitly_refuted(query, candidate_texts):
        return {
            "used": False,
            "method": "heuristic",
            "abstain": True,
            "selected_answer_ids": [],
            "selected_fact_ids": [],
            "selected_chunk_ids": [],
            "reason": "presupposed_claim_explicitly_refuted",
            "confidence": 0.94,
        }
    if premise_requirements and not premise_supported:
        return {
            "used": False,
            "method": "heuristic",
            "abstain": True,
            "selected_answer_ids": [],
            "selected_fact_ids": [],
            "selected_chunk_ids": [],
            "reason": "presupposed_entity_or_scope_not_supported",
            "confidence": 0.90,
        }

    if strict_answer_required and requested_types and not selected_answer_ids and not selected_fact_ids:
        return {
            "used": False,
            "method": "heuristic",
            "abstain": True,
            "selected_answer_ids": [],
            "selected_fact_ids": [],
            "selected_chunk_ids": [],
            "reason": "no_direct_supported_answer_candidate",
            "confidence": 0.82,
        }
    if strict_answer_required and not subject_supported:
        return {
            "used": False,
            "method": "heuristic",
            "abstain": True,
            "selected_answer_ids": [],
            "selected_fact_ids": [],
            "selected_chunk_ids": [],
            "reason": "subject_not_supported_by_candidates",
            "confidence": 0.88,
        }
    return {
        "used": False,
        "method": "heuristic",
        "abstain": False,
        "selected_answer_ids": selected_answer_ids,
        "selected_fact_ids": selected_fact_ids,
        "selected_chunk_ids": selected_chunk_ids,
        "reason": "heuristic_selection",
        "confidence": 0.66 if selected_answer_ids or selected_fact_ids else 0.52,
    }


def heuristic_adjudicate_factual_evidence(
    *,
    query: str,
    intent_summary: Mapping[str, Any],
    answer_documents: Sequence[Mapping[str, Any]],
    fact_documents: Sequence[Mapping[str, Any]],
    retrieval_documents: Sequence[Mapping[str, Any]],
    max_answer_ids: int = 4,
    max_fact_ids: int = 4,
    max_chunk_ids: int = 6,
) -> Dict[str, Any]:
    """Deterministic fail-closed fallback for bounded provider failures."""

    return _fallback_heuristic_adjudication(
        query=query,
        intent_summary=intent_summary,
        answer_documents=answer_documents,
        fact_documents=fact_documents,
        retrieval_documents=retrieval_documents,
        max_answer_ids=max_answer_ids,
        max_fact_ids=max_fact_ids,
        max_chunk_ids=max_chunk_ids,
    )


def _adjudication_prompt(
    *,
    query: str,
    intent_summary: Mapping[str, Any],
    answer_documents: Sequence[Mapping[str, Any]],
    fact_documents: Sequence[Mapping[str, Any]],
    retrieval_documents: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        f"QUERY: {query}",
        "INTENT SUMMARY:",
        f"- answer_types: {', '.join(str(value) for value in (intent_summary.get('answer_types') or [])) or 'none'}",
        f"- requested_roles: {', '.join(str(value) for value in (intent_summary.get('requested_roles') or [])) or 'none'}",
        f"- strict_answer_required: {bool(intent_summary.get('strict_answer_required', False))}",
        f"- subject_phrases: {', '.join(str(value) for value in (intent_summary.get('subject_phrases') or [])) or 'none'}",
        f"- premise_requirements: {', '.join(extract_premise_requirements(query, intent_summary)) or 'none'}",
        "",
        "ANSWER CANDIDATES:",
    ]
    for candidate in (answer_documents or [])[:6]:
        if not isinstance(candidate, Mapping):
            continue
        lines.extend(
            [
                f"- id: {candidate.get('id')}",
                f"  answer_type: {candidate.get('answer_type')}",
                f"  answer_subtype: {candidate.get('answer_subtype')}",
                f"  subject: {_truncate_words(str(candidate.get('subject_text') or ''), 24)}",
                f"  value: {_truncate_words(str(candidate.get('value') or ''), 32)}",
                f"  qualifiers: {', '.join(str(value) for value in (candidate.get('qualifiers') or [])) or 'none'}",
                f"  confidence: {float(candidate.get('confidence') or 0.0):.3f}",
                f"  authority_score: {float(candidate.get('authority_score') or 0.0):.3f}",
                f"  freshness_score: {float(candidate.get('freshness_score') or 0.0):.3f}",
                f"  source: {_truncate_words(str(candidate.get('source_url') or candidate.get('document_title') or ''), 20)}",
                f"  text: {_truncate_words(str(candidate.get('text') or ''), 80)}",
            ]
        )
    lines.append("")
    lines.append("FACT CANDIDATES:")
    for candidate in (fact_documents or [])[:6]:
        if not isinstance(candidate, Mapping):
            continue
        lines.extend(
            [
                f"- id: {candidate.get('id')}",
                f"  source: {_truncate_words(str(candidate.get('source_url') or candidate.get('document_title') or ''), 20)}",
                f"  text: {_truncate_words(str(candidate.get('text') or ''), 80)}",
            ]
        )
    lines.append("")
    lines.append("SUPPORTING CHUNK CANDIDATES:")
    supporting_docs = [
        candidate
        for candidate in (retrieval_documents or [])
        if isinstance(candidate, Mapping) and not str(candidate.get("answer_type") or "").strip()
    ]
    for candidate in supporting_docs[:8]:
        lines.extend(
            [
                f"- id: {candidate.get('id')}",
                f"  source: {_truncate_words(str(candidate.get('source_url') or candidate.get('document_title') or ''), 20)}",
                f"  text: {_truncate_words(str(candidate.get('text') or ''), 120)}",
            ]
        )
    return "\n".join(lines).strip()


def adjudicate_factual_evidence(
    *,
    query: str,
    intent_summary: Mapping[str, Any],
    answer_documents: Sequence[Mapping[str, Any]],
    fact_documents: Sequence[Mapping[str, Any]],
    retrieval_documents: Sequence[Mapping[str, Any]],
    model: str,
    reasoning_effort: str = "minimal",
    temperature: float = 0.0,
    max_completion_tokens: int = 800,
    min_confidence: float = 0.58,
    retries: int = 2,
    retry_delay_sec: float = 1.0,
    per_request_delay_sec: float = 0.0,
    provider_timeout_sec: float | None = None,
    max_answer_ids: int = 4,
    max_fact_ids: int = 4,
    max_chunk_ids: int = 6,
) -> Dict[str, Any]:
    fallback = heuristic_adjudicate_factual_evidence(
        query=query,
        intent_summary=intent_summary,
        answer_documents=answer_documents,
        fact_documents=fact_documents,
        retrieval_documents=retrieval_documents,
        max_answer_ids=max_answer_ids,
        max_fact_ids=max_fact_ids,
        max_chunk_ids=max_chunk_ids,
    )

    try:
        client = make_openai_client(timeout_sec=provider_timeout_sec)
    except RuntimeError:
        return fallback

    try:
        payload = json_completion(
            client=client,
            model=model,
            system_prompt=_EVIDENCE_ADJUDICATION_SYSTEM_PROMPT,
            user_prompt=_adjudication_prompt(
                query=query,
                intent_summary=intent_summary,
                answer_documents=answer_documents,
                fact_documents=fact_documents,
                retrieval_documents=retrieval_documents,
            ),
            json_schema=_EVIDENCE_ADJUDICATION_JSON_SCHEMA,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            retries=retries,
            retry_delay_sec=retry_delay_sec,
            per_request_delay_sec=per_request_delay_sec,
        )
    except Exception:
        return fallback

    allowed_answer_ids = {
        str(candidate.get("id") or "")
        for candidate in (answer_documents or [])
        if isinstance(candidate, Mapping) and str(candidate.get("id") or "").strip()
    }
    allowed_fact_ids = {
        str(candidate.get("id") or "")
        for candidate in (fact_documents or [])
        if isinstance(candidate, Mapping) and str(candidate.get("id") or "").strip()
    }
    allowed_chunk_ids = {
        str(candidate.get("id") or "")
        for candidate in (retrieval_documents or [])
        if isinstance(candidate, Mapping)
        and str(candidate.get("id") or "").strip()
        and not str(candidate.get("answer_type") or "").strip()
    }
    confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    selected_answer_ids = _normalize_ids(payload.get("selected_answer_ids") or [], allowed_answer_ids, limit=max_answer_ids)
    selected_fact_ids = _normalize_ids(payload.get("selected_fact_ids") or [], allowed_fact_ids, limit=max_fact_ids)
    selected_chunk_ids = _normalize_ids(payload.get("selected_chunk_ids") or [], allowed_chunk_ids, limit=max_chunk_ids)
    abstain = bool(payload.get("abstain"))
    if confidence < float(min_confidence or 0.0):
        return fallback
    return {
        "used": True,
        "method": "openai",
        "abstain": abstain,
        "selected_answer_ids": selected_answer_ids,
        "selected_fact_ids": selected_fact_ids,
        "selected_chunk_ids": selected_chunk_ids,
        "reason": str(payload.get("reason") or "").strip(),
        "confidence": confidence,
    }
