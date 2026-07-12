from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from pipeline.core.openai_client import json_completion, make_openai_client

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

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
- Pay close attention to qualifiers, scope, subtype, department, role, and currentness.
- Prefer current authoritative evidence over historical, founding, event, or incidental mentions.
- If the query asks for a specific subtype such as support hours, department contact, or current leadership role, do not choose a broader generic candidate unless it is the only direct supported answer.
- If the provided candidates do not directly support the answer, abstain.
- Only return ids that appear in the candidate lists.
- Keep selections small and precise.
"""


def _tokenize(text: str) -> List[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text or "").lower())]


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

    subject_supported = _subject_supported(
        intent_summary,
        [
            *[_candidate_text(candidate) for candidate in candidate_answers],
            *[str(candidate.get("text") or "") for candidate in (fact_documents or []) if isinstance(candidate, Mapping)],
            *[str(candidate.get("text") or "") for candidate in (retrieval_documents or []) if isinstance(candidate, Mapping)],
        ],
    )

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
    fallback = _fallback_heuristic_adjudication(
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
