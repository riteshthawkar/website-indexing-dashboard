from __future__ import annotations

import re
from typing import Any, Dict, Sequence

from pipeline.core.navigation_intent import infer_navigation_context
from pipeline.core.openai_client import json_completion, make_openai_client


_QUERY_PLANNER_SYSTEM_PROMPT = """
You are a retrieval query planner.

Return JSON only:
{
  "query_type": "fact|scoped|synthesis|multimodal|no_answer_risk",
  "vector_query": "string",
  "graph_query": "string",
  "answer_types": ["string"],
  "entity_hints": ["string"],
  "navigation_intent": "none|open_page|follow_steps|apply|register|contact|download|login|search",
  "navigation_goal": "string",
  "navigation_confidence": 0.0,
  "confidence": 0.0
}

Rules:
- Classify a request for a list, comparison, overview, or several related
  fields as synthesis. Use fact only for one narrow value and scoped for a
  clearly bounded page, program, person, event, or process.
- Keep vector_query semantically broad enough for document retrieval.
- Keep graph_query relation-oriented and entity-explicit when useful.
- Do not expand into many rewrites; produce one vector query and one graph query.
- answer_types should be short labels like email, phone, website, hours, date, location, named_after, affiliation, legal_basis, role_holder.
- Use an empty answer_types array for an ordinary narrative or list answer.
- Preserve every named entity, degree level, language-specific term, number,
  date, and constraint from the user. Add retrieval synonyms, never facts.
- Preserve the user's original-language concepts. When the likely source corpus
  may use another language, add only concise translation-equivalent retrieval
  terms (for example Arabic plus English); do not replace or reinterpret the
  original request.
- Set navigation_intent only when the user explicitly wants to reach a page, follow a process, or perform an action. Use none for ordinary factual questions.
- navigation_goal describes the requested outcome, never a URL. Do not invent URLs, page IDs, action IDs, or click steps; those are resolved from the grounded page graph after retrieval.
- confidence measures the safety of the retrieval rewrite, not confidence in an
  answer. Use 0.85-1.0 for a clear query whose entities and constraints are all
  preserved, 0.60-0.84 for a useful but somewhat ambiguous rewrite, and below
  0.55 only when the user's retrieval intent itself is unclear.
"""


_QUERY_TYPES = {"fact", "scoped", "synthesis", "multimodal", "no_answer_risk"}
_ANSWER_TYPES = {
    "email",
    "phone",
    "website",
    "hours",
    "date",
    "location",
    "named_after",
    "affiliation",
    "legal_basis",
    "role_holder",
    "service_availability",
}

_SPECULATIVE_REWRITE_RE = re.compile(
    r"(?:\.{3,}|…)|"
    r"\b(?:tbd|unknown|unclear|unsure|maybe|perhaps|possibly|presumably)\b|"
    r"\b(?:need|needs|requiring)\s+(?:current|up[- ]to[- ]date|verification|"
    r"verify|confirmation)\b|"
    r"\b(?:to\s+be\s+confirmed|not\s+sure)\b",
    flags=re.IGNORECASE,
)
_QUERY_PLANNER_JSON_SCHEMA: Dict[str, Any] = {
    "name": "retrieval_query_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "query_type",
            "vector_query",
            "graph_query",
            "answer_types",
            "entity_hints",
            "navigation_intent",
            "navigation_goal",
            "navigation_confidence",
            "confidence",
        ],
        "properties": {
            "query_type": {"type": "string", "enum": sorted(_QUERY_TYPES)},
            "vector_query": {"type": "string", "minLength": 1, "maxLength": 1200},
            "graph_query": {"type": "string", "minLength": 1, "maxLength": 1200},
            "answer_types": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(_ANSWER_TYPES)},
                "maxItems": 6,
            },
            "entity_hints": {
                "type": "array",
                "items": {"type": "string", "minLength": 1, "maxLength": 160},
                "maxItems": 12,
            },
            "navigation_intent": {
                "type": "string",
                "enum": [
                    "none",
                    "open_page",
                    "follow_steps",
                    "apply",
                    "register",
                    "contact",
                    "download",
                    "login",
                    "search",
                ],
            },
            "navigation_goal": {"type": "string", "maxLength": 500},
            "navigation_confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    },
}


def _bounded_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value or 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _rewrite_preserves_query_constraints(query: str, rewrite: str) -> bool:
    """Check that a model rewrite remains anchored to the user's request."""

    query_text = str(query or "")
    rewrite_text = str(rewrite or "")
    # Retrieval rewrites are search expressions, not draft answers. Reject
    # model uncertainty, placeholders, and newly introduced questions because
    # they can smuggle guessed entities into otherwise well-anchored rewrites.
    query_speculation = {
        match.group(0).casefold()
        for match in _SPECULATIVE_REWRITE_RE.finditer(query_text)
    }
    if any(
        match.group(0).casefold() not in query_speculation
        for match in _SPECULATIVE_REWRITE_RE.finditer(rewrite_text)
    ):
        return False
    if rewrite_text.count("?") > query_text.count("?"):
        return False

    query_tokens = re.findall(r"[^\W_]+", query_text.casefold(), flags=re.UNICODE)
    rewrite_tokens = set(
        re.findall(r"[^\W_]+", rewrite_text.casefold(), flags=re.UNICODE)
    )
    if not query_tokens or not rewrite_tokens:
        return False
    distinct_query_tokens = set(query_tokens)
    overlap = distinct_query_tokens & rewrite_tokens
    required_overlap = 1 if len(distinct_query_tokens) <= 4 else 2
    if len(overlap) < required_overlap:
        return False
    query_numbers = {token for token in distinct_query_tokens if token.isdigit()}
    return query_numbers <= rewrite_tokens


def _calibrated_planner_confidence(
    *, query: str, vector_query: str, reported_confidence: Any
) -> float:
    reported = _bounded_confidence(reported_confidence)
    if reported > 0.0:
        return reported
    # Some models copy the 0.0 JSON placeholder even while returning a useful,
    # constraint-preserving plan. Structural validation supplies a conservative
    # score so a valid rewrite is not silently discarded wholesale.
    if vector_query.strip() != query.strip() and _rewrite_preserves_query_constraints(
        query, vector_query
    ):
        return 0.65
    return 0.0


def heuristic_plan(query: str, *, query_type: str = "fact") -> Dict[str, Any]:
    navigation = infer_navigation_context(query)
    return {
        "query_type": query_type,
        "vector_query": query,
        "graph_query": query,
        "answer_types": [],
        "entity_hints": [],
        "navigation_intent": navigation["intent"],
        "navigation_goal": navigation["goal"],
        "navigation_confidence": navigation["confidence"],
        "confidence": 0.0,
    }


def plan_query(
    *,
    query: str,
    model: str,
    reasoning_effort: str = "minimal",
    temperature: float = 0.0,
    max_completion_tokens: int = 600,
    retries: int = 2,
    retry_delay_sec: float = 1.0,
    per_request_delay_sec: float = 0.0,
    fallback_query_type: str = "fact",
) -> Dict[str, Any]:
    try:
        client = make_openai_client()
    except RuntimeError:
        return heuristic_plan(query, query_type=fallback_query_type)

    try:
        payload = json_completion(
            client=client,
            model=model,
            system_prompt=_QUERY_PLANNER_SYSTEM_PROMPT,
            user_prompt=query,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            json_schema=_QUERY_PLANNER_JSON_SCHEMA,
            retries=retries,
            retry_delay_sec=retry_delay_sec,
            per_request_delay_sec=per_request_delay_sec,
        )
    except RuntimeError:
        return heuristic_plan(query, query_type=fallback_query_type)
    navigation_fallback = infer_navigation_context(query)
    # Navigation is an action-bearing contract, so model output alone must not
    # turn a factual reference to a page or website into a navigation command.
    # The deterministic explicit-intent classifier is authoritative here.
    navigation_intent = str(navigation_fallback["intent"])
    navigation_confidence = float(navigation_fallback["confidence"])
    query_type = str(payload.get("query_type") or fallback_query_type).strip().lower()
    if query_type not in _QUERY_TYPES:
        query_type = fallback_query_type
    vector_query = str(payload.get("vector_query") or query).strip() or query
    vector_rewrite_grounded = bool(
        vector_query == query
        or _rewrite_preserves_query_constraints(query, vector_query)
    )
    if not vector_rewrite_grounded:
        vector_query = query
    graph_query = str(
        payload.get("graph_query") or vector_query or query
    ).strip() or vector_query
    if not _rewrite_preserves_query_constraints(query, graph_query):
        graph_query = vector_query
    return {
        "query_type": query_type,
        "vector_query": vector_query,
        "graph_query": graph_query,
        "answer_types": [
            str(value).strip().lower()
            for value in (payload.get("answer_types") or [])
            if str(value).strip().lower() in _ANSWER_TYPES
        ],
        "entity_hints": [
            str(value).strip()
            for value in (payload.get("entity_hints") or [])
            if str(value).strip()
        ],
        "navigation_intent": navigation_intent,
        "navigation_goal": str(navigation_fallback.get("goal") or "").strip()[:500],
        "navigation_confidence": navigation_confidence,
        "confidence": _calibrated_planner_confidence(
            query=query,
            vector_query=vector_query,
            reported_confidence=(
                payload.get("confidence") if vector_rewrite_grounded else 0.0
            ),
        ),
    }
