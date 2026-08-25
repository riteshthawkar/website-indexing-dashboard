from __future__ import annotations

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
- Keep vector_query semantically broad enough for document retrieval.
- Keep graph_query relation-oriented and entity-explicit when useful.
- Do not expand into many rewrites; produce one vector query and one graph query.
- answer_types should be short labels like email, phone, website, hours, date, location, named_after, affiliation, legal_basis, role_holder.
- Set navigation_intent only when the user explicitly wants to reach a page, follow a process, or perform an action. Use none for ordinary factual questions.
- navigation_goal describes the requested outcome, never a URL. Do not invent URLs, page IDs, action IDs, or click steps; those are resolved from the grounded page graph after retrieval.
"""


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

    payload = json_completion(
        client=client,
        model=model,
        system_prompt=_QUERY_PLANNER_SYSTEM_PROMPT,
        user_prompt=query,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        retry_delay_sec=retry_delay_sec,
        per_request_delay_sec=per_request_delay_sec,
    )
    navigation_fallback = infer_navigation_context(query)
    navigation_intent = str(payload.get("navigation_intent") or "none").strip().lower()
    allowed_navigation_intents = {
        "none",
        "open_page",
        "follow_steps",
        "apply",
        "register",
        "contact",
        "download",
        "login",
        "search",
    }
    try:
        navigation_confidence = max(
            0.0,
            min(1.0, float(payload.get("navigation_confidence") or 0.0)),
        )
    except (TypeError, ValueError):
        navigation_confidence = 0.0
    if navigation_intent not in allowed_navigation_intents:
        navigation_intent = str(navigation_fallback["intent"])
        navigation_confidence = float(navigation_fallback["confidence"])
    if navigation_intent == "none" and navigation_fallback["intent"] != "none":
        navigation_intent = str(navigation_fallback["intent"])
        navigation_confidence = max(
            navigation_confidence, float(navigation_fallback["confidence"])
        )
    return {
        "query_type": str(payload.get("query_type") or fallback_query_type).strip().lower(),
        "vector_query": str(payload.get("vector_query") or query).strip() or query,
        "graph_query": str(payload.get("graph_query") or payload.get("vector_query") or query).strip() or query,
        "answer_types": [
            str(value).strip().lower()
            for value in (payload.get("answer_types") or [])
            if str(value).strip()
        ],
        "entity_hints": [
            str(value).strip()
            for value in (payload.get("entity_hints") or [])
            if str(value).strip()
        ],
        "navigation_intent": navigation_intent,
        "navigation_goal": str(
            payload.get("navigation_goal")
            or navigation_fallback.get("goal")
            or ""
        ).strip()[:500],
        "navigation_confidence": navigation_confidence,
        "confidence": max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
    }
