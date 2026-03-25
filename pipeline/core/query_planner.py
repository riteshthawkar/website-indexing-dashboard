from __future__ import annotations

from typing import Any, Dict, Sequence

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
  "confidence": 0.0
}

Rules:
- Keep vector_query semantically broad enough for document retrieval.
- Keep graph_query relation-oriented and entity-explicit when useful.
- Do not expand into many rewrites; produce one vector query and one graph query.
- answer_types should be short labels like email, phone, website, hours, date, location, named_after, affiliation, legal_basis, role_holder.
"""


def heuristic_plan(query: str, *, query_type: str = "fact") -> Dict[str, Any]:
    return {
        "query_type": query_type,
        "vector_query": query,
        "graph_query": query,
        "answer_types": [],
        "entity_hints": [],
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
        "confidence": max(0.0, min(1.0, float(payload.get("confidence") or 0.0))),
    }
