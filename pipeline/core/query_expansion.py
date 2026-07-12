from __future__ import annotations

from typing import Any, Dict

from pipeline.core.openai_client import json_completion, make_openai_client


_HYDE_SYSTEM_PROMPT = """
You write one short hypothetical retrieval passage for a broad university-website question.

Return JSON only:
{
  "hypothetical_document": "string",
  "confidence": 0.0
}

Rules:
- Do not invent names, dates, emails, phone numbers, fees, or deadlines.
- Write a generic passage that describes what a relevant MBZUAI page would contain.
- Keep it under 80 words.
- If the query asks for an exact fact, return an empty hypothetical_document and confidence 0.
"""


def hyde_expansion(
    *,
    query: str,
    model: str,
    reasoning_effort: str = "minimal",
    temperature: float = 0.0,
    max_completion_tokens: int = 300,
    retries: int = 1,
    retry_delay_sec: float = 1.0,
    per_request_delay_sec: float = 0.0,
) -> Dict[str, Any]:
    try:
        client = make_openai_client()
    except RuntimeError:
        return {"hypothetical_document": "", "confidence": 0.0}

    payload = json_completion(
        client=client,
        model=model,
        system_prompt=_HYDE_SYSTEM_PROMPT,
        user_prompt=query,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
        retries=retries,
        retry_delay_sec=retry_delay_sec,
        per_request_delay_sec=per_request_delay_sec,
    )
    document = str(payload.get("hypothetical_document") or "").strip()
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "hypothetical_document": document,
        "confidence": confidence,
    }
