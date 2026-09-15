"""Internal schema-specific calls to the structured LLM boundary."""

from __future__ import annotations

from ._util import clean_text
from .prompts import critic_prompts, question_prompts, reference_prompts


def generate_question(llm, snapshot, persona, scenario, bundle, feedback):
    system, user = question_prompts(
        snapshot, persona, scenario, bundle, feedback=feedback
    )
    payload = llm.generate_json(
        system_prompt=system,
        user_prompt=user,
        schema_name="auto_eval_question_v1",
    )
    query = clean_text(payload.get("query"))
    if not query:
        raise ValueError("Question generator returned an empty query")
    return {
        "query": query,
        "intent": clean_text(payload.get("intent")),
        "expected_response_structure": clean_text(
            payload.get("expected_response_structure")
        )
        or "concise grounded answer",
    }


def generate_reference(llm, snapshot, scenario, bundle, query):
    system, user = reference_prompts(snapshot, scenario, bundle, query)
    return llm.generate_json(
        system_prompt=system,
        user_prompt=user,
        schema_name="auto_eval_reference_v1",
    )


def run_critic(llm, snapshot, case, bundle):
    if llm is None:
        return None
    system, user = critic_prompts(snapshot, case, bundle)
    return llm.generate_json(
        system_prompt=system,
        user_prompt=user,
        schema_name="auto_eval_critic_v1",
    )
