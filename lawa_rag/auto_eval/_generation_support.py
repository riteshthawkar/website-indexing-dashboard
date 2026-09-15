"""Internal normalization helpers for generated model payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ._util import utc_now
from .cases import EvidenceCitation, GoldClaim
from .llm import StructuredLLM
from .prompts import GENERATION_PROMPT_VERSION


def claims_from(value: Any) -> tuple[GoldClaim, ...]:
    rows = value if isinstance(value, list) else []
    return tuple(GoldClaim.from_dict(item) for item in rows if isinstance(item, Mapping))


def citations_from(value: Any) -> tuple[EvidenceCitation, ...]:
    rows = value if isinstance(value, list) else []
    return tuple(EvidenceCitation.from_dict(item) for item in rows if isinstance(item, Mapping))


def llm_identity(llm: StructuredLLM) -> dict[str, str]:
    return {"provider": llm.identity.provider, "model": llm.identity.model}


def generation_metadata(question_llm, reference_llm, critic_llm) -> dict[str, Any]:
    return {
        "prompt_version": GENERATION_PROMPT_VERSION,
        "question_model": llm_identity(question_llm),
        "reference_model": llm_identity(reference_llm),
        "critic_model": llm_identity(critic_llm) if critic_llm else None,
    }


def critic_accepted(payload: Mapping[str, Any]) -> bool:
    if payload.get("passed") is not True or payload.get("answerability_correct") is not True:
        return False
    try:
        scores = {
            "grounding": float(payload.get("grounding_score")),
            "naturalness": float(payload.get("naturalness_score")),
            "difficulty": float(payload.get("difficulty_match_score")),
            "query_grounding": float(payload.get("query_grounding_score")),
            "answer_completeness": float(payload.get("answer_completeness_score")),
            "task_match": float(payload.get("task_match_score")),
            "persona_fit": float(payload.get("persona_fit_score")),
            "language_quality": float(payload.get("language_quality_score")),
        }
    except (TypeError, ValueError):
        return False
    thresholds = {
        "grounding": 0.9,
        "naturalness": 0.6,
        "difficulty": 0.6,
        "query_grounding": 0.9,
        "answer_completeness": 0.9,
        "task_match": 0.75,
        "persona_fit": 0.6,
        "language_quality": 0.8,
    }
    return all(scores[name] >= threshold for name, threshold in thresholds.items())


def failure_message(failures, output_dir) -> str:
    sample = "; ".join(f"{scenario}: {error}" for scenario, error in failures[:3])
    suffix = f" Check checkpoints in {Path(output_dir).resolve()}." if output_dir else ""
    return f"{len(failures)} scenarios failed generation: {sample}.{suffix}"


def save_failure(store, scenario, error) -> None:
    if not store:
        return
    store.save_failure(
        scenario.scenario_id,
        {
            "scenario": scenario.to_dict(),
            "error": str(error),
            "failed_at": utc_now(),
        },
    )
