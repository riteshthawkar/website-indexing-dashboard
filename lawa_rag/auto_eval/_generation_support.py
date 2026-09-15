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
        grounding = float(payload.get("grounding_score"))
        naturalness = float(payload.get("naturalness_score"))
        difficulty = float(payload.get("difficulty_match_score"))
    except (TypeError, ValueError):
        return False
    return grounding >= 0.9 and naturalness >= 0.6 and difficulty >= 0.6


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
