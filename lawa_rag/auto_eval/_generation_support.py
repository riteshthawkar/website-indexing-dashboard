"""Internal normalization helpers for generated model payloads."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Mapping

from ._util import clean_text, stable_id, string_tuple, utc_now
from .cases import EvidenceCitation, GeneratedEvalCase, GoldClaim
from .llm import StructuredLLM
from .planning import PLANNING_VERSION
from .prompts import GENERATION_PROMPT_VERSION
from .sampling import SAMPLING_VERSION
from .validation import VALIDATION_VERSION


CITATION_NORMALIZATION_VERSION = "cross-script-exact-quote.v2"
_CRITIC_SCORE_FIELDS = (
    "grounding_score",
    "naturalness_score",
    "difficulty_match_score",
    "query_grounding_score",
    "answer_completeness_score",
    "task_match_score",
    "persona_fit_score",
    "language_quality_score",
)


def claims_from(value: Any) -> tuple[GoldClaim, ...]:
    rows = value if isinstance(value, list) else []
    return tuple(GoldClaim.from_dict(item) for item in rows if isinstance(item, Mapping))


def citations_from(value: Any) -> tuple[EvidenceCitation, ...]:
    rows = value if isinstance(value, list) else []
    return tuple(EvidenceCitation.from_dict(item) for item in rows if isinstance(item, Mapping))


def normalize_citation_quotes(
    citations: tuple[EvidenceCitation, ...],
    evidence: Mapping[str, Any],
    *,
    max_quote_chars: int = 2600,
) -> tuple[tuple[EvidenceCitation, ...], tuple[str, ...]]:
    """Replace cross-script model quotes with an exact, auditable evidence excerpt.

    Models occasionally translate a citation while writing a reference answer in a
    different language.  The evidence id remains useful, but the translated text is
    no longer a quotation. Keep strict validation and deterministically substitute a
    prompt-visible excerpt instead of retrying that cross-language failure indefinitely.
    Same-script paraphrases and unknown ids remain untouched for validation to reject.
    """

    normalized: list[EvidenceCitation] = []
    repaired: list[str] = []
    for citation in citations:
        unit = evidence.get(citation.evidence_id)
        source_text = clean_text(getattr(unit, "text", ""))
        quote = clean_text(citation.quote)
        if (
            unit is not None
            and len(quote) >= 8
            and quote.casefold() in source_text.casefold()
        ):
            normalized.append(citation)
            continue
        if (
            unit is None
            or len(source_text) < 8
            or not _different_dominant_scripts(quote, source_text)
        ):
            normalized.append(citation)
            continue
        normalized.append(
            EvidenceCitation(citation.evidence_id, source_text[:max_quote_chars])
        )
        repaired.append(citation.evidence_id)
    return tuple(normalized), tuple(dict.fromkeys(repaired))


def _different_dominant_scripts(left: str, right: str) -> bool:
    left_script = _dominant_script(left)
    right_script = _dominant_script(right)
    return bool(left_script and right_script and left_script != right_script)


def _dominant_script(value: str) -> str:
    counts: dict[str, int] = {}
    scripts = ("LATIN", "ARABIC", "CYRILLIC", "HEBREW", "GREEK", "CJK", "HANGUL")
    for character in value:
        name = unicodedata.name(character, "")
        script = next((item for item in scripts if item in name), "")
        if script:
            counts[script] = counts.get(script, 0) + 1
    return max(counts, key=counts.get) if counts else ""


def build_generated_case(
    snapshot, persona, scenario, bundle, question, reference, generation
) -> GeneratedEvalCase:
    query = question["query"]
    citations, repaired = normalize_citation_quotes(
        citations_from(reference.get("citations")), snapshot.evidence_map()
    )
    return GeneratedEvalCase(
        case_id=stable_id("eval-case", snapshot.snapshot_id, scenario.scenario_id, query),
        scenario_id=scenario.scenario_id,
        persona_id=persona.persona_id,
        query=query,
        language=scenario.language,
        task=scenario.task,
        answerable=scenario.answerable,
        target_difficulty=scenario.difficulty.target_level,
        measured_difficulty=bundle.difficulty,
        reference_answer=clean_text(reference.get("reference_answer")),
        claims=claims_from(reference.get("claims")),
        citations=citations,
        evidence_ids=bundle.evidence_ids,
        distractor_ids=bundle.distractor_ids,
        source_ids=bundle.source_ids,
        expected_response_structure=question["expected_response_structure"],
        answer_must_include=string_tuple(reference.get("answer_must_include")),
        answer_must_not_include=string_tuple(reference.get("answer_must_not_include")),
        answer_should_cover=string_tuple(reference.get("answer_should_cover")),
        citation_requirements=string_tuple(reference.get("citation_requirements")),
        generation=generation,
        metadata={
            "intent": question["intent"],
            "citation_quote_repairs": len(repaired),
            "citation_quote_repair_evidence_ids": list(repaired),
        },
    )


def llm_identity(llm: StructuredLLM) -> dict[str, str]:
    return {"provider": llm.identity.provider, "model": llm.identity.model}


def generation_metadata(question_llm, reference_llm, critic_llm) -> dict[str, Any]:
    return {
        "prompt_version": GENERATION_PROMPT_VERSION,
        "planning_version": PLANNING_VERSION,
        "sampling_version": SAMPLING_VERSION,
        "validation_version": VALIDATION_VERSION,
        "citation_normalization_version": CITATION_NORMALIZATION_VERSION,
        "question_model": llm_identity(question_llm),
        "reference_model": llm_identity(reference_llm),
        "critic_model": llm_identity(critic_llm) if critic_llm else None,
    }


def critic_accepted(payload: Mapping[str, Any]) -> bool:
    if payload.get("passed") is not True or payload.get("answerability_correct") is not True:
        return False
    try:
        names = (
            "grounding",
            "naturalness",
            "difficulty",
            "query_grounding",
            "answer_completeness",
            "task_match",
            "persona_fit",
            "language_quality",
        )
        scores = dict(
            zip(names, (float(payload.get(field)) for field in _CRITIC_SCORE_FIELDS))
        )
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
    return all(
        0.0 <= scores[name] <= 1.0 and scores[name] >= threshold
        for name, threshold in thresholds.items()
    )


def normalize_critic_scores(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a provider's accidental 1–5 rubric output to the 0–1 contract."""
    output = dict(payload)
    try:
        values = [float(output.get(field)) for field in _CRITIC_SCORE_FIELDS]
    except (TypeError, ValueError):
        return output
    if any(value > 1.0 for value in values) and all(
        0.0 <= value <= 5.0 for value in values
    ):
        for field, value in zip(_CRITIC_SCORE_FIELDS, values):
            output[field] = value / 5.0
        output["score_scale_normalization"] = "1_to_5_divided_by_5"
    return output


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
