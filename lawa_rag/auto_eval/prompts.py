"""Versioned prompts for question, reference-answer, and critic calls."""

from __future__ import annotations

import json

from .cases import GeneratedEvalCase
from .corpus import CorpusSnapshot
from .personas import Persona
from .planning import ScenarioBlueprint
from .sampling import EvidenceBundle


GENERATION_PROMPT_VERSION = "grounded-auto-eval.v2"


def question_prompts(
    snapshot: CorpusSnapshot,
    persona: Persona,
    scenario: ScenarioBlueprint,
    bundle: EvidenceBundle,
    *,
    feedback: str = "",
) -> tuple[str, str]:
    system = (
        "You generate realistic evaluation questions for a retrieval-grounded assistant. "
        "Write as the supplied persona, not as an evaluator. Never mention evidence, "
        "context, chunks, "
        "or source IDs. Return one strict JSON object only."
    )
    rules = _answerability_rules(scenario)
    payload = {
        "corpus": snapshot.summary(),
        "persona": persona.to_dict(),
        "scenario": scenario.to_dict(),
        "evidence_bundle": bundle.to_prompt_dict(snapshot),
        "previous_attempt_feedback": feedback or None,
    }
    schema = {
        "query": "natural user query in the requested language",
        "intent": "short description",
        "expected_response_structure": "short_answer|paragraph|bullets|table|step_by_step",
    }
    user = (
        f"Rules:\n{rules}\nTask contract:\n{_task_rule(scenario.task)}\n"
        "Use the structural difficulty and requested reasoning operations. Distractors must not be "
        "needed to answer the question. Every named entity and constraint in the question must be "
        "supported by GOLD evidence. Do not copy long source phrases.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}\n"
        f"Output shape:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    return system, user


def reference_prompts(
    snapshot: CorpusSnapshot,
    scenario: ScenarioBlueprint,
    bundle: EvidenceBundle,
    query: str,
    *,
    feedback: str = "",
) -> tuple[str, str]:
    system = (
        "You are an independent reference-answer author. Use only GOLD evidence, never distractors "
        "or outside knowledge. Decompose the answer into atomic claims and cite exact "
        "evidence IDs. "
        "Citation quotes must be short exact substrings of their evidence text. Return JSON only."
    )
    payload = {
        "query": query,
        "answerable": scenario.answerable,
        "task": scenario.task,
        "language": scenario.language,
        "corpus_snapshot_id": snapshot.snapshot_id,
        "evidence_bundle": bundle.to_prompt_dict(snapshot),
        "previous_attempt_feedback": feedback or None,
    }
    schema = {
        "reference_answer": "complete grounded answer or explicit abstention",
        "claims": [{"text": "atomic claim", "evidence_ids": ["exact GOLD evidence id"]}],
        "citations": [{"evidence_id": "exact GOLD evidence id", "quote": "exact quote"}],
        "answer_must_include": ["essential answer requirement"],
        "answer_must_not_include": ["unsupported or misleading claim"],
        "answer_should_cover": ["coverage requirement"],
        "citation_requirements": ["claim that needs citation support"],
    }
    negative_rule = (
        "For an unanswerable case, return a useful abstention and empty claims and citations."
        if not scenario.answerable
        else (
            "Every substantive claim must be supported by at least one GOLD evidence ID. "
            "For medium and hard cases, every GOLD evidence unit must support at least one "
            "atomic claim and citation; otherwise the case is invalid."
        )
    )
    user = (
        f"{negative_rule}\nCorrect every issue in previous_attempt_feedback when present. "
        "Copy citation quotes exactly, including spelling and punctuation.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}\n"
        f"Output shape:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    return system, user


def critic_prompts(
    snapshot: CorpusSnapshot,
    case: GeneratedEvalCase,
    bundle: EvidenceBundle,
) -> tuple[str, str]:
    system = (
        "Audit a synthetic evaluation case conservatively. Reject unsupported answers, invalid "
        "citations, questions not answerable by all required gold evidence, unnatural "
        "persona wording, task-label mismatches, incomplete answers, named entities introduced "
        "outside the gold evidence, or declared no-answer cases that may be supported. "
        "Score each dimension independently; do not return uniform scores by default. "
        "Return strict JSON only."
    )
    payload = {
        "corpus": snapshot.summary(),
        "case": case.to_dict(),
        "evidence_bundle": bundle.to_prompt_dict(snapshot),
    }
    schema = {
        "passed": True,
        "answerability_correct": True,
        "grounding_score": 1.0,
        "naturalness_score": 1.0,
        "difficulty_match_score": 1.0,
        "query_grounding_score": 1.0,
        "answer_completeness_score": 1.0,
        "task_match_score": 1.0,
        "persona_fit_score": 1.0,
        "language_quality_score": 1.0,
        "issues": ["specific defect"],
    }
    user = (
        f"Audit input:\n{json.dumps(payload, ensure_ascii=False)}\n"
        f"Output shape:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    return system, user


def _answerability_rules(scenario: ScenarioBlueprint) -> str:
    if scenario.answerable:
        return (
            "The question must be fully answerable from GOLD evidence. For medium/hard "
            "cases, make the requested synthesis genuinely require the selected evidence "
            "rather than optional context."
        )
    return (
        "Ask a plausible question that the supplied corpus snapshot and nearby distractors do not "
        "answer. Do not merely create nonsense. Absence will require later human approval."
    )


def _task_rule(task: str) -> str:
    rules = {
        "fact": "Ask for one directly supported fact.",
        "explain": "Ask for an explanation of scope, purpose, process, or mechanism.",
        "compare": (
            "Require supported points from every compared source; invent no comparison axis."
        ),
        "synthesis": "Require integration of every selected GOLD unit into one coherent answer.",
        "navigate": (
            "Ask how or where to perform an action, reach a resource, use a form, or contact the "
            "right office. A plain who/what lookup is not navigation."
        ),
        "aggregate": "Require a calculation or roll-up from structured GOLD records.",
        "multimodal": "Make the media evidence necessary to answer, not decorative context.",
    }
    return rules.get(task, "Ask a realistic question matching the declared task.")


__all__ = [
    "GENERATION_PROMPT_VERSION",
    "critic_prompts",
    "question_prompts",
    "reference_prompts",
]
