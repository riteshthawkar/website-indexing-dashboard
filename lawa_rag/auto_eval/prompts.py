"""Versioned prompts for question, reference-answer, and critic calls."""

from __future__ import annotations

import json

from .cases import GeneratedEvalCase
from .corpus import CorpusSnapshot
from .personas import Persona
from .planning import ScenarioBlueprint
from .sampling import EvidenceBundle


GENERATION_PROMPT_VERSION = "grounded-auto-eval.v12"
_REFERENCE_SCHEMA = {
    "reference_answer": "complete grounded answer or explicit abstention",
    "claims": [{"text": "atomic claim", "evidence_ids": ["exact GOLD evidence id"]}],
    "citations": [{"evidence_id": "exact GOLD evidence id", "quote": "exact quote"}],
    "answer_must_include": ["essential answer requirement"],
    "answer_must_not_include": ["unsupported or misleading claim"],
    "answer_should_cover": ["coverage requirement"],
    "citation_requirements": ["claim that needs citation support"],
}


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
        f"Rules:\n{rules}\nTask contract:\n{_task_rule(scenario.task, bundle)}\n"
        "Use the structural difficulty and requested reasoning operations. Distractors must not be "
        "needed to answer the question. Every named entity and constraint in the question must be "
        "supported by GOLD evidence. The requested language controls the query language only; do "
        "not call a page English or Arabic based only on its URL or metadata. Do not copy long "
        "source phrases. Preserve names of people, organizations, programs, and courses exactly "
        "as supplied by GOLD evidence. When the requested language uses another script, retain "
        "the evidence's official spelling unless GOLD evidence provides an authoritative localized "
        "name; never invent a transliteration. Do not ask for all, complete, or exhaustive "
        "requirements, documents, "
        "steps, or categories unless GOLD evidence explicitly covers every requested item. For "
        "comparisons, name the exact comparison dimensions instead of asking vaguely for "
        "'detailed requirements' or 'everything'.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}\n"
        f"Output shape:\n{json.dumps(schema, ensure_ascii=False)}"
    )
    return system, user


def reference_prompts(
    snapshot: CorpusSnapshot, scenario: ScenarioBlueprint,
    bundle: EvidenceBundle, query: str, *, feedback: str = "",
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
        "Silently split the query into every requested subpart and make the answer address each "
        "subpart explicitly. Do not replace a requested list with a broad summary. Copy citation "
        "quotes exactly, including spelling and punctuation. Keep evidence IDs only in the "
        "claims and citations fields; never expose an evidence ID in reference_answer. Preserve "
        "proper names exactly as written in GOLD evidence, including when answering in another "
        "script; do not invent translated or transliterated names. Treat words such as all, "
        "complete, every, and detailed as literal coverage obligations when they occur in the "
        "query. For a comparison, include at least one explicit comparative atomic claim supported "
        "jointly by evidence from every compared source. Never treat an excerpt's silence as "
        "evidence that something is absent. For media "
        "claims, preserve visible counts, colors, labels, and text literally; do not substitute a "
        "different visual attribute.\n"
        f"Input:\n{json.dumps(payload, ensure_ascii=False)}\n"
        f"Output shape:\n{json.dumps(_REFERENCE_SCHEMA, ensure_ascii=False)}"
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
        "outside the gold evidence, media questions that request facts absent from the media and "
        "linked text, language claims inferred only from URL metadata, or declared no-answer cases "
        "that may be supported. "
        "Before scoring, enumerate the query's requested subparts internally and verify that the "
        "reference answer answers every one. Reject broad or exhaustive questions when the gold "
        "excerpts establish only a partial list. Reject answers that merge distinct programs or "
        "levels without preserving their separate requirements. Reject invented, misspelled, or "
        "unsubstantiated translations/transliterations of proper names; retaining the evidence's "
        "official spelling in a different-script answer is acceptable. Reject any claim that a "
        "feature is absent merely because a selected excerpt does not mention it. For multimodal "
        "cases, compare counts, colors, clothing, labels, and visible text literally and reject "
        "any contradiction with the media description. "
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
        "rather than optional context. Explicitly ask for at least one distinctive fact "
        "or scope from every GOLD evidence unit."
        )
    return (
        "Ask a plausible question that the supplied corpus snapshot and nearby distractors do not "
        "answer. Do not merely create nonsense. Absence will require later human approval."
    )


def _task_rule(task: str, bundle: EvidenceBundle) -> str:
    rules = {
        "fact": (
            "Ask for one focused factual answer. If multiple GOLD units are present, the complete "
            "question and answer must genuinely require each unit."
        ),
        "explain": "Ask for an explanation of scope, purpose, process, or mechanism.",
        "compare": (
            "Compare only equivalent fields or sections from every source. Do not infer that a "
            "requirement, feature, or policy is absent merely because one selected excerpt does "
            "not mention it. Every requested comparison dimension must be positively stated in "
            "every compared GOLD unit; omit a dimension that is absent from any unit. Require "
            "supported points from every compared source."
        ),
        "synthesis": (
            "Require integration of every selected GOLD unit into one coherent answer. Keep the "
            "question scope to equivalent aspects explicitly present in every selected source; "
            "do not request a dimension absent from any unit or a complete document or requirement "
            "inventory from partial excerpts."
        ),
        "navigate": (
            "Ask how or where to perform an action, reach a resource, use a form, or contact the "
            "right office. A plain who/what lookup is not navigation."
        ),
        "aggregate": "Require a calculation or roll-up from structured GOLD records.",
        "multimodal": _multimodal_rule(bundle),
    }
    return rules.get(task, "Ask a realistic question matching the declared task.")


def _multimodal_rule(bundle: EvidenceBundle) -> str:
    kinds = {item.kind for item in bundle.evidence}
    if kinds <= {"media", "image", "video"}:
        return (
            "Ask only about visible content, an explicit caption, or an explicit media label. Do "
            "not ask what a named program, person, or service does unless the media text says so."
        )
    return (
        "Require both the media evidence and its topically linked textual evidence. Ask an "
        "explicitly visual question about what the image, photo, figure, diagram, or chart shows "
        "and connect that observation to the linked text. The media must be necessary rather "
        "than decorative context."
    )


__all__ = [
    "GENERATION_PROMPT_VERSION",
    "critic_prompts",
    "question_prompts",
    "reference_prompts",
]
