"""Versioned generated evaluation-case contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from ._util import clean_text, json_mapping, string_tuple
from .planning import DifficultyProfile


EVAL_CASE_SCHEMA_VERSION = "lawa.auto_eval.case.v1"


@dataclass(frozen=True)
class GoldClaim:
    text: str
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GoldClaim":
        return cls(
            text=clean_text(value.get("text")),
            evidence_ids=string_tuple(value.get("evidence_ids")),
        )


@dataclass(frozen=True)
class EvidenceCitation:
    evidence_id: str
    quote: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceCitation":
        return cls(
            evidence_id=clean_text(value.get("evidence_id")),
            quote=clean_text(value.get("quote")),
        )


@dataclass(frozen=True)
class GeneratedEvalCase:
    case_id: str
    scenario_id: str
    persona_id: str
    query: str
    language: str
    task: str
    answerable: bool
    target_difficulty: str
    measured_difficulty: DifficultyProfile
    reference_answer: str
    claims: tuple[GoldClaim, ...]
    citations: tuple[EvidenceCitation, ...]
    evidence_ids: tuple[str, ...]
    distractor_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    expected_response_structure: str = "concise grounded answer"
    answer_must_include: tuple[str, ...] = ()
    answer_must_not_include: tuple[str, ...] = ()
    answer_should_cover: tuple[str, ...] = ()
    citation_requirements: tuple[str, ...] = ()
    review_status: str = "auto_validated"
    critic: Mapping[str, Any] = field(default_factory=dict)
    generation: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVAL_CASE_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedEvalCase":
        difficulty = value.get("measured_difficulty") or {}
        return cls(
            case_id=clean_text(value.get("case_id")),
            scenario_id=clean_text(value.get("scenario_id")),
            persona_id=clean_text(value.get("persona_id")),
            query=clean_text(value.get("query")),
            language=clean_text(value.get("language")),
            task=clean_text(value.get("task")),
            answerable=bool(value.get("answerable")),
            target_difficulty=clean_text(value.get("target_difficulty")),
            measured_difficulty=_difficulty_from_dict(difficulty),
            reference_answer=clean_text(value.get("reference_answer")),
            claims=tuple(GoldClaim.from_dict(item) for item in value.get("claims") or []),
            citations=tuple(
                EvidenceCitation.from_dict(item) for item in value.get("citations") or []
            ),
            evidence_ids=string_tuple(value.get("evidence_ids")),
            distractor_ids=string_tuple(value.get("distractor_ids")),
            source_ids=string_tuple(value.get("source_ids")),
            expected_response_structure=clean_text(value.get("expected_response_structure")),
            answer_must_include=string_tuple(value.get("answer_must_include")),
            answer_must_not_include=string_tuple(value.get("answer_must_not_include")),
            answer_should_cover=string_tuple(value.get("answer_should_cover")),
            citation_requirements=string_tuple(value.get("citation_requirements")),
            review_status=clean_text(value.get("review_status")) or "auto_validated",
            critic=json_mapping(value.get("critic")),
            generation=json_mapping(value.get("generation")),
            metadata=json_mapping(value.get("metadata")),
            schema_version=clean_text(value.get("schema_version")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _difficulty_from_dict(value: Mapping[str, Any]) -> DifficultyProfile:
    return DifficultyProfile(
        target_level=clean_text(value.get("target_level")),
        evidence_count=int(value.get("evidence_count") or 0),
        source_count=int(value.get("source_count") or 0),
        graph_hops=int(value.get("graph_hops") or 0),
        reasoning_operations=string_tuple(value.get("reasoning_operations")),
        modality_count=int(value.get("modality_count") or 1),
        distractor_count=int(value.get("distractor_count") or 0),
        temporal_reasoning=bool(value.get("temporal_reasoning")),
    )


__all__ = [
    "EVAL_CASE_SCHEMA_VERSION",
    "EvidenceCitation",
    "GeneratedEvalCase",
    "GoldClaim",
]
