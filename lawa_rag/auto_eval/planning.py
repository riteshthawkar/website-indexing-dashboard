"""Balanced scenario planning with structural, inspectable difficulty."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ._coverage import (
    align_tasks_and_difficulties,
    answerable_schedule,
    available_task,
    validate_weights,
    weighted_schedule,
)
from ._persona_selection import select_persona
from ._util import canonical_json, clean_text, json_mapping, stable_id, string_tuple
from .corpus import CorpusSnapshot
from .personas import Persona


DIFFICULTY_LEVELS = ("easy", "medium", "hard")
PLANNING_VERSION = "task-language-persona.v3"
TASK_TYPES = (
    "fact",
    "explain",
    "compare",
    "synthesis",
    "navigate",
    "aggregate",
    "multimodal",
    "follow_up",
)


@dataclass(frozen=True)
class DifficultyProfile:
    target_level: str
    evidence_count: int
    source_count: int
    graph_hops: int
    reasoning_operations: tuple[str, ...]
    modality_count: int = 1
    distractor_count: int = 0
    temporal_reasoning: bool = False

    @classmethod
    def for_level(cls, level: str, task: str) -> "DifficultyProfile":
        normalized = clean_text(level).lower()
        if normalized == "easy":
            return cls(normalized, 1, 1, 0, ("lookup",))
        if normalized == "medium":
            sources = 2 if task == "compare" else 1
            operations = ("retrieve", "compare") if task == "compare" else ("retrieve", "summarize")
            return cls(normalized, 2, sources, int(task == "navigate"), operations, 1, 1)
        if normalized == "hard":
            operations = ("retrieve", "integrate", "reason")
            modalities = 2 if task == "multimodal" else 1
            sources = 1 if task == "navigate" else 2
            evidence = 2 if task in {"navigate", "explain"} else 3
            hops = 2 if task == "navigate" else 1
            return cls(normalized, evidence, sources, hops, operations, modalities, 2)
        raise ValueError(f"Unsupported difficulty: {level!r}")

    @property
    def score(self) -> float:
        return round(
            max(0, self.evidence_count - 1)
            + 1.5 * max(0, self.source_count - 1)
            + float(self.graph_hops)
            + 0.8 * max(0, len(self.reasoning_operations) - 1)
            + 0.8 * max(0, self.modality_count - 1)
            + 0.3 * self.distractor_count
            + 1.2 * float(self.temporal_reasoning),
            3,
        )

    @property
    def measured_level(self) -> str:
        if self.score >= 4.5:
            return "hard"
        if self.score >= 1.8:
            return "medium"
        return "easy"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "score": self.score, "measured_level": self.measured_level}


@dataclass(frozen=True)
class CoveragePlan:
    count: int = 100
    languages: Mapping[str, float] = field(default_factory=lambda: {"English": 1.0})
    difficulties: Mapping[str, float] = field(
        default_factory=lambda: {"easy": 0.3, "medium": 0.45, "hard": 0.25}
    )
    tasks: Mapping[str, float] = field(
        default_factory=lambda: {
            "fact": 0.2,
            "explain": 0.15,
            "compare": 0.15,
            "synthesis": 0.2,
            "navigate": 0.1,
            "aggregate": 0.08,
            "multimodal": 0.07,
            "follow_up": 0.05,
        }
    )
    no_answer_fraction: float = 0.1
    seed: int = 17
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validated(self) -> "CoveragePlan":
        if self.count < 1:
            raise ValueError("CoveragePlan.count must be positive")
        validate_weights("languages", self.languages, None)
        validate_weights("difficulties", self.difficulties, set(DIFFICULTY_LEVELS))
        validate_weights("tasks", self.tasks, set(TASK_TYPES))
        if not 0.0 <= self.no_answer_fraction <= 1.0:
            raise ValueError("no_answer_fraction must be between 0 and 1")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        return stable_id("coverage-plan", canonical_json(self.to_dict()), length=16)


@dataclass(frozen=True)
class ScenarioBlueprint:
    scenario_id: str
    ordinal: int
    persona_id: str
    task: str
    language: str
    answerable: bool
    difficulty: DifficultyProfile
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScenarioBlueprint":
        difficulty = value.get("difficulty") or {}
        return cls(
            scenario_id=clean_text(value.get("scenario_id")),
            ordinal=int(value.get("ordinal") or 0),
            persona_id=clean_text(value.get("persona_id")),
            task=clean_text(value.get("task")),
            language=clean_text(value.get("language")),
            answerable=bool(value.get("answerable")),
            difficulty=DifficultyProfile(
                target_level=clean_text(difficulty.get("target_level")),
                evidence_count=int(difficulty.get("evidence_count") or 0),
                source_count=int(difficulty.get("source_count") or 0),
                graph_hops=int(difficulty.get("graph_hops") or 0),
                reasoning_operations=string_tuple(difficulty.get("reasoning_operations")),
                modality_count=int(difficulty.get("modality_count") or 1),
                distractor_count=int(difficulty.get("distractor_count") or 0),
                temporal_reasoning=bool(difficulty.get("temporal_reasoning")),
            ),
            metadata=json_mapping(value.get("metadata")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScenarioPlanner:
    def plan(
        self,
        snapshot: CorpusSnapshot,
        personas: Sequence[Persona],
        coverage: CoveragePlan,
    ) -> tuple[ScenarioBlueprint, ...]:
        plan = coverage.validated()
        if not personas:
            raise ValueError("At least one persona is required")
        rng = random.Random(plan.seed)
        tasks = weighted_schedule(plan.tasks, plan.count, rng)
        levels = align_tasks_and_difficulties(
            tasks, weighted_schedule(plan.difficulties, plan.count, rng)
        )
        languages = weighted_schedule(plan.languages, plan.count, rng)
        answerable = answerable_schedule(plan.count, plan.no_answer_fraction, rng)
        output = []
        persona_usage = {item.persona_id: 0 for item in personas}
        for index in range(plan.count):
            task = available_task(tasks[index], snapshot)
            persona = select_persona(personas, task, languages[index], persona_usage)
            persona_usage[persona.persona_id] += 1
            scenario_id = stable_id(
                "scenario",
                snapshot.snapshot_id,
                plan.fingerprint,
                index,
                persona.persona_id,
                task,
                levels[index],
                languages[index],
                length=20,
            )
            output.append(
                ScenarioBlueprint(
                    scenario_id=scenario_id,
                    ordinal=index + 1,
                    persona_id=persona.persona_id,
                    task=task,
                    language=languages[index],
                    answerable=answerable[index],
                    difficulty=DifficultyProfile.for_level(levels[index], task),
                )
            )
        return tuple(output)


__all__ = [
    "DIFFICULTY_LEVELS",
    "TASK_TYPES",
    "CoveragePlan",
    "DifficultyProfile",
    "ScenarioBlueprint",
    "ScenarioPlanner",
    "PLANNING_VERSION",
]
