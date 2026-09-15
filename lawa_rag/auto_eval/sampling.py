"""Deterministic evidence and distractor selection for evaluation scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ._sampling_support import (
    adjacency,
    deduplicated_candidates,
    modality,
    prompt_unit,
    semantic_overlap,
    stable_number,
    usable_for_generation,
)
from .corpus import CorpusSnapshot, EvidenceUnit
from .planning import DifficultyProfile, ScenarioBlueprint


_TASK_KINDS = {
    "fact": {"fact", "evidence_span", "answer", "chunk"},
    "navigate": {"action"},
    "aggregate": {"table", "database"},
    "multimodal": {"media", "image", "video"},
}

_TEXT_KINDS = {
    "chunk", "parent", "fact", "summary", "assertion", "answer", "evidence_span"
}
_CANDIDATE_KINDS = {
    "fact": _TEXT_KINDS,
    "explain": _TEXT_KINDS,
    "compare": _TEXT_KINDS,
    "synthesis": _TEXT_KINDS,
    "navigate": {"action", "page_card"},
    "aggregate": {"table", "database"},
}


@dataclass(frozen=True)
class EvidenceBundle:
    scenario_id: str
    evidence: tuple[EvidenceUnit, ...]
    distractors: tuple[EvidenceUnit, ...]
    source_ids: tuple[str, ...]
    difficulty: DifficultyProfile

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidence)

    @property
    def distractor_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.distractors)

    def to_prompt_dict(
        self,
        snapshot: CorpusSnapshot,
        *,
        max_chars_per_unit: int = 2600,
    ) -> dict[str, Any]:
        sources = snapshot.source_map()
        return {
            "scenario_id": self.scenario_id,
            "difficulty": self.difficulty.to_dict(),
            "evidence": [
                prompt_unit(item, sources.get(item.source_id), max_chars_per_unit)
                for item in self.evidence
            ],
            "distractors": [
                prompt_unit(item, sources.get(item.source_id), max_chars_per_unit)
                for item in self.distractors
            ],
        }


class EvidenceSampler:
    def __init__(self, snapshot: CorpusSnapshot, *, seed: int = 17):
        self.snapshot = snapshot
        self.seed = seed
        self._sources = snapshot.source_map()
        self._adjacency = adjacency(snapshot)

    def sample(self, scenario: ScenarioBlueprint) -> EvidenceBundle:
        candidates = self._task_candidates(scenario.task)
        if not candidates:
            raise ValueError(f"No evidence is available for task {scenario.task!r}")
        if not scenario.answerable:
            distractors = self._pick_distractors(
                (), candidates, max(2, scenario.difficulty.distractor_count)
            )
            return EvidenceBundle(
                scenario_id=scenario.scenario_id,
                evidence=(),
                distractors=distractors,
                source_ids=(),
                difficulty=scenario.difficulty,
            )
        anchors = self._anchor_candidates(scenario.task, candidates)
        anchor_index = stable_number(scenario.scenario_id, self.seed) % len(anchors)
        selected = [anchors[anchor_index]]
        target = min(len(candidates), scenario.difficulty.evidence_count)
        while len(selected) < target:
            remaining = [item for item in candidates if item not in selected]
            remaining = deduplicated_candidates(remaining, selected, self._sources)
            selected_sources = {item.source_id for item in selected}
            if len(selected_sources) < scenario.difficulty.source_count:
                diverse = [item for item in remaining if item.source_id not in selected_sources]
                remaining = diverse or remaining
            selected.append(
                max(
                    remaining,
                    key=lambda item: self._candidate_score(item, selected, scenario),
                )
            )
        distractors = self._pick_distractors(
            selected,
            list(self.snapshot.evidence),
            scenario.difficulty.distractor_count,
        )
        difficulty = self._measured_difficulty(scenario, selected, distractors)
        source_ids = tuple(dict.fromkeys(item.source_id for item in selected))
        return EvidenceBundle(
            scenario_id=scenario.scenario_id,
            evidence=tuple(selected),
            distractors=distractors,
            source_ids=source_ids,
            difficulty=difficulty,
        )

    def _task_candidates(self, task: str) -> list[EvidenceUnit]:
        useful = [item for item in self.snapshot.evidence if usable_for_generation(item)]
        allowed = _CANDIDATE_KINDS.get(task)
        scoped = [item for item in useful if item.kind in allowed] if allowed else useful
        return sorted(scoped or useful, key=lambda item: item.evidence_id)

    def _anchor_candidates(
        self, task: str, candidates: Sequence[EvidenceUnit]
    ) -> list[EvidenceUnit]:
        preferred = _TASK_KINDS.get(task)
        selected = [item for item in candidates if item.kind in preferred] if preferred else []
        return selected or list(candidates)

    def _candidate_score(
        self,
        candidate: EvidenceUnit,
        selected: Sequence[EvidenceUnit],
        scenario: ScenarioBlueprint,
    ) -> tuple[float, str]:
        overlap = max(
            (semantic_overlap(candidate, item, self._sources) for item in selected),
            default=0.0,
        )
        distance = min(
            (self._distance(candidate.evidence_id, item.evidence_id) for item in selected),
            default=0,
        )
        connected = 3.0 if distance > 0 else 0.0
        selected_sources = {item.source_id for item in selected}
        diverse = candidate.source_id not in selected_sources
        diversity = (
            2.0
            if diverse and scenario.difficulty.source_count > len(selected_sources)
            else 0.0
        )
        same_source = (
            1.5 if not diverse and scenario.difficulty.source_count <= 1 else 0.0
        )
        return connected + 4.0 * overlap + diversity + same_source, candidate.evidence_id

    def _distance(self, start: str, target: str, *, maximum: int = 3) -> int:
        if start == target:
            return 0
        frontier = {start}
        visited = {start}
        for distance in range(1, maximum + 1):
            frontier = {
                node for item in frontier for node in self._adjacency.get(item, ())
            } - visited
            if target in frontier:
                return distance
            visited.update(frontier)
            if not frontier:
                break
        return 0

    def _pick_distractors(
        self, selected, candidates, count: int
    ) -> tuple[EvidenceUnit, ...]:
        if count <= 0:
            return ()
        selected_ids = {item.evidence_id for item in selected}
        options = [item for item in candidates if item.evidence_id not in selected_ids]
        ranked = sorted(
            options,
            key=lambda item: (
                -max(
                    (
                        semantic_overlap(item, chosen, self._sources)
                        for chosen in selected
                    ),
                    default=0.0,
                ),
                item.evidence_id,
            ),
        )
        return tuple(ranked[:count])

    def _measured_difficulty(self, scenario, selected, distractors) -> DifficultyProfile:
        source_count = len({item.source_id for item in selected})
        kinds = {item.kind for item in selected}
        hops = max(
            (
                self._distance(left.evidence_id, right.evidence_id)
                for index, left in enumerate(selected)
                for right in selected[index + 1 :]
            ),
            default=0,
        )
        return DifficultyProfile(
            target_level=scenario.difficulty.target_level,
            evidence_count=len(selected),
            source_count=source_count,
            graph_hops=hops,
            reasoning_operations=scenario.difficulty.reasoning_operations,
            modality_count=max(1, len({modality(kind) for kind in kinds})),
            distractor_count=len(distractors),
            temporal_reasoning=scenario.difficulty.temporal_reasoning,
        )


__all__ = ["EvidenceBundle", "EvidenceSampler"]
