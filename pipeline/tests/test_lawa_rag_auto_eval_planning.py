"""Coverage, persona, difficulty, and evidence-sampling contract tests."""

from __future__ import annotations

from collections import Counter

from lawa_rag.auto_eval import (
    CoveragePlan,
    DifficultyProfile,
    EvidenceSampler,
    Persona,
    RetrievalBundleAdapter,
    ScenarioBlueprint,
    ScenarioPlanner,
)
from pipeline.tests.lawa_rag_auto_eval_support import sample_bundle, student_persona


def test_coverage_planner_hits_requested_quotas_deterministically() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    personas = (
        student_persona(),
        Persona.create(
            role="prospective researcher",
            goals=("compare research and study options",),
            domain="university",
        ),
    )
    plan = CoveragePlan(
        count=20,
        languages={"English": 0.5, "Arabic": 0.5},
        difficulties={"easy": 0.25, "medium": 0.5, "hard": 0.25},
        tasks={"fact": 0.5, "navigate": 0.25, "multimodal": 0.25},
        no_answer_fraction=0.2,
        seed=9,
    )
    planner = ScenarioPlanner()
    first = planner.plan(snapshot, personas, plan)
    second = planner.plan(snapshot, personas, plan)

    assert first == second
    assert Counter(item.language for item in first) == {"English": 10, "Arabic": 10}
    assert Counter(item.difficulty.target_level for item in first) == {
        "easy": 5,
        "medium": 10,
        "hard": 5,
    }
    assert sum(not item.answerable for item in first) == 4
    assert len({item.scenario_id for item in first}) == 20


def test_sampler_enforces_source_diversity_and_separates_distractors() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    scenario = ScenarioBlueprint(
        scenario_id="scenario:hard",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="synthesis",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("hard", "synthesis"),
    )
    bundle = EvidenceSampler(snapshot).sample(scenario)

    assert len(bundle.evidence) == 3
    assert len(bundle.source_ids) == 2
    assert len(bundle.distractors) == 2
    assert set(bundle.evidence_ids).isdisjoint(bundle.distractor_ids)
    assert bundle.difficulty.measured_level == "hard"


def test_unanswerable_scenarios_have_no_gold_evidence() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    scenario = ScenarioBlueprint(
        scenario_id="scenario:negative",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="fact",
        language="English",
        answerable=False,
        difficulty=DifficultyProfile.for_level("easy", "fact"),
    )
    bundle = EvidenceSampler(snapshot).sample(scenario)

    assert bundle.evidence == ()
    assert len(bundle.distractors) == 2
