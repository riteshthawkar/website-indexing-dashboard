"""Coverage, persona selection, and difficulty-planning contract tests."""

from __future__ import annotations

from collections import Counter

from lawa_rag.auto_eval import (
    CoveragePlan,
    DifficultyProfile,
    Persona,
    RetrievalBundleAdapter,
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


def test_single_query_planner_maps_follow_up_to_supported_task() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    plan = CoveragePlan(
        count=1,
        tasks={"follow_up": 1.0},
        languages={"English": 1.0},
        difficulties={"easy": 1.0},
    )

    scenario = ScenarioPlanner().plan(snapshot, (student_persona(),), plan)[0]

    assert scenario.task == "explain"


def test_planner_pairs_constrained_tasks_with_sensible_difficulty() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    plan = CoveragePlan(
        count=8,
        tasks={
            "fact": 1,
            "explain": 2,
            "compare": 1,
            "synthesis": 1,
            "navigate": 2,
            "multimodal": 1,
        },
        difficulties={"easy": 3, "medium": 3, "hard": 2},
        languages={"English": 1},
    )

    scenarios = ScenarioPlanner().plan(snapshot, (student_persona(),), plan)
    task_levels = {(item.task, item.difficulty.target_level) for item in scenarios}

    assert ("fact", "easy") in task_levels
    assert not any(task == "fact" and level == "hard" for task, level in task_levels)
    assert not any(task == "compare" and level == "easy" for task, level in task_levels)
    hard_explain = DifficultyProfile.for_level("hard", "explain")
    assert hard_explain.evidence_count == 2
    assert hard_explain.measured_level == "hard"


def test_planner_assigns_personas_that_support_the_query_language() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    english = Persona.create(
        role="English-speaking applicant",
        goals=("understand admissions",),
        preferred_languages=("en",),
        likely_tasks=("fact", "explain"),
    )
    arabic = Persona.create(
        role="Arabic-speaking applicant",
        goals=("understand admissions",),
        preferred_languages=("ar",),
        likely_tasks=("fact", "explain"),
    )
    plan = CoveragePlan(
        count=4,
        tasks={"fact": 1, "explain": 1},
        difficulties={"easy": 1},
        languages={"English": 1, "Arabic": 1},
    )

    scenarios = ScenarioPlanner().plan(snapshot, (english, arabic), plan)

    expected = {"English": english.persona_id, "Arabic": arabic.persona_id}
    assert all(item.persona_id == expected[item.language] for item in scenarios)
