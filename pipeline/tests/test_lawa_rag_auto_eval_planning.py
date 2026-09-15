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
from lawa_rag.auto_eval._sampling_support import content_signature, usable_for_generation


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


def test_navigation_sampling_requires_an_action_anchor() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    scenario = ScenarioBlueprint(
        scenario_id="scenario:navigate",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="navigate",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("hard", "navigate"),
    )

    bundle = EvidenceSampler(snapshot).sample(scenario)

    assert any(item.kind == "action" for item in bundle.evidence)
    assert scenario.difficulty.evidence_count == 2
    assert scenario.difficulty.source_count == 1
    assert scenario.difficulty.graph_hops == 2
    assert scenario.difficulty.measured_level == "hard"


def test_sampler_excludes_low_signal_related_link_blocks() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:related-links",
                    "document_id": "page",
                    "text": "SECTION: You might also be interested in # You might also be "
                    "interested in [News](https://example.edu/news)",
                },
                {
                    "id": "chunk:useful",
                    "document_id": "page",
                    "text": "Applicants must submit official transcripts and two references.",
                },
            ]
        },
        domain="university",
    )

    units = snapshot.evidence_map()
    assert not usable_for_generation(units["chunk:related-links"])
    assert usable_for_generation(units["chunk:useful"])


def test_sampler_excludes_heading_only_converted_chunks() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:heading-only",
                    "document_id": "program",
                    "text": "TITLE: Program | SECTION: Details ## Admission criteria",
                },
                {
                    "id": "chunk:substantive-section",
                    "document_id": "program",
                    "text": "TITLE: Program | SECTION: Details ## Admission criteria "
                    "Applicants need a relevant degree, transcripts, references, and "
                    "proof of English proficiency.",
                },
            ]
        },
        domain="university",
    )

    units = snapshot.evidence_map()
    assert not usable_for_generation(units["chunk:heading-only"])
    assert usable_for_generation(units["chunk:substantive-section"])


def test_content_signature_deduplicates_copied_localized_records() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:english:00001:abcdef123456",
                    "document_id": "english",
                    "text": "TITLE: English page The same substantive body.",
                },
                {
                    "id": "chunk:arabic:00001:abcdef123456",
                    "document_id": "arabic",
                    "text": "TITLE: Arabic page The same substantive body.",
                },
            ]
        },
        domain="university",
    )

    first, second = snapshot.evidence
    assert content_signature(first) == content_signature(second)


def test_sampler_does_not_count_chunk_parent_copy_as_two_gold_units() -> None:
    shared = {
        "document_id": "admissions",
        "document_title": "Admissions",
        "source_url": "https://example.edu/admissions",
    }
    copied = "Applicants submit official transcripts and two recommendation letters."
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {**shared, "id": "chunk:requirements", "text": copied},
                {
                    **shared,
                    "id": "chunk:interview",
                    "text": "Shortlisted applicants complete an online faculty interview.",
                },
            ],
            "parent_records": [
                {**shared, "id": "parent:requirements", "text": copied},
            ],
        },
        domain="university",
    )
    scenario = ScenarioBlueprint(
        scenario_id="scenario:medium-copy",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="explain",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("medium", "explain"),
    )

    selected = EvidenceSampler(snapshot).sample(scenario).evidence_ids

    assert not {"chunk:requirements", "parent:requirements"}.issubset(selected)


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
