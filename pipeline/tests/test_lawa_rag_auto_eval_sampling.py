"""Core evidence-sampling behavior for automatic evaluation."""

from lawa_rag.auto_eval import (
    DifficultyProfile,
    EvidenceSampler,
    Persona,
    RetrievalBundleAdapter,
    ScenarioBlueprint,
)
from pipeline.tests.lawa_rag_auto_eval_support import sample_bundle, student_persona


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


def test_multimodal_sampling_pairs_media_with_linked_text() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    scenario = ScenarioBlueprint(
        scenario_id="scenario:multimodal",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="multimodal",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("medium", "multimodal"),
    )

    bundle = EvidenceSampler(snapshot).sample(scenario)

    assert {item.kind for item in bundle.evidence} == {"media", "chunk"}
    assert bundle.difficulty.measured_level == "medium"


def test_navigation_sampling_avoids_reusing_the_same_action_target() -> None:
    shared = {
        "document_id": "admissions",
        "document_title": "Admissions",
        "source_url": "https://example.edu/admissions",
        "language": "English",
    }
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "action_records": [
                {
                    **shared,
                    "id": "action:email-one",
                    "label": "Email admissions for Program One",
                    "action_type": "email",
                    "target_url": "mailto:admissions@example.edu",
                },
                {
                    **shared,
                    "id": "action:email-two",
                    "label": "Email admissions for Program Two",
                    "action_type": "email",
                    "target_url": "mailto:admissions@example.edu",
                },
                {
                    **shared,
                    "id": "action:visit",
                    "label": "Book a campus visit",
                    "action_type": "visit",
                    "target_url": "https://example.edu/visit",
                },
            ]
        },
        domain="university",
    )
    sampler = EvidenceSampler(snapshot)
    first = ScenarioBlueprint(
        scenario_id="scenario:navigate-one",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="navigate",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("easy", "navigate"),
    )
    second = ScenarioBlueprint(
        scenario_id="scenario:navigate-two",
        ordinal=2,
        persona_id=student_persona().persona_id,
        task="navigate",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("easy", "navigate"),
    )

    first_unit = sampler.sample(first).evidence[0]
    second_unit = sampler.sample(second).evidence[0]

    assert "admissions@example.edu" in first_unit.text or "visit" in first_unit.text
    assert not (
        "admissions@example.edu" in first_unit.text
        and "admissions@example.edu" in second_unit.text
    )


def test_sampler_prefers_evidence_relevant_to_the_persona_goal() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:graduate",
                    "document_id": "graduate",
                    "document_title": "Graduate research",
                    "language": "en",
                    "text": "Doctoral researchers complete an advanced dissertation and viva.",
                },
                {
                    "id": "chunk:undergraduate",
                    "document_id": "undergraduate",
                    "document_title": "Undergraduate admissions",
                    "language": "en",
                    "text": "Undergraduate applicants submit school transcripts and references.",
                },
            ]
        },
        domain="university",
    )
    persona = Persona.create(
        role="prospective undergraduate student",
        goals=("understand undergraduate admission requirements",),
        likely_tasks=("fact",),
    )
    scenario = ScenarioBlueprint(
        scenario_id="scenario:persona-relevance",
        ordinal=1,
        persona_id=persona.persona_id,
        task="fact",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("easy", "fact"),
    )

    bundle = EvidenceSampler(snapshot).sample(scenario, persona=persona)

    assert bundle.evidence_ids == ("chunk:undergraduate",)
