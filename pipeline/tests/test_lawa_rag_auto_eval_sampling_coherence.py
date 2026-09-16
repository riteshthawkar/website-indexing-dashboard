"""Topical-coherence and comparison-scope tests for evidence bundles."""

from lawa_rag.auto_eval import (
    DifficultyProfile,
    EvidenceSampler,
    RetrievalBundleAdapter,
    ScenarioBlueprint,
)
from lawa_rag.auto_eval._coherence import coherent_pair, section_overlap
from pipeline.tests.lawa_rag_auto_eval_support import student_persona


def test_sampler_builds_a_connected_topic_instead_of_same_page_noise() -> None:
    shared = {
        "document_id": "admissions",
        "document_title": "Admissions",
        "source_url": "https://example.edu/admissions",
    }
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    **shared,
                    "id": "chunk:international",
                    "text": "## International students We welcome qualified applicants from "
                    "every country and educational background.",
                    "section_path": ["International students"],
                },
                {
                    **shared,
                    "id": "chunk:diversity",
                    "text": "## Diverse community International learners represent many "
                    "countries, educational systems, communities, and cultures.",
                    "section_path": ["International student diversity"],
                },
                {
                    **shared,
                    "id": "chunk:deadline",
                    "text": "## Application deadline The application closes on 15 January.",
                    "section_path": ["Important dates"],
                },
            ]
        },
        domain="university",
    )
    scenario = ScenarioBlueprint(
        scenario_id="scenario:coherent",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="explain",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("medium", "explain"),
    )

    selected = set(EvidenceSampler(snapshot).sample(scenario).evidence_ids)

    assert selected == {"chunk:international", "chunk:diversity"}


def test_generic_meet_our_team_headings_do_not_create_topic_coherence() -> None:
    source = {
        "document_id": "teams",
        "document_title": "Teams",
        "source_url": "https://example.edu/teams",
    }
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    **source,
                    "id": "chunk:housing-team",
                    "section_path": ["Housing", "Meet our team"],
                    "text": "Housing advisers help residents choose accommodation options.",
                },
                {
                    **source,
                    "id": "chunk:research-team",
                    "section_path": ["Research", "Meet our team"],
                    "text": "Research specialists help faculty commercialize inventions.",
                },
            ]
        },
        domain="university",
    )
    left, right = snapshot.evidence

    assert section_overlap(left, right) == 0.0


def test_unrelated_faculty_cards_do_not_form_an_explanation_bundle() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:eric",
                    "document_id": "division",
                    "document_title": "Division",
                    "text": "Dean and Professor of Machine Learning "
                    "](https://example.edu/faculty-directory/eric-moulines)",
                },
                {
                    "id": "chunk:kun",
                    "document_id": "program",
                    "document_title": "Program",
                    "text": "Visiting Professor of Machine Learning "
                    "](https://example.edu/faculty-directory/kun-zhang)",
                },
            ]
        },
        domain="university",
    )
    left, right = snapshot.evidence

    assert not coherent_pair(left, right, snapshot.source_map(), {})


def test_compare_sampler_requires_equivalent_section_scope() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:program-a-purpose",
                    "document_id": "program-a",
                    "document_title": "Program A",
                    "source_url": "https://example.edu/program-a",
                    "section_path": ["Admissions", "Statement of purpose"],
                    "text": "## Statement of purpose Submit an essay describing motivation, "
                    "academic preparation, career goals, and relevant project experience.",
                },
                {
                    "id": "chunk:program-b-broad",
                    "document_id": "program-b",
                    "document_title": "Program B",
                    "source_url": "https://example.edu/program-b",
                    "section_path": ["Admission criteria"],
                    "text": "## Admission criteria Applicants provide an essay, three references, "
                    "a portfolio, transcripts, an interview, and employment records.",
                },
                {
                    "id": "chunk:program-c-purpose",
                    "document_id": "program-c",
                    "document_title": "Program C",
                    "source_url": "https://example.edu/program-c",
                    "section_path": ["Admissions", "Statement of purpose"],
                    "text": "## Statement of purpose Submit an essay covering motivation, prior "
                    "education, professional goals, and applicable research experience.",
                },
            ]
        },
        domain="university",
    )
    scenario = ScenarioBlueprint(
        scenario_id="scenario:comparable-scope",
        ordinal=1,
        persona_id=student_persona().persona_id,
        task="compare",
        language="English",
        answerable=True,
        difficulty=DifficultyProfile.for_level("medium", "compare"),
    )

    selected = set(EvidenceSampler(snapshot).sample(scenario).evidence_ids)

    assert selected == {"chunk:program-a-purpose", "chunk:program-c-purpose"}
