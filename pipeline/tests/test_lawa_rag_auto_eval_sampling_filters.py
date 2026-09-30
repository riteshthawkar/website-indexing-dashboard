"""Evidence quality and duplicate-filtering tests for auto-evaluation sampling."""

from lawa_rag.auto_eval import (
    DifficultyProfile,
    EvidenceSampler,
    RetrievalBundleAdapter,
    ScenarioBlueprint,
)
from lawa_rag.auto_eval._sampling_support import (
    content_signature,
    deduplicated_candidates,
    usable_for_generation,
)
from pipeline.tests.lawa_rag_auto_eval_support import student_persona


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


def test_sampler_excludes_metadata_wrappers_with_no_substantive_body() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:empty-page",
                    "document_id": "division",
                    "text": "TITLE: Computing Division TYPE: webpage SOURCE_URL: "
                    "https://example.edu/division About",
                },
                {
                    "id": "chunk:real-page",
                    "document_id": "division",
                    "text": "TITLE: Computing Division TYPE: webpage SOURCE_URL: "
                    "https://example.edu/division The division conducts interdisciplinary "
                    "research in machine learning, robotics, vision, and data science.",
                },
            ]
        },
        domain="university",
    )

    units = snapshot.evidence_map()
    assert not usable_for_generation(units["chunk:empty-page"])
    assert usable_for_generation(units["chunk:real-page"])


def test_sampler_excludes_misaligned_entity_cards() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:mismatched-person",
                    "document_id": "faculty",
                    "section_path": ["Faculty", "Alex Morgan"],
                    "text": "## Alex Morgan Professor of Machine Learning and Director of the "
                    "AI laboratory](https://example.edu/faculty-directory/taylor-smith)",
                },
                {
                    "id": "chunk:matched-person",
                    "document_id": "faculty",
                    "section_path": ["Faculty", "Alex Morgan"],
                    "text": "## Alex Morgan Professor of Machine Learning and Director of the "
                    "AI laboratory](https://example.edu/faculty-directory/alex-morgan)",
                },
                {
                    "id": "chunk:accented-mismatch",
                    "document_id": "faculty",
                    "section_path": ["Faculty", "Éric Moulines"],
                    "text": "## Éric Moulines Professor of Machine Learning and Director of the "
                    "AI laboratory with a long biography covering research, teaching, awards, "
                    "service, publications, collaborations, supervision, and university "
                    "leadership responsibilities across many academic initiatives and centers"
                    "](https://example.edu/faculty-directory/hao-li-0)",
                },
            ]
        },
        domain="university",
    )

    units = snapshot.evidence_map()
    assert not usable_for_generation(units["chunk:mismatched-person"])
    assert not usable_for_generation(units["chunk:accented-mismatch"])
    assert usable_for_generation(units["chunk:matched-person"])


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


def test_synthesis_does_not_reuse_localized_copies_of_the_same_page() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "chunk:student-life-en",
                    "document_id": "student-life-en",
                    "document_title": "Student life",
                    "section_path": ["Meet our team"],
                    "text": "The student support team offers wellbeing and accessibility help.",
                },
                {
                    "id": "chunk:student-life-ar",
                    "document_id": "student-life-ar",
                    "document_title": "Student life",
                    "section_path": ["Meet our team"],
                    "text": "The student support team offers wellbeing and accessibility help.",
                },
                {
                    "id": "chunk:housing",
                    "document_id": "housing",
                    "document_title": "Housing",
                    "section_path": ["Meet our team"],
                    "text": "The student support team offers wellbeing and accessibility help.",
                },
            ]
        },
        domain="university",
    )
    units = snapshot.evidence_map()
    first = units["chunk:student-life-en"]
    localized = units["chunk:student-life-ar"]
    distinct_page = units["chunk:housing"]

    candidates = deduplicated_candidates(
        (localized, distinct_page),
        (first,),
        snapshot.source_map(),
        allow_cross_source_duplicates=True,
    )

    assert localized not in candidates
    assert distinct_page in candidates


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
