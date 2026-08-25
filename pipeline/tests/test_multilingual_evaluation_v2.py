from __future__ import annotations

import json

from pipeline.evaluation.dataset import EvalExample, load_eval_examples, write_eval_examples
from pipeline.evaluation.dataset_tools import summarize_eval_examples, validate_eval_examples
from pipeline.evaluation.multilingual_v2 import (
    assign_stratified_splits,
    contains_evidence_quote,
    looks_like_language,
    validate_multilingual_coverage,
    validate_source_ids,
)


def test_eval_example_round_trips_source_level_gold_ids(tmp_path):
    path = tmp_path / "source-grounded.jsonl"
    expected = EvalExample(
        id="ar-navigation-001",
        query="كيف يمكنني التقديم إلى البرنامج؟",
        query_type="scoped",
        language="Arabic",
        source_type="webpage",
        reference_answer="استخدم رابط التقديم الرسمي الموضح في صفحة البرنامج.",
        gold_document_revision_ids=["document-revision:1"],
        gold_page_card_ids=["page-card:1"],
        gold_section_ids=["page-section:1"],
        gold_action_ids=["page-action:1"],
        metadata={"expected_reference_urls": ["https://mbzuai.ac.ae/ar/study"]},
    )

    write_eval_examples(path, [expected])
    actual = load_eval_examples(path)[0]

    assert actual.gold_document_revision_ids == ["document-revision:1"]
    assert actual.gold_page_card_ids == ["page-card:1"]
    assert actual.gold_section_ids == ["page-section:1"]
    assert actual.gold_action_ids == ["page-action:1"]


def test_source_level_gold_is_valid_without_candidate_chunk_ids(tmp_path):
    path = tmp_path / "source-grounded.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "en-page-001",
                "query": "Where can I find the application page?",
                "query_type": "fact",
                "language": "English",
                "source_type": "webpage",
                "reference_answer": "Use the official application action on the admissions page.",
                "gold_page_card_ids": ["page-card:1"],
                "gold_action_ids": ["page-action:1"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_eval_examples(path)

    assert report["ok"] is True
    assert report["summary"]["with_gold_page_cards"] == 1
    assert report["summary"]["with_gold_actions"] == 1


def test_no_answer_warns_when_it_contains_source_level_gold(tmp_path):
    path = tmp_path / "bad-no-answer.jsonl"
    write_eval_examples(
        path,
        [
            EvalExample(
                id="no-answer-001",
                query="What is the unsupported policy?",
                query_type="fact",
                language="English",
                source_type="none",
                no_answer=True,
                reference_answer="The frozen corpus does not contain enough evidence.",
                gold_document_revision_ids=["document-revision:1"],
            )
        ],
    )

    report = validate_eval_examples(path)

    assert any(item["field"] == "no_answer" for item in report["warnings"])


def test_source_level_coverage_is_included_in_summary():
    summary = summarize_eval_examples(
        [
            EvalExample(
                id="coverage-001",
                query="What does the page say?",
                query_type="fact",
                gold_document_revision_ids=["document-revision:1"],
                gold_page_card_ids=["page-card:1"],
                gold_section_ids=["page-section:1"],
                gold_action_ids=["page-action:1"],
            )
        ]
    )

    assert summary["with_gold_documents"] == 1
    assert summary["with_gold_page_cards"] == 1
    assert summary["with_gold_sections"] == 1
    assert summary["with_gold_actions"] == 1


def test_evidence_quote_matching_normalizes_unicode_and_whitespace():
    assert contains_evidence_quote("الذكاءُ  الاصطناعي\nفي أبوظبي", "الذكاءُ الاصطناعي في أبوظبي")
    assert not contains_evidence_quote("Grounded source text", "invented claim")


def test_source_id_validator_checks_representation_and_media_ids():
    row = EvalExample(
        id="grounded-001",
        query="What is shown?",
        query_type="multimodal",
        language="English",
        source_type="image",
        reference_answer="The image shows the campus.",
        gold_document_revision_ids=["document-revision:1"],
        gold_page_card_ids=["page-card:1"],
        gold_section_ids=["page-section:1"],
        gold_action_ids=["page-action:1"],
        gold_media_ids=["media:1"],
        metadata={"evidence_quotes": [{"source_key": "page-card:1", "quote": "campus"}]},
    )
    representation = {
        "documents": [{"document_revision_id": "document-revision:1"}],
        "page_cards": [
            {
                "page_card_id": "page-card:1",
                "sections": [{"section_id": "page-section:1"}],
            }
        ],
        "actions": [{"action_id": "page-action:1"}],
    }
    media = {"items": [{"id": "media:1"}]}

    assert validate_source_ids([row], representation_bundle=representation, media_manifest=media)["ok"]
    missing = row.normalized()
    missing.gold_media_ids = ["media:missing"]
    report = validate_source_ids([missing], representation_bundle=representation, media_manifest=media)
    assert report["ok"] is False
    assert report["errors"][0]["field"] == "gold_media_ids"


def test_source_id_validator_rejects_actions_outside_runtime_navigation_catalog():
    row = EvalExample(
        id="navigation-001",
        query="Where do I apply?",
        query_type="scoped",
        gold_page_card_ids=["page-card:1"],
        gold_action_ids=["page-action:incidental"],
    )
    representation = {
        "documents": [],
        "page_cards": [{"page_card_id": "page-card:1", "sections": []}],
        "actions": [
            {"action_id": "page-action:incidental"},
            {"action_id": "page-action:operational"},
        ],
    }
    navigation_catalog = {
        "actions": [{"action_id": "page-action:operational"}]
    }

    report = validate_source_ids(
        [row],
        representation_bundle=representation,
        media_manifest={"items": []},
        navigation_catalog=navigation_catalog,
    )

    assert report["ok"] is False
    assert report["errors"][0]["field"] == "gold_action_ids"


def test_split_assignment_is_stable_and_stratified():
    rows = [
        EvalExample(
            id=f"en-fact-{index:03d}",
            query=f"What is fact number {index}?",
            query_type="fact",
            language="English",
            source_type="webpage",
            reference_answer="A grounded fact.",
            gold_page_card_ids=[f"page-card:{index}"],
            metadata={"evidence_quotes": [{"source_key": f"page-card:{index}", "quote": "fact"}]},
        )
        for index in range(20)
    ]

    first = assign_stratified_splits(rows)
    second = assign_stratified_splits(list(reversed(rows)))

    assert [(row.id, row.metadata["split"]) for row in first] == [
        (row.id, row.metadata["split"]) for row in second
    ]
    assert {row.metadata["split"] for row in first} == {"selection", "holdout", "regression"}


def test_split_assignment_keeps_source_connected_examples_together():
    rows = [
        EvalExample(
            id=f"en-source-group-{index:03d}",
            query=f"What is grouped fact number {index}?",
            query_type="fact",
            language="English",
            source_type="webpage",
            reference_answer="A grounded fact.",
            gold_page_card_ids=[
                "page-card:shared" if index < 2 else f"page-card:{index}"
            ],
            metadata={
                "evidence_quotes": [
                    {
                        "source_key": (
                            "page-card:shared" if index < 2 else f"page-card:{index}"
                        ),
                        "quote": "fact",
                    }
                ],
                "required_pages": [
                    "https://mbzuai.ac.ae/shared"
                    if index < 2
                    else f"https://mbzuai.ac.ae/page-{index}"
                ],
            },
        )
        for index in range(20)
    ]

    assigned = assign_stratified_splits(rows)
    by_id = {row.id: row.metadata["split"] for row in assigned}

    assert by_id["en-source-group-000"] == by_id["en-source-group-001"]
    assert {row.metadata["split"] for row in assigned} == {
        "selection",
        "holdout",
        "regression",
    }


def test_coverage_validator_rejects_cross_split_source_leakage():
    rows = [
        EvalExample(
            id=f"source-leak-{index}",
            query=f"What is source fact {index}?",
            query_type="fact",
            language="English",
            source_type="webpage",
            reference_answer="A grounded source fact.",
            gold_page_card_ids=["page-card:shared"],
            metadata={
                "split": split,
                "evidence_quotes": [
                    {"source_key": "page-card:shared", "quote": "source fact"}
                ],
            },
        )
        for index, split in enumerate(("selection", "holdout"))
    ]

    report = validate_multilingual_coverage(
        rows,
        minimum_query_count=0,
        minimum_per_language=0,
        minimum_no_answer_per_language=0,
        minimum_cross_lingual=0,
        minimum_navigation=0,
        minimum_multimodal=0,
        minimum_subdomain=0,
        require_all_splits=False,
    )

    assert any(
        "source-connected examples cross evaluation splits" in error
        for error in report["errors"]
    )


def test_coverage_validator_rejects_language_mismatch():
    row = EvalExample(
        id="bad-language",
        query="This query is in English",
        query_type="fact",
        language="Arabic",
        source_type="webpage",
        reference_answer="هذه إجابة عربية.",
        gold_page_card_ids=["page-card:1"],
        metadata={"evidence_quotes": [{"source_key": "page-card:1", "quote": "evidence"}]},
    )

    report = validate_multilingual_coverage(
        [row],
        minimum_query_count=0,
        minimum_per_language=0,
        minimum_no_answer_per_language=0,
        minimum_cross_lingual=0,
        minimum_navigation=0,
        minimum_multimodal=0,
        minimum_subdomain=0,
    )

    assert any("declared language Arabic" in error for error in report["errors"])


def test_arabic_language_check_allows_a_long_english_title_inside_arabic_question():
    assert looks_like_language(
        "من كان المضيف لمحاضرة Average Hazard for Robust Survival Analysis: "
        "From Trial Decisions to Causal Learning؟",
        "Arabic",
    )


def test_arabic_no_answer_reference_is_recognized(tmp_path):
    path = tmp_path / "arabic-no-answer.jsonl"
    write_eval_examples(
        path,
        [
            EvalExample(
                id="ar-no-answer",
                query="ما عنوان الحرم غير الموجود؟",
                query_type="fact",
                language="Arabic",
                source_type="none",
                no_answer=True,
                reference_answer="لا تحتوي المجموعة المجمدة على دليل موثق، لذا ينبغي الامتناع عن اختلاق إجابة.",
            )
        ],
    )

    report = validate_eval_examples(path)

    assert not [warning for warning in report["warnings"] if warning["field"] == "reference_answer"]
