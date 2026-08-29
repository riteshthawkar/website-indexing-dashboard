from __future__ import annotations

from types import SimpleNamespace

import pytest

from pipeline.core.io import atomic_write_json
from pipeline.evaluation.dataset import EvalExample
from pipeline.evaluation.retrieval_eval import (
    _aggregate_scores,
    _load_gold_ids_by_url,
    _score_query,
)


def test_eval_retrieval_cli_forwards_governed_splits(monkeypatch, tmp_path) -> None:
    import pipeline.cli as cli

    captured = {}

    def evaluate(**kwargs):
        captured.update(kwargs)
        return {"gates": {"passed": True, "path": "", "failures": []}}

    monkeypatch.setattr(cli, "evaluate_retrieval_dataset", evaluate)
    exit_code = cli.cmd_eval_retrieval(
        SimpleNamespace(
            config="production",
            work_dir=str(tmp_path),
            dataset="eval.jsonl",
            gates=None,
            query_cache=None,
            retrieval_cache=None,
            split=["selection"],
            parallelism=4,
            output=None,
            quiet_progress=True,
            json=True,
        )
    )

    assert exit_code == 0
    assert captured["splits"] == ["selection"]


def test_retrieval_eval_scores_representation_v2_and_navigation(tmp_path) -> None:
    work_dir = tmp_path / "run"
    bundle_dir = work_dir / "stage_outputs" / "format_retrieval"
    bundle_dir.mkdir(parents=True)
    atomic_write_json(
        bundle_dir / "retrieval_bundle.json",
        {
            "chunk_records": [
                {
                    "id": "chunk:c650:document-revision:doc-1:00000:abc",
                    "record_type": "chunk",
                    "document_revision_id": "document-revision:doc-1",
                    "page_card_ids": ["page-card:page-1"],
                    "section_ids": ["page-section:section-1"],
                    "source_url": "https://mbzuai.ac.ae/study/program-1",
                }
            ],
            "parent_records": [],
            "evidence_span_records": [],
            "page_card_records": [
                {
                    "id": "page-card:page-1",
                    "record_type": "page_card",
                    "document_revision_id": "document-revision:doc-1",
                    "page_card_ids": ["page-card:page-1"],
                    "section_ids": ["page-section:section-1"],
                    "raw_text": "Applications close on February 28, 2027.",
                    "source_url": "https://mbzuai.ac.ae/study/program-1",
                }
            ],
            "action_records": [
                {
                    "id": "page-action:apply-1",
                    "record_type": "action",
                    "action_id": "page-action:apply-1",
                    "document_revision_id": "document-revision:doc-1",
                    "page_card_ids": ["page-card:page-1"],
                    "section_ids": ["page-section:section-1"],
                    "raw_text": "Apply to Program 1",
                }
            ],
            "media_records": [
                {
                    "id": "media-1",
                    "record_type": "media",
                    "media_id": "media-1",
                    "document_revision_id": "document-revision:doc-1",
                    "page_card_ids": ["page-card:page-1"],
                    "section_ids": ["page-section:section-1"],
                    "raw_text": "Program 1 application timeline",
                }
            ],
        },
    )
    example = EvalExample(
        id="representation-v2",
        query="How do I apply and when is the deadline?",
        query_type="multimodal",
        source_type="mixed",
        gold_document_revision_ids=["document-revision:doc-1"],
        gold_page_card_ids=["page-card:page-1"],
        gold_section_ids=["page-section:section-1"],
        gold_action_ids=["page-action:apply-1"],
        gold_media_ids=["media-1"],
        metadata={
            "source_keys": ["page-card:page-1"],
            "evidence_quotes": [
                {
                    "source_key": "page-card:page-1",
                    "quote": "Applications close on February 28, 2027.",
                }
            ],
        },
    ).normalized()
    result = {
        "mode": "multimodal",
        "abstained": False,
        "selected_chunk_ids": ["chunk:c650:document-revision:doc-1:00000:abc"],
        "selected_evidence_span_ids": [],
        "selected_parent_ids": [],
        "selected_media_ids": ["media-1"],
        "dense_page_card_ids": ["page-card:page-1"],
        "dense_action_ids": ["page-action:apply-1"],
        "evidence_pack": {
            "items": [
                {
                    "id": "page-card:page-1",
                    "record_type": "page_card",
                    "document_revision_id": "document-revision:doc-1",
                    "page_card_ids": ["page-card:page-1"],
                    "section_ids": ["page-section:section-1"],
                    "text": "Applications close on February 28, 2027.",
                }
            ]
        },
        "backend_latency_ms": 125.0,
        "vector_backend_latency_ms": 80.0,
        "routing_latency_ms": 20.0,
        "graph_augment_latency_ms": 15.0,
        "navigation_plan": {
            "status": "ready",
            "target_page": {
                "page_card_id": "page-card:page-1",
                "document_revision_id": "document-revision:doc-1",
            },
            "steps": [
                {
                    "action_id": "page-action:apply-1",
                    "page_card_id": "page-card:page-1",
                    "document_revision_id": "document-revision:doc-1",
                    "section_id": "page-section:section-1",
                }
            ],
        },
    }

    score = _score_query(
        example,
        result,
        ids_by_url=_load_gold_ids_by_url(work_dir),
    )

    assert score.document_hit_at_10 == 1.0
    assert score.page_card_hit_at_5 == 1.0
    assert score.section_hit_at_10 == 1.0
    assert score.action_hit_at_5 == 1.0
    assert score.navigation_action_hit_at_5 == 1.0
    assert score.navigation_page_hit_at_1 == 1.0
    assert score.exact_media_hit_at_5 == 1.0
    assert score.source_identity_recall_at_10 == 1.0
    assert score.evidence_quote_coverage_at_10 == 1.0
    assert score.abstention_correct == 1.0

    aggregate = _aggregate_scores([score])
    assert aggregate["eligible_document_query_count"] == 1.0
    assert aggregate["eligible_action_query_count"] == 1.0
    assert aggregate["backend_latency_p50_ms"] == 125.0
    assert aggregate["backend_latency_p95_ms"] == 125.0


def test_retrieval_eval_uses_explicit_abstention_and_no_answer_denominator() -> None:
    no_answer = EvalExample(
        id="unsupported-1",
        query="What is the MBZUAI Mars shuttle timetable?",
        query_type="fact",
        source_type="none",
        no_answer=True,
    ).normalized()
    accepted_unsupported = _score_query(
        no_answer,
        {
            "abstained": True,
            # Candidates may still be retained for diagnostics after the
            # retriever has made a grounded abstention decision.
            "selected_chunk_ids": ["irrelevant-candidate"],
        },
    )
    violated_unsupported = _score_query(
        EvalExample(
            id="unsupported-2",
            query="Where is MBZUAI's lunar campus?",
            query_type="fact",
            source_type="none",
            no_answer=True,
        ).normalized(),
        {"abstained": False, "selected_chunk_ids": []},
    )
    answerable = _score_query(
        EvalExample(
            id="answerable",
            query="Where is MBZUAI?",
            query_type="fact",
            source_type="webpage",
        ).normalized(),
        {"abstained": False, "selected_chunk_ids": []},
    )

    assert accepted_unsupported.no_answer_violation == 0.0
    assert accepted_unsupported.unsupported_abstention_rate == 1.0
    assert accepted_unsupported.abstention_correct == 1.0
    assert violated_unsupported.no_answer_violation == 1.0

    aggregate = _aggregate_scores(
        [accepted_unsupported, violated_unsupported, answerable]
    )
    assert aggregate["no_answer_violation_rate"] == pytest.approx(0.5)
    assert aggregate["unsupported_abstain_accuracy"] == pytest.approx(0.5)
    assert aggregate["answerable_accept_accuracy"] == 1.0
    assert aggregate["abstention_balanced_accuracy"] == pytest.approx(0.75)


def test_retrieval_eval_scores_explicit_bridged_representation_identities() -> None:
    example = EvalExample(
        id="bridged-identities",
        query="What does this section say?",
        query_type="scoped",
        source_type="webpage",
        gold_document_revision_ids=["document-revision:1"],
        gold_page_card_ids=["page-card:1"],
        gold_section_ids=["page-section:1"],
    ).normalized()

    score = _score_query(
        example,
        {
            "abstained": False,
            "selected_document_revision_ids": ["document-revision:1"],
            "selected_page_card_ids": ["page-card:1"],
            "selected_section_ids": ["page-section:1"],
            "dense_page_card_ids": ["page-card:1"],
        },
    )

    assert score.document_hit_at_10 == 1.0
    assert score.page_card_hit_at_5 == 1.0
    assert score.section_hit_at_10 == 1.0
