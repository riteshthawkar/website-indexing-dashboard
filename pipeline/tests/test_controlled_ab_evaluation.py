from __future__ import annotations

from pipeline.evaluation.controlled_ab import (
    aggregate_quality,
    annotate_abstention_decisions,
    annotate_gold_presence,
    choose_abstention_threshold,
    reciprocal_rank_fusion,
    score_ranking,
    tokenize_multilingual,
)
from pipeline.evaluation.dataset import EvalExample


def test_multilingual_tokenizer_normalizes_arabic_diacritics_and_segments_article():
    tokens = tokenize_multilingual("وَالْقَبُولُ في الجامعة")

    assert "والقبول" in tokens
    assert "ar:قبول" in tokens
    assert "الجامعه" in tokens


def test_source_grounded_evidence_scoring_requires_quote_and_source_match():
    example = EvalExample(
        id="grounded",
        query="What is required?",
        query_type="fact",
        gold_page_card_ids=["page-card:gold"],
        metadata={
            "source_keys": ["page-card:gold"],
            "evidence_quotes": [
                {"source_key": "page-card:gold", "quote": "a master's degree is preferred"}
            ],
        },
    )
    wrong_source = {
        "id": "chunk:wrong",
        "page_card_ids": ["page-card:wrong"],
        "raw_text": "A master's degree is preferred.",
    }
    right_source = {
        "id": "chunk:right",
        "page_card_ids": ["page-card:gold"],
        "raw_text": "The notice says a master's degree is preferred.",
    }

    result = score_ranking(example, [(wrong_source, 0.9), (right_source, 0.8)])

    assert result["evidence_hit_at_10"] == 1.0
    assert result["evidence_mrr_at_10"] == 0.5
    assert result["source_recall_at_10"] == 1.0


def test_media_source_key_matches_unprefixed_runtime_media_id():
    example = EvalExample(
        id="media",
        query="What is shown?",
        query_type="multimodal",
        source_type="image",
        gold_media_ids=["abc123"],
        metadata={
            "source_keys": ["media:abc123"],
            "evidence_quotes": [
                {"source_key": "media:abc123", "quote": "visible source wording"}
            ],
        },
    )
    record = {
        "id": "abc123",
        "kind": "media",
        "media_id": "abc123",
        "raw_text": "A semantically equivalent caption.",
    }

    result = score_ranking(example, [(record, 0.9)])

    assert result["evidence_hit_at_10"] == 1.0
    assert result["source_recall_at_10"] == 1.0
    assert result["media_hit_at_5"] == 1.0


def test_aggregate_uses_navigation_and_media_subsets_not_all_queries():
    weights = {
        "evidence_ndcg_at_10": 0.0,
        "evidence_mrr_at_10": 0.0,
        "source_recall_at_10": 0.0,
        "arabic_evidence_hit_at_10": 0.0,
        "synthesis_source_recall_at_10": 0.0,
        "navigation_action_hit_at_5": 0.5,
        "media_hit_at_5": 0.5,
    }
    navigation = annotate_gold_presence(
        {
            "id": "nav",
            "language": "English",
            "query_type": "scoped",
            "no_answer": False,
            "navigation_action_hit_at_5": 1.0,
            "media_hit_at_5": 0.0,
        },
        EvalExample(
            id="nav",
            query="Apply",
            query_type="scoped",
            gold_action_ids=["page-action:1"],
        ),
    )
    media = annotate_gold_presence(
        {
            "id": "media",
            "language": "English",
            "query_type": "multimodal",
            "no_answer": False,
            "navigation_action_hit_at_5": 0.0,
            "media_hit_at_5": 1.0,
        },
        EvalExample(
            id="media",
            query="Image",
            query_type="multimodal",
            gold_media_ids=["media:1"],
        ),
    )

    result = aggregate_quality([navigation, media], weights=weights)

    assert result["navigation_action_hit_at_5"] == 1.0
    assert result["media_hit_at_5"] == 1.0
    assert result["selection_score"] == 1.0


def test_rrf_and_abstention_threshold_are_deterministic():
    assert reciprocal_rank_fusion([[1, 2], [2, 1]], output_k=2) == [
        (1, 1 / 61 + 1 / 62),
        (2, 1 / 61 + 1 / 62),
    ]
    threshold = choose_abstention_threshold(
        [
            {"top_score": 0.9, "no_answer": False},
            {"top_score": 0.8, "no_answer": False},
            {"top_score": 0.2, "no_answer": True},
            {"top_score": 0.1, "no_answer": True},
        ]
    )
    assert 0.2 < threshold["threshold"] < 0.8
    assert threshold["balanced_accuracy"] == 1.0


def test_frozen_abstention_decisions_contribute_balanced_accuracy():
    rows = [
        {"id": "answerable", "top_score": 0.8, "no_answer": False},
        {"id": "unsupported", "top_score": 0.2, "no_answer": True},
    ]
    annotate_abstention_decisions(rows, threshold=0.5)

    quality = aggregate_quality(
        rows, weights={"abstention_balanced_accuracy": 1.0}
    )

    assert quality["abstention_balanced_accuracy"] == 1.0
    assert quality["selection_score"] == 1.0
