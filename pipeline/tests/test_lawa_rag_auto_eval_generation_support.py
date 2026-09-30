"""Normalization tests for generated citations and critic scores."""

from lawa_rag.auto_eval import EvidenceCitation
from lawa_rag.auto_eval._generation_support import (
    normalize_citation_quotes,
    normalize_critic_scores,
)
from pipeline.tests.lawa_rag_auto_eval_support import simple_snapshot


def test_cross_script_quote_is_replaced_with_exact_selected_evidence() -> None:
    snapshot = simple_snapshot()
    evidence = snapshot.evidence_map()
    citation = EvidenceCitation(
        "evidence:library-hours",
        "المكتبة مفتوحة من الثامنة صباحًا حتى الثامنة مساءً.",
    )

    normalized, repaired = normalize_citation_quotes((citation,), evidence)

    assert normalized[0].quote == evidence[citation.evidence_id].text
    assert repaired == (citation.evidence_id,)


def test_five_point_critic_scores_are_normalized_to_unit_interval() -> None:
    payload = {
        "grounding_score": 5,
        "naturalness_score": 4,
        "difficulty_match_score": 3,
        "query_grounding_score": 5,
        "answer_completeness_score": 4.5,
        "task_match_score": 5,
        "persona_fit_score": 4,
        "language_quality_score": 5,
    }

    normalized = normalize_critic_scores(payload)

    assert normalized["grounding_score"] == 1.0
    assert normalized["difficulty_match_score"] == 0.6
    assert normalized["score_scale_normalization"] == "1_to_5_divided_by_5"
