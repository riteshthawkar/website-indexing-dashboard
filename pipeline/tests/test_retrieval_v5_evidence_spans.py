from __future__ import annotations

from pipeline.evaluation.dataset import EvalExample
from pipeline.evaluation.retrieval_eval import _score_query
from pipeline.retrieval.evidence_packer import build_evidence_pack
from pipeline.stages.formatters.gemini_retrieval_formatter import (
    _assertions_from_promoted_graph,
    _build_span_embedding_text,
    _classify_span_type,
    _sentence_spans,
    _stable_id,
)


def test_evidence_span_helpers_are_extractive_and_stable() -> None:
    text = (
        "Applications close on February 28, 2025. "
        "The fully funded program includes flights, accommodation, insurance, and a stipend. "
        "Students work with faculty on AI research."
    )

    spans = _sentence_spans(text, max_sentences=2, max_chars=220)
    assert spans
    assert spans[0].startswith("Applications close")
    assert _classify_span_type(spans[0], document_title="UGRIP", section_path=["Apply"], heading="Deadline") == "deadline"

    span_record = {
        "document_title": "UGRIP",
        "breadcrumb": "Study > UGRIP",
        "section_heading": "Application deadline",
        "canonical_url": "https://mbzuai.ac.ae/ugrip-campaign",
        "text": spans[0],
    }
    embedding_text = _build_span_embedding_text(span_record)
    assert "TITLE: UGRIP" in embedding_text
    assert spans[0] in embedding_text

    assert _stable_id("evidence_span", "url", "section", "chunk", spans[0]) == _stable_id(
        "evidence_span",
        "url",
        "section",
        "chunk",
        spans[0],
    )


def test_evidence_packer_orders_spans_before_chunks_and_reports_coverage() -> None:
    result = {
        "answer_documents": [
            {
                "id": "assertion-1",
                "text": "UGRIP applications close on February 28, 2025.",
                "source_url": "https://mbzuai.ac.ae/ugrip-campaign",
                "document_title": "UGRIP",
                "source_span_ids": ["span-1"],
            }
        ],
        "fact_documents": [
            {
                "id": "fact-1",
                "text": "UGRIP is a fully funded four-week AI research internship.",
                "source_url": "https://mbzuai.ac.ae/ugrip-campaign",
                "document_title": "UGRIP",
            }
        ],
        "evidence_span_documents": [
            {
                "id": "span-1",
                "text": "The program includes return flights, accommodation, health insurance, and a stipend.",
                "source_url": "https://mbzuai.ac.ae/ugrip-campaign",
                "document_title": "UGRIP",
                "section_heading": "Benefits",
                "span_type": "program",
            }
        ],
        "retrieval_documents": [
            {
                "id": "chunk-1",
                "text": "Longer chunk context about UGRIP benefits and deadlines.",
                "source_url": "https://mbzuai.ac.ae/ugrip-campaign",
                "document_title": "UGRIP",
            }
        ],
    }

    pack = build_evidence_pack(
        query="What does UGRIP offer and when is the deadline?",
        result=result,
        max_items=8,
        max_chars=4000,
        max_per_source=4,
        coverage_plan={
            "required_entities": ["February 28, 2025", "return flights"],
            "required_pages": ["https://mbzuai.ac.ae/ugrip-campaign"],
            "required_sections": ["Benefits"],
        },
    )

    assert [item["kind"] for item in pack["items"][:4]] == ["assertion", "fact", "evidence_span", "chunk"]
    assert pack["coverage_status"] == "complete"
    assert pack["citation_candidates"]


def test_retrieval_eval_scores_span_and_coverage_metrics() -> None:
    example = EvalExample(
        id="eval-1",
        query="What does UGRIP offer and when is the deadline?",
        query_type="synthesis",
        source_type="webpage",
        gold_span_ids=["span-1"],
        gold_chunk_ids=["chunk-1"],
        metadata={
            "answer_must_include": ["February 28, 2025", "return flights"],
            "expected_citation_urls": ["https://mbzuai.ac.ae/ugrip-campaign"],
            "min_distinct_sources": 1,
        },
    )
    result = {
        "mode": "synthesis",
        "seed_chunk_ids": ["chunk-1"],
        "selected_chunk_ids": ["chunk-1"],
        "selected_evidence_span_ids": ["span-1"],
        "selected_parent_ids": ["parent-1"],
        "selected_media_ids": [],
        "evidence_span_documents": [
            {
                "id": "span-1",
                "text": "Applications close on February 28, 2025 and include return flights.",
                "source_url": "https://mbzuai.ac.ae/ugrip-campaign",
                "document_title": "UGRIP",
            }
        ],
    }

    score = _score_query(example.normalized(), result)

    assert score.span_hit_at_10 == 1.0
    assert score.span_mrr_at_10 == 1.0
    assert score.required_entity_coverage == 1.0
    assert score.citation_support_rate == 1.0
    assert score.multi_page_coverage_rate == 1.0


def test_promoted_graph_assertions_are_converted_with_span_support() -> None:
    graph_payload = {
        "nodes": [
            {
                "id": "entity:program",
                "node_type": "entity",
                "label": "UGRIP",
                "properties": {"canonical_name": "UGRIP", "entity_type": "program"},
            },
            {
                "id": "entity:deadline",
                "node_type": "entity",
                "label": "February 28, 2025",
                "properties": {"canonical_name": "February 28, 2025", "entity_type": "date"},
            },
            {
                "id": "assertion:deadline",
                "node_type": "relation_assertion",
                "label": "date",
                "properties": {
                    "relation_type": "date",
                    "subject_entity_id": "entity:program",
                    "object_entity_id": "entity:deadline",
                    "subject_name": "UGRIP",
                    "object_name": "February 28, 2025",
                    "source_chunk_ids": ["chunk-1"],
                    "source_url": "https://mbzuai.ac.ae/ugrip-campaign",
                    "document_title": "UGRIP",
                    "validity_status": "active",
                },
            },
        ],
        "edges": [
            {
                "edge_type": "ASSERTION_SUPPORTED_BY_SPAN",
                "source_id": "assertion:deadline",
                "target_id": "span-1",
            }
        ],
    }

    assertions = _assertions_from_promoted_graph(graph_payload)

    assert len(assertions) == 1
    assert assertions[0]["id"] == "assertion:deadline"
    assert assertions[0]["source_span_ids"] == ["span-1"]
    assert assertions[0]["source_chunk_ids"] == ["chunk-1"]
