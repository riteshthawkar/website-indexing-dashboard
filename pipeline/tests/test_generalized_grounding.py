from types import SimpleNamespace

from pipeline.retrieval.evidence_packer import build_evidence_pack


def test_evidence_pack_collapses_identical_content_across_spa_routes():
    repeated = (
        "This category-wide FAQ body is repeated under many route URLs and must "
        "not occupy multiple evidence positions for the same query."
    )
    result = {
        "retrieval_documents": [
            {
                "id": f"chunk:faq:{index}",
                "text": repeated,
                "source_url": f"https://www.example.edu/faq/question-{index}",
                "document_title": f"Question {index}",
            }
            for index in range(4)
        ]
        + [
            {
                "id": "chunk:canonical:1",
                "text": (
                    "The canonical academic organization page describes the research "
                    "and undergraduate divisions and their respective responsibilities."
                ),
                "source_url": "https://www.example.edu/research/organization",
                "document_title": "Academic organization",
            }
        ]
    }

    pack = build_evidence_pack(
        query="How is the university organized for research and undergraduate study?",
        result=result,
        max_items=8,
        max_chars=8000,
    )

    repeated_items = [item for item in pack["items"] if item["text"] == repeated]
    assert len(repeated_items) == 1
    assert any("canonical academic organization" in item["text"] for item in pack["items"])


def test_short_enumerations_use_synthesis_but_exact_values_remain_facts():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, classify_query_mode

    assert classify_query_mode("Which divisions are there?") == QueryMode.SYNTHESIS
    assert classify_query_mode("What schools does the university have?") == QueryMode.SYNTHESIS
    assert classify_query_mode("ما هي الأقسام الأكاديمية؟") == QueryMode.SYNTHESIS
    assert (
        classify_query_mode("ما الوحدات الأكاديمية الرئيسية في الجامعة؟")
        == QueryMode.SYNTHESIS
    )
    assert classify_query_mode("What is the admissions email?") == QueryMode.FACT
    assert classify_query_mode("ما البريد الإلكتروني للقبول؟") == QueryMode.FACT
    assert classify_query_mode("ما ساعات عمل الدعم؟") == QueryMode.FACT
    assert (
        classify_query_mode("What are the IT support working hours?")
        == QueryMode.FACT
    )
    assert (
        classify_query_mode(
            "What documents must an international applicant submit?"
        )
        == QueryMode.SYNTHESIS
    )


def test_enumeration_expands_across_sibling_page_sections():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_context_chunks = 8
    retriever.max_parent_chunks = 6
    retriever.same_parent_expand_threshold = 1
    retriever.parent_candidate_top_k = 2
    retriever.chunk_map = {
        "intro": {"id": "intro"},
        "group-a": {"id": "group-a"},
        "group-b": {"id": "group-b"},
        "group-c": {"id": "group-c"},
    }
    retriever.parent_map = {
        "section:intro": {"parent_type": "section"},
        "page:groups": {"parent_type": "page"},
    }

    def parent_ids(chunk_id, *, parent_type=None):
        if parent_type == "section" and chunk_id == "intro":
            return ["section:intro"]
        if parent_type == "page" and chunk_id == "intro":
            return ["page:groups"]
        return []

    retriever._parent_ids_for_chunk = parent_ids
    retriever._rank_parent_child_chunk_ids = (
        lambda _query, parent_id, *, top_k: ["intro"]
        if parent_id == "section:intro"
        else []
    )
    retriever._rank_page_child_chunk_ids_with_section_diversity = (
        lambda _query, parent_id, *, top_k: ["group-a", "group-b", "group-c"]
        if parent_id == "page:groups"
        else []
    )
    retriever._expand_fact = lambda values: list(values)

    expanded = retriever._expand_scoped_or_synthesis(
        ["intro"],
        query="Which research groups are there?",
        mode=QueryMode.SYNTHESIS,
    )

    assert expanded[:4] == ["intro", "group-a", "group-b", "group-c"]


def test_generalized_coverage_uses_semantic_page_card_not_known_url_rules():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    groups_url = "https://www.example.edu/research/groups"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:groups": {
                "id": "card:groups",
                "source_url": groups_url,
            }
        }
    )
    retriever._coverage_page_records = [
        {
            "source_url": groups_url,
            "normalized_url": groups_url,
            "identity_text": "research groups",
            "identity_tokens": {"research", "group"},
            "tokens": {"research", "group", "laboratory", "science"},
        },
        {
            "source_url": "https://www.example.edu/admissions/fees",
            "normalized_url": "https://www.example.edu/admissions/fees",
            "identity_text": "admissions fees",
            "identity_tokens": {"admission", "fee"},
            "tokens": {"admission", "fee", "tuition"},
        },
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "Which research groups are there?",
        "multi_page_aggregation",
        {"dense_page_card_ids": ["card:groups"]},
    )

    assert inferred["required_pages"] == [groups_url]
    assert inferred["required_pages_source"] == "semantic_page_evidence"


def test_generalized_coverage_can_bridge_languages_only_with_independent_evidence():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    units_url = "https://www.example.edu/about/academic-units"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:units": {
                "id": "card:units",
                "source_url": units_url,
            }
        }
    )
    retriever._coverage_page_records = [
        {
            "source_url": units_url,
            "normalized_url": units_url,
            "identity_text": "academic units",
            "identity_tokens": {"academic", "units", "unit"},
            "tokens": {"academic", "units", "unit", "research", "education"},
        }
    ]
    payload = {
        "planner_confidence": 0.8,
        "query_retrieval_expansion": "university academic units",
        "dense_page_card_ids": ["card:units"],
    }

    inferred = retriever._infer_generalized_coverage_requirements(
        "ما الوحدات الأكاديمية الرئيسية في الجامعة؟",
        "multi_page_aggregation",
        payload,
    )
    assert inferred["required_pages"] == [units_url]

    payload["dense_page_card_ids"] = []
    uncorroborated = retriever._infer_generalized_coverage_requirements(
        "ما الوحدات الأكاديمية الرئيسية في الجامعة؟",
        "multi_page_aggregation",
        payload,
    )
    assert uncorroborated["required_pages"] == []


def test_production_mode_does_not_inject_prompt_specific_lexical_aliases():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.query_specific_retrieval_rules_enabled = False

    tokens = retriever._informative_query_tokens("ماذا قالت دانييلا روس؟")

    assert "autonomy" not in tokens
    assert "intelligence" not in tokens


def test_semantic_sufficiency_rejects_one_tangential_item_for_broad_request():
    pack = build_evidence_pack(
        query="Which research groups are there?",
        result={
            "retrieval_confidence": 0.31,
            "retrieval_documents": [
                {
                    "id": "chunk:unrelated",
                    "text": "The university welcomes visitors to its main campus.",
                    "source_url": "https://www.mbzuai.ac.ae/visit",
                }
            ],
        },
        coverage_plan={
            "intent": "multi_page_aggregation",
            "semantic_sufficiency_enabled": True,
            "query_specific_rules_enabled": False,
        },
    )

    assert pack["coverage_status"] == "partial"
    assert "weak_query_evidence_alignment" in pack["sufficiency"]["reasons"]


def test_semantic_sufficiency_accepts_relevant_complete_page_aggregate():
    pack = build_evidence_pack(
        query="Which research divisions are there?",
        result={
            "retrieval_confidence": 0.72,
            "retrieval_documents": [
                {
                    "id": "parent:divisions:page",
                    "text": (
                        "Research divisions include Biological and Life Sciences, "
                        "and Computing and Mathematical Sciences."
                    ),
                    "source_url": "https://www.mbzuai.ac.ae/research/divisions",
                    "coverage_aggregate": True,
                }
            ],
        },
        coverage_plan={
            "intent": "multi_page_aggregation",
            "semantic_sufficiency_enabled": True,
            "query_specific_rules_enabled": False,
            "required_pages": [
                "https://www.mbzuai.ac.ae/research/divisions"
            ],
        },
    )

    assert pack["coverage_status"] == "complete"
    assert pack["sufficiency"]["aggregate_item_count"] == 1
    assert pack["sufficiency"]["trusted_aggregate_item_count"] == 1


def test_semantic_sufficiency_does_not_trust_unbound_page_aggregate():
    pack = build_evidence_pack(
        query="Which academic units are responsible for research and teaching?",
        result={
            "retrieval_confidence": 0.48,
            "retrieval_documents": [
                {
                    "id": "parent:unrelated-news-page",
                    "text": (
                        "The university announced a research and teaching event "
                        "for its wider academic community."
                    ),
                    "source_url": "https://www.mbzuai.ac.ae/news/event",
                    "coverage_aggregate": True,
                }
            ],
        },
        coverage_plan={
            "intent": "multi_page_aggregation",
            "semantic_sufficiency_enabled": True,
            "query_specific_rules_enabled": False,
        },
    )

    assert pack["coverage_status"] == "partial"
    assert (
        "low_confidence_broad_evidence_without_bound_page"
        in pack["sufficiency"]["reasons"]
    )
    assert pack["sufficiency"]["trusted_aggregate_item_count"] == 0
