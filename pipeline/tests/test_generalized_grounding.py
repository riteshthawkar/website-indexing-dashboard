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


def test_generalized_coverage_bridges_languages_from_dense_representation_agreement():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    program_url = "https://www.example.edu/study/applied-artificial-intelligence"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:program": {
                "id": "card:program",
                "source_url": program_url,
            }
        },
        chunk_map={
            "chunk:program": {
                "id": "chunk:program",
                "source_url": program_url,
            }
        },
    )
    retriever._coverage_page_records = [
        {
            "source_url": program_url,
            "normalized_url": program_url,
            "identity_text": "master in applied artificial intelligence",
            "identity_tokens": {
                "master",
                "applied",
                "artificial",
                "intelligence",
            },
            "tokens": {
                "master",
                "applied",
                "artificial",
                "intelligence",
                "part-time",
                "campus",
            },
        }
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "هل البرنامج بدوام جزئي وكم تستغرق مدة إكماله؟",
        "broad_synthesis",
        {
            "dense_page_card_ids": ["card:program"],
            "dense_chunk_ids": ["chunk:program"],
        },
    )

    assert inferred["required_pages"] == [program_url]
    assert inferred["required_pages_source"] == "semantic_page_evidence"


def test_generalized_comparison_retains_each_high_ranked_dense_page():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    masters_url = "https://www.example.edu/study/msc-programs"
    doctoral_url = "https://www.example.edu/study/phd-programs"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={},
        chunk_map={
            "chunk:masters": {
                "id": "chunk:masters",
                "source_url": masters_url,
            },
            "chunk:doctoral": {
                "id": "chunk:doctoral",
                "source_url": doctoral_url,
            },
        },
    )
    retriever._coverage_page_records = [
        {
            "source_url": masters_url,
            "normalized_url": masters_url,
            "identity_text": "msc programs master's programs",
            "identity_tokens": {"msc", "master", "program"},
            "tokens": {"msc", "master", "program", "scholarship", "coverage"},
        },
        {
            "source_url": doctoral_url,
            "normalized_url": doctoral_url,
            "identity_text": "phd programs doctoral programs",
            "identity_tokens": {"phd", "doctoral", "program"},
            "tokens": {"phd", "doctoral", "program", "scholarship", "coverage"},
        },
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "Compare scholarship coverage across M.Sc. and Ph.D. programs.",
        "multi_page_aggregation",
        {"dense_chunk_ids": ["chunk:masters", "chunk:doctoral"]},
    )

    assert set(inferred["required_pages"]) == {masters_url, doctoral_url}


def test_generalized_compound_question_retains_two_complementary_pages():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    tuition_url = "https://www.example.edu/admissions/tuition"
    aid_url = "https://www.example.edu/admissions/scholarships"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:tuition": {"id": "card:tuition", "source_url": tuition_url},
            "card:aid": {"id": "card:aid", "source_url": aid_url},
        },
        chunk_map={
            "chunk:tuition": {"id": "chunk:tuition", "source_url": tuition_url},
            "chunk:aid": {"id": "chunk:aid", "source_url": aid_url},
        },
    )
    retriever._coverage_page_records = [
        {
            "source_url": tuition_url,
            "normalized_url": tuition_url,
            "identity_text": "undergraduate tuition",
            "identity_tokens": {"undergraduate", "tuition"},
            "tokens": {"undergraduate", "tuition", "annual", "cost"},
        },
        {
            "source_url": aid_url,
            "normalized_url": aid_url,
            "identity_text": "undergraduate scholarships",
            "identity_tokens": {"undergraduate", "scholarships"},
            "tokens": {"undergraduate", "scholarships", "merit", "need"},
        },
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "How much is undergraduate tuition, and what scholarships are available?",
        "multi_page_aggregation",
        {
            "dense_page_card_ids": ["card:tuition", "card:aid"],
            "dense_chunk_ids": ["chunk:tuition", "chunk:aid"],
        },
    )

    assert set(inferred["required_pages"]) == {tuition_url, aid_url}


def test_financial_detail_pack_reserves_complete_leaf_and_drops_fragment_facts():
    program_url = "https://www.example.edu/study/applied-ai"
    exact_chunk_id = "chunk:c650:document-revision:program:00014:fees"
    pack = build_evidence_pack(
        query=(
            "What are the application fee, fee-waiver conditions, seat-holding "
            "fee, per-credit tuition, and total tuition?"
        ),
        result={
            "fact_documents": [
                {"id": "fact:marker", "text": "###", "source_url": program_url},
                {"id": "fact:fragment", "text": "Th", "source_url": program_url},
            ],
            "retrieval_documents": [
                {
                    "id": "parent:c650:document-revision:program:page",
                    "text": "General program introduction. " * 300,
                    "source_url": program_url,
                    "coverage_aggregate": True,
                },
                {
                    "id": exact_chunk_id,
                    "text": (
                        "Application fee: AED 200 after screening; waived for a "
                        "screening score of 75% or higher, or reimbursed after "
                        "enrollment. Seat-holding fee: AED 5,000, credited toward "
                        "tuition. Program fee: AED 5,000 per credit. Total: AED 170,000."
                    ),
                    "source_url": program_url,
                    "retrieval_rank": 1,
                    "coverage_dense_evidence": True,
                    "dense_semantic_rank": 0,
                },
            ],
        },
        max_items=4,
        max_chars=3000,
        max_per_source=2,
        coverage_plan={
            "intent": "scoped",
            "required_pages": [program_url],
            "query_specific_rules_enabled": False,
        },
    )

    assert exact_chunk_id in [item["id"] for item in pack["items"]]
    packed_text = " ".join(item["text"] for item in pack["items"])
    assert "AED 200" in packed_text
    assert "75%" in packed_text
    assert "AED 170,000" in packed_text
    assert "###" not in packed_text
    assert not any(item["id"] == "fact:fragment" for item in pack["items"])


def test_generalized_coverage_does_not_bind_unrelated_dense_page_without_identity():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    unrelated_url = "https://www.example.edu/campus/parking"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={},
        chunk_map={
            "chunk:parking": {
                "id": "chunk:parking",
                "source_url": unrelated_url,
            }
        },
    )
    retriever._coverage_page_records = [
        {
            "source_url": unrelated_url,
            "normalized_url": unrelated_url,
            "identity_text": "campus parking",
            "identity_tokens": {"campus", "parking"},
            "tokens": {"campus", "parking", "vehicle"},
        }
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "Compare graduate scholarship programs.",
        "multi_page_aggregation",
        {"dense_chunk_ids": ["chunk:parking"]},
    )

    assert inferred["required_pages"] == []


def test_required_page_chunk_backfill_preserves_dense_semantic_order():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    program_url = "https://www.example.edu/study/program"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        chunk_map={
            "chunk:a": {
                "id": "chunk:a",
                "source_url": program_url,
                "text": "Generic program introduction.",
            },
            "chunk:z": {
                "id": "chunk:z",
                "source_url": program_url,
                "text": "Part-time and completed in two years.",
            },
        },
        _score_text_match=lambda _query, _text: 0.0,
    )
    retriever._coverage_page_records = [
        {
            "source_url": program_url,
            "normalized_url": program_url,
            "document_revision_ids": set(),
            "linked_chunk_ids": {"chunk:a", "chunk:z"},
        }
    ]

    chunks = retriever._best_required_page_chunks(
        "كم تستغرق مدة البرنامج؟",
        program_url,
        limit=1,
        preferred_chunk_ids=["chunk:z", "chunk:a"],
    )

    assert [chunk["id"] for chunk in chunks] == ["chunk:z"]
    assert chunks[0]["coverage_dense_evidence"] is True
    assert chunks[0]["dense_semantic_rank"] == 0


def test_required_page_span_backfill_follows_linked_dense_chunk_order():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    program_url = "https://www.example.edu/study/program"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        evidence_span_map={
            "span:unrelated": {
                "id": "span:unrelated",
                "source_url": program_url,
                "text": "Generic introduction.",
                "linked_chunk_ids": ["chunk:other"],
            },
            "span:duration": {
                "id": "span:duration",
                "source_url": program_url,
                "text": "Part-time and completed in two years.",
                "linked_chunk_ids": ["chunk:duration"],
            },
        },
        _score_text_match=lambda _query, _text: 0.0,
    )
    retriever._coverage_page_records = [
        {
            "source_url": program_url,
            "normalized_url": program_url,
            "document_revision_ids": set(),
            "linked_chunk_ids": {"chunk:other", "chunk:duration"},
        }
    ]

    spans = retriever._best_required_page_spans(
        "كم تستغرق مدة البرنامج؟",
        program_url,
        limit=1,
        preferred_chunk_ids=["chunk:duration", "chunk:other"],
    )

    assert [span["id"] for span in spans] == ["span:duration"]
    assert spans[0]["coverage_dense_evidence"] is True
    assert spans[0]["dense_semantic_rank"] == 0


def test_evidence_pack_keeps_cross_lingual_dense_child_of_bound_page():
    program_url = "https://www.mbzuai.ac.ae/study/program"
    exact_chunk_id = "chunk:program:duration"
    pack = build_evidence_pack(
        query="هل البرنامج بدوام جزئي وكم تستغرق مدة إكماله؟",
        result={
            "retrieval_confidence": 0.60,
            "fact_documents": [
                {
                    "id": "fact:generic",
                    "text": "The program includes elective courses.",
                    "source_url": program_url,
                }
            ],
            "retrieval_documents": [
                {
                    "id": "parent:program:page",
                    "text": "Complete official program page and study plan.",
                    "source_url": program_url,
                    "coverage_aggregate": True,
                },
                {
                    "id": exact_chunk_id,
                    "text": (
                        "The program is part-time, in-person, on campus, and "
                        "typically takes two years to complete."
                    ),
                    "source_url": program_url,
                    "coverage_dense_evidence": True,
                    "dense_semantic_rank": 0,
                },
            ],
        },
        max_items=5,
        max_chars=5000,
        max_per_source=4,
        coverage_plan={
            "intent": "broad_synthesis",
            "required_pages": [program_url],
            "required_entities": [],
            "required_sections": [],
            "query_specific_rules_enabled": False,
        },
    )

    assert exact_chunk_id in [item["id"] for item in pack["items"]]
    assert pack["coverage_status"] == "complete"


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
