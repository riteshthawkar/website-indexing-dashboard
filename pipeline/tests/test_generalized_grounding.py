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


def test_generalized_coverage_prefers_exact_content_phrase_over_broader_top_hit():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    broad_url = "https://www.example.edu/research/foundation-models"
    exact_url = "https://institute.example.edu/collaborate"
    query = (
        "What kinds of organizations does the institute partner with to co-create "
        "foundation models and explore new frontiers in AI?"
    )
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:broad": {"id": "card:broad", "source_url": broad_url},
            "card:exact": {"id": "card:exact", "source_url": exact_url},
        },
        chunk_map={
            "chunk:broad": {"id": "chunk:broad", "source_url": broad_url},
            "chunk:exact": {"id": "chunk:exact", "source_url": exact_url},
        },
    )
    broad_text = (
        "The Institute of Foundation Models is a center for model science, "
        "scale, and social value."
    )
    exact_text = (
        "The institute partners with academic institutions, research labs, "
        "startups, and enterprise leaders to co-create foundation models and "
        "explore new frontiers in AI."
    )

    def page(source_url, identity, text):
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "search_text": text.casefold(),
            "tokens": set(_tokenize(text)),
        }

    retriever._coverage_page_records = [
        page(broad_url, "Institute of Foundation Models", broad_text),
        page(exact_url, "Collaborate with the Institute", exact_text),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        query,
        "exact_fact",
        {
            "dense_page_card_ids": ["card:broad", "card:exact"],
            "dense_chunk_ids": ["chunk:broad", "chunk:exact"],
            "planner_confidence": 0.82,
            "query_retrieval_expansion": (
                "institute foundation models partner organizations collaboration"
            ),
        },
    )

    assert inferred["required_pages"] == [exact_url]


def test_spa_routes_do_not_become_aliases_from_a_shared_revision():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    requested_url = "https://events.example.edu/talks/target-talk"
    unrelated_url = "https://events.example.edu/talks/another-talk"
    revision_id = "document-revision:hydrated-spa-snapshot"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever._coverage_page_records = [
        {
            "source_url": requested_url,
            "normalized_url": retriever._normalize_source_url(requested_url),
            "document_revision_ids": {revision_id},
            "linked_chunk_ids": {"chunk:shared"},
            "explicit_alias_urls": set(),
        },
        {
            "source_url": unrelated_url,
            "normalized_url": retriever._normalize_source_url(unrelated_url),
            "document_revision_ids": {revision_id},
            "linked_chunk_ids": {"chunk:shared"},
            "explicit_alias_urls": set(),
        },
    ]
    retriever._coverage_page_records_by_url = {
        page["normalized_url"]: page for page in retriever._coverage_page_records
    }
    unrelated_record = {
        "id": "chunk:shared",
        "source_url": unrelated_url,
        "document_revision_id": revision_id,
    }

    assert not retriever._coverage_pages_share_representation(
        requested_url,
        unrelated_url,
    )
    assert not retriever._record_matches_required_page(
        unrelated_record,
        requested_url,
    )
    assert retriever._record_matches_required_page(
        {**unrelated_record, "canonical_url": requested_url},
        requested_url,
    )


def test_cross_lingual_page_binding_bridges_a_dense_aggregate_parent():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    target_url = "https://events.example.edu/press/teacher-avatar-project"
    aggregate_url = "https://events.example.edu/press"
    broader_url = "https://events.example.edu/articles/realistic-avatars"
    target_title = "Project to develop 3D avatars of teachers and students for virtual classes"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:broader": {"id": "card:broader", "source_url": broader_url},
            "card:target": {"id": "card:target", "source_url": target_url},
        },
        chunk_map={
            "chunk:aggregate": {
                "id": "chunk:aggregate",
                "source_url": aggregate_url,
                "text": f"SECTION: {target_title}. The project enhances virtual classrooms.",
            },
            "chunk:broader": {
                "id": "chunk:broader",
                "source_url": broader_url,
                "text": "Research on generating realistic avatars in virtual worlds.",
            },
        },
    )

    def page(source_url, identity, text):
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "search_text": text.casefold(),
            "tokens": set(_tokenize(text)),
            "document_revision_ids": set(),
            "linked_chunk_ids": set(),
            "explicit_alias_urls": set(),
        }

    retriever._coverage_page_records = [
        page(broader_url, "Generating realistic avatars", "Avatar research"),
        page(target_url, target_title, target_title),
    ]
    retriever._coverage_page_records_by_url = {
        item["normalized_url"]: item for item in retriever._coverage_page_records
    }
    query = (
        "ما المعلومة الأساسية في مشروع تطوير صور رمزية ثلاثية الأبعاد "
        "للمعلمين والطلاب من أجل الفصول الافتراضية؟"
    )

    inferred = retriever._infer_generalized_coverage_requirements(
        query,
        "exact_fact",
        {
            "dense_page_card_ids": ["card:broader", "card:target"],
            "dense_chunk_ids": ["chunk:aggregate", "chunk:broader"],
        },
    )

    assert inferred["required_pages"] == [target_url]
    assert retriever._record_matches_required_page(
        retriever.vector.chunk_map["chunk:aggregate"],
        target_url,
    )
    assert not retriever._record_matches_required_page(
        {
            "id": "chunk:sibling",
            "source_url": "https://events.example.edu/press/another-project",
            "text": target_title,
        },
        target_url,
    )


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
        page_card_map={
            "card:masters": {
                "id": "card:masters",
                "source_url": masters_url,
            }
        },
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
        {
            "dense_page_card_ids": ["card:masters"],
            "dense_chunk_ids": ["chunk:masters", "chunk:doctoral"],
        },
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


def test_generalized_compound_question_keeps_top_dense_partial_identity_page():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    faq_url = "https://www.example.edu/faq/annual-undergraduate-tuition"
    admissions_url = "https://www.example.edu/admissions/undergraduate"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:faq": {"id": "card:faq", "source_url": faq_url},
        },
        chunk_map={
            "chunk:admissions": {
                "id": "chunk:admissions",
                "source_url": admissions_url,
            },
            "chunk:faq": {"id": "chunk:faq", "source_url": faq_url},
        },
    )
    retriever._coverage_page_records = [
        {
            "source_url": faq_url,
            "normalized_url": faq_url,
            "identity_text": "annual undergraduate tuition scholarship",
            "identity_tokens": {
                "annual",
                "undergraduate",
                "tuition",
                "scholarship",
            },
            "tokens": {
                "annual",
                "undergraduate",
                "tuition",
                "scholarship",
            },
        },
        {
            "source_url": admissions_url,
            "normalized_url": admissions_url,
            "identity_text": "undergraduate admissions",
            "identity_tokens": {"undergraduate", "admissions"},
            "tokens": {
                "undergraduate",
                "tuition",
                "scholarship",
                "available",
            },
        },
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "How much is undergraduate tuition per year, and what scholarships are available?",
        "multi_page_aggregation",
        {
            "planner_confidence": 0.75,
            "query_retrieval_expansion": (
                "undergraduate tuition per year scholarships"
            ),
            "dense_page_card_ids": ["card:faq"],
            "dense_chunk_ids": ["chunk:admissions", "chunk:faq"],
            "retrieval_documents": [
                {
                    "id": "chunk:admissions",
                    "source_url": admissions_url,
                }
            ],
        },
    )

    assert set(inferred["required_pages"]) == {faq_url, admissions_url}


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


def test_production_graph_context_does_not_inject_query_specific_aliases(monkeypatch):
    import pipeline.retrieval.graph_rag as graph_module
    from pipeline.retrieval.graph_rag import GraphRAGRetriever

    retriever = GraphRAGRetriever.__new__(GraphRAGRetriever)
    retriever.query_specific_retrieval_rules_enabled = False
    retriever._build_relation_query_plan = lambda query, mode, media_query: None
    monkeypatch.setattr(
        graph_module,
        "_semantic_query_alias_tokens",
        lambda _query: (_ for _ in ()).throw(
            AssertionError("query-specific aliases must remain disabled")
        ),
    )

    context = retriever.prepare_query_context("ماذا تعرض لوحة مشاريع الأبحاث؟")

    assert context.rewritten_query == "ماذا تعرض لوحة مشاريع الأبحاث؟"
    assert "semantic_alias_expansion" not in context.rewrite_labels


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
