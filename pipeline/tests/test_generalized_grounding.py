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
        classify_query_mode("ما البرامج الأكاديمية التي تقدمها الجامعة؟")
        == QueryMode.SYNTHESIS
    )
    assert (
        classify_query_mode("يرجى تقديم قائمة كاملة بالبرامج والدرجات التي تتيحها الجامعة.")
        == QueryMode.SYNTHESIS
    )
    assert (
        classify_query_mode("ما الوحدات الأكاديمية الرئيسية في الجامعة؟")
        == QueryMode.SYNTHESIS
    )
    assert classify_query_mode("What is the admissions email?") == QueryMode.FACT
    assert (
        classify_query_mode(
            "According to the Human Phenotype Project page, what age range is eligible to participate?"
        )
        == QueryMode.FACT
    )
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


def test_short_direct_fact_questions_use_the_bounded_rerank_path():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)

    assert retriever._is_lightweight_fact_rerank_query(
        "What age range is eligible to participate?",
        mode=QueryMode.FACT,
    )
    assert retriever._is_lightweight_fact_rerank_query(
        "ما البريد الإلكتروني للقبول؟",
        mode=QueryMode.FACT,
    )
    assert not retriever._is_lightweight_fact_rerank_query(
        "Compare all scholarship benefits across every program.",
        mode=QueryMode.FACT,
    )
    assert not retriever._is_lightweight_fact_rerank_query(
        "Which divisions are there?",
        mode=QueryMode.SYNTHESIS,
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


def test_top_dense_consensus_beats_a_same_title_mirrored_route():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    dedicated_url = "https://study.example.edu"
    mirror_url = "https://www.example.edu/human-phenotype-project"
    query = (
        "According to the Human Phenotype Project page, what age range is "
        "eligible to participate?"
    )
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:dedicated": {
                "id": "card:dedicated",
                "source_url": dedicated_url,
            },
            "card:filler": {
                "id": "card:filler",
                "source_url": "https://www.example.edu/research",
            },
            "card:mirror": {"id": "card:mirror", "source_url": mirror_url},
        },
        chunk_map={
            "chunk:dedicated": {
                "id": "chunk:dedicated",
                "source_url": dedicated_url,
            },
            **{
                f"chunk:filler:{index}": {
                    "id": f"chunk:filler:{index}",
                    "source_url": f"https://www.example.edu/filler-{index}",
                }
                for index in range(4)
            },
            "chunk:mirror": {"id": "chunk:mirror", "source_url": mirror_url},
        },
    )

    def page(source_url, text):
        title = "Human Phenotype Project"
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": title.casefold(),
            "identity_tokens": set(_tokenize(title)),
            "search_text": text.casefold(),
            "tokens": set(_tokenize(text)),
            "document_revision_ids": set(),
            "linked_chunk_ids": set(),
            "explicit_alias_urls": set(),
        }

    retriever._coverage_page_records = [
        page(
            dedicated_url,
            "Human Phenotype Project participants aged 18 to 70 are eligible",
        ),
        page(
            mirror_url,
            "Human Phenotype Project overview, milestones, faculty and students",
        ),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        query,
        "exact_fact",
        {
            "dense_page_card_ids": [
                "card:dedicated",
                "card:filler",
                "card:mirror",
            ],
            "dense_chunk_ids": [
                "chunk:dedicated",
                "chunk:filler:0",
                "chunk:filler:1",
                "chunk:filler:2",
                "chunk:filler:3",
                "chunk:mirror",
            ],
        },
    )

    assert inferred["required_pages"] == [dedicated_url]


def test_page_coverage_tokens_include_late_sections_after_phrase_window():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    page_url = "https://long.mbzuai.ac.ae/study"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={},
        evidence_span_map={},
        chunk_map={
            **{
                f"chunk:{index}": {
                    "id": f"chunk:{index}",
                    "source_url": page_url,
                    "text": (f"introductory-section-{index} " * 100),
                }
                for index in range(12)
            },
            "chunk:late": {
                "id": "chunk:late",
                "source_url": page_url,
                "section_heading": "Eligibility",
                "text": "lateeligibilitymarker participant requirements",
            },
        },
        summary_map={},
        parent_map={},
    )

    pages = retriever._build_coverage_page_records()

    assert len(pages) == 1
    assert "lateeligibilitymarker" in pages[0]["tokens"]
    assert "lateeligibilitymarker" not in pages[0]["search_text"]
    assert pages[0]["tail_tokens"] == {"study"}
    assert pages[0]["host_identity_tokens"] == {"long"}


def test_generalized_coverage_computes_query_features_once_per_variant():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    query = "Which marine robotics laboratories are listed?"
    target_url = "https://research.example.edu/marine-robotics"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(page_card_map={}, chunk_map={})

    def page(index):
        source_url = (
            target_url
            if index == 0
            else f"https://research.example.edu/topic-{index}"
        )
        text = (
            "marine robotics laboratories autonomous systems"
            if index == 0
            else f"unrelated research topic {index}"
        )
        identity = "Marine Robotics" if index == 0 else f"Topic {index}"
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "search_text": text,
            "tokens": set(_tokenize(text)),
        }

    retriever._coverage_page_records = [page(index) for index in range(80)]
    original = retriever._generalized_page_query_features
    feature_calls = []

    def counted_features(value):
        feature_calls.append(value)
        return original(value)

    retriever._generalized_page_query_features = counted_features

    inferred = retriever._infer_generalized_coverage_requirements(
        query,
        "multi_page_aggregation",
        {},
    )

    assert inferred["required_pages"] == [target_url]
    assert feature_calls == [query]


def test_coverage_status_refresh_preserves_scope_without_reinference():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    page_url = "https://www.example.edu/research/groups"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever._infer_coverage_requirements = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("coverage scope must not be inferred during status refresh")
    )

    refreshed = retriever._refresh_coverage_plan_status(
        {
            "intent": "multi_page_aggregation",
            "required_pages": [page_url],
            "required_pages_source": "semantic_page_evidence",
        },
        {
            "selected_chunk_ids": ["chunk:groups"],
            "retrieval_documents": [
                {
                    "id": "chunk:groups",
                    "source_url": page_url,
                    "text": "The university has two research groups.",
                }
            ],
        },
    )

    assert refreshed["required_pages"] == [page_url]
    assert refreshed["coverage_status"] == "complete"


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
        "ما المعلومة الأساسية التي تعرضها صفحة مشروع تطوير صور رمزية ثلاثية الأبعاد "
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


def test_generalized_durable_program_query_does_not_bind_to_incidental_news():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    program_url = "https://www.example.edu/faq/what-programs-does-the-university-offer"
    single_program_url = "https://www.example.edu/ar/academics/access/ai-reach"
    masters_programs_url = "https://www.example.edu/study/msc-programs"
    division_programs_url = "https://www.example.edu/research/division-computing"
    news_url = (
        "https://www.example.edu/knowledge-center/the-node/"
        "government-announces-university-partnership"
    )
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:news": {"id": "card:news", "source_url": news_url},
            "card:single-program": {
                "id": "card:single-program",
                "source_url": single_program_url,
            },
            "card:masters-programs": {
                "id": "card:masters-programs",
                "source_url": masters_programs_url,
            },
            "card:division-programs": {
                "id": "card:division-programs",
                "source_url": division_programs_url,
            },
            "card:programs": {
                "id": "card:programs",
                "source_url": program_url,
            },
        },
        chunk_map={
            "chunk:news": {"id": "chunk:news", "source_url": news_url},
            "chunk:programs": {
                "id": "chunk:programs",
                "source_url": program_url,
            },
        },
    )

    def page(source_url, identity, text, page_type, *, arabic):
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "search_text": text.casefold(),
            "tokens": set(_tokenize(text)),
            "page_type": page_type,
            "page_is_arabic": arabic,
        }

    retriever._coverage_page_records = [
        page(
            news_url,
            "جامعة المثال للذكاء الاصطناعي تعلن عن شراكة معرفية استراتيجية",
            "تدعم الشراكة البرامج الأكاديمية والبحث والتعليم",
            "news_or_event",
            arabic=True,
        ),
        page(
            single_program_url,
            "برنامج ريتش للذكاء الاصطناعي",
            "برنامج أكاديمي واحد يقدم فرص البحث والتدريب لطلاب الجامعة",
            "admissions_or_program",
            arabic=True,
        ),
        page(
            masters_programs_url,
            "Master's programs",
            "Master's degree programs offered by the university",
            "admissions_or_program",
            arabic=False,
        ),
        page(
            division_programs_url,
            "Division of Computing and Mathematical Sciences",
            "The division offers several academic programs and degrees.",
            "admissions_or_program",
            arabic=False,
        ),
        page(
            program_url,
            "Academic programs",
            "Undergraduate, master's, and Ph.D. degree programs offered by the university",
            "admissions_or_program",
            arabic=False,
        ),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "ما البرامج الأكاديمية التي تقدمها جامعة المثال للذكاء الاصطناعي؟",
        "multi_page_aggregation",
        {
            # The incidental article is deliberately ranked first in both
            # semantic lanes. Durable source policy must still bind the answer
            # to the maintained program surface.
            "dense_page_card_ids": [
                "card:news",
                "card:single-program",
                "card:masters-programs",
                "card:division-programs",
                "card:programs",
            ],
            "dense_chunk_ids": ["chunk:news", "chunk:programs"],
        },
    )

    assert inferred["required_pages"] == [program_url]
    assert inferred["required_pages_source"] == "semantic_page_evidence"

    paraphrased = retriever._infer_generalized_coverage_requirements(
        "يرجى تقديم قائمة كاملة بالبرامج الأكاديمية والدرجات التي تتيحها جامعة المثال للطلاب.",
        "multi_page_aggregation",
        {
            "dense_page_card_ids": [
                "card:single-program",
                "card:division-programs",
                "card:programs",
            ],
            "dense_chunk_ids": ["chunk:programs"],
        },
    )
    assert paraphrased["required_pages"] == [program_url]


def test_generalized_news_query_can_still_bind_to_news_surface():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    news_url = (
        "https://www.example.edu/knowledge-center/the-node/"
        "government-announces-university-partnership"
    )
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:news": {"id": "card:news", "source_url": news_url},
        },
        chunk_map={
            "chunk:news": {"id": "chunk:news", "source_url": news_url},
        },
    )
    identity = "Latest academic program announcement"
    text = "The university announced its latest academic program partnership in 2026."
    retriever._coverage_page_records = [
        {
            "source_url": news_url,
            "normalized_url": retriever._normalize_source_url(news_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "search_text": text.casefold(),
            "tokens": set(_tokenize(text)),
            "page_type": "news_or_event",
            "page_is_arabic": False,
        }
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "What is the latest 2026 academic program announcement?",
        "scoped",
        {
            "dense_page_card_ids": ["card:news"],
            "dense_chunk_ids": ["chunk:news"],
        },
    )

    assert inferred["required_pages"] == [news_url]

    assert retriever._infer_generalized_coverage_requirements(
        "ماذا أعلنت الجامعة عن أحدث برامجها الأكاديمية؟",
        "scoped",
        {
            "dense_page_card_ids": ["card:news"],
            "dense_chunk_ids": ["chunk:news"],
        },
    )["required_pages"] == [news_url]


def test_cross_script_page_name_mismatch_does_not_veto_dense_source_consensus():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    careers_url = "https://careers.example.edu"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:careers": {
                "id": "card:careers",
                "source_url": careers_url,
            }
        },
        chunk_map={
            "chunk:careers": {
                "id": "chunk:careers",
                "source_url": careers_url,
            }
        },
    )
    identity = "Careers open positions"
    sequence = retriever._generalized_page_token_sequence(identity)
    retriever._coverage_page_records = [
        {
            "source_url": careers_url,
            "normalized_url": careers_url,
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "identity_sequence_text": f" {' '.join(sequence)} ",
            "search_text": "Faculty Research Engineering Vacancies global offices".casefold(),
            "tokens": set(
                _tokenize(
                    "Faculty Research Engineering Vacancies global offices"
                )
            ),
            "host_identity_tokens": {"careers"},
            "tail_tokens": set(),
            "page_is_arabic": False,
        }
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "ما هي أقسام الوظائف المفتوحة في صفحة الوظائف، وما هي المراكز العالمية؟",
        "multi_page_aggregation",
        {
            "dense_page_card_ids": ["card:careers"],
            "dense_chunk_ids": ["chunk:careers"],
        },
    )

    assert inferred["required_pages"] == [careers_url]
    assert inferred["required_pages_source"] == "semantic_page_evidence"


def test_generalized_coverage_never_requires_incidental_author_archives():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    about_url = "https://ifm.example.edu/about/"
    author_url = "https://ifm.example.edu/author/editor/"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    shared_text = (
        "Institute of Foundation Models mission partner model headquarters "
        "global hubs open collaboration"
    )
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:about": {"id": "card:about", "source_url": about_url},
            "card:author": {"id": "card:author", "source_url": author_url},
        },
        chunk_map={
            "chunk:about": {"id": "chunk:about", "source_url": about_url},
            "chunk:author": {"id": "chunk:author", "source_url": author_url},
        },
    )

    def page(source_url: str, title: str) -> dict:
        normalized_url = retriever._normalize_source_url(source_url)
        sequence = retriever._generalized_page_token_sequence(title)
        return {
            "source_url": source_url,
            "normalized_url": normalized_url,
            "identity_text": title.casefold(),
            "identity_tokens": set(_tokenize(title)),
            "identity_sequence_text": f" {' '.join(sequence)} ",
            "search_text": shared_text.casefold(),
            "tokens": set(_tokenize(shared_text)),
            "host_identity_tokens": {"ifm"},
            "tail_tokens": set(_tokenize(source_url)),
            "page_is_arabic": False,
        }

    retriever._coverage_page_records = [
        page(about_url, "IFM About"),
        page(author_url, "IFM author editor"),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "What are IFM's mission, partner model, headquarters, and global hubs?",
        "multi_page_aggregation",
        {
            "dense_page_card_ids": ["card:about", "card:author"],
            "dense_chunk_ids": ["chunk:about", "chunk:author"],
        },
    )

    assert about_url in inferred["required_pages"]
    assert author_url not in inferred["required_pages"]


def test_collection_query_prefers_retrieved_structural_overview_page():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    root_url = "https://careers.example.edu/"
    vacancies_url = "https://careers.example.edu/vacancies/"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:root": {"id": "card:root", "source_url": root_url},
            "card:vacancies": {
                "id": "card:vacancies",
                "source_url": vacancies_url,
            },
        },
        chunk_map={
            "chunk:vacancies": {
                "id": "chunk:vacancies",
                "source_url": vacancies_url,
            }
        },
    )

    def page(source_url: str, title: str, text: str) -> dict:
        normalized_url = retriever._normalize_source_url(source_url)
        sequence = retriever._generalized_page_token_sequence(title)
        return {
            "source_url": source_url,
            "normalized_url": normalized_url,
            "identity_text": title.casefold(),
            "identity_tokens": set(_tokenize(title)),
            "identity_sequence_text": f" {' '.join(sequence)} ",
            "search_text": text.casefold(),
            "tokens": set(_tokenize(text)),
            "host_identity_tokens": {"careers"},
            "tail_tokens": set(_tokenize(source_url)),
            "page_is_arabic": False,
        }

    overview_text = (
        "Careers sections categories faculty research engineering professional "
        "vacancies and global centers locations Abu Dhabi Paris Silicon Valley"
    )
    retriever._coverage_page_records = [
        page(root_url, "Careers", overview_text),
        page(vacancies_url, "All Vacancies", overview_text),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "ما هي أقسام الوظائف المفتوحة في صفحة الوظائف، وما هي المراكز العالمية المذكورة فيها؟",
        "multi_page_aggregation",
        {
            "dense_page_card_ids": ["card:root", "card:vacancies"],
            "dense_chunk_ids": ["chunk:vacancies"],
        },
    )

    assert inferred["required_pages"]
    assert retriever._normalize_source_url(inferred["required_pages"][0]) == (
        retriever._normalize_source_url(root_url)
    )


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


def test_cross_lingual_compound_question_keeps_corroborated_sibling_page_cards():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    overview_url = "https://careers.example.edu/"
    detail_url = "https://careers.example.edu/vacancies"
    unrelated_url = "https://news.example.edu/archive"
    ambiguous_url = "https://careers.example.edu/faculty"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:overview": {"id": "card:overview", "source_url": overview_url},
            "card:detail": {"id": "card:detail", "source_url": detail_url},
            "card:unrelated": {"id": "card:unrelated", "source_url": unrelated_url},
            "card:ambiguous": {"id": "card:ambiguous", "source_url": ambiguous_url},
        },
        chunk_map={
            "chunk:overview": {
                "id": "chunk:overview",
                "source_url": overview_url,
                "text": (
                    "Faculty opportunities across academic department "
                    "appointments. Faculty Vacancies Research Vacancies "
                    "Engineering Vacancies."
                ),
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
        }

    retriever._coverage_page_records = [
        page(
            overview_url,
            "Careers",
            "Faculty Vacancies Research Vacancies Engineering Vacancies",
        ),
        page(
            detail_url,
            "All Vacancies",
            "A global working community in Abu Dhabi Paris and Silicon Valley",
        ),
        page(unrelated_url, "News archive", "Institutional news"),
        page(
            ambiguous_url,
            "Faculty opportunities across academic department appointments",
            "Faculty jobs by academic department",
        ),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        "ما هي أقسام وظائف ACME، وما هي المراكز العالمية المذكورة؟",
        "multi_page_aggregation",
        {
            "dense_page_card_ids": [
                "card:overview",
                "card:detail",
                "card:unrelated",
                "card:ambiguous",
            ],
            "dense_chunk_ids": ["chunk:overview"],
            "planner_confidence": 0.8,
            "query_retrieval_expansion": (
                "ACME أقسام الوظائف jobs page sections three global centers"
            ),
        },
    )

    assert set(inferred["required_pages"]) == {
        overview_url,
        detail_url,
    }


def test_required_page_card_survives_long_spa_chunk_evidence():
    page_url = "https://events.mbzuai.ac.ae/talks/physical-intelligence"
    page_card_id = "page-card:physical-intelligence"
    long_noise = " Unrelated event schedule and speaker biography." * 180
    pack = build_evidence_pack(
        query=(
            "What features and main benefits does the Physical Intelligence "
            "abstract describe?"
        ),
        result={
            "retrieval_documents": [
                {
                    "id": "chunk:physical:00001:first",
                    "text": (
                        "Physical intelligence uses efficient models on robots and sensors."
                        + long_noise
                    ),
                    "source_url": page_url,
                    "coverage_dense_evidence": True,
                    "dense_semantic_rank": 0,
                },
                {
                    "id": "chunk:physical:00002:second",
                    "text": (
                        "Physical systems need adaptive machine intelligence."
                        + long_noise
                    ),
                    "source_url": page_url,
                    "coverage_dense_evidence": True,
                    "dense_semantic_rank": 1,
                },
                {
                    "id": page_card_id,
                    "text": (
                        "Physical AI is compact, adaptive, and embodied, inspired "
                        "by the dynamics of living systems."
                    ),
                    "source_url": page_url,
                    "span_type": "page_card_summary",
                    "coverage_page_card": True,
                },
            ]
        },
        max_items=8,
        max_chars=8000,
        max_per_source=2,
        coverage_plan={
            "intent": "scoped",
            "required_pages": [page_url],
            "query_specific_rules_enabled": False,
            "semantic_sufficiency_enabled": True,
        },
    )

    assert pack["items"][0]["id"] == page_card_id
    assert pack["items"][0]["coverage_page_card"] is True
    assert pack["coverage_status"] == "complete"
    assert len(" ".join(item["text"] for item in pack["items"])) <= 8000


def test_named_site_token_prefers_that_official_host_over_a_mirror():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    def page(source_url):
        identity = "Atlas Institute collaboration"
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "search_text": "partners research labs startups".casefold(),
            "tokens": set(_tokenize("partners research labs startups")),
        }

    named_host = page("https://atlas.example/collaborate")
    mirrored_host = page("https://university.example/research/atlas")
    query = "According to Atlas, which partners collaborate with the institute?"

    assert retriever._generalized_page_target_score(
        query,
        named_host,
    ) > retriever._generalized_page_target_score(query, mirrored_host)


def test_named_acronym_prefers_identity_page_over_incidental_content_mention():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    query = "Where are IFM headquarters and research hubs located?"

    def page(identity, content):
        return {
            "source_url": "https://example.edu/research",
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "search_text": content.casefold(),
            "tokens": set(_tokenize(content)),
            "host_identity_tokens": set(),
            "tail_tokens": set(),
        }

    exact = page("IFM headquarters and research hubs", "Global institute locations")
    umbrella = page("Research", "Our institutes include IFM and other centers")

    assert retriever._generalized_page_target_score(
        query, exact
    ) > retriever._generalized_page_target_score(query, umbrella)


def test_multilingual_bridge_covers_arabic_careers_categories_and_locations():
    from pipeline.retrieval.routed_hybrid import _multilingual_retrieval_bridge_tokens

    aliases = set(
        _multilingual_retrieval_bridge_tokens(
            "ما هي أقسام الوظائف المفتوحة وما هي المراكز المذكورة كبيئة عمل عالمية؟"
        )
    )

    assert {"categories", "careers", "vacancies", "centers", "global"} <= aliases


def test_broad_synthesis_uses_aggregation_evidence_budget():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.evidence_budget_items = 8
    retriever.evidence_budget_chars = 8000
    retriever.aggregation_evidence_budget_items = 12
    retriever.aggregation_evidence_budget_chars = 10000
    retriever.large_page_evidence_budget_items = 10
    retriever.large_page_evidence_budget_chars = 9000
    retriever.evidence_budget_max_per_source = 4

    assert retriever._evidence_budget_for_plan({"intent": "broad_synthesis"}) == (
        12,
        10000,
        4,
    )


def test_explicit_page_scope_beats_content_heavy_mirror_consensus():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    careers_url = "https://careers.example.edu"
    division_url = "https://www.example.edu/research/marine-systems"
    query = (
        "Which Example University careers page section lists vacancies for "
        "the Marine Systems Division?"
    )
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:division": {"id": "card:division", "source_url": division_url},
        },
        chunk_map={
            "chunk:division": {"id": "chunk:division", "source_url": division_url},
            "chunk:careers": {"id": "chunk:careers", "source_url": careers_url},
        },
    )

    def page(source_url, identity, content, host_tokens):
        identity_sequence = retriever._generalized_page_token_sequence(identity)
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "identity_sequence_text": f" {' '.join(identity_sequence)} ",
            "search_text": content.casefold(),
            "tokens": set(_tokenize(content)),
            "host_identity_tokens": set(host_tokens),
            "tail_tokens": set(),
        }

    retriever._coverage_page_records = [
        page(
            division_url,
            "Marine Systems Division",
            "The division has open faculty opportunities and research vacancies.",
            set(),
        ),
        page(
            careers_url,
            "Careers",
            "Faculty vacancies include the Marine Systems Division.",
            {"career", "careers"},
        ),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        query,
        "exact_fact",
        {
            "dense_page_card_ids": ["card:division"],
            "dense_chunk_ids": ["chunk:division", "chunk:careers"],
        },
    )

    assert inferred["required_pages"] == [careers_url]


def test_named_page_acronym_prefers_dedicated_host_over_top_mirror():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    dedicated_url = "https://hpp.example.edu"
    mirror_url = "https://www.example.edu/news/human-phenotype-project"
    query = (
        "On the Human Phenotype Project page, what does the image show and "
        "what does the surrounding text say?"
    )
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_min_confidence = 0.55
    retriever.vector = SimpleNamespace(
        page_card_map={
            "card:mirror": {"id": "card:mirror", "source_url": mirror_url},
            "card:dedicated": {
                "id": "card:dedicated",
                "source_url": dedicated_url,
            },
        },
        chunk_map={
            "chunk:mirror": {"id": "chunk:mirror", "source_url": mirror_url},
            "chunk:dedicated": {
                "id": "chunk:dedicated",
                "source_url": dedicated_url,
            },
        },
    )

    def page(source_url, host_tokens):
        identity = "Human Phenotype Project"
        identity_sequence = retriever._generalized_page_token_sequence(identity)
        return {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
            "identity_sequence_text": f" {' '.join(identity_sequence)} ",
            "search_text": (
                "Human Phenotype Project image and surrounding participant text"
            ).casefold(),
            "tokens": set(
                _tokenize(
                    "Human Phenotype Project image and surrounding participant text"
                )
            ),
            "host_identity_tokens": set(host_tokens),
            "tail_tokens": set(),
        }

    retriever._coverage_page_records = [
        page(mirror_url, set()),
        page(dedicated_url, {"hpp"}),
    ]

    inferred = retriever._infer_generalized_coverage_requirements(
        query,
        "broad_synthesis",
        {
            "dense_page_card_ids": ["card:mirror", "card:dedicated"],
            "dense_chunk_ids": ["chunk:mirror", "chunk:dedicated"],
        },
    )

    assert inferred["required_pages"][0] == dedicated_url


def test_verified_media_infers_generic_named_page_but_not_numbered_pdf_page():
    from pipeline.retrieval.adaptive_hybrid import QueryMode
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    page_url = "https://atlas.example.edu/diagnostics"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever._coverage_intent = lambda _query, _mode: "scoped"
    retriever._explicit_required_page_markers = lambda _query: []
    calls = []

    def infer(query, _intent, _payload=None):
        calls.append(query)
        return {
            "required_entities": [],
            "required_pages": [page_url],
            "required_sections": [],
            "required_pages_source": "semantic_page_evidence",
        }

    retriever._infer_coverage_requirements = infer
    retriever._selected_source_urls = lambda _payload: {
        retriever._normalize_source_url(page_url)
    }

    named_plan = retriever._coverage_plan_for_result(
        query="On the Atlas Diagnostics page, what does the image show?",
        payload={
            "media_evidence_verified": True,
            "selected_media_ids": ["media:atlas"],
            "selected_chunk_ids": ["chunk:atlas"],
        },
        mode=QueryMode.SCOPED,
    )
    numbered_plan = retriever._coverage_plan_for_result(
        query="What does the image on page 9 show?",
        payload={
            "media_evidence_verified": True,
            "selected_media_ids": ["media:page-9"],
            "selected_chunk_ids": ["chunk:page-9"],
        },
        mode=QueryMode.SCOPED,
    )

    assert named_plan["required_pages"] == [page_url]
    assert calls == ["On the Atlas Diagnostics page, what does the image show?"]
    assert numbered_plan["required_pages"] == []
    assert numbered_plan["required_pages_source"] == "verified_media_evidence"


def test_required_page_scope_prioritizes_media_from_that_page():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    page_url = "https://atlas.example.edu/diagnostics"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    payload = {
        "media": [
            {
                "id": "media:broad",
                "source_url": "https://www.example.edu/research-projects",
            },
            {"id": "media:target", "source_url": page_url},
        ],
        "selected_media_ids": ["media:broad", "media:target"],
    }

    retriever._prioritize_required_page_evidence(
        query="On the Atlas Diagnostics page, what does the image show?",
        payload=payload,
        coverage_plan={"required_pages": [page_url]},
    )

    assert [media["id"] for media in payload["media"]] == [
        "media:target",
        "media:broad",
    ]
    assert payload["selected_media_ids"][:2] == [
        "media:target",
        "media:broad",
    ]


def test_explicit_source_language_variant_promotes_second_dense_media_hit():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_media_results = 1
    retriever.chunk_map = {}
    retriever.parent_map = {}
    retriever.parent_ids_by_chunk = {}
    retriever.section_parent_ids_by_chunk = {}
    retriever.page_parent_ids_by_chunk = {}
    retriever.media_map = {
        "english": {
            "id": "english",
            "media_type": "image",
            "source_url": "https://cdn.mbzuai.ac.ae/event-program.pdf",
            "linked_chunk_ids": [],
        },
        "arabic": {
            "id": "arabic",
            "media_type": "image",
            "source_url": "https://cdn.mbzuai.ac.ae/event-program-AR.pdf",
            "linked_chunk_ids": [],
        },
    }
    retriever.media_texts_by_id = {
        "english": "Event identity image page 9 حفل التخرج 2025",
        "arabic": "هوية الحدث في الصفحة 9 حفل التخرج 2025",
    }
    retriever._is_low_signal_media = lambda _media: False
    retriever._score_media_relevance = lambda _query, _media: 1.0

    selected = retriever._attach_media(
        [],
        ["english", "arabic"],
        "في الصفحة 9 من البرنامج العربي، ماذا تُظهر صورة هوية الحدث؟",
        dense_media_hits=["english", "arabic"],
    )

    assert [item["id"] for item in selected] == ["arabic"]


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
                {"id": "fact:stopword-fragment", "text": "From our", "source_url": program_url},
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
    assert not any(
        item["id"] == "fact:stopword-fragment" for item in pack["items"]
    )


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


def test_required_page_chunk_backfill_prioritizes_cross_lingual_facet_coverage():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    program_url = "https://www.example.edu/study/program"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        chunk_map={
            "chunk:generic": {
                "id": "chunk:generic",
                "source_url": program_url,
                "text": "General graduate admissions information.",
            },
            "chunk:interview": {
                "id": "chunk:interview",
                "source_url": program_url,
                "text": "A subset of applicants may receive a technical interview.",
            },
            "chunk:fee": {
                "id": "chunk:fee",
                "source_url": program_url,
                "text": "The application fee is AED 200.",
            },
        },
        _score_text_match=lambda _query, _text: 0.0,
    )
    retriever._coverage_page_records = [
        {
            "source_url": program_url,
            "normalized_url": program_url,
            "document_revision_ids": set(),
            "linked_chunk_ids": {
                "chunk:generic",
                "chunk:interview",
                "chunk:fee",
            },
        }
    ]

    chunks = retriever._best_required_page_chunks(
        "ما المقابلة والرسوم المطلوبة؟",
        program_url,
        limit=2,
        preferred_chunk_ids=["chunk:generic"],
        required_facets=[
            {
                "name": "interview",
                "aliases": ["technical interview", "مقابلة"],
                "min_alias_matches": 1,
            },
            {
                "name": "fee",
                "aliases": ["application fee", "رسوم"],
                "min_alias_matches": 1,
            },
        ],
    )

    assert {chunk["id"] for chunk in chunks} == {
        "chunk:interview",
        "chunk:fee",
    }


def test_required_page_mapping_backfill_reserves_each_relation_bearing_unit_section():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    page_url = "https://www.example.edu/research/divisions"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        chunk_map={
            "chunk:cards": {
                "id": "chunk:cards",
                "source_url": page_url,
                "text": "SECTION: Meet our deans\nDean profiles and biographies.",
            },
            "chunk:biology": {
                "id": "chunk:biology",
                "source_url": page_url,
                "text": (
                    "SECTION: Division of Biological Sciences\n"
                    "## Division of Biological Sciences\nLed by Dean Amina Noor."
                ),
            },
            "chunk:computing": {
                "id": "chunk:computing",
                "source_url": page_url,
                "text": (
                    "SECTION: Division of Computing Sciences\n"
                    "## Division of Computing Sciences\nLed by Dean Ben Chen."
                ),
            },
            "chunk:undergraduate": {
                "id": "chunk:undergraduate",
                "source_url": page_url,
                "text": (
                    "SECTION: Division of Undergraduate Studies\n"
                    "## Division of Undergraduate Studies\nLed by Dean Carla Diaz."
                ),
            },
        },
        _score_text_match=lambda _query, text: 1.0 if "Meet our deans" in text else 0.1,
    )
    retriever._coverage_page_records = [
        {
            "source_url": page_url,
            "normalized_url": page_url,
            "document_revision_ids": set(),
            "linked_chunk_ids": {
                "chunk:cards",
                "chunk:biology",
                "chunk:computing",
                "chunk:undergraduate",
            },
        }
    ]

    chunks = retriever._best_required_page_chunks(
        "Who are the three deans and which division does each lead?",
        page_url,
        limit=3,
        preferred_chunk_ids=["chunk:cards"],
        required_facets=[
            {
                "name": "complete person-to-division mappings",
                "aliases": ["dean", "led by", "division of"],
                "min_alias_matches": 3,
            }
        ],
    )

    assert {chunk["id"] for chunk in chunks} == {
        "chunk:biology",
        "chunk:computing",
        "chunk:undergraduate",
    }


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


def test_semantic_sufficiency_accepts_trusted_cross_script_dense_page_binding():
    program_url = "https://www.mbzuai.ac.ae/faq/programs"
    query = "أعطني قائمة البرامج الأكاديمية التي توفرها الجامعة."
    pack = build_evidence_pack(
        query=query,
        result={
            "query_rewritten": f"{query} programs disciplines",
            "query_rewrite_labels": ["multilingual_semantic_bridge"],
            "retrieval_confidence": 0.0,
            "evidence_span_documents": [
                {
                    "id": "span:undergraduate",
                    "text": "The university offers an undergraduate artificial intelligence degree.",
                    "source_url": program_url,
                    "coverage_dense_evidence": True,
                    "dense_semantic_rank": 2,
                },
                {
                    "id": "span:masters",
                    "text": "Master of Science programs include machine learning and robotics.",
                    "source_url": program_url,
                    "coverage_dense_evidence": True,
                    "dense_semantic_rank": 2,
                },
                {
                    "id": "span:doctoral",
                    "text": "Doctoral programs include computer vision and computer science.",
                    "source_url": program_url,
                    "coverage_dense_evidence": True,
                    "dense_semantic_rank": 2,
                },
            ],
        },
        max_items=6,
        max_chars=5000,
        max_per_source=4,
        coverage_plan={
            "intent": "multi_page_aggregation",
            "required_pages": [program_url],
            "semantic_sufficiency_enabled": True,
            "query_specific_rules_enabled": False,
        },
    )

    assert pack["coverage_status"] == "complete"
    assert pack["sufficiency"]["query_alignment_source"] == (
        "multilingual_semantic_bridge"
    )
    assert pack["sufficiency"]["all_required_pages_have_dense_evidence"] is True
    assert pack["sufficiency"]["reasons"] == []


def test_semantic_sufficiency_rejects_unlabelled_cross_script_rewrite():
    program_url = "https://www.example.edu/faq/programs"
    query = "أعطني قائمة البرامج الأكاديمية التي توفرها الجامعة."
    pack = build_evidence_pack(
        query=query,
        result={
            "query_rewritten": f"{query} programs disciplines",
            "query_rewrite_labels": [],
            "retrieval_confidence": 0.0,
            "retrieval_documents": [
                {
                    "id": "chunk:generic",
                    "text": "Programs and disciplines.",
                    "source_url": program_url,
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
    assert pack["sufficiency"]["query_alignment_source"] == "original"
    assert "weak_query_evidence_alignment" in pack["sufficiency"]["reasons"]


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


def test_semantic_sufficiency_accepts_complete_facets_from_bound_official_page():
    source_url = "https://www.mbzuai.ac.ae/study/masters"
    pack = build_evidence_pack(
        query="What scholarship support is provided and which programs are excluded?",
        result={
            "retrieval_confidence": 0.0,
            "retrieval_documents": [
                {
                    "id": "chunk:scholarship-scope",
                    "text": (
                        "All eligible programs receive a full scholarship with "
                        "tuition and a monthly stipend. The applied program is excluded."
                    ),
                    "source_url": source_url,
                }
            ],
        },
        coverage_plan={
            "intent": "broad_synthesis",
            "semantic_sufficiency_enabled": True,
            "query_specific_rules_enabled": False,
            "required_pages": [source_url],
            "required_facets": [
                {
                    "name": "scholarship scope",
                    "aliases": ["scholarship", "eligible", "excluded"],
                    "min_alias_matches": 3,
                    "min_sources": 1,
                },
                {
                    "name": "benefits",
                    "aliases": ["tuition", "monthly stipend"],
                    "min_alias_matches": 2,
                    "min_sources": 1,
                },
            ],
        },
    )

    assert pack["coverage_status"] == "complete"
    assert pack["sufficiency"]["all_required_pages_have_evidence"] is True
    assert pack["sufficiency"]["complete_required_facet_count"] == 2
    assert pack["sufficiency"]["trusted_faceted_evidence"] is True


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
