from pipeline.core.admissions_routing import (
    admissions_information_audience,
    admissions_surface_preference,
)
from pipeline.retrieval.evidence_packer import build_evidence_pack
from pipeline.retrieval.routed_hybrid import (
    RoutedHybridRetriever,
    _durable_page_candidate_allowed,
    _multilingual_retrieval_bridge_tokens,
    _required_evidence_facets,
)


def test_arabic_admissions_query_gets_general_cross_script_vocabulary():
    query = "ما الوثائق والمتطلبات المطلوبة للتقديم إلى برامج الماجستير؟"

    aliases = set(_multilingual_retrieval_bridge_tokens(query))

    assert {"admissions", "application", "requirements", "documents", "masters"} <= aliases
    assert admissions_information_audience(query) == "masters"
    assert {
        "academic eligibility",
        "English-language proficiency",
        "application documents",
        "references",
        "screening",
        "interview",
    } <= {facet["name"] for facet in _required_evidence_facets(query)}


def test_arabic_admissions_query_rejects_a_faculty_biography_false_positive():
    query = "ما الوثائق والمتطلبات المطلوبة للتقديم إلى برامج الماجستير؟"
    faculty_page = {
        "source_url": "https://example.edu/ar/faculty-directory/researcher",
        "normalized_url": "https://example.edu/ar/faculty-directory/researcher",
        "page_type": "content",
        "identity_text": "Researcher biography",
        "search_text": (
            "She earned a master's degree and studies applications of machine learning. "
            "Her education includes a graduate transcript."
        ),
        "tokens": {"master", "application", "graduate", "transcript"},
    }

    assert not _durable_page_candidate_allowed(query, faculty_page)


def test_durable_admissions_surface_beats_incidental_news_mention():
    query = "What documents and requirements are needed for master's admission?"

    canonical = admissions_surface_preference(
        query,
        source_url="https://example.edu/graduate-masters-admissions",
        title="Master's admissions",
        page_type="admissions_or_program",
    )
    news = admissions_surface_preference(
        query,
        source_url="https://example.edu/knowledge-center/the-node/new-cohort",
        title="University welcomes a new cohort",
        page_type="news_or_event",
    )

    assert canonical > 0
    assert news < 0


def test_complete_admissions_request_creates_an_evidence_contract_for_every_aspect():
    query = (
        "Give me the complete admission requirements for the M.Sc. in Machine Learning, "
        "including academic eligibility, English, GRE, references, screening, and interview."
    )

    contract = _required_evidence_facets(query)
    facets = {facet["name"] for facet in contract}

    assert {
        "academic eligibility",
        "English-language proficiency",
        "standardized-test policy",
        "references",
        "screening",
        "interview",
    } <= facets
    coherence = next(
        facet for facet in contract if facet["name"] == "coherent admissions criteria"
    )
    assert coherence["report_in_answer"] is False


def test_program_format_question_requires_mode_delivery_and_bounded_duration():
    query = (
        "Is the master's program full-time or part-time, how is it delivered, "
        "and how long does it typically take?"
    )

    contract = _required_evidence_facets(query)
    names = {facet["name"] for facet in contract}

    assert {
        "study mode",
        "delivery method",
        "completion duration and any source-stated bounds",
    } <= names
    duration = next(
        facet
        for facet in contract
        if facet["name"] == "completion duration and any source-stated bounds"
    )
    assert duration["min_alias_matches"] == 2
    assert {"typical time to completion", "maximum", "must complete"} <= set(
        duration["aliases"]
    )


def test_detailed_facet_contract_scales_evidence_budget_within_aggregation_cap():
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.evidence_budget_items = 8
    retriever.evidence_budget_chars = 8000
    retriever.evidence_budget_max_per_source = 2
    retriever.aggregation_evidence_budget_items = 12
    retriever.aggregation_evidence_budget_chars = 10000
    retriever.large_page_evidence_budget_items = 12
    retriever.large_page_evidence_budget_chars = 10000
    facets = [
        {"name": f"facet-{index}", "aliases": [f"value-{index}"]}
        for index in range(10)
    ]

    budget = retriever._evidence_budget_for_plan(
        {"intent": "broad_synthesis", "required_facets": facets}
    )

    assert budget == (12, 10000, 2)


def test_detailed_pack_prioritizes_all_facets_before_generic_page_fill():
    source_url = "https://www.mbzuai.ac.ae/study/program"
    facets = [
        {
            "name": name,
            "aliases": [alias],
            "min_alias_matches": 1,
            "min_sources": 1,
        }
        for name, alias in (
            ("degree", "bachelor degree"),
            ("English", "IELTS"),
            ("documents", "transcript"),
            ("references", "two referees"),
            ("screening", "screening exam"),
            ("interview", "technical interview"),
            ("fees", "application fee"),
        )
    ]
    generic_docs = [
        {
            "id": f"chunk:generic:{index:05d}:value",
            "text": (
                "Detailed admission requirements and application process for "
                "the program."
            ),
            "source_url": source_url,
        }
        for index in range(4)
    ]
    facet_docs = [
        {
            "id": f"chunk:facet:{index:05d}:value",
            "text": f"Official requirement: {facet['aliases'][0]}.",
            "source_url": source_url,
        }
        for index, facet in enumerate(facets)
    ]

    pack = build_evidence_pack(
        query=(
            "Give me detailed admission requirements including degree, English, "
            "documents, references, screening, interview, and fees."
        ),
        result={
            "retrieval_confidence": 0.6,
            "retrieval_documents": [*generic_docs, *facet_docs],
        },
        max_items=9,
        max_chars=9000,
        max_per_source=2,
        coverage_plan={
            "intent": "broad_synthesis",
            "required_pages": [source_url],
            "required_facets": facets,
            "semantic_sufficiency_enabled": True,
            "query_specific_rules_enabled": False,
        },
    )

    assert pack["coverage_status"] == "complete"
    assert pack["missing_required_facets"] == []
    assert any("technical interview" in item["text"] for item in pack["items"])


def test_scholarship_coverage_is_partial_until_all_requested_benefits_are_supported():
    query = "What scholarships are available for master's students, and what does the scholarship cover?"
    coverage_plan = {
        "intent": "broad_synthesis",
        "required_facets": _required_evidence_facets(query),
        "semantic_sufficiency_enabled": False,
        "query_specific_rules_enabled": False,
    }
    partial = build_evidence_pack(
        query=query,
        result={
            "retrieval_documents": [
                {
                    "id": "chunk:news:00001:value",
                    "text": "The university offers a scholarship to selected master's students.",
                    "source_url": "https://example.edu/news/new-cohort",
                    "document_title": "University welcomes new cohort",
                }
            ]
        },
        coverage_plan=coverage_plan,
    )

    assert partial["coverage_status"] == "partial"
    assert {
        "tuition coverage",
        "living stipend",
        "accommodation support",
        "health coverage",
        "visa support",
    } <= set(partial["missing_required_facets"])

    complete = build_evidence_pack(
        query=query,
        result={
            "retrieval_documents": [
                {
                    "id": "chunk:program:00001:value",
                    "text": (
                        "All eligible M.Sc. programs offer a full scholarship including tuition, "
                        "accommodation, healthcare, student visa, and a generous monthly stipend."
                    ),
                    "source_url": "https://example.edu/study/msc-programs",
                    "document_title": "Master's programs",
                }
            ]
        },
        coverage_plan=coverage_plan,
    )

    assert complete["coverage_status"] == "complete"
    assert complete["missing_required_facets"] == []
    assert next(
        facet
        for facet in complete["facet_coverage"]
        if facet["name"] == "coherent scholarship coverage"
    )["report_in_answer"] is False


def test_multi_part_fee_question_requires_every_explicit_fee_facet():
    query = (
        "What are the MAAI application fee, fee-waiver conditions, "
        "seat-holding fee, per-credit tuition, and total tuition?"
    )
    facets = _required_evidence_facets(query)
    names = {facet["name"] for facet in facets}

    assert {
        "application fee",
        "fee-waiver conditions",
        "seat-holding deposit",
        "per-credit tuition rate",
        "total tuition",
        "tuition amount",
    } <= names

    partial = build_evidence_pack(
        query=query,
        result={
            "retrieval_documents": [
                {
                    "id": "chunk:generic-fee",
                    "text": "Applicants pay an application fee of AED 200.",
                    "source_url": "https://example.edu/faq/application-fee",
                }
            ]
        },
        coverage_plan={
            "intent": "broad_synthesis",
            "required_facets": facets,
            "semantic_sufficiency_enabled": False,
            "query_specific_rules_enabled": False,
        },
    )

    assert partial["coverage_status"] == "partial"
    assert {
        "fee-waiver conditions",
        "seat-holding deposit",
        "per-credit tuition rate",
        "total tuition",
    } <= set(partial["missing_required_facets"])


def test_scholarship_types_and_maximum_do_not_invent_unrequested_benefits():
    query = (
        "كم تبلغ الرسوم الدراسية السنوية لبرنامج البكالوريوس، "
        "وما أنواع المنح المتاحة وما الحد الأقصى لتغطيتها؟"
    )

    facets = _required_evidence_facets(query)
    names = {facet["name"] for facet in facets}

    assert {
        "scholarship availability and scope",
        "scholarship types",
        "maximum scholarship coverage",
        "tuition coverage",
        "tuition amount",
    } <= names
    assert {
        "living stipend",
        "accommodation support",
        "health coverage",
        "visa support",
    }.isdisjoint(names)
    assert next(
        facet
        for facet in facets
        if facet["name"] == "scholarship availability and scope"
    )["report_in_answer"] is False


def test_explicit_title_request_requires_the_complete_official_designation():
    query = (
        "Who is the current president, and what title is shown for that person "
        "on the leadership page?"
    )

    facets = _required_evidence_facets(query)
    title_facet = next(
        facet
        for facet in facets
        if facet["name"] == "complete official title or designation"
    )

    assert title_facet["min_alias_matches"] == 2
    assert {"president", "professor"} <= set(title_facet["aliases"])


def test_person_to_division_question_creates_a_mapping_coverage_facet():
    english = _required_evidence_facets(
        "Who are the deans, and which division does each person lead?"
    )
    arabic = _required_evidence_facets(
        "من هم العمداء، وأي قسم يقود كل منهم؟"
    )

    for facets in (english, arabic):
        mapping = next(
            facet
            for facet in facets
            if facet["name"] == "complete person-to-division mappings"
        )
        assert mapping["min_alias_matches"] == 3
        assert {"dean", "led by", "division of"} <= set(mapping["aliases"])


def test_gpu_range_visual_feedback_and_industry_process_get_semantic_facets():
    gpu_names = {
        facet["name"]
        for facet in _required_evidence_facets(
            "What range of GPU options does the center offer?"
        )
    }
    visual_names = {
        facet["name"]
        for facet in _required_evidence_facets(
            "What feedback and progress tracking does this screenshot show?"
        )
    }
    industry_names = {
        facet["name"]
        for facet in _required_evidence_facets(
            "How does the university engage with industry and capture value?"
        )
    }

    assert "GPU option range and intended audience" in gpu_names
    assert "visible feedback and progress indicators" in visual_names
    assert "industry engagement and value-capture process" in industry_names


def test_arabic_research_dashboard_bridge_adds_categories_not_answer_values():
    aliases = set(
        _multilingual_retrieval_bridge_tokens(
            "أي نموذج في لوحة مشاريع البحث هو نموذج هندي للغات الكبيرة "
            "وحقق أداء معرفياً في الاستدلال؟"
        )
    )

    assert {"research projects", "hindi", "indian", "knowledge", "reasoning"} <= aliases
    assert "NANDA" not in aliases


def test_person_to_division_pack_reserves_every_relation_bearing_section():
    page_url = "https://example.edu/research/divisions"
    query = "Who are the three deans, and which division does each lead?"
    records = [
        {
            "id": "page-card",
            "text": "Our divisions. Meet our deans and learn about their work.",
            "source_url": page_url,
            "coverage_page_card": True,
        },
        *[
            {
                "id": f"chunk:{slug}",
                "text": f"SECTION: {division}\n## {division}\nLed by Dean {person}.",
                "source_url": page_url,
            }
            for slug, division, person in (
                ("biology", "Division of Biological Sciences", "Amina Noor"),
                ("computing", "Division of Computing Sciences", "Ben Chen"),
                ("undergraduate", "Division of Undergraduate Studies", "Carla Diaz"),
            )
        ],
    ]

    pack = build_evidence_pack(
        query=query,
        result={"retrieval_documents": records},
        max_items=4,
        max_chars=8000,
        max_per_source=2,
        coverage_plan={
            "intent": "broad_synthesis",
            "required_pages": [page_url],
            "required_facets": _required_evidence_facets(query),
            "semantic_sufficiency_enabled": False,
            "query_specific_rules_enabled": False,
        },
    )

    selected_ids = {item["id"] for item in pack["items"]}
    assert {
        "chunk:biology",
        "chunk:computing",
        "chunk:undergraduate",
    } <= selected_ids


def test_academic_level_scope_rejects_cross_level_funding_pages():
    query = "What scholarships and tuition apply to undergraduate students?"
    masters_page = {
        "source_url": "https://example.edu/study/msc-programs",
        "normalized_url": "https://example.edu/study/msc-programs",
        "page_type": "admissions_or_program",
        "identity_text": "Master's programs and scholarships",
        "search_text": "Full scholarship including tuition and a stipend",
        "tokens": {"masters", "scholarship", "tuition", "stipend"},
    }
    undergraduate_page = {
        "source_url": "https://example.edu/admissions/undergraduate-admissions",
        "normalized_url": "https://example.edu/admissions/undergraduate-admissions",
        "page_type": "admissions_or_program",
        "identity_text": "Undergraduate admissions and scholarships",
        "search_text": "Merit-based and need-based scholarships cover tuition",
        "tokens": {"undergraduate", "scholarship", "tuition"},
    }

    assert not _durable_page_candidate_allowed(query, masters_page)
    assert _durable_page_candidate_allowed(query, undergraduate_page)


def test_named_acronym_scope_beats_a_single_facet_faq():
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    query = (
        "What are the MAAI application fee, waiver conditions, seat-holding "
        "fee, per-credit tuition, and total tuition?"
    )

    def page(url, title, text):
        identity = f"{url} {title}".casefold()
        content = f"{title} {text}".casefold()
        return {
            "source_url": url,
            "normalized_url": url,
            "page_type": "admissions_or_program",
            "identity_text": identity,
            "identity_tokens": set(identity.replace("/", " ").replace("-", " ").split()),
            "tail_tokens": set(url.rsplit("/", 1)[-1].replace("-", " ").split()),
            "host_identity_tokens": set(),
            "search_text": content,
            "tokens": set(content.replace("-", " ").split()),
            "page_is_arabic": False,
        }

    faq = page(
        "https://example.edu/faq/application-fee",
        "Application fee",
        "Applicants pay an application fee.",
    )
    program = page(
        "https://example.edu/study/master-in-applied-ai",
        "Master in Applied AI",
        (
            "The MAAI application fee has waiver conditions. A seat-holding "
            "deposit is credited toward the per-credit and total tuition."
        ),
    )

    assert retriever._generalized_page_target_score(
        query, program
    ) > retriever._generalized_page_target_score(query, faq) + 0.5


def test_program_mapping_expands_overview_to_answer_bearing_category_children():
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    root = "https://example.edu/research/our-divisions"

    def page(url, title, text):
        value = f"{title} {text}".casefold()
        return {
            "source_url": url,
            "normalized_url": url,
            "identity_text": title.casefold(),
            "identity_tokens": set(title.casefold().split()),
            "tail_tokens": set(url.rsplit("/", 1)[-1].replace("-", " ").split()),
            "host_identity_tokens": set(),
            "search_text": value,
            "tokens": set(value.split()),
            "search_sequence_text": f" {value} ",
            "identity_sequence_text": f" {title.casefold()} ",
            "page_is_arabic": False,
            "page_type": "content",
            "title": title,
        }

    computing = f"{root}/division-computing"
    biology = f"{root}/division-biological"
    undergraduate = f"{root}/division-undergraduate"
    retriever._coverage_page_records = [
        page(root, "Our divisions", "Overview of research divisions"),
        page(computing, "Division of Computing", "Programs across computer science and AI"),
        page(biology, "Division of Biological Sciences", "Graduate programs in computational biology"),
        page(undergraduate, "Division of Undergraduate Studies", "Undergraduate programs"),
    ]

    expanded = retriever._expand_mapping_page_scope(
        "Which academic programs belong to each of the two research divisions?",
        [root],
        page_card_rank={computing: 0, biology: 1, undergraduate: 2},
        dense_source_rank={computing: 0, biology: 1, undergraduate: 2},
    )

    assert expanded == [computing, biology]
