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
