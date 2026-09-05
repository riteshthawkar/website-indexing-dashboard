from __future__ import annotations

from copy import deepcopy

from pipeline.core.page_graph_bridge import (
    build_navigation_catalog,
    build_page_graph_bridge,
    validate_page_graph_bridge,
)
from pipeline.retrieval.navigation_planner import (
    GroundedNavigationPlanner,
    infer_navigation_context,
)


def _representation_bundle() -> dict:
    return {
        "schema_version": "mbzuai.representation.v2",
        "generated_at": "2026-08-22T00:00:00Z",
        "documents": [
            {
                "document_id": "document:admissions",
                "document_revision_id": "revision:admissions",
                "corpus_record_id": "corpus:admissions",
                "source_type": "webpage",
                "language": "en",
                "title": "Graduate admissions",
                "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "canonical_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "markdown_path": "/corpus/admissions.md",
                "page_card_ids": ["page:admissions"],
            },
            {
                "document_id": "document:portal",
                "document_revision_id": "revision:portal",
                "corpus_record_id": "corpus:portal",
                "source_type": "webpage",
                "language": "en",
                "title": "Application portal",
                "source_url": "https://apply.mbzuai.ac.ae/",
                "canonical_url": "https://apply.mbzuai.ac.ae/",
                "markdown_path": "/corpus/portal.md",
                "page_card_ids": ["page:portal"],
            },
        ],
        "page_cards": [
            {
                "page_card_id": "page:admissions",
                "document_revision_id": "revision:admissions",
                "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "canonical_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "canonical_family_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "title": "Graduate admissions",
                "purpose_summary": "Official admission process and application instructions.",
                "language": "en",
                "page_type": "admissions",
                "content_backed": True,
                "topics": [{"label": "Admissions"}],
                "audiences": [{"label": "Applicants"}],
                "sections": [
                    {
                        "section_id": "section:requirements",
                        "heading": "Application requirements",
                        "level": 2,
                        "html_locator": "#requirements",
                        "evidence_id": "evidence:requirements",
                    }
                ],
                "action_ids": ["action:apply", "action:email", "action:external"],
                "retrieval_action_ids": ["action:apply", "action:email", "action:external"],
            },
            {
                "page_card_id": "page:portal",
                "document_revision_id": "revision:portal",
                "source_url": "https://apply.mbzuai.ac.ae/",
                "canonical_url": "https://apply.mbzuai.ac.ae/",
                "canonical_family_url": "https://apply.mbzuai.ac.ae/",
                "title": "Application portal",
                "purpose_summary": "Official application portal.",
                "language": "en",
                "page_type": "portal",
                "content_backed": True,
                "topics": [{"label": "Applications"}],
                "audiences": [{"label": "Applicants"}],
                "sections": [
                    {
                        "section_id": "section:portal",
                        "heading": "Start an application",
                        "level": 1,
                        "html_locator": "main",
                        "evidence_id": "evidence:portal",
                    }
                ],
                "action_ids": [],
                "retrieval_action_ids": [],
            },
        ],
        "actions": [
            {
                "action_id": "action:apply",
                "page_card_id": "page:admissions",
                "source_section_id": "section:requirements",
                "source_section_heading": "Application requirements",
                "label": "Apply now",
                "context_label": "Application requirements",
                "action_type": "apply",
                "target_url": "https://apply.mbzuai.ac.ae/",
                "canonical_target_url": "https://apply.mbzuai.ac.ae/",
                "target_kind": "official_subdomain",
                "official_target": True,
                "retrieval_eligible": True,
                "opens_new_window": True,
                "authentication_requirement": "explicit",
                "evidence": [{"evidence_id": "evidence:apply"}],
            },
            {
                "action_id": "action:email",
                "page_card_id": "page:admissions",
                "source_section_id": "section:requirements",
                "source_section_heading": "Application requirements",
                "label": "admission@mbzuai.ac.ae",
                "context_label": "Admissions contact",
                "action_type": "email",
                "target_url": "mailto:admission@mbzuai.ac.ae",
                "canonical_target_url": "mailto:admission@mbzuai.ac.ae",
                "target_kind": "email",
                "official_target": False,
                "retrieval_eligible": True,
                "opens_new_window": False,
                "authentication_requirement": "none",
                "evidence": [{"evidence_id": "evidence:email"}],
            },
            {
                "action_id": "action:external",
                "page_card_id": "page:admissions",
                "source_section_id": "section:requirements",
                "source_section_heading": "Application requirements",
                "label": "External advert",
                "context_label": "Advertisement",
                "action_type": "apply",
                "target_url": "https://example.com/apply",
                "canonical_target_url": "https://example.com/apply",
                "target_kind": "external",
                "official_target": False,
                "retrieval_eligible": True,
                "opens_new_window": True,
                "authentication_requirement": "unknown",
                "evidence": [{"evidence_id": "evidence:external"}],
            },
        ],
    }


def _crawl_graph() -> dict:
    return {
        "schema_version": "mbzuai.canonical_page_link_graph.v1",
        "graph_type": "canonical_page_links",
        "nodes": [
            {"url": "https://mbzuai.ac.ae/study/graduate-admission-process/"},
            {"url": "https://apply.mbzuai.ac.ae/"},
            {"url": "https://example.com/"},
        ],
        "edges": [
            {
                "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "target_url": "https://apply.mbzuai.ac.ae/",
                "properties": {"anchor_texts": ["Apply now"]},
            },
            {
                "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "target_url": "https://apply.mbzuai.ac.ae/",
                "properties": {"anchor_texts": ["Application portal"]},
            },
            {
                "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "target_url": "https://example.com/",
            },
            {
                "source_url": "https://apply.mbzuai.ac.ae/",
                "target_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "properties": {"anchor_texts": ["Application guidance"]},
            },
            {
                "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "target_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
            },
        ],
    }


def _chunk_index() -> dict:
    return {
        "version": "test.chunks.v1",
        "chunks": [
            {
                "chunk_id": "chunk:requirements",
                "document_revision_id": "revision:admissions",
                "page_card_id": "page:admissions",
                "section_id": "section:requirements",
                "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/",
                "section_path": ["Application requirements"],
            },
            {
                "chunk_id": "chunk:portal",
                "source_markdown_path": "/corpus/portal.md",
                "source_url": "https://apply.mbzuai.ac.ae/",
                "heading": "Start an application",
                "section_path": ["Start an application"],
            },
        ],
    }


def _ready_bridge() -> dict:
    return build_page_graph_bridge(
        representation_bundle=_representation_bundle(),
        crawl_graph=_crawl_graph(),
        chunk_index=_chunk_index(),
        require_chunk_index=True,
    )


def test_page_graph_bridge_cleans_edges_and_links_all_evidence_layers():
    bridge = _ready_bridge()

    assert bridge["coverage"]["passed"] is True
    assert bridge["coverage"]["status"] == "ready"
    assert validate_page_graph_bridge(bridge, require_chunk_index=True)["passed"] is True
    assert bridge["stats"]["clean_page_link_edges"] == 2
    assert bridge["stats"]["dropped_crawl_edges_by_reason"] == {
        "self_loop": 1,
        "target_not_page_card": 1,
    }

    edge_keys = {
        (edge["edge_type"], edge["source_id"], edge["target_id"])
        for edge in bridge["edges"]
    }
    assert (
        "PAGE_LINKS_TO_PAGE",
        "page:admissions",
        "page:portal",
    ) in edge_keys
    assert (
        "PAGE_REPRESENTS_DOCUMENT",
        "page:admissions",
        "revision:admissions",
    ) in edge_keys
    assert (
        "SECTION_HAS_CHUNK",
        "section:requirements",
        "chunk:requirements",
    ) in edge_keys
    assert ("PAGE_HAS_ACTION", "page:admissions", "action:apply") in edge_keys
    assert not any("example.com" in str(edge) for edge in bridge["edges"])

    merged_link = next(
        edge for edge in bridge["edges"] if edge["edge_type"] == "PAGE_LINKS_TO_PAGE"
    )
    assert merged_link["properties"]["anchor_texts"] == [
        "Application portal",
        "Apply now",
    ]


def test_page_graph_bridge_deduplicates_identical_section_observations():
    bundle = _representation_bundle()
    section = deepcopy(bundle["page_cards"][0]["sections"][0])
    bundle["page_cards"][0]["sections"].append(section)

    bridge = build_page_graph_bridge(
        representation_bundle=bundle,
        crawl_graph=_crawl_graph(),
    )

    assert bridge["coverage"]["passed"] is True
    assert bridge["stats"]["duplicate_section_instances_removed"] == 1
    assert len([value for value in bridge["sections"] if value["section_id"] == "section:requirements"]) == 1


def test_page_graph_bridge_fails_chunk_coverage_when_section_scope_is_unresolved():
    chunk_index = _chunk_index()
    chunk_index["chunks"][0].pop("section_id")
    chunk_index["chunks"][0]["section_path"] = ["A heading not on the page"]

    bridge = build_page_graph_bridge(
        representation_bundle=_representation_bundle(),
        crawl_graph=_crawl_graph(),
        chunk_index=chunk_index,
        require_chunk_index=True,
    )

    assert bridge["coverage"]["passed"] is False
    assert bridge["coverage"]["status"] == "failed"
    assert bridge["stats"]["chunk_issue_counts"] == {"chunk_section_unresolved": 1}
    assert bridge["coverage"]["chunk_gates"]["chunk_bridge_ready"] is False

    optional_but_supplied = build_page_graph_bridge(
        representation_bundle=_representation_bundle(),
        crawl_graph=_crawl_graph(),
        chunk_index=chunk_index,
        require_chunk_index=False,
    )
    assert optional_but_supplied["coverage"]["passed"] is False


def test_page_graph_bridge_derives_grounded_document_sections_when_enabled():
    bundle = _representation_bundle()
    bundle["documents"].append(
        {
            "document_id": "document:report",
            "document_revision_id": "revision:report",
            "corpus_record_id": "corpus:report",
            "source_type": "pdf",
            "language": "en",
            "title": "Annual report",
            "source_url": "",
            "canonical_url": "",
            "markdown_path": "/corpus/report.md",
            "page_card_ids": [],
        }
    )
    chunk_index = _chunk_index()
    chunk_index["chunks"][0].pop("section_id")
    chunk_index["chunks"][0]["section_path"] = ["Generated media", "Video"]
    chunk_index["chunks"].append(
        {
            "chunk_id": "chunk:report-results",
            "source_markdown_path": "/corpus/report.md",
            "source_url": "",
            "heading": "Results",
            "section_path": ["Results"],
            "page_numbers": [7],
        }
    )

    bridge = build_page_graph_bridge(
        representation_bundle=bundle,
        crawl_graph=_crawl_graph(),
        chunk_index=chunk_index,
        require_chunk_index=True,
        derive_document_sections=True,
    )

    assert bridge["coverage"]["passed"] is True
    assert bridge["coverage"]["chunk_gates"]["chunk_bridge_ready"] is True
    assert bridge["stats"]["document_sections"] == 3
    derived = [
        section
        for section in bridge["sections"]
        if section["section_kind"] == "document_section"
    ]
    assert {section["section_path"][-1] for section in derived} == {
        "Video",
        "Results",
        "Start an application",
    }
    pdf_section = next(section for section in derived if section["heading"] == "Results")
    assert pdf_section["page_card_id"] == ""
    assert pdf_section["document_revision_id"] == "revision:report"
    assert pdf_section["page_numbers"] == [7]
    assert pdf_section["chunk_ids"] == ["chunk:report-results"]
    assert (
        "DOCUMENT_HAS_SECTION",
        "revision:report",
        pdf_section["section_id"],
    ) in {
        (edge["edge_type"], edge["source_id"], edge["target_id"])
        for edge in bridge["edges"]
    }


def test_navigation_catalog_excludes_external_actions_and_plan_is_evidence_bound():
    catalog = build_navigation_catalog(_ready_bridge())
    planner = GroundedNavigationPlanner(catalog=catalog)

    assert [action["action_id"] for action in catalog["actions"]] == [
        "action:apply",
        "action:email",
    ]
    plan = planner.plan(
        query="How do I apply to MBZUAI?",
        result={
            "evidence_pack": {
                "items": [
                    {
                        "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/"
                    }
                ]
            },
            "selected_chunk_ids": ["chunk:requirements"],
        },
    )

    assert plan["status"] == "ready"
    assert plan["intent"] == "apply"
    assert plan["target_page"]["page_card_id"] == "page:admissions"
    assert [step["action_type"] for step in plan["steps"]] == ["open_page", "apply"]
    assert plan["steps"][1]["action_id"] == "action:apply"
    assert plan["evidence"]["action_ids"] == ["action:apply"]
    assert all(step["official_target"] for step in plan["steps"])

    contact_plan = planner.plan(
        query="How can I contact MBZUAI admissions?",
        result={
            "evidence_pack": {
                "items": [
                    {
                        "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/"
                    }
                ]
            }
        },
    )
    assert contact_plan["status"] == "ready"
    assert contact_plan["steps"][1]["action_type"] == "email"
    assert contact_plan["steps"][1]["target_url"] == "mailto:admission@mbzuai.ac.ae"

    direct_endpoint_plan = planner.plan(
        query="How do I apply to MBZUAI?",
        result={
            "evidence_pack": {
                "items": [{"source_url": "https://apply.mbzuai.ac.ae/"}]
            }
        },
    )
    assert direct_endpoint_plan["status"] == "ready"
    assert direct_endpoint_plan["target_page"]["page_card_id"] == "page:portal"

    traversal_catalog = deepcopy(catalog)
    traversal_catalog["pages"].append(
        {
            "page_card_id": "page:overview",
            "document_revision_id": "revision:overview",
            "source_url": "https://mbzuai.ac.ae/overview/",
            "canonical_url": "https://mbzuai.ac.ae/overview/",
            "canonical_family_url": "https://mbzuai.ac.ae/overview/",
            "title": "University overview",
            "purpose_summary": "General information about the university.",
            "language": "en",
            "page_type": "overview",
            "topic_labels": [],
            "audience_labels": [],
            "sections": [],
            "chunk_ids": [],
            "outgoing_page_card_ids": ["page:admissions"],
            "action_ids": [],
        }
    )
    traversal_planner = GroundedNavigationPlanner(catalog=traversal_catalog)
    traversed_plan = traversal_planner.plan(
        query="How do I apply to MBZUAI?",
        result={
            "evidence_pack": {
                "items": [{"source_url": "https://mbzuai.ac.ae/overview/"}]
            }
        },
    )
    assert traversed_plan["status"] == "ready"
    assert traversed_plan["target_page"]["page_card_id"] == "page:admissions"
    assert traversed_plan["evidence"]["page_card_ids"] == [
        "page:admissions",
        "page:overview",
    ]

    unavailable = planner.plan(
        query="How do I apply to MBZUAI?",
        result={"evidence_pack": {"items": [{"source_url": "https://mbzuai.ac.ae/not-indexed/"}]}},
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["steps"] == []
    assert unavailable["target_page"] is None


def test_navigation_intent_is_explicit_and_multilingual():
    assert infer_navigation_context("What degrees does MBZUAI offer?")["intent"] == "none"
    assert infer_navigation_context("What does the IFM website say about its research centers?")["intent"] == "none"
    assert infer_navigation_context("Open the official IFM page")["intent"] == "open_page"
    assert infer_navigation_context("Please download the campus map PDF")["intent"] == "download"
    assert infer_navigation_context("كيف أتواصل مع الجامعة؟")["intent"] == "contact"
    assert infer_navigation_context("What does the Search action take me to?")["intent"] == "search"
    assert infer_navigation_context("ما الذي يحمّل هذا الزر؟")["intent"] == "download"
    assert infer_navigation_context("ما عنوان البريد الذي ستراسل به الجامعة؟")["intent"] == "contact"
    assert infer_navigation_context("أين أذهب لتقديم طلب التوظيف؟")["intent"] == "apply"
    assert infer_navigation_context(
        "ما الميزة المالية التي تقدمها جميع برامج الدكتوراه في الجامعة؟"
    )["intent"] == "none"
    assert infer_navigation_context(
        "According to the graduate admission process page, what funding is provided?"
    )["intent"] == "none"
    assert infer_navigation_context(
        "What are the steps in the graduate admission process?"
    )["intent"] == "follow_steps"


def test_upstream_planner_cannot_promote_informational_process_page_to_navigation():
    from pipeline.core.navigation_intent import normalize_navigation_context

    query = "According to the graduate admission process page, what funding is provided?"
    context = normalize_navigation_context(
        query,
        {
            "intent": "follow_steps",
            "goal": query,
            "confidence": 0.99,
            "source": "retrieval_query_planner",
        },
    )

    assert context["intent"] == "none"


def test_upstream_planner_cannot_promote_arabic_offers_fact_to_apply_action():
    from pipeline.core.navigation_intent import normalize_navigation_context

    query = "ما الميزة المالية التي تقدمها جميع برامج الدكتوراه في الجامعة؟"
    context = normalize_navigation_context(
        query,
        {
            "intent": "apply",
            "goal": query,
            "confidence": 0.99,
            "source": "retrieval_query_planner",
        },
    )

    assert context["intent"] == "none"


def test_navigation_planner_preserves_direct_page_and_action_evidence():
    catalog = build_navigation_catalog(_ready_bridge())
    login_action = deepcopy(
        next(action for action in catalog["actions"] if action["action_id"] == "action:apply")
    )
    login_action.update(
        {
            "action_id": "action:apply-login",
            "action_type": "login",
            "label": "Apply Now",
            "target_url": "https://apply.mbzuai.ac.ae/login",
            "canonical_target_url": "https://apply.mbzuai.ac.ae/login",
        }
    )
    catalog["actions"] = [
        action for action in catalog["actions"] if action["action_id"] != "action:apply"
    ]
    catalog["actions"].append(login_action)
    planner = GroundedNavigationPlanner(catalog=catalog)

    plan = planner.plan(
        query="If I use the Apply Now action, where does it lead?",
        result={
            # Repeated weak evidence from a linked page must not outweigh the
            # direct Page Card/action lanes.
            "evidence_pack": {
                "items": [
                    {"source_url": "https://apply.mbzuai.ac.ae/"}
                    for _ in range(12)
                ]
            },
            "dense_page_card_ids": ["page:admissions", "page:portal"],
            "dense_action_ids": ["action:apply-login"],
        },
    )

    assert plan["status"] == "ready"
    assert plan["target_page"]["page_card_id"] == "page:admissions"
    assert plan["steps"][1]["action_id"] == "action:apply-login"
    assert plan["steps"][1]["action_type"] == "login"


def test_navigation_planner_prefers_canonical_phd_workflow_over_action_rich_news():
    catalog = build_navigation_catalog(_ready_bridge())
    canonical_url = (
        "https://preprod.mbzuai.ac.ae/admissions/graduate-phd-admissions"
    )
    news_url = (
        "https://preprod.mbzuai.ac.ae/knowledge-center/the-node/"
        "mbzuai-opens-admissions-fall-2026-intake"
    )
    catalog["pages"] = [
        {
            "page_card_id": "page:phd-admissions",
            "document_revision_id": "revision:phd-admissions",
            "source_url": canonical_url,
            "canonical_url": canonical_url,
            "canonical_family_url": canonical_url,
            "title": "Graduate Ph.D. admissions",
            "purpose_summary": "Canonical PhD requirements and application process.",
            "language": "en",
            "page_type": "admissions_or_program",
            "topic_labels": ["Admissions"],
            "audience_labels": ["Applicants"],
            "sections": [],
            "chunk_ids": [],
            "outgoing_page_card_ids": [],
            "action_ids": [],
        },
        {
            "page_card_id": "page:intake-news",
            "document_revision_id": "revision:intake-news",
            "source_url": news_url,
            "canonical_url": news_url,
            "canonical_family_url": news_url,
            "title": "MBZUAI opens admissions for Fall 2026 intake",
            "purpose_summary": "A dated intake announcement.",
            "language": "en",
            "page_type": "news_or_event",
            "topic_labels": ["Admissions"],
            "audience_labels": ["Applicants"],
            "sections": [],
            "chunk_ids": [],
            "outgoing_page_card_ids": [],
            "action_ids": ["action:news-apply"],
        },
    ]
    catalog["chunks"] = []
    catalog["actions"] = [
        {
            "action_id": "action:news-apply",
            "page_card_id": "page:intake-news",
            "label": "Apply now",
            "context_label": "Fall 2026 intake",
            "source_section_heading": "Applications",
            "action_type": "apply",
            "target_url": "https://apply.mbzuai.ac.ae/",
            "canonical_target_url": "https://apply.mbzuai.ac.ae/",
            "target_kind": "official_subdomain",
            "official_target": True,
        }
    ]
    planner = GroundedNavigationPlanner(catalog=catalog)

    plan = planner.plan(
        query="How do I apply for a PhD at MBZUAI?",
        result={
            "dense_page_card_ids": ["page:intake-news", "page:phd-admissions"],
            "dense_action_ids": ["action:news-apply"],
        },
    )

    assert plan["status"] == "partial"
    assert plan["target_page"]["page_card_id"] == "page:phd-admissions"
    assert plan["steps"][0]["target_url"] == canonical_url
    assert plan["warnings"] == ["requested_action_not_grounded_on_target_page"]


def test_navigation_planner_prefers_directory_action_context_over_profile_tie():
    catalog = build_navigation_catalog(_ready_bridge())
    catalog["pages"] = [
        {
            "page_card_id": "page:profile",
            "document_revision_id": "revision:profile",
            "source_url": "https://mbzuai.ac.ae/people/mark-juan/",
            "canonical_url": "https://mbzuai.ac.ae/people/mark-juan/",
            "title": "Mark Juan",
            "purpose_summary": "Official profile.",
            "language": "en",
            "page_type": "profile",
            "topic_labels": [],
            "audience_labels": [],
            "sections": [],
            "chunk_ids": [],
            "outgoing_page_card_ids": [],
            "action_ids": ["action:profile-email"],
        },
        {
            "page_card_id": "page:directory",
            "document_revision_id": "revision:directory",
            "source_url": "https://mbzuai.ac.ae/directory/",
            "canonical_url": "https://mbzuai.ac.ae/directory/",
            "title": "Directory listing",
            "purpose_summary": "Official people directory.",
            "language": "en",
            "page_type": "directory",
            "topic_labels": [],
            "audience_labels": [],
            "sections": [],
            "chunk_ids": [],
            "outgoing_page_card_ids": [],
            "action_ids": ["action:directory-email"],
        },
    ]
    catalog["chunks"] = []
    catalog["actions"] = [
        {
            "action_id": "action:profile-email",
            "page_card_id": "page:profile",
            "label": "Email",
            "context_label": "",
            "source_section_heading": "",
            "action_type": "email",
            "target_url": "mailto:mark.juan@mbzuai.ac.ae",
            "canonical_target_url": "mailto:mark.juan@mbzuai.ac.ae",
            "target_kind": "email",
            "official_target": True,
        },
        {
            "action_id": "action:directory-email",
            "page_card_id": "page:directory",
            "label": "Email",
            "context_label": "Mark Juan",
            "source_section_heading": "Mark Juan",
            "source_section_id": "section:mark-juan",
            "action_type": "email",
            "target_url": "mailto:mark.juan@mbzuai.ac.ae",
            "canonical_target_url": "mailto:mark.juan@mbzuai.ac.ae",
            "target_kind": "email",
            "official_target": True,
        },
    ]
    planner = GroundedNavigationPlanner(catalog=catalog)

    plan = planner.plan(
        query="What is the email address linked to Mark Juan in the directory listing?",
        result={
            "dense_page_card_ids": ["page:profile", "page:directory"],
            "dense_action_ids": [
                "action:directory-email",
                "action:profile-email",
            ],
        },
    )

    assert plan["status"] == "ready"
    assert plan["target_page"]["page_card_id"] == "page:directory"
    assert plan["steps"][1]["action_id"] == "action:directory-email"

    profile_plan = planner.plan(
        query="What email is linked to Mark Juan on his library profile page?",
        result={
            "dense_page_card_ids": ["page:profile", "page:directory"],
            "dense_action_ids": [
                "action:directory-email",
                "action:profile-email",
            ],
        },
    )
    assert profile_plan["target_page"]["page_card_id"] == "page:profile"
    assert profile_plan["steps"][1]["action_id"] == "action:profile-email"


def test_navigation_planner_late_fuses_page_cards_with_selected_chunk_identity():
    catalog = build_navigation_catalog(_ready_bridge())
    planner = GroundedNavigationPlanner(catalog=catalog)

    fused = planner.fuse_page_card_ranking(
        {
            "dense_page_card_ids": ["page:portal", "page:admissions"],
            "selected_chunk_ids": ["chunk:requirements"],
        },
        evidence_weight=0.15,
        rrf_k=60,
    )

    assert fused == ["page:admissions", "page:portal"]


def test_navigation_planner_bridges_selected_chunks_to_query_relevant_sections():
    catalog = build_navigation_catalog(_ready_bridge())
    admissions = next(
        page
        for page in catalog["pages"]
        if page["page_card_id"] == "page:admissions"
    )
    admissions["chunk_ids"] = ["chunk:one", "chunk:two"]
    admissions["sections"] = [
        {
            "section_id": "section:funding",
            "section_kind": "page_heading",
            "heading": "Funding and benefits",
            "chunk_ids": [],
        },
        {
            "section_id": "section:requirements",
            "section_kind": "page_heading",
            "heading": "Application requirements",
            "chunk_ids": [],
        },
    ]
    catalog["chunks"] = [
        {
            "chunk_id": "chunk:one",
            "document_revision_id": "revision:admissions",
            "page_card_id": "page:admissions",
            "section_id": "",
            "page_section_ids": [],
        }
    ]
    planner = GroundedNavigationPlanner(catalog=catalog)

    identities = planner.representation_identities(
        {
            "selected_chunk_ids": ["chunk:one"],
            "dense_page_card_ids": ["page:admissions"],
        },
        query="What funding and benefits are provided to funded students?",
        chunk_records={
            "chunk:one": {
                "text": "All funded students receive comprehensive financial support."
            }
        },
    )

    assert "section:funding" in identities["section_ids"]
    assert "section:requirements" not in identities["section_ids"]


def test_navigation_planner_prefers_specific_contact_endpoint_over_general_email():
    catalog = build_navigation_catalog(_ready_bridge())
    generic_email = deepcopy(catalog["actions"][1])
    generic_email.update(
        {
            "action_id": "action:info",
            "source_section_heading": "General contact",
            "label": "info@mbzuai.ac.ae",
            "context_label": "Contact",
            "target_url": "mailto:info@mbzuai.ac.ae",
            "canonical_target_url": "mailto:info@mbzuai.ac.ae",
        }
    )
    catalog["actions"].append(generic_email)
    planner = GroundedNavigationPlanner(catalog=catalog)
    evidence = {
        "evidence_pack": {
            "items": [
                {
                    "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/"
                }
            ]
        }
    }

    admissions_plan = planner.plan(
        query="Where can I contact admissions?",
        result=evidence,
    )
    assert admissions_plan["steps"][1]["target_url"] == (
        "mailto:admission@mbzuai.ac.ae"
    )

    general_plan = planner.plan(
        query="How can I contact MBZUAI?",
        result=evidence,
    )
    assert general_plan["steps"][1]["target_url"] == "mailto:info@mbzuai.ac.ae"


def test_navigation_planner_does_not_use_page_name_to_choose_sibling_email():
    catalog = build_navigation_catalog(_ready_bridge())
    page = next(
        value for value in catalog["pages"] if value["page_card_id"] == "page:admissions"
    )
    page.update(
        {
            "title": "Research Administration",
            "purpose_summary": "Research Administration full lifecycle support.",
            "source_url": "https://research.mbzuai.ac.ae/research-administration",
            "canonical_url": "https://research.mbzuai.ac.ae/research-administration",
            "canonical_family_url": "https://research.mbzuai.ac.ae/research-administration",
        }
    )
    base = deepcopy(next(action for action in catalog["actions"] if action["action_id"] == "action:email"))
    general = deepcopy(base)
    general.update(
        {
            "action_id": "action:00-general",
            "label": "OSR@mbzuai.ac.ae",
            "context_label": "Full Lifecycle Support",
            "source_section_heading": "Full Lifecycle Support",
            "target_url": "mailto:OSR@mbzuai.ac.ae",
            "canonical_target_url": "mailto:OSR@mbzuai.ac.ae",
        }
    )
    specialized = deepcopy(base)
    specialized.update(
        {
            "action_id": "action:10-postaward",
            "label": "postaward.administration@mbzuai.ac.ae",
            "context_label": "Full Lifecycle Support",
            "source_section_heading": "Full Lifecycle Support",
            "target_url": "mailto:postaward.administration@mbzuai.ac.ae",
            "canonical_target_url": "mailto:postaward.administration@mbzuai.ac.ae",
        }
    )
    catalog["actions"] = [general, specialized]
    planner = GroundedNavigationPlanner(catalog=catalog)

    plan = planner.plan(
        query="What email does Research Administration give for full lifecycle support?",
        result={
            "evidence_pack": {
                "items": [
                    {"source_url": "https://research.mbzuai.ac.ae/research-administration"}
                ]
            },
            # The semantic action lane can rank a target containing the page
            # name first; that must not outweigh equal section evidence.
            "dense_action_ids": ["action:10-postaward", "action:00-general"],
        },
    )

    assert plan["status"] == "ready"
    assert plan["steps"][1]["target_url"] == "mailto:OSR@mbzuai.ac.ae"


def test_navigation_planner_does_not_substitute_email_for_requested_phone_number():
    catalog = build_navigation_catalog(_ready_bridge())
    planner = GroundedNavigationPlanner(catalog=catalog)

    plan = planner.plan(
        query="What phone number should I call for admissions?",
        result={
            "evidence_pack": {
                "items": [
                    {
                        "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/"
                    }
                ]
            }
        },
    )

    assert plan["status"] == "partial"
    assert len(plan["steps"]) == 1
    assert plan["steps"][0]["action_type"] == "open_page"


def test_navigation_planner_rejects_catalog_from_failed_bridge():
    catalog = build_navigation_catalog(_ready_bridge())
    catalog["source_bridge_coverage_passed"] = False

    planner = GroundedNavigationPlanner(catalog=catalog)
    plan = planner.plan(
        query="How do I apply to MBZUAI?",
        result={
            "evidence_pack": {
                "items": [
                    {
                        "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process/"
                    }
                ]
            }
        },
    )

    assert planner.available is False
    assert planner.load_error == "navigation_catalog_source_bridge_failed"
    assert plan["status"] == "unavailable"
