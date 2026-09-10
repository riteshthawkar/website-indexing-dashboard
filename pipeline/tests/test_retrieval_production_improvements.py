import json
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline.core.base import StageContext, StageStatus
from pipeline.core.chunking import build_chunk_index
from pipeline.core.io import atomic_write_json, load_json_safe

from .test_production_readiness import run_async


def _valid_modern_vector_manifest() -> dict:
    namespaces = {
        "chunks": "chunks",
        "parents": "parents",
        "media": "media",
        "facts": "facts",
        "evidence_spans": "evidence_spans",
        "summaries": "summaries",
        "assertions": "assertions",
        "entities": "entities",
        "communities": "communities",
    }
    dense = {
        "chunks": 10,
        "parents": 3,
        "media": 0,
        "facts": 4,
        "evidence_spans": 5,
        "summaries": 2,
        "assertions": 2,
        "entities": 0,
        "communities": 0,
    }
    sparse = {f"sparse_{key}": value for key, value in dense.items()}
    dense_report = {
        namespace: dense[key]
        for key, namespace in namespaces.items()
        if dense[key] > 0
    }
    sparse_report = {
        namespace: sparse[f"sparse_{key}"]
        for key, namespace in namespaces.items()
        if sparse[f"sparse_{key}"] > 0
    }
    return {
        "schema_version": 2,
        "index_name": "idx",
        "sparse_index_name": "idx-sparse",
        "namespace_strategy": "static",
        "namespace_release_id": "",
        "namespaces": namespaces,
        "planned": {**dense, **sparse},
        "uploaded": {**dense, **sparse},
        "sparse": {"record_stats": {}},
        "verification": {
            "dense": {"expected": dict(dense_report), "actual": dict(dense_report), "failures": []},
            "sparse": {"expected": dict(sparse_report), "actual": dict(sparse_report), "failures": []},
        },
    }


def _write_valid_release_artifacts(work_dir: Path) -> None:
    (work_dir / "stage_outputs" / "upload_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "upload_graph").mkdir(parents=True)
    (work_dir / "stage_outputs" / "format_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "promote_graph").mkdir(parents=True)
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        _valid_modern_vector_manifest(),
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_graph" / "neo4j_upload_manifest.json",
        {
            "neo4j_namespace": "mbzuai:test",
            "neo4j_database": "neo4j",
            "graph_type": "promoted_semantic_graph",
            "node_count": 20,
            "edge_count": 30,
            "verification": {"expected_nodes": 20, "actual_nodes": 20, "expected_edges": 30, "actual_edges": 30},
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
        {
            "version": 5,
            "stats": {
                "chunk_count": 10,
                "parent_count": 3,
                "media_count": 0,
                "fact_count": 4,
                "evidence_span_count": 5,
                "summary_count": 2,
                "assertion_count": 2,
                "answer_count": 2,
                "lexical_count": 21,
            },
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph.json",
        {
            "nodes": [
                {"id": "chunk-1", "node_type": "chunk"},
                {"id": "assertion-1", "node_type": "relation_assertion"},
            ],
            "edges": [{
                "id": "edge-1",
                "source_id": "chunk-1",
                "target_id": "assertion-1",
                "edge_type": "CHUNK_SUPPORTS_ASSERTION",
            }],
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph_index.json",
        {"outgoing_edge_ids": {"chunk-1": []}},
    )


def _mock_successful_release_checks(monkeypatch, release, gates: Path) -> None:
    fake_audit = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        to_dict=lambda: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(release, "audit_run", lambda path: fake_audit)
    monkeypatch.setattr(
        release,
        "assess_production_readiness",
        lambda *args, **kwargs: {"ok": True, "error_count": 0, "warning_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        release,
        "validate_eval_examples",
        lambda *args, **kwargs: {"ok": True, "summary": {"query_count": 1}, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        release,
        "evaluate_retrieval_dataset",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"chunk_hit_at_5": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )
    monkeypatch.setattr(
        release,
        "evaluate_answer_readiness",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"pass_rate": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )


def _load_script_module(path: str):
    script_path = Path(__file__).resolve().parents[2] / path
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[script_path.stem] = module
    spec.loader.exec_module(module)
    return module


def test_adaptive_retriever_prefers_finalized_bundle_over_stale_build(tmp_path):
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    build_dir = tmp_path / "stage_outputs" / "build_retrieval_bundle"
    finalize_dir = tmp_path / "stage_outputs" / "finalize_retrieval_bundle"
    build_dir.mkdir(parents=True)
    finalize_dir.mkdir(parents=True)
    atomic_write_json(
        build_dir / "retrieval_bundle.json",
        {
            "version": 4,
            "chunk_records": [],
            "parent_records": [],
            "media_records": [],
            "fact_records": [],
            "evidence_span_records": [{"id": "stale-span", "text": "Stale span", "source_url": ""}],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    atomic_write_json(build_dir / "lexical_corpus.json", [])
    atomic_write_json(
        finalize_dir / "retrieval_bundle.json",
        {
            "version": 5,
            "chunk_records": [],
            "parent_records": [],
            "media_records": [],
            "fact_records": [],
            "evidence_span_records": [
                {
                    "id": "final-span",
                    "text": "Finalized span",
                    "source_url": "https://mbzuai.ac.ae/final",
                }
            ],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    atomic_write_json(finalize_dir / "lexical_corpus.json", [])

    retriever = AdaptiveHybridRetriever(
        config={
            "embedder": {"pinecone_index": "test-index"},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        },
        work_dir=tmp_path,
    )

    assert "final-span" in retriever.evidence_span_map
    assert retriever.evidence_span_map["final-span"]["source_url"] == "https://mbzuai.ac.ae/final"
    assert "stale-span" not in retriever.evidence_span_map


def test_page_card_dense_hit_bridges_to_chunks_by_document_revision(tmp_path):
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    stage_dir = tmp_path / "stage_outputs" / "format_retrieval"
    stage_dir.mkdir(parents=True)
    atomic_write_json(
        stage_dir / "retrieval_bundle.json",
        {
            "version": 6,
            "chunk_records": [
                {
                    "id": "chunk:ifm:about:0",
                    "text": "The Institute of Foundation Models is based in Abu Dhabi.",
                    "dense_text": "The Institute of Foundation Models is based in Abu Dhabi.",
                    "document_revision_id": "document-revision:ifm-about",
                    "source_url": "https://ifm.ai/about",
                },
                {
                    "id": "chunk:other:0",
                    "text": "An unrelated page.",
                    "dense_text": "An unrelated page.",
                    "document_revision_id": "document-revision:other",
                    "source_url": "https://mbzuai.ac.ae/other",
                },
            ],
            "parent_records": [],
            "media_records": [],
            "page_card_records": [
                {
                    "id": "page-card:ifm-about",
                    "document_revision_id": "document-revision:ifm-about",
                    "source_url": "https://ifm.ai/about",
                    "title": "About IFM - Institute of Foundation Models",
                }
            ],
            "action_records": [],
            "fact_records": [],
            "evidence_span_records": [],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    atomic_write_json(stage_dir / "lexical_corpus.json", [])

    retriever = AdaptiveHybridRetriever(
        config={
            "embedder": {"pinecone_index": "test-index"},
            "retrieval": {"enable_sparse": False, "enable_rerank": False},
        },
        work_dir=tmp_path,
    )

    assert retriever._seed_chunk_ids(["page-card:ifm-about"]) == [
        "chunk:ifm:about:0"
    ]


def test_source_query_bonus_prefers_named_subdomain_identity():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    query = (
        "What does the IFM website say it builds with partners, and where are "
        "its headquarters and research hubs?"
    )

    ifm_bonus = retriever._source_query_bonus(
        query,
        source_url="https://ifm.ai/about",
        document_title="About IFM - Institute of Foundation Models",
        heading="Who We Are",
        text="A global center for foundation models.",
    )
    generic_bonus = retriever._source_query_bonus(
        query,
        source_url="https://mbzuai.ac.ae/research/research-centers",
        document_title="Research Centers - MBZUAI",
        heading="Research centers",
        text="Partners from academia work across research centers.",
    )

    assert ifm_bonus >= generic_bonus + 2.0


def test_named_source_evidence_precedes_generic_topic_matches():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.evidence_span_map = {
        "generic": {
            "id": "generic",
            "text": "Research centers work with partners across academia.",
            "source_url": "https://mbzuai.ac.ae/research/research-centers",
            "document_title": "Research Centers - MBZUAI",
        },
        "ifm-about": {
            "id": "ifm-about",
            "text": "IFM has its headquarters in Abu Dhabi and research hubs in Paris and Silicon Valley.",
            "source_url": "https://ifm.ai/about",
            "document_title": "About IFM",
        },
        "ifm-collaborate": {
            "id": "ifm-collaborate",
            "text": "IFM partners with academic institutions, research labs, startups, and enterprise leaders.",
            "source_url": "https://ifm.ai/collaborate",
            "document_title": "Collaborate with IFM",
        },
    }
    retriever._score_text_match = lambda query, text: (
        1.0 if "partners" in text.lower() else 0.5
    )

    ranked = retriever._promote_named_source_evidence_spans(
        "What does the IFM website say about its headquarters, research hubs, and partners?",
        ["generic", "ifm-about", "ifm-collaborate"],
    )

    assert ranked[:2] == ["ifm-collaborate", "ifm-about"]


def test_evidence_pack_prioritizes_named_official_subdomain_sources():
    from pipeline.retrieval.evidence_packer import (
        _is_official_mbzuai_url,
        build_evidence_pack,
    )

    result = {
        "evidence_span_documents": [
            {
                "id": "generic-research-centers",
                "text": "MBZUAI research centers work with partners from academia.",
                "source_url": "https://mbzuai.ac.ae/research/research-centers",
                "span_type": "general",
            },
            {
                "id": "ifm-about",
                "text": "IFM is headquartered in Abu Dhabi with research hubs in Paris and Silicon Valley.",
                "source_url": "https://ifm.ai/about",
                "document_title": "cd99ef32a7737302a82a",
                "span_type": "general",
            },
            {
                "id": "ifm-collaborate",
                "text": "IFM partners with academic institutions, labs, startups, and enterprise leaders.",
                "source_url": "https://ifm.ai/collaborate",
                "span_type": "general",
            },
            {
                "id": "corrupted-chunk",
                "text": "TI",
                "source_url": "https://ifm.ai/",
                "document_title": "c5771aec882eacb81dfa",
                "span_type": "general",
            },
        ]
    }

    pack = build_evidence_pack(
        query=(
            "What does the IFM website say about its headquarters, research "
            "hubs, and partners?"
        ),
        result=result,
        max_items=3,
        max_chars=4000,
    )

    assert [item["id"] for item in pack["items"][:2]] == [
        "ifm-about",
        "ifm-collaborate",
    ]
    assert pack["items"][0]["document_title"] == "About"
    assert all(item["id"] != "corrupted-chunk" for item in pack["items"])
    assert _is_official_mbzuai_url("https://ifm.ai/about") is True


def test_evidence_pack_does_not_emit_tiny_budget_tail_fragments():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    result = {
        "evidence_span_documents": [
            {
                "id": "a-long-ifm-span",
                "text": ("IFM " + ("research " * 87)).strip(),
                "source_url": "https://ifm.ai/about",
                "span_type": "general",
            },
            {
                "id": "z-followup-span",
                "text": "This otherwise valid evidence would only fit as a tiny fragment.",
                "source_url": "https://ifm.ai/collaborate",
                "span_type": "general",
            },
        ]
    }

    pack = build_evidence_pack(
        query="What does IFM say about research?",
        result=result,
        max_items=4,
        max_chars=800,
    )

    assert pack["items"]
    assert all(len(item["text"]) >= 20 for item in pack["items"])
    assert all(len(item["text"].split()) >= 3 for item in pack["items"])


def test_evidence_pack_keeps_complete_job_requirements_chunk_within_source_cap():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://careers.mbzuai.ac.ae/careers/senior-grants-specialist-3"
    pack = build_evidence_pack(
        query=(
            "What are the minimum education and experience requirements for "
            "the Senior Grants Specialist role?"
        ),
        result={
            "evidence_span_documents": [
                {
                    "id": "responsibilities-one",
                    "text": (
                        "Grants Lifecycle Administration: administer pre-award, "
                        "post-award, and closeout processes and provide guidance "
                        "on compliance requirements."
                    ),
                    "source_url": source_url,
                    "document_title": "Senior Grants Specialist",
                    "span_type": "requirement",
                },
                {
                    "id": "responsibilities-two",
                    "text": (
                        "Maintain grant records, conduct compliance reviews, and "
                        "support internal and external audits."
                    ),
                    "source_url": source_url,
                    "document_title": "Senior Grants Specialist",
                    "span_type": "requirement",
                },
            ],
            "retrieval_documents": [
                {
                    "id": "complete-requirements-chunk",
                    "text": (
                        "TITLE: Senior Grants Specialist. Academic Qualifications "
                        "Required: Bachelor's degree in business administration, "
                        "finance, public administration, or Higher Education "
                        "Administration. A postgraduate degree is preferred. "
                        "Professional Experience Required: minimum of 5 years of "
                        "experience in research or grants administration."
                    ),
                    "source_url": source_url,
                    "document_title": "Senior Grants Specialist",
                }
            ],
        },
        max_items=2,
        max_chars=4000,
        max_per_source=2,
    )

    selected_ids = {item["id"] for item in pack["items"]}
    assert "complete-requirements-chunk" in selected_ids
    packed_text = " ".join(item["text"] for item in pack["items"])
    assert "Bachelor's degree" in packed_text
    assert "minimum of 5 years" in packed_text


def test_evidence_pack_treats_experience_lookup_as_contiguous_detail_query():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://careers.mbzuai.ac.ae/careers/academic-appointments-partner"
    pack = build_evidence_pack(
        query="How much experience is required for the Academic Appointments Partner role?",
        result={
            "fact_documents": [
                {
                    "id": "liaison-fact",
                    "text": "Serve as the primary liaison for candidates during recruitment.",
                    "source_url": source_url,
                },
                {
                    "id": "appointments-fact",
                    "text": "Administer academic appointment documentation and reappointments.",
                    "source_url": source_url,
                },
            ],
            "retrieval_documents": [
                {
                    "id": "chunk:academic-appointments-partner",
                    "text": (
                        "Academic Appointments Partner. Professional Experience: Essential. "
                        "At least 5 years of experience in academic administration, HR "
                        "operations, or related roles in higher education."
                    ),
                    "source_url": source_url,
                    "document_title": "Academic Appointments Partner",
                }
            ],
        },
        max_items=2,
        max_chars=4000,
        max_per_source=2,
    )

    assert any(item["id"] == "chunk:academic-appointments-partner" for item in pack["items"])
    assert "At least 5 years" in " ".join(item["text"] for item in pack["items"])


def test_evidence_pack_preserves_fused_promoted_assertion_for_exact_role_fact():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://careers.mbzuai.ac.ae/careers/academic-appointments-partner"
    pack = build_evidence_pack(
        query="How much experience is required for the Academic Appointments Partner role?",
        result={
            "retrieval_documents": [
                {
                    "id": "chunk:c650:document-revision:role:00000:overview",
                    "text": (
                        "Academic Appointments Partner. Job purpose and responsibilities. "
                        "Professional Experience:"
                    ),
                    "source_url": source_url,
                    "document_title": "Academic Appointments Partner",
                },
                {
                    "id": "chunk:c650:document-revision:other:00000:other-role",
                    "text": "Another role requires a minimum of four years of experience.",
                    "source_url": "https://careers.mbzuai.ac.ae/careers/another-role",
                },
                {
                    "id": "assertion:academic-appointments-minimum-years",
                    "text": (
                        "Academic Appointments Partner minimum_years At least 5 years "
                        "of experience in academic administration, HR operations, or "
                        "related roles in higher education."
                    ),
                    "source_url": "",
                },
            ],
        },
        max_items=2,
        max_chars=4000,
        max_per_source=2,
    )

    assertion_items = [item for item in pack["items"] if item["kind"] == "assertion"]
    assert [item["id"] for item in assertion_items] == [
        "assertion:academic-appointments-minimum-years"
    ]
    assert "At least 5 years" in assertion_items[0]["text"]
    assert all(item["id"] != "chunk:c650:document-revision:other:00000:other-role" for item in pack["items"])


def test_evidence_pack_keeps_all_same_page_process_stages_for_detail_query():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://research.mbzuai.ac.ae/partnerships-and-engagements"
    stage_rows = [
        (10, "exploration", "Exploration: Brainstorming use-cases based on industry challenges."),
        (11, "refinement", "Refinement: Down-selection of use-cases based on priorities."),
        (12, "proposal", "High level proposal: Identify resources and contributions from each entity."),
        (13, "sign-off", "Engagement agreement sign-off: Full proposal generation and project start."),
    ]
    pack = build_evidence_pack(
        query="How does MBZUAI engage with industry and capture value across the process stages?",
        result={
            "retrieval_documents": [
                {
                    "id": "parent:c650:document-revision:partnerships:page",
                    "text": (
                        "How we engage and capture value across a four-stage industry process. "
                        + " ".join("Background context" for _ in range(180))
                    ),
                    "source_url": source_url,
                    "document_title": "Partnerships and Engagements",
                    "coverage_aggregate": True,
                },
                *[
                {
                    "id": (
                        "chunk:c650:document-revision:partnerships:"
                        f"{stage_index:05d}:{stage_id}"
                    ),
                    "text": (
                        "How we engage and capture value. " + stage_text
                    ),
                    "source_url": source_url,
                    "document_title": "Partnerships and Engagements",
                }
                for stage_index, stage_id, stage_text in stage_rows
                ],
            ],
            "fact_documents": [
                {
                    "id": "generic-industry-fact",
                    "text": "MBZUAI develops real-world AI systems for industry needs.",
                    "source_url": source_url,
                }
            ],
        },
        max_items=6,
        max_chars=4200,
        max_per_source=2,
    )

    packed_text = " ".join(item["text"] for item in pack["items"])
    assert all(stage_text in packed_text for _index, _stage_id, stage_text in stage_rows)


def test_evidence_pack_treats_arabic_program_list_as_structured_detail_query():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://mbzuai.ac.ae/ar/news/fall-2026"
    pack = build_evidence_pack(
        query="ما البرامج التي تفتح الجامعة باب القبول لها في خريف 2026؟",
        result={
            "retrieval_documents": [
                {
                    "id": "chunk:fall-intro",
                    "text": "فتح باب القبول في خريف 2026 لبرامج البكالوريوس والدراسات العليا.",
                    "source_url": source_url,
                },
                {
                    "id": "chunk:fall-program-list",
                    "text": (
                        "البرامج المتاحة: بكالوريوس العلوم في الذكاء الاصطناعي، "
                        "وماجستير الذكاء الاصطناعي التطبيقي، وماجستير علوم الحاسوب، "
                        "ودكتوراه علوم الحاسوب."
                    ),
                    "source_url": source_url,
                },
            ],
            "evidence_span_documents": [
                {
                    "id": "unrelated-fall-span",
                    "text": "تنطلق فعالية بحثية أخرى في خريف 2026.",
                    "source_url": "https://mbzuai.ac.ae/ar/news/other-event",
                }
            ],
        },
        max_items=3,
        max_chars=4000,
        max_per_source=2,
    )

    packed_text = " ".join(item["text"] for item in pack["items"])
    assert "ماجستير الذكاء الاصطناعي التطبيقي" in packed_text
    assert "دكتوراه علوم الحاسوب" in packed_text


def test_arabic_qualification_query_keeps_english_requirements_span():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://careers.mbzuai.ac.ae/careers/data-platform-engineer-iaai"
    pack = build_evidence_pack(
        query=(
            "ما المؤهل الدراسي المطلوب لوظيفة Data Platform Engineer "
            "في معهد IAAI؟"
        ),
        result={
            "fact_documents": [
                {
                    "id": "role-summary",
                    "text": "This position is well suited for an early-career engineer.",
                    "source_url": source_url,
                    "document_title": "Data Platform Engineer IAAI",
                },
                {
                    "id": "collaboration-summary",
                    "text": "The role works closely with an Applied Research Scientist.",
                    "source_url": source_url,
                    "document_title": "Data Platform Engineer IAAI",
                },
            ],
            "evidence_span_documents": [
                {
                    "id": "academic-qualifications",
                    "text": (
                        "Academic Qualifications Required: Bachelor's degree in "
                        "computer science, data engineering, information systems, "
                        "software engineering, or a related field. A Master's "
                        "degree is preferred but not mandatory."
                    ),
                    "source_url": source_url,
                    "document_title": "Data Platform Engineer IAAI",
                    "span_type": "requirement",
                }
            ],
            "retrieval_documents": [
                {
                    "id": "complete-qualification-chunk",
                    "text": (
                        "TITLE: Data Platform Engineer IAAI. Academic Qualifications "
                        "Required: Bachelor's degree in computer science, data "
                        "engineering, information systems, software engineering, "
                        "or a related field. A Master's degree is preferred but "
                        "not mandatory."
                    ),
                    "source_url": source_url,
                    "document_title": "Data Platform Engineer IAAI",
                }
            ],
        },
        max_items=2,
        max_chars=4000,
        max_per_source=2,
    )

    selected_ids = {item["id"] for item in pack["items"]}
    assert "academic-qualifications" in selected_ids
    assert "complete-qualification-chunk" in selected_ids
    packed_text = " ".join(item["text"] for item in pack["items"])
    assert "Bachelor's degree" in packed_text
    assert "preferred but not mandatory" in packed_text


def test_arabic_qualification_query_compacts_long_chunk_around_complete_block():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://careers.mbzuai.ac.ae/careers/data-platform-engineer-iaai"
    long_preamble = " ".join(
        f"Platform background paragraph {index} about agricultural systems."
        for index in range(90)
    )
    pack = build_evidence_pack(
        query=(
            "ما المؤهل الأكاديمي المطلوب لوظيفة Data Platform Engineer "
            "في معهد IAAI؟"
        ),
        result={
            "fact_documents": [
                {
                    "id": "bachelor-only-fragment",
                    "text": "Academic Qualifications Required: Bachelor's degree in a related field.",
                    "source_url": source_url,
                    "document_title": "Data Platform Engineer IAAI",
                }
            ],
            "retrieval_documents": [
                {
                    "id": "complete-long-qualification-chunk",
                    "text": (
                        f"TITLE: Data Platform Engineer IAAI. {long_preamble} "
                        "Academic Qualifications Required: Bachelor's degree in "
                        "computer science, data engineering, information systems, "
                        "software engineering, or a related field. A Master's "
                        "degree is preferred but not mandatory. Professional "
                        "Experience Required: two years of relevant experience."
                    ),
                    "source_url": source_url,
                    "document_title": "Data Platform Engineer IAAI",
                }
            ],
        },
        max_items=2,
        max_chars=4000,
        max_per_source=2,
    )

    assert pack["items"][0]["id"] == "complete-long-qualification-chunk"
    assert len(pack["items"][0]["text"]) <= 2400
    assert "Bachelor's degree" in pack["items"][0]["text"]
    assert "A Master's degree is preferred but not mandatory" in pack["items"][0]["text"]


def test_evidence_pack_keeps_complete_arabic_roles_chunk_over_scope_shift():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://mbzuai.ac.ae/ar/study/faculty/carlos-bustamante-ar"
    pack = build_evidence_pack(
        query=(
            "ما المناصب التي شغلها كارلوس بوستامانتي في جامعة ستانفورد "
            "قبل انضمامه إلى جامعة محمد بن زايد للذكاء الاصطناعي؟"
        ),
        result={
            "evidence_span_documents": [
                {
                    "id": "partial-stanford-role",
                    "text": (
                        "قبل انضمامه إلى جامعة محمد بن زايد للذكاء الاصطناعي، "
                        "شغل كارلوس بوستامانتي منصب أستاذ في جامعة ستانفورد."
                    ),
                    "source_url": source_url,
                    "span_type": "general",
                },
                {
                    "id": "cornell-scope-shift",
                    "text": (
                        "وقبل انضمامه إلى جامعة ستانفورد، كان عضو هيئة تدريس في "
                        "جامعة كورنيل وشغل منصب المدير المشارك لمركز كورنيل."
                    ),
                    "source_url": source_url,
                    "span_type": "general",
                },
            ],
            "retrieval_documents": [
                {
                    "id": "complete-stanford-roles",
                    "text": (
                        "قبل انضمامه إلى جامعة محمد بن زايد للذكاء الاصطناعي، شغل "
                        "كارلوس بوستامانتي في جامعة ستانفورد مناصب أستاذ في علوم "
                        "البيانات الحيوية الطبية وعلم الوراثة، وأول رئيس لقسم علوم "
                        "البيانات الحيوية الطبية، والمؤسس والمدير لمركز ستانفورد "
                        "للحوسبة الوراثية والتطورية والجينوميات البشرية."
                    ),
                    "source_url": source_url,
                    "document_title": "كارلوس بوستامانتي",
                }
            ],
        },
        max_items=2,
        max_chars=4000,
        max_per_source=2,
    )

    selected_ids = {item["id"] for item in pack["items"]}
    assert "complete-stanford-roles" in selected_ids
    packed_text = " ".join(item["text"] for item in pack["items"])
    assert "المؤسس والمدير" in packed_text
    assert "جامعة كورنيل" not in packed_text


@pytest.mark.parametrize(
    "query",
    [
        "According to the infographic, what percentage are male and female?",
        "بحسب الإنفوغراف، ما النسبة المئوية للذكور وما نسبة الإناث؟",
    ],
)
def test_evidence_pack_reserves_ranked_media_for_multilingual_visual_queries(query):
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    result = {
        "evidence_span_documents": [
            {
                "id": f"generic-span-{index}",
                "text": "General university evidence that should not displace the selected visual record.",
                "source_url": f"https://mbzuai.ac.ae/news/generic-{index}",
                "span_type": "general",
            }
            for index in range(8)
        ],
        "media": [
            {
                "id": "gender-infographic",
                "text": (
                    "SEMANTIC_CAPTION: Participant demographics infographic. "
                    "VISIBLE_TEXT: 25% إناث 75% ذكور."
                ),
                "source_url": "https://staticcdn.mbzuai.ac.ae/prospectus-arabic.pdf",
                "media_type": "image",
            }
        ],
    }

    pack = build_evidence_pack(query=query, result=result, max_items=8, max_chars=8000)

    assert pack["items"][0]["kind"] == "media"
    assert pack["items"][0]["id"] == "gender-infographic"
    assert "25% إناث 75% ذكور" in pack["items"][0]["text"]
    assert pack["items"][0]["authority_score"] == pytest.approx(0.95)


def test_evidence_pack_does_not_force_media_into_non_visual_queries():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    pack = build_evidence_pack(
        query="What are the graduate admission requirements?",
        result={
            "evidence_span_documents": [
                {
                    "id": "admissions-span",
                    "text": "Graduate admission requirements include an eligible bachelor's degree.",
                    "source_url": "https://mbzuai.ac.ae/study/graduate-admission-process",
                    "span_type": "fact",
                }
            ],
            "media": [
                {
                    "id": "decorative-photo",
                    "text": "A decorative campus photograph with no admissions details.",
                    "source_url": "https://mbzuai.ac.ae/campus",
                    "media_type": "image",
                }
            ],
        },
        max_items=1,
        max_chars=2000,
    )

    assert [item["id"] for item in pack["items"]] == ["admissions-span"]


def test_evidence_pack_promotes_qr_invitation_media_and_caps_visual_candidates():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    result = {
        "media": [
            {
                "id": "faculty-portfolio",
                "text": "MBZUAI Faculty Portfolio 2024-2025. Scan QR for a digital copy.",
                "source_url": "https://careers.mbzuai.ac.ae/engineering-vacancies",
            },
            {
                "id": "conference-speaker",
                "text": "A conference speaker profile card.",
                "source_url": "https://staticcdn.mbzuai.ac.ae/conference.pdf",
            },
            {
                "id": "brand-page",
                "text": "A brand guideline page with typography examples.",
                "source_url": "https://staticcdn.mbzuai.ac.ae/brand.pdf",
            },
            {
                "id": "unrelated-visual",
                "text": "A research laboratory photograph.",
                "source_url": "https://mbzuai.ac.ae/research",
            },
        ]
    }

    pack = build_evidence_pack(
        query="What document is shown in the image, and what does it invite the viewer to do?",
        result=result,
        max_items=8,
        max_chars=8000,
    )

    media_items = [item for item in pack["items"] if item["kind"] == "media"]
    assert media_items[0]["id"] == "faculty-portfolio"
    assert len(media_items) == 3


def test_evidence_pack_prefers_same_language_arabic_visual_evidence():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    pack = build_evidence_pack(
        query="بحسب الإنفوغراف، ما النسبة المئوية للذكور وما نسبة الإناث؟",
        result={
            "media": [
                {
                    "id": "arabic-demographics",
                    "text": "إنفوغراف المشاركين. النص المرئي: 75% ذكور و25% إناث.",
                    "source_url": "https://staticcdn.mbzuai.ac.ae/arabic-prospectus.pdf",
                },
                {
                    "id": "english-demographics",
                    "text": "Participant demographics infographic: 71% Male and 29% Female.",
                    "source_url": "https://staticcdn.mbzuai.ac.ae/english-prospectus.pdf",
                },
            ]
        },
        max_items=4,
        max_chars=4000,
    )

    assert pack["items"][0]["id"] == "arabic-demographics"


def test_evidence_pack_preserves_top_mixed_language_named_diagram():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    pack = build_evidence_pack(
        query=(
            "ما الخطوة الثانية في الإطار المعروض لبناء "
            "Cultural Commonsense Knowledge Graph، وما الفكرة العامة لها؟"
        ),
        result={
            "media": [
                {
                    "id": "gold-english-diagram",
                    "text": (
                        "Cultural Commonsense Knowledge Graph framework. "
                        "Step two: ITERATIVE EXPANSION with forward and intermediate expansion."
                    ),
                    "source_url": "https://mbzuai.ac.ae/news/cultural-archives",
                },
                {
                    "id": "arabic-adjacent-visual-one",
                    "text": "صورة عربية عن قياس المعرفة الثقافية وبناء مجموعة معيارية.",
                    "source_url": "https://mbzuai.ac.ae/ar/news/cultural-benchmark",
                },
                {
                    "id": "arabic-adjacent-visual-two",
                    "text": "رسم عربي عام عن نماذج الذكاء الاصطناعي والثقافة.",
                    "source_url": "https://mbzuai.ac.ae/ar/news/cultural-models",
                },
                {
                    "id": "arabic-adjacent-visual-three",
                    "text": "مخطط عربي عام عن أرشيفات ثقافية خفية.",
                    "source_url": "https://mbzuai.ac.ae/ar/news/cultural-archives",
                },
            ]
        },
        max_items=4,
        max_chars=4000,
    )

    assert pack["items"][0]["id"] == "gold-english-diagram"


def test_evidence_pack_preserves_visible_text_beyond_long_media_context():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    long_context = " ".join(["Generic regional context"] * 170)
    pack = build_evidence_pack(
        query="What does the map indicate about road impact and the area measured for points of interest?",
        result={
            "media": [
                {
                    "id": "rain-impact-map",
                    "text": (
                        "IMAGE: Figure 3: Analysis map\n"
                        "DOCUMENT: Rain case study\n"
                        f"CONTEXT: {long_context}\n"
                        "VISIBLE_TEXT: 140 km roads are impacted, including highways and main roads. "
                        "For points of interest, a 0.2 km radius is considered.\n"
                        "CONTEXTUAL_CAPTION: Satellite analysis map of impacted infrastructure.\n"
                        "SOURCE_URL: https://staticcdn.mbzuai.ac.ae/rain-study.pdf"
                    ),
                    "source_url": "https://staticcdn.mbzuai.ac.ae/rain-study.pdf",
                }
            ]
        },
        max_items=4,
        max_chars=2400,
    )

    evidence = pack["items"][0]["text"]
    assert "140 km roads" in evidence
    assert "0.2 km radius" in evidence
    assert "CONTEXTUAL_CAPTION" in evidence
    assert len(evidence) <= 2400


def test_evidence_pack_keeps_complete_table_visible_text_when_it_fits():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    rows = " ".join([f"Model-{index} Avg {index}.00" for index in range(35)])
    visible_text = f"Hindi MCQ Benchmarks {rows} Llama-3.1-70B Instruct Avg 47.96"
    pack = build_evidence_pack(
        query="Which model has the highest Avg in the Hindi MCQ Benchmarks table?",
        result={
            "media": [
                {
                    "id": "hindi-table",
                    "text": (
                        "IMAGE: Table 4\n"
                        "DOCUMENT: Hindi model evaluation\n"
                        f"CONTEXT: {'irrelevant prose ' * 200}\n"
                        f"VISIBLE_TEXT: {visible_text}\n"
                        "SEMANTIC_CAPTION: A table comparing Hindi model benchmarks."
                    ),
                    "source_url": "https://mbzuai.ac.ae/news/hindi-model",
                }
            ]
        },
        max_items=4,
        max_chars=2400,
    )

    evidence = pack["items"][0]["text"]
    assert "Llama-3.1-70B Instruct Avg 47.96" in evidence
    assert len(evidence) <= 2400


def _bare_media_scorer(media_by_id):
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.media_texts_by_id = {
        media_id: str(media.get("text") or "")
        for media_id, media in media_by_id.items()
    }
    return retriever


def test_media_relevance_prefers_exact_collected_data_visual():
    media_by_id = {
        "hpp-visual": {
            "id": "hpp-visual",
            "text": (
                "A glowing blue head diagram. Collected data are automatically analysed "
                "and are not reviewed or used for diagnosis."
            ),
            "source_url": "https://mbzuai.ac.ae/research/healthcare",
        },
        "generic-visual": {
            "id": "generic-visual",
            "text": "A glowing blue artificial-intelligence research illustration.",
            "source_url": "https://mbzuai.ac.ae/research",
        },
    }
    retriever = _bare_media_scorer(media_by_id)
    query = "What does the image say happens to the collected data?"

    assert retriever._score_media_relevance(query, media_by_id["hpp-visual"]) > retriever._score_media_relevance(
        query,
        media_by_id["generic-visual"],
    )


def test_media_relevance_prefers_explicit_pdf_page_number():
    media_by_id = {
        "page-10": {
            "id": "page-10",
            "media_type": "image",
            "text": (
                "IMAGE: Visual from Class of 2024 Arabic page 10. "
                "VISIBLE_TEXT: الإسهام في تطوير تقنيات الجيل التالي."
            ),
            "source_url": "https://staticcdn.mbzuai.ac.ae/class-of-2024-arabic.pdf",
        },
        "same-event-photo": {
            "id": "same-event-photo",
            "media_type": "image",
            "text": "IMAGE: A graduate portrait from the Class of 2024 ceremony.",
            "source_url": "https://mbzuai.ac.ae/ar/commencement-2024",
        },
    }
    retriever = _bare_media_scorer(media_by_id)
    query = "في الصفحة 10 من برنامج دفعة 2024 العربي، ماذا تعرض الصورة؟"

    assert retriever._score_media_relevance(
        query,
        media_by_id["page-10"],
    ) > retriever._score_media_relevance(query, media_by_id["same-event-photo"])


def test_media_relevance_prefers_qr_invitation_visual():
    media_by_id = {
        "faculty-portfolio": {
            "id": "faculty-portfolio",
            "text": "MBZUAI Faculty Portfolio 2024-2025. Scan QR for a digital copy.",
            "source_url": "https://careers.mbzuai.ac.ae/engineering-vacancies",
        },
        "generic-visual": {
            "id": "generic-visual",
            "text": "A photograph of faculty members speaking at an event.",
            "source_url": "https://mbzuai.ac.ae/faculty",
        },
    }
    retriever = _bare_media_scorer(media_by_id)
    query = "What document is shown in the image, and what does it invite the viewer to do?"

    assert retriever._score_media_relevance(
        query,
        media_by_id["faculty-portfolio"],
    ) > retriever._score_media_relevance(query, media_by_id["generic-visual"])


def test_media_relevance_prefers_complete_academic_gpa_entry():
    media_by_id = {
        "declaration": {
            "id": "declaration",
            "text": "Application tabs: Personal Details, Academic History, Declaration.",
            "source_url": "https://staticcdn.mbzuai.ac.ae/application.pdf",
        },
        "programming-course": {
            "id": "programming-course",
            "text": (
                "Programming Courses Taken. Course Name Python Programming. "
                "Final Mark 4.0. Maximum Possible Mark 4.0."
            ),
            "source_url": "https://staticcdn.mbzuai.ac.ae/application.pdf",
        },
        "academic-gpa": {
            "id": "academic-gpa",
            "text": (
                "Academic History. University Name Carnegie Mellon University. "
                "Degree Level Bachelor. Major Artificial Intelligence. "
                "Cumulative Grade Point Average (CGPA) 4.0. Maximum Possible "
                "Grade Point Average (GPA) 4.0."
            ),
            "source_url": "https://staticcdn.mbzuai.ac.ae/application.pdf",
        },
    }
    retriever = _bare_media_scorer(media_by_id)
    query = (
        "What academic history example is shown in the form, and what does it "
        "say about the GPA fields?"
    )

    target_score = retriever._score_media_relevance(
        query,
        media_by_id["academic-gpa"],
    )
    assert target_score > retriever._score_media_relevance(
        query,
        media_by_id["declaration"],
    )
    assert target_score > retriever._score_media_relevance(
        query,
        media_by_id["programming-course"],
    )


def test_media_relevance_prefers_same_language_gender_infographic():
    media_by_id = {
        "arabic-demographics": {
            "id": "arabic-demographics",
            "text": "إنفوغراف المشاركين. النص المرئي: 75% ذكور و25% إناث.",
            "source_url": "https://staticcdn.mbzuai.ac.ae/arabic-prospectus.pdf",
        },
        "english-demographics": {
            "id": "english-demographics",
            "text": "Participant demographics infographic: 71% Male and 29% Female.",
            "source_url": "https://staticcdn.mbzuai.ac.ae/english-prospectus.pdf",
        },
    }
    retriever = _bare_media_scorer(media_by_id)
    query = "بحسب الإنفوغراف، ما النسبة المئوية للذكور وما نسبة الإناث؟"

    assert retriever._score_media_relevance(
        query,
        media_by_id["arabic-demographics"],
    ) > retriever._score_media_relevance(query, media_by_id["english-demographics"])


def test_media_relevance_crosses_languages_for_after_rain_visual():
    from pipeline.retrieval.adaptive_hybrid import _semantic_query_alias_tokens

    media_by_id = {
        "after-rain-map": {
            "id": "after-rain-map",
            "text": (
                "Figure 2: After Rain. Satellite imagery captured on April 17, 2024, "
                "depicting water accumulation post-rainfall in Dubai."
            ),
            "source_url": "https://staticcdn.mbzuai.ac.ae/rain-case-study.pdf",
        },
        "arabic-watermark-visual": {
            "id": "arabic-watermark-visual",
            "text": "صورة عربية تقارن طرق إزالة العلامات المائية من الصور الاصطناعية.",
            "source_url": "https://mbzuai.ac.ae/ar/news/watermark-study",
        },
    }
    retriever = _bare_media_scorer(media_by_id)
    query = "ماذا تُظهر صورة ما بعد المطر في هذه الدراسة؟"

    aliases = set(_semantic_query_alias_tokens(query))
    assert {"after", "rain", "water", "accumulation"} <= aliases
    assert retriever._score_media_relevance(
        query,
        media_by_id["after-rain-map"],
    ) > retriever._score_media_relevance(query, media_by_id["arabic-watermark-visual"])


@pytest.mark.parametrize(
    "query",
    [
        "What does the table show?",
        "ما الخطوة الثانية في الإطار المعروض؟",
        "بحسب الإنفوغراف، ما النسب المعروضة؟",
    ],
)
def test_multilingual_visual_cues_enable_media_retrieval(query):
    from pipeline.retrieval.adaptive_hybrid import _is_media_query

    assert _is_media_query(query) is True


def test_query_embedding_cache_reuses_successful_vectors(monkeypatch):
    from pipeline.retrieval import adaptive_hybrid as mod

    with mod._QUERY_EMBEDDING_CACHE_LOCK:
        mod._QUERY_EMBEDDING_CACHE.clear()

    calls = {"count": 0}

    class FakeEmbedResponse:
        embeddings = [type("Embedding", (), {"values": [0.3, 0.7]})()]

    class FakeModels:
        def embed_content(self, **kwargs):
            calls["count"] += 1
            return FakeEmbedResponse()

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(mod, "_make_gemini_client", lambda: FakeClient())

    first = mod._embed_query(
        "cache regression probe",
        model="gemini-embedding-2-preview",
        output_dimensionality=2,
    )
    second = mod._embed_query(
        "cache regression probe",
        model="gemini-embedding-2-preview",
        output_dimensionality=2,
    )

    assert first == [0.3, 0.7]
    assert second == [0.3, 0.7]
    assert calls["count"] == 1

    with mod._QUERY_EMBEDDING_CACHE_LOCK:
        mod._QUERY_EMBEDDING_CACHE.clear()


def test_query_embedding_failure_cooldown_skips_repeated_provider_calls(monkeypatch):
    from pipeline.retrieval import adaptive_hybrid as mod

    with mod._QUERY_EMBEDDING_CACHE_LOCK:
        mod._QUERY_EMBEDDING_CACHE.clear()
    with mod._QUERY_EMBEDDING_FAILURE_LOCK:
        mod._QUERY_EMBEDDING_FAILURE_STATE.clear()
    monkeypatch.setattr(mod, "_QUERY_EMBEDDING_RETRIES", 0)
    monkeypatch.setattr(mod, "_QUERY_EMBEDDING_FAILURE_COOLDOWN_SECONDS", 60.0)

    calls = {"count": 0}

    class FakeModels:
        def embed_content(self, **kwargs):
            calls["count"] += 1
            raise RuntimeError("429 RESOURCE_EXHAUSTED")

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(mod, "_make_gemini_client", lambda: FakeClient())

    for expected in ("RESOURCE_EXHAUSTED", "temporarily disabled"):
        try:
            mod._embed_query(
                "cooldown regression probe",
                model="gemini-embedding-2-preview",
                output_dimensionality=2,
            )
        except RuntimeError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError("expected query embedding failure")

    assert calls["count"] == 1
    with mod._QUERY_EMBEDDING_FAILURE_LOCK:
        mod._QUERY_EMBEDDING_FAILURE_STATE.clear()


def test_adaptive_retriever_uses_sparse_local_fallback_when_dense_embedding_fails(tmp_path, monkeypatch):
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    stage_dir = tmp_path / "stage_outputs" / "format_retrieval"
    stage_dir.mkdir(parents=True)
    atomic_write_json(
        stage_dir / "retrieval_bundle.json",
        {
            "version": 5,
            "chunk_records": [
                {
                    "id": "chunk-location",
                    "dense_text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "document_title": "About MBZUAI",
                    "source_url": "https://mbzuai.ac.ae/about/",
                }
            ],
            "parent_records": [],
            "media_records": [],
            "fact_records": [
                {
                    "id": "fact-location",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "linked_chunk_ids": ["chunk-location"],
                    "source_url": "https://mbzuai.ac.ae/about/",
                }
            ],
            "evidence_span_records": [],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    atomic_write_json(
        stage_dir / "lexical_corpus.json",
        [
            {
                "id": "chunk-location",
                "record_type": "chunk",
                "text": "MBZUAI is located in Masdar City Abu Dhabi",
                "tokens": ["mbzuai", "located", "masdar", "city", "abu", "dhabi"],
            },
            {
                "id": "fact-location",
                "record_type": "fact",
                "text": "MBZUAI is located in Masdar City Abu Dhabi",
                "tokens": ["mbzuai", "located", "masdar", "city", "abu", "dhabi"],
            },
        ],
    )

    retriever = AdaptiveHybridRetriever(
        config={
            "embedder": {"pinecone_index": "test-index", "pinecone_sparse_index": "test-sparse-index"},
            "retrieval": {
                "enable_sparse": True,
                "enable_rerank": True,
                "parallel_lane_workers": 1,
                "max_context_chunks": 3,
            },
        },
        work_dir=tmp_path,
    )
    monkeypatch.setattr(
        retriever,
        "embed_query",
        lambda query: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")),
    )
    monkeypatch.setattr(
        retriever,
        "_pinecone_sparse_index",
        lambda: (_ for _ in ()).throw(AssertionError("sparse Pinecone should be skipped after embedding failure")),
    )
    monkeypatch.setattr(
        retriever,
        "_pinecone_client_obj",
        lambda: (_ for _ in ()).throw(AssertionError("Pinecone rerank should be skipped after embedding failure")),
    )

    result = retriever.retrieve("Where is MBZUAI located?")

    assert result["query_embedding_status"] == "failed_sparse_local_fallback"
    assert result["query_embedding_error"] == "query_embedding_rate_limited"
    assert "fact-location" in result["selected_fact_ids"]
    assert "chunk-location" in result["selected_chunk_ids"]
    assert result["fact_documents"][0]["source_url"] == "https://mbzuai.ac.ae/about/"
    assert result["sparse_chunk_ids"] == []
    assert result["sparse_fact_ids"] == []
    assert result["rerank_method"] == "fallback_external_lanes_disabled"


def test_adaptive_retriever_materializes_facts_for_contact_synthesis(tmp_path, monkeypatch):
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    stage_dir = tmp_path / "stage_outputs" / "format_retrieval"
    stage_dir.mkdir(parents=True)
    faq_url = "https://mbzuai.ac.ae/about/faq"
    undergraduate_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    screening_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf"
    chunks = [
        {
            "id": "chunk-faq",
            "dense_text": "General admissions change and document upload requests must be sent by email to admission@mbzuai.ac.ae.",
            "text": "General admissions change and document upload requests must be sent by email to admission@mbzuai.ac.ae.",
            "document_title": "FAQ",
            "source_url": faq_url,
        },
        {
            "id": "chunk-undergraduate",
            "dense_text": "Undergraduate admissions inquiries can be sent to ug.admission@mbzuai.ac.ae.",
            "text": "Undergraduate admissions inquiries can be sent to ug.admission@mbzuai.ac.ae.",
            "document_title": "Undergraduate application submission",
            "source_url": undergraduate_url,
        },
        {
            "id": "chunk-screening",
            "dense_text": "The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae. Working hours are 8:00 AM - 5:00 PM Monday to Thursday and 8:00 AM - 12:30 PM on Friday.",
            "text": "The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae. Working hours are 8:00 AM - 5:00 PM Monday to Thursday and 8:00 AM - 12:30 PM on Friday.",
            "document_title": "MBZUAI Online Screening Exam Instructions",
            "source_url": screening_url,
        },
    ]
    facts = [
        {
            "id": "fact-faq-admissions",
            "text": "General admissions change and document upload requests must be sent by email to admission@mbzuai.ac.ae.",
            "linked_chunk_ids": ["chunk-faq"],
            "source_url": faq_url,
            "document_title": "FAQ",
        },
        {
            "id": "fact-undergraduate-admissions",
            "text": "Undergraduate admissions inquiries can be sent to ug.admission@mbzuai.ac.ae.",
            "linked_chunk_ids": ["chunk-undergraduate"],
            "source_url": undergraduate_url,
            "document_title": "Undergraduate application submission",
        },
        {
            "id": "fact-screening-support",
            "text": "The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae. Working hours are 8:00 AM - 5:00 PM Monday to Thursday and 8:00 AM - 12:30 PM on Friday.",
            "linked_chunk_ids": ["chunk-screening"],
            "source_url": screening_url,
            "document_title": "MBZUAI Online Screening Exam Instructions",
        },
    ]
    atomic_write_json(
        stage_dir / "retrieval_bundle.json",
        {
            "version": 5,
            "chunk_records": chunks,
            "parent_records": [],
            "media_records": [],
            "fact_records": facts,
            "evidence_span_records": [],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    atomic_write_json(
        stage_dir / "lexical_corpus.json",
        [
            {
                "id": record["id"],
                "record_type": "chunk" if record["id"].startswith("chunk") else "fact",
                "text": record["text"],
                "tokens": record["text"].replace("@", " ").replace(".", " ").split(),
            }
            for record in [*chunks, *facts]
        ],
    )

    retriever = AdaptiveHybridRetriever(
        config={
            "embedder": {"pinecone_index": "test-index"},
            "retrieval": {
                "enable_sparse": False,
                "enable_rerank": False,
                "parallel_lane_workers": 1,
                "max_context_chunks": 4,
            },
        },
        work_dir=tmp_path,
    )
    monkeypatch.setattr(
        retriever,
        "embed_query",
        lambda query: (_ for _ in ()).throw(RuntimeError("dense unavailable")),
    )

    result = retriever.retrieve(
        "Compare the right contact paths for general admissions, undergraduate admissions, "
        "and IT support for the online screening exam, including when IT support is available."
    )

    fact_ids = set(result["selected_fact_ids"])
    fact_sources = {document["source_url"] for document in result["fact_documents"]}
    assert {"fact-faq-admissions", "fact-undergraduate-admissions", "fact-screening-support"} <= fact_ids
    assert {faq_url, undergraduate_url, screening_url} <= fact_sources


def test_routed_retriever_bridges_family_visit_query_to_parent_housing_policy(tmp_path, monkeypatch):
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    stage_dir = tmp_path / "stage_outputs" / "format_retrieval"
    stage_dir.mkdir(parents=True)
    source_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    records = {
        "chunk-parent-housing": {
            "id": "chunk-parent-housing",
            "dense_text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. However, Student Affairs can recommend nearby hotels or Airbnbs for visiting parents.",
            "text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. However, Student Affairs can recommend nearby hotels or Airbnbs for visiting parents.",
            "document_title": "Undergraduate application submission",
            "source_url": source_url,
        },
        "chunk-generic-accommodation": {
            "id": "chunk-generic-accommodation",
            "dense_text": "Student accommodation includes multi-occupancy rooms, shared bathroom facilities, laundry facilities, kitchen, TV, and internet access.",
            "text": "Student accommodation includes multi-occupancy rooms, shared bathroom facilities, laundry facilities, kitchen, TV, and internet access.",
            "document_title": "Undergraduate application submission",
            "source_url": source_url,
        },
    }
    fact = {
        "id": "fact-parent-housing",
        "text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents. However, the Student Affairs team can recommend nearby hotels or Airbnbs for visiting parents.",
        "linked_chunk_ids": ["chunk-parent-housing"],
        "source_url": source_url,
        "document_title": "Undergraduate application submission",
    }
    span = {
        "id": "span-parent-housing",
        "text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents.",
        "sparse_text": "family visiting parents stay campus housing nearby hotels Airbnbs",
        "chunk_id": "chunk-parent-housing",
        "linked_chunk_ids": ["chunk-parent-housing"],
        "source_url": source_url,
        "document_title": "Undergraduate application submission",
        "span_type": "policy",
    }
    atomic_write_json(
        stage_dir / "retrieval_bundle.json",
        {
            "version": 5,
            "chunk_records": list(records.values()),
            "parent_records": [],
            "media_records": [],
            "fact_records": [fact],
            "evidence_span_records": [span],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    lexical_rows = []
    for record_type, record in [
        ("chunk", records["chunk-parent-housing"]),
        ("chunk", records["chunk-generic-accommodation"]),
        ("fact", fact),
        ("evidence_span", span),
    ]:
        text = " ".join(str(record.get(key) or "") for key in ("text", "dense_text", "sparse_text", "document_title"))
        lexical_rows.append(
            {
                "id": record["id"],
                "record_type": record_type,
                "text": text,
                "tokens": text.replace("@", " ").replace(".", " ").replace("-", " ").split(),
            }
        )
    atomic_write_json(stage_dir / "lexical_corpus.json", lexical_rows)

    retriever = RoutedHybridRetriever(
        config={
            "embedder": {"pinecone_index": "test-index"},
            "retrieval": {
                "routed_graph_enabled": False,
                "enable_sparse": False,
                "enable_rerank": False,
                "unsupported_intent_guard_enabled": False,
                "parallel_lane_workers": 1,
                "max_context_chunks": 4,
            },
        },
        work_dir=tmp_path,
    )
    monkeypatch.setattr(
        retriever.vector,
        "embed_query",
        lambda query: (_ for _ in ()).throw(RuntimeError("dense unavailable")),
    )

    result = retriever.retrieve(
        "A student's family is visiting MBZUAI. Explain family accommodation and whether guests can stay on campus."
    )
    packed_text = " ".join(str(item.get("text") or "") for item in (result.get("evidence_pack") or {}).get("items") or [])

    assert "fact-parent-housing" in result["selected_fact_ids"]
    assert "span-parent-housing" in result["selected_evidence_span_ids"]
    assert "does not provide housing for parents" in packed_text


def test_routed_retriever_sparse_local_fallback_when_dense_embedding_fails(tmp_path, monkeypatch):
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    stage_dir = tmp_path / "stage_outputs" / "format_retrieval"
    stage_dir.mkdir(parents=True)
    atomic_write_json(
        stage_dir / "retrieval_bundle.json",
        {
            "version": 5,
            "chunk_records": [
                {
                    "id": "chunk-location",
                    "dense_text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "document_title": "FAQ",
                    "source_url": "https://mbzuai.ac.ae/about/faq",
                }
            ],
            "parent_records": [],
            "media_records": [],
            "fact_records": [
                {
                    "id": "fact-location",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "linked_chunk_ids": ["chunk-location"],
                    "source_url": "https://mbzuai.ac.ae/about/faq",
                    "document_title": "FAQ",
                }
            ],
            "evidence_span_records": [],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    atomic_write_json(
        stage_dir / "lexical_corpus.json",
        [
            {
                "id": "chunk-location",
                "record_type": "chunk",
                "text": "MBZUAI is located in Masdar City Abu Dhabi",
                "tokens": ["mbzuai", "located", "masdar", "city", "abu", "dhabi"],
            },
            {
                "id": "fact-location",
                "record_type": "fact",
                "text": "MBZUAI is located in Masdar City Abu Dhabi",
                "tokens": ["mbzuai", "located", "masdar", "city", "abu", "dhabi"],
            },
        ],
    )
    retriever = RoutedHybridRetriever(
        config={
            "embedder": {"pinecone_index": "test-index"},
            "retrieval": {
                "routed_graph_enabled": False,
                "enable_sparse": False,
                "enable_rerank": False,
                "unsupported_intent_guard_enabled": False,
                "parallel_lane_workers": 1,
                "max_context_chunks": 3,
            },
        },
        work_dir=tmp_path,
    )
    monkeypatch.setattr(
        retriever.vector,
        "embed_query",
        lambda query: (_ for _ in ()).throw(RuntimeError("429 RESOURCE_EXHAUSTED")),
    )

    result = retriever.retrieve("Where is MBZUAI located?")

    assert result["query_embedding_status"] == "failed_sparse_local_fallback"
    assert result["query_embedding_error"] == "query_embedding_rate_limited"
    assert result["abstained"] is False
    assert "fact-location" in result["selected_fact_ids"]


def test_evidence_packer_allows_required_page_spans_beyond_default_source_cap():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    required_url = "https://mbzuai.ac.ae/about/faq"
    result = {
        "answer_documents": [
            {
                "id": "assertion-location-1",
                "text": "MBZUAI is located in Masdar City.",
                "source_url": required_url,
                "confidence": 0.9,
            },
            {
                "id": "assertion-location-2",
                "text": "MBZUAI is located in Abu Dhabi.",
                "source_url": required_url,
                "confidence": 0.9,
            },
        ],
        "evidence_span_documents": [
            {
                "id": "span-identity",
                "text": (
                    "The university is named after His Highness Sheikh Mohamed bin Zayed Al Nahyan. "
                    "MBZUAI has a separate legal personality and is affiliated to the Abu Dhabi Executive Council."
                ),
                "source_url": required_url,
                "span_type": "fact",
            }
        ],
        "selected_chunk_ids": ["chunk-identity"],
        "selected_evidence_span_ids": ["span-identity"],
    }

    pack = build_evidence_pack(
        query="Explain MBZUAI's institutional identity and affiliation.",
        result=result,
        max_items=4,
        max_chars=4000,
        max_per_source=2,
        coverage_plan={
            "intent": "exact_fact",
            "required_pages": [required_url],
            "required_entities": [],
            "required_sections": [],
        },
    )

    packed_ids = [item["id"] for item in pack["items"]]
    assert "span-identity" in packed_ids
    assert pack["coverage_status"] == "complete"


def test_routed_coverage_planner_treats_official_working_hours_as_specific_target():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._query_has_specific_target("What are MBZUAI's official working hours?")
    assert "/about/faq" in retriever._explicit_required_page_markers("What are MBZUAI's official working hours?")


def test_routed_collection_query_prefers_complete_landing_page_over_child():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    landing_url = "https://example.edu/academics/departments"
    child_url = "https://example.edu/academics/departments/robotics"
    retriever._coverage_page_records = [
        {
            "source_url": landing_url,
            "normalized_url": retriever._normalize_source_url(landing_url),
            "search_text": "departments robotics computing biology",
            "tokens": set(_tokenize("departments robotics computing biology")),
            "identity_text": "academic departments",
            "identity_tokens": set(_tokenize("academic departments")),
        },
        {
            "source_url": child_url,
            "normalized_url": retriever._normalize_source_url(child_url),
            "search_text": "current robotics department degree research",
            "tokens": set(_tokenize("current robotics department degree research")),
            "identity_text": "robotics department",
            "identity_tokens": set(_tokenize("robotics department")),
        },
    ]

    query = "List all current departments available at the university."
    plan = retriever._infer_coverage_requirements(
        query,
        retriever._coverage_intent(query, QueryMode.FACT),
    )

    assert plan["required_pages"] == [landing_url]
    assert plan["required_pages_source"] == "heuristic"


def test_updated_program_and_housing_markers_resolve_current_pages():
    from pipeline.retrieval.adaptive_hybrid import _semantic_query_alias_tokens
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    masters_markers = retriever._explicit_required_page_markers(
        "List all current master's degree programs offered by MBZUAI."
    )
    maai_markers = retriever._explicit_required_page_markers(
        "What are MAAI's duration, credits, tuition, and scholarship status?"
    )
    housing_markers = retriever._explicit_required_page_markers(
        "هل أحتاج إلى إحضار مسحوق الغسيل للسكن الجامعي؟"
    )

    assert masters_markers[0] == "/study/msc-programs"
    assert maai_markers[0] == (
        "/study/msc-programs/masters-in-applied-artificial-intelligence"
    )
    assert "/study/msc-programs" in maai_markers
    assert housing_markers == ["/campus-community/housing"]
    assert {"laundry", "detergent", "housing"} <= set(
        _semantic_query_alias_tokens(
            "هل أحتاج إلى إحضار مسحوق الغسيل للسكن الجامعي؟"
        )
    )


def test_multi_aspect_program_query_injects_complete_page_parent():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    page_url = "https://example.edu/study/programs/applied-ai"
    parent_id = "parent:c650:applied-ai:page"
    retriever.vector = SimpleNamespace(
        page_card_map={},
        evidence_span_map={},
        fact_map={},
        summary_map={},
        chunk_map={},
        parent_map={
            parent_id: {
                "id": parent_id,
                "source_url": page_url,
                "document_title": "Applied AI",
                "text": "Duration, study mode, credits, tuition, and scholarships.",
            }
        },
        _score_text_match=lambda query, text: 1.0,
    )
    retriever._coverage_page_records = retriever._build_coverage_page_records()
    retriever._coverage_page_records_by_url = {
        page["normalized_url"]: page
        for page in retriever._coverage_page_records
    }
    retriever._coverage_record_indexes = retriever._build_coverage_record_indexes()
    payload = {
        "selected_chunk_ids": [],
        "selected_parent_ids": [],
        "selected_fact_ids": [],
        "selected_evidence_span_ids": [],
        "retrieval_documents": [],
        "abstained": False,
    }

    changed = retriever._augment_payload_for_required_coverage(
        query=(
            "What are the duration, study mode, credit load, tuition, and "
            "scholarship status?"
        ),
        payload=payload,
        coverage_plan={"intent": "broad_synthesis", "required_pages": [page_url]},
    )

    assert changed is True
    assert payload["retrieval_documents"][0]["id"] == parent_id
    assert payload["retrieval_documents"][0]["coverage_aggregate"] is True


def test_explicit_page_marker_establishes_specificity_for_arabic_visitor_contact():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.adaptive_hybrid import QueryMode
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    query = (
        "إلى أي عنوان بريد إلكتروني يجب على الزوار التواصل إذا احتاجوا "
        "إلى متطلبات معينة للدخول؟"
    )
    contact_url = "https://mbzuai.ac.ae/ar/about/contact"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    retriever._coverage_page_records = [
        {
            "source_url": contact_url,
            "normalized_url": retriever._normalize_source_url(contact_url),
            "search_text": query.casefold(),
            "tokens": set(_tokenize(query)),
            "identity_text": "صفحة التواصل مع الجامعة",
            "identity_tokens": set(_tokenize("صفحة التواصل مع الجامعة")),
        }
    ]

    plan = retriever._infer_coverage_requirements(query, "exact_fact")

    assert plan["required_pages"] == [contact_url]
    assert plan["required_pages_source"] == "explicit_markers"

    runtime_plan = retriever._coverage_plan_for_result(
        query=query,
        payload={},
        mode=QueryMode.FACT,
    )

    assert runtime_plan["required_pages"] == [contact_url]
    assert runtime_plan["required_pages_source"] == "explicit_markers"


def test_routed_coverage_page_records_include_page_cards_and_ifm_domain():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        page_card_map={
            "ifm-about": {
                "id": "ifm-about",
                "source_url": "https://ifm.ai/about",
                "title": "About IFM",
                "page_type": "about",
                "purpose_summary": "Explains the Institute of Foundation Models.",
            }
        },
        evidence_span_map={},
        chunk_map={},
        summary_map={},
        parent_map={},
    )

    records = retriever._build_coverage_page_records()

    assert [record["source_url"] for record in records] == ["https://ifm.ai/about"]
    assert "about ifm" in records[0]["identity_text"]


def test_routed_coverage_path_markers_match_only_exact_page_and_language_variant():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._coverage_marker_matches(
        "/about/leadership",
        "https://mbzuai.ac.ae/about/leadership/",
    )
    assert retriever._coverage_marker_matches(
        "/about/leadership",
        "https://mbzuai.ac.ae/ar/about/leadership",
    )
    assert not retriever._coverage_marker_matches(
        "/about/leadership",
        "https://mbzuai.ac.ae/about/leadership/president",
    )


def test_routed_coverage_explicit_pages_prefer_arabic_language_variant():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    english_url = "https://mbzuai.ac.ae/about/leadership"
    arabic_url = "https://mbzuai.ac.ae/ar/about/leadership"
    retriever._coverage_page_records = [
        {
            "source_url": english_url,
            "normalized_url": retriever._normalize_source_url(english_url),
            "search_text": "leadership and governance",
            "tokens": set(_tokenize("leadership and governance")),
            "identity_text": "leadership and governance",
            "identity_tokens": set(_tokenize("leadership and governance")),
        },
        {
            "source_url": arabic_url,
            "normalized_url": retriever._normalize_source_url(arabic_url),
            "search_text": "القيادة والحوكمة",
            "tokens": set(_tokenize("القيادة والحوكمة")),
            "identity_text": "القيادة والحوكمة",
            "identity_tokens": set(_tokenize("القيادة والحوكمة")),
        },
    ]

    plan = retriever._infer_coverage_requirements(
        "ماذا تعرض صفحة القيادة والحوكمة؟",
        "exact_fact",
    )

    assert plan["required_pages"] == [arabic_url]


@pytest.mark.parametrize(
    ("query", "expected_url", "other_url"),
    [
        (
            "What eligibility age does the Human Phenotype Project page specify?",
            "https://hpp.mbzuai.ac.ae/",
            "https://hpp.mbzuai.ac.ae/privacy",
        ),
        (
            "What does the MBZUAI Library homepage show?",
            "https://library.mbzuai.ac.ae/",
            "https://library.mbzuai.ac.ae/newest-technology",
        ),
        (
            "What does the IFM about page say about IFM?",
            "https://ifm.ai/about",
            "https://mbzuai.ac.ae/about",
        ),
    ],
)
def test_routed_coverage_root_and_host_specific_markers_are_exact(query, expected_url, other_url):
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever._coverage_page_records = [
        {
            "source_url": expected_url,
            "normalized_url": retriever._normalize_source_url(expected_url),
            "search_text": query.casefold(),
            "tokens": set(_tokenize(query)),
            "identity_text": query.casefold(),
            "identity_tokens": set(_tokenize(query)),
        },
        {
            "source_url": other_url,
            "normalized_url": retriever._normalize_source_url(other_url),
            "search_text": query.casefold(),
            "tokens": set(_tokenize(query)),
            "identity_text": query.casefold(),
            "identity_tokens": set(_tokenize(query)),
        },
    ]

    plan = retriever._infer_coverage_requirements(query, "exact_fact")

    assert plan["required_pages"] == [expected_url]


def test_routed_page_target_score_prefers_exact_page_identity():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    query = "What does the Office of the Registrar page provide?"

    def page(identity: str, body: str) -> dict:
        return {
            "normalized_url": "https://mbzuai.ac.ae/student-resources/example",
            "search_text": body.casefold(),
            "tokens": set(_tokenize(body)),
            "identity_text": identity.casefold(),
            "identity_tokens": set(_tokenize(identity)),
        }

    exact = page("Office of the Registrar", "student records and registration services")
    generic = page("Student Resources", "Office of the Registrar student records and registration services")

    assert retriever._page_target_score(query, exact) > retriever._page_target_score(query, generic)


def test_routed_admissions_workflow_prefers_canonical_page_over_intake_news():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    query = "How do I apply for a PhD at MBZUAI?"

    def page(url: str, title: str, page_type: str) -> dict:
        search_text = f"{title} PhD admissions application apply"
        return {
            "normalized_url": url,
            "search_text": search_text.casefold(),
            "tokens": set(_tokenize(search_text)),
            "identity_text": title.casefold(),
            "identity_tokens": set(_tokenize(title)),
            "page_type": page_type,
        }

    canonical = page(
        "https://preprod.mbzuai.ac.ae/admissions/graduate-phd-admissions",
        "Graduate Ph.D. admissions",
        "admissions_or_program",
    )
    news = page(
        "https://preprod.mbzuai.ac.ae/knowledge-center/the-node/mbzuai-opens-admissions-fall-2026-intake",
        "MBZUAI opens admissions for Fall 2026 intake",
        "news_or_event",
    )

    assert retriever._explicit_required_page_markers(query)[0] == (
        "/admissions/graduate-phd-admissions"
    )
    assert retriever._page_target_score(query, canonical) > retriever._page_target_score(
        query, news
    )


def test_routed_coverage_disambiguates_projects_and_research_projects_pages():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    projects_url = "https://mbzuai.ac.ae/ar/projects"
    research_projects_url = "https://mbzuai.ac.ae/ar/research/projects"
    retriever._coverage_page_records = [
        {
            "source_url": projects_url,
            "normalized_url": retriever._normalize_source_url(projects_url),
            "search_text": "صفحة المشاريع موضوعات بحثية",
            "tokens": set(_tokenize("صفحة المشاريع موضوعات بحثية")),
            "identity_text": "المشاريع",
            "identity_tokens": set(_tokenize("المشاريع")),
        },
        {
            "source_url": research_projects_url,
            "normalized_url": retriever._normalize_source_url(research_projects_url),
            "search_text": "مراكز البحوث والمشاريع تطبيقات الذكاء الاصطناعي",
            "tokens": set(_tokenize("مراكز البحوث والمشاريع تطبيقات الذكاء الاصطناعي")),
            "identity_text": "البحوث المشاريع",
            "identity_tokens": set(_tokenize("البحوث المشاريع")),
        },
    ]

    scoped = retriever._infer_coverage_requirements(
        "ما أنواع الموضوعات البحثية التي تغطيها صفحة المشاريع؟",
        "scoped",
    )
    synthesis = retriever._infer_coverage_requirements(
        "كيف تساهم مراكز البحوث والمشاريع في تطوير تطبيقات الذكاء الاصطناعي؟",
        "broad_synthesis",
    )

    assert scoped["required_pages"] == [projects_url]
    assert synthesis["required_pages"] == [research_projects_url]


def test_routed_coverage_resolves_ifm_about_and_collaboration_pages():
    from pipeline.retrieval.adaptive_hybrid import _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    about_url = "https://ifm.ai/about"
    collaborate_url = "https://ifm.ai/collaborate"
    retriever._coverage_page_records = [
        {
            "source_url": about_url,
            "normalized_url": retriever._normalize_source_url(about_url),
            "search_text": "institute of foundation models global center of excellence",
            "tokens": set(_tokenize("institute of foundation models global center of excellence")),
            "identity_text": "about ifm institute of foundation models",
            "identity_tokens": set(_tokenize("about ifm institute of foundation models")),
        },
        {
            "source_url": collaborate_url,
            "normalized_url": retriever._normalize_source_url(collaborate_url),
            "search_text": "collaboration career opportunities join us",
            "tokens": set(_tokenize("collaboration career opportunities join us")),
            "identity_text": "collaborate institute of foundation models",
            "identity_tokens": set(_tokenize("collaborate institute of foundation models")),
        },
    ]

    plan = retriever._infer_coverage_requirements(
        "What is the Institute of Foundation Models at MBZUAI, and how does it describe its collaboration and career opportunities?",
        "broad_synthesis",
    )

    assert plan["required_pages"] == [about_url, collaborate_url]


def test_routed_required_page_backfill_bridges_page_alias_to_shared_chunks():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    requested_url = "https://mbzuai.ac.ae/ar/projects"
    canonical_url = "https://mbzuai.ac.ae/ar/research/projects"
    revision_id = "document-revision:projects"
    chunk_id = "chunk:c650:document-revision:projects:00001:topics"
    retriever.vector = SimpleNamespace(
        page_card_map={
            "requested": {
                "id": "page-card:requested",
                "source_url": requested_url,
                "document_revision_id": revision_id,
                "linked_chunk_ids": [chunk_id],
                "title": "المشاريع",
            },
            "canonical": {
                "id": "page-card:canonical",
                "source_url": canonical_url,
                "document_revision_id": revision_id,
                "linked_chunk_ids": [chunk_id],
                "title": "المشاريع",
            },
        },
        evidence_span_map={},
        fact_map={},
        summary_map={},
        parent_map={},
        chunk_map={
            chunk_id: {
                "id": chunk_id,
                "source_url": canonical_url,
                "document_revision_id": revision_id,
                "document_title": "المشاريع",
                "text": "تشمل الموضوعات البحثية التزييف العميق وتحليل الصور الطبية.",
            }
        },
        _score_text_match=lambda query, text: 1.0,
    )
    retriever._coverage_page_records = retriever._build_coverage_page_records()
    retriever._coverage_page_records_by_url = {
        page["normalized_url"]: page
        for page in retriever._coverage_page_records
    }
    retriever._coverage_record_indexes = retriever._build_coverage_record_indexes()
    assert [
        record["id"]
        for record in retriever._coverage_candidates_for_required_page(
            record_type="chunks",
            required_page=requested_url,
            source_map=retriever.vector.chunk_map,
        )
    ] == [chunk_id]
    payload = {
        "selected_chunk_ids": [],
        "selected_fact_ids": [],
        "selected_evidence_span_ids": [],
        "retrieval_documents": [],
        "abstained": False,
    }

    changed = retriever._augment_payload_for_required_coverage(
        query="ما الموضوعات التي تغطيها صفحة المشاريع؟",
        payload=payload,
        coverage_plan={"required_pages": [requested_url]},
    )

    assert changed is True
    assert payload["selected_chunk_ids"] == [chunk_id]
    assert payload["retrieval_documents"][0]["source_url"] == requested_url
    assert payload["retrieval_documents"][0]["canonical_url"] == canonical_url


def test_required_page_backfill_preserves_multilingual_dense_section_rank():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    page_url = "https://preprod.mbzuai.ac.ae/study/phd-programs/doctoral-robotics"
    revision_id = "document-revision:robotics"
    overview_id = f"chunk:c650:{revision_id}:00000:overview"
    outcomes_id = f"chunk:c650:{revision_id}:00001:outcomes"
    requirements_id = f"chunk:c650:{revision_id}:00002:requirements"
    retriever.vector = SimpleNamespace(
        page_card_map={
            "robotics": {
                "id": "page-card:robotics",
                "source_url": page_url,
                "document_revision_id": revision_id,
                "linked_chunk_ids": [overview_id, outcomes_id, requirements_id],
                "title": "Doctor of Philosophy in Robotics",
            }
        },
        evidence_span_map={},
        fact_map={},
        summary_map={},
        parent_map={},
        chunk_map={
            overview_id: {
                "id": overview_id,
                "source_url": page_url,
                "document_revision_id": revision_id,
                "section_heading": "Overview",
                "text": "The program prepares students for robotics research.",
            },
            outcomes_id: {
                "id": outcomes_id,
                "source_url": page_url,
                "document_revision_id": revision_id,
                "section_heading": "Program learning outcomes",
                "text": "Graduates formulate and solve robotics research problems.",
            },
            requirements_id: {
                "id": requirements_id,
                "source_url": page_url,
                "document_revision_id": revision_id,
                "section_heading": "Completion requirements",
                "text": "The minimum degree requirement is 60 credits.",
            },
        },
        # Arabic query against English-only page content has no lexical overlap.
        _score_text_match=lambda query, text: 0.0,
    )
    retriever._coverage_page_records = retriever._build_coverage_page_records()
    retriever._coverage_page_records_by_url = {
        page["normalized_url"]: page
        for page in retriever._coverage_page_records
    }
    retriever._coverage_record_indexes = retriever._build_coverage_record_indexes()
    payload = {
        "dense_chunk_ids": [requirements_id, outcomes_id],
        "selected_chunk_ids": [],
        "selected_fact_ids": [],
        "selected_evidence_span_ids": [],
        "retrieval_documents": [],
        "abstained": False,
    }

    changed = retriever._augment_payload_for_required_coverage(
        query="كم عدد الساعات المعتمدة الدنيا لدرجة الدكتوراه في الروبوتات؟",
        payload=payload,
        coverage_plan={"required_pages": [page_url]},
    )

    assert changed is True
    assert payload["retrieval_documents"][0]["id"] == requirements_id
    assert payload["selected_chunk_ids"][0] == requirements_id


def test_routed_required_page_backfill_injects_complete_parent_for_list_query():
    from pipeline.retrieval.evidence_packer import build_evidence_pack
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    page_url = "https://library.mbzuai.ac.ae/newest-technology"
    revision_id = "document-revision:library-news"
    parent_id = f"parent:c650:{revision_id}:page"
    retriever.vector = SimpleNamespace(
        page_card_map={
            "page": {
                "id": "page-card:library-news",
                "source_url": page_url,
                "document_revision_id": revision_id,
                "title": "News on AI and Technology",
            }
        },
        evidence_span_map={},
        fact_map={},
        summary_map={},
        chunk_map={},
        parent_map={
            parent_id: {
                "id": parent_id,
                "source_url": page_url,
                "document_revision_id": revision_id,
                "document_title": "News on AI and Technology",
                "text": (
                    "Child-monitoring apps might need a reboot. "
                    "We still don't know how people are really using AI. "
                    "The role of the astronaut is in flux. See 15 more."
                ),
            }
        },
        _score_text_match=lambda query, text: 1.0,
    )
    retriever._coverage_page_records = retriever._build_coverage_page_records()
    payload = {
        "selected_chunk_ids": [],
        "selected_parent_ids": [],
        "selected_fact_ids": [],
        "selected_evidence_span_ids": [],
        "retrieval_documents": [],
        "abstained": False,
    }
    query = "What recent MIT Technology Review items are shown?"

    changed = retriever._augment_payload_for_required_coverage(
        query=query,
        payload=payload,
        coverage_plan={"intent": "scoped", "required_pages": [page_url]},
    )
    pack = build_evidence_pack(
        query=query,
        result=payload,
        max_items=4,
        max_chars=4000,
        max_per_source=2,
        coverage_plan={"intent": "scoped", "required_pages": [page_url]},
    )

    assert changed is True
    assert payload["selected_parent_ids"][0] == parent_id
    assert pack["items"][0]["id"] == parent_id
    assert "Child-monitoring apps might need a reboot" in pack["items"][0]["text"]
    assert "See 15 more" in pack["items"][0]["text"]


def test_evidence_pack_reserves_complete_collection_parent_before_list_fragments():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    page_url = "https://example.edu/study/masters"
    complete_inventory = (
        "Master's programs: Applied AI; Computational Biology; Computer Science; "
        "Computer Vision; Machine Learning; Natural Language Processing; Robotics; "
        "Statistics and Data Science."
    )
    result = {
        "retrieval_documents": [
            {
                "id": "chunk-admissions",
                "text": "Graduate admissions require an accredited degree.",
                "source_url": page_url,
            },
            {
                "id": "parent:masters:page",
                "text": complete_inventory,
                "source_url": page_url,
                "coverage_aggregate": True,
            },
        ]
    }

    pack = build_evidence_pack(
        query="List all master's programs.",
        result=result,
        max_items=2,
        max_chars=2000,
        max_per_source=2,
        coverage_plan={
            "intent": "multi_page_aggregation",
            "required_pages": [page_url],
        },
    )

    assert pack["items"][0]["id"] == "parent:masters:page"
    assert pack["items"][0]["text"] == complete_inventory


def test_evidence_pack_reserves_leaf_chunk_for_required_multi_detail_page():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    page_url = "https://mbzuai.ac.ae/ar/news/career-fair"
    parent_text = (
        "معرض التدريب المهني وفرص العمل وفر للطلبة دعماً مهنياً مباشراً. "
        + ("تفاصيل عامة عن المعرض. " * 180)
    )
    result = {
        "retrieval_documents": [
            {
                "id": "parent:c650:career-fair:page",
                "text": parent_text,
                "source_url": page_url,
                "coverage_aggregate": True,
            },
            {
                "id": "chunk:c650:career-fair:00000",
                "text": (
                    "مكن المعرض أكثر من 160 طالباً وطالبة من التواصل مع "
                    "48 من شركاء الجامعة والحصول على دعم مهني."
                ),
                "source_url": page_url,
            },
        ],
        "fact_documents": [
            {
                "id": f"fact-{index}",
                "text": "وفر المعرض فرصاً ودعماً مهنياً للطلبة.",
                "source_url": page_url,
            }
            for index in range(4)
        ],
    }

    pack = build_evidence_pack(
        query="ما الذي جعل المعرض مهماً للطلبة، وما الدعم الذي وفره لهم؟",
        result=result,
        max_items=4,
        max_chars=8000,
        max_per_source=2,
        coverage_plan={
            "intent": "scoped",
            "required_pages": [page_url],
            "required_entities": [],
            "required_sections": [],
        },
    )

    packed_ids = [item["id"] for item in pack["items"]]
    assert "parent:c650:career-fair:page" in packed_ids
    assert "chunk:c650:career-fair:00000" in packed_ids
    assert any("48 من شركاء الجامعة" in item["text"] for item in pack["items"])


def test_answer_readiness_arabic_term_matching_handles_clitics_and_pronouns():
    from pipeline.evaluation.answer_readiness import _required_term_supported

    assert _required_term_supported(
        "يضمن المكتب تزويد الطلاب بأفضل الخدمات طوال فترة دراستهم.",
        "أفضل الخدمات طيلة فترة دراستهم",
    )
    assert _required_term_supported(
        "تسعى الجامعة إلى مواصلة النمو وتوسيع نطاق العمل.",
        "توسيع نطاق عملنا",
    )
    assert _required_term_supported(
        "كما يتعاون أعضاء الهيئة التدريسية مع الشركاء.",
        "التعاون",
    )
    assert _required_term_supported(
        "تشمل المنحة مخصصًا شهريًا مجزيًا.",
        "مخصص شهري",
    )
    assert _required_term_supported(
        "درجة الماجستير مفضلة وغير إلزامي.",
        "غير إلزامية",
    )
    assert _required_term_supported(
        "مدة البرنامج عامان بدوام جزئي.",
        "سنتين",
    )


def test_answer_readiness_term_matching_handles_cross_language_dates_and_safe_variants():
    from pipeline.evaluation.answer_readiness import _required_term_supported

    assert _required_term_supported(
        "The talk is scheduled for July 22, 2026 at 11:00 AM.",
        "22 يوليو 2026",
    )
    assert _required_term_supported(
        "The Board helps interpret the institution to the public.",
        "interpreting the institution to the public",
    )
    assert _required_term_supported(
        "The META Wall is a multi-purpose 270-degree LED space.",
        "270 LED space",
    )
    assert _required_term_supported(
        "The action opens the applicant portal login page.",
        "login/start page",
    )


def test_routed_static_page_markers_choose_canonical_language_routes_and_avoid_program_noise():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._explicit_required_page_markers(
        "According to the leadership and mission pages, what is MBZUAI's mission and vision?"
    ) == ["/about/leadership", "/about/mission-and-vision"]
    assert retriever._explicit_required_page_markers(
        "Where are IFM's headquarters and research hubs located?"
    ) == ["https://ifm.ai/about"]
    assert retriever._explicit_required_page_markers(
        "Who can apply for onsite access to the MBZUAI Library's resources by email?"
    ) == ["https://library.mbzuai.ac.ae/the-library"]
    assert retriever._explicit_required_page_markers(
        "كيف يصف قسم تعلّم الآلة مجالات تركيزه الرئيسة وما الذي يقدمه للطلاب؟"
    ) == ["/ar/research-department/machine-learning-department"]
    assert retriever._explicit_required_page_markers(
        "اذكر مثالاً واحداً على موضوع مشروع بحثي مذكور في صفحة المشاريع."
    ) == ["/research/projects"]
    assert retriever._explicit_required_page_markers(
        "ما هي الرسالة التي تذكرها الجامعة، وكيف تربطها القيادة بخدمة دولة الإمارات؟"
    ) == ["/about/leadership", "/about/mission"]
    research_markers = retriever._explicit_required_page_markers(
        "How do the research centers and projects pages describe their work?"
    )
    assert "/research/research-centers" in research_markers
    assert "/research-centers" not in research_markers
    funding_markers = retriever._explicit_required_page_markers(
        "What funding differences are described between the M.Sc. and Ph.D. programs?"
    )
    assert "/ai-programs" not in funding_markers
    assert "mbzuai_faculty_brochure" not in funding_markers


def test_routed_static_page_markers_cover_named_multisource_arabic_routes():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    ifm_markers = retriever._explicit_required_page_markers(
        "ما الذي يصفه موقع IFM بأنه يسعى إلى بنائه مع الشركاء، وأين يقع مقره الرئيسي وما المدن التي لديه فيها مراكز أبحاث؟"
    )
    assert ifm_markers == ["https://ifm.ai/about", "https://ifm.ai/collaborate"]

    career_markers = retriever._explicit_required_page_markers(
        "ما هي أقسام الوظائف المفتوحة في صفحة الوظائف، وما هي مراكز MBZUAI الثلاثة المذكورة فيها؟"
    )
    assert career_markers == [
        "https://careers.mbzuai.ac.ae",
        "https://careers.mbzuai.ac.ae/vacancies",
    ]

    governance_markers = retriever._explicit_required_page_markers(
        "ما الذي تقوله صفحة القيادة والحوكمة، وما الذي توضحه وثيقة الحوكمة عن اللجان؟"
    )
    assert governance_markers == ["/about/leadership", "governance_structure.pdf"]

    degree_markers = retriever._explicit_required_page_markers(
        "ما المؤهلات والتوجه المهني المطلوبان من الخريجين للالتحاق ببرامج الماجستير والدكتوراه؟"
    )
    assert degree_markers == ["/study/msc-programs", "/study/phd-programs"]

    service_markers = retriever._explicit_required_page_markers(
        "ما الخدمات التي يقدمها مكتب التسجيل وفريق الخدمات المهنية والتدريب؟"
    )
    assert service_markers == [
        "/student-resources/office-of-the-registrar",
        "/student-resources/student-careers-and-internships",
    ]


def test_routed_static_page_markers_cover_time_specific_named_pages():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._explicit_required_page_markers(
        "كم عدد الطلاب الجدد الذين استقبلتهم الجامعة في العام الأكاديمي الجديد؟"
    ) == [
        "welcomes-400-students-including-inaugural-undergraduate-cohort"
    ]
    assert retriever._explicit_required_page_markers(
        "When is Xiang Meng's upcoming AI talk scheduled?"
    ) == ["https://ai-nexus.mbzuai.ac.ae/previous-ai-talks"]
    assert retriever._explicit_required_page_markers(
        "According to the Human Phenotype Project pages, who can participate, what is the study duration, and what are its goals?"
    ) == [
        "https://hpp.mbzuai.ac.ae",
        "/news/new-human-phenotype-project-findings-illuminate-pathways-to-precision-medicine",
    ]


def test_routed_static_page_markers_cover_named_answer_evidence_pages():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._explicit_required_page_markers(
        "ما الهدف الذي أُنشئ من أجله مركز الذكاء الاصطناعي التكاملي (CIAI)؟"
    ) == ["/research/research-centers/ciai"]
    assert retriever._explicit_required_page_markers(
        "ما مجالات اهتمامات البروفيسورة دانييلا روس البحثية؟"
    ) == ["/about/leadership/daniela-rus"]
    assert retriever._explicit_required_page_markers(
        "ما الدعم الذي وفره معرض التدريب المهني وفرص العمل للطلبة؟"
    ) == [
        "/news/mbzuai-students-connect-with-industry-partners-to-secure-internship-and-career-opportunities"
    ]
    assert retriever._explicit_required_page_markers(
        "ما مدة الدراسة وفرص المنح وشروط القبول في برنامج البكالوريوس؟"
    ) == ["/study/mbzuai-undergraduate", "/study/ug-admission-process"]

    assert retriever._explicit_required_page_markers(
        "What does the MBZUAI Visitor Program say visitors can get hands-on access to?"
    ) == ["https://research.mbzuai.ac.ae/visitor-program"]
    assert retriever._explicit_required_page_markers(
        "How does MBZUAI engage with industry and capture value?"
    ) == ["https://research.mbzuai.ac.ae/partnerships-and-engagements"]
    assert retriever._explicit_required_page_markers(
        "What are the META Wall uses and what GPU options does the Metaverse Center offer?"
    ) == ["https://metaverse.mbzuai.ac.ae/studio"]


def test_routed_static_page_markers_cover_multi_aspect_admissions_and_library_pages():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._explicit_required_page_markers(
        "What academic and documentation requirements apply to undergraduate applicants, including English proficiency and the application fee?"
    ) == ["/study/ug-admission-process", "/study/undergraduate-program"]
    assert retriever._explicit_required_page_markers(
        "I am visiting the MBZUAI Library as a researcher resident in the UAE. What visitor access can I request, who can borrow materials and licensed electronic resources, and where are physical resources discoverable?"
    ) == [
        "https://library.mbzuai.ac.ae/visitor-information",
        "https://library.mbzuai.ac.ae/Borrowing_Information",
    ]


def test_routed_coverage_planner_cleans_command_prefix_entities_and_arabic_static_pages():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._clean_required_entity_phrase("Explain MBZUAI") == ""
    assert retriever._clean_required_entity_phrase("Explain MBZUAI Institutional Identity") == "Institutional Identity"
    assert retriever._clean_required_entity_phrase("Using MBZUAI's campus facilities page") == "campus facilities page"
    assert retriever._english_query_page_allowed(
        "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/12/Summarized_MBZUAI-Factsheet-Nov-2025_FINAL-ARB-1.docx"
    ) is False
    identity_markers = retriever._explicit_required_page_markers(
        "Explain MBZUAI's institutional identity and what law established it."
    )
    assert "university-catalogue-2024-2025" in identity_markers
    contact_markers = retriever._explicit_required_page_markers("How can I contact admissions?")
    assert "university-catalogue-2024-2025" in contact_markers
    campus_markers = retriever._explicit_required_page_markers("Outline the core campus amenities and support facilities at MBZUAI.")
    assert "/student-resources/campus-facilities" in campus_markers
    assert "/study/undergraduate-application-submission" in campus_markers
    assert retriever._query_has_specific_target("How can I contact MBZUAI admissions?")
    mbzuai_contact_markers = retriever._explicit_required_page_markers("How can I contact MBZUAI admissions?")
    assert "university-catalogue-2024-2025" in mbzuai_contact_markers
    assert "online-screening-exam-instructions" in mbzuai_contact_markers
    parking_markers = retriever._explicit_required_page_markers("Where is parking permitted at the Masdar City campus?")
    assert "university-catalogue-2024-2025" in parking_markers
    screening_markers = retriever._explicit_required_page_markers(
        "Explain the MBZUAI online screening exam process and how applicants can contact admissions if they need help."
    )
    assert "online-screening-exam-instructions" in screening_markers
    specialization_markers = retriever._explicit_required_page_markers(
        "What five core AI specializations does MBZUAI offer in its M.Sc. and Ph.D. programs?"
    )
    assert "mbzuai_faculty_brochure" in specialization_markers


def test_routed_coverage_plan_allows_exact_named_faculty_arabic_profile_for_english_query():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    arabic_faculty_url = "https://mbzuai.ac.ae/ar/study/faculty/haiyan-huang"
    program_url = "https://mbzuai.ac.ae/study/phd-programs/doctor-of-philosophy-in-computer-vision"
    retriever._coverage_page_records = [
        {
            "source_url": program_url,
            "normalized_url": retriever._normalize_source_url(program_url),
            "search_text": "computer vision phd program faculty research interests",
            "tokens": set(_tokenize("computer vision phd program faculty research interests")),
        },
        {
            "source_url": arabic_faculty_url,
            "normalized_url": retriever._normalize_source_url(arabic_faculty_url),
            "search_text": "haiyan huang faculty profile research interests computer vision medical image analysis",
            "tokens": set(_tokenize("haiyan huang faculty profile research interests computer vision medical image analysis")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query="What are Professor Haiyan Huang's research interests?",
        payload={"selected_chunk_ids": ["faculty"]},
        mode=QueryMode.FACT,
    )

    assert retriever._english_query_page_allowed(
        arabic_faculty_url,
        query="What are Professor Haiyan Huang's research interests?",
    )
    assert arabic_faculty_url in plan["required_pages"]
    assert program_url not in plan["required_pages"]


def test_evidence_pack_replaces_hash_like_document_titles_with_source_slug():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    pack = build_evidence_pack(
        query="Where is MBZUAI located?",
        result={
            "evidence_span_documents": [
                {
                    "id": "contact-span",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "source_url": "https://mbzuai.ac.ae/about/contact/",
                    "document_title": "63a930d41e1c36fb0689eb723cceaa4af709acd0",
                    "authority_class": "official",
                }
            ]
        },
        max_items=2,
        max_chars=1000,
        max_per_source=2,
        coverage_plan={"intent": "exact_fact"},
    )

    assert pack["items"][0]["document_title"] == "Contact"
    assert pack["citations"][0]["document_title"] == "Contact"


def test_release_stats_prefers_finalized_bundle_over_stale_build(tmp_path):
    from pipeline.core import release

    build_dir = tmp_path / "stage_outputs" / "build_retrieval_bundle"
    finalize_dir = tmp_path / "stage_outputs" / "finalize_retrieval_bundle"
    build_dir.mkdir(parents=True)
    finalize_dir.mkdir(parents=True)
    atomic_write_json(
        build_dir / "retrieval_bundle.json",
        {"version": 4, "stats": {"chunk_count": 1, "evidence_span_count": 1}},
    )
    finalized_bundle = finalize_dir / "retrieval_bundle.json"
    atomic_write_json(
        finalized_bundle,
        {"version": 5, "stats": {"chunk_count": 7, "evidence_span_count": 9}},
    )

    stats = release._load_retrieval_bundle_stats(tmp_path)

    assert stats["retrieval_bundle_file"] == str(finalized_bundle)
    assert stats["retrieval_bundle_version"] == 5
    assert stats["chunk_count"] == 7
    assert stats["evidence_span_count"] == 9


def test_answer_generation_reference_maps_prefer_finalized_bundle(tmp_path):
    from pipeline.evaluation.answer_generation import _bundle_maps

    build_dir = tmp_path / "stage_outputs" / "build_retrieval_bundle"
    finalize_dir = tmp_path / "stage_outputs" / "finalize_retrieval_bundle"
    build_dir.mkdir(parents=True)
    finalize_dir.mkdir(parents=True)
    atomic_write_json(
        build_dir / "retrieval_bundle.json",
        {
            "chunk_records": [{"id": "stale-chunk", "text": "stale"}],
            "parent_records": [],
            "media_records": [],
        },
    )
    atomic_write_json(
        finalize_dir / "retrieval_bundle.json",
        {
            "chunk_records": [{"id": "final-chunk", "text": "final"}],
            "parent_records": [{"id": "final-parent", "text": "parent"}],
            "media_records": [{"id": "final-media", "text": "media"}],
        },
    )

    maps = _bundle_maps(tmp_path)

    assert "final-chunk" in maps["chunks"]
    assert "final-parent" in maps["parents"]
    assert "final-media" in maps["media"]
    assert "stale-chunk" not in maps["chunks"]


def test_synthesis_expansion_breaks_equal_section_scores_by_seed_order():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_context_chunks = 8
    retriever.max_parent_chunks = 1
    retriever.same_parent_expand_threshold = 1
    retriever.parent_candidate_top_k = 2
    retriever.chunk_map = {
        "seed-z": {"id": "seed-z"},
        "seed-a": {"id": "seed-a"},
    }
    retriever.parent_map = {
        "section:z-first": {"parent_type": "section"},
        "section:a-second": {"parent_type": "section"},
    }
    section_by_chunk = {
        "seed-z": ["section:z-first"],
        "seed-a": ["section:a-second"],
    }
    calls = []

    def parent_ids(chunk_id, *, parent_type=None):
        return section_by_chunk.get(chunk_id, []) if parent_type == "section" else []

    def rank_children(_query, parent_id, *, top_k):
        calls.append((parent_id, top_k))
        return [f"child:{parent_id}"]

    retriever._parent_ids_for_chunk = parent_ids
    retriever._rank_parent_child_chunk_ids = rank_children
    retriever._expand_fact = lambda values: list(values)

    expanded = retriever._expand_scoped_or_synthesis(
        ["seed-z", "seed-a"],
        query="Compare the two pages",
        mode=QueryMode.SYNTHESIS,
    )

    assert [parent_id for parent_id, _top_k in calls] == [
        "section:z-first",
        "section:a-second",
    ]
    assert expanded == [
        "seed-z",
        "seed-a",
        "child:section:z-first",
        "child:section:a-second",
    ]


def test_scoped_expansion_falls_back_to_page_when_no_section_expands():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_context_chunks = 8
    retriever.max_parent_chunks = 2
    retriever.same_parent_expand_threshold = 1
    retriever.parent_candidate_top_k = 2
    retriever.chunk_map = {"seed": {"id": "seed"}}
    retriever.parent_map = {"page:profile": {"parent_type": "page"}}

    def parent_ids(_chunk_id, *, parent_type=None):
        if parent_type == "section":
            return []
        if parent_type == "page":
            return ["page:profile"]
        return []

    retriever._parent_ids_for_chunk = parent_ids
    retriever._rank_parent_child_chunk_ids = (
        lambda _query, parent_id, *, top_k: ["page-child"]
        if parent_id == "page:profile"
        else []
    )
    retriever._expand_fact = lambda values: list(values)

    expanded = retriever._expand_scoped_or_synthesis(
        ["seed"],
        query="Tell me about this profile",
        mode=QueryMode.SCOPED,
    )

    assert expanded == ["seed", "page-child"]


def test_adaptive_retriever_records_lane_timings():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.parallel_lane_workers = 2
    retriever.persistent_lane_executors = True
    retriever.split_lane_executors = True
    retriever.parallel_remote_lane_workers = 2
    retriever.parallel_local_lane_workers = 2
    retriever.namespace_chunks = "chunks"
    retriever.namespace_assertions = "assertions"
    retriever.parent_map = {}
    retriever.summary_map = {}
    retriever.media_map = {}
    retriever.fact_map = {}
    retriever.evidence_span_map = {}
    retriever._dense_query_ids = lambda **_kwargs: ["dense-chunk"]
    retriever._sparse_query_ids = lambda **_kwargs: []
    retriever._local_chunk_query_ids = lambda **_kwargs: ["local-chunk"]
    retriever._local_answer_query_ids = lambda **_kwargs: []

    lanes = retriever._run_query_lanes(
        query="campus facilities",
        query_vector=[0.1],
        mode=QueryMode.SCOPED,
        lane_top_ks={
            "chunk_dense": 1,
            "chunk_sparse": 0,
            "chunk_local": 1,
            "assertion_dense": 0,
            "assertion_sparse": 0,
            "answer_local": 0,
            "parent_dense": 0,
            "parent_sparse": 0,
            "parent_local": 0,
            "summary_dense": 0,
            "summary_sparse": 0,
            "media_dense": 0,
            "media_sparse": 0,
            "media_local": 0,
            "fact_dense": 0,
            "fact_sparse": 0,
            "fact_local": 0,
            "evidence_span_dense": 0,
            "evidence_span_sparse": 0,
            "evidence_span_local": 0,
        },
    )

    assert lanes["chunk_dense_ids"] == ["dense-chunk"]
    assert lanes["local_chunk_ids"] == ["local-chunk"]
    assert set(retriever._last_lane_latency_ms) == {"chunk_dense_ids", "local_chunk_ids"}


def test_routed_retriever_route_reuses_relation_plan():
    from pipeline.retrieval.adaptive_hybrid import QueryMode
    from pipeline.retrieval.graph_rag import RelationQueryPlan
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.routed_graph_enabled = True
    retriever.parallel_graph_enabled = True
    retriever.graph = object()
    calls = {"count": 0}
    plan = RelationQueryPlan(
        family="contact",
        confidence=0.77,
        primary_relation_types=("RELATED_TO",),
        secondary_relation_types=("MENTIONS",),
        alias_tokens=("admission", "email"),
        entity_tokens=("admission",),
        graph_first=False,
    )

    def fake_plan(query):
        calls["count"] += 1
        return plan

    retriever._graph_relation_plan = fake_plan
    decision = retriever._route_query("What is the admissions email?")

    assert calls["count"] == 1
    assert decision.backend == "parallel_hybrid"
    assert decision.query_mode == QueryMode.FACT.value
    assert decision.relation_plan is plan
    assert decision.relation_family == "contact"


def test_adaptive_retriever_skips_support_parent_scan_for_fact_queries():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)

    assert not retriever._should_use_support_parent_scan(
        "What are the IT support working hours for the MBZUAI online screening exam?",
        mode=QueryMode.FACT,
    )
    assert retriever._should_use_support_parent_scan(
        "Give a detailed guide to IT support and preparation for the MBZUAI online screening exam.",
        mode=QueryMode.SCOPED,
    )


def test_relabel_v5_uses_screening_exam_pdf_for_pdf_support_hours_query():
    from pipeline.evaluation.dataset import EvalExample

    module = _load_script_module("scripts/relabel_eval_dataset_v5.py")
    example = EvalExample(
        id="fact-009",
        query="What are the IT support working hours for the MBZUAI online screening exam?",
        query_type="fact",
        source_type="pdf",
        reference_answer="IT support is available Monday to Thursday and Friday.",
    )

    urls = module._canonical_reference_urls(example)

    assert urls == [
        "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf"
    ]


def test_relabel_v5_uses_current_campus_map_pdf_for_pdf_map_query():
    from pipeline.evaluation.dataset import EvalExample

    module = _load_script_module("scripts/relabel_eval_dataset_v5.py")
    example = EvalExample(
        id="multimodal-001",
        query="What does the MBZUAI campus map show about the campus layout and facilities?",
        query_type="multimodal",
        source_type="pdf",
        reference_answer="The campus map shows buildings and facilities.",
    )

    urls = module._canonical_reference_urls(example)

    assert urls[0] == "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/11/MBZUAI_Campus_Map_V1044331768.pdf"


def test_relabel_v5_distinguishes_undergraduate_admissions_from_general_admissions():
    from pipeline.evaluation.dataset import EvalExample

    module = _load_script_module("scripts/relabel_eval_dataset_v5.py")
    undergraduate_only = EvalExample(
        id="fact-028",
        query="What is the undergraduate admissions email address?",
        query_type="fact",
        source_type="webpage",
        reference_answer="The undergraduate admissions email is ug.admission@mbzuai.ac.ae.",
    )
    comparison = EvalExample(
        id="deep-002",
        query="Compare the right contact paths for general admissions and undergraduate admissions.",
        query_type="synthesis",
        source_type="mixed",
        reference_answer="General admissions uses admission@mbzuai.ac.ae and undergraduate admissions uses ug.admission@mbzuai.ac.ae.",
    )

    assert module._canonical_reference_urls(undergraduate_only) == [
        "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    ]
    assert module._canonical_reference_urls(comparison) == [
        "https://mbzuai.ac.ae/study/undergraduate-application-submission",
        "https://mbzuai.ac.ae/about/faq",
    ]


def test_relabel_v5_routes_current_shuttle_questions_to_contact_transport_evidence():
    from pipeline.evaluation.dataset import EvalExample

    module = _load_script_module("scripts/relabel_eval_dataset_v5.py")
    example = EvalExample(
        id="fact-007",
        query="Does MBZUAI provide a shuttle bus service?",
        query_type="fact",
        source_type="webpage",
        reference_answer="Yes. MBZUAI provides a shuttle service.",
        metadata={"answer_must_include": ["shuttle", "students"]},
    )

    assert module._canonical_reference_urls(example) == ["https://mbzuai.ac.ae/about/contact"]
    assert module._curated_answer_must_include(example, ["shuttle", "students"]) == ["NAVYA bus", "if available"]
    assert "does not verify" in module._curated_reference_answer(example)


def test_release_vector_expected_counts_account_for_sparse_skips(tmp_path):
    from pipeline.core import release

    manifest = _valid_modern_vector_manifest()
    dense_counts = {
        "chunks": 10,
        "parents": 3,
        "media": 0,
        "facts": 8,
        "evidence_spans": 0,
        "summaries": 2,
        "assertions": 2,
        "entities": 0,
        "communities": 0,
    }
    sparse_counts = {f"sparse_{key}": value for key, value in dense_counts.items()}
    sparse_counts["sparse_facts"] = 6
    manifest["planned"] = {**dense_counts, **sparse_counts}
    manifest["uploaded"] = {**dense_counts, **sparse_counts}
    manifest["sparse"] = {"record_stats": {"fact_records_skipped": 2}}
    manifest["verification"] = {
        "dense": {
            "expected": {key: value for key, value in dense_counts.items() if value},
            "actual": {key: value for key, value in dense_counts.items() if value},
            "failures": [],
        },
        "sparse": {
            "expected": {
                key.removeprefix("sparse_"): value
                for key, value in sparse_counts.items()
                if value
            },
            "actual": {
                key.removeprefix("sparse_"): value
                for key, value in sparse_counts.items()
                if value
            },
            "failures": [],
        },
    }
    bundle_stats = {
        "chunk_count": 10,
        "parent_count": 3,
        "media_count": 0,
        "fact_count": 8,
        "summary_count": 2,
        "assertion_count": 2,
    }

    expected = release._vector_expected_counts_for_manifest(manifest, bundle_stats, tmp_path)

    assert expected["facts"] == 8
    assert expected["sparse_facts"] == 6
    assert release._verify_vector_manifest(manifest, bundle_stats, tmp_path) == []


def test_release_vector_counts_preserve_explicitly_disabled_dense_lane(tmp_path):
    from pipeline.core import release

    manifest = _valid_modern_vector_manifest()
    manifest["planned"]["facts"] = 0
    manifest["uploaded"]["facts"] = 0
    manifest["verification"]["dense"]["expected"].pop("facts")
    manifest["verification"]["dense"]["actual"].pop("facts")
    bundle_stats = {
        "chunk_count": 10,
        "parent_count": 3,
        "media_count": 0,
        "fact_count": 4,
        "evidence_span_count": 5,
        "summary_count": 2,
        "assertion_count": 2,
    }

    errors = release._verify_vector_manifest(
        manifest,
        bundle_stats,
        tmp_path,
        {"enable_dense_facts": False},
    )

    assert errors == []
    assert release._modern_vector_expected_counts_for_manifest(
        manifest,
        bundle_stats,
        {"enable_dense_facts": False},
    )["facts"] == 0
    assert release._modern_vector_expected_counts_for_manifest(
        manifest,
        bundle_stats,
        {"enable_dense_facts": False},
    )["sparse_facts"] == 4


def test_evidence_pack_orders_dedupes_and_caps_by_source():
    from pipeline.retrieval.evidence_packer import build_evidence_pack, score_retrieval_confidence

    result = {
        "selected_chunk_ids": ["c1", "c2"],
        "dense_chunk_ids": ["c1"],
        "sparse_chunk_ids": ["c1", "c2"],
        "answer_documents": [
            {
                "id": "a1",
                "text": "The admissions email is admissions@example.edu.",
                "source_url": "https://mbzuai.ac.ae/study",
                "document_title": "Admissions",
                "confidence": 0.91,
                "authority_score": 0.95,
            }
        ],
        "fact_documents": [
            {
                "id": "f1",
                "text": "Applications are handled by the admissions office.",
                "source_url": "https://mbzuai.ac.ae/study",
            }
        ],
        "retrieval_documents": [
            {"id": "c1", "text": "Applications are handled by the admissions office.", "source_url": "https://mbzuai.ac.ae/study"},
            {"id": "c2", "text": "Campus details live on the student resources page.", "source_url": "https://mbzuai.ac.ae/student-resources"},
        ],
    }

    pack = build_evidence_pack(query="admissions contact", result=result, max_items=3, max_chars=180, max_per_source=2)
    assert [item["kind"] for item in pack["items"]] == ["fact", "chunk", "chunk"]
    assert pack["items"][0]["id"] == "f1"
    assert pack["budget"]["used_items"] == 3

    confidence, factors = score_retrieval_confidence(result)
    assert confidence > 0.6
    assert factors["lane_agreement"] > 0


def test_evidence_pack_prioritizes_required_official_source_over_noisy_answer():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    result = {
        "answer_documents": [
            {
                "id": "bad-answer",
                "text": "A related article says MBZUAI welcomed undergraduate students.",
                "source_url": "https://mbzuai.ac.ae/news/related",
            },
            {
                "id": "blank-answer",
                "text": "A blank-source document mentions scholarships at another university.",
                "source_url": "",
            },
        ],
        "evidence_span_documents": [
            {
                "id": "span-scholarship",
                "text": "Undergraduate scholarships are awarded based on merit with no separate application.",
                "source_url": "https://mbzuai.ac.ae/study/undergraduate-application-submission",
                "document_title": "Undergraduate Application Submission",
                "span_type": "policy",
            }
        ],
    }

    pack = build_evidence_pack(
        query="Does MBZUAI offer undergraduate scholarships?",
        result=result,
        coverage_plan={
            "required_pages": ["https://mbzuai.ac.ae/study/undergraduate-application-submission"],
            "required_entities": ["scholarships"],
        },
        max_items=3,
        max_chars=1000,
        max_per_source=2,
    )

    assert pack["coverage_status"] == "complete"
    assert pack["items"][0]["id"] == "span-scholarship"
    assert pack["items"][0]["source_url"] == "https://mbzuai.ac.ae/study/undergraduate-application-submission"


def test_evidence_pack_prefers_exact_admissions_contact_span_within_required_catalogue():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    catalogue_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/05/University-Catalogue-2024-2025.pdf"
    result = {
        "evidence_span_documents": [
            {
                "id": "catalogue-generic",
                "text": "Transportation bus services, routes, and fare taxis are available for students.",
                "source_url": catalogue_url,
            },
            {
                "id": "catalogue-contact",
                "text": "For inquiries, please find below the list of contacts: Admission admission@mbzuai.ac.ae Registrar registrar@mbzuai.ac.ae.",
                "source_url": catalogue_url,
            },
            {
                "id": "emergency-contact",
                "text": "Emergency contact numbers include MBZUAI MANAGEMENT.Contact Number = +971 50 443 5552.",
                "source_url": "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2024/07/Emergency-Response-Plan.pdf",
            },
        ],
    }

    pack = build_evidence_pack(
        query="How can I contact admissions?",
        result=result,
        coverage_plan={"intent": "exact_fact", "required_pages": [catalogue_url]},
        max_items=2,
        max_chars=1000,
        max_per_source=2,
    )

    packed_ids = [item["id"] for item in pack["items"]]
    assert packed_ids[0] == "catalogue-contact"
    assert "emergency-contact" not in packed_ids


def test_evidence_pack_prefers_exact_parking_and_screening_hours_spans():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    catalogue_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/05/University-Catalogue-2024-2025.pdf"
    parking_pack = build_evidence_pack(
        query="Where is parking permitted at the Masdar City campus?",
        result={
            "evidence_span_documents": [
                {
                    "id": "catalogue-transport",
                    "text": "Transportation bus services and taxis are available around Abu Dhabi.",
                    "source_url": catalogue_url,
                },
                {
                    "id": "catalogue-parking",
                    "text": "Parking At the Masdar City campus, parking is permitted at the North Car Park.",
                    "source_url": catalogue_url,
                },
            ]
        },
        coverage_plan={"intent": "exact_fact", "required_pages": [catalogue_url]},
        max_items=2,
        max_chars=1000,
    )
    assert parking_pack["items"][0]["id"] == "catalogue-parking"

    screening_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf"
    screening_pack = build_evidence_pack(
        query="When is IT support available for the online screening exam?",
        result={
            "evidence_span_documents": [
                {
                    "id": "screening-general",
                    "text": "The online screening exam assesses applicants' knowledge and skills.",
                    "source_url": screening_url,
                },
                {
                    "id": "screening-hours",
                    "text": "The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae for technical support. Working hours are at 8:00 AM - 5:00 PM on Mondays to Thursdays and at 8:00 AM - 12:30 PM on Fridays.",
                    "source_url": screening_url,
                },
            ]
        },
        coverage_plan={"intent": "exact_fact", "required_pages": [screening_url]},
        max_items=2,
        max_chars=1000,
    )
    assert screening_pack["items"][0]["id"] == "screening-hours"


def test_evidence_pack_demotes_truncated_screening_hours_contact_record():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    screening_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf"
    pack = build_evidence_pack(
        query=(
            "Compare the right contact paths for general admissions, undergraduate admissions, "
            "and IT support for the online screening exam, including when IT support is available."
        ),
        result={
            "answer_documents": [
                {
                    "id": "truncated-it-email",
                    "text": (
                        "The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae for technical support. "
                        "Working hours are at 8:00 AM - 5:00 PM ("
                    ),
                    "source_url": screening_url,
                },
                {
                    "id": "complete-support-hours",
                    "text": (
                        "Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays to Thursdays "
                        "and at 8:00 AM - 12:30 PM (UAE time) on Fridays."
                    ),
                    "source_url": screening_url,
                },
            ],
            "evidence_span_documents": [
                {
                    "id": "complete-support-hours-span",
                    "text": (
                        "Additional reminders - The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae "
                        "for technical support. Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays "
                        "to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays."
                    ),
                    "source_url": screening_url,
                }
            ],
        },
        coverage_plan={"intent": "multi_page_aggregation", "required_pages": [screening_url]},
        max_items=3,
        max_chars=2000,
    )

    packed_ids = [item["id"] for item in pack["items"]]
    assert packed_ids[0] in {"complete-support-hours", "complete-support-hours-span"}
    assert packed_ids.index("truncated-it-email") > 0


def test_evidence_pack_guarantees_transport_facets_for_practical_arrival_query():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    contact_url = "https://mbzuai.ac.ae/about/contact"
    undergrad_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    facilities_url = "https://mbzuai.ac.ae/student-resources/campus-facilities"
    pack = build_evidence_pack(
        query=(
            "Give a detailed practical guide for a new graduate student arriving at MBZUAI, "
            "covering campus location, working hours, accommodation, parking, shuttle transport, and important facilities."
        ),
        result={
            "fact_documents": [
                {
                    "id": "prt-only",
                    "text": "Visitors can park in the North Car Park and take the PRT to Building 1A.",
                    "source_url": contact_url,
                },
                {
                    "id": "masdar-location",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "source_url": undergrad_url,
                },
                {
                    "id": "facility-247",
                    "text": "The Library provides quiet study spaces and 24/7 electronic resources.",
                    "source_url": facilities_url,
                },
            ],
            "evidence_span_documents": [
                {
                    "id": "navya-transport",
                    "text": (
                        "Visitors are requested to park in the North Car Park and take the golf cart "
                        "or electric autonomous NAVYA bus to the university's building, if available."
                    ),
                    "source_url": contact_url,
                },
                {
                    "id": "library-facilities",
                    "text": "Campus facilities include laboratories, library, a knowledge center, canteen, gyms, and a swimming pool.",
                    "source_url": facilities_url,
                },
            ],
        },
        coverage_plan={
            "intent": "broad_synthesis",
            "required_pages": [contact_url, undergrad_url, facilities_url],
            "required_entities": ["Masdar", "NAVYA bus", "library"],
        },
        max_items=5,
        max_chars=3000,
    )

    packed_text = "\n".join(item["text"] for item in pack["items"])
    packed_ids = [item["id"] for item in pack["items"]]
    assert "NAVYA bus" in packed_text
    assert "Masdar City" in packed_text
    assert "library" in packed_text.casefold()
    assert packed_ids.index("navya-transport") < packed_ids.index("prt-only")
    assert "facility-247" not in packed_ids or packed_ids.index("facility-247") > packed_ids.index("library-facilities")


def test_evidence_pack_normalizes_official_workings_hours_for_coverage():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    faq_url = "https://mbzuai.ac.ae/about/faq"
    pack = build_evidence_pack(
        query="When do MBZUAI offices operate?",
        result={
            "evidence_span_documents": [
                {
                    "id": "faq-hours",
                    "text": "Our official workings hours are from 8:00 a.m. - 6:00 p.m., Monday to Thursday, and 7.30am to 12pm on Friday.",
                    "source_url": faq_url,
                }
            ]
        },
        coverage_plan={"intent": "exact_fact", "required_pages": [faq_url], "required_entities": ["working hours"]},
        max_items=1,
        max_chars=1000,
    )

    assert pack["coverage_status"] == "complete"
    assert pack["missing_required_entities"] == []


def test_answer_record_extractor_expands_support_email_hours_context():
    from pipeline.core.answer_records import _extract_contact_records

    source = {
        "id": "screening-chunk",
        "document_title": "MBZUAI Online Screening Exam Instructions",
        "source_url": "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf",
    }
    text = (
        "Additional reminders - The MBZUAI IT team may be emailed at IT_external@mbzuai.ac.ae "
        "for technical support. Working hours are at 8:00 AM - 5:00 PM (UAE time) on Mondays "
        "to Thursdays and at 8:00 AM - 12:30 PM (UAE time) on Fridays."
    )

    records = _extract_contact_records(source, source_kind="chunk", text=text)
    support_record = next(record for record in records if record["value"].lower() == "it_external@mbzuai.ac.ae")

    assert "12:30 PM" in support_record["text"]
    assert not support_record["text"].rstrip().endswith("(")


def test_evidence_pack_fills_specific_required_page_quota_before_related_pages():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    target_url = "https://mbzuai.ac.ae/study/msc-programs/master-of-science-in-machine-learning"
    result = {
        "evidence_span_documents": [
            {
                "id": "target-referees",
                "text": "M.Sc. applicants should nominate two referees for the application.",
                "source_url": target_url,
                "span_type": "requirement",
            },
            {
                "id": "related-program",
                "text": "Machine learning appears in many MBZUAI research and program pages.",
                "source_url": "https://mbzuai.ac.ae/news/machine-learning-101",
                "span_type": "general",
            },
            {
                "id": "target-screening",
                "text": "Applicants complete an online screening exam for the M.Sc. in Machine Learning.",
                "source_url": target_url,
                "span_type": "requirement",
            },
        ],
    }

    pack = build_evidence_pack(
        query="What are referee and screening requirements for the MSc in Machine Learning?",
        result=result,
        coverage_plan={"intent": "faculty_program_detail", "required_pages": [target_url]},
        max_items=3,
        max_chars=1000,
        max_per_source=2,
    )

    assert [item["source_url"] for item in pack["items"][:2]] == [target_url, target_url]
    assert {item["source_url"] for item in pack["items"]} == {target_url}


def test_evidence_pack_scopes_single_required_page_even_for_aggregation_intent():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    target_url = "https://mbzuai.ac.ae/ai-programs"
    result = {
        "evidence_span_documents": [
            {"id": "programs", "text": "MBZUAI offers AI programs and specializations.", "source_url": target_url},
            {"id": "news", "text": "A news article mentions AI specializations.", "source_url": "https://mbzuai.ac.ae/news/related"},
        ]
    }

    pack = build_evidence_pack(
        query="What five core AI specializations does MBZUAI offer in its M.Sc. and Ph.D. programs?",
        result=result,
        coverage_plan={"intent": "multi_page_aggregation", "required_pages": [target_url]},
        max_items=4,
        max_chars=1000,
        max_per_source=2,
    )

    assert [item["source_url"] for item in pack["items"]] == [target_url]


def test_confidence_counts_official_authority_class_for_backfilled_spans():
    from pipeline.retrieval.evidence_packer import score_retrieval_confidence

    confidence, factors = score_retrieval_confidence(
        {
            "selected_evidence_span_ids": ["s1", "s2"],
            "evidence_span_documents": [
                {"id": "s1", "text": "Official evidence.", "authority_class": "official"},
                {"id": "s2", "text": "More official evidence.", "authority_class": "official"},
            ],
            "retrieval_documents": [{"id": "s1", "text": "Official evidence.", "authority_class": "official"}],
        }
    )

    assert confidence > 0.25
    assert factors["authority_signal"] == 0.95


def test_retrieval_metric_gate_fails_when_v5_metric_has_no_eligible_queries():
    from pipeline.evaluation.retrieval_eval import check_metric_gates

    report = {
        "overall": {
            "eligible_span_query_count": 0.0,
            "span_hit_at_10": 0.0,
        }
    }
    gates = {"overall": {"span_hit_at_10": {"min": 0.85}}}

    failures = check_metric_gates(report, gates)

    assert failures
    assert failures[0]["reason"] == "no_eligible_queries"
    assert failures[0]["eligible_count_metric"] == "eligible_span_query_count"


def test_retrieval_entity_coverage_uses_declared_aliases_only():
    from pipeline.evaluation.retrieval_eval import _coverage_fraction, _metadata_aliases

    metadata = {
        "required_entity_aliases": {
            "working hours": ["official workings hours", "8:00 a.m."],
            "transport": ["NAVYA bus", "PRT"],
        }
    }
    aliases = _metadata_aliases(metadata, "required_entity_aliases")
    observed = "The FAQ states official workings hours. The contact page mentions an electric autonomous NAVYA bus."

    assert _coverage_fraction(["working hours", "transport"], observed.casefold(), aliases=aliases) == 1.0
    assert _coverage_fraction(["working hours", "canteen"], observed.casefold(), aliases=aliases) == 0.5


def test_routed_coverage_plan_infers_specific_program_page_and_marks_missing_source_partial():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    source_url = "https://mbzuai.ac.ae/study/msc-programs/master-of-science-in-machine-learning"
    search_text = "master of science in machine learning referee online screening exam math programming"
    retriever._coverage_page_records = [
        {
            "source_url": source_url,
            "normalized_url": retriever._normalize_source_url(source_url),
            "search_text": search_text,
            "tokens": set(_tokenize(search_text)),
        }
    ]

    plan = retriever._coverage_plan_for_result(
        query="For the Master of Science in Machine Learning program, what are referee and screening exam requirements?",
        payload={
            "selected_chunk_ids": ["wrong"],
            "retrieval_documents": [
                {
                    "id": "wrong",
                    "text": "PhD Machine Learning screening exam topics.",
                    "source_url": "https://mbzuai.ac.ae/study/phd-programs/doctor-of-philosophy-in-machine-learning",
                }
            ],
        },
        mode=QueryMode.SYNTHESIS,
    )

    assert source_url in plan["required_pages"]
    assert plan["coverage_status"] == "partial"


def test_routed_coverage_plan_requires_campus_facilities_and_campus_map_when_both_are_named():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    facilities_url = "https://mbzuai.ac.ae/student-resources/campus-facilities"
    old_map_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2022/07/MBZUAI_Campus_Map_.pdf"
    map_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/11/MBZUAI_Campus_Map_V1044331768.pdf"
    retriever._coverage_page_records = [
        {
            "source_url": facilities_url,
            "normalized_url": retriever._normalize_source_url(facilities_url),
            "search_text": "campus facilities knowledge center medical center dining sports rooms",
            "tokens": set(_tokenize("campus facilities knowledge center medical center dining sports rooms")),
        },
        {
            "source_url": old_map_url,
            "normalized_url": retriever._normalize_source_url(old_map_url),
            "search_text": "old mbzuai campus map layout clinic library dining student services",
            "tokens": set(_tokenize("old mbzuai campus map layout clinic library dining student services")),
        },
        {
            "source_url": map_url,
            "normalized_url": retriever._normalize_source_url(map_url),
            "search_text": "mbzuai campus map layout clinic library dining student services",
            "tokens": set(_tokenize("mbzuai campus map layout clinic library dining student services")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query=(
            "Give a detailed answer about MBZUAI campus facilities using both the "
            "campus facilities page and the campus map."
        ),
        payload={
            "selected_chunk_ids": ["facilities-only"],
            "evidence_span_documents": [
                {
                    "id": "facilities-only",
                    "text": "Campus facilities include study and recreation spaces.",
                    "source_url": facilities_url,
                }
            ],
        },
        mode=QueryMode.SCOPED,
    )

    assert facilities_url in plan["required_pages"]
    assert map_url in plan["required_pages"]
    assert old_map_url not in plan["required_pages"]
    assert plan["coverage_status"] == "partial"


def test_source_markers_include_campus_map_for_student_facing_service_queries():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    markers = retriever._explicit_required_page_markers(
        "What student-facing campus services and support facilities are available at MBZUAI?"
    )

    assert "/student-resources/campus-facilities" in markers
    assert "/study/undergraduate-application-submission" in markers
    assert "campus_map" in markers
    assert "campus-map" in markers


def test_routed_coverage_plan_requires_screening_exam_pdf_for_it_support_hours():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    pdf_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf"
    faq_url = "https://mbzuai.ac.ae/about/faq"
    retriever._coverage_page_records = [
        {
            "source_url": faq_url,
            "normalized_url": retriever._normalize_source_url(faq_url),
            "search_text": "faq working hours official weekday",
            "tokens": set(_tokenize("faq working hours official weekday")),
        },
        {
            "source_url": pdf_url,
            "normalized_url": retriever._normalize_source_url(pdf_url),
            "search_text": "online screening exam instructions it support technical support working hours",
            "tokens": set(_tokenize("online screening exam instructions it support technical support working hours")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query="What are the IT support working hours for the MBZUAI online screening exam?",
        payload={"selected_chunk_ids": ["faq"], "retrieval_documents": [{"id": "faq", "source_url": faq_url, "text": "FAQ hours"}]},
        mode=QueryMode.FACT,
    )

    assert plan["required_pages"] == [pdf_url]
    assert plan["coverage_status"] == "partial"


def test_routed_coverage_plan_requires_practical_campus_pages_for_synthesis():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    faq_url = "https://mbzuai.ac.ae/about/faq"
    contact_url = "https://mbzuai.ac.ae/about/contact"
    accommodation_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    facilities_url = "https://mbzuai.ac.ae/student-resources/campus-facilities"
    retriever._coverage_page_records = [
        {
            "source_url": faq_url,
            "normalized_url": retriever._normalize_source_url(faq_url),
            "search_text": "location working hours parking masdar city faq",
            "tokens": set(_tokenize("location working hours parking masdar city faq")),
        },
        {
            "source_url": contact_url,
            "normalized_url": retriever._normalize_source_url(contact_url),
            "search_text": "contact north car park navya bus prt visitor parking",
            "tokens": set(_tokenize("contact north car park navya bus prt visitor parking")),
        },
        {
            "source_url": accommodation_url,
            "normalized_url": retriever._normalize_source_url(accommodation_url),
            "search_text": "student accommodation shuttle transport housing",
            "tokens": set(_tokenize("student accommodation shuttle transport housing")),
        },
        {
            "source_url": facilities_url,
            "normalized_url": retriever._normalize_source_url(facilities_url),
            "search_text": "campus facilities library knowledge center gym medical center",
            "tokens": set(_tokenize("campus facilities library knowledge center gym medical center")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query=(
            "Explain the most important practical campus information a new MBZUAI student "
            "should know, including location, working hours, accommodation, parking, and facilities."
        ),
        payload={"selected_chunk_ids": ["faq"], "retrieval_documents": [{"id": "faq", "source_url": faq_url, "text": "FAQ"}]},
        mode=QueryMode.SYNTHESIS,
    )

    assert faq_url in plan["required_pages"]
    assert accommodation_url in plan["required_pages"]
    assert facilities_url in plan["required_pages"]
    assert "Masdar" in plan["required_entities"]
    assert "working hours" in plan["required_entities"]
    assert "accommodation" in plan["required_entities"]
    assert "library" in plan["required_entities"]
    assert "NAVYA bus" not in plan["required_entities"]
    assert plan["coverage_status"] == "partial"

    briefing_plan = retriever._coverage_plan_for_result(
        query=(
            "Prepare a newcomer briefing covering where MBZUAI is, when offices operate, "
            "how transport and parking work, and what support facilities exist on campus."
        ),
        payload={"selected_chunk_ids": ["faq"], "retrieval_documents": [{"id": "faq", "source_url": faq_url, "text": "FAQ"}]},
        mode=QueryMode.SYNTHESIS,
    )

    assert faq_url in briefing_plan["required_pages"]
    assert accommodation_url in briefing_plan["required_pages"]
    assert facilities_url in briefing_plan["required_pages"]
    assert "Masdar" in briefing_plan["required_entities"]
    assert "NAVYA bus" in briefing_plan["required_entities"]
    assert "library" in briefing_plan["required_entities"]
    assert not any("University-Catalogue-2024-2025" in page for page in briefing_plan["required_pages"])

    visitor_plan = retriever._coverage_plan_for_result(
        query=(
            "Summarize the practical campus information a visitor should know before arriving at MBZUAI, "
            "including location, parking, transport, and facilities."
        ),
        payload={"selected_chunk_ids": ["contact"], "retrieval_documents": [{"id": "contact", "source_url": contact_url, "text": "Contact"}]},
        mode=QueryMode.SYNTHESIS,
    )

    assert faq_url in visitor_plan["required_pages"]
    assert accommodation_url in visitor_plan["required_pages"]
    assert facilities_url in visitor_plan["required_pages"]
    assert contact_url in visitor_plan["required_pages"]
    assert "Masdar" in visitor_plan["required_entities"]
    assert "NAVYA bus" in visitor_plan["required_entities"]


def test_routed_coverage_plan_requires_all_contact_sources_for_admissions_comparison():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    pdf_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf"
    undergraduate_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    admissions_url = "https://mbzuai.ac.ae/about/faq"
    retriever._coverage_page_records = [
        {
            "source_url": pdf_url,
            "normalized_url": retriever._normalize_source_url(pdf_url),
            "search_text": "online screening exam instructions it support working hours",
            "tokens": set(_tokenize("online screening exam instructions it support working hours")),
        },
        {
            "source_url": undergraduate_url,
            "normalized_url": retriever._normalize_source_url(undergraduate_url),
            "search_text": "undergraduate admissions ug admission email",
            "tokens": set(_tokenize("undergraduate admissions ug admission email")),
        },
        {
            "source_url": admissions_url,
            "normalized_url": retriever._normalize_source_url(admissions_url),
            "search_text": "faq general admissions admission email",
            "tokens": set(_tokenize("general admissions admission email")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query=(
            "Compare the right contact paths for general admissions, undergraduate admissions, "
            "and IT support for the online screening exam, including when IT support is available."
        ),
        payload={"selected_chunk_ids": ["pdf"], "retrieval_documents": [{"id": "pdf", "source_url": pdf_url, "text": "PDF"}]},
        mode=QueryMode.SYNTHESIS,
    )

    assert pdf_url in plan["required_pages"]
    assert undergraduate_url in plan["required_pages"]
    assert admissions_url in plan["required_pages"]
    assert plan["coverage_status"] == "partial"


def test_routed_coverage_plan_uses_canonical_sources_for_parking_and_admissions_email():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    catalogue_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/05/University-Catalogue-2024-2025.pdf"
    contact_url = "https://mbzuai.ac.ae/about/contact"
    application_pdf_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/PDF/MBZUAI_Application_Instructions_New_MSc-PhD.pdf"
    screening_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf"
    faq_url = "https://mbzuai.ac.ae/about/faq"
    retriever._coverage_page_records = [
        {
            "source_url": catalogue_url,
            "normalized_url": retriever._normalize_source_url(catalogue_url),
            "search_text": "university catalogue parking north car park masdar city",
            "tokens": set(_tokenize("university catalogue parking north car park masdar city")),
        },
        {
            "source_url": contact_url,
            "normalized_url": retriever._normalize_source_url(contact_url),
            "search_text": "contact visitor parking north car park navya bus",
            "tokens": set(_tokenize("contact visitor parking north car park navya bus")),
        },
        {
            "source_url": application_pdf_url,
            "normalized_url": retriever._normalize_source_url(application_pdf_url),
            "search_text": "application instructions admission email admission@mbzuai.ac.ae",
            "tokens": set(_tokenize("application instructions admission email admission@mbzuai.ac.ae")),
        },
        {
            "source_url": screening_url,
            "normalized_url": retriever._normalize_source_url(screening_url),
            "search_text": "online screening exam instructions admission related questions admission@mbzuai.ac.ae",
            "tokens": set(_tokenize("online screening exam instructions admission related questions admission@mbzuai.ac.ae")),
        },
        {
            "source_url": faq_url,
            "normalized_url": retriever._normalize_source_url(faq_url),
            "search_text": "faq admissions email admission@mbzuai.ac.ae",
            "tokens": set(_tokenize("faq admissions email admission@mbzuai.ac.ae")),
        },
    ]

    parking_plan = retriever._coverage_plan_for_result(
        query="Where can vehicles be parked on the Masdar City campus?",
        payload={"selected_chunk_ids": ["contact"], "retrieval_documents": [{"id": "contact", "source_url": contact_url, "text": "North Car Park"}]},
        mode=QueryMode.FACT,
    )
    admissions_plan = retriever._coverage_plan_for_result(
        query="What is the admissions email address?",
        payload={"selected_chunk_ids": ["faq"], "retrieval_documents": [{"id": "faq", "source_url": faq_url, "text": "admission@mbzuai.ac.ae"}]},
        mode=QueryMode.FACT,
    )
    committee_plan = retriever._coverage_plan_for_result(
        query="What is the admissions committee email address?",
        payload={"selected_chunk_ids": ["faq"], "retrieval_documents": [{"id": "faq", "source_url": faq_url, "text": "admission@mbzuai.ac.ae"}]},
        mode=QueryMode.FACT,
    )

    assert catalogue_url in parking_plan["required_pages"]
    assert contact_url in parking_plan["required_pages"]
    assert application_pdf_url in admissions_plan["required_pages"]
    assert screening_url in admissions_plan["required_pages"]
    assert screening_url in committee_plan["required_pages"]


def test_required_page_span_backfill_clears_stale_abstention():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    target_url = "https://mbzuai.ac.ae/study/ai-reach"
    retriever.vector = SimpleNamespace(
        evidence_span_map={
            "ai-reach-purpose": {
                "id": "ai-reach-purpose",
                "text": "AI Reach is fully funded and designed for participants learning AI.",
                "source_url": target_url,
                "linked_chunk_ids": ["chunk-ai-reach"],
                "span_type": "program",
            }
        },
        _score_text_match=lambda query, text: 1.0,
    )
    payload = {
        "abstained": True,
        "verification_status": "abstained",
        "selected_evidence_span_ids": [],
        "selected_chunk_ids": [],
        "evidence_span_documents": [],
        "retrieval_documents": [],
    }

    changed = retriever._augment_payload_for_required_coverage(
        query="What is AI Reach?",
        payload=payload,
        coverage_plan={"required_pages": [target_url]},
    )

    assert changed is True
    assert payload["abstained"] is False
    assert payload["verification_status"] == "backfilled_required_page_evidence"
    assert payload["selected_evidence_span_ids"] == ["ai-reach-purpose"]


def test_required_page_backfill_keeps_unsupported_premise_abstained():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    target_url = "https://preprod.mbzuai.ac.ae/campus-community/campus-facilities"
    retriever._intent_summary = lambda _query: {}
    retriever.vector = SimpleNamespace(
        evidence_span_map={
            "masdar-shuttle": {
                "id": "masdar-shuttle",
                "text": "A shuttle serves the Masdar City campus.",
                "source_url": target_url,
                "linked_chunk_ids": ["chunk-masdar"],
                "span_type": "transport",
            }
        },
        _score_text_match=lambda query, text: 1.0,
    )
    payload = {
        "abstained": True,
        "premise_grounding_required": True,
        "verification_status": "abstained",
        "adjudication_reason": "presupposed_entity_or_scope_not_supported",
        "selected_evidence_span_ids": [],
        "selected_chunk_ids": [],
        "evidence_span_documents": [],
        "retrieval_documents": [],
    }

    changed = retriever._augment_payload_for_required_coverage(
        query="What is the shuttle timetable for MBZUAI's Mars research campus?",
        payload=payload,
        coverage_plan={"required_pages": [target_url]},
    )

    assert changed is True
    assert payload["abstained"] is True
    assert payload["verification_status"] == "abstained"
    assert payload["adjudication_reason"] == "presupposed_entity_or_scope_not_supported"


def test_current_program_markers_cover_arabic_queries_and_public_routes():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert (
        "/academics/phd-programs/doctoral-robotics"
        in retriever._explicit_required_page_markers(
            "كم عدد الساعات المطلوبة لدكتوراه الفلسفة في الروبوتات؟"
        )
    )
    assert (
        "/academics/msc-programs/masters-in-computer-science"
        in retriever._explicit_required_page_markers(
            "What does the Master of Science in Computer Science require?"
        )
    )


def test_current_multisource_markers_cover_research_admissions_and_student_support():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert set(
        retriever._explicit_required_page_markers(
            "كيف تصف الجامعة أقسامها ومعاهدها ومراكزها البحثية؟"
        )
    ) >= {
        "/research/our-divisions",
        "/research/institutes-centers",
    }
    assert set(
        retriever._explicit_required_page_markers(
            "Combine the undergraduate admissions criteria and program description."
        )
    ) >= {
        "/admissions-aid/undergraduate-admissions",
        "/academics/undergraduate-program",
    }
    assert set(
        retriever._explicit_required_page_markers(
            "What campus amenities and student support services are available?"
        )
    ) >= {
        "/campus-community/campus-facilities",
        "/about-us/office-of-student-postdoctoral-affairs",
    }
    assert "governance_structure.pdf" in retriever._explicit_required_page_markers(
        "How does the chair's biography relate to university governance?"
    )
    assert (
        "mbzuai_research_showcase_20250417-final.pdf"
        in retriever._explicit_required_page_markers(
            "How do the divisions and research showcase describe innovation?"
        )
    )


def test_required_page_fact_backfill_injects_exact_faq_admissions_contact():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    faq_url = "https://mbzuai.ac.ae/about/faq"
    retriever.vector = SimpleNamespace(
        fact_map={
            "faq-admissions": {
                "id": "faq-admissions",
                "text": "Afterwards, all change and document upload requests have to be sent by email to admission@mbzuai.ac.ae.",
                "source_url": faq_url,
                "document_title": "FAQ",
                "linked_chunk_ids": ["faq-chunk"],
            }
        },
        evidence_span_map={},
        _score_text_match=lambda query, text: 0.2 if "admission@mbzuai.ac.ae" in text else 0.0,
    )
    payload = {
        "selected_fact_ids": [],
        "fact_documents": [],
        "selected_chunk_ids": [],
        "retrieval_documents": [],
        "abstained": False,
    }

    changed = retriever._augment_payload_for_required_coverage(
        query=(
            "Compare the right contact paths for general admissions, undergraduate admissions, "
            "and IT support for the online screening exam."
        ),
        payload=payload,
        coverage_plan={"required_pages": [faq_url]},
    )

    assert changed is True
    assert payload["selected_fact_ids"] == ["faq-admissions"]
    assert payload["fact_documents"][0]["source_url"] == faq_url
    assert "admission@mbzuai.ac.ae" in payload["fact_documents"][0]["text"]
    assert payload["selected_chunk_ids"][0] == "faq-chunk"


def test_explicit_required_pages_prefer_english_sources_for_english_queries():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    english_url = "https://mbzuai.ac.ae/the-sheikh-tahnoon-bin-zayed-scholarship-in-ai-excellence-at-mbzuai"
    arabic_url = "https://mbzuai.ac.ae/ar/news/in-line-with-the-vision-of-his-highness-sheikh-mohamed-bin-zayed-al-nahyan-the-newly-established-tahnoon-bin-zayed-scholarship-in-ai-excellence-program-supports-outstanding-undergraduate-talent-at-mb"
    retriever._coverage_page_records = [
        {
            "source_url": arabic_url,
            "normalized_url": retriever._normalize_source_url(arabic_url),
            "search_text": "tahnoon bin zayed scholarship undergraduate",
            "tokens": set(_tokenize("tahnoon bin zayed scholarship undergraduate")),
        },
        {
            "source_url": english_url,
            "normalized_url": retriever._normalize_source_url(english_url),
            "search_text": "tahnoon bin zayed scholarship undergraduate",
            "tokens": set(_tokenize("tahnoon bin zayed scholarship undergraduate")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query="What scholarship information is available for MBZUAI undergraduate applicants?",
        payload={"selected_chunk_ids": ["s1"]},
        mode=QueryMode.SCOPED,
    )

    assert english_url in plan["required_pages"]
    assert arabic_url not in plan["required_pages"]


def test_query_has_specific_target_for_production_operational_intents():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)

    assert retriever._query_has_specific_target("What is the admissions email address?")
    assert retriever._query_has_specific_target(
        "What five core AI specializations does MBZUAI offer in its M.Sc. and Ph.D. programs?"
    )
    assert retriever._query_has_specific_target("Where can vehicles be parked on the Masdar City campus?")


def test_required_page_span_selection_prefers_purpose_over_deadline_for_what_is_query():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    target_url = "https://mbzuai.ac.ae/study/ai-reach"
    retriever.vector = SimpleNamespace(
        evidence_span_map={
            "deadline": {
                "id": "deadline",
                "text": "Applications close soon. AI Reach program dates will be announced.",
                "source_url": target_url,
            },
            "purpose": {
                "id": "purpose",
                "text": "AI Reach is designed for participants who want a structured introduction to AI.",
                "source_url": target_url,
            },
        },
        _score_text_match=lambda query, text: 0.5,
    )

    spans = retriever._best_required_page_spans(
        "What is AI Reach and who is it for?",
        target_url,
        limit=2,
    )

    assert [span["id"] for span in spans] == ["purpose", "deadline"]


def test_unsupported_intent_guard_flags_private_confidential_live_and_future_queries():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = True
    retriever.future_year_guard_horizon = 1
    future_year = datetime.now().year + 3

    assert retriever._unsupported_intent_reason("What is my exact MBZUAI PhD interview schedule for next week?")
    assert retriever._unsupported_intent_reason("Which dorm room numbers are assigned to students?")
    assert retriever._unsupported_intent_reason("What are the exact questions on the current screening exam?")
    assert retriever._unsupported_intent_reason("What is the shuttle live location right now?")
    assert retriever._unsupported_intent_reason(f"Who won MBZUAI's {future_year} robotics hackathon and what was the prize amount?")
    assert retriever._unsupported_intent_reason(f"What will MBZUAI's exact tuition fee be in {future_year}?")
    assert retriever._unsupported_intent_reason(f"Who will deliver MBZUAI's {future_year} commencement address?")
    assert retriever._unsupported_intent_reason(f"من سيلقي كلمة حفل تخرج الجامعة لعام {future_year}؟")
    assert not retriever._unsupported_intent_reason("What PhD programs does MBZUAI offer?")

    retriever.unsupported_intent_guard_enabled = False
    assert not retriever._unsupported_intent_reason("What is my exact MBZUAI PhD interview schedule for next week?")


def test_answer_prompt_prefers_evidence_pack_context():
    from pipeline.evaluation.answer_generation import _build_answer_prompt, _retrieved_contexts_from_result

    retrieval_result = {
        "evidence_pack": {
            "items": [
                {"rank": 1, "kind": "answer", "text": "Packed answer.", "source_url": "https://example.com/answer"},
                {"rank": 2, "kind": "chunk", "text": "Packed chunk.", "source_url": "https://example.com/chunk"},
            ]
        },
        "retrieval_documents": [{"id": "raw", "text": "Raw context should not be used."}],
    }

    prompt = _build_answer_prompt(
        query="question",
        retrieval_documents=retrieval_result["retrieval_documents"],
        evidence_pack=retrieval_result["evidence_pack"],
    )

    assert "Packed answer." in prompt
    assert "Packed chunk." in prompt
    assert "Raw context should not be used." not in prompt
    assert _retrieved_contexts_from_result(retrieval_result) == ["Packed answer.", "Packed chunk."]


def test_answer_readiness_scores_generated_answers(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "answerable",
                        "query": "Where is MBZUAI located?",
                        "query_type": "fact",
                        "source_type": "webpage",
                        "reference_answer": "MBZUAI is in Masdar City, Abu Dhabi.",
                        "metadata": {
                            "benchmark_tags": ["release_readiness"],
                            "answer_must_include": ["Masdar City", "Abu Dhabi"],
                        },
                    }
                ),
                json.dumps(
                    {
                        "id": "noanswer",
                        "query": "What is MBZUAI's Toronto admissions extension?",
                        "query_type": "fact",
                        "source_type": "none",
                        "no_answer": True,
                        "reference_answer": "Insufficient evidence.",
                        "metadata": {
                            "benchmark_tags": ["release_readiness", "contact_lookup"],
                            "answer_must_not_include": ["Toronto admissions extension"],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "answer_gates.json"
    gates.write_text(
        json.dumps(
            {
                "overall": {
                    "pass_rate": {"min": 1.0},
                    "support_present_rate": {"min": 1.0},
                    "no_answer_pass_rate": {"min": 1.0},
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_generate_answer_predictions(**kwargs):
        Path(kwargs["output_path"]).write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": "answerable",
                            "response": "MBZUAI is located in Masdar City, Abu Dhabi.",
                            "retrieved_contexts": ["MBZUAI is located in Masdar City, Abu Dhabi."],
                        }
                    ),
                    json.dumps(
                        {
                            "id": "noanswer",
                            "response": "Insufficient evidence.",
                            "retrieved_contexts": [],
                            "response_kind": "grounded_no_answer",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return {"row_count": 2}

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        mode="local",
        judge_enabled=False,
    )

    assert report["gates"]["passed"] is True
    assert report["overall"]["pass_rate"] == 1.0
    assert report["by_benchmark_tag"]["contact_lookup"]["no_answer_pass_rate"] == 1.0


def test_answer_readiness_can_combine_governed_splits_without_materializing_subset(
    tmp_path,
    monkeypatch,
):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "governed.jsonl"
    rows = [
        {
            "id": "selection-row",
            "query": "Selection question",
            "query_type": "fact",
            "source_type": "webpage",
            "reference_answer": "Selection answer",
            "gold_chunk_ids": ["selection-chunk"],
            "metadata": {"split": "selection"},
        },
        {
            "id": "regression-row",
            "query": "Regression question",
            "query_type": "fact",
            "source_type": "webpage",
            "reference_answer": "Regression answer",
            "gold_chunk_ids": ["regression-chunk"],
            "metadata": {"split": "regression"},
        },
        {
            "id": "sealed-holdout-row",
            "query": "Sealed question",
            "query_type": "fact",
            "source_type": "webpage",
            "reference_answer": "Sealed answer",
            "gold_chunk_ids": ["holdout-chunk"],
            "metadata": {"split": "holdout"},
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    def fake_generate_answer_predictions(**kwargs):
        selected_examples = kwargs["examples"]
        assert [example.id for example in selected_examples] == [
            "selection-row",
            "regression-row",
        ]
        Path(kwargs["output_path"]).write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": example.id,
                        "response": example.reference_answer,
                        "retrieved_contexts": [example.reference_answer],
                    }
                )
                for example in selected_examples
            )
            + "\n",
            encoding="utf-8",
        )
        return {"row_count": len(selected_examples)}

    monkeypatch.setattr(
        answer_readiness,
        "generate_answer_predictions",
        fake_generate_answer_predictions,
    )

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        mode="local",
        judge_enabled=False,
        splits=["selection", "regression"],
    )

    assert report["requested_splits"] == ["selection", "regression"]
    assert report["query_count"] == 2
    assert {row["id"] for row in report["queries"]} == {
        "selection-row",
        "regression-row",
    }


def test_answer_readiness_query_id_filter_is_governed_by_selected_splits(
    tmp_path,
    monkeypatch,
):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "governed.jsonl"
    rows = [
        {
            "id": "selection-row",
            "query": "Selection question",
            "query_type": "fact",
            "source_type": "webpage",
            "reference_answer": "Selection answer",
            "gold_chunk_ids": ["selection-chunk"],
            "metadata": {"split": "selection"},
        },
        {
            "id": "regression-row",
            "query": "Regression question",
            "query_type": "fact",
            "source_type": "webpage",
            "reference_answer": "Regression answer",
            "gold_chunk_ids": ["regression-chunk"],
            "metadata": {"split": "regression"},
        },
        {
            "id": "sealed-holdout-row",
            "query": "Sealed question",
            "query_type": "fact",
            "source_type": "webpage",
            "reference_answer": "Sealed answer",
            "gold_chunk_ids": ["holdout-chunk"],
            "metadata": {"split": "holdout"},
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    generator_calls = []

    def fake_generate_answer_predictions(**kwargs):
        selected_examples = kwargs["examples"]
        generator_calls.append([example.id for example in selected_examples])
        Path(kwargs["output_path"]).write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": example.id,
                        "response": example.reference_answer,
                        "retrieved_contexts": [example.reference_answer],
                    }
                )
                for example in selected_examples
            )
            + "\n",
            encoding="utf-8",
        )
        return {"row_count": len(selected_examples)}

    monkeypatch.setattr(
        answer_readiness,
        "generate_answer_predictions",
        fake_generate_answer_predictions,
    )

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        mode="local",
        judge_enabled=False,
        splits=["selection", "regression"],
        example_ids=["regression-row", "selection-row", "regression-row"],
    )

    assert report["requested_example_ids"] == [
        "regression-row",
        "selection-row",
    ]
    assert report["query_count"] == 2
    assert generator_calls == [["selection-row", "regression-row"]]

    with pytest.raises(
        ValueError,
        match="not present in selected governed split.*sealed-holdout-row",
    ):
        answer_readiness.evaluate_answer_readiness(
            config_name="default",
            work_dir=tmp_path,
            dataset_path=dataset,
            mode="local",
            judge_enabled=False,
            splits=["selection", "regression"],
            example_ids=["sealed-holdout-row"],
        )

    assert generator_calls == [["selection-row", "regression-row"]]


def test_local_answer_predictions_continue_after_row_error(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_generation

    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1", "query": "First query", "query_type": "synthesis", "source_type": "webpage", "reference_answer": "A", "gold_chunk_ids": ["c1"]}),
                json.dumps({"id": "q2", "query": "Second query", "query_type": "synthesis", "source_type": "webpage", "reference_answer": "B", "gold_chunk_ids": ["c2"]}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    bundle_dir = tmp_path / "stage_outputs" / "format_retrieval"
    bundle_dir.mkdir(parents=True)
    (bundle_dir / "retrieval_bundle.json").write_text(
        json.dumps(
            {
                "chunk_records": [{"id": "c1", "text": "A"}, {"id": "c2", "text": "B"}],
                "parent_records": [],
                "media_records": [],
            }
        ),
        encoding="utf-8",
    )

    class FakeRetriever:
        def retrieve(self, query):
            return {
                "mode": "synthesis",
                "selected_chunk_ids": ["c1" if "First" in query else "c2"],
                "retrieval_documents": [{"id": "c1", "text": "context", "source_url": "https://example.edu"}],
                "abstained": False,
            }

    monkeypatch.setattr(answer_generation.AdaptiveHybridRetriever, "from_config", lambda **kwargs: FakeRetriever())

    def fake_generate_content_with_retry(**kwargs):
        if "First query" in kwargs["prompt"]:
            raise RuntimeError("transient ssl eof")
        return "Grounded answer."

    monkeypatch.setattr(answer_generation, "_generate_content_with_retry", fake_generate_content_with_retry)
    output = tmp_path / "predictions.jsonl"

    result = answer_generation.generate_answer_predictions(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        output_path=output,
        timeout_seconds=1,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert result["row_count"] == 2
    assert result["error_count"] == 1
    assert rows[0]["id"] == "q1"
    assert "transient ssl eof" in rows[0]["error"]
    assert rows[1]["id"] == "q2"
    assert rows[1]["response"] == "Grounded answer."


def test_rerank_text_is_bounded_by_character_budget():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = object.__new__(AdaptiveHybridRetriever)
    retriever.chunk_map = {"c1": {"text": "x" * 5000}}
    retriever.answer_texts_by_chunk = {}
    retriever.fact_texts_by_chunk = {}
    retriever.media_texts_by_chunk = {}

    text = retriever._build_rerank_text("c1", max_tokens=96, max_chars=80)

    assert len(text) <= 80
    assert text


def test_faculty_source_bonus_prefers_named_faculty_profile_over_program_page():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = object.__new__(AdaptiveHybridRetriever)
    query = "What are Professor Kentaro Inui's primary research interests in Natural Language Processing?"

    faculty_score = retriever._source_query_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/faculty/kentaro-inui",
        document_title="Kentaro Inui",
        heading="Faculty profile",
        text="Kentaro Inui researches computational modeling of semantics and discourse.",
    )
    program_score = retriever._source_query_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/phd-programs/doctor-of-philosophy-in-natural-language-processing",
        document_title="Doctor of Philosophy in Natural Language Processing",
        heading="Program overview",
        text="The NLP program includes faculty and course requirements.",
    )

    assert faculty_score > program_score + 2.0


def test_support_parent_expansion_finds_application_detail_chunks_before_overview():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = object.__new__(AdaptiveHybridRetriever)
    retriever.parent_candidate_top_k = 3
    retriever.max_parent_chunks = 6
    retriever.local_parent_candidate_pool = 64
    retriever.local_index_max_postings_per_token = 64
    retriever.namespace_parents = "parents"
    retriever.parent_map = {
        "program": {
            "id": "program",
            "parent_type": "page",
            "source_url": "https://mbzuai.ac.ae/study/phd-programs/phd-in-statistics-and-data-science",
            "document_title": "PhD in Statistics and Data Science",
            "text": "Curriculum, study plan, mandatory courses, and application submission information.",
            "child_chunk_ids": ["overview", "curriculum", "screening"],
        }
    }
    retriever.chunk_map = {
        "overview": {
            "id": "overview",
            "source_url": "https://mbzuai.ac.ae/study/phd-programs/phd-in-statistics-and-data-science",
            "document_title": "PhD in Statistics and Data Science",
            "heading": "Overview",
            "text": "The program develops research skills in statistics and data science.",
        },
        "curriculum": {
            "id": "curriculum",
            "source_url": "https://mbzuai.ac.ae/study/phd-programs/phd-in-statistics-and-data-science",
            "document_title": "PhD in Statistics and Data Science",
            "heading": "Mandatory courses",
            "text": "Course description, mandatory courses, elective courses, and study plan.",
        },
        "screening": {
            "id": "screening",
            "source_url": "https://mbzuai.ac.ae/study/phd-programs/phd-in-statistics-and-data-science",
            "document_title": "PhD in Statistics and Data Science",
            "heading": "Application submission",
            "text": "The online screening exam covers math, programming, and machine learning. Recommended online courses include Programming for Everybody, Python Data Structures, Mathematics for Machine Learning: Linear Algebra, and An Intuitive Introduction to Probability.",
        },
    }
    retriever.lexical_map = {
        "program": {
            "id": "program",
            "text": retriever.parent_map["program"]["text"],
        }
    }
    retriever._namespace_token_index = {
        "parents": {
            "statistics": ["program"],
            "data": ["program"],
            "science": ["program"],
            "screening": ["program"],
            "exam": ["program"],
            "courses": ["program"],
        }
    }
    retriever._namespace_tokens_by_id = {
        "parents": {
            "program": {
                "statistics",
                "data",
                "science",
                "screening",
                "exam",
                "courses",
            }
        }
    }
    retriever._bm25_by_namespace = {}

    query = "What topics are covered in the online screening exam for the PhD in Statistics and Data Science program, and are there any recommended courses to prepare?"
    parent_ids = retriever._support_parent_ids_for_query(query, top_k=3)
    chunk_ids = retriever._support_chunk_ids_for_query(query, parent_ids, top_k=3)

    assert parent_ids == ["program"]
    assert chunk_ids[0] == "screening"


def test_support_parent_expansion_never_rescores_the_complete_parent_corpus():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = object.__new__(AdaptiveHybridRetriever)
    retriever.local_parent_candidate_pool = 8
    retriever.namespace_parents = "release-parents"
    retriever.local_index_max_postings_per_token = 64
    retriever.parent_map = {
        "candidate": {
            "id": "candidate",
            "parent_type": "page",
            "source_url": "https://mbzuai.ac.ae/study/scholarships",
            "document_title": "Scholarships",
            "text": "MBZUAI scholarship funding and stipend information.",
        },
        "must-not-score": {
            "id": "must-not-score",
            "parent_type": "page",
            "source_url": "https://mbzuai.ac.ae/unrelated",
            "document_title": "Unrelated",
            "text": "Unrelated page.",
        },
    }
    retriever.lexical_map = {
        parent_id: {"id": parent_id, "text": parent["text"]}
        for parent_id, parent in retriever.parent_map.items()
    }
    retriever._namespace_token_index = {
        "release-parents": {
            "scholarship": ["candidate"],
            "funding": ["candidate"],
        }
    }
    retriever._namespace_tokens_by_id = {
        "release-parents": {
            "candidate": {"mbzuai", "scholarship", "funding", "stipend", "information"},
            "must-not-score": {"unrelated", "page"},
        }
    }
    retriever._bm25_by_namespace = {}
    seen_titles = []
    original_source_bonus = retriever._source_query_bonus

    def tracked_source_bonus(query, **kwargs):
        seen_titles.append(kwargs.get("document_title"))
        return original_source_bonus(query, **kwargs)

    retriever._source_query_bonus = tracked_source_bonus

    assert retriever._support_parent_ids_for_query(
        "What scholarship funding does MBZUAI provide?",
        top_k=3,
    ) == ["candidate"]
    assert seen_titles == ["Scholarships"]


def test_temporal_guard_abstains_when_requested_admission_cycle_is_missing():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = object.__new__(AdaptiveHybridRetriever)
    retriever.temporal_exact_year_guard_enabled = True
    retriever.temporal_guard_top_k = 3
    retriever.abstain_min_token_overlap = 0.01
    retriever.abstain_min_support_score = 0.01
    retriever.fact_abstain_min_token_overlap = 0.20
    retriever.fact_require_fact_support_overlap = 0.35
    retriever.chunk_map = {
        "current": {
            "id": "current",
            "text": "Undergraduate admission requirements for the 2025-2026 academic year include transcripts and English proficiency.",
            "source_url": "https://mbzuai.ac.ae/study/undergraduate-application-submission",
            "document_title": "Undergraduate application submission",
        }
    }
    retriever.answer_texts_by_chunk = {}
    retriever.fact_texts_by_chunk = {}

    should_abstain = retriever._should_abstain(
        query="What are the undergraduate admission requirements for MBZUAI for the academic year 2027-2028?",
        mode=QueryMode.SCOPED,
        ranked_chunks=[("current", 1.0)],
        support={"current": {"score": 1.0, "sources": {"dense_chunks", "local_chunks"}}},
    )

    assert should_abstain is True


def test_answer_readiness_uses_llm_judge_for_full_response_contract(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "component_eval",
                "query": "How can I contact MBZUAI admissions?",
                "query_type": "fact",
                "source_type": "webpage",
                "reference_answer": "Use the admissions contact published by MBZUAI.",
                "metadata": {
                    "benchmark_tags": ["release_readiness", "contact_lookup"],
                    "answer_must_include": ["admissions@mbzuai.ac.ae"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "answer_gates.json"
    gates.write_text(
        json.dumps(
            {
                "overall": {
                    "pass_rate": {"min": 1.0},
                    "llm_judge_pass_rate": {"min": 1.0},
                    "llm_overall_mean": {"min": 0.8},
                    "llm_component_quality_mean": {"min": 0.8},
                    "llm_judge_error_rate": {"max": 0.0},
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_generate_answer_predictions(**kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "id": "component_eval",
                    "response": "You can contact MBZUAI admissions at admissions@mbzuai.ac.ae.",
                    "sources": [{"title": "Admissions", "url": "https://mbzuai.ac.ae/study"}],
                    "retrieved_contexts": ["Admissions email: admissions@mbzuai.ac.ae"],
                    "ui_payload": {"cards": [{"kind": "contact", "email": "admissions@mbzuai.ac.ae"}]},
                    "suggested_actions": [{"label": "Email admissions", "href": "mailto:admissions@mbzuai.ac.ae"}],
                    "followups": ["What programs are open for applications?"],
                    "response_contract": {"schema": "chat_response_v1", "has_sources": True},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"row_count": 1}

    def fake_run_llm_judge(**kwargs):
        rows_by_id = kwargs["rows_by_id"]
        assert rows_by_id["component_eval"]["ui_payload"]["cards"][0]["kind"] == "contact"
        assert rows_by_id["component_eval"]["suggested_actions"][0]["href"].startswith("mailto:")
        return {
            "component_eval": {
                "correctness": 0.95,
                "groundedness": 0.94,
                "relevance": 0.96,
                "helpfulness": 0.93,
                "completeness": 0.90,
                "citation_quality": 0.90,
                "component_quality": 0.91,
                "safety": 1.0,
                "overall": 0.94,
                "verdict": "pass",
                "reasons": ["Response and contact component are supported."],
                "error": "",
            }
        }

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)
    monkeypatch.setattr(answer_readiness, "_run_llm_judge", fake_run_llm_judge)

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        mode="local",
        judge_enabled=True,
        judge_model="test-judge",
    )

    assert report["gates"]["passed"] is True
    assert report["llm_judge"]["enabled"] is True
    assert report["overall"]["llm_judge_pass_rate"] == 1.0
    assert report["queries"][0]["llm_verdict"] == "pass"
    assert report["queries"][0]["llm_reasons"] == ["Response and contact component are supported."]


def test_answer_readiness_judge_prompt_includes_full_generated_rubric():
    from pipeline.evaluation import answer_readiness
    from pipeline.evaluation.dataset import EvalExample

    example = EvalExample(
        id="rubric",
        query="Give an admissions checklist.",
        query_type="synthesis",
        source_type="webpage",
        reference_answer="Use the official admissions page.",
        metadata={
            "expected_response_structure": "bullets",
            "answer_must_include": ["deadline"],
            "answer_must_not_include": ["unofficial scholarship"],
            "answer_should_cover": ["requirements", "deadline"],
            "expected_source_hints": ["Admissions"],
            "expected_reference_urls": ["https://mbzuai.ac.ae/study/apply"],
            "citation_requirements": ["Deadline and requirements must be cited."],
            "expected_followup_topics": ["application documents"],
            "expected_suggested_actions": ["Open application page"],
        },
    )
    prompt = answer_readiness._build_judge_prompt(
        example,
        {
            "response": "Apply using the official page [1].",
            "sources": [{"url": "https://mbzuai.ac.ae/study/apply"}],
            "followups": ["Which documents are required?"],
            "suggested_actions": [{"label": "Open application page"}],
            "response_contract": {"schema": "chat_response_v1"},
        },
    )

    assert "expected_response_structure" in prompt
    assert "https://mbzuai.ac.ae/study/apply" in prompt
    assert "citation_requirements" in prompt
    assert "Deadline and requirements must be cited." in prompt
    assert "Do not penalize an answer for including a term or detail that appears in the reference answer" in prompt


def test_answer_readiness_judge_falls_back_to_openai_on_gemini_quota(monkeypatch):
    from pipeline.evaluation import answer_readiness
    from pipeline.evaluation.dataset import EvalExample

    example = EvalExample(
        id="judge_fallback",
        query="How can I contact admissions?",
        query_type="fact",
        source_type="webpage",
        reference_answer="Use admission@mbzuai.ac.ae.",
        metadata={"answer_must_include": ["admission@mbzuai.ac.ae"]},
    )
    row = {
        "response": "Use admission@mbzuai.ac.ae for admissions questions.",
        "sources": [{"url": "https://mbzuai.ac.ae/about/faq"}],
        "retrieved_contexts": ["Admissions questions may be sent to admission@mbzuai.ac.ae."],
    }
    gemini_calls = {"count": 0}
    openai_calls = {"count": 0}

    def fake_call_judge_model(*args, **kwargs):
        gemini_calls["count"] += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    def fake_call_openai_judge_model(**kwargs):
        openai_calls["count"] += 1
        return {
            "correctness": 0.95,
            "groundedness": 0.94,
            "relevance": 0.96,
            "helpfulness": 0.90,
            "completeness": 0.90,
            "citation_quality": 0.88,
            "component_quality": 0.86,
            "safety": 1.0,
            "overall": 0.93,
            "verdict": "pass",
            "reasons": ["Fallback judge found the answer supported."],
        }

    monkeypatch.setattr(answer_readiness, "_call_judge_model", fake_call_judge_model)
    monkeypatch.setattr(answer_readiness, "_call_openai_judge_model", fake_call_openai_judge_model)

    judge = answer_readiness._judge_answer_row(
        client=object(),
        model="gemini-2.5-flash",
        fallback_model="gpt-test",
        example=example,
        row=row,
        timeout_seconds=1.0,
    )
    score = answer_readiness._score_answer_row(example, row, judge=judge)

    assert gemini_calls["count"] == 2
    assert openai_calls["count"] == 1
    assert judge["judge_provider"] == "openai"
    assert judge["judge_model"] == "gpt-test"
    assert "RESOURCE_EXHAUSTED" in judge["primary_judge_error"]
    assert judge["error"] == ""
    assert score.llm_judge_pass == 1.0
    assert score.llm_judge_provider == "openai"


def test_answer_readiness_judge_falls_back_to_openai_on_dns_connect_error(monkeypatch):
    from pipeline.evaluation import answer_readiness
    from pipeline.evaluation.dataset import EvalExample

    example = EvalExample(
        id="judge_dns_fallback",
        query="Where is MBZUAI located?",
        query_type="fact",
        source_type="webpage",
        reference_answer="MBZUAI is in Masdar City, Abu Dhabi.",
        metadata={"answer_must_include": ["Masdar City", "Abu Dhabi"]},
    )
    row = {
        "response": "MBZUAI is located in Masdar City, Abu Dhabi.",
        "sources": [{"url": "https://mbzuai.ac.ae/about/faq"}],
        "retrieved_contexts": ["MBZUAI is located in Masdar City, Abu Dhabi."],
    }
    openai_calls = {"count": 0}

    def fake_call_judge_model(*args, **kwargs):
        raise RuntimeError("[Errno 8] nodename nor servname provided, or not known")

    def fake_call_openai_judge_model(**kwargs):
        openai_calls["count"] += 1
        return {
            "correctness": 0.96,
            "groundedness": 0.95,
            "relevance": 0.96,
            "helpfulness": 0.90,
            "completeness": 0.90,
            "citation_quality": 0.88,
            "component_quality": 0.85,
            "safety": 1.0,
            "overall": 0.94,
            "verdict": "pass",
            "reasons": ["Fallback judge found the answer supported."],
        }

    monkeypatch.setattr(answer_readiness, "_call_judge_model", fake_call_judge_model)
    monkeypatch.setattr(answer_readiness, "_call_openai_judge_model", fake_call_openai_judge_model)

    judge = answer_readiness._judge_answer_row(
        client=object(),
        model="gemini-2.5-flash",
        fallback_model="gpt-test",
        example=example,
        row=row,
        timeout_seconds=1.0,
    )
    score = answer_readiness._score_answer_row(example, row, judge=judge)

    assert openai_calls["count"] == 1
    assert judge["judge_provider"] == "openai"
    assert judge["judge_model"] == "gpt-test"
    assert "nodename nor servname" in judge["primary_judge_error"]
    assert judge["error"] == ""
    assert score.llm_judge_pass == 1.0


def test_answer_readiness_judge_falls_back_to_openai_on_gemini_internal_error(monkeypatch):
    from pipeline.evaluation import answer_readiness
    from pipeline.evaluation.dataset import EvalExample

    example = EvalExample(
        id="judge_internal_fallback",
        query="Where is MBZUAI located?",
        query_type="fact",
        source_type="webpage",
        reference_answer="MBZUAI is in Masdar City, Abu Dhabi.",
        metadata={"answer_must_include": ["Masdar City", "Abu Dhabi"]},
    )
    row = {
        "response": "MBZUAI is located in Masdar City, Abu Dhabi.",
        "sources": [{"url": "https://mbzuai.ac.ae/about/faq"}],
        "retrieved_contexts": ["MBZUAI is located in Masdar City, Abu Dhabi."],
    }
    openai_calls = {"count": 0}

    def fake_call_judge_model(*args, **kwargs):
        raise RuntimeError("500 INTERNAL. An internal error has occurred.")

    def fake_call_openai_judge_model(**kwargs):
        openai_calls["count"] += 1
        return {
            "correctness": 0.96,
            "groundedness": 0.95,
            "relevance": 0.96,
            "helpfulness": 0.90,
            "completeness": 0.90,
            "citation_quality": 0.88,
            "component_quality": 0.85,
            "safety": 1.0,
            "overall": 0.94,
            "verdict": "pass",
            "reasons": ["Fallback judge found the answer supported."],
        }

    monkeypatch.setattr(answer_readiness, "_call_judge_model", fake_call_judge_model)
    monkeypatch.setattr(answer_readiness, "_call_openai_judge_model", fake_call_openai_judge_model)

    judge = answer_readiness._judge_answer_row(
        client=object(),
        model="gemini-2.5-flash",
        fallback_model="gpt-test",
        example=example,
        row=row,
        timeout_seconds=1.0,
    )
    score = answer_readiness._score_answer_row(example, row, judge=judge)

    assert openai_calls["count"] == 1
    assert judge["judge_provider"] == "openai"
    assert judge["judge_model"] == "gpt-test"
    assert "500 INTERNAL" in judge["primary_judge_error"]
    assert judge["error"] == ""
    assert score.llm_judge_pass == 1.0


def test_answer_readiness_judge_falls_back_to_openai_on_ssl_certificate_error(monkeypatch):
    from pipeline.evaluation import answer_readiness
    from pipeline.evaluation.dataset import EvalExample

    example = EvalExample(
        id="judge_ssl_fallback",
        query="Where is MBZUAI located?",
        query_type="fact",
        source_type="webpage",
        reference_answer="MBZUAI is in Masdar City, Abu Dhabi.",
        metadata={"answer_must_include": ["Masdar City", "Abu Dhabi"]},
    )
    row = {
        "response": "MBZUAI is located in Masdar City, Abu Dhabi.",
        "sources": [{"url": "https://mbzuai.ac.ae/about/faq"}],
        "retrieved_contexts": ["MBZUAI is located in Masdar City, Abu Dhabi."],
    }
    openai_calls = {"count": 0}

    def fake_call_judge_model(*args, **kwargs):
        raise RuntimeError(
            "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
            "Hostname mismatch, certificate is not valid for 'generativelanguage.googleapis.com'."
        )

    def fake_call_openai_judge_model(**kwargs):
        openai_calls["count"] += 1
        return {
            "correctness": 0.96,
            "groundedness": 0.95,
            "relevance": 0.96,
            "helpfulness": 0.90,
            "completeness": 0.90,
            "citation_quality": 0.88,
            "component_quality": 0.85,
            "safety": 1.0,
            "overall": 0.94,
            "verdict": "pass",
            "reasons": ["Fallback judge found the answer supported."],
        }

    monkeypatch.setattr(answer_readiness, "_call_judge_model", fake_call_judge_model)
    monkeypatch.setattr(answer_readiness, "_call_openai_judge_model", fake_call_openai_judge_model)

    judge = answer_readiness._judge_answer_row(
        client=object(),
        model="gemini-2.5-flash",
        fallback_model="gpt-test",
        example=example,
        row=row,
        timeout_seconds=1.0,
    )
    score = answer_readiness._score_answer_row(example, row, judge=judge)

    assert openai_calls["count"] == 1
    assert judge["judge_provider"] == "openai"
    assert judge["judge_model"] == "gpt-test"
    assert "CERTIFICATE_VERIFY_FAILED" in judge["primary_judge_error"]
    assert judge["error"] == ""
    assert score.llm_judge_pass == 1.0


def test_answer_readiness_rejects_low_row_level_citation_quality(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "bad_citation",
                "query": "Where is MBZUAI located?",
                "query_type": "fact",
                "source_type": "webpage",
                "reference_answer": "MBZUAI is in Masdar City.",
                "metadata": {
                    "answer_must_include": ["Masdar City"],
                    "expected_reference_urls": ["https://mbzuai.ac.ae/about/faq"],
                    "citation_requirements": ["Location must be cited from the FAQ page."],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "answer_gates.json"
    gates.write_text(json.dumps({"overall": {"pass_rate": {"min": 1.0}}}), encoding="utf-8")

    def fake_generate_answer_predictions(**kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "id": "bad_citation",
                    "response": "MBZUAI is located in Masdar City.",
                    "sources": [{"url": "https://mbzuai.ac.ae/generic"}],
                    "retrieved_contexts": ["MBZUAI is located in Masdar City."],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"row_count": 1}

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)
    monkeypatch.setattr(
        answer_readiness,
        "_run_llm_judge",
        lambda **kwargs: {
            "bad_citation": {
                "correctness": 0.95,
                "groundedness": 0.95,
                "relevance": 0.95,
                "helpfulness": 0.95,
                "completeness": 0.95,
                "citation_quality": 0.20,
                "component_quality": 0.95,
                "safety": 0.95,
                "overall": 0.95,
                "verdict": "pass",
                "reasons": ["The answer is correct but the source is generic."],
                "error": "",
            }
        },
    )

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        mode="local",
        judge_enabled=True,
    )

    assert report["gates"]["passed"] is False
    assert report["overall"]["expected_reference_url_pass_rate"] == 0.0
    assert report["queries"][0]["llm_judge_pass"] == 0.0
    assert report["queries"][0]["missing_expected_reference_urls"] == ["https://mbzuai.ac.ae/about/faq"]


def test_answer_readiness_accepts_common_no_answer_wording(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "noanswer",
                "query": "What is MBZUAI's Toronto office phone number?",
                "query_type": "fact",
                "source_type": "none",
                "no_answer": True,
                "reference_answer": "Insufficient evidence.",
                "metadata": {"answer_must_not_include": ["+1 416"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "answer_gates.json"
    gates.write_text(json.dumps({"overall": {"no_answer_pass_rate": {"min": 1.0}}}), encoding="utf-8")

    def fake_generate_answer_predictions(**kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "id": "noanswer",
                    "response": "I cannot find a Toronto office phone number in the available MBZUAI sources.",
                    "sources": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"row_count": 1}

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)
    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        mode="local",
        judge_enabled=False,
    )

    assert report["gates"]["passed"] is True
    assert report["overall"]["no_answer_pass_rate"] == 1.0


def test_answer_readiness_no_answer_detector_allows_local_limitations_in_cited_answers():
    from pipeline.evaluation.answer_readiness import (
        _explicitly_denies_unsupported_premise,
        _looks_like_no_answer,
    )

    answer = (
        "MBZUAI's official working hours are 8:00 AM to 6:00 PM, Monday to Thursday, "
        "and 7:30 AM to 12:00 PM on Friday. [1] "
        "The available information does not provide separate schedules for individual offices. [1]"
    )

    assert not _looks_like_no_answer(answer, "grounded")
    assert _looks_like_no_answer(
        "I couldn't confirm that specific detail from the available MBZUAI information.",
        "grounded",
    )
    cited_denial = "MBZUAI does **not** have a private airport in the available information. [1]"
    assert not _looks_like_no_answer(cited_denial, "grounded")
    assert _explicitly_denies_unsupported_premise(cited_denial)


def test_answer_readiness_accepts_cited_explicit_premise_denial_for_no_answer_gold(
    tmp_path,
    monkeypatch,
):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "unsupported-premise",
                "query": "What is the IATA code for the university's private airport?",
                "query_type": "fact",
                "source_type": "none",
                "no_answer": True,
                "reference_answer": "The sources do not establish that it has one.",
                "metadata": {"answer_must_not_include": ["DXB"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "answer_gates.json"
    gates.write_text(
        json.dumps({"overall": {"no_answer_pass_rate": {"min": 1.0}}}),
        encoding="utf-8",
    )

    def fake_generate_answer_predictions(**kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "id": "unsupported-premise",
                    "response": (
                        "The university does **not** have a private airport mentioned in the "
                        "available information. [1]"
                    ),
                    "sources": [{"url": "https://example.edu/transport", "cite_num": "1"}],
                    "response_kind": "grounded",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return {"row_count": 1}

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)
    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        mode="local",
        judge_enabled=False,
    )

    assert report["gates"]["passed"] is True
    assert report["overall"]["no_answer_pass_rate"] == 1.0


def test_answer_readiness_required_terms_match_equivalent_time_formats():
    from pipeline.evaluation.answer_readiness import _required_term_supported

    response = (
        "MBZUAI's official working hours are 8:00 AM to 6:00 PM, Monday to Thursday, "
        "and 7:30 a.m. to 12:00 p.m. on Friday."
    )

    assert _required_term_supported(response, "7.30am to 12pm on Friday")
    assert _required_term_supported(response, "8:00")
    assert _required_term_supported(response, "6:00")
    assert _required_term_supported(response, "12:00")
    assert not _required_term_supported(response, "12:30")

    support_window = "IT support is available Monday to Thursday from 8:00 AM to 5:00 PM and Friday from 8:00 AM to 12:30 PM."
    assert _required_term_supported(support_window, "12:30")
    assert _required_term_supported(support_window, "8:00")


def test_answer_readiness_required_terms_match_arabic_clitics_and_inflection():
    from pipeline.evaluation.answer_readiness import _required_term_supported

    response = (
        "يقع المقر في أبوظبي، مع مراكز في وادي السيليكون وباريس. "
        "ويعمل IFM مع الشركاء والمؤسسات الأكاديمية ومختبرات الأبحاث "
        "والشركات الناشئة."
    )

    assert _required_term_supported(response, "وادي السيليكون وباريس")
    assert _required_term_supported(
        response,
        "شركاء أكاديميون ومختبرات وشركات ناشئة",
    )
    assert _required_term_supported(
        "تُظهر الصورة شريحة/معالج حاسوبي تحيط به مسارات إلكترونية.",
        "شريحة معالِج",
    )
    assert not _required_term_supported(
        response,
        "العلوم والنطاق والقيمة الاجتماعية",
    )


def test_answer_readiness_required_terms_match_dash_and_analysis_spelling_variants():
    from pipeline.evaluation.answer_readiness import _required_term_supported

    assert _required_term_supported(
        "The image shows the MBZUAI Faculty Portfolio 2024–2025.",
        "MBZUAI Faculty Portfolio 2024-2025",
    )
    assert _required_term_supported(
        "The collected data are automatically analyzed.",
        "automatically analysed",
    )
    assert _required_term_supported(
        "The output is a generative-AI-assisted supplier communication draft.",
        "generative AI",
    )
    assert _required_term_supported(
        "انتُخب رئيسًا لدولة الإمارات العربية المتحدة.",
        "رئيساً للدولة",
    )
    assert _required_term_supported(
        "ينبغي أن يكون المتقدمون باحثين ذوي خبرة.",
        "باحثون ذوو خبرة",
    )
    assert _required_term_supported(
        "لديهم رغبة قوية في الإسهام في العلم والإنسانية.",
        "المساهمة في العلم والإنسانية",
    )
    assert not _required_term_supported(
        "The collected data are manually reviewed.",
        "automatically analysed",
    )


def test_answer_readiness_supports_multilingual_alternative_term_groups():
    from pipeline.evaluation.answer_readiness import _required_terms_result

    response = (
        "المؤهل المطلوب هو درجة ماجستير في السياسات العامة وإدارة التعليم "
        "العالي والقانون أو مجال ذي صلة."
    )
    metadata = {
        "answer_must_include_any_groups": [
            ["public policy/guidelines", "السياسات العامة"],
            ["law", "القانون"],
        ]
    }

    missing, coverage = _required_terms_result(
        response,
        ["درجة ماجستير"],
        metadata,
    )

    assert missing == []
    assert coverage == pytest.approx(1.0)


def test_answer_readiness_accepts_faithful_cross_lingual_ifm_labels():
    from pipeline.evaluation.answer_readiness import _required_terms_result

    response = (
        "المعهد مركز عالمي مكرّس لدفع علم النماذج التأسيسية وحجمها وقيمتها الاجتماعية، "
        "ومقره في أبوظبي وله مراكز في Silicon Valley وParis، ويتعاون مع المؤسسات "
        "الأكاديمية ومختبرات الأبحاث والشركات الناشئة."
    )
    missing, coverage = _required_terms_result(
        response,
        ["أبوظبي", "شركاء أكاديميون ومختبرات وشركات ناشئة"],
        {
            "answer_must_include_any_groups": [
                [
                    "العلوم والنطاق والقيمة الاجتماعية",
                    "علم النماذج التأسيسية وحجمها وقيمتها الاجتماعية",
                    "science, scale, and social value",
                ],
                [
                    "وادي السيليكون وباريس",
                    "Silicon Valley وParis",
                    "Silicon Valley and Paris",
                ],
            ]
        },
    )

    assert missing == []
    assert coverage == pytest.approx(1.0)


def test_answer_readiness_accepts_faithful_media_instruction_paraphrases():
    from pipeline.evaluation.answer_readiness import _required_terms_result

    arabic_response = (
        "توضح الصورة شريحة معالج، وأن التدريب يتطلب استثمارات كبيرة في العتاد "
        "وقدرة حسابية عالية ومعالجات مهيأة لعمليات المصفوفات."
    )
    missing, coverage = _required_terms_result(
        arabic_response,
        ["شريحة معالِج", "عمليات المصفوفات"],
        {
            "answer_must_include_any_groups": [
                ["قدرة الحوسبة", "قدرة حاسوبية", "قدرة حسابية"],
                ["متطلبات العتاد", "استثمارات كبيرة في العتاد", "العتاد (hardware)"],
            ]
        },
    )
    assert missing == []
    assert coverage == pytest.approx(1.0)

    missing, coverage = _required_terms_result(
        "Complete the required account-creation fields, tick the reCAPTCHA, and click Submit.",
        [],
        {
            "answer_must_include_any_groups": [
                [
                    "required information",
                    "required registration fields",
                    "required details",
                    "required account-creation fields",
                ],
                ["reCAPTCHA box", "complete the reCAPTCHA", "tick the reCAPTCHA"],
                ["Submit button", "click Submit", "submit the form"],
            ]
        },
    )
    assert missing == []
    assert coverage == pytest.approx(1.0)


def test_answer_readiness_accepts_governed_alternate_reference_url():
    from pipeline.evaluation.answer_readiness import _expected_reference_url_pass

    metadata = {
        "expected_reference_urls": [
            "https://mbzuai.ac.ae/study/msc-programs",
        ],
        "alternate_expected_reference_urls": {
            "https://mbzuai.ac.ae/study/msc-programs": [
                "https://mbzuai.ac.ae/study/master-in-applied-ai",
            ]
        },
    }
    row = {
        "sources": [
            {"url": "https://mbzuai.ac.ae/study/master-in-applied-ai/"},
        ]
    }

    assert _expected_reference_url_pass(metadata, row, no_answer=False) == 1.0


def test_answer_readiness_reports_each_unsatisfied_alternative_group_once():
    from pipeline.evaluation.answer_readiness import _required_terms_result

    missing, coverage = _required_terms_result(
        "تمر العملية عبر المعالجة المسبقة ثم التنظيف وإزالة التكرار.",
        ["المعالجة المسبقة", "التنظيف", "إزالة التكرار"],
        {
            "answer_must_include_any_groups": [
                ["التصفية", "الترشيح"],
            ]
        },
    )

    assert missing == ["التصفية OR الترشيح"]
    assert coverage == pytest.approx(0.75)


def test_http_answer_readiness_does_not_send_probe_header_by_default(monkeypatch):
    from pipeline.evaluation import answer_readiness
    from pipeline.evaluation.dataset import EvalExample

    captured_headers = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"response": "ok", "sources": []}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured_headers.update(dict(request.header_items()))
        return FakeResponse()

    monkeypatch.setattr(answer_readiness.urllib.request, "urlopen", fake_urlopen)

    answer_readiness._post_chat_request(
        endpoint="http://127.0.0.1:8000/telegram-chat",
        example=EvalExample(id="q1", query="Hello", query_type="fact"),
        auth_token=None,
        timeout_seconds=1.0,
    )

    assert "x-health-probe" not in {key.casefold(): value for key, value in captured_headers.items()}
    assert {key.casefold(): value for key, value in captured_headers.items()}["x-eval-request"] == "true"


def test_answer_readiness_flexible_must_include_matching(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "msc_ml",
                "query": "What does the MSc in ML cover?",
                "query_type": "synthesis",
                "source_type": "webpage",
                "reference_answer": "The Master of Science in Machine Learning covers ML coursework.",
                "metadata": {
                    "answer_must_include": ["Master of Science in Machine Learning", "MSc ML"],
                    "answer_must_include_min_coverage": 0.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "answer_gates.json"
    gates.write_text(json.dumps({"overall": {"pass_rate": {"min": 1.0}}}), encoding="utf-8")

    def fake_generate_answer_predictions(**kwargs):
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "id": "msc_ml",
                    "response": "The M.Sc. in Machine Learning covers core machine learning topics.",
                    "sources": [{"url": "https://mbzuai.ac.ae/study/msc-programs/machine-learning"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)
    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        mode="local",
        judge_enabled=False,
    )

    assert report["gates"]["passed"] is True
    assert report["queries"][0]["must_include_coverage"] == 0.5


def test_answer_readiness_filters_stale_resume_predictions(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "Where is MBZUAI located?",
                "query_type": "fact",
                "source_type": "webpage",
                "reference_answer": "MBZUAI is located in Masdar City, Abu Dhabi.",
                "metadata": {"answer_must_include": ["Masdar City", "Abu Dhabi"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "gates.json"
    gates.write_text(
        json.dumps({"overall": {"prediction_integrity_error_rate": {"max": 0.0}, "support_present_rate": {"min": 1.0}}}),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "id": "q1",
                "response": "Stale response from an older dataset.",
                "retrieved_contexts": ["old context"],
                "metadata": {"eval_dataset_fingerprint": "stale"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state = {}

    def fake_generate_answer_predictions(**kwargs):
        existing = Path(kwargs["output_path"]).read_text(encoding="utf-8")
        state["stale_file_was_filtered"] = "Stale response" not in existing
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "id": "q1",
                    "response": "MBZUAI is located in Masdar City, Abu Dhabi.",
                    "retrieved_contexts": ["MBZUAI is located in Masdar City, Abu Dhabi."],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        predictions_path=predictions,
        resume_predictions=True,
    )

    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert state["stale_file_was_filtered"] is True
    assert report["prediction_integrity"]["ok"] is True
    assert report["gates"]["passed"] is True
    assert rows[0]["metadata"]["eval_dataset_fingerprint"] == report["dataset_fingerprint"]
    assert rows[0]["metadata"]["eval_example_fingerprint"]


def test_answer_readiness_filters_partial_resume_predictions(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "q1",
                "query": "What support do funded students receive?",
                "query_type": "fact",
                "source_type": "webpage",
                "reference_answer": "Funded students receive tuition and stipend support.",
                "metadata": {"answer_must_include": ["tuition", "stipend"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "gates.json"
    gates.write_text(
        json.dumps({"overall": {"pass_rate": {"min": 1.0}}}),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    examples = answer_readiness.load_eval_examples(dataset)
    dataset_fingerprint = answer_readiness._dataset_fingerprint(examples)
    partial_row = answer_readiness._stamp_prediction_row(
        {
            "id": "q1",
            "response": "Funded students receive tuition",
            "partial": True,
            "finish_reason": "stream_error",
            "response_contract": {"completed": False, "partial": True},
        },
        example=examples[0],
        dataset_fingerprint=dataset_fingerprint,
        config_name="default",
        work_dir=tmp_path,
        backend="local_indexing_answer_generation",
        mode="local",
    )
    predictions.write_text(json.dumps(partial_row) + "\n", encoding="utf-8")
    state = {}

    def fake_generate_answer_predictions(**kwargs):
        state["partial_file_was_filtered"] = not Path(kwargs["output_path"]).read_text(
            encoding="utf-8"
        ).strip()
        Path(kwargs["output_path"]).write_text(
            json.dumps(
                {
                    "id": "q1",
                    "response": "Funded students receive tuition and stipend support.",
                    "retrieved_contexts": ["Funded students receive tuition and stipend support."],
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(answer_readiness, "generate_answer_predictions", fake_generate_answer_predictions)

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        predictions_path=predictions,
        resume_predictions=True,
        judge_enabled=False,
    )

    assert state["partial_file_was_filtered"] is True
    assert report["prediction_integrity"]["ok"] is True
    assert report["gates"]["passed"] is True


def test_answer_readiness_websocket_mode_uses_production_chat_frames(tmp_path, monkeypatch):
    from pipeline.evaluation import answer_readiness

    dataset = tmp_path / "readiness.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "ws_q1",
                "query": "Where is MBZUAI located?",
                "query_type": "fact",
                "source_type": "webpage",
                "reference_answer": "MBZUAI is located in Masdar City, Abu Dhabi.",
                "metadata": {"answer_must_include": ["Masdar City", "Abu Dhabi"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gates = tmp_path / "gates.json"
    gates.write_text(
        json.dumps({"overall": {"prediction_integrity_error_rate": {"max": 0.0}, "support_present_rate": {"min": 1.0}}}),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    captured = {}

    def fake_websocket_chat_request(**kwargs):
        captured.update(kwargs)
        return {
            "id": kwargs["example"].id,
            "query_type": kwargs["example"].query_type,
            "source_type": kwargs["example"].source_type,
            "user_input": kwargs["example"].query,
            "response": "MBZUAI is located in Masdar City, Abu Dhabi [1].",
            "sources": [{"url": "https://mbzuai.ac.ae/about/"}],
            "followups": ["What programs does MBZUAI offer?"],
            "suggested_actions": [{"label": "Open campus page", "href": "https://mbzuai.ac.ae/about/"}],
            "response_contract": {"schema": "chat_response_v1", "has_sources": True},
            "metadata": {
                "backend": "production_chat_websocket",
                "endpoint": kwargs["endpoint"],
                "eval_request_mode": kwargs["eval_request_mode"],
                "latency_ms": 12.3,
                "error": "",
            },
            "latency_ms": 12.3,
            "error": "",
        }

    monkeypatch.setattr(answer_readiness, "_websocket_chat_request", fake_websocket_chat_request)
    events = []

    report = answer_readiness.evaluate_answer_readiness(
        config_name="default",
        work_dir=tmp_path,
        dataset_path=dataset,
        gates_path=gates,
        predictions_path=predictions,
        mode="websocket",
        endpoint="http://127.0.0.1:8000/chat",
        widget_key="widget-key",
        judge_enabled=False,
        progress_callback=lambda event, payload: events.append((event, dict(payload))),
    )

    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert captured["endpoint"] == "ws://127.0.0.1:8000/chat"
    assert captured["widget_key"] == "widget-key"
    assert captured["eval_request_mode"] is True
    assert report["backend"] == "production_chat_websocket"
    assert report["endpoint"] == "ws://127.0.0.1:8000/chat"
    assert report["prediction_integrity"]["ok"] is True
    assert report["gates"]["passed"] is True
    assert rows[0]["metadata"]["eval_backend"] == "production_chat_websocket"
    assert rows[0]["metadata"]["eval_mode"] == "websocket"
    assert rows[0]["metadata"]["eval_endpoint"] == "ws://127.0.0.1:8000/chat"
    assert any(event == "answer_prediction_done" for event, _payload in events)


def test_validate_eval_set_warns_on_suspicious_no_answer_metadata(tmp_path):
    from pipeline.evaluation.dataset_tools import validate_eval_examples

    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "bad_noanswer",
                "query": "What PhD programs does MBZUAI offer?",
                "query_type": "synthesis",
                "source_type": "none",
                "no_answer": True,
                "reference_answer": "MBZUAI offers PhD programs in several AI fields.",
                "metadata": {
                    "answer_must_include": ["PhD"],
                    "expected_reference_urls": ["https://mbzuai.ac.ae/study/phd-programs"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = validate_eval_examples(dataset)
    reasons = {warning["reason"] for warning in report["warnings"]}
    assert "no_answer example lists expected official reference URLs; audit whether this should be answerable" in reasons
    assert (
        "no_answer example has required answer facts but the reference answer does not read like an abstention"
        in reasons
    )
    assert "no_answer reference answer does not look like an abstention" in reasons


def test_llm_generated_answer_gate_matches_generated_dataset_slices():
    from pipeline.evaluation.dataset import load_eval_examples

    project_root = Path(__file__).resolve().parents[2]
    dataset = project_root / "eval" / "mbzuai_gold" / "mbzuai_llm_generated_v1.jsonl"
    gates = project_root / "eval" / "gates" / "answer_readiness_gate.llm_generated_v1.json"
    examples = load_eval_examples(dataset)
    gate_payload = json.loads(gates.read_text(encoding="utf-8"))
    query_types = {example.query_type for example in examples}
    source_types = {example.source_type for example in examples}
    tags = {
        tag
        for example in examples
        for tag in (example.metadata.get("benchmark_tags") or [])
    }

    assert not (set(gate_payload.get("by_query_type") or {}) - query_types)
    assert not (set(gate_payload.get("by_source_type") or {}) - source_types)
    assert not (set(gate_payload.get("by_benchmark_tag") or {}) - tags)


def test_release_readiness_v2_has_comprehensive_answer_rules():
    from pipeline.evaluation.dataset import load_eval_examples

    dataset = Path(__file__).resolve().parents[2] / "eval" / "mbzuai_gold" / "mbzuai_release_readiness_v2.jsonl"
    examples = load_eval_examples(dataset)
    query_types = {example.query_type for example in examples}
    source_types = {example.source_type for example in examples}
    tags = {
        tag
        for example in examples
        for tag in (example.metadata.get("benchmark_tags") or [])
    }

    assert len(examples) >= 44
    assert query_types == {"fact", "scoped", "synthesis", "multimodal"}
    assert {"webpage", "pdf", "mixed", "none"} <= source_types
    assert "contact_lookup" in tags
    assert "abstention" in tags
    assert "deep_answer" in tags
    assert "multi_doc" in tags
    assert "citation_quality" in tags
    assert "followup_quality" in tags
    assert "release_readiness_v2" in tags
    assert sum(1 for example in examples if example.no_answer) >= 5
    assert sum(1 for example in examples if "deep_answer" in (example.metadata.get("benchmark_tags") or [])) >= 5
    for example in examples:
        metadata = example.metadata
        assert metadata.get("release_suite") == "release_readiness_v2"
        if example.no_answer:
            assert metadata.get("answer_must_not_include")
        else:
            assert metadata.get("answer_must_include")
        if "deep_answer" in (metadata.get("benchmark_tags") or []):
            assert metadata.get("answer_should_cover")
            assert metadata.get("expected_source_hints")
            assert metadata.get("expected_followup_topics")


def test_llm_qa_generator_normalizes_rows_with_rubrics_and_gold_ids(tmp_path, monkeypatch):
    module = _load_script_module("scripts/generate_llm_qa_eval_set.py")
    work_dir = tmp_path / "run"
    bundle_dir = work_dir / "stage_outputs" / "build_retrieval_bundle"
    bundle_dir.mkdir(parents=True)
    atomic_write_json(
        bundle_dir / "retrieval_bundle.json",
        {
            "version": 4,
            "chunk_records": [
                {
                    "id": "chunk-campus",
                    "record_type": "chunk",
                    "text": "MBZUAI is located in Masdar City, Abu Dhabi. Student accommodation, parking, shuttle transport, and campus facilities are available.",
                    "document_id": "doc-campus",
                    "document_title": "Campus Life",
                    "document_type": "webpage",
                    "source_url": "https://mbzuai.ac.ae/campus-life",
                    "linked_parent_ids": ["parent-campus"],
                },
                {
                    "id": "chunk-admissions",
                    "record_type": "chunk",
                    "text": "The general admissions email is admission@mbzuai.ac.ae and the undergraduate admissions email is ug.admission@mbzuai.ac.ae.",
                    "document_id": "doc-admissions",
                    "document_title": "Admissions",
                    "document_type": "webpage",
                    "source_url": "https://mbzuai.ac.ae/study",
                    "linked_parent_ids": ["parent-admissions"],
                },
            ],
            "parent_records": [
                {
                    "id": "parent-campus",
                    "record_type": "parent",
                    "text": "Campus Life includes location, accommodation, parking, shuttle transport, and facilities.",
                    "document_id": "doc-campus",
                    "document_title": "Campus Life",
                    "document_type": "webpage",
                    "source_url": "https://mbzuai.ac.ae/campus-life",
                    "child_chunk_ids": ["chunk-campus"],
                },
                {
                    "id": "parent-admissions",
                    "record_type": "parent",
                    "text": "Admissions contact information for graduate and undergraduate applicants.",
                    "document_id": "doc-admissions",
                    "document_title": "Admissions",
                    "document_type": "webpage",
                    "source_url": "https://mbzuai.ac.ae/study",
                    "child_chunk_ids": ["chunk-admissions"],
                },
            ],
            "media_records": [],
            "fact_records": [],
            "answer_records": [],
        },
    )

    def fake_generate_json(**kwargs):
        assert kwargs["use_search"] is True
        return {
            "items": [
                {
                    "query": "Give a detailed campus arrival guide for a new MBZUAI student.",
                    "query_type": "synthesis",
                    "source_type": "webpage",
                    "reference_answer": "MBZUAI is in Masdar City, Abu Dhabi, and students should know about accommodation, parking, shuttle transport, and campus facilities.",
                    "expected_response_structure": "bullets",
                    "answer_must_include": ["Masdar City", "accommodation", "parking", "shuttle"],
                    "answer_must_not_include": ["Singapore campus"],
                    "answer_should_cover": ["location", "accommodation", "parking", "transport", "facilities"],
                    "expected_source_hints": ["Campus Life"],
                    "expected_reference_urls": ["https://mbzuai.ac.ae/campus-life"],
                    "expected_followup_topics": ["housing", "parking"],
                    "expected_suggested_actions": ["View campus facilities"],
                    "citation_requirements": ["Cite the campus life page for arrival logistics."],
                    "gold_chunk_ids": ["chunk-campus"],
                    "gold_parent_ids": ["parent-campus"],
                    "notes": "Checks multi-fact campus guidance.",
                    "difficulty": "hard",
                    "scenario": "campus_arrival",
                },
                {
                    "query": "What is MBZUAI's Mars campus phone number?",
                    "query_type": "fact",
                    "source_type": "none",
                    "no_answer": True,
                    "reference_answer": "Insufficient evidence.",
                    "answer_must_not_include": ["Mars campus", "+000"],
                    "expected_response_structure": "short_answer",
                    "answer_should_cover": ["state that the source corpus does not provide this"],
                    "expected_followup_topics": ["official contacts"],
                    "citation_requirements": ["Do not cite unrelated generic MBZUAI pages as support."],
                    "gold_chunk_ids": [],
                    "gold_parent_ids": [],
                    "notes": "Unsupported campus abstention.",
                    "difficulty": "medium",
                    "scenario": "unsupported_contact",
                },
            ]
        }

    monkeypatch.setattr(module, "_generate_json", fake_generate_json)
    args = module.build_parser().parse_args(
        [
            "--work-dir",
            str(work_dir),
            "--count",
            "2",
            "--min-count",
            "2",
            "--batch-size",
            "2",
            "--max-sources",
            "2",
            "--sources-per-batch",
            "2",
        ]
    )
    args.model = "test-model"
    examples = module.generate_eval_set(args)

    assert len(examples) == 2
    assert examples[0].query_type == "synthesis"
    assert examples[0].gold_chunk_ids == ["chunk-campus"]
    assert examples[0].metadata["answer_should_cover"]
    assert examples[0].metadata["expected_reference_urls"] == ["https://mbzuai.ac.ae/campus-life"]
    assert examples[1].no_answer is True
    assert examples[1].source_type == "none"
    assert examples[1].gold_chunk_ids == []


def test_release_check_manifest_and_promotion(tmp_path, monkeypatch):
    from pipeline.core import release

    work_dir = tmp_path / "runs" / "run_001"
    (work_dir / "stage_outputs" / "upload_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "upload_graph").mkdir(parents=True)
    (work_dir / "stage_outputs" / "format_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "promote_graph").mkdir(parents=True)
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        _valid_modern_vector_manifest(),
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_graph" / "neo4j_upload_manifest.json",
        {
            "neo4j_namespace": "mbzuai:run_001",
            "neo4j_database": "neo4j",
            "graph_type": "promoted_semantic_graph",
            "node_count": 20,
            "edge_count": 30,
            "verification": {"expected_nodes": 20, "actual_nodes": 20, "expected_edges": 30, "actual_edges": 30},
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
        {
            "version": 5,
            "stats": {
                "chunk_count": 10,
                "parent_count": 3,
                "media_count": 0,
                "fact_count": 4,
                "evidence_span_count": 5,
                "summary_count": 2,
                "assertion_count": 2,
                "answer_count": 2,
                "lexical_count": 21,
            },
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph.json",
        {
            "nodes": [
                {"id": "chunk-1", "node_type": "chunk"},
                {"id": "assertion-1", "node_type": "relation_assertion"},
            ],
            "edges": [{
                "id": "edge-1",
                "source_id": "chunk-1",
                "target_id": "assertion-1",
                "edge_type": "CHUNK_SUPPORTS_ASSERTION",
            }],
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph_index.json",
        {"outgoing_edge_ids": {"n1": []}},
    )
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")

    fake_audit = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        to_dict=lambda: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(release, "audit_run", lambda path: fake_audit)
    monkeypatch.setattr(
        release,
        "assess_production_readiness",
        lambda *args, **kwargs: {"ok": True, "error_count": 0, "warning_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        release,
        "validate_eval_examples",
        lambda *args, **kwargs: {"ok": True, "summary": {"query_count": 1}, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        release,
        "evaluate_retrieval_dataset",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"chunk_hit_at_5": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )
    monkeypatch.setattr(
        release,
        "evaluate_answer_readiness",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"pass_rate": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )

    manifest, passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        splits=["selection"],
        skip_stage_validation=True,
    )
    assert passed
    assert manifest["vector_index"]["uploaded"]["summaries"] == 2
    assert manifest["vector_index"]["expected_uploads"]["sparse_summaries"] == 2
    assert manifest["answer_evaluation"]["overall"]["pass_rate"] == 1.0
    manifest_path = release.write_release_manifest(manifest, work_dir)
    active_path = release.promote_release_manifest(
        manifest_path=manifest_path,
        active_release_file=tmp_path / "runs" / "active_release.json",
    )
    assert load_json_safe(active_path)["active_release_manifest"] == str(manifest_path)
    assert load_json_safe(manifest_path)["promoted"] is True


def test_release_check_accepts_local_json_graph_without_neo4j(tmp_path, monkeypatch):
    from pipeline.core import release

    work_dir = tmp_path / "runs" / "run_local_graph"
    (work_dir / "stage_outputs" / "upload_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "format_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "promote_graph").mkdir(parents=True)
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        _valid_modern_vector_manifest(),
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
        {
            "version": 5,
            "stats": {
                "chunk_count": 10,
                "parent_count": 3,
                "media_count": 0,
                "fact_count": 4,
                "evidence_span_count": 5,
                "summary_count": 2,
                "assertion_count": 2,
                "answer_count": 2,
                "lexical_count": 21,
            },
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph.json",
        {
            "nodes": [
                {"id": "n1", "node_type": "chunk"},
                {"id": "a1", "node_type": "relation_assertion"},
            ],
            "edges": [{
                "id": "edge-1",
                "source_id": "n1",
                "target_id": "a1",
                "edge_type": "CHUNK_SUPPORTS_ASSERTION",
            }],
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph_index.json",
        {"outgoing_edge_ids": {"n1": []}},
    )
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")

    fake_audit = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        to_dict=lambda: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(release, "audit_run", lambda path: fake_audit)
    monkeypatch.setattr(
        release,
        "assess_production_readiness",
        lambda *args, **kwargs: {"ok": True, "error_count": 0, "warning_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        release,
        "validate_eval_examples",
        lambda *args, **kwargs: {"ok": True, "summary": {"query_count": 1}, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        release,
        "evaluate_retrieval_dataset",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"chunk_hit_at_5": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )
    monkeypatch.setattr(
        release,
        "evaluate_answer_readiness",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"pass_rate": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )

    manifest, passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        skip_stage_validation=True,
    )

    assert passed
    assert manifest["knowledge_graph"]["store_backend"] == "local_json"
    assert manifest["knowledge_graph"]["node_count"] == 2
    assert manifest["knowledge_graph"]["assertion_count"] == 1
    assert not any("Neo4j" in error for error in manifest["errors"])


def test_release_check_blocks_missing_summary_upload(tmp_path, monkeypatch):
    from pipeline.core import release

    work_dir = tmp_path / "runs" / "run_missing_summary"
    (work_dir / "stage_outputs" / "upload_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "upload_graph").mkdir(parents=True)
    (work_dir / "stage_outputs" / "format_retrieval").mkdir(parents=True)
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        {
            "index_name": "idx",
            "sparse_index_name": "idx-sparse",
            "namespaces": {
                "chunks": "chunks",
                "parents": "parents",
                "facts": "facts",
                "evidence_spans": "evidence_spans",
                "summaries": "summaries",
                "assertions": "assertions",
            },
            "uploaded": {
                "chunks": 10,
                "parents": 3,
                "facts": 4,
                "evidence_spans": 5,
                "summaries": 0,
                "assertions": 2,
                "sparse_chunks": 10,
                "sparse_parents": 3,
                "sparse_facts": 4,
                "sparse_evidence_spans": 5,
                "sparse_summaries": 0,
                "sparse_assertions": 2,
            },
            "verification": {"dense": {"failures": []}, "sparse": {"failures": []}},
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_graph" / "neo4j_upload_manifest.json",
        {
            "neo4j_namespace": "mbzuai:run_missing_summary",
            "neo4j_database": "neo4j",
            "graph_type": "promoted_semantic_graph",
            "node_count": 20,
            "edge_count": 30,
            "verification": {"expected_nodes": 20, "actual_nodes": 20, "expected_edges": 30, "actual_edges": 30},
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
        {
            "version": 5,
            "stats": {
                "chunk_count": 10,
                "parent_count": 3,
                "media_count": 0,
                "fact_count": 4,
                "evidence_span_count": 5,
                "summary_count": 2,
                "assertion_count": 2,
                "answer_count": 2,
                "lexical_count": 21,
            },
        },
    )
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")

    fake_audit = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        to_dict=lambda: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(release, "audit_run", lambda path: fake_audit)
    monkeypatch.setattr(
        release,
        "assess_production_readiness",
        lambda *args, **kwargs: {"ok": True, "error_count": 0, "warning_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        release,
        "validate_eval_examples",
        lambda *args, **kwargs: {"ok": True, "summary": {"query_count": 1}, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        release,
        "evaluate_retrieval_dataset",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"chunk_hit_at_5": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )
    monkeypatch.setattr(
        release,
        "evaluate_answer_readiness",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"pass_rate": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )

    manifest, passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        skip_stage_validation=True,
    )

    assert not passed
    assert manifest["status"] == "failed"
    assert any("summaries" in error for error in manifest["errors"])


def test_release_check_blocks_failed_answer_readiness(tmp_path, monkeypatch):
    from pipeline.core import release

    work_dir = tmp_path / "runs" / "run_answer_failed"
    _write_valid_release_artifacts(work_dir)
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    answer_gates = tmp_path / "answer_gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")
    answer_gates.write_text("{}", encoding="utf-8")

    fake_audit = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        to_dict=lambda: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(release, "audit_run", lambda path: fake_audit)
    monkeypatch.setattr(
        release,
        "assess_production_readiness",
        lambda *args, **kwargs: {"ok": True, "error_count": 0, "warning_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        release,
        "validate_eval_examples",
        lambda *args, **kwargs: {"ok": True, "summary": {"query_count": 1}, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        release,
        "evaluate_retrieval_dataset",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"chunk_hit_at_5": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )
    monkeypatch.setattr(
        release,
        "evaluate_answer_readiness",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"pass_rate": 0.0},
            "gates": {"path": str(answer_gates), "passed": False, "failures": [{"metric": "pass_rate"}]},
        },
    )

    manifest, passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        answer_dataset_path=dataset,
        answer_gates_path=answer_gates,
        skip_stage_validation=True,
    )

    assert not passed
    assert manifest["status"] == "failed"
    assert manifest["answer_evaluation"]["gates"]["passed"] is False
    assert any("Answer readiness gates did not pass" in error for error in manifest["errors"])


def test_release_check_fails_closed_when_answer_readiness_is_skipped(tmp_path, monkeypatch):
    from pipeline.core import release

    work_dir = tmp_path / "runs" / "run_answer_skipped"
    _write_valid_release_artifacts(work_dir)
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")

    fake_audit = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        to_dict=lambda: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(release, "audit_run", lambda path: fake_audit)
    monkeypatch.setattr(
        release,
        "assess_production_readiness",
        lambda *args, **kwargs: {"ok": True, "error_count": 0, "warning_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        release,
        "validate_eval_examples",
        lambda *args, **kwargs: {"ok": True, "summary": {"query_count": 1}, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        release,
        "evaluate_retrieval_dataset",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"chunk_hit_at_5": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )

    manifest, passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        skip_answer_readiness=True,
        skip_stage_validation=True,
    )
    assert not passed
    assert manifest["status"] == "failed"
    assert manifest["answer_evaluation"]["skipped"] is True
    assert manifest["answer_evaluation"]["gates"]["passed"] is False

    waived_manifest, waived_passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        skip_answer_readiness=True,
        allow_answer_readiness_waiver=True,
        answer_readiness_waiver_reason="documented emergency rollback validation",
        skip_stage_validation=True,
    )
    assert waived_passed
    assert waived_manifest["status"] == "passed_with_waiver"
    assert waived_manifest["answer_evaluation"]["waived"] is True
    assert waived_manifest["answer_evaluation"]["waiver_reason"]


def test_release_manifest_validation_resolves_release_scoped_namespaces():
    from pipeline.core.release import _verify_vector_manifest_matches_config
    from pipeline.stages.embedders.gemini_pinecone_embedder import _resolve_upload_namespaces

    embedder = {
        "namespace_strategy": "release",
        "namespace_release_template": "{base}--{release_id}",
        "pinecone_index": "dense-v3",
        "pinecone_sparse_index": "sparse-v3",
        "namespace_chunks": "chunks",
        "namespace_parents": "parents",
        "namespace_media": "media",
        "namespace_facts": "facts",
        "namespace_evidence_spans": "evidence-spans",
        "namespace_summaries": "summaries",
        "namespace_assertions": "assertions",
        "namespace_entities": "entities",
        "namespace_communities": "communities",
    }
    run_id = "candidate/2026-07-11"
    manifest = {
        "index_name": "dense-v3",
        "sparse_index_name": "sparse-v3",
        "namespace_strategy": "release",
        "namespace_release_id": run_id,
        "namespaces": _resolve_upload_namespaces(embedder, run_id=run_id),
    }

    assert _verify_vector_manifest_matches_config(manifest, {"embedder": embedder}) == []

    manifest["namespaces"]["chunks"] = "chunks--wrong-release"
    errors = _verify_vector_manifest_matches_config(manifest, {"embedder": embedder})
    assert any("does not match uploaded chunks namespace" in error for error in errors)


def test_release_namespace_identity_is_collision_safe_and_template_is_mandatory():
    from pipeline.stages.embedders.gemini_pinecone_embedder import (
        _release_namespace_token,
        _resolve_upload_namespaces,
    )

    first = _release_namespace_token("candidate/a")
    second = _release_namespace_token("candidate-a")
    assert first != second
    assert first == _release_namespace_token("candidate/a")

    with pytest.raises(ValueError, match="exact.*release_id"):
        _resolve_upload_namespaces(
            {
                "namespace_strategy": "release",
                "namespace_release_template": "{base}",
            },
            run_id="candidate/a",
        )


def test_modern_vector_manifest_fails_closed_and_rejects_self_reported_partial_upload(tmp_path):
    from pipeline.core.release import _verify_vector_manifest

    bundle_stats = {
        "chunk_count": 10,
        "parent_count": 3,
        "media_count": 0,
        "fact_count": 4,
        "evidence_span_count": 5,
        "summary_count": 2,
        "assertion_count": 2,
    }
    manifest = _valid_modern_vector_manifest()
    assert _verify_vector_manifest(manifest, bundle_stats, tmp_path) == []

    incomplete = json.loads(json.dumps(manifest))
    incomplete.pop("index_name")
    incomplete["namespaces"].pop("facts")
    incomplete.pop("verification")
    errors = _verify_vector_manifest(incomplete, bundle_stats, tmp_path)
    assert any("missing index_name" in error for error in errors)
    assert any("missing namespaces" in error for error in errors)
    assert any("missing namespace verification" in error for error in errors)

    partial = json.loads(json.dumps(manifest))
    partial["planned"]["chunks"] = 9
    partial["uploaded"]["chunks"] = 9
    partial["verification"]["dense"]["expected"]["chunks"] = 9
    partial["verification"]["dense"]["actual"]["chunks"] = 9
    errors = _verify_vector_manifest(partial, bundle_stats, tmp_path)
    assert any("uploaded count mismatch for 'chunks'" in error for error in errors)
    assert any("does not match canonical count for 'chunks'" in error for error in errors)


def test_release_build_validates_raw_namespace_bases_without_double_suffix(tmp_path, monkeypatch):
    from pipeline.core import release
    from pipeline.stages.embedders.gemini_pinecone_embedder import _resolve_upload_namespaces

    work_dir = tmp_path / "runs" / "candidate-001"
    _write_valid_release_artifacts(work_dir)
    upload_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    upload_manifest = load_json_safe(upload_path, {})
    embedder = {
        "namespace_strategy": "release",
        "namespace_release_template": "{base}--{release_id}",
        "pinecone_index": "idx",
        "pinecone_sparse_index": "idx-sparse",
        "namespace_chunks": "chunks",
        "namespace_parents": "parents",
        "namespace_media": "media",
        "namespace_facts": "facts",
        "namespace_evidence_spans": "evidence_spans",
        "namespace_summaries": "summaries",
        "namespace_assertions": "assertions",
        "namespace_entities": "entities",
        "namespace_communities": "communities",
        "enable_dense_facts": True,
    }
    resolved_namespaces = _resolve_upload_namespaces(embedder, run_id=work_dir.name)
    upload_manifest["namespace_strategy"] = "release"
    upload_manifest["namespace_release_id"] = work_dir.name
    upload_manifest["namespaces"] = resolved_namespaces
    for label in ("dense", "sparse"):
        report = upload_manifest["verification"][label]
        report["expected"] = {
            resolved_namespaces[key]: value
            for key, value in upload_manifest["planned"].items()
            if not key.startswith("sparse_") and label == "dense" and value > 0
        } if label == "dense" else {
            resolved_namespaces[key.removeprefix("sparse_")]: value
            for key, value in upload_manifest["planned"].items()
            if key.startswith("sparse_") and value > 0
        }
        report["actual"] = dict(report["expected"])
    atomic_write_json(upload_path, upload_manifest)
    config = {
        "pipeline": {},
        "embedder": embedder,
        "retrieval": {},
        "graph": {"store_backend": "local_json"},
        "stages": [],
    }
    atomic_write_json(work_dir / "resolved_config.json", {"config": config})
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")
    _mock_successful_release_checks(monkeypatch, release, gates)

    release_manifest, passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        skip_stage_validation=True,
    )

    assert passed
    assert release_manifest["status"] == "passed"
    assert not any("namespace" in error.lower() for error in release_manifest["errors"])


def test_canonical_release_requires_snapshot_and_stage_validation(tmp_path, monkeypatch):
    from pipeline.core import release

    work_dir = tmp_path / "runs" / "candidate-no-snapshot"
    _write_valid_release_artifacts(work_dir)
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")
    config = {
        "pipeline": {"production_profile": True},
        "embedder": {
            "pinecone_index": "idx",
            "pinecone_sparse_index": "idx-sparse",
            "enable_dense_facts": True,
        },
        "retrieval": {},
        "graph": {"store_backend": "local_json"},
        "stages": [],
    }
    monkeypatch.setattr(release, "load_effective_config", lambda *args, **kwargs: config)
    _mock_successful_release_checks(monkeypatch, release, gates)

    release_manifest, passed = release.build_release_manifest(
        config_name="mbzuai_production",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        skip_stage_validation=True,
    )

    assert not passed
    assert any("missing resolved_config.json" in error for error in release_manifest["errors"])
    assert any("cannot skip stage validation" in error for error in release_manifest["errors"])


def test_promotion_revalidates_release_manifest_integrity(tmp_path):
    from pipeline.core.release import promote_release_manifest

    manifest_path = tmp_path / "release.json"
    active_path = tmp_path / "active_release.json"
    valid = {
        "status": "passed",
        "errors": [],
        "preflight": {"ok": True},
        "audit": {"ok": True},
        "evaluation": {"query_count": 1, "gates": {"passed": True}},
        "answer_evaluation": {
            "query_count": 1,
            "gates": {"passed": True},
            "skipped": False,
            "waived": False,
            "waiver_reason": "",
        },
        "run_id": "candidate-1",
        "release_id": "release-1",
    }
    atomic_write_json(manifest_path, valid)
    assert promote_release_manifest(manifest_path=manifest_path, active_release_file=active_path) == active_path

    invalid = json.loads(json.dumps(valid))
    invalid["errors"] = ["hidden failure"]
    atomic_write_json(manifest_path, invalid)
    with pytest.raises(ValueError, match="errors must be an empty list"):
        promote_release_manifest(manifest_path=manifest_path, active_release_file=active_path)

    invalid = json.loads(json.dumps(valid))
    invalid["evaluation"]["gates"]["passed"] = False
    atomic_write_json(manifest_path, invalid)
    with pytest.raises(ValueError, match="retrieval evaluation gates"):
        promote_release_manifest(manifest_path=manifest_path, active_release_file=active_path)

    invalid = json.loads(json.dumps(valid))
    invalid["evaluation"]["query_count"] = 0
    atomic_write_json(manifest_path, invalid)
    with pytest.raises(ValueError, match="retrieval evaluation query_count"):
        promote_release_manifest(manifest_path=manifest_path, active_release_file=active_path)

    invalid = json.loads(json.dumps(valid))
    invalid["answer_evaluation"]["query_count"] = 0
    atomic_write_json(manifest_path, invalid)
    with pytest.raises(ValueError, match="answer evaluation query_count"):
        promote_release_manifest(manifest_path=manifest_path, active_release_file=active_path)

    invalid = json.loads(json.dumps(valid))
    invalid["status"] = "passed_with_waiver"
    invalid["answer_evaluation"].update({"skipped": True, "waived": True, "waiver_reason": ""})
    atomic_write_json(manifest_path, invalid)
    with pytest.raises(ValueError, match="non-empty waiver reason"):
        promote_release_manifest(manifest_path=manifest_path, active_release_file=active_path)


def test_release_check_rejects_modern_upload_manifest_that_differs_from_run_snapshot(tmp_path, monkeypatch):
    from pipeline.core import release

    work_dir = tmp_path / "runs" / "run_mismatch"
    _write_valid_release_artifacts(work_dir)
    atomic_write_json(
        work_dir / "resolved_config.json",
        {
            "config": {
                "embedder": {
                    "pinecone_index": "idx",
                    "pinecone_sparse_index": "missing-sparse-index",
                    "namespace_chunks": "chunks",
                    "namespace_parents": "parents",
                    "namespace_media": "media",
                    "namespace_facts": "facts",
                    "namespace_summaries": "summaries",
                    "namespace_assertions": "assertions",
                },
                "retrieval": {},
            }
        },
    )
    dataset = tmp_path / "eval.jsonl"
    gates = tmp_path / "gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")

    fake_audit = SimpleNamespace(
        ok=True,
        errors=[],
        warnings=[],
        to_dict=lambda: {"errors": [], "warnings": []},
    )
    monkeypatch.setattr(release, "audit_run", lambda path: fake_audit)
    monkeypatch.setattr(
        release,
        "assess_production_readiness",
        lambda *args, **kwargs: {"ok": True, "error_count": 0, "warning_count": 0, "checks": []},
    )
    monkeypatch.setattr(
        release,
        "validate_eval_examples",
        lambda *args, **kwargs: {"ok": True, "summary": {"query_count": 1}, "errors": [], "warnings": []},
    )
    monkeypatch.setattr(
        release,
        "evaluate_retrieval_dataset",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"chunk_hit_at_5": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )
    monkeypatch.setattr(
        release,
        "evaluate_answer_readiness",
        lambda **kwargs: {
            "query_count": 1,
            "overall": {"pass_rate": 1.0},
            "gates": {"path": str(gates), "passed": True, "failures": []},
        },
    )

    manifest, passed = release.build_release_manifest(
        config_name="default",
        work_dir=work_dir,
        dataset_path=dataset,
        gates_path=gates,
        skip_stage_validation=True,
    )

    assert not passed
    assert manifest["status"] == "failed"
    assert manifest["vector_index"]["index_name"] == "idx"
    assert manifest["vector_index"]["sparse_index_name"] == "idx-sparse"
    assert any("missing-sparse-index" in error for error in manifest["errors"])


def test_legacy_vector_upload_manifest_overrides_stale_retrieval_config(tmp_path):
    from pipeline.retrieval.adaptive_hybrid import apply_vector_upload_manifest_config

    work_dir = tmp_path / "runs" / "legacy_run"
    (work_dir / "stage_outputs" / "upload_legacy_vectorstores").mkdir(parents=True)
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_legacy_vectorstores" / "legacy_pinecone_upload_manifest.json",
        {
            "vectorstore_contract": "mbzuai_chatbot_legacy_v1",
            "summary_index_name": "summary-live",
            "text_index_name": "text-live",
            "namespace": "",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1024,
            "use_sparse_embeddings": True,
            "bm25_model_file": str(work_dir / "MBZUAI_BM25_ENCODER.json"),
        },
    )
    config = {
        "embedder": {
            "pinecone_index": "text-stale",
            "pinecone_text_index": "text-stale",
            "pinecone_summary_index": "summary-stale",
            "pinecone_sparse_index": "text-stale-sparse",
            "namespace": "mbzuai_main",
            "namespace_chunks": "mbzuai_main-chunks",
        },
        "retrieval": {"enable_sparse": True},
    }

    resolved = apply_vector_upload_manifest_config(config, work_dir)

    assert resolved["embedder"]["pinecone_index"] == "text-live"
    assert resolved["embedder"]["pinecone_summary_index"] == "summary-live"
    assert resolved["embedder"]["pinecone_sparse_index"] == ""
    assert resolved["embedder"]["namespace"] == ""
    assert resolved["embedder"]["namespace_chunks"] == ""
    assert resolved["retrieval"]["enable_sparse"] is False
    assert resolved["retrieval"]["legacy_hybrid_sparse_enabled"] is True


def test_modern_vector_upload_manifest_takes_precedence_over_legacy_manifest(tmp_path):
    from pipeline.retrieval.adaptive_hybrid import apply_vector_upload_manifest_config

    work_dir = tmp_path / "runs" / "modern_run"
    (work_dir / "stage_outputs" / "upload_retrieval").mkdir(parents=True)
    (work_dir / "stage_outputs" / "upload_legacy_vectorstores").mkdir(parents=True)
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        {
            "index_name": "dense-v3",
            "sparse_index_name": "sparse-v3",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1536,
            "namespaces": {
                "chunks": "chunks",
                "parents": "parents",
                "facts": "facts",
                "evidence_spans": "evidence_spans",
                "summaries": "summaries",
                "assertions": "assertions",
            },
        },
    )
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_legacy_vectorstores" / "legacy_pinecone_upload_manifest.json",
        {
            "vectorstore_contract": "mbzuai_chatbot_legacy_v1",
            "summary_index_name": "summary-legacy",
            "text_index_name": "text-legacy",
            "namespace": "",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1024,
            "use_sparse_embeddings": True,
        },
    )
    config = {
        "stages": [
            {"id": "crawl_web", "type": "crawler", "plugin": "crawl4ai"},
            {"id": "finalize_retrieval_bundle", "type": "formatter", "plugin": "gemini_retrieval"},
        ],
        "embedder": {
            "pinecone_index": "stale-dense",
            "pinecone_text_index": "text-stale",
            "pinecone_summary_index": "summary-stale",
            "pinecone_sparse_index": "stale-sparse",
            "output_dimensionality": 1024,
            "namespace_chunks": "stale-chunks",
        },
        "retrieval": {
            "enable_sparse": False,
            "legacy_vectorstore_contract": "mbzuai_chatbot_legacy_v1",
        },
    }

    resolved = apply_vector_upload_manifest_config(config, work_dir)

    assert resolved["embedder"]["pinecone_index"] == "dense-v3"
    assert resolved["embedder"]["pinecone_sparse_index"] == "sparse-v3"
    assert resolved["embedder"]["output_dimensionality"] == 1536
    assert resolved["embedder"]["namespace_chunks"] == "chunks"
    assert resolved["embedder"]["namespace_evidence_spans"] == "evidence_spans"
    assert "pinecone_text_index" not in resolved["embedder"]
    assert resolved["retrieval"]["enable_sparse"] is True
    assert resolved["retrieval"]["pinecone_index"] == "dense-v3"
    assert resolved["retrieval"]["namespace_evidence_spans"] == "evidence_spans"
    assert "legacy_vectorstore_contract" not in resolved["retrieval"]
    assert resolved["retrieval"]["legacy_hybrid_sparse_enabled"] is False
    assert resolved["stages"][-1] == {"id": "upload_retrieval", "type": "embedder", "plugin": "gemini_pinecone"}


def test_mbzuai_institution_token_is_not_required_for_own_site_abstention():
    from pipeline.retrieval.adaptive_hybrid import (
        _effective_named_tokens_for_abstention,
        _missing_token_ratio,
    )

    query = "What does MBZUAI's UGRIP program provide?"
    evidence_text = "source_url: https://mbzuai.ac.ae/ugrip UGRIP is a fully funded research internship."

    named_tokens = _effective_named_tokens_for_abstention(query, evidence_text)

    assert named_tokens == ["ugrip"]
    assert _missing_token_ratio(named_tokens, evidence_text) == 0.0


def test_retrieval_cache_fingerprint_includes_pinecone_contract_fields():
    from pipeline.evaluation.retrieval_eval import _config_fingerprint, _retrieval_cache_config_payload

    base_config = {
        "embedder": {
            "pinecone_index": "text-live",
            "pinecone_sparse_index": "",
            "namespace_chunks": "",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1024,
        },
        "retrieval": {"enable_sparse": False},
    }
    changed_config = {
        **base_config,
        "embedder": {
            **base_config["embedder"],
            "namespace_chunks": "stale-chunks",
        },
    }

    assert _config_fingerprint(_retrieval_cache_config_payload(base_config)) != _config_fingerprint(
        _retrieval_cache_config_payload(changed_config)
    )


def test_retrieval_cache_fingerprint_includes_serving_implementation(monkeypatch):
    from pipeline.evaluation import retrieval_eval

    config = {
        "embedder": {
            "model": "gemini-embedding-2",
            "output_dimensionality": 1536,
        },
        "retrieval": {"retriever_backend": "routed_hybrid"},
    }
    monkeypatch.setattr(
        retrieval_eval,
        "production_serving_contract_fingerprint",
        lambda _config: "serving-code-v1",
    )
    first = retrieval_eval._config_fingerprint(
        retrieval_eval._retrieval_cache_config_payload(config)
    )
    monkeypatch.setattr(
        retrieval_eval,
        "production_serving_contract_fingerprint",
        lambda _config: "serving-code-v2",
    )
    second = retrieval_eval._config_fingerprint(
        retrieval_eval._retrieval_cache_config_payload(config)
    )

    assert first != second


def test_retrieval_eval_expands_gold_ids_from_expected_source_urls(tmp_path):
    from pipeline.evaluation.dataset import EvalExample
    from pipeline.evaluation import retrieval_eval

    work_dir = tmp_path / "run"
    bundle_dir = work_dir / "stage_outputs" / "format_retrieval"
    bundle_dir.mkdir(parents=True)
    atomic_write_json(
        bundle_dir / "retrieval_bundle.json",
        {
            "chunk_records": [
                {"id": "chunk-phd-programs", "source_url": "https://mbzuai.ac.ae/study/phd-programs/"},
            ],
            "parent_records": [
                {"id": "parent-phd-programs", "source_url": "https://mbzuai.ac.ae/study/phd-programs"},
            ],
            "media_records": [],
        },
    )
    example = EvalExample(
        id="source-url-match",
        query="What PhD programs does MBZUAI offer?",
        query_type="scoped",
        source_type="webpage",
        gold_chunk_ids=["stale-generated-chunk"],
        gold_parent_ids=["stale-generated-parent"],
        metadata={"expected_reference_urls": ["https://mbzuai.ac.ae/study/phd-programs"]},
    ).normalized()

    ids_by_url = retrieval_eval._load_gold_ids_by_url(work_dir)
    score = retrieval_eval._score_query(
        example,
        {
            "selected_chunk_ids": ["chunk-phd-programs"],
            "selected_parent_ids": ["parent-phd-programs"],
        },
        ids_by_url=ids_by_url,
    )

    assert score.chunk_hit_at_5 == 1.0
    assert score.chunk_exact_hit_at_5 == 0.0
    assert score.chunk_source_hit_at_5 == 1.0
    assert score.chunk_recall_at_10 == 0.0
    assert score.chunk_exact_ndcg_at_10 == 0.0
    assert score.chunk_source_ndcg_at_10 == 0.65
    assert score.chunk_ndcg_at_10 == 0.65
    assert score.parent_hit_at_5 == 1.0


def test_retrieval_eval_source_aware_ndcg_caps_page_expansion_denominator(tmp_path):
    from pipeline.evaluation.dataset import EvalExample
    from pipeline.evaluation import retrieval_eval

    work_dir = tmp_path / "run"
    bundle_dir = work_dir / "stage_outputs" / "format_retrieval"
    bundle_dir.mkdir(parents=True)
    atomic_write_json(
        bundle_dir / "retrieval_bundle.json",
        {
            "chunk_records": [
                {"id": f"chunk-{index}", "source_url": "https://mbzuai.ac.ae/study/phd-programs"}
                for index in range(25)
            ],
            "parent_records": [],
            "media_records": [],
        },
    )
    example = EvalExample(
        id="source-aware-ndcg",
        query="Summarize MBZUAI PhD programs.",
        query_type="synthesis",
        source_type="webpage",
        gold_chunk_ids=["exact-generated-chunk"],
        metadata={"expected_reference_urls": ["https://mbzuai.ac.ae/study/phd-programs"]},
    ).normalized()

    ids_by_url = retrieval_eval._load_gold_ids_by_url(work_dir)
    score = retrieval_eval._score_query(
        example,
        {"selected_chunk_ids": ["chunk-19"]},
        ids_by_url=ids_by_url,
    )

    assert score.chunk_hit_at_5 == 1.0
    assert score.chunk_exact_hit_at_5 == 0.0
    assert score.chunk_source_hit_at_5 == 1.0
    assert score.chunk_recall_at_10 == 0.0
    assert score.chunk_source_ndcg_at_10 == 0.65


def test_retrieval_eval_expands_section_url_to_child_program_pages(tmp_path):
    from pipeline.evaluation.dataset import EvalExample
    from pipeline.evaluation import retrieval_eval

    work_dir = tmp_path / "run"
    bundle_dir = work_dir / "stage_outputs" / "format_retrieval"
    bundle_dir.mkdir(parents=True)
    atomic_write_json(
        bundle_dir / "retrieval_bundle.json",
        {
            "chunk_records": [
                {
                    "id": "chunk-phd-cv",
                    "source_url": "https://mbzuai.ac.ae/study/phd-programs/doctor-of-philosophy-in-computer-vision",
                }
            ],
            "parent_records": [],
            "media_records": [],
        },
    )
    example = EvalExample(
        id="section-source-url-match",
        query="What PhD programs does MBZUAI offer?",
        query_type="fact",
        source_type="webpage",
        metadata={"expected_reference_urls": ["https://mbzuai.ac.ae/study/phd-programs"]},
    ).normalized()

    ids_by_url = retrieval_eval._load_gold_ids_by_url(work_dir)
    score = retrieval_eval._score_query(
        example,
        {"selected_chunk_ids": ["chunk-phd-cv"]},
        ids_by_url=ids_by_url,
    )

    assert score.chunk_hit_at_5 == 1.0
    assert score.chunk_exact_hit_at_5 == 0.0
    assert score.chunk_source_hit_at_5 == 1.0


def test_retrieval_eval_normalizes_localized_mbzuai_reference_urls(tmp_path):
    from pipeline.evaluation.dataset import EvalExample
    from pipeline.evaluation import retrieval_eval

    work_dir = tmp_path / "run"
    bundle_dir = work_dir / "stage_outputs" / "format_retrieval"
    bundle_dir.mkdir(parents=True)
    atomic_write_json(
        bundle_dir / "retrieval_bundle.json",
        {
            "chunk_records": [
                {
                    "id": "chunk-president-ar",
                    "source_url": "https://mbzuai.ac.ae/ar/news/in-conversation-with-the-president-of-the-mohamed-bin-zayed-university-of-artificial-intelligence",
                }
            ],
            "parent_records": [
                {
                    "id": "parent-president-ar",
                    "source_url": "https://mbzuai.ac.ae/ar/news/in-conversation-with-the-president-of-the-mohamed-bin-zayed-university-of-artificial-intelligence",
                }
            ],
            "media_records": [],
        },
    )
    example = EvalExample(
        id="localized-source-url-match",
        query="Who is MBZUAI's president?",
        query_type="fact",
        source_type="webpage",
        metadata={
            "expected_reference_urls": [
                "https://mbzuai.ac.ae/news/in-conversation-with-the-president-of-the-mohamed-bin-zayed-university-of-artificial-intelligence"
            ]
        },
    ).normalized()

    ids_by_url = retrieval_eval._load_gold_ids_by_url(work_dir)
    score = retrieval_eval._score_query(
        example,
        {
            "selected_chunk_ids": ["chunk-president-ar"],
            "selected_parent_ids": ["parent-president-ar"],
        },
        ids_by_url=ids_by_url,
    )

    assert score.chunk_hit_at_5 == 1.0
    assert score.chunk_source_hit_at_5 == 1.0
    assert score.parent_hit_at_5 == 1.0


def test_retrieval_gate_checks_per_query_failures():
    from pipeline.evaluation.retrieval_eval import check_metric_gates

    report = {
        "overall": {},
        "by_query_type": {},
        "by_source_type": {},
        "by_benchmark_tag": {},
        "queries": [
            {
                "id": "good",
                "query": "good query",
                "no_answer": False,
                "has_gold_chunks": True,
                "chunk_exact_ndcg_at_10": 0.9,
            },
            {
                "id": "bad",
                "query": "bad query",
                "no_answer": False,
                "has_gold_chunks": True,
                "chunk_exact_ndcg_at_10": 0.0,
            },
        ],
    }
    gates = {
        "per_query": {
            "answerable": {
                "where": {"no_answer": False, "has_gold_chunks": True},
                "min_queries": 2,
                "metrics": {
                    "chunk_exact_ndcg_at_10": {
                        "min": 0.05,
                        "max_failures": 0,
                    }
                },
            }
        }
    }

    failures = check_metric_gates(report, gates)

    assert len(failures) == 1
    assert failures[0]["section"] == "per_query"
    assert failures[0]["failure_count"] == 1
    assert failures[0]["failing_queries"][0]["id"] == "bad"


def test_retrieval_gate_rejects_empty_gate_contract():
    from pipeline.evaluation.retrieval_eval import check_metric_gates

    failures = check_metric_gates({"overall": {}}, {})

    assert failures == [
        {
            "section": "gate_contract",
            "slice": "overall",
            "metric": None,
            "reason": "missing_gate_rules",
        }
    ]


def test_retrieval_gate_fails_metrics_with_no_eligible_queries_unless_explicitly_optional():
    from pipeline.evaluation.retrieval_eval import check_metric_gates

    report = {
        "overall": {
            "eligible_span_query_count": 0.0,
            "span_hit_at_10": 0.0,
            "span_mrr_at_10": 0.0,
            "eligible_multi_page_query_count": 0.0,
            "multi_page_coverage_rate": 0.0,
            "eligible_entity_coverage_query_count": 1.0,
            "required_entity_coverage": 0.5,
        },
        "by_query_type": {},
        "by_source_type": {},
        "by_benchmark_tag": {},
    }
    gates = {
        "overall": {
            "span_hit_at_10": {"min": 0.85},
            "span_mrr_at_10": {"min": 0.65},
            "multi_page_coverage_rate": {"min": 0.9},
            "required_entity_coverage": {"min": 0.9},
        }
    }

    failures = check_metric_gates(report, gates)

    assert [failure["metric"] for failure in failures] == [
        "span_hit_at_10",
        "span_mrr_at_10",
        "multi_page_coverage_rate",
        "required_entity_coverage",
    ]
    assert failures[0]["reason"] == "no_eligible_queries"
    assert failures[-1]["actual"] == 0.5

    optional_gates = {
        "overall": {
            "span_hit_at_10": {"min": 0.85, "allow_no_eligible": True},
            "required_entity_coverage": {"min": 0.9},
        }
    }
    optional_failures = check_metric_gates(report, optional_gates)
    assert [failure["metric"] for failure in optional_failures] == ["required_entity_coverage"]


def test_retrieval_eval_infers_english_faculty_profile_for_english_person_queries(tmp_path):
    from pipeline.evaluation.dataset import EvalExample
    from pipeline.evaluation import retrieval_eval

    work_dir = tmp_path / "run"
    bundle_dir = work_dir / "stage_outputs" / "format_retrieval"
    bundle_dir.mkdir(parents=True)
    atomic_write_json(
        bundle_dir / "retrieval_bundle.json",
        {
            "chunk_records": [
                {"id": "chunk-kentaro-inui", "source_url": "https://mbzuai.ac.ae/study/faculty/kentaro-inui"},
            ],
            "parent_records": [],
            "media_records": [],
        },
    )
    example = EvalExample(
        id="faculty-url-inference",
        query="What are Professor Kentaro Inui's research interests at MBZUAI?",
        query_type="fact",
        source_type="webpage",
        gold_chunk_ids=["arabic-generated-faculty-chunk"],
        metadata={
            "expected_source_hints": ["Kentaro Inui faculty page"],
            "expected_reference_urls": ["https://mbzuai.ac.ae/ar/study/faculty/kentaro-inui"],
        },
    ).normalized()

    ids_by_url = retrieval_eval._load_gold_ids_by_url(work_dir)
    expanded_gold_ids = retrieval_eval._gold_ids(example, "gold_chunk_ids", ids_by_url=ids_by_url)

    assert "chunk-kentaro-inui" in expanded_gold_ids


def test_degree_program_source_prior_prefers_program_page_over_faculty_profile():
    from pipeline.retrieval.adaptive_hybrid import _support_intent_text_bonus

    query = "What are the goals of the Master of Science in Computer Vision program and when is the application deadline?"
    program_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/msc-programs/master-of-science-in-computer-vision",
        text="Application deadline and program goals for the Master of Science in Computer Vision.",
    )
    faculty_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/faculty/salman-khan",
        text="Professor profile with computer vision research interests.",
    )

    assert program_score > faculty_score + 3.0


def test_person_statement_source_prior_prefers_news_interview_over_profile():
    from pipeline.retrieval.adaptive_hybrid import _support_intent_text_bonus

    query = "What skills does Professor Eric Xing believe are important for future generations in the age of AI?"
    news_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/news/in-conversation-with-the-president-of-the-mohamed-bin-zayed-university-of-artificial-intelligence",
        text="In conversation with the president Eric Xing discusses skills future generations need in the age of AI.",
    )
    profile_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/faculty/professor-eric-xing",
        text="Professor Eric Xing faculty profile, biography, awards, and publications.",
    )

    assert news_score > profile_score + 4.0


def test_person_statement_source_prior_penalizes_generic_named_news():
    from pipeline.retrieval.adaptive_hybrid import _support_intent_text_bonus

    query = "What skills does Professor Eric Xing believe are important for future generations in the age of AI?"
    conversation_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/news/in-conversation-with-the-president-of-the-mohamed-bin-zayed-university-of-artificial-intelligence",
        text="Professor Eric Xing says AI literacy and the ability to continuously learn are important skills for future generations in the age of AI.",
    )
    generic_news_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/news/aiq-executives-meet-with-mbzuai-president-professor-eric-xing",
        text="MBZUAI leadership met with AIQ executives. Professor Eric Xing represented the university during the visit.",
    )

    assert conversation_score > generic_news_score + 5.0


def test_student_life_source_prior_prefers_study_page_over_faculty_profile():
    from pipeline.retrieval.adaptive_hybrid import _support_intent_text_bonus

    query = "What is it like to live in Abu Dhabi as a student at MBZUAI?"
    study_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study",
        text="Student life in Abu Dhabi, culture, entertainment, and living at MBZUAI.",
    )
    faculty_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/faculty/nancy-gleason",
        text="Faculty profile and biography.",
    )

    assert study_score > faculty_score + 3.0


def test_undergraduate_start_date_prior_prefers_overview_over_stream_curriculum():
    from pipeline.retrieval.adaptive_hybrid import _support_intent_text_bonus

    query = "When does the Bachelor of Science in Artificial Intelligence Business Stream program officially start classes?"
    overview_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/undergraduate-program",
        text="When does the 2026 Fall semester start? The 2026 Fall semester begins in mid-August 2026.",
    )
    stream_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/ar/bachelor-of-science-in-artificial-intelligence-business-stream",
        text="Arabic curriculum page with course descriptions and program learning outcomes.",
    )

    assert overview_score > stream_score


def test_scholarship_prior_prefers_scholarship_source_over_undergraduate_stream():
    from pipeline.retrieval.adaptive_hybrid import _support_intent_text_bonus

    query = "Does MBZUAI offer scholarships for undergraduate students in the Engineering Stream?"
    scholarship_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/the-sheikh-tahnoon-bin-zayed-scholarship-in-ai-excellence-at-mbzuai",
        text="Tahnoon bin Zayed Scholarship in AI Excellence covers tuition and financial aid.",
    )
    stream_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/undergraduate-program/bachelor-of-science-in-artificial-intelligence-engineering-stream",
        text="Engineering stream curriculum, courses, and program learning outcomes.",
    )

    assert scholarship_score > stream_score + 2.0


def test_news_event_prior_prefers_advisory_board_meeting_article():
    from pipeline.retrieval.adaptive_hybrid import _support_intent_text_bonus

    query = "When did MBZUAI conduct its first official Advisory Board meeting, and who attended?"
    meeting_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/news/mbzuai-holds-first-advisory-board-meeting",
        text="MBZUAI held its first official Advisory Board meeting and listed notable attendees.",
    )
    faculty_score = _support_intent_text_bonus(
        query,
        source_url="https://mbzuai.ac.ae/study/faculty/ian-reid",
        text="Faculty profile with awards and advisory board memberships.",
    )

    assert meeting_score > faculty_score + 3.0


def test_selected_chunk_parent_promotion_keeps_strong_citation_parent():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.parent_candidate_top_k = 3
    retriever.max_context_chunks = 8
    retriever.parent_map = {
        "gold-section": {"id": "gold-section"},
        "gold-page": {"id": "gold-page"},
        "noise-page": {"id": "noise-page"},
    }
    retriever.chunk_map = {
        "noise": {
            "id": "noise",
            "section_key": "",
            "page_key": "noise-page",
            "source_url": "https://mbzuai.ac.ae/news/generic-president-story",
            "text": "Professor Eric Xing attended a meeting.",
        },
        "gold": {
            "id": "gold",
            "section_key": "gold-section",
            "page_key": "gold-page",
            "source_url": "https://mbzuai.ac.ae/news/in-conversation-with-the-president-of-the-mohamed-bin-zayed-university-of-artificial-intelligence",
            "text": "AI literacy and the ability to continuously learn are important skills for future generations in the age of AI.",
        },
    }
    retriever._source_query_bonus = lambda query, **kwargs: 5.0 if "in-conversation" in kwargs.get("source_url", "") else 0.5
    retriever._score_text_match = lambda query, text: 1.0 if "AI literacy" in text else 0.1

    selected = retriever._promote_selected_chunk_parent_ids(
        "What skills does Professor Eric Xing believe are important for future generations in the age of AI?",
        ["noise-page"],
        ["noise", "gold"],
    )

    assert selected[:2] == ["gold-section", "gold-page"]


def test_selected_chunk_parent_promotion_anchors_top_evidence_chunk_parent():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.parent_candidate_top_k = 3
    retriever.max_context_chunks = 8
    retriever.parent_map = {
        "top-section": {"id": "top-section"},
        "top-page": {"id": "top-page"},
        "explicit-noise": {"id": "explicit-noise"},
    }
    retriever.chunk_map = {
        "top-chunk": {
            "id": "top-chunk",
            "section_key": "top-section",
            "page_key": "top-page",
            "source_url": "https://mbzuai.ac.ae/about/faq",
            "text": "A short exact fact selected for answer evidence.",
        },
    }
    retriever._source_query_bonus = lambda query, **kwargs: 0.2
    retriever._score_text_match = lambda query, text: 0.1

    selected = retriever._promote_selected_chunk_parent_ids(
        "What official fact is stated?",
        ["explicit-noise"],
        ["top-chunk"],
    )

    assert selected[:2] == ["top-section", "top-page"]


def test_selected_chunk_parent_promotion_prefers_current_catalogue_for_legal_scope():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.parent_candidate_top_k = 3
    retriever.max_context_chunks = 8
    retriever.parent_map = {
        "factsheet-page": {"id": "factsheet-page"},
        "old-catalogue-page": {"id": "old-catalogue-page"},
        "current-catalogue-section": {"id": "current-catalogue-section"},
        "current-catalogue-page": {"id": "current-catalogue-page"},
    }
    retriever.chunk_map = {
        "factsheet": {
            "id": "factsheet",
            "section_key": "",
            "page_key": "factsheet-page",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI-Factsheet-Eng-Nov-2025.docx",
            "document_title": "MBZUAI Factsheet",
            "text": "MBZUAI was established in 2019 as an artificial intelligence university.",
        },
        "old-catalogue": {
            "id": "old-catalogue",
            "section_key": "",
            "page_key": "old-catalogue-page",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI_University_Catalogue_2021-22.pdf",
            "document_title": "University Catalogue 2021-22",
            "text": "The University was established under Law No. 2 of 2019.",
        },
        "current-catalogue": {
            "id": "current-catalogue",
            "section_key": "current-catalogue-section",
            "page_key": "current-catalogue-page",
            "source_url": "https://staticcdn.mbzuai.ac.ae/2025/05/University-Catalogue-2024-2025.pdf",
            "document_title": "University Catalogue 2024-2025",
            "text": "The University was established under Law No. 2 of 2019 and shall be affiliated with the Abu Dhabi Executive Council.",
        },
    }
    retriever._score_text_match = lambda query, text: 0.8 if "Executive Council" in text else 0.25

    selected = retriever._promote_selected_chunk_parent_ids(
        "Under which law was MBZUAI established and to which entity is it affiliated?",
        ["factsheet-page", "old-catalogue-page"],
        ["factsheet", "old-catalogue", "current-catalogue"],
    )

    assert selected[:2] == ["current-catalogue-section", "current-catalogue-page"]


def test_parent_promotion_diversifies_legal_scope_when_one_source_dominates():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.parent_candidate_top_k = 3
    retriever.max_context_chunks = 8
    retriever.chunk_map = {}
    retriever.parent_map = {
        "factsheet-section": {
            "id": "factsheet-section",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI-Factsheet-Eng-Nov-2025.docx",
        },
        "factsheet-page": {
            "id": "factsheet-page",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI-Factsheet-Eng-Nov-2025.docx",
        },
        "old-section": {
            "id": "old-section",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI_University_Catalogue_2021-22.pdf",
        },
        "old-page": {
            "id": "old-page",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI_University_Catalogue_2021-22.pdf",
        },
        "old-overview": {
            "id": "old-overview",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI_University_Catalogue_2021-22.pdf",
        },
        "old-message": {
            "id": "old-message",
            "source_url": "https://staticcdn.mbzuai.ac.ae/MBZUAI_University_Catalogue_2021-22.pdf",
        },
        "current-section": {
            "id": "current-section",
            "source_url": "https://staticcdn.mbzuai.ac.ae/2025/05/University-Catalogue-2024-2025.pdf",
        },
        "current-page": {
            "id": "current-page",
            "source_url": "https://staticcdn.mbzuai.ac.ae/2025/05/University-Catalogue-2024-2025.pdf",
        },
    }

    selected = retriever._promote_selected_chunk_parent_ids(
        "Summarize the law that created MBZUAI and the authority it is affiliated with.",
        [
            "factsheet-section",
            "factsheet-page",
            "old-section",
            "old-page",
            "old-overview",
            "old-message",
            "current-section",
            "current-page",
        ],
        [],
    )

    assert "current-section" in selected[:5]
    assert selected[:5] == [
        "factsheet-section",
        "factsheet-page",
        "old-section",
        "old-page",
        "current-section",
    ]


def test_retrieval_formatter_emits_summary_records(tmp_path):
    from pipeline.stages.formatters.retrieval_bundle_v2_formatter import RetrievalBundleV2Formatter

    md_path = tmp_path / "page.md"
    md_path.write_text("# Admissions\nApplications include eligibility details. Funding and campus information are provided.", encoding="utf-8")
    chunks_file = tmp_path / "chunks.json"
    atomic_write_json(
        chunks_file,
        build_chunk_index(
            [
                {
                    "text": "Applications include eligibility details. Funding and campus information are provided.",
                    "document_id": "doc1",
                    "document_title": "Admissions",
                    "document_type": "webpage",
                    "source_markdown_path": str(md_path),
                    "source_url": "https://mbzuai.ac.ae/study",
                    "section_path": ["Study", "Admissions"],
                    "heading": "Admissions",
                }
            ],
            strategy="test",
        ),
    )
    ctx = StageContext(
        run_id="run",
        project_name="proj",
        config={"formatter": {}, "storage": {"plugin": "local", "copy_on_register": False}},
        work_dir=tmp_path,
        previous_outputs={"chunks_file": str(chunks_file)},
        stage_definition={"type": "formatter", "plugin": "retrieval_bundle_v2"},
        stage_id="format_retrieval",
    )

    result = run_async(RetrievalBundleV2Formatter().execute(ctx))
    assert result.status == StageStatus.COMPLETED
    assert "summary_embedding_file" in result.outputs
    summaries = load_json_safe(result.outputs["summary_embedding_file"], [])
    bundle = load_json_safe(result.outputs["retrieval_bundle_file"], {})
    lexical = load_json_safe(result.outputs["lexical_corpus_file"], [])
    assert summaries
    assert bundle["stats"]["summary_count"] == len(summaries)
    assert any(record["record_type"] == "summary" for record in lexical)


def test_assertion_promotion_marks_superseded_conflicts(tmp_path):
    from pipeline.stages.formatters.assertion_promote_formatter import AssertionPromoteFormatter

    entities_file = tmp_path / "entities.json"
    assertions_file = tmp_path / "assertions.json"
    atomic_write_json(
        entities_file,
        [
            {"id": "entity:mbzuai", "canonical_name": "MBZUAI"},
            {"id": "entity:alice", "canonical_name": "Alice"},
            {"id": "entity:bob", "canonical_name": "Bob"},
        ],
    )
    base = {
        "subject_entity_id": "entity:mbzuai",
        "subject_name": "MBZUAI",
        "predicate": "role_holder",
        "answer_type": "role_holder",
        "answer_subtype": "president",
        "validator_decision": "supported",
        "confidence": 0.9,
        "freshness_score": 0.9,
        "source_chunk_ids": ["c1"],
    }
    atomic_write_json(
        assertions_file,
        [
            {**base, "id": "assertion:1", "object_value": "Alice", "object_entity_id": "entity:alice", "authority_score": 0.95},
            {**base, "id": "assertion:2", "object_value": "Bob", "object_entity_id": "entity:bob", "authority_score": 0.55},
        ],
    )
    ctx = StageContext(
        run_id="run",
        project_name="proj",
        config={"assertions": {"promote_min_confidence": 0.5, "promote_min_authority_score": 0.4}},
        work_dir=tmp_path,
        previous_outputs={"canonical_entities_file": str(entities_file), "canonical_assertions_file": str(assertions_file)},
        stage_definition={"type": "formatter", "plugin": "assertion_promote"},
        stage_id="promote_assertions",
    )

    result = run_async(AssertionPromoteFormatter().execute(ctx))
    assert result.status == StageStatus.COMPLETED
    promoted = load_json_safe(result.outputs["promoted_assertions_file"], [])
    conflicts = load_json_safe(result.outputs["assertion_conflicts_file"], [])
    statuses = {record["id"]: record["validity_status"] for record in promoted}
    assert statuses["assertion:1"] == "active"
    assert statuses["assertion:2"] == "superseded"
    assert conflicts and conflicts[0]["selected_assertion_id"] == "assertion:1"


def test_graph_promotion_and_retriever_exclude_superseded_assertions(tmp_path):
    from pipeline.retrieval.graph_rag import GraphRAGRetriever
    from pipeline.stages.formatters.semantic_graph_promote_formatter import SemanticGraphPromoteFormatter

    base_graph_file = tmp_path / "knowledge_graph.json"
    atomic_write_json(
        base_graph_file,
        {
            "schema_version": 1,
            "graph_type": "deterministic_content_graph",
            "nodes": [
                {"id": "chunk1", "node_type": "chunk", "label": "Chunk"},
                {"id": "fact1", "node_type": "fact", "label": "Fact"},
            ],
            "edges": [
                {"id": "edge-1", "edge_type": "CHUNK_HAS_FACT", "source_id": "chunk1", "target_id": "fact1"},
            ],
            "stats": {
                "node_count": 2,
                "edge_count": 1,
                "node_type_counts": {"chunk": 1, "fact": 1},
                "edge_type_counts": {"CHUNK_HAS_FACT": 1},
            },
        },
    )
    entities_file = tmp_path / "entities.json"
    assertions_file = tmp_path / "assertions.json"
    atomic_write_json(
        entities_file,
        [
            {"id": "entity:mbzuai", "canonical_name": "MBZUAI", "entity_type": "organization"},
            {"id": "entity:alice", "canonical_name": "Alice", "entity_type": "person"},
            {"id": "entity:bob", "canonical_name": "Bob", "entity_type": "person"},
        ],
    )
    base_assertion = {
        "relation_type": "ROLE_HOLDER",
        "subject_entity_id": "entity:mbzuai",
        "subject_name": "MBZUAI",
        "source_chunk_ids": ["chunk1"],
        "source_fact_ids": ["fact1"],
        "source_parent_ids": ["page1"],
        "confidence": 0.9,
    }
    atomic_write_json(
        assertions_file,
        [
            {
                **base_assertion,
                "id": "assertion:current",
                "object_entity_id": "entity:alice",
                "object_name": "Alice",
                "validity_status": "active",
            },
            {
                **base_assertion,
                "id": "assertion:old",
                "object_entity_id": "entity:bob",
                "object_name": "Bob",
                "validity_status": "superseded",
                "superseded_by": "assertion:current",
            },
        ],
    )
    ctx = StageContext(
        run_id="run",
        project_name="proj",
        config={},
        work_dir=tmp_path,
        previous_outputs={
            "knowledge_graph_file": str(base_graph_file),
            "semantic_entities_file": str(entities_file),
            "semantic_assertions_file": str(assertions_file),
        },
        stage_definition={"type": "formatter", "plugin": "semantic_graph_promote"},
        stage_id="promote_graph",
    )

    result = run_async(SemanticGraphPromoteFormatter().execute(ctx))
    assert result.status == StageStatus.COMPLETED
    promoted = load_json_safe(result.outputs["knowledge_graph_file"], {})
    assertion_ids = {
        node["id"]
        for node in promoted["nodes"]
        if node.get("node_type") == "relation_assertion"
    }
    assert assertion_ids == {"assertion:current"}

    retriever = GraphRAGRetriever(
        config={"retrieval": {"graph_query_backend": "local"}},
        work_dir=tmp_path,
        base_retriever=SimpleNamespace(model="test-model", output_dimensionality=3),
    )
    assert set(retriever.assertion_map) == {"assertion:current"}


def test_selective_adjudication_skips_high_confidence_single_answer():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.selective_adjudication_enabled = True

    assert not retriever._should_run_evidence_adjudication(
        {
            "retrieval_confidence": 0.82,
            "answer_documents": [{"id": "a1", "text": "The answer is supported."}],
        }
    )
    assert retriever._should_run_evidence_adjudication(
        {
            "retrieval_confidence": 0.31,
            "answer_documents": [{"id": "a1", "text": "The answer is weak."}],
        }
    )
    assert not retriever._should_run_evidence_adjudication(
        {
            "retrieval_confidence": 0.72,
            "answer_documents": [],
            "fact_documents": [{"id": "fact-1", "text": "A supported fact."}],
        }
    )
    assert retriever._should_run_evidence_adjudication(
        {
            "retrieval_confidence": 0.60,
            "answer_documents": [],
            "fact_documents": [{"id": "fact-1", "text": "A weak fact."}],
        }
    )


def test_verified_media_evidence_skips_heuristic_required_page_inference():
    from pipeline.retrieval.adaptive_hybrid import QueryMode
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever._coverage_intent = lambda _query, _mode: "scoped"
    retriever._infer_coverage_requirements = lambda *_args: (_ for _ in ()).throw(
        AssertionError("verified media must not infer unrelated text pages")
    )

    plan = retriever._coverage_plan_for_result(
        query="What does the image on page 10 say?",
        payload={
            "media_evidence_verified": True,
            "selected_media_ids": ["media-page-10"],
            "selected_chunk_ids": ["chunk-page-10"],
        },
        mode=QueryMode.SCOPED,
    )

    assert plan["required_pages"] == []
    assert plan["required_pages_source"] == "verified_media_evidence"
    assert plan["coverage_status"] == "complete"


def test_verified_media_evidence_preserves_explicit_named_page_requirements():
    from pipeline.retrieval.adaptive_hybrid import QueryMode
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    page_url = "https://library.mbzuai.ac.ae/newest-technology"
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever._coverage_intent = lambda _query, _mode: "scoped"
    retriever._explicit_required_page_markers = lambda _query: ["/newest-technology"]
    retriever._infer_coverage_requirements = lambda _query, _intent: {
        "required_entities": [],
        "required_pages": [page_url],
        "required_sections": [],
        "required_pages_source": "explicit_markers",
    }
    retriever._selected_source_urls = lambda _payload: {
        retriever._normalize_source_url(page_url)
    }

    plan = retriever._coverage_plan_for_result(
        query="On the News on AI and Technology page, what items are shown?",
        payload={
            "media_evidence_verified": True,
            "selected_media_ids": ["unrelated-media"],
            "selected_chunk_ids": ["named-page-chunk"],
        },
        mode=QueryMode.SCOPED,
    )

    assert plan["required_pages"] == [page_url]
    assert plan["required_pages_source"] == "explicit_markers"
    assert plan["coverage_status"] == "complete"


def test_lookup_profile_preserves_contact_types_for_hours_plus_contact_query():
    from pipeline.retrieval.adaptive_hybrid import _lookup_query_profile

    profile = _lookup_query_profile("What are the operating hours for the MBZUAI Library, and how can I contact them?")

    assert "hours" in profile.answer_types
    assert {"email", "phone", "website"} <= set(profile.answer_types)
    assert "library" in profile.focus_tokens


def test_missing_source_url_is_recovered_from_official_url_in_text():
    from pipeline.retrieval.adaptive_hybrid import _ensure_record_source_url, _record_source_url
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    record = {
        "id": "ugrip-pdf-chunk",
        "text": "Visit the UGRIP page (https://mbzuai.ac.ae/ugrip/) to know more about application dates.",
        "source_url": "",
        "document_title": "MBZUAI Application Instructions UGRIP",
    }

    assert _record_source_url(record) == "https://mbzuai.ac.ae/ugrip/"
    _ensure_record_source_url(record)
    assert record["source_url"] == "https://mbzuai.ac.ae/ugrip/"

    pack = build_evidence_pack(
        query="UGRIP application dates",
        result={"retrieval_documents": [{**record, "source_url": ""}]},
    )

    assert pack["items"][0]["source_url"] == "https://mbzuai.ac.ae/ugrip/"
    assert pack["citation_candidates"][0]["source_url"] == "https://mbzuai.ac.ae/ugrip/"


def test_evidence_span_selection_filters_unrelated_pages_when_query_has_anchor():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.parent_candidate_top_k = 3
    retriever.evidence_span_map = {
        "ugrip-dates": {
            "id": "ugrip-dates",
            "text": "UGRIP admission cycle and important application dates.",
            "source_url": "https://mbzuai.ac.ae/ugrip",
        },
        "hackathon-dates": {
            "id": "hackathon-dates",
            "text": "Important dates for the MBZUAI x Johnson and Johnson Hackathon.",
            "source_url": "https://mbzuai.ac.ae/mbzuai-x-johnson-n-johnson-hackathon-2026",
        },
    }

    selected = retriever._select_evidence_span_ids_for_query(
        "Give me details about MBZUAI UGRIP and the important application dates.",
        ["ugrip-dates", "hackathon-dates"],
        mode=QueryMode.SYNTHESIS,
    )

    assert selected == ["ugrip-dates"]


def test_source_bonus_prefers_exact_program_slug_and_library_details():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    program_query = (
        "For the Master of Science in Machine Learning program, what are the requirements "
        "for nominating referees and what topics are covered in the online screening exam?"
    )

    exact_program_score = retriever._source_query_bonus(
        program_query,
        source_url="https://mbzuai.ac.ae/study/msc-programs/master-of-science-in-machine-learning",
        text="Applicants nominate referees and complete an online screening exam.",
    )
    wrong_program_score = retriever._source_query_bonus(
        program_query,
        source_url="https://mbzuai.ac.ae/study/msc-programs/master-of-science-in-computational-biology",
        text="Applicants nominate referees and complete an online screening exam.",
    )
    assert exact_program_score > wrong_program_score + 0.6

    library_score = retriever._source_query_bonus(
        "What are the operating hours for the MBZUAI Library, and how can I contact them?",
        source_url="https://mbzuai.ac.ae/student-resources/campus-facilities",
        text="Library hours of operation and library contact details are listed here.",
    )
    sitemap_score = retriever._source_query_bonus(
        "What are the operating hours for the MBZUAI Library, and how can I contact them?",
        source_url="https://mbzuai.ac.ae/sitemap",
        text="Sitemap link to Library.",
    )
    assert library_score > sitemap_score + 1.0


def test_synthesis_expansion_preserves_ranked_seed_chunks_before_parent_fill():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_context_chunks = 4
    retriever.max_parent_chunks = 4
    retriever.same_parent_expand_threshold = 2
    retriever.chunk_ids_by_section = {"p1": ["c1", "c2", "c3", "c4", "c5"]}
    retriever.chunk_ids_by_page = {}
    retriever.parent_map = {"p1": {"id": "p1", "parent_type": "section", "child_chunk_ids": ["c1", "c2", "c3", "c4", "c5"]}}
    retriever.chunk_map = {
        "c1": {"id": "c1", "section_key": "p1"},
        "c2": {"id": "c2", "section_key": "p1"},
        "c3": {"id": "c3", "section_key": "p1"},
        "gold": {"id": "gold", "section_key": "p1"},
    }

    selected = retriever._expand_scoped_or_synthesis(
        ["c1", "c2", "c3", "gold"],
        mode=QueryMode.SYNTHESIS,
        explicit_parent_ids=["p1"],
    )

    assert selected == ["c1", "c2", "c3", "gold"]


def test_scoped_expansion_adds_query_relevant_sibling_from_matched_parent():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever, QueryMode

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_context_chunks = 6
    retriever.max_parent_chunks = 4
    retriever.same_parent_expand_threshold = 1
    retriever.parent_candidate_top_k = 3
    retriever.chunk_ids_by_section = {"program-section": ["overview", "course", "screening", "deadline"]}
    retriever.chunk_ids_by_page = {}
    retriever.parent_map = {
        "program-section": {
            "id": "program-section",
            "parent_type": "section",
            "child_chunk_ids": ["overview", "course", "screening", "deadline"],
        }
    }
    retriever.chunk_map = {
        "overview": {"id": "overview", "section_key": "program-section", "text": "Program overview and goals."},
        "course": {"id": "course", "section_key": "program-section", "text": "Course description for an elective."},
        "screening": {"id": "screening", "section_key": "program-section", "text": "Online screening exam topics include math, programming, and machine learning."},
        "deadline": {"id": "deadline", "section_key": "program-section", "text": "Application deadline is listed here."},
    }

    selected = retriever._expand_scoped_or_synthesis(
        ["course"],
        query="What topics are covered in the online screening exam?",
        mode=QueryMode.SCOPED,
    )

    assert selected[0] == "course"
    assert "screening" in selected[:4]


def test_routed_required_page_backfill_adds_specific_parent_housing_span():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    undergrad_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    retriever.vector = SimpleNamespace(
        evidence_span_map={
            "generic-accommodation": {
                "id": "generic-accommodation",
                "text": "Student accommodation includes a multi-occupancy room and shared bathroom facilities.",
                "source_url": undergrad_url,
                "linked_chunk_ids": ["chunk-accommodation"],
            },
            "parents-housing": {
                "id": "parents-housing",
                "text": "Can my parents stay with me on campus? No, MBZUAI does not provide housing for parents.",
                "source_url": undergrad_url,
                "linked_chunk_ids": ["chunk-parents"],
                "linked_parent_ids": ["parent-parents"],
            },
        },
        _score_text_match=lambda query, text: 0.1,
    )
    payload = {
        "evidence_span_documents": [
            {"id": "generic-accommodation", "text": "Student accommodation is available.", "source_url": undergrad_url}
        ],
        "selected_evidence_span_ids": ["generic-accommodation"],
        "retrieval_documents": [],
    }

    changed = retriever._augment_payload_for_required_coverage(
        query="A student's family is visiting MBZUAI. Explain family accommodation and guest parking.",
        payload=payload,
        coverage_plan={"required_pages": [undergrad_url]},
    )

    assert changed
    assert "parents-housing" in payload["selected_evidence_span_ids"]
    assert payload["selected_chunk_ids"][0] == "chunk-parents"
    assert payload["selected_parent_ids"][0] == "parent-parents"


def test_routed_prioritizes_required_current_sources_over_old_catalog_spans():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    map_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/11/MBZUAI_Campus_Map_V1044331768.pdf"
    facilities_url = "https://mbzuai.ac.ae/student-resources/campus-facilities"
    old_catalog_url = "https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2022/08/MBZUAI_University_Catalog_2022_V9-.pdf"
    payload = {
        "evidence_span_documents": [
            {
                "id": "old-catalog",
                "text": "Old catalogue campus facilities and support services.",
                "source_url": old_catalog_url,
            },
            {
                "id": "current-map",
                "text": "Campus map labels Knowledge Center, Library, Medical Center, Gym, Swimming pool, and Canteen.",
                "source_url": map_url,
                "linked_chunk_ids": ["map-chunk"],
                "linked_parent_ids": ["map-parent"],
            },
            {
                "id": "facilities",
                "text": "Campus Facilities include student accommodation, laboratories, library, knowledge center, canteen, gyms, and pool.",
                "source_url": facilities_url,
                "linked_chunk_ids": ["facilities-chunk"],
                "linked_parent_ids": ["facilities-parent"],
            },
        ],
        "selected_evidence_span_ids": ["old-catalog", "current-map", "facilities"],
        "selected_chunk_ids": ["old-chunk"],
        "selected_parent_ids": ["old-parent"],
        "retrieval_documents": [],
    }

    retriever._prioritize_required_page_evidence(
        query="Give a detailed answer about MBZUAI campus facilities using both the campus facilities page and campus map.",
        payload=payload,
        coverage_plan={"required_pages": [map_url, facilities_url]},
    )

    assert payload["selected_evidence_span_ids"][:2] == ["current-map", "facilities"]
    assert payload["selected_evidence_span_ids"][-1] == "old-catalog"
    assert payload["selected_chunk_ids"][:2] == ["map-chunk", "facilities-chunk"]
    assert payload["selected_parent_ids"][:2] == ["map-parent", "facilities-parent"]


def test_routed_coverage_plan_includes_contact_page_for_arrival_transport_queries():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    contact_url = "https://mbzuai.ac.ae/about/contact"
    undergrad_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    retriever._coverage_page_records = [
        {
            "source_url": contact_url,
            "normalized_url": retriever._normalize_source_url(contact_url),
            "search_text": "visitor parking north car park golf cart navya bus prt transport",
            "tokens": set(_tokenize("visitor parking north car park golf cart navya bus prt transport")),
        },
        {
            "source_url": undergrad_url,
            "normalized_url": retriever._normalize_source_url(undergrad_url),
            "search_text": "student accommodation amenities get to campus",
            "tokens": set(_tokenize("student accommodation amenities get to campus")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query="Give a practical guide for a new graduate student arriving at MBZUAI, covering parking and shuttle transport.",
        payload={"retrieval_documents": []},
        mode=QueryMode.SYNTHESIS,
    )

    assert contact_url in plan["required_pages"]


def test_routed_coverage_plan_routes_generic_shuttle_query_to_contact_evidence():
    from pipeline.retrieval.adaptive_hybrid import QueryMode, _tokenize
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.unsupported_intent_guard_enabled = False
    contact_url = "https://mbzuai.ac.ae/about/contact"
    undergrad_url = "https://mbzuai.ac.ae/study/undergraduate-application-submission"
    retriever._coverage_page_records = [
        {
            "source_url": contact_url,
            "normalized_url": retriever._normalize_source_url(contact_url),
            "search_text": "contact visitor parking north car park golf cart navya bus prt",
            "tokens": set(_tokenize("contact visitor parking north car park golf cart navya bus prt")),
        },
        {
            "source_url": undergrad_url,
            "normalized_url": retriever._normalize_source_url(undergrad_url),
            "search_text": "undergraduate accommodation student services",
            "tokens": set(_tokenize("undergraduate accommodation student services")),
        },
    ]

    plan = retriever._coverage_plan_for_result(
        query="Does MBZUAI provide a shuttle bus service?",
        payload={"retrieval_documents": []},
        mode=QueryMode.FACT,
    )

    assert contact_url in plan["required_pages"]
    assert undergrad_url not in plan["required_pages"]


def test_dense_recall_floor_keeps_leading_dense_chunks_inside_top_ten():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_context_chunks = 12
    retriever.dense_recall_floor_k = 2
    retriever.dense_recall_window = 10
    retriever.chunk_map = {
        value: {"id": value}
        for value in [
            *[f"expanded-{index}" for index in range(12)],
            "dense-1",
            "dense-2",
        ]
    }

    selected = retriever._preserve_dense_chunk_recall(
        [f"expanded-{index}" for index in range(12)],
        ["dense-1", "dense-2", "dense-3"],
    )

    assert selected[:8] == [f"expanded-{index}" for index in range(8)]
    assert selected[8:10] == ["dense-1", "dense-2"]
    assert len(selected) == 12


def test_dense_parent_recall_floor_keeps_dense_page_parent_inside_top_five():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.dense_parent_recall_floor_k = 1
    retriever.dense_parent_recall_window = 5
    retriever.chunk_map = {"dense": {"id": "dense"}}
    retriever.parent_map = {
        **{f"parent-{index}": {"id": f"parent-{index}"} for index in range(6)},
        "dense-page": {"id": "dense-page", "parent_type": "page"},
    }
    retriever.parent_ids_by_chunk = {"dense": ["dense-page"]}
    retriever.section_parent_ids_by_chunk = {"dense": []}
    retriever.page_parent_ids_by_chunk = {"dense": ["dense-page"]}

    selected = retriever._preserve_dense_parent_recall(
        [f"parent-{index}" for index in range(6)],
        ["dense"],
    )

    assert selected[4] == "dense-page"
    assert len(selected) == 6


def test_explicit_dense_media_hit_is_not_discarded_by_chunk_attachment():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.max_media_results = 4
    retriever.chunk_map = {
        "selected": {"id": "selected", "media_ids": ["linked"]}
    }
    retriever.parent_map = {}
    retriever.parent_ids_by_chunk = {"selected": []}
    retriever.section_parent_ids_by_chunk = {"selected": []}
    retriever.page_parent_ids_by_chunk = {"selected": []}
    retriever.media_map = {
        "dense-gold": {
            "id": "dense-gold",
            "media_type": "image",
            "linked_chunk_ids": ["other"],
        },
        "linked": {
            "id": "linked",
            "media_type": "image",
            "linked_chunk_ids": ["selected"],
        },
    }
    retriever._is_low_signal_media = lambda media: False
    retriever._score_media_relevance = lambda query, media: (
        1.0 if media["id"] == "dense-gold" else 0.9
    )

    selected = retriever._attach_media(
        ["selected"],
        ["dense-gold"],
        "What does the image show?",
    )

    assert [item["id"] for item in selected][:2] == ["dense-gold", "linked"]


def test_strong_official_media_can_rescue_weak_surrounding_text():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.rrf_k = 60
    retriever.max_media_results = 4
    retriever.media_abstain_rescue_min_score = 1.25
    retriever.media_abstain_rescue_min_overlap = 0.35
    query = (
        "في الصفحة 10 من برنامج دفعة 2024 العربي، ما الهدف المذكور "
        "في النص على الصورة؟ image figure visual"
    )
    media_text = (
        "IMAGE: Visual from Class of 2024 Arabic page 10 "
        "VISIBLE_TEXT: الإسهام في تطوير تقنيات الجيل التالي التي ستدعم "
        "جهود الارتقاء ببنى النقل التحتية إلى مستويات أفضل هو هدفي"
    )
    retriever.media_map = {
        "gold-media": {
            "id": "gold-media",
            "media_type": "image",
            "source_url": "https://staticcdn.mbzuai.ac.ae/class-of-2024-arabic.pdf",
            "text": media_text,
        }
    }
    retriever.media_texts_by_id = {"gold-media": media_text}

    assert retriever._has_grounded_media_candidates(
        query=query,
        media_rankings=(
            ["gold-media"],
            ["repeated-a", "repeated-b", "repeated-c", "repeated-d"],
            ["repeated-a", "repeated-b", "repeated-c", "repeated-d"],
        ),
    ) is True


def test_generic_media_hit_does_not_bypass_abstention():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.rrf_k = 60
    retriever.max_media_results = 4
    retriever.media_abstain_rescue_min_score = 1.25
    retriever.media_abstain_rescue_min_overlap = 0.35
    retriever.media_map = {
        "generic": {
            "id": "generic",
            "media_type": "image",
            "source_url": "https://mbzuai.ac.ae/about",
            "text": "IMAGE: A generic photograph of the university campus.",
        }
    }
    retriever.media_texts_by_id = {
        "generic": "IMAGE: A generic photograph of the university campus."
    }

    assert retriever._has_grounded_media_candidates(
        query="Show the orbital laboratory diagram for MBZUAI's Mars campus.",
        media_rankings=(["generic"], [], []),
    ) is False


@pytest.mark.parametrize(
    "query",
    [
        "ماذا تعرض الصورة في التقرير؟",
        "ما المواقع الموضحة على الخريطة؟",
        "اشرح المخطط المرئي في ملف PDF.",
    ],
)
def test_arabic_visual_queries_route_to_the_media_lane(query):
    from pipeline.retrieval.adaptive_hybrid import _is_media_query

    assert _is_media_query(query) is True


def test_arabic_nonvisual_fact_query_does_not_route_to_the_media_lane():
    from pipeline.retrieval.adaptive_hybrid import _is_media_query

    assert _is_media_query("ما متطلبات القبول في برنامج الماجستير؟") is False


@pytest.mark.parametrize(
    "query",
    [
        "ما المؤهل الأكاديمي المطلوب لوظيفة Data Platform Engineer؟",
        "ما المؤهل الدراسي المطلوب لهذا المنصب؟",
    ],
)
def test_arabic_academic_qualification_query_adds_english_retrieval_aliases(query):
    from pipeline.retrieval.adaptive_hybrid import _semantic_query_alias_tokens

    aliases = set(_semantic_query_alias_tokens(query))

    assert {"academic", "degree", "bachelor", "master", "preferred"} <= aliases


def test_upstream_rewrite_preserves_semantic_aliases_from_original_user_query():
    from pipeline.retrieval.routed_hybrid import QueryRewriteBundle, RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    rewritten = QueryRewriteBundle(
        vector_query="leadership role and governance committees",
        graph_query="leadership role and governance committees",
        labels=("upstream_query_plan",),
    )

    result = retriever._preserve_original_query_aliases(
        rewritten,
        query="leadership role and governance committees",
        original_query="ما الذي تقوله صفحة القيادة والحوكمة عن دور الرئيس؟",
    )

    assert "التنفيذي" in result.vector_query
    assert "الصلاحيات" in result.vector_query
    assert "original_query_semantic_alias_expansion" in result.labels


@pytest.mark.parametrize(
    ("query", "expected_aliases"),
    [
        (
            "ما هي أقسام الوظائف المفتوحة في صفحة الوظائف؟",
            {"faculty", "research", "engineering", "professional", "vacancies"},
        ),
        (
            "What does the MBZUAI Visitor Program say visitors can get hands-on access to?",
            {"research", "experience", "personalized", "demos", "talks"},
        ),
        (
            "How does MBZUAI describe the way it engages with industry and captures value?",
            {"exploration", "refinement", "proposal", "engagement", "agreement"},
        ),
        (
            "ما الدعم الذي وفره معرض التدريب المهني وفرص العمل للطلبة؟",
            {"جلسات", "تدريب", "مهني", "فردية", "وكالات", "التوظيف"},
        ),
        (
            "ما المجالات التي تركز عليها اهتمامات البروفيسورة دانييلا روس البحثية؟",
            {"الاستقلالية", "الذكاء", "autonomy", "intelligence"},
        ),
        (
            "ما الذي تقوله صفحة القيادة والحوكمة عن دور الرئيس؟",
            {"التنفيذي", "مهام", "الصلاحيات", "إدارة", "chief", "executive"},
        ),
    ],
)
def test_named_multifacet_queries_add_page_local_retrieval_aliases(
    query,
    expected_aliases,
):
    from pipeline.retrieval.adaptive_hybrid import _semantic_query_alias_tokens

    assert expected_aliases <= set(_semantic_query_alias_tokens(query))


@pytest.mark.parametrize(
    "query",
    [
        "ما هي أقسام الوظائف المفتوحة في صفحة الوظائف؟",
        "ما الذي يصفه موقع IFM بأنه يبنيه مع الشركاء، وأين يقع مقره؟",
        "How does MBZUAI engage with industry and capture value?",
        "What can visitors get hands-on access to?",
        "ما الدعم الذي وفره معرض التدريب المهني للطلبة؟",
        "ما اللجان التي تشرف على شؤون الجامعة؟",
        "ما الدور الذي تشغله ريما المقرب، وما طبيعة الجهة التي تعمل فيها؟",
    ],
)
def test_multifacet_queries_request_bounded_complete_page_evidence(query):
    from pipeline.retrieval.evidence_packer import _MULTI_DETAIL_QUERY_RE
    from pipeline.retrieval.routed_hybrid import _AGGREGATE_REQUIRED_PAGE_QUERY_RE

    assert _MULTI_DETAIL_QUERY_RE.search(query)
    assert _AGGREGATE_REQUIRED_PAGE_QUERY_RE.search(query)


def test_navigation_target_resolves_to_complete_page_parent():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.vector = SimpleNamespace(
        parent_map={
            "parent:c650:document-revision:directory:page": {
                "id": "parent:c650:document-revision:directory:page",
                "document_revision_id": "document-revision:directory",
                "page_card_ids": ["page-card:directory"],
            },
            "parent:c650:document-revision:directory:section:people": {
                "id": "parent:c650:document-revision:directory:section:people",
                "document_revision_id": "document-revision:directory",
                "page_card_ids": ["page-card:directory"],
            },
        }
    )

    parent_ids = retriever._navigation_target_parent_ids(
        {
            "target_page": {
                "document_revision_id": "document-revision:directory",
                "page_card_id": "page-card:directory",
            }
        }
    )

    assert parent_ids == ["parent:c650:document-revision:directory:page"]


def test_navigation_catalog_resolves_selected_representation_identities():
    from pipeline.retrieval.navigation_planner import (
        NAVIGATION_CATALOG_SCHEMA_VERSION,
        GroundedNavigationPlanner,
    )

    planner = GroundedNavigationPlanner(
        catalog={
            "schema_version": NAVIGATION_CATALOG_SCHEMA_VERSION,
            "source_bridge_coverage_passed": True,
            "source_bridge_status": "passed",
            "pages": [
                {
                    "page_card_id": "page-card:1",
                    "document_revision_id": "document-revision:1",
                    "source_url": "https://mbzuai.ac.ae/page",
                }
            ],
            "chunks": [
                {
                    "chunk_id": "chunk:1",
                    "document_revision_id": "document-revision:1",
                    "page_card_id": "page-card:1",
                    "section_id": "document-section:1",
                    "page_section_ids": ["page-section:1"],
                }
            ],
            "actions": [],
        }
    )

    identities = planner.representation_identities(
        {
            "selected_chunk_ids": ["chunk:1"],
            "dense_page_card_ids": ["page-card:1"],
            "selected_parent_ids": [
                "parent:c650:document-revision:1:page"
            ],
        }
    )

    assert identities == {
        "document_revision_ids": ["document-revision:1"],
        "page_card_ids": ["page-card:1"],
        "section_ids": ["document-section:1", "page-section:1"],
        "chunk_ids": ["chunk:1"],
    }


def test_multi_type_answer_selection_uses_a_bounded_candidate_pool():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.local_answer_candidate_pool = 32
    retriever.local_index_max_postings_per_token = 512
    retriever.answer_map = {}
    retriever.answer_ids_by_type = {"email": [], "website": []}
    retriever.answer_ids_by_subtype = {}
    retriever.answer_token_index = {"salman": [], "khan": []}

    for answer_type in ("email", "website"):
        for index in range(5_000):
            answer_id = f"{answer_type}:{index}"
            retriever.answer_map[answer_id] = {
                "id": answer_id,
                "answer_type": answer_type,
                "answer_subtype": "generic",
                "value": f"irrelevant-{index}",
            }
            retriever.answer_ids_by_type[answer_type].append(answer_id)

    expected_ids = ["email:salman", "website:salman"]
    for answer_id, answer_type in zip(expected_ids, ("email", "website")):
        retriever.answer_map[answer_id] = {
            "id": answer_id,
            "answer_type": answer_type,
            "answer_subtype": "profile",
            "value": f"Salman Khan {answer_type}",
        }
        # Put the relevant records outside the confidence-ordered fallback so
        # the test exercises the token-index path rather than an early seed.
        retriever.answer_ids_by_type[answer_type].append(answer_id)
        retriever.answer_token_index["salman"].append(answer_id)
        retriever.answer_token_index["khan"].append(answer_id)

    scored_ids = []

    def _score(_query, answer):
        scored_ids.append(answer["id"])
        return 10.0 if answer["id"] in expected_ids else 0.1

    retriever._score_answer_record = _score

    selected = retriever._select_answer_ids_for_query(
        "What are Salman Khan's email address and website?",
        [],
        top_k=4,
    )

    assert selected[:2] == expected_ids
    assert len(scored_ids) <= retriever.local_answer_candidate_pool


def test_navigation_action_becomes_conflict_free_answer_evidence():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    payload = {
        "answer_documents": [
            {
                "id": "generic-admissions-email",
                "answer_type": "email",
                "value": "admission@mbzuai.ac.ae",
            },
            {
                "id": "profile-page",
                "answer_type": "website",
                "value": "https://mbzuai.ac.ae/ar/study/faculty/salman-khan-01",
            },
        ],
        "selected_answer_ids": ["generic-admissions-email", "profile-page"],
    }
    navigation_plan = {
        "status": "ready",
        "goal": "Open Salman Khan's email action",
        "target_page": {
            "title": "سلمان خان - MBZUAI",
            "url": "https://mbzuai.ac.ae/ar/study/faculty/salman-khan-01",
            "document_revision_id": "document-revision:salman",
        },
        "steps": [
            {
                "action_id": "page-action:salman-email",
                "action_type": "email",
                "label": "البريد الالكتروني",
                "target_url": "mailto:salman.khan@mbzuai.ac.ae",
                "section_id": "page-section:contact",
                "chunk_ids": ["chunk:salman"],
            }
        ],
    }

    applied = retriever._apply_navigation_action_evidence(
        payload,
        navigation_plan,
    )

    assert applied is True
    assert payload["answer_documents"][0]["id"] == "page-action:salman-email"
    assert payload["answer_documents"][0]["value"] == "salman.khan@mbzuai.ac.ae"
    assert payload["answer_documents"][0]["source_url"].endswith("salman-khan-01")
    assert "generic-admissions-email" not in payload["selected_answer_ids"]
    assert "profile-page" in payload["selected_answer_ids"]


def test_actionable_navigation_target_becomes_a_required_evidence_page():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    coverage_plan = {"required_pages": ["https://mbzuai.ac.ae/about/contact"]}
    navigation_plan = {
        "status": "ready",
        "target_page": {
            "url": "https://library.mbzuai.ac.ae/md-sohail/",
        },
    }

    changed = retriever._require_navigation_target_page(
        coverage_plan,
        navigation_plan,
    )
    changed_again = retriever._require_navigation_target_page(
        coverage_plan,
        navigation_plan,
    )

    assert changed is True
    assert changed_again is False
    assert coverage_plan["required_pages"] == [
        "https://library.mbzuai.ac.ae/md-sohail/",
        "https://mbzuai.ac.ae/about/contact",
    ]


def test_context_page_is_required_only_when_present_in_frozen_corpus():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    context_url = (
        "https://careers.mbzuai.ac.ae/careers/data-platform-engineer-iaai"
    )
    context_record = {
        "source_url": context_url,
        "normalized_url": context_url,
        "document_revision_ids": {"document-revision:job"},
        "linked_chunk_ids": {"chunk:job:1"},
    }
    alias_record = {
        "source_url": f"{context_url}/print",
        "normalized_url": f"{context_url}/print",
        "document_revision_ids": {"document-revision:job"},
        "linked_chunk_ids": {"chunk:job:1"},
    }
    retriever._coverage_page_records_by_url = {
        context_url: context_record,
        f"{context_url}/print": alias_record,
    }
    retriever._coverage_page_records = [context_record, alias_record]
    coverage_plan = {
        "required_pages": [
            f"{context_url}/print",
            "https://mbzuai.ac.ae/news/unrelated",
        ],
        "required_pages_source": "retrieval_payload",
    }

    resolved = retriever._require_context_page(coverage_plan, f"{context_url}/")

    assert resolved == context_url
    assert coverage_plan["required_pages"] == [
        context_url,
        "https://mbzuai.ac.ae/news/unrelated",
    ]
    assert coverage_plan["required_pages_source"] == (
        "context_page+retrieval_payload"
    )
    assert coverage_plan["context_page_url"] == context_url

    unchanged = dict(coverage_plan)
    rejected = retriever._require_context_page(
        coverage_plan,
        "https://example.com/untrusted-page",
    )
    assert rejected == ""
    assert coverage_plan == unchanged


def test_exact_navigation_action_scopes_coverage_to_its_owning_page():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    target_url = "https://library.mbzuai.ac.ae/md-sohail"
    coverage_plan = {
        "required_pages": [
            target_url,
            "https://library.mbzuai.ac.ae/directory-listing",
            "https://mbzuai.ac.ae/news/unrelated",
        ]
    }
    navigation_plan = {
        "status": "ready",
        "target_page": {"url": target_url},
        "steps": [
            {
                "action_type": "email",
                "target_url": "mailto:md.sohail@mbzuai.ac.ae",
            }
        ],
    }

    changed = retriever._require_navigation_target_page(
        coverage_plan,
        navigation_plan,
    )
    changed_again = retriever._require_navigation_target_page(
        coverage_plan,
        navigation_plan,
    )

    assert changed is True
    assert changed_again is False
    assert coverage_plan["required_pages"] == [target_url]


def test_conflicting_navigation_action_cannot_override_explicit_page_requirement():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    coverage_plan = {
        "required_pages": ["https://mbzuai.ac.ae/ar/about/contact"],
        "required_pages_source": "explicit_markers",
    }
    navigation_plan = {
        "status": "ready",
        "confidence": 1.0,
        "source": "page_graph_navigation_catalog",
        "target_page": {
            "url": "https://mbzuai.ac.ae/ar/the-node/commencement-2025-info",
        },
        "steps": [
            {
                "action_id": "page-action:commencement-email",
                "action_type": "email",
                "target_url": "mailto:campus.life@mbzuai.ac.ae",
            }
        ],
        "evidence": {"action_ids": ["page-action:commencement-email"]},
        "warnings": [],
    }

    changed = retriever._require_navigation_target_page(
        coverage_plan,
        navigation_plan,
    )

    assert changed is False
    assert coverage_plan["required_pages"] == [
        "https://mbzuai.ac.ae/ar/about/contact"
    ]
    assert navigation_plan["status"] == "not_requested"
    assert navigation_plan["target_page"] is None
    assert navigation_plan["steps"] == []
    assert navigation_plan["source"] == "explicit_coverage_guard"
    assert navigation_plan["warnings"] == [
        "navigation_action_suppressed_by_explicit_page_requirement"
    ]


def test_catalog_action_named_in_goal_can_resolve_equivalent_page_surface():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    target_url = (
        "https://mbzuai.ac.ae/about/office-of-student-postdoctoral-affairs"
    )
    coverage_plan = {
        "required_pages": ["https://mbzuai.ac.ae/ar/node/861"],
        "required_pages_source": "explicit_markers",
    }
    navigation_plan = {
        "status": "ready",
        "confidence": 1.0,
        "source": "page_graph_navigation_catalog",
        "goal": "What happens if I use the Get in touch action?",
        "target_page": {"url": target_url},
        "steps": [
            {
                "action_id": "page-action:student-affairs-contact",
                "action_type": "contact",
                "label": "Get in touch",
                "target_url": "https://mbzuai.ac.ae/contact-us",
            }
        ],
        "evidence": {"action_ids": ["page-action:student-affairs-contact"]},
        "warnings": [],
    }

    changed = retriever._require_navigation_target_page(
        coverage_plan,
        navigation_plan,
    )

    assert changed is True
    assert coverage_plan["required_pages"] == [target_url]
    assert navigation_plan["status"] == "ready"
    assert navigation_plan["warnings"] == []


def test_navigation_action_can_use_catalog_verified_previous_page_alias():
    from pipeline.core.page_graph_bridge import NAVIGATION_CATALOG_SCHEMA_VERSION
    from pipeline.retrieval.navigation_planner import GroundedNavigationPlanner
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    current_url = "https://mbzuai.ac.ae/ar/about/leadership"
    previous_url = "https://mbzuai.ac.ae/ar/about/leadership-prev"
    planner = GroundedNavigationPlanner(
        catalog={
            "schema_version": NAVIGATION_CATALOG_SCHEMA_VERSION,
            "source_bridge_coverage_passed": True,
            "source_bridge_status": "ready",
            "pages": [
                {
                    "page_card_id": "page-card:current",
                    "source_url": current_url,
                    "title": "القيادة والحوكمة - MBZUAI",
                    "language": "ar",
                    "page_type": "leadership",
                },
                {
                    "page_card_id": "page-card:previous",
                    "source_url": previous_url,
                    "title": "القيادة والحوكمة - MBZUAI",
                    "language": "ar",
                    "page_type": "leadership",
                },
            ],
            "chunks": [],
            "actions": [],
        }
    )
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.navigation_planner = planner
    coverage_plan = {
        "required_pages": [current_url],
        "required_pages_source": "explicit_markers",
    }
    navigation_plan = {
        "status": "ready",
        "target_page": {"url": previous_url},
        "steps": [
            {
                "action_id": "page-action:governance",
                "action_type": "download",
                "target_url": "https://staticcdn.mbzuai.ac.ae/governance.pdf",
            }
        ],
        "warnings": [],
    }

    assert planner.page_urls_share_identity(current_url, previous_url) is True
    assert retriever._require_navigation_target_page(coverage_plan, navigation_plan) is True
    assert coverage_plan["required_pages"] == [previous_url]
    assert navigation_plan["status"] == "ready"
    assert navigation_plan["warnings"] == [
        "navigation_target_page_alias_resolved"
    ]


def test_evidence_pack_prioritizes_validated_navigation_action():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    result = {
        "answer_documents": [
            {
                "id": "page-action:salman-email",
                "record_type": "navigation_action",
                "answer_type": "email",
                "text": "The verified email action points to salman.khan@mbzuai.ac.ae.",
                "source_url": "https://mbzuai.ac.ae/ar/study/faculty/salman-khan-01",
                "document_title": "سلمان خان - MBZUAI",
                "confidence": 1.0,
                "authority_score": 1.0,
            },
            {
                "id": "generic-admissions-email",
                "text": "The admissions email is admission@mbzuai.ac.ae.",
                "source_url": "https://mbzuai.ac.ae/about/contact",
            },
        ],
        "retrieval_documents": [
            {
                "id": "profile-context",
                "text": "Salman Khan is an associate professor of computer vision.",
                "source_url": "https://mbzuai.ac.ae/ar/study/faculty/salman-khan-01",
            }
        ],
    }

    evidence_pack = build_evidence_pack(
        query="Where does Salman Khan's email action point?",
        result=result,
        max_items=4,
    )

    assert evidence_pack["items"][0]["kind"] == "action"
    assert evidence_pack["items"][0]["id"] == "page-action:salman-email"
    assert all(
        "admission@mbzuai.ac.ae" not in item["text"]
        for item in evidence_pack["items"]
    )
    assert any(item["id"] == "profile-context" for item in evidence_pack["items"])


def test_evidence_pack_reserves_contiguous_chunk_for_using_explanation():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    source_url = "https://ai-nexus.mbzuai.ac.ae/previous-ai-talks"
    result = {
        "fact_documents": [
            {
                "id": "speaker-fact",
                "text": "Xiang Meng gave the average-hazard talk.",
                "source_url": source_url,
            }
        ],
        "evidence_span_documents": [
            {
                "id": "definition-span",
                "text": "Average hazard is an interpretable hazard-scale measure.",
                "source_url": source_url,
                "span_type": "sentence_window",
            },
            {
                "id": "title-span",
                "text": "Average Hazard for Robust Survival Analysis by Xiang Meng.",
                "source_url": source_url,
                "span_type": "sentence_window",
            },
            {
                "id": "outcome-span",
                "text": "It can support more reliable trial conclusions and inform drug approvals.",
                "source_url": source_url,
                "span_type": "sentence_window",
            },
        ],
        "retrieval_documents": [
            {
                "id": "chunk:average-hazard",
                "text": (
                    "The talk defines average hazard as an interpretable hazard-scale measure. "
                    "It compares the measure with the Cox hazard ratio and says average hazard "
                    "can support more reliable trial conclusions and inform drug approvals."
                ),
                "source_url": source_url,
            }
        ],
    }

    pack = build_evidence_pack(
        query="What does Xiang Meng's talk say about using average hazard?",
        result=result,
        max_items=4,
        max_per_source=2,
        coverage_plan={
            "intent": "broad_synthesis",
            "required_pages": [source_url],
        },
    )

    assert pack["items"][0]["id"] == "chunk:average-hazard"
    assert "more reliable trial conclusions" in pack["items"][0]["text"]


def test_evidence_pack_recognizes_example_shown_in_form_as_media_query():
    from pipeline.retrieval.evidence_packer import (
        _is_explicit_media_query,
        build_evidence_pack,
    )

    query = (
        "What academic history example is shown in the form, and what does it "
        "say about the GPA fields?"
    )
    media_text = (
        "IMAGE: Academic history form\n"
        "VISIBLE_TEXT: Carnegie Mellon University Bachelor Artificial Intelligence "
        "CGPA 4.0 Maximum Possible GPA 4.0. The maximum possible GPA is the maximum "
        "score on the university grading scale."
    )
    pack = build_evidence_pack(
        query=query,
        result={
            "evidence_span_documents": [
                {
                    "id": "generic-gpa",
                    "text": "Successful applicants generally have strong grades.",
                    "source_url": "https://mbzuai.ac.ae/study/undergraduate-program",
                    "span_type": "sentence_window",
                }
            ],
            "media": [
                {
                    "id": "academic-form",
                    "text": media_text,
                    "source_url": "https://staticcdn.mbzuai.ac.ae/application.pdf",
                }
            ],
        },
        max_items=4,
        max_chars=4000,
    )

    assert _is_explicit_media_query(query)
    assert pack["items"][0]["id"] == "academic-form"
    assert "Carnegie Mellon University" in pack["items"][0]["text"]


def test_evidence_pack_adds_arabic_president_role_answer_aliases():
    from pipeline.retrieval.evidence_packer import _query_terms

    terms = _query_terms(
        "ما الذي تقوله صفحة القيادة والحوكمة عن دور الرئيس؟"
    )

    assert {
        "الرئيس التنفيذي",
        "التنفيذي",
        "الصلاحيات",
        "إدارة الجامعة",
        "إدارة",
    } <= terms


def test_evidence_pack_prefers_exact_arabic_president_role_chunk():
    from pipeline.retrieval.evidence_packer import build_evidence_pack

    leadership_url = "https://mbzuai.ac.ae/ar/about/leadership"
    governance_url = "https://staticcdn.mbzuai.ac.ae/governance.pdf"
    pack = build_evidence_pack(
        query=(
            "ما الذي تقوله صفحة القيادة والحوكمة عن دور الرئيس، وما الذي توضحه "
            "وثيقة الحوكمة عن اللجان التي تشرف على شؤون الجامعة؟"
        ),
        result={
            "retrieval_documents": [
                {
                    "id": "chunk:generic-opening",
                    "text": "يرحب الرئيس بمجلس الأمناء ويعرض نمو الجامعة وشراكاتها.",
                    "source_url": leadership_url,
                },
                {
                    "id": "chunk:exact-role",
                    "text": (
                        "يؤدي رئيس الجامعة مهام الرئيس التنفيذي ويمارس، بتوجيه مجلس "
                        "الأمناء، الصلاحيات اللازمة لإدارة الجامعة وشؤونها."
                    ),
                    "source_url": leadership_url,
                },
                {
                    "id": "chunk:governance",
                    "text": (
                        "The board and management committees oversee strategic and "
                        "operational matters of the university."
                    ),
                    "source_url": governance_url,
                },
            ]
        },
        max_items=6,
        max_chars=5000,
        coverage_plan={
            "intent": "broad_synthesis",
            "required_pages": [leadership_url, governance_url],
        },
    )

    leadership_items = [
        item for item in pack["items"] if item["source_url"] == leadership_url
    ]
    assert leadership_items[0]["id"] == "chunk:exact-role"
