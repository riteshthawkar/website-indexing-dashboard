"""Shared fixtures for the public automatic-evaluation package tests."""

from __future__ import annotations

from lawa_rag.auto_eval import CorpusSnapshot, EvidenceUnit, Persona, SourceRecord


def sample_bundle() -> dict:
    base = {
        "document_revision_id": "document-revision:programs",
        "document_id": "programs",
        "document_title": "Academic Programs",
        "source_url": "https://example.edu/programs",
        "language": "English",
    }
    admissions = {
        "document_revision_id": "document-revision:admissions",
        "document_id": "admissions",
        "document_title": "Admissions",
        "source_url": "https://example.edu/admissions",
        "language": "English",
    }
    return {
        "version": 6,
        "chunk_records": [
            {
                **base,
                "id": "chunk:programs",
                "record_type": "chunk",
                "text": "The data science master's program includes machine learning and statistics.",
                "section_id": "section:programs",
                "neighbor_ids": ["chunk:admissions"],
            },
            {
                **admissions,
                "id": "chunk:admissions",
                "record_type": "chunk",
                "text": "Applicants submit transcripts, a statement, and two recommendation letters.",
                "section_id": "section:requirements",
            },
        ],
        "fact_records": [
            {
                **base,
                "id": "fact:duration",
                "record_type": "fact",
                "text": "The full-time master's program normally takes two years.",
                "linked_chunk_ids": ["chunk:programs"],
            }
        ],
        "media_records": [
            {
                **base,
                "id": "media:curriculum",
                "record_type": "media",
                "text": "A curriculum diagram shows foundation, core, and elective courses.",
                "linked_chunk_ids": ["chunk:programs"],
            }
        ],
        "page_card_records": [
            {
                **admissions,
                "id": "page-card:admissions",
                "record_type": "page_card",
                "purpose_summary": "Explains requirements and links applicants to the application form.",
            }
        ],
        "action_records": [
            {
                **admissions,
                "id": "page-action:apply",
                "action_id": "page-action:apply",
                "record_type": "action",
                "label": "Apply for graduate admission",
                "action_type": "apply",
                "target_url": "https://apply.example.edu/",
                "page_card_ids": ["page-card:admissions"],
            }
        ],
        "table_records": [
            {
                **base,
                "id": "table:credits",
                "record_type": "table",
                "text": "Core courses: 18 credits. Electives: 12 credits. Thesis: 6 credits.",
            }
        ],
        "graph_edges": [
            {
                "source_id": "chunk:programs",
                "target_id": "chunk:admissions",
                "relation": "requires_application",
            }
        ],
    }


def simple_snapshot() -> CorpusSnapshot:
    source = SourceRecord(
        source_id="source:handbook",
        revision_id="revision:handbook:1",
        source_type="webpage",
        title="Student Handbook",
        uri="https://example.edu/handbook",
        language="English",
    )
    evidence = EvidenceUnit(
        evidence_id="evidence:library-hours",
        source_id=source.source_id,
        kind="fact",
        text="The university library is open from 8:00 AM to 8:00 PM on weekdays.",
        locator="Student Handbook | Library",
        language="English",
        metadata={"record_group": "fact_records"},
    )
    return CorpusSnapshot.build(domain="university", sources=(source,), evidence=(evidence,))


def student_persona() -> Persona:
    return Persona.create(
        role="current student",
        goals=("find practical campus information",),
        likely_tasks=("fact", "navigate"),
        domain="university",
    )
