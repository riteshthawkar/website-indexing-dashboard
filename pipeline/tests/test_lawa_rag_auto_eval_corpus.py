"""Behavioral tests for normalized automatic-evaluation corpus snapshots."""

from __future__ import annotations

from lawa_rag.auto_eval import CorpusSnapshot, RetrievalBundleAdapter
from pipeline.tests.lawa_rag_auto_eval_support import sample_bundle


def test_retrieval_bundle_adapter_preserves_multimodal_actions_and_graph() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")

    assert len(snapshot.sources) == 2
    assert {item.kind for item in snapshot.evidence} >= {
        "chunk",
        "fact",
        "media",
        "page_card",
        "action",
        "table",
    }
    assert snapshot.actions[0].action_type == "apply"
    assert snapshot.actions[0].target == "https://apply.example.edu/"
    edges = {
        (item.source_id, item.target_id, item.relation)
        for item in snapshot.relationships
    }
    assert (
        "chunk:programs",
        "chunk:admissions",
        "requires_application",
    ) in edges
    assert (
        "page-action:apply",
        "page-card:admissions",
        "described_by_page",
    ) in edges


def test_snapshot_identity_is_content_stable_and_round_trips() -> None:
    first = RetrievalBundleAdapter.from_mapping(sample_bundle(), domain="university")
    second = RetrievalBundleAdapter.from_mapping(
        sample_bundle(), domain="university", metadata={"run": "another-run"}
    )
    restored = CorpusSnapshot.from_dict(first.to_dict())

    assert first.snapshot_id == second.snapshot_id
    assert restored == first
    assert restored.validation_errors() == ()
    assert restored.summary()["source_count"] == 2

    modified = first.to_dict()
    modified["evidence"][0]["text"] = "Tampered evidence text that is long enough."
    try:
        CorpusSnapshot.from_dict(modified)
    except ValueError as exc:
        assert "snapshot identity" in str(exc)
    else:  # pragma: no cover - protects immutable evaluation inputs
        raise AssertionError("modified snapshot retained a stale identity")


def test_adapter_rejects_bundles_without_usable_evidence() -> None:
    try:
        RetrievalBundleAdapter.from_mapping(
            {"chunk_records": [{"id": "empty", "text": "short"}]}
        )
    except ValueError as exc:
        assert "at least one source and one evidence unit" in str(exc)
    else:  # pragma: no cover - protects the fail-closed boundary
        raise AssertionError("empty retrieval bundle was accepted")


def test_adapter_normalizes_pdf_and_database_sources() -> None:
    snapshot = RetrievalBundleAdapter.from_mapping(
        {
            "chunk_records": [
                {
                    "id": "pdf:page-1",
                    "document_id": "handbook",
                    "document_title": "Handbook",
                    "source_url": "https://example.edu/handbook.pdf",
                    "text": "The handbook describes the official academic progression policy.",
                    "page_numbers": 1,
                }
            ],
            "database_records": [
                {
                    "id": "database:program-count",
                    "document_id": "approved-program-view",
                    "document_title": "Approved program reporting view",
                    "document_type": "database",
                    "text": "The approved reporting view contains twelve active programs.",
                }
            ],
        },
        domain="university",
    )

    assert {item.source_type for item in snapshot.sources} == {"pdf", "database"}
    assert {item.kind for item in snapshot.evidence} == {"chunk", "database"}
    assert "pages 1" in snapshot.evidence_map()["pdf:page-1"].locator
