from __future__ import annotations

import json
from pathlib import Path

from pipeline.core.io import atomic_write_json
from pipeline.core.knowledge_graph import (
    build_graph_bundle,
    make_graph_edge,
    make_graph_node,
    validate_graph_index_derivation,
)
from pipeline.core.retrieval_migration import (
    RetrievalMigrationOptions,
    prepare_retrieval_v2_migration,
)
from pipeline.core.run_audit import audit_run
from pipeline.stages.embedders import gemini_pinecone_embedder


def _write_source_run(root: Path, *, include_assertion: bool = True) -> Path:
    source = root / "source-run"
    bundle_dir = source / "stage_outputs" / "build_retrieval_bundle"
    graph_dir = source / "stage_outputs" / "promote_graph"
    bundle_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)

    chunk = {
        "id": "chunk-1",
        "record_type": "chunk",
        "text": "MBZUAI offers graduate artificial intelligence programs in Abu Dhabi.",
        "dense_text": "MBZUAI offers graduate artificial intelligence programs in Abu Dhabi.",
        "lexical_text": "MBZUAI offers graduate artificial intelligence programs in Abu Dhabi.",
        "sparse_text": "MBZUAI offers graduate artificial intelligence programs in Abu Dhabi.",
        "document_id": "doc-1",
        "document_title": "Academic Programs",
        "document_type": "webpage",
        "source_url": "https://mbzuai.ac.ae/study",
        "section_path": ["Study"],
        "page_numbers": [],
        "page_key": "page-1",
        "section_key": "parent-1",
    }
    parent = {
        "id": "parent-1",
        "record_type": "parent",
        "parent_type": "section",
        "document_id": "doc-1",
        "document_title": "Academic Programs",
        "document_type": "webpage",
        "source_url": "https://mbzuai.ac.ae/study",
        "section_path": ["Study"],
        "page_numbers": [],
        "page_key": "page-1",
        "child_chunk_ids": ["chunk-1"],
        "dense_text": chunk["dense_text"],
        "lexical_text": chunk["lexical_text"],
        "sparse_text": chunk["sparse_text"],
    }
    fact = {
        "id": "fact-1",
        "record_type": "fact",
        "text": "MBZUAI offers graduate artificial intelligence programs.",
        "dense_text": "FACT: MBZUAI offers graduate artificial intelligence programs.",
        "lexical_text": "MBZUAI offers graduate artificial intelligence programs.",
        "sparse_text": "MBZUAI offers graduate artificial intelligence programs.",
        "document_id": "doc-1",
        "document_title": "Academic Programs",
        "source_url": "https://mbzuai.ac.ae/study",
        "linked_chunk_ids": ["chunk-1"],
        "linked_parent_ids": ["parent-1"],
    }
    answer = {
        "id": "answer-1",
        "record_type": "answer",
        "answer_type": "program_area",
        "answer_subtype": "program_area",
        "subject_text": "MBZUAI",
        "value": "artificial intelligence",
        "text": "MBZUAI covers artificial intelligence.",
        "dense_text": "MBZUAI covers artificial intelligence.",
        "lexical_text": "MBZUAI covers artificial intelligence.",
        "sparse_text": "MBZUAI covers artificial intelligence.",
    }
    lexical = [
        {"id": "chunk-1", "record_type": "chunk", "text": chunk["sparse_text"], "tokens": ["mbzuai"]},
        {"id": "parent-1", "record_type": "parent", "text": parent["sparse_text"], "tokens": ["mbzuai"]},
        {"id": "fact-1", "record_type": "fact", "text": fact["sparse_text"], "tokens": ["mbzuai"]},
        {"id": "answer-1", "record_type": "answer", "text": answer["sparse_text"], "tokens": ["mbzuai"]},
    ]
    bundle = {
        "version": 4,
        "generated_at": "source-run",
        "chunk_records": [chunk],
        "parent_records": [parent],
        "media_records": [],
        "fact_records": [fact],
        "summary_records": [],
        "entity_records": [],
        "assertion_records": [],
        "answer_records": [answer],
        "stats": {
            "chunk_count": 1,
            "parent_count": 1,
            "media_count": 0,
            "fact_count": 1,
            "summary_count": 0,
            "entity_count": 0,
            "assertion_count": 0,
            "answer_count": 1,
            "lexical_count": 4,
        },
    }
    files = {
        "retrieval_bundle.json": bundle,
        "chunk_dense_records.json": [chunk],
        "parent_dense_records.json": [parent],
        "media_dense_records.json": [],
        "fact_dense_records.json": [fact],
        "assertion_dense_records.json": [],
        "answer_dense_records.json": [answer],
        "lexical_corpus.json": lexical,
    }
    for filename, payload in files.items():
        atomic_write_json(bundle_dir / filename, payload)

    nodes = [
        make_graph_node(
            node_id="entity:mbzuai",
            node_type="entity",
            label="MBZUAI",
            properties={"entity_type": "organization", "canonical_name": "MBZUAI"},
        ),
        make_graph_node(
            node_id="entity:ai",
            node_type="entity",
            label="artificial intelligence",
            properties={"entity_type": "program_area", "canonical_name": "artificial intelligence"},
        ),
    ]
    edges = []
    if include_assertion:
        nodes.append(
            make_graph_node(
                node_id="assertion:program-area",
                node_type="relation_assertion",
                label="PROGRAM_AREA",
                properties={
                    "relation_type": "program_area",
                    "subject_entity_id": "entity:mbzuai",
                    "object_entity_id": "entity:ai",
                    "subject_name": "MBZUAI",
                    "object_name": "artificial intelligence",
                    "evidence": "MBZUAI offers graduate artificial intelligence programs.",
                    "text": "MBZUAI covers the program area artificial intelligence.",
                    "confidence": 0.9,
                    "authority_class": "canonical_page",
                    "authority_score": 1.0,
                    "freshness_score": 1.0,
                    "source_chunk_ids": ["chunk-1"],
                    "source_parent_ids": ["parent-1"],
                    "source_fact_ids": ["fact-1"],
                    "source_url": "https://mbzuai.ac.ae/study",
                    "document_title": "Academic Programs",
                },
            )
        )
        edges.extend(
            [
                make_graph_edge(
                    edge_type="ASSERTION_SUBJECT",
                    source_id="assertion:program-area",
                    target_id="entity:mbzuai",
                    qualifier="subject",
                ),
                make_graph_edge(
                    edge_type="ASSERTION_OBJECT",
                    source_id="assertion:program-area",
                    target_id="entity:ai",
                    qualifier="object",
                ),
            ]
        )
    else:
        edges.append(
            make_graph_edge(
                edge_type="RELATED_TO",
                source_id="entity:mbzuai",
                target_id="entity:ai",
                qualifier="fallback",
            )
        )
    graph = build_graph_bundle(nodes=nodes, edges=edges, schema_version=2, graph_type="promoted_semantic_graph")
    atomic_write_json(graph_dir / "promoted_knowledge_graph.json", graph)
    return source


def test_retrieval_migration_dry_run_generates_v2_records(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    target = tmp_path / "target-run"
    prepared = prepare_retrieval_v2_migration(
        RetrievalMigrationOptions(
            source_work_dir=source,
            target_work_dir=target,
            config={"project_name": "mbzuai_main"},
            dry_run=True,
            min_summary_coverage=1.0,
        )
    )

    assert prepared.ok
    assert prepared.manifest["status"] == "dry_run_ready"
    assert prepared.manifest["target_counts"]["chunk_count"] == 1
    assert prepared.manifest["target_counts"]["summary_count"] == 1
    assert prepared.manifest["target_counts"]["assertion_count"] == 1
    assert prepared.manifest["target_counts"]["entity_count"] == 2
    assert prepared.manifest["target_counts"]["answer_count"] >= 1
    assert not target.exists()


def test_retrieval_migration_writes_auditable_target_run(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path)
    target = tmp_path / "target-run"
    prepared = prepare_retrieval_v2_migration(
        RetrievalMigrationOptions(
            source_work_dir=source,
            target_work_dir=target,
            config={"project_name": "mbzuai_main"},
            min_summary_coverage=1.0,
        )
    )

    assert prepared.ok
    manifest_path = target / "stage_outputs" / "migrate_retrieval" / "migration_manifest.json"
    bundle_path = target / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    assert manifest_path.is_file()
    assert bundle_path.is_file()

    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert bundle["version"] >= 5
    assert bundle["stats"]["summary_count"] == 1
    assert bundle["stats"]["assertion_count"] == 1
    assert bundle["summary_records"][0]["linked_chunk_ids"] == ["chunk-1"]
    assert bundle["assertion_records"][0]["source_chunk_ids"] == ["chunk-1"]
    promoted_assertions = (
        target / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
    )
    assert promoted_assertions.is_file()
    assert json.loads(promoted_assertions.read_text(encoding="utf-8"))
    assert validate_graph_index_derivation(
        target / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph.json",
        target
        / "stage_outputs"
        / "promote_graph"
        / "promoted_knowledge_graph_index.json",
    ) == []

    audit = audit_run(target)
    assert audit.ok, audit.to_dict()


def test_retrieval_migration_blocks_graph_without_assertions(tmp_path: Path) -> None:
    source = _write_source_run(tmp_path, include_assertion=False)
    target = tmp_path / "target-run"
    prepared = prepare_retrieval_v2_migration(
        RetrievalMigrationOptions(
            source_work_dir=source,
            target_work_dir=target,
            config={"project_name": "mbzuai_main"},
            dry_run=True,
            min_summary_coverage=1.0,
        )
    )

    assert not prepared.ok
    assert any("relation_assertion" in error for error in prepared.manifest["validation"]["errors"])


def test_sparse_index_creation_supports_current_pinecone_sdk(monkeypatch) -> None:
    class FakePinecone:
        def __init__(self) -> None:
            self.created = None

        def has_index(self, name: str) -> bool:
            return False

        def create_index_for_model(self, **kwargs):
            self.created = kwargs

    fake = FakePinecone()
    monkeypatch.setattr(gemini_pinecone_embedder, "_wait_for_index_ready", lambda *args, **kwargs: None)

    existed = gemini_pinecone_embedder._ensure_sparse_index(
        fake,
        index_name="sparse-test",
        cloud="aws",
        region="us-east-1",
        sparse_model="pinecone-sparse-english-v0",
        sparse_text_field="chunk_text",
    )

    assert existed is False
    assert fake.created["name"] == "sparse-test"
    embed = fake.created["embed"]
    if isinstance(embed, dict):
        assert embed["model"] == "pinecone-sparse-english-v0"
        assert embed["field_map"] == {"text": "chunk_text"}
    else:
        assert embed.model == "pinecone-sparse-english-v0"
        assert embed.field_map == {"text": "chunk_text"}


def test_retry_classifier_treats_tls_eof_as_transient() -> None:
    exc = RuntimeError(
        "[SSL: UNEXPECTED_EOF_WHILE_READING] EOF occurred in violation of protocol (_ssl.c:1000)"
    )

    assert gemini_pinecone_embedder._is_retryable_exception(exc)


def test_retry_classifier_treats_remote_disconnect_as_transient() -> None:
    exc = RuntimeError("httpx.RemoteProtocolError: Server disconnected without sending a response.")

    assert gemini_pinecone_embedder._is_retryable_exception(exc)


def test_sparse_records_use_entity_canonical_fields_when_text_is_missing() -> None:
    records, stats = gemini_pinecone_embedder._build_sparse_records(
        [
            {
                "id": "entity:board-of-trustees",
                "canonical_name": "Board of Trustees",
                "aliases": ["BOT"],
                "description": "University governance body.",
            }
        ],
        sparse_text_field="chunk_text",
        max_text_chars=4000,
        max_record_bytes=38000,
    )

    assert stats == {"trimmed": 0, "skipped": 0}
    assert records == [
        {
            "_id": "entity:board-of-trustees",
            "chunk_text": "Board of Trustees BOT University governance body.",
        }
    ]


def test_sparse_records_keep_numeric_entities_with_type_context() -> None:
    records, stats = gemini_pinecone_embedder._build_sparse_records(
        [
            {
                "id": "entity:2030",
                "canonical_name": "2030",
                "aliases": ["2030"],
                "entity_type": "date",
            }
        ],
        sparse_text_field="chunk_text",
        max_text_chars=4000,
        max_record_bytes=38000,
    )

    assert stats == {"trimmed": 0, "skipped": 0}
    assert records == [{"_id": "entity:2030", "chunk_text": "2030 date"}]


def test_graph_entity_record_uses_nested_properties_for_embedding_text() -> None:
    record = gemini_pinecone_embedder._graph_entity_record(
        {
            "id": "entity:mbzuai",
            "node_type": "entity",
            "label": "MBZUAI",
            "properties": {
                "canonical_name": "MBZUAI",
                "aliases": ["Mohamed bin Zayed University of Artificial Intelligence"],
                "description": "Graduate AI university in Abu Dhabi.",
                "entity_type": "organization",
            },
        }
    )

    assert record["text"] == (
        "MBZUAI Mohamed bin Zayed University of Artificial Intelligence "
        "Graduate AI university in Abu Dhabi."
    )
    assert record["dense_text"] == record["text"]
    assert record["entity_type"] == "organization"
