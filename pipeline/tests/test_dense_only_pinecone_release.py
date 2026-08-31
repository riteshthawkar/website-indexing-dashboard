from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from pipeline.core.base import StageContext, StageStatus


ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = ROOT / "scripts" / "deploy" / "validate-release-artifacts.py"


def _validator_module():
    spec = importlib.util.spec_from_file_location("release_artifact_validator", VALIDATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selected_upload() -> dict:
    lanes = (
        "chunks",
        "parents",
        "media",
        "page_cards",
        "actions",
        "facts",
        "evidence_spans",
        "summaries",
        "assertions",
        "entities",
        "communities",
    )
    namespaces = {lane: f"{lane}--release-1" for lane in lanes}
    counts = {lane: 0 for lane in lanes}
    counts.update({"chunks": 2, "parents": 2, "media": 1, "page_cards": 1, "actions": 1})
    return {
        "schema_version": 6,
        "provider": "pinecone",
        "production_indexing_contract_fingerprint": "a" * 64,
        "model": "gemini-embedding-2",
        "output_dimensionality": 1536,
        "index_name": "mbzuai-gemini-retrieval-v3",
        "sparse_index_name": "",
        "namespace_strategy": "release",
        "namespace_release_id": "release-1",
        "namespaces": namespaces,
        "planned": counts,
        "uploaded": counts,
        "verification": {
            "dense": {
                "expected": {
                    namespaces[lane]: count for lane, count in counts.items() if count
                },
                "actual": {
                    namespaces[lane]: count for lane, count in counts.items() if count
                },
                "failures": [],
            }
        },
    }


def test_selected_dense_only_pinecone_manifest_is_supported():
    validator = _validator_module()

    errors = validator._validate_upload_manifest(
        upload=_selected_upload(),
        run_id="release-1",
        expected_model="gemini-embedding-2",
        expected_dimension=1536,
        selected_profile=True,
    )

    assert errors == []


def test_deployment_validator_prefers_complete_summarized_community_graph(tmp_path):
    validator = _validator_module()
    community_dir = tmp_path / "stage_outputs" / "community_graph"
    community_dir.mkdir(parents=True)
    community_graph = community_dir / "community_knowledge_graph.json"
    community_index = community_dir / "community_knowledge_graph_index.json"
    community_graph.write_text("{}", encoding="utf-8")
    community_index.write_text("{}", encoding="utf-8")

    summarized_dir = tmp_path / "stage_outputs" / "summarize_community_graph"
    summarized_dir.mkdir(parents=True)
    summarized_graph = summarized_dir / "summarized_community_graph.json"
    summarized_index = summarized_dir / "summarized_community_graph_index.json"
    summarized_graph.write_text("{}", encoding="utf-8")
    summarized_index.write_text("{}", encoding="utf-8")

    graph_path, index_path, graph_kind = validator._select_runtime_graph(tmp_path)

    assert graph_path == summarized_graph
    assert index_path == summarized_index
    assert graph_kind == "summarized_community_local_graph"

    summarized_index.unlink()
    try:
        validator._select_runtime_graph(tmp_path)
    except validator.ValidationError as exc:
        assert "canonical graph artifact is incomplete" in str(exc)
    else:
        raise AssertionError("an incomplete summarized graph must fail closed")


def test_dense_only_pinecone_manifest_rejects_hidden_sparse_counts():
    validator = _validator_module()
    upload = _selected_upload()
    upload["planned"]["sparse_chunks"] = 2
    upload["uploaded"]["sparse_chunks"] = 2

    errors = validator._validate_upload_manifest(
        upload=upload,
        run_id="release-1",
        expected_model="gemini-embedding-2",
        expected_dimension=1536,
        selected_profile=True,
    )

    assert any("non-zero sparse counts" in error for error in errors)


def test_selected_pinecone_upload_includes_page_cards_and_actions(tmp_path, monkeypatch):
    from pipeline.stages.embedders import selected_pinecone as module

    lane_kinds = {
        "chunks": "chunk",
        "parents": "parent",
        "media": "media",
        "page_cards": "page_card",
        "actions": "action",
        "facts": "fact",
        "evidence_spans": "evidence_span",
        "summaries": "summary",
        "assertions": "assertion",
        "entities": "entity",
        "communities": "community",
    }
    lanes = {
        lane: ([{"id": f"{kind}-1", "text": f"{kind} text"}] if lane in {
            "chunks", "parents", "media", "page_cards", "actions"
        } else [])
        for lane, kind in lane_kinds.items()
    }
    bundle_path = tmp_path / "retrieval_bundle.json"
    bundle_path.write_text(json.dumps({"version": 6}), encoding="utf-8")
    identity = {
        "retrieval_bundle_sha256": "a" * 64,
        "lexical_corpus_sha256": "b" * 64,
        "promoted_assertions_sha256": "c" * 64,
        "knowledge_graph_sha256": "d" * 64,
        "knowledge_graph_index_sha256": "e" * 64,
        "selected_release_assembly_sha256": "f" * 64,
        "selected_release_binding_sha256": "1" * 64,
        "page_graph_navigation_catalog_sha256": "2" * 64,
        "upload_input_sha256": "3" * 64,
        "knowledge_graph_kind": "promoted_local_graph",
        "knowledge_graph_index_file": "/tmp/graph-index.json",
    }
    monkeypatch.setattr(module, "_artifact_identity", lambda *_args, **_kwargs: identity)
    monkeypatch.setattr(
        module,
        "_indexing_build_identity",
        lambda *_args, **_kwargs: ({"commit_sha": "4" * 40, "dirty": False}, "5" * 64),
    )
    monkeypatch.setattr(module, "_load_lane_records", lambda *_args, **_kwargs: lanes)
    monkeypatch.setattr(
        module,
        "_validate_selected_lanes",
        lambda *_args, **_kwargs: {
            "assembly_sha256": "1" * 64,
            "embedding_spec": {"media_input": "caption_text"},
        },
    )
    monkeypatch.setattr(
        module,
        "_apply_selected_media_input_contract",
        lambda *_args, **_kwargs: "caption_text",
    )
    monkeypatch.setattr(module, "_make_pinecone_client", lambda: object())
    monkeypatch.setattr(module, "_ensure_index", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(module, "_make_gemini_client", lambda **_kwargs: object())
    monkeypatch.setattr(
        module,
        "_embed_text_batch",
        lambda _client, *, texts, **_kwargs: [[0.1, 0.2] for _ in texts],
    )

    class FakeIndex:
        def __init__(self):
            self.counts = {}
            self.uploaded_ids = {}

        def delete(self, *, namespace, **_kwargs):
            self.counts[namespace] = 0
            self.uploaded_ids[namespace] = []

        def upsert(self, *, namespace, vectors, **_kwargs):
            self.uploaded_ids.setdefault(namespace, []).extend(item["id"] for item in vectors)
            self.counts[namespace] = len(self.uploaded_ids[namespace])

        def describe_index_stats(self):
            return {
                "namespaces": {
                    namespace: {"vector_count": count}
                    for namespace, count in self.counts.items()
                    if count
                }
            }

    index = FakeIndex()
    monkeypatch.setattr(module, "_make_index_handle", lambda _name: index)

    config = {
        "pipeline": {"production_profile": False},
        "selected_profile": {
            "variant_id": "c650__gemini2_1536__dense_graph",
            "record_kinds": [
                "chunk",
                "parent",
                "parent_section",
                "media",
                "page_card",
                "action",
            ],
        },
        "embedder": {
            "engine": "gemini",
            "model": "gemini-embedding-2",
            "output_dimensionality": 2,
            "pinecone_index": "test-index",
            "pinecone_sparse_index": "",
            "enable_sparse": False,
            "namespace_strategy": "release",
            "namespace_release_template": "{base}--{release_id}",
            "batch_size": 2,
        },
        "vector_store": {"provider": "pinecone", "dimensions": 2},
    }
    ctx = StageContext(
        run_id="release-1",
        project_name="mbzuai_main",
        config=config,
        work_dir=tmp_path,
        stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
        stage_id="upload_retrieval",
    )
    paths = {
        "bundle": str(bundle_path),
        "lexical_corpus": "/tmp/lexical.json",
        "promoted_assertions": "/tmp/assertions.json",
        "graph_bundle": "/tmp/graph.json",
        "release_assembly": "/tmp/assembly.json",
        "navigation_catalog": "/tmp/navigation.json",
    }

    result = module.execute_selected_profile_pinecone(ctx, config["embedder"], paths)

    assert result.status == StageStatus.COMPLETED
    manifest = json.loads(
        (ctx.stage_work_dir / "index_upload_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["schema_version"] == 6
    assert manifest["provider"] == "pinecone"
    assert manifest["sparse_index_name"] == ""
    assert manifest["uploaded"]["page_cards"] == 1
    assert manifest["uploaded"]["actions"] == 1
    assert index.uploaded_ids[manifest["namespaces"]["page_cards"]] == ["page_card-1"]
    assert index.uploaded_ids[manifest["namespaces"]["actions"]] == ["action-1"]


def test_existing_pinecone_index_must_match_embedding_contract(monkeypatch):
    from pipeline.stages.embedders import gemini_pinecone_embedder as module

    monkeypatch.setitem(
        __import__("sys").modules,
        "pinecone",
        type("PineconeModule", (), {"ServerlessSpec": object}),
    )

    class Client:
        def has_index(self, _name):
            return True

        def describe_index(self, _name):
            return {"dimension": 1024, "metric": "cosine"}

    try:
        module._ensure_index(
            Client(),
            index_name="mbzuai-gemini-retrieval-v3",
            dimension=1536,
            cloud="aws",
            region="us-east-1",
        )
    except ValueError as exc:
        assert "dimension 1024; expected 1536" in str(exc)
    else:
        raise AssertionError("dimension drift must fail before upload")


def test_pinecone_runtime_uses_ranked_ids_and_hydrates_records_from_local_bundle():
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever

    class Index:
        def __init__(self):
            self.query_kwargs = None

        def query(self, **kwargs):
            self.query_kwargs = kwargs
            return SimpleNamespace(
                matches=[
                    SimpleNamespace(id="page-card:president", score=0.91),
                    SimpleNamespace(id="page-card:leadership", score=0.82),
                ]
            )

    index = Index()
    retriever = AdaptiveHybridRetriever.__new__(AdaptiveHybridRetriever)
    retriever.vector_store_provider = "pinecone"
    retriever.pinecone_query_timeout_seconds = 7.0
    retriever._pinecone_index = lambda: index
    retriever._legacy_hybrid_query_payload = (
        lambda *, query, query_vector: (query_vector, None)
    )
    scores = {}

    record_ids = retriever._dense_query_ids(
        query_vector=[0.1, 0.2],
        namespace="page-cards--release-1",
        top_k=2,
        query="Who is the president?",
        score_sink=scores,
        score_key="dense_page_cards",
    )

    assert record_ids == ["page-card:president", "page-card:leadership"]
    assert index.query_kwargs["include_metadata"] is False
    assert index.query_kwargs["include_values"] is False
    assert scores["dense_page_cards"][0] == {
        "id": "page-card:president",
        "score": 0.91,
    }
