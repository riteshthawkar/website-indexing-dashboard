from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from pipeline.core.config import (
    ProductionConfigMismatchError,
    indexing_implementation_hashes,
    load_effective_config,
    production_indexing_contract_fingerprint,
    production_serving_contract_fingerprint,
)
from pipeline.core.graph_artifacts import (
    GraphArtifactContractError,
    resolve_canonical_graph_artifacts,
)
from pipeline.core.io import atomic_write_json, combine_sha256_digests, sha256_file
from pipeline.core.knowledge_graph import (
    save_graph_bundle_with_index,
    validate_graph_index_derivation,
)
from pipeline.core.runtime_contract import (
    RuntimeArtifactContractError,
    validate_runtime_artifact_contract,
)
from pipeline.core.release_policy import (
    production_answer_judge_manifest_metadata,
    production_eval_manifest_metadata,
)
from pipeline.core.release_assembly import (
    SELECTED_DENSE_RECORD_KINDS,
    SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
    SELECTED_RELEASE_BINDING_ORDER,
    SELECTED_RELEASE_SOURCE_HASH_KEYS,
)
from pipeline.stages.embedders.gemini_pinecone_embedder import (
    _resolve_indexing_input_paths,
    _resolve_upload_namespaces,
)
from pipeline.core.base import StageContext


INDEXING_BUILD = {
    "commit_sha": "a" * 40,
    "dirty": False,
    "source": "git",
    "implementation_sha256": indexing_implementation_hashes(),
}
INDEXING_BUILD_SHA256 = hashlib.sha256(
    json.dumps(INDEXING_BUILD, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _write_graph_pair(root: Path, kind: str, marker: str) -> tuple[Path, Path]:
    locations = {
        "formatted": ("format_graph", "knowledge_graph.json", "knowledge_graph_index.json"),
        "promoted": ("promote_graph", "promoted_knowledge_graph.json", "promoted_knowledge_graph_index.json"),
        "community": ("community_graph", "community_knowledge_graph.json", "community_knowledge_graph_index.json"),
        "summarized": (
            "summarize_community_graph",
            "summarized_community_graph.json",
            "summarized_community_graph_index.json",
        ),
    }
    stage_id, graph_name, index_name = locations[kind]
    directory = root / "stage_outputs" / stage_id
    graph_file = directory / graph_name
    index_file = directory / index_name
    save_graph_bundle_with_index(
        {
            "schema_version": 2,
            "graph_type": marker,
            "nodes": [{
                "id": marker,
                "node_type": "community",
                "properties": {
                    "summary": "This is a substantive community summary used to validate release integrity."
                },
            }],
            "edges": [{"id": f"{marker}-edge", "source_id": marker, "target_id": marker, "edge_type": "RELATED_TO"}],
        },
        graph_file,
        index_file,
    )
    return graph_file, index_file


def _production_config(run_id: str) -> dict:
    return {
        "project_name": "mbzuai_main",
        "pipeline": {"production_profile": True, "require_assertion_first": True},
        "stages": [{"id": "upload_retrieval", "type": "embedder", "plugin": "gemini_pinecone"}],
        "embedder": {
            "namespace_strategy": "release",
            "namespace_release_template": "{base}--{release_id}",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1536,
            "pinecone_index": "dense-v3",
            "pinecone_sparse_index": "sparse-v3",
            **{f"namespace_{lane}": f"mbzuai-{lane}" for lane in (
                "chunks",
                "parents",
                "media",
                "facts",
                "evidence_spans",
                "summaries",
                "assertions",
                "entities",
                "communities",
            )},
        },
        "retrieval": {"retriever_backend": "routed_hybrid"},
        "graph": {"store_backend": "local_json"},
        "serving": {
            "generation_model": "gpt-5.4-2026-03-05",
            "query_rewrite_model": "gpt-5.4-mini-2026-03-17",
            "reranker_model": "gpt-5.4-mini-2026-03-17",
            "grounded_finalizer_model": "gpt-5.4-mini-2026-03-17",
        },
        "run_id": run_id,
    }


def _selected_production_config(run_id: str) -> dict:
    config = _production_config(run_id)
    config["selected_profile"] = {
        "variant_id": "c650__gemini2_1536__dense_graph",
        "record_kinds": list(SELECTED_DENSE_RECORD_KINDS),
    }
    config["vector_store"] = {
        "provider": "pgvector",
        "schema": "mbzuai_retrieval",
        "records_table": "embedding_records",
    }
    config["embedder"]["enable_sparse"] = False
    return config


def _write_current_runtime_artifacts(work_dir: Path, config: dict) -> dict:
    atomic_write_json(
        work_dir / "resolved_config.json",
        {
            "config": config,
            "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(config),
            "indexing_build": INDEXING_BUILD,
        },
    )
    bundle_file = work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    atomic_write_json(bundle_file, {"version": 5, "stats": {"chunk_count": 1}})
    lexical_file = bundle_file.with_name("lexical_corpus.json")
    promoted_assertions_file = (
        work_dir / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
    )
    atomic_write_json(lexical_file, [{"id": "lexical-1", "text": "MBZUAI"}])
    atomic_write_json(
        promoted_assertions_file,
        [{"id": "assertion-1", "text": "MBZUAI is an AI university."}],
    )
    graph_file, graph_index_file = _write_graph_pair(work_dir, "community", "community")
    bundle_sha = sha256_file(bundle_file)
    lexical_sha = sha256_file(lexical_file)
    promoted_assertions_sha = sha256_file(promoted_assertions_file)
    graph_sha = sha256_file(graph_file)
    graph_index_sha = sha256_file(graph_index_file)
    manifest = {
        "schema_version": 4,
        "indexing_build": INDEXING_BUILD,
        "indexing_build_sha256": INDEXING_BUILD_SHA256,
        "index_name": "dense-v3",
        "sparse_index_name": "sparse-v3",
        "model": "gemini-embedding-2",
        "output_dimensionality": 1536,
        "namespace_strategy": "release",
        "namespace_release_id": work_dir.name,
        "namespaces": _resolve_upload_namespaces(config["embedder"], run_id=work_dir.name),
        "retrieval_bundle_sha256": bundle_sha,
        "lexical_corpus_sha256": lexical_sha,
        "promoted_assertions_sha256": promoted_assertions_sha,
        "knowledge_graph_kind": "community_local_graph",
        "knowledge_graph_sha256": graph_sha,
        "knowledge_graph_index_sha256": graph_index_sha,
        "upload_input_sha256": combine_sha256_digests(
            bundle_sha,
            lexical_sha,
            promoted_assertions_sha,
            graph_sha,
            graph_index_sha,
        ),
    }
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        manifest,
    )
    return manifest


def _write_selected_runtime_artifacts(work_dir: Path, config: dict) -> dict:
    atomic_write_json(
        work_dir / "resolved_config.json",
        {
            "config": config,
            "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(
                config
            ),
            "indexing_build": INDEXING_BUILD,
        },
    )
    assembly_dir = work_dir / "stage_outputs" / "assemble_selected_release"
    file_payloads = {
        "selected_dense_records": (
            "selected_dense_records.jsonl",
            [{"id": f"record-{index}"} for index in range(6)],
            6,
        ),
        "chunks": ("chunk_dense_records.json", [{"id": "chunk-1"}], 1),
        "parents": (
            "parent_dense_records.json",
            [{"id": "parent-1"}, {"id": "parent-section-1"}],
            2,
        ),
        "media": ("media_dense_records.json", [{"id": "media-1"}], 1),
        "page_cards": ("page_card_dense_records.json", [{"id": "page-1"}], 1),
        "actions": ("action_dense_records.json", [{"id": "action-1"}], 1),
        "chunk_index": (
            "selected_chunk_index.json",
            {"chunk_count": 1, "chunks": [{"chunk_id": "chunk-1"}]},
            1,
        ),
        "navigation_catalog": (
            "page_graph_navigation_catalog.json",
            {
                "pages": [{"page_id": "page-1"}],
                "chunks": [{"chunk_id": "chunk-1"}],
            },
            1,
        ),
        "chunk_id_bridge": (
            "chunk_id_bridge.json",
            {"mapping_count": 1, "old_to_evaluated_chunk_id": {"old-1": "chunk-1"}},
            1,
        ),
    }
    files = {}
    for key in SELECTED_RELEASE_BINDING_ORDER:
        filename, payload, count = file_payloads[key]
        path = assembly_dir / filename
        atomic_write_json(path, payload)
        files[key] = {
            "file": filename,
            "sha256": sha256_file(path),
            "record_count": count,
        }
    source = {key: "c" * 64 for key in SELECTED_RELEASE_SOURCE_HASH_KEYS}
    assembly_binding = combine_sha256_digests(
        *[source[key] for key in SELECTED_RELEASE_SOURCE_HASH_KEYS],
        *[files[key]["sha256"] for key in SELECTED_RELEASE_BINDING_ORDER],
    )
    assembly = {
        "schema_version": SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
        "status": "ready_for_embedding",
        "variant_id": config["selected_profile"]["variant_id"],
        "record_kinds": list(SELECTED_DENSE_RECORD_KINDS),
        "record_kind_counts": {
            "chunk": 1,
            "parent": 1,
            "parent_section": 1,
            "media": 1,
            "page_card": 1,
            "action": 1,
        },
        "dense_lane_counts": {
            "chunks": 1,
            "parents": 2,
            "media": 1,
            "page_cards": 1,
            "actions": 1,
        },
        "source": source,
        "coverage": {
            "checkpoint_chunk_count": 1,
            "candidate_chunk_count": 1,
            "mapped_chunk_count": 1,
            "text_exact_match_count": 1,
            "navigation_chunk_count": 1,
            "all_candidate_chunks_mapped": True,
            "all_navigation_chunks_remapped": True,
        },
        "files": files,
        "binding_order": list(SELECTED_RELEASE_BINDING_ORDER),
        "assembly_sha256": assembly_binding,
        "embedding_performed": False,
        "upload_performed": False,
    }
    assembly_file = assembly_dir / "selected_release_assembly.json"
    atomic_write_json(assembly_file, assembly)
    assembly_manifest_sha = sha256_file(assembly_file)
    navigation_sha = files["navigation_catalog"]["sha256"]

    bundle_file = work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    atomic_write_json(
        bundle_file,
        {
            "version": 6,
            "schema_version": "mbzuai.retrieval_bundle.v6",
            "selected_release_contract": {
                "schema_version": SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
                "variant_id": config["selected_profile"]["variant_id"],
                "manifest_sha256": assembly_manifest_sha,
                "assembly_sha256": assembly_binding,
                "candidate_records_sha256": source["candidate_records_sha256"],
                "navigation_catalog_sha256": navigation_sha,
            },
            "stats": {
                "chunk_count": 1,
                "parent_count": 2,
                "media_count": 1,
                "page_card_count": 1,
                "action_count": 1,
            },
        },
    )
    lexical_file = bundle_file.with_name("lexical_corpus.json")
    promoted_assertions_file = (
        work_dir / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
    )
    atomic_write_json(lexical_file, [{"id": "lexical-1", "text": "MBZUAI"}])
    atomic_write_json(
        promoted_assertions_file,
        [{"id": "assertion-1", "text": "MBZUAI is an AI university."}],
    )
    graph_file, graph_index_file = _write_graph_pair(work_dir, "community", "community")
    bundle_sha = sha256_file(bundle_file)
    lexical_sha = sha256_file(lexical_file)
    promoted_assertions_sha = sha256_file(promoted_assertions_file)
    graph_sha = sha256_file(graph_file)
    graph_index_sha = sha256_file(graph_index_file)
    uploaded = {
        "chunks": 1,
        "parents": 2,
        "media": 1,
        "page_cards": 1,
        "actions": 1,
        "facts": 0,
        "evidence_spans": 0,
        "summaries": 0,
        "assertions": 0,
        "entities": 0,
        "communities": 0,
    }
    manifest = {
        "schema_version": 6,
        "provider": "pgvector",
        "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(
            config
        ),
        "index_name": "mbzuai_retrieval.embedding_records",
        "sparse_index_name": "",
        "model": "gemini-embedding-2",
        "output_dimensionality": 1536,
        "namespace_strategy": "release",
        "namespace_release_id": work_dir.name,
        "namespaces": _resolve_upload_namespaces(config["embedder"], run_id=work_dir.name),
        "uploaded": uploaded,
        "bundle_version": 6,
        "retrieval_bundle_sha256": bundle_sha,
        "lexical_corpus_sha256": lexical_sha,
        "promoted_assertions_sha256": promoted_assertions_sha,
        "knowledge_graph_kind": "community_local_graph",
        "knowledge_graph_sha256": graph_sha,
        "knowledge_graph_index_sha256": graph_index_sha,
        "selected_release_assembly_file": str(assembly_file),
        "selected_release_assembly_sha256": assembly_manifest_sha,
        "selected_release_binding_sha256": assembly_binding,
        "page_graph_navigation_catalog_sha256": navigation_sha,
        "upload_input_sha256": combine_sha256_digests(
            bundle_sha,
            lexical_sha,
            promoted_assertions_sha,
            graph_sha,
            graph_index_sha,
            assembly_manifest_sha,
            assembly_binding,
            navigation_sha,
        ),
    }
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        manifest,
    )
    return {"manifest": manifest, "assembly_file": assembly_file, "files": files}


def test_canonical_graph_selection_prefers_community_and_rejects_partial_latest(tmp_path: Path) -> None:
    promoted_graph, _ = _write_graph_pair(tmp_path, "promoted", "promoted")
    community_graph, community_index = _write_graph_pair(tmp_path, "community", "community")

    selected = resolve_canonical_graph_artifacts(tmp_path)

    assert selected is not None
    assert selected.graph_file == community_graph
    assert selected.kind == "community_local_graph"
    community_index.unlink()
    with pytest.raises(GraphArtifactContractError, match="without its matching index"):
        resolve_canonical_graph_artifacts(tmp_path)
    assert promoted_graph.is_file()


def test_canonical_graph_selection_prefers_stage_owned_summarized_graph(
    tmp_path: Path,
) -> None:
    community_graph, _ = _write_graph_pair(tmp_path, "community", "community")
    summarized_graph, summarized_index = _write_graph_pair(
        tmp_path,
        "summarized",
        "summarized",
    )

    selected = resolve_canonical_graph_artifacts(tmp_path)

    assert selected is not None
    assert selected.graph_file == summarized_graph
    assert selected.kind == "summarized_community_local_graph"
    summarized_index.unlink()
    with pytest.raises(GraphArtifactContractError, match="without its matching index"):
        resolve_canonical_graph_artifacts(tmp_path)
    assert community_graph.is_file()


def test_canonical_graph_selection_rejects_index_from_a_different_graph(tmp_path: Path) -> None:
    graph_file, index_file = _write_graph_pair(tmp_path, "community", "community")
    foreign_root = tmp_path / "foreign"
    _, foreign_index = _write_graph_pair(foreign_root, "community", "foreign")
    index_file.write_bytes(foreign_index.read_bytes())

    issues = validate_graph_index_derivation(graph_file, index_file)
    assert {issue["code"] for issue in issues} >= {
        "graph_index_source_hash_mismatch",
        "graph_index_derivation_mismatch",
    }
    with pytest.raises(GraphArtifactContractError, match="derivation contract failed"):
        resolve_canonical_graph_artifacts(tmp_path)


def test_pinecone_upload_resolves_the_same_final_graph_as_runtime(tmp_path: Path) -> None:
    promoted_graph, _ = _write_graph_pair(tmp_path, "promoted", "promoted")
    community_graph, _ = _write_graph_pair(tmp_path, "community", "community")
    ctx = StageContext(
        run_id="candidate",
        project_name="mbzuai",
        config={},
        work_dir=tmp_path,
        previous_outputs={"promoted_knowledge_graph_file": str(promoted_graph)},
        stage_definition={"type": "embedder", "plugin": "gemini_pinecone"},
        stage_id="upload_retrieval",
    )

    resolved = _resolve_indexing_input_paths(ctx)

    assert resolved["graph_bundle"] == str(community_graph)


def test_explicit_production_config_wins_runtime_settings_but_rejects_indexing_drift(tmp_path: Path) -> None:
    config_path = tmp_path / "mbzuai_production.yaml"
    requested = _production_config("run-1")
    requested["retrieval"]["query_planner_enabled"] = True
    config_path.write_text(yaml.safe_dump(requested), encoding="utf-8")
    work_dir = tmp_path / "run-1"
    snapshot = _production_config("run-1")
    snapshot["retrieval"]["query_planner_enabled"] = False
    atomic_write_json(
        work_dir / "resolved_config.json",
        {
            "config": snapshot,
            "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(snapshot),
        },
    )

    loaded = load_effective_config(str(config_path), work_dir=work_dir)

    assert loaded["retrieval"]["query_planner_enabled"] is True
    snapshot["stages"] = [{"id": "legacy_upload_first", "plugin": "gemini_pinecone"}]
    atomic_write_json(work_dir / "resolved_config.json", {"config": snapshot})
    with pytest.raises(ProductionConfigMismatchError, match="does not match"):
        load_effective_config(str(config_path), work_dir=work_dir)


def test_current_production_runtime_validates_release_namespaces_bundle_and_graph(tmp_path: Path) -> None:
    work_dir = tmp_path / "candidate-v3"
    config = _production_config(work_dir.name)
    _write_current_runtime_artifacts(work_dir, config)

    report = validate_runtime_artifact_contract(config, work_dir)

    assert report["validated"] is True
    assert report["namespace_strategy"] == "release"
    assert report["knowledge_graph_kind"] == "community_local_graph"


def test_selected_runtime_validates_full_assembly_and_rejects_lane_tampering(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "selected-candidate-v1"
    config = _selected_production_config(work_dir.name)
    artifacts = _write_selected_runtime_artifacts(work_dir, config)

    report = validate_runtime_artifact_contract(config, work_dir)

    assert report["validated"] is True
    assert report["vector_store_provider"] == "pgvector"
    assert report["selected_release_assembly_sha256"] == artifacts["manifest"][
        "selected_release_assembly_sha256"
    ]
    assert report["selected_release_binding_sha256"] == artifacts["manifest"][
        "selected_release_binding_sha256"
    ]

    page_cards_file = artifacts["assembly_file"].parent / artifacts["files"]["page_cards"][
        "file"
    ]
    atomic_write_json(page_cards_file, [{"id": "tampered-page"}])
    with pytest.raises(RuntimeArtifactContractError, match="assembly file page_cards digest mismatch"):
        validate_runtime_artifact_contract(config, work_dir)


def test_current_production_runtime_rejects_static_namespaces_and_graph_drift(tmp_path: Path) -> None:
    work_dir = tmp_path / "candidate-v3"
    config = _production_config(work_dir.name)
    manifest = _write_current_runtime_artifacts(work_dir, config)
    manifest["namespace_strategy"] = "static"
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        manifest,
    )
    with pytest.raises(RuntimeArtifactContractError, match="namespace strategy"):
        validate_runtime_artifact_contract(config, work_dir)

    manifest["namespace_strategy"] = "release"
    atomic_write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        manifest,
    )
    community_graph = work_dir / "stage_outputs" / "community_graph" / "community_knowledge_graph.json"
    atomic_write_json(community_graph, {"nodes": [{"id": "tampered"}], "edges": []})
    with pytest.raises(RuntimeArtifactContractError, match="derivation contract"):
        validate_runtime_artifact_contract(config, work_dir)


def test_current_release_promotion_revalidates_snapshot_vector_bundle_and_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "RETRIEVAL_SERVICE_TOKEN",
        "Rtrv_7zQ9-aB3mN8.xK2pL6:sD4wF1cV5",
    )
    monkeypatch.setenv(
        "CANDIDATE_BACKEND_OPERATIONS_TOKEN",
        "Ops_7zQ9-aB3mN8.xK2pL6:sD4wF1cV5",
    )
    from pipeline.core.release import promote_release_manifest

    work_dir = tmp_path / "candidate-v3"
    config = _production_config(work_dir.name)
    upload_manifest = _write_current_runtime_artifacts(work_dir, config)
    graph = resolve_canonical_graph_artifacts(work_dir)
    assert graph is not None
    bundle_file = work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    vector_manifest_file = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    release_manifest_file = work_dir / "release" / "retrieval_release_manifest.json"
    release_manifest = {
        "schema_version": 2,
        "status": "passed",
        "config_name": "synthetic-production",
        "run_id": work_dir.name,
        "release_id": "release-v3",
        "answer_runtime": {
            "commit_sha": "a" * 40,
            "pipeline_revision": "synthetic-answer-v1",
        },
        "indexing_build": INDEXING_BUILD,
        "work_dir": str(work_dir),
        "production_indexing_contract_fingerprint": production_indexing_contract_fingerprint(config),
        "production_serving_contract_fingerprint": production_serving_contract_fingerprint(config),
        "errors": [],
        "preflight": {"ok": True},
        "audit": {"ok": True},
        "evaluation": {
            **production_eval_manifest_metadata(answer=False),
            "query_count": 65,
            "gates": {"passed": True},
        },
        "answer_evaluation": {
            **production_eval_manifest_metadata(answer=True),
            "query_count": 65,
            "llm_judge": production_answer_judge_manifest_metadata(),
            "gates": {"passed": True},
            "skipped": False,
            "waived": False,
        },
        "vector_index": {
            "manifest_file": str(vector_manifest_file),
            "manifest_schema_version": 4,
            "indexing_build": INDEXING_BUILD,
            "indexing_build_sha256": INDEXING_BUILD_SHA256,
            "index_name": upload_manifest["index_name"],
            "sparse_index_name": upload_manifest["sparse_index_name"],
            "namespaces": upload_manifest["namespaces"],
            "namespace_strategy": upload_manifest["namespace_strategy"],
            "namespace_release_id": upload_manifest["namespace_release_id"],
            "retrieval_bundle_sha256": upload_manifest["retrieval_bundle_sha256"],
            "lexical_corpus_sha256": upload_manifest["lexical_corpus_sha256"],
            "promoted_assertions_sha256": upload_manifest["promoted_assertions_sha256"],
            "knowledge_graph_kind": upload_manifest["knowledge_graph_kind"],
            "knowledge_graph_sha256": upload_manifest["knowledge_graph_sha256"],
            "knowledge_graph_index_sha256": upload_manifest["knowledge_graph_index_sha256"],
            "upload_input_sha256": upload_manifest["upload_input_sha256"],
        },
        "retrieval_bundle": {
            "retrieval_bundle_file": str(bundle_file),
            "retrieval_bundle_sha256": upload_manifest["retrieval_bundle_sha256"],
            "lexical_corpus_file": str(bundle_file.with_name("lexical_corpus.json")),
            "lexical_corpus_sha256": upload_manifest["lexical_corpus_sha256"],
            "promoted_assertions_file": str(
                work_dir / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
            ),
            "promoted_assertions_sha256": upload_manifest["promoted_assertions_sha256"],
        },
        "knowledge_graph": {
            "store_backend": "local_json",
            "manifest_file": str(graph.graph_file),
            "index_file": str(graph.index_file),
            "graph_type": graph.kind,
            "knowledge_graph_sha256": graph.graph_sha256,
            "knowledge_graph_index_sha256": graph.index_sha256,
        },
    }
    atomic_write_json(release_manifest_file, release_manifest)
    stable_manifest_sha = hashlib.sha256(
        json.dumps(
            release_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    promotion_attestation = {
        "schema_version": 1,
        "attested_at": datetime.now(timezone.utc).isoformat(),
        "release_manifest_sha256": stable_manifest_sha,
        "release_id": release_manifest["release_id"],
        "config_name": release_manifest["config_name"],
        "run_id": release_manifest["run_id"],
        "backend_commit_sha": release_manifest["answer_runtime"]["commit_sha"],
        "retriever_commit_sha": "b" * 40,
        "indexing_build_commit_sha": INDEXING_BUILD["commit_sha"],
        "retrieval_bundle_sha256": upload_manifest["retrieval_bundle_sha256"],
        "knowledge_graph_sha256": upload_manifest["knowledge_graph_sha256"],
        "knowledge_graph_index_sha256": upload_manifest[
            "knowledge_graph_index_sha256"
        ],
        "lexical_corpus_sha256": upload_manifest["lexical_corpus_sha256"],
        "promoted_assertions_sha256": upload_manifest[
            "promoted_assertions_sha256"
        ],
        "answer_models": dict(config["serving"]),
    }
    from pipeline.core.release import _promotion_attestation_signature

    promotion_attestation["signature_sha256"] = _promotion_attestation_signature(
        promotion_attestation
    )

    active_file = promote_release_manifest(
        manifest_path=release_manifest_file,
        active_release_file=tmp_path / "active_release.json",
        promotion_attestation=promotion_attestation,
    )

    assert active_file.is_file()
    promoted_manifest = json.loads(release_manifest_file.read_text(encoding="utf-8"))
    active_pointer = json.loads(active_file.read_text(encoding="utf-8"))
    assert promoted_manifest["promotion_attestation"] == promotion_attestation
    assert promoted_manifest["promotion_attestation_sha256"] == active_pointer[
        "promotion_attestation_sha256"
    ]
    validate_runtime_artifact_contract(config, work_dir)
    changed_serving_config = {
        **config,
        "retrieval": {
            **config["retrieval"],
            "query_planner_model": "different-model",
        },
    }
    with pytest.raises(RuntimeArtifactContractError, match="serving contract"):
        validate_runtime_artifact_contract(changed_serving_config, work_dir)

    upload_manifest["knowledge_graph_sha256"] = "tampered"
    atomic_write_json(vector_manifest_file, upload_manifest)
    release_manifest["promoted"] = False
    atomic_write_json(release_manifest_file, release_manifest)
    with pytest.raises(ValueError, match="does not match its upload manifest"):
        promote_release_manifest(
            manifest_path=release_manifest_file,
            active_release_file=tmp_path / "active_release-2.json",
        )


def test_canonical_production_stage_order_covers_before_processing_and_uploads_last() -> None:
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    config = load_config("mbzuai_production")
    stage_ids = [stage["id"] for stage in config["stages"]]

    assert stage_ids[:2] == ["verify_selected_profile", "assemble_selected_release"]
    assert "crawl_web" not in stage_ids
    assert "chunk_content" not in stage_ids
    assert config["selected_profile"]["pre_embedding_only"] is False
    assert config["selected_profile"]["record_kinds"] == [
        "chunk",
        "parent",
        "parent_section",
        "media",
        "page_card",
        "action",
    ]
    assert stage_ids.index("format_graph") < stage_ids.index("promote_graph")
    assert stage_ids.index("promote_graph") < stage_ids.index("community_graph")
    assert stage_ids.index("summarize_community_graph") < stage_ids.index("upload_retrieval")
    assert stage_ids[-1] == "upload_retrieval"
    assert config["crawler"]["ignore_https_errors"] is False
    assert config["crawler"]["require_https"] is True
    report = assess_production_readiness(config, config_name="mbzuai_production", validation_errors={})
    stage_order = next(check for check in report["checks"] if check["name"] == "stage_order")
    crawler_tls = next(
        check for check in report["checks"] if check["name"] == "crawler_tls_verification"
    )
    assert stage_order["status"] == "ok"
    assert crawler_tls["status"] == "ok"


def test_production_preflight_rejects_disabled_crawler_tls_verification() -> None:
    from pipeline.core.config import load_config
    from pipeline.core.preflight import assess_production_readiness

    config = load_config("mbzuai_production")
    config["crawler"]["ignore_https_errors"] = True

    report = assess_production_readiness(
        config,
        config_name="mbzuai_production",
        validation_errors={},
    )
    crawler_tls = next(
        check for check in report["checks"] if check["name"] == "crawler_tls_verification"
    )

    assert report["ok"] is False
    assert crawler_tls["status"] == "error"


def test_production_routed_retriever_refuses_vector_only_graph_downgrade(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import pipeline.retrieval.routed_hybrid as routed_module

    class FakeVector:
        model = "gemini-embedding-2"
        output_dimensionality = 1536
        evidence_span_map = {}
        chunk_map = {}
        summary_map = {}
        parent_map = {}

        def __init__(self, **_kwargs):
            pass

    class BrokenGraph:
        def __init__(self, **_kwargs):
            raise ValueError("invalid graph")

    monkeypatch.setattr(routed_module, "AdaptiveHybridRetriever", FakeVector)
    monkeypatch.setattr(routed_module, "GraphRAGRetriever", BrokenGraph)
    config = {
        "pipeline": {"production_profile": True},
        "retrieval": {"retriever_backend": "routed_hybrid", "routed_graph_enabled": True},
    }

    with pytest.raises(ValueError, match="refusing vector-only downgrade"):
        routed_module.RoutedHybridRetriever(config=config, work_dir=tmp_path)
