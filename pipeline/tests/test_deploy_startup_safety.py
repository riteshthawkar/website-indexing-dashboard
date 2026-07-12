from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
START_SCRIPT = PROJECT_ROOT / "scripts" / "deploy" / "start-retriever-from-active-release.sh"
ROLLBACK_SCRIPT = PROJECT_ROOT / "scripts" / "deploy" / "rollback-active-release.sh"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


LANES = (
    "chunks",
    "parents",
    "media",
    "facts",
    "evidence_spans",
    "summaries",
    "assertions",
    "entities",
    "communities",
)
CONTRACT_FINGERPRINT = "a" * 64
SERVING_CONTRACT_FINGERPRINT = "b" * 64
EVAL_POLICY_ID = "mbzuai-production-eval-v1"
RETRIEVAL_DATASET_SHA256 = "c1c032f12c298d7c67fc7de7bed6dec46b303ffcfa843f3eab9fcea548599314"
RETRIEVAL_GATES_SHA256 = "ff5db91a3efe04719c69db892f129a23003960b98597965366a8c9c6421c708b"
ANSWER_GATES_SHA256 = "785113d699e3d96c75bdee6686d432ec2ec08f9c6351b070e4a129fb45f66dd6"
ANSWER_RUNTIME_COMMIT_SHA = "e" * 40
INDEXING_BUILD = {
    "commit_sha": "f" * 40,
    "dirty": False,
    "source": "git",
    "implementation_sha256": {"stages/example.py": "a" * 64},
}
INDEXING_BUILD_SHA256 = hashlib.sha256(
    json.dumps(INDEXING_BUILD, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_required_artifacts(work_dir: Path) -> None:
    run_id = work_dir.name
    _write_json(
        work_dir / "resolved_config.json",
        {
            "run_id": run_id,
            "project_name": "mbzuai_main",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "indexing_build": INDEXING_BUILD,
            "config": {"pipeline": {"production_profile": True}},
        },
    )
    bundle_path = (
        work_dir
        / "stage_outputs"
        / "finalize_retrieval_bundle"
        / "retrieval_bundle.json"
    )
    graph_path = (
        work_dir
        / "stage_outputs"
        / "promote_graph"
        / "promoted_knowledge_graph.json"
    )
    graph_index_path = graph_path.with_name("promoted_knowledge_graph_index.json")
    lexical_path = bundle_path.with_name("lexical_corpus.json")
    promoted_assertions_path = (
        work_dir / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
    )
    _write_json(
        bundle_path,
        {
            "version": 5,
            "chunk_records": [
                {
                    "id": "chunk-1",
                    "text": "MBZUAI admission requirements",
                    "dense_text": "MBZUAI admission requirements",
                    "document_id": "doc-1",
                    "document_title": "Admissions",
                    "source_url": "https://mbzuai.ac.ae/admissions",
                }
            ],
            "parent_records": [],
            "media_records": [],
            "fact_records": [],
            "evidence_span_records": [],
            "summary_records": [],
            "assertion_records": [],
            "entity_records": [],
            "answer_records": [],
        },
    )
    _write_json(
        lexical_path,
        [
            {
                "id": "chunk-1",
                "record_type": "chunk",
                "text": "MBZUAI admission requirements",
                "tokens": ["mbzuai", "admission", "requirements"],
            }
        ],
    )
    _write_json(
        promoted_assertions_path,
        [
            {
                "assertion_id": "assertion-1",
                "id": "assertion-1",
                "subject": "MBZUAI",
                "predicate": "HAS_TOPIC",
                "object": "admissions",
                "status": "active",
            }
        ],
    )
    _write_json(
        graph_path,
        {
            "node_count": 2,
            "edge_count": 1,
            "nodes": [{"id": "a"}, {"id": "b"}],
            "edges": [{"source": "a", "target": "b"}],
        },
    )
    graph_sha = _sha256(graph_path)
    _write_json(graph_index_path, {"source_graph_sha256": graph_sha, "a": ["b"]})
    namespaces = {lane: f"{lane}--{run_id}" for lane in LANES}
    planned = {
        key: 1
        for lane in LANES
        for key in (lane, f"sparse_{lane}")
    }
    bundle_sha = _sha256(bundle_path)
    lexical_sha = _sha256(lexical_path)
    promoted_assertions_sha = _sha256(promoted_assertions_path)
    graph_index_sha = _sha256(graph_index_path)
    upload_input = hashlib.sha256()
    upload_input.update(bundle_sha.encode("utf-8"))
    upload_input.update(lexical_sha.encode("utf-8"))
    upload_input.update(promoted_assertions_sha.encode("utf-8"))
    upload_input.update(graph_sha.encode("utf-8"))
    upload_input.update(graph_index_sha.encode("utf-8"))
    _write_json(
        work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json",
        {
            "schema_version": 4,
            "indexing_build": INDEXING_BUILD,
            "indexing_build_sha256": INDEXING_BUILD_SHA256,
            "index_name": "dense-v3",
            "sparse_index_name": "sparse-v3",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1536,
            "namespace_strategy": "release",
            "namespace_release_id": run_id,
            "namespaces": namespaces,
            "planned": planned,
            "uploaded": planned,
            "bundle_version": 5,
            "retrieval_bundle_sha256": bundle_sha,
            "lexical_corpus_sha256": lexical_sha,
            "promoted_assertions_sha256": promoted_assertions_sha,
            "knowledge_graph_kind": "promoted_local_graph",
            "knowledge_graph_sha256": graph_sha,
            "knowledge_graph_index_sha256": graph_index_sha,
            "upload_input_sha256": upload_input.hexdigest(),
            "verification": {
                family: {
                    "expected": {namespace: 1 for namespace in namespaces.values()},
                    "actual": {namespace: 1 for namespace in namespaces.values()},
                    "failures": [],
                }
                for family in ("dense", "sparse")
            },
        },
    )


def _manifest_payload(
    *,
    status: str,
    work_dir: str,
    run_id: str = "run-1",
    release_id: str = "release-1",
) -> dict:
    waived = status == "passed_with_waiver"
    work_path = Path(work_dir)
    upload = json.loads(
        (work_path / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    bundle_path = (
        work_path
        / "stage_outputs"
        / "finalize_retrieval_bundle"
        / "retrieval_bundle.json"
    )
    return {
        "schema_version": 2,
        "status": status,
        "run_id": run_id,
        "release_id": release_id,
        "work_dir": work_dir,
        "config_name": "mbzuai_production",
        "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
        "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
        "answer_runtime": {
            "commit_sha": ANSWER_RUNTIME_COMMIT_SHA,
            "pipeline_revision": "mbzuai-agentic-grounded-v1",
        },
        "indexing_build": INDEXING_BUILD,
        "promoted": True,
        "errors": [],
        "preflight": {"ok": True},
        "audit": {"ok": True},
        "evaluation": {
            "policy_id": EVAL_POLICY_ID,
            "dataset_sha256": RETRIEVAL_DATASET_SHA256,
            "gates_sha256": RETRIEVAL_GATES_SHA256,
            "minimum_query_count": 50,
            "query_count": 50,
            "gates": {"passed": True},
        },
        "answer_evaluation": {
            "policy_id": EVAL_POLICY_ID,
            "dataset_sha256": RETRIEVAL_DATASET_SHA256,
            "gates_sha256": ANSWER_GATES_SHA256,
            "minimum_query_count": 50,
            "query_count": 0 if waived else 50,
            "llm_judge": {
                "enabled": True,
                "providers": ["gemini"],
                "models": ["gemini-2.5-flash"],
                "required_provider": "gemini",
                "required_model": "gemini-2.5-flash",
                "openai_fallback_allowed": False,
                "identity_mismatch_count": 0,
                "error_count": 0,
                "judged_count": 50,
            },
            "gates": {"passed": not waived},
            "skipped": waived,
            "waived": waived,
            "waiver_reason": "approved incident waiver" if waived else "",
        },
        "vector_index": {
            "manifest_schema_version": upload["schema_version"],
            "indexing_build": INDEXING_BUILD,
            "indexing_build_sha256": INDEXING_BUILD_SHA256,
            "index_name": upload["index_name"],
            "sparse_index_name": upload["sparse_index_name"],
            "namespaces": upload["namespaces"],
            "namespace_strategy": upload["namespace_strategy"],
            "namespace_release_id": upload["namespace_release_id"],
            "retrieval_bundle_sha256": upload["retrieval_bundle_sha256"],
            "lexical_corpus_sha256": upload["lexical_corpus_sha256"],
            "promoted_assertions_sha256": upload["promoted_assertions_sha256"],
            "knowledge_graph_kind": upload["knowledge_graph_kind"],
            "knowledge_graph_sha256": upload["knowledge_graph_sha256"],
            "knowledge_graph_index_sha256": upload["knowledge_graph_index_sha256"],
            "upload_input_sha256": upload["upload_input_sha256"],
        },
        "knowledge_graph": {
            "store_backend": "local_json",
            "graph_type": "promoted_local_graph",
            "knowledge_graph_sha256": upload["knowledge_graph_sha256"],
            "knowledge_graph_index_sha256": upload["knowledge_graph_index_sha256"],
        },
        "retrieval_bundle": {
            "retrieval_bundle_sha256": _sha256(bundle_path),
            "lexical_corpus_sha256": upload["lexical_corpus_sha256"],
            "promoted_assertions_sha256": upload["promoted_assertions_sha256"],
        },
    }


def _resolve(pointer_path: Path, runs_root: Path, **extra_env: str) -> subprocess.CompletedProcess[str]:
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        if isinstance(pointer, dict) and "indexing_build_commit_sha" not in pointer:
            pointer["indexing_build_commit_sha"] = INDEXING_BUILD["commit_sha"]
            _write_json(pointer_path, pointer)
    marker_path = pointer_path.parent / ".mbzuai-release-storage"
    marker_path.write_text('{"mode":"persistent"}\n', encoding="utf-8")
    env = {
        **os.environ,
        "ACTIVE_RELEASE_FILE": str(pointer_path),
        "RELEASE_RUNS_ROOT": str(runs_root),
        "RELEASE_STORAGE_MARKER_FILE": str(marker_path),
        "RELEASE_STORAGE_MODE": "persistent",
        "RETRIEVER_RESOLVE_ONLY": "true",
        **extra_env,
    }
    return subprocess.run(
        ["bash", str(START_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_active_release_requires_matching_passed_manifest(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    _write_json(
        manifest_path,
        _manifest_payload(status="passed", work_dir=str(work_dir)),
    )
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
            "answer_runtime_commit_sha": ANSWER_RUNTIME_COMMIT_SHA,
            "active_release_manifest": str(manifest_path),
        },
    )

    result = _resolve(pointer_path, runs_root)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(work_dir.resolve())


def test_active_release_rejects_secret_bearing_resolved_config(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    resolved_path = work_dir / "resolved_config.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["config"]["graph"] = {"neo4j_password": "must-not-be-archived"}
    _write_json(resolved_path, resolved)
    _write_json(manifest_path, _manifest_payload(status="passed", work_dir=str(work_dir)))
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
            "answer_runtime_commit_sha": ANSWER_RUNTIME_COMMIT_SHA,
            "active_release_manifest": str(manifest_path),
        },
    )

    result = _resolve(pointer_path, runs_root)

    assert result.returncode != 0
    assert "secret-bearing config fields" in result.stderr
    assert "must-not-be-archived" not in result.stderr


def test_active_release_rejects_missing_or_mismatched_manifest(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    _write_json(
        manifest_path,
        {"status": "failed", "run_id": "run-1", "work_dir": str(work_dir)},
    )
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-1",
            "active_release_manifest": str(manifest_path),
        },
    )

    result = _resolve(pointer_path, runs_root)

    assert result.returncode != 0
    assert "release status must be passed" in result.stderr


def test_active_release_accepts_relocated_passed_with_waiver_manifest(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    manifest = _manifest_payload(status="passed_with_waiver", work_dir=str(work_dir))
    manifest["work_dir"] = "/source-machine/runs/run-1"
    _write_json(manifest_path, manifest)
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed_with_waiver",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
            "answer_runtime_commit_sha": ANSWER_RUNTIME_COMMIT_SHA,
            "active_release_manifest": "/source-machine/runs/run-1/release/retrieval_release_manifest.json",
        },
    )

    result = _resolve(pointer_path, runs_root, RETRIEVER_ALLOW_WAIVED_RELEASE="true")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(work_dir.resolve())


def test_active_release_rejects_hand_edited_passed_status(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    manifest = _manifest_payload(status="passed", work_dir=str(work_dir))
    manifest["errors"] = ["retrieval gate failed before status was hand-edited"]
    _write_json(manifest_path, manifest)
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
            "active_release_manifest": str(manifest_path),
        },
    )

    result = _resolve(pointer_path, runs_root)

    assert result.returncode != 0
    assert "release gate validation failed" in result.stderr
    assert "release errors must be an empty list" in result.stderr


def test_active_release_rejects_incomplete_answer_waiver(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    manifest = _manifest_payload(
        status="passed_with_waiver",
        work_dir=str(work_dir),
    )
    manifest["answer_evaluation"]["waiver_reason"] = ""
    _write_json(manifest_path, manifest)
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed_with_waiver",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
            "active_release_manifest": str(manifest_path),
        },
    )

    result = _resolve(pointer_path, runs_root)

    assert result.returncode != 0
    assert "requires a non-empty waiver reason" in result.stderr


def test_explicit_candidate_work_dir_does_not_require_promoted_pointer(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(work_dir)


def test_retriever_start_rejects_placeholder_service_token(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
        RETRIEVER_RESOLVE_ONLY="false",
        RETRIEVAL_SERVICE_TOKEN="CHANGE_ME_WITH_AT_LEAST_32_RANDOM_CHARACTERS",
        RELEASE_COMMIT_SHA="c" * 40,
    )

    assert result.returncode != 0
    assert "RETRIEVAL_SERVICE_TOKEN must not be an example placeholder" in result.stderr


def test_active_release_rejects_zero_answer_queries(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    manifest = _manifest_payload(status="passed", work_dir=str(work_dir))
    manifest["answer_evaluation"]["query_count"] = 0
    _write_json(manifest_path, manifest)
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
        },
    )

    result = _resolve(pointer_path, runs_root)

    assert result.returncode != 0
    assert "answer evaluation query_count must be at least 50" in result.stderr


def test_active_release_rejects_unpinned_evaluation_policy(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    pointer_path = tmp_path / "active_release.json"
    _make_required_artifacts(work_dir)
    manifest = _manifest_payload(status="passed", work_dir=str(work_dir))
    manifest["evaluation"]["dataset_sha256"] = "0" * 64
    _write_json(manifest_path, manifest)
    _write_json(
        pointer_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
        },
    )

    result = _resolve(pointer_path, runs_root)

    assert result.returncode != 0
    assert "retrieval evaluation dataset_sha256 does not match mbzuai-production-eval-v1" in result.stderr


def test_candidate_rejects_static_namespaces(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)
    upload_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    upload = json.loads(upload_path.read_text(encoding="utf-8"))
    upload["namespace_strategy"] = "static"
    _write_json(upload_path, upload)

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
    )

    assert result.returncode != 0
    assert "namespace_strategy=release" in result.stderr


def test_candidate_rejects_graph_hash_drift(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)
    graph_path = work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph.json"
    graph_path.write_text('{"node_count": 2, "edge_count": 1, "nodes": [], "edges": []}\n', encoding="utf-8")

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
    )

    assert result.returncode != 0
    assert "knowledge graph SHA256" in result.stderr


def test_candidate_rejects_graph_index_hash_drift(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)
    graph_index_path = (
        work_dir
        / "stage_outputs"
        / "promote_graph"
        / "promoted_knowledge_graph_index.json"
    )
    graph_index_path.write_text('{"tampered": true}\n', encoding="utf-8")

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
    )

    assert result.returncode != 0
    assert "knowledge graph index SHA256" in result.stderr


def test_candidate_rejects_graph_index_bound_to_another_graph(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)
    upload_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    upload = json.loads(upload_path.read_text(encoding="utf-8"))
    graph_index_path = (
        work_dir
        / "stage_outputs"
        / "promote_graph"
        / "promoted_knowledge_graph_index.json"
    )
    graph_index = json.loads(graph_index_path.read_text(encoding="utf-8"))
    graph_index["source_graph_sha256"] = "0" * 64
    _write_json(graph_index_path, graph_index)
    upload["knowledge_graph_index_sha256"] = _sha256(graph_index_path)
    upload_input = hashlib.sha256()
    for key in (
        "retrieval_bundle_sha256",
        "lexical_corpus_sha256",
        "promoted_assertions_sha256",
        "knowledge_graph_sha256",
        "knowledge_graph_index_sha256",
    ):
        upload_input.update(str(upload[key]).encode("utf-8"))
    upload["upload_input_sha256"] = upload_input.hexdigest()
    _write_json(upload_path, upload)

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
    )

    assert result.returncode != 0
    assert "source_graph_sha256 does not match its graph file" in result.stderr


def test_candidate_rejects_missing_lexical_runtime_sidecar(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)
    lexical_path = (
        work_dir
        / "stage_outputs"
        / "finalize_retrieval_bundle"
        / "lexical_corpus.json"
    )
    lexical_path.unlink()

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
    )

    assert result.returncode != 0
    assert "missing required lexical retrieval corpus" in result.stderr


def test_candidate_rejects_promoted_assertion_sidecar_drift(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    work_dir = runs_root / "candidate"
    _make_required_artifacts(work_dir)
    assertions_path = (
        work_dir
        / "stage_outputs"
        / "promote_assertions"
        / "promoted_assertions.json"
    )
    assertions_path.write_text('[{"id":"tampered"}]\n', encoding="utf-8")

    result = _resolve(
        tmp_path / "missing-active-release.json",
        runs_root,
        RETRIEVAL_WORK_DIR=str(work_dir),
    )

    assert result.returncode != 0
    assert "promoted assertions SHA256" in result.stderr


def test_rollback_revalidates_and_atomically_switches_pointer(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    active_path = tmp_path / "mbzuai_main" / "active_release.json"
    marker_path = tmp_path / ".mbzuai-release-storage"
    marker_path.write_text('{"mode":"persistent"}\n', encoding="utf-8")
    for run_id, release_id in (("run-1", "release-1"), ("run-2", "release-2")):
        work_dir = runs_root / run_id
        _make_required_artifacts(work_dir)
        _write_json(
            work_dir / "release" / "retrieval_release_manifest.json",
            _manifest_payload(
                status="passed",
                work_dir=str(work_dir),
                run_id=run_id,
                release_id=release_id,
            ),
        )
    _write_json(
        active_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-2",
                "release_id": "release-2",
                "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
                "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
                "indexing_build_commit_sha": INDEXING_BUILD["commit_sha"],
            },
    )
    env = {
        **os.environ,
        "ACTIVE_RELEASE_FILE": str(active_path),
        "RELEASE_RUNS_ROOT": str(runs_root),
        "RELEASE_STORAGE_MARKER_FILE": str(marker_path),
        "RELEASE_STORAGE_MODE": "persistent",
        "ROLLBACK_RUN_ID": "run-1",
    }

    result = subprocess.run(
        ["bash", str(ROLLBACK_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    pointer = json.loads(active_path.read_text(encoding="utf-8"))
    assert pointer["run_id"] == "run-1"
    history = list((active_path.parent / "rollback_history").glob("*-run-2.json"))
    assert len(history) == 1
    assert json.loads(history[0].read_text(encoding="utf-8"))["run_id"] == "run-2"
