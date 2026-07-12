"""Fail-closed retrieval runtime validation for promoted artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Mapping

from pipeline.core.config import (
    production_indexing_contract_fingerprint,
    production_serving_contract_fingerprint,
)
from pipeline.core.graph_artifacts import (
    GraphArtifactContractError,
    resolve_canonical_graph_artifacts,
)
from pipeline.core.io import combine_sha256_digests, load_json_safe, sha256_file
from pipeline.core.knowledge_graph import (
    community_summary_quality,
    load_graph_bundle,
    validate_graph_bundle,
)
from pipeline.core.release_policy import validate_production_eval_manifest


class RuntimeArtifactContractError(ValueError):
    """Raised when retrieval artifacts do not describe one immutable release."""


_NAMESPACE_LANES = (
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


def _bundle_path(work_dir: Path) -> Path:
    candidates = (
        work_dir / "stage_outputs" / "finalize_retrieval_bundle" / "retrieval_bundle.json",
        work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
        work_dir / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _graph_backend(config: Mapping[str, Any]) -> str:
    graph_cfg = config.get("graph") if isinstance(config.get("graph"), Mapping) else {}
    configured = str(
        graph_cfg.get("store_backend")
        or graph_cfg.get("graph_store_backend")
        or "local_json"
    ).strip().lower()
    if configured in {"none", "off", "disabled"}:
        return "disabled"
    if configured == "neo4j":
        return "neo4j"
    return "local_json"


def _validate_release_manifest(
    *,
    manifest: Mapping[str, Any],
    config: Mapping[str, Any],
    work_dir: Path,
    upload_manifest: Mapping[str, Any],
    graph_sha256: str,
    graph_index_sha256: str,
    bundle_sha256: str,
    lexical_corpus_sha256: str,
    promoted_assertions_sha256: str,
    errors: list[str],
) -> None:
    status = str(manifest.get("status") or "").strip()
    if status not in {"passed", "passed_with_waiver"}:
        errors.append(f"release manifest status is not promotable: {status or '<missing>'}")
    if str(manifest.get("run_id") or "").strip() != work_dir.name:
        errors.append("release manifest run_id does not match the runtime work directory")

    evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), Mapping) else {}
    answer_evaluation = (
        manifest.get("answer_evaluation")
        if isinstance(manifest.get("answer_evaluation"), Mapping)
        else {}
    )
    errors.extend(
        validate_production_eval_manifest(
            evaluation,
            answer_evaluation,
            allow_answer_waiver=str(manifest.get("status") or "") == "passed_with_waiver",
        )
    )
    answer_runtime = (
        manifest.get("answer_runtime")
        if isinstance(manifest.get("answer_runtime"), Mapping)
        else {}
    )
    if not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})",
        str(answer_runtime.get("commit_sha") or "").strip().lower(),
    ):
        errors.append("release manifest answer runtime commit SHA is missing or invalid")
    serving_config = config.get("serving") if isinstance(config.get("serving"), Mapping) else {}
    expected_answer_revision = str(
        serving_config.get("answer_pipeline_revision") or ""
    ).strip()
    recorded_answer_revision = str(answer_runtime.get("pipeline_revision") or "").strip()
    if expected_answer_revision and recorded_answer_revision != expected_answer_revision:
        errors.append("release manifest answer pipeline revision does not match runtime config")
    elif not recorded_answer_revision:
        errors.append("release manifest answer pipeline revision is missing")

    recorded_contract = str(manifest.get("production_indexing_contract_fingerprint") or "").strip()
    expected_contract = production_indexing_contract_fingerprint(dict(config))
    if not recorded_contract:
        errors.append("release manifest is missing production_indexing_contract_fingerprint")
    elif recorded_contract != expected_contract:
        errors.append("release manifest production indexing fingerprint does not match runtime config")

    recorded_serving_contract = str(
        manifest.get("production_serving_contract_fingerprint") or ""
    ).strip()
    expected_serving_contract = production_serving_contract_fingerprint(dict(config))
    if not recorded_serving_contract:
        errors.append("release manifest is missing production_serving_contract_fingerprint")
    elif recorded_serving_contract != expected_serving_contract:
        errors.append(
            "release manifest serving contract does not match the evaluated runtime behavior"
        )

    vector = manifest.get("vector_index") if isinstance(manifest.get("vector_index"), Mapping) else {}
    if str(vector.get("index_name") or "") != str(upload_manifest.get("index_name") or ""):
        errors.append("release manifest dense index does not match the upload manifest")
    if str(vector.get("sparse_index_name") or "") != str(upload_manifest.get("sparse_index_name") or ""):
        errors.append("release manifest sparse index does not match the upload manifest")
    if dict(vector.get("namespaces") or {}) != dict(upload_manifest.get("namespaces") or {}):
        errors.append("release manifest namespaces do not match the upload manifest")
    if str(vector.get("retrieval_bundle_sha256") or "") != bundle_sha256:
        errors.append("release manifest retrieval bundle hash does not match runtime")
    if str(vector.get("lexical_corpus_sha256") or "") != lexical_corpus_sha256:
        errors.append("release manifest lexical corpus hash does not match runtime")
    if str(vector.get("promoted_assertions_sha256") or "") != promoted_assertions_sha256:
        errors.append("release manifest promoted assertions hash does not match runtime")
    if str(vector.get("knowledge_graph_sha256") or "") != graph_sha256:
        errors.append("release manifest graph hash does not match runtime")
    if graph_index_sha256 and str(vector.get("knowledge_graph_index_sha256") or "") != graph_index_sha256:
        errors.append("release manifest graph-index hash does not match runtime")


def validate_runtime_artifact_contract(
    config: Dict[str, Any],
    work_dir: str | Path,
) -> Dict[str, Any]:
    """Validate the upload, bundle, graph, namespaces, and release manifest.

    Current production profiles require schema-v4, release-scoped namespaces,
    and exact bundle/graph hashes.  Static namespaces remain supported for
    explicit non-production and legacy configurations.
    """

    root = Path(work_dir).expanduser().resolve()
    pipeline_cfg = config.get("pipeline") if isinstance(config.get("pipeline"), Mapping) else {}
    production = bool(pipeline_cfg.get("production_profile", False))
    embedder_cfg = config.get("embedder") if isinstance(config.get("embedder"), Mapping) else {}
    manifest_path = root / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    manifest = load_json_safe(manifest_path, None)
    errors: list[str] = []

    if production:
        snapshot_path = root / "resolved_config.json"
        snapshot = load_json_safe(snapshot_path, None)
        snapshot_config = snapshot.get("config") if isinstance(snapshot, Mapping) else None
        if not isinstance(snapshot_config, dict):
            errors.append(f"production resolved_config.json is missing or invalid: {snapshot_path}")
        else:
            snapshot_fingerprint = production_indexing_contract_fingerprint(snapshot_config)
            recorded_fingerprint = str(
                snapshot.get("production_indexing_contract_fingerprint") or ""
            ).strip()
            if not recorded_fingerprint:
                errors.append("production resolved_config.json is missing its indexing contract fingerprint")
            elif recorded_fingerprint != snapshot_fingerprint:
                errors.append("production resolved_config.json fingerprint does not match its config")
            if snapshot_fingerprint != production_indexing_contract_fingerprint(config):
                errors.append("production runtime config does not match the indexed run snapshot")

    if not isinstance(manifest, dict):
        if production:
            errors.append(f"current Pinecone upload manifest is missing or invalid: {manifest_path}")
        if errors:
            raise RuntimeArtifactContractError("Runtime artifact contract failed: " + "; ".join(errors))
        return {"validated": False, "legacy": True}

    schema_version = int(manifest.get("schema_version") or 0)
    strict_contract = production or schema_version >= 3
    if production and schema_version < 4:
        errors.append("production runtime requires vector manifest schema_version 4 or newer")
    if strict_contract:
        configured_dense_index = str(embedder_cfg.get("pinecone_index") or "").strip()
        configured_sparse_index = str(embedder_cfg.get("pinecone_sparse_index") or "").strip()
        if str(manifest.get("index_name") or "").strip() != configured_dense_index:
            errors.append("upload manifest dense index does not match runtime config")
        if bool(embedder_cfg.get("enable_sparse", True)) and str(
            manifest.get("sparse_index_name") or ""
        ).strip() != configured_sparse_index:
            errors.append("upload manifest sparse index does not match runtime config")
        configured_model = str(embedder_cfg.get("model") or "").strip()
        uploaded_model = str(manifest.get("model") or "").strip()
        if configured_model and uploaded_model != configured_model:
            errors.append("upload manifest embedding model does not match runtime config")
        configured_dimension = int(embedder_cfg.get("output_dimensionality") or 0)
        uploaded_dimension = int(manifest.get("output_dimensionality") or 0)
        if configured_dimension and uploaded_dimension != configured_dimension:
            errors.append("upload manifest embedding dimension does not match runtime config")

    configured_strategy = str(embedder_cfg.get("namespace_strategy") or "static").strip().lower()
    manifest_strategy = str(manifest.get("namespace_strategy") or "").strip().lower()
    if production and configured_strategy != "release":
        errors.append("production runtime config must use release-scoped namespaces")
    if strict_contract and manifest_strategy not in {"static", "release"}:
        errors.append("upload manifest namespace_strategy must be static or release")
    elif manifest_strategy in {"static", "release"} and manifest_strategy != configured_strategy:
        errors.append(
            "upload manifest namespace strategy does not match runtime config: "
            f"uploaded={manifest_strategy}, configured={configured_strategy}"
        )

    manifest_namespaces = manifest.get("namespaces") if isinstance(manifest.get("namespaces"), Mapping) else {}
    missing_namespaces = [lane for lane in _NAMESPACE_LANES if not str(manifest_namespaces.get(lane) or "").strip()]
    if strict_contract and missing_namespaces:
        errors.append(f"upload manifest is missing namespaces: {missing_namespaces}")
    populated = [str(manifest_namespaces.get(lane) or "").strip() for lane in _NAMESPACE_LANES]
    populated = [value for value in populated if value]
    if strict_contract and len(populated) != len(set(populated)):
        errors.append("upload manifest namespaces are not distinct")

    if manifest_strategy == "release":
        release_id = str(manifest.get("namespace_release_id") or "").strip()
        if release_id != root.name:
            errors.append(
                "release-scoped upload manifest does not belong to this work directory: "
                f"manifest={release_id or '<missing>'}, work_dir={root.name}"
            )
        else:
            from pipeline.stages.embedders.gemini_pinecone_embedder import _resolve_upload_namespaces

            try:
                expected_namespaces = _resolve_upload_namespaces(dict(embedder_cfg), run_id=release_id)
            except (KeyError, ValueError) as exc:
                errors.append(f"runtime could not resolve release namespaces from config: {exc}")
            else:
                if dict(manifest_namespaces) != expected_namespaces:
                    errors.append("upload manifest namespaces do not match the release-scoped runtime contract")
    elif manifest_strategy == "static" and strict_contract:
        expected_static = {
            lane: str(embedder_cfg.get(f"namespace_{lane}") or lane).strip()
            for lane in _NAMESPACE_LANES
        }
        if dict(manifest_namespaces) != expected_static:
            errors.append("upload manifest namespaces do not match the static runtime contract")

    bundle_path = _bundle_path(root)
    bundle_sha256 = sha256_file(bundle_path) if bundle_path.is_file() else ""
    uploaded_bundle_sha256 = str(manifest.get("retrieval_bundle_sha256") or "").strip()
    if production and not uploaded_bundle_sha256:
        errors.append("upload manifest is missing retrieval_bundle_sha256")
    elif uploaded_bundle_sha256 and uploaded_bundle_sha256 != bundle_sha256:
        errors.append("runtime retrieval bundle hash does not match the uploaded bundle")

    lexical_corpus_path = bundle_path.with_name("lexical_corpus.json")
    lexical_corpus = load_json_safe(lexical_corpus_path, None)
    lexical_corpus_sha256 = sha256_file(lexical_corpus_path) if lexical_corpus_path.is_file() else ""
    promoted_assertions_path = (
        root / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
    )
    promoted_assertions = load_json_safe(promoted_assertions_path, None)
    promoted_assertions_sha256 = (
        sha256_file(promoted_assertions_path) if promoted_assertions_path.is_file() else ""
    )
    current_contract = production or schema_version >= 4
    if current_contract:
        if not isinstance(lexical_corpus, list) or not lexical_corpus:
            errors.append(f"runtime lexical corpus is missing, invalid, or empty: {lexical_corpus_path}")
        if str(manifest.get("lexical_corpus_sha256") or "").strip() != lexical_corpus_sha256:
            errors.append("runtime lexical corpus hash does not match the upload manifest")
        if not isinstance(promoted_assertions, list) or not promoted_assertions:
            errors.append(
                "runtime promoted assertions sidecar is missing, invalid, or empty: "
                f"{promoted_assertions_path}"
            )
        if str(manifest.get("promoted_assertions_sha256") or "").strip() != promoted_assertions_sha256:
            errors.append("runtime promoted assertions hash does not match the upload manifest")

    graph_sha256 = ""
    graph_index_sha256 = ""
    graph_kind = ""
    backend = _graph_backend(config)
    retrieval_cfg = config.get("retrieval") if isinstance(config.get("retrieval"), Mapping) else {}
    retriever_backend = str(retrieval_cfg.get("retriever_backend") or "vector").strip().lower()
    graph_required = production or retriever_backend in {"graph_hybrid", "routed_hybrid"}
    if backend != "disabled" and graph_required:
        try:
            graph = resolve_canonical_graph_artifacts(
                root,
                required=True,
                require_index=True,
                validate_binding=current_contract,
            )
        except GraphArtifactContractError as exc:
            errors.append(str(exc))
        else:
            assert graph is not None
            graph_sha256 = graph.graph_sha256
            graph_index_sha256 = graph.index_sha256
            graph_kind = graph.kind
            graph_bundle = load_graph_bundle(graph.graph_file)
            graph_issues = validate_graph_bundle(graph_bundle)
            if graph_issues:
                errors.append(
                    "runtime knowledge graph is invalid: "
                    + str(graph_issues[0].get("message") or graph_issues[0].get("code") or "unknown error")
                )
            if production:
                graph_cfg = config.get("graph") if isinstance(config.get("graph"), Mapping) else {}
                min_summary_characters = int(
                    graph_cfg.get("community_summary_min_characters", 40) or 40
                )
                min_summary_coverage = float(
                    graph_cfg.get("community_summary_min_coverage_ratio", 1.0)
                )
                summary_quality = community_summary_quality(
                    graph_bundle,
                    min_characters=min_summary_characters,
                )
                if int(summary_quality["total_communities"]) <= 0:
                    errors.append("runtime knowledge graph contains no community nodes")
                elif float(summary_quality["coverage_ratio"]) < min_summary_coverage:
                    errors.append(
                        "runtime community-summary coverage is below the production threshold: "
                        f"actual={float(summary_quality['coverage_ratio']):.4f}, "
                        f"required={min_summary_coverage:.4f}"
                    )
            uploaded_graph_sha = str(manifest.get("knowledge_graph_sha256") or "").strip()
            if (strict_contract or uploaded_graph_sha) and uploaded_graph_sha != graph_sha256:
                errors.append("runtime graph hash does not match the graph used for vector upload")
            if production and str(manifest.get("knowledge_graph_index_sha256") or "").strip() != graph_index_sha256:
                errors.append("runtime graph-index hash does not match the graph index used for vector upload")
            if production and str(manifest.get("knowledge_graph_kind") or "").strip() != graph_kind:
                errors.append("runtime graph kind does not match the graph used for vector upload")

    uploaded_graph_sha256 = str(manifest.get("knowledge_graph_sha256") or "").strip()
    upload_input_sha256 = str(manifest.get("upload_input_sha256") or "").strip()
    if uploaded_bundle_sha256 and uploaded_graph_sha256:
        if current_contract:
            expected_upload_input_sha256 = combine_sha256_digests(
                uploaded_bundle_sha256,
                lexical_corpus_sha256,
                promoted_assertions_sha256,
                uploaded_graph_sha256,
                graph_index_sha256,
            )
        else:
            expected_upload_input_sha256 = combine_sha256_digests(
                uploaded_bundle_sha256,
                uploaded_graph_sha256,
            )
        if upload_input_sha256 != expected_upload_input_sha256:
            errors.append(
                "upload_input_sha256 does not bind the runtime bundle, sidecars, graph, and graph index"
            )

    release_manifest_path = root / "release" / "retrieval_release_manifest.json"
    release_manifest = load_json_safe(release_manifest_path, None)
    if release_manifest is not None:
        if not isinstance(release_manifest, dict):
            errors.append(f"release manifest is invalid: {release_manifest_path}")
        elif production or int(release_manifest.get("schema_version") or 0) >= 2:
            _validate_release_manifest(
                manifest=release_manifest,
                config=config,
                work_dir=root,
                upload_manifest=manifest,
                graph_sha256=graph_sha256,
                graph_index_sha256=graph_index_sha256,
                bundle_sha256=bundle_sha256,
                lexical_corpus_sha256=lexical_corpus_sha256,
                promoted_assertions_sha256=promoted_assertions_sha256,
                errors=errors,
            )

    if errors:
        raise RuntimeArtifactContractError("Runtime artifact contract failed: " + "; ".join(errors))
    return {
        "validated": True,
        "production": production,
        "vector_manifest_schema_version": schema_version,
        "namespace_strategy": manifest_strategy,
        "namespace_release_id": str(manifest.get("namespace_release_id") or ""),
        "retrieval_bundle_sha256": bundle_sha256,
        "lexical_corpus_sha256": lexical_corpus_sha256,
        "promoted_assertions_sha256": promoted_assertions_sha256,
        "knowledge_graph_kind": graph_kind,
        "knowledge_graph_sha256": graph_sha256,
        "knowledge_graph_index_sha256": graph_index_sha256,
        "release_manifest": str(release_manifest_path) if release_manifest_path.is_file() else "",
    }
