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
    "page_cards",
    "actions",
    "facts",
    "evidence_spans",
    "summaries",
    "assertions",
    "entities",
    "communities",
)
_PRE_SELECTED_NAMESPACE_LANES = tuple(
    lane for lane in _NAMESPACE_LANES if lane not in {"page_cards", "actions"}
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
    if str(vector.get("provider") or "pinecone") != str(upload_manifest.get("provider") or "pinecone"):
        errors.append("release manifest vector provider does not match the upload manifest")
    if str(vector.get("production_indexing_contract_fingerprint") or "") != str(
        upload_manifest.get("production_indexing_contract_fingerprint") or ""
    ):
        errors.append("release manifest vector indexing contract does not match the upload manifest")
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
    for key in (
        "selected_release_assembly_sha256",
        "selected_release_binding_sha256",
        "page_graph_navigation_catalog_sha256",
    ):
        if str(vector.get(key) or "") != str(upload_manifest.get(key) or ""):
            errors.append(f"release manifest {key} does not match the upload manifest")


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
    selected_profile = (
        config.get("selected_profile")
        if isinstance(config.get("selected_profile"), Mapping)
        else {}
    )
    selected = bool(str(selected_profile.get("variant_id") or ""))
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
            errors.append(f"current vector upload manifest is missing or invalid: {manifest_path}")
        if errors:
            raise RuntimeArtifactContractError("Runtime artifact contract failed: " + "; ".join(errors))
        return {"validated": False, "legacy": True}

    schema_version = int(manifest.get("schema_version") or 0)
    strict_contract = production or schema_version >= 3
    vector_store_cfg = config.get("vector_store") if isinstance(config.get("vector_store"), Mapping) else {}
    configured_provider = str(vector_store_cfg.get("provider") or "pinecone").strip().lower()
    uploaded_provider = str(manifest.get("provider") or "pinecone").strip().lower()
    if uploaded_provider not in {"pinecone", "pgvector"}:
        errors.append(f"upload manifest vector provider is unsupported: {uploaded_provider}")
    elif strict_contract and uploaded_provider != configured_provider:
        errors.append(
            "upload manifest vector provider does not match runtime config: "
            f"uploaded={uploaded_provider}, configured={configured_provider}"
        )
    if production and schema_version < 4:
        errors.append("production runtime requires vector manifest schema_version 4 or newer")
    if selected and schema_version < 6:
        errors.append("selected-profile runtime requires vector manifest schema_version 6 or newer")
    if selected and int(manifest.get("bundle_version") or 0) < 6:
        errors.append("selected-profile runtime requires retrieval bundle version 6 or newer")
    if uploaded_provider == "pgvector" and strict_contract:
        recorded_pgvector_contract = str(
            manifest.get("production_indexing_contract_fingerprint") or ""
        ).strip()
        expected_pgvector_contract = production_indexing_contract_fingerprint(config)
        if not recorded_pgvector_contract:
            errors.append("pgvector upload manifest is missing its indexing contract fingerprint")
        elif recorded_pgvector_contract != expected_pgvector_contract:
            errors.append("pgvector upload manifest indexing contract does not match runtime config")
    if strict_contract:
        if configured_provider == "pgvector":
            configured_dense_index = (
                f"{str(vector_store_cfg.get('schema') or 'mbzuai_retrieval')}."
                f"{str(vector_store_cfg.get('records_table') or 'embedding_records')}"
            )
            configured_sparse_index = ""
        else:
            configured_dense_index = str(embedder_cfg.get("pinecone_index") or "").strip()
            configured_sparse_index = str(embedder_cfg.get("pinecone_sparse_index") or "").strip()
        if str(manifest.get("index_name") or "").strip() != configured_dense_index:
            errors.append("upload manifest vector index does not match runtime config")
        if bool(embedder_cfg.get("enable_sparse", True)) and str(
            manifest.get("sparse_index_name") or ""
        ).strip() != configured_sparse_index:
            errors.append("upload manifest sparse index does not match runtime config")
        if configured_provider == "pgvector" and str(manifest.get("sparse_index_name") or "").strip():
            errors.append("selected pgvector dense-graph profile must not declare a sparse index")
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

    namespace_lanes = (
        _NAMESPACE_LANES if selected or schema_version >= 6 else _PRE_SELECTED_NAMESPACE_LANES
    )
    manifest_namespaces = manifest.get("namespaces") if isinstance(manifest.get("namespaces"), Mapping) else {}
    missing_namespaces = [lane for lane in namespace_lanes if not str(manifest_namespaces.get(lane) or "").strip()]
    if strict_contract and missing_namespaces:
        errors.append(f"upload manifest is missing namespaces: {missing_namespaces}")
    populated = [str(manifest_namespaces.get(lane) or "").strip() for lane in namespace_lanes]
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
            for lane in namespace_lanes
        }
        if dict(manifest_namespaces) != expected_static:
            errors.append("upload manifest namespaces do not match the static runtime contract")

    bundle_path = _bundle_path(root)
    bundle_payload = load_json_safe(bundle_path, None)
    bundle_sha256 = sha256_file(bundle_path) if bundle_path.is_file() else ""
    uploaded_bundle_sha256 = str(manifest.get("retrieval_bundle_sha256") or "").strip()
    if selected:
        if not isinstance(bundle_payload, Mapping):
            errors.append("selected-profile runtime retrieval bundle is invalid")
        elif int(bundle_payload.get("version") or 0) < 6 or str(
            bundle_payload.get("schema_version") or ""
        ) != "mbzuai.retrieval_bundle.v6":
            errors.append("selected-profile runtime requires retrieval bundle schema v6")
    if production and not uploaded_bundle_sha256:
        errors.append("upload manifest is missing retrieval_bundle_sha256")
    elif uploaded_bundle_sha256 and uploaded_bundle_sha256 != bundle_sha256:
        errors.append("runtime retrieval bundle hash does not match the uploaded bundle")

    selected_release_assembly_sha256 = ""
    selected_release_binding_sha256 = ""
    page_graph_navigation_catalog_sha256 = ""
    if selected:
        assembly_candidates = [
            root / "stage_outputs" / "assemble_selected_release" / "selected_release_assembly.json",
            root / "stage_outputs" / "selected_release_assembly" / "selected_release_assembly.json",
        ]
        configured_assembly = str(manifest.get("selected_release_assembly_file") or "").strip()
        if configured_assembly:
            configured_path = Path(configured_assembly).expanduser()
            if not configured_path.is_absolute():
                configured_path = root / configured_path
            configured_path = configured_path.resolve()
            try:
                configured_path.relative_to(root)
            except ValueError:
                # Archived releases retain producer-side absolute path metadata.
                # Never follow it; the canonical in-release assembly is authoritative.
                pass
            else:
                assembly_candidates.append(configured_path)
        assembly_path = next((path.resolve() for path in assembly_candidates if path.is_file()), None)
        if assembly_path is None:
            errors.append("selected release assembly manifest is missing at runtime")
        else:
            from pipeline.core.release_assembly import (
                SELECTED_DENSE_RECORD_KINDS,
                SELECTED_EMBEDDING_SPEC_KEYS,
                SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION,
                SELECTED_RELEASE_BINDING_ORDER,
                SELECTED_RELEASE_CONTENT_POLICY_SCHEMA_VERSION,
                SELECTED_RELEASE_SOURCE_HASH_KEYS,
                SelectedReleaseAssemblyError,
                selected_release_file_path,
                validate_selected_release_embedding_spec,
            )

            assembly_payload = load_json_safe(assembly_path, None)
            if not isinstance(assembly_payload, Mapping):
                errors.append("selected release assembly manifest is invalid")
            else:
                selected_release_assembly_sha256 = sha256_file(assembly_path)
                selected_release_binding_sha256 = str(
                    assembly_payload.get("assembly_sha256") or ""
                ).strip().lower()
                if str(assembly_payload.get("schema_version") or "") != SELECTED_RELEASE_ASSEMBLY_SCHEMA_VERSION:
                    errors.append("selected release assembly schema is unsupported")
                if str(assembly_payload.get("status") or "") != "ready_for_embedding":
                    errors.append("selected release assembly is not ready")
                if str(assembly_payload.get("variant_id") or "") != str(
                    selected_profile.get("variant_id") or ""
                ):
                    errors.append("selected release assembly variant does not match runtime config")
                if tuple(assembly_payload.get("record_kinds") or []) != SELECTED_DENSE_RECORD_KINDS:
                    errors.append("selected release assembly record kinds drifted")
                try:
                    embedding_spec = validate_selected_release_embedding_spec(
                        assembly_payload
                    )
                except SelectedReleaseAssemblyError as exc:
                    errors.append(str(exc))
                    embedding_spec = {}
                if embedding_spec:
                    embedder_config = (
                        config.get("embedder")
                        if isinstance(config.get("embedder"), Mapping)
                        else {}
                    )
                    configured_embedding = {
                        "provider": str(embedder_config.get("engine") or "").strip(),
                        "model": str(embedder_config.get("model") or "").strip(),
                        "dimensions": int(
                            embedder_config.get("output_dimensionality") or 0
                        ),
                        "query_format": str(
                            embedder_config.get("query_format") or ""
                        ).strip(),
                        "document_format": str(
                            embedder_config.get("document_format") or ""
                        ).strip(),
                        "media_input": str(
                            embedder_config.get("media_input") or ""
                        ).strip().casefold(),
                    }
                    for key in SELECTED_EMBEDDING_SPEC_KEYS:
                        if configured_embedding[key] != embedding_spec[key]:
                            errors.append(
                                f"runtime embedder.{key} differs from the selected release"
                            )
                    uploaded_profile = (
                        manifest.get("selected_profile")
                        if isinstance(manifest.get("selected_profile"), Mapping)
                        else {}
                    )
                    if str(manifest.get("media_input") or "") != str(
                        embedding_spec["media_input"]
                    ) or str(uploaded_profile.get("media_input") or "") != str(
                        embedding_spec["media_input"]
                    ):
                        errors.append(
                            "vector upload media input differs from the selected release"
                        )
                    if uploaded_profile.get("embedding_spec") != embedding_spec:
                        errors.append(
                            "vector upload embedding spec differs from the selected release"
                        )
                if tuple(assembly_payload.get("binding_order") or []) != SELECTED_RELEASE_BINDING_ORDER:
                    errors.append("selected release assembly binding order drifted")
                if assembly_payload.get("embedding_performed") is not False or assembly_payload.get(
                    "upload_performed"
                ) is not False:
                    errors.append(
                        "selected release assembly must remain an immutable pre-embedding artifact"
                    )

                files = (
                    assembly_payload.get("files")
                    if isinstance(assembly_payload.get("files"), Mapping)
                    else {}
                )
                if set(files) != set(SELECTED_RELEASE_BINDING_ORDER):
                    errors.append(
                        "selected release assembly file set differs from its binding order"
                    )
                resolved_assembly_files: Dict[str, Path] = {}
                assembly_file_hashes: list[str] = []
                for key in SELECTED_RELEASE_BINDING_ORDER:
                    try:
                        resolved_path = selected_release_file_path(
                            assembly_payload, assembly_path, key
                        )
                    except SelectedReleaseAssemblyError as exc:
                        errors.append(str(exc))
                    else:
                        resolved_assembly_files[key] = resolved_path
                        assembly_file_hashes.append(sha256_file(resolved_path))
                if len(set(resolved_assembly_files.values())) != len(
                    resolved_assembly_files
                ):
                    errors.append(
                        "selected release assembly file entries must resolve uniquely"
                    )

                source = (
                    assembly_payload.get("source")
                    if isinstance(assembly_payload.get("source"), Mapping)
                    else {}
                )
                source_hashes = [
                    str(source.get(key) or "").strip().lower()
                    for key in SELECTED_RELEASE_SOURCE_HASH_KEYS
                ]
                if any(not re.fullmatch(r"[0-9a-f]{64}", digest) for digest in source_hashes):
                    errors.append("selected release assembly source hash set is incomplete")
                elif len(assembly_file_hashes) == len(SELECTED_RELEASE_BINDING_ORDER):
                    computed_binding = combine_sha256_digests(
                        *source_hashes, *assembly_file_hashes
                    )
                    if selected_release_binding_sha256 != computed_binding:
                        errors.append("selected release assembly binding digest is invalid")

                navigation_path = resolved_assembly_files.get("navigation_catalog")
                if navigation_path is not None:
                    page_graph_navigation_catalog_sha256 = sha256_file(navigation_path)

                bundle_selected_contract = (
                    bundle_payload.get("selected_release_contract")
                    if isinstance(bundle_payload, Mapping)
                    and isinstance(bundle_payload.get("selected_release_contract"), Mapping)
                    else {}
                )
                expected_bundle_contract = {
                    "schema_version": str(assembly_payload.get("schema_version") or ""),
                    "variant_id": str(assembly_payload.get("variant_id") or ""),
                    "manifest_sha256": selected_release_assembly_sha256,
                    "assembly_sha256": selected_release_binding_sha256,
                    "candidate_records_sha256": str(
                        source.get("candidate_records_sha256") or ""
                    ),
                    "navigation_catalog_sha256": page_graph_navigation_catalog_sha256,
                }
                for key, expected in expected_bundle_contract.items():
                    if str(bundle_selected_contract.get(key) or "") != expected:
                        errors.append(
                            f"selected retrieval bundle contract {key} does not match the assembly"
                        )

                expected_lane_counts = (
                    assembly_payload.get("dense_lane_counts")
                    if isinstance(assembly_payload.get("dense_lane_counts"), Mapping)
                    else {}
                )
                uploaded_counts = manifest.get("uploaded") if isinstance(manifest.get("uploaded"), Mapping) else {}
                for lane in ("chunks", "parents", "media", "page_cards", "actions"):
                    if int(uploaded_counts.get(lane) or 0) != int(expected_lane_counts.get(lane) or 0):
                        errors.append(f"selected release uploaded count differs for {lane}")
                for lane in (
                    "facts",
                    "evidence_spans",
                    "summaries",
                    "assertions",
                    "entities",
                    "communities",
                ):
                    if int(uploaded_counts.get(lane) or 0) != 0:
                        errors.append(f"selected release contains unevaluated dense lane: {lane}")

                kind_counts = (
                    assembly_payload.get("record_kind_counts")
                    if isinstance(assembly_payload.get("record_kind_counts"), Mapping)
                    else {}
                )
                expected_counts_from_kinds = {
                    "chunks": int(kind_counts.get("chunk") or 0),
                    "parents": int(kind_counts.get("parent") or 0)
                    + int(kind_counts.get("parent_section") or 0),
                    "media": int(kind_counts.get("media") or 0),
                    "page_cards": int(kind_counts.get("page_card") or 0),
                    "actions": int(kind_counts.get("action") or 0),
                }
                normalized_lane_counts = {
                    lane: int(expected_lane_counts.get(lane) or 0)
                    for lane in expected_counts_from_kinds
                }
                if any(count <= 0 for count in normalized_lane_counts.values()):
                    errors.append("selected release assembly contains an empty evaluated lane")
                if normalized_lane_counts != expected_counts_from_kinds:
                    errors.append(
                        "selected release assembly lane counts differ from record-kind counts"
                    )
                for lane in expected_counts_from_kinds:
                    entry = files.get(lane) if isinstance(files.get(lane), Mapping) else {}
                    if int(entry.get("record_count") or 0) != normalized_lane_counts[lane]:
                        errors.append(f"selected release assembly file count differs for {lane}")
                exact_entry = (
                    files.get("selected_dense_records")
                    if isinstance(files.get("selected_dense_records"), Mapping)
                    else {}
                )
                if int(exact_entry.get("record_count") or 0) != sum(
                    int(kind_counts.get(kind) or 0) for kind in SELECTED_DENSE_RECORD_KINDS
                ):
                    errors.append("selected release exact record count differs from kind counts")
                coverage = (
                    assembly_payload.get("coverage")
                    if isinstance(assembly_payload.get("coverage"), Mapping)
                    else {}
                )
                if coverage.get("all_candidate_chunks_mapped") is not True or coverage.get(
                    "all_navigation_chunks_remapped"
                ) is not True:
                    errors.append("selected release assembly chunk/navigation coverage is incomplete")
                filtered_chunk_coverage_keys = (
                    "candidate_chunk_count",
                    "mapped_chunk_count",
                    "text_exact_match_count",
                    "navigation_chunk_count",
                )
                if any(
                    int(coverage.get(key) or 0) != normalized_lane_counts["chunks"]
                    for key in filtered_chunk_coverage_keys
                ):
                    errors.append("selected release assembly chunk coverage counts drifted")

                # A v3 content policy is applied after the frozen checkpoint is
                # bridged to the candidate records.  The checkpoint count must
                # therefore remain bound to the unfiltered source count, while
                # candidate/mapped/navigation counts bind to the curated lane.
                content_policy = (
                    assembly_payload.get("content_policy")
                    if isinstance(assembly_payload.get("content_policy"), Mapping)
                    else {}
                )
                source_kind_counts = (
                    assembly_payload.get("source_record_kind_counts")
                    if isinstance(
                        assembly_payload.get("source_record_kind_counts"), Mapping
                    )
                    else {}
                )
                checkpoint_chunk_count = int(
                    coverage.get("checkpoint_chunk_count") or 0
                )
                if content_policy:
                    if str(content_policy.get("schema_version") or "") != (
                        SELECTED_RELEASE_CONTENT_POLICY_SCHEMA_VERSION
                    ):
                        errors.append("selected release content policy schema is unsupported")
                    removed_kind_counts = (
                        content_policy.get("removed_record_kind_counts")
                        if isinstance(
                            content_policy.get("removed_record_kind_counts"), Mapping
                        )
                        else {}
                    )
                    for kind in SELECTED_DENSE_RECORD_KINDS:
                        source_count = int(source_kind_counts.get(kind) or 0)
                        removed_count = int(removed_kind_counts.get(kind) or 0)
                        retained_count = int(kind_counts.get(kind) or 0)
                        if source_count - removed_count != retained_count:
                            errors.append(
                                f"selected release content policy count differs for {kind}"
                            )
                    if int(content_policy.get("removed_record_count") or 0) != sum(
                        int(removed_kind_counts.get(kind) or 0)
                        for kind in SELECTED_DENSE_RECORD_KINDS
                    ):
                        errors.append("selected release content policy removed count drifted")
                    if checkpoint_chunk_count != int(
                        source_kind_counts.get("chunk") or 0
                    ):
                        errors.append(
                            "selected release assembly checkpoint chunk count drifted"
                        )
                elif checkpoint_chunk_count != normalized_lane_counts["chunks"]:
                    errors.append(
                        "selected release assembly checkpoint chunk count drifted"
                    )

                bundle_stats = (
                    bundle_payload.get("stats")
                    if isinstance(bundle_payload, Mapping)
                    and isinstance(bundle_payload.get("stats"), Mapping)
                    else {}
                )
                bundle_count_keys = {
                    "chunks": "chunk_count",
                    "parents": "parent_count",
                    "media": "media_count",
                    "page_cards": "page_card_count",
                    "actions": "action_count",
                }
                for lane, stat_key in bundle_count_keys.items():
                    if int(bundle_stats.get(stat_key) or 0) != int(expected_lane_counts.get(lane) or 0):
                        errors.append(f"selected retrieval bundle count differs for {lane}")

        for key, actual in (
            ("selected_release_assembly_sha256", selected_release_assembly_sha256),
            ("selected_release_binding_sha256", selected_release_binding_sha256),
            ("page_graph_navigation_catalog_sha256", page_graph_navigation_catalog_sha256),
        ):
            if str(manifest.get(key) or "").strip().lower() != actual:
                errors.append(f"upload manifest {key} does not match runtime")

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
        if selected:
            expected_upload_input_sha256 = combine_sha256_digests(
                uploaded_bundle_sha256,
                lexical_corpus_sha256,
                promoted_assertions_sha256,
                uploaded_graph_sha256,
                graph_index_sha256,
                selected_release_assembly_sha256,
                selected_release_binding_sha256,
                page_graph_navigation_catalog_sha256,
            )
        elif current_contract:
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
                "upload_input_sha256 does not bind all runtime retrieval artifacts"
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
        "vector_store_provider": uploaded_provider,
        "vector_manifest_schema_version": schema_version,
        "namespace_strategy": manifest_strategy,
        "namespace_release_id": str(manifest.get("namespace_release_id") or ""),
        "retrieval_bundle_sha256": bundle_sha256,
        "lexical_corpus_sha256": lexical_corpus_sha256,
        "promoted_assertions_sha256": promoted_assertions_sha256,
        "knowledge_graph_kind": graph_kind,
        "knowledge_graph_sha256": graph_sha256,
        "knowledge_graph_index_sha256": graph_index_sha256,
        "selected_release_assembly_sha256": selected_release_assembly_sha256,
        "selected_release_binding_sha256": selected_release_binding_sha256,
        "page_graph_navigation_catalog_sha256": page_graph_navigation_catalog_sha256,
        "release_manifest": str(release_manifest_path) if release_manifest_path.is_file() else "",
    }
