#!/usr/bin/env python3
"""Fail-closed validation for a candidate or promoted retrieval runtime.

This module deliberately uses only the Python standard library.  It runs before
the retriever imports pipeline code, so a broken or incomplete release can
never make the service appear healthy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


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
SELECTED_RELEASE_LANES = (
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
SELECTED_EVALUATED_DENSE_LANES = {
    "chunks",
    "parents",
    "media",
    "page_cards",
    "actions",
}
SELECTED_RECORD_KINDS = (
    "chunk",
    "parent",
    "parent_section",
    "media",
    "page_card",
    "action",
)
SELECTED_ASSEMBLY_SCHEMAS = {
    "mbzuai.selected_release_assembly.v2",
    "mbzuai.selected_release_assembly.v3",
}
SELECTED_CONTENT_POLICY_SCHEMA = "mbzuai.selected_release_content_policy.v1"
SELECTED_EMBEDDING_SPEC_KEYS = (
    "provider",
    "model",
    "dimensions",
    "query_format",
    "document_format",
    "media_input",
)
SELECTED_MEDIA_INPUT_MODES = {"caption_text", "image_and_caption_text"}
SELECTED_SOURCE_HASH_KEYS = (
    "decision_sha256",
    "embedding_spec_sha256",
    "candidate_manifest_sha256",
    "candidate_records_sha256",
    "pipeline_state_sha256",
    "artifact_catalog_sha256",
    "run_audit_sha256",
    "resolved_config_sha256",
    "chunk_index_sha256",
    "page_graph_bridge_sha256",
    "navigation_catalog_sha256",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
PRODUCTION_EVAL_POLICY_ID = "mbzuai-production-eval-v3"
PRODUCTION_RETRIEVAL_DATASET_SHA256 = (
    "fa400a69bcb9f1a61b6cdb9c8033fed3b16426d2d58b8cec1b427fe097499ac8"
)
PRODUCTION_RETRIEVAL_GATES_SHA256 = (
    "ba221e2d2507582d1566270f5116e423fefce21639cfb23c5773819bfad91128"
)
PRODUCTION_ANSWER_DATASET_SHA256 = PRODUCTION_RETRIEVAL_DATASET_SHA256
PRODUCTION_ANSWER_GATES_SHA256 = (
    "94fd1df2f00eef43f1fa957bb82147fb08be896f1a474f91b8837b74c92913eb"
)
PRODUCTION_MIN_RETRIEVAL_QUERIES = 160
PRODUCTION_MIN_ANSWER_QUERIES = 160
PREPROD_EVAL_POLICY_ID = "mbzuai-preprod-current-eval-v1"
PREPROD_RETRIEVAL_DATASET_SHA256 = (
    "671844a284042ca0cede5393f419bf017b471de6ecc5021c0f9f370a96d45135"
)
PREPROD_RETRIEVAL_GATES_SHA256 = (
    "08299b4953ac1075624ecf07cbe40c410502df0e5993f58272eae1565407b304"
)
PREPROD_ANSWER_DATASET_SHA256 = PREPROD_RETRIEVAL_DATASET_SHA256
PREPROD_ANSWER_GATES_SHA256 = (
    "79dd20b2b909117dcf52736a0551747d77dd22a47bd7b6675b3b9956d497ea82"
)
PREPROD_MIN_RETRIEVAL_QUERIES = 95
PREPROD_MIN_ANSWER_QUERIES = 95
PRODUCTION_ANSWER_JUDGE_PROVIDER = "gemini"
PRODUCTION_ANSWER_JUDGE_MODEL = "gemini-2.5-flash"
PRODUCTION_EVAL_POLICIES = {
    PRODUCTION_EVAL_POLICY_ID: {
        "retrieval_dataset_sha256": PRODUCTION_RETRIEVAL_DATASET_SHA256,
        "retrieval_gates_sha256": PRODUCTION_RETRIEVAL_GATES_SHA256,
        "answer_dataset_sha256": PRODUCTION_ANSWER_DATASET_SHA256,
        "answer_gates_sha256": PRODUCTION_ANSWER_GATES_SHA256,
        "minimum_retrieval_queries": PRODUCTION_MIN_RETRIEVAL_QUERIES,
        "minimum_answer_queries": PRODUCTION_MIN_ANSWER_QUERIES,
    },
    PREPROD_EVAL_POLICY_ID: {
        "retrieval_dataset_sha256": PREPROD_RETRIEVAL_DATASET_SHA256,
        "retrieval_gates_sha256": PREPROD_RETRIEVAL_GATES_SHA256,
        "answer_dataset_sha256": PREPROD_ANSWER_DATASET_SHA256,
        "answer_gates_sha256": PREPROD_ANSWER_GATES_SHA256,
        "minimum_retrieval_queries": PREPROD_MIN_RETRIEVAL_QUERIES,
        "minimum_answer_queries": PREPROD_MIN_ANSWER_QUERIES,
    },
}
_SECRET_CONFIG_KEYS = {
    "api_key",
    "authorization_header",
    "access_key_id",
    "aws_access_key_id",
    "aws_secret_access_key",
    "authorization",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "neo4j_password",
    "password",
    "private_key",
    "secret",
    "secret_access_key",
    "token",
    "x_api_key",
}
_SECRET_CONFIG_SUFFIXES = (
    "_access_key_id",
    "_api_key",
    "_auth_token",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_secret_access_key",
    "_token",
)


class ValidationError(RuntimeError):
    """Raised when a deployment release is unsafe to serve."""


def _bool(value: str | bool | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValidationError(f"invalid boolean value: {value!r}")


def _json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValidationError(f"missing required {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValidationError(f"could not parse {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValidationError(f"{label} must contain a JSON object: {path}")
    return payload


def _validate_nonempty_json_container(path: Path, label: str, *, opening_byte: bytes) -> None:
    """Perform a bounded structural check without materializing huge runtime JSON.

    The retriever fully parses these files after this short-lived validator
    exits. Here we verify regular-file presence, expected top-level shape, and
    non-emptiness; immutable SHA256 bindings provide integrity.
    """

    if not path.is_file():
        raise ValidationError(f"missing required {label}: {path}")
    if path.stat().st_size <= 2:
        raise ValidationError(f"{label} is empty: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(4096).lstrip()
        if not prefix.startswith(opening_byte):
            shape = "array" if opening_byte == b"[" else "object"
            raise ValidationError(f"{label} must contain a JSON {shape}: {path}")
        first_payload_byte = prefix[1:].lstrip()[:1]
        closing_byte = b"]" if opening_byte == b"[" else b"}"
        if not first_payload_byte or first_payload_byte == closing_byte:
            raise ValidationError(f"{label} must not be empty: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_graph_index_binding(index_path: Path, graph_sha256: str) -> str | None:
    """Validate the streaming-friendly graph/index source binding.

    Full deterministic graph-index reconstruction happens during release
    construction and promotion. Runtime startup only needs to stream the graph
    digest and read the bounded index prefix written by the producer.
    """

    try:
        with index_path.open("rb") as handle:
            prefix = handle.read(4096)
    except OSError as exc:
        return f"could not read knowledge graph index prefix: {exc}"
    match = re.search(
        rb'"source_graph_sha256"\s*:\s*"([0-9a-fA-F]{64})"',
        prefix,
    )
    if match is None:
        return "knowledge graph index is missing source_graph_sha256 in its JSON prefix"
    recorded_sha256 = match.group(1).decode("ascii").lower()
    if recorded_sha256 != graph_sha256:
        return "knowledge graph index source_graph_sha256 does not match its graph file"
    return None


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _secret_config_paths(value: Any, *, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            normalized = str(key).strip().lower().replace("-", "_")
            secret_key = not normalized.endswith("_per_token") and (
                normalized in _SECRET_CONFIG_KEYS
                or normalized.endswith(_SECRET_CONFIG_SUFFIXES)
            )
            if secret_key:
                if item not in (None, "", [], {}, ()):
                    paths.append(path)
                continue
            paths.extend(_secret_config_paths(item, prefix=path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_secret_config_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _validate_gates(manifest: dict[str, Any], *, allow_waiver: bool) -> list[str]:
    errors: list[str] = []
    status = str(manifest.get("status") or "").strip()
    allowed_statuses = {"passed"}
    if allow_waiver:
        allowed_statuses.add("passed_with_waiver")
    if status not in allowed_statuses:
        errors.append(
            "release status must be passed"
            + (" or passed_with_waiver" if allow_waiver else "")
            + f"; got {status or '<missing>'}"
        )
    if manifest.get("promoted") is not True:
        errors.append("active release manifest is not marked promoted")
    release_errors = manifest.get("errors")
    if not isinstance(release_errors, list) or release_errors:
        errors.append("release errors must be an empty list")
    preflight = manifest.get("preflight") if isinstance(manifest.get("preflight"), dict) else {}
    audit = manifest.get("audit") if isinstance(manifest.get("audit"), dict) else {}
    evaluation = manifest.get("evaluation") if isinstance(manifest.get("evaluation"), dict) else {}
    answer = manifest.get("answer_evaluation") if isinstance(manifest.get("answer_evaluation"), dict) else {}
    retrieval_policy_id = str(evaluation.get("policy_id") or "").strip()
    answer_policy_id = str(answer.get("policy_id") or "").strip()
    policy_id = retrieval_policy_id
    if (
        not policy_id
        or policy_id != answer_policy_id
        or policy_id not in PRODUCTION_EVAL_POLICIES
    ):
        errors.append(
            "retrieval and answer evaluation policy_id values must match one trusted "
            "production policy"
        )
        policy_id = PRODUCTION_EVAL_POLICY_ID
    policy = PRODUCTION_EVAL_POLICIES[policy_id]
    if preflight.get("ok") is not True:
        errors.append("production preflight is not marked successful")
    if audit.get("ok") is not True:
        errors.append("run audit is not marked successful")
    retrieval_gates = evaluation.get("gates") if isinstance(evaluation.get("gates"), dict) else {}
    if retrieval_gates.get("passed") is not True:
        errors.append("retrieval evaluation gates are not marked passed")
    expected_retrieval_policy = {
        "policy_id": policy_id,
        "dataset_sha256": policy["retrieval_dataset_sha256"],
        "gates_sha256": policy["retrieval_gates_sha256"],
        "minimum_query_count": policy["minimum_retrieval_queries"],
    }
    for key, expected in expected_retrieval_policy.items():
        if evaluation.get(key) != expected:
            errors.append(
                f"retrieval evaluation {key} does not match {policy_id}"
            )
    minimum_retrieval_queries = int(policy["minimum_retrieval_queries"])
    minimum_answer_queries = int(policy["minimum_answer_queries"])
    if _positive_int(evaluation.get("query_count")) < minimum_retrieval_queries:
        errors.append(
            "retrieval evaluation query_count must be at least "
            f"{minimum_retrieval_queries}"
        )

    if not answer:
        errors.append("answer evaluation is missing")
    expected_answer_policy = {
        "policy_id": policy_id,
        "dataset_sha256": policy["answer_dataset_sha256"],
        "gates_sha256": policy["answer_gates_sha256"],
        "minimum_query_count": minimum_answer_queries,
    }
    for key, expected in expected_answer_policy.items():
        if answer.get(key) != expected:
            errors.append(
                f"answer evaluation {key} does not match {policy_id}"
            )
    if status == "passed_with_waiver":
        if not allow_waiver:
            errors.append("answer-evaluation waivers are disabled for runtime startup")
        if answer.get("waived") is not True or answer.get("skipped") is not True:
            errors.append("passed_with_waiver requires an explicit skipped answer-evaluation waiver")
        if not str(answer.get("waiver_reason") or "").strip():
            errors.append("passed_with_waiver requires a non-empty waiver reason")
    else:
        answer_gates = answer.get("gates") if isinstance(answer.get("gates"), dict) else {}
        if answer.get("skipped") is True or answer.get("waived") is True:
            errors.append("passed releases cannot skip or waive answer evaluation")
        if answer_gates.get("passed") is not True:
            errors.append("answer evaluation gates are not marked passed")
        if _positive_int(answer.get("query_count")) < minimum_answer_queries:
            errors.append(
                "answer evaluation query_count must be at least "
                f"{minimum_answer_queries}"
            )
        judge = answer.get("llm_judge") if isinstance(answer.get("llm_judge"), dict) else {}
        expected_judge = {
            "enabled": True,
            "providers": [PRODUCTION_ANSWER_JUDGE_PROVIDER],
            "models": [PRODUCTION_ANSWER_JUDGE_MODEL],
            "required_provider": PRODUCTION_ANSWER_JUDGE_PROVIDER,
            "required_model": PRODUCTION_ANSWER_JUDGE_MODEL,
            "openai_fallback_allowed": False,
            "identity_mismatch_count": 0,
            "error_count": 0,
        }
        for key, expected in expected_judge.items():
            if judge.get(key) != expected:
                errors.append(
                    f"answer evaluation llm_judge.{key} does not match "
                    f"{policy_id}"
                )
        if _positive_int(judge.get("judged_count")) < minimum_answer_queries:
            errors.append(
                "answer evaluation llm_judge.judged_count must be at least "
                f"{minimum_answer_queries}"
            )
    return errors


def _select_runtime_bundle(work_dir: Path) -> Path:
    candidates = (
        work_dir / "stage_outputs" / "finalize_retrieval_bundle" / "retrieval_bundle.json",
        work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
        work_dir / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise ValidationError(
        "missing retrieval_bundle.json; checked " + ", ".join(str(path) for path in candidates)
    )


def _select_runtime_graph(work_dir: Path) -> tuple[Path, Path, str]:
    # Keep this order exactly aligned with graph_artifacts.GRAPH_ARTIFACT_CANDIDATES.
    candidates = (
        (
            work_dir
            / "stage_outputs"
            / "summarize_community_graph"
            / "summarized_community_graph.json",
            work_dir
            / "stage_outputs"
            / "summarize_community_graph"
            / "summarized_community_graph_index.json",
            "summarized_community_local_graph",
        ),
        (
            work_dir / "stage_outputs" / "community_graph" / "community_knowledge_graph.json",
            work_dir / "stage_outputs" / "community_graph" / "community_knowledge_graph_index.json",
            "community_local_graph",
        ),
        (
            work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph.json",
            work_dir / "stage_outputs" / "promote_graph" / "promoted_knowledge_graph_index.json",
            "promoted_local_graph",
        ),
        (
            work_dir / "stage_outputs" / "format_graph" / "knowledge_graph.json",
            work_dir / "stage_outputs" / "format_graph" / "knowledge_graph_index.json",
            "formatted_local_graph",
        ),
    )
    for graph_path, index_path, graph_type in candidates:
        graph_exists = graph_path.is_file()
        index_exists = index_path.is_file()
        if graph_exists != index_exists:
            raise ValidationError(
                "canonical graph artifact is incomplete: "
                f"graph={graph_path} exists={graph_exists}, index={index_path} exists={index_exists}"
            )
        if graph_exists:
            return graph_path, index_path, graph_type
    checked = [f"{graph} + {index}" for graph, index, _ in candidates]
    raise ValidationError("missing runtime graph and graph index pair; checked " + ", ".join(checked))


def _combine_sha256_digests(*digests: str) -> str:
    combined = hashlib.sha256()
    for value in digests:
        normalized = str(value or "").strip().lower()
        if normalized:
            combined.update(normalized.encode("ascii"))
    return combined.hexdigest()


def _normalize_selected_embedding_spec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("selected release embedding_spec must be an object")
    if set(value) != set(SELECTED_EMBEDDING_SPEC_KEYS):
        raise ValidationError(
            "selected release embedding_spec fields do not match the evaluated contract"
        )
    normalized: dict[str, Any] = {}
    for key in SELECTED_EMBEDDING_SPEC_KEYS:
        if key == "dimensions":
            dimensions = _positive_int(value.get(key))
            if dimensions <= 0:
                raise ValidationError(
                    "selected release embedding_spec.dimensions must be positive"
                )
            normalized[key] = dimensions
            continue
        text_value = str(value.get(key) or "").strip()
        if not text_value:
            raise ValidationError(f"selected release embedding_spec.{key} is required")
        normalized[key] = text_value
    normalized["media_input"] = str(normalized["media_input"]).casefold()
    if normalized["media_input"] not in SELECTED_MEDIA_INPUT_MODES:
        raise ValidationError(
            "selected release embedding_spec.media_input is unsupported"
        )
    return normalized


def _embedding_spec_sha256(value: Any) -> str:
    normalized = _normalize_selected_embedding_spec(value)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_selected_release_assembly(
    *,
    work_dir: Path,
    selected_profile: dict[str, Any],
    embedder_config: dict[str, Any],
    upload: dict[str, Any],
) -> tuple[Path, Path, str, str, str, dict[str, int]]:
    assembly_path = (
        work_dir
        / "stage_outputs"
        / "assemble_selected_release"
        / "selected_release_assembly.json"
    )
    assembly = _json_object(assembly_path, "selected release assembly")
    errors: list[str] = []
    assembly_schema = str(assembly.get("schema_version") or "")
    if assembly_schema not in SELECTED_ASSEMBLY_SCHEMAS:
        errors.append("selected release assembly schema is unsupported")
    if str(assembly.get("status") or "") != "ready_for_embedding":
        errors.append("selected release assembly is not ready for embedding")
    if str(assembly.get("variant_id") or "") != str(selected_profile.get("variant_id") or ""):
        errors.append("selected release assembly variant does not match resolved config")
    if tuple(assembly.get("record_kinds") or ()) != SELECTED_RECORD_KINDS:
        errors.append("selected release assembly record kinds do not match the evaluated contract")
    if assembly.get("embedding_performed") is not False or assembly.get("upload_performed") is not False:
        errors.append("selected release assembly must remain a pre-embedding immutable artifact")

    try:
        embedding_spec = _normalize_selected_embedding_spec(assembly.get("embedding_spec"))
    except ValidationError as exc:
        errors.append(str(exc))
        embedding_spec = {}
    if embedding_spec:
        configured_embedding = {
            "provider": str(embedder_config.get("engine") or "").strip(),
            "model": str(embedder_config.get("model") or "").strip(),
            "dimensions": _positive_int(embedder_config.get("output_dimensionality")),
            "query_format": str(embedder_config.get("query_format") or "").strip(),
            "document_format": str(embedder_config.get("document_format") or "").strip(),
            "media_input": str(embedder_config.get("media_input") or "").strip().casefold(),
        }
        for key in SELECTED_EMBEDDING_SPEC_KEYS:
            if configured_embedding[key] != embedding_spec[key]:
                errors.append(
                    f"resolved embedder.{key} differs from the selected release"
                )
        uploaded_profile = (
            upload.get("selected_profile")
            if isinstance(upload.get("selected_profile"), dict)
            else {}
        )
        if upload.get("media_input") != embedding_spec["media_input"]:
            errors.append("vector upload media input differs from the selected release")
        if uploaded_profile.get("media_input") != embedding_spec["media_input"]:
            errors.append(
                "vector upload selected-profile media input differs from the selected release"
            )
        if uploaded_profile.get("embedding_spec") != embedding_spec:
            errors.append(
                "vector upload embedding spec differs from the selected release"
            )

    files = assembly.get("files") if isinstance(assembly.get("files"), dict) else {}
    binding_order = tuple(assembly.get("binding_order") or ())
    expected_binding_order = (
        "selected_dense_records",
        "chunks",
        "parents",
        "media",
        "page_cards",
        "actions",
        "chunk_index",
        "navigation_catalog",
        "chunk_id_bridge",
    )
    if binding_order != expected_binding_order:
        errors.append("selected release assembly binding order is invalid")
    if set(files) != set(expected_binding_order):
        errors.append(
            "selected release assembly file set must exactly match its binding order"
        )

    file_hashes: list[str] = []
    resolved_files: dict[str, Path] = {}
    for key in expected_binding_order:
        entry = files.get(key) if isinstance(files.get(key), dict) else {}
        relative = str(entry.get("file") or "").strip()
        expected_sha = str(entry.get("sha256") or "").strip().lower()
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or ".." in relative_path.parts
            or not re.fullmatch(r"[0-9a-f]{64}", expected_sha)
        ):
            errors.append(f"selected release assembly file entry is invalid: {key}")
            continue
        unresolved_path = assembly_path.parent / relative_path
        path = unresolved_path.resolve()
        if not _is_within(path, assembly_path.parent):
            errors.append(f"selected release assembly file escapes its stage directory: {key}")
            continue
        if not path.is_file() or unresolved_path.is_symlink():
            errors.append(f"selected release assembly file is missing or unsafe: {key}")
            continue
        actual_sha = _sha256(path)
        if actual_sha != expected_sha:
            errors.append(f"selected release assembly file SHA256 drifted: {key}")
        file_hashes.append(actual_sha)
        resolved_files[key] = path
    if len(set(resolved_files.values())) != len(resolved_files):
        errors.append("selected release assembly file entries must resolve uniquely")

    source = assembly.get("source") if isinstance(assembly.get("source"), dict) else {}
    source_hash_keys = list(SELECTED_SOURCE_HASH_KEYS)
    if assembly_schema == "mbzuai.selected_release_assembly.v3":
        source_hash_keys.append("content_policy_sha256")
        policy = (
            assembly.get("content_policy")
            if isinstance(assembly.get("content_policy"), dict)
            else {}
        )
        excluded = policy.get("excluded_document_revision_ids")
        if (
            policy.get("schema_version") != SELECTED_CONTENT_POLICY_SCHEMA
            or not isinstance(excluded, list)
            or excluded != sorted(set(str(value) for value in excluded))
        ):
            errors.append("selected release content policy is invalid")
        else:
            policy_payload = {
                "schema_version": SELECTED_CONTENT_POLICY_SCHEMA,
                "excluded_document_revision_ids": excluded,
            }
            policy_sha = hashlib.sha256(
                json.dumps(
                    policy_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if str(source.get("content_policy_sha256") or "") != policy_sha:
                errors.append("selected release content policy digest is invalid")
    source_hashes = [
        str(source.get(key) or "").strip().lower()
        for key in source_hash_keys
    ]
    if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in source_hashes):
        errors.append("selected release assembly source hash set is incomplete")
    embedding_spec_digest = str(
        source.get("embedding_spec_sha256") or ""
    ).strip().lower()
    if embedding_spec and embedding_spec_digest != _embedding_spec_sha256(
        embedding_spec
    ):
        errors.append("selected release embedding spec digest is invalid")
    binding_sha = str(assembly.get("assembly_sha256") or "").strip().lower()
    if len(file_hashes) == len(expected_binding_order):
        actual_binding_sha = _combine_sha256_digests(*source_hashes, *file_hashes)
        if binding_sha != actual_binding_sha:
            errors.append("selected release assembly binding digest is invalid")

    dense_counts = (
        assembly.get("dense_lane_counts")
        if isinstance(assembly.get("dense_lane_counts"), dict)
        else {}
    )
    normalized_counts = {
        lane: _positive_int(dense_counts.get(lane))
        for lane in SELECTED_EVALUATED_DENSE_LANES
    }
    if any(count <= 0 for count in normalized_counts.values()):
        errors.append("selected release assembly contains an empty evaluated dense lane")
    kind_counts = (
        assembly.get("record_kind_counts")
        if isinstance(assembly.get("record_kind_counts"), dict)
        else {}
    )
    normalized_kind_counts = {
        kind: _positive_int(kind_counts.get(kind)) for kind in SELECTED_RECORD_KINDS
    }
    expected_dense_counts = {
        "chunks": normalized_kind_counts["chunk"],
        "parents": normalized_kind_counts["parent"]
        + normalized_kind_counts["parent_section"],
        "media": normalized_kind_counts["media"],
        "page_cards": normalized_kind_counts["page_card"],
        "actions": normalized_kind_counts["action"],
    }
    if normalized_counts != expected_dense_counts:
        errors.append("selected release dense lane counts do not match record-kind counts")
    file_count_keys = {
        "chunks": "chunks",
        "parents": "parents",
        "media": "media",
        "page_cards": "page_cards",
        "actions": "actions",
    }
    for lane, file_key in file_count_keys.items():
        entry = files.get(file_key) if isinstance(files.get(file_key), dict) else {}
        if _positive_int(entry.get("record_count")) != normalized_counts[lane]:
            errors.append(f"selected release assembly file count differs for {lane}")
    exact_entry = (
        files.get("selected_dense_records")
        if isinstance(files.get("selected_dense_records"), dict)
        else {}
    )
    if _positive_int(exact_entry.get("record_count")) != sum(normalized_kind_counts.values()):
        errors.append("selected release exact record count differs from record-kind counts")
    uploaded = upload.get("uploaded") if isinstance(upload.get("uploaded"), dict) else {}
    for lane, expected in normalized_counts.items():
        if _positive_int(uploaded.get(lane)) != expected:
            errors.append(
                f"selected release upload count differs from assembly for {lane}"
            )
    coverage = assembly.get("coverage") if isinstance(assembly.get("coverage"), dict) else {}
    if coverage.get("all_candidate_chunks_mapped") is not True or coverage.get(
        "all_navigation_chunks_remapped"
    ) is not True:
        errors.append("selected release assembly chunk/navigation coverage is incomplete")

    manifest_sha = _sha256(assembly_path)
    navigation_path = resolved_files.get("navigation_catalog")
    navigation_sha = _sha256(navigation_path) if navigation_path else ""
    expected_hashes = {
        "selected_release_assembly_sha256": manifest_sha,
        "selected_release_binding_sha256": binding_sha,
        "page_graph_navigation_catalog_sha256": navigation_sha,
    }
    for key, actual in expected_hashes.items():
        if str(upload.get(key) or "").strip().lower() != actual:
            errors.append(f"vector upload manifest {key} does not match the selected assembly")

    if errors:
        raise ValidationError("selected release assembly validation failed: " + "; ".join(errors))
    assert navigation_path is not None
    return (
        assembly_path,
        navigation_path,
        manifest_sha,
        binding_sha,
        navigation_sha,
        normalized_counts,
    )


def _validate_upload_manifest(
    *,
    upload: dict[str, Any],
    run_id: str,
    expected_model: str,
    expected_dimension: int,
    selected_profile: bool = False,
) -> list[str]:
    errors: list[str] = []
    provider = str(upload.get("provider") or "pinecone").strip().lower()
    if provider not in {"pinecone", "pgvector"}:
        errors.append(f"unsupported vector provider: {provider}")
    if _positive_int(upload.get("schema_version")) < 4:
        errors.append("vector upload manifest schema_version must be at least 4")
    if provider == "pgvector" and _positive_int(upload.get("schema_version")) < 5:
        errors.append("pgvector upload manifest schema_version must be at least 5")
    if selected_profile and _positive_int(upload.get("schema_version")) < 6:
        errors.append("selected-profile vector upload manifest schema_version must be at least 6")
    if (provider == "pgvector" or selected_profile) and not re.fullmatch(
        r"[0-9a-f]{64}",
        str(upload.get("production_indexing_contract_fingerprint") or "").strip().lower(),
    ):
        errors.append("vector upload manifest indexing contract fingerprint is missing or invalid")
    if str(upload.get("model") or "").strip() != expected_model:
        errors.append(
            f"embedding model must be {expected_model!r}; got {str(upload.get('model') or '<missing>')!r}"
        )
    if _positive_int(upload.get("output_dimensionality")) != expected_dimension:
        errors.append(
            f"embedding dimension must be {expected_dimension}; "
            f"got {_positive_int(upload.get('output_dimensionality'))}"
        )
    if not str(upload.get("index_name") or "").strip():
        errors.append("vector index target is missing")
    sparse_enabled = bool(str(upload.get("sparse_index_name") or "").strip())
    if provider == "pinecone" and not sparse_enabled and not selected_profile:
        errors.append("sparse Pinecone index name is missing")
    if provider == "pgvector" and sparse_enabled:
        errors.append("selected pgvector dense-graph release must not declare a sparse index")
    if str(upload.get("namespace_strategy") or "").strip().lower() != "release":
        errors.append("vector namespaces must use namespace_strategy=release")
    if str(upload.get("namespace_release_id") or "").strip() != run_id:
        errors.append("namespace_release_id must match the release run_id")

    lanes = SELECTED_RELEASE_LANES if selected_profile else LANES
    namespaces = upload.get("namespaces") if isinstance(upload.get("namespaces"), dict) else {}
    namespace_values = [str(namespaces.get(lane) or "").strip() for lane in lanes]
    missing = [lane for lane, value in zip(lanes, namespace_values) if not value]
    if missing:
        errors.append(f"vector namespace mappings are missing lanes: {missing}")
    if len(set(value for value in namespace_values if value)) != len([v for v in namespace_values if v]):
        errors.append("dense/sparse retrieval lanes must use distinct namespaces")
    static_lanes = [lane for lane, value in zip(lanes, namespace_values) if value == lane]
    if static_lanes:
        errors.append(f"release uses unsafe static namespace names: {static_lanes}")

    planned = upload.get("planned") if isinstance(upload.get("planned"), dict) else {}
    uploaded = upload.get("uploaded") if isinstance(upload.get("uploaded"), dict) else {}
    if not sparse_enabled:
        unexpected_sparse_counts = sorted(
            key
            for payload in (planned, uploaded)
            for key, value in payload.items()
            if str(key).startswith("sparse_") and _positive_int(value) > 0
        )
        if unexpected_sparse_counts:
            errors.append(
                "dense-only vector upload declares non-zero sparse counts: "
                f"{sorted(set(unexpected_sparse_counts))}"
            )
    for lane in lanes:
        keys = [lane]
        if sparse_enabled:
            keys.append(f"sparse_{lane}")
        for key in keys:
            expected = _positive_int(planned.get(key))
            actual = _positive_int(uploaded.get(key))
            if expected != actual:
                errors.append(f"upload count mismatch for {key}: planned={expected}, uploaded={actual}")
        dense_count = _positive_int(uploaded.get(lane))
        if selected_profile and lane not in SELECTED_EVALUATED_DENSE_LANES:
            if dense_count != 0:
                errors.append(f"selected release unevaluated dense lane must be zero for {lane}")
        elif dense_count <= 0:
            errors.append(f"dense upload count must be greater than zero for {lane}")
        if sparse_enabled and _positive_int(uploaded.get(f"sparse_{lane}")) <= 0:
            errors.append(f"sparse upload count must be greater than zero for {lane}")

    verification = upload.get("verification") if isinstance(upload.get("verification"), dict) else {}
    families = ["dense"]
    if sparse_enabled:
        families.append("sparse")
    for family in families:
        report = verification.get(family) if isinstance(verification.get(family), dict) else {}
        expected_counts = report.get("expected") if isinstance(report.get("expected"), dict) else {}
        actual_counts = report.get("actual") if isinstance(report.get("actual"), dict) else {}
        if report.get("failures") != []:
            errors.append(f"{family} namespace verification contains failures or is missing")
        for lane, namespace in zip(lanes, namespace_values):
            if not namespace:
                continue
            if _positive_int(planned.get(f"{'sparse_' if family == 'sparse' else ''}{lane}")) == 0:
                continue
            expected = _positive_int(expected_counts.get(namespace))
            actual = _positive_int(actual_counts.get(namespace))
            if expected <= 0 or actual != expected:
                errors.append(
                    f"{family} namespace verification mismatch for {lane}/{namespace}: "
                    f"expected={expected}, actual={actual}"
                )
    return errors


def validate(args: argparse.Namespace) -> dict[str, Any]:
    runs_root = Path(args.runs_root).expanduser().resolve()
    marker_path = Path(args.storage_marker_file).expanduser().resolve()
    if args.require_storage_marker and not marker_path.is_file():
        raise ValidationError(
            f"release storage marker is missing: {marker_path}; mount durable storage or hydrate a verified archive"
        )

    release_manifest: dict[str, Any] | None = None
    release_manifest_path: Path | None = None
    pointer: dict[str, Any] | None = None
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        run_id = work_dir.name
        if not args.allow_external_work_dir and not _is_within(work_dir, runs_root):
            raise ValidationError(f"candidate work_dir must be inside {runs_root}: {work_dir}")
        mode = "candidate"
        release_id = ""
    else:
        active_path = Path(args.active_release_file).expanduser().resolve()
        pointer = _json_object(active_path, "active release pointer")
        if _positive_int(pointer.get("schema_version")) != 1:
            raise ValidationError("active release pointer schema_version must be 1")
        run_id = str(pointer.get("run_id") or "").strip()
        if not RUN_ID_PATTERN.fullmatch(run_id):
            raise ValidationError(f"active release pointer has unsafe or missing run_id: {run_id!r}")
        work_dir = (runs_root / run_id).resolve()
        if not _is_within(work_dir, runs_root):
            raise ValidationError(f"active release run escapes configured runs root: {work_dir}")
        release_manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
        release_manifest = _json_object(release_manifest_path, "retrieval release manifest")
        mode = "active"
        release_id = str(release_manifest.get("release_id") or "").strip()

        gate_errors = _validate_gates(release_manifest, allow_waiver=args.allow_waived_release)
        if _positive_int(release_manifest.get("schema_version")) < 2:
            gate_errors.append("production release manifest schema_version must be at least 2")
        if str(pointer.get("status") or "").strip() != str(release_manifest.get("status") or "").strip():
            gate_errors.append("active pointer status does not match release manifest status")
        if str(release_manifest.get("run_id") or "").strip() != run_id:
            gate_errors.append("active pointer run_id does not match release manifest run_id")
        if str(pointer.get("release_id") or "").strip() != release_id:
            gate_errors.append("active pointer release_id does not match release manifest release_id")
        if str(release_manifest.get("config_name") or "").strip() != args.expected_config:
            gate_errors.append(
                f"release config must be {args.expected_config!r}; "
                f"got {str(release_manifest.get('config_name') or '<missing>')!r}"
            )
        if gate_errors:
            raise ValidationError("release gate validation failed: " + "; ".join(gate_errors))

    if not work_dir.is_dir():
        raise ValidationError(f"release work directory does not exist: {work_dir}")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValidationError(f"release work directory has unsafe run_id: {run_id!r}")

    resolved_config_path = work_dir / "resolved_config.json"
    resolved_config = _json_object(resolved_config_path, "resolved production config")
    if str(resolved_config.get("run_id") or "").strip() != run_id:
        raise ValidationError("resolved_config.json run_id does not match the runtime work directory")
    if not isinstance(resolved_config.get("config"), dict):
        raise ValidationError("resolved_config.json is missing its config object")
    resolved_pipeline_config = resolved_config["config"]
    selected_profile = (
        resolved_pipeline_config.get("selected_profile")
        if isinstance(resolved_pipeline_config.get("selected_profile"), dict)
        else {}
    )
    selected_release = bool(str(selected_profile.get("variant_id") or "").strip())
    secret_paths = _secret_config_paths(resolved_config.get("config"))
    if secret_paths:
        raise ValidationError(
            "resolved_config.json contains secret-bearing config fields: "
            + ", ".join(secret_paths)
        )
    indexing_build = (
        resolved_config.get("indexing_build")
        if isinstance(resolved_config.get("indexing_build"), dict)
        else {}
    )
    indexing_build_commit_sha = str(indexing_build.get("commit_sha") or "").strip().lower()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", indexing_build_commit_sha):
        raise ValidationError("resolved_config.json indexing build commit is missing or abbreviated")
    if indexing_build.get("dirty") is not False:
        raise ValidationError("resolved_config.json records a dirty indexing source tree")
    if not isinstance(indexing_build.get("implementation_sha256"), dict) or not indexing_build.get(
        "implementation_sha256"
    ):
        raise ValidationError("resolved_config.json indexing implementation hashes are missing")
    indexing_build_sha256 = hashlib.sha256(
        json.dumps(indexing_build, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    indexing_contract_fingerprint = str(
        resolved_config.get("production_indexing_contract_fingerprint") or ""
    ).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", indexing_contract_fingerprint):
        raise ValidationError(
            "resolved_config.json is missing a valid production_indexing_contract_fingerprint"
        )
    if release_manifest is not None:
        release_fingerprint = str(
            release_manifest.get("production_indexing_contract_fingerprint") or ""
        ).strip().lower()
        pointer_fingerprint = str(
            (pointer or {}).get("production_indexing_contract_fingerprint") or ""
        ).strip().lower()
        if release_fingerprint != indexing_contract_fingerprint:
            raise ValidationError("release manifest indexing contract fingerprint does not match resolved_config.json")
        if pointer_fingerprint != indexing_contract_fingerprint:
            raise ValidationError("active pointer indexing contract fingerprint does not match resolved_config.json")
        serving_contract_fingerprint = str(
            release_manifest.get("production_serving_contract_fingerprint") or ""
        ).strip().lower()
        pointer_serving_fingerprint = str(
            (pointer or {}).get("production_serving_contract_fingerprint") or ""
        ).strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", serving_contract_fingerprint):
            raise ValidationError(
                "release manifest is missing a valid production_serving_contract_fingerprint"
            )
        if pointer_serving_fingerprint != serving_contract_fingerprint:
            raise ValidationError(
                "active pointer serving contract fingerprint does not match release manifest"
            )
        answer_runtime = (
            release_manifest.get("answer_runtime")
            if isinstance(release_manifest.get("answer_runtime"), dict)
            else {}
        )
        answer_runtime_commit_sha = str(answer_runtime.get("commit_sha") or "").strip().lower()
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", answer_runtime_commit_sha):
            raise ValidationError("release manifest answer runtime commit SHA is missing or invalid")
        if not str(answer_runtime.get("pipeline_revision") or "").strip():
            raise ValidationError("release manifest answer runtime pipeline revision is missing")
        if str((pointer or {}).get("answer_runtime_commit_sha") or "").strip().lower() != answer_runtime_commit_sha:
            raise ValidationError("active pointer answer runtime commit does not match release manifest")
        release_indexing_build = (
            release_manifest.get("indexing_build")
            if isinstance(release_manifest.get("indexing_build"), dict)
            else {}
        )
        if release_indexing_build != indexing_build:
            raise ValidationError("release manifest indexing build does not match resolved_config.json")
        if (
            str((pointer or {}).get("indexing_build_commit_sha") or "").strip().lower()
            != indexing_build_commit_sha
        ):
            raise ValidationError("active pointer indexing build commit does not match release manifest")
    else:
        serving_contract_fingerprint = ""
        answer_runtime_commit_sha = ""

    upload_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    upload = _json_object(upload_path, "Pinecone upload manifest")
    upload_errors = _validate_upload_manifest(
        upload=upload,
        run_id=run_id,
        expected_model=args.expected_model,
        expected_dimension=args.expected_dimension,
        selected_profile=selected_release,
    )
    if upload.get("indexing_build") != indexing_build:
        upload_errors.append("Pinecone upload manifest indexing build does not match resolved config")
    if str(upload.get("indexing_build_sha256") or "").strip().lower() != indexing_build_sha256:
        upload_errors.append("Pinecone upload manifest indexing build digest does not match resolved config")

    selected_assembly_path: Path | None = None
    selected_navigation_path: Path | None = None
    selected_assembly_sha = ""
    selected_binding_sha = ""
    selected_navigation_sha = ""
    if selected_release:
        (
            selected_assembly_path,
            selected_navigation_path,
            selected_assembly_sha,
            selected_binding_sha,
            selected_navigation_sha,
            _selected_lane_counts,
        ) = _validate_selected_release_assembly(
            work_dir=work_dir,
            selected_profile=selected_profile,
            embedder_config=(
                resolved_pipeline_config.get("embedder")
                if isinstance(resolved_pipeline_config.get("embedder"), dict)
                else {}
            ),
            upload=upload,
        )

    bundle_path = _select_runtime_bundle(work_dir)
    _validate_nonempty_json_container(bundle_path, "retrieval bundle", opening_byte=b"{")
    if _positive_int(upload.get("bundle_version")) < 5:
        upload_errors.append("retrieval bundle version must be at least 5")
    if selected_release and _positive_int(upload.get("bundle_version")) < 6:
        upload_errors.append("selected-profile retrieval bundle version must be at least 6")
    expected_bundle_sha = str(upload.get("retrieval_bundle_sha256") or "").strip().lower()
    actual_bundle_sha = _sha256(bundle_path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected_bundle_sha) or expected_bundle_sha != actual_bundle_sha:
        upload_errors.append("retrieval bundle SHA256 does not match the upload manifest")

    lexical_path = bundle_path.with_name("lexical_corpus.json")
    _validate_nonempty_json_container(lexical_path, "lexical retrieval corpus", opening_byte=b"[")
    actual_lexical_sha = _sha256(lexical_path)
    if str(upload.get("lexical_corpus_sha256") or "").strip().lower() != actual_lexical_sha:
        upload_errors.append("lexical corpus SHA256 does not match the vector upload contract")

    promoted_assertions_path = (
        work_dir / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
    )
    _validate_nonempty_json_container(
        promoted_assertions_path,
        "promoted assertion sidecar",
        opening_byte=b"[",
    )
    actual_promoted_assertions_sha = _sha256(promoted_assertions_path)
    if (
        str(upload.get("promoted_assertions_sha256") or "").strip().lower()
        != actual_promoted_assertions_sha
    ):
        upload_errors.append("promoted assertions SHA256 does not match the vector upload contract")

    graph_path: Path | None = None
    graph_index_path: Path | None = None
    graph_type = "disabled"
    actual_graph_sha = ""
    actual_graph_index_sha = ""
    if args.require_graph:
        graph_path, graph_index_path, graph_type = _select_runtime_graph(work_dir)
        _validate_nonempty_json_container(graph_path, "knowledge graph", opening_byte=b"{")
        _validate_nonempty_json_container(graph_index_path, "knowledge graph index", opening_byte=b"{")
        expected_graph_sha = str(upload.get("knowledge_graph_sha256") or "").strip().lower()
        actual_graph_sha = _sha256(graph_path)
        graph_binding_error = _validate_graph_index_binding(graph_index_path, actual_graph_sha)
        if graph_binding_error:
            upload_errors.append(graph_binding_error)
        if not re.fullmatch(r"[0-9a-f]{64}", expected_graph_sha) or expected_graph_sha != actual_graph_sha:
            upload_errors.append(
                "runtime knowledge graph SHA256 does not match the graph used to build the vector upload"
            )
        expected_graph_kind = str(upload.get("knowledge_graph_kind") or "").strip()
        if expected_graph_kind != graph_type:
            upload_errors.append(
                f"runtime knowledge graph kind does not match vector upload ({graph_type} != {expected_graph_kind or '<missing>'})"
            )
        expected_graph_index_sha = str(upload.get("knowledge_graph_index_sha256") or "").strip().lower()
        actual_graph_index_sha = _sha256(graph_index_path)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", expected_graph_index_sha)
            or expected_graph_index_sha != actual_graph_index_sha
        ):
            upload_errors.append(
                "runtime knowledge graph index SHA256 does not match the index used to build the vector upload"
            )
        upload_input_sha = str(upload.get("upload_input_sha256") or "").strip().lower()
        upload_input_digest = hashlib.sha256()
        upload_artifact_hashes = [
            expected_bundle_sha,
            actual_lexical_sha,
            actual_promoted_assertions_sha,
            expected_graph_sha,
            expected_graph_index_sha,
        ]
        if selected_release:
            upload_artifact_hashes.extend(
                (
                    selected_assembly_sha,
                    selected_binding_sha,
                    selected_navigation_sha,
                )
            )
        for artifact_sha in upload_artifact_hashes:
            upload_input_digest.update(artifact_sha.encode("utf-8"))
        if upload_input_sha != upload_input_digest.hexdigest():
            upload_errors.append("upload_input_sha256 does not bind all runtime retrieval artifacts")

    if release_manifest is not None:
        vector = release_manifest.get("vector_index") if isinstance(release_manifest.get("vector_index"), dict) else {}
        if str(vector.get("provider") or "pinecone").strip().lower() != str(
            upload.get("provider") or "pinecone"
        ).strip().lower():
            upload_errors.append("release manifest vector provider does not match upload manifest")
        if _positive_int(vector.get("manifest_schema_version")) < 4:
            upload_errors.append("release vector manifest_schema_version must be at least 4")
        if str(vector.get("index_name") or "").strip() != str(upload.get("index_name") or "").strip():
            upload_errors.append("release dense index does not match upload manifest")
        if str(vector.get("sparse_index_name") or "").strip() != str(upload.get("sparse_index_name") or "").strip():
            upload_errors.append("release sparse index does not match upload manifest")
        if vector.get("namespaces") != upload.get("namespaces"):
            upload_errors.append("release namespaces do not match upload manifest")
        if vector.get("indexing_build") != indexing_build:
            upload_errors.append("release vector indexing build does not match resolved config")
        if str(vector.get("indexing_build_sha256") or "").strip().lower() != indexing_build_sha256:
            upload_errors.append("release vector indexing build digest does not match resolved config")
        for key in (
            "namespace_strategy",
            "namespace_release_id",
            "retrieval_bundle_sha256",
            "knowledge_graph_kind",
            "knowledge_graph_sha256",
            "knowledge_graph_index_sha256",
            "upload_input_sha256",
            "lexical_corpus_sha256",
            "promoted_assertions_sha256",
            "selected_release_assembly_sha256",
            "selected_release_binding_sha256",
            "page_graph_navigation_catalog_sha256",
        ):
            if str(vector.get(key) or "") != str(upload.get(key) or ""):
                upload_errors.append(f"release vector metadata {key} does not match upload manifest")
        release_graph = (
            release_manifest.get("knowledge_graph")
            if isinstance(release_manifest.get("knowledge_graph"), dict)
            else {}
        )
        if args.require_graph and str(release_graph.get("graph_type") or "").strip() != graph_type:
            upload_errors.append(
                "release graph selection does not match the graph that the runtime will load "
                f"({str(release_graph.get('graph_type') or '<missing>')} != {graph_type})"
            )
        if args.require_graph:
            if str(release_graph.get("knowledge_graph_sha256") or "").strip().lower() != _sha256(graph_path):
                upload_errors.append("release manifest knowledge graph SHA256 does not match runtime graph")
            if str(release_graph.get("knowledge_graph_index_sha256") or "").strip().lower() != _sha256(graph_index_path):
                upload_errors.append("release manifest knowledge graph index SHA256 does not match runtime graph index")
        release_bundle_sha = str(
            (release_manifest.get("retrieval_bundle") or {}).get("retrieval_bundle_sha256") or ""
        ).strip().lower()
        if release_bundle_sha != actual_bundle_sha:
            upload_errors.append("release manifest retrieval bundle SHA256 does not match runtime bundle")
        release_bundle = (
            release_manifest.get("retrieval_bundle")
            if isinstance(release_manifest.get("retrieval_bundle"), dict)
            else {}
        )
        if str(release_bundle.get("lexical_corpus_sha256") or "").strip().lower() != actual_lexical_sha:
            upload_errors.append("release manifest lexical corpus SHA256 does not match runtime")
        if (
            str(release_bundle.get("promoted_assertions_sha256") or "").strip().lower()
            != actual_promoted_assertions_sha
        ):
            upload_errors.append("release manifest promoted assertions SHA256 does not match runtime")
        if selected_release:
            for key, expected in (
                ("selected_release_assembly_sha256", selected_assembly_sha),
                ("selected_release_binding_sha256", selected_binding_sha),
                ("page_graph_navigation_catalog_sha256", selected_navigation_sha),
            ):
                if str(release_bundle.get(key) or "").strip().lower() != expected:
                    upload_errors.append(
                        f"release retrieval bundle contract {key} does not match runtime"
                    )

    if upload_errors:
        raise ValidationError("release artifact validation failed: " + "; ".join(upload_errors))

    return {
        "ok": True,
        "mode": mode,
        "work_dir": str(work_dir),
        "run_id": run_id,
        "release_id": release_id,
        "release_manifest": str(release_manifest_path or ""),
        "resolved_config": str(resolved_config_path),
        "production_indexing_contract_fingerprint": indexing_contract_fingerprint,
        "production_serving_contract_fingerprint": serving_contract_fingerprint,
        "answer_runtime_commit_sha": answer_runtime_commit_sha,
        "indexing_build_commit_sha": indexing_build_commit_sha,
        "upload_manifest": str(upload_path),
        "retrieval_bundle": str(bundle_path),
        "retrieval_bundle_sha256": actual_bundle_sha,
        "lexical_corpus": str(lexical_path),
        "lexical_corpus_sha256": actual_lexical_sha,
        "promoted_assertions": str(promoted_assertions_path),
        "promoted_assertions_sha256": actual_promoted_assertions_sha,
        "selected_release_assembly": str(selected_assembly_path or ""),
        "selected_release_assembly_sha256": selected_assembly_sha,
        "selected_release_binding_sha256": selected_binding_sha,
        "page_graph_navigation_catalog": str(selected_navigation_path or ""),
        "page_graph_navigation_catalog_sha256": selected_navigation_sha,
        "knowledge_graph": str(graph_path or ""),
        "knowledge_graph_index": str(graph_index_path or ""),
        "graph_type": graph_type,
        "knowledge_graph_sha256": actual_graph_sha,
        "knowledge_graph_index_sha256": actual_graph_index_sha,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-release-file",
        default=os.getenv("ACTIVE_RELEASE_FILE", "/data/releases/mbzuai_main/active_release.json"),
    )
    parser.add_argument(
        "--runs-root",
        default=os.getenv("RELEASE_RUNS_ROOT", "/data/releases/runs/mbzuai_main"),
    )
    parser.add_argument("--work-dir", default=os.getenv("RETRIEVAL_WORK_DIR", "").strip())
    parser.add_argument(
        "--storage-marker-file",
        default=os.getenv("RELEASE_STORAGE_MARKER_FILE", "/data/releases/.mbzuai-release-storage"),
    )
    parser.add_argument(
        "--require-storage-marker",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RETRIEVER_REQUIRE_STORAGE_MARKER"), default=True),
    )
    parser.add_argument(
        "--allow-external-work-dir",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RETRIEVER_ALLOW_EXTERNAL_WORK_DIR"), default=False),
    )
    parser.add_argument(
        "--allow-waived-release",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RETRIEVER_ALLOW_WAIVED_RELEASE"), default=False),
    )
    parser.add_argument(
        "--require-graph",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RETRIEVER_REQUIRE_GRAPH"), default=True),
    )
    parser.add_argument("--expected-config", default=os.getenv("PIPELINE_CONFIG", "mbzuai_production"))
    parser.add_argument(
        "--expected-model",
        default=os.getenv("RETRIEVER_EXPECTED_EMBEDDING_MODEL", "gemini-embedding-2"),
    )
    parser.add_argument(
        "--expected-dimension",
        type=int,
        default=int(os.getenv("RETRIEVER_EXPECTED_EMBEDDING_DIMENSIONALITY", "1536")),
    )
    return parser


def main() -> int:
    try:
        payload = validate(_parser().parse_args())
    except ValidationError as exc:
        print(f"Deployment release validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
