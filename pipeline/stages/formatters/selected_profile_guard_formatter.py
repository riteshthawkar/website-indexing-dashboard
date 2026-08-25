"""Freeze and validate the controlled A/B winner before index preparation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe, sha256_file
from pipeline.core.registry import register_stage


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _profile_errors(config: Mapping[str, Any], decision: Mapping[str, Any]) -> List[str]:
    profile = config.get("selected_profile")
    if not isinstance(profile, Mapping):
        return ["selected_profile must be a mapping"]

    errors: List[str] = []
    winner = decision.get("winner")
    if not isinstance(winner, Mapping):
        return ["controlled A/B decision has no winner object"]

    variant_id = str(profile.get("variant_id") or "")
    if str(winner.get("variant_id") or "") != variant_id:
        errors.append("selected_profile.variant_id does not match the controlled A/B winner")

    chunker = config.get("chunker") if isinstance(config.get("chunker"), Mapping) else {}
    winner_chunk = winner.get("chunk_config") if isinstance(winner.get("chunk_config"), Mapping) else {}
    for key in ("target_tokens", "max_tokens", "overlap_tokens", "min_chunk_tokens"):
        if int(chunker.get(key) or 0) != int(winner_chunk.get(key) or 0):
            errors.append(f"chunker.{key} does not match the controlled A/B winner")
    if int(chunker.get("max_chunks_per_document") or 0) != 0:
        errors.append("chunker.max_chunks_per_document must be 0 (lossless)")

    embedder = config.get("embedder") if isinstance(config.get("embedder"), Mapping) else {}
    embedding = winner.get("embedding_spec") if isinstance(winner.get("embedding_spec"), Mapping) else {}
    expected_dimensions = int(embedding.get("dimensions") or 0)
    if str(embedder.get("engine") or "") != str(embedding.get("provider") or ""):
        errors.append("embedder.engine does not match the controlled A/B winner")
    if str(embedder.get("model") or "") != str(embedding.get("model") or ""):
        errors.append("embedder.model does not match the controlled A/B winner")
    if int(embedder.get("output_dimensionality") or 0) != expected_dimensions:
        errors.append("embedder.output_dimensionality does not match the controlled A/B winner")
    if bool(embedder.get("enable_sparse", True)) or bool(
        embedder.get("use_sparse_embeddings", True)
    ):
        errors.append("selected dense_graph profile must not generate sparse embeddings")

    winner_mode = winner.get("index_mode") if isinstance(winner.get("index_mode"), Mapping) else {}
    expected_kinds = [str(value) for value in winner_mode.get("record_kinds") or []]
    configured_kinds = [str(value) for value in profile.get("record_kinds") or []]
    if configured_kinds != expected_kinds:
        errors.append("selected_profile.record_kinds does not match the controlled A/B winner")
    if not bool(winner_mode.get("dense")) or bool(winner_mode.get("sparse")):
        errors.append("controlled A/B winner is not the expected dense-only index mode")

    retrieval = config.get("retrieval") if isinstance(config.get("retrieval"), Mapping) else {}
    if str(retrieval.get("index_mode") or "") != "dense_graph":
        errors.append("retrieval.index_mode must be dense_graph")
    if bool(retrieval.get("enable_sparse", True)):
        errors.append("retrieval.enable_sparse must be false for the selected profile")
    if not bool(retrieval.get("navigation_plan_enabled", False)):
        errors.append("retrieval.navigation_plan_enabled must be true for dense_graph")

    formatter = config.get("formatter") if isinstance(config.get("formatter"), Mapping) else {}
    bridge = formatter.get("page_graph_bridge") if isinstance(formatter, Mapping) else {}
    if not isinstance(bridge, Mapping) or not bool(
        bridge.get("derive_document_sections", False)
    ):
        errors.append(
            "formatter.page_graph_bridge.derive_document_sections must be true "
            "for parent_section coverage"
        )

    stages = config.get("stages") if isinstance(config.get("stages"), list) else []
    if bool(profile.get("pre_embedding_only", False)):
        forbidden = [
            str(stage.get("id") or stage.get("plugin") or "")
            for stage in stages
            if isinstance(stage, Mapping)
            and (
                str(stage.get("type") or "") == "embedder"
                or "upload" in str(stage.get("plugin") or "").casefold()
            )
        ]
        if forbidden:
            errors.append(
                "pre-embedding selected-profile run contains forbidden stages: "
                + ", ".join(forbidden)
            )
    return errors


@register_stage
class SelectedProfileGuardFormatter(FormatterStage):
    name = "selected_profile_guard"
    description = "Pins the controlled A/B winner and blocks profile drift before chunking."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        profile = config.get("selected_profile")
        if not isinstance(profile, Mapping):
            return ["selected_profile must be a mapping"]
        digest = str(profile.get("decision_sha256") or "").strip().lower()
        if not _SHA256_RE.fullmatch(digest):
            return ["selected_profile.decision_sha256 must be a SHA-256 digest"]
        if not str(profile.get("decision_file") or "").strip():
            return ["selected_profile.decision_file is required"]
        return []

    async def execute(self, ctx: StageContext) -> StageResult:
        profile = ctx.config.get("selected_profile") or {}
        try:
            decision_file = Path(str(profile.get("decision_file") or "")).expanduser().resolve()
            if not decision_file.is_file():
                raise FileNotFoundError(f"controlled A/B decision is missing: {decision_file}")
            actual_digest = sha256_file(decision_file)
            expected_digest = str(profile.get("decision_sha256") or "").strip().lower()
            if actual_digest != expected_digest:
                raise ValueError(
                    "controlled A/B decision digest mismatch: "
                    f"expected {expected_digest}, got {actual_digest}"
                )
            decision = load_json_safe(decision_file, None)
            if not isinstance(decision, Mapping):
                raise ValueError("controlled A/B decision must be a JSON object")
            errors = _profile_errors(ctx.config, decision)
            if errors:
                raise ValueError("; ".join(errors))

            winner = dict(decision.get("winner") or {})
            contract = {
                "schema_version": "mbzuai.selected_index_profile.v1",
                "variant_id": str(profile.get("variant_id") or ""),
                "decision_file": str(decision_file),
                "decision_sha256": actual_digest,
                "pre_embedding_only": bool(profile.get("pre_embedding_only", False)),
                "chunk_config": dict(winner.get("chunk_config") or {}),
                "embedding_spec": dict(winner.get("embedding_spec") or {}),
                "index_mode": dict(winner.get("index_mode") or {}),
                "record_kinds": list(profile.get("record_kinds") or []),
                "production_mutation_performed": False,
            }
            output_file = ctx.stage_work_dir / "selected_profile_contract.json"
            atomic_write_json(output_file, contract)
            artifact = ctx.make_artifact(
                output_file,
                artifact_type="selected_profile_contract",
                role="index_configuration_contract",
                metadata={
                    "variant_id": contract["variant_id"],
                    "decision_sha256": actual_digest,
                    "pre_embedding_only": contract["pre_embedding_only"],
                },
            )
            return StageResult.success(
                outputs={
                    "selected_profile_contract_file": str(output_file),
                    "selected_profile_id": contract["variant_id"],
                    "embedding_performed": False,
                    "indexing_performed": False,
                },
                metrics={"record_kind_count": len(contract["record_kinds"])},
                artifacts=[artifact],
            )
        except (OSError, TypeError, ValueError) as exc:
            return StageResult.failure(f"Selected profile guard failed: {exc}")
