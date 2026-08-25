"""Pipeline stage for deterministic selected-release assembly."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.registry import register_stage
from pipeline.core.release_assembly import (
    SELECTED_DENSE_RECORD_KINDS,
    SelectedReleaseAssemblyError,
    assemble_selected_release,
)


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@register_stage
class SelectedReleaseAssemblyFormatter(FormatterStage):
    name = "selected_release_assembly"
    description = (
        "Joins the frozen controlled A/B records to the audited chunk and Page "
        "Graph checkpoint without changing evaluated text or IDs."
    )

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        profile = config.get("selected_profile")
        formatter = config.get("formatter") if isinstance(config.get("formatter"), Mapping) else {}
        assembly = formatter.get("selected_release_assembly") if isinstance(formatter, Mapping) else {}
        if not isinstance(profile, Mapping):
            return ["selected_profile must be a mapping"]
        if not isinstance(assembly, Mapping):
            return ["formatter.selected_release_assembly must be a mapping"]
        errors: List[str] = []
        for key in (
            "candidate_manifest_sha256",
            "candidate_records_sha256",
        ):
            if not _SHA256_RE.fullmatch(str(assembly.get(key) or "").strip().lower()):
                errors.append(f"formatter.selected_release_assembly.{key} must be a SHA-256 digest")
        evidence = assembly.get("checkpoint_evidence")
        if not isinstance(evidence, Mapping):
            errors.append("formatter.selected_release_assembly.checkpoint_evidence must be a mapping")
        else:
            for key in (
                "pipeline_state_sha256",
                "artifact_catalog_sha256",
                "run_audit_sha256",
                "resolved_config_sha256",
                "chunk_index_sha256",
                "page_graph_bridge_sha256",
                "navigation_catalog_sha256",
            ):
                if not _SHA256_RE.fullmatch(str(evidence.get(key) or "").strip().lower()):
                    errors.append(
                        "formatter.selected_release_assembly.checkpoint_evidence."
                        f"{key} must be a SHA-256 digest"
                    )
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        profile = ctx.config.get("selected_profile") or {}
        config = ctx.formatter_config.get("selected_release_assembly") or {}
        try:
            manifest = assemble_selected_release(
                output_dir=ctx.stage_work_dir,
                variant_id=str(profile.get("variant_id") or ""),
                record_kinds=list(profile.get("record_kinds") or []),
                decision_file=str(profile.get("decision_file") or ""),
                decision_sha256=str(profile.get("decision_sha256") or ""),
                candidate_manifest_file=str(config.get("candidate_manifest_file") or ""),
                candidate_manifest_sha256=str(config.get("candidate_manifest_sha256") or ""),
                candidate_records_file=str(config.get("candidate_records_file") or ""),
                candidate_records_sha256=str(config.get("candidate_records_sha256") or ""),
                checkpoint_run_dir=str(config.get("checkpoint_run_dir") or ""),
                checkpoint_evidence=dict(config.get("checkpoint_evidence") or {}),
            )
            files = manifest["files"]

            def file_path(key: str) -> Path:
                return ctx.stage_work_dir / str(files[key]["file"])

            artifact_specs = (
                ("selected_release_assembly", "production_release_assembly", Path(manifest["manifest_file"]), "assembly"),
                ("formatted_documents", "selected_dense_corpus", file_path("selected_dense_records"), "selected_dense_records"),
                ("formatted_documents", "embedding_payload_chunks", file_path("chunks"), "chunks"),
                ("formatted_documents", "embedding_payload_parents", file_path("parents"), "parents"),
                ("formatted_documents", "embedding_payload_media", file_path("media"), "media"),
                ("formatted_documents", "embedding_payload_page_cards", file_path("page_cards"), "page_cards"),
                ("formatted_documents", "embedding_payload_actions", file_path("actions"), "actions"),
                ("chunk_index", "retrieval_chunks", file_path("chunk_index"), "chunk_index"),
                ("page_graph_navigation_catalog", "retrieval_navigation_runtime", file_path("navigation_catalog"), "navigation_catalog"),
                ("chunk_id_bridge", "selected_to_evaluated_ids", file_path("chunk_id_bridge"), "chunk_id_bridge"),
            )
            artifacts = [
                ctx.make_artifact(
                    path,
                    artifact_type=artifact_type,
                    role=role,
                    metadata={
                        "variant_id": manifest["variant_id"],
                        "assembly_sha256": manifest["assembly_sha256"],
                        "records": int((files.get(file_key) or {}).get("record_count") or 0),
                    },
                )
                for artifact_type, role, path, file_key in artifact_specs
            ]
            outputs = {
                "selected_release_assembly_file": manifest["manifest_file"],
                "selected_release_assembly_sha256": manifest["manifest_sha256"],
                "selected_release_binding_sha256": manifest["assembly_sha256"],
                "selected_dense_records_file": str(file_path("selected_dense_records")),
                "chunk_embedding_file": str(file_path("chunks")),
                "parent_embedding_file": str(file_path("parents")),
                "media_embedding_file": str(file_path("media")),
                "page_card_embedding_file": str(file_path("page_cards")),
                "action_embedding_file": str(file_path("actions")),
                "chunks_file": str(file_path("chunk_index")),
                "chunk_index_file": str(file_path("chunk_index")),
                "page_graph_navigation_catalog_file": str(file_path("navigation_catalog")),
                "embedding_performed": False,
                "indexing_performed": False,
            }
            lane_counts = dict(manifest["dense_lane_counts"])
            return StageResult.success(
                outputs=outputs,
                metrics={
                    **{f"{lane}_records": count for lane, count in lane_counts.items()},
                    "dense_records": sum(lane_counts.values()),
                    "evaluated_record_kinds": len(SELECTED_DENSE_RECORD_KINDS),
                    "mapped_chunks": int(manifest["coverage"]["mapped_chunk_count"]),
                },
                artifacts=artifacts,
            )
        except (OSError, TypeError, ValueError, SelectedReleaseAssemblyError) as exc:
            return StageResult.failure(f"Selected release assembly failed: {exc}")
