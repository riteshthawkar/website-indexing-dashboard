"""Import one completed crawl as immutable input to downstream processing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json
from pipeline.core.registry import register_stage
from pipeline.stages.formatters.corpus_merge_formatter import _load_source_descriptor


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_DIRECTORIES = ("html_dir", "md_dir", "download_dir")
_REQUIRED_FILES = (
    "mapping_file",
    "page_images_file",
    "page_videos_file",
    "page_media_file",
    "page_metadata_file",
    "page_link_graph_file",
    "runtime_state_file",
    "seed_inventory_file",
)


def _import_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    formatter = config.get("formatter")
    if not isinstance(formatter, Mapping):
        return {}
    value = formatter.get("crawl_artifact_import")
    return value if isinstance(value, Mapping) else {}


@register_stage
class CrawlArtifactImportFormatter(FormatterStage):
    name = "crawl_artifact_import"
    description = "Imports an audited completed crawl without mutating its raw artifacts."

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        value = _import_config(config)
        errors: List[str] = []
        if not value:
            return ["formatter.crawl_artifact_import must be a mapping"]
        if not str(value.get("source_run_dir") or "").strip():
            errors.append(
                "formatter.crawl_artifact_import.source_run_dir is required"
            )
        if not str(value.get("source_project_name") or "").strip():
            errors.append(
                "formatter.crawl_artifact_import.source_project_name is required"
            )
        if value.get("require_source_audit_ok") is not None and not isinstance(
            value.get("require_source_audit_ok"), bool
        ):
            errors.append(
                "formatter.crawl_artifact_import.require_source_audit_ok must be a boolean"
            )
        evidence = value.get("evidence")
        if evidence is not None:
            if not isinstance(evidence, Mapping):
                errors.append(
                    "formatter.crawl_artifact_import.evidence must be a mapping"
                )
            else:
                supported = {
                    "pipeline_state_sha256",
                    "artifact_catalog_sha256",
                    "run_audit_sha256",
                    "resolved_config_sha256",
                }
                for key, digest in evidence.items():
                    if key not in supported or not _SHA256_RE.fullmatch(
                        str(digest or "").lower()
                    ):
                        errors.append(
                            "formatter.crawl_artifact_import.evidence values must be "
                            "supported SHA-256 digests"
                        )
                        break
        return errors

    async def execute(self, ctx: StageContext) -> StageResult:
        config = _import_config(ctx.config)
        source_run_dir = Path(
            str(config.get("source_run_dir") or "")
        ).expanduser().resolve()
        source_project_name = str(config.get("source_project_name") or "").strip()
        source_stage_id = str(config.get("source_stage_id") or "crawl_web").strip()

        try:
            descriptor = _load_source_descriptor(
                source_run_dir,
                required_stage_ids=[source_stage_id],
                require_audit_ok=bool(config.get("require_source_audit_ok", True)),
                allowed_projects={source_project_name},
                expected_evidence=(
                    config.get("evidence")
                    if isinstance(config.get("evidence"), Mapping)
                    else None
                ),
                source_role="raw_crawl",
            )
            outputs = dict(descriptor.get("outputs") or {})
            missing: List[str] = []
            for key in _REQUIRED_DIRECTORIES:
                path = Path(str(outputs.get(key) or "")).expanduser()
                if not path.is_dir():
                    missing.append(key)
            for key in _REQUIRED_FILES:
                path = Path(str(outputs.get(key) or "")).expanduser()
                if not path.is_file():
                    missing.append(key)
            if missing:
                raise ValueError(
                    "Source crawl is missing required outputs: "
                    + ", ".join(sorted(missing))
                )
        except (OSError, TypeError, ValueError) as exc:
            return StageResult.failure(f"Crawl artifact import failed: {exc}")

        state = descriptor["state"]
        source_stage = next(
            stage
            for stage in state.stages
            if str(stage.stage_id or stage.name) == source_stage_id
        )
        manifest = {
            "schema_version": "mbzuai.crawl_artifact_import.v1",
            "source_run_dir": str(source_run_dir),
            "source_run_id": descriptor["run_id"],
            "source_project_name": descriptor["project_name"],
            "source_stage_id": source_stage_id,
            "source_stage_metrics": dict(source_stage.metrics or {}),
            "source_evidence": dict(descriptor.get("evidence") or {}),
            "outputs": outputs,
        }
        manifest_path = ctx.stage_work_dir / "crawl_artifact_import_manifest.json"
        atomic_write_json(manifest_path, manifest)

        imported_artifacts = []
        for source_record in descriptor["catalog"].records:
            if source_record.producer_stage not in {source_stage_id, source_stage.name}:
                continue
            if not source_record.local_path or not Path(source_record.local_path).is_file():
                continue
            metadata = dict(source_record.metadata or {})
            metadata.update(
                {
                    "imported_from_run_id": descriptor["run_id"],
                    "source_artifact_id": source_record.artifact_id,
                }
            )
            imported_artifacts.append(
                ctx.make_artifact(
                    source_record.local_path,
                    artifact_type=source_record.artifact_type,
                    role=source_record.role,
                    metadata=metadata,
                )
            )
        imported_artifacts.append(
            ctx.make_artifact(
                manifest_path,
                artifact_type="crawl_artifact_import_manifest",
                role="immutable_raw_crawl_handoff",
                metadata={
                    "source_run_id": descriptor["run_id"],
                    "source_project_name": descriptor["project_name"],
                },
            )
        )

        outputs["crawler_runtime_state_file"] = outputs["runtime_state_file"]
        outputs["crawl_artifact_import_manifest_file"] = str(manifest_path)
        return StageResult.success(
            outputs=outputs,
            metrics={
                "imported_artifacts": len(imported_artifacts),
                "source_pages_scraped": int(
                    (source_stage.metrics or {}).get("pages_scraped") or 0
                ),
                "source_documents_downloaded": int(
                    (source_stage.metrics or {}).get("documents_downloaded") or 0
                ),
            },
            artifacts=imported_artifacts,
        )
