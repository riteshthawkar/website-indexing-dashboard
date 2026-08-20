"""
Run integrity auditing for pipeline outputs.

The pipeline previously relied too heavily on stage completion status. This
module validates the actual filesystem and artifact graph so a run only counts
as successful when its outputs are internally consistent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .artifacts import ArtifactCatalog, ArtifactRecord, load_artifact_catalog
from .chunking import load_chunk_index
from .io import atomic_write_json, load_json_safe, sha256_file
from .knowledge_graph import load_graph_bundle, validate_graph_bundle
from .state import PipelineState, load_state, now_iso


_PRUNABLE_CRAWLER_OUTPUT_DIR_KEYS = {"html_dir", "images_dir"}


@dataclass
class AuditIssue:
    severity: str
    code: str
    message: str
    path: str = ""
    stage_id: str = ""
    artifact_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {
            key: value
            for key, value in data.items()
            if value not in ("", [], {}, None)
        }


@dataclass
class RunAuditReport:
    work_dir: str
    checked_at: str
    errors: List[AuditIssue] = field(default_factory=list)
    warnings: List[AuditIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "work_dir": self.work_dir,
            "checked_at": self.checked_at,
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


def _resolve_path_str(value: Any) -> str:
    if not value:
        return ""
    try:
        return str(Path(str(value)).resolve())
    except Exception:
        return ""


def _iter_output_paths(outputs: Dict[str, Any]) -> Iterable[tuple[str, str]]:
    for key, value in (outputs or {}).items():
        if isinstance(value, dict):
            yield from _iter_output_paths(value)
            continue
        if not isinstance(value, str):
            continue
        if key.endswith("_dir") or key.endswith("_file"):
            yield key, value


def _has_completed_downstream_stage(state: PipelineState, stage_index: int) -> bool:
    return any(stage.status == "completed" for stage in state.stages[stage_index + 1 :])


def _is_pruned_crawler_intermediate(
    state: PipelineState,
    stage_index: int,
    stage_type: str,
    output_key: str,
) -> bool:
    return (
        stage_type == "crawler"
        and output_key in _PRUNABLE_CRAWLER_OUTPUT_DIR_KEYS
        and _has_completed_downstream_stage(state, stage_index)
    )


def _quality_report_selected_markdown_path(record: ArtifactRecord) -> str:
    metadata = dict(record.metadata or {})
    resolved = _resolve_path_str(metadata.get("selected_markdown_path"))
    if resolved:
        return resolved
    if not record.local_path:
        return ""
    payload = load_json_safe(record.local_path, {}) or {}
    if not isinstance(payload, dict):
        return ""
    return _resolve_path_str(payload.get("selected_markdown_path"))


def _audit_stage_output_paths(
    report: RunAuditReport,
    state: PipelineState,
) -> None:
    for index, stage in enumerate(state.stages):
        stage_id = stage.stage_id or f"{stage.stage_type}_{stage.name}_{index}"
        for key, value in _iter_output_paths(stage.outputs):
            path = Path(value)
            if key.endswith("_dir") and not path.is_dir():
                if _is_pruned_crawler_intermediate(state, index, stage.stage_type, key):
                    report.warnings.append(
                        AuditIssue(
                            severity="warning",
                            code="pruned_intermediate_dir",
                            message=f"Pruned crawler intermediate directory is missing: {key}",
                            path=str(path),
                            stage_id=stage_id,
                        )
                    )
                    continue
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="missing_output_dir",
                        message=f"Stage output directory is missing: {key}",
                        path=str(path),
                        stage_id=stage_id,
                    )
                )
            elif key.endswith("_file") and not path.is_file():
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="missing_output_file",
                        message=f"Stage output file is missing: {key}",
                        path=str(path),
                        stage_id=stage_id,
                    )
                )


def _audit_artifact_catalog(
    report: RunAuditReport,
    state: PipelineState,
    catalog: ArtifactCatalog,
) -> None:
    live_artifact_ids = catalog.artifact_ids()
    markdown_paths = {
        str(Path(record.local_path).resolve())
        for record in catalog.filter(artifact_type="markdown")
        if record.local_path and Path(record.local_path).is_file()
    }

    for index, stage in enumerate(state.stages):
        stage_id = stage.stage_id or f"{stage.stage_type}_{stage.name}_{index}"
        for artifact_id in stage.artifact_ids or []:
            if artifact_id not in live_artifact_ids:
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="missing_stage_artifact_record",
                        message="Stage references an artifact id that is not in the live catalog",
                        stage_id=stage_id,
                        artifact_id=artifact_id,
                    )
                )

    for record in catalog.records:
        if record.local_path and not Path(record.local_path).exists():
            report.errors.append(
                AuditIssue(
                    severity="error",
                    code="missing_artifact_path",
                    message="Artifact record points to a missing filesystem path",
                    path=str(record.local_path),
                    artifact_id=record.artifact_id,
                    stage_id=record.producer_stage,
                )
            )

        if record.artifact_type == "document_quality_report":
            accepted = bool((record.metadata or {}).get("accepted"))
            selected_markdown_path = _quality_report_selected_markdown_path(record)
            if accepted and not selected_markdown_path:
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="accepted_report_missing_selected_markdown",
                        message="Accepted document quality report has no selected markdown path",
                        path=str(record.local_path or ""),
                        artifact_id=record.artifact_id,
                        stage_id=record.producer_stage,
                    )
                )
            elif accepted and selected_markdown_path not in markdown_paths:
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="accepted_report_missing_markdown_target",
                        message="Accepted document quality report points to missing markdown",
                        path=selected_markdown_path,
                        artifact_id=record.artifact_id,
                        stage_id=record.producer_stage,
                    )
                )

        if record.artifact_type == "structured_document":
            source_markdown_path = _resolve_path_str((record.metadata or {}).get("source_markdown_path"))
            if source_markdown_path and source_markdown_path not in markdown_paths:
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="structured_document_missing_source_markdown",
                        message="Structured document points to missing markdown",
                        path=source_markdown_path,
                        artifact_id=record.artifact_id,
                        stage_id=record.producer_stage,
                    )
                )

        if record.artifact_type == "extracted_image":
            source_document_path = _resolve_path_str((record.metadata or {}).get("source_document_path"))
            if source_document_path and source_document_path not in markdown_paths:
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="extracted_image_missing_source_markdown",
                        message="Extracted image points to missing markdown",
                        path=source_document_path,
                        artifact_id=record.artifact_id,
                        stage_id=record.producer_stage,
                    )
                )


def _audit_mapping_files(report: RunAuditReport, state: PipelineState) -> None:
    for index, stage in enumerate(state.stages):
        mapping_path = stage.outputs.get("md_mapping_file")
        if not mapping_path:
            continue
        stage_id = stage.stage_id or f"{stage.stage_type}_{stage.name}_{index}"
        payload = load_json_safe(mapping_path, {}) or {}
        if not isinstance(payload, dict):
            report.errors.append(
                AuditIssue(
                    severity="error",
                    code="invalid_mapping_payload",
                    message="Markdown mapping file is not a JSON object",
                    path=str(mapping_path),
                    stage_id=stage_id,
                )
            )
            continue
        for source_url, target_path in payload.items():
            if not isinstance(target_path, str) or not Path(target_path).is_file():
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="mapping_points_to_missing_file",
                        message="Markdown mapping points to a missing file",
                        path=str(target_path),
                        stage_id=stage_id,
                        metadata={"source_url": str(source_url)},
                    )
                )


def _audit_chunk_indexes(report: RunAuditReport, catalog: ArtifactCatalog) -> None:
    for record in catalog.filter(artifact_type="chunk_index"):
        if not record.local_path or not Path(record.local_path).is_file():
            continue
        payload = load_chunk_index(record.local_path)
        for chunk in payload.get("chunks") or []:
            source_markdown_path = _resolve_path_str(chunk.get("source_markdown_path"))
            if not source_markdown_path:
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="chunk_missing_source_markdown",
                        message="Chunk record is missing source_markdown_path",
                        path=str(record.local_path),
                        artifact_id=record.artifact_id,
                        stage_id=record.producer_stage,
                        metadata={"chunk_id": str(chunk.get("chunk_id") or "")},
                    )
                )
                continue
            if not Path(source_markdown_path).is_file():
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code="chunk_missing_source_markdown_file",
                        message="Chunk record points to a missing markdown file",
                        path=source_markdown_path,
                        artifact_id=record.artifact_id,
                        stage_id=record.producer_stage,
                        metadata={"chunk_id": str(chunk.get("chunk_id") or "")},
                    )
                )


def _audit_knowledge_graphs(report: RunAuditReport, catalog: ArtifactCatalog) -> None:
    graph_artifact_types = (
        ("knowledge_graph_bundle", False),
        ("canonical_page_link_graph", True),
    )
    for artifact_type, require_stats in graph_artifact_types:
        for record in catalog.filter(artifact_type=artifact_type):
            if not record.local_path or not Path(record.local_path).is_file():
                continue
            bundle = load_graph_bundle(record.local_path)
            for issue in validate_graph_bundle(bundle, require_stats=require_stats):
                report.errors.append(
                    AuditIssue(
                        severity="error",
                        code=str(issue.get("code") or "invalid_graph_artifact"),
                        message=str(issue.get("message") or "Graph artifact is invalid"),
                        path=str(record.local_path),
                        artifact_id=record.artifact_id,
                        stage_id=record.producer_stage,
                        metadata={
                            key: value
                            for key, value in issue.items()
                            if key not in {"code", "message"}
                        },
                    )
                )


def _audit_index_coverage_gate(report: RunAuditReport, work_dir: Path) -> None:
    """Fail a run audit when its own site-coverage gate did not pass.

    Older release checks validated downstream retrieval quality without
    carrying the crawler's coverage decision into the final run audit.  That
    allowed a partially indexed site to be promoted when benchmark queries
    happened to target the pages that were present.  When a run contains the
    MBZUAI coverage artifact, it is authoritative and must pass.
    """

    gate_path = (
        work_dir
        / "stage_outputs"
        / "prepare_mbzuai_index"
        / "index_coverage_gate.json"
    )
    if not gate_path.exists():
        return

    gate = load_json_safe(gate_path)
    if not isinstance(gate, dict):
        report.errors.append(
            AuditIssue(
                severity="error",
                code="invalid_index_coverage_gate",
                message="Index coverage gate is unreadable or is not a JSON object",
                path=str(gate_path),
                stage_id="prepare_mbzuai_index",
            )
        )
        return

    if gate.get("ok") is True:
        return

    report.errors.append(
        AuditIssue(
            severity="error",
            code="index_coverage_gate_failed",
            message="Website index coverage gate did not pass",
            path=str(gate_path),
            stage_id="prepare_mbzuai_index",
            metadata={
                key: gate.get(key)
                for key in (
                    "missing_critical_count",
                    "unhealthy_critical_count",
                    "hard_failure_count",
                    "expected_site_inventory_count",
                    "effective_expected_inventory_count",
                    "minimum_inventory_coverage_ratio",
                    "inventory_coverage_ratio",
                    "inventory_gap",
                )
                if key in gate
            },
        )
    )


def _audit_content_disposition_manifests(
    report: RunAuditReport,
    state: PipelineState,
) -> None:
    """Require auditable, fail-closed manifests for quality and cleaning stages."""

    for index, stage in enumerate(state.stages):
        if stage.status != "completed":
            continue
        if stage.name == "quality_scorer":
            manifest_key = "quality_manifest_file"
            manifest_label = "quality"
            accepted_output_key = "passed_count"
        elif stage.stage_type == "cleaner":
            manifest_key = "cleaning_manifest_file"
            manifest_label = "cleaning"
            accepted_output_key = "cleaned_count"
        else:
            continue

        stage_id = stage.stage_id or f"{stage.stage_type}_{stage.name}_{index}"
        manifest_path_value = stage.outputs.get(manifest_key)
        if not manifest_path_value:
            report.errors.append(
                AuditIssue(
                    severity="error",
                    code=f"missing_{manifest_label}_manifest",
                    message=f"Completed {manifest_label} stage has no disposition manifest",
                    stage_id=stage_id,
                )
            )
            continue

        manifest_path = Path(str(manifest_path_value))
        manifest = load_json_safe(manifest_path)
        if not isinstance(manifest, dict):
            report.errors.append(
                AuditIssue(
                    severity="error",
                    code=f"invalid_{manifest_label}_manifest",
                    message=f"{manifest_label.title()} disposition manifest is unreadable",
                    path=str(manifest_path),
                    stage_id=stage_id,
                )
            )
            continue

        gate = manifest.get("gate")
        dispositions = manifest.get("dispositions")
        if (
            manifest.get("application_status") != "completed"
            or not isinstance(gate, dict)
            or gate.get("ok") is not True
            or not isinstance(dispositions, list)
        ):
            report.errors.append(
                AuditIssue(
                    severity="error",
                    code=f"failed_{manifest_label}_manifest_gate",
                    message=f"{manifest_label.title()} manifest does not prove a passed gate",
                    path=str(manifest_path),
                    stage_id=stage_id,
                )
            )
            continue

        status_counts: Dict[str, int] = {"accepted": 0, "filtered": 0, "failed": 0}
        invalid_dispositions = 0
        missing_outputs = 0
        missing_sources = 0
        in_place_outputs = 0
        for item in dispositions:
            if not isinstance(item, dict):
                invalid_dispositions += 1
                continue
            status = str(item.get("status") or "")
            reason_code = str(item.get("reason_code") or "")
            if status not in status_counts or not reason_code:
                invalid_dispositions += 1
                continue
            status_counts[status] += 1
            source_path = _resolve_path_str(item.get("source_path"))
            if source_path and not Path(source_path).is_file():
                missing_sources += 1
            if status == "accepted":
                output_path = _resolve_path_str(item.get("output_path"))
                if not output_path or not Path(output_path).is_file():
                    missing_outputs += 1
                if manifest_label == "quality" and source_path == output_path:
                    in_place_outputs += 1

        manifest_counts = {
            "input_count": len(dispositions),
            "accepted_count": status_counts["accepted"],
            "filtered_count": status_counts["filtered"],
            "failed_count": status_counts["failed"],
        }
        inconsistent_counts = {
            key: {"manifest": value, "gate": gate.get(key)}
            for key, value in manifest_counts.items()
            if gate.get(key) != value
        }
        output_accepted_count = stage.outputs.get(accepted_output_key)
        if output_accepted_count != status_counts["accepted"]:
            inconsistent_counts[accepted_output_key] = {
                "manifest": status_counts["accepted"],
                "stage_output": output_accepted_count,
            }
        if (
            invalid_dispositions
            or inconsistent_counts
            or missing_outputs
            or missing_sources
            or in_place_outputs
        ):
            report.errors.append(
                AuditIssue(
                    severity="error",
                    code=f"inconsistent_{manifest_label}_manifest",
                    message=f"{manifest_label.title()} disposition evidence is inconsistent",
                    path=str(manifest_path),
                    stage_id=stage_id,
                    metadata={
                        "invalid_dispositions": invalid_dispositions,
                        "inconsistent_counts": inconsistent_counts,
                        "missing_outputs": missing_outputs,
                        "missing_sources": missing_sources,
                        "in_place_outputs": in_place_outputs,
                    },
                )
            )


def audit_run(
    work_dir: str | Path,
    *,
    state: Optional[PipelineState] = None,
    artifact_catalog: Optional[ArtifactCatalog] = None,
) -> RunAuditReport:
    work_dir = Path(work_dir).resolve()
    state = state or load_state(work_dir)
    artifact_catalog = artifact_catalog or load_artifact_catalog(work_dir)
    report = RunAuditReport(work_dir=str(work_dir), checked_at=now_iso())

    if state is None:
        report.errors.append(
            AuditIssue(
                severity="error",
                code="missing_pipeline_state",
                message="Run is missing pipeline_state.json",
                path=str(work_dir / "pipeline_state.json"),
            )
        )
        return report

    if state.status == "completed" and any(not stage.is_terminal for stage in state.stages):
        report.errors.append(
            AuditIssue(
                severity="error",
                code="completed_run_has_unfinished_stage",
                message="Pipeline is marked completed while at least one stage is unfinished",
            )
        )

    _audit_stage_output_paths(report, state)
    _audit_artifact_catalog(report, state, artifact_catalog)
    _audit_mapping_files(report, state)
    _audit_content_disposition_manifests(report, state)
    _audit_chunk_indexes(report, artifact_catalog)
    _audit_knowledge_graphs(report, artifact_catalog)
    _audit_index_coverage_gate(report, work_dir)
    _audit_retrieval_index_manifest(report, work_dir)

    return report


def _audit_retrieval_index_manifest(report: RunAuditReport, work_dir: Path) -> None:
    manifest_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    if not manifest_path.exists():
        return

    manifest = load_json_safe(manifest_path, {}) or {}
    if not isinstance(manifest, dict):
        report.errors.append(
            AuditIssue(
                severity="error",
                code="invalid_index_upload_manifest",
                message="Index upload manifest is unreadable or invalid",
                path=str(manifest_path),
            )
        )
        return

    bundle_candidates: List[Path] = []
    manifest_bundle_file = str(manifest.get("retrieval_bundle_file") or "").strip()
    if manifest_bundle_file:
        manifest_bundle_path = Path(manifest_bundle_file)
        if not manifest_bundle_path.is_absolute():
            manifest_bundle_path = work_dir / manifest_bundle_path
        bundle_candidates.append(manifest_bundle_path)
    bundle_candidates.extend(
        [
            work_dir / "stage_outputs" / "finalize_retrieval_bundle" / "retrieval_bundle.json",
            work_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
            work_dir / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json",
        ]
    )
    bundle_path = next((candidate for candidate in bundle_candidates if candidate.exists()), None)
    if bundle_path is None:
        report.errors.append(
            AuditIssue(
                severity="error",
                code="index_manifest_bundle_missing",
                message="Index upload manifest records a bundle fingerprint but no retrieval bundle file is available for verification",
                path=str(manifest_path),
                metadata={
                    "retrieval_bundle_file": manifest_bundle_file,
                    "checked_paths": [str(candidate) for candidate in bundle_candidates],
                },
            )
        )
        return

    recorded_sha = str(manifest.get("retrieval_bundle_sha256") or "").strip()
    if not recorded_sha:
        report.warnings.append(
            AuditIssue(
                severity="warning",
                code="index_manifest_missing_bundle_fingerprint",
                message="Index upload manifest does not record a retrieval bundle fingerprint",
                path=str(manifest_path),
            )
        )
        return

    current_sha = sha256_file(bundle_path)
    if current_sha != recorded_sha:
        report.errors.append(
            AuditIssue(
                severity="error",
                code="index_manifest_bundle_mismatch",
                message="Current retrieval bundle does not match the bundle used for indexing",
                path=str(manifest_path),
                metadata={
                    "recorded_sha256": recorded_sha,
                    "current_sha256": current_sha,
                    "bundle_path": str(bundle_path),
                },
            )
        )


def reconcile_state_artifact_ids(state: Optional[PipelineState], artifact_catalog: ArtifactCatalog) -> int:
    if state is None:
        return 0
    live_artifact_ids = artifact_catalog.artifact_ids()
    removed = 0
    for stage in state.stages:
        original = list(stage.artifact_ids or [])
        filtered = [artifact_id for artifact_id in original if artifact_id in live_artifact_ids]
        removed += len(original) - len(filtered)
        stage.artifact_ids = filtered
    return removed


def save_run_audit(report: RunAuditReport, work_dir: str | Path) -> Path:
    path = Path(work_dir) / "run_audit.json"
    atomic_write_json(path, report.to_dict())
    return path
