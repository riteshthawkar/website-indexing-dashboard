"""Pipeline stage for the cleaned Representation V2 page graph bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.io import atomic_write_json, load_json_safe, sha256_file
from pipeline.core.page_graph_bridge import (
    NAVIGATION_CATALOG_SCHEMA_VERSION,
    PAGE_GRAPH_BRIDGE_SCHEMA_VERSION,
    build_navigation_catalog,
    build_page_graph_bridge,
    validate_page_graph_bridge,
)
from pipeline.core.registry import register_stage


def _resolved_path(value: Any) -> Path | None:
    raw = str(value or "").strip()
    return Path(raw).expanduser().resolve() if raw else None


def _configured_file(
    config: Mapping[str, Any],
    key: str,
    *,
    run_key: str,
    stage_id: str,
    filename: str,
) -> Path | None:
    explicit = _resolved_path(config.get(key))
    if explicit is not None:
        return explicit
    run_dir = _resolved_path(config.get(run_key))
    if run_dir is None:
        return None
    return run_dir / "stage_outputs" / stage_id / filename


@register_stage
class PageGraphBridgeFormatter(FormatterStage):
    name = "page_graph_bridge"
    description = (
        "Removes external discovery noise from the crawl graph and links Page "
        "Cards to document revisions, sections, actions, and optional chunks."
    )

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        formatter = config.get("formatter") if isinstance(config.get("formatter"), dict) else {}
        stage_config = formatter.get("page_graph_bridge") if isinstance(formatter, dict) else {}
        if not isinstance(stage_config, dict):
            return ["formatter.page_graph_bridge must be a mapping"]
        return []

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.formatter_config.get("page_graph_bridge") or {}
        if not isinstance(config, dict):
            return StageResult.failure("formatter.page_graph_bridge must be a mapping")

        try:
            representation_file = _configured_file(
                config,
                "representation_v2_bundle_file",
                run_key="source_representation_run_dir",
                stage_id=str(
                    config.get("representation_stage_id")
                    or "extract_page_cards_and_actions"
                ),
                filename="representation_v2_bundle.json",
            ) or _resolved_path(ctx.previous_outputs.get("representation_v2_bundle_file"))
            crawl_graph_file = _configured_file(
                config,
                "crawl_graph_file",
                run_key="source_corpus_run_dir",
                stage_id=str(
                    config.get("corpus_boundary_stage_id")
                    or "prepare_corpus_boundary"
                ),
                filename="prepared_canonical_page_link_graph.json",
            ) or _resolved_path(ctx.previous_outputs.get("prepared_canonical_page_link_graph_file"))
            chunk_index_file = _resolved_path(
                config.get("chunk_index_file")
                or ctx.previous_outputs.get("chunk_index_file")
                or ctx.previous_outputs.get("chunks_file")
            )
            require_chunk_index = bool(config.get("require_chunk_index", False))
            derive_document_sections = bool(
                config.get("derive_document_sections", False)
            )

            for label, path in (
                ("Representation V2 bundle", representation_file),
                ("prepared crawl graph", crawl_graph_file),
            ):
                if path is None or not path.is_file():
                    raise FileNotFoundError(f"{label} is missing: {path}")
            if chunk_index_file is not None and not chunk_index_file.is_file():
                raise FileNotFoundError(f"Chunk index is missing: {chunk_index_file}")

            representation = load_json_safe(representation_file, None)
            crawl_graph = load_json_safe(crawl_graph_file, None)
            chunk_index = (
                load_json_safe(chunk_index_file, None)
                if chunk_index_file is not None
                else None
            )
            if not isinstance(representation, Mapping):
                raise ValueError("Representation V2 bundle must be a JSON object")
            if not isinstance(crawl_graph, Mapping):
                raise ValueError("Prepared crawl graph must be a JSON object")
            if chunk_index is not None and not isinstance(chunk_index, Mapping):
                raise ValueError("Chunk index must be a JSON object")

            bridge = build_page_graph_bridge(
                representation_bundle=representation,
                crawl_graph=crawl_graph,
                chunk_index=chunk_index,
                require_chunk_index=require_chunk_index,
                derive_document_sections=derive_document_sections,
            )
            bridge["source_snapshot"].update(
                {
                    "representation_v2_bundle_file": str(representation_file),
                    "representation_v2_bundle_sha256": sha256_file(representation_file),
                    "crawl_graph_file": str(crawl_graph_file),
                    "crawl_graph_sha256": sha256_file(crawl_graph_file),
                    "chunk_index_file": str(chunk_index_file or ""),
                    "chunk_index_sha256": (
                        sha256_file(chunk_index_file) if chunk_index_file else ""
                    ),
                }
            )
            validation = validate_page_graph_bridge(
                bridge, require_chunk_index=require_chunk_index
            )
            coverage = dict(bridge.get("coverage") or {})
            coverage["independent_validation"] = validation
            coverage["passed"] = bool(coverage.get("passed")) and bool(
                validation.get("passed")
            )
            bridge["coverage"] = coverage
            navigation_catalog = build_navigation_catalog(bridge)

            bridge_path = ctx.stage_work_dir / "page_graph_bridge.json"
            coverage_path = ctx.stage_work_dir / "page_graph_bridge_coverage.json"
            navigation_catalog_path = (
                ctx.stage_work_dir / "page_graph_navigation_catalog.json"
            )
            atomic_write_json(bridge_path, bridge, indent=None)
            atomic_write_json(
                coverage_path,
                {
                    "schema_version": PAGE_GRAPH_BRIDGE_SCHEMA_VERSION,
                    "kind": "page_graph_bridge_coverage",
                    "source_snapshot": bridge["source_snapshot"],
                    "stats": bridge["stats"],
                    "coverage": coverage,
                },
            )
            atomic_write_json(
                navigation_catalog_path, navigation_catalog, indent=None
            )
            outputs = {
                "page_graph_bridge_file": str(bridge_path),
                "page_graph_bridge_coverage_file": str(coverage_path),
                "page_graph_navigation_catalog_file": str(
                    navigation_catalog_path
                ),
                "page_graph_bridge_status": coverage.get("status"),
                "page_graph_bridge_complete": bool(coverage.get("passed")),
                "chunk_bridge_ready": bool(
                    (coverage.get("chunk_gates") or {}).get("chunk_bridge_ready")
                ),
                "chunking_performed": False,
                "embedding_performed": False,
                "indexing_performed": False,
            }
            artifacts = [
                ctx.make_artifact(
                    bridge_path,
                    artifact_type="page_graph_bridge",
                    role="retrieval_navigation_graph",
                    metadata={
                        "schema_version": PAGE_GRAPH_BRIDGE_SCHEMA_VERSION,
                        "status": coverage.get("status"),
                        "pages": int((bridge.get("stats") or {}).get("pages") or 0),
                        "chunks": int((bridge.get("stats") or {}).get("chunks") or 0),
                    },
                ),
                ctx.make_artifact(
                    coverage_path,
                    artifact_type="page_graph_bridge_coverage",
                    role="quality_report",
                    metadata={
                        "passed": bool(coverage.get("passed")),
                        "chunk_bridge_ready": bool(
                            (coverage.get("chunk_gates") or {}).get(
                                "chunk_bridge_ready"
                            )
                        ),
                    },
                ),
                ctx.make_artifact(
                    navigation_catalog_path,
                    artifact_type="page_graph_navigation_catalog",
                    role="retrieval_navigation_runtime",
                    metadata={
                        "schema_version": NAVIGATION_CATALOG_SCHEMA_VERSION,
                        **dict(navigation_catalog.get("stats") or {}),
                    },
                ),
            ]
            metrics = {
                **dict(bridge.get("stats") or {}),
                "coverage_passed": bool(coverage.get("passed")),
                "chunk_bridge_ready": outputs["chunk_bridge_ready"],
            }
            if not coverage.get("passed"):
                issue = next(
                    iter(
                        (validation.get("issue_samples") or [])
                        or (coverage.get("chunk_issue_samples") or [])
                    ),
                    {},
                )
                return StageResult.failure(
                    "Page graph bridge failed coverage gates"
                    + (f": {issue.get('message') or issue.get('code')}" if issue else ""),
                    outputs=outputs,
                    metrics=metrics,
                    artifacts=artifacts,
                )
            return StageResult.success(
                outputs=outputs,
                metrics=metrics,
                artifacts=artifacts,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return StageResult.failure(f"Page graph bridge failed: {exc}")
