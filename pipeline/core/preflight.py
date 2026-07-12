"""Production-readiness checks for terminal pipeline operations."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from pipeline.core.config import configured_secret_paths


CORE_PRODUCTION_STAGE_ORDER = [
    "crawl_web",
    "prepare_mbzuai_index",
    "score_raw_content",
    "clean_html",
    "convert_documents",
    "convert_html",
    "deduplicate_markdown",
    "chunk_content",
]

ASSERTION_FIRST_STAGE_ORDER = [
    *CORE_PRODUCTION_STAGE_ORDER,
    "format_assertion_slices",
    "extract_assertions_openai",
    "validate_assertions_openai",
    "canonicalize_assertions",
    "promote_assertions",
    "format_retrieval",
    "format_graph",
    "promote_graph",
    "community_graph",
    "summarize_community_graph",
    "upload_retrieval",
]

SEMANTIC_GRAPH_STAGE_ORDER = [
    *CORE_PRODUCTION_STAGE_ORDER,
    "build_retrieval_bundle",
    "format_graph",
    "extract_semantic_graph",
    "canonicalize_semantic_graph",
    "promote_graph",
    "finalize_retrieval_bundle",
    "upload_retrieval",
]

LEGACY_STAGE_PLUGINS = {
    "mbzuai_legacy_vectorstores",
    "mbzuai_legacy_pinecone",
    "gliner_extract",
    "semantic_graph_extract",
    "semantic_graph_canonicalize",
}


@dataclass
class PreflightCheck:
    name: str
    status: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            key: value
            for key, value in asdict(self).items()
            if value not in ("", [], {}, None)
        }


def _stage_ids(config: Mapping[str, Any]) -> List[str]:
    ids: List[str] = []
    for index, stage in enumerate(config.get("stages") or []):
        if not isinstance(stage, Mapping):
            continue
        ids.append(str(stage.get("id") or stage.get("plugin") or f"stage_{index}"))
    return ids


def _stage_plugins(config: Mapping[str, Any]) -> List[str]:
    plugins: List[str] = []
    for stage in config.get("stages") or []:
        if isinstance(stage, Mapping) and stage.get("plugin"):
            plugins.append(str(stage["plugin"]))
    return plugins


def _graph_store_backend(config: Mapping[str, Any], stage_plugins: Iterable[str]) -> str:
    graph_cfg = config.get("graph", {}) if isinstance(config.get("graph"), Mapping) else {}
    configured = str(
        graph_cfg.get("store_backend")
        or graph_cfg.get("graph_store_backend")
        or ""
    ).strip().lower()
    if configured in {"none", "off", "disabled"}:
        return "disabled"
    if configured in {"local", "local_json", "json", "file", "files"}:
        return "local_json"
    if configured == "neo4j":
        return "neo4j"
    if bool(graph_cfg.get("require_neo4j_upload", False)) or "neo4j_graph_store" in set(stage_plugins):
        return "neo4j"
    return "local_json"


def _production_stage_profile(config: Mapping[str, Any], stage_plugins: Iterable[str]) -> tuple[str, List[str]]:
    plugins = set(stage_plugins)
    stage_ids = set(_stage_ids(config))
    if {"semantic_graph_extract", "semantic_graph_canonicalize"}.intersection(plugins):
        order = [
            stage_id
            for stage_id in SEMANTIC_GRAPH_STAGE_ORDER
            if stage_id != "build_retrieval_bundle" or "build_retrieval_bundle" in stage_ids
        ]
        if "build_retrieval_bundle" not in stage_ids and "format_retrieval" in stage_ids:
            order.insert(len(CORE_PRODUCTION_STAGE_ORDER), "format_retrieval")
        return "semantic-graph/v5", order

    order = list(ASSERTION_FIRST_STAGE_ORDER)
    if _graph_store_backend(config, stage_plugins) == "neo4j":
        order.append("upload_graph")
    return "assertion-first", order


def _production_stage_order(config: Mapping[str, Any], stage_plugins: Iterable[str]) -> List[str]:
    return _production_stage_profile(config, stage_plugins)[1]


def _env_or_config(section: Mapping[str, Any], key: str, env_name: str) -> str:
    value = str(section.get(key) or "").strip()
    if value:
        return value
    return str(os.environ.get(env_name) or "").strip()


def _has_any_env(names: Iterable[str]) -> bool:
    return any(str(os.environ.get(name) or "").strip() for name in names)


def _add(checks: List[PreflightCheck], name: str, status: str, message: str, **details: Any) -> None:
    checks.append(PreflightCheck(name=name, status=status, message=message, details=details))


def assess_production_readiness(
    config: Mapping[str, Any],
    *,
    config_name: str = "",
    validation_errors: Optional[Mapping[str, List[str]]] = None,
) -> Dict[str, Any]:
    """Return structured CLI preflight results for a production indexing run."""

    checks: List[PreflightCheck] = []
    stage_ids = _stage_ids(config)
    stage_plugins = _stage_plugins(config)
    graph_backend = _graph_store_backend(config, stage_plugins)

    secret_paths = configured_secret_paths(config)
    if secret_paths:
        _add(
            checks,
            "config_secret_hygiene",
            "error",
            "Production config must reference credentials from environment variables, not embed them.",
            paths=secret_paths,
        )
    else:
        _add(
            checks,
            "config_secret_hygiene",
            "ok",
            "No secret-bearing values are embedded in the effective config.",
        )

    stage_profile, expected_stage_order = _production_stage_profile(config, stage_plugins)
    missing_stages = [stage_id for stage_id in expected_stage_order if stage_id not in stage_ids]
    ordered_positions = [stage_ids.index(stage_id) for stage_id in expected_stage_order if stage_id in stage_ids]
    if missing_stages:
        _add(
            checks,
            "stage_order",
            "error",
            f"Production {stage_profile} stages are missing.",
            missing=missing_stages,
        )
    elif ordered_positions != sorted(ordered_positions):
        _add(
            checks,
            "stage_order",
            "error",
            "Production stages are present but not in dependency order.",
            expected=expected_stage_order,
            actual=stage_ids,
        )
    else:
        _add(checks, "stage_order", "ok", f"Production {stage_profile} stage order is complete.")

    legacy_plugins = sorted(set(stage_plugins).intersection(LEGACY_STAGE_PLUGINS))
    if legacy_plugins:
        _add(
            checks,
            "legacy_stages",
            "warn",
            "Legacy or experimental graph stages are still enabled.",
            plugins=legacy_plugins,
        )
    else:
        _add(checks, "legacy_stages", "ok", "No legacy vectorstore or experimental graph stages are enabled.")

    pipeline_cfg = config.get("pipeline", {}) if isinstance(config.get("pipeline"), Mapping) else {}
    crawler_cfg = config.get("crawler", {}) if isinstance(config.get("crawler"), Mapping) else {}
    if bool(crawler_cfg.get("ignore_https_errors", False)) or not bool(
        crawler_cfg.get("require_https", True)
    ):
        _add(
            checks,
            "crawler_tls_verification",
            "error",
            "Production crawling must require HTTPS and verify certificates.",
        )
    else:
        _add(
            checks,
            "crawler_tls_verification",
            "ok",
            "Crawler HTTPS-only fetching and certificate verification are enabled.",
        )
    canonical_production_requested = Path(str(config_name or "")).stem == "mbzuai_production"
    if canonical_production_requested:
        production_contract_errors: List[str] = []
        if not bool(pipeline_cfg.get("production_profile", False)):
            production_contract_errors.append("pipeline.production_profile must be true")
        if not bool(pipeline_cfg.get("require_assertion_first", False)):
            production_contract_errors.append("pipeline.require_assertion_first must be true")
        if not bool(pipeline_cfg.get("require_query_planner", False)):
            production_contract_errors.append("pipeline.require_query_planner must be true")
        retrieval_contract = config.get("retrieval", {}) if isinstance(config.get("retrieval"), Mapping) else {}
        if str(retrieval_contract.get("retriever_backend") or "").strip() != "routed_hybrid":
            production_contract_errors.append("retrieval.retriever_backend must be routed_hybrid")
        if not bool(retrieval_contract.get("routed_graph_required", False)):
            production_contract_errors.append("retrieval.routed_graph_required must be true")
        if not bool(retrieval_contract.get("evidence_adjudicator_enabled", False)):
            production_contract_errors.append("retrieval.evidence_adjudicator_enabled must be true")
        if not str(retrieval_contract.get("evidence_adjudicator_model") or "").strip():
            production_contract_errors.append("retrieval.evidence_adjudicator_model must be configured")
        if production_contract_errors:
            _add(
                checks,
                "canonical_production_contract",
                "error",
                "The effective run config does not satisfy the canonical production contract.",
                errors=production_contract_errors,
            )
        else:
            _add(
                checks,
                "canonical_production_contract",
                "ok",
                "The effective run config satisfies the canonical production contract.",
            )
    audit_required = (
        bool(pipeline_cfg.get("audit_on_stage_complete", True))
        and bool(pipeline_cfg.get("audit_on_run_complete", True))
        and bool(pipeline_cfg.get("fail_on_audit_error", True))
    )
    if audit_required:
        _add(checks, "auditing", "ok", "Stage and run audits are enabled and fail the run on errors.")
    else:
        _add(
            checks,
            "auditing",
            "error",
            "Production runs must keep stage/run audits enabled with fail_on_audit_error.",
        )

    if bool(pipeline_cfg.get("require_assertion_first", False)):
        required_assertion_plugins = {
            "extraction_slices",
            "openai_assertion_extract",
            "openai_assertion_validate",
            "assertion_canonicalize",
            "assertion_promote",
            "retrieval_bundle_v2",
        }
        missing_assertion_plugins = sorted(required_assertion_plugins - set(stage_plugins))
        if missing_assertion_plugins:
            _add(
                checks,
                "assertion_first_contract",
                "error",
                "The production profile requires the complete validated assertion-first pipeline.",
                missing_plugins=missing_assertion_plugins,
            )
        else:
            _add(checks, "assertion_first_contract", "ok", "Validated assertion-first stages are enabled.")

    embedder_cfg = config.get("embedder", {}) if isinstance(config.get("embedder"), Mapping) else {}
    graph_cfg = config.get("graph", {}) if isinstance(config.get("graph"), Mapping) else {}
    retrieval_cfg = config.get("retrieval", {}) if isinstance(config.get("retrieval"), Mapping) else {}

    if bool(pipeline_cfg.get("require_query_planner", False)):
        if bool(retrieval_cfg.get("query_planner_enabled", False)) and str(
            retrieval_cfg.get("query_planner_model") or ""
        ).strip():
            _add(checks, "query_planner_contract", "ok", "Production query planning is enabled and model-driven.")
        else:
            _add(
                checks,
                "query_planner_contract",
                "error",
                "The production profile requires retrieval.query_planner_enabled and a configured planner model.",
            )

    if "openai_assertion_extract" in stage_plugins or "openai_assertion_validate" in stage_plugins:
        if os.environ.get("OPENAI_API_KEY"):
            _add(checks, "openai_credentials", "ok", "OPENAI_API_KEY is available.")
        else:
            _add(checks, "openai_credentials", "error", "OPENAI_API_KEY is required for assertion extraction and validation.")

    if "gemini_pinecone" in stage_plugins:
        if _has_any_env(("GOOGLE_API_KEY", "GEMINI_API_KEY")):
            _add(checks, "gemini_credentials", "ok", "GOOGLE_API_KEY or GEMINI_API_KEY is available.")
        else:
            _add(checks, "gemini_credentials", "error", "GOOGLE_API_KEY or GEMINI_API_KEY is required for Gemini embeddings.")

        if os.environ.get("PINECONE_API_KEY"):
            _add(checks, "pinecone_credentials", "ok", "PINECONE_API_KEY is available.")
        else:
            _add(checks, "pinecone_credentials", "error", "PINECONE_API_KEY is required for Pinecone upload.")

        pinecone_index = str(embedder_cfg.get("pinecone_index") or "").strip()
        sparse_index = str(embedder_cfg.get("pinecone_sparse_index") or "").strip()
        sparse_required = bool(embedder_cfg.get("enable_sparse", True))
        required_namespaces = [
            "namespace_chunks",
            "namespace_parents",
            "namespace_media",
            "namespace_facts",
            "namespace_assertions",
        ]
        if "gemini_retrieval" in set(stage_plugins):
            required_namespaces.extend(
                [
                    "namespace_evidence_spans",
                    "namespace_summaries",
                    "namespace_entities",
                    "namespace_communities",
                ]
            )
        missing_namespaces = [
            key
            for key in required_namespaces
            if not str(embedder_cfg.get(key) or "").strip()
        ]
        if pinecone_index and (sparse_index or not sparse_required) and not missing_namespaces:
            _add(
                checks,
                "pinecone_targets",
                "ok",
                "Pinecone vector targets are configured.",
                dense_index=pinecone_index,
                sparse_index=sparse_index,
                sparse_required=sparse_required,
            )
        else:
            _add(
                checks,
                "pinecone_targets",
                "error",
                "Pinecone dense index, required sparse index, and production namespaces must be configured.",
                dense_index=pinecone_index,
                sparse_index=sparse_index,
                sparse_required=sparse_required,
                missing_namespaces=missing_namespaces,
            )

        namespace_strategy = str(embedder_cfg.get("namespace_strategy") or "static").strip().lower()
        if namespace_strategy == "release":
            from pipeline.stages.embedders.gemini_pinecone_embedder import _resolve_upload_namespaces

            try:
                first = _resolve_upload_namespaces(embedder_cfg, run_id="preflight/candidate")
                second = _resolve_upload_namespaces(embedder_cfg, run_id="preflight-candidate")
                if any(first.get(lane) == second.get(lane) for lane in first):
                    raise ValueError("distinct run IDs resolve to the same namespace")
            except (KeyError, ValueError) as exc:
                _add(
                    checks,
                    "pinecone_release_isolation",
                    "error",
                    "Release-scoped Pinecone namespace configuration is not collision-safe.",
                    error=str(exc),
                )
            else:
                _add(
                    checks,
                    "pinecone_release_isolation",
                    "ok",
                    "Candidate uploads use identity-bound release-scoped Pinecone namespaces.",
                )
        else:
            _add(
                checks,
                "pinecone_release_isolation",
                "error",
                "Production uploads must use embedder.namespace_strategy=release so candidate runs cannot mutate active namespaces.",
                namespace_strategy=namespace_strategy,
            )

        if bool(embedder_cfg.get("verify_index_after_upload", False)) and not bool(
            embedder_cfg.get("verify_index_min_count_only", False)
        ):
            _add(checks, "pinecone_verification", "ok", "Exact Pinecone namespace count verification is enabled after upload.")
        else:
            _add(
                checks,
                "pinecone_verification",
                "error",
                "Production uploads must verify exact Pinecone namespace counts after upload.",
            )

    if graph_backend == "local_json":
        _add(checks, "graph_store", "ok", "Knowledge graph will be verified from promoted local JSON artifacts.")
    elif graph_backend == "disabled":
        _add(checks, "graph_store", "warn", "Knowledge graph storage is disabled for this config.")

    if graph_backend == "neo4j":
        missing_neo4j = [
            label
            for label, value in (
                ("NEO4J_URI", _env_or_config(graph_cfg, "neo4j_uri", "NEO4J_URI")),
                ("NEO4J_USERNAME", _env_or_config(graph_cfg, "neo4j_username", "NEO4J_USERNAME")),
                ("NEO4J_PASSWORD", _env_or_config(graph_cfg, "neo4j_password", "NEO4J_PASSWORD")),
            )
            if not value
        ]
        if missing_neo4j:
            _add(checks, "neo4j_credentials", "error", "Neo4j credentials are required for graph upload.", missing=missing_neo4j)
        else:
            namespace = _env_or_config(graph_cfg, "neo4j_namespace", "NEO4J_NAMESPACE")
            status = "ok" if namespace else "warn"
            message = "Neo4j target is configured." if namespace else "Neo4j will use a generated per-run namespace."
            _add(checks, "neo4j_credentials", status, message, namespace=namespace or "<project:run_id>")

        if bool(graph_cfg.get("neo4j_verify_after_upload", False)):
            _add(checks, "neo4j_verification", "ok", "Neo4j node/edge count verification is enabled after upload.")
        else:
            _add(checks, "neo4j_verification", "error", "Production graph uploads must verify Neo4j node and edge counts after upload.")

    if "semantic_graph_extract" in stage_plugins and bool(graph_cfg.get("extraction_fail_open_after_retries", True)):
        _add(
            checks,
            "semantic_graph_failure_mode",
            "error",
            "semantic_graph_extract is fail-open; production extraction must fail closed or use an explicit error budget.",
        )

    if validation_errors is None:
        _add(checks, "stage_validation", "warn", "Stage validation was skipped.")
    elif validation_errors:
        for stage, errors in validation_errors.items():
            for error in errors:
                _add(checks, "stage_validation", "error", f"{stage}: {error}")
    else:
        _add(checks, "stage_validation", "ok", "Stage validation returned no errors.")

    error_count = sum(1 for check in checks if check.status == "error")
    warning_count = sum(1 for check in checks if check.status == "warn")
    return {
        "config": config_name,
        "ok": error_count == 0,
        "error_count": error_count,
        "warning_count": warning_count,
        "checks": [check.to_dict() for check in checks],
    }


def format_preflight_report(report: Mapping[str, Any]) -> str:
    lines = [
        f"Production preflight: {report.get('config') or '<config>'}",
        f"  ok: {bool(report.get('ok'))}",
        f"  errors: {int(report.get('error_count') or 0)}",
        f"  warnings: {int(report.get('warning_count') or 0)}",
    ]
    for check in report.get("checks") or []:
        status = str(check.get("status") or "").upper()
        name = str(check.get("name") or "check")
        message = str(check.get("message") or "")
        lines.append(f"  {status} {name}: {message}")
        details = check.get("details")
        if isinstance(details, Mapping):
            for key, value in details.items():
                if value not in ("", [], {}, None):
                    lines.append(f"    {key}: {value}")
    return "\n".join(lines)
