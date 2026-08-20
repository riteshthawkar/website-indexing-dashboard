"""Production-readiness checks for terminal pipeline operations."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import urlparse

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

MBZUAI_REQUIRED_CRITICAL_URL_PATTERNS = {
    "/about/office-of-the-president/?$",
    "/about/leadership/?$",
    "/about/contact/?$",
    "/study/graduate-admission-process/?$",
    "/study/(?:undergraduate|ug)-admission-process/?$",
}

MBZUAI_REQUIRED_CRAWL_HOSTS = {
    "mbzuai.ac.ae",
    "www.mbzuai.ac.ae",
    "careers.mbzuai.ac.ae",
    "research.mbzuai.ac.ae",
    "ai-nexus.mbzuai.ac.ae",
    "library.mbzuai.ac.ae",
    "metaverse.mbzuai.ac.ae",
    "buildit.mbzuai.ac.ae",
    "hpp.mbzuai.ac.ae",
    "ifm.mbzuai.ac.ae",
    "ifm.ai",
}

MBZUAI_REQUIRED_SITEMAP_ORIGINS = {
    "mbzuai.ac.ae",
    "careers.mbzuai.ac.ae",
    "research.mbzuai.ac.ae",
    "ai-nexus.mbzuai.ac.ae",
    "hpp.mbzuai.ac.ae",
    "ifm.ai",
}

MBZUAI_MINIMUM_SITEMAP_URLS_BY_HOST = {
    "mbzuai.ac.ae": 2100,
    "careers.mbzuai.ac.ae": 40,
    "research.mbzuai.ac.ae": 8,
    "ai-nexus.mbzuai.ac.ae": 8,
    "hpp.mbzuai.ac.ae": 3,
    "ifm.ai": 15,
}

MBZUAI_MINIMUM_CRAWLED_PAGES_BY_HOST = {
    **MBZUAI_MINIMUM_SITEMAP_URLS_BY_HOST,
    "careers.mbzuai.ac.ae": 30,
    "library.mbzuai.ac.ae": 15,
    "metaverse.mbzuai.ac.ae": 30,
    "buildit.mbzuai.ac.ae": 1,
}

MBZUAI_REQUIRED_NO_SITEMAP_SEED_HOSTS = {
    "library.mbzuai.ac.ae",
    "metaverse.mbzuai.ac.ae",
    "buildit.mbzuai.ac.ae",
}

MBZUAI_LINK_DISCOVERY_HOSTS = {
    "library.mbzuai.ac.ae",
    "metaverse.mbzuai.ac.ae",
}

MBZUAI_INTENTIONAL_CONTENT_EXCLUSIONS = {
    "https://buildit.mbzuai.ac.ae/about",
    "https://buildit.mbzuai.ac.ae/apply",
    "https://buildit.mbzuai.ac.ae/highlights",
    "https://buildit.mbzuai.ac.ae/benefits",
    "https://buildit.mbzuai.ac.ae/network",
    "https://buildit.mbzuai.ac.ae/faqs",
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

        start_url = str(crawler_cfg.get("start_url") or "").strip()
        try:
            parsed_start_url = urlparse(start_url)
            start_scheme = parsed_start_url.scheme.lower()
            start_hostname = (parsed_start_url.hostname or "").lower()
        except ValueError:
            start_scheme = ""
            start_hostname = ""
        if start_scheme != "https" or start_hostname != "mbzuai.ac.ae":
            production_contract_errors.append(
                "crawler.start_url must target https://mbzuai.ac.ae"
            )
        allowed_domains = {
            str(value).strip().lower().strip(".")
            for value in (crawler_cfg.get("allowed_domains") or [])
            if str(value).strip()
        }
        if allowed_domains != {"mbzuai.ac.ae", "ifm.ai"}:
            production_contract_errors.append(
                "crawler.allowed_domains must contain only mbzuai.ac.ae and ifm.ai"
            )
        allowed_hosts = {
            str(value).strip().lower().strip(".")
            for value in (crawler_cfg.get("allowed_hosts") or [])
            if str(value).strip()
        }
        if allowed_hosts != MBZUAI_REQUIRED_CRAWL_HOSTS:
            production_contract_errors.append(
                "crawler.allowed_hosts must equal the approved MBZUAI public-content host set"
            )
        robots_origin_hosts = {
            (urlparse(str(value)).hostname or "").lower()
            for value in (crawler_cfg.get("robots_origins") or [])
            if str(value).strip()
        }
        if robots_origin_hosts != MBZUAI_REQUIRED_CRAWL_HOSTS:
            production_contract_errors.append(
                "crawler.robots_origins must cover every approved MBZUAI crawl host"
            )
        if str(crawler_cfg.get("robots_unknown_host_policy") or "").lower() != "deny":
            production_contract_errors.append(
                "crawler.robots_unknown_host_policy must be deny"
            )
        if "mbzuaiknowledgeindexer" not in str(
            crawler_cfg.get("robots_user_agent") or ""
        ).lower():
            production_contract_errors.append(
                "crawler.robots_user_agent must identify the MBZUAI knowledge indexer"
            )
        crawler_headers = crawler_cfg.get("headers") or {}
        normalized_crawler_headers = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in crawler_headers.items()
        } if isinstance(crawler_headers, Mapping) else {}
        if normalized_crawler_headers.get("x-crawler-name") != "mbzuaiknowledgeindexer":
            production_contract_errors.append(
                "crawler.headers must identify MBZUAIKnowledgeIndexer"
            )
        if normalized_crawler_headers.get("x-crawler-purpose") != "search-index-reference":
            production_contract_errors.append(
                "crawler.headers must declare search-index-reference purpose"
            )
        sitemap_origin_hosts = {
            (urlparse(str(value)).hostname or "").lower()
            for value in (crawler_cfg.get("sitemap_origins") or [])
            if str(value).strip()
        }
        if sitemap_origin_hosts != MBZUAI_REQUIRED_SITEMAP_ORIGINS:
            production_contract_errors.append(
                "crawler.sitemap_origins must cover every approved sitemap origin"
            )
        sitemap_entry_urls = {
            str(value).strip().rstrip("/")
            for value in (crawler_cfg.get("sitemap_entry_urls") or [])
            if str(value).strip()
        }
        if "https://careers.mbzuai.ac.ae/wp-sitemap.xml" not in sitemap_entry_urls:
            production_contract_errors.append(
                "crawler.sitemap_entry_urls must include the Careers WordPress sitemap"
            )
        link_discovery_hosts = {
            str(value).strip().lower().strip(".")
            for value in (crawler_cfg.get("link_discovery_hosts") or [])
            if str(value).strip()
        }
        if link_discovery_hosts != MBZUAI_LINK_DISCOVERY_HOSTS:
            production_contract_errors.append(
                "crawler.link_discovery_hosts must equal the approved bounded-discovery host set"
            )
        priority_seed_urls = {
            str(value).strip().rstrip("/")
            for value in (crawler_cfg.get("priority_seed_urls") or [])
            if str(value).strip()
        }
        priority_seed_hosts = {
            (urlparse(value).hostname or "").lower()
            for value in priority_seed_urls
        }
        missing_seed_hosts = MBZUAI_REQUIRED_NO_SITEMAP_SEED_HOSTS - priority_seed_hosts
        if missing_seed_hosts:
            production_contract_errors.append(
                "crawler.priority_seed_urls is missing no-sitemap origins: "
                + ", ".join(sorted(missing_seed_hosts))
            )
        excluded_priority_seeds = (
            MBZUAI_INTENTIONAL_CONTENT_EXCLUSIONS & priority_seed_urls
        )
        if excluded_priority_seeds:
            production_contract_errors.append(
                "crawler.priority_seed_urls must not include intentional content exclusions: "
                + ", ".join(sorted(excluded_priority_seeds))
            )
        discovery_budgets = crawler_cfg.get("link_discovery_max_pages_by_host") or {}
        for host in MBZUAI_LINK_DISCOVERY_HOSTS:
            try:
                budget = int(discovery_budgets.get(host) or 0)
            except (AttributeError, TypeError, ValueError):
                budget = 0
            if budget < 1:
                production_contract_errors.append(
                    f"crawler.link_discovery_max_pages_by_host[{host}] must be >= 1"
                )
        if not str(crawler_cfg.get("origin_inventory_revision") or "").strip():
            production_contract_errors.append(
                "crawler.origin_inventory_revision must identify the researched origin set"
            )
        origin_inventory = crawler_cfg.get("origin_inventory") or {}
        intentional_exclusions = (
            origin_inventory.get("intentional_content_exclusions") or {}
            if isinstance(origin_inventory, Mapping)
            else {}
        )
        recorded_exclusions = {
            str(value).strip().rstrip("/")
            for value in (
                intentional_exclusions.keys()
                if isinstance(intentional_exclusions, Mapping)
                else []
            )
            if str(value).strip()
        }
        missing_exclusions = (
            MBZUAI_INTENTIONAL_CONTENT_EXCLUSIONS - recorded_exclusions
        )
        if missing_exclusions:
            production_contract_errors.append(
                "crawler.origin_inventory.intentional_content_exclusions is missing: "
                + ", ".join(sorted(missing_exclusions))
            )
        for key, required_values in (
            (
                "minimum_sitemap_urls_by_host",
                MBZUAI_MINIMUM_SITEMAP_URLS_BY_HOST,
            ),
            (
                "minimum_crawled_pages_by_host",
                MBZUAI_MINIMUM_CRAWLED_PAGES_BY_HOST,
            ),
        ):
            configured_values = crawler_cfg.get(key) or {}
            for host, required in required_values.items():
                try:
                    configured_minimum = int(configured_values.get(host) or 0)
                except (AttributeError, TypeError, ValueError):
                    configured_minimum = 0
                if configured_minimum < required:
                    production_contract_errors.append(
                        f"crawler.{key}[{host}] must be >= {required}"
                    )
        if not bool(crawler_cfg.get("respect_robots_txt", False)):
            production_contract_errors.append("crawler.respect_robots_txt must be true")
        if bool(crawler_cfg.get("include_external", False)):
            production_contract_errors.append("crawler.include_external must be false")
        if bool(crawler_cfg.get("allow_query_urls", False)):
            production_contract_errors.append("crawler.allow_query_urls must be false")
        if not bool(crawler_cfg.get("fail_on_empty_result", False)):
            production_contract_errors.append("crawler.fail_on_empty_result must be true")
        if not bool(crawler_cfg.get("sitemap_enabled", False)):
            production_contract_errors.append("crawler.sitemap_enabled must be true")

        try:
            minimum_sitemap_seed_count = int(
                crawler_cfg.get("minimum_sitemap_seed_count") or 0
            )
        except (TypeError, ValueError):
            minimum_sitemap_seed_count = 0
        if minimum_sitemap_seed_count < 2100:
            production_contract_errors.append(
                "crawler.minimum_sitemap_seed_count must be >= 2100"
            )
        for key in ("sitemap_seed_limit", "sitemap_frontier_seed_limit", "max_pages"):
            try:
                value = int(crawler_cfg.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if value < minimum_sitemap_seed_count:
                production_contract_errors.append(
                    f"crawler.{key} must be >= crawler.minimum_sitemap_seed_count"
                )
        bounded_crawl_limits = {
            "robots_max_response_bytes": (1, 512 * 1024),
            "sitemap_max_response_bytes": (1, 16 * 1024 * 1024),
            "sitemap_max_sources": (32, 100),
            "sitemap_max_depth": (1, 4),
        }
        for key, (minimum_value, maximum_value) in bounded_crawl_limits.items():
            try:
                value = int(crawler_cfg.get(key) or 0)
            except (TypeError, ValueError):
                value = 0
            if not minimum_value <= value <= maximum_value:
                production_contract_errors.append(
                    f"crawler.{key} must be between {minimum_value} and {maximum_value}"
                )
        if not bool(crawler_cfg.get("validate_source_html", False)):
            production_contract_errors.append("crawler.validate_source_html must be true")
        source_validation_mode = str(
            crawler_cfg.get("validate_source_html_mode") or ""
        ).strip().lower()
        if source_validation_mode != "always":
            production_contract_errors.append(
                "crawler.validate_source_html_mode must be always"
            )
        if not bool(crawler_cfg.get("retry_recoverable_skipped_on_resume", False)):
            production_contract_errors.append(
                "crawler.retry_recoverable_skipped_on_resume must be true"
            )
        known_empty_cohorts = crawler_cfg.get("known_empty_sitemap_cohorts")
        if not isinstance(known_empty_cohorts, list) or not known_empty_cohorts:
            production_contract_errors.append(
                "crawler.known_empty_sitemap_cohorts must be configured"
            )

        quality_contract = (
            config.get("quality", {})
            if isinstance(config.get("quality"), Mapping)
            else {}
        )
        for key in (
            "detect_login_walls",
            "fail_on_empty_input",
            "fail_on_zero_output",
        ):
            if not bool(quality_contract.get(key, False)):
                production_contract_errors.append(f"quality.{key} must be true")
        try:
            quality_retention = float(
                quality_contract.get("minimum_retention_ratio") or 0.0
            )
        except (TypeError, ValueError):
            quality_retention = 0.0
        if quality_retention < 0.70:
            production_contract_errors.append(
                "quality.minimum_retention_ratio must be >= 0.70"
            )
        try:
            quality_maximum_errors = int(quality_contract.get("maximum_error_count"))
        except (TypeError, ValueError):
            quality_maximum_errors = -1
        if quality_maximum_errors != 0:
            production_contract_errors.append("quality.maximum_error_count must be 0")
        try:
            quality_maximum_error_ratio = float(
                quality_contract.get("maximum_error_ratio")
            )
        except (TypeError, ValueError):
            quality_maximum_error_ratio = -1.0
        if quality_maximum_error_ratio != 0.0:
            production_contract_errors.append("quality.maximum_error_ratio must be 0")

        cleaner_contract = (
            config.get("cleaner", {})
            if isinstance(config.get("cleaner"), Mapping)
            else {}
        )
        for key in (
            "include_tables",
            "include_links",
            "preserve_embedded_media",
            "recursive",
            "fail_on_empty_input",
            "fail_on_zero_output",
            "require_critical_url_survival",
        ):
            if not bool(cleaner_contract.get(key, False)):
                production_contract_errors.append(f"cleaner.{key} must be true")
        for key, minimum in (
            ("min_content_length", 100),
            ("min_content_words", 5),
            ("minimum_host_input_count", 1),
        ):
            try:
                value = int(cleaner_contract.get(key))
            except (TypeError, ValueError):
                value = -1
            if value < minimum:
                production_contract_errors.append(
                    f"cleaner.{key} must be >= {minimum}"
                )
        try:
            minimum_host_input_count = int(
                cleaner_contract.get("minimum_host_input_count")
            )
        except (TypeError, ValueError):
            minimum_host_input_count = 0
        if minimum_host_input_count > 10:
            production_contract_errors.append(
                "cleaner.minimum_host_input_count must be <= 10"
            )
        for key, minimum in (
            ("minimum_retention_ratio", 0.75),
            ("minimum_host_retention_ratio", 0.50),
        ):
            try:
                value = float(cleaner_contract.get(key) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value < minimum:
                production_contract_errors.append(
                    f"cleaner.{key} must be >= {minimum:.2f}"
                )
        try:
            cleaner_maximum_errors = int(cleaner_contract.get("maximum_error_count"))
        except (TypeError, ValueError):
            cleaner_maximum_errors = -1
        if cleaner_maximum_errors != 0:
            production_contract_errors.append("cleaner.maximum_error_count must be 0")
        try:
            cleaner_maximum_error_ratio = float(
                cleaner_contract.get("maximum_error_ratio")
            )
        except (TypeError, ValueError):
            cleaner_maximum_error_ratio = -1.0
        if cleaner_maximum_error_ratio != 0.0:
            production_contract_errors.append("cleaner.maximum_error_ratio must be 0")

        formatter_contract = (
            config.get("formatter", {})
            if isinstance(config.get("formatter"), Mapping)
            else {}
        )
        for key in (
            "fail_on_critical_coverage",
            "fail_on_inventory_gap",
            "fail_on_hard_failure_gap",
            "require_critical_url_markdown_evidence",
        ):
            if not bool(formatter_contract.get(key, False)):
                production_contract_errors.append(f"formatter.{key} must be true")
        try:
            expected_inventory = int(
                formatter_contract.get("expected_site_inventory_count") or 0
            )
        except (TypeError, ValueError):
            expected_inventory = 0
        if expected_inventory < 2600:
            production_contract_errors.append(
                "formatter.expected_site_inventory_count must be >= 2600"
            )
        try:
            minimum_inventory_coverage = float(
                formatter_contract.get("minimum_inventory_coverage_ratio") or 0.0
            )
        except (TypeError, ValueError):
            minimum_inventory_coverage = 0.0
        if minimum_inventory_coverage < 0.90:
            production_contract_errors.append(
                "formatter.minimum_inventory_coverage_ratio must be >= 0.90"
            )
        try:
            maximum_hard_failures = int(
                formatter_contract.get("maximum_hard_failure_count")
            )
        except (TypeError, ValueError):
            maximum_hard_failures = -1
        if maximum_hard_failures < 0 or maximum_hard_failures > 25:
            production_contract_errors.append(
                "formatter.maximum_hard_failure_count must be between 0 and 25"
            )
        configured_critical_patterns = {
            str(value).strip()
            for value in (formatter_contract.get("critical_url_patterns") or [])
            if str(value).strip()
        }
        missing_critical_patterns = sorted(
            MBZUAI_REQUIRED_CRITICAL_URL_PATTERNS - configured_critical_patterns
        )
        if missing_critical_patterns:
            production_contract_errors.append(
                "formatter.critical_url_patterns is missing required exact MBZUAI routes: "
                + ", ".join(missing_critical_patterns)
            )

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
