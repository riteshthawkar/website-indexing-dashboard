"""
YAML configuration loader with inheritance support.

Configs can specify ``_inherit: parent.yaml`` to deep-merge a parent config,
allowing project-specific overrides to stay small.

Environment variables can override any config value via ``PIPELINE_<SECTION>__<KEY>``
(double-underscore separates nesting levels).
"""

import hashlib
import json
import logging
import os
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

logger = logging.getLogger(__name__)

# Default search path for config files
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
_PIPELINE_CONTROL_ENV_KEYS = {
    "PIPELINE_ARGS_JSON",
    "PIPELINE_CONFIG",
    "PIPELINE_IMAGE",
    "PIPELINE_PREFLIGHT",
    "PIPELINE_RESTART_FROM_STAGE",
    "PIPELINE_RESUME",
}


class ProductionConfigMismatchError(ValueError):
    """Raised when a run was indexed with a different production contract."""


_INDEXING_CONTRACT_SECTIONS = (
    "pipeline",
    "selected_profile",
    "crawler",
    "cleaner",
    "converter",
    "chunker",
    "summarizer",
    "quality",
    "assertions",
    "formatter",
    "embedder",
    "vector_store",
    "graph",
)
_PIPELINE_RUNTIME_ONLY_KEYS = {"active_release_file"}
_GRAPH_RUNTIME_ONLY_KEYS = {
    "neo4j_uri",
    "neo4j_username",
    "neo4j_password",
    "neo4j_database",
    "neo4j_http_timeout_sec",
}
_VECTOR_STORE_RUNTIME_ONLY_KEYS = {
    "dsn_env",
    "ingest_dsn_env",
    "reader_role",
    "writer_role",
    "allow_shared_dsn_for_ingest",
    "require_ssl",
    "require_active_release",
    "pool_min_size",
    "pool_max_size",
    "pool_timeout_seconds",
    "connect_timeout_seconds",
    "max_lifetime_seconds",
    "max_idle_seconds",
    "statement_timeout_ms",
    "idle_transaction_timeout_ms",
    "hnsw_ef_search",
    "application_name",
    "ingest_application_name",
}
_SERVING_IMPLEMENTATION_FILES = (
    "core/evidence_adjudicator.py",
    "core/navigation_intent.py",
    "core/query_expansion.py",
    "core/query_planner.py",
    "retrieval/adaptive_hybrid.py",
    "retrieval/evidence_packer.py",
    "retrieval/graph_rag.py",
    "retrieval/navigation_planner.py",
    "retrieval/routed_hybrid.py",
    "vectorstores/pgvector_store.py",
)
_INDEXING_CORE_IMPLEMENTATION_FILES = (
    "core/answer_records.py",
    "core/artifact_contracts.py",
    "core/assertions.py",
    "core/graph_artifacts.py",
    "core/knowledge_graph.py",
    "core/mbzuai_indexing.py",
    "core/page_graph_bridge.py",
    "core/release_assembly.py",
)
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
    "connection_string",
    "database_url",
    "dsn",
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


def _normalized_config_key(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def _is_secret_config_key(key: Any) -> bool:
    normalized = _normalized_config_key(key)
    # Capacity/quality settings such as ``max_postings_per_token`` describe
    # lexical tokens; they are not authentication tokens and must remain in
    # the immutable production snapshot.
    if normalized.endswith("_per_token"):
        return False
    return normalized in _SECRET_CONFIG_KEYS or normalized.endswith(_SECRET_CONFIG_SUFFIXES)


def sanitized_config_snapshot(value: Any) -> Any:
    """Return a deep copy safe for run snapshots and release archives."""
    if isinstance(value, dict):
        return {
            str(key): sanitized_config_snapshot(item)
            for key, item in value.items()
            if not _is_secret_config_key(key)
        }
    if isinstance(value, list):
        return [sanitized_config_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return [sanitized_config_snapshot(item) for item in value]
    return value


def configured_secret_paths(value: Any, *, prefix: str = "") -> List[str]:
    """List non-empty secret-bearing config paths without returning their values."""
    paths: List[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if _is_secret_config_key(key):
                if item not in (None, "", [], {}, ()):
                    paths.append(path)
                continue
            paths.extend(configured_secret_paths(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(configured_secret_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


@lru_cache(maxsize=1)
def indexing_implementation_hashes() -> Dict[str, str]:
    pipeline_root = Path(__file__).resolve().parents[1]
    paths = [pipeline_root / relative for relative in _INDEXING_CORE_IMPLEMENTATION_FILES]
    paths.extend(sorted((pipeline_root / "stages").rglob("*.py")))
    hashes: Dict[str, str] = {}
    for path in sorted(set(paths)):
        relative = str(path.relative_to(pipeline_root))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    return dict(hashes)


def indexing_build_identity() -> Dict[str, Any]:
    """Resolve the source revision and dirty-tree state without exposing paths."""
    project_root = Path(__file__).resolve().parents[2]
    commit_sha = ""
    dirty = True
    source = "unavailable"
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain", "--untracked-files=normal"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
        if commit.returncode == 0 and status.returncode == 0:
            commit_sha = commit.stdout.strip().lower()
            dirty = bool(status.stdout.strip())
            source = "git"
    except (OSError, subprocess.SubprocessError):
        pass
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit_sha):
        environment_commit = str(os.getenv("RELEASE_COMMIT_SHA") or "").strip().lower()
        if re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", environment_commit):
            commit_sha = environment_commit
            dirty = os.getenv("INDEXING_SOURCE_DIRTY", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            source = "environment"
    return {
        "commit_sha": commit_sha,
        "dirty": bool(dirty),
        "source": source,
        "implementation_sha256": indexing_implementation_hashes(),
    }


def production_indexing_contract_payload(
    config: Dict[str, Any],
    *,
    implementation_hashes: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """Return the immutable configuration that determines indexed artifacts.

    Retrieval routing and serving limits are intentionally excluded: they may
    be tuned without rebuilding vectors.  Content processing, stage order,
    embedding targets, namespace strategy, and graph construction are bound to
    the run and must match the explicitly requested production profile.
    """

    safe_config = sanitized_config_snapshot(config)
    payload: Dict[str, Any] = {
        "project_name": safe_config.get("project_name"),
        "stages": safe_config.get("stages") or [],
        "implementation_sha256": dict(
            implementation_hashes
            if implementation_hashes is not None
            else indexing_implementation_hashes()
        ),
    }
    for section in _INDEXING_CONTRACT_SECTIONS:
        value = safe_config.get(section)
        if not isinstance(value, dict):
            continue
        section_payload = dict(value)
        if section == "pipeline":
            for key in _PIPELINE_RUNTIME_ONLY_KEYS:
                section_payload.pop(key, None)
        elif section == "graph":
            for key in _GRAPH_RUNTIME_ONLY_KEYS:
                section_payload.pop(key, None)
        elif section == "vector_store":
            for key in _VECTOR_STORE_RUNTIME_ONLY_KEYS:
                section_payload.pop(key, None)
        payload[section] = section_payload
    return payload


def production_indexing_contract_fingerprint(
    config: Dict[str, Any],
    *,
    implementation_hashes: Optional[Mapping[str, str]] = None,
) -> str:
    raw = json.dumps(
        production_indexing_contract_payload(
            config,
            implementation_hashes=implementation_hashes,
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def production_serving_contract_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return runtime behavior that must be answer-evaluated as one release.

    Unlike the indexing contract, this intentionally includes retrieval
    routing/planner/adjudicator settings and an implementation digest.  A code
    or configuration change therefore cannot silently reuse answer-readiness
    gates from an older serving behavior.
    """

    pipeline_root = Path(__file__).resolve().parents[1]
    implementation_hashes: Dict[str, str] = {}
    for relative_path in _SERVING_IMPLEMENTATION_FILES:
        path = pipeline_root / relative_path
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
        implementation_hashes[relative_path] = digest

    safe_config = sanitized_config_snapshot(config)
    embedder = safe_config.get("embedder") if isinstance(safe_config.get("embedder"), dict) else {}
    graph = safe_config.get("graph") if isinstance(safe_config.get("graph"), dict) else {}
    vector_store = safe_config.get("vector_store") if isinstance(safe_config.get("vector_store"), dict) else {}
    pipeline = safe_config.get("pipeline") if isinstance(safe_config.get("pipeline"), dict) else {}
    return {
        "project_name": safe_config.get("project_name"),
        "pipeline": {
            key: pipeline.get(key)
            for key in (
                "production_profile",
                "require_query_planner",
                "require_assertion_first",
            )
        },
        "retrieval": dict(safe_config.get("retrieval") or {}),
        "serving": dict(safe_config.get("serving") or {}),
        "embedder_runtime": {
            key: embedder.get(key)
            for key in (
                "model",
                "output_dimensionality",
                "pinecone_index",
                "pinecone_sparse_index",
                "namespace_strategy",
                "namespace_release_template",
                "namespace_chunks",
                "namespace_parents",
                "namespace_media",
                "namespace_page_cards",
                "namespace_actions",
                "namespace_facts",
                "namespace_evidence_spans",
                "namespace_summaries",
                "namespace_assertions",
                "namespace_entities",
                "namespace_communities",
            )
        },
        "vector_store_runtime": {
            key: vector_store.get(key)
            for key in (
                "provider",
                "schema",
                "records_table",
                "releases_table",
                "dimensions",
                "schema_version",
                "dsn_env",
                "reader_role",
                "require_ssl",
                "require_active_release",
                "pool_min_size",
                "pool_max_size",
                "pool_timeout_seconds",
                "connect_timeout_seconds",
                "max_lifetime_seconds",
                "max_idle_seconds",
                "statement_timeout_ms",
                "idle_transaction_timeout_ms",
                "hnsw_ef_search",
            )
        },
        "graph_runtime": {
            key: graph.get(key)
            for key in (
                "store_backend",
                "graph_store_backend",
                "neo4j_database",
                "community_summary_min_coverage_ratio",
                "community_summary_min_characters",
            )
        },
        "selected_profile": {
            key: (safe_config.get("selected_profile") or {}).get(key)
            for key in ("variant_id", "record_kinds")
        }
        if isinstance(safe_config.get("selected_profile"), dict)
        else {},
        "implementation_sha256": implementation_hashes,
    }


def production_serving_contract_fingerprint(config: Dict[str, Any]) -> str:
    raw = json.dumps(
        production_serving_contract_payload(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Recursively merge *override* into a copy of *base*.

    - Dicts are merged recursively.
    - Lists and scalars in *override* replace those in *base*.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path(
    name: str,
    search_dirs: Optional[List[Path]] = None,
    *,
    allow_cwd_relative: bool = True,
) -> Path:
    """Find a config file by name, checking search dirs in order.

    ``allow_cwd_relative`` is intentionally enabled only for the top-level
    config supplied by the caller. Inherited config names must resolve against
    the child config directory (or explicit search directories), otherwise a
    same-named file in the process working directory could silently replace a
    trusted parent config.
    """
    search_dirs = search_dirs or [_CONFIG_DIR]

    # Accept an existing path exactly as supplied. Deployment scripts commonly
    # pass repository-relative paths such as ``pipeline/configs/foo.yaml``;
    # joining those to ``_CONFIG_DIR`` would duplicate the path segments.
    p = Path(name).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    if allow_cwd_relative and p.exists():
        return p.resolve()

    # Try adding .yaml extension if not present
    candidates = [name]
    if not name.endswith((".yaml", ".yml")):
        candidates.append(f"{name}.yaml")
        candidates.append(f"{name}.yml")

    for search_dir in search_dirs:
        for candidate in candidates:
            full = search_dir / candidate
            if full.exists():
                return full

    raise FileNotFoundError(
        f"Config {name!r} not found in {[str(d) for d in search_dirs]}"
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a single YAML file."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must be a YAML mapping, got {type(data).__name__}")
    return data


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides.

    Variables matching ``PIPELINE_<SECTION>__<KEY>`` override the corresponding
    config path.  Example: ``PIPELINE_CRAWLER__MAX_PAGES=500`` sets
    ``config["crawler"]["max_pages"] = 500``.
    """
    prefix = "PIPELINE_"
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix) or env_key in _PIPELINE_CONTROL_ENV_KEYS:
            continue
        parts = env_key[len(prefix):].lower().split("__")
        # Navigate to the right nesting level
        target = config
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]
        # Try to parse as int/float/bool/null, otherwise keep as string
        target[parts[-1]] = _parse_env_value(env_val)
    return config


def _parse_env_value(value: str) -> Any:
    """Parse an env var string into a Python value."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none", ""):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(
    name: str,
    search_dirs: Optional[List[Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a config by name, resolving ``_inherit`` chains and env overrides.

    Args:
        name: Config filename or path (e.g. ``"mbzuai_main"``).
        search_dirs: Directories to search for config files.
        overrides: Additional overrides applied last.

    Returns:
        Fully merged configuration dictionary.
    """
    path = _resolve_config_path(name, search_dirs)
    config = _load_config_recursive(path, search_dirs, seen=set())

    # Apply env overrides
    config = _apply_env_overrides(config)

    # Apply explicit overrides
    if overrides:
        config = _deep_merge(config, overrides)

    return config


def load_resolved_run_snapshot(work_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Load the complete run-local config snapshot written by the orchestrator."""
    path = Path(work_dir) / "resolved_config.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return dict(payload)


def load_resolved_run_config(work_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Load the config mapping from a run-local resolved snapshot."""
    payload = load_resolved_run_snapshot(work_dir)
    if payload is None:
        return None
    config = payload.get("config")
    return dict(config) if isinstance(config, dict) else None


def load_effective_config(
    name: str,
    *,
    work_dir: str | Path | None = None,
    search_dirs: Optional[List[Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load the effective config without allowing production-profile drift.

    Historical and development configs prefer a run-local snapshot. Any
    explicitly requested config marked ``pipeline.production_profile=true`` is
    authoritative for runtime capabilities, while its immutable indexing
    contract must match the run snapshot exactly. This applies equally to the
    canonical pgvector profile and an approved store-specific deployment
    profile such as the interim Pinecone release.
    """

    snapshot_path = Path(work_dir) / "resolved_config.json" if work_dir else None
    snapshot = load_resolved_run_snapshot(work_dir) if work_dir else None
    snapshot_config = (
        dict(snapshot.get("config"))
        if isinstance(snapshot, dict) and isinstance(snapshot.get("config"), dict)
        else None
    )
    try:
        requested_config = load_config(name, search_dirs=search_dirs, overrides=None)
    except FileNotFoundError:
        if snapshot_config is None:
            raise
        config = snapshot_config
        if overrides:
            config = _deep_merge(config, overrides)
        return config

    if overrides:
        requested_config = _deep_merge(requested_config, overrides)
    pipeline_cfg = requested_config.get("pipeline")
    production_requested = bool(
        pipeline_cfg.get("production_profile", False)
        if isinstance(pipeline_cfg, dict)
        else False
    )
    if not production_requested:
        if snapshot_config is None:
            return requested_config
        return _deep_merge(snapshot_config, overrides) if overrides else snapshot_config

    if snapshot_path is not None and snapshot_path.exists() and snapshot is None:
        raise ProductionConfigMismatchError(
            f"Run resolved_config.json is unreadable or invalid: {snapshot_path}"
        )
    if snapshot_config is None:
        return requested_config

    actual_snapshot_fingerprint = production_indexing_contract_fingerprint(snapshot_config)
    recorded_snapshot_fingerprint = str(
        (snapshot or {}).get("production_indexing_contract_fingerprint") or ""
    ).strip()
    if recorded_snapshot_fingerprint and recorded_snapshot_fingerprint != actual_snapshot_fingerprint:
        raise ProductionConfigMismatchError(
            "Run resolved_config.json failed its recorded production indexing fingerprint; "
            "the snapshot may have been modified after indexing"
        )

    requested_fingerprint = production_indexing_contract_fingerprint(requested_config)
    if actual_snapshot_fingerprint != requested_fingerprint:
        raise ProductionConfigMismatchError(
            "Run indexing config does not match the explicitly requested production profile "
            f"(run={actual_snapshot_fingerprint}, requested={requested_fingerprint}). "
            "Create a fresh run with the requested production profile or use the explicit "
            "migrate-release workflow; "
            "the saved run config will not silently override production."
        )
    return requested_config


def _load_config_recursive(
    path: Path,
    search_dirs: Optional[List[Path]],
    seen: set,
) -> Dict[str, Any]:
    """Load config with recursive _inherit resolution."""
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Circular config inheritance detected: {resolved}")
    seen.add(resolved)

    data = _load_yaml(path)
    inherit = data.pop("_inherit", None)

    if inherit:
        # Also search in the same directory as the current file
        extra_dirs = [path.parent] + (search_dirs or [_CONFIG_DIR])
        parent_path = _resolve_config_path(
            inherit,
            extra_dirs,
            allow_cwd_relative=False,
        )
        parent_data = _load_config_recursive(parent_path, search_dirs, seen)
        data = _deep_merge(parent_data, data)

    return data


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge multiple config dicts left-to-right (later wins)."""
    result: Dict[str, Any] = {}
    for cfg in configs:
        result = _deep_merge(result, cfg)
    return result


def list_configs(search_dirs: Optional[List[Path]] = None) -> List[Dict[str, str]]:
    """List all available config files with their project names."""
    search_dirs = search_dirs or [_CONFIG_DIR]
    configs = []
    for d in search_dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.yaml")) + sorted(d.glob("*.yml")):
            try:
                data = _load_yaml(f)
                configs.append({
                    "file": str(f),
                    "name": f.stem,
                    "project_name": data.get("project_name", f.stem),
                })
            except Exception as e:
                logger.warning("Skipping config %s: %s", f, e)
    return configs
