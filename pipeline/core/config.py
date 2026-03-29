"""
YAML configuration loader with inheritance support.

Configs can specify ``_inherit: parent.yaml`` to deep-merge a parent config,
allowing project-specific overrides to stay small.

Environment variables can override any config value via ``PIPELINE_<SECTION>__<KEY>``
(double-underscore separates nesting levels).
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

# Default search path for config files
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


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


def _resolve_config_path(name: str, search_dirs: Optional[List[Path]] = None) -> Path:
    """Find a config file by name, checking search dirs in order."""
    search_dirs = search_dirs or [_CONFIG_DIR]

    # If it's already an absolute path, use it directly
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p

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
        if not env_key.startswith(prefix):
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


def load_resolved_run_config(work_dir: str | Path) -> Optional[Dict[str, Any]]:
    """Load a run-local resolved config snapshot written by the orchestrator."""
    path = Path(work_dir) / "resolved_config.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
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
    """Load a config, preferring a run-local resolved snapshot when available."""
    config: Optional[Dict[str, Any]] = None
    if work_dir:
        config = load_resolved_run_config(work_dir)
    if config is None:
        config = load_config(name, search_dirs=search_dirs, overrides=None)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


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
        parent_path = _resolve_config_path(inherit, extra_dirs)
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
