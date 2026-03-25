"""
Configuration manager — delegates to the pipeline package's config system.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import (
    load_config as _pipeline_load_config,
    list_configs as _pipeline_list_configs,
)

CONFIGS_DIR = PROJECT_ROOT / "pipeline" / "configs"


def load_config(config_name: str) -> Optional[dict]:
    """Load a fully-resolved pipeline config by name."""
    try:
        return _pipeline_load_config(config_name)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.error(f"Error loading config {config_name}: {e}")
        return None


def save_config(config_name: str, config: dict) -> tuple[bool, str]:
    """Save a config file.  Only saves to the pipeline/configs/ directory."""
    path = CONFIGS_DIR / f"{config_name}.yaml"
    try:
        if path.exists():
            import shutil
            shutil.copy2(path, path.with_suffix(".yaml.bak"))

        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        return True, f"Config saved to {path.name}"
    except Exception as e:
        return False, f"Error saving config: {e}"


def list_configs() -> list[dict]:
    """List available pipeline configs."""
    return _pipeline_list_configs()


def get_config_schema(serializable: bool = False) -> dict:
    """Return a simple schema describing the main config sections.

    The pipeline configs are free-form YAML, so we return a description
    of the top-level sections rather than a strict schema.
    """
    # Load default config to discover sections
    try:
        default = _pipeline_load_config("default")
    except Exception:
        default = {}

    schema = {}
    for key, value in default.items():
        if key.startswith("_"):
            continue
        if isinstance(value, dict):
            schema[key] = {
                "type": "object" if serializable else dict,
                "description": f"Configuration section: {key}",
                "fields": {
                    k: {
                        "type": type(v).__name__ if serializable else type(v),
                        "value": v,
                    }
                    for k, v in value.items()
                },
            }
        elif isinstance(value, list):
            schema[key] = {
                "type": "array" if serializable else list,
                "description": f"Pipeline stage list" if key == "stages" else f"List: {key}",
            }
        else:
            schema[key] = {
                "type": type(value).__name__ if serializable else type(value),
                "description": str(key),
                "value": value,
            }
    return schema
