"""
Plugin registry for pipeline stages.

Stages register themselves with the @register_stage decorator.
The orchestrator looks up stages by (stage_type, name).
"""

import importlib
import logging
import pkgutil
from typing import Dict, Optional, Type

from .base import PipelineStage

logger = logging.getLogger(__name__)

# Global registry: {stage_type: {name: class}}
_REGISTRY: Dict[str, Dict[str, Type[PipelineStage]]] = {}
_IMPORT_ERRORS: Dict[str, str] = {}


def _format_import_errors() -> str:
    if not _IMPORT_ERRORS:
        return ""
    details = "; ".join(
        f"{module}: {error}"
        for module, error in sorted(_IMPORT_ERRORS.items())
    )
    return f" Stage modules failed to import. Install the pipeline requirements and retry. Details: {details}"


def register_stage(cls: Type[PipelineStage]) -> Type[PipelineStage]:
    """Class decorator that registers a PipelineStage subclass.

    Usage:
        @register_stage
        class MyCrawler(CrawlerStage):
            name = "my_crawler"
            ...
    """
    if not cls.name:
        raise ValueError(f"{cls.__name__} must define a 'name' class attribute")
    if not cls.stage_type:
        raise ValueError(f"{cls.__name__} must define a 'stage_type' class attribute")

    bucket = _REGISTRY.setdefault(cls.stage_type, {})
    if cls.name in bucket:
        existing = bucket[cls.name]
        if existing is not cls:
            logger.warning(
                "Overwriting stage %s/%s (%s -> %s)",
                cls.stage_type, cls.name, existing.__name__, cls.__name__,
            )
    bucket[cls.name] = cls
    logger.debug("Registered stage: %s/%s (%s)", cls.stage_type, cls.name, cls.__name__)
    return cls


def get_stage(stage_type: str, name: str) -> Type[PipelineStage]:
    """Look up a registered stage class by type and name."""
    bucket = _REGISTRY.get(stage_type)
    if not bucket:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown stage type {stage_type!r}. Available: {available}."
            f"{_format_import_errors()}"
        )
    cls = bucket.get(name)
    if not cls:
        available = list(bucket.keys())
        raise KeyError(
            f"Unknown {stage_type} plugin {name!r}. Available: {available}."
            f"{_format_import_errors()}"
        )
    return cls


def list_stages(stage_type: Optional[str] = None) -> Dict[str, Dict[str, dict]]:
    """List all registered stages with metadata.

    Returns:
        {stage_type: {name: {"description": ..., "class": ...}}}
    """
    result = {}
    for stype, bucket in _REGISTRY.items():
        if stage_type and stype != stage_type:
            continue
        result[stype] = {}
        for sname, cls in bucket.items():
            result[stype][sname] = {
                "description": cls.description or "",
                "class": f"{cls.__module__}.{cls.__name__}",
            }
    return result


def auto_discover():
    """Import all modules under pipeline.stages to trigger @register_stage decorators."""
    import pipeline.stages as stages_pkg

    _IMPORT_ERRORS.clear()
    for importer, modname, ispkg in pkgutil.walk_packages(
        stages_pkg.__path__, prefix=stages_pkg.__name__ + "."
    ):
        try:
            importlib.import_module(modname)
        except ImportError as e:
            # Don't fail if optional dependencies are missing (e.g., crawl4ai not installed)
            _IMPORT_ERRORS[modname] = str(e)
            logger.debug("Could not import stage module %s: %s", modname, e)
        except Exception as e:
            _IMPORT_ERRORS[modname] = str(e)
            logger.warning("Error importing stage module %s: %s", modname, e)
