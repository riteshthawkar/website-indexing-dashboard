"""Configuration helpers exposed by the stable package API."""

from __future__ import annotations

from copy import deepcopy
from os import PathLike
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

from pipeline.core.config import list_configs, load_config, merge_configs


ConfigMapping: TypeAlias = Mapping[str, Any]
ConfigSource: TypeAlias = str | PathLike[str] | ConfigMapping


def load_pipeline_config(
    source: ConfigSource,
    *,
    search_dirs: Sequence[str | PathLike[str]] | None = None,
    overrides: ConfigMapping | None = None,
) -> dict[str, Any]:
    """Return a detached, fully merged pipeline configuration.

    A string or path uses the pipeline YAML loader, including inheritance and
    ``PIPELINE_*`` environment overrides. A mapping is copied before explicit
    overrides are applied, so callers cannot mutate a running pipeline by
    changing their input dictionary later.
    """

    normalized_overrides = deepcopy(dict(overrides or {}))
    if isinstance(source, Mapping):
        return merge_configs(deepcopy(dict(source)), normalized_overrides)

    normalized_dirs = (
        [Path(value).expanduser() for value in search_dirs]
        if search_dirs is not None
        else None
    )
    return load_config(
        str(source),
        search_dirs=normalized_dirs,
        overrides=normalized_overrides or None,
    )


def available_pipeline_configs(
    search_dirs: Sequence[str | PathLike[str]] | None = None,
) -> list[dict[str, str]]:
    """List bundled or caller-supplied YAML pipeline configurations."""

    normalized_dirs = (
        [Path(value).expanduser() for value in search_dirs]
        if search_dirs is not None
        else None
    )
    return list_configs(search_dirs=normalized_dirs)


__all__ = [
    "ConfigMapping",
    "ConfigSource",
    "available_pipeline_configs",
    "load_pipeline_config",
]
