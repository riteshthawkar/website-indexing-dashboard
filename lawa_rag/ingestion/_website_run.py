"""Validate completed generic runs before using them as source snapshots."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit

from ._util import contained_file, digest, read_json
from .contracts import Source


CONTROL_FILES = ("pipeline_state.json", "resolved_config.json", "artifact_catalog.json")


def load_run(root: Path, source: Source) -> tuple[list[dict], tuple[str, ...], str]:
    payloads = [read_json(contained_file(root, name)) for name in CONTROL_FILES]
    state, snapshot, catalog = payloads
    if not all(isinstance(value, dict) for value in payloads):
        raise ValueError("Run control artifacts must be JSON objects")
    _check_state(state)
    config = snapshot.get("config", {})
    profile = config.get("source_profile", {})
    if profile.get("kind") != "website" or profile.get("source_id") != source.source_id:
        raise ValueError("Expected a generic website run for this exact source_id")
    if config.get("crawler", {}).get("fail_on_incomplete_frontier") is not True:
        raise ValueError("Website import requires the complete-frontier crawl guard")
    hosts = tuple(profile.get("allowed_hosts") or ())
    if not hosts or urlsplit(source.uri).hostname not in hosts:
        raise ValueError("Website source origin is outside the run's allowed hosts")
    records = catalog.get("records")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise ValueError("Run artifact catalog is invalid")
    markdown = [item for item in records if item.get("artifact_type") == "markdown"]
    return markdown, hosts, digest(payloads)


def _check_state(state: dict) -> None:
    stages = state.get("stages")
    if state.get("status") != "completed" or not isinstance(stages, list) or not stages:
        raise ValueError("Only completed runs may be imported; paused/failed runs are rejected")
    for stage in stages:
        if stage.get("status") not in {"completed", "skipped"}:
            raise ValueError("Website run has an incomplete stage")
        if float(stage.get("metrics", {}).get("failed", 0)) > 0:
            raise ValueError("Website run contains failed items; resolve them before import")
    crawls = [stage for stage in stages if stage.get("stage_type") == "crawler"]
    if not crawls or any(stage.get("status") != "completed" for stage in crawls):
        raise ValueError("Website run does not have a completed crawl stage")


def check_unchanged(root: Path, expected: str) -> None:
    current = [read_json(contained_file(root, name)) for name in CONTROL_FILES]
    if digest(current) != expected:
        raise ValueError("Run artifacts changed during import; retry a completed immutable run")
