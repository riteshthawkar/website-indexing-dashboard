"""Strict, credential-free collection manifests shared by CLI and Python callers."""

from __future__ import annotations

from pathlib import Path

from ._util import read_json
from .contracts import AccessPolicy, Collection, Source, SourceConnector, SyncPolicy
from .folders import LocalFolderConnector
from .websites import WebsiteRunConnector


MANIFEST_SCHEMA_VERSION = "lawa.ingestion.collection.v1"


def load_collection(path: str | Path) -> tuple[Collection, tuple[SourceConnector, ...]]:
    """Local paths in connector options are relative to the manifest, never to cwd."""
    manifest = Path(path).expanduser().resolve()
    data = read_json(manifest, limit=1024 * 1024)
    if not isinstance(data, dict) or set(data) != {"schema_version", "collection", "sources"}:
        raise ValueError("Collection manifest requires schema_version, collection and sources only")
    if data["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported collection manifest version")
    collection = Collection(**data["collection"])
    sources = data["sources"]
    if not isinstance(sources, list) or not sources:
        raise ValueError("At least one source is required")
    connectors = tuple(_connector(value, manifest.parent) for value in sources)
    ids = [value.source.source_id for value in connectors]
    if len(ids) != len(set(ids)):
        raise ValueError("Source IDs must be unique within a collection")
    return collection, connectors


def _connector(value: dict, root: Path) -> SourceConnector:
    if not isinstance(value, dict):
        raise ValueError("Each source must be an object")
    values = dict(value)
    options = values.pop("connector", {})
    if not isinstance(options, dict):
        raise ValueError("connector must be an object")
    values["access"] = AccessPolicy(**values.get("access", {}))
    values["sync_policy"] = SyncPolicy(**values.get("sync_policy", {}))
    source = Source(**values)
    if source.kind == "folder":
        return LocalFolderConnector(source, **options)
    if source.kind == "s3":
        from .object_storage import S3Connector

        if "client" in options:
            raise ValueError("SDK clients cannot be supplied in a JSON manifest")
        return S3Connector(source, **options)
    if source.kind == "website":
        options = dict(options)
        if "run_dir" not in options:
            raise ValueError("Website sources require connector.run_dir")
        options["run_dir"] = root / options["run_dir"]
        return WebsiteRunConnector(source, **options)
    raise ValueError(f"No connector for {source.kind!r}; use the Python extension contract")


def load_conversion_options(path: str | Path):
    from .documents import ConversionOptions

    manifest = Path(path).expanduser().resolve()
    data = read_json(manifest, limit=1024 * 1024)
    if (not isinstance(data, dict) or set(data) != {"schema_version", "options"}
            or data["schema_version"] != "lawa.ingestion.conversion.v1"):
        raise ValueError("Invalid conversion manifest")
    options = dict(data["options"])
    for key in ("docling_artifacts_path", "ocr_model_path"):
        if options.get(key):
            options[key] = str((manifest.parent / options[key]).resolve())
    return ConversionOptions(**options)


__all__ = ["MANIFEST_SCHEMA_VERSION", "load_collection", "load_conversion_options"]
