"""Versioned records preserve typed payloads, source identity and permission provenance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

from ._util import canonical_json, digest, nonempty
from .contracts import AccessContext, AccessPolicy, Collection, Source, SourceItem


RECORD_SCHEMA_VERSION = "lawa.ingestion.record.v1"


def validate_item(item: SourceItem, max_bytes: int) -> None:
    for name in ("native_id", "media_type", "locator", "title", "language"):
        nonempty(getattr(item, name), name)
    if not isinstance(item.body, bytes) or len(item.body) > max_bytes:
        raise ValueError("Item body must be bytes within the configured size limit")
    if item.kind not in {"document", "table", "record", "image", "audio", "video"}:
        raise ValueError("Unsupported content kind")
    if item.processing_state not in {"raw", "ready"}:
        raise ValueError("Unsupported processing state")
    if not isinstance(json.loads(item.metadata_json), dict):
        raise ValueError("metadata_json must contain a JSON object")
    canonical_json(json.loads(item.metadata_json))
    if item.access is not None and not isinstance(item.access, AccessPolicy):
        raise ValueError("Item access must be an AccessPolicy")


@dataclass(frozen=True)
class ContentRecord:
    record_id: str
    revision_id: str
    workspace_id: str
    collection_id: str
    source_id: str
    source_kind: str
    native_id: str
    kind: str
    media_type: str
    locator: str
    title: str
    language: str
    processing_state: str
    content_hash: str
    byte_count: int
    processing_key: str
    modified_at: str
    metadata_json: str
    source_access: AccessPolicy
    item_access: AccessPolicy
    schema_version: str = RECORD_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentRecord:
        values = dict(data)
        if values.get("schema_version") != RECORD_SCHEMA_VERSION:
            raise ValueError("Unsupported content record schema")
        for key in ("source_access", "item_access"):
            values[key] = AccessPolicy(**values[key])
        return cls(**values)

    def visible_to(self, context: AccessContext) -> bool:
        return (
            self.workspace_id == context.workspace_id
            and self.source_access.allows(context)
            and self.item_access.allows(context)
        )


def build_record(collection: Collection, source: Source, item: SourceItem) -> ContentRecord:
    content_hash = hashlib.sha256(item.body).hexdigest()
    metadata_json = canonical_json(json.loads(item.metadata_json))
    record_id = "record:" + digest([collection.key, source.source_id, item.native_id])
    values = {
        "record_id": record_id, **asdict(collection), "source_id": source.source_id,
        "source_kind": source.kind, "native_id": item.native_id, "kind": item.kind,
        "media_type": item.media_type, "locator": item.locator, "title": item.title,
        "language": item.language, "processing_state": item.processing_state,
        "content_hash": content_hash, "byte_count": len(item.body),
        "modified_at": item.modified_at, "metadata_json": metadata_json,
        "source_access": asdict(source.access),
        "item_access": asdict(item.access or AccessPolicy(public=True)),
    }
    # ACL and mtime changes must not trigger OCR, captioning or embedding again.
    processing = {key: values[key] for key in (
        "content_hash", "kind", "media_type", "title", "language", "processing_state",
        "metadata_json",
    )}
    values["processing_key"] = digest([source.processing_version, processing])
    values["revision_id"] = "revision:" + digest(values)
    values["schema_version"] = RECORD_SCHEMA_VERSION
    return ContentRecord.from_dict(values)


def classify_change(old: ContentRecord | None, new: ContentRecord) -> tuple[str, bool]:
    if old is None:
        return "added", True
    if old.content_hash != new.content_hash:
        return "updated", True
    if old.processing_key != new.processing_key:
        return "reprocess", True
    if (old.source_access, old.item_access) != (new.source_access, new.item_access):
        return "permissions_changed", False
    if old.revision_id != new.revision_id:
        return "metadata_changed", False
    return "unchanged", False


def with_source_access(record: ContentRecord, policy: AccessPolicy) -> ContentRecord:
    values = record.to_dict()
    values.pop("revision_id")
    values.pop("schema_version")
    values["source_access"] = asdict(policy)
    values["revision_id"] = "revision:" + digest(values)
    values["schema_version"] = RECORD_SCHEMA_VERSION
    return ContentRecord.from_dict(values)


__all__ = ["ContentRecord", "RECORD_SCHEMA_VERSION"]
