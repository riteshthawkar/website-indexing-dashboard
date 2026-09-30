"""Provider-neutral snapshot synchronization; does not embed, index or deploy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

from ._publish import publish
from ._sessions import begin, fail, stage
from .catalog import IngestionCatalog
from .contracts import Collection, SnapshotEnd, SourceConnector, SourceItem
from .records import build_record, validate_item


class SyncError(RuntimeError):
    def __init__(self, run_id: str, cause: BaseException):
        self.run_id = run_id
        message = f"Sync {run_id} failed ({type(cause).__name__}); this sync did not publish"
        super().__init__(message)


@dataclass(frozen=True)
class SyncReport:
    run_id: str
    source_id: str
    counts: dict[str, int]


def _validate_destination(catalog: IngestionCatalog, connector: SourceConnector) -> None:
    uri = urlsplit(connector.source.uri)
    if uri.scheme == "file":
        root = Path(unquote(uri.path)).resolve()
        if catalog.root.is_relative_to(root):
            raise ValueError("Catalog storage must be outside the ingested source directory")


def _stage_snapshot(catalog, collection, connector, session, max_bytes, max_items) -> None:
    ended = False
    count = 0
    for event in connector.scan():
        if ended:
            raise ValueError("Connector emitted data after the snapshot end")
        if isinstance(event, SnapshotEnd):
            if event.authoritative is not True:
                raise ValueError("Incomplete snapshots cannot be published")
            ended = True
            continue
        if not isinstance(event, SourceItem):
            raise TypeError("Connector emitted an unsupported event")
        count += 1
        if count > max_items:
            raise ValueError("Source item limit exceeded; snapshot was not truncated")
        validate_item(event, max_bytes)
        record = build_record(collection, connector.source, event)
        catalog.blobs.put(collection.key, record.content_hash, event.body)
        stage(catalog, session, record)
    if not ended:
        raise ValueError("Connector did not confirm a complete snapshot")


def sync_source(
    catalog: IngestionCatalog,
    collection: Collection,
    connector: SourceConnector,
    *,
    resume_run_id: str | None = None,
    max_item_bytes: int = 32 * 1024 * 1024,
    max_items: int = 100_000,
) -> SyncReport:
    """Atomic source refresh. Resume replays discovery and reuses immutable payloads.

    A caller explicitly resuming a running run takes ownership from its old worker.
    Publication is guarded against concurrent, newer successful syncs.
    """
    for value in (max_item_bytes, max_items):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Sync size limits must be positive integers")
    _validate_destination(catalog, connector)
    fingerprint = connector.fingerprint
    source = connector.source
    session = begin(catalog, collection, connector, resume_run_id)
    try:
        _stage_snapshot(catalog, collection, connector, session, max_item_bytes, max_items)
        if connector.fingerprint != fingerprint or connector.source != source:
            raise ValueError("Connector configuration changed during the sync")
        counts = publish(catalog, session, source)
    except BaseException as error:
        fail(catalog, session, error)
        if not isinstance(error, Exception):
            raise
        raise SyncError(session.run_id, error) from error
    return SyncReport(session.run_id, source.source_id, counts)


__all__ = ["SyncError", "SyncReport", "sync_source"]
