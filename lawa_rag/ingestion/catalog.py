"""Local administrative catalog with permission-filtered record and payload reads."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

from ._blobs import BlobStore
from ._database import connect, initialize
from .contracts import AccessContext, AccessPolicy, Collection
from .records import ContentRecord


class IngestionCatalog:
    """Trusted local state, not an HTTP authorization service or vector index."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db_path = self.root / "catalog.sqlite3"
        self.blobs = BlobStore(self.root / "blobs")
        initialize(self.db_path)

    def records(
        self, collection: Collection, *, access: AccessContext,
    ) -> tuple[ContentRecord, ...]:
        self._check_workspace(collection, access)
        with closing(connect(self.db_path)) as db:
            rows = db.execute(
                "SELECT r.payload,s.config FROM records r JOIN sources s USING(source_key) "
                "WHERE s.collection_key=? AND r.deleted=0 ORDER BY r.record_id",
                (collection.key,),
            )
            return tuple(record for row in rows if (record := self._visible(row, access)))

    def read(
        self, collection: Collection, record_id: str, *, access: AccessContext,
    ) -> tuple[ContentRecord, bytes]:
        """Recheck current authorization; old record handles cannot bypass revocation."""
        self._check_workspace(collection, access)
        with closing(connect(self.db_path)) as db:
            row = db.execute(
                "SELECT r.payload,s.config FROM records r JOIN sources s USING(source_key) "
                "WHERE s.collection_key=? AND r.record_id=? AND r.deleted=0",
                (collection.key, record_id),
            ).fetchone()
        record = self._visible(row, access) if row is not None else None
        if record is None:
            raise PermissionError("Record is unavailable to this caller")
        body = self.blobs.read(collection.key, record.content_hash, record.byte_count)
        return record, body

    def changes(self, run_id: str) -> tuple[dict, ...]:
        """Administrative change feed, for trusted downstream workers only."""
        with closing(connect(self.db_path)) as db:
            rows = db.execute(
                "SELECT record_id,operation,needs_processing FROM changes "
                "WHERE run_id=? ORDER BY record_id", (run_id,),
            )
            return tuple(dict(row) for row in rows)

    def run(self, run_id: str) -> dict:
        """Administrative diagnostics; never includes connector exception text/secrets."""
        with closing(connect(self.db_path)) as db:
            row = db.execute(
                "SELECT run_id,status,started_at,finished_at,counts,error_code "
                "FROM sync_runs WHERE run_id=?", (run_id,),
            ).fetchone()
        if row is None:
            raise KeyError("Unknown sync run")
        result = dict(row)
        result["counts"] = json.loads(result["counts"])
        return result

    @staticmethod
    def _check_workspace(collection: Collection, access: AccessContext) -> None:
        if collection.workspace_id != access.workspace_id:
            raise PermissionError("Workspace mismatch")

    @staticmethod
    def _visible(row, access: AccessContext) -> ContentRecord | None:
        record = ContentRecord.from_dict(json.loads(row["payload"]))
        policy = AccessPolicy(**json.loads(row["config"])["access"])
        # Current source ACL also applies to retained records absent from this scan.
        return record if record.visible_to(access) and policy.allows(access) else None


__all__ = ["IngestionCatalog"]
