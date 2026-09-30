"""Apply administrative source ACL changes without contacting an unavailable source."""

from __future__ import annotations

import json
from dataclasses import asdict
from uuid import uuid4

from ._database import transaction
from ._publish import _upsert
from ._sessions import SyncSession
from ._util import canonical_json, digest, identifier, now
from .catalog import IngestionCatalog
from .contracts import AccessPolicy, Collection
from .records import ContentRecord, with_source_access
from .sync import SyncReport


def update_source_access(
    catalog: IngestionCatalog, collection: Collection, source_id: str, policy: AccessPolicy,
) -> SyncReport:
    """Trusted administration only; update the source manifest to match before its next sync."""
    identifier(source_id, "source_id")
    if not isinstance(policy, AccessPolicy):
        raise ValueError("policy must be an AccessPolicy")
    source_key = digest([collection.key, source_id])
    with transaction(catalog.db_path) as db:
        state = db.execute("SELECT * FROM sources WHERE source_key=?", (source_key,)).fetchone()
        if state is None:
            raise KeyError("Unknown source in this collection")
        config = json.loads(state["config"])
        config["access"] = asdict(policy)
        session = SyncSession(uuid4().hex, source_key, uuid4().hex, state["generation"])
        db.execute(
            "INSERT INTO sync_runs(run_id,source_key,fingerprint,base_generation,token,"
            "status,started_at) VALUES(?,?,?,?,?,'running',?)",
            (session.run_id, source_key, digest(config), state["generation"], session.token, now()),
        )
        counts = _update_records(db, session, policy)
        db.execute(
            "UPDATE sources SET config=?,generation=generation+1 WHERE source_key=?",
            (canonical_json(config), source_key),
        )
        db.execute(
            "UPDATE sync_runs SET status='completed',counts=?,finished_at=? WHERE run_id=?",
            (canonical_json(counts), now(), session.run_id),
        )
    return SyncReport(session.run_id, source_id, counts)


def _update_records(db, session, policy) -> dict[str, int]:
    rows = db.execute(
        "SELECT payload FROM records WHERE source_key=? AND deleted=0", (session.source_key,),
    ).fetchall()
    counts = {"permissions_changed": 0, "unchanged": 0, "needs_processing": 0,
              "observed": len(rows)}
    for row in rows:
        previous = ContentRecord.from_dict(json.loads(row["payload"]))
        new = with_source_access(previous, policy)
        operation, _ = _upsert(db, session, {"payload": canonical_json(new.to_dict())})
        counts[operation] += 1
    return counts


__all__ = ["update_source_access"]
