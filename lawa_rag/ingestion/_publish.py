"""Publish a complete source snapshot and its downstream change feed atomically."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict

from ._database import transaction
from ._sessions import SyncSession, require_owner
from ._util import canonical_json, now
from .catalog import IngestionCatalog
from .contracts import Source
from .records import ContentRecord, classify_change, with_source_access


def _missing(db, session: SyncSession):
    return db.execute(
        "SELECT record_id,payload FROM records WHERE source_key=? AND deleted=0 "
        "AND record_id NOT IN "
        "(SELECT record_id FROM staged_records WHERE run_id=?)",
        (session.source_key, session.run_id),
    ).fetchall()


def _check_deletions(db, session: SyncSession, source: Source, missing: list) -> int:
    observed = db.execute(
        "SELECT COUNT(*) FROM staged_records WHERE run_id=?", (session.run_id,),
    ).fetchone()[0]
    if observed == 0 and not source.sync_policy.allow_empty:
        raise ValueError("Empty snapshot blocked; explicitly enable allow_empty if intentional")
    total = db.execute(
        "SELECT COUNT(*) FROM records WHERE source_key=? AND deleted=0", (session.source_key,),
    ).fetchone()[0]
    if source.sync_policy.delete_missing and total:
        if len(missing) / total > source.sync_policy.max_delete_fraction:
            raise ValueError("Deletion safety limit exceeded; last good snapshot is unchanged")
    return observed


def _upsert(db, session: SyncSession, row) -> tuple[str, bool]:
    new = ContentRecord.from_dict(json.loads(row["payload"]))
    previous = db.execute(
        "SELECT payload FROM records WHERE record_id=? AND deleted=0", (new.record_id,),
    ).fetchone()
    old = ContentRecord.from_dict(json.loads(previous[0])) if previous else None
    operation, needs_processing = classify_change(old, new)
    db.execute(
        "INSERT INTO records(source_key,record_id,payload,deleted,run_id) VALUES(?,?,?,0,?) "
        "ON CONFLICT(record_id) DO UPDATE SET payload=excluded.payload,deleted=0,"
        "run_id=excluded.run_id",
        (session.source_key, new.record_id, row["payload"], session.run_id),
    )
    _change(db, session.run_id, new.record_id, operation, needs_processing)
    return operation, needs_processing


def _change(db, run_id: str, record_id: str, operation: str, needs_processing: bool) -> None:
    db.execute(
        "INSERT INTO changes(run_id,record_id,operation,needs_processing) VALUES(?,?,?,?)",
        (run_id, record_id, operation, int(needs_processing)),
    )


def _reconcile(db, session: SyncSession, source: Source, missing: list, counts: Counter) -> None:
    if not source.sync_policy.delete_missing:
        counts["retained_missing"] = len(missing)
        for row in missing:
            old = ContentRecord.from_dict(json.loads(row["payload"]))
            if old.source_access != source.access:
                new = with_source_access(old, source.access)
                _upsert(db, session, {"payload": canonical_json(new.to_dict())})
                counts["permissions_changed"] += 1
        return
    for row in missing:
        db.execute(
            "UPDATE records SET deleted=1,run_id=? WHERE record_id=?",
            (session.run_id, row["record_id"]),
        )
        _change(db, session.run_id, row["record_id"], "deleted", False)
    counts["deleted"] = len(missing)


def publish(catalog: IngestionCatalog, session: SyncSession, source: Source) -> dict[str, int]:
    counts = Counter({key: 0 for key in (
        "added", "updated", "reprocess", "permissions_changed", "metadata_changed", "unchanged",
        "deleted", "retained_missing", "needs_processing",
    )})
    with transaction(catalog.db_path) as db:
        require_owner(db, session)
        generation = db.execute(
            "SELECT generation FROM sources WHERE source_key=?", (session.source_key,),
        ).fetchone()[0]
        if generation != session.base_generation:
            raise RuntimeError("A newer source snapshot was published; start a new sync")
        missing = _missing(db, session)
        counts["observed"] = _check_deletions(db, session, source, missing)
        rows = db.execute(
            "SELECT record_id,payload FROM staged_records WHERE run_id=? ORDER BY record_id",
            (session.run_id,),
        ).fetchall()
        for row in rows:
            operation, needs_processing = _upsert(db, session, row)
            counts[operation] += 1
            counts["needs_processing"] += int(needs_processing)
        _reconcile(db, session, source, missing, counts)
        db.execute(
            "UPDATE sources SET generation=generation+1,config=? WHERE source_key=?",
            (canonical_json(asdict(source)), session.source_key),
        )
        db.execute(
            "UPDATE sync_runs SET status='completed',finished_at=?,counts=? WHERE run_id=?",
            (now(), canonical_json(counts), session.run_id),
        )
        db.execute("DELETE FROM staged_records WHERE run_id=?", (session.run_id,))
    return dict(counts)
