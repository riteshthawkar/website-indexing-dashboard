"""Staging ownership prevents stale workers and unsafe resumes from publishing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from uuid import uuid4

from ._database import transaction
from ._util import canonical_json, digest, now
from .catalog import IngestionCatalog
from .contracts import Collection, SourceConnector


@dataclass(frozen=True)
class SyncSession:
    run_id: str
    source_key: str
    token: str
    base_generation: int


def begin(
    catalog: IngestionCatalog, collection: Collection, connector: SourceConnector,
    resume_run_id: str | None,
) -> SyncSession:
    source = connector.source
    source_key = source.key(collection)
    fingerprint = digest([source.fingerprint, connector.fingerprint])
    with transaction(catalog.db_path) as db:
        db.execute(
            "INSERT OR IGNORE INTO sources(source_key,collection_key,kind,uri,config) "
            "VALUES(?,?,?,?,?)",
            (source_key, collection.key, source.kind, source.uri, canonical_json(asdict(source))),
        )
        state = db.execute("SELECT * FROM sources WHERE source_key=?", (source_key,)).fetchone()
        if (state["kind"], state["uri"]) != (source.kind, source.uri):
            raise ValueError("Source identity is bound to another location; use a new source_id")
        binding = getattr(connector, "binding_key", digest([source.kind, source.uri]))
        if state["connector_binding"] and state["connector_binding"] != binding:
            raise ValueError("Connector endpoint changed; use a new source_id")
        db.execute("UPDATE sources SET connector_binding=? WHERE source_key=?",
                   (binding, source_key))
        token = uuid4().hex
        if resume_run_id:
            return _resume(db, resume_run_id, source_key, fingerprint, token, state["generation"])
        run_id = uuid4().hex
        db.execute(
            "INSERT INTO sync_runs(run_id,source_key,fingerprint,base_generation,token,"
            "status,started_at) VALUES(?,?,?,?,?,'running',?)",
            (run_id, source_key, fingerprint, state["generation"], token, now()),
        )
    return SyncSession(run_id, source_key, token, state["generation"])


def _resume(db, run_id, source_key, fingerprint, token, generation) -> SyncSession:
    row = db.execute("SELECT * FROM sync_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None or row["source_key"] != source_key or row["fingerprint"] != fingerprint:
        raise ValueError("Resume requires the same collection, source and connector configuration")
    if row["status"] == "completed" or row["base_generation"] != generation:
        raise ValueError("Cannot resume a completed or superseded sync; start a new sync")
    db.execute(
        "UPDATE sync_runs SET token=?,status='running',error_code=NULL,finished_at=NULL "
        "WHERE run_id=?", (token, run_id),
    )
    # Snapshot connectors replay discovery; do not carry vanished items across attempts.
    db.execute("DELETE FROM staged_records WHERE run_id=?", (run_id,))
    return SyncSession(run_id, source_key, token, generation)


def require_owner(db, session: SyncSession) -> None:
    row = db.execute(
        "SELECT token,status FROM sync_runs WHERE run_id=?", (session.run_id,),
    ).fetchone()
    if row is None or row["token"] != session.token or row["status"] != "running":
        raise RuntimeError("Sync ownership changed; this attempt must stop")


def stage(catalog: IngestionCatalog, session: SyncSession, record) -> None:
    with transaction(catalog.db_path) as db:
        require_owner(db, session)
        db.execute(
            "INSERT INTO staged_records(run_id,record_id,payload) VALUES(?,?,?)",
            (session.run_id, record.record_id, canonical_json(record.to_dict())),
        )


def fail(catalog: IngestionCatalog, session: SyncSession, error: BaseException) -> None:
    with transaction(catalog.db_path) as db:
        db.execute(
            "UPDATE sync_runs SET status='failed',error_code=?,finished_at=? "
            "WHERE run_id=? AND token=? AND status='running'",
            (type(error).__name__, now(), session.run_id, session.token),
        )
