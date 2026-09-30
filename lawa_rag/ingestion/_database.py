"""SQLite catalog: short transactions, staged snapshots and optimistic publication."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources(
    source_key TEXT PRIMARY KEY, collection_key TEXT NOT NULL,
    kind TEXT NOT NULL, uri TEXT NOT NULL, config TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS sync_runs(
    run_id TEXT PRIMARY KEY, source_key TEXT NOT NULL REFERENCES sources(source_key),
    fingerprint TEXT NOT NULL, base_generation INTEGER NOT NULL,
    token TEXT NOT NULL, status TEXT NOT NULL, started_at TEXT NOT NULL,
    finished_at TEXT, counts TEXT NOT NULL DEFAULT '{}', error_code TEXT
);
CREATE TABLE IF NOT EXISTS staged_records(
    run_id TEXT NOT NULL REFERENCES sync_runs(run_id), record_id TEXT NOT NULL,
    payload TEXT NOT NULL, PRIMARY KEY(run_id, record_id)
);
CREATE TABLE IF NOT EXISTS records(
    source_key TEXT NOT NULL REFERENCES sources(source_key), record_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0,
    run_id TEXT NOT NULL REFERENCES sync_runs(run_id)
);
CREATE TABLE IF NOT EXISTS changes(
    run_id TEXT NOT NULL REFERENCES sync_runs(run_id), record_id TEXT NOT NULL,
    operation TEXT NOT NULL, needs_processing INTEGER NOT NULL,
    PRIMARY KEY(run_id, record_id)
);
CREATE INDEX IF NOT EXISTS source_collection ON sources(collection_key);
CREATE INDEX IF NOT EXISTS record_source ON records(source_key, deleted);
"""


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA synchronous=FULL")
    return db


def initialize(path: Path) -> None:
    db = connect(path)
    try:
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in (0, 1, 2):
            raise ValueError("Unsupported ingestion catalog version")
        db.execute("PRAGMA journal_mode=WAL")
        db.executescript(SCHEMA)
        _migrate_v2(db)
        db.commit()
    finally:
        db.close()


def _migrate_v2(db) -> None:
    db.execute("BEGIN IMMEDIATE")
    columns = {row[1] for row in db.execute("PRAGMA table_info(sources)")}
    if "connector_binding" not in columns:
        db.execute("ALTER TABLE sources ADD COLUMN connector_binding TEXT NOT NULL DEFAULT ''")
    db.execute("""CREATE TABLE IF NOT EXISTS prepared_records(
        record_id TEXT PRIMARY KEY REFERENCES records(record_id),
        input_key TEXT NOT NULL, processor_key TEXT NOT NULL,
        output_hash TEXT NOT NULL, output_bytes INTEGER NOT NULL, created_at TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS preparation_claims(
        record_id TEXT PRIMARY KEY REFERENCES records(record_id), token TEXT NOT NULL
    )""")
    db.execute("""CREATE TABLE IF NOT EXISTS preparation_runs(
        run_id TEXT PRIMARY KEY, report TEXT NOT NULL
    )""")
    db.execute("PRAGMA user_version=2")


@contextmanager
def transaction(path: Path) -> Iterator[sqlite3.Connection]:
    db = connect(path)
    try:
        db.execute("BEGIN IMMEDIATE")
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()
