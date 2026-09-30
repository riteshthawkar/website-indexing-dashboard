"""Permission-checked conversion receipts and fenced publication of derived artifacts."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from dataclasses import dataclass
from uuid import uuid4

from ._database import connect, transaction
from ._util import canonical_json, digest, now
from .documents import PreparedDocument
from .records import ContentRecord


@dataclass(frozen=True)
class PreparedRecord:
    record: ContentRecord
    document: PreparedDocument
    processor_key: str
    output_hash: str

    @property
    def revision_id(self):
        return "prepared:" + digest([self.record.revision_id, self.processor_key, self.output_hash])


def read_prepared(catalog, collection, record_id, *, access, processor_key=None) -> PreparedRecord:
    """No stale receipts, cross-workspace access or cached-ACL bypasses."""
    with closing(connect(catalog.db_path)) as db:
        record = _record(db, catalog, collection, record_id, access)
        receipt = _receipt(db, record, processor_key)
    body = catalog.blobs.read(collection.key, receipt["output_hash"], receipt["output_bytes"])
    result = PreparedDocument.from_dict(json.loads(body))
    with closing(connect(catalog.db_path)) as db:
        current = _record(db, catalog, collection, record_id, access)
        latest = _receipt(db, current, processor_key)
        if current.revision_id != record.revision_id or dict(latest) != dict(receipt):
            raise ValueError("Conversion changed during read; retry")
    return PreparedRecord(record, result, receipt["processor_key"], receipt["output_hash"])


def _record(db, catalog, collection, record_id, access):
    catalog._check_workspace(collection, access)
    row = db.execute(
        "SELECT r.payload,s.config FROM records r JOIN sources s USING(source_key) "
        "WHERE s.collection_key=? AND r.record_id=? AND r.deleted=0",
        (collection.key, record_id),
    ).fetchone()
    record = catalog._visible(row, access) if row is not None else None
    if record is None:
        raise PermissionError("Record is unavailable to this caller")
    return record


def prepared_revision(catalog, collection, record_id, access, processor_key=None):
    with closing(connect(catalog.db_path)) as db:
        record = _record(db, catalog, collection, record_id, access)
        receipt = _receipt(db, record, processor_key)
        return "prepared:" + digest([
            record.revision_id, receipt["processor_key"], receipt["output_hash"],
        ])


def _receipt(db, record, processor_key=None):
    row = db.execute("SELECT * FROM prepared_records WHERE record_id=?",
                     (record.record_id,)).fetchone()
    if (row is None or row["input_key"] != record.processing_key
            or (processor_key is not None and row["processor_key"] != processor_key)):
        raise ValueError(f"Record {record.record_id} requires conversion before evaluation")
    return row


def claim(catalog, collection, record, access):
    token = uuid4().hex
    with transaction(catalog.db_path) as db:
        current = _record(db, catalog, collection, record.record_id, access)
        if current.revision_id != record.revision_id:
            raise ValueError("Source changed before conversion")
        db.execute("INSERT OR REPLACE INTO preparation_claims(record_id,token) VALUES(?,?)",
                   (record.record_id, token))
    return token


def publish(catalog, collection, record, document, processor_key, token, access):
    body = canonical_json(document.to_dict()).encode("utf-8")
    content_hash = hashlib.sha256(body).hexdigest()
    with transaction(catalog.db_path) as db:
        current = _record(db, catalog, collection, record.record_id, access)
        owner = db.execute("SELECT token FROM preparation_claims WHERE record_id=?",
                           (record.record_id,)).fetchone()
        if (current.processing_key != record.processing_key
                or owner is None or owner["token"] != token):
            raise ValueError("Conversion was superseded; stale output was not published")
        catalog.blobs.put(collection.key, content_hash, body)
        db.execute("INSERT OR REPLACE INTO prepared_records VALUES(?,?,?,?,?,?)",
                   (record.record_id, record.processing_key, processor_key,
                    content_hash, len(body), now()))
        db.execute("DELETE FROM preparation_claims WHERE record_id=?", (record.record_id,))


__all__ = ["PreparedRecord", "read_prepared"]
