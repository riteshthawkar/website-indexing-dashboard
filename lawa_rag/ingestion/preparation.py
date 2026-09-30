"""Resumable preparation: successful per-record conversions are reused on the next run."""

from __future__ import annotations

import json
from contextlib import closing
from dataclasses import asdict, dataclass
from uuid import uuid4

from ._database import connect, transaction
from ._prepared import claim, publish, read_prepared
from ._util import canonical_json, now
from .documents import ConversionFailure, DocumentProcessor


@dataclass(frozen=True)
class PreparationReport:
    run_id: str
    processor_key: str
    status: str
    started_at: str
    finished_at: str
    records: tuple[dict, ...]
    error_code: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    @property
    def counts(self) -> dict:
        return {key: sum(item["status"] == key for item in self.records)
                for key in ("converted", "reused", "ready", "failed")}

    def to_dict(self):
        return {**asdict(self), "ok": self.ok, "counts": self.counts}


def prepare_collection(catalog, collection, *, access, processor=None, source_ids=None):
    """Synchronous, bounded processing for authenticated local workers, not an HTTP API."""
    processor = processor or DocumentProcessor()
    key, started, run_id = processor.fingerprint, now(), uuid4().hex
    records = catalog.records(collection, access=access)
    if source_ids is not None:
        records = tuple(record for record in records if record.source_id in source_ids)
    if not records:
        raise ValueError("No authorized records are available to prepare")
    results = []
    with processor:
        for record in records:
            results.append(_prepare_one(catalog, collection, record, access, processor, key))
    current = catalog.records(collection, access=access)
    if source_ids is not None:
        current = tuple(record for record in current if record.source_id in source_ids)
    error_code = "collection_changed_during_preparation" if current != records else ""
    status = "failed" if any(item["status"] == "failed" for item in results) else "completed"
    report = PreparationReport(run_id, key, "failed" if error_code else status,
                               started, now(), tuple(results), error_code)
    with transaction(catalog.db_path) as db:
        db.execute("INSERT INTO preparation_runs VALUES(?,?)",
                   (run_id, canonical_json(report.to_dict())))
    return report


def _prepare_one(catalog, collection, record, access, processor, key):
    result = {"record_id": record.record_id}
    try:
        current, body = catalog.read(collection, record.record_id, access=access)
        if current.revision_id != record.revision_id:
            raise ValueError("Source changed before conversion")
        if record.processing_state == "ready" and record.media_type in {
            "text/plain", "text/markdown",
        }:
            return {**result, "status": "ready"}
        try:
            cached = read_prepared(catalog, collection, record.record_id,
                                   access=access, processor_key=key)
        except ValueError:
            cached = None
        if cached:
            return {**result, "status": "reused", "warnings": cached.document.warnings}
        token = claim(catalog, collection, record, access)
        document = processor.convert(record, body)
        document.validate(processor.options)
        publish(catalog, collection, record, document, key, token, access)
        return {**result, "status": "converted", "warnings": document.warnings}
    except ConversionFailure as error:
        code = error.code
    except PermissionError:
        code = "access_revoked"
    except (ValueError, OSError):
        code = "source_or_output_changed"
    return {**result, "status": "failed", "error_code": code}


def preparation_report(catalog, run_id):
    """Administrative diagnostics: only expose through an authenticated worker service."""
    with closing(connect(catalog.db_path)) as db:
        row = db.execute("SELECT report FROM preparation_runs WHERE run_id=?", (run_id,)).fetchone()
    if row is None:
        raise KeyError("Unknown preparation run")
    return json.loads(row["report"])


__all__ = ["PreparationReport", "prepare_collection", "preparation_report"]
