"""Derived-artifact lifecycle: invalidation, access checks, concurrent work and hard timeouts."""

import json
import multiprocessing
from contextlib import closing
from types import SimpleNamespace

import pytest

from lawa_rag.ingestion import (
    AccessContext, AccessPolicy, Collection, ConversionFailure, ConversionOptions,
    DocumentProcessor, IngestionCatalog, LocalFolderConnector, Source, build_evaluation_corpus,
    preparation_report, prepare_collection, read_prepared, sync_source, update_source_access,
)
from lawa_rag.ingestion.cli import main
from pipeline.tests.lawa_rag_document_support import hang_worker


def setup_collection(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    path = root / "record.json"
    path.write_text('{"code":"007","policy":"original"}')
    catalog, collection, access = (IngestionCatalog(tmp_path / "catalog"),
                                    Collection("w", "c"), AccessContext("w", ("team",)))
    connector = LocalFolderConnector(Source("docs", "folder", root.as_uri(),
                                             AccessPolicy(principals=("team",))))
    sync_source(catalog, collection, connector)
    return catalog, collection, access, connector, path


def test_preparation_cli_cache_invalidation_and_revocation(tmp_path, capsys):
    catalog, collection, access, connector, path = setup_collection(tmp_path)
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({
        "schema_version": "lawa.ingestion.collection.v1",
        "collection": {"workspace_id": "w", "collection_id": "c"},
        "sources": [{"source_id": "docs", "kind": "folder", "uri": path.parent.as_uri(),
                     "access": {"principals": ["team"]}}],
    }))
    assert main(["prepare", "--config", str(manifest), "--catalog", str(catalog.root),
                 "--principal", "team"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["converted"] == 1
    assert preparation_report(catalog, report["run_id"]) == report
    assert prepare_collection(catalog, collection, access=access).counts["reused"] == 1
    before = build_evaluation_corpus(catalog, collection, access=access)
    record_id = before.sources[0].source_id
    path.write_text('{"code":"007","policy":"changed"}')
    sync_source(catalog, collection, connector)
    with pytest.raises(ValueError, match="requires conversion"):
        read_prepared(catalog, collection, record_id, access=access)
    assert prepare_collection(catalog, collection, access=access).counts["converted"] == 1
    after = build_evaluation_corpus(catalog, collection, access=access)
    assert before.snapshot_id != after.snapshot_id
    assert all("original" not in unit.text for unit in after.evidence)
    with pytest.raises(ValueError, match="requires conversion"):
        build_evaluation_corpus(catalog, collection, access=access, processor_key="another-recipe")
    update_source_access(catalog, collection, "docs", AccessPolicy())
    with pytest.raises(PermissionError):
        read_prepared(catalog, collection, record_id, access=access)


def test_late_conversion_cannot_publish_after_newer_worker_or_permission_change(tmp_path):
    catalog, collection, access, connector, path = setup_collection(tmp_path)

    class RacingProcessor(DocumentProcessor):
        def __init__(self, callback):
            super().__init__()
            self.callback = callback

        def convert(self, record, body):
            result = super().convert(record, body)
            self.callback()
            return result

    newest = DocumentProcessor(ConversionOptions(recipe_version="new-recipe"))
    stale = prepare_collection(catalog, collection, access=access, processor=RacingProcessor(
        lambda: prepare_collection(catalog, collection, access=access, processor=newest),
    ))
    assert not stale.ok and stale.counts["failed"] == 1
    record = catalog.records(collection, access=access)[0]
    assert read_prepared(catalog, collection, record.record_id,
                         access=access).processor_key == newest.fingerprint

    def change_source():
        path.write_text('{"policy":"updated-during-conversion"}')
        sync_source(catalog, collection, connector)

    report = prepare_collection(catalog, collection, access=access,
                                 processor=RacingProcessor(change_source))
    assert not report.ok
    with pytest.raises(ValueError, match="requires conversion"):
        read_prepared(catalog, collection, record.record_id, access=access)
    report = prepare_collection(catalog, collection, access=access, processor=RacingProcessor(
        lambda: update_source_access(catalog, collection, "docs", AccessPolicy()),
    ))
    assert report.records[0]["error_code"] == "access_revoked"


def test_hung_conversion_worker_is_killed_and_next_attempt_can_succeed(monkeypatch):
    from lawa_rag.ingestion import _document_worker

    before = {child.pid for child in multiprocessing.active_children()}
    record = SimpleNamespace(locator="file:///record.json", media_type="application/json")
    with monkeypatch.context() as patch:
        patch.setattr(_document_worker, "_serve", hang_worker)
        with DocumentProcessor(ConversionOptions(timeout_seconds=1)) as processor:
            with pytest.raises(ConversionFailure, match="conversion_timeout"):
                processor.convert(record, b'{"valid":true}')
    assert {child.pid for child in multiprocessing.active_children()} == before
    result = DocumentProcessor().convert(record, b'{"valid":true}')
    assert result.blocks[0].data_json == '{"valid":true}'


def test_healthy_worker_shuts_down_gracefully_instead_of_being_terminated(tmp_path, monkeypatch):
    from lawa_rag.ingestion import _document_worker
    from pipeline.tests.lawa_rag_document_support import graceful_worker

    monkeypatch.setattr(_document_worker, "_serve", graceful_worker)
    record = SimpleNamespace(locator="file:///record.json", media_type="application/json")
    before = {child.pid for child in multiprocessing.active_children()}
    with DocumentProcessor(ConversionOptions(docling_artifacts_path=str(tmp_path))) as processor:
        assert processor.convert(record, b'{"valid":true}').blocks
    assert (tmp_path / "clean-shutdown").exists()
    assert {child.pid for child in multiprocessing.active_children()} == before


def test_catalog_v1_migration_preserves_records_and_source_bindings(tmp_path):
    from lawa_rag.ingestion._database import connect

    catalog, collection, access, _, _ = setup_collection(tmp_path)
    records = catalog.records(collection, access=access)
    # Reconstruct the previous on-disk schema, retaining its real records and change feed.
    with closing(connect(catalog.db_path)) as db:
        for name in ("prepared_records", "preparation_claims", "preparation_runs"):
            db.execute(f"DROP TABLE {name}")
        db.execute("ALTER TABLE sources DROP COLUMN connector_binding")
        db.execute("PRAGMA user_version=1")
        db.commit()
    migrated = IngestionCatalog(catalog.root)
    assert migrated.records(collection, access=access) == records
    assert prepare_collection(migrated, collection, access=access).ok
    assert read_prepared(migrated, collection, records[0].record_id, access=access)
