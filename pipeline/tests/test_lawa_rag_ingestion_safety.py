"""Fail-closed source refresh, access control and filesystem boundary contracts."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lawa_rag.ingestion import (
    AccessContext, AccessPolicy, Collection, IngestionCatalog, LocalFolderConnector, Source,
    SyncError, SyncPolicy, sync_source, update_source_access,
)
from pipeline.tests.lawa_rag_ingestion_support import FixtureConnector, make_website, source_item


def test_failed_snapshot_can_resume_without_publishing_partial_or_disappeared_items(tmp_path):
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    source = Source("api", "api", "https://source.example/", AccessPolicy(public=True))
    original = source_item("a", "Last good value")
    connector = FixtureConnector(source, (original,))
    sync_source(catalog, collection, connector)
    connector.items = (source_item("a", "New value"), source_item("removed-on-resume", "staged"))
    connector.error = True
    with pytest.raises(SyncError) as error:
        sync_source(catalog, collection, connector)
    failed_id = error.value.run_id
    assert catalog.run(failed_id)["status"] == "failed"
    assert "secret-must-not-be-stored" not in json.dumps(catalog.run(failed_id))
    access = AccessContext("w")
    record, = catalog.records(collection, access=access)
    assert catalog.read(collection, record.record_id, access=access)[1] == original.body
    connector.error = False
    connector.items = (source_item("a", "Recovered value"),)
    report = sync_source(catalog, collection, connector, resume_run_id=failed_id)
    assert report.run_id == failed_id and report.counts["updated"] == 1
    assert len(catalog.records(collection, access=access)) == 1
    with pytest.raises(ValueError, match="completed or superseded"):
        sync_source(catalog, collection, connector, resume_run_id=failed_id)


def test_missing_end_duplicate_records_empty_and_mass_deletion_cannot_replace_good_data(tmp_path):
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    source = Source("api", "api", "https://source.example/", AccessPolicy(public=True),
                    sync_policy=SyncPolicy(delete_missing=True))
    items = tuple(source_item(str(index), "Content") for index in range(5))
    sync_source(catalog, collection, FixtureConnector(source, items))
    for connector in (
        FixtureConnector(source, items[:1], complete=False),
        FixtureConnector(source, items[:1]),
        FixtureConnector(source, ()),
        FixtureConnector(source, (items[0], items[0])),
    ):
        with pytest.raises(SyncError):
            sync_source(catalog, collection, connector)
        assert len(catalog.records(collection, access=AccessContext("w"))) == 5
    approved = replace(source, sync_policy=SyncPolicy(True, 1.0, True))
    deleted = sync_source(catalog, collection, FixtureConnector(approved, ()))
    assert deleted.counts["deleted"] == 5
    assert not catalog.records(collection, access=AccessContext("w"))


def test_permissions_scope_and_revocation_apply_to_records_and_payloads(tmp_path):
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    source = Source("private", "api", "https://private.example/",
                    AccessPolicy(principals=("student", "staff")))
    item = source_item("policy", "Staff-only policy", access=AccessPolicy(principals=("staff",)))
    connector = FixtureConnector(source, (item,))
    sync_source(catalog, collection, connector)
    staff = AccessContext("w", ("staff",))
    record, = catalog.records(collection, access=staff)
    assert not catalog.records(collection, access=AccessContext("w", ("student",)))
    assert not catalog.records(Collection("w", "other"), access=staff)
    with pytest.raises(PermissionError):
        catalog.read(collection, record.record_id, access=AccessContext("other", ("staff",)))
    with pytest.raises(PermissionError):
        catalog.read(collection, record.record_id, access=AccessContext("w"))
    report = update_source_access(catalog, collection, source.source_id, AccessPolicy())
    assert report.counts["permissions_changed"] == 1 and report.counts["needs_processing"] == 0
    with pytest.raises(PermissionError):
        catalog.read(collection, record.record_id, access=staff)
    restored = update_source_access(catalog, collection, source.source_id, source.access)
    assert restored.counts["permissions_changed"] == 1
    assert catalog.read(collection, record.record_id, access=staff)[1] == item.body
    other = Collection("other", "c")
    connector.source = replace(source, access=AccessPolicy(public=True))
    sync_source(catalog, other, connector)
    second, = catalog.records(other, access=AccessContext("other", ("staff",)))
    assert second.record_id != record.record_id


def test_folder_escape_limits_and_inventory_mutation_fail_closed(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("Not permitted")
    (root / "link.txt").symlink_to(outside)
    source = Source("folder", "folder", root.as_uri(), AccessPolicy(public=True))
    connector = LocalFolderConnector(source)
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    with pytest.raises(SyncError):
        sync_source(catalog, collection, connector)
    (root / "link.txt").unlink()
    (root / "a.txt").write_text("Existing content")
    with pytest.raises(SyncError):
        sync_source(catalog, collection, replace(connector, max_file_bytes=2))
    events = iter(connector.scan())
    next(events)
    (root / "new.txt").write_text("Added mid-scan")
    with pytest.raises(ValueError, match="inventory changed"):
        list(events)
    with pytest.raises(ValueError, match="outside"):
        sync_source(IngestionCatalog(root / ".cache"), collection, connector)


def test_website_bridge_rejects_incomplete_runs_and_external_artifacts(tmp_path):
    connector = make_website(tmp_path / "site", "site", "site.example", "# Page\nUseful text")
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    sync_source(catalog, collection, connector)
    state_path = connector.run_dir / "pipeline_state.json"
    state = json.loads(state_path.read_text())
    state["status"] = "paused"
    state_path.write_text(json.dumps(state))
    with pytest.raises(SyncError):
        sync_source(catalog, collection, connector)
    state["status"] = "completed"
    state_path.write_text(json.dumps(state))
    artifact_path = connector.run_dir / "artifact_catalog.json"
    artifacts = json.loads(artifact_path.read_text())
    outside = tmp_path / "outside.md"
    outside.write_text("Should not be imported")
    artifacts["records"][0]["local_path"] = str(outside)
    artifact_path.write_text(json.dumps(artifacts))
    with pytest.raises(SyncError):
        sync_source(catalog, collection, connector)
    assert len(catalog.records(collection, access=AccessContext("w"))) == 1
