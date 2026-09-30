"""Publication races and source-identity rebinding cannot silently replace newer content."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lawa_rag.ingestion import (
    AccessContext, AccessPolicy, Collection, IngestionCatalog, Source, SyncError, sync_source,
)
from pipeline.tests.lawa_rag_ingestion_support import FixtureConnector, source_item


def test_stale_sync_cannot_publish_over_a_newer_successful_sync(tmp_path):
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    source = Source("api", "api", "https://source.example/", AccessPolicy(public=True))
    latest = FixtureConnector(source, (source_item("item", "Newer content"),))
    old = FixtureConnector(source, (source_item("item", "Stale content"),),
                           before_end=lambda: sync_source(catalog, collection, latest))
    with pytest.raises(SyncError) as error:
        sync_source(catalog, collection, old)
    access = AccessContext("w")
    record, = catalog.records(collection, access=access)
    assert catalog.read(collection, record.record_id, access=access)[1] == b"Newer content"
    with pytest.raises(ValueError, match="superseded"):
        sync_source(catalog, collection, old, resume_run_id=error.value.run_id)


def test_source_binding_and_resume_settings_cannot_be_changed_silently(tmp_path):
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    source = Source("api", "api", "https://source.example/", AccessPolicy(public=True))
    connector = FixtureConnector(source, (source_item("item", "Content"),), error=True)
    with pytest.raises(SyncError) as error:
        sync_source(catalog, collection, connector)
    changed = replace(connector, source=replace(source, processing_version="changed"), error=False)
    with pytest.raises(ValueError, match="same collection"):
        sync_source(catalog, collection, changed, resume_run_id=error.value.run_id)
    moved = replace(connector, source=replace(source, uri="https://other.example/"))
    with pytest.raises(ValueError, match="another location"):
        sync_source(catalog, collection, moved)
