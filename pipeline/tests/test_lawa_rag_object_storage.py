"""Snapshot safety at the SDK boundary, without using credentials or a live bucket."""

from dataclasses import replace

import pytest

from lawa_rag.ingestion import AccessContext, Collection, IngestionCatalog, SyncError, sync_source
from pipeline.tests.lawa_rag_object_storage_support import (
    listing, object_body, objects, snapshot, stubbed_connector,
)


def test_paginated_s3_snapshot_conditional_reads_and_unchanged_refresh(tmp_path):
    connector, stubber = stubbed_connector()
    catalog, collection, access = IngestionCatalog(tmp_path), Collection("w", "c"), AccessContext("w")
    values = {"a.md": b"# Admission\nApply online", "b.csv": "program,years\nالذكاء,2\n".encode()}
    items = objects(**values)
    with stubber:
        listing(stubber, items[:1], next_token="page-2", truncated=True)
        listing(stubber, items[1:], token="page-2")
        streams = [object_body(stubber, item, values[item["Key"][5:]]) for item in items]
        listing(stubber, items)
        report = sync_source(catalog, collection, connector)
        assert report.counts["added"] == 2 and all(stream.closed for stream in streams)
        snapshot(stubber, values)
        assert sync_source(catalog, collection, connector).counts["unchanged"] == 2
        stubber.assert_no_pending_responses()
    records = catalog.records(collection, access=access)
    assert {record.processing_state for record in records} == {"raw", "ready"}
    assert {record.locator for record in records} == {
        "s3://test-bucket/docs/a.md", "s3://test-bucket/docs/b.csv",
    }


def test_incomplete_or_changing_snapshots_never_replace_published_records(tmp_path):
    connector, stubber = stubbed_connector()
    catalog, collection, access = IngestionCatalog(tmp_path), Collection("w", "c"), AccessContext("w")
    values = {"a.md": b"Existing authoritative text"}
    with stubber:
        snapshot(stubber, values)
        sync_source(catalog, collection, connector)
        baseline = catalog.records(collection, access=access)
        # A provider says truncated but omits its cursor. The paginator alone is insufficient.
        listing(stubber, [], truncated=True)
        with pytest.raises(SyncError):
            sync_source(catalog, collection, connector)
        assert catalog.records(collection, access=access) == baseline
        item = objects(**values)[0]
        listing(stubber, [item])
        stream = object_body(stubber, item, values["a.md"])
        listing(stubber, [])  # Object disappeared during the scan.
        with pytest.raises(SyncError):
            sync_source(catalog, collection, connector)
        assert stream.closed and catalog.records(collection, access=access) == baseline
        listing(stubber, [item])
        stubber.add_client_error("get_object", service_error_code="PreconditionFailed",
                                 http_status_code=412, expected_params={
                                     "Bucket": "test-bucket", "Key": item["Key"],
                                     "IfMatch": item["ETag"],
                                 })
        with pytest.raises(SyncError):
            sync_source(catalog, collection, connector)
        assert catalog.records(collection, access=access) == baseline
        stubber.assert_no_pending_responses()


def test_source_binding_and_listing_budgets_prevent_accidental_scope_changes(tmp_path):
    connector, stubber = stubbed_connector(max_snapshot_bytes=10)
    catalog, collection = IngestionCatalog(tmp_path), Collection("w", "c")
    with stubber:
        snapshot(stubber, {"a.txt": b"Allowed"})
        sync_source(catalog, collection, connector)
        with pytest.raises(ValueError, match="endpoint changed"):
            sync_source(catalog, collection, replace(connector, endpoint_url="https://new.example"))
        listing(stubber, objects(**{"large.pdf": b"x" * 11}))
        with pytest.raises(SyncError):
            sync_source(catalog, collection, connector)
        listing(stubber, [], next_token="more-excluded-objects", truncated=True)
        with pytest.raises(SyncError):
            sync_source(catalog, collection, replace(connector, max_listing_pages=1))
        stubber.assert_no_pending_responses()
