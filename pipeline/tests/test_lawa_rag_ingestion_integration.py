"""Cross-source integration, incremental changes and SDK/CLI parity."""

from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from lawa_rag.ingestion import (
    AccessContext, AccessPolicy, Collection, IngestionCatalog, LocalFolderConnector, Source,
    SyncPolicy, build_evaluation_corpus, sync_source,
)
from lawa_rag.ingestion.cli import main
from pipeline.tests.lawa_rag_ingestion_support import make_website


def test_two_websites_and_a_folder_share_a_grounded_evaluation_collection(tmp_path):
    collection = Collection("workspace-a", "knowledge")
    catalog = IngestionCatalog(tmp_path / "catalog")
    first = make_website(tmp_path / "alpha", "alpha", "alpha.example", "# Study\nAlpha offers AI.")
    second = make_website(tmp_path / "beta", "beta", "beta.example", "# Study\nBeta offers robotics.")
    folder = tmp_path / "documents"
    folder.mkdir()
    arabic = "# القبول\nيمكن للطلاب تقديم طلباتهم عبر الموقع."
    (folder / "admissions.md").write_text(arabic)
    local = LocalFolderConnector(Source("documents", "folder", folder.as_uri(),
                                        AccessPolicy(public=True)))
    for connector in (first, second, local):
        assert sync_source(catalog, collection, connector).counts["added"] == 1
    access = AccessContext("workspace-a")
    records = catalog.records(collection, access=access)
    assert len(records) == len({item.record_id for item in records}) == 3
    snapshot = build_evaluation_corpus(catalog, collection, access=access)
    snapshot.require_valid()
    assert {item.source_type for item in snapshot.sources} == {"website", "folder"}
    assert {item.locator for item in snapshot.evidence} == {item.locator for item in records}
    assert " ".join(arabic.split()) in " ".join(
        "\n".join(item.text for item in snapshot.evidence).split()
    )
    repeat = build_evaluation_corpus(catalog, collection, access=access)
    assert repeat.snapshot_id == snapshot.snapshot_id
    (folder / "scan.pdf").write_bytes(b"%PDF-raw-not-converted")
    sync_source(catalog, collection, local)
    raw = next(item for item in catalog.records(collection, access=access) if item.kind == "document"
               and item.media_type == "application/pdf")
    assert raw.processing_state == "raw"
    with pytest.raises(ValueError, match="requires conversion"):
        build_evaluation_corpus(catalog, collection, access=access)


def test_incremental_lifecycle_reuses_content_and_separates_processing_changes(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "a.txt").write_text("First content")
    (root / "b.txt").write_text("Second content")
    source = Source("docs", "folder", root.as_uri(), AccessPolicy(public=True),
                    sync_policy=SyncPolicy(delete_missing=True, max_delete_fraction=0.5))
    connector = LocalFolderConnector(source)
    catalog = IngestionCatalog(tmp_path / "catalog")
    collection = Collection("w", "c")
    sync_source(catalog, collection, connector)
    before = sorted(path for path in catalog.blobs.root.rglob("*") if path.is_file())
    repeat = sync_source(catalog, collection, connector)
    assert repeat.counts["unchanged"] == 2 and repeat.counts["needs_processing"] == 0
    assert before == sorted(path for path in catalog.blobs.root.rglob("*") if path.is_file())
    (root / "a.txt").write_text("Updated first content")
    (root / "b.txt").unlink()
    (root / "c.txt").write_text("New third content")
    changed = sync_source(catalog, collection, connector)
    assert [changed.counts[name] for name in ("added", "updated", "deleted")] == [1, 1, 1]
    assert {item["operation"] for item in catalog.changes(changed.run_id)} == {
        "added", "updated", "deleted",
    }
    info = (root / "a.txt").stat()
    os.utime(root / "a.txt", ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    touched = sync_source(catalog, collection, connector)
    assert touched.counts["metadata_changed"] == 1 and touched.counts["needs_processing"] == 0
    new_pipeline = replace(connector, source=replace(source, processing_version="parser-v2"))
    assert sync_source(catalog, collection, new_pipeline).counts["reprocess"] == 2


def test_cli_uses_the_same_catalog_and_plan_has_no_side_effects(tmp_path, capsys):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "guide.txt").write_text("A useful guide")
    manifest = tmp_path / "collection.json"
    manifest.write_text(json.dumps({
        "schema_version": "lawa.ingestion.collection.v1",
        "collection": {"workspace_id": "w", "collection_id": "c"},
        "sources": [{"source_id": "docs", "kind": "folder", "uri": root.as_uri(),
                     "access": {"principals": ["team"]}}],
    }))
    options = ["--config", str(manifest), "--catalog", str(tmp_path / "catalog")]
    assert main(["plan", *options]) == 0
    assert json.loads(capsys.readouterr().out)["writes"] is False
    assert not (tmp_path / "catalog").exists()
    assert main(["sync", *options]) == 0
    assert json.loads(capsys.readouterr().out)["completed"][0]["counts"]["added"] == 1
    assert main(["inspect", *options]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 0
    assert main(["inspect", *options, "--principal", "team"]) == 0
    assert json.loads(capsys.readouterr().out)["count"] == 1
