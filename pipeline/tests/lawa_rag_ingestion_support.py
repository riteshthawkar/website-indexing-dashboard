"""Small source fixtures; real catalogs/connectors are used in ingestion tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lawa_rag import WebsiteProfile, build_website_config
from lawa_rag.ingestion import AccessPolicy, SnapshotEnd, Source, SourceItem, WebsiteRunConnector
from pipeline.core.artifacts import ArtifactCatalog, ArtifactRecord, save_artifact_catalog


def source_item(native_id: str, text: str, **options) -> SourceItem:
    return SourceItem(
        native_id=native_id, body=text.encode(), media_type="text/plain",
        title=native_id, locator=f"https://source.example/{native_id}",
        processing_state="ready", **options,
    )


@dataclass
class FixtureConnector:
    source: Source
    items: tuple[SourceItem, ...]
    error: bool = False
    complete: bool = True
    before_end: object = None

    @property
    def fingerprint(self) -> str:
        return "fixture-v1"

    def scan(self):
        yield from self.items
        if self.error:
            raise OSError("Synthetic interrupted connection; secret-must-not-be-stored")
        if self.before_end:
            self.before_end()
        if self.complete:
            yield SnapshotEnd()


def make_website(root: Path, source_id: str, host: str, text: str) -> WebsiteRunConnector:
    root.mkdir(parents=True)
    source = Source(source_id, "website", f"https://{host}/", AccessPolicy(public=True))
    profile = WebsiteProfile(site_id=source_id, start_urls=(source.uri,))
    config = build_website_config(profile)
    (root / "resolved_config.json").write_text(json.dumps({"config": config}))
    state = {"run_id": source_id, "project_name": source_id, "status": "completed", "stages": [
        {"name": "crawl4ai", "stage_id": "crawl_web", "stage_type": "crawler",
         "status": "completed", "metrics": {"failed": 0}},
        {"name": "markitdown", "stage_id": "convert_html", "stage_type": "converter",
         "status": "completed", "metrics": {"failed": 0}},
    ]}
    (root / "pipeline_state.json").write_text(json.dumps(state))
    markdown = root / "page.md"
    markdown.write_text(text, encoding="utf-8")
    artifact = ArtifactRecord(
        artifact_id="markdown-1", artifact_type="markdown", role="content",
        producer_stage="convert_html", uri=markdown.as_uri(), local_path=str(markdown),
        metadata={"source_url": source.uri + "study/", "document_title": "Study",
                  "language": "en", "backend": "markitdown", "source_type": "webpage"},
    )
    save_artifact_catalog(ArtifactCatalog(records=[artifact]), root)
    return WebsiteRunConnector(source, root)
