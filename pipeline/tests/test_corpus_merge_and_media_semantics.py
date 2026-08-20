from __future__ import annotations

import asyncio
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image

from pipeline.core.artifacts import (
    ArtifactCatalog,
    build_artifact_record,
    save_artifact_catalog,
)
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import build_media_embedding_text, build_media_manifest, normalize_media_item
from pipeline.core.state import PipelineState, StageState, save_state
from pipeline.stages.formatters.corpus_merge_formatter import CorpusMergeFormatter
from pipeline.stages.formatters.media_semantics_formatter import (
    MediaSemanticsFormatter,
    _image_payload,
)


def _png(path: Path, color: str = "navy") -> str:
    image = Image.new("RGB", (240, 160), color)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_run(root: Path, *, run_id: str, project: str, url: str, color: str) -> Path:
    root.mkdir(parents=True)
    markdown = root / f"{run_id}.md"
    markdown.write_text(f"# {run_id}\n\nUseful MBZUAI content for {url}.", encoding="utf-8")
    image = root / f"{run_id}.png"
    content_hash = _png(image, color)
    media_item = normalize_media_item(
        {
            "type": "image",
            "id": f"image-{run_id}",
            "url": f"{url}/content.png",
            "source_url": url,
            "source_type": "html",
            "local_path": str(image),
            "mime_type": "image/png",
            "content_hash": content_hash,
            "alt": f"Image for {run_id}",
        }
    )

    paths = {
        "mapping_file": root / "mapping.json",
        "md_mapping_file": root / "md_mapping.json",
        "page_metadata_file": root / "page_metadata.json",
        "page_link_graph_file": root / "page_link_graph.json",
        "page_media_file": root / "page_media.json",
        "page_images_file": root / "page_images.json",
        "page_videos_file": root / "page_videos.json",
        "extracted_images_index_file": root / "document_media.json",
        "media_manifest_file": root / "media_manifest.json",
    }
    atomic_write_json(paths["mapping_file"], {url: str(markdown)})
    atomic_write_json(paths["md_mapping_file"], {url: str(markdown)})
    atomic_write_json(
        paths["page_metadata_file"],
        {
            url: {
                "url": url,
                "status_code": 200,
                "title": run_id,
                "path": "/",
                "language": "en",
            }
        },
    )
    node_id = f"page:{run_id}"
    atomic_write_json(
        paths["page_link_graph_file"],
        {
            "schema_version": 1,
            "graph_type": "website_page_link_graph",
            "nodes": [{"id": node_id, "url": url, "node_type": "crawled_page"}],
            "edges": [],
            "stats": {"node_count": 1, "edge_count": 0},
        },
    )
    atomic_write_json(paths["page_media_file"], {url: [media_item]})
    atomic_write_json(paths["page_images_file"], {url: [media_item]})
    atomic_write_json(paths["page_videos_file"], {})
    atomic_write_json(paths["extracted_images_index_file"], build_media_manifest([]))
    atomic_write_json(paths["media_manifest_file"], build_media_manifest([media_item]))

    catalog = ArtifactCatalog(
        records=[
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_html",
                uri=markdown.resolve().as_uri(),
                local_path=markdown,
                metadata={"source_url": url, "source_type": "html"},
            ),
            build_artifact_record(
                artifact_type="web_image",
                role="web_content_image",
                producer_stage="enrich_media",
                uri=image.resolve().as_uri(),
                local_path=image,
                metadata=media_item,
            ),
        ]
    )
    save_artifact_catalog(catalog, root)
    save_state(
        PipelineState(
            run_id=run_id,
            project_name=project,
            status="paused",
            stages=[
                StageState(
                    name="media_enrichment",
                    stage_type="formatter",
                    stage_id="enrich_media",
                    status="completed",
                    outputs={key: str(path) for key, path in paths.items()},
                ),
                StageState(
                    name="dedup_filter",
                    stage_type="quality_gate",
                    stage_id="deduplicate_markdown",
                    status="pending",
                ),
            ],
            current_stage_index=1,
        ),
        root,
    )
    atomic_write_json(root / "run_audit.json", {"ok": True, "errors": [], "warnings": []})
    atomic_write_json(root / "resolved_config.json", {"run_id": run_id, "config": {}})
    return root


def test_corpus_merge_materializes_immutable_sources(tmp_path: Path):
    first = _write_source_run(
        tmp_path / "source-a",
        run_id="source-a",
        project="main",
        url="https://mbzuai.ac.ae/about",
        color="navy",
    )
    second = _write_source_run(
        tmp_path / "source-b",
        run_id="source-b",
        project="subdomains",
        url="https://careers.mbzuai.ac.ae/jobs",
        color="orange",
    )
    source_markdown_bytes = (second / "source-b.md").read_bytes()
    work_dir = tmp_path / "combined"
    old_markdown = tmp_path / "old.md"
    old_markdown.write_text("old", encoding="utf-8")
    existing = build_artifact_record(
        artifact_type="markdown",
        role="content",
        producer_stage="convert_html",
        uri=old_markdown.resolve().as_uri(),
        local_path=old_markdown,
    )
    context = StageContext(
        run_id="combined",
        project_name="combined",
        config={
            "formatter": {
                "corpus_merge": {
                    "use_current_artifacts": False,
                    "source_run_dirs": [str(first), str(second)],
                    "allowed_source_projects": ["main", "subdomains"],
                    "required_source_stage_ids": ["enrich_media"],
                    "require_source_audit_ok": True,
                    "minimum_source_count": 2,
                }
            }
        },
        work_dir=work_dir,
        previous_outputs={},
        stage_definition={"id": "merge_corpora", "type": "formatter", "plugin": "corpus_merge"},
        stage_id="merge_corpora",
        artifact_catalog=ArtifactCatalog(records=[existing]),
    )

    result = asyncio.run(CorpusMergeFormatter().execute(context))

    assert result.status == StageStatus.COMPLETED
    assert result.metrics["source_runs"] == 2
    assert result.metrics["markdown_artifacts"] == 2
    assert result.metrics["unique_visual_content_hashes"] == 2
    assert existing.artifact_id in result.removed_artifact_ids
    assert len(load_json_safe(result.outputs["md_mapping_file"])) == 2
    assert len(load_json_safe(result.outputs["page_metadata_file"])) == 2
    graph = load_json_safe(result.outputs["page_link_graph_file"])
    assert graph["schema_version"] == 3
    assert graph["stats"]["node_count"] == 2
    assert (second / "source-b.md").read_bytes() == source_markdown_bytes
    assert all(
        Path(record.local_path).is_file()
        for record in result.artifacts
        if getattr(record, "artifact_type", "") == "markdown"
    )


def test_media_semantics_plan_deduplicates_by_content_hash_and_propagates(tmp_path: Path):
    image_path = tmp_path / "visual.png"
    content_hash = _png(image_path)
    items = [
        {
            "type": "image",
            "url": "https://mbzuai.ac.ae/a.png",
            "source_url": "https://mbzuai.ac.ae/a",
            "source_type": "html",
            "local_path": str(image_path),
            "mime_type": "image/png",
            "content_hash": content_hash,
            "alt": "Research event",
        },
        {
            "type": "image",
            "url": "https://careers.mbzuai.ac.ae/same.png",
            "source_url": "https://careers.mbzuai.ac.ae/b",
            "source_type": "html",
            "local_path": str(image_path),
            "mime_type": "image/png",
            "content_hash": content_hash,
            "context": "Used on the careers page",
        },
    ]
    manifest = tmp_path / "media.json"
    page_media = tmp_path / "page_media.json"
    page_images = tmp_path / "page_images.json"
    document_media = tmp_path / "document_media.json"
    page_metadata = tmp_path / "page_metadata.json"
    atomic_write_json(manifest, build_media_manifest(items))
    atomic_write_json(
        page_media,
        {
            "https://mbzuai.ac.ae/a": [items[0]],
            "https://careers.mbzuai.ac.ae/b": [items[1]],
        },
    )
    atomic_write_json(
        page_images,
        {
            "https://mbzuai.ac.ae/a": [items[0]],
            "https://careers.mbzuai.ac.ae/b": [items[1]],
        },
    )
    atomic_write_json(document_media, build_media_manifest([]))
    atomic_write_json(
        page_metadata,
        {
            "https://mbzuai.ac.ae/a": {"title": "Research"},
            "https://careers.mbzuai.ac.ae/b": {"title": "Careers"},
        },
    )
    source_artifact = build_artifact_record(
        artifact_type="web_image",
        role="web_content_image",
        producer_stage="merge_corpora",
        uri=image_path.resolve().as_uri(),
        local_path=image_path,
        metadata=items[0],
    )
    context = StageContext(
        run_id="plan",
        project_name="test",
        config={
            "formatter": {
                "media_semantics": {
                    "mode": "plan",
                    "model": "gemini-3.5-flash-lite",
                    "concurrency": 2,
                    "retry_attempts": 2,
                    "image_max_side": 1600,
                    "image_max_pixels": 1_800_000,
                    "require_complete": False,
                }
            }
        },
        work_dir=tmp_path / "run",
        previous_outputs={
            "media_manifest_file": str(manifest),
            "page_media_file": str(page_media),
            "page_images_file": str(page_images),
            "extracted_images_index_file": str(document_media),
            "page_metadata_file": str(page_metadata),
        },
        stage_definition={"id": "annotate_media", "type": "formatter", "plugin": "media_semantics"},
        stage_id="annotate_media",
        artifact_catalog=ArtifactCatalog(records=[source_artifact]),
    )

    result = asyncio.run(MediaSemanticsFormatter().execute(context))

    assert result.status == StageStatus.COMPLETED
    assert result.metrics == {
        "unique_visuals": 1,
        "completed": 0,
        "failed": 0,
        "pending": 1,
        "completion_ratio": 0.0,
        "ocr_required": 0,
    }
    queue = load_json_safe(result.outputs["media_annotation_queue_file"])
    assert len(queue["items"]) == 1
    assert queue["items"][0]["reference_count"] >= 2
    assert set(queue["items"][0]["source_urls"]) == {
        "https://mbzuai.ac.ae/a",
        "https://careers.mbzuai.ac.ae/b",
    }
    annotated = load_json_safe(result.outputs["page_media_file"])
    assert annotated["https://mbzuai.ac.ae/a"][0]["annotation_status"] == "pending"
    assert annotated["https://careers.mbzuai.ac.ae/b"][0]["annotation_status"] == "pending"


def test_semantic_fields_survive_normalization_and_feed_embedding_text():
    item = normalize_media_item(
        {
            "type": "image",
            "url": "https://example.test/diagram.png",
            "semantic_caption": "A pipeline diagram with three connected stages.",
            "contextual_caption": "Architecture of the MBZUAI indexing pipeline.",
            "visual_description": "Three labeled boxes are connected left to right.",
            "visible_text": "crawl clean index",
            "image_kind": "diagram",
            "semantic_tags": ["pipeline", "indexing"],
            "semantic_relevance": "substantive",
            "annotation_confidence": 0.94,
            "needs_ocr": False,
        }
    )

    text = build_media_embedding_text([item])

    assert item["annotation_confidence"] == 0.94
    assert item["semantic_tags"] == ["pipeline", "indexing"]
    assert "visual_caption=A pipeline diagram" in text
    assert "tags=pipeline, indexing" in text


def test_unsupported_image_format_is_normalized_for_gemini(tmp_path: Path):
    bmp = tmp_path / "visual.bmp"
    Image.new("RGB", (300, 200), "teal").save(bmp, format="BMP")

    payload, mime_type, normalized = _image_payload(
        bmp,
        maximum_side=1600,
        maximum_pixels=1_800_000,
    )

    assert payload
    assert mime_type == "image/jpeg"
    assert normalized is True
    with Image.open(BytesIO(payload)) as image:
        assert image.format == "JPEG"
