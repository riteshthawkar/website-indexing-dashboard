from __future__ import annotations

import asyncio
import hashlib
import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from pipeline.core.artifacts import (
    ArtifactCatalog,
    build_artifact_record,
    save_artifact_catalog,
)
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import (
    build_media_embedding_text,
    build_media_manifest,
    media_chunk_match,
    normalize_media_item,
)
from pipeline.core.media_context import build_media_reference_contexts
from pipeline.core.state import PipelineState, StageState, save_state
from pipeline.stages.formatters.corpus_merge_formatter import CorpusMergeFormatter
from pipeline.stages.formatters.media_semantics_formatter import (
    MediaSemanticsFormatter,
    _annotation_fields,
    _effective_prompt_revision,
    _image_payload,
    _output_token_budget,
    _retryable,
    _response_contract_retryable,
    _validate_annotation,
)


def _png(path: Path, color: str = "navy") -> str:
    image = Image.new("RGB", (240, 160), color)
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_media_semantics_retries_malformed_structured_responses_and_scales_budget():
    malformed = json.JSONDecodeError("Unterminated string", "{", 1)
    assert _response_contract_retryable(malformed)
    assert _response_contract_retryable(
        ValueError("Missing contextual captions for reference IDs: media-ref:1")
    )
    assert not _response_contract_retryable(ValueError("Unsupported local image file"))
    assert _retryable(asyncio.TimeoutError())
    assert _retryable(RuntimeError("Server disconnected without sending a response"))

    one_reference = {"reference_contexts": [{"reference_id": "media-ref:1"}]}
    twelve_references = {
        "reference_contexts": [
            {"reference_id": f"media-ref:{index}"} for index in range(12)
        ]
    }
    assert _output_token_budget(one_reference, {"max_output_tokens": 1200}) == 2400
    assert _output_token_budget(twelve_references, {"max_output_tokens": 1200}) == 3720
    assert _output_token_budget(one_reference, {"max_output_tokens": 4096}) == 4096


def test_media_chunk_match_is_fail_closed_for_pdf_pages():
    item = {
        "type": "image",
        "page_number": 13,
        "section_path": [],
        "context": "A distinctive architecture diagram.",
    }

    exact = media_chunk_match(
        item,
        chunk_page_numbers=[13, 14],
        chunk_text="A distinctive architecture diagram.",
    )
    mismatched = media_chunk_match(
        item,
        chunk_page_numbers=[103, 104],
        chunk_text="A distinctive architecture diagram.",
    )

    assert exact == {"matched": True, "score": 100.0, "method": "exact_page"}
    assert mismatched == {
        "matched": False,
        "score": 0.0,
        "method": "page_mismatch",
    }


def test_media_chunk_match_uses_web_section_or_occurrence_context():
    section_match = media_chunk_match(
        {
            "type": "image",
            "section_path": ["Program Overview", "Program Guidelines"],
        },
        chunk_section_path=["Program Overview", "Program Guidelines", "Confidentiality"],
        chunk_text="Both parties commit to a respectful relationship.",
    )
    context_match = media_chunk_match(
        {
            "type": "image",
            "surrounding_text_after": (
                "Academics can thrive in business by allowing data to empower decisions."
            ),
        },
        chunk_section_path=["Career advice"],
        chunk_text=(
            "A recent guest told students that academics can thrive in business by "
            "allowing data to empower decisions."
        ),
    )
    unscoped = media_chunk_match(
        {"type": "image", "semantic_caption": "A generic campus photograph."},
        chunk_section_path=["Admissions"],
        chunk_text="Applications require academic transcripts.",
    )

    assert section_match["matched"] is True
    assert section_match["method"] == "section_ancestor"
    assert context_match["matched"] is True
    assert context_match["method"] == "surrounding_text_after_substring"
    assert unscoped == {"matched": False, "score": 0.0, "method": "unscoped"}


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
    md_mapping = tmp_path / "md_mapping.json"
    first_markdown = tmp_path / "research.md"
    second_markdown = tmp_path / "careers.md"
    first_markdown.write_text(
        "\n".join(
            [
                "# Research",
                "",
                "## Robotics laboratory",
                "",
                "Researchers build embodied AI systems.",
                "",
                "![Research event](https://mbzuai.ac.ae/a.png)",
                "",
                "Students test robots with faculty supervision.",
                "",
                "## Admissions",
                "",
                "This unrelated section must not become image context.",
            ]
        ),
        encoding="utf-8",
    )
    second_markdown.write_text(
        "\n".join(
            [
                "# Careers",
                "",
                "## Faculty opportunities",
                "",
                "MBZUAI recruits researchers across AI disciplines.",
                "",
                "![Research event](https://careers.mbzuai.ac.ae/same.png)",
                "",
                "Open roles support teaching and research.",
            ]
        ),
        encoding="utf-8",
    )
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
        md_mapping,
        {
            "https://mbzuai.ac.ae/a": str(first_markdown),
            "https://careers.mbzuai.ac.ae/b": str(second_markdown),
        },
    )
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
            "md_mapping_file": str(md_mapping),
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
    reference_contexts = queue["items"][0]["reference_contexts"]
    assert len(reference_contexts) == 2
    assert {tuple(value["section_path"]) for value in reference_contexts} == {
        ("Research", "Robotics laboratory"),
        ("Careers", "Faculty opportunities"),
    }
    research_context = next(
        value for value in reference_contexts if value["source_url"] == "https://mbzuai.ac.ae/a"
    )
    assert "Researchers build embodied AI systems" in research_context["surrounding_text_before"]
    assert "Students test robots" in research_context["surrounding_text_after"]
    assert "unrelated section" not in json.dumps(research_context).lower()
    assert queue["items"][0]["annotation_input_hash"]
    context_artifact = load_json_safe(result.outputs["media_reference_contexts_file"])
    assert context_artifact["stats"]["reference_count"] == 2
    assert context_artifact["stats"]["with_surrounding_text"] == 2
    annotated = load_json_safe(result.outputs["page_media_file"])
    assert annotated["https://mbzuai.ac.ae/a"][0]["annotation_status"] == "pending"
    assert annotated["https://careers.mbzuai.ac.ae/b"][0]["annotation_status"] == "pending"
    assert annotated["https://mbzuai.ac.ae/a"][0]["section_heading"] == "Robotics laboratory"
    assert annotated["https://careers.mbzuai.ac.ae/b"][0]["section_heading"] == "Faculty opportunities"


def test_reference_context_uses_following_card_heading_and_html_fallback(tmp_path: Path):
    image_path = tmp_path / "person.png"
    content_hash = _png(image_path)
    markdown = tmp_path / "leadership.md"
    markdown.write_text(
        "\n".join(
            [
                "# Leadership",
                "",
                "## Board of Trustees",
                "",
                "![event card image](https://example.test/person.png)",
                "",
                "### Lisa Example",
                "",
                "Chief executive and board member.",
            ]
        ),
        encoding="utf-8",
    )
    html_path = tmp_path / "research.html"
    html_path.write_text(
        """
        <main>
          <h1>Research</h1>
          <section>
            <h2>Computer Vision Lab</h2>
            <p>The lab develops robust visual perception systems.</p>
            <img src="https://example.test/lab.png" alt="Laboratory">
            <p>Projects include scene understanding and robotics.</p>
          </section>
          <h2>Admissions</h2><p>Unrelated admissions text.</p>
        </main>
        """,
        encoding="utf-8",
    )
    items = [
        {
            "type": "image",
            "id": "leadership-card",
            "url": "https://example.test/person.png",
            "source_url": "https://example.test/leadership",
            "source_type": "html",
            "local_path": str(image_path),
            "content_hash": content_hash,
            "alt": "event card image",
        },
        {
            "type": "image",
            "id": "lab-photo",
            "url": "https://example.test/lab.png",
            "source_url": "https://example.test/research",
            "source_type": "html",
            "local_path": str(image_path),
            "content_hash": content_hash,
            "alt": "Laboratory",
        },
    ]

    records, _lookup, stats = build_media_reference_contexts(
        items,
        markdown_mapping={"https://example.test/leadership": str(markdown)},
        html_mapping={"https://example.test/research": str(html_path)},
        page_metadata={
            "https://example.test/leadership": {"title": "Leadership"},
            "https://example.test/research": {"title": "Research"},
        },
        config={"context_before_blocks": 1, "context_after_blocks": 2},
    )

    by_id = {record["media_id"]: record for record in records}
    assert by_id["leadership-card"]["section_path"] == [
        "Leadership",
        "Board of Trustees",
        "Lisa Example",
    ]
    assert by_id["leadership-card"]["context_association"] == "following_heading"
    assert "Chief executive" in by_id["leadership-card"]["surrounding_text_after"]
    assert by_id["lab-photo"]["context_source"] == "html_dom"
    assert by_id["lab-photo"]["section_heading"] == "Computer Vision Lab"
    assert "visual perception" in by_id["lab-photo"]["surrounding_text_before"]
    assert "scene understanding" in by_id["lab-photo"]["surrounding_text_after"]
    assert "admissions" not in json.dumps(by_id["lab-photo"]).lower()
    assert stats["with_surrounding_text"] == 2


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
            "context_reference_id": "media-ref:diagram",
            "context_source": "markdown",
            "section_path": ["Research", "Architecture"],
            "section_heading": "Architecture",
            "surrounding_text_before": "The retrieval system has three stages.",
            "surrounding_text_after": "Each stage emits an immutable artifact.",
        }
    )

    text = build_media_embedding_text([item])

    assert item["annotation_confidence"] == 0.94
    assert item["semantic_tags"] == ["pipeline", "indexing"]
    assert item["section_path"] == ["Research", "Architecture"]
    assert "visual_caption=A pipeline diagram" in text
    assert "tags=pipeline, indexing" in text
    assert "section=Research > Architecture" in text
    assert "surrounding_after=Each stage emits" in text


def test_contextual_captions_are_validated_and_selected_per_reference():
    payload = {
        "semantic_caption": "A person standing at a lectern.",
        "contextual_caption": "A university event image.",
        "contextual_captions": [
            {
                "reference_id": "media-ref:a",
                "contextual_caption": "A speaker at the robotics symposium.",
                "confidence": 0.91,
            },
            {
                "reference_id": "media-ref:b",
                "contextual_caption": "A welcome address on the admissions page.",
                "confidence": 0.84,
            },
        ],
        "visual_description": "A person faces an audience behind a lectern.",
        "visible_text": "",
        "image_kind": "photo",
        "semantic_tags": ["speaker", "lectern"],
        "semantic_relevance": "contextual",
        "contains_text": False,
        "needs_ocr": False,
        "needs_review": False,
        "confidence": 0.9,
        "uncertain_details": [],
    }

    annotation = _validate_annotation(
        payload,
        expected_reference_ids={"media-ref:a", "media-ref:b"},
    )
    annotation["annotation_status"] = "completed"
    first = _annotation_fields(annotation, reference_id="media-ref:a")
    second = _annotation_fields(annotation, reference_id="media-ref:b")
    unselected = _annotation_fields(annotation, reference_id="media-ref:c")

    assert first["contextual_caption"] == "A speaker at the robotics symposium."
    assert second["contextual_caption"] == "A welcome address on the admissions page."
    assert first["contextual_caption_scope"] == "reference"
    assert unselected["contextual_caption"] == ""
    with pytest.raises(ValueError, match="Unknown contextual caption"):
        _validate_annotation(payload, expected_reference_ids={"media-ref:a", "media-ref:missing"})
    missing_payload = {**payload, "contextual_captions": payload["contextual_captions"][:1]}
    with pytest.raises(ValueError, match="Missing contextual captions"):
        _validate_annotation(
            missing_payload,
            expected_reference_ids={"media-ref:a", "media-ref:b"},
        )


def test_legacy_prompt_revision_is_migrated_to_section_context_contract():
    assert _effective_prompt_revision(
        {"prompt_revision": "mbzuai-media-semantics-v1"}
    ) == "mbzuai-media-semantics-v2-section-context"


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
