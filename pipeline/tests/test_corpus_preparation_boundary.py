from __future__ import annotations

import asyncio
import hashlib
import importlib.util
from pathlib import Path

import pytest

from pipeline.core.artifacts import (
    ArtifactCatalog,
    build_artifact_record,
    save_artifact_catalog,
)
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import build_media_manifest, normalize_media_item
from pipeline.core.state import PipelineState, StageState, save_state
from pipeline.stages.formatters.corpus_merge_formatter import (
    _apply_source_path_replacements,
    _load_source_descriptor,
)
from pipeline.stages.formatters.corpus_preparation_formatter import (
    CorpusPreparationFormatter,
    _expand_canonical_url_aliases,
    _expand_html_artifact_aliases,
    _is_media_only_pdf_eligible,
)
from pipeline.stages.quality.dedup_filter import DedupFilter


HAS_DATASKETCH = importlib.util.find_spec("datasketch") is not None


def _run(coro):
    return asyncio.run(coro)


def test_source_path_replacement_builds_portable_in_memory_view():
    old_root = "/legacy/workspace"
    new_root = "/portable/workspace"
    catalog = ArtifactCatalog.from_dict(
        {
            "records": [
                {
                    "artifact_id": "markdown-1",
                    "artifact_type": "markdown",
                    "role": "content",
                    "producer_stage": "prepare",
                    "uri": f"file://{old_root}/document.md",
                    "local_path": f"{old_root}/document.md",
                    "metadata": {"markdown_path": f"{old_root}/document.md"},
                }
            ]
        }
    )
    descriptor = {
        "outputs": {"md_mapping_file": f"{old_root}/mapping.json"},
        "catalog": catalog,
        "evidence": {"artifact_catalog_sha256": "immutable-digest"},
    }

    _apply_source_path_replacements(descriptor, {old_root: new_root})

    record = catalog.records[0]
    assert descriptor["outputs"]["md_mapping_file"] == f"{new_root}/mapping.json"
    assert record.local_path == f"{new_root}/document.md"
    assert record.uri == f"file://{new_root}/document.md"
    assert record.metadata["markdown_path"] == f"{new_root}/document.md"
    assert descriptor["evidence"]["artifact_catalog_sha256"] == "immutable-digest"


def test_corpus_merge_accepts_only_explicit_completed_prefix_of_failed_run(
    tmp_path: Path,
):
    run_dir = tmp_path / "source"
    run_dir.mkdir()
    save_state(
        PipelineState(
            run_id="failed-source",
            project_name="source-project",
            status="failed",
            stages=[
                StageState(
                    name="annotate",
                    stage_type="formatter",
                    stage_id="annotate_media",
                    status="completed",
                ),
                StageState(
                    name="optional",
                    stage_type="formatter",
                    stage_id="optional_stage",
                    status="completed",
                    outputs={"optional_output": "must-not-be-imported"},
                ),
                StageState(
                    name="ocr",
                    stage_type="formatter",
                    stage_id="ocr_media",
                    status="failed",
                    error_message="provider unavailable",
                ),
            ],
            current_stage_index=2,
        ),
        run_dir,
    )
    save_artifact_catalog(ArtifactCatalog(), run_dir)
    atomic_write_json(run_dir / "run_audit.json", {"ok": True})
    atomic_write_json(run_dir / "resolved_config.json", {"config": {}})

    descriptor = _load_source_descriptor(
        run_dir,
        required_stage_ids=["annotate_media"],
        require_audit_ok=True,
        allowed_projects={"source-project"},
        allow_failed_after_required_stages=True,
    )

    assert descriptor["source_status"] == "failed"
    assert descriptor["failed_source_explicitly_allowed"] is True
    assert descriptor["failed_stages"][0]["stage_id"] == "ocr_media"
    assert descriptor["completed_prefix_cutoff"] == 0
    assert descriptor["allowed_artifact_producer_stages"] == [
        "annotate",
        "annotate_media",
    ]
    assert "optional_output" not in descriptor["outputs"]
    assert set(descriptor["outputs"]["stage_outputs"]) == {"annotate_media"}
    with pytest.raises(ValueError, match="must be paused or completed"):
        _load_source_descriptor(
            run_dir,
            required_stage_ids=["annotate_media"],
            require_audit_ok=True,
            allowed_projects={"source-project"},
        )
    with pytest.raises(ValueError, match="missing completed stages"):
        _load_source_descriptor(
            run_dir,
            required_stage_ids=["ocr_media"],
            require_audit_ok=True,
            allowed_projects={"source-project"},
            allow_failed_after_required_stages=True,
        )


@pytest.mark.skipif(
    not HAS_DATASKETCH,
    reason="datasketch not installed",
)
def test_dedup_preserves_media_occurrences_and_rebinds_shared_image(
    tmp_path: Path,
):
    md_dir = tmp_path / "markdown"
    md_dir.mkdir()
    winner = md_dir / "a.md"
    loser = md_dir / "z.md"
    content = "Representation-neutral duplicate content. " * 40
    winner.write_text(content, encoding="utf-8")
    loser.write_text(content, encoding="utf-8")
    image = tmp_path / "media" / "figure.png"
    image.parent.mkdir()
    image.write_bytes(b"verified-image")
    content_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    media_item = normalize_media_item(
        {
            "type": "image",
            "url": image.resolve().as_uri(),
            "local_path": str(image),
            "content_hash": content_hash,
            "source_type": "pdf",
            "source_url": "https://example.com/z.pdf",
            "md_path": str(loser),
        }
    )
    mapping_path = tmp_path / "mapping.json"
    page_media_path = tmp_path / "page_media.json"
    document_media_path = tmp_path / "document_media.json"
    media_manifest_path = tmp_path / "media_manifest.json"
    atomic_write_json(
        mapping_path,
        {
            "https://example.com/a": str(winner),
            "https://example.com/z": str(loser),
        },
    )
    atomic_write_json(
        page_media_path,
        {
            "https://example.com/a": [],
            "https://example.com/z": [media_item],
        },
    )
    atomic_write_json(document_media_path, build_media_manifest([media_item]))
    atomic_write_json(media_manifest_path, build_media_manifest([media_item]))
    catalog = ArtifactCatalog(
        records=[
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="merge",
                uri=winner.resolve().as_uri(),
                local_path=winner,
                artifact_id="md-winner",
            ),
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="merge",
                uri=loser.resolve().as_uri(),
                local_path=loser,
                artifact_id="md-loser",
            ),
            build_artifact_record(
                artifact_type="extracted_image",
                role="document_media",
                producer_stage="merge",
                uri=image.resolve().as_uri(),
                local_path=image,
                metadata={
                    "content_hash": content_hash,
                    "source_document_path": str(loser),
                },
                artifact_id="image-loser",
            ),
        ]
    )
    context = StageContext(
        run_id="dedup",
        project_name="dedup",
        config={
            "quality": {
                "dedup_threshold": 0.85,
                "dedup_num_perm": 128,
                "dedup_ngram_size": 5,
                "preserve_duplicate_source_media": True,
            }
        },
        work_dir=tmp_path,
        previous_outputs={
            "md_dir": str(md_dir),
            "md_mapping_file": str(mapping_path),
            "page_media_file": str(page_media_path),
            "extracted_images_index_file": str(document_media_path),
            "media_manifest_file": str(media_manifest_path),
        },
        stage_definition={"type": "quality_gate", "plugin": "dedup_filter"},
        stage_id="deduplicate_markdown",
        artifact_catalog=catalog,
    )

    result = _run(DedupFilter().execute(context))

    assert result.status == StageStatus.COMPLETED
    assert result.outputs["filtered_count"] == 1
    assert winner.exists() and not loser.exists() and image.exists()
    assert "image-loser" in result.removed_artifact_ids
    rebound = [
        artifact
        for artifact in result.artifacts
        if getattr(artifact, "artifact_type", "") == "extracted_image"
    ]
    assert len(rebound) == 1
    assert rebound[0].metadata["source_document_path"] == str(winner.resolve())
    assert set(load_json_safe(page_media_path)) == {
        "https://example.com/a",
        "https://example.com/z",
    }
    document_item = load_json_safe(document_media_path)["items"][0]
    assert document_item["md_path"] == str(winner.resolve())
    aliases = load_json_safe(result.outputs["duplicate_source_aliases_file"])
    assert aliases["alias_count"] == 1
    assert aliases["preserves_source_occurrence_evidence"] is True


def test_corpus_preparation_seals_inventory_without_representation(
    tmp_path: Path,
):
    legacy_url = "https://example.com/legacy-document"
    canonical_url = "https://example.com/document"
    markdown = tmp_path / "document.md"
    markdown.write_text("# MBZUAI\n\nA grounded document with a table.\n", encoding="utf-8")
    source_pdf = tmp_path / "downloaded-document.pdf"
    source_pdf.write_bytes(b"source-pdf")
    document_markdown = tmp_path / "converted-document.md"
    document_markdown.write_text(
        "# Converted document\n\nA URL-less PDF remains a first-class document.\n",
        encoding="utf-8",
    )
    image = tmp_path / "image.png"
    image.write_bytes(b"verified-image")
    content_hash = hashlib.sha256(image.read_bytes()).hexdigest()
    media_item = normalize_media_item(
        {
            "type": "image",
            "url": image.resolve().as_uri(),
            "local_path": str(image),
            "content_hash": content_hash,
            "source_type": "html",
            "source_url": canonical_url,
            "annotation_status": "completed",
            "needs_ocr": True,
            "ocr_status": "completed",
            "ocr_text": "MBZUAI",
        }
    )
    media_only_image = tmp_path / "media-only.png"
    media_only_image.write_bytes(b"verified-media-only-image")
    media_only_hash = hashlib.sha256(media_only_image.read_bytes()).hexdigest()
    media_only_item = normalize_media_item(
        {
            "type": "image",
            "url": "https://example.com/document.pdf#page=1",
            "local_path": str(media_only_image),
            "content_hash": media_only_hash,
            "source_type": "pdf",
            "source_url": "https://example.com/document.pdf",
            "source_file": str(source_pdf),
            "document_id": "document-pdf",
            "page_number": 1,
            "annotation_status": "completed",
            "semantic_caption": "A verified chart from the PDF.",
        }
    )
    payloads = {
        "md_mapping_file": {legacy_url: str(markdown)},
        "canonical_page_metadata_file": {
            canonical_url: {
                "source_url": canonical_url,
                "redirected_from": [legacy_url],
                "title": "Document",
            }
        },
        "url_identity_map_file": {"records": []},
        "canonical_page_link_graph_file": {"nodes": [], "edges": []},
        # The crawler may preserve a root trailing slash while the URL mapping
        # canonicalizes it away; preparation must still bind the media.
        "page_media_file": {f"{canonical_url}/": [media_item]},
        "page_images_file": {canonical_url: [media_item]},
        "page_videos_file": {},
        "extracted_images_index_file": build_media_manifest([media_only_item]),
        "media_manifest_file": build_media_manifest([media_item, media_only_item]),
        "duplicate_source_aliases_file": {"aliases": []},
    }
    previous_outputs = {"md_dir": str(tmp_path)}
    for key, payload in payloads.items():
        path = tmp_path / f"{key}.json"
        atomic_write_json(path, payload)
        previous_outputs[key] = str(path)
    catalog = ArtifactCatalog(
        records=[
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="dedup",
                uri=markdown.resolve().as_uri(),
                local_path=markdown,
                metadata={"source_url": legacy_url},
            ),
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="dedup",
                uri=document_markdown.resolve().as_uri(),
                local_path=document_markdown,
                metadata={
                    "source_url": "",
                    "source_file": str(source_pdf),
                    "source_type": "pdf",
                },
            ),
        ]
    )
    context = StageContext(
        run_id="prepare",
        project_name="prepare",
        config={
            "formatter": {
                "corpus_preparation": {
                    "expand_canonical_url_aliases": True,
                    "allow_media_only_pdf_assets": True,
                    "minimum_document_count": 2,
                    "minimum_unique_media_assets": 2,
                    "maximum_missing_media_files": 0,
                    "maximum_unbound_media_assets": 0,
                    "minimum_semantically_annotated_visuals": 2,
                    "minimum_ocr_adjudicated_visuals": 1,
                }
            }
        },
        work_dir=tmp_path / "run",
        previous_outputs=previous_outputs,
        stage_definition={"type": "formatter", "plugin": "corpus_preparation"},
        stage_id="prepare_corpus_boundary",
        artifact_catalog=catalog,
    )

    result = _run(CorpusPreparationFormatter().execute(context))

    assert result.status == StageStatus.COMPLETED
    inventory = load_json_safe(result.outputs["prepared_corpus_inventory_file"])
    report = load_json_safe(result.outputs["corpus_preparation_report_file"])
    assert inventory["representation_status"] == "undecided"
    assert inventory["document_count"] == 2
    assert inventory["media_only_asset_count"] == 1
    assert inventory["media_only_assets"][0]["content_hash"] == media_only_hash
    assert sum(
        document["media_reference_count"] for document in inventory["documents"]
    ) == 1
    source_kinds = {
        document["source_locator"]["kind"] for document in inventory["documents"]
    }
    assert source_kinds == {"url", "file"}
    assert report["gates"]["passed"] is True
    assert report["counts"]["source_file_only_documents"] == 1
    assert report["counts"]["raw_unbound_media_assets"] == 1
    assert report["counts"]["media_only_pdf_assets"] == 1
    assert report["counts"]["unbound_media_assets"] == 0
    prepared_mapping = load_json_safe(result.outputs["md_mapping_file"])
    assert prepared_mapping[legacy_url] == prepared_mapping[canonical_url]
    webpage_document = next(
        document for document in inventory["documents"] if document["source_url"]
    )
    assert webpage_document["source_url"] == canonical_url
    assert legacy_url in webpage_document["source_alias_urls"]
    assert report["gates"]["inventory_matches_live_markdown"] is True
    assert report["gates"]["all_media_linked_to_documents"] is False
    assert report["gates"]["all_media_accounted_for"] is True
    assert report["gates"]["document_media_references_resolve"] is True
    assert report["next_stage_boundary"] == {
        "chunking_performed": False,
        "document_representation_selected": False,
        "embedding_performed": False,
        "indexing_performed": False,
    }


def test_canonical_alias_expansion_rejects_conflicting_document_targets():
    canonical_url = "https://example.com/page"
    old_url = "https://example.com/old-page"

    expanded, report = _expand_canonical_url_aliases(
        {canonical_url: "/tmp/a.md", old_url: "/tmp/b.md"},
        {canonical_url: {"source_url": canonical_url, "redirected_from": [old_url]}},
    )

    assert expanded == {canonical_url: "/tmp/a.md", old_url: "/tmp/b.md"}
    assert report["identity_conflicts"] == 1
    assert report["aliases_added"] == 0


def test_html_artifact_identity_expands_renamed_route_without_fuzzy_matching(
    tmp_path: Path,
):
    markdown = tmp_path / "document.md"
    markdown.write_text("# Institute", encoding="utf-8")
    raw_html = tmp_path / "raw" / "capture-identity.html"
    cleaned_html = tmp_path / "clean" / "capture-identity.html"
    raw_html.parent.mkdir()
    cleaned_html.parent.mkdir()
    raw_html.write_text("<h1>Institute</h1>", encoding="utf-8")
    cleaned_html.write_text("<h1>Institute</h1>", encoding="utf-8")
    old_url = "https://example.com/research/institutes/institute"
    new_url = "https://example.com/research/our-institutes/institute"

    expanded, report = _expand_html_artifact_aliases(
        {old_url: str(markdown)},
        {new_url: {"html_path": str(raw_html)}},
        {str(markdown.resolve()): {"source_html_path": str(cleaned_html)}},
    )

    assert expanded[old_url] == str(markdown)
    assert expanded[new_url] == str(markdown.resolve())
    assert report == {
        "aliases_added": 1,
        "identities_expanded": 1,
        "identity_conflicts": 0,
    }


def test_media_only_contract_rejects_web_and_incomplete_pdf_evidence(tmp_path: Path):
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"pdf")
    image = tmp_path / "image.png"
    image.write_bytes(b"image")
    valid = normalize_media_item(
        {
            "type": "image",
            "source_type": "pdf",
            "source_url": "https://example.com/source.pdf",
            "source_file": str(source_pdf),
            "local_path": str(image),
            "content_hash": hashlib.sha256(image.read_bytes()).hexdigest(),
            "document_id": "source",
            "page_number": 1,
            "annotation_status": "completed",
        }
    )

    assert _is_media_only_pdf_eligible(valid) is True
    assert _is_media_only_pdf_eligible({**valid, "source_type": "html"}) is False
    assert _is_media_only_pdf_eligible({**valid, "annotation_status": "failed"}) is False
    assert _is_media_only_pdf_eligible({**valid, "source_url": "not-a-url"}) is False
    assert _is_media_only_pdf_eligible({**valid, "page_number": None}) is False
