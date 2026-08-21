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
from pipeline.stages.formatters.corpus_merge_formatter import _load_source_descriptor
from pipeline.stages.formatters.corpus_preparation_formatter import (
    CorpusPreparationFormatter,
)
from pipeline.stages.quality.dedup_filter import DedupFilter


HAS_DATASKETCH = importlib.util.find_spec("datasketch") is not None


def _run(coro):
    return asyncio.run(coro)


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
            "source_url": "https://example.com/document",
            "annotation_status": "completed",
            "needs_ocr": True,
            "ocr_status": "completed",
            "ocr_text": "MBZUAI",
        }
    )
    payloads = {
        "md_mapping_file": {"https://example.com/document": str(markdown)},
        "canonical_page_metadata_file": {
            "https://example.com/document": {
                "source_url": "https://example.com/document",
                "title": "Document",
            }
        },
        "url_identity_map_file": {"records": []},
        "canonical_page_link_graph_file": {"nodes": [], "edges": []},
        "page_media_file": {"https://example.com/document": [media_item]},
        "page_images_file": {"https://example.com/document": [media_item]},
        "page_videos_file": {},
        "extracted_images_index_file": build_media_manifest([]),
        "media_manifest_file": build_media_manifest([media_item]),
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
                metadata={"source_url": "https://example.com/document"},
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
                    "minimum_document_count": 2,
                    "minimum_unique_media_assets": 1,
                    "maximum_missing_media_files": 0,
                    "minimum_semantically_annotated_visuals": 1,
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
    assert sum(
        document["media_reference_count"] for document in inventory["documents"]
    ) == 1
    source_kinds = {
        document["source_locator"]["kind"] for document in inventory["documents"]
    }
    assert source_kinds == {"url", "file"}
    assert report["gates"]["passed"] is True
    assert report["counts"]["source_file_only_documents"] == 1
    assert report["gates"]["inventory_matches_live_markdown"] is True
    assert report["next_stage_boundary"] == {
        "chunking_performed": False,
        "document_representation_selected": False,
        "embedding_performed": False,
        "indexing_performed": False,
    }
