from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record, save_artifact_catalog
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import build_media_embedding_text, build_media_manifest, normalize_media_item
from pipeline.core.state import PipelineState, StageState, save_state
from pipeline.core.unlimited_ocr import (
    assess_ocr_quality,
    extract_gradio_final_output,
    strip_layout_markers,
)
from pipeline.stages.formatters.media_ocr_formatter import (
    MediaOcrFormatter,
    _ocr_one,
    _provider_block_code,
    _retryable_error,
    _session_auth_headers,
)
from pipeline.stages.formatters.media_semantics_formatter import (
    _apply_verified_annotation_seed_manifests,
    _seed_annotations_from_completed_input,
)
from pipeline.stages.formatters.pdf_media_completion_formatter import (
    PdfMediaCompletionFormatter,
)


def _png(path: Path) -> str:
    image = Image.new("RGB", (320, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 20, 300, 200), fill="navy")
    draw.text((60, 90), "MBZUAI OCR", fill="white")
    image.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_unlimited_ocr_keeps_final_cumulative_snapshot_and_filters_layout_markup():
    sse = "\n".join(
        [
            "event: generating",
            'data: [{"text":"First","done":false}]',
            "event: generating",
            'data: [{"text":"<|det|>text [1,2,3,4]<|/det|>MBZUAI\\n<|det|>image [1,2,3,4]<|/det|>![x](crop.png)","done":true}]',
            "event: complete",
        ]
    )

    raw = extract_gradio_final_output(sse)
    quality = assess_ocr_quality(raw)

    assert "First" not in raw
    assert strip_layout_markers(raw) == "MBZUAI"
    assert quality["status"] == "completed"
    assert quality["text"] == "MBZUAI"


def test_unlimited_ocr_auth_and_quota_circuit(monkeypatch):
    monkeypatch.setenv("TEST_HF_TOKEN", "private-token")
    config = {
        "provider": "gradio_space",
        "auth_token_env": "TEST_HF_TOKEN",
        "primary_mode": "gundam",
        "retry_mode": "base",
        "retry_attempts": 3,
    }
    quota_error = RuntimeError(
        "You have exceeded your ZeroGPU quota (90s requested vs. 0s left)"
    )

    assert _session_auth_headers(config) == {"Authorization": "Bearer private-token"}
    assert _provider_block_code(quota_error) == "zerogpu_quota_exhausted"
    assert _retryable_error(quota_error) is False


def test_unlimited_ocr_provider_wide_failure_skips_remaining_requests(monkeypatch):
    calls = 0

    async def blocked_provider(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("You have exceeded your ZeroGPU quota (90s requested vs. 0s left)")

    monkeypatch.setattr(
        "pipeline.stages.formatters.media_ocr_formatter._provider_call",
        blocked_provider,
    )
    config = {
        "provider": "gradio_space",
        "primary_mode": "gundam",
        "retry_mode": "base",
        "retry_attempts": 3,
    }
    first = {"content_hash": "a" * 64}
    second = {"content_hash": "b" * 64}

    async def run():
        semaphore = asyncio.Semaphore(1)
        circuit_breaker = asyncio.Event()
        circuit_state = {}
        first_result = await _ocr_one(
            None,
            item=first,
            config=config,
            semaphore=semaphore,
            circuit_breaker=circuit_breaker,
            circuit_state=circuit_state,
        )
        second_result = await _ocr_one(
            None,
            item=second,
            config=config,
            semaphore=semaphore,
            circuit_breaker=circuit_breaker,
            circuit_state=circuit_state,
        )
        return first_result, second_result

    (_, first_result), (_, second_result) = asyncio.run(run())

    assert calls == 1
    assert first_result["attempts"] == 1
    assert second_result["attempts"] == 0
    assert second_result["failure_code"] == "zerogpu_quota_exhausted"


def test_completed_semantics_are_rekeyed_only_with_all_occurrence_captions():
    content_hash = "a" * 64
    items = [
        {
            "type": "image",
            "content_hash": content_hash,
            "context_reference_id": "media-ref:new-a",
            "semantic_caption": "A visible campus map.",
            "contextual_caption": "Campus navigation map.",
            "visual_description": "A map with labeled buildings and paths.",
            "image_kind": "map",
            "semantic_tags": ["campus", "map"],
            "semantic_relevance": "substantive",
            "annotation_status": "completed",
            "annotation_provider": "google-gemini",
            "annotation_model": "gemini-3.5-flash-lite",
            "annotation_model_revision": "gemini-3.5-flash-lite",
            "annotation_prompt_revision": "mbzuai-media-semantics-v2-section-context",
            "annotation_confidence": 0.96,
            "needs_ocr": True,
        }
    ]
    queue = [
        {
            "content_hash": content_hash,
            "annotation_input_hash": "new-input-hash",
            "reference_contexts": [{"reference_id": "media-ref:new-a"}],
        }
    ]

    seeded, stats = _seed_annotations_from_completed_input(
        items,
        queue,
        prompt_revision="mbzuai-media-semantics-v2-section-context",
    )

    assert stats["reused"] == 1
    assert seeded[content_hash]["annotation_input_hash"] == "new-input-hash"
    assert seeded[content_hash]["contextual_captions"][0]["reference_id"] == "media-ref:new-a"
    assert seeded[content_hash]["annotation_reuse_basis"].startswith("verified_content_hash")


def test_external_annotation_seed_is_sha_pinned_and_path_independent(tmp_path: Path):
    content_hash = "b" * 64
    seed_item = normalize_media_item(
        {
            "type": "image",
            "id": "stable-figure",
            "url": "file:///old/run/figure.png",
            "local_path": "/old/run/figure.png",
            "content_hash": content_hash,
            "source_type": "pdf",
            "source_url": "https://mbzuai.ac.ae/guide.pdf",
            "document_id": "guide",
            "page_number": 3,
            "crop_source": "docling_layout_recovery",
            "semantic_caption": "A three-stage process diagram.",
            "contextual_caption": "Process diagram in the guide.",
            "visual_description": "Three connected boxes.",
            "image_kind": "diagram",
            "annotation_status": "completed",
            "annotation_provider": "google-gemini",
            "annotation_model": "gemini-3.5-flash-lite",
            "annotation_prompt_revision": "mbzuai-media-semantics-v2-section-context",
            "annotation_confidence": 0.9,
        }
    )
    seed_path = tmp_path / "seed.json"
    atomic_write_json(seed_path, build_media_manifest([seed_item]))
    digest = hashlib.sha256(seed_path.read_bytes()).hexdigest()
    current = {
        **seed_item,
        "url": "file:///new/run/figure.png",
        "local_path": "/new/run/figure.png",
        "semantic_caption": "",
        "contextual_caption": "",
        "visual_description": "",
        "image_kind": "",
        "annotation_status": "",
    }

    imported, evidence = _apply_verified_annotation_seed_manifests(
        [current],
        [{"path": str(seed_path), "sha256": digest}],
        prompt_revision="mbzuai-media-semantics-v2-section-context",
    )

    assert imported[0]["annotation_status"] == "completed"
    assert imported[0]["semantic_caption"] == "A three-stage process diagram."
    assert evidence[0]["matched_unique_content_hash_count"] == 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _apply_verified_annotation_seed_manifests(
            [current],
            [{"path": str(seed_path), "sha256": "0" * 64}],
            prompt_revision="mbzuai-media-semantics-v2-section-context",
        )


def test_page_window_pdf_media_is_recovered_with_public_url(tmp_path: Path):
    raw_run = tmp_path / "raw"
    downloads = raw_run / "downloads"
    downloads.mkdir(parents=True)
    source_pdf = downloads / "windowed.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=300)
    page.draw_rect(fitz.Rect(30, 40, 270, 240), fill=(0.1, 0.3, 0.8))
    page.insert_text((75, 145), "Windowed visual", fontsize=17, color=(1, 1, 1))
    pdf.save(source_pdf)
    pdf.close()
    mapping = raw_run / "mappings.json"
    atomic_write_json(mapping, {"https://mbzuai.ac.ae/windowed.pdf": str(source_pdf)})

    conversion_run = tmp_path / "conversion"
    structured_dir = conversion_run / "structured"
    structured_dir.mkdir(parents=True)
    part = structured_dir / "windowed.docling.pages_0001_0001.json"
    atomic_write_json(
        part,
        {
            "name": "windowed",
            "pages": {"1": {"size": {"width": 300, "height": 300}}},
            "texts": [{"text": "A windowed diagram"}],
            "pictures": [
                {
                    "captions": [{"$ref": "#/texts/0"}],
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 30,
                                "t": 260,
                                "r": 270,
                                "b": 60,
                                "coord_origin": "BOTTOMLEFT",
                            },
                        }
                    ],
                }
            ],
        },
    )
    wrapper = structured_dir / "windowed.docling.json"
    atomic_write_json(
        wrapper,
        {
            "schema": "mbzuai_docling_page_windows.v1",
            "source_file": str(source_pdf),
            "page_count": 1,
            "parts": [
                {
                    "page_range": [1, 1],
                    "backend": "docling",
                    "structured_document_path": str(part),
                }
            ],
        },
    )
    quarantine = conversion_run / "quarantine"
    save_state(
        PipelineState(
            run_id="conversion",
            project_name="conversion",
            status="paused",
            stages=[
                StageState(
                    name="docling",
                    stage_type="converter",
                    stage_id="convert_documents",
                    status="completed",
                    outputs={
                        "structured_documents_dir": str(structured_dir),
                        "quarantine_dir": str(quarantine),
                    },
                )
            ],
            current_stage_index=1,
        ),
        conversion_run,
    )
    save_artifact_catalog(
        ArtifactCatalog(
            records=[
                build_artifact_record(
                    artifact_type="structured_document",
                    role="docling_page_windows",
                    producer_stage="convert_documents",
                    uri=wrapper.resolve().as_uri(),
                    local_path=wrapper,
                    metadata={
                        "source_file": str(source_pdf),
                        "backend": "docling_page_windows",
                    },
                )
            ]
        ),
        conversion_run,
    )

    markdown = tmp_path / "windowed.md"
    markdown.write_text("# Windowed\n\nA windowed PDF visual.", encoding="utf-8")
    empty_page = tmp_path / "page.json"
    empty_document = tmp_path / "document.json"
    empty_manifest = tmp_path / "manifest.json"
    atomic_write_json(empty_page, {})
    atomic_write_json(empty_document, build_media_manifest([]))
    atomic_write_json(empty_manifest, build_media_manifest([]))
    context = StageContext(
        run_id="completion",
        project_name="completion",
        config={
            "formatter": {
                "pdf_media_completion": {
                    "conversion_run_dirs": [str(conversion_run)],
                    "raw_mapping_files": [str(mapping)],
                    "include_unused_fallback_layouts": True,
                    "pdf_crop_scale": 2.0,
                    "min_pdf_crop_width": 96,
                    "min_pdf_crop_height": 72,
                    "minimum_pdf_documents": 1,
                    "maximum_failed_documents": 0,
                }
            }
        },
        work_dir=tmp_path / "completion",
        previous_outputs={
            "page_media_file": str(empty_page),
            "page_images_file": str(empty_page),
            "extracted_images_index_file": str(empty_document),
            "media_manifest_file": str(empty_manifest),
        },
        stage_definition={
            "id": "complete_pdf_media",
            "type": "formatter",
            "plugin": "pdf_media_completion",
        },
        stage_id="complete_pdf_media",
        artifact_catalog=ArtifactCatalog(
            records=[
                build_artifact_record(
                    artifact_type="markdown",
                    role="content",
                    producer_stage="merge",
                    uri=markdown.resolve().as_uri(),
                    local_path=markdown,
                    metadata={"source_file": str(source_pdf)},
                )
            ]
        ),
    )

    result = asyncio.run(PdfMediaCompletionFormatter().execute(context))

    assert result.status == StageStatus.COMPLETED
    assert result.metrics["recovered_pdf_media"] == 1
    items = load_json_safe(result.outputs["extracted_images_index_file"])["items"]
    assert items[0]["source_url"] == "https://mbzuai.ac.ae/windowed.pdf"
    assert items[0]["crop_source"] == "docling_layout_recovery"
    report = load_json_safe(result.outputs["pdf_media_completion_report_file"])
    assert report["gates"]["passed"] is True
    assert report["extraction_metrics"]["layout_boxes_page_cap_filtered"] == 0


def test_media_ocr_propagates_only_quality_gated_exact_text(tmp_path: Path, monkeypatch):
    image_path = tmp_path / "ocr.png"
    content_hash = _png(image_path)
    item = normalize_media_item(
        {
            "type": "image",
            "url": image_path.resolve().as_uri(),
            "local_path": str(image_path),
            "content_hash": content_hash,
            "mime_type": "image/png",
            "source_type": "pdf",
            "source_url": "https://mbzuai.ac.ae/test.pdf",
            "needs_ocr": True,
            "visible_text": "MBZUAI OCR",
        }
    )
    page_file = tmp_path / "page.json"
    document_file = tmp_path / "document.json"
    manifest_file = tmp_path / "manifest.json"
    atomic_write_json(page_file, {})
    atomic_write_json(document_file, build_media_manifest([item]))
    atomic_write_json(manifest_file, build_media_manifest([item]))

    async def fake_provider_call(*_args, **_kwargs):
        raw = "<|det|>text [1,2,3,4]<|/det|>MBZUAI OCR"
        return {
            **assess_ocr_quality(raw),
            "raw_output": raw,
            "raw_output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "mode": "gundam",
            "latency_ms": 12.5,
        }

    monkeypatch.setattr(
        "pipeline.stages.formatters.media_ocr_formatter._provider_call",
        fake_provider_call,
    )
    artifact = build_artifact_record(
        artifact_type="extracted_image",
        role="document_figure",
        producer_stage="annotate",
        uri=image_path.resolve().as_uri(),
        local_path=image_path,
        metadata=item,
    )
    context = StageContext(
        run_id="ocr",
        project_name="ocr",
        config={
            "formatter": {
                "media_ocr": {
                    "provider": "gradio_space",
                    "endpoint": "https://baidu-unlimited-ocr.hf.space",
                    "model": "baidu/Unlimited-OCR",
                    "model_revision": "pinned",
                    "provider_revision": "space-pinned",
                    "primary_mode": "gundam",
                    "retry_mode": "base",
                    "retry_low_quality_with_base": True,
                    "scope": "all",
                    "needs_ocr_only": True,
                    "concurrency": 1,
                    "request_timeout_sec": 30,
                    "retry_attempts": 1,
                    "require_complete": True,
                }
            }
        },
        work_dir=tmp_path / "ocr-run",
        previous_outputs={
            "page_media_file": str(page_file),
            "page_images_file": str(page_file),
            "extracted_images_index_file": str(document_file),
            "media_manifest_file": str(manifest_file),
        },
        stage_definition={"id": "enrich_exact_ocr", "type": "formatter", "plugin": "media_ocr"},
        stage_id="enrich_exact_ocr",
        artifact_catalog=ArtifactCatalog(records=[artifact]),
    )

    result = asyncio.run(MediaOcrFormatter().execute(context))

    assert result.status == StageStatus.COMPLETED
    output_item = load_json_safe(result.outputs["extracted_images_index_file"])["items"][0]
    assert output_item["ocr_status"] == "completed"
    assert output_item["ocr_text"] == "MBZUAI OCR"
    assert output_item["ocr_provider_revision"] == "space-pinned"
    assert "exact_ocr=MBZUAI OCR" in build_media_embedding_text([output_item])
    raw_result = load_json_safe(result.outputs["media_ocr_results_file"])["results"][content_hash]
    assert raw_result["raw_output"].startswith("<|det|>")
