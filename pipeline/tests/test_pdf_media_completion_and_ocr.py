from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record, save_artifact_catalog
from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.hybrid_ocr import (
    SCENE_OCR_ROUTE,
    UNLIMITED_OCR_ROUTE,
    assess_scene_ocr_lines,
    build_stratified_benchmark,
    choose_ocr_route,
    should_escalate_to_unlimited,
)
from pipeline.core.media import build_media_embedding_text, build_media_manifest, normalize_media_item
from pipeline.core.state import PipelineState, StageState, save_state
from pipeline.core.unlimited_ocr import (
    assess_ocr_quality,
    extract_gradio_final_output,
    strip_layout_markers,
)
from pipeline.stages.formatters.media_ocr_formatter import (
    MediaOcrFormatter,
    _load_adjudicated_batch_results,
    _ocr_one,
    _provider_block_code,
    _resolve_input_outputs,
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
from scripts.ocr.run_scene_ocr_batch import _resolve_image_path as _resolve_scene_image_path
from scripts.ocr.run_unlimited_ocr_batch import MODEL_REVISION as UNLIMITED_MODEL_REVISION
from scripts.ocr.run_unlimited_ocr_batch import _model_source_evidence


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


def test_unlimited_ocr_rejects_fragmented_browser_chrome_but_keeps_document_text():
    fragmented = "\n".join(
        f"<|det|>header [{index},1,{index + 1},2]<|/det|>{fragment}"
        for index, fragment in enumerate(
            [
                "or",
                "acl",
                "e.c",
                "lom",
                "d/s",
                "upp",
                "lier",
                "reg",
                "ist",
                "rat",
                "ion",
                "page",
                "Mohamed bin Zayed University of Artificial Intelligence",
            ]
        )
    )
    document = "\n".join(
        [
            "<|det|>title [1,1,2,2]<|/det|>Annual Performance Review",
            *[
                "<|det|>text [1,1,2,2]<|/det|>"
                + f"Section {_index + 1} explains the documented annual faculty review process."
                for _index in range(12)
            ],
        ]
    )

    fragmented_quality = assess_ocr_quality(fragmented)
    document_quality = assess_ocr_quality(document)

    assert fragmented_quality["status"] == "rejected_low_quality"
    assert "excessive_short_line_fragments" in fragmented_quality["quality_flags"]
    assert fragmented_quality["text"] == ""
    assert document_quality["status"] == "completed"
    assert document_quality["quality_metrics"]["short_fragment_line_ratio"] == 0.0


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


def test_hybrid_ocr_routing_and_stratified_benchmark_are_deterministic():
    bucket_sizes = {
        "document_fragment": 4,
        "screenshot": 4,
        "map": 3,
        "diagram": 2,
        "photo": 3,
        "portrait": 2,
        "logo": 1,
        "illustration": 1,
    }
    items = []
    counter = 0
    for image_kind, count in bucket_sizes.items():
        for index in range(count):
            counter += 1
            items.append(
                {
                    "type": "image",
                    "needs_ocr": True,
                    "content_hash": f"{counter:064x}",
                    "image_kind": image_kind,
                    "source_type": "pdf" if index % 2 == 0 else "html",
                    "visible_text": "نص عربي" if index == 0 else f"Visible text {counter}",
                }
            )

    first = build_stratified_benchmark(reversed(items))
    second = build_stratified_benchmark(items)

    assert len(first) == 20
    assert [item["content_hash"] for item in first] == [
        item["content_hash"] for item in second
    ]
    assert choose_ocr_route({"image_kind": "document_fragment"}) == SCENE_OCR_ROUTE
    assert choose_ocr_route({"image_kind": "map"}) == SCENE_OCR_ROUTE
    assert sum(item["ocr_route"] == UNLIMITED_OCR_ROUTE for item in first) == 0
    assert should_escalate_to_unlimited(
        {"image_kind": "screenshot"}, {"status": "rejected_low_quality"}
    )
    assert not should_escalate_to_unlimited(
        {"image_kind": "map"}, {"status": "rejected_low_quality"}
    )


def test_scene_ocr_gate_uses_exact_high_confidence_lines_only():
    quality = assess_scene_ocr_lines(
        [
            {"text": "MBZUAI", "confidence": 0.99, "box": [1, 2, 3, 4]},
            {"text": "uncertain", "confidence": 0.2, "box": [5, 6, 7, 8]},
        ]
    )
    rejected = assess_scene_ocr_lines(
        [{"text": "uncertain", "confidence": 0.6}],
        minimum_mean_confidence=0.75,
    )

    assert quality["status"] == "completed"
    assert quality["text"] == "MBZUAI"
    assert quality["quality_metrics"]["rejected_line_count"] == 1
    assert rejected["status"] == "rejected_low_quality"
    assert rejected["text"] == ""


def test_scene_ocr_gate_rejects_confident_but_incomplete_reference_mismatch():
    reference = " ".join(
        "The applicant confirms that every submitted detail is correct and authentic".split()
        * 3
    )
    quality = assess_scene_ocr_lines(
        [{"text": "subtihelcaiokehe ieiat hioa", "confidence": 0.97}],
        reference_text=reference,
    )
    expanded = assess_scene_ocr_lines(
        [
            {
                "text": reference + " Additional exact labels found elsewhere in the image",
                "confidence": 0.97,
            }
        ],
        reference_text=reference,
    )

    assert quality["status"] == "rejected_low_quality"
    assert "insufficient_visible_text_corroboration" in quality["quality_flags"]
    assert expanded["status"] == "completed"


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


def test_quarantined_pdf_layout_is_recovered_as_visual_evidence(tmp_path: Path):
    raw_run = tmp_path / "raw"
    downloads = raw_run / "downloads"
    downloads.mkdir(parents=True)
    source_pdf = downloads / "scanned-newsletter.pdf"
    pdf = fitz.open()
    page = pdf.new_page(width=300, height=300)
    page.draw_rect(fitz.Rect(30, 40, 270, 240), fill=(0.1, 0.3, 0.8))
    page.insert_text((65, 145), "Newsletter visual", fontsize=17, color=(1, 1, 1))
    pdf.save(source_pdf)
    pdf.close()
    mapping = raw_run / "mappings.json"
    source_url = "https://mbzuai.ac.ae/scanned-newsletter.pdf"
    atomic_write_json(mapping, {source_url: str(source_pdf)})

    conversion_run = tmp_path / "conversion"
    quarantine = conversion_run / "quarantine"
    quarantined_layouts = quarantine / "structured_documents"
    quarantined_layouts.mkdir(parents=True)
    atomic_write_json(
        quarantined_layouts / "scanned-newsletter.docling.json",
        {
            "name": "scanned-newsletter",
            "origin": {"filename": source_pdf.name},
            "pages": {"1": {"size": {"width": 300, "height": 300}}},
            "texts": [{"text": "Newsletter figure"}],
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
                    outputs={"quarantine_dir": str(quarantine)},
                )
            ],
            current_stage_index=1,
        ),
        conversion_run,
    )
    save_artifact_catalog(ArtifactCatalog(records=[]), conversion_run)

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
                    "include_quarantined_layouts": True,
                    "fallback_full_page_max_pages": 3,
                    "fallback_full_page_only_when_unrepresented": False,
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
        artifact_catalog=ArtifactCatalog(records=[]),
    )

    result = asyncio.run(PdfMediaCompletionFormatter().execute(context))

    assert result.status == StageStatus.COMPLETED
    assert result.metrics["discovered_pdf_documents"] == 1
    assert result.metrics["recovered_pdf_media"] == 2
    items = load_json_safe(result.outputs["extracted_images_index_file"])["items"]
    assert {item["crop_source"] for item in items} == {
        "docling_layout_recovery",
        "pdf_full_page_fallback",
    }
    assert {item["source_url"] for item in items} == {source_url}
    report = load_json_safe(result.outputs["pdf_media_completion_report_file"])
    assert report["conversion_run_evidence"][0]["quarantined_layout_count"] == 1
    assert report["documents"][0]["from_quarantine"] is True
    assert report["gates"]["passed"] is True


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


def test_batch_ocr_import_is_sha_pinned_and_coverage_complete(tmp_path: Path):
    content_hash = "a" * 64
    contract_hash = "b" * 64
    quality_revision = "hybrid-exact-ocr-quality-v1"
    raw_lines = [{"text": "MBZUAI", "confidence": 0.99, "box": [1, 2, 3, 4]}]
    raw_output_sha256 = hashlib.sha256(
        json.dumps(raw_lines, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    input_hash = hashlib.sha256(
        f"{contract_hash}:{content_hash}:{quality_revision}".encode()
    ).hexdigest()
    payload = {
        "version": 1,
        "kind": "hybrid_ocr_adjudicated_results",
        "batch_contract_sha256": contract_hash,
        "quality_revision": quality_revision,
        "results": {
            content_hash: {
                "content_hash": content_hash,
                "status": "completed",
                "text": "MBZUAI",
                "provider": "paddleocr",
                "provider_revision": "3.7.0",
                "model": "PP-OCRv5",
                "model_revision": "paddleocr-3.7.0/paddle-3.2.0",
                "raw_output_sha256": raw_output_sha256,
                "raw_evidence": {"lines": raw_lines},
                "ocr_input_hash": input_hash,
                "quality_score": 0.99,
                "quality_flags": [],
            }
        },
    }
    result_path = tmp_path / "adjudicated.json"
    atomic_write_json(result_path, payload)
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    config = {
        "batch_results": {"path": str(result_path), "sha256": digest},
        "batch_contract_sha256": contract_hash,
        "quality_revision": quality_revision,
    }

    imported, evidence = _load_adjudicated_batch_results(
        config, [{"content_hash": content_hash}]
    )

    assert imported[content_hash]["text"] == "MBZUAI"
    assert evidence["result_count"] == 1
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _load_adjudicated_batch_results(
            {**config, "batch_results": {"path": str(result_path), "sha256": "0" * 64}},
            [{"content_hash": content_hash}],
        )
    with pytest.raises(ValueError, match="coverage mismatch"):
        _load_adjudicated_batch_results(config, [{"content_hash": "c" * 64}])
    with pytest.raises(ValueError, match="coverage mismatch"):
        _load_adjudicated_batch_results(
            {**config, "allow_unused_batch_results": True},
            [{"content_hash": "c" * 64}],
        )

    unused_hash = "d" * 64
    unused = dict(payload["results"][content_hash])
    unused["content_hash"] = unused_hash
    unused["ocr_input_hash"] = hashlib.sha256(
        f"{contract_hash}:{unused_hash}:{quality_revision}".encode()
    ).hexdigest()
    payload["results"][unused_hash] = unused
    atomic_write_json(result_path, payload)
    subset_config = {
        **config,
        "batch_results": {
            "path": str(result_path),
            "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        },
        "allow_unused_batch_results": True,
    }

    imported, evidence = _load_adjudicated_batch_results(
        subset_config, [{"content_hash": content_hash}]
    )

    assert set(imported) == {content_hash}
    assert evidence["unused_result_count"] == 1


def test_standalone_ocr_inputs_are_sha_pinned(tmp_path: Path):
    specifications = {}
    for key in (
        "page_media_file",
        "page_images_file",
        "extracted_images_index_file",
        "media_manifest_file",
    ):
        path = tmp_path / f"{key}.json"
        atomic_write_json(path, {"key": key})
        specifications[key] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    context = StageContext(
        run_id="standalone-ocr",
        project_name="standalone-ocr",
        config={},
        work_dir=tmp_path / "run",
    )

    outputs, evidence = _resolve_input_outputs(
        context, {"input_artifacts": specifications}
    )

    assert set(evidence) == set(specifications)
    assert Path(outputs["media_manifest_file"]).is_file()
    specifications["media_manifest_file"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _resolve_input_outputs(context, {"input_artifacts": specifications})


def test_remote_ocr_runner_proves_model_revision_and_contains_batch_paths(
    tmp_path: Path,
):
    model_dir = tmp_path / "model"
    metadata_dir = model_dir / ".cache/huggingface/download"
    metadata_dir.mkdir(parents=True)
    model_files = {
        "config.json": b"{}",
        "tokenizer_config.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"layer": "model.safetensors"}}
        ).encode(),
        "model.safetensors": b"weights",
    }
    for filename, value in model_files.items():
        (model_dir / filename).write_bytes(value)
        (metadata_dir / f"{filename}.metadata").write_text(
            f"{UNLIMITED_MODEL_REVISION}\netag-{filename}\n",
            encoding="utf-8",
        )

    evidence = _model_source_evidence(model_dir)

    assert evidence["revision"] == UNLIMITED_MODEL_REVISION
    assert set(evidence["files"]) == set(model_files)
    (metadata_dir / "config.json.metadata").write_text(
        "wrong-revision\netag\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="not from the pinned revision"):
        _model_source_evidence(model_dir)

    batch_dir = tmp_path / "batch"
    assets_dir = batch_dir / "assets"
    assets_dir.mkdir(parents=True)
    manifest_path = batch_dir / "batch.json"
    manifest_path.write_text("{}", encoding="utf-8")
    image_path = assets_dir / "image.png"
    image_path.write_bytes(b"image")
    assert _resolve_scene_image_path(manifest_path, "assets/image.png") == image_path
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"outside")
    with pytest.raises(ValueError, match="escapes the batch directory"):
        _resolve_scene_image_path(manifest_path, "../outside.png")
