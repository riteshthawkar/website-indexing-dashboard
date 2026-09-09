from __future__ import annotations

import asyncio
from collections import Counter
from io import BytesIO
from pathlib import Path

import fitz
from PIL import Image, ImageDraw

from pipeline.core.artifacts import ArtifactCatalog, build_artifact_record
from pipeline.core.base import StageContext
from pipeline.core.io import atomic_write_json
from pipeline.core.media import normalize_media_item
from pipeline.core.media_context import build_media_reference_contexts
from pipeline.core.unlimited_ocr import image_regions, normalized_bbox_to_pixels
from pipeline.stages.formatters.media_enrichment_formatter import (
    _download_image_bounded,
    _extract_pdf_figures,
    _image_inspection,
)


def _png_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((5, 5, width - 5, height - 5), fill="navy")
    draw.line((0, 0, width, height), fill="orange", width=3)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _avif_bytes(width: int, height: int) -> bytes:
    image = Image.new("RGB", (width, height), "teal")
    draw = ImageDraw.Draw(image)
    draw.ellipse((10, 10, width - 10, height - 10), fill="orange")
    buffer = BytesIO()
    image.save(buffer, format="AVIF")
    return buffer.getvalue()


def test_unlimited_ocr_image_boxes_are_page_aware_and_convert_to_pixels():
    output = "\n".join(
        [
            "<PAGE><|det|>image [10, 20, 900, 800]<|/det|>",
            "<|det|>text [20, 820, 900, 900]<|/det|>caption",
            "<PAGE><|det|>figure [100, 200, 500, 700]<|/det|>",
        ]
    )

    regions = image_regions(output)

    assert [region["page_number"] for region in regions] == [1, 2]
    assert normalized_bbox_to_pixels(regions[0]["bbox_1000"], width=1000, height=500) == (
        10,
        10,
        900,
        400,
    )


def test_image_inspection_rejects_icons_and_retains_hashes_for_content_images():
    icon, icon_reason = _image_inspection(
        _png_bytes(32, 32),
        minimum_width=120,
        minimum_height=90,
        minimum_area=30000,
        maximum_pixels=50_000_000,
        maximum_aspect_ratio=12.0,
        minimum_stddev=2.0,
    )
    content, content_reason = _image_inspection(
        _png_bytes(400, 300),
        minimum_width=120,
        minimum_height=90,
        minimum_area=30000,
        maximum_pixels=50_000_000,
        maximum_aspect_ratio=12.0,
        minimum_stddev=2.0,
    )

    assert icon is None
    assert icon_reason == "too_small"
    assert content_reason == ""
    assert content and len(content["content_hash"]) == 64
    assert len(content["perceptual_hash"]) == 16


def test_image_inspection_accepts_avif_content_images():
    content, reason = _image_inspection(
        _avif_bytes(400, 300),
        minimum_width=120,
        minimum_height=90,
        minimum_area=30000,
        maximum_pixels=50_000_000,
        maximum_aspect_ratio=12.0,
        minimum_stddev=2.0,
    )

    assert reason == ""
    assert content and content["extension"] == ".avif"
    assert content["mime_type"] == "image/avif"


def test_web_download_queue_is_bounded_before_request_timeout(monkeypatch, tmp_path):
    from urllib.parse import urlsplit

    import pipeline.stages.formatters.media_enrichment_formatter as module

    active_global = 0
    active_by_host: Counter[str] = Counter()
    maximum_global = 0
    maximum_by_host: Counter[str] = Counter()

    async def fake_download(
        session,
        *,
        url,
        output_dir,
        allowed_hosts,
        dns_cache,
        config,
        request_headers_by_host=None,
    ):
        nonlocal active_global, maximum_global
        host = urlsplit(url).hostname or ""
        active_global += 1
        active_by_host[host] += 1
        maximum_global = max(maximum_global, active_global)
        maximum_by_host[host] = max(maximum_by_host[host], active_by_host[host])
        await asyncio.sleep(0.01)
        active_by_host[host] -= 1
        active_global -= 1
        return {"url": url, "status": "accepted", "reason": ""}

    monkeypatch.setattr(module, "_download_image", fake_download)

    async def exercise_queue():
        global_semaphore = asyncio.Semaphore(3)
        host_semaphores = {}
        urls = [f"https://one.example/image-{index}.jpg" for index in range(8)]
        urls += [f"https://two.example/image-{index}.jpg" for index in range(8)]
        return await asyncio.gather(
            *(
                _download_image_bounded(
                    object(),
                    url=url,
                    output_dir=tmp_path,
                    allowed_hosts={"one.example", "two.example"},
                    dns_cache={},
                    config={},
                    global_semaphore=global_semaphore,
                    host_semaphores=host_semaphores,
                    per_host_concurrency=2,
                )
                for url in urls
            )
        )

    results = asyncio.run(exercise_queue())

    assert len(results) == 16
    assert maximum_global == 3
    assert maximum_by_host == {"one.example": 2, "two.example": 2}


def test_media_credentials_are_not_forwarded_to_allowlisted_redirect_host(monkeypatch, tmp_path):
    import pipeline.stages.formatters.media_enrichment_formatter as module

    class FakeResponse:
        def __init__(self, status, headers=None):
            self.status = status
            self.headers = headers or {}
            self.content_length = 0
            self.connection = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

    class FakeSession:
        def __init__(self):
            self.calls = []
            self.responses = [
                FakeResponse(302, {"Location": "https://cdn.example/image.jpg"}),
                FakeResponse(404),
            ]

        def get(self, url, *, allow_redirects, headers=None):
            self.calls.append({"url": url, "headers": headers})
            return self.responses.pop(0)

    async def public_host(host, cache):
        return True

    monkeypatch.setattr(module, "_host_resolves_publicly", public_host)
    monkeypatch.setattr(module, "_peer_is_public", lambda response: True)
    session = FakeSession()

    result = asyncio.run(
        module._download_image(
            session,
            url="https://protected.example/image.jpg",
            output_dir=tmp_path,
            allowed_hosts={"protected.example", "access.example", "cdn.example"},
            dns_cache={},
            config={
                "download_retry_attempts": 1,
                "fetch_host_aliases": {"protected.example": "access.example"},
            },
            request_headers_by_host={
                "access.example": {"Cookie": "private-session"}
            },
        )
    )

    assert result["reason"] == "http_404"
    assert session.calls == [
        {
            "url": "https://access.example/image.jpg",
            "headers": {"Cookie": "private-session"},
        },
        {"url": "https://cdn.example/image.jpg", "headers": None},
    ]


def test_media_normalization_preserves_multimodal_provenance():
    item = normalize_media_item(
        {
            "type": "image",
            "url": "https://example.test/figure.png",
            "content_hash": "abc",
            "perceptual_hash": "1234",
            "ocr_text": "Visible label",
            "ocr_model": "baidu/Unlimited-OCR",
            "ocr_model_revision": "pinned-revision",
            "crop_source": "unlimited_ocr_layout",
            "bbox": {"l": 1, "t": 2, "r": 3, "b": 4, "coord_origin": "TOPLEFT"},
        }
    )

    assert item["ocr_text"] == "Visible label"
    assert item["ocr_model"] == "baidu/Unlimited-OCR"
    assert item["crop_source"] == "unlimited_ocr_layout"
    assert item["bbox"]["coord_origin"] == "TOPLEFT"


def test_docling_layout_boxes_materialize_pdf_figure_crops(tmp_path: Path):
    source_pdf = tmp_path / "sample.pdf"
    document = fitz.open()
    page = document.new_page(width=300, height=300)
    page.draw_rect(fitz.Rect(30, 40, 270, 240), color=(0, 0, 0.8), fill=(0.2, 0.6, 0.9))
    page.insert_text((70, 140), "MBZUAI figure", fontsize=18, color=(1, 1, 1))
    document.save(source_pdf)
    document.close()

    markdown = tmp_path / "sample.md"
    markdown.write_text("# Sample\n\nDocument context", encoding="utf-8")
    structured = tmp_path / "sample.docling.json"
    atomic_write_json(
        structured,
        {
            "name": "sample",
            "origin": {"filename": source_pdf.name, "mimetype": "application/pdf"},
            "pages": {"1": {"page_no": 1, "size": {"width": 300, "height": 300}}},
                "texts": [
                    {
                        "text": "An informative architecture diagram",
                    "prov": [
                        {
                            "page_no": 1,
                            "bbox": {
                                "l": 40,
                                "t": 275,
                                "r": 260,
                                "b": 250,
                                "coord_origin": "BOTTOMLEFT",
                            },
                            }
                        ],
                    },
                    {
                        "text": "Document context",
                        "prov": [
                            {
                                "page_no": 1,
                                "bbox": {
                                    "l": 40,
                                    "t": 45,
                                    "r": 260,
                                    "b": 20,
                                    "coord_origin": "BOTTOMLEFT",
                                },
                            }
                        ],
                    },
                ],
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
    catalog = ArtifactCatalog(
        records=[
            build_artifact_record(
                artifact_type="structured_document",
                role="docling_document",
                producer_stage="convert_documents",
                uri=structured.resolve().as_uri(),
                local_path=structured,
                metadata={
                    "source_file": str(source_pdf),
                    "source_markdown_path": str(markdown),
                    "source_url": "https://mbzuai.ac.ae/sample.pdf",
                },
            ),
            build_artifact_record(
                artifact_type="markdown",
                role="content",
                producer_stage="convert_documents",
                uri=markdown.resolve().as_uri(),
                local_path=markdown,
                metadata={"source_file": str(source_pdf)},
            ),
        ]
    )
    context = StageContext(
        run_id="test-run",
        project_name="test",
        config={
            "formatter": {
                "media_enrichment": {
                    "min_pdf_bbox_area_ratio": 0.003,
                    "max_pdf_bbox_area_ratio": 0.88,
                }
            }
        },
        work_dir=tmp_path,
        previous_outputs={},
        stage_definition={"id": "enrich_media", "type": "formatter", "plugin": "media_enrichment"},
        stage_id="enrich_media",
        artifact_catalog=catalog,
    )

    items, metrics = _extract_pdf_figures(
        context,
        output_dir=context.output_dir("pdf_crops"),
        config=context.formatter_config["media_enrichment"],
    )

    assert metrics["pdf_crops_accepted"] == 1
    assert len(items) == 1
    assert Path(items[0]["local_path"]).is_file()
    assert items[0]["caption"] == "An informative architecture diagram"
    assert items[0]["page_number"] == 1
    assert items[0]["crop_source"] == "docling_layout"

    records, _lookup, stats = build_media_reference_contexts(
        items,
        markdown_mapping={},
        html_mapping={},
        page_metadata={
            "https://mbzuai.ac.ae/sample.pdf": {"title": "Sample architecture"}
        },
        config={"context_before_blocks": 1, "context_after_blocks": 2},
    )
    assert len(records) == 1
    assert records[0]["context_source"] == "docling_layout+markdown"
    assert records[0]["section_path"] == ["Sample"]
    assert records[0]["nearby_text"] == "Document context"
    assert stats["with_surrounding_text"] == 1
