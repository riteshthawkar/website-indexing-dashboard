"""Real Docling integration and preservation of provenance at the adapter boundary."""

from dataclasses import replace
from io import BytesIO
from types import SimpleNamespace

import pytest

pytest.importorskip("docling")

from lawa_rag.ingestion import (
    AccessContext, AccessPolicy, Collection, ConversionFailure, ConversionOptions,
    DocumentProcessor, IngestionCatalog, LocalFolderConnector, Source, build_evaluation_corpus,
    prepare_collection, read_prepared, sync_source,
)
from pipeline.tests.lawa_rag_document_support import mixed_documents, pdf_bytes


def test_docling_office_html_csv_preparation_preserves_arabic_tables_and_citations(tmp_path):
    root = tmp_path / "documents"
    root.mkdir()
    originals = mixed_documents()
    for name, body in originals.items():
        (root / name).write_bytes(body)
    collection, access = Collection("w", "docs"), AccessContext("w", ("team",))
    catalog = IngestionCatalog(tmp_path / "catalog")
    connector = LocalFolderConnector(Source("docs", "folder", root.as_uri(),
                                             AccessPolicy(principals=("team",))))
    sync_source(catalog, collection, connector)
    report = prepare_collection(catalog, collection, access=access)
    assert report.ok, report.to_dict()
    assert report.counts["converted"] == 5
    assert prepare_collection(catalog, collection, access=access).counts["reused"] == 5
    prepared = [read_prepared(catalog, collection, item.record_id, access=access,
                             processor_key=report.processor_key)
                for item in catalog.records(collection, access=access)]
    assert all(item.document.structured_json for item in prepared)
    assert {item.record.processing_state for item in prepared} == {"raw"}
    blocks = [block for item in prepared for block in item.document.blocks]
    assert any("شروط القبول" in block.text for block in blocks)
    assert any(block.locator["kind"] == "slide_item" and block.locator["pages"] == [1]
               for block in blocks)
    tables = [block for block in blocks if block.kind == "table"]
    assert any("007" in block.text for block in tables)
    assert any(block.locator.get("sheet_name") == "Programs" for block in tables)
    snapshot = build_evaluation_corpus(catalog, collection, access=access,
                                       processor_key=report.processor_key)
    assert len(snapshot.sources) == 5 and any(unit.kind == "table" for unit in snapshot.evidence)
    assert all("item=" in unit.locator for unit in snapshot.evidence)
    assert {name: (root / name).read_bytes() for name in originals} == originals


def test_conversion_rejects_unprovisioned_ocr_formula_loss_and_oversized_input():
    from openpyxl import Workbook

    workbook = Workbook()
    workbook.active["A1"] = "=1+1"  # openpyxl does not calculate a cached result.
    stream = BytesIO()
    workbook.save(stream)
    options = ConversionOptions(max_input_bytes=32 * 1024)
    cases = [
        (".pdf", b"%PDF-1.4\n", "ocr_models_not_provisioned"),
        (".xlsx", stream.getvalue(), "formula_cache_missing"),
        (".png", b"not-a-captioned-image", "unsupported_format"),
        (".csv", b"a,b\n1,2,3\n", "inconsistent_csv_columns"),
        (".docx", b"x" * (options.max_input_bytes + 1), "input_limit_exceeded"),
    ]
    with DocumentProcessor(options) as processor:
        for suffix, body, code in cases:
            record = SimpleNamespace(locator="file:///example" + suffix, media_type="binary")
            with pytest.raises(ConversionFailure) as caught:
                processor.convert(record, body)
            assert caught.value.code == code


def test_docling_pdf_provenance_and_partial_success_fail_closed(tmp_path, monkeypatch):
    from docling.datamodel.base_models import ConversionStatus
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import (
        BoundingBox, DocItemLabel, DoclingDocument, ProvenanceItem, Size,
    )
    from lawa_rag.ingestion._docling import convert_document
    from lawa_rag.ingestion._docling_output import prepare_output

    doc = DoclingDocument(name="two-page")
    for page in (1, 2):
        doc.add_page(page_no=page, size=Size(width=100, height=100))
        doc.add_text(label=DocItemLabel.TEXT, text=f"Page {page} requirements", prov=ProvenanceItem(
            page_no=page, bbox=BoundingBox(l=1, t=2, r=90, b=20), charspan=(0, 19),
        ))
    options = ConversionOptions()
    output = prepare_output(doc, ".pdf", options, 2)
    assert [block.locator["pages"] for block in output.blocks] == [[1], [2]]
    assert output.blocks[1].locator["provenance"][0]["bbox"]["l"] == 1
    assert "pdf_extraction_not_quality_verified" in output.warnings
    with pytest.raises(ConversionFailure, match="pdf_page_coverage_incomplete"):
        prepare_output(doc, ".pdf", options, 3)
    # Mock only the upstream converter result: our adapter must not accept partial success.
    monkeypatch.setattr(DocumentConverter, "convert", lambda *a, **kw: SimpleNamespace(
        status=ConversionStatus.PARTIAL_SUCCESS, errors=[], document=doc,
    ))
    options = replace(options, docling_artifacts_path=str(tmp_path), ocr_model_path=str(tmp_path))
    with pytest.raises(ConversionFailure, match="docling_incomplete_conversion"):
        convert_document(".pdf", pdf_bytes((595, 842)), options)


def test_pdf_resource_guards_reject_large_pages_counts_and_bad_files_before_model_loading(tmp_path):
    options = ConversionOptions(docling_artifacts_path=str(tmp_path),
                                ocr_model_path=str(tmp_path), max_pages=1)
    record = SimpleNamespace(locator="file:///document.pdf", media_type="application/pdf")
    cases = [
        (pdf_bytes((2835, 3402)), "pdf_page_pixel_limit_exceeded"),
        (pdf_bytes((595, 842), (595, 842)), "page_limit_exceeded"),
        (b"%PDF-not-a-valid-document", "invalid_or_encrypted_pdf"),
    ]
    # Empty model directories: expected failures must precede any OCR model loading.
    with DocumentProcessor(options) as processor:
        for body, expected in cases:
            with pytest.raises(ConversionFailure, match=expected):
                processor.convert(record, body)
