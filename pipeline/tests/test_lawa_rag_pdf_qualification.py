"""Qualification must not equate successful parsing with release-quality evidence."""

import hashlib
import json

import pytest

from ci import qualify_lawa_rag_pdf as qualification
from lawa_rag.ingestion.documents import DocumentBlock, PreparedDocument


def test_fact_checks_require_correct_arabic_numbers_page_and_provenance():
    case = {"pages": [2], "checks": [{"page": 2, "contains": "ساعات العمل 8:00"}]}
    locator = {"kind": "pdf_item", "pages": [2], "provenance": [{"bbox": {"l": 1}}]}
    good = PreparedDocument((DocumentBlock("ساعات العمل 8:00", locator),))
    assert qualification.assess_document(good, case)["quality_passed"]
    for text, reference in [
        ("ساعات العمل 9:00", locator),
        ("8:00 لمعلا تاعاس", locator),
        ("ساعات العمل 8:00", {**locator, "pages": [1]}),
        ("ساعات العمل 8:00", {**locator, "provenance": []}),
    ]:
        document = PreparedDocument((DocumentBlock(text, reference),))
        assert not qualification.assess_document(document, case)["quality_passed"]


def test_failed_quality_stops_before_evaluation_even_when_conversion_succeeds(tmp_path, monkeypatch):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"qualification fixture; converter boundary injected below")
    manifest = tmp_path / "cases.json"
    manifest.write_text(json.dumps({"cases": [{
        "name": "case.pdf", "path": str(source), "pages": [1],
        "checks": [{"page": 1, "contains": "Opening hours: 8:00"}],
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }]}))
    conversion = tmp_path / "conversion.json"
    conversion.write_text(json.dumps({"schema_version": "lawa.ingestion.conversion.v1",
                                       "options": {}}))
    document = PreparedDocument((DocumentBlock("Opening hours: 9:00", {
        "kind": "pdf_item", "pages": [1], "provenance": [{"bbox": {"l": 1}}],
    }),))
    monkeypatch.setattr(qualification.MeasuredProcessor, "convert", lambda *args: document)
    monkeypatch.setattr(qualification, "build_evaluation_corpus", lambda *args, **kw:
                        pytest.fail("Failed quality must not reach evaluation"))
    output = tmp_path / "result"
    assert not qualification.qualify(manifest, conversion, output)
    report = json.loads((output / "report.json").read_text())
    assert report["conversion_complete"] and not report["quality_passed"]
    assert "evaluation_sources" not in report
    assert source.read_bytes().startswith(b"qualification fixture")
    # A stale/replaced fixture must be rejected before processing.
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs from the reviewed fixture"):
        qualification.qualify(manifest, conversion, tmp_path / "changed-result")
