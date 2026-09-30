"""Thin Docling converter adapter. No provider APIs, URL fetching or fallback parsers."""

from __future__ import annotations

import io
import json
import csv
from functools import lru_cache

from ._document_checks import preflight
from .documents import ConversionFailure, DocumentBlock, PreparedDocument


def convert_document(suffix, body, options):
    body = preflight(suffix, body, options)
    if suffix == ".json":
        text = body.decode("utf-8-sig", errors="strict")
        json.loads(text, parse_constant=_reject_constant)
        return PreparedDocument((DocumentBlock(
            "```json\n" + text + "\n```", {"kind": "json_pointer", "pointer": ""},
            kind="record", data_json=text,
        ),))
    from docling.datamodel.base_models import ConversionStatus, DocumentStream
    from ._docling_output import prepare_output

    result = _converter(options, suffix == ".pdf").convert(
        DocumentStream(name="document" + suffix, stream=io.BytesIO(body)),
        raises_on_error=False, max_num_pages=options.max_pages,
        max_file_size=options.max_input_bytes,
    )
    if result.status != ConversionStatus.SUCCESS or result.errors:
        raise ConversionFailure("docling_incomplete_conversion")
    if suffix == ".csv":
        _validate_csv(result.document, body)
    return prepare_output(result.document, suffix, options, result.input.page_count)


def _reject_constant(_):
    raise ConversionFailure("nonstandard_json_number")


def _validate_csv(document, body):
    rows = list(csv.reader(io.StringIO(body.decode("utf-8")), strict=True))
    if len(document.tables) != 1 or not rows:
        raise ConversionFailure("csv_coverage_incomplete")
    data = document.tables[0].data
    if (data.num_rows, data.num_cols) != (len(rows), len(rows[0])):
        raise ConversionFailure("csv_coverage_incomplete")
    actual = {(cell.start_row_offset_idx, cell.start_col_offset_idx): cell.text
              for cell in data.table_cells}
    expected = {(r, c): value for r, row in enumerate(rows) for c, value in enumerate(row)}
    if actual != expected:
        raise ConversionFailure("csv_values_changed")


@lru_cache(maxsize=2)
def _converter(options, with_pdf):
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.backend_options import HTMLBackendOptions
    from docling.document_converter import DocumentConverter, HTMLFormatOption

    formats = [InputFormat.DOCX, InputFormat.PPTX, InputFormat.XLSX,
               InputFormat.CSV, InputFormat.HTML]
    format_options = {InputFormat.HTML: HTMLFormatOption(backend_options=HTMLBackendOptions(
        fetch_images=False, render_page=False, enable_remote_fetch=False,
        enable_local_fetch=False, infer_furniture=False,
    ))}
    if with_pdf:
        formats = [InputFormat.PDF]
        format_options = {InputFormat.PDF: _pdf_options(options)}
    return DocumentConverter(allowed_formats=formats, format_options=format_options)


def _pdf_options(options):
    from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
    from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions
    from docling.document_converter import PdfFormatOption

    pipeline = PdfPipelineOptions(
        artifacts_path=options.docling_artifacts_path,
        do_ocr=True, do_table_structure=True, enable_remote_services=False,
        allow_external_plugins=False, document_timeout=options.timeout_seconds,
        accelerator_options=AcceleratorOptions(
            device=AcceleratorDevice.CPU, num_threads=options.cpu_threads,
        ),
        ocr_options=EasyOcrOptions(
            lang=list(options.ocr_languages), use_gpu=False, download_enabled=False,
            model_storage_directory=options.ocr_model_path,
        ),
    )
    return PdfFormatOption(pipeline_options=pipeline)
