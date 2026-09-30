"""Map Docling's native structured output into citation-bearing evidence blocks."""

from __future__ import annotations

from ._util import canonical_json
from .documents import ConversionFailure, DocumentBlock, PreparedDocument


def prepare_output(document, suffix, options, page_count=None):
    from docling_core.types.doc import ContentLayer, TableItem, TextItem

    blocks, covered, cells, rows = [], set(), 0, 0
    for item, _ in document.iterate_items(included_content_layers=set(ContentLayer)):
        if isinstance(item, TableItem):
            rows += item.data.num_rows
            cells += item.data.num_rows * item.data.num_cols
            if rows > options.max_rows or cells > options.max_cells:
                raise ConversionFailure("table_limit_exceeded")
            text, kind = item.export_to_markdown(doc=document), "table"
        elif isinstance(item, TextItem):
            text, kind = item.text, "text"
            if str(item.label) in {"section_header", "title"}:
                text = "## " + text
        else:
            continue
        if not text.strip():
            continue
        locator = _locator(document, item, suffix)
        covered.update(value.page_no for value in item.prov)
        blocks.append(DocumentBlock(text, locator, kind))
        if len(blocks) > options.max_blocks:
            raise ConversionFailure("block_limit_exceeded")
    if len(document.pages) > options.max_pages:
        raise ConversionFailure("page_limit_exceeded")
    if suffix == ".pdf":
        expected = set(range(1, (page_count or 0) + 1))
        if not expected or set(document.pages) != expected or not expected.issubset(covered):
            raise ConversionFailure("pdf_page_coverage_incomplete")
    return PreparedDocument(tuple(blocks), _warnings(document, suffix),
                            canonical_json(document.export_to_dict()))


def _warnings(document, suffix):
    warnings = ["visual_content_not_captioned"] if document.pictures else []
    if suffix == ".pdf":
        warnings.append("pdf_extraction_not_quality_verified")
    if suffix in {".docx", ".pptx", ".xlsx"}:
        warnings.append("office_rendering_and_embedded_objects_not_verified")
    return tuple(warnings)


def _locator(document, item, suffix):
    kind = {".pdf": "pdf_item", ".pptx": "slide_item", ".xlsx": "sheet_item",
            ".csv": "csv_table"}.get(suffix, "document_item")
    locator = {"kind": kind, "docling_ref": item.self_ref, "label": str(item.label),
               "provenance": [value.model_dump(mode="json") for value in item.prov]}
    if item.prov:
        locator["pages"] = sorted({value.page_no for value in item.prov})
    if suffix == ".xlsx" and item.parent:
        parent = item.parent.resolve(document)
        locator["sheet_name"] = getattr(parent, "name", "").removeprefix("sheet: ")
    if suffix == ".csv":
        locator["row_start"], locator["row_end"] = 1, item.data.num_rows
    return locator
