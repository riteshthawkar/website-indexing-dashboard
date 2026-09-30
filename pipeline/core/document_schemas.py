"""Versioned document-conversion schemas shared across ingestion sources."""

DOCLING_PAGE_WINDOWS_SCHEMA = "docling_page_windows.v1"
LEGACY_DOCLING_PAGE_WINDOWS_SCHEMA = "mbzuai_docling_page_windows.v1"
SUPPORTED_DOCLING_PAGE_WINDOWS_SCHEMAS = frozenset(
    {DOCLING_PAGE_WINDOWS_SCHEMA, LEGACY_DOCLING_PAGE_WINDOWS_SCHEMA}
)

__all__ = [
    "DOCLING_PAGE_WINDOWS_SCHEMA",
    "LEGACY_DOCLING_PAGE_WINDOWS_SCHEMA",
    "SUPPORTED_DOCLING_PAGE_WINDOWS_SCHEMAS",
]
