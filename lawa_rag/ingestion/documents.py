"""Bounded document conversion contracts; original source records are never overwritten."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from ._util import canonical_json, digest, nonempty


DOCUMENT_SCHEMA_VERSION = "lawa.prepared.document.v1"


class ConversionFailure(ValueError):
    """Machine-readable reason, without document text or credentials in error messages."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ConversionOptions:
    recipe_version: str = "documents-v1"
    docling_artifacts_path: str = ""
    ocr_model_path: str = ""
    ocr_languages: tuple[str, ...] = ("en", "ar")
    max_input_bytes: int = 32 * 1024 * 1024
    max_output_bytes: int = 32 * 1024 * 1024
    max_archive_bytes: int = 128 * 1024 * 1024
    max_pages: int = 500
    max_pdf_page_pixels: int = 16_000_000
    max_blocks: int = 20_000
    max_rows: int = 50_000
    max_cells: int = 500_000
    timeout_seconds: int = 120
    cpu_threads: int = 2
    csv_delimiter: str = ","

    def __post_init__(self) -> None:
        nonempty(self.recipe_version, "recipe_version")
        for name in ("max_input_bytes", "max_output_bytes", "max_archive_bytes", "max_pages",
                     "max_pdf_page_pixels", "max_blocks", "max_rows", "max_cells",
                     "timeout_seconds", "cpu_threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.csv_delimiter not in {",", ";", "\t", "|"}:
            raise ValueError("Unsupported CSV delimiter")
        if isinstance(self.ocr_languages, str) or not self.ocr_languages:
            raise ValueError("ocr_languages must be a non-empty sequence")
        for language in self.ocr_languages:
            nonempty(language, "OCR language")
        object.__setattr__(self, "ocr_languages", tuple(self.ocr_languages))


@dataclass(frozen=True)
class DocumentBlock:
    text: str
    locator: dict[str, Any]
    kind: str = "text"
    data_json: str = ""

    def __post_init__(self) -> None:
        nonempty(self.text, "block text")
        if not isinstance(self.locator, dict) or not self.locator.get("kind"):
            raise ValueError("Document block requires a typed locator")
        if self.kind not in {"text", "table", "record"}:
            raise ValueError("Unsupported document block kind")
        canonical_json(self.locator)
        if self.data_json:
            json.loads(self.data_json)


@dataclass(frozen=True)
class PreparedDocument:
    blocks: tuple[DocumentBlock, ...]
    warnings: tuple[str, ...] = ()
    structured_json: str = ""
    schema_version: str = DOCUMENT_SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> PreparedDocument:
        if value.get("schema_version") != DOCUMENT_SCHEMA_VERSION:
            raise ValueError("Unsupported prepared document schema")
        return cls(tuple(DocumentBlock(**item) for item in value["blocks"]),
                   tuple(value.get("warnings", ())), value.get("structured_json", ""))

    def validate(self, options: ConversionOptions) -> None:
        if not self.blocks:
            raise ConversionFailure("empty_document")
        if len(self.blocks) > options.max_blocks:
            raise ConversionFailure("block_limit_exceeded")
        if len(canonical_json(self.to_dict()).encode()) > options.max_output_bytes:
            raise ConversionFailure("output_limit_exceeded")


class DocumentProcessor:
    """Docling adapter. Use as a context manager to reuse a bounded, warm worker."""

    def __init__(self, options: ConversionOptions | None = None):
        self.options = options or ConversionOptions()
        self._worker = None

    def __enter__(self):
        from ._document_worker import ConversionWorker

        if self._worker is not None:
            raise RuntimeError("Processor is already open")
        self._worker = ConversionWorker(self.options)
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None

    @property
    def fingerprint(self) -> str:
        packages = ("docling", "docling-core", "docling-parse", "pypdfium2", "easyocr",
                    "python-docx", "python-pptx", "openpyxl", "beautifulsoup4")
        versions = {}
        for package in packages:
            try:
                versions[package] = version(package)
            except PackageNotFoundError:
                versions[package] = "not-installed"
        return digest([asdict(self.options), versions, "conversion-worker-v2"])

    def convert(self, record, body: bytes) -> PreparedDocument:
        from ._document_worker import ConversionWorker

        if len(body) > self.options.max_input_bytes:
            raise ConversionFailure("input_limit_exceeded")
        suffix = PurePosixPath(urlsplit(record.locator).path).suffix.lower()
        if record.media_type == "application/pdf":
            suffix = ".pdf"
        if self._worker is None:
            with ConversionWorker(self.options) as worker:
                result = worker.convert(suffix, body)
        else:
            result = self._worker.convert(suffix, body)
        result.validate(self.options)
        return result


__all__ = [
    "ConversionFailure", "ConversionOptions", "DOCUMENT_SCHEMA_VERSION", "DocumentBlock",
    "DocumentProcessor", "PreparedDocument",
]
