"""
PDF/Office document converter stage.

Extracts text from PDF, DOCX, PPTX, and XLSX files and converts to Markdown.
Uses PyMuPDF (fitz) for PDFs and python-docx/python-pptx for Office files.
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.core.base import ConverterStage, StageContext, StageResult
from pipeline.core.document_quality import assess_markdown_document
from pipeline.core.io import atomic_write_json, ensure_dir
from pipeline.core.media import build_media_manifest
from pipeline.core.registry import register_stage

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".doc"}


def _extract_pdf(path: Path) -> Optional[str]:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(f"## Page {i + 1}\n\n{text}")
        doc.close()
        return "\n\n".join(pages) if pages else None
    except ImportError:
        logger.warning("PyMuPDF not installed; trying pdfplumber")
        return _extract_pdf_plumber(path)
    except Exception as e:
        logger.error("PyMuPDF failed for %s: %s", path.name, e)
        return _extract_pdf_plumber(path)


def _extract_pdf_plumber(path: Path) -> Optional[str]:
    """Fallback PDF extraction using pdfplumber."""
    try:
        import pdfplumber

        pages = []
        with pdfplumber.open(str(path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(f"## Page {i + 1}\n\n{text}")
        return "\n\n".join(pages) if pages else None
    except ImportError:
        logger.error("Neither pymupdf nor pdfplumber installed")
        return None
    except Exception as e:
        logger.error("pdfplumber failed for %s: %s", path.name, e)
        return None


def _extract_docx(path: Path) -> Optional[str]:
    """Extract text from DOCX."""
    try:
        from docx import Document

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs) if paragraphs else None
    except ImportError:
        logger.error("python-docx not installed")
        return None
    except Exception as e:
        logger.error("DOCX extraction failed for %s: %s", path.name, e)
        return None


def _extract_pptx(path: Path) -> Optional[str]:
    """Extract text from PPTX."""
    try:
        from pptx import Presentation

        prs = Presentation(str(path))
        slides = []
        for i, slide in enumerate(prs.slides):
            texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    texts.append(shape.text_frame.text)
            if texts:
                slides.append(f"## Slide {i + 1}\n\n" + "\n\n".join(texts))
        return "\n\n".join(slides) if slides else None
    except ImportError:
        logger.error("python-pptx not installed")
        return None
    except Exception as e:
        logger.error("PPTX extraction failed for %s: %s", path.name, e)
        return None


def _extract_xlsx(path: Path) -> Optional[str]:
    """Extract readable sheet content from XLSX files."""
    try:
        from openpyxl import load_workbook

        workbook = load_workbook(filename=str(path), read_only=True, data_only=True)
        sections = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                values = [str(value).strip() for value in row if value not in (None, "")]
                if values:
                    rows.append(" | ".join(values))
            if rows:
                sections.append(f"## Sheet: {sheet.title}\n\n" + "\n".join(rows))
        workbook.close()
        return "\n\n".join(sections) if sections else None
    except ImportError:
        logger.error("openpyxl not installed")
        return None
    except Exception as e:
        logger.error("XLSX extraction failed for %s: %s", path.name, e)
        return None


def _extract_pdf_images(path: Path, image_output_dir: Path) -> List[Dict[str, Any]]:
    """Extract embedded images from a PDF into a sidecar directory."""
    try:
        import fitz
    except ImportError:
        return []

    extracted: List[Dict[str, Any]] = []
    try:
        doc = fitz.open(str(path))
        for page_index, page in enumerate(doc, start=1):
            for image_index, image_info in enumerate(page.get_images(full=True), start=1):
                xref = image_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image.get("image")
                image_ext = base_image.get("ext", "png")
                if not image_bytes:
                    continue

                image_path = image_output_dir / f"page_{page_index:03d}_img_{image_index:03d}.{image_ext}"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)
                extracted.append(
                    {
                        "type": "image",
                        "url": str(image_path.resolve().as_uri()),
                        "asset_uri": str(image_path.resolve().as_uri()),
                        "local_path": str(image_path.resolve()),
                        "source_file": str(path),
                        "page_number": page_index,
                        "alt": f"{path.stem} page {page_index} image {image_index}",
                        "md_path": "",
                        "source_type": "pdf",
                        "document_id": "",
                    }
                )
        doc.close()
    except Exception as exc:
        logger.warning("PDF image extraction failed for %s: %s", path.name, exc)
    return extracted


EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".doc": _extract_docx,
    ".pptx": _extract_pptx,
    ".xlsx": _extract_xlsx,
}


def _move_to_quarantine(path: Path, *, source_root: Path, quarantine_root: Path) -> Optional[Path]:
    if not path.exists():
        return None
    try:
        relative = path.resolve().relative_to(source_root.resolve())
    except Exception:
        relative = Path(path.name)
    destination = quarantine_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return destination


@register_stage
class PDFOfficeConverter(ConverterStage):
    name = "pdf_converter"
    description = "Extracts text from PDF, DOCX, PPTX files into Markdown."

    async def execute(self, ctx: StageContext) -> StageResult:
        document_artifacts = ctx.find_artifacts(artifact_type="downloaded_document") or ctx.find_artifacts(artifact_type="document")
        download_dir = ctx.previous_outputs.get("download_dir")
        if not document_artifacts:
            if not download_dir:
                return StageResult.skipped("No download_dir in previous outputs")

            download_dir = Path(download_dir)
            if not download_dir.is_dir():
                return StageResult.skipped(f"download_dir does not exist: {download_dir}")

        md_dir = ensure_dir(ctx.output_dir("markdown"))
        images_dir = ensure_dir(ctx.output_dir("extracted_images"))

        if document_artifacts:
            files = [
                Path(record.local_path)
                for record in document_artifacts
                if record.local_path and Path(record.local_path).is_file() and Path(record.local_path).suffix.lower() in SUPPORTED_EXTENSIONS
            ]
        else:
            files = [f for f in download_dir.rglob("*") if f.suffix.lower() in SUPPORTED_EXTENSIONS]
        if not files:
            return StageResult.skipped("No supported documents found in download_dir")

        logger.info("PDF/Office converter: %d files", len(files))

        converted = 0
        failed = 0
        validation_failed = 0
        extracted_images: List[Dict[str, Any]] = []
        artifacts = []
        quality_reports_dir = ensure_dir(ctx.output_dir("quality_reports"))
        quarantine_root = ensure_dir(ctx.output_dir("quarantine"))
        quarantine_markdown_dir = ensure_dir(quarantine_root / "markdown")
        quarantine_images_dir = ensure_dir(quarantine_root / "extracted_images")
        source_artifact_by_file = {
            str(Path(record.local_path).resolve()): record
            for record in document_artifacts
            if record.local_path
        }

        for fp in files:
            ext = fp.suffix.lower()
            extractor = EXTRACTORS.get(ext)
            if not extractor:
                continue

            text = extractor(fp)
            if text:
                relative = fp.relative_to(download_dir) if download_dir else Path(fp.name)
                out = md_dir / relative.with_suffix(".md")
                markdown = f"# {fp.stem}\n\n{text}"
                quality_report_path = quality_reports_dir / relative.with_suffix(".validation.json")
                report_metadata = None
                images: List[Dict[str, Any]] = []
                if ext == ".pdf":
                    images = _extract_pdf_images(fp, images_dir / relative.with_suffix(""))
                    for image in images:
                        image["md_path"] = str(out)
                        image["document_id"] = str(out.with_suffix("").name)
                assessment = assess_markdown_document(
                    markdown,
                    source_ext=ext,
                    config=ctx.converter_config,
                    media_items=images,
                )
                source_record = source_artifact_by_file.get(str(fp.resolve()))
                source_metadata = dict(source_record.metadata or {}) if source_record else {}
                atomic_write_json(
                    quality_report_path,
                    {
                        "source_file": str(fp),
                        "selected_backend": "pdf_converter",
                        "selected_markdown_path": str(out) if assessment["accepted"] else "",
                        "quarantined": not assessment["accepted"],
                        "selected_assessment": assessment,
                    },
                )
                artifacts.append(
                    ctx.make_artifact(
                        quality_report_path,
                        artifact_type="document_quality_report",
                        role="quality_report",
                        metadata={
                            "source_file": str(fp),
                            "selected_backend": "pdf_converter",
                            "accepted": assessment["accepted"],
                            "score": assessment["score"],
                        },
                        source_artifact_ids=[source_record.artifact_id] if source_record else None,
                    )
                )
                if assessment["accepted"]:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(markdown, encoding="utf-8")
                    converted += 1
                    extracted_images.extend(images)
                    artifacts.append(
                        ctx.make_artifact(
                            out,
                            artifact_type="markdown",
                            role="content",
                            metadata={
                                "source_file": str(fp),
                                "source_url": str(source_metadata.get("source_url") or ""),
                                "source_type": ext.lstrip(".") or "document",
                                "backend": "pdf_converter",
                                "quality_score": assessment["score"],
                                "quality_warnings": assessment["warnings"],
                                "quality_metrics": assessment["metrics"],
                            },
                            source_artifact_ids=[source_record.artifact_id] if source_record else None,
                        )
                    )
                else:
                    validation_failed += 1
                    failed += 1
                    quarantine_md_path = quarantine_markdown_dir / relative.with_suffix(".md")
                    quarantine_md_path.parent.mkdir(parents=True, exist_ok=True)
                    quarantine_md_path.write_text(markdown, encoding="utf-8")
                    if ext == ".pdf" and (images_dir / relative.with_suffix("")).exists():
                        _move_to_quarantine(
                            images_dir / relative.with_suffix(""),
                            source_root=images_dir,
                            quarantine_root=quarantine_images_dir,
                        )
                    logger.warning(
                        "Quarantined %s from pdf_converter because extracted content was not fit for indexing: %s",
                        fp.name,
                        ",".join(assessment["reasons"]),
                    )
                    continue
            else:
                failed += 1

        for image in extracted_images:
            artifacts.append(
                ctx.make_artifact(
                    image["local_path"],
                    artifact_type="extracted_image",
                    role="document_media",
                    metadata=image,
                )
            )

        images_index_path = ctx.stage_work_dir / "extracted_images_index.json"
        atomic_write_json(images_index_path, build_media_manifest(extracted_images, kind="document_media"))
        artifacts.append(
            ctx.make_artifact(
                images_index_path,
                artifact_type="media_manifest",
                role="document_media_index",
                metadata={"entries": len(extracted_images)},
            )
        )

        logger.info(
            "PDF/Office converter done: converted=%d failed=%d quarantined=%d",
            converted,
            failed,
            validation_failed,
        )

        return StageResult.success(
            outputs={
                "md_dir": str(md_dir),
                "doc_converted": converted,
                "images_dir": str(images_dir),
                "extracted_images_index_file": str(images_index_path),
                "extracted_images_count": len(extracted_images),
                "quality_reports_dir": str(quality_reports_dir),
                "quarantine_dir": str(quarantine_root),
            },
            metrics={
                "converted": converted,
                "failed": failed,
                "validation_failed": validation_failed,
                "images_extracted": len(extracted_images),
            },
            artifacts=artifacts,
        )
