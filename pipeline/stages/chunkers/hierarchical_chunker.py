"""
Hierarchical chunker plugin.

Uses Docling's native hierarchical chunker when a structured document artifact
is available, and falls back to heading/section-aware markdown chunking for
web content and non-Docling pipelines.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import ChunkerStage, StageContext, StageResult
from pipeline.core.registry import register_stage
from pipeline.stages.chunkers.common import (
    collect_markdown_sources,
    collect_structured_documents,
    docling_chunks_from_json,
    hierarchical_markdown_chunks,
    write_chunk_outputs,
)

logger = logging.getLogger(__name__)


@register_stage
class HierarchicalChunker(ChunkerStage):
    name = "hierarchical"
    description = "Structure-aware hierarchical chunking for markdown and Docling documents."

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.chunker_config
        sources = collect_markdown_sources(ctx)
        if not sources:
            return StageResult.skipped("No markdown sources available for chunking")

        structured_docs = collect_structured_documents(ctx)
        max_tokens = int(config.get("max_tokens", 650))
        max_chunks_per_document = int(config.get("max_chunks_per_document", 64))
        include_section_headings = bool(config.get("include_section_headings", True))
        always_emit_headings = bool(config.get("always_emit_headings", False))
        use_docling_native = bool(config.get("use_docling_native", True))

        chunks: List[Dict[str, Any]] = []
        source_artifact_ids: List[str] = []

        for source in sources:
            path = Path(source["path"]).resolve()
            structured_doc_path = structured_docs.get(str(path))
            doc_chunks: List[Dict[str, Any]] = []

            if use_docling_native and structured_doc_path and structured_doc_path.is_file():
                try:
                    doc_chunks = docling_chunks_from_json(
                        structured_doc_path,
                        strategy=self.name,
                        source_info=source,
                        max_tokens=max_tokens,
                        overlap_tokens=0,
                        max_chunks_per_document=max_chunks_per_document,
                        always_emit_headings=always_emit_headings,
                    )
                except Exception as exc:
                    logger.warning("Docling hierarchical chunking failed for %s: %s", path.name, exc)

            if not doc_chunks:
                text = path.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    continue
                doc_chunks = hierarchical_markdown_chunks(
                    text,
                    source_info=source,
                    max_tokens=max_tokens,
                    max_chunks_per_document=max_chunks_per_document,
                    include_section_headings=include_section_headings,
                )

            chunks.extend(doc_chunks)
            if source.get("artifact_id"):
                source_artifact_ids.append(source["artifact_id"])

        output_file, chunk_index, artifact = write_chunk_outputs(
            ctx,
            strategy=self.name,
            chunks=chunks,
            source_artifact_ids=source_artifact_ids,
            metadata={
                "max_tokens": max_tokens,
                "include_section_headings": include_section_headings,
                "use_docling_native": use_docling_native,
            },
        )
        logger.info(
            "Hierarchical chunker: %d documents -> %d chunks",
            chunk_index["document_count"],
            chunk_index["chunk_count"],
        )
        return StageResult.success(
            outputs={
                "chunks_file": str(output_file),
                "chunk_count": chunk_index["chunk_count"],
                "chunk_document_count": chunk_index["document_count"],
            },
            metrics={
                "documents": chunk_index["document_count"],
                "chunks": chunk_index["chunk_count"],
            },
            artifacts=[artifact],
        )
