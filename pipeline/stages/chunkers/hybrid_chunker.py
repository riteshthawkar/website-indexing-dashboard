"""
Hybrid chunker plugin.

Combines document hierarchy with token-aware balancing. Uses Docling's native
hybrid chunker for structured PDF artifacts when available and falls back to a
heading-aware markdown hybrid strategy for web and markdown content.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import ChunkerStage, StageContext, StageResult
from pipeline.core.chunking import ChunkLimitExceededError, token_counting_method
from pipeline.core.registry import register_stage
from pipeline.stages.chunkers.common import (
    collect_markdown_sources,
    collect_structured_documents,
    docling_chunks_from_json,
    hybrid_markdown_chunks,
    write_chunk_outputs,
)

logger = logging.getLogger(__name__)


@register_stage
class HybridChunker(ChunkerStage):
    name = "hybrid"
    description = "Hybrid hierarchical + token-aware chunking with Docling-native support."

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.chunker_config
        sources = collect_markdown_sources(ctx)
        if not sources:
            return StageResult.skipped("No markdown sources available for chunking")

        structured_docs = collect_structured_documents(ctx)
        target_tokens = int(config.get("target_tokens", 450))
        max_tokens = int(config.get("max_tokens", 650))
        overlap_tokens = int(config.get("overlap_tokens", 80))
        min_chunk_tokens = int(config.get("min_chunk_tokens", 140))
        max_chunks_per_document = int(config.get("max_chunks_per_document", 0))
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
                        overlap_tokens=overlap_tokens,
                        max_chunks_per_document=max_chunks_per_document,
                        always_emit_headings=always_emit_headings,
                    )
                except ChunkLimitExceededError:
                    raise
                except Exception as exc:
                    logger.warning("Docling hybrid chunking failed for %s: %s", path.name, exc)

            if not doc_chunks:
                text = path.read_text(encoding="utf-8", errors="replace")
                if not text.strip():
                    continue
                doc_chunks = hybrid_markdown_chunks(
                    text,
                    source_info=source,
                    target_tokens=target_tokens,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    min_chunk_tokens=min_chunk_tokens,
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
                "target_tokens": target_tokens,
                "max_tokens": max_tokens,
                "overlap_tokens": overlap_tokens,
                "min_chunk_tokens": min_chunk_tokens,
                "max_chunks_per_document": max_chunks_per_document,
                "chunk_limit_mode": "unlimited" if max_chunks_per_document <= 0 else "fail",
                "token_counting_method": token_counting_method(),
                "include_section_headings": include_section_headings,
                "use_docling_native": use_docling_native,
            },
        )
        logger.info("Hybrid chunker: %d documents -> %d chunks", chunk_index["document_count"], chunk_index["chunk_count"])
        return StageResult.success(
            outputs={
                "chunks_file": str(output_file),
                # Canonical alias consumed by graph/index preparation stages.
                "chunk_index_file": str(output_file),
                "chunk_count": chunk_index["chunk_count"],
                "chunk_document_count": chunk_index["document_count"],
            },
            metrics={
                "documents": chunk_index["document_count"],
                "chunks": chunk_index["chunk_count"],
            },
            artifacts=[artifact],
        )
