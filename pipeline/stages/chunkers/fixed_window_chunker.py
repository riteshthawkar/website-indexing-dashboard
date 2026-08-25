"""
Fixed-window chunker plugin.

Useful as a baseline or fallback when structure is unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from pipeline.core.base import ChunkerStage, StageContext, StageResult
from pipeline.core.chunking import token_counting_method
from pipeline.core.registry import register_stage
from pipeline.stages.chunkers.common import (
    collect_markdown_sources,
    fixed_window_chunks,
    write_chunk_outputs,
)

logger = logging.getLogger(__name__)


@register_stage
class FixedWindowChunker(ChunkerStage):
    name = "fixed_window"
    description = "Fixed-window paragraph chunking baseline."

    async def execute(self, ctx: StageContext) -> StageResult:
        config = ctx.chunker_config
        sources = collect_markdown_sources(ctx)
        if not sources:
            return StageResult.skipped("No markdown sources available for chunking")

        target_tokens = int(config.get("target_tokens", 450))
        overlap_tokens = int(config.get("overlap_tokens", 80))
        max_chunks_per_document = int(config.get("max_chunks_per_document", 0))

        chunks: List[Dict[str, Any]] = []
        source_artifact_ids: List[str] = []
        for source in sources:
            text = Path(source["path"]).read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                continue
            chunks.extend(
                fixed_window_chunks(
                    text,
                    source_info=source,
                    target_tokens=target_tokens,
                    overlap_tokens=overlap_tokens,
                    max_chunks_per_document=max_chunks_per_document,
                )
            )
            if source.get("artifact_id"):
                source_artifact_ids.append(source["artifact_id"])

        output_file, chunk_index, artifact = write_chunk_outputs(
            ctx,
            strategy=self.name,
            chunks=chunks,
            source_artifact_ids=source_artifact_ids,
            metadata={
                "target_tokens": target_tokens,
                "overlap_tokens": overlap_tokens,
                "max_chunks_per_document": max_chunks_per_document,
                "chunk_limit_mode": "unlimited" if max_chunks_per_document <= 0 else "fail",
                "token_counting_method": token_counting_method(),
            },
        )
        logger.info("Fixed chunker: %d documents -> %d chunks", chunk_index["document_count"], chunk_index["chunk_count"])
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
