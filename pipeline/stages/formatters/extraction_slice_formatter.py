from __future__ import annotations

from collections import defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pipeline.core.assertions import authority_score, clean_text, infer_authority_class, unique_strings
from pipeline.core.base import FormatterStage, StageContext, StageResult
from pipeline.core.chunking import load_chunk_index
from pipeline.core.io import atomic_write_json
from pipeline.core.registry import register_stage


def _slice_id(*parts: Any) -> str:
    raw = "|".join(clean_text(part) for part in parts if clean_text(part))
    if not raw:
        raw = "slice"
    return f"slice:{sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def _estimate_tokens(text: str) -> int:
    words = [token for token in clean_text(text).split() if token]
    return max(1, int(len(words) * 1.33)) if words else 0


def _section_anchor(chunk: Dict[str, Any]) -> Tuple[str, ...]:
    section_path = [clean_text(part) for part in (chunk.get("section_path") or []) if clean_text(part)]
    if section_path:
        return tuple(section_path[:2])
    heading = clean_text(chunk.get("heading"))
    if heading:
        return (heading,)
    pages = [str(value) for value in (chunk.get("page_numbers") or []) if value not in (None, "")]
    if pages:
        return (f"pages:{','.join(pages[:3])}",)
    return ("__root__",)


def _chunk_text_for_slice(chunk: Dict[str, Any]) -> str:
    heading = clean_text(chunk.get("heading"))
    text = clean_text(chunk.get("text"))
    if heading and heading.lower() not in text.lower():
        return f"{heading}\n{text}".strip()
    return text


@register_stage
class ExtractionSliceFormatter(FormatterStage):
    name = "extraction_slices"
    description = "Builds coherent section-aligned extraction slices for LLM assertion extraction."

    async def execute(self, ctx: StageContext) -> StageResult:
        chunk_index_artifacts = ctx.find_artifacts(artifact_type="chunk_index")
        chunk_index_payload: Dict[str, Any] = {}
        if chunk_index_artifacts:
            latest = chunk_index_artifacts[-1].local_path
            if latest:
                chunk_index_payload = load_chunk_index(latest)
        elif ctx.previous_outputs.get("chunks_file"):
            chunk_index_payload = load_chunk_index(ctx.previous_outputs.get("chunks_file"))

        chunk_records = list(chunk_index_payload.get("chunks") or [])
        if not chunk_records:
            return StageResult.failure("No chunk_index available for extraction slice formatting")

        cfg = dict(ctx.assertions_config or {})
        target_tokens = max(800, int(cfg.get("slice_target_tokens") or 2200))
        max_tokens = max(target_tokens, int(cfg.get("slice_max_tokens") or 3200))
        section_margin = max(200, int(cfg.get("slice_section_margin_tokens") or 320))

        chunks_by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for chunk in chunk_records:
            if not isinstance(chunk, dict):
                continue
            doc_key = (
                clean_text(chunk.get("source_markdown_path"))
                or clean_text(chunk.get("document_id"))
                or clean_text(chunk.get("source_url"))
                or clean_text(chunk.get("document_title"))
            )
            chunks_by_doc[doc_key].append(chunk)

        slices: List[Dict[str, Any]] = []
        for doc_key, chunks in chunks_by_doc.items():
            chunks.sort(key=lambda item: int(item.get("chunk_index") or 0))
            current_chunks: List[Dict[str, Any]] = []
            current_tokens = 0
            current_anchor: Tuple[str, ...] | None = None

            def flush() -> None:
                nonlocal current_chunks, current_tokens, current_anchor
                if not current_chunks:
                    return
                first = current_chunks[0]
                linked_chunk_ids = [clean_text(chunk.get("chunk_id")) or clean_text(chunk.get("id")) for chunk in current_chunks]
                page_numbers = unique_strings(
                    str(page)
                    for chunk in current_chunks
                    for page in (chunk.get("page_numbers") or [])
                    if page not in (None, "")
                )
                section_paths = [
                    [clean_text(part) for part in (chunk.get("section_path") or []) if clean_text(part)]
                    for chunk in current_chunks
                ]
                merged_sections = []
                for section in section_paths:
                    if section and section not in merged_sections:
                        merged_sections.append(section)
                source_markdown_path = clean_text(first.get("source_markdown_path"))
                source_url = clean_text(first.get("source_url"))
                document_title = clean_text(first.get("document_title"))
                document_type = clean_text(first.get("document_type"))
                text = "\n\n".join(_chunk_text_for_slice(chunk) for chunk in current_chunks if _chunk_text_for_slice(chunk))
                authority_class = infer_authority_class(
                    source_url=source_url,
                    document_title=document_title,
                    document_type=document_type,
                    source_markdown_path=source_markdown_path,
                )
                slices.append(
                    {
                        "id": _slice_id(
                            source_markdown_path,
                            document_title,
                            current_anchor or (),
                            linked_chunk_ids[0] if linked_chunk_ids else "",
                            linked_chunk_ids[-1] if linked_chunk_ids else "",
                        ),
                        "document_id": clean_text(first.get("document_id")),
                        "document_title": document_title,
                        "document_type": document_type,
                        "source_backend": clean_text(first.get("source_backend")),
                        "source_file": clean_text(first.get("source_file")),
                        "source_markdown_path": source_markdown_path,
                        "source_url": source_url,
                        "section_anchor": list(current_anchor or ()),
                        "section_paths": merged_sections,
                        "page_numbers": [int(value) for value in page_numbers if str(value).isdigit()],
                        "linked_chunk_ids": [chunk_id for chunk_id in linked_chunk_ids if chunk_id],
                        "chunk_start_index": int(current_chunks[0].get("chunk_index") or 0),
                        "chunk_end_index": int(current_chunks[-1].get("chunk_index") or 0),
                        "chunk_count": len(current_chunks),
                        "token_count": _estimate_tokens(text),
                        "text": text,
                        "authority_class": authority_class,
                        "authority_score": authority_score(
                            source_url=source_url,
                            document_title=document_title,
                            document_type=document_type,
                            source_markdown_path=source_markdown_path,
                        ),
                    }
                )
                current_chunks = []
                current_tokens = 0
                current_anchor = None

            for chunk in chunks:
                chunk_tokens = int(chunk.get("token_count") or _estimate_tokens(chunk.get("text") or ""))
                chunk_anchor = _section_anchor(chunk)
                if not current_chunks:
                    current_chunks = [chunk]
                    current_tokens = chunk_tokens
                    current_anchor = chunk_anchor
                    continue

                same_anchor = chunk_anchor == current_anchor
                projected_tokens = current_tokens + chunk_tokens
                if projected_tokens <= target_tokens:
                    current_chunks.append(chunk)
                    current_tokens = projected_tokens
                    continue
                if same_anchor and projected_tokens <= max_tokens:
                    current_chunks.append(chunk)
                    current_tokens = projected_tokens
                    continue
                if not same_anchor and projected_tokens <= target_tokens + section_margin:
                    current_chunks.append(chunk)
                    current_tokens = projected_tokens
                    continue

                flush()
                current_chunks = [chunk]
                current_tokens = chunk_tokens
                current_anchor = chunk_anchor

            flush()

        slices_file = ctx.stage_work_dir / "extraction_slices.json"
        atomic_write_json(slices_file, slices)

        artifact = ctx.make_artifact(
            slices_file,
            artifact_type="extraction_slices",
            role="assertion_extraction_slices",
            metadata={
                "slice_count": len(slices),
                "document_count": len(chunks_by_doc),
                "target_tokens": target_tokens,
                "max_tokens": max_tokens,
            },
        )

        return StageResult.success(
            outputs={
                "extraction_slices_file": str(slices_file),
            },
            metrics={
                "extraction_slices": len(slices),
                "extraction_slice_documents": len(chunks_by_doc),
            },
            artifacts=[artifact],
        )
