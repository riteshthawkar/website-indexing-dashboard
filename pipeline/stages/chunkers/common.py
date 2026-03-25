"""
Shared helpers for pluggable chunker stages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline.core.base import StageContext
from pipeline.core.chunking import (
    build_chunk_index,
    estimate_token_count,
    stable_document_id,
)
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def collect_markdown_sources(ctx: StageContext) -> List[Dict[str, Any]]:
    markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
    sources: List[Dict[str, Any]] = []

    if markdown_artifacts:
        for record in markdown_artifacts:
            if not record.local_path or not Path(record.local_path).is_file():
                continue
            metadata = dict(record.metadata or {})
            path = Path(record.local_path).resolve()
            sources.append(
                {
                    "path": path,
                    "artifact_id": record.artifact_id,
                    "metadata": metadata,
                    "source_url": str(metadata.get("source_url") or ""),
                    "source_backend": str(metadata.get("backend") or ""),
                    "document_title": metadata.get("document_title") or path.stem,
                    "document_type": str(metadata.get("source_type") or ""),
                    "source_file": str(metadata.get("source_file") or ""),
                }
            )
        return sources

    md_dir = ctx.previous_outputs.get("md_dir")
    if not md_dir:
        return []

    mapping_file = ctx.previous_outputs.get("md_mapping_file")
    url_to_md = load_json_safe(mapping_file, {}) if mapping_file else {}
    md_to_url = {str(Path(v).resolve()): k for k, v in url_to_md.items()}

    for path in Path(md_dir).glob("**/*.md"):
        resolved = path.resolve()
        sources.append(
            {
                "path": resolved,
                "artifact_id": None,
                "metadata": {},
                "source_url": md_to_url.get(str(resolved), ""),
                "source_backend": "",
                "document_title": resolved.stem,
                "document_type": "",
                "source_file": "",
            }
        )
    return sources


def collect_structured_documents(ctx: StageContext) -> Dict[str, Path]:
    by_md_path: Dict[str, Path] = {}
    for record in ctx.find_artifacts(artifact_type="structured_document"):
        if not record.local_path:
            continue
        path = Path(record.local_path)
        if not path.is_file():
            continue
        md_path = str(record.metadata.get("source_markdown_path") or "")
        if md_path:
            by_md_path[str(Path(md_path).resolve())] = path.resolve()
    return by_md_path


def write_chunk_outputs(
    ctx: StageContext,
    *,
    strategy: str,
    chunks: Iterable[Dict[str, Any]],
    source_artifact_ids: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    chunk_dir = ensure_dir(ctx.output_dir("chunks"))
    chunk_index = build_chunk_index(chunks, strategy=strategy, metadata=metadata)
    output_file = chunk_dir / "chunk_index.json"
    atomic_write_json(output_file, chunk_index)
    artifact = ctx.make_artifact(
        output_file,
        artifact_type="chunk_index",
        role="retrieval_chunks",
        metadata={
            "strategy": strategy,
            "documents": chunk_index["document_count"],
            "chunks": chunk_index["chunk_count"],
        },
        source_artifact_ids=source_artifact_ids,
    )
    return output_file, chunk_index, artifact


def classify_block(lines: List[str]) -> str:
    nonempty = [line for line in lines if line.strip()]
    if not nonempty:
        return "paragraph"
    if nonempty[0].strip().startswith("```"):
        return "code"
    if len(nonempty) >= 2 and "|" in nonempty[0] and TABLE_SEPARATOR_RE.match(nonempty[1]):
        return "table"
    if all(LIST_RE.match(line) for line in nonempty):
        return "list"
    return "paragraph"


def extract_markdown_blocks(text: str) -> List[Dict[str, Any]]:
    lines = (text or "").splitlines()
    blocks: List[Dict[str, Any]] = []
    current_lines: List[str] = []
    in_code = False
    section_path: List[str] = []

    def flush_current() -> None:
        nonlocal current_lines
        if not current_lines:
            return
        block_text = "\n".join(current_lines).strip()
        if block_text:
            blocks.append(
                {
                    "text": block_text,
                    "element_type": classify_block(current_lines),
                    "section_path": list(section_path),
                }
            )
        current_lines = []

    for line in lines:
        if line.strip().startswith("```"):
            if in_code:
                current_lines.append(line)
                flush_current()
                in_code = False
                continue
            flush_current()
            in_code = True
            current_lines = [line]
            continue

        if in_code:
            current_lines.append(line)
            continue

        match = HEADING_RE.match(line)
        if match:
            flush_current()
            level = len(match.group(1))
            heading = match.group(2).strip()
            section_path = section_path[: level - 1] + [heading]
            blocks.append(
                {
                    "text": line.strip(),
                    "element_type": "heading",
                    "section_path": list(section_path),
                    "heading": heading,
                }
            )
            continue

        if not line.strip():
            flush_current()
            continue

        current_lines.append(line)

    flush_current()
    return blocks


def group_blocks_by_section(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for block in blocks:
        section_key = tuple(block.get("section_path") or [])
        if current is None or tuple(current["section_path"]) != section_key:
            current = {
                "section_path": list(block.get("section_path") or []),
                "blocks": [],
                "element_types": [],
            }
            sections.append(current)
        current["blocks"].append(block)
        current["element_types"].append(block.get("element_type") or "paragraph")

    normalized: List[Dict[str, Any]] = []
    pending_heading_blocks: List[Dict[str, Any]] = []

    for section in sections:
        has_non_heading_content = any(
            (block.get("element_type") or "paragraph") != "heading"
            for block in section["blocks"]
        )
        if not has_non_heading_content:
            pending_heading_blocks.extend(section["blocks"])
            continue

        if pending_heading_blocks:
            section["blocks"] = [*pending_heading_blocks, *section["blocks"]]
            section["element_types"] = [
                *(block.get("element_type") or "paragraph" for block in pending_heading_blocks),
                *section["element_types"],
            ]
            pending_heading_blocks = []
        normalized.append(section)

    if pending_heading_blocks:
        if normalized:
            normalized[-1]["blocks"].extend(pending_heading_blocks)
            normalized[-1]["element_types"].extend(
                block.get("element_type") or "paragraph"
                for block in pending_heading_blocks
            )
        else:
            normalized.append(
                {
                    "section_path": list(pending_heading_blocks[-1].get("section_path") or []),
                    "blocks": pending_heading_blocks,
                    "element_types": [block.get("element_type") or "paragraph" for block in pending_heading_blocks],
                }
            )

    return normalized


def render_section_prefix(section_path: List[str]) -> str:
    if not section_path:
        return ""
    return "\n".join(f"{'#' * min(idx + 1, 6)} {heading}" for idx, heading in enumerate(section_path))


def build_section_text(section: Dict[str, Any], *, include_section_headings: bool = True) -> str:
    block_text = "\n\n".join(block["text"].strip() for block in section["blocks"] if block.get("text"))
    if not include_section_headings:
        return block_text.strip()
    if block_text.startswith("#") or not section.get("section_path"):
        return block_text.strip()
    prefix = render_section_prefix(list(section.get("section_path") or []))
    return f"{prefix}\n\n{block_text}".strip() if prefix else block_text.strip()


def split_text_by_budget(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> List[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if estimate_token_count(cleaned) <= max_tokens:
        return [cleaned]

    units = [unit.strip() for unit in re.split(r"\n{2,}", cleaned) if unit.strip()]
    if len(units) <= 1:
        units = [unit.strip() for unit in SENTENCE_RE.split(cleaned) if unit.strip()]
    if len(units) <= 1:
        words = cleaned.split()
        approx_words = max(50, int(max_tokens / 1.33))
        overlap_words = max(0, int(overlap_tokens / 1.33))
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + approx_words)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(words):
                break
            start = max(start + 1, end - overlap_words)
        return chunks

    chunks: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    for unit in units:
        unit_tokens = estimate_token_count(unit)
        if current_units and current_tokens + unit_tokens > max_tokens:
            chunks.append("\n\n".join(current_units).strip())
            if overlap_tokens > 0:
                overlap: List[str] = []
                overlap_count = 0
                for existing in reversed(current_units):
                    existing_tokens = estimate_token_count(existing)
                    overlap.insert(0, existing)
                    overlap_count += existing_tokens
                    if overlap_count >= overlap_tokens:
                        break
                current_units = overlap
                current_tokens = sum(estimate_token_count(item) for item in current_units)
            else:
                current_units = []
                current_tokens = 0

        current_units.append(unit)
        current_tokens += unit_tokens

    if current_units:
        chunks.append("\n\n".join(current_units).strip())
    return [chunk for chunk in chunks if chunk]


def fixed_window_chunks(
    text: str,
    *,
    source_info: Dict[str, Any],
    target_tokens: int,
    overlap_tokens: int,
    max_chunks_per_document: int,
) -> List[Dict[str, Any]]:
    pieces = split_text_by_budget(text, max_tokens=target_tokens, overlap_tokens=overlap_tokens)
    document_id = stable_document_id(
        source_info.get("path"),
        source_info.get("source_url"),
        source_info.get("source_file"),
        source_info.get("document_title"),
    )
    chunks = []
    for idx, piece in enumerate(pieces[:max_chunks_per_document]):
        chunks.append(
            {
                "document_id": document_id,
                "chunk_index": idx,
                "text": piece,
                "token_count": estimate_token_count(piece),
                "section_path": [],
                "element_types": ["paragraph"],
                "document_title": source_info.get("document_title") or Path(source_info["path"]).stem,
                "document_type": source_info.get("document_type") or "",
                "source_backend": source_info.get("source_backend") or "",
                "source_file": source_info.get("source_file") or "",
                "source_markdown_path": str(source_info["path"]),
                "source_url": source_info.get("source_url") or "",
            }
        )
    return chunks


def hierarchical_markdown_chunks(
    text: str,
    *,
    source_info: Dict[str, Any],
    max_tokens: int,
    max_chunks_per_document: int,
    include_section_headings: bool = True,
) -> List[Dict[str, Any]]:
    blocks = extract_markdown_blocks(text)
    sections = group_blocks_by_section(blocks)
    document_id = stable_document_id(
        source_info.get("path"),
        source_info.get("source_url"),
        source_info.get("source_file"),
        source_info.get("document_title"),
    )

    chunks: List[Dict[str, Any]] = []
    for section in sections:
        section_text = build_section_text(section, include_section_headings=include_section_headings)
        pieces = split_text_by_budget(section_text, max_tokens=max_tokens, overlap_tokens=0)
        for piece in pieces:
            chunks.append(
                {
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "text": piece,
                    "token_count": estimate_token_count(piece),
                    "section_path": list(section.get("section_path") or []),
                    "heading": (section.get("section_path") or [""])[-1] if section.get("section_path") else "",
                    "element_types": sorted(set(section.get("element_types") or ["paragraph"])),
                    "document_title": source_info.get("document_title") or Path(source_info["path"]).stem,
                    "document_type": source_info.get("document_type") or "",
                    "source_backend": source_info.get("source_backend") or "",
                    "source_file": source_info.get("source_file") or "",
                    "source_markdown_path": str(source_info["path"]),
                    "source_url": source_info.get("source_url") or "",
                }
            )
            if len(chunks) >= max_chunks_per_document:
                return chunks
    return chunks


def hybrid_markdown_chunks(
    text: str,
    *,
    source_info: Dict[str, Any],
    target_tokens: int,
    max_tokens: int,
    overlap_tokens: int,
    min_chunk_tokens: int,
    max_chunks_per_document: int,
    include_section_headings: bool = True,
) -> List[Dict[str, Any]]:
    blocks = extract_markdown_blocks(text)
    sections = group_blocks_by_section(blocks)
    document_id = stable_document_id(
        source_info.get("path"),
        source_info.get("source_url"),
        source_info.get("source_file"),
        source_info.get("document_title"),
    )
    chunks: List[Dict[str, Any]] = []
    current_units: List[Dict[str, Any]] = []
    current_tokens = 0

    def emit_current() -> None:
        nonlocal current_units, current_tokens
        if not current_units:
            return
        text_parts = [build_section_text(unit, include_section_headings=include_section_headings) for unit in current_units]
        piece = "\n\n".join(part for part in text_parts if part).strip()
        if not piece:
            current_units = []
            current_tokens = 0
            return
        section_path = list(current_units[-1].get("section_path") or [])
        element_types = sorted({etype for unit in current_units for etype in unit.get("element_types") or []})
        chunks.append(
            {
                "document_id": document_id,
                "chunk_index": len(chunks),
                "text": piece,
                "token_count": estimate_token_count(piece),
                "section_path": section_path,
                "heading": section_path[-1] if section_path else "",
                "element_types": element_types or ["paragraph"],
                "document_title": source_info.get("document_title") or Path(source_info["path"]).stem,
                "document_type": source_info.get("document_type") or "",
                "source_backend": source_info.get("source_backend") or "",
                "source_file": source_info.get("source_file") or "",
                "source_markdown_path": str(source_info["path"]),
                "source_url": source_info.get("source_url") or "",
            }
        )
        current_units = []
        current_tokens = 0

    def can_merge(current_section: List[str], next_section: List[str]) -> bool:
        if not current_section or not next_section:
            return True
        return current_section[:1] == next_section[:1]

    for section in sections:
        section_text = build_section_text(section, include_section_headings=include_section_headings)
        section_tokens = estimate_token_count(section_text)

        if section_tokens > max_tokens:
            emit_current()
            pieces = split_text_by_budget(section_text, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
            for piece in pieces:
                chunks.append(
                    {
                        "document_id": document_id,
                        "chunk_index": len(chunks),
                        "text": piece,
                        "token_count": estimate_token_count(piece),
                        "section_path": list(section.get("section_path") or []),
                        "heading": (section.get("section_path") or [""])[-1] if section.get("section_path") else "",
                        "element_types": sorted(set(section.get("element_types") or ["paragraph"])),
                        "document_title": source_info.get("document_title") or Path(source_info["path"]).stem,
                        "document_type": source_info.get("document_type") or "",
                        "source_backend": source_info.get("source_backend") or "",
                        "source_file": source_info.get("source_file") or "",
                        "source_markdown_path": str(source_info["path"]),
                        "source_url": source_info.get("source_url") or "",
                    }
                )
                if len(chunks) >= max_chunks_per_document:
                    return chunks
            continue

        next_section_path = list(section.get("section_path") or [])
        current_section_path = list(current_units[-1].get("section_path") or []) if current_units else []
        if current_units and (
            current_tokens + section_tokens > target_tokens
            or not can_merge(current_section_path, next_section_path)
        ):
            if current_tokens >= min_chunk_tokens or current_tokens + section_tokens > max_tokens:
                emit_current()

        current_units.append(section)
        current_tokens += section_tokens
        if len(chunks) >= max_chunks_per_document:
            return chunks

    emit_current()
    return chunks[:max_chunks_per_document]


def _walk_nested_values(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key, nested
            yield from _walk_nested_values(nested)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_nested_values(item)


def extract_docling_meta(meta: Any) -> Tuple[List[str], List[int], List[str]]:
    try:
        if hasattr(meta, "export_json_dict"):
            data = meta.export_json_dict()
        elif hasattr(meta, "model_dump"):
            data = meta.model_dump(mode="json")
        elif hasattr(meta, "dict"):
            data = meta.dict()
        else:
            data = {}
    except Exception:
        data = {}

    headings: List[str] = []
    page_numbers: List[int] = []
    element_types: List[str] = []

    for key, value in _walk_nested_values(data):
        key_lower = str(key).lower()
        if key_lower in {"heading", "headings", "title"}:
            if isinstance(value, list):
                headings.extend(str(item).strip() for item in value if str(item).strip())
            elif str(value).strip():
                headings.append(str(value).strip())
        elif key_lower in {"page_no", "page_num", "page_number"}:
            try:
                page_numbers.append(int(value))
            except Exception:
                pass
        elif key_lower in {"label", "doc_item_label", "group_label", "type"} and str(value).strip():
            element_types.append(str(value).strip())

    return headings, sorted(set(page_numbers)), sorted(set(element_types))


def docling_chunks_from_json(
    json_path: Path,
    *,
    strategy: str,
    source_info: Dict[str, Any],
    max_tokens: int,
    overlap_tokens: int,
    max_chunks_per_document: int,
    always_emit_headings: bool = False,
) -> List[Dict[str, Any]]:
    from docling.chunking import HierarchicalChunker, HybridChunker
    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument.load_from_json(json_path)
    if strategy == "hybrid":
        chunker = HybridChunker(always_emit_headings=always_emit_headings)
    else:
        chunker = HierarchicalChunker(always_emit_headings=always_emit_headings)

    document_id = stable_document_id(
        source_info.get("path"),
        source_info.get("source_url"),
        source_info.get("source_file"),
        source_info.get("document_title"),
    )
    chunks: List[Dict[str, Any]] = []

    for base_index, chunk in enumerate(chunker.chunk(doc)):
        text = ""
        try:
            text = chunker.contextualize(chunk=chunk).strip()
        except Exception:
            text = str(getattr(chunk, "text", "") or "").strip()
        if not text:
            continue

        headings, page_numbers, element_types = extract_docling_meta(getattr(chunk, "meta", None))
        pieces = split_text_by_budget(text, max_tokens=max_tokens, overlap_tokens=overlap_tokens if strategy == "hybrid" else 0)
        for piece in pieces:
            section_path = headings or []
            chunks.append(
                {
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "text": piece,
                    "token_count": estimate_token_count(piece),
                    "section_path": section_path,
                    "heading": section_path[-1] if section_path else "",
                    "element_types": element_types or ["docling"],
                    "page_numbers": page_numbers,
                    "document_title": source_info.get("document_title") or Path(source_info["path"]).stem,
                    "document_type": source_info.get("document_type") or "",
                    "source_backend": source_info.get("source_backend") or "docling",
                    "source_file": source_info.get("source_file") or "",
                    "source_markdown_path": str(source_info["path"]),
                    "source_url": source_info.get("source_url") or "",
                }
            )
            if len(chunks) >= max_chunks_per_document:
                return chunks

    return chunks
