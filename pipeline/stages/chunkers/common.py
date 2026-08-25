"""
Shared helpers for pluggable chunker stages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pipeline.core.base import StageContext
from pipeline.core.chunking import (
    TOKENIZER_ENCODING,
    ChunkLimitExceededError,
    build_chunk_index,
    estimate_token_count,
    stable_document_id,
)
from pipeline.core.io import atomic_write_json, ensure_dir, load_json_safe


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
LIST_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-+:?\s*\|)+\s*$")
SENTENCE_RE = re.compile(r"(?<=[.!?\u061f\u06d4\u3002\uff01\uff1f])\s+")


def _enforce_chunk_safety_limit(
    chunk_count: int,
    *,
    max_chunks_per_document: int,
    source_info: Dict[str, Any],
) -> None:
    """Fail loudly instead of returning a partial document.

    A non-positive limit means unlimited. Positive values are safety guards,
    not truncation instructions.
    """

    if max_chunks_per_document <= 0 or chunk_count <= max_chunks_per_document:
        return
    source = (
        source_info.get("source_url")
        or source_info.get("path")
        or source_info.get("source_file")
        or source_info.get("document_title")
        or "unknown document"
    )
    raise ChunkLimitExceededError(
        f"Chunk safety limit exceeded for {source}: generated at least "
        f"{chunk_count} chunks, configured maximum is {max_chunks_per_document}. "
        "No partial chunks were returned. Set max_chunks_per_document to 0 for "
        "lossless unlimited chunking or raise the explicit safety limit."
    )


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _path_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return str(Path(text).resolve())
    except Exception:
        return text


def _document_stem_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    stem = Path(text).stem
    stem = re.sub(r"\.pages_\d+_\d+$", "", stem)
    return stem.casefold()


def _source_markdown_path(source_info: Dict[str, Any]) -> str:
    """Return the immutable Markdown identity, not a stage working-copy path.

    Corpus merge stages materialize files below the current run so downstream
    stages can work without mutating their audited source run.  The merge keeps
    the original artifact path in ``corpus_source_local_path``.  Publishing the
    materialized path as chunk provenance breaks joins to Representation V2,
    especially for file-only PDF documents that have no URL fallback.
    """

    return str(source_info.get("source_markdown_path") or source_info.get("path") or "")


def _load_download_source_lookup(ctx: StageContext) -> Dict[str, str]:
    mapping_file = ctx.work_dir / "mappings.json"
    payload = load_json_safe(mapping_file, {}) if mapping_file.exists() else {}
    if not isinstance(payload, dict):
        return {}
    lookup: Dict[str, str] = {}
    for source_url, local_path in payload.items():
        source = _clean_text(source_url)
        if not source:
            continue
        for key in (
            _path_key(local_path),
            _document_stem_key(local_path),
            Path(str(local_path or "")).name.casefold(),
        ):
            if key:
                lookup.setdefault(key, source)
    return lookup


def _resolve_source_url(
    source_url: Any,
    *,
    source_file: Any = "",
    source_markdown_path: Any = "",
    document_title: Any = "",
    download_source_lookup: Dict[str, str] | None = None,
) -> str:
    existing = _clean_text(source_url)
    if existing:
        return existing
    lookup = download_source_lookup or {}
    for key in (
        _path_key(source_file),
        _path_key(source_markdown_path),
        _document_stem_key(source_file),
        _document_stem_key(source_markdown_path),
        _document_stem_key(document_title),
        Path(str(source_file or "")).name.casefold(),
        Path(str(source_markdown_path or "")).name.casefold(),
    ):
        if key and key in lookup:
            return lookup[key]
    return ""


def collect_markdown_sources(ctx: StageContext) -> List[Dict[str, Any]]:
    markdown_artifacts = ctx.find_artifacts(artifact_type="markdown")
    sources: List[Dict[str, Any]] = []
    download_source_lookup = _load_download_source_lookup(ctx)

    if markdown_artifacts:
        for record in markdown_artifacts:
            if not record.local_path or not Path(record.local_path).is_file():
                continue
            metadata = dict(record.metadata or {})
            path = Path(record.local_path).resolve()
            source_markdown_path = str(
                metadata.get("corpus_source_local_path") or path
            )
            source_file = str(metadata.get("source_file") or "")
            source_url = _resolve_source_url(
                metadata.get("source_url"),
                source_file=source_file,
                source_markdown_path=path,
                document_title=metadata.get("document_title") or path.stem,
                download_source_lookup=download_source_lookup,
            )
            sources.append(
                {
                    "path": path,
                    "source_markdown_path": source_markdown_path,
                    "artifact_id": record.artifact_id,
                    "metadata": metadata,
                    "source_url": source_url,
                    "source_backend": str(metadata.get("backend") or ""),
                    "document_title": metadata.get("document_title") or path.stem,
                    "document_type": str(metadata.get("source_type") or ""),
                    "source_file": source_file,
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
        source_url = _resolve_source_url(
            md_to_url.get(str(resolved), ""),
            source_markdown_path=resolved,
            document_title=resolved.stem,
            download_source_lookup=download_source_lookup,
        )
        sources.append(
            {
                "path": resolved,
                "source_markdown_path": str(resolved),
                "artifact_id": None,
                "metadata": {},
                "source_url": source_url,
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
    section_headings: Dict[int, str] = {}

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
            # Track actual Markdown levels rather than indexing into a compact
            # path. This keeps consecutive H2/H3 headings as siblings even
            # when the source omits an H1 or skips a heading level.
            section_headings = {
                existing_level: existing_heading
                for existing_level, existing_heading in section_headings.items()
                if existing_level < level
            }
            section_headings[level] = heading
            section_path = [
                section_headings[heading_level]
                for heading_level in sorted(section_headings)
            ]
            blocks.append(
                {
                    "text": line.strip(),
                    "element_type": "heading",
                    "section_path": list(section_path),
                    "heading": heading,
                    "heading_level": level,
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
    pending_heading_sections: List[Dict[str, Any]] = []

    for section in sections:
        has_non_heading_content = any(
            (block.get("element_type") or "paragraph") != "heading"
            for block in section["blocks"]
        )
        if not has_non_heading_content:
            pending_heading_sections.append(section)
            continue

        if pending_heading_sections:
            section_path = list(section.get("section_path") or [])
            ancestor_heading_blocks: List[Dict[str, Any]] = []
            for pending in pending_heading_sections:
                pending_path = list(pending.get("section_path") or [])
                is_ancestor = (
                    bool(pending_path)
                    and len(pending_path) < len(section_path)
                    and section_path[: len(pending_path)] == pending_path
                )
                if is_ancestor:
                    ancestor_heading_blocks.extend(pending["blocks"])
                else:
                    # A heading-only sibling/top-level section is real source
                    # content. Preserve it separately instead of relabeling it
                    # as part of the following section.
                    normalized.append(pending)
            section["blocks"] = [*ancestor_heading_blocks, *section["blocks"]]
            section["element_types"] = [
                *(
                    block.get("element_type") or "paragraph"
                    for block in ancestor_heading_blocks
                ),
                *section["element_types"],
            ]
            pending_heading_sections = []
        normalized.append(section)

    # Preserve trailing headings as their own sections; attaching them to the
    # previous section would corrupt both the text and section metadata.
    normalized.extend(pending_heading_sections)

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


def _split_single_lexeme_by_budget(value: str, *, max_tokens: int) -> List[str]:
    """Split a URL or other no-whitespace value without exceeding the budget."""

    pieces: List[str] = []
    start = 0
    while start < len(value):
        low = start + 1
        high = len(value)
        best_end = start
        while low <= high:
            midpoint = (low + high) // 2
            if estimate_token_count(value[start:midpoint]) <= max_tokens:
                best_end = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best_end <= start:
            raise ValueError(
                f"Token budget {max_tokens} is too small to encode one source character"
            )
        pieces.append(value[start:best_end])
        start = best_end
    return pieces


def _split_words_by_budget(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """Token-aware fallback for prose with no usable paragraph boundaries."""

    words = re.findall(r"\S+", text)
    chunks: List[str] = []
    start = 0
    while start < len(words):
        low = start + 1
        high = len(words)
        best_end = start
        while low <= high:
            midpoint = (low + high) // 2
            candidate = " ".join(words[start:midpoint])
            if estimate_token_count(candidate) <= max_tokens:
                best_end = midpoint
                low = midpoint + 1
            else:
                high = midpoint - 1

        if best_end == start:
            chunks.extend(
                _split_single_lexeme_by_budget(
                    words[start],
                    max_tokens=max_tokens,
                )
            )
            start += 1
            continue

        chunks.append(" ".join(words[start:best_end]))
        if best_end >= len(words):
            break

        next_start = best_end
        if overlap_tokens > 0:
            low = start + 1  # Always make forward progress.
            high = best_end
            best_overlap_start = best_end
            while low <= high:
                midpoint = (low + high) // 2
                overlap_text = " ".join(words[midpoint:best_end])
                if estimate_token_count(overlap_text) <= overlap_tokens:
                    best_overlap_start = midpoint
                    high = midpoint - 1
                else:
                    low = midpoint + 1
            next_start = best_overlap_start
        start = max(start + 1, next_start)
    return chunks


def split_text_by_budget(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> List[str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero")
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if estimate_token_count(cleaned) <= max_tokens:
        return [cleaned]

    units = [unit.strip() for unit in re.split(r"\n{2,}", cleaned) if unit.strip()]
    if len(units) <= 1:
        units = [unit.strip() for unit in SENTENCE_RE.split(cleaned) if unit.strip()]
    if len(units) <= 1:
        return _split_words_by_budget(
            cleaned,
            max_tokens=max_tokens,
            overlap_tokens=max(0, min(overlap_tokens, max_tokens - 1)),
        )

    chunks: List[str] = []
    current_units: List[str] = []
    current_tokens = 0

    def estimate_units(candidate_units: List[str]) -> int:
        return estimate_token_count("\n\n".join(candidate_units).strip())

    for unit in units:
        unit_tokens = estimate_token_count(unit)
        if unit_tokens > max_tokens:
            if current_units:
                chunks.append("\n\n".join(current_units).strip())
                current_units = []
                current_tokens = 0
            chunks.extend(
                split_text_by_budget(
                    unit,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                )
            )
            continue
        prospective_tokens = estimate_units(current_units + [unit]) if current_units else unit_tokens
        if current_units and prospective_tokens > max_tokens:
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
                current_tokens = estimate_units(current_units) if current_units else 0
                if current_units and estimate_units(current_units + [unit]) > max_tokens:
                    while current_units and estimate_units(current_units + [unit]) > max_tokens:
                        current_units.pop(0)
                    current_tokens = estimate_units(current_units) if current_units else 0
            else:
                current_units = []
                current_tokens = 0

        current_units.append(unit)
        current_tokens = estimate_units(current_units)

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
        _source_markdown_path(source_info),
        source_info.get("source_url"),
        source_info.get("source_file"),
        source_info.get("document_title"),
    )
    chunks = []
    for idx, piece in enumerate(pieces):
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
                "source_markdown_path": _source_markdown_path(source_info),
                "source_url": source_info.get("source_url") or "",
            }
        )
        _enforce_chunk_safety_limit(
            len(chunks),
            max_chunks_per_document=max_chunks_per_document,
            source_info=source_info,
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
        _source_markdown_path(source_info),
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
                    "source_markdown_path": _source_markdown_path(source_info),
                    "source_url": source_info.get("source_url") or "",
                }
            )
            _enforce_chunk_safety_limit(
                len(chunks),
                max_chunks_per_document=max_chunks_per_document,
                source_info=source_info,
            )
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
        _source_markdown_path(source_info),
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
        section_path = list(current_units[0].get("section_path") or [])
        element_types = sorted({etype for unit in current_units for etype in unit.get("element_types") or []})
        pieces = split_text_by_budget(piece, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        for emitted_piece in pieces:
            chunks.append(
                {
                    "document_id": document_id,
                    "chunk_index": len(chunks),
                    "text": emitted_piece,
                    "token_count": estimate_token_count(emitted_piece),
                    "section_path": section_path,
                    "heading": section_path[-1] if section_path else "",
                    "element_types": element_types or ["paragraph"],
                    "document_title": source_info.get("document_title") or Path(source_info["path"]).stem,
                    "document_type": source_info.get("document_type") or "",
                    "source_backend": source_info.get("source_backend") or "",
                    "source_file": source_info.get("source_file") or "",
                    "source_markdown_path": _source_markdown_path(source_info),
                    "source_url": source_info.get("source_url") or "",
                }
            )
            _enforce_chunk_safety_limit(
                len(chunks),
                max_chunks_per_document=max_chunks_per_document,
                source_info=source_info,
            )
        current_units = []
        current_tokens = 0

    def can_merge(current_section: List[str], next_section: List[str]) -> bool:
        return current_section == next_section

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
                        "source_markdown_path": _source_markdown_path(source_info),
                        "source_url": source_info.get("source_url") or "",
                    }
                )
                _enforce_chunk_safety_limit(
                    len(chunks),
                    max_chunks_per_document=max_chunks_per_document,
                    source_info=source_info,
                )
            continue

        next_section_path = list(section.get("section_path") or [])
        current_section_path = list(current_units[-1].get("section_path") or []) if current_units else []
        if current_units:
            crosses_semantic_section = not can_merge(
                current_section_path,
                next_section_path,
            )
            exceeds_target = current_tokens + section_tokens > target_tokens
            if crosses_semantic_section:
                # Every heading-defined section is a hard boundary, even when
                # the preceding section is smaller than min_chunk_tokens.
                emit_current()
            elif exceeds_target and (
                current_tokens >= min_chunk_tokens
                or current_tokens + section_tokens > max_tokens
            ):
                emit_current()

        current_units.append(section)
        current_tokens += section_tokens

    emit_current()
    return chunks


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


def _build_docling_hybrid_chunker(max_tokens: int, *, always_emit_headings: bool):
    from docling.chunking import HybridChunker

    try:
        import tiktoken
        from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer
        from pydantic import ConfigDict

        class TiktokenBudgetTokenizer(BaseTokenizer):
            model_config = ConfigDict(arbitrary_types_allowed=True)

            tokenizer: Any
            max_tokens: int

            def count_tokens(self, text: str) -> int:
                return len(self.tokenizer.encode(text=text or "", disallowed_special=()))

            def get_max_tokens(self) -> int:
                return self.max_tokens

            def get_tokenizer(self) -> Any:
                return self.tokenizer

        tokenizer = TiktokenBudgetTokenizer(
            tokenizer=tiktoken.get_encoding(TOKENIZER_ENCODING),
            max_tokens=max_tokens,
        )
        return HybridChunker(tokenizer=tokenizer, always_emit_headings=always_emit_headings)
    except Exception:
        return HybridChunker(always_emit_headings=always_emit_headings)


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
    window_index = load_json_safe(json_path, {})
    if isinstance(window_index, dict) and window_index.get("schema") == "mbzuai_docling_page_windows.v1":
        chunks: List[Dict[str, Any]] = []
        for part in window_index.get("parts") or []:
            structured_path = str(part.get("structured_document_path") or "").strip()
            if not structured_path:
                continue
            part_path = Path(structured_path)
            if not part_path.is_absolute():
                part_path = (json_path.parent / part_path).resolve()
            if not part_path.is_file():
                continue
            part_chunks = docling_chunks_from_json(
                part_path,
                strategy=strategy,
                source_info=source_info,
                max_tokens=max_tokens,
                overlap_tokens=overlap_tokens,
                # Enforce the limit over the complete source document below,
                # not independently inside each page-window part.
                max_chunks_per_document=0,
                always_emit_headings=always_emit_headings,
            )
            page_range = part.get("page_range") or []
            for chunk in part_chunks:
                chunk["chunk_index"] = len(chunks)
                if page_range and not chunk.get("page_numbers"):
                    try:
                        start_page, end_page = int(page_range[0]), int(page_range[1])
                        chunk["page_numbers"] = list(range(start_page, end_page + 1))
                    except Exception:
                        pass
                chunks.append(chunk)
                _enforce_chunk_safety_limit(
                    len(chunks),
                    max_chunks_per_document=max_chunks_per_document,
                    source_info=source_info,
                )
        return chunks

    from docling.chunking import HierarchicalChunker, HybridChunker
    from docling_core.types.doc import DoclingDocument

    doc = DoclingDocument.load_from_json(json_path)
    if strategy == "hybrid":
        chunker = _build_docling_hybrid_chunker(max_tokens, always_emit_headings=always_emit_headings)
    else:
        chunker = HierarchicalChunker(always_emit_headings=always_emit_headings)

    document_id = stable_document_id(
        _source_markdown_path(source_info),
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
                    "source_markdown_path": _source_markdown_path(source_info),
                    "source_url": source_info.get("source_url") or "",
                }
            )
            _enforce_chunk_safety_limit(
                len(chunks),
                max_chunks_per_document=max_chunks_per_document,
                source_info=source_info,
            )

    return chunks
