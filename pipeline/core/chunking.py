"""
Shared chunking contracts and helpers.

Chunkers emit a stable manifest so alternate strategies can be swapped without
changing the formatter or embedder stages.
"""

from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import load_json_safe


TOKENIZER_ENCODING = "cl100k_base"


class ChunkLimitExceededError(RuntimeError):
    """Raised when an explicit safety limit would otherwise drop content."""


@lru_cache(maxsize=1)
def _budget_tokenizer():
    """Return the deterministic tokenizer used for local chunk budgets.

    Gemini does not expose an offline tokenizer. The pipeline therefore uses
    the same multilingual-safe proxy as the Docling chunker and records the
    method in the chunk manifest. Exact provider counts can be sampled with
    Gemini's countTokens API without making chunking depend on a network call.
    """

    try:
        import tiktoken

        return tiktoken.get_encoding(TOKENIZER_ENCODING)
    except Exception:
        return None


def token_counting_method() -> str:
    """Describe the active local token-budget implementation."""

    if _budget_tokenizer() is not None:
        return f"tiktoken:{TOKENIZER_ENCODING}"
    return "unicode_conservative_fallback:v1"


def _unicode_conservative_token_count(text: str) -> int:
    """Conservative dependency-free fallback for multilingual content.

    ASCII alphanumeric runs are estimated at four characters per token. Each
    punctuation mark counts separately, and non-ASCII characters are charged
    by UTF-8 byte length. This deliberately overestimates Arabic/CJK content
    rather than recreating the former whitespace-based undercount.
    """

    count = 0
    ascii_run_length = 0

    def flush_ascii_run() -> None:
        nonlocal count, ascii_run_length
        if ascii_run_length:
            count += max(1, (ascii_run_length + 3) // 4)
            ascii_run_length = 0

    for character in text:
        if character.isspace():
            flush_ascii_run()
        elif character.isascii() and character.isalnum():
            ascii_run_length += 1
        else:
            flush_ascii_run()
            count += len(character.encode("utf-8")) if not character.isascii() else 1
    flush_ascii_run()
    return count


def estimate_token_count(text: str) -> int:
    """Count local budget tokens consistently for every language/source type."""

    value = str(text or "")
    if not value.strip():
        return 0
    tokenizer = _budget_tokenizer()
    if tokenizer is not None:
        return max(1, len(tokenizer.encode(value, disallowed_special=())))
    return max(1, _unicode_conservative_token_count(value))


def window_text_to_token_budget(
    text: str,
    *,
    max_tokens: int,
    omission_marker: str = "\n\n[...content window omitted...]\n\n",
) -> str:
    """Return explicit head/middle/tail windows within a multilingual token budget.

    This is intended for synopsis-style retrieval records, not lossless chunking.
    Callers must retain the complete source separately. The omission marker makes
    the bounded representation auditable instead of relying on provider-side
    silent truncation.
    """

    value = str(text or "").strip()
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not value or estimate_token_count(value) <= max_tokens:
        return value

    tokenizer = _budget_tokenizer()
    if tokenizer is not None:
        token_ids = tokenizer.encode(value, disallowed_special=())
        marker_ids = tokenizer.encode(omission_marker, disallowed_special=())
        available = max_tokens - 2 * len(marker_ids)
        if available <= 2:
            return tokenizer.decode(token_ids[:max_tokens]).strip()
        while available > 2:
            first_count = max(1, int(available * 0.55))
            middle_count = max(1, int(available * 0.20))
            last_count = max(1, available - first_count - middle_count)
            midpoint = max(0, len(token_ids) // 2 - middle_count // 2)
            bounded_ids = [
                *token_ids[:first_count],
                *marker_ids,
                *token_ids[midpoint : midpoint + middle_count],
                *marker_ids,
                *token_ids[-last_count:],
            ]
            bounded = tokenizer.decode(bounded_ids).strip()
            actual_tokens = estimate_token_count(bounded)
            if actual_tokens <= max_tokens:
                return bounded
            # Decoding a token slice at an arbitrary Unicode boundary can
            # re-encode to a few extra tokens. Leave measured headroom and
            # rebuild all three windows instead of dropping the tail.
            available -= max(4, actual_tokens - max_tokens + 2)
        return tokenizer.decode(token_ids[:max_tokens]).strip()

    def by_character_budget(maximum_chars: int) -> str:
        first = max(1, int(maximum_chars * 0.55))
        middle = max(1, int(maximum_chars * 0.20))
        last = max(1, maximum_chars - first - middle)
        midpoint = max(0, len(value) // 2 - middle // 2)
        return (
            value[:first].rstrip()
            + omission_marker
            + value[midpoint : midpoint + middle].strip()
            + omission_marker
            + value[-last:].lstrip()
        ).strip()

    low, high = 1, len(value)
    best = ""
    while low <= high:
        midpoint = (low + high) // 2
        candidate = by_character_budget(midpoint)
        if estimate_token_count(candidate) <= max_tokens:
            best = candidate
            low = midpoint + 1
        else:
            high = midpoint - 1
    if best:
        return best
    # Extremely small budgets can be shorter than two omission markers.
    prefix = ""
    for character in value:
        candidate = prefix + character
        if estimate_token_count(candidate) > max_tokens:
            break
        prefix = candidate
    return prefix.strip()


def stable_document_id(*parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not raw:
        raw = "document"
    return sha1(raw.encode("utf-8")).hexdigest()[:20]


def stable_chunk_id(document_id: str, strategy: str, chunk_index: int, text: str) -> str:
    digest = sha1(f"{document_id}|{strategy}|{chunk_index}|{text[:256]}".encode("utf-8")).hexdigest()[:12]
    return f"{document_id}::chunk::{chunk_index + 1:03d}:{digest}"


def normalize_chunk_record(item: Dict[str, Any], *, strategy: str, default_index: int = 0) -> Dict[str, Any]:
    record = dict(item)
    text = str(record.get("text") or "").strip()
    if not text:
        raise ValueError("Chunk text cannot be empty")

    document_id = str(record.get("document_id") or "")
    if not document_id:
        document_id = stable_document_id(
            record.get("source_markdown_path"),
            record.get("source_file"),
            record.get("source_url"),
            record.get("document_title"),
        )

    chunk_index = int(record.get("chunk_index", default_index) or default_index)
    section_path = [
        str(value).strip()
        for value in (record.get("section_path") or [])
        if str(value).strip()
    ]
    element_types = sorted(
        {
            str(value).strip()
            for value in (record.get("element_types") or [])
            if str(value).strip()
        }
    )
    page_numbers = sorted(
        {
            int(value)
            for value in (record.get("page_numbers") or [])
            if value not in (None, "")
        }
    )

    normalized = {
        "document_id": document_id,
        "chunk_id": str(record.get("chunk_id") or stable_chunk_id(document_id, strategy, chunk_index, text)),
        "chunk_index": chunk_index,
        "chunk_count": int(record.get("chunk_count") or 0),
        "strategy": str(record.get("strategy") or strategy),
        "text": text,
        "token_count": int(record.get("token_count") or estimate_token_count(text)),
        "section_path": section_path,
        "element_types": element_types,
        "page_numbers": page_numbers,
        "document_title": str(record.get("document_title") or ""),
        "document_type": str(record.get("document_type") or ""),
        "source_backend": str(record.get("source_backend") or ""),
        "source_file": str(record.get("source_file") or ""),
        "source_markdown_path": str(record.get("source_markdown_path") or ""),
        "source_url": str(record.get("source_url") or ""),
    }
    if record.get("heading"):
        normalized["heading"] = str(record["heading"]).strip()
    if record.get("summary_path"):
        normalized["summary_path"] = str(record["summary_path"])
    return normalized


def build_chunk_index(
    chunks: Iterable[Dict[str, Any]],
    *,
    strategy: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    documents: Dict[str, Dict[str, Any]] = {}
    normalized_chunks: List[Dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        normalized = normalize_chunk_record(chunk, strategy=strategy, default_index=idx)
        normalized_chunks.append(normalized)

        document = documents.setdefault(
            normalized["document_id"],
            {
                "document_id": normalized["document_id"],
                "document_title": normalized.get("document_title", ""),
                "document_type": normalized.get("document_type", ""),
                "source_backend": normalized.get("source_backend", ""),
                "source_file": normalized.get("source_file", ""),
                "source_markdown_path": normalized.get("source_markdown_path", ""),
                "source_url": normalized.get("source_url", ""),
                "chunk_ids": [],
                "chunk_count": 0,
                "section_paths": [],
            },
        )
        document["chunk_ids"].append(normalized["chunk_id"])
        document["chunk_count"] += 1
        if normalized["section_path"]:
            document["section_paths"].append(normalized["section_path"])

    chunk_count_by_doc = {
        document_id: int(document["chunk_count"])
        for document_id, document in documents.items()
    }
    for chunk in normalized_chunks:
        chunk["chunk_count"] = chunk_count_by_doc.get(chunk["document_id"], 1)

    return {
        "version": 1,
        "strategy": strategy,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(documents),
        "chunk_count": len(normalized_chunks),
        "documents": list(documents.values()),
        "chunks": normalized_chunks,
        "metadata": dict(metadata or {}),
    }


def load_chunk_index(value: Any) -> Dict[str, Any]:
    if isinstance(value, (str, Path)):
        value = load_json_safe(value, {})
    if not isinstance(value, dict):
        return {"version": 1, "strategy": "", "documents": [], "chunks": [], "metadata": {}}
    chunks = value.get("chunks")
    if not isinstance(chunks, list):
        chunks = []
    strategy = str(value.get("strategy") or "")
    normalized = []
    for idx, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        try:
            normalized.append(normalize_chunk_record(chunk, strategy=strategy or "unknown", default_index=idx))
        except ValueError:
            continue
    documents = value.get("documents")
    if not isinstance(documents, list):
        documents = []
    return {
        "version": int(value.get("version", 1)),
        "strategy": strategy,
        "generated_at": value.get("generated_at"),
        "document_count": int(value.get("document_count", len(documents))),
        "chunk_count": len(normalized),
        "documents": documents,
        "chunks": normalized,
        "metadata": dict(value.get("metadata") or {}),
    }
