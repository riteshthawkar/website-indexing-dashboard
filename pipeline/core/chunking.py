"""
Shared chunking contracts and helpers.

Chunkers emit a stable manifest so alternate strategies can be swapped without
changing the formatter or embedder stages.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .io import load_json_safe


def estimate_token_count(text: str) -> int:
    words = re.findall(r"\S+", text or "")
    if not words:
        return 0
    return max(1, int(len(words) * 1.33))


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
