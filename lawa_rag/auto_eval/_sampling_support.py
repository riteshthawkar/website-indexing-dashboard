"""Internal graph and similarity primitives for evidence sampling."""

from __future__ import annotations

import hashlib
from dataclasses import asdict
from typing import Any

from ._util import text_tokens
from .corpus import CorpusSnapshot, EvidenceUnit


def stable_number(value: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def adjacency(snapshot: CorpusSnapshot) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    for relationship in snapshot.relationships:
        output.setdefault(relationship.source_id, set()).add(relationship.target_id)
        output.setdefault(relationship.target_id, set()).add(relationship.source_id)
    return output


def semantic_overlap(unit: EvidenceUnit, other: EvidenceUnit, sources) -> float:
    left_source = sources.get(unit.source_id)
    right_source = sources.get(other.source_id)
    left = text_tokens(unit.text[:500], unit.locator, getattr(left_source, "title", ""))
    right = text_tokens(other.text[:500], other.locator, getattr(right_source, "title", ""))
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def modality(kind: str) -> str:
    if kind in {"media", "image", "video"}:
        return "visual"
    if kind in {"table", "database"}:
        return "structured"
    return "text"


def prompt_unit(unit: EvidenceUnit, source, max_chars: int) -> dict[str, Any]:
    payload = asdict(unit)
    payload["text"] = unit.text[:max_chars]
    payload["source"] = asdict(source) if source is not None else None
    return payload
