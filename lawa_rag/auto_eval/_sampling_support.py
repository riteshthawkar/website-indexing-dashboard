"""Internal graph and similarity primitives for evidence sampling."""

from __future__ import annotations

import hashlib
import re
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


def usable_for_generation(unit: EvidenceUnit) -> bool:
    text = unit.text.strip()
    if len(text) < 24:
        return False
    if unit.kind in {
        "action", "fact", "media", "image", "video", "table", "database"
    }:
        return True
    normalized = text.casefold()
    low_signal_markers = (
        "section: embedded media",
        "## embedded media",
        "section: you might also be interested in",
        "# you might also be interested in",
    )
    if any(marker in normalized for marker in low_signal_markers):
        return False

    # Converted pages can contain chunks that are little more than a breadcrumb
    # followed by a final Markdown heading (for example, ``## Admission
    # criteria``).  They are useful graph nodes, but they do not contain enough
    # evidence to ground a question or reference answer.
    heading = re.search(r"(?:^|\s)#{1,6}\s+", text)
    if heading is not None and len(text_tokens(text[heading.end() :])) < 8:
        return False
    return True


def content_signature(unit: EvidenceUnit) -> str:
    suffix = unit.evidence_id.rsplit(":", 1)[-1].lower()
    if len(suffix) >= 12 and re.fullmatch(r"[0-9a-f]+", suffix):
        return f"{unit.kind}:{suffix}"
    return " ".join(unit.text.casefold().split())


def deduplicated_candidates(candidates, selected, sources):
    signatures = {content_signature(item) for item in selected}
    distinct = [item for item in candidates if content_signature(item) not in signatures]
    distinct = distinct or list(candidates)
    semantic = [
        item
        for item in distinct
        if max(semantic_overlap(item, chosen, sources) for chosen in selected) < 0.68
    ]
    return semantic or distinct


def prompt_unit(unit: EvidenceUnit, source, max_chars: int) -> dict[str, Any]:
    payload = asdict(unit)
    payload["text"] = unit.text[:max_chars]
    payload["source"] = asdict(source) if source is not None else None
    return payload
