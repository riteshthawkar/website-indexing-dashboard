"""Internal graph and similarity primitives for evidence sampling."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict
from typing import Any

from ._coherence import content_overlap
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
        "ready to start your new education journey",
        "what are you waiting for",
    )
    if any(marker in normalized for marker in low_signal_markers):
        return False

    source_body = re.search(
        r"\bSOURCE_URL:\s*\S+\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL
    )
    if source_body is not None and len(text_tokens(source_body.group(1))) < 8:
        return False

    # Converted pages can contain chunks that are little more than a breadcrumb
    # followed by a final Markdown heading (for example, ``## Admission
    # criteria``).  They are useful graph nodes, but they do not contain enough
    # evidence to ground a question or reference answer.
    heading = re.search(r"(?:^|\s)#{1,6}\s+", text)
    if heading is not None and len(text_tokens(text[heading.end() :])) < 8:
        return False
    if _mismatched_entity_link(unit):
        return False
    return True


def _mismatched_entity_link(unit: EvidenceUnit) -> bool:
    path = unit.metadata.get("section_path") or ()
    values = (path,) if isinstance(path, str) else tuple(path)
    terminal = str(values[-1]) if values else ""
    words = re.findall(r"[^\W\d_][\w.'’-]*", terminal, flags=re.UNICODE)
    if not 2 <= len(words) <= 4 or not all(word[:1].isupper() for word in words):
        return False
    links = re.findall(r"\]\(([^)]+)\)", unit.text)
    if len(links) != 1:
        return False
    slug = links[0].rstrip("/").rsplit("/", 1)[-1]
    slug_tokens = _name_tokens(slug.replace("-", " "))
    return len(slug_tokens) >= 2 and not (_name_tokens(terminal) & slug_tokens)


def _name_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token.casefold()
        for token in re.findall(r"[^\W\d_][\w.'’-]*", value, flags=re.UNICODE)
        if len(token) >= 2
    )


def content_signature(unit: EvidenceUnit) -> str:
    suffix = unit.evidence_id.rsplit(":", 1)[-1].lower()
    if len(suffix) >= 12 and re.fullmatch(r"[0-9a-f]+", suffix):
        return f"{unit.kind}:{suffix}"
    return " ".join(unit.text.casefold().split())


def reuse_key(unit: EvidenceUnit) -> str:
    """Return a semantic key for records whose IDs may hide duplicate behavior."""
    if unit.kind != "action":
        return ""
    text = " ".join(unit.text.split())
    target = re.search(r"\bTARGET:\s*(\S+)", text, flags=re.IGNORECASE)
    if target:
        return f"action-target:{target.group(1).casefold().rstrip('/')}"
    email = re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
    if email:
        return f"action-target:mailto:{email.group(0).casefold()}"
    url = re.search(r"https?://\S+", text, flags=re.IGNORECASE)
    if url:
        return f"action-target:{url.group(0).casefold().rstrip('/')}"
    return ""


def deduplicated_candidates(
    candidates, selected, sources, *, allow_cross_source_duplicates=False
):
    return [
        item
        for item in candidates
        if all(
            (
                content_signature(item) != content_signature(chosen)
                and content_overlap(item, chosen) < 0.48
                and not _same_text_scope(item, chosen, sources)
            )
            or _distinct_comparison_sources(
                item, chosen, sources, allow_cross_source_duplicates
            )
            for chosen in selected
        )
    ]


def _distinct_comparison_sources(item, chosen, sources, allowed) -> bool:
    if not allowed or item.source_id == chosen.source_id:
        return False
    left = sources.get(item.source_id)
    right = sources.get(chosen.source_id)
    left_title = " ".join(str(getattr(left, "title", "")).casefold().split())
    right_title = " ".join(str(getattr(right, "title", "")).casefold().split())
    return bool(left_title and right_title and left_title != right_title)


def _same_text_scope(item, chosen, sources) -> bool:
    if {item.kind, chosen.kind} - {"chunk", "parent"}:
        return False
    if item.source_id == chosen.source_id and item.kind == chosen.kind:
        return False
    left = sources.get(item.source_id)
    right = sources.get(chosen.source_id)
    left_title = " ".join(str(getattr(left, "title", "")).casefold().split())
    right_title = " ".join(str(getattr(right, "title", "")).casefold().split())
    left_path = tuple(item.metadata.get("section_path") or ())
    right_path = tuple(chosen.metadata.get("section_path") or ())
    return bool(left_title and left_title == right_title and left_path == right_path)


def prompt_unit(unit: EvidenceUnit, source, max_chars: int) -> dict[str, Any]:
    payload = asdict(unit)
    payload["text"] = unit.text[:max_chars]
    payload["source"] = asdict(source) if source is not None else None
    return payload
