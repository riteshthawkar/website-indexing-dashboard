"""
Helpers for semantic knowledge graph extraction and promotion.
"""

from __future__ import annotations

import re
import unicodedata
from hashlib import sha1
from typing import Any, Dict, Iterable, List


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def stable_semantic_id(kind: str, *parts: Any) -> str:
    raw = "|".join(str(part or "").strip() for part in parts if str(part or "").strip())
    if not raw:
        raw = kind
    return f"{kind}:{sha1(raw.encode('utf-8')).hexdigest()[:24]}"


def normalize_entity_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value))
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def entity_merge_key(entity_type: Any, name: Any) -> str:
    entity_type = clean_text(entity_type).lower() or "other"
    label = normalize_entity_label(name).lower()
    label = re.sub(r"_+", " ", label)
    label = re.sub(r"[^\w]+", " ", label, flags=re.UNICODE).strip()
    if not label:
        label = normalize_entity_label(name).lower()
    return f"{entity_type}|{label}"


def unique_strings(values: Iterable[Any]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values or []:
        text = clean_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def coerce_confidence(value: Any, *, default: float = 0.5) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return float(default)
    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def semantic_assertion_text(subject: str, relation_type: str, obj: str, evidence: str = "") -> str:
    parts = [clean_text(subject), clean_text(relation_type), clean_text(obj)]
    base = " ".join(part for part in parts if part).strip()
    evidence = clean_text(evidence)
    if evidence:
        return f"{base}. Evidence: {evidence}"
    return base
