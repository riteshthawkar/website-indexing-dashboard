"""
Document extraction quality assessment helpers.

This module evaluates whether converter output is fit for downstream indexing.
It separates operational success ("a file was written") from semantic success
("the extracted content is usable").
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List


FORMULA_MARKER = "<!-- formula-not-decoded -->"
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
MARKDOWN_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
WORD_RE = re.compile(r"[^\W_]+(?:['/-][^\W_]+)*", re.UNICODE)


def _plain_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _word_count(value: Any) -> int:
    return len(WORD_RE.findall(_plain_text(value)))


def _semantic_media_words(media_items: Iterable[Dict[str, Any]]) -> int:
    total = 0
    for item in media_items or []:
        total += _word_count(item.get("description"))
        total += _word_count(item.get("caption"))
        total += _word_count(item.get("context"))
        total += _word_count(item.get("alt"))
    return total


def _semantic_media_count(media_items: Iterable[Dict[str, Any]], *, min_words: int) -> int:
    count = 0
    for item in media_items or []:
        words = max(
            _word_count(item.get("description")),
            _word_count(item.get("caption")),
            _word_count(item.get("context")),
            _word_count(item.get("alt")),
        )
        if words >= min_words:
            count += 1
    return count


def _normalize_markdown(markdown: str) -> str:
    text = markdown or ""
    text = MARKDOWN_IMAGE_RE.sub(" ", text)
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = MARKDOWN_HTML_COMMENT_RE.sub(" ", text)
    return text


def _looks_suspicious_latin_token(token: str) -> bool:
    token = token.strip(".,;:!?()[]{}<>\"'`*_")
    if len(token) < 28 or not token.isalpha():
        return False
    if any(ord(ch) > 127 for ch in token):
        return False
    if len(token) >= 40:
        return True
    uppercase = sum(1 for ch in token if ch.isupper())
    lowercase = sum(1 for ch in token if ch.islower())
    transitions = sum(
        1
        for idx in range(1, len(token))
        if token[idx].isupper() != token[idx - 1].isupper()
    )
    return uppercase >= 2 and lowercase >= 6 and transitions >= 2


def _suspicious_token_length(token: str) -> int:
    token = token.strip(".,;:!?()[]{}<>\"'`*_")
    return len(token) if _looks_suspicious_latin_token(token) else 0


def assess_markdown_document(
    markdown: str,
    *,
    source_ext: str,
    config: Dict[str, Any],
    media_items: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Assess whether extracted markdown is acceptable for indexing.

    Returns a dict with:
      - accepted: bool
      - score: float in [0, 1]
      - reasons: list[str]
      - warnings: list[str]
      - metrics: dict
    """

    media_items = list(media_items or [])
    normalized = _normalize_markdown(markdown)
    lines = [line.rstrip() for line in normalized.splitlines()]
    nonempty_lines = [line for line in lines if line.strip()]
    words = WORD_RE.findall(normalized)
    suspicious_tokens = [token for token in re.split(r"\s+", normalized) if _looks_suspicious_latin_token(token)]
    max_suspicious_token_length = max((_suspicious_token_length(token) for token in suspicious_tokens), default=0)
    formula_marker_count = markdown.count(FORMULA_MARKER)
    replacement_char_count = markdown.count("\ufffd")
    semantic_media_word_count = _semantic_media_words(media_items)
    semantic_media_item_count = _semantic_media_count(
        media_items,
        min_words=int(config.get("validation_min_semantic_media_item_words", 6)),
    )
    single_word_lines = sum(1 for line in nonempty_lines if len(WORD_RE.findall(line)) == 1)

    informative_words = len(words)
    char_count = len(normalized)
    line_count = len(nonempty_lines)
    suspicious_count = len(suspicious_tokens)
    suspicious_ratio = suspicious_count / max(1, informative_words)
    replacement_ratio = replacement_char_count / max(1, char_count)
    single_word_line_ratio = single_word_lines / max(1, line_count)

    is_pdf = source_ext.lower() == ".pdf"
    min_words = int(
        config.get(
            "validation_min_informative_words" if is_pdf else "validation_min_informative_words_office",
            40 if is_pdf else 6,
        )
    )
    min_chars = int(
        config.get(
            "validation_min_text_chars" if is_pdf else "validation_min_text_chars_office",
            180 if is_pdf else 20,
        )
    )
    max_suspicious_tokens = int(config.get("validation_max_suspicious_tokens", 4 if is_pdf else 6))
    max_suspicious_ratio = float(config.get("validation_max_suspicious_token_ratio", 0.025 if is_pdf else 0.04))
    critical_suspicious_token_length = int(config.get("validation_critical_suspicious_token_length", 40))
    max_replacement_ratio = float(config.get("validation_max_replacement_char_ratio", 0.002))
    formula_warn_count = int(config.get("validation_formula_warn_count", 3))
    formula_fail_count = int(config.get("validation_formula_fail_count", 12))
    formula_fail_word_floor = int(config.get("validation_formula_fail_word_floor", 200))
    max_single_word_line_ratio = float(config.get("validation_max_single_word_line_ratio", 0.58))
    min_semantic_media_items = int(config.get("validation_min_semantic_media_items", 1))
    min_semantic_media_words = int(config.get("validation_min_semantic_media_words", 10))

    reasons: List[str] = []
    warnings: List[str] = []

    if informative_words < min_words or char_count < min_chars:
        if (
            semantic_media_item_count >= min_semantic_media_items
            and semantic_media_word_count >= min_semantic_media_words
        ):
            warnings.append("media_heavy_low_text")
        else:
            reasons.append("too_short")

    if max_suspicious_token_length >= critical_suspicious_token_length:
        reasons.append("collapsed_spacing")
    elif suspicious_count >= max_suspicious_tokens and suspicious_ratio > max_suspicious_ratio:
        reasons.append("collapsed_spacing")
    elif suspicious_count > 0:
        warnings.append("collapsed_spacing_warning")

    if replacement_ratio > max_replacement_ratio:
        reasons.append("decoding_noise")

    if formula_marker_count >= formula_fail_count and informative_words < formula_fail_word_floor:
        reasons.append("formula_loss")
    elif formula_marker_count >= formula_warn_count:
        warnings.append("formula_loss_warning")

    if line_count >= 30 and single_word_line_ratio > max_single_word_line_ratio:
        warnings.append("line_fragmentation_warning")

    score = 1.0
    if "too_short" in reasons:
        score -= 0.35
    if "collapsed_spacing" in reasons:
        score -= min(0.45, suspicious_ratio * 10.0)
    elif "collapsed_spacing_warning" in warnings:
        score -= min(0.15, suspicious_ratio * 5.0)
    if "decoding_noise" in reasons:
        score -= min(0.25, replacement_ratio * 40.0)
    if "formula_loss" in reasons:
        score -= 0.2
    elif "formula_loss_warning" in warnings:
        score -= min(0.1, formula_marker_count * 0.01)
    if "line_fragmentation_warning" in warnings:
        score -= min(0.1, max(0.0, single_word_line_ratio - max_single_word_line_ratio) * 0.4)
    if "media_heavy_low_text" in warnings:
        score += 0.05
    score = max(0.0, min(1.0, score))

    return {
        "accepted": not reasons,
        "score": round(score, 4),
        "reasons": reasons,
        "warnings": warnings,
        "metrics": {
            "source_ext": source_ext.lower(),
            "char_count": char_count,
            "informative_words": informative_words,
            "line_count": line_count,
            "image_ref_count": len(MARKDOWN_IMAGE_RE.findall(markdown or "")),
            "formula_marker_count": formula_marker_count,
            "replacement_char_count": replacement_char_count,
            "replacement_char_ratio": round(replacement_ratio, 6),
            "suspicious_token_count": suspicious_count,
            "suspicious_token_ratio": round(suspicious_ratio, 6),
            "max_suspicious_token_length": max_suspicious_token_length,
            "single_word_line_ratio": round(single_word_line_ratio, 6),
            "semantic_media_item_count": semantic_media_item_count,
            "semantic_media_word_count": semantic_media_word_count,
        },
    }


def prefer_fallback_candidate(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    *,
    margin: float,
) -> bool:
    """Return True when the secondary candidate should replace the primary."""
    if secondary.get("accepted") and not primary.get("accepted"):
        return True
    if not secondary.get("accepted"):
        return False
    if not primary.get("accepted"):
        return True
    return float(secondary.get("score", 0.0)) > float(primary.get("score", 0.0)) + margin
