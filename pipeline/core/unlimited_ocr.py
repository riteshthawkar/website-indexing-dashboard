"""Small, dependency-free helpers for Unlimited-OCR layout output.

Unlimited-OCR emits layout markers such as::

    <|det|>image [42, 80, 958, 620]<|/det|>

Coordinates use the model's normalized 0-1000 page space.  The model's
``crop_mode`` is an inference-time tiling strategy; it does not materialize
standalone figure files.  These helpers make the emitted image boxes usable by
the ingestion pipeline without coupling it to a particular GPU runtime.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


_DETECTION_RE = re.compile(
    r"<\|det\|>\s*([^\s\[]+)\s*"
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    r"\s*<\|/det\|>\s*(.*)$",
    re.IGNORECASE,
)
_INLINE_DETECTION_RE = re.compile(
    r"^<\|det\|>\s*([^<\s]+)(?:\s*\[[^\]]*\])?\s*<\|/det\|>\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|>]+\|>")
_NON_TEXT_SENTINELS = {
    "[non-text]",
    "[non text]",
    "non-text",
    "non text",
}


def _clamp(value: float, lower: float = 0.0, upper: float = 1000.0) -> float:
    return max(lower, min(upper, float(value)))


def parse_layout_detections(raw_output: str) -> List[Dict[str, Any]]:
    """Parse layout-grounded Unlimited-OCR output.

    ``<PAGE>`` increments the one-based page number.  Invalid or empty boxes
    are ignored so downstream crop code can fail closed.
    """

    detections: List[Dict[str, Any]] = []
    page_number = 1
    saw_page_marker = False
    for raw_line in str(raw_output or "").splitlines():
        line = raw_line.strip()
        while line.startswith("<PAGE>"):
            if saw_page_marker:
                page_number += 1
            else:
                saw_page_marker = True
            line = line[len("<PAGE>") :].lstrip()
        if not line:
            continue
        match = _DETECTION_RE.match(line)
        if not match:
            continue
        category = match.group(1).strip().lower()
        left, top, right, bottom = (_clamp(float(match.group(index))) for index in range(2, 6))
        left, right = sorted((left, right))
        top, bottom = sorted((top, bottom))
        if right <= left or bottom <= top:
            continue
        detections.append(
            {
                "category": category,
                "bbox_1000": [left, top, right, bottom],
                "page_number": page_number,
                "text": match.group(6).strip(),
            }
        )
    return detections


def image_regions(raw_output: str) -> List[Dict[str, Any]]:
    """Return only usable image/figure detections from model output."""

    accepted = {"image", "figure", "chart", "diagram", "photo"}
    return [
        detection
        for detection in parse_layout_detections(raw_output)
        if detection.get("category") in accepted
    ]


def normalized_bbox_to_pixels(
    bbox_1000: Sequence[float],
    *,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Convert a normalized model box to a clipped Pillow-style pixel box."""

    if len(bbox_1000) != 4:
        raise ValueError("bbox_1000 must contain exactly four coordinates")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    left, top, right, bottom = (_clamp(float(value)) for value in bbox_1000)
    left, right = sorted((left, right))
    top, bottom = sorted((top, bottom))
    pixel_box = (
        max(0, min(width, round(left * width / 1000.0))),
        max(0, min(height, round(top * height / 1000.0))),
        max(0, min(width, round(right * width / 1000.0))),
        max(0, min(height, round(bottom * height / 1000.0))),
    )
    if pixel_box[2] <= pixel_box[0] or pixel_box[3] <= pixel_box[1]:
        raise ValueError("bbox_1000 resolves to an empty pixel box")
    return pixel_box


def filter_regions(
    detections: Iterable[Dict[str, Any]],
    *,
    minimum_area_ratio: float = 0.003,
    maximum_area_ratio: float = 0.90,
) -> List[Dict[str, Any]]:
    """Filter normalized regions by page coverage before rasterization."""

    output: List[Dict[str, Any]] = []
    for detection in detections:
        bbox = detection.get("bbox_1000") or []
        if len(bbox) != 4:
            continue
        area_ratio = max(0.0, float(bbox[2]) - float(bbox[0])) * max(
            0.0, float(bbox[3]) - float(bbox[1])
        ) / 1_000_000.0
        if minimum_area_ratio <= area_ratio <= maximum_area_ratio:
            output.append({**detection, "area_ratio": area_ratio})
    return output


def strip_layout_markers(raw_output: str) -> str:
    """Convert model layout output into conservative exact OCR text.

    Image-region markers and Markdown image references are deliberately
    omitted: they describe layout, not text visible in the source image.
    Unlike a captioning model, this function never invents replacement text.
    """

    blocks: List[List[str]] = []
    current: List[str] = []
    for raw_line in str(raw_output or "").replace("<PAGE>", "\n").splitlines():
        line = raw_line.strip()
        if not line:
            if current:
                blocks.append(current)
                current = []
            continue
        match = _INLINE_DETECTION_RE.match(line)
        if match:
            category = match.group(1).strip().lower()
            content = match.group(2).strip()
            if current:
                blocks.append(current)
                current = []
            if category in {"image", "figure", "photo", "chart", "diagram"}:
                continue
            line = content
        line = _MARKDOWN_IMAGE_RE.sub(" ", line)
        line = _SPECIAL_TOKEN_RE.sub(" ", line)
        line = " ".join(line.split()).strip()
        if not line or line.casefold() in _NON_TEXT_SENTINELS:
            continue
        current.append(line)
    if current:
        blocks.append(current)
    return "\n\n".join("\n".join(block) for block in blocks if block).strip()


def assess_ocr_quality(
    raw_output: str,
    *,
    minimum_alphanumeric_chars: int = 2,
    maximum_single_character_token_ratio: float = 0.62,
    minimum_unique_token_ratio: float = 0.14,
    maximum_repeated_line_ratio: float = 0.55,
    minimum_quality_score: float = 0.55,
) -> Dict[str, Any]:
    """Quality-gate exact OCR without fabricating a model confidence value."""

    text = strip_layout_markers(raw_output)
    alphanumeric_chars = sum(1 for character in text if character.isalnum())
    tokens = re.findall(r"[^\W_]+", text.casefold(), flags=re.UNICODE)
    token_count = len(tokens)
    unique_token_ratio = len(set(tokens)) / max(1, token_count)
    single_character_ratio = sum(len(token) == 1 for token in tokens) / max(1, token_count)
    lines = [line.strip().casefold() for line in text.splitlines() if line.strip()]
    repeated_line_count = sum(count - 1 for count in Counter(lines).values() if count > 1)
    repeated_line_ratio = repeated_line_count / max(1, len(lines))

    flags: List[str] = []
    if alphanumeric_chars < max(0, int(minimum_alphanumeric_chars)):
        flags.append("no_readable_text")
    if token_count >= 12 and single_character_ratio > maximum_single_character_token_ratio:
        flags.append("excessive_single_character_tokens")
    if token_count >= 24 and unique_token_ratio < minimum_unique_token_ratio:
        flags.append("excessive_token_repetition")
    if len(lines) >= 6 and repeated_line_ratio > maximum_repeated_line_ratio:
        flags.append("excessive_line_repetition")
    if _SPECIAL_TOKEN_RE.search(text):
        flags.append("unremoved_special_tokens")

    score = 1.0
    if "no_readable_text" in flags:
        score = 0.0
    else:
        score -= min(0.45, max(0.0, single_character_ratio - 0.15))
        score -= min(0.35, max(0.0, 0.45 - unique_token_ratio))
        score -= min(0.35, repeated_line_ratio)
        score = max(0.0, min(1.0, score))
    if (
        "no_readable_text" not in flags
        and score < min(1.0, max(0.0, float(minimum_quality_score)))
    ):
        flags.append("quality_score_below_threshold")

    if "no_readable_text" in flags:
        status = "no_readable_text"
    elif flags:
        status = "rejected_low_quality"
    else:
        status = "completed"
    return {
        "status": status,
        "text": text if status == "completed" else "",
        "candidate_text": text,
        # This is a deterministic heuristic score, not model confidence.
        "quality_score": round(score, 6),
        "quality_flags": flags,
        "quality_metrics": {
            "character_count": len(text),
            "alphanumeric_character_count": alphanumeric_chars,
            "token_count": token_count,
            "unique_token_ratio": round(unique_token_ratio, 6),
            "single_character_token_ratio": round(single_character_ratio, 6),
            "repeated_line_ratio": round(repeated_line_ratio, 6),
        },
    }


def _walk_gradio_payload(value: Any) -> Iterable[Tuple[str, bool]]:
    if isinstance(value, Mapping):
        text = value.get("text")
        if isinstance(text, str):
            yield text, bool(value.get("done", False))
        for key, nested in value.items():
            if key != "text":
                yield from _walk_gradio_payload(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _walk_gradio_payload(nested)


def extract_gradio_final_output(raw_sse: str) -> str:
    """Return the last cumulative OCR snapshot from a Gradio SSE response."""

    latest = ""
    latest_done = ""
    for raw_line in str(raw_sse or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue
        for text, done in _walk_gradio_payload(payload):
            latest = text
            if done:
                latest_done = text
    return latest_done or latest


def extract_openai_stream_output(raw_sse: str) -> str:
    """Concatenate OpenAI-compatible streaming deltas from SGLang/vLLM."""

    chunks: List[str] = []
    for raw_line in str(raw_sse or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        payload_text = line[5:].strip()
        if not payload_text or payload_text == "[DONE]":
            continue
        try:
            payload = json.loads(payload_text)
            choices = payload.get("choices") or []
            delta = choices[0].get("delta") if choices and isinstance(choices[0], dict) else {}
            content = delta.get("content") if isinstance(delta, dict) else ""
        except (AttributeError, IndexError, json.JSONDecodeError):
            content = ""
        if isinstance(content, str) and content:
            chunks.append(content)
    return "".join(chunks)


def select_better_result(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Prefer usable text, then the stronger deterministic quality score."""

    status_rank = {
        "completed": 3,
        "no_readable_text": 2,
        "rejected_low_quality": 1,
        "failed": 0,
    }
    candidates = [dict(value) for value in results if isinstance(value, Mapping)]
    if not candidates:
        return {"status": "failed", "error": "no_ocr_attempt_result"}
    return max(
        candidates,
        key=lambda value: (
            status_rank.get(str(value.get("status") or ""), -1),
            float(value.get("quality_score") or 0.0),
            -float(value.get("latency_ms") or math.inf),
        ),
    )
