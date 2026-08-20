"""Small, dependency-free helpers for Unlimited-OCR layout output.

Unlimited-OCR emits layout markers such as::

    <|det|>image [42, 80, 958, 620]<|/det|>

Coordinates use the model's normalized 0-1000 page space.  The model's
``crop_mode`` is an inference-time tiling strategy; it does not materialize
standalone figure files.  These helpers make the emitted image boxes usable by
the ingestion pipeline without coupling it to a particular GPU runtime.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple


_DETECTION_RE = re.compile(
    r"<\|det\|>\s*([^\s\[]+)\s*"
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*"
    r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    r"\s*<\|/det\|>\s*(.*)$",
    re.IGNORECASE,
)


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
