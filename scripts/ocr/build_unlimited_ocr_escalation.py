#!/usr/bin/env python3
"""Build the hash-pinned Unlimited-OCR escalation queue after scene OCR."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.hybrid_ocr import (
    UNLIMITED_OCR_ROUTE,
    assess_scene_ocr_lines,
    batch_contract_sha256,
    should_escalate_to_unlimited,
)
from pipeline.core.io import atomic_write_json, load_json_safe


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--scene-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _args()
    batch = load_json_safe(args.batch, {}) or {}
    scene = load_json_safe(args.scene_results, {}) or {}
    full_contract = str(batch.get("batch_contract_sha256") or "")
    if str(scene.get("batch_contract_sha256") or "") != full_contract:
        raise ValueError("Scene OCR result contract does not match the full batch")
    scene_results = scene.get("results") or {}
    selected: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for item in batch.get("items") or []:
        content_hash = str(item["content_hash"])
        raw = scene_results.get(content_hash)
        if not isinstance(raw, dict):
            quality: Dict[str, Any] = {"status": "failed", "quality_flags": ["missing_result"]}
        elif raw.get("status") == "raw_completed":
            quality = assess_scene_ocr_lines(
                raw.get("raw_lines") or [],
                reference_text=str(item.get("visible_text_reference") or ""),
            )
        else:
            quality = {"status": "failed", "quality_flags": ["provider_failure"]}
        escalate = should_escalate_to_unlimited(item, quality)
        decisions.append(
            {
                "content_hash": content_hash,
                "image_kind": str(item.get("image_kind") or ""),
                "scene_status": str(quality.get("status") or "failed"),
                "scene_quality_flags": list(quality.get("quality_flags") or []),
                "escalated": escalate,
            }
        )
        if escalate:
            selected.append(
                {
                    **item,
                    "ocr_route": UNLIMITED_OCR_ROUTE,
                    "scene_status": str(quality.get("status") or "failed"),
                }
            )
    output = {
        "version": 1,
        "kind": "unlimited_ocr_escalation_batch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_batch_contract_sha256": full_contract,
        "item_count": len(selected),
        "batch_contract_sha256": batch_contract_sha256(selected),
        "items": selected,
        "decisions": decisions,
    }
    atomic_write_json(args.output, output)
    print(
        json.dumps(
            {
                "item_count": len(selected),
                "batch_contract_sha256": output["batch_contract_sha256"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
