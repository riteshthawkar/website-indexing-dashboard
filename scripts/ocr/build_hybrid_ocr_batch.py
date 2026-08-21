#!/usr/bin/env python3
"""Materialize a hash-verified, path-independent hybrid OCR work batch."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.hybrid_ocr import (
    batch_contract_sha256,
    build_stratified_benchmark,
    choose_ocr_route,
    validate_local_assets,
)
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.core.media import load_media_manifest_items


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Build the fixed stratified 20-image benchmark instead of the full OCR queue.",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    payload = load_json_safe(args.manifest, {})
    source_items = load_media_manifest_items(payload)
    if args.benchmark:
        selected = build_stratified_benchmark(source_items)
    else:
        by_hash: Dict[str, Dict[str, Any]] = {}
        for raw in source_items:
            if raw.get("type") != "image" or raw.get("needs_ocr") is not True:
                continue
            content_hash = str(raw.get("content_hash") or "").lower()
            current = by_hash.get(content_hash)
            if current is None or len(str(raw.get("visible_text") or "")) > len(
                str(current.get("visible_text") or "")
            ):
                by_hash[content_hash] = dict(raw)
        selected = [by_hash[key] for key in sorted(by_hash)]
        for item in selected:
            item["ocr_route"] = choose_ocr_route(item)

    validate_local_assets(selected)
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    batch_items: List[Dict[str, Any]] = []
    for item in selected:
        source = Path(str(item["local_path"])).resolve()
        suffix = source.suffix.lower() or ".img"
        image_file = f"assets/{item['content_hash']}{suffix}"
        target = args.assets_dir / Path(image_file).name
        if not target.exists():
            shutil.copy2(source, target)
        batch_items.append(
            {
                "content_hash": str(item["content_hash"]),
                "image_file": image_file,
                "image_kind": str(item.get("image_kind") or ""),
                "source_type": str(item.get("source_type") or ""),
                "source_url": str(item.get("source_url") or ""),
                "document_id": str(item.get("document_id") or ""),
                "page_number": item.get("page_number"),
                "visible_text_reference": str(item.get("visible_text") or ""),
                "benchmark_bucket": str(item.get("benchmark_bucket") or ""),
                "ocr_route": str(item.get("ocr_route") or choose_ocr_route(item)),
            }
        )
    output = {
        "version": 1,
        "kind": "hybrid_ocr_benchmark" if args.benchmark else "hybrid_ocr_batch",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(args.manifest.resolve()),
        "item_count": len(batch_items),
        "batch_contract_sha256": batch_contract_sha256(batch_items),
        "items": batch_items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output, output)
    print(json.dumps({key: output[key] for key in ("kind", "item_count", "batch_contract_sha256")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
