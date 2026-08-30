#!/usr/bin/env python3
"""Seed a new Unlimited-OCR batch with byte-identical completed results."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROVENANCE_FIELDS = (
    "provider",
    "provider_revision",
    "model",
    "model_revision",
    "model_source_evidence",
    "mode",
    "prompt",
    "prompt_revision",
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed-manifest", type=Path, required=True)
    parser.add_argument("--seed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _items_by_hash(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item["content_hash"]): dict(item)
        for item in manifest.get("items") or []
        if isinstance(item, dict) and item.get("content_hash")
    }


def _verify_asset(manifest_path: Path, item: Dict[str, Any], content_hash: str) -> None:
    batch_root = manifest_path.parent.resolve()
    image_path = (batch_root / str(item.get("image_file") or "")).resolve()
    try:
        image_path.relative_to(batch_root)
    except ValueError as exc:
        raise ValueError(f"Input image escapes the batch directory: {image_path}") from exc
    if not image_path.is_file() or _sha256_file(image_path) != content_hash:
        raise ValueError(f"Input image does not match content hash: {image_path}")


def main() -> int:
    args = _args()
    manifest = _load(args.manifest)
    seed_manifest = _load(args.seed_manifest)
    seed_results = _load(args.seed_results)
    output = _load(args.output) if args.output.is_file() else {}

    target_contract = str(manifest.get("batch_contract_sha256") or "")
    seed_contract = str(seed_manifest.get("batch_contract_sha256") or "")
    if not target_contract or not seed_contract:
        raise ValueError("Both manifests must contain a batch contract SHA-256")
    if str(seed_results.get("batch_contract_sha256") or "") != seed_contract:
        raise ValueError("Seed result does not match its source batch contract")
    if seed_results.get("kind") != "unlimited_ocr_batch_results":
        raise ValueError("Seed file is not an Unlimited-OCR result manifest")
    if output and str(output.get("batch_contract_sha256") or "") != target_contract:
        raise ValueError("Existing output belongs to a different target batch contract")
    for field in PROVENANCE_FIELDS:
        if output.get(field) and output.get(field) != seed_results.get(field):
            raise ValueError(f"OCR provenance mismatch for {field}")

    current_items = _items_by_hash(manifest)
    seed_items = _items_by_hash(seed_manifest)
    merged_results = {
        str(key): dict(value)
        for key, value in (output.get("results") or {}).items()
        if isinstance(value, dict) and str(key) in current_items
    }
    reused = 0
    seed_file_sha256 = _sha256_file(args.seed_results)
    for content_hash in sorted(set(current_items) & set(seed_items)):
        if merged_results.get(content_hash, {}).get("status") == "raw_completed":
            continue
        raw = (seed_results.get("results") or {}).get(content_hash)
        if not isinstance(raw, dict) or raw.get("status") != "raw_completed":
            continue
        if str(raw.get("content_hash") or "") != content_hash:
            raise ValueError(f"Seed result content hash mismatch: {content_hash}")
        _verify_asset(args.manifest, current_items[content_hash], content_hash)
        record = copy.deepcopy(raw)
        record["result_reused"] = True
        record["result_reuse_evidence"] = {
            "seed_batch_contract_sha256": seed_contract,
            "seed_results_sha256": seed_file_sha256,
        }
        merged_results[content_hash] = record
        reused += 1

    payload = {
        "version": 1,
        "kind": "unlimited_ocr_batch_results",
        "batch_contract_sha256": target_contract,
        **{field: output.get(field) or seed_results.get(field) for field in PROVENANCE_FIELDS},
        "model_load_ms": output.get("model_load_ms") or seed_results.get("model_load_ms"),
        "results": merged_results,
        "result_reuse": {
            "seed_manifest": str(args.seed_manifest.resolve()),
            "seed_manifest_sha256": _sha256_file(args.seed_manifest),
            "seed_results": str(args.seed_results.resolve()),
            "seed_results_sha256": seed_file_sha256,
            "reused_result_count": reused,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write(args.output, payload)
    print(
        json.dumps(
            {
                "target_item_count": len(current_items),
                "result_count": len(merged_results),
                "reused_result_count": reused,
                "remaining_count": len(current_items) - len(merged_results),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
