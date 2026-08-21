#!/usr/bin/env python3
"""Run a resumable direct-Transformers Unlimited-OCR batch on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


MODEL_NAME = "baidu/Unlimited-OCR"
MODEL_REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="Benchmark every item, ignoring its route.")
    parser.add_argument("--mode", choices=("gundam", "base"), default="gundam")
    return parser.parse_args()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _selected(items: Iterable[Dict[str, Any]], include_all: bool) -> List[Dict[str, Any]]:
    return [
        item
        for item in items
        if include_all or str(item.get("ocr_route") or "") == "unlimited_ocr"
    ]


def _resolve_image_path(manifest_path: Path, image_file: Any) -> Path:
    batch_root = manifest_path.parent.resolve()
    path = (batch_root / str(image_file or "")).resolve()
    try:
        path.relative_to(batch_root)
    except ValueError as exc:
        raise ValueError(f"Input image escapes the batch directory: {path}") from exc
    if not path.is_file():
        raise ValueError(f"Input image is missing: {path}")
    return path


def _model_source_evidence(
    model_dir: Path, *, expected_revision: str = MODEL_REVISION
) -> Dict[str, Any]:
    """Prove that a local model directory came from the pinned HF revision."""

    model_dir = model_dir.resolve()
    metadata_dir = model_dir / ".cache" / "huggingface" / "download"
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError(f"Pinned model index is missing: {index_path}")
    index = _load(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Pinned model index does not contain a weight map")
    required_files = {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        *(str(value) for value in weight_map.values()),
    }
    file_evidence: Dict[str, Dict[str, Any]] = {}
    if metadata_dir.is_dir():
        for filename in sorted(required_files):
            source_path = model_dir / filename
            metadata_path = metadata_dir / f"{filename}.metadata"
            if not source_path.is_file() or not metadata_path.is_file():
                raise ValueError(f"Pinned model file or metadata is missing: {filename}")
            metadata_lines = metadata_path.read_text(encoding="utf-8").splitlines()
            if not metadata_lines or metadata_lines[0].strip() != expected_revision:
                raise ValueError(f"Model file is not from the pinned revision: {filename}")
            file_evidence[filename] = {
                "revision": metadata_lines[0].strip(),
                "etag": metadata_lines[1].strip() if len(metadata_lines) > 1 else "",
                "size_bytes": source_path.stat().st_size,
            }
        source = "huggingface_local_dir_metadata"
    elif model_dir.parent.name == "snapshots" and model_dir.name == expected_revision:
        for filename in sorted(required_files):
            source_path = model_dir / filename
            if not source_path.is_file():
                raise ValueError(f"Pinned snapshot model file is missing: {filename}")
            file_evidence[filename] = {"size_bytes": source_path.stat().st_size}
        source = "huggingface_snapshot_path"
    else:
        raise ValueError(
            "Local model directory lacks Hugging Face revision evidence for "
            f"{expected_revision}"
        )
    return {
        "repository": MODEL_NAME,
        "revision": expected_revision,
        "verification_source": source,
        "files": file_evidence,
    }


def main() -> int:
    args = _args()
    manifest = _load(args.manifest)
    items = _selected(list(manifest.get("items") or []), args.all)
    contract_hash = str(manifest.get("batch_contract_sha256") or "")
    existing = _load(args.output) if args.output.is_file() else {}
    if existing and str(existing.get("batch_contract_sha256") or "") != contract_hash:
        raise ValueError("Existing Unlimited-OCR result belongs to a different batch contract")
    results = {
        str(key): dict(value)
        for key, value in (existing.get("results") or {}).items()
        if isinstance(value, dict) and value.get("status") in {"raw_completed", "failed"}
    }

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_source_evidence = _model_source_evidence(args.model_dir)
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    model = AutoModel.from_pretrained(
        str(args.model_dir),
        trust_remote_code=True,
        use_safetensors=True,
        torch_dtype=torch.bfloat16,
    ).eval().cuda()
    model_load_ms = round((time.monotonic() - started) * 1000, 3)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    output: Dict[str, Any] = {
        "version": 1,
        "kind": "unlimited_ocr_batch_results",
        "batch_contract_sha256": contract_hash,
        "provider": "transformers_direct",
        "provider_revision": "transformers-4.51.3/torch-2.7.1-cu118",
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "model_source_evidence": model_source_evidence,
        "mode": args.mode,
        "prompt": "<image>document parsing.",
        "prompt_revision": "unlimited-ocr-document-v1",
        "model_load_ms": model_load_ms,
        "results": results,
    }
    for index, item in enumerate(items, start=1):
        content_hash = str(item["content_hash"])
        if content_hash in results and results[content_hash].get("status") == "raw_completed":
            continue
        image_path = _resolve_image_path(args.manifest, item.get("image_file"))
        actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if actual_hash != content_hash:
            raise ValueError(f"Input image hash mismatch: {image_path}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        try:
            if args.mode == "gundam":
                image_size, crop_mode, ngram_window = 640, True, 128
            else:
                image_size, crop_mode, ngram_window = 1024, False, 1024
            raw_output = model.infer(
                tokenizer,
                prompt="<image>document parsing.",
                image_file=str(image_path),
                output_path=str(args.work_dir / content_hash),
                base_size=1024,
                image_size=image_size,
                crop_mode=crop_mode,
                max_length=32768,
                no_repeat_ngram_size=35,
                ngram_window=ngram_window,
                save_results=False,
                eval_mode=True,
                temperature=0.0,
            )
            result = {
                "status": "raw_completed",
                "content_hash": content_hash,
                "provider": "transformers_direct",
                "provider_revision": output["provider_revision"],
                "model": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "mode": args.mode,
                "prompt_revision": output["prompt_revision"],
                "raw_output": str(raw_output or ""),
                "raw_output_sha256": hashlib.sha256(
                    str(raw_output or "").encode("utf-8")
                ).hexdigest(),
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "peak_allocated_gib": round(
                    torch.cuda.max_memory_allocated() / (1024**3), 6
                ),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "content_hash": content_hash,
                "provider": "transformers_direct",
                "provider_revision": output["provider_revision"],
                "model": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "mode": args.mode,
                "raw_output": "",
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "error": " ".join(str(exc).split())[:600],
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        results[content_hash] = result
        output["results"] = results
        output["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(args.output, output)
        print(f"[{index}/{len(items)}] {content_hash[:12]} {result['status']} {result['latency_ms']}ms", flush=True)
    output["finished_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
