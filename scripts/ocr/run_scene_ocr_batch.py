#!/usr/bin/env python3
"""Run a resumable PP-OCRv5 Arabic+English scene-text batch on one GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


MODEL_REVISION = "paddleocr-3.7.0/paddle-3.2.0"
DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "arabic_PP-OCRv5_mobile_rec"
SUPPORTED_INPUT_SUFFIXES = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
    ".pbm",
    ".pgm",
    ".ppm",
    ".pnm",
    ".sr",
    ".ras",
    ".tiff",
    ".tif",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all", action="store_true", help="Benchmark every item, ignoring its route.")
    parser.add_argument("--line-confidence", type=float, default=0.0)
    parser.add_argument("--recognition-batch-size", type=int, default=32)
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
        if include_all or str(item.get("ocr_route") or "") == "paddleocr_scene"
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


def _prepare_prediction_path(
    image_path: Path, output_path: Path, content_hash: str
) -> tuple[Path, Dict[str, Any]]:
    if image_path.suffix.lower() in SUPPORTED_INPUT_SUFFIXES:
        return image_path, {"converted": False, "source_suffix": image_path.suffix.lower()}
    from PIL import Image, ImageOps

    converted_dir = output_path.parent / "preprocessed"
    converted_dir.mkdir(parents=True, exist_ok=True)
    converted_path = converted_dir / f"{content_hash}.png"
    if not converted_path.is_file():
        with Image.open(image_path) as source:
            ImageOps.exif_transpose(source).convert("RGB").save(converted_path, format="PNG")
    return converted_path, {
        "converted": True,
        "source_suffix": image_path.suffix.lower(),
        "prediction_suffix": ".png",
        "converted_sha256": hashlib.sha256(converted_path.read_bytes()).hexdigest(),
    }


def main() -> int:
    args = _args()
    manifest = _load(args.manifest)
    items = _selected(list(manifest.get("items") or []), args.all)
    contract_hash = str(manifest.get("batch_contract_sha256") or "")
    existing = _load(args.output) if args.output.is_file() else {}
    if existing and str(existing.get("batch_contract_sha256") or "") != contract_hash:
        raise ValueError("Existing scene OCR result belongs to a different batch contract")
    results = {
        str(key): dict(value)
        for key, value in (existing.get("results") or {}).items()
        if isinstance(value, dict) and value.get("status") in {"raw_completed", "failed"}
    }

    import paddle
    import paddleocr
    from paddleocr import PaddleOCR

    started = time.monotonic()
    pipeline = PaddleOCR(
        lang="ar",
        ocr_version="PP-OCRv5",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_rec_score_thresh=max(0.0, min(1.0, args.line_confidence)),
        text_recognition_batch_size=max(1, args.recognition_batch_size),
        device="gpu:0",
    )
    model_load_ms = round((time.monotonic() - started) * 1000, 3)
    output: Dict[str, Any] = {
        "version": 1,
        "kind": "scene_ocr_batch_results",
        "batch_contract_sha256": contract_hash,
        "provider": "paddleocr",
        "provider_revision": paddleocr.__version__,
        "model": f"{DETECTION_MODEL}+{RECOGNITION_MODEL}",
        "model_revision": MODEL_REVISION,
        "paddle_version": paddle.__version__,
        "language": "Arabic+English",
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
        prediction_path, preprocessing = _prepare_prediction_path(
            image_path, args.output, content_hash
        )
        started = time.monotonic()
        try:
            predictions = list(pipeline.predict(str(prediction_path)))
            if not predictions:
                raise RuntimeError("PaddleOCR returned no prediction record for a validated image")
            payloads = [value.json.get("res", {}) for value in predictions]
            lines: List[Dict[str, Any]] = []
            for payload in payloads:
                texts = list(payload.get("rec_texts") or [])
                scores = list(payload.get("rec_scores") or [])
                boxes = list(payload.get("rec_boxes") or [])
                for line_index, text in enumerate(texts):
                    lines.append(
                        {
                            "text": str(text),
                            "confidence": float(scores[line_index])
                            if line_index < len(scores)
                            else 0.0,
                            "box": boxes[line_index] if line_index < len(boxes) else [],
                        }
                    )
            canonical_raw = json.dumps(lines, ensure_ascii=False, sort_keys=True)
            result = {
                "status": "raw_completed",
                "content_hash": content_hash,
                "provider": "paddleocr",
                "provider_revision": paddleocr.__version__,
                "model": output["model"],
                "model_revision": MODEL_REVISION,
                "preprocessing": preprocessing,
                "raw_lines": lines,
                "raw_output_sha256": hashlib.sha256(canonical_raw.encode("utf-8")).hexdigest(),
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "content_hash": content_hash,
                "provider": "paddleocr",
                "provider_revision": paddleocr.__version__,
                "model": output["model"],
                "model_revision": MODEL_REVISION,
                "preprocessing": preprocessing,
                "raw_lines": [],
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
