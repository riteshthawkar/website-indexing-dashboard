from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "runs/evaluation/mbzuai-multilingual-controlled-ab-v1"
)
DEFAULT_REMOTE_DIR = "/tmp/mbzuai-multilingual-controlled-ab-qwen-v1"


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: Sequence[str]) -> None:
    print("Running:", " ".join(shlex.quote(value) for value in command), flush=True)
    subprocess.run(list(command), check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the controlled Qwen3 embedding candidate on the configured GPU host"
    )
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument("--host", default="ritesh_hpc")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    args = parser.parse_args(argv)

    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    input_manifest_path = experiment_dir / "embedding_inputs/manifest.json"
    input_manifest = _read_json(input_manifest_path)
    documents_path = Path(str(input_manifest["documents_path"]))
    queries_path = Path(str(input_manifest["queries_path"]))
    if _sha256_file(documents_path) != input_manifest["documents_sha256"]:
        raise RuntimeError("Document embedding input hash mismatch")
    if _sha256_file(queries_path) != input_manifest["queries_sha256"]:
        raise RuntimeError("Query embedding input hash mismatch")

    worker_path = PROJECT_ROOT / "scripts/qwen_embedding_worker.py"
    remote_dir = str(args.remote_dir).rstrip("/")
    _run(["ssh", "-o", "BatchMode=yes", args.host, "mkdir", "-p", f"{remote_dir}/input", f"{remote_dir}/output"])
    _run(
        [
            "rsync",
            "-az",
            "--partial",
            str(documents_path),
            str(queries_path),
            str(worker_path),
            f"{args.host}:{remote_dir}/input/",
        ]
    )
    remote_command = " ".join(
        shlex.quote(value)
        for value in [
            "python3",
            f"{remote_dir}/input/qwen_embedding_worker.py",
            "--documents",
            f"{remote_dir}/input/documents.jsonl",
            "--queries",
            f"{remote_dir}/input/queries.jsonl",
            "--output-dir",
            f"{remote_dir}/output",
            "--max-length",
            str(args.max_length),
            "--max-batch-size",
            str(args.max_batch_size),
            "--max-batch-tokens",
            str(args.max_batch_tokens),
        ]
    )
    started = time.perf_counter()
    _run(["ssh", "-o", "BatchMode=yes", args.host, remote_command])
    remote_seconds = time.perf_counter() - started

    local_output = experiment_dir / "embeddings/qwen3_06b_1024"
    local_output.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-az",
            "--partial",
            f"{args.host}:{remote_dir}/output/",
            f"{local_output}/",
        ]
    )
    manifest_path = local_output / "manifest.json"
    manifest = _read_json(manifest_path)
    expected = {
        "documents_input_sha256": input_manifest["documents_sha256"],
        "queries_input_sha256": input_manifest["queries_sha256"],
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"Remote output provenance mismatch for {key}")
    transfer_manifest = {
        "schema_version": "mbzuai.multilingual.ab_remote_transfer.v1",
        "created_at_epoch": int(time.time()),
        "host_alias": args.host,
        "remote_directory": remote_dir,
        "remote_run_wall_seconds": remote_seconds,
        "worker_sha256": _sha256_file(worker_path),
        "embedding_manifest": str(manifest_path),
        "embedding_manifest_sha256": _sha256_file(manifest_path),
        "production_mutation_performed": False,
    }
    transfer_path = local_output / "remote_transfer.json"
    transfer_path.write_text(
        json.dumps(transfer_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(transfer_manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
