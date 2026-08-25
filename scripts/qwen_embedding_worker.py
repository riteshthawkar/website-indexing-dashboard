from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DIMENSION = 1024
QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id") or "")
            text = str(row.get("text") or "").strip()
            if not row_id or not text:
                raise ValueError(f"Invalid row on line {line_number} of {path}")
            rows.append({"id": row_id, "text": text})
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nvidia_smi() -> Dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=15,
        ).strip()
        name, total, used, utilization = [part.strip() for part in output.splitlines()[0].split(",")]
        return {
            "name": name,
            "memory_total_mib": int(total),
            "memory_used_mib": int(used),
            "utilization_percent": int(utilization),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _last_token_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class QwenEmbedder:
    def __init__(self, *, max_length: int) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the Qwen embedding benchmark")
        self.max_length = max_length
        before = _nvidia_smi()
        started = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            padding_side="left",
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda")
        self.model.eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        self.gpu_before_load = before
        self.gpu_after_load = _nvidia_smi()
        self.parameter_count = sum(parameter.numel() for parameter in self.model.parameters())
        self.parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in self.model.parameters()
        )
        self.possible_truncation_count = 0
        self.internal_batch_latencies: List[float] = []
        self.internal_batch_sizes: List[int] = []

    def formatted_text(self, text: str, *, task_type: str) -> str:
        if task_type == "RETRIEVAL_QUERY":
            return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"
        return text

    @torch.inference_mode()
    def embed_texts(self, texts: Sequence[str], *, task_type: str) -> np.ndarray:
        formatted = [self.formatted_text(text, task_type=task_type) for text in texts]
        started = time.perf_counter()
        batch = self.tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        lengths = batch["attention_mask"].sum(dim=1)
        self.possible_truncation_count += int((lengths >= self.max_length).sum().item())
        batch = {key: value.to("cuda", non_blocking=True) for key, value in batch.items()}
        outputs = self.model(**batch)
        embeddings = _last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
        embeddings = F.normalize(embeddings.float(), p=2, dim=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.internal_batch_latencies.append(elapsed)
        self.internal_batch_sizes.append(len(texts))
        return embeddings.cpu().numpy().astype(np.float32, copy=False)

    def embed_indices(
        self,
        rows: Sequence[Mapping[str, str]],
        indices: Sequence[int],
        *,
        task_type: str,
    ) -> np.ndarray:
        try:
            return self.embed_texts(
                [str(rows[index]["text"]) for index in indices], task_type=task_type
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(indices) <= 1:
                raise
            midpoint = len(indices) // 2
            first = self.embed_indices(
                rows, indices[:midpoint], task_type=task_type
            )
            second = self.embed_indices(
                rows, indices[midpoint:], task_type=task_type
            )
            return np.concatenate([first, second], axis=0)


def _dynamic_batches(
    rows: Sequence[Mapping[str, str]],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
    max_length: int,
) -> List[List[int]]:
    ordered = sorted(range(len(rows)), key=lambda index: len(str(rows[index]["text"])))
    batches: List[List[int]] = []
    current: List[int] = []
    current_estimated_tokens = 0
    for index in ordered:
        text = str(rows[index]["text"])
        estimated_tokens = min(max_length, max(8, int(len(text) / 2.5)))
        if current and (
            len(current) >= max_batch_size
            or current_estimated_tokens + estimated_tokens > max_batch_tokens
        ):
            batches.append(current)
            current = []
            current_estimated_tokens = 0
        current.append(index)
        current_estimated_tokens += estimated_tokens
    if current:
        batches.append(current)
    return batches


def _valid_shard(
    path: Path,
    metadata_path: Path,
    *,
    rows: Sequence[Mapping[str, str]],
    task_type: str,
    max_length: int,
    max_batch_size: int,
    max_batch_tokens: int,
) -> bool:
    if not path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("ids") != [str(row["id"]) for row in rows]:
            return False
        if metadata.get("model") != MODEL_NAME or metadata.get("task_type") != task_type:
            return False
        if int(metadata.get("max_length") or 0) != max_length:
            return False
        if "possible_truncation_count" not in metadata:
            return False
        if int(metadata.get("possible_truncation_count") or 0) != 0:
            return False
        if int(metadata.get("row_count") or 0) != len(rows):
            return False
        if int(metadata.get("dimension") or 0) != DIMENSION:
            return False
        if int(metadata.get("max_batch_size") or 0) != max_batch_size:
            return False
        if int(metadata.get("max_batch_tokens") or 0) != max_batch_tokens:
            return False
        vectors = np.load(path, mmap_mode="r")
        if vectors.shape != (len(rows), DIMENSION) or vectors.dtype != np.float32:
            return False
        if metadata.get("vectors_sha256") != _sha256_file(path):
            return False
        norms = np.linalg.norm(np.asarray(vectors), axis=1)
        return bool(
            np.all(np.isfinite(norms))
            and float(np.max(np.abs(norms - 1.0))) <= 2e-4
        )
    except Exception:
        return False


def _run_collection(
    embedder: QwenEmbedder,
    *,
    name: str,
    rows: Sequence[Mapping[str, str]],
    output_root: Path,
    task_type: str,
    shard_size: int,
    max_batch_size: int,
    max_batch_tokens: int,
) -> Dict[str, Any]:
    output_dir = output_root / name
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    shard_rows = [rows[start : start + shard_size] for start in range(0, len(rows), shard_size)]
    started = time.perf_counter()
    resumed = 0
    for shard_number, current_rows in enumerate(shard_rows):
        vector_path = shards_dir / f"{shard_number:06d}.npy"
        metadata_path = shards_dir / f"{shard_number:06d}.json"
        if _valid_shard(
            vector_path,
            metadata_path,
            rows=current_rows,
            task_type=task_type,
            max_length=embedder.max_length,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
        ):
            resumed += 1
            continue
        shard_started = time.perf_counter()
        truncation_count_before = embedder.possible_truncation_count
        vectors = np.empty((len(current_rows), DIMENSION), dtype=np.float32)
        batches = _dynamic_batches(
            current_rows,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
            max_length=embedder.max_length,
        )
        for indices in batches:
            vectors[np.asarray(indices)] = embedder.embed_indices(
                current_rows, indices, task_type=task_type
            )
        temporary = vector_path.with_name(vector_path.name + ".tmp")
        with temporary.open("wb") as handle:
            np.save(handle, vectors, allow_pickle=False)
        os.replace(temporary, vector_path)
        _write_json(
            metadata_path,
            {
                "shard_number": shard_number,
                "ids": [str(row["id"]) for row in current_rows],
                "row_count": len(current_rows),
                "dimension": DIMENSION,
                "model": MODEL_NAME,
                "task_type": task_type,
                "max_length": embedder.max_length,
                "max_batch_size": max_batch_size,
                "max_batch_tokens": max_batch_tokens,
                "possible_truncation_count": (
                    embedder.possible_truncation_count - truncation_count_before
                ),
                "internal_batch_count": len(batches),
                "elapsed_seconds": time.perf_counter() - shard_started,
                "vectors_sha256": _sha256_file(vector_path),
            },
        )
        if (shard_number + 1) % 10 == 0 or shard_number + 1 == len(shard_rows):
            print(
                f"{name}: completed {shard_number + 1}/{len(shard_rows)} shards",
                flush=True,
            )

    vectors_path = output_dir / "vectors.npy"
    temporary = vectors_path.with_name(vectors_path.name + ".tmp")
    merged = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), DIMENSION),
    )
    offset = 0
    for shard_number, current_rows in enumerate(shard_rows):
        shard_vectors = np.load(
            shards_dir / f"{shard_number:06d}.npy", mmap_mode="r"
        )
        merged[offset : offset + len(current_rows)] = shard_vectors
        offset += len(current_rows)
    merged.flush()
    del merged
    os.replace(temporary, vectors_path)
    ids_path = output_dir / "ids.json"
    _write_json(ids_path, [str(row["id"]) for row in rows])
    result = {
        "name": name,
        "model": MODEL_NAME,
        "dimension": DIMENSION,
        "task_type": task_type,
        "row_count": len(rows),
        "shard_size": shard_size,
        "shard_count": len(shard_rows),
        "resumed_shard_count": resumed,
        "max_batch_size": max_batch_size,
        "max_batch_tokens": max_batch_tokens,
        "max_length": embedder.max_length,
        "possible_truncation_count": sum(
            int(
                json.loads(
                    (shards_dir / f"{shard_number:06d}.json").read_text(
                        encoding="utf-8"
                    )
                ).get("possible_truncation_count")
                or 0
            )
            for shard_number in range(len(shard_rows))
        ),
        "wall_seconds_this_run": time.perf_counter() - started,
        "vectors_path": str(vectors_path.resolve()),
        "vectors_sha256": _sha256_file(vectors_path),
        "ids_path": str(ids_path.resolve()),
        "ids_sha256": _sha256_file(ids_path),
    }
    _write_json(output_dir / "manifest.json", result)
    return result


def _percentile(values: Sequence[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


def _query_latency_benchmark(
    embedder: QwenEmbedder, queries: Sequence[Mapping[str, str]]
) -> Dict[str, Any]:
    results = []
    iterations_by_batch_size = {1: 12, 8: 5, 32: 3}
    for batch_size in (1, 8, 32):
        latencies = []
        for iteration in range(iterations_by_batch_size[batch_size]):
            start = (iteration * batch_size) % len(queries)
            batch = list(queries[start : start + batch_size])
            if len(batch) < batch_size:
                batch += list(queries[: batch_size - len(batch)])
            started = time.perf_counter()
            embedder.embed_texts(
                [str(row["text"]) for row in batch], task_type="RETRIEVAL_QUERY"
            )
            latencies.append(time.perf_counter() - started)
        results.append(
            {
                "batch_size": batch_size,
                "iterations": len(latencies),
                "p50_seconds": _percentile(latencies, 50),
                "p95_seconds": _percentile(latencies, 95),
                "mean_queries_per_second": batch_size / (sum(latencies) / len(latencies)),
            }
        )
    return {"batching": results}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remote Qwen3 embedding worker")
    parser.add_argument("--documents", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=32768)
    parser.add_argument("--shard-size", type=int, default=256)
    args = parser.parse_args(argv)

    documents_path = Path(args.documents).resolve()
    queries_path = Path(args.queries).resolve()
    output_root = Path(args.output_dir).resolve()
    documents = _read_jsonl(documents_path)
    queries = _read_jsonl(queries_path)
    torch.cuda.reset_peak_memory_stats()
    embedder = QwenEmbedder(max_length=args.max_length)
    load_peak = int(torch.cuda.max_memory_allocated())
    document_manifest = _run_collection(
        embedder,
        name="documents",
        rows=documents,
        output_root=output_root,
        task_type="RETRIEVAL_DOCUMENT",
        shard_size=args.shard_size,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    query_manifest = _run_collection(
        embedder,
        name="queries",
        rows=queries,
        output_root=output_root,
        task_type="RETRIEVAL_QUERY",
        shard_size=args.shard_size,
        max_batch_size=args.max_batch_size,
        max_batch_tokens=args.max_batch_tokens,
    )
    if (
        int(document_manifest["possible_truncation_count"]) > 0
        or int(query_manifest["possible_truncation_count"]) > 0
    ):
        raise RuntimeError(
            "Qwen input truncation detected; increase --max-length before comparing "
            "this candidate"
        )
    query_latency = _query_latency_benchmark(embedder, queries)
    result = {
        "schema_version": "mbzuai.multilingual.ab_embeddings.v1",
        "created_at_epoch": int(time.time()),
        "provider": "self_hosted_cuda",
        "model": MODEL_NAME,
        "dimension": DIMENSION,
        "production_mutation_performed": False,
        "query_instruction": QUERY_INSTRUCTION,
        "pooling": "last_token",
        "normalization": "l2",
        "documents_input": str(documents_path),
        "documents_input_sha256": _sha256_file(documents_path),
        "queries_input": str(queries_path),
        "queries_input_sha256": _sha256_file(queries_path),
        "documents": document_manifest,
        "queries": query_manifest,
        "query_latency_benchmark": query_latency,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cuda": torch.version.cuda,
            "gpu_before_model_load": embedder.gpu_before_load,
            "gpu_after_model_load": embedder.gpu_after_load,
            "gpu_after_benchmark": _nvidia_smi(),
            "model_load_seconds": embedder.load_seconds,
            "parameter_count": embedder.parameter_count,
            "parameter_bytes": embedder.parameter_bytes,
            "peak_cuda_allocated_bytes_after_load": load_peak,
            "peak_cuda_allocated_bytes_total": int(torch.cuda.max_memory_allocated()),
            "peak_cuda_reserved_bytes_total": int(torch.cuda.max_memory_reserved()),
            "possible_truncation_count": embedder.possible_truncation_count,
            "internal_batch_count": len(embedder.internal_batch_latencies),
            "internal_batch_latency_seconds": {
                "p50": _percentile(embedder.internal_batch_latencies, 50),
                "p95": _percentile(embedder.internal_batch_latencies, 95),
                "maximum": max(embedder.internal_batch_latencies)
                if embedder.internal_batch_latencies
                else 0.0,
            },
        },
    }
    _write_json(output_root / "manifest.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
