from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stages.embedders.gemini_pinecone_embedder import (
    _call_with_retry,
    _embed_multimodal_batch,
    _embed_text_batch,
    _format_embedding_text,
    _make_gemini_client,
)
from pipeline.core.chunking import estimate_token_count, token_counting_method


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "runs/evaluation/mbzuai-multilingual-controlled-ab-v1"
)
MODEL = "gemini-embedding-2"
PRIMARY_DIMENSION = 1536
DERIVED_DIMENSION = 768
GEMINI_INPUT_TOKEN_LIMIT = 8192
LOCAL_TEXT_SAFETY_BUDGET = 6000
GEMINI_IMAGE_TOKEN_ALLOWANCE = 258
_THREAD_LOCAL = threading.local()
_RATE_LOCK = threading.Lock()
_MINIMUM_REQUEST_INTERVAL_SECONDS = 0.0
_NEXT_REQUEST_TIME = 0.0


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            row_id = str(row.get("id") or "")
            text = str(row.get("text") or "").strip()
            if not row_id or not text:
                raise ValueError(f"Invalid embedding input on line {line_number} of {path}")
            normalized = {"id": row_id, "text": text}
            if row.get("local_path"):
                normalized["local_path"] = str(row["local_path"])
            if row.get("content_hash"):
                normalized["content_hash"] = str(row["content_hash"])
            rows.append(normalized)
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


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _client():
    client = getattr(_THREAD_LOCAL, "gemini_client", None)
    if client is None:
        client = _make_gemini_client(request_timeout_ms=180_000)
        _THREAD_LOCAL.gemini_client = client
    return client


def _configure_rate_limit(requests_per_minute: float) -> None:
    global _MINIMUM_REQUEST_INTERVAL_SECONDS, _NEXT_REQUEST_TIME
    with _RATE_LOCK:
        _MINIMUM_REQUEST_INTERVAL_SECONDS = (
            60.0 / requests_per_minute if requests_per_minute > 0.0 else 0.0
        )
        _NEXT_REQUEST_TIME = 0.0


def _wait_for_rate_slot() -> None:
    global _NEXT_REQUEST_TIME
    with _RATE_LOCK:
        now = time.monotonic()
        wait_seconds = max(0.0, _NEXT_REQUEST_TIME - now)
        _NEXT_REQUEST_TIME = max(now, _NEXT_REQUEST_TIME) + _MINIMUM_REQUEST_INTERVAL_SECONDS
    if wait_seconds > 0.0:
        time.sleep(wait_seconds)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0.0):
        raise RuntimeError("Embedding response contained an invalid vector")
    return (vectors / norms).astype(np.float32, copy=False)


def _local_token_budget_audit(
    rows: Sequence[Mapping[str, str]],
    *,
    task_type: str,
    image_tokens_each: int = 0,
) -> Dict[str, Any]:
    counts = [
        estimate_token_count(
            _format_embedding_text(
                row.get("text") or "", task_type=task_type, model=MODEL
            )
        )
        + image_tokens_each
        for row in rows
    ]
    maximum = max(counts) if counts else 0
    if maximum > LOCAL_TEXT_SAFETY_BUDGET:
        raise RuntimeError(
            f"Gemini input proxy count {maximum} exceeds the preregistered "
            f"{LOCAL_TEXT_SAFETY_BUDGET}-token safety budget"
        )
    return {
        "row_count": len(rows),
        "token_counting_method": token_counting_method(),
        "provider_input_limit": GEMINI_INPUT_TOKEN_LIMIT,
        "local_safety_budget": LOCAL_TEXT_SAFETY_BUDGET,
        "image_token_allowance_each": image_tokens_each,
        "maximum_proxy_tokens": maximum,
        "at_or_above_provider_limit": sum(
            value >= GEMINI_INPUT_TOKEN_LIMIT for value in counts
        ),
    }


def _embed(rows: Sequence[Mapping[str, str]], *, task_type: str, dimension: int) -> np.ndarray:
    def request() -> List[List[float]]:
        _wait_for_rate_slot()
        return _embed_text_batch(
            _client(),
            model=MODEL,
            texts=[str(row["text"]) for row in rows],
            task_type=task_type,
            output_dimensionality=dimension,
        )

    vectors = _call_with_retry(
        f"{MODEL} {task_type} batch",
        request,
        max_attempts=6,
        base_delay_sec=2.0,
        max_delay_sec=30.0,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (len(rows), dimension):
        raise RuntimeError(
            f"Unexpected embedding shape {array.shape}; expected {(len(rows), dimension)}"
        )
    return _normalize(array)


def _embed_multimodal(
    rows: Sequence[Mapping[str, str]], *, task_type: str, dimension: int
) -> np.ndarray:
    def request() -> List[List[float]]:
        _wait_for_rate_slot()
        return _embed_multimodal_batch(
            _client(),
            model=MODEL,
            items=[dict(row) for row in rows],
            task_type=task_type,
            output_dimensionality=dimension,
        )

    vectors = _call_with_retry(
        f"{MODEL} multimodal {task_type} batch",
        request,
        max_attempts=6,
        base_delay_sec=2.0,
        max_delay_sec=30.0,
    )
    array = np.asarray(vectors, dtype=np.float32)
    if array.shape != (len(rows), dimension):
        raise RuntimeError(
            f"Unexpected multimodal embedding shape {array.shape}; "
            f"expected {(len(rows), dimension)}"
        )
    return _normalize(array)


def _shard_paths(output_dir: Path, shard_number: int) -> tuple[Path, Path]:
    stem = f"{shard_number:06d}"
    return output_dir / "shards" / f"{stem}.npy", output_dir / "shards" / f"{stem}.json"


def _load_valid_shard(
    output_dir: Path,
    *,
    shard_number: int,
    expected_ids: Sequence[str],
    dimension: int,
    task_type: str,
) -> tuple[np.ndarray, Dict[str, Any]] | None:
    vector_path, metadata_path = _shard_paths(output_dir, shard_number)
    if not vector_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = _read_json(metadata_path)
        if metadata.get("ids") != list(expected_ids):
            return None
        if metadata.get("model") != MODEL or int(metadata.get("dimension") or 0) != dimension:
            return None
        if metadata.get("task_type") != task_type:
            return None
        if int(metadata.get("row_count") or 0) != len(expected_ids):
            return None
        vectors = np.load(vector_path, mmap_mode="r")
        if vectors.shape != (len(expected_ids), dimension) or vectors.dtype != np.float32:
            return None
        if metadata.get("vectors_sha256") != _sha256_file(vector_path):
            return None
        norms = np.linalg.norm(np.asarray(vectors), axis=1)
        if not np.all(np.isfinite(norms)) or float(np.max(np.abs(norms - 1.0))) > 2e-4:
            return None
        return vectors, metadata
    except Exception:
        return None


def _save_shard(
    output_dir: Path,
    *,
    shard_number: int,
    rows: Sequence[Mapping[str, str]],
    vectors: np.ndarray,
    elapsed_seconds: float,
    task_type: str,
) -> Dict[str, Any]:
    vector_path, metadata_path = _shard_paths(output_dir, shard_number)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    vector_temp = vector_path.with_name(vector_path.name + ".tmp")
    with vector_temp.open("wb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    os.replace(vector_temp, vector_path)
    metadata = {
        "shard_number": shard_number,
        "ids": [str(row["id"]) for row in rows],
        "model": MODEL,
        "dimension": int(vectors.shape[1]),
        "task_type": task_type,
        "row_count": len(rows),
        "input_characters": sum(len(str(row["text"])) for row in rows),
        "elapsed_seconds": round(float(elapsed_seconds), 6),
        "vectors_sha256": _sha256_file(vector_path),
    }
    _write_json(metadata_path, metadata)
    return metadata


def _embed_and_save_shard(
    output_dir: Path,
    *,
    shard_number: int,
    rows: Sequence[Mapping[str, str]],
    task_type: str,
    dimension: int,
    multimodal: bool,
) -> Dict[str, Any]:
    started = time.perf_counter()
    vectors = (
        _embed_multimodal(rows, task_type=task_type, dimension=dimension)
        if multimodal
        else _embed(rows, task_type=task_type, dimension=dimension)
    )
    return _save_shard(
        output_dir,
        shard_number=shard_number,
        rows=rows,
        vectors=vectors,
        elapsed_seconds=time.perf_counter() - started,
        task_type=task_type,
    )


def _run_collection(
    *,
    name: str,
    rows: Sequence[Mapping[str, str]],
    output_dir: Path,
    task_type: str,
    dimension: int,
    batch_size: int,
    workers: int,
    multimodal: bool = False,
) -> Dict[str, Any]:
    collection_dir = output_dir / name
    batches = [rows[start : start + batch_size] for start in range(0, len(rows), batch_size)]
    shard_metadata: Dict[int, Dict[str, Any]] = {}
    missing: List[tuple[int, Sequence[Mapping[str, str]]]] = []
    for shard_number, batch in enumerate(batches):
        cached = _load_valid_shard(
            collection_dir,
            shard_number=shard_number,
            expected_ids=[str(row["id"]) for row in batch],
            dimension=dimension,
            task_type=task_type,
        )
        if cached is None:
            missing.append((shard_number, batch))
        else:
            shard_metadata[shard_number] = cached[1]
    print(
        f"{name}: {len(rows)} rows, {len(batches)} shards, "
        f"{len(missing)} requests remaining",
        flush=True,
    )

    started = time.perf_counter()
    completed_now = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                _embed_and_save_shard,
                collection_dir,
                shard_number=shard_number,
                rows=batch,
                task_type=task_type,
                dimension=dimension,
                multimodal=multimodal,
            ): shard_number
            for shard_number, batch in missing
        }
        for future in as_completed(futures):
            shard_number = futures[future]
            shard_metadata[shard_number] = future.result()
            completed_now += 1
            if completed_now % 25 == 0 or completed_now == len(missing):
                print(
                    f"{name}: completed {completed_now}/{len(missing)} new shards",
                    flush=True,
                )

    vectors_path = collection_dir / "vectors.npy"
    vector_temp = vectors_path.with_name(vectors_path.name + ".tmp")
    merged = np.lib.format.open_memmap(
        vector_temp,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), dimension),
    )
    offset = 0
    for shard_number, batch in enumerate(batches):
        shard_path, _ = _shard_paths(collection_dir, shard_number)
        shard_vectors = np.load(shard_path, mmap_mode="r")
        merged[offset : offset + len(batch)] = shard_vectors
        offset += len(batch)
    merged.flush()
    del merged
    os.replace(vector_temp, vectors_path)

    # Earlier resumable trials may have used a smaller batch size. Keep only
    # shards addressable by this frozen collection so stale files cannot be
    # mistaken for part of the final artifact during later audits.
    stale_shards_removed = 0
    shards_dir = collection_dir / "shards"
    for shard_path in shards_dir.glob("*.*"):
        if shard_path.suffix not in {".json", ".npy"} or not shard_path.stem.isdigit():
            continue
        if int(shard_path.stem) >= len(batches):
            shard_path.unlink()
            stale_shards_removed += 1

    ids_path = collection_dir / "ids.json"
    _write_json(ids_path, [str(row["id"]) for row in rows])
    latencies = [float(shard_metadata[index]["elapsed_seconds"]) for index in sorted(shard_metadata)]
    characters = sum(int(value.get("input_characters") or 0) for value in shard_metadata.values())
    api_seconds = sum(latencies)
    collection_manifest = {
        "name": name,
        "model": MODEL,
        "dimension": dimension,
        "task_type": task_type,
        "row_count": len(rows),
        "batch_size": batch_size,
        "worker_count": workers,
        "input_mode": "image_and_caption_text" if multimodal else "text",
        "request_count": len(batches),
        "resumed_request_count": len(batches) - len(missing),
        "stale_shard_files_removed": stale_shards_removed,
        "wall_seconds_this_run": round(time.perf_counter() - started, 6),
        "summed_api_seconds": round(api_seconds, 6),
        "request_latency_seconds": {
            "p50": round(_percentile(latencies, 50), 6),
            "p95": round(_percentile(latencies, 95), 6),
            "maximum": round(max(latencies) if latencies else 0.0, 6),
        },
        "input_characters": characters,
        "vectors_path": str(vectors_path.resolve()),
        "vectors_sha256": _sha256_file(vectors_path),
        "ids_path": str(ids_path.resolve()),
        "ids_sha256": _sha256_file(ids_path),
    }
    _write_json(collection_dir / "manifest.json", collection_manifest)
    return collection_manifest


def _derive_collection(
    *,
    name: str,
    source_manifest: Mapping[str, Any],
    output_dir: Path,
    dimension: int,
) -> Dict[str, Any]:
    source_path = Path(str(source_manifest["vectors_path"]))
    source = np.load(source_path, mmap_mode="r")
    if source.shape[1] < dimension:
        raise ValueError(f"Cannot derive {dimension} dimensions from {source.shape[1]}")
    collection_dir = output_dir / name
    collection_dir.mkdir(parents=True, exist_ok=True)
    vector_path = collection_dir / "vectors.npy"
    temporary = vector_path.with_name(vector_path.name + ".tmp")
    target = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(source.shape[0], dimension),
    )
    block_size = 2048
    for start in range(0, source.shape[0], block_size):
        block = np.asarray(source[start : start + block_size, :dimension], dtype=np.float32)
        target[start : start + len(block)] = _normalize(block)
    target.flush()
    del target
    os.replace(temporary, vector_path)
    source_ids = Path(str(source_manifest["ids_path"]))
    ids_path = collection_dir / "ids.json"
    ids_path.write_bytes(source_ids.read_bytes())
    result = {
        "name": name,
        "model": MODEL,
        "dimension": dimension,
        "task_type": source_manifest["task_type"],
        "row_count": int(source.shape[0]),
        "derivation": "prefix_truncation_then_l2_normalization",
        "source_dimension": int(source.shape[1]),
        "source_vectors_sha256": str(source_manifest["vectors_sha256"]),
        "vectors_path": str(vector_path.resolve()),
        "vectors_sha256": _sha256_file(vector_path),
        "ids_path": str(ids_path.resolve()),
        "ids_sha256": _sha256_file(ids_path),
    }
    _write_json(collection_dir / "manifest.json", result)
    return result


def _validate_mrl(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    sample = list(rows[: min(8, len(rows))])
    direct = _embed(sample, task_type="RETRIEVAL_DOCUMENT", dimension=DERIVED_DIMENSION)
    full = _embed(sample, task_type="RETRIEVAL_DOCUMENT", dimension=PRIMARY_DIMENSION)
    truncated = _normalize(full[:, :DERIVED_DIMENSION])
    similarities = np.sum(direct * truncated, axis=1)
    result = {
        "sample_count": len(sample),
        "minimum_cosine_similarity": float(np.min(similarities)),
        "mean_cosine_similarity": float(np.mean(similarities)),
        "maximum_absolute_component_difference": float(np.max(np.abs(direct - truncated))),
        "required_minimum_cosine_similarity": 0.99999,
    }
    result["passed"] = result["minimum_cosine_similarity"] >= result["required_minimum_cosine_similarity"]
    if not result["passed"]:
        raise RuntimeError(f"Gemini MRL conformance check failed: {result}")
    return result


def _validate_multimodal_mrl(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    sample = list(rows[: min(4, len(rows))])
    direct = _embed_multimodal(
        sample, task_type="RETRIEVAL_DOCUMENT", dimension=DERIVED_DIMENSION
    )
    full = _embed_multimodal(
        sample, task_type="RETRIEVAL_DOCUMENT", dimension=PRIMARY_DIMENSION
    )
    truncated = _normalize(full[:, :DERIVED_DIMENSION])
    similarities = np.sum(direct * truncated, axis=1)
    result = {
        "sample_count": len(sample),
        "minimum_cosine_similarity": float(np.min(similarities)),
        "mean_cosine_similarity": float(np.mean(similarities)),
        "maximum_absolute_component_difference": float(
            np.max(np.abs(direct - truncated))
        ),
        "required_minimum_cosine_similarity": 0.99999,
    }
    result["passed"] = (
        result["minimum_cosine_similarity"]
        >= result["required_minimum_cosine_similarity"]
    )
    if not result["passed"]:
        raise RuntimeError(f"Gemini multimodal MRL conformance check failed: {result}")
    return result


def _query_latency_benchmark(
    queries: Sequence[Mapping[str, str]], *, dimension: int
) -> Dict[str, Any]:
    batch_results = []
    iterations_by_batch_size = {1: 12, 8: 5, 32: 3}
    for batch_size in (1, 8, 32):
        latencies = []
        for iteration in range(iterations_by_batch_size[batch_size]):
            start = (iteration * batch_size) % len(queries)
            batch = list(queries[start : start + batch_size])
            if len(batch) < batch_size:
                batch += list(queries[: batch_size - len(batch)])
            started = time.perf_counter()
            _embed(batch, task_type="RETRIEVAL_QUERY", dimension=dimension)
            latencies.append(time.perf_counter() - started)
        batch_results.append(
            {
                "batch_size": batch_size,
                "iterations": len(latencies),
                "p50_seconds": _percentile(latencies, 50),
                "p95_seconds": _percentile(latencies, 95),
                "mean_queries_per_second": batch_size / (sum(latencies) / len(latencies)),
            }
        )

    concurrency_results = []
    benchmark_rows = list(queries[:8])
    for concurrency in (1, 4, 8):
        started = time.perf_counter()
        individual_latencies: List[float] = []

        def one(row: Mapping[str, str]) -> float:
            request_started = time.perf_counter()
            _embed([row], task_type="RETRIEVAL_QUERY", dimension=dimension)
            return time.perf_counter() - request_started

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            for latency in executor.map(one, benchmark_rows):
                individual_latencies.append(latency)
        wall = time.perf_counter() - started
        concurrency_results.append(
            {
                "concurrency": concurrency,
                "query_count": len(benchmark_rows),
                "wall_seconds": wall,
                "throughput_queries_per_second": len(benchmark_rows) / wall,
                "request_p50_seconds": _percentile(individual_latencies, 50),
                "request_p95_seconds": _percentile(individual_latencies, 95),
            }
        )
    return {
        "dimension": dimension,
        "batching": batch_results,
        "parallel_single_query_requests": concurrency_results,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if not 1 <= int(args.batch_size) <= 100:
        raise ValueError("--batch-size must be between 1 and Gemini's limit of 100")
    if not 1 <= int(args.multimodal_batch_size) <= 6:
        raise ValueError(
            "--multimodal-batch-size must be between 1 and Gemini's limit of 6 images"
        )
    load_dotenv(PROJECT_ROOT / ".env")
    _configure_rate_limit(args.requests_per_minute)
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    input_manifest_path = experiment_dir / "embedding_inputs/manifest.json"
    input_manifest = _read_json(input_manifest_path)
    documents_path = Path(str(input_manifest["documents_path"]))
    queries_path = Path(str(input_manifest["queries_path"]))
    media_path = Path(str(input_manifest["media_path"]))
    if _sha256_file(documents_path) != input_manifest["documents_sha256"]:
        raise RuntimeError("Document embedding input hash mismatch")
    if _sha256_file(queries_path) != input_manifest["queries_sha256"]:
        raise RuntimeError("Query embedding input hash mismatch")
    if _sha256_file(media_path) != input_manifest["media_sha256"]:
        raise RuntimeError("Media embedding input hash mismatch")
    documents = _read_jsonl(documents_path)
    queries = _read_jsonl(queries_path)
    media = _read_jsonl(media_path)
    token_budget_audit = {
        "documents": _local_token_budget_audit(
            documents, task_type="RETRIEVAL_DOCUMENT"
        ),
        "queries": _local_token_budget_audit(
            queries, task_type="RETRIEVAL_QUERY"
        ),
        "multimodal_media": _local_token_budget_audit(
            media,
            task_type="RETRIEVAL_DOCUMENT",
            image_tokens_each=GEMINI_IMAGE_TOKEN_ALLOWANCE,
        ),
    }
    output_root = experiment_dir / "embeddings"
    primary_root = output_root / "gemini2_1536"
    derived_root = output_root / "gemini2_768"
    primary_multimodal_root = output_root / "gemini2_1536_mm"
    derived_multimodal_root = output_root / "gemini2_768_mm"

    conformance = _validate_mrl(documents)
    document_manifest = _run_collection(
        name="documents",
        rows=documents,
        output_dir=primary_root,
        task_type="RETRIEVAL_DOCUMENT",
        dimension=PRIMARY_DIMENSION,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    query_manifest = _run_collection(
        name="queries",
        rows=queries,
        output_dir=primary_root,
        task_type="RETRIEVAL_QUERY",
        dimension=PRIMARY_DIMENSION,
        batch_size=min(args.batch_size, 32),
        workers=min(args.workers, 4),
    )
    multimodal_conformance = _validate_multimodal_mrl(media)
    media_manifest = _run_collection(
        name="media",
        rows=media,
        output_dir=primary_multimodal_root,
        task_type="RETRIEVAL_DOCUMENT",
        dimension=PRIMARY_DIMENSION,
        batch_size=args.multimodal_batch_size,
        workers=min(args.workers, 4),
        multimodal=True,
    )
    _configure_rate_limit(0.0)
    primary_latency = _query_latency_benchmark(queries, dimension=PRIMARY_DIMENSION)
    primary_manifest = {
        "schema_version": "mbzuai.multilingual.ab_embeddings.v1",
        "created_at_epoch": int(time.time()),
        "provider": "Google Gemini API",
        "model": MODEL,
        "dimension": PRIMARY_DIMENSION,
        "requests_per_minute_limit_during_corpus_embedding": args.requests_per_minute,
        "production_mutation_performed": False,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": _sha256_file(input_manifest_path),
        "documents": document_manifest,
        "queries": query_manifest,
        "query_latency_benchmark": primary_latency,
        "mrl_conformance": conformance,
        "input_token_budget_audit": token_budget_audit,
    }
    _write_json(primary_root / "manifest.json", primary_manifest)

    derived_documents = _derive_collection(
        name="documents",
        source_manifest=document_manifest,
        output_dir=derived_root,
        dimension=DERIVED_DIMENSION,
    )
    derived_queries = _derive_collection(
        name="queries",
        source_manifest=query_manifest,
        output_dir=derived_root,
        dimension=DERIVED_DIMENSION,
    )
    derived_latency = _query_latency_benchmark(queries, dimension=DERIVED_DIMENSION)
    derived_manifest = {
        "schema_version": "mbzuai.multilingual.ab_embeddings.v1",
        "created_at_epoch": int(time.time()),
        "provider": "Google Gemini API",
        "model": MODEL,
        "dimension": DERIVED_DIMENSION,
        "production_mutation_performed": False,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": _sha256_file(input_manifest_path),
        "documents": derived_documents,
        "queries": derived_queries,
        "query_latency_benchmark": derived_latency,
        "derivation_validation": conformance,
        "input_token_budget_audit": token_budget_audit,
    }
    _write_json(derived_root / "manifest.json", derived_manifest)
    derived_media = _derive_collection(
        name="media",
        source_manifest=media_manifest,
        output_dir=derived_multimodal_root,
        dimension=DERIVED_DIMENSION,
    )
    primary_multimodal_manifest = {
        "schema_version": "mbzuai.multilingual.ab_embeddings.v1",
        "created_at_epoch": int(time.time()),
        "provider": "Google Gemini API",
        "model": MODEL,
        "dimension": PRIMARY_DIMENSION,
        "media_input": "image_and_caption_text",
        "base_text_embedding": "gemini2_1536",
        "production_mutation_performed": False,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": _sha256_file(input_manifest_path),
        "media": media_manifest,
        "queries": query_manifest,
        "query_latency_benchmark": primary_latency,
        "mrl_conformance": multimodal_conformance,
        "input_token_budget_audit": token_budget_audit,
    }
    _write_json(
        primary_multimodal_root / "manifest.json", primary_multimodal_manifest
    )
    derived_multimodal_manifest = {
        "schema_version": "mbzuai.multilingual.ab_embeddings.v1",
        "created_at_epoch": int(time.time()),
        "provider": "Google Gemini API",
        "model": MODEL,
        "dimension": DERIVED_DIMENSION,
        "media_input": "image_and_caption_text",
        "base_text_embedding": "gemini2_768",
        "production_mutation_performed": False,
        "input_manifest": str(input_manifest_path),
        "input_manifest_sha256": _sha256_file(input_manifest_path),
        "media": derived_media,
        "queries": derived_queries,
        "query_latency_benchmark": derived_latency,
        "derivation_validation": multimodal_conformance,
        "input_token_budget_audit": token_budget_audit,
    }
    _write_json(
        derived_multimodal_root / "manifest.json", derived_multimodal_manifest
    )
    return {
        "gemini2_1536": primary_manifest,
        "gemini2_768": derived_manifest,
        "gemini2_1536_mm": primary_multimodal_manifest,
        "gemini2_768_mm": derived_multimodal_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed the controlled multilingual A/B corpus with Gemini embedding 2"
    )
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Text inputs per request; Gemini accepts at most 100",
    )
    parser.add_argument("--multimodal-batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--requests-per-minute",
        type=float,
        default=15.0,
        help="Global request-start limit across workers; set 0 to disable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(
        json.dumps(
            {
                key: {
                    "dimension": value["dimension"],
                    "documents": (value.get("documents") or {}).get("row_count"),
                    "media": (value.get("media") or {}).get("row_count"),
                    "queries": value["queries"]["row_count"],
                }
                for key, value in result.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
