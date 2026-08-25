from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.controlled_ab import (
    aggregate_quality,
    annotate_abstention_decisions,
    annotate_gold_presence,
    choose_abstention_threshold,
    evaluate_abstention_threshold,
    paired_bootstrap_delta,
    reciprocal_rank_fusion,
    score_ranking,
    tokenize_multilingual,
)
from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.core.chunking import estimate_token_count, token_counting_method


DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "runs/evaluation/mbzuai-multilingual-controlled-ab-v1"
)
TOP_K = 200
SELECTION_COMPONENTS = {
    "evidence_ndcg_at_10",
    "evidence_mrr_at_10",
    "source_recall_at_10",
    "arabic_evidence_hit_at_10",
    "synthesis_source_recall_at_10",
    "navigation_action_hit_at_5",
    "media_hit_at_5",
    "abstention_balanced_accuracy",
}
COST_ASSUMPTIONS = {
    "scenario": "one production re-index and 1,000,000 cold chat turns per month",
    "monthly_cold_chat_turns": 1_000_000,
    "dense_embedding_inputs_per_cold_turn": 2,
    "measured_mean_tokens_per_embedding_input": 19.33,
    "monthly_reindexes": 1,
    "qwen_current_app_container_usd": 98.0,
    "gemini_backend_container_usd": 49.0,
    "gemini_standard_text_usd_per_million_tokens": 0.20,
    "gemini_batch_text_usd_per_million_tokens": 0.10,
    "gemini_standard_image_usd_each": 0.00012,
    "gemini_batch_image_usd_each": 0.00006,
    "document_tokens_estimation": "multilingual cl100k proxy recorded by the experiment",
    "document_token_counting_method": token_counting_method(),
    "excludes_shared_vector_database_and_generation_costs": True,
    "sources": [
        "https://ai.google.dev/gemini-api/docs/pricing",
        "https://docs.digitalocean.com/products/app-platform/details/pricing/",
    ],
}
EVALUATION_SOURCE_FILES = (
    Path(__file__).resolve(),
    PROJECT_ROOT / "pipeline/evaluation/controlled_ab.py",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _variant_id(config_id: str, embedding_id: str, mode_id: str) -> str:
    return f"{config_id}__{embedding_id}__{mode_id}"


def _evaluation_source_hashes() -> Dict[str, str]:
    return {
        str(path.resolve()): _sha256_file(path.resolve())
        for path in EVALUATION_SOURCE_FILES
    }


def _parse_variant(value: str) -> tuple[str, str, str]:
    parts = str(value).split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid variant id: {value}")
    return parts[0], parts[1], parts[2]


def _top_indices(
    scores: np.ndarray,
    allowed_indices: np.ndarray,
    record_ids: Sequence[str],
    *,
    k: int,
    positive_only: bool = False,
) -> List[tuple[int, float]]:
    if allowed_indices.size == 0:
        return []
    allowed_scores = scores[allowed_indices]
    if positive_only:
        positive = np.flatnonzero(allowed_scores > 0.0)
        if positive.size == 0:
            return []
        allowed_indices = allowed_indices[positive]
        allowed_scores = allowed_scores[positive]
    take = min(k, len(allowed_indices))
    if take < len(allowed_indices):
        selected = np.argpartition(allowed_scores, -take)[-take:]
    else:
        selected = np.arange(len(allowed_indices))
    pairs = [
        (int(allowed_indices[index]), float(allowed_scores[index]))
        for index in selected
    ]
    pairs.sort(key=lambda item: (-item[1], record_ids[item[0]]))
    return pairs[:take]


class QueryScopedBM25:
    def __init__(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        query_vocabulary: set[str],
    ) -> None:
        self.document_count = len(records)
        self.lengths = np.zeros(self.document_count, dtype=np.float32)
        posting_indices: Dict[str, List[int]] = defaultdict(list)
        posting_counts: Dict[str, List[int]] = defaultdict(list)
        for document_index, record in enumerate(records):
            tokens = tokenize_multilingual(record.get("sparse_text") or record.get("text") or "")
            self.lengths[document_index] = len(tokens)
            counts = Counter(token for token in tokens if token in query_vocabulary)
            for token, count in counts.items():
                posting_indices[token].append(document_index)
                posting_counts[token].append(count)
        self.average_length = float(np.mean(self.lengths)) if self.document_count else 0.0
        self.postings = {
            token: (
                np.asarray(posting_indices[token], dtype=np.int32),
                np.asarray(posting_counts[token], dtype=np.float32),
            )
            for token in posting_indices
        }

    def scores(self, query: str, *, k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        output = np.zeros(self.document_count, dtype=np.float32)
        if self.document_count == 0 or self.average_length <= 0.0:
            return output
        query_counts = Counter(tokenize_multilingual(query))
        for token, query_count in query_counts.items():
            posting = self.postings.get(token)
            if posting is None:
                continue
            indices, frequencies = posting
            document_frequency = len(indices)
            idf = math.log(
                1.0
                + (self.document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = frequencies + k1 * (
                1.0 - b + b * self.lengths[indices] / self.average_length
            )
            output[indices] += float(query_count) * idf * (
                frequencies * (k1 + 1.0) / denominator
            )
        return output


def _embedding_paths(experiment_dir: Path, embedding_id: str) -> Dict[str, Path]:
    root = experiment_dir / "embeddings" / embedding_id
    return {
        "root": root,
        "manifest": root / "manifest.json",
        "document_vectors": root / "documents/vectors.npy",
        "document_ids": root / "documents/ids.json",
        "query_vectors": root / "queries/vectors.npy",
        "query_ids": root / "queries/ids.json",
    }


def _load_embedding(
    experiment_dir: Path,
    embedding_id: str,
    embedding_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    expected_dimension = int(embedding_spec["dimensions"])
    input_manifest_path = experiment_dir / "embedding_inputs/manifest.json"
    input_manifest = _read_json(input_manifest_path)
    input_manifest_sha256 = _sha256_file(input_manifest_path)
    base_embedding_id = str(embedding_spec.get("base_text_embedding") or "")
    if base_embedding_id:
        base = _load_embedding(
            experiment_dir,
            base_embedding_id,
            {"dimensions": expected_dimension},
        )
        root = experiment_dir / "embeddings" / embedding_id
        manifest_path = root / "manifest.json"
        media_vectors_path = root / "media/vectors.npy"
        media_ids_path = root / "media/ids.json"
        for path in (manifest_path, media_vectors_path, media_ids_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing {embedding_id} multimodal artifact: {path}"
                )
        media_ids = [str(value) for value in _read_json(media_ids_path)]
        media_vectors = np.load(media_vectors_path, mmap_mode="r")
        if media_vectors.shape != (len(media_ids), expected_dimension):
            raise RuntimeError(
                f"Invalid media vector shape for {embedding_id}: {media_vectors.shape}"
            )
        multimodal_manifest = _read_json(manifest_path)
        if (
            multimodal_manifest.get("input_manifest_sha256")
            != input_manifest_sha256
        ):
            raise RuntimeError(
                f"{embedding_id} was not built from the locked embedding inputs"
            )
        return {
            **base,
            "paths": {"root": root, "manifest": manifest_path},
            "manifest": multimodal_manifest,
            "manifest_sha256": _sha256_file(manifest_path),
            "media_vectors": media_vectors,
            "media_id_to_index": {
                value: index for index, value in enumerate(media_ids)
            },
            "base_embedding_id": base_embedding_id,
        }
    paths = _embedding_paths(experiment_dir, embedding_id)
    for key, path in paths.items():
        if key != "root" and not path.is_file():
            raise FileNotFoundError(f"Missing {embedding_id} embedding artifact: {path}")
    document_ids = [str(value) for value in _read_json(paths["document_ids"])]
    query_ids = [str(value) for value in _read_json(paths["query_ids"])]
    document_vectors = np.load(paths["document_vectors"], mmap_mode="r")
    query_vectors = np.load(paths["query_vectors"], mmap_mode="r")
    embedding_manifest = _read_json(paths["manifest"])
    if embedding_id.startswith("gemini"):
        if embedding_manifest.get("input_manifest_sha256") != input_manifest_sha256:
            raise RuntimeError(
                f"{embedding_id} was not built from the locked embedding inputs"
            )
    else:
        if (
            embedding_manifest.get("documents_input_sha256")
            != input_manifest.get("documents_sha256")
            or embedding_manifest.get("queries_input_sha256")
            != input_manifest.get("queries_sha256")
        ):
            raise RuntimeError(
                f"{embedding_id} input hashes do not match the locked embedding inputs"
            )
    if document_vectors.shape != (len(document_ids), expected_dimension):
        raise RuntimeError(f"Invalid document vector shape for {embedding_id}: {document_vectors.shape}")
    if query_vectors.shape != (len(query_ids), expected_dimension):
        raise RuntimeError(f"Invalid query vector shape for {embedding_id}: {query_vectors.shape}")
    return {
        "paths": paths,
        "manifest": embedding_manifest,
        "document_ids": document_ids,
        "document_id_to_index": {value: index for index, value in enumerate(document_ids)},
        "document_vectors": document_vectors,
        "query_ids": query_ids,
        "query_id_to_index": {value: index for index, value in enumerate(query_ids)},
        "query_vectors": query_vectors,
        "manifest_sha256": _sha256_file(paths["manifest"]),
        "media_vectors": None,
        "media_id_to_index": {},
        "base_embedding_id": "",
    }


def _query_p95_seconds(manifest: Mapping[str, Any]) -> float | None:
    benchmark = manifest.get("query_latency_benchmark") or {}
    for row in benchmark.get("batching") or []:
        if int(row.get("batch_size") or 0) == 1:
            value = row.get("p95_seconds")
            return float(value) if value is not None else None
    return None


def _operational_profile(
    *,
    embedding_id: str,
    embedding_manifest: Mapping[str, Any],
    dimension: int,
    indexed_record_count: int,
    indexed_document_tokens: int,
    indexed_media_count: int,
) -> Dict[str, Any]:
    record_count = int(indexed_record_count)
    vector_bytes = record_count * dimension * 4
    if embedding_id.startswith("gemini"):
        estimated_document_tokens = float(indexed_document_tokens)
        query_tokens = (
            COST_ASSUMPTIONS["monthly_cold_chat_turns"]
            * COST_ASSUMPTIONS["dense_embedding_inputs_per_cold_turn"]
            * COST_ASSUMPTIONS["measured_mean_tokens_per_embedding_input"]
        )
        monthly_cost = (
            COST_ASSUMPTIONS["gemini_backend_container_usd"]
            + query_tokens
            / 1_000_000.0
            * COST_ASSUMPTIONS["gemini_standard_text_usd_per_million_tokens"]
            + estimated_document_tokens
            / 1_000_000.0
            * COST_ASSUMPTIONS["gemini_batch_text_usd_per_million_tokens"]
            * COST_ASSUMPTIONS["monthly_reindexes"]
        )
        monthly_index_cost = (
            estimated_document_tokens
            / 1_000_000.0
            * COST_ASSUMPTIONS["gemini_batch_text_usd_per_million_tokens"]
            * COST_ASSUMPTIONS["monthly_reindexes"]
        )
        if embedding_id.endswith("_mm"):
            monthly_index_cost += (
                indexed_media_count
                * COST_ASSUMPTIONS["gemini_batch_image_usd_each"]
                * COST_ASSUMPTIONS["monthly_reindexes"]
            )
            monthly_cost += (
                indexed_media_count
                * COST_ASSUMPTIONS["gemini_batch_image_usd_each"]
                * COST_ASSUMPTIONS["monthly_reindexes"]
            )
        cost_per_cold_turn = (
            COST_ASSUMPTIONS["dense_embedding_inputs_per_cold_turn"]
            * COST_ASSUMPTIONS["measured_mean_tokens_per_embedding_input"]
            * COST_ASSUMPTIONS["gemini_standard_text_usd_per_million_tokens"]
            / 1_000_000.0
        )
        break_even_turns = max(
            0.0,
            (
                COST_ASSUMPTIONS["qwen_current_app_container_usd"]
                - COST_ASSUMPTIONS["gemini_backend_container_usd"]
                - monthly_index_cost
            )
            / cost_per_cold_turn,
        )
    else:
        estimated_document_tokens = None
        monthly_cost = COST_ASSUMPTIONS["qwen_current_app_container_usd"]
        break_even_turns = None
    return {
        "record_count": record_count,
        "dimension": dimension,
        "vector_bytes": vector_bytes,
        "vector_mebibytes": vector_bytes / (1024.0 * 1024.0),
        "query_embedding_p95_seconds_batch_1": _query_p95_seconds(embedding_manifest),
        "estimated_monthly_cost_usd": monthly_cost,
        "estimated_document_tokens": estimated_document_tokens,
        "estimated_break_even_cold_turns_vs_current_qwen": break_even_turns,
        "indexed_media_count": indexed_media_count,
        "cost_scenario": COST_ASSUMPTIONS["scenario"],
    }


def _rank_for_config(
    *,
    experiment_dir: Path,
    config_id: str,
    candidate_manifest: Mapping[str, Any],
    embedding_specs: Mapping[str, Mapping[str, Any]],
    index_modes: Mapping[str, Mapping[str, Any]],
    examples: Sequence[EvalExample],
    requested_variants: set[str] | None,
) -> Dict[str, Dict[str, Any]]:
    records_path = Path(str(candidate_manifest["records_path"]))
    if _sha256_file(records_path) != candidate_manifest["records_sha256"]:
        raise RuntimeError(f"Candidate corpus hash mismatch: {config_id}")
    records = _read_jsonl(records_path)
    record_ids = [str(row.get("id") or "") for row in records]
    if len(records) != int(candidate_manifest["record_count"]):
        raise RuntimeError(f"Candidate record count mismatch: {config_id}")

    requested_modes = {
        mode_id
        for mode_id in index_modes
        if requested_variants is None
        or any(
            _parse_variant(variant)[0] == config_id
            and _parse_variant(variant)[2] == mode_id
            for variant in requested_variants
        )
    }
    scopes = {
        tuple(str(kind) for kind in index_modes[mode_id].get("record_kinds") or [])
        for mode_id in requested_modes
    }
    allowed_by_scope = {
        scope: np.asarray(
            [index for index, row in enumerate(records) if str(row.get("kind")) in set(scope)],
            dtype=np.int32,
        )
        for scope in scopes
    }
    token_counts = [
        estimate_token_count(str(record.get("text") or "")) for record in records
    ]
    document_tokens_by_scope = {
        scope: sum(token_counts[index] for index in allowed)
        for scope, allowed in allowed_by_scope.items()
    }

    sparse_rankings: Dict[tuple[str, ...], Dict[str, List[tuple[int, float]]]] = {}
    sparse_build_seconds = 0.0
    sparse_query_seconds = 0.0
    if any(bool(index_modes[mode].get("sparse")) for mode in requested_modes):
        query_vocabulary = {
            token for example in examples for token in tokenize_multilingual(example.query)
        }
        started = time.perf_counter()
        bm25 = QueryScopedBM25(records, query_vocabulary=query_vocabulary)
        sparse_build_seconds = time.perf_counter() - started
        sparse_rankings = {scope: {} for scope in scopes}
        started = time.perf_counter()
        for example in examples:
            scores = bm25.scores(example.query)
            for scope, allowed in allowed_by_scope.items():
                sparse_rankings[scope][example.id] = _top_indices(
                    scores,
                    allowed,
                    record_ids,
                    k=TOP_K,
                    positive_only=True,
                )
        sparse_query_seconds = time.perf_counter() - started
        del bm25

    output: Dict[str, Dict[str, Any]] = {}
    for embedding_id, embedding_spec in embedding_specs.items():
        relevant_variants = {
            variant
            for variant in (requested_variants or set())
            if _parse_variant(variant)[:2] == (config_id, embedding_id)
        }
        if requested_variants is not None and not relevant_variants:
            continue
        dimension = int(embedding_spec["dimensions"])
        embedding = _load_embedding(
            experiment_dir, embedding_id, embedding_spec
        )
        try:
            global_document_indices = np.asarray(
                [embedding["document_id_to_index"][str(row["text_sha256"])] for row in records],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing {embedding_id} vector for candidate text {exc}") from exc
        try:
            query_indices = np.asarray(
                [embedding["query_id_to_index"][example.id] for example in examples],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise RuntimeError(f"Missing {embedding_id} query vector {exc}") from exc

        matrix_started = time.perf_counter()
        document_matrix = np.asarray(
            embedding["document_vectors"][global_document_indices], dtype=np.float32
        )
        if embedding["media_vectors"] is not None:
            for record_index, record in enumerate(records):
                if str(record.get("kind") or "") != "media":
                    continue
                media_id = str(record.get("media_id") or record.get("id") or "")
                try:
                    media_index = embedding["media_id_to_index"][media_id]
                except KeyError as exc:
                    raise RuntimeError(
                        f"Missing {embedding_id} native media vector for {media_id}"
                    ) from exc
                document_matrix[record_index] = embedding["media_vectors"][
                    media_index
                ]
        query_matrix = np.asarray(
            embedding["query_vectors"][query_indices], dtype=np.float32
        )
        matrix_materialization_seconds = time.perf_counter() - matrix_started
        dense_started = time.perf_counter()
        dense_scores = query_matrix @ document_matrix.T
        dense_compute_seconds = time.perf_counter() - dense_started
        dense_rankings: Dict[tuple[str, ...], Dict[str, List[tuple[int, float]]]] = {
            scope: {} for scope in scopes
        }
        rank_started = time.perf_counter()
        for query_index, example in enumerate(examples):
            for scope, allowed in allowed_by_scope.items():
                dense_rankings[scope][example.id] = _top_indices(
                    dense_scores[query_index], allowed, record_ids, k=TOP_K
                )
        dense_rank_seconds = time.perf_counter() - rank_started

        for mode_id in requested_modes:
            variant = _variant_id(config_id, embedding_id, mode_id)
            if requested_variants is not None and variant not in requested_variants:
                continue
            mode = index_modes[mode_id]
            scope = tuple(str(kind) for kind in mode.get("record_kinds") or [])
            query_rows: List[Dict[str, Any]] = []
            for example in examples:
                dense = dense_rankings[scope][example.id]
                if mode.get("sparse"):
                    sparse = sparse_rankings[scope][example.id]
                    fused = reciprocal_rank_fusion(
                        [[index for index, _ in dense], [index for index, _ in sparse]],
                        top_k_per_ranking=int(mode.get("dense_top_k") or 100),
                        rrf_k=int(mode.get("rrf_k") or 60),
                        output_k=TOP_K,
                    )
                    ranked = [(records[index], score) for index, score in fused]
                else:
                    ranked = [(records[index], score) for index, score in dense]
                row = score_ranking(example, ranked)
                query_rows.append(annotate_gold_presence(row, example))
            output[variant] = {
                "variant_id": variant,
                "chunk_config": config_id,
                "embedding": embedding_id,
                "index_mode": mode_id,
                "per_query": query_rows,
                "runtime": {
                    "sparse_build_seconds_shared_by_config": sparse_build_seconds,
                    "sparse_query_seconds_shared_by_config": sparse_query_seconds,
                    "matrix_materialization_seconds": matrix_materialization_seconds,
                    "dense_batch_compute_seconds": dense_compute_seconds,
                    "dense_ranking_seconds": dense_rank_seconds,
                    "evaluated_query_count": len(examples),
                },
                "operational": _operational_profile(
                    embedding_id=embedding_id,
                    embedding_manifest=embedding["manifest"],
                    dimension=dimension,
                    indexed_record_count=len(allowed_by_scope[scope]),
                    indexed_document_tokens=document_tokens_by_scope[scope],
                    indexed_media_count=sum(
                        1
                        for index in allowed_by_scope[scope]
                        if str(records[index].get("kind") or "") == "media"
                    ),
                ),
                "embedding_manifest_sha256": embedding["manifest_sha256"],
            }
        del dense_scores, document_matrix, query_matrix
    return output


def _score_variants(
    *,
    experiment_dir: Path,
    manifest: Mapping[str, Any],
    examples: Sequence[EvalExample],
    requested_variants: set[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    controlled = manifest["controlled_variables"]
    embedding_specs = controlled["embedding_specs"]
    index_modes = controlled["index_modes"]
    candidate_manifests = manifest["candidate_manifests"]
    output: Dict[str, Dict[str, Any]] = {}
    for config_id, candidate_manifest in candidate_manifests.items():
        if requested_variants is not None and not any(
            _parse_variant(variant)[0] == config_id for variant in requested_variants
        ):
            continue
        print(f"Scoring {config_id} on {len(examples)} queries", flush=True)
        output.update(
            _rank_for_config(
                experiment_dir=experiment_dir,
                config_id=config_id,
                candidate_manifest=candidate_manifest,
                embedding_specs=embedding_specs,
                index_modes=index_modes,
                examples=examples,
                requested_variants=requested_variants,
            )
        )
    weights = controlled["selection_policy"]["selection_score_weights"]
    for result in output.values():
        _attach_quality_summaries(result, weights=weights)
    return output


def _attach_quality_summaries(
    result: Dict[str, Any], *, weights: Mapping[str, float]
) -> None:
    result["quality"] = aggregate_quality(result["per_query"], weights=weights)
    result["quality_slices"] = {
        "by_language": {
            value: aggregate_quality(
                [row for row in result["per_query"] if row.get("language") == value],
                weights=weights,
            )
            for value in sorted(
                {str(row.get("language")) for row in result["per_query"]}
            )
        },
        "by_query_type": {
            value: aggregate_quality(
                [row for row in result["per_query"] if row.get("query_type") == value],
                weights=weights,
            )
            for value in sorted(
                {str(row.get("query_type")) for row in result["per_query"]}
            )
        },
        "by_source_type": {
            value: aggregate_quality(
                [row for row in result["per_query"] if row.get("source_type") == value],
                weights=weights,
            )
            for value in sorted(
                {str(row.get("source_type")) for row in result["per_query"]}
            )
        },
    }


def _dataset_context(manifest: Mapping[str, Any]) -> tuple[Path, List[EvalExample]]:
    dataset = manifest["evaluation_dataset"]
    path = Path(str(dataset["path"]))
    if _sha256_file(path) != dataset["sha256"]:
        raise RuntimeError("Evaluation dataset hash differs from the experiment lock")
    return path, load_eval_examples(path)


def _protocol_context(experiment_dir: Path) -> tuple[Path, Dict[str, Any]]:
    manifest_path = experiment_dir / "experiment_manifest.json"
    manifest = _read_json(manifest_path)
    required_configs = set(manifest["controlled_variables"]["chunk_configs"])
    if set(manifest.get("candidate_manifests") or {}) != required_configs:
        raise RuntimeError("Not every preregistered chunk candidate was prepared")
    weights = manifest["controlled_variables"]["selection_policy"][
        "selection_score_weights"
    ]
    if set(weights) != SELECTION_COMPONENTS:
        raise RuntimeError("Selection score components differ from the frozen contract")
    if abs(sum(float(value) for value in weights.values()) - 1.0) > 1e-9:
        raise RuntimeError("Selection score weights must sum to one")
    return manifest_path, manifest


def _selection_phase(experiment_dir: Path) -> Dict[str, Any]:
    manifest_path, manifest = _protocol_context(experiment_dir)
    dataset_path, all_examples = _dataset_context(manifest)
    policy = manifest["controlled_variables"]["selection_policy"]
    examples = [
        row
        for row in all_examples
        if str((row.metadata or {}).get("split") or "") == policy["selection_split"]
    ]
    if not examples:
        raise RuntimeError("Selection split is empty")
    results = _score_variants(
        experiment_dir=experiment_dir, manifest=manifest, examples=examples
    )
    baseline_id = str(policy["baseline_variant"])
    if baseline_id not in results:
        raise RuntimeError(f"Missing baseline variant {baseline_id}")
    weights = policy["selection_score_weights"]
    for result in results.values():
        result["abstention"] = choose_abstention_threshold(result["per_query"])
        annotate_abstention_decisions(
            result["per_query"], float(result["abstention"]["threshold"])
        )
        _attach_quality_summaries(result, weights=weights)
    baseline_rows = results[baseline_id]["per_query"]
    for variant_id, result in results.items():
        result["paired_bootstrap_vs_baseline"] = paired_bootstrap_delta(
            result["per_query"],
            baseline_rows,
            weights=weights,
            samples=int(policy["paired_bootstrap_samples"]),
            seed=int(policy["bootstrap_seed"]),
        )
    ordered = sorted(
        results,
        key=lambda key: (
            -float(results[key]["quality"]["selection_score"]),
            float(results[key]["operational"]["estimated_monthly_cost_usd"]),
            float(results[key]["operational"]["vector_bytes"]),
            key,
        ),
    )
    finalist_ids = ordered[: int(policy["finalist_count"])]
    finalist_pairwise_bootstrap = {
        f"{left}__versus__{right}": paired_bootstrap_delta(
            results[left]["per_query"],
            results[right]["per_query"],
            weights=weights,
            samples=int(policy["paired_bootstrap_samples"]),
            seed=int(policy["bootstrap_seed"]),
        )
        for left, right in combinations(finalist_ids, 2)
    }
    output_dir = experiment_dir / "results"
    selection_payload = {
        "schema_version": "mbzuai.multilingual.controlled_ab.selection.v1",
        "created_at_epoch": int(time.time()),
        "phase": "selection",
        "holdout_opened": False,
        "production_mutation_performed": False,
        "experiment_manifest": str(manifest_path),
        "experiment_manifest_sha256": _sha256_file(manifest_path),
        "dataset": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "query_count": len(examples),
        "cost_assumptions": COST_ASSUMPTIONS,
        "evaluation_source_sha256": _evaluation_source_hashes(),
        "ordered_variants": ordered,
        "finalist_pairwise_bootstrap": finalist_pairwise_bootstrap,
        "results": results,
    }
    selection_path = output_dir / "selection_results.json"
    _write_json(selection_path, selection_payload)
    threshold_by_variant = {
        variant_id: float(results[variant_id]["abstention"]["threshold"])
        for variant_id in set(finalist_ids) | {baseline_id}
    }
    frozen = {
        "schema_version": "mbzuai.multilingual.controlled_ab.frozen_finalists.v1",
        "created_at_epoch": int(time.time()),
        "holdout_opened": False,
        "production_mutation_performed": False,
        "experiment_manifest_sha256": _sha256_file(manifest_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "selection_results": str(selection_path),
        "selection_results_sha256": _sha256_file(selection_path),
        "baseline_variant": baseline_id,
        "finalists": finalist_ids,
        "selection_scores": {
            variant_id: results[variant_id]["quality"]["selection_score"]
            for variant_id in finalist_ids
        },
        "finalist_pairwise_bootstrap": finalist_pairwise_bootstrap,
        "frozen_abstention_thresholds": threshold_by_variant,
        "selection_policy": policy,
        "evaluation_source_sha256": _evaluation_source_hashes(),
    }
    frozen_path = output_dir / "frozen_finalists.json"
    _write_json(frozen_path, frozen)
    print(f"Frozen finalists: {finalist_ids}", flush=True)
    return frozen


def _load_and_validate_freeze(
    experiment_dir: Path, manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[Path, Dict[str, Any]]:
    frozen_path = experiment_dir / "results/frozen_finalists.json"
    frozen = _read_json(frozen_path)
    if frozen.get("experiment_manifest_sha256") != _sha256_file(manifest_path):
        raise RuntimeError("Experiment manifest changed after finalist freeze")
    dataset_path, _ = _dataset_context(manifest)
    if frozen.get("dataset_sha256") != _sha256_file(dataset_path):
        raise RuntimeError("Dataset changed after finalist freeze")
    selection_path = Path(str(frozen["selection_results"]))
    if frozen.get("selection_results_sha256") != _sha256_file(selection_path):
        raise RuntimeError("Selection results changed after finalist freeze")
    if frozen.get("evaluation_source_sha256") != _evaluation_source_hashes():
        raise RuntimeError("Evaluation scoring code changed after finalist freeze")
    return frozen_path, frozen


def _regression_phase(experiment_dir: Path) -> Dict[str, Any]:
    manifest_path, manifest = _protocol_context(experiment_dir)
    frozen_path, frozen = _load_and_validate_freeze(
        experiment_dir, manifest_path, manifest
    )
    _dataset_path, all_examples = _dataset_context(manifest)
    policy = frozen["selection_policy"]
    examples = [
        row
        for row in all_examples
        if str((row.metadata or {}).get("split") or "") == policy["regression_split"]
    ]
    baseline_id = str(frozen["baseline_variant"])
    requested = set(str(value) for value in frozen["finalists"]) | {baseline_id}
    results = _score_variants(
        experiment_dir=experiment_dir,
        manifest=manifest,
        examples=examples,
        requested_variants=requested,
    )
    thresholds = frozen["frozen_abstention_thresholds"]
    for variant_id, result in results.items():
        result["abstention"] = evaluate_abstention_threshold(
            result["per_query"], float(thresholds[variant_id])
        )
        annotate_abstention_decisions(
            result["per_query"], float(thresholds[variant_id])
        )
        _attach_quality_summaries(
            result, weights=policy["selection_score_weights"]
        )
    baseline_score = float(results[baseline_id]["quality"]["selection_score"])
    maximum_drop = float(policy["regression_max_absolute_drop"])
    gate_rows = {}
    passed = []
    for variant_id in frozen["finalists"]:
        score = float(results[variant_id]["quality"]["selection_score"])
        delta = score - baseline_score
        did_pass = delta >= -maximum_drop
        gate_rows[variant_id] = {
            "regression_score": score,
            "baseline_score": baseline_score,
            "delta": delta,
            "minimum_allowed_delta": -maximum_drop,
            "passed": did_pass,
        }
        if did_pass:
            passed.append(variant_id)
    if not passed:
        passed = [baseline_id]
    payload = {
        "schema_version": "mbzuai.multilingual.controlled_ab.regression_gate.v1",
        "created_at_epoch": int(time.time()),
        "phase": "regression",
        "holdout_opened": False,
        "production_mutation_performed": False,
        "frozen_finalists": str(frozen_path),
        "frozen_finalists_sha256": _sha256_file(frozen_path),
        "query_count": len(examples),
        "gate": gate_rows,
        "passed_finalists": passed,
        "results": results,
    }
    output_path = experiment_dir / "results/regression_gate.json"
    _write_json(output_path, payload)
    print(f"Regression-passing finalists: {passed}", flush=True)
    return payload


def _holdout_phase(experiment_dir: Path) -> Dict[str, Any]:
    manifest_path, manifest = _protocol_context(experiment_dir)
    frozen_path, frozen = _load_and_validate_freeze(
        experiment_dir, manifest_path, manifest
    )
    gate_path = experiment_dir / "results/regression_gate.json"
    gate = _read_json(gate_path)
    if gate.get("frozen_finalists_sha256") != _sha256_file(frozen_path):
        raise RuntimeError("Frozen-finalist file changed after regression gating")
    _dataset_path, all_examples = _dataset_context(manifest)
    policy = frozen["selection_policy"]
    examples = [
        row
        for row in all_examples
        if str((row.metadata or {}).get("split") or "")
        == policy["sealed_holdout_split"]
    ]
    requested = set(str(value) for value in gate["passed_finalists"])
    if not requested:
        raise RuntimeError("No finalist passed the regression gate")
    results = _score_variants(
        experiment_dir=experiment_dir,
        manifest=manifest,
        examples=examples,
        requested_variants=requested,
    )
    thresholds = frozen["frozen_abstention_thresholds"]
    for variant_id, result in results.items():
        result["abstention"] = evaluate_abstention_threshold(
            result["per_query"], float(thresholds[variant_id])
        )
        annotate_abstention_decisions(
            result["per_query"], float(thresholds[variant_id])
        )
        _attach_quality_summaries(
            result, weights=policy["selection_score_weights"]
        )
    best_quality = max(
        float(result["quality"]["selection_score"]) for result in results.values()
    )
    margin = float(policy["holdout_tie_margin"])
    tied = [
        variant_id
        for variant_id, result in results.items()
        if best_quality - float(result["quality"]["selection_score"]) <= margin
    ]
    tied.sort(
        key=lambda variant_id: (
            float(results[variant_id]["operational"]["estimated_monthly_cost_usd"]),
            float(
                results[variant_id]["operational"].get(
                    "query_embedding_p95_seconds_batch_1"
                )
                or float("inf")
            ),
            int(results[variant_id]["operational"]["vector_bytes"]),
            -float(results[variant_id]["quality"]["selection_score"]),
            variant_id,
        )
    )
    winner_id = tied[0]
    config_id, embedding_id, mode_id = _parse_variant(winner_id)
    payload = {
        "schema_version": "mbzuai.multilingual.controlled_ab.final_selection.v1",
        "created_at_epoch": int(time.time()),
        "phase": "holdout",
        "holdout_opened_after_finalists_frozen_and_regression_gated": True,
        "production_mutation_performed": False,
        "frozen_finalists_sha256": _sha256_file(frozen_path),
        "regression_gate": str(gate_path),
        "regression_gate_sha256": _sha256_file(gate_path),
        "query_count": len(examples),
        "quality_tie_margin": margin,
        "quality_tied_finalists": tied,
        "tie_break_order": policy["tie_break_order"],
        "cost_assumptions": COST_ASSUMPTIONS,
        "winner": {
            "variant_id": winner_id,
            "chunk_config_id": config_id,
            "chunk_config": manifest["controlled_variables"]["chunk_configs"][config_id],
            "embedding_id": embedding_id,
            "embedding_spec": manifest["controlled_variables"]["embedding_specs"][embedding_id],
            "index_mode_id": mode_id,
            "index_mode": manifest["controlled_variables"]["index_modes"][mode_id],
            "holdout_quality": results[winner_id]["quality"],
            "holdout_abstention": results[winner_id]["abstention"],
            "operational": results[winner_id]["operational"],
        },
        "results": results,
        "selection_is_recommendation_only": True,
        "production_promotion_authorized": False,
    }
    output_path = experiment_dir / "results/final_selection.json"
    _write_json(output_path, payload)
    print(f"Final selected variant: {winner_id}", flush=True)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the gated controlled multilingual retrieval A/B evaluation"
    )
    parser.add_argument("phase", choices=("selection", "regression", "holdout"))
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_dir = Path(args.experiment_dir).expanduser().resolve()
    if args.phase == "selection":
        result = _selection_phase(experiment_dir)
    elif args.phase == "regression":
        result = _regression_phase(experiment_dir)
    else:
        result = _holdout_phase(experiment_dir)
    print(json.dumps({"phase": args.phase, "schema_version": result["schema_version"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
