from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.evaluation.multilingual_v2 import normalize_evidence_text

DEFAULT_EXPERIMENT_DIR = (
    PROJECT_ROOT / "runs/evaluation/mbzuai-multilingual-controlled-ab-v1"
)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object on line {line_number} of {path}")
            yield row


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
            count += 1
    return count


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_evidence_coverage(
    records_path: Path,
    examples: Sequence[EvalExample],
    *,
    core_kinds: set[str],
) -> Dict[str, Any]:
    records_by_source: Dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for row in _read_jsonl(records_path):
        kind = str(row.get("kind") or "")
        action_id = str(row.get("action_id") or "")
        media_id = str(row.get("media_id") or "")
        evidence_text = normalize_evidence_text(
            f"{row.get('raw_text') or ''}\n{row.get('text') or ''}"
        )
        source_keys = [
            str(row.get("id") or ""),
            str(row.get("document_revision_id") or ""),
            action_id,
            media_id,
            *(str(value) for value in row.get("page_card_ids") or []),
            *(str(value) for value in row.get("section_ids") or []),
        ]
        if media_id:
            source_keys.append(f"media:{media_id}")
        item = (kind, evidence_text, action_id, media_id)
        for source_key in dict.fromkeys(value for value in source_keys if value):
            records_by_source[source_key].append(item)

    evidence_unit_count = 0
    graph_missing: list[Dict[str, str]] = []
    core_missing: list[Dict[str, str]] = []
    for example in examples:
        gold_actions = set(example.gold_action_ids)
        gold_media = set(example.gold_media_ids)
        for evidence in (example.metadata or {}).get("evidence_quotes") or []:
            if not isinstance(evidence, Mapping):
                continue
            source_key = str(evidence.get("source_key") or "")
            quote = normalize_evidence_text(evidence.get("quote"))
            if not source_key or not quote:
                continue
            evidence_unit_count += 1
            matched_kinds = []
            for kind, evidence_text, action_id, media_id in records_by_source.get(
                source_key, []
            ):
                if (
                    quote in evidence_text
                    or (action_id and action_id in gold_actions)
                    or (media_id and media_id in gold_media)
                ):
                    matched_kinds.append(kind)
            issue = {
                "id": example.id,
                "source_key": source_key,
                "quote_prefix": quote[:120],
            }
            if not matched_kinds:
                graph_missing.append(issue)
            if not any(kind in core_kinds for kind in matched_kinds):
                core_missing.append(issue)
    return {
        "evidence_unit_count": evidence_unit_count,
        "graph_reachable_evidence_units": evidence_unit_count - len(graph_missing),
        "graph_missing_count": len(graph_missing),
        "graph_missing_samples": graph_missing[:20],
        "core_reachable_evidence_units": evidence_unit_count - len(core_missing),
        "core_missing_count": len(core_missing),
        "core_missing_samples": core_missing[:20],
        "passed": not graph_missing,
    }


def build(experiment_dir: Path) -> Dict[str, Any]:
    experiment_dir = experiment_dir.resolve()
    experiment_manifest_path = experiment_dir / "experiment_manifest.json"
    manifest = _read_json(experiment_manifest_path)
    candidates = manifest.get("candidate_manifests") or {}
    dataset_path = Path(str((manifest.get("evaluation_dataset") or {}).get("path") or ""))
    dataset_sha256 = str(
        (manifest.get("evaluation_dataset") or {}).get("sha256") or ""
    )
    if not dataset_path.is_file() or _sha256_file(dataset_path) != dataset_sha256:
        raise RuntimeError(f"Evaluation dataset hash mismatch: {dataset_path}")
    examples = load_eval_examples(dataset_path)
    core_kinds = set(
        ((manifest.get("controlled_variables") or {}).get("index_modes") or {})
        .get("dense_core", {})
        .get("record_kinds", [])
    )
    expected_configs = set(
        (manifest.get("controlled_variables") or {}).get("chunk_configs") or {}
    )
    if set(candidates) != expected_configs:
        raise RuntimeError(
            "Embedding inputs require every preregistered chunk candidate; "
            f"expected {sorted(expected_configs)}, got {sorted(candidates)}"
        )

    unique_documents: Dict[str, Dict[str, Any]] = {}
    unique_media: Dict[str, Dict[str, Any]] = {}
    per_config: Dict[str, Dict[str, Any]] = {}
    evidence_coverage: Dict[str, Dict[str, Any]] = {}
    for config_id in sorted(candidates):
        records_path = Path(str(candidates[config_id]["records_path"])).resolve()
        expected_sha = str(candidates[config_id]["records_sha256"])
        actual_sha = _sha256_file(records_path)
        if actual_sha != expected_sha:
            raise RuntimeError(f"Candidate hash mismatch for {config_id}: {records_path}")
        record_count = 0
        new_text_count = 0
        for row in _read_jsonl(records_path):
            record_count += 1
            text = str(row.get("text") or "").strip()
            text_sha = str(row.get("text_sha256") or "")
            if not text or not text_sha:
                raise ValueError(f"Empty embedding text in {records_path}: {row.get('id')}")
            calculated_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if calculated_sha != text_sha:
                raise ValueError(f"Invalid text hash in {records_path}: {row.get('id')}")
            existing = unique_documents.get(text_sha)
            if existing is not None and existing["text"] != text:
                raise RuntimeError(f"SHA-256 collision for embedding text {text_sha}")
            if existing is None:
                unique_documents[text_sha] = {
                    "id": text_sha,
                    "text": text,
                    "representative_title": str(row.get("title") or ""),
                    "representative_kind": str(row.get("kind") or ""),
                }
                new_text_count += 1
            if str(row.get("kind") or "") == "media":
                media_id = str(row.get("media_id") or row.get("id") or "")
                local_path = Path(str(row.get("local_path") or "")).expanduser().resolve()
                if not media_id or not local_path.is_file():
                    raise ValueError(
                        f"Media record {row.get('id')} has no readable local image"
                    )
                media_input = {
                    "id": media_id,
                    "text": text,
                    "local_path": str(local_path),
                    "content_hash": str(
                        (row.get("metadata") or {}).get("content_hash") or ""
                    ),
                }
                prior_media = unique_media.get(media_id)
                if prior_media is not None and prior_media != media_input:
                    raise RuntimeError(
                        f"Media input {media_id} differs across chunk candidates"
                    )
                unique_media[media_id] = media_input
        if record_count != int(candidates[config_id]["record_count"]):
            raise RuntimeError(
                f"Candidate row-count mismatch for {config_id}: "
                f"{record_count} != {candidates[config_id]['record_count']}"
            )
        per_config[config_id] = {
            "record_count": record_count,
            "new_unique_text_count": new_text_count,
            "records_sha256": actual_sha,
        }
        evidence_coverage[config_id] = _candidate_evidence_coverage(
            records_path, examples, core_kinds=core_kinds
        )
        if not evidence_coverage[config_id]["passed"]:
            raise RuntimeError(
                f"Candidate {config_id} cannot represent every gold evidence unit: "
                f"{evidence_coverage[config_id]['graph_missing_samples']}"
            )

    queries_path = Path(
        str((manifest.get("evaluation_dataset") or {}).get("query_file") or "")
    ).resolve()
    expected_query_sha = str(
        (manifest.get("evaluation_dataset") or {}).get("query_file_sha256") or ""
    )
    if _sha256_file(queries_path) != expected_query_sha:
        raise RuntimeError(f"Query hash mismatch: {queries_path}")
    queries = []
    seen_query_ids = set()
    for row in _read_jsonl(queries_path):
        query_id = str(row.get("id") or "")
        query = str(row.get("query") or "").strip()
        if not query_id or not query or query_id in seen_query_ids:
            raise ValueError(f"Invalid or duplicate query row: {query_id!r}")
        seen_query_ids.add(query_id)
        queries.append({"id": query_id, "text": query})

    inputs_dir = experiment_dir / "embedding_inputs"
    documents_path = inputs_dir / "documents.jsonl"
    embedding_queries_path = inputs_dir / "queries.jsonl"
    media_path = inputs_dir / "media.jsonl"
    document_count = _write_jsonl(
        documents_path,
        (unique_documents[key] for key in sorted(unique_documents)),
    )
    query_count = _write_jsonl(embedding_queries_path, queries)
    media_count = _write_jsonl(
        media_path, (unique_media[key] for key in sorted(unique_media))
    )
    output = {
        "schema_version": "mbzuai.multilingual.ab_embedding_inputs.v1",
        "created_at_epoch": int(time.time()),
        "production_mutation_performed": False,
        "experiment_manifest": str(experiment_manifest_path),
        "experiment_manifest_sha256": _sha256_file(experiment_manifest_path),
        "documents_path": str(documents_path),
        "documents_sha256": _sha256_file(documents_path),
        "unique_document_text_count": document_count,
        "queries_path": str(embedding_queries_path),
        "queries_sha256": _sha256_file(embedding_queries_path),
        "query_count": query_count,
        "media_path": str(media_path),
        "media_sha256": _sha256_file(media_path),
        "media_count": media_count,
        "per_config": per_config,
        "candidate_evidence_coverage": evidence_coverage,
    }
    _write_json(inputs_dir / "manifest.json", output)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deduplicated embedding inputs for the controlled multilingual A/B run"
    )
    parser.add_argument("--experiment-dir", default=str(DEFAULT_EXPERIMENT_DIR))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build(Path(args.experiment_dir).expanduser())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
