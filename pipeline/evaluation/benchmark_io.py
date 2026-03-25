from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from pipeline.evaluation.retrieval_eval import _hit_at_k, _mrr_at_k, _ndcg_at_k, _recall_at_k


@dataclass
class BenchmarkDocument:
    id: str
    text: str
    title: str = ""
    metadata: Dict[str, Any] | None = None


@dataclass
class BenchmarkQuery:
    id: str
    text: str
    metadata: Dict[str, Any] | None = None


@dataclass
class BenchmarkQrel:
    query_id: str
    doc_id: str
    relevance: float = 1.0


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), ensure_ascii=True) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL in {path} on line {line_number}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object rows in {path}, got {type(payload).__name__}")
        rows.append(payload)
    return rows


def _load_rankings(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("{") and "\n" not in text:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            if any(key in payload for key in ("query_id", "id", "ranked_ids", "doc_ids")):
                return [payload]
            rows: List[Dict[str, Any]] = []
            for query_id, value in payload.items():
                ranked_ids: List[str] = []
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            doc_id = str(item.get("doc_id") or item.get("id") or "").strip()
                            if doc_id:
                                ranked_ids.append(doc_id)
                        else:
                            doc_id = str(item).strip()
                            if doc_id:
                                ranked_ids.append(doc_id)
                rows.append({"query_id": str(query_id).strip(), "ranked_ids": ranked_ids})
            return rows
    return _load_jsonl(path)


def summarize_standard_benchmark(dataset_dir: str | Path) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    corpus_rows = _load_jsonl(dataset_dir / "corpus.jsonl")
    query_rows = _load_jsonl(dataset_dir / "queries.jsonl")
    qrel_rows = _load_jsonl(dataset_dir / "qrels.jsonl")
    query_ids = {str(row.get("id") or "") for row in query_rows if str(row.get("id") or "")}
    doc_ids = {str(row.get("id") or "") for row in corpus_rows if str(row.get("id") or "")}
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "document_count": len(doc_ids),
        "query_count": len(query_ids),
        "qrel_count": len(qrel_rows),
        "queries_with_qrels": len({str(row.get("query_id") or "") for row in qrel_rows if str(row.get("query_id") or "")}),
    }


def evaluate_standard_rankings(
    *,
    dataset_dir: str | Path,
    rankings_path: str | Path,
    k: int = 10,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    ranking_rows = _load_rankings(Path(rankings_path))
    qrel_rows = _load_jsonl(dataset_dir / "qrels.jsonl")
    queries = _load_jsonl(dataset_dir / "queries.jsonl")

    gold_by_query: Dict[str, List[str]] = {}
    for row in qrel_rows:
        query_id = str(row.get("query_id") or "").strip()
        doc_id = str(row.get("doc_id") or "").strip()
        relevance = float(row.get("relevance") or 0.0)
        if not query_id or not doc_id or relevance <= 0.0:
            continue
        gold_by_query.setdefault(query_id, []).append(doc_id)

    rankings_by_query: Dict[str, List[str]] = {}
    for row in ranking_rows:
        query_id = str(row.get("query_id") or row.get("id") or "").strip()
        ranked_ids = row.get("ranked_ids") or row.get("doc_ids") or []
        if not query_id or not isinstance(ranked_ids, list):
            continue
        rankings_by_query[query_id] = [str(value).strip() for value in ranked_ids if str(value).strip()]

    query_ids = [str(row.get("id") or "").strip() for row in queries if str(row.get("id") or "").strip()]
    scores: List[Dict[str, Any]] = []
    for query_id in query_ids:
        gold_ids = gold_by_query.get(query_id, [])
        ranked_ids = rankings_by_query.get(query_id, [])
        scores.append(
            {
                "query_id": query_id,
                "has_qrels": bool(gold_ids),
                "hit_at_k": _hit_at_k(ranked_ids, gold_ids, k),
                "recall_at_k": _recall_at_k(ranked_ids, gold_ids, k),
                "mrr_at_k": _mrr_at_k(ranked_ids, gold_ids, k),
                "ndcg_at_k": _ndcg_at_k(ranked_ids, gold_ids, k),
            }
        )

    eligible = [row for row in scores if row["has_qrels"]]
    metric_names = ("hit_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k")
    overall = {
        "query_count": len(query_ids),
        "eligible_query_count": len(eligible),
        **{
            metric: (
                sum(float(row[metric]) for row in eligible) / float(len(eligible))
                if eligible
                else 0.0
            )
            for metric in metric_names
        },
    }
    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "rankings_path": str(Path(rankings_path).resolve()),
        "k": int(k),
        "overall": overall,
        "queries": scores,
    }


def export_ir_datasets_benchmark(
    *,
    dataset_id: str,
    output_dir: str | Path,
    max_queries: int | None = None,
    max_docs: int | None = None,
    full_corpus: bool = False,
) -> Dict[str, Any]:
    try:
        import ir_datasets
    except Exception as exc:  # pragma: no cover - dependency is present in env
        raise RuntimeError("ir_datasets is required for export-ir-benchmark") from exc

    dataset = ir_datasets.load(dataset_id)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    query_rows: List[Dict[str, Any]] = []
    query_id_set = set()
    for idx, query in enumerate(dataset.queries_iter()):
        if max_queries is not None and idx >= int(max_queries):
            break
        query_id = str(getattr(query, "query_id", None) or getattr(query, "doc_id", None) or getattr(query, "id", None) or "")
        query_text = str(getattr(query, "text", None) or getattr(query, "query", None) or "")
        if not query_id or not query_text:
            continue
        query_id_set.add(query_id)
        query_rows.append({"id": query_id, "text": query_text})

    qrel_rows: List[Dict[str, Any]] = []
    relevant_doc_ids = set()
    for qrel in dataset.qrels_iter():
        query_id = str(getattr(qrel, "query_id", None) or "")
        doc_id = str(getattr(qrel, "doc_id", None) or "")
        relevance = float(getattr(qrel, "relevance", 0.0) or 0.0)
        if not query_id or not doc_id:
            continue
        if query_id_set and query_id not in query_id_set:
            continue
        qrel_rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": relevance})
        if relevance > 0.0:
            relevant_doc_ids.add(doc_id)

    corpus_rows: List[Dict[str, Any]] = []
    exported_doc_ids = set()
    for idx, doc in enumerate(dataset.docs_iter()):
        doc_id = str(getattr(doc, "doc_id", None) or getattr(doc, "id", None) or "")
        if not doc_id:
            continue
        if not full_corpus and doc_id not in relevant_doc_ids:
            continue
        if max_docs is not None and len(corpus_rows) >= int(max_docs):
            break
        title = str(getattr(doc, "title", None) or "")
        text = str(getattr(doc, "text", None) or getattr(doc, "body", None) or "")
        if not text and not title:
            continue
        corpus_rows.append({"id": doc_id, "title": title, "text": text})
        exported_doc_ids.add(doc_id)

    qrel_rows = [row for row in qrel_rows if row["doc_id"] in exported_doc_ids]

    _write_jsonl(output_dir / "corpus.jsonl", corpus_rows)
    _write_jsonl(output_dir / "queries.jsonl", query_rows)
    _write_jsonl(output_dir / "qrels.jsonl", qrel_rows)

    metadata = {
        "source": "ir_datasets",
        "dataset_id": dataset_id,
        "full_corpus": bool(full_corpus),
        "max_queries": max_queries,
        "max_docs": max_docs,
        **summarize_standard_benchmark(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def export_hf_benchmark(
    *,
    mapping_path: str | Path,
    output_dir: str | Path,
    max_queries: int | None = None,
    max_docs: int | None = None,
    max_qrels: int | None = None,
) -> Dict[str, Any]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency is present in env
        raise RuntimeError("datasets is required for export-hf-benchmark") from exc

    mapping = json.loads(Path(mapping_path).read_text(encoding="utf-8"))
    dataset_name = str(mapping.get("dataset_name") or "").strip()
    if not dataset_name:
        raise ValueError("dataset_name is required in mapping file")

    def _load_split(spec: Dict[str, Any]):
        return load_dataset(
            dataset_name,
            name=spec.get("config"),
            split=spec.get("split"),
        )

    corpus_spec = dict(mapping.get("corpus") or {})
    query_spec = dict(mapping.get("queries") or {})
    qrel_spec = dict(mapping.get("qrels") or {})
    for spec_name, spec in (("corpus", corpus_spec), ("queries", query_spec), ("qrels", qrel_spec)):
        if not spec.get("split"):
            raise ValueError(f"{spec_name}.split is required in mapping file")

    corpus_ds = _load_split(corpus_spec)
    query_ds = _load_split(query_spec)
    qrel_ds = _load_split(qrel_spec)

    def _row_value(row: Dict[str, Any], key: str, *, required: bool = True) -> str:
        value = row.get(key)
        if value is None:
            if required:
                raise ValueError(f"Missing field {key!r} in row {row}")
            return ""
        return str(value)

    query_rows: List[Dict[str, Any]] = []
    query_id_set = set()
    for idx, row in enumerate(query_ds):
        if max_queries is not None and idx >= int(max_queries):
            break
        query_id = _row_value(row, str(query_spec.get("id_field") or "id"))
        query_text = _row_value(row, str(query_spec.get("text_field") or "text"))
        query_rows.append({"id": query_id, "text": query_text})
        query_id_set.add(query_id)

    qrel_rows: List[Dict[str, Any]] = []
    relevant_doc_ids = set()
    for idx, row in enumerate(qrel_ds):
        if max_qrels is not None and idx >= int(max_qrels):
            break
        query_id = _row_value(row, str(qrel_spec.get("query_id_field") or "query_id"))
        doc_id = _row_value(row, str(qrel_spec.get("doc_id_field") or "doc_id"))
        if query_id_set and query_id not in query_id_set:
            continue
        relevance_key = str(qrel_spec.get("relevance_field") or "relevance")
        relevance = float(row.get(relevance_key) or 0.0)
        qrel_rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": relevance})
        if relevance > 0.0:
            relevant_doc_ids.add(doc_id)

    corpus_rows: List[Dict[str, Any]] = []
    exported_doc_ids = set()
    for idx, row in enumerate(corpus_ds):
        doc_id = _row_value(row, str(corpus_spec.get("id_field") or "id"))
        if max_docs is not None and len(corpus_rows) >= int(max_docs):
            break
        if relevant_doc_ids and doc_id not in relevant_doc_ids and not corpus_spec.get("full_corpus"):
            continue
        title = _row_value(row, str(corpus_spec.get("title_field") or ""), required=False) if corpus_spec.get("title_field") else ""
        text = _row_value(row, str(corpus_spec.get("text_field") or "text"))
        metadata_fields = list(corpus_spec.get("metadata_fields") or [])
        metadata = {field: row.get(field) for field in metadata_fields if field in row}
        corpus_rows.append({"id": doc_id, "title": title, "text": text, "metadata": metadata})
        exported_doc_ids.add(doc_id)

    qrel_rows = [row for row in qrel_rows if row["doc_id"] in exported_doc_ids]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "corpus.jsonl", corpus_rows)
    _write_jsonl(output_dir / "queries.jsonl", query_rows)
    _write_jsonl(output_dir / "qrels.jsonl", qrel_rows)

    metadata = {
        "source": "huggingface",
        "dataset_name": dataset_name,
        "mapping_path": str(Path(mapping_path).resolve()),
        "max_queries": max_queries,
        "max_docs": max_docs,
        "max_qrels": max_qrels,
        **summarize_standard_benchmark(output_dir),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata
