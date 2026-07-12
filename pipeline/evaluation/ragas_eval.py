from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from pipeline.core.google_genai import import_genai


DEFAULT_RAGAS_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
    "answer_similarity",
]


def _load_prediction_rows(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        return rows
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected JSON array or JSONL in {path}")


def _make_google_clients():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for RAGAS evaluation")
    genai = import_genai()
    client = genai.Client(api_key=api_key)
    return client


def _add_legacy_embedding_methods(embeddings: Any) -> Any:
    """Bridge modern RAGAS embeddings to legacy metric expectations."""
    if not hasattr(embeddings, "embed_query"):
        embeddings.embed_query = embeddings.embed_text  # type: ignore[attr-defined]
    if not hasattr(embeddings, "embed_documents"):
        embeddings.embed_documents = embeddings.embed_texts  # type: ignore[attr-defined]
    if not hasattr(embeddings, "aembed_query"):
        embeddings.aembed_query = embeddings.aembed_text  # type: ignore[attr-defined]
    if not hasattr(embeddings, "aembed_documents"):
        embeddings.aembed_documents = embeddings.aembed_texts  # type: ignore[attr-defined]
    return embeddings


def _import_ragas_components():
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import embedding_factory
        from ragas.llms import llm_factory
        from ragas.metrics import (
            answer_correctness,
            answer_relevancy,
            answer_similarity,
            context_precision,
            context_recall,
            faithfulness,
        )
        from ragas.run_config import RunConfig
    except ImportError as exc:
        raise RuntimeError(
            "RAGAS evaluation requires optional dependencies. Install `ragas` in the active venv first."
        ) from exc

    metric_registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "llm_context_precision_with_reference": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
        "factual_correctness": answer_correctness,
        "answer_similarity": answer_similarity,
    }
    return Dataset, evaluate, llm_factory, embedding_factory, metric_registry, RunConfig


def _build_metrics(
    *,
    metric_names: Iterable[str],
    llm: Any,
    embeddings: Any,
    metric_registry: Dict[str, Any],
) -> List[Any]:
    metrics = []
    for metric_name in metric_names:
        metric_template = metric_registry.get(metric_name)
        if metric_template is None:
            raise ValueError(f"Unsupported RAGAS metric: {metric_name}")
        metric = copy.deepcopy(metric_template)
        if hasattr(metric, "llm"):
            metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
        if metric_name in {"answer_correctness", "factual_correctness"} and hasattr(metric, "answer_similarity"):
            metric.answer_similarity = copy.deepcopy(metric_registry["answer_similarity"])
            if hasattr(metric.answer_similarity, "embeddings"):
                metric.answer_similarity.embeddings = embeddings
        metrics.append(metric)
    return metrics


def run_ragas_evaluation(
    *,
    predictions_path: str | Path,
    metric_names: Iterable[str] | None = None,
    llm_model: str = "gemini-2.5-flash",
    embedding_model: str = "gemini-embedding-2",
) -> Dict[str, Any]:
    Dataset, evaluate, llm_factory, embedding_factory, metric_registry, RunConfig = _import_ragas_components()
    rows = _load_prediction_rows(predictions_path)
    if not rows:
        raise ValueError("No prediction rows found for RAGAS evaluation")

    client = _make_google_clients()
    llm = llm_factory(
        llm_model,
        provider="google",
        client=client,
        temperature=0.0,
    )
    embeddings = embedding_factory(
        provider="google",
        model=embedding_model,
        client=client,
    )
    embeddings = _add_legacy_embedding_methods(embeddings)

    metric_names = list(metric_names or DEFAULT_RAGAS_METRICS)
    metrics = _build_metrics(
        metric_names=metric_names,
        llm=llm,
        embeddings=embeddings,
        metric_registry=metric_registry,
    )

    dataset = Dataset.from_list(rows)
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        show_progress=True,
        raise_exceptions=False,
        run_config=RunConfig(timeout=300, max_workers=4, max_retries=3),
    )
    scores = dict(getattr(result, "_repr_dict", {})) or result.to_pandas().mean(numeric_only=True).to_dict()

    return {
        "predictions_path": str(Path(predictions_path).resolve()),
        "row_count": len(rows),
        "llm_model": llm_model,
        "embedding_model": embedding_model,
        "metrics": {
            metric_name: (float(scores.get(metric_name)) if scores.get(metric_name) is not None else None)
            for metric_name in metric_names
        },
    }
