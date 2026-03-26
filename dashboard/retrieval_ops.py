"""
Run-scoped retrieval helpers for the dashboard.

These helpers expose the production retriever directly from the dashboard so the
UI can run live smoke queries against an indexed run without shelling out to the
CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.answer_generation import _compose_structured_answer
from pipeline.retrieval import AdaptiveHybridRetriever


_retriever_cache: Dict[Tuple[str, str], AdaptiveHybridRetriever] = {}
_retriever_cache_lock = Lock()


def get_retriever(config_name: str, work_dir: str | Path) -> AdaptiveHybridRetriever:
    cache_key = (str(config_name), str(Path(work_dir).resolve()))
    with _retriever_cache_lock:
        retriever = _retriever_cache.get(cache_key)
        if retriever is not None:
            return retriever
        retriever = AdaptiveHybridRetriever.from_config(
            config_name=config_name,
            work_dir=str(Path(work_dir).resolve()),
        )
        _retriever_cache[cache_key] = retriever
        return retriever


def run_retrieval_query(
    *,
    config_name: str,
    work_dir: str | Path,
    query: str,
) -> Dict[str, Any]:
    retriever = get_retriever(config_name=config_name, work_dir=work_dir)
    result = retriever.retrieve(query)
    try:
        answer_preview = _compose_structured_answer(
            query=query,
            retriever=retriever,
            retrieval_result=result,
        )
    except Exception:
        answer_preview = None

    return {
        "query": query,
        "config_name": config_name,
        "work_dir": str(Path(work_dir).resolve()),
        "answer_preview": answer_preview,
        "result": result,
    }
