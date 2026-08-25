from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.evaluation.retrieval_eval import check_metric_gates


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_release_dataset_manifest_matches_multilingual_dataset():
    dataset_path = REPO_ROOT / "eval" / "mbzuai_gold" / "mbzuai_llm_generated_v1.jsonl"
    manifest_path = REPO_ROOT / "eval" / "mbzuai_gold" / "mbzuai_llm_generated_v1.manifest.json"

    examples = load_eval_examples(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    language_counts = Counter(example.language for example in examples)
    answerability_counts = Counter("no_answer" if example.no_answer else "answerable" for example in examples)

    assert len(examples) == 65
    assert manifest["row_count"] == len(examples)
    assert manifest["generated_row_count"] == 50
    assert manifest["release_control_row_count"] == 15
    assert manifest["language_counts"] == dict(language_counts)
    assert manifest["answerability_counts"] == dict(answerability_counts)


def test_eval_example_normalizes_supported_language_aliases():
    assert EvalExample(id="ar", query="سؤال", query_type="fact", language="ar").normalized().language == "Arabic"
    assert EvalExample(id="en", query="Question", query_type="fact", language="en").normalized().language == "English"


def test_release_gates_require_arabic_coverage_and_latency_metrics():
    answer_gates = json.loads(
        (REPO_ROOT / "eval" / "gates" / "answer_readiness_gate.llm_generated_v1.json").read_text(
            encoding="utf-8"
        )
    )
    retrieval_gates = json.loads(
        (REPO_ROOT / "eval" / "gates" / "retrieval_gate.v5_span_strict.json").read_text(
            encoding="utf-8"
        )
    )

    assert answer_gates["by_language"]["Arabic"]["query_count"]["min"] == 8
    assert retrieval_gates["by_language"]["Arabic"]["query_count"]["min"] == 8
    assert answer_gates["overall"]["p95_first_content_latency_ms"]["max"] == 15000
    assert answer_gates["overall"]["p95_latency_ms"]["max"] == 60000


def test_metric_gates_enforce_language_slices():
    failures = check_metric_gates(
        {
            "overall": {},
            "by_query_type": {},
            "by_source_type": {},
            "by_language": {"Arabic": {"query_count": 7.0}},
            "by_benchmark_tag": {},
        },
        {"by_language": {"Arabic": {"query_count": {"min": 8}}}},
    )

    assert failures == [
        {
            "section": "by_language",
            "slice": "Arabic",
            "metric": "query_count",
            "expected_min": 8.0,
            "actual": 7.0,
        }
    ]
