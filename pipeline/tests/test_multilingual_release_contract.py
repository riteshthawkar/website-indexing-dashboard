from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.evaluation.retrieval_eval import check_metric_gates


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_release_dataset_manifest_matches_multilingual_dataset():
    dataset_path = REPO_ROOT / "eval" / "mbzuai_gold" / "mbzuai_multilingual_v2.jsonl"
    manifest_path = REPO_ROOT / "eval" / "mbzuai_gold" / "mbzuai_multilingual_v2.manifest.json"

    examples = load_eval_examples(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    language_counts = Counter(example.language for example in examples)
    answerability_counts = Counter("no_answer" if example.no_answer else "answerable" for example in examples)
    split_counts = Counter(str(example.metadata.get("split") or "") for example in examples)
    dataset_manifest = manifest["dataset"]
    coverage = dataset_manifest["coverage"]

    assert len(examples) == 160
    assert manifest["suite"] == "mbzuai_multilingual_v2"
    assert manifest["candidate_independent_gold"] is True
    assert manifest["experimental_policy"]["production_promotion_authorized"] is True
    assert dataset_manifest["path"] == "eval/mbzuai_gold/mbzuai_multilingual_v2.jsonl"
    assert dataset_manifest["sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert coverage["query_count"] == len(examples)
    assert coverage["language_counts"] == dict(language_counts)
    assert coverage["answerability_counts"] == dict(answerability_counts)
    assert coverage["split_counts"] == dict(split_counts)
    assert coverage["source_group_split_leak_count"] == 0


def test_preprod_release_manifest_and_gate_counts_match_the_committed_dataset():
    dataset_path = (
        REPO_ROOT
        / "eval"
        / "mbzuai_gold"
        / "mbzuai_preprod_multilingual_current_v1.jsonl"
    )
    manifest = json.loads(
        (
            REPO_ROOT
            / "eval"
            / "mbzuai_gold"
            / "mbzuai_preprod_multilingual_current_v1.manifest.json"
        ).read_text(encoding="utf-8")
    )
    examples = load_eval_examples(dataset_path)
    language_counts = Counter(example.language for example in examples)
    query_type_counts = Counter(example.query_type for example in examples)
    answerable_count = sum(not example.no_answer for example in examples)
    no_answer_count = sum(example.no_answer for example in examples)
    dataset_manifest = manifest["dataset"]

    assert dataset_manifest["sha256"] == hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    assert dataset_manifest["query_count"] == len(examples)
    assert dataset_manifest["answerable_query_count"] == answerable_count
    assert dataset_manifest["no_answer_query_count"] == no_answer_count
    assert dataset_manifest["languages"] == dict(language_counts)
    assert dataset_manifest["query_types"] == dict(query_type_counts)

    for gate_name in (
        "retrieval_gate.preprod_current_v1.json",
        "answer_readiness_gate.preprod_current_v1.json",
    ):
        gates = json.loads(
            (REPO_ROOT / "eval" / "gates" / gate_name).read_text(encoding="utf-8")
        )
        assert gates["overall"]["query_count"]["min"] == len(examples)
        assert gates["overall"]["answerable_query_count"]["min"] == answerable_count
        assert gates["overall"]["no_answer_query_count"]["min"] == no_answer_count
        for language, count in language_counts.items():
            assert gates["by_language"][language]["query_count"]["min"] == count
        for query_type, count in query_type_counts.items():
            assert gates["by_query_type"][query_type]["query_count"]["min"] == count


def test_eval_example_normalizes_supported_language_aliases():
    assert EvalExample(id="ar", query="سؤال", query_type="fact", language="ar").normalized().language == "Arabic"
    assert EvalExample(id="en", query="Question", query_type="fact", language="en").normalized().language == "English"


def test_release_gates_require_arabic_coverage_and_latency_metrics():
    answer_gates = json.loads(
        (
            REPO_ROOT
            / "eval"
            / "gates"
            / "answer_readiness_gate.multilingual_v2_release.json"
        ).read_text(encoding="utf-8")
    )
    retrieval_gates = json.loads(
        (
            REPO_ROOT
            / "eval"
            / "gates"
            / "retrieval_gate.multilingual_v2_release.json"
        ).read_text(encoding="utf-8")
    )

    assert answer_gates["overall"]["query_count"]["min"] == 160
    assert retrieval_gates["overall"]["query_count"]["min"] == 160
    assert answer_gates["by_language"]["Arabic"]["query_count"]["min"] == 80
    assert retrieval_gates["by_language"]["Arabic"]["query_count"]["min"] == 80
    assert answer_gates["by_query_type"]["multimodal"]["query_count"]["min"] == 24
    assert retrieval_gates["by_benchmark_tag"]["navigation"]["query_count"]["min"] == 16
    assert answer_gates["overall"]["p95_first_content_latency_ms"]["max"] == 15000
    assert answer_gates["overall"]["p95_latency_ms"]["max"] == 60000


def test_release_gate_slices_are_present_in_multilingual_v2_dataset():
    dataset_path = REPO_ROOT / "eval" / "mbzuai_gold" / "mbzuai_multilingual_v2.jsonl"
    answer_gates = json.loads(
        (
            REPO_ROOT
            / "eval"
            / "gates"
            / "answer_readiness_gate.multilingual_v2_release.json"
        ).read_text(encoding="utf-8")
    )
    retrieval_gates = json.loads(
        (
            REPO_ROOT
            / "eval"
            / "gates"
            / "retrieval_gate.multilingual_v2_release.json"
        ).read_text(encoding="utf-8")
    )
    examples = load_eval_examples(dataset_path)
    query_types = {example.query_type for example in examples}
    source_types = {example.source_type for example in examples}
    languages = {example.language for example in examples}
    tags = {
        tag
        for example in examples
        for tag in (example.metadata.get("benchmark_tags") or [])
    }

    for gates in (answer_gates, retrieval_gates):
        assert not (set(gates.get("by_query_type") or {}) - query_types)
        assert not (set(gates.get("by_source_type") or {}) - source_types)
        assert not (set(gates.get("by_language") or {}) - languages)
        assert not (set(gates.get("by_benchmark_tag") or {}) - tags)


def test_multilingual_v2_selection_gate_covers_production_dimensions():
    gates = json.loads(
        (
            REPO_ROOT
            / "eval"
            / "gates"
            / "retrieval_gate.multilingual_v2_selection.json"
        ).read_text(encoding="utf-8")
    )

    overall = gates["overall"]
    assert overall["query_count"]["min"] == 96
    assert overall["retrieval_error_count"]["max"] == 0
    assert overall["no_answer_violation_rate"]["max"] == 0.0
    assert overall["document_hit_at_10"]["min"] >= 0.90
    assert overall["exact_media_hit_at_5"]["min"] >= 0.70
    assert overall["navigation_action_hit_at_5"]["min"] == 1.0
    assert overall["backend_latency_p95_ms"]["max"] <= 12000
    assert gates["by_language"]["Arabic"]["query_count"]["min"] == 48


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
