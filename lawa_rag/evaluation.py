"""Lazy public imports for retrieval and answer-quality evaluation."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DEFAULT_RAGAS_METRICS": ("pipeline.evaluation.ragas_eval", "DEFAULT_RAGAS_METRICS"),
    "EVALUATION_PRESETS": ("pipeline.evaluation.presets", "EVALUATION_PRESETS"),
    "EvalExample": ("pipeline.evaluation.dataset", "EvalExample"),
    "aggregate_quality": ("pipeline.evaluation.controlled_ab", "aggregate_quality"),
    "choose_abstention_threshold": ("pipeline.evaluation.controlled_ab", "choose_abstention_threshold"),
    "compare_retrieval_report_files": ("pipeline.evaluation.ablation", "compare_retrieval_report_files"),
    "compare_retrieval_reports": ("pipeline.evaluation.ablation", "compare_retrieval_reports"),
    "evaluate_answer_readiness": ("pipeline.evaluation.answer_readiness", "evaluate_answer_readiness"),
    "evaluate_retrieval_dataset": ("pipeline.evaluation.retrieval_eval", "evaluate_retrieval_dataset"),
    "evaluate_standard_rankings": ("pipeline.evaluation.benchmark_io", "evaluate_standard_rankings"),
    "export_hf_benchmark": ("pipeline.evaluation.benchmark_io", "export_hf_benchmark"),
    "export_ir_datasets_benchmark": ("pipeline.evaluation.benchmark_io", "export_ir_datasets_benchmark"),
    "generate_answer_predictions": ("pipeline.evaluation.answer_generation", "generate_answer_predictions"),
    "load_eval_examples": ("pipeline.evaluation.dataset", "load_eval_examples"),
    "mbzuai_eval_template": ("pipeline.evaluation.dataset", "mbzuai_eval_template"),
    "paired_bootstrap_delta": ("pipeline.evaluation.controlled_ab", "paired_bootstrap_delta"),
    "run_ragas_evaluation": ("pipeline.evaluation.ragas_eval", "run_ragas_evaluation"),
    "run_standard_benchmark_retrieval": ("pipeline.evaluation.benchmark_runner", "run_standard_benchmark_retrieval"),
    "score_ranking": ("pipeline.evaluation.controlled_ab", "score_ranking"),
    "summarize_eval_examples": ("pipeline.evaluation.dataset_tools", "summarize_eval_examples"),
    "summarize_standard_benchmark": ("pipeline.evaluation.benchmark_io", "summarize_standard_benchmark"),
    "validate_eval_examples": ("pipeline.evaluation.dataset_tools", "validate_eval_examples"),
    "write_eval_examples": ("pipeline.evaluation.dataset", "write_eval_examples"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
