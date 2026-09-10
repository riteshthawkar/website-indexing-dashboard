"""Immutable production evaluation policy for release promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pipeline.core.io import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_EVAL_POLICY_ID = "mbzuai-production-eval-v4"
PRODUCTION_RETRIEVAL_DATASET = (
    PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_multilingual_v2.jsonl"
)
PRODUCTION_RETRIEVAL_GATES = (
    PROJECT_ROOT / "eval" / "gates" / "retrieval_gate.multilingual_v2_release.json"
)
PRODUCTION_ANSWER_DATASET = PRODUCTION_RETRIEVAL_DATASET
PRODUCTION_ANSWER_GATES = (
    PROJECT_ROOT / "eval" / "gates" / "answer_readiness_gate.multilingual_v2_release.json"
)
PRODUCTION_RETRIEVAL_DATASET_SHA256 = (
    "2f34a329411d5de5a7749ac64c0906a1b9bccf237f45cd69ad89b0f3ddcd0db2"
)
PRODUCTION_RETRIEVAL_GATES_SHA256 = (
    "ba221e2d2507582d1566270f5116e423fefce21639cfb23c5773819bfad91128"
)
PRODUCTION_ANSWER_DATASET_SHA256 = PRODUCTION_RETRIEVAL_DATASET_SHA256
PRODUCTION_ANSWER_GATES_SHA256 = (
    "94fd1df2f00eef43f1fa957bb82147fb08be896f1a474f91b8837b74c92913eb"
)
PRODUCTION_MIN_RETRIEVAL_QUERIES = 160
PRODUCTION_MIN_ANSWER_QUERIES = 160
PRODUCTION_ANSWER_JUDGE_PROVIDER = "gemini"
PRODUCTION_ANSWER_JUDGE_MODEL = "gemini-2.5-flash"


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _validate_file(
    *,
    label: str,
    actual_path: str | Path,
    expected_path: Path,
    expected_sha256: str,
    require_gate_rules: bool = False,
) -> list[str]:
    path = Path(actual_path).expanduser().resolve()
    errors: list[str] = []
    if path != expected_path.resolve():
        errors.append(
            f"{label} must use the canonical production file {expected_path.resolve()}; got {path}"
        )
        return errors
    if not path.is_file():
        errors.append(f"{label} is missing: {path}")
        return errors
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        errors.append(
            f"{label} SHA256 does not match {PRODUCTION_EVAL_POLICY_ID}: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    if require_gate_rules:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = None
        recognized = {
            "overall",
            "by_query_type",
            "by_source_type",
            "by_language",
            "by_benchmark_tag",
            "per_query",
        }
        if not isinstance(payload, dict) or not any(
            key in recognized and isinstance(value, dict) and bool(value)
            for key, value in payload.items()
        ):
            errors.append(f"{label} contains no recognized non-empty metric gates")
    return errors


def validate_production_eval_inputs(
    *,
    retrieval_dataset: str | Path,
    retrieval_gates: str | Path,
    answer_dataset: str | Path,
    answer_gates: str | Path,
) -> list[str]:
    errors: list[str] = []
    errors.extend(
        _validate_file(
            label="retrieval evaluation dataset",
            actual_path=retrieval_dataset,
            expected_path=PRODUCTION_RETRIEVAL_DATASET,
            expected_sha256=PRODUCTION_RETRIEVAL_DATASET_SHA256,
        )
    )
    errors.extend(
        _validate_file(
            label="retrieval evaluation gates",
            actual_path=retrieval_gates,
            expected_path=PRODUCTION_RETRIEVAL_GATES,
            expected_sha256=PRODUCTION_RETRIEVAL_GATES_SHA256,
            require_gate_rules=True,
        )
    )
    errors.extend(
        _validate_file(
            label="answer evaluation dataset",
            actual_path=answer_dataset,
            expected_path=PRODUCTION_ANSWER_DATASET,
            expected_sha256=PRODUCTION_ANSWER_DATASET_SHA256,
        )
    )
    errors.extend(
        _validate_file(
            label="answer evaluation gates",
            actual_path=answer_gates,
            expected_path=PRODUCTION_ANSWER_GATES,
            expected_sha256=PRODUCTION_ANSWER_GATES_SHA256,
            require_gate_rules=True,
        )
    )
    return errors


def production_eval_manifest_metadata(*, answer: bool = False) -> dict[str, Any]:
    return {
        "policy_id": PRODUCTION_EVAL_POLICY_ID,
        "dataset_sha256": (
            PRODUCTION_ANSWER_DATASET_SHA256 if answer else PRODUCTION_RETRIEVAL_DATASET_SHA256
        ),
        "gates_sha256": (
            PRODUCTION_ANSWER_GATES_SHA256 if answer else PRODUCTION_RETRIEVAL_GATES_SHA256
        ),
        "minimum_query_count": (
            PRODUCTION_MIN_ANSWER_QUERIES if answer else PRODUCTION_MIN_RETRIEVAL_QUERIES
        ),
    }


def production_answer_judge_manifest_metadata() -> dict[str, Any]:
    return {
        "enabled": True,
        "providers": [PRODUCTION_ANSWER_JUDGE_PROVIDER],
        "models": [PRODUCTION_ANSWER_JUDGE_MODEL],
        "required_provider": PRODUCTION_ANSWER_JUDGE_PROVIDER,
        "required_model": PRODUCTION_ANSWER_JUDGE_MODEL,
        "openai_fallback_allowed": False,
        "identity_mismatch_count": 0,
        "error_count": 0,
        "judged_count": PRODUCTION_MIN_ANSWER_QUERIES,
    }


def validate_production_eval_manifest(
    retrieval: Mapping[str, Any],
    answer: Mapping[str, Any],
    *,
    allow_answer_waiver: bool = False,
) -> list[str]:
    errors: list[str] = []
    expected_retrieval = production_eval_manifest_metadata(answer=False)
    expected_answer = production_eval_manifest_metadata(answer=True)
    for label, payload, expected in (
        ("retrieval evaluation", retrieval, expected_retrieval),
        ("answer evaluation", answer, expected_answer),
    ):
        for key in ("policy_id", "dataset_sha256", "gates_sha256", "minimum_query_count"):
            if payload.get(key) != expected[key]:
                errors.append(f"{label} {key} does not match {PRODUCTION_EVAL_POLICY_ID}")
    if _positive_int(retrieval.get("query_count")) < PRODUCTION_MIN_RETRIEVAL_QUERIES:
        errors.append(
            f"retrieval evaluation query_count must be at least {PRODUCTION_MIN_RETRIEVAL_QUERIES}"
        )
    if not allow_answer_waiver and _positive_int(answer.get("query_count")) < PRODUCTION_MIN_ANSWER_QUERIES:
        errors.append(f"answer evaluation query_count must be at least {PRODUCTION_MIN_ANSWER_QUERIES}")
    if not allow_answer_waiver:
        judge = answer.get("llm_judge") if isinstance(answer.get("llm_judge"), Mapping) else {}
        expected_judge_values = production_answer_judge_manifest_metadata()
        for key, expected in expected_judge_values.items():
            if key == "judged_count":
                continue
            if judge.get(key) != expected:
                errors.append(
                    f"answer evaluation llm_judge.{key} does not match "
                    f"{PRODUCTION_EVAL_POLICY_ID}"
                )
        if _positive_int(judge.get("judged_count")) < PRODUCTION_MIN_ANSWER_QUERIES:
            errors.append(
                f"answer evaluation judged_count must be at least {PRODUCTION_MIN_ANSWER_QUERIES}"
            )
    return errors
