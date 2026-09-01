"""Immutable production evaluation policy for release promotion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from pipeline.core.io import sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_EVAL_POLICY_ID = "mbzuai-production-eval-v3"
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
    "fa400a69bcb9f1a61b6cdb9c8033fed3b16426d2d58b8cec1b427fe097499ac8"
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

PREPROD_EVAL_POLICY_ID = "mbzuai-preprod-current-eval-v1"
PREPROD_RETRIEVAL_DATASET = (
    PROJECT_ROOT
    / "eval"
    / "mbzuai_gold"
    / "mbzuai_preprod_multilingual_current_v1.jsonl"
)
PREPROD_RETRIEVAL_GATES = (
    PROJECT_ROOT / "eval" / "gates" / "retrieval_gate.preprod_current_v1.json"
)
PREPROD_ANSWER_DATASET = PREPROD_RETRIEVAL_DATASET
PREPROD_ANSWER_GATES = (
    PROJECT_ROOT / "eval" / "gates" / "answer_readiness_gate.preprod_current_v1.json"
)
PREPROD_RETRIEVAL_DATASET_SHA256 = (
    "d48b847fa1a12d49d5f10009988dfc1c47f9e7fb813cde365993f6d3c4762f5c"
)
PREPROD_RETRIEVAL_GATES_SHA256 = (
    "08299b4953ac1075624ecf07cbe40c410502df0e5993f58272eae1565407b304"
)
PREPROD_ANSWER_DATASET_SHA256 = PREPROD_RETRIEVAL_DATASET_SHA256
PREPROD_ANSWER_GATES_SHA256 = (
    "79dd20b2b909117dcf52736a0551747d77dd22a47bd7b6675b3b9956d497ea82"
)
PREPROD_MIN_RETRIEVAL_QUERIES = 94
PREPROD_MIN_ANSWER_QUERIES = 94

_PRODUCTION_EVAL_POLICIES: dict[str, dict[str, Any]] = {
    PRODUCTION_EVAL_POLICY_ID: {
        "retrieval_dataset": PRODUCTION_RETRIEVAL_DATASET,
        "retrieval_gates": PRODUCTION_RETRIEVAL_GATES,
        "answer_dataset": PRODUCTION_ANSWER_DATASET,
        "answer_gates": PRODUCTION_ANSWER_GATES,
        "retrieval_dataset_sha256": PRODUCTION_RETRIEVAL_DATASET_SHA256,
        "retrieval_gates_sha256": PRODUCTION_RETRIEVAL_GATES_SHA256,
        "answer_dataset_sha256": PRODUCTION_ANSWER_DATASET_SHA256,
        "answer_gates_sha256": PRODUCTION_ANSWER_GATES_SHA256,
        "minimum_retrieval_queries": PRODUCTION_MIN_RETRIEVAL_QUERIES,
        "minimum_answer_queries": PRODUCTION_MIN_ANSWER_QUERIES,
    },
    PREPROD_EVAL_POLICY_ID: {
        "retrieval_dataset": PREPROD_RETRIEVAL_DATASET,
        "retrieval_gates": PREPROD_RETRIEVAL_GATES,
        "answer_dataset": PREPROD_ANSWER_DATASET,
        "answer_gates": PREPROD_ANSWER_GATES,
        "retrieval_dataset_sha256": PREPROD_RETRIEVAL_DATASET_SHA256,
        "retrieval_gates_sha256": PREPROD_RETRIEVAL_GATES_SHA256,
        "answer_dataset_sha256": PREPROD_ANSWER_DATASET_SHA256,
        "answer_gates_sha256": PREPROD_ANSWER_GATES_SHA256,
        "minimum_retrieval_queries": PREPROD_MIN_RETRIEVAL_QUERIES,
        "minimum_answer_queries": PREPROD_MIN_ANSWER_QUERIES,
    },
}


def production_eval_policy_id_for_config(config_name: str | Path) -> str:
    """Select the immutable evaluation contract for a production corpus."""

    stem = Path(str(config_name or "")).stem.casefold()
    if stem.startswith("mbzuai_preprod_") or stem == "mbzuai_preprod":
        return PREPROD_EVAL_POLICY_ID
    return PRODUCTION_EVAL_POLICY_ID


def _production_eval_policy(policy_id: str) -> dict[str, Any]:
    normalized = str(policy_id or "").strip()
    try:
        return _PRODUCTION_EVAL_POLICIES[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown production evaluation policy: {normalized or '<empty>'}") from exc


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
    policy_id: str,
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
            f"{label} SHA256 does not match {policy_id}: "
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
    policy_id: str = PRODUCTION_EVAL_POLICY_ID,
) -> list[str]:
    try:
        policy = _production_eval_policy(policy_id)
    except ValueError as exc:
        return [str(exc)]
    errors: list[str] = []
    errors.extend(
        _validate_file(
            label="retrieval evaluation dataset",
            actual_path=retrieval_dataset,
            expected_path=policy["retrieval_dataset"],
            expected_sha256=policy["retrieval_dataset_sha256"],
            policy_id=policy_id,
        )
    )
    errors.extend(
        _validate_file(
            label="retrieval evaluation gates",
            actual_path=retrieval_gates,
            expected_path=policy["retrieval_gates"],
            expected_sha256=policy["retrieval_gates_sha256"],
            policy_id=policy_id,
            require_gate_rules=True,
        )
    )
    errors.extend(
        _validate_file(
            label="answer evaluation dataset",
            actual_path=answer_dataset,
            expected_path=policy["answer_dataset"],
            expected_sha256=policy["answer_dataset_sha256"],
            policy_id=policy_id,
        )
    )
    errors.extend(
        _validate_file(
            label="answer evaluation gates",
            actual_path=answer_gates,
            expected_path=policy["answer_gates"],
            expected_sha256=policy["answer_gates_sha256"],
            policy_id=policy_id,
            require_gate_rules=True,
        )
    )
    return errors


def production_eval_manifest_metadata(
    *,
    answer: bool = False,
    policy_id: str = PRODUCTION_EVAL_POLICY_ID,
) -> dict[str, Any]:
    policy = _production_eval_policy(policy_id)
    return {
        "policy_id": policy_id,
        "dataset_sha256": (
            policy["answer_dataset_sha256"]
            if answer
            else policy["retrieval_dataset_sha256"]
        ),
        "gates_sha256": (
            policy["answer_gates_sha256"]
            if answer
            else policy["retrieval_gates_sha256"]
        ),
        "minimum_query_count": (
            policy["minimum_answer_queries"]
            if answer
            else policy["minimum_retrieval_queries"]
        ),
    }


def production_answer_judge_manifest_metadata(
    *,
    policy_id: str = PRODUCTION_EVAL_POLICY_ID,
) -> dict[str, Any]:
    policy = _production_eval_policy(policy_id)
    return {
        "enabled": True,
        "providers": [PRODUCTION_ANSWER_JUDGE_PROVIDER],
        "models": [PRODUCTION_ANSWER_JUDGE_MODEL],
        "required_provider": PRODUCTION_ANSWER_JUDGE_PROVIDER,
        "required_model": PRODUCTION_ANSWER_JUDGE_MODEL,
        "openai_fallback_allowed": False,
        "identity_mismatch_count": 0,
        "error_count": 0,
        "judged_count": policy["minimum_answer_queries"],
    }


def validate_production_eval_manifest(
    retrieval: Mapping[str, Any],
    answer: Mapping[str, Any],
    *,
    allow_answer_waiver: bool = False,
) -> list[str]:
    errors: list[str] = []
    retrieval_policy_id = str(retrieval.get("policy_id") or "").strip()
    answer_policy_id = str(answer.get("policy_id") or "").strip()
    policy_id = retrieval_policy_id
    if (
        not policy_id
        or policy_id != answer_policy_id
        or policy_id not in _PRODUCTION_EVAL_POLICIES
    ):
        trusted = ", ".join(sorted(_PRODUCTION_EVAL_POLICIES))
        errors.append(
            "retrieval and answer evaluation policy_id values must match one trusted "
            f"production policy ({trusted})"
        )
        policy_id = PRODUCTION_EVAL_POLICY_ID
    policy = _production_eval_policy(policy_id)
    expected_retrieval = production_eval_manifest_metadata(
        answer=False,
        policy_id=policy_id,
    )
    expected_answer = production_eval_manifest_metadata(
        answer=True,
        policy_id=policy_id,
    )
    for label, payload, expected in (
        ("retrieval evaluation", retrieval, expected_retrieval),
        ("answer evaluation", answer, expected_answer),
    ):
        for key in ("policy_id", "dataset_sha256", "gates_sha256", "minimum_query_count"):
            if payload.get(key) != expected[key]:
                errors.append(f"{label} {key} does not match {policy_id}")
    minimum_retrieval_queries = int(policy["minimum_retrieval_queries"])
    minimum_answer_queries = int(policy["minimum_answer_queries"])
    if _positive_int(retrieval.get("query_count")) < minimum_retrieval_queries:
        errors.append(
            f"retrieval evaluation query_count must be at least {minimum_retrieval_queries}"
        )
    if not allow_answer_waiver and _positive_int(answer.get("query_count")) < minimum_answer_queries:
        errors.append(f"answer evaluation query_count must be at least {minimum_answer_queries}")
    if not allow_answer_waiver:
        judge = answer.get("llm_judge") if isinstance(answer.get("llm_judge"), Mapping) else {}
        expected_judge_values = production_answer_judge_manifest_metadata(
            policy_id=policy_id,
        )
        for key, expected in expected_judge_values.items():
            if key == "judged_count":
                continue
            if judge.get(key) != expected:
                errors.append(
                    f"answer evaluation llm_judge.{key} does not match "
                    f"{policy_id}"
                )
        if _positive_int(judge.get("judged_count")) < minimum_answer_queries:
            errors.append(
                f"answer evaluation judged_count must be at least {minimum_answer_queries}"
            )
    return errors
