from __future__ import annotations

from pipeline.core.release import _validation_config_for_release
from pipeline.core.release_policy import (
    PRODUCTION_ANSWER_DATASET,
    PRODUCTION_ANSWER_GATES,
    PRODUCTION_EVAL_POLICY_ID,
    PRODUCTION_MIN_ANSWER_QUERIES,
    PRODUCTION_MIN_RETRIEVAL_QUERIES,
    PRODUCTION_RETRIEVAL_DATASET,
    PRODUCTION_RETRIEVAL_GATES,
    production_answer_judge_manifest_metadata,
    production_eval_manifest_metadata,
    validate_production_eval_inputs,
    validate_production_eval_manifest,
)


def test_release_stage_validation_uses_release_purpose_without_mutating_config(tmp_path) -> None:
    config = {
        "pipeline": {"production_profile": True},
        "stages": [
            {"id": "verify", "plugin": "verify_selected_profile"},
            {"id": "upload", "plugin": "gemini_pgvector"},
        ],
    }

    release_config = _validation_config_for_release(config, tmp_path)

    assert release_config["pipeline"]["validation_purpose"] == "release"
    assert [stage["id"] for stage in release_config["stages"]] == ["verify"]
    assert "validation_purpose" not in config["pipeline"]
    assert [stage["id"] for stage in config["stages"]] == ["verify", "upload"]


def test_committed_production_evaluation_policy_matches_pinned_hashes() -> None:
    assert validate_production_eval_inputs(
        retrieval_dataset=PRODUCTION_RETRIEVAL_DATASET,
        retrieval_gates=PRODUCTION_RETRIEVAL_GATES,
        answer_dataset=PRODUCTION_ANSWER_DATASET,
        answer_gates=PRODUCTION_ANSWER_GATES,
    ) == []


def test_production_evaluation_policy_rejects_arbitrary_inputs(tmp_path) -> None:
    dataset = tmp_path / "one-row.jsonl"
    gates = tmp_path / "empty-gates.json"
    dataset.write_text('{"query":"x"}\n', encoding="utf-8")
    gates.write_text("{}", encoding="utf-8")

    errors = validate_production_eval_inputs(
        retrieval_dataset=dataset,
        retrieval_gates=gates,
        answer_dataset=dataset,
        answer_gates=gates,
    )

    assert len(errors) == 4
    assert all("canonical production file" in error for error in errors)


def test_production_evaluation_manifest_requires_policy_hashes_and_minimum_counts() -> None:
    retrieval = {
        **production_eval_manifest_metadata(answer=False),
        "query_count": PRODUCTION_MIN_RETRIEVAL_QUERIES,
    }
    answer = {
        **production_eval_manifest_metadata(answer=True),
        "query_count": PRODUCTION_MIN_ANSWER_QUERIES,
        "llm_judge": production_answer_judge_manifest_metadata(),
    }
    assert validate_production_eval_manifest(retrieval, answer) == []

    retrieval["policy_id"] = "untrusted"
    answer["query_count"] = 1
    errors = validate_production_eval_manifest(retrieval, answer)
    assert any(PRODUCTION_EVAL_POLICY_ID in error for error in errors)
    assert any(f"at least {PRODUCTION_MIN_ANSWER_QUERIES}" in error for error in errors)


def test_production_evaluation_manifest_rejects_judge_provider_fallback() -> None:
    retrieval = {
        **production_eval_manifest_metadata(answer=False),
        "query_count": PRODUCTION_MIN_RETRIEVAL_QUERIES,
    }
    answer = {
        **production_eval_manifest_metadata(answer=True),
        "query_count": PRODUCTION_MIN_ANSWER_QUERIES,
        "llm_judge": {
            **production_answer_judge_manifest_metadata(),
            "providers": ["openai"],
            "models": ["gpt-4.1"],
            "identity_mismatch_count": PRODUCTION_MIN_ANSWER_QUERIES,
            "openai_fallback_allowed": True,
        },
    }

    errors = validate_production_eval_manifest(retrieval, answer)

    assert any("llm_judge" in error for error in errors)
