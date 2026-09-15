"""End-to-end tests for generation, validation, resume, and compatibility export."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from lawa_rag.auto_eval import (
    AutoEvalBuilder,
    AutoEvalGenerationError,
    CoveragePlan,
    EvalSuite,
    LLMIdentity,
    export_eval_examples,
    generate_personas,
)
from pipeline.evaluation.dataset import load_eval_examples
from pipeline.tests.lawa_rag_auto_eval_support import simple_snapshot, student_persona


class ScriptedLLM:
    def __init__(self, *, invalid_reference_calls: int = 0, fail_on_call: bool = False):
        self.invalid_reference_calls = invalid_reference_calls
        self.fail_on_call = fail_on_call
        self.calls: list[str] = []
        self.reference_calls = 0

    @property
    def identity(self) -> LLMIdentity:
        return LLMIdentity("test", "scripted-v1")

    def generate_json(self, *, system_prompt, user_prompt, schema_name):
        if self.fail_on_call:
            raise AssertionError("LLM should not be called for a completed checkpoint")
        self.calls.append(schema_name)
        if schema_name == "auto_eval_personas_v1":
            return {
                "personas": [
                    {
                        "role": "student",
                        "goals": ["find campus services"],
                        "preferred_languages": ["English"],
                    },
                    {
                        "role": "visitor",
                        "goals": ["plan a campus visit"],
                        "preferred_languages": ["English"],
                    },
                ]
            }
        if schema_name == "auto_eval_question_v1":
            query = "When is the university library open on weekdays?"
            if "query already exists" in user_prompt:
                query = "What are the university library's weekday opening hours?"
            return {
                "query": query,
                "intent": "find library hours",
                "expected_response_structure": "short_answer",
            }
        if schema_name == "auto_eval_reference_v1":
            self.reference_calls += 1
            if '"answerable": false' in user_prompt:
                return {
                    "reference_answer": "The frozen corpus does not establish this fact.",
                    "claims": [],
                    "citations": [],
                }
            quote = "not present in the evidence"
            if self.reference_calls > self.invalid_reference_calls:
                quote = "The university library is open from 8:00 AM to 8:00 PM on weekdays."
            return {
                "reference_answer": "The library is open from 8:00 AM to 8:00 PM on weekdays.",
                "claims": [
                    {
                        "text": "Weekday opening hours are 8:00 AM to 8:00 PM.",
                        "evidence_ids": ["evidence:library-hours"],
                    }
                ],
                "citations": [
                    {"evidence_id": "evidence:library-hours", "quote": quote}
                ],
                "answer_must_include": ["8:00 AM to 8:00 PM"],
                "answer_must_not_include": ["unverified weekend hours"],
                "answer_should_cover": ["weekday schedule"],
                "citation_requirements": ["opening hours"],
            }
        if schema_name == "auto_eval_critic_v1":
            return {
                "passed": True,
                "answerability_correct": True,
                "grounding_score": 1.0,
                "naturalness_score": 1.0,
                "difficulty_match_score": 1.0,
                "issues": [],
            }
        raise AssertionError(schema_name)


def _plan(language: str = "English") -> CoveragePlan:
    return CoveragePlan(
        count=1,
        languages={language: 1.0},
        difficulties={"easy": 1.0},
        tasks={"fact": 1.0},
        no_answer_fraction=0.0,
    )


def test_builder_generates_valid_resumable_suite_and_legacy_export(tmp_path: Path) -> None:
    llm = ScriptedLLM()
    snapshot = simple_snapshot()
    output = tmp_path / "suite"
    suite = AutoEvalBuilder(
        question_llm=llm,
        reference_llm=llm,
        critic_llm=llm,
        max_concurrency=1,
    ).generate(snapshot, _plan("French"), personas=(student_persona(),), output_dir=output)

    assert suite.validate(snapshot).ok
    assert suite.cases[0].review_status == "auto_verified"
    assert suite.cases[0].citations[0].evidence_id == "evidence:library-hours"
    assert EvalSuite.from_directory(output).suite_id == suite.suite_id
    assert json.loads((output / "generation_state.json").read_text())["status"] == "complete"

    legacy_path = export_eval_examples(suite, snapshot, tmp_path / "legacy.jsonl")
    legacy = load_eval_examples(legacy_path)
    assert legacy[0].language == "French"
    assert legacy[0].gold_chunk_ids == []
    assert legacy[0].metadata["persona_id"] == student_persona().persona_id

    tampered = replace(
        suite,
        cases=(replace(suite.cases[0], reference_answer="Modified answer"),),
    )
    assert not tampered.validate(snapshot).ok

    no_call = ScriptedLLM(fail_on_call=True)
    resumed = AutoEvalBuilder(question_llm=no_call, max_concurrency=1).generate(
        snapshot,
        _plan("French"),
        personas=(student_persona(),),
        output_dir=output,
    )
    assert resumed.suite_id == suite.suite_id


def test_invalid_grounding_is_retried_before_case_is_accepted() -> None:
    llm = ScriptedLLM(invalid_reference_calls=1)
    suite = AutoEvalBuilder(
        question_llm=llm,
        reference_llm=llm,
        max_concurrency=1,
        generation_attempts=2,
    ).generate(simple_snapshot(), _plan(), personas=(student_persona(),))

    assert llm.reference_calls == 2
    assert suite.cases[0].review_status == "auto_validated"


def test_duplicate_queries_are_regenerated() -> None:
    llm = ScriptedLLM()
    plan = CoveragePlan(
        count=2,
        languages={"English": 1.0},
        difficulties={"easy": 1.0},
        tasks={"fact": 1.0},
        no_answer_fraction=0.0,
    )
    suite = AutoEvalBuilder(
        question_llm=llm,
        reference_llm=llm,
        max_concurrency=2,
    ).generate(simple_snapshot(), plan, personas=(student_persona(),))

    assert len({item.query for item in suite.cases}) == 2


def test_no_answer_generation_is_flagged_for_human_review() -> None:
    plan = CoveragePlan(
        count=1,
        languages={"English": 1.0},
        difficulties={"easy": 1.0},
        tasks={"fact": 1.0},
        no_answer_fraction=1.0,
    )
    suite = AutoEvalBuilder(
        question_llm=ScriptedLLM(),
        max_concurrency=1,
    ).generate(simple_snapshot(), plan, personas=(student_persona(),))

    assert suite.cases[0].evidence_ids == ()
    assert suite.cases[0].review_status == "needs_review"
    assert suite.validate(simple_snapshot()).warnings[0].code == "negative_needs_review"


def test_failed_scenario_is_checkpointed_and_can_resume(tmp_path: Path) -> None:
    output = tmp_path / "recoverable"
    invalid = ScriptedLLM(invalid_reference_calls=5)
    builder = AutoEvalBuilder(
        question_llm=invalid,
        reference_llm=invalid,
        max_concurrency=1,
        generation_attempts=1,
    )
    with pytest.raises(AutoEvalGenerationError):
        builder.generate(
            simple_snapshot(),
            _plan(),
            personas=(student_persona(),),
            output_dir=output,
        )
    assert json.loads((output / "generation_state.json").read_text())["status"] == "incomplete"
    assert len(list((output / "failed_scenarios").glob("*.json"))) == 1

    valid = ScriptedLLM()
    suite = AutoEvalBuilder(
        question_llm=valid,
        reference_llm=valid,
        max_concurrency=1,
    ).generate(simple_snapshot(), _plan(), output_dir=output)
    assert len(suite.cases) == 1
    assert json.loads((output / "generation_state.json").read_text())["status"] == "complete"


def test_personas_can_be_generated_from_the_corpus() -> None:
    personas = generate_personas(simple_snapshot(), ScriptedLLM(), count=2)

    assert [item.role for item in personas] == ["student", "visitor"]
    assert all(item.metadata["generated"] for item in personas)
