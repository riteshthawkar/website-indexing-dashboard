"""Retry, no-answer, and persona-generation policy tests."""

from lawa_rag.auto_eval import AutoEvalBuilder, CoveragePlan, generate_personas
from pipeline.tests.lawa_rag_auto_eval_support import simple_snapshot, student_persona
from pipeline.tests.test_lawa_rag_auto_eval_generation import ScriptedLLM, _plan


def test_same_script_non_verbatim_quote_is_retried() -> None:
    llm = ScriptedLLM(invalid_reference_calls=1)
    suite = AutoEvalBuilder(
        question_llm=llm,
        reference_llm=llm,
        max_concurrency=1,
        generation_attempts=2,
    ).generate(simple_snapshot(), _plan(), personas=(student_persona(),))

    assert llm.reference_calls == 2
    assert suite.cases[0].citations[0].quote == (
        "The university library is open from 8:00 AM to 8:00 PM on weekdays."
    )
    assert suite.cases[0].metadata["citation_quote_repairs"] == 0
    assert suite.cases[0].metadata["citation_quote_repair_evidence_ids"] == []
    assert suite.cases[0].review_status == "auto_validated"


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


def test_personas_can_be_generated_from_the_corpus() -> None:
    personas = generate_personas(simple_snapshot(), ScriptedLLM(), count=2)

    assert [item.role for item in personas] == ["student", "visitor"]
    assert all(item.metadata["generated"] for item in personas)
