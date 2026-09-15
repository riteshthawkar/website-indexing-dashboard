"""Human-review workflow tests for generated no-answer cases."""

from __future__ import annotations

import pytest

from lawa_rag.auto_eval import (
    CoveragePlan,
    DifficultyProfile,
    EvalSuite,
    GeneratedEvalCase,
    approve_cases,
    export_eval_examples,
)
from lawa_rag.auto_eval._util import stable_id
from pipeline.tests.lawa_rag_auto_eval_support import simple_snapshot, student_persona


def _unanswerable_suite() -> tuple[EvalSuite, object]:
    snapshot = simple_snapshot()
    persona = student_persona()
    query = "Does the university operate a private airport?"
    scenario_id = "scenario:no-airport"
    case = GeneratedEvalCase(
        case_id=stable_id("eval-case", snapshot.snapshot_id, scenario_id, query),
        scenario_id=scenario_id,
        persona_id=persona.persona_id,
        query=query,
        language="English",
        task="fact",
        answerable=False,
        target_difficulty="easy",
        measured_difficulty=DifficultyProfile.for_level("easy", "fact"),
        reference_answer="The frozen corpus does not establish a private airport.",
        claims=(),
        citations=(),
        evidence_ids=(),
        distractor_ids=(),
        source_ids=(),
        review_status="needs_review",
    )
    plan = CoveragePlan(
        count=1,
        languages={"English": 1.0},
        difficulties={"easy": 1.0},
        tasks={"fact": 1.0},
        no_answer_fraction=1.0,
    )
    return (
        EvalSuite.build(
            snapshot_id=snapshot.snapshot_id,
            plan=plan,
            personas=(persona,),
            cases=(case,),
        ),
        snapshot,
    )


def test_approval_records_provenance_and_rebuilds_suite_identity() -> None:
    suite, snapshot = _unanswerable_suite()
    assert suite.validate(snapshot).warnings

    reviewed = approve_cases(
        suite,
        snapshot,
        [suite.cases[0].case_id],
        reviewer="quality-owner",
        notes="Checked all frozen sources",
    )

    assert reviewed.suite_id != suite.suite_id
    assert reviewed.cases[0].review_status == "human_approved"
    assert reviewed.cases[0].metadata["human_review"]["reviewer"] == "quality-owner"
    assert reviewed.validate(snapshot).warnings == ()


def test_approval_rejects_unknown_or_empty_case_selection() -> None:
    suite, snapshot = _unanswerable_suite()

    with pytest.raises(ValueError, match="At least one"):
        approve_cases(suite, snapshot, [])
    with pytest.raises(ValueError, match="Unknown"):
        approve_cases(suite, snapshot, ["not-a-case"])


def test_legacy_export_fails_closed_until_negative_cases_are_reviewed(tmp_path) -> None:
    suite, snapshot = _unanswerable_suite()
    with pytest.raises(ValueError, match="requiring review"):
        export_eval_examples(suite, snapshot, tmp_path / "blocked.jsonl")

    reviewed = approve_cases(suite, snapshot, [suite.cases[0].case_id])
    output = export_eval_examples(reviewed, snapshot, tmp_path / "approved.jsonl")
    assert output.exists()
