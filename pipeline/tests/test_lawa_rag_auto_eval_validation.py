"""Deterministic semantic-quality gates for generated evaluation cases."""

from types import SimpleNamespace

from lawa_rag.auto_eval import (
    CorpusSnapshot,
    DifficultyProfile,
    EvidenceCitation,
    EvidenceUnit,
    GeneratedEvalCase,
    GoldClaim,
    SourceRecord,
    validate_case,
)
from lawa_rag.auto_eval._validation_support import (
    answerable_semantic_issues,
    has_cross_scope_claim,
    infers_absence_from_comparison,
    mentions_visual_evidence,
    query_uncovered_evidence,
)
from lawa_rag.auto_eval._util import stable_id


def test_medium_case_must_use_every_selected_gold_evidence() -> None:
    source = SourceRecord(
        source_id="source:program",
        revision_id="revision:program",
        source_type="webpage",
        title="Program",
    )
    first = EvidenceUnit(
        evidence_id="evidence:first",
        source_id=source.source_id,
        kind="chunk",
        text="The program requires two foundation courses before advanced study.",
    )
    second = EvidenceUnit(
        evidence_id="evidence:second",
        source_id=source.source_id,
        kind="chunk",
        text="Students complete an industry capstone during the final semester.",
    )
    snapshot = CorpusSnapshot.build(
        domain="university",
        sources=(source,),
        evidence=(first, second),
    )
    query = "Explain the program requirements."
    case = GeneratedEvalCase(
        case_id=stable_id("eval-case", snapshot.snapshot_id, "scenario:medium", query),
        scenario_id="scenario:medium",
        persona_id="persona:student",
        query=query,
        language="English",
        task="explain",
        answerable=True,
        target_difficulty="medium",
        measured_difficulty=DifficultyProfile.for_level("medium", "explain"),
        reference_answer="The program requires two foundation courses.",
        claims=(GoldClaim("Two foundation courses are required.", (first.evidence_id,)),),
        citations=(EvidenceCitation(first.evidence_id, first.text),),
        evidence_ids=(first.evidence_id, second.evidence_id),
        distractor_ids=(),
        source_ids=(source.source_id,),
    )

    codes = {item.code for item in validate_case(case, snapshot).errors}

    assert "unused_gold_evidence" in codes
    assert "incoherent_gold_evidence" in codes


def test_reference_answer_cannot_expose_internal_evidence_ids() -> None:
    source = SourceRecord(
        source_id="source:one",
        revision_id="revision:one",
        source_type="webpage",
        title="Program",
    )
    evidence = EvidenceUnit(
        evidence_id="chunk:program:001",
        source_id=source.source_id,
        kind="chunk",
        text="The program requires one official transcript for admission.",
    )
    snapshot = CorpusSnapshot.build(
        domain="university", sources=(source,), evidence=(evidence,)
    )
    query = "What document is required?"
    case = GeneratedEvalCase(
        case_id=stable_id("eval-case", snapshot.snapshot_id, "scenario:one", query),
        scenario_id="scenario:one",
        persona_id="persona:student",
        query=query,
        language="English",
        task="fact",
        answerable=True,
        target_difficulty="easy",
        measured_difficulty=DifficultyProfile.for_level("easy", "fact"),
        reference_answer="One transcript is required (chunk:program:001).",
        claims=(GoldClaim("One transcript is required.", (evidence.evidence_id,)),),
        citations=(EvidenceCitation(evidence.evidence_id, "one official transcript"),),
        evidence_ids=(evidence.evidence_id,),
        distractor_ids=(),
        source_ids=(source.source_id,),
    )

    codes = {item.code for item in validate_case(case, snapshot).errors}

    assert "reference_leaks_evidence_id" in codes


def test_comparison_absence_and_visual_query_guards() -> None:
    assert infers_absence_from_comparison(
        "The general page does not mention an opt-out policy."
    )
    assert infers_absence_from_comparison("Exam timing is not explicitly mentioned.")
    assert infers_absence_from_comparison("There is no specific mention of timing.")
    assert infers_absence_from_comparison("The overview is without explicit opt-out details.")
    assert infers_absence_from_comparison("الخيار غير مذكور في الصفحة العامة")
    assert not infers_absence_from_comparison(
        "Both excerpts explicitly require a screening exam."
    )
    assert mentions_visual_evidence("What does the image show about campus life?")
    assert mentions_visual_evidence("ما الذي يظهر في الصورة عن حياة الطلاب؟")
    assert not mentions_visual_evidence("What events improve campus life?")


def test_silence_based_absence_is_rejected_for_non_comparison_tasks() -> None:
    case = SimpleNamespace(
        task="synthesis",
        query="Summarize the requirements.",
        reference_answer="The selected source does not mention any other skills.",
    )

    codes = {code for code, _ in answerable_semantic_issues(case)}

    assert "reference_infers_absence" in codes


def test_comparison_requires_a_claim_supported_by_every_compared_source() -> None:
    first = EvidenceUnit("evidence:a", "source:a", "chunk", "Program A requires calculus.")
    second = EvidenceUnit("evidence:b", "source:b", "chunk", "Program B requires calculus.")
    evidence = {first.evidence_id: first, second.evidence_id: second}
    selected = set(evidence)
    separate = SimpleNamespace(
        claims=(
            GoldClaim("Program A requires calculus.", (first.evidence_id,)),
            GoldClaim("Program B requires calculus.", (second.evidence_id,)),
        )
    )
    integrated = SimpleNamespace(
        claims=(GoldClaim("Both programs require calculus.", tuple(selected)),)
    )

    assert not has_cross_scope_claim(separate, selected, evidence)
    assert has_cross_scope_claim(integrated, selected, evidence)


def test_medium_question_must_require_each_same_language_gold_unit() -> None:
    research = EvidenceUnit(
        "evidence:research",
        "source:division",
        "chunk",
        "## Research We investigate robotics and machine learning.",
    )
    programs = EvidenceUnit(
        "evidence:programs",
        "source:division",
        "chunk",
        "## Programs We offer undergraduate and doctoral degrees.",
    )
    case = SimpleNamespace(
        target_difficulty="medium",
        query="Explain the division's research scope and robotics work.",
        evidence_ids=(research.evidence_id, programs.evidence_id),
    )

    assert query_uncovered_evidence(
        case, {research.evidence_id: research, programs.evidence_id: programs}
    ) == (programs.evidence_id,)
