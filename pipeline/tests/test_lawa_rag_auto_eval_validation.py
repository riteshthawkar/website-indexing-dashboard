"""Deterministic semantic-quality gates for generated evaluation cases."""

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
