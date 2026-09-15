"""Audited human-approval operations for generated evaluation suites."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ._util import utc_now
from .corpus import CorpusSnapshot
from .suite import EvalSuite


def approve_cases(
    suite: EvalSuite,
    snapshot: CorpusSnapshot,
    case_ids: Sequence[str],
    *,
    reviewer: str = "operator",
    notes: str = "",
) -> EvalSuite:
    requested = {str(value).strip() for value in case_ids if str(value).strip()}
    known = {case.case_id for case in suite.cases}
    unknown = sorted(requested - known)
    if not requested:
        raise ValueError("At least one case id is required for approval")
    if unknown:
        raise ValueError(f"Unknown evaluation case ids: {unknown}")
    reviewed_at = utc_now()
    cases = []
    for case in suite.cases:
        if case.case_id not in requested:
            cases.append(case)
            continue
        metadata = dict(case.metadata)
        metadata["human_review"] = {
            "reviewer": str(reviewer or "operator"),
            "reviewed_at": reviewed_at,
            "notes": str(notes or ""),
        }
        cases.append(replace(case, review_status="human_approved", metadata=metadata))
    metadata = dict(suite.metadata)
    metadata["previous_suite_id"] = suite.suite_id
    reviewed = EvalSuite.build(
        snapshot_id=suite.snapshot_id,
        plan=suite.plan,
        personas=suite.personas,
        cases=cases,
        metadata=metadata,
    )
    reviewed.validate(snapshot).require_valid()
    return reviewed


__all__ = ["approve_cases"]
