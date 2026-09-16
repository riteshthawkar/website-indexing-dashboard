"""Deterministic validation gates for generated evaluation cases and suites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from ._coherence import coherent_bundle, relationship_index
from ._util import clean_text, stable_id
from ._validation_support import (
    answerable_semantic_issues,
    evidence_scope_labels,
    has_cross_scope_claim,
    query_uncovered_evidence,
)
from .cases import EVAL_CASE_SCHEMA_VERSION, GeneratedEvalCase
from .corpus import CorpusSnapshot
from .personas import Persona


VALIDATION_VERSION = "semantic-grounding.v2"

@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str
    case_id: str = ""


@dataclass(frozen=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(item.severity == "error" for item in self.issues)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(item for item in self.issues if item.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(item for item in self.issues if item.severity == "warning")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [asdict(item) for item in self.issues],
        }

    def require_valid(self) -> None:
        if self.errors:
            details = "; ".join(f"{item.code}: {item.message}" for item in self.errors)
            raise ValueError(f"Generated evaluation data failed validation: {details}")


def validate_case(case: GeneratedEvalCase, snapshot: CorpusSnapshot) -> ValidationReport:
    issues: list[ValidationIssue] = []
    add = lambda severity, code, message: issues.append(
        ValidationIssue(severity, code, message, case.case_id)
    )
    if case.schema_version != EVAL_CASE_SCHEMA_VERSION:
        add("error", "schema_version", f"Unsupported case schema {case.schema_version!r}")
    for field, value in (
        ("case_id", case.case_id),
        ("scenario_id", case.scenario_id),
        ("persona_id", case.persona_id),
        ("query", case.query),
    ):
        if not clean_text(value):
            add("error", f"missing_{field}", f"{field} must be non-empty")
    expected_id = stable_id(
        "eval-case", snapshot.snapshot_id, case.scenario_id, case.query
    )
    if case.case_id != expected_id:
        add("error", "case_identity", "Case identity does not match its query and scenario")
    evidence = snapshot.evidence_map()
    selected = set(case.evidence_ids)
    unknown = sorted((selected | set(case.distractor_ids)) - set(evidence))
    if unknown:
        add("error", "unknown_evidence", f"Unknown evidence ids: {unknown[:5]}")
    if case.answerable:
        _validate_answerable(case, selected, evidence, add)
        _validate_coherence(case, selected, evidence, snapshot, add)
    else:
        _validate_unanswerable(case, add)
    if case.target_difficulty != case.measured_difficulty.target_level:
        add("error", "difficulty_target", "Target difficulty is inconsistent with its profile")
    if case.measured_difficulty.measured_level != case.target_difficulty:
        add(
            "warning",
            "difficulty_shortfall",
            "Available evidence did not realize the requested structural difficulty",
        )
    expected_sources = {
        evidence[item].source_id for item in selected if item in evidence
    }
    if set(case.source_ids) != expected_sources:
        add("error", "source_ids", "Case source IDs do not match selected evidence")
    return ValidationReport(tuple(issues))


def validate_cases(
    cases: Sequence[GeneratedEvalCase],
    snapshot: CorpusSnapshot,
    personas: Sequence[Persona] = (),
) -> ValidationReport:
    issues = [item for case in cases for item in validate_case(case, snapshot).issues]
    issues.extend(_uniqueness_issues(cases))
    persona_ids = {persona.persona_id for persona in personas}
    for case in cases:
        if persona_ids and case.persona_id not in persona_ids:
            issues.append(
                ValidationIssue(
                    "error", "unknown_persona", "Case references an unknown persona", case.case_id
                )
            )
    return ValidationReport(tuple(issues))


def _validate_answerable(case, selected, evidence, add) -> None:
    if not selected:
        add("error", "missing_evidence", "Answerable case requires selected evidence")
    if not case.reference_answer:
        add("error", "missing_reference_answer", "Answerable case requires a reference answer")
    for code, message in answerable_semantic_issues(case):
        add("error", code, message)
    if not case.claims:
        add("error", "missing_claims", "Answerable case requires atomic gold claims")
    if not case.citations:
        add("error", "missing_citations", "Answerable case requires evidence citations")
    _validate_claims(case, selected, evidence, add)
    _validate_citations(case, selected, evidence, add)


def _validate_claims(case, selected, evidence, add) -> None:
    for claim in case.claims:
        if not claim.text or not claim.evidence_ids:
            add("error", "invalid_claim", "Each claim requires text and evidence ids")
        if not set(claim.evidence_ids).issubset(selected):
            add("error", "claim_outside_bundle", "Claim cites evidence outside the selected bundle")
    cited_ids = {citation.evidence_id for citation in case.citations}
    claim_ids = {item for claim in case.claims for item in claim.evidence_ids}
    if not claim_ids.issubset(cited_ids):
        add("error", "citation_coverage", "Gold citations do not cover every claim")
    if case.target_difficulty in {"medium", "hard"} and not selected.issubset(claim_ids):
        unused = sorted(selected - claim_ids)
        add(
            "error",
            "unused_gold_evidence",
            f"Medium/hard case does not require all selected evidence: {unused[:5]}",
        )
    if case.task == "compare" and not has_cross_scope_claim(case, selected, evidence):
        add(
            "error",
            "comparison_missing_cross_scope_claim",
            "Comparison requires an atomic claim jointly supported by every compared scope",
        )
    uncovered = query_uncovered_evidence(case, evidence)
    if uncovered:
        add(
            "error",
            "query_does_not_require_gold_evidence",
            f"Missing selected scope: {list(evidence_scope_labels(uncovered, evidence))[:5]}",
        )
def _validate_citations(case, selected, evidence, add) -> None:
    for citation in case.citations:
        unit = evidence.get(citation.evidence_id)
        if citation.evidence_id not in selected or unit is None:
            add("error", "citation_outside_bundle", "Citation is outside the selected bundle")
        elif not _quote_is_supported(citation.quote, unit.text):
            add("error", "invalid_quote", f"Citation quote is not in {citation.evidence_id}")


def _validate_coherence(case, selected, evidence, snapshot, add) -> None:
    units = [evidence[item] for item in case.evidence_ids if item in selected and item in evidence]
    if len(units) < 2:
        return
    if not coherent_bundle(
        units,
        snapshot.source_map(),
        relationship_index(snapshot.relationships),
    ):
        add(
            "error",
            "incoherent_gold_evidence",
            "Selected gold evidence does not form one connected topical bundle",
        )


def _validate_unanswerable(case, add) -> None:
    if case.evidence_ids:
        add("error", "negative_has_evidence", "Unanswerable cases cannot declare gold evidence")
    if case.claims or case.citations:
        add(
            "error",
            "negative_has_claims",
            "Unanswerable cases cannot declare gold claims/citations",
        )
    if not case.reference_answer:
        add(
            "error",
            "negative_missing_response",
            "Unanswerable case requires an expected abstention",
        )
    if case.review_status != "human_approved":
        add("warning", "negative_needs_review", "Corpus absence requires verifier or human review")


def _quote_is_supported(quote: str, text: str) -> bool:
    normalized_quote = clean_text(quote).casefold()
    normalized_text = clean_text(text).casefold()
    return len(normalized_quote) >= 8 and normalized_quote in normalized_text


def _uniqueness_issues(cases: Iterable[GeneratedEvalCase]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for case in cases:
        normalized_query = clean_text(case.query).casefold()
        if case.case_id in seen_ids:
            issues.append(ValidationIssue("error", "duplicate_case_id", case.case_id, case.case_id))
        if normalized_query in seen_queries:
            issues.append(ValidationIssue("error", "duplicate_query", case.query, case.case_id))
        seen_ids.add(case.case_id)
        seen_queries.add(normalized_query)
    return issues

__all__ = [
    "VALIDATION_VERSION",
    "ValidationIssue",
    "ValidationReport",
    "validate_case",
    "validate_cases",
]
