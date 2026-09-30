"""Immutable evaluation-suite persistence and summary contracts."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._util import json_mapping, read_json, read_jsonl, stable_id, utc_now, write_json, write_jsonl
from .cases import GeneratedEvalCase
from .corpus import CorpusSnapshot
from .personas import Persona
from .planning import CoveragePlan
from .validation import ValidationIssue, ValidationReport, validate_cases


EVAL_SUITE_SCHEMA_VERSION = "lawa.auto_eval.suite.v1"


@dataclass(frozen=True)
class EvalSuite:
    suite_id: str
    snapshot_id: str
    created_at: str
    plan: CoveragePlan
    personas: tuple[Persona, ...]
    cases: tuple[GeneratedEvalCase, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVAL_SUITE_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        snapshot_id: str,
        plan: CoveragePlan,
        personas: Sequence[Persona],
        cases: Sequence[GeneratedEvalCase],
        metadata: Mapping[str, Any] | None = None,
    ) -> "EvalSuite":
        ordered = tuple(sorted(cases, key=lambda item: (item.scenario_id, item.case_id)))
        suite_id = _suite_id(snapshot_id, plan, ordered)
        return cls(
            suite_id=suite_id,
            snapshot_id=snapshot_id,
            created_at=utc_now(),
            plan=plan,
            personas=tuple(personas),
            cases=ordered,
            metadata=json_mapping(metadata),
        )

    @classmethod
    def from_directory(cls, directory: str | Path) -> "EvalSuite":
        root = Path(directory).expanduser().resolve()
        manifest = read_json(root / "manifest.json")
        plan = CoveragePlan(**read_json(root / "plan.json")).validated()
        personas = tuple(Persona.from_dict(item) for item in read_jsonl(root / "personas.jsonl"))
        cases = tuple(
            GeneratedEvalCase.from_dict(item)
            for item in read_jsonl(root / "cases.jsonl")
        )
        return cls(
            suite_id=str(manifest.get("suite_id") or ""),
            snapshot_id=str(manifest.get("snapshot_id") or ""),
            created_at=str(manifest.get("created_at") or ""),
            plan=plan,
            personas=personas,
            cases=cases,
            metadata=json_mapping(manifest.get("metadata")),
            schema_version=str(manifest.get("schema_version") or ""),
        )

    def validate(self, snapshot: CorpusSnapshot) -> ValidationReport:
        issues = list(validate_cases(self.cases, snapshot, self.personas).issues)
        if snapshot.snapshot_id != self.snapshot_id:
            issues.append(
                ValidationIssue(
                    "error",
                    "snapshot_mismatch",
                    f"Suite expects {self.snapshot_id}; received {snapshot.snapshot_id}",
                )
            )
        if self.schema_version != EVAL_SUITE_SCHEMA_VERSION:
            issues.append(ValidationIssue("error", "suite_schema", self.schema_version))
        if len(self.cases) != self.plan.count:
            message = f"Coverage plan requires {self.plan.count} cases; found {len(self.cases)}"
            issues.append(ValidationIssue("error", "case_count", message))
        if self.suite_id != _suite_id(self.snapshot_id, self.plan, self.cases):
            issues.append(ValidationIssue("error", "suite_identity", "Suite content was modified"))
        return ValidationReport(tuple(issues))

    def summary(self) -> dict[str, Any]:
        return {
            "suite_id": self.suite_id,
            "snapshot_id": self.snapshot_id,
            "case_count": len(self.cases),
            "persona_count": len(self.personas),
            "answerable_count": sum(item.answerable for item in self.cases),
            "no_answer_count": sum(not item.answerable for item in self.cases),
            "needs_review_count": sum(item.review_status == "needs_review" for item in self.cases),
            "unique_source_count": len(
                {source_id for item in self.cases for source_id in item.source_ids}
            ),
            "unique_evidence_count": len(
                {evidence_id for item in self.cases for evidence_id in item.evidence_ids}
            ),
            "tasks": dict(sorted(Counter(item.task for item in self.cases).items())),
            "languages": dict(sorted(Counter(item.language for item in self.cases).items())),
            "difficulties": dict(
                sorted(Counter(item.target_difficulty for item in self.cases).items())
            ),
        }

    def write(
        self,
        directory: str | Path,
        *,
        snapshot: CorpusSnapshot | None = None,
        validation: ValidationReport | None = None,
    ) -> Path:
        root = Path(directory).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        write_json(root / "manifest.json", self._manifest())
        write_json(root / "plan.json", self.plan.to_dict())
        write_jsonl(root / "personas.jsonl", (item.to_dict() for item in self.personas))
        write_jsonl(root / "cases.jsonl", (item.to_dict() for item in self.cases))
        if snapshot is not None:
            write_json(root / "corpus_snapshot.json", snapshot.to_dict())
        if validation is not None:
            write_json(root / "validation.json", validation.to_dict())
        return root

    def _manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "snapshot_id": self.snapshot_id,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "summary": self.summary(),
        }


def _suite_id(
    snapshot_id: str,
    plan: CoveragePlan,
    cases: Sequence[GeneratedEvalCase],
) -> str:
    ordered_cases = [
        item.to_dict() for item in sorted(cases, key=lambda item: item.case_id)
    ]
    return stable_id("eval-suite", snapshot_id, plan.fingerprint, ordered_cases)


__all__ = ["EVAL_SUITE_SCHEMA_VERSION", "EvalSuite"]
