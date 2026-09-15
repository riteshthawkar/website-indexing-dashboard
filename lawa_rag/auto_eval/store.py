"""Crash-safe per-scenario checkpoints for resumable suite generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from ._util import read_json, read_jsonl, utc_now, write_json, write_jsonl
from .cases import GeneratedEvalCase
from .corpus import CorpusSnapshot
from .personas import Persona
from .planning import CoveragePlan, ScenarioBlueprint
from .suite import EvalSuite
from .validation import ValidationReport


STORE_SCHEMA_VERSION = "lawa.auto_eval.generation.v1"


class GenerationStore:
    def __init__(self, directory: str | Path):
        self.root = Path(directory).expanduser().resolve()
        self.state_path = self.root / "generation_state.json"
        self.case_dir = self.root / "case_checkpoints"
        self.failure_dir = self.root / "failed_scenarios"

    @property
    def exists(self) -> bool:
        return self.state_path.exists()

    def restore_inputs(
        self,
        snapshot: CorpusSnapshot,
        plan: CoveragePlan,
    ) -> tuple[tuple[Persona, ...], tuple[ScenarioBlueprint, ...]]:
        state = read_json(self.state_path)
        _verify_state(state, snapshot, plan)
        personas = tuple(
            Persona.from_dict(item)
            for item in read_jsonl(self.root / "personas.jsonl")
        )
        scenarios = tuple(
            ScenarioBlueprint.from_dict(item) for item in read_jsonl(self.root / "scenarios.jsonl")
        )
        return personas, scenarios

    def initialize(
        self,
        *,
        snapshot: CorpusSnapshot,
        plan: CoveragePlan,
        personas: Sequence[Persona],
        scenarios: Sequence[ScenarioBlueprint],
        generation: Mapping[str, Any],
    ) -> None:
        if self.exists:
            raise FileExistsError(f"Generation store already exists: {self.root}")
        self.case_dir.mkdir(parents=True, exist_ok=True)
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.root / "corpus_snapshot.json", snapshot.to_dict())
        write_json(self.root / "plan.json", plan.to_dict())
        write_jsonl(self.root / "personas.jsonl", (item.to_dict() for item in personas))
        write_jsonl(self.root / "scenarios.jsonl", (item.to_dict() for item in scenarios))
        write_json(
            self.state_path,
            {
                "schema_version": STORE_SCHEMA_VERSION,
                "status": "running",
                "created_at": utc_now(),
                "snapshot_id": snapshot.snapshot_id,
                "plan_fingerprint": plan.fingerprint,
                "scenario_count": len(scenarios),
                "generation": dict(generation),
            },
        )

    def load_cases(self) -> tuple[GeneratedEvalCase, ...]:
        if not self.case_dir.exists():
            return ()
        rows = [read_json(path) for path in sorted(self.case_dir.glob("*.json"))]
        return tuple(GeneratedEvalCase.from_dict(item) for item in rows)

    def save_case(self, case: GeneratedEvalCase) -> None:
        self.case_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.case_dir / f"{_safe_name(case.scenario_id)}.json", case.to_dict())

    def save_failure(self, scenario_id: str, payload: Mapping[str, Any]) -> None:
        self.failure_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.failure_dir / f"{_safe_name(scenario_id)}.json", dict(payload))

    def finalize(
        self,
        suite: EvalSuite,
        snapshot: CorpusSnapshot,
        validation: ValidationReport,
        *,
        failure_count: int = 0,
    ) -> None:
        suite.write(self.root, snapshot=snapshot, validation=validation)
        state = read_json(self.state_path)
        state.update(
            {
                "status": _final_status(validation, failure_count),
                "completed_at": utc_now(),
                "suite_id": suite.suite_id,
                "case_count": len(suite.cases),
                "failure_count": failure_count,
            }
        )
        write_json(self.state_path, state)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "scenario"


def _final_status(validation: ValidationReport, failure_count: int) -> str:
    if failure_count:
        return "incomplete"
    return "complete" if validation.ok else "invalid"


def _verify_state(state: Mapping[str, Any], snapshot: CorpusSnapshot, plan: CoveragePlan) -> None:
    if state.get("schema_version") != STORE_SCHEMA_VERSION:
        raise ValueError("Unsupported auto-evaluation generation checkpoint")
    if state.get("snapshot_id") != snapshot.snapshot_id:
        raise ValueError("Cannot resume with a different corpus snapshot")
    if state.get("plan_fingerprint") != plan.fingerprint:
        raise ValueError("Cannot resume with a different coverage plan")


__all__ = ["GenerationStore", "STORE_SCHEMA_VERSION"]
