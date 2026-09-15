"""Resumable orchestration for grounded automatic evaluation generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from ._util import clean_text, stable_id, string_tuple
from ._generation_support import (
    citations_from,
    claims_from,
    critic_accepted,
    failure_message,
    generation_metadata,
    save_failure,
)
from ._model_calls import generate_question, generate_reference, run_critic
from .cases import GeneratedEvalCase
from .corpus import CorpusSnapshot
from .llm import StructuredLLM
from .personas import Persona, generate_personas
from .planning import CoveragePlan, ScenarioPlanner
from .sampling import EvidenceSampler
from .store import GenerationStore
from .suite import EvalSuite
from .validation import validate_case


class AutoEvalGenerationError(RuntimeError):
    """Raised when a scenario cannot pass generation validation."""


class AutoEvalBuilder:
    def __init__(
        self,
        *,
        question_llm: StructuredLLM,
        reference_llm: StructuredLLM | None = None,
        critic_llm: StructuredLLM | None = None,
        max_concurrency: int = 4,
        generation_attempts: int = 2,
    ):
        if max_concurrency < 1 or generation_attempts < 1:
            raise ValueError("Concurrency and generation attempts must be positive")
        self.question_llm = question_llm
        self.reference_llm = reference_llm or question_llm
        self.critic_llm = critic_llm
        self.max_concurrency = max_concurrency
        self.generation_attempts = generation_attempts

    def generate(
        self,
        snapshot: CorpusSnapshot,
        coverage: CoveragePlan,
        *,
        personas: Sequence[Persona] | None = None,
        persona_llm: StructuredLLM | None = None,
        persona_count: int = 6,
        user_context: str = "",
        output_dir: str | Path | None = None,
        resume: bool = True,
    ) -> EvalSuite:
        snapshot.require_valid()
        plan = coverage.validated()
        store = GenerationStore(output_dir) if output_dir is not None else None
        persona_limit = min(persona_count, plan.count)
        selected_personas, scenarios = self._inputs(
            snapshot, plan, personas, persona_llm, persona_limit, user_context, store, resume
        )
        existing = {item.scenario_id: item for item in store.load_cases()} if store else {}
        pending = [item for item in scenarios if item.scenario_id not in existing]
        generated, failures = self._generate_pending(
            snapshot, selected_personas, pending, store, tuple(existing.values())
        )
        cases = tuple(existing.values()) + tuple(generated)
        suite = EvalSuite.build(
            snapshot_id=snapshot.snapshot_id,
            plan=plan,
            personas=selected_personas,
            cases=cases,
            metadata={"generation": self._generation_metadata(), "failure_count": len(failures)},
        )
        validation = suite.validate(snapshot)
        if store:
            store.finalize(
                suite, snapshot, validation, failure_count=len(failures)
            )
        if failures:
            raise AutoEvalGenerationError(failure_message(failures, output_dir))
        validation.require_valid()
        return suite

    def _inputs(self, snapshot, plan, personas, persona_llm, count, context, store, resume):
        if store and store.exists:
            if not resume:
                raise FileExistsError(f"Evaluation output already exists: {store.root}")
            return store.restore_inputs(snapshot, plan)
        selected = tuple(personas or ())
        if not selected:
            selected = generate_personas(
                snapshot,
                persona_llm or self.question_llm,
                count=count,
                user_context=context,
            )
        scenarios = ScenarioPlanner().plan(snapshot, selected, plan)
        if store:
            store.initialize(
                snapshot=snapshot,
                plan=plan,
                personas=selected,
                scenarios=scenarios,
                generation=self._generation_metadata(),
            )
        return selected, scenarios

    def _generate_pending(self, snapshot, personas, scenarios, store, existing):
        if not scenarios:
            return [], []
        persona_map = {item.persona_id: item for item in personas}
        sampler = EvidenceSampler(snapshot)
        generated: list[GeneratedEvalCase] = []
        failures: list[tuple[str, str]] = []
        seen_queries = {clean_text(item.query).casefold() for item in existing}
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            futures = {}
            for scenario in scenarios:
                bundle = sampler.sample(scenario)
                future = executor.submit(
                    self._generate_case,
                    snapshot,
                    persona_map[scenario.persona_id],
                    scenario,
                    bundle,
                )
                futures[future] = (scenario, bundle)
            for future in as_completed(futures):
                scenario, bundle = futures[future]
                try:
                    case = future.result()
                    case, normalized = self._ensure_unique(
                        case, seen_queries, snapshot, persona_map, scenario, bundle
                    )
                except Exception as exc:
                    failures.append((scenario.scenario_id, str(exc)))
                    save_failure(store, scenario, exc)
                    continue
                seen_queries.add(normalized)
                generated.append(case)
                if store:
                    store.save_case(case)
        return generated, failures

    def _ensure_unique(
        self, case, seen_queries, snapshot, persona_map, scenario, bundle
    ):
        normalized = clean_text(case.query).casefold()
        if normalized in seen_queries:
            case = self._generate_case(
                snapshot,
                persona_map[scenario.persona_id],
                scenario,
                bundle,
                initial_feedback=(
                    "Generate a different query; that query already exists."
                ),
            )
            normalized = clean_text(case.query).casefold()
        if normalized in seen_queries:
            raise AutoEvalGenerationError("Question duplicates an existing case")
        return case, normalized

    def _generate_case(
        self, snapshot, persona, scenario, bundle, *, initial_feedback=""
    ):
        feedback = initial_feedback
        last_error = "generation failed"
        for attempt in range(1, self.generation_attempts + 1):
            question = generate_question(
                self.question_llm, snapshot, persona, scenario, bundle, feedback
            )
            reference = generate_reference(
                self.reference_llm, snapshot, scenario, bundle, question["query"]
            )
            case = self._case_from_payloads(
                snapshot, persona, scenario, bundle, question, reference, attempt
            )
            report = validate_case(case, snapshot)
            if report.errors:
                last_error = "; ".join(item.message for item in report.errors)
                feedback = last_error
                continue
            checked = run_critic(self.critic_llm, snapshot, case, bundle)
            if checked is not None and not critic_accepted(checked):
                issues = string_tuple(checked.get("issues"))
                last_error = "; ".join(issues) or "critic rejected the case"
                feedback = last_error
                continue
            status = "needs_review" if not scenario.answerable else (
                "auto_verified" if checked is not None else "auto_validated"
            )
            return replace(case, review_status=status, critic=dict(checked or {}))
        raise AutoEvalGenerationError(last_error)

    def _case_from_payloads(
        self, snapshot, persona, scenario, bundle, question, reference, attempt
    ):
        query = question["query"]
        return GeneratedEvalCase(
            case_id=stable_id("eval-case", snapshot.snapshot_id, scenario.scenario_id, query),
            scenario_id=scenario.scenario_id,
            persona_id=persona.persona_id,
            query=query,
            language=scenario.language,
            task=scenario.task,
            answerable=scenario.answerable,
            target_difficulty=scenario.difficulty.target_level,
            measured_difficulty=bundle.difficulty,
            reference_answer=clean_text(reference.get("reference_answer")),
            claims=claims_from(reference.get("claims")),
            citations=citations_from(reference.get("citations")),
            evidence_ids=bundle.evidence_ids,
            distractor_ids=bundle.distractor_ids,
            source_ids=bundle.source_ids,
            expected_response_structure=question["expected_response_structure"],
            answer_must_include=string_tuple(reference.get("answer_must_include")),
            answer_must_not_include=string_tuple(reference.get("answer_must_not_include")),
            answer_should_cover=string_tuple(reference.get("answer_should_cover")),
            citation_requirements=string_tuple(reference.get("citation_requirements")),
            generation={**self._generation_metadata(), "attempt": attempt},
            metadata={"intent": question["intent"]},
        )

    def _generation_metadata(self) -> dict[str, Any]:
        return generation_metadata(
            self.question_llm, self.reference_llm, self.critic_llm
        )
__all__ = ["AutoEvalBuilder", "AutoEvalGenerationError"]
