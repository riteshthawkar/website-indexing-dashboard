"""Command-line entry point for building and validating auto-evaluation suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from ._util import read_json, write_json
from .adapters import RetrievalBundleAdapter
from .builder import AutoEvalBuilder
from .compat import export_eval_examples
from .corpus import CorpusSnapshot
from .llm import create_structured_llm
from .personas import Persona
from .planning import CoveragePlan
from .review import approve_cases
from .store import GenerationStore
from .suite import EvalSuite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lawa-rag-auto-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="Normalize a retrieval bundle")
    source = snapshot.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle")
    source.add_argument("--work-dir")
    snapshot.add_argument("--domain", default="generic")
    snapshot.add_argument("--output", required=True)

    generate = commands.add_parser("generate", help="Generate a grounded evaluation suite")
    generate.add_argument("--snapshot", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--provider", choices=("openai", "gemini"), default="openai")
    generate.add_argument("--model")
    generate.add_argument("--reference-provider", choices=("openai", "gemini"))
    generate.add_argument("--reference-model")
    generate.add_argument("--critic-provider", choices=("openai", "gemini"))
    generate.add_argument("--critic-model")
    generate.add_argument("--plan")
    generate.add_argument("--personas")
    generate.add_argument("--count", type=int)
    generate.add_argument("--persona-count", type=int, default=6)
    generate.add_argument("--max-concurrency", type=int, default=4)
    generate.add_argument("--generation-attempts", type=int, default=2)
    generate.add_argument("--user-context", default="")
    generate.add_argument("--no-resume", action="store_true")

    validate = commands.add_parser("validate", help="Validate a completed suite")
    validate.add_argument("--suite-dir", required=True)
    approve = commands.add_parser("approve", help="Record human approval for reviewed cases")
    approve.add_argument("--suite-dir", required=True)
    approve.add_argument("--case-id", action="append", default=[])
    approve.add_argument("--all-needs-review", action="store_true")
    approve.add_argument("--reviewer", default="operator")
    approve.add_argument("--notes", default="")
    export = commands.add_parser("export-legacy", help="Export legacy EvalExample JSONL")
    export.add_argument("--suite-dir", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--allow-unreviewed", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "snapshot":
        return _snapshot(args)
    if args.command == "generate":
        return _generate(args)
    if args.command == "validate":
        return _validate(args)
    if args.command == "approve":
        return _approve(args)
    if args.command == "export-legacy":
        return _export(args)
    raise AssertionError(f"Unknown command: {args.command}")


def _snapshot(args) -> int:
    snapshot = (
        RetrievalBundleAdapter.from_file(args.bundle, domain=args.domain)
        if args.bundle
        else RetrievalBundleAdapter.from_run(args.work_dir, domain=args.domain)
    )
    write_json(args.output, snapshot.to_dict())
    _print(snapshot.summary())
    return 0


def _generate(args) -> int:
    snapshot = CorpusSnapshot.from_dict(read_json(args.snapshot))
    plan = _load_plan(args.plan, args.count)
    personas = _load_personas(args.personas)
    primary = create_structured_llm(args.provider, model=args.model)
    reference = (
        create_structured_llm(
            args.reference_provider or args.provider,
            model=args.reference_model,
        )
        if args.reference_provider or args.reference_model
        else primary
    )
    critic = (
        create_structured_llm(
            args.critic_provider or args.provider,
            model=args.critic_model,
        )
        if args.critic_provider or args.critic_model
        else None
    )
    builder = AutoEvalBuilder(
        question_llm=primary,
        reference_llm=reference,
        critic_llm=critic,
        max_concurrency=args.max_concurrency,
        generation_attempts=args.generation_attempts,
    )
    suite = builder.generate(
        snapshot,
        plan,
        personas=personas,
        persona_count=args.persona_count,
        user_context=args.user_context,
        output_dir=args.output_dir,
        resume=not args.no_resume,
    )
    _print(suite.summary())
    return 0


def _validate(args) -> int:
    root = Path(args.suite_dir).expanduser().resolve()
    suite = EvalSuite.from_directory(root)
    snapshot = CorpusSnapshot.from_dict(read_json(root / "corpus_snapshot.json"))
    report = suite.validate(snapshot)
    _print(report.to_dict())
    return 0 if report.ok else 1


def _export(args) -> int:
    root = Path(args.suite_dir).expanduser().resolve()
    suite = EvalSuite.from_directory(root)
    snapshot = CorpusSnapshot.from_dict(read_json(root / "corpus_snapshot.json"))
    output = export_eval_examples(
        suite,
        snapshot,
        args.output,
        allow_unreviewed=args.allow_unreviewed,
    )
    _print({"output": str(output), "case_count": len(suite.cases)})
    return 0


def _approve(args) -> int:
    root = Path(args.suite_dir).expanduser().resolve()
    suite = EvalSuite.from_directory(root)
    snapshot = CorpusSnapshot.from_dict(read_json(root / "corpus_snapshot.json"))
    case_ids = list(args.case_id)
    if args.all_needs_review:
        case_ids.extend(
            item.case_id for item in suite.cases if item.review_status == "needs_review"
        )
    reviewed = approve_cases(
        suite,
        snapshot,
        case_ids,
        reviewer=args.reviewer,
        notes=args.notes,
    )
    report = reviewed.validate(snapshot)
    store = GenerationStore(root)
    for case in reviewed.cases:
        store.save_case(case)
    store.finalize(reviewed, snapshot, report)
    _print(reviewed.summary())
    return 0


def _load_plan(path: str | None, count: int | None) -> CoveragePlan:
    payload: Mapping[str, Any] = {}
    if path:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("Coverage plan must be a YAML/JSON object")
        payload = loaded
    values = dict(payload)
    if count is not None:
        values["count"] = count
    return CoveragePlan(**values).validated()


def _load_personas(path: str | None) -> tuple[Persona, ...] | None:
    if not path:
        return None
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rows = loaded.get("personas") if isinstance(loaded, Mapping) else loaded
    if not isinstance(rows, list):
        raise ValueError("Persona file must contain a list or a personas list")
    return tuple(Persona.from_dict(item) for item in rows)


def _print(payload: Mapping[str, Any]) -> None:
    print(json.dumps(dict(payload), indent=2, ensure_ascii=False))


__all__ = ["build_parser", "main"]
