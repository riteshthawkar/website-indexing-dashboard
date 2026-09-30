"""Operator-facing automatic-evaluation CLI contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from lawa_rag.auto_eval import CorpusSnapshot, CoveragePlan, Persona
from lawa_rag.auto_eval.cli import build_parser, main
from pipeline.tests.lawa_rag_auto_eval_support import sample_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_snapshot_command_writes_valid_content_addressed_snapshot(
    tmp_path: Path,
    capsys,
) -> None:
    bundle = tmp_path / "retrieval_bundle.json"
    output = tmp_path / "corpus.json"
    bundle.write_text(json.dumps(sample_bundle()), encoding="utf-8")

    code = main(
        [
            "snapshot",
            "--bundle",
            str(bundle),
            "--domain",
            "university",
            "--output",
            str(output),
        ]
    )
    snapshot = CorpusSnapshot.from_dict(json.loads(output.read_text(encoding="utf-8")))

    assert code == 0
    assert snapshot.domain == "university"
    assert '"source_count": 2' in capsys.readouterr().out


def test_supplied_coverage_and_persona_templates_are_executable() -> None:
    coverage_path = PROJECT_ROOT / "examples" / "auto_eval" / "coverage.yaml"
    personas_path = PROJECT_ROOT / "examples" / "auto_eval" / "personas.yaml"
    coverage = CoveragePlan(**yaml.safe_load(coverage_path.read_text())).validated()
    persona_rows = yaml.safe_load(personas_path.read_text())["personas"]
    personas = [Persona.from_dict(item) for item in persona_rows]

    assert coverage.count == 100
    assert coverage.no_answer_fraction == 0.1
    assert [item.role for item in personas] == [
        "prospective student",
        "research collaborator",
    ]


def test_generate_parser_supports_independent_provider_roles() -> None:
    args = build_parser().parse_args(
        [
            "generate",
            "--snapshot",
            "corpus.json",
            "--output-dir",
            "suite",
            "--provider",
            "gemini",
            "--reference-provider",
            "openai",
            "--critic-provider",
            "openai",
            "--critic-model",
            "gpt-5-mini",
        ]
    )

    assert args.provider == "gemini"
    assert args.reference_provider == "openai"
    assert args.critic_provider == "openai"
