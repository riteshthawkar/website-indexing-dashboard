from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

import lawa_rag
from lawa_rag import IndexingPipeline, load_pipeline_config
from pipeline.tests.lawa_rag_test_support import behavior_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _requirement_lines(path: Path) -> set[str]:
    values: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            values.update(_requirement_lines((path.parent / line[3:].strip()).resolve()))
            continue
        values.add(line.casefold())
    return values


def test_public_version_matches_distribution_metadata() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert lawa_rag.__version__ == metadata["project"]["version"]


def test_package_extras_match_canonical_runtime_requirements() -> None:
    metadata = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    base = {value.casefold() for value in metadata["project"]["dependencies"]}
    extras = metadata["project"]["optional-dependencies"]

    assert base | {value.casefold() for value in extras["indexing"]} == _requirement_lines(
        PROJECT_ROOT / "requirements.txt"
    )
    assert base | {value.casefold() for value in extras["retrieval"]} == _requirement_lines(
        PROJECT_ROOT / "requirements-retriever.txt"
    )


def test_config_mapping_and_returned_config_are_defensive_copies(tmp_path: Path) -> None:
    source = behavior_config(tmp_path)
    pipeline = IndexingPipeline.from_config(source, run_id="copy-test")
    source["package_behavior"]["mode"] = "mutated input"
    returned = pipeline.config
    returned["package_behavior"]["mode"] = "mutated output"

    assert pipeline.config["package_behavior"]["mode"] == "success"
    assert load_pipeline_config({"nested": {"one": 1}}, overrides={"nested": {"two": 2}}) == {
        "nested": {"one": 1, "two": 2}
    }
