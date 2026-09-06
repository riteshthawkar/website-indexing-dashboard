#!/usr/bin/env python3
"""Fail closed when a built lawa-rag wheel omits its public contracts."""

from __future__ import annotations

import argparse
import json
import re
from email.parser import BytesParser
from pathlib import Path
from zipfile import BadZipFile, ZipFile


REQUIRED_FILES = {
    "lawa_rag/__init__.py",
    "lawa_rag/config.py",
    "lawa_rag/contracts.py",
    "lawa_rag/evaluation.py",
    "lawa_rag/indexing.py",
    "lawa_rag/py.typed",
    "lawa_rag/release.py",
    "lawa_rag/representation.py",
    "lawa_rag/retrieval.py",
    "lawa_rag/service.py",
    "pipeline/__init__.py",
    "pipeline/configs/default.yaml",
    "pipeline/configs/mbzuai_production.yaml",
}


def _expected_version() -> str:
    version_file = Path(__file__).resolve().parents[1] / "lawa_rag" / "_version.py"
    match = re.search(
        r'^__version__\s*=\s*["\']([^"\']+)["\']',
        version_file.read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"Could not read package version from {version_file}")
    return match.group(1)


def validate_wheel(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Wheel does not exist: {path}")
    try:
        with ZipFile(path) as archive:
            names = set(archive.namelist())
            metadata_names = sorted(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            if len(metadata_names) != 1:
                raise ValueError("Wheel must contain exactly one METADATA file")
            metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
    except BadZipFile as exc:
        raise ValueError(f"Not a valid wheel archive: {path}") from exc

    missing = sorted(REQUIRED_FILES - names)
    if missing:
        raise ValueError(f"Wheel is missing required package files: {missing}")
    forbidden = sorted(
        name
        for name in names
        if name.startswith("pipeline/tests/")
        or "/__pycache__/" in name
        or name.endswith((".pyc", ".env"))
    )
    if forbidden:
        raise ValueError(f"Wheel contains forbidden development files: {forbidden[:20]}")
    if metadata.get("Name") != "lawa-rag":
        raise ValueError(f"Unexpected distribution name: {metadata.get('Name')!r}")
    classifiers = set(metadata.get_all("Classifier") or [])
    if "Private :: Do Not Upload" not in classifiers:
        raise ValueError("Wheel is missing the private-distribution classifier")
    expected_version = _expected_version()
    if metadata.get("Version") != expected_version:
        raise ValueError(f"Unexpected distribution version: {metadata.get('Version')!r}")
    python_specifiers = {
        value.strip() for value in str(metadata.get("Requires-Python") or "").split(",")
    }
    if python_specifiers != {">=3.10", "<3.13"}:
        raise ValueError(
            f"Unexpected Python compatibility: {metadata.get('Requires-Python')!r}"
        )

    config_count = sum(name.startswith("pipeline/configs/") for name in names)
    return {
        "wheel": str(path.resolve()),
        "version": expected_version,
        "file_count": len(names),
        "bundled_config_count": config_count,
        "valid": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_wheel(args.wheel), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
