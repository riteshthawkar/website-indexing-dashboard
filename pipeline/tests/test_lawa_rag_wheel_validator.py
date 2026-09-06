from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from ci.validate_package_wheel import REQUIRED_FILES, validate_wheel


def _metadata(
    *,
    name: str = "lawa-rag",
    version: str = "0.1.0",
    requires_python: str = ">=3.10,<3.13",
    private: bool = True,
) -> str:
    value = (
        "Metadata-Version: 2.4\n"
        f"Name: {name}\n"
        f"Version: {version}\n"
        f"Requires-Python: {requires_python}\n"
    )
    if private:
        value += "Classifier: Private :: Do Not Upload\n"
    return value + "\n"


def _wheel(
    path: Path,
    *,
    files: set[str] | None = None,
    metadata: str | None = None,
    extra_files: dict[str, str] | None = None,
) -> Path:
    with ZipFile(path, "w") as archive:
        for name in sorted(REQUIRED_FILES if files is None else files):
            archive.writestr(name, "")
        archive.writestr(
            "lawa_rag-0.1.0.dist-info/METADATA",
            _metadata() if metadata is None else metadata,
        )
        for name, value in (extra_files or {}).items():
            archive.writestr(name, value)
    return path


def test_wheel_validator_accepts_complete_private_distribution(tmp_path: Path) -> None:
    report = validate_wheel(_wheel(tmp_path / "lawa_rag-0.1.0-py3-none-any.whl"))

    assert report["valid"] is True
    assert report["version"] == "0.1.0"
    assert report["bundled_config_count"] == 2
    assert report["file_count"] == len(REQUIRED_FILES) + 1


def test_wheel_validator_rejects_missing_public_contract_file(tmp_path: Path) -> None:
    files = set(REQUIRED_FILES)
    files.remove("lawa_rag/indexing.py")

    with pytest.raises(ValueError, match="missing required package files"):
        validate_wheel(_wheel(tmp_path / "missing.whl", files=files))


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "pipeline/tests/test_internal.py",
        "lawa_rag/__pycache__/indexing.pyc",
        "lawa_rag/secret.env",
    ],
)
def test_wheel_validator_rejects_development_or_secret_files(
    tmp_path: Path,
    forbidden_name: str,
) -> None:
    with pytest.raises(ValueError, match="forbidden development files"):
        validate_wheel(
            _wheel(
                tmp_path / "forbidden.whl",
                extra_files={forbidden_name: "must not ship"},
            )
        )


@pytest.mark.parametrize(
    ("metadata", "expected_message"),
    [
        (_metadata(name="different"), "Unexpected distribution name"),
        (_metadata(private=False), "private-distribution classifier"),
        (_metadata(version="9.9.9"), "Unexpected distribution version"),
        (_metadata(requires_python=">=3.11"), "Unexpected Python compatibility"),
    ],
)
def test_wheel_validator_rejects_incompatible_metadata(
    tmp_path: Path,
    metadata: str,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        validate_wheel(_wheel(tmp_path / "metadata.whl", metadata=metadata))


def test_wheel_validator_accepts_normalized_python_specifier_order(tmp_path: Path) -> None:
    report = validate_wheel(
        _wheel(
            tmp_path / "normalized.whl",
            metadata=_metadata(requires_python="<3.13,>=3.10"),
        )
    )
    assert report["valid"] is True


def test_wheel_validator_rejects_multiple_metadata_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one METADATA"):
        validate_wheel(
            _wheel(
                tmp_path / "multiple.whl",
                extra_files={"another.dist-info/METADATA": _metadata()},
            )
        )


def test_wheel_validator_rejects_non_wheel_and_missing_paths(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.whl"
    invalid.write_text("not a zip archive", encoding="utf-8")
    with pytest.raises(ValueError, match="Not a valid wheel archive"):
        validate_wheel(invalid)
    with pytest.raises(FileNotFoundError, match="Wheel does not exist"):
        validate_wheel(tmp_path / "missing.whl")
