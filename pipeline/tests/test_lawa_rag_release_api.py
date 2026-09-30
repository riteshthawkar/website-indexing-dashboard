from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lawa_rag import ActiveRelease, ReleaseContractError, load_active_release


def _write_release(
    tmp_path: Path,
    *,
    pointer_overrides: dict[str, Any] | None = None,
    manifest_overrides: dict[str, Any] | None = None,
    relative_manifest: bool = False,
) -> tuple[Path, Path, Path]:
    work_dir = tmp_path / "run-1"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest = {
        "run_id": "run-1",
        "release_id": "release-1",
        "config_name": "test_config",
        "status": "passed",
        "work_dir": str(work_dir),
    }
    manifest.update(manifest_overrides or {})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pointer_path = tmp_path / "active_release.json"
    pointer = {
        "run_id": "run-1",
        "release_id": "release-1",
        "status": "passed",
        "active_release_manifest": (
            str(manifest_path.relative_to(tmp_path))
            if relative_manifest
            else str(manifest_path)
        ),
    }
    pointer.update(pointer_overrides or {})
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    return pointer_path, manifest_path, work_dir


def test_release_loader_accepts_relative_manifest_and_derives_work_dir(tmp_path: Path) -> None:
    pointer_path, manifest_path, work_dir = _write_release(
        tmp_path,
        manifest_overrides={"work_dir": "", "status": "passed_with_waiver"},
        pointer_overrides={"status": "passed_with_waiver"},
        relative_manifest=True,
    )

    release = ActiveRelease.load(pointer_path)

    assert release.pointer_path == pointer_path.resolve()
    assert release.manifest_path == manifest_path.resolve()
    assert release.work_dir == work_dir.resolve()
    assert release.status == "passed_with_waiver"
    assert release.config_name == "test_config"


@pytest.mark.parametrize("field", ["run_id", "release_id", "status"])
def test_release_loader_rejects_identity_mismatch(tmp_path: Path, field: str) -> None:
    pointer_path, _manifest_path, _work_dir = _write_release(
        tmp_path,
        pointer_overrides={field: "mismatch"},
    )
    with pytest.raises(ReleaseContractError, match=field):
        load_active_release(pointer_path)


def test_release_loader_rejects_missing_invalid_and_unpromotable_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="pointer not found"):
        load_active_release(tmp_path / "missing.json")

    invalid_pointer = tmp_path / "invalid.json"
    invalid_pointer.write_bytes(b"\xff\xfe")
    with pytest.raises(ReleaseContractError, match="pointer is not valid JSON"):
        load_active_release(invalid_pointer)

    array_pointer = tmp_path / "array.json"
    array_pointer.write_text("[]", encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="JSON object"):
        load_active_release(array_pointer)

    missing_manifest_field = tmp_path / "no-manifest.json"
    missing_manifest_field.write_text("{}", encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="missing active_release_manifest"):
        load_active_release(missing_manifest_field)

    pointer_path, manifest_path, _work_dir = _write_release(
        tmp_path / "failed",
        pointer_overrides={"status": "failed"},
        manifest_overrides={"status": "failed"},
    )
    with pytest.raises(ReleaseContractError, match="not promotable"):
        load_active_release(pointer_path)

    manifest_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ReleaseContractError, match="manifest is not valid JSON"):
        load_active_release(pointer_path)


def test_release_runtime_validation_delegates_with_a_defensive_config_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipeline.core.runtime_contract as runtime_contract

    pointer_path, _manifest_path, work_dir = _write_release(tmp_path)
    release = load_active_release(pointer_path)
    captured: dict[str, Any] = {}

    def fake_validate(
        config: dict[str, Any],
        received_work_dir: Path,
        *,
        validate_existing_release_manifest: bool,
    ) -> dict[str, Any]:
        captured["config"] = config
        captured["work_dir"] = received_work_dir
        captured["strict"] = validate_existing_release_manifest
        config["nested"]["value"] = "mutated inside validator"
        return {"validated": True}

    monkeypatch.setattr(runtime_contract, "validate_runtime_artifact_contract", fake_validate)
    config = {"nested": {"value": "original"}}

    assert release.validate_runtime(
        config,
        validate_existing_release_manifest=False,
    ) == {"validated": True}
    assert config == {"nested": {"value": "original"}}
    assert captured["work_dir"] == work_dir.resolve()
    assert captured["strict"] is False
