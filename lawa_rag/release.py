"""Read-only release inspection and runtime contract validation."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


class ReleaseContractError(ValueError):
    """Raised when an active-release pointer is missing or inconsistent."""


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseContractError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ReleaseContractError(f"{label} must be a JSON object: {path}")
    return value


@dataclass(frozen=True)
class ActiveRelease:
    """An immutable view of a promoted release pointer and its manifest."""

    pointer_path: Path
    manifest_path: Path
    _pointer: Mapping[str, Any]
    _manifest: Mapping[str, Any]

    @classmethod
    def load(cls, pointer_path: str | Path) -> "ActiveRelease":
        resolved_pointer = Path(pointer_path).expanduser().resolve()
        pointer = _load_json_object(resolved_pointer, label="Active release pointer")
        raw_manifest_path = str(pointer.get("active_release_manifest") or "").strip()
        if not raw_manifest_path:
            raise ReleaseContractError(
                f"Active release pointer is missing active_release_manifest: {resolved_pointer}"
            )
        manifest_path = Path(raw_manifest_path).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = resolved_pointer.parent / manifest_path
        manifest_path = manifest_path.resolve()
        manifest = _load_json_object(manifest_path, label="Release manifest")

        for field in ("run_id", "release_id", "status"):
            pointer_value = str(pointer.get(field) or "").strip()
            manifest_value = str(manifest.get(field) or "").strip()
            if pointer_value and manifest_value and pointer_value != manifest_value:
                raise ReleaseContractError(
                    f"Active release {field} does not match its manifest "
                    f"({pointer_value!r} != {manifest_value!r})"
                )
        if str(manifest.get("status") or "") not in {"passed", "passed_with_waiver"}:
            raise ReleaseContractError(
                f"Release manifest is not promotable: {manifest_path}"
            )
        return cls(
            pointer_path=resolved_pointer,
            manifest_path=manifest_path,
            _pointer=deepcopy(pointer),
            _manifest=deepcopy(manifest),
        )

    @property
    def run_id(self) -> str:
        return str(self._manifest.get("run_id") or self._pointer.get("run_id") or "")

    @property
    def release_id(self) -> str:
        return str(self._manifest.get("release_id") or self._pointer.get("release_id") or "")

    @property
    def status(self) -> str:
        return str(self._manifest.get("status") or self._pointer.get("status") or "")

    @property
    def config_name(self) -> str:
        return str(self._manifest.get("config_name") or "")

    @property
    def work_dir(self) -> Path:
        configured = str(self._manifest.get("work_dir") or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return self.manifest_path.parent.parent

    @property
    def pointer(self) -> dict[str, Any]:
        return deepcopy(dict(self._pointer))

    @property
    def manifest(self) -> dict[str, Any]:
        return deepcopy(dict(self._manifest))

    def validate_runtime(
        self,
        config: Mapping[str, Any],
        *,
        validate_existing_release_manifest: bool = True,
    ) -> dict[str, Any]:
        """Validate all immutable runtime artifact bindings for this release."""

        from pipeline.core.runtime_contract import validate_runtime_artifact_contract

        return validate_runtime_artifact_contract(
            deepcopy(dict(config)),
            self.work_dir,
            validate_existing_release_manifest=validate_existing_release_manifest,
        )


def load_active_release(pointer_path: str | Path) -> ActiveRelease:
    return ActiveRelease.load(pointer_path)


__all__ = ["ActiveRelease", "ReleaseContractError", "load_active_release"]
