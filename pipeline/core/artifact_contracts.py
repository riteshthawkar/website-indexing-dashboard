"""Helpers for stage-to-stage artifact contracts.

The pipeline still supports legacy ``previous_outputs`` keys, but production
stages should prefer cataloged artifacts. These helpers make that preference
explicit and keep fallback behavior consistent while the pipeline migrates.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from .artifacts import ArtifactRecord
from .base import StageContext


@dataclass(frozen=True)
class ArtifactContract:
    artifact_type: str
    role: Optional[str] = None
    legacy_output_key: Optional[str] = None
    label: Optional[str] = None
    must_exist: bool = True

    @property
    def display_name(self) -> str:
        if self.label:
            return self.label
        if self.role:
            return f"{self.artifact_type}/{self.role}"
        return self.artifact_type


@dataclass(frozen=True)
class ArtifactResolution:
    path: str
    source: str
    contract: ArtifactContract
    artifact_id: str = ""


def _path_ok(path: str | Path | None, *, must_exist: bool) -> bool:
    if not path:
        return False
    if not must_exist:
        return True
    return Path(path).is_file()


def latest_artifact_record(ctx: StageContext, contract: ArtifactContract) -> Optional[ArtifactRecord]:
    records = ctx.find_artifacts(artifact_type=contract.artifact_type, role=contract.role)
    for record in reversed(records):
        if _path_ok(record.local_path, must_exist=contract.must_exist):
            return record
    return None


def resolve_artifact_path(ctx: StageContext, contract: ArtifactContract) -> Optional[ArtifactResolution]:
    record = latest_artifact_record(ctx, contract)
    if record and record.local_path:
        return ArtifactResolution(
            path=str(Path(record.local_path).resolve()),
            source="artifact_catalog",
            artifact_id=record.artifact_id,
            contract=contract,
        )

    if contract.legacy_output_key:
        legacy_path = ctx.previous_outputs.get(contract.legacy_output_key)
        if _path_ok(legacy_path, must_exist=contract.must_exist):
            return ArtifactResolution(
                path=str(Path(str(legacy_path)).resolve()),
                source=f"previous_outputs.{contract.legacy_output_key}",
                contract=contract,
            )

    return None


def resolve_first_artifact_path(
    ctx: StageContext,
    contracts: Iterable[ArtifactContract],
) -> Optional[ArtifactResolution]:
    for contract in contracts:
        resolved = resolve_artifact_path(ctx, contract)
        if resolved is not None:
            return resolved
    return None


def missing_contract_message(contract: ArtifactContract) -> str:
    if contract.legacy_output_key:
        return (
            f"Required artifact {contract.display_name} is missing "
            f"(legacy output key: {contract.legacy_output_key})"
        )
    return f"Required artifact {contract.display_name} is missing"
