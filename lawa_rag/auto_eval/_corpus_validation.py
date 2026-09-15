"""Internal integrity checks for evaluation corpus snapshots."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def snapshot_errors(snapshot, expected_schema: str) -> list[str]:
    errors: list[str] = []
    if snapshot.schema_version != expected_schema:
        errors.append(f"unsupported schema version {snapshot.schema_version!r}")
    source_ids = [item.source_id for item in snapshot.sources]
    evidence_ids = [item.evidence_id for item in snapshot.evidence]
    action_ids = [item.action_id for item in snapshot.actions]
    id_groups = (("source", source_ids), ("evidence", evidence_ids), ("action", action_ids))
    for label, values in id_groups:
        _check_ids(errors, label, values)
    known_sources = set(source_ids)
    missing_sources = sorted(
        {unit.source_id for unit in snapshot.evidence if unit.source_id not in known_sources}
        | {action.source_id for action in snapshot.actions if action.source_id not in known_sources}
    )
    if missing_sources:
        errors.append(f"records reference unknown sources: {missing_sources[:5]}")
    if not snapshot.sources or not snapshot.evidence:
        errors.append("snapshot requires at least one source and one evidence unit")
    return errors


def _check_ids(errors: list[str], label: str, values: Sequence[str]) -> None:
    if not all(values):
        errors.append(f"{label} ids must be non-empty")
    duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
    if duplicates:
        errors.append(f"duplicate {label} ids: {duplicates[:5]}")
