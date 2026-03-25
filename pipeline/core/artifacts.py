"""
Artifact catalog helpers for modular pipeline stages.

Stages publish structured artifacts instead of relying only on flat output
keys. This keeps cross-stage contracts explicit and makes alternate scrapers,
cleaners, and converters pluggable without hidden filename conventions.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from .io import atomic_write_json, load_json_safe


ARTIFACT_CATALOG_FILENAME = "artifact_catalog.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_artifact_id(
    *,
    producer_stage: str,
    artifact_type: str,
    role: str,
    local_path: str = "",
    uri: str = "",
) -> str:
    raw = "|".join([producer_stage, artifact_type, role, local_path, uri])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    return f"{producer_stage}:{artifact_type}:{digest}"


@dataclass
class ArtifactRecord:
    artifact_id: str
    artifact_type: str
    role: str
    producer_stage: str
    uri: str
    local_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_artifact_ids: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return {
            key: value
            for key, value in data.items()
            if value not in (None, "", [], {})
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactRecord":
        return cls(
            artifact_id=str(data.get("artifact_id") or ""),
            artifact_type=str(data.get("artifact_type") or ""),
            role=str(data.get("role") or ""),
            producer_stage=str(data.get("producer_stage") or ""),
            uri=str(data.get("uri") or ""),
            local_path=data.get("local_path"),
            metadata=dict(data.get("metadata") or {}),
            source_artifact_ids=list(data.get("source_artifact_ids") or []),
            created_at=str(data.get("created_at") or _now_iso()),
        )


@dataclass
class ArtifactCatalog:
    version: int = 1
    records: List[ArtifactRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "records": [record.to_dict() for record in self.records],
        }

    @classmethod
    def from_dict(cls, data: Any) -> "ArtifactCatalog":
        if isinstance(data, list):
            records = [ArtifactRecord.from_dict(item) for item in data if isinstance(item, dict)]
            return cls(records=records)

        if not isinstance(data, dict):
            return cls()

        raw_records = data.get("records") or []
        records = [ArtifactRecord.from_dict(item) for item in raw_records if isinstance(item, dict)]
        return cls(version=int(data.get("version", 1)), records=records)

    def add(self, record: ArtifactRecord | Dict[str, Any]) -> ArtifactRecord:
        artifact = record if isinstance(record, ArtifactRecord) else ArtifactRecord.from_dict(record)
        for idx, existing in enumerate(self.records):
            if existing.artifact_id == artifact.artifact_id:
                self.records[idx] = artifact
                return artifact
        self.records.append(artifact)
        return artifact

    def extend(self, records: Iterable[ArtifactRecord | Dict[str, Any]]) -> List[ArtifactRecord]:
        added: List[ArtifactRecord] = []
        for record in records:
            added.append(self.add(record))
        return added

    def remove_many(self, artifact_ids: Iterable[str]) -> None:
        wanted = {artifact_id for artifact_id in artifact_ids if artifact_id}
        if not wanted:
            return
        self.records = [record for record in self.records if record.artifact_id not in wanted]

    def artifact_ids(self) -> Set[str]:
        return {record.artifact_id for record in self.records if record.artifact_id}

    def remove_by_producer_stage(self, producer_stage: str) -> List[str]:
        removed = [
            record.artifact_id
            for record in self.records
            if record.producer_stage == producer_stage
        ]
        self.remove_many(removed)
        return removed

    def prune_missing_local_paths(self) -> List[str]:
        removed: List[str] = []
        kept: List[ArtifactRecord] = []
        for record in self.records:
            local_path = record.local_path
            if local_path and not Path(local_path).exists():
                removed.append(record.artifact_id)
                continue
            kept.append(record)
        self.records = kept
        return removed

    def filter(
        self,
        *,
        artifact_type: Optional[str] = None,
        producer_stage: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        records = self.records
        if artifact_type:
            records = [record for record in records if record.artifact_type == artifact_type]
        if producer_stage:
            records = [record for record in records if record.producer_stage == producer_stage]
        if role:
            records = [record for record in records if record.role == role]
        return list(records)

    def find_by_local_path(self, path: str | Path) -> Optional[ArtifactRecord]:
        target = str(Path(path).resolve())
        for record in self.records:
            if not record.local_path:
                continue
            if str(Path(record.local_path).resolve()) == target:
                return record
        return None


def load_artifact_catalog(work_dir: str | Path) -> ArtifactCatalog:
    path = Path(work_dir) / ARTIFACT_CATALOG_FILENAME
    data = load_json_safe(path)
    if data is None:
        return ArtifactCatalog()
    return ArtifactCatalog.from_dict(data)


def save_artifact_catalog(catalog: ArtifactCatalog, work_dir: str | Path) -> Path:
    path = Path(work_dir) / ARTIFACT_CATALOG_FILENAME
    atomic_write_json(path, catalog.to_dict())
    return path


def build_artifact_record(
    *,
    artifact_type: str,
    role: str,
    producer_stage: str,
    uri: str,
    local_path: str | Path | None = None,
    metadata: Optional[Dict[str, Any]] = None,
    source_artifact_ids: Optional[Iterable[str]] = None,
    artifact_id: Optional[str] = None,
) -> ArtifactRecord:
    local_path_str = ""
    if local_path:
        local_path_str = str(Path(local_path).resolve())
    artifact_id = artifact_id or _stable_artifact_id(
        producer_stage=producer_stage,
        artifact_type=artifact_type,
        role=role,
        local_path=local_path_str,
        uri=uri,
    )
    return ArtifactRecord(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        role=role,
        producer_stage=producer_stage,
        uri=uri,
        local_path=local_path_str or None,
        metadata=dict(metadata or {}),
        source_artifact_ids=list(source_artifact_ids or []),
    )
