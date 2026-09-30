"""Provider-neutral, immutable corpus snapshot used for evaluation generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Mapping, Sequence

from ._corpus_validation import snapshot_errors
from ._util import canonical_json, json_mapping, string_tuple, utc_now


CORPUS_SCHEMA_VERSION = "lawa.auto_eval.corpus.v1"


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    revision_id: str
    source_type: str
    title: str
    uri: str = ""
    language: str = "und"
    authority: str = "source"
    updated_at: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceRecord":
        return cls(
            source_id=str(value.get("source_id") or ""),
            revision_id=str(value.get("revision_id") or ""),
            source_type=str(value.get("source_type") or "unknown"),
            title=str(value.get("title") or "Untitled source"),
            uri=str(value.get("uri") or ""),
            language=str(value.get("language") or "und"),
            authority=str(value.get("authority") or "source"),
            updated_at=str(value.get("updated_at") or ""),
            metadata=json_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class EvidenceUnit:
    evidence_id: str
    source_id: str
    kind: str
    text: str
    locator: str = ""
    language: str = "und"
    section_id: str = ""
    parent_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceUnit":
        return cls(
            evidence_id=str(value.get("evidence_id") or ""),
            source_id=str(value.get("source_id") or ""),
            kind=str(value.get("kind") or "text"),
            text=str(value.get("text") or ""),
            locator=str(value.get("locator") or ""),
            language=str(value.get("language") or "und"),
            section_id=str(value.get("section_id") or ""),
            parent_id=str(value.get("parent_id") or ""),
            metadata=json_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class CorpusRelationship:
    source_id: str
    target_id: str
    relation: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusRelationship":
        return cls(
            source_id=str(value.get("source_id") or ""),
            target_id=str(value.get("target_id") or ""),
            relation=str(value.get("relation") or "related_to"),
            metadata=json_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class CorpusAction:
    action_id: str
    source_id: str
    label: str
    action_type: str
    target: str
    evidence_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusAction":
        return cls(
            action_id=str(value.get("action_id") or ""),
            source_id=str(value.get("source_id") or ""),
            label=str(value.get("label") or ""),
            action_type=str(value.get("action_type") or "navigate"),
            target=str(value.get("target") or ""),
            evidence_ids=string_tuple(value.get("evidence_ids")),
            metadata=json_mapping(value.get("metadata")),
        )


@dataclass(frozen=True)
class CorpusSnapshot:
    snapshot_id: str
    domain: str
    created_at: str
    sources: tuple[SourceRecord, ...]
    evidence: tuple[EvidenceUnit, ...]
    relationships: tuple[CorpusRelationship, ...] = ()
    actions: tuple[CorpusAction, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = CORPUS_SCHEMA_VERSION

    @classmethod
    def build(
        cls,
        *,
        domain: str,
        sources: Sequence[SourceRecord],
        evidence: Sequence[EvidenceUnit],
        relationships: Sequence[CorpusRelationship] = (),
        actions: Sequence[CorpusAction] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> "CorpusSnapshot":
        snapshot = cls(
            snapshot_id=_snapshot_id(domain, sources, evidence, relationships, actions),
            domain=str(domain or "generic"),
            created_at=utc_now(),
            sources=tuple(sources),
            evidence=tuple(evidence),
            relationships=tuple(relationships),
            actions=tuple(actions),
            metadata=json_mapping(metadata),
        )
        snapshot.require_valid()
        return snapshot

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CorpusSnapshot":
        snapshot = cls(
            snapshot_id=str(value.get("snapshot_id") or ""),
            domain=str(value.get("domain") or "generic"),
            created_at=str(value.get("created_at") or ""),
            sources=tuple(SourceRecord.from_dict(item) for item in value.get("sources") or []),
            evidence=tuple(EvidenceUnit.from_dict(item) for item in value.get("evidence") or []),
            relationships=tuple(
                CorpusRelationship.from_dict(item) for item in value.get("relationships") or []
            ),
            actions=tuple(CorpusAction.from_dict(item) for item in value.get("actions") or []),
            metadata=json_mapping(value.get("metadata")),
            schema_version=str(value.get("schema_version") or ""),
        )
        snapshot.require_valid()
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def source_map(self) -> dict[str, SourceRecord]:
        return {source.source_id: source for source in self.sources}

    def evidence_map(self) -> dict[str, EvidenceUnit]:
        return {unit.evidence_id: unit for unit in self.evidence}

    def summary(self, *, max_sources: int = 30) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "domain": self.domain,
            "source_count": len(self.sources),
            "evidence_count": len(self.evidence),
            "relationship_count": len(self.relationships),
            "action_count": len(self.actions),
            "source_types": sorted({item.source_type for item in self.sources}),
            "languages": sorted({item.language for item in self.sources}),
            "sample_titles": [item.title for item in self.sources[:max_sources]],
        }

    def validation_errors(self) -> tuple[str, ...]:
        errors = snapshot_errors(self, CORPUS_SCHEMA_VERSION)
        expected_id = _snapshot_id(
            self.domain,
            self.sources,
            self.evidence,
            self.relationships,
            self.actions,
        )
        if self.snapshot_id != expected_id:
            errors.append("snapshot identity does not match its content")
        return tuple(errors)

    def require_valid(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise ValueError("Invalid evaluation corpus: " + "; ".join(errors))


def _snapshot_id(domain, sources, evidence, relationships, actions) -> str:
    digest = sha256()
    digest.update(canonical_json(str(domain or "generic")).encode("utf-8"))
    groups = (
        ("sources", sorted(sources, key=lambda item: item.source_id)),
        ("evidence", sorted(evidence, key=lambda item: item.evidence_id)),
        (
            "relationships",
            sorted(relationships, key=lambda item: (item.source_id, item.target_id, item.relation)),
        ),
        ("actions", sorted(actions, key=lambda item: item.action_id)),
    )
    for label, records in groups:
        digest.update(f"\0{label}\0".encode("utf-8"))
        for record in records:
            digest.update(canonical_json(asdict(record)).encode("utf-8"))
            digest.update(b"\n")
    return f"eval-corpus:{digest.hexdigest()[:24]}"


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "CorpusAction",
    "CorpusRelationship",
    "CorpusSnapshot",
    "EvidenceUnit",
    "SourceRecord",
]
