"""Source-neutral identity, permission and synchronization contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Protocol

from ._util import digest, identifier, nonempty, safe_uri


@dataclass(frozen=True)
class AccessContext:
    """Trusted caller identity; a hosting application must authenticate it."""

    workspace_id: str
    principals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        identifier(self.workspace_id, "workspace_id")
        object.__setattr__(self, "principals", _principals(self.principals))


def _principals(values: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError("principals must be a sequence, not a string")
    for value in values:
        nonempty(value, "principal")
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class AccessPolicy:
    """Deny by default. Public means public within the requested workspace."""

    public: bool = False
    principals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.public, bool):
            raise ValueError("public must be a boolean")
        object.__setattr__(self, "principals", _principals(self.principals))

    def allows(self, context: AccessContext) -> bool:
        return self.public or bool(set(self.principals).intersection(context.principals))


@dataclass(frozen=True)
class Collection:
    workspace_id: str
    collection_id: str

    def __post_init__(self) -> None:
        identifier(self.workspace_id, "workspace_id")
        identifier(self.collection_id, "collection_id")

    @property
    def key(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class SyncPolicy:
    """Deletion reconciliation is opt-in and blocked on incomplete scans."""

    delete_missing: bool = False
    max_delete_fraction: float = 0.2
    allow_empty: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.delete_missing, bool) or not isinstance(self.allow_empty, bool):
            raise ValueError("Sync policy flags must be booleans")
        value = self.max_delete_fraction
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value <= 1:
            raise ValueError("max_delete_fraction must be between zero and one")


@dataclass(frozen=True)
class Source:
    source_id: str
    kind: str
    uri: str
    access: AccessPolicy = field(default_factory=AccessPolicy)
    processing_version: str = "default-v1"
    sync_policy: SyncPolicy = field(default_factory=SyncPolicy)

    def __post_init__(self) -> None:
        identifier(self.source_id, "source_id")
        identifier(self.kind, "kind")
        safe_uri(self.uri)
        nonempty(self.processing_version, "processing_version")
        valid_policies = isinstance(self.access, AccessPolicy) and isinstance(
            self.sync_policy, SyncPolicy,
        )
        if not valid_policies:
            raise ValueError("Source requires typed access and sync policies")

    def key(self, collection: Collection) -> str:
        return digest([collection.key, self.source_id])

    @property
    def fingerprint(self) -> str:
        return digest(asdict(self))


@dataclass(frozen=True)
class SourceItem:
    """One bounded payload. Locator identifies the original, not a working copy."""

    native_id: str
    body: bytes
    media_type: str
    locator: str
    title: str
    kind: str = "document"
    language: str = "und"
    processing_state: str = "raw"
    modified_at: str = ""
    metadata_json: str = "{}"
    access: AccessPolicy | None = None


@dataclass(frozen=True)
class SnapshotEnd:
    """Required final event; never emit it after a partial listing or read error."""

    authoritative: bool = True


class SourceConnector(Protocol):
    """Snapshot discovery; future cursor/CDC adapters must preserve this boundary."""

    source: Source

    @property
    def fingerprint(self) -> str:
        """Include options affecting discovery so unsafe resume is rejected."""
        ...

    def scan(self) -> Iterable[SourceItem | SnapshotEnd]:
        """Replayable scan ending in exactly one SnapshotEnd event."""
        ...


__all__ = [
    "AccessContext", "AccessPolicy", "Collection", "SnapshotEnd", "Source", "SourceConnector",
    "SourceItem", "SyncPolicy",
]
