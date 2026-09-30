"""S3/Spaces source connector; credentials use the SDK chain, never source manifests."""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Iterable
from urllib.parse import quote, unquote, urlsplit

from ._object_store import download, inventory
from ._util import canonical_json, digest, safe_uri
from .contracts import SnapshotEnd, Source, SourceItem
from .folders import DEFAULT_EXTENSIONS


@dataclass(frozen=True)
class S3Connector:
    source: Source
    endpoint_url: str | None = None
    region_name: str | None = None
    profile_name: str | None = None
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    max_objects: int = 100_000
    max_object_bytes: int = 32 * 1024 * 1024
    max_listing_pages: int = 1000
    max_snapshot_bytes: int = 1024 * 1024 * 1024
    client: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        uri = urlsplit(self.source.uri)
        if self.source.kind != "s3" or uri.scheme != "s3" or not uri.netloc:
            raise ValueError("S3Connector requires an s3 source with an s3://bucket/prefix/ URI")
        if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", uri.netloc):
            raise ValueError("Invalid bucket name")
        if self.prefix and not self.prefix.endswith("/"):
            raise ValueError("Use a directory prefix ending in / to avoid adjacent-prefix matches")
        if self.endpoint_url:
            safe_uri(self.endpoint_url)
            endpoint = urlsplit(self.endpoint_url)
            if (endpoint.scheme != "https" or not endpoint.hostname
                    or endpoint.path not in {"", "/"}):
                raise ValueError("Object-store endpoint must be an HTTPS origin")
        if self.client is not None:
            actual = self.client.meta.endpoint_url.rstrip("/")
            if self.endpoint_url and actual != self.endpoint_url.rstrip("/"):
                raise ValueError("Injected client endpoint changed; use a matching client")
        self._validate_limits()

    def _validate_limits(self) -> None:
        if isinstance(self.extensions, str) or not self.extensions:
            raise ValueError("A non-empty extension allowlist is required")
        extensions = tuple(sorted(set(value.lower() for value in self.extensions)))
        if any(not re.fullmatch(r"\.[a-z0-9]{1,12}", value) for value in extensions):
            raise ValueError("Invalid extension allowlist")
        object.__setattr__(self, "extensions", extensions)
        for value in (self.max_objects, self.max_object_bytes, self.max_listing_pages,
                      self.max_snapshot_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("Object-store limits must be positive integers")

    @property
    def bucket(self) -> str:
        return urlsplit(self.source.uri).netloc

    @property
    def prefix(self) -> str:
        return unquote(urlsplit(self.source.uri).path.lstrip("/"))

    @property
    def binding_key(self) -> str:
        endpoint = self.client.meta.endpoint_url if self.client is not None else self.endpoint_url
        return digest(["s3", (endpoint or "aws").rstrip("/"), self.bucket, self.prefix])

    @property
    def fingerprint(self) -> str:
        return digest([self.binding_key, self.region_name, self.profile_name, self.extensions,
                       self.max_objects, self.max_object_bytes, self.max_listing_pages,
                       self.max_snapshot_bytes, "s3-snapshot-v1"])

    def _client(self):
        import boto3
        from botocore.config import Config

        return boto3.Session(profile_name=self.profile_name).client(
            "s3", endpoint_url=self.endpoint_url, region_name=self.region_name,
            config=Config(signature_version="s3v4", connect_timeout=10, read_timeout=30,
                          retries={"mode": "standard", "total_max_attempts": 3}),
        )

    def scan(self) -> Iterable[SourceItem | SnapshotEnd]:
        client = self.client if self.client is not None else self._client()
        try:
            initial = self._inventory(client)
            for key, identity in sorted(initial.items()):
                body = download(client, self.bucket, key, identity, self.max_object_bytes)
                yield self._item(key, body, identity)
            current = self._inventory(client)
            if initial != current:
                raise ValueError("Object inventory changed during sync; retry a stable snapshot")
            yield SnapshotEnd()
        finally:
            if self.client is None:
                client.close()

    def _inventory(self, client):
        return inventory(client, self.bucket, self.prefix, self.extensions, self.max_objects,
                         self.max_listing_pages, self.max_snapshot_bytes)

    def _item(self, key: str, body: bytes, identity: tuple) -> SourceItem:
        path = PurePosixPath(key)
        suffix = path.suffix.lower()
        media = "text/markdown" if suffix == ".md" else mimetypes.guess_type(key)[0]
        ready = suffix in {".md", ".txt"} and bool(body.strip())
        if ready:
            body.decode("utf-8", errors="strict")
        kind = "table" if suffix in {".csv", ".xlsx"} else "document"
        if suffix == ".json":
            kind = "record"
        elif (media or "").startswith("image/"):
            kind = "image"
        return SourceItem(
            native_id=key, body=body, media_type=media or "application/octet-stream",
            locator=f"s3://{self.bucket}/{quote(key, safe='/')}", title=path.stem, kind=kind,
            processing_state="ready" if ready else "raw", modified_at=identity[2],
            metadata_json=canonical_json({"object_store_binding": self.binding_key}),
        )


__all__ = ["S3Connector"]
