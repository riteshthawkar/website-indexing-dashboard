"""Bridge completed generic website runs into collections without recrawling them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit, urlunsplit

from ._files import read_stable, stamp
from ._util import canonical_json, contained_file, digest
from ._website_run import check_unchanged, load_run
from .contracts import SnapshotEnd, Source, SourceItem


@dataclass(frozen=True)
class WebsiteRunConnector:
    source: Source
    run_dir: str | Path
    max_file_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        uri = urlsplit(self.source.uri)
        if self.source.kind != "website" or uri.scheme not in {"https", "http"} or not uri.hostname:
            raise ValueError("WebsiteRunConnector requires a website source with an HTTP(S) URI")
        if (isinstance(self.max_file_bytes, bool) or not isinstance(self.max_file_bytes, int)
                or self.max_file_bytes < 1):
            raise ValueError("max_file_bytes must be a positive integer")
        object.__setattr__(self, "run_dir", Path(self.run_dir).expanduser().resolve())

    @property
    def fingerprint(self) -> str:
        return digest(["website-run-v1", str(self.run_dir), self.max_file_bytes])

    def scan(self) -> Iterable[SourceItem | SnapshotEnd]:
        root = Path(self.run_dir)
        artifacts, hosts, fingerprint = load_run(root, self.source)
        files = []
        for artifact in artifacts:
            item, path, info = self._read(root, artifact, hosts)
            files.append((path, stamp(info)))
            yield item
        if any(stamp(path.stat()) != expected for path, expected in files):
            raise ValueError("Website content changed during import")
        check_unchanged(root, fingerprint)
        yield SnapshotEnd()

    def _read(self, root: Path, artifact: dict, hosts: tuple[str, ...]):
        metadata = artifact.get("metadata") or {}
        raw_url = metadata.get("canonical_url") or metadata.get("source_url") or ""
        url = urlsplit(raw_url)
        if (url.scheme not in {"http", "https"} or url.hostname not in hosts
                or url.username is not None or url.password is not None):
            raise ValueError("Markdown artifact has no permitted source URL")
        locator = urlunsplit((url.scheme, url.netloc, url.path or "/", url.query, ""))
        path = contained_file(root, artifact.get("local_path") or "")
        body, info = read_stable(path, self.max_file_bytes)
        text = body.decode("utf-8", errors="strict")
        if not text.strip():
            raise ValueError("Empty website document cannot become ready evidence")
        item = SourceItem(
            native_id=locator, body=body, media_type="text/markdown", locator=locator,
            title=metadata.get("document_title") or url.path.strip("/") or url.hostname,
            language=metadata.get("language") or "und", processing_state="ready",
            metadata_json=canonical_json({
                "original_type": metadata.get("source_type") or "webpage",
                "conversion_backend": metadata.get("backend") or "unknown",
            }),
        )
        return item, path, info


__all__ = ["WebsiteRunConnector"]
