"""Local-folder connector. Binary and structured files remain raw until converted."""

from __future__ import annotations

import mimetypes
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlsplit

from ._files import inventory, read_stable, stamp
from ._util import digest
from .contracts import SnapshotEnd, Source, SourceItem


DEFAULT_EXTENSIONS = (
    ".md", ".txt", ".html", ".pdf", ".docx", ".pptx", ".xlsx", ".csv", ".json",
    ".png", ".jpg", ".jpeg", ".webp",
)


@dataclass(frozen=True)
class LocalFolderConnector:
    source: Source
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    max_file_bytes: int = 32 * 1024 * 1024
    max_files: int = 100_000

    def __post_init__(self) -> None:
        uri = urlsplit(self.source.uri)
        if self.source.kind != "folder" or uri.scheme != "file" or uri.netloc:
            raise ValueError("LocalFolderConnector requires a folder source with a local file URI")
        if not self.extensions or isinstance(self.extensions, str):
            raise ValueError("Provide a non-empty sequence of allowed extensions")
        values = tuple(sorted(set(value.lower() for value in self.extensions)))
        if any(not re.fullmatch(r"\.[a-z0-9]{1,12}", value) for value in values):
            raise ValueError("Invalid extension allowlist")
        object.__setattr__(self, "extensions", values)
        for limit in (self.max_file_bytes, self.max_files):
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValueError("Folder limits must be positive integers")

    @property
    def root(self) -> Path:
        return Path(unquote(urlsplit(self.source.uri).path)).resolve()

    @property
    def fingerprint(self) -> str:
        return digest(["folder-v1", str(self.root), self.extensions,
                       self.max_file_bytes, self.max_files])

    def scan(self) -> Iterable[SourceItem | SnapshotEnd]:
        root = self.root
        initial = inventory(root, self.extensions, self.max_files)
        for relative, expected in initial:
            path = root / relative
            if not path.resolve().is_relative_to(root):
                raise ValueError("Source path escaped the folder")
            body, info = read_stable(path, self.max_file_bytes)
            if stamp(info) != expected:
                raise ValueError("Folder changed during scan; retry with a stable snapshot")
            yield _item(path, relative, body, info)
        if initial != inventory(root, self.extensions, self.max_files):
            raise ValueError("Folder inventory changed during scan; no deletions were applied")
        yield SnapshotEnd()


def _item(path: Path, relative: str, body: bytes, info) -> SourceItem:
    suffix = path.suffix.lower()
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    if suffix == ".md":
        media_type = "text/markdown"
    ready = suffix in {".md", ".txt"}
    if ready:
        body.decode("utf-8", errors="strict")
        ready = bool(body.strip())
    kind = "document"
    if suffix in {".csv", ".xlsx"}:
        kind = "table"
    elif suffix == ".json":
        kind = "record"
    elif media_type.startswith("image/"):
        kind = "image"
    return SourceItem(
        native_id=relative, body=body, media_type=media_type, locator=path.as_uri(),
        title=path.stem, kind=kind, processing_state="ready" if ready else "raw",
        modified_at=datetime.fromtimestamp(info.st_mtime, timezone.utc).isoformat(),
    )


__all__ = ["DEFAULT_EXTENSIONS", "LocalFolderConnector"]
