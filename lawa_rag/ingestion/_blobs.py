"""Bounded, content-addressed local payload storage; no automatic destructive cleanup."""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path


class BlobStore:
    def __init__(self, root: Path):
        self.root = root

    def path(self, collection_key: str, content_hash: str) -> Path:
        for value in (collection_key, content_hash):
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("Invalid content address")
        return self.root / collection_key / content_hash[:2] / content_hash

    def put(self, collection_key: str, content_hash: str, body: bytes) -> None:
        path = self.path(collection_key, content_hash)
        if hashlib.sha256(body).hexdigest() != content_hash:
            raise ValueError("Payload digest mismatch")
        if path.exists():
            self.read(collection_key, content_hash, len(body))
            return
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(body)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_directory(path.parent)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def read(self, collection_key: str, content_hash: str, expected_size: int) -> bytes:
        with self.path(collection_key, content_hash).open("rb") as stream:
            body = stream.read(expected_size + 1)
        if len(body) != expected_size or hashlib.sha256(body).hexdigest() != content_hash:
            raise ValueError("Stored payload failed its size/digest check")
        return body


def _sync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
