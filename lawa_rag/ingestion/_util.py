"""Small serialization and input-validation helpers for ingestion."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def identifier(value: str, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", value):
        raise ValueError(f"Invalid {label}; use 1–128 letters, digits, dots, dashes or underscores")


def nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string without null bytes")


def safe_uri(value: str) -> None:
    nonempty(value, "uri")
    parsed = urlsplit(value)
    if not parsed.scheme or parsed.username is not None or parsed.password is not None:
        raise ValueError("Source URIs must have a scheme and must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Source URIs must not contain queries, tokens or fragments")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, limit: int = 64 * 1024 * 1024) -> Any:
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("JSON artifact exceeds the configured size limit")
    return json.loads(data)


def contained_file(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Artifact must be a regular file inside the source directory")
    return path
