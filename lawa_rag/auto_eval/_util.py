"""Small serialization and identity helpers for automatic evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable as IterableABC
from typing import Any, Iterable, Mapping, Sequence


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, IterableABC):
        values = [value]
    elif isinstance(value, Mapping):
        values = [value]
    else:
        values = list(value)
    return tuple(dict.fromkeys(clean_text(item) for item in values if clean_text(item)))


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    material = "\0".join(canonical_json(part) for part in parts)
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}:{digest}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def atomic_write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    lines = [canonical_json(dict(row)) for row in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    if not input_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {input_path}:{number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {input_path}:{number}")
        rows.append(value)
    return rows


_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", flags=re.UNICODE)


def text_tokens(*values: Any) -> frozenset[str]:
    text = " ".join(clean_text(value).casefold() for value in values)
    return frozenset(token for token in _TOKEN_RE.findall(text) if len(token) > 2)


def compact_mapping(value: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: value[key] for key in keys if value.get(key) not in (None, "", [], {})}
