"""Crash-resilient incremental cache storage for long-running pipeline stages."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from pipeline.core.io import atomic_write_json

logger = logging.getLogger(__name__)


class IncrementalJsonObjectCache:
    """Maintain a JSON-object cache without rewriting it after every update.

    Completed entries are appended to a newline-delimited journal and fsynced.
    On successful stage completion, :meth:`compact` atomically materializes the
    complete JSON snapshot and removes the journal. Replaying an existing
    journal is idempotent because later values replace earlier values by key.
    """

    def __init__(self, snapshot_path: str | Path) -> None:
        self.snapshot_path = Path(snapshot_path)
        self.journal_path = self.snapshot_path.with_name(
            f"{self.snapshot_path.stem}.journal.jsonl"
        )
        self._payload = self._load()

    @property
    def payload(self) -> Dict[str, Any]:
        return self._payload

    def _load(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if self.snapshot_path.exists():
            try:
                with self.snapshot_path.open("r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Incremental cache snapshot is unreadable: {self.snapshot_path}"
                ) from exc
            if not isinstance(loaded, dict):
                raise ValueError(
                    f"Incremental cache snapshot must contain a JSON object: {self.snapshot_path}"
                )
            payload.update(loaded)

        if not self.journal_path.exists():
            return payload

        truncate_at: int | None = None
        append_separator = False
        valid_bytes = 0
        try:
            with self.journal_path.open("rb") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    try:
                        line = raw_line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        if not raw_line.endswith(b"\n"):
                            truncate_at = valid_bytes
                            logger.warning(
                                "Discarding incomplete trailing cache journal record at %s:%d",
                                self.journal_path,
                                line_number,
                            )
                            break
                        raise ValueError(
                            f"Incremental cache journal is not UTF-8 at "
                            f"{self.journal_path}:{line_number}"
                        ) from exc
                    if not line.strip():
                        valid_bytes += len(raw_line)
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        if not raw_line.endswith(b"\n"):
                            truncate_at = valid_bytes
                            logger.warning(
                                "Discarding incomplete trailing cache journal record at %s:%d",
                                self.journal_path,
                                line_number,
                            )
                            break
                        raise ValueError(
                            f"Incremental cache journal is corrupt at "
                            f"{self.journal_path}:{line_number}"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(
                            f"Incremental cache journal record must be an object at "
                            f"{self.journal_path}:{line_number}"
                        )
                    key = str(record.get("key") or "").strip()
                    if not key or "value" not in record:
                        raise ValueError(
                            f"Incremental cache journal record is invalid at "
                            f"{self.journal_path}:{line_number}"
                        )
                    payload[key] = record["value"]
                    valid_bytes += len(raw_line)
                    if not raw_line.endswith(b"\n"):
                        append_separator = True
        except OSError as exc:
            raise ValueError(
                f"Incremental cache journal is unreadable: {self.journal_path}"
            ) from exc

        if truncate_at is not None:
            try:
                with self.journal_path.open("r+b") as handle:
                    handle.truncate(truncate_at)
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise ValueError(
                    f"Incremental cache journal could not be repaired: {self.journal_path}"
                ) from exc
        elif append_separator:
            try:
                with self.journal_path.open("ab") as handle:
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except OSError as exc:
                raise ValueError(
                    f"Incremental cache journal could not be repaired: {self.journal_path}"
                ) from exc
        return payload

    def put(self, key: str, value: Any) -> None:
        normalized_key = str(key or "").strip()
        if not normalized_key:
            raise ValueError("Incremental cache keys must be non-empty")

        record = json.dumps(
            {"key": normalized_key, "value": value},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as handle:
            handle.write(record)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._payload[normalized_key] = value

    def compact(self) -> None:
        atomic_write_json(self.snapshot_path, self._payload)
        try:
            self.journal_path.unlink()
        except FileNotFoundError:
            pass
