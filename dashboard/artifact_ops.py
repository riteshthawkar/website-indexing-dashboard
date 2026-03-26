"""
Artifact catalog and run-file browser helpers for the dashboard.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.artifacts import load_artifact_catalog


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".log",
    ".csv",
    ".tsv",
    ".html",
    ".xml",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def _resolve_run_path(work_dir: str | Path, relative_path: str | Path | None = None) -> Path:
    root = Path(work_dir).resolve()
    target = root if not relative_path else (root / relative_path).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must stay within run work dir: {relative_path}") from exc
    return target


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def list_artifacts(
    work_dir: str | Path,
    *,
    artifact_type: str | None = None,
    producer_stage: str | None = None,
    role: str | None = None,
    query: str | None = None,
    limit: int = 500,
) -> Dict[str, Any]:
    catalog = load_artifact_catalog(work_dir)
    records = catalog.records
    if artifact_type:
        records = [record for record in records if record.artifact_type == artifact_type]
    if producer_stage:
        records = [record for record in records if record.producer_stage == producer_stage]
    if role:
        records = [record for record in records if record.role == role]
    if query:
        query_lower = str(query).strip().lower()
        filtered = []
        for record in records:
            haystack = " ".join(
                [
                    record.artifact_id,
                    record.artifact_type,
                    record.role,
                    record.producer_stage,
                    record.uri,
                    record.local_path or "",
                    json.dumps(record.metadata, ensure_ascii=True),
                ]
            ).lower()
            if query_lower in haystack:
                filtered.append(record)
        records = filtered
    records = records[: max(1, int(limit))]
    items = []
    root = _resolve_run_path(work_dir)
    for record in records:
        local_path = Path(record.local_path).resolve() if record.local_path else None
        exists = bool(local_path and local_path.exists())
        relative_local_path = None
        if exists and local_path:
            try:
                relative_local_path = str(local_path.relative_to(root))
            except ValueError:
                relative_local_path = str(local_path)
        items.append(
            {
                **record.to_dict(),
                "exists": exists,
                "file_name": local_path.name if local_path else None,
                "relative_local_path": relative_local_path,
            }
        )
    return {
        "total": len(catalog.records),
        "returned": len(items),
        "artifact_types": sorted({record.artifact_type for record in catalog.records if record.artifact_type}),
        "producer_stages": sorted({record.producer_stage for record in catalog.records if record.producer_stage}),
        "roles": sorted({record.role for record in catalog.records if record.role}),
        "items": items,
    }


def list_run_files(work_dir: str | Path, relative_path: str | Path | None = None) -> Dict[str, Any]:
    target = _resolve_run_path(work_dir, relative_path)
    if not target.exists():
        raise FileNotFoundError(target)
    if not target.is_dir():
        raise ValueError(f"Path is not a directory: {target}")
    root = _resolve_run_path(work_dir)
    items: List[Dict[str, Any]] = []
    for child in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        entry = {
            "name": child.name,
            "relative_path": str(child.relative_to(root)),
            "is_dir": child.is_dir(),
            "size": child.stat().st_size if child.is_file() else None,
            "modified_at": _iso_mtime(child),
            "extension": child.suffix.lower() if child.is_file() else "",
            "preview_type": _preview_type(child) if child.is_file() else "directory",
        }
        items.append(entry)
    parent_path = None
    if target != root:
        parent_path = str(target.parent.relative_to(root))
    return {
        "root": str(root),
        "path": str(target.relative_to(root)) if target != root else "",
        "parent_path": parent_path,
        "items": items,
    }


def _preview_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "binary"


def read_run_file(work_dir: str | Path, relative_path: str | Path, *, max_bytes: int = 200_000) -> Dict[str, Any]:
    target = _resolve_run_path(work_dir, relative_path)
    if not target.exists():
        raise FileNotFoundError(target)
    if not target.is_file():
        raise ValueError(f"Path is not a file: {target}")
    size = target.stat().st_size
    preview_type = _preview_type(target)
    content = None
    parsed_json = None
    truncated = False
    if preview_type == "text":
        raw = target.read_text(encoding="utf-8", errors="replace")
        if len(raw.encode("utf-8")) > max_bytes:
            truncated = True
            raw = raw.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        content = raw
        if target.suffix.lower() == ".json":
            try:
                parsed_json = json.loads(target.read_text(encoding="utf-8"))
            except Exception:
                parsed_json = None
    return {
        "relative_path": str(target.relative_to(_resolve_run_path(work_dir))),
        "size": size,
        "modified_at": _iso_mtime(target),
        "preview_type": preview_type,
        "content": content,
        "parsed_json": parsed_json,
        "truncated": truncated,
    }
