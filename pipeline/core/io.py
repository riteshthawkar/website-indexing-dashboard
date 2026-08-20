"""
Filesystem utilities shared across pipeline stages.
"""

import hashlib
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns the path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(filepath: str | Path, data: Any, indent: int = 2) -> None:
    """Write JSON atomically using a temp file + os.replace.

    Prevents partial writes if the process crashes mid-write.
    """
    filepath = str(filepath)
    dir_name = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        os.replace(tmp_path, filepath)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_write_text(
    filepath: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> None:
    """Write text atomically using a sibling temporary file."""

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(filepath.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, filepath)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def atomic_copy_file(source: str | Path, destination: str | Path) -> None:
    """Copy a file into place atomically without modifying the source."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(destination.parent), suffix=".tmp")
    os.close(fd)
    try:
        shutil.copy2(source, tmp_path)
        os.replace(tmp_path, destination)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def reset_stage_output_directory(path: str | Path, stage_work_dir: str | Path) -> Path:
    """Reset one named output directory after proving it belongs to a stage."""

    path = Path(path).resolve()
    stage_work_dir = Path(stage_work_dir).resolve()
    if path == stage_work_dir:
        raise ValueError("Refusing to reset the stage work directory itself")
    try:
        path.relative_to(stage_work_dir)
    except ValueError as exc:
        raise ValueError(f"Output directory escapes stage work directory: {path}") from exc
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json_safe(filepath: str | Path, default: Any = None) -> Any:
    """Load JSON file, returning *default* if the file is missing or corrupt."""
    filepath = Path(filepath)
    if not filepath.exists():
        return default
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load %s: %s", filepath, e)
        return default


def sha256_file(filepath: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hex digest of a file."""
    filepath = Path(filepath)
    digest = hashlib.sha256()
    with filepath.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def combine_sha256_digests(*digests: str) -> str:
    """Bind an ordered set of artifact SHA-256 digests into one digest."""

    combined = hashlib.sha256()
    for value in digests:
        normalized = str(value or "").strip().lower()
        if normalized:
            combined.update(normalized.encode("ascii"))
    return combined.hexdigest()


def safe_filename(name: str, max_length: int = 200) -> str:
    """Convert an arbitrary string into a safe filename."""
    # Replace problematic characters
    for ch in r'<>:"/\|?*':
        name = name.replace(ch, "_")
    # Collapse whitespace
    name = "_".join(name.split())
    return name[:max_length]
