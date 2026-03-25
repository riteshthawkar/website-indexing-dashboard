"""
Filesystem utilities shared across pipeline stages.
"""

import hashlib
import json
import logging
import os
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


def safe_filename(name: str, max_length: int = 200) -> str:
    """Convert an arbitrary string into a safe filename."""
    # Replace problematic characters
    for ch in r'<>:"/\|?*':
        name = name.replace(ch, "_")
    # Collapse whitespace
    name = "_".join(name.split())
    return name[:max_length]
