"""Bounded reads and inventory checks for local snapshot connectors."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def stamp(value: os.stat_result) -> tuple[int, ...]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def read_stable(path: Path, max_bytes: int) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("Source is not a regular file within the size limit")
        body = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    if len(body) > max_bytes or stamp(before) != stamp(after) or stamp(after) != stamp(path.stat()):
        raise ValueError("Source file changed while being read; retry with a stable snapshot")
    return body, after


def _raise_walk_error(error: OSError) -> None:
    raise error


def inventory(root: Path, extensions: tuple[str, ...], max_files: int) -> tuple:
    if not root.is_dir():
        raise FileNotFoundError("Source directory is unavailable")
    result = []
    for directory, folders, files in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        folders[:] = sorted(name for name in folders if not name.startswith("."))
        for name in folders:
            if (Path(directory) / name).is_symlink():
                raise ValueError("Directory symlinks are not supported in source snapshots")
        for name in sorted(files):
            path = Path(directory) / name
            if name.startswith(".") or path.suffix.lower() not in extensions:
                continue
            if path.is_symlink() or not path.is_file():
                raise ValueError("Source files must be regular files, not symlinks")
            result.append((path.relative_to(root).as_posix(), stamp(path.stat())))
            if len(result) > max_files:
                raise ValueError("Folder file limit exceeded; snapshot was not truncated")
    return tuple(sorted(result))
