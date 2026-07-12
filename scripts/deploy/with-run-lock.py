#!/usr/bin/env python3
"""Hold a persistent run lock for the complete lifetime of a command."""

from __future__ import annotations

import argparse
import errno
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        parser.error("a command is required after --")
    if args.timeout_seconds < 0:
        parser.error("--timeout-seconds must be non-negative")

    lock_path = Path(args.lock_file).expanduser().resolve()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    # Never unlink this inode. Every indexing, validation, and promotion caller
    # must contend on the same persistent file.
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o640)
    deadline = time.monotonic() + args.timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    print(
                        f"Timed out waiting for exclusive run lock after {args.timeout_seconds}s: {lock_path}",
                        file=sys.stderr,
                    )
                    return 75
                time.sleep(0.2)
        completed = subprocess.run(command, check=False)
        return int(completed.returncode)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
