#!/usr/bin/env python3
"""Build a minimal, deterministic runtime archive for verified release hydration."""

from __future__ import annotations

import argparse
import errno
import fcntl
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATOR = SCRIPT_DIR / "validate-release-artifacts.py"


class ArchiveError(RuntimeError):
    """Raised when a runtime release archive cannot be built safely."""


@contextmanager
def _run_lock(path: Path, *, timeout_seconds: float, held_env_var: str):
    if os.getenv(held_env_var, "false").lower() == "true":
        yield
        return
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o640)
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                if time.monotonic() >= deadline:
                    raise ArchiveError(f"timed out waiting for release run lock: {path}") from exc
                time.sleep(0.2)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_release(args: argparse.Namespace) -> dict[str, Any]:
    command = [
        sys.executable,
        str(VALIDATOR),
        "--active-release-file",
        str(args.active_release_file),
        "--runs-root",
        str(args.runs_root),
        "--no-require-storage-marker",
    ]
    if args.allow_waived_release:
        command.append("--allow-waived-release")
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise ArchiveError(completed.stderr.strip() or "release validation failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ArchiveError("release validator returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("mode") != "active":
        raise ArchiveError("release validator did not return a validated active release")
    return payload


def _archive_entries(
    *,
    payload: dict[str, Any],
    active_release_file: Path,
    runs_root: Path,
) -> list[tuple[Path, str]]:
    work_dir = Path(str(payload["work_dir"])).resolve()
    try:
        work_dir.relative_to(runs_root.resolve())
    except ValueError as exc:
        raise ArchiveError(f"validated work directory escapes runs root: {work_dir}") from exc
    run_id = str(payload["run_id"])
    files = {
        "active pointer": active_release_file.resolve(),
        "release manifest": Path(str(payload["release_manifest"])).resolve(),
        "resolved production config": Path(str(payload["resolved_config"])).resolve(),
        "upload manifest": Path(str(payload["upload_manifest"])).resolve(),
        "retrieval bundle": Path(str(payload["retrieval_bundle"])).resolve(),
        "lexical retrieval corpus": Path(str(payload["lexical_corpus"])).resolve(),
        "promoted assertion sidecar": Path(str(payload["promoted_assertions"])).resolve(),
        "knowledge graph": Path(str(payload["knowledge_graph"])).resolve(),
        "knowledge graph index": Path(str(payload["knowledge_graph_index"])).resolve(),
    }
    entries: list[tuple[Path, str]] = []
    for label, source in files.items():
        if not source.is_file() or source.is_symlink():
            raise ArchiveError(f"{label} is missing, not a regular file, or is a symlink: {source}")
        if label == "active pointer":
            archive_name = "mbzuai_main/active_release.json"
        else:
            try:
                relative = source.relative_to(work_dir)
            except ValueError as exc:
                raise ArchiveError(f"{label} escapes validated work directory: {source}") from exc
            archive_name = str(Path("runs") / "mbzuai_main" / run_id / relative)
        entries.append((source, archive_name))
    return entries


def _write_archive(entries: list[tuple[Path, str]], output_path: Path) -> tuple[int, int]:
    total_bytes = sum(source.stat().st_size for source, _ in entries)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
        delete=False,
    ) as raw_handle:
        temp_path = Path(raw_handle.name)
        try:
            with gzip.GzipFile(fileobj=raw_handle, mode="wb", filename="", mtime=0, compresslevel=6) as gzip_handle:
                # USTAR avoids hidden PAX/GNU extended-header payloads. The
                # hydrator rejects such metadata records before tarfile parses
                # them, keeping archive limits effective before allocation.
                with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                    for source, archive_name in sorted(entries, key=lambda item: item[1]):
                        info = tarfile.TarInfo(name=archive_name)
                        info.size = source.stat().st_size
                        info.mode = 0o640
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        with source.open("rb") as input_handle:
                            archive.addfile(info, input_handle)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
            os.replace(temp_path, output_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    return len(entries), total_bytes


def _paths_alias(first: Path, second: Path) -> bool:
    if first == second:
        return True
    try:
        return first.exists() and second.exists() and os.path.samefile(first, second)
    except OSError:
        return False


def _reject_output_aliases(
    *,
    output_path: Path,
    checksum_path: Path,
    protected_sources: list[Path],
) -> None:
    for destination, label in ((output_path, "archive output"), (checksum_path, "checksum output")):
        for source in protected_sources:
            if _paths_alias(destination, source):
                raise ArchiveError(f"{label} aliases a validated release source and would overwrite it")
    if _paths_alias(output_path, checksum_path):
        raise ArchiveError("archive output and checksum output must be distinct files")


def _write_checksum(path: Path, payload: str) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        mode="w",
        encoding="utf-8",
        delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise


def _build_locked(
    args: argparse.Namespace,
    *,
    active_release_file: Path,
    runs_root: Path,
    expected_run_id: str,
) -> dict[str, Any]:
    output_path = Path(args.output).expanduser().resolve()
    checksum_path = output_path.with_name(output_path.name + ".sha256")
    payload = _validate_release(args)
    if str(payload.get("run_id") or "") != expected_run_id:
        raise ArchiveError("active release changed while validating the locked archive input")
    pointer_fd, pointer_name = tempfile.mkstemp(prefix="mbzuai-active-release-", suffix=".json")
    os.close(pointer_fd)
    pointer_snapshot = Path(pointer_name)
    try:
        pointer_bytes = active_release_file.read_bytes()
        pointer_payload = json.loads(pointer_bytes)
        if not isinstance(pointer_payload, dict) or str(pointer_payload.get("run_id") or "") != expected_run_id:
            raise ArchiveError("active release changed before its pointer snapshot was captured")
        pointer_snapshot.write_bytes(pointer_bytes)
        entries = _archive_entries(
            payload=payload,
            active_release_file=pointer_snapshot,
            runs_root=runs_root,
        )
        _reject_output_aliases(
            output_path=output_path,
            checksum_path=checksum_path,
            protected_sources=[active_release_file, *(source for source, _ in entries)],
        )
        total_bytes = sum(source.stat().st_size for source, _ in entries)
        if total_bytes > args.max_uncompressed_bytes:
            raise ArchiveError(
                f"runtime artifacts total {total_bytes} bytes, exceeding max {args.max_uncompressed_bytes}"
            )
        file_count, uncompressed_bytes = _write_archive(entries, output_path)
    finally:
        pointer_snapshot.unlink(missing_ok=True)
    archive_sha256 = _sha256(output_path)
    _write_checksum(checksum_path, f"{archive_sha256}  {output_path.name}\n")
    return {
        "ok": True,
        "output": str(output_path),
        "checksum_file": str(checksum_path),
        "sha256": archive_sha256,
        "compressed_bytes": output_path.stat().st_size,
        "uncompressed_bytes": uncompressed_bytes,
        "file_count": file_count,
        "run_id": payload["run_id"],
        "release_id": payload["release_id"],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    active_release_file = Path(args.active_release_file).expanduser().resolve()
    runs_root = Path(args.runs_root).expanduser().resolve()
    try:
        initial_pointer = json.loads(active_release_file.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ArchiveError(f"could not read active release pointer: {active_release_file}") from exc
    initial_run_id = str(initial_pointer.get("run_id") or "").strip() if isinstance(initial_pointer, dict) else ""
    if not initial_run_id or Path(initial_run_id).name != initial_run_id:
        raise ArchiveError(f"active release pointer has an unsafe or missing run_id: {initial_run_id!r}")
    pointer_lock_path = active_release_file.with_name(active_release_file.name + ".lock")
    run_lock_path = runs_root / initial_run_id / ".run.lock"
    with _run_lock(
        pointer_lock_path,
        timeout_seconds=args.lock_timeout_seconds,
        held_env_var="RELEASE_POINTER_LOCK_HELD",
    ):
        with _run_lock(
            run_lock_path,
            timeout_seconds=args.lock_timeout_seconds,
            held_env_var="RELEASE_RUN_LOCK_HELD",
        ):
            try:
                locked_pointer = json.loads(active_release_file.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ArchiveError(
                    f"could not reread active release pointer under lock: {active_release_file}"
                ) from exc
            locked_run_id = (
                str(locked_pointer.get("run_id") or "").strip()
                if isinstance(locked_pointer, dict)
                else ""
            )
            if locked_run_id != initial_run_id:
                raise ArchiveError(
                    "active release changed while acquiring its locks: "
                    f"{initial_run_id!r} -> {locked_run_id!r}"
                )
            return _build_locked(
                args,
                active_release_file=active_release_file,
                runs_root=runs_root,
                expected_run_id=initial_run_id,
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--active-release-file",
        default=os.getenv("ACTIVE_RELEASE_FILE", "/data/releases/mbzuai_main/active_release.json"),
    )
    parser.add_argument(
        "--runs-root",
        default=os.getenv("RELEASE_RUNS_ROOT", "/data/releases/runs/mbzuai_main"),
    )
    parser.add_argument("--output", required=True, help="Output .tar.gz path")
    parser.add_argument(
        "--max-uncompressed-bytes",
        type=int,
        default=int(os.getenv("RELEASE_ARCHIVE_MAX_EXTRACTED_BYTES", str(2 * 1024**3))),
    )
    parser.add_argument("--allow-waived-release", action="store_true")
    parser.add_argument(
        "--lock-timeout-seconds",
        type=float,
        default=float(os.getenv("RELEASE_LOCK_TIMEOUT_SECONDS", "30")),
    )
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        if args.max_uncompressed_bytes <= 0:
            raise ArchiveError("--max-uncompressed-bytes must be positive")
        payload = build(args)
    except (ArchiveError, OSError, tarfile.TarError) as exc:
        print(f"Runtime release archive build failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
