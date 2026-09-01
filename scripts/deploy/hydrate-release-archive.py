#!/usr/bin/env python3
"""Download and safely hydrate a checksum-pinned retrieval release archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
S3_BUCKET_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]{1,61}[a-z0-9])?$")
MARKER_NAME = ".mbzuai-release-storage"
ALLOWED_ARCHIVE_ROOTS = {"mbzuai_main", "runs"}
SCRIPT_DIR = Path(__file__).resolve().parent
VALIDATOR = SCRIPT_DIR / "validate-release-artifacts.py"


class HydrationError(RuntimeError):
    """Raised when a remote release archive is not safe to install."""


def _bool(value: str | bool | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise HydrationError(f"invalid boolean value: {value!r}")


def _allowed_hosts(raw: str) -> set[str]:
    return {host.strip().lower().rstrip(".") for host in raw.split(",") if host.strip()}


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise HydrationError("release archive download exceeded its end-to-end deadline")
    return remaining


def _set_stream_timeout(stream: Any, timeout_seconds: float) -> None:
    """Best-effort per-read deadline propagation for urllib and botocore streams."""

    setter = getattr(stream, "set_socket_timeout", None)
    if callable(setter):
        try:
            setter(max(0.001, timeout_seconds))
            return
        except (AttributeError, OSError):
            # Botocore's StreamingBody can detach its urllib3 socket as soon as
            # ContentLength bytes have been consumed. Treat timeout propagation
            # as best effort in that state; the client-level read timeout and
            # the explicit end-to-end deadline remain enforced below.
            pass
    fp = getattr(stream, "fp", None)
    raw = getattr(fp, "raw", None)
    candidates = (
        getattr(raw, "_sock", None),
        getattr(fp, "_sock", None),
        getattr(getattr(raw, "_fp", None), "fp", None),
    )
    for candidate in candidates:
        socket_setter = getattr(candidate, "settimeout", None)
        if callable(socket_setter):
            socket_setter(max(0.001, timeout_seconds))
            return


def _validate_url(url: str, *, allowed_hosts: set[str], allow_http: bool) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    schemes = {"https"} | ({"http"} if allow_http else set())
    if parsed.scheme.lower() not in schemes:
        raise HydrationError("release archive URL must use HTTPS")
    if not parsed.hostname:
        raise HydrationError("release archive URL is missing a hostname")
    if parsed.username or parsed.password:
        raise HydrationError("release archive URL must not contain userinfo")
    if parsed.fragment:
        raise HydrationError("release archive URL must not contain a fragment")
    hostname = parsed.hostname.lower().rstrip(".")
    if not allowed_hosts:
        raise HydrationError("RELEASE_ARCHIVE_ALLOWED_HOSTS must contain the archive hostname")
    if hostname not in allowed_hosts:
        raise HydrationError(f"release archive hostname is not allowlisted: {hostname}")
    return parsed


def _validate_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme.lower() != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
        raise HydrationError("RELEASE_ARCHIVE_S3_URI must use s3://bucket/object-key")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise HydrationError("RELEASE_ARCHIVE_S3_URI must not contain credentials, a query, or a fragment")
    bucket = parsed.netloc.strip().lower()
    if not 3 <= len(bucket) <= 63 or not S3_BUCKET_PATTERN.fullmatch(bucket) or ".." in bucket:
        raise HydrationError("RELEASE_ARCHIVE_S3_URI contains an invalid bucket name")
    key = urllib.parse.unquote(parsed.path.lstrip("/"))
    if not key or any(ord(character) < 32 or ord(character) == 127 for character in key):
        raise HydrationError("RELEASE_ARCHIVE_S3_URI contains an invalid object key")
    return bucket, key


def _validate_s3_endpoint(endpoint_url: str, *, allowed_hosts: set[str]) -> urllib.parse.ParseResult:
    parsed = _validate_url(endpoint_url, allowed_hosts=allowed_hosts, allow_http=False)
    if parsed.path not in {"", "/"} or parsed.query:
        raise HydrationError("release archive S3 endpoint must not contain a path or query")
    if parsed.port not in {None, 443}:
        raise HydrationError("release archive S3 endpoint must use the default HTTPS port")
    return parsed


def _validate_s3_credentials(access_key_id: str, secret_access_key: str, session_token: str) -> None:
    values = {
        "RELEASE_ARCHIVE_S3_ACCESS_KEY_ID": (access_key_id, 16),
        "RELEASE_ARCHIVE_S3_SECRET_ACCESS_KEY": (secret_access_key, 32),
    }
    for name, (value, minimum_length) in values.items():
        normalized = value.strip().lower()
        if not value or len(value) < minimum_length:
            raise HydrationError(f"{name} is missing or too short")
        if value != value.strip() or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise HydrationError(f"{name} contains whitespace or control characters")
        if any(marker in normalized for marker in ("change_me", "change-me", "placeholder", "example")):
            raise HydrationError(f"{name} is still an example placeholder")
    if session_token and (
        session_token != session_token.strip()
        or any(
            character.isspace() or ord(character) < 32 or ord(character) == 127
            for character in session_token
        )
    ):
        raise HydrationError("RELEASE_ARCHIVE_S3_SESSION_TOKEN contains whitespace or control characters")


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, *, allowed_hosts: set[str], allow_http: bool, deadline: float) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts
        self.allow_http = allow_http
        self.deadline = deadline

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target = urllib.parse.urljoin(req.full_url, newurl)
        _validate_url(target, allowed_hosts=self.allowed_hosts, allow_http=self.allow_http)
        req.timeout = min(float(getattr(req, "timeout", float("inf"))), _remaining_seconds(self.deadline))
        return super().redirect_request(req, fp, code, msg, headers, target)


def _download(
    *,
    url: str,
    destination: Path,
    expected_sha256: str,
    allowed_hosts: set[str],
    allow_http: bool,
    timeout_seconds: float,
    max_bytes: int,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    parsed = _validate_url(url, allowed_hosts=allowed_hosts, allow_http=allow_http)
    redacted = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
    print(f"Hydrating checksum-pinned release archive from {redacted}")
    opener = urllib.request.build_opener(
        _SafeRedirectHandler(
            allowed_hosts=allowed_hosts,
            allow_http=allow_http,
            deadline=deadline,
        ),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
    )
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/gzip, application/x-tar, application/octet-stream",
            "User-Agent": "mbzuai-retriever-release-hydrator/1.0",
        },
        method="GET",
    )
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with opener.open(request, timeout=_remaining_seconds(deadline)) as response, destination.open(
            "wb"
        ) as output:
            content_length = response.headers.get("Content-Length")
            if content_length:
                try:
                    declared = int(content_length)
                except ValueError as exc:
                    raise HydrationError("archive response has an invalid Content-Length") from exc
                if declared <= 0 or declared > max_bytes:
                    raise HydrationError(
                        f"archive Content-Length {declared} is outside the allowed range (1..{max_bytes})"
                    )
            while True:
                _set_stream_timeout(response, _remaining_seconds(deadline))
                block = response.read(1024 * 1024)
                _remaining_seconds(deadline)
                if not block:
                    break
                downloaded += len(block)
                if downloaded > max_bytes:
                    raise HydrationError(f"archive exceeds RELEASE_ARCHIVE_MAX_BYTES={max_bytes}")
                digest.update(block)
                output.write(block)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise HydrationError(f"release archive download failed: {type(exc).__name__}") from exc
    if downloaded <= 0:
        raise HydrationError("release archive download was empty")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise HydrationError(
            f"release archive SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    return downloaded


def _download_s3(
    *,
    uri: str,
    endpoint_url: str,
    region: str,
    destination: Path,
    expected_sha256: str,
    allowed_hosts: set[str],
    access_key_id: str,
    secret_access_key: str,
    session_token: str,
    timeout_seconds: float,
    max_bytes: int,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    bucket, key = _validate_s3_uri(uri)
    endpoint = _validate_s3_endpoint(endpoint_url, allowed_hosts=allowed_hosts)
    endpoint_hostname = str(endpoint.hostname or "").lower().rstrip(".")
    if "." in bucket:
        raise HydrationError("S3 bucket names containing dots are not supported with verified HTTPS virtual hosting")
    virtual_hostname = f"{bucket}.{endpoint_hostname}"
    if virtual_hostname not in allowed_hosts:
        raise HydrationError(
            "RELEASE_ARCHIVE_ALLOWED_HOSTS must contain the exact virtual-hosted S3 archive hostname"
        )
    if not region.strip() or any(character.isspace() for character in region):
        raise HydrationError("RELEASE_ARCHIVE_S3_REGION is required and must not contain whitespace")
    _validate_s3_credentials(access_key_id, secret_access_key, session_token)

    try:
        import boto3
        from botocore.config import Config
    except ImportError as exc:
        raise HydrationError("boto3 is required for authenticated release archive hydration") from exc

    print(f"Hydrating checksum-pinned release archive from {endpoint_hostname} object storage")
    try:
        _remaining_seconds(deadline)
        operation_timeout = max(0.001, timeout_seconds / 2.0)
        client = boto3.client(
            "s3",
            endpoint_url=f"https://{endpoint_hostname}",
            region_name=region.strip(),
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token or None,
            config=Config(
                connect_timeout=operation_timeout,
                read_timeout=operation_timeout,
                retries={"total_max_attempts": 1, "mode": "standard"},
                signature_version="s3v4",
                # DigitalOcean Spaces documents virtual-hosted addressing for
                # Boto3. The configured base endpoint remains HTTPS-only and
                # exact-host allowlisted before the client is constructed.
                s3={"addressing_style": "virtual"},
            ),
        )
        events = getattr(getattr(client, "meta", None), "events", None)
        register = getattr(events, "register", None)
        if not callable(register):
            raise HydrationError("S3 client cannot enforce the exact final request hostname")

        def enforce_final_request_host(request: Any, **_kwargs: Any) -> None:
            parsed_request = urllib.parse.urlparse(str(getattr(request, "url", "") or ""))
            request_hostname = str(parsed_request.hostname or "").lower().rstrip(".")
            if (
                parsed_request.scheme != "https"
                or request_hostname != virtual_hostname
                or parsed_request.username
                or parsed_request.password
                or parsed_request.port not in {None, 443}
            ):
                raise HydrationError("signed S3 archive request attempted to leave the exact allowlisted host")

        register("before-send.s3.GetObject", enforce_final_request_host)
        _remaining_seconds(deadline)
        response = client.get_object(Bucket=bucket, Key=key)
        _remaining_seconds(deadline)
        body = response.get("Body")
        if body is None or not callable(getattr(body, "read", None)):
            raise HydrationError("archive object response has no readable body")
        try:
            declared_raw = response.get("ContentLength")
            try:
                declared = int(declared_raw)
            except (TypeError, ValueError) as exc:
                raise HydrationError("archive object has an invalid ContentLength") from exc
            if declared <= 0 or declared > max_bytes:
                raise HydrationError(
                    f"archive object ContentLength {declared} is outside the allowed range (1..{max_bytes})"
                )

            digest = hashlib.sha256()
            downloaded = 0
            with destination.open("wb") as output:
                while downloaded < declared:
                    _set_stream_timeout(body, _remaining_seconds(deadline))
                    block = body.read(min(1024 * 1024, declared - downloaded))
                    _remaining_seconds(deadline)
                    if not block:
                        break
                    downloaded += len(block)
                    if downloaded > max_bytes:
                        raise HydrationError(f"archive exceeds RELEASE_ARCHIVE_MAX_BYTES={max_bytes}")
                    digest.update(block)
                    output.write(block)
            if downloaded != declared:
                raise HydrationError(
                    f"archive object length mismatch: expected={declared}, downloaded={downloaded}"
                )
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
    except HydrationError:
        raise
    except Exception as exc:
        # Keep credentials, signed request details, and object names out of logs.
        raise HydrationError(f"release archive object download failed: {type(exc).__name__}") from exc

    if downloaded <= 0:
        raise HydrationError("release archive download was empty")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise HydrationError(
            f"release archive SHA256 mismatch: expected={expected_sha256}, actual={actual_sha256}"
        )
    return downloaded


def _safe_member_path(name: str) -> PurePosixPath:
    if not name or any(ord(character) < 32 or ord(character) == 127 for character in name):
        raise HydrationError("archive contains an unsafe path")
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise HydrationError(f"archive contains an unsafe path: {name!r}")
    if path.parts[0] not in ALLOWED_ARCHIVE_ROOTS:
        raise HydrationError(f"archive contains an unexpected top-level path: {name!r}")
    if path.parts[0] == "runs" and (len(path.parts) < 2 or path.parts[1] != "mbzuai_main"):
        raise HydrationError(f"archive run path must be under runs/mbzuai_main: {name!r}")
    return path


def _discard_tar_bytes(stream: Any, size: int, *, expanded_limit: int) -> None:
    remaining = size
    consumed = 0
    while remaining:
        block = stream.read(min(1024 * 1024, remaining))
        if not block:
            raise HydrationError(
                "release archive could not be parsed safely: truncated member payload"
            )
        consumed += len(block)
        if consumed > expanded_limit:
            raise HydrationError("release archive raw expansion exceeds its configured limit")
        remaining -= len(block)


def _tar_header_size(field: bytes) -> int:
    # The canonical builder emits POSIX USTAR octal sizes. Reject base-256 and
    # extended numeric forms so no parser-specific large allocation happens.
    if field and field[0] & 0x80:
        raise HydrationError("archive uses a non-canonical tar size encoding")
    value = field.rstrip(b"\0 ").lstrip(b" ")
    if not value:
        return 0
    if any(character not in b"01234567" for character in value):
        raise HydrationError("archive contains an invalid tar member size")
    return int(value, 8)


def _prescan_tar_stream(
    *,
    archive_path: Path,
    max_files: int,
    max_extracted_bytes: int,
) -> tuple[int, int]:
    """Bound raw headers/payloads before Python's tar parser sees metadata."""

    member_count = 0
    total_payload_bytes = 0
    expanded_limit = max_extracted_bytes + (max_files + 2) * 512
    with archive_path.open("rb") as raw:
        magic = raw.read(2)
        raw.seek(0)
        stream: Any = gzip.GzipFile(fileobj=raw, mode="rb") if magic == b"\x1f\x8b" else raw
        try:
            zero_blocks = 0
            while True:
                header = stream.read(512)
                if not header:
                    raise HydrationError(
                        "release archive could not be parsed safely: missing tar end marker"
                    )
                if len(header) != 512:
                    raise HydrationError(
                        "release archive could not be parsed safely: truncated tar header"
                    )
                if header == b"\0" * 512:
                    zero_blocks += 1
                    if zero_blocks == 2:
                        break
                    continue
                if zero_blocks:
                    raise HydrationError("release archive has an invalid tar end marker")
                member_count += 1
                if member_count > max_files:
                    raise HydrationError(
                        f"archive member count exceeds the allowed range (1..{max_files})"
                    )
                type_flag = header[156:157]
                if type_flag in {b"1", b"2", b"3", b"4", b"6"}:
                    raise HydrationError("archive links and special files are forbidden")
                if type_flag in {b"g", b"x", b"L", b"K", b"X", b"S"}:
                    raise HydrationError("archive extended metadata records are forbidden")
                if type_flag not in {b"\0", b"0", b"5"}:
                    raise HydrationError("archive member type is forbidden")
                size = _tar_header_size(header[124:136])
                total_payload_bytes += size
                if total_payload_bytes > max_extracted_bytes:
                    raise HydrationError(
                        "archive expands beyond "
                        f"RELEASE_ARCHIVE_MAX_EXTRACTED_BYTES={max_extracted_bytes}"
                    )
                padded_size = ((size + 511) // 512) * 512
                if padded_size:
                    _discard_tar_bytes(
                        stream,
                        padded_size,
                        expanded_limit=expanded_limit,
                    )
        except (EOFError, OSError) as exc:
            raise HydrationError(
                f"release archive could not be pre-scanned safely: {type(exc).__name__}"
            ) from exc
        finally:
            if stream is not raw:
                stream.close()
    if member_count <= 0:
        raise HydrationError(f"archive member count 0 is outside the allowed range (1..{max_files})")
    return member_count, total_payload_bytes


def _extract_safely(
    *,
    archive_path: Path,
    destination: Path,
    max_files: int,
    max_extracted_bytes: int,
) -> tuple[int, int]:
    expected_member_count, expected_total_size = _prescan_tar_stream(
        archive_path=archive_path,
        max_files=max_files,
        max_extracted_bytes=max_extracted_bytes,
    )
    destination.mkdir(parents=True, exist_ok=False)
    destination_root = destination.resolve()
    member_count = 0
    total_size = 0
    try:
        # Stream members and extract only after each member passes policy. This
        # bounds archive metadata memory by the configured member limit.
        with tarfile.open(archive_path, mode="r|*") as archive:
            for member in archive:
                member_count += 1
                if member_count > max_files:
                    raise HydrationError(
                        f"archive member count exceeds the allowed range (1..{max_files})"
                    )
                member_path = _safe_member_path(member.name)
                if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                    raise HydrationError(
                        f"archive links and special files are forbidden: {member.name!r}"
                    )
                if not (member.isdir() or member.isfile()):
                    raise HydrationError(f"archive member type is forbidden: {member.name!r}")
                if member.mode & 0o6000:
                    raise HydrationError(
                        f"archive setuid/setgid bits are forbidden: {member.name!r}"
                    )
                if member.size < 0:
                    raise HydrationError(f"archive member has an invalid size: {member.name!r}")
                if member.isfile():
                    total_size += int(member.size)
                    if total_size > max_extracted_bytes:
                        raise HydrationError(
                            "archive expands beyond "
                            f"RELEASE_ARCHIVE_MAX_EXTRACTED_BYTES={max_extracted_bytes}"
                        )

                target = destination.joinpath(*member_path.parts)
                try:
                    target.resolve().relative_to(destination_root)
                except ValueError as exc:
                    raise HydrationError(
                        f"archive member escapes extraction root: {member.name!r}"
                    ) from exc
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True, mode=0o750)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
                source = archive.extractfile(member)
                if source is None:
                    raise HydrationError(
                        f"archive file has no readable content: {member.name!r}"
                    )
                with source, target.open("xb") as output:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                if target.stat().st_size != member.size:
                    raise HydrationError(
                        f"archive member length does not match its header: {member.name!r}"
                    )
                target.chmod(0o640)
    except HydrationError:
        raise
    except (tarfile.TarError, EOFError, OSError, UnicodeError) as exc:
        raise HydrationError(
            f"release archive could not be parsed safely: {type(exc).__name__}"
        ) from exc
    if member_count <= 0:
        raise HydrationError(f"archive member count 0 is outside the allowed range (1..{max_files})")
    if member_count != expected_member_count or total_size != expected_total_size:
        raise HydrationError("archive tar parser disagrees with the bounded raw-header scan")
    return member_count, total_size


def _load_marker(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _managed_marker(root: Path) -> dict[str, Any]:
    marker = _load_marker(root / MARKER_NAME)
    if (
        marker.get("schema_version") != 1
        or marker.get("mode") != "hydrated"
        or not SHA256_PATTERN.fullmatch(str(marker.get("archive_sha256") or ""))
        or marker.get("archive_source") not in {"https", "authenticated_s3"}
        or not RUN_ID_PATTERN.fullmatch(str(marker.get("run_id") or ""))
    ):
        raise HydrationError("release target is not a verified hydrator-managed directory")
    return marker


def _write_marker(root: Path, payload: dict[str, Any]) -> None:
    marker_path = root / MARKER_NAME
    marker_tmp = marker_path.with_name(marker_path.name + ".tmp")
    marker_tmp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    marker_tmp.chmod(0o640)
    os.replace(marker_tmp, marker_path)


def _validate_release_tree(root: Path, args: argparse.Namespace) -> dict[str, Any]:
    marker = _managed_marker(root)
    command = [
        sys.executable,
        str(VALIDATOR),
        "--active-release-file",
        str(root / "mbzuai_main" / "active_release.json"),
        "--runs-root",
        str(root / "runs" / "mbzuai_main"),
        "--storage-marker-file",
        str(root / MARKER_NAME),
        "--require-storage-marker",
        "--expected-config",
        args.expected_config,
        "--expected-model",
        args.expected_model,
        "--expected-dimension",
        str(args.expected_dimension),
        "--require-graph" if args.require_graph else "--no-require-graph",
    ]
    if args.allow_waived_release:
        command.append("--allow-waived-release")
    try:
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            timeout=args.validation_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise HydrationError("staged release validation exceeded its deadline") from exc
    if completed.returncode != 0:
        detail = (completed.stderr.strip().splitlines() or ["release validator rejected the artifacts"])[-1]
        detail = detail.replace(str(root), "<staged-release>")[:500]
        raise HydrationError(f"staged release validation failed: {detail}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise HydrationError("staged release validator returned invalid JSON") from exc
    if not isinstance(payload, dict) or payload.get("ok") is not True or payload.get("mode") != "active":
        raise HydrationError("staged release validator did not attest an active release")
    if str(payload.get("run_id") or "") != str(marker.get("run_id") or ""):
        raise HydrationError("hydration marker run_id does not match the validated active release")
    return payload


def _backup_path(target_root: Path) -> Path:
    return target_root.with_name(f".{target_root.name}.mbzuai-previous")


def _recover_interrupted_install(target_root: Path, args: argparse.Namespace) -> None:
    backup = _backup_path(target_root)
    if not backup.exists() and not backup.is_symlink():
        return
    if backup.is_symlink() or not backup.is_dir():
        raise HydrationError("release hydration backup path is not a safe directory")
    _managed_marker(backup)
    if not target_root.exists() and not target_root.is_symlink():
        _validate_release_tree(backup, args)
        os.replace(backup, target_root)
        return
    if target_root.is_symlink() or not target_root.is_dir():
        raise HydrationError("release hydration target path is not a safe directory")
    # A crash after the second rename leaves both the new target and the old
    # backup. Validate the new target before deleting the managed old copy.
    _managed_marker(target_root)
    _validate_release_tree(target_root, args)
    shutil.rmtree(backup)


def _activate_staged_target(
    *,
    staged_root: Path,
    target_root: Path,
    args: argparse.Namespace,
) -> None:
    backup = _backup_path(target_root)
    if backup.exists() or backup.is_symlink():
        raise HydrationError("release hydration backup must be recovered before activation")
    if not target_root.exists() and not target_root.is_symlink():
        os.replace(staged_root, target_root)
        return
    if target_root.is_symlink() or not target_root.is_dir():
        raise HydrationError("release hydration target path is not a safe directory")
    if not args.allow_replace:
        raise HydrationError(
            "release target already exists; set RELEASE_ARCHIVE_ALLOW_REPLACE=true for atomic managed upgrades"
        )
    _managed_marker(target_root)
    _validate_release_tree(target_root, args)
    os.replace(target_root, backup)
    try:
        os.replace(staged_root, target_root)
    except Exception as activation_error:
        try:
            os.replace(backup, target_root)
        except Exception as recovery_error:
            raise HydrationError(
                "atomic release activation failed and the previous release remains in the managed backup"
            ) from recovery_error
        raise HydrationError("atomic release activation failed; the previous release was restored") from activation_error
    try:
        shutil.rmtree(backup)
    except OSError:
        # The new target is already atomically active and validated. A later
        # startup will validate it again before removing this managed backup.
        print("Release activated; previous managed release will be cleaned up on the next startup", file=sys.stderr)


def hydrate(args: argparse.Namespace) -> dict[str, Any]:
    expected_sha256 = str(args.sha256 or "").strip().lower()
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise HydrationError("RELEASE_ARCHIVE_SHA256 must be exactly 64 lowercase hexadecimal characters")
    target_root = Path(os.path.abspath(Path(args.target_root).expanduser()))
    if target_root.is_symlink():
        raise HydrationError("release hydration target must not be a symlink")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    _recover_interrupted_install(target_root, args)
    marker = _load_marker(target_root / MARKER_NAME)
    if marker.get("mode") == "hydrated" and marker.get("archive_sha256") == expected_sha256:
        _managed_marker(target_root)
        _validate_release_tree(target_root, args)
        print(f"Verified release archive {expected_sha256} is already hydrated")
        return {"ok": True, "reused": True, "target_root": str(target_root), "sha256": expected_sha256}
    if target_root.exists():
        _managed_marker(target_root)
        if not args.allow_replace:
            raise HydrationError(
                "release target already contains a different hydrated release; enable atomic managed replacement"
            )
    with tempfile.TemporaryDirectory(prefix="mbzuai-release-download-") as download_tmp:
        archive_path = Path(download_tmp) / "release.tar.gz"
        allowed_hosts = _allowed_hosts(args.allowed_hosts)
        if args.s3_uri:
            downloaded = _download_s3(
                uri=args.s3_uri,
                endpoint_url=args.s3_endpoint_url,
                region=args.s3_region,
                destination=archive_path,
                expected_sha256=expected_sha256,
                allowed_hosts=allowed_hosts,
                access_key_id=os.getenv("RELEASE_ARCHIVE_S3_ACCESS_KEY_ID", ""),
                secret_access_key=os.getenv("RELEASE_ARCHIVE_S3_SECRET_ACCESS_KEY", ""),
                session_token=os.getenv("RELEASE_ARCHIVE_S3_SESSION_TOKEN", ""),
                timeout_seconds=args.timeout_seconds,
                max_bytes=args.max_bytes,
            )
            archive_source = "authenticated_s3"
        else:
            downloaded = _download(
                url=args.url,
                destination=archive_path,
                expected_sha256=expected_sha256,
                allowed_hosts=allowed_hosts,
                allow_http=args.allow_http,
                timeout_seconds=args.timeout_seconds,
                max_bytes=args.max_bytes,
            )
            archive_source = "https"
        extract_parent = Path(tempfile.mkdtemp(prefix=".mbzuai-release-extract-", dir=target_root.parent))
        extract_root = extract_parent / "payload"
        try:
            members, extracted_bytes = _extract_safely(
                archive_path=archive_path,
                destination=extract_root,
                max_files=args.max_files,
                max_extracted_bytes=args.max_extracted_bytes,
            )
            active_pointer = extract_root / "mbzuai_main" / "active_release.json"
            if not active_pointer.is_file():
                raise HydrationError("archive is missing mbzuai_main/active_release.json")
            pointer = json.loads(active_pointer.read_text(encoding="utf-8"))
            if not isinstance(pointer, dict) or not str(pointer.get("run_id") or "").strip():
                raise HydrationError("archive active release pointer is invalid")
            run_id = str(pointer["run_id"]).strip()
            manifest = extract_root / "runs" / "mbzuai_main" / run_id / "release" / "retrieval_release_manifest.json"
            if not manifest.is_file():
                raise HydrationError(f"archive is missing release manifest for run {run_id!r}")
            marker_payload = {
                "schema_version": 1,
                "mode": "hydrated",
                "archive_source": archive_source,
                "archive_sha256": expected_sha256,
                "downloaded_bytes": downloaded,
                "extracted_bytes": extracted_bytes,
                "member_count": members,
                "run_id": run_id,
            }
            _write_marker(extract_root, marker_payload)
            _validate_release_tree(extract_root, args)
            _activate_staged_target(staged_root=extract_root, target_root=target_root, args=args)
        finally:
            shutil.rmtree(extract_parent, ignore_errors=True)
    return {
        "ok": True,
        "reused": False,
        "target_root": str(target_root),
        "sha256": expected_sha256,
        "run_id": run_id,
        "downloaded_bytes": downloaded,
        "extracted_bytes": extracted_bytes,
        "member_count": members,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.getenv("RELEASE_ARCHIVE_URL", ""))
    parser.add_argument("--s3-uri", default=os.getenv("RELEASE_ARCHIVE_S3_URI", ""))
    parser.add_argument(
        "--s3-endpoint-url",
        default=os.getenv("RELEASE_ARCHIVE_S3_ENDPOINT_URL", ""),
    )
    parser.add_argument("--s3-region", default=os.getenv("RELEASE_ARCHIVE_S3_REGION", ""))
    parser.add_argument("--sha256", default=os.getenv("RELEASE_ARCHIVE_SHA256", ""))
    parser.add_argument(
        "--allowed-hosts",
        default=os.getenv("RELEASE_ARCHIVE_ALLOWED_HOSTS", ""),
        help="Comma-separated exact hostnames allowed for downloads and redirects.",
    )
    parser.add_argument(
        "--target-root",
        default=os.getenv("RELEASE_ARCHIVE_TARGET_ROOT", "/data/releases"),
    )
    parser.add_argument("--expected-config", default=os.getenv("PIPELINE_CONFIG", "mbzuai_production"))
    parser.add_argument(
        "--expected-model",
        default=os.getenv("RETRIEVER_EXPECTED_EMBEDDING_MODEL", "gemini-embedding-2"),
    )
    parser.add_argument(
        "--expected-dimension",
        type=int,
        default=int(os.getenv("RETRIEVER_EXPECTED_EMBEDDING_DIMENSIONALITY", "1536")),
    )
    parser.add_argument(
        "--require-graph",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RETRIEVER_REQUIRE_GRAPH"), default=True),
    )
    parser.add_argument(
        "--allow-waived-release",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RETRIEVER_ALLOW_WAIVED_RELEASE"), default=False),
    )
    parser.add_argument(
        "--validation-timeout-seconds",
        type=float,
        default=float(os.getenv("RELEASE_ARCHIVE_VALIDATION_TIMEOUT_SECONDS", "600")),
    )
    parser.add_argument(
        "--allow-http",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RELEASE_ARCHIVE_ALLOW_HTTP"), default=False),
    )
    parser.add_argument(
        "--allow-replace",
        action=argparse.BooleanOptionalAction,
        default=_bool(os.getenv("RELEASE_ARCHIVE_ALLOW_REPLACE"), default=False),
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=float(os.getenv("RELEASE_ARCHIVE_TIMEOUT_SECONDS", "300")),
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=int(os.getenv("RELEASE_ARCHIVE_MAX_BYTES", str(2 * 1024**3))),
    )
    parser.add_argument(
        "--max-extracted-bytes",
        type=int,
        default=int(os.getenv("RELEASE_ARCHIVE_MAX_EXTRACTED_BYTES", str(4 * 1024**3))),
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=int(os.getenv("RELEASE_ARCHIVE_MAX_FILES", "64")),
    )
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        configured_sources = int(bool(args.url)) + int(bool(args.s3_uri))
        if configured_sources != 1:
            raise HydrationError(
                "configure exactly one release archive source: RELEASE_ARCHIVE_S3_URI or RELEASE_ARCHIVE_URL"
            )
        if args.s3_uri and (not args.s3_endpoint_url or not args.s3_region):
            raise HydrationError(
                "RELEASE_ARCHIVE_S3_ENDPOINT_URL and RELEASE_ARCHIVE_S3_REGION are required for S3 hydration"
            )
        if not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 3600:
            raise HydrationError("archive download timeout must be finite, positive, and at most 3600 seconds")
        if not math.isfinite(args.validation_timeout_seconds) or not (
            0 < args.validation_timeout_seconds <= 3600
        ):
            raise HydrationError("archive validation timeout must be finite, positive, and at most 3600 seconds")
        if not 0 < args.max_bytes <= 8 * 1024**3:
            raise HydrationError("archive byte limit must be positive and at most 8 GiB")
        if not 0 < args.max_extracted_bytes <= 8 * 1024**3:
            raise HydrationError("archive extracted-byte limit must be positive and at most 8 GiB")
        if not 0 < args.max_files <= 10_000:
            raise HydrationError("archive member limit must be between 1 and 10000")
        if not 0 < args.expected_dimension <= 65_536:
            raise HydrationError("expected embedding dimension is outside the safe range")
        payload = hydrate(args)
    except (HydrationError, json.JSONDecodeError) as exc:
        print(f"Release archive hydration failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        # Never print provider exceptions, signed request data, or malformed
        # archive internals. The exception type is enough for operator triage.
        print(
            f"Release archive hydration failed unexpectedly: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
