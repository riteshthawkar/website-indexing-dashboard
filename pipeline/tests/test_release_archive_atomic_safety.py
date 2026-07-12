from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import types
from pathlib import Path

import pytest

from pipeline.tests.test_release_archive_deploy import (
    BUILDER,
    PROJECT_ROOT,
    _hydrate_from_local_archive,
    _load_hydrator_module,
    _make_active_release,
)


def _run_builder(active_path: Path, runs_root: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--active-release-file",
            str(active_path),
            "--runs-root",
            str(runs_root),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_managed_marker(root: Path, *, digest: str = "a" * 64) -> None:
    root.mkdir(parents=True)
    (root / ".mbzuai-release-storage").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "mode": "hydrated",
                "archive_source": "https",
                "archive_sha256": digest,
                "run_id": "run-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_archive_builder_rejects_output_and_checksum_aliases(tmp_path: Path) -> None:
    active_path, runs_root = _make_active_release(tmp_path / "source")
    pointer_before = active_path.read_bytes()

    output_alias = _run_builder(active_path, runs_root, active_path)
    assert output_alias.returncode != 0
    assert "aliases a validated release source" in output_alias.stderr
    assert active_path.read_bytes() == pointer_before

    output = tmp_path / "release.tar.gz"
    checksum = output.with_name(output.name + ".sha256")
    os.link(active_path, checksum)
    checksum_alias = _run_builder(active_path, runs_root, output)
    assert checksum_alias.returncode != 0
    assert "checksum output aliases" in checksum_alias.stderr
    assert active_path.read_bytes() == pointer_before
    assert not output.exists()


def test_malformed_archive_error_is_sanitized(tmp_path: Path) -> None:
    archive = tmp_path / "malformed.tar.gz"
    archive.write_bytes(b"not-a-tar-and-must-not-appear-in-logs")
    result = _hydrate_from_local_archive(archive, target=tmp_path / "target")

    assert result.returncode != 0
    assert "could not be parsed safely" in result.stderr
    assert "Traceback" not in result.stderr
    assert "must-not-appear-in-logs" not in result.stderr
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("timeout", ["inf", "nan", "0", "3601"])
def test_hydrator_rejects_unbounded_download_timeouts(tmp_path: Path, timeout: str) -> None:
    archive = tmp_path / "unused.tar.gz"
    archive.write_bytes(b"unused")

    result = _hydrate_from_local_archive(
        archive,
        target=tmp_path / "target",
        extra_args=["--timeout-seconds", timeout],
    )

    assert result.returncode != 0
    assert "download timeout must be finite" in result.stderr


def test_archive_rejects_extended_metadata_before_tar_parser_allocation(tmp_path: Path) -> None:
    module = _load_hydrator_module()
    archive = tmp_path / "pax-metadata.tar.gz"
    content = b"{}"
    with tarfile.open(archive, mode="w:gz", format=tarfile.PAX_FORMAT) as handle:
        member = tarfile.TarInfo("mbzuai_main/active_release.json")
        member.size = len(content)
        member.pax_headers = {"comment": "x" * 4096}
        handle.addfile(member, io.BytesIO(content))

    with pytest.raises(module.HydrationError, match="extended metadata records are forbidden"):
        module._extract_safely(
            archive_path=archive,
            destination=tmp_path / "extract",
            max_files=64,
            max_extracted_bytes=1024 * 1024,
        )

    assert not (tmp_path / "extract").exists()


def test_http_download_uses_monotonic_end_to_end_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hydrator_module()
    clock = [0.0]

    class FakeResponse:
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _size: int) -> bytes:
            clock[0] += 0.6
            return b"late"

    class FakeOpener:
        def open(self, _request: object, *, timeout: float) -> FakeResponse:
            assert 0 < timeout <= 0.5
            return FakeResponse()

    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.urllib.request, "build_opener", lambda *_args: FakeOpener())

    with pytest.raises(module.HydrationError, match="end-to-end deadline"):
        module._download(
            url="https://archives.example.invalid/release.tar.gz",
            destination=tmp_path / "release.tar.gz",
            expected_sha256=hashlib.sha256(b"late").hexdigest(),
            allowed_hosts={"archives.example.invalid"},
            allow_http=False,
            timeout_seconds=0.5,
            max_bytes=1024,
        )


def test_s3_requires_final_virtual_host_and_enforces_stream_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hydrator_module()
    common = {
        "uri": "s3://mbzuai-releases/production/release.tar.gz",
        "endpoint_url": "https://nyc3.digitaloceanspaces.com",
        "region": "nyc3",
        "destination": tmp_path / "release.tar.gz",
        "expected_sha256": hashlib.sha256(b"late").hexdigest(),
        "access_key_id": "A1B2C3D4E5F6G7H8I9J0",
        "secret_access_key": "correct-horse-battery-staple-archive-key-2026",
        "session_token": "",
        "timeout_seconds": 0.5,
        "max_bytes": 1024,
    }
    with pytest.raises(module.HydrationError, match="exact virtual-hosted"):
        module._download_s3(
            **common,
            allowed_hosts={"nyc3.digitaloceanspaces.com"},
        )

    clock = [0.0]

    class FakeBody:
        def set_socket_timeout(self, timeout: float) -> None:
            assert 0 < timeout <= 0.5

        def read(self, _size: int) -> bytes:
            clock[0] += 0.6
            return b"late"

        def close(self) -> None:
            return None

    class FakeEvents:
        def register(self, _event: str, _handler: object) -> None:
            return None

    class FakeClient:
        meta = types.SimpleNamespace(events=FakeEvents())

        def get_object(self, **_kwargs: object) -> dict[str, object]:
            return {"ContentLength": 4, "Body": FakeBody()}

    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *_args, **_kwargs: FakeClient()  # type: ignore[attr-defined]
    botocore = types.ModuleType("botocore")
    botocore_config = types.ModuleType("botocore.config")
    botocore_config.Config = lambda **kwargs: types.SimpleNamespace(**kwargs)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    with pytest.raises(module.HydrationError, match="end-to-end deadline"):
        module._download_s3(
            **common,
            allowed_hosts={
                "nyc3.digitaloceanspaces.com",
                "mbzuai-releases.nyc3.digitaloceanspaces.com",
            },
        )


def test_atomic_activation_restores_previous_target_on_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hydrator_module()
    target = tmp_path / "current"
    staged = tmp_path / "staged"
    _write_managed_marker(target, digest="a" * 64)
    _write_managed_marker(staged, digest="b" * 64)
    (target / "identity.txt").write_text("old", encoding="utf-8")
    (staged / "identity.txt").write_text("new", encoding="utf-8")
    monkeypatch.setattr(module, "_validate_release_tree", lambda *_args, **_kwargs: {"ok": True})
    real_replace = module.os.replace

    def fail_new_activation(source: Path, destination: Path) -> None:
        if Path(source) == staged and Path(destination) == target:
            raise OSError("injected activation failure")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", fail_new_activation)
    args = types.SimpleNamespace(allow_replace=True)

    with pytest.raises(module.HydrationError, match="previous release was restored"):
        module._activate_staged_target(staged_root=staged, target_root=target, args=args)

    assert (target / "identity.txt").read_text(encoding="utf-8") == "old"
    assert (staged / "identity.txt").read_text(encoding="utf-8") == "new"
    assert not module._backup_path(target).exists()


def test_interrupted_install_recovers_managed_backup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hydrator_module()
    target = tmp_path / "current"
    backup = module._backup_path(target)
    _write_managed_marker(backup)
    (backup / "identity.txt").write_text("old", encoding="utf-8")
    monkeypatch.setattr(module, "_validate_release_tree", lambda *_args, **_kwargs: {"ok": True})

    module._recover_interrupted_install(target, types.SimpleNamespace())

    assert (target / "identity.txt").read_text(encoding="utf-8") == "old"
    assert not backup.exists()
