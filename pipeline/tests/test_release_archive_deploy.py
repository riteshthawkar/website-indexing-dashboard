from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import threading
import types
from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pipeline.tests.test_deploy_startup_safety import (
    CONTRACT_FINGERPRINT,
    ANSWER_RUNTIME_COMMIT_SHA,
    INDEXING_BUILD,
    SERVING_CONTRACT_FINGERPRINT,
    _make_required_artifacts,
    _manifest_payload,
    _write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_DIR = PROJECT_ROOT / "scripts" / "deploy"
BUILDER = DEPLOY_DIR / "build-runtime-release-archive.py"
HYDRATOR = DEPLOY_DIR / "hydrate-release-archive.py"
VALIDATOR = DEPLOY_DIR / "validate-release-artifacts.py"


def _load_hydrator_module():
    spec = importlib.util.spec_from_file_location("mbzuai_release_hydrator", HYDRATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeStreamingBody:
    def __init__(self, content: bytes) -> None:
        self._content = io.BytesIO(content)
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        return self._content.read(size)

    def close(self) -> None:
        self.closed = True


def _install_fake_s3(monkeypatch: pytest.MonkeyPatch, *, content: bytes, declared: int | None = None):
    captured: dict[str, object] = {}
    body = _FakeStreamingBody(content)

    class FakeEvents:
        def register(self, event: str, handler: object) -> None:
            captured["registered_event"] = event
            captured["before_send_handler"] = handler

    class FakeClient:
        meta = types.SimpleNamespace(events=FakeEvents())

        def get_object(self, **kwargs: object) -> dict[str, object]:
            captured["get_object"] = kwargs
            captured["before_send_handler"](
                types.SimpleNamespace(
                    url="https://mbzuai-releases.nyc3.digitaloceanspaces.com/production/release-2026.tar.gz"
                )
            )
            return {"ContentLength": len(content) if declared is None else declared, "Body": body}

    boto3 = types.ModuleType("boto3")

    def client(service_name: str, **kwargs: object) -> FakeClient:
        captured["service_name"] = service_name
        captured["client"] = kwargs
        return FakeClient()

    boto3.client = client  # type: ignore[attr-defined]
    botocore = types.ModuleType("botocore")
    botocore_config = types.ModuleType("botocore.config")

    def config(**kwargs: object) -> types.SimpleNamespace:
        captured["config"] = kwargs
        return types.SimpleNamespace(**kwargs)

    botocore_config.Config = config  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", boto3)
    monkeypatch.setitem(sys.modules, "botocore", botocore)
    monkeypatch.setitem(sys.modules, "botocore.config", botocore_config)
    return captured, body


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


def _hydrate_from_local_archive(
    archive_path: Path,
    *,
    target: Path,
    sha256: str | None = None,
    allowed_hosts: str = "127.0.0.1",
    extra_args: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args,
        directory=str(archive_path.parent),
        **kwargs,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        return subprocess.run(
            [
                sys.executable,
                str(HYDRATOR),
                "--url",
                f"http://127.0.0.1:{server.server_port}/{archive_path.name}",
                "--sha256",
                sha256 or hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "--allowed-hosts",
                allowed_hosts,
                "--target-root",
                str(target),
                "--allow-http",
                *(extra_args or []),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _make_active_release(root: Path) -> tuple[Path, Path]:
    runs_root = root / "runs" / "mbzuai_main"
    work_dir = runs_root / "run-1"
    active_path = root / "mbzuai_main" / "active_release.json"
    manifest_path = work_dir / "release" / "retrieval_release_manifest.json"
    _make_required_artifacts(work_dir)
    _write_json(manifest_path, _manifest_payload(status="passed", work_dir=str(work_dir)))
    _write_json(
        active_path,
        {
            "schema_version": 1,
            "status": "passed",
            "run_id": "run-1",
            "release_id": "release-1",
            "production_indexing_contract_fingerprint": CONTRACT_FINGERPRINT,
            "production_serving_contract_fingerprint": SERVING_CONTRACT_FINGERPRINT,
            "answer_runtime_commit_sha": ANSWER_RUNTIME_COMMIT_SHA,
            "indexing_build_commit_sha": INDEXING_BUILD["commit_sha"],
            "active_release_manifest": str(manifest_path),
        },
    )
    return active_path, runs_root


def test_runtime_archive_is_deterministic_and_hydrates_safely(tmp_path: Path) -> None:
    source = tmp_path / "source"
    active_path, runs_root = _make_active_release(source)
    output_one = tmp_path / "release-one.tar.gz"
    output_two = tmp_path / "release-two.tar.gz"
    for output in (output_one, output_two):
        result = subprocess.run(
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
        assert result.returncode == 0, result.stderr

    digest = hashlib.sha256(output_one.read_bytes()).hexdigest()
    assert hashlib.sha256(output_two.read_bytes()).hexdigest() == digest
    with tarfile.open(output_one, mode="r:gz") as archive:
        names = set(archive.getnames())
    assert "mbzuai_main/active_release.json" in names
    assert "runs/mbzuai_main/run-1/release/retrieval_release_manifest.json" in names
    assert "runs/mbzuai_main/run-1/resolved_config.json" in names
    assert "runs/mbzuai_main/run-1/stage_outputs/finalize_retrieval_bundle/lexical_corpus.json" in names
    assert "runs/mbzuai_main/run-1/stage_outputs/promote_assertions/promoted_assertions.json" in names
    assert len(names) == 9

    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args,
        directory=str(tmp_path),
        **kwargs,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    target = tmp_path / "hydrated"
    try:
        hydrate = subprocess.run(
            [
                sys.executable,
                str(HYDRATOR),
                "--url",
                f"http://127.0.0.1:{server.server_port}/{output_one.name}",
                "--sha256",
                digest,
                "--allowed-hosts",
                "127.0.0.1",
                "--target-root",
                str(target),
                "--allow-http",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert hydrate.returncode == 0, hydrate.stderr
    marker = json.loads((target / ".mbzuai-release-storage").read_text(encoding="utf-8"))
    assert marker["mode"] == "hydrated"
    assert marker["archive_sha256"] == digest

    validate = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--active-release-file",
            str(target / "mbzuai_main" / "active_release.json"),
            "--runs-root",
            str(target / "runs" / "mbzuai_main"),
            "--storage-marker-file",
            str(target / ".mbzuai-release-storage"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert validate.returncode == 0, validate.stderr

    hydrated_work_dir = target / "runs" / "mbzuai_main" / "run-1"
    runtime_config = {
        "pipeline": {"production_profile": False},
        "embedder": {
            "pinecone_index": "dense-v3",
            "pinecone_sparse_index": "sparse-v3",
            "model": "gemini-embedding-2",
            "output_dimensionality": 1536,
        },
        "retrieval": {
            "enable_sparse": False,
            "enable_rerank": False,
            "routed_graph_enabled": False,
        },
    }
    from pipeline.retrieval.adaptive_hybrid import AdaptiveHybridRetriever
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    adaptive = AdaptiveHybridRetriever(config=runtime_config, work_dir=hydrated_work_dir)
    assert adaptive.lexical_records
    assert adaptive.lexical_map["chunk-1"]["record_type"] == "chunk"
    assert adaptive._namespace_token_index["chunks"]["admission"] == ["chunk-1"]
    routed = RoutedHybridRetriever(config=runtime_config, work_dir=hydrated_work_dir)
    try:
        assert routed.vector.lexical_records
        assert "chunk-1" in routed.vector.lexical_map
    finally:
        routed.close()


def test_hydrator_rejects_path_traversal_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "malicious.tar.gz"
    content = b"not allowed"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo("../escape.txt")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    handler = lambda *args, **kwargs: _QuietHandler(  # noqa: E731
        *args,
        directory=str(tmp_path),
        **kwargs,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(HYDRATOR),
                "--url",
                f"http://127.0.0.1:{server.server_port}/{archive_path.name}",
                "--sha256",
                digest,
                "--allowed-hosts",
                "127.0.0.1",
                "--target-root",
                str(tmp_path / "target"),
                "--allow-http",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode != 0
    assert "unsafe path" in result.stderr
    assert not (tmp_path / "escape.txt").exists()


def test_hydrator_rejects_checksum_mismatch(tmp_path: Path) -> None:
    archive_path = tmp_path / "release.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        content = b'{"run_id":"run-1"}'
        info = tarfile.TarInfo("mbzuai_main/active_release.json")
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    result = _hydrate_from_local_archive(
        archive_path,
        target=tmp_path / "target",
        sha256="0" * 64,
    )

    assert result.returncode != 0
    assert "SHA256 mismatch" in result.stderr


def test_hydrator_rejects_non_allowlisted_host(tmp_path: Path) -> None:
    archive_path = tmp_path / "release.tar.gz"
    archive_path.write_bytes(b"not reached")

    result = _hydrate_from_local_archive(
        archive_path,
        target=tmp_path / "target",
        allowed_hosts="spaces.example.invalid",
    )

    assert result.returncode != 0
    assert "hostname is not allowlisted" in result.stderr


def test_authenticated_s3_hydration_streams_checksum_pinned_object(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_hydrator_module()
    content = b"checksum-pinned private release archive"
    captured, body = _install_fake_s3(monkeypatch, content=content)
    destination = tmp_path / "release.tar.gz"
    access_key = "A1B2C3D4E5F6G7H8I9J0"
    secret_key = "correct-horse-battery-staple-archive-key-2026"

    downloaded = module._download_s3(
        uri="s3://mbzuai-releases/production/release-2026.tar.gz",
        endpoint_url="https://nyc3.digitaloceanspaces.com",
        region="nyc3",
        destination=destination,
        expected_sha256=hashlib.sha256(content).hexdigest(),
        allowed_hosts={
            "nyc3.digitaloceanspaces.com",
            "mbzuai-releases.nyc3.digitaloceanspaces.com",
        },
        access_key_id=access_key,
        secret_access_key=secret_key,
        session_token="",
        timeout_seconds=30,
        max_bytes=1024,
    )

    assert downloaded == len(content)
    assert destination.read_bytes() == content
    assert body.closed is True
    assert captured["service_name"] == "s3"
    assert captured["get_object"] == {
        "Bucket": "mbzuai-releases",
        "Key": "production/release-2026.tar.gz",
    }
    assert captured["registered_event"] == "before-send.s3.GetObject"
    with pytest.raises(module.HydrationError, match="leave the exact allowlisted host"):
        captured["before_send_handler"](
            types.SimpleNamespace(url="https://redirected.example.invalid/release.tar.gz")
        )
    client_options = captured["client"]
    assert isinstance(client_options, dict)
    assert client_options["endpoint_url"] == "https://nyc3.digitaloceanspaces.com"
    assert captured["config"] == {
        "connect_timeout": 15,
        "read_timeout": 15,
        "retries": {"total_max_attempts": 1, "mode": "standard"},
        "signature_version": "s3v4",
        "s3": {"addressing_style": "virtual"},
    }
    output = capsys.readouterr()
    assert access_key not in output.out + output.err
    assert secret_key not in output.out + output.err


def test_authenticated_s3_hydration_closes_oversized_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_hydrator_module()
    _, body = _install_fake_s3(monkeypatch, content=b"small", declared=2048)

    with pytest.raises(module.HydrationError, match="ContentLength"):
        module._download_s3(
            uri="s3://mbzuai-releases/production/release.tar.gz",
            endpoint_url="https://nyc3.digitaloceanspaces.com",
            region="nyc3",
            destination=tmp_path / "release.tar.gz",
            expected_sha256="0" * 64,
            allowed_hosts={
                "nyc3.digitaloceanspaces.com",
                "mbzuai-releases.nyc3.digitaloceanspaces.com",
            },
            access_key_id="A1B2C3D4E5F6G7H8I9J0",
            secret_access_key="correct-horse-battery-staple-archive-key-2026",
            session_token="",
            timeout_seconds=30,
            max_bytes=1024,
        )

    assert body.closed is True


def test_hydrator_requires_exactly_one_archive_source(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(HYDRATOR),
            "--url",
            "https://archives.example.invalid/release.tar.gz",
            "--s3-uri",
            "s3://mbzuai-releases/production/release.tar.gz",
            "--sha256",
            "0" * 64,
            "--target-root",
            str(tmp_path / "target"),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "exactly one release archive source" in result.stderr


def test_hydrator_revalidates_redirect_hostname(tmp_path: Path) -> None:
    class RedirectHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{self.server.server_port}/redirected.tar.gz")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = subprocess.run(
            [
                sys.executable,
                str(HYDRATOR),
                "--url",
                f"http://127.0.0.1:{server.server_port}/release.tar.gz",
                "--sha256",
                "0" * 64,
                "--allowed-hosts",
                "127.0.0.1",
                "--target-root",
                str(tmp_path / "target"),
                "--allow-http",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert result.returncode != 0
    assert "hostname is not allowlisted" in result.stderr


@pytest.mark.parametrize("unsafe_name", ["../escape.txt", "/absolute/escape.txt"])
def test_hydrator_rejects_relative_and_absolute_escape_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    content = b"forbidden"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo(unsafe_name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    result = _hydrate_from_local_archive(archive_path, target=tmp_path / "target")

    assert result.returncode != 0
    assert "unsafe path" in result.stderr


def test_hydrator_rejects_symlinks(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        info = tarfile.TarInfo("mbzuai_main/active_release.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        archive.addfile(info)

    result = _hydrate_from_local_archive(archive_path, target=tmp_path / "target")

    assert result.returncode != 0
    assert "links and special files are forbidden" in result.stderr


@pytest.mark.parametrize(
    ("limit_args", "expected_error"),
    [
        (["--max-files", "1"], "archive member count"),
        (["--max-extracted-bytes", "1"], "archive expands beyond"),
        (["--max-bytes", "1"], "Content-Length"),
    ],
)
def test_hydrator_enforces_archive_limits(
    tmp_path: Path,
    limit_args: list[str],
    expected_error: str,
) -> None:
    archive_path = tmp_path / "limits.tar.gz"
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for name, content in (
            ("mbzuai_main/active_release.json", b'{"run_id":"run-1"}'),
            ("runs/mbzuai_main/run-1/release/retrieval_release_manifest.json", b"{}"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    result = _hydrate_from_local_archive(
        archive_path,
        target=tmp_path / "target",
        extra_args=limit_args,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_hydrated_release_without_resolved_config_fails_runtime_validation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    active_path, runs_root = _make_active_release(source)
    complete = tmp_path / "complete.tar.gz"
    build = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--active-release-file",
            str(active_path),
            "--runs-root",
            str(runs_root),
            "--output",
            str(complete),
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    incomplete = tmp_path / "missing-config.tar.gz"
    with tarfile.open(complete, mode="r:gz") as source_archive, tarfile.open(
        incomplete,
        mode="w:gz",
    ) as target_archive:
        for member in source_archive.getmembers():
            if member.name.endswith("/resolved_config.json"):
                continue
            extracted = source_archive.extractfile(member) if member.isfile() else None
            target_archive.addfile(member, extracted)

    target = tmp_path / "hydrated"
    initial_hydrate = _hydrate_from_local_archive(complete, target=target)
    assert initial_hydrate.returncode == 0, initial_hydrate.stderr
    pointer_before = (target / "mbzuai_main" / "active_release.json").read_bytes()
    marker_before = (target / ".mbzuai-release-storage").read_bytes()

    hydrate = _hydrate_from_local_archive(
        incomplete,
        target=target,
        extra_args=["--allow-replace"],
    )
    assert hydrate.returncode != 0
    assert "missing required resolved production config" in hydrate.stderr
    assert (target / "mbzuai_main" / "active_release.json").read_bytes() == pointer_before
    assert (target / ".mbzuai-release-storage").read_bytes() == marker_before
