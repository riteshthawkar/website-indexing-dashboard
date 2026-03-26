"""
Local retriever service lifecycle helpers for the dashboard.

This is intentionally a local-ops control plane. It starts and stops the
retriever service as an external process, then polls its health endpoints.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _service_dir(work_dir: str | Path) -> Path:
    target = Path(work_dir).resolve() / "dashboard_services"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _manifest_path(work_dir: str | Path) -> Path:
    return _service_dir(work_dir) / "retriever_service.manifest.json"


def _log_path(work_dir: str | Path) -> Path:
    return _service_dir(work_dir) / "retriever_service.log"


def _read_manifest(work_dir: str | Path) -> Dict[str, Any] | None:
    path = _manifest_path(work_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_manifest(work_dir: str | Path, payload: Dict[str, Any]) -> None:
    _manifest_path(work_dir).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _health_get(url: str) -> Dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=2.0) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
        return payload if isinstance(payload, dict) else None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return None


def get_retriever_service_status(work_dir: str | Path) -> Dict[str, Any]:
    manifest = _read_manifest(work_dir) or {}
    pid = int(manifest.get("pid") or 0) if str(manifest.get("pid") or "").strip() else 0
    process_alive = _process_alive(pid)
    host = str(manifest.get("host") or "127.0.0.1")
    port = int(manifest.get("port") or 0) if str(manifest.get("port") or "").strip() else 0
    base_url = f"http://{host}:{port}" if port else None
    health = _health_get(f"{base_url}/healthz") if (process_alive and base_url) else None
    ready = _health_get(f"{base_url}/readyz") if (process_alive and base_url) else None
    return {
        **manifest,
        "running": bool(process_alive),
        "process_alive": bool(process_alive),
        "base_url": base_url,
        "health": health,
        "ready": ready,
        "manifest_path": str(_manifest_path(work_dir)),
        "log_path": str(_log_path(work_dir)),
        "checked_at": _utcnow_iso(),
    }


def start_retriever_service(
    *,
    config_name: str,
    work_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8060,
    max_concurrency: int = 4,
    request_timeout_seconds: float = 90.0,
) -> Dict[str, Any]:
    resolved_work_dir = Path(work_dir).resolve()
    existing = get_retriever_service_status(resolved_work_dir)
    if existing.get("running"):
        raise ValueError(f"Retriever service already running on port {existing.get('port')}")

    log_path = _log_path(resolved_work_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = open(log_path, "a", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "pipeline",
        "serve-retriever",
        "--config",
        str(config_name),
        "--work-dir",
        str(resolved_work_dir),
        "--host",
        str(host),
        "--port",
        str(int(port)),
        "--max-concurrency",
        str(int(max_concurrency)),
        "--request-timeout-seconds",
        str(float(request_timeout_seconds)),
    ]
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    payload = {
        "service": "retriever",
        "status": "starting",
        "pid": process.pid,
        "host": host,
        "port": int(port),
        "config_name": config_name,
        "work_dir": str(resolved_work_dir),
        "max_concurrency": int(max_concurrency),
        "request_timeout_seconds": float(request_timeout_seconds),
        "command": command,
        "started_at": _utcnow_iso(),
    }
    _write_manifest(resolved_work_dir, payload)
    time.sleep(1.0)
    return get_retriever_service_status(resolved_work_dir)


def stop_retriever_service(work_dir: str | Path) -> Dict[str, Any]:
    resolved_work_dir = Path(work_dir).resolve()
    manifest = _read_manifest(resolved_work_dir)
    if not manifest:
        return {
            "service": "retriever",
            "status": "not_running",
            "running": False,
            "manifest_path": str(_manifest_path(resolved_work_dir)),
            "log_path": str(_log_path(resolved_work_dir)),
        }
    pid = int(manifest.get("pid") or 0) if str(manifest.get("pid") or "").strip() else 0
    if _process_alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    deadline = time.time() + 10.0
    while _process_alive(pid) and time.time() < deadline:
        time.sleep(0.25)

    final_status = "stopped" if not _process_alive(pid) else "stop_timeout"
    manifest["status"] = final_status
    manifest["stopped_at"] = _utcnow_iso()
    _write_manifest(resolved_work_dir, manifest)
    status = get_retriever_service_status(resolved_work_dir)
    status["status"] = final_status
    return status
