from __future__ import annotations

import copy
from pathlib import Path
import sys

from fastapi.testclient import TestClient


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import app
from database import Run, get_db
from config_manager import load_config


def test_public_config_endpoints_only_expose_default() -> None:
    with TestClient(app) as client:
        response = client.get("/api/configs")
        assert response.status_code == 200
        items = response.json()
        assert items
        assert {item["name"] for item in items} == {"default"}

        response = client.get("/api/pipeline-configs")
        assert response.status_code == 200
        items = response.json()
        assert items
        assert {item["name"] for item in items} == {"default"}


def test_create_run_persists_launch_time_config_snapshot() -> None:
    base = load_config("default")
    assert base is not None
    snapshot = copy.deepcopy(base)
    snapshot.setdefault("crawler", {})
    snapshot["crawler"]["start_url"] = "https://example.org/custom"
    snapshot["crawler"]["max_pages"] = 17

    with TestClient(app) as client:
        response = client.post(
            "/api/runs",
            json={
                "run_name": "pytest-single-config-run",
                "config_name": "default",
                "config_snapshot": snapshot,
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        run_id = int(payload["id"])
        try:
            detail = client.get(f"/api/runs/{run_id}")
            assert detail.status_code == 200, detail.text
            run_payload = detail.json()
            assert run_payload["config_name"] == "default"
            assert run_payload["start_url"] == "https://example.org/custom"
            assert run_payload["config_snapshot"]["crawler"]["max_pages"] == 17

            db = get_db()
            try:
                run = db.get(Run, run_id)
                assert run is not None
                assert run.start_url == "https://example.org/custom"
                assert run.config_snapshot_json is not None
            finally:
                db.close()
        finally:
            client.delete(f"/api/runs/{run_id}")
