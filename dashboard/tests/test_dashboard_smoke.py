from __future__ import annotations

import time
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient

DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def existing_run(client: TestClient) -> dict:
    response = client.get("/api/runs")
    assert response.status_code == 200
    runs = response.json()
    if not runs:
        pytest.skip("No dashboard runs are available in the local database.")
    run = next((item for item in runs if item.get("work_dir")), None)
    if not run:
        pytest.skip("No imported or executed run has a work directory yet.")
    return run


def test_dashboard_global_routes(client: TestClient) -> None:
    for path in (
        "/api/runs",
        "/api/configs",
        "/api/configs/schema",
        "/api/pipeline-configs",
        "/api/evaluation/assets",
        "/api/evaluation/presets",
        "/api/stages",
    ):
        response = client.get(path)
        assert response.status_code == 200, f"{path} failed: {response.text}"


def test_dashboard_run_detail_routes(client: TestClient, existing_run: dict) -> None:
    run_id = existing_run["id"]

    for path in (
        f"/api/runs/{run_id}",
        f"/api/runs/{run_id}/stages",
        f"/api/runs/{run_id}/logs",
        f"/api/runs/{run_id}/structured-logs",
        f"/api/runs/{run_id}/evaluation/assets",
        f"/api/runs/{run_id}/evaluation/jobs",
        f"/api/runs/{run_id}/knowledge-base",
        f"/api/runs/{run_id}/knowledge-base/assertions?source=promoted&limit=3",
        f"/api/runs/{run_id}/audit",
        f"/api/runs/{run_id}/retriever-service",
        f"/api/runs/{run_id}/artifacts?limit=3",
        f"/api/runs/{run_id}/files",
        f"/api/runs/{run_id}/urls",
        f"/api/runs/{run_id}/indexed-urls",
        f"/api/runs/{run_id}/media",
    ):
        response = client.get(path)
        assert response.status_code == 200, f"{path} failed: {response.text}"


def test_dashboard_file_preview_route(client: TestClient, existing_run: dict) -> None:
    run_id = existing_run["id"]
    files_response = client.get(f"/api/runs/{run_id}/files")
    assert files_response.status_code == 200
    payload = files_response.json()
    entries = payload.get("entries") or []
    if not entries:
        pytest.skip("Run has no browsable files for preview.")

    selected = next((item for item in entries if item.get("type") == "file"), None)
    if not selected:
        pytest.skip("Run root does not contain a previewable file.")

    preview_response = client.get(
        f"/api/runs/{run_id}/file-content",
        params={"path": selected["path"], "max_bytes": 2048},
    )
    assert preview_response.status_code == 200, preview_response.text
    preview_payload = preview_response.json()
    assert preview_payload["path"] == selected["path"]


def test_dashboard_eval_dataset_tools_and_job_cancel(client: TestClient, existing_run: dict) -> None:
    run_id = existing_run["id"]
    output_path = "eval/tmp/dashboard_pytest_eval.jsonl"

    init_response = client.post(
        "/api/evaluation/datasets/init",
        json={"output_path": output_path, "force": True},
    )
    assert init_response.status_code == 200, init_response.text
    init_payload = init_response.json()
    assert init_payload["output_path"].endswith(output_path)

    summarize_response = client.post(
        "/api/evaluation/datasets/summarize",
        json={"dataset_path": output_path},
    )
    assert summarize_response.status_code == 200, summarize_response.text
    assert summarize_response.json()["query_count"] >= 0

    assets_response = client.get("/api/evaluation/assets")
    assert assets_response.status_code == 200, assets_response.text
    datasets = assets_response.json().get("datasets") or []
    assert datasets, "Expected at least one evaluation dataset in eval/mbzuai_gold."

    validate_response = client.post(
        "/api/evaluation/datasets/validate",
        json={"dataset_path": datasets[0], "work_dir": existing_run["work_dir"]},
    )
    assert validate_response.status_code == 200, validate_response.text
    assert validate_response.json()["ok"] is True

    export_response = client.post(
        f"/api/runs/{run_id}/evaluation/benchmarks/export-ir",
        json={
            "dataset_id": "beir/scifact/test",
            "output_dir": "eval/dashboard_exports/scifact_pytest_smoke",
            "max_queries": 10,
            "max_docs": 100,
            "full_corpus": False,
        },
    )
    assert export_response.status_code == 200, export_response.text
    job_payload = export_response.json()
    job_id = job_payload["job_id"]

    cancel_response = client.post(f"/api/runs/{run_id}/evaluation/jobs/{job_id}/cancel")
    assert cancel_response.status_code == 200, cancel_response.text
    assert cancel_response.json()["status"] in {"cancelled", "completed"}

    time.sleep(0.2)
    jobs_response = client.get(f"/api/runs/{run_id}/evaluation/jobs")
    assert jobs_response.status_code == 200, jobs_response.text
    jobs = jobs_response.json()
    matching = next((item for item in jobs if item.get("job_id") == job_id), None)
    assert matching is not None
    assert matching["status"] in {"cancelled", "completed"}
