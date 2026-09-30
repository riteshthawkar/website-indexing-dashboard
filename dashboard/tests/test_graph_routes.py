"""Isolated API tests; no production DB, pipeline worker or external API is used."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from graph_routes import make_graph_router
from graph_service import GraphService
from graph_preview_guard import preview_guard
from lawa_rag.graphs import GraphIndex


@pytest.fixture
def graph_client(tmp_path):
    source = tmp_path / "page_link_graph.json"
    source.write_text(
        json.dumps(
            {"nodes": [{"id": "page/a", "label": "Admissions", "node_type": "page"}], "edges": []}
        )
    )
    service = GraphService(tmp_path / "cache")

    def run(identity):
        if identity != 1:
            raise HTTPException(404, "Run not found")
        return SimpleNamespace(work_dir=str(tmp_path))

    app = FastAPI()
    app.include_router(make_graph_router(run, service))
    with TestClient(app) as client:
        graph = client.get("/api/runs/1/graphs").json()["items"][0]
        yield client, graph["id"], source, service
    service.shutdown()


def test_graph_routes_validate_run_artifact_and_query_bounds(graph_client):
    client, identity, source, service = graph_client
    assert client.get("/api/runs/999/graphs").status_code == 404
    assert client.get("/api/runs/1/graphs/unknown").status_code == 404
    prefix = f"/api/runs/1/graphs/{identity}"
    assert client.get(f"{prefix}/nodes").status_code == 409
    GraphIndex.build(source, service.cache_dir)
    assert client.get(prefix).json()["status"] == "ready"
    assert (
        client.get(f"{prefix}/nodes", params={"q": "Admissions"}).json()["items"][0]["id"]
        == "page/a"
    )
    assert client.get(f"{prefix}/node", params={"node_id": "page/a"}).status_code == 200
    assert client.get(f"{prefix}/node", params={"node_id": "missing"}).status_code == 404
    for args in ({"hops": 9}, {"max_nodes": 251}, {"max_edges": 1501}, {"direction": "arbitrary"}, {"edge_mode": "arbitrary"}):
        assert (
            client.get(f"{prefix}/neighborhood", params={"node_id": "page/a", **args}).status_code
            == 422
        )
    result = client.get(f"{prefix}/neighborhood", params={"node_id": "page/a"})
    assert result.status_code == 200 and result.json()["root"]["id"] == "page/a"
    assert client.get(f"{prefix}/neighborhood", params={"node_id": "page/a", "hops": 8, "edge_mode": "tree"}).status_code == 200
    overview = client.get(f"{prefix}/website").json()
    assert overview["overview"]["total_pages"] == 1
    assert overview["limits"]["hops"] is None
    assert client.get(f"{prefix}/website", params={"edge_mode": "all"}).status_code == 422
    assert client.get("/api/runs/999/graphs/unknown/website").status_code == 404


def test_background_preparation_and_failure_are_visible(graph_client):
    client, identity, source, service = graph_client
    prefix = f"/api/runs/1/graphs/{identity}"
    assert client.get(prefix).json()["status"] == "building"
    for job in list(service._jobs.values()):
        job.result(timeout=5)
    assert client.get(prefix).json()["status"] == "ready"
    source.write_text("{broken")
    assert client.get(prefix).json()["status"] == "building"
    for job in list(service._jobs.values()):
        try:
            job.result(timeout=5)
        except ValueError:
            pass
    assert client.get(prefix).json()["status"] == "failed"


def test_preview_registration_cannot_start_jobs_or_modify_run_artifacts():
    app = FastAPI()
    app.middleware("http")(
        preview_guard(lambda identity: "graph_preview" if identity == 1 else "full")
    )

    @app.api_route("/api/runs/{identity}/{action:path}", methods=["GET", "POST", "DELETE"])
    def action(identity: int, action: str):
        return {"ok": True}

    with TestClient(app) as client:
        for operation in (
            "start",
            "resume",
            "restart",
            "stages/crawl/retry",
            "evaluation/answers",
            "retriever-service/start",
        ):
            assert client.post(f"/api/runs/1/{operation}", json={}).status_code == 403
        assert client.get("/api/runs/1/graphs").status_code == 200
        assert client.post("/api/runs/2/start").status_code == 200


def test_cache_eviction_preserves_unrelated_files(tmp_path):
    service = GraphService(tmp_path / "cache")
    service.cache_dir.mkdir()
    unrelated = service.cache_dir / ("f" * 64 + ".sqlite")
    with sqlite3.connect(unrelated) as database:
        database.execute("CREATE TABLE unrelated(value TEXT)")
    before = unrelated.read_bytes()
    source = tmp_path / "page_link_graph.json"
    for number in range(4):
        source.write_text(json.dumps({"nodes": [{"id": str(number)}], "edges": []}))
        path = GraphIndex.cache_path(source, service.cache_dir)
        service._build(source, path)
    assert unrelated.read_bytes() == before
    # At most two prior viewer snapshots plus the new one; unrelated DB excluded.
    assert len(list(service.cache_dir.glob("*.sqlite"))) == 4
    service.shutdown()
