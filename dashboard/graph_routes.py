"""Graph explorer HTTP boundary: run-scoped catalog IDs and bounded queries only."""

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Query

from graph_service import graph_service
from lawa_rag.graphs import discover_graphs, resolve_graph


def make_graph_router(load_run, service=graph_service):
    router = APIRouter(prefix="/api/runs/{run_id}/graphs", tags=["graph explorer"])

    def root(run_id):
        run = load_run(run_id)
        if not run.work_dir:
            raise HTTPException(404, "Run has no saved artifacts yet")
        return Path(run.work_dir)

    async def source(run_id, graph_id):
        try:
            return await asyncio.to_thread(resolve_graph, root(run_id), graph_id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc

    async def execute(run_id, graph_id, method, *args, **kwargs):
        path = await source(run_id, graph_id)

        def query():
            return getattr(service.ready(path), method)(*args, **kwargs)

        try:
            return await asyncio.to_thread(query)
        except KeyError as exc:
            raise HTTPException(404, "Node not found in this snapshot") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except (RuntimeError, FileNotFoundError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.get("")
    async def catalog(run_id: int):
        return {"items": await asyncio.to_thread(discover_graphs, root(run_id))}

    @router.get("/{graph_id}")
    async def status(run_id: int, graph_id: str, retry: bool = False):
        path = await source(run_id, graph_id)
        return await asyncio.to_thread(service.status, path, retry=retry)

    @router.get("/{graph_id}/nodes")
    async def search(
        run_id: int,
        graph_id: str,
        q: str = Query("", max_length=300),
        node_type: str = Query("", max_length=120),
        limit: int = Query(30, ge=1, le=100),
        offset: int = Query(0, ge=0, le=10000),
    ):
        return await execute(run_id, graph_id, "search", q, node_type, limit, offset)

    @router.get("/{graph_id}/node")
    async def node(
        run_id: int, graph_id: str, node_id: str = Query(..., min_length=1, max_length=2048)
    ):
        return await execute(run_id, graph_id, "node", node_id)

    @router.get("/{graph_id}/neighborhood")
    async def neighborhood(
        run_id: int,
        graph_id: str,
        node_id: str = Query(..., min_length=1, max_length=2048),
        hops: int = Query(1, ge=1, le=8),
        direction: Literal["both", "in", "out"] = "both",
        edge_type: str = Query("", max_length=120),
        max_nodes: int = Query(40, ge=1, le=250),
        max_edges: int = Query(120, ge=1, le=1500),
        edge_mode: Literal["all", "forward", "tree"] = "all",
    ):
        return await execute(
            run_id,
            graph_id,
            "neighborhood",
            node_id,
            hops=hops,
            direction=direction,
            edge_type=edge_type,
            max_nodes=max_nodes,
            max_edges=max_edges,
            edge_mode=edge_mode,
        )

    @router.get("/{graph_id}/website")
    async def website(
        run_id: int,
        graph_id: str,
        edge_mode: Literal["tree", "forward"] = "tree",
    ):
        return await execute(run_id, graph_id, "website", edge_mode=edge_mode)

    return router
