from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from pipeline.retrieval import AdaptiveHybridRetriever


logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query text.")
    request_id: str | None = Field(default=None, description="Optional caller-supplied request identifier.")


def _load_env_files() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _health_payload(app: FastAPI) -> Dict[str, Any]:
    started_at = float(getattr(app.state, "started_at_monotonic", time.monotonic()))
    return {
        "ok": True,
        "service": "retriever",
        "ready": bool(getattr(app.state, "ready", False)),
        "config_name": getattr(app.state, "config_name", None),
        "work_dir": str(getattr(app.state, "work_dir", "")),
        "request_count": int(getattr(app.state, "request_count", 0)),
        "error_count": int(getattr(app.state, "error_count", 0)),
        "max_concurrency": int(getattr(app.state, "max_concurrency", 0)),
        "request_timeout_seconds": float(getattr(app.state, "request_timeout_seconds", 0.0)),
        "uptime_seconds": round(max(0.0, time.monotonic() - started_at), 3),
    }


def create_retrieval_service_app(
    *,
    config_name: str,
    work_dir: str | Path,
    max_concurrency: int = 4,
    request_timeout_seconds: float = 90.0,
) -> FastAPI:
    config_name = str(config_name or "").strip()
    resolved_work_dir = Path(work_dir).resolve()
    if not config_name:
        raise ValueError("config_name is required")
    if not resolved_work_dir.exists():
        raise FileNotFoundError(f"Retrieval work directory does not exist: {resolved_work_dir}")
    bounded_concurrency = max(1, int(max_concurrency))
    bounded_timeout = max(1.0, float(request_timeout_seconds))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _load_env_files()
        app.state.started_at_monotonic = time.monotonic()
        app.state.ready = False
        app.state.request_count = 0
        app.state.error_count = 0
        app.state.config_name = config_name
        app.state.work_dir = resolved_work_dir
        app.state.max_concurrency = bounded_concurrency
        app.state.request_timeout_seconds = bounded_timeout
        app.state.semaphore = asyncio.Semaphore(bounded_concurrency)
        logger.info(
            "Loading retrieval service: config=%s work_dir=%s max_concurrency=%s timeout=%ss",
            config_name,
            resolved_work_dir,
            bounded_concurrency,
            bounded_timeout,
        )
        app.state.retriever = AdaptiveHybridRetriever.from_config(
            config_name=config_name,
            work_dir=resolved_work_dir,
        )
        app.state.ready = True
        logger.info("Retrieval service ready: config=%s work_dir=%s", config_name, resolved_work_dir)
        yield

    app = FastAPI(
        title="MBZUAI Retrieval Service",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        return _health_payload(app)

    @app.get("/readyz")
    async def readyz() -> Dict[str, Any]:
        if not bool(getattr(app.state, "ready", False)):
            raise HTTPException(status_code=503, detail="retriever_not_ready")
        return _health_payload(app)

    @app.post("/retrieve")
    async def retrieve_endpoint(payload: RetrieveRequest, request: Request) -> Dict[str, Any]:
        if not bool(getattr(app.state, "ready", False)):
            raise HTTPException(status_code=503, detail="retriever_not_ready")
        query = payload.query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="query must not be empty")

        request_id = payload.request_id or str(uuid.uuid4())
        started_at = time.perf_counter()
        retriever = app.state.retriever
        semaphore = app.state.semaphore
        app.state.request_count += 1
        try:
            async with semaphore:
                result = await asyncio.wait_for(
                    asyncio.to_thread(retriever.retrieve, query),
                    timeout=app.state.request_timeout_seconds,
                )
        except asyncio.TimeoutError as exc:
            app.state.error_count += 1
            raise HTTPException(status_code=504, detail="retrieval_timeout") from exc
        except HTTPException:
            app.state.error_count += 1
            raise
        except Exception as exc:  # pragma: no cover - exercised in live validation
            app.state.error_count += 1
            logger.exception("Retrieval request failed: request_id=%s path=%s", request_id, request.url.path)
            raise HTTPException(status_code=500, detail=f"retrieval_failed: {exc}") from exc

        output = dict(result or {})
        output["service_request_id"] = request_id
        output["service_latency_ms"] = round((time.perf_counter() - started_at) * 1000.0, 3)
        output["service_backend"] = "retrieval_service"
        output["service_config_name"] = app.state.config_name
        output["service_work_dir"] = str(app.state.work_dir)
        return output

    return app
