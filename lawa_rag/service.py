"""Retriever service application factory."""

from __future__ import annotations

from os import PathLike
from typing import Any


def create_retrieval_service_app(
    *,
    config_name: str,
    work_dir: str | PathLike[str],
    max_concurrency: int = 3,
    request_timeout_seconds: float = 90.0,
    queue_timeout_seconds: float = 3.0,
) -> Any:
    """Create the production FastAPI retriever app.

    FastAPI and provider modules are imported only when this factory is called,
    keeping the core package usable for config and artifact tooling.
    """

    from pipeline.service.retrieval_api import create_retrieval_service_app as factory

    return factory(
        config_name=config_name,
        work_dir=work_dir,
        max_concurrency=max_concurrency,
        request_timeout_seconds=request_timeout_seconds,
        queue_timeout_seconds=queue_timeout_seconds,
    )


__all__ = ["create_retrieval_service_app"]
