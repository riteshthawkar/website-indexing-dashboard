"""Minimal command-line entrypoint for the long-lived retrieval service.

Unlike ``python -m pipeline serve-retriever``, this module does not import the
indexing orchestrator, crawler, converters, or evaluation stack.  It is the
entrypoint for the retriever-only image and deliberately keeps that image's
dependency and attack surface small.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

import uvicorn

from pipeline.service.retrieval_api import create_retrieval_service_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="default", help="Config name or path")
    parser.add_argument("--work-dir", required=True, help="Indexed release work directory")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8060, help="Bind port")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=3,
        help="Maximum concurrent retrieval requests",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=90.0,
        help="Per-request timeout",
    )
    parser.add_argument(
        "--queue-timeout-seconds",
        type=float,
        default=3.0,
        help="Maximum wait for retrieval worker capacity",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    app = create_retrieval_service_app(
        config_name=args.config,
        work_dir=args.work_dir,
        max_concurrency=args.max_concurrency,
        request_timeout_seconds=args.request_timeout_seconds,
        queue_timeout_seconds=args.queue_timeout_seconds,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.verbose else "info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
