from __future__ import annotations

import asyncio
import logging
from pathlib import Path
import sys


DASHBOARD_DIR = Path(__file__).resolve().parents[1]
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from run_executor import _DashboardStageLogHandler, _resolve_terminal_error_message


def test_stage_log_handler_forwards_stage_logs() -> None:
    writes: list[dict] = []
    broadcasts: list[dict] = []

    def _write_log(level: str, message: str, **kwargs):
        payload = {
            "level": level,
            "message": message,
            **kwargs,
        }
        writes.append(payload)
        return payload

    def _broadcast_log(**kwargs):
        broadcasts.append(kwargs)

    async def _exercise() -> None:
        handler = _DashboardStageLogHandler(
            loop=asyncio.get_running_loop(),
            write_log=_write_log,
            broadcast_log=_broadcast_log,
            current_stage_getter=lambda: "crawler/crawl4ai",
        )
        record = logging.LogRecord(
            name="pipeline.stages.crawlers.crawl4ai_crawler",
            level=logging.INFO,
            pathname=__file__,
            lineno=42,
            msg="Fetched %d pages",
            args=(12,),
            exc_info=None,
        )
        handler.emit(record)
        await asyncio.sleep(0.05)

    asyncio.run(_exercise())

    assert len(writes) == 1
    event = writes[0]
    assert event["level"] == "info"
    assert event["message"] == "Fetched 12 pages"
    assert event["stage"] == "crawler/crawl4ai"
    assert event["event_type"] == "stage_log"
    assert event["data"]["logger"] == "pipeline.stages.crawlers.crawl4ai_crawler"
    assert len(broadcasts) == 1


def test_resolve_terminal_error_message_prefers_failed_stage() -> None:
    class Stage:
        def __init__(self, status: str, error_message: str | None) -> None:
            self.status = status
            self.error_message = error_message

    class State:
        stages = [
            Stage("skipped", "No supported documents found in download_dir"),
            Stage("failed", "No chunk_index available for extraction slice formatting"),
        ]

    assert _resolve_terminal_error_message(State()) == "No chunk_index available for extraction slice formatting"
