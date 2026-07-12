import asyncio
import json

from pipeline.evaluation.answer_readiness import (
    _chat_prediction_row_from_payload,
    _post_chat_request,
    _websocket_chat_request_async,
)
from pipeline.evaluation.dataset import EvalExample


def test_http_answer_prediction_preserves_retrieval_diagnostics(monkeypatch):
    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps(
                {
                    "response": "MBZUAI is in Abu Dhabi [1].",
                    "sources": [{"url": "https://mbzuai.ac.ae/about/"}],
                    "retrieval_diagnostics": {
                        "used_retrieval_service": True,
                        "retrieval_service": {"coverage_status": "complete"},
                    },
                    "evidence_pack": {
                        "items": [
                            {
                                "text": "MBZUAI is in Abu Dhabi.",
                                "source_url": "https://mbzuai.ac.ae/about/",
                            }
                        ]
                    },
                    "response_kind": "grounded",
                }
            ).encode("utf-8")

    monkeypatch.setattr(
        "pipeline.evaluation.answer_readiness.urllib.request.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    row = _post_chat_request(
        endpoint="http://127.0.0.1:8080/telegram-chat",
        example=EvalExample(id="q1", query="Where is MBZUAI?", query_type="fact"),
        auth_token=None,
        timeout_seconds=5,
    )

    assert row["retrieval_diagnostics"]["used_retrieval_service"] is True
    assert row["retrieval_diagnostics"]["retrieval_service"]["coverage_status"] == "complete"
    assert row["evidence_pack"]["items"][0]["text"] == "MBZUAI is in Abu Dhabi."


def test_answer_prediction_preserves_websocket_timeout_error_without_http_status():
    row = _chat_prediction_row_from_payload(
        payload={
            "response": "",
            "sources": [],
            "status": "error",
            "error": "websocket_answer_readiness_timeout",
        },
        example=EvalExample(id="timeout-row", query="What services are available?", query_type="scoped"),
        backend="production_chat_websocket",
        endpoint="ws://127.0.0.1:8000/chat",
        latency_ms=180000.0,
        eval_request_mode=True,
        terminal_event="timeout",
    )

    assert row["error"] == "websocket_answer_readiness_timeout"
    assert row["metadata"]["error"] == "websocket_answer_readiness_timeout"
    assert row["metadata"]["terminal_event"] == "timeout"


def test_websocket_answer_prediction_reports_receive_timeout(monkeypatch):
    import websockets

    class FakeConnection:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def recv(self):
            raise asyncio.TimeoutError()

    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: FakeConnection())

    row = asyncio.run(
        _websocket_chat_request_async(
            endpoint="ws://127.0.0.1:8000/chat",
            example=EvalExample(id="timeout-row", query="What services are available?", query_type="scoped"),
            auth_token=None,
            timeout_seconds=1,
        )
    )

    assert row["error"] == "websocket_answer_readiness_timeout"
    assert row["metadata"]["terminal_event"] == "timeout"


def test_openai_judge_fallback_can_be_disabled(monkeypatch):
    from pipeline.evaluation import answer_readiness

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ANSWER_READINESS_DISABLE_OPENAI_JUDGE_FALLBACK", "true")

    runtime_config = answer_readiness._judge_runtime_config()

    assert answer_readiness._openai_judge_fallback_model() == ""
    assert runtime_config["openai_fallback_enabled"] is False
    assert runtime_config["openai_fallback_disabled_by_env"] is True


def test_openai_judge_uses_capped_retry_settings(monkeypatch):
    from pipeline.evaluation import answer_readiness

    fake_client = object()
    captured = {}

    def fake_json_completion(**kwargs):
        captured.update(kwargs)
        return {"verdict": "pass", "overall": 1.0}

    monkeypatch.setenv("ANSWER_READINESS_OPENAI_JUDGE_ATTEMPTS", "1")
    monkeypatch.setenv("ANSWER_READINESS_OPENAI_JUDGE_RETRY_DELAY_SECONDS", "0.25")
    monkeypatch.setenv("ANSWER_READINESS_OPENAI_JUDGE_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("ANSWER_READINESS_OPENAI_CLIENT_MAX_RETRIES", "0")
    monkeypatch.setattr(answer_readiness, "_make_openai_judge_client", lambda: fake_client)
    monkeypatch.setattr(answer_readiness, "json_completion", fake_json_completion)

    result = answer_readiness._call_openai_judge_model(model="gpt-test", prompt="{}")

    assert result["verdict"] == "pass"
    assert captured["model"] == "gpt-test"
    assert captured["retries"] == 1
    assert captured["retry_delay_sec"] == 0.25
    assert captured["client"] is fake_client
