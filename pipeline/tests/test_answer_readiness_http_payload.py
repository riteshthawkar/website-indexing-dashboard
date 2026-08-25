import asyncio
import json
import threading
import time

import pytest

from pipeline.evaluation.answer_readiness import (
    _chat_prediction_row_from_payload,
    _looks_like_no_answer,
    _post_chat_request,
    _required_term_supported,
    _run_websocket_answer_predictions,
    _websocket_chat_request_async,
    evaluate_answer_readiness,
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


def test_http_answer_prediction_sends_dataset_language_and_protocol(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return json.dumps({"response": "تقع الجامعة في أبوظبي.", "sources": []}).encode("utf-8")

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "pipeline.evaluation.answer_readiness.urllib.request.urlopen",
        fake_urlopen,
    )

    row = _post_chat_request(
        endpoint="http://127.0.0.1:8080/telegram-chat",
        example=EvalExample(
            id="ar-q1",
            query="أين تقع جامعة محمد بن زايد للذكاء الاصطناعي؟",
            query_type="fact",
            language="Arabic",
        ),
        auth_token=None,
        timeout_seconds=5,
    )

    assert captured["payload"]["language"] == "Arabic"
    assert captured["payload"]["protocol_version"] == "1.0"
    assert row["language"] == "Arabic"


def test_arabic_answer_matching_normalizes_diacritics_digits_and_articles():
    assert _required_term_supported("تتوفر مختبرات حديثة للطلاب.", "المختبرات")
    assert _required_term_supported("يبدأ الدعم الساعة ٨:٠٠ صباحا.", "8:00 صباحًا")
    assert _looks_like_no_answer("لا توجد معلومات موثوقة عن هذا المكتب في المصادر المتاحة.")


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


def test_websocket_answer_prediction_captures_first_content_latency_and_arabic_payload(monkeypatch):
    import websockets

    sent_payloads = []

    class FakeConnection:
        def __init__(self):
            self.messages = [
                json.dumps({"status": "connected"}),
                json.dumps({"event": "chunk", "delta": "تقع الجامعة"}),
                json.dumps(
                    {
                        "event": "final",
                        "terminal": True,
                        "status": "done",
                        "response": "تقع الجامعة في أبوظبي [1].",
                        "sources": [{"url": "https://mbzuai.ac.ae/about/"}],
                    }
                ),
            ]

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def recv(self):
            return self.messages.pop(0)

        async def send(self, payload):
            sent_payloads.append(json.loads(payload))

    monkeypatch.setattr(websockets, "connect", lambda *args, **kwargs: FakeConnection())

    row = asyncio.run(
        _websocket_chat_request_async(
            endpoint="ws://127.0.0.1:8000/chat",
            example=EvalExample(
                id="ar-stream",
                query="أين تقع الجامعة؟",
                query_type="fact",
                language="Arabic",
            ),
            auth_token=None,
            timeout_seconds=1,
        )
    )

    assert sent_payloads[0]["language"] == "Arabic"
    assert sent_payloads[0]["protocol_version"] == "1.0"
    assert row["metadata"]["terminal_event"] == "final"
    assert 0.0 <= row["first_content_latency_ms"] <= row["latency_ms"]


def test_websocket_prediction_runner_honors_parallelism_and_preserves_dataset_order(
    tmp_path,
    monkeypatch,
):
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0
    examples = [
        EvalExample(id=f"q{index}", query=f"Question {index}", query_type="fact")
        for index in range(4)
    ]

    def fake_websocket_request(*, example, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        barrier.wait(timeout=2.0)
        time.sleep(0.01)
        with lock:
            active -= 1
        return _chat_prediction_row_from_payload(
            payload={"response": f"Answer {example.id}", "sources": []},
            example=example,
            backend="production_chat_websocket",
            endpoint="ws://chat.test/chat",
            latency_ms=10.0,
            first_content_latency_ms=4.0,
            eval_request_mode=True,
            terminal_event="final",
        )

    monkeypatch.setattr(
        "pipeline.evaluation.answer_readiness._websocket_chat_request",
        fake_websocket_request,
    )

    rows = _run_websocket_answer_predictions(
        endpoint="ws://chat.test/chat",
        dataset_path=tmp_path / "dataset.jsonl",
        predictions_path=tmp_path / "predictions.jsonl",
        examples=examples,
        dataset_fingerprint="dataset-fingerprint",
        config_name="cfg",
        work_dir=tmp_path,
        auth_token=None,
        timeout_seconds=5,
        parallelism=2,
    )

    assert max_active == 2
    assert [row["id"] for row in rows] == [example.id for example in examples]
    assert all(row["first_content_latency_ms"] == 4.0 for row in rows)


@pytest.mark.parametrize(
    ("mode", "runner_name"),
    [
        ("http", "_run_http_answer_predictions"),
        ("websocket", "_run_websocket_answer_predictions"),
    ],
)
def test_answer_readiness_forwards_parallelism_to_network_runners(
    tmp_path,
    monkeypatch,
    mode,
    runner_name,
):
    dataset_path = tmp_path / "dataset.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps({"id": f"q{index}", "query": f"Question {index}", "query_type": "fact"})
            for index in range(3)
        )
        + "\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_runner(**kwargs):
        captured["parallelism"] = kwargs["parallelism"]
        return []

    monkeypatch.setattr(f"pipeline.evaluation.answer_readiness.{runner_name}", fake_runner)

    report = evaluate_answer_readiness(
        config_name="cfg",
        work_dir=tmp_path,
        dataset_path=dataset_path,
        predictions_path=tmp_path / f"{mode}-predictions.jsonl",
        mode=mode,
        endpoint="ws://chat.test/chat" if mode == "websocket" else "http://chat.test/telegram-chat",
        parallelism=3,
    )

    assert captured["parallelism"] == 3
    assert report["execution"] == {
        "parallelism_requested": 3,
        "parallelism_effective": 3,
    }


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
