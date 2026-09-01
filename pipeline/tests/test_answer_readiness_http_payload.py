import asyncio
import json
import threading
import time

import pytest

from pipeline.evaluation.answer_readiness import (
    _build_judge_prompt,
    _chat_prediction_row_from_payload,
    _forbidden_term_present,
    _eval_context_referrer,
    _looks_like_no_answer,
    _post_chat_request,
    _required_term_supported,
    _normalize_url_for_match,
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


def test_answer_readiness_supplies_referrer_only_for_page_deictic_examples():
    deictic = EvalExample(
        id="deictic",
        query="What is the main point on this page?",
        query_type="fact",
        metadata={"required_pages": ["https://metaverse.mbzuai.ac.ae/press/example"]},
    )
    ordinary = EvalExample(
        id="ordinary",
        query="What is the main point of the project?",
        query_type="fact",
        metadata={"required_pages": ["https://metaverse.mbzuai.ac.ae/press/example"]},
    )

    assert _eval_context_referrer(deictic) == "https://metaverse.mbzuai.ac.ae/press/example"
    assert _eval_context_referrer(ordinary) == ""


def test_arabic_answer_matching_normalizes_diacritics_digits_and_articles():
    assert _required_term_supported("تتوفر مختبرات حديثة للطلاب.", "المختبرات")
    assert _required_term_supported("يبدأ الدعم الساعة ٨:٠٠ صباحا.", "8:00 صباحًا")
    assert _looks_like_no_answer("لا توجد معلومات موثوقة عن هذا المكتب في المصادر المتاحة.")


def test_arabic_answer_matching_handles_possessive_taa_marbuta_and_visual_labels():
    assert _required_term_supported(
        "ترسيخ مكانتها كمركز رائد لمجتمع الذكاء الاصطناعي في أبوظبي.",
        "ترسيخ مكانة أبوظبي",
    )
    assert _required_term_supported(
        "تُظهر الصورة معالجاً حاسوبياً (computer processor chip).",
        "شريحة معالج",
    )
    assert _required_term_supported(
        "يمر المخطط بمرحلة Filtering ثم التنظيف.",
        "التصفية",
    )


def test_arabic_answer_matching_accepts_governed_lexical_equivalents():
    assert _required_term_supported("تُظهر الصورة تجمّع المياه بعد المطر.", "تراكم المياه")
    assert _required_term_supported("يعتمد العمل على نموذج تعاوني مع الشركاء.", "التعاون")
    assert _required_term_supported("تشرف لجان المجلس والإدارة على شؤون الجامعة.", "لجان إدارية")


def test_arabic_answer_matching_handles_feminine_plural_with_attached_pronoun():
    assert _required_term_supported(
        "رقم الهاتف المنشور لهذا الاستفسار هو +971 (0) 2 811 3203.",
        "استفساراتكم",
    )


def test_judge_prompt_uses_explicit_dubai_reference_datetime(monkeypatch):
    monkeypatch.setenv("ANSWER_READINESS_REFERENCE_DATETIME", "2026-08-29T12:30:00+04:00")

    prompt = _build_judge_prompt(
        EvalExample(
            id="time-aware-judge",
            query="Which listed news dates have passed?",
            query_type="scoped",
        ),
        {"response": "The August 20, 2026 item has passed."},
    )

    assert "mbzuai-answer-readiness-judge-v3" in prompt
    assert "2026-08-29T12:30:00+04:00" in prompt
    assert "do not substitute a model training date" in prompt
    assert "evaluation_reference_datetime" in prompt


def test_answer_matching_normalizes_thousands_separators():
    assert _required_term_supported("The cohort includes 5,000 participants.", "5000 participants")
    assert _required_term_supported("تضم المبادرة ٥٬٠٠٠ مشارك.", "5000 مشارك")


def test_answer_matching_accepts_interpretable_hazard_scale_paraphrase():
    assert _required_term_supported(
        "Average hazard is an interpretable, hazard-scale integral measure.",
        "hazard-scale interpretation",
    )


def test_answer_matching_recognizes_production_no_relevant_information_wording():
    assert _looks_like_no_answer("No relevant information found in the supplied MBZUAI sources.")
    assert _looks_like_no_answer(
        "I do not have any information showing that MBZUAI has a private airport. [1]"
    )
    assert _looks_like_no_answer(
        "I don’t currently have information confirming that MBZUAI operates a private airport."
    )


def test_answer_matching_decodes_visible_percent_encoded_titles():
    assert _required_term_supported(
        "Open Statistics%20for%20Business in the official site.",
        "Statistics for Business",
    )


def test_answer_matching_accepts_ahead_before_visit_paraphrase():
    assert _required_term_supported(
        "Visitors should call before visiting MBZUAI.",
        "ahead of their visit",
    )


def test_forbidden_term_matching_ignores_explicit_negative_caveat():
    assert not _forbidden_term_present(
        "The funded package does not guarantee airfare.",
        "airfare",
    )
    assert not _forbidden_term_present(
        "Airfare is not guaranteed as part of the funded package.",
        "airfare",
    )
    assert _forbidden_term_present(
        "The funded package includes airfare and health insurance.",
        "airfare",
    )


def test_answer_matching_recognizes_supported_context_followed_by_premise_denial():
    from pipeline.evaluation.answer_readiness import (
        _explicitly_denies_unsupported_premise,
    )

    response = (
        "MBZUAI is five kilometers from Abu Dhabi International Airport, but I do "
        "not have any information indicating that MBZUAI has a private airport or an "
        "IATA code for one. [1]"
    )

    assert _explicitly_denies_unsupported_premise(response)


def test_answer_url_matching_treats_percent_encoded_arabic_path_as_equivalent():
    encoded = "https://mbzuai.ac.ae/ar/news/%D8%A7%D9%84%D8%B0%D9%83%D8%A7%D8%A1"
    decoded = "https://mbzuai.ac.ae/ar/news/الذكاء"

    assert _normalize_url_for_match(encoded) == _normalize_url_for_match(decoded)
    assert _normalize_url_for_match(encoded.replace("%", "%25")) == _normalize_url_for_match(decoded)


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


def test_answer_prediction_preserves_structured_navigation_plan():
    navigation_plan = {
        "schema_version": "mbzuai.navigation_plan.v1",
        "status": "ready",
        "intent": "search",
        "steps": [
            {
                "action_type": "search",
                "target_url": "https://metaverse.mbzuai.ac.ae/",
            }
        ],
    }
    row = _chat_prediction_row_from_payload(
        payload={
            "response": "The Search action opens the Metaverse homepage. [1]",
            "sources": [{"url": "https://metaverse.mbzuai.ac.ae/publications"}],
            "navigation_plan": navigation_plan,
        },
        example=EvalExample(
            id="navigation-row",
            query="Where does Search go?",
            query_type="scoped",
        ),
        backend="production_chat_http",
        endpoint="http://127.0.0.1:8000/telegram-chat",
        latency_ms=10.0,
        eval_request_mode=True,
    )

    assert row["navigation_plan"] == navigation_plan


def test_answer_prediction_preserves_backend_phase_timings():
    timings_ms = {
        "query_rewrite": 1250.0,
        "retrieval": 3400.5,
        "draft_generation": 725.25,
        "time_to_first_chunk": 5400.0,
        "time_to_final": 5600.0,
    }
    row = _chat_prediction_row_from_payload(
        payload={
            "response": "MBZUAI is in Abu Dhabi [1].",
            "sources": [{"url": "https://mbzuai.ac.ae/about/"}],
            "timings_ms": timings_ms,
        },
        example=EvalExample(
            id="timed-row",
            query="Where is MBZUAI?",
            query_type="fact",
        ),
        backend="production_chat_websocket",
        endpoint="ws://127.0.0.1:8000/chat",
        latency_ms=5610.0,
        first_content_latency_ms=5410.0,
        eval_request_mode=True,
        terminal_event="final",
    )

    assert row["timings_ms"] == timings_ms
    assert row["metadata"]["timings_ms"] == timings_ms


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


def test_llm_judge_honors_bounded_parallelism_and_preserves_dataset_order(monkeypatch):
    from pipeline.evaluation import answer_readiness

    barrier = threading.Barrier(2)
    lock = threading.Lock()
    active = 0
    max_active = 0
    progress_events = []
    examples = [
        EvalExample(id=f"q{index}", query=f"Question {index}", query_type="fact")
        for index in range(4)
    ]

    def fake_judge_answer_row(*, example, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        if example.id in {"q0", "q1"}:
            barrier.wait(timeout=2.0)
        time.sleep(0.01)
        with lock:
            active -= 1
        return {
            "verdict": "pass",
            "overall": 1.0,
            "judge_provider": "gemini",
            "judge_model": "gemini-test",
        }

    monkeypatch.setenv("ANSWER_READINESS_JUDGE_MAX_PARALLELISM", "2")
    monkeypatch.setattr(answer_readiness, "_make_judge_client", lambda **_kwargs: object())
    monkeypatch.setattr(answer_readiness, "_judge_answer_row", fake_judge_answer_row)

    judged = answer_readiness._run_llm_judge(
        examples=examples,
        rows_by_id={example.id: {"response": "answer"} for example in examples},
        model="gemini-test",
        timeout_seconds=5.0,
        allow_openai_fallback=False,
        parallelism=8,
        progress_callback=lambda event, payload: progress_events.append((event, payload)),
    )

    row_events = [
        payload
        for event, payload in progress_events
        if event == "answer_readiness_judge_row_done"
    ]
    assert max_active == 2
    assert list(judged) == [example.id for example in examples]
    assert sorted(payload["id"] for payload in row_events) == [example.id for example in examples]
    assert sorted(payload["completed"] for payload in row_events) == [1, 2, 3, 4]
    assert all(result["judge_provider"] == "gemini" for result in judged.values())


def test_llm_judge_uses_transport_timeout_without_nested_worker(monkeypatch):
    from pipeline.evaluation import answer_readiness

    captured = {}

    class FakeHttpOptions:
        def __init__(self, *, timeout):
            captured["timeout"] = timeout

    class FakeTypes:
        HttpOptions = FakeHttpOptions

    class FakeModels:
        def generate_content(self, *, model, contents):
            captured["thread_id"] = threading.get_ident()
            captured["model"] = model
            captured["contents"] = contents
            return type("Response", (), {"text": "ok"})()

    class FakeGenai:
        class Client:
            def __init__(self, *, api_key, http_options):
                captured["api_key"] = api_key
                captured["http_options"] = http_options
                self.models = FakeModels()

    monkeypatch.setenv("GOOGLE_API_KEY", "judge-key")
    monkeypatch.setattr(answer_readiness, "import_genai", lambda: FakeGenai)
    monkeypatch.setattr(answer_readiness, "import_genai_types", lambda: FakeTypes)

    caller_thread_id = threading.get_ident()
    client = answer_readiness._make_judge_client(timeout_seconds=12.5)
    result = answer_readiness._call_judge_model(
        client,
        model="gemini-test",
        prompt="Judge this answer",
        timeout_seconds=12.5,
    )

    assert captured["timeout"] == 12_500
    assert captured["thread_id"] == caller_thread_id
    assert captured["model"] == "gemini-test"
    assert captured["contents"] == "Judge this answer"
    assert result == "ok"


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
