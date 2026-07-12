from __future__ import annotations

from pipeline.service import retrieval_runner


def test_minimal_runner_forwards_runtime_settings(monkeypatch, tmp_path):
    captured = {}
    fake_app = object()

    def fake_create_app(**kwargs):
        captured["app_kwargs"] = kwargs
        return fake_app

    def fake_run(app, **kwargs):
        captured["uvicorn_app"] = app
        captured["uvicorn_kwargs"] = kwargs

    monkeypatch.setattr(retrieval_runner, "create_retrieval_service_app", fake_create_app)
    monkeypatch.setattr(retrieval_runner.uvicorn, "run", fake_run)

    result = retrieval_runner.main(
        [
            "--config",
            "mbzuai_production",
            "--work-dir",
            str(tmp_path),
            "--host",
            "0.0.0.0",
            "--port",
            "8061",
            "--max-concurrency",
            "7",
            "--request-timeout-seconds",
            "45",
            "--queue-timeout-seconds",
            "0.5",
        ]
    )

    assert result == 0
    assert captured["app_kwargs"] == {
        "config_name": "mbzuai_production",
        "work_dir": str(tmp_path),
        "max_concurrency": 7,
        "request_timeout_seconds": 45.0,
        "queue_timeout_seconds": 0.5,
    }
    assert captured["uvicorn_app"] is fake_app
    assert captured["uvicorn_kwargs"] == {
        "host": "0.0.0.0",
        "port": 8061,
        "log_level": "info",
    }
