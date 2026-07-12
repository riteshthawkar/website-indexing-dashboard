from __future__ import annotations

import json
import os
import re
import time
from threading import local
from typing import Any, Dict, Iterable, List, Mapping, Sequence


_OPENAI_STATE = local()


def _sanitize_json(raw: str) -> str:
    return re.sub(r",(\s*[\]}])", r"\1", str(raw or ""))


def make_openai_client(*, timeout_sec: float | None = None) -> Any:
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - import path validated in stage config
        raise RuntimeError("openai is not installed. Run: pip install openai") from exc

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")

    resolved_timeout_sec = max(
        0.1,
        float(timeout_sec) if timeout_sec is not None else float(os.getenv("OPENAI_TIMEOUT_SEC", "120")),
    )
    clients = getattr(_OPENAI_STATE, "clients", None)
    if clients is None:
        clients = {}
        _OPENAI_STATE.clients = clients
    cache_key = round(resolved_timeout_sec, 3)
    client = clients.get(cache_key)
    if client is None:
        client = openai.OpenAI(api_key=api_key, timeout=resolved_timeout_sec)
        clients[cache_key] = client
        if timeout_sec is None:
            # Preserve the legacy attribute for callers that inspect thread-local
            # state while allowing deadline-specific clients to coexist safely.
            _OPENAI_STATE.client = client
    return client


def _supports_explicit_temperature(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return not normalized.startswith("gpt-5")


def json_completion(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 8000,
    json_schema: Dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
    retries: int = 3,
    retry_delay_sec: float = 3.0,
    per_request_delay_sec: float = 0.0,
    client: Any | None = None,
) -> Dict[str, Any]:
    active_client = client or make_openai_client()
    last_error: Exception | None = None

    for attempt in range(max(1, int(retries))):
        try:
            if per_request_delay_sec > 0:
                time.sleep(float(per_request_delay_sec))
            request_kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=int(max_completion_tokens),
            )
            if json_schema:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": dict(json_schema),
                }
            else:
                request_kwargs["response_format"] = {"type": "json_object"}
            if _supports_explicit_temperature(model):
                request_kwargs["temperature"] = float(temperature)
            if reasoning_effort:
                request_kwargs["reasoning_effort"] = str(reasoning_effort)
            response = active_client.chat.completions.create(**request_kwargs)
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(_sanitize_json(content))
            if not isinstance(parsed, dict):
                raise ValueError("OpenAI JSON response was not an object")
            return parsed
        except Exception as exc:  # pragma: no cover - network path exercised in integration
            last_error = exc
            if attempt + 1 >= max(1, int(retries)):
                break
            time.sleep(float(retry_delay_sec))

    raise RuntimeError(f"OpenAI JSON completion failed: {last_error}")


def batched(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    size = max(1, int(batch_size or 1))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def merge_overrides(base: Mapping[str, Any] | None, override: Mapping[str, Any] | None) -> Dict[str, Any]:
    payload = dict(base or {})
    for key, value in dict(override or {}).items():
        if isinstance(value, dict) and isinstance(payload.get(key), dict):
            payload[key] = merge_overrides(payload[key], value)
        else:
            payload[key] = value
    return payload
