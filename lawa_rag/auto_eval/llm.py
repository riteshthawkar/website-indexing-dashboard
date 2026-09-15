"""Minimal structured-LLM boundary with lazy OpenAI and Gemini providers."""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol


class StructuredLLMError(RuntimeError):
    """Raised when a provider does not return a usable JSON object."""


@dataclass(frozen=True)
class LLMIdentity:
    provider: str
    model: str


class StructuredLLM(Protocol):
    @property
    def identity(self) -> LLMIdentity: ...

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
    ) -> Mapping[str, Any]: ...


def parse_json_object(value: str) -> dict[str, Any]:
    raw = str(value or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise StructuredLLMError("Provider response did not contain a JSON object")


@dataclass
class OpenAIStructuredLLM:
    model: str = "gpt-5-mini"
    api_key: str = field(default="", repr=False)
    timeout_seconds: float = 180.0
    max_output_tokens: int = 5000
    temperature: float | None = 0.2

    @property
    def identity(self) -> LLMIdentity:
        return LLMIdentity(provider="openai", model=self.model)

    def generate_json(self, *, system_prompt, user_prompt, schema_name):
        import openai

        key = self.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise StructuredLLMError("OPENAI_API_KEY is required")
        client = openai.OpenAI(api_key=key, timeout=self.timeout_seconds)
        options: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.temperature is not None:
            options["temperature"] = self.temperature
        response = client.chat.completions.create(**options)
        content = response.choices[0].message.content or ""
        return parse_json_object(content)


@dataclass
class GeminiStructuredLLM:
    model: str = "gemini-2.5-flash"
    api_key: str = field(default="", repr=False)
    max_output_tokens: int = 5000
    temperature: float = 0.2

    @property
    def identity(self) -> LLMIdentity:
        return LLMIdentity(provider="gemini", model=self.model)

    def generate_json(self, *, system_prompt, user_prompt, schema_name):
        from google import genai
        from google.genai import types

        key = self.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise StructuredLLMError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
        client = genai.Client(api_key=key)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        response = client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return parse_json_object(str(getattr(response, "text", "") or ""))


@dataclass
class RetryingStructuredLLM:
    delegate: StructuredLLM
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0

    @property
    def identity(self) -> LLMIdentity:
        return self.delegate.identity

    def generate_json(self, *, system_prompt, user_prompt, schema_name):
        error: Exception | None = None
        for attempt in range(max(1, self.max_attempts)):
            try:
                return self.delegate.generate_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema_name=schema_name,
                )
            except Exception as exc:  # provider errors are normalized at this boundary
                error = exc
                if attempt + 1 < self.max_attempts:
                    time.sleep(self.initial_delay_seconds * (2**attempt))
        raise StructuredLLMError(str(error or "Structured generation failed")) from error


def create_structured_llm(
    provider: str,
    *,
    model: str | None = None,
    max_attempts: int = 3,
) -> RetryingStructuredLLM:
    normalized = str(provider or "").strip().lower()
    if normalized == "openai":
        delegate = OpenAIStructuredLLM(model=model or "gpt-5-mini")
    elif normalized == "gemini":
        delegate = GeminiStructuredLLM(model=model or "gemini-2.5-flash")
    else:
        raise ValueError("provider must be 'openai' or 'gemini'")
    return RetryingStructuredLLM(delegate=delegate, max_attempts=max_attempts)


__all__ = [
    "GeminiStructuredLLM",
    "LLMIdentity",
    "OpenAIStructuredLLM",
    "RetryingStructuredLLM",
    "StructuredLLM",
    "StructuredLLMError",
    "create_structured_llm",
    "parse_json_object",
]
