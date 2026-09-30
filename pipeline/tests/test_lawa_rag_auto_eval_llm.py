"""Provider-boundary and retry tests for structured auto-evaluation calls."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from lawa_rag.auto_eval import (
    LLMIdentity,
    RetryingStructuredLLM,
    StructuredLLMError,
    create_structured_llm,
)
from lawa_rag.auto_eval.llm import parse_json_object
from lawa_rag.auto_eval._generation_support import critic_accepted


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FlakyLLM:
    def __init__(self):
        self.calls = 0

    @property
    def identity(self) -> LLMIdentity:
        return LLMIdentity("test", "flaky")

    def generate_json(self, **kwargs):
        self.calls += 1
        if self.calls < 3:
            raise RuntimeError("temporary provider failure")
        return {"passed": True}


def test_json_parser_accepts_fenced_output_but_rejects_non_objects() -> None:
    assert parse_json_object("```json\n{\"query\": \"hello\"}\n```") == {
        "query": "hello"
    }
    with pytest.raises(StructuredLLMError):
        parse_json_object("[1, 2, 3]")


def test_retry_wrapper_retries_transient_provider_failures() -> None:
    delegate = FlakyLLM()
    llm = RetryingStructuredLLM(
        delegate=delegate,
        max_attempts=3,
        initial_delay_seconds=0,
    )

    result = llm.generate_json(system_prompt="system", user_prompt="user", schema_name="test")

    assert result == {"passed": True}
    assert delegate.calls == 3
    assert llm.identity == delegate.identity


def test_provider_factory_is_lazy_and_validates_provider_name() -> None:
    openai_llm = create_structured_llm("openai")
    assert openai_llm.identity.provider == "openai"
    assert openai_llm.identity.model == "gpt-4.1-mini"
    assert openai_llm.delegate.temperature is None
    gemini_llm = create_structured_llm("gemini")
    assert gemini_llm.identity.provider == "gemini"
    assert gemini_llm.identity.model == "gemini-3.5-flash-lite"
    assert gemini_llm.delegate.timeout_seconds == 120
    with pytest.raises(ValueError, match="provider"):
        create_structured_llm("unknown")


def test_critic_gate_requires_boolean_pass_and_calibrated_scores() -> None:
    passing = {
        "passed": True,
        "answerability_correct": True,
        "grounding_score": 0.9,
        "naturalness_score": 0.6,
        "difficulty_match_score": 0.6,
        "query_grounding_score": 0.9,
        "answer_completeness_score": 0.9,
        "task_match_score": 0.75,
        "persona_fit_score": 0.6,
        "language_quality_score": 0.8,
    }

    assert critic_accepted(passing)
    assert not critic_accepted({**passing, "passed": "true"})
    assert not critic_accepted({**passing, "grounding_score": 0.89})
    assert not critic_accepted({**passing, "task_match_score": 0.74})


def test_import_does_not_initialize_external_provider_sdks() -> None:
    script = """
import sys
import lawa_rag.auto_eval
assert 'openai' not in sys.modules
assert 'google.genai' not in sys.modules
print('providers remain lazy')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "providers remain lazy"
