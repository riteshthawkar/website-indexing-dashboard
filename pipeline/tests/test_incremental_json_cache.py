from __future__ import annotations

import asyncio
import json
import time

import pytest

from pipeline.core.incremental_json_cache import IncrementalJsonObjectCache
from pipeline.core.base import StageContext
from pipeline.core.io import atomic_write_json, load_json_safe
from pipeline.stages.formatters.openai_assertion_extract_formatter import (
    OpenAIAssertionExtractFormatter,
)
from pipeline.stages.formatters.openai_assertion_validate_formatter import (
    OpenAIAssertionValidateFormatter,
    _group_assertions_by_slice,
)


def test_incremental_cache_replays_and_compacts(tmp_path):
    snapshot = tmp_path / "cache.json"
    atomic_write_json(snapshot, {"slice-1": {"value": 1}})

    cache = IncrementalJsonObjectCache(snapshot)
    cache.put("slice-2", {"value": 2})

    assert load_json_safe(snapshot) == {"slice-1": {"value": 1}}
    assert cache.journal_path.exists()

    resumed = IncrementalJsonObjectCache(snapshot)
    assert resumed.payload == {
        "slice-1": {"value": 1},
        "slice-2": {"value": 2},
    }

    resumed.compact()
    assert load_json_safe(snapshot) == resumed.payload
    assert not resumed.journal_path.exists()


def test_incremental_cache_ignores_an_incomplete_trailing_record(tmp_path):
    snapshot = tmp_path / "cache.json"
    cache = IncrementalJsonObjectCache(snapshot)
    cache.put("slice-1", {"value": 1})
    with cache.journal_path.open("a", encoding="utf-8") as handle:
        handle.write('{"key":"slice-2"')

    resumed = IncrementalJsonObjectCache(snapshot)
    assert resumed.payload == {"slice-1": {"value": 1}}
    resumed.put("slice-2", {"value": 2})
    assert IncrementalJsonObjectCache(snapshot).payload == {
        "slice-1": {"value": 1},
        "slice-2": {"value": 2},
    }


def test_incremental_cache_repairs_a_valid_record_without_final_newline(tmp_path):
    snapshot = tmp_path / "cache.json"
    cache = IncrementalJsonObjectCache(snapshot)
    cache.journal_path.write_text(
        json.dumps({"key": "slice-1", "value": {"value": 1}}),
        encoding="utf-8",
    )

    resumed = IncrementalJsonObjectCache(snapshot)
    resumed.put("slice-2", {"value": 2})
    assert IncrementalJsonObjectCache(snapshot).payload == {
        "slice-1": {"value": 1},
        "slice-2": {"value": 2},
    }


def test_incremental_cache_rejects_corrupt_complete_record(tmp_path):
    snapshot = tmp_path / "cache.json"
    cache = IncrementalJsonObjectCache(snapshot)
    cache.journal_path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="journal is corrupt"):
        IncrementalJsonObjectCache(snapshot)


def test_incremental_cache_uses_latest_value_for_duplicate_key(tmp_path):
    snapshot = tmp_path / "cache.json"
    cache = IncrementalJsonObjectCache(snapshot)
    cache.put("slice-1", {"value": 1})
    cache.put("slice-1", {"value": 2})

    records = [
        json.loads(line)
        for line in cache.journal_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 2
    assert IncrementalJsonObjectCache(snapshot).payload == {
        "slice-1": {"value": 2}
    }


def test_assertion_extraction_resumes_from_journal_after_failure(
    tmp_path,
    monkeypatch,
):
    slices_file = tmp_path / "extraction_slices.json"
    atomic_write_json(
        slices_file,
        [
            {
                "id": "slice-1",
                "document_id": "doc-1",
                "document_title": "First",
                "document_type": "webpage",
                "source_url": "https://mbzuai.ac.ae/first/",
                "linked_chunk_ids": ["chunk-1"],
                "text": "FIRST source text.",
                "authority_class": "canonical_page",
                "authority_score": 1.0,
            },
            {
                "id": "slice-2",
                "document_id": "doc-2",
                "document_title": "Second",
                "document_type": "webpage",
                "source_url": "https://mbzuai.ac.ae/second/",
                "linked_chunk_ids": ["chunk-2"],
                "text": "SECOND source text.",
                "authority_class": "canonical_page",
                "authority_score": 1.0,
            },
        ],
    )
    ctx = StageContext(
        run_id="resume-test",
        project_name="p",
        config={"assertions": {"extract_concurrency": 1}},
        work_dir=tmp_path,
        previous_outputs={"extraction_slices_file": str(slices_file)},
        stage_definition={"type": "formatter", "plugin": "openai_assertion_extract"},
        stage_id="extract_assertions_openai",
    )
    journal_path = (
        ctx.stage_work_dir
        / "openai_assertion_extract_cache.journal.jsonl"
    )

    import pipeline.stages.formatters.openai_assertion_extract_formatter as extract_mod

    def _flaky_completion(**kwargs):
        prompt = kwargs["user_prompt"]
        if "FIRST source text" in prompt:
            return {"entities": [], "assertions": [], "quality_flags": []}
        deadline = time.monotonic() + 2.0
        while not journal_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        raise RuntimeError("simulated provider failure")

    monkeypatch.setattr(extract_mod, "json_completion", _flaky_completion)
    with pytest.raises(RuntimeError, match="simulated provider failure"):
        asyncio.run(OpenAIAssertionExtractFormatter().execute(ctx))

    cache = IncrementalJsonObjectCache(
        ctx.stage_work_dir / "openai_assertion_extract_cache.json"
    )
    assert set(cache.payload) == {"slice-1"}

    resumed_prompts = []

    def _successful_completion(**kwargs):
        resumed_prompts.append(kwargs["user_prompt"])
        return {"entities": [], "assertions": [], "quality_flags": []}

    monkeypatch.setattr(extract_mod, "json_completion", _successful_completion)
    result = asyncio.run(OpenAIAssertionExtractFormatter().execute(ctx))

    assert result.status.value == "completed"
    assert len(resumed_prompts) == 1
    assert "SECOND source text" in resumed_prompts[0]
    assert not journal_path.exists()
    assert set(
        load_json_safe(
            ctx.stage_work_dir / "openai_assertion_extract_cache.json"
        )
    ) == {"slice-1", "slice-2"}


@pytest.mark.parametrize(
    ("formatter", "config_key", "module_name"),
    [
        (
            OpenAIAssertionExtractFormatter,
            "extract_concurrency",
            "pipeline.stages.formatters.openai_assertion_extract_formatter",
        ),
        (
            OpenAIAssertionValidateFormatter,
            "validate_concurrency",
            "pipeline.stages.formatters.openai_assertion_validate_formatter",
        ),
    ],
)
def test_assertion_stage_concurrency_is_bounded(
    formatter,
    config_key,
    module_name,
    monkeypatch,
):
    module = __import__(module_name, fromlist=["make_openai_client"])
    monkeypatch.setattr(module, "make_openai_client", lambda: object())

    errors = asyncio.run(
        formatter().validate_config(
            {"assertions": {config_key: 33}}
        )
    )

    assert errors == [f"assertions.{config_key} must be between 1 and 32"]


def test_assertion_grouping_prefers_exact_slice_and_indexes_chunk_fallback():
    slices = [
        {
            "id": "slice-1",
            "linked_chunk_ids": ["shared-chunk"],
            "text": "First source text.",
        },
        {
            "id": "slice-2",
            "linked_chunk_ids": ["shared-chunk", "second-chunk"],
            "text": "Second source text with the exact support span.",
        },
    ]
    assertions = [
        {
            "id": "exact",
            "source_slice_id": "slice-2",
            "source_chunk_ids": ["shared-chunk"],
        },
        {
            "id": "fallback",
            "source_chunk_ids": ["shared-chunk"],
            "support_span": "exact support span",
        },
    ]

    slices_by_id, grouped, unmapped = _group_assertions_by_slice(
        slices,
        assertions,
    )

    assert set(slices_by_id) == {"slice-1", "slice-2"}
    assert [item["id"] for item in grouped["slice-2"]] == [
        "exact",
        "fallback",
    ]
    assert unmapped == []


def test_assertion_grouping_reports_unmapped_records():
    _, grouped, unmapped = _group_assertions_by_slice(
        [{"id": "slice-1", "linked_chunk_ids": ["chunk-1"], "text": "Text"}],
        [{"id": "missing", "source_chunk_ids": ["unknown"]}],
    )

    assert grouped == {}
    assert unmapped == ["missing"]
