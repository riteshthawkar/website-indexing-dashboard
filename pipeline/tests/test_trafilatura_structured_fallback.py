import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

from pipeline.core.base import StageContext, StageStatus
from pipeline.core.io import load_json_safe
from pipeline.stages.cleaners.trafilatura_cleaner import (
    TrafilaturaCleaner,
    _structured_fallback_comparison,
)


def _words(label: str, count: int) -> str:
    return " ".join(f"{label}{index}" for index in range(count))


def test_structured_fallback_recovers_material_missing_records():
    primary = "<main><h1>Leadership</h1><p>" + _words("intro", 45) + "</p></main>"
    fallback = (
        "<main><h1>Leadership</h1><p>"
        + _words("intro", 45)
        + "</p>"
        + "".join(
            f"<h2>Team {index}</h2><p>{_words(f'person{index}', 30)}</p>"
            for index in range(1, 5)
        )
        + "</main>"
    )

    result = _structured_fallback_comparison(primary, fallback)

    assert result["prefer_fallback"] is True
    assert result["word_gain"] >= 80
    assert result["missing_heading_count"] == 4


def test_structured_fallback_does_not_replace_article_for_text_growth_alone():
    primary = "<article><h1>Research story</h1><p>" + _words("body", 120) + "</p></article>"
    fallback = (
        "<main><h1>Research story</h1><p>"
        + _words("body", 120)
        + "</p><h2>Related links</h2><p>"
        + _words("link", 120)
        + "</p></main>"
    )

    result = _structured_fallback_comparison(primary, fallback)

    assert result["word_gain"] >= 80
    assert result["prefer_fallback"] is False
    assert result["missing_heading_count"] == 1


def test_cleaner_selects_structurally_richer_main_content(tmp_path, monkeypatch):
    html_dir = tmp_path / "html"
    html_dir.mkdir()
    source = html_dir / "directory.html"
    primary = "<main><h1>Leadership</h1><p>" + _words("intro", 45) + "</p></main>"
    source.write_text(
        "<html><body><main><h1>Leadership</h1><p>"
        + _words("intro", 45)
        + "</p>"
        + "".join(
            f"<section><h2>Group {index}</h2><h3>Person {index}</h3>"
            f"<p>{_words(f'profile{index}', 30)}</p></section>"
            for index in range(1, 5)
        )
        + "</main></body></html>",
        encoding="utf-8",
    )
    monkeypatch.setitem(
        sys.modules,
        "trafilatura",
        SimpleNamespace(extract=lambda *args, **kwargs: primary),
    )
    ctx = StageContext(
        run_id="structured-fallback",
        project_name="test",
        config={"cleaner": {"min_content_length": 10}},
        work_dir=tmp_path,
        previous_outputs={"html_dir": str(html_dir)},
        stage_definition={"type": "cleaner", "plugin": "trafilatura"},
        stage_id="clean_html",
    )

    result = asyncio.run(TrafilaturaCleaner().execute(ctx))

    assert result.status == StageStatus.COMPLETED
    assert result.metrics["fallback_compared"] == 1
    assert result.metrics["structurally_richer_fallbacks"] == 1
    cleaned = (Path(result.outputs["cleaned_dir"]) / source.name).read_text(
        encoding="utf-8"
    )
    assert "Person 4" in cleaned
    manifest = load_json_safe(result.outputs["cleaning_manifest_file"])
    disposition = manifest["dispositions"][0]
    assert disposition["selected_backend"] == "bs4_structured_fallback"
    assert disposition["reason_code"] == "accepted_bs4_structured_fallback"
    assert disposition["structured_fallback_comparison"]["prefer_fallback"] is True


def test_structured_fallback_configuration_is_validated():
    errors = asyncio.run(
        TrafilaturaCleaner().validate_config(
            {
                "cleaner": {
                    "compare_bs4_when_structurally_richer": "yes",
                    "structured_fallback_min_word_gain": -1,
                    "structured_fallback_min_word_ratio": 0.5,
                }
            }
        )
    )

    assert "cleaner.compare_bs4_when_structurally_richer must be a boolean" in errors
    assert "cleaner.structured_fallback_min_word_gain must be a non-negative integer" in errors
    assert "cleaner.structured_fallback_min_word_ratio must be a number >= 1" in errors
