from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.core.chunking import (
    ChunkLimitExceededError,
    estimate_token_count,
    stable_document_id,
    token_counting_method,
    window_text_to_token_budget,
)
from pipeline.stages.chunkers.common import (
    fixed_window_chunks,
    hybrid_markdown_chunks,
    split_text_by_budget,
)
from pipeline.stages.formatters.pinecone_formatter import _split_text_into_chunks
from pipeline.stages.formatters.gemini_retrieval_formatter import (
    _classify_span_type,
    _sentence_spans,
    _span_signal_score,
)


def _source_info(name: str = "page.md") -> dict[str, object]:
    return {
        "path": Path("/tmp") / name,
        "source_url": f"https://example.com/{Path(name).stem}",
        "document_title": Path(name).stem,
        "document_type": "webpage",
    }


def test_unlimited_chunking_preserves_content_beyond_the_legacy_cap() -> None:
    text = " ".join(f"unique_term_{index}" for index in range(2_000))

    chunks = fixed_window_chunks(
        text,
        source_info=_source_info("large.md"),
        target_tokens=40,
        overlap_tokens=0,
        max_chunks_per_document=0,
    )

    assert len(chunks) > 64
    assert "unique_term_1999" in chunks[-1]["text"]
    assert all(chunk["token_count"] <= 40 for chunk in chunks)


def test_positive_chunk_limit_fails_instead_of_returning_partial_content() -> None:
    text = " ".join(f"unique_term_{index}" for index in range(300))

    with pytest.raises(ChunkLimitExceededError, match="No partial chunks were returned"):
        fixed_window_chunks(
            text,
            source_info=_source_info("guarded.md"),
            target_tokens=30,
            overlap_tokens=0,
            max_chunks_per_document=2,
        )


def test_legacy_formatter_fallback_is_also_lossless_or_fail_fast() -> None:
    text = " ".join(f"legacy_term_{index}" for index in range(2_000))

    chunks = _split_text_into_chunks(
        text,
        chunk_size_chars=500,
        chunk_overlap_chars=0,
        max_chunks=0,
    )
    assert len(chunks) > 64
    assert "legacy_term_1999" in chunks[-1]

    with pytest.raises(ChunkLimitExceededError, match="No partial chunks were returned"):
        _split_text_into_chunks(
            text,
            chunk_size_chars=500,
            chunk_overlap_chars=0,
            max_chunks=2,
        )


def test_multilingual_token_budget_counts_arabic_and_url_structure() -> None:
    arabic = " ".join(["تقدم جامعة محمد بن زايد برامج الذكاء الاصطناعي"] * 40)
    url_heavy = "https://mbzuai.ac.ae/ar/programs?degree=phd&track=machine-learning" * 12

    legacy_arabic_estimate = int(len(arabic.split()) * 1.33)
    assert estimate_token_count(arabic) > legacy_arabic_estimate
    assert estimate_token_count(url_heavy) > 12
    assert token_counting_method() in {
        "tiktoken:cl100k_base",
        "unicode_conservative_fallback:v1",
    }

    pieces = split_text_by_budget(arabic, max_tokens=80, overlap_tokens=10)
    assert len(pieces) > 1
    assert max(estimate_token_count(piece) for piece in pieces) <= 80
    assert "الاصطناعي" in pieces[-1]


def test_synopsis_windows_are_explicit_and_never_exceed_the_token_budget() -> None:
    text = " ".join(
        ["بداية المستند"]
        + [f"معلومة-{index}" for index in range(1_000)]
        + ["نهاية المستند"]
    )

    bounded = window_text_to_token_budget(text, max_tokens=240)

    assert estimate_token_count(bounded) <= 240
    assert "بداية المستند" in bounded
    assert "نهاية المستند" in bounded
    assert "content window omitted" in bounded


def test_top_level_sections_are_hard_chunk_boundaries() -> None:
    chunks = hybrid_markdown_chunks(
        "# Admissions\n\nAdmission details.\n\n# Research\n\nResearch details.",
        source_info=_source_info(),
        target_tokens=450,
        max_tokens=650,
        overlap_tokens=80,
        min_chunk_tokens=140,
        max_chunks_per_document=0,
    )

    assert [chunk["section_path"] for chunk in chunks] == [
        ["Admissions"],
        ["Research"],
    ]
    assert "Research" not in chunks[0]["text"]
    assert "Admissions" not in chunks[1]["text"]


def test_materialized_working_copy_preserves_immutable_markdown_identity() -> None:
    working_copy = Path("/tmp/current-run/corpus/markdown/materialized.md")
    prepared_source = Path("/tmp/prepared-run/corpus/markdown/original.md")
    source_info = {
        "path": working_copy,
        "source_markdown_path": str(prepared_source),
        "source_url": "",
        "source_file": "/tmp/downloads/source.pdf",
        "document_title": "Source PDF",
        "document_type": "pdf",
    }

    chunks = hybrid_markdown_chunks(
        "# Results\n\nGrounded result text.",
        source_info=source_info,
        target_tokens=650,
        max_tokens=900,
        overlap_tokens=100,
        min_chunk_tokens=160,
        max_chunks_per_document=0,
    )

    assert chunks
    assert {chunk["source_markdown_path"] for chunk in chunks} == {
        str(prepared_source)
    }
    assert {chunk["document_id"] for chunk in chunks} == {
        stable_document_id(
            str(prepared_source),
            "",
            "/tmp/downloads/source.pdf",
            "Source PDF",
        )
    }


def test_sibling_sections_are_hard_boundaries_with_exact_metadata() -> None:
    chunks = hybrid_markdown_chunks(
        (
            "# Programs\n\n"
            "## MSc\n\nMaster program details.\n\n"
            "## PhD\n\nDoctoral program details."
        ),
        source_info=_source_info("programs.md"),
        target_tokens=450,
        max_tokens=650,
        overlap_tokens=0,
        min_chunk_tokens=140,
        max_chunks_per_document=0,
    )

    assert len(chunks) == 2
    assert [chunk["section_path"] for chunk in chunks] == [
        ["Programs", "MSc"],
        ["Programs", "PhD"],
    ]
    assert "## PhD" not in chunks[0]["text"]
    assert "## MSc" not in chunks[1]["text"]


def test_same_level_headings_remain_siblings_when_h1_is_missing() -> None:
    chunks = hybrid_markdown_chunks(
        (
            "## MBZUAI Speakers\n\nSpeaker details.\n\n"
            "## RIKEN-AIP Speakers\n\nPartner speaker details."
        ),
        source_info=_source_info("workshop.md"),
        target_tokens=450,
        max_tokens=650,
        overlap_tokens=0,
        min_chunk_tokens=140,
        max_chunks_per_document=0,
    )

    assert [chunk["section_path"] for chunk in chunks] == [
        ["MBZUAI Speakers"],
        ["RIKEN-AIP Speakers"],
    ]
    assert "RIKEN-AIP" not in chunks[0]["text"]
    assert "MBZUAI Speakers" not in chunks[1]["text"]


def test_arabic_sentences_are_segmented_and_semantically_typed() -> None:
    text = (
        "آخر موعد للتقديم هو ٢٨ فبراير ٢٠٢٦. "
        "البرنامج ممول بالكامل ويشمل السكن والتأمين. "
        "للتواصل، استخدم البريد الإلكتروني admissions@example.ae. "
        "تشمل شروط القبول درجة البكالوريوس؟"
    )

    spans = _sentence_spans(text, max_sentences=1, max_chars=300)
    assert len(spans) == 4

    span_types = [
        _classify_span_type(
            span,
            document_title="القبول",
            section_path=["القبول"],
            heading="",
        )
        for span in spans
    ]
    assert span_types == ["deadline", "program", "contact", "requirement"]
    assert all(
        _span_signal_score(
            span,
            document_title="القبول",
            section_path=["القبول"],
            heading="",
        )
        > 0
        for span in spans
    )
