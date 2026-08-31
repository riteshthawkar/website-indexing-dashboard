from __future__ import annotations

import pytest

from pipeline.core.document_titles import (
    looks_like_opaque_title,
    resolve_document_title,
    title_from_source_file,
    title_from_source_url,
)
from scripts.prepare_multilingual_ab import _validate_candidate_title_quality


def test_resolves_truncated_artifact_ids_from_original_download_name():
    title = resolve_document_title(
        "1a58b097075809df3f7c",
        source_file=(
            "/crawl/downloads/"
            "MBZUAI_BRAND-GUIDELINES_MARCH_2026_V1_993eed6e3bf8.pdf"
        ),
    )

    assert title == "MBZUAI Brand Guidelines March 2026 V1"
    assert looks_like_opaque_title("1a58b097075809df3f7c") is True
    assert looks_like_opaque_title(title) is False


def test_uses_canonical_page_or_site_identity_for_web_sources():
    assert title_from_source_url(
        "https://preprod.mbzuai.ac.ae/admissions/graduate-phd-admissions"
    ) == "Graduate PhD Admissions"
    assert title_from_source_url("https://careers.mbzuai.ac.ae/") == "MBZUAI Careers"
    assert title_from_source_url("https://ifm.ai/") == "IFM"


def test_strips_only_the_crawl_digest_from_download_titles():
    assert title_from_source_file(
        "/downloads/MBZUAI_University_Catalogue_2021-22_08082021_3e4ac17553e4.pdf"
    ) == "MBZUAI University Catalogue 2021 22 08082021"


def test_candidate_title_gate_rejects_missing_and_truncated_hash_titles():
    with pytest.raises(RuntimeError, match=r"missing=1 opaque=1"):
        _validate_candidate_title_quality(
            [
                {"id": "chunk:missing", "kind": "chunk", "title": ""},
                {
                    "id": "chunk:opaque",
                    "kind": "chunk",
                    "title": "ee72c8444780dd009b50",
                },
                {"id": "chunk:ok", "kind": "chunk", "title": "Admissions"},
            ],
            config_id="c650",
        )


def test_candidate_title_gate_reports_a_passed_quality_contract():
    assert _validate_candidate_title_quality(
        [{"id": "chunk:ok", "kind": "chunk", "title": "Admissions"}],
        config_id="c650",
    ) == {
        "required_record_count": 1,
        "missing_title_count": 0,
        "opaque_title_count": 0,
        "passed": True,
    }
