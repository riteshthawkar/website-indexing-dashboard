"""Fail-closed safety tests for generic website profiles."""

from __future__ import annotations

import importlib.util

import pytest

from lawa_rag import WebsiteProfile, WebsiteProfileError, build_website_config


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"start_urls": ("http://site.example/",)}, "must use HTTPS"),
        ({"languages": ("not a language",)}, "languages"),
        ({"fetch_concurrency": 0}, "between 1 and 64"),
        ({"max_pages": "100"}, "must be integers"),
        ({"extract_images": "yes"}, "must be booleans"),
        ({"excluded_path_prefixes": ("api",)}, "must start with"),
    ],
)
def test_invalid_profile_values_fail_before_pipeline_construction(
    options: dict[str, object], message: str
) -> None:
    values = {"start_urls": ("https://site.example/",), **options}
    with pytest.raises(WebsiteProfileError, match=message):
        WebsiteProfile(site_id="safe-site", **values)


def test_generic_config_rejects_embedded_credentials() -> None:
    profile = WebsiteProfile.from_url("https://private.example/")

    with pytest.raises(WebsiteProfileError, match="embedded secret"):
        build_website_config(
            profile,
            overrides={"crawler": {"headers": {"Authorization": "Bearer secret"}}},
        )


def test_completion_and_tls_guards_are_profile_owned() -> None:
    profile = WebsiteProfile.from_url("https://complete.example/")
    config = build_website_config(
        profile,
        overrides={
            "crawler": {
                "ignore_https_errors": True,
                "fail_on_empty_result": False,
                "fail_on_incomplete_frontier": False,
                "require_complete_priority_seeds": False,
                "legacy_site_content_heuristics": True,
            }
        },
    )

    assert config["crawler"]["ignore_https_errors"] is False
    assert config["crawler"]["fail_on_empty_result"] is True
    assert config["crawler"]["fail_on_incomplete_frontier"] is True
    assert config["crawler"]["require_complete_priority_seeds"] is True
    assert config["crawler"]["legacy_site_content_heuristics"] is False


@pytest.mark.skipif(
    importlib.util.find_spec("crawl4ai") is None,
    reason="the optional indexing extra is not installed",
)
def test_generic_profile_disables_legacy_site_content_rules() -> None:
    from pipeline.stages.crawlers.crawl4ai_crawler import (
        _clean_page_title_for_markdown,
        _html_quality_report,
    )

    html = (
        "<html><head><title>MBZUAI - Mohamed bin Zayed University of "
        "Artificial Intelligence</title></head><body>"
        + ("Unrelated rendered shell content. " * 20)
        + "</body></html>"
    )
    legacy = _html_quality_report(html, "https://site.example/program")
    generic = _html_quality_report(
        html,
        "https://site.example/program",
        legacy_site_heuristics=False,
    )

    assert "generic_mbzuai_title" in legacy["reasons"]
    assert "generic_mbzuai_title" not in generic["reasons"]
    assert _clean_page_title_for_markdown(
        "Research - MBZUAI", legacy_site_heuristics=False
    ) == "Research - MBZUAI"
    assert build_website_config(
        WebsiteProfile.from_url("https://site.example/")
    )["crawler"]["legacy_site_content_heuristics"] is False
