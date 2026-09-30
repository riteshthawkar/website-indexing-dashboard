"""Behavioral contracts for tenant-neutral website ingestion profiles."""

from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

from lawa_rag import (
    DynamicCollection,
    IndexingPipeline,
    RobotsPolicy,
    WebsiteProfile,
    WebsiteProfileError,
    build_website_config,
)
from pipeline.core.config import production_indexing_contract_fingerprint
from pipeline.core.dynamic_collections import normalize_dynamic_collection_specs
from pipeline.core.orchestrator import PipelineOrchestrator, RunConfigMismatchError


def _static_profile() -> WebsiteProfile:
    return WebsiteProfile(
        site_id="example-docs",
        start_urls=("https://docs.example.com/",),
        allowed_hosts=("docs.example.com", "www.example.com"),
        priority_urls=("https://www.example.com/about",),
        sitemap_urls=("https://docs.example.com/sitemap.xml",),
        languages=("en", "ar"),
        excluded_path_prefixes=("/api/", "/preview/"),
    )


def _indexing_extra_available() -> bool:
    modules = ("crawl4ai", "docling", "markitdown", "datasketch", "trafilatura")
    return all(importlib.util.find_spec(name) is not None for name in modules)


def test_static_multihost_profile_compiles_without_tenant_assumptions() -> None:
    config = build_website_config(_static_profile())

    assert config["project_name"] == "example-docs"
    assert config["source_profile"] == {
        "schema_version": "source-profile/v1",
        "kind": "website",
        "source_id": "example-docs",
        "website_profile_version": "website-profile/v1",
        "allowed_hosts": ["docs.example.com", "www.example.com"],
        "languages": ["en", "ar"],
        "robots_policy": "respect",
        "robots_authorization_note": "",
    }
    assert config["crawler"]["allowed_hosts"] == [
        "docs.example.com",
        "www.example.com",
    ]
    assert config["crawler"]["robots_origins"] == [
        "https://docs.example.com",
        "https://www.example.com",
    ]
    assert config["crawler"]["headers"]["Accept-Language"] == "en,ar;q=0.9"
    assert config["quality"]["target_languages"] == ["en", "ar"]
    assert "mbzuai" not in json.dumps(config).lower()


def test_generic_stage_plan_contains_no_site_specific_plugin() -> None:
    config = build_website_config(_static_profile())
    plugins = [stage["plugin"] for stage in config["stages"]]

    assert plugins == [
        "crawl4ai",
        "quality_scorer",
        "trafilatura",
        "docling",
        "markitdown",
        "dedup_filter",
        "hybrid",
    ]
    assert all("mbzuai" not in plugin for plugin in plugins)


def test_profile_owned_security_boundaries_cannot_be_overridden() -> None:
    config = build_website_config(
        _static_profile(),
        overrides={
            "project_name": "wrong-tenant",
            "pipeline": {"production_profile": True, "immutable_run": False},
            "crawler": {
                "start_url": "https://attacker.example/",
                "allowed_hosts": ["attacker.example"],
                "respect_robots_txt": False,
            },
            "stages": [
                {"id": "tenant-rule", "type": "formatter", "plugin": "mbzuai_index_readiness"}
            ],
            "chunker": {"target_tokens": 512},
        },
    )

    assert config["project_name"] == "example-docs"
    assert config["pipeline"]["production_profile"] is False
    assert config["pipeline"]["immutable_run"] is True
    assert config["crawler"]["start_url"] == "https://docs.example.com/"
    assert config["crawler"]["allowed_hosts"] == [
        "docs.example.com",
        "www.example.com",
    ]
    assert config["crawler"]["respect_robots_txt"] is True
    assert config["stages"][0]["plugin"] == "crawl4ai"
    assert config["chunker"]["target_tokens"] == 512


def test_spa_collection_uses_existing_exhaustive_browser_contract() -> None:
    profile = WebsiteProfile(
        site_id="faculty-site",
        start_urls=("https://faculty.example.edu/",),
        languages=("en", "ar"),
        dynamic_collections=(
            DynamicCollection(
                id="faculty",
                url="/directory",
                item_selector="main article",
                item_url_pattern=r"^https://faculty\.example\.edu/people/[^/?#]+/?$",
                minimum_items=10,
                max_states=20,
                augment_listing_page=True,
            ),
        ),
    )
    crawler = build_website_config(profile)["crawler"]
    specs, errors = normalize_dynamic_collection_specs(
        crawler["dynamic_collections"],
        start_url=crawler["start_url"],
        allowed_hosts=crawler["allowed_hosts"],
    )

    assert errors == []
    assert specs[0].url == "https://faculty.example.edu/directory"
    assert specs[0].minimum_items == 10
    assert crawler["require_complete_dynamic_collections"] is True


@pytest.mark.parametrize(
    "field",
    ["priority_urls", "sitemap_urls", "sitemap_origins"],
)
def test_profile_rejects_cross_host_discovery_urls(field: str) -> None:
    values = {field: ("https://outside.example/path",)}
    with pytest.raises(WebsiteProfileError, match="outside allowed_hosts"):
        WebsiteProfile(
            site_id="safe-site",
            start_urls=("https://safe.example/",),
            **values,
        )


def test_profile_rejects_cross_host_dynamic_collection() -> None:
    collection = DynamicCollection(
        id="people",
        url="https://outside.example/people",
        item_selector="article",
    )
    with pytest.raises(WebsiteProfileError, match="outside allowed_hosts"):
        WebsiteProfile(
            site_id="safe-site",
            start_urls=("https://safe.example/",),
            dynamic_collections=(collection,),
        )


def test_robots_override_requires_and_records_explicit_authorization() -> None:
    with pytest.raises(WebsiteProfileError, match="authorization note"):
        WebsiteProfile.from_url(
            "https://owner.example/",
            robots_policy=RobotsPolicy.AUTHORIZED_OVERRIDE,
        )

    profile = WebsiteProfile.from_url(
        "https://owner.example/",
        robots_policy=RobotsPolicy.AUTHORIZED_OVERRIDE,
        robots_authorization_note="Written authorization from the site owner.",
    )
    config = build_website_config(profile)

    assert config["crawler"]["respect_robots_txt"] is False
    assert config["crawler"]["robots_origins"] == []
    assert config["source_profile"]["robots_policy"] == "authorized_override"


def test_source_profile_is_part_of_the_immutable_indexing_identity() -> None:
    config = build_website_config(_static_profile())
    altered = deepcopy(config)
    altered["source_profile"]["source_id"] = "different-source"

    assert production_indexing_contract_fingerprint(config) != (
        production_indexing_contract_fingerprint(altered)
    )


def test_profile_detaches_caller_owned_sequences() -> None:
    starts = ["https://mutable.example/"]
    prefixes = ["/api/"]
    profile = WebsiteProfile(
        site_id="mutable-site",
        start_urls=starts,
        excluded_path_prefixes=prefixes,
    )
    starts.append("https://outside.example/")
    prefixes.append("/private/")

    assert profile.start_urls == ("https://mutable.example/",)
    assert profile.excluded_path_prefixes == ("/api/",)


def test_generic_immutable_run_refuses_to_relabel_an_existing_directory(
    tmp_path: Path,
) -> None:
    work_dir = tmp_path / "existing"
    work_dir.mkdir()
    (work_dir / "pipeline_state.json").write_text("{}", encoding="utf-8")
    orchestrator = PipelineOrchestrator(
        build_website_config(_static_profile()),
        work_dir=work_dir,
    )

    with pytest.raises(RunConfigMismatchError, match="already initialized"):
        orchestrator._validate_fresh_production_run_directory()


@pytest.mark.skipif(
    not _indexing_extra_available(),
    reason="the optional indexing extra is not installed",
)
def test_real_generic_plugins_resolve_and_validate_without_writes(tmp_path: Path) -> None:
    work_dir = tmp_path / "must-not-exist"
    pipeline = IndexingPipeline.for_website(_static_profile(), work_dir=work_dir)

    assert [stage.plugin for stage in pipeline.plan_sync()][-2:] == [
        "dedup_filter",
        "hybrid",
    ]
    assert pipeline.validate_sync().ok
    assert not work_dir.exists()
