"""Compile website source profiles into tenant-neutral pipeline configs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping
from urllib.parse import urlsplit, urlunsplit

from pipeline.core.config import configured_secret_paths, merge_configs

from .config import load_pipeline_config
from .websites import (
    RobotsPolicy,
    WEBSITE_PROFILE_SCHEMA_VERSION,
    WebsiteProfile,
    WebsiteProfileError,
)


GENERIC_WEBSITE_CONFIG = "website_generic"


def _origin(value: str) -> str:
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _origins(profile: WebsiteProfile) -> list[str]:
    known = {_origin(value) for value in profile.start_urls}
    primary_scheme = urlsplit(profile.start_urls[0]).scheme
    for host in profile.allowed_hosts:
        known.add(f"{primary_scheme}://{host}")
    return sorted(known)


def _accept_language(languages: tuple[str, ...]) -> str:
    values = [languages[0]]
    for index, language in enumerate(languages[1:], start=1):
        quality = max(0.2, 1.0 - (index * 0.1))
        values.append(f"{language};q={quality:.1f}")
    return ",".join(values)


def _source_profile(profile: WebsiteProfile) -> dict[str, Any]:
    return {
        "schema_version": "source-profile/v1",
        "kind": "website",
        "source_id": profile.site_id,
        "website_profile_version": WEBSITE_PROFILE_SCHEMA_VERSION,
        "allowed_hosts": list(profile.allowed_hosts),
        "languages": list(profile.languages),
        "robots_policy": profile.robots_policy.value,
        "robots_authorization_note": profile.robots_authorization_note.strip(),
    }


def _crawler_config(profile: WebsiteProfile) -> dict[str, Any]:
    robots_respected = profile.robots_policy is RobotsPolicy.RESPECT
    sitemap_origins = profile.sitemap_origins or tuple(
        _origin(value) for value in profile.start_urls
    )
    return {
        "start_url": profile.start_urls[0],
        "allowed_domains": list(profile.allowed_hosts),
        "allowed_hosts": list(profile.allowed_hosts),
        "priority_seed_urls": list(
            dict.fromkeys((*profile.start_urls, *profile.priority_urls))
        ),
        "sitemap_origins": list(sitemap_origins),
        "sitemap_entry_urls": list(profile.sitemap_urls),
        "robots_origins": _origins(profile) if robots_respected else [],
        "respect_robots_txt": robots_respected,
        "robots_unknown_host_policy": "deny" if robots_respected else "allow",
        "robots_override_authorization": profile.robots_authorization_note.strip(),
        "require_https": profile.require_https,
        "ignore_https_errors": False,
        "fail_on_empty_result": True,
        "fail_on_incomplete_frontier": True,
        "require_complete_priority_seeds": True,
        "legacy_site_content_heuristics": False,
        "max_pages": profile.max_pages,
        "max_depth": profile.max_depth,
        "fetch_concurrency": profile.fetch_concurrency,
        "download_concurrency": profile.download_concurrency,
        "excluded_path_prefixes": list(profile.excluded_path_prefixes),
        "dynamic_collections": [
            asdict(value) for value in profile.dynamic_collections
        ],
        "require_complete_dynamic_collections": bool(profile.dynamic_collections),
        "extract_images": profile.extract_images,
        "download_page_images": profile.download_images,
        "extract_videos": profile.extract_videos,
        "headers": {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": _accept_language(profile.languages),
            "X-Crawler-Name": "LawaRAGWebsiteIndexer",
        },
    }


def _profile_overlay(
    profile: WebsiteProfile, *, canonical_stages: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "project_name": profile.site_id,
        "source_profile": _source_profile(profile),
        "pipeline": {
            "production_profile": False,
            "require_production_preflight": False,
            "immutable_run": True,
        },
        "stages": deepcopy(canonical_stages),
        "crawler": _crawler_config(profile),
        "quality": {"target_languages": list(profile.languages)},
    }


def build_website_config(
    profile: WebsiteProfile,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detached config while protecting source and stage boundaries.

    Advanced users can tune conversion, quality, chunking, and operational
    settings. Website identity, egress policy, robots policy, and the canonical
    generic stage plan always come from the validated profile.
    """

    if not isinstance(profile, WebsiteProfile):
        raise TypeError("profile must be a WebsiteProfile")
    base = load_pipeline_config(GENERIC_WEBSITE_CONFIG)
    canonical_stages = list(base.get("stages") or [])
    caller = deepcopy(dict(overrides or {}))
    config = merge_configs(
        base,
        caller,
        _profile_overlay(profile, canonical_stages=canonical_stages),
    )
    secret_paths = configured_secret_paths(config)
    if secret_paths:
        raise WebsiteProfileError(
            "Website configs must reference credentials through environment-backed "
            f"adapters; embedded secret fields found at: {', '.join(secret_paths)}"
        )
    return config


__all__ = ["GENERIC_WEBSITE_CONFIG", "build_website_config"]
