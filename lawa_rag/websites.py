"""Validated source profiles for website ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any
from urllib.parse import urljoin, urlsplit

from ._website_validation import (
    WebsiteProfileError,
    dedupe,
    normalize_host,
    normalize_site_id,
    normalize_url,
    normalized_languages,
    normalized_path_prefixes,
    normalized_urls,
    validate_allowed_urls,
)
from .website_collections import DynamicCollection


WEBSITE_PROFILE_SCHEMA_VERSION = "website-profile/v1"


class RobotsPolicy(str, Enum):
    """How the crawler treats robots directives for this source."""

    RESPECT = "respect"
    AUTHORIZED_OVERRIDE = "authorized_override"


@dataclass(frozen=True)
class WebsiteProfile:
    """Tenant-neutral website crawl policy compiled into an indexing pipeline."""

    site_id: str
    start_urls: tuple[str, ...]
    allowed_hosts: tuple[str, ...] = ()
    languages: tuple[str, ...] = ("en",)
    priority_urls: tuple[str, ...] = ()
    sitemap_urls: tuple[str, ...] = ()
    sitemap_origins: tuple[str, ...] = ()
    excluded_path_prefixes: tuple[str, ...] = ()
    dynamic_collections: tuple[DynamicCollection, ...] = ()
    robots_policy: RobotsPolicy = RobotsPolicy.RESPECT
    robots_authorization_note: str = ""
    require_https: bool = True
    max_pages: int = 5000
    max_depth: int = 8
    fetch_concurrency: int = 5
    download_concurrency: int = 8
    extract_images: bool = True
    download_images: bool = False
    extract_videos: bool = True

    def __post_init__(self) -> None:
        self._validate_scalar_types()
        site_id = normalize_site_id(self.site_id)
        authorization_note = str(self.robots_authorization_note or "").strip()
        policy = self._normalized_robots_policy()
        starts = normalized_urls(
            tuple(self.start_urls),
            label="start_urls",
            require_https=self.require_https,
        )
        if not starts:
            raise WebsiteProfileError("At least one start URL is required")
        hosts = self._normalized_allowed_hosts(starts)
        priority = normalized_urls(
            tuple(self.priority_urls),
            label="priority_urls",
            require_https=self.require_https,
        )
        sitemaps = normalized_urls(
            tuple(self.sitemap_urls),
            label="sitemap_urls",
            require_https=self.require_https,
        )
        origins = normalized_urls(
            tuple(self.sitemap_origins),
            label="sitemap_origins",
            require_https=self.require_https,
        )
        collections = self._normalized_collections(starts[0])
        exclusions = normalized_path_prefixes(tuple(self.excluded_path_prefixes))
        self._validate_values(
            hosts, starts, priority, sitemaps, origins, collections,
            policy, authorization_note,
        )
        normalized = (
            site_id, starts, hosts, normalized_languages(tuple(self.languages)),
            priority, sitemaps, origins, exclusions, collections, policy,
            authorization_note,
        )
        self._apply_normalized_values(normalized)

    def _apply_normalized_values(self, values: tuple[Any, ...]) -> None:
        names = (
            "site_id", "start_urls", "allowed_hosts", "languages",
            "priority_urls", "sitemap_urls", "sitemap_origins",
            "excluded_path_prefixes", "dynamic_collections", "robots_policy",
            "robots_authorization_note",
        )
        for name, value in zip(names, values, strict=True):
            object.__setattr__(self, name, value)

    @classmethod
    def from_url(cls, url: str, *, site_id: str | None = None,
                 **options: Any) -> "WebsiteProfile":
        host = urlsplit(str(url)).hostname or "website"
        derived = re.sub(r"[^a-z0-9]+", "-", host.lower()).strip("-")[:63]
        return cls(site_id=site_id or derived, start_urls=(url,), **options)

    def _normalized_robots_policy(self) -> RobotsPolicy:
        try:
            return RobotsPolicy(self.robots_policy)
        except ValueError as exc:
            raise WebsiteProfileError("robots_policy is invalid") from exc

    def _validate_scalar_types(self) -> None:
        integers = (
            self.max_pages,
            self.max_depth,
            self.fetch_concurrency,
            self.download_concurrency,
        )
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in integers):
            raise WebsiteProfileError("Crawler limits must be integers")
        flags = (
            self.require_https,
            self.extract_images,
            self.download_images,
            self.extract_videos,
        )
        if any(not isinstance(value, bool) for value in flags):
            raise WebsiteProfileError("Website profile flags must be booleans")

    def _normalized_allowed_hosts(self, starts: tuple[str, ...]) -> tuple[str, ...]:
        configured = tuple(
            normalize_host(value) for value in self.allowed_hosts
        )
        derived = tuple((urlsplit(value).hostname or "").lower() for value in starts)
        return dedupe(configured or derived)

    def _normalized_collections(self, base_url: str) -> tuple[DynamicCollection, ...]:
        collections = tuple(self.dynamic_collections)
        if any(not isinstance(value, DynamicCollection) for value in collections):
            raise WebsiteProfileError(
                "dynamic_collections must contain DynamicCollection values"
            )
        return tuple(
            replace(
                value,
                url=normalize_url(
                    urljoin(base_url, value.url),
                    label=f"dynamic_collections[{value.id}].url",
                    require_https=self.require_https,
                ),
            )
            for value in collections
        )

    def _validate_values(
        self,
        hosts: tuple[str, ...],
        starts: tuple[str, ...],
        priority: tuple[str, ...],
        sitemaps: tuple[str, ...],
        origins: tuple[str, ...],
        collections: tuple[DynamicCollection, ...],
        policy: RobotsPolicy,
        authorization_note: str,
    ) -> None:
        groups = (
            (starts, "start"),
            (priority, "priority"),
            (sitemaps, "sitemap"),
            (origins, "sitemap origin"),
            (tuple(value.url for value in collections), "collection"),
        )
        for values, label in groups:
            validate_allowed_urls(values, hosts, label=label)
        if len({value.id for value in collections}) != len(collections):
            raise WebsiteProfileError("Dynamic collection ids must be unique")
        if (
            policy is RobotsPolicy.AUTHORIZED_OVERRIDE
            and not authorization_note
        ):
            raise WebsiteProfileError(
                "A robots authorization note is required for an override"
            )
        if self.max_pages < 1 or self.max_depth < 0:
            raise WebsiteProfileError("max_pages must be positive and max_depth non-negative")
        valid_fetch = 1 <= self.fetch_concurrency <= 64
        valid_download = 1 <= self.download_concurrency <= 64
        if not valid_fetch or not valid_download:
            raise WebsiteProfileError("Crawler concurrency must be between 1 and 64")


__all__ = [
    "DynamicCollection", "RobotsPolicy", "WEBSITE_PROFILE_SCHEMA_VERSION",
    "WebsiteProfile", "WebsiteProfileError",
]
