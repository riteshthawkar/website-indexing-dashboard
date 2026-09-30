"""Typed browser traversal contracts for client-rendered collections."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ._website_validation import WebsiteProfileError


@dataclass(frozen=True)
class DynamicCollection:
    """Describe how to exhaust one paginated or lazy-loaded listing."""

    id: str
    url: str
    item_selector: str
    next_selector: str = 'button[aria-label="Next Page"]'
    action: str = "next"
    expected_count_pattern: str = (
        r"Showing\s+\d+\s*[-\u2013]\s*\d+\s+of\s+([\d,]+)\s+results"
    )
    require_expected_count: bool = True
    minimum_items: int = 1
    max_states: int = 250
    item_url_pattern: str = ""
    require_item_urls: bool = True
    augment_listing_page: bool = False

    def __post_init__(self) -> None:
        self._normalize_text_fields()
        self._validate_fields()
        self._validate_patterns()

    def _normalize_text_fields(self) -> None:
        for name in (
            "id", "url", "item_selector", "next_selector", "action",
            "expected_count_pattern", "item_url_pattern",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        object.__setattr__(self, "action", self.action.lower())

    def _validate_fields(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.id):
            raise WebsiteProfileError("Dynamic collection id is invalid")
        if not self.url.strip() or not self.item_selector.strip():
            raise WebsiteProfileError(
                "Dynamic collection URL and item selector are required"
            )
        if self.action not in {"next", "load_more", "scroll"}:
            raise WebsiteProfileError("Dynamic collection action is invalid")
        if self.action != "scroll" and not self.next_selector.strip():
            raise WebsiteProfileError("Dynamic collection next selector is required")
        integer_values = (self.minimum_items, self.max_states)
        if any(isinstance(value, bool) or not isinstance(value, int)
               for value in integer_values):
            raise WebsiteProfileError("Dynamic collection bounds must be integers")
        if self.minimum_items < 0 or self.max_states < 1:
            raise WebsiteProfileError("Dynamic collection bounds are invalid")
        flag_values = (
            self.require_expected_count,
            self.require_item_urls,
            self.augment_listing_page,
        )
        if any(not isinstance(value, bool) for value in flag_values):
            raise WebsiteProfileError("Dynamic collection flags must be booleans")

    def _validate_patterns(self) -> None:
        try:
            if re.compile(self.expected_count_pattern, re.IGNORECASE).groups != 1:
                raise WebsiteProfileError(
                    "Dynamic collection expected-count pattern needs one capture group"
                )
            if self.item_url_pattern:
                re.compile(self.item_url_pattern)
        except re.error as exc:
            raise WebsiteProfileError(
                f"Dynamic collection regex is invalid: {exc}"
            ) from exc


__all__ = ["DynamicCollection"]
