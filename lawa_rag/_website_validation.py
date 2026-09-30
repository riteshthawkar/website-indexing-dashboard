"""Small validation helpers shared by website profile contracts."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


_SITE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]{1,62}")
_LANGUAGE_RE = re.compile(r"[A-Za-z]{2,8}(?:-[A-Za-z0-9]{1,8})*")


class WebsiteProfileError(ValueError):
    """Raised when a website source profile is unsafe or incomplete."""


def normalize_site_id(value: Any) -> str:
    site_id = str(value or "").strip().lower()
    if not _SITE_ID_RE.fullmatch(site_id):
        raise WebsiteProfileError(
            "site_id must be 2-63 lowercase letters, numbers, '_' or '-'"
        )
    return site_id


def normalize_host(value: Any) -> str:
    raw = str(value or "").strip().lower().strip(".")
    if not raw or "://" in raw or any(char in raw for char in "/?#@"):
        raise WebsiteProfileError(f"Invalid exact hostname: {value!r}")
    try:
        host = (urlsplit(f"//{raw}").hostname or "").encode("idna").decode("ascii")
    except (UnicodeError, ValueError) as exc:
        raise WebsiteProfileError(f"Invalid exact hostname: {value!r}") from exc
    if not host or (":" in raw and host != raw):
        raise WebsiteProfileError(f"Invalid exact hostname: {value!r}")
    return host


def normalize_url(
    value: Any,
    *,
    label: str,
    require_https: bool,
) -> str:
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
        host = (parsed.hostname or "").encode("idna").decode("ascii")
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise WebsiteProfileError(f"{label} must be a valid HTTP(S) URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not host:
        raise WebsiteProfileError(f"{label} must be a valid HTTP(S) URL")
    if parsed.username or parsed.password:
        raise WebsiteProfileError(f"{label} must not contain embedded credentials")
    if require_https and parsed.scheme.lower() != "https":
        raise WebsiteProfileError(f"{label} must use HTTPS")
    display_host = f"[{host}]" if ":" in host else host
    netloc = f"{display_host}:{port}" if port is not None else display_host
    return urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, "")
    )


def dedupe(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def normalized_urls(
    values: tuple[str, ...],
    *,
    label: str,
    require_https: bool,
) -> tuple[str, ...]:
    return dedupe(
        tuple(
            normalize_url(
                value,
                label=f"{label}[{index}]",
                require_https=require_https,
            )
            for index, value in enumerate(values)
        )
    )


def validate_allowed_urls(
    values: tuple[str, ...],
    allowed_hosts: tuple[str, ...],
    *,
    label: str,
) -> None:
    allowed = set(allowed_hosts)
    for value in values:
        if (urlsplit(value).hostname or "").lower() not in allowed:
            raise WebsiteProfileError(
                f"{label} URL is outside allowed_hosts: {value}"
            )


def normalized_languages(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = dedupe(tuple(str(value).strip() for value in values))
    if not normalized or any(not _LANGUAGE_RE.fullmatch(value) for value in normalized):
        raise WebsiteProfileError("languages must contain valid BCP-47-like tags")
    return normalized


def normalized_path_prefixes(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = dedupe(tuple(str(value or "").strip() for value in values))
    if any(not value or not value.startswith("/") for value in normalized):
        raise WebsiteProfileError("Excluded path prefixes must start with '/'")
    return normalized
