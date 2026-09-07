from __future__ import annotations

import pytest

from pipeline.core.config import configured_secret_paths, sanitized_config_snapshot
from pipeline.core.request_headers import (
    request_header_config_errors,
    resolve_request_headers,
)
from pipeline.stages.crawlers.crawl4ai_crawler import (
    _browser_cookie_records,
    _parse_environment_cookie_header,
)


_BINDINGS = [
    {
        "name": "CF-Access-Client-Id",
        "env": "CLOUDFLARE_ACCESS_CLIENT_ID",
        "required": True,
    },
    {
        "name": "CF-Access-Client-Secret",
        "env": "CLOUDFLARE_ACCESS_CLIENT_SECRET",
        "required": True,
    },
]


def test_resolves_environment_headers_without_mutating_static_config() -> None:
    static = {"Accept": "text/html", "X-Crawler-Name": "indexer"}
    environment = {
        "CLOUDFLARE_ACCESS_CLIENT_ID": "client-id",
        "CLOUDFLARE_ACCESS_CLIENT_SECRET": "top-secret",
    }

    resolved = resolve_request_headers(static, _BINDINGS, environ=environment)

    assert resolved["CF-Access-Client-Id"] == "client-id"
    assert resolved["CF-Access-Client-Secret"] == "top-secret"
    assert static == {"Accept": "text/html", "X-Crawler-Name": "indexer"}


def test_required_environment_header_fails_closed_without_exposing_values() -> None:
    errors = request_header_config_errors({}, _BINDINGS, environ={})

    assert errors == [
        "Required crawler credential environment variable is missing: "
        "CLOUDFLARE_ACCESS_CLIENT_ID",
        "Required crawler credential environment variable is missing: "
        "CLOUDFLARE_ACCESS_CLIENT_SECRET",
    ]


@pytest.mark.parametrize(
    "bindings",
    [
        [{"name": "Host", "env": "SITE_HOST", "required": True}],
        [{"name": "X-Test\nInjected", "env": "SITE_TOKEN", "required": True}],
        [{"name": "X-Test", "env": "lowercase", "required": True}],
        [{"name": "X-Test", "env": "SITE_TOKEN", "required": "yes"}],
    ],
)
def test_invalid_environment_header_contract_is_rejected(bindings) -> None:
    assert request_header_config_errors(
        {},
        bindings,
        environ={"SITE_HOST": "value", "SITE_TOKEN": "value"},
    )


def test_environment_binding_contract_contains_no_secret_value() -> None:
    config = {"crawler": {"request_header_env": _BINDINGS}}

    assert configured_secret_paths(config) == []
    assert sanitized_config_snapshot(config) == config


def test_credential_header_value_cannot_be_persisted_in_static_config() -> None:
    errors = request_header_config_errors(
        {"CF-Access-Client-Secret": "must-not-be-persisted"},
        [],
    )

    assert errors == [
        "crawler.headers credential header must use request_header_env: "
        "'CF-Access-Client-Secret'"
    ]


def test_cookie_credential_is_allowed_only_through_environment_binding() -> None:
    binding = [
        {
            "name": "Cookie",
            "env": "CLOUDFLARE_ACCESS_COOKIE",
            "required": True,
        }
    ]
    environment = {
        "CLOUDFLARE_ACCESS_COOKIE": "CF_Authorization=short-lived-jwt"
    }

    assert request_header_config_errors(
        {},
        binding,
        environ=environment,
    ) == []
    assert resolve_request_headers(
        {},
        binding,
        environ=environment,
    ) == {"Cookie": "CF_Authorization=short-lived-jwt"}
    assert request_header_config_errors(
        {"Cookie": "CF_Authorization=must-not-be-persisted"},
        [],
    ) == [
        "crawler.headers credential header must use request_header_env: 'Cookie'"
    ]


def test_environment_cookie_is_parsed_and_scoped_to_crawl_origin() -> None:
    parsed = _parse_environment_cookie_header(
        "CF_Authorization=short-lived-jwt; session_hint=opaque"
    )

    assert parsed == {
        "CF_Authorization": "short-lived-jwt",
        "session_hint": "opaque",
    }
    assert _browser_cookie_records(
        parsed,
        "https://preprod.mbzuai.ac.ae/start",
    ) == [
        {
            "name": "CF_Authorization",
            "value": "short-lived-jwt",
            "url": "https://preprod.mbzuai.ac.ae/",
        },
        {
            "name": "session_hint",
            "value": "opaque",
            "url": "https://preprod.mbzuai.ac.ae/",
        },
    ]


def test_malformed_environment_cookie_fails_closed() -> None:
    with pytest.raises(ValueError, match="credential cookie is malformed"):
        _parse_environment_cookie_header("not-a-cookie")
