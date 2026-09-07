"""Environment-backed HTTP request-header configuration.

Crawler authentication values must never be written into YAML, resolved run
snapshots, logs, or release artifacts.  Configurations therefore name the
environment variables that supply sensitive header values at runtime.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Mapping, Sequence


_HEADER_NAME_RE = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_ENV_NAME_RE = re.compile(r"[A-Z_][A-Z0-9_]*")
_FORBIDDEN_HEADERS = {
    "connection",
    "content-length",
    "cookie",
    "host",
    "proxy-connection",
    "set-cookie",
    "transfer-encoding",
}
_ENVIRONMENT_ONLY_HEADERS = {
    "authorization",
    "cf-access-client-id",
    "cf-access-client-secret",
    "proxy-authorization",
    "x-api-key",
}


def request_header_config_errors(
    static_headers: Any,
    environment_bindings: Any,
    *,
    environ: Mapping[str, str] | None = None,
    require_values: bool = True,
) -> List[str]:
    """Return safe validation errors without exposing configured values."""

    errors: List[str] = []
    static_names: set[str] = set()
    if static_headers is not None and not isinstance(static_headers, Mapping):
        errors.append("crawler.headers must be a mapping")
    elif isinstance(static_headers, Mapping):
        for raw_name, raw_value in static_headers.items():
            name = str(raw_name or "").strip()
            normalized = name.casefold()
            if not _valid_header_name(name):
                errors.append(f"crawler.headers contains an invalid header name: {name!r}")
            elif normalized in static_names:
                errors.append(f"crawler.headers contains duplicate header: {name!r}")
            elif normalized in _FORBIDDEN_HEADERS:
                errors.append(f"crawler.headers cannot configure reserved header: {name!r}")
            elif _requires_environment_binding(normalized):
                errors.append(
                    f"crawler.headers credential header must use request_header_env: {name!r}"
                )
            static_names.add(normalized)
            if raw_value is None or _contains_newline(raw_value):
                errors.append(f"crawler.headers contains an invalid value for {name!r}")

    bindings = environment_bindings or []
    if not isinstance(bindings, Sequence) or isinstance(bindings, (str, bytes)):
        return [*errors, "crawler.request_header_env must be a list"]

    resolved_environment = os.environ if environ is None else environ
    bound_names: set[str] = set()
    for index, binding in enumerate(bindings):
        label = f"crawler.request_header_env[{index}]"
        if not isinstance(binding, Mapping):
            errors.append(f"{label} must be a mapping")
            continue
        name = str(binding.get("name") or "").strip()
        environment_name = str(binding.get("env") or "").strip()
        required = binding.get("required", True)
        normalized = name.casefold()
        if not _valid_header_name(name):
            errors.append(f"{label}.name is not a valid HTTP header name")
        elif normalized in _FORBIDDEN_HEADERS:
            errors.append(f"{label}.name is a reserved HTTP header")
        elif normalized in static_names or normalized in bound_names:
            errors.append(f"{label}.name duplicates another configured header")
        bound_names.add(normalized)
        if not _ENV_NAME_RE.fullmatch(environment_name):
            errors.append(f"{label}.env must be an uppercase environment variable name")
        if not isinstance(required, bool):
            errors.append(f"{label}.required must be a boolean")
        value = resolved_environment.get(environment_name, "")
        if require_values and required is True and not value:
            errors.append(f"Required crawler credential environment variable is missing: {environment_name}")
        elif value and _contains_newline(value):
            errors.append(f"Crawler credential environment variable contains a newline: {environment_name}")
    return errors


def resolve_request_headers(
    static_headers: Any,
    environment_bindings: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    """Resolve request headers while keeping credential values out of config."""

    errors = request_header_config_errors(
        static_headers,
        environment_bindings,
        environ=environ,
        require_values=True,
    )
    if errors:
        raise ValueError("; ".join(errors))
    resolved = {
        str(name).strip(): str(value)
        for name, value in (static_headers or {}).items()
    }
    source = os.environ if environ is None else environ
    for binding in environment_bindings or []:
        value = source.get(str(binding["env"]), "")
        if value:
            resolved[str(binding["name"]).strip()] = value
    return resolved


def _valid_header_name(value: str) -> bool:
    return bool(value and _HEADER_NAME_RE.fullmatch(value))


def _requires_environment_binding(normalized_name: str) -> bool:
    return (
        normalized_name in _ENVIRONMENT_ONLY_HEADERS
        or normalized_name.endswith(("-api-key", "-secret", "-token"))
    )


def _contains_newline(value: Any) -> bool:
    text = str(value)
    return "\r" in text or "\n" in text


__all__ = ["request_header_config_errors", "resolve_request_headers"]
