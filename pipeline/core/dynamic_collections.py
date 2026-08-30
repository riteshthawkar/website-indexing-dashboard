"""Deterministic discovery for client-rendered paginated collections.

The ordinary crawler sees only the DOM state that exists when a URL finishes
loading.  This module walks explicitly configured pagination controls before
the main crawl starts, records every visible collection item, and returns the
detail URLs that must be admitted to the crawl frontier.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence
from urllib.parse import urljoin, urlparse


DEFAULT_EXPECTED_COUNT_PATTERN = (
    r"Showing\s+\d+\s*[-\u2013]\s*\d+\s+of\s+([\d,]+)\s+results"
)


def _compact_text(value: Any, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit].rstrip()


def _stable_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DynamicCollectionSpec:
    """Validated browser-walk contract for one dynamic listing."""

    collection_id: str
    url: str
    item_selector: str
    next_selector: str = 'button[aria-label="Next Page"]'
    action: str = "next"
    expected_count_pattern: str = DEFAULT_EXPECTED_COUNT_PATTERN
    require_expected_count: bool = True
    required: bool = True
    minimum_items: int = 1
    max_states: int = 250
    wait_after_load_sec: float = 1.0
    state_change_timeout_sec: float = 12.0
    state_change_stability_sec: float = 0.45
    item_url_pattern: str = ""
    require_item_urls: bool = True
    require_successful_item_urls: bool = True
    allowed_terminal_item_urls: tuple[str, ...] = ()
    max_item_text_chars: int = 2000
    augment_listing_page: bool = False

    def fingerprint_payload(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_dynamic_collection_specs(
    raw_specs: Any,
    *,
    start_url: str,
    allowed_hosts: Sequence[str],
    require_https: bool = True,
) -> tuple[List[DynamicCollectionSpec], List[str]]:
    """Validate and normalize dynamic-collection configuration."""

    if raw_specs in (None, []):
        return [], []
    if not isinstance(raw_specs, list):
        return [], ["crawler.dynamic_collections must be a list"]

    normalized_hosts = {
        str(host or "").strip().lower().strip(".")
        for host in allowed_hosts
        if str(host or "").strip()
    }
    specs: List[DynamicCollectionSpec] = []
    errors: List[str] = []
    seen_ids: set[str] = set()

    for index, raw in enumerate(raw_specs):
        label = f"crawler.dynamic_collections[{index}]"
        if not isinstance(raw, Mapping):
            errors.append(f"{label} must be a mapping")
            continue

        collection_id = str(raw.get("id") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", collection_id):
            errors.append(f"{label}.id must contain only letters, numbers, '.', '_' or '-'")
        elif collection_id in seen_ids:
            errors.append(f"{label}.id duplicates {collection_id!r}")
        else:
            seen_ids.add(collection_id)

        url = urljoin(start_url, str(raw.get("url") or "").strip())
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            errors.append(f"{label}.url must be a valid HTTP(S) URL")
        elif require_https and parsed.scheme != "https":
            errors.append(f"{label}.url must use HTTPS")
        elif normalized_hosts and parsed.hostname.lower() not in normalized_hosts:
            errors.append(f"{label}.url must use crawler.allowed_hosts")

        item_selector = str(raw.get("item_selector") or "").strip()
        if not item_selector:
            errors.append(f"{label}.item_selector is required")

        next_selector = str(
            raw.get("next_selector") or 'button[aria-label="Next Page"]'
        ).strip()
        action = str(raw.get("action") or "next").strip().lower()
        if action not in {"next", "load_more", "scroll"}:
            errors.append(f"{label}.action must be next, load_more, or scroll")
        if action != "scroll" and not next_selector:
            errors.append(f"{label}.next_selector is required for {action}")

        expected_count_pattern = str(
            raw.get("expected_count_pattern") or DEFAULT_EXPECTED_COUNT_PATTERN
        )
        try:
            compiled_expected_count = re.compile(expected_count_pattern, re.IGNORECASE)
            if compiled_expected_count.groups != 1:
                errors.append(
                    f"{label}.expected_count_pattern must contain exactly one capture group"
                )
        except re.error as exc:
            errors.append(f"{label}.expected_count_pattern is invalid: {exc}")

        item_url_pattern = str(raw.get("item_url_pattern") or "").strip()
        compiled_item_url_pattern = None
        if item_url_pattern:
            try:
                compiled_item_url_pattern = re.compile(item_url_pattern)
            except re.error as exc:
                errors.append(f"{label}.item_url_pattern is invalid: {exc}")

        raw_allowed_terminal_urls = raw.get("allowed_terminal_item_urls") or []
        allowed_terminal_urls: List[str] = []
        if not isinstance(raw_allowed_terminal_urls, list):
            errors.append(f"{label}.allowed_terminal_item_urls must be a list")
            raw_allowed_terminal_urls = []
        for terminal_index, value in enumerate(raw_allowed_terminal_urls):
            terminal_label = f"{label}.allowed_terminal_item_urls[{terminal_index}]"
            terminal_url = urljoin(url, str(value or "").strip())
            terminal_parsed = urlparse(terminal_url)
            if (
                terminal_parsed.scheme not in {"http", "https"}
                or not terminal_parsed.hostname
            ):
                errors.append(f"{terminal_label} must be a valid HTTP(S) URL")
                continue
            if require_https and terminal_parsed.scheme != "https":
                errors.append(f"{terminal_label} must use HTTPS")
                continue
            if (
                normalized_hosts
                and terminal_parsed.hostname.lower() not in normalized_hosts
            ):
                errors.append(f"{terminal_label} must use crawler.allowed_hosts")
                continue
            terminal_url = terminal_parsed._replace(fragment="").geturl()
            if (
                compiled_item_url_pattern is not None
                and not compiled_item_url_pattern.search(terminal_url)
            ):
                errors.append(f"{terminal_label} must match item_url_pattern")
                continue
            if terminal_url not in allowed_terminal_urls:
                allowed_terminal_urls.append(terminal_url)
        if allowed_terminal_urls and not bool(
            raw.get("require_successful_item_urls", True)
        ):
            errors.append(
                f"{label}.allowed_terminal_item_urls requires "
                "require_successful_item_urls=true"
            )

        integer_fields = {
            "minimum_items": (raw.get("minimum_items", 1), 0),
            "max_states": (raw.get("max_states", 250), 1),
            "max_item_text_chars": (raw.get("max_item_text_chars", 2000), 40),
        }
        parsed_integers: Dict[str, int] = {}
        for field, (value, minimum) in integer_fields.items():
            try:
                parsed_value = int(value)
                if parsed_value < minimum:
                    raise ValueError
                parsed_integers[field] = parsed_value
            except (TypeError, ValueError):
                errors.append(f"{label}.{field} must be an integer >= {minimum}")
                parsed_integers[field] = max(minimum, 1)

        parsed_floats: Dict[str, float] = {}
        for field, default in (
            ("wait_after_load_sec", 1.0),
            ("state_change_timeout_sec", 12.0),
            ("state_change_stability_sec", 0.45),
        ):
            try:
                parsed_value = float(raw.get(field, default))
                if parsed_value < 0 or (field == "state_change_timeout_sec" and parsed_value <= 0):
                    raise ValueError
                parsed_floats[field] = parsed_value
            except (TypeError, ValueError):
                comparator = "> 0" if field == "state_change_timeout_sec" else ">= 0"
                errors.append(f"{label}.{field} must be {comparator}")
                parsed_floats[field] = default

        specs.append(
            DynamicCollectionSpec(
                collection_id=collection_id,
                url=url,
                item_selector=item_selector,
                next_selector=next_selector,
                action=action,
                expected_count_pattern=expected_count_pattern,
                require_expected_count=bool(raw.get("require_expected_count", True)),
                required=bool(raw.get("required", True)),
                minimum_items=parsed_integers["minimum_items"],
                max_states=parsed_integers["max_states"],
                wait_after_load_sec=parsed_floats["wait_after_load_sec"],
                state_change_timeout_sec=parsed_floats["state_change_timeout_sec"],
                state_change_stability_sec=parsed_floats[
                    "state_change_stability_sec"
                ],
                item_url_pattern=item_url_pattern,
                require_item_urls=bool(raw.get("require_item_urls", True)),
                require_successful_item_urls=bool(
                    raw.get("require_successful_item_urls", True)
                ),
                allowed_terminal_item_urls=tuple(allowed_terminal_urls),
                max_item_text_chars=parsed_integers["max_item_text_chars"],
                augment_listing_page=bool(raw.get("augment_listing_page", False)),
            )
        )

    return specs, errors


def dynamic_collection_specs_fingerprint(specs: Sequence[DynamicCollectionSpec]) -> str:
    return _stable_sha256([spec.fingerprint_payload() for spec in specs])


class DynamicCollectionPage(Protocol):
    """Minimal adapter used by the deterministic pagination walker."""

    final_url: str
    response_status: Optional[int]

    async def load(self) -> None: ...

    async def body_text(self) -> str: ...

    async def collect_items(self) -> List[Dict[str, Any]]: ...

    async def advance(self, before_signature: str) -> tuple[bool, str]: ...

    async def close(self) -> None: ...


def _normalize_items(
    raw_items: Sequence[Mapping[str, Any]],
    *,
    spec: DynamicCollectionSpec,
    base_url: str,
) -> List[Dict[str, Any]]:
    url_pattern = re.compile(spec.item_url_pattern) if spec.item_url_pattern else None
    normalized: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for raw in raw_items:
        text = _compact_text(raw.get("text"), limit=spec.max_item_text_chars)
        urls: List[str] = []
        for value in raw.get("urls") or []:
            absolute = urljoin(base_url, str(value or "").strip())
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname:
                continue
            if url_pattern and not url_pattern.search(absolute):
                continue
            absolute = parsed._replace(fragment="").geturl()
            if absolute not in urls:
                urls.append(absolute)

        identity = urls[0] if urls else _stable_sha256({"text": text})
        if not text and not urls:
            continue
        if identity in seen:
            continue
        seen.add(identity)
        normalized.append(
            {
                "item_id": identity,
                "text": text,
                "urls": urls,
            }
        )
    return normalized


def _items_signature(items: Sequence[Mapping[str, Any]]) -> str:
    return _stable_sha256([str(item.get("item_id") or "") for item in items])


async def walk_dynamic_collection(
    spec: DynamicCollectionSpec,
    page: DynamicCollectionPage,
    *,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Walk one collection until its expected item count or terminal control state."""

    started_at = time.time()
    items_by_id: Dict[str, Dict[str, Any]] = {}
    states: List[Dict[str, Any]] = []
    expected_count: Optional[int] = None
    termination_reason = "max_states_reached"

    try:
        await page.load()
        body_text = await page.body_text()
        expected_matches = re.findall(
            spec.expected_count_pattern,
            body_text,
            re.IGNORECASE,
        )
        expected_counts = {
            int(str(value).replace(",", ""))
            for value in expected_matches
            if str(value).replace(",", "").isdigit()
        }
        expected_count = (
            next(iter(expected_counts)) if len(expected_counts) == 1 else None
        )

        for state_index in range(1, spec.max_states + 1):
            state_items = _normalize_items(
                await page.collect_items(),
                spec=spec,
                base_url=page.final_url or spec.url,
            )
            state_signature = _items_signature(state_items)
            before_count = len(items_by_id)
            for item in state_items:
                items_by_id.setdefault(str(item["item_id"]), item)
            new_items = len(items_by_id) - before_count
            state_record = {
                "state": state_index,
                "visible_items": len(state_items),
                "new_items": new_items,
                "total_unique_items": len(items_by_id),
                "signature": state_signature,
                "first_item_id": str(state_items[0]["item_id"]) if state_items else "",
                "last_item_id": str(state_items[-1]["item_id"]) if state_items else "",
            }
            states.append(state_record)
            if progress_callback:
                progress_callback(
                    {
                        "collection_id": spec.collection_id,
                        "expected_count": expected_count,
                        **state_record,
                    }
                )

            if expected_count is not None and len(items_by_id) >= expected_count:
                termination_reason = "expected_count_reached"
                break

            advanced, reason = await page.advance(state_signature)
            if not advanced:
                termination_reason = reason or "next_unavailable"
                break
    finally:
        with contextlib.suppress(Exception):
            await page.close()

    discovered_count = len(items_by_id)
    errors: List[str] = []
    if page.response_status is not None and page.response_status >= 400:
        errors.append(f"collection returned HTTP {page.response_status}")
    if spec.require_expected_count and expected_count is None:
        errors.append("expected item count was not found or was ambiguous")
    if discovered_count < spec.minimum_items:
        errors.append(
            f"discovered item count {discovered_count} is below minimum {spec.minimum_items}"
        )
    if expected_count is not None and discovered_count != expected_count:
        errors.append(
            f"discovered item count {discovered_count} does not equal expected {expected_count}"
        )

    items = list(items_by_id.values())
    items_without_urls = sum(1 for item in items if not item.get("urls"))
    if spec.require_item_urls and items_without_urls:
        errors.append(f"{items_without_urls} discovered item(s) have no matching detail URL")
    urls = sorted({url for item in items for url in item.get("urls") or []})
    return {
        "version": 1,
        "collection_id": spec.collection_id,
        "configured_url": spec.url,
        "final_url": page.final_url or spec.url,
        "response_status": page.response_status,
        "required": spec.required,
        "require_successful_item_urls": spec.require_successful_item_urls,
        "allowed_terminal_item_urls": list(spec.allowed_terminal_item_urls),
        "augment_listing_page": spec.augment_listing_page,
        "expected_count": expected_count,
        "discovered_item_count": discovered_count,
        "discovered_url_count": len(urls),
        "items_without_urls": items_without_urls,
        "states_traversed": len(states),
        "termination_reason": termination_reason,
        "complete": not errors,
        "errors": errors,
        "states": states,
        "items": items,
        "urls": urls,
        "started_at": started_at,
        "finished_at": time.time(),
    }


class PlaywrightDynamicCollectionPage:
    """Playwright implementation of :class:`DynamicCollectionPage`."""

    def __init__(self, page: Any, spec: DynamicCollectionSpec, *, timeout_ms: int):
        self.page = page
        self.spec = spec
        self.timeout_ms = timeout_ms
        self.final_url = spec.url
        self.response_status: Optional[int] = None

    async def load(self) -> None:
        response = await self.page.goto(
            self.spec.url,
            wait_until="domcontentloaded",
            timeout=self.timeout_ms,
        )
        self.final_url = self.page.url
        self.response_status = response.status if response is not None else None
        await self.page.locator(self.spec.item_selector).first.wait_for(
            state="attached",
            timeout=self.timeout_ms,
        )
        if self.spec.wait_after_load_sec:
            await asyncio.sleep(self.spec.wait_after_load_sec)

    async def body_text(self) -> str:
        return await self.page.locator("body").inner_text(timeout=self.timeout_ms)

    async def collect_items(self) -> List[Dict[str, Any]]:
        return await self.page.locator(self.spec.item_selector).evaluate_all(
            """elements => elements.map(element => {
                const urls = [];
                const closest = element.closest('a[href]');
                if (closest && closest.href) urls.push(closest.href);
                for (const anchor of element.querySelectorAll('a[href]')) {
                    if (anchor.href && !urls.includes(anchor.href)) urls.push(anchor.href);
                }
                return {text: element.innerText || element.textContent || '', urls};
            })"""
        )

    async def _current_signature(self) -> str:
        normalized = _normalize_items(
            await self.collect_items(),
            spec=self.spec,
            base_url=self.page.url or self.final_url,
        )
        # SPA list components commonly remove their cards while the next state
        # is loading.  An empty DOM is transitional evidence, not a completed
        # state change.
        return _items_signature(normalized) if normalized else ""

    async def advance(self, before_signature: str) -> tuple[bool, str]:
        if self.spec.action == "scroll":
            previous_height = await self.page.evaluate("document.body.scrollHeight")
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            terminal_reason = "scroll_exhausted"
        else:
            controls = self.page.locator(self.spec.next_selector)
            active = None
            for index in range(await controls.count()):
                candidate = controls.nth(index)
                if await candidate.is_visible() and await candidate.is_enabled():
                    active = candidate
                    break
            if active is None:
                return False, "next_disabled"
            await active.click(timeout=self.timeout_ms)
            previous_height = None
            terminal_reason = "no_progress"

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.spec.state_change_timeout_sec
        candidate_signature = ""
        candidate_since: Optional[float] = None
        while loop.time() < deadline:
            await asyncio.sleep(0.15)
            current_signature = await self._current_signature()
            if current_signature and current_signature != before_signature:
                now = loop.time()
                if current_signature != candidate_signature:
                    candidate_signature = current_signature
                    candidate_since = now
                elif (
                    candidate_since is not None
                    and now - candidate_since >= self.spec.state_change_stability_sec
                ):
                    self.final_url = self.page.url or self.final_url
                    return True, "advanced"
            else:
                candidate_signature = ""
                candidate_since = None
            if self.spec.action == "scroll":
                current_height = await self.page.evaluate("document.body.scrollHeight")
                if current_height != previous_height:
                    previous_height = current_height

        return False, terminal_reason

    async def close(self) -> None:
        await self.page.close()


class PlaywrightDynamicCollectionBrowser:
    """Small, isolated browser used only for collection enumeration."""

    def __init__(
        self,
        *,
        headless: bool,
        timeout_sec: float,
        user_agent: str = "",
        headers: Optional[Mapping[str, Any]] = None,
        ignore_https_errors: bool = False,
        viewport: Optional[Mapping[str, Any]] = None,
        proxy: Any = None,
        storage_state: Any = None,
    ):
        self.headless = headless
        self.timeout_ms = max(1000, int(timeout_sec * 1000))
        self.user_agent = user_agent
        self.headers = {
            str(key): str(value)
            for key, value in (headers or {}).items()
            if str(key).strip() and value is not None
        }
        self.ignore_https_errors = ignore_https_errors
        self.viewport = dict(viewport or {}) or None
        self.proxy = proxy
        self.storage_state = storage_state
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None

    async def __aenter__(self) -> "PlaywrightDynamicCollectionBrowser":
        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - validated by the crawler
            raise RuntimeError(
                "playwright is required for crawler.dynamic_collections"
            ) from exc

        self._playwright = await async_playwright().start()
        launch_kwargs: Dict[str, Any] = {"headless": self.headless}
        if self.proxy:
            launch_kwargs["proxy"] = (
                dict(self.proxy)
                if isinstance(self.proxy, Mapping)
                else {"server": str(self.proxy)}
            )
        self._browser = await self._playwright.chromium.launch(**launch_kwargs)
        context_kwargs: Dict[str, Any] = {
            "ignore_https_errors": self.ignore_https_errors,
        }
        if self.user_agent:
            context_kwargs["user_agent"] = self.user_agent
        if self.headers:
            context_kwargs["extra_http_headers"] = self.headers
        if self.viewport:
            context_kwargs["viewport"] = self.viewport
        if self.storage_state:
            context_kwargs["storage_state"] = self.storage_state
        self._context = await self._browser.new_context(**context_kwargs)
        self._context.set_default_timeout(self.timeout_ms)
        return self

    async def discover(
        self,
        spec: DynamicCollectionSpec,
        *,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        if self._context is None:
            raise RuntimeError("dynamic collection browser is not open")
        page = await self._context.new_page()
        adapter = PlaywrightDynamicCollectionPage(
            page,
            spec,
            timeout_ms=self.timeout_ms,
        )
        return await walk_dynamic_collection(
            spec,
            adapter,
            progress_callback=progress_callback,
        )

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.close()
        if self._browser is not None:
            with contextlib.suppress(Exception):
                await self._browser.close()
        if self._playwright is not None:
            with contextlib.suppress(Exception):
                await self._playwright.stop()
