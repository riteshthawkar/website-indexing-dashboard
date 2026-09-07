import asyncio

from pipeline.core.dynamic_collections import (
    DynamicCollectionSpec,
    PlaywrightDynamicCollectionBrowser,
    PlaywrightDynamicCollectionPage,
    normalize_dynamic_collection_specs,
    walk_dynamic_collection,
)
from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler


class _FakeCollectionPage:
    def __init__(self, states, *, body_text, final_url="https://example.com/directory"):
        self.states = states
        self.current = 0
        self._body_text = body_text
        self.final_url = final_url
        self.response_status = 200
        self.closed = False

    async def load(self):
        return None

    async def body_text(self):
        return self._body_text

    async def collect_items(self):
        return self.states[self.current]

    async def advance(self, _before_signature):
        if self.current + 1 >= len(self.states):
            return False, "next_disabled"
        self.current += 1
        return True, "advanced"

    async def close(self):
        self.closed = True


class _FailingLoadCollectionPage(_FakeCollectionPage):
    async def load(self):
        raise RuntimeError("navigation failed")


class _FakeNextControl:
    def __init__(self):
        self.clicked = False

    async def is_visible(self):
        return True

    async def is_enabled(self):
        return True

    async def click(self, **_kwargs):
        self.clicked = True


class _FakeControls:
    def __init__(self, control):
        self.control = control

    async def count(self):
        return 1

    def nth(self, _index):
        return self.control


class _FakePlaywrightPage:
    def __init__(self):
        self.url = "https://example.com/directory"
        self.control = _FakeNextControl()

    def locator(self, _selector):
        return _FakeControls(self.control)


class _TransitioningPlaywrightCollectionPage(PlaywrightDynamicCollectionPage):
    def __init__(self, signatures, spec):
        self.fake_page = _FakePlaywrightPage()
        super().__init__(self.fake_page, spec, timeout_ms=2000)
        self.signatures = iter(signatures)

    async def _current_signature(self):
        return next(self.signatures)


def _item(slug):
    return {
        "text": slug.replace("-", " ").title(),
        "urls": [f"/profiles/{slug}"],
    }


def test_dynamic_collection_config_is_exact_host_and_fail_closed():
    specs, errors = normalize_dynamic_collection_specs(
        [
            {
                "id": "faculty",
                "url": "/directory",
                "item_selector": "main article",
                "expected_count_pattern": r"Showing \d+ of (\d+) results",
                "item_url_pattern": r"^https://example\.com/profiles/",
                "allowed_terminal_item_urls": [
                    "https://example.com/profiles/retired"
                ],
            }
        ],
        start_url="https://example.com/",
        allowed_hosts=["example.com"],
    )

    assert errors == []
    assert specs[0].url == "https://example.com/directory"
    assert specs[0].allowed_terminal_item_urls == (
        "https://example.com/profiles/retired",
    )

    _specs, errors = normalize_dynamic_collection_specs(
        [
            {
                "id": "faculty",
                "url": "https://untrusted.example/directory",
                "item_selector": "main article",
                "expected_count_pattern": r"(\d+)(\d+)",
            }
        ],
        start_url="https://example.com/",
        allowed_hosts=["example.com"],
    )

    assert any("allowed_hosts" in error for error in errors)
    assert any("exactly one capture group" in error for error in errors)


def test_dynamic_collection_walk_collects_every_paginated_item():
    spec = DynamicCollectionSpec(
        collection_id="faculty",
        url="https://example.com/directory",
        item_selector="main article",
        expected_count_pattern=r"Showing \d+-\d+ of (\d+) results",
        item_url_pattern=r"^https://example\.com/profiles/",
        max_states=10,
    )
    page = _FakeCollectionPage(
        [
            [_item("one"), _item("two")],
            [_item("three"), _item("four")],
            [_item("five")],
        ],
        body_text="Showing 1-2 of 5 results",
    )

    result = asyncio.run(walk_dynamic_collection(spec, page))

    assert result["complete"] is True
    assert result["expected_count"] == 5
    assert result["discovered_item_count"] == 5
    assert result["discovered_url_count"] == 5
    assert result["states_traversed"] == 3
    assert result["termination_reason"] == "expected_count_reached"
    assert page.closed is True


def test_dynamic_collection_walk_rejects_early_terminal_state():
    spec = DynamicCollectionSpec(
        collection_id="faculty",
        url="https://example.com/directory",
        item_selector="main article",
        expected_count_pattern=r"Showing \d+-\d+ of (\d+) results",
        item_url_pattern=r"^https://example\.com/profiles/",
        max_states=10,
    )
    page = _FakeCollectionPage(
        [[_item("one"), _item("two")]],
        body_text="Showing 1-2 of 5 results",
    )

    result = asyncio.run(walk_dynamic_collection(spec, page))

    assert result["complete"] is False
    assert result["termination_reason"] == "next_disabled"
    assert "does not equal expected 5" in " ".join(result["errors"])


def test_dynamic_collection_closes_page_when_initial_load_fails():
    spec = DynamicCollectionSpec(
        collection_id="faculty",
        url="https://example.com/directory",
        item_selector="main article",
    )
    page = _FailingLoadCollectionPage([], body_text="")

    try:
        asyncio.run(walk_dynamic_collection(spec, page))
    except RuntimeError as exc:
        assert str(exc) == "navigation failed"
    else:
        raise AssertionError("load failure should propagate")

    assert page.closed is True


def test_spa_advance_ignores_empty_loading_state_and_waits_for_stability():
    spec = DynamicCollectionSpec(
        collection_id="news",
        url="https://example.com/directory",
        item_selector="main article",
        state_change_timeout_sec=2,
        state_change_stability_sec=0.1,
    )
    page = _TransitioningPlaywrightCollectionPage(
        ["", "before", "after", "after"],
        spec,
    )

    advanced, reason = asyncio.run(page.advance("before"))

    assert advanced is True
    assert reason == "advanced"
    assert page.fake_page.control.clicked is True


def test_dynamic_collection_can_preserve_cards_without_detail_urls():
    spec = DynamicCollectionSpec(
        collection_id="startups",
        url="https://example.com/startups",
        item_selector="main article",
        expected_count_pattern=r"Showing \d+-\d+ of (\d+) results",
        require_item_urls=False,
    )
    page = _FakeCollectionPage(
        [[{"text": "Startup Alpha", "urls": []}]],
        body_text="Showing 1-1 of 1 results",
        final_url=spec.url,
    )

    result = asyncio.run(walk_dynamic_collection(spec, page))

    assert result["complete"] is True
    assert result["items_without_urls"] == 1
    assert result["items"][0]["text"] == "Startup Alpha"


def test_crawler_augments_listing_html_with_complete_collection():
    crawler = Crawl4AICrawler()
    crawler.stats = {}
    crawler.dynamic_collection_inventory = {
        "collections": [
            {
                "collection_id": "faculty_directory_en",
                "configured_url": "https://example.com/directory",
                "final_url": "https://example.com/directory",
                "augment_listing_page": True,
                "expected_count": 2,
                "discovered_item_count": 2,
                "states_traversed": 2,
                "items": [
                    {
                        "text": "Professor One",
                        "crawl_urls": ["https://example.com/profiles/one"],
                    },
                    {
                        "text": "Professor Two",
                        "crawl_urls": ["https://example.com/profiles/two"],
                    },
                ],
            }
        ]
    }

    augmented, changed = crawler._augment_dynamic_collection_html(
        "https://example.com/directory",
        "<html><body><main><h1>Directory</h1></main></body></html>",
    )

    assert changed is True
    assert "Collected 2 of 2 listed items" in augmented
    assert "https://example.com/profiles/one" in augmented
    assert "Professor Two" in augmented
    assert crawler.stats["dynamic_collection_pages_augmented"] == 1


def test_completion_gate_rejects_missing_or_failed_dynamic_urls():
    crawler = Crawl4AICrawler()
    crawler.config = {"require_complete_dynamic_collections": True}
    crawler.require_complete_dynamic_collections = True
    crawler.max_pages = 100
    crawler.crawl_state = {"pending": []}
    crawler.seed_inventory = {"urls": []}
    crawler.dynamic_collection_inventory = {
        "urls": [
            "https://example.com/profiles/captured",
            "https://example.com/profiles/failed",
            "https://example.com/profiles/unmapped",
        ]
    }
    crawler.url_mapping = {
        "https://example.com/profiles/captured": "/tmp/captured.html",
        "https://example.com/profiles/failed": "SKIPPED_HTTP_404",
    }
    crawler.stats = {}

    errors = crawler._crawl_completion_errors()

    assert len(errors) == 2
    assert "unmapped=1" in errors[0]
    assert "failed=1" in errors[1]
    assert crawler.stats["dynamic_collection_urls_unmapped"] == 1
    assert crawler.stats["dynamic_collection_urls_failed"] == 1


def test_completion_gate_reports_but_does_not_require_stale_collection_aliases():
    crawler = Crawl4AICrawler()
    crawler.config = {"require_complete_dynamic_collections": True}
    crawler.require_complete_dynamic_collections = True
    crawler.max_pages = 100
    crawler.crawl_state = {"pending": []}
    crawler.seed_inventory = {"urls": []}
    crawler.dynamic_collection_inventory = {
        "urls": [
            "https://example.com/directory",
            "https://example.com/stale-alias",
        ],
        "required_success_urls": ["https://example.com/directory"],
    }
    crawler.url_mapping = {
        "https://example.com/directory": "/tmp/directory.html",
        "https://example.com/stale-alias": "SKIPPED_HTTP_404",
    }
    crawler.stats = {}

    errors = crawler._crawl_completion_errors()

    assert errors == []
    assert crawler.stats["dynamic_collection_urls_unmapped"] == 0
    assert crawler.stats["dynamic_collection_urls_failed"] == 0


def test_browser_json_fetch_primes_same_origin_and_enforces_size_limit():
    class Response:
        status = 200

    class Page:
        def __init__(self):
            self.goto_calls = []
            self.evaluate_calls = []
            self.body = '{"rows": []}'

        async def goto(self, url, **kwargs):
            self.goto_calls.append((url, kwargs))
            return Response()

        async def evaluate(self, script, argument):
            self.evaluate_calls.append((script, argument))
            return {"status": 200, "body": self.body}

    class Context:
        def __init__(self, page):
            self.page = page

        async def new_page(self):
            return self.page

    page = Page()
    browser = PlaywrightDynamicCollectionBrowser(headless=False, timeout_sec=5)
    browser._context = Context(page)

    status, payload = asyncio.run(
        browser.fetch_json(
            "https://example.com/api/items?page=1",
            context_url="https://example.com/items",
            max_bytes=1024,
        )
    )

    assert status == 200
    assert payload == b'{"rows": []}'
    assert page.goto_calls[0][0] == "https://example.com/items"
    assert page.evaluate_calls[0][1] == {
        "url": "https://example.com/api/items?page=1"
    }

    page.body = "payload-is-too-large"
    try:
        asyncio.run(
            browser.fetch_json(
                "https://example.com/api/items?page=2",
                context_url="https://example.com/items",
                max_bytes=5,
            )
        )
    except ValueError as exc:
        assert "exceeds configured size limit" in str(exc)
    else:
        raise AssertionError("oversized browser JSON response must be rejected")

    try:
        asyncio.run(
            browser.fetch_json(
                "https://outside.example/api/items",
                context_url="https://example.com/items",
                max_bytes=1024,
            )
        )
    except ValueError as exc:
        assert "same-origin" in str(exc)
    else:
        raise AssertionError("cross-origin browser JSON fetch must be rejected")
