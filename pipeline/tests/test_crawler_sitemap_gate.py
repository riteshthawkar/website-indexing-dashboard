import asyncio
from types import SimpleNamespace

from pipeline.core.base import StageContext, StageStatus
from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module


def _context(work_dir, *, minimum_sitemap_seed_count=3):
    return StageContext(
        run_id="sitemap-coverage-gate",
        project_name="test",
        config={
            "crawler": {
                "start_url": "https://example.com",
                "allowed_domains": ["example.com"],
                "max_pages": 10,
                "max_depth": 1,
                "fetch_concurrency": 1,
                "download_concurrency": 1,
                "timeout": 5,
                "sitemap_enabled": True,
                "sitemap_seed_limit": 10,
                "minimum_sitemap_seed_count": minimum_sitemap_seed_count,
                "extract_images": False,
                "extract_videos": False,
                "download_page_images": False,
                "fetch_video_transcripts": False,
                "respect_robots_txt": False,
                "fail_on_empty_result": True,
            },
            "converter": {"content_filter_threshold": 0.48},
        },
        work_dir=work_dir,
    )


def test_page_metadata_preserves_only_indexability_response_header():
    metadata = crawler_module._extract_page_metadata(
        "<html lang='en'><head><meta name='robots' content='index, follow'></head></html>",
        "https://preprod.mbzuai.ac.ae/about-us",
        response_headers={
            "Content-Type": "text/html",
            "Set-Cookie": "must-not-be-persisted=secret",
            "X-Robots-Tag": ["noindex, nofollow", "noimageindex, noarchive"],
        },
    )

    assert metadata["robots"] == "index, follow"
    assert metadata["robots_meta"] == "index, follow"
    assert metadata["robots_http"] == [
        "noindex, nofollow",
        "noimageindex, noarchive",
    ]
    assert metadata["x-robots-tag"] == metadata["robots_http"]
    assert "Content-Type" not in metadata
    assert "Set-Cookie" not in metadata


def test_sitemap_coverage_gate_fails_before_browser_start(tmp_path, monkeypatch):
    browser_started = False

    class BrowserMustNotStart:
        def __init__(self, config=None):
            nonlocal browser_started
            browser_started = True
            raise AssertionError("browser crawl must not start after a failed sitemap gate")

    async def fake_open_http_session(self):
        self._session = None

    async def fake_discover_sitemap_urls(self):
        self.discovered_sitemaps = {
            "sources": ["https://example.com/sitemap.xml"],
            "urls": ["https://example.com/a", "https://example.com/b"],
        }
        return list(self.discovered_sitemaps["urls"])

    monkeypatch.setattr(crawler_module, "AsyncWebCrawler", BrowserMustNotStart)
    monkeypatch.setattr(
        crawler_module.Crawl4AICrawler,
        "_open_http_session",
        fake_open_http_session,
    )
    monkeypatch.setattr(
        crawler_module.Crawl4AICrawler,
        "_discover_sitemap_urls",
        fake_discover_sitemap_urls,
    )

    result = asyncio.run(crawler_module.Crawl4AICrawler().execute(_context(tmp_path)))

    assert result.status == StageStatus.FAILED
    assert "discovered=2 required=3" in str(result.error_message)
    assert not browser_started


def test_sitemap_coverage_gate_configuration_is_fail_closed():
    crawler = crawler_module.Crawl4AICrawler()

    disabled_errors = asyncio.run(
        crawler.validate_config(
            {
                "crawler": {
                    "start_url": "https://example.com",
                    "sitemap_enabled": False,
                    "sitemap_seed_limit": 10,
                    "minimum_sitemap_seed_count": 3,
                }
            }
        )
    )
    over_limit_errors = asyncio.run(
        crawler.validate_config(
            {
                "crawler": {
                    "start_url": "https://example.com",
                    "sitemap_enabled": True,
                    "sitemap_seed_limit": 2,
                    "minimum_sitemap_seed_count": 3,
                }
            }
        )
    )

    assert any("sitemap_enabled must be true" in error for error in disabled_errors)
    assert any("must not exceed" in error for error in over_limit_errors)


def test_per_host_sitemap_coverage_cannot_be_masked_by_main_site_volume():
    crawler = crawler_module.Crawl4AICrawler()
    urls = [f"https://mbzuai.ac.ae/page-{index}" for index in range(100)]
    urls.extend(
        [f"https://careers.mbzuai.ac.ae/job-{index}" for index in range(2)]
    )

    try:
        crawler._enforce_host_minimums(
            urls,
            {"mbzuai.ac.ae": 50, "careers.mbzuai.ac.ae": 3},
            label="Sitemap origin coverage",
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("missing Careers coverage must fail the origin gate")

    assert "careers.mbzuai.ac.ae: found=2 required=3" in message


def test_bounded_link_discovery_stays_on_the_approved_source_host(monkeypatch):
    crawler = crawler_module.Crawl4AICrawler()
    crawler.start_url = "https://library.mbzuai.ac.ae/"
    crawler.allowed_domains = {"mbzuai.ac.ae"}
    crawler.allowed_hosts = {"library.mbzuai.ac.ae", "careers.mbzuai.ac.ae"}
    crawler.excluded_subdomains = set()
    crawler.excluded_path_prefixes = set()
    crawler.allow_query_urls = False
    crawler.allowed_query_param_names = set()
    crawler.require_https = True
    crawler.max_depth = 4
    crawler.max_pages = 20
    crawler.link_discovery_hosts = {"library.mbzuai.ac.ae"}
    crawler.link_discovery_max_pages_by_host = {"library.mbzuai.ac.ae": 3}
    source_url = "https://library.mbzuai.ac.ae"
    crawler.page_links = {
        source_url: [
            {"target_url": "https://library.mbzuai.ac.ae/user/login"},
            {"target_url": "https://library.mbzuai.ac.ae/services"},
            {"target_url": "https://library.mbzuai.ac.ae/research"},
            {"target_url": "https://library.mbzuai.ac.ae/third"},
            {"target_url": "https://careers.mbzuai.ac.ae/vacancies/"},
            {"target_url": "https://admin.hci.mbzuai.ac.ae/"},
        ]
    }
    monkeypatch.setattr(
        crawler_module,
        "_host_resolves_to_private_or_reserved",
        lambda _host: False,
    )
    monkeypatch.setattr(
        crawler,
        "_robots_allows_url",
        lambda url: not url.endswith("/user/login"),
    )

    discovered = crawler._discover_link_frontier_items(
        [source_url],
        visited=[source_url],
        pending=[],
        depths={source_url: 1},
    )

    assert [item["url"] for item in discovered] == [
        "https://library.mbzuai.ac.ae/services",
        "https://library.mbzuai.ac.ae/research",
    ]
    assert all(item["parent_url"] == source_url for item in discovered)


def test_no_sitemap_link_discovery_uses_bounded_seed_batches():
    crawler = crawler_module.Crawl4AICrawler()
    crawler.sitemap_batch_crawl = True
    crawler.sitemap_crawl_batch_size = 40
    crawler.crawl_state = {
        "pending": [{"url": "https://preprod.mbzuai.ac.ae/", "parent_url": None}]
    }
    crawler.stats = {"sitemap_urls_seeded": 0}
    crawler.config = {"sitemap_enabled": False}
    crawler.link_discovery_hosts = {"preprod.mbzuai.ac.ae"}

    assert crawler._should_use_seed_batch_crawl() is True

    crawler.link_discovery_hosts = set()
    assert crawler._should_use_seed_batch_crawl() is False


def test_json_seed_inventory_extracts_only_configured_result_items():
    payload = {
        "navigation": {"url": "/about-us"},
        "view": {
            "pager": {"totalPages": 3, "current": 0},
            "rows": [
                {
                    "element": "search-result",
                    "props": {"url": "/research/one"},
                },
                {
                    "element": "search-result",
                    "props": {"url": "/study/two"},
                },
            ],
        },
    }

    assert crawler_module._seed_inventory_total_pages(payload, "totalPages") == 3
    assert crawler_module._seed_inventory_total_pages(
        {"pager_info": {"totalItems": 228, "itemsPerPage": 12}},
        "totalPages",
        total_items_key="totalItems",
        items_per_page_key="itemsPerPage",
    ) == 19
    assert crawler_module._extract_seed_inventory_urls(
        payload,
        base_url="https://preprod.mbzuai.ac.ae/",
        url_keys=["url"],
        item_element="search-result",
    ) == [
        "https://preprod.mbzuai.ac.ae/research/one",
        "https://preprod.mbzuai.ac.ae/study/two",
    ]
    assert crawler_module._url_with_query_param(
        "https://preprod.mbzuai.ac.ae/api/drupal-ce/search?lang=en",
        "page",
        2,
    ) == "https://preprod.mbzuai.ac.ae/api/drupal-ce/search?lang=en&page=2"


def test_json_seed_inventory_captures_card_only_semantic_items():
    payload = {
        "pager_info": {"totalItems": 2, "itemsPerPage": 1},
        "rows": [
            {
                "element": "node-startup-listing-card",
                "props": {
                    "title": "Ortho AI",
                    "industryName": "Healthcare and pharma",
                    "stageName": "Building in beta",
                },
                "slots": {
                    "description": {
                        "processed": "<p>Clinical workflow platform.</p>"
                    }
                },
            }
        ],
    }

    assert crawler_module._extract_seed_inventory_items(
        payload,
        base_url="https://preprod.mbzuai.ac.ae/",
        item_element="node-startup-listing-card",
        url_keys=["url"],
        text_keys=["title", "industryName", "stageName", "processed"],
    ) == [
        {
            "item_id": crawler_module.hashlib.sha256(
                (
                    "Ortho AI | Healthcare and pharma | Building in beta | "
                    "Clinical workflow platform."
                ).encode("utf-8")
            ).hexdigest(),
            "text": (
                "Ortho AI | Healthcare and pharma | Building in beta | "
                "Clinical workflow platform."
            ),
            "urls": [],
        }
    ]


def test_seed_inventory_card_items_are_appended_to_saved_listing_html():
    crawler = crawler_module.Crawl4AICrawler()
    listing_url = "https://preprod.mbzuai.ac.ae/startups"
    crawler.seed_inventory = {
        "endpoints": [
            {
                "url": "https://preprod.mbzuai.ac.ae/api/startups",
                "augment_listing_url": listing_url,
                "expected_item_count": 2,
                "captured_item_count": 2,
                "fetched_pages": 2,
                "items": [
                    {"item_id": "one", "text": "Startup One", "urls": []},
                    {"item_id": "two", "text": "Startup Two", "urls": []},
                ],
            }
        ]
    }
    crawler.stats = {}

    augmented, changed = crawler._augment_seed_inventory_html(
        listing_url,
        "<html><body><main><h1>Startups</h1></main></body></html>",
    )

    assert changed is True
    assert "Collected 2 of 2 listed items across 2 API pages." in augmented
    assert "Startup One" in augmented
    assert "Startup Two" in augmented
    assert crawler.stats["seed_inventory_pages_augmented"] == 1


def test_completion_gate_rejects_pending_or_unmapped_inventory_urls():
    crawler = crawler_module.Crawl4AICrawler()
    crawler.config = {
        "fail_on_incomplete_frontier": True,
        "require_complete_seed_inventory": True,
    }
    crawler.max_pages = 100
    crawler.crawl_state = {
        "pending": [{"url": "https://preprod.mbzuai.ac.ae/pending"}]
    }
    crawler.seed_inventory = {
        "urls": [
            "https://preprod.mbzuai.ac.ae/captured",
            "https://preprod.mbzuai.ac.ae/unmapped",
        ]
    }
    crawler.url_mapping = {
        "https://preprod.mbzuai.ac.ae/captured": "/tmp/captured.html"
    }
    crawler.stats = {}

    errors = crawler._crawl_completion_errors()

    assert len(errors) == 2
    assert "pending=1" in errors[0]
    assert "unmapped=1" in errors[1]
    assert crawler.stats["frontier_pending_remaining"] == 1
    assert crawler.stats["seed_inventory_urls_unmapped"] == 1


def test_completion_gate_requires_successful_priority_seeds():
    crawler = crawler_module.Crawl4AICrawler()
    crawler.config = {"require_complete_priority_seeds": True}
    crawler.max_pages = 100
    crawler.crawl_state = {"pending": []}
    crawler.seed_inventory = {"urls": []}
    crawler.dynamic_collection_inventory = {"urls": []}
    crawler.priority_seed_urls = [
        "https://preprod.mbzuai.ac.ae/captured",
        "https://preprod.mbzuai.ac.ae/failed",
        "https://preprod.mbzuai.ac.ae/unmapped",
    ]
    crawler.url_mapping = {
        "https://preprod.mbzuai.ac.ae/captured": "/tmp/captured.html",
        "https://preprod.mbzuai.ac.ae/failed": "SKIPPED_HTTP_500",
    }
    crawler.stats = {}

    errors = crawler._crawl_completion_errors()

    assert len(errors) == 2
    assert "unmapped=1" in errors[0]
    assert "failed=1" in errors[1]
    assert crawler.stats["priority_seed_urls_unmapped"] == 1
    assert crawler.stats["priority_seed_urls_failed"] == 1


def test_completion_gate_requires_successful_seed_inventory_with_explicit_404_policy():
    allowed_404 = "https://preprod.mbzuai.ac.ae/annette-black-1"
    unexpected_500 = "https://preprod.mbzuai.ac.ae/programs/current"
    crawler = crawler_module.Crawl4AICrawler()
    crawler.config = {
        "require_complete_seed_inventory": True,
        "require_successful_seed_inventory": True,
        "seed_inventory_allowed_terminal_url_patterns": [
            r"^https://preprod\.mbzuai\.ac\.ae/annette-black(?:-[0-9]+)?/?$"
        ],
    }
    crawler.max_pages = 100
    crawler.crawl_state = {"pending": []}
    crawler.seed_inventory = {"urls": [allowed_404, unexpected_500]}
    crawler.url_mapping = {
        allowed_404: "SKIPPED_HTTP_404:browser_fetch_http_404",
        unexpected_500: "SKIPPED_HTTP_500:browser_fetch_http_500",
    }
    crawler.stats = {}

    errors = crawler._crawl_completion_errors()

    assert len(errors) == 1
    assert "not successfully captured" in errors[0]
    assert unexpected_500 in errors[0]
    assert allowed_404 not in errors[0]
    assert crawler.stats["seed_inventory_urls_failed"] == 2
    assert crawler.stats["seed_inventory_allowed_terminal_urls"] == 1
    assert crawler.stats["seed_inventory_unexpected_failed_urls"] == 1


def test_completion_gate_allows_explicit_terminal_500_fixture_policy():
    broken_fixture = (
        "https://preprod.mbzuai.ac.ae/"
        "keegan-test-page-aspire-phd-fellowship-program"
    )
    crawler = crawler_module.Crawl4AICrawler()
    crawler.config = {
        "require_complete_seed_inventory": True,
        "require_successful_seed_inventory": True,
        "seed_inventory_allowed_terminal_url_patterns": [
            r"^https://preprod\.mbzuai\.ac\.ae/keegan-test-page-"
            r"aspire-phd-fellowship-program/?$"
        ],
        "seed_inventory_allowed_terminal_statuses": [404, 410, 500],
    }
    crawler.max_pages = 100
    crawler.crawl_state = {"pending": []}
    crawler.seed_inventory = {"urls": [broken_fixture]}
    crawler.url_mapping = {
        broken_fixture: "SKIPPED_HTTP_500:browser_fetch_http_500"
    }
    crawler.stats = {}

    errors = crawler._crawl_completion_errors()

    assert errors == []
    assert crawler.stats["seed_inventory_allowed_terminal_urls"] == 1
    assert crawler.stats["seed_inventory_unexpected_failed_urls"] == 0


def test_refreshed_terminal_evidence_allows_inventory_404(monkeypatch):
    url = "https://preprod.mbzuai.ac.ae/ar/retired-page"
    observations = [
        {
            "attempt": attempt,
            "status": 404,
            "final_url": url,
            "title": "404 - Page Not Found",
            "response_bytes": 128,
            "response_sha256": str(attempt) * 64,
        }
        for attempt in range(1, 4)
    ]

    class Browser:
        async def fetch_html_with_refreshes(self, requested_url, **kwargs):
            assert requested_url == url
            assert kwargs == {
                "max_bytes": 2048,
                "attempts": 3,
                "backoff_sec": 0,
            }
            return 404, "error", url, {"content-type": "text/html"}, observations

    crawler = crawler_module.Crawl4AICrawler()
    crawler.config = {
        "require_complete_seed_inventory": True,
        "require_successful_seed_inventory": True,
        "seed_inventory_allowed_terminal_statuses": [404, 500],
    }
    crawler.max_pages = 100
    crawler.crawl_state = {"pending": []}
    crawler.seed_inventory = {"urls": [url]}
    crawler.dynamic_collection_inventory = {"urls": []}
    crawler.priority_seed_urls = []
    crawler.url_mapping = {url: "SKIPPED_HTTP_404:browser_fetch_http_404"}
    crawler.recoverable_skip_exhausted_urls = {url}
    crawler.recoverable_skip_retries = {url: 6}
    crawler.browser_terminal_verification_enabled = True
    crawler.browser_terminal_verification_statuses = {404, 500}
    crawler.browser_terminal_verification_attempts = 3
    crawler.browser_terminal_verification_backoff = 0
    crawler.browser_terminal_verification_request_delay = 0
    crawler.browser_terminal_verification_max_urls = 10
    crawler.browser_fetch_max_response_bytes = 2048
    crawler._active_browser_fetch = Browser()
    crawler.terminal_page_verification = {}
    crawler.stats = {"pages_failed": 1, "skipped_urls": 1}
    monkeypatch.setattr(crawler, "_flush_runtime_state", lambda *args, **kwargs: None)

    recovered = asyncio.run(crawler._verify_exhausted_browser_pages())

    assert recovered == []
    assert crawler.stats["terminal_pages_verified"] == 1
    assert crawler._crawl_completion_errors() == []
    verified = crawler_module._verified_terminal_page_urls(
        crawler.terminal_page_verification,
        allowed_statuses={404, 500},
        minimum_attempts=3,
    )
    assert verified == {url: 404}

    crawler.terminal_page_verification["records"][0]["observations"][0][
        "title"
    ] = "Not an error"
    assert crawler_module._verified_terminal_page_urls(
        crawler.terminal_page_verification,
        allowed_statuses={404, 500},
        minimum_attempts=3,
    ) == {}


def test_rendered_http_404_is_never_saved_as_successful_page(monkeypatch):
    crawler = crawler_module.Crawl4AICrawler()
    crawler.stats = {"pages_failed": 0, "skipped_urls": 0}
    crawler.url_mapping = {}
    monkeypatch.setattr(crawler, "_flush_runtime_state", lambda *args, **kwargs: None)

    processed = asyncio.run(
        crawler._process_result(
            SimpleNamespace(
                url="https://careers.mbzuai.ac.ae/careers/stale-vacancy",
                status_code=404,
                success=True,
                html="<html><body>Careers site shell</body></html>",
                error_message="",
            )
        )
    )

    assert processed is False
    assert crawler.url_mapping == {
        "https://careers.mbzuai.ac.ae/careers/stale-vacancy": "SKIPPED_HTTP_404"
    }
    assert crawler.stats == {"pages_failed": 1, "skipped_urls": 1}


def test_raw_source_404_overrides_renderer_http_200(monkeypatch):
    crawler = crawler_module.Crawl4AICrawler()
    crawler.validate_source_html = True
    crawler.validate_source_html_mode = "always"
    crawler.stats = {
        "pages_failed": 0,
        "skipped_urls": 0,
        "source_html_validations": 0,
    }
    crawler.url_mapping = {}
    monkeypatch.setattr(crawler, "_should_validate_source_html", lambda _url: True)
    monkeypatch.setattr(
        crawler,
        "_fetch_raw_source_page",
        lambda _url: asyncio.sleep(0, result=("", 404)),
    )
    monkeypatch.setattr(crawler, "_flush_runtime_state", lambda *args, **kwargs: None)

    processed = asyncio.run(
        crawler._process_result(
            SimpleNamespace(
                url="https://careers.mbzuai.ac.ae/careers/stale-vacancy",
                status_code=200,
                success=True,
                html="<html><body>Rendered WordPress 404 shell</body></html>",
                error_message="",
                markdown=None,
            )
        )
    )

    assert processed is False
    assert crawler.url_mapping == {
        "https://careers.mbzuai.ac.ae/careers/stale-vacancy": (
            "SKIPPED_HTTP_404:raw_source_terminal_http_status"
        )
    }
    assert crawler.stats["pages_failed"] == 1
    assert crawler.stats["source_html_validations"] == 0


def test_terminal_http_statuses_are_not_retried_as_browser_failures():
    assert not crawler_module._should_retry_page_failure(404)
    assert not crawler_module._should_retry_page_failure(401)
    assert crawler_module._should_retry_page_failure(None)
    assert crawler_module._should_retry_page_failure(301)
    assert crawler_module._should_retry_page_failure(403)
    assert crawler_module._should_retry_page_failure(503)


def test_site_specific_transient_statuses_can_include_preprod_404():
    assert crawler_module._should_retry_page_failure(404, {404, 500})
    assert not crawler_module._should_retry_page_failure(410, {404, 500})


def test_transient_source_404_cannot_invalidate_a_rendered_success():
    assert not crawler_module._is_authoritative_terminal_source_status(404, {404})
    assert crawler_module._is_authoritative_terminal_source_status(404, set())
    assert crawler_module._is_authoritative_terminal_source_status(410, {404})


def test_browser_fetch_adapter_returns_processable_html_results():
    class Browser:
        async def fetch_html(self, url, **kwargs):
            assert kwargs["context_url"] == "https://example.com"
            assert kwargs["max_bytes"] == 2048
            if url.endswith("/retry"):
                return 503, "temporary", url, {"content-type": "text/html"}
            return 200, "<html><body>Ready</body></html>", f"{url}/canonical", {
                "content-type": "text/html"
            }

    adapter = crawler_module._BrowserFetchCrawlerAdapter(
        Browser(),
        context_url="https://example.com",
        max_response_bytes=2048,
        request_delay=0,
    )
    results = asyncio.run(
        adapter.arun_many(
            urls=["https://example.com/ready", "https://example.com/retry"],
            config=SimpleNamespace(),
        )
    )

    assert [(result.status_code, result.success) for result in results] == [
        (200, True),
        (503, False),
    ]
    assert results[0].html == "<html><body>Ready</body></html>"
    assert results[0].final_url == "https://example.com/ready/canonical"
    assert results[1].error_message == "browser_fetch_http_503"


def test_successful_internal_redirect_maps_alias_to_one_canonical_artifact(
    tmp_path,
    monkeypatch,
):
    requested_url = "https://example.com/old-admissions"
    final_url = "https://example.com/current-admissions"
    crawler = crawler_module.Crawl4AICrawler()
    crawler.html_dir = tmp_path / "html"
    crawler.html_dir.mkdir()
    crawler.md_dir = tmp_path / "markdown"
    crawler.url_mapping = {}
    crawler.url_to_md_mapping = {}
    crawler.page_images = {}
    crawler.page_videos = {}
    crawler.page_media = {}
    crawler.page_links = {}
    crawler.page_metadata = {}
    crawler.dynamic_collection_inventory = {"collections": [], "urls": []}
    crawler.seed_inventory = {"endpoints": [], "urls": []}
    crawler.crawl_state = {"depths": {requested_url: 1, final_url: 1}}
    crawler.allowed_domains = {"example.com"}
    crawler.allowed_hosts = {"example.com"}
    crawler.excluded_subdomains = set()
    crawler.validate_source_html = False
    crawler.validate_source_html_mode = "never"
    crawler.write_markdown = False
    crawler.extract_images = False
    crawler.extract_videos = False
    crawler.stats = {
        "pages_scraped": 0,
        "bytes_downloaded": 0,
        "documents_downloaded": 0,
        "images_extracted": 0,
        "videos_extracted": 0,
        "content_quality_warnings": 0,
        "source_html_replacements": 0,
    }
    monkeypatch.setattr(crawler, "_url_allowed_for_fetch", lambda _url: True)
    monkeypatch.setattr(crawler, "_allow_frontier_url", lambda _url: True)
    monkeypatch.setattr(crawler, "_flush_runtime_state", lambda *args, **kwargs: None)

    html = (
        "<html><head><title>Current Admissions</title></head><body><main>"
        + "Current admissions requirements, deadlines, programs, and application "
        * 12
        + "</main></body></html>"
    )
    processed = asyncio.run(
        crawler._process_result(
            SimpleNamespace(
                url=requested_url,
                final_url=final_url,
                status_code=200,
                success=True,
                html=html,
                error_message="",
                response_headers={"content-type": "text/html"},
                links={"internal": [], "external": []},
                markdown=None,
            )
        )
    )

    assert processed is True
    assert crawler.stats["pages_scraped"] == 1
    assert crawler.url_mapping[requested_url] == crawler.url_mapping[final_url]
    assert crawler.page_metadata[final_url]["redirected_from"] == [requested_url]
    assert crawler.page_metadata[final_url]["final_url"] == final_url
