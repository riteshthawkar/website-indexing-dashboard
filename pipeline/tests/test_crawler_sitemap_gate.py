import asyncio

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
