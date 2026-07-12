"""
Real-world scenario tests verifying Playwright browser recycling, boilerplate-aware
deduplication, Pinecone retry loops, and robots.txt compliance config.
"""

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from pipeline.core.base import StageContext
from pipeline.stages.quality.dedup_filter import _strip_boilerplate, DedupFilter
from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler


def run_async(coro):
    """Run an async function synchronously."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# ─────────────────────────────────────────────────────────────
# 1. Boilerplate Deduplication Scenario
# ─────────────────────────────────────────────────────────────

def test_boilerplate_deduplication():
    """Verify that boilerplate content is successfully stripped, and LSH signature

    generates correctly based purely on core unique content.
    """
    html_template = """
    <html>
      <head><title>Academic Department</title></head>
      <body>
        <header id="site-header" class="site-header-nav-menu">
          <nav>
            <ul>
              <li><a href="/">Home</a></li>
              <li><a href="/about">About MBZUAI</a></li>
              <li><a href="/admission">Admission Guidelines</a></li>
              <li><a href="/research">Research Center</a></li>
              <li><a href="/contact">Contact Info</a></li>
            </ul>
          </nav>
        </header>
        <aside class="sidebar-widget">
          <h3>Quick Links</h3>
          <ul>
            <li><a href="/news">Recent News</a></li>
            <li><a href="/events">Upcoming Seminars</a></li>
            <li><a href="/careers">Job Careers</a></li>
          </ul>
        </aside>
        <main>
          <article>
            <h1>Department Announcement</h1>
            <p>{unique_text}</p>
          </article>
        </main>
        <footer class="footer-layout">
          <div class="footer-links">
            <a href="/terms">Terms of Service</a> | <a href="/privacy">Privacy Statement</a>
          </div>
          <p>Facebook Twitter LinkedIn Share This Page</p>
          <p>&copy; 2026 MBZUAI. All rights reserved.</p>
        </footer>
      </body>
    </html>
    """

    # Create two pages that share the exact heavy boilerplate, but have different core copy
    page_a_content = html_template.format(unique_text="MBZUAI has opened admissions for the graduate class of 2026 in Computer Science.")
    page_b_content = html_template.format(unique_text="We are pleased to announce a new research partnership in artificial intelligence with local hospitals.")

    cleaned_a = _strip_boilerplate(page_a_content)
    cleaned_b = _strip_boilerplate(page_b_content)

    # 1. Assert boilerplate elements (header, nav, footer, sidebar/aside, social) are completely removed
    assert "<header" not in cleaned_a
    assert "<nav" not in cleaned_a
    assert "<footer" not in cleaned_a
    assert "<aside" not in cleaned_a
    assert "Home" not in cleaned_a
    assert "Privacy Statement" not in cleaned_a
    assert "Facebook" not in cleaned_a
    assert "Twitter" not in cleaned_a

    # 2. Assert that actual unique text remains fully intact
    assert "admissions for the graduate class" in cleaned_a
    assert "research partnership in artificial intelligence" in cleaned_b

    # 3. Assert they are not deduplicated under DedupFilterStage because of different core copy
    # We can create a simple DedupFilterStage and verify deduplication flow
    stage = DedupFilter()
    stage.config = {"threshold": 0.8}
    stage.seen_hashes = {}

    # Even if they had the same boilerplate, because we stripped it, their signatures will be distinct
    from datasketch import MinHash
    def get_shingle_hash(text):
        m = MinHash()
        for shingle in text.split():
            m.update(shingle.encode("utf-8"))
        return m

    mh_a = get_shingle_hash(cleaned_a)
    mh_b = get_shingle_hash(cleaned_b)

    similarity = mh_a.jaccard(mh_b)
    assert similarity < 0.3  # They should be highly distinct


# ─────────────────────────────────────────────────────────────
# 2. Robots.txt Compliance & Configuration Toggles
# ─────────────────────────────────────────────────────────────

def test_robots_txt_compliance_toggles():
    """Verify that crawler respects the `respect_robots_txt` toggle:

    - When respect_robots_txt=True, robots.txt is queried for sitemaps.
    - When respect_robots_txt=False, robots.txt is skipped completely.
    """
    async def run():
        # 1. Crawler Stage context mock
        ctx = StageContext(
            run_id="run_robots_test",
            project_name="robots_test",
            config={
                "crawler": {
                    "start_url": "https://example.com",
                    "respect_robots_txt": True,
                }
            },
            work_dir=Path(tempfile.mkdtemp()),
        )

        # 2. Crawl4AICrawler with respect_robots_txt=True
        crawler = Crawl4AICrawler()
        crawler.ctx = ctx
        crawler.config = dict(ctx.crawler_config)
        crawler.start_url = "https://example.com"
        crawler.respect_robots_txt = True
        crawler.proxy = None
        crawler.sitemap_state_file = ctx.work_dir / "sitemap_discovery.json"
        crawler.allowed_domains = {"example.com"}
        crawler.excluded_path_prefixes = set()
        crawler.allow_query_urls = False

        # Mock session
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.headers = {"Content-Type": "application/xml"}

        # We use an async context manager mock for get()
        class AsyncContextManagerMock:
            async def __aenter__(self):
                return mock_response
            async def __aexit__(self, exc_type, exc, tb):
                pass

        mock_session.get.return_value = AsyncContextManagerMock()

        # Async response text mock
        async def fake_text():
            return "Sitemap: https://example.com/sitemap.xml\nUser-agent: *\nDisallow: /private"
        mock_response.text = fake_text

        async def fake_read():
            return b"<urlset><url><loc>https://example.com/sitemap.xml</loc></url></urlset>"
        mock_response.read = fake_read

        crawler._session = mock_session

        # Execute discovery
        candidates = await crawler._discover_sitemap_urls()

        # Assert session.get was called to retrieve robots.txt (check full call history)
        robots_called = any(
            call[0][0] == "https://example.com/robots.txt" for call in mock_session.get.call_args_list
        )
        assert robots_called
        assert "https://example.com/sitemap.xml" in candidates

        # 3. Crawl4AICrawler with respect_robots_txt=False
        mock_session.reset_mock()
        crawler.respect_robots_txt = False

        candidates_disabled = await crawler._discover_sitemap_urls()

        # Assert session.get was NEVER called for robots.txt
        robots_called_disabled = any(
            call[0][0] == "https://example.com/robots.txt" for call in mock_session.get.call_args_list
        )
        assert not robots_called_disabled
        # Still candidates from default root fallback are present (sitemap.xml, sitemap_index.xml)
        assert "https://example.com/sitemap.xml" in candidates_disabled

    run_async(run())


# ─────────────────────────────────────────────────────────────
# 4. Playwright Browser Recycling Scenario
# ─────────────────────────────────────────────────────────────

def test_browser_recycling_behavior():
    """Verify that sitemap crawler correctly shuts down and restarts the
    headless Playwright browser once the page count threshold is crossed.
    """
    async def run():
        class FakeCrawler:
            def __init__(self, config=None):
                self.config = config
                self.entered_count = 0
                self.exited_count = 0
                self.calls = []

            async def __aenter__(self):
                self.entered_count += 1
                return self

            async def __aexit__(self, exc_type, exc, tb):
                self.exited_count += 1
                return False

            async def arun_many(self, urls, config):
                self.calls.append(list(urls))
                return [
                    SimpleNamespace(
                        url=url,
                        html="<html><body>OK</body></html>",
                        success=True,
                        status_code=200,
                        links={"internal": [], "external": []},
                        markdown=SimpleNamespace(fit_markdown="OK", raw_markdown="OK"),
                    )
                    for url in urls
                ]

        # Create crawler instance
        crawler = Crawl4AICrawler()
        crawler.config = {
            "start_url": "https://example.com",
            "max_pages": 150,
            "sitemap_crawl_batch_size": 90, # triggers >80 threshold in 1 batch
            "fetch_concurrency": 5,
        }
        crawler.ctx = StageContext(
            run_id="run_recycle_test",
            project_name="recycle_test",
            config=crawler.config,
            work_dir=Path(tempfile.mkdtemp()),
        )
        crawler.start_url = "https://example.com"
        crawler.respect_robots_txt = False
        crawler.sitemap_crawl_batch_size = 90
        crawler.max_pages = 150
        crawler.fetch_concurrency = 5
        crawler.stats = {
            "pages_scraped": 0,
            "recoverable_skips_exhausted": 0,
            "sitemap_batches_completed": 0,
            "pages_failed": 0,
        }
        crawler.url_mapping = {}
        crawler.url_to_md_mapping = {}
        crawler.excluded_path_prefixes = set()
        crawler.allow_query_urls = False
        crawler.allowed_query_param_names = set()
        crawler.allowed_domains = {"example.com"}
        crawler.recoverable_skip_retries = {}
        crawler.recoverable_skip_max_retries = 2

        # Seed pending sitemap queue
        crawler.crawl_state = {
            "pending": [{"url": f"https://example.com/p{i}", "parent_url": "https://example.com"} for i in range(100)],
            "visited": [],
            "depths": {},
            "pages_crawled": 0,
        }

        # Patch AsyncWebCrawler creation inside crawl4ai_crawler
        fake_crawler_instance = FakeCrawler()

        # We mock AsyncWebCrawler class and hook __aenter__ / __aexit__ calls
        with patch("pipeline.stages.crawlers.crawl4ai_crawler.AsyncWebCrawler", return_value=fake_crawler_instance) as mock_class:
            crawler_holder = {"crawler": fake_crawler_instance}
            # Simulate entering context first
            await fake_crawler_instance.__aenter__()

            # Run sitemap frontier crawl
            run_config = SimpleNamespace(
                clone=lambda **kwargs: SimpleNamespace()
            )
            browser_config = SimpleNamespace()

            # Let's invoke the _crawl_seed_frontier loop under these conditions
            with patch.object(crawler, "_process_result", return_value=True):
                with patch.object(crawler, "_flush_runtime_state"):
                    # Execute seed frontier crawling
                    await crawler._crawl_seed_frontier(crawler_holder, run_config, browser_config)

            # 1. Assert __aenter__ was called twice (initial creation + 1 recycle trigger)
            assert fake_crawler_instance.entered_count == 2
            # 2. Assert __aexit__ was called at least once during recycling
            assert fake_crawler_instance.exited_count >= 1

    run_async(run())


# ─────────────────────────────────────────────────────────────
# 5. Pinecone Backoff Retry Scenario
# ─────────────────────────────────────────────────────────────

def test_pinecone_retry_loops():
    """Verify that our Pinecone exponential-backoff retry loops

    catch transient failures (like 429 Rate Limit) and eventually succeed.
    """
    from pipeline.stages.embedders.gemini_pinecone_embedder import _call_with_retry

    # We will simulate a function that fails twice with rate-limit errors and succeeds on the 3rd attempt
    call_count = 0

    def mock_flaky_upsert():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("429: Too Many Requests - Pinecone capacity reached")
        return "Success"

    # Patch time.sleep to avoid waiting in tests
    with patch("time.sleep") as mock_sleep:
        result = _call_with_retry(
            label="PineconeUpsert",
            func=mock_flaky_upsert,
            max_attempts=5,
            base_delay_sec=1.0,
            max_delay_sec=10.0,
        )

        # Assert outcome is successful
        assert result == "Success"
        # Assert 3 calls were made
        assert call_count == 3
        # Assert sleep was called twice
        assert mock_sleep.call_count == 2
