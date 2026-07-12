import asyncio
import json
from types import SimpleNamespace

from pipeline.core.base import StageContext, StageStatus
from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module


def _context(work_dir, *, max_pages=4):
    return StageContext(
        run_id="resume-integrity",
        project_name="test",
        config={
            "crawler": {
                "start_url": "https://example.com/a",
                "max_pages": max_pages,
                "max_depth": 3,
                "fetch_concurrency": 1,
                "download_concurrency": 1,
                "timeout": 5,
                "sitemap_enabled": False,
                "extract_images": False,
                "extract_videos": False,
                "download_page_images": False,
                "fetch_video_transcripts": False,
                "respect_robots_txt": False,
                "stream_results": False,
                "fail_on_empty_result": True,
                "validate_source_html": False,
                "checkpoint_flush_interval_sec": 60,
            },
            "converter": {"content_filter_threshold": 0.48},
        },
        work_dir=work_dir,
    )


def _result(url, title):
    body = " ".join([f"{title} content for crawler resume integrity."] * 20)
    return SimpleNamespace(
        url=url,
        html=f"<html><head><title>{title}</title></head><body><h1>{title}</h1><p>{body}</p></body></html>",
        success=True,
        status_code=200,
        links={"internal": [], "external": []},
        markdown=SimpleNamespace(
            fit_markdown=f"# {title}\n\n{body}",
            raw_markdown=f"# {title}\n\n{body}",
        ),
    )


def test_list_mode_interruption_requeues_only_unconsumed_visited_url(tmp_path, monkeypatch):
    first_url = "https://example.com/a"
    interrupted_url = "https://example.com/b"

    class InterruptedResults(list):
        def __iter__(self):
            yield self[0]
            raise RuntimeError("simulated interruption during result consumption")

    class FakeAsyncCrawler:
        calls = 0

        def __init__(self, config=None):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def arun(self, url, config):
            type(self).calls += 1
            strategy = config.deep_crawl_strategy
            if self.calls == 1:
                await strategy._on_state_change(
                    {
                        "strategy_type": "bfs",
                        "visited": [first_url, interrupted_url],
                        "pending": [],
                        "depths": {first_url: 0, interrupted_url: 1},
                        "pages_crawled": 2,
                    }
                )
                return InterruptedResults(
                    [_result(first_url, "First"), _result(interrupted_url, "Interrupted")]
                )

            resume_state = strategy._resume_state
            assert resume_state["visited"] == [first_url]
            assert resume_state["pending"] == [
                {"url": interrupted_url, "parent_url": None}
            ]
            assert resume_state["depths"] == {first_url: 0, interrupted_url: 1}
            assert resume_state["pages_crawled"] == 1
            await strategy._on_state_change(
                {
                    "strategy_type": "bfs",
                    "visited": [first_url, interrupted_url],
                    "pending": [],
                    "depths": {first_url: 0, interrupted_url: 1},
                    "pages_crawled": 2,
                }
            )
            return [_result(interrupted_url, "Interrupted")]

    monkeypatch.setattr(crawler_module, "AsyncWebCrawler", FakeAsyncCrawler)
    monkeypatch.setattr(
        crawler_module,
        "_host_resolves_to_private_or_reserved",
        lambda _host: False,
    )

    first = asyncio.run(crawler_module.Crawl4AICrawler().execute(_context(tmp_path)))
    assert first.status == StageStatus.FAILED
    assert json.loads((tmp_path / "mappings.json").read_text()).keys() == {first_url}

    resumed = asyncio.run(crawler_module.Crawl4AICrawler().execute(_context(tmp_path)))
    assert resumed.status == StageStatus.COMPLETED
    assert resumed.metrics["pages_scraped"] == 2
    assert set(json.loads((tmp_path / "mappings.json").read_text())) == {
        first_url,
        interrupted_url,
    }


def test_missing_mapped_output_is_requeued_and_stale_mapping_removed(tmp_path, monkeypatch):
    completed_url = "https://example.com/completed"
    skipped_url = "https://example.com/skipped"
    stale_url = "https://example.com/stale"
    completed_output = tmp_path / "completed.html"
    completed_output.write_text("<html>complete</html>", encoding="utf-8")

    crawler = object.__new__(crawler_module.Crawl4AICrawler)
    crawler.crawl_state = {
        "visited": [completed_url, skipped_url, stale_url],
        "pending": [],
        "depths": {completed_url: 0, skipped_url: 1, stale_url: 1},
        "pages_crawled": 3,
    }
    crawler.url_mapping = {
        completed_url: str(completed_output),
        skipped_url: "SKIPPED_HTTP_404",
        stale_url: str(tmp_path / "missing.html"),
    }
    crawler.max_pages = 3
    crawler.start_url = completed_url
    crawler.allowed_domains = {"example.com"}
    crawler.excluded_subdomains = set()
    crawler.excluded_path_prefixes = set()
    crawler.require_https = True
    crawler.allow_query_urls = False
    crawler.allowed_query_param_names = set()
    crawler.stats = {"skipped_urls": 1, "excluded_frontier_urls": 0}
    monkeypatch.setattr(
        crawler_module,
        "_host_resolves_to_private_or_reserved",
        lambda _host: False,
    )

    recovered = crawler._requeue_unprocessed_visited_urls()

    assert recovered == [stale_url]
    assert crawler.crawl_state["visited"] == [completed_url, skipped_url]
    assert crawler.crawl_state["pending"] == [{"url": stale_url, "parent_url": None}]
    assert crawler.crawl_state["pages_crawled"] == 2
    assert stale_url not in crawler.url_mapping
    assert crawler.url_mapping[completed_url] == str(completed_output)
    assert crawler.url_mapping[skipped_url] == "SKIPPED_HTTP_404"


def test_newer_public_frontier_wins_and_merges_lagging_runtime_checkpoint(tmp_path, monkeypatch):
    root = "https://example.com/a"
    target = "https://example.com/target"
    public_only = "https://example.com/public"
    stale_only = "https://example.com/stale"
    root_output = tmp_path / "html" / "root.html"
    root_output.parent.mkdir(parents=True)
    root_output.write_text("<html>root</html>", encoding="utf-8")

    (tmp_path / "crawler_checkpoint.json").write_text(
        json.dumps(
            {
                "updated_at": 100,
                "url_mapping": {root: str(root_output)},
                "crawl_state": {
                    "visited": [root],
                    "pending": [
                        {"url": target, "parent_url": "https://example.com/old-parent"},
                        {"url": stale_only, "parent_url": root},
                    ],
                    "depths": {root: 0, target: 3, stale_only: 2},
                    "pages_crawled": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "crawl_state.json").write_text(
        json.dumps(
            {
                "updated_at": 200,
                "visited": [root],
                "pending": [
                    {"url": target, "parent_url": root},
                    {"url": public_only, "parent_url": target},
                ],
                "depths": {root: 0, target: 1, public_only: 2},
                "pages_crawled": 1,
            }
        ),
        encoding="utf-8",
    )

    class InspectingCrawler:
        def __init__(self, config=None):
            self.config = config

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def arun(self, url, config):
            resume_state = config.deep_crawl_strategy._resume_state
            assert resume_state["visited"] == [root]
            assert resume_state["pending"] == [
                {"url": target, "parent_url": root},
                {"url": public_only, "parent_url": target},
                {"url": stale_only, "parent_url": root},
            ]
            assert resume_state["depths"][target] == 1
            assert resume_state["pages_crawled"] == 1
            raise RuntimeError("frontier inspected")

    monkeypatch.setattr(crawler_module, "AsyncWebCrawler", InspectingCrawler)
    monkeypatch.setattr(
        crawler_module,
        "_host_resolves_to_private_or_reserved",
        lambda _host: False,
    )

    result = asyncio.run(crawler_module.Crawl4AICrawler().execute(_context(tmp_path)))
    assert result.status == StageStatus.FAILED
    assert result.error_message == "frontier inspected"
