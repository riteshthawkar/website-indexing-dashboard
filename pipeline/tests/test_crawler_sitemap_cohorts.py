import asyncio
import json
from pathlib import Path

import pytest

from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
from pipeline.stages.crawlers.crawl4ai_crawler import (
    Crawl4AICrawler,
    _normalize_known_empty_cohort_policies,
)


SOURCE = "https://mbzuai.ac.ae/publication-sitemap.xml"
EMPTY_URL = "https://mbzuai.ac.ae/publication/legacy-author"
SOURCE_FETCHES = {
    SOURCE: {"status": 200, "payload_sha256": "a" * 64},
}


def _crawler(tmp_path: Path):
    crawler = object.__new__(Crawl4AICrawler)
    crawler.known_empty_sitemap_cohorts = [
        {
            "id": "legacy-publications-v1",
            "source_url": SOURCE,
            "allowed_path_prefixes": ["/publication"],
            "exact_urls": [],
            "expected_member_count": 1,
            "max_members": 2,
        }
    ]
    crawler.config = {}
    crawler.timeout = 5
    crawler.ignore_https_errors = False
    crawler.proxy = None
    crawler.fetch_concurrency = 2
    crawler.cohort_probe_concurrency = 2
    crawler.cohort_probe_max_bytes = 64
    crawler.cohort_probe_attempts = 1
    crawler.cohort_probe_backoff = 0
    crawler.url_mapping = {}
    crawler.stats = {"skipped_urls": 0, "verified_empty_urls": 0}
    crawler.sitemap_cohort_verification_file = tmp_path / "cohort.json"
    return crawler


def test_policy_normalization_fails_closed_on_invalid_counts():
    policies, errors = _normalize_known_empty_cohort_policies(
        [
            {
                "id": "legacy-publications-v1",
                "source_url": SOURCE,
                "allowed_path_prefixes": ["/publication"],
                "expected_member_count": 2,
                "max_members": 1,
            }
        ],
        start_url="https://mbzuai.ac.ae",
    )

    assert policies[0]["source_url"] == SOURCE
    assert errors == [
        "crawler.known_empty_sitemap_cohorts[0].max_members must be a positive "
        "integer >= expected_member_count"
    ]


def test_probe_bounds_are_validated_before_execution(monkeypatch):
    monkeypatch.setattr(
        "pipeline.stages.crawlers.crawl4ai_crawler.AsyncWebCrawler",
        object,
    )
    errors = asyncio.run(
        Crawl4AICrawler().validate_config(
            {
                "crawler": {
                    "start_url": "https://mbzuai.ac.ae",
                    "cohort_probe_concurrency": 0,
                    "cohort_probe_max_bytes": "invalid",
                    "cohort_probe_attempts": -1,
                    "cohort_probe_backoff_sec": -0.1,
                    "cohort_probe_min_interval_sec": -0.1,
                }
            }
        )
    )

    assert "crawler.cohort_probe_concurrency must be a positive integer" in errors
    assert "crawler.cohort_probe_max_bytes must be a positive integer" in errors
    assert "crawler.cohort_probe_attempts must be a positive integer" in errors
    assert (
        "crawler.cohort_probe_backoff_sec must be a non-negative number" in errors
    )
    assert (
        "crawler.cohort_probe_min_interval_sec must be a non-negative number"
        in errors
    )


def test_concurrent_cohort_probes_keep_one_global_start_interval(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)
    crawler.cohort_probe_min_interval = 1.5
    crawler._cohort_probe_rate_lock = asyncio.Lock()
    crawler._cohort_probe_last_started_at = 0.0
    clock = {"now": 10.0}
    sleeps = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds
        await real_sleep(0)

    monkeypatch.setattr(crawler_module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(crawler_module.asyncio, "sleep", fake_sleep)

    async def run_probes():
        await asyncio.gather(
            *(crawler._pace_cohort_probe_request() for _ in range(4))
        )

    asyncio.run(run_probes())

    assert sleeps == [1.5, 1.5, 1.5]
    assert crawler._cohort_probe_last_started_at == 14.5


def test_exact_source_verified_empty_is_excluded_but_other_source_is_not(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)
    other_url = "https://mbzuai.ac.ae/publication/from-another-source"

    async def probe(**kwargs):
        return {
            "url": kwargs["url"],
            "policy_id": kwargs["policy_id"],
            "source_url": kwargs["source_url"],
            "final_url": f"{kwargs['url']}/",
            "final_status": 200,
            "body_bytes": 0,
            "reached_eof": True,
            "classification": "verified_empty",
        }

    monkeypatch.setattr(crawler, "_probe_known_empty_url", probe)
    eligible = asyncio.run(
        crawler._verify_known_empty_sitemap_cohorts(
            [EMPTY_URL, other_url],
            {
                EMPTY_URL: [SOURCE],
                other_url: ["https://mbzuai.ac.ae/another-sitemap.xml"],
            },
            SOURCE_FETCHES,
        )
    )

    assert eligible == [other_url]
    assert crawler.verified_empty_urls == {EMPTY_URL: "legacy-publications-v1"}
    assert crawler.url_mapping[EMPTY_URL] == (
        "SKIPPED_VERIFIED_EMPTY_COHORT:legacy-publications-v1"
    )
    assert crawler.stats["skipped_urls"] == 1
    assert crawler.sitemap_cohort_verification_file.is_file()


def test_member_that_becomes_nonempty_is_automatically_reincluded(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)

    async def probe(**kwargs):
        return {
            "url": kwargs["url"],
            "policy_id": kwargs["policy_id"],
            "source_url": kwargs["source_url"],
            "final_url": f"{kwargs['url']}/",
            "final_status": 200,
            "body_bytes": 128,
            "reached_eof": True,
            "classification": "content",
        }

    monkeypatch.setattr(crawler, "_probe_known_empty_url", probe)
    eligible = asyncio.run(
        crawler._verify_known_empty_sitemap_cohorts(
            [EMPTY_URL],
            {EMPTY_URL: [SOURCE]},
            SOURCE_FETCHES,
        )
    )

    assert eligible == [EMPTY_URL]
    assert crawler.verified_empty_urls == {}
    assert crawler.url_mapping == {}


def test_unexpected_cohort_member_count_stops_before_exclusion(tmp_path):
    crawler = _crawler(tmp_path)

    with pytest.raises(RuntimeError, match="member count changed"):
        asyncio.run(
            crawler._verify_known_empty_sitemap_cohorts([], {}, SOURCE_FETCHES)
        )

    assert crawler.url_mapping == {}


class _FakeContent:
    def __init__(self, body):
        self.body = body

    async def read(self, _limit):
        return self.body

    def at_eof(self):
        return True


class _FakeResponse:
    def __init__(self, *, status, body):
        self.status = status
        self.url = f"{EMPTY_URL}/"
        self.content = _FakeContent(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class _FakeSession:
    def __init__(self, *, status, body):
        self.status = status
        self.body = body

    def get(self, *_args, **_kwargs):
        return _FakeResponse(status=self.status, body=self.body)


@pytest.mark.parametrize(
    ("status", "body", "classification"),
    [(200, b"", "verified_empty"), (200, b"content", "content"), (403, b"", "http_error")],
)
def test_probe_requires_exact_empty_200(tmp_path, monkeypatch, status, body, classification):
    crawler = _crawler(tmp_path)
    monkeypatch.setattr(crawler, "_url_allowed_for_fetch", lambda _url: True)

    record = asyncio.run(
        crawler._probe_known_empty_url(
            session=_FakeSession(status=status, body=body),
            semaphore=asyncio.Semaphore(1),
            url=EMPTY_URL,
            policy_id="legacy-publications-v1",
            source_url=SOURCE,
        )
    )

    assert record["classification"] == classification


def test_probe_preserves_canonical_trailing_slash_redirect(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)
    monkeypatch.setattr(crawler, "_url_allowed_for_fetch", lambda _url: True)
    paced_requests = []

    async def pace_request():
        paced_requests.append(True)

    monkeypatch.setattr(crawler, "_pace_cohort_probe_request", pace_request)

    class SlashResponse(_FakeResponse):
        def __init__(self, request_url):
            super().__init__(
                status=200 if request_url.endswith("/") else 301,
                body=b"",
            )
            self.url = request_url
            self.headers = (
                {} if request_url.endswith("/") else {"Location": f"{request_url}/"}
            )

    class SlashSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            return SlashResponse(url)

    session = SlashSession()
    record = asyncio.run(
        crawler._probe_known_empty_url(
            session=session,
            semaphore=asyncio.Semaphore(1),
            url=EMPTY_URL,
            policy_id="legacy-publications-v1",
            source_url=SOURCE,
        )
    )

    assert record["classification"] == "verified_empty"
    assert [call[0] for call in session.calls] == [EMPTY_URL, f"{EMPTY_URL}/"]
    assert len(paced_requests) == 2


def test_atomic_runtime_checkpoint_precedes_public_frontier_projection(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)
    crawler.start_url = "https://mbzuai.ac.ae"
    crawler.crawl_state = {
        "visited": [],
        "pending": [{"url": "https://mbzuai.ac.ae/", "parent_url": None}],
        "depths": {"https://mbzuai.ac.ae/": 0},
        "pages_crawled": 0,
    }
    crawler.url_mapping = {
        EMPTY_URL: "SKIPPED_VERIFIED_EMPTY_COHORT:legacy-publications-v1"
    }
    crawler.url_to_md_mapping = {}
    crawler.page_images = {}
    crawler.page_videos = {}
    crawler.page_media = {}
    crawler.page_metadata = {}
    crawler.page_links = {}
    crawler.downloaded_images = {}
    crawler.recoverable_skip_retries = {}
    crawler.recoverable_skip_exhausted_urls = set()
    crawler.sitemap_cohort_verification = {"evidence_sha256": "proof"}
    crawler.discovered_sitemaps = {"raw_urls": [EMPTY_URL]}
    crawler.runtime_state_file = tmp_path / "crawler_checkpoint.json"
    crawler.checkpoint_flush_interval = 0
    crawler._last_flush_at = 0

    def interrupted_projection():
        raise KeyboardInterrupt

    monkeypatch.setattr(crawler, "_write_crawl_state_file", interrupted_projection)

    with pytest.raises(KeyboardInterrupt):
        crawler._flush_runtime_state(force=True)

    persisted = json.loads(crawler.runtime_state_file.read_text())
    assert persisted["crawl_state"] == crawler.crawl_state
    assert persisted["url_mapping"] == crawler.url_mapping
    assert persisted["sitemap_cohort_verification"] == crawler.sitemap_cohort_verification
    assert persisted["discovered_sitemaps"] == crawler.discovered_sitemaps


def test_stage_outputs_omit_unconfigured_cohort_evidence(tmp_path):
    crawler = _crawler(tmp_path)
    for attribute, filename in {
        "html_dir": "html",
        "md_dir": "markdown",
        "download_dir": "downloads",
        "mapping_file": "mappings.json",
        "page_images_file": "page_images.json",
        "page_videos_file": "page_videos.json",
        "page_media_file": "page_media.json",
        "page_metadata_file": "page_metadata.json",
        "page_link_graph_file": "page_link_graph.json",
        "runtime_state_file": "crawler_checkpoint.json",
        "sitemap_state_file": "sitemap_discovery.json",
        "images_dir": "downloaded_page_images",
        "url_to_md_mapping_file": "url_to_markdown.json",
    }.items():
        setattr(crawler, attribute, tmp_path / filename)

    crawler.sitemap_cohort_verification = None
    outputs = crawler._build_stage_outputs()

    assert "sitemap_cohort_verification_file" not in outputs

    crawler.sitemap_cohort_verification = {"evidence_sha256": "proof"}
    outputs = crawler._build_stage_outputs()

    assert outputs["sitemap_cohort_verification_file"] == str(
        crawler.sitemap_cohort_verification_file
    )


def test_probe_never_requests_external_redirect_target(tmp_path, monkeypatch):
    crawler = _crawler(tmp_path)
    monkeypatch.setattr(
        crawler,
        "_url_allowed_for_fetch",
        lambda url: url.startswith("https://mbzuai.ac.ae/"),
    )

    class RedirectResponse:
        status = 302
        url = EMPTY_URL
        headers = {"Location": "https://untrusted.example/empty"}
        content = _FakeContent(b"")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

    class RedirectSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if len(self.calls) > 1:
                raise AssertionError("external redirect target must never be requested")
            return RedirectResponse()

    session = RedirectSession()
    record = asyncio.run(
        crawler._probe_known_empty_url(
            session=session,
            semaphore=asyncio.Semaphore(1),
            url=EMPTY_URL,
            policy_id="legacy-publications-v1",
            source_url=SOURCE,
        )
    )

    assert record["classification"] == "egress_rejected"
    assert record["error_type"] == "redirect_egress_policy"
    assert [call[0] for call in session.calls] == [EMPTY_URL]
    assert session.calls[0][1]["allow_redirects"] is False
