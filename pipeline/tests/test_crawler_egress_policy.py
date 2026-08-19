import asyncio
from types import SimpleNamespace

from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler


def _crawler():
    crawler = Crawl4AICrawler()
    crawler.start_url = "https://mbzuai.ac.ae"
    crawler.config = {}
    crawler.allowed_domains = {"mbzuai.ac.ae"}
    crawler.excluded_subdomains = {"library.mbzuai.ac.ae"}
    crawler.excluded_path_prefixes = set()
    crawler.allow_query_urls = False
    crawler.allowed_query_param_names = set()
    crawler.proxy = None
    return crawler


def test_validate_config_rejects_plaintext_start_and_unknown_source_mode():
    errors = asyncio.run(
        Crawl4AICrawler().validate_config(
            {
                "crawler": {
                    "start_url": "http://mbzuai.ac.ae",
                    "require_https": True,
                    "validate_source_html_mode": "sometimes",
                }
            }
        )
    )

    assert (
        "crawler.start_url must use HTTPS when crawler.require_https is true"
        in errors
    )
    assert (
        "crawler.validate_source_html_mode must be always, on_quality_warning, or never"
        in errors
    )


def test_egress_policy_blocks_private_addresses(monkeypatch):
    crawler = _crawler()
    crawler_module._host_resolves_to_private_or_reserved.cache_clear()

    monkeypatch.setattr(
        crawler_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )

    assert crawler._url_allowed_for_fetch("https://mbzuai.ac.ae/sitemap.xml")
    assert not crawler._url_allowed_for_fetch("http://mbzuai.ac.ae/sitemap.xml")
    assert not crawler._url_allowed_for_fetch("http://127.0.0.1/admin.pdf")
    assert not crawler._url_allowed_for_fetch("http://169.254.169.254/latest/meta-data/foo.pdf")
    assert not crawler._url_allowed_for_fetch("https://evil.example/file.pdf")
    assert not crawler._url_allowed_for_fetch("https://library.mbzuai.ac.ae/private.pdf")


def test_egress_policy_derives_missing_allowlist_only_from_start_host(monkeypatch):
    crawler = Crawl4AICrawler()
    crawler.start_url = "https://mbzuai.ac.ae"
    crawler.excluded_subdomains = set()
    crawler_module._host_resolves_to_private_or_reserved.cache_clear()
    monkeypatch.setattr(
        crawler_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )

    assert crawler._url_allowed_for_fetch("https://mbzuai.ac.ae/admissions")
    assert not crawler._url_allowed_for_fetch("https://untrusted.example/admissions")


def test_egress_policy_denies_when_allowlist_and_start_host_are_missing(monkeypatch):
    crawler = Crawl4AICrawler()
    crawler.excluded_subdomains = set()
    crawler_module._host_resolves_to_private_or_reserved.cache_clear()
    monkeypatch.setattr(
        crawler_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )

    assert not crawler._url_allowed_for_fetch("https://untrusted.example/admissions")


def test_frontier_policy_blocks_allowed_domain_with_private_dns(monkeypatch):
    crawler = _crawler()
    crawler_module._host_resolves_to_private_or_reserved.cache_clear()

    monkeypatch.setattr(
        crawler_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("10.0.0.7", 443))],
    )

    assert not crawler._allow_frontier_url("https://mbzuai.ac.ae/admissions")


def test_downloadable_urls_keep_only_allowed_public_documents(monkeypatch):
    crawler = _crawler()
    crawler_module._host_resolves_to_private_or_reserved.cache_clear()

    monkeypatch.setattr(
        crawler_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )

    result = SimpleNamespace(
        links={
            "internal": [{"href": "/programs/catalog.pdf"}],
            "external": [
                {"href": "https://evil.example/malicious.pdf"},
                {"href": "http://169.254.169.254/latest/meta-data/iam.pdf"},
            ],
        }
    )
    html = '<a href="/admissions/guide.pdf">guide</a><a href="https://library.mbzuai.ac.ae/private.pdf">private</a>'

    assert crawler._extract_downloadable_urls(result, html, "https://mbzuai.ac.ae/study") == [
        "https://mbzuai.ac.ae/programs/catalog.pdf",
        "https://mbzuai.ac.ae/admissions/guide.pdf",
    ]


def test_raw_source_fetch_blocks_unsafe_redirect(monkeypatch):
    crawler = _crawler()
    crawler.raw_source_retry_attempts = 1
    crawler.raw_source_retry_backoff = 0
    crawler_module._host_resolves_to_private_or_reserved.cache_clear()

    monkeypatch.setattr(
        crawler_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [(None, None, None, None, ("93.184.216.34", 443))],
    )

    class FakeResponse:
        status = 200
        url = "http://169.254.169.254/latest/meta-data"

        async def text(self):
            return "unsafe redirect"

    class FakeGetContext:
        async def __aenter__(self):
            return FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            return False

    crawler._session = SimpleNamespace(get=lambda *args, **kwargs: FakeGetContext())

    html, status = asyncio.run(crawler._fetch_raw_source_page("https://mbzuai.ac.ae/admissions"))

    assert html == ""
    assert status == 200


def test_raw_source_fetch_never_requests_external_redirect_target(monkeypatch):
    crawler = _crawler()
    crawler.raw_source_retry_attempts = 1
    crawler.raw_source_retry_backoff = 0

    class RedirectResponse:
        status = 302
        url = "https://mbzuai.ac.ae/admissions"
        headers = {"Location": "https://untrusted.example/source"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class RedirectSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if len(self.calls) > 1:
                raise AssertionError("external redirect target must never be requested")
            return RedirectResponse()

    session = RedirectSession()
    crawler._session = session
    monkeypatch.setattr(
        crawler,
        "_url_allowed_for_fetch",
        lambda url: url.startswith("https://mbzuai.ac.ae/"),
    )

    html, status = asyncio.run(
        crawler._fetch_raw_source_page("https://mbzuai.ac.ae/admissions")
    )

    assert html == ""
    assert status == 302
    assert [call[0] for call in session.calls] == [
        "https://mbzuai.ac.ae/admissions"
    ]
    assert session.calls[0][1]["allow_redirects"] is False


def test_safe_redirects_stop_before_request_beyond_hop_limit(monkeypatch):
    crawler = _crawler()
    monkeypatch.setattr(crawler_module, "SAFE_REDIRECT_MAX_HOPS", 1)
    monkeypatch.setattr(crawler, "_url_allowed_for_fetch", lambda _url: True)

    class RedirectResponse:
        status = 302
        headers = {"Location": "/next"}

        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class RedirectSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if len(self.calls) > 2:
                raise AssertionError("redirect target beyond the limit must not be requested")
            return RedirectResponse(url)

    session = RedirectSession()

    async def fetch():
        async with crawler._get_with_safe_redirects(
            session=session,
            url="https://mbzuai.ac.ae/start",
            enforce_allowed_domain=True,
        ):
            raise AssertionError("a redirect-only chain must not yield a response")

    try:
        asyncio.run(fetch())
    except crawler_module._SafeRedirectLimitError:
        pass
    else:
        raise AssertionError("redirect chain must fail closed at the hop limit")

    assert [call[0] for call in session.calls] == [
        "https://mbzuai.ac.ae/start",
        "https://mbzuai.ac.ae/next",
    ]
    assert all(call[1]["allow_redirects"] is False for call in session.calls)


def test_public_external_fetch_mode_still_blocks_private_redirect_target(monkeypatch):
    crawler = _crawler()
    crawler.require_https = True
    crawler_module._host_resolves_to_private_or_reserved.cache_clear()
    monkeypatch.setattr(
        crawler_module.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: [
            (None, None, None, None, ("93.184.216.34", 443))
        ],
    )

    class RedirectResponse:
        status = 302
        headers = {"Location": "http://169.254.169.254/latest/meta-data"}

        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class RedirectSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if len(self.calls) > 1:
                raise AssertionError("private redirect target must never be requested")
            return RedirectResponse(url)

    session = RedirectSession()

    async def fetch():
        async with crawler._get_with_safe_redirects(
            session=session,
            url="https://cdn.example.edu/campus.jpg",
            enforce_allowed_domain=False,
        ):
            raise AssertionError("a blocked redirect must not yield a response")

    try:
        asyncio.run(fetch())
    except crawler_module._RedirectEgressPolicyError:
        pass
    else:
        raise AssertionError("private redirect target must fail the request")

    assert [call[0] for call in session.calls] == [
        "https://cdn.example.edu/campus.jpg"
    ]
    assert session.calls[0][1]["allow_redirects"] is False


def test_video_transcript_redirect_never_requests_external_target(monkeypatch):
    crawler = _crawler()
    crawler.require_https = True
    crawler._download_semaphore = asyncio.Semaphore(1)
    crawler.max_video_transcript_bytes = 1024
    crawler.stats = {"video_transcripts_fetched": 0}
    monkeypatch.setattr(
        crawler,
        "_url_allowed_for_fetch",
        lambda url: url.startswith("https://mbzuai.ac.ae/"),
    )

    class RedirectResponse:
        status = 302
        headers = {"Location": "https://untrusted.example/captions.vtt"}

        def __init__(self, url):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class RedirectSession:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if len(self.calls) > 1:
                raise AssertionError("external transcript target must never be requested")
            return RedirectResponse(url)

    session = RedirectSession()
    crawler._session = session
    video = {}

    asyncio.run(
        crawler._fetch_video_transcript(
            video,
            "https://mbzuai.ac.ae/media/captions.vtt",
        )
    )

    assert video == {}
    assert crawler.stats["video_transcripts_fetched"] == 0
    assert [call[0] for call in session.calls] == [
        "https://mbzuai.ac.ae/media/captions.vtt"
    ]
    assert session.calls[0][1]["allow_redirects"] is False
