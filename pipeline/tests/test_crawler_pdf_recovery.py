import asyncio
import contextlib
from pathlib import Path

from pipeline.stages.crawlers import crawl4ai_crawler as crawler_module
from pipeline.stages.crawlers.crawl4ai_crawler import Crawl4AICrawler


class _FakeContent:
    def __init__(self, payload: bytes):
        self.payload = payload

    async def iter_chunked(self, _size: int):
        if self.payload:
            yield self.payload


class _FakeResponse:
    def __init__(
        self,
        status: int,
        url: str,
        *,
        payload: bytes = b"",
        content_type: str = "application/pdf",
        location: str | None = None,
    ):
        self.status = status
        self.url = url
        self.headers = {"Content-Type": content_type}
        if location is not None:
            self.headers["Location"] = location
        self.content = _FakeContent(payload)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError("unexpected extra request")
        return self.responses.pop(0)


def _crawler(main_session: _FakeSession) -> Crawl4AICrawler:
    crawler = Crawl4AICrawler()
    crawler._session = main_session
    crawler._download_semaphore = asyncio.Semaphore(1)
    crawler.retry_attempts = 2
    crawler.retry_backoff = 0
    crawler.proxy = None
    crawler.config = {"user_agent": "MBZUAIIndexer/1.0"}
    crawler.ignore_https_errors = False
    crawler.timeout = 30
    crawler.url_mapping = {}
    crawler.stats = {"skipped_urls": 0, "bytes_downloaded": 0}
    crawler._url_allowed_for_fetch = lambda url: url.startswith("https://mbzuai.ac.ae/")
    return crawler


def _download(crawler: Crawl4AICrawler, url: str, destination: Path):
    return asyncio.run(
        crawler._download_binary(
            url=url,
            destination_dir=destination,
            max_bytes=1024 * 1024,
            expected_prefix=None,
            status_on_failure="SKIPPED_DOWNLOAD_FAILED",
            validate_document=True,
            enforce_allowed_domain=True,
        )
    )


def test_pdf_403_recovers_with_fresh_cookie_isolated_session(tmp_path: Path):
    url = "https://mbzuai.ac.ae/uploads/public-guide.pdf"
    main = _FakeSession([_FakeResponse(403, url)])
    fresh = _FakeSession(
        [_FakeResponse(200, url, payload=b"%PDF-1.7\nvalid public document")]
    )
    crawler = _crawler(main)

    @contextlib.asynccontextmanager
    async def fresh_session():
        yield fresh

    crawler._fresh_cookie_isolated_download_session = fresh_session

    path = _download(crawler, url, tmp_path)

    assert path is not None
    assert path.read_bytes().startswith(b"%PDF-")
    assert len(main.calls) == 1
    assert len(fresh.calls) == 1
    assert crawler.url_mapping == {}
    assert crawler.stats["skipped_urls"] == 0
    assert crawler.stats["bytes_downloaded"] == len(path.read_bytes())


def test_same_site_pdf_redirect_is_followed_and_streamed(tmp_path: Path):
    url = "https://mbzuai.ac.ae/uploads/guide.pdf"
    final_url = "https://mbzuai.ac.ae/documents/guide.pdf"
    session = _FakeSession(
        [
            _FakeResponse(302, url, location="/documents/guide.pdf"),
            _FakeResponse(
                200,
                final_url,
                payload=b"%PDF-1.7\nvalid redirected document",
            ),
        ]
    )
    crawler = _crawler(session)

    path = _download(crawler, url, tmp_path)

    assert path is not None
    assert path.read_bytes().startswith(b"%PDF-")
    assert [call[0] for call in session.calls] == [url, final_url]
    assert all(call[1]["allow_redirects"] is False for call in session.calls)
    assert crawler.url_mapping == {}
    assert crawler.stats["skipped_urls"] == 0


def test_persistent_pdf_403_is_bounded_and_recorded_once(tmp_path: Path):
    url = "https://mbzuai.ac.ae/uploads/still-blocked.pdf"
    main = _FakeSession([_FakeResponse(403, url)])
    fresh = _FakeSession([_FakeResponse(403, url), _FakeResponse(403, url)])
    crawler = _crawler(main)

    @contextlib.asynccontextmanager
    async def fresh_session():
        yield fresh

    crawler._fresh_cookie_isolated_download_session = fresh_session

    assert _download(crawler, url, tmp_path) is None
    assert len(main.calls) == 1
    assert len(fresh.calls) == crawler.retry_attempts
    assert crawler.url_mapping[url] == "SKIPPED_HTTP_403"
    assert crawler.stats["skipped_urls"] == 1


def test_generic_403_and_pdf_404_do_not_use_fresh_recovery(tmp_path: Path):
    generic_url = "https://mbzuai.ac.ae/uploads/data.csv"
    generic_main = _FakeSession([_FakeResponse(403, generic_url)])
    generic_crawler = _crawler(generic_main)

    @contextlib.asynccontextmanager
    async def forbidden_fresh_session():
        raise AssertionError("fresh recovery must be limited to a same-site PDF 403")
        yield

    generic_crawler._fresh_cookie_isolated_download_session = forbidden_fresh_session
    assert _download(generic_crawler, generic_url, tmp_path) is None
    assert generic_crawler.url_mapping[generic_url] == "SKIPPED_HTTP_403"

    pdf_url = "https://mbzuai.ac.ae/uploads/missing.pdf"
    pdf_main = _FakeSession([_FakeResponse(404, pdf_url)])
    pdf_crawler = _crawler(pdf_main)
    pdf_crawler._fresh_cookie_isolated_download_session = forbidden_fresh_session
    assert _download(pdf_crawler, pdf_url, tmp_path) is None
    assert pdf_crawler.url_mapping[pdf_url] == "SKIPPED_HTTP_404"


def test_pdf_requires_compatible_content_type_and_magic(tmp_path: Path):
    wrong_mime_url = "https://mbzuai.ac.ae/uploads/wrong-mime.pdf"
    wrong_mime = _crawler(
        _FakeSession(
            [
                _FakeResponse(
                    200,
                    wrong_mime_url,
                    payload=b"%PDF-1.7\nlooks like a PDF",
                    content_type="text/plain",
                )
            ]
        )
    )
    assert _download(wrong_mime, wrong_mime_url, tmp_path) is None
    assert wrong_mime.url_mapping[wrong_mime_url] == "SKIPPED_INVALID_DOCUMENT"

    wrong_magic_url = "https://mbzuai.ac.ae/uploads/wrong-magic.pdf"
    wrong_magic = _crawler(
        _FakeSession(
            [
                _FakeResponse(
                    200,
                    wrong_magic_url,
                    payload=b"not a PDF",
                    content_type="application/pdf",
                )
            ]
        )
    )
    assert _download(wrong_magic, wrong_magic_url, tmp_path) is None
    assert wrong_magic.url_mapping[wrong_magic_url] == "SKIPPED_INVALID_DOCUMENT"
    assert not list(tmp_path.glob("*.part"))


def test_fresh_session_removes_cookie_header_and_retains_tls_policy(monkeypatch):
    captured = {}

    class FakeConnector:
        def __init__(self, **kwargs):
            captured["connector"] = kwargs

    class FakeCookieJar:
        pass

    class FakeClientSession:
        def __init__(self, **kwargs):
            captured["session"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(crawler_module.aiohttp, "TCPConnector", FakeConnector)
    monkeypatch.setattr(crawler_module.aiohttp, "DummyCookieJar", FakeCookieJar)
    monkeypatch.setattr(crawler_module.aiohttp, "ClientSession", FakeClientSession)

    crawler = Crawl4AICrawler()
    crawler.config = {
        "headers": {"Accept": "application/pdf", "Cookie": "affinity=blocked"},
        "user_agent": "MBZUAIIndexer/1.0",
    }
    crawler.ignore_https_errors = False
    crawler.timeout = 30

    async def open_session():
        async with crawler._fresh_cookie_isolated_download_session():
            pass

    asyncio.run(open_session())

    assert captured["connector"]["ssl"] is True
    assert captured["connector"]["limit"] == 1
    assert captured["session"]["headers"] == {
        "Accept": "application/pdf",
        "User-Agent": "MBZUAIIndexer/1.0",
    }
    assert isinstance(captured["session"]["cookie_jar"], FakeCookieJar)


def test_fresh_pdf_redirect_still_enforces_egress_policy(tmp_path: Path):
    url = "https://mbzuai.ac.ae/uploads/redirected.pdf"
    main = _FakeSession([_FakeResponse(403, url)])
    fresh = _FakeSession(
        [
            _FakeResponse(
                302,
                url,
                location="https://untrusted.example/redirected.pdf",
            )
        ]
    )
    crawler = _crawler(main)

    @contextlib.asynccontextmanager
    async def fresh_session():
        yield fresh

    crawler._fresh_cookie_isolated_download_session = fresh_session

    assert _download(crawler, url, tmp_path) is None
    assert crawler.url_mapping[url] == "SKIPPED_EGRESS_POLICY"
    assert crawler.stats["skipped_urls"] == 1
    assert [call[0] for call in fresh.calls] == [url]
    assert fresh.calls[0][1]["allow_redirects"] is False
