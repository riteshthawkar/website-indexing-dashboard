from pipeline.core.sitemap_cohorts import (
    finalize_evidence,
    member_urls_sha256,
    parse_verified_empty_reason,
    validate_evidence,
    verified_empty_reason,
)


def _evidence(*, classification="verified_empty", body_bytes=0, status=200):
    url = "https://mbzuai.ac.ae/publication/legacy-author"
    policy = {
        "id": "legacy-publications-v1",
        "source_url": "https://mbzuai.ac.ae/publication-sitemap.xml",
        "source_payload_sha256": "a" * 64,
        "expected_member_count": 1,
        "member_count": 1,
        "member_urls_sha256": member_urls_sha256([url]),
        "records": [
            {
                "url": url,
                "policy_id": "legacy-publications-v1",
                "source_url": "https://mbzuai.ac.ae/publication-sitemap.xml",
                "final_url": f"{url}/",
                "final_status": status,
                "body_bytes": body_bytes,
                "reached_eof": True,
                "classification": classification,
            }
        ],
    }
    return finalize_evidence({"policies": [policy]})


def test_valid_verified_empty_evidence_round_trips():
    payload = _evidence()

    verified, errors = validate_evidence(payload)

    assert errors == []
    assert verified == {
        "https://mbzuai.ac.ae/publication/legacy-author": "legacy-publications-v1"
    }
    reason = verified_empty_reason("legacy-publications-v1")
    assert parse_verified_empty_reason(reason) == "legacy-publications-v1"


def test_nonempty_record_is_not_excluded_but_remains_valid_evidence():
    payload = _evidence(classification="content", body_bytes=128)

    verified, errors = validate_evidence(payload)

    assert errors == []
    assert verified == {}


def test_invalid_status_cannot_prove_empty():
    payload = _evidence(status=403)

    verified, errors = validate_evidence(payload)

    assert verified == {}
    assert errors == [
        "invalid verified-empty proof for https://mbzuai.ac.ae/publication/legacy-author"
    ]


def test_tampered_evidence_fails_closed():
    payload = _evidence()
    payload["policies"][0]["records"][0]["body_bytes"] = 1

    verified, errors = validate_evidence(payload)

    assert verified == {}
    assert "sitemap cohort evidence digest mismatch" in errors


def test_malformed_numeric_fields_fail_closed_without_raising():
    payload = _evidence()
    payload["policies"][0]["member_count"] = "not-an-integer"
    payload["policies"][0]["records"][0]["body_bytes"] = "not-an-integer"
    payload = finalize_evidence(payload)

    verified, errors = validate_evidence(payload)

    assert verified == {}
    assert any("member_count mismatch" in error for error in errors)
    assert any("invalid verified-empty proof" in error for error in errors)


def test_records_must_equal_config_selector_derived_sitemap_members():
    expected_url = "https://mbzuai.ac.ae/ar/news"
    wrong_url = "https://mbzuai.ac.ae/news/some-article"
    source_url = "https://mbzuai.ac.ae/news-sitemap.xml"
    policy = {
        "id": "empty-news-hub-v1",
        "source_url": source_url,
        "source_payload_sha256": "b" * 64,
        "allowed_path_prefixes": [],
        "exact_urls": [expected_url],
        "expected_member_count": 1,
        "max_members": 1,
        "member_count": 1,
        "member_urls_sha256": member_urls_sha256([wrong_url]),
        "records": [
            {
                "url": wrong_url,
                "policy_id": "empty-news-hub-v1",
                "source_url": source_url,
                "final_url": f"{wrong_url}/",
                "final_status": 200,
                "body_bytes": 0,
                "reached_eof": True,
                "classification": "verified_empty",
            }
        ],
    }
    payload = finalize_evidence({"policies": [policy]})

    verified, errors = validate_evidence(
        payload,
        expected_policies=[
            {
                "id": "empty-news-hub-v1",
                "source_url": source_url,
                "allowed_path_prefixes": [],
                "exact_urls": [expected_url],
                "expected_member_count": 1,
                "max_members": 1,
            }
        ],
        sitemap_snapshot={
            "raw_urls": [wrong_url],
            "url_sources": {wrong_url: [source_url]},
            "source_fetches": {
                source_url: {"status": 200, "payload_sha256": "b" * 64}
            },
        },
    )

    assert verified == {}
    assert any("selector-derived sitemap members" in error for error in errors)
