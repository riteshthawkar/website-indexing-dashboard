from pipeline.core.sitemap_cohorts import (
    finalize_evidence,
    member_urls_sha256,
    verified_empty_reason,
)
from pipeline.stages.formatters.mbzuai_index_readiness_formatter import _failure_manifest


URL = "https://mbzuai.ac.ae/publication/legacy-author"
POLICY_ID = "legacy-publications-v1"
SOURCE_URL = "https://mbzuai.ac.ae/publication-sitemap.xml"
COHORT_CONFIG = {
    "known_empty_sitemap_cohorts": [
        {
            "id": POLICY_ID,
            "source_url": SOURCE_URL,
            "allowed_path_prefixes": ["/publication"],
            "expected_member_count": 1,
            "max_members": 1,
        }
    ]
}


def _runtime_state(*, include_mapping=True):
    record = {
        "url": URL,
        "policy_id": POLICY_ID,
        "source_url": SOURCE_URL,
        "final_url": f"{URL}/",
        "final_status": 200,
        "body_bytes": 0,
        "reached_eof": True,
        "classification": "verified_empty",
    }
    evidence = finalize_evidence(
        {
            "policies": [
                {
                    "id": POLICY_ID,
                    "source_url": SOURCE_URL,
                    "source_payload_sha256": "a" * 64,
                    "allowed_path_prefixes": ["/publication"],
                    "exact_urls": [],
                    "expected_member_count": 1,
                    "max_members": 1,
                    "member_count": 1,
                    "member_urls_sha256": member_urls_sha256([URL]),
                    "records": [record],
                }
            ]
        }
    )
    return {
        "stats": {"pages_scraped": 2400, "documents_downloaded": 100},
        "crawl_state": {"visited": [], "pending": [], "pages_crawled": 0},
        "url_mapping": {URL: verified_empty_reason(POLICY_ID)} if include_mapping else {},
        "sitemap_cohort_verification": evidence,
        "discovered_sitemaps": {
            "raw_urls": [URL],
            "url_sources": {URL: [SOURCE_URL]},
            "source_fetches": {
                SOURCE_URL: {"status": 200, "payload_sha256": "a" * 64}
            },
        },
    }


def test_evidence_backed_empty_cohort_is_intentional_not_hard():
    manifest = _failure_manifest(
        _runtime_state(),
        {"expected_site_inventory_count": 2600},
    )

    assert manifest["verified_empty_count"] == 1
    assert manifest["intentional_excluded_count"] == 1
    assert manifest["hard_failure_count"] == 0
    assert manifest["effective_expected_inventory_count"] == 2599
    assert manifest["cohort_evidence_errors"] == []


def test_verified_empty_reason_without_evidence_is_hard_failure():
    runtime = _runtime_state()
    runtime.pop("sitemap_cohort_verification")

    manifest = _failure_manifest(
        runtime,
        {"expected_site_inventory_count": 2600},
    )

    assert manifest["intentional_excluded_count"] == 0
    assert manifest["hard_failure_count"] == 1


def test_verified_evidence_without_matching_mapping_fails_closed():
    manifest = _failure_manifest(
        _runtime_state(include_mapping=False),
        {"expected_site_inventory_count": 2600},
    )

    assert manifest["intentional_excluded_count"] == 0
    assert manifest["cohort_evidence_errors"]
    assert "missing matching URL mappings" in manifest["cohort_evidence_errors"][0]


def test_tampered_evidence_never_reduces_hard_failure_count():
    runtime = _runtime_state()
    runtime["sitemap_cohort_verification"]["policies"][0]["records"][0][
        "body_bytes"
    ] = 1

    manifest = _failure_manifest(
        runtime,
        {"expected_site_inventory_count": 2600},
    )

    assert manifest["intentional_excluded_count"] == 0
    assert manifest["hard_failure_count"] == 1
    assert "digest mismatch" in manifest["cohort_evidence_errors"][0]


def test_configured_cohorts_require_evidence_and_discovery_snapshot():
    runtime = _runtime_state()
    runtime.pop("sitemap_cohort_verification")

    manifest = _failure_manifest(
        runtime,
        {"expected_site_inventory_count": 2600},
        COHORT_CONFIG,
    )

    assert manifest["intentional_excluded_count"] == 0
    assert "configured sitemap cohort evidence is missing" in manifest[
        "cohort_evidence_errors"
    ]


def test_configured_cohort_evidence_is_bound_to_sitemap_source_digest():
    runtime = _runtime_state()
    runtime["discovered_sitemaps"]["source_fetches"][SOURCE_URL][
        "payload_sha256"
    ] = "b" * 64

    manifest = _failure_manifest(
        runtime,
        {"expected_site_inventory_count": 2600},
        COHORT_CONFIG,
    )

    assert manifest["intentional_excluded_count"] == 0
    assert any(
        "source payload digest mismatch" in error
        for error in manifest["cohort_evidence_errors"]
    )
