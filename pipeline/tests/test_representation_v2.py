from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator

from pipeline.core.base import StageContext, StageStatus
from pipeline.core.page_cards import (
    build_document_revisions,
    extract_page_draft,
    finalize_page_drafts,
    link_page_cards_to_revisions,
)
from pipeline.core.representation_v2 import (
    REPRESENTATION_V2_KIND,
    REPRESENTATION_V2_SCHEMA_VERSION,
    attach_page_links_to_documents,
    now_iso,
    representation_v2_json_schema,
    validate_representation_v2,
)
from pipeline.stages.formatters.representation_v2_formatter import (
    RepresentationV2Formatter,
)


def _metadata(url: str, html_path: Path, **overrides: object) -> dict:
    payload = {
        "url": url,
        "source_url": url,
        "canonical_url": url,
        "canonical_family_url": url,
        "host": url.split("/", 3)[2],
        "path": "/" + url.split("/", 3)[3] if len(url.split("/", 3)) > 3 else "/",
        "status_code": 200,
        "depth": 1,
        "title": "Admissions at MBZUAI",
        "description": "Learn about the evidence-backed admissions process and how to submit an application.",
        "language": "en",
        "page_type": "admissions_or_program",
        "indexable": True,
        "html_path": str(html_path),
        "meta_tags": {"og:locale": "en_US"},
    }
    payload.update(overrides)
    return payload


def _inventory_document(url: str, markdown_path: Path, **overrides: object) -> dict:
    digest = hashlib.sha256(markdown_path.read_bytes()).hexdigest()
    payload = {
        "record_id": f"corpus-document:{hashlib.sha1(url.encode()).hexdigest()[:24]}",
        "source_url": url,
        "source_alias_urls": [],
        "source_file": "",
        "source_locator": {"kind": "url", "value": url},
        "markdown_path": str(markdown_path),
        "markdown_sha256": digest,
        "source_type": "webpage",
        "language": "en",
        "title": "Admissions at MBZUAI",
        "canonical_url": url,
        "canonical_family_url": url,
        "content_statistics": {"character_count": len(markdown_path.read_text())},
        "media_references": [],
    }
    payload.update(overrides)
    return payload


def test_page_card_actions_are_evidence_backed_and_template_noise_is_removed(tmp_path: Path):
    html_path = tmp_path / "admissions.html"
    html_path.write_text(
        """
        <html lang="en"><head>
          <title>Admissions at MBZUAI</title>
          <link rel="canonical" href="https://mbzuai.ac.ae/study/admissions/">
        </head><body>
          <header><nav><a href="/about">About</a></nav></header>
          <main>
            <h1>Graduate admissions</h1>
            <p>Applicants can review the requirements and submit an application to MBZUAI.</p>
            <section>
              <h2>Application process</h2>
              <a href="/apply">Apply now</a>
              <a href="/files/admissions-guide.pdf" download>Download admissions guide</a>
              <a href="javascript:alert(1)">Unsafe action</a>
            </section>
            <form action="/search" method="get">
              <input type="search" name="query"><button type="submit">Search</button>
            </form>
          </main>
          <footer><h2>Follow us</h2><a href="https://facebook.com/mbzuai">Facebook</a></footer>
        </body></html>
        """,
        encoding="utf-8",
    )
    url = "https://mbzuai.ac.ae/study/admissions"
    draft = extract_page_draft(
        url,
        _metadata(url, html_path),
        official_hosts={"mbzuai.ac.ae"},
    )
    pages, actions, stats = finalize_page_drafts([draft])

    page = pages[0]
    assert page["schema_version"] == REPRESENTATION_V2_SCHEMA_VERSION
    assert {section["heading"] for section in page["sections"]} == {
        "Graduate admissions",
        "Application process",
    }
    assert "Follow us" not in {section["heading"] for section in page["sections"]}
    assert {value["label"] for value in page["audiences"]} >= {"applicants"}
    evidence_ids = {value["evidence_id"] for value in page["evidence"]}
    for field in (
        "source_url",
        "canonical_url",
        "canonical_family_url",
        "host",
        "path",
        "language",
        "locale",
        "page_type",
        "title",
        "purpose_summary",
    ):
        assert page["field_evidence"][field]
        assert set(page["field_evidence"][field]) <= evidence_ids

    assert {action["action_type"] for action in actions} == {"apply", "download", "search"}
    assert all(action["retrieval_eligible"] for action in actions)
    assert all(action["evidence"] for action in actions)
    assert not any("javascript:" in action["target_url"] for action in actions)
    assert not any(action["label"] in {"About", "Facebook"} for action in actions)
    assert set(page["action_ids"]) == {action["action_id"] for action in actions}
    assert stats["raw_action_candidates"] >= len(actions)


def test_arabic_page_card_preserves_arabic_sections_and_classifies_actions(tmp_path: Path):
    html_path = tmp_path / "arabic.html"
    html_path.write_text(
        """
        <html lang="ar" dir="rtl"><body><main>
          <h1>القبول في جامعة محمد بن زايد للذكاء الاصطناعي</h1>
          <p>يمكن للطلاب مراجعة متطلبات القبول وإرسال طلب الالتحاق بالجامعة.</p>
          <h2>كيفية التقديم</h2>
          <a href="/ar/apply">قدّم الآن</a>
          <a href="/ar/contact">تواصل معنا</a>
        </main></body></html>
        """,
        encoding="utf-8",
    )
    url = "https://mbzuai.ac.ae/ar/study/admissions"
    metadata = _metadata(
        url,
        html_path,
        title="القبول في جامعة محمد بن زايد للذكاء الاصطناعي",
        description="يمكن للطلاب مراجعة متطلبات القبول وإرسال طلب الالتحاق بالجامعة.",
        language="ar",
        meta_tags={"og:locale": "ar"},
    )
    draft = extract_page_draft(
        url,
        metadata,
        official_hosts={"mbzuai.ac.ae"},
    )
    pages, actions, _ = finalize_page_drafts([draft])

    assert pages[0]["language"] == "ar"
    assert [section["heading"] for section in pages[0]["sections"]] == [
        "القبول في جامعة محمد بن زايد للذكاء الاصطناعي",
        "كيفية التقديم",
    ]
    assert {value["label"] for value in pages[0]["audiences"]} >= {
        "applicants",
        "students",
    }
    assert {action["action_type"] for action in actions} == {"apply", "contact"}


def test_template_actions_are_retained_but_excluded_from_retrieval(tmp_path: Path):
    html_path = tmp_path / "template-action.html"
    html_path.write_text(
        """
        <html><body>
          <header><a href="https://apply.mbzuai.ac.ae/ApplicantPortal">Apply now</a></header>
          <main><h1>Program page</h1><p>This page explains one of the academic programs at MBZUAI.</p></main>
        </body></html>
        """,
        encoding="utf-8",
    )
    url = "https://mbzuai.ac.ae/study/program"
    draft = extract_page_draft(
        url,
        _metadata(url, html_path),
        official_hosts={"mbzuai.ac.ae", "apply.mbzuai.ac.ae"},
    )

    pages, actions, stats = finalize_page_drafts([draft])

    assert len(actions) == 1
    assert actions[0]["action_type"] == "apply"
    assert actions[0]["is_template"] is True
    assert actions[0]["retrieval_eligible"] is False
    assert pages[0]["action_ids"] == [actions[0]["action_id"]]
    assert pages[0]["retrieval_action_ids"] == []
    assert stats["retained_template_actions"] == 1
    assert stats["retrieval_eligible_actions"] == 0


def test_document_revision_index_does_not_bind_ambiguous_canonical_urls(tmp_path: Path):
    first_md = tmp_path / "first.md"
    second_md = tmp_path / "second.md"
    first_md.write_text("# First", encoding="utf-8")
    second_md.write_text("# Second", encoding="utf-8")
    first_url = "https://mbzuai.ac.ae/first"
    second_url = "https://mbzuai.ac.ae/second"
    shared_bad_canonical = "https://mbzuai.ac.ae/"
    sources = [
        _inventory_document(first_url, first_md, canonical_url=shared_bad_canonical),
        _inventory_document(second_url, second_md, canonical_url=shared_bad_canonical),
    ]

    documents, url_index, web_ids = build_document_revisions(sources)

    assert len(documents) == 2
    assert len(web_ids) == 2
    assert shared_bad_canonical not in url_index
    assert first_url in url_index
    assert second_url in url_index


def test_representation_schema_is_valid_and_reserves_lossless_evidence_units():
    schema = representation_v2_json_schema()
    Draft202012Validator.check_schema(schema)
    assert schema["$defs"]["evidence_unit"]["required"] == [
        "evidence_unit_id",
        "document_revision_id",
        "text",
        "structural_locator",
        "content_sha256",
    ]
    assert schema["properties"]["schema_version"]["const"] == REPRESENTATION_V2_SCHEMA_VERSION


def test_representation_formatter_enforces_exact_coverage_and_writes_artifacts(tmp_path: Path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    urls = ["https://mbzuai.ac.ae/one", "https://mbzuai.ac.ae/ar/two"]
    metadata = {}
    inventory_documents = []
    for index, url in enumerate(urls, start=1):
        html_path = source_dir / f"page-{index}.html"
        markdown_path = source_dir / f"page-{index}.md"
        language = "ar" if "/ar/" in url else "en"
        title = "الصفحة الثانية" if language == "ar" else "First page"
        purpose = (
            "هذه صفحة للطلاب وتحتوي على معلومات موثقة حول القبول والتقديم."
            if language == "ar"
            else "This evidence-backed page explains the admissions process for applicants."
        )
        html_path.write_text(
            f"<html lang='{language}'><body><main><h1>{title}</h1>"
            f"<p>{purpose}</p><a href='/apply'>Apply now</a></main></body></html>",
            encoding="utf-8",
        )
        markdown_path.write_text(f"# {title}\n\n{purpose}\n", encoding="utf-8")
        metadata[url] = _metadata(
            url,
            html_path,
            title=title,
            description=purpose,
            language=language,
            meta_tags={"og:locale": language},
        )
        inventory_documents.append(_inventory_document(url, markdown_path, language=language, title=title))

    inventory_path = source_dir / "prepared_corpus_inventory.json"
    metadata_path = source_dir / "prepared_canonical_page_metadata.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "kind": "representation_neutral_corpus_inventory",
                "document_count": len(inventory_documents),
                "documents": inventory_documents,
            }
        ),
        encoding="utf-8",
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    config = {
        "formatter": {
            "representation_v2": {
                "inventory_file": str(inventory_path),
                "page_metadata_file": str(metadata_path),
                "expected_document_count": 2,
                "expected_page_count": 2,
                "expected_web_document_count": 2,
                "maximum_workers": 2,
                "template_minimum_page_count": 20,
                "template_minimum_page_ratio": 0.5,
            }
        }
    }
    ctx = StageContext(
        run_id="representation-test",
        project_name="representation-test",
        config=config,
        work_dir=tmp_path / "run",
        stage_definition={"type": "formatter", "plugin": "representation_v2"},
        stage_id="extract_page_cards_and_actions",
    )

    result = asyncio.run(RepresentationV2Formatter().execute(ctx))

    assert result.status is StageStatus.COMPLETED, result.error_message
    bundle = json.loads(Path(result.outputs["representation_v2_bundle_file"]).read_text())
    coverage = json.loads(
        Path(result.outputs["representation_v2_coverage_report_file"]).read_text()
    )
    assert bundle["kind"] == REPRESENTATION_V2_KIND
    assert len(bundle["documents"]) == 2
    assert len(bundle["page_cards"]) == 2
    assert coverage["passed"] is True
    assert all(coverage["gates"].values())
    assert bundle["stats"]["chunking_performed"] is False
    assert bundle["stats"]["embedding_performed"] is False
    assert bundle["stats"]["indexing_performed"] is False
    Draft202012Validator(representation_v2_json_schema()).validate(bundle)

    broken = copy.deepcopy(bundle)
    broken["page_cards"][0]["field_evidence"]["title"] = ["missing-evidence"]
    broken_report = validate_representation_v2(
        broken,
        expected_corpus_record_ids={value["record_id"] for value in inventory_documents},
        expected_page_urls=set(urls),
        expected_web_revision_ids={
            value["document_revision_id"] for value in bundle["documents"]
        },
    )
    assert broken_report["passed"] is False
    assert broken_report["counts"]["issue_codes"]["page_field_evidence"] == 1


def test_low_level_bundle_coverage_is_exact_and_reciprocal(tmp_path: Path):
    url = "https://mbzuai.ac.ae/page"
    html_path = tmp_path / "page.html"
    markdown_path = tmp_path / "page.md"
    html_path.write_text(
        "<html><body><main><h1>Page</h1><p>A complete source paragraph for the Page Card.</p></main></body></html>",
        encoding="utf-8",
    )
    markdown_path.write_text("# Page\n\nA complete source paragraph.", encoding="utf-8")
    source = _inventory_document(url, markdown_path)
    documents, index, web_ids = build_document_revisions([source])
    draft = extract_page_draft(
        url,
        _metadata(url, html_path),
        official_hosts={"mbzuai.ac.ae"},
    )
    pages, actions, stats = finalize_page_drafts([draft])
    link_page_cards_to_revisions(pages, index)
    attach_page_links_to_documents(documents, pages)
    bundle = {
        "schema_version": REPRESENTATION_V2_SCHEMA_VERSION,
        "kind": REPRESENTATION_V2_KIND,
        "generated_at": now_iso(),
        "source_snapshot": {},
        "documents": documents,
        "page_cards": pages,
        "actions": actions,
        "stats": stats,
    }

    report = validate_representation_v2(
        bundle,
        expected_corpus_record_ids={source["record_id"]},
        expected_page_urls={url},
        expected_web_revision_ids=web_ids,
    )

    assert report["passed"] is True
    assert report["counts"]["linked_web_document_revisions"] == 1
    assert documents[0]["page_card_ids"] == [pages[0]["page_card_id"]]
