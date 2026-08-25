"""Versioned contracts and coverage validation for Representation V2.

Representation V2 is intentionally pre-embedding. It establishes immutable
document revisions, evidence-backed Page Cards, typed webpage actions, and the
IDs/provenance that later evidence units and media vectors must reference.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPRESENTATION_V2_SCHEMA_VERSION = "mbzuai.representation.v2"
REPRESENTATION_V2_KIND = "page_aware_multimodal_representation"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_representation_id(prefix: str, *parts: Any) -> str:
    material = "\0".join(str(part or "").strip() for part in parts)
    digest = sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"{prefix}:{digest}"


def representation_v2_json_schema() -> Dict[str, Any]:
    """Return the JSON Schema 2020-12 contract emitted with every bundle."""

    evidence = {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence_id", "source_kind", "locator", "excerpt"],
        "properties": {
            "evidence_id": {"type": "string", "minLength": 1},
            "source_kind": {
                "type": "string",
                "enum": [
                    "html_text",
                    "html_attribute",
                    "meta_tag",
                    "crawl_metadata",
                    "url",
                ],
            },
            "locator": {"type": "string", "minLength": 1},
            "excerpt": {"type": "string", "minLength": 1},
            "attribute": {"type": "string"},
            "extraction_method": {"type": "string"},
        },
    }
    document_revision = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "document_id",
            "document_revision_id",
            "corpus_record_id",
            "revision_status",
            "source_type",
            "language",
            "source_locator",
            "markdown_path",
            "markdown_sha256",
            "page_card_ids",
            "media_references",
        ],
        "properties": {
            "document_id": {"type": "string", "minLength": 1},
            "document_revision_id": {"type": "string", "minLength": 1},
            "corpus_record_id": {"type": "string", "minLength": 1},
            "revision_status": {"const": "current"},
            "source_type": {"type": "string"},
            "language": {"type": "string"},
            "title": {"type": "string"},
            "source_locator": {
                "type": "object",
                "required": ["kind", "value"],
                "properties": {
                    "kind": {"type": "string", "minLength": 1},
                    "value": {"type": "string", "minLength": 1},
                },
            },
            "source_url": {"type": "string"},
            "canonical_url": {"type": "string"},
            "canonical_family_url": {"type": "string"},
            "source_file": {"type": "string"},
            "markdown_path": {"type": "string", "minLength": 1},
            "markdown_sha256": {
                "type": "string",
                "pattern": "^[a-f0-9]{64}$",
            },
            "content_statistics": {"type": "object"},
            "page_card_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "media_references": {"type": "array", "items": {"type": "object"}},
        },
    }
    section = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "section_id",
            "level",
            "heading",
            "html_locator",
            "evidence_id",
        ],
        "properties": {
            "section_id": {"type": "string", "minLength": 1},
            "level": {"type": "integer", "minimum": 1, "maximum": 6},
            "heading": {"type": "string", "minLength": 1},
            "html_locator": {"type": "string", "minLength": 1},
            "evidence_id": {"type": "string", "minLength": 1},
        },
    }
    semantic_label = {
        "type": "object",
        "additionalProperties": False,
        "required": ["label", "method", "evidence_ids"],
        "properties": {
            "label": {"type": "string", "minLength": 1},
            "method": {"type": "string", "minLength": 1},
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
        },
    }
    page_card = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "page_card_id",
            "schema_version",
            "source_url",
            "canonical_url",
            "canonical_family_url",
            "document_revision_id",
            "content_backed",
            "host",
            "path",
            "language",
            "locale",
            "page_type",
            "title",
            "purpose_summary",
            "purpose_method",
            "audiences",
            "topics",
            "sections",
            "action_ids",
            "retrieval_action_ids",
            "source_html",
            "crawl",
            "field_evidence",
            "evidence",
        ],
        "properties": {
            "page_card_id": {"type": "string", "minLength": 1},
            "schema_version": {"const": REPRESENTATION_V2_SCHEMA_VERSION},
            "source_url": {"type": "string", "minLength": 1},
            "canonical_url": {"type": "string", "minLength": 1},
            "canonical_family_url": {"type": "string"},
            "document_revision_id": {"type": ["string", "null"]},
            "content_backed": {"type": "boolean"},
            "host": {"type": "string", "minLength": 1},
            "path": {"type": "string"},
            "language": {"type": "string", "minLength": 1},
            "locale": {"type": "string"},
            "page_type": {"type": "string", "minLength": 1},
            "title": {"type": "string", "minLength": 1},
            "purpose_summary": {"type": "string", "minLength": 1},
            "purpose_method": {"type": "string", "minLength": 1},
            "audiences": {"type": "array", "items": semantic_label},
            "topics": {"type": "array", "items": semantic_label},
            "sections": {"type": "array", "items": section},
            "action_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "retrieval_action_ids": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
            "source_html": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "sha256", "byte_count"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
                    "byte_count": {"type": "integer", "minimum": 1},
                },
            },
            "crawl": {"type": "object"},
            "field_evidence": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
            },
            "evidence": {"type": "array", "items": evidence},
        },
    }
    action = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "action_id",
            "page_card_id",
            "label",
            "context_label",
            "action_type",
            "target_url",
            "canonical_target_url",
            "target_kind",
            "official_target",
            "element_role",
            "dom_region",
            "html_locator",
            "source_section_id",
            "source_section_heading",
            "opens_new_window",
            "authentication_requirement",
            "template_page_count",
            "template_page_ratio",
            "is_template",
            "retrieval_eligible",
            "evidence",
        ],
        "properties": {
            "action_id": {"type": "string", "minLength": 1},
            "page_card_id": {"type": "string", "minLength": 1},
            "label": {"type": "string", "minLength": 1},
            "context_label": {"type": "string"},
            "label_source": {"type": "string"},
            "action_type": {
                "type": "string",
                "enum": [
                    "navigate",
                    "apply",
                    "register",
                    "login",
                    "download",
                    "email",
                    "telephone",
                    "contact",
                    "search",
                    "submit_form",
                    "fragment_navigation",
                ],
            },
            "target_url": {"type": "string", "minLength": 1},
            "canonical_target_url": {"type": "string", "minLength": 1},
            "target_kind": {
                "type": "string",
                "enum": [
                    "official_page",
                    "external_page",
                    "download",
                    "email",
                    "telephone",
                    "fragment",
                    "form_endpoint",
                ],
            },
            "official_target": {"type": "boolean"},
            "element_role": {"type": "string", "minLength": 1},
            "dom_region": {
                "type": "string",
                "enum": ["main", "article", "navigation", "header", "footer", "aside", "body"],
            },
            "html_locator": {"type": "string", "minLength": 1},
            "source_section_id": {"type": ["string", "null"]},
            "source_section_heading": {"type": "string"},
            "form_method": {"type": "string"},
            "opens_new_window": {"type": "boolean"},
            "authentication_requirement": {
                "type": "string",
                "enum": ["explicit", "not_indicated"],
            },
            "template_page_count": {"type": "integer", "minimum": 1},
            "template_page_ratio": {"type": "number", "minimum": 0, "maximum": 1},
            "is_template": {"type": "boolean"},
            "retrieval_eligible": {"type": "boolean"},
            "evidence": {"type": "array", "minItems": 1, "items": evidence},
        },
    }
    evidence_unit = {
        "description": "Reserved contract for the subsequent lossless chunking stage.",
        "type": "object",
        "required": [
            "evidence_unit_id",
            "document_revision_id",
            "text",
            "structural_locator",
            "content_sha256",
        ],
        "properties": {
            "evidence_unit_id": {"type": "string", "minLength": 1},
            "document_revision_id": {"type": "string", "minLength": 1},
            "page_card_id": {"type": ["string", "null"]},
            "section_id": {"type": ["string", "null"]},
            "text": {"type": "string", "minLength": 1},
            "structural_locator": {"type": "object"},
            "content_sha256": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://mbzuai.ac.ae/schemas/representation-v2.json",
        "title": "MBZUAI Representation V2",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "kind",
            "generated_at",
            "source_snapshot",
            "documents",
            "page_cards",
            "actions",
            "stats",
        ],
        "properties": {
            "schema_version": {"const": REPRESENTATION_V2_SCHEMA_VERSION},
            "kind": {"const": REPRESENTATION_V2_KIND},
            "generated_at": {"type": "string", "format": "date-time"},
            "source_snapshot": {"type": "object"},
            "documents": {
                "type": "array",
                "items": {"$ref": "#/$defs/document_revision"},
            },
            "page_cards": {
                "type": "array",
                "items": {"$ref": "#/$defs/page_card"},
            },
            "actions": {
                "type": "array",
                "items": {"$ref": "#/$defs/page_action"},
            },
            "stats": {"type": "object"},
        },
        "$defs": {
            "evidence_locator": evidence,
            "document_revision": document_revision,
            "page_section": section,
            "semantic_label": semantic_label,
            "page_card": page_card,
            "page_action": action,
            "evidence_unit": evidence_unit,
        },
    }


def _nonempty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _append_issue(
    issues: List[Dict[str, Any]],
    *,
    code: str,
    message: str,
    record_id: str = "",
) -> None:
    issues.append(
        {
            "code": code,
            "message": message,
            **({"record_id": record_id} if record_id else {}),
        }
    )


def validate_representation_v2(
    bundle: Mapping[str, Any],
    *,
    expected_corpus_record_ids: Iterable[str] = (),
    expected_page_urls: Iterable[str] = (),
    expected_web_revision_ids: Iterable[str] = (),
    maximum_issue_samples: int = 200,
) -> Dict[str, Any]:
    """Validate schema-critical fields, evidence, linkage, and full coverage."""

    issues: List[Dict[str, Any]] = []
    documents = [dict(value) for value in bundle.get("documents") or [] if isinstance(value, Mapping)]
    page_cards = [dict(value) for value in bundle.get("page_cards") or [] if isinstance(value, Mapping)]
    actions = [dict(value) for value in bundle.get("actions") or [] if isinstance(value, Mapping)]

    if bundle.get("schema_version") != REPRESENTATION_V2_SCHEMA_VERSION:
        _append_issue(issues, code="schema_version", message="Unexpected schema_version")
    if bundle.get("kind") != REPRESENTATION_V2_KIND:
        _append_issue(issues, code="kind", message="Unexpected representation kind")

    document_ids = [str(value.get("document_revision_id") or "") for value in documents]
    corpus_record_ids = [str(value.get("corpus_record_id") or "") for value in documents]
    page_ids = [str(value.get("page_card_id") or "") for value in page_cards]
    action_ids = [str(value.get("action_id") or "") for value in actions]
    for label, values in (
        ("document_revision_id", document_ids),
        ("corpus_record_id", corpus_record_ids),
        ("page_card_id", page_ids),
        ("action_id", action_ids),
    ):
        duplicates = [value for value, count in Counter(values).items() if value and count > 1]
        if duplicates:
            _append_issue(
                issues,
                code="duplicate_id",
                message=f"Duplicate {label}: {duplicates[:5]}",
            )
        if any(not value for value in values):
            _append_issue(issues, code="missing_id", message=f"Missing {label}")

    document_id_set = set(document_ids)
    page_id_set = set(page_ids)
    action_by_id = {str(value.get("action_id") or ""): value for value in actions}
    page_actions: Dict[str, set[str]] = defaultdict(set)

    required_document_fields = (
        "document_id",
        "document_revision_id",
        "corpus_record_id",
        "source_locator",
        "markdown_path",
        "markdown_sha256",
        "page_card_ids",
        "media_references",
    )
    for document in documents:
        revision_id = str(document.get("document_revision_id") or "")
        for field in required_document_fields:
            if field not in document or document.get(field) in (None, ""):
                _append_issue(
                    issues,
                    code="document_schema",
                    message=f"Missing document field {field}",
                    record_id=revision_id,
                )
        sha = str(document.get("markdown_sha256") or "")
        if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
            _append_issue(
                issues,
                code="document_sha256",
                message="Invalid markdown_sha256",
                record_id=revision_id,
            )

    for action in actions:
        action_id = str(action.get("action_id") or "")
        page_id = str(action.get("page_card_id") or "")
        if page_id not in page_id_set:
            _append_issue(
                issues,
                code="action_page_link",
                message="Action references an unknown Page Card",
                record_id=action_id,
            )
        page_actions[page_id].add(action_id)
        for field in (
            "label",
            "action_type",
            "target_url",
            "canonical_target_url",
            "target_kind",
            "element_role",
            "dom_region",
            "html_locator",
        ):
            if not _nonempty_text(action.get(field)):
                _append_issue(
                    issues,
                    code="action_schema",
                    message=f"Missing action field {field}",
                    record_id=action_id,
                )
        evidence = [value for value in action.get("evidence") or [] if isinstance(value, Mapping)]
        evidence_ids = {str(value.get("evidence_id") or "") for value in evidence}
        if not evidence or "" in evidence_ids:
            _append_issue(
                issues,
                code="action_evidence",
                message="Action has missing/invalid evidence",
                record_id=action_id,
            )
        target = str(action.get("target_url") or "").lower()
        if bool(action.get("retrieval_eligible")) and target.startswith(
            ("javascript:", "data:")
        ):
            _append_issue(
                issues,
                code="unsafe_action",
                message="Retrieval-eligible action has an unsafe target",
                record_id=action_id,
            )

    required_field_evidence = (
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
    )
    linked_web_revision_ids = set()
    for page in page_cards:
        page_id = str(page.get("page_card_id") or "")
        for field in (
            "source_url",
            "canonical_url",
            "language",
            "page_type",
            "title",
            "purpose_summary",
            "host",
        ):
            if not _nonempty_text(page.get(field)):
                _append_issue(
                    issues,
                    code="page_schema",
                    message=f"Missing Page Card field {field}",
                    record_id=page_id,
                )
        revision_id = page.get("document_revision_id")
        if revision_id is not None:
            revision_id = str(revision_id)
            if revision_id not in document_id_set:
                _append_issue(
                    issues,
                    code="page_document_link",
                    message="Page Card references an unknown document revision",
                    record_id=page_id,
                )
            else:
                linked_web_revision_ids.add(revision_id)
        if bool(page.get("content_backed")) != bool(revision_id):
            _append_issue(
                issues,
                code="content_backed_flag",
                message="content_backed does not match document linkage",
                record_id=page_id,
            )

        evidence = [value for value in page.get("evidence") or [] if isinstance(value, Mapping)]
        evidence_ids = {str(value.get("evidence_id") or "") for value in evidence}
        field_evidence = page.get("field_evidence") if isinstance(page.get("field_evidence"), Mapping) else {}
        for field in required_field_evidence:
            references = [str(value) for value in field_evidence.get(field) or []]
            if not references or any(reference not in evidence_ids for reference in references):
                _append_issue(
                    issues,
                    code="page_field_evidence",
                    message=f"Field {field} lacks resolvable evidence",
                    record_id=page_id,
                )
        for collection_name in ("audiences", "topics"):
            for value in page.get(collection_name) or []:
                references = [str(item) for item in value.get("evidence_ids") or []]
                if not references or any(reference not in evidence_ids for reference in references):
                    _append_issue(
                        issues,
                        code="semantic_label_evidence",
                        message=f"{collection_name} label lacks evidence",
                        record_id=page_id,
                    )
        for section in page.get("sections") or []:
            if str(section.get("evidence_id") or "") not in evidence_ids:
                _append_issue(
                    issues,
                    code="section_evidence",
                    message="Section lacks resolvable evidence",
                    record_id=page_id,
                )

        listed_action_ids = {str(value) for value in page.get("action_ids") or []}
        if listed_action_ids != page_actions.get(page_id, set()):
            _append_issue(
                issues,
                code="page_action_coverage",
                message="Page action_ids do not exactly match owned actions",
                record_id=page_id,
            )
        retrieval_ids = {str(value) for value in page.get("retrieval_action_ids") or []}
        if not retrieval_ids <= listed_action_ids:
            _append_issue(
                issues,
                code="retrieval_action_link",
                message="retrieval_action_ids are not a subset of action_ids",
                record_id=page_id,
            )
        for action_id in retrieval_ids:
            if not bool(action_by_id.get(action_id, {}).get("retrieval_eligible")):
                _append_issue(
                    issues,
                    code="retrieval_action_flag",
                    message="Page lists a non-eligible retrieval action",
                    record_id=action_id,
                )

    document_page_links: Dict[str, set[str]] = defaultdict(set)
    for page in page_cards:
        revision_id = page.get("document_revision_id")
        if revision_id:
            document_page_links[str(revision_id)].add(str(page.get("page_card_id") or ""))
    for document in documents:
        revision_id = str(document.get("document_revision_id") or "")
        if set(document.get("page_card_ids") or []) != document_page_links.get(revision_id, set()):
            _append_issue(
                issues,
                code="document_page_coverage",
                message="Document page_card_ids do not exactly match linked Page Cards",
                record_id=revision_id,
            )

    expected_corpus = {str(value) for value in expected_corpus_record_ids if str(value)}
    expected_pages = {str(value) for value in expected_page_urls if str(value)}
    expected_web = {str(value) for value in expected_web_revision_ids if str(value)}
    actual_corpus = set(corpus_record_ids)
    actual_pages = {str(value.get("source_url") or "") for value in page_cards}
    if expected_corpus and actual_corpus != expected_corpus:
        _append_issue(
            issues,
            code="corpus_coverage",
            message=(
                f"Corpus coverage mismatch: missing={len(expected_corpus - actual_corpus)}, "
                f"unexpected={len(actual_corpus - expected_corpus)}"
            ),
        )
    if expected_pages and actual_pages != expected_pages:
        _append_issue(
            issues,
            code="page_coverage",
            message=(
                f"Page coverage mismatch: missing={len(expected_pages - actual_pages)}, "
                f"unexpected={len(actual_pages - expected_pages)}"
            ),
        )
    if expected_web and linked_web_revision_ids != expected_web:
        _append_issue(
            issues,
            code="web_document_coverage",
            message=(
                f"Web document coverage mismatch: missing={len(expected_web - linked_web_revision_ids)}, "
                f"unexpected={len(linked_web_revision_ids - expected_web)}"
            ),
        )

    issue_counts = Counter(str(issue.get("code") or "unknown") for issue in issues)
    gates = {
        "schema_version_valid": bundle.get("schema_version") == REPRESENTATION_V2_SCHEMA_VERSION,
        "all_corpus_documents_represented": not expected_corpus or actual_corpus == expected_corpus,
        "all_html_pages_represented": not expected_pages or actual_pages == expected_pages,
        "all_web_documents_linked": not expected_web or linked_web_revision_ids == expected_web,
        "unique_identifiers": not bool(issue_counts.get("duplicate_id") or issue_counts.get("missing_id")),
        "page_fields_evidenced": not bool(
            issue_counts.get("page_field_evidence")
            or issue_counts.get("semantic_label_evidence")
            or issue_counts.get("section_evidence")
        ),
        "actions_evidenced_and_safe": not bool(
            issue_counts.get("action_evidence") or issue_counts.get("unsafe_action")
        ),
        "document_page_links_complete": not bool(
            issue_counts.get("page_document_link")
            or issue_counts.get("document_page_coverage")
        ),
        "page_action_links_complete": not bool(
            issue_counts.get("action_page_link")
            or issue_counts.get("page_action_coverage")
            or issue_counts.get("retrieval_action_link")
            or issue_counts.get("retrieval_action_flag")
        ),
    }
    gates["passed"] = not issues and all(gates.values())
    return {
        "schema_version": REPRESENTATION_V2_SCHEMA_VERSION,
        "kind": "representation_v2_coverage_report",
        "created_at": now_iso(),
        "passed": bool(gates["passed"]),
        "gates": gates,
        "counts": {
            "documents": len(documents),
            "page_cards": len(page_cards),
            "content_backed_page_cards": sum(bool(page.get("content_backed")) for page in page_cards),
            "actions": len(actions),
            "retrieval_eligible_actions": sum(bool(action.get("retrieval_eligible")) for action in actions),
            "linked_web_document_revisions": len(linked_web_revision_ids),
            "issues": len(issues),
            "issue_codes": dict(sorted(issue_counts.items())),
        },
        "issue_samples": issues[: max(0, int(maximum_issue_samples))],
    }


def attach_page_links_to_documents(
    documents: Sequence[Dict[str, Any]],
    page_cards: Sequence[Mapping[str, Any]],
) -> None:
    """Populate reciprocal Page Card links after all pages have been mapped."""

    page_ids_by_revision: Dict[str, List[str]] = defaultdict(list)
    for page in page_cards:
        revision_id = str(page.get("document_revision_id") or "")
        page_id = str(page.get("page_card_id") or "")
        if revision_id and page_id:
            page_ids_by_revision[revision_id].append(page_id)
    for document in documents:
        revision_id = str(document.get("document_revision_id") or "")
        document["page_card_ids"] = sorted(set(page_ids_by_revision.get(revision_id, [])))
