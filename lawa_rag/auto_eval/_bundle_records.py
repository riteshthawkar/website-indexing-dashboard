"""Internal conversion of retrieval records into sources and evidence."""

from __future__ import annotations

from typing import Any, Mapping

from ._util import clean_text, compact_mapping, stable_id, string_tuple
from .corpus import CorpusAction, EvidenceUnit, SourceRecord


RECORD_GROUPS = {
    "chunk_records": "chunk",
    "evidence_span_records": "evidence_span",
    "parent_records": "parent",
    "media_records": "media",
    "fact_records": "fact",
    "summary_records": "summary",
    "assertion_records": "assertion",
    "answer_records": "answer",
    "page_card_records": "page_card",
    "action_records": "action",
    "table_records": "table",
    "database_records": "database",
}

_METADATA_FIELDS = (
    "record_type",
    "document_revision_id",
    "document_id",
    "page_card_ids",
    "section_ids",
    "section_id",
    "section_path",
    "page_numbers",
    "media_id",
    "action_id",
    "authority_class",
    "source_last_seen",
    "updated_at",
)

_SOURCE_KIND_PRIORITY = {
    "page_card": 5,
    "parent": 4,
    "chunk": 4,
    "summary": 4,
    "fact": 3,
    "assertion": 3,
    "answer": 3,
    "evidence_span": 3,
    "table": 3,
    "database": 3,
    "media": 2,
    "action": 1,
}


def bundle_records(bundle: Mapping[str, Any]):
    for group, kind in RECORD_GROUPS.items():
        for raw in bundle.get(group) or []:
            if isinstance(raw, Mapping):
                yield group, kind, raw


def source_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    revision = clean_text(record.get("document_revision_id"))
    identity = clean_text(
        revision
        or record.get("document_id")
        or record.get("canonical_url")
        or record.get("source_url")
        or record.get("source_markdown_path")
        or record.get("source_file")
    )
    if not identity:
        identity = stable_id("source", record.get("id"), record.get("document_title"))
    return identity, revision or identity


def build_sources(records) -> tuple[SourceRecord, ...]:
    sources: dict[str, SourceRecord] = {}
    scores: dict[str, tuple[int, int, int]] = {}
    for _group, kind, record in records:
        candidate = _source_from_record(record)
        score = (
            _SOURCE_KIND_PRIORITY.get(kind, 2),
            int(bool(clean_text(record.get("document_title")))),
            len(candidate.title) + len(candidate.uri),
        )
        if score > scores.get(candidate.source_id, (-1, -1, -1)):
            sources[candidate.source_id] = candidate
            scores[candidate.source_id] = score
    return tuple(sorted(sources.values(), key=lambda item: item.source_id))


def build_evidence(records) -> tuple[EvidenceUnit, ...]:
    evidence: dict[str, EvidenceUnit] = {}
    for group, kind, record in records:
        candidate = _evidence_from_record(group, kind, record)
        if candidate is None:
            continue
        current = evidence.get(candidate.evidence_id)
        if current is None or len(candidate.text) > len(current.text):
            evidence[candidate.evidence_id] = candidate
    return tuple(sorted(evidence.values(), key=lambda item: item.evidence_id))


def build_actions(records) -> tuple[CorpusAction, ...]:
    actions: dict[str, CorpusAction] = {}
    for _group, kind, record in records:
        if kind != "action":
            continue
        action_id = clean_text(record.get("action_id") or record.get("id"))
        source_id, _revision_id = source_identity(record)
        if not action_id or not source_id:
            continue
        actions[action_id] = CorpusAction(
            action_id=action_id,
            source_id=source_id,
            label=clean_text(record.get("label") or record.get("raw_text")),
            action_type=clean_text(record.get("action_type")) or "navigate",
            target=clean_text(record.get("canonical_target_url") or record.get("target_url")),
            evidence_ids=(action_id,),
            metadata=compact_mapping(record, ("page_card_id", "section_id", "official_target")),
        )
    return tuple(sorted(actions.values(), key=lambda item: item.action_id))


def _source_type(record: Mapping[str, Any]) -> str:
    declared = clean_text(record.get("document_type") or record.get("source_type")).lower()
    uri = clean_text(record.get("source_url") or record.get("source_file")).lower()
    if "pdf" in declared or uri.endswith(".pdf"):
        return "pdf"
    if declared in {"database", "table", "sql", "csv"}:
        return "database"
    if uri.startswith(("http://", "https://")):
        return "webpage"
    return declared or "document"


def _source_from_record(record: Mapping[str, Any]) -> SourceRecord:
    source_id, revision_id = source_identity(record)
    uri = clean_text(
        record.get("canonical_url")
        or record.get("source_url")
        or record.get("source_file")
        or record.get("source_markdown_path")
    )
    return SourceRecord(
        source_id=source_id,
        revision_id=revision_id,
        source_type=_source_type(record),
        title=clean_text(record.get("document_title") or record.get("title"))
        or uri
        or source_id,
        uri=uri,
        language=clean_text(record.get("language")) or "und",
        authority=clean_text(record.get("authority_class")) or "source",
        updated_at=clean_text(record.get("updated_at") or record.get("source_last_seen")),
        metadata=compact_mapping(record, ("document_id", "document_revision_id")),
    )


def _record_text(record: Mapping[str, Any], kind: str) -> str:
    fields = ("text", "raw_text", "purpose_summary", "caption", "description", "value")
    text = next(
        (clean_text(record.get(field)) for field in fields if clean_text(record.get(field))),
        "",
    )
    if text or kind != "action":
        return text
    return clean_text(
        " ".join(
            str(record.get(key) or "")
            for key in ("label", "context_label", "action_type", "target_url")
        )
    )


def _locator(record: Mapping[str, Any]) -> str:
    parts = [
        clean_text(record.get("source_url") or record.get("source_file")),
        clean_text(record.get("breadcrumb") or record.get("heading")),
    ]
    pages = string_tuple(record.get("page_numbers"))
    if pages:
        parts.append("pages " + ",".join(pages))
    return " | ".join(value for value in parts if value)


def _evidence_from_record(group: str, kind: str, record: Mapping[str, Any]):
    text = _record_text(record, kind)
    record_id = clean_text(record.get("id") or record.get(f"{kind}_id"))
    if not record_id or len(text) < 12:
        return None
    source_id, _revision_id = source_identity(record)
    metadata = {"record_group": group, "record_id": record_id}
    metadata.update(compact_mapping(record, _METADATA_FIELDS))
    return EvidenceUnit(
        evidence_id=record_id,
        source_id=source_id,
        kind=kind,
        text=text,
        locator=_locator(record),
        language=clean_text(record.get("language")) or "und",
        section_id=clean_text(record.get("section_id") or record.get("section_key")),
        parent_id=clean_text(record.get("parent_id") or record.get("page_key")),
        metadata=metadata,
    )
