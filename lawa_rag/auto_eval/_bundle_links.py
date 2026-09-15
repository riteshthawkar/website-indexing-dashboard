"""Internal graph-link extraction for retrieval-bundle adapters."""

from __future__ import annotations

from typing import Mapping

from ._bundle_records import source_identity
from ._util import clean_text, string_tuple
from .corpus import CorpusRelationship


_LINK_FIELDS = {
    "linked_chunk_ids": "supports_chunk",
    "child_chunk_ids": "contains_chunk",
    "linked_parent_ids": "belongs_to_parent",
    "linked_span_ids": "supports_span",
    "child_span_ids": "contains_span",
    "neighbor_ids": "neighbor",
    "media_ids": "has_media",
    "page_card_ids": "described_by_page",
}


def build_relationships(records, bundle, evidence, sources, actions):
    known = {item.evidence_id for item in evidence}
    known.update(item.source_id for item in sources)
    known.update(item.action_id for item in actions)
    output: dict[tuple[str, str, str], CorpusRelationship] = {}
    for _group, _kind, record in records:
        record_id = clean_text(record.get("id"))
        if record_id not in known:
            continue
        source_id, _revision_id = source_identity(record)
        _add(output, record_id, source_id, "belongs_to_source", known)
        for field, relation in _LINK_FIELDS.items():
            for target in string_tuple(record.get(field)):
                _add(output, record_id, target, relation, known)
    for edge in bundle.get("graph_edges") or bundle.get("relationships") or []:
        if not isinstance(edge, Mapping):
            continue
        source = clean_text(edge.get("source_id") or edge.get("source"))
        target = clean_text(edge.get("target_id") or edge.get("target"))
        relation = clean_text(edge.get("relation") or edge.get("type")) or "related_to"
        _add(output, source, target, relation, known)
    return tuple(
        sorted(output.values(), key=lambda item: (item.source_id, item.target_id, item.relation))
    )


def _add(output, source: str, target: str, relation: str, known: set[str]) -> None:
    if not source or not target or source == target or source not in known or target not in known:
        return
    key = (source, target, relation)
    output[key] = CorpusRelationship(source_id=source, target_id=target, relation=relation)
