"""Deterministic topical-coherence checks for generated evidence bundles."""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations
from typing import Mapping, Sequence

from ._util import text_tokens
from .corpus import CorpusRelationship, EvidenceUnit, SourceRecord


_HEADING = re.compile(r"(?:^|\s)#{1,6}\s+")
_STOPWORDS = frozenset(
    """
    about admissions all also and application applications are can does for from graduate
    have how into mbzuai more not only our page program programmes source section student
    students studies study that the their them there they this title type undergraduate
    university url webpage what when where which who why will with you your
    available education journey new ready start
    leadership meet team teams overview programs system systems
    إلى الاصطناعي التي الذي الذكاء الجامعة الطلبة الطلاب البرنامج بن جامعة عن على في
    ما محمد مع من هذه هذا هل هو هي و أو
    """.split()
)
_GENERIC_RELATIONS = frozenset({"belongs_to_source"})


def relationship_index(
    relationships: Sequence[CorpusRelationship],
) -> dict[frozenset[str], frozenset[str]]:
    grouped: dict[frozenset[str], set[str]] = defaultdict(set)
    for edge in relationships:
        grouped[frozenset((edge.source_id, edge.target_id))].add(edge.relation)
    return {key: frozenset(value) for key, value in grouped.items()}


def topical_coherence(
    left: EvidenceUnit,
    right: EvidenceUnit,
    sources: Mapping[str, SourceRecord],
    relations: Mapping[frozenset[str], frozenset[str]],
) -> float:
    content = content_overlap(left, right)
    sections = _jaccard(_section_tokens(left), _section_tokens(right))
    relation = _relation_score(left, right, relations)
    if left.source_id == right.source_id:
        return round(max(content, sections, relation), 6)
    return round(max(content, sections, relation), 6)


def content_overlap(left: EvidenceUnit, right: EvidenceUnit) -> float:
    return _jaccard(_content_tokens(left), _content_tokens(right))


def content_terms(unit: EvidenceUnit) -> frozenset[str]:
    return _content_tokens(unit)


def topic_terms(value: str) -> frozenset[str]:
    return _tokens(value)


def section_overlap(left: EvidenceUnit, right: EvidenceUnit) -> float:
    return _jaccard(_section_tokens(left), _section_tokens(right))


def comparison_compatible(left, right, relations) -> bool:
    if section_overlap(left, right) >= 0.25 or content_overlap(left, right) >= 0.6:
        return True
    if left.kind in {"table", "database"} and right.kind == left.kind:
        return True
    return _relation_score(left, right, relations) >= 0.9


def coherent_pair(
    left: EvidenceUnit,
    right: EvidenceUnit,
    sources: Mapping[str, SourceRecord],
    relations: Mapping[frozenset[str], frozenset[str]],
) -> bool:
    left_person = _person_identity(left)
    right_person = _person_identity(right)
    if left_person and right_person and not (left_person & right_person):
        return False
    score = topical_coherence(left, right, sources, relations)
    if {left.kind, right.kind} <= {"action", "page_card"}:
        return score >= 0.4
    if "media" in {left.kind, right.kind}:
        return score >= 0.12
    return score >= 0.25


def coherent_bundle(
    units: Sequence[EvidenceUnit],
    sources: Mapping[str, SourceRecord],
    relations: Mapping[frozenset[str], frozenset[str]],
) -> bool:
    if len(units) < 2:
        return True
    pairs = tuple(combinations(units, 2))
    if all(coherent_pair(left, right, sources, relations) for left, right in pairs):
        return True
    return _explicit_graph_connected(units, relations)


def _content_tokens(unit: EvidenceUnit) -> frozenset[str]:
    match = _HEADING.search(unit.text)
    content = unit.text[match.end() :] if match else unit.text
    return _tokens(content)


def _section_tokens(unit: EvidenceUnit) -> frozenset[str]:
    path = unit.metadata.get("section_path") or ()
    values = (path,) if isinstance(path, str) else tuple(path)
    return _tokens(values[-1] if values else "")


def _tokens(value) -> frozenset[str]:
    return text_tokens(value) - _STOPWORDS


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _relation_score(left, right, relations) -> float:
    kinds = relations.get(frozenset((left.evidence_id, right.evidence_id)), ())
    meaningful = set(kinds) - _GENERIC_RELATIONS
    if not meaningful:
        return 0.0
    if meaningful == {"described_by_page"}:
        return 0.9 if {left.kind, right.kind} <= {"action", "page_card"} else 0.0
    return 0.95


def _explicit_graph_connected(units, relations) -> bool:
    reached = {units[0].evidence_id}
    pending = [units[0]]
    expected = {item.evidence_id for item in units}
    while pending:
        current = pending.pop()
        for candidate in units:
            if candidate.evidence_id in reached:
                continue
            if _relation_score(current, candidate, relations) >= 0.9:
                reached.add(candidate.evidence_id)
                pending.append(candidate)
    return reached == expected


def _person_identity(unit: EvidenceUnit) -> frozenset[str]:
    links = re.findall(
        r"/(?:faculty-directory|study/faculty)/([^/?#)]+)",
        unit.text,
        flags=re.IGNORECASE,
    )
    if len(links) == 1:
        return _tokens(links[0].replace("-", " "))
    path = unit.metadata.get("section_path") or ()
    values = (path,) if isinstance(path, str) else tuple(path)
    terminal = str(values[-1]) if values else ""
    words = re.findall(r"[^\W\d_][\w.'’-]*", terminal, flags=re.UNICODE)
    if 2 <= len(words) <= 4 and all(word[:1].isupper() for word in words):
        return _tokens(terminal)
    return frozenset()
