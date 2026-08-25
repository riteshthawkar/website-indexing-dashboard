"""Validation helpers for the source-grounded multilingual MBZUAI benchmark."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

from pipeline.evaluation.dataset import EvalExample


_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_WORD_RE = re.compile(r"[\w\u0600-\u06ff]+", flags=re.UNICODE)


def normalize_evidence_text(value: Any) -> str:
    value = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(value.split()).casefold()


def contains_evidence_quote(source_text: str, quote: str) -> bool:
    normalized_quote = normalize_evidence_text(quote)
    return bool(normalized_quote) and normalized_quote in normalize_evidence_text(source_text)


def looks_like_language(value: str, language: str) -> bool:
    letters = [character for character in str(value or "") if character.isalpha()]
    if not letters:
        return False
    arabic_letters = sum(1 for character in letters if _ARABIC_RE.match(character))
    ratio = arabic_letters / float(len(letters))
    if language == "Arabic":
        # Arabic questions can legitimately contain long English program, paper,
        # job, or event titles. Require a real Arabic clause instead of rejecting
        # those mixed-script questions solely on a global character ratio.
        return arabic_letters >= 8 and (ratio >= 0.20 or arabic_letters >= 15)
    non_arabic_letters = len(letters) - arabic_letters
    return non_arabic_letters >= 8 and ratio <= 0.35


def normalized_query_key(value: str) -> str:
    return " ".join(_WORD_RE.findall(normalize_evidence_text(value)))


def _normalized_source_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    parsed = urlsplit(raw)
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.casefold(), parsed.netloc.casefold(), path, parsed.query, "")
    )


def source_identity_keys(example: EvalExample) -> Tuple[str, ...]:
    """Return stable identities used to keep a source out of multiple splits."""

    metadata = dict(example.metadata or {})
    keys = {
        f"source:{value}"
        for value in (
            *example.gold_document_revision_ids,
            *example.gold_page_card_ids,
            *example.gold_section_ids,
            *example.gold_action_ids,
            *example.gold_media_ids,
            *(str(value) for value in metadata.get("source_keys") or []),
        )
        if str(value)
    }
    for field in (
        "required_pages",
        "expected_reference_urls",
        "expected_citation_urls",
    ):
        for value in metadata.get(field) or []:
            normalized = _normalized_source_url(value)
            if normalized:
                keys.add(f"url:{normalized}")
    return tuple(sorted(keys))


def _source_connected_groups(
    examples: Sequence[EvalExample],
) -> List[List[EvalExample]]:
    """Connect examples that share any gold source or canonical source URL."""

    rows = list(examples)
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_by_key: Dict[str, int] = {}
    for index, row in enumerate(rows):
        for key in source_identity_keys(row):
            previous = first_by_key.setdefault(key, index)
            union(index, previous)

    grouped: Dict[int, List[EvalExample]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[find(index)].append(row)
    return [
        sorted(group, key=lambda row: row.id)
        for group in grouped.values()
    ]


def assign_stratified_splits(
    examples: Sequence[EvalExample],
    *,
    selection_ratio: float = 0.60,
    holdout_ratio: float = 0.25,
) -> List[EvalExample]:
    """Assign stable, exactly stratified, source-disjoint evaluation splits."""

    if selection_ratio <= 0.0 or holdout_ratio <= 0.0 or selection_ratio + holdout_ratio >= 1.0:
        raise ValueError("Split ratios must be positive and leave room for the regression split")
    grouped: Dict[Tuple[str, str, bool], List[EvalExample]] = defaultdict(list)
    for example in examples:
        grouped[(example.language, example.query_type, example.no_answer)].append(example)

    split_names = ("selection", "holdout", "regression")
    targets: Dict[str, Counter[Tuple[str, str, bool]]] = {
        split: Counter() for split in split_names
    }
    for key in sorted(grouped):
        count = len(grouped[key])
        selection_count = max(1, round(count * selection_ratio)) if count >= 3 else max(0, count - 1)
        holdout_count = max(1, round(count * holdout_ratio)) if count >= 3 else min(1, count)
        if selection_count + holdout_count >= count and count >= 3:
            selection_count = max(1, count - holdout_count - 1)
        targets["selection"][key] = selection_count
        targets["holdout"][key] = holdout_count
        targets["regression"][key] = count - selection_count - holdout_count

    strata = sorted(grouped)
    source_groups = _source_connected_groups(examples)
    multi_groups = [group for group in source_groups if len(group) > 1]
    singleton_rows = [group[0] for group in source_groups if len(group) == 1]
    multi_groups.sort(
        key=lambda group: (
            -len(group),
            hashlib.sha256(
                "\0".join(row.id for row in group).encode("utf-8")
            ).hexdigest(),
        )
    )

    def vector(group: Sequence[EvalExample]) -> Tuple[int, ...]:
        counts = Counter(
            (row.language, row.query_type, row.no_answer) for row in group
        )
        return tuple(counts[stratum] for stratum in strata)

    vectors = [vector(group) for group in multi_groups]
    remaining = [
        [targets[split][stratum] for stratum in strata]
        for split in split_names
    ]
    assignments: Dict[int, int] = {}
    failed_states: set[Tuple[int, Tuple[int, ...], Tuple[int, ...]]] = set()

    def place_group(position: int) -> bool:
        if position == len(multi_groups):
            singleton_counts = Counter(
                (row.language, row.query_type, row.no_answer)
                for row in singleton_rows
            )
            return all(
                sum(remaining[split_index][stratum_index] for split_index in range(3))
                == singleton_counts[stratum]
                for stratum_index, stratum in enumerate(strata)
            )
        state = (position, tuple(remaining[0]), tuple(remaining[1]))
        if state in failed_states:
            return False
        group = multi_groups[position]
        counts = vectors[position]
        group_key = "\0".join(row.id for row in group)
        split_order = sorted(
            range(3),
            key=lambda index: hashlib.sha256(
                f"mbzuai-multilingual-v2\0{group_key}\0{split_names[index]}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )
        for split_index in split_order:
            if any(
                count > remaining[split_index][stratum_index]
                for stratum_index, count in enumerate(counts)
            ):
                continue
            for stratum_index, count in enumerate(counts):
                remaining[split_index][stratum_index] -= count
            assignments[position] = split_index
            if place_group(position + 1):
                return True
            assignments.pop(position, None)
            for stratum_index, count in enumerate(counts):
                remaining[split_index][stratum_index] += count
        failed_states.add(state)
        return False

    if not place_group(0):
        raise ValueError(
            "Source-disjoint groups cannot satisfy the requested exact stratification"
        )

    split_by_id: Dict[str, str] = {}
    for position, group in enumerate(multi_groups):
        split = split_names[assignments[position]]
        split_by_id.update({row.id: split for row in group})

    singletons_by_stratum: Dict[
        Tuple[str, str, bool], List[EvalExample]
    ] = defaultdict(list)
    for row in singleton_rows:
        singletons_by_stratum[(row.language, row.query_type, row.no_answer)].append(row)
    for stratum_index, stratum in enumerate(strata):
        rows = sorted(
            singletons_by_stratum[stratum],
            key=lambda row: hashlib.sha256(
                f"mbzuai-multilingual-v2\0singleton\0{row.id}".encode("utf-8")
            ).hexdigest(),
        )
        slots = [
            (split_names[split_index], slot_index)
            for split_index in range(3)
            for slot_index in range(remaining[split_index][stratum_index])
        ]
        slots.sort(
            key=lambda slot: hashlib.sha256(
                f"mbzuai-multilingual-v2\0{stratum}\0{slot[0]}\0{slot[1]}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        if len(rows) != len(slots):
            raise RuntimeError("Internal split assignment accounting mismatch")
        for row, (split, _slot_index) in zip(rows, slots):
            split_by_id[row.id] = split

    output: List[EvalExample] = []
    for row in examples:
        metadata = {**dict(row.metadata or {}), "split": split_by_id[row.id]}
        output.append(replace(row, metadata=metadata).normalized())
    return sorted(output, key=lambda row: row.id)


def summarize_multilingual_coverage(examples: Sequence[EvalExample]) -> Dict[str, Any]:
    rows = list(examples)

    def counts(values: Iterable[Any]) -> Dict[str, int]:
        return dict(sorted(Counter(str(value) for value in values).items()))

    tags = [
        str(tag)
        for row in rows
        for tag in (dict(row.metadata or {}).get("benchmark_tags") or [])
        if str(tag)
    ]
    hosts = [
        str(host)
        for row in rows
        for host in (dict(row.metadata or {}).get("source_hosts") or [])
        if str(host)
    ]
    source_groups = _source_connected_groups(rows)
    source_group_split_leaks = 0
    for group in source_groups:
        splits = {
            str((row.metadata or {}).get("split") or "unassigned") for row in group
        }
        if len(splits) > 1:
            source_group_split_leaks += 1
    return {
        "query_count": len(rows),
        "language_counts": counts(row.language for row in rows),
        "query_type_counts": counts(row.query_type for row in rows),
        "source_type_counts": counts(row.source_type for row in rows),
        "split_counts": counts(dict(row.metadata or {}).get("split") or "unassigned" for row in rows),
        "answerability_counts": counts("no_answer" if row.no_answer else "answerable" for row in rows),
        "benchmark_tag_counts": counts(tags),
        "source_host_counts": counts(hosts),
        "with_evidence_quotes": sum(
            1 for row in rows if (dict(row.metadata or {}).get("evidence_quotes") or [])
        ),
        "with_page_cards": sum(1 for row in rows if row.gold_page_card_ids),
        "with_sections": sum(1 for row in rows if row.gold_section_ids),
        "with_actions": sum(1 for row in rows if row.gold_action_ids),
        "with_media": sum(1 for row in rows if row.gold_media_ids),
        "source_connected_group_count": len(source_groups),
        "source_group_split_leak_count": source_group_split_leaks,
    }


def validate_multilingual_coverage(
    examples: Sequence[EvalExample],
    *,
    minimum_query_count: int = 150,
    minimum_per_language: int = 70,
    minimum_no_answer_per_language: int = 10,
    minimum_cross_lingual: int = 12,
    minimum_navigation: int = 12,
    minimum_multimodal: int = 20,
    minimum_subdomain: int = 20,
    require_all_splits: bool = True,
) -> Dict[str, Any]:
    rows = list(examples)
    summary = summarize_multilingual_coverage(rows)
    errors: List[str] = []
    warnings: List[str] = []
    if len(rows) < minimum_query_count:
        errors.append(f"query_count {len(rows)} is below {minimum_query_count}")
    language_counts = Counter(row.language for row in rows)
    no_answer_by_language = Counter(row.language for row in rows if row.no_answer)
    for language in ("English", "Arabic"):
        if language_counts[language] < minimum_per_language:
            errors.append(f"{language} count {language_counts[language]} is below {minimum_per_language}")
        if no_answer_by_language[language] < minimum_no_answer_per_language:
            errors.append(
                f"{language} no-answer count {no_answer_by_language[language]} is below "
                f"{minimum_no_answer_per_language}"
            )
    tag_counts = Counter(
        str(tag)
        for row in rows
        for tag in (dict(row.metadata or {}).get("benchmark_tags") or [])
        if str(tag)
    )
    for tag, minimum in (
        ("cross_lingual", minimum_cross_lingual),
        ("navigation", minimum_navigation),
        ("multimodal", minimum_multimodal),
        ("subdomain", minimum_subdomain),
    ):
        if tag_counts[tag] < minimum:
            errors.append(f"{tag} count {tag_counts[tag]} is below {minimum}")
    if require_all_splits:
        for split in ("selection", "holdout", "regression"):
            if not any((row.metadata or {}).get("split") == split for row in rows):
                errors.append(f"split {split!r} is empty")
    for group in _source_connected_groups(rows):
        splits = sorted(
            {
                str((row.metadata or {}).get("split") or "unassigned")
                for row in group
            }
        )
        if len(splits) > 1:
            errors.append(
                "source-connected examples cross evaluation splits: "
                f"{', '.join(row.id for row in group)} -> {splits}"
            )

    query_keys: Dict[str, str] = {}
    for row in rows:
        key = normalized_query_key(row.query)
        if key in query_keys:
            errors.append(f"duplicate query text: {query_keys[key]} and {row.id}")
        query_keys[key] = row.id
        if not looks_like_language(row.query, row.language):
            errors.append(f"{row.id} query does not match declared language {row.language}")
        if not looks_like_language(row.reference_answer, row.language):
            warnings.append(f"{row.id} reference answer may not match declared language {row.language}")
        if not row.no_answer and not (row.metadata or {}).get("evidence_quotes"):
            errors.append(f"{row.id} has no evidence quotes")
        if row.no_answer and (row.metadata or {}).get("evidence_quotes"):
            errors.append(f"{row.id} is no-answer but contains evidence quotes")

    return {
        "ok": not errors,
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
    }


def validate_source_ids(
    examples: Sequence[EvalExample],
    *,
    representation_bundle: Mapping[str, Any],
    media_manifest: Mapping[str, Any],
    navigation_catalog: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    documents = {
        str(row.get("document_revision_id")): row
        for row in representation_bundle.get("documents") or []
        if isinstance(row, Mapping) and str(row.get("document_revision_id") or "")
    }
    pages = {
        str(row.get("page_card_id")): row
        for row in representation_bundle.get("page_cards") or []
        if isinstance(row, Mapping) and str(row.get("page_card_id") or "")
    }
    sections = {
        str(section.get("section_id")): section
        for page in pages.values()
        for section in (page.get("sections") or [])
        if isinstance(section, Mapping) and str(section.get("section_id") or "")
    }
    action_source = (
        navigation_catalog.get("actions")
        if isinstance(navigation_catalog, Mapping)
        else representation_bundle.get("actions")
    )
    actions = {
        str(row.get("action_id")): row
        for row in action_source or []
        if isinstance(row, Mapping) and str(row.get("action_id") or "")
    }
    media = {
        str(row.get("id")): row
        for row in media_manifest.get("items") or []
        if isinstance(row, Mapping) and str(row.get("id") or "")
    }
    known = {
        "gold_document_revision_ids": documents,
        "gold_page_card_ids": pages,
        "gold_section_ids": sections,
        "gold_action_ids": actions,
        "gold_media_ids": media,
    }
    errors: List[Dict[str, str]] = []
    for row in examples:
        for field_name, values_by_id in known.items():
            for value in getattr(row, field_name) or []:
                if value not in values_by_id:
                    errors.append({"id": row.id, "field": field_name, "value": value})
    return {"ok": not errors, "errors": errors}
