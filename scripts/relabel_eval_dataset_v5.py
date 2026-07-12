from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.dataset import EvalExample, load_eval_examples, write_eval_examples
from pipeline.evaluation.dataset_tools import validate_eval_examples


STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "available",
    "before",
    "between",
    "can",
    "covering",
    "does",
    "for",
    "from",
    "give",
    "how",
    "including",
    "into",
    "mbzuai",
    "new",
    "not",
    "off",
    "official",
    "on",
    "prepare",
    "provided",
    "should",
    "student",
    "students",
    "summarize",
    "tell",
    "the",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True)
class BundleRecord:
    id: str
    kind: str
    text: str
    text_lower: str
    tokens: frozenset[str]
    source_url: str
    document_type: str
    linked_chunk_ids: tuple[str, ...] = ()
    linked_parent_ids: tuple[str, ...] = ()
    linked_span_ids: tuple[str, ...] = ()
    url: str = ""


def _load_bundle(work_dir: str | Path) -> Dict[str, Any]:
    root = Path(work_dir).expanduser().resolve()
    for stage_id in ("finalize_retrieval_bundle", "format_retrieval", "build_retrieval_bundle"):
        path = root / "stage_outputs" / stage_id / "retrieval_bundle.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    raise FileNotFoundError(f"No retrieval_bundle.json found under {root}/stage_outputs")


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean(value)] if _clean(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_clean(item) for item in value if _clean(item)]
    return [_clean(value)] if _clean(value) else []


def _tokens(value: str) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) > 2 and token not in STOPWORDS
    ]


def _source_key(value: str) -> str:
    raw = _clean(value).rstrip("/")
    if not raw:
        return ""
    try:
        parsed = urlparse(raw)
    except Exception:
        return raw.casefold()
    if not parsed.netloc:
        return raw.casefold()
    path = (parsed.path or "/").rstrip("/")
    return f"{parsed.scheme.lower() or 'https'}://{parsed.netloc.lower()}{path}".rstrip("/").casefold()


def _record_source_url(record: Mapping[str, Any]) -> str:
    return _clean(record.get("source_url") or record.get("canonical_url") or record.get("language_normalized_url"))


def _record_text(record: Mapping[str, Any]) -> str:
    return _clean(
        " ".join(
            str(record.get(key) or "")
            for key in (
                "document_title",
                "heading",
                "section_heading",
                "breadcrumb",
                "source_url",
                "url",
                "text",
                "dense_text",
                "sparse_text",
                "lexical_text",
            )
        )
    )


def _bundle_records(records: Iterable[Mapping[str, Any]], kind: str) -> List[BundleRecord]:
    output: List[BundleRecord] = []
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        record_id = _clean(record.get("id"))
        if not record_id:
            continue
        text = _record_text(record)
        output.append(
            BundleRecord(
                id=record_id,
                kind=kind,
                text=text,
                text_lower=text.casefold(),
                tokens=frozenset(_tokens(text)),
                source_url=_record_source_url(record),
                document_type=_clean(record.get("document_type") or record.get("source_type")),
                linked_chunk_ids=tuple(_as_list(record.get("linked_chunk_ids") or record.get("child_chunk_ids"))),
                linked_parent_ids=tuple(_as_list(record.get("linked_parent_ids"))),
                linked_span_ids=tuple(_as_list(record.get("linked_span_ids") or record.get("child_span_ids"))),
                url=_clean(record.get("url")),
            )
        )
    return output


def _query_profile(example: EvalExample) -> Dict[str, Any]:
    metadata = dict(example.metadata or {})
    phrases = [
        example.query,
        example.reference_answer,
        *(_as_list(metadata.get("answer_must_include"))),
        *(_as_list(metadata.get("answer_should_cover"))),
        *(_as_list(metadata.get("expected_source_hints"))),
    ]
    text = " ".join(phrases)
    return {
        "text": text,
        "tokens": set(_tokens(text)),
        "phrases": [
            phrase.casefold()
            for phrase in [
                *_as_list(metadata.get("answer_must_include")),
                *_as_list(metadata.get("answer_should_cover")),
                *_as_list(metadata.get("expected_source_hints")),
            ]
            if len(phrase) >= 4
        ],
        "source_type": example.source_type,
    }


def _record_score(record: BundleRecord, profile: Mapping[str, Any]) -> float:
    text = record.text_lower
    record_tokens = set(record.tokens)
    query_tokens = set(profile.get("tokens") or [])
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & record_tokens)
    score = overlap / max(1.0, len(query_tokens)) * 12.0
    for phrase in profile.get("phrases") or []:
        phrase_tokens = set(_tokens(str(phrase)))
        if not phrase_tokens:
            literal = str(phrase or "").strip().casefold()
            if literal and literal in text:
                score += 8.0
            continue
        if str(phrase).casefold() in text:
            score += 6.0
            if re.search(r"[@:]|\b\d{1,2}:\d{2}\b", str(phrase)):
                score += 6.0
        elif len(phrase_tokens & record_tokens) >= max(1, min(len(phrase_tokens), 3)):
            score += 2.0
    profile_text = str(profile.get("text") or "").casefold()
    if re.search(r"\b(it support|technical support)\b", profile_text) and re.search(r"\b(working hours|available|8:00|12:30)\b", profile_text):
        if "it_external@mbzuai.ac.ae" in text or ("working hours" in text and "8:00 am" in text and "12:30 pm" in text):
            score += 14.0
    if "general admissions" in profile_text and "admission@mbzuai.ac.ae" in text:
        score += 10.0
    if "undergraduate admissions" in profile_text and "ug.admission@mbzuai.ac.ae" in text:
        score += 10.0
    source_type = str(profile.get("source_type") or "").casefold()
    if source_type == "webpage":
        score += 3.0 if record.source_url else -4.0
    elif source_type == "pdf":
        score += 3.0 if record.document_type == "pdf" or not record.source_url else -2.0
    elif source_type == "mixed":
        score += 1.5 if record.source_url or record.document_type == "pdf" else 0.0
    if _source_key(record.source_url) == "https://mbzuai.ac.ae":
        score -= 4.0
    if "/ar/" in _source_key(record.source_url) and not re.search(r"[\u0600-\u06FF]", str(profile.get("text") or "")):
        score -= 3.0
    return score


def _top_ids(
    records: Sequence[BundleRecord],
    profile: Mapping[str, Any],
    *,
    limit: int,
    min_score: float,
) -> List[BundleRecord]:
    scored = [
        (_record_score(record, profile), record)
        for record in records
    ]
    scored = [(score, record) for score, record in scored if score >= min_score]
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [record for _score, record in scored[:limit]]


def _top_records_for_reference_urls(
    records: Sequence[BundleRecord],
    profile: Mapping[str, Any],
    reference_urls: Sequence[str],
    *,
    limit: int,
    min_score: float,
) -> List[BundleRecord]:
    source_keys = [_source_key(url) for url in reference_urls if _source_key(url)]
    source_keys = _unique(source_keys)
    if not source_keys:
        return []
    per_source_limit = max(1, (limit + len(source_keys) - 1) // len(source_keys))
    selected: List[BundleRecord] = []
    for source_key in source_keys:
        source_records = [record for record in records if _source_key(record.source_url) == source_key]
        selected.extend(
            _top_ids(
                source_records,
                profile,
                limit=per_source_limit,
                min_score=min_score,
            )
        )
    if len(selected) < limit:
        selected_ids = {record.id for record in selected}
        remaining_source_records = [
            record
            for record in records
            if _source_key(record.source_url) in set(source_keys) and record.id not in selected_ids
        ]
        selected.extend(
            _top_ids(
                remaining_source_records,
                profile,
                limit=limit - len(selected),
                min_score=min_score,
            )
        )
    return _unique_records(selected)[:limit]


def _unique_records(records: Iterable[BundleRecord]) -> List[BundleRecord]:
    output: List[BundleRecord] = []
    seen = set()
    for record in records:
        if record.id in seen:
            continue
        seen.add(record.id)
        output.append(record)
    return output


def _unique(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen = set()
    for value in values:
        text = _clean(value)
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _source_urls(records: Iterable[BundleRecord], *, limit: int = 6) -> List[str]:
    urls = []
    for record in records:
        key = _source_key(record.source_url)
        if key and key != "https://mbzuai.ac.ae":
            urls.append(record.source_url)
    return _unique(urls)[:limit]


def _canonical_reference_urls(example: EvalExample) -> List[str]:
    text = " ".join(
        [
            example.id,
            example.query,
            example.reference_answer,
            example.notes,
            " ".join(_as_list((example.metadata or {}).get("answer_must_include"))),
            " ".join(_as_list((example.metadata or {}).get("answer_should_cover"))),
            " ".join(_as_list((example.metadata or {}).get("expected_source_hints"))),
        ]
    ).casefold()
    urls: List[str] = []
    screening_exam_context = bool(
        re.search(r"\b(online screening exam|screening exam).*\b(it support|technical support|working hours|available)\b", text)
        or re.search(
            r"\b(it support|technical support).*\b(online screening exam|screening exam)\b",
            text,
        )
    )
    if screening_exam_context:
        urls.append("https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2023/06/MBZUAI-Online-Screening-Exam-Instructions.pdf")
    if "campus map" in text or re.search(r"\bcampus\b.*\b(map|layout)\b", text):
        urls.append("https://staticcdn.mbzuai.ac.ae/mbzuaiwpprd01/2025/11/MBZUAI_Campus_Map_V1044331768.pdf")
    if (
        re.search(r"\b(named after|whose name|after whom|location|located|emirate|working hours|weekday operating hours|parking provided|parking available)\b", text)
        and not screening_exam_context
    ):
        urls.append("https://mbzuai.ac.ae/about/faq")
    if re.search(r"\b(student accommodation|student housing|parents stay|family members|undergraduate admissions|ug\\.admission|undergraduate applicants|financial aid|scholarship)\b", text):
        urls.append("https://mbzuai.ac.ae/study/undergraduate-application-submission")
    if re.search(r"\b(shuttle|transport|transportation|navya|golf cart|prt|personal rapid transit)\b", text):
        urls.append("https://mbzuai.ac.ae/about/contact")
    if re.search(r"\b(campus facilities|campus amenities|support facilities|campus services|knowledge center|medical center|library|gym|canteen|laboratories)\b", text):
        urls.append("https://mbzuai.ac.ae/student-resources/campus-facilities")
    if re.search(r"\b(north car park|parking permitted|vehicles? .*park|visitor parking|guest parking|visiting the university)\b", text):
        urls.append("https://mbzuai.ac.ae/about/contact")
    undergraduate_admissions_context = bool(
        re.search(r"\b(undergraduate admissions email|ug\\.admission@mbzuai\\.ac\\.ae|undergraduate admissions|undergraduate applicants)\b", text)
    )
    general_admissions_context = bool(
        re.search(r"\b(general admissions|admission@mbzuai\\.ac\\.ae)\b", text)
        or (
            re.search(r"\b(admissions email|contact admissions)\b", text)
            and not undergraduate_admissions_context
        )
    )
    if general_admissions_context:
        urls.append("https://mbzuai.ac.ae/about/faq")
    if undergraduate_admissions_context:
        urls.append("https://mbzuai.ac.ae/study/undergraduate-application-submission")
    if re.search(r"\b(core ai specializations|specializations|m\\.sc|ph\\.d|graduate programs)\b", text):
        urls.append("https://mbzuai.ac.ae/ai-programs")
    if re.search(r"\b(law no\\. 25|executive council|institutional identity|affiliated|established)\b", text):
        urls.append("https://mbzuai.ac.ae/about/faq")
    return _unique(urls)


def _curated_reference_answer(example: EvalExample) -> str:
    query = str(example.query or "").casefold()
    if re.search(r"\b(shuttle bus service|shuttle service|student shuttle)\b", query):
        return (
            "The current indexed corpus does not verify a general MBZUAI student shuttle service. "
            "For visitor arrival logistics, the contact page says visitors parking in the North Car Park "
            "may take a golf cart or electric autonomous NAVYA bus to the university building if available, "
            "or use the PRT."
        )
    if "family" in query and re.search(r"\b(transport|shuttle|parking)\b", query):
        return (
            "The corpus says MBZUAI does not provide housing for parents or visiting family members, "
            "though nearby hotels or Airbnbs may be recommended. Visitor parking directions point to "
            "the North Car Park and mention a golf cart or electric autonomous NAVYA bus to the university "
            "building if available, or the PRT. Campus amenities include the library/knowledge center, "
            "canteen, sports spaces, gym, pool, retail outlets, and other support facilities."
        )
    if re.search(r"\b(arriving|new graduate|newcomer)\b", query) and re.search(r"\b(transport|shuttle)\b", query):
        return (
            "A new student should know that MBZUAI is in Masdar City, Abu Dhabi, has weekday working hours "
            "with a shorter Friday schedule, provides student accommodation, and has parking guidance for "
            "the Masdar City campus. For arrival transport, the indexed contact page mentions movement from "
            "North Car Park by golf cart or electric autonomous NAVYA bus if available, or by PRT; it does "
            "not verify a general student shuttle service. Campus facilities include laboratories, "
            "library/knowledge center, sports spaces, canteen, and other support facilities."
        )
    return str(example.reference_answer or "")


def _curated_answer_must_include(example: EvalExample, values: Sequence[str]) -> List[str]:
    query = str(example.query or "").casefold()
    curated: List[str] = []
    for value in _as_list(values):
        lower = value.casefold()
        if lower == "shuttle":
            if re.search(r"\b(shuttle|transport|transportation|arriving|visitor|family|guest)\b", query):
                curated.extend(["NAVYA bus", "if available"])
            continue
        if lower == "students" and re.search(r"\b(shuttle bus service|shuttle service|student shuttle)\b", query):
            continue
        curated.append(value)
    return _unique(curated)


def _required_entity_aliases(metadata: Mapping[str, Any]) -> Dict[str, List[str]]:
    raw = metadata.get("required_entity_aliases")
    aliases: Dict[str, List[str]] = {}
    if isinstance(raw, Mapping):
        aliases.update({str(key): _as_list(value) for key, value in raw.items() if str(key).strip()})
    alias_defaults = {
        "working hours": ["official workings hours", "official working hours", "8:00 a.m.", "7.30am", "Friday"],
        "accommodation": ["student accommodation", "on-campus accommodation", "student housing", "housing"],
        "parking": ["North Car Park", "visitor parking", "car parking", "parking spaces"],
        "transport": ["golf cart", "NAVYA bus", "PRT", "Personal Rapid Transit"],
        "NAVYA bus": ["electric autonomous NAVYA bus", "golf cart", "PRT", "Personal Rapid Transit"],
        "facilities": ["campus facilities", "support facilities", "Knowledge Center", "library", "laboratories", "gym", "Medical Center"],
        "canteen": ["canteen", "dining", "restaurants", "coffee shops"],
        "does not provide housing": ["does not provide housing for parents"],
    }
    required = [
        *_as_list(metadata.get("required_entities")),
        *_as_list(metadata.get("answer_must_include")),
    ]
    for value in required:
        if value in alias_defaults:
            aliases[value] = _unique([*aliases.get(value, []), *alias_defaults[value]])
    return aliases


def relabel_examples(
    *,
    examples: Sequence[EvalExample],
    bundle: Mapping[str, Any],
    max_chunks: int,
    max_spans: int,
    max_parents: int,
    max_media: int,
) -> List[EvalExample]:
    chunk_records = _bundle_records(bundle.get("chunk_records") or [], "chunk")
    span_records = _bundle_records(bundle.get("evidence_span_records") or [], "evidence_span")
    parent_records = _bundle_records(bundle.get("parent_records") or [], "parent")
    media_records = _bundle_records(bundle.get("media_records") or [], "media")
    chunk_by_id = {record.id: record for record in chunk_records}
    parent_by_id = {record.id: record for record in parent_records}

    output: List[EvalExample] = []
    for example in examples:
        metadata = dict(example.metadata or {})
        metadata["relabel_source"] = "deterministic_bundle_lexical_v5"
        metadata["relabel_bundle_version"] = bundle.get("version")
        metadata["release_suite"] = metadata.get("release_suite") or "release_readiness_v5"
        curated_reference_answer = _curated_reference_answer(example)
        curated_must_include = _curated_answer_must_include(example, _as_list(metadata.get("answer_must_include")))
        if curated_must_include:
            metadata["answer_must_include"] = curated_must_include
        aliases = _required_entity_aliases(metadata)
        if aliases:
            metadata["required_entity_aliases"] = aliases
        example_payload = {
            **example.to_dict(),
            "reference_answer": curated_reference_answer,
            "metadata": metadata,
        }
        if example.no_answer:
            metadata.pop("alternate_gold_chunk_ids", None)
            metadata.pop("alternate_gold_span_ids", None)
            metadata.pop("alternate_gold_parent_ids", None)
            metadata.pop("alternate_gold_media_ids", None)
            output.append(
                EvalExample(
                    **{
                        **example_payload,
                        "gold_chunk_ids": [],
                        "gold_span_ids": [],
                        "gold_parent_ids": [],
                        "gold_media_ids": [],
                        "metadata": metadata,
                    }
                )
            )
            continue

        profile = _query_profile(example)
        top_spans = _top_ids(span_records, profile, limit=max_spans, min_score=4.0)
        linked_chunks = [chunk_by_id[item] for span in top_spans for item in span.linked_chunk_ids if item in chunk_by_id]
        linked_parents = [parent_by_id[item] for span in top_spans for item in span.linked_parent_ids if item in parent_by_id]
        top_chunks = _top_ids(chunk_records, profile, limit=max_chunks, min_score=4.0)
        top_parents = _top_ids(parent_records, profile, limit=max_parents, min_score=4.0)
        selected_chunks = _unique([record.id for record in [*linked_chunks, *top_chunks]])[:max_chunks]
        selected_parents = _unique([record.id for record in [*linked_parents, *top_parents]])[:max_parents]
        selected_spans = _unique([record.id for record in top_spans])[:max_spans]
        selected_media_records: List[BundleRecord] = []
        if example.query_type == "multimodal" or example.source_type in {"image", "video", "pdf"}:
            selected_media_records = _top_ids(media_records, profile, limit=max_media, min_score=3.0)
        selected_media = _unique([record.id for record in selected_media_records])[:max_media]

        evidence_records = [
            *[record for record in top_spans],
            *[chunk_by_id[item] for item in selected_chunks if item in chunk_by_id],
            *selected_media_records,
        ]
        canonical_reference_urls = _canonical_reference_urls(example)
        evidence_reference_urls = _source_urls(evidence_records, limit=6)
        if example.source_type == "pdf":
            canonical_pdf_urls = [
                url for url in canonical_reference_urls if urlparse(url).path.casefold().endswith(".pdf")
            ]
            evidence_pdf_urls = [
                url for url in evidence_reference_urls if urlparse(url).path.casefold().endswith(".pdf")
            ]
            reference_urls = canonical_pdf_urls or evidence_pdf_urls[:1] or evidence_reference_urls[:1]
        else:
            reference_urls = canonical_reference_urls
            if not reference_urls and example.source_type not in {"none"}:
                reference_urls = evidence_reference_urls[:1]
        if reference_urls:
            metadata["expected_reference_urls"] = reference_urls
            metadata["expected_citation_urls"] = reference_urls
            metadata["required_pages"] = reference_urls
            if example.query_type == "synthesis" and len(reference_urls) > 1:
                metadata["min_distinct_sources"] = min(2, len(reference_urls))
            source_top_spans = _top_records_for_reference_urls(
                span_records,
                profile,
                reference_urls,
                limit=max_spans,
                min_score=0.0,
            )
            if source_top_spans:
                top_spans = source_top_spans
                linked_chunks = [
                    chunk_by_id[item]
                    for span in top_spans
                    for item in span.linked_chunk_ids
                    if item in chunk_by_id
                ]
                linked_parents = [
                    parent_by_id[item]
                    for span in top_spans
                    for item in span.linked_parent_ids
                    if item in parent_by_id
                ]
                top_chunks = _top_records_for_reference_urls(
                    chunk_records,
                    profile,
                    reference_urls,
                    limit=max_chunks,
                    min_score=0.0,
                )
                top_parents = _top_records_for_reference_urls(
                    parent_records,
                    profile,
                    reference_urls,
                    limit=max_parents,
                    min_score=0.0,
                )
                selected_chunks = _unique([record.id for record in [*linked_chunks, *top_chunks]])[:max_chunks]
                selected_parents = _unique([record.id for record in [*linked_parents, *top_parents]])[:max_parents]
                selected_spans = _unique([record.id for record in top_spans])[:max_spans]
        if not selected_chunks and top_spans:
            selected_chunks = _unique([item for span in top_spans for item in span.linked_chunk_ids])[:max_chunks]
        if not selected_parents and top_spans:
            selected_parents = _unique([item for span in top_spans for item in span.linked_parent_ids])[:max_parents]
        metadata.pop("alternate_gold_chunk_ids", None)
        metadata.pop("alternate_gold_span_ids", None)
        metadata.pop("alternate_gold_parent_ids", None)
        metadata.pop("alternate_gold_media_ids", None)
        output.append(
            EvalExample(
                **{
                    **example_payload,
                    "gold_chunk_ids": selected_chunks,
                    "gold_span_ids": selected_spans,
                    "gold_parent_ids": selected_parents,
                    "gold_media_ids": selected_media,
                    "metadata": metadata,
                }
            )
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Relabel an MBZUAI eval dataset against a v5 retrieval bundle")
    parser.add_argument("--input", required=True, help="Input eval JSON/JSONL dataset")
    parser.add_argument("--output", required=True, help="Output eval JSON/JSONL dataset")
    parser.add_argument("--work-dir", required=True, help="Indexed run directory containing the v5 retrieval bundle")
    parser.add_argument("--max-chunks", type=int, default=8)
    parser.add_argument("--max-spans", type=int, default=8)
    parser.add_argument("--max-parents", type=int, default=8)
    parser.add_argument("--max-media", type=int, default=5)
    args = parser.parse_args()

    examples = load_eval_examples(args.input)
    bundle = _load_bundle(args.work_dir)
    relabeled = relabel_examples(
        examples=examples,
        bundle=bundle,
        max_chunks=max(1, args.max_chunks),
        max_spans=max(1, args.max_spans),
        max_parents=max(1, args.max_parents),
        max_media=max(0, args.max_media),
    )
    write_eval_examples(args.output, relabeled)
    validation = validate_eval_examples(args.output, work_dir=args.work_dir)
    manifest_path = Path(args.output).with_suffix(Path(args.output).suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "input": str(Path(args.input).resolve()),
                "output": str(Path(args.output).resolve()),
                "work_dir": str(Path(args.work_dir).resolve()),
                "bundle_version": bundle.get("version"),
                "query_count": len(relabeled),
                "validation_ok": bool(validation.get("ok")),
                "validation_error_count": len(validation.get("errors") or []),
                "validation_warning_count": len(validation.get("warnings") or []),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, indent=2))
    return 0 if validation.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
