from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.google_genai import import_genai, import_genai_types
from pipeline.evaluation.dataset import EvalExample, write_eval_examples


DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_llm_generated_v1.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_llm_generated_v1.manifest.json"
DEFAULT_TOPICS = [
    "admissions and application process",
    "undergraduate admissions",
    "graduate programs and specializations",
    "campus location and visitor logistics",
    "student accommodation and family visitor policy",
    "parking and transport",
    "campus facilities and services",
    "working hours and support contacts",
    "institutional identity, legal basis, leadership, and governance",
    "research, faculty, labs, and academic life",
    "events, news, scholarships, and public engagement",
    "no-answer controls for unsupported offices, contacts, campuses, dates, or policies",
]
def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _slug(value: str, *, max_len: int = 48) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return cleaned[:max_len].strip("-") or "item"


def _stable_id(prefix: str, query: str) -> str:
    return f"{prefix}-{hashlib.sha1(query.encode('utf-8')).hexdigest()[:10]}"


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_text(value)] if _text(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_text(item) for item in value if _text(item)]
    return [_text(value)] if _text(value) else []


def _json_from_text(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM response did not contain a JSON object")


def _resolve_bundle_path(work_dir: str | Path) -> Path:
    run_dir = Path(work_dir).expanduser().resolve()
    candidates = [
        run_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json",
        run_dir / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json",
        run_dir / "retrieval_bundle.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No retrieval_bundle.json found under {run_dir}")


def _source_key(record: Mapping[str, Any]) -> str:
    return (
        _text(record.get("source_url"))
        or _text(record.get("document_title"))
        or _text(record.get("document_id"))
        or _text(record.get("source_markdown_path"))
    )


def _record_text(record: Mapping[str, Any], *, max_chars: int) -> str:
    text = _text(record.get("text") or record.get("dense_text") or record.get("lexical_text"))
    return text[:max_chars]


def _is_useful_record(record: Mapping[str, Any]) -> bool:
    text = _text(record.get("text") or record.get("dense_text") or "")
    if len(text) < 120:
        return False
    title = _text(record.get("document_title"))
    url = _text(record.get("source_url"))
    if not url and not title:
        return False
    noisy_markers = ("cookie", "javascript", "nav menu", "skip to content")
    return not any(marker in text.lower() for marker in noisy_markers)


def _load_source_packs(
    *,
    work_dir: str | Path,
    max_sources: int,
    max_records_per_source: int,
    max_record_chars: int,
) -> List[Dict[str, Any]]:
    bundle = _load_json(_resolve_bundle_path(work_dir))
    records = []
    for record_type in ("chunk_records", "parent_records", "media_records", "fact_records", "answer_records"):
        for record in bundle.get(record_type) or []:
            if isinstance(record, dict) and _is_useful_record(record):
                records.append({**record, "_bundle_record_type": record_type})

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = _source_key(record)
        if key:
            grouped[key].append(record)

    def score_group(items: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
        text = " ".join(_text(item.get("text") or item.get("dense_text")) for item in items).lower()
        official = 1 if "mbzuai.ac.ae" in " ".join(_text(item.get("source_url")) for item in items).lower() else 0
        policy_terms = sum(
            1
            for term in (
                "admission",
                "program",
                "campus",
                "facility",
                "accommodation",
                "parking",
                "student",
                "research",
                "faculty",
                "calendar",
                "law",
                "support",
            )
            if term in text
        )
        return official, policy_terms, len(items)

    packs: List[Dict[str, Any]] = []
    for source_key, items in sorted(grouped.items(), key=lambda item: score_group(item[1]), reverse=True):
        selected = sorted(
            items,
            key=lambda item: (
                1 if item.get("_bundle_record_type") in {"chunk_records", "parent_records"} else 0,
                len(_text(item.get("text") or item.get("dense_text"))),
            ),
            reverse=True,
        )[: max(1, max_records_per_source)]
        if not selected:
            continue
        first = selected[0]
        pack_records = []
        for item in selected:
            pack_records.append(
                {
                    "id": item.get("id"),
                    "record_type": str(item.get("_bundle_record_type") or item.get("record_type") or ""),
                    "document_title": item.get("document_title"),
                    "document_type": item.get("document_type"),
                    "source_url": item.get("source_url"),
                    "page_numbers": item.get("page_numbers") or ([item.get("page_number")] if item.get("page_number") else []),
                    "section_path": item.get("section_path") or item.get("section_keys") or [],
                    "linked_chunk_ids": item.get("linked_chunk_ids") or item.get("child_chunk_ids") or ([item.get("id")] if str(item.get("_bundle_record_type")) == "chunk_records" else []),
                    "linked_parent_ids": item.get("linked_parent_ids") or item.get("child_parent_ids") or ([item.get("id")] if str(item.get("_bundle_record_type")) == "parent_records" else []),
                    "media_ids": item.get("media_ids") or item.get("linked_media_ids") or [],
                    "text": _record_text(item, max_chars=max_record_chars),
                }
            )
        packs.append(
            {
                "source_key": source_key,
                "document_id": first.get("document_id"),
                "document_title": first.get("document_title"),
                "document_type": first.get("document_type"),
                "source_url": first.get("source_url"),
                "records": pack_records,
            }
        )
        if len(packs) >= max_sources:
            break
    return packs


def _batch(items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
    size = max(1, int(batch_size or 1))
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def _system_prompt() -> str:
    return (
        "You are creating a production evaluation set for an MBZUAI retrieval-grounded chatbot. "
        "Generate rigorous QA examples only when the answer can be grounded in the provided source packs "
        "or in official MBZUAI web-search results. Prefer official MBZUAI sources. "
        "The goal is not simple trivia only: include detailed, multi-page, multi-document, policy, contact, "
        "citation-quality, follow-up-quality, and no-answer cases. Return strict JSON only."
    )


def _user_prompt(
    *,
    source_packs: Sequence[Mapping[str, Any]],
    batch_count: int,
    topics: Sequence[str],
    batch_index: int,
    total_batches: int,
) -> str:
    compact_sources = []
    for pack in source_packs:
        compact_sources.append(
            {
                "source_key": pack.get("source_key"),
                "document_title": pack.get("document_title"),
                "document_type": pack.get("document_type"),
                "source_url": pack.get("source_url"),
                "records": pack.get("records"),
            }
        )
    return (
        f"Generate {batch_count} evaluation rows for batch {batch_index} of {total_batches}.\n"
        "Use these target topic families across the whole dataset:\n"
        f"{json.dumps(list(topics), ensure_ascii=True, indent=2)}\n\n"
        "Output JSON schema:\n"
        "{\n"
        "  \"items\": [\n"
        "    {\n"
        "      \"query\": \"user-style question\",\n"
        "      \"query_type\": \"fact|scoped|synthesis|multimodal\",\n"
        "      \"source_type\": \"webpage|pdf|mixed|image|video|none\",\n"
        "      \"no_answer\": false,\n"
        "      \"reference_answer\": \"complete source-grounded answer\",\n"
        "      \"expected_response_structure\": \"paragraph|bullets|table|short_answer|step_by_step\",\n"
        "      \"answer_must_include\": [\"specific required fact\"],\n"
        "      \"answer_must_not_include\": [\"likely wrong or unsupported claim\"],\n"
        "      \"answer_should_cover\": [\"coverage requirement\"],\n"
        "      \"expected_source_hints\": [\"source topic/title/url/page hint\"],\n"
        "      \"expected_reference_urls\": [\"official URL if known\"],\n"
        "      \"expected_followup_topics\": [\"useful follow-up topic\"],\n"
        "      \"expected_suggested_actions\": [\"useful action, if applicable\"],\n"
        "      \"citation_requirements\": [\"what citations/references must substantively support\"],\n"
        "      \"gold_chunk_ids\": [\"ids from provided records\"],\n"
        "      \"gold_parent_ids\": [\"ids from provided records\"],\n"
        "      \"gold_media_ids\": [\"ids from provided records when image/map/media is required\"],\n"
        "      \"notes\": \"why this is a useful production eval case\",\n"
        "      \"difficulty\": \"easy|medium|hard\",\n"
        "      \"scenario\": \"short scenario label\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- At least half of this batch should require detailed responses, not one-line facts.\n"
        "- Include at least one no_answer row when appropriate; no_answer rows must have source_type \"none\" and empty gold ids.\n"
        "- For answerable rows, include gold ids only from the supplied source records.\n"
        "- A source/reference is valid only if it would substantively support the answer, not merely mention MBZUAI.\n"
        "- expected_followup_topics must be relevant next user needs, not generic prompts.\n"
        "- expected_suggested_actions should be concrete only when the answer naturally supports an action, such as applying, contacting admissions, viewing a campus map, or checking a calendar.\n"
        "- expected_response_structure should describe how the agent should format the answer for best usability.\n"
        "- Do not invent contacts, policies, campuses, phone numbers, deadlines, scholarships, or program details.\n\n"
        "Source packs:\n"
        f"{json.dumps(compact_sources, ensure_ascii=True, indent=2, default=str)}"
    )


def _gemini_generate_json(*, model: str, prompt: str, use_search: bool, max_output_tokens: int) -> Dict[str, Any]:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY or GEMINI_API_KEY is required for --provider gemini")
    genai = import_genai()
    types = import_genai_types()
    client = genai.Client(api_key=api_key)
    tools = [types.Tool(googleSearch=types.GoogleSearch())] if use_search else None
    config_kwargs = {
        "systemInstruction": _system_prompt(),
        "temperature": 0.2,
        "maxOutputTokens": max_output_tokens,
        "tools": tools,
    }
    if not use_search:
        config_kwargs["responseMimeType"] = "application/json"
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return _json_from_text(str(getattr(response, "text", "") or ""))


def _openai_generate_json(*, model: str, prompt: str, use_search: bool, max_output_tokens: int) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --provider openai")
    import openai

    client = openai.OpenAI(api_key=api_key, timeout=float(os.environ.get("OPENAI_TIMEOUT_SEC", "180")))
    if use_search:
        response = client.responses.create(
            model=model,
            instructions=_system_prompt(),
            input=prompt,
            tools=[{"type": "web_search_preview"}],
            max_output_tokens=max_output_tokens,
        )
        return _json_from_text(str(getattr(response, "output_text", "") or ""))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_completion_tokens=max_output_tokens,
    )
    return _json_from_text(response.choices[0].message.content or "{}")


def _generate_json(
    *,
    provider: str,
    model: str,
    prompt: str,
    use_search: bool,
    max_output_tokens: int,
) -> Dict[str, Any]:
    provider = str(provider or "").strip().lower()
    if provider == "gemini":
        return _gemini_generate_json(
            model=model,
            prompt=prompt,
            use_search=use_search,
            max_output_tokens=max_output_tokens,
        )
    if provider == "openai":
        return _openai_generate_json(
            model=model,
            prompt=prompt,
            use_search=use_search,
            max_output_tokens=max_output_tokens,
        )
    raise ValueError("provider must be 'gemini' or 'openai'")


def _known_ids(source_packs: Sequence[Mapping[str, Any]]) -> Dict[str, set[str]]:
    chunk_ids: set[str] = set()
    parent_ids: set[str] = set()
    media_ids: set[str] = set()
    for pack in source_packs:
        for record in pack.get("records") or []:
            if not isinstance(record, Mapping):
                continue
            record_id = _text(record.get("id"))
            record_type = _text(record.get("record_type"))
            if record_id and record_type == "chunk_records":
                chunk_ids.add(record_id)
            if record_id and record_type == "parent_records":
                parent_ids.add(record_id)
            for item in _as_list(record.get("linked_chunk_ids")):
                chunk_ids.add(item)
            for item in _as_list(record.get("linked_parent_ids")):
                parent_ids.add(item)
            for item in _as_list(record.get("media_ids")):
                media_ids.add(item)
            if record_id and record_type == "media_records":
                media_ids.add(record_id)
    return {"chunks": chunk_ids, "parents": parent_ids, "media": media_ids}


def _normalize_item(
    *,
    item: Mapping[str, Any],
    index: int,
    known_ids: Mapping[str, set[str]],
    suite_name: str,
) -> EvalExample | None:
    query = _text(item.get("query"))
    if not query:
        return None
    no_answer = bool(item.get("no_answer"))
    query_type = _text(item.get("query_type")).lower() or ("fact" if no_answer else "synthesis")
    if query_type == "no_answer":
        query_type = "fact"
        no_answer = True
    if query_type not in {"fact", "scoped", "synthesis", "multimodal"}:
        query_type = "synthesis"
    source_type = _text(item.get("source_type")).lower() or ("none" if no_answer else "mixed")
    if source_type not in {"webpage", "pdf", "mixed", "image", "video", "none"}:
        source_type = "none" if no_answer else "mixed"

    gold_chunk_ids = [value for value in _as_list(item.get("gold_chunk_ids")) if value in known_ids["chunks"]]
    gold_parent_ids = [value for value in _as_list(item.get("gold_parent_ids")) if value in known_ids["parents"]]
    gold_media_ids = [value for value in _as_list(item.get("gold_media_ids")) if value in known_ids["media"]]
    if no_answer:
        gold_chunk_ids, gold_parent_ids, gold_media_ids = [], [], []
        source_type = "none"
    elif not (gold_chunk_ids or gold_parent_ids or gold_media_ids):
        return None

    scenario = _text(item.get("scenario")) or _slug(query)
    prefix = f"llm-{index:03d}-{_slug(scenario, max_len=20)}"
    example_id = _text(item.get("id")) or _stable_id(prefix, query)
    metadata = {
        "benchmark_tags": list(
            dict.fromkeys(
                [
                    suite_name,
                    "llm_generated",
                    _text(item.get("difficulty")) or "medium",
                    "abstention" if no_answer else "",
                    "deep_answer" if query_type == "synthesis" else "",
                    "citation_quality",
                    "followup_quality",
                ]
            )
        ),
        "release_suite": suite_name,
        "generator": "scripts/generate_llm_qa_eval_set.py",
        "scenario": scenario,
        "difficulty": _text(item.get("difficulty")) or "medium",
        "expected_response_structure": _text(item.get("expected_response_structure")) or "concise structured answer",
        "answer_must_include": _as_list(item.get("answer_must_include")),
        "answer_must_not_include": _as_list(item.get("answer_must_not_include")),
        "answer_should_cover": _as_list(item.get("answer_should_cover")),
        "expected_source_hints": _as_list(item.get("expected_source_hints")),
        "expected_reference_urls": _as_list(item.get("expected_reference_urls")),
        "expected_followup_topics": _as_list(item.get("expected_followup_topics")),
        "expected_suggested_actions": _as_list(item.get("expected_suggested_actions")),
        "citation_requirements": _as_list(item.get("citation_requirements")),
    }
    metadata["benchmark_tags"] = [tag for tag in metadata["benchmark_tags"] if tag]
    return EvalExample(
        id=example_id,
        query=query,
        query_type=query_type,
        source_type=source_type,
        no_answer=no_answer,
        reference_answer=_text(item.get("reference_answer")) or ("Insufficient evidence." if no_answer else ""),
        gold_chunk_ids=list(dict.fromkeys(gold_chunk_ids)),
        gold_parent_ids=list(dict.fromkeys(gold_parent_ids)),
        gold_media_ids=list(dict.fromkeys(gold_media_ids)),
        notes=_text(item.get("notes")) or "LLM-generated evaluation row.",
        metadata=metadata,
    )


def _dedupe_examples(examples: Sequence[EvalExample], *, target_count: int) -> List[EvalExample]:
    seen_queries: set[str] = set()
    output: List[EvalExample] = []
    for example in examples:
        key = re.sub(r"\W+", " ", example.query.lower()).strip()
        if key in seen_queries:
            continue
        seen_queries.add(key)
        output.append(example)
        if len(output) >= target_count:
            break
    return output


def generate_eval_set(args: argparse.Namespace) -> List[EvalExample]:
    source_packs = _load_source_packs(
        work_dir=args.work_dir,
        max_sources=args.max_sources,
        max_records_per_source=args.max_records_per_source,
        max_record_chars=args.max_record_chars,
    )
    if not source_packs:
        raise RuntimeError("No usable source packs found in retrieval bundle")
    known = _known_ids(source_packs)
    raw_items: List[Dict[str, Any]] = []
    total_batches = max(1, (args.count + args.batch_size - 1) // args.batch_size)
    for batch_index, packs in enumerate(_batch(source_packs, args.sources_per_batch), start=1):
        remaining = args.count - len(raw_items)
        if remaining <= 0:
            break
        batch_count = min(args.batch_size, remaining)
        prompt = _user_prompt(
            source_packs=packs,
            batch_count=batch_count,
            topics=args.topic or DEFAULT_TOPICS,
            batch_index=batch_index,
            total_batches=total_batches,
        )
        payload = _generate_json(
            provider=args.provider,
            model=args.model,
            prompt=prompt,
            use_search=not args.disable_search,
            max_output_tokens=args.max_output_tokens,
        )
        items = payload.get("items")
        if not isinstance(items, list):
            raise ValueError(f"LLM payload for batch {batch_index} did not include an items list")
        raw_items.extend([item for item in items if isinstance(item, dict)])
        if args.request_delay_seconds > 0:
            time.sleep(float(args.request_delay_seconds))

    examples = []
    for index, item in enumerate(raw_items, start=1):
        example = _normalize_item(
            item=item,
            index=index,
            known_ids=known,
            suite_name=args.suite_name,
        )
        if example is not None:
            examples.append(example)
    examples = _dedupe_examples(examples, target_count=args.count)
    if len(examples) < args.min_count:
        raise RuntimeError(f"Only generated {len(examples)} valid examples; minimum requested is {args.min_count}")
    return examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an MBZUAI QA evaluation set with Gemini/OpenAI and provider web search.",
    )
    parser.add_argument("--work-dir", required=True, help="Indexed run directory containing a retrieval bundle")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output eval JSONL path")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Output generation manifest path")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini", help="LLM provider")
    parser.add_argument("--model", default=None, help="Model name. Defaults by provider.")
    parser.add_argument("--count", type=int, default=50, help="Target number of eval rows")
    parser.add_argument("--min-count", type=int, default=45, help="Minimum valid rows before failing")
    parser.add_argument("--batch-size", type=int, default=8, help="Rows requested per LLM batch")
    parser.add_argument("--max-sources", type=int, default=36, help="Maximum source documents sampled from the bundle")
    parser.add_argument("--sources-per-batch", type=int, default=6, help="Source packs sent to each LLM batch")
    parser.add_argument("--max-records-per-source", type=int, default=5, help="Records sampled per source document")
    parser.add_argument("--max-record-chars", type=int, default=1800, help="Maximum characters per sampled source record")
    parser.add_argument("--max-output-tokens", type=int, default=12000, help="LLM max output tokens per batch")
    parser.add_argument("--request-delay-seconds", type=float, default=0.0, help="Delay between LLM requests")
    parser.add_argument("--disable-search", action="store_true", help="Disable provider web-search grounding tool")
    parser.add_argument("--suite-name", default="llm_generated_v1", help="Metadata suite name")
    parser.add_argument("--topic", action="append", default=None, help="Topic family to emphasize; may be repeated")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model is None:
        args.model = "gemini-2.5-flash" if args.provider == "gemini" else "gpt-5-mini"
    examples = generate_eval_set(args)
    write_eval_examples(args.output, examples)
    _write_json(
        args.manifest,
        {
            "schema_version": 1,
            "provider": args.provider,
            "model": args.model,
            "search_enabled": not args.disable_search,
            "work_dir": str(Path(args.work_dir).expanduser().resolve()),
            "output": str(Path(args.output).expanduser().resolve()),
            "row_count": len(examples),
            "suite_name": args.suite_name,
            "topics": args.topic or DEFAULT_TOPICS,
        },
    )
    print(f"Wrote {len(examples)} generated QA eval rows to {args.output}")
    print(f"Wrote generation manifest to {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
