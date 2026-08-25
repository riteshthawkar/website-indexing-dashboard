from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

from pipeline.evaluation.dataset import EvalExample, load_eval_examples


def _first_existing_stage_file(work_dir: Path, stage_ids: Sequence[str], filename: str) -> Path:
    for stage_id in stage_ids:
        candidate = work_dir / "stage_outputs" / stage_id / filename
        if candidate.exists():
            return candidate
    return work_dir / "stage_outputs" / stage_ids[0] / filename


def _count_by(values: Iterable[str]) -> Dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items()))


def summarize_eval_examples(examples: Sequence[EvalExample]) -> Dict[str, Any]:
    rows = list(examples)
    return {
        "query_count": len(rows),
        "query_type_counts": _count_by(example.query_type for example in rows),
        "source_type_counts": _count_by(example.source_type for example in rows),
        "no_answer_count": sum(1 for example in rows if example.no_answer),
        "answerable_count": sum(1 for example in rows if not example.no_answer),
        "with_gold_chunks": sum(1 for example in rows if example.gold_chunk_ids),
        "with_gold_parents": sum(1 for example in rows if example.gold_parent_ids),
        "with_gold_media": sum(1 for example in rows if example.gold_media_ids),
        "with_gold_documents": sum(1 for example in rows if example.gold_document_revision_ids),
        "with_gold_page_cards": sum(1 for example in rows if example.gold_page_card_ids),
        "with_gold_sections": sum(1 for example in rows if example.gold_section_ids),
        "with_gold_actions": sum(1 for example in rows if example.gold_action_ids),
    }


def _load_retrieval_bundle_ids(work_dir: str | Path) -> Dict[str, Set[str]]:
    bundle_path = _first_existing_stage_file(
        Path(work_dir),
        ("finalize_retrieval_bundle", "build_retrieval_bundle", "format_retrieval"),
        "retrieval_bundle.json",
    )
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    return {
        "gold_chunk_ids": {str(item.get("id")) for item in payload.get("chunk_records", []) if str(item.get("id") or "")},
        "gold_span_ids": {str(item.get("id")) for item in payload.get("evidence_span_records", []) if str(item.get("id") or "")},
        "gold_parent_ids": {str(item.get("id")) for item in payload.get("parent_records", []) if str(item.get("id") or "")},
        "gold_media_ids": {str(item.get("id")) for item in payload.get("media_records", []) if str(item.get("id") or "")},
    }


def _all_gold_values(example: EvalExample, field_name: str) -> List[str]:
    values = list(getattr(example, field_name) or [])
    alternate_values = list((example.metadata or {}).get(f"alternate_{field_name}") or [])
    output: List[str] = []
    for item in [*values, *alternate_values]:
        value = str(item or "").strip()
        if value:
            output.append(value)
    return output


def _metadata_list(metadata: Dict[str, Any], key: str) -> List[str]:
    value = metadata.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _looks_like_no_answer_reference(value: str) -> bool:
    normalized = " ".join(str(value or "").split()).lower()
    return any(
        marker in normalized
        for marker in (
            "insufficient evidence",
            "not enough evidence",
            "cannot provide",
            "cannot find",
            "could not verify",
            "cannot verify",
            "does not verify",
            "does not establish",
            "contains no",
            "should abstain",
            "must abstain",
            "should not",
            "should not invent",
            "must not fabricate",
            "not available",
            "not found",
            "not in the provided sources",
            "does not contain",
            "do not contain",
            "lack of information",
            "no answer",
            "لا تحتوي",
            "لا تتضمن",
            "لا تتحقق",
            "لا تثبت",
            "لا تقدم",
            "لا تحدد",
            "غير متاح",
            "غير متوفرة",
            "لا يمكن تقديم",
            "الامتناع عن",
            "عدم اختلاق",
        )
    )


def validate_eval_examples(
    dataset_path: str | Path,
    *,
    work_dir: str | Path | None = None,
) -> Dict[str, Any]:
    examples = load_eval_examples(dataset_path)
    summary = summarize_eval_examples(examples)
    errors: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    available_ids: Dict[str, Set[str]] = {}
    if work_dir:
        available_ids = _load_retrieval_bundle_ids(work_dir)

    for example in examples:
        metadata = dict(example.metadata or {})
        source_grounded_ids = (
            example.gold_document_revision_ids,
            example.gold_page_card_ids,
            example.gold_section_ids,
            example.gold_action_ids,
        )
        retrieval_grounded_ids = (
            example.gold_chunk_ids,
            example.gold_span_ids,
            example.gold_parent_ids,
            example.gold_media_ids,
        )
        if example.no_answer:
            if any((*retrieval_grounded_ids, *source_grounded_ids)):
                warnings.append(
                    {
                        "id": example.id,
                        "field": "no_answer",
                        "reason": "no_answer example contains gold evidence ids",
                    }
                )
            if _metadata_list(metadata, "expected_reference_urls"):
                warnings.append(
                    {
                        "id": example.id,
                        "field": "expected_reference_urls",
                        "reason": "no_answer example lists expected official reference URLs; audit whether this should be answerable",
                    }
                )
            if _metadata_list(metadata, "answer_must_include") and not _looks_like_no_answer_reference(
                example.reference_answer
            ):
                warnings.append(
                    {
                        "id": example.id,
                        "field": "answer_must_include",
                        "reason": "no_answer example has required answer facts but the reference answer does not read like an abstention",
                    }
                )
            if example.reference_answer and not _looks_like_no_answer_reference(example.reference_answer):
                warnings.append(
                    {
                        "id": example.id,
                        "field": "reference_answer",
                        "reason": "no_answer reference answer does not look like an abstention",
                    }
                )
            continue

        if not any((*retrieval_grounded_ids, *source_grounded_ids)):
            errors.append(
                {
                    "id": example.id,
                    "field": "gold_evidence",
                    "reason": "answerable example is missing all gold evidence ids",
                }
            )

        if not example.reference_answer:
            warnings.append(
                {
                    "id": example.id,
                    "field": "reference_answer",
                    "reason": "answerable example is missing a reference answer",
                }
            )

        if not available_ids:
            continue

        for field_name, known_ids in available_ids.items():
            for value in _all_gold_values(example, field_name):
                if value not in known_ids:
                    errors.append(
                        {
                            "id": example.id,
                            "field": field_name,
                            "reason": "missing_index_id",
                            "value": value,
                        }
                    )

    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "work_dir": str(Path(work_dir).resolve()) if work_dir else None,
        "summary": summary,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
    }
