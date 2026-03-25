from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pipeline.core.google_genai import import_genai
from pipeline.core.io import load_json_safe
from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.retrieval import AdaptiveHybridRetriever
from pipeline.retrieval.adaptive_hybrid import _ROLE_QUERY_LABELS, _clean_text, _query_intent, _tokenize

_EXACT_LOOKUP_ANSWER_TYPES = {"email", "phone", "website", "hours", "date", "service_availability"}


def _make_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    genai = import_genai()
    return genai.Client(api_key=api_key)


def _selected_answer_records(retriever: Any, retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    answer_map = getattr(retriever, "answer_map", {}) or {}
    selected_ids = [str(value) for value in (retrieval_result.get("selected_answer_ids") or []) if str(value)]
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    for answer_id in selected_ids:
        if answer_id in seen_ids:
            continue
        seen_ids.add(answer_id)
        answer = answer_map.get(answer_id)
        if isinstance(answer, dict) and answer:
            records.append(answer)
    if records:
        return records
    seen_doc_ids = set()
    for doc in retrieval_result.get("answer_documents") or []:
        if not isinstance(doc, dict):
            continue
        answer_id = str(doc.get("id") or "")
        if not answer_id or answer_id in seen_doc_ids:
            continue
        seen_doc_ids.add(answer_id)
        if str(doc.get("answer_type") or ""):
            records.append(doc)
    return records


def _rank_answer_records(query: str, retriever: Any, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not records:
        return []
    score_fn = getattr(retriever, "_score_answer_record", None)
    if not callable(score_fn):
        return list(records)
    scored: List[tuple[float, int, Dict[str, Any]]] = []
    for idx, answer in enumerate(records):
        try:
            score = float(score_fn(query, answer))
        except Exception:
            score = 0.0
        scored.append((score, idx, answer))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [answer for _score, _idx, answer in scored]


def _preferred_subject_label(query: str, answer: Dict[str, Any]) -> str:
    subject_text = _clean_text(answer.get("subject_text") or "")
    if subject_text and subject_text.lower() not in {"the institution", "institution", "the university", "university"}:
        return subject_text
    lower_query = _clean_text(query).lower()
    if "mbzuai" in lower_query or "mohamed bin zayed university of artificial intelligence" in lower_query:
        return "MBZUAI"
    return "the institution"


def _answer_supports_query_subject(query: str, answer: Dict[str, Any]) -> bool:
    intent = _query_intent(query)
    evidence_text = _clean_text(
        " ".join(
            str(part or "")
            for part in (
                answer.get("subject_text"),
                answer.get("value"),
                answer.get("text"),
                answer.get("document_title"),
                answer.get("source_url"),
            )
        )
    ).lower()
    if not evidence_text:
        return False
    subject_phrases = [phrase for phrase in intent.subject_phrases if phrase and phrase != "mbzuai"]
    phrase_hit = not subject_phrases or any(phrase in evidence_text for phrase in subject_phrases)
    subject_tokens = {token for token in intent.subject_tokens if token and token != "mbzuai"}
    token_hit = True
    if subject_tokens:
        evidence_tokens = set(_tokenize(evidence_text))
        token_hit = bool(subject_tokens & evidence_tokens)
    if subject_phrases and subject_tokens:
        return phrase_hit or token_hit
    if subject_phrases:
        return phrase_hit
    if subject_tokens:
        return token_hit
    return True


def _compose_role_holder_sentence(query: str, answer: Dict[str, Any]) -> Optional[str]:
    holder = _clean_text(answer.get("value") or answer.get("text") or "")
    if not holder:
        return None
    subtype = str(answer.get("answer_subtype") or "")
    subject_label = _preferred_subject_label(query, answer)
    if subtype == "board_chair":
        if subject_label == "MBZUAI":
            return f"The chair of MBZUAI's Board of Trustees is {holder}."
        return f"The chair of {subject_label}'s Board of Trustees is {holder}."
    role_label = _ROLE_QUERY_LABELS.get(subtype, subtype.replace("_", " ").strip())
    return f"The {role_label} of {subject_label} is {holder}."


def _compose_atomic_structured_answer(query: str, answer: Dict[str, Any]) -> Optional[str]:
    answer_type = str(answer.get("answer_type") or "")
    value = _clean_text(answer.get("value") or "")
    text = _clean_text(answer.get("text") or value)
    if answer_type == "role_holder":
        return _compose_role_holder_sentence(query, answer)
    if answer_type == "email":
        return f"The email address is {value}." if value else None
    if answer_type == "phone":
        return f"The phone number is {value}." if value else None
    if answer_type == "website":
        return f"The website is {value}." if value else None
    if answer_type == "location":
        subject_label = _preferred_subject_label(query, answer)
        return f"{subject_label} is located in {value}." if value else (text or None)
    if answer_type == "named_after":
        subject_label = _preferred_subject_label(query, answer)
        return f"{subject_label} is named after {value}." if value else (text or None)
    if answer_type == "affiliation":
        subject_label = _preferred_subject_label(query, answer)
        return f"{subject_label} is affiliated with {value}." if value else (text or None)
    if answer_type in {"legal_basis", "hours", "date", "service_availability"}:
        return text or value or None
    return text or value or None


def _compose_structured_answer(
    *,
    query: str,
    retriever: Any,
    retrieval_result: Dict[str, Any],
) -> Optional[str]:
    query_intent = _query_intent(query)
    if not query_intent.answer_types:
        return None
    answer_records = _rank_answer_records(query, retriever, _selected_answer_records(retriever, retrieval_result))
    if not answer_records:
        return "Insufficient evidence."
    requested_types = list(dict.fromkeys(query_intent.answer_types))
    exact_lookup_only = bool(requested_types) and set(requested_types) <= _EXACT_LOOKUP_ANSWER_TYPES
    if exact_lookup_only:
        supported_answers = list(answer_records)
    else:
        supported_answers = [answer for answer in answer_records if _answer_supports_query_subject(query, answer)]
    if not supported_answers:
        return "Insufficient evidence."

    requested_roles = list(query_intent.requested_role_subtypes)
    if requested_roles:
        sentences: List[str] = []
        for role in requested_roles:
            matching = [
                answer
                for answer in supported_answers
                if str(answer.get("answer_type") or "") == "role_holder"
                and str(answer.get("answer_subtype") or "") == role
            ]
            if not matching:
                return "Insufficient evidence."
            best_answer = matching[0]
            sentence = _compose_role_holder_sentence(query, best_answer)
            if not sentence:
                return "Insufficient evidence."
            sentences.append(sentence)
        return " ".join(dict.fromkeys(sentences))

    if not requested_types:
        return None
    if len(requested_types) > 1:
        sentences: List[str] = []
        for answer_type in requested_types:
            matching = [
                answer
                for answer in supported_answers
                if str(answer.get("answer_type") or "") == answer_type
            ]
            if not matching:
                return "Insufficient evidence."
            best_answer = matching[0]
            sentence = _compose_atomic_structured_answer(query, best_answer)
            if not sentence:
                return "Insufficient evidence."
            sentences.append(sentence)
        return " ".join(dict.fromkeys(sentences))

    best_answers = [answer for answer in supported_answers if str(answer.get("answer_type") or "") in requested_types]
    if not best_answers:
        return "Insufficient evidence."
    return _compose_atomic_structured_answer(query, best_answers[0]) or "Insufficient evidence."


def _bundle_maps(work_dir: str | Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    bundle_path = Path(work_dir) / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
    bundle = load_json_safe(bundle_path, {}) or {}
    if not isinstance(bundle, dict):
        raise ValueError(f"Invalid retrieval bundle at {bundle_path}")
    return {
        "chunks": {item["id"]: item for item in bundle.get("chunk_records", []) if isinstance(item, dict) and item.get("id")},
        "parents": {item["id"]: item for item in bundle.get("parent_records", []) if isinstance(item, dict) and item.get("id")},
        "media": {item["id"]: item for item in bundle.get("media_records", []) if isinstance(item, dict) and item.get("id")},
    }


def _reference_contexts(example: EvalExample, maps: Dict[str, Dict[str, Dict[str, Any]]]) -> List[str]:
    contexts: List[str] = []
    for chunk_id in example.gold_chunk_ids:
        chunk = maps["chunks"].get(chunk_id)
        if chunk and chunk.get("text"):
            contexts.append(str(chunk["text"]))
    for parent_id in example.gold_parent_ids:
        parent = maps["parents"].get(parent_id)
        if parent and parent.get("text"):
            contexts.append(str(parent["text"]))
    for media_id in example.gold_media_ids:
        media = maps["media"].get(media_id)
        if media and media.get("text"):
            contexts.append(str(media["text"]))
    seen = set()
    deduped = []
    for item in contexts:
        normalized = " ".join(item.split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(item)
    return deduped


def _build_answer_prompt(
    *,
    query: str,
    retrieval_documents: List[Dict[str, Any]],
    answer_documents: List[Dict[str, Any]] | None = None,
    fact_documents: List[Dict[str, Any]] | None = None,
) -> str:
    structured_blocks = []
    for i, doc in enumerate((answer_documents or [])[:6], start=1):
        text = str(doc.get("text") or "").strip()
        if not text:
            continue
        structured_blocks.append(f"[Answer {i}] {text}")
    for i, doc in enumerate((fact_documents or [])[:6], start=1):
        text = str(doc.get("text") or "").strip()
        if not text:
            continue
        structured_blocks.append(f"[Fact {i}] {text}")
    context_blocks = []
    for i, doc in enumerate(retrieval_documents[:8], start=1):
        source = doc.get("source_url") or doc.get("document_title") or doc.get("id") or f"doc-{i}"
        text = str(doc.get("text") or "").strip()
        if not text:
            continue
        context_blocks.append(f"[Source {i}: {source}]\n{text}")
    joined_context = "\n\n".join(context_blocks)
    joined_structured = "\n".join(structured_blocks)
    return (
        "Answer the user question using only the supplied context.\n"
        "Use structured evidence first when it is present.\n"
        "If the context does not support a grounded answer, say exactly: Insufficient evidence.\n"
        "If the question asks for multiple items, answer all of them only when the context supports each requested item.\n"
        "Keep the answer concise and factual.\n\n"
        f"Question:\n{query}\n\n"
        f"Structured evidence:\n{joined_structured or 'None'}\n\n"
        f"Context:\n{joined_context}"
    )


def generate_answer_predictions(
    *,
    config_name: str,
    work_dir: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    retriever = AdaptiveHybridRetriever.from_config(config_name=config_name, work_dir=work_dir)
    examples = load_eval_examples(dataset_path)
    maps = _bundle_maps(work_dir)
    client = None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for example in examples:
        retrieval_result = retriever.retrieve(example.query)
        if retrieval_result.get("abstained"):
            answer = "Insufficient evidence."
        else:
            structured_answer = _compose_structured_answer(
                query=example.query,
                retriever=retriever,
                retrieval_result=retrieval_result,
            )
            if structured_answer is not None:
                answer = structured_answer
            else:
                if client is None:
                    client = _make_gemini_client()
                prompt = _build_answer_prompt(
                    query=example.query,
                    retrieval_documents=retrieval_result.get("retrieval_documents") or [],
                    answer_documents=retrieval_result.get("answer_documents") or [],
                    fact_documents=retrieval_result.get("fact_documents") or [],
                )
                response = client.models.generate_content(model=model, contents=prompt)
                answer = str(getattr(response, "text", "") or "").strip()
        rows.append(
            {
                "id": example.id,
                "query_type": example.query_type,
                "source_type": example.source_type,
                "user_input": example.query,
                "response": answer,
                "reference": example.reference_answer,
                "retrieved_contexts": [
                    str(item.get("text") or "")
                    for item in (retrieval_result.get("retrieval_documents") or [])
                    if isinstance(item, dict) and str(item.get("text") or "").strip()
                ],
                "reference_contexts": _reference_contexts(example, maps),
                "retrieved_context_ids": list(retrieval_result.get("selected_chunk_ids") or []),
                "reference_context_ids": list(example.gold_chunk_ids),
                "metadata": {
                    "mode": retrieval_result.get("mode"),
                    "seed_chunk_ids": list(retrieval_result.get("seed_chunk_ids") or []),
                    "selected_answer_ids": list(retrieval_result.get("selected_answer_ids") or []),
                    "dense_parent_ids": list(retrieval_result.get("dense_parent_ids") or []),
                    "media_ids": [str(item.get("id") or "") for item in (retrieval_result.get("media") or []) if isinstance(item, dict)],
                },
            }
        )

    output_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )
    return {
        "output_path": str(output_path.resolve()),
        "row_count": len(rows),
        "model": model,
    }
