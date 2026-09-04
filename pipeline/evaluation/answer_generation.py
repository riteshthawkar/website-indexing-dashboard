from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pipeline.core.google_genai import import_genai
from pipeline.core.io import load_json_safe
from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.retrieval import AdaptiveHybridRetriever
from pipeline.retrieval.adaptive_hybrid import _ROLE_QUERY_LABELS, _clean_text, _query_intent, _tokenize

_EXACT_LOOKUP_ANSWER_TYPES = {"email", "phone", "website", "hours", "date", "service_availability"}


def _query_requires_contextual_answer(query: str) -> bool:
    """Return whether an atomic record cannot satisfy the requested fields.

    A promoted answer record can be a useful low-latency fact, but it is not a
    substitute for the surrounding page when the user explicitly requests a
    displayed title, designation, label, or person-to-category mapping.  In
    those cases generation must inspect the packed evidence rather than stop
    after the first matching name.
    """

    normalized = " ".join(str(query or "").casefold().split())
    return bool(
        re.search(
            r"\b(?:full|official|displayed|shown)\s+(?:title|designation|position|label)\b"
            r"|\bwhat\s+(?:title|designation|position|label)\b"
            r"|\bwhich\s+(?:division|department|school|unit|category)\b",
            normalized,
        )
        or any(
            marker in normalized
            for marker in (
                "المسمى الرسمي",
                "المسمى الوظيفي",
                "ما المسمى",
                "ما اللقب",
                "أي قسم",
                "اي قسم",
                "أي إدارة",
                "اي ادارة",
            )
        )
    )


def _compose_abstention_answer(query: str, retrieval_result: Dict[str, Any]) -> str:
    """Produce a useful fail-closed answer without inventing missing facts."""

    normalized = " ".join(str(query or "").casefold().split())
    reason = str(retrieval_result.get("adjudication_reason") or "").casefold()
    is_arabic = bool(re.search(r"[\u0600-\u06ff]", str(query or "")))
    # Trust the retriever's temporal guard rather than guessing from a year in
    # the query; an exact historical-year question is not a future request.
    future_fact = reason == "unsupported_future_mutable_fact"
    fee_request = bool(
        re.search(r"\b(?:tuition|fees?|cost|price)\b", normalized)
        or any(marker in normalized for marker in ("الرسوم", "التكلفة", "السعر"))
    )
    program_request = bool(
        re.search(r"\b(?:program|degree|course|admissions?|requirements?)\b", normalized)
        or any(
            marker in normalized
            for marker in ("برنامج", "درجة", "تخصص", "القبول", "المتطلبات")
        )
    )

    if is_arabic:
        if future_fact and fee_request:
            return (
                "لا توجد أدلة كافية. لا تحدد مصادر MBZUAI المتاحة الرسوم الدراسية أو "
                "التكلفة المستقبلية الدقيقة المطلوبة، لذلك لا يمكنني تقديم قيمة موثقة."
            )
        if reason in {
            "presupposed_claim_explicitly_refuted",
            "presupposed_entity_or_scope_not_supported",
        } and program_request:
            return (
                "لا توجد أدلة كافية. لم أتمكن من التحقق من وجود البرنامج أو الدرجة المطلوبة "
                "في مصادر MBZUAI المتاحة، لذلك لا يمكنني تقديم متطلبات خاصة بها."
            )
        return (
            "لا توجد أدلة كافية. لم أتمكن من التحقق من المعلومة المطلوبة مباشرةً في "
            "مصادر MBZUAI المتاحة، لذلك لا يمكنني تقديم إجابة موثقة."
        )

    if future_fact and fee_request:
        return (
            "Insufficient evidence. The available MBZUAI sources do not establish the exact "
            "future tuition or fee requested, so I cannot provide a source-grounded amount."
        )
    if reason in {
        "presupposed_claim_explicitly_refuted",
        "presupposed_entity_or_scope_not_supported",
    } and program_request:
        return (
            "Insufficient evidence. I could not verify that the requested program or degree "
            "exists in the available MBZUAI sources, so I cannot provide program-specific requirements."
        )
    return (
        "Insufficient evidence. I could not verify the requested fact directly in the available "
        "MBZUAI sources, so I cannot provide a source-grounded answer."
    )


def _make_gemini_client():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
    genai = import_genai()
    return genai.Client(api_key=api_key)


def _read_prediction_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_prediction_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=True, default=str) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _row_error(row: Dict[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(row.get("error") or metadata.get("error") or "").strip()


def _call_generate_content_with_timeout(client: Any, *, model: str, prompt: str, timeout_seconds: float) -> str:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: client.models.generate_content(model=model, contents=prompt))
    try:
        response = future.result(timeout=max(1.0, float(timeout_seconds or 1.0)))
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"Answer generation timed out after {timeout_seconds} seconds") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return str(getattr(response, "text", "") or "").strip()


def _generate_content_with_retry(
    *,
    prompt: str,
    model: str,
    timeout_seconds: float,
    max_retries: int,
) -> str:
    attempts = max(1, int(max_retries or 0) + 1)
    last_error = ""
    for attempt in range(attempts):
        try:
            client = _make_gemini_client()
            return _call_generate_content_with_timeout(
                client,
                model=model,
                prompt=prompt,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            last_error = str(exc)
            if attempt + 1 >= attempts:
                break
            time.sleep(min(8.0, 1.5 * (2**attempt)))
    raise RuntimeError(last_error or "answer generation failed")


def _selected_answer_records(retriever: Any, retrieval_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    answer_map = getattr(retriever, "answer_map", {}) or {}
    selected_ids = [str(value) for value in (retrieval_result.get("selected_answer_ids") or []) if str(value)]
    records: List[Dict[str, Any]] = []
    seen_ids = set()
    # Validated page actions are synthesized at retrieval time and therefore
    # do not exist in the immutable answer map. Merge direct documents first
    # instead of treating them only as a fallback.
    for doc in retrieval_result.get("answer_documents") or []:
        if not isinstance(doc, dict):
            continue
        answer_id = str(doc.get("id") or "")
        if not answer_id or answer_id in seen_ids:
            continue
        if str(doc.get("answer_type") or ""):
            records.append(doc)
            seen_ids.add(answer_id)
    for answer_id in selected_ids:
        if answer_id in seen_ids:
            continue
        answer = answer_map.get(answer_id)
        if isinstance(answer, dict) and answer:
            records.append(answer)
            seen_ids.add(answer_id)
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
    if _query_requires_contextual_answer(query):
        return None
    query_intent = _query_intent(query)
    if not query_intent.answer_types:
        return None
    answer_records = _rank_answer_records(query, retriever, _selected_answer_records(retriever, retrieval_result))
    if not answer_records:
        return None
    requested_types = list(dict.fromkeys(query_intent.answer_types))
    exact_lookup_only = bool(requested_types) and set(requested_types) <= _EXACT_LOOKUP_ANSWER_TYPES
    if exact_lookup_only:
        supported_answers = list(answer_records)
    else:
        supported_answers = [answer for answer in answer_records if _answer_supports_query_subject(query, answer)]
    if not supported_answers:
        return None

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
                return None
            best_answer = matching[0]
            sentence = _compose_role_holder_sentence(query, best_answer)
            if not sentence:
                return None
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
                return None
            best_answer = matching[0]
            sentence = _compose_atomic_structured_answer(query, best_answer)
            if not sentence:
                return None
            sentences.append(sentence)
        return " ".join(dict.fromkeys(sentences))

    best_answers = [answer for answer in supported_answers if str(answer.get("answer_type") or "") in requested_types]
    if not best_answers:
        return None
    return _compose_atomic_structured_answer(query, best_answers[0])


def _bundle_maps(work_dir: str | Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    run_dir = Path(work_dir)
    bundle_path = run_dir / "stage_outputs" / "finalize_retrieval_bundle" / "retrieval_bundle.json"
    if not bundle_path.exists():
        candidate = run_dir / "stage_outputs" / "format_retrieval" / "retrieval_bundle.json"
        if candidate.exists():
            bundle_path = candidate
    if not bundle_path.exists():
        candidate = run_dir / "stage_outputs" / "build_retrieval_bundle" / "retrieval_bundle.json"
        if candidate.exists():
            bundle_path = candidate
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
    evidence_pack: Dict[str, Any] | None = None,
) -> str:
    structured_blocks = []
    context_blocks = []
    coverage_requirements: List[str] = []
    if isinstance(evidence_pack, dict):
        for raw_facet in evidence_pack.get("facet_coverage") or []:
            if not isinstance(raw_facet, dict):
                continue
            name = str(raw_facet.get("name") or "").strip()
            if not name or not bool(raw_facet.get("report_in_answer", True)):
                continue
            coverage_requirements.append(name)
    coverage_requirements = list(dict.fromkeys(coverage_requirements))
    pack_items = (
        list(evidence_pack.get("items") or [])
        if isinstance(evidence_pack, dict)
        else []
    )
    pack_items = [item for item in pack_items if isinstance(item, dict)]
    if pack_items:
        for item in pack_items:
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            rank = int(item.get("rank") or (len(context_blocks) + 1))
            kind = str(item.get("kind") or "evidence").strip().title()
            source = item.get("source_url") or item.get("document_title") or item.get("id") or f"evidence-{rank}"
            if kind.lower() in {"answer", "fact"}:
                structured_blocks.append(f"[{kind} {rank}] {text}")
            context_blocks.append(f"[Evidence {rank}: {kind}; {source}]\n{text}")
    else:
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
        for i, doc in enumerate(retrieval_documents[:8], start=1):
            source = doc.get("source_url") or doc.get("document_title") or doc.get("id") or f"doc-{i}"
            text = str(doc.get("text") or "").strip()
            if not text:
                continue
            context_blocks.append(f"[Source {i}: {source}]\n{text}")
    joined_context = "\n\n".join(context_blocks)
    joined_structured = "\n".join(structured_blocks)
    joined_requirements = "\n".join(
        f"- {name}" for name in coverage_requirements
    )
    return (
        "Answer the user question using only the supplied context.\n"
        "Respond in the same language as the user's question.\n"
        "Use structured evidence first when it is present, but do not assume one atomic fact is a complete "
        "answer: inspect the surrounding evidence for every field the user explicitly requested.\n"
        "If the context does not support a grounded answer, begin with 'Insufficient evidence.' and briefly "
        "identify which requested fact could not be verified without guessing.\n"
        "Silently identify every requested field before drafting; answer each supported field explicitly and "
        "state any unsupported field as unavailable.\n"
        "If the question asks for multiple items, answer every supported item explicitly.\n"
        "When the user asks for an official title, designation, label, count, or mapping, preserve the complete "
        "source wording. For person-to-unit mappings, prefer the canonical organizational-unit heading over "
        "a shortened nearby role caption, and do not repeat a conflicting shorthand label as a second name.\n"
        "For a types-or-categories question, list only items the source explicitly classifies as that requested "
        "type; do not promote adjacent sponsorship mechanisms, seats, optional benefits, or examples into the "
        "classification. For a maximum-or-extent question, state the maximum and what it applies to without "
        "expanding into other optional benefits unless the user also asks what is included.\n"
        "When asked about an organization's, board's, committee's, or unit's role, include all distinct "
        "source-stated functions and any composition or mission contribution that materially explains that role.\n"
        "For each requested value, preserve closely attached conditions, timing, scope, exceptions, "
        "dependencies, reimbursements, credits, or limitations that are needed to use the answer correctly.\n"
        "A qualifier belongs in the response only when it changes how a requested value should be interpreted or "
        "used; a separate optional benefit is not a qualifier for an otherwise complete requested maximum.\n"
        "Treat the answer coverage checklist as the response scope: cover every supported checklist facet, but "
        "do not add adjacent programs, benefits, funding mechanisms, people, or categories merely because they "
        "appear in the context.\n"
        "Treat Page Card PURPOSE, TOPICS, AUDIENCES, and SECTIONS fields, plus standalone headings, as discovery "
        "metadata rather than proof of a factual claim. A label only shows that a topic exists; state a requirement, "
        "eligibility rule, benefit, or policy only when an evidence sentence explicitly supports it.\n"
        "Preserve polarity and modality exactly: required, not required, optional, recommended, preferred, may, "
        "and must are materially different. Never infer one of them from a heading or topic label.\n"
        "For a duration or timeline question, include the ordinary or typical duration and any source-stated "
        "minimum, maximum, or completion deadline that materially limits it.\n"
        "Before returning the answer, silently check every factual clause against an explicit evidence sentence "
        "and remove or qualify any clause that is not directly entailed.\n"
        "Use compact bullets when they make a multi-part answer clearer.\n"
        "Be concise but complete; do not omit a supported qualification merely to shorten the answer.\n\n"
        f"Question:\n{query}\n\n"
        f"Answer coverage checklist:\n{joined_requirements or 'No additional structured checklist.'}\n\n"
        f"Structured evidence:\n{joined_structured or 'None'}\n\n"
        f"Context:\n{joined_context}"
    )


def _retrieved_contexts_from_result(retrieval_result: Dict[str, Any]) -> List[str]:
    evidence_pack = retrieval_result.get("evidence_pack")
    if isinstance(evidence_pack, dict):
        items = [item for item in (evidence_pack.get("items") or []) if isinstance(item, dict)]
        contexts = [str(item.get("text") or "") for item in items if str(item.get("text") or "").strip()]
        if contexts:
            return contexts
    return [
        str(item.get("text") or "")
        for item in (retrieval_result.get("retrieval_documents") or [])
        if isinstance(item, dict) and str(item.get("text") or "").strip()
    ]


def _sources_from_result(retrieval_result: Dict[str, Any], *, limit: int = 12) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()

    def add_source(kind: str, item: Dict[str, Any]) -> None:
        if len(sources) >= limit:
            return
        source_url = str(item.get("source_url") or "").strip()
        title = str(item.get("document_title") or item.get("title") or "").strip()
        source_id = str(item.get("id") or item.get("source_id") or "").strip()
        key = (source_url, title, source_id)
        if not any(key) or key in seen:
            return
        seen.add(key)
        sources.append(
            {
                "id": source_id,
                "kind": kind,
                "title": title,
                "url": source_url,
                "text_preview": str(item.get("text") or "").strip()[:400],
            }
        )

    evidence_pack = retrieval_result.get("evidence_pack")
    if isinstance(evidence_pack, dict):
        for item in evidence_pack.get("items") or []:
            if isinstance(item, dict):
                add_source(str(item.get("kind") or "evidence"), item)
    for key, kind in (
        ("answer_documents", "answer"),
        ("fact_documents", "fact"),
        ("retrieval_documents", "chunk"),
        ("media", "media"),
    ):
        for item in retrieval_result.get(key) or []:
            if isinstance(item, dict):
                add_source(kind, item)
    return sources


def generate_answer_predictions(
    *,
    config_name: str,
    work_dir: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    model: str = "gemini-2.5-flash",
    timeout_seconds: float = 120.0,
    max_retries: int = 2,
    resume_predictions: bool = False,
    examples: Sequence[EvalExample] | None = None,
) -> Dict[str, Any]:
    retriever = AdaptiveHybridRetriever.from_config(config_name=config_name, work_dir=work_dir)
    examples = list(examples) if examples is not None else load_eval_examples(dataset_path)
    maps = _bundle_maps(work_dir)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    if resume_predictions:
        for row in _read_prediction_rows(output_path):
            row_id = str(row.get("id") or "").strip()
            if row_id and not _row_error(row):
                rows_by_id[row_id] = row
    else:
        _write_prediction_rows(output_path, [])

    def ordered_rows() -> List[Dict[str, Any]]:
        return [rows_by_id[example.id] for example in examples if example.id in rows_by_id]

    for example in examples:
        if example.id in rows_by_id:
            continue
        started = time.perf_counter()
        retrieval_result: Dict[str, Any] = {}
        error = ""
        try:
            retrieval_result = retriever.retrieve(example.query)
            if retrieval_result.get("abstained"):
                answer = _compose_abstention_answer(example.query, retrieval_result)
            else:
                structured_answer = _compose_structured_answer(
                    query=example.query,
                    retriever=retriever,
                    retrieval_result=retrieval_result,
                )
                if structured_answer is not None:
                    answer = structured_answer
                else:
                    prompt = _build_answer_prompt(
                        query=example.query,
                        retrieval_documents=retrieval_result.get("retrieval_documents") or [],
                        answer_documents=retrieval_result.get("answer_documents") or [],
                        fact_documents=retrieval_result.get("fact_documents") or [],
                        evidence_pack=retrieval_result.get("evidence_pack") or {},
                    )
                    answer = _generate_content_with_retry(
                        prompt=prompt,
                        model=model,
                        timeout_seconds=timeout_seconds,
                        max_retries=max_retries,
                    )
        except Exception as exc:
            answer = ""
            error = str(exc)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
        response_evidence_pack = (
            retrieval_result.get("evidence_pack")
            if isinstance(retrieval_result.get("evidence_pack"), dict)
            else {}
        )
        metadata = {
            "mode": retrieval_result.get("mode"),
            "abstained": bool(retrieval_result.get("abstained")),
            "verification_status": retrieval_result.get("verification_status"),
            "adjudication_reason": retrieval_result.get("adjudication_reason"),
            "coverage_status": retrieval_result.get("coverage_status"),
            "missing_required_facets": list(
                retrieval_result.get("missing_required_facets")
                or response_evidence_pack.get("missing_required_facets")
                or []
            ),
            "facet_coverage": list(
                response_evidence_pack.get("facet_coverage")
                or []
            ),
            "seed_chunk_ids": list(retrieval_result.get("seed_chunk_ids") or []),
            "selected_answer_ids": list(retrieval_result.get("selected_answer_ids") or []),
            "dense_parent_ids": list(retrieval_result.get("dense_parent_ids") or []),
            "media_ids": [str(item.get("id") or "") for item in (retrieval_result.get("media") or []) if isinstance(item, dict)],
            "backend": "local_indexing_answer_generation",
            "latency_ms": latency_ms,
            "error": error,
        }
        row = {
            "id": example.id,
            "query_type": example.query_type,
            "source_type": example.source_type,
            "user_input": example.query,
            "response": answer,
            "reference": example.reference_answer,
            "sources": _sources_from_result(retrieval_result),
            "retrieved_contexts": _retrieved_contexts_from_result(retrieval_result),
            "reference_contexts": _reference_contexts(example, maps),
            "retrieved_context_ids": list(retrieval_result.get("selected_chunk_ids") or []),
            "reference_context_ids": list(example.gold_chunk_ids),
            "metadata": metadata,
            "latency_ms": latency_ms,
            "error": error,
        }
        rows_by_id[example.id] = row
        _write_prediction_rows(output_path, ordered_rows())

    rows = ordered_rows()
    _write_prediction_rows(output_path, rows)
    error_count = sum(1 for row in rows if _row_error(row))
    return {
        "output_path": str(output_path.resolve()),
        "row_count": len(rows),
        "error_count": error_count,
        "model": model,
    }
