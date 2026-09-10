from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

from pipeline.evaluation.dataset import EvalExample, write_eval_examples
from pipeline.evaluation.dataset_tools import validate_eval_examples
from pipeline.evaluation.multilingual_v2 import (
    assign_stratified_splits,
    contains_evidence_quote,
    looks_like_language,
    normalize_evidence_text,
    summarize_multilingual_coverage,
    validate_multilingual_coverage,
    validate_source_ids,
)


PROMPT_REVISION = "mbzuai-multilingual-eval-v2-current-preprod-20260910"
SUITE_NAME = "mbzuai_multilingual_v2"
DEFAULT_REPRESENTATION = (
    PROJECT_ROOT
    / "runs/mbzuai_representation_v2/mbzuai-representation-v2-20260821-v2"
    / "stage_outputs/extract_page_cards_and_actions/representation_v2_bundle.json"
)
DEFAULT_MEDIA = (
    PROJECT_ROOT
    / "runs/mbzuai_corpus_preparation/mbzuai-corpus-preparation-20260821-v4"
    / "stage_outputs/prepare_corpus_boundary/prepared_media_manifest.json"
)
DEFAULT_NAVIGATION_CATALOG = (
    PROJECT_ROOT
    / "runs/mbzuai_page_graph_bridge/mbzuai-page-graph-bridge-20260822-v4"
    / "stage_outputs/bridge_page_graph/page_graph_navigation_catalog.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "eval/mbzuai_gold/mbzuai_multilingual_v2.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "eval/mbzuai_gold/mbzuai_multilingual_v2.manifest.json"
DEFAULT_PLAN = PROJECT_ROOT / "eval/mbzuai_gold/mbzuai_multilingual_v2.plan.json"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "runs/evaluation/mbzuai-multilingual-v2-generation-cache"

MAIN_HOST = "mbzuai.ac.ae"
MAIN_HOSTS = frozenset(
    {
        MAIN_HOST,
        "www.mbzuai.ac.ae",
        "preprod.mbzuai.ac.ae",
    }
)
PRODUCTION_MIN_SUBDOMAIN_TASKS = 59
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
WORD_RE = re.compile(
    r"[a-z0-9]+|[\u0621-\u063a\u0641-\u064a\u0660-\u0669]+",
    flags=re.IGNORECASE,
)
IMPORTANT_TERMS = (
    "admission",
    "apply",
    "program",
    "study",
    "research",
    "faculty",
    "campus",
    "student",
    "career",
    "collaborat",
    "partner",
    "governance",
    "leadership",
    "mission",
    "library",
    "contact",
    "scholarship",
    "academic",
    "القبول",
    "التقديم",
    "برنامج",
    "البحث",
    "الطلاب",
    "الحرم",
)
TOPIC_RULES = {
    "admissions_programs": (
        "admission",
        "apply",
        "application",
        "program",
        "study",
        "graduate",
        "undergraduate",
        "screening",
        "catalog",
        "prospectus",
        "القبول",
        "التقديم",
        "برنامج",
    ),
    "research_faculty": (
        "research",
        "faculty",
        "institute",
        "laboratory",
        "lab",
        "center",
        "centre",
        "researcher",
        "البحث",
        "هيئة التدريس",
    ),
    "campus_student_life": (
        "campus",
        "student",
        "accommodation",
        "library",
        "pool",
        "masdar",
        "map",
        "internship",
        "الحرم",
        "الطلاب",
    ),
    "careers_partnerships": (
        "career",
        "vacancy",
        "job",
        "collaborat",
        "partner",
        "employer",
        "supplier",
        "وظائف",
        "شراكة",
    ),
    "institution_governance": (
        "about",
        "mission",
        "leadership",
        "president",
        "governance",
        "strategy",
        "annual report",
        "factsheet",
        "القيادة",
        "الحوكمة",
    ),
    "events_public_engagement": (
        "event",
        "news",
        "commencement",
        "speaker",
        "magazine",
        "showcase",
        "فعاليات",
        "أخبار",
    ),
}
INFORMATIVE_MEDIA_KINDS = {
    "chart",
    "diagram",
    "document_fragment",
    "illustration",
    "infographic",
    "map",
    "screenshot",
}

NO_ANSWER_CASES = [
    (
        "tokyo-campus-address",
        "What is the street address of MBZUAI's Tokyo campus?",
        "ما عنوان الشارع لحرم جامعة محمد بن زايد للذكاء الاصطناعي في طوكيو؟",
        "The frozen MBZUAI corpus does not verify that MBZUAI has a Tokyo campus, so the assistant should abstain instead of inventing an address.",
        "لا تتحقق مجموعة بيانات جامعة محمد بن زايد للذكاء الاصطناعي المجمدة من وجود حرم للجامعة في طوكيو، لذا ينبغي للمساعد الامتناع عن اختلاق عنوان.",
    ),
    (
        "singapore-office-phone",
        "What is the phone number for MBZUAI's Singapore office?",
        "ما رقم هاتف مكتب جامعة محمد بن زايد للذكاء الاصطناعي في سنغافورة؟",
        "The frozen corpus contains no verified Singapore office phone number, so no number should be supplied.",
        "لا تتضمن المجموعة المجمدة رقم هاتف موثقاً لمكتب في سنغافورة، لذلك يجب عدم تقديم أي رقم.",
    ),
    (
        "veterinary-degree",
        "What are the admission requirements for MBZUAI's veterinary medicine degree?",
        "ما متطلبات القبول في برنامج الطب البيطري بجامعة محمد بن زايد للذكاء الاصطناعي؟",
        "The frozen corpus does not establish that MBZUAI offers a veterinary medicine degree, so the assistant should say that it cannot verify such a program.",
        "لا تثبت المجموعة المجمدة أن الجامعة تقدم برنامجاً في الطب البيطري، لذا ينبغي للمساعد توضيح أنه لا يستطيع التحقق من وجوده.",
    ),
    (
        "mars-shuttle",
        "What is the shuttle timetable for MBZUAI's Mars research campus?",
        "ما جدول الحافلات إلى الحرم البحثي لجامعة محمد بن زايد للذكاء الاصطناعي على كوكب المريخ؟",
        "The frozen corpus has no evidence of an MBZUAI Mars campus or shuttle timetable, so the assistant must abstain.",
        "لا تحتوي المجموعة المجمدة على دليل لوجود حرم للجامعة على المريخ أو جدول حافلات إليه، لذا يجب على المساعد الامتناع عن الإجابة.",
    ),
    (
        "crypto-wallet",
        "Which cryptocurrency wallet should applicants use to pay MBZUAI application fees?",
        "ما محفظة العملات المشفرة التي ينبغي للمتقدمين استخدامها لدفع رسوم التقديم إلى الجامعة؟",
        "The frozen corpus does not provide a verified cryptocurrency wallet for application fees, so the assistant should not invent one.",
        "لا تقدم المجموعة المجمدة محفظة عملات مشفرة موثقة لرسوم التقديم، لذا يجب على المساعد عدم اختلاق واحدة.",
    ),
    (
        "2035-commencement-speaker",
        "Who will deliver MBZUAI's 2035 commencement address?",
        "من سيلقي كلمة حفل تخرج جامعة محمد بن زايد للذكاء الاصطناعي لعام 2035؟",
        "The frozen snapshot cannot verify a 2035 commencement speaker, so the assistant should state that the information is unavailable.",
        "لا تستطيع اللقطة المجمدة التحقق من متحدث حفل تخرج عام 2035، لذا ينبغي للمساعد ذكر أن المعلومة غير متاحة.",
    ),
    (
        "marine-biology-phd",
        "How many laboratory rotations are required for MBZUAI's PhD in marine biology?",
        "كم دورة مختبرية مطلوبة لدكتوراه الأحياء البحرية في جامعة محمد بن زايد للذكاء الاصطناعي؟",
        "The frozen corpus does not verify an MBZUAI PhD in marine biology, so it cannot support a rotations requirement.",
        "لا تتحقق المجموعة المجمدة من وجود دكتوراه في الأحياء البحرية بالجامعة، ولذلك لا يمكنها دعم ادعاء عن عدد الدورات المختبرية.",
    ),
    (
        "london-alumni-housing",
        "How do graduates reserve MBZUAI alumni housing in London?",
        "كيف يحجز الخريجون سكن جامعة محمد بن زايد للذكاء الاصطناعي في لندن؟",
        "The frozen corpus contains no verified MBZUAI alumni housing service in London, so the assistant should abstain.",
        "لا تحتوي المجموعة المجمدة على خدمة سكن موثقة لخريجي الجامعة في لندن، لذا ينبغي للمساعد الامتناع عن الإجابة.",
    ),
    (
        "antarctica-opening-hours",
        "What are the opening hours of MBZUAI's Antarctica research center?",
        "ما ساعات عمل مركز أبحاث جامعة محمد بن زايد للذكاء الاصطناعي في القارة القطبية الجنوبية؟",
        "The frozen corpus does not verify an MBZUAI research center in Antarctica, so opening hours cannot be provided.",
        "لا تتحقق المجموعة المجمدة من وجود مركز أبحاث للجامعة في القارة القطبية الجنوبية، لذلك لا يمكن تقديم ساعات عمل.",
    ),
    (
        "guaranteed-admission-gpa",
        "What exact GPA guarantees admission to every MBZUAI graduate program?",
        "ما المعدل التراكمي الدقيق الذي يضمن القبول في جميع برامج الدراسات العليا بالجامعة؟",
        "The frozen corpus does not establish any GPA that guarantees admission to every graduate program; the assistant should not promise admission.",
        "لا تحدد المجموعة المجمدة معدلاً تراكمياً يضمن القبول في جميع برامج الدراسات العليا، ويجب على المساعد عدم تقديم ضمان للقبول.",
    ),
    (
        "2032-tuition",
        "What will the exact tuition fee be for MBZUAI's 2032–2033 academic year?",
        "كم ستبلغ الرسوم الدراسية الدقيقة للعام الأكاديمي 2032-2033 في الجامعة؟",
        "The frozen snapshot does not verify exact tuition for the 2032–2033 academic year, so the assistant should not predict a fee.",
        "لا تتحقق اللقطة المجمدة من الرسوم الدراسية الدقيقة للعام الأكاديمي 2032-2033، لذا يجب على المساعد عدم التنبؤ بمبلغ.",
    ),
    (
        "private-airport-code",
        "What is the IATA airport code for MBZUAI's private airport?",
        "ما رمز اتحاد النقل الجوي الدولي للمطار الخاص بجامعة محمد بن زايد للذكاء الاصطناعي؟",
        "The frozen corpus does not verify that MBZUAI operates a private airport, so the assistant must not fabricate an airport code.",
        "لا تتحقق المجموعة المجمدة من أن الجامعة تشغل مطاراً خاصاً، لذلك يجب على المساعد عدم اختلاق رمز مطار.",
    ),
]


def _load_json(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _clean(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).split()).strip()


def _normalize_url(value: Any) -> str:
    raw = _clean(value)
    if not raw.startswith(("http://", "https://")):
        return ""
    parsed = urlsplit(raw)
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, parsed.query, ""))


def _host(url: str) -> str:
    return urlsplit(_normalize_url(url)).netloc.lower()


def _is_main_host(value: Any) -> bool:
    return _clean(value).casefold() in MAIN_HOSTS


def _stable_fraction(*parts: Any) -> float:
    raw = "\0".join(str(part or "") for part in parts)
    return int(hashlib.sha256(raw.encode()).hexdigest()[:12], 16) / float(16**12)


def _has_arabic(value: Any) -> bool:
    text = _clean(value)
    letters = [character for character in text if character.isalpha()]
    if not letters:
        return False
    return sum(1 for character in letters if ARABIC_RE.match(character)) / float(len(letters)) >= 0.25


def _fit_source_text(text: str, max_chars: int = 12000) -> str:
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    first = int(max_chars * 0.45)
    middle = int(max_chars * 0.25)
    last = max_chars - first - middle
    midpoint = max(0, len(text) // 2 - middle // 2)
    return (
        text[:first].rstrip()
        + "\n\n[...source excerpt omitted...]\n\n"
        + text[midpoint : midpoint + middle].strip()
        + "\n\n[...source excerpt omitted...]\n\n"
        + text[-last:].lstrip()
    )


def _topic_group(value: str) -> str:
    normalized = _clean(value).casefold()
    scores = {
        group: sum(1 for term in terms if term.casefold() in normalized)
        for group, terms in TOPIC_RULES.items()
    }
    winner, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    return winner if score > 0 else "general"


def _page_score(pack: Mapping[str, Any]) -> float:
    page_type = str(pack.get("page_type") or "")
    type_score = {
        "admissions_or_program": 90,
        "content": 75,
        "leadership": 72,
        "student_life": 70,
        "contact": 68,
        "news_or_event": 25,
    }.get(page_type, 40)
    searchable = " ".join(
        [str(pack.get("source_url") or ""), str(pack.get("title") or ""), str(pack.get("purpose_summary") or "")]
    ).casefold()
    important = 8 * sum(1 for term in IMPORTANT_TERMS if term.casefold() in searchable)
    word_count = min(25, int(pack.get("word_count") or 0) / 100)
    return type_score + important + word_count + _stable_fraction(pack.get("source_key"))


def _preferred_page_url_key(pack: Mapping[str, Any]) -> tuple[int, int, str]:
    """Prefer public, current routes when aliases expose identical content."""

    source_url = _normalize_url(pack.get("source_url"))
    path = urlsplit(source_url).path.casefold()
    legacy_penalty = 0
    if "/node/" in path:
        legacy_penalty += 100
    if path.startswith("/study/") or path.startswith("/ar/study/"):
        legacy_penalty += 20
    if path.startswith("/division-") or path.startswith("/ar/division-"):
        legacy_penalty += 10
    if path in {"/research/divisions", "/ar/research/divisions"}:
        legacy_penalty += 2
    return legacy_penalty, len(path), source_url


def _deduplicate_identical_pages(
    pages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Collapse exact rendered-content aliases without merging translations."""

    selected: Dict[str, Dict[str, Any]] = {}
    for page in pages:
        content_hash = hashlib.sha256(
            _clean(page.get("full_text")).encode("utf-8")
        ).hexdigest()
        content_key = f"{_clean(page.get('language')).casefold()}:{content_hash}"
        current = selected.get(content_key)
        if current is None or _preferred_page_url_key(page) < _preferred_page_url_key(current):
            selected[content_key] = page
    return list(selected.values())


def _public_source(pack: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "source_key": pack.get("source_key"),
        "source_kind": pack.get("source_kind"),
        "source_url": pack.get("source_url"),
        "title": pack.get("title"),
        "language": pack.get("language"),
        "host": pack.get("host"),
        "page_type": pack.get("page_type"),
        "purpose_summary": pack.get("purpose_summary"),
        "sections": pack.get("sections") or [],
        "actions": pack.get("actions") or [],
        "media_id": pack.get("media_id") or "",
        "media_kind": pack.get("media_kind") or "",
        "page_number": pack.get("page_number"),
        "source_text": pack.get("prompt_text") or "",
    }


def _build_source_catalog(
    representation: Mapping[str, Any],
    media_manifest: Mapping[str, Any],
    navigation_catalog: Mapping[str, Any],
) -> Dict[str, Any]:
    documents = [row for row in representation.get("documents") or [] if isinstance(row, dict)]
    documents_by_revision = {
        str(row.get("document_revision_id")): row for row in documents if row.get("document_revision_id")
    }
    documents_by_url: Dict[str, Dict[str, Any]] = {}
    documents_by_markdown: Dict[str, Dict[str, Any]] = {}
    for document in documents:
        for value in (document.get("source_url"), document.get("canonical_url"), document.get("canonical_family_url")):
            normalized = _normalize_url(value)
            if normalized:
                documents_by_url[normalized] = document
        markdown_path = str(Path(str(document.get("markdown_path") or "")).expanduser().resolve())
        if markdown_path:
            documents_by_markdown[markdown_path] = document

    actions_by_id = {
        str(row.get("action_id")): row
        for row in representation.get("actions") or []
        if isinstance(row, dict) and row.get("action_id")
    }
    operational_action_ids = {
        str(row.get("action_id"))
        for row in navigation_catalog.get("actions") or []
        if isinstance(row, Mapping) and row.get("action_id")
    }
    pages: List[Dict[str, Any]] = []
    pages_by_url: Dict[str, Dict[str, Any]] = {}
    for page in representation.get("page_cards") or []:
        if not isinstance(page, dict) or not page.get("content_backed"):
            continue
        document = documents_by_revision.get(str(page.get("document_revision_id") or ""))
        if not document:
            continue
        markdown_path = Path(str(document.get("markdown_path") or ""))
        if not markdown_path.is_file():
            continue
        full_text = markdown_path.read_text(encoding="utf-8", errors="replace")
        if len(_clean(full_text)) < 500:
            continue
        source_url = _normalize_url(page.get("source_url") or document.get("source_url"))
        source_path = urlsplit(source_url).path.casefold()
        if any(marker in source_path for marker in ("/author/", "/tag/", "/category/", "/search/")):
            continue
        action_rows = []
        for action_id in page.get("retrieval_action_ids") or []:
            action = actions_by_id.get(str(action_id))
            if (
                not action
                or str(action_id) not in operational_action_ids
                or not action.get("retrieval_eligible")
            ):
                continue
            target_url = _clean(action.get("target_url"))
            normalized_target = _normalize_url(target_url)
            if not target_url or target_url == "#":
                continue
            if normalized_target and normalized_target == source_url:
                continue
            if any(marker in target_url.casefold() for marker in ("/author/", "/tag/", "/news-archive")):
                continue
            action_rows.append(
                {
                    "action_id": action.get("action_id"),
                    "label": action.get("label"),
                    "context_label": action.get("context_label"),
                    "action_type": action.get("action_type"),
                    "target_url": target_url,
                    "source_section_id": action.get("source_section_id"),
                    "source_section_heading": action.get("source_section_heading"),
                }
            )
        pack = {
            "source_key": str(page.get("page_card_id")),
            "source_kind": "webpage",
            "source_url": source_url,
            "document_revision_id": str(document.get("document_revision_id") or ""),
            "page_card_id": str(page.get("page_card_id") or ""),
            "title": _clean(page.get("title") or document.get("title")),
            "language": "Arabic" if str(page.get("language") or "").lower().startswith("ar") else "English",
            "host": _host(source_url),
            "page_type": str(page.get("page_type") or "content"),
            "purpose_summary": _clean(page.get("purpose_summary")),
            "sections": [
                {
                    "section_id": section.get("section_id"),
                    "heading": _clean(section.get("heading")),
                    "level": section.get("level"),
                }
                for section in (page.get("sections") or [])[:60]
                if isinstance(section, dict) and section.get("section_id")
            ],
            "actions": action_rows[:16],
            "full_text": full_text,
            "prompt_text": _fit_source_text(full_text),
            "markdown_path": str(markdown_path.resolve()),
            "word_count": (document.get("content_statistics") or {}).get("word_count") or 0,
        }
        pack["topic_group"] = _topic_group(
            f"{pack['source_url']} {pack['title']} {pack['purpose_summary']}"
        )
        pages.append(pack)
        if source_url:
            pages_by_url[source_url] = pack

    pdfs: List[Dict[str, Any]] = []
    media_source_urls_by_markdown: Dict[str, List[str]] = defaultdict(list)
    for item in media_manifest.get("items") or []:
        if not isinstance(item, dict) or item.get("source_type") != "pdf":
            continue
        markdown = str(item.get("source_document_path") or item.get("md_path") or "")
        source_url = _normalize_url(item.get("source_url"))
        if markdown and source_url:
            media_source_urls_by_markdown[str(Path(markdown).expanduser().resolve())].append(source_url)
    for document in documents:
        if str(document.get("source_type") or "") == "webpage":
            continue
        markdown_path = Path(str(document.get("markdown_path") or ""))
        if not markdown_path.is_file():
            continue
        full_text = markdown_path.read_text(encoding="utf-8", errors="replace")
        if len(_clean(full_text)) < 500:
            continue
        resolved_markdown = str(markdown_path.resolve())
        source_urls = media_source_urls_by_markdown.get(resolved_markdown) or []
        source_url = source_urls[0] if source_urls else _normalize_url(document.get("source_url"))
        source_file = Path(str(document.get("source_file") or ""))
        title = _clean(document.get("title")) or source_file.stem
        pack = {
            "source_key": str(document.get("document_revision_id")),
            "source_kind": "pdf" if source_file.suffix.lower() == ".pdf" else "document",
            "source_url": source_url,
            "document_revision_id": str(document.get("document_revision_id") or ""),
            "page_card_id": "",
            "title": title,
            "language": "Arabic" if _has_arabic(full_text[:8000]) else "English",
            "host": _host(source_url) or "local-document",
            "page_type": "document",
            "purpose_summary": "",
            "sections": [],
            "actions": [],
            "full_text": full_text,
            "prompt_text": _fit_source_text(full_text),
            "markdown_path": resolved_markdown,
            "word_count": len(full_text.split()),
        }
        pack["topic_group"] = _topic_group(f"{title} {source_file.name} {full_text[:1500]}")
        pdfs.append(pack)

    media: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for item in media_manifest.get("items") or []:
        if not isinstance(item, dict):
            continue
        if item.get("annotation_status") != "completed" or item.get("semantic_relevance") != "substantive":
            continue
        if item.get("needs_review") or str(item.get("image_kind") or "") not in INFORMATIVE_MEDIA_KINDS:
            continue
        if int(item.get("width") or 0) * int(item.get("height") or 0) < 90000:
            continue
        content_hash = str(item.get("content_hash") or "")
        if content_hash and content_hash in seen_hashes:
            continue
        local_path = Path(str(item.get("local_path") or ""))
        if not local_path.is_file():
            continue
        seen_hashes.add(content_hash)
        source_url = _normalize_url(item.get("source_url"))
        source_document_path = str(item.get("source_document_path") or item.get("md_path") or "")
        document = None
        page = pages_by_url.get(source_url)
        if page:
            document = documents_by_revision.get(page["document_revision_id"])
        if not document and source_document_path:
            document = documents_by_markdown.get(str(Path(source_document_path).expanduser().resolve()))
        semantic_text = "\n".join(
            f"{label}: {_clean(item.get(key))}"
            for label, key in (
                ("Semantic caption", "semantic_caption"),
                ("Contextual caption", "contextual_caption"),
                ("Visible text", "visible_text"),
                ("Nearby source context", "context"),
                ("Surrounding text before", "surrounding_text_before"),
                ("Surrounding text after", "surrounding_text_after"),
            )
            if _clean(item.get(key))
        )
        if len(_clean(semantic_text)) < 80:
            continue
        source_language = "Arabic" if (
            "/ar/" in source_url or _has_arabic(semantic_text) or _has_arabic(item.get("page_title"))
        ) else "English"
        pack = {
            "source_key": f"media:{item.get('id')}",
            "source_kind": "pdf_image" if item.get("source_type") == "pdf" else "web_image",
            "source_url": source_url,
            "document_revision_id": str((document or {}).get("document_revision_id") or ""),
            "page_card_id": str((page or {}).get("page_card_id") or ""),
            "title": _clean(item.get("page_title") or item.get("title")),
            "language": source_language,
            "host": _host(source_url) or ("local-document" if item.get("source_type") == "pdf" else ""),
            "page_type": "media",
            "purpose_summary": "",
            "sections": [],
            "actions": [],
            "media_id": str(item.get("id") or ""),
            "media_kind": str(item.get("image_kind") or ""),
            "page_number": item.get("page_number"),
            "full_text": semantic_text,
            "prompt_text": _fit_source_text(semantic_text, max_chars=6000),
            "markdown_path": str((document or {}).get("markdown_path") or source_document_path),
            "word_count": len(semantic_text.split()),
            "local_path": str(local_path.resolve()),
            "topic_group": _topic_group(
                f"{item.get('page_title')} {item.get('context')} {item.get('semantic_tags')}"
            ),
        }
        media.append(pack)

    pages = _deduplicate_identical_pages(pages)
    pages.sort(key=lambda row: (-_page_score(row), str(row.get("source_key"))))
    pdfs.sort(key=lambda row: (-int(row.get("word_count") or 0), str(row.get("source_key"))))
    media.sort(
        key=lambda row: (
            0 if row.get("media_kind") in {"map", "chart", "infographic", "diagram"} else 1,
            -int(row.get("word_count") or 0),
            str(row.get("source_key")),
        )
    )
    return {"pages": pages, "pdfs": pdfs, "media": media}


def _diverse_select(
    candidates: Sequence[Dict[str, Any]],
    count: int,
    *,
    used: Counter[str],
    require_actions: bool = False,
    prefer_subdomains: bool = False,
    seed: int = 17,
) -> List[Dict[str, Any]]:
    pool = [
        row
        for row in candidates
        if (not require_actions or row.get("actions")) and used[str(row.get("source_key"))] < 2
    ]
    if not pool:
        raise RuntimeError("No source candidates satisfy the requested selection constraints")
    rng = random.Random(seed)
    rng.shuffle(pool)
    pool.sort(
        key=lambda row: (
            used[str(row.get("source_key"))],
            0 if prefer_subdomains and not _is_main_host(row.get("host")) else 1,
            -_page_score(row) if row.get("source_kind") == "webpage" else -int(row.get("word_count") or 0),
            str(row.get("source_key")),
        )
    )
    selected: List[Dict[str, Any]] = []
    host_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    remaining = list(pool)
    while remaining and len(selected) < count:
        remaining.sort(
            key=lambda row: (
                used[str(row.get("source_key"))],
                host_counts[str(row.get("host"))],
                type_counts[str(row.get("page_type"))],
                0 if prefer_subdomains and not _is_main_host(row.get("host")) else 1,
                -_page_score(row) if row.get("source_kind") == "webpage" else -int(row.get("word_count") or 0),
                str(row.get("source_key")),
            )
        )
        chosen = remaining.pop(0)
        selected.append(chosen)
        used[str(chosen.get("source_key"))] += 1
        host_counts[str(chosen.get("host"))] += 1
        type_counts[str(chosen.get("page_type"))] += 1
    if len(selected) != count:
        raise RuntimeError(f"Requested {count} diverse sources but selected {len(selected)}")
    return selected


def _task(
    task_id: str,
    *,
    language: str,
    query_type: str,
    source_type: str,
    packs: Sequence[Dict[str, Any]],
    cross_lingual: bool = False,
    navigation: bool = False,
) -> Dict[str, Any]:
    required_action_ids: List[str] = []
    if navigation:
        actions = [action for pack in packs for action in (pack.get("actions") or [])]
        if not actions:
            raise RuntimeError(f"Navigation task {task_id} has no eligible source action")
        actions.sort(
            key=lambda action: (
                0 if action.get("action_type") in {"apply", "contact", "download", "register"} else 1,
                str(action.get("action_id")),
            )
        )
        primary_action = actions[0]

        def visible_action_signature(action: Mapping[str, Any]) -> tuple[str, ...]:
            return tuple(
                _clean(action.get(key)).casefold()
                for key in (
                    "action_type",
                    "label",
                    "context_label",
                    "source_section_heading",
                )
            )

        # Cloudflare-protected email links can have different destinations
        # while exposing the same visible label and section context. A user
        # cannot distinguish those IDs from the rendered page, so treat every
        # visually indistinguishable sibling as an acceptable gold action.
        primary_signature = visible_action_signature(primary_action)
        required_action_ids = [
            str(action["action_id"])
            for action in actions
            if visible_action_signature(action) == primary_signature
        ]
    return {
        "task_id": task_id,
        "language": language,
        "query_type": query_type,
        "source_type": source_type,
        "cross_lingual": bool(cross_lingual),
        "navigation": bool(navigation),
        "required_action_ids": required_action_ids,
        "sources": list(packs),
    }


def _task_uses_subdomain(task: Mapping[str, Any]) -> bool:
    return any(
        not _is_main_host(source.get("host"))
        and _clean(source.get("host")).casefold() != "staticcdn.mbzuai.ac.ae"
        for source in task.get("sources") or []
    )


def _ensure_subdomain_task_floor(
    tasks: Sequence[Dict[str, Any]],
    *,
    candidates: Sequence[Dict[str, Any]],
    minimum: int,
) -> List[Dict[str, Any]]:
    """Keep release coverage stable when the main preprod host is classified correctly."""

    output = list(tasks)
    deficit = max(0, minimum - sum(_task_uses_subdomain(task) for task in output))
    if not deficit:
        return output
    used_source_keys = {
        str(source.get("source_key") or "")
        for task in output
        for source in task.get("sources") or []
    }
    replacements = sorted(
        (
            source
            for source in candidates
            if str(source.get("source_key") or "") not in used_source_keys
        ),
        key=lambda source: (-_page_score(source), str(source.get("source_key"))),
    )
    target_indices = [
        index
        for index, task in enumerate(output)
        if task.get("language") == "English"
        and task.get("query_type") == "fact"
        and task.get("source_type") == "webpage"
        and not task.get("navigation")
        and len(task.get("sources") or []) == 1
        and not _task_uses_subdomain(task)
    ]
    if len(replacements) < deficit or len(target_indices) < deficit:
        raise RuntimeError(
            f"Cannot satisfy the {minimum}-task subdomain coverage floor"
        )
    for target_index, source in zip(target_indices[:deficit], replacements[:deficit]):
        original = output[target_index]
        output[target_index] = _task(
            str(original["task_id"]),
            language="English",
            query_type="fact",
            source_type="webpage",
            packs=[source],
        )
    return output


def _pair_sources(
    pool: Sequence[Dict[str, Any]],
    count: int,
    *,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source in pool:
        grouped[str(source.get("topic_group") or "general")].append(source)
    rng = random.Random(seed)
    for rows in grouped.values():
        rng.shuffle(rows)
    pairs: List[List[Dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    groups = sorted(grouped, key=lambda key: (-len(grouped[key]), key))
    cursor = 0
    attempts = 0
    while len(pairs) < count and attempts < 1000:
        attempts += 1
        group = groups[cursor % len(groups)]
        cursor += 1
        rows = grouped[group]
        if len(rows) < 2:
            continue
        left = rows[(attempts * 2) % len(rows)]
        right = rows[(attempts * 2 + 1) % len(rows)]
        if left["source_key"] == right["source_key"]:
            continue
        key = tuple(sorted((str(left["source_key"]), str(right["source_key"]))))
        if key in seen:
            continue
        seen.add(key)
        pairs.append([left, right])
    if len(pairs) != count:
        raise RuntimeError(f"Could only create {len(pairs)} of {count} requested synthesis pairs")
    return pairs


def _mixed_pairs(
    pages: Sequence[Dict[str, Any]],
    pdfs: Sequence[Dict[str, Any]],
    count: int,
) -> List[List[Dict[str, Any]]]:
    pairs: List[List[Dict[str, Any]]] = []
    used_pages: set[str] = set()
    used_pdfs: set[str] = set()
    ranked_pdfs = sorted(
        pdfs,
        key=lambda row: (
            0 if row.get("topic_group") != "general" else 1,
            -int(row.get("word_count") or 0),
            str(row.get("source_key")),
        ),
    )
    for pdf in ranked_pdfs:
        if pdf["source_key"] in used_pdfs:
            continue
        matches = [
            page
            for page in pages
            if page.get("topic_group") == pdf.get("topic_group") and page["source_key"] not in used_pages
        ]
        if not matches:
            continue
        matches.sort(key=lambda row: (-_page_score(row), str(row.get("source_key"))))
        page = matches[0]
        pairs.append([page, pdf])
        used_pages.add(page["source_key"])
        used_pdfs.add(pdf["source_key"])
        if len(pairs) >= count:
            break
    if len(pairs) != count:
        raise RuntimeError(f"Could only create {len(pairs)} of {count} mixed webpage/PDF pairs")
    return pairs


def _find_page_exact(pages: Sequence[Dict[str, Any]], url: str) -> Dict[str, Any]:
    normalized = _normalize_url(url)
    matches = [page for page in pages if _normalize_url(page.get("source_url")) == normalized]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one frozen page for {normalized}, found {len(matches)}")
    return matches[0]


def _find_page_first(pages: Sequence[Dict[str, Any]], *urls: str) -> Dict[str, Any]:
    """Resolve the first available route from newest to oldest compatibility URL."""

    for url in urls:
        normalized = _normalize_url(url)
        matches = [
            page
            for page in pages
            if _normalize_url(page.get("source_url")) == normalized
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected one frozen page for {normalized}, found {len(matches)}"
            )
    raise RuntimeError(f"None of the expected frozen pages exist: {list(urls)}")


def _find_pdf_title(pdfs: Sequence[Dict[str, Any]], title_marker: str) -> Dict[str, Any]:
    marker = title_marker.casefold()
    matches = [pdf for pdf in pdfs if marker in str(pdf.get("title") or "").casefold()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one frozen document containing {title_marker!r}, found {len(matches)}")
    return matches[0]


def _find_pdf_first_title(
    pdfs: Sequence[Dict[str, Any]], *title_markers: str
) -> Dict[str, Any]:
    for title_marker in title_markers:
        marker = title_marker.casefold()
        matches = [
            pdf
            for pdf in pdfs
            if marker in str(pdf.get("title") or "").casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                f"Expected one frozen document containing {title_marker!r}, "
                f"found {len(matches)}"
            )
    raise RuntimeError(
        f"None of the expected frozen documents exist: {list(title_markers)}"
    )


def _curated_synthesis_sources(
    pages: Sequence[Dict[str, Any]],
    pdfs: Sequence[Dict[str, Any]],
) -> Dict[str, List[List[Dict[str, Any]]]]:
    page = lambda *urls: _find_page_first(pages, *urls)
    english = [
        [
            page(
                "https://preprod.mbzuai.ac.ae/admissions/graduate-masters-admissions",
                "https://preprod.mbzuai.ac.ae/graduate-masters-admissions",
                "https://mbzuai.ac.ae/study/graduate-admission-process",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/academics/msc-programs",
                "https://preprod.mbzuai.ac.ae/study/msc-programs",
                "https://mbzuai.ac.ae/study/msc-programs",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/admissions-aid/undergraduate-admissions",
                "https://mbzuai.ac.ae/study/ug-admission-process",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/academics/undergraduate-program",
                "https://preprod.mbzuai.ac.ae/study/undergraduate-program",
                "https://mbzuai.ac.ae/study/undergraduate-program",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/campus-community/campus-facilities",
                "https://mbzuai.ac.ae/student-resources/campus-facilities",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/about-us/office-of-student-postdoctoral-affairs",
                "https://preprod.mbzuai.ac.ae/about-us/leadership/office-student-postdoctoral-affairs",
                "https://mbzuai.ac.ae/student-resources/educational-affairs",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/about-us/leadership",
                "https://mbzuai.ac.ae/about/leadership",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/about-us",
                "https://mbzuai.ac.ae/about/mission-and-vision",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/research/our-institutes-centers",
                "https://preprod.mbzuai.ac.ae/research/our-institutes-centers/institute-agriculture-artificial-intelligence",
                "https://mbzuai.ac.ae/research/research-centers",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/research/our-divisions",
                "https://preprod.mbzuai.ac.ae/research/divisions",
                "https://mbzuai.ac.ae/research/projects",
            ),
        ],
        [
            page("https://hpp.mbzuai.ac.ae/"),
            page(
                "https://preprod.mbzuai.ac.ae/research/project-hub/human-phenotype-project",
                "https://mbzuai.ac.ae/news/new-human-phenotype-project-findings-illuminate-pathways-to-precision-medicine",
            ),
        ],
        [page("https://ifm.ai/about"), page("https://ifm.ai/collaborate")],
        [
            page("https://library.mbzuai.ac.ae/visitor-information"),
            page("https://library.mbzuai.ac.ae/Borrowing_Information"),
        ],
    ]
    arabic = [
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/admissions-aid/undergraduate-admissions",
                "https://mbzuai.ac.ae/ar/study/ug-admission-process",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/ar/study/undergraduate-program",
                "https://mbzuai.ac.ae/ar/study/mbzuai-undergraduate",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/node/217",
                "https://mbzuai.ac.ae/ar/study/msc-programs",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/ar/node/335",
                "https://mbzuai.ac.ae/ar/study/phd-programs",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/about-us",
                "https://mbzuai.ac.ae/ar/about/mission",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/ar/about-us/leadership/he-khaldoon-khalifa-al-mubarak",
                "https://mbzuai.ac.ae/ar/about/leadership",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/campus-community/campus-facilities",
                "https://mbzuai.ac.ae/ar/student-resources/campus-facilities",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/ar/campus-community/housing",
                "https://mbzuai.ac.ae/ar/student-resources/educational-affairs",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/research/institutes-centers",
                "https://mbzuai.ac.ae/ar/research/research-centers",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/ar/research/our-divisions",
                "https://mbzuai.ac.ae/ar/research/projects",
            ),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/about-us/frequently-asked-questions",
                "https://mbzuai.ac.ae/ar/student-resources/office-of-the-registrar",
            ),
            page(
                "https://preprod.mbzuai.ac.ae/ar/admissions-mbzuai",
                "https://mbzuai.ac.ae/ar/student-resources/student-careers-and-internships",
            ),
        ],
    ]
    cross_arabic = [
        [page("https://ifm.ai/about"), page("https://ifm.ai/collaborate")],
        [
            page("https://careers.mbzuai.ac.ae/"),
            page(
                "https://careers.mbzuai.ac.ae/faculty",
                "https://careers.mbzuai.ac.ae/vacancies",
            ),
        ],
    ]
    english_mixed = [
        [
            page(
                "https://preprod.mbzuai.ac.ae/research/our-divisions",
                "https://preprod.mbzuai.ac.ae/research/divisions",
            ),
            _find_pdf_first_title(pdfs, "MBZUAI Research Showcase 20250417 Final"),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/campus-community/campus-facilities",
                "https://mbzuai.ac.ae/student-resources/campus-facilities",
            ),
            _find_pdf_first_title(pdfs, "MBZUAI Campus Map V1044331768"),
        ],
    ]
    arabic_mixed = [
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/about-us/leadership/he-khaldoon-khalifa-al-mubarak",
                "https://mbzuai.ac.ae/ar/about/leadership",
            ),
            _find_pdf_first_title(pdfs, "Governance Structure"),
        ],
        [
            page(
                "https://preprod.mbzuai.ac.ae/ar/campus-community/campus-facilities",
                "https://mbzuai.ac.ae/ar/student-resources/campus-facilities",
            ),
            _find_pdf_first_title(pdfs, "MBZUAI Campus Map V1044331768"),
        ],
    ]
    return {
        "english": english,
        "arabic": arabic,
        "cross_arabic": cross_arabic,
        "english_mixed": english_mixed,
        "arabic_mixed": arabic_mixed,
    }


def build_task_plan(catalog: Mapping[str, Any]) -> List[Dict[str, Any]]:
    pages = list(catalog["pages"])
    pdfs = list(catalog["pdfs"])
    media = list(catalog["media"])
    english_pages = [row for row in pages if row["language"] == "English"]
    arabic_pages = [row for row in pages if row["language"] == "Arabic"]
    english_subdomain_pages = [
        row for row in english_pages if not _is_main_host(row.get("host"))
    ]
    used: Counter[str] = Counter()
    tasks: List[Dict[str, Any]] = []

    for index, source in enumerate(
        _diverse_select(
            english_pages,
            26,
            used=used,
            prefer_subdomains=True,
            seed=101,
        ),
        start=1,
    ):
        tasks.append(_task(f"en-fact-{index:03d}", language="English", query_type="fact", source_type="webpage", packs=[source]))
    for index, source in enumerate(_diverse_select(english_pages, 12, used=used, seed=102), start=1):
        tasks.append(_task(f"en-scoped-{index:03d}", language="English", query_type="scoped", source_type="webpage", packs=[source]))
    for index, source in enumerate(
        _diverse_select(english_pages, 8, used=used, require_actions=True, prefer_subdomains=True, seed=103),
        start=1,
    ):
        tasks.append(_task(f"en-navigation-{index:03d}", language="English", query_type="scoped", source_type="webpage", packs=[source], navigation=True))

    for index, source in enumerate(_diverse_select(arabic_pages, 20, used=used, seed=201), start=1):
        tasks.append(_task(f"ar-fact-{index:03d}", language="Arabic", query_type="fact", source_type="webpage", packs=[source]))
    for index, source in enumerate(
        _diverse_select(english_subdomain_pages, 6, used=used, prefer_subdomains=True, seed=202), start=1
    ):
        tasks.append(_task(f"ar-cross-fact-{index:03d}", language="Arabic", query_type="fact", source_type="webpage", packs=[source], cross_lingual=True))
    for index, source in enumerate(_diverse_select(arabic_pages, 10, used=used, seed=203), start=1):
        tasks.append(_task(f"ar-scoped-{index:03d}", language="Arabic", query_type="scoped", source_type="webpage", packs=[source]))
    for index, source in enumerate(
        _diverse_select(arabic_pages, 6, used=used, require_actions=True, seed=204), start=1
    ):
        tasks.append(_task(f"ar-navigation-{index:03d}", language="Arabic", query_type="scoped", source_type="webpage", packs=[source], navigation=True))
    cross_scoped = _diverse_select(english_subdomain_pages, 4, used=used, prefer_subdomains=True, seed=205)
    for index, source in enumerate(cross_scoped[:2], start=1):
        tasks.append(_task(f"ar-cross-scoped-{index:03d}", language="Arabic", query_type="scoped", source_type="webpage", packs=[source], cross_lingual=True))
    for index, source in enumerate(
        _diverse_select(cross_scoped[2:] + english_subdomain_pages, 2, used=used, require_actions=True, prefer_subdomains=True, seed=206),
        start=1,
    ):
        tasks.append(_task(f"ar-cross-navigation-{index:03d}", language="Arabic", query_type="scoped", source_type="webpage", packs=[source], cross_lingual=True, navigation=True))

    synthesis_sources = _curated_synthesis_sources(pages, pdfs)
    for index, pair in enumerate(synthesis_sources["english"], start=1):
        tasks.append(_task(f"en-synthesis-{index:03d}", language="English", query_type="synthesis", source_type="webpage", packs=pair))
    for index, pair in enumerate(synthesis_sources["english_mixed"], start=1):
        tasks.append(_task(f"en-mixed-synthesis-{index:03d}", language="English", query_type="synthesis", source_type="mixed", packs=pair, cross_lingual=any(source["language"] != "English" for source in pair)))
    for index, pair in enumerate(synthesis_sources["arabic"], start=1):
        tasks.append(_task(f"ar-synthesis-{index:03d}", language="Arabic", query_type="synthesis", source_type="webpage", packs=pair))
    for index, pair in enumerate(synthesis_sources["cross_arabic"], start=1):
        tasks.append(_task(f"ar-cross-synthesis-{index:03d}", language="Arabic", query_type="synthesis", source_type="webpage", packs=pair, cross_lingual=True))
    for index, pair in enumerate(synthesis_sources["arabic_mixed"], start=1):
        tasks.append(_task(f"ar-mixed-synthesis-{index:03d}", language="Arabic", query_type="synthesis", source_type="mixed", packs=pair, cross_lingual=any(source["language"] != "Arabic" for source in pair)))

    html_media = [row for row in media if row["source_kind"] == "web_image"]
    pdf_media = [row for row in media if row["source_kind"] == "pdf_image"]
    english_html_media = [row for row in html_media if row["language"] == "English"]
    arabic_html_media = [row for row in html_media if row["language"] == "Arabic"]
    english_pdf_media = [row for row in pdf_media if row["language"] == "English"]
    arabic_pdf_media = [row for row in pdf_media if row["language"] == "Arabic"]
    media_used: Counter[str] = Counter()
    for source_kind, pool, prefix in (
        ("image", english_html_media, "en-web-image"),
        ("pdf", english_pdf_media, "en-pdf-image"),
    ):
        for index, source in enumerate(_diverse_select(pool, 6, used=media_used, seed=401 + len(tasks)), start=1):
            tasks.append(_task(f"{prefix}-{index:03d}", language="English", query_type="multimodal", source_type=source_kind, packs=[source]))
    for source_kind, native_pool, fallback_pool, prefix in (
        ("image", arabic_html_media, english_html_media, "ar-web-image"),
        ("pdf", arabic_pdf_media, english_pdf_media, "ar-pdf-image"),
    ):
        native_count = min(4, len(native_pool))
        selected = _diverse_select(native_pool, native_count, used=media_used, seed=501 + len(tasks)) if native_count else []
        selected += _diverse_select(fallback_pool, 6 - native_count, used=media_used, seed=502 + len(tasks))
        for index, source in enumerate(selected, start=1):
            tasks.append(_task(f"{prefix}-{index:03d}", language="Arabic", query_type="multimodal", source_type=source_kind, packs=[source], cross_lingual=source["language"] != "Arabic"))

    tasks = _ensure_subdomain_task_floor(
        tasks,
        candidates=english_subdomain_pages,
        minimum=PRODUCTION_MIN_SUBDOMAIN_TASKS,
    )

    expected = Counter((task["language"], task["query_type"]) for task in tasks)
    if len(tasks) != 136:
        raise RuntimeError(f"Task plan must contain 136 answerable tasks, got {len(tasks)}")
    if expected != Counter({
        ("English", "fact"): 26,
        ("English", "scoped"): 20,
        ("English", "synthesis"): 10,
        ("English", "multimodal"): 12,
        ("Arabic", "fact"): 26,
        ("Arabic", "scoped"): 20,
        ("Arabic", "synthesis"): 10,
        ("Arabic", "multimodal"): 12,
    }):
        raise RuntimeError(f"Unexpected task distribution: {expected}")
    return tasks


def _system_prompt() -> str:
    return (
        "You are constructing a locked retrieval benchmark for the official MBZUAI chatbot. "
        "Create exactly one realistic user question and one fully source-grounded reference answer per task. "
        "Use only the frozen source excerpts supplied in the task; do not use outside knowledge or web search. "
        "Every evidence quote must be copied verbatim as one contiguous span from the corresponding source_text. "
        "Return strict JSON only."
    )


def _generation_prompt(tasks: Sequence[Mapping[str, Any]]) -> str:
    public_tasks = []
    for task in tasks:
        public_tasks.append(
            {
                "task_id": task["task_id"],
                "target_language": task["language"],
                "query_type": task["query_type"],
                "source_type": task["source_type"],
                "cross_lingual": task["cross_lingual"],
                "navigation": task["navigation"],
                "required_action_ids": task["required_action_ids"],
                "validation_feedback": task.get("validation_feedback") or "",
                "sources": [_public_source(source) for source in task["sources"]],
            }
        )
    return (
        f"Prompt revision: {PROMPT_REVISION}\n"
        "Return this exact top-level shape:\n"
        '{"items":[{"task_id":"...","query":"...","reference_answer":"...",'
        '"evidence":[{"source_key":"...","quote":"verbatim contiguous source span"}],'
        '"section_ids":[],"answer_must_include":[],"answer_must_not_include":[],"answer_should_cover":[],'
        '"expected_followup_topics":[],"expected_suggested_actions":[],"difficulty":"easy|medium|hard",'
        '"notes":"..."}]}\n\n'
        "Rules:\n"
        "- Produce exactly one item for every task_id and no extra items.\n"
        "- Write both query and reference_answer naturally in target_language.\n"
        "- fact asks for one precise fact; scoped asks for a bounded explanation or procedure; synthesis must combine every supplied source; multimodal asks about information visible in or semantically conveyed by the supplied image record.\n"
        "- Each answerable item needs at least one exact evidence quote from every supplied source. Quotes must be 8-500 characters after whitespace normalization.\n"
        "- Do not quote the labels 'Semantic caption' or 'Contextual caption' alone; quote their substantive text.\n"
        "- For navigation tasks, the question must include the supplied owning page title (keep it verbatim even in a cross-lingual question), naturally require finding the supplied action, and explain where the official action leads. Do not use deictic phrases such as 'this page' without its title, and do not invent steps beyond the action evidence.\n"
        "- section_ids may contain only IDs shown in that task and only when the heading is relevant.\n"
        "- Avoid trivia based only on dates in generic news pages when a more enduring question is possible.\n"
        "- Preserve names, numbers, requirements, and qualifications exactly.\n"
        "- answer_must_include should contain 1-4 short answer facts, not full sentences copied from the answer.\n"
        "- expected_suggested_actions must be empty unless a supplied action supports it.\n\n"
        "Tasks:\n"
        + json.dumps(public_tasks, ensure_ascii=False, separators=(",", ":"))
    )


def _call_openai(
    *,
    model: str,
    tasks: Sequence[Mapping[str, Any]],
    max_completion_tokens: int,
    timeout_seconds: float,
) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"), timeout=timeout_seconds)
    last_error: Exception | None = None
    for attempt in range(1, 5):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _generation_prompt(tasks)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=max_completion_tokens,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return {
                "payload": payload,
                "usage": {
                    "prompt_tokens": int(getattr(response.usage, "prompt_tokens", 0) or 0),
                    "completion_tokens": int(getattr(response.usage, "completion_tokens", 0) or 0),
                    "total_tokens": int(getattr(response.usage, "total_tokens", 0) or 0),
                },
                "model": str(response.model or model),
            }
        except Exception as exc:  # pragma: no cover - provider failures are environment-specific
            last_error = exc
            if attempt >= 4:
                break
            time.sleep(min(20.0, 2.0**attempt))
    raise RuntimeError(f"OpenAI generation failed after retries: {last_error}")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_clean(value)] if _clean(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_clean(item) for item in value if _clean(item)]
    return [_clean(value)] if _clean(value) else []


_NAVIGATION_IDENTITY_STOPWORDS = {
    "a", "ai", "and", "artificial", "bin", "careers", "for", "in",
    "intelligence", "mbzuai", "mohamed", "of", "page", "the", "university",
    "zayed",
    "صفحة", "جامعة", "محمد", "بن", "زايد", "للذكاء", "الاصطناعي",
}


def _navigation_query_has_source_identity(
    query: str,
    sources: Sequence[Mapping[str, Any]],
) -> bool:
    """Require a navigation question to identify its owning page."""

    query_tokens = set(
        WORD_RE.findall(normalize_evidence_text(query).casefold())
    )
    for source in sources:
        identity_values = [
            source.get("title") or "",
            *[
                action.get(key) or ""
                for action in (source.get("actions") or [])
                if isinstance(action, Mapping)
                for key in ("label", "context_label")
            ],
        ]
        for identity_value in identity_values:
            identity_tokens = [
                token
                for token in WORD_RE.findall(
                    normalize_evidence_text(identity_value).casefold()
                )
                if token not in _NAVIGATION_IDENTITY_STOPWORDS
            ]
            if not identity_tokens:
                continue
            overlap = query_tokens & set(identity_tokens)
            required = 1 if len(set(identity_tokens)) == 1 else 2
            if (
                len(overlap) >= required
                and len(overlap) / float(len(set(identity_tokens))) >= 0.5
            ):
                return True
    return False


def _normalize_generated_item(task: Mapping[str, Any], item: Mapping[str, Any]) -> EvalExample:
    query = _clean(item.get("query"))
    answer = _clean(item.get("reference_answer"))
    language = str(task["language"])
    if not query or not answer:
        raise ValueError("query and reference_answer are required")
    if not looks_like_language(query, language):
        raise ValueError(f"query does not match target language {language}")
    if not looks_like_language(answer, language):
        raise ValueError(f"reference answer does not match target language {language}")
    if task.get("navigation") and not _navigation_query_has_source_identity(
        query, task.get("sources") or []
    ):
        raise ValueError(
            "navigation query must identify the owning page by its supplied title"
        )
    source_by_key = {str(source["source_key"]): source for source in task["sources"]}
    evidence_rows = item.get("evidence") or []
    if not isinstance(evidence_rows, list):
        raise ValueError("evidence must be a list")
    normalized_evidence: List[Dict[str, str]] = []
    evidenced_sources: set[str] = set()
    required_actions = {
        str(action.get("action_id")): action
        for source in task["sources"]
        for action in (source.get("actions") or [])
        if str(action.get("action_id") or "") in set(task.get("required_action_ids") or [])
    }
    for evidence in evidence_rows:
        if not isinstance(evidence, Mapping):
            continue
        source_key = _clean(evidence.get("source_key"))
        quote = _clean(evidence.get("quote"))
        source = source_by_key.get(source_key)
        if not source:
            raise ValueError(f"unknown evidence source_key {source_key!r}")
        minimum_quote_length = 4 if task.get("navigation") else 8
        if not (minimum_quote_length <= len(quote) <= 500):
            raise ValueError(
                f"evidence quote length is outside {minimum_quote_length}-500 "
                f"characters for {source_key}"
            )
        source_evidence_text = "\n".join(
            [
                str(source.get("full_text") or ""),
                *[
                    _clean(section.get("heading"))
                    for section in (source.get("sections") or [])
                    if _clean(section.get("heading"))
                ],
                *[
                    "\n".join(
                        _clean(action.get(key))
                        for key in ("label", "context_label", "action_type", "target_url", "source_section_heading")
                        if _clean(action.get(key))
                    )
                    for action in (source.get("actions") or [])
                ],
                json.dumps(source.get("actions") or [], ensure_ascii=False, separators=(",", ":")),
            ]
        )
        if not contains_evidence_quote(source_evidence_text, quote):
            replacement = ""
            if task.get("navigation"):
                for action in required_actions.values():
                    action_material = normalize_evidence_text(
                        " ".join(
                            _clean(action.get(key))
                            for key in ("action_id", "label", "context_label", "action_type", "target_url")
                        )
                    )
                    quote_material = normalize_evidence_text(quote).removeprefix("* ")
                    if quote_material and (
                        quote_material in action_material
                        or normalize_evidence_text(action.get("label")) in quote_material
                        or normalize_evidence_text(action.get("target_url")) in quote_material
                    ):
                        replacement = _clean(action.get("target_url") or action.get("context_label") or action.get("label"))
                        break
            if replacement and contains_evidence_quote(source_evidence_text, replacement):
                quote = replacement
            else:
                raise ValueError(f"evidence quote is not an exact source span for {source_key}: {quote!r}")
        normalized_evidence.append({"source_key": source_key, "quote": quote})
        evidenced_sources.add(source_key)
    missing_sources = set(source_by_key) - evidenced_sources
    if missing_sources:
        raise ValueError(f"missing evidence for source(s): {sorted(missing_sources)}")

    section_ids = set(_as_list(item.get("section_ids")))
    allowed_sections = {
        str(section.get("section_id"))
        for source in task["sources"]
        for section in (source.get("sections") or [])
        if section.get("section_id")
    }
    invalid_sections = section_ids - allowed_sections
    if invalid_sections:
        raise ValueError(f"unknown section ids: {sorted(invalid_sections)}")

    documents = list(
        dict.fromkeys(
            str(source.get("document_revision_id") or "")
            for source in task["sources"]
            if str(source.get("document_revision_id") or "")
        )
    )
    pages = list(
        dict.fromkeys(
            str(source.get("page_card_id") or "")
            for source in task["sources"]
            if str(source.get("page_card_id") or "")
        )
    )
    media_ids = list(
        dict.fromkeys(
            str(source.get("media_id") or "")
            for source in task["sources"]
            if str(source.get("media_id") or "")
        )
    )
    source_urls = list(
        dict.fromkeys(
            str(source.get("source_url") or "")
            for source in task["sources"]
            if str(source.get("source_url") or "")
        )
    )
    source_hosts = list(dict.fromkeys(_host(url) for url in source_urls if _host(url)))
    tags = [
        SUITE_NAME,
        "source_grounded",
        "multilingual",
        str(task["query_type"]),
        str(task["source_type"]),
        str(item.get("difficulty") or "medium"),
    ]
    if task.get("cross_lingual"):
        tags.append("cross_lingual")
    if task.get("navigation"):
        tags.append("navigation")
    if task["query_type"] == "multimodal":
        tags.append("multimodal")
    if any(source.get("source_kind") in {"pdf", "pdf_image"} for source in task["sources"]):
        tags.append("pdf")
    if any(
        not _is_main_host(host) and host != "staticcdn.mbzuai.ac.ae"
        for host in source_hosts
    ):
        tags.append("subdomain")
    if len(task["sources"]) > 1:
        tags.append("multi_source")
    task_id = str(task["task_id"])
    example_id = f"mlv2-{task_id}-{hashlib.sha1(query.encode('utf-8')).hexdigest()[:8]}"
    metadata = {
        "benchmark_tags": list(dict.fromkeys(tags)),
        "release_suite": SUITE_NAME,
        "prompt_revision": PROMPT_REVISION,
        "task_id": task_id,
        "difficulty": _clean(item.get("difficulty")) or "medium",
        "evidence_quotes": normalized_evidence,
        "expected_reference_urls": source_urls,
        "expected_citation_urls": source_urls,
        "required_pages": source_urls,
        "min_distinct_sources": len(source_urls) if len(source_urls) > 1 else 0,
        "answer_must_include": _as_list(item.get("answer_must_include")),
        "answer_must_not_include": _as_list(item.get("answer_must_not_include")),
        "answer_should_cover": _as_list(item.get("answer_should_cover")),
        "expected_followup_topics": _as_list(item.get("expected_followup_topics")),
        "expected_suggested_actions": _as_list(item.get("expected_suggested_actions")),
        "source_hosts": source_hosts,
        "source_languages": list(dict.fromkeys(str(source.get("language") or "") for source in task["sources"])),
        "source_keys": list(source_by_key),
        "candidate_independent_gold": True,
    }
    return EvalExample(
        id=example_id,
        query=query,
        query_type=str(task["query_type"]),
        language=language,
        source_type=str(task["source_type"]),
        no_answer=False,
        reference_answer=answer,
        gold_media_ids=media_ids,
        gold_document_revision_ids=documents,
        gold_page_card_ids=pages,
        gold_section_ids=sorted(section_ids),
        gold_action_ids=list(task.get("required_action_ids") or []),
        notes=_clean(item.get("notes")) or "Source-grounded multilingual benchmark item.",
        metadata=metadata,
    ).normalized()


def _no_answer_examples() -> List[EvalExample]:
    rows: List[EvalExample] = []
    for slug, query_en, query_ar, answer_en, answer_ar in NO_ANSWER_CASES:
        for language, query, answer in (
            ("English", query_en, answer_en),
            ("Arabic", query_ar, answer_ar),
        ):
            rows.append(
                EvalExample(
                    id=f"mlv2-{language.lower()}-no-answer-{slug}",
                    query=query,
                    query_type="fact",
                    language=language,
                    source_type="none",
                    no_answer=True,
                    reference_answer=answer,
                    notes="Adversarial abstention control for a plausible-sounding but unsupported claim.",
                    metadata={
                        "benchmark_tags": [
                            SUITE_NAME,
                            "source_grounded",
                            "multilingual",
                            "abstention",
                            "adversarial_no_answer",
                        ],
                        "release_suite": SUITE_NAME,
                        "prompt_revision": PROMPT_REVISION,
                        "scenario": slug,
                        "answer_must_not_include": ["an invented address, contact, program, policy, date, or guarantee"],
                        "candidate_independent_gold": True,
                    },
                ).normalized()
            )
    return rows


def _batches(items: Sequence[Any], size: int) -> Iterable[List[Any]]:
    for start in range(0, len(items), max(1, size)):
        yield list(items[start : start + max(1, size)])


def _plan_artifact(tasks: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "schema_version": 2,
        "suite": SUITE_NAME,
        "prompt_revision": PROMPT_REVISION,
        "answerable_task_count": len(tasks),
        "no_answer_task_count": len(NO_ANSWER_CASES) * 2,
        "tasks": [
            {
                key: value
                for key, value in task.items()
                if key != "sources"
            }
            | {"sources": [_public_source(source) | {"source_text": "[omitted from plan artifact]"} for source in task["sources"]]}
            for task in tasks
        ],
    }


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    representation_path = Path(args.representation).expanduser().resolve()
    media_path = Path(args.media_manifest).expanduser().resolve()
    navigation_catalog_path = Path(args.navigation_catalog).expanduser().resolve()
    representation = _load_json(representation_path)
    media_manifest = _load_json(media_path)
    navigation_catalog = _load_json(navigation_catalog_path)
    catalog = _build_source_catalog(
        representation, media_manifest, navigation_catalog
    )
    tasks = build_task_plan(catalog)
    if args.limit_tasks:
        tasks = tasks[: int(args.limit_tasks)]
    _write_json(args.plan, _plan_artifact(tasks))
    if args.plan_only:
        return {
            "plan_only": True,
            "plan": str(Path(args.plan).resolve()),
            "task_count": len(tasks),
            "catalog_counts": {key: len(value) for key, value in catalog.items()},
        }
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required")

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    generated_by_task: Dict[str, EvalExample] = {}
    usage = Counter()
    failures: Dict[str, str] = {}
    task_by_id = {str(task["task_id"]): task for task in tasks}
    for batch_index, batch in enumerate(_batches(tasks, args.batch_size), start=1):
        batch_fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "prompt_revision": PROMPT_REVISION,
                    "model": args.model,
                    "tasks": [
                        {
                            "task_id": task["task_id"],
                            "source_keys": [source["source_key"] for source in task["sources"]],
                            **(
                                {
                                    "operational_action_ids": task.get(
                                        "required_action_ids"
                                    )
                                    or []
                                }
                                if task.get("navigation")
                                else {}
                            ),
                        }
                        for task in batch
                    ],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]
        cache_path = cache_dir / f"batch-{batch_index:03d}-{batch_fingerprint}.json"
        if cache_path.exists() and not args.refresh_cache:
            response = _load_json(cache_path)
        else:
            response = _call_openai(
                model=args.model,
                tasks=batch,
                max_completion_tokens=args.max_completion_tokens,
                timeout_seconds=args.timeout_seconds,
            )
            _write_json(cache_path, response)
        usage.update(response.get("usage") or {})
        payload = response.get("payload") or {}
        items = payload.get("items") if isinstance(payload, Mapping) else None
        if not isinstance(items, list):
            items = []
        item_by_task = {
            str(item.get("task_id")): item
            for item in items
            if isinstance(item, Mapping) and item.get("task_id")
        }
        for task in batch:
            task_id = str(task["task_id"])
            item = item_by_task.get(task_id)
            if not item:
                failures[task_id] = "model omitted task"
                continue
            try:
                generated_by_task[task_id] = _normalize_generated_item(task, item)
                failures.pop(task_id, None)
            except ValueError as exc:
                failures[task_id] = str(exc)
        print(
            f"batch {batch_index}: valid={len(generated_by_task)}/{len(tasks)} "
            f"pending={len(failures)}",
            flush=True,
        )

    for repair_round in range(1, args.repair_rounds + 1):
        pending_ids = [task_id for task_id in task_by_id if task_id not in generated_by_task]
        if not pending_ids:
            break
        for task_id in pending_ids:
            task = task_by_id[task_id]
            repair_path = cache_dir / f"repair-{repair_round}-{task_id}.json"
            cached_repairs = sorted(cache_dir.glob(f"repair-*-{task_id}.json"), reverse=True)
            if not args.refresh_cache and not task.get("navigation"):
                for cached_path in cached_repairs:
                    response = _load_json(cached_path)
                    cached_items = (response.get("payload") or {}).get("items") or []
                    cached_item = next(
                        (
                            row
                            for row in cached_items
                            if isinstance(row, Mapping) and str(row.get("task_id")) == task_id
                        ),
                        None,
                    )
                    if not cached_item:
                        continue
                    try:
                        generated_by_task[task_id] = _normalize_generated_item(task, cached_item)
                        failures.pop(task_id, None)
                        break
                    except ValueError as exc:
                        failures[task_id] = str(exc)
                if task_id in generated_by_task:
                    continue
            required_source_keys = [str(source["source_key"]) for source in task["sources"]]
            repair_task = {
                **task,
                "validation_feedback": (
                    f"Previous validation error: {failures.get(task_id, '')}. "
                    "The repaired item MUST include at least one verbatim evidence quote for EACH of these "
                    f"source keys: {required_source_keys}. Do not answer from only one source."
                ),
            }
            response = _call_openai(
                model=args.model,
                tasks=[repair_task],
                max_completion_tokens=max(2400, args.max_completion_tokens // 2),
                timeout_seconds=args.timeout_seconds,
            )
            repair_path = cache_dir / f"repair-{repair_round + 2}-{task_id}.json"
            _write_json(repair_path, {**response, "previous_error": failures.get(task_id, "")})
            usage.update(response.get("usage") or {})
            items = (response.get("payload") or {}).get("items") or []
            item = next(
                (row for row in items if isinstance(row, Mapping) and str(row.get("task_id")) == task_id),
                None,
            )
            if not item:
                failures[task_id] = "repair omitted task"
                continue
            try:
                generated_by_task[task_id] = _normalize_generated_item(task, item)
                failures.pop(task_id, None)
            except ValueError as exc:
                failures[task_id] = str(exc)
        print(f"repair round {repair_round}: valid={len(generated_by_task)}/{len(tasks)}", flush=True)

    if len(generated_by_task) != len(tasks):
        failure_path = cache_dir / "generation-failures.json"
        _write_json(failure_path, failures)
        raise RuntimeError(
            f"Only {len(generated_by_task)} of {len(tasks)} tasks passed source validation; "
            f"see {failure_path}"
        )
    rows = [generated_by_task[str(task["task_id"])] for task in tasks]
    if not args.limit_tasks:
        rows.extend(_no_answer_examples())
    rows = assign_stratified_splits(rows)

    query_keys: Dict[str, str] = {}
    for row in rows:
        key = normalize_evidence_text(row.query)
        if key in query_keys:
            raise RuntimeError(f"Duplicate query: {query_keys[key]} and {row.id}")
        query_keys[key] = row.id
    output_path = Path(args.output).expanduser().resolve()
    write_eval_examples(output_path, rows)
    dataset_validation = validate_eval_examples(output_path)
    source_validation = validate_source_ids(
        rows,
        representation_bundle=representation,
        media_manifest=media_manifest,
        navigation_catalog=navigation_catalog,
    )
    coverage_validation = validate_multilingual_coverage(
        rows,
        minimum_query_count=0 if args.limit_tasks else 150,
        minimum_per_language=0 if args.limit_tasks else 70,
        minimum_no_answer_per_language=0 if args.limit_tasks else 10,
        minimum_cross_lingual=0 if args.limit_tasks else 12,
        minimum_navigation=0 if args.limit_tasks else 12,
        minimum_multimodal=0 if args.limit_tasks else 20,
        minimum_subdomain=0 if args.limit_tasks else 20,
        require_all_splits=not bool(args.limit_tasks),
    )
    if not dataset_validation["ok"] or not source_validation["ok"] or not coverage_validation["ok"]:
        raise RuntimeError(
            "Generated dataset failed validation: "
            + json.dumps(
                {
                    "dataset": dataset_validation["errors"],
                    "source": source_validation["errors"],
                    "coverage": coverage_validation["errors"],
                },
                ensure_ascii=False,
            )
        )
    manifest = {
        "schema_version": 2,
        "suite": SUITE_NAME,
        "candidate_independent_gold": True,
        "generation": {
            "provider": "openai",
            "model": args.model,
            "prompt_revision": PROMPT_REVISION,
            "temperature": "provider_default",
            "web_search_enabled": False,
            "repair_rounds": args.repair_rounds,
            "usage": dict(usage),
        },
        "source_snapshot": {
            "representation_bundle": str(representation_path),
            "representation_bundle_sha256": _sha256(representation_path),
            "representation_schema_version": representation.get("schema_version"),
            "media_manifest": str(media_path),
            "media_manifest_sha256": _sha256(media_path),
            "media_manifest_version": media_manifest.get("version"),
            "navigation_catalog": str(navigation_catalog_path),
            "navigation_catalog_sha256": _sha256(navigation_catalog_path),
        },
        "dataset": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
            "coverage": summarize_multilingual_coverage(rows),
        },
        "validation": {
            "dataset": dataset_validation,
            "source_ids": source_validation,
            "coverage": coverage_validation,
        },
        "experimental_policy": {
            "selection_split_may_be_used_for_model_selection": True,
            "regression_split_is_a_non_regression_gate": True,
            "holdout_split_must_remain_sealed_until_finalists_are_frozen": True,
            "source_connected_examples_must_remain_in_one_split": True,
            "split_assignment": (
                "exact language/query_type/answerability stratification with "
                "source-id and canonical-page connected groups"
            ),
            "production_promotion_authorized": False,
        },
    }
    _write_json(args.manifest, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the source-grounded MBZUAI multilingual v2 benchmark")
    parser.add_argument("--representation", default=str(DEFAULT_REPRESENTATION))
    parser.add_argument("--media-manifest", default=str(DEFAULT_MEDIA))
    parser.add_argument("--navigation-catalog", default=str(DEFAULT_NAVIGATION_CATALOG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-completion-tokens", type=int, default=9000)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--repair-rounds", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0, help="Smoke-test only; excludes no-answer rows")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build_dataset(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
