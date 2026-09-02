"""
Adaptive hybrid retrieval for indexed pipeline runs.

Strategy:
- dense chunk retrieval for precision
- lexical BM25 retrieval for exact facts/names
- optional dense parent retrieval for broader queries
- optional dense media retrieval when the query is visually oriented
- conditional parent expansion at section/page granularity
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from collections.abc import Mapping, MutableMapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

try:
    from rank_bm25 import BM25Okapi
except Exception:  # pragma: no cover - exercised via fallback behavior tests
    class BM25Okapi:  # type: ignore[override]
        def __init__(self, corpus: Sequence[Sequence[str]]):
            self.corpus = [list(doc or []) for doc in corpus]

        def get_scores(self, query_tokens: Sequence[str]) -> List[float]:
            query_set = set(query_tokens or [])
            scores: List[float] = []
            for doc in self.corpus:
                if not query_set or not doc:
                    scores.append(0.0)
                    continue
                doc_set = set(doc)
                scores.append(len(query_set & doc_set) / float(len(query_set)))
            return scores

from pipeline.core.config import load_config
from pipeline.core.answer_records import (
    _service_availability_subtype,
    answer_record_looks_valid,
    derive_answer_records_from_bundle,
)
from pipeline.core.assertions import build_answer_records_from_assertions
from pipeline.core.google_genai import import_genai, import_genai_types
from pipeline.core.io import load_json_safe
from pipeline.core.media import build_retrieval_documents, response_agent_media_instructions
from pipeline.core.navigation_intent import infer_navigation_context

logger = logging.getLogger(__name__)
_GEMINI_CLIENT_STATE = threading.local()
_DEFAULT_GEMINI_REQUEST_TIMEOUT_MS = 120_000
_LEGACY_VECTORSTORE_CONTRACT = "mbzuai_chatbot_legacy_v1"
_PINECONE_INDEX_HOST_CACHE: Dict[str, str] = {}
_PINECONE_INDEX_HOST_CACHE_LOCK = threading.Lock()
_QUERY_EMBEDDING_CACHE_MAX = max(0, int(os.getenv("RETRIEVAL_QUERY_EMBEDDING_CACHE_SIZE", "4096") or "0"))
_QUERY_EMBEDDING_RETRIES = max(0, int(os.getenv("RETRIEVAL_QUERY_EMBEDDING_RETRIES", "1") or "0"))
_QUERY_EMBEDDING_RETRY_DELAY_SECONDS = max(
    0.0,
    float(os.getenv("RETRIEVAL_QUERY_EMBEDDING_RETRY_DELAY_SECONDS", "0.75") or "0"),
)
_GEMINI_QUERY_EMBEDDING_BATCH_SIZE = min(
    100,
    max(1, int(os.getenv("RETRIEVAL_QUERY_EMBEDDING_BATCH_SIZE", "100") or "100")),
)
_QUERY_EMBEDDING_FAILURE_COOLDOWN_SECONDS = max(
    0.0,
    float(os.getenv("RETRIEVAL_QUERY_EMBEDDING_FAILURE_COOLDOWN_SECONDS", "300") or "0"),
)
_TOKENIZE_CACHE_SIZE = max(0, int(os.getenv("RETRIEVAL_TOKENIZE_CACHE_SIZE", "32768") or "0"))
_TOKEN_SET_CACHE_SIZE = max(0, int(os.getenv("RETRIEVAL_TOKEN_SET_CACHE_SIZE", "32768") or "0"))
_SYMBOLIC_TOKENIZE_CACHE_SIZE = max(0, int(os.getenv("RETRIEVAL_SYMBOLIC_TOKENIZE_CACHE_SIZE", "32768") or "0"))
_SYMBOLIC_TOKEN_SET_CACHE_SIZE = max(0, int(os.getenv("RETRIEVAL_SYMBOLIC_TOKEN_SET_CACHE_SIZE", "32768") or "0"))
_QUERY_EMBEDDING_CACHE: "OrderedDict[Tuple[str, int | None, str, str], List[float]]" = OrderedDict()
_QUERY_EMBEDDING_CACHE_LOCK = threading.Lock()
_QUERY_EMBEDDING_FAILURE_STATE: Dict[Tuple[str, int | None, str], Tuple[float, str]] = {}
_QUERY_EMBEDDING_FAILURE_LOCK = threading.Lock()
_SHARED_LANE_EXECUTORS: Dict[Tuple[str, int], ThreadPoolExecutor] = {}
_SHARED_LANE_EXECUTORS_LOCK = threading.Lock()


def _shared_lane_executor(kind: str, max_workers: int) -> ThreadPoolExecutor:
    bounded_workers = max(1, int(max_workers or 1))
    key = (str(kind or "lane"), bounded_workers)
    with _SHARED_LANE_EXECUTORS_LOCK:
        executor = _SHARED_LANE_EXECUTORS.get(key)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=bounded_workers,
                thread_name_prefix=f"retrieval-{key[0]}-{bounded_workers}",
            )
            _SHARED_LANE_EXECUTORS[key] = executor
        return executor


def _is_local_lane_name(name: str) -> bool:
    return str(name or "").startswith("local_")


def _first_existing_stage_file(work_dir: Path, stage_ids: Sequence[str], filename: str) -> Path:
    for stage_id in stage_ids:
        candidate = work_dir / "stage_outputs" / stage_id / filename
        if candidate.exists():
            return candidate
    return work_dir / "stage_outputs" / stage_ids[0] / filename


def _legacy_vector_manifest_path(work_dir: Path) -> Path:
    return work_dir / "stage_outputs" / "upload_legacy_vectorstores" / "legacy_pinecone_upload_manifest.json"


def _modern_vector_manifest_path(work_dir: Path) -> Path:
    return work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"


def _apply_modern_vector_manifest_config(payload: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Any]:
    index_name = str(manifest.get("index_name") or "").strip()
    if not index_name:
        return payload

    embed_cfg = dict(payload.get("embedder") or {})
    retrieval_cfg = dict(payload.get("retrieval") or {})
    vector_store_cfg = dict(payload.get("vector_store") or {})
    provider = str(manifest.get("provider") or vector_store_cfg.get("provider") or "pinecone").strip().lower()
    if provider not in {"pinecone", "pgvector"}:
        raise ValueError(f"Unsupported vector upload manifest provider: {provider}")
    sparse_index_name = str(manifest.get("sparse_index_name") or "").strip()

    vector_store_cfg["provider"] = provider
    vector_store_cfg["release_id"] = str(manifest.get("namespace_release_id") or "").strip()
    if provider == "pinecone":
        embed_cfg["pinecone_index"] = index_name
        embed_cfg["pinecone_sparse_index"] = sparse_index_name
        retrieval_cfg["pinecone_index"] = index_name
        retrieval_cfg["pinecone_sparse_index"] = sparse_index_name
    else:
        embed_cfg.pop("pinecone_index", None)
        embed_cfg.pop("pinecone_sparse_index", None)
        retrieval_cfg.pop("pinecone_index", None)
        retrieval_cfg.pop("pinecone_sparse_index", None)
    embed_cfg.pop("pinecone_text_index", None)
    embed_cfg.pop("pinecone_summary_index", None)
    embed_cfg.pop("namespace", None)

    namespaces = manifest.get("namespaces") if isinstance(manifest.get("namespaces"), dict) else {}
    for key in (
        "chunks",
        "parents",
        "media",
        "page_cards",
        "actions",
        "facts",
        "evidence_spans",
        "summaries",
        "assertions",
        "entities",
        "communities",
    ):
        namespace = str(namespaces.get(key) or "").strip()
        if namespace:
            embed_cfg[f"namespace_{key}"] = namespace
            retrieval_cfg[f"namespace_{key}"] = namespace

    if manifest.get("model"):
        embed_cfg["model"] = manifest.get("model")
    dimension = manifest.get("output_dimensionality") or manifest.get("dimension")
    if dimension:
        embed_cfg["output_dimensionality"] = dimension

    stages = payload.get("stages")
    if isinstance(stages, list) and stages:
        has_modern_upload_stage = any(
            isinstance(stage, dict)
            and (
                str(stage.get("id") or "") == "upload_retrieval"
                or str(stage.get("plugin") or "") in {"gemini_pinecone", "gemini_pgvector"}
            )
            for stage in stages
        )
        if not has_modern_upload_stage:
            payload["stages"] = [
                *stages,
                {
                    "id": "upload_retrieval",
                    "type": "embedder",
                    "plugin": "gemini_pgvector" if provider == "pgvector" else "gemini_pinecone",
                },
            ]

    retrieval_cfg["enable_sparse"] = bool(sparse_index_name)
    for key in (
        "legacy_vectorstore_contract",
        "legacy_summary_index",
        "legacy_text_index",
        "legacy_namespace",
        "legacy_bm25_model_file",
    ):
        retrieval_cfg.pop(key, None)
    retrieval_cfg["legacy_hybrid_sparse_enabled"] = False

    payload["embedder"] = embed_cfg
    payload["retrieval"] = retrieval_cfg
    payload["vector_store"] = vector_store_cfg
    return payload


def apply_vector_upload_manifest_config(config: Dict[str, Any], work_dir: str | Path) -> Dict[str, Any]:
    """
    Align retrieval runtime config with the vectorstore contract that was actually uploaded.

    Older MBZUAI runs may carry a stale resolved_config.json with per-namespace
    Pinecone settings, while the production chatbot vectorstores are the legacy
    two-index contract: summary index + text index, both in the default namespace,
    with dense and sparse values stored together. The upload manifest is the
    authoritative release artifact for that deployed contract.
    """
    payload = deepcopy(config or {})
    resolved_work_dir = Path(work_dir).resolve()
    modern_manifest = load_json_safe(_modern_vector_manifest_path(resolved_work_dir), {}) or {}
    if isinstance(modern_manifest, dict) and str(modern_manifest.get("index_name") or "").strip():
        return _apply_modern_vector_manifest_config(payload, modern_manifest)

    manifest = load_json_safe(_legacy_vector_manifest_path(resolved_work_dir), {}) or {}
    if not isinstance(manifest, dict):
        return payload
    if str(manifest.get("vectorstore_contract") or "").strip() != _LEGACY_VECTORSTORE_CONTRACT:
        return payload

    text_index = str(manifest.get("text_index_name") or "").strip()
    summary_index = str(manifest.get("summary_index_name") or "").strip()
    namespace = str(manifest.get("namespace") or "").strip()
    if not text_index:
        return payload

    embed_cfg = dict(payload.get("embedder") or {})
    retrieval_cfg = dict(payload.get("retrieval") or {})

    embed_cfg["pinecone_index"] = text_index
    embed_cfg["pinecone_text_index"] = text_index
    if summary_index:
        embed_cfg["pinecone_summary_index"] = summary_index
    embed_cfg["pinecone_sparse_index"] = ""
    embed_cfg["namespace"] = namespace
    for key in (
        "namespace_chunks",
        "namespace_parents",
        "namespace_media",
        "namespace_facts",
        "namespace_evidence_spans",
        "namespace_summaries",
        "namespace_assertions",
    ):
        embed_cfg[key] = namespace
    if manifest.get("model"):
        embed_cfg["model"] = manifest.get("model")
    dimension = manifest.get("output_dimensionality") or manifest.get("dimension")
    if dimension:
        embed_cfg["output_dimensionality"] = dimension

    retrieval_cfg["legacy_vectorstore_contract"] = _LEGACY_VECTORSTORE_CONTRACT
    retrieval_cfg["legacy_summary_index"] = summary_index
    retrieval_cfg["legacy_text_index"] = text_index
    retrieval_cfg["legacy_namespace"] = namespace
    retrieval_cfg["legacy_hybrid_sparse_enabled"] = bool(manifest.get("use_sparse_embeddings"))
    retrieval_cfg["legacy_bm25_model_file"] = str(
        manifest.get("bm25_model_file") or manifest.get("legacy_bm25_model_file") or ""
    )
    retrieval_cfg.setdefault("hybrid_alpha", 0.5)
    # The legacy store keeps sparse values in the same Pinecone indexes. Do not
    # query a guessed sidecar sparse index; local lexical and hybrid dense lanes
    # cover sparse recall for this contract.
    retrieval_cfg["enable_sparse"] = False

    payload["embedder"] = embed_cfg
    payload["retrieval"] = retrieval_cfg
    return payload


_NAMESPACE_RECORD_TYPES = {
    "chunks": {"chunk"},
    "parents": {"parent"},
    "media": {"media"},
    "page_cards": {"page_card"},
    "actions": {"action"},
    "facts": {"fact"},
    "evidence_spans": {"evidence_span"},
    "summaries": {"summary"},
    "assertions": {"assertion"},
}


def _register_configured_namespace_aliases(
    *,
    configured_to_canonical: Mapping[str, str],
    token_indexes: MutableMapping[str, Dict[str, List[str]]],
    tokens_by_id: MutableMapping[str, Dict[str, set[str]]],
    bm25_indexes: MutableMapping[str, Tuple[List[str], BM25Okapi]],
) -> None:
    """Expose canonical lexical indexes under physical vector namespaces."""
    for configured_namespace, canonical_namespace in configured_to_canonical.items():
        if configured_namespace == canonical_namespace:
            continue
        if canonical_namespace not in token_indexes:
            continue
        token_indexes[configured_namespace] = token_indexes[canonical_namespace]
        tokens_by_id[configured_namespace] = tokens_by_id.get(canonical_namespace, {})
        if canonical_namespace in bm25_indexes:
            bm25_indexes[configured_namespace] = bm25_indexes[canonical_namespace]

_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "can", "do", "does", "for", "from",
    "have", "has", "how", "in", "is", "it", "many", "much", "of", "on", "or", "the",
    "their", "this", "to", "was", "were", "what", "when", "where", "which", "who", "why", "with",
}

_SOURCE_ANCHOR_STOPWORDS = {
    *_QUERY_STOPWORDS,
    "about",
    "academics",
    "ac",
    "ae",
    "ar",
    "en",
    "html",
    "https",
    "http",
    "mbzuai",
    "old",
    "page",
    "pages",
    "program",
    "programs",
    "resources",
    "study",
    "www",
}

_INSTITUTION_CONTEXT_TOKENS = {"mbzuai"}

_ANSWER_FOCUS_STOPWORDS = {
    *_SOURCE_ANCHOR_STOPWORDS,
    "bachelor",
    "comprehensive",
    "current",
    "detail",
    "details",
    "doctor",
    "include",
    "including",
    "master",
    "mbzuai",
    "msc",
    "number",
    "offer",
    "offered",
    "phd",
    "philosophy",
    "provide",
    "requirement",
    "requirements",
    "science",
    "specific",
}

_EVIDENCE_SPAN_ANCHOR_STOPWORDS = {
    *_SOURCE_ANCHOR_STOPWORDS,
    "application",
    "applications",
    "apply",
    "date",
    "dates",
    "deadline",
    "deadlines",
    "detail",
    "details",
    "document",
    "documents",
    "eligibility",
    "fee",
    "fees",
    "give",
    "important",
    "include",
    "includes",
    "including",
    "key",
    "requirement",
    "requirements",
    "tell",
}

_FACT_ATTRIBUTE_TOKENS = {
    "address",
    "contact",
    "city",
    "email",
    "hours",
    "location",
    "located",
    "named",
    "name",
    "number",
    "office",
    "phone",
    "telephone",
    "website",
}

_FACT_QUERY_PHRASES = {
    "in which city",
    "what city",
    "whose name",
    "carry the name",
    "named after",
}

_LOOKUP_ENTITY_ROLE_TOKENS = {
    "committee",
    "desk",
    "department",
    "group",
    "office",
    "staff",
    "team",
    "unit",
}

_GENERIC_CONTACT_QUERY_TOKENS = {
    "contact",
    "contacts",
    "reach",
    "reaching",
}

_GENERIC_CONTACT_SUBTYPES = {
    "contact",
    "contact_point",
    "admissions_contact",
    "general_contact",
    "office_contact",
}

_CONTACT_NARROW_SCOPE_TOKENS = {
    "after",
    "aid",
    "application",
    "applications",
    "document",
    "documents",
    "financial",
    "further",
    "information",
    "internship",
    "inquiries",
    "inquiry",
    "media",
    "more",
    "press",
    "program",
    "programs",
    "scholarship",
    "scholarships",
    "submission",
    "submitted",
    "support",
    "technical",
    "update",
    "updates",
}

_SUBORDINATE_LOCATION_SUBJECT_TOKENS = {
    "auditorium",
    "building",
    "buildings",
    "camp",
    "campus",
    "center",
    "centre",
    "hall",
    "lab",
    "labs",
    "library",
    "medical",
    "multi",
    "office",
    "room",
    "venue",
}

_LOOKUP_ATTRIBUTE_RULES = {
    "location": {
        "tokens": {"address", "metro", "station", "عنوان", "محطة"},
        "phrases": {"campus address", "metro station", "train station", "عنوان الحرم الجامعي"},
        "strict": False,
    },
    "email": {
        "tokens": {"email", "mail", "بريد", "البريد", "إيميل"},
        "phrases": {"email address", "contact email", "email id", "البريد الإلكتروني", "عنوان البريد"},
        "strict": True,
    },
    "phone": {
        "tokens": {"phone", "telephone", "mobile", "hotline", "extension", "ext", "هاتف", "الهاتف"},
        "phrases": {"phone number", "telephone number", "extension number", "contact number", "رقم الهاتف", "رقم التواصل"},
        "strict": True,
    },
    "website": {
        "tokens": {"website", "url", "web", "webpage", "link", "رابط"},
        "phrases": {"website address", "web address", "official website", "الموقع الإلكتروني", "الموقع الرسمي"},
        "strict": True,
    },
    "hours": {
        "tokens": {"hours", "hour", "time", "times", "opening", "working", "operating", "ساعات", "الدوام", "مواعيد"},
        "phrases": {"working hours", "operating hours", "opening hours", "ساعات العمل", "ساعات الدوام", "مواعيد العمل"},
        "strict": False,
    },
    "date": {
        "tokens": {"deadline", "date", "dates", "due", "start", "starts", "begin", "begins", "موعد", "الموعد", "تاريخ"},
        "phrases": {"application deadline", "start date", "semester start", "الموعد النهائي", "تاريخ البدء"},
        "strict": False,
    },
}

_CONTACT_LOOKUP_TYPES = {"email", "phone", "website"}
_EXPLICIT_LOCATION_LOOKUP_TOKENS = {
    "campus",
    "city",
    "emirate",
    "location",
    "located",
    "map",
    "metro",
    "postal",
    "road",
    "station",
    "street",
    "where",
}
_INSTITUTIONAL_AFFILIATION_SUBTYPES = {
    "entity_relation",
    "executive_council_affiliation",
    "government_affiliation",
    "institutional_affiliation",
}
_NON_INSTITUTIONAL_AFFILIATION_SUBTYPE_TOKENS = {
    "academic",
    "advis",
    "alumn",
    "author",
    "collaboration",
    "component",
    "department",
    "employer",
    "faculty",
    "fund",
    "group",
    "host",
    "industry",
    "instructor",
    "mentor",
    "news",
    "organizer",
    "owner",
    "partner",
    "policy",
    "professor",
    "program",
    "project",
    "research",
    "school",
    "specialization",
    "sponsor",
    "startup",
    "student",
    "website",
}
_FOUNDING_LEGAL_BASIS_SUBTYPES = {
    "establishment_law",
    "founding_law",
    "policy_relation",
}
_NON_FOUNDING_LEGAL_BASIS_SUBTYPE_TOKENS = {
    "accredit",
    "appointment",
    "compliance",
    "ethic",
    "fund",
    "generated",
    "generator",
    "health",
    "policy",
    "privacy",
    "property",
    "standards",
}

_LOOKUP_ATTRIBUTE_TOKENS = {
    token
    for payload in _LOOKUP_ATTRIBUTE_RULES.values()
    for token in payload["tokens"]
}
_LOOKUP_CONTEXT_QUALIFIER_TOKENS = {
    "ug",
    "pg",
    "undergraduate",
    "graduate",
    "phd",
    "doctoral",
    "master",
    "masters",
    "registrar",
    "finance",
    "career",
    "internship",
    "campus",
    "life",
    "helpdesk",
    "research",
    "facilities",
    "security",
    "provost",
    "president",
}
_LOOKUP_QUALIFIER_EQUIVALENTS = {
    "ug": {"undergraduate"},
    "undergraduate": {"ug"},
    "pg": {"graduate", "postgraduate"},
    "graduate": {"pg", "postgraduate"},
    "postgraduate": {"pg", "graduate"},
    "phd": {"doctoral"},
    "doctoral": {"phd"},
    "master": {"masters"},
    "masters": {"master"},
}
for payload in _LOOKUP_ATTRIBUTE_RULES.values():
    for phrase in payload["phrases"]:
        _LOOKUP_ATTRIBUTE_TOKENS.update(
            part.strip().lower()
            for part in re.findall(r"[A-Za-z0-9]+", phrase)
            if part.strip()
        )

_EMAIL_RE = re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?:(?:\+\d{1,3}[\s\-]?)?(?:\(?\d{2,4}\)?[\s\-]?){2,}\d{3,4})")
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:ac\.ae|edu|com|org|net)\b", re.IGNORECASE)
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\s*(?:am|pm)\b", re.IGNORECASE)
_DATE_HINT_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|"
    r"sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    re.IGNORECASE,
)
_LOOKUP_FRAGMENT_SPLIT_RE = re.compile(r"(?:[\n\r|;•]+|(?<=[.!?])\s+(?=[A-Z0-9]))")


@dataclass(frozen=True)
class LookupQueryProfile:
    answer_types: Tuple[str, ...]
    focus_tokens: Tuple[str, ...]
    strict_answer_required: bool

    @property
    def is_exact_lookup(self) -> bool:
        return bool(self.answer_types)

    @property
    def is_contact_lookup(self) -> bool:
        return bool({"email", "phone", "website"} & set(self.answer_types))


@dataclass(frozen=True)
class QueryAnswerSlot:
    answer_type: str
    qualifier: str = ""


@dataclass(frozen=True)
class QueryIntent:
    answer_types: Tuple[str, ...]
    slots: Tuple[QueryAnswerSlot, ...]
    subject_tokens: Tuple[str, ...]
    subject_phrases: Tuple[str, ...]
    strict_answer_required: bool

    @property
    def requested_role_subtypes(self) -> Tuple[str, ...]:
        return tuple(
            slot.qualifier
            for slot in self.slots
            if slot.answer_type == "role_holder" and slot.qualifier
        )

_SCOPED_QUERY_TOKENS = {
    "amenity",
    "amenities",
    "affiliation",
    "affiliated",
    "authority",
    "curriculum",
    "established",
    "facilities",
    "facility",
    "graduate",
    "graduates",
    "institutional",
    "law",
    "legal",
    "phd",
    "program",
    "programs",
    "services",
    "specialization",
    "specializations",
    "student-facing",
    "البرامج",
    "التخصصات",
    "الخدمات",
    "المرافق",
    "متطلبات",
    "القبول",
}

_SCOPED_QUERY_PHRASES = {
    "campus amenities",
    "core ai specializations",
    "everyday campus amenities",
    "graduate programs",
    "institutional affiliation",
    "legal basis",
    "law that created",
    "specialization areas",
    "under which law",
    "authority it is affiliated",
    "established under law",
    "student-facing campus services",
    "student facing campus services",
    "متطلبات القبول",
    "برامج الدراسات العليا",
    "مرافق الحرم الجامعي",
    "خدمات الطلاب",
}

_VISUAL_INTENT_TOKENS = {
    "building",
    "buildings",
    "campus",
    "chart",
    "diagram",
    "facility",
    "facilities",
    "figure",
    "figures",
    "framework",
    "image",
    "images",
    "infographic",
    "layout",
    "map",
    "parking",
    "plan",
    "pool",
    "show",
    "shown",
    "screenshot",
    "table",
    "visual",
    "صورة",
    "صور",
    "الصورة",
    "الصور",
    "الخريطة",
    "خريطة",
    "الرسم",
    "رسم",
    "الشكل",
    "شكل",
    "المخطط",
    "مخطط",
    "المرئي",
    "مرئي",
    "الفيديو",
    "فيديو",
    "الشريحة",
    "شريحة",
    "الإنفوغراف",
    "إنفوغراف",
    "الجدول",
    "جدول",
    "الإطار",
    "إطار",
    "لقطة",
    "لوحة",
    "اللوحة",
}

_EXPLICIT_VISUAL_QUERY_TOKENS = {
    "chart",
    "diagram",
    "figure",
    "figures",
    "framework",
    "image",
    "images",
    "infographic",
    "labelled",
    "labeled",
    "layout",
    "map",
    "plan",
    "poster",
    "screenshot",
    "show",
    "shown",
    "slide",
    "slides",
    "table",
    "video",
    "videos",
    "visual",
    "صورة",
    "صور",
    "الصورة",
    "الصور",
    "الخريطة",
    "خريطة",
    "الرسم",
    "رسم",
    "الشكل",
    "شكل",
    "المخطط",
    "مخطط",
    "المرئي",
    "مرئي",
    "الفيديو",
    "فيديو",
    "الشريحة",
    "شريحة",
    "الملصق",
    "ملصق",
    "الإنفوغراف",
    "إنفوغراف",
    "الجدول",
    "جدول",
    "الإطار",
    "إطار",
    "لقطة",
    "لوحة",
    "اللوحة",
}

_MEDIA_PRIORITY_TOKENS = {
    "assembly",
    "building",
    "buildings",
    "campus",
    "center",
    "centre",
    "facilities",
    "facility",
    "gym",
    "layout",
    "library",
    "labeled",
    "labelled",
    "map",
    "parking",
    "pool",
    "prayer",
    "residence",
    "residential",
    "residences",
    "الخريطة",
    "خريطة",
    "المخطط",
    "مخطط",
}

_LOW_SIGNAL_MEDIA_PHRASES = {
    "i'm sorry, but i cannot provide a description",
    "i cannot provide a description or answer",
    "doesn't contain any text or information",
    "does not contain any text or information",
}

_EVENT_SCHEDULE_TOKENS = {
    "agenda",
    "arrival",
    "booklet",
    "conference",
    "details",
    "keynote",
    "panel",
    "program",
    "registration",
    "schedule",
    "session",
    "speaker",
    "summit",
    "talk",
    "welcome",
    "workshop",
}

_FACILITY_HOURS_TOKENS = {
    "canteen",
    "gym",
    "library",
    "medical",
    "pool",
    "security",
    "space",
    "support",
}

_ROLE_QUERY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "president": (
        "president",
        "interim president",
        "university president",
    ),
    "provost": (
        "provost",
        "acting provost",
    ),
    "board_chair": (
        "chair of mbzuai's board of trustees",
        "chairman of mbzuai's board of trustees",
        "chair of the board of trustees",
        "chairman of the board of trustees",
        "board chair",
        "chairs mbzuai's board of trustees",
        "chairs the board of trustees",
    ),
    "vice_president_chief_of_staff": (
        "vice president and chief of staff",
        "vp and chief of staff",
    ),
}
_ROLE_QUERY_LABELS = {
    "president": "president",
    "provost": "provost",
    "board_chair": "chair of the Board of Trustees",
    "vice_president_chief_of_staff": "Vice President and Chief of Staff",
}
_ROLE_QUERY_TOKENS = {
    "president",
    "provost",
    "chair",
    "chairman",
    "board",
    "trustees",
    "vice",
    "chief",
    "staff",
}

_GENERIC_SUBJECT_REFERENCE_TOKENS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "university",
    "institution",
    "campus",
    "school",
    "program",
    "team",
    "office",
}


class QueryMode(str, Enum):
    FACT = "fact"
    SCOPED = "scoped"
    SYNTHESIS = "synthesis"


_ENUMERATION_QUERY_RE = re.compile(
    r"^(?:please\s+)?(?:list|name|enumerate|identify)\b"
    r"|\b(?:list|listing)\s+(?:all|every|the|of)\b"
    r"|\bhow\s+many\b"
    r"|\b(?:all|every|each|both)\b"
    r"|\bwhat\s+are\b"
    r"|\bwhich\s+(?:are|were)\b"
    r"|\b(?:what|which)\b.{0,100}\b(?:are\s+there|does\b.{0,40}\bhave|do\b.{0,40}\bhave)\b"
    # Auxiliary-led plural objects: "What documents must applicants
    # submit?", "Which programs can students choose?". Exact typed lookups
    # (hours, email, date, phone) are classified before this check.
    r"|\b(?:what|which)\s+(?:[\w'-]+\s+){0,3}[\w'-]+s\s+"
    r"(?:do|does|did|are|were|can|could|should|must|will|would|have|has)\b"
    r"|(?:اذكر|عدّد|عدد|جميع|كل|كافة|كم|ما\s+هي|ما\s+هما|أي\s+من)"
    # Arabic often omits the copula: "ما الوحدات ... الرئيسية؟". Detect
    # plural morphology or a plural-list adjective without encoding any
    # domain noun or expected answer.
    r"|^(?:ما|ماذا|أي|اي)\s+(?:هي\s+)?(?:ال)?[\u0600-\u06ff]+(?:ات|ون|ين)\b"
    r"|^(?:ما|ماذا|أي|اي)\s+.{1,100}\b(?:الرئيسية|المتاحة|الموجودة|المقدمة|المتوفرة|المعروضة)\b",
    re.IGNORECASE,
)


def _is_enumeration_query(query: str) -> bool:
    """Return whether answering requires a set rather than one isolated fact.

    This is deliberately domain-neutral. It recognizes linguistic request shape
    only; it never maps a question to a known answer, page, or MBZUAI fact.
    """

    normalized = _clean_text(query)
    if not normalized:
        return False
    return bool(_ENUMERATION_QUERY_RE.search(normalized))


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _token_variants(token: str) -> List[str]:
    value = "".join(ch for ch in str(token or "").lower() if ch.isalnum() or ch in {"-", "_", "'"})
    value = value.strip("'")
    if not value:
        return []
    if value.endswith("'s"):
        value = value[:-2]
    variants = [value]
    if value.endswith("ies") and len(value) > 4:
        variants.append(value[:-3] + "y")
    if value.endswith("s") and len(value) > 4 and not value.endswith("ss"):
        variants.append(value[:-1])
    return list(dict.fromkeys(item for item in variants if item))


@lru_cache(maxsize=_TOKENIZE_CACHE_SIZE)
def _tokenize_cached(text: str) -> Tuple[str, ...]:
    tokens: List[str] = []
    for token in _clean_text(text).split():
        tokens.extend(_token_variants(token))
    return tuple(token for token in tokens if token)


@lru_cache(maxsize=_TOKEN_SET_CACHE_SIZE)
def _token_set_cached(text: str) -> frozenset[str]:
    return frozenset(_tokenize_cached(text))


def _tokenize(text: str) -> List[str]:
    return list(_tokenize_cached(str(text or "")))


def _token_set(text: str) -> frozenset[str]:
    return _token_set_cached(str(text or ""))


@lru_cache(maxsize=_SYMBOLIC_TOKENIZE_CACHE_SIZE)
def _symbolic_tokens_cached(text: str) -> Tuple[str, ...]:
    tokens: List[str] = []
    for raw in re.split(r"[^a-z0-9]+", _clean_text(text).lower()):
        if not raw:
            continue
        tokens.extend(_token_variants(raw))
    return tuple(token for token in tokens if token)


@lru_cache(maxsize=_SYMBOLIC_TOKEN_SET_CACHE_SIZE)
def _symbolic_token_set_cached(text: str) -> frozenset[str]:
    return frozenset(_symbolic_tokens_cached(text))


def _symbolic_tokens(text: str) -> List[str]:
    return list(_symbolic_tokens_cached(str(text or "")))


def _symbolic_token_set(text: str) -> frozenset[str]:
    return _symbolic_token_set_cached(str(text or ""))


def _source_anchor_tokens(text: str) -> List[str]:
    return [
        token
        for token in _symbolic_tokens(text)
        if len(token) >= 3 and token not in _SOURCE_ANCHOR_STOPWORDS
    ]


def _evidence_span_anchor_tokens(query: str) -> List[str]:
    tokens = [
        token
        for token in [*_named_query_tokens(query), *_source_anchor_tokens(query)]
        if len(token) >= 3
        and token not in _EVIDENCE_SPAN_ANCHOR_STOPWORDS
        and token not in _INSTITUTION_CONTEXT_TOKENS
    ]
    return list(dict.fromkeys(tokens))


def _record_matches_anchor_tokens(record: Mapping[str, Any], anchor_tokens: Sequence[str]) -> bool:
    if not anchor_tokens:
        return True
    text = " ".join(
        str(record.get(key) or "")
        for key in (
            "text",
            "dense_text",
            "embedding_text",
            "sparse_text",
            "source_url",
            "canonical_url",
            "language_normalized_url",
            "document_title",
            "section_heading",
            "breadcrumb",
        )
    ).lower()
    if not text:
        return False
    symbolic = set(_symbolic_tokens(text))
    return any(token in symbolic or token in text for token in anchor_tokens)


def _source_url_path_text(source_url: str) -> str:
    try:
        parsed = urlparse(str(source_url or ""))
        value = unquote(parsed.path or str(source_url or ""))
    except Exception:
        value = str(source_url or "")
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


_OFFICIAL_SOURCE_URL_RE = re.compile(
    r"https?://(?:(?:[a-z0-9-]+\.)*mbzuai\.ac\.ae|(?:[a-z0-9-]+\.)*ifm\.ai|mbzuai\.gitbook\.io)/[^\s\]\)\"'<>,]+",
    re.IGNORECASE,
)


def _extract_official_source_url_from_text(*values: Any) -> str:
    for value in values:
        text = str(value or "")
        if not text:
            continue
        match = _OFFICIAL_SOURCE_URL_RE.search(text)
        if match:
            return match.group(0).rstrip(".,;:")
    return ""


def _record_source_url(record: Mapping[str, Any] | None, *fallback_records: Mapping[str, Any] | None) -> str:
    records = [record, *fallback_records]
    metadata_records: List[Mapping[str, Any]] = []
    for candidate in records:
        if not isinstance(candidate, Mapping):
            continue
        for key in (
            "source_url",
            "language_normalized_url",
            "canonical_url",
            "document_source",
            "page_source",
            "source",
            "url",
        ):
            value = str(candidate.get(key) or "").strip()
            if value:
                return value
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping):
            metadata_records.append(metadata)
    for metadata in metadata_records:
        for key in (
            "source_url",
            "language_normalized_url",
            "canonical_url",
            "document_source",
            "page_source",
            "source",
            "url",
        ):
            value = str(metadata.get(key) or "").strip()
            if value:
                return value
    text_candidates: List[Any] = []
    for candidate in [*records, *metadata_records]:
        if not isinstance(candidate, Mapping):
            continue
        for key in (
            "text",
            "dense_text",
            "embedding_text",
            "sparse_text",
            "document_title",
            "section_heading",
            "breadcrumb",
        ):
            text_candidates.append(candidate.get(key))
    return _extract_official_source_url_from_text(*text_candidates)


def _ensure_record_source_url(record: MutableMapping[str, Any], *fallback_records: Mapping[str, Any] | None) -> str:
    source_url = _record_source_url(record, *fallback_records)
    if source_url:
        record["source_url"] = source_url
    return source_url


def _source_phrase_bonus(query: str, source_anchor_text: str) -> float:
    query_tokens = _source_anchor_tokens(query)
    if len(query_tokens) < 2 or not source_anchor_text:
        return 0.0
    bonus = 0.0
    seen_phrases: set[str] = set()
    for width in range(min(5, len(query_tokens)), 1, -1):
        for idx in range(0, len(query_tokens) - width + 1):
            phrase_tokens = query_tokens[idx : idx + width]
            phrase = " ".join(phrase_tokens)
            if phrase in seen_phrases:
                continue
            seen_phrases.add(phrase)
            if phrase in source_anchor_text:
                if width >= 4:
                    bonus += 1.10
                elif width == 3:
                    bonus += 0.82
                else:
                    bonus += 0.52
    return min(bonus, 1.65)


def _source_identity_bonus(
    query: str,
    *,
    source_url: str = "",
    document_title: str = "",
    heading: str = "",
) -> float:
    """Prefer sources that explicitly identify a named query subject.

    Dense page-card recall can correctly identify a site or organization while
    generic synthesis terms (for example, ``research centers``) dominate the
    final chunk rerank. Only title/heading/URL identity fields participate in
    this bonus; body-text mentions are deliberately excluded so an unrelated
    page that merely mentions the entity cannot win the source match.
    """

    identity_tokens = {
        token
        for token in _named_query_tokens(query)
        if len(token) >= 3
        and token not in _SOURCE_ANCHOR_STOPWORDS
        and token not in _INSTITUTION_CONTEXT_TOKENS
    }
    if not identity_tokens:
        return 0.0

    try:
        parsed = urlparse(str(source_url or ""))
        host = parsed.netloc.lower()
        path = unquote(parsed.path or "")
    except Exception:
        host = ""
        path = str(source_url or "")
    identity_text = " ".join(
        part
        for part in (
            host,
            path,
            _clean_text(document_title).lower(),
            _clean_text(heading).lower(),
        )
        if part
    )
    source_tokens = set(_symbolic_tokens(identity_text))
    matched = identity_tokens & source_tokens
    if not matched:
        return 0.0

    match_ratio = len(matched) / float(len(identity_tokens))
    bonus = min(2.25, (0.45 * len(matched)) + (1.10 * match_ratio))
    host_tokens = {
        token
        for token in _symbolic_tokens(host)
        if token not in {"www", "ac", "ae", "ai", "com", "edu", "org"}
    }
    if matched & host_tokens:
        bonus += 1.25
    return bonus


def _answer_focus_tokens(query: str) -> List[str]:
    tokens = [
        token
        for token in _symbolic_tokens(query)
        if len(token) >= 4 and token not in _ANSWER_FOCUS_STOPWORDS
    ]
    return list(dict.fromkeys(tokens))


def _answer_focus_match_bonus(query: str, text: str) -> float:
    focus_tokens = _answer_focus_tokens(query)
    if not focus_tokens:
        return 0.0
    text_tokens = _symbolic_token_set(text)
    if not text_tokens:
        return 0.0
    hits = [token for token in focus_tokens if token in text_tokens]
    if not hits:
        return 0.0
    hit_ratio = len(hits) / float(len(focus_tokens))
    bonus = (0.22 * len(hits)) + (0.85 * hit_ratio)
    lower_text = _clean_text(text).lower()
    for phrase in ("screening exam", "research statement", "collaborative learning", "operating hours", "library contact"):
        if phrase in _clean_text(query).lower() and phrase in lower_text:
            bonus += 0.55
    if "operating hours" in _clean_text(query).lower() and "hours of operation" in lower_text:
        bonus += 0.55
    return min(bonus, 2.10)


def _email_local_part_tokens(value: str) -> set[str]:
    local_part = str(value or "").split("@", 1)[0]
    return set(_symbolic_tokens(local_part))


def _contact_scope_tokens(answer: Dict[str, Any], *, effective_subtype: str = "") -> set[str]:
    fragments: List[str] = []
    subtype = str(effective_subtype or answer.get("answer_subtype") or "")
    if subtype:
        fragments.append(subtype.replace("_", " "))
    subject_text = _clean_text(answer.get("subject_text") or "")
    if subject_text:
        fragments.append(subject_text)
    qualifiers = " ".join(str(value) for value in (answer.get("qualifiers") or []) if str(value))
    if qualifiers:
        fragments.append(qualifiers)
    scope_tokens = set(_tokenize(" ".join(fragments)))
    scope_tokens -= {"contact", "contacts", "email", "emails", "phone", "phones", "website", "websites"}
    return scope_tokens


def _is_generic_contact_query(query: str) -> bool:
    query_tokens = set(_tokenize(query))
    return bool(query_tokens & _GENERIC_CONTACT_QUERY_TOKENS) and not bool(
        query_tokens & {"email", "emails", "phone", "phones", "website", "websites", "telephone", "number"}
    )


def _website_lookup_requested(normalized_query: str, query_tokens: set[str]) -> bool:
    """Distinguish a URL lookup from a factual question that cites a website."""
    if query_tokens & {"url", "link", "رابط"}:
        return True
    if any(
        phrase in normalized_query
        for phrase in (
            "website address",
            "web address",
            "link to the website",
            "رابط الموقع",
            "عنوان الموقع الإلكتروني",
        )
    ):
        return True
    return bool(
        re.search(
            r"\b(?:what|which)\s+(?:is|are)\b[^?]{0,80}\b(?:official\s+)?website\b"
            r"|\bwhere\s+(?:can|do)\s+(?:i|we)\s+find\b[^?]{0,80}\bwebsite\b"
            r"|(?:ما|أين|اين)\s+(?:هو|هي|أجد|اجد)?\s*(?:رابط\s+)?(?:الموقع الإلكتروني|الموقع الرسمي)",
            normalized_query,
            flags=re.IGNORECASE,
        )
    )


@lru_cache(maxsize=2048)
def _lookup_query_profile(query: str) -> LookupQueryProfile:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    answer_types: List[str] = []
    strict_answer_required = False
    for answer_type, rule in _LOOKUP_ATTRIBUTE_RULES.items():
        if answer_type == "website":
            phrase_hit = _website_lookup_requested(normalized, query_tokens)
            token_hit = False
        else:
            phrase_hit = any(phrase in normalized for phrase in rule["phrases"])
            token_hit = bool(query_tokens & set(rule["tokens"]))
        if phrase_hit or token_hit:
            answer_types.append(answer_type)
            strict_answer_required = strict_answer_required or bool(rule["strict"])

    answer_type_set = set(answer_types)
    if "location" in answer_type_set and (_CONTACT_LOOKUP_TYPES & answer_type_set):
        explicit_location_signal = bool(query_tokens & _EXPLICIT_LOCATION_LOOKUP_TOKENS) or any(
            phrase in normalized
            for phrase in (
                "campus address",
                "office address",
                "mailing address",
                "postal address",
                "street address",
                "metro station",
                "train station",
                "where is",
                "where are",
            )
        )
        ambiguous_address_lookup = "address" in query_tokens and any(
            phrase in normalized for phrase in ("email address", "website address", "web address")
        )
        if ambiguous_address_lookup and not explicit_location_signal:
            answer_types = [answer_type for answer_type in answer_types if answer_type != "location"]

    if _is_generic_contact_query(query):
        # Email is the safest default for an unspecified contact request. Do
        # not broaden it to phone/website records, which can retrieve unrelated
        # offices and reduce citation precision. Preserve other requested
        # attributes such as opening hours in compound questions.
        non_contact_answer_types = [
            answer_type for answer_type in answer_types if answer_type not in _CONTACT_LOOKUP_TYPES
        ]
        answer_types = non_contact_answer_types + (
            ["email", "phone", "website"] if non_contact_answer_types else ["email"]
        )
        strict_answer_required = True

    if not answer_types:
        return LookupQueryProfile(answer_types=tuple(), focus_tokens=tuple(), strict_answer_required=False)

    attribute_tokens = set(_LOOKUP_ATTRIBUTE_TOKENS)
    focus_tokens = [
        token
        for token in _tokenize(query)
        if token not in _QUERY_STOPWORDS and len(token) >= 3 and token not in attribute_tokens
    ]
    if {"email", "phone", "website"} & set(answer_types):
        stripped = [token for token in focus_tokens if token not in _LOOKUP_ENTITY_ROLE_TOKENS]
        if stripped:
            focus_tokens = stripped
    if not focus_tokens:
        focus_tokens = [
            token
            for token in _tokenize(query)
            if token not in attribute_tokens and token not in _QUERY_STOPWORDS and len(token) >= 3
        ]
    return LookupQueryProfile(
        answer_types=tuple(dict.fromkeys(answer_types)),
        focus_tokens=tuple(dict.fromkeys(focus_tokens)),
        strict_answer_required=strict_answer_required,
    )


@lru_cache(maxsize=2048)
def _requested_role_subtypes(query: str) -> Tuple[str, ...]:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    roles: List[str] = []
    for canonical_role, aliases in _ROLE_QUERY_ALIASES.items():
        alias_hit = any(alias in normalized for alias in aliases)
        if canonical_role == "president" and alias_hit and {"vice", "chief", "staff"} <= query_tokens:
            alias_hit = False
        if alias_hit:
            roles.append(canonical_role)
            continue
        if canonical_role == "board_chair" and {"board", "trustee"} <= query_tokens and ("chair" in query_tokens or "chairman" in query_tokens):
            roles.append(canonical_role)
        elif canonical_role == "vice_president_chief_of_staff" and {"vice", "president", "chief", "staff"} <= query_tokens:
            roles.append(canonical_role)
        elif canonical_role == "provost" and "provost" in query_tokens:
            roles.append(canonical_role)
        elif canonical_role == "president" and "president" in query_tokens and not (
            {"vice", "chief", "staff"} <= query_tokens
        ) and not (
            {"board", "trustee"} & query_tokens and ("chair" in query_tokens or "chairman" in query_tokens)
        ):
            roles.append(canonical_role)
    return tuple(dict.fromkeys(roles))


@lru_cache(maxsize=2048)
def _query_intent(query: str) -> QueryIntent:
    lookup_profile = _lookup_query_profile(query)
    answer_types = list(_structured_answer_types(query))
    slots: List[QueryAnswerSlot] = []
    for answer_type in lookup_profile.answer_types:
        slots.append(QueryAnswerSlot(answer_type=answer_type))
    requested_roles = _requested_role_subtypes(query)
    for role in requested_roles:
        slots.append(QueryAnswerSlot(answer_type="role_holder", qualifier=role))
    subject_tokens = [
        token
        for token in _named_query_tokens(query)
        if token not in _ROLE_QUERY_TOKENS and token not in _LOOKUP_ENTITY_ROLE_TOKENS
    ]
    subject_phrases = [
        phrase
        for phrase in _named_query_phrases(query)
        if not any(token in _ROLE_QUERY_TOKENS for token in _tokenize(phrase))
    ]
    if not subject_tokens and answer_types:
        fallback_tokens, fallback_phrases = _fallback_subject_tokens_and_phrases(
            query,
            answer_types=tuple(answer_types),
        )
        subject_tokens = list(fallback_tokens)
        subject_phrases = list(fallback_phrases)
    strict_answer_required = bool(slots) or lookup_profile.strict_answer_required
    return QueryIntent(
        answer_types=tuple(dict.fromkeys(answer_types)),
        slots=tuple(slots),
        subject_tokens=tuple(dict.fromkeys(subject_tokens)),
        subject_phrases=tuple(dict.fromkeys(subject_phrases)),
        strict_answer_required=strict_answer_required,
    )


def _text_matches_answer_type(text: str, answer_type: str) -> bool:
    value = _clean_text(text)
    lower_value = value.lower()
    if not value:
        return False
    if answer_type == "email":
        return bool(_EMAIL_RE.search(value))
    if answer_type == "phone":
        return bool(_PHONE_RE.search(value))
    if answer_type == "website":
        return bool(_URL_RE.search(value)) or ".ac.ae" in lower_value or ".edu" in lower_value or ".com" in lower_value
    if answer_type == "hours":
        return bool(re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)\b", lower_value)) or "working hours" in lower_value or "operating hours" in lower_value
    if answer_type == "date":
        return bool(_DATE_HINT_RE.search(value)) or "deadline" in lower_value
    if answer_type == "location":
        return any(
            phrase in lower_value
            for phrase in (
                "located in",
                "located at",
                "based in",
                "situated in",
                "situated at",
                "emirate of",
                "masdar city",
                "abu dhabi",
                "address",
            )
        )
    if answer_type == "role_holder":
        return any(alias in lower_value for aliases in _ROLE_QUERY_ALIASES.values() for alias in aliases)
    return False


def _lookup_text_fragments(text: str) -> List[str]:
    clean_text = _clean_text(text)
    if not clean_text:
        return []
    fragments = [
        _clean_text(fragment)
        for fragment in _LOOKUP_FRAGMENT_SPLIT_RE.split(clean_text)
        if _clean_text(fragment)
    ]
    return fragments or [clean_text]


def _lookup_answer_windows(text: str, profile: LookupQueryProfile) -> List[str]:
    clean_text = _clean_text(text)
    if not clean_text or not profile.is_exact_lookup:
        return []
    regexes = []
    for answer_type in profile.answer_types:
        if answer_type == "email":
            regexes.append(_EMAIL_RE)
        elif answer_type == "phone":
            regexes.append(_PHONE_RE)
        elif answer_type == "website":
            regexes.extend([_URL_RE, _DOMAIN_RE])
        elif answer_type == "hours":
            regexes.append(_TIME_RE)
        elif answer_type == "date":
            regexes.append(_DATE_HINT_RE)
    windows: List[str] = []
    for regex in regexes:
        for match in regex.finditer(clean_text):
            start = max(0, match.start() - 56)
            end = min(len(clean_text), match.end() + 56)
            window = _clean_text(clean_text[start:end])
            if window:
                windows.append(window)
    return list(dict.fromkeys(windows))


def _expanded_lookup_qualifiers(tokens: Iterable[str]) -> set[str]:
    expanded = {str(token) for token in tokens if str(token)}
    for token in list(expanded):
        expanded.update(_LOOKUP_QUALIFIER_EQUIVALENTS.get(token, set()))
    return expanded


def _best_email_lookup_alignment(text: str, profile: LookupQueryProfile) -> Tuple[float, bool, bool]:
    if "email" not in set(profile.answer_types):
        return 0.0, False, False
    focus_tokens = set(profile.focus_tokens)
    requested_qualifiers = _expanded_lookup_qualifiers(focus_tokens & _LOOKUP_CONTEXT_QUALIFIER_TOKENS)
    best_alignment = 0.0
    qualifier_mismatch = False
    matched_requested_qualifier = False
    for match in _EMAIL_RE.finditer(_clean_text(text)):
        local_part = match.group(0).split("@", 1)[0].lower()
        local_tokens = {
            token.strip()
            for token in re.split(r"[._+\-]+", local_part)
            if token.strip()
        }
        if focus_tokens:
            focus_match = bool(local_tokens & focus_tokens)
            best_alignment = max(best_alignment, 1.0 if focus_match else 0.0)
        local_qualifiers = _expanded_lookup_qualifiers(local_tokens & _LOOKUP_CONTEXT_QUALIFIER_TOKENS)
        if local_qualifiers:
            if requested_qualifiers:
                if local_qualifiers & requested_qualifiers:
                    matched_requested_qualifier = True
                else:
                    qualifier_mismatch = True
            else:
                qualifier_mismatch = True
    qualifier_missing = bool(requested_qualifiers) and not matched_requested_qualifier
    return best_alignment, qualifier_mismatch, qualifier_missing


def _lookup_answer_label_bonus(text: str, profile: LookupQueryProfile) -> float:
    clean_text = _clean_text(text)
    if not clean_text or not profile.focus_tokens:
        return 0.0
    regexes = []
    for answer_type in profile.answer_types:
        if answer_type == "email":
            regexes.append(_EMAIL_RE)
        elif answer_type == "phone":
            regexes.append(_PHONE_RE)
        elif answer_type == "website":
            regexes.extend([_URL_RE, _DOMAIN_RE])
        elif answer_type == "hours":
            regexes.append(_TIME_RE)
        elif answer_type == "date":
            regexes.append(_DATE_HINT_RE)
    focus_tokens = set(profile.focus_tokens)
    best = 0.0
    for regex in regexes:
        for match in regex.finditer(clean_text):
            prefix = clean_text[max(0, match.start() - 64) : match.start()]
            prefix_tokens = [token for token in _tokenize(prefix) if token]
            if not prefix_tokens:
                continue
            tail_tokens = prefix_tokens[-5:]
            overlap = len(set(tail_tokens) & focus_tokens) / float(len(focus_tokens))
            if overlap <= 0.0:
                continue
            score = 0.20 + (overlap * 0.42)
            if any(token in focus_tokens for token in tail_tokens[-2:]):
                score += 0.14
            if ":" in prefix[-12:] or "-" in prefix[-8:]:
                score += 0.06
            best = max(best, score)
    return best


def _best_lookup_fragment_bonus(text: str, profile: LookupQueryProfile) -> float:
    if not profile.is_exact_lookup:
        return 0.0
    focus_tokens = set(profile.focus_tokens)
    best = 0.0
    candidates = [*_lookup_answer_windows(text, profile), *_lookup_text_fragments(text)[:6]]
    if not candidates:
        return 0.0
    requested_qualifiers = _expanded_lookup_qualifiers(focus_tokens & _LOOKUP_CONTEXT_QUALIFIER_TOKENS)
    for fragment in dict.fromkeys(candidates):
        matched_types = [
            answer_type
            for answer_type in profile.answer_types
            if _text_matches_answer_type(fragment, answer_type)
        ]
        if not matched_types:
            continue
        fragment_tokens = set(_tokenize(fragment))
        overlap = 0.0
        if focus_tokens:
            overlap = len(fragment_tokens & focus_tokens) / float(len(focus_tokens))
        score = 0.12 + (overlap * 0.34)
        if len(matched_types) > 1:
            score += 0.08
        lower_fragment = fragment.lower()
        label_bonus = _lookup_answer_label_bonus(fragment, profile)
        score += label_bonus
        if any(token in lower_fragment for token in ("contact", "contacts", "directory", "reach")):
            score += 0.10
        if any(phrase in lower_fragment for phrase in ("list of contacts", "contact directory", "email directory")):
            score += 0.18
        if ":" in fragment or " - " in lower_fragment:
            score += 0.04
        word_count = len(fragment.split())
        if word_count <= 10:
            score += 0.16
        elif word_count <= 20:
            score += 0.10
        elif word_count >= 36:
            score -= 0.10
        if "email" in matched_types:
            best_email_alignment, email_qualifier_mismatch, email_qualifier_missing = _best_email_lookup_alignment(fragment, profile)
            if focus_tokens and best_email_alignment <= 0.0:
                score -= 0.92
            score += best_email_alignment * 0.50
            if email_qualifier_mismatch:
                score -= 0.82
            if email_qualifier_missing:
                score -= 0.78
            elif requested_qualifiers:
                score += 0.24
            if "@" in fragment and label_bonus <= 0.0 and any(
                phrase in lower_fragment
                for phrase in (
                    "submitted application",
                    "updated documents",
                    "application process",
                    "screening exam",
                    "scores will be automatically forwarded",
                )
            ):
                score -= 0.18
        fragment_qualifiers = set(_tokenize(fragment)) & _LOOKUP_CONTEXT_QUALIFIER_TOKENS
        if fragment_qualifiers:
            if requested_qualifiers:
                if not (fragment_qualifiers & requested_qualifiers):
                    score -= 0.12
            else:
                score -= min(0.24, 0.12 * len(fragment_qualifiers))
        if profile.strict_answer_required and focus_tokens and overlap <= 0.0:
            score -= 0.10
        best = max(best, score)
    return best


def _lookup_signal_bonus(query: str, text: str) -> float:
    profile = _lookup_query_profile(query)
    if not profile.is_exact_lookup:
        return 0.0
    clean_text = _clean_text(text)
    if not clean_text:
        return 0.0
    text_tokens = set(_tokenize(clean_text))
    lower_text = clean_text.lower()
    requested_qualifiers = _expanded_lookup_qualifiers(
        set(profile.focus_tokens) & _LOOKUP_CONTEXT_QUALIFIER_TOKENS
    )
    bonus = 0.0
    matched_types = [answer_type for answer_type in profile.answer_types if _text_matches_answer_type(clean_text, answer_type)]
    if matched_types:
        bonus += 0.42 + (0.12 * max(len(matched_types) - 1, 0))
        bonus += _best_lookup_fragment_bonus(clean_text, profile)
    elif profile.strict_answer_required:
        bonus -= 0.48

    if profile.focus_tokens:
        overlap = len(text_tokens & set(profile.focus_tokens)) / float(len(profile.focus_tokens))
        bonus += overlap * (0.55 if profile.is_contact_lookup else 0.32)
        if profile.is_contact_lookup and overlap == 0.0:
            bonus -= 0.30

    if profile.is_contact_lookup:
        email_alignment, email_qualifier_mismatch, email_qualifier_missing = _best_email_lookup_alignment(clean_text, profile)
        if "email" in set(profile.answer_types):
            if profile.focus_tokens and email_alignment <= 0.0 and _EMAIL_RE.search(clean_text):
                bonus -= 0.95
            else:
                bonus += email_alignment * 0.36
            if email_qualifier_mismatch:
                bonus -= 0.80
            if email_qualifier_missing:
                bonus -= 0.70
            elif requested_qualifiers:
                bonus += 0.16
        if "email address" in lower_text and not _EMAIL_RE.search(clean_text):
            bonus -= 0.22
        if "committee" in lower_text and not any(_text_matches_answer_type(clean_text, answer_type) for answer_type in profile.answer_types):
            bonus -= 0.28
        if "contact" in lower_text and any(_text_matches_answer_type(clean_text, answer_type) for answer_type in profile.answer_types):
            bonus += 0.12
    return bonus


@lru_cache(maxsize=2048)
def _structured_answer_types(query: str) -> Tuple[str, ...]:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    lookup_profile = _lookup_query_profile(query)
    answer_types: List[str] = list(lookup_profile.answer_types)
    if _requested_role_subtypes(query):
        answer_types.append("role_holder")

    if any(phrase in normalized for phrase in ("whose name", "carry the name", "named after")) or (
        {"name", "carry"} <= query_tokens
    ):
        answer_types.append("named_after")

    if any(
        phrase in normalized
        for phrase in (
            "institutional affiliation",
            "authority it is affiliated",
            "affiliated with",
            "affiliated to",
        )
    ) or bool({"affiliation", "affiliated", "authority"} & query_tokens):
        answer_types.append("affiliation")

    if any(
        phrase in normalized
        for phrase in (
            "legal basis",
            "law that created",
            "under which law",
            "established under law",
            "established by law",
        )
    ) or bool({"law", "legal", "established", "created"} & query_tokens):
        answer_types.append("legal_basis")

    non_location_fact_tokens = {
        "park",
        "parked",
        "parking",
        "accommodation",
        "housing",
        "shuttle",
        "transport",
        "bus",
        "visitor",
        "visitors",
        "guest",
        "guests",
        "family",
        "families",
        "parent",
        "parents",
        "email",
        "phone",
        "website",
        "hours",
        "hour",
        "date",
        "deadline",
    }
    if not (query_tokens & non_location_fact_tokens):
        if any(
            phrase in normalized
            for phrase in ("where is", "where are", "where was", "in which city", "what city", "which emirate")
        ) or bool({"location", "located", "based", "city", "emirate"} & query_tokens):
            answer_types.append("location")

    if query_tokens & {
        "park",
        "parked",
        "parking",
        "shuttle",
        "transport",
        "transportation",
        "bus",
        "accommodation",
        "housing",
        "stay",
        "staying",
        "parent",
        "parents",
        "family",
        "families",
        "amenity",
        "amenities",
        "service",
        "services",
        "facility",
        "facilities",
    }:
        answer_types.append("service_availability")

    return tuple(dict.fromkeys(str(value) for value in answer_types if str(value)))


def _synthesis_structured_lane_enabled(
    query: str,
    *,
    mode: QueryMode,
    lookup_profile: LookupQueryProfile | None = None,
    structured_answer_types: Sequence[str] | None = None,
) -> bool:
    if mode != QueryMode.SYNTHESIS:
        return False
    profile = lookup_profile or _lookup_query_profile(query)
    answer_types = set(structured_answer_types or _structured_answer_types(query))
    return bool(
        profile.is_contact_lookup
        or bool({"affiliation", "legal_basis", "named_after", "role_holder"} & answer_types)
    )


def _token_overlap(query: str, text: str) -> float:
    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.0
    text_tokens = _token_set(text)
    if not text_tokens:
        return 0.0
    return len(query_tokens & text_tokens) / float(len(query_tokens))


def _fact_lookup_text(fact: Mapping[str, Any]) -> str:
    return _clean_text(
        " ".join(
            str(fact.get(key) or "")
            for key in ("text", "dense_text", "document_title", "heading", "section_heading", "source_url")
        )
    )


def _contextual_contact_fact_match(query: str, fact_text: str, fact_lookup_text: str) -> bool:
    if not _EMAIL_RE.search(fact_text):
        return False
    query_tokens = set(_tokenize(query))
    lookup_lower = _clean_text(fact_lookup_text).lower()
    screening_query = bool({"screening", "exam", "online"} & query_tokens) or "screening exam" in query.lower()
    support_query = bool({"it", "support", "technical", "help"} & query_tokens) or "it support" in query.lower()
    if screening_query and support_query:
        return any(
            marker in lookup_lower
            for marker in (
                "it_external@mbzuai.ac.ae",
                "it team",
                "technical support",
                "online screening exam",
                "screening exam instructions",
            )
        )
    return False


def _contextual_family_accommodation_match(query: str, record_text: str) -> bool:
    query_lower = _clean_text(query).lower()
    text_lower = _clean_text(record_text).lower()
    if not query_lower or not text_lower:
        return False
    family_query = bool(re.search(r"\b(family|families|parents?|visitors?|visiting|guests?)\b", query_lower))
    accommodation_query = bool(
        re.search(r"\b(accommodation|housing|stay|staying|lodging|visit|campus|student accommodation)\b", query_lower)
    )
    if not (family_query and accommodation_query):
        return False
    return any(
        marker in text_lower
        for marker in (
            "does not provide housing for parents",
            "can my parents stay",
            "parents stay with me on campus",
            "housing for parents",
            "visiting parents",
            "nearby hotels",
            "airbnbs",
            "airbnb",
        )
    )


def _phrase_match_bonus(query_terms: Sequence[str], text: str) -> float:
    normalized_text = _clean_text(text).lower()
    if not normalized_text or not query_terms:
        return 0.0
    bonus = 0.0
    if any(term in normalized_text for term in query_terms):
        bonus += 0.15
    for idx in range(len(query_terms) - 1):
        if f"{query_terms[idx]} {query_terms[idx + 1]}" in normalized_text:
            bonus += 0.35
            break
    return bonus


def _query_starts_with(query: str, prefixes: Sequence[str]) -> bool:
    normalized = _clean_text(query).lower()
    return any(normalized.startswith(prefix) for prefix in prefixes)


def _is_generic_figure_label(text: str) -> bool:
    normalized = _clean_text(text).lower()
    if not normalized.startswith("figure"):
        return False
    suffix = normalized[len("figure") :].strip(" .:#-")
    return not suffix or suffix.isdigit()


def _named_query_tokens(query: str) -> List[str]:
    tokens: List[str] = []
    for index, raw_token in enumerate(str(query or "").split()):
        token = "".join(ch for ch in raw_token if ch.isalnum() or ch in {"-", "_", "'"}).strip()
        if not token:
            continue
        if token.lower() in {"who", "what", "when", "where", "which", "why", "how", "is", "are", "does", "do", "can", "has", "have"}:
            continue
        if (
            any(ch.isupper() for ch in token[1:])
            or (token.isupper() and len(token) >= 3)
            or (index > 0 and token[:1].isupper() and any(ch.islower() for ch in token[1:]))
        ):
            tokens.extend(_token_variants(token))
    return list(dict.fromkeys(tokens))


def _named_query_phrases(query: str) -> List[str]:
    phrases: List[str] = []
    current: List[str] = []
    for index, raw_token in enumerate(str(query or "").split()):
        token = "".join(ch for ch in raw_token if ch.isalnum() or ch in {"-", "_", "'"}).strip()
        if not token:
            if len(current) >= 2:
                phrases.append(" ".join(current))
            current = []
            continue
        lower = token.lower()
        is_named = (
            lower not in {"who", "what", "when", "where", "which", "why", "how", "is", "are", "does", "do", "can", "has", "have"}
            and (
                any(ch.isupper() for ch in token[1:])
                or (token.isupper() and len(token) >= 3)
                or (index > 0 and token[:1].isupper() and any(ch.islower() for ch in token[1:]))
            )
        )
        if is_named:
            current.append(_clean_text(token))
            continue
        if len(current) >= 2:
            phrases.append(" ".join(current))
        current = []
    if len(current) >= 2:
        phrases.append(" ".join(current))
    return list(dict.fromkeys(phrase.strip().lower() for phrase in phrases if phrase.strip()))


_PERSON_TITLE_TOKENS = {"professor", "prof", "dr", "doctor"}
_PERSON_NAME_EXCLUDE_TOKENS = {
    *_PERSON_TITLE_TOKENS,
    "admission",
    "admissions",
    "campus",
    "catalogue",
    "center",
    "centre",
    "contact",
    "directory",
    "graduate",
    "governance",
    "institute",
    "laboratory",
    "library",
    "program",
    "programme",
    "research",
    "school",
    "team",
    "visitor",
    "mbzuai",
    "mohamed",
    "bin",
    "zayed",
    "university",
    "artificial",
    "intelligence",
}


def _person_name_tokens(query: str) -> List[str]:
    raw_tokens = [
        "".join(ch for ch in raw_token if ch.isalnum() or ch in {"-", "_", "'"}).strip()
        for raw_token in str(query or "").split()
    ]
    raw_tokens = [token for token in raw_tokens if token]
    for index, token in enumerate(raw_tokens):
        if token.lower().strip("'") not in _PERSON_TITLE_TOKENS:
            continue
        collected: List[str] = []
        for candidate in raw_tokens[index + 1 : index + 5]:
            lower = candidate.lower().strip("'")
            if lower in _PERSON_NAME_EXCLUDE_TOKENS:
                continue
            is_name_like = (
                any(ch.isupper() for ch in candidate[:1])
                or any(ch.isupper() for ch in candidate[1:])
                or candidate.isupper()
            )
            if not is_name_like:
                break
            collected.extend(
                variant
                for variant in _token_variants(lower)
                if variant and variant not in _PERSON_NAME_EXCLUDE_TOKENS
            )
        if len(dict.fromkeys(collected)) >= 2:
            return list(dict.fromkeys(collected))

    for phrase in _named_query_phrases(query):
        phrase_tokens = [
            token
            for token in _tokenize(phrase)
            if token not in _PERSON_NAME_EXCLUDE_TOKENS and len(token) >= 3
        ]
        if len(phrase_tokens) >= 2:
            return list(dict.fromkeys(phrase_tokens[:4]))
    return []


def _requested_year_ranges(query: str) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    for match in re.finditer(r"\b(20\d{2})\s*[-/]\s*(20\d{2})\b", _clean_text(query)):
        start_year = int(match.group(1))
        end_year = int(match.group(2))
        if start_year <= end_year:
            ranges.append((start_year, end_year))
    return list(dict.fromkeys(ranges))


def _requested_years(query: str) -> List[int]:
    years = [int(value) for value in re.findall(r"\b(20\d{2})\b", _clean_text(query))]
    return list(dict.fromkeys(years))


def _requires_exact_temporal_evidence(query: str) -> bool:
    query_tokens = set(_tokenize(query))
    if not _requested_years(query):
        return False
    temporal_scope_tokens = {
        "academic",
        "admission",
        "admissions",
        "application",
        "applications",
        "class",
        "classes",
        "deadline",
        "fall",
        "intake",
        "requirement",
        "requirements",
        "semester",
        "spring",
        "start",
        "starts",
        "year",
    }
    program_scope_tokens = {
        "bachelor",
        "graduate",
        "master",
        "msc",
        "m.sc",
        "phd",
        "program",
        "programs",
        "undergraduate",
    }
    return bool(query_tokens & temporal_scope_tokens) and bool(query_tokens & program_scope_tokens)


def _text_has_requested_temporal_anchor(query: str, text: str) -> bool:
    if not _requires_exact_temporal_evidence(query):
        return True
    normalized_text = _clean_text(text).lower()
    if not normalized_text:
        return False
    for start_year, end_year in _requested_year_ranges(query):
        compact_patterns = {
            f"{start_year}-{end_year}",
            f"{start_year}/{end_year}",
            f"{start_year} {end_year}",
        }
        if any(pattern in normalized_text for pattern in compact_patterns):
            continue
        if str(start_year) in normalized_text and str(end_year) in normalized_text:
            continue
        return False
    ranged_years = {year for year_range in _requested_year_ranges(query) for year in year_range}
    for year in _requested_years(query):
        if year in ranged_years:
            continue
        if str(year) not in normalized_text:
            return False
    return True


def _support_query_intents(query: str) -> set[str]:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    intents: set[str] = set()
    if _person_name_tokens(query) or query_tokens & _PERSON_TITLE_TOKENS:
        intents.add("faculty_person")
    statement_tokens = {
        "age",
        "background",
        "believe",
        "believes",
        "conversation",
        "future",
        "generation",
        "generations",
        "interview",
        "joining",
        "prepared",
        "said",
        "says",
        "skill",
        "skills",
    }
    if (
        (_person_name_tokens(query) or query_tokens & ({"president"} | _PERSON_TITLE_TOKENS))
        and query_tokens & statement_tokens
    ):
        intents.add("person_statement")
    if "president" in query_tokens and query_tokens & {"background", "joining", "professional"}:
        intents.add("president_background")
    if (
        {"first", "cohort"} <= query_tokens
        or ("first cohort" in normalized and query_tokens & {"current", "expanded", "expansion", "majors", "programs"})
    ):
        intents.add("program_expansion_history")
    if "advisory board" in normalized and query_tokens & {"attended", "meeting", "official", "first"}:
        intents.add("news_event")
    if (
        query_tokens & {"student", "students"}
        and query_tokens & {"live", "living", "life"}
        and ("abu dhabi" in normalized or query_tokens & {"cultural", "culture", "entertainment", "city"})
    ):
        intents.add("student_life")
    if "ai reach" in normalized or {"ai", "reach"} <= query_tokens:
        intents.add("ai_reach")
    if query_tokens & {"scholarship", "scholarships", "financial", "aid", "tuition", "stipend", "funding", "funded"}:
        intents.add("scholarship")
    if (
        query_tokens & {"start", "starts", "begin", "begins", "class", "classes", "semester", "fall", "academic"}
        and query_tokens & {"undergraduate", "bachelor", "business", "engineering", "program", "stream"}
    ):
        intents.add("start_date")
    prep_tokens = {
        "applicant",
        "applicants",
        "application",
        "courses",
        "course",
        "exam",
        "online",
        "prepare",
        "preparing",
        "preparatory",
        "qualification",
        "qualifications",
        "recommended",
        "referee",
        "screening",
    }
    if query_tokens & prep_tokens:
        intents.add("application_prep")
        if query_tokens & {"screening", "exam", "prepare", "preparing", "recommended", "online"}:
            intents.add("screening_prep")
    degree_signal_tokens = {
        "application",
        "applicant",
        "applicants",
        "bachelor",
        "course",
        "courses",
        "curriculum",
        "deadline",
        "degree",
        "doctor",
        "graduate",
        "master",
        "msc",
        "phd",
        "program",
        "programs",
        "referee",
        "requirement",
        "requirements",
        "screening",
        "undergraduate",
    }
    program_phrase_hit = any(
        phrase in normalized
        for phrase in (
            "computer vision",
            "machine learning",
            "computational biology",
            "natural language processing",
            "computer science",
            "statistics and data science",
            "applied artificial intelligence",
            "applied ai",
            "engineering stream",
            "business stream",
            "ai reach",
            "ugrip",
        )
    )
    if (query_tokens & degree_signal_tokens) or (program_phrase_hit and "faculty_person" not in intents):
        intents.add("degree_program_page")
    return intents


def _degree_program_source_bonus(query: str, lower_url: str, source_blob: str) -> float:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    program_aliases: Tuple[Tuple[Tuple[str, ...], Tuple[str, ...]], ...] = (
        (("computer vision",), ("computer-vision",)),
        (("machine learning",), ("machine-learning",)),
        (("computational biology",), ("computational-biology",)),
        (("natural language processing", "nlp"), ("natural-language-processing",)),
        (("robotics",), ("robotics",)),
        (("computer science",), ("computer-science",)),
        (("statistics and data science", "statistics data science", "data science"), ("statistics-and-data-science",)),
        (("applied artificial intelligence", "applied ai", "maai"), ("master-in-applied-ai", "applied-ai")),
        (("artificial intelligence engineering stream", "engineering stream"), ("artificial-intelligence-engineering-stream",)),
        (("artificial intelligence business stream", "business stream"), ("artificial-intelligence-business-stream",)),
        (("undergraduate research internship", "ugrip"), ("undergraduate-research-internship-program", "ugrip")),
        (("ai reach",), ("ai-reach",)),
    )
    wanted_slugs: List[str] = []
    for aliases, slugs in program_aliases:
        if any(alias in normalized for alias in aliases):
            wanted_slugs.extend(slugs)
    degree_or_program_signal = bool(
        {"application", "applicant", "applicants", "bachelor", "course", "courses", "curriculum", "deadline", "degree", "doctor", "graduate", "master", "msc", "phd", "program", "programs", "referee", "requirement", "requirements", "screening", "undergraduate"}
        & query_tokens
    )
    if not degree_or_program_signal and not wanted_slugs:
        return 0.0

    bonus = 0.0
    if wanted_slugs:
        if any(slug in lower_url for slug in wanted_slugs):
            bonus += 2.85
        elif "/study/faculty/" in lower_url:
            bonus -= 1.85
        elif "/news/" in lower_url and not ("news" in query_tokens or "announcement" in query_tokens):
            bonus -= 0.65
    if degree_or_program_signal:
        if any(marker in lower_url for marker in ("/study/msc-programs/", "/study/phd-programs/", "/study/master-in-applied-ai", "/study/undergraduate-program")):
            bonus += 0.75
        if "/study/faculty/" in lower_url and not (_person_name_tokens(query) or query_tokens & _PERSON_TITLE_TOKENS):
            bonus -= 1.35
        if "catalogue" in source_blob or "catalog" in source_blob:
            bonus -= 0.35
    if query_tokens & {"deadline", "application", "applicant", "applicants", "referee", "screening", "recommended", "online"}:
        if wanted_slugs and any(slug in lower_url for slug in wanted_slugs):
            bonus += 0.80
        if "/study/faculty/" in lower_url:
            bonus -= 0.75
    return bonus


def _support_intent_text_bonus(
    query: str,
    *,
    source_url: str = "",
    document_title: str = "",
    heading: str = "",
    text: str = "",
) -> float:
    intents = _support_query_intents(query)
    if not intents:
        return 0.0
    lower_url = str(source_url or "").lower()
    lower_title = _clean_text(document_title).lower()
    lower_heading = _clean_text(heading).lower()
    lower_text = _clean_text(text).lower()
    query_tokens = set(_tokenize(query))
    blob = " ".join(part for part in (lower_url, lower_title, lower_heading, lower_text) if part)
    if not blob:
        return 0.0
    bonus = 0.0
    is_news = "/news/" in lower_url or "news-archive" in lower_url or "/the-node/" in lower_url
    is_program_page = any(marker in lower_url for marker in ("/study/msc-programs/", "/study/phd-programs/", "/undergraduate-program"))
    is_faculty_profile = "/study/faculty/" in lower_url
    is_application_page = "application-submission" in lower_url or "admission-process" in lower_url

    if "degree_program_page" in intents:
        bonus += _degree_program_source_bonus(query, lower_url, blob)

    if "person_statement" in intents:
        person_tokens = set(_person_name_tokens(query))
        blob_tokens = set(_source_anchor_tokens(blob))
        matched = len(person_tokens & blob_tokens)
        match_ratio = matched / float(len(person_tokens)) if person_tokens else 0.0
        statement_specific_query = bool(
            query_tokens
            & {
                "believe",
                "believes",
                "future",
                "generation",
                "generations",
                "prepared",
                "skill",
                "skills",
            }
        )
        statement_page = any(
            marker in lower_url
            for marker in (
                "in-conversation-with-the-president",
                "interview",
                "fortune-magazine",
                "course-to-the-future",
            )
        ) or any(
            marker in blob
            for marker in (
                "future generations",
                "future of education",
                "future of skills",
                "age of ai",
                "ai literacy",
                "continuously learn",
                "which skills",
                "important skill",
                "interview",
            )
        )
        if is_news and statement_page:
            bonus += 5.30
        elif is_news and match_ratio >= 0.50 and not statement_specific_query:
            bonus += 0.85
        elif is_news and match_ratio >= 0.50:
            # A named person appearing in a news story is not enough for quote
            # or belief questions. These queries need interview/statement text.
            bonus -= 1.20
        if is_faculty_profile:
            bonus -= 2.25

    if "president_background" in intents:
        if "in-conversation-with-the-president" in lower_url:
            bonus += 3.25
        elif "/about/office-of-the-president" in lower_url or "/about/leadership" in lower_url:
            bonus += 1.45
        if is_faculty_profile:
            bonus -= 1.25

    if "program_expansion_history" in intents:
        if "in-conversation-with-the-president" in lower_url:
            bonus += 2.60
        elif "first-cohort" in lower_url:
            bonus += 1.20
        if is_faculty_profile:
            bonus -= 1.10

    if "news_event" in intents:
        if is_news and "advisory-board-meeting" in lower_url:
            bonus += 4.20
        elif is_news and "advisory board" in blob:
            bonus += 1.60
        if is_faculty_profile:
            bonus -= 2.20

    if "student_life" in intents:
        path = urlparse(lower_url).path.rstrip("/")
        if path == "/study":
            bonus += 2.75
        elif any(marker in lower_url for marker in ("student-resources", "campus-facilities", "educational-affairs")):
            bonus += 1.05
        if is_faculty_profile:
            bonus -= 1.45
        if is_news:
            bonus -= 0.50

    if "faculty_person" in intents and "person_statement" not in intents:
        person_tokens = set(_person_name_tokens(query))
        blob_tokens = set(_source_anchor_tokens(blob))
        matched = len(person_tokens & blob_tokens)
        match_ratio = matched / float(len(person_tokens)) if person_tokens else 0.0
        if is_faculty_profile and match_ratio >= 0.75:
            bonus += 2.85
        elif is_faculty_profile and match_ratio >= 0.50:
            bonus += 1.35
        elif is_faculty_profile:
            bonus += 0.35
        if person_tokens and not is_faculty_profile and is_program_page:
            bonus -= 1.35
        if is_news and match_ratio < 0.75:
            bonus -= 0.65

    if "ai_reach" in intents:
        if "/study/ai-reach" in lower_url:
            bonus += 1.85
        if any(marker in blob for marker in ("participants will", "high school", "two-week", "fully funded", "on-campus research")):
            bonus += 1.25
        if is_news and "/study/ai-reach" not in lower_url:
            bonus -= 0.75

    if "scholarship" in intents:
        scholarship_markers = (
            "scholarship",
            "scholarships",
            "financial aid",
            "full scholarship",
            "tuition",
            "living costs",
            "tahnoon",
            "stipend",
        )
        if any(marker in blob for marker in scholarship_markers):
            bonus += 3.25
        if "undergraduate-application-submission" in lower_url:
            bonus += 0.80
        if "tahnoon" in lower_url or "scholarship" in lower_url:
            bonus += 2.60
        if is_program_page and not any(marker in blob for marker in scholarship_markers):
            bonus -= 2.45

    if "start_date" in intents:
        start_markers = (
            "fall semester begins",
            "semester begins",
            "classes begin",
            "start date",
            "mid-august",
            "mid august",
            "fall 2026",
            "orientation week",
            "official enrollment",
        )
        if any(marker in blob for marker in start_markers):
            bonus += 3.45
        if "undergraduate-application-submission" in lower_url or lower_url.rstrip("/").endswith("/study/undergraduate-program"):
            bonus += 3.35
        if "/bachelor-of-science-" in lower_url or "/ar/bachelor-of-science-" in lower_url:
            if not any(marker in blob for marker in start_markers):
                bonus -= 3.20
        if is_news:
            bonus -= 0.55

    if "application_prep" in intents or "screening_prep" in intents:
        prep_markers = (
            "recommended online courses",
            "online screening exam",
            "screening exam",
            "programming for everybody",
            "python data structures",
            "mathematics for machine learning",
            "intuitive introduction to probability",
            "referee",
            "application submission",
            "improve qualification",
            "improve qualifications",
        )
        if any(marker in blob for marker in prep_markers):
            bonus += 2.25
        elif any(marker in blob for marker in ("application", "applicant", "admission")):
            bonus += 0.45
        if query_tokens & {"age", "limit", "fee"} and is_application_page:
            bonus += 2.10
        if query_tokens & {"age", "limit", "fee"} and is_program_page and not is_application_page:
            bonus -= 0.85
        curriculum_only = any(marker in blob for marker in ("course description", "mandatory courses", "elective courses", "study plan"))
        if curriculum_only and not any(marker in blob for marker in prep_markers):
            bonus -= 0.70

    return bonus


def _fallback_subject_tokens_and_phrases(
    query: str,
    *,
    answer_types: Sequence[str],
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    if not answer_types:
        return tuple(), tuple()
    excluded_tokens = {
        *_QUERY_STOPWORDS,
        *_LOOKUP_ATTRIBUTE_TOKENS,
        *_FACT_ATTRIBUTE_TOKENS,
        *_ROLE_QUERY_TOKENS,
        *_LOOKUP_ENTITY_ROLE_TOKENS,
        *_GENERIC_CONTACT_QUERY_TOKENS,
        "address",
        "answer",
        "contacting",
        "details",
        "information",
        "i",
        "me",
        "my",
        "number",
        "official",
        "we",
        "you",
    }
    ordered_tokens: List[str] = []
    for raw_token in str(query or "").split():
        clean = _clean_text(raw_token).lower()
        if not clean:
            continue
        variants = [variant for variant in _token_variants(clean) if variant and variant not in excluded_tokens]
        if variants:
            ordered_tokens.append(variants[0])
    if not ordered_tokens:
        return tuple(), tuple()
    tokens = tuple(dict.fromkeys(ordered_tokens))
    phrases: List[str] = []
    if len(tokens) >= 2:
        phrases.append(" ".join(tokens[:2]))
    if len(tokens) >= 3:
        phrases.append(" ".join(tokens[:3]))
    return tokens, tuple(dict.fromkeys(phrase for phrase in phrases if phrase))


def _missing_named_token_ratio(query: str, text: str) -> float:
    named_tokens = set(_named_query_tokens(query))
    if not named_tokens:
        return 0.0
    text_tokens = _token_set(text)
    if not text_tokens:
        return 1.0
    missing = sum(1 for token in named_tokens if token not in text_tokens)
    return missing / float(len(named_tokens))


def _effective_named_tokens_for_abstention(query: str, evidence_text: str) -> List[str]:
    named_tokens = _named_query_tokens(query)
    if "mbzuai.ac.ae" in str(evidence_text or "").lower():
        named_tokens = [token for token in named_tokens if token not in _INSTITUTION_CONTEXT_TOKENS]
    return named_tokens


def _missing_token_ratio(tokens: Sequence[str], text: str) -> float:
    token_set = {str(token) for token in tokens if str(token)}
    if not token_set:
        return 0.0
    text_tokens = _token_set(text)
    if not text_tokens:
        return 1.0
    missing = sum(1 for token in token_set if token not in text_tokens)
    return missing / float(len(token_set))


def _is_generic_subject_reference(subject_text: str) -> bool:
    subject_tokens = [token for token in _tokenize(subject_text) if token]
    if not subject_tokens:
        return True
    return all(token in _GENERIC_SUBJECT_REFERENCE_TOKENS for token in subject_tokens)


def _subject_alignment_score(*, query: str, subject_text: str, context_text: str) -> float:
    normalized_subject = _clean_text(subject_text)
    if not normalized_subject:
        return 0.0
    subject_tokens = set(_tokenize(normalized_subject))
    query_subject_tokens = set(_query_intent(query).subject_tokens)
    if not query_subject_tokens:
        query_subject_tokens = set(_named_query_tokens(query))
    if not query_subject_tokens:
        return 0.0
    context_tokens = set(_tokenize(context_text))
    subject_overlap = len(subject_tokens & query_subject_tokens) / float(len(query_subject_tokens)) if query_subject_tokens else 0.0
    context_overlap = len(context_tokens & query_subject_tokens) / float(len(query_subject_tokens)) if query_subject_tokens else 0.0
    return max(subject_overlap, context_overlap)


def _query_subject_label(query: str) -> str:
    normalized = _clean_text(query)
    lower_query = normalized.lower()
    if "mbzuai" in lower_query or "mohamed bin zayed university of artificial intelligence" in lower_query:
        return "MBZUAI"
    phrases = _named_query_phrases(query)
    if phrases:
        return phrases[0]
    return "the institution"


def _slot_query_text(query: str, slot: QueryAnswerSlot) -> str:
    subject_label = _query_subject_label(query)
    if slot.answer_type == "role_holder" and slot.qualifier:
        role_label = _ROLE_QUERY_LABELS.get(slot.qualifier, slot.qualifier.replace("_", " "))
        if slot.qualifier == "board_chair":
            return f"Who is the {role_label} of {subject_label}?"
        if slot.qualifier == "vice_president_chief_of_staff":
            return f"Who is {subject_label}'s {role_label}?"
        return f"Who is the {role_label} of {subject_label}?"
    return query


def _requested_service_availability_subtypes(query: str) -> set[str]:
    query_tokens = set(_tokenize(query))
    requested: set[str] = set()
    if query_tokens & {"shuttle", "transport", "transportation", "bus"}:
        requested.add("transport")
    if query_tokens & {"park", "parked", "parking", "vehicle", "vehicles", "visitor", "visitors"}:
        requested.add("parking")
    if query_tokens & {"accommodation", "housing", "stay", "staying", "parent", "parents", "family", "families"}:
        requested.add("accommodation")
    explicit_amenity_tokens = query_tokens & {"amenity", "amenities", "support"}
    generic_amenity_tokens = query_tokens & {"service", "services", "facility", "facilities"}
    if explicit_amenity_tokens or (generic_amenity_tokens and not requested):
        requested.add("amenities")
    return requested


def _truncate_chars(text: str, *, max_chars: int) -> str:
    value = _clean_text(text)
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    clipped = value[:max_chars].rstrip()
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0] or clipped
    return clipped.strip()


def _truncate_tokens(text: str, *, max_tokens: int, max_chars: int | None = None) -> str:
    tokens = _clean_text(text).split()
    if len(tokens) <= max_tokens:
        value = " ".join(tokens)
    else:
        value = " ".join(tokens[:max_tokens])
    if max_chars is not None:
        value = _truncate_chars(value, max_chars=max_chars)
    return value


def _truncate_fragments(
    fragments: Sequence[str],
    *,
    max_tokens: int,
    max_chars: int | None = None,
) -> str:
    remaining = max(0, int(max_tokens))
    remaining_chars = None if max_chars is None else max(0, int(max_chars))
    output: List[str] = []
    for fragment in fragments:
        candidate = _clean_text(fragment)
        if not candidate or remaining <= 0 or remaining_chars == 0:
            continue
        words = candidate.split()
        if not words:
            continue
        if len(words) <= remaining:
            selected = " ".join(words)
        else:
            selected = " ".join(words[:remaining])
        if remaining_chars is not None:
            selected = _truncate_chars(selected, max_chars=remaining_chars)
        if not selected:
            break
        output.append(selected)
        remaining -= len(selected.split())
        if remaining_chars is not None:
            remaining_chars -= len(selected)
    return "\n".join(output).strip()


def _rrf_merge(rankings: Sequence[Sequence[str]], *, k: int = 60) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, record_id in enumerate(ranking, start=1):
            if not record_id:
                continue
            scores[record_id] = scores.get(record_id, 0.0) + (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _is_media_query(query: str) -> bool:
    # Do not treat a visual-domain word inside a quoted work/event title as a
    # request to inspect an image. For example, “the talk titled \"Applying
    # Image Analysis …\"” is an exact event lookup, not a media query.
    intent_text = re.sub(
        r"\b(?:titled|called|named)\s+[\"'“‘][^\"'”’]{0,240}[\"'”’]",
        " ",
        str(query or ""),
        flags=re.IGNORECASE,
    )
    query_tokens = set(_tokenize(intent_text))
    if not query_tokens:
        return False
    normalized_intent = _clean_text(intent_text).casefold()
    if (
        "policy page" in normalized_intent
        and re.search(r"\b(?:promotion|professor|faculty review)\b", normalized_intent)
    ):
        return True
    if query_tokens & _EXPLICIT_VISUAL_QUERY_TOKENS:
        return True
    # Do not route ordinary factual campus/facilities/parking questions into
    # the visual lane unless they also carry an explicit visual cue.
    if {"building", "buildings", "facility", "facilities", "parking"} & query_tokens:
        return bool({"map", "layout", "diagram", "figure", "image", "images", "show", "shown", "labelled", "labeled", "visual"} & query_tokens)
    return False


def _support_hours_query(query: str) -> bool:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    if query_tokens & {"support", "helpdesk", "technical", "screening", "exam", "internship", "host", "organization", "it"}:
        return True
    return any(
        phrase in normalized
        for phrase in (
            "technical support",
            "online screening exam",
            "screening exam",
            "it support",
            "help desk",
            "helpdesk",
            "host organization",
        )
    )


def _hours_query_alias_tokens(query: str) -> List[str]:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    aliases: List[str] = ["working", "time", "times"]
    if _support_hours_query(query):
        aliases.extend(["support", "technical", "it"])
        if {"screening", "exam"} & query_tokens or "screening exam" in normalized:
            aliases.extend(["screening", "exam"])
        if "helpdesk" in query_tokens or "help" in query_tokens:
            aliases.append("helpdesk")
        return aliases
    if {"official", "operating", "weekday", "weekdays"} & query_tokens or any(
        phrase in normalized for phrase in ("official working hours", "official workings hours", "operating hours", "office hours")
    ):
        aliases.append("official")
    if {"weekday", "weekdays", "operating"} & query_tokens:
        aliases.extend(["monday", "thursday", "friday"])
    return aliases


def _semantic_query_alias_tokens(query: str) -> List[str]:
    normalized = _clean_text(query).lower()
    query_tokens = set(_tokenize(query))
    synthesis_like_topics = {
        "location",
        "parking",
        "transport",
        "shuttle",
        "facilities",
        "amenities",
        "support",
        "hours",
        "law",
        "affiliation",
    }
    if (
        any(term in normalized for term in {"prepare", "briefing", "covering", "summary", "summarize", "guide"})
        and len(query_tokens & synthesis_like_topics) >= 2
    ):
        return []
    aliases: List[str] = []
    if (
        re.search(r"(?:لوحة|اللوحة).{0,80}(?:مشاريع|المشاريع).{0,120}(?:هندي|هندية)", normalized)
        or re.search(r"(?:هندي|هندية).{0,120}(?:لوحة|اللوحة).{0,80}(?:مشاريع|المشاريع)", normalized)
    ):
        aliases.extend(["NANDA", "Hindi", "LLM", "knowledge", "reasoning", "performance"])
    if any(term in normalized for term in ("بعد المطر", "بعد هطول المطر", "ما بعد المطر")):
        aliases.extend(["after", "rain", "rainfall", "post", "water", "accumulation", "flood"])
    if "صورة" in normalized or "الصورة" in normalized:
        aliases.extend(["image", "figure", "visual"])
    if "خريطة" in normalized or "الخريطة" in normalized:
        aliases.append("map")
    if "جدول" in normalized or "الجدول" in normalized:
        aliases.append("table")
    if any(term in normalized for term in ("مخطط", "المخطط", "إطار", "الإطار")):
        aliases.extend(["diagram", "framework"])
    if "إنفوغراف" in normalized or "الإنفوغراف" in normalized:
        aliases.append("infographic")
    if any(
        marker in normalized
        for marker in (
            "المؤهل الأكاديمي",
            "المؤهلات الأكاديمية",
            "المؤهلات",
        )
    ):
        aliases.extend(
            [
                "academic",
                "qualification",
                "qualifications",
                "degree",
                "bachelor",
                "master",
                "required",
                "preferred",
                "mandatory",
            ]
        )
    if (
        any(marker in normalized for marker in ("أقسام الوظائف", "الوظائف المفتوحة"))
        or re.search(r"\b(?:job|vacancy|vacancies)\s+(?:sections|categories)\b", normalized)
    ):
        aliases.extend(
            [
                "faculty",
                "research",
                "engineering",
                "professional",
                "vacancy",
                "vacancies",
            ]
        )
    if "ifm" in normalized and any(
        marker in normalized
        for marker in ("شركاء", "الشركاء", "partners", "collaborat")
    ):
        aliases.extend(
            [
                "science",
                "scale",
                "social",
                "value",
                "academic",
                "institutions",
                "labs",
                "startups",
                "enterprise",
                "partners",
            ]
        )
    if any(marker in normalized for marker in ("دانييلا روس", "daniela rus")):
        aliases.extend(["الاستقلالية", "الذكاء", "autonomy", "intelligence"])
    if any(
        marker in normalized
        for marker in ("دور الرئيس", "مهام الرئيس", "صلاحيات الرئيس")
    ):
        aliases.extend(
            ["التنفيذي", "مهام", "الصلاحيات", "إدارة", "chief", "executive"]
        )
    if (
        any(marker in normalized for marker in ("معرض التدريب المهني", "career fair"))
        and any(marker in normalized for marker in ("الدعم", "دعم", "support"))
    ):
        aliases.extend(
            [
                "جلسات",
                "تدريب",
                "مهني",
                "فردية",
                "وكالات",
                "التوظيف",
                "صور",
                "احترافية",
                "coaching",
                "recruitment",
            ]
        )
    if (
        any(marker in normalized for marker in ("الدكتوراه", "doctorate", "doctoral", "phd"))
        and any(marker in normalized for marker in ("التوجه المهني", "career orientation", "career path"))
    ):
        aliases.extend(
            [
                "contribute",
                "science",
                "humanity",
                "experienced",
                "researchers",
                "academia",
                "industry",
                "startup",
            ]
        )
    if (
        any(marker in normalized for marker in ("visitor program", "برنامج الزوار"))
        and any(marker in normalized for marker in ("hands-on", "عملي", "تجربة"))
    ):
        aliases.extend(
            [
                "research",
                "experience",
                "program",
                "personalized",
                "demos",
                "talks",
            ]
        )
    if (
        ("engage" in normalized and "capture" in normalized and "value" in normalized)
        or ("يتفاعل" in normalized and "القيمة" in normalized)
    ):
        aliases.extend(
            [
                "exploration",
                "refinement",
                "high",
                "level",
                "proposal",
                "engagement",
                "agreement",
                "sign-off",
            ]
        )
    if "whose name" in normalized or ({"name", "carry"} <= query_tokens):
        aliases.extend(["named", "after"])
    if "in which city" in normalized or "what city" in normalized:
        aliases.extend(["based", "located", "location", "city"])
    if "emirate" in query_tokens:
        aliases.extend(["abu", "dhabi", "based", "located", "location", "city"])
    if "parking" in query_tokens:
        aliases.extend(["parking", "provided", "available", "permitted"])
        if {"visitor", "visitors"} & query_tokens:
            aliases.extend(["guest", "guests"])
    if {"hour", "hours"} & query_tokens:
        aliases.extend(_hours_query_alias_tokens(query))
    if {"transportation", "transport"} & query_tokens:
        aliases.extend(["shuttle", "bus", "service"])
    if {"housing", "accommodation"} & query_tokens:
        aliases.extend(["housing", "accommodation", "apartments", "campus", "stay"])
    if {"family", "families"} & query_tokens or {"parents", "parent"} & query_tokens:
        aliases.extend(["parents", "family", "visiting", "hotels", "airbnbs"])
    if {"specialization", "specializations"} & query_tokens:
        aliases.extend(["program", "programs", "graduate"])
    if {"legal", "basis"} <= query_tokens:
        aliases.extend(["law", "established", "under"])
    if "law" in query_tokens or "established" in query_tokens:
        aliases.extend(["law", "established", "under", "affiliated"])
    if {"affiliation", "affiliated"} & query_tokens or {"institutional", "affiliation"} <= query_tokens:
        aliases.extend(["affiliated", "under", "authority", "executive", "council"])
    if "authority" in query_tokens:
        aliases.extend(["affiliated", "executive", "council"])
    if {"amenity", "amenities", "facility", "facilities", "services"} & query_tokens:
        aliases.extend(["facilities", "services", "support", "campus", "accommodation", "canteen", "gym", "parking"])
    requested_roles = _requested_role_subtypes(query)
    if requested_roles:
        aliases.extend(["leadership", "office"])
        if "board_chair" in requested_roles:
            aliases.extend(["board", "trustees", "chairman"])
        if "president" in requested_roles:
            aliases.extend(["president", "university"])
        if "provost" in requested_roles:
            aliases.extend(["provost", "academic"])
        if "vice_president_chief_of_staff" in requested_roles:
            aliases.extend(["chief", "staff"])
    lookup_profile = _lookup_query_profile(query)
    if lookup_profile.is_contact_lookup:
        aliases.extend(["contact", "reach"])
        if "email" in lookup_profile.answer_types:
            aliases.extend(["email", "admission"])
        if "phone" in lookup_profile.answer_types:
            aliases.extend(["phone", "telephone", "number"])
        if "website" in lookup_profile.answer_types:
            aliases.extend(["website", "url", "link"])
    elif lookup_profile.is_exact_lookup:
        aliases.extend(list(lookup_profile.answer_types))
    return list(dict.fromkeys(token for token in aliases if token))


def classify_query_mode(query: str) -> QueryMode:
    normalized = _clean_text(query).lower()
    intent_normalized = _clean_text(
        re.sub(
            r"\b(?:titled|called|named)\s+[\"'“‘][^\"'”’]{0,240}[\"'”’]",
            " ",
            str(query or ""),
            flags=re.IGNORECASE,
        )
    ).lower()
    words = normalized.split()
    query_tokens = set(_tokenize(query))
    intent_tokens = set(_tokenize(intent_normalized))
    lookup_profile = _lookup_query_profile(query)
    broad_terms = {
        "explain",
        "summarize",
        "summary",
        "overview",
        "detailed",
        "detail",
        "compare",
        "comparison",
        "analyze",
        "analysis",
        "briefing",
        "covering",
        "history",
        "prepare",
        "describe",
        "guide",
        "policy",
        "process",
        "اشرح",
        "لخص",
        "نظرة عامة",
        "نظرة شاملة",
        "قارن",
        "حلل",
        "صف",
        "بالتفصيل",
    }
    broad_query = any(
        (
            term in intent_tokens
            if " " not in term
            else bool(
                re.search(
                    rf"(?<!\w){re.escape(term)}(?!\w)",
                    intent_normalized,
                    flags=re.IGNORECASE,
                )
            )
        )
        for term in broad_terms
    )
    synthesis_topic_terms = {
        "location",
        "parking",
        "transport",
        "shuttle",
        "facilities",
        "amenities",
        "support",
        "hours",
        "law",
        "affiliation",
        "specialization",
        "specializations",
    }
    fact_starts = (
        "who ",
        "when ",
        "where ",
        "which ",
        "whose ",
        "is ",
        "in which ",
        "are ",
        "does ",
        "do ",
        "can ",
        "has ",
        "have ",
        "how many ",
        "how much ",
        "what is ",
        "what are ",
        "what was ",
        "what does ",
        "what do ",
        "how does ",
        "من ",
        "متى ",
        "أين ",
        "اين ",
        "ما ",
        "ما هو ",
        "ما هي ",
        "كم ",
        "هل ",
        "أي ",
        "اي ",
    )
    narrow_fact_terms = {
        "address",
        "bus",
        "contact",
        "city",
        "email",
        "hour",
        "hours",
        "housing",
        "located",
        "location",
        "name",
        "named",
        "phone",
        "shuttle",
        "stay",
        "time",
        "times",
    }
    scoped_anchor = bool(query_tokens & _SCOPED_QUERY_TOKENS) or any(
        phrase in normalized for phrase in _SCOPED_QUERY_PHRASES
    )
    fact_phrase = any(phrase in normalized for phrase in _FACT_QUERY_PHRASES)
    scoped_legal_relation = (
        ("law" in query_tokens or "established" in query_tokens)
        and ("affiliated" in query_tokens or "affiliation" in query_tokens or "authority" in query_tokens)
    )
    if (
        any(term in normalized for term in {"prepare", "briefing", "covering", "guide"})
        and len(query_tokens & synthesis_topic_terms) >= 2
    ):
        return QueryMode.SYNTHESIS

    # A request for one member of a named page/list still needs list-level
    # context. Routing it as a narrow fact can suppress the Page Card and
    # parent lanes, leaving only an isolated item with no proof that it belongs
    # to the list the user named.
    list_membership_query = bool(
        re.search(
            r"\b(?:one|any)\s+(?:item\s+)?(?:of|from)\b.{0,120}"
            r"\b(?:list|listed|latest|publication|publications|page)\b",
            normalized,
        )
        or re.search(
            r"(?:إحدى|احدى|أحد|احد).{0,120}"
            r"(?:المدرجة|القائمة|المنشورات|الصفحة)",
            normalized,
        )
    )
    if list_membership_query:
        return QueryMode.SYNTHESIS

    # Questions that explicitly join several interrogative clauses require a
    # combined evidence set even when they begin with a fact-style phrase.
    # Detect clause starts, not incidental words inside a single question.
    interrogative_clauses = re.findall(
        r"(?:^|,\s*|\band\s+)(?:(?:in\s+)?which|what|where|when|who|how)\b",
        normalized,
    )
    if len(interrogative_clauses) >= 2:
        return QueryMode.SYNTHESIS

    if _is_media_query(query):
        return QueryMode.SCOPED
    if _requested_role_subtypes(query):
        return QueryMode.FACT
    if lookup_profile.is_exact_lookup and len(words) <= 32 and not broad_query:
        return QueryMode.FACT
    # Short questions such as "Which divisions are there?" look like narrow
    # facts syntactically, but a correct answer requires collecting sibling
    # sections from a page or several documents. Treat the request shape as
    # synthesis so page-card and parent lanes stay active. Exact typed lookups
    # (one email, date, phone number, or office-hours value) remain facts above.
    # This is a general retrieval decision, not a fact-specific answer shortcut.
    if _is_enumeration_query(query):
        return QueryMode.SYNTHESIS
    if fact_phrase and len(words) <= 18 and not broad_query:
        return QueryMode.FACT
    if len(words) <= 12 and normalized.startswith(fact_starts) and not broad_query and not scoped_anchor:
        return QueryMode.FACT
    if (
        len(words) <= 28
        and normalized.startswith(fact_starts)
        and not broad_query
        and not scoped_anchor
    ):
        return QueryMode.FACT
    if scoped_legal_relation:
        return QueryMode.SCOPED
    if scoped_anchor and len(words) <= 22:
        return QueryMode.SCOPED
    if len(words) >= 14 or broad_query:
        return QueryMode.SYNTHESIS
    return QueryMode.SCOPED


def _make_gemini_client(*, request_timeout_ms: int | None = None):
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required")
    timeout_ms = int(
        _DEFAULT_GEMINI_REQUEST_TIMEOUT_MS
        if request_timeout_ms is None
        else request_timeout_ms
    )
    if timeout_ms <= 0:
        raise ValueError("gemini_request_timeout_ms must be greater than zero")
    cached_key = getattr(_GEMINI_CLIENT_STATE, "api_key", None)
    cached_timeout_ms = getattr(_GEMINI_CLIENT_STATE, "request_timeout_ms", None)
    cached_client = getattr(_GEMINI_CLIENT_STATE, "client", None)
    if (
        cached_client is not None
        and cached_key == api_key
        and cached_timeout_ms == timeout_ms
    ):
        return cached_client
    genai = import_genai()
    types = import_genai_types()
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=timeout_ms),
    )
    _GEMINI_CLIENT_STATE.api_key = api_key
    _GEMINI_CLIENT_STATE.request_timeout_ms = timeout_ms
    _GEMINI_CLIENT_STATE.client = client
    return client


def _query_embedding_cache_key(
    *,
    model: str,
    output_dimensionality: int | None,
    task_type: str,
    embed_text: str,
) -> Tuple[str, int | None, str, str]:
    return (
        str(model or "").strip(),
        output_dimensionality,
        str(task_type or "").strip(),
        re.sub(r"\s+", " ", embed_text or "").strip(),
    )


def _get_cached_query_embedding(key: Tuple[str, int | None, str, str]) -> List[float] | None:
    if _QUERY_EMBEDDING_CACHE_MAX <= 0:
        return None
    with _QUERY_EMBEDDING_CACHE_LOCK:
        cached = _QUERY_EMBEDDING_CACHE.get(key)
        if cached is None:
            return None
        _QUERY_EMBEDDING_CACHE.move_to_end(key)
        return list(cached)


def _cache_query_embedding(key: Tuple[str, int | None, str, str], vector: Sequence[float]) -> None:
    if _QUERY_EMBEDDING_CACHE_MAX <= 0:
        return
    with _QUERY_EMBEDDING_CACHE_LOCK:
        _QUERY_EMBEDDING_CACHE[key] = [float(value) for value in vector]
        _QUERY_EMBEDDING_CACHE.move_to_end(key)
        while len(_QUERY_EMBEDDING_CACHE) > _QUERY_EMBEDDING_CACHE_MAX:
            _QUERY_EMBEDDING_CACHE.popitem(last=False)


def _query_embedding_failure_key(
    *,
    model: str,
    output_dimensionality: int | None,
    task_type: str,
) -> Tuple[str, int | None, str]:
    return (str(model or "").strip(), output_dimensionality, str(task_type or "").strip())


def _recent_query_embedding_failure(key: Tuple[str, int | None, str]) -> str:
    if _QUERY_EMBEDDING_FAILURE_COOLDOWN_SECONDS <= 0:
        return ""
    now = time.monotonic()
    with _QUERY_EMBEDDING_FAILURE_LOCK:
        failure = _QUERY_EMBEDDING_FAILURE_STATE.get(key)
        if not failure:
            return ""
        expires_at, message = failure
        if expires_at <= now:
            _QUERY_EMBEDDING_FAILURE_STATE.pop(key, None)
            return ""
        return message


def _record_query_embedding_failure(key: Tuple[str, int | None, str], exc: Exception) -> None:
    if _QUERY_EMBEDDING_FAILURE_COOLDOWN_SECONDS <= 0:
        return
    expires_at = time.monotonic() + _QUERY_EMBEDDING_FAILURE_COOLDOWN_SECONDS
    message = str(exc)
    with _QUERY_EMBEDDING_FAILURE_LOCK:
        _QUERY_EMBEDDING_FAILURE_STATE[key] = (expires_at, message)


def _clear_query_embedding_failure(key: Tuple[str, int | None, str]) -> None:
    with _QUERY_EMBEDDING_FAILURE_LOCK:
        _QUERY_EMBEDDING_FAILURE_STATE.pop(key, None)


def _retryable_query_embedding_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in message
        for marker in (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "rate_limit",
            "temporarily unavailable",
            "timeout",
        )
    )


def _public_query_embedding_error_code(exc: Exception) -> str:
    message = f"{type(exc).__name__}: {exc}".lower()
    if any(marker in message for marker in ("429", "resource_exhausted", "quota", "rate limit", "rate_limit")):
        return "query_embedding_rate_limited"
    if "timeout" in message:
        return "query_embedding_timeout"
    if "temporarily unavailable" in message or "unavailable" in message:
        return "query_embedding_unavailable"
    return "query_embedding_failed"


def _embed_query(
    query: str,
    *,
    model: str,
    output_dimensionality: int | None,
    task_type: str = "RETRIEVAL_QUERY",
    request_timeout_ms: int | None = None,
) -> List[float]:
    use_prompt_instruction = str(model or "").strip().lower() == "gemini-embedding-2"
    embed_text = f"task: search result | query: {query}" if use_prompt_instruction else query
    cache_key = _query_embedding_cache_key(
        model=model,
        output_dimensionality=output_dimensionality,
        task_type=task_type,
        embed_text=embed_text,
    )
    cached = _get_cached_query_embedding(cache_key)
    if cached is not None:
        return cached
    failure_key = _query_embedding_failure_key(
        model=model,
        output_dimensionality=output_dimensionality,
        task_type=task_type,
    )
    recent_failure = _recent_query_embedding_failure(failure_key)
    if recent_failure:
        raise RuntimeError(f"query embedding temporarily disabled after recent provider failure: {recent_failure}")

    try:
        types = import_genai_types()
        config_kwargs = {"output_dimensionality": output_dimensionality}
        if not use_prompt_instruction:
            config_kwargs["task_type"] = task_type
        config = types.EmbedContentConfig(**config_kwargs)
    except Exception:
        from types import SimpleNamespace
        config_kwargs = {"output_dimensionality": output_dimensionality}
        if not use_prompt_instruction:
            config_kwargs["task_type"] = task_type
        config = SimpleNamespace(**config_kwargs)

    client = (
        _make_gemini_client(request_timeout_ms=request_timeout_ms)
        if request_timeout_ms is not None
        else _make_gemini_client()
    )
    attempts = max(1, _QUERY_EMBEDDING_RETRIES + 1)
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            response = client.models.embed_content(
                model=model,
                contents=embed_text,
                config=config,
            )
            vector = list(response.embeddings[0].values)
            _cache_query_embedding(cache_key, vector)
            _clear_query_embedding_failure(failure_key)
            return vector
        except Exception as exc:
            last_exc = exc
            if _retryable_query_embedding_error(exc):
                _record_query_embedding_failure(failure_key, exc)
            if attempt >= attempts - 1 or not _retryable_query_embedding_error(exc):
                raise
            delay = _QUERY_EMBEDDING_RETRY_DELAY_SECONDS * (2 ** attempt)
            if delay > 0:
                time.sleep(delay)
    raise RuntimeError(f"query embedding failed: {last_exc}")


def _embed_queries(
    queries: Sequence[str],
    *,
    model: str,
    output_dimensionality: int | None,
    task_type: str = "RETRIEVAL_QUERY",
    request_timeout_ms: int | None = None,
) -> List[List[float]]:
    if not queries:
        return []
    use_prompt_instruction = str(model or "").strip().lower() == "gemini-embedding-2"
    embed_queries = [
        f"task: search result | query: {query}"
        for query in queries
    ] if use_prompt_instruction else list(queries)
    types = import_genai_types()
    config_kwargs = {"output_dimensionality": output_dimensionality}
    if not use_prompt_instruction:
        config_kwargs["task_type"] = task_type
    config = types.EmbedContentConfig(**config_kwargs)
    client = (
        _make_gemini_client(request_timeout_ms=request_timeout_ms)
        if request_timeout_ms is not None
        else _make_gemini_client()
    )
    vectors: List[List[float]] = []
    batch_size = _GEMINI_QUERY_EMBEDDING_BATCH_SIZE
    for start in range(0, len(embed_queries), batch_size):
        batch_queries = embed_queries[start : start + batch_size]
        # The Google Gen AI SDK treats a plain list[str] as consecutive parts
        # of one Content object. Keep each query in its own Content so every
        # provider batch has a strict one-to-one query/vector contract.
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=query)],
            )
            for query in batch_queries
        ]
        attempts = max(1, _QUERY_EMBEDDING_RETRIES + 1)
        for attempt in range(attempts):
            try:
                response = client.models.embed_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                batch_vectors = [list(embedding.values) for embedding in response.embeddings]
                if len(batch_vectors) != len(batch_queries):
                    raise RuntimeError(
                        "Gemini query embedding cardinality mismatch: "
                        f"requested={len(batch_queries)} returned={len(batch_vectors)} "
                        f"batch_start={start}"
                    )
                vectors.extend(batch_vectors)
                break
            except Exception as exc:
                if attempt >= attempts - 1 or not _retryable_query_embedding_error(exc):
                    raise
                delay = _QUERY_EMBEDDING_RETRY_DELAY_SECONDS * (2 ** attempt)
                if delay > 0:
                    time.sleep(delay)
    if len(vectors) != len(embed_queries):
        raise RuntimeError(
            "Gemini query embedding cardinality mismatch: "
            f"requested={len(embed_queries)} returned={len(vectors)}"
        )
    return vectors


@dataclass
class RetrievedRecord:
    record_id: str
    record_type: str
    score: float
    source: str


class AdaptiveHybridRetriever:
    def __init__(self, *, config: Dict[str, Any], work_dir: str | Path):
        self.work_dir = Path(work_dir).resolve()
        self.config = apply_vector_upload_manifest_config(config, self.work_dir)
        self.runtime_artifact_contract = dict(
            self.config.get("_runtime_artifact_contract") or {}
        )
        retrieval_cfg = self.config.get("retrieval", {}) or {}
        embed_cfg = self.config.get("embedder", {}) or {}
        vector_store_cfg = self.config.get("vector_store", {}) or {}

        # Historical profiles used prompt-specific semantic aliases to rescue
        # individual evaluation cases. Production V2 relies on the query
        # planner, multilingual embeddings, hybrid recall, and reranking. Keep
        # the aliases only as an explicit rollback compatibility path.
        self.query_specific_retrieval_rules_enabled = bool(
            retrieval_cfg.get("query_specific_retrieval_rules_enabled", True)
        )

        self.vector_store_provider = str(
            vector_store_cfg.get("provider") or "pinecone"
        ).strip().lower()
        if self.vector_store_provider not in {"pinecone", "pgvector"}:
            raise ValueError(
                f"Unsupported vector_store.provider: {self.vector_store_provider}"
            )

        self.index_name = str(embed_cfg.get("pinecone_index") or "").strip()
        if self.vector_store_provider == "pinecone" and not self.index_name:
            raise ValueError("embedder.pinecone_index is required for retrieval")
        if self.vector_store_provider == "pgvector":
            self.index_name = (
                f"{str(vector_store_cfg.get('schema') or 'mbzuai_retrieval')}."
                f"{str(vector_store_cfg.get('records_table') or 'embedding_records')}"
            )
        self.vector_release_id = str(
            vector_store_cfg.get("release_id")
            or self.runtime_artifact_contract.get("namespace_release_id")
            or self.work_dir.name
        ).strip()

        self.model = str(embed_cfg.get("model") or "gemini-embedding-2")
        self.output_dimensionality = int(embed_cfg.get("output_dimensionality") or 1536)
        self.gemini_request_timeout_ms = int(
            embed_cfg.get("gemini_request_timeout_ms")
            if embed_cfg.get("gemini_request_timeout_ms") is not None
            else _DEFAULT_GEMINI_REQUEST_TIMEOUT_MS
        )
        if self.gemini_request_timeout_ms <= 0:
            raise ValueError("embedder.gemini_request_timeout_ms must be greater than zero")
        self.namespace_chunks = str(embed_cfg.get("namespace_chunks") or "chunks")
        self.namespace_parents = str(embed_cfg.get("namespace_parents") or "parents")
        self.namespace_media = str(embed_cfg.get("namespace_media") or "media")
        self.namespace_page_cards = str(embed_cfg.get("namespace_page_cards") or "page_cards")
        self.namespace_actions = str(embed_cfg.get("namespace_actions") or "actions")
        self.namespace_facts = str(embed_cfg.get("namespace_facts") or "facts")
        self.namespace_evidence_spans = str(
            embed_cfg.get("namespace_evidence_spans")
            or retrieval_cfg.get("namespace_evidence_spans")
            or "evidence_spans"
        )
        self.namespace_summaries = str(embed_cfg.get("namespace_summaries") or retrieval_cfg.get("namespace_summaries") or "summaries")
        self.namespace_assertions = str(embed_cfg.get("namespace_assertions") or "assertions")
        self.sparse_index_name = str(embed_cfg.get("pinecone_sparse_index") or "").strip()

        self.dense_chunk_top_k = int(retrieval_cfg.get("dense_chunk_top_k", 12))
        self.dense_parent_top_k = int(retrieval_cfg.get("dense_parent_top_k", 6))
        self.dense_media_top_k = int(retrieval_cfg.get("dense_media_top_k", 6))
        self.dense_page_card_top_k = int(retrieval_cfg.get("dense_page_card_top_k", 6))
        self.dense_action_top_k = int(retrieval_cfg.get("dense_action_top_k", 6))
        self.dense_fact_top_k = int(retrieval_cfg.get("dense_fact_top_k", 8))
        self.dense_evidence_span_top_k = int(retrieval_cfg.get("dense_evidence_span_top_k", max(8, self.dense_fact_top_k)))
        self.dense_summary_top_k = int(retrieval_cfg.get("dense_summary_top_k", max(4, self.dense_parent_top_k)))
        self.dense_assertion_top_k = int(retrieval_cfg.get("dense_assertion_top_k", max(8, self.dense_fact_top_k)))
        self.lexical_top_k = int(retrieval_cfg.get("lexical_top_k", 12))
        self.sparse_chunk_top_k = int(retrieval_cfg.get("sparse_chunk_top_k", self.lexical_top_k))
        self.sparse_parent_top_k = int(retrieval_cfg.get("sparse_parent_top_k", max(4, self.lexical_top_k // 2)))
        self.sparse_media_top_k = int(retrieval_cfg.get("sparse_media_top_k", max(4, self.lexical_top_k // 2)))
        self.sparse_fact_top_k = int(retrieval_cfg.get("sparse_fact_top_k", max(6, self.lexical_top_k)))
        self.sparse_evidence_span_top_k = int(retrieval_cfg.get("sparse_evidence_span_top_k", max(8, self.sparse_fact_top_k)))
        self.sparse_summary_top_k = int(retrieval_cfg.get("sparse_summary_top_k", max(4, self.sparse_parent_top_k)))
        self.sparse_assertion_top_k = int(retrieval_cfg.get("sparse_assertion_top_k", max(8, self.sparse_fact_top_k)))
        self.rrf_k = int(retrieval_cfg.get("rrf_k", 60))
        self.fact_neighbor_window = int(retrieval_cfg.get("fact_neighbor_window", 1))
        self.max_context_chunks = int(retrieval_cfg.get("max_context_chunks", 12))
        self.max_parent_chunks = int(retrieval_cfg.get("max_parent_chunks", 10))
        self.max_media_results = int(retrieval_cfg.get("max_media_results", 4))
        self.media_abstain_rescue_min_score = max(
            0.0,
            float(retrieval_cfg.get("media_abstain_rescue_min_score", 1.25)),
        )
        self.media_abstain_rescue_min_overlap = max(
            0.0,
            min(
                1.0,
                float(retrieval_cfg.get("media_abstain_rescue_min_overlap", 0.35)),
            ),
        )
        self.dense_recall_floor_k = max(
            0,
            int(retrieval_cfg.get("dense_recall_floor_k", 2) or 0),
        )
        self.dense_recall_window = max(
            self.dense_recall_floor_k,
            min(
                self.max_context_chunks,
                int(retrieval_cfg.get("dense_recall_window", 10) or 10),
            ),
        )
        self.dense_parent_recall_floor_k = max(
            0,
            int(retrieval_cfg.get("dense_parent_recall_floor_k", 1) or 0),
        )
        self.dense_parent_recall_window = max(
            self.dense_parent_recall_floor_k,
            int(retrieval_cfg.get("dense_parent_recall_window", 5) or 5),
        )
        self.same_parent_expand_threshold = int(retrieval_cfg.get("same_parent_expand_threshold", 2))
        self.enable_sparse = bool(retrieval_cfg.get("enable_sparse", True))
        self.external_lanes_on_embedding_failure = bool(
            retrieval_cfg.get("external_lanes_on_embedding_failure", False)
        )
        self.pinecone_query_timeout_seconds = max(
            1.0,
            float(retrieval_cfg.get("pinecone_query_timeout_seconds", 10.0) or 10.0),
        )
        self.legacy_vectorstore_contract = str(retrieval_cfg.get("legacy_vectorstore_contract") or "").strip()
        self.legacy_hybrid_sparse_enabled = bool(retrieval_cfg.get("legacy_hybrid_sparse_enabled", False))
        self.legacy_bm25_model_file = str(retrieval_cfg.get("legacy_bm25_model_file") or "").strip()
        self.hybrid_alpha = float(retrieval_cfg.get("hybrid_alpha") or 0.5)
        self.enable_rerank = bool(retrieval_cfg.get("enable_rerank", True))
        self.rerank_model = str(retrieval_cfg.get("rerank_model") or "pinecone-rerank-v0")
        self.rerank_top_n = int(retrieval_cfg.get("rerank_top_n", 24))
        self.rerank_return_top_k = int(retrieval_cfg.get("rerank_return_top_k", 8))
        self.rerank_fact_top_n = int(retrieval_cfg.get("rerank_fact_top_n", min(self.rerank_top_n, 12)))
        self.rerank_fact_return_top_k = int(
            retrieval_cfg.get("rerank_fact_return_top_k", min(self.rerank_return_top_k, 6))
        )
        self.rerank_query_max_tokens = int(retrieval_cfg.get("rerank_query_max_tokens", 32))
        self.rerank_doc_max_tokens = int(retrieval_cfg.get("rerank_doc_max_tokens", 96))
        self.rerank_doc_max_chars = int(retrieval_cfg.get("rerank_doc_max_chars", 800))
        self.rerank_request_max_chars = int(retrieval_cfg.get("rerank_request_max_chars", 9000))
        self.rerank_retry_doc_max_tokens = [
            int(value)
            for value in (retrieval_cfg.get("rerank_retry_doc_max_tokens") or [72, 56, 40])
            if int(value) > 0
        ]
        self.rerank_skip_high_confidence_fact = bool(
            retrieval_cfg.get("rerank_skip_high_confidence_fact", True)
        )
        self.rerank_skip_fact_overlap = float(retrieval_cfg.get("rerank_skip_fact_overlap", 0.72))
        self.rerank_skip_fact_support_score = float(
            retrieval_cfg.get("rerank_skip_fact_support_score", 0.025)
        )
        self.abstain_min_token_overlap = float(retrieval_cfg.get("abstain_min_token_overlap", 0.12))
        self.fact_abstain_min_token_overlap = float(retrieval_cfg.get("fact_abstain_min_token_overlap", 0.20))
        self.fact_require_fact_support_overlap = float(retrieval_cfg.get("fact_require_fact_support_overlap", 0.35))
        self.abstain_min_support_score = float(retrieval_cfg.get("abstain_min_support_score", 0.05))
        self.temporal_exact_year_guard_enabled = bool(
            retrieval_cfg.get("temporal_exact_year_guard_enabled", True)
        )
        self.temporal_guard_top_k = max(1, int(retrieval_cfg.get("temporal_guard_top_k", 5) or 5))
        self.parent_candidate_top_k = int(retrieval_cfg.get("parent_candidate_top_k", 3))
        self.source_weights = {
            "dense_chunks": float(retrieval_cfg.get("weight_dense_chunks", 1.0)),
            "sparse_chunks": float(retrieval_cfg.get("weight_sparse_chunks", 1.2)),
            "local_chunks": float(retrieval_cfg.get("weight_local_chunks", 1.5)),
            "graph_relation_chunks": float(retrieval_cfg.get("weight_graph_relation_chunks", 2.2)),
            "dense_parents": float(retrieval_cfg.get("weight_dense_parents", 0.8)),
            "sparse_parents": float(retrieval_cfg.get("weight_sparse_parents", 0.9)),
            "local_parents": float(retrieval_cfg.get("weight_local_parents", 1.4)),
            "graph_relation_parents": float(retrieval_cfg.get("weight_graph_relation_parents", 1.6)),
            "dense_media": float(retrieval_cfg.get("weight_dense_media", 0.8)),
            "dense_page_cards": float(retrieval_cfg.get("weight_dense_page_cards", 1.0)),
            "dense_actions": float(retrieval_cfg.get("weight_dense_actions", 1.2)),
            "sparse_media": float(retrieval_cfg.get("weight_sparse_media", 0.8)),
            "local_media": float(retrieval_cfg.get("weight_local_media", 1.4)),
            "dense_facts": float(retrieval_cfg.get("weight_dense_facts", 1.3)),
            "sparse_facts": float(retrieval_cfg.get("weight_sparse_facts", 1.5)),
            "dense_evidence_spans": float(retrieval_cfg.get("weight_dense_evidence_spans", 1.8)),
            "sparse_evidence_spans": float(retrieval_cfg.get("weight_sparse_evidence_spans", 2.0)),
            "local_evidence_spans": float(retrieval_cfg.get("weight_local_evidence_spans", 2.1)),
            "dense_summaries": float(retrieval_cfg.get("weight_dense_summaries", 1.15)),
            "sparse_summaries": float(retrieval_cfg.get("weight_sparse_summaries", 1.25)),
            "dense_assertions": float(retrieval_cfg.get("weight_dense_assertions", 2.0)),
            "sparse_assertions": float(retrieval_cfg.get("weight_sparse_assertions", 2.2)),
            "local_facts": float(retrieval_cfg.get("weight_local_facts", 1.9)),
            "local_answers": float(retrieval_cfg.get("weight_local_answers", 2.1)),
            "graph_relation_facts": float(retrieval_cfg.get("weight_graph_relation_facts", 2.4)),
            "support_parents": float(retrieval_cfg.get("weight_support_parents", 1.8)),
        }
        self._dense_index = None
        self._sparse_index = None
        self._pinecone_client = None
        self._legacy_bm25_encoder = None
        self._legacy_bm25_encoder_lock = threading.Lock()
        self._thread_state = threading.local()
        self._pgvector_store = None
        if self.vector_store_provider == "pgvector":
            if self.enable_sparse:
                raise ValueError(
                    "pgvector runtime is configured for the selected dense-graph profile; "
                    "retrieval.enable_sparse must be false"
                )
            if self.enable_rerank:
                raise ValueError(
                    "Pinecone inference reranking is unavailable with pgvector; "
                    "set retrieval.enable_rerank=false or configure a provider-neutral reranker"
                )
            if not self.vector_release_id:
                raise ValueError("A release_id is required for pgvector retrieval")
            from pipeline.vectorstores.pgvector_store import PgVectorStore

            self._pgvector_store = PgVectorStore.from_config(
                self.config,
                purpose="read",
            )
        # Retrieval request diagnostics and provider handles are request-local
        # or thread-local. Immutable in-memory indexes can therefore be safely
        # shared by concurrent service requests.
        self.supports_shared_parallel_retrieval = True
        self.parallel_lane_workers = max(1, int(retrieval_cfg.get("parallel_lane_workers", 6) or 6))
        default_local_lane_workers = max(1, min(4, self.parallel_lane_workers))
        self.persistent_lane_executors = bool(retrieval_cfg.get("persistent_lane_executors", True))
        self.split_lane_executors = bool(retrieval_cfg.get("split_lane_executors", True))
        self.parallel_remote_lane_workers = max(
            1,
            int(retrieval_cfg.get("parallel_remote_lane_workers", self.parallel_lane_workers) or self.parallel_lane_workers),
        )
        self.parallel_local_lane_workers = max(
            1,
            int(retrieval_cfg.get("parallel_local_lane_workers", default_local_lane_workers) or default_local_lane_workers),
        )
        self.enable_local_bm25_fallback = bool(retrieval_cfg.get("enable_local_bm25_fallback", False))
        self.local_index_max_postings_per_token = max(
            16,
            int(retrieval_cfg.get("local_index_max_postings_per_token", 512) or 512),
        )
        self.local_chunk_candidate_pool = max(
            8,
            int(retrieval_cfg.get("local_chunk_candidate_pool", 96) or 96),
        )
        self.local_answer_top_k = max(
            4,
            int(retrieval_cfg.get("local_answer_top_k", max(self.sparse_fact_top_k, 10)) or max(self.sparse_fact_top_k, 10)),
        )
        self.local_answer_candidate_pool = max(
            8,
            int(retrieval_cfg.get("local_answer_candidate_pool", 96) or 96),
        )
        self.local_parent_candidate_pool = max(
            8,
            int(retrieval_cfg.get("local_parent_candidate_pool", 64) or 64),
        )
        self.local_media_candidate_pool = max(
            8,
            int(retrieval_cfg.get("local_media_candidate_pool", 64) or 64),
        )
        self.local_evidence_span_candidate_pool = max(
            12,
            int(retrieval_cfg.get("local_evidence_span_candidate_pool", 128) or 128),
        )

        bundle_path = _first_existing_stage_file(
            self.work_dir,
            ("finalize_retrieval_bundle", "format_retrieval", "build_retrieval_bundle"),
            "retrieval_bundle.json",
        )
        lexical_path = _first_existing_stage_file(
            self.work_dir,
            ("finalize_retrieval_bundle", "format_retrieval", "build_retrieval_bundle"),
            "lexical_corpus.json",
        )
        promoted_assertions_path = self.work_dir / "stage_outputs" / "promote_assertions" / "promoted_assertions.json"
        self.bundle = load_json_safe(bundle_path, {}) or {}
        self.lexical_records = load_json_safe(lexical_path, []) or []
        self.promoted_assertions = load_json_safe(promoted_assertions_path, []) or []
        if not isinstance(self.bundle, dict):
            raise ValueError(f"Invalid retrieval bundle: {bundle_path}")
        if not isinstance(self.lexical_records, list):
            raise ValueError(f"Invalid lexical corpus: {lexical_path}")
        if not isinstance(self.promoted_assertions, list):
            self.promoted_assertions = []

        self.chunk_map = {record["id"]: record for record in self.bundle.get("chunk_records", []) if isinstance(record, dict)}
        self.parent_map = {record["id"]: record for record in self.bundle.get("parent_records", []) if isinstance(record, dict)}
        self.media_map = {record["id"]: record for record in self.bundle.get("media_records", []) if isinstance(record, dict)}
        self.page_card_map = {
            record["id"]: record
            for record in self.bundle.get("page_card_records", [])
            if isinstance(record, dict) and str(record.get("id") or "")
        }
        self.action_map = {
            record["id"]: record
            for record in self.bundle.get("action_records", [])
            if isinstance(record, dict) and str(record.get("id") or "")
        }
        self.fact_map = {record["id"]: record for record in self.bundle.get("fact_records", []) if isinstance(record, dict)}
        self.evidence_span_map = {
            record["id"]: record
            for record in self.bundle.get("evidence_span_records", [])
            if isinstance(record, dict) and str(record.get("id") or "")
        }
        self.summary_map = {
            record["id"]: record
            for record in self.bundle.get("summary_records", [])
            if isinstance(record, dict) and str(record.get("id") or "")
        }
        self.assertion_map = {
            record["id"]: record
            for record in self.bundle.get("assertion_records", [])
            if isinstance(record, dict) and str(record.get("id") or "")
        }
        self.entity_map = {
            record["id"]: record
            for record in self.bundle.get("entity_records", [])
            if isinstance(record, dict) and str(record.get("id") or "")
        }
        answer_records = [
            record
            for record in (self.bundle.get("answer_records") or [])
            if isinstance(record, dict) and str(record.get("id") or "") and answer_record_looks_valid(record)
        ]
        if self.promoted_assertions:
            refreshed_assertion_answers = [
                record
                for record in build_answer_records_from_assertions(self.promoted_assertions)
                if isinstance(record, dict) and str(record.get("id") or "") and answer_record_looks_valid(record)
            ]
            if refreshed_assertion_answers:
                answer_by_id = {str(record.get("id")): record for record in answer_records}
                for record in refreshed_assertion_answers:
                    answer_by_id[str(record.get("id"))] = record
                answer_records = sorted(
                    answer_by_id.values(),
                    key=lambda item: (
                        -float(item.get("confidence") or 0.0),
                        -float(item.get("authority_score") or 0.0),
                        -float(item.get("freshness_score") or 0.0),
                        str(item.get("answer_type") or ""),
                        str(item.get("answer_subtype") or ""),
                        str(item.get("value") or ""),
                    ),
                )
        if not answer_records:
            answer_records = derive_answer_records_from_bundle(self.bundle)
            if answer_records:
                self.bundle["answer_records"] = answer_records
        elif self.bundle.get("answer_records") != answer_records:
            self.bundle["answer_records"] = answer_records
        self.answer_map = {record["id"]: record for record in answer_records}
        self.lexical_map = {record["id"]: record for record in self.lexical_records if isinstance(record, dict)}
        self.fact_texts_by_chunk: Dict[str, List[str]] = {}
        self.fact_tokens_by_id: Dict[str, List[str]] = {}
        self.fact_token_index: Dict[str, List[str]] = defaultdict(list)
        self.answer_tokens_by_id: Dict[str, List[str]] = {}
        self.answer_token_index: Dict[str, List[str]] = defaultdict(list)
        self.answer_ids_by_type: Dict[str, List[str]] = defaultdict(list)
        self.answer_ids_by_subtype: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        self.answer_context_texts_by_id: Dict[str, str] = {}
        self.answer_subjects_by_id: Dict[str, str] = {}
        self.answer_texts_by_chunk: Dict[str, List[str]] = {}
        for fact in self.fact_map.values():
            fact_text = _clean_text(fact.get("text") or fact.get("dense_text") or "")
            if not fact_text:
                continue
            fact_tokens = _tokenize(_fact_lookup_text(fact) or fact_text)
            self.fact_tokens_by_id[str(fact["id"])] = fact_tokens
            for token in dict.fromkeys(fact_tokens):
                self.fact_token_index[token].append(str(fact["id"]))
            for chunk_id in fact.get("linked_chunk_ids") or []:
                self.fact_texts_by_chunk.setdefault(str(chunk_id), []).append(fact_text)
        self.media_texts_by_chunk: Dict[str, List[str]] = {}
        self.media_texts_by_id: Dict[str, str] = {}
        for media in self.media_map.values():
            text = _clean_text(
                " ".join(
                    str(media.get(key) or "")
                    for key in ("title", "caption", "description", "context", "transcript", "text")
                )
            )
            if not text:
                continue
            self.media_texts_by_id[str(media["id"])] = text
            for chunk_id in media.get("linked_chunk_ids") or []:
                self.media_texts_by_chunk.setdefault(str(chunk_id), []).append(text)
        for answer in self.answer_map.values():
            answer_id = str(answer.get("id") or "")
            if not answer_id:
                continue
            answer_type = str(answer.get("answer_type") or "")
            answer_subtype = str(answer.get("answer_subtype") or "")
            self.answer_ids_by_type[answer_type].append(answer_id)
            self.answer_ids_by_subtype[(answer_type, answer_subtype)].append(answer_id)
            subject_text = _clean_text(answer.get("subject_text") or "")
            self.answer_subjects_by_id[answer_id] = subject_text
            anchor_fragments: List[str] = []
            for chunk_id in answer.get("linked_chunk_ids") or []:
                chunk = self.chunk_map.get(str(chunk_id)) or {}
                chunk_text = _clean_text(chunk.get("dense_text") or chunk.get("text") or "")
                if chunk_text:
                    anchor_fragments.append(_truncate_tokens(chunk_text, max_tokens=80))
                    break
            if not anchor_fragments:
                for fact_id in answer.get("linked_fact_ids") or []:
                    fact = self.fact_map.get(str(fact_id)) or {}
                    fact_text = _clean_text(fact.get("text") or fact.get("dense_text") or "")
                    if fact_text:
                        anchor_fragments.append(_truncate_tokens(fact_text, max_tokens=48))
                        break
            answer_text = " ".join(
                part
                for part in (
                    answer.get("value"),
                    answer.get("text"),
                    " ".join(answer.get("qualifiers") or []),
                    answer.get("document_title"),
                    answer.get("heading"),
                    answer.get("source_url"),
                    subject_text,
                    *anchor_fragments,
                )
                if part
            )
            self.answer_context_texts_by_id[answer_id] = answer_text
            answer_tokens = _tokenize(answer_text)
            self.answer_tokens_by_id[answer_id] = answer_tokens
            for token in dict.fromkeys(answer_tokens):
                self.answer_token_index[token].append(answer_id)
            answer_display_text = _clean_text(answer.get("text") or answer.get("value") or "")
            if answer_display_text:
                for chunk_id in answer.get("linked_chunk_ids") or []:
                    self.answer_texts_by_chunk.setdefault(str(chunk_id), []).append(answer_display_text)
                for fact_id in answer.get("linked_fact_ids") or []:
                    for chunk_id in (self.fact_map.get(str(fact_id)) or {}).get("linked_chunk_ids") or []:
                        self.answer_texts_by_chunk.setdefault(str(chunk_id), []).append(answer_display_text)
        for answer_type, answer_ids in list(self.answer_ids_by_type.items()):
            self.answer_ids_by_type[answer_type] = sorted(
                answer_ids,
                key=lambda answer_id: (
                    -float((self.answer_map.get(answer_id) or {}).get("confidence") or 0.0),
                    str(answer_id),
                ),
            )
        for answer_key, answer_ids in list(self.answer_ids_by_subtype.items()):
            self.answer_ids_by_subtype[answer_key] = sorted(
                answer_ids,
                key=lambda answer_id: (
                    -float((self.answer_map.get(answer_id) or {}).get("confidence") or 0.0),
                    str(answer_id),
                ),
            )

        self.chunk_ids_by_section: Dict[str, List[str]] = {}
        self.chunk_ids_by_page: Dict[str, List[str]] = {}
        parent_ids_by_chunk: Dict[str, List[str]] = defaultdict(list)
        section_parent_ids_by_chunk: Dict[str, List[str]] = defaultdict(list)
        page_parent_ids_by_chunk: Dict[str, List[str]] = defaultdict(list)
        for parent in self.parent_map.values():
            parent_id = str(parent.get("id") or "")
            child_chunk_ids = [
                str(value)
                for value in (parent.get("child_chunk_ids") or [])
                if str(value) in self.chunk_map
            ]
            if parent.get("parent_type") == "section":
                self.chunk_ids_by_section[parent_id] = child_chunk_ids
                for chunk_id in child_chunk_ids:
                    section_parent_ids_by_chunk[chunk_id].append(parent_id)
            elif parent.get("parent_type") == "page":
                self.chunk_ids_by_page[parent_id] = child_chunk_ids
                for chunk_id in child_chunk_ids:
                    page_parent_ids_by_chunk[chunk_id].append(parent_id)
            for chunk_id in child_chunk_ids:
                parent_ids_by_chunk[chunk_id].append(parent_id)
        self.parent_ids_by_chunk = {
            chunk_id: list(dict.fromkeys(parent_ids))
            for chunk_id, parent_ids in parent_ids_by_chunk.items()
        }
        self.section_parent_ids_by_chunk = {
            chunk_id: list(dict.fromkeys(parent_ids))
            for chunk_id, parent_ids in section_parent_ids_by_chunk.items()
        }
        self.page_parent_ids_by_chunk = {
            chunk_id: list(dict.fromkeys(parent_ids))
            for chunk_id, parent_ids in page_parent_ids_by_chunk.items()
        }
        chunk_ids_by_revision: Dict[str, List[str]] = defaultdict(list)
        chunk_ids_by_page_card: Dict[str, List[str]] = defaultdict(list)
        for chunk_id, chunk in self.chunk_map.items():
            revision_id = str(chunk.get("document_revision_id") or "")
            if revision_id:
                chunk_ids_by_revision[revision_id].append(chunk_id)
            for page_card_id in chunk.get("page_card_ids") or []:
                page_card_id = str(page_card_id or "")
                if page_card_id:
                    chunk_ids_by_page_card[page_card_id].append(chunk_id)
        for page_card_id, page_card in self.page_card_map.items():
            linked_chunk_ids = [
                str(value)
                for value in (page_card.get("linked_chunk_ids") or [])
                if str(value) in self.chunk_map
            ]
            linked_chunk_ids.extend(chunk_ids_by_page_card.get(page_card_id, []))
            revision_id = str(page_card.get("document_revision_id") or "")
            if revision_id:
                linked_chunk_ids.extend(chunk_ids_by_revision.get(revision_id, []))
            chunk_ids_by_page_card[page_card_id] = list(
                dict.fromkeys(linked_chunk_ids)
            )
        self.chunk_ids_by_page_card = dict(chunk_ids_by_page_card)
        parent_aliases: Dict[str, List[str]] = defaultdict(list)
        for parent_id in self.parent_map:
            parent_aliases[parent_id].append(parent_id)
        for chunk_id, chunk in self.chunk_map.items():
            for raw_key, canonical_ids in (
                (
                    str(chunk.get("section_key") or ""),
                    self.section_parent_ids_by_chunk.get(chunk_id, []),
                ),
                (
                    str(chunk.get("page_key") or ""),
                    self.page_parent_ids_by_chunk.get(chunk_id, []),
                ),
            ):
                if raw_key:
                    parent_aliases[raw_key].extend(canonical_ids)
        self._parent_id_aliases = {
            alias: list(dict.fromkeys(parent_ids))
            for alias, parent_ids in parent_aliases.items()
        }
        self._namespace_token_index: Dict[str, Dict[str, List[str]]] = {}
        self._namespace_tokens_by_id: Dict[str, Dict[str, set[str]]] = {}
        self._bm25_by_namespace: Dict[str, Tuple[List[str], BM25Okapi]] = {}
        for namespace, record_types in _NAMESPACE_RECORD_TYPES.items():
            scoped_records = [
                record
                for record in self.lexical_records
                if str(record.get("record_type") or "") in record_types
            ]
            if not scoped_records:
                continue
            token_index: Dict[str, List[str]] = defaultdict(list)
            tokens_by_id: Dict[str, set[str]] = {}
            for record in scoped_records:
                record_id = str(record.get("id") or "")
                if not record_id:
                    continue
                tokens = set(_tokenize(record.get("text", "")))
                if not tokens:
                    continue
                tokens_by_id[record_id] = tokens
                for token in tokens:
                    token_index[token].append(record_id)
            self._namespace_token_index[namespace] = dict(token_index)
            self._namespace_tokens_by_id[namespace] = tokens_by_id
            if self.enable_local_bm25_fallback:
                tokenized_corpus = [_tokenize(record.get("text", "")) for record in scoped_records]
                self._bm25_by_namespace[namespace] = (
                    [str(record.get("id") or "") for record in scoped_records],
                    BM25Okapi(tokenized_corpus),
                )
        # Lexical records use canonical record-kind namespaces while vector
        # stores may use release/project-prefixed physical namespaces.  Every
        # local/sparse lane must resolve both names to the same immutable
        # in-memory index; otherwise a valid lexical query silently misses and
        # older code falls back to an unbounded corpus scan.
        configured_namespace_aliases = {
            self.namespace_chunks: "chunks",
            self.namespace_parents: "parents",
            self.namespace_media: "media",
            self.namespace_page_cards: "page_cards",
            self.namespace_actions: "actions",
            self.namespace_facts: "facts",
            self.namespace_evidence_spans: "evidence_spans",
            self.namespace_summaries: "summaries",
            self.namespace_assertions: "assertions",
        }
        _register_configured_namespace_aliases(
            configured_to_canonical=configured_namespace_aliases,
            token_indexes=self._namespace_token_index,
            tokens_by_id=self._namespace_tokens_by_id,
            bm25_indexes=self._bm25_by_namespace,
        )

    @classmethod
    def from_config(cls, *, config_name: str, work_dir: str | Path) -> "AdaptiveHybridRetriever":
        from ..core.config import load_effective_config
        from ..core.runtime_contract import validate_runtime_artifact_contract

        try:
            config = load_effective_config(config_name, work_dir=work_dir)
        except FileNotFoundError:
            config = load_config(config_name)
        runtime_contract = validate_runtime_artifact_contract(config, work_dir)
        config = apply_vector_upload_manifest_config(config, work_dir)
        config["_runtime_artifact_contract"] = runtime_contract
        backend = str((config.get("retrieval") or {}).get("retriever_backend") or "vector").strip().lower()
        if backend == "graph_hybrid":
            from .graph_rag import GraphRAGRetriever

            return GraphRAGRetriever(config=config, work_dir=work_dir)
        if backend == "routed_hybrid":
            from .routed_hybrid import RoutedHybridRetriever

            return RoutedHybridRetriever(config=config, work_dir=work_dir)
        return cls(config=config, work_dir=work_dir)

    def _pinecone_client_obj(self):
        override = getattr(self, "_pinecone_client", None)
        if override is not None:
            return override
        from pinecone import Pinecone
        client = getattr(self._thread_state, "pinecone_client", None)
        if client is None:
            client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
            self._thread_state.pinecone_client = client
        return client

    def _parent_ids_for_chunk(
        self,
        chunk_id: str,
        *,
        parent_type: str | None = None,
    ) -> List[str]:
        chunk_id = str(chunk_id or "")
        if parent_type == "section":
            parent_ids = list(getattr(self, "section_parent_ids_by_chunk", {}).get(chunk_id, []))
        elif parent_type == "page":
            parent_ids = list(getattr(self, "page_parent_ids_by_chunk", {}).get(chunk_id, []))
        else:
            parent_ids = list(getattr(self, "parent_ids_by_chunk", {}).get(chunk_id, []))
        # Backward compatibility for legacy bundles where section_key/page_key
        # already were canonical parent IDs.
        chunk = self.chunk_map.get(chunk_id) or {}
        legacy_keys = (
            ("section_key",) if parent_type == "section"
            else ("page_key",) if parent_type == "page"
            else ("section_key", "page_key")
        )
        for key in legacy_keys:
            candidate = str(chunk.get(key) or "")
            if candidate in self.parent_map:
                parent_ids.append(candidate)
        return list(dict.fromkeys(parent_ids))

    def _canonical_parent_ids(
        self,
        parent_ids: Iterable[str],
        *,
        chunk_ids: Iterable[str] | None = None,
    ) -> List[str]:
        contextual_chunk_ids = {str(value) for value in (chunk_ids or []) if str(value)}
        output: List[str] = []
        for raw_parent_id in parent_ids or []:
            raw_parent_id = str(raw_parent_id or "")
            if not raw_parent_id:
                continue
            candidates = list(getattr(self, "_parent_id_aliases", {}).get(raw_parent_id, []))
            if raw_parent_id in self.parent_map:
                candidates.insert(0, raw_parent_id)
            if contextual_chunk_ids and candidates:
                scoped = [
                    parent_id
                    for parent_id in candidates
                    if contextual_chunk_ids
                    & {
                        str(value)
                        for value in (self.parent_map.get(parent_id, {}).get("child_chunk_ids") or [])
                        if str(value)
                    }
                ]
                if scoped:
                    candidates = scoped
            output.extend(candidates)
        return list(dict.fromkeys(output))

    def _pinecone_index_handle(self, index_name: str, *, cache_attr: str):
        index = getattr(self._thread_state, cache_attr, None)
        if index is not None:
            return index

        client = self._pinecone_client_obj()
        with _PINECONE_INDEX_HOST_CACHE_LOCK:
            host = _PINECONE_INDEX_HOST_CACHE.get(index_name)
            if not host:
                try:
                    description = client.describe_index(index_name)
                    if isinstance(description, dict):
                        host = description.get("host")
                    else:
                        host = getattr(description, "host", None)
                except Exception:
                    host = None
                if host:
                    _PINECONE_INDEX_HOST_CACHE[index_name] = str(host)
            if host:
                host = str(host)

        if host:
            index = client.Index(host=str(host))
        else:
            index = client.Index(index_name)
        setattr(self._thread_state, cache_attr, index)
        return index

    def _pinecone_index(self):
        override = getattr(self, "_dense_index", None)
        if override is not None:
            return override
        return self._pinecone_index_handle(self.index_name, cache_attr="dense_index")

    def _pinecone_sparse_index(self):
        override = getattr(self, "_sparse_index", None)
        if override is not None:
            return override
        return self._pinecone_index_handle(self.sparse_index_name, cache_attr="sparse_index")

    def _legacy_bm25_encoder_obj(self):
        encoder = getattr(self, "_legacy_bm25_encoder", None)
        if encoder is not None:
            return encoder
        if not self.legacy_bm25_model_file:
            return None
        with self._legacy_bm25_encoder_lock:
            encoder = getattr(self, "_legacy_bm25_encoder", None)
            if encoder is not None:
                return encoder
            try:
                from pinecone_text.sparse import BM25Encoder

                encoder = BM25Encoder().load(self.legacy_bm25_model_file)
            except Exception as exc:
                logger.warning("Failed to load legacy BM25 model %s: %s", self.legacy_bm25_model_file, exc)
                encoder = None
            self._legacy_bm25_encoder = encoder
            return encoder

    def _legacy_hybrid_query_payload(
        self,
        *,
        query: str | None,
        query_vector: List[float],
    ) -> Tuple[List[float], Dict[str, Any] | None]:
        if (
            not self.legacy_hybrid_sparse_enabled
            or self.legacy_vectorstore_contract != _LEGACY_VECTORSTORE_CONTRACT
            or not query
        ):
            return query_vector, None
        encoder = self._legacy_bm25_encoder_obj()
        if encoder is None:
            return query_vector, None
        try:
            from pinecone_text.hybrid import hybrid_convex_scale

            sparse_vector = encoder.encode_queries(query)
            dense_vector, sparse_vector = hybrid_convex_scale(query_vector, sparse_vector, alpha=self.hybrid_alpha)
            return list(dense_vector), sparse_vector
        except Exception as exc:
            logger.warning("Failed to build legacy hybrid query vector: %s", exc)
            return query_vector, None

    def embed_query(self, query: str) -> List[float]:
        return _embed_query(
            query,
            model=self.model,
            output_dimensionality=self.output_dimensionality,
            task_type="RETRIEVAL_QUERY",
            request_timeout_ms=self.gemini_request_timeout_ms,
        )

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        return _embed_queries(
            queries,
            model=self.model,
            output_dimensionality=self.output_dimensionality,
            task_type="RETRIEVAL_QUERY",
            request_timeout_ms=self.gemini_request_timeout_ms,
        )

    def _dense_query_ids(
        self,
        *,
        query_vector: List[float],
        namespace: str,
        top_k: int,
        query: str | None = None,
        score_sink: MutableMapping[str, List[Dict[str, Any]]] | None = None,
        score_key: str = "",
    ) -> List[str]:
        if top_k <= 0:
            if score_sink is not None and score_key:
                score_sink[score_key] = []
            return []
        if self.vector_store_provider == "pgvector":
            if self._pgvector_store is None:
                raise RuntimeError("pgvector store is not initialized")
            matches = self._pgvector_store.query(
                release_id=self.vector_release_id,
                namespace=namespace,
                vector=query_vector,
                top_k=top_k,
            )
        else:
            dense_vector, sparse_vector = self._legacy_hybrid_query_payload(query=query, query_vector=query_vector)
            kwargs: Dict[str, Any] = {}
            if sparse_vector:
                kwargs["sparse_vector"] = sparse_vector
            matches = self._pinecone_index().query(
                vector=dense_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=False,
                include_values=False,
                timeout=self.pinecone_query_timeout_seconds,
                **kwargs,
            ).matches
        normalized_matches = [
            {
                "id": str(
                    getattr(match, "record_id", "")
                    or getattr(match, "id", "")
                ),
                "score": float(getattr(match, "score", 0.0) or 0.0),
            }
            for match in matches
            if str(
                getattr(match, "record_id", "")
                or getattr(match, "id", "")
            )
        ]
        if score_sink is not None and score_key:
            # The sink is owned by one retrieval request and every concurrent
            # lane writes a distinct key.  This preserves score provenance
            # without leaking mutable diagnostics across requests.
            score_sink[score_key] = normalized_matches
        return [value["id"] for value in normalized_matches]

    def close(self) -> None:
        store = getattr(self, "_pgvector_store", None)
        if store is not None:
            store.close()
            self._pgvector_store = None

    def _informative_query_tokens(self, query: str) -> List[str]:
        lookup_profile = _lookup_query_profile(query)
        if lookup_profile.is_exact_lookup:
            focus_tokens = [
                token
                for token in lookup_profile.focus_tokens
                if token not in _QUERY_STOPWORDS and len(token) >= 3
            ]
            attribute_tokens = [
                token
                for token in _tokenize(" ".join(lookup_profile.answer_types))
                if token not in _QUERY_STOPWORDS and len(token) >= 3
            ]
            raw_tokens = [
                token
                for token in _tokenize(query)
                if token not in _QUERY_STOPWORDS and len(token) >= 3
            ]
            if focus_tokens and lookup_profile.is_contact_lookup:
                raw_tokens = [
                    token
                    for token in raw_tokens
                    if token not in _LOOKUP_ENTITY_ROLE_TOKENS
                ]
            base_tokens = [*focus_tokens, *attribute_tokens, *raw_tokens]
            alias_tokens = (
                _semantic_query_alias_tokens(query)
                if getattr(
                    self,
                    "query_specific_retrieval_rules_enabled",
                    True,
                )
                else []
            )
            return list(dict.fromkeys([*base_tokens, *alias_tokens]))
        base_tokens = [
            token
            for token in _tokenize(query)
            if token not in _QUERY_STOPWORDS and len(token) >= 3
        ]
        alias_tokens = (
            _semantic_query_alias_tokens(query)
            if getattr(
                self,
                "query_specific_retrieval_rules_enabled",
                True,
            )
            else []
        )
        return list(dict.fromkeys([*base_tokens, *alias_tokens]))

    def _lexical_query_ids(self, query: str, top_k: int, *, namespace: str) -> List[str]:
        if top_k <= 0:
            return []
        informative_tokens = list(dict.fromkeys(self._informative_query_tokens(query)))
        if not informative_tokens:
            return []
        token_index = self._namespace_token_index.get(namespace) or {}
        tokens_by_id = self._namespace_tokens_by_id.get(namespace) or {}
        if token_index and tokens_by_id:
            candidate_hits: Dict[str, int] = defaultdict(int)
            for token in informative_tokens:
                for record_id in token_index.get(token, [])[: self.local_index_max_postings_per_token]:
                    candidate_hits[record_id] += 1
            if candidate_hits:
                candidate_cap = max(top_k * 8, 64)
                scored: List[Tuple[str, float]] = []
                informative_token_set = set(informative_tokens)
                phrase_tokens = [token for token in informative_tokens if len(token) >= 4]
                ordered_candidates = sorted(
                    candidate_hits.items(),
                    key=lambda item: (-item[1], item[0]),
                )[:candidate_cap]
                for record_id, hit_count in ordered_candidates:
                    record_tokens = tokens_by_id.get(record_id) or set()
                    if not record_tokens:
                        continue
                    overlap = len(record_tokens & informative_token_set) / float(len(informative_token_set))
                    text = str((self.lexical_map.get(record_id) or {}).get("text") or "")
                    score = overlap + _phrase_match_bonus(phrase_tokens, text)
                    score += min(max(hit_count - 1, 0) * 0.03, 0.15)
                    if score > 0.0:
                        scored.append((record_id, score))
                scored.sort(key=lambda item: (-item[1], item[0]))
                return [record_id for record_id, _score in scored[:top_k]]
        payload = self._bm25_by_namespace.get(namespace)
        if payload is None:
            return []
        record_ids, bm25 = payload
        scores = bm25.get_scores(_tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        result = []
        for idx, _score in ranked[:top_k]:
            record_id = record_ids[idx]
            if record_id:
                result.append(str(record_id))
        return result

    def _score_text_match(self, query: str, text: str) -> float:
        return self._score_preindexed_text_match(
            query,
            text=text,
            text_tokens=_token_set(text),
        )

    def _score_preindexed_text_match(
        self,
        query: str,
        *,
        text: str,
        text_tokens: Iterable[str],
    ) -> float:
        informative_tokens = list(dict.fromkeys(self._informative_query_tokens(query)))
        if not informative_tokens:
            return 0.0
        indexed_tokens = set(text_tokens or [])
        if not indexed_tokens:
            return 0.0
        overlap = len(indexed_tokens & set(informative_tokens)) / float(len(informative_tokens))
        phrase_bonus = _phrase_match_bonus(
            [token for token in informative_tokens if len(token) >= 4],
            text,
        )
        return overlap + phrase_bonus

    def _local_chunk_query_ids(self, query: str, *, top_k: int) -> List[str]:
        if top_k <= 0 or not self.chunk_map:
            return []
        candidate_ids = self._lexical_query_ids(
            query,
            max(top_k, self.local_chunk_candidate_pool),
            namespace=self.namespace_chunks,
        )
        if not candidate_ids:
            return []
        scored: List[Tuple[str, float]] = []
        for chunk_id in candidate_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            chunk_text = chunk.get("dense_text") or chunk.get("text") or ""
            score = self._score_text_match(query, chunk_text)
            score += _lookup_signal_bonus(query, chunk_text)
            score += self._fact_query_bonus(query, chunk_text)
            score += _answer_focus_match_bonus(query, chunk_text)
            if score <= 0.0:
                continue
            scored.append((chunk_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, _score in scored[:top_k]]

    def _local_parent_query_ids(self, query: str, *, top_k: int) -> List[str]:
        if top_k <= 0 or not self.parent_map:
            return []
        candidate_ids = self._lexical_query_ids(
            query,
            max(top_k, self.local_parent_candidate_pool),
            namespace=self.namespace_parents,
        )
        if not candidate_ids:
            # Dense parent retrieval remains available.  Never turn a lexical
            # miss into an unbounded scan of every parent in the serving path.
            return []
        scored: List[Tuple[str, float]] = []
        for parent_id in candidate_ids:
            parent = self.parent_map.get(parent_id)
            if not parent:
                continue
            score = self._scoped_parent_bonus(
                query,
                parent,
                indexed_tokens=(
                    self._namespace_tokens_by_id.get(self.namespace_parents, {}).get(parent_id)
                    or set()
                ),
            )
            if score <= 0.0:
                continue
            scored.append((parent_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [parent_id for parent_id, _score in scored[:top_k]]

    def _should_use_local_parent_lane(self, query: str, *, mode: QueryMode) -> bool:
        if mode == QueryMode.FACT or not self.parent_map:
            return False
        normalized = _clean_text(query).lower()
        query_tokens = set(self._informative_query_tokens(query))
        if any(
            phrase in normalized
            for phrase in (
                "legal basis",
                "institutional affiliation",
                "ai specializations",
                "graduate programs",
                "under which law",
                "law that created",
                "authority it is affiliated",
            )
        ):
            return True
        return bool(
            query_tokens
            & {
                "amenities",
                "affiliation",
                "affiliated",
                "authority",
                "basis",
                "established",
                "facilities",
                "facility",
                "graduate",
                "institutional",
                "law",
                "legal",
                "program",
                "programs",
                "services",
                "specialization",
                "specializations",
                "student-facing",
                "support",
            }
        )

    def _should_prefer_explicit_parent_chunks(self, query: str, *, mode: QueryMode, media_query: bool) -> bool:
        if media_query:
            return True
        if mode == QueryMode.FACT:
            return False
        normalized = _clean_text(query).lower()
        query_tokens = set(self._informative_query_tokens(query))
        if any(
            phrase in normalized
            for phrase in (
                "campus amenities",
                "everyday campus amenities",
                "student-facing campus services",
                "student facing campus services",
                "legal basis",
                "institutional affiliation",
                "under which law",
                "law that created",
                "authority it is affiliated",
            )
        ):
            return True
        return (
            ("law" in query_tokens or "established" in query_tokens)
            and ("affiliated" in query_tokens or "affiliation" in query_tokens or "authority" in query_tokens)
        ) or bool(query_tokens & {"amenity", "amenities", "facility", "facilities", "services"})

    def _media_keywords(self, query: str) -> List[str]:
        return [
            token
            for token in self._informative_query_tokens(query)
            if token in _MEDIA_PRIORITY_TOKENS or token in _VISUAL_INTENT_TOKENS
        ]

    def _specific_visual_tokens(self, query: str) -> List[str]:
        specific = {"map", "layout", "building", "buildings", "labeled", "labelled", "parking"}
        return [token for token in self._media_keywords(query) if token in specific]

    def _is_low_signal_media(self, media: Dict[str, Any]) -> bool:
        media_type = str(media.get("media_type") or "")
        title = _clean_text(media.get("title") or "")
        caption = _clean_text(media.get("caption") or "")
        description = _clean_text(media.get("description") or "")
        context = _clean_text(media.get("context") or "")
        transcript = _clean_text(media.get("transcript") or "")
        text = _clean_text(media.get("text") or "")
        combined = " ".join(
            part for part in (title, caption, description, context, transcript, text) if part
        ).lower()
        if any(phrase in combined for phrase in _LOW_SIGNAL_MEDIA_PHRASES):
            return True
        if media_type == "page_visual":
            combined_prefix = combined[:240]
            if combined_prefix.startswith("contents ") or combined_prefix.startswith("table of contents"):
                return True
            if "contents " in combined_prefix and sum(ch.isdigit() for ch in combined_prefix) >= 8:
                return True
        if _is_generic_figure_label(title) and not any((caption, description, context, transcript)):
            return True
        return False

    def _score_media_relevance(self, query: str, media: Dict[str, Any]) -> float:
        if self._is_low_signal_media(media):
            return -1.0
        media_id = str(media.get("id") or "")
        media_text = self.media_texts_by_id.get(media_id) or _clean_text(media.get("text") or "")
        if not media_text:
            return 0.0
        score = self._score_text_match(query, media_text) + self._media_query_bonus(query, media)
        query_keywords = set(self._media_keywords(query))
        media_tokens = set(_tokenize(media_text))
        if query_keywords:
            overlap = len(query_keywords & media_tokens) / float(len(query_keywords))
            score += overlap
            specific_tokens = set(self._specific_visual_tokens(query))
            if specific_tokens and not (specific_tokens & media_tokens):
                score -= 0.90
            if "map" in query_keywords and "map" not in media_tokens:
                score -= 0.45
            if "layout" in query_keywords and "layout" not in media_tokens:
                score -= 0.30
        query_has_arabic = bool(re.search(r"[\u0600-\u06ff]", str(query or "")))
        media_has_arabic = bool(re.search(r"[\u0600-\u06ff]", media_text))
        if query_has_arabic:
            score += 0.55 if media_has_arabic else -0.35
        elif media_has_arabic and "/ar/" in str(media.get("source_url") or "").casefold():
            score -= 0.45
        score += self._media_specificity_bonus(query, media_text)
        return score

    def _media_specificity_bonus(self, query: str, media_text: str) -> float:
        query_tokens = set(_tokenize(query))
        normalized_query = _clean_text(query).lower()
        text = _clean_text(media_text).lower()
        if not query_tokens or not text:
            return 0.0
        bonus = 0.0
        query_years = set(re.findall(r"\b20\d{2}\b", normalized_query))
        if query_years:
            media_years = set(re.findall(r"\b20\d{2}\b", text))
            if query_years & media_years:
                bonus += 3.2
            elif media_years:
                bonus -= 2.8
        page_match = re.search(r"\bpage\s+(\d{1,3})\b", normalized_query)
        if page_match:
            page_number = page_match.group(1)
            if re.search(rf"\b(?:page|pdf[_ ]page)\s*[:#-]?\s*{re.escape(page_number)}\b", text):
                bonus += 2.6
        if {"which", "labelled", "labeled"} & query_tokens and {"facility", "facilities"} & query_tokens:
            comma_count = text.count(",")
            bonus += min(comma_count * 0.12, 0.84)
            if comma_count < 2:
                bonus -= 0.30
                if any(term in text for term in ("map", "layout", "showing location")):
                    bonus -= 0.90
            if ":" in text:
                bonus += 0.10
        if "layout" in query_tokens and "layout" in text:
            bonus += 0.15
        if "collected data" in normalized_query:
            if "collected data" in text:
                bonus += 2.2
                if "automatically analys" in text:
                    bonus += 0.35
            else:
                bonus -= 0.35
        if any(marker in normalized_query for marker in ("بعد المطر", "بعد هطول المطر", "ما بعد المطر")):
            if any(marker in text for marker in ("after rain", "post-rainfall", "water accumulation")):
                bonus += 2.4
            else:
                bonus -= 0.35
        if re.search(r"\b(?:invite|invites|inviting)\b", normalized_query):
            if any(marker in text for marker in ("scan qr", "qr code", "digital copy", "download")):
                bonus += 2.4
            else:
                bonus -= 0.25
        if "copilot" in normalized_query:
            bonus += 2.0 if "copilot" in text else -0.45
        gpa_field_query = "gpa" in query_tokens or "grade point average" in normalized_query
        if gpa_field_query:
            has_complete_gpa_fields = (
                "cumulative grade point average" in text
                and "maximum possible grade point average" in text
            )
            if has_complete_gpa_fields:
                bonus += 4.5
            elif "gpa" in text or "grade point average" in text:
                bonus += 0.8
            else:
                bonus -= 1.25
            if "academic history" in normalized_query:
                academic_entry_fields = (
                    "university name",
                    "degree level",
                    "major",
                )
                if all(field in text for field in academic_entry_fields):
                    bonus += 1.5
                if "programming courses taken" in text:
                    bonus -= 1.25
        gender_query = (
            bool({"male", "female", "gender"} & query_tokens)
            or any(marker in normalized_query for marker in ("الذكور", "للذكور", "ذكور", "الإناث", "للإناث", "إناث"))
        )
        if gender_query:
            gender_markers = ("male", "female", "gender", "ذكور", "إناث")
            matched_gender_markers = sum(marker in text for marker in gender_markers)
            if matched_gender_markers >= 2:
                bonus += 1.8
            elif matched_gender_markers == 0:
                bonus -= 0.35
        if any(marker in normalized_query for marker in ("commencement 2025", "حفل التخرج 2025")):
            bonus += 1.8 if any(marker in text for marker in ("commencement 2025", "حفل التخرج 2025")) else -0.30
        requested_pages = list(
            dict.fromkeys(
                re.findall(
                    r"(?:\bpage|الصفحة)\s*[:#-]?\s*(\d{1,4})\b",
                    normalized_query,
                    flags=re.IGNORECASE,
                )
            )
        )
        if requested_pages:
            page_matched = any(
                re.search(
                    rf"(?:\bpage|الصفحة)\s*[:#-]?\s*{re.escape(page_number)}\b",
                    text,
                    flags=re.IGNORECASE,
                )
                for page_number in requested_pages
            )
            bonus += 3.0 if page_matched else -1.0

        generic_media_terms = {
            "image",
            "shown",
            "show",
            "visual",
            "diagram",
            "figure",
            "chart",
            "table",
            "photo",
            "picture",
            "surrounding",
            "text",
            "what",
            "does",
            "about",
            "the",
            "and",
        }
        ordered_terms = [
            token
            for token in self._informative_query_tokens(query)
            if token not in generic_media_terms and len(token) >= 3
        ]
        distinctive_phrase_matches = 0
        for index in range(len(ordered_terms) - 1):
            phrase = f"{ordered_terms[index]} {ordered_terms[index + 1]}"
            if phrase in text:
                distinctive_phrase_matches += 1
        bonus += min(0.55 * distinctive_phrase_matches, 1.65)
        return bonus

    def _media_query_bonus(self, query: str, media: Dict[str, Any]) -> float:
        query_tokens = set(self._informative_query_tokens(query))
        if not query_tokens:
            return 0.0
        media_type = str(media.get("media_type") or "")
        bonus = 0.0
        if query_tokens & _VISUAL_INTENT_TOKENS:
            if media_type == "page_visual":
                bonus += 0.45
            elif media_type in {"image", "video"}:
                bonus += 0.25
        if {"map", "layout", "building", "buildings", "parking", "facilities", "facility"} & query_tokens:
            if media_type == "page_visual":
                bonus += 0.35
            media_text = self.media_texts_by_id.get(str(media.get("id") or ""), "")
            if any(token in media_text.lower() for token in ("map", "layout", "parking", "facility", "facilities", "building", "buildings")):
                bonus += 0.20
        return bonus

    def _local_media_query_ids(self, query: str, *, top_k: int) -> List[str]:
        if top_k <= 0 or not self.media_map:
            return []
        candidate_ids = self._lexical_query_ids(
            query,
            max(top_k, self.local_media_candidate_pool),
            namespace=self.namespace_media,
        )
        if not candidate_ids:
            # A local lexical miss is not permission to rescore the complete
            # media corpus; the independent dense media lane preserves recall.
            return []
        scored: List[Tuple[str, float]] = []
        for media_id in candidate_ids:
            media = self.media_map.get(media_id)
            if not media:
                continue
            score = self._score_media_relevance(query, media)
            if score <= 0.0:
                continue
            scored.append((media_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [media_id for media_id, _score in scored[:top_k]]

    def _source_query_bonus(
        self,
        query: str,
        *,
        source_url: str = "",
        document_title: str = "",
        heading: str = "",
        text: str = "",
        mode: QueryMode | None = None,
    ) -> float:
        mode = mode or classify_query_mode(query)
        normalized = _clean_text(query).lower()
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return 0.0
        support_intents = _support_query_intents(query)
        lower_url = str(source_url or "").lower()
        lower_title = _clean_text(document_title).lower()
        lower_heading = _clean_text(heading).lower()
        lower_text = _clean_text(text).lower()
        source_blob = " ".join(part for part in (lower_url, lower_title, lower_heading, lower_text) if part)
        if not source_blob:
            return 0.0
        source_path_text = _source_url_path_text(lower_url)
        source_anchor_text = " ".join(
            part for part in (source_path_text, lower_title, lower_heading) if part
        )

        is_news = (
            "/news/" in lower_url
            or "news-archive" in lower_url
            or "/the-node/" in lower_url
            or "magazine" in lower_title
        )
        is_application = any(
            marker in lower_url
            for marker in (
                "application-submission",
                "admission-process",
                "apply.",
                "/study/undergraduate-application",
            )
        )
        is_contact = "/about/contact" in lower_url or "contact" in lower_title or "contact" in lower_heading
        is_faq = "/about/faq" in lower_url or lower_title == "faq" or "faq" in lower_heading
        is_campus = any(
            marker in lower_url
            for marker in (
                "campus-facilities",
                "educational-affairs",
                "student-resources",
            )
        ) or any(
            marker in source_blob
            for marker in (
                "campus facilities",
                "available services for students on campus",
                "support services and dedicated student facilities",
                "office of student affairs",
            )
        )
        is_institutional = any(
            marker in source_blob
            for marker in (
                "our history",
                "institutional history",
                "the university",
                "about mbzuai",
                "mohamed bin zayed university of artificial intelligence began",
            )
        ) or "/about/" in lower_url
        is_catalogue = (
            "catalogue" in lower_title
            or "catalog" in lower_title
            or "catalogue" in lower_url
            or "catalog" in lower_url
        )
        is_leadership = any(
            marker in source_blob
            for marker in (
                "office of the president",
                "office of the provost",
                "leadership",
                "board of trustees",
                "vice president and chief of staff",
                "chairman of mbzuai's board of trustees",
                "chair of mbzuai's board of trustees",
            )
        ) or "/leadership" in lower_url or "/office-of-the-president" in lower_url or "/office-of-the-provost" in lower_url

        bonus = 0.0
        institutional_relation_tokens = {
            "name",
            "named",
            "carry",
            "law",
            "legal",
            "established",
            "affiliated",
            "affiliation",
            "authority",
        }
        leadership_tokens = {
            "president",
            "provost",
            "chair",
            "chairman",
            "trustee",
            "trustees",
            "vice",
            "chief",
            "staff",
        }
        transport_tokens = {
            "parking",
            "visitor",
            "visitors",
            "shuttle",
            "transport",
            "transportation",
            "bus",
        }
        accommodation_tokens = {
            "accommodation",
            "housing",
            "stay",
            "staying",
            "residence",
            "residences",
            "parent",
            "parents",
            "family",
            "families",
        }
        amenities_tokens = {
            "amenity",
            "amenities",
            "support",
            "service",
            "services",
            "facility",
            "facilities",
            "campus",
            "onsite",
            "site",
        }
        hours_tokens = {"hour", "hours", "working", "operating", "office"}
        library_query = "library" in query_tokens

        query_anchor_tokens = set(_source_anchor_tokens(query))
        source_path_tokens = set(_source_anchor_tokens(source_path_text))
        if query_anchor_tokens and source_path_tokens:
            overlap = query_anchor_tokens & source_path_tokens
            if overlap:
                overlap_ratio = len(overlap) / float(max(1, min(len(query_anchor_tokens), len(source_path_tokens))))
                bonus += min(1.20, (0.16 * len(overlap)) + (0.70 * overlap_ratio))
        bonus += _source_phrase_bonus(query, source_anchor_text)
        bonus += _source_identity_bonus(
            query,
            source_url=source_url,
            document_title=document_title,
            heading=heading,
        )
        bonus += _support_intent_text_bonus(
            query,
            source_url=source_url,
            document_title=document_title,
            heading=heading,
            text=text,
        )

        degree_program_query = bool(
            {"master", "msc", "m.sc", "phd", "doctor", "philosophy", "bachelor", "undergraduate"} & query_tokens
        )
        if degree_program_query:
            wants_msc = bool({"master", "msc", "m.sc"} & query_tokens) or "master of science" in normalized
            wants_phd = bool({"phd", "doctor", "philosophy"} & query_tokens) or "doctor of philosophy" in normalized
            wants_undergraduate = bool({"bachelor", "undergraduate"} & query_tokens)
            is_msc_url = "/study/msc-programs/" in lower_url or "master-of-science" in lower_url
            is_phd_url = "/study/phd-programs/" in lower_url or "doctor-of-philosophy" in lower_url or "/phd-" in lower_url
            is_undergraduate_url = "/undergraduate" in lower_url or "bachelor-of-science" in lower_url
            if wants_msc:
                bonus += 0.42 if is_msc_url else (-0.42 if is_phd_url or is_undergraduate_url else 0.0)
            if wants_phd:
                bonus += 0.42 if is_phd_url else (-0.42 if is_msc_url or is_undergraduate_url else 0.0)
            if wants_undergraduate:
                bonus += 0.42 if is_undergraduate_url else (-0.36 if is_msc_url or is_phd_url else 0.0)

        if {"faculty", "professor", "dr"} & query_tokens and "person_statement" not in support_intents:
            is_faculty_profile = "/study/faculty/" in lower_url
            if is_faculty_profile:
                bonus += 0.46
            if is_news and not is_faculty_profile:
                bonus -= 0.72

        if query_tokens & institutional_relation_tokens:
            if is_news:
                bonus -= 0.95
            if is_institutional or is_faq or is_catalogue:
                bonus += 0.42
            if is_catalogue:
                years = [
                    int(match)
                    for match in re.findall(r"\b20\d{2}\b", " ".join([lower_url, lower_title]))
                ]
                latest_year = max(years) if years else 0
                bonus += 0.72
                if latest_year >= 2024:
                    bonus += 0.68
                elif 0 < latest_year <= 2022:
                    bonus -= 0.34
            if "factsheet" in source_blob or "fact sheet" in source_blob:
                bonus -= 0.28

        if query_tokens & leadership_tokens:
            if is_news and not (support_intents & {"person_statement", "president_background", "program_expansion_history"}):
                bonus -= 0.95
            if is_leadership or is_institutional or is_catalogue:
                bonus += 0.56

        if query_tokens & transport_tokens:
            if is_news:
                bonus -= 0.85
            if is_contact or is_campus or is_faq:
                bonus += 0.30

        if query_tokens & accommodation_tokens:
            if is_news:
                bonus -= 0.75
            if is_campus or is_faq or is_application:
                bonus += 0.26

        if query_tokens & amenities_tokens:
            if is_news:
                bonus -= 0.90
            if is_campus or is_contact or is_catalogue:
                bonus += 0.44
            if is_application and not (query_tokens & accommodation_tokens):
                bonus -= 0.42

        if query_tokens & hours_tokens:
            if is_news:
                bonus -= 0.70
            if is_contact or is_faq:
                bonus += 0.22
            if is_application and not (query_tokens & (transport_tokens | accommodation_tokens)):
                bonus -= 0.24

        if library_query:
            library_detail_hit = any(
                marker in source_blob
                for marker in (
                    "library hours",
                    "hours of operation",
                    "library contact",
                    "libraryservices",
                    "library.mbzuai",
                )
            )
            if library_detail_hit:
                bonus += 1.85
            elif "library" in source_blob:
                bonus += 0.28
            else:
                bonus -= 0.72
            if any(marker in lower_url for marker in ("/event/", "/news/", "/alumni", "/sitemap")):
                bonus -= 1.05
            if "emergency contact numbers" in source_blob or "mbzuai management.contact number" in source_blob:
                bonus -= 1.15

        if mode == QueryMode.SYNTHESIS:
            if is_news and not (support_intents & {"person_statement", "president_background", "program_expansion_history"}):
                bonus -= 0.72
            if is_contact or is_campus or is_faq or is_catalogue:
                bonus += 0.22

        return bonus

    def _effective_answer_subtype(
        self,
        answer: Dict[str, Any],
        *,
        combined_text: str,
        context_text: str,
    ) -> str:
        answer_type = str(answer.get("answer_type") or "")
        answer_subtype = str(answer.get("answer_subtype") or "")
        lower_text = _clean_text(combined_text).lower()
        lower_context = _clean_text(context_text).lower()
        if answer_type == "hours":
            support_markers = (
                "technical support",
                "helpdesk",
                "it team",
                "screening exam",
                "host organization",
                "internship",
                "support",
            )
            if any(marker in lower_context for marker in support_markers):
                return "support_hours"
            if any(
                marker in lower_text or marker in lower_context
                for marker in ("official working hours", "official workings hours", "working hours", "operating hours", "office hours")
            ):
                return "operational_hours"
            if any(marker in lower_text or marker in lower_context for marker in _EVENT_SCHEDULE_TOKENS):
                return "event_schedule"
            if (
                "hours of operation" in lower_text
                or "hours of operation" in lower_context
                or any(marker in lower_text or marker in lower_context for marker in _FACILITY_HOURS_TOKENS)
            ):
                return "facility_hours"
            return answer_subtype or "generic_hours"
        if answer_type == "service_availability":
            normalized = _service_availability_subtype(" ".join(part for part in (answer_subtype, combined_text, context_text) if part))
            if normalized:
                return normalized
        return answer_subtype

    def _role_holder_currentness_bonus(
        self,
        query: str,
        answer: Dict[str, Any],
        *,
        combined_text: str,
        context_text: str,
        effective_subtype: str,
    ) -> float:
        if str(answer.get("answer_type") or "") != "role_holder":
            return 0.0
        lower_context = _clean_text(context_text).lower()
        lower_combined = _clean_text(combined_text).lower()
        lower_title = _clean_text(answer.get("document_title") or "").lower()
        lower_url = str(answer.get("source_url") or "").lower()
        lower_source = " ".join(part for part in (lower_title, lower_url, lower_context) if part)
        bonus = 0.0

        is_official_leadership = any(
            marker in lower_source
            for marker in (
                "/about/leadership",
                "/office-of-the-president",
                "/office-of-the-provost",
                "office of the president",
                "office of the provost",
                "leadership and governance",
                "mbzuai leadership",
            )
        )
        is_catalogue = "catalogue" in lower_title or "catalog" in lower_title
        if is_official_leadership:
            bonus += 0.62
        elif is_catalogue:
            bonus -= 0.30

        years = [int(value) for value in re.findall(r"\b(20\d{2})\b", lower_source)]
        if years:
            newest_year = max(years)
            if newest_year >= 2025:
                bonus += 0.42
            elif newest_year == 2024:
                bonus += 0.30
            elif newest_year <= 2022:
                bonus -= 0.28

        current_markers = {
            "president": ("is the president of", "president and university professor"),
            "provost": ("is the provost of", "provost and professor", "appointed as the provost", "appointed as provost"),
            "board_chair": ("chairman of mbzuai's board of trustees", "chair of mbzuai's board of trustees"),
            "vice_president_chief_of_staff": ("vice president and chief of staff",),
        }
        if any(marker in lower_combined or marker in lower_context for marker in current_markers.get(effective_subtype, ())):
            bonus += 0.24

        if effective_subtype == "provost":
            if "associate provost" in lower_context or "associate provost" in lower_combined:
                bonus -= 1.40
            if "acting provost" in lower_context or "acting provost" in lower_combined:
                bonus -= 0.08
        if effective_subtype == "board_chair":
            if "founding chair" in lower_context or "founding chair" in lower_combined:
                bonus -= 0.92

        requested_roles = set(_query_intent(query).requested_role_subtypes)
        if requested_roles and effective_subtype in requested_roles:
            bonus += 0.10
        return bonus

    def _answer_subtype_bonus(
        self,
        query: str,
        answer: Dict[str, Any],
        *,
        effective_subtype: str | None = None,
    ) -> float:
        answer_type = str(answer.get("answer_type") or "")
        answer_subtype = str(effective_subtype or answer.get("answer_subtype") or "")
        text = _clean_text(answer.get("text") or answer.get("value") or "").lower()
        query_tokens = set(_tokenize(query))
        bonus = 0.0
        if answer_type in {"email", "phone", "website"}:
            bonus += 0.18
            if answer_subtype in _GENERIC_CONTACT_SUBTYPES:
                bonus += 0.18
            query_scope_tokens = set(_tokenize(query))
            scope_tokens = _contact_scope_tokens(answer, effective_subtype=answer_subtype)
            narrow_scope_tokens = scope_tokens & _CONTACT_NARROW_SCOPE_TOKENS
            if narrow_scope_tokens:
                matched_scope_tokens = narrow_scope_tokens & query_scope_tokens
                if matched_scope_tokens:
                    bonus += min(0.10 * len(matched_scope_tokens), 0.24)
                else:
                    bonus -= min(0.42 + (0.14 * len(narrow_scope_tokens)), 1.10)
        elif answer_type == "hours":
            support_hours_query = bool(query_tokens & {"support", "screening", "exam", "technical", "helpdesk", "it", "internship", "host", "organization"})
            if answer_subtype == "operational_hours":
                bonus += 0.56
            elif answer_subtype == "support_hours":
                bonus += 0.08 if query_tokens & {"support", "it", "helpdesk", "screening", "exam", "technical"} else -0.72
            elif answer_subtype == "facility_hours":
                bonus += 0.10 if query_tokens & _FACILITY_HOURS_TOKENS else -0.18
            elif answer_subtype == "event_schedule":
                bonus += 0.08 if query_tokens & _EVENT_SCHEDULE_TOKENS else -1.05
            if support_hours_query:
                if answer_subtype == "support_hours":
                    bonus += 0.82
                elif answer_subtype == "operational_hours":
                    bonus -= 1.08
                elif answer_subtype == "event_schedule":
                    bonus -= 1.32
                elif answer_subtype == "facility_hours":
                    bonus -= 0.54
            if {"official", "working", "operating", "weekday", "weekdays"} & query_tokens:
                if answer_subtype == "operational_hours":
                    bonus += 0.34
                elif answer_subtype in {"event_schedule", "support_hours"}:
                    bonus -= 0.45
            if answer_subtype == "operational_hours" and any(
                day in text for day in ("monday", "tuesday", "wednesday", "thursday", "friday")
            ):
                bonus += 0.18
            if (
                {"official", "working", "operating"} & query_tokens
                and answer_subtype != "operational_hours"
                and not any(term in text for term in ("working hours", "operating hours", "official working", "official workings"))
            ):
                bonus -= 0.38
            if not (query_tokens & _EVENT_SCHEDULE_TOKENS) and any(token in text for token in _EVENT_SCHEDULE_TOKENS):
                bonus -= 0.35
        elif answer_type == "date":
            if answer_subtype == "deadline":
                bonus += 0.34 if query_tokens & {"deadline", "due"} else 0.16
            elif answer_subtype == "start_date":
                bonus += 0.34 if query_tokens & {"start", "semester", "orientation", "begin"} else 0.14
            elif answer_subtype == "event_date":
                bonus += 0.10 if query_tokens & _EVENT_SCHEDULE_TOKENS else -0.28
        elif answer_type == "location":
            bonus += 0.20
            if answer_subtype == "emirate" and "emirate" in query_tokens:
                bonus += 0.24
            elif answer_subtype == "location" and query_tokens & {"city", "located", "location", "based"}:
                bonus += 0.16
        elif answer_type == "service_availability":
            requested_subtypes = _requested_service_availability_subtypes(query)
            service_bonus = 0.10
            if requested_subtypes and answer_subtype not in requested_subtypes:
                service_bonus -= 1.10
            if answer_subtype == "transport":
                service_bonus += 0.48 if query_tokens & {"shuttle", "transport", "transportation", "bus"} else -0.30
                if "student" in query_tokens and "student" in text:
                    service_bonus += 0.18
                if "student" in query_tokens and {"visitor", "visitors"} & set(_tokenize(text)):
                    service_bonus -= 0.32
            elif answer_subtype == "parking":
                service_bonus += 0.48 if query_tokens & {"park", "parked", "parking", "vehicle", "vehicles", "visitor", "visitors"} else -0.24
                if "visitor" not in query_tokens and {"visitor", "visitors"} & set(_tokenize(text)):
                    service_bonus -= 0.20
            elif answer_subtype == "accommodation":
                service_bonus += 0.48 if query_tokens & {"accommodation", "housing", "stay", "parent", "parents", "family", "families"} else -0.18
                if not (query_tokens & {"parent", "parents", "family", "families"}) and any(
                    token in text for token in ("parent", "parents", "family", "families", "hotels", "airbnbs", "airbnb")
                ):
                    service_bonus -= 0.60
                if query_tokens & {"student", "housing", "accommodation"} and any(
                    token in text for token in ("student accommodation", "provided accommodation", "university accommodation", "residences")
                ):
                    service_bonus += 0.22
            elif answer_subtype == "amenities":
                service_bonus += 0.44 if query_tokens & {"amenity", "amenities", "service", "services", "facility", "facilities", "support"} else -0.16
                if any(
                    token in text
                    for token in (
                        "support services",
                        "student facilities",
                        "health services",
                        "prayer rooms",
                        "dining facilities",
                        "student lounges",
                    )
                ):
                    service_bonus += 0.18
            if text.startswith("yes"):
                service_bonus += 0.14
            elif text.startswith("no"):
                service_bonus += 0.10
            if "?" in text:
                service_bonus += 0.18
            bonus += service_bonus
        elif answer_type == "named_after":
            bonus += 0.28
        elif answer_type == "affiliation":
            bonus += 0.24
        elif answer_type == "legal_basis":
            bonus += 0.24
        elif answer_type == "role_holder":
            requested_roles = set(_requested_role_subtypes(query))
            if requested_roles:
                if answer_subtype in requested_roles:
                    bonus += 0.62
                else:
                    bonus -= 1.40
            if answer_subtype == "board_chair" and {"board", "trustee", "trustees"} & query_tokens:
                bonus += 0.34
            if answer_subtype == "president" and "president" in query_tokens:
                bonus += 0.24
            if answer_subtype == "provost" and "provost" in query_tokens:
                bonus += 0.24
            if answer_subtype == "vice_president_chief_of_staff" and {"chief", "staff"} & query_tokens:
                bonus += 0.30
        return bonus

    def _score_answer_record(self, query: str, answer: Dict[str, Any]) -> float:
        answer_id = str(answer.get("id") or "")
        answer_type = str(answer.get("answer_type") or "")
        query_intent = _query_intent(query)
        query_tokens = set(_tokenize(query))
        desired_types = set(query_intent.answer_types)
        if desired_types and answer_type not in desired_types:
            return -1.0
        text = _clean_text(answer.get("text") or "")
        value = _clean_text(answer.get("value") or "")
        if not text and not value:
            return -1.0
        combined_text = " ".join(part for part in (value, text) if part)
        context_text = _clean_text(self.answer_context_texts_by_id.get(answer_id) or combined_text)
        subject_text = _clean_text(self.answer_subjects_by_id.get(answer_id) or answer.get("subject_text") or "")
        effective_subtype = self._effective_answer_subtype(
            answer,
            combined_text=combined_text,
            context_text=context_text,
        )
        if answer_type == "service_availability":
            requested_service_subtypes = _requested_service_availability_subtypes(query)
            if requested_service_subtypes and effective_subtype not in requested_service_subtypes:
                return -1.0
        lower_combined = combined_text.lower()
        lower_context = context_text.lower()
        requested_roles = set(query_intent.requested_role_subtypes)
        if answer_type == "role_holder" and requested_roles and effective_subtype not in requested_roles:
            return -1.0
        score = self._score_text_match(query, combined_text)
        if context_text and context_text != combined_text:
            score += self._score_text_match(query, context_text) * 0.28
        if value:
            score += self._score_text_match(query, value) * 0.35
        score += _lookup_signal_bonus(query, combined_text)
        score += self._fact_query_bonus(query, combined_text) * 0.55
        score += self._source_query_bonus(
            query,
            source_url=str(answer.get("source_url") or ""),
            document_title=str(answer.get("document_title") or ""),
            heading=str(answer.get("heading") or ""),
            text=context_text,
            mode=classify_query_mode(query),
        )
        if str(answer.get("source_record_type") or "") == "assertion":
            score += 0.26
            if answer_type in {"role_holder", "affiliation", "legal_basis", "named_after"}:
                score += 0.14
        score += self._answer_subtype_bonus(query, answer, effective_subtype=effective_subtype)
        score += self._role_holder_currentness_bonus(
            query,
            answer,
            combined_text=combined_text,
            context_text=context_text,
            effective_subtype=effective_subtype,
        )
        score += float(answer.get("confidence") or 0.0) * 0.55

        subject_alignment = _subject_alignment_score(
            query=query,
            subject_text=subject_text,
            context_text=context_text,
        )
        if answer_type in {"role_holder", "named_after", "affiliation", "legal_basis", "location", "email", "phone", "website"}:
            if subject_alignment > 0.0:
                score += 0.48 * subject_alignment
            elif query_intent.subject_tokens and not _is_generic_subject_reference(subject_text):
                score -= 0.80

        if answer_type == "location" and "mbzuai" in query_tokens and subject_text:
            subject_tokens = set(_tokenize(subject_text))
            query_subject_tokens = set(query_intent.subject_tokens)
            subordinate_overlap = subject_tokens & _SUBORDINATE_LOCATION_SUBJECT_TOKENS
            if subordinate_overlap and not (query_subject_tokens & subordinate_overlap):
                score -= 0.85
            if "emirate" in query_tokens:
                if effective_subtype == "emirate" or "abu dhabi" in lower_context or "abu dhabi" in lower_combined:
                    score += 0.32
                else:
                    score -= 0.64

        qualifiers = set(_tokenize(" ".join(answer.get("qualifiers") or [])))
        lookup_profile = _lookup_query_profile(query)
        requested_qualifiers = _expanded_lookup_qualifiers(
            set(lookup_profile.focus_tokens) & _LOOKUP_CONTEXT_QUALIFIER_TOKENS
        )
        if requested_qualifiers:
            if qualifiers & requested_qualifiers:
                score += 0.18
            elif qualifiers:
                score -= 0.12
        elif qualifiers and answer_type in {"email", "phone", "website"}:
            score -= min(0.18, 0.06 * len(qualifiers))
        if answer_type in {"email", "phone", "website"} and qualifiers:
            subject_like_tokens = set(query_intent.subject_tokens)
            if not subject_like_tokens:
                subject_like_tokens = set(
                    token
                    for token in _tokenize(query)
                    if token not in _QUERY_STOPWORDS and token not in _LOOKUP_ATTRIBUTE_TOKENS
                )
            qualifier_overlap = len(qualifiers & subject_like_tokens) / float(len(subject_like_tokens)) if subject_like_tokens else 0.0
            if qualifier_overlap > 0.0:
                score += 0.28 * qualifier_overlap
            elif subject_like_tokens:
                score -= 0.42
        if answer_type == "email" and value:
            local_part_tokens = _email_local_part_tokens(value)
            subject_like_tokens = set(_tokenize(" ".join(query_intent.subject_tokens)))
            if not subject_like_tokens:
                subject_like_tokens = {
                    token
                    for token in _tokenize(query)
                    if token not in _QUERY_STOPWORDS and token not in _LOOKUP_ATTRIBUTE_TOKENS
                }
            if local_part_tokens and subject_like_tokens:
                local_overlap = len(local_part_tokens & subject_like_tokens) / float(len(subject_like_tokens))
                if local_overlap > 0.0:
                    score += 0.34 * local_overlap
                if {"admission", "admissions"} & subject_like_tokens:
                    if "admission" in local_part_tokens or "admissions" in local_part_tokens:
                        score += 0.42
                    else:
                        score -= 0.78
                if {"undergraduate", "undergrad", "ug"} & subject_like_tokens:
                    if {"ug", "undergraduate"} & local_part_tokens:
                        score += 0.18
                    else:
                        score -= 0.16
                if {"registrar"} & subject_like_tokens and "registrar" not in local_part_tokens:
                    score -= 0.26

        named_tokens = set(_named_query_tokens(query))
        context_tokens = set(_tokenize(context_text))
        if named_tokens:
            context_overlap = len(named_tokens & context_tokens) / float(len(named_tokens))
            if context_overlap:
                score += context_overlap * 0.18
            missing_ratio = _missing_named_token_ratio(query, context_text)
            if answer_type in {"email", "phone", "website", "hours", "date"} and missing_ratio >= 0.5:
                score -= 0.95 * missing_ratio
            elif answer_type in {"location", "affiliation", "legal_basis", "role_holder"} and missing_ratio >= 1.0:
                score -= 0.45
            if subject_text and answer_type in {"named_after", "location", "affiliation", "legal_basis", "role_holder"}:
                subject_tokens = set(_tokenize(subject_text))
                subject_overlap = len(named_tokens & subject_tokens) / float(len(named_tokens)) if subject_tokens else 0.0
                if subject_overlap:
                    score += subject_overlap * 0.34
                elif not _is_generic_subject_reference(subject_text):
                    score -= 1.10

        if answer_type == "legal_basis":
            founding_query = bool({"law", "legal", "created", "established", "founded"} & query_tokens)
            if founding_query:
                if effective_subtype in _FOUNDING_LEGAL_BASIS_SUBTYPES:
                    score += 1.05
                elif any(token in effective_subtype for token in _NON_FOUNDING_LEGAL_BASIS_SUBTYPE_TOKENS):
                    score -= 1.45
                if any(term in lower_context or term in lower_combined for term in ("law no.", "law no ", "established under law", "established by law", "decree")):
                    score += 0.72
                elif "law" not in lower_combined and "law" not in lower_context:
                    score -= 1.10

        if answer_type == "affiliation":
            institutional_query = bool({"authority", "affiliation", "affiliated", "institutional"} & query_tokens)
            if institutional_query:
                if effective_subtype in _INSTITUTIONAL_AFFILIATION_SUBTYPES:
                    score += 0.58
                elif any(token in effective_subtype for token in _NON_INSTITUTIONAL_AFFILIATION_SUBTYPE_TOKENS):
                    score -= 1.05
                if any(term in lower_context or term in lower_combined for term in ("executive council", "affiliated to", "affiliated with")):
                    score += 0.32
                if "executive council" in lower_context or "executive council" in lower_combined:
                    score += 0.42

        if answer_type == "hours" and lookup_profile.is_exact_lookup and "hours" in lookup_profile.answer_types:
            support_hours_query = bool(set(_tokenize(query)) & {"support", "screening", "internship", "organization", "event", "conference", "technical", "it", "helpdesk", "exam"})
            official_hours_query = bool(set(_tokenize(query)) & {"official", "working", "operating", "weekday", "weekdays"})
            if "library" in query_tokens:
                if "library" in lower_context or "library" in lower_combined:
                    score += 0.72
                else:
                    return -1.0
            if support_hours_query:
                support_markers = ("technical support", "screening exam", "online screening exam", "internship", "host organization", "it team", "helpdesk", "support hours")
                if any(term in lower_context or term in lower_combined for term in support_markers):
                    score += 0.36
                else:
                    score -= 0.56
            if official_hours_query and not support_hours_query:
                if effective_subtype == "support_hours":
                    score -= 1.05
                elif effective_subtype == "facility_hours":
                    score -= 0.42
            if (
                not support_hours_query
                and any(term in lower_context for term in ("technical support", "screening exam", "internship", "host organization", "it team", "helpdesk"))
            ):
                score -= 1.25
            if effective_subtype == "support_hours" and not (
                support_hours_query
            ):
                score -= 0.65
        return score

    def _local_answer_query_ids(self, query: str, *, top_k: int) -> List[str]:
        if top_k <= 0 or not self.answer_map:
            return []
        query_intent = _query_intent(query)
        desired_types = set(query_intent.answer_types)
        if not desired_types:
            return []
        informative_tokens = list(dict.fromkeys(self._informative_query_tokens(query)))
        candidate_hits: Dict[str, int] = defaultdict(int)
        requested_service_subtypes = _requested_service_availability_subtypes(query)
        requested_roles = set(query_intent.requested_role_subtypes)
        for token in informative_tokens:
            for answer_id in self.answer_token_index.get(token, [])[: self.local_index_max_postings_per_token]:
                candidate_hits[answer_id] += 1
        for answer_type in desired_types:
            seeded = 0
            scoped_answer_ids: Sequence[str]
            if answer_type == "role_holder" and requested_roles:
                scoped_answer_ids = [
                    answer_id
                    for role in requested_roles
                    for answer_id in (self.answer_ids_by_subtype.get((answer_type, role)) or [])
                ]
            else:
                scoped_answer_ids = self.answer_ids_by_type.get(answer_type) or []
            for rank, answer_id in enumerate(scoped_answer_ids):
                if rank >= self.local_answer_candidate_pool:
                    break
                answer = self.answer_map.get(str(answer_id)) or {}
                if answer_type == "service_availability" and requested_service_subtypes:
                    subtype = str(answer.get("answer_subtype") or "")
                    if subtype not in requested_service_subtypes:
                        continue
                if answer_type == "role_holder" and requested_roles:
                    subtype = str(answer.get("answer_subtype") or "")
                    if subtype not in requested_roles:
                        continue
                candidate_hits[answer_id] += max(1, self.local_answer_candidate_pool - rank)
                seeded += 1
                if answer_type == "service_availability" and requested_service_subtypes and seeded >= self.local_answer_candidate_pool:
                    break
        if not candidate_hits:
            return []
        candidate_cap = max(top_k * 8, self.local_answer_candidate_pool, 48)
        ordered_candidates = sorted(
            candidate_hits.items(),
            key=lambda item: (-item[1], item[0]),
        )[:candidate_cap]
        scored: List[Tuple[str, float]] = []
        for answer_id, hit_count in ordered_candidates:
            answer = self.answer_map.get(answer_id)
            if not answer:
                continue
            if str(answer.get("answer_type") or "") == "service_availability" and requested_service_subtypes:
                subtype = str(answer.get("answer_subtype") or "")
                if subtype not in requested_service_subtypes:
                    continue
            score = self._score_answer_record(query, answer)
            if score <= 0.0:
                continue
            score += min(max(hit_count - 1, 0) * 0.03, 0.18)
            scored.append((answer_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [answer_id for answer_id, _score in scored[:top_k]]

    def _rank_answer_ids(
        self,
        query: str,
        answer_ids: Sequence[str],
        *,
        top_k: int,
    ) -> List[str]:
        scored: List[Tuple[str, float]] = []
        seen_answer_ids = set()
        for answer_id in answer_ids:
            answer_id = str(answer_id)
            if not answer_id or answer_id in seen_answer_ids:
                continue
            seen_answer_ids.add(answer_id)
            answer = self.answer_map.get(answer_id)
            if not answer:
                continue
            score = self._score_answer_record(query, answer)
            if score > 0.0:
                scored.append((answer_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [answer_id for answer_id, _score in scored[:top_k]]

    def _bounded_answer_candidates(
        self,
        query: str,
        *,
        answer_type: str,
        preferred_answer_ids: Sequence[str],
        candidate_limit: int,
        answer_subtype: str = "",
    ) -> List[str]:
        """Build a query-relevant answer pool without scanning a whole answer class.

        ``answer_ids_by_type`` can contain thousands of records (URLs are the
        largest class in the MBZUAI corpus).  The final answer selector used to
        rescore that complete list whenever a query requested more than one
        answer type.  Besides duplicating the bounded local-answer lane, that
        made tail latency grow linearly with the corpus.

        Keep already retrieved answers, add matches from the in-memory token
        index, and reserve a small confidence-ordered fallback.  Scoring still
        makes the final decision; this helper only places a hard ceiling on the
        number of records that reach the expensive scorer.
        """

        limit = max(1, int(candidate_limit or 1))
        selected: List[str] = []
        seen: set[str] = set()

        def _matches_scope(answer_id: str) -> bool:
            answer = self.answer_map.get(str(answer_id)) or {}
            if str(answer.get("answer_type") or "") != answer_type:
                return False
            if answer_subtype and str(answer.get("answer_subtype") or "") != answer_subtype:
                return False
            return True

        def _append(answer_id: str) -> None:
            normalized_id = str(answer_id)
            if (
                normalized_id
                and normalized_id not in seen
                and len(selected) < limit
                and _matches_scope(normalized_id)
            ):
                selected.append(normalized_id)
                seen.add(normalized_id)

        # Preserve the candidates already surfaced by dense, sparse, and local
        # retrieval.  They are the strongest corpus-level relevance signal.
        for answer_id in preferred_answer_ids:
            _append(str(answer_id))
            if len(selected) >= limit:
                return selected

        candidate_hits: Dict[str, int] = defaultdict(int)
        for token in dict.fromkeys(self._informative_query_tokens(query)):
            for answer_id in self.answer_token_index.get(token, [])[
                : self.local_index_max_postings_per_token
            ]:
                normalized_id = str(answer_id)
                if normalized_id not in seen and _matches_scope(normalized_id):
                    candidate_hits[normalized_id] += 1
        for answer_id, _hit_count in sorted(
            candidate_hits.items(),
            key=lambda item: (-item[1], item[0]),
        ):
            _append(answer_id)
            if len(selected) >= limit:
                return selected

        scoped_answer_ids = (
            self.answer_ids_by_subtype.get((answer_type, answer_subtype))
            if answer_subtype
            else self.answer_ids_by_type.get(answer_type)
        ) or []
        for answer_id in scoped_answer_ids:
            _append(str(answer_id))
            if len(selected) >= limit:
                break
        return selected

    def _select_answer_ids_for_query(
        self,
        query: str,
        ranked_answer_ids: Sequence[str],
        *,
        top_k: int,
    ) -> List[str]:
        query_intent = _query_intent(query)
        def _answer_signature(answer_id: str) -> Tuple[str, str, str]:
            answer = self.answer_map.get(str(answer_id)) or {}
            return (
                str(answer.get("answer_type") or ""),
                str(answer.get("answer_subtype") or ""),
                _clean_text(answer.get("value") or answer.get("text") or "").lower(),
            )

        requested_roles = list(query_intent.requested_role_subtypes)
        requested_types = list(dict.fromkeys(query_intent.answer_types))
        if not requested_roles:
            if len(requested_types) > 1:
                selected: List[str] = []
                seen_signatures: set[Tuple[str, str, str]] = set()
                per_type_candidate_limit = max(
                    8,
                    self.local_answer_candidate_pool // len(requested_types),
                )
                for answer_type in requested_types:
                    candidate_ids = self._bounded_answer_candidates(
                        query,
                        answer_type=answer_type,
                        preferred_answer_ids=ranked_answer_ids,
                        candidate_limit=per_type_candidate_limit,
                    )
                    scored: List[Tuple[str, float]] = []
                    for answer_id in candidate_ids:
                        answer = self.answer_map.get(str(answer_id)) or {}
                        score = self._score_answer_record(query, answer)
                        if score > 0.0:
                            scored.append((str(answer_id), score))
                    scored.sort(key=lambda item: item[1], reverse=True)
                    for answer_id, _score in scored:
                        signature = _answer_signature(answer_id)
                        if signature in seen_signatures:
                            continue
                        selected.append(answer_id)
                        seen_signatures.add(signature)
                        break
                for answer_id in ranked_answer_ids:
                    answer_id = str(answer_id)
                    signature = _answer_signature(answer_id)
                    if not answer_id or answer_id in selected or signature in seen_signatures:
                        continue
                    selected.append(answer_id)
                    seen_signatures.add(signature)
                    if len(selected) >= top_k:
                        break
                return selected[:top_k]
            selected: List[str] = []
            seen_signatures: set[Tuple[str, str, str]] = set()
            for answer_id in ranked_answer_ids:
                answer_id = str(answer_id)
                signature = _answer_signature(answer_id)
                if answer_id and signature not in seen_signatures:
                    selected.append(answer_id)
                    seen_signatures.add(signature)
                if len(selected) >= top_k:
                    break
            return selected
        selected: List[str] = []
        seen_signatures: set[Tuple[str, str, str]] = set()
        role_candidate_limit = max(
            8,
            self.local_answer_candidate_pool // max(len(requested_roles), 1),
        )
        for slot in query_intent.slots:
            if slot.answer_type != "role_holder" or not slot.qualifier:
                continue
            slot_query = _slot_query_text(query, slot)
            candidate_ids = self._bounded_answer_candidates(
                slot_query,
                answer_type=slot.answer_type,
                answer_subtype=slot.qualifier,
                preferred_answer_ids=ranked_answer_ids,
                candidate_limit=role_candidate_limit,
            )
            scored: List[Tuple[str, float]] = []
            for answer_id in candidate_ids:
                answer = self.answer_map.get(str(answer_id)) or {}
                score = self._score_answer_record(slot_query, answer)
                if score > 0.0:
                    scored.append((str(answer_id), score))
            scored.sort(key=lambda item: item[1], reverse=True)
            for answer_id, _score in scored:
                signature = _answer_signature(answer_id)
                if signature in seen_signatures:
                    continue
                selected.append(answer_id)
                seen_signatures.add(signature)
                break
        for answer_id in ranked_answer_ids:
            answer_id = str(answer_id)
            signature = _answer_signature(answer_id)
            if not answer_id or answer_id in selected or signature in seen_signatures:
                continue
            selected.append(answer_id)
            seen_signatures.add(signature)
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    def _rank_answer_anchor_chunk_ids(
        self,
        query: str,
        answer_ids: Sequence[str],
        *,
        top_k: int,
    ) -> List[str]:
        ranked_answer_ids = self._rank_answer_ids(query, answer_ids, top_k=max(top_k * 2, top_k))
        if not ranked_answer_ids:
            return []
        scored: List[Tuple[str, float]] = []
        seen_chunk_ids = set()
        for answer_id in ranked_answer_ids:
            answer = self.answer_map.get(str(answer_id))
            if not answer:
                continue
            answer_score = self._score_answer_record(query, answer)
            linked_chunk_ids = [str(value) for value in (answer.get("linked_chunk_ids") or []) if str(value)]
            if not linked_chunk_ids:
                for fact_id in answer.get("linked_fact_ids") or []:
                    linked_chunk_ids.extend(
                        str(chunk_id)
                        for chunk_id in (self.fact_map.get(str(fact_id)) or {}).get("linked_chunk_ids") or []
                        if str(chunk_id)
                    )
            for chunk_id in dict.fromkeys(linked_chunk_ids):
                if chunk_id in seen_chunk_ids or chunk_id not in self.chunk_map:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunk = self.chunk_map.get(chunk_id) or {}
                chunk_text = _clean_text(chunk.get("dense_text") or chunk.get("text") or "")
                chunk_heading = _clean_text(
                    " ".join(
                        str(value)
                        for value in (
                            chunk.get("heading") or "",
                            " > ".join(chunk.get("section_path") or []),
                            chunk.get("document_title") or "",
                        )
                    )
                )
                chunk_score = self._score_text_match(query, chunk_text) + _lookup_signal_bonus(query, chunk_text)
                heading_bonus = (self._score_text_match(query, chunk_heading) * 0.25) + (_lookup_signal_bonus(query, chunk_heading) * 0.15)
                scored.append((chunk_id, answer_score + (chunk_score * 0.95) + heading_bonus))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, score in scored[:top_k] if score > 0.0]

    def _local_fact_query_ids(self, query: str, *, top_k: int) -> List[str]:
        if top_k <= 0 or not self.fact_map:
            return []
        informative_tokens = list(dict.fromkeys(self._informative_query_tokens(query)))
        if not informative_tokens:
            return []
        lookup_profile = _lookup_query_profile(query)
        candidate_hits: Dict[str, int] = defaultdict(int)
        candidate_rarity: Dict[str, float] = defaultdict(float)
        for token in informative_tokens:
            postings = self.fact_token_index.get(token, [])
            if not postings:
                continue
            rarity_weight = 1.0 / max(1.0, float(len(postings)) ** 0.5)
            for fact_id in postings[: self.local_index_max_postings_per_token]:
                candidate_hits[fact_id] += 1
                candidate_rarity[fact_id] += rarity_weight
        if not candidate_hits:
            return []
        # Full-corpus fact rescoring dominated concurrent request latency.  The
        # inverted index already supplies the lexical candidate generator, so
        # apply the expensive semantic/source bonuses only to a bounded set.
        # Multi-token and rare-token matches are kept first; dense fact search
        # remains an independent recall lane for candidates outside this pool.
        candidate_cap = max(256, top_k * 24)
        candidate_ids = [
            fact_id
            for fact_id, _hit_count in sorted(
                candidate_hits.items(),
                key=lambda item: (
                    -item[1],
                    -candidate_rarity.get(item[0], 0.0),
                    item[0],
                ),
            )[:candidate_cap]
        ]
        informative_token_set = set(informative_tokens)
        scored: List[Tuple[str, float]] = []
        for fact_id in candidate_ids:
            fact_tokens = set(self.fact_tokens_by_id.get(fact_id) or [])
            if not fact_tokens:
                continue
            overlap = len(fact_tokens & informative_token_set) / float(len(informative_token_set))
            fact = self.fact_map[fact_id]
            fact_text = _clean_text(fact.get("text") or fact.get("dense_text") or "")
            fact_lookup_text = _fact_lookup_text(fact) or fact_text
            lookup_bonus = _lookup_signal_bonus(query, fact_text)
            context_lookup_bonus = _lookup_signal_bonus(query, fact_lookup_text)
            contextual_contact_match = _contextual_contact_fact_match(query, fact_text, fact_lookup_text)
            contextual_family_match = _contextual_family_accommodation_match(query, fact_lookup_text)
            if lookup_profile.is_exact_lookup:
                if not any(_text_matches_answer_type(fact_text, answer_type) for answer_type in lookup_profile.answer_types):
                    continue
                if max(lookup_bonus, context_lookup_bonus) <= 0.0 and not contextual_contact_match and not contextual_family_match:
                    continue
            scored.append(
                (
                    fact_id,
                    overlap
                    + self._fact_query_bonus(query, fact_text)
                    + max(lookup_bonus, context_lookup_bonus)
                    + (0.85 if contextual_contact_match else 0.0)
                    + (1.15 if contextual_family_match else 0.0)
                    + self._source_query_bonus(
                        query,
                        source_url=str(fact.get("source_url") or ""),
                        document_title=str(fact.get("document_title") or ""),
                        heading=str(fact.get("heading") or fact.get("section_heading") or ""),
                        text=fact_lookup_text,
                    ),
                )
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        return [fact_id for fact_id, score in scored[:top_k] if score > 0.0]

    def _fact_query_bonus(self, query: str, fact_text: str) -> float:
        text = _clean_text(fact_text)
        if not text:
            return 0.0
        query_tokens = set(_tokenize(query))
        normalized_query = _clean_text(query).lower()
        informative_tokens = [token for token in self._informative_query_tokens(query) if len(token) >= 4]
        bonus = _phrase_match_bonus(informative_tokens, text)
        bonus += _lookup_signal_bonus(query, text)
        token_count = len(text.split())
        if 6 <= token_count <= 24:
            bonus += 0.12
        elif token_count >= 40:
            bonus -= 0.10

        lower_text = text.lower()
        if _query_starts_with(query, ("where ",)):
            if any(phrase in lower_text for phrase in ("located in", "located at", "based in", "based at")):
                bonus += 0.12
            location_markers = ("city", "campus", "airport", "district", "street", "road", "avenue", "boulevard")
            marker_count = sum(1 for marker in location_markers if marker in lower_text)
            bonus += min(0.04 * marker_count, 0.16)
        if "in which city" in normalized_query or "what city" in normalized_query:
            if any(phrase in lower_text for phrase in ("based in", "located in", "located at")):
                bonus += 0.20
            if "city" in lower_text:
                bonus += 0.06
        if "whose name" in normalized_query or ({"name", "carry"} <= query_tokens):
            if "named after" in lower_text:
                bonus += 0.36
        if "parking" in query_tokens:
            if "parking" in lower_text:
                bonus += 0.18
            if any(term in lower_text for term in ("provided", "available", "permitted", "car park", "car parking")):
                bonus += 0.16
            if any(term in lower_text for term in ("guest", "guests", "visitor", "visitors")):
                bonus += 0.10
            if any(term in lower_text for term in ("student", "students", "registered students")):
                bonus += 0.10
        if "parent" in query_tokens or "parents" in query_tokens:
            if "parent" in lower_text or "parents" in lower_text:
                bonus += 0.18
            if any(term in lower_text for term in ("housing for parents", "visiting parents", "hotels", "airbnbs")):
                bonus += 0.18
        if "hour" in query_tokens or "hours" in query_tokens:
            if "working hours" in lower_text:
                bonus += 0.20
            if re.search(r"\b\d{1,2}:\d{2}\s*(am|pm)\b", lower_text):
                bonus += 0.12
            if {"weekday", "weekdays", "operating"} & query_tokens:
                if any(term in lower_text for term in ("monday to thursday", "monday - thursday", "friday")):
                    bonus += 0.20
                if any(term in lower_text for term in ("24/7", "all day", "sunday to saturday")):
                    bonus -= 0.38
            if "official" in query_tokens and ("official working hours" in lower_text or "official workings hours" in lower_text):
                bonus += 0.28
            if not (query_tokens & {"it", "support", "exam", "screening", "internship", "host", "organization"}):
                if any(term in lower_text for term in ("technical support", "it team", "screening exam", "internship", "host organization", "your organization", "24/7 security", "hours of operation")):
                    bonus -= 0.78
            if {"official", "weekday", "weekdays", "operating"} & query_tokens:
                if any(term in lower_text for term in ("technical support", "it team", "helpdesk", "support hours")):
                    bonus -= 0.35
                if "official working hours" in lower_text or "official workings hours" in lower_text:
                    bonus += 0.18
        if "shuttle" in query_tokens:
            if "shuttle" in lower_text:
                bonus += 0.18
            if "bus service" in lower_text:
                bonus += 0.12
        if {"family", "families", "parents", "parent"} & query_tokens:
            if any(term in lower_text for term in ("parents stay", "parents stay with me", "housing for parents", "visiting parents")):
                bonus += 0.42
            if any(term in lower_text for term in ("hotels", "airbnbs", "airbnb")):
                bonus += 0.22
        if _contextual_family_accommodation_match(query, text):
            bonus += 0.85
        if _is_generic_contact_query(query):
            narrow_scope_hits = [
                token for token in _CONTACT_NARROW_SCOPE_TOKENS
                if token in lower_text and token not in query_tokens
            ]
            if narrow_scope_hits:
                bonus -= min(0.40 + (0.12 * len(narrow_scope_hits)), 1.05)
            if any(
                phrase in lower_text
                for phrase in (
                    "contact",
                    "contact us",
                    "questions may be sent",
                    "for inquiries",
                    "for more information",
                )
            ):
                bonus += 0.18
        if {"specialization", "specializations"} & query_tokens:
            if "specialization" in lower_text or "specializations" in lower_text:
                bonus += 0.16
            if "machine learning" in lower_text and "computer vision" in lower_text:
                bonus += 0.16
        if {"specialization", "specializations", "program", "programs", "graduate"} & query_tokens:
            specialization_terms = (
                "machine learning",
                "computer vision",
                "natural language processing",
                "robotics",
                "computer science",
            )
            specialization_hits = sum(1 for term in specialization_terms if term in lower_text)
            if specialization_hits >= 3:
                bonus += 0.22
            if specialization_hits >= 4:
                bonus += 0.14
            if any(
                phrase in lower_text
                for phrase in ("five ai specializations", "currently offers ph.d.", "m.sc. programs", "ph.d. and m.sc. programs")
            ):
                bonus += 0.30
        if {"amenity", "amenities", "services", "facilities", "facility"} & query_tokens:
            amenity_terms = (
                "student accommodation",
                "laboratories",
                "knowledge center",
                "library",
                "auditorium",
                "multipurpose hall",
                "sports facility",
                "gym",
                "canteen",
                "retail outlets",
                "medical center",
                "pool",
            )
            if any(term in lower_text for term in amenity_terms):
                bonus += 0.24
        if {"legal", "basis"} <= query_tokens:
            if "law no." in lower_text or "established under law" in lower_text or "established by law" in lower_text:
                bonus += 0.24
        if {"affiliation", "affiliated"} & query_tokens or {"institutional", "affiliation"} <= query_tokens:
            if "affiliated to" in lower_text or "affiliated with" in lower_text or "under and shall be affiliated" in lower_text:
                bonus += 0.24
        if _query_starts_with(query, ("can ", "does ", "is ", "are ")):
            if "?" in text[:120]:
                bonus += 0.10
        return bonus

    def _rank_fact_anchor_chunk_ids(
        self,
        query: str,
        fact_ids: Sequence[str],
        *,
        top_k: int,
    ) -> List[str]:
        ranked_fact_ids = self._rank_fact_ids(query, fact_ids, top_k=max(top_k * 2, top_k))
        if not ranked_fact_ids:
            return []
        lookup_profile = _lookup_query_profile(query)
        scored: List[Tuple[str, float]] = []
        seen_chunk_ids = set()
        for fact_id in ranked_fact_ids:
            fact = self.fact_map.get(str(fact_id))
            if not fact:
                continue
            fact_text = _clean_text(fact.get("text") or fact.get("dense_text") or "")
            if not fact_text:
                continue
            fact_score = self._score_text_match(query, fact_text) + self._fact_query_bonus(query, fact_text)
            for chunk_id in fact.get("linked_chunk_ids") or []:
                chunk_id = str(chunk_id)
                if not chunk_id or chunk_id in seen_chunk_ids or chunk_id not in self.chunk_map:
                    continue
                seen_chunk_ids.add(chunk_id)
                chunk = self.chunk_map.get(chunk_id) or {}
                chunk_text = _clean_text(chunk.get("dense_text") or chunk.get("text") or "")
                chunk_heading = _clean_text(
                    " ".join(
                        str(value)
                        for value in (
                            chunk.get("heading") or "",
                            " > ".join(chunk.get("section_path") or []),
                            chunk.get("document_title") or "",
                        )
                    )
                )
                chunk_bonus = self._score_text_match(query, chunk_text) + _lookup_signal_bonus(query, chunk_text)
                heading_bonus = (self._score_text_match(query, chunk_heading) * 0.35) + (_lookup_signal_bonus(query, chunk_heading) * 0.25)
                if lookup_profile.is_exact_lookup:
                    combined_score = (fact_score * 1.75) + (chunk_bonus * 0.70) + (heading_bonus * 0.60)
                else:
                    combined_score = (fact_score * 1.25) + (chunk_bonus * 0.95) + heading_bonus
                scored.append((chunk_id, combined_score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [chunk_id for chunk_id, score in scored[:top_k] if score > 0.0]

    def _rank_fact_ids(
        self,
        query: str,
        fact_ids: Sequence[str],
        *,
        top_k: int,
    ) -> List[str]:
        scored: List[Tuple[str, float]] = []
        seen_fact_ids = set()
        lookup_profile = _lookup_query_profile(query)
        for fact_id in fact_ids:
            fact_id = str(fact_id)
            if not fact_id or fact_id in seen_fact_ids:
                continue
            seen_fact_ids.add(fact_id)
            fact = self.fact_map.get(fact_id)
            if not fact:
                continue
            fact_text = _clean_text(fact.get("text") or fact.get("dense_text") or "")
            if not fact_text:
                continue
            score = self._score_text_match(query, fact_text) + self._fact_query_bonus(query, fact_text)
            fact_lookup_text = _fact_lookup_text(fact) or fact_text
            if _contextual_contact_fact_match(query, fact_text, fact_lookup_text):
                score += 0.85
            if _contextual_family_accommodation_match(query, fact_lookup_text):
                score += 1.15
            score += self._source_query_bonus(
                query,
                source_url=str(fact.get("source_url") or ""),
                document_title=str(fact.get("document_title") or ""),
                heading=str(fact.get("heading") or ""),
                text=fact_lookup_text,
            )
            if lookup_profile.is_exact_lookup:
                answer_matches = any(
                    _text_matches_answer_type(fact_text, answer_type)
                    for answer_type in lookup_profile.answer_types
                )
                if not answer_matches:
                    continue
                score += _lookup_signal_bonus(query, fact_text) * 1.35
                if 4 <= len(fact_text.split()) <= 24:
                    score += 0.12
            if score > 0.0:
                scored.append((fact_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [fact_id for fact_id, _score in scored[:top_k]]

    def _local_evidence_span_query_ids(self, query: str, *, top_k: int) -> List[str]:
        if top_k <= 0 or not self.evidence_span_map:
            return []
        candidate_ids = self._lexical_query_ids(
            query,
            max(top_k, self.local_evidence_span_candidate_pool),
            namespace=self.namespace_evidence_spans,
        )
        if not candidate_ids:
            return []
        scored: List[Tuple[str, float]] = []
        for span_id in candidate_ids:
            span = self.evidence_span_map.get(str(span_id))
            if not span:
                continue
            text = _clean_text(
                " ".join(
                    str(span.get(key) or "")
                    for key in (
                        "document_title",
                        "section_heading",
                        "breadcrumb",
                        "span_type",
                        "sparse_text",
                        "text",
                    )
                )
            )
            score = self._score_text_match(query, text)
            score += _lookup_signal_bonus(query, text)
            score += self._fact_query_bonus(query, text)
            score += _answer_focus_match_bonus(query, text)
            score += self._source_query_bonus(
                query,
                source_url=str(
                    span.get("source_url")
                    or span.get("canonical_url")
                    or span.get("language_normalized_url")
                    or ""
                ),
                document_title=str(span.get("document_title") or ""),
                heading=str(
                    span.get("section_heading")
                    or span.get("breadcrumb")
                    or ""
                ),
                text=text,
            )
            if _contextual_family_accommodation_match(query, text):
                score += 1.15
            if score <= 0.0:
                continue
            scored.append((str(span_id), score))
        scored.sort(key=lambda item: (-item[1], item[0]))
        return [span_id for span_id, _score in scored[:top_k]]

    def _extract_search_ids(self, response: Any) -> List[str]:
        hits = None
        if isinstance(response, dict):
            hits = response.get("result", {}).get("hits") or response.get("hits")
        else:
            result = getattr(response, "result", None)
            hits = getattr(result, "hits", None) if result is not None else getattr(response, "hits", None)
        output: List[str] = []
        for hit in hits or []:
            record_id = None
            if isinstance(hit, dict):
                record_id = hit.get("_id") or hit.get("id")
            else:
                record_id = getattr(hit, "_id", None) or getattr(hit, "id", None)
            if record_id:
                output.append(str(record_id))
        return output

    def _sparse_query_ids(self, *, namespace: str, query: str, top_k: int) -> List[str]:
        if top_k <= 0:
            return []
        lexical_ids = self._lexical_query_ids(query, top_k, namespace=namespace)
        if not self.enable_sparse or not self.sparse_index_name:
            return lexical_ids
        try:
            response = self._pinecone_sparse_index().search(
                namespace=namespace,
                top_k=top_k,
                inputs={"text": query},
                fields=[],
                timeout=self.pinecone_query_timeout_seconds,
            )
            result = self._extract_search_ids(response)
            merged = _rrf_merge([result, lexical_ids], k=self.rrf_k)
            return [record_id for record_id, _score in merged[:top_k]]
        except Exception as exc:
            logger.warning("Sparse retrieval failed for namespace %s: %s", namespace, exc)
            return lexical_ids

    def _seed_chunk_ids(self, fused_ids: Iterable[str]) -> List[str]:
        chunk_ids: List[str] = []
        for record_id in fused_ids:
            if record_id in self.chunk_map:
                chunk_ids.append(record_id)
                continue
            if record_id in self.answer_map:
                answer = self.answer_map[record_id]
                chunk_ids.extend(answer.get("linked_chunk_ids") or [])
                for fact_id in answer.get("linked_fact_ids") or []:
                    chunk_ids.extend((self.fact_map.get(str(fact_id)) or {}).get("linked_chunk_ids") or [])
                continue
            if record_id in self.parent_map:
                chunk_ids.extend(self.parent_map[record_id].get("child_chunk_ids") or [])
                continue
            if record_id in self.summary_map:
                chunk_ids.extend(self.summary_map[record_id].get("linked_chunk_ids") or [])
                continue
            if record_id in self.evidence_span_map:
                span = self.evidence_span_map[record_id]
                chunk_ids.extend(span.get("linked_chunk_ids") or [])
                if span.get("chunk_id"):
                    chunk_ids.append(str(span.get("chunk_id")))
                continue
            if record_id in self.media_map:
                chunk_ids.extend(self.media_map[record_id].get("linked_chunk_ids") or [])
                continue
            if record_id in self.page_card_map:
                chunk_ids.extend(
                    getattr(self, "chunk_ids_by_page_card", {}).get(record_id, [])
                    or self.page_card_map[record_id].get("linked_chunk_ids")
                    or []
                )
                continue
            if record_id in self.action_map:
                chunk_ids.extend(self.action_map[record_id].get("linked_chunk_ids") or [])
                continue
            if record_id in self.fact_map:
                chunk_ids.extend(self.fact_map[record_id].get("linked_chunk_ids") or [])
        return list(dict.fromkeys(chunk_ids))

    def _span_anchor_chunk_ids(self, span_ids: Iterable[str]) -> List[str]:
        chunk_ids: List[str] = []
        for span_id in span_ids:
            span = self.evidence_span_map.get(str(span_id))
            if not span:
                continue
            chunk_ids.extend(str(value) for value in (span.get("linked_chunk_ids") or []) if str(value))
            if span.get("chunk_id"):
                chunk_ids.append(str(span.get("chunk_id")))
        return list(dict.fromkeys(chunk_ids))

    def _select_evidence_span_ids_for_query(
        self,
        query: str,
        ranked_span_ids: Sequence[str],
        *,
        mode: QueryMode,
    ) -> List[str]:
        if not ranked_span_ids:
            return []
        anchor_tokens = _evidence_span_anchor_tokens(query)
        if anchor_tokens:
            anchored_span_ids = [
                str(span_id)
                for span_id in ranked_span_ids
                if _record_matches_anchor_tokens(self.evidence_span_map.get(str(span_id)) or {}, anchor_tokens)
            ]
            if anchored_span_ids:
                ranked_span_ids = anchored_span_ids
        query_lower = query.lower()
        aggregation_query = bool(
            re.search(r"\b(compare|all|list|across|multiple|programs|departments|schools|faculty members|aggregate)\b", query_lower)
        )
        max_items = (
            max(12, self.parent_candidate_top_k + 6)
            if aggregation_query
            else max(6, self.parent_candidate_top_k + 4)
            if mode != QueryMode.SYNTHESIS
            else max(4, self.parent_candidate_top_k + 1)
        )
        max_per_source = 2 if aggregation_query else 3
        source_counts: Dict[str, int] = {}
        selected: List[str] = []

        def _span_source(span_id: str) -> str:
            span = self.evidence_span_map.get(str(span_id)) or {}
            return str(
                span.get("language_normalized_url")
                or span.get("canonical_url")
                or span.get("source_url")
                or span.get("page_id")
                or "local"
            )

        if aggregation_query:
            seen_sources = set()
            for span_id in ranked_span_ids:
                span_id = str(span_id)
                if span_id in selected or span_id not in self.evidence_span_map:
                    continue
                source = _span_source(span_id)
                if source in seen_sources:
                    continue
                selected.append(span_id)
                seen_sources.add(source)
                source_counts[source] = 1
                if len(selected) >= max_items:
                    return selected

        for span_id in ranked_span_ids:
            span_id = str(span_id)
            if span_id in selected or span_id not in self.evidence_span_map:
                continue
            source = _span_source(span_id)
            if source_counts.get(source, 0) >= max_per_source:
                continue
            selected.append(span_id)
            source_counts[source] = source_counts.get(source, 0) + 1
            if len(selected) >= max_items:
                break
        return selected

    def _promote_named_source_evidence_spans(
        self,
        query: str,
        ranked_span_ids: Sequence[str],
    ) -> List[str]:
        """Keep evidence from an explicitly named source ahead of topic matches."""

        ranked = [
            str(span_id)
            for span_id in dict.fromkeys(str(value) for value in ranked_span_ids)
            if str(span_id) in self.evidence_span_map
        ]
        if not ranked:
            return []
        identity_scores: Dict[str, float] = {}
        for span_id in ranked:
            span = self.evidence_span_map.get(span_id) or {}
            identity_scores[span_id] = _source_identity_bonus(
                query,
                source_url=str(
                    span.get("source_url")
                    or span.get("canonical_url")
                    or span.get("language_normalized_url")
                    or ""
                ),
                document_title=str(span.get("document_title") or ""),
                heading=str(
                    span.get("section_heading")
                    or span.get("breadcrumb")
                    or ""
                ),
            )
        if not any(score > 0.0 for score in identity_scores.values()):
            return ranked

        original_rank = {span_id: index for index, span_id in enumerate(ranked)}
        return sorted(
            ranked,
            key=lambda span_id: (
                -identity_scores[span_id],
                -self._score_text_match(
                    query,
                    str(
                        (self.evidence_span_map.get(span_id) or {}).get("text")
                        or (self.evidence_span_map.get(span_id) or {}).get("dense_text")
                        or ""
                    ),
                ),
                original_rank[span_id],
                span_id,
            ),
        )

    def _accumulate_chunk_support(self, rankings: Dict[str, Sequence[str]]) -> Dict[str, Dict[str, Any]]:
        support: Dict[str, Dict[str, Any]] = {}
        for source_name, ranking in rankings.items():
            weight = self.source_weights.get(source_name, 1.0)
            for rank, record_id in enumerate(ranking, start=1):
                for chunk_id in self._seed_chunk_ids([record_id]):
                    payload = support.setdefault(
                        chunk_id,
                        {"score": 0.0, "sources": set(), "record_ids": set()},
                    )
                    payload["score"] += weight / float(rank + self.rrf_k)
                    payload["sources"].add(source_name)
                    payload["record_ids"].add(record_id)
        return support

    def _build_rerank_text(
        self,
        chunk_id: str,
        *,
        max_tokens: int,
        max_chars: int | None = None,
    ) -> str:
        chunk = self.chunk_map.get(chunk_id, {})
        lines = []
        if chunk.get("document_title"):
            lines.append(f"TITLE: {chunk['document_title']}")
        if chunk.get("heading"):
            lines.append(f"HEADING: {chunk['heading']}")
        if chunk.get("section_path"):
            lines.append(f"SECTION: {' > '.join(chunk['section_path'])}")
        if chunk.get("page_numbers"):
            lines.append(f"PAGES: {', '.join(str(v) for v in chunk['page_numbers'])}")
        for answer_text in (self.answer_texts_by_chunk.get(chunk_id) or [])[:2]:
            lines.append(f"ANSWER: {answer_text}")
        for fact_text in (self.fact_texts_by_chunk.get(chunk_id) or [])[:2]:
            lines.append(f"FACT: {fact_text}")
        for media_text in (self.media_texts_by_chunk.get(chunk_id) or [])[:2]:
            lines.append(f"MEDIA: {media_text}")
        lines.append(chunk.get("text") or chunk.get("dense_text") or "")
        return _truncate_fragments(lines, max_tokens=max_tokens, max_chars=max_chars)

    def _parse_rerank_results(self, response: Any, documents: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
        results = getattr(response, "data", None) or getattr(response, "results", None)
        if results is None and isinstance(response, dict):
            results = response.get("data") or response.get("results")
        ranked: List[Tuple[str, float]] = []
        for item in results or []:
            if isinstance(item, dict):
                index = item.get("index")
                score = float(item.get("score") or 0.0)
                document = item.get("document") or {}
            else:
                index = getattr(item, "index", None)
                score = float(getattr(item, "score", 0.0) or 0.0)
                document = getattr(item, "document", None)
            record_id = None
            if isinstance(document, dict):
                record_id = document.get("id")
            if record_id is None and index is not None and 0 <= int(index) < len(documents):
                record_id = documents[int(index)]["id"]
            if record_id:
                ranked.append((str(record_id), score))
        return ranked

    def _fallback_rerank(
        self,
        query: str,
        candidate_chunk_ids: Sequence[str],
        support: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        mode = classify_query_mode(query)
        ranked: List[Tuple[str, float]] = []
        for chunk_id in candidate_chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            chunk_text = chunk.get("dense_text") or chunk.get("text") or ""
            overlap = self._score_text_match(query, chunk_text)
            overlap += _lookup_signal_bonus(query, chunk_text)
            overlap += self._fact_query_bonus(query, chunk_text)
            overlap += _answer_focus_match_bonus(query, chunk_text)
            overlap += self._source_query_bonus(
                query,
                source_url=str(chunk.get("source_url") or ""),
                document_title=str(chunk.get("document_title") or ""),
                heading=str(chunk.get("heading") or ""),
                text=chunk_text,
            )
            support_payload = support.get(chunk_id, {})
            prior = float(support_payload.get("score") or 0.0)
            support_sources = set(support_payload.get("sources") or set())
            if mode != QueryMode.FACT:
                corroboration_bonus = 0.0
                if {"dense_chunks", "sparse_chunks"} <= support_sources:
                    corroboration_bonus += 0.75
                if {"dense_parents", "sparse_parents"} <= support_sources:
                    corroboration_bonus += 0.25
                if support_sources & {"dense_media", "sparse_media", "local_media"}:
                    corroboration_bonus += 0.20
                if mode == QueryMode.SYNTHESIS:
                    corroboration_bonus += min(max(len(support_sources) - 1, 0) * 0.10, 0.40)
                overlap += corroboration_bonus
            ranked.append((chunk_id, prior + overlap))
        return sorted(ranked, key=lambda item: item[1], reverse=True)

    def _merge_reranked_with_remaining(
        self,
        reranked: Sequence[Tuple[str, float]],
        fallback_ranked: Sequence[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        if not reranked:
            return list(fallback_ranked)
        ranked_ids = {chunk_id for chunk_id, _score in reranked}
        rerank_base = max((score for _chunk_id, score in reranked), default=0.0) + 1.0
        merged: List[Tuple[str, float]] = [
            (chunk_id, rerank_base + float(score))
            for chunk_id, score in reranked
        ]
        for chunk_id, score in fallback_ranked:
            if chunk_id in ranked_ids:
                continue
            merged.append((chunk_id, float(score)))
        return merged

    def _promote_source_matched_chunks(
        self,
        query: str,
        ranked_chunks: Sequence[Tuple[str, float]],
        support: Dict[str, Dict[str, Any]],
    ) -> List[Tuple[str, float]]:
        mode = classify_query_mode(query)
        rescored: List[Tuple[str, float, int]] = []
        for rank, (chunk_id, score) in enumerate(ranked_chunks):
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            chunk_text = _clean_text(chunk.get("dense_text") or chunk.get("text") or "")
            source_bonus = self._source_query_bonus(
                query,
                source_url=str(chunk.get("source_url") or ""),
                document_title=str(chunk.get("document_title") or ""),
                heading=str(chunk.get("heading") or ""),
                text=chunk_text,
                mode=mode,
            )
            text_score = self._score_text_match(query, chunk_text)
            focus_score = _answer_focus_match_bonus(query, chunk_text)
            support_score = float((support.get(chunk_id) or {}).get("score") or 0.0)
            exact_source_bonus = 0.0
            if source_bonus >= 1.0 and (text_score >= 0.16 or focus_score >= 0.6):
                exact_source_bonus += 0.95
            if source_bonus >= 1.5:
                exact_source_bonus += 0.45
            if focus_score >= 1.9:
                exact_source_bonus += 0.90
            boosted = (
                float(score)
                + (1.85 * source_bonus)
                + (0.45 * text_score)
                + (2.65 * focus_score)
                + (1.35 * support_score)
                + exact_source_bonus
            )
            rescored.append((chunk_id, boosted, rank))
        rescored.sort(key=lambda item: (-item[1], item[2]))
        return [(chunk_id, score) for chunk_id, score, _rank in rescored]

    def _promote_fact_supported_chunks(
        self,
        query: str,
        ranked_chunks: Sequence[Tuple[str, float]],
        support: Dict[str, Dict[str, Any]],
        *,
        anchored_chunk_ids: Sequence[str] | None = None,
    ) -> List[Tuple[str, float]]:
        query_tokens = set(_tokenize(query))
        normalized_query = _clean_text(query).lower()
        anchored_chunk_ids = {str(value) for value in (anchored_chunk_ids or []) if str(value)}
        question_style_query = _query_starts_with(query, ("can ", "does ", "is ", "are ", "where ", "what ", "whose ", "in which "))
        rescored: List[Tuple[str, float, int]] = []
        for rank, (chunk_id, score) in enumerate(ranked_chunks):
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            support_payload = support.get(chunk_id, {})
            support_score = float(support_payload.get("score") or 0.0)
            support_sources = set(support_payload.get("sources") or set())
            fact_texts = list(self.fact_texts_by_chunk.get(chunk_id) or [])[:4]
            answer_texts = list(self.answer_texts_by_chunk.get(chunk_id) or [])[:4]
            fact_overlap = max(
                (
                    self._score_text_match(query, fact_text) + self._fact_query_bonus(query, fact_text)
                    for fact_text in fact_texts
                ),
                default=0.0,
            )
            answer_overlap = max(
                (
                    self._score_text_match(query, answer_text) + _lookup_signal_bonus(query, answer_text)
                    for answer_text in answer_texts
                ),
                default=0.0,
            )
            chunk_overlap = self._score_text_match(query, chunk.get("dense_text") or chunk.get("text") or "")
            has_fact_lane_support = bool(
                {"dense_facts", "sparse_facts", "dense_assertions", "sparse_assertions", "local_facts", "local_answers"}
                & support_sources
            )
            has_fact_text_support = bool(fact_texts or answer_texts)
            anchored_bonus = 1.4 if chunk_id in anchored_chunk_ids else 0.0
            fact_support_bonus = 1.10 if has_fact_lane_support else 0.0
            question_style_bonus = 0.0
            if question_style_query and any("?" in fact_text[:160] for fact_text in fact_texts):
                question_style_bonus += 0.45
            if chunk_id in anchored_chunk_ids and any(6 <= len(_clean_text(fact_text).split()) <= 24 for fact_text in fact_texts):
                question_style_bonus += 0.25
            if ("whose name" in normalized_query or ({"name", "carry"} <= query_tokens)) and any("named after" in fact_text.lower() for fact_text in fact_texts):
                question_style_bonus += 0.35
            if "parking" in query_tokens and any(
                "parking" in fact_text.lower() and any(term in fact_text.lower() for term in ("provided", "available", "permitted"))
                for fact_text in fact_texts
            ):
                question_style_bonus += 0.30
            narrative_penalty = 0.0
            chunk_text = _clean_text(chunk.get("dense_text") or chunk.get("text") or "")
            chunk_token_count = len(chunk_text.split())
            if not has_fact_lane_support and not has_fact_text_support:
                if question_style_query:
                    narrative_penalty += 0.55
                if chunk_token_count >= 80:
                    narrative_penalty += 0.35
                if query_tokens & {"parent", "parents", "parking", "permitted", "stay", "hours", "hour", "accommodation", "shuttle"}:
                    narrative_penalty += 0.30
            boosted_score = (
                float(score)
                + (2.8 * fact_overlap)
                + (2.4 * answer_overlap)
                + (0.9 * chunk_overlap)
                + (2.5 * support_score)
                + anchored_bonus
                + fact_support_bonus
                + question_style_bonus
                - narrative_penalty
            )
            rescored.append((chunk_id, boosted_score, rank))
        rescored.sort(key=lambda item: (-item[1], item[2]))
        return [(chunk_id, score) for chunk_id, score, _rank in rescored]

    def _scoped_parent_bonus(
        self,
        query: str,
        parent: Dict[str, Any],
        *,
        indexed_tokens: Iterable[str] | None = None,
    ) -> float:
        text = _clean_text(parent.get("dense_text") or parent.get("text") or "")
        if not text:
            return 0.0
        lower_text = text.lower()
        query_tokens = set(self._informative_query_tokens(query))
        scoring_text = text
        if len(text) > 8000:
            scoring_text = f"{text[:4000]} {text[-4000:]}"
        if indexed_tokens:
            bonus = self._score_preindexed_text_match(
                query,
                text=scoring_text,
                text_tokens=indexed_tokens,
            )
        else:
            bonus = self._score_text_match(query, scoring_text)
        bonus += self._source_query_bonus(
            query,
            source_url=str(parent.get("source_url") or ""),
            document_title=str(parent.get("document_title") or ""),
            heading=" > ".join(str(value) for value in (parent.get("section_path") or [])),
            text=scoring_text,
            mode=QueryMode.SCOPED,
        )
        heading_text = _clean_text(
            " ".join(
                [
                    str(parent.get("document_title") or ""),
                    " ".join(str(value) for value in (parent.get("section_path") or [])),
                ]
            )
        ).lower()
        if {"specialization", "specializations"} & query_tokens:
            if "specialization" in lower_text or "specializations" in lower_text:
                bonus += 0.18
            if "machine learning" in lower_text and "computer vision" in lower_text:
                bonus += 0.14
            if "five ai specializations" in lower_text or "specializations including" in lower_text:
                bonus += 0.40
        if {"specialization", "specializations", "program", "programs", "graduate"} & query_tokens:
            specialization_terms = (
                "machine learning",
                "computer vision",
                "natural language processing",
                "robotics",
                "computer science",
            )
            specialization_hits = sum(1 for term in specialization_terms if term in lower_text)
            if specialization_hits >= 3:
                bonus += 0.24
            if specialization_hits >= 4:
                bonus += 0.16
            if any(
                phrase in lower_text
                for phrase in ("currently offers ph.d.", "m.sc. programs", "ph.d. and m.sc. programs", "five ai specializations")
            ):
                bonus += 0.26
            if "about mbzuai" in heading_text and "specializations including" not in lower_text:
                bonus -= 0.28
            if "overview" in heading_text and "specializations including" not in lower_text:
                bonus -= 0.18
        if {"program", "programs", "graduate"} & query_tokens and any(term in lower_text for term in ("program", "programs", "m.sc", "ph.d", "graduate")):
            bonus += 0.16
            if "offers ph.d." in lower_text or "currently offers" in lower_text:
                bonus += 0.18
        legal_relation_tokens = {"law", "legal", "basis", "established", "affiliated", "affiliation", "authority", "institutional"}
        if (query_tokens & legal_relation_tokens) and any(
            term in lower_text
            for term in ("law no.", "established under law", "established by law", "established under law no.", "established by law no.")
        ):
            bonus += 0.24
        if (query_tokens & legal_relation_tokens) and all(
            term not in lower_text
            for term in ("law no.", "established under law", "established by law", "established under law no.", "established by law no.")
        ):
            bonus -= 0.16
        if {"affiliation", "affiliated", "authority"} & query_tokens or {"institutional", "affiliation"} <= query_tokens:
            if any(term in lower_text for term in ("affiliated to", "affiliated with", "under and shall be affiliated", "executive council")):
                bonus += 0.24
            elif "affiliated" not in lower_text and "executive council" not in lower_text:
                bonus -= 0.12
        if {"amenity", "amenities", "services", "facilities", "facility"} & query_tokens:
            if any(term in lower_text for term in ("services", "facilities", "facility", "amenities")):
                bonus += 0.16
            if any(
                term in lower_text
                for term in (
                    "student accommodation",
                    "knowledge center",
                    "library",
                    "canteen",
                    "gym",
                    "medical center",
                    "pool",
                    "sports facility",
                    "retail outlets",
                )
            ):
                bonus += 0.20
        if parent.get("parent_type") == "section":
            bonus += 0.05
        return bonus

    def _rank_parent_candidates(self, query: str, parent_ids: Sequence[str]) -> List[str]:
        scored: List[Tuple[str, float]] = []
        seen = set()
        for parent_id in parent_ids:
            parent_id = str(parent_id)
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)
            parent = self.parent_map.get(parent_id)
            if not parent:
                continue
            scored.append(
                (
                    parent_id,
                    self._scoped_parent_bonus(
                        query,
                        parent,
                        indexed_tokens=(
                            self._namespace_tokens_by_id.get(self.namespace_parents, {}).get(parent_id)
                            or set()
                        ),
                    ),
                )
            )
        scored.sort(key=lambda item: item[1], reverse=True)
        return [parent_id for parent_id, _score in scored]

    def _promote_media_supported_chunks(
        self,
        query: str,
        ranked_chunks: Sequence[Tuple[str, float]],
        media_hits: Sequence[str],
    ) -> List[Tuple[str, float]]:
        media_hit_set = {str(value) for value in media_hits if str(value)}
        rescored: List[Tuple[str, float, int]] = []
        for rank, (chunk_id, score) in enumerate(ranked_chunks):
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            chunk_media_ids = list(dict.fromkeys(chunk.get("media_ids") or []))
            best_media_score = 0.0
            explicit_hit_bonus = 0.0
            low_signal_penalty = 0.0
            for media_id in chunk_media_ids:
                media = self.media_map.get(media_id)
                if not media:
                    continue
                media_score = self._score_media_relevance(query, media)
                if media_score < 0.0:
                    low_signal_penalty += 0.6
                    continue
                if media_id in media_hit_set:
                    explicit_hit_bonus = max(explicit_hit_bonus, 0.45)
                best_media_score = max(best_media_score, media_score)
            section_text = " ".join(
                str(value)
                for value in (
                    chunk.get("heading") or "",
                    " > ".join(chunk.get("section_path") or []),
                    chunk.get("document_title") or "",
                )
            )
            chunk_text = _clean_text(chunk.get("dense_text") or chunk.get("text") or "")
            specific_tokens = set(self._specific_visual_tokens(query))
            chunk_tokens = set(_tokenize(" ".join([chunk_text, section_text])))
            missing_specific_penalty = 0.0
            if specific_tokens and not (specific_tokens & chunk_tokens):
                missing_specific_penalty = 0.9
            section_bonus = _phrase_match_bonus(self._media_keywords(query), section_text)
            rescored.append(
                (
                    chunk_id,
                    float(score)
                    + (2.2 * best_media_score)
                    + explicit_hit_bonus
                    + section_bonus
                    - low_signal_penalty
                    - missing_specific_penalty,
                    rank,
                )
            )
        rescored.sort(key=lambda item: (-item[1], item[2]))
        return [(chunk_id, score) for chunk_id, score, _rank in rescored]

    def _rerank_chunk_candidates(
        self,
        query: str,
        candidate_chunk_ids: Sequence[str],
        support: Dict[str, Dict[str, Any]],
        *,
        mode: QueryMode | None = None,
        graph_seed_chunk_ids: Sequence[str] | None = None,
        graph_seed_fact_ids: Sequence[str] | None = None,
        force_fallback: bool = False,
        diagnostics: Dict[str, Any] | None = None,
    ) -> List[Tuple[str, float]]:
        started = time.perf_counter()

        def _record_rerank(method: str) -> None:
            latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
            if diagnostics is not None:
                diagnostics["rerank_latency_ms"] = latency_ms
                diagnostics["rerank_method"] = method
            else:
                # Compatibility for direct callers. Production retrieval passes a
                # request-local diagnostics object and never relies on these fields.
                self._last_rerank_latency_ms = latency_ms
                self._last_rerank_method = method

        mode = mode or classify_query_mode(query)
        rerank_top_n = self.rerank_top_n
        rerank_return_top_k = self.rerank_return_top_k
        lightweight_fact_rerank = self._is_lightweight_fact_rerank_query(query, mode=mode)
        if lightweight_fact_rerank:
            rerank_top_n = min(rerank_top_n, max(1, self.rerank_fact_top_n))
            rerank_return_top_k = min(rerank_return_top_k, max(1, self.rerank_fact_return_top_k))

        candidate_chunk_ids = list(dict.fromkeys(candidate_chunk_ids))[: max(rerank_top_n, self.max_context_chunks)]
        if not candidate_chunk_ids:
            _record_rerank("none")
            return []
        fallback_ranked = self._fallback_rerank(query, candidate_chunk_ids, support)
        if force_fallback:
            _record_rerank("fallback_external_lanes_disabled")
            return fallback_ranked
        if not self.enable_rerank:
            _record_rerank("fallback_disabled")
            return fallback_ranked
        if self._should_skip_rerank(
            query=query,
            mode=mode,
            fallback_ranked=fallback_ranked,
            support=support,
            graph_seed_chunk_ids=graph_seed_chunk_ids,
            graph_seed_fact_ids=graph_seed_fact_ids,
            lightweight_fact_rerank=lightweight_fact_rerank,
        ):
            _record_rerank("fallback_high_confidence_fact")
            return fallback_ranked
        rerank_query = _truncate_tokens(query, max_tokens=self.rerank_query_max_tokens)
        token_caps = [self.rerank_doc_max_tokens, *self.rerank_retry_doc_max_tokens]
        seen_caps = set()
        for token_cap in token_caps:
            if token_cap in seen_caps:
                continue
            seen_caps.add(token_cap)
            doc_char_cap = max(0, self.rerank_doc_max_chars)
            if self.rerank_request_max_chars > 0 and candidate_chunk_ids:
                request_cap = max(180, self.rerank_request_max_chars // max(len(candidate_chunk_ids), 1))
                doc_char_cap = min(doc_char_cap or request_cap, request_cap)
            documents = [
                {
                    "id": chunk_id,
                    "text": self._build_rerank_text(
                        chunk_id,
                        max_tokens=token_cap,
                        max_chars=doc_char_cap or None,
                    ),
                }
                for chunk_id in candidate_chunk_ids
                if chunk_id in self.chunk_map
            ]
            documents = [document for document in documents if document["text"]]
            if not documents:
                break
            try:
                response = self._pinecone_client_obj().inference.rerank(
                    model=self.rerank_model,
                    query=rerank_query,
                    documents=documents,
                    rank_fields=["text"],
                    top_n=min(rerank_return_top_k, len(documents)),
                    return_documents=True,
                )
                ranked = self._parse_rerank_results(response, documents)
                if ranked:
                    _record_rerank("pinecone")
                    return self._merge_reranked_with_remaining(ranked, fallback_ranked)
            except Exception as exc:
                if "exceeds the maximum token limit" in str(exc).lower():
                    logger.warning(
                        "Reranking exceeded token limit with doc cap %d; retrying with a smaller cap",
                        token_cap,
                    )
                    continue
                logger.warning("Reranking failed, using fallback ranking: %s", exc)
                break
        _record_rerank("fallback_error")
        return fallback_ranked

    def _is_lightweight_fact_rerank_query(self, query: str, *, mode: QueryMode) -> bool:
        if mode != QueryMode.FACT:
            return False
        query_tokens = set(_tokenize(query))
        if not _query_starts_with(query, ("can ", "does ", "is ", "are ")):
            return False
        return bool(
            query_tokens
            & {
                "parking",
                "visitor",
                "visitors",
                "guest",
                "guests",
                "accommodation",
                "housing",
                "shuttle",
                "transport",
                "transportation",
                "support",
                "hour",
                "hours",
                "family",
                "families",
                "parent",
                "parents",
            }
        )

    def _should_skip_rerank(
        self,
        *,
        query: str,
        mode: QueryMode,
        fallback_ranked: Sequence[Tuple[str, float]],
        support: Dict[str, Dict[str, Any]],
        graph_seed_chunk_ids: Sequence[str] | None = None,
        graph_seed_fact_ids: Sequence[str] | None = None,
        lightweight_fact_rerank: bool = False,
    ) -> bool:
        if (
            mode != QueryMode.FACT
            or not lightweight_fact_rerank
            or not self.rerank_skip_high_confidence_fact
            or not fallback_ranked
        ):
            return False
        top_chunk_id = str(fallback_ranked[0][0] or "")
        if not top_chunk_id:
            return False
        chunk = self.chunk_map.get(top_chunk_id)
        if not chunk:
            return False
        support_payload = support.get(top_chunk_id, {})
        support_sources = set(support_payload.get("sources") or set())
        support_score = float(support_payload.get("score") or 0.0)
        chunk_overlap = self._score_text_match(query, chunk.get("dense_text") or chunk.get("text") or "")
        fact_texts = list(self.fact_texts_by_chunk.get(top_chunk_id) or [])[:4]
        answer_texts = list(self.answer_texts_by_chunk.get(top_chunk_id) or [])[:4]
        fact_overlap = max(
            (
                self._score_text_match(query, fact_text) + self._fact_query_bonus(query, fact_text)
                for fact_text in fact_texts
            ),
            default=0.0,
        )
        answer_overlap = max(
            (
                self._score_text_match(query, answer_text) + _lookup_signal_bonus(query, answer_text)
                for answer_text in answer_texts
            ),
            default=0.0,
        )
        effective_overlap = max(chunk_overlap, fact_overlap, answer_overlap)
        has_fact_support = bool(
            support_sources
            & {
                "dense_facts",
                "sparse_facts",
                "dense_assertions",
                "sparse_assertions",
                "local_facts",
                "local_answers",
                "graph_relation_facts",
            }
        )
        if not has_fact_support or support_score < self.rerank_skip_fact_support_score:
            return False
        graph_seed_chunk_ids = {str(value) for value in (graph_seed_chunk_ids or []) if str(value)}
        has_graph_seed_support = bool(
            top_chunk_id in graph_seed_chunk_ids
            or support_sources & {"graph_relation_chunks", "graph_relation_facts"}
        )
        if has_graph_seed_support and effective_overlap >= max(0.62, self.rerank_skip_fact_overlap - 0.08):
            return True
        if effective_overlap >= self.rerank_skip_fact_overlap:
            return True
        return False

    def _expand_fact(self, seed_chunk_ids: List[str]) -> List[str]:
        ordered_seed_ids = list(dict.fromkeys(chunk_id for chunk_id in seed_chunk_ids if chunk_id in self.chunk_map))
        selected: List[str] = ordered_seed_ids[: self.max_context_chunks]
        if len(selected) >= self.max_context_chunks:
            return selected
        for chunk_id in ordered_seed_ids:
            if chunk_id not in self.chunk_map:
                continue
            chunk = self.chunk_map[chunk_id]
            neighbors = list(chunk.get("neighbor_ids") or [])
            for neighbor_id in neighbors[: self.fact_neighbor_window * 2]:
                selected.append(neighbor_id)
            if len(dict.fromkeys(selected)) >= self.max_context_chunks:
                break
        return list(dict.fromkeys(selected))[: self.max_context_chunks]

    def _should_use_support_parent_scan(self, query: str, *, mode: QueryMode) -> bool:
        if mode == QueryMode.FACT:
            return False
        return bool(_support_query_intents(query))

    def _support_parent_ids_for_query(self, query: str, *, top_k: int) -> List[str]:
        mode = classify_query_mode(query)
        if top_k <= 0 or not self._should_use_support_parent_scan(query, mode=mode):
            return []
        candidate_ids = self._lexical_query_ids(
            query,
            max(128, self.local_parent_candidate_pool, top_k * 16),
            namespace=self.namespace_parents,
        )
        if not candidate_ids:
            # The independent dense and ordinary local parent lanes remain
            # available. A support hint must never trigger a complete parent
            # corpus scan in the serving path.
            return []
        scored: List[Tuple[str, float, int]] = []
        for index, parent_id in enumerate(candidate_ids):
            parent_id = str(parent_id or "")
            parent = self.parent_map.get(parent_id)
            if not parent:
                continue
            text = _clean_text(parent.get("dense_text") or parent.get("text") or "")
            scoring_text = text
            if len(scoring_text) > 8000:
                scoring_text = f"{scoring_text[:4000]} {scoring_text[-4000:]}"
            heading = " > ".join(str(value) for value in (parent.get("section_path") or []))
            support_bonus = self._source_query_bonus(
                query,
                source_url=str(parent.get("source_url") or ""),
                document_title=str(parent.get("document_title") or ""),
                heading=heading,
                text=scoring_text,
                mode=mode,
            )
            lexical_score = self._score_preindexed_text_match(
                query,
                text=" ".join([heading, scoring_text]),
                text_tokens=(
                    self._namespace_tokens_by_id.get(self.namespace_parents, {}).get(parent_id)
                    or set()
                ),
            )
            score = support_bonus + (0.35 * lexical_score)
            if parent.get("parent_type") == "page":
                score += 0.08
            if score < 0.90:
                continue
            scored.append((parent_id, score, index))
        scored.sort(key=lambda item: (-item[1], item[2], item[0]))
        return [parent_id for parent_id, _score, _index in scored[:top_k]]

    def _support_chunk_ids_for_query(
        self,
        query: str,
        parent_ids: Sequence[str],
        *,
        top_k: int,
    ) -> List[str]:
        if top_k <= 0 or not parent_ids:
            return []
        per_parent = max(2, min(self.max_parent_chunks, max(3, top_k // max(1, min(len(parent_ids), 3)))))
        selected: List[str] = []
        for parent_id in parent_ids[: max(2, self.parent_candidate_top_k + 1)]:
            selected.extend(self._rank_parent_child_chunk_ids(query, str(parent_id), top_k=per_parent))
            if len(dict.fromkeys(selected)) >= top_k:
                break
        return list(dict.fromkeys(selected))[:top_k]

    def _rank_parent_child_chunk_ids(self, query: str, parent_id: str, *, top_k: int) -> List[str]:
        parent = self.parent_map.get(str(parent_id))
        if not parent:
            return []
        child_ids = [str(value) for value in (parent.get("child_chunk_ids") or []) if str(value) in self.chunk_map]
        if not child_ids:
            return []
        query = str(query or "")
        scored: List[Tuple[str, float, int]] = []
        for index, chunk_id in enumerate(child_ids):
            chunk = self.chunk_map.get(chunk_id) or {}
            text = str(chunk.get("dense_text") or chunk.get("text") or "")
            heading = str(chunk.get("heading") or "")
            source_url = str(chunk.get("source_url") or "")
            document_title = str(chunk.get("document_title") or "")
            content_score = self._score_text_match(query, " ".join([heading, text]))
            source_score = self._source_query_bonus(
                query,
                source_url=source_url,
                document_title=document_title,
                heading=heading,
                text=text,
                mode=classify_query_mode(query),
            )
            # The first child of MBZUAI web pages usually contains the page overview,
            # deadline, credits, and other high-value fields hidden by repeated course sections.
            lead_bonus = max(0.0, 0.55 - (0.06 * index))
            if _support_query_intents(query):
                # Support-style questions usually need FAQ/application/scholarship
                # detail chunks, not just the lead curriculum/overview chunk.
                lead_bonus = min(lead_bonus, 0.16)
            scored.append((chunk_id, content_score + (0.78 * source_score) + lead_bonus, index))
        scored.sort(key=lambda item: (-item[1], item[2], item[0]))
        return [chunk_id for chunk_id, _score, _index in scored[: max(1, int(top_k or 1))]]

    def _rank_page_child_chunk_ids_with_section_diversity(
        self,
        query: str,
        parent_id: str,
        *,
        top_k: int,
    ) -> List[str]:
        """Rank a page's children while covering distinct sibling sections.

        List and comparison answers commonly live in separate H2/H3 sections.
        Ordinary relevance ranking can spend the context window on several
        chunks from one long section and omit equally important siblings. The
        first pass keeps the best chunk from each section; remaining slots use
        the normal relevance order.
        """

        parent = self.parent_map.get(str(parent_id)) or {}
        child_ids = [
            str(value)
            for value in (parent.get("child_chunk_ids") or [])
            if str(value) in self.chunk_map
        ]
        if not child_ids:
            return []
        ranked = self._rank_parent_child_chunk_ids(
            query,
            parent_id,
            top_k=len(child_ids),
        )
        first_by_section: List[str] = []
        remainder: List[str] = []
        seen_sections: set[str] = set()
        for chunk_id in ranked:
            chunk = self.chunk_map.get(chunk_id) or {}
            section_ids = self._parent_ids_for_chunk(
                chunk_id,
                parent_type="section",
            )
            section_key = str(section_ids[0]) if section_ids else _clean_text(
                chunk.get("heading")
                or " > ".join(chunk.get("section_path") or [])
                or chunk_id
            ).casefold()
            if section_key and section_key not in seen_sections:
                seen_sections.add(section_key)
                first_by_section.append(chunk_id)
            else:
                remainder.append(chunk_id)
        return list(dict.fromkeys([*first_by_section, *remainder]))[
            : max(1, int(top_k or 1))
        ]

    def _expand_scoped_or_synthesis(
        self,
        seed_chunk_ids: List[str],
        *,
        query: str = "",
        mode: QueryMode,
        explicit_parent_ids: Sequence[str] | None = None,
        prioritize_explicit_parents: bool = False,
        prefer_explicit_parent_chunks_first: bool = False,
    ) -> List[str]:
        support_intents = _support_query_intents(query)
        if mode == QueryMode.SYNTHESIS:
            preserve_seed_limit = max(4, min(self.max_context_chunks // 2, 8))
        else:
            preserve_seed_limit = max(4, min(self.max_context_chunks // 2, 8))
        if support_intents & {"person_statement", "president_background", "program_expansion_history", "news_event"}:
            preserve_seed_limit = max(preserve_seed_limit, self.max_context_chunks)
        preserved_seed_ids: List[str] = list(dict.fromkeys(seed_chunk_ids))[:preserve_seed_limit]
        selected: List[str] = list(preserved_seed_ids)
        by_section: Dict[str, List[str]] = {}
        by_page: Dict[str, List[str]] = {}
        for chunk_id in seed_chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            for parent_id in self._parent_ids_for_chunk(chunk_id, parent_type="section"):
                by_section.setdefault(parent_id, []).append(chunk_id)
            for parent_id in self._parent_ids_for_chunk(chunk_id, parent_type="page"):
                by_page.setdefault(parent_id, []).append(chunk_id)

        explicit_parent_ids = [str(value) for value in (explicit_parent_ids or []) if str(value)]
        explicit_parent_rank = {
            parent_id: float(len(explicit_parent_ids) - idx)
            for idx, parent_id in enumerate(explicit_parent_ids)
        }

        if prioritize_explicit_parents:
            added_explicit_parent_chunks = False
            explicit_parent_chunk_ids: List[str] = []
            per_parent_limit = max(3, min(self.max_parent_chunks, max(4, self.max_context_chunks // max(1, min(len(explicit_parent_ids), self.parent_candidate_top_k or 1)))))
            for parent_id in explicit_parent_ids[: max(2, self.parent_candidate_top_k)]:
                parent = self.parent_map.get(parent_id)
                if not parent:
                    continue
                added_explicit_parent_chunks = True
                ranked_parent_chunks = self._rank_parent_child_chunk_ids(
                    query,
                    parent_id,
                    top_k=per_parent_limit,
                )
                explicit_parent_chunk_ids.extend(ranked_parent_chunks)
                if prefer_explicit_parent_chunks_first:
                    selected = list(dict.fromkeys([*explicit_parent_chunk_ids, *preserved_seed_ids]))
                else:
                    selected.extend(ranked_parent_chunks)
                if len(dict.fromkeys(selected)) >= self.max_context_chunks:
                    return list(dict.fromkeys(selected))[: self.max_context_chunks]
            if added_explicit_parent_chunks and mode != QueryMode.SYNTHESIS:
                if prefer_explicit_parent_chunks_first:
                    selected = list(dict.fromkeys([*explicit_parent_chunk_ids, *preserved_seed_ids]))
                return list(dict.fromkeys(selected))[: self.max_context_chunks]

        section_candidates = set(section_id for section_id in by_section if section_id)
        section_candidates.update(
            parent_id
            for parent_id in explicit_parent_ids
            if parent_id in self.parent_map and self.parent_map[parent_id].get("parent_type") == "section"
        )
        section_first_seen = {
            parent_id: index for index, parent_id in enumerate(by_section)
        }
        section_explicit_first_seen = {
            parent_id: index for index, parent_id in enumerate(explicit_parent_ids)
        }
        section_ids = sorted(
            section_candidates,
            key=lambda item: (
                -(
                    (len(by_section.get(item, [])) * 10.0)
                    + explicit_parent_rank.get(item, 0.0)
                ),
                section_first_seen.get(
                    item,
                    len(section_first_seen)
                    + section_explicit_first_seen.get(item, len(explicit_parent_ids)),
                ),
                item,
            ),
        )
        expanded_section = False
        for section_id in section_ids:
            hit_count = len(by_section.get(section_id, []))
            explicit_hit = section_id in explicit_parent_rank
            if hit_count < self.same_parent_expand_threshold and mode != QueryMode.SYNTHESIS and not explicit_hit:
                continue
            ranked_section_chunks = self._rank_parent_child_chunk_ids(
                query,
                section_id,
                top_k=self.max_parent_chunks,
            )
            if ranked_section_chunks:
                expanded_section = True
                selected.extend(ranked_section_chunks)
            if len(dict.fromkeys(selected)) >= self.max_context_chunks:
                return list(dict.fromkeys(selected))[: self.max_context_chunks]

        aggregate_page_query = bool(
            mode == QueryMode.SYNTHESIS and _is_enumeration_query(query)
        )

        # Seeds are copied into ``selected`` before expansion, so checking
        # ``not selected`` made page fallback unreachable for any real query.
        # For ordinary scoped queries, fall back to page parents only when no
        # section expanded. Enumeration queries are different: their evidence
        # often lives in sibling sections, so a successful single-section
        # expansion must not prevent page-level assembly.
        if not expanded_section or aggregate_page_query:
            page_candidates = set(page_id for page_id in by_page if page_id)
            page_candidates.update(
                parent_id
                for parent_id in explicit_parent_ids
                if parent_id in self.parent_map and self.parent_map[parent_id].get("parent_type") == "page"
            )
            page_first_seen = {
                parent_id: index for index, parent_id in enumerate(by_page)
            }
            page_explicit_first_seen = {
                parent_id: index for index, parent_id in enumerate(explicit_parent_ids)
            }
            page_ids = sorted(
                page_candidates,
                key=lambda item: (
                    -(
                        (len(by_page.get(item, [])) * 10.0)
                        + explicit_parent_rank.get(item, 0.0)
                    ),
                    page_first_seen.get(
                        item,
                        len(page_first_seen)
                        + page_explicit_first_seen.get(item, len(explicit_parent_ids)),
                    ),
                    item,
                ),
            )
            aggregate_page_chunks: List[str] = []
            for page_id in page_ids:
                hit_count = len(by_page.get(page_id, []))
                explicit_hit = page_id in explicit_parent_rank
                if hit_count < self.same_parent_expand_threshold and mode != QueryMode.SYNTHESIS and not explicit_hit:
                    continue
                if aggregate_page_query:
                    ranked_page_chunks = self._rank_page_child_chunk_ids_with_section_diversity(
                        query,
                        page_id,
                        top_k=self.max_parent_chunks,
                    )
                    aggregate_page_chunks.extend(ranked_page_chunks)
                else:
                    selected.extend(
                        self._rank_parent_child_chunk_ids(
                            query,
                            page_id,
                            top_k=self.max_parent_chunks,
                        )
                    )
                if len(dict.fromkeys(selected)) >= self.max_context_chunks:
                    return list(dict.fromkeys(selected))[: self.max_context_chunks]
                if aggregate_page_query and aggregate_page_chunks:
                    # One strongly ranked page normally owns the complete set.
                    # Keep a small number of top seeds as independent recall
                    # anchors, then reserve the context window for its sibling
                    # sections before unrelated lower-ranked seeds.
                    selected = list(
                        dict.fromkeys(
                            [
                                *preserved_seed_ids[:2],
                                *aggregate_page_chunks,
                                *preserved_seed_ids[2:],
                                *selected,
                            ]
                        )
                    )
                    return selected[: self.max_context_chunks]

        if not selected:
            return self._expand_fact(seed_chunk_ids)
        return list(dict.fromkeys(selected))[: self.max_context_chunks]

    def _select_parent_ids(
        self,
        chunk_ids: Sequence[str],
        *,
        explicit_parent_ids: Sequence[str] | None = None,
        prefer_explicit_parents: bool = False,
    ) -> List[str]:
        parent_scores: Dict[str, float] = {}
        parent_first_seen: Dict[str, int] = {}
        section_counts: Dict[str, int] = {}
        page_counts: Dict[str, int] = {}
        for chunk_rank, chunk_id in enumerate(chunk_ids):
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            for section_parent_id in self._parent_ids_for_chunk(chunk_id, parent_type="section"):
                section_counts[section_parent_id] = section_counts.get(section_parent_id, 0) + 1
                parent_scores[section_parent_id] = parent_scores.get(section_parent_id, 0.0) + max(6.0 - float(chunk_rank), 1.0) * 2.0
                parent_first_seen.setdefault(section_parent_id, chunk_rank * 2)
            for page_parent_id in self._parent_ids_for_chunk(chunk_id, parent_type="page"):
                page_counts[page_parent_id] = page_counts.get(page_parent_id, 0) + 1
                parent_scores[page_parent_id] = parent_scores.get(page_parent_id, 0.0) + max(6.0 - float(chunk_rank), 1.0) * 1.5
                parent_first_seen.setdefault(page_parent_id, chunk_rank * 2 + 1)
        selected_chunk_ids = set(str(value) for value in chunk_ids if str(value))
        for explicit_rank, parent_id in enumerate(explicit_parent_ids or []):
            parent = self.parent_map.get(str(parent_id))
            if not parent:
                continue
            parent_key = str(parent_id)
            if prefer_explicit_parents:
                explicit_bonus = max(8.0 - float(explicit_rank), 1.0)
            else:
                explicit_bonus = max(4.0 - float(explicit_rank), 0.5)
            child_chunk_ids = set(str(value) for value in (parent.get("child_chunk_ids") or []))
            if child_chunk_ids & selected_chunk_ids:
                parent_scores[parent_key] = parent_scores.get(parent_key, 0.0) + explicit_bonus * 1.25
                parent_first_seen.setdefault(parent_key, 1000 + explicit_rank)
            else:
                background_multiplier = 2.5 if prefer_explicit_parents else 0.45
                parent_scores[parent_key] = parent_scores.get(parent_key, 0.0) + explicit_bonus * background_multiplier
                parent_first_seen.setdefault(parent_key, 2000 + explicit_rank)
        for parent_id, count in section_counts.items():
            if parent_id in self.parent_map:
                parent_scores[parent_id] = parent_scores.get(parent_id, 0.0) + count * 1.5
        for parent_id, count in page_counts.items():
            if parent_id in self.parent_map:
                parent_scores[parent_id] = parent_scores.get(parent_id, 0.0) + count * 1.0
        limit = max(4, self.parent_candidate_top_k * 2)
        if prefer_explicit_parents:
            seeded: List[str] = []
            if chunk_ids:
                first_chunk = self.chunk_map.get(str(chunk_ids[0]))
                if first_chunk:
                    first_chunk_id = str(first_chunk.get("id") or chunk_ids[0])
                    first_sections = self._parent_ids_for_chunk(first_chunk_id, parent_type="section")
                    first_pages = self._parent_ids_for_chunk(first_chunk_id, parent_type="page")
                    seeded.extend(first_sections[:1] or first_pages[:1])
            seeded.extend(str(parent_id) for parent_id in (explicit_parent_ids or []) if str(parent_id) in self.parent_map)
            ordered = seeded + sorted(
                (parent_id for parent_id in parent_scores if parent_id in self.parent_map),
                key=lambda parent_id: (-parent_scores[parent_id], parent_first_seen.get(parent_id, 10_000), parent_id),
            )
            return list(dict.fromkeys(ordered))[:limit]
        ordered = sorted(
            (parent_id for parent_id in parent_scores if parent_id in self.parent_map),
            key=lambda parent_id: (-parent_scores[parent_id], parent_first_seen.get(parent_id, 10_000), parent_id),
        )
        return ordered[:limit]

    def _preserve_dense_chunk_recall(
        self,
        selected_chunk_ids: Sequence[str],
        dense_chunk_ids: Sequence[str],
    ) -> List[str]:
        """Keep selected dense-graph anchors inside the final context window."""

        selected = list(
            dict.fromkeys(
                str(value)
                for value in selected_chunk_ids
                if str(value) in self.chunk_map
            )
        )
        if self.dense_recall_floor_k <= 0 or self.dense_recall_window <= 0:
            return selected[: self.max_context_chunks]
        anchors = list(
            dict.fromkeys(
                str(value)
                for value in dense_chunk_ids
                if str(value) in self.chunk_map
            )
        )[: self.dense_recall_floor_k]
        if not anchors:
            return selected[: self.max_context_chunks]
        window = min(self.max_context_chunks, self.dense_recall_window)
        missing = [value for value in anchors if value not in selected[:window]]
        if not missing:
            return selected[: self.max_context_chunks]
        retained = selected[: max(0, window - len(missing))]
        return list(dict.fromkeys([*retained, *missing, *selected]))[
            : self.max_context_chunks
        ]

    def _preserve_dense_parent_recall(
        self,
        selected_parent_ids: Sequence[str],
        dense_chunk_ids: Sequence[str],
    ) -> List[str]:
        """Keep page parents for leading dense chunks inside the parent window."""

        selected = list(
            dict.fromkeys(
                str(value)
                for value in selected_parent_ids
                if str(value) in self.parent_map
            )
        )
        if (
            self.dense_parent_recall_floor_k <= 0
            or self.dense_parent_recall_window <= 0
        ):
            return selected
        anchors: List[str] = []
        for chunk_id in dense_chunk_ids:
            anchors.extend(
                self._parent_ids_for_chunk(str(chunk_id), parent_type="page")
            )
            if len(list(dict.fromkeys(anchors))) >= self.dense_parent_recall_floor_k:
                break
        anchors = list(dict.fromkeys(anchors))[: self.dense_parent_recall_floor_k]
        if not anchors:
            return selected
        window = self.dense_parent_recall_window
        missing = [value for value in anchors if value not in selected[:window]]
        if not missing:
            return selected
        retained = selected[: max(0, window - len(missing))]
        limit = max(len(selected), window)
        return list(dict.fromkeys([*retained, *missing, *selected]))[:limit]

    def _promote_selected_chunk_parent_ids(
        self,
        query: str,
        selected_parent_ids: Sequence[str],
        selected_chunk_ids: Sequence[str],
    ) -> List[str]:
        limit = max(4, self.parent_candidate_top_k * 2)
        existing = [str(parent_id) for parent_id in selected_parent_ids if str(parent_id)]
        parent_candidates: List[Tuple[str, float, int]] = []
        for chunk_rank, chunk_id in enumerate(selected_chunk_ids[: self.max_context_chunks]):
            chunk = self.chunk_map.get(str(chunk_id))
            if not chunk:
                continue
            chunk_text = _clean_text(chunk.get("dense_text") or chunk.get("text") or "")
            source_bonus = self._source_query_bonus(
                query,
                source_url=str(chunk.get("source_url") or ""),
                document_title=str(chunk.get("document_title") or ""),
                heading=str(chunk.get("heading") or ""),
                text=chunk_text,
                mode=classify_query_mode(query),
            )
            text_score = self._score_text_match(query, chunk_text)
            focus_score = _answer_focus_match_bonus(query, chunk_text)
            top_evidence_anchor = chunk_rank < max(3, self.parent_candidate_top_k)
            if not top_evidence_anchor and (source_bonus < 2.4 or max(text_score, focus_score) < 0.55):
                continue
            rank_bonus = max(0.0, 1.0 - (0.08 * float(chunk_rank)))
            parent_score = (1.25 * source_bonus) + (0.55 * text_score) + (0.90 * focus_score) + rank_bonus
            if top_evidence_anchor:
                parent_score = max(parent_score, 4.35 - (0.08 * float(chunk_rank)))
            for parent_offset, parent_id in enumerate(self._parent_ids_for_chunk(str(chunk_id))):
                if parent_id and parent_id in self.parent_map:
                    parent_candidates.append((parent_id, parent_score - (0.05 * parent_offset), chunk_rank))

        if not parent_candidates:
            return self._diversify_parent_ids_for_query(query, existing, limit=limit)
        best_by_parent: Dict[str, Tuple[float, int]] = {}
        for parent_id, score, rank in parent_candidates:
            previous = best_by_parent.get(parent_id)
            if previous is None or (score, -rank) > (previous[0], -previous[1]):
                best_by_parent[parent_id] = (score, rank)
        promoted = [
            parent_id
            for parent_id, (score, rank) in sorted(
                best_by_parent.items(),
                key=lambda item: (-item[1][0], item[1][1], item[0]),
            )
            if score >= 3.8
        ]
        return self._diversify_parent_ids_for_query(
            query,
            list(dict.fromkeys([*promoted, *existing])),
            limit=limit,
        )

    def _diversify_parent_ids_for_query(
        self,
        query: str,
        parent_ids: Sequence[str],
        *,
        limit: int,
    ) -> List[str]:
        ordered = [str(parent_id) for parent_id in parent_ids if str(parent_id) in self.parent_map]
        if not ordered or limit <= 0:
            return []
        query_tokens = set(_tokenize(query))
        legal_relation_tokens = {
            "law",
            "legal",
            "basis",
            "established",
            "created",
            "affiliated",
            "affiliation",
            "authority",
            "institutional",
        }
        should_diversify = classify_query_mode(query) in {QueryMode.SCOPED, QueryMode.SYNTHESIS} or bool(
            query_tokens & legal_relation_tokens
        )
        if not should_diversify:
            return list(dict.fromkeys(ordered))[:limit]

        max_per_source = 2 if query_tokens & legal_relation_tokens else 3
        selected: List[str] = []
        overflow: List[str] = []
        source_counts: Dict[str, int] = {}
        seen = set()
        for parent_id in ordered:
            if parent_id in seen:
                continue
            seen.add(parent_id)
            parent = self.parent_map.get(parent_id) or {}
            source_key = _clean_text(
                str(parent.get("source_url") or parent.get("canonical_url") or parent.get("document_title") or parent_id)
            ).lower().rstrip("/")
            count = source_counts.get(source_key, 0)
            if count < max_per_source:
                selected.append(parent_id)
                source_counts[source_key] = count + 1
            else:
                overflow.append(parent_id)

        return [*selected, *overflow][:limit]

    def _media_parent_hints(self, query: str, media_ids: Sequence[str], *, top_k: int) -> List[str]:
        ranked_media: List[Tuple[str, float, int]] = []
        for rank, media_id in enumerate(media_ids):
            media = self.media_map.get(str(media_id))
            if not media:
                continue
            score = self._score_media_relevance(query, media)
            if score <= 0.0:
                continue
            ranked_media.append((str(media_id), score + (1.0 / float(rank + 1)), rank))
        ranked_media.sort(key=lambda item: (-item[1], item[2], item[0]))
        parent_ids: List[str] = []
        for media_id, _score, _rank in ranked_media[: max(1, top_k)]:
            media = self.media_map.get(media_id) or {}
            parent_ids.extend(
                self._canonical_parent_ids(
                    media.get("linked_parent_ids") or [],
                    chunk_ids=media.get("linked_chunk_ids") or [],
                )
            )
        return list(dict.fromkeys(parent_ids))

    def _attach_media(self, chunk_ids: List[str], media_hits: List[str], query: str) -> List[Dict[str, Any]]:
        wanted_chunk_ids = set(chunk_ids)
        wanted_parent_ids = set()
        media_query = _is_media_query(query)
        chunk_rank = {chunk_id: idx for idx, chunk_id in enumerate(chunk_ids)}
        for chunk_id in chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            wanted_parent_ids.update(self._parent_ids_for_chunk(chunk_id))
        scored_media: Dict[str, float] = {}
        media_rank_hint: Dict[str, int] = {}
        for explicit_rank, media_id in enumerate(media_hits):
            media = self.media_map.get(media_id)
            if not media:
                continue
            if self._is_low_signal_media(media):
                continue
            media_type = str(media.get("media_type") or "")
            media_rank_hint.setdefault(media_id, explicit_rank)
            if media_query:
                relevance = self._score_media_relevance(query, media)
                if relevance > 0.0:
                    # Dense media retrieval embeds grounded caption/OCR/context
                    # directly. Preserve that independent signal even when
                    # chunk expansion has selected another page first.
                    lane_rank_bonus = max(5.0 - (0.40 * explicit_rank), 1.0)
                    scored_media[media_id] = (
                        scored_media.get(media_id, 0.0)
                        + lane_rank_bonus
                        + (2.0 * relevance)
                    )
            if wanted_chunk_ids.intersection(media.get("linked_chunk_ids") or []):
                explicit_bonus = 2.2
                if media_query and media_type == "page_visual":
                    explicit_bonus = 1.5
                if media_query and media_type != "page_visual":
                    explicit_bonus += 1.0
                scored_media[media_id] = scored_media.get(media_id, 0.0) + explicit_bonus
            elif wanted_parent_ids.intersection(
                self._canonical_parent_ids(
                    media.get("linked_parent_ids") or [],
                    chunk_ids=media.get("linked_chunk_ids") or [],
                )
            ):
                parent_bonus = 1.2
                if media_query and media_type == "page_visual":
                    parent_bonus -= 0.35
                scored_media[media_id] = scored_media.get(media_id, 0.0) + parent_bonus
        for chunk_id in chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            for media_position, media_id in enumerate(chunk.get("media_ids") or []):
                media = self.media_map.get(media_id)
                if not media or self._is_low_signal_media(media):
                    continue
                media_type = str(media.get("media_type") or "")
                media_rank_hint.setdefault(media_id, len(media_rank_hint))
                chunk_order_bonus = max(2.0 - (chunk_rank.get(chunk_id, 0) * 0.20), 0.8)
                position_bonus = max(1.8 - (media_position * 0.12), 1.0)
                if media_query:
                    if media_type == "page_visual":
                        chunk_order_bonus *= 0.65
                        position_bonus *= 0.75
                    else:
                        chunk_order_bonus *= 1.15
                        position_bonus *= 1.10
                scored_media[media_id] = scored_media.get(media_id, 0.0) + chunk_order_bonus + position_bonus
        ranked_media = []
        for media_id, base_score in scored_media.items():
            media = self.media_map.get(media_id)
            if not media:
                continue
            relevance = self._score_media_relevance(query, media)
            if relevance <= 0.0:
                continue
            media_type = str(media.get("media_type") or "")
            relevance_weight = 1.0
            if media_query:
                relevance_weight = 2.2 if media_type != "page_visual" else 1.35
            direct_chunk_link = bool(wanted_chunk_ids.intersection(media.get("linked_chunk_ids") or []))
            media_specific_bonus = 0.0
            if media_query and direct_chunk_link and media_type != "page_visual":
                media_specific_bonus += 0.45
            ranked_media.append(
                (
                    media_id,
                    base_score + (relevance_weight * relevance) + media_specific_bonus,
                    media_rank_hint.get(media_id, 10_000),
                )
            )
        ranked_media.sort(key=lambda item: (-item[1], item[2], item[0]))
        return [
            self.media_map[media_id]
            for media_id, _score, _rank_hint in ranked_media[: self.max_media_results]
            if media_id in self.media_map
        ]

    def _has_grounded_media_candidates(
        self,
        *,
        query: str,
        media_rankings: Sequence[Sequence[str]],
    ) -> bool:
        """Allow strong media evidence to survive weak text-chunk abstention.

        PDF and webpage visuals are independently embedded and can answer a
        visual question even when their surrounding prose is too weak to pass
        the text-only chunk guard. Rescue remains fail-closed: the candidate
        must be near the top of a media lane, come from an official source,
        and have both strong overall relevance and direct query overlap.
        """

        if not _is_media_query(query):
            return False
        ranked_lanes = [
            [str(value) for value in ranking if str(value)]
            for ranking in media_rankings
            if ranking
        ]
        if not ranked_lanes:
            return False
        fused_candidate_ids = [
            media_id
            for media_id, _score in _rrf_merge(ranked_lanes, k=self.rrf_k)
        ]
        minimum_score = float(
            getattr(self, "media_abstain_rescue_min_score", 1.25)
        )
        minimum_overlap = float(
            getattr(self, "media_abstain_rescue_min_overlap", 0.35)
        )
        rank_window = max(2, min(4, int(getattr(self, "max_media_results", 4))))
        # RRF can rank a mediocre record repeated in sparse/local lanes ahead
        # of a decisive dense-only hit. Inspect each lane's leading window
        # first, then use fusion for the remaining cross-lane candidates.
        candidate_ids = list(
            dict.fromkeys(
                [
                    media_id
                    for lane in ranked_lanes
                    for media_id in lane[:rank_window]
                ]
                + fused_candidate_ids
            )
        )
        for media_id in candidate_ids[: max(12, rank_window * len(ranked_lanes))]:
            media = self.media_map.get(media_id)
            if not media or self._is_low_signal_media(media):
                continue
            ranks = [
                lane.index(media_id)
                for lane in ranked_lanes
                if media_id in lane
            ]
            if not ranks or (min(ranks) >= rank_window and len(ranks) < 2):
                continue
            source_url = _record_source_url(media)
            if not _OFFICIAL_SOURCE_URL_RE.match(source_url):
                continue
            media_text = self.media_texts_by_id.get(media_id) or _clean_text(
                media.get("text") or ""
            )
            direct_overlap = self._score_text_match(query, media_text)
            relevance = self._score_media_relevance(query, media)
            if relevance >= minimum_score and direct_overlap >= minimum_overlap:
                return True
        return False

    def _should_abstain(
        self,
        *,
        query: str,
        mode: QueryMode,
        ranked_chunks: Sequence[Tuple[str, float]],
        support: Dict[str, Dict[str, Any]],
    ) -> bool:
        if not ranked_chunks:
            return True
        top_chunk_id, top_score = ranked_chunks[0]
        top_chunk = self.chunk_map.get(top_chunk_id)
        if not top_chunk:
            return True
        overlap = self._score_text_match(query, top_chunk.get("dense_text") or top_chunk.get("text") or "")
        fact_overlap = max(
            (
                self._score_text_match(query, fact_text) + self._fact_query_bonus(query, fact_text)
                for fact_text in (self.fact_texts_by_chunk.get(top_chunk_id) or [])[:4]
            ),
            default=0.0,
        )
        answer_overlap = max(
            (
                self._score_text_match(query, answer_text) + _lookup_signal_bonus(query, answer_text)
                for answer_text in (self.answer_texts_by_chunk.get(top_chunk_id) or [])[:4]
            ),
            default=0.0,
        )
        support_score = float(support.get(top_chunk_id, {}).get("score") or 0.0)
        support_sources = support.get(top_chunk_id, {}).get("sources") or set()
        evidence_text = " ".join(
            [
                str(top_chunk.get("dense_text") or top_chunk.get("text") or ""),
                *list(self.answer_texts_by_chunk.get(top_chunk_id) or [])[:2],
                *list(self.fact_texts_by_chunk.get(top_chunk_id) or [])[:2],
            ]
        ).lower()
        if self.temporal_exact_year_guard_enabled and _requires_exact_temporal_evidence(query):
            temporal_fragments: List[str] = []
            for chunk_id, _score in ranked_chunks[: self.temporal_guard_top_k]:
                chunk = self.chunk_map.get(str(chunk_id))
                if not chunk:
                    continue
                temporal_fragments.extend(
                    [
                        str(chunk.get("source_url") or ""),
                        str(chunk.get("document_title") or ""),
                        str(chunk.get("heading") or ""),
                        str(chunk.get("dense_text") or chunk.get("text") or ""),
                    ]
                )
                temporal_fragments.extend(list(self.answer_texts_by_chunk.get(str(chunk_id)) or [])[:2])
                temporal_fragments.extend(list(self.fact_texts_by_chunk.get(str(chunk_id)) or [])[:2])
            if not _text_has_requested_temporal_anchor(query, " ".join(temporal_fragments)):
                return True
        if mode == QueryMode.FACT:
            has_fact_support = bool(
                {"dense_facts", "sparse_facts", "dense_assertions", "sparse_assertions", "local_facts", "local_answers"}
                & set(support_sources)
            )
            named_tokens = _effective_named_tokens_for_abstention(query, evidence_text)
            lookup_profile = _lookup_query_profile(query)
            informative_tokens = self._informative_query_tokens(query)
            required_fact_overlap = self.fact_abstain_min_token_overlap
            if len(informative_tokens) >= 4:
                required_fact_overlap = max(required_fact_overlap, 0.45)
            if set(informative_tokens) & _FACT_ATTRIBUTE_TOKENS:
                required_fact_overlap = max(required_fact_overlap, 0.45)
            evidence_overlap = max(overlap, fact_overlap, answer_overlap)
            structured_fact_overlap = max(fact_overlap, answer_overlap)
            missing_named_ratio = _missing_token_ratio(named_tokens, evidence_text)
            missing_named_phrases = [
                phrase for phrase in _named_query_phrases(query)
                if phrase not in evidence_text and not all(token in evidence_text for token in _tokenize(phrase))
            ]
            source_match = self._source_query_bonus(
                query,
                source_url=str(top_chunk.get("source_url") or ""),
                document_title=str(top_chunk.get("document_title") or ""),
                heading=str(top_chunk.get("heading") or ""),
                text=evidence_text,
                mode=mode,
            )
            legacy_text_only = self.legacy_vectorstore_contract == _LEGACY_VECTORSTORE_CONTRACT
            allow_implicit_subject = (
                lookup_profile.is_exact_lookup
                and not lookup_profile.strict_answer_required
                and len(named_tokens) <= 1
                and not missing_named_phrases
                and has_fact_support
                and structured_fact_overlap >= required_fact_overlap
            )
            if (
                lookup_profile.is_exact_lookup
                and (missing_named_ratio >= 0.5 or missing_named_phrases)
                and not allow_implicit_subject
            ):
                return True
            if evidence_overlap < self.fact_abstain_min_token_overlap and support_score < self.abstain_min_support_score:
                return True
            if len(support_sources) < 2 and evidence_overlap < self.fact_abstain_min_token_overlap:
                return True
            if has_fact_support and structured_fact_overlap >= required_fact_overlap:
                return False
            if legacy_text_only and lookup_profile.is_exact_lookup:
                if evidence_overlap >= self.fact_require_fact_support_overlap:
                    return False
                if source_match >= 0.75 and evidence_overlap >= self.fact_abstain_min_token_overlap:
                    return False
            if named_tokens and any(token not in evidence_text for token in named_tokens):
                return True
            if not has_fact_support and evidence_overlap < self.fact_require_fact_support_overlap:
                return True
            if has_fact_support and structured_fact_overlap < required_fact_overlap and evidence_overlap < required_fact_overlap:
                return True
        else:
            if overlap < self.abstain_min_token_overlap and support_score < self.abstain_min_support_score:
                return True
        return False

    def _has_grounded_navigation_candidates(
        self,
        *,
        query: str,
        page_card_ids: Sequence[str],
        action_ids: Sequence[str],
    ) -> bool:
        navigation = infer_navigation_context(query)
        intent = str(navigation.get("intent") or "none")
        if intent == "none" or not page_card_ids or not action_ids:
            return False
        desired_types = {
            "apply": {"apply", "submit_form"},
            "register": {"register", "submit_form"},
            "contact": {"contact", "email", "telephone", "submit_form"},
            "download": {"download"},
            "login": {"login"},
            "search": {"search"},
            "follow_steps": {
                "apply", "register", "contact", "download", "login", "submit_form"
            },
        }.get(intent, set())
        if not desired_types:
            return False
        top_page_ids = {
            str(value)
            for value in list(page_card_ids)[:3]
            if str(value) in self.page_card_map
        }
        intent_patterns = {
            "apply": r"\b(?:apply|application|submit application)\b|التقديم|تقديم|طلب الالتحاق",
            "register": r"\b(?:register|registration|sign up|enrol|enroll)\b|التسجيل|سج[ّ]?ل",
            "contact": r"\b(?:contact|email|e-mail|phone|telephone|call)\b|تواصل|اتصل|بريد|هاتف",
            "download": r"\b(?:download|pdf|brochure|prospectus)\b|تحميل|تنزيل",
            "login": r"\b(?:log[ -]?in|sign[ -]?in|portal)\b|تسجيل الدخول|بوابة",
            "search": r"\bsearch\b|بحث",
        }
        for action_id in list(action_ids)[:6]:
            action = self.action_map.get(str(action_id))
            if not isinstance(action, Mapping):
                continue
            action_page_ids = {
                str(value)
                for value in (
                    [action.get("page_card_id")]
                    + list(action.get("page_card_ids") or [])
                )
                if str(value)
            }
            if not action_page_ids.intersection(top_page_ids):
                continue
            metadata = action.get("metadata")
            if not isinstance(metadata, Mapping):
                metadata = {}
            action_type = str(
                action.get("action_type") or metadata.get("action_type") or ""
            ).strip().lower()
            if action_type in desired_types:
                return True
            material = " ".join(
                [
                    *(
                        str(action.get(key) or "")
                        for key in (
                            "label",
                            "title",
                            "context_label",
                            "source_section_heading",
                            "raw_text",
                        )
                    ),
                    str(metadata.get("label") or ""),
                    str(metadata.get("context_label") or ""),
                ]
            ).casefold()
            pattern = intent_patterns.get(intent)
            if pattern and re.search(pattern, material, flags=re.IGNORECASE):
                return True
        return False

    def retrieve(
        self,
        query: str,
        *,
        query_vector: Optional[List[float]] = None,
        seed_overrides: Optional[Dict[str, Sequence[str]]] = None,
        skip_query_planner: bool = False,
        mode_override: QueryMode | str | None = None,
    ) -> Dict[str, Any]:
        retrieval_started = time.perf_counter()
        stage_started = retrieval_started
        stage_latency_ms: Dict[str, float] = {}
        request_diagnostics: Dict[str, Any] = {
            "lane_latency_ms": {},
            "dense_lane_scores": {},
            "rerank_latency_ms": 0.0,
            "rerank_method": "",
        }

        def _mark_stage(name: str) -> None:
            nonlocal stage_started
            now = time.perf_counter()
            stage_latency_ms[name] = round((now - stage_started) * 1000.0, 3)
            stage_started = now

        if mode_override is None:
            mode = classify_query_mode(query)
        elif isinstance(mode_override, QueryMode):
            mode = mode_override
        else:
            mode = QueryMode(str(mode_override).strip().lower())
        media_query = _is_media_query(query)
        lookup_profile = _lookup_query_profile(query)
        exact_lookup = mode == QueryMode.FACT and lookup_profile.is_exact_lookup
        legacy_text_only = self.legacy_vectorstore_contract == _LEGACY_VECTORSTORE_CONTRACT
        query_embedding_status = "ok"
        query_embedding_error = ""
        if query_vector is None:
            try:
                query_vector = self.embed_query(query)
            except Exception as exc:
                query_embedding_status = "failed_sparse_local_fallback"
                query_embedding_error = _public_query_embedding_error_code(exc)
                query_vector = []
                logger.warning(
                    "Dense query embedding failed; continuing with sparse/local retrieval fallback: %s",
                    exc,
                )
        else:
            query_vector = list(query_vector)
            if not query_vector:
                query_embedding_status = "skipped_dense_no_query_vector"
        _mark_stage("query_embedding_ms")
        seed_overrides = dict(seed_overrides or {})
        graph_seed_chunk_ids = [str(value) for value in (seed_overrides.get("graph_relation_chunk_ids") or []) if str(value)]
        graph_seed_parent_ids = [str(value) for value in (seed_overrides.get("graph_relation_parent_ids") or []) if str(value)]
        graph_seed_fact_ids = [str(value) for value in (seed_overrides.get("graph_relation_fact_ids") or []) if str(value)]

        lane_top_ks = self._lane_top_ks(query=query, mode=mode, media_query=media_query)
        disable_external_lanes_after_embedding_failure = (
            not query_vector
            and not self.external_lanes_on_embedding_failure
        )
        if not query_vector:
            for dense_lane in (
                "chunk_dense",
                "assertion_dense",
                "parent_dense",
                "summary_dense",
                "media_dense",
                "page_card_dense",
                "action_dense",
                "fact_dense",
                "evidence_span_dense",
            ):
                lane_top_ks[dense_lane] = 0
        if disable_external_lanes_after_embedding_failure:
            for sparse_lane in (
                "chunk_sparse",
                "assertion_sparse",
                "parent_sparse",
                "summary_sparse",
                "media_sparse",
                "fact_sparse",
                "evidence_span_sparse",
            ):
                lane_top_ks[sparse_lane] = 0
        lane_results = self._run_query_lanes(
            query=query,
            query_vector=query_vector,
            lane_top_ks=lane_top_ks,
            mode=mode,
            diagnostics=request_diagnostics,
        )
        _mark_stage("lane_query_ms")
        chunk_dense_ids = lane_results["chunk_dense_ids"]
        sparse_chunk_ids = lane_results["sparse_chunk_ids"]
        parent_dense_ids = lane_results["parent_dense_ids"]
        sparse_parent_ids = lane_results["sparse_parent_ids"]
        local_parent_ids = lane_results["local_parent_ids"]
        summary_dense_ids = lane_results["summary_dense_ids"]
        sparse_summary_ids = lane_results["sparse_summary_ids"]
        media_dense_ids = lane_results["media_dense_ids"]
        sparse_media_ids = lane_results["sparse_media_ids"]
        local_media_ids = lane_results["local_media_ids"]
        dense_page_card_ids = lane_results["dense_page_card_ids"]
        dense_action_ids = lane_results["dense_action_ids"]
        fact_dense_ids = lane_results["fact_dense_ids"]
        sparse_fact_ids = lane_results["sparse_fact_ids"]
        local_fact_ids = lane_results["local_fact_ids"]
        dense_evidence_span_ids = lane_results["dense_evidence_span_ids"]
        sparse_evidence_span_ids = lane_results["sparse_evidence_span_ids"]
        local_evidence_span_ids = lane_results["local_evidence_span_ids"]
        dense_assertion_ids = lane_results["dense_assertion_ids"]
        sparse_assertion_ids = lane_results["sparse_assertion_ids"]
        local_answer_ids = lane_results["local_answer_ids"]
        structured_answer_types = _structured_answer_types(query)
        synthesis_structured_lane_enabled = _synthesis_structured_lane_enabled(
            query,
            mode=mode,
            lookup_profile=lookup_profile,
            structured_answer_types=structured_answer_types,
        )
        structured_answer_query = bool(structured_answer_types) and (
            mode != QueryMode.SYNTHESIS or synthesis_structured_lane_enabled
        )
        fact_selection_query = mode == QueryMode.FACT or synthesis_structured_lane_enabled
        ranked_answer_ids = self._rank_answer_ids(
            query,
            [*dense_assertion_ids, *sparse_assertion_ids, *local_answer_ids],
            top_k=max(4, min(8, self.parent_candidate_top_k + 3)),
        ) if structured_answer_query else []
        answer_anchor_chunk_ids = self._rank_answer_anchor_chunk_ids(
            query,
            ranked_answer_ids or local_answer_ids,
            top_k=max(4, self.parent_candidate_top_k + 1),
        ) if structured_answer_query else []
        ranked_fact_ids = self._rank_fact_ids(
            query,
            [*graph_seed_fact_ids, *local_fact_ids, *sparse_fact_ids, *fact_dense_ids],
            top_k=max(4, self.parent_candidate_top_k + 2),
        ) if fact_selection_query else []
        ranked_evidence_span_ids = [
            span_id
            for span_id, _score in _rrf_merge(
                [dense_evidence_span_ids, sparse_evidence_span_ids, local_evidence_span_ids],
                k=self.rrf_k,
            )
            if span_id in self.evidence_span_map
        ]
        ranked_evidence_span_ids = self._promote_named_source_evidence_spans(
            query,
            ranked_evidence_span_ids,
        )
        evidence_span_anchor_chunk_ids = self._span_anchor_chunk_ids(
            ranked_evidence_span_ids[: max(4, self.parent_candidate_top_k + 3)]
        )
        fact_anchor_chunk_ids = self._rank_fact_anchor_chunk_ids(
            query,
            ranked_fact_ids or [*local_fact_ids, *sparse_fact_ids, *fact_dense_ids],
            top_k=max(4, self.parent_candidate_top_k + 1),
        ) if fact_selection_query else []
        local_chunk_ids = lane_results["local_chunk_ids"]
        if mode == QueryMode.SYNTHESIS and (chunk_dense_ids or sparse_chunk_ids):
            local_chunk_ids = []
        if mode == QueryMode.FACT and not legacy_text_only and not any((fact_dense_ids, sparse_fact_ids, local_fact_ids)):
            local_chunk_ids = []
        support_parent_ids = self._support_parent_ids_for_query(
            query,
            top_k=max(3, self.parent_candidate_top_k + 2),
        )
        support_parent_chunk_ids = self._support_chunk_ids_for_query(
            query,
            support_parent_ids,
            top_k=max(6, min(self.max_context_chunks, self.parent_candidate_top_k * 3 + 3)),
        )

        rankings = {
            "dense_chunks": chunk_dense_ids,
            "sparse_chunks": sparse_chunk_ids,
            "local_chunks": local_chunk_ids,
            "dense_assertions": dense_assertion_ids,
            "sparse_assertions": sparse_assertion_ids,
            "local_answers": local_answer_ids,
            "graph_relation_chunks": graph_seed_chunk_ids,
            "dense_parents": parent_dense_ids,
            "sparse_parents": sparse_parent_ids,
            "local_parents": local_parent_ids,
            "support_parents": support_parent_ids,
            "graph_relation_parents": graph_seed_parent_ids,
            "dense_summaries": summary_dense_ids,
            "sparse_summaries": sparse_summary_ids,
            "dense_media": media_dense_ids,
            "sparse_media": sparse_media_ids,
            "local_media": local_media_ids,
            "dense_page_cards": dense_page_card_ids,
            "dense_actions": dense_action_ids,
            "dense_facts": fact_dense_ids,
            "sparse_facts": sparse_fact_ids,
            "local_facts": local_fact_ids,
            "dense_evidence_spans": dense_evidence_span_ids,
            "sparse_evidence_spans": sparse_evidence_span_ids,
            "local_evidence_spans": local_evidence_span_ids,
            "graph_relation_facts": graph_seed_fact_ids,
        }
        fused = _rrf_merge(
            [ranking for ranking in rankings.values() if ranking],
            k=self.rrf_k,
        )
        fused_ids = [record_id for record_id, _score in fused]
        support = self._accumulate_chunk_support(rankings)
        candidate_chunk_ids = self._seed_chunk_ids(fused_ids[: max(self.dense_chunk_top_k, self.sparse_chunk_top_k, 24)])
        if evidence_span_anchor_chunk_ids:
            candidate_chunk_ids = list(dict.fromkeys([*evidence_span_anchor_chunk_ids, *candidate_chunk_ids]))
        if support_parent_chunk_ids:
            candidate_chunk_ids = list(dict.fromkeys([*support_parent_chunk_ids, *candidate_chunk_ids]))
        if mode == QueryMode.SYNTHESIS and (summary_dense_ids or sparse_summary_ids):
            summary_seed_chunk_ids = self._seed_chunk_ids([*summary_dense_ids, *sparse_summary_ids])
            candidate_chunk_ids = list(dict.fromkeys([*summary_seed_chunk_ids, *candidate_chunk_ids]))
        if graph_seed_chunk_ids:
            candidate_chunk_ids = list(dict.fromkeys([*graph_seed_chunk_ids, *candidate_chunk_ids]))
        if media_query:
            media_seed_chunk_ids = self._seed_chunk_ids([*local_media_ids, *sparse_media_ids, *media_dense_ids])
            candidate_chunk_ids = list(dict.fromkeys([*media_seed_chunk_ids, *candidate_chunk_ids]))
        if mode == QueryMode.FACT:
            fact_seed_chunk_ids = list(
                dict.fromkeys(
                    [
                        *graph_seed_chunk_ids,
                        *answer_anchor_chunk_ids,
                        *fact_anchor_chunk_ids,
                        *evidence_span_anchor_chunk_ids,
                        *self._seed_chunk_ids(
                            [
                                *dense_assertion_ids,
                                *sparse_assertion_ids,
                                *local_answer_ids,
                                *graph_seed_fact_ids,
                                *fact_dense_ids,
                                *sparse_fact_ids,
                                *local_fact_ids,
                                *dense_evidence_span_ids,
                                *sparse_evidence_span_ids,
                                *local_evidence_span_ids,
                            ]
                        ),
                    ]
                )
            )
            candidate_chunk_ids = list(dict.fromkeys([*fact_seed_chunk_ids, *candidate_chunk_ids]))
        elif fact_selection_query and fact_anchor_chunk_ids:
            candidate_chunk_ids = list(dict.fromkeys([*fact_anchor_chunk_ids, *candidate_chunk_ids]))
        elif structured_answer_query and answer_anchor_chunk_ids:
            candidate_chunk_ids = list(
                dict.fromkeys(
                    [
                        *answer_anchor_chunk_ids,
                        *self._seed_chunk_ids([*dense_assertion_ids, *sparse_assertion_ids, *local_answer_ids]),
                        *candidate_chunk_ids,
                    ]
                )
            )
        ranked_chunks = self._rerank_chunk_candidates(
            query,
            candidate_chunk_ids,
            support,
            mode=mode,
            graph_seed_chunk_ids=graph_seed_chunk_ids,
            graph_seed_fact_ids=graph_seed_fact_ids,
            force_fallback=disable_external_lanes_after_embedding_failure,
            diagnostics=request_diagnostics,
        )
        ranked_chunks = self._promote_source_matched_chunks(query, ranked_chunks, support)
        if mode == QueryMode.FACT:
            ranked_chunks = self._promote_fact_supported_chunks(
                query,
                ranked_chunks,
                support,
                anchored_chunk_ids=[*answer_anchor_chunk_ids, *fact_anchor_chunk_ids],
            )
        if media_query:
            ranked_chunks = self._promote_media_supported_chunks(
                query,
                ranked_chunks,
                [*local_media_ids, *sparse_media_ids, *media_dense_ids],
            )
        _mark_stage("candidate_ranking_ms")
        ranked_chunk_ids = [chunk_id for chunk_id, _score in ranked_chunks]
        if fact_selection_query and (answer_anchor_chunk_ids or fact_anchor_chunk_ids):
            if _is_generic_contact_query(query):
                ranked_chunk_ids = list(dict.fromkeys([*answer_anchor_chunk_ids, *fact_anchor_chunk_ids, *ranked_chunk_ids]))
            else:
                ranked_chunk_ids = list(dict.fromkeys([*fact_anchor_chunk_ids, *answer_anchor_chunk_ids, *ranked_chunk_ids]))
        seed_chunk_ids = ranked_chunk_ids[: max(self.dense_chunk_top_k, self.sparse_chunk_top_k, 12)]

        should_abstain = self._should_abstain(
            query=query,
            mode=mode,
            ranked_chunks=ranked_chunks,
            support=support,
        )
        navigation_evidence_rescued = bool(
            should_abstain
            and self._has_grounded_navigation_candidates(
                query=query,
                page_card_ids=dense_page_card_ids,
                action_ids=dense_action_ids,
            )
        )
        media_evidence_rescued = bool(
            should_abstain
            and self._has_grounded_media_candidates(
                query=query,
                media_rankings=(
                    media_dense_ids,
                    sparse_media_ids,
                    local_media_ids,
                ),
            )
        )
        if should_abstain and not navigation_evidence_rescued and not media_evidence_rescued:
            return {
                "query": query,
                "mode": mode.value,
                "seed_chunk_ids": [],
                "selected_chunk_ids": [],
                "dense_chunk_ids": [],
                "sparse_chunk_ids": [],
                "local_chunk_ids": [],
                "dense_assertion_ids": [],
                "sparse_assertion_ids": [],
                "local_answer_ids": [],
                "dense_parent_ids": [],
                "sparse_parent_ids": [],
                "dense_summary_ids": [],
                "sparse_summary_ids": [],
                "dense_media_ids": [],
                "sparse_media_ids": [],
                "local_media_ids": [],
                "dense_page_card_ids": [],
                "dense_action_ids": [],
                "dense_fact_ids": [],
                "sparse_fact_ids": [],
                "local_fact_ids": [],
                "dense_evidence_span_ids": [],
                "sparse_evidence_span_ids": [],
                "local_evidence_span_ids": [],
                "selected_evidence_span_ids": [],
                "selected_answer_ids": [],
                "answer_documents": [],
                "graph_relation_chunk_ids": graph_seed_chunk_ids,
                "graph_relation_parent_ids": graph_seed_parent_ids,
                "graph_relation_fact_ids": graph_seed_fact_ids,
                "selected_fact_ids": [],
                "fact_documents": [],
                "evidence_span_documents": [],
                "selected_parent_ids": [],
                "selected_media_ids": [],
                "retrieval_documents": [],
                "media": [],
                "abstained": True,
                "media_evidence_rescued": False,
                "query_embedding_status": query_embedding_status,
                "query_embedding_error": query_embedding_error,
                "lane_latency_ms": dict(request_diagnostics.get("lane_latency_ms") or {}),
                "dense_lane_scores": dict(request_diagnostics.get("dense_lane_scores") or {}),
                "rerank_latency_ms": float(request_diagnostics.get("rerank_latency_ms") or 0.0),
                "rerank_method": str(request_diagnostics.get("rerank_method") or ""),
                "stage_latency_ms": dict(stage_latency_ms),
                "vector_backend_latency_ms": round((time.perf_counter() - retrieval_started) * 1000.0, 3),
                "debug_candidates": {
                    "dense_chunk_ids": chunk_dense_ids,
                    "sparse_chunk_ids": sparse_chunk_ids,
                    "local_chunk_ids": local_chunk_ids,
                    "dense_assertion_ids": dense_assertion_ids,
                    "sparse_assertion_ids": sparse_assertion_ids,
                    "local_answer_ids": local_answer_ids,
                    "dense_parent_ids": parent_dense_ids,
                    "sparse_parent_ids": sparse_parent_ids,
                    "local_parent_ids": local_parent_ids,
                    "support_parent_ids": support_parent_ids,
                    "support_parent_chunk_ids": support_parent_chunk_ids,
                    "dense_summary_ids": summary_dense_ids,
                    "sparse_summary_ids": sparse_summary_ids,
                    "dense_media_ids": media_dense_ids,
                    "sparse_media_ids": sparse_media_ids,
                    "local_media_ids": local_media_ids,
                    "dense_page_card_ids": dense_page_card_ids,
                    "dense_action_ids": dense_action_ids,
                    "dense_fact_ids": fact_dense_ids,
                    "sparse_fact_ids": sparse_fact_ids,
                    "local_fact_ids": local_fact_ids,
                    "dense_evidence_span_ids": dense_evidence_span_ids,
                    "sparse_evidence_span_ids": sparse_evidence_span_ids,
                    "local_evidence_span_ids": local_evidence_span_ids,
                    "graph_relation_chunk_ids": graph_seed_chunk_ids,
                    "graph_relation_parent_ids": graph_seed_parent_ids,
                    "graph_relation_fact_ids": graph_seed_fact_ids,
                },
                "response_agent_instructions": response_agent_media_instructions(),
            }

        selected_fact_ids = (
            ranked_fact_ids[: max(3, min(6, self.parent_candidate_top_k + 1))]
            if fact_selection_query
            else []
        )
        selected_evidence_span_ids = self._select_evidence_span_ids_for_query(
            query,
            ranked_evidence_span_ids,
            mode=mode,
        )
        selected_answer_ids = (
            self._select_answer_ids_for_query(
                query,
                ranked_answer_ids,
                top_k=max(3, min(6, self.parent_candidate_top_k + 2)),
            )
            if structured_answer_query
            else []
        )
        explicit_parent_ids = self._rank_parent_candidates(
            query,
            [*support_parent_ids, *graph_seed_parent_ids, *local_parent_ids, *parent_dense_ids, *sparse_parent_ids],
        )
        if media_query:
            media_parent_ids = self._media_parent_hints(
                query,
                [*local_media_ids, *sparse_media_ids, *media_dense_ids],
                top_k=max(2, self.parent_candidate_top_k),
            )
            if media_parent_ids:
                explicit_parent_ids = self._rank_parent_candidates(query, media_parent_ids)
        if mode == QueryMode.FACT:
            selected_chunk_ids = self._expand_fact(seed_chunk_ids)
        else:
            selected_chunk_ids = self._expand_scoped_or_synthesis(
                seed_chunk_ids,
                query=query,
                mode=mode,
                explicit_parent_ids=explicit_parent_ids,
                prioritize_explicit_parents=bool(explicit_parent_ids) and (media_query or mode == QueryMode.SCOPED),
                prefer_explicit_parent_chunks_first=self._should_prefer_explicit_parent_chunks(
                    query,
                    mode=mode,
                    media_query=media_query,
                ),
            )
        if fact_selection_query and (selected_answer_ids or selected_fact_ids):
            if _is_generic_contact_query(query) or _lookup_query_profile(query).is_exact_lookup:
                promoted_chunk_ids = [
                    *self._rank_answer_anchor_chunk_ids(
                        query,
                        selected_answer_ids,
                        top_k=max(4, self.parent_candidate_top_k + 1),
                    ),
                    *self._rank_fact_anchor_chunk_ids(
                        query,
                        selected_fact_ids,
                        top_k=max(4, self.parent_candidate_top_k + 1),
                    ),
                ]
            else:
                promoted_chunk_ids = [
                    *self._rank_fact_anchor_chunk_ids(
                        query,
                        selected_fact_ids,
                        top_k=max(4, self.parent_candidate_top_k + 1),
                    ),
                    *self._rank_answer_anchor_chunk_ids(
                        query,
                        selected_answer_ids,
                        top_k=max(4, self.parent_candidate_top_k + 1),
                    ),
                ]
            if promoted_chunk_ids:
                selected_chunk_ids = list(dict.fromkeys([*promoted_chunk_ids, *selected_chunk_ids]))
        if selected_evidence_span_ids:
            selected_chunk_ids = list(
                dict.fromkeys([*self._span_anchor_chunk_ids(selected_evidence_span_ids), *selected_chunk_ids])
            )
        selected_chunk_ids = self._preserve_dense_chunk_recall(
            selected_chunk_ids,
            chunk_dense_ids,
        )
        selected_parent_ids = self._select_parent_ids(
            selected_chunk_ids,
            explicit_parent_ids=explicit_parent_ids,
            prefer_explicit_parents=bool(explicit_parent_ids) and (mode == QueryMode.SCOPED or media_query),
        )
        selected_parent_ids = self._promote_selected_chunk_parent_ids(
            query,
            selected_parent_ids,
            selected_chunk_ids,
        )
        selected_parent_ids = self._preserve_dense_parent_recall(
            selected_parent_ids,
            chunk_dense_ids,
        )
        _mark_stage("selection_expansion_ms")

        explicit_media_hits = list(dict.fromkeys([*media_dense_ids, *sparse_media_ids, *local_media_ids]))
        selected_media = self._attach_media(selected_chunk_ids, explicit_media_hits, query)
        selected_media_ids = [str(media.get("id") or "") for media in selected_media if str(media.get("id") or "")]
        selected_docs = []
        for chunk_id in selected_chunk_ids:
            chunk = self.chunk_map.get(chunk_id)
            if not chunk:
                continue
            chunk_media = [
                media
                for media in selected_media
                if (
                    chunk_id in (media.get("linked_chunk_ids") or [])
                    or bool(
                        set(self._parent_ids_for_chunk(chunk_id))
                        & set(
                            self._canonical_parent_ids(
                                media.get("linked_parent_ids") or [],
                                chunk_ids=media.get("linked_chunk_ids") or [],
                            )
                        )
                    )
                )
            ]
            selected_docs.append(
                {
                    "id": chunk["id"],
                    "text": chunk.get("dense_text") or chunk.get("text") or "",
                    "metadata": {
                        "document_source": chunk.get("source_url") or "",
                        "document_title": chunk.get("document_title") or "",
                        "document_summary": "",
                        "media": json.dumps(
                            [
                                {
                                    "type": media.get("media_type", "image"),
                                    "url": media.get("url", ""),
                                    "asset_uri": media.get("asset_uri", ""),
                                    "title": media.get("title", ""),
                                    "caption": media.get("caption", ""),
                                    "description": media.get("description", ""),
                                    "context": media.get("context", ""),
                                    "transcript": media.get("transcript", ""),
                                }
                                for media in chunk_media
                            ]
                        ),
                    },
                }
            )

        retrieval_payload = build_retrieval_documents(
            selected_docs,
            max_media_per_doc=self.max_media_results,
            max_total_media=self.max_media_results,
        )
        answer_payload: List[Dict[str, Any]] = []
        if selected_answer_ids:
            for answer_id in selected_answer_ids:
                answer = self.answer_map.get(str(answer_id))
                if not answer:
                    continue
                linked_chunk = None
                for chunk_id in (answer.get("linked_chunk_ids") or []):
                    linked_chunk = self.chunk_map.get(str(chunk_id))
                    if linked_chunk:
                        break
                if linked_chunk is None:
                    for fact_id in (answer.get("linked_fact_ids") or []):
                        fact = self.fact_map.get(str(fact_id)) or {}
                        for chunk_id in fact.get("linked_chunk_ids") or []:
                            linked_chunk = self.chunk_map.get(str(chunk_id))
                            if linked_chunk:
                                break
                        if linked_chunk:
                            break
                answer_payload.append(
                    {
                        "id": str(answer.get("id") or answer_id),
                        "text": str(answer.get("text") or answer.get("value") or ""),
                        "value": str(answer.get("value") or ""),
                        "answer_type": str(answer.get("answer_type") or ""),
                        "answer_subtype": str(answer.get("answer_subtype") or ""),
                        "confidence": float(answer.get("confidence") or 0.0),
                        "subject_text": str(answer.get("subject_text") or ""),
                        "qualifiers": list(answer.get("qualifiers") or []),
                        "linked_chunk_ids": [str(value) for value in (answer.get("linked_chunk_ids") or []) if str(value)],
                        "linked_span_ids": [str(value) for value in (answer.get("linked_span_ids") or answer.get("source_span_ids") or []) if str(value)],
                        "source_span_ids": [str(value) for value in (answer.get("source_span_ids") or answer.get("linked_span_ids") or []) if str(value)],
                        "linked_fact_ids": [str(value) for value in (answer.get("linked_fact_ids") or []) if str(value)],
                        "linked_parent_ids": self._canonical_parent_ids(
                            answer.get("linked_parent_ids") or [],
                            chunk_ids=answer.get("linked_chunk_ids") or [],
                        ),
                        "source_url": _record_source_url(answer, linked_chunk),
                        "document_title": str(answer.get("document_title") or (linked_chunk or {}).get("document_title") or ""),
                        "document_summary": "",
                        "media": [],
                    }
                )
        fact_payload: List[Dict[str, Any]] = []
        if selected_fact_ids:
            for fact_id in selected_fact_ids:
                fact = self.fact_map.get(str(fact_id))
                if not fact:
                    continue
                linked_chunk = None
                for chunk_id in (fact.get("linked_chunk_ids") or []):
                    linked_chunk = self.chunk_map.get(str(chunk_id))
                    if linked_chunk:
                        break
                fact_payload.append(
                    {
                        "id": str(fact.get("id") or fact_id),
                        "text": str(fact.get("text") or fact.get("dense_text") or ""),
                        "source_url": _record_source_url(fact, linked_chunk),
                        "document_title": str((linked_chunk or {}).get("document_title") or ""),
                        "document_summary": "",
                        "media": [],
                    }
                )
        evidence_span_payload: List[Dict[str, Any]] = []
        if selected_evidence_span_ids:
            for span_id in selected_evidence_span_ids:
                span = self.evidence_span_map.get(str(span_id))
                if not span or str(span.get("validity_status") or "active") == "quarantined":
                    continue
                linked_chunk = None
                for chunk_id in [
                    *[str(value) for value in (span.get("linked_chunk_ids") or []) if str(value)],
                    str(span.get("chunk_id") or ""),
                ]:
                    if not chunk_id:
                        continue
                    linked_chunk = self.chunk_map.get(chunk_id)
                    if linked_chunk:
                        break
                evidence_span_payload.append(
                    {
                        "id": str(span.get("id") or span_id),
                        "text": str(span.get("text") or span.get("dense_text") or ""),
                        "span_type": str(span.get("span_type") or "general"),
                        "source_url": _record_source_url(span, linked_chunk),
                        "canonical_url": str(span.get("canonical_url") or ""),
                        "document_title": str(span.get("document_title") or (linked_chunk or {}).get("document_title") or ""),
                        "section_heading": str(span.get("section_heading") or span.get("heading") or ""),
                        "breadcrumb": str(span.get("breadcrumb") or ""),
                        "linked_chunk_ids": [str(value) for value in (span.get("linked_chunk_ids") or []) if str(value)],
                        "linked_parent_ids": self._canonical_parent_ids(
                            span.get("linked_parent_ids") or [],
                            chunk_ids=span.get("linked_chunk_ids") or [],
                        ),
                        "page_id": str(span.get("page_id") or span.get("page_key") or ""),
                        "section_id": str(span.get("section_id") or span.get("section_key") or ""),
                        "authority_class": str(span.get("authority_class") or ""),
                        "source_last_seen": str(span.get("source_last_seen") or ""),
                        "validity_status": str(span.get("validity_status") or "active"),
                        "document_summary": "",
                        "media": [],
                    }
                )
        if answer_payload or fact_payload or evidence_span_payload:
            retrieval_payload = [*answer_payload, *fact_payload, *evidence_span_payload, *retrieval_payload]
        chunk_source_fallbacks: Dict[str, Dict[str, str]] = {}
        for span_id in selected_evidence_span_ids:
            span = self.evidence_span_map.get(str(span_id))
            if not isinstance(span, Mapping):
                continue
            source_url = _record_source_url(span)
            if not source_url:
                continue
            for chunk_id in [
                *[str(value) for value in (span.get("linked_chunk_ids") or []) if str(value)],
                str(span.get("chunk_id") or ""),
            ]:
                if chunk_id:
                    chunk_source_fallbacks.setdefault(chunk_id, {"source_url": source_url})
        for document in [*answer_payload, *fact_payload, *evidence_span_payload, *retrieval_payload]:
            if isinstance(document, MutableMapping):
                document_id = str(document.get("id") or "")
                fallback_chunk = self.chunk_map.get(document_id) if hasattr(self, "chunk_map") else None
                _ensure_record_source_url(document, chunk_source_fallbacks.get(document_id), fallback_chunk)
        _mark_stage("payload_build_ms")

        return {
            "query": query,
            "mode": mode.value,
            "seed_chunk_ids": seed_chunk_ids,
            "selected_chunk_ids": selected_chunk_ids,
            "selected_answer_ids": selected_answer_ids,
            "selected_fact_ids": selected_fact_ids,
            "selected_evidence_span_ids": selected_evidence_span_ids,
            "selected_parent_ids": selected_parent_ids,
            "selected_media_ids": selected_media_ids,
            "dense_chunk_ids": chunk_dense_ids,
            "sparse_chunk_ids": sparse_chunk_ids,
            "local_chunk_ids": local_chunk_ids,
            "dense_assertion_ids": dense_assertion_ids,
            "sparse_assertion_ids": sparse_assertion_ids,
            "local_answer_ids": local_answer_ids,
            "dense_parent_ids": parent_dense_ids,
            "sparse_parent_ids": sparse_parent_ids,
            "support_parent_ids": support_parent_ids,
            "support_parent_chunk_ids": support_parent_chunk_ids,
            "dense_summary_ids": summary_dense_ids,
            "sparse_summary_ids": sparse_summary_ids,
            "dense_media_ids": media_dense_ids,
            "sparse_media_ids": sparse_media_ids,
            "local_media_ids": local_media_ids,
            "dense_page_card_ids": dense_page_card_ids,
            "dense_action_ids": dense_action_ids,
            "dense_fact_ids": fact_dense_ids,
            "sparse_fact_ids": sparse_fact_ids,
            "local_fact_ids": local_fact_ids,
            "dense_evidence_span_ids": dense_evidence_span_ids,
            "sparse_evidence_span_ids": sparse_evidence_span_ids,
            "local_evidence_span_ids": local_evidence_span_ids,
            "graph_relation_chunk_ids": graph_seed_chunk_ids,
            "graph_relation_parent_ids": graph_seed_parent_ids,
            "graph_relation_fact_ids": graph_seed_fact_ids,
            "answer_documents": answer_payload,
            "fact_documents": fact_payload,
            "evidence_span_documents": evidence_span_payload,
            "retrieval_documents": retrieval_payload,
            "media": selected_media[: self.max_media_results],
            "abstained": False,
            "navigation_evidence_rescued": navigation_evidence_rescued,
            "media_evidence_rescued": media_evidence_rescued,
            "query_embedding_status": query_embedding_status,
            "query_embedding_error": query_embedding_error,
            "lane_latency_ms": dict(request_diagnostics.get("lane_latency_ms") or {}),
            "dense_lane_scores": dict(request_diagnostics.get("dense_lane_scores") or {}),
            "rerank_latency_ms": float(request_diagnostics.get("rerank_latency_ms") or 0.0),
            "rerank_method": str(request_diagnostics.get("rerank_method") or ""),
            "stage_latency_ms": dict(stage_latency_ms),
            "vector_backend_latency_ms": round((time.perf_counter() - retrieval_started) * 1000.0, 3),
            "response_agent_instructions": response_agent_media_instructions(),
        }

    def _lane_top_ks(self, *, query: str, mode: QueryMode, media_query: bool) -> Dict[str, int]:
        lookup_profile = _lookup_query_profile(query)
        exact_lookup = mode == QueryMode.FACT and lookup_profile.is_exact_lookup
        legacy_text_only = self.legacy_vectorstore_contract == _LEGACY_VECTORSTORE_CONTRACT
        structured_answer_types = _structured_answer_types(query)
        synthesis_structured_lane_enabled = _synthesis_structured_lane_enabled(
            query,
            mode=mode,
            lookup_profile=lookup_profile,
            structured_answer_types=structured_answer_types,
        )
        answer_lane_enabled = bool(self.answer_map) and bool(structured_answer_types) and (
            mode != QueryMode.SYNTHESIS or synthesis_structured_lane_enabled
        )
        parent_lane_enabled = bool(self.parent_map) and mode != QueryMode.FACT and not legacy_text_only
        summary_lane_enabled = bool(self.summary_map) and mode == QueryMode.SYNTHESIS and not legacy_text_only
        local_parent_lane_enabled = self._should_use_local_parent_lane(query, mode=mode)
        media_lane_enabled = bool(self.media_map) and (media_query or mode == QueryMode.SYNTHESIS) and not legacy_text_only
        fact_lane_enabled = bool(self.fact_map) and (
            mode != QueryMode.SYNTHESIS or synthesis_structured_lane_enabled
        ) and not legacy_text_only
        evidence_span_lane_enabled = bool(self.evidence_span_map) and not legacy_text_only
        page_card_lane_enabled = bool(self.page_card_map) and not legacy_text_only
        navigation_intent = str(infer_navigation_context(query).get("intent") or "none")
        action_lane_enabled = bool(self.action_map) and not legacy_text_only
        answer_local_top_k = self.local_answer_top_k if answer_lane_enabled else 0
        local_chunk_top_k = self.sparse_chunk_top_k if mode != QueryMode.SYNTHESIS else max(self.sparse_chunk_top_k, self.dense_chunk_top_k)
        if exact_lookup:
            local_chunk_top_k = max(4, min(local_chunk_top_k, 6))
            answer_local_top_k = max(answer_local_top_k, 10)
        scoped_fact_top_k = max(1, self.dense_fact_top_k // 2)
        scoped_sparse_fact_top_k = max(1, self.sparse_fact_top_k // 2)
        chunk_dense_top_k = self.dense_chunk_top_k
        chunk_sparse_top_k = self.sparse_chunk_top_k
        fact_dense_top_k = self.dense_fact_top_k
        fact_sparse_top_k = self.sparse_fact_top_k
        fact_local_top_k = self.sparse_fact_top_k
        evidence_span_dense_top_k = self.dense_evidence_span_top_k
        evidence_span_sparse_top_k = self.sparse_evidence_span_top_k
        evidence_span_local_top_k = self.sparse_evidence_span_top_k
        assertion_dense_top_k = self.dense_assertion_top_k
        assertion_sparse_top_k = self.sparse_assertion_top_k
        if exact_lookup:
            chunk_dense_top_k = max(4, min(self.dense_chunk_top_k, 8))
            chunk_sparse_top_k = max(4, min(self.sparse_chunk_top_k, 8))
            fact_dense_top_k = max(self.dense_fact_top_k, 12)
            fact_sparse_top_k = max(self.sparse_fact_top_k, 12)
            fact_local_top_k = max(self.sparse_fact_top_k, 12)
            evidence_span_dense_top_k = max(self.dense_evidence_span_top_k, 12)
            evidence_span_sparse_top_k = max(self.sparse_evidence_span_top_k, 12)
            evidence_span_local_top_k = max(self.sparse_evidence_span_top_k, 12)
            assertion_dense_top_k = max(self.dense_assertion_top_k, 12)
            assertion_sparse_top_k = max(self.sparse_assertion_top_k, 12)
        elif mode == QueryMode.SYNTHESIS:
            evidence_span_dense_top_k = max(4, self.dense_evidence_span_top_k // 2)
            evidence_span_sparse_top_k = max(4, self.sparse_evidence_span_top_k // 2)
            evidence_span_local_top_k = max(4, self.sparse_evidence_span_top_k // 2)
        return {
            "chunk_dense": chunk_dense_top_k,
            "chunk_sparse": chunk_sparse_top_k,
            "chunk_local": local_chunk_top_k,
            "assertion_dense": assertion_dense_top_k if answer_lane_enabled and not legacy_text_only else 0,
            "assertion_sparse": assertion_sparse_top_k if answer_lane_enabled and not legacy_text_only else 0,
            "answer_local": answer_local_top_k,
            "parent_dense": self.dense_parent_top_k if parent_lane_enabled else 0,
            "parent_sparse": self.sparse_parent_top_k if parent_lane_enabled else 0,
            "parent_local": self.sparse_parent_top_k if local_parent_lane_enabled else 0,
            "summary_dense": self.dense_summary_top_k if summary_lane_enabled else 0,
            "summary_sparse": self.sparse_summary_top_k if summary_lane_enabled else 0,
            "media_dense": self.dense_media_top_k if media_lane_enabled else 0,
            "media_sparse": self.sparse_media_top_k if media_lane_enabled else 0,
            "media_local": self.sparse_media_top_k if media_lane_enabled else 0,
            "page_card_dense": self.dense_page_card_top_k if page_card_lane_enabled else 0,
            "action_dense": (
                self.dense_action_top_k
                if action_lane_enabled and navigation_intent != "none"
                else max(1, self.dense_action_top_k // 3) if action_lane_enabled else 0
            ),
            "fact_dense": (
                fact_dense_top_k
                if fact_lane_enabled and mode == QueryMode.FACT
                else scoped_fact_top_k if fact_lane_enabled else 0
            ),
            "fact_sparse": (
                fact_sparse_top_k
                if fact_lane_enabled and mode == QueryMode.FACT
                else scoped_sparse_fact_top_k if fact_lane_enabled else 0
            ),
            "fact_local": (
                fact_local_top_k
                if fact_lane_enabled and mode == QueryMode.FACT
                else scoped_sparse_fact_top_k if fact_lane_enabled else 0
            ),
            "evidence_span_dense": evidence_span_dense_top_k if evidence_span_lane_enabled else 0,
            "evidence_span_sparse": evidence_span_sparse_top_k if evidence_span_lane_enabled else 0,
            "evidence_span_local": evidence_span_local_top_k if evidence_span_lane_enabled else 0,
        }

    def _run_lane_task(self, name: str, func, **kwargs) -> Tuple[str, List[str], float]:
        started = time.perf_counter()
        try:
            return name, list(func(**kwargs)), round((time.perf_counter() - started) * 1000.0, 3)
        except Exception as exc:
            logger.warning("Retrieval lane %s failed: %s", name, exc)
            return name, [], round((time.perf_counter() - started) * 1000.0, 3)

    def _run_query_lanes(
        self,
        *,
        query: str,
        query_vector: List[float],
        lane_top_ks: Dict[str, int],
        mode: QueryMode,
        diagnostics: Dict[str, Any] | None = None,
    ) -> Dict[str, List[str]]:
        dense_lane_scores: Dict[str, List[Dict[str, Any]]] = {}

        def dense_kwargs(
            *,
            namespace: str,
            top_k: int,
            score_key: str,
        ) -> Dict[str, Any]:
            return {
                "query_vector": query_vector,
                "namespace": namespace,
                "top_k": top_k,
                "query": query,
                "score_sink": dense_lane_scores,
                "score_key": score_key,
            }

        tasks: Dict[str, Tuple[Any, Dict[str, Any]]] = {
            "chunk_dense_ids": (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_chunks,
                    top_k=lane_top_ks["chunk_dense"],
                    score_key="chunk_dense_ids",
                ),
            ),
            "sparse_chunk_ids": (
                self._sparse_query_ids,
                {"namespace": self.namespace_chunks, "query": query, "top_k": lane_top_ks["chunk_sparse"]},
            ),
            "local_chunk_ids": (
                self._local_chunk_query_ids,
                {"query": query, "top_k": lane_top_ks["chunk_local"]},
            ),
            "local_answer_ids": (
                self._local_answer_query_ids,
                {"query": query, "top_k": lane_top_ks["answer_local"]},
            ),
            "dense_assertion_ids": (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_assertions,
                    top_k=lane_top_ks["assertion_dense"],
                    score_key="dense_assertion_ids",
                ),
            ),
            "sparse_assertion_ids": (
                self._sparse_query_ids,
                {"namespace": self.namespace_assertions, "query": query, "top_k": lane_top_ks["assertion_sparse"]},
            ),
        }
        if self.parent_map:
            tasks["parent_dense_ids"] = (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_parents,
                    top_k=lane_top_ks["parent_dense"],
                    score_key="parent_dense_ids",
                ),
            )
            tasks["sparse_parent_ids"] = (
                self._sparse_query_ids,
                {"namespace": self.namespace_parents, "query": query, "top_k": lane_top_ks["parent_sparse"]},
            )
            tasks["local_parent_ids"] = (
                self._local_parent_query_ids,
                {"query": query, "top_k": lane_top_ks["parent_local"]},
            )
        if self.summary_map:
            tasks["summary_dense_ids"] = (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_summaries,
                    top_k=lane_top_ks["summary_dense"],
                    score_key="summary_dense_ids",
                ),
            )
            tasks["sparse_summary_ids"] = (
                self._sparse_query_ids,
                {"namespace": self.namespace_summaries, "query": query, "top_k": lane_top_ks["summary_sparse"]},
            )
        if self.media_map:
            tasks["media_dense_ids"] = (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_media,
                    top_k=lane_top_ks["media_dense"],
                    score_key="media_dense_ids",
                ),
            )
            tasks["sparse_media_ids"] = (
                self._sparse_query_ids,
                {"namespace": self.namespace_media, "query": query, "top_k": lane_top_ks["media_sparse"]},
            )
            tasks["local_media_ids"] = (
                self._local_media_query_ids,
                {"query": query, "top_k": lane_top_ks["media_local"]},
            )
        if getattr(self, "page_card_map", None):
            tasks["dense_page_card_ids"] = (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_page_cards,
                    top_k=lane_top_ks.get("page_card_dense", 0),
                    score_key="dense_page_card_ids",
                ),
            )
        if getattr(self, "action_map", None):
            tasks["dense_action_ids"] = (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_actions,
                    top_k=lane_top_ks.get("action_dense", 0),
                    score_key="dense_action_ids",
                ),
            )
        if self.fact_map:
            tasks["fact_dense_ids"] = (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_facts,
                    top_k=lane_top_ks["fact_dense"],
                    score_key="fact_dense_ids",
                ),
            )
            tasks["sparse_fact_ids"] = (
                self._sparse_query_ids,
                {"namespace": self.namespace_facts, "query": query, "top_k": lane_top_ks["fact_sparse"]},
            )
            tasks["local_fact_ids"] = (
                self._local_fact_query_ids,
                {"query": query, "top_k": lane_top_ks["fact_local"]},
            )
        if self.evidence_span_map:
            tasks["dense_evidence_span_ids"] = (
                self._dense_query_ids,
                dense_kwargs(
                    namespace=self.namespace_evidence_spans,
                    top_k=lane_top_ks["evidence_span_dense"],
                    score_key="dense_evidence_span_ids",
                ),
            )
            tasks["sparse_evidence_span_ids"] = (
                self._sparse_query_ids,
                {"namespace": self.namespace_evidence_spans, "query": query, "top_k": lane_top_ks["evidence_span_sparse"]},
            )
            tasks["local_evidence_span_ids"] = (
                self._local_evidence_span_query_ids,
                {"query": query, "top_k": lane_top_ks["evidence_span_local"]},
            )

        defaults = {
            "chunk_dense_ids": [],
            "sparse_chunk_ids": [],
            "local_chunk_ids": [],
            "dense_assertion_ids": [],
            "sparse_assertion_ids": [],
            "local_answer_ids": [],
            "parent_dense_ids": [],
            "sparse_parent_ids": [],
            "local_parent_ids": [],
            "summary_dense_ids": [],
            "sparse_summary_ids": [],
            "media_dense_ids": [],
            "sparse_media_ids": [],
            "local_media_ids": [],
            "dense_page_card_ids": [],
            "dense_action_ids": [],
            "fact_dense_ids": [],
            "sparse_fact_ids": [],
            "local_fact_ids": [],
            "dense_evidence_span_ids": [],
            "sparse_evidence_span_ids": [],
            "local_evidence_span_ids": [],
        }
        enabled = {
            name: (func, kwargs)
            for name, (func, kwargs) in tasks.items()
            if int(kwargs.get("top_k") or 0) > 0
        }
        if not enabled:
            if diagnostics is not None:
                diagnostics["lane_latency_ms"] = {}
                diagnostics["dense_lane_scores"] = {}
            else:
                self._last_lane_latency_ms = {}
            return defaults
        lane_latency_ms: Dict[str, float] = {}
        parallel_lane_workers = max(1, int(getattr(self, "parallel_lane_workers", 1) or 1))
        if len(enabled) == 1 or parallel_lane_workers <= 1:
            for name, (func, kwargs) in enabled.items():
                _name, values, elapsed_ms = self._run_lane_task(name, func, **kwargs)
                defaults[name] = values
                lane_latency_ms[name] = elapsed_ms
            if diagnostics is not None:
                diagnostics["lane_latency_ms"] = dict(lane_latency_ms)
                diagnostics["dense_lane_scores"] = dict(dense_lane_scores)
            else:
                self._last_lane_latency_ms = lane_latency_ms
            return defaults

        future_map = {}
        persistent_lane_executors = bool(getattr(self, "persistent_lane_executors", True))
        split_lane_executors = bool(getattr(self, "split_lane_executors", True))
        if persistent_lane_executors and split_lane_executors:
            remote_workers = max(
                1,
                int(getattr(self, "parallel_remote_lane_workers", parallel_lane_workers) or parallel_lane_workers),
            )
            local_workers = max(
                1,
                int(getattr(self, "parallel_local_lane_workers", max(1, min(4, parallel_lane_workers))) or 1),
            )
            for name, (func, kwargs) in enabled.items():
                if _is_local_lane_name(name):
                    executor = _shared_lane_executor("local-lane", min(local_workers, len(enabled)))
                else:
                    executor = _shared_lane_executor("remote-lane", min(remote_workers, len(enabled)))
                future_map[executor.submit(self._run_lane_task, name, func, **kwargs)] = name
        elif persistent_lane_executors:
            executor = _shared_lane_executor("lane", min(parallel_lane_workers, len(enabled)))
            future_map = {
                executor.submit(self._run_lane_task, name, func, **kwargs): name
                for name, (func, kwargs) in enabled.items()
            }
        else:
            executor = ThreadPoolExecutor(max_workers=min(parallel_lane_workers, len(enabled)))
            try:
                future_map = {
                    executor.submit(self._run_lane_task, name, func, **kwargs): name
                    for name, (func, kwargs) in enabled.items()
                }
            finally:
                executor.shutdown(wait=False)
        for future in as_completed(future_map):
            name = future_map[future]
            try:
                _name, values, elapsed_ms = future.result()
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.warning("Retrieval lane %s failed: %s", name, exc)
                values = []
                elapsed_ms = 0.0
            defaults[name] = list(values)
            lane_latency_ms[name] = elapsed_ms
        if diagnostics is not None:
            diagnostics["lane_latency_ms"] = dict(lane_latency_ms)
            diagnostics["dense_lane_scores"] = dict(dense_lane_scores)
        else:
            self._last_lane_latency_ms = lane_latency_ms
        return defaults
