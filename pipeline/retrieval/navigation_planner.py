"""Grounded navigation plans backed only by the Page Graph catalog.

The query planner may identify a navigation intent, but it is never allowed to
invent a URL or action.  This module resolves every target and step against the
deterministically extracted Page Card/action catalog.
"""

from __future__ import annotations

import re
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import urlsplit

from pipeline.core.io import load_json_safe
from pipeline.core.admissions_routing import admissions_surface_preference
from pipeline.core.navigation_intent import (
    infer_navigation_context,
    normalize_navigation_context,
)
from pipeline.core.page_cards import normalize_url
from pipeline.core.page_graph_bridge import NAVIGATION_CATALOG_SCHEMA_VERSION


NAVIGATION_PLAN_SCHEMA_VERSION = "mbzuai.navigation_plan.v1"
NAVIGATION_EVIDENCE_MAX_PAGE_IDS = 20
_ACTION_TYPES_BY_INTENT = {
    "apply": {"apply", "submit_form"},
    "register": {"register", "submit_form"},
    "contact": {"contact", "email", "telephone", "submit_form"},
    "download": {"download"},
    "login": {"login"},
    "search": {"search"},
    "follow_steps": {
        "apply",
        "register",
        "contact",
        "download",
        "login",
        "submit_form",
    },
}
_TOKEN_RE = re.compile(r"[\w\u0600-\u06ff]+", re.UNICODE)
_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "where",
    "which",
    "how",
    "does",
    "do",
    "should",
    "about",
    "to",
    "mbzuai",
    "please",
    "tell",
    "show",
    "page",
    "website",
    "link",
    "can",
    "you",
    "الى",
    "إلى",
    "على",
    "في",
    "من",
    "عن",
    "كيف",
    "أين",
    "اين",
    "جامعة",
}
_CONTACT_INTENT_TERMS = {
    "call",
    "contact",
    "contacting",
    "contacts",
    "email",
    "emails",
    "mail",
    "phone",
    "reach",
    "telephone",
    "touch",
    "اتصال",
    "اتصل",
    "الاتصال",
    "التواصل",
    "تواصل",
    "بريد",
    "هاتف",
}
_GENERAL_CONTACT_EMAIL_LOCAL_PARTS = {
    "contact",
    "enquiries",
    "hello",
    "info",
    "inquiries",
}

_EMAIL_CONTACT_QUERY_TERMS = {
    "email",
    "emails",
    "e-mail",
    "mail",
    "بريد",
    "البريد",
    "إلكتروني",
    "الكتروني",
}
_TELEPHONE_CONTACT_QUERY_TERMS = {
    "phone",
    "phones",
    "telephone",
    "telephones",
    "هاتف",
    "الهاتف",
}
_DIRECTORY_SURFACE_TERMS = {
    "directory",
    "listing",
    "دليل",
    "الدليل",
    "قائمة",
    "القائمة",
}
_PROFILE_SURFACE_TERMS = {
    "profile",
    "biography",
    "bio",
    "بروفايل",
    "البروفايل",
    "الملف",
    "السيرة",
}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(_clean_text(value))
        if len(token) > 1 and token.casefold() not in _STOP_WORDS
    }


def _ordered_tokens(value: Any) -> List[str]:
    return [
        token.casefold()
        for token in _TOKEN_RE.findall(_clean_text(value))
        if len(token) > 1 and token.casefold() not in _STOP_WORDS
    ]


def _query_phrase_match_score(query: str, page_text: str) -> float:
    query_tokens = _ordered_tokens(query)
    normalized_page = " ".join(_ordered_tokens(page_text))
    if len(query_tokens) < 2 or not normalized_page:
        return 0.0
    matched: set[str] = set()
    score = 0.0
    for width, weight in ((4, 3.5), (3, 2.75), (2, 2.0)):
        if len(query_tokens) < width:
            continue
        for index in range(len(query_tokens) - width + 1):
            phrase = " ".join(query_tokens[index : index + width])
            if phrase in matched or phrase not in normalized_page:
                continue
            matched.add(phrase)
            score += weight
            if score >= 6.0:
                return 6.0
    return score


def _normalized_boundary_text(value: Any) -> str:
    return " ".join(
        token.casefold() for token in _TOKEN_RE.findall(_clean_text(value))
    )


def _heading_matches_chunk_boundary(
    heading: Any,
    chunk: Mapping[str, Any],
) -> bool:
    """Match a Page Card heading only to a heading-shaped chunk text line."""

    normalized_heading = _normalized_boundary_text(heading)
    if not normalized_heading:
        return False
    heading_tokens = normalized_heading.split()
    text = str(
        chunk.get("text") or chunk.get("raw_text") or chunk.get("dense_text") or ""
    )
    for raw_line in text.splitlines():
        normalized_line = _normalized_boundary_text(raw_line)
        line_tokens = normalized_line.split()
        if not normalized_line or len(normalized_line) > 180 or len(line_tokens) > 14:
            continue
        if normalized_line == normalized_heading:
            return True
        if (
            normalized_line.startswith(f"{normalized_heading} ")
            and len(line_tokens) <= len(heading_tokens) + 2
        ):
            return True
    return False


def _page_surface_constraint_score(
    query: str,
    page: Mapping[str, Any],
) -> float:
    """Honor an explicit directory/profile surface named by the user."""

    query_tokens = _tokens(query)
    requested_directory = bool(query_tokens & _DIRECTORY_SURFACE_TERMS)
    requested_profile = bool(query_tokens & _PROFILE_SURFACE_TERMS)
    if requested_directory == requested_profile:
        return 0.0
    identity_text = " ".join(
        _clean_text(page.get(key))
        for key in ("title", "purpose_summary", "page_type", "source_url")
    )
    identity_tokens = _tokens(identity_text)
    is_directory = bool(identity_tokens & _DIRECTORY_SURFACE_TERMS)
    if requested_directory:
        return 8.0 if is_directory else 0.0
    if is_directory:
        return -8.0
    # Some extracted people pages are typed as generic content and do not
    # literally contain "profile". A two-token person/title match plus the
    # absence of directory markers is sufficient corroboration.
    title_overlap = len(query_tokens & _tokens(page.get("title")))
    is_profile = bool(identity_tokens & _PROFILE_SURFACE_TERMS) or title_overlap >= 2
    return 4.0 if is_profile else 0.0


def _normalized_url(value: Any) -> str:
    return normalize_url(value, keep_fragment=False)


def _stable_id(prefix: str, *parts: Any) -> str:
    material = "\0".join(_clean_text(value) for value in parts)
    return f"{prefix}:{sha256(material.encode('utf-8')).hexdigest()[:24]}"


def _is_arabic(value: Any) -> bool:
    return bool(re.search(r"[\u0600-\u06ff]", str(value or "")))


def _safe_action_target(value: Any) -> bool:
    raw = _clean_text(value)
    if not raw or any(character in raw for character in ("\r", "\n", "\x00")):
        return False
    lowered = raw.casefold()
    if lowered.startswith("mailto:"):
        address = raw.split(":", 1)[1].strip()
        return bool(address and "@" in address and not any(ch.isspace() for ch in address))
    if lowered.startswith("tel:"):
        number = raw.split(":", 1)[1].strip()
        return bool(number and re.fullmatch(r"[+()0-9 .-]+", number))
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return False
    return parsed.scheme.casefold() in {"http", "https"} and bool(parsed.hostname)


def _official_navigation_target(value: Any) -> bool:
    raw = _clean_text(value)
    if raw.casefold().startswith("mailto:"):
        address = raw.split(":", 1)[1].strip()
        domain = address.rsplit("@", 1)[-1].casefold() if "@" in address else ""
        return bool(
            domain == "mbzuai.ac.ae"
            or domain.endswith(".mbzuai.ac.ae")
            or domain == "ifm.ai"
            or domain.endswith(".ifm.ai")
        )
    if raw.casefold().startswith("tel:"):
        return True
    try:
        host = (urlsplit(raw).hostname or "").casefold().removeprefix("www.")
    except ValueError:
        return False
    return bool(
        host == "mbzuai.ac.ae"
        or host.endswith(".mbzuai.ac.ae")
        or host == "ifm.ai"
        or host.endswith(".ifm.ai")
    )


def _contact_action_is_semantic(action: Mapping[str, Any]) -> bool:
    action_type = _clean_text(action.get("action_type"))
    if action_type != "contact":
        return True
    material = " ".join(
        _clean_text(action.get(key))
        for key in ("label", "target_url", "canonical_target_url")
    ).casefold()
    return bool(
        re.search(
            r"\b(?:contact|collaborat\w*|partner\w*|visit\w*|tour|get in touch)\b|تواصل|اتصل|تعاون|زيارة",
            material,
        )
    )


def _action_satisfies_intent(action: Mapping[str, Any], intent: str) -> bool:
    action_type = _clean_text(action.get("action_type"))
    desired = _ACTION_TYPES_BY_INTENT.get(intent, set())
    if action_type in desired:
        return _contact_action_is_semantic(action)
    # Extraction records the target mechanism (for example, a login endpoint)
    # while the visible control can express the user's goal (for example,
    # "Apply Now").  Keep both facts: accept the action only when its grounded
    # label/context explicitly expresses the requested intent.
    material = " ".join(
        _clean_text(action.get(key))
        for key in ("label", "context_label", "source_section_heading")
    ).casefold()
    intent_patterns = {
        "apply": r"\b(?:apply|application|submit application)\b|التقديم|تقديم|طلب الالتحاق",
        "register": r"\b(?:register|registration|sign up|enrol|enroll)\b|التسجيل|سج[ّ]?ل",
        "contact": r"\b(?:contact|email|e-mail|phone|telephone|call)\b|تواصل|اتصل|بريد|هاتف",
        "download": r"\b(?:download|pdf|brochure|prospectus)\b|تحميل|تنزيل",
        "login": r"\b(?:log[ -]?in|sign[ -]?in|portal)\b|تسجيل الدخول|بوابة",
        "search": r"\bsearch\b|بحث",
    }
    pattern = intent_patterns.get(intent)
    return bool(pattern and re.search(pattern, material, flags=re.IGNORECASE))


def _page_satisfies_intent(page: Mapping[str, Any], intent: str) -> bool:
    material = " ".join(
        [
            _clean_text(page.get("title")),
            _clean_text(page.get("purpose_summary")),
            _clean_text(page.get("page_type")),
            _clean_text(page.get("source_url")).replace("-", " ").replace("/", " "),
        ]
    ).casefold()
    patterns = {
        "apply": r"\b(?:apply|application portal|start an application)\b|التقديم|طلب الالتحاق",
        "register": r"\b(?:register|registration|sign up|enrol|enroll)\b|التسجيل",
        "contact": r"\b(?:contact|get in touch|visit us)\b|تواصل|اتصل",
        "login": r"\b(?:log[ -]?in|sign[ -]?in|portal)\b|تسجيل الدخول|بوابة",
        "search": r"\b(?:site search|search results|catalogue|catalog)\b|بحث الموقع",
        "follow_steps": r"\b(?:process|procedure|steps|instructions|guide)\b|الخطوات|الإجراءات|الاجراءات",
        "open_page": r".+",
    }
    pattern = patterns.get(intent)
    return bool(pattern and re.search(pattern, material, flags=re.IGNORECASE))


def _candidate_source_urls(result: Mapping[str, Any]) -> List[str]:
    values: List[str] = []
    evidence_pack = result.get("evidence_pack")
    if isinstance(evidence_pack, Mapping):
        for item in evidence_pack.get("items") or []:
            if isinstance(item, Mapping) and item.get("source_url"):
                values.append(_clean_text(item.get("source_url")))
    for key in (
        "answer_documents",
        "fact_documents",
        "evidence_span_documents",
        "retrieval_documents",
    ):
        for item in result.get(key) or []:
            if not isinstance(item, Mapping):
                continue
            metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
            source_url = (
                item.get("source_url")
                or item.get("language_normalized_url")
                or item.get("canonical_url")
                or item.get("document_source")
                or item.get("page_source")
                or metadata.get("source_url")
                or metadata.get("page_source")
                or metadata.get("source")
            )
            if source_url:
                values.append(_clean_text(source_url))
    return list(dict.fromkeys(value for value in values if value))


class GroundedNavigationPlanner:
    """Resolve navigation steps from retrieved evidence and a frozen catalog."""

    def __init__(
        self,
        *,
        catalog: Mapping[str, Any] | None,
        catalog_path: str = "",
    ) -> None:
        self.catalog_path = _clean_text(catalog_path)
        self.available = False
        self.load_error = ""
        self.source_bridge_status = ""
        self.pages_by_id: Dict[str, Dict[str, Any]] = {}
        self.pages_by_url: Dict[str, str] = {}
        self.chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self.actions_by_id: Dict[str, Dict[str, Any]] = {}
        self.actions_by_page: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        if catalog is None:
            self.load_error = "navigation_catalog_missing"
            return
        if catalog.get("schema_version") != NAVIGATION_CATALOG_SCHEMA_VERSION:
            self.load_error = "navigation_catalog_schema_mismatch"
            return
        if catalog.get("source_bridge_coverage_passed") is not True:
            self.load_error = "navigation_catalog_source_bridge_failed"
            return
        self.source_bridge_status = _clean_text(catalog.get("source_bridge_status"))
        page_url_claims: Dict[str, set[str]] = defaultdict(set)
        for value in catalog.get("pages") or []:
            if not isinstance(value, Mapping):
                continue
            page = dict(value)
            page_id = _clean_text(page.get("page_card_id"))
            if not page_id or page_id in self.pages_by_id:
                self.load_error = "navigation_catalog_duplicate_page_id"
                return
            self.pages_by_id[page_id] = page
            for raw_url in (
                page.get("source_url"),
                page.get("canonical_url"),
                page.get("canonical_family_url"),
            ):
                normalized = _normalized_url(raw_url)
                if normalized:
                    page_url_claims[normalized].add(page_id)
        self.pages_by_url = {
            normalized: next(iter(page_ids))
            for normalized, page_ids in page_url_claims.items()
            if len(page_ids) == 1
        }
        for value in catalog.get("chunks") or []:
            if not isinstance(value, Mapping) or not _clean_text(value.get("chunk_id")):
                continue
            chunk_id = _clean_text(value.get("chunk_id"))
            if chunk_id in self.chunks_by_id:
                self.load_error = "navigation_catalog_duplicate_chunk_id"
                return
            self.chunks_by_id[chunk_id] = dict(value)
        for value in catalog.get("actions") or []:
            if not isinstance(value, Mapping):
                continue
            action = dict(value)
            action_id = _clean_text(action.get("action_id"))
            page_id = _clean_text(action.get("page_card_id"))
            if action_id and action_id in self.actions_by_id:
                self.load_error = "navigation_catalog_duplicate_action_id"
                return
            if (
                not action_id
                or page_id not in self.pages_by_id
                or not bool(action.get("official_target"))
                or not _safe_action_target(
                    action.get("canonical_target_url") or action.get("target_url")
                )
                or not _official_navigation_target(
                    action.get("canonical_target_url") or action.get("target_url")
                )
            ):
                continue
            self.actions_by_id[action_id] = action
            self.actions_by_page[page_id].append(action)
        for page_actions in self.actions_by_page.values():
            page_actions.sort(key=lambda value: _clean_text(value.get("action_id")))
        self.available = bool(self.pages_by_id)

    @classmethod
    def from_runtime(
        cls,
        *,
        work_dir: str | Path,
        configured_path: str | Path | None = None,
    ) -> "GroundedNavigationPlanner":
        work_dir = Path(work_dir).resolve()
        candidates: List[Path] = []
        if configured_path:
            configured = Path(configured_path).expanduser()
            if configured.is_absolute():
                candidates.append(configured.resolve())
            else:
                candidates.extend(
                    [configured.resolve(), (work_dir / configured).resolve()]
                )
        candidates.extend(
            [
                work_dir
                / "stage_outputs"
                / "assemble_selected_release"
                / "page_graph_navigation_catalog.json",
                work_dir
                / "stage_outputs"
                / "bridge_page_graph"
                / "page_graph_navigation_catalog.json",
                work_dir / "page_graph_navigation_catalog.json",
            ]
        )
        selected = next((value for value in candidates if value.is_file()), None)
        if selected is None:
            return cls(catalog=None)
        payload = load_json_safe(selected, None)
        if not isinstance(payload, Mapping):
            planner = cls(catalog=None, catalog_path=str(selected))
            planner.load_error = "navigation_catalog_invalid_json_object"
            return planner
        return cls(catalog=payload, catalog_path=str(selected))

    def representation_identities(
        self,
        result: Mapping[str, Any],
        *,
        query: str = "",
        chunk_records: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Dict[str, List[str]]:
        """Resolve selected records to the frozen document/page/section IDs."""

        document_revision_ids: List[str] = []
        page_card_ids: List[str] = []
        section_ids: List[str] = []
        chunk_ids = [
            _clean_text(value)
            for value in result.get("selected_chunk_ids") or []
            if _clean_text(value)
        ]
        corroborated_page_ids: set[str] = set()
        chunk_records = chunk_records or {}
        for chunk_id in chunk_ids:
            bridge = self.chunks_by_id.get(chunk_id) or {}
            page_card_id = _clean_text(bridge.get("page_card_id"))
            document_revision_ids.append(
                _clean_text(bridge.get("document_revision_id"))
            )
            page_card_ids.append(page_card_id)
            section_ids.append(_clean_text(bridge.get("section_id")))
            section_ids.extend(
                _clean_text(value)
                for value in bridge.get("page_section_ids") or []
            )
            if page_card_id in self.pages_by_id:
                corroborated_page_ids.add(page_card_id)
                page = self.pages_by_id[page_card_id]
                page_sections = [
                    section
                    for section in page.get("sections") or []
                    if isinstance(section, Mapping)
                    and _clean_text(section.get("section_id"))
                    and _clean_text(section.get("section_kind")) == "page_heading"
                ]
                page_chunk_ids = [
                    _clean_text(value)
                    for value in page.get("chunk_ids") or []
                    if _clean_text(value)
                ]
                if page_chunk_ids == [chunk_id]:
                    # The selected chunk is the complete page, so every raw
                    # heading is a valid identity for that evidence item.
                    section_ids.extend(
                        _clean_text(section.get("section_id"))
                        for section in page_sections
                    )
                else:
                    chunk_record = chunk_records.get(chunk_id) or {}
                    section_ids.extend(
                        _clean_text(section.get("section_id"))
                        for section in page_sections
                        if _heading_matches_chunk_boundary(
                            section.get("heading"), chunk_record
                        )
                    )
        for page_card_id in result.get("dense_page_card_ids") or []:
            page_card_id = _clean_text(page_card_id)
            page = self.pages_by_id.get(page_card_id) or {}
            page_card_ids.append(page_card_id)
            document_revision_ids.append(
                _clean_text(page.get("document_revision_id"))
            )

        # Traverse Page Card -> section only when two independent lanes agree:
        # the Page Card is in the top-three semantic lane and a selected chunk
        # corroborates the same page. The overlap threshold keeps this a
        # high-precision identity bridge rather than a catalog-wide search.
        query_tokens = _tokens(query)
        if query_tokens:
            section_candidates: List[tuple[float, int, str]] = []
            normalized_query = " ".join(_ordered_tokens(query))
            for page_rank, raw_page_id in enumerate(
                list(result.get("dense_page_card_ids") or [])[:3], start=1
            ):
                page_id = _clean_text(raw_page_id)
                if page_id not in corroborated_page_ids:
                    continue
                page = self.pages_by_id.get(page_id) or {}
                for section in page.get("sections") or []:
                    if not isinstance(section, Mapping):
                        continue
                    if _clean_text(section.get("section_kind")) != "page_heading":
                        continue
                    section_id = _clean_text(section.get("section_id"))
                    heading_tokens = _tokens(section.get("heading"))
                    if not section_id or not heading_tokens:
                        continue
                    overlap = len(query_tokens & heading_tokens)
                    if overlap < 2:
                        continue
                    heading_coverage = overlap / float(len(heading_tokens))
                    query_coverage = overlap / float(len(query_tokens))
                    normalized_heading = " ".join(
                        _ordered_tokens(section.get("heading"))
                    )
                    phrase_match = bool(
                        normalized_heading
                        and normalized_heading in normalized_query
                    )
                    if heading_coverage < 0.5 and not phrase_match:
                        continue
                    score = (
                        (3.0 * heading_coverage)
                        + query_coverage
                        + (1.0 if phrase_match else 0.0)
                    )
                    section_candidates.append((-score, page_rank, section_id))
            section_candidates.sort()
            section_ids.extend(
                section_id
                for _score, _page_rank, section_id in section_candidates[:3]
            )

        navigation_plan = result.get("navigation_plan")
        if isinstance(navigation_plan, Mapping):
            target_page = navigation_plan.get("target_page")
            if isinstance(target_page, Mapping):
                page_card_ids.append(_clean_text(target_page.get("page_card_id")))
                document_revision_ids.append(
                    _clean_text(target_page.get("document_revision_id"))
                )
            navigation_evidence = navigation_plan.get("evidence")
            if isinstance(navigation_evidence, Mapping):
                page_card_ids.extend(
                    _clean_text(value)
                    for value in navigation_evidence.get("page_card_ids") or []
                )
                document_revision_ids.extend(
                    _clean_text(value)
                    for value in navigation_evidence.get("document_revision_ids") or []
                )
                section_ids.extend(
                    _clean_text(value)
                    for value in navigation_evidence.get("section_ids") or []
                )
        for parent_id in result.get("selected_parent_ids") or []:
            parent_id = _clean_text(parent_id)
            if "document-revision:" not in parent_id:
                continue
            suffix = parent_id.split("document-revision:", 1)[1].split(":", 1)[0]
            if suffix:
                document_revision_ids.append(f"document-revision:{suffix}")
        return {
            "document_revision_ids": list(
                dict.fromkeys(value for value in document_revision_ids if value)
            ),
            "page_card_ids": list(
                dict.fromkeys(value for value in page_card_ids if value)
            ),
            "section_ids": list(
                dict.fromkeys(value for value in section_ids if value)
            ),
            "chunk_ids": list(dict.fromkeys(chunk_ids)),
        }

    def fuse_page_card_ranking(
        self,
        result: Mapping[str, Any],
        *,
        evidence_weight: float = 0.15,
        rrf_k: int = 60,
    ) -> List[str]:
        """Late-fuse Page Card semantics with selected-chunk page identity.

        The candidate set remains the dense Page Card lane. Selected chunks
        only corroborate and reorder those existing candidates, so this cannot
        invent a Page Card that the semantic lane did not retrieve.
        """

        dense_page_ids = list(
            dict.fromkeys(
                _clean_text(value)
                for value in result.get("dense_page_card_ids") or []
                if _clean_text(value)
            )
        )
        if len(dense_page_ids) < 2 or evidence_weight <= 0.0:
            return dense_page_ids
        evidence_page_ids: List[str] = []
        for chunk_id in result.get("selected_chunk_ids") or []:
            chunk = self.chunks_by_id.get(_clean_text(chunk_id)) or {}
            page_id = _clean_text(chunk.get("page_card_id"))
            if page_id and page_id not in evidence_page_ids:
                evidence_page_ids.append(page_id)
        if not evidence_page_ids:
            return dense_page_ids
        evidence_ranks = {
            page_id: rank
            for rank, page_id in enumerate(evidence_page_ids, start=1)
        }
        scored: List[tuple[float, int, str]] = []
        for dense_rank, page_id in enumerate(dense_page_ids, start=1):
            score = 1.0 / float(max(1, rrf_k) + dense_rank)
            evidence_rank = evidence_ranks.get(page_id)
            if evidence_rank is not None:
                score += evidence_weight / float(max(1, rrf_k) + evidence_rank)
            scored.append((-score, dense_rank, page_id))
        scored.sort()
        return [page_id for _score, _dense_rank, page_id in scored]

    def _empty_plan(
        self,
        *,
        context: Mapping[str, Any],
        status: str,
        warning: str = "",
    ) -> Dict[str, Any]:
        warnings = [warning] if warning else []
        return {
            "schema_version": NAVIGATION_PLAN_SCHEMA_VERSION,
            "plan_id": _stable_id(
                "navigation-plan", context.get("intent"), context.get("goal"), status
            ),
            "status": status,
            "intent": _clean_text(context.get("intent")) or "none",
            "goal": _clean_text(context.get("goal")),
            "confidence": round(float(context.get("confidence") or 0.0), 3),
            "source": "page_graph_navigation_catalog",
            "target_page": None,
            "steps": [],
            "evidence": {
                "page_card_ids": [],
                "document_revision_ids": [],
                "section_ids": [],
                "chunk_ids": [],
                "action_ids": [],
            },
            "warnings": warnings,
        }

    def _page_search_text(self, page: Mapping[str, Any]) -> str:
        return " ".join(
            [
                _clean_text(page.get("title")),
                _clean_text(page.get("purpose_summary")),
                " ".join(_clean_text(value) for value in page.get("topic_labels") or []),
                " ".join(_clean_text(value) for value in page.get("audience_labels") or []),
                " ".join(
                    _clean_text(section.get("heading"))
                    for section in page.get("sections") or []
                    if isinstance(section, Mapping)
                ),
                " ".join(
                    " ".join(
                        _clean_text(action.get(key))
                        for key in ("action_type", "label", "context_label")
                    )
                    for action in self.actions_by_page.get(
                        _clean_text(page.get("page_card_id")), []
                    )
                ),
                _clean_text(page.get("source_url")).replace("-", " ").replace("/", " "),
            ]
        )

    def page_urls_share_identity(self, first_url: Any, second_url: Any) -> bool:
        """Return whether two catalog URLs are aliases of the same page surface.

        A refreshed page and a retained ``-prev`` route can have the same
        official title, language, and page type while only one preserves a
        historical action such as a PDF download. This check is deliberately
        catalog-bound and same-host so an action cannot jump to a merely
        similar page on another site.
        """

        first_normalized = _normalized_url(first_url)
        second_normalized = _normalized_url(second_url)
        if not first_normalized or not second_normalized:
            return False
        if first_normalized == second_normalized:
            return True
        first_id = self.pages_by_url.get(first_normalized)
        second_id = self.pages_by_url.get(second_normalized)
        if not first_id or not second_id:
            return False
        first_page = self.pages_by_id.get(first_id) or {}
        second_page = self.pages_by_id.get(second_id) or {}
        try:
            first_host = (urlsplit(first_normalized).hostname or "").casefold()
            second_host = (urlsplit(second_normalized).hostname or "").casefold()
        except ValueError:
            return False
        if not first_host or first_host != second_host:
            return False
        first_title = _normalized_boundary_text(first_page.get("title"))
        second_title = _normalized_boundary_text(second_page.get("title"))
        if not first_title or first_title != second_title:
            return False
        for key in ("page_type", "language"):
            first_value = _clean_text(first_page.get(key)).casefold().split("-", 1)[0]
            second_value = _clean_text(second_page.get(key)).casefold().split("-", 1)[0]
            if first_value and second_value and first_value != second_value:
                return False
        return True

    def _score_pages(
        self,
        *,
        query: str,
        result: Mapping[str, Any],
        intent: str,
    ) -> tuple[List[tuple[float, str]], set[str]]:
        query_tokens = _tokens(query)
        lane_scores: Dict[str, Dict[str, float]] = {
            "source": {},
            "chunk": {},
            "page_card": {},
            "action": {},
        }
        page_card_ranks: Dict[str, int] = {}
        action_page_ranks: Dict[str, int] = {}
        evidence_page_ids: set[str] = set()

        def record_lane_score(lane: str, page_id: str, score: float) -> None:
            lane_scores[lane][page_id] = max(
                float(score), float(lane_scores[lane].get(page_id, 0.0))
            )

        for rank, source_url in enumerate(_candidate_source_urls(result), start=1):
            page_id = self.pages_by_url.get(_normalized_url(source_url))
            if page_id:
                record_lane_score("source", page_id, max(5.0, 13.0 - float(rank)))
                evidence_page_ids.add(page_id)
        for rank, chunk_id in enumerate(result.get("selected_chunk_ids") or [], start=1):
            chunk = self.chunks_by_id.get(_clean_text(chunk_id))
            page_id = _clean_text((chunk or {}).get("page_card_id"))
            if page_id in self.pages_by_id:
                record_lane_score("chunk", page_id, max(6.0, 16.0 - float(rank)))
                evidence_page_ids.add(page_id)
        for rank, page_id in enumerate(result.get("dense_page_card_ids") or [], start=1):
            page_id = _clean_text(page_id)
            if page_id in self.pages_by_id:
                page_card_ranks.setdefault(page_id, rank)
                record_lane_score(
                    "page_card", page_id, max(9.0, 26.0 - (3.0 * float(rank - 1)))
                )
                evidence_page_ids.add(page_id)
        for rank, action_id in enumerate(result.get("dense_action_ids") or [], start=1):
            action = self.actions_by_id.get(_clean_text(action_id))
            page_id = _clean_text((action or {}).get("page_card_id"))
            if (
                page_id in self.pages_by_id
                and isinstance(action, Mapping)
                and _action_satisfies_intent(action, intent)
            ):
                action_page_ranks[page_id] = min(
                    rank, action_page_ranks.get(page_id, rank)
                )
                context_tokens = _tokens(
                    " ".join(
                        _clean_text(action.get(key))
                        for key in ("context_label", "source_section_heading")
                    )
                )
                context_overlap_bonus = min(
                    4.0,
                    2.0 * float(len(query_tokens & context_tokens)),
                )
                record_lane_score(
                    "action",
                    page_id,
                    max(10.0, 30.0 - (3.0 * float(rank - 1)))
                    + context_overlap_bonus,
                )
                evidence_page_ids.add(page_id)
                target_url = _clean_text(
                    action.get("canonical_target_url") or action.get("target_url")
                )
                target_page_id = self.pages_by_url.get(_normalized_url(target_url))
                # An internal action can bridge a listing/card to the exact
                # content page.  Use it only when the independent Page Card
                # lane also retrieved that target, so traversal cannot invent
                # a destination from an action alone.
                if target_page_id in page_card_ranks:
                    action_page_ranks[target_page_id] = min(
                        rank, action_page_ranks.get(target_page_id, rank)
                    )
                    record_lane_score(
                        "action",
                        target_page_id,
                        max(9.0, 28.0 - (3.0 * float(rank - 1))),
                    )
                    evidence_page_ids.add(target_page_id)

        scores: Dict[str, float] = defaultdict(float)
        candidate_page_ids = {
            page_id for values in lane_scores.values() for page_id in values
        }
        for page_id in candidate_page_ids:
            direct_score = (
                lane_scores["page_card"].get(page_id, 0.0)
                + lane_scores["action"].get(page_id, 0.0)
            )
            weak_score = (
                lane_scores["source"].get(page_id, 0.0)
                + lane_scores["chunk"].get(page_id, 0.0)
            )
            # Repeated chunks or evidence-pack items from one page are useful
            # corroboration, but must not overturn direct Page Card/action
            # retrieval.  When no direct lane resolved the page, retain the
            # full weak score for graph traversal fallback.
            scores[page_id] = direct_score + (
                min(8.0, weak_score) if direct_score > 0.0 else weak_score
            )
            if page_id in page_card_ranks and page_id in action_page_ranks:
                scores[page_id] += max(
                    8.0,
                    24.0
                    - float(page_card_ranks[page_id] - 1)
                    - (3.0 * float(action_page_ranks[page_id] - 1)),
                )

        # Traverse one cleaned graph hop, but require semantic agreement before
        # a linked page can outrank the page that supplied the evidence.
        traversed_page_ids: set[str] = set()
        for page_id in evidence_page_ids:
            for linked_id in self.pages_by_id[page_id].get("outgoing_page_card_ids") or []:
                linked_id = _clean_text(linked_id)
                if linked_id in self.pages_by_id:
                    traversed_page_ids.add(linked_id)
                    scores.setdefault(linked_id, 6.5)

        # A planner intent is not evidence. If retrieval did not resolve at
        # least one Page Card/chunk, fail closed instead of searching the whole
        # catalog and potentially choosing an unrelated action-rich page.
        candidate_ids = set(scores)
        if not candidate_ids:
            return [], evidence_page_ids
        for page_id in candidate_ids:
            page = self.pages_by_id[page_id]
            page_search_text = self._page_search_text(page)
            page_tokens = _tokens(page_search_text)
            overlap = len(query_tokens & page_tokens)
            if query_tokens:
                scores[page_id] += 5.0 * overlap / float(len(query_tokens))
            scores[page_id] += _query_phrase_match_score(query, page_search_text)
            scores[page_id] += _page_surface_constraint_score(query, page)
            if page_id in evidence_page_ids:
                # Canonical-surface routing may break ties among independently
                # retrieved pages, but it must never let a graph-only neighbor
                # replace an exact source URL supplied by retrieval.
                scores[page_id] += 40.0 * admissions_surface_preference(
                    query,
                    source_url=page.get("source_url"),
                    title=page.get("title"),
                    page_type=page.get("page_type"),
                )
            if any(
                _action_satisfies_intent(action, intent)
                for action in self.actions_by_page.get(page_id, [])
            ):
                scores[page_id] += 3.5
            if page_id in traversed_page_ids and overlap == 0:
                scores[page_id] -= 1.0
            if _is_arabic(query) == _is_arabic(
                f"{page.get('title')} {page.get('purpose_summary')}"
            ):
                scores[page_id] += 0.3
        return (
            sorted(
                ((score, page_id) for page_id, score in scores.items()),
                key=lambda value: (-value[0], value[1]),
            ),
            evidence_page_ids,
        )

    def _best_action(
        self,
        *,
        query: str,
        page_id: str,
        intent: str,
        retrieved_action_ids: Sequence[str] = (),
    ) -> Dict[str, Any] | None:
        if intent not in _ACTION_TYPES_BY_INTENT:
            return None
        query_tokens = _tokens(query)
        required_contact_mechanism = ""
        if intent == "contact":
            if query_tokens & _TELEPHONE_CONTACT_QUERY_TERMS:
                required_contact_mechanism = "telephone"
            elif query_tokens & _EMAIL_CONTACT_QUERY_TERMS:
                required_contact_mechanism = "email"
        page = self.pages_by_id.get(page_id) or {}
        page_identity_tokens = _tokens(
            " ".join(
                _clean_text(page.get(key))
                for key in ("title", "purpose_summary", "page_type", "source_url")
            )
        )
        relevance_tokens = (
            query_tokens - _CONTACT_INTENT_TERMS
            if intent == "contact"
            else query_tokens
        )
        retrieved_ranks = {
            _clean_text(action_id): rank
            for rank, action_id in enumerate(retrieved_action_ids, start=1)
            if _clean_text(action_id)
        }
        prepared_actions: List[tuple[Dict[str, Any], str, set[str]]] = []
        for action in self.actions_by_page.get(page_id, []):
            action_type = _clean_text(action.get("action_type"))
            if not _action_satisfies_intent(action, intent):
                continue
            target_url = _clean_text(
                action.get("canonical_target_url") or action.get("target_url")
            ).casefold()
            action_mechanism = (
                "telephone"
                if target_url.startswith("tel:")
                else "email"
                if target_url.startswith("mailto:")
                else action_type
            )
            if (
                required_contact_mechanism
                and action_mechanism != required_contact_mechanism
            ):
                continue
            context_tokens = _tokens(
                " ".join(
                    _clean_text(action.get(key))
                    for key in (
                        "context_label",
                        "source_section_heading",
                    )
                )
            )
            label = _clean_text(action.get("label"))
            target_tokens = _tokens(
                " ".join(
                    (
                        label if "@" in label or "://" in label else "",
                        _clean_text(action.get("target_url")),
                    )
                )
            )
            if label and "@" not in label and "://" not in label:
                context_tokens.update(_tokens(label))
            # Page-identity words remain meaningful when an action's visible
            # context names the service (for example, "Admissions contact").
            # They are removed only from raw endpoint text, where an address
            # can mechanically repeat the page name without identifying the
            # requested sibling action.
            action_tokens = context_tokens | (target_tokens - page_identity_tokens)
            prepared_actions.append((action, action_type, action_tokens))

        overlap_counts = [
            len(relevance_tokens & action_tokens)
            for _action, _action_type, action_tokens in prepared_actions
        ]
        minimum_overlap = min(overlap_counts) if overlap_counts else 0
        scored: List[tuple[float, str, Dict[str, Any]]] = []
        for action, action_type, action_tokens in prepared_actions:
            if intent == "contact":
                score = {
                    "email": 7.0,
                    "telephone": 6.5,
                    "submit_form": 5.0,
                    "contact": 4.0,
                }.get(action_type, 3.0)
                target_url = _clean_text(
                    action.get("canonical_target_url") or action.get("target_url")
                )
                if target_url.casefold().startswith("mailto:"):
                    local_part = (
                        target_url.split(":", 1)[1].split("@", 1)[0].casefold()
                    )
                    endpoint_tokens = _ordered_tokens(
                        re.sub(r"[._+-]+", " ", local_part)
                    )
                    query_sequence = _ordered_tokens(query)
                    if len(endpoint_tokens) >= 2 and any(
                        query_sequence[index : index + len(endpoint_tokens)]
                        == endpoint_tokens
                        for index in range(
                            max(0, len(query_sequence) - len(endpoint_tokens) + 1)
                        )
                    ):
                        # A service-qualified mailbox such as ``cloud.hpc``
                        # is stronger sibling-action evidence than a generic
                        # mailbox on the same page. Requiring the complete,
                        # ordered local-part phrase avoids rewarding an
                        # incidental page-name token in another endpoint.
                        score += 6.0
                    if local_part in _GENERAL_CONTACT_EMAIL_LOCAL_PARTS:
                        score += 1.0
            else:
                score = 5.0 if action_type == intent else 3.0
            if relevance_tokens:
                score += 4.0 * len(relevance_tokens & action_tokens) / float(
                    len(relevance_tokens)
                )
            retrieved_rank = retrieved_ranks.get(_clean_text(action.get("action_id")))
            action_overlap = len(relevance_tokens & action_tokens)
            if retrieved_rank is not None and (
                len(prepared_actions) == 1 or action_overlap > minimum_overlap
            ):
                # Preserve the vector lane's direct action evidence.  The
                # rank may differentiate siblings only when action-specific
                # evidence—not the already-selected page identity—supports
                # one of them. This prevents an address such as
                # postaward.administration@... from winning merely because
                # the page itself is named Research Administration.
                score += max(2.0, 8.0 - float(retrieved_rank - 1))
            if action.get("source_section_id"):
                score += 0.2
            scored.append((score, _clean_text(action.get("action_id")), action))
        scored.sort(key=lambda value: (-value[0], value[1]))
        return scored[0][2] if scored else None

    def plan(
        self,
        *,
        query: str,
        result: Mapping[str, Any],
        navigation_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        context = normalize_navigation_context(query, navigation_context)
        if context["intent"] == "none":
            return self._empty_plan(context=context, status="not_requested")
        if not self.available:
            return self._empty_plan(
                context=context,
                status="unavailable",
                warning=self.load_error or "navigation_catalog_unavailable",
            )

        ranked_pages, retrieved_page_ids = self._score_pages(
            query=query, result=result, intent=context["intent"]
        )
        if not ranked_pages or ranked_pages[0][0] <= 0:
            return self._empty_plan(
                context=context,
                status="unavailable",
                warning="no_grounded_navigation_target",
            )
        page_score, page_id = ranked_pages[0]
        page = self.pages_by_id[page_id]
        grounding_page_ids = [
            page_id,
            *sorted(value for value in retrieved_page_ids if value != page_id),
        ][:NAVIGATION_EVIDENCE_MAX_PAGE_IDS]
        source_url = _clean_text(page.get("source_url"))
        if not _safe_action_target(source_url) or not _official_navigation_target(source_url):
            return self._empty_plan(
                context=context,
                status="unavailable",
                warning="unsafe_or_missing_page_target",
            )

        action = self._best_action(
            query=query,
            page_id=page_id,
            intent=context["intent"],
            retrieved_action_ids=result.get("dense_action_ids") or [],
        )
        plan_id = _stable_id(
            "navigation-plan",
            context["intent"],
            page_id,
            (action or {}).get("action_id"),
        )
        arabic = _is_arabic(query)
        page_title = _clean_text(page.get("title")) or source_url
        open_step_id = _stable_id("navigation-step", plan_id, "open-page", page_id)
        steps: List[Dict[str, Any]] = [
            {
                "step_id": open_step_id,
                "order": 1,
                "instruction": (
                    f"افتح صفحة «{page_title}» الرسمية."
                    if arabic
                    else f'Open the official “{page_title}” page.'
                ),
                "action_id": None,
                "action_type": "open_page",
                "label": page_title,
                "target_url": source_url,
                "target_kind": "official_page",
                "official_target": True,
                "page_card_id": page_id,
                "document_revision_id": _clean_text(
                    page.get("document_revision_id")
                ),
                "section_id": None,
                "chunk_ids": list(page.get("chunk_ids") or [])[:20],
                "evidence_ids": [],
                "requires_authentication": False,
                "opens_new_window": False,
            }
        ]
        if action is not None:
            action_label = _clean_text(action.get("label"))
            target_url = _clean_text(
                action.get("canonical_target_url") or action.get("target_url")
            )
            action_step_id = _stable_id(
                "navigation-step", plan_id, action.get("action_id"), target_url
            )
            section_heading = _clean_text(action.get("source_section_heading"))
            if arabic:
                instruction = f"اختر «{action_label}»"
                if section_heading:
                    instruction += f" ضمن قسم «{section_heading}»"
                instruction += "."
            else:
                instruction = f'Select “{action_label}”'
                if section_heading:
                    instruction += f' in the “{section_heading}” section'
                instruction += "."
            steps.append(
                {
                    "step_id": action_step_id,
                    "order": 2,
                    "instruction": instruction,
                    "action_id": _clean_text(action.get("action_id")),
                    "action_type": _clean_text(action.get("action_type")),
                    "label": action_label,
                    "target_url": target_url,
                    "target_kind": _clean_text(action.get("target_kind")),
                    "official_target": bool(action.get("official_target")),
                    "page_card_id": page_id,
                    "document_revision_id": _clean_text(
                        page.get("document_revision_id")
                    ),
                    "section_id": _clean_text(action.get("source_section_id")) or None,
                    "chunk_ids": list(page.get("chunk_ids") or [])[:20],
                    "evidence_ids": list(action.get("evidence_ids") or [])[:20],
                    "requires_authentication": _clean_text(
                        action.get("authentication_requirement")
                    )
                    == "explicit",
                    "opens_new_window": bool(action.get("opens_new_window")),
                }
            )

        section_ids = list(
            dict.fromkeys(
                _clean_text(step.get("section_id"))
                for step in steps
                if _clean_text(step.get("section_id"))
            )
        )
        chunk_ids = list(
            dict.fromkeys(
                _clean_text(value)
                for step in steps
                for value in step.get("chunk_ids") or []
                if _clean_text(value)
            )
        )
        action_ids = [
            _clean_text(step.get("action_id"))
            for step in steps
            if _clean_text(step.get("action_id"))
        ]
        warnings: List[str] = []
        status = "ready"
        if (
            context["intent"] in _ACTION_TYPES_BY_INTENT
            and action is None
            and not _page_satisfies_intent(page, context["intent"])
        ):
            status = "partial"
            warnings.append("requested_action_not_grounded_on_target_page")
        if self.source_bridge_status != "ready":
            warnings.append("chunk_bridge_not_ready")
        confidence = min(
            1.0,
            max(float(context["confidence"]), 0.55)
            + min(0.15, max(0.0, page_score) / 100.0),
        )
        target_sections = [
            {
                "section_id": _clean_text(value.get("section_id")),
                "heading": _clean_text(value.get("heading")),
            }
            for value in page.get("sections") or []
            if isinstance(value, Mapping)
        ][:12]
        return {
            "schema_version": NAVIGATION_PLAN_SCHEMA_VERSION,
            "plan_id": plan_id,
            "status": status,
            "intent": context["intent"],
            "goal": context["goal"],
            "confidence": round(confidence, 3),
            "source": "page_graph_navigation_catalog",
            "target_page": {
                "page_card_id": page_id,
                "document_revision_id": _clean_text(
                    page.get("document_revision_id")
                ),
                "title": page_title,
                "url": source_url,
                "purpose_summary": _clean_text(page.get("purpose_summary")),
                "language": _clean_text(page.get("language")),
                "page_type": _clean_text(page.get("page_type")),
                "sections": target_sections,
            },
            "steps": steps,
            "evidence": {
                "page_card_ids": grounding_page_ids,
                "document_revision_ids": list(
                    dict.fromkeys(
                        _clean_text(self.pages_by_id[value].get("document_revision_id"))
                        for value in grounding_page_ids
                        if _clean_text(
                            self.pages_by_id[value].get("document_revision_id")
                        )
                    )
                ),
                "section_ids": section_ids,
                "chunk_ids": chunk_ids,
                "action_ids": action_ids,
            },
            "warnings": warnings,
        }
