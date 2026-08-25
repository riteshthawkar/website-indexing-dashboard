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
from pipeline.core.navigation_intent import (
    infer_navigation_context,
    normalize_navigation_context,
)
from pipeline.core.page_cards import normalize_url
from pipeline.core.page_graph_bridge import NAVIGATION_CATALOG_SCHEMA_VERSION


NAVIGATION_PLAN_SCHEMA_VERSION = "mbzuai.navigation_plan.v1"
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


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in _TOKEN_RE.findall(_clean_text(value))
        if len(token) > 1 and token.casefold() not in _STOP_WORDS
    }


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

    def _score_pages(
        self,
        *,
        query: str,
        result: Mapping[str, Any],
        intent: str,
    ) -> tuple[List[tuple[float, str]], set[str]]:
        query_tokens = _tokens(query)
        scores: Dict[str, float] = defaultdict(float)
        evidence_page_ids: set[str] = set()
        for rank, source_url in enumerate(_candidate_source_urls(result), start=1):
            page_id = self.pages_by_url.get(_normalized_url(source_url))
            if page_id:
                scores[page_id] += max(5.0, 13.0 - float(rank))
                evidence_page_ids.add(page_id)
        for rank, chunk_id in enumerate(result.get("selected_chunk_ids") or [], start=1):
            chunk = self.chunks_by_id.get(_clean_text(chunk_id))
            page_id = _clean_text((chunk or {}).get("page_card_id"))
            if page_id in self.pages_by_id:
                scores[page_id] += max(6.0, 16.0 - float(rank))
                evidence_page_ids.add(page_id)
        for rank, page_id in enumerate(result.get("dense_page_card_ids") or [], start=1):
            page_id = _clean_text(page_id)
            if page_id in self.pages_by_id:
                scores[page_id] += max(7.0, 17.0 - float(rank))
                evidence_page_ids.add(page_id)
        for rank, action_id in enumerate(result.get("dense_action_ids") or [], start=1):
            action = self.actions_by_id.get(_clean_text(action_id))
            page_id = _clean_text((action or {}).get("page_card_id"))
            if page_id in self.pages_by_id:
                scores[page_id] += max(6.5, 15.0 - float(rank))
                evidence_page_ids.add(page_id)

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
            page_tokens = _tokens(self._page_search_text(page))
            overlap = len(query_tokens & page_tokens)
            if query_tokens:
                scores[page_id] += 5.0 * overlap / float(len(query_tokens))
            desired_actions = _ACTION_TYPES_BY_INTENT.get(intent, set())
            if desired_actions and any(
                _clean_text(action.get("action_type")) in desired_actions
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
        self, *, query: str, page_id: str, intent: str
    ) -> Dict[str, Any] | None:
        desired = _ACTION_TYPES_BY_INTENT.get(intent, set())
        if not desired:
            return None
        query_tokens = _tokens(query)
        relevance_tokens = (
            query_tokens - _CONTACT_INTENT_TERMS
            if intent == "contact"
            else query_tokens
        )
        scored: List[tuple[float, str, Dict[str, Any]]] = []
        for action in self.actions_by_page.get(page_id, []):
            action_type = _clean_text(action.get("action_type"))
            if action_type not in desired or not _contact_action_is_semantic(action):
                continue
            action_tokens = _tokens(
                " ".join(
                    _clean_text(action.get(key))
                    for key in (
                        "label",
                        "context_label",
                        "source_section_heading",
                        "target_url",
                    )
                )
            )
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
                    if local_part in _GENERAL_CONTACT_EMAIL_LOCAL_PARTS:
                        score += 1.0
            else:
                score = 5.0 if action_type == intent else 3.0
            if relevance_tokens:
                score += 4.0 * len(relevance_tokens & action_tokens) / float(
                    len(relevance_tokens)
                )
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
        ]
        source_url = _clean_text(page.get("source_url"))
        if not _safe_action_target(source_url) or not _official_navigation_target(source_url):
            return self._empty_plan(
                context=context,
                status="unavailable",
                warning="unsafe_or_missing_page_target",
            )

        action = self._best_action(
            query=query, page_id=page_id, intent=context["intent"]
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
