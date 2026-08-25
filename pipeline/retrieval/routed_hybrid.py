from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Any, Dict, List, Mapping, Sequence
from urllib.parse import unquote, urlparse

from pipeline.core.evidence_adjudicator import adjudicate_factual_evidence
from pipeline.core.navigation_intent import normalize_navigation_context
from pipeline.core.query_expansion import hyde_expansion
from pipeline.core.query_planner import plan_query
from pipeline.retrieval.evidence_packer import build_evidence_pack, score_retrieval_confidence
from pipeline.retrieval.navigation_planner import (
    GroundedNavigationPlanner,
)

from .adaptive_hybrid import (
    AdaptiveHybridRetriever,
    QueryMode,
    _contextual_family_accommodation_match,
    _is_generic_contact_query,
    _is_media_query,
    _public_query_embedding_error_code,
    _query_intent,
    _semantic_query_alias_tokens,
    _tokenize,
    classify_query_mode,
)
from .graph_rag import GraphQueryContext, GraphRAGRetriever, RelationCandidateSet, RelationQueryPlan


logger = logging.getLogger(__name__)
_ADJUDICATOR_RUNTIME_INIT_LOCK = Lock()


def _with_retriever_backend(config: Dict[str, Any], backend: str) -> Dict[str, Any]:
    payload = deepcopy(config or {})
    retrieval_cfg = dict(payload.get("retrieval") or {})
    retrieval_cfg["retriever_backend"] = str(backend)
    payload["retrieval"] = retrieval_cfg
    return payload


def _looks_like_hash_title(value: Any) -> bool:
    return bool(re.fullmatch(r"[a-f0-9]{24,64}", str(value or "").strip().casefold()))


def _title_from_source_url(source_url: str) -> str:
    try:
        parsed = urlparse(str(source_url or "").strip())
    except Exception:
        parsed = None
    path = unquote(parsed.path or "") if parsed is not None else str(source_url or "")
    parts = [
        part
        for part in path.strip("/").split("/")
        if part and not re.fullmatch(r"20\d{2}|\d{1,2}", part)
    ]
    slug = parts[-1] if parts else (parsed.netloc if parsed is not None else "")
    slug = re.sub(r"\.(?:html?|pdf|docx?|pptx?)$", "", slug, flags=re.IGNORECASE)
    slug = re.sub(r"[_-]+", " ", slug).strip()
    if not slug:
        return ""
    return " ".join(
        word.upper() if word.casefold() in {"mbzuai", "faq", "ai", "uae", "phd", "msc", "ugrip"} else word.capitalize()
        for word in slug.split()
    ).strip()


def _clean_document_title(title: Any, source_url: str = "") -> str:
    value = str(title or "").strip()
    if value and not _looks_like_hash_title(value):
        return value
    return _title_from_source_url(source_url)


@dataclass(frozen=True)
class RoutingDecision:
    backend: str
    reason: str
    query_mode: str
    relation_family: str = ""
    relation_confidence: float = 0.0
    graph_available: bool = False
    relation_plan: RelationQueryPlan | None = None


@dataclass(frozen=True)
class QueryRewriteBundle:
    vector_query: str
    graph_query: str
    labels: tuple[str, ...]
    navigation_intent: str = "none"
    navigation_goal: str = ""
    navigation_confidence: float = 0.0
    navigation_source: str = "deterministic_query_intent"


class RoutedHybridRetriever:
    """
    Production-facing retrieval API.

    Vector retrieval remains the default path. Graph retrieval is invoked only
    for routed relation-heavy queries where the graph has measured value.
    """

    def __init__(self, *, config: Dict[str, Any], work_dir: str | Path):
        self.config = dict(config or {})
        self.work_dir = Path(work_dir).resolve()
        retrieval_cfg = dict(self.config.get("retrieval") or {})
        pipeline_cfg = dict(self.config.get("pipeline") or {})

        self.routed_graph_enabled = bool(retrieval_cfg.get("routed_graph_enabled", True))
        self.routed_graph_required = bool(pipeline_cfg.get("production_profile", False)) or bool(
            retrieval_cfg.get("routed_graph_required", False)
        )
        self.routed_graph_query_types = {
            str(value).strip().lower()
            for value in (retrieval_cfg.get("routed_graph_query_types") or ["fact"])
            if str(value).strip()
        }
        self.routed_graph_min_confidence = float(
            retrieval_cfg.get("routed_graph_min_confidence")
            or retrieval_cfg.get("graph_relation_route_min_confidence")
            or 0.38
        )
        allowed_families = retrieval_cfg.get("routed_graph_relation_families") or []
        self.routed_graph_relation_families = {
            str(value).strip().lower()
            for value in allowed_families
            if str(value).strip()
        }
        self.routed_fallback_to_vector = bool(retrieval_cfg.get("routed_fallback_to_vector", True))
        self.parallel_graph_enabled = bool(retrieval_cfg.get("parallel_graph_enabled", True))
        self.parallel_query_rewriting_enabled = bool(retrieval_cfg.get("parallel_query_rewriting_enabled", True))
        self.parallel_graph_augment_all_queries = bool(retrieval_cfg.get("parallel_graph_augment_all_queries", True))
        self.query_planner_enabled = bool(retrieval_cfg.get("query_planner_enabled", False))
        self.query_planner_model = str(retrieval_cfg.get("query_planner_model") or "gpt-5-nano")
        self.query_planner_reasoning_effort = str(
            retrieval_cfg.get("query_planner_reasoning_effort") or "minimal"
        )
        self.query_planner_min_confidence = float(retrieval_cfg.get("query_planner_min_confidence") or 0.55)
        self.navigation_plan_enabled = bool(
            retrieval_cfg.get("navigation_plan_enabled", True)
        )
        self.navigation_plan_required = bool(
            retrieval_cfg.get("navigation_plan_required", False)
        )
        self.navigation_planner = GroundedNavigationPlanner.from_runtime(
            work_dir=self.work_dir,
            configured_path=retrieval_cfg.get("page_graph_navigation_catalog_file"),
        )
        if (
            self.navigation_plan_enabled
            and self.navigation_plan_required
            and not self.navigation_planner.available
        ):
            raise ValueError(
                "Required page-graph navigation catalog is unavailable: "
                f"{self.navigation_planner.load_error or 'not found'}"
            )
        self.evidence_adjudicator_enabled = bool(retrieval_cfg.get("evidence_adjudicator_enabled", False))
        self.selective_adjudication_enabled = bool(retrieval_cfg.get("selective_adjudication_enabled", True))
        self.evidence_adjudicator_model = str(retrieval_cfg.get("evidence_adjudicator_model") or "gpt-5-nano")
        self.evidence_adjudicator_reasoning_effort = str(
            retrieval_cfg.get("evidence_adjudicator_reasoning_effort") or "minimal"
        )
        self.evidence_adjudicator_min_confidence = float(
            retrieval_cfg.get("evidence_adjudicator_min_confidence") or 0.58
        )
        self.evidence_adjudicator_max_completion_tokens = int(
            retrieval_cfg.get("evidence_adjudicator_max_completion_tokens") or 800
        )
        self.evidence_adjudicator_retries = int(retrieval_cfg.get("evidence_adjudicator_retries") or 2)
        self.evidence_adjudicator_retry_delay_sec = float(
            retrieval_cfg.get("evidence_adjudicator_retry_delay_sec") or 1.0
        )
        self.evidence_adjudicator_timeout_sec = float(
            retrieval_cfg.get("evidence_adjudicator_timeout_sec") or 12.0
        )
        configured_provider_timeout = retrieval_cfg.get("evidence_adjudicator_provider_timeout_sec")
        default_provider_timeout = max(0.1, self.evidence_adjudicator_timeout_sec - 1.0)
        self.evidence_adjudicator_provider_timeout_sec = min(
            self.evidence_adjudicator_timeout_sec,
            max(
                0.1,
                float(configured_provider_timeout)
                if configured_provider_timeout is not None
                else default_provider_timeout,
            ),
        )
        self.evidence_adjudicator_max_workers = max(
            1,
            int(retrieval_cfg.get("evidence_adjudicator_max_workers") or 2),
        )
        self.evidence_adjudicator_per_request_delay_sec = float(
            retrieval_cfg.get("evidence_adjudicator_per_request_delay_sec") or 0.0
        )
        self.evidence_adjudicator_answer_limit = int(
            retrieval_cfg.get("evidence_adjudicator_answer_limit") or 4
        )
        self.evidence_adjudicator_fact_limit = int(
            retrieval_cfg.get("evidence_adjudicator_fact_limit") or 4
        )
        self.evidence_adjudicator_chunk_limit = int(
            retrieval_cfg.get("evidence_adjudicator_chunk_limit") or 6
        )
        self.evidence_budget_items = int(retrieval_cfg.get("evidence_budget_items") or 8)
        self.evidence_budget_chars = int(retrieval_cfg.get("evidence_budget_chars") or 8000)
        self.evidence_budget_max_per_source = int(retrieval_cfg.get("evidence_budget_max_per_source") or 2)
        self.aggregation_evidence_budget_items = int(
            retrieval_cfg.get("aggregation_evidence_budget_items") or 12
        )
        self.aggregation_evidence_budget_chars = int(
            retrieval_cfg.get("aggregation_evidence_budget_chars") or 10000
        )
        self.large_page_evidence_budget_items = int(
            retrieval_cfg.get("large_page_evidence_budget_items") or 12
        )
        self.large_page_evidence_budget_chars = int(
            retrieval_cfg.get("large_page_evidence_budget_chars") or 10000
        )
        self.unsupported_intent_guard_enabled = bool(retrieval_cfg.get("unsupported_intent_guard_enabled", True))
        self.future_year_guard_horizon = int(retrieval_cfg.get("future_year_guard_horizon") or 1)
        self.hyde_enabled = bool(retrieval_cfg.get("hyde_enabled", False))
        self.hyde_query_modes = {
            str(value).strip().lower()
            for value in (retrieval_cfg.get("hyde_query_modes") or ["synthesis"])
            if str(value).strip()
        }
        self.hyde_model = str(retrieval_cfg.get("hyde_model") or "gpt-5-nano")
        self.hyde_min_confidence = float(retrieval_cfg.get("hyde_min_confidence") or 0.55)
        self.hyde_max_chars = int(retrieval_cfg.get("hyde_max_chars") or 600)
        self.hyde_retries = int(retrieval_cfg.get("hyde_retries") or 1)
        self.hyde_retry_delay_sec = float(retrieval_cfg.get("hyde_retry_delay_sec") or 1.0)
        self.hyde_per_request_delay_sec = float(retrieval_cfg.get("hyde_per_request_delay_sec") or 0.0)
        self.supports_shared_parallel_retrieval = False

        vector_config = _with_retriever_backend(self.config, "vector")
        self.vector = AdaptiveHybridRetriever(config=vector_config, work_dir=self.work_dir)
        self.model = self.vector.model
        self.output_dimensionality = self.vector.output_dimensionality
        self._coverage_page_records = self._build_coverage_page_records()

        self.graph: GraphRAGRetriever | None = None
        self.graph_init_error: str | None = None
        if self.routed_graph_enabled:
            graph_config = _with_retriever_backend(self.config, "graph_hybrid")
            if self.parallel_graph_augment_all_queries:
                graph_retrieval_cfg = dict(graph_config.get("retrieval") or {})
                graph_retrieval_cfg["graph_relation_only"] = False
                graph_config["retrieval"] = graph_retrieval_cfg
            try:
                self.graph = GraphRAGRetriever(
                    config=graph_config,
                    work_dir=self.work_dir,
                    base_retriever=self.vector,
                )
            except Exception as exc:
                if self.routed_graph_required:
                    raise ValueError(
                        "Production graph retriever initialization failed; refusing vector-only downgrade"
                    ) from exc
                logger.warning("Graph retriever initialization failed; graph routing is disabled: %s", exc)
                self.graph_init_error = "graph_initialization_failed"
                self.graph = None

        self._initialize_evidence_adjudicator_runtime()
        self.supports_shared_parallel_retrieval = bool(
            getattr(self.vector, "supports_shared_parallel_retrieval", False)
            and (
                self.graph is None
                or getattr(self.graph, "supports_shared_parallel_retrieval", False)
            )
        )

    def _initialize_evidence_adjudicator_runtime(self) -> None:
        """Create one bounded adjudication pool per retriever instance.

        The capacity semaphore deliberately covers running work, not only calls
        waiting for ``Future.result``. A provider call that outlives the caller's
        deadline therefore retains its slot and cannot cause an unbounded queue
        of background requests.
        """
        if getattr(self, "_evidence_adjudicator_executor", None) is not None:
            return
        with _ADJUDICATOR_RUNTIME_INIT_LOCK:
            if getattr(self, "_evidence_adjudicator_executor", None) is not None:
                return
            workers = max(1, int(getattr(self, "evidence_adjudicator_max_workers", 2) or 2))
            self._evidence_adjudicator_capacity = BoundedSemaphore(workers)
            self._evidence_adjudicator_executor = ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="mbzuai-evidence-adjudicator",
            )

    def close(self) -> None:
        executor = getattr(self, "_evidence_adjudicator_executor", None)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            self._evidence_adjudicator_executor = None
        vector_close = getattr(getattr(self, "vector", None), "close", None)
        if callable(vector_close):
            vector_close()

    def embed_query(self, query: str) -> List[float]:
        return self.vector.embed_query(query)

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        return self.vector.embed_queries(queries)

    def _append_alias_tokens(
        self,
        query: str,
        alias_tokens: Sequence[str],
        *,
        max_new_tokens: int = 6,
    ) -> str:
        additions: List[str] = []
        existing_tokens = set(_tokenize(query))
        for token in alias_tokens:
            token = str(token or "").strip().lower()
            if not token or token in existing_tokens or token in additions:
                continue
            additions.append(token)
            if len(additions) >= max_new_tokens:
                break
        if not additions:
            return query
        return f"{query} {' '.join(additions)}".strip()

    def _graph_relation_plan(self, query: str) -> RelationQueryPlan | None:
        if self.graph is None:
            return None
        mode = classify_query_mode(query)
        media_query = _is_media_query(query)
        return self.graph._build_relation_query_plan(query, mode=mode, media_query=media_query)

    def _build_query_rewrite_bundle(
        self,
        query: str,
        *,
        relation_plan: RelationQueryPlan | None = None,
        query_mode: str = "fact",
        use_query_planner: bool = True,
    ) -> QueryRewriteBundle:
        labels: List[str] = []
        vector_query = query
        graph_query = query
        navigation = normalize_navigation_context(query)
        generic_contact_query = _is_generic_contact_query(query)
        if self.query_planner_enabled and use_query_planner:
            plan = plan_query(
                query=query,
                model=self.query_planner_model,
                fallback_query_type=query_mode,
            )
            planner_confidence = float(plan.get("confidence") or 0.0)
            navigation = normalize_navigation_context(
                query,
                {
                    "intent": plan.get("navigation_intent"),
                    "goal": plan.get("navigation_goal"),
                    "confidence": plan.get("navigation_confidence"),
                    "source": "retrieval_query_planner",
                },
            )
            if planner_confidence >= self.query_planner_min_confidence:
                planned_vector = str(plan.get("vector_query") or "").strip()
                planned_graph = str(plan.get("graph_query") or "").strip()
                if planned_vector and planned_vector != vector_query:
                    vector_query = planned_vector
                    labels.append("openai_vector_plan")
                if planned_graph and planned_graph != graph_query:
                    graph_query = planned_graph
                    labels.append("openai_graph_plan")
        if self.parallel_query_rewriting_enabled and not generic_contact_query:
            semantic_aliases = _semantic_query_alias_tokens(query)
            if semantic_aliases:
                candidate = self._append_alias_tokens(vector_query, semantic_aliases, max_new_tokens=6)
                if candidate != vector_query:
                    vector_query = candidate
                    labels.append("semantic_alias_expansion")
            graph_query = vector_query
            if relation_plan is not None and self.graph is not None:
                candidate = self.graph._expanded_relation_query(graph_query, relation_plan)
                if candidate != graph_query:
                    graph_query = candidate
                    labels.append("relation_alias_expansion")
        if bool(getattr(self, "hyde_enabled", False)) and query_mode in getattr(self, "hyde_query_modes", {"synthesis"}):
            expansion = hyde_expansion(
                query=query,
                model=str(getattr(self, "hyde_model", "gpt-5-nano")),
                reasoning_effort=self.query_planner_reasoning_effort
                if hasattr(self, "query_planner_reasoning_effort")
                else "minimal",
                retries=int(getattr(self, "hyde_retries", 1)),
                retry_delay_sec=float(getattr(self, "hyde_retry_delay_sec", 1.0)),
                per_request_delay_sec=float(getattr(self, "hyde_per_request_delay_sec", 0.0)),
            )
            hypothetical = str(expansion.get("hypothetical_document") or "").strip()
            confidence = float(expansion.get("confidence") or 0.0)
            if hypothetical and confidence >= float(getattr(self, "hyde_min_confidence", 0.55)):
                vector_query = f"{vector_query}\n\nHypothetical relevant passage: {hypothetical[: int(getattr(self, 'hyde_max_chars', 600))]}"
                labels.append("hyde_expansion")
        return QueryRewriteBundle(
            vector_query=vector_query,
            graph_query=graph_query,
            labels=tuple(dict.fromkeys(labels)),
            navigation_intent=str(navigation.get("intent") or "none"),
            navigation_goal=str(navigation.get("goal") or ""),
            navigation_confidence=float(navigation.get("confidence") or 0.0),
            navigation_source=str(
                navigation.get("source") or "deterministic_query_intent"
            ),
        )

    def _empty_graph_context(self, query: str) -> GraphQueryContext:
        return GraphQueryContext(
            mode=classify_query_mode(query),
            media_query=False,
            relation_plan=None,
            relation_candidates=RelationCandidateSet(),
            rewritten_query=query,
            rewrite_labels=tuple(),
        )

    def _intent_summary(self, query: str) -> Dict[str, Any]:
        intent = _query_intent(query)
        return {
            "answer_types": list(intent.answer_types),
            "requested_roles": list(intent.requested_role_subtypes),
            "subject_tokens": list(intent.subject_tokens),
            "subject_phrases": list(intent.subject_phrases),
            "strict_answer_required": bool(intent.strict_answer_required),
        }

    def _unsupported_intent_reason(self, query: str) -> str:
        if not self.unsupported_intent_guard_enabled:
            return ""
        text = " ".join(str(query or "").casefold().split())
        if not text:
            return ""

        private_markers = (" my ", " personal ", " private ", " assigned to ", " assignment ")
        private_targets = ("interview schedule", "admissions interview", "dorm room", "room number", "room numbers")
        padded = f" {text} "
        if any(marker in padded for marker in private_markers) and any(target in text for target in private_targets):
            return "unsupported_private_or_user_specific_request"

        if ("room number" in text or "room numbers" in text) and any(
            marker in text for marker in ("assigned", "assignment", "dorm", "housing")
        ):
            return "unsupported_private_or_user_specific_request"

        if any(
            marker in text
            for marker in (
                "موعد مقابلتي",
                "جدول مقابلتي",
                "مقابلتي الشخصية",
                "غرفتي",
                "رقم غرفتي",
                "المخصص لي",
            )
        ):
            return "unsupported_private_or_user_specific_request"

        if "exact questions" in text and any(marker in text for marker in ("exam", "screening", "test", "assessment")):
            return "unsupported_confidential_exam_content"
        if any(marker in text for marker in ("الأسئلة الدقيقة", "الاسئلة الدقيقة", "أسئلة الاختبار نفسها")) and any(
            marker in text for marker in ("اختبار", "امتحان", "تقييم")
        ):
            return "unsupported_confidential_exam_content"

        if any(marker in text for marker in ("right now", "live location", "current live", "currently live")) and any(
            marker in text for marker in ("shuttle", "bus", "vehicle", "airport")
        ):
            return "unsupported_live_operational_status"
        if any(marker in text for marker in ("الآن", "الان", "الموقع المباشر", "موقعه الحالي")) and any(
            marker in text for marker in ("الحافلة", "حافلة", "مركبة", "المطار")
        ):
            return "unsupported_live_operational_status"

        years = [int(value) for value in re.findall(r"\b(20\d{2})\b", text)]
        if years:
            latest_supported_year = datetime.now().year + max(0, self.future_year_guard_horizon)
            if max(years) > latest_supported_year and any(
                marker in text
                for marker in (
                    "winner",
                    "won ",
                    "who won",
                    "prize",
                    "prize amount",
                    "result",
                    "results",
                    "awardee",
                    "الفائز",
                    "فاز",
                    "الجائزة",
                    "النتائج",
                )
            ):
                return "unsupported_future_event_result"

        return ""

    def _abstained_payload_from_result(
        self,
        *,
        result: Dict[str, Any],
        reason: str,
        confidence: float,
        method: str,
    ) -> Dict[str, Any]:
        payload = dict(result or {})
        payload.update(
            {
                "selected_answer_ids": [],
                "selected_fact_ids": [],
                "selected_chunk_ids": [],
                "selected_parent_ids": [],
                "selected_media_ids": [],
                "answer_documents": [],
                "fact_documents": [],
                "retrieval_documents": [],
                "media": [],
                "abstained": True,
                "adjudication_used": True,
                "adjudication_method": method,
                "adjudication_reason": reason,
                "adjudication_confidence": round(float(confidence or 0.0), 3),
                "verification_status": "abstained",
            }
        )
        return payload

    def _reorder_documents(
        self,
        *,
        documents: Sequence[Dict[str, Any]],
        selected_ids: Sequence[str],
    ) -> List[Dict[str, Any]]:
        if not documents:
            return []
        by_id = {
            str(doc.get("id") or ""): doc
            for doc in documents
            if isinstance(doc, dict) and str(doc.get("id") or "")
        }
        ordered: List[Dict[str, Any]] = []
        seen = set()
        for record_id in selected_ids:
            record_id = str(record_id or "")
            if not record_id or record_id in seen or record_id not in by_id:
                continue
            seen.add(record_id)
            ordered.append(by_id[record_id])
        return ordered

    def _apply_evidence_adjudication(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not self.evidence_adjudicator_enabled:
            result.setdefault("verification_status", "not_requested")
            return result
        payload = dict(result or {})
        if payload.get("abstained"):
            payload.setdefault("verification_status", "not_required_abstained")
            return payload
        if str(payload.get("mode") or "").strip().lower() != QueryMode.FACT.value:
            payload.setdefault("adjudication_used", False)
            payload.setdefault("verification_status", "skipped_non_fact")
            return payload

        if bool(getattr(self, "selective_adjudication_enabled", False)) and not self._should_run_evidence_adjudication(payload):
            payload.setdefault("adjudication_used", False)
            payload.setdefault("verification_status", "skipped_high_confidence")
            return payload

        answer_documents = [
            doc for doc in (payload.get("answer_documents") or []) if isinstance(doc, dict) and str(doc.get("id") or "")
        ]
        fact_documents = [
            doc for doc in (payload.get("fact_documents") or []) if isinstance(doc, dict) and str(doc.get("id") or "")
        ]
        retrieval_documents = [
            doc for doc in (payload.get("retrieval_documents") or []) if isinstance(doc, dict) and str(doc.get("id") or "")
        ]
        if not answer_documents and not fact_documents and not retrieval_documents:
            payload.setdefault("adjudication_used", False)
            payload.setdefault("verification_status", "skipped_no_evidence")
            return payload

        adjudication_kwargs = {
            "query": query,
            "intent_summary": self._intent_summary(query),
            "answer_documents": answer_documents[: self.evidence_adjudicator_answer_limit],
            "fact_documents": fact_documents[: self.evidence_adjudicator_fact_limit],
            "retrieval_documents": retrieval_documents[: max(8, self.evidence_adjudicator_chunk_limit + 2)],
            "model": self.evidence_adjudicator_model,
            "reasoning_effort": self.evidence_adjudicator_reasoning_effort,
            "min_confidence": self.evidence_adjudicator_min_confidence,
            "max_completion_tokens": self.evidence_adjudicator_max_completion_tokens,
            "retries": self.evidence_adjudicator_retries,
            "retry_delay_sec": self.evidence_adjudicator_retry_delay_sec,
            "per_request_delay_sec": self.evidence_adjudicator_per_request_delay_sec,
            "provider_timeout_sec": min(
                max(0.1, float(getattr(self, "evidence_adjudicator_provider_timeout_sec", 11.0))),
                max(0.1, float(getattr(self, "evidence_adjudicator_timeout_sec", 12.0))),
            ),
            "max_answer_ids": self.evidence_adjudicator_answer_limit,
            "max_fact_ids": self.evidence_adjudicator_fact_limit,
            "max_chunk_ids": self.evidence_adjudicator_chunk_limit,
        }
        self._initialize_evidence_adjudicator_runtime()
        capacity = self._evidence_adjudicator_capacity
        if not capacity.acquire(blocking=False):
            payload.setdefault("adjudication_used", False)
            payload["verification_status"] = "skipped_busy"
            payload["adjudication_reason"] = "evidence_adjudicator_capacity_exhausted"
            return payload

        try:
            future = self._evidence_adjudicator_executor.submit(
                adjudicate_factual_evidence,
                **adjudication_kwargs,
            )
        except RuntimeError:
            capacity.release()
            payload.setdefault("adjudication_used", False)
            payload["verification_status"] = "skipped_unavailable"
            payload["adjudication_reason"] = "evidence_adjudicator_unavailable"
            return payload

        # Release only when provider work actually exits. ``Future.cancel`` does
        # not stop a running network call and must not free capacity early.
        future.add_done_callback(lambda _future: capacity.release())
        try:
            adjudication = future.result(timeout=max(0.1, float(getattr(self, "evidence_adjudicator_timeout_sec", 12.0))))
        except FutureTimeoutError:
            future.cancel()
            payload.setdefault("adjudication_used", False)
            payload["verification_status"] = "skipped_timeout"
            payload["adjudication_reason"] = "evidence_adjudicator_timeout"
            return payload

        payload["adjudication_used"] = bool(
            adjudication.get("used")
            or adjudication.get("abstain")
            or adjudication.get("selected_answer_ids")
            or adjudication.get("selected_fact_ids")
            or adjudication.get("selected_chunk_ids")
        )
        payload["adjudication_method"] = str(adjudication.get("method") or "")
        payload["adjudication_reason"] = str(adjudication.get("reason") or "")
        payload["adjudication_confidence"] = round(float(adjudication.get("confidence") or 0.0), 3)
        payload["verification_status"] = "verified" if payload["adjudication_used"] else "verification_no_selection"

        if adjudication.get("abstain"):
            return self._abstained_payload_from_result(
                result=payload,
                reason=str(adjudication.get("reason") or "adjudicated_no_support"),
                confidence=float(adjudication.get("confidence") or 0.0),
                method=str(adjudication.get("method") or ""),
            )

        selected_answer_ids = [
            str(value)
            for value in (adjudication.get("selected_answer_ids") or [])
            if str(value)
        ]
        selected_fact_ids = [
            str(value)
            for value in (adjudication.get("selected_fact_ids") or [])
            if str(value)
        ]
        if selected_answer_ids:
            payload["selected_answer_ids"] = selected_answer_ids
            payload["answer_documents"] = self._reorder_documents(
                documents=answer_documents,
                selected_ids=selected_answer_ids,
            )
        if selected_fact_ids:
            payload["selected_fact_ids"] = selected_fact_ids
            payload["fact_documents"] = self._reorder_documents(
                documents=fact_documents,
                selected_ids=selected_fact_ids,
            )
        if selected_answer_ids or selected_fact_ids:
            leading_ids = set(selected_answer_ids) | set(selected_fact_ids)
            trailing_docs = [
                doc
                for doc in retrieval_documents
                if str(doc.get("id") or "") not in leading_ids
            ]
            payload["retrieval_documents"] = [
                *list(payload.get("answer_documents") or []),
                *list(payload.get("fact_documents") or []),
                *trailing_docs,
            ]
        return payload

    def _should_run_evidence_adjudication(self, payload: Dict[str, Any]) -> bool:
        confidence = float(payload.get("retrieval_confidence") or 0.0)
        if confidence <= 0.55:
            return True
        answer_documents = [
            doc for doc in (payload.get("answer_documents") or []) if isinstance(doc, dict)
        ]
        if not answer_documents:
            return True
        answer_values = {
            str(doc.get("value") or doc.get("text") or "").strip().casefold()
            for doc in answer_documents
            if str(doc.get("value") or doc.get("text") or "").strip()
        }
        if len(answer_values) > 1:
            return True
        if payload.get("routing_graph_error"):
            return True
        return False

    def _coverage_intent(self, query: str, mode: QueryMode) -> str:
        query_lower = query.lower()
        if self._unsupported_intent_reason(query):
            return "unsupported"
        if re.search(r"\b(compare|all|list|across|multiple|programs|departments|schools|faculty members|aggregate)\b", query_lower):
            return "multi_page_aggregation"
        if re.search(r"\b(overview|summary|summarize|complete page|whole page|full page|entire page|large page)\b", query_lower):
            return "large_page"
        if mode == QueryMode.SYNTHESIS:
            return "broad_synthesis"
        if re.search(r"\b(faculty|professor|program|phd|master|department)\b", query_lower):
            return "faculty_program_detail"
        return "exact_fact" if mode == QueryMode.FACT else mode.value

    def _coverage_plan_for_result(
        self,
        *,
        query: str,
        payload: Dict[str, Any],
        mode: QueryMode,
    ) -> Dict[str, Any]:
        intent = self._coverage_intent(query, mode)
        inferred = self._infer_coverage_requirements(query, intent)
        required_entities = [
            str(value)
            for value in (payload.get("required_entities") or inferred.get("required_entities") or [])
            if str(value).strip()
        ]
        required_pages = [
            str(value)
            for value in (
                payload.get("required_pages")
                or payload.get("required_source_urls")
                or inferred.get("required_pages")
                or []
            )
            if str(value).strip()
        ]
        required_sections = [
            str(value)
            for value in (payload.get("required_sections") or inferred.get("required_sections") or [])
            if str(value).strip()
        ]
        selected_span_ids = [
            str(value)
            for value in (payload.get("selected_evidence_span_ids") or [])
            if str(value).strip()
        ]
        has_evidence = bool(
            selected_span_ids
            or payload.get("selected_chunk_ids")
            or payload.get("answer_documents")
            or payload.get("fact_documents")
        )
        coverage_status = "complete" if has_evidence else "insufficient"
        if required_pages:
            selected_sources = self._selected_source_urls(payload)
            missing_pages = [
                page
                for page in required_pages
                if self._normalize_source_url(page) not in selected_sources
            ]
            if missing_pages:
                coverage_status = "partial" if has_evidence else "insufficient"
        elif (required_entities or required_sections) and not selected_span_ids:
            coverage_status = "partial" if has_evidence else "insufficient"
        return {
            "intent": intent,
            "required_entities": required_entities,
            "required_pages": required_pages,
            "required_sections": required_sections,
            "selected_span_ids": selected_span_ids,
            "coverage_status": coverage_status,
        }

    def _evidence_budget_for_plan(self, coverage_plan: Dict[str, Any]) -> tuple[int, int, int]:
        intent = str((coverage_plan or {}).get("intent") or "")
        if intent == "multi_page_aggregation":
            return (
                self.aggregation_evidence_budget_items,
                self.aggregation_evidence_budget_chars,
                self.evidence_budget_max_per_source,
            )
        if intent == "large_page":
            return (
                self.large_page_evidence_budget_items,
                self.large_page_evidence_budget_chars,
                self.evidence_budget_max_per_source,
            )
        return (
            self.evidence_budget_items,
            self.evidence_budget_chars,
            self.evidence_budget_max_per_source,
        )

    def _route_query(self, query: str) -> RoutingDecision:
        mode = classify_query_mode(query)
        query_mode = mode.value
        if not self.routed_graph_enabled or not self.parallel_graph_enabled:
            return RoutingDecision(
                backend="vector",
                reason="graph_disabled",
                query_mode=query_mode,
                graph_available=False,
            )
        if self.graph is None:
            return RoutingDecision(
                backend="vector",
                reason="graph_unavailable",
                query_mode=query_mode,
                graph_available=False,
            )
        relation_plan = self._graph_relation_plan(query)
        return RoutingDecision(
            backend="parallel_hybrid",
            reason="parallel_vector_graph",
            query_mode=query_mode,
            relation_family=(relation_plan.family.strip().lower() if relation_plan is not None else ""),
            relation_confidence=float(relation_plan.confidence or 0.0) if relation_plan is not None else 0.0,
            graph_available=True,
            relation_plan=relation_plan,
        )

    def _normalize_source_url(self, value: Any) -> str:
        raw = str(value or "").strip().rstrip("/")
        if not raw:
            return ""
        try:
            parsed = urlparse(raw)
        except Exception:
            return raw.casefold()
        if not parsed.netloc:
            return raw.casefold()
        path = (unquote(parsed.path or "/")).rstrip("/")
        return f"{parsed.scheme.lower() or 'https'}://{parsed.netloc.lower()}{path}".rstrip("/").casefold()

    def _coverage_page_family_key(self, value: Any) -> str:
        normalized = self._normalize_source_url(value)
        try:
            path = unquote(urlparse(str(value or "")).path or "").casefold()
        except Exception:
            path = normalized
        if re.search(r"campus[_-]?map", path):
            return "pdf:campus-map"
        return normalized

    def _coverage_page_recency_key(self, value: Any) -> tuple[int, int, int]:
        try:
            path = unquote(urlparse(str(value or "")).path or "").casefold()
        except Exception:
            path = str(value or "").casefold()
        date_match = re.search(r"/(20\d{2})/(\d{1,2})/", path)
        date_score = 0
        if date_match:
            date_score = (int(date_match.group(1)) * 100) + int(date_match.group(2))
        version_match = re.search(r"(?:^|[_-])v(\d+)", path)
        version_score = int(version_match.group(1)) if version_match else 0
        return (date_score, version_score, len(path))

    def _dedupe_explicit_pages_by_family(self, pages: Sequence[str]) -> List[str]:
        selected_by_family: Dict[str, str] = {}
        for page in pages:
            if not str(page or "").strip():
                continue
            family = self._coverage_page_family_key(page)
            current = selected_by_family.get(family)
            if current is None or self._coverage_page_recency_key(page) > self._coverage_page_recency_key(current):
                selected_by_family[family] = str(page)
        return list(selected_by_family.values())

    def _source_url_from_record(self, record: Dict[str, Any]) -> str:
        metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        for key in (
            "source_url",
            "language_normalized_url",
            "canonical_url",
            "document_source",
            "page_source",
            "source",
        ):
            value = record.get(key) or metadata.get(key)
            if str(value or "").strip():
                return str(value).strip()
        return ""

    def _build_coverage_page_records(self) -> List[Dict[str, Any]]:
        by_url: Dict[str, Dict[str, Any]] = {}
        sources = [
            getattr(self.vector, "evidence_span_map", {}),
            getattr(self.vector, "chunk_map", {}),
            getattr(self.vector, "summary_map", {}),
            getattr(self.vector, "parent_map", {}),
        ]
        for source_map in sources:
            if not isinstance(source_map, dict):
                continue
            for record in source_map.values():
                if not isinstance(record, dict):
                    continue
                source_url = self._source_url_from_record(record)
                key = self._normalize_source_url(source_url)
                if not key or "mbzuai.ac.ae" not in key:
                    continue
                page = by_url.setdefault(
                    key,
                    {
                        "source_url": source_url.rstrip("/"),
                        "normalized_url": key,
                        "parts": [],
                    },
                )
                for value in (
                    record.get("document_title"),
                    record.get("title"),
                    record.get("section_heading"),
                    record.get("heading"),
                    record.get("breadcrumb"),
                    " ".join(str(part) for part in (record.get("section_path") or [])),
                    record.get("span_type"),
                    record.get("text"),
                    record.get("dense_text"),
                    record.get("sparse_text"),
                ):
                    text = str(value or "").strip()
                    if text:
                        page["parts"].append(text[:1200])
        records: List[Dict[str, Any]] = []
        for page in by_url.values():
            parsed = urlparse(page["source_url"])
            slug_text = " ".join(part.replace("-", " ") for part in unquote(parsed.path or "").split("/") if part)
            search_text = " ".join([slug_text, *page["parts"]])[:12000].casefold()
            records.append(
                {
                    "source_url": page["source_url"],
                    "normalized_url": page["normalized_url"],
                    "search_text": search_text,
                    "tokens": set(_tokenize(search_text)),
                }
            )
        return records

    def _query_has_specific_target(self, query: str) -> bool:
        lower = query.casefold()
        if re.search(r"\bcontact\b.{0,60}\badmissions?\b", lower) or re.search(
            r"\badmissions?\b.{0,60}\bcontact\b",
            lower,
        ):
            return True
        if any(
            marker in lower
            for marker in (
                "scholarship",
                "library",
                "ai reach",
                "ugrip",
                "undergraduate research internship",
                "campus facilities",
                "campus amenities",
                "campus map",
                "official working hours",
                "official workings hours",
                "working hours",
                "offices operate",
                "operating hours",
                "where is mbzuai",
                "student-facing campus services",
                "student facing campus services",
                "support facilities",
                "student wellbeing",
                "admissions email",
                "admission email",
                "admissions committee",
                "contact admissions",
                "general admissions",
                "undergraduate admissions",
                "core ai specializations",
                "specializations",
                "ai programs",
                "law no. 25",
                "executive council",
                "affiliated",
                "institutional identity",
                "north car park",
                "parking permitted",
                "visitor parking",
                "guest parking",
                "vehicles be parked",
                "vehicles can be parked",
                "where can vehicles",
                "where can cars",
                "shuttle",
                "student accommodation",
                "student housing",
                "online screening exam",
                "screening exam",
                "practical campus information",
                "newcomer briefing",
                "visitor should know",
                "arriving at mbzuai",
            )
        ):
            return True
        if self._query_faculty_person_names(query):
            return True
        if "campus" in lower and re.search(r"\b(facilities|facility|services|amenities|amenity)\b", lower):
            return True
        if (
            re.search(r"\b(location|located|where|working hours|offices operate|parking|transport|shuttle|accommodation|facilities)\b", lower)
            and re.search(r"\b(newcomer|visitor|arriving|campus|student|practical)\b", lower)
        ):
            return True
        if re.search(r"\b(master|msc|m\.sc|doctor|phd|ph\.d|bachelor|undergraduate)\b", lower) and re.search(
            r"\b(machine learning|computer vision|natural language processing|robotics|computational biology|computer science|statistics|data science|human-computer interaction|hci|applied artificial intelligence|engineering stream|business stream)\b",
            lower,
        ):
            return True
        if re.search(r"\b(professor|faculty)\b", lower) and re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", query):
            return True
        return False

    def _query_faculty_person_names(self, query: str) -> List[str]:
        if not re.search(
            r"\b(professor|faculty|research interests?|publications?|profile|supervisor|lab|biography|bio)\b",
            str(query or ""),
            flags=re.IGNORECASE,
        ):
            return []
        names: List[str] = []
        for raw in re.findall(r"\b(?:Professor|Prof\.?|Dr\.?|Faculty)?\s*([A-Z][A-Za-z]+(?:[-' ][A-Z][A-Za-z]+){1,4})\b", str(query or "")):
            cleaned = re.sub(r"'s\b", "", raw).strip(" ,.;:?!")
            cleaned = re.sub(r"^(?:Professor|Prof\.?|Dr\.?|Faculty)\s+", "", cleaned, flags=re.IGNORECASE).strip()
            lowered = cleaned.casefold()
            if not cleaned or lowered in {
                "natural language processing",
                "machine learning",
                "computer vision",
                "artificial intelligence",
                "mohamed bin zayed",
            }:
                continue
            if any(token in lowered.split() for token in ("mbzuai", "phd", "msc", "master", "doctor")):
                continue
            names.append(cleaned)
        return list(dict.fromkeys(names))

    def _english_query_page_allowed(self, source_url: str, query: str = "") -> bool:
        normalized = self._normalize_source_url(source_url)
        if "/ar/study/faculty/" in normalized and self._query_faculty_person_names(query):
            return True
        return not any(marker in normalized for marker in ("/ar/", "-arb", "_arb", "arabic"))

    def _clean_required_entity_phrase(self, phrase: str) -> str:
        cleaned = str(phrase or "").strip()
        cleaned = re.sub(
            r"^(?:using|explain|describe|summarize|list|compare|give(?: me)?(?: a)?(?: detailed)?(?: answer)?(?: about)?|tell me about)\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip(" :,.")
        cleaned = re.sub(r"^mbzuai'?s?\s+", "", cleaned, flags=re.IGNORECASE).strip(" :,.")
        cleaned = re.sub(r"^mbzuai\s+", "", cleaned, flags=re.IGNORECASE).strip(" :,.")
        if cleaned.casefold() in {
            "",
            "mbzuai",
            "using mbzuai",
            "mohamed bin zayed university",
            "artificial intelligence",
            "mohamed bin zayed university of artificial intelligence",
        }:
            return ""
        return cleaned

    def _facet_required_entities(self, query: str) -> List[str]:
        lower = query.casefold()
        entities: List[str] = []
        practical_campus_query = bool(
            re.search(r"\b(practical|newcomer|visitor|arriv(?:e|ing|al)|campus|student)\b", lower)
            and re.search(r"\b(location|where|working hours|offices operate|parking|transport|shuttle|facilities|accommodation)\b", lower)
        )
        if practical_campus_query and re.search(r"\b(location|where|arriv(?:e|ing|al)|campus)\b", lower):
            entities.append("Masdar")
        if practical_campus_query and re.search(r"\b(transport|transportation|shuttle|bus|visitor|visitors|arriv(?:e|ing|al))\b", lower):
            entities.append("NAVYA bus")
        if practical_campus_query and re.search(r"\b(working hours|offices operate|operating hours)\b", lower):
            entities.append("working hours")
        if practical_campus_query and re.search(r"\b(accommodation|housing)\b", lower):
            entities.append("accommodation")
        if practical_campus_query and re.search(r"\b(facilities|library)\b", lower):
            entities.append("library")
        return list(dict.fromkeys(entities))

    def _page_target_score(self, query: str, page: Dict[str, Any]) -> float:
        lower_query = query.casefold()
        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return 0.0
        page_tokens = set(page.get("tokens") or set())
        search_text = str(page.get("search_text") or "")
        url = str(page.get("normalized_url") or "")
        score = len(query_tokens & page_tokens) / float(len(query_tokens))
        for phrase in (
            "machine learning",
            "computer vision",
            "natural language processing",
            "computational biology",
            "computer science",
            "statistics and data science",
            "human-computer interaction",
            "applied artificial intelligence",
            "engineering stream",
            "business stream",
            "ai reach",
            "undergraduate research internship",
            "kentaro inui",
            "haiyan huang",
        ):
            if phrase in lower_query and phrase in search_text:
                score += 0.42
        faculty_names = self._query_faculty_person_names(query)
        if faculty_names:
            for name in faculty_names:
                name_tokens = [token for token in _tokenize(name) if len(token) > 1]
                if not name_tokens:
                    continue
                if all(token in search_text or token in url for token in name_tokens):
                    score += 1.35
                    if "/study/faculty/" in url:
                        score += 0.45
                elif "/study/faculty/" in url:
                    score -= 0.45
        if "scholarship" in lower_query and ("scholarship" in search_text or "scholarship" in url):
            score += 0.55
        if "library" in lower_query and ("library" in search_text or "campus-facilities" in url):
            score += 0.50
        if ("campus" in lower_query or "facilities" in lower_query) and (
            "campus-facilities" in url or "/about/faq" in url
        ):
            score += 0.35
        if re.search(r"\b(master|msc|m\.sc)\b", lower_query):
            if "/phd-programs/" in url or "doctor-of-philosophy" in url:
                score -= 0.70
            if "/msc-programs/" in url or "/master-" in url or "/master-in-" in url:
                score += 0.28
        if re.search(r"\b(doctor|phd|ph\.d)\b", lower_query):
            if "/msc-programs/" in url or "/master-" in url or "/master-in-" in url:
                score -= 0.70
            if "/phd-programs/" in url or "doctor-of-philosophy" in url:
                score += 0.28
        if re.search(r"\b(bachelor|undergraduate)\b", lower_query):
            if "/graduate-" in url or "/phd-programs/" in url or "/msc-programs/" in url:
                score -= 0.55
            if "undergraduate" in url or "bachelor" in url:
                score += 0.30
        if re.search(r"\b(professor|faculty)\b", lower_query):
            if "/study/faculty/" in url:
                score += 0.35
            else:
                score -= 0.30
        return score

    def _explicit_required_page_markers(self, query: str) -> List[str]:
        lower = query.casefold()
        markers: List[str] = []
        if "ai reach" in lower:
            markers.append("/study/ai-reach")
        if "kentaro inui" in lower:
            markers.append("/study/faculty/kentaro-inui")
        for name in self._query_faculty_person_names(query):
            slug = re.sub(r"[^a-z0-9]+", "-", name.casefold()).strip("-")
            if slug:
                markers.append(f"/study/faculty/{slug}")
        if re.search(
            r"\b(core ai specializations|specializations|m\.sc\. and ph\.d\.|m\.sc|ph\.d|msc and phd|masters? and phd|ai programs)\b",
            lower,
        ):
            if "five" in lower or "core ai specializations" in lower or "m.sc" in lower or "ph.d" in lower or "msc and phd" in lower:
                markers.append("mbzuai_faculty_brochure")
            markers.append("/ai-programs")
        if re.search(r"\b(law|established|affiliated|executive council|institutional identity)\b", lower):
            markers.append("university-catalogue-2024-2025")
            markers.append("/about/faq")
            markers.append("summarized_mbzuai-factsheet")
            markers.append("mbzuai_university_catalogue")
        if re.search(r"\b(campus facilities|campus amenities|support facilities|campus services|knowledge center|medical center)\b", lower) or (
            "campus" in lower and re.search(r"\b(facilities|facility|services|amenities|amenity)\b", lower)
        ):
            markers.append("/student-resources/campus-facilities")
            if re.search(r"\b(core|amenities|support|student|accommodation|available)\b", lower):
                markers.append("/study/undergraduate-application-submission")
            if re.search(
                r"\b(student[-\s]?facing|support|services?|medical|first[-\s]?aid|prayer|washrooms?|map[-\s]?marked|visibly\s+marked)\b",
                lower,
            ):
                markers.extend(["campus_map", "campus-map"])
        screening_exam_context = bool(re.search(r"\b(online screening exam|screening exam)\b", lower))
        if (
            re.search(r"\b(location|located|where mbzuai|working hours|offices operate|weekday|parking)\b", lower)
            and not screening_exam_context
            and not re.search(
                r"\b(guest|visitor|visitors|visiting)\b",
                lower,
            )
        ):
            markers.append("/about/faq")
        if re.search(r"\b(airport|abu dhabi international airport|how far|taxi|careem|transportation options?)\b", lower):
            markers.append("/about/faq")
            if "airport" in lower:
                markers.append("/study/undergraduate-application-submission")
        if (
            re.search(r"\b(practical campus information|visitor should know|before arriving|arriving at mbzuai|newcomer briefing|new graduate student)\b", lower)
            and re.search(r"\b(location|where|campus|parking|transport|facilities)\b", lower)
            and not screening_exam_context
        ):
            markers.append("/about/faq")
        if ("campus" in lower and re.search(r"\b(map|layout)\b", lower)) or "campus map" in lower:
            markers.extend(["campus_map", "campus-map"])
        if screening_exam_context:
            markers.append("online-screening-exam-instructions")
            if re.search(r"\b(process|instructions?|technical|specifications?|opt(?:ing)? out|available|help|support|contact)\b", lower):
                markers.append("/study/admission-process")
        if re.search(r"\b(online screening exam|screening exam)\b", lower) and re.search(
            r"\b(it support|technical support|working hours|available|instructions?)\b",
            lower,
        ):
            markers.append("online-screening-exam-instructions")
        if re.search(r"\b(north car park|where .*park|vehicles? .*park|vehicles? be parked|where can vehicles|where can cars|parking permitted|visitor parking|guest parking)\b", lower):
            markers.append("/about/contact")
        if (
            re.search(r"\b(transport|transportation|shuttle|bus|arriving|arrival)\b", lower)
            and re.search(r"\b(campus|visitor|visitors|visiting|family|guest|parking|practical|newcomer|arriving)\b", lower)
        ):
            markers.append("/about/contact")
        if "parking" in lower and any(token in lower for token in ("provided", "available", "guests", "visitors", "students")):
            markers.append("/about/faq")
            markers.append("/about/contact")
        exact_vehicle_parking_query = bool(
            re.search(
                r"\b(parking permitted|permitted .{0,40}parking|"
                r"where\s+(?:can\s+)?(?:vehicles?|cars?|guests?|visitors?)\b.{0,50}\bpark(?:ed|ing)?|"
                r"(?:vehicles?|cars?)\s+(?:can\s+)?be\s+parked|"
                r"masdar city campus.{0,50}\bpark(?:ed|ing)?|north car park)\b",
                lower,
            )
        )
        if exact_vehicle_parking_query:
            markers.append("university-catalogue-2024-2025")
            markers.append("/about/contact")
        if re.search(r"\b(student housing|student accommodation|family members?|parents? stay|accommodation)\b", lower):
            markers.append("/study/undergraduate-application-submission")
        if re.search(r"\b(shuttle|transport|transportation|navya|golf cart|prt|bus)\b", lower):
            markers.append("/about/contact")
        if (
            re.search(r"\b(student-facing|student facing|for students|new student|new graduate student|arriving at mbzuai|campus facilities|facilities are available)\b", lower)
            and re.search(r"\b(facilities|services|amenities|accommodation|campus)\b", lower)
        ):
            markers.append("/study/undergraduate-application-submission")
        if re.search(
            r"\b(admissions?(?:\s+\w+){0,3}\s+email|admission email|admissions?\s+committee|admissions?\s+contact|contact\b.{0,60}\badmissions?)\b",
            lower,
        ):
            if "undergraduate" in lower or "ug." in lower:
                markers.append("/study/undergraduate-application-submission")
            else:
                markers.append("mbzuai_application_instructions_new_msc-phd")
                markers.append("university-catalogue-2024-2025")
                markers.append("online-screening-exam-instructions")
                markers.append("/study/admissions")
                markers.append("/study/admission-process")
                markers.append("/about/faq")
        if re.search(r"\b(general admissions|admission@mbzuai\\.ac\\.ae)\b", lower):
            markers.append("/about/faq")
        if re.search(r"\b(undergraduate admissions|ug\\.admission@mbzuai\\.ac\\.ae)\b", lower):
            markers.append("/study/undergraduate-application-submission")
        if "undergraduate" in lower and "scholarship" in lower:
            markers.extend(
                [
                    "/study/undergraduate-application-submission",
                    "tahnoon-bin-zayed-scholarship",
                ]
            )
        if "engineering stream" in lower and ("bachelor" in lower or "undergraduate" in lower):
            markers.append("/study/undergraduate-program/bachelor-of-science-in-artificial-intelligence-engineering-stream")
        if "business stream" in lower and ("bachelor" in lower or "undergraduate" in lower):
            markers.append("/study/undergraduate-program/bachelor-of-science-in-artificial-intelligence-business-stream")

        program_slug_by_phrase = {
            "machine learning": "machine-learning",
            "computer vision": "computer-vision",
            "natural language processing": "natural-language-processing",
            "computational biology": "computational-biology",
            "computer science": "computer-science",
            "robotics": "robotics",
            "statistics and data science": "statistics-and-data-science",
            "statistics & data science": "statistics-and-data-science",
            "human-computer interaction": "human-computer-interaction",
            "hci": "human-computer-interaction",
        }
        matched_program_slug = ""
        for phrase, slug in program_slug_by_phrase.items():
            if phrase in lower:
                matched_program_slug = slug
                break
        if matched_program_slug:
            if re.search(r"\b(master|msc|m\.sc)\b", lower):
                markers.append(f"/study/msc-programs/master-of-science-in-{matched_program_slug}")
            if re.search(r"\b(doctor|phd|ph\.d)\b", lower):
                markers.append(f"/study/phd-programs/doctor-of-philosophy-in-{matched_program_slug}")
        if "master in applied artificial intelligence" in lower or "maai" in lower or "applied ai" in lower:
            markers.append("/study/master-in-applied-ai")
        return list(dict.fromkeys(markers))

    def _infer_coverage_requirements(self, query: str, intent: str) -> Dict[str, List[str]]:
        if not self._query_has_specific_target(query):
            return {"required_pages": [], "required_entities": [], "required_sections": []}
        explicit_markers = self._explicit_required_page_markers(query)
        if explicit_markers:
            explicit_pages = []
            for page in self._coverage_page_records:
                normalized_url = str(page.get("normalized_url") or "")
                if any(marker.casefold().strip("/") in normalized_url for marker in explicit_markers):
                    explicit_pages.append(str(page.get("source_url") or ""))
            if explicit_pages and not re.search(r"[\u0600-\u06FF]", query):
                english_pages = [
                    page
                    for page in explicit_pages
                    if self._english_query_page_allowed(page, query=query)
                ]
                if english_pages:
                    explicit_pages = english_pages
            explicit_pages = self._dedupe_explicit_pages_by_family(explicit_pages)
            if explicit_pages:
                entities: List[str] = []
                if intent == "multi_page_aggregation" or len(explicit_pages) > 1:
                    for phrase in re.findall(r"\b[A-Z][A-Za-z]+(?:[- ][A-Z][A-Za-z]+){1,5}\b", query):
                        cleaned = self._clean_required_entity_phrase(phrase)
                        if cleaned:
                            entities.append(cleaned)
                entities.extend(self._facet_required_entities(query))
                return {
                    "required_pages": list(dict.fromkeys(explicit_pages))[:6 if intent == "multi_page_aggregation" else 4],
                    "required_entities": list(dict.fromkeys(entities)),
                    "required_sections": [],
                }
        scored = [
            (self._page_target_score(query, page), page)
            for page in self._coverage_page_records
        ]
        scored = [(score, page) for score, page in scored if score >= 0.58]
        scored.sort(key=lambda item: (-item[0], item[1]["normalized_url"]))
        max_pages = 6 if intent == "multi_page_aggregation" else 3
        pages = [page["source_url"] for _score, page in scored[:max_pages]]
        entities: List[str] = []
        for phrase in re.findall(r"\b[A-Z][A-Za-z]+(?:[- ][A-Z][A-Za-z]+){1,5}\b", query):
            cleaned = self._clean_required_entity_phrase(phrase)
            if cleaned:
                entities.append(cleaned)
        entities.extend(self._facet_required_entities(query))
        return {
            "required_pages": pages,
            "required_entities": list(dict.fromkeys(entities)),
            "required_sections": [],
        }

    def _selected_source_urls(self, payload: Dict[str, Any]) -> set[str]:
        urls: set[str] = set()
        for key in ("answer_documents", "fact_documents", "evidence_span_documents", "retrieval_documents"):
            for item in payload.get(key) or []:
                if isinstance(item, dict):
                    normalized = self._normalize_source_url(self._source_url_from_record(item))
                    if normalized:
                        urls.add(normalized)
        return urls

    def _selected_direct_evidence_source_urls(self, payload: Dict[str, Any]) -> set[str]:
        urls: set[str] = set()
        for key in ("answer_documents", "fact_documents", "evidence_span_documents"):
            for item in payload.get(key) or []:
                if isinstance(item, dict):
                    normalized = self._normalize_source_url(self._source_url_from_record(item))
                    if normalized:
                        urls.add(normalized)
        return urls

    def _span_payload_from_record(self, span: Dict[str, Any], *, required_page: str = "") -> Dict[str, Any]:
        source_url = self._source_url_from_record(span) or required_page
        return {
            "id": str(span.get("id") or ""),
            "text": str(span.get("text") or span.get("dense_text") or ""),
            "span_type": str(span.get("span_type") or "general"),
            "source_url": source_url,
            "canonical_url": str(span.get("canonical_url") or ""),
            "document_title": _clean_document_title(span.get("document_title") or span.get("title"), source_url),
            "section_heading": str(span.get("section_heading") or span.get("heading") or ""),
            "breadcrumb": str(span.get("breadcrumb") or ""),
            "linked_chunk_ids": [str(value) for value in (span.get("linked_chunk_ids") or []) if str(value)],
            "linked_parent_ids": [str(value) for value in (span.get("linked_parent_ids") or []) if str(value)],
            "authority_class": str(span.get("authority_class") or "official"),
            "source_last_seen": str(span.get("source_last_seen") or ""),
            "validity_status": str(span.get("validity_status") or "active"),
            "coverage_injected": True,
        }

    def _fact_payload_from_record(self, fact: Dict[str, Any], *, required_page: str = "") -> Dict[str, Any]:
        source_url = self._source_url_from_record(fact) or required_page
        return {
            "id": str(fact.get("id") or ""),
            "text": str(fact.get("text") or fact.get("dense_text") or ""),
            "source_url": source_url,
            "document_title": _clean_document_title(fact.get("document_title") or fact.get("title"), source_url),
            "section_heading": str(fact.get("section_heading") or fact.get("heading") or ""),
            "breadcrumb": str(fact.get("breadcrumb") or ""),
            "linked_chunk_ids": [str(value) for value in (fact.get("linked_chunk_ids") or []) if str(value)],
            "linked_parent_ids": [str(value) for value in (fact.get("linked_parent_ids") or []) if str(value)],
            "authority_class": str(fact.get("authority_class") or "official"),
            "source_last_seen": str(fact.get("source_last_seen") or ""),
            "validity_status": str(fact.get("validity_status") or "active"),
            "coverage_injected": True,
        }

    def _required_page_match_rank(self, source_url: str, required_pages: Sequence[str]) -> tuple[int, bool] | None:
        normalized_source = self._normalize_source_url(source_url)
        if not normalized_source:
            return None
        source_family = self._coverage_page_family_key(source_url)
        family_match: tuple[int, bool] | None = None
        for index, required_page in enumerate(required_pages):
            normalized_required = self._normalize_source_url(required_page)
            if normalized_source == normalized_required:
                return (index, True)
            if source_family == self._coverage_page_family_key(required_page):
                family_match = (index, False)
        return family_match

    def _facet_relevance_bonus(self, query: str, text: str, source_url: str = "") -> float:
        query_lower = query.casefold()
        text_lower = text.casefold()
        source_lower = self._normalize_source_url(source_url)
        bonus = 0.0
        if re.search(r"\b(working hours|workings hours|offices operate|operating hours|weekday)\b", query_lower):
            if (
                "official workings hours" in text_lower
                or "official working hours" in text_lower
                or re.search(r"\b8:00\s*a\.?m", text_lower)
                or re.search(r"\b7\.?30\s*a?m", text_lower)
            ):
                bonus += 0.85
        if re.search(r"\b(family|parents?|stay|housing|accommodation)\b", query_lower):
            if _contextual_family_accommodation_match(query, text):
                bonus += 1.10
            if any(
                marker in text_lower
                for marker in (
                    "does not provide housing",
                    "parents stay",
                    "nearby hotels",
                    "airbnbs",
                    "student accommodation",
                    "on-campus accommodation",
                    "multi-occupancy room",
                )
            ):
                bonus += 0.80
            if "does not provide housing for parents" in text_lower:
                bonus += 0.45
        if re.search(r"\b(parking|park|car park|guest|visitor)\b", query_lower):
            if any(
                marker in text_lower
                for marker in (
                    "north car park",
                    "visitor parking",
                    "car parking is provided",
                    "parking spaces",
                    "parking is permitted",
                )
            ):
                bonus += 0.75
        if re.search(r"\b(transport|transportation|shuttle|bus|arriving|arrival)\b", query_lower):
            if any(
                marker in text_lower
                for marker in (
                    "golf cart",
                    "navya bus",
                    "prt",
                    "personal rapid transit",
                    "taxi",
                    "transportation",
                    "if available",
                )
            ):
                bonus += 0.72
            if "study/undergraduate-application-submission" in source_lower and "get to campus" in text_lower:
                bonus += 0.35
        if re.search(r"\b(campus facilities|facilities|amenities|support facilities|library|canteen|gym|knowledge center|medical center|laboratories)\b", query_lower):
            if any(
                marker in text_lower
                for marker in (
                    "campus facilities",
                    "purpose-built facilities",
                    "knowledge center",
                    "medical center",
                    "library",
                    "laborator",
                    "canteen",
                    "gym",
                    "sports",
                    "student residences",
                )
            ):
                bonus += 0.70
        admissions_contact_query = bool(
            re.search(
                r"\b(general admissions|admissions?\s+committee|admission@mbzuai\.ac\.ae|admissions?\s+contact|admissions?(?:\s+\w+){0,3}\s+email|admission email)\b",
                query_lower,
            )
            or re.search(r"\bcontact\b.{0,60}\badmissions?\b", query_lower)
            or re.search(r"\badmissions?\b.{0,60}\bcontact\b", query_lower)
        )
        if admissions_contact_query:
            if "admission@mbzuai.ac.ae" in text_lower:
                bonus += 1.15
                if "university-catalogue-2024-2025" in source_lower:
                    bonus += 0.45
                if "mbzuai_application_instructions_new_msc-phd" in source_lower:
                    bonus += 0.85
                if "online-screening-exam-instructions" in source_lower:
                    bonus += 0.65
                    if "committee" in query_lower:
                        bonus += 0.55
            if "ug.admission@mbzuai.ac.ae" in text_lower and "undergraduate" not in query_lower:
                bonus -= 1.00
            if any(marker in text_lower for marker in ("emergency response", "emergency contact", "mbzuai management.contact number")):
                bonus -= 0.90
        if re.search(r"\b(undergraduate admissions|ug\.admission@mbzuai\.ac\.ae|undergraduate applicants?)\b", query_lower):
            if "ug.admission@mbzuai.ac.ae" in text_lower:
                bonus += 0.90
        if re.search(r"\b(it support|technical support|screening exam)\b", query_lower):
            support_hours_query = bool(re.search(r"\b(it support|technical support|available|working hours|hours)\b", query_lower))
            has_complete_support_hours = (
                "working hours" in text_lower
                and "8:00 am" in text_lower
                and ("12:30 pm" in text_lower or "12:30" in text_lower)
            )
            has_truncated_support_hours = (
                "working hours" in text_lower
                and "8:00 am" in text_lower
                and ("5:00 pm (" in text_lower or text_lower.rstrip().endswith("("))
                and not ("12:30 pm" in text_lower or "12:30" in text_lower)
            )
            if support_hours_query and has_complete_support_hours:
                bonus += 1.95
            elif support_hours_query and has_truncated_support_hours:
                bonus -= 1.20
            elif any(
                marker in text_lower
                for marker in (
                    "it_external@mbzuai.ac.ae",
                    "working hours are at 8:00 am",
                    "admission-related questions may be sent to admission@mbzuai.ac.ae",
                    "online screening exam",
                    "screening exam instructions",
                    "exam topics",
                    "process, opting out criteria, and technical specifications",
                )
            ):
                bonus += 0.90
        if re.search(r"\b(institutional identity|named after|established|law|legal personality|affiliated|executive council)\b", query_lower):
            if any(
                marker in text_lower
                for marker in (
                    "named after",
                    "his highness sheikh mohamed bin zayed",
                    "established in 2019",
                    "established as an independent local entity",
                    "legal personality",
                    "affiliated to the",
                    "executive council",
                )
            ):
                bonus += 0.95
        return bonus

    def _best_required_page_spans(self, query: str, required_page: str, *, limit: int = 2) -> List[Dict[str, Any]]:
        required_normalized = self._normalize_source_url(required_page)
        scored: List[tuple[float, Dict[str, Any]]] = []
        for span in getattr(self.vector, "evidence_span_map", {}).values():
            if not isinstance(span, dict):
                continue
            if self._normalize_source_url(self._source_url_from_record(span)) != required_normalized:
                continue
            text = " ".join(
                str(span.get(key) or "")
                for key in ("document_title", "section_heading", "breadcrumb", "span_type", "text", "sparse_text")
            )
            try:
                score = float(self.vector._score_text_match(query, text))
            except Exception:
                score = 0.0
            query_lower = query.casefold()
            text_lower = text.casefold()
            if any(token in query_lower and token in text_lower for token in ("scholarship", "deadline", "credit", "full-time", "library", "referee", "screening", "python", "award", "research")):
                score += 0.30
            score += self._facet_relevance_bonus(query, text, self._source_url_from_record(span))
            if re.search(r"\b(what is|who is|who .* for|designed for|for whom)\b", query_lower):
                if any(token in text_lower for token in ("designed", "participants", "students", "program", "provides", "aims", "intended")):
                    score += 0.45
            if not re.search(r"\b(date|deadline|when|application|admission|decision|close|screening)\b", query_lower):
                if any(token in text_lower for token in ("applications close", "admission decision", "program dates", "postponed", "deadline")):
                    score -= 0.35
            if score > 0.0:
                scored.append((score, span))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id") or "")))
        output: List[Dict[str, Any]] = []
        for _score, span in scored[:limit]:
            output.append(self._span_payload_from_record(span, required_page=required_page))
        return output

    def _best_required_page_facts(self, query: str, required_page: str, *, limit: int = 1) -> List[Dict[str, Any]]:
        required_normalized = self._normalize_source_url(required_page)
        scored: List[tuple[float, Dict[str, Any]]] = []
        for fact in getattr(self.vector, "fact_map", {}).values():
            if not isinstance(fact, dict):
                continue
            if self._normalize_source_url(self._source_url_from_record(fact)) != required_normalized:
                continue
            text = " ".join(
                str(fact.get(key) or "")
                for key in ("document_title", "section_heading", "breadcrumb", "text", "dense_text", "source_url")
            )
            if not text.strip():
                continue
            try:
                score = float(self.vector._score_text_match(query, text))
            except Exception:
                score = 0.0
            score += self._facet_relevance_bonus(query, text, self._source_url_from_record(fact))
            if score > 0.0:
                scored.append((score, fact))
        scored.sort(key=lambda item: (-item[0], str(item[1].get("id") or "")))
        return [self._fact_payload_from_record(fact, required_page=required_page) for _score, fact in scored[:limit]]

    def _prioritize_required_page_evidence(
        self,
        *,
        query: str,
        payload: Dict[str, Any],
        coverage_plan: Dict[str, Any],
    ) -> None:
        required_pages = [
            str(value)
            for value in (coverage_plan.get("required_pages") or [])
            if str(value).strip()
        ]
        if not required_pages:
            return

        def document_sort_key(index_doc: tuple[int, Dict[str, Any]]) -> tuple[int, int, int, float, int, int, int]:
            index, doc = index_doc
            source_url = self._source_url_from_record(doc)
            match = self._required_page_match_rank(source_url, required_pages)
            if match is None:
                return (1, 999, 1, 0.0, 0, 0, index)
            rank, exact = match
            recency = self._coverage_page_recency_key(source_url)
            text = " ".join(
                str(doc.get(key) or "")
                for key in ("document_title", "section_heading", "breadcrumb", "span_type", "text", "sparse_text")
            )
            return (
                0,
                rank,
                0 if exact else 1,
                -self._facet_relevance_bonus(query, text, source_url),
                -recency[0],
                -recency[1],
                index,
            )

        span_docs = [doc for doc in (payload.get("evidence_span_documents") or []) if isinstance(doc, dict)]
        if span_docs:
            payload["evidence_span_documents"] = [
                doc for _index, doc in sorted(enumerate(span_docs), key=document_sort_key)
            ]
            original_ids = [
                str(value)
                for value in (payload.get("selected_evidence_span_ids") or [])
                if str(value)
            ]
            sorted_ids = [
                str(doc.get("id") or "")
                for doc in payload["evidence_span_documents"]
                if str(doc.get("id") or "")
            ]
            payload["selected_evidence_span_ids"] = list(dict.fromkeys([*sorted_ids, *original_ids]))
            linked_chunk_ids = [
                str(chunk_id)
                for doc in payload["evidence_span_documents"]
                for chunk_id in (doc.get("linked_chunk_ids") or [])
                if str(chunk_id)
            ]
            if linked_chunk_ids:
                payload["selected_chunk_ids"] = list(
                    dict.fromkeys([*linked_chunk_ids, *[str(value) for value in (payload.get("selected_chunk_ids") or []) if str(value)]])
                )
            linked_parent_ids = [
                str(parent_id)
                for doc in payload["evidence_span_documents"]
                for parent_id in (doc.get("linked_parent_ids") or [])
                if str(parent_id)
            ]
            if linked_parent_ids:
                payload["selected_parent_ids"] = list(
                    dict.fromkeys([*linked_parent_ids, *[str(value) for value in (payload.get("selected_parent_ids") or []) if str(value)]])
                )

        retrieval_docs = [doc for doc in (payload.get("retrieval_documents") or []) if isinstance(doc, dict)]
        if retrieval_docs:
            required_docs: List[Dict[str, Any]] = []
            other_docs: List[Dict[str, Any]] = []
            for doc in retrieval_docs:
                source_url = self._source_url_from_record(doc)
                if str(doc.get("id") or "") in set(payload.get("selected_evidence_span_ids") or []) and self._required_page_match_rank(source_url, required_pages):
                    required_docs.append(doc)
                else:
                    other_docs.append(doc)
            if required_docs:
                ordered_required = [
                    doc for _index, doc in sorted(enumerate(required_docs), key=document_sort_key)
                ]
                payload["retrieval_documents"] = [*ordered_required, *other_docs]

    def _augment_payload_for_required_coverage(
        self,
        *,
        query: str,
        payload: Dict[str, Any],
        coverage_plan: Dict[str, Any],
    ) -> bool:
        required_pages = [
            str(value)
            for value in (coverage_plan.get("required_pages") or [])
            if str(value).strip()
        ]
        if not required_pages:
            return False
        existing_span_ids = {str(value) for value in (payload.get("selected_evidence_span_ids") or []) if str(value)}
        existing_fact_ids = {str(value) for value in (payload.get("selected_fact_ids") or []) if str(value)}
        changed = False
        for required_page in required_pages:
            normalized_required = self._normalize_source_url(required_page)
            injected_facts = self._best_required_page_facts(query, required_page, limit=1)
            if injected_facts:
                payload.setdefault("fact_documents", [])
                payload.setdefault("selected_fact_ids", [])
                payload.setdefault("retrieval_documents", [])
                for fact in injected_facts:
                    fact_id = str(fact.get("id") or "")
                    if fact_id and fact_id in existing_fact_ids:
                        continue
                    if self._normalize_source_url(self._source_url_from_record(fact)) != normalized_required:
                        continue
                    payload["fact_documents"].insert(0, fact)
                    payload["retrieval_documents"].insert(0, fact)
                    if fact_id:
                        payload["selected_fact_ids"].insert(0, fact_id)
                        existing_fact_ids.add(fact_id)
                    for chunk_id in fact.get("linked_chunk_ids") or []:
                        if chunk_id:
                            payload.setdefault("selected_chunk_ids", [])
                            if chunk_id not in payload["selected_chunk_ids"]:
                                payload["selected_chunk_ids"].insert(0, chunk_id)
                    for parent_id in fact.get("linked_parent_ids") or []:
                        if parent_id:
                            payload.setdefault("selected_parent_ids", [])
                            if parent_id not in payload["selected_parent_ids"]:
                                payload["selected_parent_ids"].insert(0, parent_id)
                    changed = True
            injected_spans = self._best_required_page_spans(query, required_page, limit=2)
            if not injected_spans:
                continue
            payload.setdefault("evidence_span_documents", [])
            payload.setdefault("selected_evidence_span_ids", [])
            payload.setdefault("retrieval_documents", [])
            for span in injected_spans:
                span_id = str(span.get("id") or "")
                if span_id and span_id in existing_span_ids:
                    continue
                if self._normalize_source_url(self._source_url_from_record(span)) != normalized_required:
                    continue
                payload["evidence_span_documents"].append(span)
                payload["retrieval_documents"].insert(0, span)
                if span_id:
                    payload["selected_evidence_span_ids"].append(span_id)
                    existing_span_ids.add(span_id)
                for chunk_id in span.get("linked_chunk_ids") or []:
                    if chunk_id:
                        payload.setdefault("selected_chunk_ids", [])
                        if chunk_id not in payload["selected_chunk_ids"]:
                            payload["selected_chunk_ids"].insert(0, chunk_id)
                for parent_id in span.get("linked_parent_ids") or []:
                    if parent_id:
                        payload.setdefault("selected_parent_ids", [])
                        if parent_id not in payload["selected_parent_ids"]:
                            payload["selected_parent_ids"].insert(0, parent_id)
                changed = True
        if changed and payload.get("abstained"):
            payload["abstained"] = False
            payload["adjudication_used"] = False
            payload["adjudication_method"] = "required_page_evidence_backfill"
            payload["adjudication_reason"] = "cleared_abstention_after_required_page_span_backfill"
            payload["adjudication_confidence"] = 0.0
            payload["verification_status"] = "backfilled_required_page_evidence"
        return changed

    def retrieve(
        self,
        query: str,
        *,
        query_vector: List[float] | None = None,
        skip_query_planner: bool = False,
        navigation_context: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        routing_started = time.perf_counter()
        mode = classify_query_mode(query)
        media_query = _is_media_query(query)
        unsupported_reason = self._unsupported_intent_reason(query)
        if unsupported_reason:
            routing_latency_ms = (time.perf_counter() - routing_started) * 1000.0
            payload = self._abstained_payload_from_result(
                result={"mode": mode.value},
                reason=unsupported_reason,
                confidence=0.0,
                method="unsupported_intent_guard",
            )
            payload.update(
                {
                    "query": query,
                    "query_rewritten": query,
                    "query_rewrite_labels": [],
                    "graph_query_rewritten": query,
                    "retriever_backend": "abstention_guard",
                    "routing_backend": "abstention_guard",
                    "routing_reason": unsupported_reason,
                    "routing_query_mode": mode.value,
                    "routing_relation_family": "",
                    "routing_relation_confidence": 0.0,
                    "routing_graph_available": False,
                    "routing_graph_init_error": self.graph_init_error,
                    "routing_parallel_vector_graph": False,
                    "routing_parallel_graph_used": False,
                    "routing_graph_rewrite_applied": False,
                    "routing_latency_ms": round(routing_latency_ms, 3),
                    "backend_latency_ms": 0.0,
                    "verification_status": "abstained_unsupported_intent",
                }
            )
            confidence, factors = score_retrieval_confidence(payload)
            payload["retrieval_confidence"] = confidence
            payload["confidence_factors"] = factors
            coverage_plan = self._coverage_plan_for_result(
                query=query,
                payload=payload,
                mode=mode,
            )
            budget_items, budget_chars, budget_max_per_source = self._evidence_budget_for_plan(coverage_plan)
            payload["evidence_pack"] = build_evidence_pack(
                query=query,
                result=payload,
                max_items=budget_items,
                max_chars=budget_chars,
                max_per_source=budget_max_per_source,
                coverage_plan=coverage_plan,
            )
            coverage_plan["coverage_status"] = payload["evidence_pack"].get("coverage_status") or coverage_plan["coverage_status"]
            payload["coverage_status"] = coverage_plan["coverage_status"]
            payload["missing_required_entities"] = payload["evidence_pack"].get("missing_required_entities") or []
            payload["missing_required_pages"] = payload["evidence_pack"].get("missing_required_pages") or []
            payload["missing_required_sections"] = payload["evidence_pack"].get("missing_required_sections") or []
            payload["retrieval_trace"] = {
                "backend": payload["routing_backend"],
                "reason": payload["routing_reason"],
                "query_mode": payload["routing_query_mode"],
                "query_rewrite_labels": [],
                "routing_latency_ms": payload["routing_latency_ms"],
                "backend_latency_ms": payload["backend_latency_ms"],
                "lane_latency_ms": {},
                "rerank_latency_ms": 0.0,
                "rerank_method": "",
                "graph_available": False,
                "graph_error": "",
                "candidate_counts": {
                    "dense_chunks": 0,
                    "sparse_chunks": 0,
                    "dense_parents": 0,
                    "sparse_parents": 0,
                    "dense_summaries": 0,
                    "sparse_summaries": 0,
                    "dense_facts": 0,
                    "sparse_facts": 0,
                    "dense_evidence_spans": 0,
                    "sparse_evidence_spans": 0,
                    "local_evidence_spans": 0,
                    "dense_assertions": 0,
                    "sparse_assertions": 0,
                },
                "selected": {
                    "chunks": [],
                    "answers": [],
                    "facts": [],
                    "evidence_spans": [],
                    "parents": [],
                    "media": [],
                },
                "coverage_plan": coverage_plan,
                "confidence": confidence,
                "verification_status": payload["verification_status"],
            }
            return payload

        decision = self._route_query(query)
        relation_plan = decision.relation_plan if decision.graph_available else None
        rewrites = self._build_query_rewrite_bundle(
            query,
            relation_plan=relation_plan,
            query_mode=mode.value,
            use_query_planner=not skip_query_planner,
        )
        if skip_query_planner:
            rewrites = QueryRewriteBundle(
                vector_query=rewrites.vector_query,
                graph_query=rewrites.graph_query,
                labels=tuple(dict.fromkeys(["upstream_query_plan", *rewrites.labels])),
                navigation_intent=rewrites.navigation_intent,
                navigation_goal=rewrites.navigation_goal,
                navigation_confidence=rewrites.navigation_confidence,
                navigation_source=rewrites.navigation_source,
            )
        planned_navigation_context = normalize_navigation_context(
            query,
            navigation_context
            or {
                "intent": rewrites.navigation_intent,
                "goal": rewrites.navigation_goal,
                "confidence": rewrites.navigation_confidence,
                "source": rewrites.navigation_source,
            },
        )
        query_embedding_status = "ok"
        query_embedding_error = ""
        if query_vector is not None and "hyde_expansion" not in rewrites.labels:
            query_vector = list(query_vector)
            if not query_vector:
                query_embedding_status = "skipped_dense_no_query_vector"
        else:
            try:
                query_vector = self.vector.embed_query(rewrites.vector_query if "hyde_expansion" in rewrites.labels else query)
            except Exception as exc:
                query_vector = []
                query_embedding_status = "failed_sparse_local_fallback"
                query_embedding_error = _public_query_embedding_error_code(exc)
                logger.warning(
                    "Routed dense query embedding failed; continuing with sparse/local retrieval fallback: %s",
                    exc,
                )
        routing_latency_ms = (time.perf_counter() - routing_started) * 1000.0

        backend_started = time.perf_counter()
        graph_context = self._empty_graph_context(query)
        graph_error: str | None = None
        vector_backend_latency_ms = 0.0
        graph_context_latency_ms = 0.0
        graph_augment_latency_ms = 0.0

        def _timed_vector_retrieve() -> tuple[Dict[str, Any], float]:
            started = time.perf_counter()
            retrieved = self.vector.retrieve(
                rewrites.vector_query,
                query_vector=query_vector,
            )
            return retrieved, round((time.perf_counter() - started) * 1000.0, 3)

        def _timed_graph_context() -> tuple[GraphQueryContext, float]:
            started = time.perf_counter()
            context = self.graph.prepare_query_context(
                rewrites.graph_query,
                mode=mode,
                media_query=media_query,
                relation_plan=relation_plan,
            )
            return context, round((time.perf_counter() - started) * 1000.0, 3)

        try:
            if decision.backend == "parallel_hybrid" and self.graph is not None:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    vector_future = executor.submit(_timed_vector_retrieve)
                    graph_future = executor.submit(_timed_graph_context)
                    result, vector_backend_latency_ms = vector_future.result()
                    graph_context, graph_context_latency_ms = graph_future.result()
            else:
                result, vector_backend_latency_ms = _timed_vector_retrieve()
        except Exception as exc:
            if decision.backend == "parallel_hybrid" and self.routed_fallback_to_vector:
                graph_error = "graph_context_failed"
                logger.warning("Graph context preparation failed; falling back to vector retrieval: %s", exc)
                backend_started = time.perf_counter()
                result, vector_backend_latency_ms = _timed_vector_retrieve()
                graph_context = self._empty_graph_context(query)
                decision = RoutingDecision(
                    backend="vector",
                    reason="graph_error_fallback",
                    query_mode=decision.query_mode,
                    relation_family=decision.relation_family,
                    relation_confidence=decision.relation_confidence,
                    graph_available=decision.graph_available,
                )
            else:
                raise

        if decision.backend == "parallel_hybrid" and self.graph is not None:
            try:
                graph_augment_started = time.perf_counter()
                result = self.graph.augment_result(
                    query,
                    result,
                    relation_plan=graph_context.relation_plan,
                    relation_candidates=graph_context.relation_candidates,
                )
                graph_augment_latency_ms = round((time.perf_counter() - graph_augment_started) * 1000.0, 3)
            except Exception as exc:
                if self.routed_fallback_to_vector:
                    graph_augment_latency_ms = round((time.perf_counter() - graph_augment_started) * 1000.0, 3)
                    graph_error = "graph_augmentation_failed"
                    logger.warning("Graph result augmentation failed; falling back to vector retrieval: %s", exc)
                    decision = RoutingDecision(
                        backend="vector",
                        reason="graph_error_fallback",
                        query_mode=decision.query_mode,
                        relation_family=decision.relation_family,
                        relation_confidence=decision.relation_confidence,
                        graph_available=decision.graph_available,
                    )
                else:
                    raise
        backend_latency_ms = (time.perf_counter() - backend_started) * 1000.0

        payload = dict(result or {})
        preliminary_confidence, preliminary_factors = score_retrieval_confidence(payload)
        payload["retrieval_confidence"] = preliminary_confidence
        payload["confidence_factors"] = preliminary_factors
        payload = self._apply_evidence_adjudication(query, payload)
        payload["query"] = query
        payload["query_rewritten"] = rewrites.vector_query
        payload["query_rewrite_labels"] = list(dict.fromkeys([*rewrites.labels, *graph_context.rewrite_labels]))
        payload["graph_query_rewritten"] = graph_context.rewritten_query if decision.graph_available else query
        payload["retriever_backend"] = decision.backend
        payload["routing_backend"] = decision.backend
        payload["routing_reason"] = decision.reason
        payload["routing_query_mode"] = decision.query_mode
        payload["routing_relation_family"] = (
            graph_context.relation_plan.family
            if graph_context.relation_plan is not None
            else decision.relation_family
        )
        payload["routing_relation_confidence"] = float(
            graph_context.relation_candidates.confidence
            if graph_context.relation_plan is not None
            else decision.relation_confidence
        )
        payload["routing_graph_available"] = bool(decision.graph_available)
        payload["routing_graph_init_error"] = self.graph_init_error
        payload["routing_parallel_vector_graph"] = decision.backend == "parallel_hybrid"
        payload["routing_parallel_graph_used"] = bool(decision.backend == "parallel_hybrid" and self.graph is not None)
        payload["routing_graph_rewrite_applied"] = graph_context.rewritten_query != query
        payload["routing_latency_ms"] = round(routing_latency_ms, 3)
        payload["backend_latency_ms"] = round(backend_latency_ms, 3)
        payload["vector_backend_latency_ms"] = payload.get("vector_backend_latency_ms") or vector_backend_latency_ms
        if query_embedding_status != "ok":
            payload["query_embedding_status"] = query_embedding_status
            payload["query_embedding_error"] = query_embedding_error
        else:
            payload["query_embedding_status"] = payload.get("query_embedding_status") or query_embedding_status
            payload["query_embedding_error"] = payload.get("query_embedding_error") or query_embedding_error
        payload["graph_context_latency_ms"] = graph_context_latency_ms
        payload["graph_augment_latency_ms"] = graph_augment_latency_ms
        if graph_error:
            payload["routing_graph_error"] = graph_error
        confidence, factors = score_retrieval_confidence(payload)
        payload["retrieval_confidence"] = confidence
        payload["confidence_factors"] = factors
        coverage_plan = self._coverage_plan_for_result(
            query=query,
            payload=payload,
            mode=mode,
        )
        if self._augment_payload_for_required_coverage(
            query=query,
            payload=payload,
            coverage_plan=coverage_plan,
        ):
            confidence, factors = score_retrieval_confidence(payload)
            payload["retrieval_confidence"] = confidence
            payload["confidence_factors"] = factors
            coverage_plan = self._coverage_plan_for_result(
                query=query,
                payload=payload,
                mode=mode,
            )
        self._prioritize_required_page_evidence(
            query=query,
            payload=payload,
            coverage_plan=coverage_plan,
        )
        selected_parent_ids = [
            str(value)
            for value in (payload.get("selected_parent_ids") or [])
            if str(value)
        ]
        if selected_parent_ids:
            payload["selected_parent_ids"] = self.vector._diversify_parent_ids_for_query(
                query,
                selected_parent_ids,
                limit=len(selected_parent_ids),
            )
        budget_items, budget_chars, budget_max_per_source = self._evidence_budget_for_plan(coverage_plan)
        payload["evidence_pack"] = build_evidence_pack(
            query=query,
            result=payload,
            max_items=budget_items,
            max_chars=budget_chars,
            max_per_source=budget_max_per_source,
            coverage_plan=coverage_plan,
        )
        coverage_plan["coverage_status"] = payload["evidence_pack"].get("coverage_status") or coverage_plan["coverage_status"]
        payload["coverage_status"] = coverage_plan["coverage_status"]
        payload["missing_required_entities"] = payload["evidence_pack"].get("missing_required_entities") or []
        payload["missing_required_pages"] = payload["evidence_pack"].get("missing_required_pages") or []
        payload["missing_required_sections"] = payload["evidence_pack"].get("missing_required_sections") or []
        if self.navigation_plan_enabled:
            payload["navigation_plan"] = self.navigation_planner.plan(
                query=query,
                result=payload,
                navigation_context=planned_navigation_context,
            )
            payload["navigation_intent"] = planned_navigation_context["intent"]
        payload["retrieval_trace"] = {
            "backend": decision.backend,
            "reason": decision.reason,
            "query_mode": decision.query_mode,
            "query_rewrite_labels": payload.get("query_rewrite_labels") or [],
            "routing_latency_ms": payload.get("routing_latency_ms"),
            "backend_latency_ms": payload.get("backend_latency_ms"),
            "lane_latency_ms": payload.get("lane_latency_ms") or {},
            "rerank_latency_ms": payload.get("rerank_latency_ms") or 0.0,
            "rerank_method": payload.get("rerank_method") or "",
            "vector_backend_latency_ms": payload.get("vector_backend_latency_ms") or 0.0,
            "graph_context_latency_ms": payload.get("graph_context_latency_ms") or 0.0,
            "graph_augment_latency_ms": payload.get("graph_augment_latency_ms") or 0.0,
            "stage_latency_ms": payload.get("stage_latency_ms") if isinstance(payload.get("stage_latency_ms"), dict) else {},
            "graph_available": bool(decision.graph_available),
            "graph_error": graph_error or "",
            "candidate_counts": {
                "dense_chunks": len(payload.get("dense_chunk_ids") or []),
                "sparse_chunks": len(payload.get("sparse_chunk_ids") or []),
                "dense_parents": len(payload.get("dense_parent_ids") or []),
                "sparse_parents": len(payload.get("sparse_parent_ids") or []),
                "dense_summaries": len(payload.get("dense_summary_ids") or []),
                "sparse_summaries": len(payload.get("sparse_summary_ids") or []),
                "dense_facts": len(payload.get("dense_fact_ids") or []),
                "sparse_facts": len(payload.get("sparse_fact_ids") or []),
                "dense_evidence_spans": len(payload.get("dense_evidence_span_ids") or []),
                "sparse_evidence_spans": len(payload.get("sparse_evidence_span_ids") or []),
                "local_evidence_spans": len(payload.get("local_evidence_span_ids") or []),
                "dense_assertions": len(payload.get("dense_assertion_ids") or []),
                "sparse_assertions": len(payload.get("sparse_assertion_ids") or []),
            },
            "selected": {
                "chunks": payload.get("selected_chunk_ids") or [],
                "answers": payload.get("selected_answer_ids") or [],
                "facts": payload.get("selected_fact_ids") or [],
                "evidence_spans": payload.get("selected_evidence_span_ids") or [],
                "parents": payload.get("selected_parent_ids") or [],
                "media": payload.get("selected_media_ids") or [],
            },
            "coverage_plan": coverage_plan,
            "confidence": confidence,
            "verification_status": payload.get("verification_status") or "",
            "navigation": {
                "enabled": bool(self.navigation_plan_enabled),
                "catalog_available": bool(self.navigation_planner.available),
                "intent": planned_navigation_context["intent"],
                "status": (
                    payload.get("navigation_plan") or {}
                ).get("status"),
                "step_count": len(
                    (payload.get("navigation_plan") or {}).get("steps") or []
                ),
            },
        }
        return payload
