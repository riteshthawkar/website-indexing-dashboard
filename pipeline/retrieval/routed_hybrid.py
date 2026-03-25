from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pipeline.core.evidence_adjudicator import adjudicate_factual_evidence
from pipeline.core.query_planner import plan_query

from .adaptive_hybrid import (
    AdaptiveHybridRetriever,
    QueryMode,
    _is_generic_contact_query,
    _is_media_query,
    _query_intent,
    _semantic_query_alias_tokens,
    _tokenize,
    classify_query_mode,
)
from .graph_rag import GraphQueryContext, GraphRAGRetriever, RelationCandidateSet, RelationQueryPlan


def _with_retriever_backend(config: Dict[str, Any], backend: str) -> Dict[str, Any]:
    payload = deepcopy(config or {})
    retrieval_cfg = dict(payload.get("retrieval") or {})
    retrieval_cfg["retriever_backend"] = str(backend)
    payload["retrieval"] = retrieval_cfg
    return payload


@dataclass(frozen=True)
class RoutingDecision:
    backend: str
    reason: str
    query_mode: str
    relation_family: str = ""
    relation_confidence: float = 0.0
    graph_available: bool = False


@dataclass(frozen=True)
class QueryRewriteBundle:
    vector_query: str
    graph_query: str
    labels: tuple[str, ...]


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

        self.routed_graph_enabled = bool(retrieval_cfg.get("routed_graph_enabled", True))
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
        self.query_planner_min_confidence = float(retrieval_cfg.get("query_planner_min_confidence") or 0.55)
        self.evidence_adjudicator_enabled = bool(retrieval_cfg.get("evidence_adjudicator_enabled", False))
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
        self.supports_shared_parallel_retrieval = False

        vector_config = _with_retriever_backend(self.config, "vector")
        self.vector = AdaptiveHybridRetriever(config=vector_config, work_dir=self.work_dir)
        self.model = self.vector.model
        self.output_dimensionality = self.vector.output_dimensionality

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
                self.graph_init_error = str(exc)
                self.graph = None

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
    ) -> QueryRewriteBundle:
        labels: List[str] = []
        vector_query = query
        graph_query = query
        generic_contact_query = _is_generic_contact_query(query)
        if self.query_planner_enabled:
            plan = plan_query(
                query=query,
                model=self.query_planner_model,
                fallback_query_type=query_mode,
            )
            planner_confidence = float(plan.get("confidence") or 0.0)
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
        return QueryRewriteBundle(
            vector_query=vector_query,
            graph_query=graph_query,
            labels=tuple(dict.fromkeys(labels)),
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
            return result
        payload = dict(result or {})
        if payload.get("abstained"):
            return payload
        if str(payload.get("mode") or "").strip().lower() != QueryMode.FACT.value:
            payload.setdefault("adjudication_used", False)
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
            return payload

        adjudication = adjudicate_factual_evidence(
            query=query,
            intent_summary=self._intent_summary(query),
            answer_documents=answer_documents[: self.evidence_adjudicator_answer_limit],
            fact_documents=fact_documents[: self.evidence_adjudicator_fact_limit],
            retrieval_documents=retrieval_documents[: max(8, self.evidence_adjudicator_chunk_limit + 2)],
            model=self.evidence_adjudicator_model,
            reasoning_effort=self.evidence_adjudicator_reasoning_effort,
            min_confidence=self.evidence_adjudicator_min_confidence,
            max_completion_tokens=self.evidence_adjudicator_max_completion_tokens,
            retries=self.evidence_adjudicator_retries,
            retry_delay_sec=self.evidence_adjudicator_retry_delay_sec,
            per_request_delay_sec=self.evidence_adjudicator_per_request_delay_sec,
            max_answer_ids=self.evidence_adjudicator_answer_limit,
            max_fact_ids=self.evidence_adjudicator_fact_limit,
            max_chunk_ids=self.evidence_adjudicator_chunk_limit,
        )
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
        )

    def retrieve(self, query: str, *, query_vector: List[float] | None = None) -> Dict[str, Any]:
        routing_started = time.perf_counter()
        mode = classify_query_mode(query)
        media_query = _is_media_query(query)
        query_vector = list(query_vector) if query_vector is not None else self.vector.embed_query(query)
        decision = self._route_query(query)
        relation_plan = self._graph_relation_plan(query) if decision.graph_available else None
        rewrites = self._build_query_rewrite_bundle(query, relation_plan=relation_plan, query_mode=mode.value)
        routing_latency_ms = (time.perf_counter() - routing_started) * 1000.0

        backend_started = time.perf_counter()
        graph_context = self._empty_graph_context(query)
        graph_error: str | None = None
        try:
            if decision.backend == "parallel_hybrid" and self.graph is not None:
                with ThreadPoolExecutor(max_workers=2) as executor:
                    vector_future = executor.submit(
                        self.vector.retrieve,
                        rewrites.vector_query,
                        query_vector=query_vector,
                    )
                    graph_future = executor.submit(
                        self.graph.prepare_query_context,
                        rewrites.graph_query,
                        mode=mode,
                        media_query=media_query,
                        relation_plan=relation_plan,
                    )
                    result = vector_future.result()
                    graph_context = graph_future.result()
            else:
                result = self.vector.retrieve(rewrites.vector_query, query_vector=query_vector)
        except Exception as exc:
            if decision.backend == "parallel_hybrid" and self.routed_fallback_to_vector:
                graph_error = str(exc)
                backend_started = time.perf_counter()
                result = self.vector.retrieve(rewrites.vector_query, query_vector=query_vector)
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
                result = self.graph.augment_result(
                    query,
                    result,
                    relation_plan=graph_context.relation_plan,
                    relation_candidates=graph_context.relation_candidates,
                )
            except Exception as exc:
                if self.routed_fallback_to_vector:
                    graph_error = str(exc)
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

        payload = self._apply_evidence_adjudication(query, dict(result or {}))
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
        if graph_error:
            payload["routing_graph_error"] = graph_error
        return payload
