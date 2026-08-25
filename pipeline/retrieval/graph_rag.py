from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from pipeline.core.graph_artifacts import resolve_canonical_graph_artifacts
from pipeline.core.io import load_json_safe
from pipeline.core.knowledge_graph import load_graph_bundle, validate_graph_bundle
from pipeline.core.media import build_retrieval_documents
from pipeline.retrieval.adaptive_hybrid import (
    QueryMode,
    _clean_text,
    _hours_query_alias_tokens,
    _is_media_query,
    _lookup_query_profile,
    _named_query_tokens,
    _query_starts_with,
    _semantic_query_alias_tokens,
    _tokenize,
    classify_query_mode,
)
from pipeline.stages.embedders.neo4j_graph_store import _env_or_config, _neo4j_endpoint, _post_query


_GENERIC_RELATION_TYPES = {"RELATED_TO", "ABOUT", "MENTIONS", "OTHER"}
_GRAPH_FIRST_RELATION_FAMILIES = {"location", "naming", "affiliation"}
_ACTIVE_ASSERTION_STATUSES = {"active", "valid"}
_OFFERING_ROUTER_TOKENS = {
    "parking",
    "accommodation",
    "housing",
    "shuttle",
    "transport",
    "transportation",
    "service",
    "services",
    "hours",
    "support",
    "family",
    "parents",
}
_EXPLICIT_LOCATION_ROUTER_TOKENS = {"city", "emirate", "located", "location", "based", "situated"}
_STRONG_OFFERING_ROUTER_TOKENS = {"parking", "accommodation", "housing", "shuttle", "transport", "transportation"}

_RELATION_QUERY_RULES: Dict[str, Dict[str, Any]] = {
    "contact": {
        "phrases": (
            "email address",
            "contact email",
            "phone number",
            "telephone number",
            "website address",
            "how can i contact",
            "how do i contact",
            "who should i email",
            "who should i contact",
        ),
        "tokens": {"contact", "email", "mail", "phone", "telephone", "extension", "website", "url", "reach"},
        "aliases": ("contact", "email", "mail", "phone", "telephone", "number", "website", "url", "reach", "office", "admission", "admissions"),
        "primary_relation_types": ("CONTACTS",),
        "secondary_relation_types": ("ABOUT", "RELATED_TO", "MENTIONS"),
    },
    "location": {
        "phrases": ("where is", "where in", "in which city", "what city", "which city", "which emirate", "in which emirate"),
        "tokens": {"where", "city", "emirate", "located", "location", "based", "situated"},
        "aliases": ("located", "based", "location", "city", "campus", "abu", "dhabi", "masdar"),
        "primary_relation_types": ("LOCATED_IN", "PART_OF"),
        "secondary_relation_types": ("ABOUT", "RELATED_TO"),
    },
    "naming": {
        "phrases": ("named after", "whose name", "carry the name", "after whom"),
        "tokens": {"named", "name", "after", "carry"},
        "aliases": ("named", "after", "name", "fahad", "bin", "sultan", "zayed"),
        "primary_relation_types": ("NAMED_AFTER", "ABOUT"),
        "secondary_relation_types": ("RELATED_TO", "MENTIONS"),
    },
    "affiliation": {
        "phrases": ("affiliated with", "institutional affiliation", "authority it is affiliated", "under which law", "legal basis", "established under law"),
        "tokens": {"affiliation", "affiliated", "authority", "law", "legal", "basis", "established"},
        "aliases": ("affiliated", "authority", "law", "legal", "established", "executive", "council", "part"),
        "primary_relation_types": ("AFFILIATED_WITH", "PART_OF"),
        "secondary_relation_types": ("ABOUT", "RELATED_TO"),
    },
    "transport": {
        "phrases": (
            "shuttle bus service",
            "shuttle service",
            "parking provided",
            "parking available",
            "parking permitted",
            "transport service",
        ),
        "tokens": {"parking", "shuttle", "transport", "transportation", "bus", "visitor", "visitors"},
        "aliases": ("parking", "shuttle", "transport", "transportation", "bus", "service", "visitor", "visitors", "guest", "guests"),
        "primary_relation_types": ("OFFERS", "HOSTS", "RELATED_TO"),
        "secondary_relation_types": ("ABOUT", "MENTIONS"),
    },
    "accommodation": {
        "phrases": (
            "student accommodation",
            "student housing",
            "on-campus accommodation",
            "stay on campus",
            "parents stay",
            "family members stay",
            "visiting family members stay",
        ),
        "tokens": {"accommodation", "housing", "stay", "parents", "family", "visiting", "residence", "residences"},
        "aliases": ("accommodation", "housing", "stay", "campus", "residence", "residences", "parents", "family", "visiting", "hotel", "airbnb"),
        "primary_relation_types": ("OFFERS", "HOSTS", "RELATED_TO"),
        "secondary_relation_types": ("ABOUT", "MENTIONS"),
    },
    "hours": {
        "phrases": (
            "working hours",
            "operating hours",
            "office hours",
            "weekday operating hours",
            "weekday working hours",
        ),
        "tokens": {"hours", "hour", "time", "times", "working", "operating", "weekday", "weekdays"},
        "aliases": ("hours", "working", "operating", "weekday", "official", "monday", "thursday", "friday"),
        "primary_relation_types": ("OFFERS", "RELATED_TO"),
        "secondary_relation_types": ("ABOUT", "MENTIONS"),
    },
    "amenities": {
        "phrases": (
            "campus amenities",
            "support facilities",
            "support services",
            "facilities on campus",
            "amenities on campus",
        ),
        "tokens": {"amenity", "amenities", "facility", "facilities", "support", "services", "campus"},
        "aliases": ("amenities", "facilities", "support", "services", "campus", "library", "canteen", "gym", "lounge", "health"),
        "primary_relation_types": ("OFFERS", "HOSTS", "RELATED_TO"),
        "secondary_relation_types": ("RELATED_TO", "ABOUT", "MENTIONS"),
    },
}


@dataclass(frozen=True)
class RelationQueryPlan:
    family: str
    confidence: float
    primary_relation_types: Tuple[str, ...]
    secondary_relation_types: Tuple[str, ...]
    alias_tokens: Tuple[str, ...]
    entity_tokens: Tuple[str, ...]
    graph_first: bool


@dataclass
class RelationCandidateSet:
    family: str = ""
    confidence: float = 0.0
    graph_first: bool = False
    assertion_ids: List[str] = None  # type: ignore[assignment]
    fact_ids: List[str] = None  # type: ignore[assignment]
    chunk_ids: List[str] = None  # type: ignore[assignment]
    parent_ids: List[str] = None  # type: ignore[assignment]
    media_ids: List[str] = None  # type: ignore[assignment]
    chunk_scores: Dict[str, float] = None  # type: ignore[assignment]
    parent_scores: Dict[str, float] = None  # type: ignore[assignment]
    assertion_scores: Dict[str, float] = None  # type: ignore[assignment]
    fact_scores: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.assertion_ids = list(self.assertion_ids or [])
        self.fact_ids = list(self.fact_ids or [])
        self.chunk_ids = list(self.chunk_ids or [])
        self.parent_ids = list(self.parent_ids or [])
        self.media_ids = list(self.media_ids or [])
        self.chunk_scores = dict(self.chunk_scores or {})
        self.parent_scores = dict(self.parent_scores or {})
        self.assertion_scores = dict(self.assertion_scores or {})
        self.fact_scores = dict(self.fact_scores or {})


@dataclass(frozen=True)
class GraphQueryContext:
    mode: QueryMode
    media_query: bool
    relation_plan: RelationQueryPlan | None
    relation_candidates: RelationCandidateSet
    rewritten_query: str
    rewrite_labels: Tuple[str, ...]


class GraphRAGRetriever:
    """
    Graph-augmented wrapper around the existing vector-first retriever.

    This is intentionally a sidecar, not a replacement for the dense+sparse
    vector stack. Candidate generation stays vector-first; the graph adds
    bounded structural/fact/media expansion.
    """

    def __init__(self, *, config: Dict[str, Any], work_dir: str | Path, base_retriever: Any | None = None):
        from .adaptive_hybrid import AdaptiveHybridRetriever

        self.config = config
        self.work_dir = Path(work_dir).resolve()
        self._owns_base_retriever = base_retriever is None
        self.base = base_retriever or AdaptiveHybridRetriever(config=config, work_dir=self.work_dir)
        self.model = self.base.model
        self.output_dimensionality = self.base.output_dimensionality

        retrieval_cfg = dict(config.get("retrieval") or {})
        self.graph_max_fact_results = int(retrieval_cfg.get("graph_max_fact_results") or 4)
        self.graph_max_additional_media_results = int(retrieval_cfg.get("graph_max_additional_media_results") or 2)
        self.graph_max_context_documents = int(retrieval_cfg.get("graph_max_context_documents") or 4)
        self.graph_max_assertion_results = int(retrieval_cfg.get("graph_max_assertion_results") or 4)
        self.graph_chunk_fact_bonus = float(retrieval_cfg.get("graph_chunk_fact_bonus") or 0.25)
        self.graph_parent_fact_bonus = float(retrieval_cfg.get("graph_parent_fact_bonus") or 0.15)
        self.graph_chunk_media_bonus = float(retrieval_cfg.get("graph_chunk_media_bonus") or 0.20)
        self.graph_parent_media_bonus = float(retrieval_cfg.get("graph_parent_media_bonus") or 0.12)
        self.graph_assertion_bonus = float(retrieval_cfg.get("graph_assertion_bonus") or 0.30)
        self.graph_max_additional_chunk_candidates = int(retrieval_cfg.get("graph_max_additional_chunk_candidates") or 6)
        self.graph_max_additional_parent_candidates = int(retrieval_cfg.get("graph_max_additional_parent_candidates") or 4)
        self.graph_query_backend = str(retrieval_cfg.get("graph_query_backend") or "auto").strip().lower()
        self.neo4j_query_limit = int(retrieval_cfg.get("graph_neo4j_query_limit") or 64)
        self.graph_relation_router_enabled = bool(retrieval_cfg.get("graph_relation_router_enabled", True))
        self.graph_relation_only = bool(retrieval_cfg.get("graph_relation_only", True))
        self.graph_relation_route_min_confidence = float(retrieval_cfg.get("graph_relation_route_min_confidence") or 0.38)
        self.graph_relation_graph_first_min_confidence = float(retrieval_cfg.get("graph_relation_graph_first_min_confidence") or 0.62)
        self.graph_relation_reorder_min_confidence = float(retrieval_cfg.get("graph_relation_reorder_min_confidence") or 0.86)
        self.graph_relation_assertion_limit = int(retrieval_cfg.get("graph_relation_assertion_limit") or 4)
        self.graph_relation_fact_limit = int(retrieval_cfg.get("graph_relation_fact_limit") or 6)
        self.graph_relation_local_candidate_limit = int(retrieval_cfg.get("graph_relation_local_candidate_limit") or 12)
        self.graph_relation_prefer_local = bool(retrieval_cfg.get("graph_relation_prefer_local", True))
        self.graph_relation_max_query_tokens = int(retrieval_cfg.get("graph_relation_max_query_tokens") or 18)
        self.graph_relation_chunk_seed_bonus = float(retrieval_cfg.get("graph_relation_chunk_seed_bonus") or 2.2)
        self.graph_relation_parent_seed_bonus = float(retrieval_cfg.get("graph_relation_parent_seed_bonus") or 1.4)
        self.graph_relation_fact_seed_bonus = float(retrieval_cfg.get("graph_relation_fact_seed_bonus") or 2.5)
        graph_cfg = dict(config.get("graph") or {})
        self.neo4j_timeout_sec = int(graph_cfg.get("neo4j_http_timeout_sec") or 60)
        self.supports_shared_parallel_retrieval = bool(
            getattr(self.base, "supports_shared_parallel_retrieval", False)
        )

        self._thread_state = threading.local()
        self._local_graph_load_lock = threading.Lock()

        manifest_file = self.work_dir / "stage_outputs" / "upload_graph" / "neo4j_upload_manifest.json"
        self.neo4j_manifest = load_json_safe(manifest_file, {}) if manifest_file.is_file() else {}
        if not isinstance(self.neo4j_manifest, dict):
            self.neo4j_manifest = {}
        self.neo4j_namespace = str(
            self.neo4j_manifest.get("neo4j_namespace")
            or _env_or_config(graph_cfg, "neo4j_namespace", "NEO4J_NAMESPACE")
            or ""
        ).strip()
        self.neo4j_uri = _env_or_config(graph_cfg, "neo4j_uri", "NEO4J_URI")
        self.neo4j_database = _env_or_config(graph_cfg, "neo4j_database", "NEO4J_DATABASE") or "neo4j"
        self.neo4j_username = _env_or_config(graph_cfg, "neo4j_username", "NEO4J_USERNAME")
        self.neo4j_password = _env_or_config(graph_cfg, "neo4j_password", "NEO4J_PASSWORD")
        self.neo4j_enabled = bool(
            self.neo4j_uri and self.neo4j_username and self.neo4j_password and self.neo4j_namespace
        )

        self.local_graph_available = False
        self._graph_file: Path | None = None
        self._graph_index_file: Path | None = None
        self.local_graph_kind = ""
        self.local_graph_sha256 = ""
        self.local_graph_index_sha256 = ""
        self._local_graph_loaded = False
        graph_artifacts = resolve_canonical_graph_artifacts(
            self.work_dir,
            required=False,
            require_index=True,
            validate_binding=bool(
                (config.get("pipeline") or {}).get("production_profile", False)
            ),
        )
        if graph_artifacts is not None:
            self.local_graph_available = True
            self._graph_file = graph_artifacts.graph_file
            self._graph_index_file = graph_artifacts.index_file
            self.local_graph_kind = graph_artifacts.kind
            self.local_graph_sha256 = graph_artifacts.graph_sha256
            self.local_graph_index_sha256 = graph_artifacts.index_sha256
        if not self.local_graph_available and not self.neo4j_enabled:
            raise ValueError(
                "GraphRAG requested but neither local graph artifacts nor Neo4j graph settings are available. "
                "Run the knowledge_graph formatter stage, sync the graph to Neo4j, or use a vector-only retrieval config."
            )
        self.graph_bundle = {}
        self.graph_index = {}
        self.node_map: Dict[str, Dict[str, Any]] = {}
        self.edge_map: Dict[str, Dict[str, Any]] = {}
        self.entity_map: Dict[str, Dict[str, Any]] = {}
        self.assertion_map: Dict[str, Dict[str, Any]] = {}
        self.community_map: Dict[str, Dict[str, Any]] = {}
        self.outgoing_edge_ids: Dict[str, List[str]] = {}
        should_eager_load_local = self.local_graph_available and (
            not self.neo4j_enabled or self.graph_query_backend == "local"
        )
        if should_eager_load_local:
            self._ensure_local_graph_loaded()

    def close(self) -> None:
        if not getattr(self, "_owns_base_retriever", False):
            return
        close_base = getattr(getattr(self, "base", None), "close", None)
        if callable(close_base):
            close_base()

    def embed_query(self, query: str) -> List[float]:
        return self.base.embed_query(query)

    def embed_queries(self, queries: Sequence[str]) -> List[List[float]]:
        return self.base.embed_queries(queries)

    def _append_query_aliases(
        self,
        query: str,
        alias_tokens: Sequence[str],
        *,
        max_new_tokens: int = 8,
    ) -> Tuple[str, Tuple[str, ...]]:
        existing_tokens = set(_tokenize(query))
        additions: List[str] = []
        for token in alias_tokens:
            token = str(token or "").strip().lower()
            if not token or token in existing_tokens or token in additions:
                continue
            additions.append(token)
            if len(additions) >= max_new_tokens:
                break
        if not additions:
            return query, tuple()
        return f"{query} {' '.join(additions)}".strip(), tuple(additions)

    def _neo4j_node_cache(self) -> Dict[str, Dict[str, Any]]:
        cache = getattr(self._thread_state, "neo4j_node_cache", None)
        if cache is None:
            cache = {}
            self._thread_state.neo4j_node_cache = cache
        return cache

    def _neo4j_last_error(self) -> str | None:
        return getattr(self._thread_state, "neo4j_last_error", None)

    def _set_neo4j_last_error(self, value: str | None) -> None:
        self._thread_state.neo4j_last_error = value

    def _is_active_assertion_node(self, node: Dict[str, Any]) -> bool:
        props = dict(node.get("properties") or {})
        status = _clean_text(props.get("validity_status") or "active").lower()
        return status in _ACTIVE_ASSERTION_STATUSES

    def _ensure_local_graph_loaded(self) -> None:
        if self._local_graph_loaded or not self.local_graph_available:
            return
        with self._local_graph_load_lock:
            if self._local_graph_loaded or not self.local_graph_available:
                return
            graph_file = self._graph_file
            graph_index_file = self._graph_index_file
            if graph_file is None or graph_index_file is None:
                raise ValueError("Local graph artifacts are not configured")
            graph_bundle = load_graph_bundle(graph_file)
            issues = validate_graph_bundle(graph_bundle)
            if issues:
                raise ValueError(f"Knowledge graph bundle is invalid: {issues[0].get('message')}")
            graph_index = load_json_safe(graph_index_file, {}) or {}
            node_map = {
                str(node.get("id") or ""): node
                for node in (graph_bundle.get("nodes") or [])
                if isinstance(node, dict) and str(node.get("id") or "")
            }
            edge_map = {
                str(edge.get("id") or ""): edge
                for edge in (graph_bundle.get("edges") or [])
                if isinstance(edge, dict) and str(edge.get("id") or "")
            }
            entity_map = {
                str(node.get("id") or ""): node
                for node in (graph_bundle.get("nodes") or [])
                if isinstance(node, dict) and str(node.get("node_type") or "") == "entity"
            }
            assertion_map = {
                str(node.get("id") or ""): node
                for node in (graph_bundle.get("nodes") or [])
                if (
                    isinstance(node, dict)
                    and str(node.get("node_type") or "") == "relation_assertion"
                    and self._is_active_assertion_node(node)
                )
            }
            community_map = {
                str(node.get("id") or ""): node
                for node in (graph_bundle.get("nodes") or [])
                if isinstance(node, dict) and str(node.get("node_type") or "") == "community"
            }
            outgoing = graph_index.get("outgoing_edge_ids") or {}
            outgoing_edge_ids = {
                str(node_id): [str(edge_id) for edge_id in edge_ids if str(edge_id)]
                for node_id, edge_ids in outgoing.items()
                if isinstance(edge_ids, list)
            }
            # Publish the complete immutable graph snapshot only after every
            # structure has validated, so concurrent readers never see a
            # partially initialized graph.
            self.graph_bundle = graph_bundle
            self.graph_index = graph_index
            self.node_map = node_map
            self.edge_map = edge_map
            self.entity_map = entity_map
            self.assertion_map = assertion_map
            self.community_map = community_map
            self.outgoing_edge_ids = outgoing_edge_ids
            self._local_graph_loaded = True

    def _build_relation_query_plan(self, query: str, *, mode: QueryMode, media_query: bool) -> RelationQueryPlan | None:
        if not self.graph_relation_router_enabled or media_query:
            return None
        if mode not in {QueryMode.FACT, QueryMode.SCOPED}:
            return None
        normalized = _clean_text(query).lower()
        query_tokens = set(_tokenize(query))
        lookup_profile = _lookup_query_profile(query)
        entity_tokens = tuple(dict.fromkeys([*_named_query_tokens(query), *lookup_profile.focus_tokens]))
        if len(normalized.split()) > self.graph_relation_max_query_tokens and mode != QueryMode.SCOPED:
            return None
        best_family = ""
        best_score = 0.0
        best_rule: Dict[str, Any] | None = None
        for family, rule in _RELATION_QUERY_RULES.items():
            phrase_hits = sum(1 for phrase in rule["phrases"] if phrase in normalized)
            token_hits = len(set(rule["tokens"]) & query_tokens)
            alias_hits = len(set(rule["aliases"]) & query_tokens)
            score = (phrase_hits * 0.42) + min(token_hits * 0.10, 0.30) + min(alias_hits * 0.05, 0.15)
            if entity_tokens:
                score += 0.08
            if mode == QueryMode.FACT and _query_starts_with(query, ("who ", "what ", "where ", "which ", "whose ", "is ", "are ", "does ", "do ", "can ", "in which ")):
                score += 0.05
            if family == "contact":
                if lookup_profile.is_contact_lookup:
                    score += 0.42
                elif lookup_profile.is_exact_lookup:
                    score -= 0.18
            if family == "affiliation" and mode == QueryMode.SCOPED:
                score += 0.08
            if family == "location" and (query_tokens & _OFFERING_ROUTER_TOKENS):
                score -= 0.32
                if not (query_tokens & _EXPLICIT_LOCATION_ROUTER_TOKENS):
                    score -= 0.28
                if query_tokens & _STRONG_OFFERING_ROUTER_TOKENS:
                    score -= 0.32
            if family in {"transport", "accommodation", "hours", "amenities"}:
                scoped_bonus = 0.10 if mode == QueryMode.SCOPED else 0.0
                score += 0.12 + scoped_bonus
            if score > best_score:
                best_family = family
                best_score = score
                best_rule = rule
        if not best_rule or best_score < self.graph_relation_route_min_confidence:
            return None
        graph_first = (
            best_family in _GRAPH_FIRST_RELATION_FAMILIES
            and best_score >= self.graph_relation_graph_first_min_confidence
        )
        alias_tokens = tuple(best_rule["aliases"])
        if best_family == "hours":
            alias_tokens = tuple(_hours_query_alias_tokens(query))
        return RelationQueryPlan(
            family=best_family,
            confidence=min(best_score, 1.0),
            primary_relation_types=tuple(best_rule["primary_relation_types"]),
            secondary_relation_types=tuple(best_rule["secondary_relation_types"]),
            alias_tokens=alias_tokens,
            entity_tokens=entity_tokens,
            graph_first=graph_first,
        )

    def _expanded_relation_query(self, query: str, plan: RelationQueryPlan) -> str:
        rewritten, _ = self._append_query_aliases(query, plan.alias_tokens)
        return rewritten

    def prepare_query_context(
        self,
        query: str,
        *,
        mode: QueryMode | None = None,
        media_query: bool | None = None,
        relation_plan: RelationQueryPlan | None = None,
    ) -> GraphQueryContext:
        mode = mode or classify_query_mode(query)
        if media_query is None:
            media_query = _is_media_query(query)
        if relation_plan is None:
            relation_plan = self._build_relation_query_plan(query, mode=mode, media_query=media_query)

        rewritten_query = query
        rewrite_labels: List[str] = []
        semantic_aliases = _semantic_query_alias_tokens(query)
        if semantic_aliases:
            candidate, added = self._append_query_aliases(rewritten_query, semantic_aliases, max_new_tokens=6)
            if candidate != rewritten_query and added:
                rewritten_query = candidate
                rewrite_labels.append("semantic_alias_expansion")

        if relation_plan is not None:
            candidate = self._expanded_relation_query(rewritten_query, relation_plan)
            if candidate != rewritten_query:
                rewritten_query = candidate
                rewrite_labels.append("relation_alias_expansion")
            relation_candidates = self._build_relation_candidate_set(rewritten_query, relation_plan)
        else:
            relation_candidates = RelationCandidateSet()

        return GraphQueryContext(
            mode=mode,
            media_query=bool(media_query),
            relation_plan=relation_plan,
            relation_candidates=relation_candidates,
            rewritten_query=rewritten_query,
            rewrite_labels=tuple(dict.fromkeys(rewrite_labels)),
        )

    def _score_relation_assertion_candidate(self, query: str, plan: RelationQueryPlan, node: Dict[str, Any]) -> float:
        if not self._is_active_assertion_node(node):
            return -1.0
        props = dict(node.get("properties") or {})
        relation_type = str(props.get("relation_type") or node.get("label") or "").upper()
        if relation_type in plan.primary_relation_types:
            relation_bonus = 1.05
        elif relation_type in plan.secondary_relation_types:
            relation_bonus = 0.38
        elif relation_type in _GENERIC_RELATION_TYPES:
            relation_bonus = 0.12
        else:
            return -1.0
        expanded_query = self._expanded_relation_query(query, plan)
        subject_name = _clean_text(props.get("subject_name") or "")
        object_name = _clean_text(props.get("object_name") or "")
        evidence_text = _clean_text(props.get("text") or props.get("evidence") or "")
        score = relation_bonus + self.base._score_text_match(expanded_query, " ".join([subject_name, object_name, evidence_text]))
        entity_tokens = set(plan.entity_tokens)
        subject_tokens = set(_tokenize(subject_name))
        object_tokens = set(_tokenize(object_name))
        if entity_tokens:
            if entity_tokens & subject_tokens:
                score += 0.20
            if entity_tokens & object_tokens:
                score += 0.20
        confidence = float(props.get("confidence") or 0.0)
        score += max(0.0, min(confidence, 1.0)) * 0.40
        if relation_type in _GENERIC_RELATION_TYPES and not (entity_tokens & (subject_tokens | object_tokens)):
            score -= 0.18
        return score

    def _local_relation_assertion_candidates(self, query: str, plan: RelationQueryPlan) -> List[Tuple[str, float]]:
        if not self.local_graph_available:
            return []
        self._ensure_local_graph_loaded()
        if not self.assertion_map:
            return []
        scored: List[Tuple[str, float]] = []
        for assertion_id, node in self.assertion_map.items():
            score = self._score_relation_assertion_candidate(query, plan, node)
            if score <= 0.0:
                continue
            scored.append((assertion_id, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: self.graph_relation_local_candidate_limit]

    def _local_relation_fact_candidates(self, query: str, plan: RelationQueryPlan) -> List[Tuple[str, float]]:
        expanded_query = self._expanded_relation_query(query, plan)
        fact_ids = self.base._local_fact_query_ids(expanded_query, top_k=max(self.graph_relation_fact_limit * 3, 8))
        scored: List[Tuple[str, float]] = []
        for fact_id in fact_ids:
            fact = self.base.fact_map.get(str(fact_id))
            if not fact:
                continue
            fact_text = _clean_text(fact.get("text") or fact.get("dense_text") or "")
            if not fact_text:
                continue
            score = self.base._score_text_match(expanded_query, fact_text) + self.base._fact_query_bonus(query, fact_text)
            score += self.base._score_text_match(query, fact_text) * 0.20
            if any(token in fact_text.lower() for token in plan.alias_tokens):
                score += 0.12
            entity_tokens = set(plan.entity_tokens)
            if entity_tokens and not (entity_tokens & set(_tokenize(fact_text))):
                score -= 0.10
            if score <= 0.0:
                continue
            scored.append((str(fact_id), score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(self.graph_relation_fact_limit, self.graph_relation_local_candidate_limit // 2)]

    def _build_relation_candidate_set(self, query: str, plan: RelationQueryPlan) -> RelationCandidateSet:
        assertion_scores = self._local_relation_assertion_candidates(query, plan)
        fact_scores = self._local_relation_fact_candidates(query, plan)
        chunk_scores: Dict[str, float] = {}
        parent_scores: Dict[str, float] = {}
        assertion_score_map = {assertion_id: score for assertion_id, score in assertion_scores}
        fact_score_map = {fact_id: score for fact_id, score in fact_scores}
        for assertion_id, score in assertion_scores[: self.graph_relation_assertion_limit]:
            props = self._assertion_props(assertion_id)
            bonus = self.graph_relation_chunk_seed_bonus * score
            parent_bonus = self.graph_relation_parent_seed_bonus * score
            for chunk_id in props.get("source_chunk_ids") or []:
                chunk_id = str(chunk_id)
                if chunk_id:
                    chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), bonus)
            for parent_id in props.get("source_parent_ids") or []:
                parent_id = str(parent_id)
                if parent_id:
                    parent_scores[parent_id] = max(parent_scores.get(parent_id, 0.0), parent_bonus)
        for fact_id, score in fact_scores[: self.graph_relation_fact_limit]:
            fact = self.base.fact_map.get(str(fact_id)) or {}
            bonus = self.graph_relation_chunk_seed_bonus * 0.9 * score
            parent_bonus = self.graph_relation_parent_seed_bonus * 0.9 * score
            for chunk_id in fact.get("linked_chunk_ids") or []:
                chunk_id = str(chunk_id)
                if chunk_id:
                    chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), bonus)
            for parent_id in fact.get("linked_parent_ids") or []:
                parent_id = str(parent_id)
                if parent_id:
                    parent_scores[parent_id] = max(parent_scores.get(parent_id, 0.0), parent_bonus)
        top_assertion = assertion_scores[0][1] if assertion_scores else 0.0
        top_fact = fact_scores[0][1] if fact_scores else 0.0
        confidence = min(
            1.0,
            (plan.confidence * 0.35)
            + (min(top_assertion / 1.8, 1.0) * 0.40)
            + (min(top_fact / 1.6, 1.0) * 0.25),
        )
        return RelationCandidateSet(
            family=plan.family,
            confidence=confidence,
            graph_first=confidence >= self.graph_relation_graph_first_min_confidence and plan.graph_first,
            assertion_ids=[assertion_id for assertion_id, _score in assertion_scores[: self.graph_relation_assertion_limit]],
            fact_ids=[fact_id for fact_id, _score in fact_scores[: self.graph_relation_fact_limit]],
            chunk_ids=list(dict.fromkeys(sorted(chunk_scores, key=lambda item: chunk_scores[item], reverse=True)))[: self.graph_max_additional_chunk_candidates],
            parent_ids=list(dict.fromkeys(sorted(parent_scores, key=lambda item: parent_scores[item], reverse=True)))[: self.graph_max_additional_parent_candidates],
            chunk_scores=chunk_scores,
            parent_scores=parent_scores,
            assertion_scores=assertion_score_map,
            fact_scores=fact_score_map,
        )

    def _targets_for_edge_types_local(self, source_ids: Sequence[str], edge_types: Sequence[str]) -> Tuple[List[str], List[str]]:
        self._ensure_local_graph_loaded()
        wanted = {str(edge_type) for edge_type in edge_types if str(edge_type)}
        targets: List[str] = []
        used_edges: List[str] = []
        seen_targets = set()
        seen_edges = set()
        for source_id in source_ids:
            source_node = self.node_map.get(str(source_id))
            if (
                isinstance(source_node, dict)
                and str(source_node.get("node_type") or "") == "relation_assertion"
                and not self._is_active_assertion_node(source_node)
            ):
                continue
            for edge_id in self.outgoing_edge_ids.get(str(source_id), []):
                edge = self.edge_map.get(str(edge_id))
                if not edge:
                    continue
                if str(edge.get("edge_type") or "") not in wanted:
                    continue
                target_id = str(edge.get("target_id") or "")
                if not target_id:
                    continue
                target_node = self.node_map.get(target_id)
                if (
                    isinstance(target_node, dict)
                    and str(target_node.get("node_type") or "") == "relation_assertion"
                    and not self._is_active_assertion_node(target_node)
                ):
                    continue
                if edge_id not in seen_edges:
                    used_edges.append(str(edge_id))
                    seen_edges.add(str(edge_id))
                if target_id in seen_targets:
                    continue
                seen_targets.add(target_id)
                targets.append(target_id)
        return targets, used_edges

    def _neo4j_query_rows(self, *, statement: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        endpoint = _neo4j_endpoint(self.neo4j_uri, self.neo4j_database)
        payload = _post_query(
            endpoint=endpoint,
            username=self.neo4j_username,
            password=self.neo4j_password,
            statement=statement,
            parameters=parameters,
            timeout_sec=self.neo4j_timeout_sec,
        )
        data = payload.get("data") or {}
        fields = list(data.get("fields") or [])
        values = list(data.get("values") or [])
        rows: List[Dict[str, Any]] = []
        for value_row in values:
            if not isinstance(value_row, list):
                continue
            row: Dict[str, Any] = {}
            for idx, field in enumerate(fields):
                if idx < len(value_row):
                    row[str(field)] = value_row[idx]
            rows.append(row)
        return rows

    def _targets_for_edge_types_neo4j(
        self,
        source_ids: Sequence[str],
        edge_types: Sequence[str],
        *,
        node_types: Sequence[str] | None = None,
    ) -> Tuple[List[str], List[str]]:
        source_ids = [str(value) for value in source_ids if str(value)]
        edge_types = [str(value) for value in edge_types if str(value)]
        node_types = [str(value) for value in (node_types or []) if str(value)]
        if not source_ids or not edge_types:
            return [], []
        rows = self._neo4j_query_rows(
            statement=(
                "MATCH (source:KGNode {namespace: $namespace})-[r:KG_EDGE]->(target:KGNode {namespace: $namespace}) "
                "WHERE source.id IN $source_ids AND r.edge_type IN $edge_types "
                "AND (size($node_types) = 0 OR target.node_type IN $node_types) "
                "RETURN target.id AS target_id, "
                "       target.node_type AS node_type, "
                "       target.label AS label, "
                "       target.text AS text, "
                "       target.evidence AS evidence, "
                "       target.source_url AS source_url, "
                "       target.document_title AS document_title, "
                "       target.title AS title, "
                "       target.caption AS caption, "
                "       target.description AS description, "
                "       target.context AS context, "
                "       target.transcript AS transcript, "
                "       target.media_type AS media_type, "
                "       target.url AS url, "
                "       target.asset_uri AS asset_uri, "
                "       target.local_path AS local_path, "
                "       target.source_chunk_ids AS source_chunk_ids, "
                "       target.source_fact_ids AS source_fact_ids, "
                "       target.source_parent_ids AS source_parent_ids, "
                "       target.validity_status AS validity_status, "
                "       collect(DISTINCT r.id) AS edge_ids "
                "LIMIT $limit"
            ),
            parameters={
                "namespace": self.neo4j_namespace,
                "source_ids": source_ids,
                "edge_types": edge_types,
                "node_types": node_types,
                "limit": max(1, self.neo4j_query_limit),
            },
        )
        targets: List[str] = []
        used_edges: List[str] = []
        seen_targets = set()
        seen_edges = set()
        for row in rows:
            target_id = str(row.get("target_id") or "")
            if not target_id:
                continue
            node_type = str(row.get("node_type") or "")
            validity_status = _clean_text(row.get("validity_status") or "active").lower()
            if node_type == "relation_assertion" and validity_status not in _ACTIVE_ASSERTION_STATUSES:
                continue
            if target_id not in seen_targets:
                seen_targets.add(target_id)
                targets.append(target_id)
            for edge_id in row.get("edge_ids") or []:
                edge_id = str(edge_id or "")
                if not edge_id or edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)
                used_edges.append(edge_id)
            self._neo4j_node_cache()[target_id] = {
                "id": target_id,
                "node_type": node_type,
                "label": str(row.get("label") or ""),
                "properties": {
                    "text": row.get("text"),
                    "evidence": row.get("evidence"),
                    "source_url": row.get("source_url"),
                    "document_title": row.get("document_title"),
                    "title": row.get("title"),
                    "caption": row.get("caption"),
                    "description": row.get("description"),
                    "context": row.get("context"),
                    "transcript": row.get("transcript"),
                    "media_type": row.get("media_type"),
                    "url": row.get("url"),
                    "asset_uri": row.get("asset_uri"),
                    "local_path": row.get("local_path"),
                    "source_chunk_ids": row.get("source_chunk_ids") or [],
                    "source_fact_ids": row.get("source_fact_ids") or [],
                    "source_parent_ids": row.get("source_parent_ids") or [],
                    "validity_status": row.get("validity_status"),
                },
            }
        return targets, used_edges

    def _targets_for_edge_types(
        self,
        source_ids: Sequence[str],
        edge_types: Sequence[str],
        *,
        node_types: Sequence[str] | None = None,
        prefer_local: bool = False,
    ) -> Tuple[List[str], List[str], str]:
        self._set_neo4j_last_error(None)
        prefer_neo4j = (not prefer_local) and self.graph_query_backend in {"auto", "neo4j"}
        if prefer_neo4j and self.neo4j_enabled:
            try:
                targets, edge_ids = self._targets_for_edge_types_neo4j(source_ids, edge_types, node_types=node_types)
                if targets or self.graph_query_backend == "neo4j":
                    return targets, edge_ids, "neo4j"
            except Exception as exc:
                self._set_neo4j_last_error(str(exc))
                if self.graph_query_backend == "neo4j" and not self.local_graph_available:
                    raise
        if self.local_graph_available:
            targets, edge_ids = self._targets_for_edge_types_local(source_ids, edge_types)
            return targets, edge_ids, "local"
        return [], [], "none"

    def _score_graph_fact(self, query: str, fact_id: str, *, chunk_ids: Sequence[str], parent_ids: Sequence[str]) -> float:
        fact = self.base.fact_map.get(str(fact_id))
        if not fact:
            return -1.0
        score = self.base._score_text_match(query, fact.get("dense_text") or fact.get("text") or "")
        linked_chunk_ids = {str(value) for value in (fact.get("linked_chunk_ids") or []) if str(value)}
        linked_parent_ids = {str(value) for value in (fact.get("linked_parent_ids") or []) if str(value)}
        if linked_chunk_ids & {str(value) for value in chunk_ids if str(value)}:
            score += self.graph_chunk_fact_bonus
        if linked_parent_ids & {str(value) for value in parent_ids if str(value)}:
            score += self.graph_parent_fact_bonus
        return score

    def _score_graph_media(self, query: str, media_id: str, *, chunk_ids: Sequence[str], parent_ids: Sequence[str]) -> float:
        media = self.base.media_map.get(str(media_id))
        if not media:
            return -1.0
        score = self.base._score_media_relevance(query, media)
        linked_chunk_ids = {str(value) for value in (media.get("linked_chunk_ids") or []) if str(value)}
        linked_parent_ids = {str(value) for value in (media.get("linked_parent_ids") or []) if str(value)}
        if linked_chunk_ids & {str(value) for value in chunk_ids if str(value)}:
            score += self.graph_chunk_media_bonus
        if linked_parent_ids & {str(value) for value in parent_ids if str(value)}:
            score += self.graph_parent_media_bonus
        return score

    def _build_fact_document(self, fact_id: str) -> Dict[str, Any] | None:
        fact = self.base.fact_map.get(str(fact_id))
        if not fact:
            return None
        return {
            "id": str(fact.get("id") or ""),
            "text": str(fact.get("text") or fact.get("dense_text") or "").strip(),
            "source_url": str(fact.get("source_url") or ""),
            "document_title": str(fact.get("document_title") or ""),
            "document_summary": "",
            "media": [],
        }

    def _build_media_document(self, media_id: str) -> Dict[str, Any] | None:
        media = self.base.media_map.get(str(media_id))
        if not media:
            return None
        text = " ".join(
            str(media.get(key) or "")
            for key in ("title", "caption", "description", "context", "transcript", "text")
        ).strip()
        if not text:
            return None
        return {
            "id": str(media.get("id") or ""),
            "text": text,
            "source_url": str(media.get("source_url") or media.get("url") or ""),
            "document_title": str(media.get("document_title") or ""),
            "document_summary": "",
            "media": [media],
        }

    def _build_assertion_document(self, assertion_id: str) -> Dict[str, Any] | None:
        node = self.assertion_map.get(str(assertion_id))
        if not node:
            node = self._neo4j_node_cache().get(str(assertion_id))
        if not node:
            return None
        props = dict(node.get("properties") or {})
        text = str(props.get("text") or props.get("evidence") or "").strip()
        if not text:
            return None
        return {
            "id": str(node.get("id") or ""),
            "text": text,
            "source_url": str(props.get("source_url") or ""),
            "document_title": str(props.get("document_title") or ""),
            "source_span_ids": [str(value) for value in (props.get("source_span_ids") or []) if str(value)],
            "linked_span_ids": [str(value) for value in (props.get("source_span_ids") or []) if str(value)],
            "linked_chunk_ids": [str(value) for value in (props.get("source_chunk_ids") or []) if str(value)],
            "linked_parent_ids": [str(value) for value in (props.get("source_parent_ids") or []) if str(value)],
            "document_summary": "",
            "media": [],
        }

    def _assertion_props(self, assertion_id: str) -> Dict[str, Any]:
        node = self.assertion_map.get(str(assertion_id))
        if not node:
            node = self._neo4j_node_cache().get(str(assertion_id))
        return dict(node.get("properties") or {}) if isinstance(node, dict) else {}

    def _score_community_summary(self, query: str, community_id: str) -> float:
        node = self.community_map.get(str(community_id))
        if not node:
            node = self._neo4j_node_cache().get(str(community_id))
        if not node:
            return -1.0
        if not self._is_active_assertion_node(node):
            return -1.0
        props = dict(node.get("properties") or {})
        text = str(props.get("summary") or "").strip()
        if not text:
            return -1.0
        return self.base._score_text_match(query, text)

    def _local_community_candidates(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        if not self.local_graph_available:
            return []
        self._ensure_local_graph_loaded()
        scored: List[Tuple[str, float]] = []
        for cid in self.community_map:
            score = self._score_community_summary(query, cid)
            if score > 0.0:
                scored.append((cid, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _neo4j_community_candidates(self, query: str, top_k: int = 2) -> List[Tuple[str, float]]:
        self._set_neo4j_last_error(None)
        try:
            rows = self._neo4j_query_rows(
                statement=(
                    "MATCH (n:KGNode {namespace: $namespace, node_type: 'community'}) "
                    "RETURN n.id AS id, n.summary AS summary, n.title AS title LIMIT 100"
                ),
                parameters={"namespace": self.neo4j_namespace},
            )
            for row in rows:
                cid = str(row.get("id") or "")
                if cid:
                    self._neo4j_node_cache()[cid] = {
                        "id": cid,
                        "node_type": "community",
                        "properties": {
                            "summary": row.get("summary"),
                            "title": row.get("title")
                        }
                    }
            scored: List[Tuple[str, float]] = []
            for row in rows:
                cid = str(row.get("id") or "")
                if cid:
                    score = self._score_community_summary(query, cid)
                    if score > 0.0:
                        scored.append((cid, score))
            scored.sort(key=lambda item: item[1], reverse=True)
            return scored[:top_k]
        except Exception as exc:
            self._set_neo4j_last_error(str(exc))
            return []

    def _get_top_communities(self, query: str, prefer_local: bool = False, top_k: int = 2) -> List[str]:
        prefer_neo4j = (not prefer_local) and self.graph_query_backend in {"auto", "neo4j"}
        if prefer_neo4j and self.neo4j_enabled:
            candidates = self._neo4j_community_candidates(query, top_k=top_k)
            if candidates or self.graph_query_backend == "neo4j":
                return [cid for cid, score in candidates]
        if self.local_graph_available:
            return [cid for cid, score in self._local_community_candidates(query, top_k=top_k)]
        return []

    def _build_community_document(self, community_id: str) -> Dict[str, Any] | None:
        node = self.community_map.get(str(community_id))
        if not node:
            node = self._neo4j_node_cache().get(str(community_id))
        if not node:
            return None
        props = dict(node.get("properties") or {})
        text = str(props.get("summary") or "").strip()
        title = str(props.get("title") or "Community Summary").strip()
        if not text:
            return None
        return {
            "id": str(node.get("id") or ""),
            "text": text,
            "source_url": "",
            "document_title": title,
            "document_summary": "",
            "media": [],
        }

    def _build_chunk_document(self, chunk_id: str, selected_media: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
        chunk = self.base.chunk_map.get(str(chunk_id))
        if not chunk:
            return None
        chunk_media = [
            media
            for media in selected_media
            if (
                chunk_id in (media.get("linked_chunk_ids") or [])
                or chunk.get("page_key") in (media.get("linked_parent_ids") or [])
                or chunk.get("section_key") in (media.get("linked_parent_ids") or [])
            )
        ]
        return {
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

    def _reorder_with_graph_support(
        self,
        *,
        query: str,
        selected_ids: Sequence[str],
        max_ids: int,
        base_bonus: float,
        graph_link_scores: Dict[str, float],
        text_getter: Any,
    ) -> List[str]:
        # Keep the base vector order intact. Graph evidence augments recall and
        # context, but it should not demote already-strong vector hits unless we
        # later add a calibrated graph-aware reranker.
        base_ids = []
        seen = set()
        for value in selected_ids:
            record_id = str(value)
            if not record_id or record_id in seen:
                continue
            seen.add(record_id)
            base_ids.append(record_id)
        if len(base_ids) >= max_ids:
            return base_ids[:max_ids]

        extra_scores: List[Tuple[str, float]] = []
        for record_id, graph_score in graph_link_scores.items():
            record_id = str(record_id)
            if not record_id or record_id in seen:
                continue
            text = str(text_getter(record_id) or "")
            match_score = self.base._score_text_match(query, text) if text else 0.0
            extra_scores.append((record_id, max(0.0, graph_score) + (match_score * 0.35) + base_bonus))
        extra_scores.sort(key=lambda item: item[1], reverse=True)
        for record_id, _score in extra_scores:
            if record_id in seen:
                continue
            seen.add(record_id)
            base_ids.append(record_id)
            if len(base_ids) >= max_ids:
                break
        return base_ids

    def _score_assertion(
        self,
        query: str,
        assertion_id: str,
        *,
        chunk_ids: Sequence[str],
        fact_ids: Sequence[str],
        parent_ids: Sequence[str],
    ) -> float:
        node = self.assertion_map.get(str(assertion_id))
        if not node:
            node = self._neo4j_node_cache().get(str(assertion_id))
        if not node:
            return -1.0
        props = dict(node.get("properties") or {})
        text = str(props.get("text") or props.get("evidence") or "")
        score = self.base._score_text_match(query, text)
        source_chunk_ids = {str(value) for value in (props.get("source_chunk_ids") or []) if str(value)}
        source_fact_ids = {str(value) for value in (props.get("source_fact_ids") or []) if str(value)}
        source_parent_ids = {str(value) for value in (props.get("source_parent_ids") or []) if str(value)}
        if source_chunk_ids & {str(value) for value in chunk_ids if str(value)}:
            score += self.graph_assertion_bonus
        if source_fact_ids & {str(value) for value in fact_ids if str(value)}:
            score += self.graph_assertion_bonus
        if source_parent_ids & {str(value) for value in parent_ids if str(value)}:
            score += self.graph_assertion_bonus * 0.5
        return score

    def _bfs_multi_hop_traversal(
        self,
        start_ids: Sequence[str],
        max_hops: int = 2,
        prefer_local: bool = False,
    ) -> List[str]:
        """
        Performs BFS multi-hop traversal from seed IDs to connect entities across multiple steps.
        """
        queue = [(sid, 0) for sid in start_ids if sid]
        visited = set(sid for sid in start_ids if sid)
        expanded_ids = list(visited)

        # Broad edge types to connect entities to chunks, facts, assertions
        traversal_edge_types = [
            "CHUNK_HAS_FACT", "PAGE_HAS_FACT", "SECTION_HAS_FACT",
            "CHUNK_HAS_EVIDENCE_SPAN", "ASSERTION_SUPPORTED_BY_SPAN",
            "FACT_SUPPORTS_ASSERTION", "CHUNK_SUPPORTS_ASSERTION",
            "ENTITY_IN_ASSERTION", "RELATED_TO", "MENTIONS", "ABOUT",
            "ENTITY_MENTIONED_IN_SPAN", "ENTITY_HAS_COMMUNITY", "PART_OF", "LOCATED_IN", "AFFILIATED_WITH"
        ]

        while queue:
            current_id, depth = queue.pop(0)
            if depth >= max_hops:
                continue

            targets, _, _ = self._targets_for_edge_types(
                [current_id],
                traversal_edge_types,
                prefer_local=prefer_local
            )
            for target in targets:
                if target not in visited:
                    visited.add(target)
                    expanded_ids.append(target)
                    queue.append((target, depth + 1))

        return expanded_ids

    def augment_result(
        self,
        query: str,
        result: Dict[str, Any],
        *,
        relation_plan: RelationQueryPlan | None = None,
        relation_candidates: RelationCandidateSet | None = None,
    ) -> Dict[str, Any]:
        result = dict(result or {})
        result["retriever_backend"] = "graph_hybrid"
        if result.get("abstained"):
            result["graph_used"] = False
            result["graph_fact_ids"] = []
            result["graph_edge_ids"] = []
            return result

        if relation_candidates is None:
            relation_candidates = RelationCandidateSet()

        selected_chunk_ids = [str(value) for value in (result.get("selected_chunk_ids") or []) if str(value)]
        selected_parent_ids = [str(value) for value in (result.get("selected_parent_ids") or []) if str(value)]
        existing_media_ids = [str(value) for value in (result.get("selected_media_ids") or []) if str(value)]
        relation_query = relation_plan is not None
        prefer_local_lookup = bool(relation_query and self.graph_relation_prefer_local and self.local_graph_available)

        if self.graph_relation_only and not relation_query:
            result["graph_used"] = False
            result["graph_store_backend"] = "none"
            result["graph_store_error"] = None
            result["graph_fact_ids"] = []
            result["graph_assertion_ids"] = []
            result["graph_edge_ids"] = []
            result["graph_media_ids"] = []
            result["graph_relation_family"] = ""
            result["graph_relation_confidence"] = 0.0
            result["graph_relation_graph_first"] = False
            return result

        seed_fact_ids = [str(value) for value in relation_candidates.fact_ids if str(value)]
        seed_assertion_ids = [str(value) for value in relation_candidates.assertion_ids if str(value)]
        seed_chunk_scores = dict(relation_candidates.chunk_scores or {})
        seed_parent_scores = dict(relation_candidates.parent_scores or {})

        # Entity-aware Multi-Hop BFS Traversal
        bfs_expanded_ids = self._bfs_multi_hop_traversal(
            start_ids=[*selected_chunk_ids, *selected_parent_ids, *seed_fact_ids, *seed_assertion_ids],
            max_hops=2,
            prefer_local=prefer_local_lookup
        )
        expanded_source_ids = list(dict.fromkeys([*selected_chunk_ids, *selected_parent_ids, *bfs_expanded_ids]))

        candidate_fact_ids, fact_edge_ids, fact_backend = self._targets_for_edge_types(
            expanded_source_ids,
            ["CHUNK_HAS_FACT", "PAGE_HAS_FACT", "SECTION_HAS_FACT"],
            node_types=["fact"],
            prefer_local=prefer_local_lookup,
        )
        candidate_fact_ids = list(dict.fromkeys([*seed_fact_ids, *candidate_fact_ids]))
        scored_fact_ids = sorted(
            (
                (fact_id, self._score_graph_fact(query, fact_id, chunk_ids=selected_chunk_ids, parent_ids=selected_parent_ids))
                for fact_id in candidate_fact_ids
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        graph_fact_ids = [
            fact_id
            for fact_id, score in scored_fact_ids
            if score > 0.0
        ][: self.graph_max_fact_results]
        fact_score_map = {fact_id: score for fact_id, score in scored_fact_ids if score > 0.0}

        candidate_media_ids, media_edge_ids, media_backend = self._targets_for_edge_types(
            expanded_source_ids,
            ["CHUNK_HAS_MEDIA", "PAGE_HAS_MEDIA", "SECTION_HAS_MEDIA"],
            node_types=["media"],
            prefer_local=prefer_local_lookup,
        )
        scored_media_ids = sorted(
            (
                (media_id, self._score_graph_media(query, media_id, chunk_ids=selected_chunk_ids, parent_ids=selected_parent_ids))
                for media_id in candidate_media_ids
                if media_id not in set(existing_media_ids)
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        graph_media_ids = [
            media_id
            for media_id, score in scored_media_ids
            if score > 0.0
        ][: self.graph_max_additional_media_results]
        media_score_map = {media_id: score for media_id, score in scored_media_ids if score > 0.0}

        assertion_source_ids = list(dict.fromkeys([*expanded_source_ids, *graph_fact_ids]))
        candidate_assertion_ids, assertion_edge_ids, assertion_backend = self._targets_for_edge_types(
            assertion_source_ids,
            ["FACT_SUPPORTS_ASSERTION", "CHUNK_SUPPORTS_ASSERTION", "ENTITY_IN_ASSERTION"],
            node_types=["relation_assertion"],
            prefer_local=prefer_local_lookup,
        )
        candidate_assertion_ids = list(dict.fromkeys([*seed_assertion_ids, *candidate_assertion_ids]))
        scored_assertion_ids = sorted(
            (
                (
                    assertion_id,
                    self._score_assertion(
                        query,
                        assertion_id,
                        chunk_ids=selected_chunk_ids,
                        fact_ids=graph_fact_ids,
                        parent_ids=selected_parent_ids,
                    ),
                )
                for assertion_id in candidate_assertion_ids
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        graph_assertion_ids = [
            assertion_id
            for assertion_id, score in scored_assertion_ids
            if score > 0.0
        ][: self.graph_max_assertion_results]
        assertion_score_map = {assertion_id: score for assertion_id, score in scored_assertion_ids if score > 0.0}

        chunk_graph_scores: Dict[str, float] = dict(seed_chunk_scores)
        parent_graph_scores: Dict[str, float] = dict(seed_parent_scores)
        for fact_id in graph_fact_ids:
            fact = self.base.fact_map.get(str(fact_id)) or {}
            fact_bonus = fact_score_map.get(str(fact_id), 0.0) * self.graph_chunk_fact_bonus
            parent_bonus = fact_score_map.get(str(fact_id), 0.0) * self.graph_parent_fact_bonus
            for chunk_id in fact.get("linked_chunk_ids") or []:
                chunk_id = str(chunk_id)
                if not chunk_id:
                    continue
                chunk_graph_scores[chunk_id] = chunk_graph_scores.get(chunk_id, 0.0) + fact_bonus
            for parent_id in fact.get("linked_parent_ids") or []:
                parent_id = str(parent_id)
                if not parent_id:
                    continue
                parent_graph_scores[parent_id] = parent_graph_scores.get(parent_id, 0.0) + parent_bonus
        for media_id in graph_media_ids:
            media = self.base.media_map.get(str(media_id)) or {}
            media_bonus = media_score_map.get(str(media_id), 0.0) * self.graph_chunk_media_bonus
            parent_bonus = media_score_map.get(str(media_id), 0.0) * self.graph_parent_media_bonus
            for chunk_id in media.get("linked_chunk_ids") or []:
                chunk_id = str(chunk_id)
                if not chunk_id:
                    continue
                chunk_graph_scores[chunk_id] = chunk_graph_scores.get(chunk_id, 0.0) + media_bonus
            for parent_id in media.get("linked_parent_ids") or []:
                parent_id = str(parent_id)
                if not parent_id:
                    continue
                parent_graph_scores[parent_id] = parent_graph_scores.get(parent_id, 0.0) + parent_bonus
        for assertion_id in graph_assertion_ids:
            props = self._assertion_props(assertion_id)
            assertion_score = assertion_score_map.get(str(assertion_id), 0.0)
            assertion_bonus = (self.graph_assertion_bonus * 2.0) + (assertion_score * self.graph_assertion_bonus)
            for span_id in props.get("source_span_ids") or []:
                span = getattr(self.base, "evidence_span_map", {}).get(str(span_id)) or {}
                for chunk_id in [*list(span.get("linked_chunk_ids") or []), span.get("chunk_id")]:
                    chunk_id = str(chunk_id or "")
                    if not chunk_id:
                        continue
                    chunk_graph_scores[chunk_id] = chunk_graph_scores.get(chunk_id, 0.0) + assertion_bonus
            for chunk_id in props.get("source_chunk_ids") or []:
                chunk_id = str(chunk_id)
                if not chunk_id:
                    continue
                chunk_graph_scores[chunk_id] = chunk_graph_scores.get(chunk_id, 0.0) + assertion_bonus
            for parent_id in props.get("source_parent_ids") or []:
                parent_id = str(parent_id)
                if not parent_id:
                    continue
                parent_graph_scores[parent_id] = parent_graph_scores.get(parent_id, 0.0) + (assertion_bonus * 0.75)

        allow_graph_chunk_reorder = bool(
            relation_query
            and relation_candidates.family in _GRAPH_FIRST_RELATION_FAMILIES
            and relation_candidates.confidence >= self.graph_relation_reorder_min_confidence
        )
        if allow_graph_chunk_reorder:
            max_selected_chunks = max(
                len(selected_chunk_ids),
                min(len(selected_chunk_ids) + self.graph_max_additional_chunk_candidates, len(self.base.chunk_map)),
            )
            selected_chunk_ids = self._reorder_with_graph_support(
                query=query,
                selected_ids=[*selected_chunk_ids, *chunk_graph_scores.keys()],
                max_ids=max_selected_chunks,
                base_bonus=0.0,
                graph_link_scores=chunk_graph_scores,
                text_getter=lambda chunk_id: (
                    (self.base.chunk_map.get(str(chunk_id)) or {}).get("dense_text")
                    or (self.base.chunk_map.get(str(chunk_id)) or {}).get("text")
                    or ""
                ),
            )
            max_selected_parents = max(
                len(selected_parent_ids),
                min(len(selected_parent_ids) + self.graph_max_additional_parent_candidates, len(self.base.parent_map)),
            )
            selected_parent_ids = self._reorder_with_graph_support(
                query=query,
                selected_ids=[*selected_parent_ids, *parent_graph_scores.keys()],
                max_ids=max_selected_parents,
                base_bonus=0.0,
                graph_link_scores=parent_graph_scores,
                text_getter=lambda parent_id: (
                    (self.base.parent_map.get(str(parent_id)) or {}).get("text")
                    or (self.base.parent_map.get(str(parent_id)) or {}).get("dense_text")
                    or ""
                ),
            )

        if allow_graph_chunk_reorder:
            promoted_chunk_ids = [
                chunk_id
                for chunk_id, _score in sorted(chunk_graph_scores.items(), key=lambda item: item[1], reverse=True)
                if chunk_id in set(selected_chunk_ids)
            ]
            if promoted_chunk_ids:
                seen_chunk_ids = set()
                selected_chunk_ids = [
                    chunk_id
                    for chunk_id in [*promoted_chunk_ids, *selected_chunk_ids]
                    if chunk_id and not (chunk_id in seen_chunk_ids or seen_chunk_ids.add(chunk_id))
                ]
            promoted_parent_ids = [
                parent_id
                for parent_id, _score in sorted(parent_graph_scores.items(), key=lambda item: item[1], reverse=True)
                if parent_id in set(selected_parent_ids)
            ]
            if promoted_parent_ids:
                seen_parent_ids = set()
                selected_parent_ids = [
                    parent_id
                    for parent_id in [*promoted_parent_ids, *selected_parent_ids]
                    if parent_id and not (parent_id in seen_parent_ids or seen_parent_ids.add(parent_id))
                ]

        existing_docs = list(result.get("retrieval_documents") or [])
        existing_doc_ids = {str(doc.get("id") or "") for doc in existing_docs if isinstance(doc, dict)}
        augmented_docs = list(existing_docs)
        for fact_id in graph_fact_ids:
            if fact_id in existing_doc_ids:
                continue
            doc = self._build_fact_document(fact_id)
            if doc and doc["text"]:
                augmented_docs.append(doc)
                existing_doc_ids.add(fact_id)
        for media_id in graph_media_ids:
            if media_id in existing_doc_ids:
                continue
            doc = self._build_media_document(media_id)
            if doc and doc["text"]:
                augmented_docs.append(doc)
                existing_doc_ids.add(media_id)
        for assertion_id in graph_assertion_ids:
            if assertion_id in existing_doc_ids:
                continue
            doc = self._build_assertion_document(assertion_id)
            if doc and doc["text"]:
                augmented_docs.append(doc)
                existing_doc_ids.add(assertion_id)

        graph_community_ids = self._get_top_communities(query, prefer_local=prefer_local_lookup, top_k=2)
        for cid in graph_community_ids:
            if cid in existing_doc_ids:
                continue
            doc = self._build_community_document(cid)
            if doc and doc["text"]:
                augmented_docs.append(doc)
                existing_doc_ids.add(cid)

        if self.graph_max_context_documents > 0:
            augmented_docs = augmented_docs[: max(len(existing_docs), self.graph_max_context_documents + len(existing_docs))]

        merged_media: List[Dict[str, Any]] = []
        seen_media = set()
        for media in list(result.get("media") or []) + [self.base.media_map[mid] for mid in graph_media_ids if mid in self.base.media_map]:
            media_id = str(media.get("id") or "")
            if not media_id or media_id in seen_media:
                continue
            seen_media.add(media_id)
            merged_media.append(media)

        selected_docs = []
        for chunk_id in selected_chunk_ids:
            doc = self._build_chunk_document(chunk_id, merged_media)
            if doc:
                selected_docs.append(doc)
        leading_answer_docs = [
            doc
            for doc in (result.get("answer_documents") or [])
            if isinstance(doc, dict) and str(doc.get("id") or "")
        ]
        leading_answer_doc_ids = {
            str(doc.get("id") or "")
            for doc in leading_answer_docs
            if str(doc.get("id") or "")
        }
        leading_fact_docs = [
            doc
            for doc in (result.get("fact_documents") or [])
            if isinstance(doc, dict) and str(doc.get("id") or "")
        ]
        leading_fact_doc_ids = {
            str(doc.get("id") or "")
            for doc in leading_fact_docs
            if str(doc.get("id") or "")
        }
        selected_doc_ids = {str(doc.get("id") or "") for doc in selected_docs if str(doc.get("id") or "")}
        trailing_docs = [
            doc
            for doc in augmented_docs
            if str(doc.get("id") or "") not in selected_doc_ids
            and str(doc.get("id") or "") not in leading_fact_doc_ids
            and str(doc.get("id") or "") not in leading_answer_doc_ids
        ]

        graph_used = bool(graph_fact_ids or graph_media_ids or graph_assertion_ids or graph_community_ids)
        result["graph_used"] = graph_used
        result["graph_store_backend"] = (
            "neo4j"
            if "neo4j" in {fact_backend, media_backend, assertion_backend}
            else "local"
            if self.local_graph_available
            else "none"
        )
        result["graph_store_error"] = self._neo4j_last_error()
        result["graph_fact_ids"] = graph_fact_ids
        result["graph_assertion_ids"] = graph_assertion_ids
        result["graph_community_ids"] = graph_community_ids
        result["graph_edge_ids"] = list(dict.fromkeys([*fact_edge_ids, *media_edge_ids, *assertion_edge_ids]))
        result["graph_media_ids"] = graph_media_ids
        result["graph_relation_family"] = relation_candidates.family if relation_query else ""
        result["graph_relation_confidence"] = float(relation_candidates.confidence if relation_query else 0.0)
        result["graph_relation_graph_first"] = bool(relation_candidates.graph_first if relation_query else False)
        result["selected_chunk_ids"] = selected_chunk_ids
        result["selected_parent_ids"] = selected_parent_ids
        result["selected_media_ids"] = list(dict.fromkeys([*existing_media_ids, *graph_media_ids]))
        result["retrieval_documents"] = build_retrieval_documents(
            [*leading_answer_docs, *leading_fact_docs, *selected_docs, *trailing_docs],
            max_media_per_doc=self.base.max_media_results,
            max_total_media=self.base.max_media_results,
        )
        result["media"] = merged_media[: self.base.max_media_results]
        return result

    def retrieve(self, query: str, *, query_vector: List[float] | None = None) -> Dict[str, Any]:
        query_vector = list(query_vector) if query_vector is not None else self.base.embed_query(query)
        graph_context = self.prepare_query_context(query)
        relation_plan = graph_context.relation_plan
        relation_candidates = graph_context.relation_candidates
        retrieval_query = graph_context.rewritten_query or query
        seed_overrides: Dict[str, Sequence[str]] | None = None
        if relation_plan is not None and relation_candidates.graph_first:
            seed_overrides = {
                "graph_relation_chunk_ids": relation_candidates.chunk_ids,
                "graph_relation_parent_ids": relation_candidates.parent_ids,
                "graph_relation_fact_ids": relation_candidates.fact_ids,
            }
        try:
            result = self.base.retrieve(retrieval_query, query_vector=query_vector, seed_overrides=seed_overrides)
        except TypeError:
            result = self.base.retrieve(retrieval_query, query_vector=query_vector)
        result["query"] = query
        result["query_rewritten"] = retrieval_query
        result["query_rewrite_labels"] = list(graph_context.rewrite_labels)
        return self.augment_result(
            query,
            result,
            relation_plan=relation_plan,
            relation_candidates=relation_candidates,
        )
