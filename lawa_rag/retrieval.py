"""Lazy public imports for retrieval and vector-store capabilities."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "AdaptiveHybridRetriever": ("pipeline.retrieval.adaptive_hybrid", "AdaptiveHybridRetriever"),
    "GraphRAGRetriever": ("pipeline.retrieval.graph_rag", "GraphRAGRetriever"),
    "GroundedNavigationPlanner": ("pipeline.retrieval.navigation_planner", "GroundedNavigationPlanner"),
    "PgVectorConfigurationError": ("pipeline.vectorstores.pgvector_store", "PgVectorConfigurationError"),
    "PgVectorReleaseError": ("pipeline.vectorstores.pgvector_store", "PgVectorReleaseError"),
    "PgVectorSettings": ("pipeline.vectorstores.pgvector_store", "PgVectorSettings"),
    "PgVectorStore": ("pipeline.vectorstores.pgvector_store", "PgVectorStore"),
    "QueryMode": ("pipeline.retrieval.adaptive_hybrid", "QueryMode"),
    "QueryRewriteBundle": ("pipeline.retrieval.routed_hybrid", "QueryRewriteBundle"),
    "RetrievedRecord": ("pipeline.retrieval.adaptive_hybrid", "RetrievedRecord"),
    "RoutedHybridRetriever": ("pipeline.retrieval.routed_hybrid", "RoutedHybridRetriever"),
    "RoutingDecision": ("pipeline.retrieval.routed_hybrid", "RoutingDecision"),
    "VectorMatch": ("pipeline.vectorstores.pgvector_store", "VectorMatch"),
    "apply_vector_upload_manifest_config": ("pipeline.retrieval.adaptive_hybrid", "apply_vector_upload_manifest_config"),
    "build_evidence_pack": ("pipeline.retrieval.evidence_packer", "build_evidence_pack"),
    "classify_query_mode": ("pipeline.retrieval.adaptive_hybrid", "classify_query_mode"),
    "score_retrieval_confidence": ("pipeline.retrieval.evidence_packer", "score_retrieval_confidence"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
