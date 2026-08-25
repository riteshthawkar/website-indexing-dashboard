from __future__ import annotations


def test_routed_rewrite_bundle_skips_llm_planner_for_upstream_rewrites(monkeypatch):
    import pipeline.retrieval.routed_hybrid as module
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_enabled = True
    retriever.query_planner_model = "gpt-5-nano"
    retriever.query_planner_min_confidence = 0.5
    retriever.parallel_query_rewriting_enabled = False
    retriever.hyde_enabled = False

    def fail_if_called(**_kwargs):
        raise AssertionError("retrieval planner must not run after an upstream query rewrite")

    monkeypatch.setattr(module, "plan_query", fail_if_called)

    bundle = retriever._build_query_rewrite_bundle(
        "MBZUAI graduate admissions requirements",
        relation_plan=None,
        query_mode="fact",
        use_query_planner=False,
    )

    assert bundle.vector_query == "MBZUAI graduate admissions requirements"
    assert bundle.graph_query == "MBZUAI graduate admissions requirements"
    assert "openai_vector_plan" not in bundle.labels
    assert "openai_graph_plan" not in bundle.labels


def test_routed_rewrite_bundle_preserves_deterministic_navigation_intent():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_enabled = False
    retriever.query_planner_model = "gpt-5-nano"
    retriever.query_planner_min_confidence = 0.5
    retriever.parallel_query_rewriting_enabled = False
    retriever.hyde_enabled = False

    bundle = retriever._build_query_rewrite_bundle(
        "How do I apply to MBZUAI?",
        relation_plan=None,
        query_mode="fact",
        use_query_planner=False,
    )

    assert bundle.navigation_intent == "apply"
    assert bundle.navigation_goal == "How do I apply to MBZUAI?"
    assert bundle.navigation_confidence >= 0.9
