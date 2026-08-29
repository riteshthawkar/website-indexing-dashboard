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


def test_query_planner_cannot_invent_navigation_for_factual_website_reference(monkeypatch):
    import pipeline.core.query_planner as module

    monkeypatch.setattr(module, "make_openai_client", lambda: object())
    monkeypatch.setattr(
        module,
        "json_completion",
        lambda **_kwargs: {
            "query_type": "synthesis",
            "vector_query": "IFM partners headquarters research centers",
            "graph_query": "IFM locations partnerships",
            "navigation_intent": "open_page",
            "navigation_goal": "Open IFM website",
            "navigation_confidence": 0.99,
            "confidence": 0.9,
        },
    )

    result = module.plan_query(
        query="What does the IFM website say about its partners and research centers?",
        model="gpt-test",
    )

    assert result["navigation_intent"] == "none"
    assert result["navigation_goal"] == ""
    assert result["navigation_confidence"] == 0.0
