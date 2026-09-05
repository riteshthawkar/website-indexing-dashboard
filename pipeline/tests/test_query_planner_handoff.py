from __future__ import annotations

import threading


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


def test_routed_planner_expands_without_replacing_original_query(monkeypatch):
    import pipeline.retrieval.routed_hybrid as module
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_enabled = True
    retriever.query_planner_model = "gpt-test"
    retriever.query_planner_min_confidence = 0.55
    retriever.parallel_query_rewriting_enabled = True
    retriever.query_specific_retrieval_rules_enabled = False
    retriever.hyde_enabled = False
    monkeypatch.setattr(
        module,
        "plan_query",
        lambda **_kwargs: {
            "query_type": "synthesis",
            "vector_query": "MBZUAI academic organization divisions",
            "graph_query": "MBZUAI academic units organization",
            "answer_types": [],
            "entity_hints": ["MBZUAI"],
            "navigation_intent": "none",
            "navigation_goal": "",
            "navigation_confidence": 0.0,
            "confidence": 0.82,
        },
    )
    query = "How is MBZUAI organized for research and undergraduate education?"

    bundle = retriever._build_query_rewrite_bundle(
        query,
        relation_plan=None,
        query_mode="synthesis",
    )

    assert bundle.vector_query.startswith(query)
    assert bundle.vector_query.endswith("MBZUAI academic organization divisions")
    assert bundle.retrieval_expansion == "MBZUAI academic organization divisions"
    assert "openai_vector_plan" in bundle.labels
    assert "semantic_alias_expansion" not in bundle.labels


def test_routed_planner_and_query_embedding_run_in_parallel():
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.parallel_query_embedding_enabled = True
    embedding_started = threading.Event()
    planner_started = threading.Event()
    expected_bundle = object()

    class FakeVector:
        @staticmethod
        def embed_query(_query):
            embedding_started.set()
            assert planner_started.wait(timeout=1.0)
            return [0.1, 0.2]

    def fake_build(_query, **_kwargs):
        assert embedding_started.wait(timeout=1.0)
        planner_started.set()
        return expected_bundle

    retriever.vector = FakeVector()
    retriever._build_query_rewrite_bundle = fake_build

    bundle, vector, status, error = retriever._prepare_query_rewrites_and_embedding(
        "What are the admissions requirements?",
        relation_plan=None,
        query_mode="synthesis",
        use_query_planner=True,
        query_vector=None,
    )

    assert bundle is expected_bundle
    assert vector == [0.1, 0.2]
    assert status == "ok"
    assert error == ""


def test_routed_planner_passes_production_deadline_and_single_attempt(monkeypatch):
    import pipeline.retrieval.routed_hybrid as module
    from pipeline.retrieval.routed_hybrid import RoutedHybridRetriever

    captured = {}
    retriever = RoutedHybridRetriever.__new__(RoutedHybridRetriever)
    retriever.query_planner_enabled = True
    retriever.query_planner_model = "gpt-test"
    retriever.query_planner_reasoning_effort = "minimal"
    retriever.query_planner_timeout_sec = 6.0
    retriever.query_planner_retries = 1
    retriever.query_planner_min_confidence = 0.55
    retriever.parallel_query_rewriting_enabled = False
    retriever.hyde_enabled = False

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return {
            "query_type": "fact",
            "vector_query": kwargs["query"],
            "graph_query": kwargs["query"],
            "answer_types": [],
            "entity_hints": [],
            "navigation_intent": "none",
            "navigation_goal": "",
            "navigation_confidence": 0.0,
            "confidence": 0.8,
        }

    monkeypatch.setattr(module, "plan_query", fake_plan)
    retriever._build_query_rewrite_bundle("Where is MBZUAI located?")

    assert captured["timeout_sec"] == 6.0
    assert captured["retries"] == 1
    assert captured["reasoning_effort"] == "minimal"


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


def test_query_planner_keeps_grounded_rewrite_when_model_copies_zero_placeholder(monkeypatch):
    import pipeline.core.query_planner as module

    monkeypatch.setattr(module, "make_openai_client", lambda: object())
    monkeypatch.setattr(
        module,
        "json_completion",
        lambda **_kwargs: {
            "query_type": "synthesis",
            "vector_query": "MBZUAI academic organization research undergraduate divisions",
            "graph_query": "MBZUAI academic divisions research undergraduate organization",
            "answer_types": ["organization_structure", "affiliation"],
            "entity_hints": ["MBZUAI", "research", "undergraduate education"],
            "navigation_intent": "none",
            "navigation_goal": "",
            "navigation_confidence": 0.0,
            "confidence": 0.0,
        },
    )

    result = module.plan_query(
        query="How is MBZUAI organized across research and undergraduate education?",
        model="gpt-test",
    )

    assert result["vector_query"].startswith("MBZUAI academic organization")
    assert result["confidence"] == 0.65
    assert result["answer_types"] == ["affiliation"]


def test_query_planner_rejects_rewrite_that_drops_user_constraints(monkeypatch):
    import pipeline.core.query_planner as module

    monkeypatch.setattr(module, "make_openai_client", lambda: object())
    monkeypatch.setattr(
        module,
        "json_completion",
        lambda **_kwargs: {
            "query_type": "fact",
            "vector_query": "unrelated university weather forecast",
            "graph_query": "unrelated weather",
            "answer_types": [],
            "entity_hints": [],
            "navigation_intent": "none",
            "navigation_goal": "",
            "navigation_confidence": 0.0,
            "confidence": 0.99,
        },
    )
    query = "What documents are required for the MBZUAI MSc application in 2027?"

    result = module.plan_query(query=query, model="gpt-test")

    assert result["vector_query"] == query
    assert result["graph_query"] == query
    assert result["confidence"] == 0.0


def test_query_planner_rejects_speculative_facts_in_anchored_rewrite(monkeypatch):
    import pipeline.core.query_planner as module

    monkeypatch.setattr(module, "make_openai_client", lambda: object())
    monkeypatch.setattr(
        module,
        "json_completion",
        lambda **_kwargs: {
            "query_type": "synthesis",
            "vector_query": (
                "MBZUAI current deans three divisions: Computer Science, "
                "Robotics, Science? (Need up-to-date deans from the leadership page.)"
            ),
            "graph_query": (
                "MBZUAI divisions and their current deans, perhaps Computer "
                "Science and Robotics"
            ),
            "answer_types": ["role_holder"],
            "entity_hints": ["MBZUAI"],
            "navigation_intent": "none",
            "navigation_goal": "",
            "navigation_confidence": 0.0,
            "confidence": 0.94,
        },
    )
    query = (
        "Who are the current deans of MBZUAI's three divisions, and which "
        "division does each lead?"
    )

    result = module.plan_query(query=query, model="gpt-test")

    assert result["vector_query"] == query
    assert result["graph_query"] == query
    assert result["confidence"] == 0.0
