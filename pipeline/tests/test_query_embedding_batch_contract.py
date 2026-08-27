from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def _fake_genai_types():
    class FakePart:
        @staticmethod
        def from_text(*, text):
            return ("text", text)

    class FakeContent:
        def __init__(self, *, role, parts):
            self.role = role
            self.parts = parts

    class FakeConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    return SimpleNamespace(
        EmbedContentConfig=FakeConfig,
        Content=FakeContent,
        Part=FakePart,
    )


def test_embed_queries_sends_one_content_per_gemini_query(monkeypatch):
    from pipeline.retrieval import adaptive_hybrid

    captured = {}

    class FakeModels:
        def embed_content(self, *, model, contents, config):
            captured.update(model=model, contents=contents, config=config)
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[float(index), 1.0])
                    for index, _content in enumerate(contents)
                ]
            )

    monkeypatch.setattr(adaptive_hybrid, "import_genai_types", _fake_genai_types)
    monkeypatch.setattr(
        adaptive_hybrid,
        "_make_gemini_client",
        lambda: SimpleNamespace(models=FakeModels()),
    )

    vectors = adaptive_hybrid._embed_queries(
        ["first query", "second query"],
        model="gemini-embedding-2",
        output_dimensionality=1536,
    )

    assert vectors == [[0.0, 1.0], [1.0, 1.0]]
    assert captured["model"] == "gemini-embedding-2"
    assert captured["config"].kwargs == {"output_dimensionality": 1536}
    assert [content.role for content in captured["contents"]] == ["user", "user"]
    assert [content.parts[0][1] for content in captured["contents"]] == [
        "task: search result | query: first query",
        "task: search result | query: second query",
    ]


def test_embed_queries_rejects_provider_cardinality_mismatch(monkeypatch):
    from pipeline.retrieval import adaptive_hybrid

    class FakeModels:
        def embed_content(self, **_kwargs):
            return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])

    monkeypatch.setattr(adaptive_hybrid, "import_genai_types", _fake_genai_types)
    monkeypatch.setattr(
        adaptive_hybrid,
        "_make_gemini_client",
        lambda: SimpleNamespace(models=FakeModels()),
    )

    with pytest.raises(RuntimeError, match=r"requested=2 returned=1"):
        adaptive_hybrid._embed_queries(
            ["first query", "second query"],
            model="gemini-embedding-2",
            output_dimensionality=1536,
        )


def test_embed_queries_splits_requests_at_gemini_batch_limit(monkeypatch):
    from pipeline.retrieval import adaptive_hybrid

    batch_sizes = []

    class FakeModels:
        def embed_content(self, *, model, contents, config):
            batch_sizes.append(len(contents))
            offset = sum(batch_sizes[:-1])
            return SimpleNamespace(
                embeddings=[
                    SimpleNamespace(values=[float(offset + index), 1.0])
                    for index, _content in enumerate(contents)
                ]
            )

    monkeypatch.setattr(adaptive_hybrid, "import_genai_types", _fake_genai_types)
    monkeypatch.setattr(
        adaptive_hybrid,
        "_make_gemini_client",
        lambda: SimpleNamespace(models=FakeModels()),
    )

    queries = [f"query {index}" for index in range(101)]
    vectors = adaptive_hybrid._embed_queries(
        queries,
        model="gemini-embedding-2",
        output_dimensionality=1536,
    )

    assert batch_sizes == [100, 1]
    assert len(vectors) == len(queries)
    assert vectors[0] == [0.0, 1.0]
    assert vectors[-1] == [100.0, 1.0]


def test_gemini_document_embedder_fails_closed_before_silent_truncation(monkeypatch):
    from pipeline.stages.embedders import gemini_pinecone_embedder as embedder

    monkeypatch.setattr(embedder, "estimate_token_count", lambda _text: 6001)

    with pytest.raises(ValueError, match="fail-closed local safety budget"):
        embedder._enforce_gemini_embedding_budget(
            ["oversized input"],
            model="gemini-embedding-2",
        )


def test_gemini_document_embedder_budget_includes_image_allowance(monkeypatch):
    from pipeline.stages.embedders import gemini_pinecone_embedder as embedder

    monkeypatch.setattr(embedder, "estimate_token_count", lambda _text: 5800)

    with pytest.raises(ValueError, match=r"6058 > 6000"):
        embedder._enforce_gemini_embedding_budget(
            ["multimodal input"],
            model="gemini-embedding-2",
            image_count_by_input=[1],
        )


def test_retrieval_evaluation_rejects_partial_batch_before_cache_write(tmp_path, monkeypatch):
    from pipeline.evaluation.retrieval_eval import evaluate_retrieval_dataset

    dataset_path = tmp_path / "gold.jsonl"
    dataset_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"q{index}",
                    "query": query,
                    "query_type": "fact",
                    "source_type": "webpage",
                    "gold_chunk_ids": [f"chunk-{index}"],
                    "gold_parent_ids": [f"parent-{index}"],
                }
            )
            for index, query in enumerate(("first query", "second query"), start=1)
        )
        + "\n",
        encoding="utf-8",
    )

    class FakeRetriever:
        model = "gemini-embedding-2"
        output_dimensionality = 1536

        def embed_queries(self, _queries):
            return [[0.1, 0.2]]

        def embed_query(self, _query):
            raise AssertionError("partial batches must fail before single-query fallback")

    monkeypatch.setattr(
        "pipeline.evaluation.retrieval_eval.AdaptiveHybridRetriever.from_config",
        lambda **_kwargs: FakeRetriever(),
    )
    cache_path = tmp_path / "query_cache.json"

    with pytest.raises(RuntimeError, match=r"requested=2 returned=1"):
        evaluate_retrieval_dataset(
            config_name="unused",
            work_dir=tmp_path,
            dataset_path=dataset_path,
            query_cache_path=cache_path,
        )

    assert not cache_path.exists()
