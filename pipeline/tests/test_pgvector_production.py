from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from psycopg import sql

from pipeline.core.config import load_config
from pipeline.retrieval.adaptive_hybrid import (
    AdaptiveHybridRetriever,
    apply_vector_upload_manifest_config,
)
from pipeline.service.retrieval_api import _run_startup_probe
from pipeline.vectorstores.pgvector_store import (
    PGVECTOR_SCHEMA_VERSION,
    PgVectorConfigurationError,
    PgVectorSettings,
    PgVectorStore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LANES = (
    "chunks",
    "parents",
    "media",
    "facts",
    "evidence_spans",
    "summaries",
    "assertions",
    "entities",
    "communities",
)


def _base_config() -> dict:
    return {
        "pipeline": {"production_profile": True},
        "embedder": {"output_dimensionality": 1536},
        "vector_store": {
            "provider": "pgvector",
            "dimensions": 1536,
            "dsn_env": "PGVECTOR_DSN",
            "ingest_dsn_env": "PGVECTOR_INGEST_DSN",
            "require_ssl": True,
        },
    }


def test_pgvector_settings_require_tls_and_dedicated_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PGVECTOR_DSN",
        "postgresql://reader:strong-password@pgvector.internal/vectors?sslmode=verify-full",
    )
    monkeypatch.delenv("PGVECTOR_INGEST_DSN", raising=False)

    reader = PgVectorSettings.from_config(_base_config(), purpose="read")
    assert reader.purpose == "read"
    assert reader.dimensions == 1536

    with pytest.raises(PgVectorConfigurationError, match="PGVECTOR_INGEST_DSN"):
        PgVectorSettings.from_config(_base_config(), purpose="write")

    monkeypatch.setenv(
        "PGVECTOR_INGEST_DSN",
        "postgresql://writer:other-strong-password@pgvector.internal/vectors?sslmode=require",
    )
    writer = PgVectorSettings.from_config(_base_config(), purpose="write")
    assert writer.purpose == "write"


def test_pgvector_settings_reject_plaintext_production_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "PGVECTOR_DSN",
        "postgresql://reader:strong-password@pgvector.internal/vectors?sslmode=disable",
    )
    with pytest.raises(PgVectorConfigurationError, match="sslmode"):
        PgVectorSettings.from_config(_base_config(), purpose="read")


def test_pgvector_schema_v1_rejects_dimension_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    config["vector_store"]["dimensions"] = 768
    monkeypatch.setenv(
        "PGVECTOR_DSN",
        "postgresql://reader:strong-password@pgvector.internal/vectors?sslmode=require",
    )
    with pytest.raises(PgVectorConfigurationError, match="fixed to 1536"):
        PgVectorSettings.from_config(config, purpose="read")


def test_pgvector_settings_enforce_distinct_named_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _base_config()
    config["vector_store"].update(
        {"reader_role": "mbzuai_retriever", "writer_role": "mbzuai_indexer"}
    )
    reader_dsn = (
        "postgresql://mbzuai_retriever:strong-password@pgvector.internal/"
        "vectors?sslmode=require"
    )
    monkeypatch.setenv("PGVECTOR_DSN", reader_dsn)
    monkeypatch.setenv("PGVECTOR_INGEST_DSN", reader_dsn)
    with pytest.raises(PgVectorConfigurationError, match="distinct credentials"):
        PgVectorSettings.from_config(config, purpose="write")

    monkeypatch.setenv(
        "PGVECTOR_INGEST_DSN",
        "postgresql://wrong_writer:other-password@pgvector.internal/vectors?sslmode=require",
    )
    with pytest.raises(PgVectorConfigurationError, match="dedicated write role"):
        PgVectorSettings.from_config(config, purpose="write")


def test_pgvector_vector_validation_is_dimensioned_and_finite() -> None:
    store = object.__new__(PgVectorStore)
    store.settings = PgVectorSettings(
        dsn="postgresql://unused?sslmode=require",
        dimensions=3,
    )
    assert store._normalize_vector([1, 2, 3]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="dimension mismatch"):
        store._normalize_vector([1, 2])
    with pytest.raises(ValueError, match="zero vector"):
        store._normalize_vector([0, 0, 0])
    with pytest.raises(ValueError, match="non-finite"):
        store._normalize_vector([1, float("nan"), 2])


def test_pgvector_upsert_uses_psycopg_cursor_executemany() -> None:
    executed_batches = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return None

        def executemany(self, query, values):
            executed_batches.append((query, list(values)))

    class Result:
        @staticmethod
        def fetchone():
            return {"status": "building"}

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return None

    class Connection:
        @staticmethod
        def transaction():
            return Transaction()

        @staticmethod
        def execute(_query, _parameters):
            return Result()

        @staticmethod
        def cursor():
            return Cursor()

    class Pool:
        @contextmanager
        def connection(self, **_kwargs):
            yield Connection()

    store = object.__new__(PgVectorStore)
    store.settings = PgVectorSettings(
        dsn="postgresql://unused?sslmode=require",
        purpose="write",
        dimensions=3,
    )
    store._pool = Pool()
    store._sql = sql
    store._Jsonb = lambda value: value

    uploaded = store.upsert_records(
        release_id="release-1",
        lane="chunks",
        namespace="chunks--release-1",
        records=[
            {
                "id": "chunk-1",
                "text": "First chunk",
                "embedding": [1, 2, 3],
            },
            {
                "id": "chunk-2",
                "text": "Second chunk",
                "embedding": [3, 2, 1],
            },
        ],
    )

    assert uploaded == 2
    assert len(executed_batches) == 1
    assert len(executed_batches[0][1]) == 2


def test_pgvector_query_casts_list_parameters_to_vector() -> None:
    executed_queries = []

    class Result:
        def __init__(self, rows=None):
            self.rows = list(rows or [])

        def fetchall(self):
            return self.rows

    class Transaction:
        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return None

    class Connection:
        @staticmethod
        def transaction():
            return Transaction()

        @staticmethod
        def execute(query, _parameters=None):
            executed_queries.append(query)
            if isinstance(query, str):
                return Result()
            return Result([{"record_id": "chunk-1", "score": 0.75}])

    class Pool:
        @contextmanager
        def connection(self, **_kwargs):
            yield Connection()

    store = object.__new__(PgVectorStore)
    store.settings = PgVectorSettings(
        dsn="postgresql://unused?sslmode=require",
        dimensions=3,
        require_active_release=False,
    )
    store._pool = Pool()
    store._sql = sql

    matches = store.query(
        release_id="release-1",
        namespace="chunks--release-1",
        vector=[1, 2, 3],
        top_k=5,
    )

    vector_query = next(query for query in executed_queries if not isinstance(query, str))
    rendered = vector_query.as_string(None)
    assert rendered.count("%s::vector") == 2
    assert [(match.record_id, match.score) for match in matches] == [("chunk-1", 0.75)]


def test_adaptive_dense_lane_dispatches_to_pgvector() -> None:
    calls = []

    class Store:
        def query_ids(self, **kwargs):
            calls.append(kwargs)
            return ["chunk-1"]

    retriever = object.__new__(AdaptiveHybridRetriever)
    retriever.vector_store_provider = "pgvector"
    retriever.vector_release_id = "release-1"
    retriever._pgvector_store = Store()

    result = retriever._dense_query_ids(
        query_vector=[0.25, 0.75],
        namespace="chunks--release-1",
        top_k=4,
    )

    assert result == ["chunk-1"]
    assert calls == [
        {
            "release_id": "release-1",
            "namespace": "chunks--release-1",
            "vector": [0.25, 0.75],
            "top_k": 4,
        }
    ]


def test_pgvector_manifest_becomes_runtime_authority(tmp_path: Path) -> None:
    work_dir = tmp_path / "release-1"
    manifest_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    namespaces = {lane: f"{lane}--release-1" for lane in LANES}
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 5,
                "provider": "pgvector",
                "index_name": "mbzuai_retrieval.embedding_records",
                "sparse_index_name": "",
                "namespace_release_id": "release-1",
                "namespaces": namespaces,
                "model": "gemini-embedding-2",
                "output_dimensionality": 1536,
            }
        ),
        encoding="utf-8",
    )
    config = {
        "embedder": {"pinecone_index": "stale", "pinecone_sparse_index": "stale-sparse"},
        "retrieval": {"enable_sparse": True},
        "vector_store": {"provider": "pinecone"},
        "stages": [{"id": "format_retrieval", "type": "formatter", "plugin": "retrieval_bundle_v2"}],
    }

    resolved = apply_vector_upload_manifest_config(config, work_dir)

    assert resolved["vector_store"]["provider"] == "pgvector"
    assert resolved["vector_store"]["release_id"] == "release-1"
    assert "pinecone_index" not in resolved["embedder"]
    assert resolved["retrieval"]["enable_sparse"] is False
    assert resolved["stages"][-1]["plugin"] == "gemini_pgvector"


def test_pgvector_startup_probe_checks_counts_and_live_query(tmp_path: Path) -> None:
    work_dir = tmp_path / "release-1"
    manifest_path = work_dir / "stage_outputs" / "upload_retrieval" / "index_upload_manifest.json"
    manifest_path.parent.mkdir(parents=True)
    namespaces = {lane: f"{lane}--release-1" for lane in LANES}
    uploaded = {lane: 1 for lane in LANES}
    manifest_path.write_text(
        json.dumps(
            {
                "provider": "pgvector",
                "namespace_release_id": "release-1",
                "namespaces": namespaces,
                "uploaded": uploaded,
            }
        ),
        encoding="utf-8",
    )

    class Store:
        def health_check(self, **kwargs):
            assert kwargs["release_id"] == "release-1"
            assert kwargs["expected_namespaces"][namespaces["chunks"]] == 1
            return {"namespace_counts": dict(kwargs["expected_namespaces"])}

        def query_ids(self, **kwargs):
            assert kwargs["namespace"] == namespaces["chunks"]
            return ["chunk-1"]

    vector = SimpleNamespace(
        vector_store_provider="pgvector",
        vector_release_id="release-1",
        output_dimensionality=3,
        _pgvector_store=Store(),
        embed_query=lambda _query: [0.1, 0.2, 0.3],
    )
    retriever = SimpleNamespace(
        vector=vector,
        retrieve=lambda _query, query_vector: {
            "query_embedding_status": "ok",
            "answer_documents": [{"id": "chunk-1"}],
        },
    )

    report = _run_startup_probe(
        retriever,
        work_dir=work_dir,
        query="Where is MBZUAI?",
        operation_timeout_seconds=2,
    )

    assert report["provider"] == "pgvector"
    assert report["dense_namespace_count"] == len(LANES)
    assert report["sparse_namespace_count"] == 0


def test_canonical_production_profile_uses_selected_pgvector_contract() -> None:
    config = load_config("mbzuai_production")
    assert config["vector_store"]["provider"] == "pgvector"
    assert config["vector_store"]["schema_version"] == PGVECTOR_SCHEMA_VERSION
    assert config["vector_store"]["pool_max_size"] == 12
    assert config["chunker"]["target_tokens"] == 650
    assert config["chunker"]["max_tokens"] == 900
    assert config["retrieval"]["enable_sparse"] is False
    assert config["retrieval"]["enable_rerank"] is False
    upload = next(stage for stage in config["stages"] if stage["id"] == "upload_retrieval")
    assert upload["plugin"] == "gemini_pgvector"


def test_pgvector_deployment_is_pinned_private_and_least_privilege() -> None:
    compose = (PROJECT_ROOT / "deploy" / "pgvector" / "docker-compose.yml").read_text(encoding="utf-8")
    schema = (PROJECT_ROOT / "deploy" / "pgvector" / "sql" / "001_schema.sql").read_text(encoding="utf-8")
    hba = (PROJECT_ROOT / "deploy" / "pgvector" / "sql" / "003_configure_hba.sh").read_text(encoding="utf-8")

    assert (
        "pgvector/pgvector:0.8.6-pg17-bookworm@sha256:"
        "cf134a767f474095eeba57e0117be8e568e011a63f33fbf252f14c9b760f8e6f"
    ) in compose
    assert "PGVECTOR_BIND_IP" in compose
    assert "ssl=on" in compose
    assert "latest" not in compose
    assert "CREATE INDEX IF NOT EXISTS embedding_records_embedding_hnsw_idx" in schema
    assert "vector_cosine_ops" in schema
    assert "mbzuai_retrieval_reader" in schema
    assert "mbzuai_retrieval_writer" in schema
    assert "hostnossl all all 0.0.0.0/0 reject" in hba
