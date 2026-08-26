"""Production pgvector storage with immutable, release-scoped records.

The module deliberately keeps database imports lazy. Legacy Pinecone runs can
still inspect or validate their artifacts without installing PostgreSQL client
dependencies, while pgvector configurations fail early with a useful message.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlsplit


PGVECTOR_SCHEMA_VERSION = 2
PGVECTOR_LANES = frozenset(
    {
        "chunks",
        "parents",
        "media",
        "facts",
        "evidence_spans",
        "summaries",
        "assertions",
        "entities",
        "communities",
        "page_cards",
        "actions",
    }
)
_IDENTIFIER = re.compile(r"^[a-z_][a-z0-9_]*$")
_PLACEHOLDER_TOKENS = ("change_me", "replace_me", "example.com")


class PgVectorConfigurationError(ValueError):
    """Raised when pgvector is configured unsafely or incompletely."""


class PgVectorReleaseError(RuntimeError):
    """Raised when an immutable release transition or write is invalid."""


@dataclass(frozen=True)
class VectorMatch:
    record_id: str
    score: float


def _positive_int(value: Any, *, name: str, default: int) -> int:
    parsed = int(default if value in (None, "") else value)
    if parsed <= 0:
        raise PgVectorConfigurationError(f"vector_store.{name} must be greater than zero")
    return parsed


def _non_negative_int(value: Any, *, name: str, default: int) -> int:
    parsed = int(default if value in (None, "") else value)
    if parsed < 0:
        raise PgVectorConfigurationError(f"vector_store.{name} must not be negative")
    return parsed


def _sslmode_from_dsn(dsn: str) -> str:
    configured = str(os.getenv("PGSSLMODE") or "").strip().lower()
    if configured:
        return configured
    if "://" in dsn:
        query = parse_qs(urlsplit(dsn).query)
        values = query.get("sslmode") or []
        return str(values[-1] if values else "").strip().lower()
    match = re.search(r"(?:^|\s)sslmode\s*=\s*([^\s]+)", dsn, flags=re.IGNORECASE)
    return str(match.group(1) if match else "").strip(" '\"").lower()


def _username_from_dsn(dsn: str) -> str:
    if "://" in dsn:
        return unquote(urlsplit(dsn).username or "").strip()
    match = re.search(r"(?:^|\s)user\s*=\s*([^\s]+)", dsn, flags=re.IGNORECASE)
    return str(match.group(1) if match else "").strip(" '\"")


def _validate_identifier(value: Any, *, name: str) -> str:
    identifier = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(identifier):
        raise PgVectorConfigurationError(
            f"vector_store.{name} must be a lowercase PostgreSQL identifier"
        )
    return identifier


@dataclass(frozen=True)
class PgVectorSettings:
    dsn: str
    purpose: str = "read"
    schema: str = "mbzuai_retrieval"
    records_table: str = "embedding_records"
    releases_table: str = "releases"
    dimensions: int = 1536
    schema_version: int = PGVECTOR_SCHEMA_VERSION
    pool_min_size: int = 1
    pool_max_size: int = 8
    pool_timeout_seconds: int = 5
    connect_timeout_seconds: int = 5
    max_lifetime_seconds: int = 1800
    max_idle_seconds: int = 300
    statement_timeout_ms: int = 10_000
    idle_transaction_timeout_ms: int = 5_000
    hnsw_ef_search: int = 100
    require_ssl: bool = True
    require_active_release: bool = True
    application_name: str = "mbzuai-retriever"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        purpose: str = "read",
    ) -> "PgVectorSettings":
        vector_cfg = config.get("vector_store")
        if not isinstance(vector_cfg, Mapping):
            vector_cfg = {}
        normalized_purpose = str(purpose or "read").strip().lower()
        if normalized_purpose not in {"read", "write"}:
            raise PgVectorConfigurationError("pgvector purpose must be 'read' or 'write'")

        production = bool(
            (config.get("pipeline") or {}).get("production_profile", False)
            if isinstance(config.get("pipeline"), Mapping)
            else False
        ) or str(os.getenv("SERVICE_ENVIRONMENT") or "").strip().lower() == "production"
        reader_env = str(vector_cfg.get("dsn_env") or "PGVECTOR_DSN").strip()
        writer_env = str(vector_cfg.get("ingest_dsn_env") or "PGVECTOR_INGEST_DSN").strip()
        dsn_env = writer_env if normalized_purpose == "write" else reader_env
        dsn = str(os.getenv(dsn_env) or "").strip()
        allow_shared_writer = bool(vector_cfg.get("allow_shared_dsn_for_ingest", False))
        if normalized_purpose == "write" and not dsn and allow_shared_writer:
            dsn = str(os.getenv(reader_env) or "").strip()
        if not dsn:
            raise PgVectorConfigurationError(f"{dsn_env} is required for pgvector {normalized_purpose} access")
        if any(token in dsn.lower() for token in _PLACEHOLDER_TOKENS):
            raise PgVectorConfigurationError(f"{dsn_env} contains a deployment placeholder")

        require_ssl = bool(vector_cfg.get("require_ssl", production))
        sslmode = _sslmode_from_dsn(dsn)
        if require_ssl and sslmode not in {"require", "verify-ca", "verify-full"}:
            raise PgVectorConfigurationError(
                f"{dsn_env} must set sslmode=require, verify-ca, or verify-full"
            )
        if production and normalized_purpose == "write" and not os.getenv(writer_env) and not allow_shared_writer:
            raise PgVectorConfigurationError(
                "Production ingestion requires a dedicated PGVECTOR_INGEST_DSN writer credential"
            )
        reader_dsn = str(os.getenv(reader_env) or "").strip()
        writer_dsn = str(os.getenv(writer_env) or "").strip()
        if production and reader_dsn and writer_dsn and reader_dsn == writer_dsn:
            raise PgVectorConfigurationError(
                "Production pgvector reader and writer DSNs must use distinct credentials"
            )
        expected_role = str(
            vector_cfg.get("writer_role" if normalized_purpose == "write" else "reader_role")
            or ""
        ).strip()
        actual_role = _username_from_dsn(dsn)
        if expected_role and actual_role != expected_role:
            raise PgVectorConfigurationError(
                f"{dsn_env} must use the configured dedicated {normalized_purpose} role"
            )

        dimensions = _positive_int(
            vector_cfg.get("dimensions")
            or ((config.get("embedder") or {}).get("output_dimensionality") if isinstance(config.get("embedder"), Mapping) else None),
            name="dimensions",
            default=1536,
        )
        configured_schema_version = _positive_int(
            vector_cfg.get("schema_version"),
            name="schema_version",
            default=PGVECTOR_SCHEMA_VERSION,
        )
        if configured_schema_version in {1, 2} and dimensions != 1536:
            raise PgVectorConfigurationError(
                "pgvector schema versions 1 and 2 are fixed to 1536 embedding dimensions"
            )
        pool_min = _non_negative_int(
            vector_cfg.get("pool_min_size"), name="pool_min_size", default=1
        )
        pool_max = _positive_int(
            vector_cfg.get("pool_max_size"), name="pool_max_size", default=8
        )
        if pool_min > pool_max:
            raise PgVectorConfigurationError(
                "vector_store.pool_min_size must not exceed pool_max_size"
            )

        application_name = str(
            vector_cfg.get("ingest_application_name" if normalized_purpose == "write" else "application_name")
            or ("mbzuai-indexer" if normalized_purpose == "write" else "mbzuai-retriever")
        ).strip()
        if not application_name or len(application_name) > 63:
            raise PgVectorConfigurationError(
                "vector_store application_name must contain 1 to 63 characters"
            )

        return cls(
            dsn=dsn,
            purpose=normalized_purpose,
            schema=_validate_identifier(
                vector_cfg.get("schema") or "mbzuai_retrieval", name="schema"
            ),
            records_table=_validate_identifier(
                vector_cfg.get("records_table") or "embedding_records",
                name="records_table",
            ),
            releases_table=_validate_identifier(
                vector_cfg.get("releases_table") or "releases",
                name="releases_table",
            ),
            dimensions=dimensions,
            schema_version=configured_schema_version,
            pool_min_size=pool_min,
            pool_max_size=pool_max,
            pool_timeout_seconds=_positive_int(
                vector_cfg.get("pool_timeout_seconds"),
                name="pool_timeout_seconds",
                default=5,
            ),
            connect_timeout_seconds=_positive_int(
                vector_cfg.get("connect_timeout_seconds"),
                name="connect_timeout_seconds",
                default=5,
            ),
            max_lifetime_seconds=_positive_int(
                vector_cfg.get("max_lifetime_seconds"),
                name="max_lifetime_seconds",
                default=1800,
            ),
            max_idle_seconds=_positive_int(
                vector_cfg.get("max_idle_seconds"),
                name="max_idle_seconds",
                default=300,
            ),
            statement_timeout_ms=_positive_int(
                vector_cfg.get("statement_timeout_ms"),
                name="statement_timeout_ms",
                default=10_000,
            ),
            idle_transaction_timeout_ms=_positive_int(
                vector_cfg.get("idle_transaction_timeout_ms"),
                name="idle_transaction_timeout_ms",
                default=5_000,
            ),
            hnsw_ef_search=_positive_int(
                vector_cfg.get("hnsw_ef_search"),
                name="hnsw_ef_search",
                default=100,
            ),
            require_ssl=require_ssl,
            require_active_release=bool(
                vector_cfg.get("require_active_release", production)
            ),
            application_name=application_name,
        )


def _load_dependencies():
    try:
        from pgvector.psycopg import register_vector
        from psycopg import sql
        from psycopg.rows import dict_row
        from psycopg.types.json import Jsonb
        from psycopg_pool import ConnectionPool
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError(
            "pgvector runtime dependencies are missing; install psycopg[binary,pool] and pgvector"
        ) from exc
    return ConnectionPool, Jsonb, dict_row, register_vector, sql


class PgVectorStore:
    """Pooled pgvector reader/writer with explicit release transitions."""

    def __init__(self, settings: PgVectorSettings, *, open_pool: bool = True):
        self.settings = settings
        ConnectionPool, Jsonb, dict_row, register_vector, sql = _load_dependencies()
        self._Jsonb = Jsonb
        self._register_vector = register_vector
        self._sql = sql
        self._pool = ConnectionPool(
            conninfo=settings.dsn,
            min_size=settings.pool_min_size,
            max_size=settings.pool_max_size,
            timeout=settings.pool_timeout_seconds,
            max_lifetime=settings.max_lifetime_seconds,
            max_idle=settings.max_idle_seconds,
            reconnect_timeout=settings.connect_timeout_seconds,
            kwargs={
                "autocommit": True,
                "connect_timeout": settings.connect_timeout_seconds,
                "row_factory": dict_row,
            },
            configure=self._configure_connection,
            open=False,
            name=settings.application_name,
        )
        if open_pool:
            self.open()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        purpose: str = "read",
        open_pool: bool = True,
    ) -> "PgVectorStore":
        return cls(
            PgVectorSettings.from_config(config, purpose=purpose),
            open_pool=open_pool,
        )

    def _configure_connection(self, connection: Any) -> None:
        self._register_vector(connection)
        connection.execute(
            "SELECT set_config('application_name', %s, false)",
            (self.settings.application_name,),
        )
        connection.execute(
            "SELECT set_config('statement_timeout', %s, false)",
            (f"{self.settings.statement_timeout_ms}ms",),
        )
        connection.execute(
            "SELECT set_config('idle_in_transaction_session_timeout', %s, false)",
            (f"{self.settings.idle_transaction_timeout_ms}ms",),
        )

    def open(self) -> None:
        self._pool.open(wait=True, timeout=self.settings.connect_timeout_seconds)

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def _table(self, name: str):
        return self._sql.Identifier(self.settings.schema, name)

    def _normalize_vector(self, values: Sequence[float]) -> list[float]:
        vector = [float(value) for value in values]
        if len(vector) != self.settings.dimensions:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.settings.dimensions}, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Embedding contains a non-finite value")
        if math.sqrt(sum(value * value for value in vector)) <= 0.0:
            raise ValueError("Embedding must not be a zero vector")
        return vector

    def check_schema(self) -> Dict[str, Any]:
        query = self._sql.SQL(
            """
            SELECT
                (SELECT extversion FROM pg_extension WHERE extname = 'vector') AS vector_version,
                (SELECT MAX(version) FROM {}.schema_migrations) AS schema_version,
                (
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute AS a
                    JOIN pg_class AS c ON c.oid = a.attrelid
                    JOIN pg_namespace AS n ON n.oid = c.relnamespace
                    WHERE n.nspname = %s
                      AND c.relname = %s
                      AND a.attname = 'embedding'
                      AND a.attnum > 0
                      AND NOT a.attisdropped
                ) AS embedding_type
            """
        ).format(self._sql.Identifier(self.settings.schema))
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            row = connection.execute(
                query,
                (self.settings.schema, self.settings.records_table),
            ).fetchone() or {}
        actual = int(row.get("schema_version") or 0)
        if not row.get("vector_version"):
            raise RuntimeError("PostgreSQL extension 'vector' is not installed")
        if actual != self.settings.schema_version:
            raise RuntimeError(
                f"pgvector schema version mismatch: expected {self.settings.schema_version}, got {actual}"
            )
        expected_type = f"vector({self.settings.dimensions})"
        if str(row.get("embedding_type") or "") != expected_type:
            raise RuntimeError(
                "pgvector embedding column type mismatch: "
                f"expected {expected_type}, got {row.get('embedding_type') or '<missing>'}"
            )
        return {
            "ok": True,
            "vector_version": str(row["vector_version"]),
            "schema_version": actual,
        }

    def namespace_counts(self, release_id: str) -> Dict[str, int]:
        query = self._sql.SQL(
            "SELECT namespace, COUNT(*)::bigint AS count FROM {} "
            "WHERE release_id = %s GROUP BY namespace"
        ).format(self._table(self.settings.records_table))
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            rows = connection.execute(query, (str(release_id),)).fetchall()
        return {str(row["namespace"]): int(row["count"]) for row in rows}

    def health_check(
        self,
        *,
        release_id: str | None = None,
        expected_namespaces: Mapping[str, int] | None = None,
        expected_model: str | None = None,
        expected_contract_sha256: str | None = None,
        expected_lane_counts: Mapping[str, int] | None = None,
        expected_artifact_hashes: Mapping[str, str] | None = None,
    ) -> Dict[str, Any]:
        schema = self.check_schema()
        result: Dict[str, Any] = dict(schema)
        if not release_id:
            return result
        release_query = self._sql.SQL(
            "SELECT status, project_name, embedding_model, embedding_dimensions, "
            "contract_sha256, expected_counts, artifact_hashes "
            "FROM {} WHERE release_id = %s"
        ).format(self._table(self.settings.releases_table))
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            release = connection.execute(release_query, (str(release_id),)).fetchone()
        if not release:
            raise RuntimeError(f"pgvector release does not exist: {release_id}")
        allowed = (
            {"active"}
            if self.settings.require_active_release and self.settings.purpose == "read"
            else {"ready", "active", "retired"}
        )
        if str(release["status"]) not in allowed:
            raise RuntimeError(
                f"pgvector release {release_id} is {release['status']}, expected one of {sorted(allowed)}"
            )
        if int(release["embedding_dimensions"]) != self.settings.dimensions:
            raise RuntimeError("pgvector release embedding dimension does not match runtime")
        if expected_model and str(release["embedding_model"]) != str(expected_model):
            raise RuntimeError("pgvector release embedding model does not match runtime")
        if expected_contract_sha256 and str(release["contract_sha256"]) != str(
            expected_contract_sha256
        ):
            raise RuntimeError("pgvector release indexing contract does not match runtime")
        expected_lanes = {
            str(key): int(value) for key, value in (expected_lane_counts or {}).items()
        }
        if expected_lanes and dict(release.get("expected_counts") or {}) != expected_lanes:
            raise RuntimeError("pgvector release upload plan does not match runtime manifest")
        expected_artifacts = {
            str(key): str(value)
            for key, value in (expected_artifact_hashes or {}).items()
            if str(key) and str(value)
        }
        if expected_artifacts and dict(release.get("artifact_hashes") or {}) != expected_artifacts:
            raise RuntimeError("pgvector release artifact hashes do not match runtime manifest")
        counts = self.namespace_counts(release_id)
        expected = {str(key): int(value) for key, value in (expected_namespaces or {}).items()}
        mismatches = {
            namespace: {"expected": count, "actual": int(counts.get(namespace, 0))}
            for namespace, count in expected.items()
            if int(counts.get(namespace, 0)) != count
        }
        if mismatches:
            raise RuntimeError(f"pgvector namespace count mismatch: {mismatches}")
        result.update(
            {
                "release_id": str(release_id),
                "release_status": str(release["status"]),
                "project_name": str(release["project_name"]),
                "namespace_counts": counts,
            }
        )
        return result

    def query(
        self,
        *,
        release_id: str,
        namespace: str,
        vector: Sequence[float],
        top_k: int,
    ) -> list[VectorMatch]:
        limit = int(top_k)
        if limit <= 0:
            return []
        if limit > 1000:
            raise ValueError("pgvector top_k must not exceed 1000")
        query_vector = self._normalize_vector(vector)
        status_sql = (
            self._sql.SQL("AND r.status = 'active'")
            if self.settings.require_active_release
            else self._sql.SQL("AND r.status IN ('ready', 'active', 'retired')")
        )
        query = self._sql.SQL(
            """
            SELECT e.record_id, (1 - (e.embedding <=> %s))::double precision AS score
            FROM {} AS e
            JOIN {} AS r ON r.release_id = e.release_id
            WHERE e.release_id = %s AND e.namespace = %s {}
            ORDER BY e.embedding <=> %s
            LIMIT %s
            """
        ).format(
            self._table(self.settings.records_table),
            self._table(self.settings.releases_table),
            status_sql,
        )
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(self.settings.hnsw_ef_search),),
                )
                connection.execute(
                    "SELECT set_config('hnsw.iterative_scan', 'strict_order', true)"
                )
                rows = connection.execute(
                    query,
                    (query_vector, str(release_id), str(namespace), query_vector, limit),
                ).fetchall()
        return [
            VectorMatch(record_id=str(row["record_id"]), score=float(row["score"]))
            for row in rows
        ]

    def query_ids(self, **kwargs: Any) -> list[str]:
        return [match.record_id for match in self.query(**kwargs)]

    def begin_release(
        self,
        *,
        release_id: str,
        project_name: str,
        embedding_model: str,
        contract_sha256: str,
        expected_counts: Mapping[str, int],
        artifact_hashes: Mapping[str, str] | None = None,
    ) -> None:
        self._require_writer()
        release_id = str(release_id or "").strip()
        project_name = str(project_name or "").strip()
        if not release_id or not project_name or not embedding_model or not contract_sha256:
            raise ValueError("release_id, project_name, embedding_model, and contract_sha256 are required")
        expected = {str(key): int(value) for key, value in expected_counts.items()}
        requested_artifacts = {
            str(key): str(value)
            for key, value in (artifact_hashes or {}).items()
            if str(key) and str(value)
        }
        if any(value < 0 for value in expected.values()):
            raise ValueError("expected release counts must not be negative")
        select_query = self._sql.SQL(
            "SELECT status, project_name, embedding_model, embedding_dimensions, contract_sha256, "
            "expected_counts, artifact_hashes "
            "FROM {} WHERE release_id = %s FOR UPDATE"
        ).format(self._table(self.settings.releases_table))
        insert_query = self._sql.SQL(
            """
            INSERT INTO {} (
                release_id, project_name, status, embedding_model,
                embedding_dimensions, distance_metric, contract_sha256,
                expected_counts, artifact_hashes
            ) VALUES (%s, %s, 'building', %s, %s, 'cosine', %s, %s, %s)
            """
        ).format(self._table(self.settings.releases_table))
        update_query = self._sql.SQL(
            "UPDATE {} SET updated_at = now() WHERE release_id = %s"
        ).format(self._table(self.settings.releases_table))
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            with connection.transaction():
                existing = connection.execute(select_query, (release_id,)).fetchone()
                if existing:
                    if str(existing["status"]) != "building":
                        raise PgVectorReleaseError(
                            f"release {release_id} is immutable in status {existing['status']}"
                        )
                    identity = (
                        str(existing["project_name"]),
                        str(existing["embedding_model"]),
                        int(existing["embedding_dimensions"]),
                        str(existing["contract_sha256"]),
                    )
                    requested = (
                        project_name,
                        str(embedding_model),
                        self.settings.dimensions,
                        str(contract_sha256),
                    )
                    if identity != requested:
                        raise PgVectorReleaseError(
                            f"building release {release_id} has a different immutable contract"
                        )
                    if dict(existing.get("expected_counts") or {}) != expected:
                        raise PgVectorReleaseError(
                            f"building release {release_id} has a different immutable upload plan"
                        )
                    if dict(existing.get("artifact_hashes") or {}) != requested_artifacts:
                        raise PgVectorReleaseError(
                            f"building release {release_id} has different immutable input artifacts"
                        )
                    connection.execute(update_query, (release_id,))
                else:
                    connection.execute(
                        insert_query,
                        (
                            release_id,
                            project_name,
                            str(embedding_model),
                            self.settings.dimensions,
                            str(contract_sha256),
                            self._Jsonb(expected),
                            self._Jsonb(requested_artifacts),
                        ),
                    )

    def upsert_records(
        self,
        *,
        release_id: str,
        lane: str,
        namespace: str,
        records: Sequence[Mapping[str, Any]],
    ) -> int:
        self._require_writer()
        lane = str(lane or "").strip()
        namespace = str(namespace or "").strip()
        if lane not in PGVECTOR_LANES:
            raise ValueError(f"Unsupported pgvector lane: {lane}")
        if not namespace:
            raise ValueError("pgvector namespace must not be empty")
        if not records:
            return 0
        lock_query = self._sql.SQL(
            "SELECT status FROM {} WHERE release_id = %s FOR SHARE"
        ).format(self._table(self.settings.releases_table))
        upsert_query = self._sql.SQL(
            """
            INSERT INTO {} (
                release_id, namespace, lane, record_id, retrieval_text,
                source_url, language, content_sha256, metadata, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (release_id, lane, record_id) DO UPDATE SET
                namespace = EXCLUDED.namespace,
                retrieval_text = EXCLUDED.retrieval_text,
                source_url = EXCLUDED.source_url,
                language = EXCLUDED.language,
                content_sha256 = EXCLUDED.content_sha256,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding,
                updated_at = now()
            """
        ).format(self._table(self.settings.records_table))
        values = []
        for record in records:
            record_id = str(record.get("record_id") or record.get("id") or "").strip()
            text = str(record.get("retrieval_text") or record.get("text") or "").strip()
            if not record_id or not text:
                raise ValueError(f"{lane} record_id and retrieval_text are required")
            vector = self._normalize_vector(record.get("embedding") or [])
            content_sha = str(record.get("content_sha256") or "").strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", content_sha):
                content_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
            metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
            values.append(
                (
                    str(release_id),
                    namespace,
                    lane,
                    record_id,
                    text,
                    str(record.get("source_url") or "") or None,
                    str(record.get("language") or "") or None,
                    content_sha,
                    self._Jsonb(dict(metadata)),
                    vector,
                )
            )
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            with connection.transaction():
                release = connection.execute(lock_query, (str(release_id),)).fetchone()
                if not release:
                    raise PgVectorReleaseError(f"release does not exist: {release_id}")
                if str(release["status"]) != "building":
                    raise PgVectorReleaseError(
                        f"release {release_id} is immutable in status {release['status']}"
                    )
                # psycopg 3 exposes batch execution on cursors, not directly
                # on Connection. Keep the release lock and the whole batch in
                # the same transaction while using the supported API.
                with connection.cursor() as cursor:
                    cursor.executemany(upsert_query, values)
        return len(values)

    def verify_counts(
        self,
        *,
        release_id: str,
        expected_by_namespace: Mapping[str, int],
    ) -> Dict[str, Any]:
        actual = self.namespace_counts(release_id)
        expected = {str(key): int(value) for key, value in expected_by_namespace.items()}
        failures = {
            namespace: {"expected": count, "actual": int(actual.get(namespace, 0))}
            for namespace, count in expected.items()
            if int(actual.get(namespace, 0)) != count
        }
        unexpected = {
            namespace: count
            for namespace, count in actual.items()
            if namespace not in expected and count
        }
        if unexpected:
            failures["__unexpected_namespaces__"] = unexpected
        if failures:
            raise PgVectorReleaseError(f"pgvector release count verification failed: {failures}")
        return {"expected": expected, "actual": actual, "failures": {}}

    def mark_ready(
        self,
        *,
        release_id: str,
        expected_by_namespace: Mapping[str, int],
    ) -> Dict[str, Any]:
        self._require_writer()
        expected = {str(key): int(value) for key, value in expected_by_namespace.items()}
        lock = self._sql.SQL(
            "SELECT status FROM {} WHERE release_id = %s FOR UPDATE"
        ).format(self._table(self.settings.releases_table))
        count_query = self._sql.SQL(
            "SELECT namespace, COUNT(*)::bigint AS count FROM {} "
            "WHERE release_id = %s GROUP BY namespace"
        ).format(self._table(self.settings.records_table))
        update = self._sql.SQL(
            "UPDATE {} SET status = 'ready', ready_at = now(), updated_at = now() "
            "WHERE release_id = %s RETURNING release_id"
        ).format(self._table(self.settings.releases_table))
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            with connection.transaction():
                release = connection.execute(lock, (str(release_id),)).fetchone()
                if not release or str(release["status"]) != "building":
                    status = str(release["status"]) if release else "missing"
                    raise PgVectorReleaseError(
                        f"release {release_id} could not transition from {status} to ready"
                    )
                rows = connection.execute(count_query, (str(release_id),)).fetchall()
                actual = {str(row["namespace"]): int(row["count"]) for row in rows}
                failures = {
                    namespace: {"expected": count, "actual": int(actual.get(namespace, 0))}
                    for namespace, count in expected.items()
                    if int(actual.get(namespace, 0)) != count
                }
                unexpected = {
                    namespace: count
                    for namespace, count in actual.items()
                    if namespace not in expected and count
                }
                if unexpected:
                    failures["__unexpected_namespaces__"] = unexpected
                if failures:
                    raise PgVectorReleaseError(
                        f"pgvector release count verification failed: {failures}"
                    )
                row = connection.execute(update, (str(release_id),)).fetchone()
                if not row:
                    raise PgVectorReleaseError(
                        f"release {release_id} could not transition from building to ready"
                    )
        return {"expected": expected, "actual": actual, "failures": {}}

    def activate_release(self, *, release_id: str, project_name: str) -> None:
        """Atomically activate a verified release under a project advisory lock."""
        self._require_writer()
        releases = self._table(self.settings.releases_table)
        select_target = self._sql.SQL(
            "SELECT status, project_name FROM {} WHERE release_id = %s FOR UPDATE"
        ).format(releases)
        retire = self._sql.SQL(
            "UPDATE {} SET status = 'retired', retired_at = now(), updated_at = now() "
            "WHERE project_name = %s AND status = 'active' AND release_id <> %s"
        ).format(releases)
        activate = self._sql.SQL(
            "UPDATE {} SET status = 'active', activated_at = COALESCE(activated_at, now()), "
            "retired_at = NULL, updated_at = now() WHERE release_id = %s"
        ).format(releases)
        with self._pool.connection(timeout=self.settings.pool_timeout_seconds) as connection:
            with connection.transaction():
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (f"mbzuai-pgvector:{project_name}",),
                )
                target = connection.execute(select_target, (str(release_id),)).fetchone()
                if not target:
                    raise PgVectorReleaseError(f"release does not exist: {release_id}")
                if str(target["project_name"]) != str(project_name):
                    raise PgVectorReleaseError("release project does not match activation target")
                if str(target["status"]) not in {"ready", "active"}:
                    raise PgVectorReleaseError(
                        f"release {release_id} must be ready before activation"
                    )
                connection.execute(retire, (str(project_name), str(release_id)))
                connection.execute(activate, (str(release_id),))

    def _require_writer(self) -> None:
        if self.settings.purpose != "write":
            raise PermissionError("pgvector write operation requires the ingest connection pool")
