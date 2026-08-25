"""Vector-store adapters used by indexing and retrieval runtimes."""

from .pgvector_store import (
    PGVECTOR_SCHEMA_VERSION,
    PgVectorSettings,
    PgVectorStore,
    VectorMatch,
)

__all__ = [
    "PGVECTOR_SCHEMA_VERSION",
    "PgVectorSettings",
    "PgVectorStore",
    "VectorMatch",
]
