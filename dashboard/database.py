"""
Database models for the MBZUAI Vectorstore Pipeline Dashboard.

Slim schema — stage details live in pipeline_state.json (the pipeline's own
checkpoint file).  The DB only stores run metadata and cached aggregate metrics
for fast listing queries.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
    Boolean,
    Index,
    inspect,
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "dashboard.db"

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Run(Base):
    """A pipeline run.  Stage progress is read from pipeline_state.json on disk."""
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_name = Column(String(255), nullable=False, index=True)
    run_type = Column(String(50), nullable=False, default="full")
    config_name = Column(String(255), nullable=False, default="default")
    status = Column(String(50), nullable=False, default="pending")
    start_url = Column(String(1024), nullable=True)
    work_dir = Column(String(512), nullable=True)

    created_at = Column(DateTime, default=utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Cached aggregate metrics (refreshed from filesystem)
    pages_scraped = Column(Integer, default=0)
    documents_downloaded = Column(Integer, default=0)
    pages_cleaned = Column(Integer, default=0)
    docs_converted = Column(Integer, default=0)
    summaries_generated = Column(Integer, default=0)
    embeddings_created = Column(Integer, default=0)
    images_extracted = Column(Integer, default=0)
    videos_extracted = Column(Integer, default=0)
    media_items_extracted = Column(Integer, default=0)
    structured_documents_created = Column(Integer, default=0)
    chunks_created = Column(Integer, default=0)
    artifact_count = Column(Integer, default=0)
    chunk_strategy = Column(String(80), nullable=True)
    total_bytes = Column(Integer, default=0)

    error_message = Column(Text, nullable=True)
    is_imported = Column(Boolean, default=False)  # True = discovered from filesystem
    config_snapshot_json = Column(Text, nullable=True)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self, include_config_snapshot: bool = False) -> dict:
        data = {
            "id": self.id,
            "run_name": self.run_name,
            "run_type": self.run_type,
            "config_name": self.config_name,
            "status": self.status,
            "start_url": self.start_url,
            "work_dir": self.work_dir,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "pages_scraped": self.pages_scraped,
            "documents_downloaded": self.documents_downloaded,
            "pages_cleaned": self.pages_cleaned,
            "docs_converted": self.docs_converted,
            "summaries_generated": self.summaries_generated,
            "embeddings_created": self.embeddings_created,
            "images_extracted": self.images_extracted,
            "videos_extracted": self.videos_extracted,
            "media_items_extracted": self.media_items_extracted,
            "structured_documents_created": self.structured_documents_created,
            "chunks_created": self.chunks_created,
            "artifact_count": self.artifact_count,
            "chunk_strategy": self.chunk_strategy,
            "total_bytes": self.total_bytes,
            "error_message": self.error_message,
            "is_imported": self.is_imported,
        }
        if include_config_snapshot:
            data["config_snapshot"] = (
                json.loads(self.config_snapshot_json) if self.config_snapshot_json else None
            )
        return data


class RunLog(Base):
    """Structured log entry for a run — powers the live output and history views."""
    __tablename__ = "run_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, nullable=False, index=True)
    level = Column(String(10), nullable=False, default="info")
    stage = Column(String(100), nullable=True)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "level": self.level,
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class PineconeSnapshot(Base):
    """Point-in-time snapshot of a Pinecone index."""
    __tablename__ = "pinecone_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    index_name = Column(String(255), nullable=False, index=True)
    vector_count = Column(Integer, default=0)
    dimension = Column(Integer, default=0)
    metric = Column(String(50), nullable=True)
    namespaces_json = Column(Text, nullable=True)
    captured_at = Column(DateTime, default=utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "index_name": self.index_name,
            "vector_count": self.vector_count,
            "dimension": self.dimension,
            "metric": self.metric,
            "namespaces": json.loads(self.namespaces_json) if self.namespaces_json else {},
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
        }


# Indexes for common queries
Index("ix_runs_status", Run.status)
Index("ix_runs_created", Run.created_at.desc())
Index("ix_run_logs_run", RunLog.run_id, RunLog.created_at)


# ---------------------------------------------------------------------------
# Database lifecycle
# ---------------------------------------------------------------------------

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)
    _run_migrations()


def get_db() -> Session:
    """Get a new database session."""
    return SessionLocal()


def _run_migrations():
    """Apply lightweight schema migrations for older local databases."""
    inspector = inspect(engine)
    if "runs" not in inspector.get_table_names():
        return

    existing_columns = {col["name"] for col in inspector.get_columns("runs")}
    with engine.begin() as conn:
        if "config_snapshot_json" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN config_snapshot_json TEXT")
        if "documents_downloaded" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN documents_downloaded INTEGER DEFAULT 0")
        if "images_extracted" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN images_extracted INTEGER DEFAULT 0")
        if "videos_extracted" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN videos_extracted INTEGER DEFAULT 0")
        if "media_items_extracted" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN media_items_extracted INTEGER DEFAULT 0")
        if "structured_documents_created" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN structured_documents_created INTEGER DEFAULT 0")
        if "chunks_created" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN chunks_created INTEGER DEFAULT 0")
        if "artifact_count" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN artifact_count INTEGER DEFAULT 0")
        if "chunk_strategy" not in existing_columns:
            conn.exec_driver_sql("ALTER TABLE runs ADD COLUMN chunk_strategy VARCHAR(80)")
