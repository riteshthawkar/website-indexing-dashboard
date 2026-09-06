"""Stable extension contracts for custom pipeline stages and artifacts."""

from pipeline.core.artifact_contracts import (
    ArtifactContract,
    ArtifactResolution,
    latest_artifact_record,
    missing_contract_message,
    resolve_artifact_path,
    resolve_first_artifact_path,
)
from pipeline.core.artifacts import ArtifactCatalog, ArtifactRecord
from pipeline.core.base import (
    ChunkerStage,
    CleanerStage,
    ConverterStage,
    CrawlerStage,
    EmbedderStage,
    FormatterStage,
    PipelineStage,
    QualityGate,
    StageContext,
    StageResult,
    StageStatus,
    SummarizerStage,
)
from pipeline.core.registry import list_stages, register_stage
from pipeline.core.storage import ArtifactStore, LocalArtifactStore

__all__ = [
    "ArtifactCatalog",
    "ArtifactContract",
    "ArtifactRecord",
    "ArtifactResolution",
    "ArtifactStore",
    "ChunkerStage",
    "CleanerStage",
    "ConverterStage",
    "CrawlerStage",
    "EmbedderStage",
    "FormatterStage",
    "LocalArtifactStore",
    "PipelineStage",
    "QualityGate",
    "StageContext",
    "StageResult",
    "StageStatus",
    "SummarizerStage",
    "latest_artifact_record",
    "list_stages",
    "missing_contract_message",
    "register_stage",
    "resolve_artifact_path",
    "resolve_first_artifact_path",
]
