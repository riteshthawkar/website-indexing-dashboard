"""
Abstract base classes for all pipeline stages.

Every stage implements `execute(ctx) -> StageResult`. The orchestrator
threads a StageContext through stages sequentially, passing outputs
from one stage as inputs to the next.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from .artifacts import ArtifactCatalog, ArtifactRecord
from .config import merge_configs
from .io import ensure_dir
from .storage import create_artifact_store


_CONFIG_SECTION_BY_STAGE_TYPE = {
    "quality_gate": "quality",
}
_KNOWN_CONFIG_SECTIONS = {
    "assertions",
    "crawler",
    "cleaner",
    "converter",
    "chunker",
    "graph",
    "summarizer",
    "quality",
    "quality_gate",
    "formatter",
    "embedder",
    "storage",
}


class StageStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageContext:
    """Immutable context passed to each stage."""

    run_id: str
    project_name: str
    config: Dict[str, Any]
    work_dir: Path
    previous_outputs: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[Dict] = None
    stage_definition: Dict[str, Any] = field(default_factory=dict)
    stage_index: int = 0
    stage_id: str = ""
    artifact_catalog: Optional[ArtifactCatalog] = None

    @property
    def current_stage_type(self) -> str:
        return str(self.stage_definition.get("type") or "")

    @property
    def current_config_section(self) -> str:
        stage_type = self.current_stage_type
        return _CONFIG_SECTION_BY_STAGE_TYPE.get(stage_type, stage_type)

    @property
    def stage_config(self) -> Dict[str, Any]:
        section = self.current_config_section
        if not section:
            return {}
        return self._config_for_section(section)

    @property
    def stage_outputs(self) -> Dict[str, Dict[str, Any]]:
        outputs = self.previous_outputs.get("stage_outputs")
        return outputs if isinstance(outputs, dict) else {}

    @property
    def crawler_config(self) -> Dict[str, Any]:
        return self._config_for_section("crawler")

    @property
    def cleaner_config(self) -> Dict[str, Any]:
        return self._config_for_section("cleaner")

    @property
    def converter_config(self) -> Dict[str, Any]:
        return self._config_for_section("converter")

    @property
    def summarizer_config(self) -> Dict[str, Any]:
        return self._config_for_section("summarizer")

    @property
    def chunker_config(self) -> Dict[str, Any]:
        return self._config_for_section("chunker")

    @property
    def graph_config(self) -> Dict[str, Any]:
        return self._config_for_section("graph")

    @property
    def quality_config(self) -> Dict[str, Any]:
        return self._config_for_section("quality")

    @property
    def formatter_config(self) -> Dict[str, Any]:
        return self._config_for_section("formatter")

    @property
    def assertions_config(self) -> Dict[str, Any]:
        return self._config_for_section("assertions")

    @property
    def embedder_config(self) -> Dict[str, Any]:
        return self._config_for_section("embedder")

    @property
    def storage_config(self) -> Dict[str, Any]:
        cfg = self.config.get("storage", {})
        return cfg if isinstance(cfg, dict) else {}

    @property
    def stage_work_dir(self) -> Path:
        if not self.stage_id:
            return self.work_dir
        return ensure_dir(self.work_dir / "stage_outputs" / self.stage_id)

    def output_dir(self, name: Optional[str] = None) -> Path:
        base = self.stage_work_dir
        return ensure_dir(base / name) if name else base

    def get_stage_outputs(self, stage_id: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.stage_outputs.get(stage_id, default or {})

    def find_artifacts(
        self,
        *,
        artifact_type: Optional[str] = None,
        producer_stage: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        if not self.artifact_catalog:
            return []
        return self.artifact_catalog.filter(
            artifact_type=artifact_type,
            producer_stage=producer_stage,
            role=role,
        )

    def make_artifact(
        self,
        local_path: str | Path,
        *,
        artifact_type: str,
        role: str,
        metadata: Optional[Dict[str, Any]] = None,
        source_artifact_ids: Optional[List[str]] = None,
    ) -> ArtifactRecord:
        store = create_artifact_store(
            self.storage_config,
            default_root=self.work_dir / "artifact_store",
        )
        producer_stage = self.stage_id or self.current_stage_type or "stage"
        return store.register(
            local_path=local_path,
            artifact_type=artifact_type,
            role=role,
            producer_stage=producer_stage,
            metadata=metadata,
            source_artifact_ids=source_artifact_ids,
        )

    def _config_for_section(self, section: str) -> Dict[str, Any]:
        base = self.config.get(section, {})
        if not isinstance(base, dict):
            return {}

        overlay_root = self.stage_definition.get("config") or {}
        if not isinstance(overlay_root, dict):
            return dict(base)

        overlay = {}
        nested = overlay_root.get(section)
        if isinstance(nested, dict):
            overlay = nested
        elif section == self.current_config_section:
            explicit_section_keys = [key for key in overlay_root if key in _KNOWN_CONFIG_SECTIONS]
            if not explicit_section_keys:
                overlay = overlay_root

        return merge_configs(base, overlay)


@dataclass
class StageResult:
    """Return value from a stage execution."""

    status: StageStatus
    outputs: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    checkpoint: Optional[Dict] = None
    artifacts: List[Dict[str, Any] | ArtifactRecord] = field(default_factory=list)
    removed_artifact_ids: List[str] = field(default_factory=list)

    @staticmethod
    def success(outputs: Dict[str, Any] = None, metrics: Dict[str, Any] = None,
                checkpoint: Dict = None, artifacts: List[Dict[str, Any] | ArtifactRecord] = None,
                removed_artifact_ids: List[str] = None) -> "StageResult":
        return StageResult(
            status=StageStatus.COMPLETED,
            outputs=outputs or {},
            metrics=metrics or {},
            checkpoint=checkpoint,
            artifacts=artifacts or [],
            removed_artifact_ids=removed_artifact_ids or [],
        )

    @staticmethod
    def failure(error_message: str, checkpoint: Dict = None) -> "StageResult":
        return StageResult(
            status=StageStatus.FAILED,
            error_message=error_message,
            checkpoint=checkpoint,
        )

    @staticmethod
    def skipped(reason: str = "") -> "StageResult":
        return StageResult(
            status=StageStatus.SKIPPED,
            error_message=reason,
        )


class PipelineStage(ABC):
    """Base class for all pipeline stages."""

    name: str = ""
    stage_type: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, ctx: StageContext) -> StageResult:
        """Run the stage. Must be idempotent when given the same checkpoint."""
        ...

    async def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Return list of config validation errors (empty = valid)."""
        return []

    async def cleanup(self, ctx: StageContext) -> None:
        """Optional cleanup hook called after stage completes or fails."""
        pass

    def estimate_work(self, ctx: StageContext) -> Optional[int]:
        """Optional: return estimated number of items for progress reporting."""
        return None

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} type={self.stage_type!r}>"


class CrawlerStage(PipelineStage):
    """Base for crawlers that produce raw files + URL mappings."""

    stage_type = "crawler"
    # Expected outputs:
    #   html_dir: Path — directory containing raw HTML files
    #   md_dir: Path — directory containing markdown files (if crawler produces them)
    #   mapping_file: Path — URL-to-file mapping JSON
    #   download_dir: Path — directory containing downloaded files (PDFs, etc.)


class CleanerStage(PipelineStage):
    """Base for content cleaners that remove boilerplate from HTML."""

    stage_type = "cleaner"
    # Expected outputs:
    #   cleaned_dir: Path — directory with cleaned HTML
    #   cleaned_count: int
    #   removed_count: int


class ConverterStage(PipelineStage):
    """Base for format converters (HTML/PDF/DOCX → Markdown)."""

    stage_type = "converter"
    # Expected outputs:
    #   md_dir: Path — directory with markdown files
    #   md_mapping_file: Path — URL-to-MD mapping JSON


class SummarizerStage(PipelineStage):
    """Base for LLM-powered summary generation."""

    stage_type = "summarizer"
    # Expected outputs:
    #   summaries_dir: Path
    #   summary_index_file: Path


class ChunkerStage(PipelineStage):
    """Base for chunkers that turn content into retrieval/indexing chunks."""

    stage_type = "chunker"
    # Expected outputs:
    #   chunks_file: Path
    #   chunk_count: int


class FormatterStage(PipelineStage):
    """Base for embedding formatters that prepare data for vector stores."""

    stage_type = "formatter"
    # Expected outputs:
    #   formatted_file: Path — JSON ready for embedding


class EmbedderStage(PipelineStage):
    """Base for embedding generation and vector store upload."""

    stage_type = "embedder"
    # Expected outputs:
    #   vectors_uploaded: int
    #   index_name: str


class QualityGate(PipelineStage):
    """A filter that runs between stages to assess/filter content quality."""

    stage_type = "quality_gate"
    # Expected outputs:
    #   passed_count: int
    #   filtered_count: int
    #   filtered_items: List[str] — IDs or paths of filtered items
