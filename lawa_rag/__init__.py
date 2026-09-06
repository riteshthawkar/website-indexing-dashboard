"""Stable public API for the LawaAI RAG package.

Provider-specific retrieval and evaluation APIs live in their corresponding
submodules so importing :mod:`lawa_rag` does not initialize network clients.
"""

from ._version import __version__
from .config import available_pipeline_configs, load_pipeline_config
from .contracts import (
    ArtifactCatalog,
    ArtifactContract,
    ArtifactRecord,
    PipelineStage,
    StageContext,
    StageResult,
    StageStatus,
    register_stage,
)
from .indexing import (
    IndexingPipeline,
    PipelinePlanStage,
    PipelineRun,
    PipelineValidationError,
    PipelineValidationReport,
    StageRun,
)
from .release import ActiveRelease, ReleaseContractError, load_active_release
from .representation import (
    REPRESENTATION_V2_KIND,
    REPRESENTATION_V2_SCHEMA_VERSION,
    representation_v2_json_schema,
    stable_representation_id,
    validate_representation_v2,
)

__all__ = [
    "ActiveRelease",
    "ArtifactCatalog",
    "ArtifactContract",
    "ArtifactRecord",
    "IndexingPipeline",
    "PipelinePlanStage",
    "PipelineRun",
    "PipelineStage",
    "PipelineValidationError",
    "PipelineValidationReport",
    "REPRESENTATION_V2_KIND",
    "REPRESENTATION_V2_SCHEMA_VERSION",
    "ReleaseContractError",
    "StageContext",
    "StageResult",
    "StageRun",
    "StageStatus",
    "__version__",
    "available_pipeline_configs",
    "load_active_release",
    "load_pipeline_config",
    "register_stage",
    "representation_v2_json_schema",
    "stable_representation_id",
    "validate_representation_v2",
]
