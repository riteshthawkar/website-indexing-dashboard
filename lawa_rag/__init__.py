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
from .website_config import GENERIC_WEBSITE_CONFIG, build_website_config
from .websites import (
    DynamicCollection,
    RobotsPolicy,
    WEBSITE_PROFILE_SCHEMA_VERSION,
    WebsiteProfile,
    WebsiteProfileError,
)

__all__ = [
    "ActiveRelease",
    "ArtifactCatalog",
    "ArtifactContract",
    "ArtifactRecord",
    "DynamicCollection",
    "GENERIC_WEBSITE_CONFIG",
    "IndexingPipeline",
    "PipelinePlanStage",
    "PipelineRun",
    "PipelineStage",
    "PipelineValidationError",
    "PipelineValidationReport",
    "REPRESENTATION_V2_KIND",
    "REPRESENTATION_V2_SCHEMA_VERSION",
    "ReleaseContractError",
    "RobotsPolicy",
    "StageContext",
    "StageResult",
    "StageRun",
    "StageStatus",
    "WEBSITE_PROFILE_SCHEMA_VERSION",
    "WebsiteProfile",
    "WebsiteProfileError",
    "__version__",
    "available_pipeline_configs",
    "build_website_config",
    "load_active_release",
    "load_pipeline_config",
    "register_stage",
    "representation_v2_json_schema",
    "stable_representation_id",
    "validate_representation_v2",
]
