"""Public indexing API.

Execution lives in :mod:`lawa_rag.runner`; immutable result models live in
:mod:`lawa_rag.results`. This module preserves the original compact import
surface for callers.
"""

from .results import (
    PipelinePlanStage,
    PipelineRun,
    PipelineValidationReport,
    StageRun,
)
from .runner import IndexingPipeline, PipelineValidationError

__all__ = [
    "IndexingPipeline",
    "PipelinePlanStage",
    "PipelineRun",
    "PipelineValidationError",
    "PipelineValidationReport",
    "StageRun",
]
