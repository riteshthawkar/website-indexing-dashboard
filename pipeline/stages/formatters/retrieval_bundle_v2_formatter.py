from __future__ import annotations

from pipeline.core.registry import register_stage
from pipeline.stages.formatters.gemini_retrieval_formatter import GeminiRetrievalFormatter


@register_stage
class RetrievalBundleV2Formatter(GeminiRetrievalFormatter):
    name = "retrieval_bundle_v2"
    description = "Builds retrieval artifacts and promotes assertion-first answer records when assertion stages are available."
