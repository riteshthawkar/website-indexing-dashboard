"""Grounded, role-based automatic evaluation generation for Lawa RAG."""

from .adapters import RECORD_GROUPS, RetrievalBundleAdapter
from .builder import AutoEvalBuilder, AutoEvalGenerationError
from .cases import (
    EVAL_CASE_SCHEMA_VERSION,
    EvidenceCitation,
    GeneratedEvalCase,
    GoldClaim,
)
from .compat import export_eval_examples, to_eval_example
from .corpus import (
    CORPUS_SCHEMA_VERSION,
    CorpusAction,
    CorpusRelationship,
    CorpusSnapshot,
    EvidenceUnit,
    SourceRecord,
)
from .llm import (
    GeminiStructuredLLM,
    LLMIdentity,
    OpenAIStructuredLLM,
    RetryingStructuredLLM,
    StructuredLLM,
    StructuredLLMError,
    create_structured_llm,
)
from .personas import Persona, generate_personas
from .planning import (
    DIFFICULTY_LEVELS,
    TASK_TYPES,
    CoveragePlan,
    DifficultyProfile,
    ScenarioBlueprint,
    ScenarioPlanner,
)
from .review import approve_cases
from .sampling import EvidenceBundle, EvidenceSampler
from .suite import EVAL_SUITE_SCHEMA_VERSION, EvalSuite
from .validation import ValidationIssue, ValidationReport, validate_case, validate_cases


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "DIFFICULTY_LEVELS",
    "EVAL_CASE_SCHEMA_VERSION",
    "EVAL_SUITE_SCHEMA_VERSION",
    "RECORD_GROUPS",
    "TASK_TYPES",
    "AutoEvalBuilder",
    "AutoEvalGenerationError",
    "CorpusAction",
    "CorpusRelationship",
    "CorpusSnapshot",
    "CoveragePlan",
    "DifficultyProfile",
    "EvalSuite",
    "EvidenceBundle",
    "EvidenceCitation",
    "EvidenceSampler",
    "EvidenceUnit",
    "GeminiStructuredLLM",
    "GeneratedEvalCase",
    "GoldClaim",
    "LLMIdentity",
    "OpenAIStructuredLLM",
    "Persona",
    "RetryingStructuredLLM",
    "ScenarioBlueprint",
    "ScenarioPlanner",
    "SourceRecord",
    "StructuredLLM",
    "StructuredLLMError",
    "ValidationIssue",
    "ValidationReport",
    "create_structured_llm",
    "approve_cases",
    "export_eval_examples",
    "generate_personas",
    "to_eval_example",
    "validate_case",
    "validate_cases",
]
