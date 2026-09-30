"""Task-focused user personas for grounded query generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

from ._util import clean_text, json_mapping, stable_id, string_tuple
from .corpus import CorpusSnapshot
from .llm import StructuredLLM


PERSONA_PROMPT_VERSION = "persona.v1"


@dataclass(frozen=True)
class Persona:
    persona_id: str
    role: str
    goals: tuple[str, ...]
    knowledge_level: str = "general"
    preferred_languages: tuple[str, ...] = ("English",)
    query_style: str = "conversational"
    likely_tasks: tuple[str, ...] = ()
    domain: str = "generic"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        role: str,
        goals: Sequence[str],
        knowledge_level: str = "general",
        preferred_languages: Sequence[str] = ("English",),
        query_style: str = "conversational",
        likely_tasks: Sequence[str] = (),
        domain: str = "generic",
        metadata: Mapping[str, Any] | None = None,
    ) -> "Persona":
        identity = stable_id("persona", domain, role, list(goals), length=16)
        return cls(
            persona_id=identity,
            role=clean_text(role),
            goals=string_tuple(goals),
            knowledge_level=clean_text(knowledge_level) or "general",
            preferred_languages=string_tuple(preferred_languages) or ("English",),
            query_style=clean_text(query_style) or "conversational",
            likely_tasks=string_tuple(likely_tasks),
            domain=clean_text(domain) or "generic",
            metadata=json_mapping(metadata),
        ).validated()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Persona":
        persona = cls(
            persona_id=clean_text(value.get("persona_id")),
            role=clean_text(value.get("role")),
            goals=string_tuple(value.get("goals")),
            knowledge_level=clean_text(value.get("knowledge_level")) or "general",
            preferred_languages=string_tuple(value.get("preferred_languages")) or ("English",),
            query_style=clean_text(value.get("query_style")) or "conversational",
            likely_tasks=string_tuple(value.get("likely_tasks")),
            domain=clean_text(value.get("domain")) or "generic",
            metadata=json_mapping(value.get("metadata")),
        )
        if not persona.persona_id:
            values = {
                key: value
                for key, value in asdict(persona).items()
                if key != "persona_id"
            }
            return cls.create(**values)
        return persona.validated()

    def validated(self) -> "Persona":
        if not self.persona_id or not self.role or not self.goals:
            raise ValueError("Persona requires persona_id, role, and at least one goal")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_personas(
    snapshot: CorpusSnapshot,
    llm: StructuredLLM,
    *,
    count: int = 6,
    user_context: str = "",
) -> tuple[Persona, ...]:
    if count < 1:
        raise ValueError("persona count must be positive")
    payload = llm.generate_json(
        system_prompt=_persona_system_prompt(),
        user_prompt=_persona_user_prompt(snapshot, count, user_context),
        schema_name="auto_eval_personas_v1",
    )
    raw_personas = payload.get("personas")
    if not isinstance(raw_personas, list):
        raise ValueError("Persona generator response requires a personas list")
    personas = [
        _persona_from_generated(item, snapshot.domain) for item in raw_personas
    ]
    unique = {persona.persona_id: persona for persona in personas}
    if len(unique) < count:
        raise ValueError(
            f"Persona generator returned {len(unique)} valid unique personas; expected {count}"
        )
    return tuple(list(unique.values())[:count])


def _persona_from_generated(value: Any, domain: str) -> Persona:
    if not isinstance(value, Mapping):
        raise ValueError("Each generated persona must be an object")
    return Persona.create(
        role=clean_text(value.get("role")),
        goals=string_tuple(value.get("goals")),
        knowledge_level=clean_text(value.get("knowledge_level")) or "general",
        preferred_languages=string_tuple(value.get("preferred_languages"))
        or ("English",),
        query_style=clean_text(value.get("query_style")) or "conversational",
        likely_tasks=string_tuple(value.get("likely_tasks")),
        domain=domain,
        metadata={"generated": True, "prompt_version": PERSONA_PROMPT_VERSION},
    )


def _persona_system_prompt() -> str:
    return (
        "Design task-focused user personas for evaluating a grounded information system. "
        "Avoid demographic stereotypes. Roles must differ in goals, prior knowledge, "
        "and realistic tasks. "
        "Return JSON only and do not invent facts about the corpus."
    )


def _persona_user_prompt(snapshot: CorpusSnapshot, count: int, user_context: str) -> str:
    schema = {
        "personas": [
            {
                "role": "short role name",
                "goals": ["concrete information goal"],
                "knowledge_level": "beginner|general|expert",
                "preferred_languages": ["language"],
                "query_style": "natural query behavior",
                "likely_tasks": ["fact|explain|compare|synthesis|navigate|aggregate|multimodal"],
            }
        ]
    }
    return (
        f"Create exactly {count} distinct personas.\n"
        f"Optional operator context: {clean_text(user_context) or 'none'}\n"
        f"Corpus summary: {json.dumps(snapshot.summary(), ensure_ascii=False)}\n"
        f"Required shape: {json.dumps(schema, ensure_ascii=False)}"
    )


__all__ = ["PERSONA_PROMPT_VERSION", "Persona", "generate_personas"]
