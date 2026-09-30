"""Deterministic persona assignment for language- and task-aware scenarios."""


def select_persona(personas, task, language, usage):
    matching_language = [
        item for item in personas if _supports_language(item, language)
    ]
    candidates = matching_language or list(personas)
    matching_task = [item for item in candidates if task in item.likely_tasks]
    candidates = matching_task or candidates
    order = {item.persona_id: index for index, item in enumerate(personas)}
    return min(candidates, key=lambda item: (usage[item.persona_id], order[item.persona_id]))


def _supports_language(persona, language) -> bool:
    aliases = {
        "arabic": {"arabic", "ar"},
        "english": {"english", "en"},
    }
    expected = aliases.get(str(language).casefold(), {str(language).casefold()})
    preferred = {str(item).casefold() for item in persona.preferred_languages}
    return bool(expected & preferred)
