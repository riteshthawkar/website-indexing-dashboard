"""Build complete, deduplicated, topically connected evidence bundles."""

from __future__ import annotations

from typing import Mapping, Sequence

from ._coherence import comparison_compatible, coherent_bundle, topical_coherence
from ._sampling_support import deduplicated_candidates, reuse_key, stable_number
from ._util import text_tokens
from .corpus import EvidenceUnit, SourceRecord


def select_evidence(
    *,
    candidates: Sequence[EvidenceUnit],
    anchors: Sequence[EvidenceUnit],
    scenario,
    sources: Mapping[str, SourceRecord],
    relations,
    seed: int,
    usage: Mapping[str, int],
    persona=None,
) -> list[EvidenceUnit]:
    preferred = _preferred_language(anchors, scenario.language)
    offset = stable_number(scenario.scenario_id, seed) % len(preferred)
    ordered = preferred[offset:] + preferred[:offset]
    order_index = {item.evidence_id: index for index, item in enumerate(ordered)}
    ordered.sort(
        key=lambda item: (
            _reuse_penalty(item, usage),
            -_persona_relevance(item, persona, sources),
            order_index[item.evidence_id],
        )
    )
    best: list[EvidenceUnit] = []
    for anchor in ordered[: min(96, len(ordered))]:
        selected = _grow_bundle(
            anchor, candidates, scenario, sources, relations, usage, persona
        )
        if _bundle_rank(selected, scenario) > _bundle_rank(best, scenario):
            best = selected
        if _bundle_complete(selected, candidates, scenario):
            return selected
    return best


def _grow_bundle(anchor, candidates, scenario, sources, relations, usage, persona):
    selected = [anchor]
    target = min(len(candidates), scenario.difficulty.evidence_count)
    while len(selected) < target:
        remaining = [item for item in candidates if item not in selected]
        remaining = deduplicated_candidates(
            remaining,
            selected,
            sources,
            allow_cross_source_duplicates=scenario.task == "synthesis",
        )
        remaining = [
            item
            for item in remaining
            if coherent_bundle((*selected, item), sources, relations)
        ]
        if scenario.task == "compare":
            remaining = _comparison_candidates(
                remaining, selected[0], persona, sources, relations
            )
        if not remaining:
            break
        pool = _diverse_candidates(remaining, selected, scenario)
        selected.append(
            max(
                pool,
                key=lambda item: _candidate_score(
                    item, selected, scenario, sources, relations, usage, persona
                ),
            )
        )
    return selected


def _candidate_score(candidate, selected, scenario, sources, relations, usage, persona):
    coherence = max(
        topical_coherence(candidate, item, sources, relations) for item in selected
    )
    selected_sources = {item.source_id for item in selected}
    diverse = candidate.source_id not in selected_sources
    diversity = 2.0 if (
        diverse and scenario.difficulty.source_count > len(selected_sources)
    ) else 0.0
    same_source = 1.5 if (
        not diverse and scenario.difficulty.source_count <= 1
    ) else 0.0
    code = {"arabic": "ar", "english": "en"}.get(scenario.language.casefold())
    language = 0.5 if candidate.language.casefold() == code else 0.0
    reuse_penalty = min(6.0, _reuse_penalty(candidate, usage))
    relevance = 2.0 * _persona_relevance(candidate, persona, sources)
    score = 8.0 * coherence + diversity + same_source + language + relevance - reuse_penalty
    return score, candidate.evidence_id


def _diverse_candidates(remaining, selected, scenario):
    if scenario.task == "multimodal":
        media = {"media", "image", "video"}
        selected_modalities = {item.kind in media for item in selected}
        mixed = [item for item in remaining if (item.kind in media) not in selected_modalities]
        if mixed:
            return mixed
    selected_sources = {item.source_id for item in selected}
    if len(selected_sources) >= scenario.difficulty.source_count:
        return remaining
    diverse = [item for item in remaining if item.source_id not in selected_sources]
    return diverse or remaining


def _comparison_candidates(candidates, anchor, persona, sources, relations):
    comparable = [
        item for item in candidates if comparison_compatible(anchor, item, relations)
    ]
    if persona is None:
        return comparable
    anchor_relevance = _persona_relevance(anchor, persona, sources)
    minimum = max(0.1, 0.6 * anchor_relevance)
    anchor_scope = _source_relevance(anchor, persona, sources)
    scope_minimum = max(0.04, 0.5 * anchor_scope)
    return [
        item
        for item in comparable
        if _persona_relevance(item, persona, sources) >= minimum
        and _source_relevance(item, persona, sources) >= scope_minimum
        and _academic_scope_matches(item, persona, sources)
    ]


def _bundle_complete(selected, candidates, scenario) -> bool:
    target = min(len(candidates), scenario.difficulty.evidence_count)
    available_sources = len({item.source_id for item in candidates})
    required_sources = min(scenario.difficulty.source_count, available_sources)
    actual_sources = len({item.source_id for item in selected})
    return len(selected) == target and actual_sources >= required_sources


def _bundle_rank(selected, scenario) -> tuple[int, int]:
    source_count = len({item.source_id for item in selected})
    return min(source_count, scenario.difficulty.source_count), len(selected)


def _preferred_language(candidates, requested) -> list[EvidenceUnit]:
    code = {"arabic": "ar", "english": "en"}.get(str(requested).casefold())
    matching = [item for item in candidates if item.language.casefold() == code]
    return matching or list(candidates)


def _persona_relevance(unit, persona, sources) -> float:
    if persona is None:
        return 0.0
    interests = text_tokens(persona.role, persona.goals, persona.query_style)
    source = sources.get(unit.source_id)
    section_path = unit.metadata.get("section_path") or ()
    evidence = text_tokens(
        getattr(source, "title", ""), section_path, unit.text[:1000]
    )
    return len(interests & evidence) / len(interests) if interests else 0.0


def _source_relevance(unit, persona, sources) -> float:
    interests = text_tokens(persona.role, persona.goals)
    source = sources.get(unit.source_id)
    title = text_tokens(getattr(source, "title", ""))
    return len(interests & title) / len(interests) if interests else 0.0


def _academic_scope_matches(unit, persona, sources) -> bool:
    expected = _academic_levels(" ".join((persona.role, *persona.goals)))
    source = sources.get(unit.source_id)
    actual = _academic_levels(getattr(source, "title", ""))
    return not expected or not actual or bool(expected & actual)


def _academic_levels(value) -> set[str]:
    normalized = str(value).casefold().replace("’", "'")
    output = set()
    if any(term in normalized for term in ("undergraduate", "bachelor", "b.sc")):
        output.add("undergraduate")
    if any(term in normalized for term in ("master", "msc", "m.sc")):
        output.add("masters")
    if any(term in normalized for term in ("phd", "ph.d", "doctoral", "doctorate")):
        output.add("doctoral")
    return output


def _reuse_penalty(item, usage) -> float:
    semantic_key = reuse_key(item)
    return float(usage.get(item.evidence_id, 0)) + 4.0 * float(
        usage.get(semantic_key, 0) if semantic_key else 0
    )
