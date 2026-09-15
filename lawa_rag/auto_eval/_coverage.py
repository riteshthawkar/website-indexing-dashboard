"""Internal weighted scheduling and task-availability helpers."""

from __future__ import annotations

import math
from typing import Mapping

from .corpus import CorpusSnapshot


def validate_weights(name, weights, allowed: set[str] | None) -> None:
    if not weights or sum(float(value) for value in weights.values()) <= 0:
        raise ValueError(f"{name} weights must contain a positive value")
    if any(float(value) < 0 for value in weights.values()):
        raise ValueError(f"{name} weights cannot be negative")
    unknown = set(weights) - allowed if allowed is not None else set()
    if unknown:
        raise ValueError(f"Unsupported {name}: {sorted(unknown)}")


def weighted_schedule(weights: Mapping[str, float], count: int, rng) -> list[str]:
    total = sum(float(value) for value in weights.values())
    exact = {key: count * float(value) / total for key, value in weights.items()}
    allocations = {key: math.floor(value) for key, value in exact.items()}
    remaining = count - sum(allocations.values())
    ranked = sorted(weights, key=lambda key: (-(exact[key] - allocations[key]), key))
    for key in ranked[:remaining]:
        allocations[key] += 1
    values = [key for key in weights for _item in range(allocations[key])]
    rng.shuffle(values)
    return values


def answerable_schedule(count: int, no_answer_fraction: float, rng) -> list[bool]:
    unsupported = min(count, max(0, round(count * no_answer_fraction)))
    values = [False] * unsupported + [True] * (count - unsupported)
    rng.shuffle(values)
    return values


def available_task(task: str, snapshot: CorpusSnapshot) -> str:
    kinds = {unit.kind for unit in snapshot.evidence}
    if task == "navigate" and not snapshot.actions:
        return "explain"
    if task == "multimodal" and not kinds.intersection({"media", "image", "video"}):
        return "synthesis"
    if task == "aggregate" and not kinds.intersection({"table", "database"}):
        return "compare" if len(snapshot.sources) > 1 else "explain"
    if task == "compare" and len(snapshot.sources) < 2:
        return "explain"
    return task
