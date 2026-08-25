"""Deterministic navigation-intent classification shared across the pipeline."""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping


NAVIGATION_INTENTS = frozenset(
    {
        "none",
        "open_page",
        "follow_steps",
        "apply",
        "register",
        "contact",
        "download",
        "login",
        "search",
    }
)


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def infer_navigation_context(query: str) -> Dict[str, Any]:
    """Classify only explicit navigation needs without calling a model."""

    text = _clean_text(query).casefold()
    intent = "none"
    confidence = 0.0
    patterns = (
        (
            "login",
            r"\b(?:log[ -]?in|sign[ -]?in|portal access)\b|تسجيل الدخول|بوابة الطالب",
            0.96,
        ),
        (
            "download",
            r"\b(?:download|pdf|brochure|prospectus|downloadable)\b|تحميل|تنزيل|ملف pdf|كتيب",
            0.94,
        ),
        (
            "contact",
            r"\b(?:contact|email|e-mail|phone|telephone|call)\b|تواصل|اتصل|البريد الإلكتروني|البريد الالكتروني|هاتف",
            0.94,
        ),
        (
            "apply",
            r"\b(?:apply|application portal|submit (?:my |an )?application)\b|(?:قد[ّ]?م|التقديم|طلب الالتحاق)",
            0.94,
        ),
        (
            "register",
            r"\b(?:register|registration|sign up|enrol|enroll)\b|التسجيل|سج[ّ]?ل",
            0.92,
        ),
        (
            "search",
            r"\b(?:site search|search the (?:site|website|library|catalogue|catalog))\b|ابحث في الموقع|بحث الموقع",
            0.90,
        ),
        (
            "follow_steps",
            r"\b(?:how (?:do|can|should) i|steps?|process|procedure|instructions?|what (?:do i do|comes) next|guide me)\b|(?:ما هي الخطوات|ما الخطوات|كيفية|الإجراءات|الاجراءات|أرشدني|ارشدني)",
            0.84,
        ),
        (
            "open_page",
            r"\b(?:open|take me to|where (?:can|do) i find|official page|website|link to)\b|(?:افتح|خذني إلى|خذني الى|الصفحة الرسمية|رابط)",
            0.82,
        ),
    )
    for candidate, pattern, candidate_confidence in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            intent = candidate
            confidence = candidate_confidence
            break
    return {
        "intent": intent,
        "goal": _clean_text(query)[:500] if intent != "none" else "",
        "confidence": confidence,
        "source": "deterministic_query_intent",
    }


def normalize_navigation_context(
    query: str, value: Mapping[str, Any] | None = None
) -> Dict[str, Any]:
    """Merge a trusted upstream intent with the deterministic fallback."""

    fallback = infer_navigation_context(query)
    if not isinstance(value, Mapping):
        return fallback
    intent = _clean_text(value.get("intent") or value.get("navigation_intent")).casefold()
    if intent not in NAVIGATION_INTENTS:
        return fallback
    try:
        confidence = max(0.0, min(1.0, float(value.get("confidence") or 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0
    if intent == "none" and fallback["intent"] != "none":
        return fallback
    if intent != "none" and confidence >= 0.5:
        return {
            "intent": intent,
            "goal": _clean_text(value.get("goal") or query)[:500],
            "confidence": confidence,
            "source": _clean_text(value.get("source")) or "upstream_query_planner",
        }
    return fallback
