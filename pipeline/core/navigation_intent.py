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

_FOLLOW_STEPS_PATTERN = (
    r"\b(?:"
    r"how (?:do|can|should) i|"
    r"what (?:do i do|comes) next|"
    r"where (?:do|should) i (?:start|begin)|"
    r"guide me|walk me through|"
    r"(?:what|which) (?:are|is) (?:the )?(?:steps?|process|procedure|instructions?)|"
    r"steps? (?:to|for|in)|"
    r"instructions? (?:to|for|on)|"
    r"(?:explain|describe|show me) (?:the )?(?:application|admission|registration) (?:process|procedure)"
    r")\b|"
    r"(?:ما هي الخطوات|ما الخطوات|كيفية|الإجراءات|الاجراءات|أرشدني|ارشدني)"
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
            r"\b(?:download|pdf|brochure|prospectus|downloadable)\b|تحميل|تنزيل|(?:يحم[ّ]?ل|يحمل)\s+(?:هذا|الزر)|ملف pdf|كتيب",
            0.94,
        ),
        (
            "contact",
            r"\b(?:contact|email|e-mail|phone|telephone|call)\b|تواصل|اتصل|مراسلة|تراسل|(?:عنوان\s+)?البريد|هاتف",
            0.94,
        ),
        (
            "apply",
            r"\b(?:apply|application portal|submit (?:my |an )?application)\b|(?:\bقد[ّ]?م\b|\bتقديم\b|\bالتقديم\b|\bطلب (?:التوظيف|الالتحاق)\b)",
            0.94,
        ),
        (
            "register",
            r"\b(?:register|registration|sign up|enrol|enroll)\b|التسجيل|سج[ّ]?ل",
            0.92,
        ),
        (
            "search",
            r"\b(?:site search|search action|search (?:button|link)|search the (?:site|website|library|catalogue|catalog))\b|(?:إجراء|زر|رابط) البحث|ابحث في الموقع|بحث الموقع",
            0.90,
        ),
        (
            "follow_steps",
            _FOLLOW_STEPS_PATTERN,
            0.84,
        ),
        (
            "open_page",
            r"\b(?:open|take me to|where (?:can|do) i find|show me (?:the )?(?:official )?page|give me (?:the )?(?:official )?(?:page|link)|link to)\b"
            r"|(?:افتح|خذني إلى|خذني الى|أين أجد|اين اجد|أعطني رابط|اعطني رابط|رابط إلى|رابط الى)",
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
    # A model planner must not turn an informational reference such as
    # "According to the admission process page, what funding is provided?"
    # into a navigation workflow merely because the page title contains the
    # word "process". Require an explicit request for steps before accepting
    # this one intent from upstream.
    if intent == "follow_steps" and not re.search(
        _FOLLOW_STEPS_PATTERN,
        _clean_text(query).casefold(),
        flags=re.IGNORECASE,
    ):
        return fallback
    # Action plans can replace otherwise relevant evidence with a button,
    # email address, or destination page. Require an explicit deterministic
    # signal for every action intent instead of trusting a model-only
    # promotion. This also protects Arabic informational verbs such as
    # "تقدمها" ("it offers") from being treated as the imperative "قدّم"
    # ("apply").
    explicit_action_intents = NAVIGATION_INTENTS - {"none", "follow_steps"}
    if intent in explicit_action_intents and fallback["intent"] != intent:
        return fallback
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
