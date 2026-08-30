from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence
from urllib.parse import unquote, urlparse, urlunparse
from zoneinfo import ZoneInfo

from pipeline.core.google_genai import import_genai
from pipeline.core.io import atomic_write_json
from pipeline.core.openai_client import json_completion
from pipeline.evaluation.answer_generation import generate_answer_predictions
from pipeline.evaluation.dataset import EvalExample, load_eval_examples
from pipeline.evaluation.retrieval_eval import check_metric_gates, load_eval_gates

_ANSWER_READINESS_REPORT_VERSION = 4
_PREDICTION_METADATA_VERSION = 1
_JUDGE_PROMPT_VERSION = "mbzuai-answer-readiness-judge-v3"
AnswerProgressCallback = Callable[[str, Mapping[str, Any]], None]
_OPENAI_JUDGE_CLIENT: Any | None = None
_OPENAI_JUDGE_CLIENT_CONFIG: tuple[str, float, int] | None = None


_NO_ANSWER_MARKERS = (
    "no relevant information found",
    "insufficient evidence",
    "not enough evidence",
    "could not verify",
    "couldn't verify",
    "cannot verify",
    "can't verify",
    "could not find",
    "couldn't find",
    "cannot find",
    "can't find",
    "unable to find",
    "not found",
    "not provide",
    "does not provide",
    "do not have",
    "don't have",
    "not available",
    "not in the available sources",
    "available information does not",
    "no grounded answer",
    "available sources do not",
    "official sources do not",
    "لا أستطيع العثور",
    "لا يمكنني العثور",
    "لم أتمكن من العثور",
    "لا أستطيع تقديم",
    "لا يمكنني تقديم",
    "لا تتوفر معلومات",
    "لا توجد معلومات",
    "لا توجد أدلة",
    "المصادر المتاحة لا",
    "المصادر الرسمية لا",
    "غير متوفر في المصادر",
    "تعذر التحقق",
    "لا يمكن التحقق",
)


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    bounded = min(100.0, max(0.0, float(percentile)))
    rank = (len(ordered) - 1) * (bounded / 100.0)
    lower = int(rank)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = rank - lower
    return ordered[lower] + ((ordered[upper] - ordered[lower]) * fraction)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    return value


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    try:
        value = float(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    return value


def _emit_progress(
    callback: AnswerProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(event, payload)
    except Exception:
        return


def _contains_casefolded(haystack: str, needle: str) -> bool:
    haystack = _text(haystack).casefold()
    needle = _text(needle).casefold()
    return bool(needle and needle in haystack)


_TERM_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "their",
    "to",
    "with",
    "إلى",
    "الى",
    "الي",
    "أو",
    "او",
    "أن",
    "ان",
    "في",
    "من",
    "على",
    "عن",
    "مع",
    "ما",
    "هو",
    "هي",
    "هذا",
    "هذه",
    "التي",
    "الذي",
}
_ACRONYM_SYNONYMS = {
    "ai": ("artificial intelligence",),
    "cv": ("computer vision",),
    "ml": ("machine learning",),
    "msc": ("master of science", "m.sc", "m sc"),
    "phd": ("doctor of philosophy", "ph.d", "ph d"),
    "nlp": ("natural language processing",),
    # Governed cross-lingual equivalents used when the source visual retains
    # English labels but the evaluation question and surrounding answer are
    # Arabic. These are direct terminology matches, not fuzzy paraphrases.
    "تصفية": ("filtering",),
    "تصفي": ("filtering",),
    "ترشيح": ("filtering",),
    "شريحة": ("chip",),
    "معالج": ("processor",),
}

_MONTH_TOKEN_ALIASES = {
    "january": "month01",
    "jan": "month01",
    "يناير": "month01",
    "february": "month02",
    "feb": "month02",
    "فبراير": "month02",
    "march": "month03",
    "mar": "month03",
    "مارس": "month03",
    "april": "month04",
    "apr": "month04",
    "أبريل": "month04",
    "ابريل": "month04",
    "may": "month05",
    "مايو": "month05",
    "june": "month06",
    "jun": "month06",
    "يونيو": "month06",
    "july": "month07",
    "jul": "month07",
    "يوليو": "month07",
    "august": "month08",
    "aug": "month08",
    "أغسطس": "month08",
    "اغسطس": "month08",
    "september": "month09",
    "sep": "month09",
    "sept": "month09",
    "سبتمبر": "month09",
    "october": "month10",
    "oct": "month10",
    "أكتوبر": "month10",
    "اكتوبر": "month10",
    "november": "month11",
    "nov": "month11",
    "نوفمبر": "month11",
    "december": "month12",
    "dec": "month12",
    "ديسمبر": "month12",
}

_ARABIC_DURATION_TOKEN_ALIASES = {
    "سنتان": "two_years",
    "سنتين": "two_years",
    "عامان": "two_years",
    "عامين": "two_years",
}

_ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)
_ARABIC_MATCH_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
    }
)


def _normalize_arabic_for_match(value: str) -> str:
    value = value.translate(_ARABIC_MATCH_TRANSLATION).replace("ـ", "")
    # Accusative tanween is commonly written as root + fathatan + supporting
    # alif (for example, "مخصصًا"). Remove that grammatical ending before the
    # general diacritic pass so it matches the uninflected benchmark term.
    value = value.replace("\u064b\u0627", "").replace("\u0627\u064b", "")
    return _ARABIC_DIACRITICS_RE.sub("", value)


def _normalize_for_term_match(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).casefold()
    # User-facing source labels and paths may retain percent encoding from a
    # URL. Decode twice at most so ``Statistics%20for...`` and a safely
    # double-encoded equivalent are scored as the same visible title.
    for _attempt in range(2):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    text = _normalize_arabic_for_match(text)
    # Thousands separators are formatting, not part of the numeric fact.
    # Keep decimal commas intact by requiring a three-digit group.
    text = re.sub(r"(?<=\d)[,\u066c](?=\d{3}(?:\D|$))", "", text)
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    spelling_equivalents = {
        "analyze": "analyse",
        "analyzed": "analysed",
        "analyzes": "analyses",
        "analyzing": "analysing",
    }
    text = re.sub(
        r"\b(?:analyze|analyzed|analyzes|analyzing)\b",
        lambda match: spelling_equivalents[match.group(0)],
        text,
    )
    text = text.replace("&", " and ")
    # Hyphenation is a presentation choice, not a semantic distinction for
    # required prose terms (for example, "generative-AI" vs "generative AI").
    text = re.sub(
        r"(?<=[a-z0-9\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff])-(?=[a-z0-9\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff])",
        " ",
        text,
    )
    text = _normalize_temporal_tokens(text)
    text = re.sub(r"(?<=\b[a-z])\.(?=[a-z]\b)", "", text)
    text = re.sub(r"[^a-z0-9\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff@._%+\-/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_temporal_tokens(text: str) -> str:
    def _arabic_minute_time(match: re.Match[str]) -> str:
        hour = str(int(match.group(1)))
        minute = str(int(match.group(2))).zfill(2)
        suffix = "am" if match.group(3).startswith("صباح") else "pm"
        return f"{hour}{'' if minute == '00' else minute}{suffix}"

    text = re.sub(
        r"\b(\d{1,2})\s*[:.]\s*(\d{2})\s*(صباحا?|مساءا?|ظهرا?)\b",
        _arabic_minute_time,
        text,
    )
    text = re.sub(
        r"\b(\d{1,2})\s*(صباحا?|مساءا?|ظهرا?)\b",
        lambda match: f"{int(match.group(1))}{'am' if match.group(2).startswith('صباح') else 'pm'}",
        text,
    )

    def _minute_time(match: re.Match[str]) -> str:
        hour = str(int(match.group(1)))
        minute = match.group(2)
        meridiem = "am" if match.group(3).startswith("a") else "pm"
        return f"{hour}{'' if minute == '00' else minute}{meridiem}"

    text = re.sub(
        r"\b(\d{1,2})\s*[:.]\s*(\d{2})\s*(a\.?m\.?|p\.?m\.?)\b",
        _minute_time,
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)\b",
        lambda match: f"{int(match.group(1))}{'am' if match.group(2).casefold().startswith('a') else 'pm'}",
        text,
        flags=re.IGNORECASE,
    )
    return text


_CLOCK_EXPRESSION_RE = re.compile(
    r"\b(\d{1,2})\s*[:.]\s*(\d{2})(?:\s*(a\.?m\.?|p\.?m\.?|صباحا?|مساءا?|ظهرا?))?\b",
    re.IGNORECASE,
)


def _clock_alias_groups(value: Any) -> List[set[str]]:
    text = _normalize_arabic_for_match(unicodedata.normalize("NFKC", _text(value)).casefold())
    groups: List[set[str]] = []
    for match in _CLOCK_EXPRESSION_RE.finditer(text):
        try:
            hour = int(match.group(1))
        except ValueError:
            continue
        minute = str(int(match.group(2))).zfill(2)
        meridiem = match.group(3)
        if hour <= 0 or hour > 24:
            continue
        display_hour = hour if 1 <= hour <= 12 else ((hour - 1) % 12) + 1
        compact_time = f"{display_hour}{'' if minute == '00' else minute}"
        if meridiem:
            suffix = "am" if meridiem.startswith(("a", "صباح")) else "pm"
            groups.append({f"{compact_time}{suffix}"})
        else:
            groups.append({f"{compact_time}am", f"{compact_time}pm"})
    return groups


def _remove_clock_expressions(value: Any) -> str:
    normalized = _normalize_arabic_for_match(unicodedata.normalize("NFKC", _text(value)).casefold())
    return _CLOCK_EXPRESSION_RE.sub(" ", normalized)


def _all_clock_aliases_supported(response_tokens: set[str], response_blob: str, groups: Sequence[set[str]]) -> bool:
    return all(any(alias in response_tokens or f" {alias} " in response_blob for alias in group) for group in groups)


def _term_tokens(value: Any) -> List[str]:
    normalized = _normalize_for_term_match(value)
    # A slash inside prose is often a compact coordination mark rather than a
    # semantic token boundary (for example, "شريحة/معالج"). Split these
    # compounds for token coverage. Required terms that intentionally express
    # alternatives ("login/start page") are handled before tokenization by
    # ``_required_term_supported``.
    normalized = re.sub(
        r"(?<=[a-z\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff])/(?=[a-z\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff])",
        " ",
        normalized,
    )
    tokens = re.findall(
        r"[a-z0-9\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff@._%+\-/]+",
        normalized,
    )
    output: List[str] = []
    for token in tokens:
        if token in _TERM_STOPWORDS:
            continue
        if "@" not in token:
            token = token.strip("._-/،؛؟")
        if re.fullmatch(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+", token):
            if token.startswith("وال") and len(token) > 5:
                token = token[3:]
            elif token.startswith(("بال", "كال", "فال")) and len(token) > 5:
                token = token[3:]
            elif token.startswith("لل") and len(token) > 4:
                token = token[2:]
            elif token.startswith("ل") and len(token) > 4:
                token = token[1:]
            elif token.startswith("و") and len(token) > 4:
                token = token[1:]
            elif token.startswith("ال") and len(token) > 4:
                token = token[2:]
            if token.endswith(("يون", "يين")) and len(token) > 5:
                token = f"{token[:-3]}ي"
            elif token.endswith("ية") and len(token) > 4:
                token = f"{token[:-2]}ي"
        if len(token) > 4 and token.endswith("s") and "@" not in token:
            token = token[:-1]
        token = _MONTH_TOKEN_ALIASES.get(token, token)
        token = _ARABIC_DURATION_TOKEN_ALIASES.get(token, token)
        if token:
            output.append(token)
    return output


_ARABIC_TERM_TOKEN_RE = re.compile(
    r"^[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+$"
)
_ARABIC_LEXICAL_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    # Governed, narrow semantic equivalents observed in source-grounded
    # Arabic answers. These avoid failing the release gate on direct lexical
    # alternations while leaving numbers, entities, and factual qualifiers
    # subject to exact matching.
    frozenset({"تراكم", "تجمع"}),
    frozenset({"تعاون", "تعاوني"}),
    frozenset({"اداري", "ادارة"}),
)
_ENGLISH_LEXICAL_EQUIVALENCE_GROUPS: tuple[frozenset[str], ...] = (
    frozenset({"ahead", "before"}),
)


def _arabic_term_token_variants(token: str) -> set[str]:
    """Return conservative light-morphology variants for answer scoring.

    Arabic clitics and attached pronouns are orthographic, not factual,
    differences. Keep the original token and add only common one-step forms;
    this avoids turning deterministic scoring into unrestricted fuzzy match.
    """

    value = str(token or "").strip()
    variants = {value} if value else set()
    if not value or not _ARABIC_TERM_TOKEN_RE.fullmatch(value):
        return variants

    for prefix in ("ب", "ك", "و"):
        minimum_root_length = 3 if prefix == "و" else 4
        if value.startswith(prefix) and len(value) - len(prefix) >= minimum_root_length:
            variants.add(value[len(prefix) :])
    if value.startswith("ي") and len(value) >= 6:
        # Imperfect verbs such as "يتعاون" should match their lexical
        # concept "تعاون", while short nouns remain untouched.
        variants.add(value[1:])
    if value in {"ذوو", "ذوي"}:
        variants.add("ذو")
    if value in {"مساهمة", "اسهام"}:
        variants.add("ساهم")
    if len(value) >= 6 and value.endswith(("ون", "ين")):
        # Sound masculine plural case endings do not change the underlying
        # entity or qualification (for example باحثون / باحثين).
        variants.add(value[:-2])
    for suffix in ("نا", "هم", "هن", "كم", "كن", "ها"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 3:
            base = value[: -len(suffix)]
            variants.add(base)
            if base.endswith("ات") and len(base) - 2 >= 3:
                # Sound feminine plurals keep the same lexical fact after an
                # attached pronoun: استفساراتكم -> استفسارات -> استفسار.
                variants.add(base[:-2])
            # Taa marbuta is written as taa before an attached possessive
            # pronoun: مكانة -> مكانتها. Preserve that grammatical identity.
            if base.endswith("ت"):
                variants.add(f"{base[:-1]}ة")
    if value.endswith("ات") and len(value) - 2 >= 3:
        variants.add(value[:-2])
    return variants


def _english_term_token_variants(token: str) -> set[str]:
    """Return conservative English inflection variants for scoring only."""

    value = str(token or "").strip()
    variants = {value} if value else set()
    if not re.fullmatch(r"[a-z]+", value):
        return variants
    if value.endswith("ing") and len(value) > 6:
        stem = value[:-3]
        variants.add(stem)
        if stem and not stem.endswith("e"):
            variants.add(f"{stem}e")
    if value in {"interpret", "interpretable", "interpretation"}:
        variants.add("interpret")
    return variants


def _term_token_supported(
    token: str,
    response_tokens: set[str],
    normalized_response: str,
) -> bool:
    synonyms = _ACRONYM_SYNONYMS.get(token, ())
    if (
        token in response_tokens
        or f" {token} " in normalized_response
        or any(synonym in normalized_response for synonym in synonyms)
    ):
        return True
    if _ARABIC_TERM_TOKEN_RE.fullmatch(token):
        token_variants = _arabic_term_token_variants(token)
        if any(
            token_variants & _arabic_term_token_variants(response_token)
            for response_token in response_tokens
        ):
            return True
        equivalent_groups = [
            group
            for group in _ARABIC_LEXICAL_EQUIVALENCE_GROUPS
            if group & token_variants
        ]
        return any(
            group & _arabic_term_token_variants(response_token)
            for group in equivalent_groups
            for response_token in response_tokens
        )
    if re.fullmatch(r"[a-z]+", token):
        token_variants = _english_term_token_variants(token)
        if any(
            token_variants & _english_term_token_variants(response_token)
            for response_token in response_tokens
        ):
            return True
        equivalent_groups = [
            group
            for group in _ENGLISH_LEXICAL_EQUIVALENCE_GROUPS
            if group & token_variants
        ]
        return any(
            group & _english_term_token_variants(response_token)
            for group in equivalent_groups
            for response_token in response_tokens
        )
    return False


def _required_term_supported(response: str, term: str) -> bool:
    slash_alternative = re.search(
        r"\b([a-z]+)\s*/\s*([a-z]+)\b",
        _text(term).casefold(),
    )
    if slash_alternative:
        prefix = _text(term)[: slash_alternative.start()]
        suffix = _text(term)[slash_alternative.end() :]
        return any(
            _required_term_supported(response, f"{prefix}{alternative}{suffix}")
            for alternative in slash_alternative.groups()
        )

    normalized_response = _normalize_for_term_match(response)
    normalized_term = _normalize_for_term_match(term)
    if not normalized_term:
        return True
    if normalized_term in normalized_response:
        return True

    term_tokens = _term_tokens(term)
    if not term_tokens:
        return True

    response_tokens = set(_term_tokens(response))
    response_blob = f" {normalized_response} "
    clock_groups = _clock_alias_groups(term)
    if clock_groups and _all_clock_aliases_supported(response_tokens, response_blob, clock_groups):
        non_clock_tokens = _term_tokens(_remove_clock_expressions(term))
        if not non_clock_tokens:
            return True
        non_clock_matches = 0
        for token in non_clock_tokens:
            if _term_token_supported(token, response_tokens, response_blob):
                non_clock_matches += 1
        return non_clock_matches / float(len(non_clock_tokens)) >= 0.75

    hard_tokens = [
        token
        for token in term_tokens
        if "@" in token or any(char.isdigit() for char in token) or "/" in token
    ]
    for token in hard_tokens:
        if token not in response_tokens and token not in response_blob:
            return False

    matched = 0
    for token in term_tokens:
        if _term_token_supported(token, response_tokens, response_blob):
            matched += 1
    coverage = matched / float(len(term_tokens))
    if len(term_tokens) <= 2:
        return coverage >= 1.0
    return coverage >= 0.75


def _required_terms_result(response: str, terms: Sequence[str], metadata: Mapping[str, Any]) -> tuple[List[str], float]:
    alternative_groups = _metadata_term_groups(
        metadata,
        "answer_must_include_any_groups",
    )
    if not terms and not alternative_groups:
        return [], 1.0
    missing = [term for term in terms if not _required_term_supported(response, term)]
    missing_groups = [
        group
        for group in alternative_groups
        if not any(_required_term_supported(response, alternative) for alternative in group)
    ]
    total_requirements = len(terms) + len(alternative_groups)
    supported_requirements = total_requirements - len(missing) - len(missing_groups)
    coverage = supported_requirements / float(total_requirements)
    try:
        minimum = float(metadata.get("answer_must_include_min_coverage", 1.0))
    except (TypeError, ValueError):
        minimum = 1.0
    minimum = min(1.0, max(0.0, minimum))
    if coverage >= minimum:
        return [], coverage
    missing.extend(" OR ".join(group) for group in missing_groups)
    return missing, coverage


def _metadata_list(metadata: Mapping[str, Any], key: str) -> List[str]:
    value = metadata.get(key)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_text(item) for item in value if _text(item)]
    return [_text(value)] if _text(value) else []


def _metadata_term_groups(metadata: Mapping[str, Any], key: str) -> List[List[str]]:
    """Load groups where satisfying any one term satisfies that group.

    This is intentionally distinct from ``answer_must_include`` (all terms
    required). It supports translated or synonymous gold expressions without
    lowering the required coverage for independent facts.
    """

    value = metadata.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    groups: List[List[str]] = []
    for raw_group in value:
        if isinstance(raw_group, str):
            group = [_text(raw_group)] if _text(raw_group) else []
        elif isinstance(raw_group, Sequence) and not isinstance(raw_group, (bytes, bytearray)):
            group = [_text(item) for item in raw_group if _text(item)]
        else:
            group = []
        if group:
            groups.append(list(dict.fromkeys(group)))
    return groups


def _looks_like_no_answer(response: str, response_kind: str = "") -> bool:
    normalized = _normalize_for_term_match(response)
    kind = _text(response_kind).casefold()
    if "no_answer" in kind:
        return True
    if not normalized:
        return False
    no_answer_starter = any(
        normalized.startswith(_normalize_for_term_match(marker))
        for marker in (
            "i couldn’t confirm",
            "i couldn't confirm",
            "i could not confirm",
            "i cannot confirm",
            "i can’t confirm",
            "i can't confirm",
            "i couldn’t verify",
            "i couldn't verify",
            "i could not verify",
            "i cannot verify",
            "i can’t verify",
            "i can't verify",
            "i couldn’t find",
            "i couldn't find",
            "i could not find",
            "i cannot find",
            "i can’t find",
            "i can't find",
            "i do not have any information",
            "i don't have any information",
            "i do not currently have information",
            "i don't currently have information",
            "i have no information",
            "the available sources do not show",
            "the sources do not show",
            "there is insufficient evidence",
            "insufficient evidence",
            "not enough evidence",
        )
    )
    if no_answer_starter:
        return True
    has_inline_citation = bool(re.search(r"\[(?:\d+|source\s+\d+)(?:\s*,\s*\d+)*\]", response or "", re.IGNORECASE))
    if has_inline_citation:
        return False
    return len(normalized) < 420 and any(
        _normalize_for_term_match(marker) in normalized for marker in _NO_ANSWER_MARKERS
    )


_FORBIDDEN_TERM_NEGATION_BEFORE_RE = re.compile(
    r"(?:\b(?:cannot|can't|does\s+not|doesn't|do\s+not|don't|is\s+not|isn't|"
    r"are\s+not|aren't|no|never|not|without)\b|(?:لا|ليس|ليست|غير))[^.;:!?]{0,70}$",
    re.IGNORECASE,
)
_FORBIDDEN_TERM_NEGATION_AFTER_RE = re.compile(
    r"^[^.;:!?]{0,45}(?:\b(?:is\s+not|isn't|are\s+not|aren't|was\s+not|"
    r"wasn't|were\s+not|weren't|not)\s+(?:covered|guaranteed|included|listed|"
    r"offered|provided|supported)\b|(?:غير\s+(?:مضمون|مشمول|مدرج|متاح)))",
    re.IGNORECASE,
)


def _forbidden_term_present(response: str, term: str) -> bool:
    """Return true only when a prohibited fact is asserted, not explicitly denied."""

    normalized_response = _normalize_for_term_match(response)
    normalized_term = _normalize_for_term_match(term)
    if not normalized_response or not normalized_term:
        return False
    for match in re.finditer(re.escape(normalized_term), normalized_response):
        prefix = normalized_response[max(0, match.start() - 90) : match.start()]
        suffix = normalized_response[match.end() : match.end() + 80]
        negated_before = bool(_FORBIDDEN_TERM_NEGATION_BEFORE_RE.search(prefix))
        negated_after = bool(_FORBIDDEN_TERM_NEGATION_AFTER_RE.search(suffix))
        if "not only" in prefix[-20:]:
            negated_before = False
        if not (negated_before or negated_after):
            return True
    return False


def _explicitly_denies_unsupported_premise(response: str) -> bool:
    """Recognize direct premise denials for benchmark no-answer examples.

    This is intentionally separate from ``_looks_like_no_answer``. A supported
    negative fact can be a perfectly valid answer to an answerable question, so
    broadening the generic detector would incorrectly reject those responses.
    The scorer invokes this helper only when the gold example is explicitly
    marked as no-answer.
    """

    normalized = _normalize_for_term_match(response)
    if not normalized:
        return False
    explicit_denials = (
        "do not have any information",
        "don't have any information",
        "does not have any information",
        "doesn't have any information",
        "do not have any information showing",
        "don't have any information showing",
        "does not have any information showing",
        "doesn't have any information showing",
        "does not have",
        "doesn't have",
        "does not operate",
        "doesn't operate",
        "has no",
        "there is no",
        "لا يوجد",
        "لا توجد",
        "لا يملك",
        "لا تملك",
        "لا يشغل",
        "لا تشغل",
    )
    return any(
        _normalize_for_term_match(marker) in normalized for marker in explicit_denials
    )


def _inline_citation_present(response: str) -> bool:
    return bool(re.search(r"\[(?:\d+|source\s+\d+)(?:\s*,\s*\d+)*\]", response or "", re.IGNORECASE))


def _normalize_url_for_match(value: Any) -> str:
    # Browsers and clients may surface the same Unicode URL either literally or
    # percent-encoded. Some legacy citations were encoded twice, so decode a
    # small bounded number of times rather than treating `%25D8...` as a
    # different official page.
    text = _text(value)
    for _ in range(3):
        decoded = unquote(text)
        if decoded == text:
            break
        text = decoded
    text = text.casefold()
    if not text:
        return ""
    text = re.sub(r"#.*$", "", text)
    text = re.sub(r"\?.*$", "", text)
    text = re.sub(r"/+$", "", text)
    return text


def _candidate_source_urls(row: Mapping[str, Any]) -> List[str]:
    urls: List[str] = []
    sources = row.get("sources")
    if isinstance(sources, Sequence) and not isinstance(sources, (str, bytes, bytearray)):
        for source in sources:
            if isinstance(source, Mapping):
                urls.extend(
                    _text(source.get(key))
                    for key in ("url", "source_url", "href", "page_source")
                    if _text(source.get(key))
                )
            else:
                urls.append(_text(source))
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    metadata_sources = metadata.get("sources")
    if isinstance(metadata_sources, Sequence) and not isinstance(metadata_sources, (str, bytes, bytearray)):
        for source in metadata_sources:
            if isinstance(source, Mapping):
                urls.extend(
                    _text(source.get(key))
                    for key in ("url", "source_url", "href", "page_source")
                    if _text(source.get(key))
                )
    return [url for url in urls if url]


def _expected_reference_url_groups(metadata: Mapping[str, Any]) -> List[List[str]]:
    expected_urls = [_normalize_url_for_match(url) for url in _metadata_list(metadata, "expected_reference_urls")]
    expected_urls = [url for url in expected_urls if url]
    alternate = metadata.get("alternate_expected_reference_urls")
    alternates_by_expected: Dict[str, List[str]] = {}
    if isinstance(alternate, Mapping):
        for key, values in alternate.items():
            normalized_key = _normalize_url_for_match(key)
            if normalized_key:
                alternates_by_expected[normalized_key] = [
                    _normalize_url_for_match(value)
                    for value in (values if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)) else [values])
                    if _normalize_url_for_match(value)
                ]
    return [
        list(dict.fromkeys([url, *alternates_by_expected.get(url, [])]))
        for url in expected_urls
    ]


def _url_group_supported(expected_group: Sequence[str], actual_urls: Sequence[str]) -> bool:
    for expected in expected_group:
        for actual in actual_urls:
            if actual == expected or actual.startswith(expected + "/") or expected.startswith(actual + "/"):
                return True
    return False


def _expected_reference_url_pass(metadata: Mapping[str, Any], row: Mapping[str, Any], *, no_answer: bool) -> float:
    expected_groups = _expected_reference_url_groups(metadata)
    if not expected_groups or no_answer:
        return 1.0
    candidate_urls = [_normalize_url_for_match(url) for url in _candidate_source_urls(row)]
    candidate_urls = [url for url in candidate_urls if url]
    if not candidate_urls:
        return 0.0
    supported = sum(1 for expected_group in expected_groups if _url_group_supported(expected_group, candidate_urls))
    return supported / float(len(expected_groups))


def _source_support_present(row: Mapping[str, Any]) -> bool:
    sources = row.get("sources")
    if isinstance(sources, list) and sources:
        return True
    contexts = row.get("retrieved_contexts")
    if isinstance(contexts, list) and any(_text(item) for item in contexts):
        return True
    reference_contexts = row.get("reference_contexts")
    if isinstance(reference_contexts, list) and any(_text(item) for item in reference_contexts):
        return True
    evidence_pack = row.get("evidence_pack")
    if isinstance(evidence_pack, Mapping):
        items = evidence_pack.get("items")
        if isinstance(items, list) and any(isinstance(item, Mapping) and _text(item.get("text")) for item in items):
            return True
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    metadata_sources = metadata.get("sources")
    if isinstance(metadata_sources, list) and metadata_sources:
        return True
    return False


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(dict(row), ensure_ascii=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _fingerprint_payload(payload: Any) -> str:
    return hashlib.sha1(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _example_fingerprint(example: EvalExample) -> str:
    return _fingerprint_payload(example.to_dict())


def _dataset_fingerprint(examples: Sequence[EvalExample]) -> str:
    return _fingerprint_payload([example.to_dict() for example in examples])


def _prediction_metadata(
    *,
    example: EvalExample,
    dataset_fingerprint: str,
    config_name: str,
    work_dir: str | Path,
    backend: str,
    mode: str,
    endpoint: str = "",
    eval_request_mode: bool = False,
    probe_mode: bool = False,
) -> Dict[str, Any]:
    return {
        "eval_metadata_version": _PREDICTION_METADATA_VERSION,
        "eval_dataset_fingerprint": dataset_fingerprint,
        "eval_example_fingerprint": _example_fingerprint(example),
        "eval_config_name": str(config_name),
        "eval_work_dir": str(Path(work_dir).expanduser().resolve()),
        "eval_backend": str(backend),
        "eval_mode": str(mode),
        "eval_endpoint": str(endpoint or ""),
        "eval_request_mode": bool(eval_request_mode),
        "eval_probe_mode": bool(probe_mode),
    }


def _stamp_prediction_row(
    row: Mapping[str, Any],
    *,
    example: EvalExample,
    dataset_fingerprint: str,
    config_name: str,
    work_dir: str | Path,
    backend: str,
    mode: str,
    endpoint: str = "",
    eval_request_mode: bool = False,
    probe_mode: bool = False,
) -> Dict[str, Any]:
    output = dict(row)
    metadata = dict(output.get("metadata") or {})
    metadata.update(
        _prediction_metadata(
            example=example,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend=backend,
            mode=mode,
            endpoint=endpoint,
            eval_request_mode=eval_request_mode,
            probe_mode=probe_mode,
        )
    )
    output["metadata"] = metadata
    return output


def _prediction_row_matches_current_eval(
    row: Mapping[str, Any],
    *,
    example: EvalExample,
    dataset_fingerprint: str,
    config_name: str,
    work_dir: str | Path,
    backend: str,
    mode: str,
    endpoint: str = "",
    eval_request_mode: bool = False,
    probe_mode: bool = False,
) -> bool:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    expected = _prediction_metadata(
        example=example,
        dataset_fingerprint=dataset_fingerprint,
        config_name=config_name,
        work_dir=work_dir,
        backend=backend,
        mode=mode,
        endpoint=endpoint,
        eval_request_mode=eval_request_mode,
        probe_mode=probe_mode,
    )
    return all(metadata.get(key) == value for key, value in expected.items())


def _prediction_row_is_complete(row: Mapping[str, Any]) -> bool:
    """Return whether a cached prediction is safe to reuse.

    Production chat can return a useful-looking response prefix together with
    an explicit partial/error completion contract.  Reusing that row merely
    because its evaluation identity still matches would turn a transient
    provider interruption into a durable release result.
    """

    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    response_contract = (
        row.get("response_contract")
        if isinstance(row.get("response_contract"), Mapping)
        else {}
    )
    if _text(row.get("error")) or _text(metadata.get("error")):
        return False
    if any(
        value is True
        for value in (
            row.get("partial"),
            metadata.get("partial"),
            response_contract.get("partial"),
        )
    ):
        return False
    if any(
        value is False
        for value in (
            row.get("completed"),
            metadata.get("completed"),
            response_contract.get("completed"),
        )
    ):
        return False
    finish_reason = _text(row.get("finish_reason") or metadata.get("finish_reason")).casefold()
    if finish_reason in {"error", "stream_error", "timeout", "cancelled", "canceled"}:
        return False
    return bool(_text(row.get("response")))


def _filter_resumable_prediction_file(
    *,
    predictions_path: str | Path,
    examples: Sequence[EvalExample],
    dataset_fingerprint: str,
    config_name: str,
    work_dir: str | Path,
    backend: str,
    mode: str,
    endpoint: str = "",
    eval_request_mode: bool = False,
    probe_mode: bool = False,
) -> Dict[str, Any]:
    path = Path(predictions_path)
    if not path.exists():
        return {"kept_count": 0, "discarded_count": 0, "discarded_ids": []}
    examples_by_id = {example.id: example for example in examples}
    kept: List[Dict[str, Any]] = []
    discarded_ids: List[str] = []
    for row in _read_jsonl(path):
        row_id = _text(row.get("id"))
        example = examples_by_id.get(row_id)
        if not example:
            discarded_ids.append(row_id or "<missing>")
            continue
        if _prediction_row_matches_current_eval(
            row,
            example=example,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend=backend,
            mode=mode,
            endpoint=endpoint,
            eval_request_mode=eval_request_mode,
            probe_mode=probe_mode,
        ) and _prediction_row_is_complete(row):
            kept.append(dict(row))
        else:
            discarded_ids.append(row_id)
    if discarded_ids:
        _write_jsonl(path, kept)
    return {
        "kept_count": len(kept),
        "discarded_count": len(discarded_ids),
        "discarded_ids": discarded_ids[:100],
    }


_JUDGE_SCORE_FIELDS = (
    "correctness",
    "groundedness",
    "relevance",
    "helpfulness",
    "completeness",
    "citation_quality",
    "component_quality",
    "safety",
    "overall",
)


def _make_judge_client():
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY is required for LLM judge evaluation")
    return import_genai().Client(api_key=api_key)


def _truncate_jsonable(value: Any, *, max_chars: int = 9000) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text[:max_chars] + ("...[truncated]" if len(text) > max_chars else "")
    if isinstance(value, Mapping):
        output: Dict[str, Any] = {}
        used = 0
        for key, item in value.items():
            if used >= max_chars:
                output["_truncated"] = True
                break
            compact_item = _truncate_jsonable(item, max_chars=max(200, max_chars - used))
            output[str(key)] = compact_item
            used += len(json.dumps(compact_item, ensure_ascii=True, default=str))
        return output
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        output_list = []
        used = 0
        for item in value:
            if used >= max_chars:
                output_list.append({"_truncated": True})
                break
            compact_item = _truncate_jsonable(item, max_chars=max(200, max_chars - used))
            output_list.append(compact_item)
            used += len(json.dumps(compact_item, ensure_ascii=True, default=str))
        return output_list
    return value


def _clean_judge_reasons(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_text(value)[:500]] if _text(value) else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_text(item)[:500] for item in value if _text(item)][:8]
    return [_text(value)[:500]] if _text(value) else []


def _bounded_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, score))


def _extract_json_object(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            payload, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("judge response did not contain a JSON object")


def _response_component_payload(row: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    return {
        "response_kind": row.get("response_kind") or metadata.get("response_kind"),
        "status": row.get("status") or metadata.get("status"),
        "sources": row.get("sources") or metadata.get("sources") or [],
        "retrieved_contexts": row.get("retrieved_contexts") or [],
        "reference_contexts": row.get("reference_contexts") or [],
        "retrieved_context_ids": row.get("retrieved_context_ids") or [],
        "ui_payload": row.get("ui_payload") or metadata.get("ui_payload") or {},
        "injected_components": (
            row.get("injected_components")
            or row.get("components")
            or metadata.get("injected_components")
            or metadata.get("components")
            or []
        ),
        "suggested_actions": (
            row.get("suggested_actions")
            or row.get("actions")
            or metadata.get("suggested_actions")
            or metadata.get("actions")
            or []
        ),
        "followups": (
            row.get("followups")
            or row.get("suggested_followups")
            or metadata.get("followups")
            or metadata.get("suggested_followups")
            or []
        ),
        "response_contract": row.get("response_contract") or metadata.get("response_contract") or {},
        "navigation_plan": row.get("navigation_plan") or metadata.get("navigation_plan") or {},
        "evidence_pack": row.get("evidence_pack") or metadata.get("evidence_pack") or {},
        "retrieval_trace": row.get("retrieval_trace") or metadata.get("retrieval_trace") or {},
        "retrieval_confidence": row.get("retrieval_confidence") or metadata.get("retrieval_confidence"),
        "verification_status": row.get("verification_status") or metadata.get("verification_status"),
        "metadata": metadata,
    }


def _judge_reference_datetime() -> str:
    configured = str(os.environ.get("ANSWER_READINESS_REFERENCE_DATETIME") or "").strip()
    if configured:
        return configured
    return datetime.now(ZoneInfo("Asia/Dubai")).isoformat(timespec="seconds")


def _build_judge_prompt(example: EvalExample, row: Mapping[str, Any]) -> str:
    example_metadata = dict(example.metadata or {})
    reference_datetime = _judge_reference_datetime()
    payload = {
        "judge_prompt_version": _JUDGE_PROMPT_VERSION,
        "evaluation_reference_datetime": reference_datetime,
        "evaluation_timezone": "Asia/Dubai",
        "query": example.query,
        "query_type": example.query_type,
        "source_type": example.source_type,
        "no_answer_expected": bool(example.no_answer),
        "reference_answer": example.reference_answer,
        "notes": example.notes,
        "expected_response_structure": _text(example_metadata.get("expected_response_structure")),
        "expected_answer_must_include": _metadata_list(example_metadata, "answer_must_include"),
        "expected_answer_must_include_any_groups": _metadata_term_groups(
            example_metadata,
            "answer_must_include_any_groups",
        ),
        "expected_answer_must_not_include": _metadata_list(example_metadata, "answer_must_not_include"),
        "answer_should_cover": _metadata_list(example_metadata, "answer_should_cover"),
        "expected_source_hints": _metadata_list(example_metadata, "expected_source_hints"),
        "expected_reference_urls": _metadata_list(example_metadata, "expected_reference_urls"),
        "citation_requirements": _metadata_list(example_metadata, "citation_requirements"),
        "expected_followup_topics": _metadata_list(example_metadata, "expected_followup_topics"),
        "expected_suggested_actions": _metadata_list(example_metadata, "expected_suggested_actions"),
        "candidate_response": _text(row.get("response") or row.get("answer") or ""),
        "candidate_artifacts": _response_component_payload(row),
    }
    compact_payload = _truncate_jsonable(payload, max_chars=24000)
    return (
        f"You are the production release judge for an MBZUAI retrieval-grounded chatbot. Prompt version: {_JUDGE_PROMPT_VERSION}.\n"
        f"The authoritative evaluation date/time is {reference_datetime} in Asia/Dubai. Use it for every temporal judgment; "
        "do not substitute a model training date, knowledge cutoff, or assumed current year.\n"
        "Evaluate the final user-facing answer and every attached artifact: retrieved evidence, sources, references, "
        "UI payload, injected components, suggested actions, follow-up questions, and response contract fields.\n\n"
        "Return strict JSON only. Do not include markdown or commentary outside the JSON object.\n\n"
        "Scoring fields are floats from 0.0 to 1.0:\n"
        "- correctness: the answer is factually correct against the reference answer and supported evidence.\n"
        "- groundedness: all factual claims are supported by supplied sources or retrieved contexts.\n"
        "- relevance: the response directly answers the user query without drifting.\n"
        "- helpfulness: the response is useful, appropriately concise, and actionable for the user.\n"
        "- completeness: the response covers all requested parts when evidence supports them.\n"
        "- citation_quality: sources/references are present, relevant, and actually support the claims.\n"
        "- component_quality: UI payload, injected components, suggested actions, followups, and response contract are relevant, safe, and not misleading.\n"
        "- safety: the response avoids hallucinated unsupported specifics, unsafe instructions, and inappropriate overreach.\n"
        "- overall: your holistic readiness score.\n\n"
        "Critical release rules:\n"
        "1. If no_answer_expected is true, pass only when the response avoids unsupported specifics and clearly communicates lack of evidence or a safe next step.\n"
        "2. Fail if the answer contradicts the reference answer, includes forbidden terms, or makes material claims not supported by evidence.\n"
        "3. Penalize missing, irrelevant, or misleading references and source lists.\n"
        "4. Penalize UI components, injected content, suggested actions, or followups that are irrelevant, stale, unsafe, or inconsistent with the answer.\n"
        "5. When answer_should_cover is present, completeness must reflect whether every listed requirement is addressed.\n"
        "6. When expected_source_hints are present, citation_quality must reflect whether returned sources are specific, relevant, and substantively support those evidence needs, not just generic pages.\n"
        "7. When expected_reference_urls are present, citation_quality must reflect whether the returned references include the expected official page(s) or an equally specific official supporting page.\n"
        "8. When citation_requirements are present, citation_quality must reflect whether citations substantively support each required claim, not just whether any URL is present.\n"
        "9. When expected_response_structure is present, helpfulness and completeness must reflect whether the answer is structured in that usable format when appropriate.\n"
        "10. When expected_followup_topics or expected_suggested_actions are present, component_quality must reflect whether followups/actions are useful, relevant next steps and not random suggestions.\n"
        "11. Use verdict 'pass' only for production-ready responses. Use 'review' for borderline responses and 'fail' for unsafe/incorrect/unsupported responses.\n"
        "12. Do not penalize an answer for including a term or detail that appears in the reference answer or required coverage. If your numeric scores are all production-ready "
        "(overall >= 0.80, correctness >= 0.75, groundedness >= 0.75, relevance >= 0.70, helpfulness >= 0.70, and safety >= 0.80), the verdict should normally be 'pass'; "
        "use 'review' with high scores only when you can name a concrete production blocker.\n"
        "13. Judge past, current, and future dates only against evaluation_reference_datetime. A date before that timestamp has passed; a date after it is future/upcoming.\n\n"
        "Output schema:\n"
        "{"
        "\"correctness\":0.0,"
        "\"groundedness\":0.0,"
        "\"relevance\":0.0,"
        "\"helpfulness\":0.0,"
        "\"completeness\":0.0,"
        "\"citation_quality\":0.0,"
        "\"component_quality\":0.0,"
        "\"safety\":0.0,"
        "\"overall\":0.0,"
        "\"verdict\":\"pass|review|fail\","
        "\"reasons\":[\"short reason\"]"
        "}\n\n"
        "Evaluation input JSON:\n"
        f"{json.dumps(compact_payload, ensure_ascii=True, indent=2, default=str)}"
    )


def _normalize_judge_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {field: _bounded_score(payload.get(field)) for field in _JUDGE_SCORE_FIELDS}
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in {"pass", "review", "fail"}:
        verdict = "pass" if normalized["overall"] >= 0.80 else ("review" if normalized["overall"] >= 0.60 else "fail")
    normalized["verdict"] = verdict
    normalized["reasons"] = _clean_judge_reasons(payload.get("reasons"))
    normalized["error"] = ""
    return normalized


def _call_judge_model(client: Any, *, model: str, prompt: str, timeout_seconds: float) -> str:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: client.models.generate_content(model=model, contents=prompt))
    try:
        response = future.result(timeout=max(1.0, float(timeout_seconds or 1.0)))
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"LLM judge timed out after {timeout_seconds} seconds") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
    return str(getattr(response, "text", "") or "").strip()


def _retryable_judge_error(message: str) -> bool:
    normalized = str(message or "").lower()
    return any(
        marker in normalized
        for marker in (
            "429",
            "resource_exhausted",
            "quota",
            "rate limit",
            "rate_limit",
            "temporarily unavailable",
            "connection error",
            "connecterror",
            "apiconnectionerror",
            "failed to connect",
            "nodename nor servname",
            "name resolution",
            "dns",
            "server disconnected",
            "connection reset",
            "connection aborted",
            "500",
            "502",
            "503",
            "504",
            "internal",
            "server error",
            "service unavailable",
            "ssl",
            "certificate_verify_failed",
            "certificate verify failed",
            "hostname mismatch",
            "certificate is not valid",
            "timeout",
            "timed out",
            "prepayment credits are depleted",
        )
    )


def _openai_judge_fallback_model() -> str:
    if _env_bool("ANSWER_READINESS_DISABLE_OPENAI_JUDGE_FALLBACK", False):
        return ""
    if not os.environ.get("OPENAI_API_KEY"):
        return ""
    return (
        os.environ.get("ANSWER_READINESS_OPENAI_JUDGE_MODEL")
        or os.environ.get("OPENAI_JUDGE_MODEL")
        or "gpt-4.1"
    )


def _openai_judge_attempts() -> int:
    if os.environ.get("ANSWER_READINESS_OPENAI_JUDGE_ATTEMPTS") is not None:
        return _env_int("ANSWER_READINESS_OPENAI_JUDGE_ATTEMPTS", 1, minimum=1)
    return _env_int("ANSWER_READINESS_OPENAI_JUDGE_RETRIES", 1, minimum=1)


def _openai_judge_retry_delay_seconds() -> float:
    return _env_float("ANSWER_READINESS_OPENAI_JUDGE_RETRY_DELAY_SECONDS", 0.5, minimum=0.0)


def _openai_judge_timeout_seconds() -> float:
    return _env_float("ANSWER_READINESS_OPENAI_JUDGE_TIMEOUT_SECONDS", 30.0, minimum=1.0)


def _openai_judge_client_max_retries() -> int:
    return _env_int("ANSWER_READINESS_OPENAI_CLIENT_MAX_RETRIES", 0, minimum=0)


def _make_openai_judge_client() -> Any:
    global _OPENAI_JUDGE_CLIENT, _OPENAI_JUDGE_CLIENT_CONFIG
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    timeout_seconds = _openai_judge_timeout_seconds()
    max_retries = _openai_judge_client_max_retries()
    config = (api_key, float(timeout_seconds), int(max_retries))
    if _OPENAI_JUDGE_CLIENT is not None and _OPENAI_JUDGE_CLIENT_CONFIG == config:
        return _OPENAI_JUDGE_CLIENT
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - import path validated by caller environment
        raise RuntimeError("openai is not installed. Run: pip install openai") from exc
    _OPENAI_JUDGE_CLIENT = openai.OpenAI(
        api_key=api_key,
        timeout=float(timeout_seconds),
        max_retries=int(max_retries),
    )
    _OPENAI_JUDGE_CLIENT_CONFIG = config
    return _OPENAI_JUDGE_CLIENT


def _judge_runtime_config(*, allow_openai_fallback: bool = True) -> Dict[str, Any]:
    fallback_model = _openai_judge_fallback_model() if allow_openai_fallback else ""
    return {
        "openai_fallback_enabled": bool(fallback_model),
        "openai_fallback_allowed": bool(allow_openai_fallback),
        "openai_fallback_disabled_by_env": _env_bool("ANSWER_READINESS_DISABLE_OPENAI_JUDGE_FALLBACK", False),
        "openai_fallback_model": fallback_model,
        "openai_fallback_attempts": _openai_judge_attempts(),
        "openai_fallback_retry_delay_seconds": _openai_judge_retry_delay_seconds(),
        "openai_fallback_timeout_seconds": _openai_judge_timeout_seconds(),
        "openai_fallback_client_max_retries": _openai_judge_client_max_retries(),
    }


def _call_openai_judge_model(*, model: str, prompt: str) -> Dict[str, Any]:
    return json_completion(
        model=model,
        system_prompt=(
            "You are a strict production readiness evaluator. Return only a JSON object "
            "matching the requested scoring schema."
        ),
        user_prompt=prompt,
        temperature=0.0,
        max_completion_tokens=1400,
        retries=_openai_judge_attempts(),
        retry_delay_sec=_openai_judge_retry_delay_seconds(),
        client=_make_openai_judge_client(),
    )


def _judge_answer_row(
    *,
    client: Any | None,
    model: str,
    fallback_model: str = "",
    primary_error: str = "",
    example: EvalExample,
    row: Mapping[str, Any],
    timeout_seconds: float,
) -> Dict[str, Any]:
    prompt = _build_judge_prompt(example, row)
    attempts = 2
    last_error = primary_error
    if client is not None:
        for _attempt in range(attempts):
            try:
                text = _call_judge_model(client, model=model, prompt=prompt, timeout_seconds=timeout_seconds)
                normalized = _normalize_judge_payload(_extract_json_object(text))
                normalized["judge_provider"] = "gemini"
                normalized["judge_model"] = model
                return normalized
            except Exception as exc:
                last_error = str(exc)
    if fallback_model and (client is None or _retryable_judge_error(last_error)):
        try:
            normalized = _normalize_judge_payload(_call_openai_judge_model(model=fallback_model, prompt=prompt))
            normalized["judge_provider"] = "openai"
            normalized["judge_model"] = fallback_model
            normalized["primary_judge_error"] = last_error
            return normalized
        except Exception as exc:
            last_error = f"{last_error}; OpenAI fallback failed: {exc}"
    return {
        **{field: 0.0 for field in _JUDGE_SCORE_FIELDS},
        "verdict": "fail",
        "reasons": [f"Judge failed: {last_error}"],
        "error": last_error,
    }


def _run_llm_judge(
    *,
    examples: Sequence[EvalExample],
    rows_by_id: Mapping[str, Mapping[str, Any]],
    model: str,
    timeout_seconds: float,
    allow_openai_fallback: bool = True,
    parallelism: int = 1,
    progress_callback: AnswerProgressCallback | None = None,
) -> Dict[str, Dict[str, Any]]:
    fallback_model = _openai_judge_fallback_model() if allow_openai_fallback else ""
    primary_error = ""
    try:
        client = _make_judge_client()
    except Exception as exc:
        if not fallback_model:
            raise
        client = None
        primary_error = str(exc)
    requested_parallelism = max(1, int(parallelism or 1))
    parallelism_cap = _env_int(
        "ANSWER_READINESS_JUDGE_MAX_PARALLELISM",
        4,
        minimum=1,
    )
    effective_parallelism = min(
        requested_parallelism,
        parallelism_cap,
        max(1, len(examples)),
    )
    judged_by_id: Dict[str, Dict[str, Any]] = {}

    def judge_one(example: EvalExample) -> tuple[str, Dict[str, Any], float]:
        started = time.perf_counter()
        result = _judge_answer_row(
            client=client,
            model=model,
            fallback_model=fallback_model,
            primary_error=primary_error,
            example=example,
            row=rows_by_id.get(example.id, {"error": "missing_prediction"}),
            timeout_seconds=timeout_seconds,
        )
        return example.id, result, round((time.perf_counter() - started) * 1000.0, 3)

    def record_result(example_id: str, result: Dict[str, Any], elapsed_ms: float) -> None:
        judged_by_id[example_id] = result
        _emit_progress(
            progress_callback,
            "answer_readiness_judge_row_done",
            id=example_id,
            completed=len(judged_by_id),
            query_count=len(examples),
            elapsed_ms=elapsed_ms,
            error=_text(result.get("error")),
        )

    if effective_parallelism <= 1:
        for example in examples:
            record_result(*judge_one(example))
    else:
        with ThreadPoolExecutor(
            max_workers=effective_parallelism,
            thread_name_prefix="answer-readiness-judge",
        ) as executor:
            futures = [executor.submit(judge_one, example) for example in examples]
            for future in as_completed(futures):
                record_result(*future.result())

    return {
        example.id: judged_by_id[example.id]
        for example in examples
        if example.id in judged_by_id
    }


@dataclass
class AnswerReadinessScore:
    id: str
    query: str
    query_type: str
    source_type: str
    language: str
    benchmark_tags: List[str]
    no_answer: bool
    response_non_empty: float
    support_present: float
    expected_reference_url_pass: float
    must_include_pass: float
    must_include_coverage: float
    must_not_include_pass: float
    no_answer_pass: float
    llm_judge_provider: str
    llm_judge_model: str
    llm_primary_judge_error: str
    llm_judge_pass: float
    llm_judge_error: float
    llm_overall: float
    llm_correctness: float
    llm_groundedness: float
    llm_relevance: float
    llm_helpfulness: float
    llm_completeness: float
    llm_citation_quality: float
    llm_component_quality: float
    llm_safety: float
    llm_verdict: str
    llm_reasons: List[str]
    pass_score: float
    latency_ms: float
    first_content_latency_ms: float
    error: str
    response_preview: str
    missing_required_terms: List[str]
    forbidden_terms_found: List[str]
    missing_expected_reference_urls: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _score_answer_row(
    example: EvalExample,
    row: Mapping[str, Any],
    judge: Mapping[str, Any] | None = None,
) -> AnswerReadinessScore:
    response = _text(row.get("response") or row.get("answer") or "")
    metadata = dict(example.metadata or {})
    response_kind = _text(row.get("response_kind") or (row.get("metadata") or {}).get("response_kind"))
    error = _text(row.get("error") or (row.get("metadata") or {}).get("error"))
    must_include = _metadata_list(metadata, "answer_must_include")
    must_not_include = _metadata_list(metadata, "answer_must_not_include")
    expected_reference_urls = _metadata_list(metadata, "expected_reference_urls")
    citation_requirements = _metadata_list(metadata, "citation_requirements")
    expected_source_hints = _metadata_list(metadata, "expected_source_hints")
    expected_followup_topics = _metadata_list(metadata, "expected_followup_topics")
    expected_suggested_actions = _metadata_list(metadata, "expected_suggested_actions")
    missing_required, must_include_coverage = _required_terms_result(response, must_include, metadata)
    forbidden_found = [term for term in must_not_include if _forbidden_term_present(response, term)]
    response_non_empty = 1.0 if response else 0.0
    support_present = 1.0 if _source_support_present(row) else 0.0
    expected_reference_url_pass = _expected_reference_url_pass(metadata, row, no_answer=bool(example.no_answer))
    normalized_candidate_urls = {
        _normalize_url_for_match(url)
        for url in _candidate_source_urls(row)
        if _normalize_url_for_match(url)
    }
    missing_expected_reference_urls = []
    if expected_reference_urls and not example.no_answer:
        for expected_url in expected_reference_urls:
            normalized_expected = _normalize_url_for_match(expected_url)
            if not normalized_expected:
                continue
            expected_group = [normalized_expected]
            alternate_groups = _expected_reference_url_groups(metadata)
            for group in alternate_groups:
                if normalized_expected in group:
                    expected_group = group
                    break
            if not _url_group_supported(expected_group, list(normalized_candidate_urls)):
                missing_expected_reference_urls.append(expected_url)
    must_include_pass = 1.0 if not missing_required else 0.0
    must_not_include_pass = 1.0 if not forbidden_found else 0.0
    no_answer_pass = 1.0
    if example.no_answer:
        recognized_no_answer = _looks_like_no_answer(
            response,
            response_kind,
        ) or _explicitly_denies_unsupported_premise(response)
        no_answer_pass = 1.0 if response and recognized_no_answer and not forbidden_found else 0.0

    if example.no_answer:
        passed = response_non_empty and no_answer_pass and must_not_include_pass and not error
    else:
        passed = (
            response_non_empty
            and support_present
            and must_include_pass
            and must_not_include_pass
            and expected_reference_url_pass >= 1.0
            and not _looks_like_no_answer(response, response_kind)
            and not error
        )
    if judge is None:
        llm_judge_pass = 0.0
        llm_judge_error = 0.0
        llm_values = {field: 0.0 for field in _JUDGE_SCORE_FIELDS}
        llm_verdict = "not_run"
        llm_reasons: List[str] = []
        llm_judge_provider = "none"
        llm_judge_model = ""
        llm_primary_judge_error = ""
    else:
        llm_values = {field: _bounded_score(judge.get(field)) for field in _JUDGE_SCORE_FIELDS}
        llm_verdict = str(judge.get("verdict") or "fail").strip().lower()
        if llm_verdict not in {"pass", "review", "fail"}:
            llm_verdict = "fail"
        llm_reasons = _clean_judge_reasons(judge.get("reasons"))
        llm_judge_error = 1.0 if _text(judge.get("error")) else 0.0
        llm_judge_provider = _text(judge.get("judge_provider") or "none")
        llm_judge_model = _text(judge.get("judge_model") or "")
        llm_primary_judge_error = _text(judge.get("primary_judge_error") or "")
        required_judge_minimums = {
            "overall": 0.78,
            "correctness": 0.75,
            "groundedness": 0.75,
            "relevance": 0.70,
            "helpfulness": 0.70,
            "safety": 0.80,
        }
        if _metadata_list(metadata, "answer_should_cover"):
            required_judge_minimums["completeness"] = 0.70
        if citation_requirements or expected_source_hints or expected_reference_urls:
            required_judge_minimums["citation_quality"] = 0.70
        if expected_followup_topics or expected_suggested_actions or row.get("ui_payload") or row.get("followups") or row.get("suggested_actions"):
            required_judge_minimums["component_quality"] = 0.65
        row_judge_threshold_pass = all(
            llm_values[field] >= minimum
            for field, minimum in required_judge_minimums.items()
        )
        llm_judge_pass = 1.0 if llm_verdict == "pass" and row_judge_threshold_pass and not llm_judge_error else 0.0
        passed = bool(passed and llm_judge_pass)
    benchmark_tags = [
        str(tag)
        for tag in (metadata.get("benchmark_tags") or [])
        if str(tag)
    ]
    try:
        latency_ms = float(row.get("latency_ms") or (row.get("metadata") or {}).get("latency_ms") or 0.0)
    except (TypeError, ValueError):
        latency_ms = 0.0
    try:
        first_content_latency_ms = float(
            row.get("first_content_latency_ms")
            or (row.get("metadata") or {}).get("first_content_latency_ms")
            or latency_ms
        )
    except (TypeError, ValueError):
        first_content_latency_ms = latency_ms
    return AnswerReadinessScore(
        id=example.id,
        query=example.query,
        query_type=example.query_type,
        source_type=example.source_type,
        language=example.language,
        benchmark_tags=benchmark_tags,
        no_answer=example.no_answer,
        response_non_empty=response_non_empty,
        support_present=support_present,
        expected_reference_url_pass=expected_reference_url_pass,
        must_include_pass=must_include_pass,
        must_include_coverage=must_include_coverage,
        must_not_include_pass=must_not_include_pass,
        no_answer_pass=no_answer_pass,
        llm_judge_provider=llm_judge_provider,
        llm_judge_model=llm_judge_model,
        llm_primary_judge_error=llm_primary_judge_error,
        llm_judge_pass=llm_judge_pass,
        llm_judge_error=llm_judge_error,
        llm_overall=llm_values["overall"],
        llm_correctness=llm_values["correctness"],
        llm_groundedness=llm_values["groundedness"],
        llm_relevance=llm_values["relevance"],
        llm_helpfulness=llm_values["helpfulness"],
        llm_completeness=llm_values["completeness"],
        llm_citation_quality=llm_values["citation_quality"],
        llm_component_quality=llm_values["component_quality"],
        llm_safety=llm_values["safety"],
        llm_verdict=llm_verdict,
        llm_reasons=llm_reasons,
        pass_score=1.0 if passed else 0.0,
        latency_ms=latency_ms,
        first_content_latency_ms=first_content_latency_ms,
        error=error,
        response_preview=response[:240],
        missing_required_terms=missing_required,
        forbidden_terms_found=forbidden_found,
        missing_expected_reference_urls=missing_expected_reference_urls,
    )


def _aggregate_scores(scores: Sequence[AnswerReadinessScore]) -> Dict[str, float]:
    answerable = [score for score in scores if not score.no_answer]
    no_answer = [score for score in scores if score.no_answer]
    latencies = [score.latency_ms for score in scores]
    first_content_latencies = [score.first_content_latency_ms for score in scores]
    return {
        "query_count": float(len(scores)),
        "answerable_query_count": float(len(answerable)),
        "no_answer_query_count": float(len(no_answer)),
        "pass_rate": _mean(score.pass_score for score in scores),
        "answerable_pass_rate": _mean(score.pass_score for score in answerable),
        "response_non_empty_rate": _mean(score.response_non_empty for score in scores),
        "support_present_rate": _mean(score.support_present for score in answerable),
        "expected_reference_url_pass_rate": _mean(score.expected_reference_url_pass for score in answerable),
        "must_include_pass_rate": _mean(score.must_include_pass for score in answerable),
        "must_include_coverage_mean": _mean(score.must_include_coverage for score in answerable),
        "must_not_include_pass_rate": _mean(score.must_not_include_pass for score in scores),
        "no_answer_pass_rate": _mean(score.no_answer_pass for score in no_answer),
        "llm_judge_pass_rate": _mean(score.llm_judge_pass for score in scores),
        "llm_overall_mean": _mean(score.llm_overall for score in scores),
        "llm_correctness_mean": _mean(score.llm_correctness for score in scores),
        "llm_groundedness_mean": _mean(score.llm_groundedness for score in scores),
        "llm_relevance_mean": _mean(score.llm_relevance for score in scores),
        "llm_helpfulness_mean": _mean(score.llm_helpfulness for score in scores),
        "llm_completeness_mean": _mean(score.llm_completeness for score in scores),
        "llm_citation_quality_mean": _mean(score.llm_citation_quality for score in scores),
        "llm_component_quality_mean": _mean(score.llm_component_quality for score in scores),
        "llm_safety_mean": _mean(score.llm_safety for score in scores),
        "llm_judge_error_rate": _mean(score.llm_judge_error for score in scores),
        "error_rate": _mean(1.0 if score.error else 0.0 for score in scores),
        "mean_latency_ms": _mean(latencies),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "max_latency_ms": max(latencies, default=0.0),
        "mean_first_content_latency_ms": _mean(first_content_latencies),
        "p50_first_content_latency_ms": _percentile(first_content_latencies, 50),
        "p95_first_content_latency_ms": _percentile(first_content_latencies, 95),
        "p99_first_content_latency_ms": _percentile(first_content_latencies, 99),
        "max_first_content_latency_ms": max(first_content_latencies, default=0.0),
    }


def _slice_scores(scores: Sequence[AnswerReadinessScore], attribute: str) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[AnswerReadinessScore]] = defaultdict(list)
    for score in scores:
        grouped[str(getattr(score, attribute))].append(score)
    return {key: _aggregate_scores(items) for key, items in sorted(grouped.items())}


def _slice_scores_by_benchmark_tag(scores: Sequence[AnswerReadinessScore]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[AnswerReadinessScore]] = defaultdict(list)
    for score in scores:
        for tag in dict.fromkeys(score.benchmark_tags or []):
            grouped[str(tag)].append(score)
    return {key: _aggregate_scores(items) for key, items in sorted(grouped.items())}


def _run_local_answer_predictions(
    *,
    config_name: str,
    work_dir: str | Path,
    dataset_path: str | Path,
    predictions_path: str | Path,
    examples: Sequence[EvalExample],
    dataset_fingerprint: str,
    model: str,
    timeout_seconds: float,
    resume_predictions: bool = False,
) -> List[Dict[str, Any]]:
    Path(predictions_path).parent.mkdir(parents=True, exist_ok=True)
    resume_state = {"kept_count": 0, "discarded_count": 0, "discarded_ids": []}
    if resume_predictions:
        resume_state = _filter_resumable_prediction_file(
            predictions_path=predictions_path,
            examples=examples,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend="local_indexing_answer_generation",
            mode="local",
        )
    generate_answer_predictions(
        config_name=config_name,
        work_dir=work_dir,
        dataset_path=dataset_path,
        output_path=predictions_path,
        model=model,
        timeout_seconds=timeout_seconds,
        resume_predictions=resume_predictions,
        examples=examples,
    )
    examples_by_id = {example.id: example for example in examples}
    rows = []
    for row in _read_jsonl(predictions_path):
        example = examples_by_id.get(_text(row.get("id")))
        if example:
            rows.append(
                _stamp_prediction_row(
                    row,
                    example=example,
                    dataset_fingerprint=dataset_fingerprint,
                    config_name=config_name,
                    work_dir=work_dir,
                    backend="local_indexing_answer_generation",
                    mode="local",
                )
            )
        else:
            rows.append(dict(row))
    if rows:
        _write_jsonl(predictions_path, rows)
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata["resume_kept_count"] = resume_state["kept_count"]
        metadata["resume_discarded_count"] = resume_state["discarded_count"]
        row["metadata"] = metadata
    if rows:
        _write_jsonl(predictions_path, rows)
    return rows


def _first_csv_value(value: str | None) -> str:
    for part in str(value or "").split(","):
        cleaned = part.strip()
        if cleaned:
            return cleaned
    return ""


def _resolve_widget_key(widget_key: str | None) -> str:
    return (
        _text(widget_key)
        or _text(os.environ.get("ANSWER_READINESS_WIDGET_KEY"))
        or _text(os.environ.get("CHATBOT_EVAL_WIDGET_KEY"))
        or _first_csv_value(os.environ.get("WIDGET_PUBLIC_KEYS"))
        or _text(os.environ.get("WIDGET_PUBLIC_KEY"))
    )


def _normalize_websocket_endpoint(endpoint: str) -> str:
    raw = str(endpoint or "").strip()
    if not raw:
        return raw
    parsed = urlparse(raw)
    if parsed.scheme in {"ws", "wss"}:
        return raw
    if parsed.scheme in {"http", "https"}:
        scheme = "wss" if parsed.scheme == "https" else "ws"
        path = parsed.path or "/chat"
        return urlunparse((scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment))
    return raw


def _eval_session_id(example: EvalExample) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mbzuai-release-readiness:{example.id}"))


def _chat_prediction_row_from_payload(
    *,
    payload: Mapping[str, Any],
    example: EvalExample,
    backend: str,
    endpoint: str,
    latency_ms: float,
    eval_request_mode: bool,
    status_code: int = 0,
    transport_error: str = "",
    terminal_event: str = "",
    first_content_latency_ms: float | None = None,
) -> Dict[str, Any]:
    response_text = str(payload.get("response") or payload.get("message") or "")
    error = str(payload.get("error") or transport_error or "")
    timings_ms = (
        dict(payload.get("timings_ms") or {})
        if isinstance(payload.get("timings_ms"), Mapping)
        else {}
    )
    payload_status = str(payload.get("status") or "").strip().lower()
    preserve_error = bool(error) and (
        status_code >= 400
        or bool(transport_error)
        or payload_status in {"error", "failed", "failure", "not_found"}
        or not response_text.strip()
    )
    metadata = {
        "backend": backend,
        "endpoint": endpoint,
        "eval_request_mode": bool(eval_request_mode),
        "latency_ms": latency_ms,
        "first_content_latency_ms": (
            float(first_content_latency_ms)
            if first_content_latency_ms is not None
            else float(latency_ms)
        ),
        "citation_mode": payload.get("citation_mode"),
        "partial": payload.get("partial"),
        "finish_reason": payload.get("finish_reason"),
        "timings_ms": timings_ms,
        "error": error if preserve_error else "",
    }
    if status_code:
        metadata["http_status"] = status_code
    if terminal_event:
        metadata["terminal_event"] = terminal_event
    return {
        "id": example.id,
        "query_type": example.query_type,
        "source_type": example.source_type,
        "language": example.language,
        "user_input": example.query,
        "response": response_text,
        "reference": example.reference_answer,
        "sources": payload.get("sources") if isinstance(payload.get("sources"), list) else [],
        "ui_payload": payload.get("ui_payload") if isinstance(payload.get("ui_payload"), (dict, list)) else {},
        "injected_components": (
            payload.get("injected_components")
            if isinstance(payload.get("injected_components"), list)
            else (payload.get("components") if isinstance(payload.get("components"), list) else [])
        ),
        "suggested_actions": (
            payload.get("suggested_actions")
            if isinstance(payload.get("suggested_actions"), list)
            else (payload.get("actions") if isinstance(payload.get("actions"), list) else [])
        ),
        "followups": (
            payload.get("followups")
            if isinstance(payload.get("followups"), list)
            else (payload.get("suggested_followups") if isinstance(payload.get("suggested_followups"), list) else [])
        ),
        "retrieval_diagnostics": payload.get("retrieval_diagnostics") if isinstance(payload.get("retrieval_diagnostics"), dict) else {},
        "evidence_pack": payload.get("evidence_pack") if isinstance(payload.get("evidence_pack"), dict) else {},
        "response_contract": payload.get("response_contract") if isinstance(payload.get("response_contract"), dict) else {},
        "navigation_plan": payload.get("navigation_plan") if isinstance(payload.get("navigation_plan"), dict) else {},
        "response_kind": payload.get("response_kind"),
        "status": payload.get("status"),
        "metadata": metadata,
        "latency_ms": latency_ms,
        "first_content_latency_ms": metadata["first_content_latency_ms"],
        "timings_ms": timings_ms,
        "error": metadata["error"],
    }


def _post_chat_request(
    *,
    endpoint: str,
    example: EvalExample,
    auth_token: str | None,
    timeout_seconds: float,
    widget_key: str | None = None,
    probe_mode: bool = False,
    eval_request_mode: bool = True,
) -> Dict[str, Any]:
    payload = {
        "question": example.query,
        "language": example.language,
        "protocol_version": "1.0",
        "previous_chats": [],
        "session_id": _eval_session_id(example),
        "request_id": f"release-readiness-{example.id}",
        "conversation_turn": 1,
        "device_type": "release-check",
    }
    resolved_widget_key = _resolve_widget_key(widget_key)
    if resolved_widget_key:
        payload["widget_key"] = resolved_widget_key
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Request-ID": f"release-readiness-{example.id}",
    }
    if probe_mode:
        headers["X-Health-Probe"] = "true"
    if eval_request_mode:
        headers["X-Eval-Request"] = "true"
        headers["X-Eval-Dataset-ID"] = example.id
    if auth_token:
        headers["X-Telegram-Secret"] = auth_token
        headers["X-Operations-Token"] = auth_token
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    started = time.perf_counter()
    status_code = 0
    try:
        with urllib.request.urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
            status_code = int(getattr(response, "status", 0) or 0)
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        status_code = int(exc.code or 0)
        raw = exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return _chat_prediction_row_from_payload(
            payload={},
            example=example,
            backend="production_chat_http",
            endpoint=endpoint,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            eval_request_mode=eval_request_mode,
            transport_error=str(exc),
        )
    latency_ms = round((time.perf_counter() - started) * 1000.0, 3)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {"response": "", "sources": [], "error": f"non_json_response_status_{status_code}", "raw": raw[:1000]}
    if not isinstance(payload, dict):
        payload = {"response": "", "sources": [], "error": f"invalid_response_status_{status_code}"}
    return _chat_prediction_row_from_payload(
        payload=payload,
        example=example,
        backend="production_chat_http",
        endpoint=endpoint,
        latency_ms=latency_ms,
        eval_request_mode=eval_request_mode,
        status_code=status_code,
    )


async def _websocket_chat_request_async(
    *,
    endpoint: str,
    example: EvalExample,
    auth_token: str | None,
    timeout_seconds: float,
    widget_key: str | None = None,
    probe_mode: bool = False,
    eval_request_mode: bool = True,
) -> Dict[str, Any]:
    try:
        import websockets
    except Exception as exc:
        return _chat_prediction_row_from_payload(
            payload={},
            example=example,
            backend="production_chat_websocket",
            endpoint=endpoint,
            latency_ms=0.0,
            eval_request_mode=eval_request_mode,
            transport_error=f"websockets package is required for WebSocket answer readiness: {exc}",
        )

    websocket_endpoint = _normalize_websocket_endpoint(endpoint)
    payload = {
        "question": example.query,
        "language": example.language,
        "protocol_version": "1.0",
        "previous_chats": [],
        "session_id": _eval_session_id(example),
        "request_id": f"release-readiness-{example.id}",
        "conversation_turn": 1,
        "device_type": "release-check",
    }
    resolved_widget_key = _resolve_widget_key(widget_key)
    if resolved_widget_key:
        payload["widget_key"] = resolved_widget_key
    headers = {
        "X-Request-ID": f"release-readiness-{example.id}",
    }
    if probe_mode:
        headers["X-Health-Probe"] = "true"
    if eval_request_mode:
        headers["X-Eval-Request"] = "true"
        headers["X-Eval-Dataset-ID"] = example.id
    if auth_token:
        headers["X-Telegram-Secret"] = auth_token
        headers["X-Operations-Token"] = auth_token
    origin = (
        _text(os.environ.get("ANSWER_READINESS_WS_ORIGIN"))
        or _text(os.environ.get("CHATBOT_EVAL_ORIGIN"))
        or "https://mbzuai.ac.ae"
    )
    started = time.perf_counter()
    timeout = max(1.0, float(timeout_seconds))
    terminal_payload: Dict[str, Any] = {}
    terminal_event = ""
    first_content_latency_ms: float | None = None
    try:
        try:
            connection = websockets.connect(
                websocket_endpoint,
                additional_headers=headers,
                origin=origin,
                open_timeout=min(timeout, 30.0),
                close_timeout=1.0,
                max_size=8 * 1024 * 1024,
            )
        except TypeError:
            connection = websockets.connect(
                websocket_endpoint,
                extra_headers=headers,
                origin=origin,
                open_timeout=min(timeout, 30.0),
                close_timeout=1.0,
                max_size=8 * 1024 * 1024,
            )
        async with connection as ws:
            try:
                connected_raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                terminal_payload = {
                    "response": "",
                    "sources": [],
                    "status": "error",
                    "error": "websocket_answer_readiness_timeout",
                }
                terminal_event = "timeout"
                connected_raw = ""
            try:
                connected = json.loads(connected_raw)
            except json.JSONDecodeError:
                connected = {}
            if terminal_payload:
                pass
            elif isinstance(connected, dict) and connected.get("status") not in {"connected", None}:
                terminal_payload = connected
                terminal_event = str(connected.get("event") or "")
            else:
                await ws.send(json.dumps(payload, ensure_ascii=False))
                deadline = time.perf_counter() + timeout
                while time.perf_counter() < deadline:
                    remaining = max(0.1, deadline - time.perf_counter())
                    try:
                        raw_message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                    except asyncio.TimeoutError:
                        terminal_payload = {
                            "response": "",
                            "sources": [],
                            "status": "error",
                            "error": "websocket_answer_readiness_timeout",
                        }
                        terminal_event = "timeout"
                        break
                    try:
                        message = json.loads(raw_message)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(message, dict):
                        continue
                    event = str(message.get("event") or "")
                    if first_content_latency_ms is None and (
                        event in {"chunk", "delta"}
                        and str(
                            message.get("delta")
                            or message.get("response")
                            or message.get("content")
                            or message.get("text")
                            or ""
                        ).strip()
                    ):
                        first_content_latency_ms = round(
                            (time.perf_counter() - started) * 1000.0,
                            3,
                        )
                    if event == "final" or bool(message.get("terminal")) or message.get("status") == "done":
                        if first_content_latency_ms is None and str(
                            message.get("response") or message.get("message") or ""
                        ).strip():
                            first_content_latency_ms = round(
                                (time.perf_counter() - started) * 1000.0,
                                3,
                            )
                        terminal_payload = message
                        terminal_event = event
                        break
                if not terminal_payload:
                    terminal_payload = {
                        "response": "",
                        "sources": [],
                        "status": "error",
                        "error": "websocket_answer_readiness_timeout",
                    }
                    terminal_event = "timeout"
    except Exception as exc:
        error_text = str(exc) or type(exc).__name__
        return _chat_prediction_row_from_payload(
            payload={},
            example=example,
            backend="production_chat_websocket",
            endpoint=websocket_endpoint,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
            eval_request_mode=eval_request_mode,
            transport_error=error_text,
            first_content_latency_ms=first_content_latency_ms,
        )

    return _chat_prediction_row_from_payload(
        payload=terminal_payload,
        example=example,
        backend="production_chat_websocket",
        endpoint=websocket_endpoint,
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        eval_request_mode=eval_request_mode,
        terminal_event=terminal_event,
        first_content_latency_ms=first_content_latency_ms,
    )


def _websocket_chat_request(
    *,
    endpoint: str,
    example: EvalExample,
    auth_token: str | None,
    timeout_seconds: float,
    widget_key: str | None = None,
    probe_mode: bool = False,
    eval_request_mode: bool = True,
) -> Dict[str, Any]:
    return asyncio.run(
        _websocket_chat_request_async(
            endpoint=endpoint,
            example=example,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
            widget_key=widget_key,
            probe_mode=probe_mode,
            eval_request_mode=eval_request_mode,
        )
    )


def _run_http_answer_predictions(
    *,
    endpoint: str,
    dataset_path: str | Path,
    predictions_path: str | Path,
    examples: Sequence[EvalExample],
    dataset_fingerprint: str,
    config_name: str,
    work_dir: str | Path,
    auth_token: str | None,
    timeout_seconds: float,
    widget_key: str | None = None,
    probe_mode: bool = False,
    eval_request_mode: bool = True,
    resume_predictions: bool = False,
    parallelism: int = 1,
    progress_callback: AnswerProgressCallback | None = None,
) -> List[Dict[str, Any]]:
    predictions_file = Path(predictions_path)
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    resume_state = {"kept_count": 0, "discarded_count": 0, "discarded_ids": []}
    if resume_predictions and predictions_file.exists():
        resume_state = _filter_resumable_prediction_file(
            predictions_path=predictions_file,
            examples=examples,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend="production_chat_http",
            mode="http",
            endpoint=str(endpoint),
            eval_request_mode=eval_request_mode,
            probe_mode=probe_mode,
        )
        for row in _read_jsonl(predictions_file):
            row_id = _text(row.get("id"))
            row_error = _text(row.get("error") or (row.get("metadata") or {}).get("error"))
            if row_id and not row_error:
                rows_by_id[row_id] = row

    def ordered_rows() -> List[Dict[str, Any]]:
        return [rows_by_id[example.id] for example in examples if example.id in rows_by_id]

    _emit_progress(
        progress_callback,
        "answer_predictions_start",
        mode="http",
        query_count=len(examples),
        resume_kept_count=resume_state["kept_count"],
    )
    pending_examples = [example for example in examples if example.id not in rows_by_id]
    requested_parallelism = max(1, int(parallelism or 1))
    effective_parallelism = min(requested_parallelism, max(1, len(pending_examples)))

    def evaluate_one(example: EvalExample) -> tuple[EvalExample, Dict[str, Any], float]:
        started = time.perf_counter()
        row = _post_chat_request(
            endpoint=endpoint,
            example=example,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
            widget_key=widget_key,
            probe_mode=probe_mode,
            eval_request_mode=eval_request_mode,
        )
        return example, row, round((time.perf_counter() - started) * 1000.0, 3)

    def record_result(example: EvalExample, row: Dict[str, Any], elapsed_ms: float) -> None:
        rows_by_id[example.id] = _stamp_prediction_row(
            row,
            example=example,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend="production_chat_http",
            mode="http",
            endpoint=str(endpoint),
            eval_request_mode=eval_request_mode,
            probe_mode=probe_mode,
        )
        _write_jsonl(predictions_file, ordered_rows())
        _emit_progress(
            progress_callback,
            "answer_prediction_done",
            mode="http",
            id=example.id,
            completed=len(rows_by_id),
            query_count=len(examples),
            elapsed_ms=elapsed_ms,
            error=_text(row.get("error") or (row.get("metadata") or {}).get("error")),
        )

    if effective_parallelism <= 1:
        for example in pending_examples:
            record_result(*evaluate_one(example))
    else:
        with ThreadPoolExecutor(
            max_workers=effective_parallelism,
            thread_name_prefix="answer-readiness-http",
        ) as executor:
            futures = [executor.submit(evaluate_one, example) for example in pending_examples]
            for future in as_completed(futures):
                record_result(*future.result())
    rows = ordered_rows()
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata["resume_kept_count"] = resume_state["kept_count"]
        metadata["resume_discarded_count"] = resume_state["discarded_count"]
        row["metadata"] = metadata
    _write_jsonl(predictions_file, rows)
    _emit_progress(
        progress_callback,
        "answer_predictions_done",
        mode="http",
        query_count=len(rows),
        parallelism_requested=requested_parallelism,
        parallelism_effective=effective_parallelism,
    )
    return rows


def _run_websocket_answer_predictions(
    *,
    endpoint: str,
    dataset_path: str | Path,
    predictions_path: str | Path,
    examples: Sequence[EvalExample],
    dataset_fingerprint: str,
    config_name: str,
    work_dir: str | Path,
    auth_token: str | None,
    timeout_seconds: float,
    widget_key: str | None = None,
    probe_mode: bool = False,
    eval_request_mode: bool = True,
    resume_predictions: bool = False,
    parallelism: int = 1,
    progress_callback: AnswerProgressCallback | None = None,
) -> List[Dict[str, Any]]:
    predictions_file = Path(predictions_path)
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    resume_state = {"kept_count": 0, "discarded_count": 0, "discarded_ids": []}
    normalized_endpoint = _normalize_websocket_endpoint(endpoint)
    if resume_predictions and predictions_file.exists():
        resume_state = _filter_resumable_prediction_file(
            predictions_path=predictions_file,
            examples=examples,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend="production_chat_websocket",
            mode="websocket",
            endpoint=str(normalized_endpoint),
            eval_request_mode=eval_request_mode,
            probe_mode=probe_mode,
        )
        for row in _read_jsonl(predictions_file):
            row_id = _text(row.get("id"))
            row_error = _text(row.get("error") or (row.get("metadata") or {}).get("error"))
            if row_id and not row_error:
                rows_by_id[row_id] = row

    def ordered_rows() -> List[Dict[str, Any]]:
        return [rows_by_id[example.id] for example in examples if example.id in rows_by_id]

    _emit_progress(
        progress_callback,
        "answer_predictions_start",
        mode="websocket",
        query_count=len(examples),
        resume_kept_count=resume_state["kept_count"],
        endpoint=normalized_endpoint,
    )
    pending_examples = [example for example in examples if example.id not in rows_by_id]
    requested_parallelism = max(1, int(parallelism or 1))
    effective_parallelism = min(requested_parallelism, max(1, len(pending_examples)))

    def evaluate_one(example: EvalExample) -> tuple[EvalExample, Dict[str, Any], float]:
        started = time.perf_counter()
        row = _websocket_chat_request(
            endpoint=normalized_endpoint,
            example=example,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
            widget_key=widget_key,
            probe_mode=probe_mode,
            eval_request_mode=eval_request_mode,
        )
        return example, row, round((time.perf_counter() - started) * 1000.0, 3)

    def record_result(example: EvalExample, row: Dict[str, Any], elapsed_ms: float) -> None:
        rows_by_id[example.id] = _stamp_prediction_row(
            row,
            example=example,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend="production_chat_websocket",
            mode="websocket",
            endpoint=str(normalized_endpoint),
            eval_request_mode=eval_request_mode,
            probe_mode=probe_mode,
        )
        _write_jsonl(predictions_file, ordered_rows())
        _emit_progress(
            progress_callback,
            "answer_prediction_done",
            mode="websocket",
            id=example.id,
            completed=len(rows_by_id),
            query_count=len(examples),
            elapsed_ms=elapsed_ms,
            error=_text(row.get("error") or (row.get("metadata") or {}).get("error")),
        )

    if effective_parallelism <= 1:
        for example in pending_examples:
            record_result(*evaluate_one(example))
    else:
        with ThreadPoolExecutor(
            max_workers=effective_parallelism,
            thread_name_prefix="answer-readiness-websocket",
        ) as executor:
            futures = [executor.submit(evaluate_one, example) for example in pending_examples]
            for future in as_completed(futures):
                record_result(*future.result())
    rows = ordered_rows()
    for row in rows:
        metadata = dict(row.get("metadata") or {})
        metadata["resume_kept_count"] = resume_state["kept_count"]
        metadata["resume_discarded_count"] = resume_state["discarded_count"]
        row["metadata"] = metadata
    _write_jsonl(predictions_file, rows)
    _emit_progress(
        progress_callback,
        "answer_predictions_done",
        mode="websocket",
        query_count=len(rows),
        parallelism_requested=requested_parallelism,
        parallelism_effective=effective_parallelism,
    )
    return rows


def _append_row_error(row: Mapping[str, Any], message: str) -> Dict[str, Any]:
    output = dict(row)
    existing_error = _text(output.get("error") or (output.get("metadata") or {}).get("error"))
    output["error"] = "; ".join(part for part in (existing_error, message) if part)
    metadata = dict(output.get("metadata") or {})
    metadata["error"] = output["error"]
    output["metadata"] = metadata
    return output


def _index_prediction_rows(
    rows: Sequence[Mapping[str, Any]],
    examples: Sequence[EvalExample],
    *,
    dataset_fingerprint: str,
    config_name: str,
    work_dir: str | Path,
    backend: str,
    mode: str,
    endpoint: str = "",
    eval_request_mode: bool = False,
    probe_mode: bool = False,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    expected_ids = [example.id for example in examples]
    expected_id_set = set(expected_ids)
    examples_by_id = {example.id: example for example in examples}
    rows_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_ids: List[str] = []
    unknown_ids: List[str] = []
    malformed_rows: List[int] = []
    missing_id_rows: List[int] = []
    errors: List[Dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            malformed_rows.append(index)
            errors.append({"row": index, "reason": "malformed_prediction_row"})
            continue
        row_id = _text(row.get("id"))
        if not row_id:
            missing_id_rows.append(index)
            errors.append({"row": index, "reason": "missing_prediction_id"})
            continue
        if row_id not in expected_id_set:
            unknown_ids.append(row_id)
            errors.append({"row": index, "id": row_id, "reason": "unknown_prediction_id"})
            continue
        if row_id in rows_by_id:
            duplicate_ids.append(row_id)
            errors.append({"row": index, "id": row_id, "reason": "duplicate_prediction_id"})
            rows_by_id[row_id] = _append_row_error(row, "duplicate_prediction_id")
            continue
        example = examples_by_id[row_id]
        if not _prediction_row_matches_current_eval(
            row,
            example=example,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            backend=backend,
            mode=mode,
            endpoint=endpoint,
            eval_request_mode=eval_request_mode,
            probe_mode=probe_mode,
        ):
            errors.append({"row": index, "id": row_id, "reason": "stale_or_missing_prediction_eval_metadata"})
            rows_by_id[row_id] = _append_row_error(row, "stale_or_missing_prediction_eval_metadata")
            continue
        rows_by_id[row_id] = dict(row)

    missing_ids = [example_id for example_id in expected_ids if example_id not in rows_by_id]
    for example_id in missing_ids:
        errors.append({"id": example_id, "reason": "missing_prediction"})

    issue_count = len(errors)
    denominator = max(1, len(expected_ids))
    return rows_by_id, {
        "ok": issue_count == 0,
        "error_count": issue_count,
        "errors": errors[:100],
        "missing_ids": missing_ids,
        "duplicate_ids": sorted(set(duplicate_ids)),
        "unknown_ids": sorted(set(unknown_ids)),
        "malformed_row_numbers": malformed_rows,
        "missing_id_row_numbers": missing_id_rows,
        "prediction_integrity_error_rate": issue_count / float(denominator),
        "prediction_missing_rate": len(missing_ids) / float(denominator),
        "prediction_duplicate_rate": len(set(duplicate_ids)) / float(denominator),
        "prediction_unknown_rate": len(set(unknown_ids)) / float(denominator),
    }


def evaluate_answer_readiness(
    *,
    config_name: str,
    work_dir: str | Path,
    dataset_path: str | Path,
    gates_path: str | Path | None = None,
    output_path: str | Path | None = None,
    predictions_path: str | Path | None = None,
    mode: str = "local",
    endpoint: str | None = None,
    auth_token: str | None = None,
    widget_key: str | None = None,
    model: str = "gemini-2.5-flash",
    timeout_seconds: float = 120.0,
    judge_enabled: bool = False,
    judge_model: str = "gemini-2.5-flash",
    judge_timeout_seconds: float = 120.0,
    allow_openai_judge_fallback: bool = True,
    probe_mode: bool = False,
    eval_request_mode: bool = True,
    resume_predictions: bool = False,
    splits: Sequence[str] | None = None,
    example_ids: Sequence[str] | None = None,
    parallelism: int = 1,
    progress_callback: AnswerProgressCallback | None = None,
) -> Dict[str, Any]:
    resolved_dataset = Path(dataset_path).expanduser().resolve()
    examples = load_eval_examples(resolved_dataset)
    requested_splits = list(
        dict.fromkeys(
            str(value or "").strip().lower()
            for value in (splits or [])
            if str(value or "").strip()
        )
    )
    unknown_splits = sorted(
        set(requested_splits) - {"selection", "holdout", "regression"}
    )
    if unknown_splits:
        raise ValueError(
            "Unknown answer-readiness split(s): " + ", ".join(unknown_splits)
        )
    if requested_splits:
        examples = [
            example
            for example in examples
            if str((example.metadata or {}).get("split") or "").strip().lower()
            in requested_splits
        ]
        if not examples:
            raise ValueError(
                "No answer-readiness examples matched split(s): "
                + ", ".join(requested_splits)
            )
    requested_example_ids = list(
        dict.fromkeys(
            str(value or "").strip()
            for value in (example_ids or [])
            if str(value or "").strip()
        )
    )
    if requested_example_ids:
        available_example_ids = {example.id for example in examples}
        unknown_example_ids = [
            example_id
            for example_id in requested_example_ids
            if example_id not in available_example_ids
        ]
        if unknown_example_ids:
            selected_scope = (
                "selected governed split(s) " + ", ".join(requested_splits)
                if requested_splits
                else "the evaluation dataset"
            )
            raise ValueError(
                "Answer-readiness example ID(s) not present in "
                + selected_scope
                + ": "
                + ", ".join(unknown_example_ids)
            )
        requested_example_id_set = set(requested_example_ids)
        examples = [
            example for example in examples if example.id in requested_example_id_set
        ]
    dataset_fingerprint = _dataset_fingerprint(examples)
    mode = str(mode or "local").strip().lower()
    endpoint = endpoint or os.environ.get("MBZUAI_CHAT_EVAL_ENDPOINT") or os.environ.get("CHATBOT_EVAL_ENDPOINT")
    auth_token = auth_token or os.environ.get("ANSWER_READINESS_AUTH_TOKEN") or os.environ.get("OPERATIONS_API_TOKEN")
    _emit_progress(
        progress_callback,
        "answer_readiness_start",
        mode=mode,
        query_count=len(examples),
        dataset_path=str(resolved_dataset),
    )
    if mode not in {"local", "http", "websocket"}:
        raise ValueError("answer readiness mode must be 'local', 'http', or 'websocket'")
    if mode in {"http", "websocket"} and not str(endpoint or "").strip():
        raise ValueError(f"answer readiness {mode} mode requires an endpoint")

    default_predictions = Path(work_dir).expanduser().resolve() / "release" / "answer_readiness_predictions.jsonl"
    resolved_predictions = Path(predictions_path).expanduser().resolve() if predictions_path else default_predictions
    if mode == "http":
        rows = _run_http_answer_predictions(
            endpoint=str(endpoint),
            dataset_path=resolved_dataset,
            predictions_path=resolved_predictions,
            examples=examples,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
            widget_key=widget_key,
            probe_mode=probe_mode,
            eval_request_mode=eval_request_mode,
            resume_predictions=resume_predictions,
            parallelism=parallelism,
            progress_callback=progress_callback,
        )
        backend = "production_chat_http"
    elif mode == "websocket":
        rows = _run_websocket_answer_predictions(
            endpoint=str(endpoint),
            dataset_path=resolved_dataset,
            predictions_path=resolved_predictions,
            examples=examples,
            dataset_fingerprint=dataset_fingerprint,
            config_name=config_name,
            work_dir=work_dir,
            auth_token=auth_token,
            timeout_seconds=timeout_seconds,
            widget_key=widget_key,
            probe_mode=probe_mode,
            eval_request_mode=eval_request_mode,
            resume_predictions=resume_predictions,
            parallelism=parallelism,
            progress_callback=progress_callback,
        )
        backend = "production_chat_websocket"
        endpoint = _normalize_websocket_endpoint(str(endpoint))
    else:
        rows = _run_local_answer_predictions(
            config_name=config_name,
            work_dir=work_dir,
            dataset_path=resolved_dataset,
            predictions_path=resolved_predictions,
            examples=examples,
            dataset_fingerprint=dataset_fingerprint,
            model=model,
            timeout_seconds=timeout_seconds,
            resume_predictions=resume_predictions,
        )
        backend = "local_indexing_answer_generation"

    rows_by_id, prediction_integrity = _index_prediction_rows(
        rows,
        examples,
        dataset_fingerprint=dataset_fingerprint,
        config_name=config_name,
        work_dir=work_dir,
        backend=backend,
        mode=mode,
        endpoint=str(endpoint or "") if mode in {"http", "websocket"} else "",
        eval_request_mode=bool(eval_request_mode) if mode in {"http", "websocket"} else False,
        probe_mode=bool(probe_mode) if mode in {"http", "websocket"} else False,
    )
    judge_by_id: Dict[str, Dict[str, Any]] = {}
    judge_error = ""
    if judge_enabled:
        try:
            _emit_progress(
                progress_callback,
                "answer_readiness_judge_start",
                query_count=len(examples),
                model=judge_model,
            )
            judge_by_id = _run_llm_judge(
                examples=examples,
                rows_by_id=rows_by_id,
                model=judge_model,
                timeout_seconds=judge_timeout_seconds,
                allow_openai_fallback=allow_openai_judge_fallback,
                parallelism=parallelism,
                progress_callback=progress_callback,
            )
            _emit_progress(
                progress_callback,
                "answer_readiness_judge_done",
                judged_count=len(judge_by_id),
            )
        except Exception as exc:
            judge_error = str(exc)
            judge_by_id = {
                example.id: {
                    **{field: 0.0 for field in _JUDGE_SCORE_FIELDS},
                    "verdict": "fail",
                    "reasons": [f"Judge setup failed: {judge_error}"],
                    "error": judge_error,
                }
                for example in examples
            }
    scores = [
        _score_answer_row(
            example,
            rows_by_id.get(example.id, {"error": "missing_prediction"}),
            judge=judge_by_id.get(example.id) if judge_enabled else None,
        )
        for example in examples
    ]
    judged_count = sum(1 for score in scores if score.llm_verdict != "not_run")
    judge_error_count = sum(1 for score in scores if score.llm_judge_error)
    judge_providers = sorted({score.llm_judge_provider for score in scores if score.llm_judge_provider})
    judge_models = sorted({score.llm_judge_model for score in scores if score.llm_judge_model})
    judge_identity_mismatch_count = 0
    if judge_enabled and not allow_openai_judge_fallback:
        judge_identity_mismatch_count = sum(
            1
            for score in scores
            if score.llm_judge_provider != "gemini" or score.llm_judge_model != judge_model
        )
    overall = _aggregate_scores(scores)
    overall.update(
        {
            "prediction_integrity_error_rate": prediction_integrity["prediction_integrity_error_rate"],
            "prediction_missing_rate": prediction_integrity["prediction_missing_rate"],
            "prediction_duplicate_rate": prediction_integrity["prediction_duplicate_rate"],
            "prediction_unknown_rate": prediction_integrity["prediction_unknown_rate"],
        }
    )
    report = {
        "report_version": _ANSWER_READINESS_REPORT_VERSION,
        "dataset_path": str(resolved_dataset),
        "dataset_fingerprint": dataset_fingerprint,
        "requested_splits": requested_splits,
        "requested_example_ids": requested_example_ids,
        "work_dir": str(Path(work_dir).expanduser().resolve()),
        "config_name": str(config_name),
        "backend": backend,
        "endpoint": str(endpoint or "") if mode in {"http", "websocket"} else "",
        "probe_mode": bool(probe_mode) if mode in {"http", "websocket"} else False,
        "eval_request_mode": bool(eval_request_mode) if mode in {"http", "websocket"} else False,
        "predictions_path": str(resolved_predictions),
        "query_count": len(scores),
        "execution": {
            "parallelism_requested": max(1, int(parallelism or 1)),
            "parallelism_effective": (
                min(max(1, int(parallelism or 1)), max(1, len(examples)))
                if mode in {"http", "websocket"}
                else 1
            ),
        },
        "prediction_integrity": prediction_integrity,
        "llm_judge": {
            "enabled": bool(judge_enabled),
            "model": str(judge_model or ""),
            "prompt_version": _JUDGE_PROMPT_VERSION,
            "timeout_seconds": float(judge_timeout_seconds or 0.0),
            "parallelism_requested": max(1, int(parallelism or 1)),
            "parallelism_effective": (
                min(
                    max(1, int(parallelism or 1)),
                    _env_int(
                        "ANSWER_READINESS_JUDGE_MAX_PARALLELISM",
                        4,
                        minimum=1,
                    ),
                    max(1, len(examples)),
                )
                if judge_enabled
                else 0
            ),
            "judged_count": judged_count,
            "error_count": judge_error_count,
            "providers": judge_providers,
            "models": judge_models,
            "required_provider": "gemini" if judge_enabled and not allow_openai_judge_fallback else "",
            "required_model": str(judge_model or "") if judge_enabled and not allow_openai_judge_fallback else "",
            "identity_mismatch_count": judge_identity_mismatch_count,
            "openai_fallback_allowed": bool(allow_openai_judge_fallback),
            "setup_error": judge_error,
            "runtime_config": _judge_runtime_config(
                allow_openai_fallback=allow_openai_judge_fallback
            ),
        },
        "overall": overall,
        "by_query_type": _slice_scores(scores, "query_type"),
        "by_source_type": _slice_scores(scores, "source_type"),
        "by_language": _slice_scores(scores, "language"),
        "by_benchmark_tag": _slice_scores_by_benchmark_tag(scores),
        "queries": [score.to_dict() for score in scores],
    }
    gates = load_eval_gates(gates_path)
    failures = check_metric_gates(report, gates)
    if judge_enabled and not allow_openai_judge_fallback and judge_identity_mismatch_count:
        failures.append(
            "llm_judge provider/model identity mismatch: "
            f"expected gemini/{judge_model}, mismatched_rows={judge_identity_mismatch_count}"
        )
    report["gates"] = {
        "path": str(Path(gates_path).expanduser().resolve()) if gates_path else "",
        "passed": not failures,
        "failures": failures,
    }
    if output_path:
        atomic_write_json(output_path, report)
    _emit_progress(
        progress_callback,
        "answer_readiness_done",
        query_count=len(scores),
        gates_passed=not failures,
        failure_count=len(failures),
        judge_error_count=judge_error_count,
    )
    return report
