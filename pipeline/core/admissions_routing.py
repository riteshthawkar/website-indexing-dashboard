"""Canonical admissions workflow routing shared by retrieval and navigation."""

from __future__ import annotations

import re
from typing import Any


_WORKFLOW_RE = re.compile(
    r"\b(?:apply|applying|application process|application steps?|submit(?:ting)? (?:an? )?application|"
    r"start(?:ing)? (?:an? )?(?:\w+\s+){0,4}application|"
    r"(?:undergraduate|graduate|phd|doctoral|masters?|msc) application|"
    r"(?:undergraduate|graduate|phd|doctoral|masters?|msc)(?:['’]s)? admissions?|"
    r"admissions? (?:requirements?|eligibility|criteria|documents?|materials?|process|steps?)|"
    r"admission process|admissions process)\b"
    r"|(?:كيفية التقديم|كيف\s+(?:أ|ا)?قدّم|كيف\s+(?:أ|ا)?قدم|التقديم|إجراءات القبول|"
    r"اجراءات القبول|طلب الالتحاق|تقديم الطلب)",
    flags=re.IGNORECASE,
)
_NON_ADMISSIONS_APPLICATION_RE = re.compile(
    r"\b(?:job|jobs|career|careers|vacancy|vacancies|employment|supplier|grant proposal|"
    r"library|visitor|onsite access|borrow|research study|competition|event)\b"
    r"|(?:وظيفة|وظائف|مهنة|شواغر|مورد|مكتبة|زائر|فعالية|مسابقة)",
    flags=re.IGNORECASE,
)
_TIME_BOUND_RE = re.compile(
    r"\b(?:20\d{2}(?:\s*[-/]\s*\d{2,4})?|fall|spring|autumn|intake|cycle|"
    r"deadline|currently|current|latest|open now)\b"
    r"|(?:خريف|ربيع|دفعة|دورة القبول|الموعد النهائي|حاليا|حالياً)",
    flags=re.IGNORECASE,
)
_DESCRIPTIVE_APPLY_RE = re.compile(
    r"\b(?:requirements?|rules?|conditions?|policies|fees?)\s+apply\s+to\b",
    flags=re.IGNORECASE,
)
_ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061a\u0640\u064b-\u065f\u0670\u06d6-\u06ed]"
)
_ARABIC_ADMISSIONS_TOPIC_RE = re.compile(
    r"(?:القبول|التقديم|طلب\s+الالتحاق|الالتحاق|التسجيل)"
)
_ARABIC_ADMISSIONS_DETAIL_RE = re.compile(
    r"(?:الحد\s+الادني|المعدل|المستندات?|الوثائق?|الشهادات?|"
    r"(?:ال)?متطلب(?:ات)?|الشروط?|المعايير|معايير|الاهليه|المهارات?|مهارات|"
    r"البرمجه|البرمجة|الخبره|الخبرة)"
)


def _routing_text(query: Any) -> str:
    """Return query text plus safe Arabic-normalized routing aliases.

    Arabic conjunctions and prepositions attach to nouns. Adding de-prefixed
    aliases lets general intent rules recognize forms such as ``للتقديم`` and
    ``والقبول`` without maintaining a phrase list for every surface form.
    """

    text = " ".join(str(query or "").split()).casefold()
    if not re.search(r"[\u0600-\u06ff]", text):
        return text

    normalized = _ARABIC_DIACRITICS_RE.sub("", text).translate(
        str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})
    )
    aliases: list[str] = []
    for token in re.findall(r"[\u0621-\u064a]+", normalized):
        if token.startswith(("وال", "فال", "بال", "كال")) and len(token) > 4:
            aliases.append(token[1:])
        if token.startswith("لل") and len(token) > 3:
            aliases.append("ال" + token[2:])
    return " ".join(dict.fromkeys((text, normalized, *aliases)))


def admissions_workflow_audience(query: Any) -> str:
    """Classify an application-process query without treating news as workflow."""

    text = _routing_text(query)
    workflow_match = _WORKFLOW_RE.search(text)
    arabic_detail_request = bool(
        _ARABIC_ADMISSIONS_TOPIC_RE.search(text)
        and _ARABIC_ADMISSIONS_DETAIL_RE.search(text)
    )
    if (
        not text
        or (not workflow_match and not arabic_detail_request)
        or _NON_ADMISSIONS_APPLICATION_RE.search(text)
    ):
        return ""
    if (
        workflow_match
        and workflow_match.group(0).casefold() == "apply"
        and _DESCRIPTIVE_APPLY_RE.search(text)
    ):
        # “Requirements apply to undergraduate applicants” describes scope;
        # it is not a request to perform the application workflow.
        return ""
    if re.search(r"\b(?:undergraduate|bachelor|bsc|b\.sc)\b|(?:البكالوريوس|الجامعية)", text):
        return "undergraduate"
    phd_requested = bool(
        re.search(r"\b(?:phd|ph\.d|doctorate|doctoral)\b|(?:الدكتوراه|دكتوراه)", text)
    )
    masters_requested = bool(
        re.search(r"\b(?:master|masters|msc|m\.sc)\b|(?:الماجستير|ماجستير)", text)
    )
    if phd_requested and masters_requested:
        # A comparison or shared-requirements question spans both graduate
        # admissions surfaces; routing it as Ph.D.-only silently drops the
        # master's evidence.
        return "graduate"
    if phd_requested:
        return "phd"
    if masters_requested:
        return "masters"
    if re.search(r"\bgraduate\b|(?:الدراسات العليا|برامج الدراسات العليا)", text):
        return "graduate"
    if "admission" in text or "mbzuai" in text or "القبول" in text or "الجامعة" in text:
        return "generic"
    return ""


def is_time_bound_admissions_query(query: Any) -> bool:
    return bool(_TIME_BOUND_RE.search(" ".join(str(query or "").split()).casefold()))


def canonical_admissions_marker(query: Any) -> str:
    audience = admissions_workflow_audience(query)
    return {
        "phd": "/admissions/graduate-phd-admissions",
        "masters": "/graduate-masters-admissions",
        "undergraduate": "/admissions-aid/undergraduate-admissions",
        "graduate": "/admissions",
        "generic": "/admissions",
    }.get(audience, "")


def admissions_surface_preference(
    query: Any,
    *,
    source_url: Any,
    title: Any = "",
    page_type: Any = "",
) -> float:
    """Return a bounded preference for canonical workflow vs. news surfaces."""

    audience = admissions_workflow_audience(query)
    if not audience:
        return 0.0
    url = str(source_url or "").casefold().rstrip("/")
    page_title = " ".join(str(title or "").split()).casefold()
    surface = str(page_type or "").casefold()
    is_news = (
        surface == "news_or_event"
        or "/news/" in url
        or "/news-events/news/" in url
        or "/knowledge-center/the-node/" in url
        or bool(re.search(r"\b(?:news|opens? admissions|admissions? (?:cycle|now open))\b", page_title))
    )
    if is_news:
        return -0.10 if is_time_bound_admissions_query(query) else -1.0
    if "/faq/" in url or "/faqs/" in url:
        return -0.25

    desired_markers = {
        "phd": ("/admissions/graduate-phd-admissions",),
        "masters": ("/graduate-masters-admissions",),
        "undergraduate": (
            "/admissions-aid/undergraduate-admissions",
            "/admissions/undergraduate-admissions",
            "/undergraduate-admissions",
        ),
        "graduate": ("/admissions", "/admissions-mbzuai"),
        "generic": ("/admissions", "/admissions-mbzuai"),
    }[audience]
    if any(url.endswith(marker) for marker in desired_markers):
        return 1.0

    if audience == "phd" and any(marker in url for marker in ("master", "undergraduate")):
        return -0.70
    if audience == "masters" and any(marker in url for marker in ("phd", "undergraduate")):
        return -0.70
    if audience == "undergraduate" and any(marker in url for marker in ("graduate-phd", "graduate-master")):
        return -0.70
    if surface == "admissions_or_program" or "admission" in url:
        return 0.25
    return 0.0
