"""Canonical admissions workflow routing shared by retrieval and navigation."""

from __future__ import annotations

import re
from typing import Any


_WORKFLOW_RE = re.compile(
    r"\b(?:apply|applying|application process|application steps?|submit(?:ting)? (?:an? )?application|"
    r"start(?:ing)? (?:an? )?(?:\w+\s+){0,4}application|"
    r"(?:undergraduate|graduate|phd|doctoral|masters?|msc) application|"
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
_TITLED_APPLY_RE = re.compile(
    r"\b(?:titled|called|named)\s+[\"'“‘][^\"'”’]{0,180}\bapply(?:ing)?\b",
    flags=re.IGNORECASE,
)
_ADMISSIONS_INFORMATION_RE = re.compile(
    r"\b(?:admissions?|entry requirements?|admission requirements?|eligibility|eligible|"
    r"application documents?|supporting documents?|academic requirements?|english proficiency|"
    r"ielts|toefl|gre|graduate record examination|referees?|recommendation letters?|"
    r"screening exam|admission interview)\b"
    r"|(?:القبول|متطلبات القبول|شروط القبول|المتطلبات|الشروط|الأهلية|اهلية|المؤهلات الأكاديمية|"
    r"الوثائق|المستندات|الوثائق المطلوبة|المستندات المطلوبة|أوراق التقديم|إثبات اللغة الإنجليزية|"
    r"اختبار اللغة الإنجليزية|اختبار القبول|المقابلة|خطابات التوصية|المراجع)",
    flags=re.IGNORECASE,
)
_DEGREE_AUDIENCE_RE = re.compile(
    r"\b(?:undergraduate|bachelor|bsc|b\.sc|graduate|master|masters|msc|m\.sc|"
    r"phd|ph\.d|doctorate|doctoral|applicant)\b"
    r"|(?:البكالوريوس|الجامعية|الدراسات العليا|الماجستير|ماجستير|الدكتوراه|دكتوراه|متقدم)",
    flags=re.IGNORECASE,
)


def admissions_workflow_audience(query: Any) -> str:
    """Classify an application-process query without treating news as workflow."""

    text = " ".join(str(query or "").split()).casefold()
    workflow_match = _WORKFLOW_RE.search(text)
    if not text or not workflow_match or _NON_ADMISSIONS_APPLICATION_RE.search(text):
        return ""
    if workflow_match.group(0).casefold() == "apply" and _DESCRIPTIVE_APPLY_RE.search(text):
        # “Requirements apply to undergraduate applicants” describes scope;
        # it is not a request to perform the application workflow.
        return ""
    if workflow_match.group(0).casefold() in {"apply", "applying"} and _TITLED_APPLY_RE.search(text):
        # A work, event, or talk title such as “Applying Image Analysis …” is
        # content identity, not an instruction to enter the admissions flow.
        return ""
    if re.search(r"\b(?:undergraduate|bachelor|bsc|b\.sc)\b|(?:البكالوريوس|الجامعية)", text):
        return "undergraduate"
    if re.search(r"\b(?:phd|ph\.d|doctorate|doctoral)\b|(?:الدكتوراه|دكتوراه)", text):
        return "phd"
    if re.search(r"\b(?:master|masters|msc|m\.sc)\b|(?:الماجستير|ماجستير)", text):
        return "masters"
    if re.search(r"\bgraduate\b|(?:الدراسات العليا|برامج الدراسات العليا)", text):
        return "graduate"
    if "admission" in text or "mbzuai" in text or "القبول" in text or "الجامعة" in text:
        return "generic"
    return ""


def admissions_information_audience(query: Any) -> str:
    """Classify an admissions-information query without encoding an answer.

    Users often ask about eligibility, documents, tests, or interviews without
    using the verb ``apply``. Those questions still need authoritative
    admissions/program pages rather than a news story that merely mentions an
    intake. Requiring either explicit admissions language or a degree audience
    plus an admissions facet keeps unrelated uses of words such as
    ``requirements`` out of this route.
    """

    text = " ".join(str(query or "").split()).casefold()
    if not text or _NON_ADMISSIONS_APPLICATION_RE.search(text):
        return ""
    direct_admissions = bool(re.search(r"\badmissions?\b|(?:القبول)", text))
    if not _ADMISSIONS_INFORMATION_RE.search(text):
        return ""
    if not direct_admissions and not _DEGREE_AUDIENCE_RE.search(text):
        return ""
    if re.search(r"\b(?:undergraduate|bachelor|bsc|b\.sc)\b|(?:البكالوريوس|الجامعية)", text):
        return "undergraduate"
    if re.search(r"\b(?:phd|ph\.d|doctorate|doctoral)\b|(?:الدكتوراه|دكتوراه)", text):
        return "phd"
    if re.search(r"\b(?:master|masters|msc|m\.sc)\b|(?:الماجستير|ماجستير)", text):
        return "masters"
    if re.search(r"\bgraduate\b|(?:الدراسات العليا|برامج الدراسات العليا)", text):
        return "graduate"
    return "generic"


def admissions_query_audience(query: Any) -> str:
    """Return the audience for workflow or informational admissions queries."""

    return admissions_workflow_audience(query) or admissions_information_audience(query)


def is_time_bound_admissions_query(query: Any) -> bool:
    return bool(_TIME_BOUND_RE.search(" ".join(str(query or "").split()).casefold()))


def canonical_admissions_marker(query: Any) -> str:
    # URL pinning is retained only for explicit workflow requests. Informational
    # questions use the authority score below so retrieval stays semantic and
    # does not acquire a new query-to-route shortcut.
    audience = admissions_workflow_audience(query)
    return {
        "phd": "/admissions/graduate-phd-admissions",
        "masters": "/graduate-masters-admissions",
        "undergraduate": "/admissions/undergraduate-admissions",
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

    audience = admissions_query_audience(query)
    if not audience:
        return 0.0
    url = str(source_url or "").casefold().rstrip("/")
    page_title = " ".join(str(title or "").split()).casefold()
    surface = str(page_type or "").casefold()
    is_news = (
        surface == "news_or_event"
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
