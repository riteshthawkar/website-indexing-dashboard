"""Small semantic checks shared by automatic-evaluation validation."""

import re

from ._coherence import content_terms, topic_terms
from ._util import clean_text


_INTERNAL_EVIDENCE_ID = re.compile(
    r"(?:\bevidence[_ -]?id\s*:|\b(?:chunk|parent|fact|media|page-action|page-card):[\w-])",
    flags=re.IGNORECASE,
)
_ABSENCE_COMPARISON = re.compile(
    r"(?:\b(?:does not|doesn't|did not)\s+"
    r"(?:explicitly\s+|specifically\s+)?"
    r"(?:detail|mention|specify|state|include|provide|list)|"
    r"\b(?:is|are|was|were)\s+not\s+(?:explicitly\s+|specifically\s+)?"
    r"(?:itemized|mentioned|specified|detailed|provided|stated|listed)|"
    r"\bno\s+(?:explicit\s+|specific\s+)?(?:mention|detail|information)|"
    r"\bwithout\s+(?:any\s+)?(?:explicit\s+|specific\s+)?"
    r"(?:[\w-]+\s+){0,4}(?:mention|details?|information)|"
    r"\bnot (?:itemized|mentioned|specified|detailed)\b|"
    r"لا يذكر|لم يذكر|غير مذكور|لا توجد إشارة)",
    flags=re.IGNORECASE,
)
_VISUAL_CUE = re.compile(
    r"(?:\bimage\b|\bphoto(?:graph)?\b|\bfigure\b|\bdiagram\b|\bchart\b|"
    r"\bshown\b|\bvisible\b|\bdepict(?:s|ed)?\b|\bpictured\b|\bvisual\b|"
    r"الصورة|الصور|يظهر|تظهر|المرئي|المخطط|الرسم|الموضح)",
    flags=re.IGNORECASE,
)
_QUERY_COVERAGE_GENERIC = frozenset({"course", "courses", "division"})


def looks_like_abstention(answer: str) -> bool:
    normalized = clean_text(answer).casefold()
    phrases = (
        "the provided evidence does not",
        "the evidence does not",
        "information is not available",
        "no information is available",
        "does not establish this",
        "لا تتوفر معلومات",
        "لا تتوفر في الأدلة",
        "لا توجد معلومات",
        "الأدلة المقدمة لا",
        "لا يمكن تحديد",
    )
    return any(phrase in normalized for phrase in phrases)


def contains_internal_evidence_id(value: str) -> bool:
    return bool(_INTERNAL_EVIDENCE_ID.search(str(value or "")))


def infers_absence_from_comparison(value: str) -> bool:
    return bool(_ABSENCE_COMPARISON.search(str(value or "")))


def mentions_visual_evidence(value: str) -> bool:
    return bool(_VISUAL_CUE.search(str(value or "")))


def has_cross_scope_claim(case, selected, evidence) -> bool:
    source_ids = {evidence[item].source_id for item in selected if item in evidence}
    for claim in case.claims:
        claim_ids = set(claim.evidence_ids) & selected
        claim_sources = {
            evidence[item].source_id for item in claim_ids if item in evidence
        }
        required = source_ids if len(source_ids) > 1 else selected
        covered = claim_sources if len(source_ids) > 1 else claim_ids
        if required and required.issubset(covered):
            return True
    return False


def query_uncovered_evidence(case, evidence) -> tuple[str, ...]:
    if case.target_difficulty not in {"medium", "hard"}:
        return ()
    query_terms = topic_terms(case.query) - _QUERY_COVERAGE_GENERIC
    query_script = _dominant_script(" ".join(query_terms))
    uncovered = []
    for evidence_id in case.evidence_ids:
        unit = evidence.get(evidence_id)
        if unit is None:
            continue
        unit_terms = content_terms(unit) - _QUERY_COVERAGE_GENERIC
        if _dominant_script(" ".join(unit_terms)) != query_script:
            continue
        if not (query_terms & unit_terms):
            uncovered.append(evidence_id)
    return tuple(uncovered)


def evidence_scope_labels(evidence_ids, evidence) -> tuple[str, ...]:
    labels = []
    for evidence_id in evidence_ids:
        unit = evidence.get(evidence_id)
        path = tuple((unit.metadata.get("section_path") or ())) if unit else ()
        scope = " > ".join(str(item) for item in path) or getattr(unit, "kind", "unknown")
        labels.append(f"{evidence_id} [{scope}]")
    return tuple(labels)


def _dominant_script(value: str) -> str:
    arabic = len(re.findall(r"[\u0600-\u06ff]", value))
    latin = len(re.findall(r"[A-Za-z]", value))
    if arabic > latin:
        return "arabic"
    if latin:
        return "latin"
    return "other"


def answerable_semantic_issues(case) -> tuple[tuple[str, str], ...]:
    issues: list[tuple[str, str]] = []
    if looks_like_abstention(case.reference_answer):
        issues.append((
            "answerable_abstention",
            "Answerable case reference answer says the requested information is unavailable",
        ))
    elif contains_internal_evidence_id(case.reference_answer):
        issues.append((
            "reference_leaks_evidence_id",
            "Reference answer exposes an internal evidence identifier",
        ))
    if infers_absence_from_comparison(case.reference_answer):
        code = (
            "comparison_infers_absence"
            if case.task == "compare"
            else "reference_infers_absence"
        )
        issues.append((
            code,
            "Reference answer treats a selected excerpt's silence as proof of absence",
        ))
    if case.task == "multimodal" and not mentions_visual_evidence(case.query):
        issues.append((
            "multimodal_media_not_required",
            "Multimodal query does not require a visual observation",
        ))
    return tuple(issues)
