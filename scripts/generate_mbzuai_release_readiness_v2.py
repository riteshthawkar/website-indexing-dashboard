from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.dataset import EvalExample, load_eval_examples, write_eval_examples


INPUT_PATH = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_internal_v4.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_release_readiness_v2.jsonl"

SELECTED_IDS = [
    # Canonical exact facts and policy facts.
    "fact-001",
    "fact-002",
    "fact-003",
    "fact-004",
    "fact-005",
    "fact-006",
    "fact-007",
    "fact-008",
    "fact-009",
    # Paraphrase and near-match robustness.
    "fact-015",
    "fact-017",
    "fact-018",
    "fact-019",
    "fact-020",
    "fact-022",
    "fact-024",
    # Contact precision and fallback behavior.
    "fact-025",
    "fact-026",
    "fact-027",
    "fact-028",
    # Scoped retrieval and graph/relation style queries.
    "scoped-001",
    "scoped-002",
    "scoped-004",
    "scoped-007",
    "scoped-009",
    "scoped-011",
    # Multi-evidence synthesis.
    "synthesis-001",
    "synthesis-003",
    "synthesis-004",
    # PDF/multimodal map evidence.
    "multimodal-001",
    "multimodal-002",
    "multimodal-003",
    "multimodal-006",
    "multimodal-007",
    # Abstention and hallucination guards.
    "noanswer-001",
    "noanswer-002",
    "noanswer-003",
    "noanswer-004",
    "noanswer-005",
]

ANSWER_RULES = {
    "fact-001": (["Masdar", "Abu Dhabi"], ["Dubai campus", "Singapore campus", "New York campus"]),
    "fact-002": (["Mohamed bin Zayed"], ["Khalifa", "Sheikh Zayed bin Sultan"]),
    "fact-003": (["8:00", "6:00", "Friday"], ["24/7", "Saturday"]),
    "fact-004": (["parking", "students", "guests"], ["no parking"]),
    "fact-005": (["North Car Park"], ["South Car Park", "airport parking"]),
    "fact-006": (["student accommodation"], ["does not provide accommodation"]),
    "fact-007": (["shuttle", "students"], ["metro service"]),
    "fact-008": (["does not provide housing", "parents"], ["parents can stay on campus"]),
    "fact-009": (["8:00", "5:00", "12:30"], ["24/7"]),
    "fact-015": (["Abu Dhabi"], ["Dubai", "Sharjah"]),
    "fact-017": (["8:00", "6:00", "Friday"], ["24/7"]),
    "fact-018": (["parking", "guests"], ["not available to guests"]),
    "fact-019": (["North Car Park"], ["South Car Park", "airport parking"]),
    "fact-020": (["student accommodation"], ["no student housing"]),
    "fact-022": (["does not provide housing", "parents"], ["family members can stay in student accommodation"]),
    "fact-024": (["Mohamed bin Zayed"], ["Khalifa", "Sheikh Zayed bin Sultan"]),
    "fact-025": (["admission@mbzuai.ac.ae"], ["ug.admission@mbzuai.ac.ae"]),
    "fact-026": (["admission@mbzuai.ac.ae"], ["committee@mbzuai.ac.ae"]),
    "fact-027": (["admission@mbzuai.ac.ae"], ["ug.admission@mbzuai.ac.ae"]),
    "fact-028": (["ug.admission@mbzuai.ac.ae"], ["general admissions email is admission"]),
    "scoped-001": (["Machine Learning", "Computer Vision", "Natural Language Processing"], ["Data Science only"]),
    "scoped-002": (["Law No. 25", "Abu Dhabi Executive Council"], ["federal university", "Ministry of Education"]),
    "scoped-004": (["accommodation", "laboratories", "library"], ["airport terminal"]),
    "scoped-007": (["accommodation", "laboratories", "canteen"], ["football stadium"]),
    "scoped-009": (["Law No. 25", "Abu Dhabi Executive Council"], ["federal university", "Ministry of Education"]),
    "scoped-011": (["library", "sports", "canteen"], ["airport terminal"]),
    "synthesis-001": (["Masdar", "working hours", "accommodation", "parking"], ["tuition waiver"]),
    "synthesis-003": (["Masdar", "transport", "parking", "facilities"], ["Singapore campus"]),
    "synthesis-004": (["Masdar", "parking", "transport", "facilities"], ["Paris campus"]),
    "multimodal-001": (["campus map", "buildings", "facilities"], ["football stadium"]),
    "multimodal-002": (["Knowledge Center", "library", "Medical Center"], ["football stadium"]),
    "multimodal-003": (["Medical Center", "campus map"], ["off campus hospital"]),
    "multimodal-006": (["campus map", "buildings", "services"], ["New York campus"]),
    "multimodal-007": (["Knowledge Center", "Medical Center", "gym"], ["football stadium"]),
    "noanswer-001": ([], ["Singapore office phone", "+65"]),
    "noanswer-002": ([], ["New York campus", "Manhattan", "Brooklyn"]),
    "noanswer-003": ([], ["London campus", "+44"]),
    "noanswer-004": ([], ["Paris campus", "metro station"]),
    "noanswer-005": ([], ["Toronto admissions desk", "extension "]),
}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys([str(value) for value in values if str(value)]))


def _metadata_values(example: EvalExample, key: str) -> list[str]:
    value = (example.metadata or {}).get(key)
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value:
        return [value]
    return []


def _combined_example(
    *,
    source_examples: dict[str, EvalExample],
    new_id: str,
    source_ids: list[str],
    query: str,
    reference_answer: str,
    must_include: list[str],
    must_not_include: list[str],
    answer_should_cover: list[str],
    expected_source_hints: list[str],
    expected_followup_topics: list[str],
    notes: str,
) -> EvalExample:
    sources = [source_examples[source_id] for source_id in source_ids]
    gold_chunk_ids = _unique([item for source in sources for item in source.gold_chunk_ids])
    gold_parent_ids = _unique([item for source in sources for item in source.gold_parent_ids])
    gold_media_ids = _unique([item for source in sources for item in source.gold_media_ids])
    alternate_gold_chunk_ids = _unique(
        [item for source in sources for item in _metadata_values(source, "alternate_gold_chunk_ids")]
    )
    alternate_gold_parent_ids = _unique(
        [item for source in sources for item in _metadata_values(source, "alternate_gold_parent_ids")]
    )
    alternate_gold_media_ids = _unique(
        [item for source in sources for item in _metadata_values(source, "alternate_gold_media_ids")]
    )
    metadata = {
        "benchmark_tags": [
            "release_readiness_v2",
            "deep_answer",
            "multi_doc",
            "citation_quality",
            "followup_quality",
        ],
        "source_example_ids": source_ids,
        "answer_must_include": must_include,
        "answer_must_not_include": must_not_include,
        "answer_should_cover": answer_should_cover,
        "expected_source_hints": expected_source_hints,
        "expected_followup_topics": expected_followup_topics,
        "release_suite": "release_readiness_v2",
    }
    if alternate_gold_chunk_ids:
        metadata["alternate_gold_chunk_ids"] = alternate_gold_chunk_ids
    if alternate_gold_parent_ids:
        metadata["alternate_gold_parent_ids"] = alternate_gold_parent_ids
    if alternate_gold_media_ids:
        metadata["alternate_gold_media_ids"] = alternate_gold_media_ids
    return EvalExample(
        id=new_id,
        query=query,
        query_type="synthesis",
        source_type="mixed",
        no_answer=False,
        reference_answer=reference_answer,
        gold_chunk_ids=gold_chunk_ids,
        gold_parent_ids=gold_parent_ids,
        gold_media_ids=gold_media_ids,
        notes=notes,
        metadata=metadata,
    )


def _deep_answer_examples(source_examples: dict[str, EvalExample]) -> list[EvalExample]:
    return [
        _combined_example(
            source_examples=source_examples,
            new_id="deep-001",
            source_ids=["fact-001", "fact-003", "fact-004", "fact-006", "fact-007", "scoped-004"],
            query=(
                "Give a detailed practical guide for a new graduate student arriving at MBZUAI, "
                "covering campus location, working hours, accommodation, parking, shuttle transport, "
                "and important facilities."
            ),
            reference_answer=(
                "A new graduate student should know that MBZUAI is in Masdar City, Abu Dhabi, "
                "has weekday working hours with a shorter Friday schedule, provides student accommodation, "
                "has parking for registered students and guests, offers shuttle transport, and provides campus "
                "facilities such as laboratories, library/knowledge center, sports spaces, canteen, and other support facilities."
            ),
            must_include=["Masdar", "working hours", "accommodation", "parking", "shuttle", "library"],
            must_not_include=["Singapore campus", "tuition waiver", "24/7"],
            answer_should_cover=[
                "location",
                "weekday and Friday working-hours distinction",
                "student accommodation",
                "student/guest parking",
                "student shuttle transport",
                "campus facilities with several concrete examples",
            ],
            expected_source_hints=["FAQ", "campus facilities", "student accommodation", "shuttle"],
            expected_followup_topics=["admissions", "housing", "transport", "campus facilities"],
            notes="Deep release check: multi-page student orientation answer with completeness, citations, and follow-up quality requirements.",
        ),
        _combined_example(
            source_examples=source_examples,
            new_id="deep-002",
            source_ids=["fact-025", "fact-028", "fact-009"],
            query=(
                "Compare the right contact paths for general admissions, undergraduate admissions, "
                "and IT support for the online screening exam, including when IT support is available."
            ),
            reference_answer=(
                "The general admissions contact is admission@mbzuai.ac.ae, undergraduate admissions uses "
                "ug.admission@mbzuai.ac.ae, and online screening exam IT support is available from 8:00 AM "
                "to 5:00 PM UAE time Monday to Thursday and 8:00 AM to 12:30 PM UAE time on Friday."
            ),
            must_include=["admission@mbzuai.ac.ae", "ug.admission@mbzuai.ac.ae", "8:00", "12:30"],
            must_not_include=["committee@mbzuai.ac.ae", "24/7", "general admissions email is ug.admission"],
            answer_should_cover=[
                "general admissions email",
                "undergraduate admissions email",
                "online screening exam IT support purpose",
                "Monday-Thursday IT support hours",
                "Friday IT support hours",
                "clear distinction between the two admissions emails",
            ],
            expected_source_hints=["admissions", "undergraduate", "online screening exam", "IT support"],
            expected_followup_topics=["application requirements", "application deadlines", "screening exam process"],
            notes="Deep release check: contact disambiguation across PDF and webpage sources.",
        ),
        _combined_example(
            source_examples=source_examples,
            new_id="deep-003",
            source_ids=["fact-001", "fact-002", "scoped-002"],
            query=(
                "Explain MBZUAI's institutional identity: who it is named after, where it is located, "
                "what law established it, and which entity it is affiliated with."
            ),
            reference_answer=(
                "MBZUAI is named after His Highness Sheikh Mohamed bin Zayed Al Nahyan, is located in "
                "Masdar City, Abu Dhabi, was established under Law No. 25 of 2019, and is affiliated with "
                "the Abu Dhabi Executive Council."
            ),
            must_include=["Mohamed bin Zayed", "Masdar", "Law No. 25", "Abu Dhabi Executive Council"],
            must_not_include=["Ministry of Education", "federal university", "Dubai"],
            answer_should_cover=[
                "named-after person",
                "campus location",
                "legal establishment instrument",
                "institutional affiliation",
            ],
            expected_source_hints=["FAQ", "university catalogue", "law", "affiliated"],
            expected_followup_topics=["leadership", "governance", "campus location"],
            notes="Deep release check: relation-heavy institutional answer requiring webpage and PDF evidence.",
        ),
        _combined_example(
            source_examples=source_examples,
            new_id="deep-004",
            source_ids=["scoped-004", "multimodal-002", "multimodal-003"],
            query=(
                "Give a detailed answer about MBZUAI campus facilities using both the campus facilities page "
                "and the campus map: what facilities are available and which support services are visibly marked?"
            ),
            reference_answer=(
                "MBZUAI campus facilities include student accommodation, laboratories, a library and knowledge center, "
                "an auditorium, sports and gym spaces, pool, canteen, retail outlets, and collaboration spaces. "
                "The campus map also labels services and places such as the Knowledge Center, library, Medical Center, "
                "gym, prayer rooms, swimming pool, canteen, student residences, and public washrooms."
            ),
            must_include=["accommodation", "laboratories", "Knowledge Center", "library", "Medical Center", "gym"],
            must_not_include=["football stadium", "airport terminal"],
            answer_should_cover=[
                "facilities page amenities",
                "campus map labeled services",
                "medical center",
                "library or knowledge center",
                "sports/gym or pool",
                "student residences or accommodation",
            ],
            expected_source_hints=["campus facilities", "campus map", "Medical Center", "Knowledge Center"],
            expected_followup_topics=["campus map", "student accommodation", "sports facilities"],
            notes="Deep release check: answer must combine webpage facilities with multimodal/PDF map evidence.",
        ),
        _combined_example(
            source_examples=source_examples,
            new_id="deep-005",
            source_ids=["fact-008", "fact-018", "fact-021", "scoped-011"],
            query=(
                "A student's family is visiting MBZUAI. Explain what the corpus says about family accommodation, "
                "guest parking, transport options, and useful campus amenities, without inventing unsupported visitor services."
            ),
            reference_answer=(
                "The corpus says MBZUAI does not provide housing for parents or visiting family members, though nearby "
                "hotels or Airbnbs may be recommended. Parking is available for guests, students have shuttle transport, "
                "and the campus includes amenities such as the library/knowledge center, canteen, sports spaces, gym, "
                "pool, retail outlets, and other support facilities."
            ),
            must_include=["does not provide housing", "parents", "parking", "shuttle", "canteen"],
            must_not_include=["family members can stay in student accommodation", "free hotel", "visitor visa service"],
            answer_should_cover=[
                "family/parent accommodation limitation",
                "guest parking",
                "student shuttle transport",
                "several campus amenities",
                "explicit avoidance of unsupported visitor-service claims",
            ],
            expected_source_hints=["parents", "parking", "shuttle", "campus amenities"],
            expected_followup_topics=["nearby hotels", "campus access", "visitor parking"],
            notes="Deep release check: nuanced visitor answer with a negative policy and multiple practical support facts.",
        ),
    ]


def build_examples() -> list[EvalExample]:
    source_examples = {example.id: example for example in load_eval_examples(INPUT_PATH)}
    examples: list[EvalExample] = []
    missing = [example_id for example_id in SELECTED_IDS if example_id not in source_examples]
    if missing:
        raise ValueError(f"Missing source examples: {missing}")

    for example_id in SELECTED_IDS:
        source = source_examples[example_id]
        metadata = dict(source.metadata or {})
        tags = list(dict.fromkeys([*metadata.get("benchmark_tags", []), "release_readiness_v2"]))
        if source.no_answer:
            tags = list(dict.fromkeys([*tags, "abstention"]))
        must_include, must_not_include = ANSWER_RULES.get(example_id, ([], []))
        metadata.update(
            {
                "benchmark_tags": tags,
                "answer_must_include": list(must_include),
                "answer_must_not_include": list(must_not_include),
                "release_suite": "release_readiness_v2",
            }
        )
        examples.append(
            EvalExample(
                id=source.id,
                query=source.query,
                query_type=source.query_type,
                source_type=source.source_type,
                no_answer=source.no_answer,
                reference_answer=source.reference_answer,
                gold_chunk_ids=list(source.gold_chunk_ids or []),
                gold_parent_ids=list(source.gold_parent_ids or []),
                gold_media_ids=list(source.gold_media_ids or []),
                notes=f"Release readiness v2: {source.notes}",
                metadata=metadata,
            )
        )
    examples.extend(_deep_answer_examples(source_examples))
    return examples


def main() -> int:
    examples = build_examples()
    write_eval_examples(OUTPUT_PATH, examples)
    print(f"Wrote {len(examples)} release-readiness examples to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
