from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.dataset import EvalExample, load_eval_examples, write_eval_examples

INPUT_PATH = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_internal_v3.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_internal_v4.jsonl"

_RELATION_HEAVY_FACT_IDS = {
    "fact-001",
    "fact-002",
    "fact-004",
    "fact-005",
    "fact-006",
    "fact-007",
    "fact-008",
    "fact-010",
    "fact-011",
    "fact-012",
    "fact-013",
    "fact-014",
    "fact-015",
    "fact-016",
    "fact-018",
    "fact-019",
    "fact-020",
    "fact-021",
    "fact-022",
    "fact-024",
    "fact-025",
    "fact-026",
    "fact-027",
    "fact-028",
}

_RELATION_HEAVY_SCOPED_IDS = {
    "scoped-002",
    "scoped-009",
}

_CONTACT_LOOKUP_IDS = {
    "fact-025",
    "fact-026",
    "fact-027",
    "fact-028",
    "noanswer-005",
}


def _clone(
    base: EvalExample,
    *,
    new_id: str,
    query: str,
    query_type: str | None = None,
    source_type: str | None = None,
    reference_answer: str | None = None,
    no_answer: bool | None = None,
    notes: str | None = None,
    scenario: str | None = None,
) -> EvalExample:
    metadata = dict(base.metadata or {})
    metadata["paraphrase_of"] = base.id
    if scenario:
        metadata["scenario"] = scenario
    if "root_example_id" not in metadata:
        metadata["root_example_id"] = metadata.get("paraphrase_of") or base.id
    return EvalExample(
        id=new_id,
        query=query,
        query_type=query_type or base.query_type,
        source_type=source_type or base.source_type,
        no_answer=base.no_answer if no_answer is None else bool(no_answer),
        reference_answer=reference_answer if reference_answer is not None else base.reference_answer,
        gold_chunk_ids=list(base.gold_chunk_ids or []),
        gold_parent_ids=list(base.gold_parent_ids or []),
        gold_media_ids=list(base.gold_media_ids or []),
        notes=notes or f"Paraphrase of {base.id}.",
        metadata=metadata,
    )


def build_v4_examples() -> list[EvalExample]:
    base_examples = load_eval_examples(INPUT_PATH)
    by_id = {example.id: example for example in base_examples}
    additions = [
        _clone(
            by_id["fact-001"],
            new_id="fact-015",
            query="In which emirate is MBZUAI located?",
            notes="Paraphrase of fact-001 with emirate wording.",
            scenario="location_alias",
        ),
        _clone(
            by_id["fact-001"],
            new_id="fact-016",
            query="Where in Abu Dhabi is MBZUAI situated?",
            notes="Paraphrase of fact-001 with situated wording.",
            scenario="location_alias",
        ),
        _clone(
            by_id["fact-003"],
            new_id="fact-017",
            query="What are MBZUAI's weekday operating hours?",
            notes="Paraphrase of fact-003 with operating-hours wording.",
            scenario="hours_alias",
        ),
        _clone(
            by_id["fact-004"],
            new_id="fact-018",
            query="Does MBZUAI offer parking for guests and visitors?",
            notes="Paraphrase of fact-004 emphasizing guest parking.",
            scenario="parking_policy",
        ),
        _clone(
            by_id["fact-005"],
            new_id="fact-019",
            query="Where can vehicles be parked on the Masdar City campus?",
            notes="Paraphrase of fact-005 with campus-parking wording.",
            scenario="parking_location",
        ),
        _clone(
            by_id["fact-006"],
            new_id="fact-020",
            query="Is student housing available at MBZUAI?",
            notes="Paraphrase of fact-006 with student-housing wording.",
            scenario="accommodation",
        ),
        _clone(
            by_id["fact-007"],
            new_id="fact-021",
            query="Do MBZUAI students have shuttle transportation?",
            notes="Paraphrase of fact-007 with transport wording.",
            scenario="transport",
        ),
        _clone(
            by_id["fact-008"],
            new_id="fact-022",
            query="Can visiting family members stay in MBZUAI student accommodation?",
            notes="Paraphrase of fact-008 with family/accommodation wording.",
            scenario="accommodation_policy",
        ),
        _clone(
            by_id["fact-009"],
            new_id="fact-023",
            query="During what hours can applicants reach IT support for the MBZUAI online screening exam?",
            notes="Paraphrase of fact-009 with support-hours wording.",
            scenario="support_hours",
        ),
        _clone(
            by_id["fact-002"],
            new_id="fact-024",
            query="After whom is MBZUAI named?",
            notes="Paraphrase of fact-002 with explicit naming wording.",
            scenario="named_after",
        ),
        EvalExample(
            id="fact-025",
            query="What is the admissions email address?",
            query_type="fact",
            source_type="pdf",
            no_answer=False,
            reference_answer="The general admissions email address is admission@mbzuai.ac.ae.",
            gold_chunk_ids=["7293cf362e89b5ac9ed6::chunk::029:c43af33eef63"],
            gold_parent_ids=["9243f47d97ae3f42fe844be1", "f05d853af1f9d42a4832a0e4"],
            notes="Direct contact lookup. Prefer the contact-directory chunk, but other chunks stating the same admissions email are acceptable.",
            metadata={
                "alternate_gold_chunk_ids": [
                    "8693a56b12278e50e7c5::chunk::012:d2353ea48e8a",
                    "7f7cf7b96cfee31bd373::chunk::024:8e1122486b31",
                    "05d69ac14cafa7d44bee::chunk::004:5ad3bcf3c2bc",
                    "9175600f20c1418732aa::chunk::6509:0d65a21d4c0d",
                ],
                "alternate_gold_parent_ids": [
                    "4a932e787891961a3c8e572d",
                    "f0eea5c3603a77b6da9d3546",
                    "cc7374a4c28497664ebfc2d0",
                    "bbbe3a5783d379c868996734",
                    "3c16b4da42435a6034588cc5",
                    "05c43ec4b5945569c45a3198",
                    "7dd59213b2451ad90faab10a",
                    "7be76cbf3e860d1724e9f35d",
                ],
                "scenario": "contact_lookup",
            },
        ),
        EvalExample(
            id="fact-026",
            query="What is the admissions committee email address?",
            query_type="fact",
            source_type="pdf",
            no_answer=False,
            reference_answer="The corpus does not provide a distinct committee-only email, so the best grounded admissions contact is admission@mbzuai.ac.ae.",
            gold_chunk_ids=["7293cf362e89b5ac9ed6::chunk::029:c43af33eef63"],
            gold_parent_ids=["9243f47d97ae3f42fe844be1", "f05d853af1f9d42a4832a0e4"],
            notes="Specific-role fallback. The retriever should return the general admissions email when no committee-specific email is present.",
            metadata={
                "alternate_gold_chunk_ids": [
                    "8693a56b12278e50e7c5::chunk::012:d2353ea48e8a",
                    "7f7cf7b96cfee31bd373::chunk::024:8e1122486b31",
                    "05d69ac14cafa7d44bee::chunk::004:5ad3bcf3c2bc",
                ],
                "alternate_gold_parent_ids": [
                    "4a932e787891961a3c8e572d",
                    "f0eea5c3603a77b6da9d3546",
                    "cc7374a4c28497664ebfc2d0",
                    "bbbe3a5783d379c868996734",
                    "3c16b4da42435a6034588cc5",
                    "05c43ec4b5945569c45a3198",
                ],
                "scenario": "contact_role_fallback",
            },
        ),
        EvalExample(
            id="fact-027",
            query="How can I contact admissions?",
            query_type="fact",
            source_type="pdf",
            no_answer=False,
            reference_answer="Admissions can be contacted at admission@mbzuai.ac.ae.",
            gold_chunk_ids=["7293cf362e89b5ac9ed6::chunk::029:c43af33eef63"],
            gold_parent_ids=["9243f47d97ae3f42fe844be1", "f05d853af1f9d42a4832a0e4"],
            notes="Contact-intent wording should still retrieve the admissions contact chunk.",
            metadata={
                "alternate_gold_chunk_ids": [
                    "8693a56b12278e50e7c5::chunk::012:d2353ea48e8a",
                    "7f7cf7b96cfee31bd373::chunk::024:8e1122486b31",
                    "9175600f20c1418732aa::chunk::6509:0d65a21d4c0d",
                ],
                "alternate_gold_parent_ids": [
                    "4a932e787891961a3c8e572d",
                    "f0eea5c3603a77b6da9d3546",
                    "cc7374a4c28497664ebfc2d0",
                    "bbbe3a5783d379c868996734",
                    "7dd59213b2451ad90faab10a",
                    "7be76cbf3e860d1724e9f35d",
                ],
                "scenario": "contact_lookup",
            },
        ),
        EvalExample(
            id="fact-028",
            query="What is the undergraduate admissions email address?",
            query_type="fact",
            source_type="webpage",
            no_answer=False,
            reference_answer="The undergraduate admissions email address is ug.admission@mbzuai.ac.ae.",
            gold_chunk_ids=["50e9160a9acf5acbaff7::chunk::002:c7b30b7493f0"],
            gold_parent_ids=["d198d5c6985f93ba1a7f2b1e", "2ae9a7fa5f23a0a903c7f490"],
            notes="Exact undergraduate-contact lookup.",
            metadata={
                "alternate_gold_chunk_ids": [
                    "9175600f20c1418732aa::chunk::6509:0d65a21d4c0d",
                ],
                "alternate_gold_parent_ids": [
                    "7dd59213b2451ad90faab10a",
                    "7be76cbf3e860d1724e9f35d",
                ],
                "scenario": "contact_lookup_undergraduate",
            },
        ),
        _clone(
            by_id["scoped-001"],
            new_id="scoped-008",
            query="Summarize the main research and study specializations in MBZUAI graduate degrees.",
            notes="Paraphrase of scoped-001 with research/study wording.",
            scenario="academic_programs",
        ),
        _clone(
            by_id["scoped-002"],
            new_id="scoped-009",
            query="Summarize the law that created MBZUAI and the authority it is affiliated with.",
            notes="Paraphrase of scoped-002 with governance wording.",
            scenario="legal_affiliation",
        ),
        _clone(
            by_id["scoped-003"],
            new_id="scoped-010",
            query="What everyday campus amenities can MBZUAI students use on site?",
            notes="Paraphrase of scoped-003 focused on daily student amenities.",
            scenario="campus_amenities",
        ),
        _clone(
            by_id["scoped-004"],
            new_id="scoped-011",
            query="Outline the core campus amenities and support facilities at MBZUAI.",
            notes="Paraphrase of scoped-004 with amenities/support wording.",
            scenario="campus_amenities",
        ),
        _clone(
            by_id["synthesis-001"],
            new_id="synthesis-003",
            query="Prepare a newcomer briefing covering where MBZUAI is, when offices operate, how transport and parking work, and what support facilities exist on campus.",
            notes="Paraphrase of synthesis-001 with briefing framing.",
            scenario="orientation_summary",
        ),
        _clone(
            by_id["synthesis-002"],
            new_id="synthesis-004",
            query="Summarize the practical campus information a visitor should know before arriving at MBZUAI, including location, parking, transport, and facilities.",
            notes="Paraphrase of synthesis-002 with visitor framing.",
            scenario="visitor_summary",
        ),
        _clone(
            by_id["multimodal-001"],
            new_id="multimodal-006",
            query="How does the campus map organize buildings, parking, and campus services?",
            notes="Paraphrase of multimodal-001 emphasizing layout organization.",
            scenario="map_layout",
        ),
        _clone(
            by_id["multimodal-002"],
            new_id="multimodal-007",
            query="Which support services are visibly marked on the MBZUAI campus map?",
            notes="Paraphrase of multimodal-002 with support-services wording.",
            scenario="map_services",
        ),
        _clone(
            by_id["multimodal-003"],
            new_id="multimodal-008",
            query="Where on the campus map is the Medical Center located relative to nearby services?",
            notes="Paraphrase of multimodal-003 with relative-location wording.",
            scenario="map_relative_location",
        ),
        EvalExample(
            id="noanswer-004",
            query="What is the metro station for MBZUAI's Paris campus?",
            query_type="fact",
            source_type="none",
            no_answer=True,
            reference_answer="No grounded answer should be returned because the corpus does not contain this information.",
            notes="Additional abstention case for a nonexistent overseas campus.",
            metadata={"paraphrase_of": "noanswer-001", "scenario": "nonexistent_overseas_campus"},
        ),
        EvalExample(
            id="noanswer-005",
            query="What is the extension number for MBZUAI's Toronto admissions desk?",
            query_type="fact",
            source_type="none",
            no_answer=True,
            reference_answer="No grounded answer should be returned because the corpus does not contain this information.",
            notes="Additional abstention case for a nonexistent overseas admissions desk.",
            metadata={"paraphrase_of": "noanswer-001", "scenario": "nonexistent_contact"},
        ),
    ]
    output = [*base_examples, *additions]
    tagged: list[EvalExample] = []
    for example in output:
        metadata = dict(example.metadata or {})
        tags = list(metadata.get("benchmark_tags") or [])
        if example.id in _RELATION_HEAVY_FACT_IDS:
            tags.append("relation_heavy_fact")
            tags.append("relation_heavy")
        if example.id in _RELATION_HEAVY_SCOPED_IDS:
            tags.append("relation_heavy_scoped")
            tags.append("relation_heavy")
        if example.id in _CONTACT_LOOKUP_IDS:
            tags.append("contact_lookup")
        metadata["benchmark_tags"] = list(dict.fromkeys(str(tag) for tag in tags if str(tag)))
        tagged.append(
            EvalExample(
                id=example.id,
                query=example.query,
                query_type=example.query_type,
                source_type=example.source_type,
                no_answer=example.no_answer,
                reference_answer=example.reference_answer,
                gold_chunk_ids=list(example.gold_chunk_ids or []),
                gold_parent_ids=list(example.gold_parent_ids or []),
                gold_media_ids=list(example.gold_media_ids or []),
                notes=example.notes,
                metadata=metadata,
            )
        )
    return tagged


def main() -> int:
    examples = build_v4_examples()
    write_eval_examples(OUTPUT_PATH, examples)
    print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
