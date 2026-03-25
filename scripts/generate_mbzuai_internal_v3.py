from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.evaluation.dataset import EvalExample, load_eval_examples, write_eval_examples

INPUT_PATH = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_internal_v2.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "eval" / "mbzuai_gold" / "mbzuai_internal_v3.jsonl"


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
) -> EvalExample:
    metadata = dict(base.metadata or {})
    metadata["paraphrase_of"] = base.id
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


def build_v3_examples() -> list[EvalExample]:
    base_examples = load_eval_examples(INPUT_PATH)
    by_id = {example.id: example for example in base_examples}
    additions = [
        _clone(
            by_id["fact-001"],
            new_id="fact-010",
            query="In which city is MBZUAI based?",
            notes="Paraphrase of fact-001 for wording robustness.",
        ),
        _clone(
            by_id["fact-002"],
            new_id="fact-011",
            query="Whose name does MBZUAI carry?",
            notes="Paraphrase of fact-002 for named-entity robustness.",
        ),
        _clone(
            by_id["fact-004"],
            new_id="fact-012",
            query="Is parking available for students and visitors at MBZUAI?",
            notes="Paraphrase of fact-004 with broader parking wording.",
        ),
        _clone(
            by_id["fact-007"],
            new_id="fact-013",
            query="Is there a shuttle service for MBZUAI students?",
            notes="Paraphrase of fact-007 for transport wording.",
        ),
        _clone(
            by_id["fact-008"],
            new_id="fact-014",
            query="Are parents allowed to stay with students in MBZUAI accommodation?",
            notes="Paraphrase of fact-008 for accommodation phrasing.",
        ),
        _clone(
            by_id["scoped-001"],
            new_id="scoped-005",
            query="Describe the main AI specialization areas covered by MBZUAI graduate programs.",
            notes="Paraphrase of scoped-001 with broader academic wording.",
        ),
        _clone(
            by_id["scoped-002"],
            new_id="scoped-006",
            query="Explain MBZUAI's legal basis and institutional affiliation.",
            notes="Paraphrase of scoped-002 with legal-status wording.",
        ),
        _clone(
            by_id["scoped-003"],
            new_id="scoped-007",
            query="What student-facing campus services and facilities are available at MBZUAI?",
            notes="Paraphrase of scoped-003 focusing on student services.",
        ),
        _clone(
            by_id["synthesis-001"],
            new_id="synthesis-002",
            query="Give a practical orientation summary for a new MBZUAI student covering campus location, transport, accommodation, parking, and facilities.",
            notes="Paraphrase of synthesis-001 with orientation framing.",
        ),
        _clone(
            by_id["multimodal-001"],
            new_id="multimodal-004",
            query="What does the campus map indicate about where the main MBZUAI buildings and services are located?",
            notes="Paraphrase of multimodal-001 with building/service emphasis.",
        ),
        _clone(
            by_id["multimodal-002"],
            new_id="multimodal-005",
            query="Which campus services and facilities are explicitly labeled on the MBZUAI map?",
            notes="Paraphrase of multimodal-002 with services wording.",
        ),
        EvalExample(
            id="noanswer-003",
            query="What is the phone number of MBZUAI's London campus admissions office?",
            query_type="fact",
            source_type="none",
            no_answer=True,
            reference_answer="No grounded answer should be returned because the corpus does not contain this information.",
            notes="Additional abstention case for nonexistent overseas campus/contact details.",
            metadata={"paraphrase_of": "noanswer-001"},
        ),
    ]
    return [*base_examples, *additions]


def main() -> int:
    examples = build_v3_examples()
    write_eval_examples(OUTPUT_PATH, examples)
    print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
