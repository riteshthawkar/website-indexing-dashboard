from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List


ALLOWED_QUERY_TYPES = {"fact", "scoped", "synthesis", "multimodal"}
ALLOWED_SOURCE_TYPES = {"webpage", "pdf", "mixed", "image", "video", "none"}


@dataclass
class EvalExample:
    id: str
    query: str
    query_type: str
    source_type: str = "mixed"
    no_answer: bool = False
    reference_answer: str = ""
    gold_chunk_ids: List[str] = field(default_factory=list)
    gold_span_ids: List[str] = field(default_factory=list)
    gold_parent_ids: List[str] = field(default_factory=list)
    gold_media_ids: List[str] = field(default_factory=list)
    notes: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalized(self) -> "EvalExample":
        query_type = str(self.query_type or "").strip().lower()
        source_type = str(self.source_type or "").strip().lower()
        if query_type not in ALLOWED_QUERY_TYPES:
            raise ValueError(f"Unsupported query_type {self.query_type!r} for eval example {self.id}")
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"Unsupported source_type {self.source_type!r} for eval example {self.id}")
        if not str(self.id or "").strip():
            raise ValueError("Eval example id is required")
        if not str(self.query or "").strip():
            raise ValueError(f"Eval example {self.id} is missing query text")
        return EvalExample(
            id=str(self.id).strip(),
            query=str(self.query).strip(),
            query_type=query_type,
            source_type=source_type,
            no_answer=bool(self.no_answer),
            reference_answer=str(self.reference_answer or "").strip(),
            gold_chunk_ids=[str(item).strip() for item in self.gold_chunk_ids if str(item).strip()],
            gold_span_ids=[str(item).strip() for item in self.gold_span_ids if str(item).strip()],
            gold_parent_ids=[str(item).strip() for item in self.gold_parent_ids if str(item).strip()],
            gold_media_ids=[str(item).strip() for item in self.gold_media_ids if str(item).strip()],
            notes=str(self.notes or "").strip(),
            metadata=dict(self.metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.normalized())


def _load_raw_examples(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".jsonl":
        payload = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        return payload
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected list or JSONL payload in {path}")


def load_eval_examples(path: str | Path) -> List[EvalExample]:
    raw_examples = _load_raw_examples(path)
    examples = [EvalExample(**dict(item or {})).normalized() for item in raw_examples]
    seen = set()
    for example in examples:
        if example.id in seen:
            raise ValueError(f"Duplicate eval example id: {example.id}")
        seen.add(example.id)
    return examples


def write_eval_examples(path: str | Path, examples: Iterable[EvalExample]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".jsonl":
        lines = [json.dumps(example.to_dict(), ensure_ascii=True) for example in examples]
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return
    payload = [example.to_dict() for example in examples]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def mbzuai_eval_template() -> List[EvalExample]:
    return [
        EvalExample(
            id="fact-001",
            query="Where is MBZUAI located?",
            query_type="fact",
            source_type="webpage",
            reference_answer="MBZUAI is located in Masdar City, Abu Dhabi, United Arab Emirates.",
            gold_chunk_ids=["replace-with-gold-chunk-id"],
            gold_parent_ids=["replace-with-gold-page-or-section-id"],
            notes="Short factual query. Use precise supporting chunk ids from the indexed run.",
        ),
        EvalExample(
            id="scoped-001",
            query="Summarize the MBZUAI admissions requirements for graduate applicants.",
            query_type="scoped",
            source_type="mixed",
            reference_answer="Add a concise gold answer here.",
            gold_chunk_ids=["replace-with-gold-chunk-id-1", "replace-with-gold-chunk-id-2"],
            gold_parent_ids=["replace-with-gold-section-id"],
            notes="Scoped multi-evidence query that should prefer section expansion over full-page expansion.",
        ),
        EvalExample(
            id="synthesis-001",
            query="Explain MBZUAI's mission, location, and academic focus in detail.",
            query_type="synthesis",
            source_type="mixed",
            reference_answer="Add a fuller reference answer here.",
            gold_chunk_ids=["replace-with-gold-chunk-id-1", "replace-with-gold-chunk-id-2", "replace-with-gold-chunk-id-3"],
            gold_parent_ids=["replace-with-gold-parent-id-1", "replace-with-gold-parent-id-2"],
            notes="Broad synthesis question spanning multiple sources.",
        ),
        EvalExample(
            id="multimodal-001",
            query="What does the MBZUAI campus map show about the campus layout?",
            query_type="multimodal",
            source_type="pdf",
            reference_answer="Add a grounded answer tied to the campus map PDF and extracted images.",
            gold_chunk_ids=["replace-with-gold-chunk-id"],
            gold_parent_ids=["replace-with-gold-page-id"],
            gold_media_ids=["replace-with-gold-media-id"],
            notes="Use at least one PDF page/image media id here.",
        ),
        EvalExample(
            id="noanswer-001",
            query="What is MBZUAI's office in Singapore phone number?",
            query_type="fact",
            source_type="none",
            no_answer=True,
            reference_answer="No grounded answer should be returned because the corpus does not contain this information.",
            notes="No-answer / abstention case.",
        ),
    ]
