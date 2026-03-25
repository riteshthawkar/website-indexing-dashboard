from __future__ import annotations

from typing import Dict


EVALUATION_PRESETS: Dict[str, Dict[str, str]] = {
    "mbzuai-internal-retrieval": {
        "type": "internal",
        "focus": "Primary production gate for chunk/page/media retrieval on the MBZUAI corpus.",
        "datasets": "Custom gold set in eval/mbzuai_gold/*.jsonl",
        "metrics": "MRR@10, nDCG@10, Recall@10, parent hit@5, media hit@5, no-answer violation rate",
    },
    "beir": {
        "type": "external-text",
        "focus": "General zero-shot text retrieval sanity across diverse IR tasks.",
        "datasets": "BEIR benchmark suites",
        "metrics": "nDCG@10, Recall@100, MRR@10",
    },
    "miracl": {
        "type": "external-multilingual",
        "focus": "Multilingual retrieval robustness.",
        "datasets": "MIRACL benchmark",
        "metrics": "nDCG@10, Recall@100",
    },
    "vidore-v3": {
        "type": "external-multimodal",
        "focus": "Visually rich document retrieval for PDFs/pages.",
        "datasets": "ViDoRe V3 collections",
        "metrics": "page/document retrieval quality on visually rich corpora",
    },
    "ragas": {
        "type": "answer-quality",
        "focus": "Faithfulness, answer relevance, context precision/recall, noise sensitivity on your own eval set.",
        "datasets": "Internal generated-answer eval rows",
        "metrics": "faithfulness, answer_relevancy, context_precision, context_recall, noise_sensitivity",
    },
    "crag-rgb-ragtruth": {
        "type": "external-answer-quality",
        "focus": "End-to-end grounded answer quality and hallucination behavior.",
        "datasets": "CRAG, RGB, RAGTruth",
        "metrics": "answer correctness, groundedness, hallucination/unsupported claims",
    },
}
