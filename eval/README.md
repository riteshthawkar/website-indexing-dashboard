# Evaluation

This directory is the home for internal and external evaluation assets.

Recommended structure:

- `mbzuai_gold/`
  - Internal gold-set queries and judgments for the MBZUAI corpus.
- `gates/`
  - Threshold files used by CI or release checks.
- `predictions/`
  - Generated-answer rows for RAGAS or human-review workflows.

## Internal Retrieval Evaluation

Create a starter gold set:

```bash
./env/bin/python -m pipeline init-eval-set --output eval/mbzuai_gold/template.jsonl
```

Run retrieval evaluation:

```bash
./env/bin/python -m pipeline eval-retrieval \
  --config mbzuai_main_gemini_retrieval \
  --work-dir runs/mbzuai_main_processing/mbzuai_live_full_20260317 \
  --dataset eval/mbzuai_gold/template.jsonl \
  --gates eval/gates/retrieval_gate.example.json \
  --query-cache eval/cache/template.query_embeddings.json \
  --retrieval-cache eval/cache/template.retrieval_results.json \
  --parallelism 4 \
  --output eval/reports/retrieval_report.json
```

`--query-cache` is optional, but you should use it for repeated eval runs. It avoids paying Gemini query-embedding latency every time you rerun the same dataset.
`--retrieval-cache` is optional, but it is the bigger win once retrieval/rerank dominates runtime. It reuses prior per-query retrieval results for the same config/work-dir/query combination.
`--parallelism` is optional. Use it once your query cache is warm; it parallelizes retrieval work across queries.

Generate the expanded internal MBZUAI set:

```bash
./env/bin/python scripts/generate_mbzuai_internal_v3.py
./env/bin/python -m pipeline validate-eval-set \
  --dataset eval/mbzuai_gold/mbzuai_internal_v3.jsonl \
  --work-dir runs/mbzuai_main_processing/mbzuai_live_full_20260317 \
  --json
```

Generate the broader `v4` internal set:

```bash
./env/bin/python scripts/generate_mbzuai_internal_v4.py
./env/bin/python -m pipeline validate-eval-set \
  --dataset eval/mbzuai_gold/mbzuai_internal_v4.jsonl \
  --work-dir runs/mbzuai_main_processing/mbzuai_live_full_20260317 \
  --json
```

Run the stricter `v4` gate:

```bash
./env/bin/python -m pipeline eval-retrieval \
  --config mbzuai_main_gemini_retrieval \
  --work-dir runs/mbzuai_main_processing/mbzuai_live_full_20260317 \
  --dataset eval/mbzuai_gold/mbzuai_internal_v4.jsonl \
  --gates eval/gates/retrieval_gate.v4_strict.json \
  --retrieval-cache eval/cache/mbzuai_internal_v4.retrieval_results.json \
  --output /tmp/mbzuai_retrieval_report_v4.json
```

Summarize and validate an eval set:

```bash
./env/bin/python -m pipeline summarize-eval-set \
  --dataset eval/mbzuai_gold/mbzuai_internal_v2.jsonl

./env/bin/python -m pipeline validate-eval-set \
  --dataset eval/mbzuai_gold/mbzuai_internal_v2.jsonl \
  --work-dir runs/mbzuai_main_processing/mbzuai_live_full_20260317
```

## Gold-Set Row Format

Each row is JSON or JSONL with these fields:

- `id`
- `query`
- `query_type`: `fact`, `scoped`, `synthesis`, `multimodal`
- `source_type`: `webpage`, `pdf`, `mixed`, `image`, `video`, `none`
- `no_answer`
- `reference_answer`
- `gold_chunk_ids`
- `gold_parent_ids`
- `gold_media_ids`
- `notes`
- `metadata`

Use chunk, parent, and media ids from the indexed run so the evaluator can score:

- seed chunk quality
- expanded chunk quality
- parent/page hit rate
- media hit rate
- no-answer violation rate

## RAGAS

`pipeline eval-ragas` is optional. It requires `ragas` to be installed in the active venv.

Example:

```bash
./env/bin/python -m pipeline eval-ragas \
  --predictions eval/predictions/answers.jsonl \
  --metric faithfulness \
  --metric answer_relevancy \
  --metric context_precision \
  --metric context_recall
```

Prediction rows should include the fields expected by RAGAS, typically:

- `user_input`
- `response`
- `retrieved_contexts`
- `reference`
- `reference_contexts` when available

## External Benchmarks

Use `pipeline list-eval-presets` to see the recommended benchmark layers:

- internal MBZUAI gold set
- BEIR
- MIRACL
- ViDoRe V3
- RAGAS
- CRAG / RGB / RAGTruth

Export an `ir_datasets` benchmark subset in standard JSONL format:

```bash
./env/bin/python -m pipeline export-ir-benchmark \
  --dataset-id beir/scifact/test \
  --output-dir eval/benchmarks/scifact_subset \
  --max-queries 100
```

Export a Hugging Face retrieval benchmark using a mapping file:

```bash
./env/bin/python -m pipeline export-hf-benchmark \
  --mapping eval/benchmarks/huggingface_retrieval_mapping.template.json \
  --output-dir eval/benchmarks/hf_subset
```

Standard benchmark exports write:

- `corpus.jsonl`
- `queries.jsonl`
- `qrels.jsonl`
- `metadata.json`

You can score any rankings file against those qrels:

```bash
./env/bin/python -m pipeline eval-benchmark-rankings \
  --dataset-dir eval/benchmarks/scifact_subset \
  --rankings eval/benchmarks/scifact_subset/rankings.jsonl
```

You can also run the repo's Gemini hybrid benchmark retriever directly on a standard benchmark export:

```bash
./env/bin/python -m pipeline run-benchmark-retrieval \
  --config mbzuai_main_gemini_retrieval \
  --dataset-dir /tmp/scifact_subset_eval \
  --output-rankings /tmp/scifact_subset_eval/rankings.jsonl \
  --output /tmp/scifact_subset_eval/report.json \
  --json
```
