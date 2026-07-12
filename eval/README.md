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

## Release Readiness Evaluation

Production promotion uses the governed LLM-generated v1 suite and strict gates by default:

- `eval/mbzuai_gold/mbzuai_llm_generated_v1.jsonl`
- `eval/gates/retrieval_gate.v5_span_strict.json`
- `eval/gates/answer_readiness_gate.llm_generated_v1.json`

The production policy requires at least 50 retrieval queries and 50 judged answers. The release-readiness v2 suite remains available as a supplemental regression set for PDF-derived, multi-document, and multimodal coverage, but it is not the promotion policy.

The v2 suite is generated from the internal v4 gold set with:

```bash
./env/bin/python scripts/generate_mbzuai_release_readiness_v2.py
```

You can also generate a fresh LLM-assisted QA suite from a completed indexed run. This uses sampled retrieval-bundle evidence plus the provider's web-search grounding tool, then writes evaluation rows with reference answers, expected response structure, required coverage, source/citation requirements, expected follow-up topics, suggested actions, and gold ids from the local bundle:

```bash
GOOGLE_API_KEY=... ./env/bin/python scripts/generate_llm_qa_eval_set.py \
  --provider gemini \
  --model gemini-2.5-flash \
  --work-dir runs/mbzuai_main/<run_id> \
  --count 50 \
  --output eval/mbzuai_gold/mbzuai_llm_generated_v1.jsonl \
  --manifest eval/mbzuai_gold/mbzuai_llm_generated_v1.manifest.json
```

OpenAI is also supported:

```bash
OPENAI_API_KEY=... ./env/bin/python scripts/generate_llm_qa_eval_set.py \
  --provider openai \
  --model gpt-5-mini \
  --work-dir runs/mbzuai_main/<run_id> \
  --count 50 \
  --output eval/mbzuai_gold/mbzuai_llm_generated_v1.jsonl
```

Review generated rows before using them as a promotion gate. The script constrains gold ids to the local retrieval bundle, but an LLM-generated reference answer should still be spot-checked against official source pages/PDFs before it becomes a release benchmark.

Run the full release gate against a completed candidate run:

```bash
./env/bin/python -m pipeline release-check \
  --config pipeline/configs/default.yaml \
  --work-dir runs/mbzuai_indexing/<run_id> \
  --promote
```

`release-check` now gates both retrieval quality and generated-answer readiness. Production release checks default to the live HTTP chat path, so start the backend against the candidate retrieval configuration and pass `--answer-endpoint` or set `MBZUAI_CHAT_EVAL_ENDPOINT`. Use `--answer-eval-mode local` only for indexing-side dry runs.

Generated-answer readiness uses an LLM-as-judge by default for production release checks. The judge evaluates the final response plus supporting evidence, references/sources, expected reference URLs, citation requirements, response structure, UI payload, injected components, suggested actions, follow-up questions, and response contract fields. Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` before running a production release gate.

To exercise the exact live widget/chatbot HTTP path, start the backend against the candidate retrieval configuration, then run:

```bash
./env/bin/python -m pipeline release-check \
  --config pipeline/configs/default.yaml \
  --work-dir runs/mbzuai_indexing/<run_id> \
  --answer-endpoint http://127.0.0.1:8000/telegram-chat \
  --answer-auth-token "$OPERATIONS_API_TOKEN" \
  --promote
```

The HTTP evaluator does not send `X-Health-Probe` by default because probe mode can intentionally bypass normal user-facing responses. Use `--answer-probe-mode` only when you explicitly want a health-probe style check instead of production answer grading.

To grade the LLM-generated 50-query suite, use its matching answer gate:

```bash
./env/bin/python -m pipeline eval-answer-readiness \
  --config pipeline/configs/default.yaml \
  --work-dir runs/mbzuai_indexing/<run_id> \
  --dataset eval/mbzuai_gold/mbzuai_llm_generated_v1.jsonl \
  --gates eval/gates/answer_readiness_gate.llm_generated_v1.json \
  --mode http \
  --endpoint http://127.0.0.1:8000/telegram-chat \
  --output eval/reports/answer_readiness_llm_generated_v1_report.json
```

For local deterministic dry runs only, you can skip the LLM judge, but the default production answer gate will still fail because it requires judge metrics. Use a non-LLM gate file for deterministic-only checks:

```bash
./env/bin/python -m pipeline release-check \
  --config pipeline/configs/default.yaml \
  --work-dir runs/mbzuai_indexing/<run_id> \
  --answer-gates eval/gates/answer_readiness_deterministic_only.json \
  --skip-llm-judge
```

You can run only the generated-answer readiness check with:

```bash
./env/bin/python -m pipeline eval-answer-readiness \
  --config pipeline/configs/default.yaml \
  --work-dir runs/mbzuai_indexing/<run_id> \
  --dataset eval/mbzuai_gold/mbzuai_release_readiness_v2.jsonl \
  --gates eval/gates/answer_readiness_gate.v2.json \
  --output eval/reports/answer_readiness_report.json
```

That command also runs the LLM judge by default. Use `--skip-llm-judge` only with non-production gates when checking deterministic include/exclude rules without production promotion confidence.

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
