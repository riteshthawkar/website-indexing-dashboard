# MBZUAI Website Indexing Pipeline

This repository contains the terminal-first indexing pipeline for the MBZUAI chatbot knowledge base.

The production path scrapes the MBZUAI website, normalizes website and document content, creates assertion-first retrieval artifacts, uploads dense and sparse Pinecone namespaces scoped to one release, and promotes a local JSON semantic knowledge graph. Neo4j remains an optional connector for deployments that explicitly enable graph upload.

## Stack

- Python pipeline and CLI
- Crawl4AI, Trafilatura, MarkItDown, and Docling for website/document processing
- OpenAI for structured assertion extraction and validation
- Gemini embeddings for Pinecone upload
- Pinecone for vector retrieval
- Local JSON artifacts for the promoted semantic graph; optional Neo4j connector support

## Repository Layout

- [`pipeline/`](pipeline/): pipeline stages, retrieval, evaluation, CLI
- [`pipeline/configs/mbzuai_production.yaml`](pipeline/configs/mbzuai_production.yaml): canonical production contract
- [`pipeline/configs/default.yaml`](pipeline/configs/default.yaml): shared defaults inherited by environment-specific configs
- [`docs/`](docs/): operator and system docs
- [`scripts/`](scripts/): bootstrap and CLI helpers

## Prerequisites

- Linux or macOS workstation
- Python `3.10` to `3.12`
- outbound access to OpenAI, Google Gemini, and Pinecone

## Setup

```bash
cp .env.example .env
bash scripts/bootstrap.sh
```

Fill in these required values in `.env` before a production run:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `PINECONE_API_KEY`

Neo4j values are optional and only needed when `graph.store_backend: neo4j` is configured.

## Terminal Workflow

Run the production preflight first:

```bash
bash scripts/pipeline.sh doctor --config mbzuai_production
```

Show the exact stage order:

```bash
bash scripts/pipeline.sh dry-run --config mbzuai_production
```

Validate registered stages and required dependencies:

```bash
bash scripts/pipeline.sh validate-config --config mbzuai_production
```

Run the production pipeline:

```bash
bash scripts/pipeline.sh run --config mbzuai_production
```

Run through one stage and persist a resumable checkpoint for inspection:

```bash
bash scripts/pipeline.sh run --config mbzuai_production \
  --run-id <run_id> \
  --stop-after-stage <stage_id>
```

Continue the same run through the next checkpoint:

```bash
bash scripts/pipeline.sh run --config mbzuai_production \
  --run-id <run_id> \
  --resume \
  --stop-after-stage <next_stage_id>
```

Resume a run:

```bash
bash scripts/pipeline.sh run --config mbzuai_production --resume --run-id <run_id>
```

Restart from a stage:

```bash
bash scripts/pipeline.sh run --config mbzuai_production --restart-from-stage <stage_id> --run-id <run_id>
```

Audit a completed run:

```bash
bash scripts/pipeline.sh audit-run --work-dir runs/<project>/<run_id>
```

Run a retrieval smoke query:

```bash
bash scripts/pipeline.sh retrieve --work-dir runs/<project>/<run_id> --query "Who is the president of MBZUAI?" --trace
```

Start the retriever HTTP service for agents:

```bash
bash scripts/pipeline.sh serve-retriever --work-dir runs/<project>/<run_id> --host 127.0.0.1 --port 8663
```

Migrate an existing indexed run into the current v2 retrieval contract without re-scraping:

```bash
bash scripts/pipeline.sh migrate-release \
  --config mbzuai_production \
  --source-work-dir runs/<project>/<old_run_id> \
  --target-work-dir runs/<project>/<old_run_id>-v2-migrated \
  --dry-run

bash scripts/pipeline.sh migrate-release \
  --config mbzuai_production \
  --source-work-dir runs/<project>/<old_run_id> \
  --target-work-dir runs/<project>/<old_run_id>-v2-migrated \
  --force
```

Upload or resume a migrated run to the configured v3 Pinecone indexes:

```bash
bash scripts/pipeline.sh migrate-release \
  --config mbzuai_production \
  --source-work-dir runs/<project>/<old_run_id> \
  --target-work-dir runs/<project>/<old_run_id>-v2-migrated \
  --upload-existing \
  --embed-batch-size 96 \
  --media-text-batch-size 96
```

Gate and promote a completed retrieval release through the protected release-check flow. The protected control plane must supply the exact candidate commits, candidate service URLs, operations token, and shared retrieval-service token:

```bash
RELEASE_WORK_DIR=/data/releases/runs/mbzuai_main/$RUN_ID \
RELEASE_COMMIT_SHA="$RETRIEVER_COMMIT_SHA" \
CANDIDATE_BACKEND_COMMIT_SHA="$BACKEND_COMMIT_SHA" \
CANDIDATE_BACKEND_OPERATIONS_TOKEN="$OPERATIONS_TOKEN" \
RETRIEVAL_SERVICE_TOKEN="$RETRIEVER_TOKEN" \
ANSWER_EVAL_ENDPOINT=ws://backend-candidate:8080/chat \
CANDIDATE_BACKEND_DETAILED_URL=http://backend-candidate:8080/health/detailed \
CANDIDATE_RETRIEVER_ATTESTATION_URL=http://retriever-candidate:8060/attestationz \
bash scripts/deploy/release-check-promote.sh

bash scripts/pipeline.sh show-release --config mbzuai_production
```

Do not bypass the protected wrapper with the lower-level promotion subcommand; the wrapper validates immutable artifacts, both candidate identities, the canonical retrieval and answer gates, and the locked active-release pointer.

## Production Checks

The default config enables:

- assertion-first extraction before retrieval formatting
- Pinecone dense and sparse uploads under release-scoped `chunks`, `parents`, `media`, `facts`, `evidence_spans`, `summaries`, `assertions`, `entities`, and `communities` lanes
- post-upload Pinecone namespace count verification
- guarded v2 migration from existing runs, including summary/assertion/entity enrichment and no-regression record-count checks
- local promoted graph artifact verification
- optional Neo4j graph upload and node/edge verification when `graph.store_backend: neo4j` is configured
- release manifests with retrieval eval gates and an active release pointer
- retrieval confidence, routing traces, and bounded evidence packs for terminal and agent use
- stage and run audits that fail the run on integrity errors

Default stores:

- Pinecone dense index: `mbzuai-gemini-retrieval-v3`
- Pinecone sparse index: `mbzuai-gemini-retrieval-v3-sparse`
- Knowledge graph store: promoted local JSON artifact under the run directory

## Useful Config Overrides

Any config value can be overridden through environment variables using `PIPELINE_<SECTION>__<KEY>`.

Examples:

```bash
PIPELINE_CRAWLER__MAX_PAGES=500 bash scripts/pipeline.sh run
PIPELINE_GRAPH__NEO4J_NAMESPACE=mbzuai-prod-2026-05 bash scripts/pipeline.sh run
PIPELINE_EMBEDDER__VERIFY_INDEX_MIN_COUNT_ONLY=true bash scripts/pipeline.sh run
```

## Key Docs

- setup guide: [`docs/setup.md`](docs/setup.md)
- retriever handoff: [`docs/retriever_system.md`](docs/retriever_system.md)
