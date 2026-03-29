# Website Indexing Dashboard

This repository contains the full website indexing product:

- the modular scraping and indexing pipeline
- the assertion-first Pinecone + Neo4j knowledge-base build
- the React dashboard control plane
- the retrieval and evaluation tooling

The product is launched from a single user-facing config file:

- [`pipeline/configs/default.yaml`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/pipeline/configs/default.yaml)

The dashboard creates a per-run snapshot of that config and executes the snapshot, so launch-time edits are reproducible.

## Stack

- Python pipeline and dashboard backend
- Next.js + shadcn/ui dashboard frontend
- OpenAI for structured assertion extraction, validation, planning, and adjudication
- Gemini embeddings for Pinecone upload
- Pinecone for vector retrieval
- Neo4j for the promoted semantic graph

## Repository Layout

- [`pipeline/`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/pipeline): pipeline stages, retrieval, evaluation, CLI
- [`dashboard/`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard): FastAPI control plane and worker management
- [`dashboard-ui/`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui): React dashboard UI
- [`docs/`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs): product and system docs
- [`scripts/`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/scripts): local bootstrap and launch helpers

## Prerequisites

- Linux or macOS workstation
- Python `3.10`
- Node.js `20+`
- npm `10+`
- outbound access to:
  - OpenAI
  - Google Gemini
  - Pinecone
  - Neo4j

## Setup

1. Create the environment file.

```bash
cp .env.example .env
```

2. Fill in the required secrets in `.env`.

3. Bootstrap the workstation.

```bash
bash scripts/bootstrap.sh
```

This script:

- creates `env/` if it does not exist
- installs Python dependencies from [`requirements.txt`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/requirements.txt)
- installs Chromium for Crawl4AI / Playwright
- installs frontend dependencies with `npm ci`

## Start The Dashboard

Backend:

```bash
bash scripts/dashboard.sh
```

Frontend development server:

```bash
cd dashboard-ui
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Open:

- `http://127.0.0.1:3000`

Production-like local frontend serve:

```bash
cd dashboard-ui
npm run build
npx serve out -l 3000
```

## Run The Pipeline From CLI

Validate the current product config:

```bash
bash scripts/pipeline.sh validate-config --config default
```

Show the planned stages:

```bash
bash scripts/pipeline.sh dry-run --config default
```

Run the pipeline:

```bash
bash scripts/pipeline.sh run --config default
```

Resume a run:

```bash
bash scripts/pipeline.sh run --config default --resume --run-id <run_id>
```

Restart from a stage:

```bash
bash scripts/pipeline.sh run --config default --restart-from-stage <stage_id> --run-id <run_id>
```

## Retriever Service

Start the long-lived retriever service for a completed run:

```bash
bash scripts/pipeline.sh serve-retriever --config default --work-dir <run_work_dir> --host 0.0.0.0 --port 8663
```

The dashboard can also start and stop the retriever service from the run detail page.

## Required Environment Variables

Required for the default product path:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `NEO4J_URI`
- `NEO4J_DATABASE`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

Optional but supported:

- `OPENAI_TIMEOUT_SEC`
- `NEO4J_NAMESPACE`
- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_WS_URL`
- `FASTTEXT_LID_MODEL`
- any config override using `PIPELINE_<SECTION>__<KEY>`

See:

- [`.env.example`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/.env.example)
- [`docs/setup.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/setup.md)

## Default Stores

The default config currently targets:

- Pinecone dense index: `mbzuai-gemini-retrieval-v2`
- Pinecone sparse index: `mbzuai-gemini-retrieval-v2-sparse`

Neo4j namespace is per-run unless explicitly overridden.

## Key Docs

- setup guide: [`docs/setup.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/setup.md)
- dashboard control plane: [`docs/dashboard_control_plane.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/dashboard_control_plane.md)
- retriever system handoff: [`docs/retriever_system.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/retriever_system.md)
- structured logs: [`docs/structured_logs.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/structured_logs.md)
