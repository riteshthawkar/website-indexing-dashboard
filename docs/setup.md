# Setup Guide

This is the machine-setup guide for bringing the product up on a new workstation.

## 1. Prerequisites

- Python `3.10`
- Node.js `20+`
- npm `10+`
- network access to OpenAI, Gemini, Pinecone, and Neo4j

## 2. Clone And Configure

```bash
git clone <repo-url>
cd website-indexing-dashboard
cp .env.example .env
```

Fill in:

- `OPENAI_API_KEY`
- `GEMINI_API_KEY`
- `PINECONE_API_KEY`
- `NEO4J_URI`
- `NEO4J_DATABASE`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`

## 3. Bootstrap

```bash
bash scripts/bootstrap.sh
```

The bootstrap script performs:

- Python virtualenv creation at `env/`
- `pip install -r requirements.txt`
- `python -m playwright install chromium`
- `npm ci` in [`dashboard-ui/`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui)

If Playwright browser installation fails because the host is missing system packages, run:

```bash
env/bin/python -m playwright install-deps chromium
env/bin/python -m playwright install chromium
```

## 4. Start The Product

### Backend

```bash
bash scripts/dashboard.sh
```

This starts the FastAPI dashboard backend on `:8050`.

### Frontend

Development mode:

```bash
cd dashboard-ui
npm run dev -- --hostname 0.0.0.0 --port 3000
```

Production-like static serve:

```bash
cd dashboard-ui
npm run build
npx serve out -l 3000
```

The frontend is configured as a static export build in:

- [`dashboard-ui/next.config.ts`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/next.config.ts)

## 5. Default Product Config

There is a single user-facing launch config:

- [`pipeline/configs/default.yaml`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/pipeline/configs/default.yaml)

Important operational details:

- the dashboard uses `default` for launch
- the dashboard stores a run-local config snapshot
- each run executes its own saved snapshot, not the mutable file on disk

## 6. Core Commands

Validate config:

```bash
bash scripts/pipeline.sh validate-config --config default
```

Show stage plan:

```bash
bash scripts/pipeline.sh dry-run --config default
```

Run the full pipeline:

```bash
bash scripts/pipeline.sh run --config default
```

Run the retriever service:

```bash
bash scripts/pipeline.sh serve-retriever --config default --work-dir <run_work_dir> --host 0.0.0.0 --port 8663
```

## 7. Current Default External Stores

Default Pinecone indexes from [`pipeline/configs/default.yaml`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/pipeline/configs/default.yaml):

- `mbzuai-gemini-retrieval-v2`
- `mbzuai-gemini-retrieval-v2-sparse`

Neo4j namespace:

- generated per run unless overridden with `NEO4J_NAMESPACE` or `graph.neo4j_namespace`

## 8. Docs For Operators

- dashboard control plane: [`docs/dashboard_control_plane.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/dashboard_control_plane.md)
- retriever system: [`docs/retriever_system.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/retriever_system.md)
- structured logs: [`docs/structured_logs.md`](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/structured_logs.md)
