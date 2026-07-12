# Setup Guide

This is the operator setup guide for the terminal-first MBZUAI indexing pipeline.

## 1. Prerequisites

- Python `3.10` to `3.12`
- network access to OpenAI, Gemini, and Pinecone

## 2. Configure Secrets

```bash
cp .env.example .env
```

Required for the production config:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `PINECONE_API_KEY`

Optional:

- `NEO4J_URI`, `NEO4J_DATABASE`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, and `NEO4J_NAMESPACE` only when `graph.store_backend: neo4j` is enabled
- `FASTTEXT_LID_MODEL` for local language detection
- `PIPELINE_<SECTION>__<KEY>` overrides for any YAML config value

## 3. Bootstrap

```bash
bash scripts/bootstrap.sh
```

The bootstrap script creates `env/`, installs Python dependencies from `requirements.txt`, and installs Chromium for Crawl4AI/Playwright.

If browser installation fails because system packages are missing:

```bash
env/bin/python -m playwright install-deps chromium
env/bin/python -m playwright install chromium
```

## 4. Preflight

Run this before a production crawl:

```bash
bash scripts/pipeline.sh doctor
```

The doctor command checks:

- required assertion-first stage order
- OpenAI, Gemini, and Pinecone credentials
- Pinecone index and namespace targets
- Pinecone post-upload verification
- local promoted graph artifact verification
- Neo4j credentials and post-upload verification only when configured as the graph store
- stage registration and stage-level config validation

## 5. Core Commands

```bash
bash scripts/pipeline.sh dry-run
bash scripts/pipeline.sh validate-config
bash scripts/pipeline.sh run
```

Resume a run:

```bash
bash scripts/pipeline.sh run --resume --run-id <run_id>
```

Restart from a stage:

```bash
bash scripts/pipeline.sh run --restart-from-stage <stage_id> --run-id <run_id>
```

Audit run artifacts:

```bash
bash scripts/pipeline.sh audit-run --work-dir runs/<project>/<run_id>
```

Run a retrieval smoke query:

```bash
bash scripts/pipeline.sh retrieve --work-dir runs/<project>/<run_id> --query "What programs does MBZUAI offer?"
```

Serve retrieval over HTTP for agents:

```bash
bash scripts/pipeline.sh serve-retriever --work-dir runs/<project>/<run_id> --host 127.0.0.1 --port 8663
```

Migrate a trusted existing run to the current v2 retrieval contract without re-scraping:

```bash
bash scripts/pipeline.sh migrate-release \
  --config mbzuai_main \
  --source-work-dir runs/<project>/<old_run_id> \
  --target-work-dir runs/<project>/<old_run_id>-v2-migrated \
  --dry-run

bash scripts/pipeline.sh migrate-release \
  --config mbzuai_main \
  --source-work-dir runs/<project>/<old_run_id> \
  --target-work-dir runs/<project>/<old_run_id>-v2-migrated \
  --force
```

Resume Pinecone upload for a migrated run:

```bash
bash scripts/pipeline.sh migrate-release \
  --config mbzuai_main \
  --source-work-dir runs/<project>/<old_run_id> \
  --target-work-dir runs/<project>/<old_run_id>-v2-migrated \
  --upload-existing \
  --embed-batch-size 96 \
  --media-text-batch-size 96
```

## 6. Default External Stores

Default Pinecone indexes:

- `mbzuai-gemini-retrieval-v3`
- `mbzuai-gemini-retrieval-v3-sparse`

Canonical Pinecone namespace bases:

- `mbzuai_main-chunks`
- `mbzuai_main-parents`
- `mbzuai_main-media`
- `mbzuai_main-facts`
- `mbzuai_main-evidence-spans`
- `mbzuai_main-summaries`
- `mbzuai_main-assertions`
- `mbzuai_main-entities`
- `mbzuai_main-communities`

Production uploads append `--<release_id>` to every base. The retriever resolves the exact dense and sparse namespace names from the promoted run's `index_upload_manifest.json`; operators must not hard-code or reconstruct them.

Default knowledge graph store:

- promoted local JSON graph artifact under `runs/<project>/<run_id>/stage_outputs/promote_graph/`

Optional Neo4j namespace:

- generated per run unless `graph.store_backend: neo4j` is enabled and overridden with `NEO4J_NAMESPACE` or `graph.neo4j_namespace`
