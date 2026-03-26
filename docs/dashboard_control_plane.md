# Dashboard Control Plane

This dashboard is the operational surface for the MBZUAI retrieval pipeline. It now manages more than run execution and logs. The run detail page exposes the main control-plane functions needed to inspect, query, and evaluate an indexed run.

## Current Run Detail Tabs

### `Overview`
- run metadata
- artifact counts
- media counts
- ingestion/indexing totals

### `URLs`
- scraped URLs
- skipped URLs
- indexed URL listing

### `Media`
- extracted images and videos for the run

### `Stages`
- stage state from `pipeline_state.json`
- per-stage metrics
- stage outputs summary

### `Operations`
- resume a run from existing `pipeline_state.json`
- restart a run from a selected stage
- retry a selected stage
- validate the current config
- dry-run the current config to inspect the execution plan
- audit the run work directory and optionally repair artifact references
- start, stop, and inspect the local long-lived retriever service for the run
- inspect available evaluation presets

Backend endpoints:
- `POST /api/runs/{run_id}/resume`
- `POST /api/runs/{run_id}/restart`
- `POST /api/runs/{run_id}/stages/{stage_selector}/retry`
- `GET /api/configs/{config_name}/validate`
- `GET /api/configs/{config_name}/dry-run`
- `GET /api/runs/{run_id}/audit`
- `GET /api/evaluation/presets`
- `GET /api/runs/{run_id}/retriever-service`
- `POST /api/runs/{run_id}/retriever-service/start`
- `POST /api/runs/{run_id}/retriever-service/stop`

### `Retrieval`
- live retrieval playground against the selected run
- config selection for the retriever
- bounded answer preview from structured answer records
- raw retrieval payload inspection
- top evidence inspection

Backend endpoint:
- `POST /api/runs/{run_id}/retrieve`

Request body:
```json
{
  "config_name": "mbzuai_main_openai_routed_retrieval",
  "query": "Who is the president of MBZUAI?"
}
```

### `Evaluation`
- benchmark asset discovery
- eval dataset template generation
- eval dataset summarize / validate actions
- run-scoped retrieval benchmark launcher
- grounded answer generation launcher
- RAGAS launcher
- standard benchmark export helpers:
  - `ir_datasets`
  - Hugging Face mapping exports
- standard benchmark summarize action
- standard benchmark retrieval launcher
- benchmark ranking evaluation launcher
- evaluation job listing with live refresh
- evaluation job cancellation
- report path and headline metrics

Backend endpoints:
- `GET /api/evaluation/assets`
- `GET /api/runs/{run_id}/evaluation/assets`
- `POST /api/evaluation/datasets/init`
- `POST /api/evaluation/datasets/summarize`
- `POST /api/evaluation/datasets/validate`
- `POST /api/evaluation/benchmarks/summarize`
- `GET /api/runs/{run_id}/benchmarks/retrieval`
- `POST /api/runs/{run_id}/benchmarks/retrieval`
- `GET /api/runs/{run_id}/evaluation/jobs`
- `POST /api/runs/{run_id}/evaluation/jobs/{job_id}/cancel`
- `POST /api/runs/{run_id}/evaluation/answers`
- `POST /api/runs/{run_id}/evaluation/ragas`
- `POST /api/runs/{run_id}/evaluation/benchmarks/run-standard-retrieval`
- `POST /api/runs/{run_id}/evaluation/benchmarks/evaluate-rankings`
- `POST /api/runs/{run_id}/evaluation/benchmarks/export-ir`
- `POST /api/runs/{run_id}/evaluation/benchmarks/export-hf`

### `Knowledge`
- Pinecone dense/sparse index names and live stats
- namespace inspection
- Pinecone snapshot action
- delete-by-source maintenance action
- retrieval bundle record counts
- assertion layer counts
- assertion browser across candidate/validated/canonical/promoted/quarantined layers
- Neo4j upload manifest / namespace / graph counts

Backend endpoint:
- `GET /api/runs/{run_id}/knowledge-base`
- `GET /api/runs/{run_id}/knowledge-base/assertions`
- `POST /api/indexes/{index_name}/delete-by-source`

### `Artifacts`
- artifact catalog browser with filters by type, role, and producer stage
- run-local file browser rooted at the run work directory
- text/JSON preview
- image/video preview through the dashboard asset endpoint

Backend endpoints:
- `GET /api/runs/{run_id}/artifacts`
- `GET /api/runs/{run_id}/files`
- `GET /api/runs/{run_id}/file-content`

### `Logs`
- structured log stream
- server-side pagination
- filters for level, event type, and stage
- free-text search

See also:
- [structured_logs.md](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/docs/structured_logs.md)

### `Config`
- effective config snapshot or current config view

## Backend Modules

### Dashboard API
- [app.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/app.py)

### Run execution and env loading
- [run_executor.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/run_executor.py)

### Run-control helpers
- [control_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/control_ops.py)

### Retriever service lifecycle helpers
- [retriever_service_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/retriever_service_ops.py)

### Structured log storage
- [structured_logs.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/structured_logs.py)

### Retrieval control-plane helpers
- [retrieval_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/retrieval_ops.py)

### Benchmark control-plane helpers
- [evaluation_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/evaluation_ops.py)

### Knowledge-base inspection helpers
- [knowledge_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/knowledge_ops.py)

### Artifact browser helpers
- [artifact_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/artifact_ops.py)

## Frontend Components

Run detail page:
- [client.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/app/projects/detail/client.tsx)

New operational tabs:
- [operations-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/operations-tab.tsx)
- [retrieval-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/retrieval-tab.tsx)
- [evaluation-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/evaluation-tab.tsx)
- [knowledge-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/knowledge-tab.tsx)
- [artifacts-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/artifacts-tab.tsx)

API client/types:
- [api.ts](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/lib/api.ts)
- [types.ts](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/lib/types.ts)

## Operational Notes

- The dashboard API now loads repo `.env` values during app startup so retrieval and benchmark operations can access model and Pinecone credentials.
- Retrieval playground queries are slower on first use because they instantiate and warm the retriever against the indexed run.
- Knowledge-base status prefers the actual run upload manifests when present, instead of trusting imported run metadata.
- Benchmark jobs persist manifest JSON files under:
  - `<work_dir>/dashboard_reports/retrieval_benchmarks`
- Expanded evaluation jobs persist manifest JSON files under:
  - `<work_dir>/dashboard_reports/evaluation_jobs`
- Retriever service manifests and logs persist under:
  - `<work_dir>/dashboard_services`
- The local retriever service controls are suitable for workstation use. They are not a durable distributed process supervisor.

## What Still Does Not Exist

The dashboard is broader now, but it is still not a complete platform admin surface. Missing pieces include:
- Neo4j query explorer
- broader Pinecone / Neo4j maintenance actions beyond snapshotting and delete-by-source
- broader graph maintenance workflows beyond stage restarts and manifest inspection
- richer artifact previews for PDFs and structured diffs
- durable worker/process orchestration across dashboard backend restarts
- scheduled jobs / refresh automation
- secrets management

Those should be implemented as additional control-plane layers, not mixed into the current run detail tabs ad hoc.
