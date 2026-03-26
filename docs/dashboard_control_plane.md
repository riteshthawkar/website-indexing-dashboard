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
- retrieval benchmark launcher for a run
- benchmark job listing with live refresh
- report path and headline metrics
- gate pass/fail status

Backend endpoints:
- `GET /api/evaluation/assets`
- `GET /api/runs/{run_id}/benchmarks/retrieval`
- `POST /api/runs/{run_id}/benchmarks/retrieval`

### `Knowledge`
- Pinecone dense/sparse index names and live stats
- namespace inspection
- retrieval bundle record counts
- Neo4j upload manifest / namespace / graph counts

Backend endpoint:
- `GET /api/runs/{run_id}/knowledge-base`

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

### Structured log storage
- [structured_logs.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/structured_logs.py)

### Retrieval control-plane helpers
- [retrieval_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/retrieval_ops.py)

### Benchmark control-plane helpers
- [evaluation_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/evaluation_ops.py)

### Knowledge-base inspection helpers
- [knowledge_ops.py](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard/knowledge_ops.py)

## Frontend Components

Run detail page:
- [client.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/app/projects/detail/client.tsx)

New operational tabs:
- [retrieval-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/retrieval-tab.tsx)
- [evaluation-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/evaluation-tab.tsx)
- [knowledge-tab.tsx](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/components/project-detail/knowledge-tab.tsx)

API client/types:
- [api.ts](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/lib/api.ts)
- [types.ts](/home/fahadkhan/ritesh/Final-MBZUAI-vectorstore/dashboard-ui/lib/types.ts)

## Operational Notes

- The dashboard API now loads repo `.env` values during app startup so retrieval and benchmark operations can access model and Pinecone credentials.
- Retrieval playground queries are slower on first use because they instantiate and warm the retriever against the indexed run.
- Knowledge-base status prefers the actual run upload manifests when present, instead of trusting imported run metadata.
- Benchmark jobs persist manifest JSON files under:
  - `<work_dir>/dashboard_reports/retrieval_benchmarks`

## What Still Does Not Exist

The dashboard is broader now, but it is still not a complete platform admin surface. Missing pieces include:
- retriever service start/stop/health management
- stage retry / rerun / resume controls
- artifact browser across all stage outputs
- Neo4j query explorer
- assertion candidate/promoted assertion browser
- scheduled jobs / refresh automation
- secrets management

Those should be implemented as additional control-plane layers, not mixed into the current run detail tabs ad hoc.
