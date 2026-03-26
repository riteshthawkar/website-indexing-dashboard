# Structured Logs

## Purpose

The dashboard now exposes a structured per-run log stream instead of only line-oriented terminal output.

These logs are intended for:

- dashboard inspection
- downstream automation
- run debugging
- stage-level audit trails

## Storage

For each run, the dashboard writes a JSONL file at:

`<work_dir>/dashboard_logs/structured_logs.jsonl`

Example:

`runs/mbzuai_main_processing/run_42/dashboard_logs/structured_logs.jsonl`

Each line is one JSON object.

## Record Schema

Each record has this shape:

```json
{
  "sequence": 17,
  "run_id": 42,
  "pipeline_run_id": "run_42",
  "created_at": "2026-03-25T18:01:22.123456+00:00",
  "level": "info",
  "event_type": "stage_complete",
  "stage": "formatter/gemini_retrieval",
  "message": "Stage formatter/gemini_retrieval completed: {...}",
  "data": {
    "status": "completed",
    "metrics": {
      "documents_total": 1200
    }
  }
}
```

## Field Semantics

- `sequence`: monotonic per-run sequence number
- `run_id`: dashboard database run id
- `pipeline_run_id`: pipeline execution id used in the work directory
- `created_at`: UTC timestamp in ISO-8601
- `level`: `info`, `warning`, `error`, or other log severity
- `event_type`: structured event kind
- `stage`: stage key such as `chunker/hybrid_chunker`
- `message`: human-readable summary
- `data`: structured payload for the event

## Event Types

Current event types:

- `log`
- `stage_start`
- `stage_complete`

The `data` field carries event-specific payloads:

- `log`: usually empty unless the caller attaches structured metadata
- `stage_start`: stage info payload
- `stage_complete`: full stage completion payload including metrics and outputs when available

## Dashboard API

Structured logs are exposed at:

`GET /api/runs/{run_id}/structured-logs`

Query params:

- `limit`: number of records to return, default `200`
- `tail`: backward-compatible alias for `limit`
- `before_sequence`: optional cursor for older records
- `stage`: optional exact stage filter
- `event_type`: optional exact event-type filter
- `level`: optional exact severity filter

Response:

```json
{
  "items": [...],
  "path": "/abs/path/to/runs/.../dashboard_logs/structured_logs.jsonl",
  "has_more": true,
  "next_before_sequence": 482
}
```

Pagination semantics:

- records are returned in ascending `sequence` order within each page
- to fetch older records, call the same endpoint with:
  - `before_sequence = next_before_sequence`
- if `has_more` is `false`, there are no older matching records

## WebSocket Behavior

The existing WebSocket endpoint remains:

`/ws/runs/{run_id}/logs`

It now includes a `record` field on pushed messages so the UI can render structured entries live.

Log message shape:

```json
{
  "type": "log",
  "level": "info",
  "stage": "formatter/gemini_retrieval",
  "message": "Pipeline starting with 14 stages",
  "record": { "...structured record..." }
}
```

Stage message shape:

```json
{
  "type": "stage",
  "event": "complete",
  "stage": "formatter/gemini_retrieval",
  "info": { "...stage payload..." },
  "record": { "...structured record..." }
}
```

## Dashboard UI

The run detail page now shows a persistent `Logs` tab.

The UI:

- loads historical structured logs from the REST API
- appends live structured events from the WebSocket
- pages older records from the REST API using `before_sequence`
- renders:
  - severity
  - event type
  - stage
  - timestamp
  - sequence
  - expandable JSON payload
- supports client-side filters for:
  - severity
  - event type
  - stage
  - free-text search across message and payload

## Implementation Files

Backend:

- `dashboard/structured_logs.py`
- `dashboard/run_executor.py`
- `dashboard/app.py`

Frontend:

- `dashboard-ui/lib/types.ts`
- `dashboard-ui/lib/api.ts`
- `dashboard-ui/lib/hooks/use-websocket.ts`
- `dashboard-ui/components/project-detail/live-output-tab.tsx`
- `dashboard-ui/app/projects/detail/client.tsx`

## Compatibility

Legacy endpoints still work:

- `GET /api/runs/{run_id}/logs`
- `GET /api/runs/{run_id}/stages/{stage_name}/log`

Those remain line-oriented for backward compatibility.
