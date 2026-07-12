#!/usr/bin/env bash
set -euo pipefail

PIPELINE_CONFIG="${PIPELINE_CONFIG:-mbzuai_production}"
RELEASE_PROJECT_NAME="${RELEASE_PROJECT_NAME:-mbzuai_main}"
PIPELINE_WORK_DIR="${PIPELINE_WORK_DIR:-/data/releases/runs}"
RUN_ID="${RUN_ID:-mbzuai-production-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON:-python}"
RELEASE_STORAGE_MARKER_FILE="${RELEASE_STORAGE_MARKER_FILE:-/data/releases/.mbzuai-release-storage}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3 in PATH." >&2
    exit 1
  fi
fi

if [[ "${RELEASE_STORAGE_MODE:-persistent}" != "persistent" ]]; then
  echo "Production indexing requires RELEASE_STORAGE_MODE=persistent." >&2
  exit 1
fi
if [[ ! -f "$RELEASE_STORAGE_MARKER_FILE" ]]; then
  echo "Release storage marker is missing: $RELEASE_STORAGE_MARKER_FILE" >&2
  exit 1
fi
if [[ "$PIPELINE_CONFIG" != "mbzuai_production" && "${ALLOW_NON_PRODUCTION_INDEXING:-false}" != "true" ]]; then
  echo "Deployment indexing requires PIPELINE_CONFIG=mbzuai_production." >&2
  exit 1
fi
if [[ "${PIPELINE_PREFLIGHT:-true}" != "true" ]]; then
  echo "PIPELINE_PREFLIGHT must be true for deployment indexing." >&2
  exit 1
fi
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "RUN_ID contains unsafe characters: $RUN_ID" >&2
  exit 1
fi

export PIPELINE_WORK_DIR
mkdir -p "$PIPELINE_WORK_DIR/$RELEASE_PROJECT_NAME"

args=(run --config "$PIPELINE_CONFIG" --run-id "$RUN_ID")
if [[ "${PIPELINE_RESUME:-false}" == "true" ]]; then
  args+=(--resume)
fi
if [[ -n "${PIPELINE_RESTART_FROM_STAGE:-}" ]]; then
  args+=(--restart-from-stage "$PIPELINE_RESTART_FROM_STAGE")
fi
if [[ "${PIPELINE_PREFLIGHT:-true}" == "true" ]]; then
  args+=(--preflight)
fi

echo "Starting indexing run_id=$RUN_ID config=$PIPELINE_CONFIG work_root=$PIPELINE_WORK_DIR"
exec "$PYTHON_BIN" -m pipeline "${args[@]}" "$@"
