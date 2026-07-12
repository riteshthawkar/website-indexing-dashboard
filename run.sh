#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONFIG="${CONFIG:-default}"
RUN_ID="${RUN_ID:-}"
RESUME="${RESUME:-false}"
RESTART_FROM_STAGE="${RESTART_FROM_STAGE:-}"
SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-false}"
VERBOSE="${VERBOSE:-false}"

export PIPELINE_CRAWLER__START_URL="${START_URL:-https://mbzuai.ac.ae}"
export PIPELINE_CRAWLER__MAX_PAGES="${MAX_PAGES:-5000}"
export PIPELINE_CRAWLER__MAX_DEPTH="${MAX_DEPTH:-4}"
export PIPELINE_CRAWLER__RESPECT_ROBOTS_TXT="${RESPECT_ROBOTS_TXT:-true}"
export PIPELINE_CRAWLER__FETCH_CONCURRENCY="${FETCH_CONCURRENCY:-5}"

echo "MBZUAI indexing pipeline"
echo "  config: $CONFIG"
echo "  start_url: $PIPELINE_CRAWLER__START_URL"
echo "  max_pages: $PIPELINE_CRAWLER__MAX_PAGES"
echo "  max_depth: $PIPELINE_CRAWLER__MAX_DEPTH"
echo "  respect_robots_txt: $PIPELINE_CRAWLER__RESPECT_ROBOTS_TXT"
echo "  fetch_concurrency: $PIPELINE_CRAWLER__FETCH_CONCURRENCY"

if [[ "$SKIP_PREFLIGHT" != "true" ]]; then
  bash scripts/pipeline.sh doctor --config "$CONFIG"
fi

ARGS=(run --config "$CONFIG")
if [[ "$RESUME" == "true" ]]; then
  ARGS+=(--resume)
fi
if [[ -n "$RESTART_FROM_STAGE" ]]; then
  ARGS+=(--restart-from-stage "$RESTART_FROM_STAGE")
fi
if [[ -n "$RUN_ID" ]]; then
  ARGS+=(--run-id "$RUN_ID")
fi
if [[ "$SKIP_PREFLIGHT" == "true" ]]; then
  ARGS+=(--skip-preflight)
fi

if [[ "$VERBOSE" == "true" ]]; then
  exec bash scripts/pipeline.sh -v "${ARGS[@]}"
fi
exec bash scripts/pipeline.sh "${ARGS[@]}"
