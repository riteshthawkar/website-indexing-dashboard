#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/env"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

cd "$ROOT_DIR"

if [[ "${1:-}" != "run" ]]; then
  echo "Usage: ./scripts/pipeline-detached.sh run --config <config> --run-id <run_id> [--resume|--restart-from-stage <stage>] [other args...]"
  exit 1
fi

CONFIG_NAME=""
RUN_ID=""

for ((i=1; i<=$#; i++)); do
  arg="${!i}"
  if [[ "$arg" == "--config" ]]; then
    next=$((i+1))
    CONFIG_NAME="${!next:-}"
  elif [[ "$arg" == "--run-id" ]]; then
    next=$((i+1))
    RUN_ID="${!next:-}"
  fi
done

if [[ -z "$CONFIG_NAME" || -z "$RUN_ID" ]]; then
  echo "Detached launcher requires both --config and --run-id."
  exit 1
fi

PROJECT_NAME="$(python - <<'PY' "$CONFIG_NAME"
import sys
from pipeline.core.config import load_config
cfg = load_config(sys.argv[1])
print(cfg.get("project_name", "default"))
PY
)"

RUN_DIR="$ROOT_DIR/runs/$PROJECT_NAME/$RUN_ID"
mkdir -p "$RUN_DIR"
LOG_PATH="$RUN_DIR/terminal.log"

{
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] starting detached pipeline: python -m pipeline $*"
} >>"$LOG_PATH"

ARGS_JSON="$(python - <<'PY' "$@"
import json
import sys
print(json.dumps(sys.argv[1:]))
PY
)"

PID="$(LOG_PATH="$LOG_PATH" PIPELINE_ARGS_JSON="$ARGS_JSON" python - <<'PY'
import json
import os
import subprocess
import sys

log_path = os.environ["LOG_PATH"]
args = json.loads(os.environ["PIPELINE_ARGS_JSON"])

log_handle = open(log_path, "ab", buffering=0)
proc = subprocess.Popen(
    [sys.executable, "-m", "pipeline", *args],
    stdin=subprocess.DEVNULL,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env={**os.environ, "PYTHONUNBUFFERED": "1"},
)
print(proc.pid)
PY
)"

echo "$PID"
echo "log=$LOG_PATH"
