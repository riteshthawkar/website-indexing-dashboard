#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTIVE_RELEASE_FILE="${ACTIVE_RELEASE_FILE:-/data/releases/mbzuai_main/active_release.json}"
RELEASE_PROJECT_NAME="${RELEASE_PROJECT_NAME:-mbzuai_main}"
RELEASE_RUNS_ROOT="${RELEASE_RUNS_ROOT:-/data/releases/runs/${RELEASE_PROJECT_NAME}}"
ROLLBACK_RUN_ID="${ROLLBACK_RUN_ID:-${1:-}}"
PYTHON_BIN="${PYTHON:-python}"

if [[ "${RELEASE_STORAGE_MODE:-persistent}" != "persistent" ]]; then
  echo "Pointer rollback requires RELEASE_STORAGE_MODE=persistent; publish a previous archive for hydrate mode." >&2
  exit 1
fi
if [[ -z "$ROLLBACK_RUN_ID" ]]; then
  echo "ROLLBACK_RUN_ID is required (or pass the run id as the first argument)." >&2
  exit 1
fi
if [[ ! "$ROLLBACK_RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "ROLLBACK_RUN_ID contains unsafe characters: $ROLLBACK_RUN_ID" >&2
  exit 1
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi
if [[ "${RELEASE_POINTER_LOCK_HELD:-false}" != "true" ]]; then
  exec env RELEASE_POINTER_LOCK_HELD=true \
    "$PYTHON_BIN" "$SCRIPT_DIR/with-run-lock.py" \
    --lock-file "${ACTIVE_RELEASE_FILE}.lock" \
    --timeout-seconds "${RELEASE_LOCK_TIMEOUT_SECONDS:-30}" \
    -- bash "$0" "$@"
fi
if [[ "${RELEASE_RUN_LOCK_HELD:-false}" != "true" ]]; then
  exec env RELEASE_RUN_LOCK_HELD=true \
    "$PYTHON_BIN" "$SCRIPT_DIR/with-run-lock.py" \
    --lock-file "$RELEASE_RUNS_ROOT/$ROLLBACK_RUN_ID/.run.lock" \
    --timeout-seconds "${RELEASE_LOCK_TIMEOUT_SECONDS:-30}" \
    -- bash "$0" "$@"
fi

target_manifest="$RELEASE_RUNS_ROOT/$ROLLBACK_RUN_ID/release/retrieval_release_manifest.json"
if [[ ! -f "$target_manifest" ]]; then
  echo "Rollback release manifest does not exist: $target_manifest" >&2
  exit 1
fi

active_dir="$(dirname "$ACTIVE_RELEASE_FILE")"
mkdir -p "$active_dir"
candidate_pointer="$(mktemp "${active_dir}/.rollback-pointer.XXXXXX")"
cleanup() {
  rm -f "$candidate_pointer"
}
trap cleanup EXIT

"$PYTHON_BIN" - "$target_manifest" "$candidate_pointer" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).resolve()
candidate_path = Path(sys.argv[2])
payload = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("rollback release manifest must contain a JSON object")
pointer = {
    "schema_version": 1,
    "active_release_manifest": str(manifest_path),
    "release_id": payload.get("release_id"),
    "run_id": payload.get("run_id"),
    "production_indexing_contract_fingerprint": payload.get(
        "production_indexing_contract_fingerprint"
    ),
    "production_serving_contract_fingerprint": payload.get(
        "production_serving_contract_fingerprint"
    ),
    "answer_runtime_commit_sha": (
        (payload.get("answer_runtime") or {}).get("commit_sha")
        if isinstance(payload.get("answer_runtime"), dict)
        else ""
    ),
    "indexing_build_commit_sha": (
        (payload.get("indexing_build") or {}).get("commit_sha")
        if isinstance(payload.get("indexing_build"), dict)
        else ""
    ),
    "promoted_at": payload.get("promoted_at"),
    "status": payload.get("status"),
}
candidate_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
candidate_path.chmod(0o640)
PY

ACTIVE_RELEASE_FILE="$candidate_pointer" \
RELEASE_RUNS_ROOT="$RELEASE_RUNS_ROOT" \
RETRIEVER_RESOLVE_ONLY=true \
RETRIEVER_ALLOW_WAIVED_RELEASE="${ROLLBACK_ALLOW_WAIVED_RELEASE:-false}" \
RELEASE_STORAGE_MODE=persistent \
bash "$SCRIPT_DIR/start-retriever-from-active-release.sh" >/dev/null

if [[ "${ROLLBACK_DRY_RUN:-false}" == "true" ]]; then
  echo "Rollback validation passed for run_id=$ROLLBACK_RUN_ID; active pointer was not changed."
  exit 0
fi

"$PYTHON_BIN" - "$ACTIVE_RELEASE_FILE" "$candidate_pointer" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

active_path = Path(sys.argv[1])
candidate_path = Path(sys.argv[2])
history_dir = active_path.parent / "rollback_history"
history_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
if active_path.is_file():
    current = json.loads(active_path.read_text(encoding="utf-8"))
    current_run = str(current.get("run_id") or "unknown").replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    history_path = history_dir / f"{stamp}-{current_run}.json"
    history_path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    history_path.chmod(0o640)
os.replace(candidate_path, active_path)
PY

echo "Active retrieval release rolled back to run_id=$ROLLBACK_RUN_ID. Restart retriever and backend, then run smoke checks."
