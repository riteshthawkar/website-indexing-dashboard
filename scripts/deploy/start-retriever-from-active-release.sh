#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_CONFIG="${PIPELINE_CONFIG:-mbzuai_production}"
ACTIVE_RELEASE_FILE="${ACTIVE_RELEASE_FILE:-/data/releases/mbzuai_main/active_release.json}"
RELEASE_PROJECT_NAME="${RELEASE_PROJECT_NAME:-mbzuai_main}"
RELEASE_RUNS_ROOT="${RELEASE_RUNS_ROOT:-/data/releases/runs/${RELEASE_PROJECT_NAME}}"
RELEASE_STORAGE_MODE="$(printf '%s' "${RELEASE_STORAGE_MODE:-persistent}" | tr '[:upper:]' '[:lower:]')"
RETRIEVER_HOST="${RETRIEVER_HOST:-0.0.0.0}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8060}"
RETRIEVER_MAX_CONCURRENCY="${RETRIEVER_MAX_CONCURRENCY:-4}"
RETRIEVER_REQUEST_TIMEOUT_SECONDS="${RETRIEVER_REQUEST_TIMEOUT_SECONDS:-110}"
RETRIEVER_QUEUE_TIMEOUT_SECONDS="${RETRIEVER_QUEUE_TIMEOUT_SECONDS:-10}"
PYTHON_BIN="${PYTHON:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3 in PATH." >&2
    exit 127
  fi
fi

if [[ "${RETRIEVER_VALIDATE_JSON:-false}" != "true" \
  && "${RETRIEVER_RESOLVE_ONLY:-false}" != "true" ]]; then
  "$PYTHON_BIN" - <<'PY'
import os

token = os.getenv("RETRIEVAL_SERVICE_TOKEN", "")
normalized = token.strip().lower()
markers = (
    "your",
    "example",
    "test",
    "dummy",
    "change-me",
    "change_me",
    "changeme",
    "replace-me",
    "replace_me",
    "placeholder",
)


def repeated_pattern(value: str) -> bool:
    return any(
        len(value) % width == 0 and value == value[:width] * (len(value) // width)
        for width in range(1, (len(value) // 2) + 1)
    )


errors = []
if len(token) < 32:
    errors.append("must contain at least 32 characters")
elif token != token.strip():
    errors.append("must not have leading or trailing whitespace")
elif any(character.isspace() or ord(character) < 32 or ord(character) == 127 for character in token):
    errors.append("must not contain whitespace or control characters")
elif any(marker in normalized for marker in markers):
    errors.append("must not be an example placeholder")
elif len(set(token)) <= 1:
    errors.append("has insufficient character diversity")
elif repeated_pattern(token):
    errors.append("must not be a repeated pattern")
elif len(set(token)) < 10:
    errors.append("has insufficient character diversity")
if errors:
    raise SystemExit("RETRIEVAL_SERVICE_TOKEN " + "; ".join(errors))
PY
fi

case "$RELEASE_STORAGE_MODE" in
  persistent)
    # The validator requires the marker created when durable storage is
    # provisioned. The marker is intentionally not baked into the image.
    ;;
  hydrate)
    "$PYTHON_BIN" "$SCRIPT_DIR/hydrate-release-archive.py"
    ;;
  *)
    echo "RELEASE_STORAGE_MODE must be persistent or hydrate; got ${RELEASE_STORAGE_MODE}." >&2
    exit 1
    ;;
esac

validation_json="$("$PYTHON_BIN" "$SCRIPT_DIR/validate-release-artifacts.py")"

eval "$("$PYTHON_BIN" - "$validation_json" <<'PY'
import json
import shlex
import sys

payload = json.loads(sys.argv[1])
for env_name, key in (
    ("RESOLVED_RETRIEVAL_WORK_DIR", "work_dir"),
    ("RETRIEVAL_RELEASE_ID", "release_id"),
    ("RETRIEVAL_RELEASE_RUN_ID", "run_id"),
    ("RETRIEVAL_RELEASE_MANIFEST_PATH", "release_manifest"),
    ("RETRIEVAL_BUNDLE_SHA256", "retrieval_bundle_sha256"),
    ("RETRIEVAL_KNOWLEDGE_GRAPH_SHA256", "knowledge_graph_sha256"),
    ("RETRIEVAL_KNOWLEDGE_GRAPH_INDEX_SHA256", "knowledge_graph_index_sha256"),
    ("RETRIEVAL_LEXICAL_CORPUS_SHA256", "lexical_corpus_sha256"),
    ("RETRIEVAL_PROMOTED_ASSERTIONS_SHA256", "promoted_assertions_sha256"),
    ("RETRIEVAL_ANSWER_RUNTIME_COMMIT_SHA", "answer_runtime_commit_sha"),
    ("RETRIEVAL_INDEXING_BUILD_COMMIT_SHA", "indexing_build_commit_sha"),
):
    print(f"export {env_name}={shlex.quote(str(payload.get(key) or ''))}")
PY
)"

if [[ "${RETRIEVER_VALIDATE_JSON:-false}" == "true" ]]; then
  printf '%s\n' "$validation_json"
  exit 0
fi
if [[ "${RETRIEVER_RESOLVE_ONLY:-false}" == "true" ]]; then
  printf '%s\n' "$RESOLVED_RETRIEVAL_WORK_DIR"
  exit 0
fi

if [[ -z "${RELEASE_COMMIT_SHA:-}" ]]; then
  RELEASE_COMMIT_SHA="${SOURCE_COMMIT_HASH:-${GITHUB_SHA:-}}"
fi
if [[ -z "${RELEASE_COMMIT_SHA:-}" ]]; then
  echo "RELEASE_COMMIT_SHA is required; on DigitalOcean set it to the bindable value \${_self.COMMIT_HASH}." >&2
  exit 1
fi
export RELEASE_COMMIT_SHA

echo "Starting retriever: config=${PIPELINE_CONFIG} run_id=${RETRIEVAL_RELEASE_RUN_ID} release_id=${RETRIEVAL_RELEASE_ID:-candidate} commit_sha=${RELEASE_COMMIT_SHA}"
exec "$PYTHON_BIN" -m pipeline.service.retrieval_runner \
  --config "$PIPELINE_CONFIG" \
  --work-dir "$RESOLVED_RETRIEVAL_WORK_DIR" \
  --host "$RETRIEVER_HOST" \
  --port "$RETRIEVER_PORT" \
  --max-concurrency "$RETRIEVER_MAX_CONCURRENCY" \
  --request-timeout-seconds "$RETRIEVER_REQUEST_TIMEOUT_SECONDS" \
  --queue-timeout-seconds "$RETRIEVER_QUEUE_TIMEOUT_SECONDS"
