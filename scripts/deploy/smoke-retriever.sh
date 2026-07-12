#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRIEVER_BASE_URL="${RETRIEVER_BASE_URL:-http://127.0.0.1:8060}"
SMOKE_QUERY="${SMOKE_QUERY:-What masters programs does MBZUAI offer?}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-30}"
PYTHON_BIN="${PYTHON:-python}"
EXPECTED_RETRIEVER_COMMIT_SHA="${EXPECTED_RETRIEVER_COMMIT_SHA:-${RELEASE_COMMIT_SHA:-}}"
EXPECTED_RETRIEVAL_RELEASE_ID="${EXPECTED_RETRIEVAL_RELEASE_ID:-}"
EXPECTED_RETRIEVAL_RUN_ID="${EXPECTED_RETRIEVAL_RUN_ID:-}"
EXPECTED_RETRIEVAL_BUNDLE_SHA256="${EXPECTED_RETRIEVAL_BUNDLE_SHA256:-}"
EXPECTED_KNOWLEDGE_GRAPH_SHA256="${EXPECTED_KNOWLEDGE_GRAPH_SHA256:-}"
EXPECTED_KNOWLEDGE_GRAPH_INDEX_SHA256="${EXPECTED_KNOWLEDGE_GRAPH_INDEX_SHA256:-}"
EXPECTED_LEXICAL_CORPUS_SHA256="${EXPECTED_LEXICAL_CORPUS_SHA256:-}"
EXPECTED_PROMOTED_ASSERTIONS_SHA256="${EXPECTED_PROMOTED_ASSERTIONS_SHA256:-}"
EXPECTED_ANSWER_RUNTIME_COMMIT_SHA="${EXPECTED_ANSWER_RUNTIME_COMMIT_SHA:-}"
RETRIEVAL_SERVICE_TOKEN="${RETRIEVAL_SERVICE_TOKEN:-}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required for the retriever smoke test." >&2
  exit 127
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN="python3"
fi

if [[ "${SMOKE_VALIDATE_LOCAL_RELEASE:-true}" == "true" ]] \
  && { [[ -n "${RETRIEVAL_WORK_DIR:-}" ]] || [[ -f "${ACTIVE_RELEASE_FILE:-/data/releases/mbzuai_main/active_release.json}" ]]; }; then
  validation_json="$("$PYTHON_BIN" "$SCRIPT_DIR/validate-release-artifacts.py")"
  eval "$("$PYTHON_BIN" - "$validation_json" <<'PY'
import json
import shlex
import sys

payload = json.loads(sys.argv[1])
for env_name, key in (
    ("EXPECTED_RETRIEVAL_RELEASE_ID", "release_id"),
    ("EXPECTED_RETRIEVAL_RUN_ID", "run_id"),
    ("EXPECTED_RETRIEVAL_BUNDLE_SHA256", "retrieval_bundle_sha256"),
    ("EXPECTED_KNOWLEDGE_GRAPH_SHA256", "knowledge_graph_sha256"),
    ("EXPECTED_KNOWLEDGE_GRAPH_INDEX_SHA256", "knowledge_graph_index_sha256"),
    ("EXPECTED_LEXICAL_CORPUS_SHA256", "lexical_corpus_sha256"),
    ("EXPECTED_PROMOTED_ASSERTIONS_SHA256", "promoted_assertions_sha256"),
    ("EXPECTED_ANSWER_RUNTIME_COMMIT_SHA", "answer_runtime_commit_sha"),
):
    if payload.get(key):
        print(f"export {env_name}={shlex.quote(str(payload[key]))}")
PY
)"
fi
if [[ -z "$EXPECTED_RETRIEVER_COMMIT_SHA" \
  || -z "$EXPECTED_RETRIEVAL_RUN_ID" \
  || -z "$EXPECTED_RETRIEVAL_BUNDLE_SHA256" \
  || -z "$EXPECTED_KNOWLEDGE_GRAPH_SHA256" \
  || -z "$EXPECTED_KNOWLEDGE_GRAPH_INDEX_SHA256" \
  || -z "$EXPECTED_LEXICAL_CORPUS_SHA256" \
  || -z "$EXPECTED_PROMOTED_ASSERTIONS_SHA256" \
  || -z "$EXPECTED_ANSWER_RUNTIME_COMMIT_SHA" ]]; then
  echo "Expected retriever commit/run and all runtime artifact SHA256 values are required for an identity-safe smoke." >&2
  exit 1
fi
if (( ${#RETRIEVAL_SERVICE_TOKEN} < 32 )); then
  echo "RETRIEVAL_SERVICE_TOKEN with at least 32 characters is required for the production smoke." >&2
  exit 1
fi

base_url="${RETRIEVER_BASE_URL%/}"
ready_file="$(mktemp)"
retrieve_file="$(mktemp)"
auth_header_file="$(mktemp)"
chmod 600 "$auth_header_file"
printf 'X-Retrieval-Service-Token: %s\n' "$RETRIEVAL_SERVICE_TOKEN" >"$auth_header_file"
cleanup() { rm -f "$ready_file" "$retrieve_file" "$auth_header_file"; }
trap cleanup EXIT

curl --fail --silent --show-error \
  --max-time "$SMOKE_TIMEOUT_SECONDS" \
  --header "@$auth_header_file" \
  "$base_url/attestationz" >"$ready_file"

request_payload="$("$PYTHON_BIN" - "$SMOKE_QUERY" <<'PY'
import json
import sys
print(json.dumps({"query": sys.argv[1], "request_id": "production-smoke"}))
PY
)"
curl --fail --silent --show-error \
  --max-time "$SMOKE_TIMEOUT_SECONDS" \
  --header 'Content-Type: application/json' \
  --header "@$auth_header_file" \
  --data "$request_payload" \
  "$base_url/retrieve" >"$retrieve_file"

"$PYTHON_BIN" - \
  "$ready_file" \
  "$retrieve_file" \
  "$EXPECTED_RETRIEVER_COMMIT_SHA" \
  "$EXPECTED_RETRIEVAL_RUN_ID" \
  "$EXPECTED_RETRIEVAL_BUNDLE_SHA256" \
  "$EXPECTED_RETRIEVAL_RELEASE_ID" \
  "$EXPECTED_KNOWLEDGE_GRAPH_SHA256" \
  "$EXPECTED_KNOWLEDGE_GRAPH_INDEX_SHA256" \
  "$EXPECTED_LEXICAL_CORPUS_SHA256" \
  "$EXPECTED_PROMOTED_ASSERTIONS_SHA256" \
  "$EXPECTED_ANSWER_RUNTIME_COMMIT_SHA" <<'PY'
import json
import sys
from pathlib import Path

ready = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
expected_commit, expected_run, expected_bundle, expected_release = sys.argv[3:7]
expected_graph, expected_graph_index, expected_lexical, expected_assertions = sys.argv[7:11]
expected_answer_runtime_commit = sys.argv[11]
if ready.get("ready") is not True:
    raise SystemExit("retriever /attestationz did not report ready=true")
if ready.get("config_name") != "mbzuai_production":
    raise SystemExit(f"retriever is using unexpected config: {ready.get('config_name')!r}")
if ready.get("commit_sha") != expected_commit:
    raise SystemExit(f"retriever commit mismatch: {ready.get('commit_sha')!r} != {expected_commit!r}")
if ready.get("run_id") != expected_run:
    raise SystemExit(f"retriever run mismatch: {ready.get('run_id')!r} != {expected_run!r}")
if ready.get("retrieval_bundle_sha256") != expected_bundle:
    raise SystemExit("retriever bundle SHA256 does not match the promoted release")
if expected_release and ready.get("release_id") != expected_release:
    raise SystemExit(f"retriever release mismatch: {ready.get('release_id')!r} != {expected_release!r}")
for key, expected in (
    ("knowledge_graph_sha256", expected_graph),
    ("knowledge_graph_index_sha256", expected_graph_index),
    ("lexical_corpus_sha256", expected_lexical),
    ("promoted_assertions_sha256", expected_assertions),
):
    if ready.get(key) != expected:
        raise SystemExit(f"retriever runtime artifact mismatch for {key}")
if ready.get("answer_runtime_commit_sha") != expected_answer_runtime_commit:
    raise SystemExit("retriever answer-runtime commit does not match the evaluated backend")
documents = result.get("retrieval_documents")
if not isinstance(documents, list) or not documents:
    raise SystemExit("retriever smoke query returned no retrieval_documents")
if result.get("service_backend") != "retrieval_service":
    raise SystemExit("retriever smoke response is missing service_backend=retrieval_service")
if not str(result.get("service_request_id") or "").strip():
    raise SystemExit("retriever smoke response is missing a service request id")
print(
    json.dumps(
        {
            "ok": True,
            "config_name": ready.get("config_name"),
            "commit_sha": ready.get("commit_sha"),
            "run_id": ready.get("run_id"),
            "release_id": ready.get("release_id"),
            "document_count": len(documents),
            "retrieval_confidence": result.get("retrieval_confidence"),
            "verification_status": result.get("verification_status"),
        },
        sort_keys=True,
    )
)
PY

echo "Retriever production smoke passed."
