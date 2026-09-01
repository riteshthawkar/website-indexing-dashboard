#!/usr/bin/env bash
set -euo pipefail

PIPELINE_CONFIG="${PIPELINE_CONFIG:-mbzuai_production}"
ACTIVE_RELEASE_FILE="${ACTIVE_RELEASE_FILE:-/data/releases/mbzuai_main/active_release.json}"
ANSWER_EVAL_MODE="${ANSWER_EVAL_MODE:-websocket}"
ANSWER_EVAL_ENDPOINT="${ANSWER_EVAL_ENDPOINT:-ws://backend-candidate:8080/chat}"
CANDIDATE_RETRIEVER_ATTESTATION_URL="${CANDIDATE_RETRIEVER_ATTESTATION_URL:-http://retriever-candidate:8060/attestationz}"
CANDIDATE_BACKEND_DETAILED_URL="${CANDIDATE_BACKEND_DETAILED_URL:-http://backend-candidate:8080/health/detailed}"
CANDIDATE_BACKEND_OPERATIONS_TOKEN="${CANDIDATE_BACKEND_OPERATIONS_TOKEN:-}"
CANDIDATE_BACKEND_COMMIT_SHA="${CANDIDATE_BACKEND_COMMIT_SHA:-}"
RETRIEVAL_SERVICE_TOKEN="${RETRIEVAL_SERVICE_TOKEN:-}"
CANDIDATE_ATTESTATION_TIMEOUT_SECONDS="${CANDIDATE_ATTESTATION_TIMEOUT_SECONDS:-30}"
CANDIDATE_ATTESTATION_ALLOWED_HOSTS="${CANDIDATE_ATTESTATION_ALLOWED_HOSTS:-backend-candidate,retriever-candidate}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
ANSWER_MODEL="${ANSWER_MODEL:-gemini-2.5-flash}"
PARALLELISM="${RELEASE_CHECK_PARALLELISM:-2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
pipeline_config_stem="$(basename "$PIPELINE_CONFIG")"
pipeline_config_stem="${pipeline_config_stem%.yaml}"
pipeline_config_stem="${pipeline_config_stem%.yml}"
if [[ "$pipeline_config_stem" == "mbzuai_preprod" \
  || "$pipeline_config_stem" == mbzuai_preprod_* ]]; then
  CANONICAL_RELEASE_DATASET="$PROJECT_ROOT/eval/mbzuai_gold/mbzuai_preprod_multilingual_current_v1.jsonl"
  CANONICAL_RELEASE_GATES="$PROJECT_ROOT/eval/gates/retrieval_gate.preprod_current_v1.json"
  CANONICAL_ANSWER_GATES="$PROJECT_ROOT/eval/gates/answer_readiness_gate.preprod_current_v1.json"
else
  CANONICAL_RELEASE_DATASET="$PROJECT_ROOT/eval/mbzuai_gold/mbzuai_multilingual_v2.jsonl"
  CANONICAL_RELEASE_GATES="$PROJECT_ROOT/eval/gates/retrieval_gate.multilingual_v2_release.json"
  CANONICAL_ANSWER_GATES="$PROJECT_ROOT/eval/gates/answer_readiness_gate.multilingual_v2_release.json"
fi
CANONICAL_ANSWER_DATASET="$CANONICAL_RELEASE_DATASET"
PYTHON_BIN="${PYTHON:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Could not find python or python3 in PATH." >&2
    exit 1
  fi
fi

if [[ -z "${RELEASE_WORK_DIR:-}" ]]; then
  echo "RELEASE_WORK_DIR is required, for example /data/releases/runs/mbzuai_main/<run_id>." >&2
  exit 1
fi

if [[ ! -d "$RELEASE_WORK_DIR" ]]; then
  echo "RELEASE_WORK_DIR does not exist: $RELEASE_WORK_DIR" >&2
  exit 1
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
    --lock-file "$RELEASE_WORK_DIR/.run.lock" \
    --timeout-seconds "${RELEASE_LOCK_TIMEOUT_SECONDS:-30}" \
    -- bash "$0" "$@"
fi
if [[ "$ANSWER_EVAL_MODE" == "disabled" ]]; then
  echo "ANSWER_EVAL_MODE=disabled is forbidden for production promotion." >&2
  exit 1
fi
if ! "$PYTHON_BIN" - "$PROJECT_ROOT" "$PIPELINE_CONFIG" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from pipeline.core.config import load_config

config = load_config(sys.argv[2])
if not bool((config.get("pipeline") or {}).get("production_profile", False)):
    raise SystemExit(1)
PY
then
  echo "Production promotion requires a config with pipeline.production_profile=true." >&2
  exit 1
fi
if [[ "$ANSWER_EVAL_MODE" != "websocket" ]]; then
  echo "Production promotion requires ANSWER_EVAL_MODE=websocket." >&2
  exit 1
fi
for override_name in RELEASE_DATASET RELEASE_GATES ANSWER_DATASET ANSWER_GATES; do
  if [[ -n "${!override_name:-}" ]]; then
    echo "$override_name cannot override the canonical committed production evaluation contract." >&2
    exit 1
  fi
done
if [[ "$ANSWER_MODEL" != "gemini-2.5-flash" || "$JUDGE_MODEL" != "gemini-2.5-flash" ]]; then
  echo "Production answer generation and judging must use the pinned gemini-2.5-flash model." >&2
  exit 1
fi
if [[ -z "$ANSWER_EVAL_ENDPOINT" ]]; then
  echo "ANSWER_EVAL_ENDPOINT is required for production promotion." >&2
  exit 1
fi
if (( ${#CANDIDATE_BACKEND_OPERATIONS_TOKEN} < 32 )) || [[ "$CANDIDATE_BACKEND_OPERATIONS_TOKEN" =~ [[:space:][:cntrl:]] ]]; then
  echo "CANDIDATE_BACKEND_OPERATIONS_TOKEN must contain at least 32 non-whitespace characters." >&2
  exit 1
fi
if [[ ! "$CANDIDATE_BACKEND_COMMIT_SHA" =~ ^([0-9a-fA-F]{40}|[0-9a-fA-F]{64})$ ]]; then
  echo "CANDIDATE_BACKEND_COMMIT_SHA must identify the exact backend-candidate image revision." >&2
  exit 1
fi
if (( ${#RETRIEVAL_SERVICE_TOKEN} < 32 )) || [[ "$RETRIEVAL_SERVICE_TOKEN" =~ [[:space:][:cntrl:]] ]]; then
  echo "RETRIEVAL_SERVICE_TOKEN must contain at least 32 non-whitespace characters." >&2
  exit 1
fi
if [[ ! "${RELEASE_COMMIT_SHA:-}" =~ ^([0-9a-fA-F]{40}|[0-9a-fA-F]{64})$ ]]; then
  echo "RELEASE_COMMIT_SHA must be the full retriever-candidate Git revision." >&2
  exit 1
fi

# Validate immutable runtime artifacts before spending time and model calls on
# answer evaluation. This does not require an already-promoted release pointer.
candidate_validation_json="$(
  RETRIEVAL_WORK_DIR="$RELEASE_WORK_DIR" \
  "$PYTHON_BIN" "$SCRIPT_DIR/validate-release-artifacts.py"
)"

# Validate every destination before creating header files or making a network
# request. This prevents a typo or hostile override from receiving candidate
# credentials before the post-response identity checks run.
"$PYTHON_BIN" - \
  "$ANSWER_EVAL_ENDPOINT" \
  "$CANDIDATE_BACKEND_DETAILED_URL" \
  "$CANDIDATE_RETRIEVER_ATTESTATION_URL" \
  "$CANDIDATE_ATTESTATION_ALLOWED_HOSTS" <<'PY'
import ipaddress
import sys
from urllib.parse import urlparse

answer_url = urlparse(sys.argv[1])
backend_url = urlparse(sys.argv[2])
retriever_url = urlparse(sys.argv[3])
allowed_hosts = {
    value.strip().lower().rstrip(".")
    for value in sys.argv[4].split(",")
    if value.strip()
}
if not allowed_hosts:
    raise SystemExit("CANDIDATE_ATTESTATION_ALLOWED_HOSTS must not be empty")


def validate_endpoint(parsed, *, schemes, path, label):
    if (
        parsed.scheme not in schemes
        or not parsed.hostname
        or parsed.path != path
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise SystemExit(f"{label} is not a safe candidate endpoint")
    hostname = parsed.hostname.lower().rstrip(".")
    if hostname not in allowed_hosts:
        raise SystemExit(f"{label} hostname is not exactly allowlisted")
    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return
    if not ip.is_loopback:
        raise SystemExit(f"{label} must not use a non-loopback IP literal")


validate_endpoint(answer_url, schemes={"ws", "wss"}, path="/chat", label="ANSWER_EVAL_ENDPOINT")
validate_endpoint(
    backend_url,
    schemes={"http", "https"},
    path="/health/detailed",
    label="CANDIDATE_BACKEND_DETAILED_URL",
)
validate_endpoint(
    retriever_url,
    schemes={"http", "https"},
    path="/attestationz",
    label="CANDIDATE_RETRIEVER_ATTESTATION_URL",
)


def effective_origin(parsed):
    default_ports = {"http": 80, "https": 443, "ws": 80, "wss": 443}
    return parsed.hostname.lower().rstrip("."), parsed.port or default_ports[parsed.scheme]


expected_backend_scheme = "https" if answer_url.scheme == "wss" else "http"
if backend_url.scheme != expected_backend_scheme or effective_origin(answer_url) != effective_origin(backend_url):
    raise SystemExit("answer evaluation and backend attestation must use the same candidate origin")
PY

candidate_retriever_file="$(mktemp)"
candidate_backend_file="$(mktemp)"
candidate_retriever_header_file="$(mktemp)"
candidate_backend_header_file="$(mktemp)"
chmod 600 \
  "$candidate_retriever_file" \
  "$candidate_backend_file" \
  "$candidate_retriever_header_file" \
  "$candidate_backend_header_file"
printf 'X-Retrieval-Service-Token: %s\n' "$RETRIEVAL_SERVICE_TOKEN" >"$candidate_retriever_header_file"
printf 'X-Operations-Token: %s\n' "$CANDIDATE_BACKEND_OPERATIONS_TOKEN" >"$candidate_backend_header_file"
cleanup_attestation() {
  rm -f \
    "$candidate_retriever_file" \
    "$candidate_backend_file" \
    "$candidate_retriever_header_file" \
    "$candidate_backend_header_file"
}
trap cleanup_attestation EXIT
attest_candidate() {
  local evidence_output="${1:-}"
  local release_manifest_file="${2:-}"
  curl --fail --silent --show-error \
    --max-time "$CANDIDATE_ATTESTATION_TIMEOUT_SECONDS" \
    --header "@$candidate_retriever_header_file" \
    "$CANDIDATE_RETRIEVER_ATTESTATION_URL" >"$candidate_retriever_file"
  curl --fail --silent --show-error \
    --max-time "$CANDIDATE_ATTESTATION_TIMEOUT_SECONDS" \
    --header "@$candidate_backend_header_file" \
    "$CANDIDATE_BACKEND_DETAILED_URL" >"$candidate_backend_file"

  "$PYTHON_BIN" - \
    "$candidate_validation_json" \
    "$candidate_retriever_file" \
    "$candidate_backend_file" \
    "$RELEASE_COMMIT_SHA" \
    "$ANSWER_EVAL_ENDPOINT" \
    "$CANDIDATE_BACKEND_DETAILED_URL" \
    "$CANDIDATE_RETRIEVER_ATTESTATION_URL" \
    "$CANDIDATE_BACKEND_COMMIT_SHA" \
    "$evidence_output" \
    "$release_manifest_file" \
    "$PIPELINE_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

expected = json.loads(sys.argv[1])
retriever = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
backend = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
expected_commit = sys.argv[4]
answer_url = urlparse(sys.argv[5])
backend_url = urlparse(sys.argv[6])
retriever_url = urlparse(sys.argv[7])
expected_backend_commit = sys.argv[8]
evidence_output = sys.argv[9]
release_manifest_file = sys.argv[10]
expected_config_name = sys.argv[11]
if (
    answer_url.scheme not in {"ws", "wss"}
    or not answer_url.hostname
    or answer_url.path != "/chat"
    or answer_url.username
    or answer_url.password
    or answer_url.query
    or answer_url.fragment
):
    raise SystemExit("ANSWER_EVAL_ENDPOINT must be the candidate WebSocket /chat endpoint")
if (
    backend_url.scheme not in {"http", "https"}
    or not backend_url.hostname
    or backend_url.path != "/health/detailed"
    or backend_url.username
    or backend_url.password
    or backend_url.query
    or backend_url.fragment
):
    raise SystemExit("CANDIDATE_BACKEND_DETAILED_URL must be the candidate HTTP /health/detailed endpoint")


def effective_origin(parsed):
    default_ports = {"http": 80, "https": 443, "ws": 80, "wss": 443}
    return parsed.hostname.lower(), parsed.port or default_ports[parsed.scheme]


expected_backend_scheme = "https" if answer_url.scheme == "wss" else "http"
if backend_url.scheme != expected_backend_scheme or effective_origin(answer_url) != effective_origin(backend_url):
    raise SystemExit(
        "answer evaluation and candidate attestation must target the same backend origin"
    )
if (
    retriever_url.scheme not in {"http", "https"}
    or not retriever_url.hostname
    or retriever_url.path != "/attestationz"
    or retriever_url.username
    or retriever_url.password
    or retriever_url.query
    or retriever_url.fragment
):
    raise SystemExit("CANDIDATE_RETRIEVER_ATTESTATION_URL must be the retriever /attestationz endpoint")
expected_identity = {
    "run_id": expected.get("run_id"),
    "commit_sha": expected_commit,
    "indexing_build_commit_sha": expected.get("indexing_build_commit_sha"),
    "retrieval_bundle_sha256": expected.get("retrieval_bundle_sha256"),
    "knowledge_graph_sha256": expected.get("knowledge_graph_sha256"),
    "knowledge_graph_index_sha256": expected.get("knowledge_graph_index_sha256"),
    "lexical_corpus_sha256": expected.get("lexical_corpus_sha256"),
    "promoted_assertions_sha256": expected.get("promoted_assertions_sha256"),
}
if expected.get("selected_release_assembly_sha256"):
    expected_identity.update(
        {
            "selected_release_assembly_sha256": expected.get(
                "selected_release_assembly_sha256"
            ),
            "selected_release_binding_sha256": expected.get(
                "selected_release_binding_sha256"
            ),
            "page_graph_navigation_catalog_sha256": expected.get(
                "page_graph_navigation_catalog_sha256"
            ),
        }
    )
if retriever.get("ready") is not True or retriever.get("config_name") != expected_config_name:
    raise SystemExit(
        f"retriever-candidate is not ready with {expected_config_name}"
    )
for key, value in expected_identity.items():
    if not value or retriever.get(key) != value:
        raise SystemExit(
            f"retriever-candidate identity mismatch for {key}: "
            f"expected={value!r}, actual={retriever.get(key)!r}"
        )
if backend.get("status") != "healthy":
    raise SystemExit(f"backend-candidate is not healthy: {backend.get('status')!r}")
backend_openai = (backend.get("checks") or {}).get("openai") or {}
expected_answer_models = {
    "generation_model": "gpt-5.4-2026-03-05",
    "query_rewrite_model": "gpt-5.4-mini-2026-03-17",
    "reranker_model": "gpt-5.4-mini-2026-03-17",
    "grounded_finalizer_model": "gpt-5.4-mini-2026-03-17",
}
if backend_openai.get("status") != "healthy":
    raise SystemExit("backend-candidate OpenAI configuration is not healthy")
for key, expected_model in expected_answer_models.items():
    if backend_openai.get(key) != expected_model:
        raise SystemExit(
            f"backend-candidate answer model mismatch for {key}: "
            f"expected={expected_model!r}, actual={backend_openai.get(key)!r}"
        )
actual_backend_commit = str((backend.get("release") or {}).get("commit_sha") or "")
if actual_backend_commit != expected_backend_commit:
    raise SystemExit(
        "backend-candidate code revision mismatch: "
        f"expected={expected_backend_commit!r}, actual={actual_backend_commit!r}"
    )
backend_retrieval = (backend.get("checks") or {}).get("retrieval_service") or {}
if (
    backend_retrieval.get("status") != "healthy"
    or backend_retrieval.get("ready") is not True
    or backend_retrieval.get("mode") != "required"
):
    raise SystemExit("backend-candidate is not using a healthy required retrieval service")
for key, value in expected_identity.items():
    if backend_retrieval.get(key) != value:
        raise SystemExit(
            f"backend-candidate retrieval identity mismatch for {key}: "
            f"expected={value!r}, actual={backend_retrieval.get(key)!r}"
        )
if evidence_output:
    if not release_manifest_file:
        raise SystemExit("release manifest path is required when writing promotion evidence")
    from pipeline.core.io import atomic_write_json
    from pipeline.core.release import build_promotion_attestation_evidence

    evidence = build_promotion_attestation_evidence(
        manifest_path=release_manifest_file,
        retriever_attestation=retriever,
        backend_health=backend,
        expected_retriever_commit_sha=expected_commit,
        expected_backend_commit_sha=expected_backend_commit,
    )
    atomic_write_json(evidence_output, evidence)
print(
    "Candidate attestation passed: "
    f"run_id={expected_identity['run_id']} bundle={expected_identity['retrieval_bundle_sha256']}"
)
PY
}

# A cheap first attestation avoids spending evaluation time against an already
# mismatched candidate. This result is intentionally not promotion evidence.
attest_candidate

args=(
  release-check
  --config "$PIPELINE_CONFIG"
  --work-dir "$RELEASE_WORK_DIR"
  --dataset "$CANONICAL_RELEASE_DATASET"
  --gates "$CANONICAL_RELEASE_GATES"
  --answer-dataset "$CANONICAL_ANSWER_DATASET"
  --answer-gates "$CANONICAL_ANSWER_GATES"
  --answer-eval-mode "$ANSWER_EVAL_MODE"
  --answer-endpoint "$ANSWER_EVAL_ENDPOINT"
  --answer-model "$ANSWER_MODEL"
  --judge-model "$JUDGE_MODEL"
  --answer-runtime-commit-sha "$CANDIDATE_BACKEND_COMMIT_SHA"
  --resume-answer-predictions
  --parallelism "$PARALLELISM"
)

if [[ -n "${ANSWER_WIDGET_KEY:-}" ]]; then
  args+=(--answer-widget-key "$ANSWER_WIDGET_KEY")
fi
mkdir -p "$(dirname "$ACTIVE_RELEASE_FILE")"
"$PYTHON_BIN" -m pipeline "${args[@]}"
release_manifest_file="$RELEASE_WORK_DIR/release/retrieval_release_manifest.json"
promotion_evidence_file="$RELEASE_WORK_DIR/release/promotion_attestation.json"
if [[ ! -f "$release_manifest_file" ]]; then
  echo "Release checks did not produce the required release manifest." >&2
  exit 1
fi

# Re-attest after every evaluation has completed. The resulting non-secret
# evidence is bound to the exact pre-promotion manifest and is short-lived.
attest_candidate "$promotion_evidence_file" "$release_manifest_file"
"$PYTHON_BIN" -m pipeline promote-release \
  --manifest "$release_manifest_file" \
  --attestation-file "$promotion_evidence_file" \
  --active-release-file "$ACTIVE_RELEASE_FILE"

cleanup_attestation
trap - EXIT
if [[ -n "${RELEASE_ARCHIVE_OUTPUT:-}" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/build-runtime-release-archive.py" \
    --active-release-file "$ACTIVE_RELEASE_FILE" \
    --runs-root "${RELEASE_RUNS_ROOT:-/data/releases/runs/mbzuai_main}" \
    --output "$RELEASE_ARCHIVE_OUTPUT"
fi
echo "Release checks passed and active pointer was promoted: $ACTIVE_RELEASE_FILE"
