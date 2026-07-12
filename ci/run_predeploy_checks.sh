#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

for dockerfile in Dockerfile Dockerfile.retriever; do
  if grep -E '^FROM ' "$dockerfile" | grep -vq '@sha256:'; then
    echo "Every base image in $dockerfile must be pinned by digest." >&2
    exit 1
  fi
done

for required_pattern in \
  '.env' \
  '.env.*' \
  '**/*.env' \
  '**/*.pem' \
  '**/*.key' \
  '**/*.crt' \
  '**/*.cer' \
  '**/*.p12' \
  '**/*.pfx' \
  '**/*.jks' \
  '**/*credentials*.json' \
  '**/*service-account*.json' \
  '**/*.db' \
  '**/*.sqlite3' \
  '**/*.backup' \
  'fix_state.py'
do
  if ! grep -Fqx "$required_pattern" .dockerignore; then
    echo "Missing required .dockerignore rule: $required_pattern" >&2
    exit 1
  fi
done

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="$PYTHON"
elif [[ -x "$ROOT_DIR/env/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No usable Python interpreter found." >&2
  exit 127
fi

"$PYTHON_BIN" -m pip check
"$PYTHON_BIN" -m pip_audit \
  --requirement "$ROOT_DIR/requirements.txt" \
  --strict \
  --progress-spinner off
"$PYTHON_BIN" "$ROOT_DIR/ci/validate_app_platform_spec.py"
"$PYTHON_BIN" -m pipeline dry-run --config mbzuai_production >/dev/null
"$PYTHON_BIN" -m pipeline validate-config --config mbzuai_production
"$PYTHON_BIN" -m pipeline doctor --config mbzuai_production
"$PYTHON_BIN" -m pytest pipeline/tests -q --disable-warnings
