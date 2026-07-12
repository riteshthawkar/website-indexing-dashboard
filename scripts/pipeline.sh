#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="$ROOT_DIR/env"

if [[ -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$VENV_PATH/bin/activate"
fi

cd "$ROOT_DIR"
if command -v python >/dev/null 2>&1; then
  exec python -m pipeline "$@"
fi

exec python3 -m pipeline "$@"
