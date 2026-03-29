#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/env"

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m playwright install chromium

pushd "$ROOT_DIR/dashboard-ui" >/dev/null
npm ci
popd >/dev/null

echo
echo "Bootstrap complete."
echo "Backend:  bash scripts/dashboard.sh"
echo "Frontend: cd dashboard-ui && npm run dev -- --hostname 0.0.0.0 --port 3000"
