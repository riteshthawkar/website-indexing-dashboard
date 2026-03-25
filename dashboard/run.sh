#!/usr/bin/env bash
# Start the MBZUAI Pipeline Dashboard
set -e

cd "$(dirname "$0")"
PROJECT_ROOT="$(cd .. && pwd)"

# Activate project virtual environment
VENV_PATH="$PROJECT_ROOT/env"
if [ -f "$VENV_PATH/bin/activate" ]; then
    echo "Activating virtual environment: $VENV_PATH"
    source "$VENV_PATH/bin/activate"
else
    echo "Warning: Virtual environment not found at $VENV_PATH, using system Python"
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

echo "Starting MBZUAI Pipeline Dashboard on http://0.0.0.0:8050"
exec python3 -m uvicorn app:app --host 0.0.0.0 --port 8050 --reload
