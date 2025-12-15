#!/bin/bash
# Start script for Desktop App2 (App4 + ChatGPT) on Linux/Mac

set -e  # Exit on error

echo "========================================="
echo "Starting Desktop App2 - App4 + ChatGPT"
echo "========================================="
echo ""

# Check if setup was run
if [ ! -d "python-backend/venv" ]; then
    echo "❌ Python virtual environment not found"
    echo "   Please run ./setup.sh first"
    exit 1
fi

if [ ! -d "electron/node_modules" ]; then
    echo "❌ Node modules not found"
    echo "   Please run ./setup.sh first"
    exit 1
fi

# Activate Python virtual environment (for backend to use)
echo "Activating Python environment..."

# Resolve script base directory (so the script works when run from anywhere)
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source the virtualenv if present
if [ -f "$BASE_DIR/python-backend/venv/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$BASE_DIR/python-backend/venv/bin/activate"
fi

# Ensure venv bin is first on PATH so 'python' resolves to the venv executable
export PATH="$BASE_DIR/python-backend/venv/bin:$PATH"

# If venv provides python3 but not a 'python' symlink, create one so processes
# that call 'python' (not 'python3') will succeed.
if [ ! -x "$BASE_DIR/python-backend/venv/bin/python" ]; then
    if command -v python3 >/dev/null 2>&1; then
        echo "Creating 'python' symlink in venv to python3"
        ln -sf "$(command -v python3)" "$BASE_DIR/python-backend/venv/bin/python" || true
    fi
fi

echo "Using python: $(command -v python || command -v python3 || echo 'none')"
if ! command -v python >/dev/null 2>&1; then
    echo "❌ No 'python' executable found in venv or system PATH."
    echo "   If you have only python3, ensure the venv was created with python3 or install a 'python' command."
    echo "   You can recreate the venv with: python3 -m venv python-backend/venv && source python-backend/venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Export explicit python environment variables so spawned Electron child processes
# (which may run from an app bundle) can find the venv python even if PATH is
# sanitized by macOS or Electron.
export PYTHON="$BASE_DIR/python-backend/venv/bin/python"
export PYTHON_BIN="$BASE_DIR/python-backend/venv/bin/python"
echo "Exported PYTHON=$PYTHON"

# Export AWS credentials if configured
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    echo "✓ AWS credentials configured"
fi

# Start Electron app (which will start Python backend)
echo "Launching Electron app..."
echo ""
cd electron
# Pass --dev so Electron main.js uses the repository python-backend (dev mode)
# which contains the app.py and allows using the venv in the repo during development.
npm start -- --dev
