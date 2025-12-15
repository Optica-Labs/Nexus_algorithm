#!/bin/bash
# Test Python backend in browser mode (WSL compatible - no Electron needed)

set -e

echo "========================================="
echo "Testing Desktop App2 Backend (Browser Mode)"
echo "========================================="
echo ""
echo "This will test the Python backend without Electron."
echo "Perfect for WSL environments!"
echo ""

# Check if virtual environment exists
if [ ! -d "python-backend/venv" ]; then
    echo "❌ Virtual environment not found"
    echo "   Please run ./setup.sh first"
    exit 1
fi

# Activate virtual environment
echo "Activating Python environment..."
source python-backend/venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1
export ELECTRON_MODE=false  # Signal we're in browser mode

# Export AWS credentials if available
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    export AWS_ACCESS_KEY_ID
    export AWS_SECRET_ACCESS_KEY
    export AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1}
    echo "✓ AWS credentials configured"
fi

echo ""
echo "Starting Streamlit backend..."
echo "Once started, open your browser to: http://localhost:8502"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================="
echo ""

cd python-backend
streamlit run app.py --server.port 8502 --server.headless true --browser.gatherUsageStats false
