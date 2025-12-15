#!/bin/bash
# Final, bulletproof startup script for Desktop App2

echo "============================================="
echo "Desktop App2 - Final Start Script"
echo "============================================="
echo ""

# Step 1: Kill any existing Streamlit
echo "[1/3] Cleaning up existing processes..."
sudo pkill -9 -f "streamlit.*8501" 2>/dev/null || true
sudo pkill -9 -f "streamlit.*app.py" 2>/dev/null || true
sleep 2
echo "      Done!"
echo ""

# Step 2: Verify we're in the right directory
echo "[2/3] Verifying directory structure..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/python-backend"

if [ ! -f "app.py" ]; then
    echo "ERROR: app.py not found!"
    echo "Current directory: $(pwd)"
    exit 1
fi

if [ ! -d "../../deployment/shared" ]; then
    echo "ERROR: deployment/shared directory not found!"
    echo "Make sure you're running from desktop-app2/"
    exit 1
fi

echo "      Directory structure OK!"
echo ""

# Step 3: Activate venv and start
echo "[3/3] Starting Streamlit..."
source venv/bin/activate

echo ""
echo "============================================="
echo "Starting Vector Precognition Desktop App2"
echo ""
echo "  Open your browser to:"
echo "    http://localhost:8502"
echo ""
echo "  Press Ctrl+C to stop"
echo "============================================="
echo ""

streamlit run app.py --server.port 8502 --server.headless true --browser.gatherUsageStats false
