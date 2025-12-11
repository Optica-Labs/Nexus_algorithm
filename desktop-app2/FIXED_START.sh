#!/bin/bash
# Fixed start script with all import paths corrected

echo "============================================="
echo "Desktop App2 - Starting Backend"
echo "============================================="
echo ""

# Kill any existing processes
echo "Cleaning up any existing Streamlit processes..."
sudo pkill -9 -f streamlit 2>/dev/null
sleep 2

echo "Starting Streamlit backend..."
echo ""
echo "Once loaded, open your browser to:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""
echo "============================================="
echo ""

cd /home/aya/work/optica_labs/algorithm_work/desktop-app2/python-backend
source venv/bin/activate

streamlit run app.py --server.port 8501
