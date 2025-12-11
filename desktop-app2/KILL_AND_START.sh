#!/bin/bash
# Kill any running Streamlit and start fresh

echo "Killing any running Streamlit processes..."
sudo pkill -9 -f streamlit
sudo pkill -9 -f "python.*app.py"

echo "Waiting for processes to terminate..."
sleep 2

echo "Checking if port 8501 is free..."
if sudo lsof -ti:8501 > /dev/null 2>&1; then
    echo "Port still in use, force killing..."
    sudo kill -9 $(sudo lsof -ti:8501)
    sleep 2
fi

echo ""
echo "Port 8501 is now free!"
echo ""
echo "Starting Desktop App2 backend..."
echo "================================================"
echo ""

cd /home/aya/work/optica_labs/algorithm_work/desktop-app2
source python-backend/venv/bin/activate
cd python-backend

echo "Streamlit starting on http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

streamlit run app.py
