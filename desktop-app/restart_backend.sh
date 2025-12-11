#!/bin/bash
# Force restart Streamlit backend with cache clearing

echo "🔄 Restarting Streamlit Backend..."
echo ""

# Kill any existing Streamlit processes
echo "Stopping existing Streamlit processes..."
pkill -f "streamlit run" || echo "No existing processes found"
sleep 2

# Clear Streamlit cache
echo "Clearing Streamlit cache..."
cd python-backend
rm -rf .streamlit
rm -rf __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Activate venv
echo "Activating Python environment..."
source venv/bin/activate

echo ""
echo "🚀 Starting Streamlit with fresh cache..."
echo ""
echo "Open in your browser: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

# Start Streamlit with cache cleared
streamlit run app.py \
  --server.port 8501 \
  --server.headless true \
  --browser.gatherUsageStats false \
  --server.address localhost \
  --server.runOnSave true \
  --server.fileWatcherType auto
