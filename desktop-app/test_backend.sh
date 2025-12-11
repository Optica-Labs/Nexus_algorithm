#!/bin/bash
# Test Python backend without Electron
# This runs Streamlit directly so you can test ChatGPT integration

echo "========================================="
echo "Testing Python Backend (No Electron)"
echo "========================================="
echo ""

cd python-backend

# Activate venv
echo "🐍 Activating Python environment..."
source venv/bin/activate

# Check if PCA models exist
if [ ! -f "models/pca_model.pkl" ]; then
    echo "⚠️  PCA models not found. Initializing..."
    python -c "from shared.pca_pipeline import PCATransformer; PCATransformer()" 2>/dev/null || echo "Note: PCA will initialize on first run"
fi

echo ""
echo "🚀 Starting Streamlit backend..."
echo ""
echo "Open in your browser: http://localhost:8501"
echo "Press Ctrl+C to stop"
echo ""

# Run Streamlit
streamlit run app.py --server.port 8501 --server.headless true
