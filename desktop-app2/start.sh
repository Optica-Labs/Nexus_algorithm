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
source python-backend/venv/bin/activate

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
npm start
