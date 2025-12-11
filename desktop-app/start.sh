#!/bin/bash
# Start script for Vector Precognition Desktop App

set -e

echo "========================================="
echo "Starting Vector Precognition Desktop"
echo "========================================="
echo ""

# Check if setup was run
if [ ! -d "python-backend/venv" ]; then
    echo "❌ Error: Virtual environment not found"
    echo "Please run ./setup.sh first"
    exit 1
fi

if [ ! -d "electron/node_modules" ]; then
    echo "❌ Error: Node modules not found"
    echo "Please run ./setup.sh first"
    exit 1
fi

# Activate Python venv
echo "🐍 Activating Python environment..."
source python-backend/venv/bin/activate

# Start Electron app (which will spawn Python backend)
echo "🚀 Launching desktop application..."
cd electron
npm start
