#!/bin/bash
# Desktop App Launcher for Unified Dashboard (App4)
# 
# This script launches the Electron desktop app with the unified
# AI safety dashboard from deployment/app4_unified_dashboard

set -e

echo "=========================================="
echo "  Unified Dashboard Desktop App Launcher"
echo "=========================================="
echo ""

# Navigate to electron directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELECTRON_DIR="$SCRIPT_DIR/electron"

cd "$ELECTRON_DIR"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing Electron dependencies..."
    npm install
fi

# Check for Python dependencies
echo ""
echo "🐍 Checking Python environment..."
echo "   Make sure you have installed dependencies from:"
echo "   - deployment/app4_unified_dashboard/requirements.txt"
echo "   - deployment/shared/requirements.txt (if exists)"
echo ""
echo "   Quick install:"
echo "   pip install -r ../../deployment/app4_unified_dashboard/requirements.txt"
echo ""

# Ask user if they want to proceed
read -p "Ready to launch? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Launch the Electron app with app4 backend
echo ""
echo "🚀 Starting Electron with Unified Dashboard..."
echo ""

npm run start:app4
