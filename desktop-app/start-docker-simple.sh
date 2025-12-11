#!/bin/bash
# Simple Docker launcher for Unified Dashboard (no docker-compose)

set -e

echo "=========================================="
echo "  Unified Dashboard - Docker Direct"
echo "=========================================="
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd .. && pwd)"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Stop and remove existing container
echo "🧹 Cleaning up old containers..."
docker rm -f ai-safety-unified-dashboard 2>/dev/null || true

echo ""
echo "🏗️  Building Docker image..."
echo "   (First build takes 5-10 minutes - please wait)"
echo ""

# Build image
cd "$REPO_ROOT"
docker build -f desktop-app/Dockerfile.unified -t ai-safety-unified:latest .

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✓ Build successful"
echo ""
echo "🚀 Starting container..."

# Create output and logs directories if they don't exist
mkdir -p "$REPO_ROOT/output"
mkdir -p "$REPO_ROOT/logs"

# Run container
docker run -d \
    --name ai-safety-unified-dashboard \
    -p 8501:8501 \
    -v "$REPO_ROOT/output:/app/output" \
    -v "$REPO_ROOT/logs:/app/logs" \
    ai-safety-unified:latest

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Failed to start container"
    exit 1
fi

echo ""
echo "✅ Container started successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🌐 Open your browser:"
echo "     http://localhost:8501"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Useful commands:"
echo "  View logs:     docker logs -f ai-safety-unified-dashboard"
echo "  Stop:          docker stop ai-safety-unified-dashboard"
echo "  Restart:       docker restart ai-safety-unified-dashboard"
echo "  Remove:        docker rm -f ai-safety-unified-dashboard"
echo ""
echo "Checking startup logs..."
sleep 3
echo ""
docker logs ai-safety-unified-dashboard 2>&1 | tail -20
