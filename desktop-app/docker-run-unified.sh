#!/bin/bash
# Build and run the Unified Dashboard in Docker (WSL-friendly)

set -e

echo "=========================================="
echo "  Unified Dashboard - Docker Deployment"
echo "=========================================="
echo ""

# Navigate to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    echo "   Please start Docker Desktop or Docker daemon"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Build the Docker image
echo "🏗️  Building Docker image..."
docker build -f desktop-app/Dockerfile.unified -t ai-safety-unified:latest .

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo ""
echo "✓ Build successful"
echo ""

# Stop and remove existing container if running
echo "🧹 Cleaning up old containers..."
docker rm -f ai-safety-unified-dashboard 2>/dev/null || true

echo ""
echo "🚀 Starting Unified Dashboard..."
echo ""

# Run the container
docker run -d \
    --name ai-safety-unified-dashboard \
    -p 8501:8501 \
    -v "$REPO_ROOT/output:/app/output" \
    -v "$REPO_ROOT/logs:/app/logs" \
    ai-safety-unified:latest

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container"
    exit 1
fi

echo ""
echo "✅ Container started successfully!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Access the dashboard at:"
echo "  🌐 http://localhost:8501"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Useful commands:"
echo "  View logs:     docker logs -f ai-safety-unified-dashboard"
echo "  Stop:          docker stop ai-safety-unified-dashboard"
echo "  Restart:       docker restart ai-safety-unified-dashboard"
echo "  Remove:        docker rm -f ai-safety-unified-dashboard"
echo ""

# Show logs
echo "Waiting for Streamlit to start..."
sleep 3
docker logs ai-safety-unified-dashboard
