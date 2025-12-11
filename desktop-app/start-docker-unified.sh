#!/bin/bash
# Quick Docker Compose launcher for Unified Dashboard

set -e

echo "=========================================="
echo "  Unified Dashboard - Docker Compose"
echo "=========================================="
echo ""

cd "$(dirname "${BASH_SOURCE[0]}")"

# Check Docker
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running"
    exit 1
fi

echo "🏗️  Building and starting container..."
echo ""

# Use docker compose (v2 - no hyphen)
docker compose -f docker-compose.unified.yml up --build -d

echo ""
echo "✅ Dashboard is starting!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🌐 http://localhost:8501"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Commands:"
echo "  Logs:    docker compose -f docker-compose.unified.yml logs -f"
echo "  Stop:    docker compose -f docker-compose.unified.yml down"
echo "  Restart: docker compose -f docker-compose.unified.yml restart"
echo ""
