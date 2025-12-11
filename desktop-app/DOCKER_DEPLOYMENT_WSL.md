# Docker Deployment for Unified Dashboard (WSL Guide)

This guide shows how to run the **Unified AI Safety Dashboard** in Docker on **WSL** (Windows Subsystem for Linux).

---

## Why Docker for WSL?

Electron requires a GUI display server (X11), which isn't available in standard WSL environments. Docker provides a **headless** solution where you access the Streamlit app via your **browser** instead of a native window.

---

## Prerequisites

### 1. Docker Desktop for Windows

**Install Docker Desktop:**
1. Download from: https://www.docker.com/products/docker-desktop
2. Install and enable **WSL 2 backend**
3. Start Docker Desktop

**Verify in WSL:**
```bash
docker --version
docker-compose --version
```

### 2. Python Dependencies (for development)

The Docker image handles all Python dependencies automatically. No manual installation needed!

---

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
cd desktop-app
./start-docker-unified.sh
```

Or manually:
```bash
cd desktop-app
docker-compose -f docker-compose.unified.yml up --build -d
```

### Option 2: Using Docker Run Script

```bash
cd desktop-app
./docker-run-unified.sh
```

### Option 3: Manual Docker Commands

**Build:**
```bash
cd /path/to/repository
docker build -f desktop-app/Dockerfile.unified -t ai-safety-unified:latest .
```

**Run:**
```bash
docker run -d \
  --name ai-safety-unified-dashboard \
  -p 8501:8501 \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/logs:/app/logs" \
  ai-safety-unified:latest
```

---

## Access the Dashboard

Once the container is running:

🌐 **Open your browser:** http://localhost:8501

The Streamlit interface will load automatically.

---

## Managing the Container

### View Logs
```bash
# Docker Compose
docker-compose -f docker-compose.unified.yml logs -f

# Docker run
docker logs -f ai-safety-unified-dashboard
```

### Stop Container
```bash
# Docker Compose
docker-compose -f docker-compose.unified.yml down

# Docker run
docker stop ai-safety-unified-dashboard
```

### Restart Container
```bash
# Docker Compose
docker-compose -f docker-compose.unified.yml restart

# Docker run
docker restart ai-safety-unified-dashboard
```

### Remove Container
```bash
# Docker Compose
docker-compose -f docker-compose.unified.yml down -v

# Docker run
docker rm -f ai-safety-unified-dashboard
```

### Rebuild After Code Changes
```bash
# Docker Compose
docker-compose -f docker-compose.unified.yml up --build -d

# Docker run
./docker-run-unified.sh
```

---

## Configuration

### API Keys

You can set API keys via environment variables:

**Method 1: Environment File (Recommended)**

Create `.env` in `desktop-app/`:
```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
MISTRAL_API_KEY=...
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=us-east-1
```

Then run:
```bash
docker-compose -f docker-compose.unified.yml up -d
```

**Method 2: Command Line**

```bash
docker run -d \
  --name ai-safety-unified-dashboard \
  -p 8501:8501 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  ai-safety-unified:latest
```

### Ports

Default port: **8501**

To change:
```bash
# Map to different host port
docker run -d -p 9000:8501 ai-safety-unified:latest
# Access at http://localhost:9000
```

### Persistent Storage

The Docker setup mounts two volumes:
- `./output` → Conversation exports
- `./logs` → Application logs

These persist on your host machine even if the container is removed.

---

## Troubleshooting

### "Cannot connect to Docker daemon"

**Fix:**
1. Start Docker Desktop for Windows
2. Wait for "Docker is running" indicator
3. In WSL, run: `docker ps`

### "Port 8501 already in use"

**Fix:**
```bash
# Find what's using the port
lsof -i :8501

# Kill existing Streamlit
pkill -f streamlit

# Or use different port
docker run -d -p 8502:8501 ai-safety-unified:latest
```

### "Container exits immediately"

**Check logs:**
```bash
docker logs ai-safety-unified-dashboard
```

Common issues:
- Missing Python dependencies (rebuild image)
- Missing model files in `deployment/models/`
- Port conflict

### "Module not found" errors

**Cause:** Missing dependencies in the build context.

**Fix:**
```bash
# Rebuild with no cache
docker build --no-cache -f desktop-app/Dockerfile.unified -t ai-safety-unified:latest .
```

### Build takes too long

**Solution:** The first build installs all dependencies (~5-10 minutes). Subsequent builds use cached layers and are much faster.

**Speed up:**
```bash
# Use Docker BuildKit
DOCKER_BUILDKIT=1 docker build -f desktop-app/Dockerfile.unified -t ai-safety-unified:latest .
```

---

## Architecture

```
┌─────────────────────────────────────────┐
│     Browser (Windows/WSL)               │
│     http://localhost:8501               │
└────────────────┬────────────────────────┘
                 │
                 │ HTTP
                 │
┌────────────────▼────────────────────────┐
│   Docker Container (WSL)                │
│                                         │
│   ┌──────────────────────────────────┐ │
│   │   Streamlit (Port 8501)          │ │
│   │   - UI rendering                 │ │
│   │   - WebSocket for live updates   │ │
│   └──────────────┬───────────────────┘ │
│                  │                      │
│   ┌──────────────▼───────────────────┐ │
│   │   App4 Unified Dashboard         │ │
│   │   - Pipeline orchestration       │ │
│   │   - Safety monitoring            │ │
│   │   - Multi-stage analysis         │ │
│   └──────────────┬───────────────────┘ │
│                  │                      │
│   ┌──────────────▼───────────────────┐ │
│   │   Shared Modules                 │ │
│   │   - PCA transformer              │ │
│   │   - API clients (OpenAI, etc)    │ │
│   │   - Vector processing            │ │
│   └──────────────────────────────────┘ │
│                                         │
│   Volumes:                              │
│   - /app/output → ./output              │
│   - /app/logs → ./logs                  │
└─────────────────────────────────────────┘
```

---

## Differences: Docker vs Electron Desktop App

| Feature | Docker (WSL) | Electron (Native) |
|---------|--------------|-------------------|
| **Access** | Browser (http://localhost:8501) | Native desktop window |
| **Platform** | WSL, Linux, headless servers | Windows, macOS, Linux (with GUI) |
| **Deployment** | `docker run` or `docker-compose` | Double-click .exe, .app, or AppImage |
| **Updates** | Pull new image | Auto-update or manual install |
| **API Keys** | Environment variables or Streamlit UI | Electron secure store |
| **Performance** | Slightly slower (containerization overhead) | Native speed |
| **Portability** | Highly portable, consistent environment | Platform-specific builds |

**Recommendation:**
- **WSL users:** Use Docker (this guide)
- **Windows/macOS users with GUI:** Use Electron desktop app
- **Server deployments:** Use Docker

---

## Production Deployment

### Using Docker in Production

**1. Multi-stage build for smaller images:**

```dockerfile
FROM python:3.11-slim as builder
# ... build dependencies ...

FROM python:3.11-slim
# Copy only necessary artifacts from builder
```

**2. Use proper secrets management:**

```bash
# Don't commit .env files
# Use Docker secrets or external secret managers
docker secret create openai_key ./openai_key.txt
```

**3. Add reverse proxy (nginx) for SSL:**

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./certs:/etc/nginx/certs
  
  unified-dashboard:
    # ... existing config ...
    ports:
      - "8501"  # Internal only
```

**4. Health monitoring:**

```bash
# Docker Compose already includes healthcheck
# Monitor with:
docker ps
docker inspect ai-safety-unified-dashboard
```

---

## Next Steps

1. **Test the Docker deployment** with sample conversations
2. **Configure API keys** for LLM endpoints
3. **Review logs** for any startup issues
4. **Export results** from the mounted `./output` directory
5. **Consider production hardening** if deploying to servers

---

## Additional Resources

- **Docker Documentation:** https://docs.docker.com/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **WSL Documentation:** https://docs.microsoft.com/en-us/windows/wsl/

---

**Last Updated:** December 11, 2025
