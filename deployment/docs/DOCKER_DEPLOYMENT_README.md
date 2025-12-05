# Docker Deployment Guide - Vector Precognition Suite

## Quick Start

### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB RAM minimum
- AWS credentials (for embeddings)

### One-Command Launch

**Start all 4 applications:**
```bash
cd deployment
docker-compose up -d
```

**Access the applications:**
- App 1 (Guardrail Erosion): http://localhost:8501
- App 2 (RHO Calculator): http://localhost:8502
- App 3 (PHI Evaluator): http://localhost:8503
- App 4 (Unified Dashboard): http://localhost:8504

**Stop all applications:**
```bash
docker-compose down
```

---

## Table of Contents
1. [Setup Instructions](#setup-instructions)
2. [Development Mode](#development-mode)
3. [Production Mode](#production-mode)
4. [Testing](#testing)
5. [Troubleshooting](#troubleshooting)

---

## Setup Instructions

### 1. Environment Configuration

Create `.env` file in `deployment/` directory:

```bash
# Copy example configuration
cp .env.example .env

# Edit with your credentials
nano .env
```

**Required variables:**
```env
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-east-1
```

**Optional variables (for App 4):**
```env
GPT35_ENDPOINT=https://your-endpoint.amazonaws.com/prod/chat
CLAUDE_ENDPOINT=https://your-endpoint.amazonaws.com/prod/chat
MISTRAL_ENDPOINT=https://your-endpoint.amazonaws.com/prod/chat
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
SESSION_TIMEOUT_MINUTES=60
```

### 2. Verify Model Files

Ensure PCA models exist:
```bash
ls -lh models/
# Should show:
# pca_model.pkl
# embedding_scaler.pkl
```

### 3. Build Docker Images

**Build all images:**
```bash
docker-compose build
```

**Build specific app:**
```bash
docker-compose build app1-guardrail-erosion
docker-compose build app2-rho-calculator
docker-compose build app3-phi-evaluator
docker-compose build app4-unified-dashboard
```

**Build with no cache (clean build):**
```bash
docker-compose build --no-cache
```

---

## Development Mode

### Run Single Application

**App 1 only:**
```bash
docker-compose up app1-guardrail-erosion
```

**App 2 only:**
```bash
docker-compose up app2-rho-calculator
```

**App 3 only:**
```bash
docker-compose up app3-phi-evaluator
```

**App 4 only:**
```bash
docker-compose up app4-unified-dashboard
```

### Live Development with Volume Mounts

For active development, mount source code:

```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

**Create `docker-compose.dev.yml`:**
```yaml
version: '3.8'
services:
  app1-guardrail-erosion:
    volumes:
      - ./app1_guardrail_erosion:/app/app1_guardrail_erosion
      - ./shared:/app/shared
```

### View Logs

**All apps:**
```bash
docker-compose logs -f
```

**Specific app:**
```bash
docker-compose logs -f app1-guardrail-erosion
docker-compose logs -f app2-rho-calculator
docker-compose logs -f app3-phi-evaluator
docker-compose logs -f app4-unified-dashboard
```

### Execute Commands Inside Container

```bash
# Open shell in running container
docker exec -it vp-app1-guardrail-erosion bash

# Run Python script
docker exec -it vp-app1-guardrail-erosion python -c "import sys; print(sys.version)"

# Check environment
docker exec -it vp-app1-guardrail-erosion env
```

---

## Production Mode

### 1. Build Optimized Images

```bash
# Build with production target
docker-compose build --parallel

# Tag images for registry
docker tag vp-app1-guardrail-erosion:latest your-registry.com/vp-app1:v1.0
docker tag vp-app2-rho-calculator:latest your-registry.com/vp-app2:v1.0
docker tag vp-app3-phi-evaluator:latest your-registry.com/vp-app3:v1.0
docker tag vp-app4-unified-dashboard:latest your-registry.com/vp-app4:v1.0
```

### 2. Push to Registry

```bash
# Login to registry
docker login your-registry.com

# Push all images
docker push your-registry.com/vp-app1:v1.0
docker push your-registry.com/vp-app2:v1.0
docker push your-registry.com/vp-app3:v1.0
docker push your-registry.com/vp-app4:v1.0
```

### 3. Deploy to Another Machine

**On target machine:**

```bash
# 1. Clone repository (or copy deployment folder)
git clone <repository-url>
cd deployment

# 2. Copy .env file
cp .env.example .env
nano .env  # Add credentials

# 3. Copy model files
# Transfer models/ directory from source machine
scp -r user@source:/path/to/models ./

# 4. Pull images from registry
docker-compose pull

# 5. Start services
docker-compose up -d

# 6. Verify health
docker-compose ps
```

### 4. Health Checks

```bash
# Check container status
docker-compose ps

# Check health endpoints
curl http://localhost:8501/_stcore/health
curl http://localhost:8502/_stcore/health
curl http://localhost:8503/_stcore/health
curl http://localhost:8504/_stcore/health

# Check logs
docker-compose logs --tail=50
```

---

## Testing

### Test Inside Docker

**1. Run all apps:**
```bash
docker-compose up -d
```

**2. Test App 1:**
```bash
# Open browser
open http://localhost:8501

# Or use curl
curl -I http://localhost:8501/_stcore/health
```

**3. Test data pipeline:**
```bash
# Generate test data in App 1
# Export results
# Import into App 2
# Verify RHO calculation
# Import into App 3
# Verify PHI calculation
```

**4. Test App 4 unified dashboard:**
```bash
open http://localhost:8504
# Test mock client (no API keys needed)
# Test real LLM endpoints (if configured)
```

### Integration Testing

```bash
# Run API endpoint tests
docker exec -it vp-app1-guardrail-erosion pytest /app/tests/

# Run with coverage
docker exec -it vp-app1-guardrail-erosion pytest --cov=/app --cov-report=html
```

### Load Testing

```bash
# Use Apache Bench
ab -n 100 -c 10 http://localhost:8501/

# Use wrk
wrk -t4 -c100 -d30s http://localhost:8501/
```

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker-compose logs app1-guardrail-erosion
```

**Common issues:**
- Missing `.env` file
- Invalid AWS credentials
- Missing model files
- Port already in use

**Solution:**
```bash
# Fix .env
cp .env.example .env && nano .env

# Check ports
netstat -an | grep 8501

# Recreate containers
docker-compose down -v
docker-compose up -d
```

### Model Files Not Found

**Error:** `FileNotFoundError: models/pca_model.pkl not found`

**Solution:**
```bash
# Ensure models exist on host
ls -l models/

# Check volume mount
docker-compose config | grep -A 3 volumes

# Rebuild with models
docker-compose build --no-cache
```

### AWS Credentials Error

**Error:** `botocore.exceptions.NoCredentialsError`

**Solution:**
```bash
# Verify .env file
cat .env | grep AWS

# Verify environment inside container
docker exec -it vp-app1-guardrail-erosion env | grep AWS

# Restart with new credentials
docker-compose down
docker-compose up -d
```

### Port Already in Use

**Error:** `Bind for 0.0.0.0:8501 failed: port is already allocated`

**Solution:**
```bash
# Find process using port
lsof -i :8501

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
# Change "8501:8501" to "8601:8501"
```

### Permission Denied

**Error:** `Permission denied: '/app/output'`

**Solution:**
```bash
# Fix volume permissions
docker-compose down
docker volume rm vp-app1-output
docker-compose up -d
```

### Container Memory Issues

**Error:** Container crashes or OOM

**Solution:**
```bash
# Increase Docker memory limit
# Docker Desktop: Preferences > Resources > Memory > 4GB+

# Or add to docker-compose.yml:
services:
  app1-guardrail-erosion:
    mem_limit: 2g
    memswap_limit: 2g
```

### Slow Build Times

**Issue:** Docker build takes too long

**Solution:**
```bash
# Use BuildKit for parallel builds
DOCKER_BUILDKIT=1 docker-compose build

# Build specific stage
docker build --target dependencies .

# Use layer caching
docker-compose build --parallel
```

---

## Advanced Usage

### Custom Network Configuration

```yaml
# docker-compose.override.yml
services:
  app1-guardrail-erosion:
    networks:
      - vp-network
      - external-network
```

### Volume Backup

```bash
# Backup output volumes
docker run --rm -v vp-app1-output:/data -v $(pwd):/backup alpine tar czf /backup/app1-output.tar.gz /data

# Restore
docker run --rm -v vp-app1-output:/data -v $(pwd):/backup alpine tar xzf /backup/app1-output.tar.gz -C /
```

### Multi-Stage Deployment

```bash
# Stage 1: Build base
docker build --target base -t vp-base:latest .

# Stage 2: Build dependencies
docker build --target dependencies -t vp-deps:latest .

# Stage 3: Build apps
docker build --target app1 -t vp-app1:latest .
```

---

## Performance Optimization

### Build Cache

```dockerfile
# Order from least to most frequently changed
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY shared/ ./shared/
COPY models/ ./models/
COPY app1_guardrail_erosion/ ./app1_guardrail_erosion/  # Last!
```

### Image Size Reduction

```bash
# Use slim Python image: python:3.10-slim
# Remove build tools after install
# Use multi-stage builds
# Check image size
docker images | grep vp-app
```

### Runtime Optimization

```yaml
# docker-compose.yml
services:
  app1-guardrail-erosion:
    cpus: '1.5'
    mem_limit: 2g
    restart: unless-stopped
```

---

## Monitoring

### Container Stats

```bash
# Real-time stats
docker stats

# Specific container
docker stats vp-app1-guardrail-erosion
```

### Logs Management

```bash
# Configure log rotation in docker-compose.yml
services:
  app1-guardrail-erosion:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Security Best Practices

1. **Use non-root user** (already implemented)
2. **Scan for vulnerabilities:**
   ```bash
   docker scan vp-app1-guardrail-erosion:latest
   ```
3. **Use secrets for credentials:**
   ```bash
   echo "my_secret" | docker secret create aws_access_key -
   ```
4. **Limit container resources** (see docker-compose.yml)
5. **Keep images updated:**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

---

## Shipping to Another Machine

### Method 1: Using Docker Registry (Recommended)

```bash
# On source machine
docker-compose build
docker tag deployment_app1-guardrail-erosion:latest registry.com/vp-app1:v1.0
docker push registry.com/vp-app1:v1.0

# On target machine
docker pull registry.com/vp-app1:v1.0
docker-compose up -d
```

### Method 2: Export/Import Images

```bash
# On source machine
docker save deployment_app1-guardrail-erosion:latest | gzip > vp-app1.tar.gz
docker save deployment_app2-rho-calculator:latest | gzip > vp-app2.tar.gz
docker save deployment_app3-phi-evaluator:latest | gzip > vp-app3.tar.gz
docker save deployment_app4-unified-dashboard:latest | gzip > vp-app4.tar.gz

# Transfer files to target machine
scp vp-app*.tar.gz user@target:/path/

# On target machine
gunzip -c vp-app1.tar.gz | docker load
gunzip -c vp-app2.tar.gz | docker load
gunzip -c vp-app3.tar.gz | docker load
gunzip -c vp-app4.tar.gz | docker load
```

### Method 3: Copy Entire Deployment Folder

```bash
# On source machine
cd /path/to/algorithm_work
tar czf deployment.tar.gz deployment/

# Transfer
scp deployment.tar.gz user@target:/path/

# On target machine
tar xzf deployment.tar.gz
cd deployment
docker-compose up -d
```

---

## Summary Commands

```bash
# Build
docker-compose build

# Start all
docker-compose up -d

# Start specific app
docker-compose up -d app1-guardrail-erosion

# View logs
docker-compose logs -f

# Stop all
docker-compose down

# Clean everything
docker-compose down -v --rmi all

# Health check
docker-compose ps
curl http://localhost:8501/_stcore/health

# Access container
docker exec -it vp-app1-guardrail-erosion bash
```

---

**Ready to deploy!** Follow the steps above to ship your Vector Precognition suite to any machine.
