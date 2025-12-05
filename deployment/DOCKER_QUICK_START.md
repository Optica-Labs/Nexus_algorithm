# Docker Quick Start - Vector Precognition Suite

## One-Minute Setup

```bash
# 1. Navigate to deployment directory
cd deployment

# 2. Configure environment (add your AWS credentials)
nano .env

# 3. Build and start all 4 apps
docker-compose up -d

# 4. Access applications
open http://localhost:8501  # App 1: Guardrail Erosion
open http://localhost:8502  # App 2: RHO Calculator
open http://localhost:8503  # App 3: PHI Evaluator
open http://localhost:8504  # App 4: Unified Dashboard
```

## What You Need

### Required
- **Docker Desktop** (or Docker Engine 20.10+)
- **AWS Credentials** (Access Key ID + Secret)
  - See [docs/ENVIRONMENT_SETUP_GUIDE.md](docs/ENVIRONMENT_SETUP_GUIDE.md) for how to get these

### Optional
- LLM API endpoints (for App 4 live chat)
- Or use Mock Client (no API keys needed)

## Environment Setup

Edit the `.env` file and add your AWS credentials:

```env
# Required
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1

# Optional (for App 4)
GPT35_ENDPOINT=
CLAUDE_ENDPOINT=
MISTRAL_ENDPOINT=
```

**Where to get AWS credentials?** See [docs/ENVIRONMENT_SETUP_GUIDE.md](docs/ENVIRONMENT_SETUP_GUIDE.md)

## Common Commands

```bash
# Build images
docker-compose build

# Start all apps (detached)
docker-compose up -d

# Start specific app
docker-compose up app1-guardrail-erosion

# View logs
docker-compose logs -f

# Stop all apps
docker-compose down

# Restart after .env changes
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check status
docker-compose ps
```

## Verification

```bash
# Check health
curl http://localhost:8501/_stcore/health
curl http://localhost:8502/_stcore/health
curl http://localhost:8503/_stcore/health
curl http://localhost:8504/_stcore/health

# All should return: {"status": "ok"}
```

## Shipping to Another Machine

### Method 1: Copy Deployment Folder (Easiest)

```bash
# On source machine
cd /path/to/algorithm_work
tar czf deployment.tar.gz deployment/
scp deployment.tar.gz user@target:/path/

# On target machine
tar xzf deployment.tar.gz
cd deployment
nano .env  # Add AWS credentials
docker-compose up -d
```

### Method 2: Export Docker Images

```bash
# On source machine
docker-compose build
docker save $(docker-compose config | grep image: | awk '{print $2}') | gzip > vp-suite.tar.gz
scp vp-suite.tar.gz user@target:/path/

# On target machine
gunzip -c vp-suite.tar.gz | docker load
cd deployment
docker-compose up -d
```

## Troubleshooting

### Issue: "No such file or directory: .env"
```bash
cp .env.example .env
nano .env  # Add AWS credentials
```

### Issue: "NoCredentialsError"
```bash
# Verify .env has AWS credentials
cat .env | grep AWS

# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Issue: "Port already in use"
```bash
# Find and kill process
lsof -i :8501
kill -9 <PID>

# Or change ports in docker-compose.yml
```

### Issue: "Models not found"
```bash
# Ensure models directory exists
ls models/
# Should show: pca_model.pkl, embedding_scaler.pkl

# If missing, copy from source machine
```

## Documentation

- **[DOCKER_DEPLOYMENT_README.md](docs/DOCKER_DEPLOYMENT_README.md)** - Complete deployment guide
- **[ENVIRONMENT_SETUP_GUIDE.md](docs/ENVIRONMENT_SETUP_GUIDE.md)** - How to get AWS credentials
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Developer onboarding
- **[BUSINESS_VALUE.md](docs/BUSINESS_VALUE.md)** - Business case and features
- **[PROJECT_STATUS.md](docs/PROJECT_STATUS.md)** - Current project status

## Application Overview

| App | Port | Purpose | Key Features |
|-----|------|---------|--------------|
| **App 1** | 8501 | Guardrail Erosion Analyzer | Turn-by-turn risk analysis, 4 input methods |
| **App 2** | 8502 | RHO Robustness Calculator | Per-conversation model evaluation |
| **App 3** | 8503 | PHI Model Evaluator | Multi-model benchmarking |
| **App 4** | 8504 | Unified Dashboard | Real-time monitoring + live chat |

## Testing Without AWS Credentials

You can test the apps without AWS credentials using:

1. **Manual Input** (App 1) - Enter 2D vectors directly
2. **Mock Client** (App 4) - Simulated LLM responses
3. **Demo Mode** (App 3) - Sample data pre-loaded

For full functionality, AWS credentials are required.

## Production Checklist

- [ ] AWS credentials configured in `.env`
- [ ] PCA models present in `models/` directory
- [ ] Docker and Docker Compose installed
- [ ] Ports 8501-8504 available
- [ ] `.env` file permissions: `chmod 600 .env`
- [ ] `.env` not committed to git
- [ ] Health checks passing
- [ ] All 4 apps accessible in browser

## Need Help?

1. Check [docs/ENVIRONMENT_SETUP_GUIDE.md](docs/ENVIRONMENT_SETUP_GUIDE.md) for AWS setup
2. Check [docs/DOCKER_DEPLOYMENT_README.md](docs/DOCKER_DEPLOYMENT_README.md) for deployment issues
3. Check `docker-compose logs <app-name>` for error messages
4. Verify `.env` file has correct format (no spaces around `=`)

---

**Ready to deploy!** Run `docker-compose up -d` and access the apps at http://localhost:8501-8504
