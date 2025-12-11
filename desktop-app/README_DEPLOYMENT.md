# Desktop App Deployment Options

This directory contains **two deployment methods** for running the Unified AI Safety Dashboard as a desktop application:

---

## 🖥️ Option 1: Electron Desktop App (Native Windows/macOS/Linux with GUI)

**For users with:** Windows Desktop, macOS, or Linux with X11 display server

**Features:**
- Native desktop window
- Secure credential storage via Electron
- Offline packaging and distribution
- Auto-update support

**Quick Start:**
```bash
# Linux/macOS
./start_unified.sh

# Windows PowerShell
.\start_unified.ps1

# Or manually
cd electron
npm run start:app4
```

**Documentation:** See [UNIFIED_DASHBOARD_GUIDE.md](./UNIFIED_DASHBOARD_GUIDE.md)

---

## 🐳 Option 2: Docker Deployment (WSL/Headless/Servers)

**For users with:** WSL, headless servers, or containerized deployments

**Features:**
- Browser-based access (no GUI required)
- Consistent environment across platforms
- Easy deployment and scaling
- Perfect for WSL users

**Quick Start:**
```bash
# Using Docker Compose (recommended)
./start-docker-unified.sh

# Or manually
docker-compose -f docker-compose.unified.yml up -d

# Access at http://localhost:8501
```

**Documentation:** See [DOCKER_DEPLOYMENT_WSL.md](./DOCKER_DEPLOYMENT_WSL.md)

---

## Which Option Should I Use?

| Your Situation | Recommended Option |
|----------------|-------------------|
| Running on **WSL** (Windows Subsystem for Linux) | 🐳 **Docker** |
| Running on **Windows Desktop** with GUI | 🖥️ **Electron** |
| Running on **macOS** | 🖥️ **Electron** |
| Running on **Linux Desktop** with X11 | 🖥️ **Electron** |
| Running on **headless server** | 🐳 **Docker** |
| Need **portable executable** for distribution | 🖥️ **Electron** |
| Want **browser-based** interface | 🐳 **Docker** |
| Deploying to **production cloud** | 🐳 **Docker** |

---

## What's Included

### Electron Desktop App Files
```
electron/
├── main.js              # Electron main process
├── preload.js           # IPC bridge
├── package.json         # Node dependencies & build config
└── node_modules/        # Electron dependencies

start_unified.sh         # Linux/macOS launcher
start_unified.ps1        # Windows PowerShell launcher
UNIFIED_DASHBOARD_GUIDE.md  # Full Electron guide
```

### Docker Deployment Files
```
Dockerfile.unified             # Docker image definition
docker-compose.unified.yml     # Docker Compose config
docker-run-unified.sh          # Simple Docker launcher
start-docker-unified.sh        # Docker Compose launcher
DOCKER_DEPLOYMENT_WSL.md       # Full Docker guide
```

### Other Files
```
python-backend/          # Legacy Streamlit backend (optional)
docs/                    # Additional documentation
resources/               # Icons and assets
*.sh, *.ps1             # Various setup/start scripts
README.md               # This file
```

---

## Prerequisites

### For Electron Desktop App
- **Node.js** 16+ and npm
- **Python** 3.9+ with required packages
- **Streamlit** and dependencies

### For Docker Deployment
- **Docker** Desktop (Windows/macOS) or Docker Engine (Linux)
- **Docker Compose** (usually included with Docker Desktop)

---

## Unified Dashboard (App4) Details

Both deployment options run the **same backend**: `deployment/app4_unified_dashboard`

**Features:**
- ✅ Live chat with multiple LLM endpoints (GPT, Claude, Mistral, etc.)
- ✅ Real-time guardrail erosion monitoring (Stage 1)
- ✅ Per-conversation RHO robustness calculation (Stage 2)
- ✅ Multi-conversation PHI fragility aggregation (Stage 3)
- ✅ Interactive visualizations and safety alerts
- ✅ Conversation export and reporting

**Tech Stack:**
- **Frontend:** Streamlit (Python web framework)
- **Backend:** Python with NumPy, scikit-learn, pandas
- **AI Integration:** OpenAI, Anthropic, Mistral APIs or AWS Lambda endpoints
- **Embeddings:** AWS Bedrock or local sentence-transformers

---

## Quick Reference

### Electron Commands
```bash
# Start with unified dashboard
npm run start:app4

# Development mode (with DevTools)
npm run dev -- --app=app4_unified_dashboard

# Build installer
npm run build

# Build for specific platform
npm run build:win    # Windows
npm run build:mac    # macOS
npm run build:linux  # Linux
```

### Docker Commands
```bash
# Start (Docker Compose)
docker-compose -f docker-compose.unified.yml up -d

# View logs
docker-compose -f docker-compose.unified.yml logs -f

# Stop
docker-compose -f docker-compose.unified.yml down

# Rebuild after changes
docker-compose -f docker-compose.unified.yml up --build -d
```

---

## Troubleshooting

### Electron Issues
- **"libnss3.so not found"**: You're in WSL - use Docker instead
- **"Cannot find module 'electron'"**: Run `npm install` in `electron/` directory
- **"Python backend failed"**: Install Python dependencies from `deployment/app4_unified_dashboard/requirements.txt`

### Docker Issues
- **"Cannot connect to Docker daemon"**: Start Docker Desktop
- **"Port 8501 in use"**: Stop other Streamlit instances or change port
- **Build takes too long**: First build takes 5-10 minutes - subsequent builds are faster

---

## Next Steps

1. **Choose your deployment method** (Electron for native GUI, Docker for WSL/headless)
2. **Follow the respective guide** (UNIFIED_DASHBOARD_GUIDE.md or DOCKER_DEPLOYMENT_WSL.md)
3. **Configure API keys** for LLM endpoints
4. **Test with sample conversations** to verify functionality
5. **Review safety monitoring** outputs and alerts

---

## Support & Documentation

- **Electron Guide:** [UNIFIED_DASHBOARD_GUIDE.md](./UNIFIED_DASHBOARD_GUIDE.md)
- **Docker Guide:** [DOCKER_DEPLOYMENT_WSL.md](./DOCKER_DEPLOYMENT_WSL.md)
- **Testing Guide:** [TESTING_GUIDE.md](./TESTING_GUIDE.md)
- **Installation Guide:** [INSTALLATION_GUIDE.md](./INSTALLATION_GUIDE.md)
- **Current Status:** [CURRENT_STATUS.md](./CURRENT_STATUS.md)

---

**Project:** Nexus Algorithm - AI Safety Monitoring System  
**Owner:** Optica Labs  
**Last Updated:** December 11, 2025
