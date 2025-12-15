# Desktop App2 - Vector Precognition Desktop Application

AI Safety Desktop Application combining **App4 (Unified Safety Dashboard)** with **ChatGPT Integration**.

Every conversation with ChatGPT is analyzed in real-time using the **Vector Precognition algorithm** for guardrail erosion and risk velocity detection.

## Features

- **Live ChatGPT Integration**: Chat with GPT-3.5/GPT-4 directly in the app
- **Real-Time Risk Analysis**: Every turn analyzed with Vector Precognition metrics:
  - R(N): Risk Severity
  - v(N): Risk Rate  
  - a(N): Guardrail Erosion
  - z(N): Failure Potential
  - L(N): Likelihood of Breach
- **Conversation-Level Metrics**: RHO (ρ) Robustness Index per conversation
- **Visual Analytics**: 4-panel dynamics plots showing risk evolution
- **Data Export**: CSV download of all metrics
- **Secure API Key Storage**: Encrypted local storage (Electron Store)

## Quick Start

### For macOS Users (Native Desktop App)

```bash
# 1. Clone on macOS
git clone https://github.com/Optica-Labs/Nexus_algorithm.git
cd Nexus_algorithm/desktop-app2

# 2. Install dependencies
cd electron && npm install && cd ..
cd python-backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && cd ..

# 3. Run
./start.sh
```

See [QUICKSTART_MACOS.md](QUICKSTART_MACOS.md) for details.

### For WSL/Linux Users (Browser Mode)

```bash
# 1. Navigate to desktop-app2
cd /home/aya/work/optica_labs/algorithm_work/desktop-app2

# 2. Run in browser mode
./START_FINAL.sh

# 3. Open browser
# Go to: http://localhost:8502
```

See [START_HERE_WSL.md](START_HERE_WSL.md) for details.

## Platform Support

| Platform | Mode | Command | Output |
|----------|------|---------|--------|
| **macOS** | Native Desktop | `./start.sh` | Electron window |
| **macOS** | Browser | `./test-backend.sh` | http://localhost:8502 |
| **WSL** | Browser Only | `./START_FINAL.sh` | http://localhost:8502 |
| **Linux** | Native Desktop | `./start.sh` | Electron window |
| **Linux** | Browser | `./test-backend.sh` | http://localhost:8502 |
| **Windows** | Browser Only | `START_FINAL.sh` (WSL) | http://localhost:8502 |

## Architecture

```
Desktop App2
│
├── Electron Layer (macOS/Linux only)
│   ├── Native window wrapper
│   ├── Secure API key storage
│   └── Python process manager
│
├── Python Backend (All platforms)
│   ├── Streamlit UI (4 tabs)
│   ├── ChatGPT client (OpenAI API)
│   └── App4 integration
│
└── App4 Components (Vector Precognition)
    ├── Stage 1: Guardrail Erosion (embeddings → PCA → metrics)
    ├── Stage 2: RHO Calculator (robustness index)
    └── Stage 3: PHI Evaluator (model fragility score)
```

## Documentation

- **[QUICKSTART_MACOS.md](QUICKSTART_MACOS.md)** - 5-minute setup for macOS
- **[MACOS_TESTING_GUIDE.md](MACOS_TESTING_GUIDE.md)** - Complete macOS guide (building .app, DMG, troubleshooting)
- **[START_HERE_WSL.md](START_HERE_WSL.md)** - WSL/Linux browser mode guide
- **[WSL_TESTING_GUIDE.md](WSL_TESTING_GUIDE.md)** - Detailed WSL instructions
- **[FINAL_INSTRUCTIONS.md](FINAL_INSTRUCTIONS.md)** - Original development documentation

## Requirements

### All Platforms
- Python 3.8+
- Node.js 16+ (for Electron)
- OpenAI API key

### macOS/Linux (for native desktop app)
- GUI libraries (automatically available on desktop systems)

### WSL
- Use browser mode (Electron not supported in headless WSL)

## Building for Distribution

### macOS .app Bundle

```bash
cd electron
npm run build:mac
# Output: dist/Vector Precognition Desktop App4.app
```

### macOS DMG Installer

```bash
cd electron
npm run build:mac:dmg
# Output: dist/Vector-Precognition-Desktop-App4.dmg
```

See [MACOS_TESTING_GUIDE.md](MACOS_TESTING_GUIDE.md) for code signing and distribution.

## Project Structure

```
desktop-app2/
├── electron/                    # Electron main process
│   ├── main.js                 # App initialization, Python manager
│   ├── preload.js              # IPC security bridge
│   └── package.json            # Build configuration
│
├── python-backend/              # Streamlit backend
│   ├── app.py                  # Main UI (4 tabs)
│   ├── chatgpt_integration.py  # OpenAI client
│   ├── venv/                   # Virtual environment
│   └── requirements.txt        # Dependencies
│
├── start.sh                     # macOS/Linux launcher (native)
├── START_FINAL.sh              # WSL/Browser launcher
├── test-backend.sh             # Backend-only testing
│
└── docs/
    ├── QUICKSTART_MACOS.md
    ├── MACOS_TESTING_GUIDE.md
    └── START_HERE_WSL.md
```

## Usage Example

1. **Launch the app** (macOS: `./start.sh`, WSL: `./START_FINAL.sh` + browser)
2. **Setup Tab**: Enter OpenAI API key, select model (gpt-4)
3. **Chat Tab**: Send message: "Tell me about AI safety"
4. **See real-time metrics**:
   ```
   Turn 1 (User): Risk = 0.12, Velocity = 0.05, Erosion = 0.01
   Turn 2 (GPT): Risk = 0.08, Velocity = -0.04, Erosion = -0.03
   ```
5. **Conversation Analysis Tab**: View dynamics plot
6. **Safety Dashboard**: RHO = 0.67 (Robust - model resisted risk)

## Testing

### Test Backend Only

```bash
./test-backend.sh
```

Runs import verification and starts Streamlit without Electron.

### Test Full Stack (macOS)

```bash
./start.sh
```

Tests Electron + Python + App4 integration.

## Troubleshooting

### "Port 8502 already in use"

**WSL:**
```bash
./KILL_ROOT_STREAMLIT.sh
./START_FINAL.sh
```

**macOS:**
```bash
lsof -ti:8502 | xargs kill -9
./start.sh
```

### "libnss3.so: cannot open shared object file" (WSL)

This is expected - use browser mode:
```bash
./START_FINAL.sh
# Then open http://localhost:8502 in Windows browser
```

### "Failed to import App4 components"

Ensure you have the full project structure including `deployment/` folder:
```bash
ls -la ../deployment/app4_unified_dashboard
ls -la ../deployment/shared
```

See platform-specific guides for more troubleshooting.

## Development

### Run backend in watch mode

```bash
cd python-backend
source venv/bin/activate
streamlit run app.py --server.port 8502
```

Edit `app.py`, save, and Streamlit auto-reloads.

### Run Electron in dev mode

```bash
cd electron
npm start
```

Opens Electron with DevTools enabled (`Cmd+Option+I`).

## Contributing

When adding features:
1. Test in both browser mode (WSL) and native mode (macOS)
2. Update relevant documentation
3. Follow MLOps best practices (modular code, logging, error handling)
4. Update project status documents

## License

Part of the Vector Precognition AI Safety Research Project.

See main repository README for citation and white paper details.

## Support

- **macOS Issues**: See [MACOS_TESTING_GUIDE.md](MACOS_TESTING_GUIDE.md)
- **WSL Issues**: See [START_HERE_WSL.md](START_HERE_WSL.md)
- **Backend Issues**: Run `./test-backend.sh` for diagnostics

---

**Ready to test AI safety in action!** 🚀
