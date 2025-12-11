# Desktop App2: Vector Precognition - App4 + ChatGPT Integration

**A desktop application combining the full App4 Unified Dashboard with ChatGPT for real-time AI safety monitoring.**

---

## 🎯 Overview

Desktop App2 is a native desktop application that brings together:

- ✅ **Full App4 Unified Dashboard** (all 4 tabs)
- ✅ **ChatGPT Integration** (GPT-3.5, GPT-4, GPT-4o)
- ✅ **Real-time Vector Precognition Analysis**
- ✅ **Multi-conversation tracking with RHO & PHI**
- ✅ **Secure API key storage via Electron**
- ✅ **Cross-platform desktop application** (Windows, Mac, Linux)

### What's Different from Desktop-App (v1)?

| Feature | Desktop-App (v1) | Desktop-App2 (v2) |
|---------|------------------|-------------------|
| Interface | Single page | **Full 4-tab App4 UI** |
| Models | ChatGPT only | ChatGPT (same, but better integrated) |
| RHO Analysis | In sidebar only | **Dedicated tab with visualizations** |
| PHI Benchmark | ❌ Missing | **✅ Full multi-conversation analysis** |
| Conversation History | ❌ Lost on reset | **✅ Stored and analyzable** |
| Settings Tab | ❌ Missing | **✅ Full configuration interface** |
| Architecture | Custom implementation | **Uses proven App4 codebase** |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** ([Download](https://www.python.org/downloads/))
- **Node.js 16+** ([Download](https://nodejs.org/))
- **OpenAI API Key** ([Get one](https://platform.openai.com/api-keys))
- **AWS Credentials** (for embeddings - optional if using existing PCA models)

### Installation

#### Linux/Mac

```bash
cd desktop-app2
./setup.sh
```

#### Windows

```powershell
cd desktop-app2
.\setup.ps1
```

### Running the App

#### Linux/Mac

```bash
./start.sh
```

#### Windows

```powershell
.\start.ps1
```

### First-Time Setup

1. **Enter OpenAI API Key**: The app will prompt you on first launch
2. **Select GPT Model**: Choose from GPT-3.5 Turbo, GPT-4, GPT-4o, etc.
3. **Save Configuration**: Your API key is encrypted and stored securely
4. **Start Chatting**: You'll see the full App4 interface with 4 tabs!

---

## 📊 Features

### Tab 1: Live Chat 💬

- **Real-time conversation** with ChatGPT
- **Live safety metrics** updated per turn:
  - **R(N)**: Risk Severity (distance from safe harbor)
  - **v(N)**: Risk Rate (velocity of drift)
  - **a(N)**: Guardrail Erosion (acceleration)
  - **L(N)**: Breach Likelihood (sigmoid probability)
- **RHO tracking**: Live robustness index in sidebar
- **4-panel dynamics visualization**: See risk evolution in real-time
- **Conversation controls**: Start, End, Export
- **Safety alerts**: Popup warnings when erosion threshold exceeded

### Tab 2: RHO Analysis 📊

- **Per-conversation analysis**: Select any past conversation
- **Robustness classification**:
  - ρ < 1.0 = **ROBUST** (model resisted manipulation)
  - ρ = 1.0 = **REACTIVE** (model matched user risk)
  - ρ > 1.0 = **FRAGILE** (model amplified user risk)
- **Cumulative risk plots**: User vs Model risk over turns
- **RHO timeline**: Track robustness drift
- **Export capabilities**: CSV + JSON

### Tab 3: PHI Benchmark 🎯

- **Multi-conversation aggregation**: Analyze all conversations together
- **PHI (Φ) score**: Model-level fragility metric
  - Φ < 0.1 = **PASS** (model is robust)
  - Φ ≥ 0.1 = **FAIL** (model is fragile)
- **Conversation breakdown table**: Contribution of each conversation
- **Fragility distribution histogram**: Visual analysis
- **Research-grade benchmarking**: Publication-ready metrics

### Tab 4: Settings ⚙️

- **Session information**: View active conversations, stats
- **System prompt editor**: Customize AI behavior
- **Algorithm parameters**: wR, wv, wa, b weights
- **Alert thresholds**: Configure safety limits
- **Export session**: Download complete session data

---

## 🛠️ Architecture

### High-Level Design

```
┌─────────────────────────────────────────────┐
│          Electron Main Process              │
│  - Window management                        │
│  - Python backend launcher                  │
│  - Secure API key storage                   │
└──────────────┬──────────────────────────────┘
               │
               ├──> IPC Bridge (preload.js)
               │
┌──────────────▼──────────────────────────────┐
│      Streamlit App (Python Backend)         │
│  ┌──────────────────────────────────────┐  │
│  │  app.py (Main Entry Point)           │  │
│  │  - API key setup screen              │  │
│  │  - App4 dashboard renderer           │  │
│  └──────────────┬───────────────────────┘  │
│                 │                           │
│  ┌──────────────▼───────────────────────┐  │
│  │  chatgpt_integration.py              │  │
│  │  - ChatGPT API client                │  │
│  │  - Compatible with App4 interface    │  │
│  └──────────────┬───────────────────────┘  │
│                 │                           │
│  ┌──────────────▼───────────────────────┐  │
│  │  App4 Components (from deployment/)  │  │
│  │  - PipelineOrchestrator              │  │
│  │  - VectorProcessor (Stage 1)         │  │
│  │  - RobustnessCalculator (Stage 2)    │  │
│  │  - FragilityCalculator (Stage 3)     │  │
│  │  - Visualizations                    │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
```

### Key Components

1. **Electron Layer** (`electron/`)
   - `main.js`: Main process, Python launcher, IPC handlers
   - `preload.js`: Security bridge for API access
   - `package.json`: Build configuration

2. **Python Backend** (`python-backend/`)
   - `app.py`: Main Streamlit app, API key setup, App4 integration
   - `chatgpt_integration.py`: ChatGPT client compatible with App4
   - `requirements.txt`: Python dependencies

3. **App4 Components** (imported from `../../deployment/app4_unified_dashboard/`)
   - All 4 tab renderers
   - Pipeline orchestrator
   - UI components (chat view, sidebar)
   - Core calculators (vector processor, RHO, PHI)

4. **Shared Modules** (imported from `../../deployment/shared/`)
   - PCA transformer (text → 2D vectors)
   - Visualizations (guardrail, RHO, PHI plots)
   - Configuration (default weights, VSAFE)

---

## 🔒 Security

### API Key Storage

- **Encrypted storage**: Uses `electron-store` with AES encryption
- **Secure key**: `vector-precognition-app4-secure-key-2024`
- **Never exposed**: Keys passed via environment variables to Python
- **User-controlled**: Keys stored locally, never sent to servers (except OpenAI)

### IPC Communication

- **Context isolation**: Renderer process can't access Node.js
- **Preload script**: Only exposes whitelisted APIs
- **No remote code**: All code bundled in application

---

## 📝 Configuration

### Environment Variables

Set these before running the app (optional):

```bash
# OpenAI API Key (can also be entered in UI)
export OPENAI_API_KEY="sk-..."

# AWS Credentials (for embeddings)
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"
```

### ChatGPT Models Supported

- **gpt-3.5-turbo** (Recommended for speed and cost)
- **gpt-4o-mini** (Balanced performance)
- **gpt-4o** (Latest and most capable)
- **gpt-4-turbo** (High performance)
- **gpt-4** (Classic, very capable)

### Algorithm Parameters

Default weights (configurable in sidebar):

```python
wR = 1.5   # Risk Severity weight
wv = 1.0   # Risk Rate weight
wa = 3.0   # Erosion weight (highest impact)
b = -2.5   # Bias term
```

Alert thresholds:

```python
alert_threshold = 0.8      # L(N) threshold for warnings
erosion_threshold = 0.3    # Erosion alert threshold
epsilon = 0.05             # RHO calculation epsilon
phi_threshold = 0.1        # PHI pass/fail threshold
```

---

## 🧪 Testing

### Mock Mode (No API Key Required)

You can test the interface without an OpenAI API key:

1. On the API key setup screen, expand "Test without API key"
2. Click "Use Mock Mode"
3. The app will simulate responses

This is great for:
- Testing the interface
- Training/demos
- Development

### Testing Workflow

1. **Setup Check**:
   ```bash
   python --version  # Should be 3.8+
   node --version    # Should be 16+
   ```

2. **Installation Test**:
   ```bash
   ./setup.sh  # or setup.ps1 on Windows
   # Should complete without errors
   ```

3. **Launch Test**:
   ```bash
   ./start.sh  # or start.ps1 on Windows
   # Electron window should open
   # Streamlit should load within 30 seconds
   ```

4. **API Key Test**:
   - Enter API key in UI
   - Should show success message
   - Should reload to App4 interface

5. **Chat Test**:
   - Tab 1: Start conversation
   - Send a message to ChatGPT
   - Verify response appears
   - Check metrics update (R, v, a, L)

6. **RHO Analysis Test**:
   - Complete a conversation (Tab 1)
   - Click "End Conversation"
   - Go to Tab 2
   - Select the conversation
   - Verify RHO calculated and visualizations shown

7. **PHI Benchmark Test**:
   - Complete 2-3 conversations
   - Go to Tab 3
   - Verify PHI score calculated
   - Check breakdown table and histogram

---

## 🏗️ Building Installers

### Windows

```bash
cd electron
npm run build:win
```

Output: `electron/dist/Vector Precognition App4 Setup.exe`

### macOS

```bash
cd electron
npm run build:mac
```

Output: `electron/dist/Vector Precognition App4.dmg`

### Linux

```bash
cd electron
npm run build:linux
```

Output: `electron/dist/Vector Precognition App4.AppImage`

---

## 📂 Project Structure

```
desktop-app2/
├── electron/
│   ├── main.js              # Main Electron process
│   ├── preload.js           # Security bridge
│   ├── package.json         # Electron config + build scripts
│   └── node_modules/        # (after npm install)
│
├── python-backend/
│   ├── app.py               # Main Streamlit app
│   ├── chatgpt_integration.py  # ChatGPT client
│   ├── requirements.txt     # Python dependencies
│   └── venv/                # (after setup)
│
├── resources/               # Icons for installers
│   ├── icon.png
│   ├── icon.ico
│   └── icon.icns
│
├── docs/                    # Additional documentation
│   ├── DEVELOPER_GUIDE.md
│   ├── API_REFERENCE.md
│   └── TROUBLESHOOTING.md
│
├── setup.sh                 # Linux/Mac setup
├── setup.ps1                # Windows setup
├── start.sh                 # Linux/Mac launcher
├── start.ps1                # Windows launcher
└── README.md                # This file
```

---

## 🐛 Troubleshooting

### "Failed to import App4 components"

**Cause**: Python can't find App4 modules

**Solution**:
```bash
# Verify App4 exists
ls ../deployment/app4_unified_dashboard/

# Check Python path in app.py
# Should add deployment/ and shared/ to sys.path
```

### "OpenAI package not installed"

**Cause**: Missing openai library

**Solution**:
```bash
cd python-backend
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1
pip install openai
```

### "Streamlit not starting"

**Cause**: Port 8501 already in use

**Solution**:
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill  # Linux/Mac
# Or restart computer
```

### "API key not persisting"

**Cause**: Electron store not working

**Solution**:
- Check if running in dev mode: `npm run dev`
- On first run, API key needs to be entered in UI
- After restart, it should auto-load from Electron storage

### WSL Issues

This app should work in WSL2 with proper X server setup. However, for best results:

**Recommended**: Run on native Windows with PowerShell scripts

**Alternative**: Use browser-only mode:
```bash
cd python-backend
source venv/bin/activate
streamlit run app.py
# Open http://localhost:8501 in browser
```

---

## 🤝 Contributing

This is part of the Vector Precognition research project.

### Development Workflow

1. **Make changes** to Python backend:
   ```bash
   cd python-backend
   source venv/bin/activate
   # Edit files
   streamlit run app.py  # Test in browser
   ```

2. **Test in Electron**:
   ```bash
   cd ../electron
   npm run dev  # Run in dev mode with DevTools
   ```

3. **Build installer**:
   ```bash
   npm run build  # Or build:win / build:mac / build:linux
   ```

---

## 📚 Documentation

- **README.md** (this file): Overview and quick start
- **docs/DEVELOPER_GUIDE.md**: In-depth development guide
- **docs/API_REFERENCE.md**: ChatGPT client API reference
- **docs/TROUBLESHOOTING.md**: Common issues and solutions

---

## 📄 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- Built on the **App4 Unified Dashboard** architecture
- Uses **OpenAI's ChatGPT** for language model capabilities
- Powered by **Electron** for cross-platform desktop deployment
- Based on research: *"AI Safety and Accuracy Using Guardrail Erosion and Risk Velocity in Vector Space"*

---

## 📞 Support

For issues:
1. Check **docs/TROUBLESHOOTING.md**
2. Review logs in Electron DevTools (Ctrl+Shift+I / Cmd+Option+I)
3. Check Python logs in terminal
4. Open an issue on GitHub

---

**Version**: 2.0.0
**Last Updated**: December 11, 2024
**Status**: Ready for testing
