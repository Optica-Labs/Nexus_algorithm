# Vector Precognition Desktop App

**AI Safety Monitoring Desktop Application with ChatGPT Integration**

A standalone desktop application that provides real-time AI safety monitoring for ChatGPT conversations using the Vector Precognition algorithm.

## 🌟 Features

### Core Functionality
- **Live ChatGPT Integration**: Chat directly with GPT-4/GPT-3.5 through the desktop app
- **Real-Time Risk Analysis**: Every conversation turn analyzed through Vector Precognition
- **Dynamic Visualizations**: 4-panel risk dynamics updated in real-time
- **Multi-Model Support**: Switch between GPT-4, GPT-4 Turbo, GPT-3.5
- **Secure API Key Storage**: Keys encrypted and stored locally via Electron

### Safety Metrics (Per Turn)
- **R(N)**: Risk Severity - Cosine distance from safe harbor
- **v(N)**: Risk Rate - Velocity of drift in vector space
- **a(N)**: Guardrail Erosion - Acceleration (early warning signal)
- **z(N)**: Failure Potential - Weighted risk combination
- **L(N)**: Likelihood of Breach - Sigmoid probability (0-1)
- **ρ (rho)**: Robustness Index - Model resistance to manipulation

### Advanced Features
- **Multi-Conversation Tracking**: Analyze safety across multiple chat sessions
- **Alert System**: Visual warnings when breach likelihood exceeds 80%
- **Conversation Export**: Save full chat history with metrics to JSON
- **Configurable VSAFE**: Customize safe harbor anchor point
- **Cross-Platform**: Windows, macOS, Linux support

---

## 📋 Prerequisites

### Required Software
- **Python 3.8+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** - [Download](https://nodejs.org/)
- **npm** (comes with Node.js)

### Required Credentials
1. **OpenAI API Key** - Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. **AWS Credentials** (for embeddings) - Configure via AWS CLI or environment variables:
   ```bash
   export AWS_ACCESS_KEY_ID=your_key_here
   export AWS_SECRET_ACCESS_KEY=your_secret_here
   export AWS_DEFAULT_REGION=us-east-1
   ```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone/navigate to desktop-app directory
cd desktop-app

# Run setup script (installs all dependencies)
./setup.sh
```

**What setup.sh does:**
- Creates Python virtual environment
- Installs Python dependencies (Streamlit, OpenAI, scikit-learn, etc.)
- Installs Node.js dependencies (Electron, electron-store)
- Verifies all requirements

### 2. Configure AWS (Required for Embeddings)

```bash
# Option A: Environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Option B: AWS CLI (preferred)
aws configure
```

### 3. Initialize PCA Models (First Time Only)

```bash
cd python-backend
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# Quick initialization
python -c "from shared.pca_pipeline import PCATransformer; PCATransformer()"

cd ..
```

### 4. Launch Desktop App

```bash
./start.sh
```

The app will:
1. Start Python backend (Streamlit server on port 8501)
2. Launch Electron window
3. Display API key configuration screen

---

## 🎯 Usage Guide

### First Launch - Configuration

1. **Enter OpenAI API Key**
   - Paste your API key (starts with `sk-`)
   - Key is stored securely using Electron's encrypted storage

2. **Select Model**
   - GPT-4 Turbo (recommended for quality)
   - GPT-4 (balanced)
   - GPT-3.5 Turbo (faster, cheaper)

3. **Configure VSAFE** (Safe Harbor Anchor)
   - Default: "I am a helpful, harmless, and honest AI assistant."
   - Safety Focused: "I prioritize user safety and ethical guidelines..."
   - Custom: Write your own anchor text

4. **Click "Save & Continue"**

### Chatting with Safety Monitoring

Once configured, you'll see:

**Left Panel** - Chat Interface
- Type messages in the input box
- Conversation history displays above
- Real-time timestamps for each turn

**Right Panel** - Risk Analysis Dashboard
- **User Turn Metrics**: Your message risk profile
- **Assistant Turn Metrics**: ChatGPT's response risk
- **Live Chart**: Risk trajectory over time
- **Robustness Index (ρ)**: Overall model safety
  - ρ < 1.0 = **ROBUST** ✅ (model resisted manipulation)
  - ρ = 1.0 = **REACTIVE** ⚠️ (model matched user risk)
  - ρ > 1.0 = **FRAGILE** ❌ (model amplified risk)

**Sidebar** - Controls
- Temperature slider (0.0-2.0)
- Max tokens slider
- Current turn counter
- Robustness gauge
- **New Conversation** button (saves current to history)
- **Change API Key** button

### Understanding Alerts

When **L(N) > 0.8** (80% breach probability):
- 🚨 Red alert appears in right panel
- Full-screen warning modal shows:
  > "⚠️ AI Guardrail Erosion Threshold Exceeded"
  > "Please close this context window and restart..."

**Action**: Start a new conversation immediately.

---

## 📊 How It Works

### Architecture Overview

```
User Input → ChatGPT API → Response
     ↓                         ↓
   Embed                     Embed
     ↓                         ↓
  1536-D                    1536-D
     ↓                         ↓
   PCA (2-D)               PCA (2-D)
     ↓                         ↓
   Vector                   Vector
     ↓___________↓___________↓
              ↓
    Vector Precognition Algorithm
              ↓
    R(N), v(N), a(N), z(N), L(N), ρ
              ↓
       Live Visualization
```

### Algorithm Flow

1. **User types message** → Sent to ChatGPT API
2. **ChatGPT responds** → Response captured
3. **Both messages embedded** → AWS Bedrock Titan (1536-D)
4. **Dimensionality reduction** → PCA to 2-D vectors
5. **Risk calculation**:
   - **R(N)**: Distance from VSAFE anchor
   - **v(N)**: Distance from previous turn (velocity)
   - **a(N)**: Change in velocity (acceleration)
   - **z(N)**: Weighted combination → z = 1.5R + 1.0v + 3.0a - 2.5
   - **L(N)**: Sigmoid(z) → Breach probability
6. **Robustness tracking**:
   - Cumulative user risk vs. model risk
   - ρ = Σ(model_risk) / Σ(user_risk)
7. **UI update** → Charts, metrics, alerts

---

## 🔧 Configuration Files

### `electron/package.json`
- Electron app metadata
- Build configuration for installers
- Dependencies (electron, electron-store, axios)

### `python-backend/requirements.txt`
- Python dependencies
- Key packages: `streamlit`, `openai`, `boto3`, `scikit-learn`

### `python-backend/shared/config.py`
- Algorithm weights (wR, wv, wa, b)
- Default VSAFE text
- Alert thresholds

---

## 📁 Project Structure

```
desktop-app/
├── electron/                    # Electron main process
│   ├── main.js                 # Window management, IPC handlers
│   ├── preload.js              # Secure bridge to renderer
│   ├── package.json            # Dependencies & build config
│   └── node_modules/           # (generated)
│
├── python-backend/              # Streamlit backend
│   ├── app.py                  # Main Streamlit app with ChatGPT
│   ├── chatgpt_client.py       # OpenAI API client + risk monitor
│   ├── requirements.txt        # Python dependencies
│   ├── venv/                   # (generated) Virtual environment
│   │
│   ├── shared/                 # Shared modules (from deployment/)
│   │   ├── pca_pipeline.py    # Text → 2D vector pipeline
│   │   ├── config.py          # Algorithm configuration
│   │   └── visualizations.py  # Plotting utilities
│   │
│   ├── models/                 # PCA trained models
│   │   ├── pca_model.pkl
│   │   └── embedding_scaler.pkl
│   │
│   ├── core/                   # Core processing
│   ├── ui/                     # UI components
│   └── utils/                  # Utilities
│
├── resources/                   # App icons & assets
│   ├── icon.png
│   ├── icon.ico  (Windows)
│   └── icon.icns (macOS)
│
├── docs/                        # Documentation
│   ├── ARCHITECTURE.md
│   ├── DEVELOPER_GUIDE.md
│   └── API_REFERENCE.md
│
├── setup.sh                     # One-time setup script
├── start.sh                     # Launch script
└── README.md                    # This file
```

---

## 🛠️ Development

### Running in Development Mode

```bash
# Terminal 1: Start Python backend manually
cd python-backend
source venv/bin/activate
streamlit run app.py --server.port 8501

# Terminal 2: Start Electron in dev mode
cd electron
npm run dev
```

**Dev mode features:**
- Hot reload for Python changes (Streamlit auto-reloads)
- Electron DevTools open by default
- Verbose logging

### Testing ChatGPT Integration

```python
# Test ChatGPT client standalone
cd python-backend
source venv/bin/activate
python

>>> from chatgpt_client import create_chatgpt_monitor
>>> import numpy as np
>>>
>>> monitor = create_chatgpt_monitor(api_key="sk-...", model="gpt-3.5-turbo")
>>> monitor.set_vsafe_anchor(np.array([0.0, 0.0]))
>>>
>>> # Send test message
>>> response = await monitor.send_message(
...     user_message="Hello!",
...     user_vector=np.array([0.1, 0.1])
... )
>>> print(response['assistant_message'])
```

### Modifying Algorithm Weights

Edit `python-backend/shared/config.py`:

```python
DEFAULT_WEIGHTS = {
    'wR': 1.5,   # Risk severity weight
    'wv': 1.0,   # Risk rate weight
    'wa': 3.0,   # Erosion weight (highest impact)
    'b': -2.5    # Bias term
}
```

**Effect**: Higher `wa` = more sensitive to rapid drift.

---

## 📦 Building Installers

### Windows (.exe)

```bash
cd electron
npm run build:win
# Output: electron/dist/Vector Precognition Setup.exe
```

### macOS (.dmg)

```bash
cd electron
npm run build:mac
# Output: electron/dist/Vector Precognition.dmg
```

### Linux (.AppImage, .deb)

```bash
cd electron
npm run build:linux
# Output: electron/dist/vector-precognition.AppImage
```

**Note**: Installers bundle Python backend automatically.

---

## 🐛 Troubleshooting

### "PCA models not found"

```bash
cd python-backend
source venv/bin/activate
python -c "from shared.pca_pipeline import PCATransformer; PCATransformer()"
```

### "Invalid API key" Error

- Verify key starts with `sk-`
- Check balance at [platform.openai.com/usage](https://platform.openai.com/usage)
- Try regenerating key

### "AWS Credentials Error"

```bash
# Verify credentials
aws sts get-caller-identity

# Or set environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### Streamlit Not Starting

```bash
# Check if port 8501 is already in use
lsof -i :8501  # Linux/Mac
netstat -ano | findstr :8501  # Windows

# Kill process or change port in electron/main.js
```

### Electron Window Blank

- Wait 30 seconds for Streamlit to fully start
- Check console logs in `electron/main.js` output
- Try restarting: `./start.sh`

---

## 🔒 Security

### API Key Storage
- Keys stored using `electron-store` with AES encryption
- Never transmitted to external servers (except OpenAI/AWS)
- Stored location:
  - Windows: `%APPDATA%/vector-precognition-desktop/config.json`
  - macOS: `~/Library/Application Support/vector-precognition-desktop/config.json`
  - Linux: `~/.config/vector-precognition-desktop/config.json`

### Data Privacy
- All conversations processed locally
- No telemetry or analytics sent to third parties
- Export files saved to user's local disk only

---

## 📚 Additional Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical system design
- **[DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** - Contributing guide
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Python module docs

---

## 🤝 Support

For issues, questions, or feedback:
- Open GitHub issue: [github.com/optica-labs/vector-precognition](https://github.com/optica-labs/vector-precognition)
- Email: support@opticalabs.com

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🎓 Citation

Based on research: *"AI Safety and Accuracy Using Guardrail Erosion and Risk Velocity in Vector Space"*

If using this application for research, please cite the white paper.
