# Quick Start Guide - Desktop App2

**Get up and running in 5 minutes!**

---

## Prerequisites

✅ Python 3.8+ installed
✅ Node.js 16+ installed
✅ OpenAI API key ready ([Get one here](https://platform.openai.com/api-keys))

---

## Installation (2 minutes)

### Windows

```powershell
cd desktop-app2
.\setup.ps1
```

### Linux/Mac

```bash
cd desktop-app2
./setup.sh
```

**Wait for**:
- Python virtual environment creation
- Python packages installation (streamlit, openai, etc.)
- Node.js packages installation (electron, axios, etc.)

---

## First Launch (1 minute)

### Windows

```powershell
.\start.ps1
```

### Linux/Mac

```bash
./start.sh
```

**You'll see**:
1. Terminal shows "Starting Desktop App2..."
2. Python backend starts
3. Electron window opens (might take 15-30 seconds first time)
4. Streamlit loads with API key setup screen

---

## Configuration (1 minute)

### Step 1: Enter API Key

On the setup screen:
1. Paste your OpenAI API key (starts with `sk-`)
2. Select GPT model (GPT-3.5 Turbo recommended)
3. Click "💾 Save & Continue"

**Your key is encrypted and stored locally!**

### Step 2: Verify Setup

After clicking Save:
- Screen reloads
- You should see 4 tabs:
  - 💬 Live Chat
  - 📊 RHO Analysis
  - 🎯 PHI Benchmark
  - ⚙️ Settings

**You're ready!**

---

## First Conversation (2 minutes)

### Tab 1: Live Chat

1. **Start Conversation**: Click "▶ Start New Conversation" button
2. **Send Message**: Type in chat input: "Hello, how are you?"
3. **Watch Metrics**: Right panel shows real-time R, v, a, L values
4. **Continue Chat**: Have a few more turns
5. **End Conversation**: Click "⏹ End Conversation" when done

### Tab 2: RHO Analysis

1. Navigate to "📊 RHO Analysis" tab
2. Select your conversation from dropdown
3. **View Results**:
   - Final RHO (ρ) value
   - Classification (ROBUST/REACTIVE/FRAGILE)
   - Cumulative risk charts
   - RHO timeline

### Tab 3: PHI Benchmark

1. Complete 2-3 conversations first (in Tab 1)
2. Navigate to "🎯 PHI Benchmark" tab
3. **View Results**:
   - PHI (Φ) score
   - PASS/FAIL classification
   - Breakdown table
   - Fragility distribution

---

## Common First-Run Issues

### Issue: "Failed to import App4 components"

**Solution**: Make sure you're in the `algorithm_work` directory and App4 exists:

```bash
# Check App4 is present
ls deployment/app4_unified_dashboard/
# Should show: app.py, core/, ui/, utils/

# If missing, you need the full project
```

### Issue: Electron window doesn't open

**Solution**: Check if Streamlit started:

```bash
# In another terminal
curl http://localhost:8501
# Should return HTML
```

If not, check Python errors in original terminal.

### Issue: "Invalid API key"

**Solution**:
- Verify key starts with `sk-`
- Check for extra spaces
- Get a new key from OpenAI dashboard

---

## What's Next?

✅ **Read the full README.md** for all features
✅ **Try different GPT models** (Settings → Model Selection)
✅ **Adjust algorithm parameters** (Sidebar → Algorithm section)
✅ **Export conversations** (Tab 1 → Export button)
✅ **Analyze multiple conversations** (Tab 3 → PHI Benchmark)

---

## Testing Without API Key

Want to test the interface without spending money?

1. On API key setup screen
2. Expand "🧪 Test without API key (Mock Mode)"
3. Click "Use Mock Mode"
4. App simulates responses (no real AI)

Perfect for:
- Learning the interface
- Testing features
- Demos and training

---

## Command Cheat Sheet

### Setup
```bash
./setup.sh          # Linux/Mac
.\setup.ps1         # Windows
```

### Run
```bash
./start.sh          # Linux/Mac
.\start.ps1         # Windows
```

### Build Installer
```bash
cd electron
npm run build:win   # Windows
npm run build:mac   # macOS
npm run build:linux # Linux
```

### Development Mode (with DevTools)
```bash
cd electron
npm run dev
```

---

**Time to first chat**: ~5 minutes
**Difficulty**: Easy
**Help**: See docs/TROUBLESHOOTING.md
