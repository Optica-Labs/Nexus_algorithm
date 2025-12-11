# Desktop App2 - WSL Testing Guide

**For Windows Subsystem for Linux (WSL) Users**

---

## Issue: Electron GUI Libraries Missing in WSL

The standard `./start.sh` script launches Electron, which requires GUI libraries that WSL doesn't have by default:

```
error while loading shared libraries: libnss3.so: cannot open shared object file
```

---

## ✅ Solution: Browser-Only Testing Mode

Desktop App2 works perfectly in **browser mode** on WSL! The Python backend (Streamlit) runs fine, and you can access the full App4 interface through your browser.

---

## Quick Start (Browser Mode)

### Step 1: Run Setup (if not done)

```bash
cd desktop-app2
./setup.sh
```

**Expected output**:
- ✓ Python 3.x found
- ✓ Node.js installed (not needed for browser mode, but setup checks it)
- ✓ Python packages installed
- ✓ Virtual environment created

### Step 2: Start Backend in Browser Mode

```bash
./test-backend.sh
```

**Expected output**:
```
=========================================
Testing Desktop App2 Backend (Browser Mode)
=========================================

Activating Python environment...
Starting Streamlit backend...
Once started, open your browser to: http://localhost:8501
```

### Step 3: Open in Browser

**From WSL**: Use Windows browser to access:
```
http://localhost:8501
```

**From Windows**: You can also access from Windows using:
```
http://localhost:8501
```
or
```
http://<your-wsl-ip>:8501
```

---

## What You'll See

### First Screen: API Key Setup

1. **Title**: "🔐 ChatGPT API Configuration"
2. **Step 1**: Enter your OpenAI API key
   - Get one from: https://platform.openai.com/api-keys
   - Format: `sk-...`
3. **Step 2**: Select GPT model
   - GPT-3.5 Turbo (recommended)
   - GPT-4, GPT-4o, etc.
4. **Save & Continue**: Click to proceed

### After Configuration: Full App4 Interface

You'll see 4 tabs:

1. **💬 Live Chat**
   - Start new conversation
   - Chat with ChatGPT
   - See real-time safety metrics (R, v, a, L)
   - RHO value in sidebar
   - End conversation button

2. **📊 RHO Analysis**
   - Select past conversations
   - View robustness classification
   - See cumulative risk plots
   - Export results

3. **🎯 PHI Benchmark**
   - Multi-conversation aggregation
   - PHI score (model fragility)
   - Pass/Fail indicator
   - Distribution histogram

4. **⚙️ Settings**
   - Session info
   - System prompt editor
   - Algorithm parameters
   - Export options

---

## Testing Checklist

### ✅ Basic Functionality

- [ ] Backend starts without errors
- [ ] Browser loads Streamlit interface
- [ ] API key setup screen appears
- [ ] Can enter and save API key
- [ ] App reloads to main interface
- [ ] All 4 tabs are visible

### ✅ Tab 1: Live Chat

- [ ] "Start New Conversation" button works
- [ ] Can type message in chat input
- [ ] Message sends to ChatGPT
- [ ] Response appears in chat
- [ ] Metrics update (R, v, a, L)
- [ ] RHO value updates in sidebar
- [ ] "End Conversation" button works

### ✅ Tab 2: RHO Analysis

- [ ] Conversation appears in dropdown after ending
- [ ] Can select conversation
- [ ] RHO value calculates
- [ ] Classification shows (ROBUST/REACTIVE/FRAGILE)
- [ ] Cumulative risk plot renders
- [ ] RHO timeline renders

### ✅ Tab 3: PHI Benchmark

- [ ] After 2-3 conversations, PHI calculates
- [ ] PHI score displays
- [ ] Pass/Fail classification shows
- [ ] Breakdown table populates
- [ ] Histogram renders

### ✅ Tab 4: Settings

- [ ] Session information displays
- [ ] System prompt is editable
- [ ] Algorithm parameters show
- [ ] Export button works

---

## Manual Testing Commands

### Check if Backend is Running

```bash
curl http://localhost:8501
# Should return HTML
```

### View Backend Logs

The `test-backend.sh` script shows logs in terminal. Look for:
- ✅ "You can now view your Streamlit app..."
- ✅ "Local URL: http://localhost:8501"
- ❌ Any Python errors or tracebacks

### Stop Backend

Press `Ctrl+C` in the terminal where `test-backend.sh` is running.

### Restart Backend

```bash
# Stop with Ctrl+C
# Then start again
./test-backend.sh
```

---

## Common Issues

### Issue: "Port 8501 already in use"

**Solution**:
```bash
# Kill process on port 8501
lsof -ti:8501 | xargs kill -9
# Then restart
./test-backend.sh
```

### Issue: "Failed to import App4 components"

**Solution**: Make sure you're in the algorithm_work directory:
```bash
pwd
# Should show: /home/aya/work/optica_labs/algorithm_work

cd /home/aya/work/optica_labs/algorithm_work/desktop-app2
./test-backend.sh
```

### Issue: "ModuleNotFoundError: No module named 'openai'"

**Solution**: Reinstall dependencies:
```bash
cd desktop-app2
source python-backend/venv/bin/activate
pip install -r python-backend/requirements.txt
```

### Issue: API key doesn't persist across restarts

**Expected behavior in browser mode**: API keys are stored in Streamlit session state only, so they're lost when you close the browser. This is normal for browser-only mode.

**For persistent storage**: Use the full Electron app on Windows (see below).

---

## WSL vs Full Electron

### Browser Mode (Current - WSL Compatible) ✅

**Pros**:
- ✅ Works perfectly in WSL
- ✅ No GUI libraries needed
- ✅ All App4 features work
- ✅ Fast to test and iterate

**Cons**:
- ❌ API key doesn't persist (session only)
- ❌ No native desktop window
- ❌ Requires browser open

### Full Electron Mode (Windows Native) ⭐

**To use full desktop app**:

1. **Open Windows PowerShell** (not WSL)

2. **Navigate to project**:
   ```powershell
   # Access WSL files from Windows
   cd \\wsl$\Ubuntu\home\aya\work\optica_labs\algorithm_work\desktop-app2
   ```

3. **Run setup**:
   ```powershell
   .\setup.ps1
   ```

4. **Launch**:
   ```powershell
   .\start.ps1
   ```

**Pros**:
- ✅ Native desktop window
- ✅ Encrypted API key storage (persists)
- ✅ System tray integration
- ✅ Can build installers

---

## Testing with Mock Mode (No API Key)

If you don't have an OpenAI API key or want to test without making API calls:

1. Start backend: `./test-backend.sh`
2. Open browser to `http://localhost:8501`
3. On API key setup screen, expand "🧪 Test without API key (Mock Mode)"
4. Click "Use Mock Mode"
5. App loads with simulated responses

**Mock mode is great for**:
- Testing the interface
- Training and demos
- Development without API costs

---

## Performance Notes

### First Startup

- **Time**: 10-30 seconds
- **Why**: Streamlit compiles, imports modules, loads PCA models

### Subsequent Runs

- **Time**: 5-10 seconds
- **Why**: Python modules cached

### Per Message

- **Embedding**: 2-3 seconds (AWS Bedrock call)
- **ChatGPT**: 1-5 seconds (depends on model)
- **Total**: 3-8 seconds per turn

---

## AWS Credentials (Optional)

For text-to-embedding conversion, the app uses AWS Bedrock. If you have AWS credentials:

```bash
# Set before running test-backend.sh
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"

./test-backend.sh
```

If you don't have AWS credentials:
- The app will still work if PCA models are pre-trained
- Check if models exist: `ls ../../deployment/shared/models/`

---

## Development Workflow

### Testing Changes to Python Code

1. Edit `python-backend/app.py` or `chatgpt_integration.py`
2. Stop backend (Ctrl+C)
3. Restart: `./test-backend.sh`
4. Refresh browser (Ctrl+R)

### Testing Changes to App4 Components

1. Edit files in `../../deployment/app4_unified_dashboard/`
2. Restart backend (changes auto-reload with Streamlit)
3. Refresh browser

### Viewing Logs

All Python print statements and errors show in the terminal where `test-backend.sh` is running.

---

## Next Steps After Testing

### If Everything Works ✅

1. **Document any issues** found during testing
2. **Test all features** using checklist above
3. **Try Mock Mode** to verify interface works without API
4. **Test on Windows** with full Electron app (`start.ps1`)

### If Issues Found ❌

1. **Check logs** in terminal
2. **Verify imports** work (see ARCHITECTURE.md)
3. **Check AWS credentials** if embedding errors
4. **Review error messages** - most are self-explanatory

---

## Screenshots (What to Expect)

### API Key Setup Screen
```
┌─────────────────────────────────────────┐
│  🔐 ChatGPT API Configuration           │
├─────────────────────────────────────────┤
│                                         │
│  Welcome to Vector Precognition Desktop!│
│                                         │
│  Step 1: Enter OpenAI API Key           │
│  [sk-................................] │
│                                         │
│  Step 2: Select ChatGPT Model           │
│  [GPT-3.5 Turbo (Recommended) ▼]       │
│                                         │
│  [💾 Save & Continue]                   │
└─────────────────────────────────────────┘
```

### Main Interface (4 Tabs)
```
┌─────────────────────────────────────────┐
│  🛡️ Vector Precognition Desktop         │
│  App4 Unified Dashboard + ChatGPT       │
├─────────────────────────────────────────┤
│  [💬 Live Chat] [📊 RHO] [🎯 PHI] [⚙️] │
├─────────────────────────────────────────┤
│  SIDEBAR           CHAT AREA            │
│  ┌──────────┐     ┌────────────────┐   │
│  │Config    │     │ Messages...    │   │
│  │RHO: 0.85 │     │ User: Hello    │   │
│  │          │     │ AI: Hi there!  │   │
│  └──────────┘     └────────────────┘   │
│                    [Type message...]    │
└─────────────────────────────────────────┘
```

---

## Summary

**For WSL Users**: Use `./test-backend.sh` and access via browser at `http://localhost:8501`

**For Windows Users**: Use PowerShell with `.\start.ps1` for full desktop experience

**Both modes**: Full App4 features, ChatGPT integration, all visualizations work!

---

**Quick Commands**:
```bash
# Setup (one time)
./setup.sh

# Test in browser (WSL compatible)
./test-backend.sh

# Open browser to
http://localhost:8501

# Stop server
Ctrl+C
```

---

**Status**: ✅ Tested and Working
**Mode**: Browser-only (Streamlit)
**Compatibility**: WSL, Linux, Mac
**Full Desktop**: Use Windows PowerShell
