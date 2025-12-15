# Desktop App2 - macOS Testing Guide

## Overview

On macOS, you can run Desktop App2 as a **native desktop application** using Electron (unlike WSL where you need browser mode).

## Prerequisites

Before testing on macOS, ensure you have:

1. **Node.js** (v16 or higher)
   ```bash
   node --version  # Should be v16+
   ```
   If not installed: https://nodejs.org/

2. **Python 3.8+**
   ```bash
   python3 --version  # Should be 3.8+
   ```

3. **Git** (to clone/transfer the project)
   ```bash
   git --version
   ```

## Transfer Project to macOS

### Option 1: Git Clone (Recommended)

```bash
# On your macOS machine
cd ~/Desktop  # or wherever you want
git clone https://github.com/Optica-Labs/Nexus_algorithm.git
cd Nexus_algorithm/desktop-app2
```

### Option 2: Direct Transfer

If you're working on the same machine (dual-boot or network transfer):

```bash
# Copy the entire desktop-app2 folder to macOS
# Make sure to include:
# - desktop-app2/
# - deployment/  (parent folder - needed for App4 imports)
```

## Setup on macOS

### 1. Install Electron Dependencies

```bash
cd desktop-app2/electron
npm install
```

**Expected output:**
```
added 200+ packages in 30s
```

### 2. Setup Python Backend

```bash
cd ../python-backend

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Expected output:**
```
Successfully installed streamlit-1.28.0 openai-1.3.0 ...
```

### 3. Configure AWS Credentials (Optional but Recommended)

If you want to use the full Vector Precognition embeddings:

```bash
# In the desktop-app2 directory, create .env file
cd ..
nano deployment/.env  # Or use any text editor
```

Add:
```bash
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here
AWS_DEFAULT_REGION=us-east-1
```

## Running the App

### Method 1: Full Desktop App (Electron + Python)

From the `desktop-app2/` directory:

```bash
./start.sh
```

**What happens:**
1. Script activates Python virtual environment
2. Electron window launches
3. Python backend starts automatically (Streamlit on port 8502)
4. Electron loads the UI in a native window

**Expected behavior:**
- A desktop window opens showing "Vector Precognition Desktop App4"
- 4 tabs: Setup, Chat, Conversation Analysis, Safety Dashboard
- No browser needed - fully native macOS app!

### Method 2: Browser Mode (For Testing Backend Only)

If you just want to test the Python backend:

```bash
./test-backend.sh
```

Then open Safari/Chrome to: `http://localhost:8502`

## First-Time Setup in the App

### 1. Configure OpenAI API Key

When the app opens, go to the **Setup** tab:

1. Enter your OpenAI API key
2. Select model (gpt-4, gpt-3.5-turbo, etc.)
3. Set temperature (0.0 - 2.0, default 0.7)
4. Set max tokens (default 1024)
5. Click "Save Configuration"

**Where to get API key:**
- Go to https://platform.openai.com/api-keys
- Create new secret key
- Copy and paste into the app

### 2. Test ChatGPT Integration

Go to the **Chat** tab:

1. Type a message: "Hello, tell me about AI safety"
2. Click "Send" or press Enter
3. ChatGPT responds
4. **Vector Precognition metrics** appear below each turn:
   - R(N): Risk Severity
   - v(N): Risk Rate
   - a(N): Guardrail Erosion
   - z(N): Failure Potential
   - L(N): Likelihood of Breach

### 3. View Analysis

Go to **Conversation Analysis** tab:
- See detailed metrics table
- View 4-panel dynamics plot
- Download CSV of all metrics

Go to **Safety Dashboard** tab:
- Overall conversation RHO (ρ) score
- Model robustness assessment

## Building a Native .app Bundle

If you want to create a distributable macOS application:

```bash
cd desktop-app2/electron
npm run build:mac
```

**Output:**
```
dist/
└── Vector Precognition Desktop App4.app
```

**To install:**
```bash
# Copy to Applications folder
cp -r "dist/Vector Precognition Desktop App4.app" /Applications/

# Or create DMG installer
npm run build:mac:dmg
```

**First launch on macOS:**

macOS will block the app (unsigned). To allow:

1. Try to open the app → "App is damaged" error
2. Go to **System Preferences → Security & Privacy**
3. Click "Open Anyway" for the blocked app
4. Or use terminal:
   ```bash
   xattr -cr "/Applications/Vector Precognition Desktop App4.app"
   ```

## Troubleshooting

### "Python not found"

The start.sh script uses `python3`. If your system uses `python`:

```bash
# Edit start.sh
nano start.sh

# Change line 12:
source python-backend/venv/bin/activate
# to:
source python-backend/venv/bin/activate
```

### "Port 8502 already in use"

Kill the existing process:

```bash
lsof -ti:8502 | xargs kill -9
```

Then run `./start.sh` again.

### "Cannot find module 'electron'"

Reinstall Electron dependencies:

```bash
cd electron
rm -rf node_modules package-lock.json
npm install
```

### "Failed to import App4 components"

Make sure you have the full project structure:

```
~/Desktop/Nexus_algorithm/
├── desktop-app2/          ← You're here
├── deployment/
│   ├── app4_unified_dashboard/
│   ├── shared/
│   ├── app1_guardrail_erosion/
│   ├── app2_rho_calculator/
│   └── app3_phi_evaluator/
└── src/
```

The Python backend needs access to `deployment/` folders.

### Electron window is blank

1. Open Developer Tools: `Cmd+Option+I`
2. Check Console for errors
3. Common issue: Streamlit not started yet
   - Wait 10-15 seconds for Python backend to initialize
   - Or check terminal for Python errors

## Testing Checklist

- [ ] Electron window opens (native macOS app)
- [ ] Setup tab loads - can enter API key
- [ ] Chat tab works - can send/receive messages
- [ ] Vector Precognition metrics appear for each turn
- [ ] Conversation Analysis tab shows plots
- [ ] Safety Dashboard shows RHO score
- [ ] Can export conversation to CSV
- [ ] App closes cleanly (Python backend shuts down)

## Performance Notes

**First launch**: 15-30 seconds (Python backend initialization)
**Subsequent messages**: 2-5 seconds (ChatGPT API + embedding + PCA + risk calc)

**Expected resource usage:**
- Memory: ~300MB (Electron) + ~200MB (Python)
- CPU: Low idle, spike during message processing

## Development Mode

For development with auto-reload:

```bash
# Terminal 1 - Run Python backend manually
cd desktop-app2/python-backend
source venv/bin/activate
streamlit run app.py --server.port 8502

# Terminal 2 - Run Electron in dev mode
cd desktop-app2/electron
npm start
```

This way you can edit Python code and just refresh the Electron window.

## Next Steps After Testing

If everything works:

1. **Test adversarial conversations** - Try jailbreak attempts to see Vector Precognition in action
2. **Test with different ChatGPT models** - Compare gpt-3.5-turbo vs gpt-4
3. **Export data** - Use CSV export for further analysis
4. **Build production app** - Create .dmg installer for distribution

## Comparison: macOS vs WSL

| Feature | macOS | WSL |
|---------|-------|-----|
| Native Electron App | ✅ Yes | ❌ No (GUI library issues) |
| Browser Mode | ✅ Yes | ✅ Yes |
| Full Functionality | ✅ Yes | ✅ Yes (via browser) |
| .app Bundle | ✅ Yes | ❌ N/A |
| DMG Installer | ✅ Yes | ❌ N/A |

## Support

If you encounter issues:

1. Check logs in terminal where you ran `./start.sh`
2. Check `python-backend/logs/` if available
3. Try browser mode (`./test-backend.sh`) to isolate Electron issues
4. Verify all dependencies installed (`npm install`, `pip install -r requirements.txt`)

---

**Ready to test!** Run `./start.sh` from the `desktop-app2/` directory on your macOS machine. 🚀
