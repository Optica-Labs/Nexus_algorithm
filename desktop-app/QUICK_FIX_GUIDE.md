# Quick Fix: Enable Full App4 Features in Desktop App

**Time Required**: 5 minutes
**Difficulty**: Easy
**Result**: Full 4-tab dashboard with all features!

---

## The Discovery 🎉

The desktop app **already supports** running the full App4 Unified Dashboard!
It's just not the default. Here's how to enable it:

---

## Option 1: Run App4 Right Now (No Changes)

### Method A: Using npm script
```bash
cd desktop-app/electron
npm run start:app4
```

### Method B: Using environment variable
```bash
cd desktop-app/electron
APP_BACKEND=app4_unified_dashboard npm start
```

### Method C: Using command-line argument
```bash
cd desktop-app/electron
electron . --app=app4_unified_dashboard
```

**All three methods do the same thing!** Try it now to verify it works.

---

## Option 2: Make App4 the Permanent Default (Recommended)

This makes the full dashboard the default when you run `npm start`.

### Step 1: Edit main.js
Open: `desktop-app/electron/main.js`

Find line 28:
```javascript
const appChoice = explicitAppArg ? explicitAppArg.split('=')[1] : (envApp || 'python-backend');
```

Change to:
```javascript
const appChoice = explicitAppArg ? explicitAppArg.split('=')[1] : (envApp || 'app4_unified_dashboard');
```

**That's it!** Just change `'python-backend'` to `'app4_unified_dashboard'`.

### Step 2: Update package.json (Optional)
Open: `desktop-app/electron/package.json`

Update the scripts section:
```json
{
  "scripts": {
    "start": "APP_BACKEND=app4_unified_dashboard electron .",
    "start:simple": "electron .",
    "start:app4": "APP_BACKEND=app4_unified_dashboard electron .",
    "dev": "electron . --dev",
    "build": "electron-builder",
    "build:win": "electron-builder --win",
    "build:mac": "electron-builder --mac",
    "build:linux": "electron-builder --linux"
  }
}
```

Now:
- `npm start` → Runs full App4 dashboard
- `npm run start:simple` → Runs simple ChatGPT-only version

---

## What You Get with App4

### Before (python-backend)
- ❌ Single page only
- ❌ ChatGPT only
- ❌ No historical analysis
- ❌ No PHI scores

### After (app4_unified_dashboard)
- ✅ **4-Tab Interface**:
  - Tab 1: Live Chat with real-time monitoring
  - Tab 2: RHO Analysis (per-conversation)
  - Tab 3: PHI Benchmark (multi-conversation)
  - Tab 4: Settings & Configuration
- ✅ **Multiple Models**:
  - GPT-3.5 Turbo
  - GPT-4
  - Claude Sonnet 3
  - Mistral Large
- ✅ **Historical Analysis**:
  - Review past conversations
  - Compare robustness across sessions
  - Export comprehensive reports
- ✅ **PHI Fragility Scoring**:
  - Aggregate metrics across conversations
  - Model-level benchmarking
  - Pass/Fail classification
- ✅ **Mock Mode**:
  - Test without API keys
  - Demo mode for presentations
- ✅ **Advanced Visualizations**:
  - 4-panel dynamics plots
  - Cumulative risk charts
  - RHO timeline graphs
  - PHI distribution histograms

---

## Testing the App4 Integration

After making the change, test these features:

### Test 1: Start the App
```bash
cd desktop-app/electron
npm start  # Should now run app4
```

**Expected**: You should see 4 tabs at the top:
- 💬 Live Chat
- 📊 RHO Analysis
- 🎯 PHI Benchmark
- ⚙️ Settings

### Test 2: Multi-Model Support
1. Click "Live Chat" tab
2. Open sidebar
3. Find "Model Selection" dropdown
4. **Expected**: Should see:
   - GPT-3.5 Turbo
   - GPT-4
   - Claude Sonnet 3
   - Mistral Large
   - Mock LLM (for testing)

### Test 3: Conversation History
1. Start a conversation in Tab 1
2. Have a few turns of chat
3. Click "End Conversation" button
4. Go to Tab 2 (RHO Analysis)
5. **Expected**: Should see your conversation in the dropdown
6. Select it and view detailed metrics

### Test 4: PHI Benchmark
1. Complete 2-3 conversations in Tab 1
2. Go to Tab 3 (PHI Benchmark)
3. **Expected**: Should see:
   - PHI score calculated
   - Table of all conversations
   - Distribution histogram
   - Pass/Fail classification

### Test 5: Settings Tab
1. Go to Tab 4 (Settings)
2. **Expected**: Should see:
   - Session information
   - System prompt editor
   - Algorithm parameters
   - Export options

---

## Troubleshooting

### Problem: "Module not found" errors

**Solution**: App4 needs access to shared modules. Verify paths:
```bash
ls -la ../deployment/app4_unified_dashboard/
# Should show: app.py, core/, ui/, utils/

ls -la ../deployment/shared/
# Should show: pca_pipeline.py, vector_processor.py, etc.
```

If missing, the paths in main.js might be wrong. Check:
```javascript
// Line 32 in main.js should resolve to correct path
const backendPath = (appChoice === 'app4_unified_dashboard')
    ? (isDev
        ? path.join(__dirname, '..', '..', 'deployment', 'app4_unified_dashboard')
        : path.join(process.resourcesPath, 'app4_unified_dashboard'))
```

### Problem: Streamlit won't start

**Check**:
1. Python virtual environment activated?
2. Requirements installed?
   ```bash
   cd ../../deployment/app4_unified_dashboard
   pip install -r requirements.txt
   ```

### Problem: API keys not working

**Note**: App4 uses **AWS Lambda endpoints** by default, which don't need API keys for GPT-3.5, GPT-4, Claude, Mistral. Only if you want to use **direct API mode** do you need keys.

To use Mock mode (no API needed):
1. Sidebar → Model Selection
2. Select "Mock LLM"
3. Chat away!

---

## Reverting Back to Simple Mode

If you want to go back to the ChatGPT-only version:

### Quick revert:
```bash
npm run start:simple
```

### Permanent revert:
Edit `main.js` line 28 back to:
```javascript
const appChoice = explicitAppArg ? explicitAppArg.split('=')[1] : (envApp || 'python-backend');
```

---

## Building Desktop Installers with App4

Once you've verified app4 works, you can build installers:

### Windows Installer
```bash
npm run build:win
# Creates: desktop-app/electron/dist/Vector Precognition Setup.exe
```

### macOS DMG
```bash
npm run build:mac
# Creates: desktop-app/electron/dist/Vector Precognition.dmg
```

### Linux AppImage
```bash
npm run build:linux
# Creates: desktop-app/electron/dist/Vector Precognition.AppImage
```

**Important**: The `package.json` already includes app4 in the build:
```json
{
  "build": {
    "extraResources": [
      {
        "from": "../../deployment/app4_unified_dashboard",
        "to": "app4_unified_dashboard",
        "filter": ["**/*"]
      }
    ]
  }
}
```

So the full app4 code **will be included** in your installers!

---

## Summary

### What We Discovered
The desktop app **already has** all the code to run the full App4 dashboard. It just defaults to a simpler version.

### What You Need to Do
**Option A**: Nothing! Just run `npm run start:app4`
**Option B**: Change 1 line in main.js to make it permanent

### What You Get
- 4x more features
- Multi-model support
- Historical analysis
- Research-grade benchmarking
- Professional visualizations

### Time Investment
- Testing: 5 minutes
- Making permanent: 2 minutes
- Total: 7 minutes for 4x functionality! 🚀

---

## Next Steps

1. **Test it now**:
   ```bash
   cd desktop-app/electron
   npm run start:app4
   ```

2. **If it works**, make it permanent (edit main.js line 28)

3. **Update documentation** to reflect full feature set

4. **Consider removing** `python-backend/` to reduce confusion (or keep as "simple mode")

5. **Build installers** and distribute!

---

**Quick Reference**:
- **File to edit**: `desktop-app/electron/main.js` (line 28)
- **Change from**: `'python-backend'`
- **Change to**: `'app4_unified_dashboard'`
- **Result**: Full dashboard with all features! 🎉
