# Desktop App2 - macOS Application Guide

**Complete guide to running Desktop App2 as a native macOS application**

---

## 🍎 Overview

Desktop App2 can run as a **native macOS application** with:
- ✅ Native macOS window (not browser)
- ✅ App icon in Dock
- ✅ Menu bar integration
- ✅ Encrypted API key storage
- ✅ Launch from Applications folder
- ✅ Distributable DMG installer

---

## 📋 Prerequisites

### Required Software:

1. **macOS** 10.15 (Catalina) or later
2. **Python 3.8+**
   ```bash
   python3 --version
   ```
   Install from: https://www.python.org/downloads/macos/

3. **Node.js 16+**
   ```bash
   node --version
   ```
   Install from: https://nodejs.org/

4. **Homebrew** (optional, for easy installs)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

---

## 🚀 Method 1: Run in Development Mode (Fastest)

This runs the app locally without building an installer.

### Step 1: Clone/Download the Project

```bash
# If on macOS, clone from WSL or download the desktop-app2 folder
# Copy to your Mac (use USB, cloud storage, or git)

# On macOS:
cd ~/Desktop
# (Paste or clone the desktop-app2 folder here)
```

### Step 2: Run Setup

```bash
cd ~/Desktop/desktop-app2  # Or wherever you put it
./setup.sh
```

**What this does**:
- Creates Python virtual environment
- Installs Python packages (streamlit, openai, etc.)
- Installs Node packages (electron, electron-store)
- Takes ~3-5 minutes

### Step 3: Start the Application

```bash
./start.sh
```

**What happens**:
- Python backend starts (Streamlit)
- Electron window opens automatically
- App appears in your Dock
- Full native macOS experience!

### Step 4: Configure & Use

1. **API Key Setup Screen** appears
2. Enter your OpenAI API key (or use Mock Mode)
3. Select GPT model
4. Click "Save & Continue"
5. **Full 4-tab interface loads!**

---

## 🏗️ Method 2: Build macOS Application (DMG Installer)

This creates a `.dmg` installer you can distribute.

### Step 1: Setup (if not done)

```bash
cd desktop-app2
./setup.sh
```

### Step 2: Build the Application

```bash
cd electron
npm run build:mac
```

**This creates**:
```
electron/dist/
├── Vector Precognition App4.app     # The application
└── Vector Precognition App4.dmg     # Installer
```

**Build time**: ~2-5 minutes

### Step 3: Install the Application

#### Option A: From .app file
```bash
# Copy to Applications folder
cp -r "electron/dist/Vector Precognition App4.app" /Applications/

# Or drag & drop in Finder:
# 1. Open electron/dist/ in Finder
# 2. Drag "Vector Precognition App4.app" to Applications
```

#### Option B: From .dmg installer
```bash
# Open the DMG
open "electron/dist/Vector Precognition App4.dmg"

# In the window that opens:
# 1. Drag app icon to Applications folder
# 2. Eject the DMG
```

### Step 4: Launch

```bash
# From terminal:
open /Applications/"Vector Precognition App4.app"

# OR: Click app icon in Launchpad
# OR: Spotlight search (Cmd+Space): "Vector Precognition"
```

---

## 🔐 macOS Security & Permissions

### First Launch: "App can't be opened" Warning

If you see: **"Vector Precognition App4.app can't be opened because it is from an unidentified developer"**

**Solution 1: Right-click bypass**
1. Right-click (or Ctrl+click) the app in Applications
2. Select "Open"
3. Click "Open" in the dialog
4. App will open and remember this choice

**Solution 2: System Preferences**
1. Go to System Preferences → Security & Privacy
2. Click "Open Anyway" next to the blocked app message
3. App will open

**Solution 3: Command line**
```bash
xattr -cr /Applications/"Vector Precognition App4.app"
open /Applications/"Vector Precognition App4.app"
```

### For Distribution: Code Signing (Optional)

To avoid security warnings for other users:

1. **Get Apple Developer Account** ($99/year)
   - https://developer.apple.com/

2. **Get Developer Certificate**
   ```bash
   # In Keychain Access, request certificate
   # Download from Apple Developer portal
   ```

3. **Sign the app**
   ```bash
   # Update electron/package.json
   {
     "build": {
       "mac": {
         "identity": "Developer ID Application: Your Name (TEAMID)",
         "hardenedRuntime": true,
         "gatekeeperAssess": false,
         "entitlements": "build/entitlements.mac.plist",
         "entitlementsInherit": "build/entitlements.mac.plist"
       }
     }
   }
   ```

4. **Notarize with Apple**
   ```bash
   xcrun altool --notarize-app \
     --primary-bundle-id "com.opticalabs.vectorprecognition.app4" \
     --username "your@email.com" \
     --password "@keychain:AC_PASSWORD" \
     --file "electron/dist/Vector Precognition App4.dmg"
   ```

---

## ⚙️ Configuration Files

### Icon File

The app needs an icon file for macOS:

**Required**: `desktop-app2/resources/icon.icns`

**To create from PNG**:
```bash
# If you have a PNG (icon.png), convert it:
mkdir icon.iconset
sips -z 16 16     icon.png --out icon.iconset/icon_16x16.png
sips -z 32 32     icon.png --out icon.iconset/icon_16x16@2x.png
sips -z 32 32     icon.png --out icon.iconset/icon_32x32.png
sips -z 64 64     icon.png --out icon.iconset/icon_32x32@2x.png
sips -z 128 128   icon.png --out icon.iconset/icon_128x128.png
sips -z 256 256   icon.png --out icon.iconset/icon_128x128@2x.png
sips -z 256 256   icon.png --out icon.iconset/icon_256x256.png
sips -z 512 512   icon.png --out icon.iconset/icon_256x256@2x.png
sips -z 512 512   icon.png --out icon.iconset/icon_512x512.png
sips -z 1024 1024 icon.png --out icon.iconset/icon_512x512@2x.png

# Convert to .icns
iconutil -c icns icon.iconset

# Move to resources
mv icon.icns desktop-app2/resources/
```

### Build Configuration

**File**: `electron/package.json`

```json
{
  "build": {
    "appId": "com.opticalabs.vectorprecognition.app4",
    "productName": "Vector Precognition App4",
    "mac": {
      "target": "dmg",
      "icon": "../resources/icon.icns",
      "category": "public.app-category.productivity"
    }
  }
}
```

---

## 📂 Application Structure (macOS)

### Development Mode:
```
desktop-app2/
├── electron/
│   ├── main.js              # Electron main process
│   ├── preload.js           # Security bridge
│   └── node_modules/        # Node dependencies
├── python-backend/
│   ├── app.py               # Streamlit app
│   ├── venv/                # Python environment
│   └── ...
└── start.sh                 # Launch script
```

### Built Application:
```
Vector Precognition App4.app/
└── Contents/
    ├── Info.plist           # App metadata
    ├── MacOS/
    │   └── Vector Precognition App4  # Electron binary
    ├── Resources/
    │   ├── icon.icns        # App icon
    │   ├── python-backend/  # Bundled Python code
    │   └── electron.asar    # Electron code
    └── Frameworks/          # Electron framework
```

---

## 🎯 Running Options Compared

| Feature | Dev Mode (./start.sh) | Built App (.app) | DMG Installer |
|---------|----------------------|------------------|---------------|
| **Setup Time** | 5 min | 10 min | 10 min |
| **Launch Speed** | 5-10s | 2-5s | 2-5s |
| **Native Look** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Icon in Dock** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Easy to Share** | ❌ No | ⚠️ Folder | ✅ DMG file |
| **Updates** | Manual | Manual | Manual |
| **File Size** | ~200 MB | ~250 MB | ~150 MB |
| **Requires Terminal** | ✅ First time | ❌ No | ❌ No |

**Recommendation**:
- **Testing**: Use Dev Mode (`./start.sh`)
- **Personal Use**: Use Built App (`.app`)
- **Distribution**: Use DMG Installer (`.dmg`)

---

## 🚨 Troubleshooting macOS

### Issue: "Python not found"

**Solution**: Install Python 3.8+
```bash
# Using Homebrew:
brew install python@3.11

# Or download from:
# https://www.python.org/downloads/macos/
```

### Issue: "Node not found"

**Solution**: Install Node.js
```bash
# Using Homebrew:
brew install node

# Or download from:
# https://nodejs.org/
```

### Issue: "Electron window doesn't open"

**Check**:
```bash
# Make sure Streamlit started:
curl http://localhost:8501
# Should return HTML

# Check for errors in terminal
# Look for Python errors or port conflicts
```

### Issue: "Port 8501 already in use"

**Solution**:
```bash
# Kill existing process:
lsof -ti:8501 | xargs kill -9

# Then restart:
./start.sh
```

### Issue: App crashes on startup

**Check logs**:
```bash
# View Console app logs:
open /Applications/Utilities/Console.app

# Or check terminal output when running:
./start.sh
# Look for Python tracebacks or errors
```

### Issue: "Can't install packages" (Permission denied)

**Solution**:
```bash
# Don't use sudo with setup script!
# Just run:
./setup.sh

# If still fails, check Python permissions:
ls -la $(which python3)
```

---

## 🔄 Updates & Maintenance

### Updating the Application

#### Dev Mode:
```bash
cd desktop-app2
git pull  # If using git
./setup.sh  # Reinstall dependencies if needed
./start.sh  # Restart
```

#### Built App:
1. Rebuild the application:
   ```bash
   cd electron
   npm run build:mac
   ```
2. Replace old app with new one
3. Or create new DMG and reinstall

---

## 📦 Distribution Checklist

If you want to share the app with others:

- [ ] Build DMG: `npm run build:mac`
- [ ] Test DMG on clean Mac (no dev tools installed)
- [ ] Create README for users
- [ ] Document API key setup
- [ ] (Optional) Code sign & notarize
- [ ] Upload DMG to cloud storage or website
- [ ] Provide download link

### Sample Distribution README:

```markdown
# Vector Precognition App4 for macOS

## Installation

1. Download `Vector Precognition App4.dmg`
2. Double-click to open
3. Drag app to Applications folder
4. Launch from Applications or Launchpad

## First Launch

1. Right-click app → Open (to bypass security)
2. Enter your OpenAI API key
3. Start chatting!

## Requirements

- macOS 10.15 or later
- OpenAI API key (get from platform.openai.com)
```

---

## 🎨 Customization

### Change App Name

**File**: `electron/package.json`
```json
{
  "name": "your-app-name",
  "productName": "Your App Display Name",
  "build": {
    "appId": "com.yourcompany.yourapp"
  }
}
```

### Change App Icon

1. Create/edit `resources/icon.icns`
2. Rebuild: `npm run build:mac`

### Change Window Size

**File**: `electron/main.js`
```javascript
mainWindow = new BrowserWindow({
  width: 1600,   // Change this
  height: 1000,  // Change this
  minWidth: 1200,
  minHeight: 800,
  // ...
});
```

---

## 📊 Performance on macOS

### Recommended Specs:

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **macOS** | 10.15 Catalina | 12.0 Monterey+ |
| **CPU** | Intel i5 / M1 | Intel i7 / M1 Pro+ |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 2 GB free | 5 GB free |
| **Internet** | Required | Broadband |

### Typical Performance:

- **Startup**: 5-10 seconds
- **Message Response**: 3-8 seconds (with real API)
- **Memory Usage**: 200-400 MB
- **CPU Usage**: 5-15% (idle), 30-50% (active)

---

## 🔐 macOS-Specific Features

### 1. Encrypted Storage

API keys stored in:
```
~/Library/Application Support/vector-precognition-app4/config.json
```
Encrypted with `electron-store`

### 2. Dock Integration

- App icon appears in Dock when running
- Right-click for quit/hide options
- Badge notifications (can be added)

### 3. Menu Bar

Standard macOS menu bar with:
- File → Quit
- Edit → Copy/Paste
- View → Reload
- Window → Minimize/Zoom

### 4. Spotlight Integration

After installation:
- Cmd+Space → Type "Vector Precognition"
- App appears in search results

---

## 🎯 Quick Start Commands

### Development Mode:
```bash
# First time:
cd desktop-app2
./setup.sh
./start.sh

# Subsequent runs:
./start.sh
```

### Build Mode:
```bash
# Build once:
cd electron
npm run build:mac

# Install:
open "dist/Vector Precognition App4.dmg"
# Drag to Applications

# Launch:
open /Applications/"Vector Precognition App4.app"
```

---

## 📞 Support & Resources

### Documentation:
- Main README: `README.md`
- Testing Guide: `README_TESTING.md`
- Final Instructions: `FINAL_INSTRUCTIONS.md`
- WSL Testing: `WSL_TESTING_GUIDE.md`

### Common Issues:
- Check `FINAL_INSTRUCTIONS.md` for troubleshooting
- macOS-specific: This file
- General: `WSL_TESTING_GUIDE.md` (applies to macOS too)

### Electron Resources:
- Electron Docs: https://www.electronjs.org/docs
- Electron Builder: https://www.electron.build/
- macOS Signing: https://www.electron.build/code-signing

---

## ✅ Success Checklist

### Development Mode:
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Ran `./setup.sh` successfully
- [ ] Ran `./start.sh`
- [ ] Electron window opened
- [ ] API key setup screen appeared
- [ ] Can chat with ChatGPT
- [ ] All 4 tabs work

### Built Application:
- [ ] Built DMG successfully
- [ ] Installed to Applications folder
- [ ] Launched without security warnings (after bypass)
- [ ] App appears in Launchpad
- [ ] API key persists across restarts
- [ ] All features work

---

## 🎉 Summary

**To run on macOS**:

1. **Quick Test** (5 min):
   ```bash
   ./setup.sh
   ./start.sh
   ```

2. **Full Native App** (10 min):
   ```bash
   cd electron
   npm run build:mac
   open dist/*.dmg
   # Install & Launch
   ```

**That's it!** You now have a native macOS application with full App4 + ChatGPT integration! 🍎✨

---

**Version**: 2.0.0
**Last Updated**: December 11, 2024
**Platform**: macOS 10.15+
**Status**: ✅ Ready for macOS!
