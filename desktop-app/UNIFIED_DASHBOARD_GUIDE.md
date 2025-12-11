# Unified Dashboard Desktop App

This guide shows how to run the **Unified AI Safety Dashboard** (`deployment/app4_unified_dashboard`) as an **Electron desktop application**.

---

## Overview

The desktop app uses **Electron** to wrap a **Streamlit** backend, providing:
- Native desktop window for the unified dashboard
- Secure credential storage (API keys, AWS credentials)
- Cross-platform support (Windows, macOS, Linux)
- Standalone packaging for distribution

The Electron wrapper can load either:
1. **Original backend** (`desktop-app/python-backend`) - legacy ChatGPT integration
2. **Unified dashboard** (`deployment/app4_unified_dashboard`) - **new default**

---

## Quick Start

### 1. Install Node.js Dependencies

```bash
cd desktop-app/electron
npm install
```

### 2. Install Python Dependencies

```bash
# From repository root
pip install -r deployment/app4_unified_dashboard/requirements.txt

# Optional: install shared dependencies
pip install -r deployment/shared/requirements.txt
```

### 3. Launch Desktop App with Unified Dashboard

**Linux/macOS:**
```bash
cd desktop-app
./start_unified.sh
```

**Windows (PowerShell):**
```powershell
cd desktop-app
.\start_unified.ps1
```

**Manual (from electron directory):**
```bash
cd desktop-app/electron
npm run start:app4
```

---

## How It Works

### Backend Selection

The Electron app (`desktop-app/electron/main.js`) supports switching backends:

- **Environment variable:**
  ```bash
  export APP_BACKEND=app4_unified_dashboard
  npm start
  ```

- **CLI argument:**
  ```bash
  electron . --app=app4_unified_dashboard
  ```

- **npm script** (recommended):
  ```bash
  npm run start:app4
  ```

### Startup Flow

1. Electron launches and reads backend choice
2. Python backend (Streamlit) starts on `localhost:8501`
3. Electron waits for Streamlit to be ready
4. Desktop window opens and loads the Streamlit UI

---

## Package Scripts

Defined in `desktop-app/electron/package.json`:

| Script | Command | Description |
|--------|---------|-------------|
| `npm start` | `electron .` | Run with default backend (python-backend) |
| `npm run start:app4` | `APP_BACKEND=app4_unified_dashboard electron .` | Run with unified dashboard |
| `npm run dev` | `electron . --dev` | Development mode (opens DevTools) |
| `npm run build` | `electron-builder` | Build installer for current platform |
| `npm run build:win` | `electron-builder --win` | Build Windows installer |
| `npm run build:mac` | `electron-builder --mac` | Build macOS installer |
| `npm run build:linux` | `electron-builder --linux` | Build Linux installer |

---

## Building Installers

### Prerequisites

Install `electron-builder` (already in `devDependencies`):
```bash
npm install
```

### Build for Current Platform

```bash
npm run build
```

Output: `desktop-app/electron/dist/`

### Build for Specific Platform

```bash
# Windows (NSIS installer)
npm run build:win

# macOS (DMG)
npm run build:mac

# Linux (AppImage + deb)
npm run build:linux
```

### What Gets Packaged?

The `package.json` `build.extraResources` section bundles:
- `python-backend/` → legacy backend
- `deployment/app4_unified_dashboard/` → unified dashboard
- `deployment/shared/` → shared Python modules
- `deployment/models/` → PCA models

**Note:** You must ensure all Python dependencies and models are present before building.

---

## Configuration

### API Keys & Credentials

The Electron app provides secure storage for:
- OpenAI API keys
- AWS credentials (access key, secret key, region)

These are stored using `electron-store` with encryption.

**To configure:**
1. Launch the app
2. Use the Streamlit UI to input credentials
3. Credentials persist across app restarts

### Streamlit Port

Default: `8501`

To change, edit `desktop-app/electron/main.js`:
```javascript
const STREAMLIT_PORT = 8502; // Change port
```

---

## Development Mode

For development with live reloading:

```bash
cd desktop-app/electron
npm run dev -- --app=app4_unified_dashboard
```

This:
- Opens DevTools automatically
- Loads backend from source directories (not packaged resources)
- Enables hot-reloading for Python changes (restart Streamlit)

---

## Troubleshooting

### "Python backend failed to start"

**Check:**
1. Python is installed and in PATH
2. Streamlit is installed: `pip install streamlit`
3. All dependencies are installed from `requirements.txt`
4. Port 8501 is not already in use

**Test manually:**
```bash
cd deployment/app4_unified_dashboard
streamlit run app.py
```

### "Cannot find module 'electron'"

**Fix:**
```bash
cd desktop-app/electron
npm install
```

### "ModuleNotFoundError" in Python

**Fix:**
```bash
pip install -r deployment/app4_unified_dashboard/requirements.txt
pip install -r deployment/shared/requirements.txt
```

### Window opens but shows "Cannot connect"

**Wait:** Streamlit takes 10-30 seconds to start. The app retries automatically.

**Check logs:** Look for Python backend output in the terminal.

---

## Architecture

```
desktop-app/
├── electron/                    # Electron wrapper
│   ├── main.js                 # Main process (backend launcher)
│   ├── preload.js              # IPC bridge
│   ├── package.json            # Electron config
│   └── node_modules/           # Node dependencies
│
├── python-backend/             # Legacy backend (optional)
│
└── start_unified.sh/ps1        # Quick launchers

deployment/
├── app4_unified_dashboard/     # Unified dashboard backend
│   ├── app.py                  # Streamlit entry point
│   ├── requirements.txt        # Python dependencies
│   └── ...                     # Dashboard modules
│
└── shared/                     # Shared Python modules
    └── ...                     # PCA, API clients, etc.
```

---

## Next Steps

### Recommended Improvements

1. **Auto-updates:** Integrate `electron-updater` for automatic updates
2. **Custom installer:** Add custom branding, license agreements
3. **Code signing:** Sign installers for Windows/macOS
4. **Portable mode:** Add option to run without installation
5. **Multiple backends:** Add UI to switch backends without restarting
6. **Health checks:** Better backend startup detection and error handling

### Production Checklist

- [ ] Test on all target platforms (Windows, macOS, Linux)
- [ ] Bundle all required Python dependencies
- [ ] Include PCA models in `deployment/models/`
- [ ] Configure code signing certificates
- [ ] Set up update server for auto-updates
- [ ] Write user documentation
- [ ] Test installer/uninstaller flows

---

## Additional Resources

- **Electron Documentation:** https://www.electronjs.org/docs
- **electron-builder:** https://www.electron.build/
- **Streamlit Docs:** https://docs.streamlit.io/

---

## Support

For issues or questions:
1. Check existing documentation in `desktop-app/docs/`
2. Review `TESTING_GUIDE.md` for testing procedures
3. Check `CURRENT_STATUS.md` for known issues

---

**Last Updated:** December 11, 2025
