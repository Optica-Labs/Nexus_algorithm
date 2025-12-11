# Multi-Platform Guide - Vector Precognition Desktop

Guide for running on **WSL**, **macOS**, and **Windows**.

---

## 🖥️ **Platform Support**

| Platform | Desktop App | Backend-Only | Recommended |
|----------|-------------|--------------|-------------|
| **macOS** | ✅ Full support | ✅ Works | Desktop App |
| **Windows** | ✅ Full support | ✅ Works | Desktop App |
| **WSL** | ⚠️ Needs setup | ✅ Works | Backend-Only |
| **Linux** | ✅ Full support | ✅ Works | Desktop App |

---

## 🍎 **macOS Setup**

### Prerequisites
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3
brew install python@3.11

# Install Node.js
brew install node
```

### Installation
```bash
cd desktop-app

# Run setup
chmod +x setup.sh
./setup.sh

# Launch desktop app
./start.sh
```

### Troubleshooting macOS

#### "App cannot be opened because it is from an unidentified developer"
```bash
# Allow Electron to run
xattr -cr electron/node_modules/electron/dist/Electron.app

# Or in System Preferences:
# Security & Privacy → Allow apps downloaded from: App Store and identified developers
```

#### "Python not found"
```bash
# Use python3 explicitly
python3 -m venv venv
source venv/bin/activate
```

#### "Cannot find module 'electron'"
```bash
cd electron
npm install
```

---

## 🪟 **Windows Setup**

### Option A: Native Windows (PowerShell)

#### Prerequisites
- Install Python: [python.org/downloads](https://www.python.org/downloads/)
- Install Node.js: [nodejs.org](https://nodejs.org/)

#### Installation
```powershell
cd desktop-app

# Run setup
.\setup.ps1

# Launch app
.\start.ps1
```

### Option B: WSL (Backend-Only)

See WSL section below.

---

## 🐧 **WSL Setup**

### Option 1: Backend-Only (Recommended)

**Fastest and most reliable for WSL:**

```bash
cd ~/work/optica_labs/algorithm_work/desktop-app

# Test backend only
./test_backend.sh
```

Open browser: `http://localhost:8501`

**Features available:**
- ✅ Full ChatGPT integration
- ✅ Real-time risk analysis
- ✅ Live metrics & charts
- ✅ Safety alerts
- ✅ Conversation export
- ❌ Desktop window (browser instead)

---

### Option 2: Full Desktop with X Server

#### Step 1: Install X Server on Windows

**Download VcXsrv**: [sourceforge.net/projects/vcxsrv](https://sourceforge.net/projects/vcxsrv/)

**Configure XLaunch:**
1. Multiple windows
2. Start no client
3. **Disable access control** ✅
4. Save configuration

**Start X Server** before launching app!

#### Step 2: Configure WSL

Add to `~/.bashrc`:

```bash
# X11 Display
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
export LIBGL_ALWAYS_INDIRECT=1

# Disable SSL verification issues (if needed)
export NO_AT_BRIDGE=1
```

Apply changes:
```bash
source ~/.bashrc
```

#### Step 3: Install Required Libraries

```bash
sudo apt update
sudo apt install -y \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
  libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
  libxfixes3 libxrandr2 libgbm1 libasound2 \
  libatspi2.0-0 libxshmfence1 libgtk-3-0 libgconf-2-4
```

#### Step 4: Test X Server

```bash
# Test if X server is reachable
xeyes
# Should show a pair of eyes following your cursor
```

If that works, you're ready!

#### Step 5: Launch Desktop App

```bash
cd ~/work/optica_labs/algorithm_work/desktop-app
./start.sh
```

---

## 🐧 **Linux (Ubuntu/Debian)**

### Prerequisites

```bash
sudo apt update
sudo apt install -y \
  python3 python3-pip python3-venv \
  nodejs npm \
  libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 \
  libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 \
  libxfixes3 libxrandr2 libgbm1 libasound2
```

### Installation

```bash
cd desktop-app

# Run setup
chmod +x setup.sh start.sh
./setup.sh

# Launch app
./start.sh
```

---

## 🔧 **Platform-Specific Configuration**

### AWS Credentials

#### macOS/Linux
```bash
# Add to ~/.bashrc or ~/.zshrc
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1

# Or use AWS CLI
brew install awscli  # macOS
sudo apt install awscli  # Linux
aws configure
```

#### Windows (PowerShell)
```powershell
$env:AWS_ACCESS_KEY_ID="your_key"
$env:AWS_SECRET_ACCESS_KEY="your_secret"
$env:AWS_DEFAULT_REGION="us-east-1"

# Or use AWS CLI
choco install awscli
aws configure
```

### Python Path Issues

#### macOS
```bash
# If "python not found"
alias python=python3
# Add to ~/.zshrc for persistence
```

#### WSL
```bash
# Usually python3 is available by default
python3 --version
```

---

## 🚀 **Quick Start by Platform**

### macOS
```bash
brew install python@3.11 node
cd desktop-app
./setup.sh
./start.sh
```

### Windows (PowerShell)
```powershell
cd desktop-app
.\setup.ps1
.\start.ps1
```

### WSL (Backend-Only)
```bash
cd desktop-app
./test_backend.sh
# Open http://localhost:8501
```

### Linux
```bash
sudo apt install python3 nodejs npm libnss3
cd desktop-app
./setup.sh
./start.sh
```

---

## 🐛 **Common Issues by Platform**

### macOS

#### "Operation not permitted"
```bash
# macOS Catalina+: Grant Terminal full disk access
# System Preferences → Security & Privacy → Privacy → Full Disk Access
# Add Terminal.app
```

#### "xcrun: error"
```bash
# Install Xcode Command Line Tools
xcode-select --install
```

#### "Library not loaded: @rpath/..."
```bash
# Reinstall Node modules
cd electron
rm -rf node_modules
npm install
```

---

### WSL

#### "Cannot open display"
```bash
# Ensure X Server is running on Windows
# Check DISPLAY variable
echo $DISPLAY
# Should show: <IP>:0

# Set it manually if needed
export DISPLAY=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}'):0
```

#### "Connection refused to X server"
```bash
# In VcXsrv, disable access control:
# XLaunch → Extra settings → "Disable access control" ✅
```

---

### Windows

#### "Cannot be loaded because running scripts is disabled"
```powershell
# Enable script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### "Python was not found"
```powershell
# Reinstall Python and check "Add to PATH"
# Or find path:
where.exe python
# Add to PATH manually
```

---

## 📊 **Performance Comparison**

| Platform | Startup Time | Response Time | Notes |
|----------|--------------|---------------|-------|
| macOS | ~5s | Fast | Native support |
| Windows | ~5s | Fast | Native support |
| WSL (X Server) | ~10s | Medium | X11 overhead |
| WSL (Browser) | ~3s | Fast | No GUI overhead |
| Linux | ~5s | Fast | Native support |

---

## 🎯 **Recommendations by Platform**

### For macOS Users
✅ Use full desktop app (`./start.sh`)
- Native performance
- Best user experience
- Full feature set

### For Windows Users
✅ Use PowerShell version (`.\start.ps1`)
- Native Windows integration
- Can build installers
- Full functionality

### For WSL Users
✅ Use backend-only mode (`./test_backend.sh`)
- Fastest setup
- No X server needed
- Access from Windows browser at `localhost:8501`

Alternative: Set up X server if you need desktop window

### For Linux Users
✅ Use full desktop app (`./start.sh`)
- Native support
- All features work
- Best performance

---

## 🔄 **Cross-Platform Development**

### Sharing Between WSL and Windows

```bash
# Access WSL files from Windows
\\wsl$\Ubuntu\home\aya\work\optica_labs\algorithm_work

# Access Windows files from WSL
/mnt/c/Users/<YourUsername>/...
```

### Port Forwarding

All platforms can access:
- `localhost:8501` (Streamlit backend)
- Works in browser on any OS

---

## ✅ **Platform-Specific Checklist**

### macOS
- [ ] Homebrew installed
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed
- [ ] Xcode Command Line Tools installed
- [ ] AWS credentials configured
- [ ] OpenAI API key ready

### Windows
- [ ] Python installed (with PATH)
- [ ] Node.js installed
- [ ] PowerShell execution policy set
- [ ] AWS credentials configured
- [ ] OpenAI API key ready

### WSL
- [ ] Python 3.8+ installed
- [ ] Node.js 16+ installed (for full desktop)
- [ ] X Server running (if using full desktop)
- [ ] DISPLAY variable set (if using X server)
- [ ] AWS credentials configured
- [ ] OpenAI API key ready

---

## 📞 **Support**

**Platform-specific issues?**
- macOS: Check [README.md](README.md) troubleshooting
- Windows: Use [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)
- WSL: See [WSL_SETUP_GUIDE.md](WSL_SETUP_GUIDE.md)

**General help**: Open issue or check documentation

---

**Last Updated**: December 9, 2024
**Tested On**: macOS 13+, Windows 11, WSL2 (Ubuntu 22.04), Ubuntu 22.04
