# Installation Guide - Vector Precognition Desktop App

Complete installation instructions for all platforms.

---

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10+, macOS 10.14+, Ubuntu 18.04+ (or equivalent Linux)
- **RAM**: 4 GB
- **Disk Space**: 2 GB free
- **Internet**: Required for ChatGPT API and AWS Bedrock

### Software Requirements
- **Python**: 3.8 or higher
- **Node.js**: 16.0 or higher
- **npm**: 7.0 or higher (included with Node.js)

---

## 🚀 Installation Steps

### Step 1: Install Python

#### Windows
1. Download from [python.org/downloads](https://www.python.org/downloads/)
2. Run installer
3. ✅ **IMPORTANT**: Check "Add Python to PATH"
4. Click "Install Now"
5. Verify:
   ```cmd
   python --version
   ```

#### macOS
```bash
# Using Homebrew (recommended)
brew install python@3.11

# Or download from python.org
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### Step 2: Install Node.js

#### Windows
1. Download from [nodejs.org](https://nodejs.org/)
2. Run installer (choose LTS version)
3. Accept defaults
4. Verify:
   ```cmd
   node --version
   npm --version
   ```

#### macOS
```bash
# Using Homebrew
brew install node

# Or download from nodejs.org
```

#### Linux (Ubuntu/Debian)
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
```

### Step 3: Configure AWS Credentials

#### Option A: AWS CLI (Recommended)
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: us-east-1
# - Default output format: json
```

#### Option B: Environment Variables
```bash
# Linux/macOS (add to ~/.bashrc or ~/.zshrc)
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1

# Windows (Command Prompt)
setx AWS_ACCESS_KEY_ID your_access_key_here
setx AWS_SECRET_ACCESS_KEY your_secret_key_here
setx AWS_DEFAULT_REGION us-east-1

# Windows (PowerShell)
$env:AWS_ACCESS_KEY_ID="your_access_key_here"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key_here"
$env:AWS_DEFAULT_REGION="us-east-1"
```

### Step 4: Get OpenAI API Key

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Sign up or log in
3. Click "Create new secret key"
4. Name it: "Vector Precognition Desktop"
5. Copy the key (starts with `sk-`)
6. **Save it securely** - you'll enter it in the app

### Step 5: Download/Clone Desktop App

```bash
# If using Git
git clone https://github.com/optica-labs/algorithm_work.git
cd algorithm_work/desktop-app

# Or download ZIP and extract
# Then navigate to desktop-app folder
```

### Step 6: Run Setup

#### Linux/macOS
```bash
chmod +x setup.sh start.sh
./setup.sh
```

#### Windows (Git Bash or WSL)
```bash
./setup.sh
```

#### Windows (PowerShell)
```powershell
# Run setup manually:
cd python-backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

cd ..\electron
npm install
```

**Wait time**: 2-3 minutes (downloads ~500 MB of dependencies)

### Step 7: Initialize PCA Models

```bash
# Linux/macOS
cd python-backend
source venv/bin/activate
python -c "from shared.pca_pipeline import PCATransformer; PCATransformer()"

# Windows
cd python-backend
.\venv\Scripts\activate
python -c "from shared.pca_pipeline import PCATransformer; PCATransformer()"
```

**This creates**:
- `python-backend/models/pca_model.pkl`
- `python-backend/models/embedding_scaler.pkl`

### Step 8: Launch Application

#### Linux/macOS
```bash
./start.sh
```

#### Windows (Git Bash/WSL)
```bash
./start.sh
```

#### Windows (PowerShell)
```powershell
# Run manually:
cd python-backend
.\venv\Scripts\activate
cd ..\electron
npm start
```

**Expected output**:
```
=========================================
Starting Vector Precognition Desktop
=========================================

🐍 Activating Python environment...
🚀 Launching desktop application...
[Python Backend] Streamlit server starting...
[Python Backend] Server running on http://localhost:8501
Electron app is ready
Streamlit server is ready!
```

**Desktop window opens automatically!**

---

## 🎯 First-Time Configuration

1. **Enter API Key Screen**
   - Paste your OpenAI API key
   - Select model (GPT-4 recommended)
   - Choose VSAFE preset
   - Click "Save & Continue"

2. **Chat Interface Loads**
   - Left: Chat messages
   - Right: Risk analysis
   - Sidebar: Controls

3. **Start Chatting**
   - Type a message
   - Press Enter
   - Watch real-time analysis!

---

## 🐛 Troubleshooting Installation

### "Python not found"

**Windows**:
```cmd
# Reinstall Python with PATH option checked
# Or add manually:
setx PATH "%PATH%;C:\Python311;C:\Python311\Scripts"
```

**Linux/macOS**:
```bash
# Use python3 explicitly
python3 --version
# Update setup.sh to use python3 instead of python
```

### "Node not found"

**All Platforms**:
1. Close and reopen terminal
2. Verify installation:
   ```bash
   node --version
   ```
3. If still missing, reinstall Node.js

### "Permission denied" on setup.sh

```bash
chmod +x setup.sh start.sh
./setup.sh
```

### "Port 8501 already in use"

```bash
# Linux/macOS
lsof -i :8501
kill -9 <PID>

# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

### "AWS credentials not found"

```bash
# Verify configuration
aws sts get-caller-identity

# If fails, reconfigure:
aws configure
```

### "PCA models not found"

```bash
cd python-backend
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
python -c "from shared.pca_pipeline import PCATransformer; PCATransformer()"
```

### "OpenAI API error"

**Check API key**:
- Starts with `sk-`
- No extra spaces
- Account has credits: [platform.openai.com/usage](https://platform.openai.com/usage)

**Try regenerating key**:
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Revoke old key
3. Create new key
4. Enter in app

### "Module not found" errors

```bash
# Reinstall Python dependencies
cd python-backend
source venv/bin/activate
pip install -r requirements.txt --force-reinstall
```

### Electron window is blank

**Wait 30 seconds** - Streamlit takes time to start

If still blank:
```bash
# Check Python backend logs in terminal
# Look for errors in the console output
```

---

## 🔧 Advanced Configuration

### Change Streamlit Port

Edit `electron/main.js`:
```javascript
const STREAMLIT_PORT = 8502;  // Change to your preferred port
```

### Use Custom Python Path

Edit `electron/main.js`:
```javascript
pythonProcess = spawn('/path/to/your/python', [
    '-m', 'streamlit', 'run',
    // ...
]);
```

### Offline Mode (No AWS)

Not currently supported - AWS Bedrock required for embeddings.

**Alternative**: Use local embeddings (requires code modification):
- Replace `shared/embeddings.py` with local model (e.g., SentenceTransformers)
- Update `pca_pipeline.py` to use local embeddings

---

## 📦 Building Installers

### Prerequisites
```bash
cd electron
npm install electron-builder --save-dev
```

### Windows Installer
```bash
npm run build:win
# Output: electron/dist/Vector Precognition Setup.exe
```

### macOS Installer
```bash
npm run build:mac
# Output: electron/dist/Vector Precognition.dmg
```

### Linux Installer
```bash
npm run build:linux
# Output: electron/dist/vector-precognition.AppImage
```

**Note**: Installers include bundled Python backend.

---

## 🔄 Updating

### Update Python Dependencies
```bash
cd python-backend
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

### Update Node Dependencies
```bash
cd electron
npm update
```

### Update Application Code
```bash
git pull origin main
./setup.sh  # Reinstall if requirements changed
```

---

## 🗑️ Uninstallation

### Remove Application Files
```bash
# Delete entire directory
rm -rf desktop-app/

# Or keep for reinstall, just remove dependencies:
rm -rf python-backend/venv
rm -rf electron/node_modules
```

### Remove Configuration
```bash
# Linux/macOS
rm -rf ~/.config/vector-precognition-desktop

# Windows
# Delete: %APPDATA%\vector-precognition-desktop
```

### Revoke API Keys
1. OpenAI: [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. AWS: `aws iam delete-access-key --access-key-id <KEY_ID>`

---

## 📞 Support

**Installation Issues?**
- Check [README.md](README.md) troubleshooting section
- Review [docs/QUICKSTART.md](docs/QUICKSTART.md)
- Open GitHub issue with:
  - OS version
  - Python version (`python --version`)
  - Node version (`node --version`)
  - Error message (full output)

**Email**: support@opticalabs.com

---

## ✅ Post-Installation Checklist

- [ ] Python 3.8+ installed and in PATH
- [ ] Node.js 16+ installed and in PATH
- [ ] AWS credentials configured (`aws sts get-caller-identity` works)
- [ ] OpenAI API key obtained
- [ ] `./setup.sh` completed without errors
- [ ] PCA models initialized (files exist in `python-backend/models/`)
- [ ] `./start.sh` launches desktop window
- [ ] API key configuration screen appears
- [ ] Can enter API key and select model
- [ ] Chat interface loads after saving
- [ ] Can send message and receive response
- [ ] Risk metrics display in right panel
- [ ] Charts update after each turn

**All checked?** 🎉 **You're ready to use Vector Precognition Desktop!**

---

**Last Updated**: December 9, 2024
