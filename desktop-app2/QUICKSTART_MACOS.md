# Desktop App2 - macOS Quick Start

## 🚀 5-Minute Setup

### 1. Transfer Project to macOS

```bash
# Clone the repository on your macOS machine
git clone https://github.com/Optica-Labs/Nexus_algorithm.git
cd Nexus_algorithm/desktop-app2
```

### 2. Install Dependencies

```bash
# Install Electron dependencies
cd electron
npm install

# Setup Python backend
cd ../python-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..
```

### 3. Run the App

```bash
./start.sh
```

A native macOS window will open with the full Desktop App4 interface!

## 📝 First Use

1. **Setup Tab**: Enter your OpenAI API key
   - Get one at: https://platform.openai.com/api-keys
   
2. **Chat Tab**: Start chatting with ChatGPT
   - Every message is analyzed by Vector Precognition
   - See real-time risk metrics

3. **Analysis Tabs**: View detailed safety metrics and plots

## 🔧 Troubleshooting

**Port already in use?**
```bash
lsof -ti:8502 | xargs kill -9
./start.sh
```

**Module not found?**
```bash
cd electron
npm install
```

**Python errors?**
```bash
cd python-backend
source venv/bin/activate
pip install -r requirements.txt
```

## 📖 Full Documentation

See [MACOS_TESTING_GUIDE.md](MACOS_TESTING_GUIDE.md) for:
- Building native .app bundle
- Creating DMG installer
- Development mode
- Complete troubleshooting guide

---

**That's it! You're ready to test on macOS.** 🎉
