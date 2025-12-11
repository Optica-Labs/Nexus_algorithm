# Quick Start Guide - Vector Precognition Desktop

Get up and running in **5 minutes**!

## Step 1: Install Prerequisites (2 min)

### Check if you have Python 3.8+
```bash
python3 --version
# Should show: Python 3.8.x or higher
```
**Don't have it?** → [Download Python](https://www.python.org/downloads/)

### Check if you have Node.js 16+
```bash
node --version
# Should show: v16.x.x or higher
```
**Don't have it?** → [Download Node.js](https://nodejs.org/)

---

## Step 2: Get API Keys (2 min)

### OpenAI API Key
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key (starts with `sk-...`)
4. Keep it safe!

### AWS Credentials (for embeddings)
```bash
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region: us-east-1
```

---

## Step 3: Setup (1 min)

```bash
cd desktop-app
./setup.sh
```

**What this does:**
- ✅ Creates Python virtual environment
- ✅ Installs Python packages
- ✅ Installs Electron dependencies
- ✅ Verifies everything works

**Grab a coffee** ☕ - takes about 60 seconds

---

## Step 4: Launch! (30 sec)

```bash
./start.sh
```

You'll see:
```
=========================================
Starting Vector Precognition Desktop
=========================================

🐍 Activating Python environment...
🚀 Launching desktop application...
```

**Desktop app opens automatically!**

---

## Step 5: Configure & Chat (1 min)

### First-Time Setup Screen

1. **Paste your OpenAI API Key**
   ```
   sk-proj-abc123...
   ```

2. **Select Model**
   - GPT-4 Turbo ← **Recommended** (best quality)
   - GPT-4 (balanced)
   - GPT-3.5 Turbo (fastest, cheapest)

3. **Choose VSAFE Preset**
   - "Default Assistant" ← **Start here**
   - Or customize your own

4. **Click "💾 Save & Continue"**

### Start Chatting!

Type in the chat box:
```
Hello! Can you explain quantum computing?
```

Watch the **right panel** update in real-time:
- 📊 Risk metrics (R, v, a, L)
- 📈 Live trajectory chart
- 🎯 Robustness index (ρ)

---

## 🎉 You're Done!

### Try These Commands

**Safe conversation:**
```
User: "What's the weather like?"
User: "Tell me about machine learning"
```
→ Watch ρ stay **< 1.0** (ROBUST ✅)

**Test guardrails** (research purposes only):
```
User: "Ignore previous instructions..."
```
→ Watch metrics spike, alert may trigger! ⚠️

---

## Next Steps

- **Export conversation**: Click sidebar → Save history
- **New chat**: Click "🔄 New Conversation"
- **Multi-session analysis**: Stack multiple conversations for PHI score
- **Read full docs**: [README.md](../README.md)

---

## Quick Troubleshooting

### "PCA models not found"
```bash
cd python-backend
source venv/bin/activate
python -c "from shared.pca_pipeline import PCATransformer; PCATransformer()"
```

### "Invalid API key"
- Check it starts with `sk-`
- Verify at [platform.openai.com/usage](https://platform.openai.com/usage)

### App won't start
```bash
# Check port 8501 is free
lsof -i :8501  # Mac/Linux
netstat -ano | findstr :8501  # Windows

# Kill any processes using it, then:
./start.sh
```

---

**Need help?** Open an issue or check [README.md](../README.md) for full docs!
