# 🚀 Start Here - Desktop App2 Quick Launch

**For WSL Users - 3 Simple Steps**

---

## Step 1: Kill Any Running Processes

```bash
pkill -f streamlit
```

**Output**: (may show nothing - that's okay)

---

## Step 2: Start the Backend

```bash
cd /home/aya/work/optica_labs/algorithm_work/desktop-app2
source python-backend/venv/bin/activate
cd python-backend
streamlit run app.py
```

**Expected Output**:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://172.x.x.x:8501
```

---

## Step 3: Open Your Browser

**In your Windows browser**, go to:
```
http://localhost:8501
```

---

## What You'll See

### First Screen: API Key Setup

1. Enter your OpenAI API key (get from https://platform.openai.com/api-keys)
2. Select GPT model (GPT-3.5 Turbo recommended)
3. Click "Save & Continue"

### OR: Test Without API Key

1. Expand "🧪 Test without API key (Mock Mode)"
2. Click "Use Mock Mode"
3. App loads with simulated responses

### After Setup: 4-Tab Interface

- **💬 Live Chat** - Chat with ChatGPT + real-time safety metrics
- **📊 RHO Analysis** - Per-conversation robustness
- **🎯 PHI Benchmark** - Multi-conversation fragility scoring
- **⚙️ Settings** - Configuration and session info

---

## To Stop the Server

Press `Ctrl+C` in the terminal where Streamlit is running.

---

## If Port 8501 is Already in Use

```bash
# Kill the process using the port
pkill -f streamlit

# Then start again
cd /home/aya/work/optica_labs/algorithm_work/desktop-app2
source python-backend/venv/bin/activate
cd python-backend
streamlit run app.py
```

---

## Alternative: Use the Test Script

```bash
# First, kill any running processes
pkill -f streamlit

# Then run
./test-backend.sh
```

---

## Quick Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution**:
```bash
cd /home/aya/work/optica_labs/algorithm_work/desktop-app2
./setup.sh
```

### Issue: "Failed to import App4 components"

**Solution**: Make sure you're in the right directory:
```bash
pwd
# Should show: /home/aya/work/optica_labs/algorithm_work/desktop-app2
```

### Issue: Blank page in browser

**Solution**:
1. Wait 10-15 seconds for Streamlit to fully start
2. Refresh the page (Ctrl+R)
3. Check terminal for error messages

---

## The Easiest Way (Copy-Paste)

Just run these commands in order:

```bash
pkill -f streamlit
cd /home/aya/work/optica_labs/algorithm_work/desktop-app2
source python-backend/venv/bin/activate
cd python-backend
streamlit run app.py
```

Then open: `http://localhost:8501` in your browser

---

**That's it!** Your full App4 + ChatGPT desktop app is now running! 🎉
