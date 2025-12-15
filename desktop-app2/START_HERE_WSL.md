# Desktop App2 - Quick Start Guide (WSL)

## Problem: Electron doesn't work in WSL

The full Electron desktop app requires GUI libraries that aren't available in WSL.

**Solution**: Use **browser-only mode** - works perfectly in WSL!

## Step-by-Step Instructions

### 1. Kill the blocking process (one-time fix)

There's a Streamlit process from another app blocking port 8501.

Run this command and enter your sudo password when prompted:

```bash
./KILL_ROOT_STREAMLIT.sh
```

### 2. Start Desktop App2

```bash
./START_FINAL.sh
```

You should see:

```
=============================================
Starting Vector Precognition Desktop App2

  Open your browser to:
    http://localhost:8501

  Press Ctrl+C to stop
=============================================
```

### 3. Open in your browser

Open your Windows browser (Chrome, Edge, Firefox) and go to:

```
http://localhost:8501
```

You'll see the full Desktop App2 interface with 4 tabs:
- **Setup** - Configure your OpenAI API key
- **Chat** - Talk to ChatGPT with real-time risk analysis
- **Conversation Analysis** - View detailed metrics and dynamics plots
- **Safety Dashboard** - Overall conversation safety overview

## If you still get "Port already in use"

Check what's running on port 8501:

```bash
ps aux | grep streamlit
```

If you see a process, note the PID and kill it:

```bash
sudo kill -9 <PID>
```

Then run `./START_FINAL.sh` again.

## To stop the app

Press `Ctrl+C` in the terminal where Streamlit is running.

## Running on macOS (native desktop app)

If you're on macOS, you can run the full Electron app:

```bash
./start.sh
```

See [MACOS_APP_GUIDE.md](MACOS_APP_GUIDE.md) for building a native .app bundle.

## Troubleshooting

### "Failed to import App4 components"

Make sure you're in the correct directory:

```bash
cd /home/aya/work/optica_labs/algorithm_work/desktop-app2
```

### "Missing Python dependencies"

Activate the virtual environment and install requirements:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Need to test backend only

```bash
./test-backend.sh
```

This will show detailed import testing and error messages.
