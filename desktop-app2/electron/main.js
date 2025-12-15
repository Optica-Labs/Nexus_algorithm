const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const fs = require('fs');
const Store = require('electron-store');
const axios = require('axios');

// Initialize secure store for API keys
const store = new Store({
  encryptionKey: 'vector-precognition-app4-secure-key-2024'
});

let mainWindow;
let pythonProcess;
const STREAMLIT_PORT = 8502;
const STREAMLIT_URL = `http://localhost:${STREAMLIT_PORT}`;

/**
 * Start Python backend server running App4 with ChatGPT integration
 */
function startPythonBackend() {
  const isDev = process.argv.includes('--dev');

  // Determine Python backend path
  const backendPath = isDev
    ? path.join(__dirname, '..', 'python-backend')
    : path.join(process.resourcesPath, 'python-backend');

  console.log(`Starting Python backend (App4 + ChatGPT) from: ${backendPath}`);

  // Get stored OpenAI API key and pass to environment
  const apiKey = store.get('openai_api_key', '');

  // Determine python executable: prefer env override, then venv python, then 'python'
  let pythonExec = process.env.PYTHON || process.env.PYTHON_BIN || 'python';

  // If a venv exists under the backend path, prefer its python executable
  try {
    const venvPython = path.join(backendPath, 'venv', 'bin', 'python');
    if (fs.existsSync(venvPython)) {
      pythonExec = venvPython;
    }
  } catch (e) {
    // ignore and fall back to env or system python
  }

  console.log(`Using python executable: ${pythonExec}`);

  // Start Streamlit server
  // Choose a safe working directory for the child process. If pythonExec is
  // an absolute path that exists, use its directory as cwd so we avoid
  // spawning from inside the Electron app bundle which can cause permission
  // or translocation issues on macOS.
  const spawnCwd = (path.isAbsolute(pythonExec) && fs.existsSync(pythonExec))
    ? path.dirname(pythonExec)
    : backendPath;

  pythonProcess = spawn(pythonExec, [
    '-m', 'streamlit', 'run',
    path.join(backendPath, 'app.py'),
    '--server.port', STREAMLIT_PORT.toString(),
    '--server.headless', 'true',
    '--browser.gatherUsageStats', 'false',
    '--server.address', 'localhost'
  ], {
    cwd: spawnCwd,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      OPENAI_API_KEY: apiKey,  // Pass API key to Python
      ELECTRON_MODE: 'true'     // Signal that we're running in Electron
    }
  });

  pythonProcess.stdout.on('data', (data) => {
    console.log(`[Python Backend] ${data.toString()}`);
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`[Python Backend Error] ${data.toString()}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python backend exited with code ${code}`);
  });
}

/**
 * Wait for Streamlit server to be ready
 */
async function waitForStreamlit(maxRetries = 30, delay = 1000) {
  for (let i = 0; i < maxRetries; i++) {
    try {
      await axios.get(STREAMLIT_URL);
      console.log('Streamlit server is ready!');
      return true;
    } catch (error) {
      console.log(`Waiting for Streamlit... (${i + 1}/${maxRetries})`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }
  return false;
}

/**
 * Create main application window
 */
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    minWidth: 1200,
    minHeight: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true
    },
    icon: path.join(__dirname, '..', 'resources', 'icon.png'),
    title: 'Vector Precognition - AI Safety Dashboard (App4 + ChatGPT)',
    backgroundColor: '#1E1E1E'
  });

  // Load Streamlit app
  mainWindow.loadURL(STREAMLIT_URL);

  // Open DevTools in development mode
  if (process.argv.includes('--dev')) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

/**
 * IPC Handlers for secure communication with renderer process
 */

// Store OpenAI API Key
ipcMain.handle('store-openai-key', async (event, apiKey) => {
  try {
    store.set('openai_api_key', apiKey);

    // Restart Python backend with new key
    if (pythonProcess) {
      pythonProcess.kill();
      await new Promise(resolve => setTimeout(resolve, 1000));
      startPythonBackend();
      await waitForStreamlit();
    }

    return { success: true, message: 'OpenAI API key stored securely and backend restarted' };
  } catch (error) {
    return { success: false, message: error.message };
  }
});

// Retrieve OpenAI API Key
ipcMain.handle('get-openai-key', async () => {
  try {
    const apiKey = store.get('openai_api_key', '');
    return { success: true, apiKey };
  } catch (error) {
    return { success: false, message: error.message };
  }
});

// Delete OpenAI API Key
ipcMain.handle('delete-openai-key', async () => {
  try {
    store.delete('openai_api_key');
    return { success: true, message: 'OpenAI API key deleted' };
  } catch (error) {
    return { success: false, message: error.message };
  }
});

// Store AWS credentials (for embeddings)
ipcMain.handle('store-aws-credentials', async (event, credentials) => {
  try {
    store.set('aws_credentials', credentials);

    // Restart Python backend with new credentials
    if (pythonProcess) {
      pythonProcess.kill();
      await new Promise(resolve => setTimeout(resolve, 1000));
      startPythonBackend();
      await waitForStreamlit();
    }

    return { success: true, message: 'AWS credentials stored securely and backend restarted' };
  } catch (error) {
    return { success: false, message: error.message };
  }
});

// Get AWS credentials
ipcMain.handle('get-aws-credentials', async () => {
  try {
    const credentials = store.get('aws_credentials', {});
    return { success: true, credentials };
  } catch (error) {
    return { success: false, message: error.message };
  }
});

// Check if API key is configured
ipcMain.handle('check-api-key', async () => {
  try {
    const apiKey = store.get('openai_api_key', '');
    const isConfigured = apiKey && apiKey.startsWith('sk-');
    return { success: true, isConfigured, hasKey: !!apiKey };
  } catch (error) {
    return { success: false, message: error.message };
  }
});

/**
 * App lifecycle events
 */

app.whenReady().then(async () => {
  console.log('Electron app is ready');

  // Start Python backend
  startPythonBackend();

  // Wait for Streamlit to be ready
  const isReady = await waitForStreamlit();

  if (isReady) {
    createWindow();
  } else {
    console.error('Failed to start Streamlit server');
    app.quit();
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  // Kill Python backend process
  if (pythonProcess) {
    console.log('Terminating Python backend...');
    pythonProcess.kill();
  }
});

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('Uncaught Exception:', error);
});
