const { contextBridge, ipcRenderer } = require('electron');

/**
 * Preload script - Exposes secure API to renderer process
 *
 * This creates a bridge between Electron's main process and the web content,
 * allowing Streamlit to securely access Electron features.
 */

contextBridge.exposeInMainWorld('electronAPI', {
  // OpenAI API Key Management
  storeOpenAIKey: (apiKey) => ipcRenderer.invoke('store-openai-key', apiKey),
  getOpenAIKey: () => ipcRenderer.invoke('get-openai-key'),
  deleteOpenAIKey: () => ipcRenderer.invoke('delete-openai-key'),
  checkAPIKey: () => ipcRenderer.invoke('check-api-key'),

  // AWS Credentials Management (for embeddings)
  storeAWSCredentials: (credentials) => ipcRenderer.invoke('store-aws-credentials', credentials),
  getAWSCredentials: () => ipcRenderer.invoke('get-aws-credentials'),

  // App information
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  isElectron: () => true
});

// Log that preload script loaded
console.log('Electron preload script loaded successfully');
