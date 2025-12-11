const { contextBridge, ipcRenderer } = require('electron');

/**
 * Preload script for secure IPC communication
 * Exposes limited API to renderer process
 */

contextBridge.exposeInMainWorld('electronAPI', {
  // API Key Management
  storeApiKey: (apiKey) => ipcRenderer.invoke('store-api-key', apiKey),
  getApiKey: () => ipcRenderer.invoke('get-api-key'),
  deleteApiKey: () => ipcRenderer.invoke('delete-api-key'),

  // AWS Credentials Management
  storeAWSCredentials: (credentials) => ipcRenderer.invoke('store-aws-credentials', credentials),
  getAWSCredentials: () => ipcRenderer.invoke('get-aws-credentials'),

  // App Info
  platform: process.platform,
  version: process.versions.electron
});

console.log('Preload script loaded - electronAPI exposed to window');
