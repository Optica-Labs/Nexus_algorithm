# PowerShell Start Script for Windows
# Vector Precognition Desktop App

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Starting Vector Precognition Desktop" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if setup was run
if (-Not (Test-Path "python-backend\venv")) {
    Write-Host "❌ Error: Virtual environment not found" -ForegroundColor Red
    Write-Host "Please run .\setup.ps1 first" -ForegroundColor Yellow
    exit 1
}

if (-Not (Test-Path "electron\node_modules")) {
    Write-Host "❌ Error: Node modules not found" -ForegroundColor Red
    Write-Host "Please run .\setup.ps1 first" -ForegroundColor Yellow
    exit 1
}

# Activate Python venv
Write-Host "🐍 Activating Python environment..." -ForegroundColor Yellow
& ".\python-backend\venv\Scripts\Activate.ps1"

# Start Electron app (which will spawn Python backend)
Write-Host "🚀 Launching desktop application..." -ForegroundColor Yellow
Set-Location electron
npm start
