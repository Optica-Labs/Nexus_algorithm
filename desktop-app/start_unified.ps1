#!/usr/bin/env pwsh
# Desktop App Launcher for Unified Dashboard (App4) - PowerShell
# 
# This script launches the Electron desktop app with the unified
# AI safety dashboard from deployment/app4_unified_dashboard

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Unified Dashboard Desktop App Launcher" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Navigate to electron directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ElectronDir = Join-Path $ScriptDir "electron"

Set-Location $ElectronDir

# Check if node_modules exists
if (-Not (Test-Path "node_modules")) {
    Write-Host "📦 Installing Electron dependencies..." -ForegroundColor Yellow
    npm install
}

# Check for Python dependencies
Write-Host ""
Write-Host "🐍 Checking Python environment..." -ForegroundColor Yellow
Write-Host "   Make sure you have installed dependencies from:" -ForegroundColor Gray
Write-Host "   - deployment/app4_unified_dashboard/requirements.txt" -ForegroundColor Gray
Write-Host "   - deployment/shared/requirements.txt (if exists)" -ForegroundColor Gray
Write-Host ""
Write-Host "   Quick install:" -ForegroundColor Gray
Write-Host "   pip install -r ../../deployment/app4_unified_dashboard/requirements.txt" -ForegroundColor Gray
Write-Host ""

# Ask user if they want to proceed
$response = Read-Host "Ready to launch? (y/n)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "Cancelled." -ForegroundColor Red
    exit 0
}

# Launch the Electron app with app4 backend
Write-Host ""
Write-Host "🚀 Starting Electron with Unified Dashboard..." -ForegroundColor Green
Write-Host ""

npm run start:app4
