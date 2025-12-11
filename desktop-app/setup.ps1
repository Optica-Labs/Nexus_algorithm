# PowerShell Setup Script for Windows
# Vector Precognition Desktop App

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Vector Precognition Desktop - Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "🐍 Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Python is not installed" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

# Check Node.js
Write-Host ""
Write-Host "📦 Checking Node.js version..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version
    Write-Host "Found: Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Error: Node.js is not installed" -ForegroundColor Red
    Write-Host "Download from: https://nodejs.org/" -ForegroundColor Yellow
    exit 1
}

# Setup Python backend
Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setting up Python backend..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

Set-Location python-backend

# Create virtual environment
if (-Not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "✅ Python backend setup complete" -ForegroundColor Green

# Setup Electron app
Set-Location ..\electron

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "Setting up Electron frontend..." -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Install npm packages
Write-Host "Installing Node.js dependencies..." -ForegroundColor Yellow
npm install

Write-Host "✅ Electron frontend setup complete" -ForegroundColor Green

# Return to root
Set-Location ..

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "✅ Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Configure AWS credentials for embeddings:" -ForegroundColor White
Write-Host '   $env:AWS_ACCESS_KEY_ID="your_key"' -ForegroundColor Gray
Write-Host '   $env:AWS_SECRET_ACCESS_KEY="your_secret"' -ForegroundColor Gray
Write-Host '   $env:AWS_DEFAULT_REGION="us-east-1"' -ForegroundColor Gray
Write-Host ""
Write-Host "2. Get your OpenAI API key from:" -ForegroundColor White
Write-Host "   https://platform.openai.com/api-keys" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start the desktop app:" -ForegroundColor White
Write-Host "   .\start.ps1" -ForegroundColor Gray
Write-Host ""
