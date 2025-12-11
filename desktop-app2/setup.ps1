# Setup script for Desktop App2 (App4 + ChatGPT) on Windows
# Run this in PowerShell

Write-Host "=========================================" -ForegroundColor Blue
Write-Host "Desktop App2 Setup - App4 + ChatGPT" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue
Write-Host ""

# Step 1: Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion found" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 3 not found. Please install Python 3.8 or higher" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 2: Check Node.js
Write-Host "[2/5] Checking Node.js installation..." -ForegroundColor Cyan
try {
    $nodeVersion = node --version
    Write-Host "✓ Node.js $nodeVersion found" -ForegroundColor Green
} catch {
    Write-Host "❌ Node.js not found. Please install Node.js 16 or higher" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Step 3: Setup Python backend
Write-Host "[3/5] Setting up Python environment..." -ForegroundColor Cyan

Set-Location python-backend

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment..."
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Install Python dependencies
Write-Host "Installing Python dependencies..."
python -m pip install --upgrade pip
pip install -r requirements.txt

Set-Location ..
Write-Host "✓ Python environment setup complete" -ForegroundColor Green
Write-Host ""

# Step 4: Setup Electron
Write-Host "[4/5] Setting up Electron..." -ForegroundColor Cyan

Set-Location electron

# Install Node dependencies
Write-Host "Installing Node.js dependencies..."
npm install

Set-Location ..
Write-Host "✓ Electron setup complete" -ForegroundColor Green
Write-Host ""

# Step 5: Check AWS credentials
Write-Host "[5/5] Checking AWS configuration..." -ForegroundColor Cyan
if (-not $env:AWS_ACCESS_KEY_ID -or -not $env:AWS_SECRET_ACCESS_KEY) {
    Write-Host "⚠  AWS credentials not found in environment" -ForegroundColor Yellow
    Write-Host "   Embeddings will require AWS Bedrock access"
    Write-Host "   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
} else {
    Write-Host "✓ AWS credentials found" -ForegroundColor Green
}
Write-Host ""

# Success
Write-Host "=========================================" -ForegroundColor Green
Write-Host "✓ Setup Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Get your OpenAI API key from https://platform.openai.com/api-keys"
Write-Host "  2. Run the application:"
Write-Host ""
Write-Host "     .\start.ps1"
Write-Host ""
Write-Host "The app will prompt you to enter your API key on first run."
Write-Host ""
