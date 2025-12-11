# Start script for Desktop App2 (App4 + ChatGPT) on Windows
# Run this in PowerShell

Write-Host "=========================================" -ForegroundColor Blue
Write-Host "Starting Desktop App2 - App4 + ChatGPT" -ForegroundColor Blue
Write-Host "=========================================" -ForegroundColor Blue
Write-Host ""

# Check if setup was run
if (-not (Test-Path "python-backend\venv")) {
    Write-Host "❌ Python virtual environment not found" -ForegroundColor Red
    Write-Host "   Please run .\setup.ps1 first"
    exit 1
}

if (-not (Test-Path "electron\node_modules")) {
    Write-Host "❌ Node modules not found" -ForegroundColor Red
    Write-Host "   Please run .\setup.ps1 first"
    exit 1
}

# Activate Python virtual environment
Write-Host "Activating Python environment..."
.\python-backend\venv\Scripts\Activate.ps1

# Export AWS credentials if configured
if ($env:AWS_ACCESS_KEY_ID) {
    if (-not $env:AWS_DEFAULT_REGION) {
        $env:AWS_DEFAULT_REGION = "us-east-1"
    }
    Write-Host "✓ AWS credentials configured" -ForegroundColor Green
}

# Start Electron app
Write-Host "Launching Electron app..."
Write-Host ""
Set-Location electron
npm start
