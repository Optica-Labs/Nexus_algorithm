#!/bin/bash
# Setup script for Vector Precognition Desktop App
# Installs Python dependencies and Node.js packages

set -e  # Exit on error

echo "========================================="
echo "Vector Precognition Desktop - Setup"
echo "========================================="
echo ""

# Check Python version
echo "🐍 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    exit 1
fi

# Check Node.js version
echo ""
echo "📦 Checking Node.js version..."
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    echo "Please install Node.js from: https://nodejs.org/"
    exit 1
fi

node_version=$(node --version)
echo "Found Node.js $node_version"

# Setup Python backend
echo ""
echo "========================================="
echo "Setting up Python backend..."
echo "========================================="

cd python-backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Python backend setup complete"

# Setup Electron app
cd ../electron

echo ""
echo "========================================="
echo "Setting up Electron frontend..."
echo "========================================="

# Install npm packages
echo "Installing Node.js dependencies..."
npm install

echo "✅ Electron frontend setup complete"

# Return to root
cd ..

echo ""
echo "========================================="
echo "✅ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Configure AWS credentials for embeddings:"
echo "   export AWS_ACCESS_KEY_ID=your_key"
echo "   export AWS_SECRET_ACCESS_KEY=your_secret"
echo "   export AWS_DEFAULT_REGION=us-east-1"
echo ""
echo "2. Train PCA models (if not already done):"
echo "   cd python-backend"
echo "   source venv/bin/activate"
echo "   python -c 'from shared.pca_pipeline import PCATransformer; PCATransformer()'"
echo ""
echo "3. Get your OpenAI API key from:"
echo "   https://platform.openai.com/api-keys"
echo ""
echo "4. Start the desktop app:"
echo "   ./start.sh"
echo ""
