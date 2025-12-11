#!/bin/bash
# Setup script for Desktop App2 (App4 + ChatGPT) on Linux/Mac

set -e  # Exit on error

echo "========================================="
echo "Desktop App2 Setup - App4 + ChatGPT"
echo "========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check Python
echo -e "${BLUE}[1/5]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python 3 not found. Please install Python 3.8 or higher${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION found"
echo ""

# Step 2: Check Node.js
echo -e "${BLUE}[2/5]${NC} Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Node.js not found. Please install Node.js 16 or higher${NC}"
    exit 1
fi

NODE_VERSION=$(node --version)
echo -e "${GREEN}✓${NC} Node.js $NODE_VERSION found"
echo ""

# Step 3: Setup Python backend
echo -e "${BLUE}[3/5]${NC} Setting up Python environment..."

cd python-backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

cd ..
echo -e "${GREEN}✓${NC} Python environment setup complete"
echo ""

# Step 4: Setup Electron
echo -e "${BLUE}[4/5]${NC} Setting up Electron..."

cd electron

# Install Node dependencies
echo "Installing Node.js dependencies..."
npm install

cd ..
echo -e "${GREEN}✓${NC} Electron setup complete"
echo ""

# Step 5: Check AWS credentials (for embeddings)
echo -e "${BLUE}[5/5]${NC} Checking AWS configuration..."
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo -e "${YELLOW}⚠${NC}  AWS credentials not found in environment"
    echo "   Embeddings will require AWS Bedrock access"
    echo "   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
else
    echo -e "${GREEN}✓${NC} AWS credentials found"
fi
echo ""

# Success
echo "========================================="
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Get your OpenAI API key from https://platform.openai.com/api-keys"
echo "  2. Run the application:"
echo ""
echo "     ./start.sh"
echo ""
echo "The app will prompt you to enter your API key on first run."
echo ""
