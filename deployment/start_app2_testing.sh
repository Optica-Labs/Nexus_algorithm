#!/bin/bash

# App 2: RHO Calculator Testing Script
# Automated setup and launch for testing

set -e  # Exit on error

echo "=========================================="
echo "  App 2: RHO Calculator - Test Setup"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP2_DIR="${SCRIPT_DIR}/app2_rho_calculator"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check if we're in the right directory
echo -e "${BLUE}[1/6]${NC} Checking directory structure..."
if [ ! -d "$APP2_DIR" ]; then
    echo -e "${RED}✗ Error: app2_rho_calculator directory not found${NC}"
    echo "Current directory: $(pwd)"
    exit 1
fi
echo -e "${GREEN}✓ Directory structure OK${NC}"
echo ""

# Step 2: Check Python version
echo -e "${BLUE}[2/6]${NC} Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python version: ${PYTHON_VERSION}${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo ""

# Step 3: Check if virtual environment exists
echo -e "${BLUE}[3/6]${NC} Checking virtual environment..."
if [ -d "${SCRIPT_DIR}/venv" ]; then
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    source "${SCRIPT_DIR}/venv/bin/activate"
else
    echo -e "${YELLOW}⚠ No virtual environment found${NC}"
    echo "Using system Python installation"
fi
echo ""

# Step 4: Check required packages
echo -e "${BLUE}[4/6]${NC} Checking required packages..."
MISSING_PACKAGES=()

for package in streamlit pandas numpy boto3 matplotlib seaborn; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=($package)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All required packages installed${NC}"
else
    echo -e "${YELLOW}⚠ Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    echo "Installing missing packages..."
    if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
        pip install -q -r "${SCRIPT_DIR}/requirements.txt"
        echo -e "${GREEN}✓ Packages installed${NC}"
    else
        echo -e "${RED}✗ requirements.txt not found${NC}"
        echo "Please install packages manually: pip install streamlit pandas numpy boto3 matplotlib seaborn"
    fi
fi
echo ""

# Step 5: Check environment variables
echo -e "${BLUE}[5/6]${NC} Checking AWS credentials..."
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo -e "${GREEN}✓ .env file found${NC}"
    source "${SCRIPT_DIR}/.env"

    if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
        echo -e "${YELLOW}⚠ AWS credentials not set in .env${NC}"
        echo "You may need to configure AWS credentials for embeddings"
    else
        echo -e "${GREEN}✓ AWS credentials configured${NC}"
    fi
else
    echo -e "${YELLOW}⚠ No .env file found${NC}"
    echo "Create .env file with AWS credentials if needed"
fi
echo ""

# Step 6: Check models directory
echo -e "${BLUE}[6/6]${NC} Checking models directory..."
if [ -d "${SCRIPT_DIR}/models" ]; then
    MODEL_COUNT=$(find "${SCRIPT_DIR}/models" -name "*.pkl" 2>/dev/null | wc -l)
    if [ $MODEL_COUNT -gt 0 ]; then
        echo -e "${GREEN}✓ Found ${MODEL_COUNT} model file(s)${NC}"
    else
        echo -e "${YELLOW}⚠ No .pkl files found in models directory${NC}"
        echo "Make sure PCA models are in deployment/models/"
    fi
else
    echo -e "${YELLOW}⚠ Models directory not found${NC}"
    echo "Create models/ directory and add PCA pipeline files"
fi
echo ""

# Summary
echo "=========================================="
echo "  Pre-flight Check Complete"
echo "=========================================="
echo ""
echo -e "${GREEN}App 2: RHO Calculator${NC}"
echo "Description: Calculate Robustness Index (ρ) for conversations"
echo ""
echo "Input Methods:"
echo "  1. Single Conversation - Analyze one conversation"
echo "  2. Import App 1 Results - Load CSV/JSON from App 1"
echo "  3. Batch Upload - Analyze multiple conversations"
echo ""
echo "Expected Output:"
echo "  • RHO value (ρ < 1.0 = ROBUST, ρ > 1.0 = FRAGILE)"
echo "  • Cumulative risk chart"
echo "  • RHO timeline plot"
echo "  • Statistics summary"
echo "  • Export options (CSV, JSON, PNG)"
echo ""
echo -e "${BLUE}Fixes Applied:${NC}"
echo "  ✓ Import path resolution"
echo "  ✓ NumPy comparison bug"
echo "  ✓ st.rerun() interruption"
echo "  ✓ JSON export serialization"
echo ""
echo "=========================================="
echo ""

# Ask for confirmation
read -p "Launch App 2: RHO Calculator? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo -e "${GREEN}Starting App 2...${NC}"
    echo ""
    echo "=========================================="
    echo "  Streamlit will open in your browser"
    echo "  Default URL: http://localhost:8502"
    echo "=========================================="
    echo ""
    echo -e "${YELLOW}Quick Test:${NC}"
    echo "  1. Select '1. Single Conversation'"
    echo "  2. Paste a test conversation"
    echo "  3. Click analyze button"
    echo "  4. Check RHO value and visualizations"
    echo "  5. Test exports (CSV, JSON, PNG)"
    echo ""
    echo -e "${BLUE}Press Ctrl+C to stop the app${NC}"
    echo ""
    sleep 2

    # Launch Streamlit
    cd "$APP2_DIR"
    streamlit run app.py
else
    echo ""
    echo "Launch cancelled. To start manually, run:"
    echo "  cd $APP2_DIR"
    echo "  streamlit run app.py"
    echo ""
fi
