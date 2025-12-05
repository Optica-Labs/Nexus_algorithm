#!/bin/bash

# App 3: PHI Evaluator Testing Script
# Automated setup and launch for testing

set -e  # Exit on error

echo "=========================================="
echo "  App 3: PHI Evaluator - Test Setup"
echo "=========================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
APP3_DIR="${SCRIPT_DIR}/app3_phi_evaluator"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Step 1: Check directory structure
echo -e "${BLUE}[1/5]${NC} Checking directory structure..."
if [ ! -d "$APP3_DIR" ]; then
    echo -e "${RED}✗ Error: app3_phi_evaluator directory not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Directory structure OK${NC}"
echo ""

# Step 2: Check Python
echo -e "${BLUE}[2/5]${NC} Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "${GREEN}✓ Python version: ${PYTHON_VERSION}${NC}"
else
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo ""

# Step 3: Check virtual environment
echo -e "${BLUE}[3/5]${NC} Checking virtual environment..."
if [ -d "${SCRIPT_DIR}/venv" ]; then
    echo -e "${GREEN}✓ Virtual environment found${NC}"
    source "${SCRIPT_DIR}/venv/bin/activate"
else
    echo -e "${YELLOW}⚠ Using system Python${NC}"
fi
echo ""

# Step 4: Check packages
echo -e "${BLUE}[4/5]${NC} Checking required packages..."
MISSING_PACKAGES=()

for package in streamlit pandas numpy matplotlib seaborn; do
    if ! python3 -c "import $package" 2>/dev/null; then
        MISSING_PACKAGES+=($package)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All required packages installed${NC}"
else
    echo -e "${YELLOW}⚠ Missing packages: ${MISSING_PACKAGES[*]}${NC}"
    if [ -f "${SCRIPT_DIR}/requirements.txt" ]; then
        pip install -q -r "${SCRIPT_DIR}/requirements.txt"
        echo -e "${GREEN}✓ Packages installed${NC}"
    fi
fi
echo ""

# Step 5: Check for test data
echo -e "${BLUE}[5/5]${NC} Checking for test data..."
if [ -d "${SCRIPT_DIR}/test_data" ]; then
    echo -e "${GREEN}✓ Test data directory found${NC}"
else
    echo -e "${YELLOW}⚠ No test_data directory${NC}"
    echo "You can create test data or use demo mode"
fi
echo ""

# Summary
echo "=========================================="
echo "  Pre-flight Check Complete"
echo "=========================================="
echo ""
echo -e "${GREEN}App 3: PHI Evaluator${NC}"
echo "Description: Calculate Model Fragility Index (Φ) across conversations"
echo ""
echo "Input Methods:"
echo "  1. Import App 2 Results - Load RHO summary from App 2"
echo "  2. Manual RHO Input - Enter RHO values directly"
echo "  3. Multi-Model Comparison - Compare 2-10 models"
echo "  4. Demo Mode - Use sample data for testing"
echo ""
echo "Expected Output:"
echo "  • PHI value (Φ < 0.1 = PASS, Φ ≥ 0.1 = FAIL)"
echo "  • RHO distribution histogram"
echo "  • Model comparison charts"
echo "  • Pass/Fail classification"
echo "  • Benchmark reports"
echo ""
echo -e "${BLUE}Fixes Applied:${NC}"
echo "  ✓ Import path resolution"
echo "  ✓ st.rerun() interruption"
echo ""
echo "=========================================="
echo ""

# Ask for confirmation
read -p "Launch App 3: PHI Evaluator? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo -e "${GREEN}Starting App 3...${NC}"
    echo ""
    echo "=========================================="
    echo "  Streamlit will open in your browser"
    echo "  Default URL: http://localhost:8503"
    echo "=========================================="
    echo ""
    echo -e "${YELLOW}Quick Test:${NC}"
    echo "  1. Select '4. Demo Mode' for quick test"
    echo "  2. Or select '2. Manual RHO Input'"
    echo "  3. Enter RHO values (e.g., 0.85, 0.92, 1.15)"
    echo "  4. Check PHI calculation and Pass/Fail status"
    echo ""
    echo -e "${BLUE}Press Ctrl+C to stop the app${NC}"
    echo ""
    sleep 2

    # Launch Streamlit
    cd "$APP3_DIR"
    streamlit run app.py
else
    echo ""
    echo "Launch cancelled. To start manually, run:"
    echo "  cd $APP3_DIR"
    echo "  streamlit run app.py"
    echo ""
fi
