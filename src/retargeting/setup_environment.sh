#!/bin/bash
# Techunited Data Generation - Automated Environment Setup
# This script automates the environment creation and package installation

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "================================================="
echo "  Techunited Data Generation Setup"
echo "================================================="
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}✗ Conda not found!${NC}"
    echo "Please install Anaconda or Miniconda first:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ Conda found${NC}"
echo ""

# Check if environment already exists
ENV_NAME="HumanoidDataGeneration"
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${RED}⚠ Environment '${ENV_NAME}' already exists${NC}"
    read -p "Do you want to remove it and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Keeping existing environment. Skipping creation."
        SKIP_ENV_CREATE=true
    fi
fi

# Create conda environment
if [ "$SKIP_ENV_CREATE" != "true" ]; then
    echo ""
    echo "Step 1: Creating conda environment '${ENV_NAME}'..."
    conda create -y -n ${ENV_NAME} python=3.10
    echo -e "${GREEN}✓ Environment created${NC}"
fi

# Activate environment
echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}
echo -e "${GREEN}✓ Environment activated${NC}"

# Install GVHMR
echo ""
echo "Step 3: Installing GVHMR..."
cd GVHMR
pip install -r requirements.txt
pip install -e .
cd ..
echo -e "${GREEN}✓ GVHMR installed${NC}"

# Install GMR
echo ""
echo "Step 4: Installing GMR..."
pip install -e .
echo -e "${GREEN}✓ GMR installed${NC}"

# Install libstdcxx
echo ""
echo "Step 5: Installing rendering dependencies..."
conda install -c conda-forge libstdcxx-ng -y
echo -e "${GREEN}✓ Rendering dependencies installed${NC}"

# Create necessary directories
echo ""
echo "Step 6: Creating directory structure..."
mkdir -p data
mkdir -p videos
mkdir -p outputs
mkdir -p GVHMR/inputs/checkpoints/body_models/smplx
mkdir -p GVHMR/inputs/checkpoints/body_models/smpl
mkdir -p GVHMR/inputs/checkpoints/gvhmr
mkdir -p GVHMR/inputs/checkpoints/hmr2
mkdir -p GVHMR/inputs/checkpoints/vitpose
mkdir -p GVHMR/inputs/checkpoints/yolo
mkdir -p GVHMR/outputs
echo -e "${GREEN}✓ Directories created${NC}"

# Summary
echo ""
echo "Step 7: Reinstalling GMR with fixes..."
cd "$SCRIPT_DIR"
pip uninstall general-motion-retargeting -y > /dev/null 2>&1 || true
pip install -e . > /dev/null 2>&1
echo -e "${GREEN}✓ GMR reinstalled${NC}"

echo ""
echo -e "${GREEN}=================================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}=================================================${NC}"
echo ""
echo "Environment: ${ENV_NAME}"
echo "Python: $(python --version)"
echo ""
echo "Next steps:"
echo "  1. Run: ./download_checkpoints.sh"
echo "  2. Run: ./verify_installation.sh"
echo "  3. Run: ./run_pipeline.sh --video /path/to/video.mp4"
echo ""

