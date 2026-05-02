#!/bin/bash
# Setup script for CoOp Graph on CPU
# This script sets up the environment and installs all dependencies

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setting up CoOp Graph for CPU${NC}"
echo -e "${GREEN}================================${NC}\n"

# Step 1: Create environment
echo -e "${YELLOW}[1/5] Creating Conda environment...${NC}"
if conda env list | grep -q "coop_cpu"; then
    echo "Environment 'coop_cpu' already exists. Activating..."
else
    conda create -n coop_cpu python=3.8 -y
fi

# Activate environment (source for bash)
eval "$(conda shell.bash hook)"
conda activate coop_cpu

# Step 2: Install PyTorch (CPU)
echo -e "${YELLOW}[2/5] Installing PyTorch (CPU version)...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q

# Step 3: Install dependencies
echo -e "${YELLOW}[3/5] Installing dependencies...${NC}"
pip install ftfy regex tqdm -q

# Step 4: Install CLIP
echo -e "${YELLOW}[4/5] Installing OpenAI CLIP...${NC}"
pip install git+https://github.com/openai/CLIP.git -q

# Step 5: Install Dassl
echo -e "${YELLOW}[5/5] Installing Dassl.pytorch...${NC}"
if [ -d "Dassl.pytorch" ]; then
    cd Dassl.pytorch
    python setup.py develop -q > /dev/null 2>&1
    cd ..
    echo -e "${GREEN}✓ Dassl installed${NC}"
else
    echo -e "${YELLOW}Note: Dassl.pytorch not found. Make sure it's cloned.${NC}"
fi

# Verification
echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Verification${NC}"
echo -e "${GREEN}================================${NC}\n"

python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import clip; print('✓ CLIP installed')"
python -c "import dassl; print('✓ Dassl installed')" || echo "⚠ Dassl may need setup.py develop to be run"
python -c "import ftfy; import regex; import tqdm; print('✓ Dependencies installed')"

echo -e "\n${GREEN}================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}================================${NC}\n"

echo -e "${YELLOW}Next steps:${NC}"
echo "1. Prepare your dataset using DATASETS.md"
echo "2. Run: bash quick_start.sh /path/to/datasets caltech-101"
echo "3. Or read: RUN_GUIDE_CPU.md for detailed instructions"
echo ""
echo -e "${YELLOW}To activate environment later:${NC}"
echo "  conda activate coop_cpu"
