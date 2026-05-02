#!/bin/bash
# Quick Start Script for CoOp Graph on CPU
# Usage: bash quick_start.sh /path/to/datasets caltech101

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}CoOp Graph - Quick Start (CPU)${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Check arguments
if [ -z "$1" ]; then
    echo -e "${RED}Error: Please provide dataset path${NC}"
    echo "Usage: bash quick_start.sh /path/to/datasets [dataset_name]"
    echo "Example: bash quick_start.sh ~/datasets caltech-101"
    exit 1
fi

DATASET_PATH="$1"
DATASET_NAME="${2:-caltech-101}"
BACKBONE="rn50"

# Verify dataset exists
echo -e "${YELLOW}[1/5] Checking dataset...${NC}"
if [ ! -d "$DATASET_PATH/$DATASET_NAME" ]; then
    echo -e "${RED}Dataset not found at: $DATASET_PATH/$DATASET_NAME${NC}"
    echo -e "${YELLOW}Available datasets:${NC}"
    ls -la "$DATASET_PATH" 2>/dev/null || echo "  (dataset directory not found)"
    exit 1
fi
echo -e "${GREEN}✓ Dataset found${NC}\n"

# Check Python
echo -e "${YELLOW}[2/5] Checking Python environment...${NC}"
python --version
TORCH_CHECK=$(python -c "import torch; print(f'torch {torch.__version__}, cuda: {torch.cuda.is_available()}')" 2>/dev/null)
echo -e "${GREEN}✓ $TORCH_CHECK${NC}\n"

# Check required modules
echo -e "${YELLOW}[3/5] Checking dependencies...${NC}"
python -c "import clip; print('✓ CLIP installed')" || (echo "Installing CLIP..." && pip install git+https://github.com/openai/CLIP.git > /dev/null)
python -c "import dassl; print('✓ Dassl installed')" || (echo "Installing Dassl..." && cd Dassl.pytorch && python setup.py develop > /dev/null && cd ..)
echo -e "${GREEN}✓ All dependencies ready${NC}\n"

# Determine config file
CONFIG_FILE="configs/trainers/CoOp/${BACKBONE}_ep50.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    CONFIG_FILE="configs/trainers/CoOp/${BACKBONE}.yaml"
fi

# Determine dataset config
DATASET_CONFIG="configs/datasets/${DATASET_NAME}.yaml"
if [ ! -f "$DATASET_CONFIG" ]; then
    echo -e "${RED}Dataset config not found: $DATASET_CONFIG${NC}"
    echo "Available dataset configs:"
    ls -1 configs/datasets/
    exit 1
fi

echo -e "${YELLOW}[4/5] Configuration${NC}"
echo "  Dataset: $DATASET_NAME"
echo "  Dataset Path: $DATASET_PATH"
echo "  Backbone: $BACKBONE"
echo "  Config: $CONFIG_FILE"
echo ""

# Run training
OUTPUT_DIR="output/coop_graph/${DATASET_NAME}/quick_test"
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}[5/5] Starting training...${NC}"
echo -e "${GREEN}Command:${NC}"
echo "python train.py \\"
echo "  --root $DATASET_PATH \\"
echo "  --trainer CoOpGraph \\"
echo "  --dataset-config-file $DATASET_CONFIG \\"
echo "  --config-file $CONFIG_FILE \\"
echo "  --output-dir $OUTPUT_DIR \\"
echo "  TRAINER.COOP.PREC fp32 DATALOADER.NUM_WORKERS 0 OPTIM.MAX_EPOCH 5 DATASET.NUM_SHOTS 4"
echo ""

python train.py \
  --root "$DATASET_PATH" \
  --trainer CoOpGraph \
  --dataset-config-file "$DATASET_CONFIG" \
  --config-file "$CONFIG_FILE" \
  --output-dir "$OUTPUT_DIR" \
  TRAINER.COOP.PREC fp32 \
  DATALOADER.NUM_WORKERS 0 \
  OPTIM.MAX_EPOCH 5 \
  DATASET.NUM_SHOTS 4

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}✓ Training complete!${NC}"
echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
echo -e "${GREEN}========================================${NC}"
