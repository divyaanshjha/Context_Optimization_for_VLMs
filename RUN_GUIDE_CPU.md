# Complete Guide to Running CoOp Graph on CPU

This guide covers everything you need to run the CoOp Graph model on CPU (no GPU required).

---

## Step 1: Environment Setup

### 1a. Create Conda Environment

```bash
cd /Users/divyanshjha/Desktop/4th_Year/4-2/Graph_Mining_project/CLIP_CoOp/CoOp

# Create environment (CPU-optimized)
conda create -n coop_cpu python=3.8 -y
conda activate coop_cpu
```

### 1b. Install PyTorch (CPU version)

```bash
# For CPU, install PyTorch without CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Or for specific version:
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
```

### 1c. Install Dependencies

```bash
# Install required packages
pip install ftfy regex tqdm

# Install OpenAI CLIP
pip install git+https://github.com/openai/CLIP.git

# Install Dassl.pytorch
cd Dassl.pytorch
python setup.py develop
cd ..
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
# Should show: CUDA available: False (or True but we don't use it)
```

---

## Step 2: Prepare Datasets

### Option A: Quick Start with Oxford Flowers (smallest dataset)

```bash
# Create data directory
mkdir -p /path/to/datasets  # e.g., ~/datasets

# Oxford Flowers 102 - ~330 MB
# 1. Download from: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
# 2. Extract to: /path/to/datasets/oxford_flowers/
```

### Option B: Caltech-101 (recommended for testing)

```bash
# Download from official source
# http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz

# Extract and organize:
cd /path/to/datasets
mkdir -p caltech-101
cd caltech-101
# Extract the tar.gz file here

# Download split file from:
# https://drive.google.com/file/d/1hyarUivQE36mY6jSomru6Fjd-JzwcCzN/view?usp=sharing
# Place as: caltech-101/split_zhou_Caltech101.json
```

### Data Directory Structure

```
/path/to/datasets/
├── caltech-101/
│   ├── 101_ObjectCategories/
│   │   ├── accordion/
│   │   ├── airplanes/
│   │   ├── ... (101 categories)
│   └── split_zhou_Caltech101.json
├── oxford_flowers/
│   ├── jpg/
│   ├── split_fewshot/
│   ├── split_zhou_OxfordFlowers.json
│   ├── cat_to_name.json
│   └── imagelabels.mat
```

---

## Step 3: Verify Tests Pass (Optional but Recommended)

```bash
python test_graph_utils.py
python test_gcn.py
python test_forward_backward.py

# Expected output: All should show PASS
```

---

## Step 4: Run CoOp Graph Experiments

### Core Configuration

The key file is `configs/trainers/CoOp/rn50_ep50.yaml` which now includes:

- `TRAINER.NAME: CoOpGraph` - Uses the graph-enhanced trainer
- `TRAINER.COOP.ALPHA_SMOOTH: 0.1` - Graph label smoothing weight
- `TRAINER.COOP.LAMBDA_LAP: 0.01` - Laplacian regularization weight
- `TRAINER.COOP.USE_GCN: False` - Toggle GCN module
- `TRAINER.COOP.PREC: "fp32"` - CPU-compatible precision

### Experiment 1: Basic CoOp Graph (Recommended First Test)

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --output-dir output/coop_graph/caltech101/test \
  --opts TRAINER.COOP.PREC fp32
```

**Expected output:**

```
Setting up trainer for dataset: Caltech101
Building model...
Loading CLIP (backbone: RN50)
Building graph from class names...
Graph built: XXX edges among 101 classes
Training started...
```

### Experiment 2: Graph Label Smoothing Only

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --output-dir output/coop_graph/caltech101/smooth_only \
  --opts \
    TRAINER.COOP.PREC fp32 \
    DATASET.NUM_SHOTS 16 \
    TRAINER.COOP.ALPHA_SMOOTH 0.1 \
    TRAINER.COOP.LAMBDA_LAP 0.0 \
    TRAINER.COOP.USE_GCN False
```

### Experiment 3: Laplacian Regularization Only

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --output-dir output/coop_graph/caltech101/lap_only \
  --opts \
    TRAINER.COOP.PREC fp32 \
    DATASET.NUM_SHOTS 16 \
    TRAINER.COOP.ALPHA_SMOOTH 0.0 \
    TRAINER.COOP.LAMBDA_LAP 0.01 \
    TRAINER.COOP.USE_GCN False
```

### Experiment 4: Smoothing + Laplacian Combined

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --output-dir output/coop_graph/caltech101/smooth_lap \
  --opts \
    TRAINER.COOP.PREC fp32 \
    DATASET.NUM_SHOTS 16 \
    TRAINER.COOP.ALPHA_SMOOTH 0.1 \
    TRAINER.COOP.LAMBDA_LAP 0.01 \
    TRAINER.COOP.USE_GCN False
```

### Experiment 5: With GCN Module

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --output-dir output/coop_graph/caltech101/with_gcn \
  --opts \
    TRAINER.COOP.PREC fp32 \
    DATASET.NUM_SHOTS 16 \
    TRAINER.COOP.ALPHA_SMOOTH 0.1 \
    TRAINER.COOP.LAMBDA_LAP 0.01 \
    TRAINER.COOP.USE_GCN True
```

---

## Step 5: Important CPU Configuration Options

### Key Hyperparameters for CPU

```bash
# Reduce batch size for lower memory usage
--opts DATALOADER.TRAIN_X.BATCH_SIZE 8

# Use fewer workers for data loading
--opts DATALOADER.NUM_WORKERS 0

# Reduce epochs for faster testing
--opts OPTIM.MAX_EPOCH 10

# Reduce number of shots for faster training
--opts DATASET.NUM_SHOTS 4
```

### Example: Minimal Config for Quick Testing

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --output-dir output/test \
  --opts \
    TRAINER.COOP.PREC fp32 \
    DATALOADER.TRAIN_X.BATCH_SIZE 8 \
    DATALOADER.NUM_WORKERS 0 \
    OPTIM.MAX_EPOCH 5 \
    DATASET.NUM_SHOTS 4
```

---

## Step 6: Evaluate Pre-trained Models

### Test Only Mode

```bash
python train.py \
  --root /path/to/datasets \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/caltech101.yaml \
  --config-file configs/trainers/CoOp/rn50_ep50.yaml \
  --model-dir output/coop_graph/caltech101/test \
  --eval-only \
  --opts TRAINER.COOP.PREC fp32
```

---

## Complete Example Script

Create `run_coop_graph.sh`:

```bash
#!/bin/bash

# Configuration
DATASET_PATH="/path/to/datasets"
DATASET="caltech101"
BACKBONE="rn50"
OUTPUT_DIR="output/coop_graph/${DATASET}"

# Activate environment
conda activate coop_cpu

# Run training
python train.py \
  --root ${DATASET_PATH} \
  --trainer CoOpGraph \
  --dataset-config-file configs/datasets/${DATASET}.yaml \
  --config-file configs/trainers/CoOp/${BACKBONE}_ep50.yaml \
  --output-dir ${OUTPUT_DIR} \
  --opts \
    TRAINER.COOP.PREC fp32 \
    DATALOADER.NUM_WORKERS 0 \
    DATASET.NUM_SHOTS 16 \
    TRAINER.COOP.ALPHA_SMOOTH 0.1 \
    TRAINER.COOP.LAMBDA_LAP 0.01 \
    TRAINER.COOP.USE_GCN False

echo "Training complete! Results saved to ${OUTPUT_DIR}"
```

Run with:

```bash
bash run_coop_graph.sh
```

---

## Troubleshooting

### Issue 1: "No module named 'clip'"

**Solution:**

```bash
pip install git+https://github.com/openai/CLIP.git
```

### Issue 2: "No module named 'dassl'"

**Solution:**

```bash
cd Dassl.pytorch
python setup.py develop
cd ..
```

### Issue 3: "CUDA out of memory" (on GPU) or very slow (on CPU)

**Solution:** Reduce batch size and workers:

```bash
--opts \
  DATALOADER.TRAIN_X.BATCH_SIZE 4 \
  DATALOADER.NUM_WORKERS 0 \
  OPTIM.MAX_EPOCH 5
```

### Issue 4: "Dataset not found"

**Solution:** Verify your dataset directory structure matches DATASETS.md and use correct `--root` path:

```bash
# Check if dataset exists
ls -la /path/to/datasets/caltech-101/
```

### Issue 5: Error with mixed precision on CPU

**Solution:** Always use `TRAINER.COOP.PREC fp32` for CPU:

```bash
--opts TRAINER.COOP.PREC fp32
```

### Issue 6: Very slow training on CPU

**Expected behavior** - CPU training is slower. To speed up testing:

- Reduce `MAX_EPOCH` to 5-10
- Reduce `NUM_SHOTS` to 4-8
- Reduce `BATCH_SIZE` to 4-8
- Use `NUM_WORKERS 0` for data loading

---

## Parameter Reference

| Parameter                     | Default | CPU Recommendation | Description              |
| ----------------------------- | ------- | ------------------ | ------------------------ |
| TRAINER.COOP.PREC             | "fp16"  | "fp32"             | Precision (fp32 for CPU) |
| DATALOADER.NUM_WORKERS        | 8       | 0                  | Data loading threads     |
| DATALOADER.TRAIN_X.BATCH_SIZE | 32      | 8                  | Training batch size      |
| OPTIM.MAX_EPOCH               | 50      | 10-20              | Number of epochs         |
| DATASET.NUM_SHOTS             | 16      | 4-16               | Shots per class          |
| TRAINER.COOP.ALPHA_SMOOTH     | 0.1     | 0.1                | Label smoothing weight   |
| TRAINER.COOP.LAMBDA_LAP       | 0.01    | 0.01               | Laplacian regularization |
| TRAINER.COOP.USE_GCN          | False   | False/True         | Enable GCN module        |

---

## Next Steps

1. **Start with quick test:** Run Experiment 1 with minimal epochs
2. **Verify graph building:** Check the "Graph built" message in output
3. **Run full training:** Use full epoch count once verified
4. **Experiment with hyperparameters:** Try different ALPHA_SMOOTH and LAMBDA_LAP values
5. **Compare results:** Run multiple experiments and compare accuracy

---

## Output Files

After training, check `output/coop_graph/{dataset}/`:

```
├── log.txt                 # Training log
├── model-best.pth.tar      # Best model weights
├── model-final.pth.tar     # Final model weights
├── result.json             # Final accuracies
├── summaries/              # TensorBoard logs (if available)
```
