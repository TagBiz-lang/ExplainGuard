#!/bin/bash
# Quick Setup & Training Script for ExplainGuard

set -e

echo "===================================="
echo "ExplainGuard Setup & Training"
echo "===================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Environment setup
echo -e "${YELLOW}Step 1: Setting up environment...${NC}"
pip install -q torch scikit-learn numpy pandas tqdm -q 2>/dev/null || echo "Dependencies already installed"
echo -e "${GREEN}✓ Environment ready${NC}\n"

# Step 2: Initialize directories
echo -e "${YELLOW}Step 2: Initializing directories...${NC}"
python scripts/setup.py > /dev/null 2>&1 || echo "Directories already initialized"
echo -e "${GREEN}✓ Directories initialized${NC}\n"

# Step 3: Check datasets
echo -e "${YELLOW}Step 3: Checking datasets...${NC}"
num_datasets=$(find data/raw -name "*.csv" 2>/dev/null | wc -l)
if [ $num_datasets -eq 0 ]; then
    echo -e "${YELLOW}⚠ No datasets found in data/raw/${NC}"
    echo "  Please add CSV files to data/raw/"
    exit 1
fi
echo -e "${GREEN}✓ Found $num_datasets datasets${NC}\n"

# Step 4: Train models
echo -e "${YELLOW}Step 4: Training dataset-specific models...${NC}"
echo "  This will take 10-30 minutes depending on your hardware"
echo ""
python train_all_datasets.py \
    --device cuda \
    --epochs 150 \
    --patience 20 \
    --lr 1e-3

echo ""
echo -e "${GREEN}✓ Model training complete${NC}\n"

# Step 5: Evaluate models
echo -e "${YELLOW}Step 5: Evaluating models on all datasets...${NC}"
python scripts/evaluate_multi_dataset.py --output output/reports/evaluation_after_training.json

echo ""
echo -e "${GREEN}✓ Evaluation complete${NC}\n"

# Step 6: Display results
echo "===================================="
echo "Setup & Training Complete!"
echo "===================================="
echo ""
echo "Results:"
echo "  Models: models/pretrained/transformer_*_best.pt"
echo "  Summary: models/pretrained/training_summary.json"
echo "  Evaluation: output/reports/evaluation_after_training.json"
echo "  Logs: output/logs/explainguard.log"
echo ""
echo "Next steps:"
echo "  1. Check evaluation results"
echo "  2. Verify model accuracy has improved"
echo "  3. Generate certificates: python scripts/evaluate.py <dataset>"
echo ""
