# Quick Reference - Multi-Model Training Commands

## Essential Commands

### 1. Train Single Model Type
```bash
# Train Transformer (best accuracy)
python train_individual_dataset.py data/raw/cicddos2019_dataset.csv \
  --model-type transformer

# Train CNN-LSTM (fastest, smallest)
python train_individual_dataset.py data/raw/cicddos2019_dataset.csv \
  --model-type cnn_lstm
```

### 2. Train Both Models
```bash
# For one dataset
python train_all_models.py --dataset data/raw/cicddos2019_dataset.csv

# For all datasets in data/raw
python train_all_models.py
```

### 3. Check Available Models
```bash
# List all trained models
ls -lh models/pretrained/*_best.pt

# Count by type
ls models/pretrained/transformer_*_best.pt | wc -l  # Transformers
ls models/pretrained/cnn_lstm_*_best.pt | wc -l      # CNN-LSTMs
```

### 4. Run Evaluation (Automatic Model Selection)
```bash
# Will automatically use trained models if available
python scripts/evaluate_multi_dataset.py
```

---

## Model Naming Convention

### Transformer
- Pattern: `transformer_{DATASET_STEM}_best.pt`
- Examples:
  - `transformer_cicddos2019_dataset_best.pt`
  - `transformer_workinghours_merged_best.pt`
  - `transformer_bcc_cpacket_cloud_ddos_best.pt`

### CNN-LSTM
- Pattern: `cnn_lstm_{DATASET_STEM}_best.pt`
- Examples:
  - `cnn_lstm_cicddos2019_dataset_best.pt`
  - `cnn_lstm_workinghours_merged_best.pt`
  - `cnn_lstm_bcc_cpacket_cloud_ddos_best.pt`

---

## Common Use Cases

### Use Case 1: Maximum Accuracy (Research)
```bash
# Train Transformer model
python train_individual_dataset.py data/raw/cicddos2019_dataset.csv \
  --model-type transformer --epochs 30

# Result: 97.67% accuracy on DDoS2019
```

### Use Case 2: Edge Deployment (Small & Fast)
```bash
# Train CNN-LSTM model
python train_individual_dataset.py data/raw/cicddos2019_dataset.csv \
  --model-type cnn_lstm --epochs 30

# Result: 1.2MB model, 10-20x faster inference
```

### Use Case 3: Complete Comparison
```bash
# Train both models for all datasets
python train_all_models.py --epochs 30 --batch-size 128

# Generates:
# - 8 total models (2 types × 4 datasets)
# - Training histories for each
# - Ready for comparative evaluation
```

### Use Case 4: Quick Testing (Small Batch)
```bash
# Train with larger batch to reduce iterations
python train_individual_dataset.py data/raw/cicddos2019_dataset.csv \
  --model-type transformer --batch-size 256 --epochs 10

# Result: Faster training, reasonable accuracy
```

### Use Case 5: GPU Acceleration (If Available)
```bash
# Train on GPU (if CUDA available)
python train_individual_dataset.py data/raw/cicddos2019_dataset.csv \
  --model-type transformer --device cuda

# Result: 3-5x faster training
```

---

## Parameter Tuning

### Training Speed Trade-offs
```bash
# FASTEST (use for testing)
--batch-size 256 --epochs 5 --device cuda

# BALANCED
--batch-size 128 --epochs 30 --device cpu

# MOST THOROUGH
--batch-size 64 --epochs 50 --device cuda
```

### Batch Size Guidelines
```bash
# 32     - Maximum accuracy, slow
# 64     - Good balance, recommended
# 128    - Faster, minimal accuracy drop
# 256    - For quick testing
# 512+   - Speed focused
```

---

## File Locations

### Training Scripts
```
/home/Tawa/ExplainGuardAI/explainguard_clean/
├── train_individual_dataset.py      Single model trainer
└── train_all_models.py              Multi-model orchestrator
```

### Datasets
```
/home/Tawa/ExplainGuardAI/explainguard_clean/data/raw/
├── cicddos2019_dataset.csv          431K samples, 18 classes
├── workinghours_merged.csv          1.8M samples, 11 classes
├── BCC-Cpacket-Cloud-DDoS.csv       Large dataset
└── CIC-IoT2023-dataset.csv          IoT security data
```

### Model Outputs
```
/home/Tawa/ExplainGuardAI/explainguard_clean/models/pretrained/
├── transformer_*.pt                 13MB each
├── transformer_*_history.json       Training logs
├── cnn_lstm_*.pt                    1.2MB each
└── cnn_lstm_*_history.json          Training logs
```

---

## Current Status

### Already Trained 
- transformer_cicddos2019_dataset_best.pt (97.67%)
- transformer_workinghours_merged_best.pt
- transformer_bcc_cpacket_cloud_ddos_best.pt
- transformer_cic_iot2023_dataset_best.pt
- cnn_lstm_bcc_cpacket_cloud_ddos_best.pt
- cnn_lstm_cic_iot2023_dataset_best.pt

### In Progress 
- cnn_lstm_cicddos2019_dataset_best.pt (started 00:23:56)

### Queued 
- cnn_lstm_workinghours_merged_best.pt

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Model not found | Check naming: `{model}_{dataset}_best.pt` |
| Training too slow | Increase `--batch-size` or reduce `--epochs` |
| Out of memory | Switch to `cnn_lstm` (90% smaller) |
| CUDA errors | Use `--device cpu` instead |
| No label column found | Ensure CSV has 'Label' or ' Label' column |
| Models not used in eval | Check `models/pretrained/` exists |

---

## Performance Comparison

| Metric | Transformer | CNN-LSTM |
|--------|-------------|----------|
| Accuracy | 97.67% | ~94% |
| Model Size | 13M | 1.2M |
| Speed | Baseline | 10-20x faster |
| Parameters | 3.2M | 300K |
| Best For | Research, Accuracy | Production, Edge |

---

## Documentation Files

| File | Purpose |
|------|---------|
| MULTI_MODEL_TRAINING_GUIDE.md | Complete user guide |
| MULTI_MODEL_IMPLEMENTATION.md | Technical details |
| README_MULTIMODEL.md | This comprehensive summary |
| TRAINING_QUICK_REFERENCE.md | Quick commands (this file) |

---

## Example Workflow

### Step 1: Train Both Models
```bash
python train_all_models.py --epochs 30
```

### Step 2: Check Results
```bash
ls -lh models/pretrained/*_best.pt
```

### Step 3: Run Evaluation
```bash
python scripts/evaluate_multi_dataset.py
```

### Step 4: Compare Results
```bash
# Check generated reports
ls output/reports/
```

---

## Monitor Training

### Check if Training is Running
```bash
ps aux | grep train_individual
```

### View Training Progress
```bash
# Real-time logs
tail -f nohup.out

# Training history (after completion)
cat models/pretrained/cnn_lstm_*.json | head -20
```

### Get Training Summary
```bash
# After training, check accuracy
python -c "
import json
with open('models/pretrained/cnn_lstm_cicddos2019_dataset_history.json') as f:
    h = json.load(f)
    print(f'Best validation accuracy: {max(h[\"val_acc\"]):.4f}')
"
```

---

## Advanced Options

### Custom Dataset Directory
```bash
python train_all_models.py --dataset-dir /path/to/datasets
```

### Specific Learning Rate
```bash
python train_individual_dataset.py data.csv --lr 5e-4
```

### Single Dataset, All Models
```bash
# Transformer
python train_individual_dataset.py data.csv --model-type transformer

# CNN-LSTM
python train_individual_dataset.py data.csv --model-type cnn_lstm
```

---

## What Happens When You Run Commands

### train_individual_dataset.py
1. Loads dataset from CSV
2. Detects label column
3. Encodes labels (0-indexed)
4. Splits data (60/20/20)
5. Creates model based on `--model-type`
6. Trains with early stopping
7. Saves best model as `{model}_{dataset}_best.pt`
8. Saves training history as JSON
9. Prints summary with accuracy

### train_all_models.py
1. Discovers all CSV files in directory
2. For each dataset:
   - Trains Transformer model
   - Trains CNN-LSTM model
3. Logs progress and results
4. Prints per-dataset and aggregate summaries

### scripts/evaluate_multi_dataset.py
1. Discovers datasets
2. For each dataset:
   - Looks for trained models automatically
   - Uses Transformer if available
   - Falls back to CNN-LSTM if needed
   - Generates 600 security certificates
   - Tests RQ1, RQ2, RQ3 metrics
3. Saves detailed reports

---

## Key Differences from Original

### Before
- Only Transformer model training
- Single `train_all_datasets.py` with complex logic
- Limited flexibility

### After
- Transformer AND CNN-LSTM support 
- Simplified `train_individual_dataset.py` with model selection 
- New `train_all_models.py` for orchestration 
- Better file naming and organization 

---

## Tips & Tricks

### Save Time
```bash
# Use larger batch size for faster training
--batch-size 256

# Reduce epochs for testing
--epochs 10
```

### Better Accuracy
```bash
# Use smaller batch size
--batch-size 32

# More epochs
--epochs 50

# Use GPU if available
--device cuda
```

### Monitor Both Models
```bash
# Terminal 1: Train Transformer
python train_individual_dataset.py data.csv --model-type transformer &

# Terminal 2: Train CNN-LSTM
python train_individual_dataset.py data.csv --model-type cnn_lstm &

# Terminal 3: Monitor
watch ls -lh models/pretrained/*_best.pt
```

---

**Version**: 1.0
**Last Updated**: 2026-04-27
**Status**: Ready for production use 
