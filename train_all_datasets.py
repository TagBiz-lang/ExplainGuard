#!/usr/bin/env python3
"""
Comprehensive multi-dataset model trainer.
Trains dataset-specific models for all available datasets.
Handles class imbalance with weighted loss and focal loss.
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
try:
    from sklearn.class_weight import compute_class_weight
except ImportError:
    # Fallback if sklearn is not installed
    def compute_class_weight(strategy, classes, y):
        """Fallback class weight computation"""
        if strategy == "balanced":
            unique, counts = np.unique(y, return_counts=True)
            weights = len(y) / (len(unique) * counts)
            return weights
        return np.ones(len(np.unique(y)))

from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.core.models import TabularTransformer
from src.utils.dataset_loader import SecurityDataset
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        # Ensure alpha is on the same device as inputs
        alpha = self.alpha
        if alpha is not None and alpha.device != inputs.device:
            alpha = alpha.to(inputs.device)
        
        ce_loss = nn.CrossEntropyLoss(weight=alpha)(inputs, targets)
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma) * ce_loss
        return focal_loss.mean()


class DatasetSpecificTrainer:
    """Train models optimized for each dataset"""

    def __init__(
        self,
        device="cpu",
        learning_rate=1e-3,
        epochs=150,
        patience=20,
        batch_size=64,
        random_seed=42,
    ):
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.save_dir = Config.PRETRAINED_MODEL_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(random_seed)

    def prepare_dataset(self, csv_path, dataset_name):
        """Load and prepare dataset with proper class weighting"""
        logger.info(f"Loading dataset: {dataset_name}")
        
        dataset = SecurityDataset(csv_path, domain=dataset_name.replace("_", "-").replace("-dataset", ""))
        
        # Get all unique classes in the full dataset
        all_unique_classes = torch.unique(dataset.labels).sort()[0].numpy()
        num_classes = len(all_unique_classes)
        
        # Create dataloaders
        n_samples = len(dataset)
        train_idx = int(0.6 * n_samples)
        val_idx = int(0.8 * n_samples)
        
        train_set = torch.utils.data.Subset(dataset, range(train_idx))
        val_set = torch.utils.data.Subset(dataset, range(train_idx, val_idx))
        test_set = torch.utils.data.Subset(dataset, range(val_idx, n_samples))
        
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        # Compute class weights for ALL classes (including those not in training set)
        train_labels = dataset.labels[:train_idx].numpy()
        
        try:
            # Get weights for all classes (including those with 0 samples in training)
            class_weights = np.zeros(num_classes)
            for idx, class_val in enumerate(all_unique_classes):
                mask = (train_labels == class_val)
                if mask.sum() > 0:
                    # Weight for this class
                    class_weight = len(train_labels) / (num_classes * mask.sum())
                    class_weights[idx] = class_weight
                else:
                    # Class not in training set - use a default weight
                    class_weights[idx] = 1.0
        except Exception as e:
            logger.warning(f"Error computing class weights: {e}. Using uniform weights.")
            class_weights = np.ones(num_classes)
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        if self.device.type == 'cuda':
            try:
                class_weights = class_weights.to(self.device)
            except Exception as e:
                logger.warning(f"Failed to move class_weights to CUDA: {e}. Using CPU.")
                self.device = torch.device('cpu')
        
        logger.info(f"  Samples: {n_samples} (train: {len(train_set)}, val: {len(val_set)}, test: {len(test_set)})")
        logger.info(f"  Features: {dataset.features.shape[1]}")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Class weights shape: {class_weights.shape}")
        logger.info(f"  Class weights: {class_weights.tolist()[:5]}..." if len(class_weights) > 5 else f"  Class weights: {class_weights.tolist()}")
        
        return train_loader, val_loader, test_loader, num_classes, dataset.features.shape[1], class_weights

    def create_model(self, input_dim, num_classes):
        """Create model architecture optimized for the task"""
        model = TabularTransformer(
            input_dim=input_dim,
            hidden_dim=256,
            num_heads=8,
            num_layers=4,
            num_classes=num_classes,
            dropout=0.1,
        )
        return model.to(self.device)

    def train_epoch(self, model, train_loader, optimizer, loss_fn, scaler=None):
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(train_loader)

    def validate(self, model, val_loader):
        """Validate model"""
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return acc, f1

    def train_dataset(self, csv_path, dataset_name):
        """Train model for specific dataset"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Training model for dataset: {dataset_name}")
        logger.info(f"{'='*80}")
        
        # Prepare data
        train_loader, val_loader, test_loader, num_classes, input_dim, class_weights = self.prepare_dataset(
            csv_path, dataset_name
        )
        
        # Create model
        model = self.create_model(input_dim, num_classes)
        logger.info(f"Model created: {model.__class__.__name__}")
        
        # Setup training
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        loss_fn = FocalLoss(gamma=2.0, alpha=class_weights)
        
        best_val_f1 = -np.inf
        patience_counter = 0
        history = {"train_loss": [], "val_acc": [], "val_f1": []}
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(self.epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, loss_fn)
            val_acc, val_f1 = self.validate(model, val_loader)
            
            history["train_loss"].append(float(train_loss))
            history["val_acc"].append(float(val_acc))
            history["val_f1"].append(float(val_f1))
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                
                # Save best model
                model_name = f"transformer_{dataset_name.lower().replace('-', '_')}_best.pt"
                model_path = self.save_dir / model_name
                torch.save(model.state_dict(), model_path)
                logger.info(f"  Saved best model to {model_name}")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                
                y_true.extend(batch_y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        test_mcc = matthews_corrcoef(y_true, y_pred)
        
        logger.info(f"Test Results:")
        logger.info(f"  Accuracy: {test_acc:.4f}")
        logger.info(f"  F1-Score: {test_f1:.4f}")
        logger.info(f"  MCC: {test_mcc:.4f}")
        
        # Save history
        history_name = f"transformer_{dataset_name}_history.json"
        history_path = self.save_dir / history_name
        history["test_accuracy"] = float(test_acc)
        history["test_f1"] = float(test_f1)
        history["test_mcc"] = float(test_mcc)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f" Training completed for {dataset_name}\n")
        
        return {
            "dataset": dataset_name,
            "test_accuracy": test_acc,
            "test_f1": test_f1,
            "test_mcc": test_mcc,
            "model_path": str(model_path),
            "history_path": str(history_path),
        }

    def train_all(self, dataset_dir=None):
        """Train models for all available datasets"""
        dataset_dir = Path(dataset_dir or Config.RAW_DATA_DIR)
        
        if not dataset_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_dir}")
            return {}
        
        csv_files = sorted(dataset_dir.glob("*.csv"))
        logger.info(f"Found {len(csv_files)} datasets")
        
        results = {}
        for csv_file in csv_files:
            dataset_name = csv_file.stem
            try:
                result = self.train_dataset(str(csv_file), dataset_name)
                results[dataset_name] = result
            except Exception as e:
                logger.error(f"Failed to train on {dataset_name}: {e}", exc_info=True)
                results[dataset_name] = {"error": str(e)}
        
        # Print summary
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        for dataset, result in results.items():
            if "error" in result:
                logger.info(f" {dataset}: {result['error']}")
            else:
                logger.info(f" {dataset}: Acc={result['test_accuracy']:.4f}, F1={result['test_f1']:.4f}")
        
        # Save summary
        summary_path = self.save_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSummary saved to {summary_path}")
        
        return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train models for all datasets")
    parser.add_argument("--dataset_dir", type=str, help="Path to datasets directory")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    args = parser.parse_args()
    
    trainer = DatasetSpecificTrainer(
        device=args.device,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=args.patience,
    )
    
    trainer.train_all(dataset_dir=args.dataset_dir)


if __name__ == "__main__":
    main()
