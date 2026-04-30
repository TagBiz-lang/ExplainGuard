#!/usr/bin/env python3
"""
Robust Training Script for CNN-LSTM and Transformer Models
Fixes label remapping issue for proper multi-class training
"""

import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import Config
from src.core.models import TabularTransformer, CNNLSTMHybrid
from src.utils.dataset_loader import SecurityDataset
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance without explicit class weights"""

    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Focal Loss: down-weights easy examples
        """
        ce_loss = nn.CrossEntropyLoss()(inputs, targets)
        p = torch.exp(-ce_loss)
        focal_loss = ((1 - p) ** self.gamma) * ce_loss
        return focal_loss.mean()


class ModelTrainer:
    """Robust trainer with proper label handling"""

    def __init__(
        self,
        model_type="transformer",
        device="cuda",
        learning_rate=1e-4,
        batch_size=64,
        epochs=100,
        patience=15,
        save_dir=None,
        random_seed=42,
    ):
        self.model_type = model_type
        self.device = device
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.save_dir = Path(save_dir or Config.PRETRAINED_MODEL_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed

        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if device == "cuda":
            torch.cuda.manual_seed(random_seed)

        self.best_val_metric = -np.inf
        self.patience_counter = 0
        self.train_history = []
        self.val_history = []
        self.best_model_path = None

    def create_model(self, input_dim, num_classes):
        """Create appropriate model"""
        if self.model_type == "transformer":
            model = TabularTransformer(
                input_dim=input_dim,
                hidden_dim=256,
                num_heads=8,
                num_layers=4,
                num_classes=num_classes,
                dropout=0.1,
            )
        elif self.model_type == "cnn_lstm":
            model = CNNLSTMHybrid(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=128,
                dropout=0.1,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model.to(self.device)

    def load_and_prepare_data(self, csv_path):
        """Load dataset and create dataloaders with proper label remapping"""
        logger.info("Loading dataset...")
        
        # Load raw dataset
        dataset = SecurityDataset(csv_path=csv_path)
        
        # Get labels and remap them to 0-num_classes
        labels = dataset.labels.numpy()
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Create remapping
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        remapped_labels = np.array([label_map[l] for l in labels])
        
        # Get features
        features = dataset.features.numpy()
        
        logger.info(f"  Original labels: {unique_labels}")
        logger.info(f"  Remapped to: {np.unique(remapped_labels)}")
        logger.info(f"  Num classes: {num_classes}")
        
        # Create train/val/test split
        n = len(features)
        perm = np.random.permutation(n)
        
        test_size = int(0.2 * n)
        val_size = int(0.1 * n)
        train_size = n - val_size - test_size
        
        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size + val_size]
        test_idx = perm[train_size + val_size:]
        
        # Create datasets
        train_features = torch.tensor(features[train_idx], dtype=torch.float32)
        train_labels = torch.tensor(remapped_labels[train_idx], dtype=torch.long)
        
        val_features = torch.tensor(features[val_idx], dtype=torch.float32)
        val_labels = torch.tensor(remapped_labels[val_idx], dtype=torch.long)
        
        test_features = torch.tensor(features[test_idx], dtype=torch.float32)
        test_labels = torch.tensor(remapped_labels[test_idx], dtype=torch.long)
        
        # Create loaders
        train_loader = DataLoader(
            TensorDataset(train_features, train_labels),
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            TensorDataset(val_features, val_labels),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        test_loader = DataLoader(
            TensorDataset(test_features, test_labels),
            batch_size=self.batch_size,
            shuffle=False
        )
        
        logger.info(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}, Test samples: {len(test_idx)}")
        
        return train_loader, val_loader, test_loader, features.shape[1], num_classes

    def train_epoch(self, model, train_loader, optimizer, criterion):
        """Train one epoch"""
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for X, y in pbar:
            X, y = X.to(self.device), y.to(self.device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().detach().numpy())
            all_labels.extend(y.cpu().detach().numpy())

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        avg_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return avg_loss, avg_acc

    def validate(self, model, val_loader):
        """Validate"""
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0

        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validating", leave=False)
            for X, y in pbar:
                X, y = X.to(self.device), y.to(self.device)

                logits = model(X)
                loss = nn.CrossEntropyLoss()(logits, y)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        accuracy = np.mean(all_preds == all_labels)
        balanced_acc = balanced_accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
        mcc = matthews_corrcoef(all_labels, all_preds)

        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "f1": f1,
            "mcc": mcc,
        }

    def train(self, csv_path, dataset_name=None):
        """Train model"""
        logger.info("=" * 80)
        logger.info(f"TRAINING {self.model_type.upper()} ON {dataset_name or Path(csv_path).stem}")
        logger.info("=" * 80)

        # Load and prepare data
        train_loader, val_loader, test_loader, input_dim, num_classes = self.load_and_prepare_data(csv_path)

        # Create model
        logger.info("Creating model...")
        model = self.create_model(input_dim, num_classes)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Setup training
        optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = FocalLoss(gamma=2.0)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5
        )

        # Training loop
        logger.info("\nStarting training...")
        logger.info(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} "
                   f"{'Val Acc':<12} {'Val Bal.Acc':<12} {'Val F1':<12}")
        logger.info("-" * 80)

        for epoch in range(self.epochs):
            # Train
            train_loss, train_acc = self.train_epoch(model, train_loader, optimizer, criterion)

            # Validate
            val_metrics = self.validate(model, val_loader)

            # Log
            logger.info(
                f"{epoch+1:<6} {train_loss:<12.4f} {train_acc:<12.4f} "
                f"{val_metrics['accuracy']:<12.4f} {val_metrics['balanced_accuracy']:<12.4f} "
                f"{val_metrics['f1']:<12.4f}"
            )

            self.train_history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc})
            self.val_history.append({"epoch": epoch + 1, **val_metrics})

            # Scheduler
            scheduler.step(val_metrics["balanced_accuracy"])

            # Early stopping
            if val_metrics["balanced_accuracy"] > self.best_val_metric:
                self.best_val_metric = val_metrics["balanced_accuracy"]
                self.patience_counter = 0

                # Save best model with dataset-specific name
                dataset_suffix = Path(csv_path).stem.replace('-', '_').lower()
                model_path = self.save_dir / f"{self.model_type}_{dataset_suffix}_best.pt"
                torch.save(model.state_dict(), model_path)
                self.best_model_path = model_path
                logger.info(f"  ✓ New best model saved to {model_path.name} (val_bal_acc: {self.best_val_metric:.4f})")

            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # Test set evaluation
        logger.info("\n" + "=" * 80)
        logger.info("FINAL EVALUATION ON TEST SET")
        logger.info("=" * 80)

        # Load best model
        if hasattr(self, 'best_model_path') and self.best_model_path.exists():
            model.load_state_dict(torch.load(self.best_model_path))
            logger.info(f"Loaded best model from {self.best_model_path.name}")

        test_metrics = self.validate(model, test_loader)

        logger.info("\nTest Results:")
        logger.info(f"  Accuracy:          {test_metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        logger.info(f"  F1-Score:          {test_metrics['f1']:.4f}")
        logger.info(f"  MCC:               {test_metrics['mcc']:.4f}")

        # Save history
        history_path = self.save_dir / f"{self.model_type}_{Path(csv_path).stem}_history.json"
        with open(history_path, "w") as f:
            json.dump({
                "train_history": self.train_history,
                "val_history": self.val_history,
                "test_metrics": test_metrics,
                "dataset": Path(csv_path).stem,
                "model_type": self.model_type,
            }, f, indent=2)

        logger.info(f"Training history saved to {history_path}")
        logger.info("=" * 80 + "\n")

        return model, test_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train models with robust label handling")

    parser.add_argument("--model", choices=["transformer", "cnn_lstm"], default="transformer")
    parser.add_argument("--dataset", type=str, help="Dataset CSV path")
    parser.add_argument("--train_all", action="store_true", help="Train on all datasets")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ROBUST MODEL TRAINING (WITH PROPER LABEL HANDLING)")
    logger.info("=" * 80 + "\n")

    trainer = ModelTrainer(
        model_type=args.model,
        device=args.device,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        random_seed=args.seed,
    )

    if args.train_all:
        dataset_dir = Config.RAW_DATA_DIR
        datasets = sorted(dataset_dir.glob("*.csv"))
        for dataset_path in datasets:
            model, metrics = trainer.train(str(dataset_path), dataset_name=dataset_path.stem)
    else:
        if not args.dataset:
            parser.error("Either --dataset or --train_all must be specified")
        model, metrics = trainer.train(args.dataset, dataset_name=Path(args.dataset).stem)

    logger.info("Training completed!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)
