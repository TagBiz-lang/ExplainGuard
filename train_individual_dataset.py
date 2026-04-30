#!/usr/bin/env python3
"""
Individual Dataset Trainer for ExplainGuard
Generates dataset-specific pre-trained weights with proper class handling
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from src.core.models import TabularTransformer, CNNLSTMHybrid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleSecurityDataset(Dataset):
    """Lightweight dataset loader with proper label handling"""
    
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DatasetTrainer:
     
    def __init__(self, dataset_path, model_type='transformer', device='cpu', epochs=50, batch_size=64, lr=1e-3):
        """
        Initialize trainer
        
        Args:
            dataset_path: Path to CSV file
            model_type: 'transformer' or 'cnn_lstm'
            device: 'cpu' or 'cuda'
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        self.dataset_path = Path(dataset_path)
        self.model_type = model_type
        self.device = torch.device(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = None
        self.label_encoder = None
        self.scaler = None
        
        logger.info(f"Initializing trainer for {self.dataset_path.name}")
        logger.info(f"Model type: {self.model_type}")
        logger.info(f"Device: {self.device}")
    
    def load_and_prepare_data(self):
        """Load CSV and prepare features/labels with proper encoding"""
        logger.info(f"Loading dataset from {self.dataset_path}")
        
        df = pd.read_csv(self.dataset_path)
        logger.info(f"Dataset shape: {df.shape}")
        
        # Find label column
        label_col = None
        for candidate in ['label', 'Label', 'class', 'Class', 'Label', ' Label']:
            if candidate in df.columns:
                label_col = candidate
                break
        
        if label_col is None:
            raise ValueError(f"Could not find label column in {df.columns.tolist()}")
        
        logger.info(f"Using label column: '{label_col}'")
        logger.info(f"Unique labels (raw): {df[label_col].nunique()}")
        
        # Extract labels BEFORE encoding
        raw_labels = df[label_col].astype(str).values
        
        # Create label encoder and transform labels to 0-indexed
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(raw_labels)
        num_classes = len(self.label_encoder.classes_)
        
        logger.info(f"Label classes: {self.label_encoder.classes_}")
        logger.info(f"Encoded labels range: {labels.min()}-{labels.max()}")
        logger.info(f"Total classes: {num_classes}")
        
        # Extract features
        features = df.drop(columns=[label_col])
        features = features.select_dtypes(include=['number'])
        
        logger.info(f"Feature shape: {features.shape}")
        
        # Remove NaN/Inf
        valid_mask = ~(features.isna().any(axis=1) | np.isinf(features.values).any(axis=1))
        features = features[valid_mask].values
        labels = labels[valid_mask]
        
        logger.info(f"After NaN/Inf removal: features {features.shape}, labels {labels.shape}")
        
        # Normalize features
        self.scaler = StandardScaler()
        features = self.scaler.fit_transform(features)
        
        # Create train/val/test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            features, labels, test_size=0.4, random_state=42, stratify=labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        logger.info(f"Train/Val/Test split: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]}")
        logger.info(f"Class distribution in training: {np.bincount(y_train)}")
        
        return (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            features.shape[1],  # input_dim
            num_classes
        )
    
    def train(self):
        """Train model on dataset"""
        logger.info(f"Starting training on {self.dataset_path.name}")
        
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, input_dim, num_classes = \
            self.load_and_prepare_data()
        
        # Create datasets and loaders
        train_dataset = SimpleSecurityDataset(X_train, y_train)
        val_dataset = SimpleSecurityDataset(X_val, y_val)
        test_dataset = SimpleSecurityDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        if self.model_type == 'transformer':
            self.model = TabularTransformer(
                input_dim=input_dim,
                hidden_dim=256,
                num_heads=8,
                num_layers=4,
                num_classes=num_classes,
                dropout=0.1
            ).to(self.device)
        elif self.model_type == 'cnn_lstm':
            self.model = CNNLSTMHybrid(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=128,
                dropout=0.1
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Model created: {self.model.__class__.__name__}")
        logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Compute class weights for imbalanced data
        class_counts = np.bincount(y_train, minlength=num_classes)
        class_weights = torch.tensor(
            1.0 / (class_counts + 1e-6),  # Add small epsilon to avoid division by zero
            dtype=torch.float32
        )
        class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
        class_weights = class_weights.to(self.device)
        
        logger.info(f"Class weights: {class_weights.tolist()}")
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        # Training loop
        best_val_acc = 0
        patience = 15
        patience_counter = 0
        history = {'train_loss': [], 'val_acc': [], 'val_loss': []}
        
        logger.info(f"Training for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs} (train)", leave=False)
            for batch_x, batch_y in pbar:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss /= len(train_loader)
            history['train_loss'].append(float(train_loss))
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    
                    val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    val_correct += (preds == batch_y).sum().item()
                    val_total += batch_y.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            history['val_acc'].append(float(val_acc))
            history['val_loss'].append(float(val_loss))
            
            logger.info(f"Epoch {epoch+1}/{self.epochs} - Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model with model type in filename
                model_path = Config.PRETRAINED_MODEL_DIR / f"{self.model_type}_{self.dataset_path.stem}_best.pt"
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"✓ Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1} (patience {patience} reached)")
                    break
        
        # Test phase
        logger.info("Evaluating on test set...")
        self.model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                logits = self.model(batch_x)
                preds = torch.argmax(logits, dim=1)
                
                test_correct += (preds == batch_y).sum().item()
                test_total += batch_y.size(0)
        
        test_acc = test_correct / test_total
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        # Save training history
        history_path = Config.PRETRAINED_MODEL_DIR / f"{self.model_type}_{self.dataset_path.stem}_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"✓ Saved training history to {history_path}")
        
        return {
            'dataset': self.dataset_path.stem,
            'model_type': self.model_type,
            'best_val_acc': float(best_val_acc),
            'test_acc': float(test_acc),
            'epochs_trained': epoch + 1,
            'model_path': str(Config.PRETRAINED_MODEL_DIR / f"{self.model_type}_{self.dataset_path.stem}_best.pt"),
            'input_dim': input_dim,
            'num_classes': num_classes,
            'timestamp': datetime.now().isoformat(),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Train individual dataset model and generate pre-trained weights"
    )
    parser.add_argument('dataset', help='Path to CSV dataset file')
    parser.add_argument('--model-type', choices=['transformer', 'cnn_lstm'], default='transformer',
                        help='Model architecture to train (default: transformer)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    # Verify dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)
    
    # Verify output directory exists
    Config.PRETRAINED_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Train
    trainer = DatasetTrainer(
        dataset_path=dataset_path,
        model_type=args.model_type,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
    
    result = trainer.train()
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY")
    logger.info("="*80)
    logger.info(json.dumps(result, indent=2))
    logger.info("="*80)
    
    return result


if __name__ == '__main__':
    main()
