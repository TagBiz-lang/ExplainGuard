#!/usr/bin/env python3
"""
Multi-Model Dataset Trainer for ExplainGuard
Trains both Transformer and CNN-LSTM models for each dataset
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiModelTrainer:
    """Train both Transformer and CNN-LSTM models for datasets"""
    
    def __init__(self, dataset_dir=None, epochs=50, batch_size=64, device='cpu'):
        """
        Initialize multi-model trainer
        
        Args:
            dataset_dir: Directory containing CSV files (if None, uses data/raw)
            epochs: Number of training epochs per model
            batch_size: Batch size for training
            device: 'cpu' or 'cuda'
        """
        self.script_dir = Path(__file__).parent
        self.dataset_dir = Path(dataset_dir) if dataset_dir else self.script_dir / 'data' / 'raw'
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.results = {}
        
        logger.info(f"Multi-Model Trainer initialized")
        logger.info(f"Dataset directory: {self.dataset_dir}")
        logger.info(f"Epochs: {self.epochs}, Batch size: {self.batch_size}, Device: {self.device}")
    
    def discover_datasets(self):
        """Find all CSV files in dataset directory"""
        if not self.dataset_dir.exists():
            logger.error(f"Dataset directory not found: {self.dataset_dir}")
            return []
        
        csv_files = sorted(self.dataset_dir.glob('*.csv'))
        logger.info(f"Discovered {len(csv_files)} datasets:")
        for csv_file in csv_files:
            logger.info(f"  - {csv_file.name}")
        
        return csv_files
    
    def train_model(self, dataset_path, model_type):
        """Train a single model using train_individual_dataset.py"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Training {model_type.upper()} model for {dataset_path.stem}")
        logger.info(f"{'='*80}")
        
        cmd = [
            'python', str(self.script_dir / 'train_individual_dataset.py'),
            str(dataset_path),
            '--model-type', model_type,
            '--epochs', str(self.epochs),
            '--batch-size', str(self.batch_size),
            '--device', self.device,
            '--lr', '1e-3'
        ]
        
        logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"✓ {model_type.upper()} training completed successfully")
                return True
            else:
                logger.error(f"✗ {model_type.upper()} training failed with exit code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output:\n{result.stderr[-500:]}")
                return False
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {model_type.upper()} training timed out (>1 hour)")
            return False
        except Exception as e:
            logger.error(f"✗ Exception during {model_type.upper()} training: {e}")
            return False
    
    def train_all_datasets(self):
        """Train both models for all discovered datasets"""
        datasets = self.discover_datasets()
        
        if not datasets:
            logger.warning("No datasets found!")
            return
        
        total_datasets = len(datasets)
        models_per_dataset = 2  # transformer + cnn_lstm
        total_models = total_datasets * models_per_dataset
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING PLAN: {total_datasets} datasets × 2 models = {total_models} total models")
        logger.info(f"{'='*80}\n")
        
        model_count = 0
        success_count = 0
        
        for dataset_idx, dataset_path in enumerate(datasets, 1):
            logger.info(f"\n[Dataset {dataset_idx}/{total_datasets}] {dataset_path.stem}")
            
            for model_type in ['transformer', 'cnn_lstm']:
                model_count += 1
                logger.info(f"  Model {model_count}/{total_models}: {model_type}")
                
                if self.train_model(dataset_path, model_type):
                    success_count += 1
                    if dataset_path.stem not in self.results:
                        self.results[dataset_path.stem] = {}
                    self.results[dataset_path.stem][model_type] = 'SUCCESS'
                else:
                    if dataset_path.stem not in self.results:
                        self.results[dataset_path.stem] = {}
                    self.results[dataset_path.stem][model_type] = 'FAILED'
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"TRAINING SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total models trained: {model_count}")
        logger.info(f"Successful: {success_count}")
        logger.info(f"Failed: {model_count - success_count}")
        logger.info(f"Success rate: {100*success_count/model_count:.1f}%")
        
        # Per-dataset summary
        logger.info(f"\nPer-Dataset Results:")
        for dataset_name, models in sorted(self.results.items()):
            logger.info(f"  {dataset_name}:")
            for model_type, status in models.items():
                status_icon = "✓" if status == "SUCCESS" else "✗"
                logger.info(f"    {status_icon} {model_type}: {status}")
        
        logger.info(f"{'='*80}")
        
        return success_count, model_count
    
    def train_single_dataset(self, dataset_path):
        """Train both models for a single dataset"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return False
        
        logger.info(f"\n{'='*80}")
        logger.info(f"SINGLE DATASET: Training both models for {dataset_path.stem}")
        logger.info(f"{'='*80}\n")
        
        success = True
        for model_type in ['transformer', 'cnn_lstm']:
            if not self.train_model(dataset_path, model_type):
                success = False
        
        return success


def main():
    parser = argparse.ArgumentParser(
        description="Train both Transformer and CNN-LSTM models for all or specific datasets"
    )
    parser.add_argument('--dataset', help='Path to single CSV dataset (if omitted, trains all in data/raw)')
    parser.add_argument('--dataset-dir', help='Directory containing CSV files (default: data/raw)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs per model (default: 50)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use (default: cpu)')
    
    args = parser.parse_args()
    
    trainer = MultiModelTrainer(
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if args.dataset:
        # Train single dataset with both models
        trainer.train_single_dataset(args.dataset)
    else:
        # Train all datasets with both models
        trainer.train_all_datasets()


if __name__ == '__main__':
    main()
