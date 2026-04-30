#!/usr/bin/env python3
"""
Proper evaluation of trained models using test sets from training.
Loads models with correct architectures and evaluates on multi-class datasets.
"""

import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix,
)

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.core.models import TabularTransformer, CNNLSTMHybrid
from src.utils.dataset_loader import SecurityDataset
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


class RobustModelEvaluator:
    """Evaluate trained models on proper test sets"""

    def __init__(self, device="cuda", random_seed=42):
        self.device = device
        self.random_seed = random_seed
        self.results = {}
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def find_training_history(self, dataset_name, model_type):
        """Find the training history JSON for a dataset"""
        history_dir = Config.PRETRAINED_MODEL_DIR
        
        # Try different naming conventions
        candidates = [
            f"{model_type}_{dataset_name}_history.json",
            f"{model_type.lower()}_{dataset_name}_history.json",
        ]
        
        for candidate in candidates:
            path = history_dir / candidate
            if path.exists():
                return path
        
        return None

    def find_model_file(self, dataset_name, model_type):
        """Find the trained model file for a dataset"""
        model_dir = Config.PRETRAINED_MODEL_DIR
        
        # Normalize dataset name: lowercase and replace hyphens with underscores
        normalized_name = dataset_name.lower().replace("-", "_").replace(".", "_")
        
        # Try different naming conventions
        candidates = [
            f"{model_type}_{normalized_name}_best.pt",
            f"{model_type.lower()}_{normalized_name}_best.pt",
            f"{model_type}_{dataset_name}_best.pt",
            f"{model_type.lower()}_{dataset_name}_best.pt",
        ]
        
        for candidate in candidates:
            path = model_dir / candidate
            if path.exists():
                logger.info(f"Found model: {candidate}")
                return path
        
        logger.warning(f"Could not find model for {model_type} on {dataset_name}. Tried: {candidates[0]}")
        return None

    def load_and_prepare_data(self, csv_path):
        """Load dataset and create train/val/test split (same as training)"""
        logger.info(f"Loading dataset: {csv_path}")
        
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
        
        logger.info(f"  Samples: {len(features)}")
        logger.info(f"  Features: {features.shape[1]}")
        logger.info(f"  Original labels: {sorted(unique_labels)}")
        logger.info(f"  Num classes: {num_classes}")
        logger.info(f"  Class distribution: {np.bincount(remapped_labels)}")
        
        # Create train/val/test split (same as training)
        n = len(features)
        perm = np.random.permutation(n)
        
        test_size = int(0.2 * n)
        val_size = int(0.1 * n)
        train_size = n - val_size - test_size
        
        train_idx = perm[:train_size]
        val_idx = perm[train_size:train_size + val_size]
        test_idx = perm[train_size + val_size:]
        
        logger.info(f"  Train set: {len(train_idx)} samples")
        logger.info(f"  Val set: {len(val_idx)} samples")
        logger.info(f"  Test set: {len(test_idx)} samples")
        
        return {
            "features": features,
            "labels": remapped_labels,
            "original_labels": labels,
            "label_map": label_map,
            "num_classes": num_classes,
            "input_dim": features.shape[1],
            "test_idx": test_idx,
            "test_features": torch.tensor(features[test_idx], dtype=torch.float32),
            "test_labels": torch.tensor(remapped_labels[test_idx], dtype=torch.long),
        }

    def load_model(self, model_type, model_path, input_dim, num_classes):
        """Load trained model with correct architecture"""
        logger.info(f"Loading {model_type} model from {model_path}")
        
        if model_type == "transformer":
            model = TabularTransformer(
                input_dim=input_dim,
                hidden_dim=256,
                num_heads=8,
                num_layers=4,
                num_classes=num_classes,
                dropout=0.1,
            )
        elif model_type == "cnn_lstm":
            model = CNNLSTMHybrid(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dim=128,
                dropout=0.1,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        try:
            state_dict = torch.load(str(model_path), map_location=self.device)
            model.load_state_dict(state_dict)
            logger.info(f"✓ Successfully loaded weights")
        except Exception as e:
            logger.error(f"Could not load model weights: {e}")
            raise

        model = model.to(self.device)
        model.eval()
        return model

    def predict_on_dataset(self, model, X, batch_size=64):
        """Generate predictions on dataset"""
        predictions = []
        probabilities = []

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size].to(self.device)
                
                logits = model(batch_X)
                
                # Get predictions and probabilities
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                
                # For multi-class, get max probability
                probs = F.softmax(logits, dim=1).max(dim=1)[0].cpu().numpy()
                
                predictions.extend(preds)
                probabilities.extend(probs)

        return np.array(predictions), np.array(probabilities)

    def compute_metrics(self, y_true, y_pred, y_proba, num_classes):
        """Compute comprehensive metrics"""
        metrics = {}

        # Basic metrics
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

        # Per-class metrics
        try:
            metrics["precision"] = float(
                precision_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            metrics["recall"] = float(
                recall_score(y_true, y_pred, average="weighted", zero_division=0)
            )
            metrics["f1"] = float(
                f1_score(y_true, y_pred, average="weighted", zero_division=0)
            )
        except Exception as e:
            logger.warning(f"Could not compute weighted metrics: {e}")
            metrics["precision"] = None
            metrics["recall"] = None
            metrics["f1"] = None

        # Matthews correlation coefficient (works for multi-class)
        try:
            metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        except Exception as e:
            logger.warning(f"Could not compute MCC: {e}")
            metrics["mcc"] = None

        # ROC-AUC for binary classification
        if num_classes == 2:
            try:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        # Confusion matrix stats
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        return metrics

    def evaluate_dataset(self, dataset_path, dataset_name, model_type):
        """Evaluate model on single dataset"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating {model_type.upper()} on {dataset_name}")
        logger.info(f"{'='*80}")

        # Load data
        try:
            data = self.load_and_prepare_data(dataset_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None

        # Find model file
        model_path = self.find_model_file(dataset_name, model_type)
        if not model_path or not model_path.exists():
            logger.error(f"Could not find model file for {model_type} on {dataset_name}")
            return None

        # Load model
        try:
            model = self.load_model(
                model_type,
                model_path,
                input_dim=data["input_dim"],
                num_classes=data["num_classes"],
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

        # Generate predictions on test set
        logger.info("Generating predictions on test set...")
        y_pred, y_proba = self.predict_on_dataset(model, data["test_features"])

        # Compute metrics
        logger.info("Computing metrics...")
        y_test = data["test_labels"].numpy()
        metrics = self.compute_metrics(y_test, y_pred, y_proba, data["num_classes"])

        # Log results
        logger.info(f"\n✓ Results for {model_type}:")
        logger.info(f"  Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        if metrics["precision"] is not None:
            logger.info(f"  Precision (weighted): {metrics['precision']:.4f}")
        if metrics["recall"] is not None:
            logger.info(f"  Recall (weighted):    {metrics['recall']:.4f}")
        if metrics["f1"] is not None:
            logger.info(f"  F1 (weighted):        {metrics['f1']:.4f}")
        if metrics["mcc"] is not None:
            logger.info(f"  MCC:                {metrics['mcc']:.4f}")
        if metrics["roc_auc"] is not None:
            logger.info(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")

        # Add dataset metadata
        metrics["dataset"] = dataset_name
        metrics["model"] = model_type
        metrics["num_classes"] = data["num_classes"]
        metrics["test_samples"] = len(y_test)

        return metrics

    def evaluate_all(self):
        """Evaluate all available models"""
        # Discover datasets
        dataset_dir = Config.RAW_DATA_DIR
        datasets = sorted(dataset_dir.glob("*.csv"))
        
        if not datasets:
            logger.error(f"No datasets found in {dataset_dir}")
            return {}

        # Model types
        model_types = ["transformer", "cnn_lstm"]

        # Evaluate each model on each dataset
        for dataset_path in datasets:
            dataset_name = dataset_path.stem
            
            # # Skip some datasets if they have too few samples
            # if "final-merged" in dataset_name or "workinghours" in dataset_name:
            #     logger.info(f"Skipping {dataset_name} (merged dataset)")
            #     continue

            for model_type in model_types:
                key = f"{model_type}_{dataset_name}"
                
                metrics = self.evaluate_dataset(dataset_path, dataset_name, model_type)
                
                if metrics:
                    self.results[key] = metrics

        return self.results

    def generate_report(self, output_path=None):
        """Generate comprehensive comparison report"""
        if output_path is None:
            output_path = Config.REPORTS_DIR / "model_evaluation_comprehensive.json"
        else:
            output_path = Path(output_path)

        # Create summary
        summary = {
            "evaluation_date": str(Path(__file__).stat().st_mtime),
            "device": self.device,
            "random_seed": self.random_seed,
            "results": self.results,
        }

        # Save JSON report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n{'='*80}")
        logger.info(f"Report saved to: {output_path}")
        logger.info(f"{'='*80}")

        # Print summary table
        self._print_summary_table()

        return output_path

    def _print_summary_table(self):
        """Print formatted summary table"""
        logger.info("\n" + "=" * 120)
        logger.info("MODEL PERFORMANCE SUMMARY (TEST SET EVALUATION)")
        logger.info("=" * 120)

        # Organize by model type
        results_by_model = {}
        for key, metrics in self.results.items():
            model = metrics["model"]
            if model not in results_by_model:
                results_by_model[model] = {}
            results_by_model[model][metrics["dataset"]] = metrics

        for model_type in sorted(results_by_model.keys()):
            datasets = results_by_model[model_type]
            
            logger.info(f"\n{model_type.upper()}")
            logger.info("-" * 120)

            header = (
                f"{'Dataset':<40} {'Accuracy':<12} {'Balanced':<12} {'Precision':<12} "
                f"{'Recall':<12} {'F1':<12} {'MCC':<12} {'Classes':<8}"
            )
            logger.info(header)
            logger.info("-" * 120)

            for dataset_name in sorted(datasets.keys()):
                m = datasets[dataset_name]
                row = (
                    f"{dataset_name:<40} "
                    f"{m.get('accuracy', 0.0):<12.4f} "
                    f"{m.get('balanced_accuracy', 0.0):<12.4f} "
                    f"{m.get('precision', 0.0) or 0.0:<12.4f} "
                    f"{m.get('recall', 0.0) or 0.0:<12.4f} "
                    f"{m.get('f1', 0.0) or 0.0:<12.4f} "
                    f"{m.get('mcc', 0.0) or 0.0:<12.4f} "
                    f"{m.get('num_classes', 0):<8}"
                )
                logger.info(row)

            # Summary stats
            logger.info("-" * 120)
            acc_values = [m["accuracy"] for m in datasets.values()]
            bal_acc_values = [m["balanced_accuracy"] for m in datasets.values()]
            f1_values = [m["f1"] for m in datasets.values() if m["f1"] is not None]
            mcc_values = [m["mcc"] for m in datasets.values() if m["mcc"] is not None]

            row = (
                f"{'AVERAGE':<40} "
                f"{np.mean(acc_values):<12.4f} "
                f"{np.mean(bal_acc_values):<12.4f} "
                f"{'N/A':<12} "
                f"{'N/A':<12} "
                f"{np.mean(f1_values) if f1_values else 0.0:<12.4f} "
                f"{np.mean(mcc_values) if mcc_values else 0.0:<12.4f} "
                f"{'N/A':<8}"
            )
            logger.info(row)

        logger.info("=" * 120)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Proper evaluation of trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Compute device (default: cuda)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output report path",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("TRAINED MODEL EVALUATION (TEST SET EVALUATION)")
    logger.info("=" * 80)

    # Run evaluation
    evaluator = RobustModelEvaluator(device=args.device, random_seed=args.seed)
    results = evaluator.evaluate_all()

    if not results:
        logger.error("No results generated. Check logs above.")
        sys.exit(1)

    # Generate report
    output_path = evaluator.generate_report(output_path=args.output)

    logger.info(f"\n✓ Evaluation completed successfully!")
    logger.info(f"✓ Results saved to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n✓ Interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
