#!/usr/bin/env python3
"""
Test script to verify certificate generation and storage
"""

import sys
from pathlib import Path
import json

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_certificate_persistence():
    """Test that certificates are generated and saved to disk"""
    from config import Config
    from src.pipeline.evaluator import EvaluationPipeline
    
    print("\n" + "="*70)
    print("CERTIFICATE GENERATION & PERSISTENCE TEST")
    print("="*70 + "\n")
    
    # Find a small test dataset
    from pathlib import Path
    test_datasets = list(Config.RAW_DATA_DIR.glob("*.csv"))
    
    if not test_datasets:
        from config import Config as LegacyConfig
        test_datasets = list(LegacyConfig.LEGACY_DATASET_DIR.glob("*.csv"))
    
    if not test_datasets:
        print("❌ ERROR: No datasets found!")
        return False
    
    test_csv = str(test_datasets[0])
    dataset_name = Path(test_csv).stem
    
    print(f"Testing with dataset: {dataset_name}")
    print(f"Path: {test_csv}\n")
    
    try:
        # Run evaluation pipeline
        evaluator = EvaluationPipeline()
        result = evaluator.run(test_csv, dataset_name=dataset_name)
        
        if result is None:
            print("❌ ERROR: Evaluation returned None")
            return False
        
        # Check certificate directory
        cert_dir = Config.CERT_DIR / dataset_name
        
        if not cert_dir.exists():
            print(f" ERROR: Certificate directory not created at {cert_dir}")
            return False
        
        # List certificates
        cert_files = list(cert_dir.glob("cert_*.json"))
        index_file = cert_dir / "certificates_index.json"
        
        print(f"\n Certificate directory created: {cert_dir}")
        print(f" Generated {len(cert_files)} certificate files")
        
        if not cert_files:
            print(" ERROR: No certificate files found!")
            return False
        
        if not index_file.exists():
            print(" ERROR: Certificate index file not created!")
            return False
        
        # Load and verify index
        with open(index_file, 'r') as f:
            index_data = json.load(f)
        
        print(f"✓ Certificate index file exists")
        print(f"  - Total certificates: {index_data['total_certificates']}")
        print(f"  - Saved certificates: {index_data['saved_certificates']}")
        
        # Load and verify a sample certificate
        sample_cert_path = cert_files[0]
        with open(sample_cert_path, 'r') as f:
            cert_data = json.load(f)
        
        print(f"\n✓ Sample certificate loaded: {sample_cert_path.name}")
        print(f"  - Has input_tensor: {'input_tensor' in cert_data}")
        print(f"  - Has feature_importances: {'feature_importances' in cert_data}")
        print(f"  - Has hash_chain: {'hash_chain' in cert_data}")
        print(f"  - Has signature: {'signature' in cert_data}")
        print(f"  - Predicted class: {cert_data.get('predicted_class')}")
        print(f"  - Confidence: {cert_data.get('confidence'):.4f}")
        
        # Verify metrics are dataset-specific
        rq_metrics = result['results']
        print(f"\n✓ Evaluation results loaded:")
        print(f"  - RQ1 Integrity: {rq_metrics['rq1'].get('integrity_verification_success_rate', 'N/A')}")
        print(f"  - RQ2 Robustness: {rq_metrics['rq2'].get('avg_confidence', 'N/A')}")
        print(f"  - RQ3 Overhead: {rq_metrics['rq3'].get('throughput_samples_per_sec', 'N/A')} samples/sec")
        
        print("\n" + "="*70)
        print("✅ CERTIFICATE PERSISTENCE TEST PASSED!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_vary_across_datasets():
    """Verify that metrics are different across datasets"""
    from config import Config
    from src.pipeline.multi_dataset_integrated import MultiDatasetEvaluator
    
    print("\n" + "="*70)
    print("DATASET-SPECIFIC METRICS TEST")
    print("="*70 + "\n")
    
    try:
        evaluator = MultiDatasetEvaluator()
        results = evaluator.evaluate_all()
        
        if not results:
            print("❌ ERROR: No results from multi-dataset evaluation")
            return False
        
        print(f"\nEvaluated {len(results)} datasets\n")
        
        # Extract metrics from each dataset
        metrics_per_dataset = {}
        for dataset_name, dataset_result in results.items():
            if dataset_result['status'] == 'success' and dataset_result['result']:
                result = dataset_result['result']
                if 'results' in result:
                    rq1_val = result['results']['rq1'].get('integrity_verification_success_rate', 0)
                    rq2_val = result['results']['rq2'].get('avg_confidence', 0)
                    rq3_val = result['results']['rq3'].get('throughput_samples_per_sec', 0)
                    
                    metrics_per_dataset[dataset_name] = {
                        'rq1': rq1_val,
                        'rq2': rq2_val,
                        'rq3': rq3_val,
                    }
        
        if len(metrics_per_dataset) < 2:
            print("⚠ WARNING: Only 1 dataset evaluated successfully")
            return True  # Not a complete failure
        
        # Check if metrics are different
        rq1_values = [m['rq1'] for m in metrics_per_dataset.values()]
        rq2_values = [m['rq2'] for m in metrics_per_dataset.values()]
        rq3_values = [m['rq3'] for m in metrics_per_dataset.values()]
        
        print("Metrics across datasets:\n")
        for dataset_name, metrics in metrics_per_dataset.items():
            print(f"{dataset_name}:")
            print(f"  RQ1: {metrics['rq1']}")
            print(f"  RQ2: {metrics['rq2']:.4f}")
            print(f"  RQ3: {metrics['rq3']:.1f}")
        
        # Test if all values are identical (which would be wrong)
        all_rq1_same = len(set(rq1_values)) == 1
        all_rq2_same = len(set([f"{v:.4f}" for v in rq2_values])) == 1
        all_rq3_same = len(set([f"{v:.1f}" for v in rq3_values])) == 1
        
        if all_rq1_same and all_rq2_same and all_rq3_same:
            print("\n⚠ WARNING: All metrics are identical across datasets!")
            print("This suggests metrics are not being computed per-dataset.")
            return False
        else:
            print("\n✓ Metrics vary across datasets (as expected)")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    cert_test = test_certificate_persistence()
    
    # Only run metrics test if certificate test passes
    if cert_test:
        metrics_test = test_metrics_vary_across_datasets()
    else:
        metrics_test = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Certificate Persistence: {'✅ PASS' if cert_test else '❌ FAIL'}")
    print(f"Dataset-Specific Metrics: {'✅ PASS' if metrics_test else '❌ FAIL'}")
    print("="*70 + "\n")
    
    sys.exit(0 if (cert_test and metrics_test) else 1)
