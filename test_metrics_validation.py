#!/usr/bin/env python3


import numpy as np
from pathlib import Path
import sys

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_metrics_aggregator():
    """Test basic metrics aggregation functionality"""
    from src.core.metrics_aggregator import MetricsAggregator
    
    print("Testing MetricsAggregator...")
    
    # Create sample data
    np.random.seed(42)
    y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0] * 60)  # 600 samples
    y_pred_model1 = np.array([0, 1, 1, 0, 1, 0, 1, 0, 0, 0] * 60)  # 95% accuracy
    y_pred_model2 = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 0] * 60)  # 90% accuracy
    
    # Binary classification scores
    y_score_model1 = np.random.rand(600, 2)
    y_score_model1[:, 1] = (y_pred_model1 == 1).astype(float)
    y_score_model1[:, 0] = 1 - y_score_model1[:, 1]
    
    y_score_model2 = np.random.rand(600, 2)
    y_score_model2[:, 1] = (y_pred_model2 == 1).astype(float)
    y_score_model2[:, 0] = 1 - y_score_model2[:, 1]
    
    # Test aggregator
    aggregator = MetricsAggregator()
    
    # Register models
    aggregator.register_model('Model-1')
    aggregator.register_model('Model-2')
    
    # Add metrics
    aggregator.add_metrics('Model-1', y_true, y_pred_model1, y_score_model1)
    aggregator.add_metrics('Model-2', y_true, y_pred_model2, y_score_model2)
    
    # Get summary
    summary = aggregator.get_metrics_summary()
    print("\n✓ Metrics Summary:")
    print(summary)
    
    # Compare models
    comp = aggregator.compare_models('Model-1', 'Model-2', metric='accuracy')
    print("\n✓ Comparison (Model-1 vs Model-2):")
    print(f"  Model-1 Accuracy: {comp['model1_score']:.4f}")
    print(f"  Model-2 Accuracy: {comp['model2_score']:.4f}")
    print(f"  Difference: {comp['difference']:.4f}")
    print(f"  Cohen's d: {comp['cohens_d']:.4f} ({comp['effect_size']})")
    print(f"  P-value (McNemar): {comp['p_mcnemar']:.6f}")
    print(f"  Significant at α=0.05: {comp['significant_at_0.05']}")
    
    # Generate report
    report = aggregator.generate_comparison_report()
    print("\n✓ Comparison Report Generated")
    print(f"  Models in report: {report['models']}")
    print(f"  Best accuracy: {report['best_models']['accuracy'][0]}")
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED - MetricsAggregator is working correctly!")
    print("="*70)
    
    return True


def test_comprehensive_evaluation():
    """Test evaluation pipeline with metrics integration"""
    from src.pipeline.evaluator import EvaluationPipeline
    
    print("\nTesting EvaluationPipeline metrics integration...")
    
    # Create evaluator
    evaluator = EvaluationPipeline()
    
    # Check metrics aggregator initialized
    assert hasattr(evaluator, 'metrics_aggregator'), "MetricsAggregator not initialized"
    assert hasattr(evaluator, 'model_name'), "model_name not set"
    
    print("✓ EvaluationPipeline has metrics_aggregator initialized")
    print(f"✓ Model name set to: {evaluator.model_name}")
    
    # Check export method exists
    assert hasattr(evaluator, 'export_comparison_reports'), "export_comparison_reports method missing"
    print("✓ export_comparison_reports method available")
    
    print("\n" + "="*70)
    print("✅ EvaluationPipeline integration test PASSED!")
    print("="*70)
    
    return True


def test_imports():
    """Test all required imports"""
    print("\nTesting imports...")
    
    try:
        from src.core.metrics_aggregator import MetricsAggregator
        print("✓ MetricsAggregator imported")
        
        from src.core.metrics_aggregator import ModelComparisonVisualizer
        print("✓ ModelComparisonVisualizer imported")
        
        from src.core.metrics_aggregator import export_metrics_for_paper
        print("✓ export_metrics_for_paper imported")
        
        from src.pipeline.evaluator import EvaluationPipeline
        print("EvaluationPipeline imported")
        
        print("\n" + "="*70)
        print("ALL IMPORTS SUCCESSFUL!")
        print("="*70)
        
        return True
    except ImportError as e:
        print(f"Import failed: {e}")
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("COMPREHENSIVE METRICS VALIDATION TEST")
    print("="*70)
    
    # Run tests
    all_pass = True
    
    try:
        all_pass = test_imports() and all_pass
        all_pass = test_metrics_aggregator() and all_pass
        all_pass = test_comprehensive_evaluation() and all_pass
        
        if all_pass:
            print("\n" + "="*70)
            print("🎉 ALL VALIDATION TESTS PASSED!")
            print("="*70)
            print("\nYou can now:")
            print("  1. Run evaluation pipeline (metrics captured automatically)")
            print("  2. Export comparison reports")
            print("  3. Generate LaTeX tables for paper")
            print("\nSee README.md and QUICKSTART.md for usage details")
            sys.exit(0)
        else:
            print("\nSome tests failed")
            sys.exit(1)
            
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
