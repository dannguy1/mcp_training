#!/usr/bin/env python3
"""
Test script for enhanced evaluation system.
"""

import numpy as np
import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_training.models.evaluation import ModelEvaluator
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def test_enhanced_evaluation():
    """Test the enhanced evaluation system."""
    print("Testing Enhanced Evaluation System")
    print("=" * 50)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    print(f"✓ Created ModelEvaluator with quality thresholds: {evaluator.quality_thresholds}")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create normal data
    X_normal = np.random.normal(0, 1, (n_samples, n_features))
    
    # Create some anomalies (5% of data)
    n_anomalies = int(n_samples * 0.05)
    X_anomalies = np.random.normal(5, 2, (n_anomalies, n_features))
    
    # Combine data
    X = np.vstack([X_normal, X_anomalies])
    print(f"✓ Generated synthetic data: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Scaled features using StandardScaler")
    
    # Train Isolation Forest
    model = IsolationForest(contamination=0.1, random_state=42)
    model.fit(X_scaled)
    print("✓ Trained IsolationForest model")
    
    # Evaluate model
    print("\nRunning enhanced evaluation...")
    results = evaluator.evaluate_model(model, X_scaled, feature_names=[f"feature_{i}" for i in range(n_features)])
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 30)
    
    # Basic metrics
    print(f"Basic Metrics ({len(results['basic_metrics'])} metrics):")
    for key, value in list(results['basic_metrics'].items())[:5]:  # Show first 5
        print(f"  {key}: {value:.4f}")
    
    # Clustering metrics
    print(f"\nClustering Metrics ({len(results['clustering_metrics'])} metrics):")
    for key, value in results['clustering_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    # Quality metrics
    print(f"\nQuality Metrics ({len(results['quality_metrics'])} metrics):")
    for key, value in results['quality_metrics'].items():
        print(f"  {key}: {value:.4f}")
    
    # Quality assessment
    assessment = results['quality_assessment']
    print(f"\nQuality Assessment:")
    print(f"  Overall Score: {assessment['overall_score']:.4f}")
    print(f"  Quality Level: {assessment['quality_level']}")
    print(f"  Validation Status: {assessment['validation_status']}")
    print(f"  Issues: {len(assessment['issues'])}")
    for issue in assessment['issues']:
        print(f"    - {issue}")
    
    # Recommendations
    print(f"\nRecommendations ({len(results['recommendations'])}):")
    for i, rec in enumerate(results['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    # Evaluation summary
    summary = results['evaluation_summary']
    print(f"\nEvaluation Summary:")
    print(f"  Model Quality Score: {summary['model_quality_score']:.4f}")
    print(f"  Quality Level: {summary['quality_level']}")
    print(f"  Validation Status: {summary['validation_status']}")
    print(f"  Total Samples: {summary['total_samples']}")
    print(f"  Total Features: {summary['total_features']}")
    
    # Test quality thresholds
    print(f"\nQuality Threshold Tests:")
    thresholds = evaluator.quality_thresholds
    for key, threshold in thresholds.items():
        print(f"  {key}: {threshold}")
    
    print("\n" + "=" * 50)
    print("Enhanced Evaluation Test Completed Successfully!")
    
    return results

if __name__ == "__main__":
    try:
        results = test_enhanced_evaluation()
        print(f"\nTest passed! Model quality score: {results['evaluation_summary']['model_quality_score']:.4f}")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 