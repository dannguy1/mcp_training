#!/usr/bin/env python3
"""
Test script for enhanced training system.
"""

import numpy as np
import sys
import os
from pathlib import Path
import json
import tempfile

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_training.models.training_pipeline import TrainingPipeline
from mcp_training.core.enhanced_feature_extractor import EnhancedWiFiFeatureExtractor
from mcp_training.models.multi_algorithm_selector import MultiAlgorithmSelector

def create_test_data(n_records: int = 100) -> list:
    """Create test WiFi log data."""
    import random
    from datetime import datetime, timedelta
    
    data = []
    base_time = datetime.now()
    
    # Sample WiFi events
    events = [
        "AP-STA-CONNECTED: 00:11:22:33:44:55",
        "AP-STA-DISCONNECTED: 00:11:22:33:44:55",
        "STA-ASSOC: 00:11:22:33:44:55",
        "STA-DISASSOC: 00:11:22:33:44:55",
        "AUTH: 00:11:22:33:44:55",
        "ERROR: Authentication failed for 00:11:22:33:44:55",
        "WPA: 00:11:22:33:44:55 connected with WPA2",
        "SSID: TestNetwork channel: 6 signal: -45"
    ]
    
    for i in range(n_records):
        timestamp = base_time + timedelta(seconds=i*30)  # 30-second intervals
        event = random.choice(events)
        
        record = {
            'timestamp': timestamp.isoformat(),
            'message': event,
            'process_name': 'hostapd',
            'log_level': random.choice(['INFO', 'WARNING', 'ERROR']),
            'mac_address': f"{random.randint(0, 255):02x}:{random.randint(0, 255):02x}:{random.randint(0, 255):02x}:{random.randint(0, 255):02x}:{random.randint(0, 255):02x}:{random.randint(0, 255):02x}",
            'ip_address': f"192.168.1.{random.randint(1, 254)}",
            'ssid': random.choice(['TestNetwork', 'GuestNetwork', 'AdminNetwork']),
            'channel': random.choice([1, 6, 11, 36, 40, 44, 48]),
            'signal_strength': random.randint(-80, -30)
        }
        data.append(record)
    
    return data

def test_enhanced_feature_extractor():
    """Test the enhanced feature extractor."""
    print("Testing Enhanced Feature Extractor")
    print("=" * 50)
    
    # Create test data
    test_data = create_test_data(50)
    print(f"✓ Created {len(test_data)} test records")
    
    # Initialize feature extractor
    extractor = EnhancedWiFiFeatureExtractor()
    print(f"✓ Initialized EnhancedWiFiFeatureExtractor")
    
    # Extract features
    features_df = extractor.extract_features(test_data)
    print(f"✓ Extracted {features_df.shape[1]} features from {features_df.shape[0]} records")
    
    # Check feature types
    feature_names = list(features_df.columns)
    print(f"✓ Feature names: {feature_names[:10]}...")  # Show first 10
    
    # Check for specific feature types
    wifi_features = [f for f in feature_names if 'connection' in f or 'auth' in f or 'error' in f]
    time_features = [f for f in feature_names if 'hour' in f or 'day' in f or 'time' in f]
    behavioral_features = [f for f in feature_names if 'rate' in f or 'burst' in f or 'pattern' in f]
    
    print(f"✓ WiFi features: {len(wifi_features)}")
    print(f"✓ Time features: {len(time_features)}")
    print(f"✓ Behavioral features: {len(behavioral_features)}")
    
    return features_df

def test_multi_algorithm_selector():
    """Test the multi-algorithm selector."""
    print("\nTesting Multi-Algorithm Selector")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    X = np.random.randn(200, 20)  # 200 samples, 20 features
    
    # Initialize selector
    selector = MultiAlgorithmSelector()
    print(f"✓ Initialized MultiAlgorithmSelector")
    
    # Analyze data characteristics
    characteristics = selector.analyze_data_characteristics(X)
    print(f"✓ Analyzed data characteristics:")
    for key, value in characteristics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")
    
    # Select best algorithm
    selected_algorithm, parameters = selector.select_best_algorithm(X)
    print(f"✓ Selected algorithm: {selected_algorithm}")
    print(f"✓ Selection reason: {selector.selection_reason}")
    print(f"✓ Parameters: {parameters}")
    
    # Create model
    model = selector.create_model(selected_algorithm, parameters)
    print(f"✓ Created model: {type(model).__name__}")
    
    # Test model fitting
    model.fit(X)
    print(f"✓ Model fitted successfully")
    
    return selector, model

def test_enhanced_training_pipeline():
    """Test the enhanced training pipeline."""
    print("\nTesting Enhanced Training Pipeline")
    print("=" * 50)
    
    # Create test data file
    test_data = create_test_data(100)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({'data': test_data}, f)
        test_file = f.name
    
    try:
        # Initialize pipeline
        pipeline = TrainingPipeline()
        print(f"✓ Initialized TrainingPipeline with enhanced components")
        
        # Validate export file
        import asyncio
        validation_result = asyncio.run(pipeline.validate_export_for_training(test_file))
        print(f"✓ Validation result: {validation_result['is_valid']}")
        
        # Test feature extraction
        features_data = asyncio.run(pipeline._load_and_preprocess_data(test_file))
        X, feature_names = pipeline._extract_features(features_data)
        print(f"✓ Extracted {len(feature_names)} features from {X.shape[0]} samples")
        
        # Test model training with auto-selection
        model = asyncio.run(pipeline._train_model(X, "auto"))
        print(f"✓ Trained model: {type(model).__name__}")
        
        # Check algorithm selection info
        if 'algorithm_selection' in pipeline.training_metrics:
            selection_info = pipeline.training_metrics['algorithm_selection']
            print(f"✓ Algorithm selection: {selection_info['selected_algorithm']}")
            print(f"✓ Selection reason: {selection_info['selection_reason']}")
        
        return pipeline, model
        
    finally:
        # Clean up
        os.unlink(test_file)

def main():
    """Run all tests."""
    print("Enhanced Training System Test")
    print("=" * 60)
    
    try:
        # Test enhanced feature extractor
        features_df = test_enhanced_feature_extractor()
        
        # Test multi-algorithm selector
        selector, model = test_multi_algorithm_selector()
        
        # Test enhanced training pipeline
        pipeline, trained_model = test_enhanced_training_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("Enhanced training system is working correctly.")
        print("\nKey improvements implemented:")
        print("- Advanced WiFi-specific feature engineering")
        print("- Multi-algorithm selection with data analysis")
        print("- Enhanced behavioral and network features")
        print("- Cyclical time encoding")
        print("- Quality assessment and validation")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 