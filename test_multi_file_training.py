#!/usr/bin/env python3
"""
Test script for multi-file training functionality.
"""

import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mcp_training.models.training_pipeline import TrainingPipeline
from mcp_training.models.config import ModelConfig

def create_test_export_file(filename: str, num_entries: int = 100):
    """Create a test export file with sample data."""
    test_data = {
        "data": [
            {
                "timestamp": "2025-06-23T10:00:00Z",
                "level": "INFO",
                "message": f"Test log entry {i}",
                "process": "test_process",
                "mac_address": f"00:11:22:33:44:5{i%10}",
                "ip_address": f"192.168.1.{i%255}",
                "word_count": i % 20,
                "special_char_count": i % 5,
                "uppercase_ratio": (i % 10) / 10.0,
                "number_count": i % 8,
                "unique_chars": i % 15,
                "process_rank": i % 100,
                "log_level_numeric": i % 5
            }
            for i in range(num_entries)
        ],
        "metadata": {
            "source": "test",
            "created_at": "2025-06-23T10:00:00Z",
            "entry_count": num_entries
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test export file: {filename} with {num_entries} entries")

def test_multi_file_training():
    """Test the multi-file training functionality."""
    print("=== Testing Multi-File Training ===")
    
    # Create test export files
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    
    test_file1 = exports_dir / "test_export_1.json"
    test_file2 = exports_dir / "test_export_2.json"
    
    create_test_export_file(test_file1, 50)
    create_test_export_file(test_file2, 75)
    
    # Test the training pipeline
    try:
        config = ModelConfig()
        pipeline = TrainingPipeline(config)
        
        # Test combining export data
        print("\n--- Testing data combination ---")
        
        # Load data from both files
        with open(test_file1, 'r') as f:
            data1 = json.load(f)
        with open(test_file2, 'r') as f:
            data2 = json.load(f)
        
        # Test the combine method
        combined = pipeline._combine_export_data([data1, data2])
        print(f"Combined {len(combined['data'])} entries from 2 files")
        print(f"Combined metadata: {combined['metadata']}")
        
        print("\n✅ Multi-file training test completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_multi_file_training() 