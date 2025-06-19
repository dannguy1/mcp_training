"""
Pytest configuration for MCP Training Service tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcp_training.core.config import TrainingConfig
from mcp_training.core.feature_extractor import WiFiFeatureExtractor
from mcp_training.core.model_trainer import ModelTrainer
from mcp_training.core.export_validator import ExportValidator


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_export_data():
    """Sample export data for testing."""
    return {
        "export_metadata": {
            "created_at": "2024-01-01T12:00:00Z",
            "total_records": 100,
            "format": "json"
        },
        "data": [
            {
                "timestamp": "2024-01-01T12:00:00Z",
                "message": "AP-STA-CONNECTED 00:11:22:33:44:55",
                "process_name": "hostapd",
                "log_level": "INFO"
            },
            {
                "timestamp": "2024-01-01T12:01:00Z",
                "message": "AP-STA-DISCONNECTED 00:11:22:33:44:55",
                "process_name": "hostapd",
                "log_level": "INFO"
            },
            {
                "timestamp": "2024-01-01T12:02:00Z",
                "message": "ERROR: Authentication failed for 00:11:22:33:44:55",
                "process_name": "hostapd",
                "log_level": "ERROR"
            }
        ]
    }


@pytest.fixture
def sample_export_file(temp_dir, sample_export_data):
    """Create a sample export file for testing."""
    import json
    
    export_file = Path(temp_dir) / "sample_export.json"
    with open(export_file, 'w') as f:
        json.dump(sample_export_data, f)
    
    return str(export_file)


@pytest.fixture
def config_instance():
    """Create a test configuration instance."""
    return TrainingConfig(
        models_dir="test_models",
        exports_dir="test_exports",
        logs_dir="test_logs"
    )


@pytest.fixture
def feature_extractor():
    """Create a feature extractor instance."""
    return WiFiFeatureExtractor()


@pytest.fixture
def model_trainer():
    """Create a model trainer instance."""
    return ModelTrainer(model_type="isolation_forest")


@pytest.fixture
def export_validator():
    """Create an export validator instance."""
    return ExportValidator() 