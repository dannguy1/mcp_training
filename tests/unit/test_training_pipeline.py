"""
Tests for the training pipeline.
"""

import pytest
import json
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from mcp_training.models.training_pipeline import TrainingPipeline
from mcp_training.models.config import ModelConfig


class TestTrainingPipeline:
    """Test the TrainingPipeline class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        return ModelConfig()
    
    @pytest.fixture
    def pipeline(self, config):
        """Create a test pipeline."""
        return TrainingPipeline(config)
    
    @pytest.fixture
    def sample_export_data(self):
        """Create sample export data."""
        return {
            "data": [
                {
                    "timestamp": "2024-01-01T10:00:00Z",
                    "message": "hostapd: AP-STA-CONNECTED 00:11:22:33:44:55",
                    "process_name": "hostapd",
                    "log_level": "INFO"
                },
                {
                    "timestamp": "2024-01-01T10:01:00Z",
                    "message": "hostapd: AP-STA-DISCONNECTED 00:11:22:33:44:55",
                    "process_name": "hostapd",
                    "log_level": "INFO"
                },
                {
                    "timestamp": "2024-01-01T10:02:00Z",
                    "message": "wpa_supplicant: Trying to authenticate with 00:11:22:33:44:55",
                    "process_name": "wpa_supplicant",
                    "log_level": "INFO"
                }
            ]
        }
    
    @pytest.fixture
    def temp_export_file(self, sample_export_data):
        """Create a temporary export file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_export_data, f)
            temp_file = f.name
        
        yield temp_file
        
        # Cleanup
        Path(temp_file).unlink(missing_ok=True)
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.config is not None
        assert pipeline.trainer is not None
        assert pipeline.evaluator is not None
        assert pipeline.metadata_manager is not None
        assert pipeline.registry is not None
    
    @pytest.mark.asyncio
    async def test_load_exported_data(self, pipeline, temp_export_file, sample_export_data):
        """Test loading exported data."""
        # Test successful loading
        loaded_data = await pipeline._load_exported_data(temp_export_file)
        
        assert loaded_data == sample_export_data
        assert 'data' in loaded_data
        assert len(loaded_data['data']) == 3
    
    @pytest.mark.asyncio
    async def test_load_exported_data_invalid_structure(self, pipeline):
        """Test loading exported data with invalid structure."""
        # Create invalid export data
        invalid_data = {"invalid_key": []}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Export file must contain 'data' section"):
                await pipeline._load_exported_data(temp_file)
        finally:
            Path(temp_file).unlink(missing_ok=True)
    
    def test_check_model_requirements(self, pipeline):
        """Test model requirements checking."""
        # Test with good performance
        good_evaluation = {
            'threshold_checks': {
                'accuracy': True,
                'precision': True,
                'recall': True,
                'f1_score': True
            },
            'basic_metrics': {
                'roc_auc': 0.8
            }
        }
        
        assert pipeline._check_model_requirements(good_evaluation) is True
        
        # Test with poor performance
        poor_evaluation = {
            'threshold_checks': {
                'accuracy': False,
                'precision': True,
                'recall': True,
                'f1_score': True
            },
            'basic_metrics': {
                'roc_auc': 0.8
            }
        }
        
        assert pipeline._check_model_requirements(poor_evaluation) is False
        
        # Test with low ROC AUC
        low_roc_evaluation = {
            'threshold_checks': {
                'accuracy': True,
                'precision': True,
                'recall': True,
                'f1_score': True
            },
            'basic_metrics': {
                'roc_auc': 0.3
            }
        }
        
        assert pipeline._check_model_requirements(low_roc_evaluation) is False
    
    def test_generate_training_report(self, pipeline):
        """Test training report generation."""
        model_path = Path("/tmp/test_model")
        evaluation_results = {
            'basic_metrics': {'accuracy': 0.85},
            'advanced_metrics': {'confusion_matrix': [[80, 20], [10, 90]]},
            'threshold_checks': {'accuracy': True}
        }
        pipeline_duration = 120.5
        feature_names = ['feature1', 'feature2', 'feature3']
        
        # Mock the evaluator
        pipeline.evaluator.get_evaluation_summary = Mock(return_value={
            'overall_performance': {'threshold_pass_rate': 0.8},
            'recommendations': ['Test recommendation']
        })
        
        report = pipeline._generate_training_report(
            model_path, evaluation_results, pipeline_duration, feature_names
        )
        
        assert 'pipeline_info' in report
        assert 'model_info' in report
        assert 'model_performance' in report
        assert 'evaluation_details' in report
        assert 'threshold_checks' in report
        assert 'evaluation_summary' in report
        assert 'recommendations' in report
        
        # Check specific values
        assert report['pipeline_info']['duration'] == 120.5
        assert report['model_info']['feature_count'] == 3
        assert report['model_info']['feature_names'] == feature_names
        assert report['model_performance'] == {'accuracy': 0.85}
    
    def test_get_pipeline_status(self, pipeline):
        """Test getting pipeline status."""
        status = pipeline.get_pipeline_status()
        
        assert 'config' in status
        assert 'registry_stats' in status
        
        config = status['config']
        assert 'model_type' in config
        assert 'evaluation_thresholds' in config
        assert 'storage_directory' in config
    
    @pytest.mark.asyncio
    async def test_validate_export_for_training(self, pipeline, temp_export_file):
        """Test export validation for training."""
        validation_results = await pipeline.validate_export_for_training(temp_export_file)
        
        assert 'is_valid' in validation_results
        assert 'errors' in validation_results
        assert 'warnings' in validation_results
        assert 'stats' in validation_results
        
        # Should be valid for our test data
        assert validation_results['is_valid'] is True
        assert validation_results['stats']['total_records'] == 3
        assert validation_results['stats']['wifi_logs'] == 3
        assert validation_results['stats']['wifi_ratio'] == 1.0
    
    @pytest.mark.asyncio
    async def test_validate_export_for_training_invalid_file(self, pipeline):
        """Test export validation with invalid file."""
        validation_results = await pipeline.validate_export_for_training("/nonexistent/file.json")
        
        assert validation_results['is_valid'] is False
        assert len(validation_results['errors']) > 0
    
    @pytest.mark.asyncio
    @patch('mcp_training.models.training_pipeline.joblib.dump')
    async def test_save_model_with_metadata(self, mock_joblib_dump, pipeline, tmp_path):
        """Test saving model with metadata."""
        # Mock the model and other components
        mock_model = Mock()
        X = np.random.randn(10, 5)
        evaluation_results = {'basic_metrics': {'accuracy': 0.85}}
        training_duration = 60.0
        export_file_path = str(tmp_path / "test_export.json")
        model_type = "isolation_forest"
        feature_names = ['feature1', 'feature2']
        
        # Create a real export file for testing
        with open(export_file_path, 'w') as f:
            json.dump({"data": []}, f)
        
        # Mock metadata
        mock_metadata = Mock()
        pipeline.metadata_manager.create = Mock(return_value=mock_metadata)
        
        # Mock config storage directory to use tmp_path
        pipeline.config.storage.directory = str(tmp_path)
        
        result = await pipeline._save_model_with_metadata(
            mock_model, X, evaluation_results, training_duration,
            export_file_path, model_type, feature_names
        )
        
        # Verify calls
        assert mock_metadata.update_evaluation.called
        assert mock_metadata.update_training_duration.called
        assert mock_metadata.update_export_file_size.called
        assert mock_metadata.save.called
        assert mock_joblib_dump.call_count >= 2  # Model and scaler
        
        # Verify result is a Path
        assert isinstance(result, Path)
    
    @pytest.mark.asyncio
    @patch('mcp_training.models.training_pipeline.json.load')
    @patch('mcp_training.models.training_pipeline.json.dump')
    @patch('builtins.open', create=True)
    async def test_update_model_registry(self, mock_open, mock_json_dump, mock_json_load, pipeline, tmp_path):
        """Test updating model registry."""
        # Mock registry file operations
        mock_json_load.return_value = {'existing_model': {'status': 'available'}}
        
        # Mock config storage directory to use tmp_path
        pipeline.config.storage.directory = str(tmp_path)
        
        model_path = Mock()
        model_path.name = "test_version"
        
        evaluation_results = {
            'basic_metrics': {'accuracy': 0.85},
            'threshold_checks': {'accuracy': True}
        }
        
        # Mock evaluator
        pipeline.evaluator.get_evaluation_summary = Mock(return_value={
            'overall_performance': {'threshold_pass_rate': 0.8}
        })
        
        # Mock file context manager
        mock_file = Mock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_open.return_value.__exit__.return_value = None
        
        await pipeline._update_model_registry(model_path, evaluation_results)
        
        # Verify registry was updated
        assert mock_json_dump.called
        # Get the arguments passed to json.dump
        args, kwargs = mock_json_dump.call_args
        registry_data = args[0]  # First argument is the registry data (dict)
        assert 'test_version' in registry_data
        assert registry_data['test_version']['status'] == 'available' 