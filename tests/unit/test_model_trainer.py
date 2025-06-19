"""
Unit tests for model trainer.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from mcp_training.core.model_trainer import ModelTrainer
from mcp_training.core.feature_extractor import FeatureExtractor
from mcp_training.models.config import ModelConfig


class TestModelTrainer:
    """Test cases for ModelTrainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        return [
            {
                "timestamp": "2024-01-01T10:00:00Z",
                "message": "WiFi connection established",
                "process_name": "wpa_supplicant",
                "log_level": "INFO",
                "structured_data": {
                    "ssid": "TestNetwork",
                    "signal_strength": -45,
                    "channel": 6,
                    "frequency": 2437,
                    "bitrate": 54,
                    "tx_power": 20,
                    "rx_packets": 1000,
                    "tx_packets": 500,
                    "rx_bytes": 50000,
                    "tx_bytes": 25000,
                    "rx_errors": 5,
                    "tx_errors": 2,
                    "rx_dropped": 1,
                    "tx_dropped": 0,
                    "beacon_loss": 0,
                    "auth_failures": 0,
                    "assoc_failures": 0,
                    "roaming_events": 0,
                    "disconnect_events": 0,
                    "reconnect_events": 0
                }
            },
            {
                "timestamp": "2024-01-01T10:01:00Z",
                "message": "WiFi connection lost",
                "process_name": "wpa_supplicant",
                "log_level": "WARNING",
                "structured_data": {
                    "ssid": "TestNetwork",
                    "signal_strength": -85,
                    "channel": 6,
                    "frequency": 2437,
                    "bitrate": 6,
                    "tx_power": 20,
                    "rx_packets": 1200,
                    "tx_packets": 600,
                    "rx_bytes": 60000,
                    "tx_bytes": 30000,
                    "rx_errors": 10,
                    "tx_errors": 5,
                    "rx_dropped": 3,
                    "tx_dropped": 1,
                    "beacon_loss": 2,
                    "auth_failures": 1,
                    "assoc_failures": 0,
                    "roaming_events": 1,
                    "disconnect_events": 1,
                    "reconnect_events": 0
                }
            }
        ]
    
    @pytest.fixture
    def model_config(self):
        """Sample model configuration."""
        return ModelConfig(
            model_type="isolation_forest",
            contamination=0.1,
            random_state=42,
            n_estimators=100,
            max_samples="auto"
        )
    
    @pytest.fixture
    def trainer(self, model_config):
        """Model trainer instance."""
        return ModelTrainer(model_config)
    
    def test_init(self, trainer, model_config):
        """Test trainer initialization."""
        assert trainer.config == model_config
        assert trainer.model is None
        assert trainer.scaler is None
        assert trainer.feature_extractor is not None
        assert isinstance(trainer.feature_extractor, FeatureExtractor)
    
    def test_prepare_features(self, trainer, sample_data):
        """Test feature preparation."""
        features = trainer.prepare_features(sample_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_data)
        assert features.shape[1] > 0  # Should have some features
        
        # Check that features are numeric
        assert np.issubdtype(features.dtype, np.number)
    
    def test_train_model(self, trainer, sample_data):
        """Test model training."""
        # Prepare features
        features = trainer.prepare_features(sample_data)
        
        # Train model
        result = trainer.train_model(features)
        
        assert result['success'] is True
        assert trainer.model is not None
        assert trainer.scaler is not None
        assert 'training_time' in result
        assert 'model_info' in result
    
    def test_train_model_with_insufficient_data(self, trainer):
        """Test training with insufficient data."""
        # Create minimal data
        minimal_data = [{"timestamp": "2024-01-01T10:00:00Z", "message": "test"}]
        
        with pytest.raises(ValueError, match="Insufficient data"):
            trainer.prepare_features(minimal_data)
    
    def test_save_model(self, trainer, sample_data, tmp_path):
        """Test model saving."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        scaler_path = tmp_path / "test_scaler.joblib"
        
        result = trainer.save_model(str(model_path), str(scaler_path))
        
        assert result['success'] is True
        assert model_path.exists()
        assert scaler_path.exists()
    
    def test_load_model(self, trainer, sample_data, tmp_path):
        """Test model loading."""
        # Train and save model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        model_path = tmp_path / "test_model.joblib"
        scaler_path = tmp_path / "test_scaler.joblib"
        trainer.save_model(str(model_path), str(scaler_path))
        
        # Create new trainer and load model
        new_trainer = ModelTrainer(trainer.config)
        result = new_trainer.load_model(str(model_path), str(scaler_path))
        
        assert result['success'] is True
        assert new_trainer.model is not None
        assert new_trainer.scaler is not None
    
    def test_predict(self, trainer, sample_data):
        """Test model prediction."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Make predictions
        predictions = trainer.predict(features)
        
        assert isinstance(predictions, dict)
        assert 'scores' in predictions
        assert 'predictions' in predictions
        assert 'threshold' in predictions
        assert len(predictions['scores']) == len(sample_data)
        assert len(predictions['predictions']) == len(sample_data)
    
    def test_evaluate_model(self, trainer, sample_data):
        """Test model evaluation."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Create test labels (simulate some anomalies)
        test_labels = np.array([0, 1])  # Second sample is anomaly
        
        # Evaluate model
        evaluation = trainer.evaluate_model(features, test_labels)
        
        assert isinstance(evaluation, dict)
        assert 'score_statistics' in evaluation
        assert 'supervised_metrics' in evaluation
        assert 'accuracy' in evaluation['supervised_metrics']
    
    def test_evaluate_model_without_labels(self, trainer, sample_data):
        """Test model evaluation without labels."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Evaluate model without labels
        evaluation = trainer.evaluate_model(features)
        
        assert isinstance(evaluation, dict)
        assert 'score_statistics' in evaluation
        assert 'supervised_metrics' not in evaluation
    
    def test_get_model_info(self, trainer, sample_data):
        """Test getting model information."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Get model info
        info = trainer.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_type' in info
        assert 'training_samples' in info
        assert 'feature_count' in info
        assert 'model_params' in info
    
    @patch('mcp_training.core.model_trainer.joblib.dump')
    def test_save_model_error(self, mock_dump, trainer, sample_data, tmp_path):
        """Test model saving error handling."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Mock joblib.dump to raise exception
        mock_dump.side_effect = Exception("Save error")
        
        model_path = tmp_path / "test_model.joblib"
        scaler_path = tmp_path / "test_scaler.joblib"
        
        result = trainer.save_model(str(model_path), str(scaler_path))
        
        assert result['success'] is False
        assert 'error' in result
    
    @patch('mcp_training.core.model_trainer.joblib.load')
    def test_load_model_error(self, mock_load, trainer, tmp_path):
        """Test model loading error handling."""
        # Mock joblib.load to raise exception
        mock_load.side_effect = Exception("Load error")
        
        model_path = tmp_path / "test_model.joblib"
        scaler_path = tmp_path / "test_scaler.joblib"
        
        result = trainer.load_model(str(model_path), str(scaler_path))
        
        assert result['success'] is False
        assert 'error' in result
    
    def test_invalid_model_type(self):
        """Test initialization with invalid model type."""
        config = ModelConfig(model_type="invalid_model")
        
        with pytest.raises(ValueError, match="Unsupported model type"):
            ModelTrainer(config)
    
    def test_custom_threshold(self, trainer, sample_data):
        """Test prediction with custom threshold."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Make predictions with custom threshold
        custom_threshold = 0.5
        predictions = trainer.predict(features, threshold=custom_threshold)
        
        assert predictions['threshold'] == custom_threshold
    
    def test_feature_scaling(self, trainer, sample_data):
        """Test that features are properly scaled."""
        # Train model first
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Check that scaler was fitted
        assert trainer.scaler is not None
        
        # Check that features are scaled
        scaled_features = trainer.scaler.transform(features)
        assert not np.array_equal(features, scaled_features)
    
    def test_model_persistence(self, trainer, sample_data, tmp_path):
        """Test that model can be saved and loaded correctly."""
        # Train model
        features = trainer.prepare_features(sample_data)
        trainer.train_model(features)
        
        # Save model
        model_path = tmp_path / "test_model.joblib"
        scaler_path = tmp_path / "test_scaler.joblib"
        trainer.save_model(str(model_path), str(scaler_path))
        
        # Load model in new trainer
        new_trainer = ModelTrainer(trainer.config)
        new_trainer.load_model(str(model_path), str(scaler_path))
        
        # Make predictions with both trainers
        original_predictions = trainer.predict(features)
        loaded_predictions = new_trainer.predict(features)
        
        # Predictions should be identical
        np.testing.assert_array_equal(
            original_predictions['predictions'],
            loaded_predictions['predictions']
        )
        np.testing.assert_array_almost_equal(
            original_predictions['scores'],
            loaded_predictions['scores']
        ) 