# Training Enhancement Implementation Plan

## Overview

This document outlines the implementation plan to enhance the model training system for the MCP service, addressing the gaps identified between the current implementation and the documented requirements. The goal is to create a robust, production-ready training pipeline that supports advanced feature engineering, comprehensive evaluation, and seamless integration with the existing system.

## Current State Analysis

### ✅ Implemented Components
1. **Basic Model Training**: `ModelTrainer` class with Isolation Forest
2. **Model Configuration**: `ModelConfig` with basic parameters
3. **Model Loading**: `ModelLoader` for model persistence
4. **Basic Monitoring**: `ModelMonitor` with Prometheus metrics
5. **Training Script**: `train_model.py` with command-line interface
6. **Feature Extraction**: Basic WiFi feature extraction
7. **UI Model Management**: Frontend interface for model loading and management
8. **Export Data Integration**: Training from exported data files

### ❌ Missing or Incomplete Components
1. **Advanced Feature Engineering**: Sophisticated feature extraction
2. **Database Integration**: Proper integration with actual schema
3. **Comprehensive Evaluation**: Multiple evaluation metrics
4. **Model Versioning**: Robust versioning with metadata
5. **Training Pipeline**: End-to-end training workflow
6. **Integration Points**: Connection with ModelManager and monitoring
7. **UI Model Management**: Frontend interface for model training and management
8. **Export Data Integration**: Training from exported data files

## Implementation Phases

### Phase 1: Export Data Integration (Priority: Critical)

#### 1.1 Export Data Training Integration
```python
# backend/app/models/training.py - Update to use exported data
class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.feature_extractor = FeatureExtractor()
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
    
    async def train_from_export_data(self, export_file_path: str) -> Dict[str, Any]:
        """Train model using exported data from the exporter system."""
        try:
            logger.info(f"Training model from export data: {export_file_path}")
            
            # Load exported data
            exported_data = await self._load_exported_data(export_file_path)
            
            # Extract features from exported logs
            features = await self.feature_extractor.extract_wifi_features(exported_data['logs'])
            
            # Prepare training data
            X = self._prepare_feature_matrix(features)
            y = np.zeros(len(X))  # Unsupervised learning
            
            # Train model
            model = self._train_model(X, y)
            
            # Evaluate model
            evaluation_results = self._evaluate_model(model, X, y)
            
            # Save model with metadata
            model_path = await self._save_model_with_metadata(
                model, features, evaluation_results, export_file_path
            )
            
            return {
                'model_path': str(model_path),
                'evaluation_results': evaluation_results,
                'training_samples': len(X),
                'export_file': export_file_path
            }
            
        except Exception as e:
            logger.error(f"Error training from export data: {e}")
            raise
    
    async def _load_exported_data(self, export_file_path: str) -> Dict[str, Any]:
        """Load data from exported JSON file."""
        try:
            with open(export_file_path, 'r') as f:
                data = json.load(f)
            
            # Validate export data structure
            if 'logs' not in data:
                raise ValueError("Export file must contain 'logs' section")
            
            logger.info(f"Loaded {len(data['logs'])} log entries from export file")
            return data
            
        except Exception as e:
            logger.error(f"Error loading exported data: {e}")
            raise
    
    async def _save_model_with_metadata(self, model: Any, features: Dict[str, Any],
                                      evaluation_results: Dict[str, Any],
                                      export_file_path: str) -> Path:
        """Save model with comprehensive metadata including export source."""
        # Generate version
        version = datetime.now().strftime(self.config.storage.version_format)
        
        # Create model directory
        model_dir = Path(self.config.storage.directory) / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'model.joblib'
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = model_dir / 'scaler.joblib'
        joblib.dump(self.scaler, scaler_path)
        
        # Create metadata with export information
        metadata = {
            'model_info': {
                'version': version,
                'created_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'training_source': 'export_data',
                'export_file': export_file_path
            },
            'training_info': {
                'training_samples': len(features),
                'feature_names': list(features.keys()),
                'export_file_size': os.path.getsize(export_file_path)
            },
            'evaluation_info': evaluation_results,
            'deployment_info': {
                'status': 'available',
                'deployed_at': None,
                'deployed_by': None
            }
        }
        
        # Save metadata
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return model_dir
```

#### 1.2 Export Data Validation
```python
# backend/app/models/export_validator.py - New module for export validation
class ExportDataValidator:
    """Validate exported data for training suitability."""
    
    def __init__(self):
        self.logger = logging.getLogger("ExportDataValidator")
    
    async def validate_export_for_training(self, export_file_path: str) -> Dict[str, Any]:
        """Validate that exported data is suitable for training."""
        try:
            # Load export data
            with open(export_file_path, 'r') as f:
                data = json.load(f)
            
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }
            
            # Check required sections
            if 'logs' not in data:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Missing 'logs' section in export file")
            
            if 'metadata' not in data:
                validation_results['warnings'].append("Missing 'metadata' section in export file")
            
            # Validate log entries
            if 'logs' in data:
                log_validation = self._validate_log_entries(data['logs'])
                validation_results['stats']['log_count'] = len(data['logs'])
                validation_results['stats']['valid_logs'] = log_validation['valid_count']
                validation_results['stats']['invalid_logs'] = log_validation['invalid_count']
                
                if log_validation['errors']:
                    validation_results['errors'].extend(log_validation['errors'])
                
                if log_validation['warnings']:
                    validation_results['warnings'].extend(log_validation['warnings'])
            
            # Check data quality
            quality_check = self._check_data_quality(data)
            validation_results['stats'].update(quality_check['stats'])
            
            if quality_check['errors']:
                validation_results['errors'].extend(quality_check['errors'])
            
            # Determine overall validity
            if validation_results['errors']:
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error validating export file: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'stats': {}
            }
    
    def _validate_log_entries(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate individual log entries."""
        valid_count = 0
        invalid_count = 0
        errors = []
        warnings = []
        
        required_fields = ['timestamp', 'message', 'process_name']
        
        for i, log in enumerate(logs):
            # Check required fields
            missing_fields = [field for field in required_fields if field not in log]
            if missing_fields:
                invalid_count += 1
                errors.append(f"Log {i}: Missing required fields: {missing_fields}")
                continue
            
            # Check data types
            if not isinstance(log['timestamp'], str):
                invalid_count += 1
                errors.append(f"Log {i}: Invalid timestamp format")
                continue
            
            if not isinstance(log['message'], str):
                invalid_count += 1
                errors.append(f"Log {i}: Invalid message format")
                continue
            
            valid_count += 1
        
        return {
            'valid_count': valid_count,
            'invalid_count': invalid_count,
            'errors': errors,
            'warnings': warnings
        }
    
    def _check_data_quality(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check overall data quality."""
        stats = {}
        errors = []
        
        if 'logs' in data:
            logs = data['logs']
            stats['total_logs'] = len(logs)
            
            # Check for WiFi-related logs
            wifi_logs = [log for log in logs if log.get('process_name') in ['hostapd', 'wpa_supplicant']]
            stats['wifi_logs'] = len(wifi_logs)
            stats['wifi_ratio'] = len(wifi_logs) / len(logs) if logs else 0
            
            if stats['wifi_ratio'] < 0.1:
                warnings.append("Low ratio of WiFi-related logs (< 10%)")
            
            # Check time range
            if logs:
                timestamps = [log.get('timestamp') for log in logs if log.get('timestamp')]
                if timestamps:
                    try:
                        times = [pd.to_datetime(ts) for ts in timestamps]
                        time_range = max(times) - min(times)
                        stats['time_range_days'] = time_range.days
                        
                        if time_range.days < 1:
                            warnings.append("Export contains less than 1 day of data")
                    except:
                        pass
        
        return {
            'stats': stats,
            'errors': errors,
            'warnings': warnings
        }
```

### Phase 2: Enhanced Model Configuration (Priority: High)

#### 2.1 Update ModelConfig Structure
```python
# backend/app/models/config.py - Enhanced configuration
class ModelConfig(BaseModel):
    """Enhanced configuration for model training and serving."""
    version: str = Field(default="1.0.0", description="Configuration version")
    
    # Model parameters
    model: ModelParameters = Field(default_factory=ModelParameters)
    
    # Enhanced feature configuration
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    
    # Training configuration
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    # Storage configuration
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    # Evaluation configuration
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    # Logging configuration
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Database configuration
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    
    # Monitoring configuration
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)

class DatabaseConfig(BaseModel):
    """Database configuration for training."""
    table_name: str = Field(default="log_entries", description="Main log table")
    wifi_programs: List[str] = Field(default=["hostapd", "wpa_supplicant"], description="WiFi-related programs")
    batch_size: int = Field(default=1000, description="Data fetching batch size")
    max_records: int = Field(default=100000, description="Maximum records to fetch")

class MonitoringConfig(BaseModel):
    """Monitoring configuration for training."""
    enable_drift_detection: bool = Field(default=True, description="Enable drift detection")
    drift_threshold: float = Field(default=0.1, description="Drift detection threshold")
    performance_tracking: bool = Field(default=True, description="Enable performance tracking")
    resource_monitoring: bool = Field(default=True, description="Enable resource monitoring")
```

#### 2.2 Enhanced Configuration YAML
```yaml
# backend/app/config/model_config.yaml - Updated structure
version: '2.0.0'

model:
  type: isolation_forest
  n_estimators: 100
  max_samples: auto
  contamination: 0.1
  random_state: 42
  bootstrap: true
  max_features: 1.0

features:
  numeric:
    - auth_failure_ratio
    - deauth_ratio
    - beacon_ratio
    - unique_mac_count
    - unique_ssid_count
    - mean_signal_strength
    - std_signal_strength
    - mean_data_rate
    - mean_packet_loss
    - error_ratio
    - warning_ratio
  categorical:
    - device_type
    - connection_type
  temporal:
    - mean_hour_of_day
    - mean_day_of_week
    - mean_time_between_events
  derived:
    - signal_strength_trend
    - connection_stability
    - network_load

training:
  test_size: 0.2
  validation_size: 0.1
  random_state: 42
  n_jobs: -1
  cross_validation_folds: 5
  early_stopping: true
  patience: 10

storage:
  directory: models
  version_format: '%Y%m%d_%H%M%S'
  keep_last_n_versions: 5
  backup_enabled: true
  compression: true

evaluation:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
    - roc_auc
    - average_precision
  thresholds:
    accuracy: 0.8
    precision: 0.7
    recall: 0.7
    f1_score: 0.7
    roc_auc: 0.8
  cross_validation: true

database:
  table_name: log_entries
  wifi_programs:
    - hostapd
    - wpa_supplicant
  batch_size: 1000
  max_records: 100000

monitoring:
  enable_drift_detection: true
  drift_threshold: 0.1
  performance_tracking: true
  resource_monitoring: true
  alerting:
    enabled: true
    email_notifications: false
    slack_notifications: false

logging:
  level: INFO
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  file: logs/model_training.log
  rotation:
    max_size: 100MB
    backup_count: 10
```

### Phase 3: Comprehensive Evaluation System (Priority: High)

#### 3.1 Enhanced Model Evaluation
```python
# backend/app/models/evaluation.py - New evaluation module
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metrics = {}
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray = None) -> Dict[str, Any]:
        """Evaluate model performance comprehensively."""
        try:
            # For unsupervised learning, create pseudo-labels
            if y is None:
                y_pred = model.predict(X)
                y = (y_pred == -1).astype(int)  # Convert to binary labels
            
            # Get anomaly scores
            scores = -model.score_samples(X)
            
            # Calculate all metrics
            evaluation_results = {
                'basic_metrics': self._calculate_basic_metrics(y, scores),
                'advanced_metrics': self._calculate_advanced_metrics(y, scores),
                'cross_validation': self._calculate_cross_validation(model, X, y),
                'feature_importance': self._calculate_feature_importance(model, X),
                'performance_analysis': self._analyze_performance(y, scores)
            }
            
            # Check against thresholds
            evaluation_results['threshold_checks'] = self._check_thresholds(evaluation_results)
            
            # Store metrics
            self.metrics = evaluation_results
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def _calculate_basic_metrics(self, y: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Calculate basic classification metrics."""
        # Use threshold to convert scores to predictions
        threshold = np.percentile(scores, 90)  # 90th percentile as threshold
        y_pred = (scores > threshold).astype(int)
        
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, scores),
            'average_precision': average_precision_score(y, scores)
        }
    
    def _calculate_advanced_metrics(self, y: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        """Calculate advanced evaluation metrics."""
        threshold = np.percentile(scores, 90)
        y_pred = (scores > threshold).astype(int)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'threshold_analysis': self._analyze_thresholds(y, scores),
            'score_distribution': {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'percentiles': {
                    '25': float(np.percentile(scores, 25)),
                    '50': float(np.percentile(scores, 50)),
                    '75': float(np.percentile(scores, 75)),
                    '90': float(np.percentile(scores, 90)),
                    '95': float(np.percentile(scores, 95)),
                    '99': float(np.percentile(scores, 99))
                }
            }
        }
    
    def _calculate_cross_validation(self, model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate cross-validation scores."""
        cv_scores = cross_val_score(model, X, y, cv=self.config.training.cross_validation_folds)
        
        return {
            'mean_cv_score': float(np.mean(cv_scores)),
            'std_cv_score': float(np.std(cv_scores)),
            'cv_scores': cv_scores.tolist()
        }
    
    def _calculate_feature_importance(self, model: Any, X: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for Isolation Forest."""
        # For Isolation Forest, we can use feature importances if available
        if hasattr(model, 'feature_importances_'):
            return dict(zip(range(X.shape[1]), model.feature_importances_))
        else:
            # Fallback: use permutation importance
            return self._calculate_permutation_importance(model, X)
    
    def _check_thresholds(self, results: Dict[str, Any]) -> Dict[str, bool]:
        """Check if metrics meet configured thresholds."""
        thresholds = self.config.evaluation.thresholds
        basic_metrics = results['basic_metrics']
        
        checks = {}
        for metric, threshold in thresholds.items():
            if metric in basic_metrics:
                checks[metric] = basic_metrics[metric] >= threshold
        
        return checks
```

### Phase 4: Enhanced Model Versioning and Metadata (Priority: Medium)

#### 4.1 Comprehensive Model Metadata
```python
# backend/app/models/metadata.py - New metadata management
import json
import yaml
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import hashlib
import joblib

class ModelMetadata:
    """Comprehensive model metadata management."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.metadata = {}
    
    def create_metadata(self, model: Any, features: Dict[str, Any], 
                       evaluation_results: Dict[str, Any], 
                       training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive model metadata."""
        metadata = {
            'model_info': {
                'version': self._generate_version(),
                'created_at': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'model_hash': self._calculate_model_hash(model),
                'config_hash': self._calculate_config_hash()
            },
            'training_info': {
                'config': training_config,
                'data_info': {
                    'feature_count': len(features),
                    'feature_names': list(features.keys()),
                    'training_samples': training_config.get('training_samples', 0),
                    'validation_samples': training_config.get('validation_samples', 0)
                },
                'training_duration': training_config.get('training_duration', 0),
                'training_memory_usage': training_config.get('memory_usage', 0)
            },
            'evaluation_info': {
                'metrics': evaluation_results['basic_metrics'],
                'advanced_metrics': evaluation_results['advanced_metrics'],
                'threshold_checks': evaluation_results['threshold_checks'],
                'cross_validation': evaluation_results['cross_validation']
            },
            'deployment_info': {
                'status': 'pending',
                'deployed_at': None,
                'deployed_by': None,
                'deployment_environment': None
            },
            'monitoring_info': {
                'drift_detection_enabled': self.config.monitoring.enable_drift_detection,
                'performance_tracking_enabled': self.config.monitoring.performance_tracking,
                'last_monitoring_check': None,
                'drift_scores': {}
            }
        }
        
        self.metadata = metadata
        return metadata
    
    def save_metadata(self, metadata: Dict[str, Any], model_path: Path) -> None:
        """Save metadata to file."""
        metadata_path = model_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def load_metadata(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """Load metadata from file."""
        metadata_path = model_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def _generate_version(self) -> str:
        """Generate model version string."""
        return datetime.now().strftime(self.config.storage.version_format)
    
    def _calculate_model_hash(self, model: Any) -> str:
        """Calculate hash of model parameters."""
        # For Isolation Forest, hash the key parameters
        params = {
            'n_estimators': model.n_estimators,
            'max_samples': model.max_samples,
            'contamination': model.contamination,
            'random_state': model.random_state
        }
        return hashlib.md5(str(params).encode()).hexdigest()
    
    def _calculate_config_hash(self) -> str:
        """Calculate hash of configuration."""
        config_str = self.config.model_dump_json()
        return hashlib.md5(config_str.encode()).hexdigest()
```

### Phase 5: Training Pipeline Integration (Priority: Medium)

#### 5.1 Enhanced Training Pipeline
```python
# backend/app/models/training_pipeline.py - New training pipeline
import asyncio
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from .training import ModelTrainer
from .evaluation import ModelEvaluator
from .metadata import ModelMetadata
from .config import ModelConfig

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.trainer = ModelTrainer(config)
        self.evaluator = ModelEvaluator(config)
        self.metadata_manager = ModelMetadata(config)
    
    async def run_training_pipeline(self, start_date: Optional[datetime] = None,
                                  end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """Run complete training pipeline."""
        pipeline_start = datetime.now()
        
        try:
            logger.info("Starting training pipeline")
            
            # Step 1: Prepare training data
            logger.info("Preparing training data")
            X, y = await self.trainer.prepare_training_data(start_date, end_date)
            
            # Step 2: Train model
            logger.info("Training model")
            training_start = datetime.now()
            model = self.trainer.train_model(X, y)
            training_duration = (datetime.now() - training_start).total_seconds()
            
            # Step 3: Evaluate model
            logger.info("Evaluating model")
            evaluation_results = self.evaluator.evaluate_model(model, X, y)
            
            # Step 4: Check if model meets requirements
            if not self._check_model_requirements(evaluation_results):
                raise ValueError("Model does not meet performance requirements")
            
            # Step 5: Save model and metadata
            logger.info("Saving model and metadata")
            model_path = await self._save_model_with_metadata(
                model, X, evaluation_results, training_duration
            )
            
            # Step 6: Update model registry
            logger.info("Updating model registry")
            await self._update_model_registry(model_path, evaluation_results)
            
            # Step 7: Generate training report
            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
            report = self._generate_training_report(
                model_path, evaluation_results, pipeline_duration
            )
            
            logger.info("Training pipeline completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def _check_model_requirements(self, evaluation_results: Dict[str, Any]) -> bool:
        """Check if model meets performance requirements."""
        threshold_checks = evaluation_results['threshold_checks']
        
        # All required metrics must pass thresholds
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            if metric in threshold_checks and not threshold_checks[metric]:
                logger.warning(f"Model failed {metric} threshold check")
                return False
        
        return True
    
    async def _save_model_with_metadata(self, model: Any, X: np.ndarray,
                                      evaluation_results: Dict[str, Any],
                                      training_duration: float) -> Path:
        """Save model with comprehensive metadata."""
        # Generate version
        version = datetime.now().strftime(self.config.storage.version_format)
        
        # Create model directory
        model_dir = Path(self.config.storage.directory) / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'model.joblib'
        joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = model_dir / 'scaler.joblib'
        joblib.dump(self.trainer.scaler, scaler_path)
        
        # Create and save metadata
        training_config = {
            'training_samples': len(X),
            'training_duration': training_duration,
            'memory_usage': self._get_memory_usage()
        }
        
        metadata = self.metadata_manager.create_metadata(
            model, self.trainer.feature_names, evaluation_results, training_config
        )
        self.metadata_manager.save_metadata(metadata, model_dir)
        
        return model_dir
    
    async def _update_model_registry(self, model_path: Path, 
                                   evaluation_results: Dict[str, Any]) -> None:
        """Update model registry with new model."""
        # This would integrate with the existing ModelManager
        # For now, we'll create a simple registry update
        registry_file = Path(self.config.storage.directory) / 'model_registry.json'
        
        registry = {}
        if registry_file.exists():
            with open(registry_file, 'r') as f:
                registry = json.load(f)
        
        # Add new model to registry
        model_version = model_path.name
        registry[model_version] = {
            'path': str(model_path),
            'created_at': datetime.now().isoformat(),
            'metrics': evaluation_results['basic_metrics'],
            'status': 'available'
        }
        
        # Save updated registry
        with open(registry_file, 'w') as f:
            json.dump(registry, f, indent=2)
    
    def _generate_training_report(self, model_path: Path,
                                evaluation_results: Dict[str, Any],
                                pipeline_duration: float) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        return {
            'pipeline_info': {
                'status': 'completed',
                'duration': pipeline_duration,
                'model_path': str(model_path),
                'created_at': datetime.now().isoformat()
            },
            'model_performance': evaluation_results['basic_metrics'],
            'evaluation_details': evaluation_results['advanced_metrics'],
            'threshold_checks': evaluation_results['threshold_checks'],
            'recommendations': self._generate_recommendations(evaluation_results)
        }
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        metrics = evaluation_results['basic_metrics']
        
        if metrics.get('precision', 0) < 0.7:
            recommendations.append("Consider increasing contamination parameter to reduce false positives")
        
        if metrics.get('recall', 0) < 0.7:
            recommendations.append("Consider decreasing contamination parameter to catch more anomalies")
        
        if metrics.get('f1_score', 0) < 0.7:
            recommendations.append("Model may need hyperparameter tuning for better balance")
        
        return recommendations
```

## Implementation Timeline

### Week 1-2: Export Data Integration
- [ ] Implement export data validation
- [ ] Update ModelTrainer to use exported data
- [ ] Add export file processing
- [ ] Test export data training

### Week 3-4: Enhanced Configuration and Evaluation
- [ ] Update ModelConfig structure
- [ ] Implement comprehensive evaluation system
- [ ] Add cross-validation and threshold checking
- [ ] Test evaluation metrics

### Week 5-6: Metadata and Versioning
- [ ] Implement comprehensive metadata management
- [ ] Add model versioning with hashes
- [ ] Create metadata storage and retrieval
- [ ] Test metadata system

### Week 7-8: Training Pipeline Integration
- [ ] Implement end-to-end training pipeline
- [ ] Add model registry integration
- [ ] Create training reports and recommendations
- [ ] Test complete pipeline

### Week 9-10: Testing and Documentation
- [ ] Comprehensive testing of all components
- [ ] Performance testing on target hardware
- [ ] Update documentation
- [ ] Create deployment guide

## Success Criteria

1. **Functional Requirements**:
   - Training from exported data works correctly
   - Comprehensive evaluation system implemented
   - Model versioning and metadata complete
   - End-to-end training pipeline functional

2. **Integration Requirements**:
   - Seamless integration with export system
   - Proper integration with existing ModelManager
   - Compatible with current deployment process

3. **Quality Requirements**:
   - All tests passing
   - Documentation complete and accurate
   - Error handling robust
   - Logging comprehensive

## Next Steps

1. **Review and Approve**: Review this plan with stakeholders
2. **Resource Allocation**: Assign developers to implementation phases
3. **Environment Setup**: Prepare development and testing environments
4. **Implementation**: Begin Phase 1 implementation
5. **Regular Reviews**: Weekly progress reviews and adjustments

This enhancement plan will transform the current basic training system into a production-ready, comprehensive training pipeline that meets all documented requirements and integrates seamlessly with the existing MCP service architecture. 