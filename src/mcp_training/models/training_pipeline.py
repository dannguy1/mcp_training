"""
End-to-end training pipeline for MCP Training Service.
"""

import asyncio
import logging
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path

from ..core.model_trainer import ModelTrainer
from .evaluation import ModelEvaluator
from .metadata import ModelMetadata
from .config import ModelConfig
from .registry import ModelRegistry

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the training pipeline."""
        self.config = config
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator(config)
        self.metadata_manager = ModelMetadata
        self.registry = ModelRegistry(config.storage.directory)
    
    async def run_training_pipeline(self, 
                                  export_file_path: str,
                                  model_type: str = "isolation_forest",
                                  model_name: Optional[str] = None) -> Dict[str, Any]:
        """Run complete training pipeline."""
        pipeline_start = datetime.now()
        
        try:
            logger.info("Starting training pipeline")
            
            # Step 1: Load and validate export data
            logger.info("Loading and validating export data")
            exported_data = await self._load_exported_data(export_file_path)
            
            # Step 2: Extract features
            logger.info("Extracting features")
            features_df = self.trainer.feature_extractor.extract_features(exported_data['data'])
            features_df = self.trainer._handle_missing_values(features_df)
            
            # Step 3: Prepare training data
            logger.info("Preparing training data")
            X = self.trainer.scaler.fit_transform(features_df)
            y = np.zeros(len(X))  # Unsupervised learning
            
            # Step 4: Train model
            logger.info("Training model")
            training_start = datetime.now()
            self.trainer.model = self.trainer._create_model()
            self.trainer.model.fit(X)
            training_duration = (datetime.now() - training_start).total_seconds()
            
            # Step 5: Evaluate model
            logger.info("Evaluating model")
            evaluation_results = self.evaluator.evaluate_model(self.trainer.model, X, y)
            
            # Step 6: Check if model meets requirements
            logger.info("Checking model requirements")
            if not self._check_model_requirements(evaluation_results):
                raise ValueError("Model does not meet performance requirements")
            
            # Step 7: Save model and metadata
            logger.info("Saving model and metadata")
            model_path = await self._save_model_with_metadata(
                self.trainer.model, X, evaluation_results, training_duration, 
                export_file_path, model_type, features_df.columns.tolist()
            )
            
            # Step 8: Update model registry
            logger.info("Updating model registry")
            await self._update_model_registry(model_path, evaluation_results)
            
            # Step 9: Generate training report
            pipeline_duration = (datetime.now() - pipeline_start).total_seconds()
            report = self._generate_training_report(
                model_path, evaluation_results, pipeline_duration, features_df.columns.tolist()
            )
            
            logger.info("Training pipeline completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    async def _load_exported_data(self, export_file_path: str) -> Dict[str, Any]:
        """Load data from exported JSON file."""
        try:
            with open(export_file_path, 'r') as f:
                data = json.load(f)
            
            # Validate export data structure
            if 'data' not in data:
                raise ValueError("Export file must contain 'data' section")
            
            logger.info(f"Loaded {len(data['data'])} log entries from export file")
            return data
            
        except Exception as e:
            logger.error(f"Error loading exported data: {e}")
            raise
    
    def _check_model_requirements(self, evaluation_results: Dict[str, Any]) -> bool:
        """Check if model meets performance requirements."""
        threshold_checks = evaluation_results.get('threshold_checks', {})
        
        # All required metrics must pass thresholds
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            if metric in threshold_checks and not threshold_checks[metric]:
                logger.warning(f"Model failed {metric} threshold check")
                return False
        
        # Check overall performance
        basic_metrics = evaluation_results.get('basic_metrics', {})
        if basic_metrics.get('roc_auc', 0) < 0.5:  # Minimum ROC AUC
            logger.warning("Model ROC AUC below minimum threshold")
            return False
        
        return True
    
    async def _save_model_with_metadata(self, 
                                      model: Any, 
                                      X: np.ndarray,
                                      evaluation_results: Dict[str, Any],
                                      training_duration: float,
                                      export_file_path: str,
                                      model_type: str,
                                      feature_names: List[str]) -> Path:
        """Save model with comprehensive metadata."""
        # Generate version
        version = datetime.now().strftime(self.config.storage.version_format)
        
        # Create model metadata
        metadata = self.metadata_manager.create(
            version=version,
            model_type=model_type,
            training_samples=len(X),
            feature_names=feature_names,
            export_file=export_file_path,
            training_id=None,  # Will be set by training service
            model_parameters=self.config.model.dict()
        )
        
        # Update evaluation results
        metadata.update_evaluation(evaluation_results)
        metadata.update_training_duration(training_duration)
        metadata.update_export_file_size(Path(export_file_path).stat().st_size)
        
        # Save model files
        model_dir = Path(self.config.storage.directory) / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_file = model_dir / "model.joblib"
        joblib.dump(model, model_file)
        
        # Save scaler
        scaler_file = model_dir / "scaler.joblib"
        joblib.dump(self.trainer.scaler, scaler_file)
        
        # Save metadata
        metadata.save(model_dir)
        
        return model_dir
    
    async def _update_model_registry(self, model_path: Path, 
                                   evaluation_results: Dict[str, Any]) -> None:
        """Update model registry with new model."""
        try:
            # Load existing registry
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
                'metrics': evaluation_results.get('basic_metrics', {}),
                'status': 'available',
                'evaluation_summary': self.evaluator.get_evaluation_summary()
            }
            
            # Save updated registry
            with open(registry_file, 'w') as f:
                json.dump(registry, f, indent=2)
                
            logger.info(f"Model registry updated with version {model_version}")
            
        except Exception as e:
            logger.error(f"Error updating model registry: {e}")
            # Don't fail the pipeline for registry update errors
    
    def _generate_training_report(self, 
                                model_path: Path,
                                evaluation_results: Dict[str, Any],
                                pipeline_duration: float,
                                feature_names: List[str]) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        evaluation_summary = self.evaluator.get_evaluation_summary()
        
        return {
            'pipeline_info': {
                'status': 'completed',
                'duration': pipeline_duration,
                'model_path': str(model_path),
                'created_at': datetime.now().isoformat()
            },
            'model_info': {
                'model_type': type(self.trainer.model).__name__,
                'feature_count': len(feature_names),
                'feature_names': feature_names,
                'training_samples': evaluation_results.get('basic_metrics', {}).get('training_samples', 0)
            },
            'model_performance': evaluation_results.get('basic_metrics', {}),
            'evaluation_details': evaluation_results.get('advanced_metrics', {}),
            'threshold_checks': evaluation_results.get('threshold_checks', {}),
            'evaluation_summary': evaluation_summary,
            'recommendations': evaluation_summary.get('recommendations', [])
        }
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return {
            'config': {
                'model_type': self.config.model.dict(),
                'evaluation_thresholds': self.config.evaluation.thresholds,
                'storage_directory': self.config.storage.directory
            },
            'registry_stats': self.registry.get_model_stats() if self.registry else {}
        }
    
    async def validate_export_for_training(self, export_file_path: str) -> Dict[str, Any]:
        """Validate that export file is suitable for training."""
        try:
            # Load export data
            exported_data = await self._load_exported_data(export_file_path)
            
            validation_results = {
                'is_valid': True,
                'errors': [],
                'warnings': [],
                'stats': {}
            }
            
            # Check data structure
            if 'data' not in exported_data:
                validation_results['is_valid'] = False
                validation_results['errors'].append("Missing 'data' section in export file")
            
            # Check data quality
            if 'data' in exported_data:
                data = exported_data['data']
                validation_results['stats']['total_records'] = len(data)
                
                # Check for WiFi-related logs
                wifi_logs = [log for log in data if log.get('process_name') in ['hostapd', 'wpa_supplicant']]
                validation_results['stats']['wifi_logs'] = len(wifi_logs)
                validation_results['stats']['wifi_ratio'] = len(wifi_logs) / len(data) if data else 0
                
                if validation_results['stats']['wifi_ratio'] < 0.1:
                    validation_results['warnings'].append("Low ratio of WiFi-related logs (< 10%)")
                
                # Check time range
                if data:
                    timestamps = [log.get('timestamp') for log in data if log.get('timestamp')]
                    if timestamps:
                        try:
                            times = [pd.to_datetime(ts) for ts in timestamps]
                            time_range = max(times) - min(times)
                            validation_results['stats']['time_range_days'] = time_range.days
                            
                            if time_range.days < 1:
                                validation_results['warnings'].append("Export contains less than 1 day of data")
                        except:
                            pass
            
            # Determine overall validity
            if validation_results['errors']:
                validation_results['is_valid'] = False
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating export file: {e}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'stats': {}
            } 