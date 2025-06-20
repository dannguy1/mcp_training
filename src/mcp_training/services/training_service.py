"""
Training service for MCP Training Service.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
import uuid
import json

from ..core.feature_extractor import WiFiFeatureExtractor
from ..core.model_trainer import ModelTrainer
from ..core.export_validator import ExportValidator
from ..models.config import ModelConfig
from ..models.metadata import ModelMetadata
from ..models.registry import ModelRegistry
from ..models.evaluation import ModelEvaluator
from ..models.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class TrainingService:
    """Service for managing training operations."""
    
    def __init__(self, config: Optional[ModelConfig] = None, model_service=None, storage_service=None):
        """Initialize training service."""
        self.config = config or ModelConfig()
        self.model_service = model_service
        self.storage_service = storage_service
        self.feature_extractor = WiFiFeatureExtractor()
        self.model_trainer = ModelTrainer()
        self.export_validator = ExportValidator()
        self.model_registry = ModelRegistry(self.config.storage.directory) if self.config else None
        self.model_evaluator = ModelEvaluator(self.config)
        self.training_pipeline = TrainingPipeline(self.config)
        self.training_tasks: Dict[str, Dict[str, Any]] = {}
    
    async def start_training(self, 
                           export_file: str,
                           model_type: str = "isolation_forest",
                           model_name: Optional[str] = None,
                           config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """Start a training job."""
        training_id = str(uuid.uuid4())
        
        # Initialize training task
        self.training_tasks[training_id] = {
            'id': training_id,
            'status': 'initializing',
            'progress': 0,
            'step': 'Validating export data',
            'error': None,
            'result': None,
            'start_time': datetime.now().isoformat(),
            'export_file': export_file,
            'model_type': model_type,
            'model_name': model_name
        }
        
        # Start training in background
        asyncio.create_task(self._run_training_task(
            training_id, export_file, model_type, model_name, config_overrides
        ))
        
        return training_id
    
    async def _run_training_task(self, 
                                training_id: str,
                                export_file: str,
                                model_type: str,
                                model_name: Optional[str],
                                config_overrides: Optional[Dict[str, Any]]):
        """Run training task in background."""
        try:
            # Step 1: Validate export data
            await self._update_progress(training_id, 5, 'Validating export data')
            validation_results = await self.training_pipeline.validate_export_for_training(export_file)
            
            if not validation_results['is_valid']:
                await self._update_progress(training_id, 0, 'Validation failed', 
                                          error='; '.join(validation_results['errors']))
                return
            
            # Step 2: Run comprehensive training pipeline
            await self._update_progress(training_id, 10, 'Starting training pipeline')
            training_result = await self.training_pipeline.run_training_pipeline(
                export_file_path=export_file,
                model_type=model_type,
                model_name=model_name
            )
            
            # Step 3: Complete
            await self._update_progress(training_id, 100, 'Training completed', 
                                      result=training_result)
            
            logger.info(f"Training completed successfully: {training_id}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            await self._update_progress(training_id, 0, 'Training failed', error=str(e))
    
    async def _load_exported_data(self, export_file: str) -> Dict[str, Any]:
        """Load data from exported JSON file."""
        try:
            with open(export_file, 'r') as f:
                data = json.load(f)
            
            # Validate export data structure
            if 'data' not in data:
                raise ValueError("Export file must contain 'data' section")
            
            logger.info(f"Loaded {len(data['data'])} log entries from export file")
            return data
            
        except Exception as e:
            logger.error(f"Error loading exported data: {e}")
            raise
    
    def _prepare_training_data(self, features: Dict[str, Any]) -> tuple:
        """Prepare training data from extracted features."""
        # Convert features to feature matrix
        feature_matrix = []
        for feature_name in self.config.features.numeric:
            if feature_name in features:
                feature_matrix.append(float(features[feature_name]))
            else:
                feature_matrix.append(0.0)
        
        import numpy as np
        X = np.array([feature_matrix])
        y = np.zeros(len(X))  # Unsupervised learning
        
        return X, y
    
    async def _save_model_with_metadata(self, 
                                      model: Any, 
                                      features: Dict[str, Any],
                                      evaluation_results: Dict[str, Any],
                                      export_file: str,
                                      training_id: str,
                                      model_type: str) -> Path:
        """Save model with comprehensive metadata."""
        # Generate version
        version = datetime.now().strftime(self.config.storage.version_format)
        
        # Create model metadata
        metadata = ModelMetadata.create(
            version=version,
            model_type=model_type,
            training_samples=len(features),
            feature_names=list(features.keys()),
            export_file=export_file,
            training_id=training_id,
            model_parameters=self.config.model.dict()
        )
        
        # Update evaluation results
        metadata.update_evaluation(evaluation_results)
        metadata.update_export_file_size(Path(export_file).stat().st_size)
        metadata.update_training_duration(self._get_training_duration(training_id))
        
        # Save model files
        import joblib
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Save model
            model_file = temp_path / "model.joblib"
            joblib.dump(model, model_file)
            
            # Save scaler if available
            scaler_file = None
            if hasattr(self.model_trainer, 'scaler') and self.model_trainer.scaler:
                scaler_file = temp_path / "scaler.joblib"
                joblib.dump(self.model_trainer.scaler, scaler_file)
            
            # Save to registry
            model_path = self.model_registry.save_model(
                version, metadata, model_file, scaler_file
            )
        
        return model_path
    
    async def _update_progress(self, training_id: str, progress: int, step: str,
                             error: Optional[str] = None, result: Optional[Dict] = None):
        """Update training progress."""
        if training_id in self.training_tasks:
            self.training_tasks[training_id].update({
                'progress': progress,
                'step': step,
                'updated_at': datetime.now().isoformat()
            })
            
            if error:
                self.training_tasks[training_id].update({
                    'status': 'failed',
                    'error': error
                })
            elif result:
                self.training_tasks[training_id].update({
                    'status': 'completed',
                    'result': result
                })
            else:
                self.training_tasks[training_id]['status'] = 'running'
    
    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status."""
        return self.training_tasks.get(training_id)
    
    def list_training_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all training tasks."""
        return self.training_tasks
    
    def _get_training_duration(self, training_id: str) -> float:
        """Get training duration for a task."""
        task = self.training_tasks.get(training_id)
        if task and 'start_time' in task:
            start_time = datetime.fromisoformat(task['start_time'])
            end_time = datetime.now()
            return (end_time - start_time).total_seconds()
        return 0.0
    
    async def validate_export(self, export_file: str) -> Dict[str, Any]:
        """Validate export file for training."""
        return await self.training_pipeline.validate_export_for_training(export_file)
    
    def get_model_registry(self) -> ModelRegistry:
        """Get model registry."""
        return self.model_registry
    
    async def shutdown(self):
        """Shutdown the training service."""
        # Cancel any running tasks
        for task_id, task_info in self.training_tasks.items():
            if task_info['status'] == 'running':
                task_info['status'] = 'cancelled'
                task_info['error'] = 'Service shutdown'
        
        logger.info("Training service shutdown complete") 