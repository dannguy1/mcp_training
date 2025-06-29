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
                           export_files: List[str],
                           model_type: str = "isolation_forest",
                           model_name: Optional[str] = None,
                           config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """Start a training job with multiple export files."""
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
            'export_files': export_files,
            'model_type': model_type,
            'model_name': model_name
        }
        
        # Start training in background
        asyncio.create_task(self._run_training_task(
            training_id, export_files, model_type, model_name, config_overrides
        ))
        
        return training_id
    
    async def _run_training_task(self, 
                                training_id: str,
                                export_files: List[str],
                                model_type: str,
                                model_name: Optional[str],
                                config_overrides: Optional[Dict[str, Any]]):
        """Run training task in background."""
        try:
            # Create progress callback for 5% interval updates
            async def progress_callback(progress: int, step: str):
                await self._update_progress(training_id, progress, step)
            
            # Step 1: Validate export data
            await self._update_progress(training_id, 5, 'Validating export data')
            validation_results = await self.training_pipeline.validate_export_for_training(export_files[0])
            
            if not validation_results['is_valid']:
                await self._update_progress(training_id, 0, 'Validation failed', 
                                          error='; '.join(validation_results['errors']))
                return
            
            # Step 2: Run comprehensive training pipeline with progress callback
            await self._update_progress(training_id, 10, 'Starting training pipeline')
            training_result = await self.training_pipeline.run_training_pipeline(
                export_file_paths=export_files,
                model_type=model_type,
                model_name=model_name,
                training_id=training_id,
                progress_callback=progress_callback
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
            export_files=[export_file],
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
            
            # Broadcast progress update via WebSocket
            try:
                from ..api.routes.websocket import broadcast_training_update
                
                # Create async task to broadcast with timeout
                async def broadcast():
                    try:
                        status = 'running'
                        if error:
                            status = 'failed'
                        elif result:
                            status = 'completed'
                        
                        await asyncio.wait_for(
                            broadcast_training_update(
                                job_id=training_id,
                                progress=progress,
                                status=status,
                                step=step,
                                error=error,
                                result=result
                            ),
                            timeout=5.0  # 5 second timeout
                        )
                    except asyncio.TimeoutError:
                        logger.warning(f"WebSocket broadcast timeout for training {training_id}")
                    except Exception as e:
                        logger.error(f"Failed to broadcast training update: {e}")
                
                # Run in event loop if available
                try:
                    loop = asyncio.get_running_loop()
                    # Schedule the broadcast task without blocking
                    loop.create_task(broadcast())
                except RuntimeError:
                    # No running event loop, try to get the current one
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(broadcast())
                        else:
                            # Create a new task in the event loop with timeout
                            future = asyncio.run_coroutine_threadsafe(broadcast(), loop)
                            try:
                                future.result(timeout=5)  # Wait up to 5 seconds
                            except Exception as e:
                                logger.warning(f"WebSocket broadcast failed: {e}")
                    except Exception as e:
                        logger.warning(f"Could not broadcast training update: {e}")
                        
            except ImportError:
                logger.warning("WebSocket broadcasting not available")
            except Exception as e:
                logger.error(f"Failed to broadcast training update: {e}")
    
    def get_training_status(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Get training job status."""
        # First check in-memory tasks
        task = self.training_tasks.get(training_id)
        if task:
            return task
        
        # If not found in memory, check model registry for completed jobs
        return self._load_completed_job_from_registry(training_id)
    
    def _get_evaluation_summary_from_registry(self, version: str) -> Dict[str, Any]:
        """Get evaluation summary from model registry."""
        try:
            registry_file = Path(self.config.storage.directory) / 'model_registry.json'
            if registry_file.exists():
                with open(registry_file, 'r') as f:
                    registry = json.load(f)
                
                if version in registry:
                    return registry[version].get('evaluation_summary', {})
            return {}
        except Exception as e:
            logger.error(f"Error getting evaluation summary from registry: {e}")
            return {}
    
    def _load_completed_job_from_registry(self, training_id: str) -> Optional[Dict[str, Any]]:
        """Load completed job information from model registry."""
        try:
            # Search through all models to find one with matching training_id
            models = self.model_registry.list_models()
            for model in models:
                version = model['version']
                metadata = self.model_registry.get_model(version)
                if metadata:
                    # Check both model_info.training_id and training_info.training_id for robustness
                    model_info_tid = getattr(metadata.model_info, 'training_id', None)
                    training_info_tid = getattr(metadata.training_info, 'training_id', None)
                    if model_info_tid == training_id or training_info_tid == training_id:
                        return {
                            'id': training_id,
                            'status': 'completed',
                            'progress': 100,
                            'step': 'Training completed',
                            'error': None,
                            'result': {
                                'model_version': version,
                                'model_type': metadata.model_info.model_type,
                                'training_samples': metadata.training_info.training_samples,
                                'evaluation_results': metadata.evaluation_info.basic_metrics if hasattr(metadata, 'evaluation_info') else None,
                                'export_files': metadata.model_info.export_files if hasattr(metadata.model_info, 'export_files') else None,
                                'training_duration': metadata.training_info.training_duration if hasattr(metadata.training_info, 'training_duration') else None,
                                'feature_names': metadata.training_info.feature_names if hasattr(metadata.training_info, 'feature_names') else None,
                                'export_files_size': metadata.training_info.export_files_size if hasattr(metadata.training_info, 'export_files_size') else None,
                                'model_parameters': metadata.training_info.model_parameters if hasattr(metadata.training_info, 'model_parameters') else None
                            },
                            'start_time': metadata.model_info.created_at,
                            'updated_at': metadata.model_info.created_at,
                            'export_files': metadata.model_info.export_files if hasattr(metadata.model_info, 'export_files') else None,
                            'model_type': metadata.model_info.model_type,
                            'model_name': None,
                            # Add comprehensive statistics
                            'comprehensive_stats': {
                                'training_info': {
                                    'samples': metadata.training_info.training_samples,
                                    'features': len(metadata.training_info.feature_names) if hasattr(metadata.training_info, 'feature_names') else 0,
                                    'feature_names': metadata.training_info.feature_names if hasattr(metadata.training_info, 'feature_names') else [],
                                    'duration_seconds': metadata.training_info.training_duration if hasattr(metadata.training_info, 'training_duration') else 0,
                                    'export_size_mb': round(metadata.training_info.export_file_size / 1024 / 1024, 2) if hasattr(metadata.training_info, 'export_file_size') else 0,
                                    'model_parameters': metadata.training_info.model_parameters if hasattr(metadata.training_info, 'model_parameters') else {}
                                },
                                'evaluation_summary': self._get_evaluation_summary_from_registry(version),
                                'performance_metrics': metadata.evaluation_info.basic_metrics if hasattr(metadata, 'evaluation_info') else {}
                            }
                        }
            return None
        except Exception as e:
            logger.error(f"Error loading completed job from registry: {e}")
            return None
    
    def list_training_tasks(self) -> Dict[str, Dict[str, Any]]:
        """List all training tasks (including completed ones from registry)."""
        # Start with in-memory tasks
        all_tasks = self.training_tasks.copy()
        
        # Add completed jobs from registry
        try:
            models = self.model_registry.list_models()
            for model in models:
                version = model['version']
                metadata = self.model_registry.get_model(version)
                if metadata:
                    # Get training_id from metadata
                    model_info_tid = getattr(metadata.model_info, 'training_id', None)
                    training_info_tid = getattr(metadata.training_info, 'training_id', None)
                    training_id = model_info_tid or training_info_tid
                    
                    if training_id and training_id not in all_tasks:
                        completed_job = self._load_completed_job_from_registry(training_id)
                        if completed_job:
                            all_tasks[training_id] = completed_job
        except Exception as e:
            logger.error(f"Error loading completed jobs from registry: {e}")
        
        return all_tasks
    
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
    
    async def delete_training_job(self, training_id: str) -> bool:
        """Delete a training job and its associated data.
        
        Args:
            training_id: Training job ID to delete
            
        Returns:
            True if job was deleted successfully, False otherwise
        """
        try:
            logger.info(f"Attempting to delete training job: {training_id}")
            
            # Check if job exists in memory
            if training_id in self.training_tasks:
                task = self.training_tasks[training_id]
                
                # Don't allow deletion of running jobs
                if task['status'] == 'running':
                    logger.warning(f"Cannot delete running training job: {training_id}")
                    return False
                
                # Remove from memory
                del self.training_tasks[training_id]
                logger.info(f"Removed training job from memory: {training_id}")
            
            # Check if job exists in registry (completed jobs)
            try:
                models = self.model_registry.list_models()
                for model in models:
                    version = model['version']
                    metadata = self.model_registry.get_model(version)
                    if metadata:
                        # Check both model_info.training_id and training_info.training_id
                        model_info_tid = getattr(metadata.model_info, 'training_id', None)
                        training_info_tid = getattr(metadata.training_info, 'training_id', None)
                        
                        if model_info_tid == training_id or training_info_tid == training_id:
                            # Delete the model from registry
                            success = self.model_registry.delete_model(version)
                            if success:
                                logger.info(f"Deleted completed training job from registry: {training_id}")
                                return True
                            else:
                                logger.error(f"Failed to delete model from registry: {version}")
                                return False
                
                # If we get here, the job wasn't found in registry
                logger.info(f"Training job not found in registry: {training_id}")
                return True  # Consider it a success if not found
                
            except Exception as e:
                logger.error(f"Error checking registry for training job {training_id}: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error deleting training job {training_id}: {e}")
            return False 