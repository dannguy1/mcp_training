"""
Dependency injection functions for FastAPI.
"""

from typing import Optional
from .training_service import TrainingService
from .model_service import ModelService
from .storage_service import StorageService

# Global service instances (will be initialized by app.py)
_training_service: Optional[TrainingService] = None
_model_service: Optional[ModelService] = None
_storage_service: Optional[StorageService] = None

def set_services(training_service: TrainingService, model_service: ModelService, storage_service: StorageService):
    """Set the global service instances. Called by app.py during startup."""
    global _training_service, _model_service, _storage_service
    _training_service = training_service
    _model_service = model_service
    _storage_service = storage_service

def get_training_service() -> TrainingService:
    """Get training service instance."""
    if _training_service is None:
        raise RuntimeError("Training service not initialized. Make sure the app has started.")
    return _training_service

def get_model_service() -> ModelService:
    """Get model service instance."""
    if _model_service is None:
        raise RuntimeError("Model service not initialized. Make sure the app has started.")
    return _model_service

def get_storage_service() -> StorageService:
    """Get storage service instance."""
    if _storage_service is None:
        raise RuntimeError("Storage service not initialized. Make sure the app has started.")
    return _storage_service 