"""
Model registry for MCP Training Service.
"""

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import shutil
import logging
from datetime import datetime

from .metadata import ModelMetadata

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for managing trained models."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model registry."""
        # Get project root (assuming this is the directory containing the main.py file)
        project_root = Path(__file__).parent.parent.parent.parent
        
        # Convert relative path to absolute path
        self.models_dir = project_root / models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._models_cache: Optional[Dict[str, ModelMetadata]] = None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models."""
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                # Skip non-model directories
                if model_dir.name in ['deployments', '.git', '__pycache__']:
                    continue
                
                # Check if directory contains model metadata
                metadata_file = model_dir / "metadata.json"
                if not metadata_file.exists():
                    logger.debug(f"Skipping directory without metadata: {model_dir}")
                    continue
                
                try:
                    metadata = ModelMetadata.load(model_dir)
                    models.append({
                        'version': metadata.model_info.version,
                        'model_type': metadata.model_info.model_type,
                        'created_at': metadata.model_info.created_at,
                        'training_samples': metadata.training_info.training_samples,
                        'deployment_status': metadata.deployment_info.status,
                        'path': str(model_dir)
                    })
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {model_dir}: {e}")
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return models
    
    def get_model(self, version: str) -> Optional[ModelMetadata]:
        """Get model metadata by version."""
        model_dir = self.models_dir / version
        if not model_dir.exists():
            return None
        
        try:
            return ModelMetadata.load(model_dir)
        except Exception as e:
            logger.error(f"Failed to load model {version}: {e}")
            return None
    
    def save_model(self, 
                  version: str,
                  model_metadata: ModelMetadata,
                  model_file: Path,
                  scaler_file: Optional[Path] = None) -> Path:
        """Save a trained model."""
        model_dir = self.models_dir / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        target_model_path = model_dir / "model.joblib"
        shutil.copy2(model_file, target_model_path)
        
        # Save scaler if provided
        if scaler_file and scaler_file.exists():
            target_scaler_path = model_dir / "scaler.joblib"
            shutil.copy2(scaler_file, target_scaler_path)
        
        # Save metadata
        model_metadata.save(model_dir)
        
        # Clear cache
        self._models_cache = None
        
        logger.info(f"Model saved: {version}")
        return model_dir
    
    def delete_model(self, version: str) -> bool:
        """Delete a model."""
        model_dir = self.models_dir / version
        if not model_dir.exists():
            logger.warning(f"Model not found: {version}")
            return False
        
        try:
            shutil.rmtree(model_dir)
            self._models_cache = None
            logger.info(f"Model deleted: {version}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {version}: {e}")
            return False
    
    def deploy_model(self, version: str, deployed_by: Optional[str] = None) -> bool:
        """Deploy a model (mark as deployed)."""
        model_dir = self.models_dir / version
        if not model_dir.exists():
            logger.warning(f"Model not found: {version}")
            return False
        
        try:
            metadata = ModelMetadata.load(model_dir)
            metadata.deploy(deployed_by)
            metadata.save(model_dir)
            self._models_cache = None
            logger.info(f"Model deployed: {version}")
            return True
        except Exception as e:
            logger.error(f"Failed to deploy model {version}: {e}")
            return False
    
    def get_latest_model(self) -> Optional[ModelMetadata]:
        """Get the latest trained model."""
        models = self.list_models()
        if not models:
            return None
        
        latest_version = models[0]['version']
        return self.get_model(latest_version)
    
    def get_deployed_model(self) -> Optional[ModelMetadata]:
        """Get the currently deployed model."""
        models = self.list_models()
        for model in models:
            if model['deployment_status'] == 'deployed':
                return self.get_model(model['version'])
        return None
    
    def model_exists(self, version: str) -> bool:
        """Check if a model exists."""
        model_dir = self.models_dir / version
        return model_dir.exists()
    
    def get_model_path(self, version: str) -> Optional[Path]:
        """Get the path to a model."""
        model_dir = self.models_dir / version
        if not model_dir.exists():
            return None
        
        model_file = model_dir / "model.joblib"
        return model_file if model_file.exists() else None
    
    def get_scaler_path(self, version: str) -> Optional[Path]:
        """Get the path to a model's scaler."""
        model_dir = self.models_dir / version
        if not model_dir.exists():
            return None
        
        scaler_file = model_dir / "scaler.joblib"
        return scaler_file if scaler_file.exists() else None
    
    def cleanup_old_models(self, keep_last_n: int = 5) -> List[str]:
        """Clean up old models, keeping only the last N."""
        models = self.list_models()
        if len(models) <= keep_last_n:
            return []
        
        models_to_delete = models[keep_last_n:]
        deleted_models = []
        
        for model in models_to_delete:
            version = model['version']
            if self.delete_model(version):
                deleted_models.append(version)
        
        logger.info(f"Cleaned up {len(deleted_models)} old models")
        return deleted_models
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about stored models."""
        models = self.list_models()
        
        stats = {
            'total_models': len(models),
            'deployed_models': len([m for m in models if m['deployment_status'] == 'deployed']),
            'available_models': len([m for m in models if m['deployment_status'] == 'available']),
            'model_types': {},
            'total_training_samples': 0,
            'oldest_model': None,
            'newest_model': None
        }
        
        if models:
            # Model type distribution
            for model in models:
                model_type = model['model_type']
                stats['model_types'][model_type] = stats['model_types'].get(model_type, 0) + 1
            
            # Training samples
            for model in models:
                stats['total_training_samples'] += model['training_samples']
            
            # Date range
            dates = [model['created_at'] for model in models]
            stats['oldest_model'] = min(dates)
            stats['newest_model'] = max(dates)
        
        return stats 