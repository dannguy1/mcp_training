"""
Model service for MCP Training Service.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import joblib
import numpy as np
from datetime import datetime

from ..models.registry import ModelRegistry
from ..models.metadata import ModelMetadata
from ..utils.logger import get_logger
from ..utils.file_utils import ensure_directory, copy_file, delete_file

logger = get_logger(__name__)


class ModelService:
    """Service for managing model operations."""
    
    def __init__(self, models_dir: str = "models"):
        """Initialize model service.
        
        Args:
            models_dir: Directory for storing models
        """
        self.models_dir = Path(models_dir)
        self.registry = ModelRegistry(models_dir)
        ensure_directory(self.models_dir)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            return self.registry.list_models()
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_model(self, version: str) -> Optional[ModelMetadata]:
        """Get model metadata by version.
        
        Args:
            version: Model version
            
        Returns:
            Model metadata or None if not found
        """
        try:
            return self.registry.get_model(version)
        except Exception as e:
            logger.error(f"Error getting model {version}: {e}")
            return None
    
    def load_model(self, version: str) -> Tuple[Optional[Any], Optional[Any]]:
        """Load a trained model and its scaler.
        
        Args:
            version: Model version
            
        Returns:
            Tuple of (model, scaler) or (None, None) if error
        """
        try:
            model_path = self.registry.get_model_path(version)
            scaler_path = self.registry.get_scaler_path(version)
            
            if not model_path:
                logger.error(f"Model file not found for version: {version}")
                return None, None
            
            # Load model
            model = joblib.load(model_path)
            
            # Load scaler if available
            scaler = None
            if scaler_path:
                scaler = joblib.load(scaler_path)
            
            logger.info(f"Model {version} loaded successfully")
            return model, scaler
            
        except Exception as e:
            logger.error(f"Error loading model {version}: {e}")
            return None, None
    
    def save_model(
        self,
        model: Any,
        scaler: Optional[Any],
        metadata: ModelMetadata,
        model_file: Path,
        scaler_file: Optional[Path] = None
    ) -> Optional[Path]:
        """Save a trained model with metadata.
        
        Args:
            model: Trained model object
            scaler: Feature scaler object
            metadata: Model metadata
            model_file: Path to model file
            scaler_file: Path to scaler file
            
        Returns:
            Path to saved model directory or None if error
        """
        try:
            # Save to registry
            model_dir = self.registry.save_model(
                metadata.model_info.version,
                metadata,
                model_file,
                scaler_file
            )
            
            logger.info(f"Model saved successfully: {model_dir}")
            return model_dir
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return None
    
    def delete_model(self, version: str) -> bool:
        """Delete a model.
        
        Args:
            version: Model version to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.registry.delete_model(version)
            if success:
                logger.info(f"Model {version} deleted successfully")
            return success
        except Exception as e:
            logger.error(f"Error deleting model {version}: {e}")
            return False
    
    def deploy_model(self, version: str, deployed_by: Optional[str] = None) -> bool:
        """Deploy a model (mark as deployed).
        
        Args:
            version: Model version to deploy
            deployed_by: User deploying the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.registry.deploy_model(version, deployed_by)
            if success:
                logger.info(f"Model {version} deployed successfully")
                
                # Broadcast model ready notification via WebSocket
                try:
                    import asyncio
                    from ..api.routes.websocket import broadcast_model_ready
                    
                    # Create async task to broadcast
                    async def broadcast():
                        await broadcast_model_ready(
                            model_id=version,
                            deployed_by=deployed_by,
                            deployed_at=datetime.now().isoformat()
                        )
                    
                    # Run in event loop if available
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(broadcast())
                        else:
                            loop.run_until_complete(broadcast())
                    except RuntimeError:
                        # No event loop, skip broadcasting
                        pass
                        
                except ImportError:
                    logger.warning("WebSocket broadcasting not available")
                except Exception as e:
                    logger.error(f"Failed to broadcast model ready: {e}")
                    
            return success
        except Exception as e:
            logger.error(f"Error deploying model {version}: {e}")
            return False
    
    def get_latest_model(self) -> Optional[ModelMetadata]:
        """Get the latest trained model.
        
        Returns:
            Latest model metadata or None if no models exist
        """
        try:
            return self.registry.get_latest_model()
        except Exception as e:
            logger.error(f"Error getting latest model: {e}")
            return None
    
    def get_deployed_model(self) -> Optional[ModelMetadata]:
        """Get the currently deployed model.
        
        Returns:
            Deployed model metadata or None if no deployed model
        """
        try:
            return self.registry.get_deployed_model()
        except Exception as e:
            logger.error(f"Error getting deployed model: {e}")
            return None
    
    def predict(
        self,
        model_version: str,
        features: np.ndarray,
        threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Make predictions using a trained model.
        
        Args:
            model_version: Model version to use
            features: Feature array for prediction
            threshold: Anomaly threshold (optional)
            
        Returns:
            Prediction results dictionary
        """
        try:
            # Load model and scaler
            model, scaler = self.load_model(model_version)
            if model is None:
                raise ValueError(f"Failed to load model {model_version}")
            
            # Scale features if scaler is available
            if scaler is not None:
                features_scaled = scaler.transform(features)
            else:
                features_scaled = features
            
            # Make predictions
            scores = -model.score_samples(features_scaled)
            
            # Determine threshold if not provided
            if threshold is None:
                threshold = np.percentile(scores, 90)  # 90th percentile
            
            # Classify anomalies
            predictions = (scores > threshold).astype(int)
            
            # Calculate statistics
            result = {
                'model_version': model_version,
                'total_samples': len(features),
                'normal_count': int(np.sum(predictions == 0)),
                'anomaly_count': int(np.sum(predictions == 1)),
                'anomaly_rate': float(np.mean(predictions)),
                'average_score': float(np.mean(scores)),
                'score_std': float(np.std(scores)),
                'threshold': float(threshold),
                'scores': scores.tolist(),
                'predictions': predictions.tolist()
            }
            
            logger.info(f"Predictions completed for {len(features)} samples")
            return result
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def evaluate_model(
        self,
        model_version: str,
        test_features: np.ndarray,
        test_labels: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Evaluate a trained model.
        
        Args:
            model_version: Model version to evaluate
            test_features: Test feature array
            test_labels: Test labels (optional, for supervised evaluation)
            
        Returns:
            Evaluation results dictionary
        """
        try:
            # Load model and scaler
            model, scaler = self.load_model(model_version)
            if model is None:
                raise ValueError(f"Failed to load model {model_version}")
            
            # Scale features if scaler is available
            if scaler is not None:
                test_features_scaled = scaler.transform(test_features)
            else:
                test_features_scaled = test_features
            
            # Get anomaly scores
            scores = -model.score_samples(test_features_scaled)
            
            # Calculate basic statistics
            evaluation = {
                'model_version': model_version,
                'test_samples': len(test_features),
                'score_statistics': {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'percentiles': {
                        '25': float(np.percentile(scores, 25)),
                        '50': float(np.percentile(scores, 50)),
                        '75': float(np.percentile(scores, 75)),
                        '90': float(np.percentile(scores, 90)),
                        '95': float(np.percentile(scores, 95))
                    }
                }
            }
            
            # Add supervised metrics if labels are provided
            if test_labels is not None:
                # For anomaly detection, we assume -1 is anomaly, 1 is normal
                # Convert to binary: -1 -> 1 (anomaly), 1 -> 0 (normal)
                binary_labels = (test_labels == -1).astype(int)
                
                # Use 90th percentile as threshold
                threshold = np.percentile(scores, 90)
                binary_predictions = (scores > threshold).astype(int)
                
                # Calculate metrics
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                
                evaluation['supervised_metrics'] = {
                    'accuracy': float(accuracy_score(binary_labels, binary_predictions)),
                    'precision': float(precision_score(binary_labels, binary_predictions, zero_division=0)),
                    'recall': float(recall_score(binary_labels, binary_predictions, zero_division=0)),
                    'f1_score': float(f1_score(binary_labels, binary_predictions, zero_division=0)),
                    'threshold': float(threshold)
                }
            
            logger.info(f"Model evaluation completed for {len(test_features)} samples")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics about stored models.
        
        Returns:
            Model statistics dictionary
        """
        try:
            return self.registry.get_model_stats()
        except Exception as e:
            logger.error(f"Error getting model stats: {e}")
            return {}
    
    def cleanup_old_models(self, keep_last_n: int = 5) -> List[str]:
        """Clean up old models, keeping only the last N.
        
        Args:
            keep_last_n: Number of recent models to keep
            
        Returns:
            List of deleted model versions
        """
        try:
            deleted_models = self.registry.cleanup_old_models(keep_last_n)
            logger.info(f"Cleanup completed: {len(deleted_models)} models deleted")
            return deleted_models
        except Exception as e:
            logger.error(f"Error during model cleanup: {e}")
            return []
    
    def export_model(self, version: str, export_path: Path) -> bool:
        """Export a model to a different location.
        
        Args:
            version: Model version to export
            export_path: Path to export the model to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            model_dir = self.models_dir / version
            if not model_dir.exists():
                logger.error(f"Model directory not found: {model_dir}")
                return False
            
            # Create export directory
            ensure_directory(export_path)
            
            # Copy model files
            success = copy_file(
                model_dir / "model.joblib",
                export_path / "model.joblib",
                overwrite=True
            )
            
            if not success:
                return False
            
            # Copy scaler if exists
            scaler_file = model_dir / "scaler.joblib"
            if scaler_file.exists():
                copy_file(scaler_file, export_path / "scaler.joblib", overwrite=True)
            
            # Copy metadata
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                copy_file(metadata_file, export_path / "metadata.json", overwrite=True)
            
            logger.info(f"Model {version} exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting model {version}: {e}")
            return False
    
    def import_model(self, import_path: Path, version: Optional[str] = None) -> Optional[str]:
        """Import a model from a different location.
        
        Args:
            import_path: Path containing model files
            version: Version to assign (optional, uses timestamp if not provided)
            
        Returns:
            Imported model version or None if error
        """
        try:
            if not import_path.exists():
                logger.error(f"Import path does not exist: {import_path}")
                return None
            
            # Generate version if not provided
            if version is None:
                version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Check if model files exist
            model_file = import_path / "model.joblib"
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return None
            
            # Create model directory
            model_dir = self.models_dir / version
            ensure_directory(model_dir)
            
            # Copy files
            copy_file(model_file, model_dir / "model.joblib", overwrite=True)
            
            # Copy scaler if exists
            scaler_file = import_path / "scaler.joblib"
            if scaler_file.exists():
                copy_file(scaler_file, model_dir / "scaler.joblib", overwrite=True)
            
            # Copy metadata if exists
            metadata_file = import_path / "metadata.json"
            if metadata_file.exists():
                copy_file(metadata_file, model_dir / "metadata.json", overwrite=True)
            
            logger.info(f"Model imported successfully as version {version}")
            return version
            
        except Exception as e:
            logger.error(f"Error importing model: {e}")
            return None 