"""
Models API routes for MCP Training Service.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import os
import json

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, ConfigDict
import numpy as np

from ...services.model_service import ModelService
from ...services.storage_service import StorageService
from ...services.deps import get_model_service, get_storage_service
from ...utils.logger import get_logger
from ...utils.validation import validate_file_extension, validate_file_size
from ...models.metadata import ModelInfo, ModelMetadata

logger = get_logger(__name__)
router = APIRouter()


# Request/Response models
class ModelInfo(BaseModel):
    """Model information model."""
    model_config = ConfigDict(protected_namespaces=())
    
    version: str = Field(..., description="Model version")
    id: str = Field(..., description="Model ID (same as version)")
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    status: str = Field(..., description="Model status")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    deployed_at: Optional[datetime] = Field(None, description="Deployment timestamp")
    deployed_by: Optional[str] = Field(None, description="User who deployed the model")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Model metrics")
    file_size: Optional[int] = Field(None, description="Model file size in bytes")
    size: int = Field(0, description="Model file size in bytes (for frontend)")


class ModelList(BaseModel):
    """Model list model."""
    model_config = ConfigDict(protected_namespaces=())
    
    models: List[ModelInfo] = Field(..., description="List of models")
    total: int = Field(..., description="Total number of models")


class PredictionRequest(BaseModel):
    """Prediction request model."""
    model_config = ConfigDict(protected_namespaces=())
    
    features: List[List[float]] = Field(..., description="Feature arrays for prediction")
    threshold: Optional[float] = Field(None, description="Anomaly threshold")


class PredictionResponse(BaseModel):
    """Prediction response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_version: str = Field(..., description="Model version used")
    predictions: List[int] = Field(..., description="Prediction results (0=normal, 1=anomaly)")
    scores: List[float] = Field(..., description="Anomaly scores")
    threshold: float = Field(..., description="Threshold used")
    statistics: Dict[str, Any] = Field(..., description="Prediction statistics")


class EvaluationRequest(BaseModel):
    """Model evaluation request model."""
    model_config = ConfigDict(protected_namespaces=())
    
    test_features: List[List[float]] = Field(..., description="Test feature arrays")
    test_labels: Optional[List[int]] = Field(None, description="Test labels (optional)")


class EvaluationResponse(BaseModel):
    """Model evaluation response model."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_version: str = Field(..., description="Model version evaluated")
    evaluation: Dict[str, Any] = Field(..., description="Evaluation results")


# Model endpoints
@router.get("/", response_model=ModelList)
async def get_models(
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    model_service: ModelService = Depends(get_model_service)
):
    """Get list of models.
    
    Args:
        status: Filter by status
        model_type: Filter by model type
        limit: Maximum number of models to return
        offset: Number of models to skip
        model_service: Model service
        
    Returns:
        List of models
    """
    try:
        models = model_service.list_models()
        
        # Apply filters
        if status:
            models = [m for m in models if m.get('deployment_status') == status]
        if model_type:
            models = [m for m in models if m.get('model_type') == model_type]
        
        # Apply pagination
        total = len(models)
        models = models[offset:offset + limit]
        
        # Convert to ModelInfo format
        model_list = []
        for model in models:
            # Map deployment status to frontend expected status
            status = model['deployment_status']
            if status == 'available':
                status = 'ready'  # Frontend expects 'ready' for available models
            elif status == 'deployed':
                status = 'deployed'  # Keep as is
            else:
                status = 'ready'  # Default to ready for other statuses
            
            # Try to get file size from metadata, else from model.joblib
            file_size = model.get('file_size')
            if not file_size:
                # Try to get from model.joblib
                model_path = Path(model['path']) / 'model.joblib'
                if model_path.exists():
                    file_size = model_path.stat().st_size
                else:
                    file_size = 0
            
            model_info = ModelInfo(
                version=model['version'],
                id=model['version'],
                name=f"Model {model['version']}",
                type=model['model_type'],
                status=status,
                created_at=model['created_at'],
                updated_at=model['created_at'],  # Use created_at as updated_at for now
                deployed_at=None,  # Will be set when deployed
                deployed_by=None,  # Will be set when deployed
                metrics=None,  # Could be added later
                file_size=file_size,
                size=file_size
            )
            model_list.append(model_info)
        
        return ModelList(
            models=model_list,
            total=total
        )
        
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/{version}", response_model=ModelInfo)
async def get_model(
    version: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Get model information.
    
    Args:
        version: Model version
        model_service: Model service
        
    Returns:
        Model information
    """
    try:
        model_metadata = model_service.get_model(version)
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail=f"Model not found: {version}")
        
        # Convert ModelMetadata to ModelInfo format
        # Map deployment status to frontend expected status
        status = model_metadata.deployment_info.status
        if status == 'available':
            status = 'ready'  # Frontend expects 'ready' for available models
        elif status == 'deployed':
            status = 'deployed'  # Keep as is
        else:
            status = 'ready'  # Default to ready for other statuses
        
        # Try to get file size from metadata, else from model.joblib
        file_size = model_metadata.training_info.export_file_size
        if not file_size:
            # Try to get from model.joblib
            model_dir = Path(model_metadata.model_info.version)
            model_path = Path('models') / model_metadata.model_info.version / 'model.joblib'
            if model_path.exists():
                file_size = model_path.stat().st_size
            else:
                file_size = 0
        
        model_info = ModelInfo(
            version=model_metadata.model_info.version,
            id=model_metadata.model_info.version,
            name=f"Model {model_metadata.model_info.version}",
            type=model_metadata.model_info.model_type,
            status=status,
            created_at=model_metadata.model_info.created_at,
            updated_at=model_metadata.model_info.created_at,  # Use created_at as updated_at for now
            deployed_at=model_metadata.deployment_info.deployed_at,
            deployed_by=model_metadata.deployment_info.deployed_by,
            metrics=model_metadata.evaluation_info.basic_metrics if model_metadata.evaluation_info.basic_metrics else None,
            file_size=file_size,
            size=file_size
        )
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")


@router.delete("/{version}")
async def delete_model(
    version: str,
    model_service: ModelService = Depends(get_model_service)
):
    """Delete a model.
    
    Args:
        version: Model version
        model_service: Model service
        
    Returns:
        Deletion result
    """
    try:
        success = model_service.delete_model(version)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model not found: {version}")
        
        return {"message": f"Model {version} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post("/{version}/deploy")
async def deploy_model(
    version: str,
    deployed_by: Optional[str] = None,
    model_service: ModelService = Depends(get_model_service)
):
    """Deploy a model.
    
    Args:
        version: Model version
        deployed_by: User deploying the model
        model_service: Model service
        
    Returns:
        Deployment result
    """
    try:
        success = model_service.deploy_model(version, deployed_by)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model not found: {version}")
        
        return {"message": f"Model {version} deployed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deploying model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to deploy model: {str(e)}")


@router.get("/latest", response_model=ModelInfo)
async def get_latest_model(
    model_service: ModelService = Depends(get_model_service)
):
    """Get the latest trained model.
    
    Args:
        model_service: Model service
        
    Returns:
        Latest model information
    """
    try:
        model_metadata = model_service.get_latest_model()
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail="No models found")
        
        # Convert ModelMetadata to ModelInfo format
        # Map deployment status to frontend expected status
        status = model_metadata.deployment_info.status
        if status == 'available':
            status = 'ready'  # Frontend expects 'ready' for available models
        elif status == 'deployed':
            status = 'deployed'  # Keep as is
        else:
            status = 'ready'  # Default to ready for other statuses
        
        # Try to get file size from metadata, else from model.joblib
        file_size = model_metadata.training_info.export_file_size
        if not file_size:
            # Try to get from model.joblib
            model_dir = Path(model_metadata.model_info.version)
            model_path = Path('models') / model_metadata.model_info.version / 'model.joblib'
            if model_path.exists():
                file_size = model_path.stat().st_size
            else:
                file_size = 0
        
        model_info = ModelInfo(
            version=model_metadata.model_info.version,
            id=model_metadata.model_info.version,
            name=f"Model {model_metadata.model_info.version}",
            type=model_metadata.model_info.model_type,
            status=status,
            created_at=model_metadata.model_info.created_at,
            updated_at=model_metadata.model_info.created_at,  # Use created_at as updated_at for now
            deployed_at=model_metadata.deployment_info.deployed_at,
            deployed_by=model_metadata.deployment_info.deployed_by,
            metrics=model_metadata.evaluation_info.basic_metrics if model_metadata.evaluation_info.basic_metrics else None,
            file_size=file_size,
            size=file_size
        )
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest model: {str(e)}")


@router.get("/deployed", response_model=ModelInfo)
async def get_deployed_model(
    model_service: ModelService = Depends(get_model_service)
):
    """Get the currently deployed model.
    
    Args:
        model_service: Model service
        
    Returns:
        Deployed model information
    """
    try:
        model_metadata = model_service.get_deployed_model()
        
        if not model_metadata:
            raise HTTPException(status_code=404, detail="No deployed model found")
        
        # Convert ModelMetadata to ModelInfo format
        # Map deployment status to frontend expected status
        status = model_metadata.deployment_info.status
        if status == 'available':
            status = 'ready'  # Frontend expects 'ready' for available models
        elif status == 'deployed':
            status = 'deployed'  # Keep as is
        else:
            status = 'ready'  # Default to ready for other statuses
        
        # Try to get file size from metadata, else from model.joblib
        file_size = model_metadata.training_info.export_file_size
        if not file_size:
            # Try to get from model.joblib
            model_dir = Path(model_metadata.model_info.version)
            model_path = Path('models') / model_metadata.model_info.version / 'model.joblib'
            if model_path.exists():
                file_size = model_path.stat().st_size
            else:
                file_size = 0
        
        model_info = ModelInfo(
            version=model_metadata.model_info.version,
            id=model_metadata.model_info.version,
            name=f"Model {model_metadata.model_info.version}",
            type=model_metadata.model_info.model_type,
            status=status,
            created_at=model_metadata.model_info.created_at,
            updated_at=model_metadata.model_info.created_at,  # Use created_at as updated_at for now
            deployed_at=model_metadata.deployment_info.deployed_at,
            deployed_by=model_metadata.deployment_info.deployed_by,
            metrics=model_metadata.evaluation_info.basic_metrics if model_metadata.evaluation_info.basic_metrics else None,
            file_size=file_size,
            size=file_size
        )
        
        return model_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting deployed model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get deployed model: {str(e)}")


@router.post("/{version}/predict", response_model=PredictionResponse)
async def predict(
    version: str,
    request: PredictionRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """Make predictions using a model.
    
    Args:
        version: Model version
        request: Prediction request
        model_service: Model service
        
    Returns:
        Prediction results
    """
    try:
        # Convert features to numpy array
        features = np.array(request.features)
        
        # Make predictions
        result = model_service.predict(
            model_version=version,
            features=features,
            threshold=request.threshold
        )
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions with model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to make predictions: {str(e)}")


@router.post("/{version}/evaluate", response_model=EvaluationResponse)
async def evaluate_model(
    version: str,
    request: EvaluationRequest,
    model_service: ModelService = Depends(get_model_service)
):
    """Evaluate a model.
    
    Args:
        version: Model version
        request: Evaluation request
        model_service: Model service
        
    Returns:
        Evaluation results
    """
    try:
        # Convert features to numpy array
        test_features = np.array(request.test_features)
        
        # Convert labels if provided
        test_labels = None
        if request.test_labels:
            test_labels = np.array(request.test_labels)
        
        # Evaluate model
        evaluation = model_service.evaluate_model(
            model_version=version,
            test_features=test_features,
            test_labels=test_labels
        )
        
        return EvaluationResponse(
            model_version=version,
            evaluation=evaluation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to evaluate model: {str(e)}")


@router.get("/{version}/download")
async def download_model(
    version: str,
    model_service: ModelService = Depends(get_model_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """Download a model.
    
    Args:
        version: Model version
        model_service: Model service
        storage_service: Storage service
        
    Returns:
        Model file
    """
    try:
        # Get model path
        model_path = model_service.registry.get_model_path(version)
        
        if not model_path or not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model file not found: {version}")
        
        return FileResponse(
            path=model_path,
            filename=f"model_{version}.joblib",
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading model {version}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")


@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    version: Optional[str] = None,
    model_service: ModelService = Depends(get_model_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """Upload a model.
    
    Args:
        file: Model file
        version: Model version (optional)
        model_service: Model service
        storage_service: Storage service
        
    Returns:
        Upload result
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not validate_file_extension(file.filename, ["joblib"]):
            raise HTTPException(status_code=400, detail="File must be a joblib file")
        
        # Import model
        imported_version = model_service.import_model(file.file, version)
        
        if not imported_version:
            raise HTTPException(status_code=500, detail="Failed to import model")
        
        return {
            "message": f"Model uploaded successfully",
            "version": imported_version
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload model: {str(e)}")


@router.get("/stats")
async def get_model_stats(
    model_service: ModelService = Depends(get_model_service)
):
    """Get model statistics.
    
    Args:
        model_service: Model service
        
    Returns:
        Model statistics
    """
    try:
        stats = model_service.get_model_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting model stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model stats: {str(e)}")


@router.delete("/cleanup")
async def cleanup_old_models(
    keep_last_n: int = 5,
    model_service: ModelService = Depends(get_model_service)
):
    """Clean up old models.
    
    Args:
        keep_last_n: Number of recent models to keep
        model_service: Model service
        
    Returns:
        Cleanup result
    """
    try:
        deleted_models = model_service.cleanup_old_models(keep_last_n)
        
        return {
            "message": f"Cleaned up {len(deleted_models)} old models",
            "deleted_models": deleted_models
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup models: {str(e)}")


@router.get("/{version}/deployment-package")
async def download_deployment_package(version: str, model_service: ModelService = Depends(get_model_service)):
    """Download the deployment package for a deployed model."""
    try:
        # Check if model exists and is deployed
        model = model_service.get_model(version)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {version} not found")
        
        if model.deployment_info.status != "deployed":
            raise HTTPException(status_code=400, detail=f"Model {version} is not deployed")
        
        # Check if deployment package exists
        deployments_dir = model_service.models_dir / "deployments"
        package_name = f"model_{version}_deployment.zip"
        package_path = deployments_dir / package_name
        
        if not package_path.exists():
            raise HTTPException(status_code=404, detail=f"Deployment package for {version} not found")
        
        # Return the file
        return FileResponse(
            path=str(package_path),
            filename=package_name,
            media_type="application/zip"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download deployment package: {str(e)}") 