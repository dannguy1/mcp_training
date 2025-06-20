"""
Training API routes for MCP Training Service.
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict

from ...services.training_service import TrainingService
from ...services.storage_service import StorageService
from ...services.deps import get_training_service, get_storage_service
from ...utils.logger import get_logger
from ...utils.validation import validate_file_extension, validate_file_size

logger = get_logger(__name__)
router = APIRouter()


# Request/Response models
class TrainingRequest(BaseModel):
    """Training request model."""
    model_config = ConfigDict(protected_namespaces=())
    
    export_file: str = Field(..., description="Path to export file")
    model_cfg: Optional[Dict[str, Any]] = Field(None, description="Model configuration")
    training_config: Optional[Dict[str, Any]] = Field(None, description="Training configuration")


class TrainingResponse(BaseModel):
    """Training response model."""
    training_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Training status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Creation timestamp")


class TrainingStatus(BaseModel):
    """Training status model."""
    training_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Training status")
    progress: float = Field(..., description="Progress percentage")
    current_step: str = Field(..., description="Current training step")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Training results")


class TrainingList(BaseModel):
    """Training jobs list model."""
    trainings: List[TrainingStatus] = Field(..., description="List of training jobs")
    total: int = Field(..., description="Total number of training jobs")


# Training endpoints
@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    training_service: TrainingService = Depends(get_training_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """Start a new training job.
    
    Args:
        request: Training request
        background_tasks: Background tasks
        training_service: Training service
        storage_service: Storage service
        
    Returns:
        Training response with job ID
    """
    try:
        logger.info(f"Starting training job for export: {request.export_file}")
        
        # Validate export file
        export_path = Path(request.export_file)
        if not export_path.exists():
            raise HTTPException(status_code=404, detail=f"Export file not found: {request.export_file}")
        
        # Validate file extension
        if not validate_file_extension(export_path, ["json"]):
            raise HTTPException(status_code=400, detail="Export file must be a JSON file")
        
        # Validate file size (max 100MB)
        if not validate_file_size(export_path, 100):
            raise HTTPException(status_code=400, detail="Export file too large (max 100MB)")
        
        # Start training job
        training_id = await training_service.start_training(
            export_file=str(export_path),
            model_type=request.model_cfg.get("type", "isolation_forest") if request.model_cfg else "isolation_forest",
            model_name=request.model_cfg.get("name") if request.model_cfg else None,
            config_overrides=request.training_config
        )
        
        logger.info(f"Training job started: {training_id}")
        
        return TrainingResponse(
            training_id=training_id,
            status="started",
            message="Training job started successfully",
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start training: {str(e)}")


@router.post("/upload", response_model=TrainingResponse)
async def upload_and_train(
    file: UploadFile = File(...),
    model_cfg: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None,
    background_tasks: BackgroundTasks = None,
    training_service: TrainingService = Depends(get_training_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """Upload export file and start training.
    
    Args:
        file: Uploaded export file
        model_cfg: Model configuration
        training_config: Training configuration
        background_tasks: Background tasks
        training_service: Training service
        storage_service: Storage service
        
    Returns:
        Training response with job ID
    """
    try:
        logger.info(f"Uploading export file: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not validate_file_extension(file.filename, ["json"]):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        # Store uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        stored_path = storage_service.store_export(file.file, filename)
        
        if not stored_path:
            raise HTTPException(status_code=500, detail="Failed to store uploaded file")
        
        # Start training job
        training_id = await training_service.start_training(
            export_file=str(stored_path),
            model_type=model_cfg.get("type", "isolation_forest") if model_cfg else "isolation_forest",
            model_name=model_cfg.get("name") if model_cfg else None,
            config_overrides=training_config
        )
        
        logger.info(f"Upload and training job started: {training_id}")
        
        return TrainingResponse(
            training_id=training_id,
            status="started",
            message="File uploaded and training started successfully",
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading and training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload and train: {str(e)}")


@router.get("/status/{training_id}", response_model=TrainingStatus)
async def get_training_status(
    training_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Get training job status.
    
    Args:
        training_id: Training job ID
        training_service: Training service
        
    Returns:
        Training status
    """
    try:
        status = training_service.get_training_status(training_id)
        
        if not status:
            raise HTTPException(status_code=404, detail=f"Training job not found: {training_id}")
        
        return TrainingStatus(**status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training status: {str(e)}")


@router.get("/list", response_model=TrainingList)
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    training_service: TrainingService = Depends(get_training_service)
):
    """List training jobs.
    
    Args:
        status: Filter by status
        limit: Maximum number of jobs to return
        offset: Number of jobs to skip
        training_service: Training service
        
    Returns:
        List of training jobs
    """
    try:
        jobs = await training_service.list_training_jobs(
            status=status,
            limit=limit,
            offset=offset
        )
        
        return TrainingList(
            trainings=[TrainingStatus(**job) for job in jobs],
            total=len(jobs)
        )
        
    except Exception as e:
        logger.error(f"Error listing training jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list training jobs: {str(e)}")


@router.delete("/cancel/{training_id}")
async def cancel_training(
    training_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Cancel a training job.
    
    Args:
        training_id: Training job ID
        training_service: Training service
        
    Returns:
        Cancellation result
    """
    try:
        success = await training_service.cancel_training(training_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Training job not found or cannot be cancelled: {training_id}")
        
        return {"message": f"Training job {training_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling training: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cancel training: {str(e)}")


@router.delete("/cleanup")
async def cleanup_completed_jobs(
    days_old: int = 7,
    training_service: TrainingService = Depends(get_training_service)
):
    """Clean up completed training jobs.
    
    Args:
        days_old: Delete jobs older than this many days
        training_service: Training service
        
    Returns:
        Cleanup result
    """
    try:
        deleted_count = await training_service.cleanup_completed_jobs(days_old)
        
        return {
            "message": f"Cleaned up {deleted_count} completed training jobs",
            "deleted_count": deleted_count
        }
        
    except Exception as e:
        logger.error(f"Error cleaning up training jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup training jobs: {str(e)}")


@router.get("/logs/{training_id}")
async def get_training_logs(
    training_id: str,
    lines: int = 100,
    training_service: TrainingService = Depends(get_training_service)
):
    """Get training job logs.
    
    Args:
        training_id: Training job ID
        lines: Number of log lines to return
        training_service: Training service
        
    Returns:
        Training logs
    """
    try:
        logs = await training_service.get_training_logs(training_id, lines)
        
        if logs is None:
            raise HTTPException(status_code=404, detail=f"Training job not found: {training_id}")
        
        return {
            "training_id": training_id,
            "logs": logs,
            "line_count": len(logs)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training logs: {str(e)}")


@router.get("/stats")
async def get_training_stats(
    training_service: TrainingService = Depends(get_training_service)
):
    """Get training statistics.
    
    Args:
        training_service: Training service
        
    Returns:
        Training statistics
    """
    try:
        stats = await training_service.get_training_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting training stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training stats: {str(e)}")


@router.get("/exports")
async def list_export_files(
    storage_service: StorageService = Depends(get_storage_service)
):
    """Get list of available export files.
    
    Args:
        storage_service: Storage service
        
    Returns:
        List of export files
    """
    try:
        exports = storage_service.list_exports()
        return exports
    except Exception as e:
        logger.error(f"Error listing export files: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list export files: {str(e)}")


@router.post("/exports/upload")
async def upload_export_file(
    file: UploadFile = File(...),
    storage_service: StorageService = Depends(get_storage_service)
):
    """Upload an export file to the exports directory.
    
    Args:
        file: Uploaded export file
        storage_service: Storage service
        
    Returns:
        Upload result with file information
    """
    try:
        logger.info(f"Uploading export file: {file.filename}")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not validate_file_extension(file.filename, ["json"]):
            raise HTTPException(status_code=400, detail="File must be a JSON file")
        
        # Store uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"upload_{timestamp}_{file.filename}"
        stored_path = storage_service.store_export(file.file, filename)
        
        if not stored_path:
            raise HTTPException(status_code=500, detail="Failed to store uploaded file")
        
        # Get file information
        file_info = storage_service.get_export_info(filename)
        
        logger.info(f"Export file uploaded successfully: {filename}")
        
        return {
            "message": "Export file uploaded successfully",
            "filename": filename,
            "path": str(stored_path),
            "file_info": file_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading export file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload export file: {str(e)}")


@router.get("/jobs")
async def get_training_jobs(
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    training_service: TrainingService = Depends(get_training_service)
):
    """Get list of training jobs."""
    try:
        jobs_dict = training_service.list_training_tasks()
        jobs = list(jobs_dict.values())
        # Optionally filter by status
        if status:
            jobs = [job for job in jobs if job.get('status') == status]
        # Apply offset and limit
        jobs = jobs[offset:offset+limit]
        return jobs
    except Exception as e:
        logger.error(f"Error getting training jobs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training jobs: {str(e)}")


@router.post("/jobs", response_model=TrainingResponse)
async def create_training_job(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    training_service: TrainingService = Depends(get_training_service),
    storage_service: StorageService = Depends(get_storage_service)
):
    """Create a new training job.
    
    Args:
        request: Training request
        background_tasks: Background tasks
        training_service: Training service
        storage_service: Storage service
        
    Returns:
        Training response with job ID
    """
    try:
        logger.info(f"Creating training job for export: {request.export_file}")
        
        # Validate export file
        export_path = Path(request.export_file)
        if not export_path.exists():
            raise HTTPException(status_code=404, detail=f"Export file not found: {request.export_file}")
        
        # Validate file extension
        if not validate_file_extension(export_path, ["json"]):
            raise HTTPException(status_code=400, detail="Export file must be a JSON file")
        
        # Validate file size (max 100MB)
        if not validate_file_size(export_path, 100):
            raise HTTPException(status_code=400, detail="Export file too large (max 100MB)")
        
        # Start training job
        training_id = await training_service.start_training(
            export_file=str(export_path),
            model_type=request.model_cfg.get("type", "isolation_forest") if request.model_cfg else "isolation_forest",
            model_name=request.model_cfg.get("name") if request.model_cfg else None,
            config_overrides=request.training_config
        )
        
        logger.info(f"Training job created: {training_id}")
        
        return TrainingResponse(
            training_id=training_id,
            status="started",
            message="Training job created successfully",
            created_at=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating training job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_training_job(
    job_id: str,
    training_service: TrainingService = Depends(get_training_service)
):
    """Get individual training job details.
    
    Args:
        job_id: Training job ID
        training_service: Training service
        
    Returns:
        Training job details
    """
    try:
        job = training_service.get_training_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
        
        return job
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting training job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training job: {str(e)}") 