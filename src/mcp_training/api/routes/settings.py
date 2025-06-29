"""
Settings API routes
"""

from fastapi import APIRouter, Request, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime
import logging

from ...core.config import get_global_config, get_config
from ...services.storage_service import StorageService
from ...api.deps import get_storage_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.options("/")
async def options_settings():
    """Handle CORS preflight requests for settings."""
    return {"message": "OK"}

@router.get("/")
async def get_settings() -> Dict[str, Any]:
    """Get current system settings."""
    try:
        config = get_config()
        return {
            "general": {
                "service_name": "MCP Training Service",
                "version": "1.0.0",
                "timezone": "UTC",
                "date_format": "YYYY-MM-DD",
                "auto_refresh": True,
                "notifications": True,
                "performance_mode": "training",
                "live_updates": True
            },
            "training": {
                "max_concurrent_jobs": 3,
                "default_max_iterations": 1000,
                "default_learning_rate": 0.01,
                "job_timeout": 24,
                "default_config": '{"algorithm": "random_forest", "n_estimators": 100, "max_depth": 10}'
            },
            "storage": {
                "models_dir": "models",
                "exports_dir": "exports",
                "logs_dir": "logs",
                "max_storage_gb": 10,
                "auto_cleanup": True,
                "retention_days": 30
            },
            "logging": {
                "level": "INFO",
                "format": "structured",
                "training_only": True,
                "rotation": "daily",
                "retention": 30
            },
            "security": {
                "authentication_enabled": False,
                "ssl_enabled": False,
                "max_login_attempts": 5
            },
            "advanced": {
                "debug_mode": False,
                "profiling_enabled": False,
                "cache_enabled": True
            }
        }
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")

@router.put("/")
async def update_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update system settings."""
    try:
        # For now, just return success - actual implementation would save to config
        logger.info("Settings update requested")
        return {
            "status": "success",
            "message": "Settings updated successfully",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@router.get("/training/config")
async def get_training_config() -> Dict[str, Any]:
    """Get training configuration."""
    try:
        return {
            "config": {
                "algorithm": "random_forest",
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42
            }
        }
    except Exception as e:
        logger.error(f"Failed to get training config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get training config: {str(e)}")

@router.get("/storage/info")
async def get_storage_info(storage_service: StorageService = Depends(get_storage_service)) -> Dict[str, Any]:
    """Get storage information."""
    try:
        # Get basic storage info
        storage_info = {
            "used": 0,  # Placeholder
            "total": 10,  # Placeholder
            "models_count": 0,
            "exports_count": 0,
            "logs_count": 0
        }
        
        return {
            "storage_info": storage_info
        }
    except Exception as e:
        logger.error(f"Failed to get storage info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage info: {str(e)}") 