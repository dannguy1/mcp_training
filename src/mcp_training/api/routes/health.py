"""
Health check API routes.
"""

from fastapi import APIRouter, Request
from datetime import datetime
from typing import Dict, Any
import psutil
import os

router = APIRouter()


@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "mcp-training",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "training": "/api/training",
            "models": "/api/models",
            "docs": "/docs"
        }
    }


@router.get("/status")
async def health_status(request: Request) -> Dict[str, Any]:
    """Health status endpoint for dashboard."""
    try:
        # Get system information
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "system_status": "healthy",
            "active_jobs": 0,  # TODO: Get from training service
            "total_models": 0,  # TODO: Get from model service
            "storage_used": f"{disk.used // (1024**3):.1f} GB",
            "version": "1.0.0",
            "uptime": "0 days",  # TODO: Implement uptime tracking
            "cpu_usage": f"{psutil.cpu_percent():.1f}%",
            "memory_usage": f"{memory.percent:.1f}%"
        }
    except Exception as e:
        return {
            "system_status": "error",
            "active_jobs": 0,
            "total_models": 0,
            "storage_used": "0 GB",
            "version": "1.0.0",
            "uptime": "0 days",
            "cpu_usage": "0%",
            "memory_usage": "0%",
            "error": str(e)
        }


@router.get("/api/v1/health")
async def api_health_check(request: Request) -> Dict[str, Any]:
    """API health check endpoint."""
    return {
        "status": "healthy",
        "service": "mcp-training",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/api/v1/system/info")
async def system_info(request: Request) -> Dict[str, Any]:
    """Get system information."""
    
    return {
        "system": {
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total,
            "memory_available": psutil.virtual_memory().available,
            "disk_usage": psutil.disk_usage('/').percent
        },
        "process": {
            "pid": os.getpid(),
            "memory_usage": psutil.Process().memory_info().rss,
            "cpu_percent": psutil.Process().cpu_percent()
        },
        "service": {
            "name": "mcp-training",
            "version": "1.0.0",
            "uptime": "TODO"  # TODO: Implement uptime tracking
        }
    }


@router.get("/settings")
async def get_settings(request: Request) -> Dict[str, Any]:
    """Get application settings."""
    try:
        return {
            "general": {
                "service_name": "MCP Training Service",
                "service_version": "1.0.0",
                "timezone": "UTC",
                "date_format": "YYYY-MM-DD",
                "auto_refresh": True,
                "notifications": True
            },
            "training": {
                "max_concurrent_jobs": 3,
                "default_max_iterations": 1000,
                "default_learning_rate": 0.01,
                "job_timeout": 24,
                "default_config": {
                    "algorithm": "random_forest",
                    "n_estimators": 100,
                    "max_depth": 10
                }
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
                "log_level": "INFO",
                "log_format": "structured",
                "log_file": "logs/mcp_training.log",
                "max_log_size_mb": 100,
                "log_to_console": True,
                "log_to_file": True
            },
            "security": {
                "auth_enabled": False,
                "api_key": "",
                "cors_origins": "http://localhost:3000,https://example.com",
                "rate_limit": 100,
                "https_only": False,
                "secure_headers": True
            },
            "advanced": {
                "debug_mode": False,
                "performance_monitoring": True,
                "websocket_enabled": True,
                "auto_backup": False,
                "custom_config": {
                    "custom_setting": "value"
                }
            }
        }
    except Exception as e:
        return {
            "error": f"Failed to load settings: {str(e)}",
            "general": {},
            "training": {},
            "storage": {},
            "logging": {},
            "security": {},
            "advanced": {}
        } 