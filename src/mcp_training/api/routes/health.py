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