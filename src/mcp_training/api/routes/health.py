"""
Health check API routes.
"""

from fastapi import APIRouter, Request
from datetime import datetime
from typing import Dict, Any

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
            "training": "/api/v1/training",
            "models": "/api/v1/models",
            "docs": "/docs"
        }
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
    import psutil
    import os
    
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