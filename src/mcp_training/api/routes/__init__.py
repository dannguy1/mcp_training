"""
API routes for MCP Training Service.
"""

from .training import router as training_router
from .models import router as models_router
from .health import router as health_router

__all__ = ['training_router', 'models_router', 'health_router'] 