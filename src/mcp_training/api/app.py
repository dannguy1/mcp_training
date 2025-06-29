"""
FastAPI application for MCP Training Service.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from ..core.config import get_config, get_global_config as core_config
from ..utils.logger import setup_logging, get_logger
from ..services.training_service import TrainingService
from ..services.model_service import ModelService
from ..services.storage_service import StorageService
from ..services.deps import set_services
from .middleware.logging import LoggingMiddleware, RequestIDMiddleware, PerformanceMiddleware
from .middleware.cors import setup_cors
from .middleware.auth import get_auth_middleware
from .routes import health, training, models, web, settings, logs, websocket
from ..models.config import ModelConfig


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger = get_logger(__name__)
    logger.info("Starting MCP Training Service...")
    
    # Background task for periodic system status broadcasts
    background_task = None
    
    try:
        # Load configuration
        config = get_config()
        
        # Load model config from YAML
        model_config_data = core_config().model_config_data
        model_config = ModelConfig(**model_config_data)
        
        # Setup logging
        setup_logging(
            log_level=config.get("logging", {}).get("level", "INFO"),
            log_file=config.get("logging", {}).get("file"),
            log_format=config.get("logging", {}).get("format", "structured"),
            training_only=config.get("logging", {}).get("training_only", True)
        )
        
        # Initialize services
        # Get the project root directory (current working directory)
        project_root = Path.cwd()
        
        models_dir = str(project_root / config.get("storage", {}).get("models_dir", "models"))
        exports_dir = str(project_root / config.get("storage", {}).get("exports_dir", "exports"))
        logs_dir = str(project_root / config.get("storage", {}).get("logs_dir", "logs"))
        
        logger.info(f"Project root: {project_root}")
        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Exports directory: {exports_dir}")
        logger.info(f"Logs directory: {logs_dir}")
        
        storage_service = StorageService(
            models_dir=models_dir,
            exports_dir=exports_dir,
            logs_dir=logs_dir
        )
        
        model_service = ModelService(
            models_dir=models_dir
        )
        
        training_service = TrainingService(
            config=model_config,
            model_service=model_service,
            storage_service=storage_service
        )
        
        # Register services with dependency injection
        set_services(training_service, model_service, storage_service)
        
        # Start periodic system status broadcast task
        import asyncio
        from .routes.websocket import broadcast_system_status
        from .routes.health import health_status
        
        async def get_system_status():
            """Get system status without requiring a request object."""
            try:
                # Create a mock request object for the health_status function
                from fastapi import Request
                from starlette.datastructures import State
                
                # Create a minimal mock request
                mock_request = Request(scope={
                    'type': 'http',
                    'method': 'GET',
                    'path': '/api/health/status',
                    'headers': [],
                    'state': State()
                })
                
                return await health_status(mock_request)
            except Exception as e:
                logger.warning(f"Failed to get system status: {e}")
                return {
                    "system_status": "error",
                    "active_jobs": 0,
                    "total_models": 0,
                    "storage_used": "0 GB",
                    "version": "1.0.0",
                    "uptime": "0 days",
                    "cpu_usage": "0%",
                    "memory_usage": "0%"
                }
        
        async def periodic_status_broadcast():
            """Periodically broadcast system status to keep WebSocket connections alive."""
            while True:
                try:
                    await asyncio.sleep(30)  # Broadcast every 30 seconds
                    status = await get_system_status()
                    await broadcast_system_status(status)
                except Exception as e:
                    logger.warning(f"Failed to broadcast system status: {e}")
                    await asyncio.sleep(5)  # Wait a bit before retrying
        
        background_task = asyncio.create_task(periodic_status_broadcast())
        logger.info("Periodic system status broadcast task started")
        
        logger.info("MCP Training Service started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start MCP Training Service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Training Service...")
    
    # Cancel background task
    if background_task:
        background_task.cancel()
        try:
            await background_task
        except asyncio.CancelledError:
            pass
        logger.info("Periodic system status broadcast task cancelled")
    
    # Cleanup services
    if training_service:
        await training_service.shutdown()
    
    logger.info("MCP Training Service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI application
    """
    # Load configuration
    config = get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="MCP Training Service",
        description="Machine Learning model training service for MCP data analysis",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app, config)
    
    # Setup routes
    setup_routes(app)
    
    # Setup exception handlers
    setup_exception_handlers(app)
    
    return app


def setup_middleware(app: FastAPI, config: dict):
    """Setup application middleware.
    
    Args:
        app: FastAPI application
        config: Application configuration
    """
    logger = get_logger(__name__)
    
    # Debug logging for config
    logger.info(f"Setting up middleware with config: {config}")
    
    # CORS middleware
    cors_config = config.get("cors", {})
    logger.info(f"CORS config from config dict: {cors_config}")
    
    setup_cors(
        app,
        allowed_origins=cors_config.get("allowed_origins"),
        allowed_methods=cors_config.get("allowed_methods"),
        allowed_headers=cors_config.get("allowed_headers"),
        allow_credentials=cors_config.get("allow_credentials", True)
    )
    
    # Logging middleware
    logging_config = config.get("logging", {})
    app.add_middleware(
        LoggingMiddleware,
        log_requests=logging_config.get("log_requests", True),
        log_responses=logging_config.get("log_responses", True)
    )
    
    # Request ID middleware
    app.add_middleware(RequestIDMiddleware)
    
    # Performance middleware
    performance_config = config.get("performance", {})
    app.add_middleware(
        PerformanceMiddleware,
        slow_request_threshold=performance_config.get("slow_request_threshold", 1.0)
    )
    
    # Authentication middleware (if enabled)
    auth_config = config.get("auth", {})
    if auth_config.get("enabled", False):
        app.add_middleware(
            get_auth_middleware(
                api_key=auth_config.get("api_key"),
                require_auth=auth_config.get("require_auth", False),
                exclude_paths=auth_config.get("exclude_paths")
            )
        )
    
    logger.info("Middleware configured")


def setup_routes(app: FastAPI):
    """Setup application routes.
    
    Args:
        app: FastAPI application
    """
    logger = get_logger(__name__)
    
    # Setup static files
    static_dir = Path(__file__).parent.parent / "web" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"Static files mounted at /static from {static_dir}")
    
    # Include API route modules
    app.include_router(health.router, prefix="/api/health", tags=["health"])
    app.include_router(training.router, prefix="/api/training", tags=["training"])
    app.include_router(models.router, prefix="/api/models", tags=["models"])
    app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
    app.include_router(logs.router, prefix="/api/logs", tags=["logs"])
    
    # Include WebSocket routes
    app.include_router(websocket.router, tags=["websocket"])
    
    # Include web routes (HTML pages)
    app.include_router(web.router, tags=["web"])
    
    # API root endpoint
    @app.get("/api", tags=["api"])
    async def api_root():
        """API root endpoint."""
        return {
            "service": "MCP Training Service",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/api/health",
                "training": "/api/training",
                "models": "/api/models",
                "web": "/"
            }
        }
    
    logger.info("Routes configured")


def setup_exception_handlers(app: FastAPI):
    """Setup exception handlers.
    
    Args:
        app: FastAPI application
    """
    logger = get_logger(__name__)
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors."""
        logger.warning(f"Validation error: {exc.errors()}")
        return JSONResponse(
            status_code=422,
            content={
                "error": "Validation error",
                "details": exc.errors()
            }
        )
    
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        logger.warning(f"HTTP error {exc.status_code}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error"
            }
        )
    
    logger.info("Exception handlers configured")


# Create app instance
app = create_app()


# Export app for uvicorn
if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    uvicorn.run(
        "app:app",
        host=config.get("api", {}).get("host", "0.0.0.0"),
        port=config.get("api", {}).get("port", 8000),
        reload=config.get("api", {}).get("reload", False)
    ) 