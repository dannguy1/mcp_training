from fastapi import APIRouter, Request
from typing import Dict, Any

router = APIRouter()

@router.get("/")
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