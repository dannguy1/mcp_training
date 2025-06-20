from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any
import json
import os

router = APIRouter()

# Global settings storage (in a real app, this would be persisted to a database or config file)
_settings = {
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

@router.get("/")
async def get_settings(request: Request) -> Dict[str, Any]:
    """Get application settings."""
    try:
        return _settings
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

@router.put("/")
async def update_settings(request: Request, settings: Dict[str, Any]) -> Dict[str, Any]:
    """Update application settings."""
    try:
        global _settings
        
        # Validate the settings structure
        if not isinstance(settings, dict):
            raise HTTPException(status_code=400, detail="Settings must be a dictionary")
        
        # Update settings with provided values
        for section, section_data in settings.items():
            if section in _settings and isinstance(section_data, dict):
                _settings[section].update(section_data)
        
        # Optionally persist settings to a file (for development)
        try:
            config_dir = "config"
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            settings_file = os.path.join(config_dir, "user_settings.json")
            with open(settings_file, 'w') as f:
                json.dump(_settings, f, indent=2)
        except Exception as e:
            # Log the error but don't fail the request
            print(f"Warning: Could not persist settings to file: {e}")
        
        return {
            "message": "Settings updated successfully",
            "settings": _settings
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}") 