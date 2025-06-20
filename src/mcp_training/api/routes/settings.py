from fastapi import APIRouter, Request, HTTPException
from typing import Dict, Any
from ...core.config import get_global_config

router = APIRouter()

@router.options("/")
async def options_settings():
    """Handle CORS preflight requests for settings."""
    return {"message": "OK"}

@router.get("/")
async def get_settings(request: Request) -> Dict[str, Any]:
    """Get application settings."""
    try:
        config = get_global_config()
        return config.to_dict()
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
        # Validate the settings structure
        if not isinstance(settings, dict):
            raise HTTPException(status_code=400, detail="Settings must be a dictionary")
        
        # Update configuration from the provided settings
        config = get_global_config()
        config.update_from_dict(settings)
        
        return {
            "message": "Settings updated successfully and saved to .env file",
            "settings": config.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}") 