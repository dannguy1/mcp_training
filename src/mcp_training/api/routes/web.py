"""
Web route handlers for MCP Training Service UI.
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path

# Get the template directory path
template_dir = Path(__file__).parent.parent.parent / "web" / "templates"
templates = Jinja2Templates(directory=str(template_dir))

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard page."""
    return templates.TemplateResponse("pages/dashboard.html", {"request": request})


@router.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """Training management page."""
    return templates.TemplateResponse("pages/training.html", {"request": request})


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """Model management page."""
    return templates.TemplateResponse("pages/models.html", {"request": request})


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(request: Request):
    """Logs viewer page."""
    return templates.TemplateResponse("pages/logs.html", {"request": request})


@router.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    return templates.TemplateResponse("pages/settings.html", {"request": request}) 