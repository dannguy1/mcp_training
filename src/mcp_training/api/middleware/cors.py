"""
CORS middleware for FastAPI.
"""

from typing import List, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware as FastAPICORSMiddleware

from ...utils.logger import get_logger

logger = get_logger(__name__)


def setup_cors(
    app: FastAPI,
    allowed_origins: Optional[List[str]] = None,
    allowed_methods: Optional[List[str]] = None,
    allowed_headers: Optional[List[str]] = None,
    allow_credentials: bool = True,
    max_age: int = 600
):
    """Setup CORS middleware for the application.
    
    Args:
        app: FastAPI application
        allowed_origins: List of allowed origins (defaults to ["*"])
        allowed_methods: List of allowed HTTP methods
        allowed_headers: List of allowed headers
        allow_credentials: Whether to allow credentials
        max_age: Max age for preflight requests
    """
    # Default allowed origins
    if allowed_origins is None:
        allowed_origins = ["*"]
    
    # Default allowed methods
    if allowed_methods is None:
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    
    # Default allowed headers
    if allowed_headers is None:
        allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-Process-Time"
        ]
    
    # Add CORS middleware
    app.add_middleware(
        FastAPICORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        max_age=max_age
    )
    
    logger.info("CORS middleware configured")


class CORSMiddleware:
    """Custom CORS middleware wrapper."""
    
    def __init__(
        self,
        allowed_origins: Optional[List[str]] = None,
        allowed_methods: Optional[List[str]] = None,
        allowed_headers: Optional[List[str]] = None,
        allow_credentials: bool = True,
        max_age: int = 600
    ):
        """Initialize CORS middleware.
        
        Args:
            allowed_origins: List of allowed origins
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed headers
            allow_credentials: Whether to allow credentials
            max_age: Max age for preflight requests
        """
        self.allowed_origins = allowed_origins or ["*"]
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.allowed_headers = allowed_headers or [
            "Accept",
            "Accept-Language", 
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Request-ID",
            "X-Process-Time"
        ]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    def apply(self, app: FastAPI):
        """Apply CORS middleware to FastAPI app.
        
        Args:
            app: FastAPI application
        """
        setup_cors(
            app,
            allowed_origins=self.allowed_origins,
            allowed_methods=self.allowed_methods,
            allowed_headers=self.allowed_headers,
            allow_credentials=self.allow_credentials,
            max_age=self.max_age
        ) 