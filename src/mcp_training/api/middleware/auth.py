"""
Authentication middleware for FastAPI.
"""

import os
from typing import Optional, Dict, Any
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...utils.logger import get_logger

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Basic authentication middleware."""
    
    def __init__(
        self,
        app: ASGIApp,
        api_key: Optional[str] = None,
        require_auth: bool = False,
        exclude_paths: Optional[list] = None
    ):
        """Initialize authentication middleware.
        
        Args:
            app: ASGI application
            api_key: API key for authentication
            require_auth: Whether authentication is required
            exclude_paths: Paths to exclude from authentication
        """
        super().__init__(app)
        self.api_key = api_key or os.getenv("API_KEY")
        self.require_auth = require_auth
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/openapi.json"]
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        # Check if path should be excluded
        if self._should_exclude_path(request.url.path):
            return await call_next(request)
        
        # Check if authentication is required
        if not self.require_auth and not self.api_key:
            return await call_next(request)
        
        # Validate authentication
        if not await self._validate_auth(request):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return await call_next(request)
    
    def _should_exclude_path(self, path: str) -> bool:
        """Check if path should be excluded from authentication.
        
        Args:
            path: Request path
            
        Returns:
            True if path should be excluded
        """
        for exclude_path in self.exclude_paths:
            if path.startswith(exclude_path):
                return True
        return False
    
    async def _validate_auth(self, request: Request) -> bool:
        """Validate authentication credentials.
        
        Args:
            request: HTTP request
            
        Returns:
            True if authentication is valid
        """
        try:
            # Check for API key in header
            api_key_header = request.headers.get("X-API-Key")
            if api_key_header and api_key_header == self.api_key:
                return True
            
            # Check for Bearer token
            credentials: HTTPAuthorizationCredentials = await self.security(request)
            if credentials and credentials.credentials == self.api_key:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Authentication validation error: {e}")
            return False


class APIKeyAuth:
    """API key authentication dependency."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize API key authentication.
        
        Args:
            api_key: API key for authentication
        """
        self.api_key = api_key or os.getenv("API_KEY")
        self.security = HTTPBearer(auto_error=False)
    
    async def __call__(self, request: Request) -> Dict[str, Any]:
        """Validate API key authentication.
        
        Args:
            request: HTTP request
            
        Returns:
            Authentication context
            
        Raises:
            HTTPException: If authentication fails
        """
        if not self.api_key:
            # No API key configured, allow access
            return {"authenticated": False, "user": None}
        
        try:
            # Check for API key in header
            api_key_header = request.headers.get("X-API-Key")
            if api_key_header and api_key_header == self.api_key:
                return {"authenticated": True, "user": "api_user"}
            
            # Check for Bearer token
            credentials: HTTPAuthorizationCredentials = await self.security(request)
            if credentials and credentials.credentials == self.api_key:
                return {"authenticated": True, "user": "api_user"}
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"}
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )


def get_auth_middleware(
    api_key: Optional[str] = None,
    require_auth: bool = False,
    exclude_paths: Optional[list] = None
) -> AuthMiddleware:
    """Get authentication middleware instance.
    
    Args:
        api_key: API key for authentication
        require_auth: Whether authentication is required
        exclude_paths: Paths to exclude from authentication
        
    Returns:
        Authentication middleware instance
    """
    return AuthMiddleware(
        app=None,  # Will be set by FastAPI
        api_key=api_key,
        require_auth=require_auth,
        exclude_paths=exclude_paths
    )


def get_api_key_auth(api_key: Optional[str] = None) -> APIKeyAuth:
    """Get API key authentication dependency.
    
    Args:
        api_key: API key for authentication
        
    Returns:
        API key authentication dependency
    """
    return APIKeyAuth(api_key=api_key) 