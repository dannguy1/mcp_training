"""
Logging middleware for FastAPI.
"""

import time
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ...utils.logger import get_logger, log_with_context

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging HTTP requests and responses."""
    
    def __init__(self, app: ASGIApp, log_requests: bool = True, log_responses: bool = True):
        """Initialize logging middleware.
        
        Args:
            app: ASGI application
            log_requests: Whether to log requests
            log_responses: Whether to log responses
        """
        super().__init__(app)
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response with logging.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        # Log request
        if self.log_requests:
            await self._log_request(request)
        
        # Process request
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Log response
            if self.log_responses:
                await self._log_response(request, response, process_time)
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"Request failed: {request.method} {request.url} - {e}")
            raise
    
    async def _log_request(self, request: Request):
        """Log incoming request details.
        
        Args:
            request: HTTP request
        """
        # Extract request details
        method = request.method
        url = str(request.url)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request
        log_with_context(
            logger, "info", f"Request: {method} {url}",
            method=method,
            url=url,
            client_ip=client_ip,
            user_agent=user_agent,
            content_type=request.headers.get("content-type"),
            content_length=request.headers.get("content-length")
        )
    
    async def _log_response(self, request: Request, response: Response, process_time: float):
        """Log response details.
        
        Args:
            request: HTTP request
            response: HTTP response
            process_time: Request processing time
        """
        # Extract response details
        status_code = response.status_code
        method = request.method
        url = str(request.url)
        
        # Determine log level based on status code
        if status_code >= 500:
            log_level = "error"
        elif status_code >= 400:
            log_level = "warning"
        else:
            log_level = "info"
        
        # Log response
        log_with_context(
            logger, log_level, f"Response: {method} {url} - {status_code}",
            method=method,
            url=url,
            status_code=status_code,
            process_time=process_time,
            content_type=response.headers.get("content-type"),
            content_length=response.headers.get("content-length")
        )


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware for adding request ID to requests and responses."""
    
    def __init__(self, app: ASGIApp, header_name: str = "X-Request-ID"):
        """Initialize request ID middleware.
        
        Args:
            app: ASGI application
            header_name: Header name for request ID
        """
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Add request ID to request and response.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        import uuid
        
        # Generate or extract request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring."""
    
    def __init__(self, app: ASGIApp, slow_request_threshold: float = 1.0):
        """Initialize performance middleware.
        
        Args:
            app: ASGI application
            slow_request_threshold: Threshold for slow request logging (seconds)
        """
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Monitor request performance.
        
        Args:
            request: HTTP request
            call_next: Next middleware/endpoint
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        # Log slow requests
        if process_time > self.slow_request_threshold:
            log_with_context(
                logger, "warning", f"Slow request detected: {request.method} {request.url}",
                method=request.method,
                url=str(request.url),
                process_time=process_time,
                threshold=self.slow_request_threshold,
                status_code=response.status_code
            )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response 