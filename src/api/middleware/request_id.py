"""
Request ID Middleware
Adds request IDs for distributed tracing and request correlation
"""

import uuid
from typing import Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

structlog = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Middleware to add request IDs for tracing"""
    
    def __init__(self, app, header_name: str = "X-Request-ID"):
        super().__init__(app)
        self.header_name = header_name
    
    async def dispatch(self, request: Request, call_next):
        # Get or generate request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = str(uuid.uuid4())
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        # Log request
        structlog.info(
            "Request started",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown"
        )
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response headers
        response.headers[self.header_name] = request_id
        
        # Log response
        structlog.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code
        )
        
        return response