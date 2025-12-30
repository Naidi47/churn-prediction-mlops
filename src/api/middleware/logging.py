"""
Enhanced Logging Middleware
Structured logging with request/response correlation
"""

import time
import json
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import logging

structlog = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced logging middleware with structured logging"""
    
    def __init__(self, app):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Extract request details
        request_id = getattr(request.state, 'request_id', 'unknown')
        method = request.method
        path = request.url.path
        query_params = str(request.query_params) if request.query_params else None
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        content_type = request.headers.get("content-type", "unknown")
        
        # Log request
        request_log = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "query_params": query_params,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "content_type": content_type,
            "event": "request_started",
            "timestamp": time.time()
        }
        
        structlog.info("Request started", extra=request_log)
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            error_log = {
                "request_id": request_id,
                "method": method,
                "path": path,
                "error": str(e),
                "event": "request_exception",
                "timestamp": time.time()
            }
            structlog.error("Request failed with exception", extra=error_log)
            raise
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Extract response details
        status_code = response.status_code
        response_headers = dict(response.headers)
        content_length = response_headers.get("content-length", "unknown")
        
        # Log response
        response_log = {
            "request_id": request_id,
            "method": method,
            "path": path,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "content_length": content_length,
            "event": "request_completed",
            "timestamp": time.time()
        }
        
        # Log level based on status code
        if status_code >= 500:
            structlog.error("Server error", extra=response_log)
        elif status_code >= 400:
            structlog.warning("Client error", extra=response_log)
        elif status_code >= 300:
            structlog.info("Redirect", extra=response_log)
        else:
            structlog.info("Success", extra=response_log)
        
        return response