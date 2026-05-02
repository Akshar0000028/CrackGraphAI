"""
Database integration for FastAPI endpoints.
Provides middleware and utilities to automatically save analysis results to database.
"""

import hashlib
import logging
import time
from typing import Optional

from fastapi import Request
from sqlalchemy.orm import Session

from db.service import AnalysisService, APIAuditService

logger = logging.getLogger(__name__)


class DatabaseMiddleware:
    """Middleware to log API requests to database."""

    def __init__(self, get_db_session):
        self.get_db_session = get_db_session

    async def log_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time_ms: float,
        request_id: Optional[str] = None,
        api_key: Optional[str] = None,
        error_message: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> None:
        """Log API request to database."""
        try:
            session = self.get_db_session()
            try:
                audit_service = APIAuditService(session)
                audit_service.log_request(
                    endpoint=endpoint,
                    method=method,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    request_id=request_id,
                    api_key=api_key,
                    error_message=error_message,
                    ip_address=ip_address,
                )
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to log request to database: {e}")


class AnalysisResultSaver:
    """Utility to save analysis results to database."""

    def __init__(self, get_db_session):
        self.get_db_session = get_db_session

    def save_result(
        self,
        request_id: str,
        result: dict,
        input_filename: Optional[str] = None,
        input_file_size: Optional[int] = None,
        input_file_bytes: Optional[bytes] = None,
        api_key: Optional[str] = None,
        model_version: Optional[str] = None,
        image_output_dir: Optional[str] = None,
    ) -> Optional[int]:
        """
        Save analysis result to database.

        Returns:
            analysis_id if successful, None if failed
        """
        try:
            session = self.get_db_session()
            try:
                service = AnalysisService(session)
                analysis_id = service.save_analysis_result(
                    request_id=request_id,
                    result=result,
                    input_filename=input_filename,
                    input_file_size=input_file_size,
                    input_file_bytes=input_file_bytes,
                    api_key=api_key,
                    model_version=model_version,
                    image_output_dir=image_output_dir,
                )
                return analysis_id
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to save analysis result: {e}", exc_info=True)
            return None

    def get_result(self, request_id: str) -> Optional[dict]:
        """Retrieve analysis result from database."""
        try:
            session = self.get_db_session()
            try:
                service = AnalysisService(session)
                return service.get_analysis_with_details(request_id)
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to retrieve analysis result: {e}")
            return None

    def get_statistics(self) -> dict:
        """Get aggregate statistics."""
        try:
            session = self.get_db_session()
            try:
                service = AnalysisService(session)
                return service.get_statistics()
            finally:
                session.close()
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


def extract_api_key_from_request(request: Request) -> Optional[str]:
    """Extract API key from Authorization header."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:]
    return None


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check X-Forwarded-For header first (for proxies)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    
    # Fall back to client connection
    if request.client:
        return request.client.host
    
    return "unknown"
