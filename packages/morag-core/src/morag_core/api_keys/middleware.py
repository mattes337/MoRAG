"""API Key middleware for MoRAG core."""

from typing import Optional
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from .service import ApiKeyService

logger = structlog.get_logger(__name__)


class ApiKeyMiddleware:
    """Middleware for API key authentication."""

    def __init__(self):
        self.api_key_service = ApiKeyService()
        self.security = HTTPBearer(auto_error=False)

    async def authenticate_request(self, request: Request) -> Optional[str]:
        """Authenticate request using API key and return user ID."""
        try:
            # Try to get API key from Authorization header
            api_key = None
            
            # Check Authorization header (Bearer token)
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
            
            # Check X-API-Key header
            elif request.headers.get("X-API-Key"):
                api_key = request.headers.get("X-API-Key")
            
            # Check api_key query parameter
            elif request.query_params.get("api_key"):
                api_key = request.query_params.get("api_key")

            if not api_key:
                return None

            # Authenticate API key
            user_id = self.api_key_service.authenticate_api_key(api_key)
            
            if user_id:
                logger.debug("API key authentication successful",
                           user_id=user_id,
                           endpoint=str(request.url.path))
            else:
                logger.warning("API key authentication failed",
                             endpoint=str(request.url.path),
                             ip=request.client.host if request.client else "unknown")

            return user_id

        except Exception as e:
            logger.error("API key authentication error",
                        endpoint=str(request.url.path),
                        error=str(e))
            return None

    def extract_api_key_context(self, user_id: Optional[str]) -> dict:
        """Extract API key context for logging and processing."""
        if not user_id:
            return {
                "user_id": None,
                "user_email": None,
                "authentication_method": "none",
                "collection_name": "morag_documents"
            }

        return {
            "user_id": user_id,
            "user_email": None,  # Would need to fetch from database
            "authentication_method": "api_key",
            "collection_name": f"user_{user_id}_documents"
        }
