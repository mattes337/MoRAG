"""Authentication middleware for API key validation."""

from typing import Optional, Dict, Any
from fastapi import Request, HTTPException
import structlog

from morag.services.auth_service import APIKeyService

logger = structlog.get_logger(__name__)

class APIKeyAuth:
    """Authentication middleware for API key validation."""

    def __init__(self, api_key_service: APIKeyService):
        self.api_key_service = api_key_service

    async def get_current_user(self, request: Request) -> Optional[Dict[str, Any]]:
        """Get current user from API key, return None for anonymous access."""
        try:
            # Check for API key in headers
            api_key = None
            
            # Try Authorization header first (Bearer token)
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                api_key = auth_header[7:]  # Remove "Bearer " prefix
            
            # Try X-API-Key header as fallback
            if not api_key:
                api_key = request.headers.get("X-API-Key")
            
            # If no API key provided, allow anonymous access
            if not api_key:
                return None
            
            # Validate API key
            user_data = await self.api_key_service.validate_api_key(api_key)
            if not user_data:
                logger.warning("Invalid API key provided", 
                             client_ip=request.client.host if request.client else "unknown")
                return None
            
            logger.debug("User authenticated via API key", 
                        user_id=user_data.get("user_id"),
                        client_ip=request.client.host if request.client else "unknown")
            
            return user_data
            
        except Exception as e:
            logger.error("Error during API key authentication", error=str(e))
            return None

    async def get_user_id(self, request: Request) -> Optional[str]:
        """Get user ID from API key, return None for anonymous."""
        user_data = await self.get_current_user(request)
        return user_data.get("user_id") if user_data else None

    async def require_authentication(self, request: Request) -> Dict[str, Any]:
        """Require valid API key authentication."""
        user_data = await self.get_current_user(request)
        if not user_data:
            raise HTTPException(
                status_code=401,
                detail="Valid API key required",
                headers={"WWW-Authenticate": "Bearer"}
            )
        return user_data

    def get_api_key_from_request(self, request: Request) -> Optional[str]:
        """Extract API key from request headers."""
        # Try Authorization header first (Bearer token)
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        
        # Try X-API-Key header as fallback
        return request.headers.get("X-API-Key")
