"""API Key dependencies for FastAPI."""

from typing import Optional
from fastapi import Depends, HTTPException, Request
import structlog

from morag_core.auth.models import UserResponse
from morag_core.auth.service import UserService
from .middleware import ApiKeyMiddleware

logger = structlog.get_logger(__name__)

# Global instances
api_key_middleware = ApiKeyMiddleware()
user_service = UserService()


async def get_api_key_user(request: Request) -> Optional[UserResponse]:
    """Get user from API key authentication (optional)."""
    try:
        user_id = await api_key_middleware.authenticate_request(request)
        if not user_id:
            return None

        # Get user details
        user = user_service.get_user_by_id(user_id)
        return user

    except Exception as e:
        logger.error("Failed to get API key user", error=str(e))
        return None


async def require_api_key(request: Request) -> UserResponse:
    """Require valid API key authentication."""
    user = await get_api_key_user(request)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Valid API key required"
        )
    return user


# Type aliases for dependency injection
ApiKeyUser = Optional[UserResponse]
RequireApiKey = UserResponse
