"""FastAPI dependencies for authentication."""

from fastapi import Depends, Request
from typing import Optional

from .middleware import AuthenticationMiddleware
from .models import UserResponse

# Global middleware instance
auth_middleware = AuthenticationMiddleware()


async def get_current_user(request: Request) -> Optional[UserResponse]:
    """FastAPI dependency to get current user."""
    return await auth_middleware.get_current_user(request)


async def require_authentication(request: Request) -> UserResponse:
    """FastAPI dependency to require authentication."""
    return await auth_middleware.require_authentication(request)


async def require_admin(request: Request) -> UserResponse:
    """FastAPI dependency to require admin role."""
    return await auth_middleware.require_admin(request)


async def require_user(request: Request) -> UserResponse:
    """FastAPI dependency to require user role or higher."""
    return await auth_middleware.require_user(request)


# Convenience dependencies
CurrentUser = Depends(get_current_user)
RequireAuth = Depends(require_authentication)
RequireAdmin = Depends(require_admin)
RequireUser = Depends(require_user)
