"""Authentication package for MoRAG core."""

from .models import (
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
    UserSettingsUpdate,
    UserSettingsResponse,
    TokenResponse,
    PasswordChangeRequest,
    UserRole,
    Theme,
)
from .service import UserService
from .security import PasswordManager, JWTManager
from .middleware import AuthenticationMiddleware
from .dependencies import (
    get_current_user,
    require_authentication,
    require_admin,
    require_user,
    CurrentUser,
    RequireAuth,
    RequireAdmin,
    RequireUser,
)

__all__ = [
    # Models
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "UserUpdate",
    "UserSettingsUpdate",
    "UserSettingsResponse",
    "TokenResponse",
    "PasswordChangeRequest",
    "UserRole",
    "Theme",
    # Services
    "UserService",
    "PasswordManager",
    "JWTManager",
    "AuthenticationMiddleware",
    # Dependencies
    "get_current_user",
    "require_authentication",
    "require_admin",
    "require_user",
    "CurrentUser",
    "RequireAuth",
    "RequireAdmin",
    "RequireUser",
]
