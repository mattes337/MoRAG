"""FastAPI authentication middleware."""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import structlog

from .service import UserService
from .security import JWTManager
from .models import UserResponse

logger = structlog.get_logger(__name__)


class AuthenticationMiddleware:
    """Authentication middleware for FastAPI."""
    
    def __init__(self):
        self.user_service = UserService()
        self.jwt_manager = JWTManager()
        self.security = HTTPBearer(auto_error=False)
    
    async def get_current_user(self, request: Request) -> Optional[UserResponse]:
        """Get current authenticated user from request."""
        try:
            credentials: HTTPAuthorizationCredentials = await self.security(request)
            
            if not credentials:
                return None
            
            token_data = self.jwt_manager.decode_token(credentials.credentials)
            if not token_data:
                return None
            
            user = self.user_service.get_user_by_id(token_data["sub"])
            return user
            
        except Exception as e:
            logger.warning("Failed to get current user", error=str(e))
            return None
    
    async def require_authentication(self, request: Request) -> UserResponse:
        """Require authentication, raise exception if not authenticated."""
        user = await self.get_current_user(request)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user
    
    async def require_role(self, request: Request, required_role: str) -> UserResponse:
        """Require specific role, raise exception if insufficient permissions."""
        user = await self.require_authentication(request)
        
        # Role hierarchy: ADMIN > USER > VIEWER
        role_hierarchy = {"ADMIN": 3, "USER": 2, "VIEWER": 1}
        user_level = role_hierarchy.get(user.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role {required_role} or higher required",
            )
        
        return user
    
    async def require_admin(self, request: Request) -> UserResponse:
        """Require admin role."""
        return await self.require_role(request, "ADMIN")
    
    async def require_user(self, request: Request) -> UserResponse:
        """Require user role or higher."""
        return await self.require_role(request, "USER")
    
    def extract_user_context(self, user: Optional[UserResponse]) -> dict:
        """Extract user context for task processing."""
        if not user:
            return {
                "user_id": None,
                "user_email": None,
                "user_role": None,
                "collection_name": "morag_documents"  # Default collection
            }
        
        return {
            "user_id": user.id,
            "user_email": user.email,
            "user_role": user.role,
            "collection_name": f"user_{user.id}_documents"
        }
    
    def get_user_collection_name(self, user: Optional[UserResponse], database_name: str = "default") -> str:
        """Get collection name for user and database."""
        if not user:
            return "morag_documents"
        return f"user_{user.id}_{database_name}"
