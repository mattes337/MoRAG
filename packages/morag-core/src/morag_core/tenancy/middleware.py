"""Multi-tenancy middleware for MoRAG core."""

from typing import Optional, Dict, Any
import structlog

from morag_core.auth.models import UserResponse
from .service import TenantService
from .models import ResourceType

logger = structlog.get_logger(__name__)


class TenantMiddleware:
    """Middleware for multi-tenant operations."""

    def __init__(self):
        self.tenant_service = TenantService()

    def extract_tenant_context(self, current_user: Optional[UserResponse]) -> Dict[str, Any]:
        """Extract tenant context for processing."""
        if not current_user:
            return {
                "user_id": None,
                "user_email": None,
                "collection_name": "morag_documents",
                "tenant_status": "anonymous",
                "quotas_enabled": False
            }

        # Get user-specific collection name
        collection_name = self.tenant_service.get_user_collection_name(
            current_user.id, 
            "default"
        )

        return {
            "user_id": current_user.id,
            "user_email": current_user.email,
            "collection_name": collection_name,
            "tenant_status": "authenticated",
            "quotas_enabled": True
        }

    def check_resource_quota(self, user_id: str, resource_type: ResourceType, amount: int = 1) -> bool:
        """Check if user can allocate resources."""
        if not user_id:
            return True  # No quotas for anonymous users

        try:
            return self.tenant_service.check_resource_quota(user_id, resource_type, amount)
        except Exception as e:
            logger.error("Failed to check resource quota",
                        user_id=user_id,
                        resource_type=resource_type,
                        error=str(e))
            return False

    def get_user_collection_name(self, user_id: Optional[str], database_name: str = "default") -> str:
        """Get collection name for user."""
        if not user_id:
            return "morag_documents"
        
        return self.tenant_service.get_user_collection_name(user_id, database_name)
