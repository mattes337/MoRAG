"""Administrative endpoints for MoRAG API."""

import structlog
from fastapi import APIRouter, HTTPException

from morag.services.cleanup_service import force_cleanup

logger = structlog.get_logger(__name__)

admin_router = APIRouter(prefix="/api/v1/admin", tags=["Administration"])


def setup_admin_endpoints(morag_api_getter):
    """Setup administrative endpoints with MoRAG API getter function."""
    
    @admin_router.post("/cleanup")
    async def force_temp_cleanup():
        """Force immediate cleanup of old temporary files."""
        try:
            deleted_count = force_cleanup()

            logger.info("Manual cleanup completed", files_deleted=deleted_count)
            return {
                "message": f"Cleanup completed. {deleted_count} files deleted.",
                "files_deleted": deleted_count
            }

        except Exception as e:
            logger.error("Manual cleanup failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    return admin_router
