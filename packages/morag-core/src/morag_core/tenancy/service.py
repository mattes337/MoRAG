"""Multi-tenancy service for MoRAG core."""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import and_, func
import structlog

from morag_core.database import (
    get_database_manager,
    User,
    Document,
    Job,
    ApiKey,
    Database,
    DocumentState,
    JobStatus,
    get_session_context
)
from morag_core.exceptions import NotFoundError, ValidationError
from .models import (
    TenantInfo,
    TenantQuotas,
    TenantConfiguration,
    TenantUsageStats,
    TenantStatus,
    ResourceQuota,
    ResourceType,
    TenantCreate,
    TenantUpdate,
    TenantSearchRequest,
    TenantSearchResponse,
)

logger = structlog.get_logger(__name__)


class TenantService:
    """Multi-tenant management service."""

    def __init__(self):
        self.db_manager = get_database_manager()

    def get_tenant_info(self, user_id: str) -> Optional[TenantInfo]:
        """Get comprehensive tenant information."""
        try:
            with get_session_context(self.db_manager) as session:
                user = session.query(User).filter_by(id=user_id).first()
                if not user:
                    return None

                # Calculate tenant statistics
                usage_stats = self._calculate_usage_stats(session, user_id)
                quotas = self._get_default_quotas(usage_stats)
                configuration = self._get_default_configuration()

                return TenantInfo(
                    user_id=user.id,
                    user_name=user.name,
                    user_email=user.email,
                    status=TenantStatus.ACTIVE,  # Default status
                    created_at=user.created_at,
                    last_login=None,  # TODO: Track login times
                    quotas=quotas,
                    configuration=configuration,
                    usage_stats=usage_stats,
                    metadata={}
                )

        except Exception as e:
            logger.error("Failed to get tenant info",
                        user_id=user_id,
                        error=str(e))
            raise

    def check_resource_quota(self, user_id: str, resource_type: ResourceType, requested_amount: int = 1) -> bool:
        """Check if user can allocate requested resources."""
        try:
            tenant_info = self.get_tenant_info(user_id)
            if not tenant_info:
                return False

            return tenant_info.quotas.check_quota(resource_type, requested_amount)

        except Exception as e:
            logger.error("Failed to check resource quota",
                        user_id=user_id,
                        resource_type=resource_type,
                        error=str(e))
            return False

    def get_user_collection_name(self, user_id: str, database_name: str = "default") -> str:
        """Get collection name for user and database."""
        return f"user_{user_id}_{database_name}"

    def get_user_collections(self, user_id: str) -> List[str]:
        """Get all collections for a user."""
        try:
            with get_session_context(self.db_manager) as session:
                databases = session.query(Database).filter_by(user_id=user_id).all()
                return [self.get_user_collection_name(user_id, db.name) for db in databases]

        except Exception as e:
            logger.error("Failed to get user collections",
                        user_id=user_id,
                        error=str(e))
            return []

    def _calculate_usage_stats(self, session, user_id: str) -> TenantUsageStats:
        """Calculate usage statistics for a tenant."""
        try:
            # Document statistics
            total_documents = session.query(Document).filter(
                and_(
                    Document.user_id == user_id,
                    Document.state != DocumentState.DELETED
                )
            ).count()

            # Job statistics
            successful_jobs = session.query(Job).filter(
                and_(
                    Job.user_id == user_id,
                    Job.status == JobStatus.FINISHED
                )
            ).count()

            failed_jobs = session.query(Job).filter(
                and_(
                    Job.user_id == user_id,
                    Job.status == JobStatus.FAILED
                )
            ).count()

            # API key count
            total_api_keys = session.query(ApiKey).filter_by(user_id=user_id).count()

            # Database count
            total_databases = session.query(Database).filter_by(user_id=user_id).count()

            # Average processing time (placeholder)
            avg_processing_time = 0.0

            # Last activity (most recent job)
            last_job = session.query(Job).filter_by(user_id=user_id).order_by(
                Job.created_at.desc()
            ).first()
            last_activity = last_job.created_at if last_job else None

            return TenantUsageStats(
                total_documents=total_documents,
                total_storage_mb=0.0,  # TODO: Calculate actual storage
                total_api_calls_today=0,  # TODO: Implement API call tracking
                total_api_calls_this_month=0,
                total_vector_points=0,  # TODO: Calculate from vector database
                total_databases=total_databases,
                total_api_keys=total_api_keys,
                last_activity=last_activity,
                avg_processing_time_seconds=avg_processing_time,
                successful_jobs=successful_jobs,
                failed_jobs=failed_jobs
            )

        except Exception as e:
            logger.error("Failed to calculate usage stats",
                        user_id=user_id,
                        error=str(e))
            return TenantUsageStats()

    def _get_default_quotas(self, usage_stats: TenantUsageStats) -> TenantQuotas:
        """Get default quotas for a tenant."""
        return TenantQuotas(
            quotas=[
                ResourceQuota(
                    resource_type=ResourceType.DOCUMENTS,
                    limit=1000,
                    used=usage_stats.total_documents
                ),
                ResourceQuota(
                    resource_type=ResourceType.STORAGE_MB,
                    limit=10000,  # 10GB
                    used=int(usage_stats.total_storage_mb)
                ),
                ResourceQuota(
                    resource_type=ResourceType.API_CALLS_PER_HOUR,
                    limit=1000,
                    used=0  # Reset hourly
                ),
                ResourceQuota(
                    resource_type=ResourceType.VECTOR_POINTS,
                    limit=100000,
                    used=usage_stats.total_vector_points
                ),
                ResourceQuota(
                    resource_type=ResourceType.DATABASES,
                    limit=10,
                    used=usage_stats.total_databases
                ),
                ResourceQuota(
                    resource_type=ResourceType.API_KEYS,
                    limit=20,
                    used=usage_stats.total_api_keys
                ),
            ]
        )

    def _get_default_configuration(self) -> TenantConfiguration:
        """Get default configuration for a tenant."""
        return TenantConfiguration(
            default_database_id=None,
            default_collection_name=None,
            allowed_file_types=["*"],
            max_file_size_mb=100,
            enable_remote_processing=True,
            enable_webhooks=True,
            custom_settings={}
        )

    def list_tenants(self, search_request: TenantSearchRequest) -> TenantSearchResponse:
        """List tenants with filtering."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(User)

                # Apply filters
                if search_request.email_contains:
                    query = query.filter(User.email.contains(search_request.email_contains))

                if search_request.name_contains:
                    query = query.filter(User.name.contains(search_request.name_contains))

                if search_request.created_after:
                    query = query.filter(User.created_at >= search_request.created_after)

                if search_request.created_before:
                    query = query.filter(User.created_at <= search_request.created_before)

                # Get total count
                total = query.count()

                # Apply pagination
                users = query.offset(search_request.skip).limit(search_request.limit).all()

                # Convert to tenant info
                tenants = []
                for user in users:
                    tenant_info = self.get_tenant_info(user.id)
                    if tenant_info:
                        tenants.append(tenant_info)

                return TenantSearchResponse(
                    tenants=tenants,
                    total=total,
                    skip=search_request.skip,
                    limit=search_request.limit
                )

        except Exception as e:
            logger.error("Failed to list tenants", error=str(e))
            raise
