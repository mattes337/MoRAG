"""Multi-tenancy models for MoRAG core."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field


class TenantStatus(str, Enum):
    """Tenant status."""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    INACTIVE = "inactive"
    TRIAL = "trial"


class ResourceType(str, Enum):
    """Resource types for quotas."""
    DOCUMENTS = "documents"
    STORAGE_MB = "storage_mb"
    API_CALLS_PER_HOUR = "api_calls_per_hour"
    VECTOR_POINTS = "vector_points"
    DATABASES = "databases"
    API_KEYS = "api_keys"


class ResourceQuota(BaseModel):
    """Resource quota definition."""
    resource_type: ResourceType
    limit: int
    used: int = 0
    warning_threshold: float = Field(default=0.8, ge=0.0, le=1.0)  # 80% warning
    
    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage."""
        if self.limit == 0:
            return 0.0
        return min(self.used / self.limit, 1.0)
    
    @property
    def is_warning(self) -> bool:
        """Check if usage is above warning threshold."""
        return self.usage_percentage >= self.warning_threshold
    
    @property
    def is_exceeded(self) -> bool:
        """Check if quota is exceeded."""
        return self.used >= self.limit


class TenantQuotas(BaseModel):
    """Tenant resource quotas."""
    quotas: List[ResourceQuota]
    
    def get_quota(self, resource_type: ResourceType) -> Optional[ResourceQuota]:
        """Get quota for specific resource type."""
        for quota in self.quotas:
            if quota.resource_type == resource_type:
                return quota
        return None
    
    def check_quota(self, resource_type: ResourceType, requested_amount: int = 1) -> bool:
        """Check if resource can be allocated."""
        quota = self.get_quota(resource_type)
        if not quota:
            return True  # No quota defined, allow
        return quota.used + requested_amount <= quota.limit


class TenantConfiguration(BaseModel):
    """Tenant-specific configuration."""
    default_database_id: Optional[str] = None
    default_collection_name: Optional[str] = None
    allowed_file_types: List[str] = Field(default_factory=lambda: ["*"])
    max_file_size_mb: int = Field(default=100)
    enable_remote_processing: bool = Field(default=True)
    enable_webhooks: bool = Field(default=True)
    custom_settings: Dict[str, Any] = Field(default_factory=dict)


class TenantUsageStats(BaseModel):
    """Tenant usage statistics."""
    total_documents: int = 0
    total_storage_mb: float = 0.0
    total_api_calls_today: int = 0
    total_api_calls_this_month: int = 0
    total_vector_points: int = 0
    total_databases: int = 0
    total_api_keys: int = 0
    last_activity: Optional[datetime] = None
    avg_processing_time_seconds: float = 0.0
    successful_jobs: int = 0
    failed_jobs: int = 0


class TenantInfo(BaseModel):
    """Complete tenant information."""
    user_id: str
    user_name: str
    user_email: str
    status: TenantStatus
    created_at: datetime
    last_login: Optional[datetime]
    quotas: TenantQuotas
    configuration: TenantConfiguration
    usage_stats: TenantUsageStats
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TenantCreate(BaseModel):
    """Tenant creation request."""
    user_id: str
    quotas: Optional[TenantQuotas] = None
    configuration: Optional[TenantConfiguration] = None
    metadata: Optional[Dict[str, Any]] = None


class TenantUpdate(BaseModel):
    """Tenant update request."""
    status: Optional[TenantStatus] = None
    quotas: Optional[TenantQuotas] = None
    configuration: Optional[TenantConfiguration] = None
    metadata: Optional[Dict[str, Any]] = None


class TenantSearchRequest(BaseModel):
    """Tenant search request."""
    status: Optional[TenantStatus] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    last_login_after: Optional[datetime] = None
    last_login_before: Optional[datetime] = None
    email_contains: Optional[str] = None
    name_contains: Optional[str] = None
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class TenantSearchResponse(BaseModel):
    """Tenant search response."""
    tenants: List[TenantInfo]
    total: int
    skip: int
    limit: int
