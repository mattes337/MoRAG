# Task 08: Multi-tenancy Implementation

## 📋 Task Overview

**Objective**: Implement comprehensive multi-tenancy support ensuring complete data isolation between users, user-specific configurations, and scalable architecture for multiple organizations.

**Priority**: High - Core requirement for production deployment
**Estimated Time**: 1-2 weeks
**Dependencies**: Task 07 (API Key Management)

## 🎯 Goals

1. Implement complete data isolation between users
2. Add user-specific vector database collections
3. Create tenant-aware processing pipelines
4. Implement user-specific configurations
5. Add resource quotas and usage tracking
6. Create tenant management and monitoring
7. Ensure scalable multi-tenant architecture

## 📊 Current State Analysis

### Current MoRAG Architecture
- **Data Storage**: Shared collections in Qdrant
- **Processing**: No user context in tasks
- **Configuration**: Global configuration only
- **Isolation**: No data isolation between users

### Required Multi-tenancy Features
- **Data Isolation**: User-specific collections and namespaces
- **Resource Isolation**: Per-user quotas and limits
- **Configuration Isolation**: User-specific settings
- **Processing Isolation**: User context in all operations

## 🔧 Implementation Plan

### Step 1: Create Tenant Management Service

**Files to Create**:
```
packages/morag-core/src/morag_core/
├── tenancy/
│   ├── __init__.py
│   ├── models.py          # Tenant and quota models
│   ├── service.py         # Tenant management service
│   ├── isolation.py       # Data isolation utilities
│   ├── quotas.py          # Resource quota management
│   └── middleware.py      # Tenant-aware middleware
```

**Implementation Details**:

1. **Tenant Models**:
```python
# packages/morag-core/src/morag_core/tenancy/models.py
"""Multi-tenancy models and data structures."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class TenantStatus(str, Enum):
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    INACTIVE = "INACTIVE"

class ResourceType(str, Enum):
    DOCUMENTS = "DOCUMENTS"
    STORAGE_MB = "STORAGE_MB"
    API_CALLS_PER_HOUR = "API_CALLS_PER_HOUR"
    PROCESSING_JOBS = "PROCESSING_JOBS"
    VECTOR_COLLECTIONS = "VECTOR_COLLECTIONS"

class TenantInfo(BaseModel):
    user_id: str
    email: str
    name: str
    status: TenantStatus
    created_at: datetime
    last_active: Optional[datetime]
    total_documents: int
    total_storage_mb: float
    total_api_calls: int
    active_jobs: int

class ResourceQuota(BaseModel):
    resource_type: ResourceType
    limit: int
    current_usage: int
    percentage_used: float
    
    @validator('percentage_used', always=True)
    def calculate_percentage(cls, v, values):
        limit = values.get('limit', 1)
        current = values.get('current_usage', 0)
        return (current / limit * 100) if limit > 0 else 0

class TenantQuotas(BaseModel):
    user_id: str
    quotas: List[ResourceQuota]
    is_over_limit: bool
    warnings: List[str]

class TenantConfiguration(BaseModel):
    user_id: str
    default_database_id: Optional[str] = None
    default_chunk_size: int = Field(default=4000, ge=500, le=16000)
    default_chunk_overlap: int = Field(default=200, ge=0, le=1000)
    default_chunking_strategy: str = Field(default="SENTENCE")
    enable_auto_processing: bool = Field(default=True)
    webhook_url: Optional[str] = None
    notification_preferences: Dict[str, bool] = Field(default_factory=lambda: {
        "job_completed": True,
        "job_failed": True,
        "quota_warning": True,
        "quota_exceeded": False
    })
    processing_preferences: Dict[str, Any] = Field(default_factory=lambda: {
        "use_docling": False,
        "enable_thumbnails": False,
        "audio_model_size": "base",
        "enable_speaker_diarization": True,
        "enable_topic_segmentation": True
    })

class TenantUsageStats(BaseModel):
    user_id: str
    period_start: datetime
    period_end: datetime
    documents_processed: int
    total_chunks_created: int
    api_calls_made: int
    storage_used_mb: float
    processing_time_seconds: int
    successful_jobs: int
    failed_jobs: int
    average_job_duration: float
```

2. **Tenant Service**:
```python
# packages/morag-core/src/morag_core/tenancy/service.py
"""Tenant management service."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
import structlog
from datetime import datetime, timedelta

from morag_core.database import (
    User, Document, Job, ApiKey, Database, get_database_manager,
    DocumentState, JobStatus
)
from .models import (
    TenantInfo, TenantQuotas, TenantConfiguration, TenantUsageStats,
    ResourceQuota, ResourceType, TenantStatus
)
from morag_core.exceptions import NotFoundError, ValidationError

logger = structlog.get_logger(__name__)

class TenantService:
    """Multi-tenant management service."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def get_tenant_info(self, user_id: str) -> Optional[TenantInfo]:
        """Get comprehensive tenant information."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return None
            
            # Calculate tenant statistics
            total_documents = session.query(Document).filter(
                and_(
                    Document.user_id == user_id,
                    Document.state != DocumentState.DELETED
                )
            ).count()
            
            # Calculate storage usage (approximate)
            storage_query = session.query(func.sum(Document.chunks)).filter(
                and_(
                    Document.user_id == user_id,
                    Document.state == DocumentState.INGESTED
                )
            ).scalar()
            total_storage_mb = (storage_query or 0) * 0.001  # Rough estimate
            
            # Count API calls (from API key usage)
            api_calls_query = session.query(ApiKey).filter_by(user_id=user_id).all()
            total_api_calls = sum(
                key.metadata.get('usage_count', 0) for key in api_calls_query
                if key.metadata
            )
            
            # Count active jobs
            active_jobs = session.query(Job).filter(
                and_(
                    Job.user_id == user_id,
                    Job.status.in_([JobStatus.PENDING, JobStatus.PROCESSING])
                )
            ).count()
            
            # Get last activity
            last_job = session.query(Job).filter_by(user_id=user_id)\
                             .order_by(desc(Job.start_date)).first()
            last_active = last_job.start_date if last_job else None
            
            return TenantInfo(
                user_id=user_id,
                email=user.email,
                name=user.name,
                status=TenantStatus.ACTIVE,  # TODO: Add status to User model
                created_at=user.created_at,
                last_active=last_active,
                total_documents=total_documents,
                total_storage_mb=total_storage_mb,
                total_api_calls=total_api_calls,
                active_jobs=active_jobs
            )
    
    def get_tenant_quotas(self, user_id: str) -> TenantQuotas:
        """Get tenant resource quotas and usage."""
        tenant_info = self.get_tenant_info(user_id)
        if not tenant_info:
            raise NotFoundError(f"Tenant {user_id} not found")
        
        # Default quotas (should be configurable per user/plan)
        default_limits = {
            ResourceType.DOCUMENTS: 1000,
            ResourceType.STORAGE_MB: 5000,  # 5GB
            ResourceType.API_CALLS_PER_HOUR: 1000,
            ResourceType.PROCESSING_JOBS: 10,
            ResourceType.VECTOR_COLLECTIONS: 5
        }
        
        # Calculate current usage
        current_usage = {
            ResourceType.DOCUMENTS: tenant_info.total_documents,
            ResourceType.STORAGE_MB: int(tenant_info.total_storage_mb),
            ResourceType.API_CALLS_PER_HOUR: self._get_hourly_api_calls(user_id),
            ResourceType.PROCESSING_JOBS: tenant_info.active_jobs,
            ResourceType.VECTOR_COLLECTIONS: self._get_collections_count(user_id)
        }
        
        # Create quota objects
        quotas = []
        warnings = []
        is_over_limit = False
        
        for resource_type, limit in default_limits.items():
            usage = current_usage.get(resource_type, 0)
            percentage = (usage / limit * 100) if limit > 0 else 0
            
            quota = ResourceQuota(
                resource_type=resource_type,
                limit=limit,
                current_usage=usage,
                percentage_used=percentage
            )
            quotas.append(quota)
            
            # Check for warnings and limits
            if percentage >= 100:
                is_over_limit = True
                warnings.append(f"{resource_type.value} limit exceeded ({usage}/{limit})")
            elif percentage >= 80:
                warnings.append(f"{resource_type.value} usage high ({percentage:.1f}%)")
        
        return TenantQuotas(
            user_id=user_id,
            quotas=quotas,
            is_over_limit=is_over_limit,
            warnings=warnings
        )
    
    def get_tenant_configuration(self, user_id: str) -> TenantConfiguration:
        """Get tenant-specific configuration."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Get user settings
            user_settings = user.user_settings
            default_database_id = user_settings.default_database if user_settings else None
            
            # TODO: Add tenant configuration table for more settings
            # For now, return defaults with user settings
            return TenantConfiguration(
                user_id=user_id,
                default_database_id=default_database_id
            )
    
    def update_tenant_configuration(self, user_id: str, config: TenantConfiguration) -> TenantConfiguration:
        """Update tenant configuration."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Update user settings
            if user.user_settings:
                if config.default_database_id is not None:
                    user.user_settings.default_database = config.default_database_id
            
            # TODO: Store additional configuration in tenant config table
            
            logger.info("Tenant configuration updated", user_id=user_id)
            return config
    
    def get_tenant_usage_stats(self, user_id: str, days: int = 30) -> TenantUsageStats:
        """Get tenant usage statistics."""
        with self.db_manager.get_session() as session:
            period_end = datetime.utcnow()
            period_start = period_end - timedelta(days=days)
            
            # Documents processed
            documents_processed = session.query(Document).filter(
                and_(
                    Document.user_id == user_id,
                    Document.upload_date >= period_start,
                    Document.state == DocumentState.INGESTED
                )
            ).count()
            
            # Total chunks created
            chunks_query = session.query(func.sum(Document.chunks)).filter(
                and_(
                    Document.user_id == user_id,
                    Document.upload_date >= period_start,
                    Document.state == DocumentState.INGESTED
                )
            ).scalar()
            total_chunks = chunks_query or 0
            
            # Job statistics
            jobs = session.query(Job).filter(
                and_(
                    Job.user_id == user_id,
                    Job.start_date >= period_start
                )
            ).all()
            
            successful_jobs = sum(1 for job in jobs if job.status == JobStatus.FINISHED)
            failed_jobs = sum(1 for job in jobs if job.status == JobStatus.FAILED)
            
            # Calculate processing time and average duration
            total_processing_time = 0
            job_durations = []
            
            for job in jobs:
                if job.end_date:
                    duration = (job.end_date - job.start_date).total_seconds()
                    total_processing_time += duration
                    job_durations.append(duration)
            
            average_duration = sum(job_durations) / len(job_durations) if job_durations else 0
            
            # API calls (approximate from usage count)
            api_keys = session.query(ApiKey).filter_by(user_id=user_id).all()
            api_calls = sum(
                key.metadata.get('usage_count', 0) for key in api_keys
                if key.metadata
            )
            
            # Storage used (approximate)
            storage_used_mb = total_chunks * 0.001  # Rough estimate
            
            return TenantUsageStats(
                user_id=user_id,
                period_start=period_start,
                period_end=period_end,
                documents_processed=documents_processed,
                total_chunks_created=total_chunks,
                api_calls_made=api_calls,
                storage_used_mb=storage_used_mb,
                processing_time_seconds=int(total_processing_time),
                successful_jobs=successful_jobs,
                failed_jobs=failed_jobs,
                average_job_duration=average_duration
            )
    
    def check_quota_limits(self, user_id: str, resource_type: ResourceType, 
                          requested_amount: int = 1) -> bool:
        """Check if user can consume additional resources."""
        quotas = self.get_tenant_quotas(user_id)
        
        for quota in quotas.quotas:
            if quota.resource_type == resource_type:
                return quota.current_usage + requested_amount <= quota.limit
        
        return True  # No quota defined, allow
    
    def _get_hourly_api_calls(self, user_id: str) -> int:
        """Get API calls in the last hour."""
        # TODO: Implement proper API call tracking
        # For now, return 0 as placeholder
        return 0
    
    def _get_collections_count(self, user_id: str) -> int:
        """Get number of vector collections for user."""
        with self.db_manager.get_session() as session:
            return session.query(Database).filter_by(user_id=user_id).count()
```

### Step 2: Create Data Isolation Utilities

**File to Create**: `packages/morag-core/src/morag_core/tenancy/isolation.py`

```python
"""Data isolation utilities for multi-tenancy."""

import structlog
from typing import Optional, Dict, Any, List
from morag_core.auth.models import UserResponse

logger = structlog.get_logger(__name__)

class TenantIsolation:
    """Utilities for ensuring tenant data isolation."""
    
    @staticmethod
    def get_tenant_collection_name(user_id: str, database_name: str = "default") -> str:
        """Generate tenant-specific collection name."""
        # Ensure collection names are unique per tenant
        return f"tenant_{user_id}_{database_name}".replace("-", "_")
    
    @staticmethod
    def get_tenant_namespace(user_id: str) -> str:
        """Get tenant namespace for data isolation."""
        return f"tenant_{user_id}"
    
    @staticmethod
    def validate_tenant_access(current_user: UserResponse, resource_user_id: str) -> bool:
        """Validate that user can access resource."""
        if not current_user:
            return False
        
        # Users can only access their own resources
        # TODO: Add support for shared resources and admin access
        return current_user.id == resource_user_id
    
    @staticmethod
    def add_tenant_filter(query_params: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Add tenant filtering to query parameters."""
        query_params = query_params.copy()
        query_params['user_id'] = user_id
        return query_params
    
    @staticmethod
    def sanitize_metadata(metadata: Dict[str, Any], user_id: str) -> Dict[str, Any]:
        """Add tenant information to metadata."""
        metadata = metadata.copy()
        metadata['tenant_id'] = user_id
        metadata['tenant_namespace'] = TenantIsolation.get_tenant_namespace(user_id)
        return metadata
    
    @staticmethod
    def get_tenant_temp_path(user_id: str, filename: str) -> str:
        """Get tenant-specific temporary file path."""
        import os
        from morag_core.config import get_settings
        
        settings = get_settings()
        tenant_dir = os.path.join(settings.temp_dir, f"tenant_{user_id}")
        os.makedirs(tenant_dir, exist_ok=True)
        
        return os.path.join(tenant_dir, filename)
```

### Step 3: Update Processing Pipeline for Multi-tenancy

**Modify Existing Files**:

1. **Update Ingestion Tasks**:
```python
# In packages/morag/src/morag/ingest_tasks.py
# Add tenant context to all task functions

@celery_app.task(bind=True, name="ingest_file_task")
def ingest_file_task(self, file_path: str, source_type: str, task_options: dict, 
                    tenant_context: dict = None):
    """Ingest file task with tenant isolation."""
    
    tenant_context = tenant_context or {}
    user_id = tenant_context.get('user_id')
    
    if not user_id:
        logger.error("No tenant context provided for ingestion task", task_id=self.request.id)
        raise ValueError("Tenant context required for ingestion")
    
    # Check tenant quotas before processing
    from morag_core.tenancy import TenantService
    tenant_service = TenantService()
    
    if not tenant_service.check_quota_limits(user_id, ResourceType.DOCUMENTS):
        logger.error("Document quota exceeded", user_id=user_id, task_id=self.request.id)
        raise ValueError("Document quota exceeded")
    
    # Use tenant-specific collection
    collection_name = TenantIsolation.get_tenant_collection_name(
        user_id, 
        tenant_context.get('database_name', 'default')
    )
    
    # Add tenant metadata
    enhanced_metadata = TenantIsolation.sanitize_metadata(
        task_options.get('metadata', {}),
        user_id
    )
    
    # Continue with existing processing logic using tenant-specific settings
    # ...
```

2. **Update Vector Storage**:
```python
# In packages/morag-services/src/morag_services/storage.py
# Add tenant-aware collection management

class QdrantVectorStorage:
    def __init__(self, host: str = None, port: int = None, api_key: str = None,
                 collection_name: str = None, tenant_id: str = None):
        # ... existing initialization ...
        
        self.tenant_id = tenant_id
        if tenant_id and collection_name:
            # Use tenant-specific collection name
            from morag_core.tenancy.isolation import TenantIsolation
            self.collection_name = TenantIsolation.get_tenant_collection_name(
                tenant_id, collection_name
            )
    
    async def store_vectors(self, vectors: List[List[float]], 
                           metadata: List[Dict[str, Any]], 
                           collection_name: Optional[str] = None) -> List[str]:
        """Store vectors with tenant isolation."""
        
        # Ensure tenant metadata is added
        if self.tenant_id:
            from morag_core.tenancy.isolation import TenantIsolation
            metadata = [
                TenantIsolation.sanitize_metadata(meta, self.tenant_id) 
                for meta in metadata
            ]
        
        # Continue with existing storage logic
        # ...
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_multi_tenancy.py
import pytest
from morag_core.tenancy import TenantService, TenantIsolation
from morag_core.tenancy.models import ResourceType

def test_tenant_collection_naming():
    """Test tenant-specific collection naming."""
    user_id = "user123"
    database_name = "documents"
    
    collection_name = TenantIsolation.get_tenant_collection_name(user_id, database_name)
    assert collection_name == "tenant_user123_documents"
    assert user_id in collection_name

def test_tenant_quota_checking():
    """Test tenant quota validation."""
    service = TenantService()
    
    # Test quota checking (requires test data setup)
    can_create = service.check_quota_limits("user123", ResourceType.DOCUMENTS, 1)
    assert isinstance(can_create, bool)

def test_tenant_access_validation():
    """Test tenant access control."""
    from morag_core.auth.models import UserResponse
    from datetime import datetime
    
    user = UserResponse(
        id="user123",
        name="Test User",
        email="test@example.com",
        avatar=None,
        role="USER",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )
    
    # User can access their own resources
    assert TenantIsolation.validate_tenant_access(user, "user123")
    
    # User cannot access other user's resources
    assert not TenantIsolation.validate_tenant_access(user, "user456")
```

## 📋 Acceptance Criteria

- [ ] Complete data isolation between tenants implemented
- [ ] Tenant-specific vector database collections working
- [ ] Resource quotas and usage tracking functional
- [ ] Tenant configuration management implemented
- [ ] Multi-tenant processing pipeline working
- [ ] Tenant access control enforced
- [ ] Usage statistics and monitoring available
- [ ] Comprehensive unit tests passing
- [ ] Performance testing with multiple tenants
- [ ] Documentation for multi-tenant deployment

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 09: User Settings and Preferences](./task-09-user-settings-preferences.md)
2. Test multi-tenant scenarios thoroughly
3. Add tenant monitoring and alerting
4. Implement tenant billing and usage reporting

## 📝 Notes

- Ensure complete data isolation at all levels
- Implement proper resource quotas to prevent abuse
- Add comprehensive logging for tenant operations
- Consider implementing tenant-specific rate limiting
- Plan for horizontal scaling with tenant sharding
