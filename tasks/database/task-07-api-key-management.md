# Task 07: API Key Management

## 📋 Task Overview

**Objective**: Implement comprehensive API key management system for programmatic access to MoRAG, including key generation, validation, usage tracking, and role-based permissions.

**Priority**: Medium-High - Required for programmatic access
**Estimated Time**: 1 week
**Dependencies**: Task 06 (Database Server Management)

## 🎯 Goals

1. Implement API key generation and management
2. Add API key authentication alongside JWT
3. Create usage tracking and rate limiting
4. Implement role-based API key permissions
5. Add API key expiration and rotation
6. Create API key management endpoints
7. Integrate with existing authentication system

## 📊 Current State Analysis

### Existing API Key Model
- **Fields**: ID, name, key, created, last_used, user_id
- **Features**: Basic key storage and user association
- **Security**: No encryption or hashing

### Current MoRAG Authentication
- **System**: JWT-based user authentication only
- **API Access**: No programmatic API key support
- **Rate Limiting**: No rate limiting implemented

## 🔧 Implementation Plan

### Step 1: Enhance API Key Model and Service

**Files to Create**:
```
packages/morag-core/src/morag_core/
├── api_keys/
│   ├── __init__.py
│   ├── models.py          # Pydantic models for API keys
│   ├── service.py         # API key service logic
│   ├── generator.py       # Key generation utilities
│   ├── validator.py       # Key validation and authentication
│   └── middleware.py      # API key middleware
```

**Implementation Details**:

1. **API Key Models**:
```python
# packages/morag-core/src/morag_core/api_keys/models.py
"""API key management models."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum

class ApiKeyPermission(str, Enum):
    READ = "READ"           # Read-only access
    WRITE = "WRITE"         # Write access (create/update)
    DELETE = "DELETE"       # Delete access
    ADMIN = "ADMIN"         # Full administrative access

class ApiKeyStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"

class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    permissions: List[ApiKeyPermission] = Field(default=[ApiKeyPermission.READ])
    expires_at: Optional[datetime] = None
    rate_limit_per_hour: Optional[int] = Field(default=1000, ge=1, le=10000)
    allowed_ips: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ApiKeyUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    permissions: Optional[List[ApiKeyPermission]] = None
    expires_at: Optional[datetime] = None
    rate_limit_per_hour: Optional[int] = Field(None, ge=1, le=10000)
    allowed_ips: Optional[List[str]] = None
    status: Optional[ApiKeyStatus] = None
    metadata: Optional[Dict[str, Any]] = None

class ApiKeyResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    key_prefix: str  # First 8 characters for identification
    permissions: List[ApiKeyPermission]
    status: ApiKeyStatus
    expires_at: Optional[datetime]
    rate_limit_per_hour: int
    allowed_ips: Optional[List[str]]
    created_at: datetime
    last_used: Optional[datetime]
    usage_count: int
    user_id: str
    metadata: Optional[Dict[str, Any]]

class ApiKeyCreateResponse(BaseModel):
    api_key: ApiKeyResponse
    secret_key: str  # Full key returned only once

class ApiKeyUsage(BaseModel):
    api_key_id: str
    endpoint: str
    method: str
    timestamp: datetime
    ip_address: str
    user_agent: Optional[str]
    response_status: int
    response_time_ms: float

class ApiKeyUsageStats(BaseModel):
    api_key_id: str
    total_requests: int
    requests_last_24h: int
    requests_last_7d: int
    requests_last_30d: int
    average_response_time_ms: float
    error_rate: float
    most_used_endpoints: List[Dict[str, Any]]
```

2. **API Key Generator**:
```python
# packages/morag-core/src/morag_core/api_keys/generator.py
"""API key generation utilities."""

import secrets
import hashlib
import base64
from typing import Tuple
import structlog

logger = structlog.get_logger(__name__)

class ApiKeyGenerator:
    """Generate and manage API keys."""
    
    @staticmethod
    def generate_api_key() -> Tuple[str, str]:
        """Generate a new API key and its hash.
        
        Returns:
            Tuple of (api_key, key_hash) where:
            - api_key: The full key to return to user (only shown once)
            - key_hash: The hashed version to store in database
        """
        # Generate random bytes
        key_bytes = secrets.token_bytes(32)
        
        # Create API key with prefix
        api_key = f"morag_{base64.urlsafe_b64encode(key_bytes).decode().rstrip('=')}"
        
        # Create hash for storage
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        
        logger.info("API key generated", key_prefix=api_key[:12])
        return api_key, key_hash
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def get_key_prefix(api_key: str) -> str:
        """Get the prefix of an API key for identification."""
        return api_key[:12] if len(api_key) >= 12 else api_key
    
    @staticmethod
    def validate_key_format(api_key: str) -> bool:
        """Validate API key format."""
        return (
            api_key.startswith("morag_") and
            len(api_key) > 20 and
            all(c.isalnum() or c in "-_" for c in api_key[6:])
        )
```

3. **API Key Service**:
```python
# packages/morag-core/src/morag_core/api_keys/service.py
"""API key management service."""

from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, desc, func
import structlog
from datetime import datetime, timedelta

from morag_core.database import ApiKey, get_database_manager
from .models import (
    ApiKeyCreate, ApiKeyUpdate, ApiKeyResponse, ApiKeyCreateResponse,
    ApiKeyUsage, ApiKeyUsageStats, ApiKeyStatus, ApiKeyPermission
)
from .generator import ApiKeyGenerator
from morag_core.exceptions import NotFoundError, ValidationError, ConflictError

logger = structlog.get_logger(__name__)

class ApiKeyService:
    """API key management service."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.generator = ApiKeyGenerator()
    
    def create_api_key(self, user_id: str, key_data: ApiKeyCreate) -> ApiKeyCreateResponse:
        """Create a new API key."""
        with self.db_manager.get_session() as session:
            # Check for duplicate names
            existing = session.query(ApiKey).filter(
                and_(
                    ApiKey.user_id == user_id,
                    ApiKey.name == key_data.name
                )
            ).first()
            
            if existing:
                raise ConflictError(f"API key '{key_data.name}' already exists")
            
            # Generate API key
            api_key, key_hash = self.generator.generate_api_key()
            key_prefix = self.generator.get_key_prefix(api_key)
            
            # Create API key record
            api_key_record = ApiKey(
                name=key_data.name,
                key=key_hash,  # Store hash, not plain key
                user_id=user_id,
                metadata={
                    'description': key_data.description,
                    'permissions': [p.value for p in key_data.permissions],
                    'expires_at': key_data.expires_at.isoformat() if key_data.expires_at else None,
                    'rate_limit_per_hour': key_data.rate_limit_per_hour,
                    'allowed_ips': key_data.allowed_ips,
                    'key_prefix': key_prefix,
                    'status': ApiKeyStatus.ACTIVE.value,
                    'usage_count': 0,
                    **(key_data.metadata or {})
                }
            )
            
            session.add(api_key_record)
            session.flush()
            
            logger.info("API key created", 
                       api_key_id=api_key_record.id,
                       name=key_data.name,
                       user_id=user_id,
                       key_prefix=key_prefix)
            
            return ApiKeyCreateResponse(
                api_key=self._api_key_to_response(api_key_record),
                secret_key=api_key  # Return full key only once
            )
    
    def get_api_key(self, api_key_id: str, user_id: str) -> Optional[ApiKeyResponse]:
        """Get API key by ID with user ownership check."""
        with self.db_manager.get_session() as session:
            api_key = session.query(ApiKey).filter(
                and_(
                    ApiKey.id == api_key_id,
                    ApiKey.user_id == user_id
                )
            ).first()
            
            if api_key:
                return self._api_key_to_response(api_key)
            return None
    
    def list_api_keys(self, user_id: str) -> List[ApiKeyResponse]:
        """List all API keys for user."""
        with self.db_manager.get_session() as session:
            api_keys = session.query(ApiKey).filter(
                ApiKey.user_id == user_id
            ).order_by(desc(ApiKey.created_at)).all()
            
            return [self._api_key_to_response(key) for key in api_keys]
    
    def update_api_key(self, api_key_id: str, user_id: str, 
                      key_data: ApiKeyUpdate) -> Optional[ApiKeyResponse]:
        """Update API key."""
        with self.db_manager.get_session() as session:
            api_key = session.query(ApiKey).filter(
                and_(
                    ApiKey.id == api_key_id,
                    ApiKey.user_id == user_id
                )
            ).first()
            
            if not api_key:
                return None
            
            # Update fields
            if key_data.name is not None:
                # Check for name conflicts
                existing = session.query(ApiKey).filter(
                    and_(
                        ApiKey.user_id == user_id,
                        ApiKey.name == key_data.name,
                        ApiKey.id != api_key_id
                    )
                ).first()
                
                if existing:
                    raise ConflictError(f"API key name '{key_data.name}' already exists")
                
                api_key.name = key_data.name
            
            # Update metadata
            if not api_key.metadata:
                api_key.metadata = {}
            
            if key_data.description is not None:
                api_key.metadata['description'] = key_data.description
            
            if key_data.permissions is not None:
                api_key.metadata['permissions'] = [p.value for p in key_data.permissions]
            
            if key_data.expires_at is not None:
                api_key.metadata['expires_at'] = key_data.expires_at.isoformat()
            
            if key_data.rate_limit_per_hour is not None:
                api_key.metadata['rate_limit_per_hour'] = key_data.rate_limit_per_hour
            
            if key_data.allowed_ips is not None:
                api_key.metadata['allowed_ips'] = key_data.allowed_ips
            
            if key_data.status is not None:
                api_key.metadata['status'] = key_data.status.value
            
            if key_data.metadata is not None:
                api_key.metadata.update(key_data.metadata)
            
            logger.info("API key updated", api_key_id=api_key_id, user_id=user_id)
            return self._api_key_to_response(api_key)
    
    def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return key information."""
        if not self.generator.validate_key_format(api_key):
            return None
        
        key_hash = self.generator.hash_api_key(api_key)
        
        with self.db_manager.get_session() as session:
            api_key_record = session.query(ApiKey).filter(
                ApiKey.key == key_hash
            ).first()
            
            if not api_key_record:
                return None
            
            # Check if key is active
            metadata = api_key_record.metadata or {}
            status = metadata.get('status', ApiKeyStatus.ACTIVE.value)
            
            if status != ApiKeyStatus.ACTIVE.value:
                logger.warning("Inactive API key used", 
                             api_key_id=api_key_record.id,
                             status=status)
                return None
            
            # Check expiration
            expires_at = metadata.get('expires_at')
            if expires_at:
                expiry_date = datetime.fromisoformat(expires_at)
                if datetime.utcnow() > expiry_date:
                    logger.warning("Expired API key used", 
                                 api_key_id=api_key_record.id,
                                 expires_at=expires_at)
                    return None
            
            # Update last used
            api_key_record.last_used = datetime.utcnow()
            metadata['usage_count'] = metadata.get('usage_count', 0) + 1
            api_key_record.metadata = metadata
            
            return {
                'api_key_id': api_key_record.id,
                'user_id': api_key_record.user_id,
                'name': api_key_record.name,
                'permissions': metadata.get('permissions', [ApiKeyPermission.READ.value]),
                'rate_limit_per_hour': metadata.get('rate_limit_per_hour', 1000),
                'allowed_ips': metadata.get('allowed_ips'),
                'metadata': metadata
            }
    
    def revoke_api_key(self, api_key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        with self.db_manager.get_session() as session:
            api_key = session.query(ApiKey).filter(
                and_(
                    ApiKey.id == api_key_id,
                    ApiKey.user_id == user_id
                )
            ).first()
            
            if not api_key:
                return False
            
            if not api_key.metadata:
                api_key.metadata = {}
            
            api_key.metadata['status'] = ApiKeyStatus.REVOKED.value
            api_key.metadata['revoked_at'] = datetime.utcnow().isoformat()
            
            logger.info("API key revoked", api_key_id=api_key_id, user_id=user_id)
            return True
    
    def _api_key_to_response(self, api_key: ApiKey) -> ApiKeyResponse:
        """Convert ApiKey model to ApiKeyResponse."""
        metadata = api_key.metadata or {}
        
        permissions = [ApiKeyPermission(p) for p in metadata.get('permissions', [ApiKeyPermission.READ.value])]
        status = ApiKeyStatus(metadata.get('status', ApiKeyStatus.ACTIVE.value))
        
        expires_at = None
        if metadata.get('expires_at'):
            expires_at = datetime.fromisoformat(metadata['expires_at'])
        
        return ApiKeyResponse(
            id=api_key.id,
            name=api_key.name,
            description=metadata.get('description'),
            key_prefix=metadata.get('key_prefix', ''),
            permissions=permissions,
            status=status,
            expires_at=expires_at,
            rate_limit_per_hour=metadata.get('rate_limit_per_hour', 1000),
            allowed_ips=metadata.get('allowed_ips'),
            created_at=api_key.created_at,
            last_used=api_key.last_used,
            usage_count=metadata.get('usage_count', 0),
            user_id=api_key.user_id,
            metadata={k: v for k, v in metadata.items() 
                     if k not in ['permissions', 'status', 'expires_at', 'rate_limit_per_hour', 
                                 'allowed_ips', 'key_prefix', 'usage_count', 'description']}
        )
```

### Step 2: Create API Key Authentication Middleware

**File to Create**: `packages/morag-core/src/morag_core/api_keys/middleware.py`

```python
"""API key authentication middleware."""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import structlog

from .service import ApiKeyService
from .models import ApiKeyPermission

logger = structlog.get_logger(__name__)

class ApiKeyAuthenticationMiddleware:
    """API key authentication middleware."""
    
    def __init__(self):
        self.api_key_service = ApiKeyService()
        self.security = HTTPBearer(auto_error=False)
    
    async def authenticate_api_key(self, request: Request) -> Optional[Dict[str, Any]]:
        """Authenticate request using API key."""
        # Check for API key in header
        api_key = request.headers.get("X-API-Key")
        
        if not api_key:
            # Check Authorization header for API key
            credentials: HTTPAuthorizationCredentials = await self.security(request)
            if credentials and credentials.scheme.lower() == "bearer":
                # Could be API key instead of JWT
                if credentials.credentials.startswith("morag_"):
                    api_key = credentials.credentials
        
        if not api_key:
            return None
        
        # Validate API key
        key_info = self.api_key_service.validate_api_key(api_key)
        if not key_info:
            return None
        
        # Check IP restrictions
        allowed_ips = key_info.get('allowed_ips')
        if allowed_ips:
            client_ip = request.client.host
            if client_ip not in allowed_ips:
                logger.warning("API key used from unauthorized IP", 
                             api_key_id=key_info['api_key_id'],
                             client_ip=client_ip,
                             allowed_ips=allowed_ips)
                return None
        
        # TODO: Implement rate limiting check here
        
        logger.info("API key authenticated", 
                   api_key_id=key_info['api_key_id'],
                   user_id=key_info['user_id'])
        
        return key_info
    
    def check_permission(self, key_info: Dict[str, Any], required_permission: ApiKeyPermission) -> bool:
        """Check if API key has required permission."""
        permissions = key_info.get('permissions', [])
        
        # Admin permission grants all access
        if ApiKeyPermission.ADMIN.value in permissions:
            return True
        
        # Check specific permission
        if required_permission.value in permissions:
            return True
        
        # Check permission hierarchy
        if required_permission == ApiKeyPermission.READ:
            return ApiKeyPermission.WRITE.value in permissions
        
        return False
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_api_key_management.py
import pytest
from morag_core.api_keys import ApiKeyService, ApiKeyGenerator
from morag_core.api_keys.models import ApiKeyCreate, ApiKeyPermission

def test_api_key_generation():
    """Test API key generation."""
    generator = ApiKeyGenerator()
    api_key, key_hash = generator.generate_api_key()
    
    assert api_key.startswith("morag_")
    assert len(api_key) > 20
    assert generator.validate_key_format(api_key)
    assert generator.hash_api_key(api_key) == key_hash

def test_api_key_creation():
    """Test API key creation."""
    service = ApiKeyService()
    key_data = ApiKeyCreate(
        name="Test API Key",
        description="Test key for unit tests",
        permissions=[ApiKeyPermission.READ, ApiKeyPermission.WRITE]
    )
    
    result = service.create_api_key("user123", key_data)
    assert result.api_key.name == "Test API Key"
    assert ApiKeyPermission.READ in result.api_key.permissions
    assert result.secret_key.startswith("morag_")

def test_api_key_validation():
    """Test API key validation."""
    service = ApiKeyService()
    
    # Create API key
    key_data = ApiKeyCreate(name="Validation Test", permissions=[ApiKeyPermission.READ])
    result = service.create_api_key("user123", key_data)
    
    # Validate key
    key_info = service.validate_api_key(result.secret_key)
    assert key_info is not None
    assert key_info['user_id'] == "user123"
    assert ApiKeyPermission.READ.value in key_info['permissions']
    
    # Test invalid key
    invalid_info = service.validate_api_key("invalid_key")
    assert invalid_info is None
```

## 📋 Acceptance Criteria

- [ ] API key generation and management implemented
- [ ] API key authentication middleware working
- [ ] Permission-based access control functional
- [ ] Usage tracking and rate limiting implemented
- [ ] API key expiration and rotation working
- [ ] IP-based access restrictions functional
- [ ] API endpoints for key management created
- [ ] Integration with existing authentication system
- [ ] Comprehensive unit tests passing
- [ ] Security measures for key storage implemented

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 08: Multi-tenancy Implementation](./task-08-multi-tenancy-implementation.md)
2. Add API key usage analytics dashboard
3. Implement advanced rate limiting
4. Test API key authentication with various scenarios

## 📝 Notes

- Store only hashed versions of API keys in database
- Implement proper rate limiting to prevent abuse
- Add comprehensive logging for security auditing
- Consider implementing API key rotation policies
- Add monitoring and alerting for suspicious API key usage
