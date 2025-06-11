"""API Key models for MoRAG core."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class ApiKeyPermission(str, Enum):
    """API key permissions."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    PROCESS = "process"
    INGEST = "ingest"
    SEARCH = "search"


class ApiKeyStatus(str, Enum):
    """API key status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    REVOKED = "revoked"
    EXPIRED = "expired"


class ApiKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    permissions: List[ApiKeyPermission] = Field(default=[ApiKeyPermission.READ])
    expires_at: Optional[datetime] = None
    rate_limit_per_hour: int = Field(default=1000, ge=1, le=10000)
    allowed_ips: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ApiKeyUpdate(BaseModel):
    """API key update request."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    permissions: Optional[List[ApiKeyPermission]] = None
    status: Optional[ApiKeyStatus] = None
    expires_at: Optional[datetime] = None
    rate_limit_per_hour: Optional[int] = Field(None, ge=1, le=10000)
    allowed_ips: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ApiKeyResponse(BaseModel):
    """API key response."""
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
    """API key creation response with full key."""
    api_key: ApiKeyResponse
    secret_key: str  # Full key returned only once


class ApiKeyUsage(BaseModel):
    """API key usage record."""
    api_key_id: str
    endpoint: str
    method: str
    timestamp: datetime
    ip_address: str
    user_agent: Optional[str]
    response_status: int
    response_time_ms: float
    request_size_bytes: Optional[int]
    response_size_bytes: Optional[int]


class ApiKeyUsageStats(BaseModel):
    """API key usage statistics."""
    api_key_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    total_data_transferred_bytes: int
    last_24h_requests: int
    last_7d_requests: int
    last_30d_requests: int
    most_used_endpoints: List[Dict[str, Any]]


class ApiKeySearchRequest(BaseModel):
    """API key search request."""
    user_id: Optional[str] = None
    status: Optional[ApiKeyStatus] = None
    permissions: Optional[List[ApiKeyPermission]] = None
    name_contains: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    last_used_after: Optional[datetime] = None
    last_used_before: Optional[datetime] = None
    skip: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=100)


class ApiKeySearchResponse(BaseModel):
    """API key search response."""
    api_keys: List[ApiKeyResponse]
    total: int
    skip: int
    limit: int
