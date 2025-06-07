# Task 9: Authentication & Security

## Objective
Implement comprehensive authentication and security measures for remote worker connections, including JWT-based authentication, API key management, secure file transfers, and audit logging.

## Current State Analysis

### Existing Security
- Basic auth token placeholders in worker communication
- File encryption in transfer service (Task 3)
- No comprehensive authentication system
- No audit logging or security monitoring

### Security Requirements
- JWT-based worker authentication
- API key management for server access
- Secure WebSocket connections (WSS)
- File transfer encryption and integrity checks
- Audit logging for security events
- Rate limiting and DDoS protection

## Implementation Plan

### Step 1: Authentication Models and Services

#### 1.1 Create Authentication Models
**File**: `packages/morag-core/src/morag_core/models/auth.py`

```python
"""Authentication and security models."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uuid

class AuthRole(str, Enum):
    ADMIN = "admin"
    WORKER = "worker"
    CLIENT = "client"
    READONLY = "readonly"

class AuthScope(str, Enum):
    WORKER_REGISTER = "worker:register"
    WORKER_TASKS = "worker:tasks"
    FILE_TRANSFER = "file:transfer"
    API_ACCESS = "api:access"
    ADMIN_ACCESS = "admin:access"

class APIKey(BaseModel):
    """API key for authentication."""
    key_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    key_hash: str  # Hashed version of the actual key
    role: AuthRole
    scopes: List[AuthScope]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

class WorkerCredentials(BaseModel):
    """Credentials for worker authentication."""
    worker_id: str
    api_key: str
    certificate_path: Optional[str] = None
    private_key_path: Optional[str] = None

class AuthToken(BaseModel):
    """JWT authentication token."""
    token: str
    token_type: str = "Bearer"
    expires_in: int  # seconds
    scope: List[AuthScope]
    issued_at: datetime = Field(default_factory=datetime.utcnow)

class SecurityEvent(BaseModel):
    """Security-related event for audit logging."""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str  # "auth_success", "auth_failure", "token_expired", etc.
    user_id: Optional[str] = None
    worker_id: Optional[str] = None
    ip_address: str
    user_agent: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    details: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical

class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10
    block_duration_minutes: int = 15
```

### Step 2: Authentication Service

#### 2.1 Create Authentication Service
**File**: `packages/morag/src/morag/services/auth_service.py`

```python
"""Authentication service for MoRAG system."""

import hashlib
import secrets
import jwt
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import structlog
from redis import Redis
import bcrypt

from morag_core.models.auth import (
    APIKey, AuthRole, AuthScope, AuthToken, WorkerCredentials,
    SecurityEvent, RateLimitConfig
)

logger = structlog.get_logger(__name__)

class AuthenticationService:
    """Handles authentication and authorization for the MoRAG system."""
    
    def __init__(self, redis_client: Redis, jwt_secret: str):
        self.redis = redis_client
        self.jwt_secret = jwt_secret
        self.api_keys: Dict[str, APIKey] = {}
        self.rate_limits = RateLimitConfig()
        
    async def start(self):
        """Start the authentication service."""
        logger.info("Starting authentication service")
        await self._load_api_keys()
        logger.info("Authentication service started")
    
    async def stop(self):
        """Stop the authentication service."""
        logger.info("Stopping authentication service")
        await self._save_api_keys()
        logger.info("Authentication service stopped")
    
    async def create_api_key(self, name: str, role: AuthRole, 
                           scopes: List[AuthScope], expires_days: Optional[int] = None) -> Tuple[str, APIKey]:
        """Create a new API key."""
        try:
            # Generate secure random key
            raw_key = secrets.token_urlsafe(32)
            key_hash = self._hash_key(raw_key)
            
            # Create API key record
            api_key = APIKey(
                name=name,
                key_hash=key_hash,
                role=role,
                scopes=scopes,
                expires_at=datetime.utcnow() + timedelta(days=expires_days) if expires_days else None
            )
            
            # Store API key
            self.api_keys[api_key.key_id] = api_key
            await self._save_api_key(api_key)
            
            logger.info("API key created",
                       key_id=api_key.key_id,
                       name=name,
                       role=role.value)
            
            return raw_key, api_key
            
        except Exception as e:
            logger.error("Failed to create API key", name=name, error=str(e))
            raise
    
    async def validate_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Validate an API key and return the associated record."""
        try:
            key_hash = self._hash_key(raw_key)
            
            # Find matching API key
            for api_key in self.api_keys.values():
                if api_key.key_hash == key_hash and api_key.is_active:
                    # Check expiration
                    if api_key.expires_at and datetime.utcnow() > api_key.expires_at:
                        logger.warning("API key expired", key_id=api_key.key_id)
                        return None
                    
                    # Update last used
                    api_key.last_used = datetime.utcnow()
                    await self._save_api_key(api_key)
                    
                    return api_key
            
            return None
            
        except Exception as e:
            logger.error("API key validation failed", error=str(e))
            return None
    
    async def create_jwt_token(self, api_key: APIKey, worker_id: Optional[str] = None,
                             expires_hours: int = 24) -> AuthToken:
        """Create a JWT token for authenticated access."""
        try:
            payload = {
                'key_id': api_key.key_id,
                'role': api_key.role.value,
                'scopes': [scope.value for scope in api_key.scopes],
                'iat': datetime.utcnow(),
                'exp': datetime.utcnow() + timedelta(hours=expires_hours)
            }
            
            if worker_id:
                payload['worker_id'] = worker_id
            
            token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
            
            auth_token = AuthToken(
                token=token,
                expires_in=expires_hours * 3600,
                scope=api_key.scopes
            )
            
            logger.info("JWT token created",
                       key_id=api_key.key_id,
                       worker_id=worker_id,
                       expires_hours=expires_hours)
            
            return auth_token
            
        except Exception as e:
            logger.error("JWT token creation failed", error=str(e))
            raise
    
    async def validate_jwt_token(self, token: str) -> Optional[Dict]:
        """Validate a JWT token and return the payload."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            
            # Check if API key is still valid
            key_id = payload.get('key_id')
            if key_id and key_id in self.api_keys:
                api_key = self.api_keys[key_id]
                if not api_key.is_active:
                    logger.warning("JWT token for inactive API key", key_id=key_id)
                    return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None
        except Exception as e:
            logger.error("JWT token validation failed", error=str(e))
            return None
    
    async def check_rate_limit(self, identifier: str, ip_address: str) -> bool:
        """Check if request is within rate limits."""
        try:
            current_time = datetime.utcnow()
            minute_key = f"rate_limit:{identifier}:{current_time.strftime('%Y%m%d%H%M')}"
            hour_key = f"rate_limit:{identifier}:{current_time.strftime('%Y%m%d%H')}"
            
            # Check minute limit
            minute_count = self.redis.get(minute_key)
            if minute_count and int(minute_count) >= self.rate_limits.requests_per_minute:
                await self._log_security_event("rate_limit_exceeded", 
                                              details={'limit': 'minute', 'ip': ip_address})
                return False
            
            # Check hour limit
            hour_count = self.redis.get(hour_key)
            if hour_count and int(hour_count) >= self.rate_limits.requests_per_hour:
                await self._log_security_event("rate_limit_exceeded",
                                              details={'limit': 'hour', 'ip': ip_address})
                return False
            
            # Increment counters
            pipe = self.redis.pipeline()
            pipe.incr(minute_key)
            pipe.expire(minute_key, 60)
            pipe.incr(hour_key)
            pipe.expire(hour_key, 3600)
            pipe.execute()
            
            return True
            
        except Exception as e:
            logger.error("Rate limit check failed", error=str(e))
            return True  # Allow on error
    
    async def revoke_api_key(self, key_id: str) -> bool:
        """Revoke an API key."""
        try:
            if key_id in self.api_keys:
                self.api_keys[key_id].is_active = False
                await self._save_api_key(self.api_keys[key_id])
                
                logger.info("API key revoked", key_id=key_id)
                return True
            
            return False
            
        except Exception as e:
            logger.error("Failed to revoke API key", key_id=key_id, error=str(e))
            return False
    
    async def log_security_event(self, event_type: str, user_id: Optional[str] = None,
                                worker_id: Optional[str] = None, ip_address: str = "",
                                details: Dict = None, severity: str = "info"):
        """Log a security event."""
        await self._log_security_event(event_type, user_id, worker_id, ip_address, details, severity)
    
    def _hash_key(self, raw_key: str) -> str:
        """Hash an API key for secure storage."""
        return hashlib.sha256(raw_key.encode()).hexdigest()
    
    async def _log_security_event(self, event_type: str, user_id: Optional[str] = None,
                                 worker_id: Optional[str] = None, ip_address: str = "",
                                 details: Dict = None, severity: str = "info"):
        """Internal method to log security events."""
        try:
            event = SecurityEvent(
                event_type=event_type,
                user_id=user_id,
                worker_id=worker_id,
                ip_address=ip_address,
                details=details or {},
                severity=severity
            )
            
            # Store in Redis with TTL
            key = f"security_event:{event.event_id}"
            self.redis.setex(key, 86400 * 30, event.model_dump_json())  # 30 days
            
            # Log to structured logger
            logger.info("Security event logged",
                       event_type=event_type,
                       severity=severity,
                       user_id=user_id,
                       worker_id=worker_id)
            
        except Exception as e:
            logger.error("Failed to log security event", error=str(e))
    
    async def _load_api_keys(self):
        """Load API keys from Redis."""
        try:
            pattern = "api_key:*"
            for key in self.redis.scan_iter(match=pattern):
                data = self.redis.get(key)
                if data:
                    api_key = APIKey.model_validate_json(data)
                    self.api_keys[api_key.key_id] = api_key
            
            logger.info("API keys loaded", count=len(self.api_keys))
            
        except Exception as e:
            logger.error("Failed to load API keys", error=str(e))
    
    async def _save_api_keys(self):
        """Save all API keys to Redis."""
        try:
            for api_key in self.api_keys.values():
                await self._save_api_key(api_key)
            
            logger.info("API keys saved", count=len(self.api_keys))
            
        except Exception as e:
            logger.error("Failed to save API keys", error=str(e))
    
    async def _save_api_key(self, api_key: APIKey):
        """Save a single API key to Redis."""
        try:
            key = f"api_key:{api_key.key_id}"
            self.redis.set(key, api_key.model_dump_json())
            
        except Exception as e:
            logger.error("Failed to save API key", key_id=api_key.key_id, error=str(e))
```

### Step 3: Security Middleware

#### 3.1 Create Security Middleware
**File**: `packages/morag/src/morag/middleware/security.py`

```python
"""Security middleware for FastAPI."""

import time
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from morag.services.auth_service import AuthenticationService
from morag_core.models.auth import AuthScope

logger = structlog.get_logger(__name__)

class SecurityMiddleware:
    """Security middleware for request authentication and authorization."""
    
    def __init__(self, auth_service: AuthenticationService):
        self.auth_service = auth_service
        self.bearer_scheme = HTTPBearer(auto_error=False)
    
    async def authenticate_request(self, request: Request) -> Optional[dict]:
        """Authenticate a request and return user context."""
        try:
            # Get client IP
            client_ip = self._get_client_ip(request)
            
            # Check rate limits
            identifier = client_ip  # Could be enhanced with user ID
            if not await self.auth_service.check_rate_limit(identifier, client_ip):
                await self.auth_service.log_security_event(
                    "rate_limit_exceeded",
                    ip_address=client_ip,
                    severity="warning"
                )
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded"
                )
            
            # Extract authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return None
            
            # Validate token format
            if not auth_header.startswith("Bearer "):
                await self.auth_service.log_security_event(
                    "invalid_auth_format",
                    ip_address=client_ip,
                    severity="warning"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authorization format"
                )
            
            token = auth_header[7:]  # Remove "Bearer " prefix
            
            # Validate JWT token
            payload = await self.auth_service.validate_jwt_token(token)
            if not payload:
                await self.auth_service.log_security_event(
                    "invalid_token",
                    ip_address=client_ip,
                    severity="warning"
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired token"
                )
            
            # Log successful authentication
            await self.auth_service.log_security_event(
                "auth_success",
                user_id=payload.get('key_id'),
                worker_id=payload.get('worker_id'),
                ip_address=client_ip
            )
            
            return payload
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Authentication failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication error"
            )
    
    def require_scope(self, required_scope: AuthScope):
        """Decorator to require specific scope for endpoint access."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                # Get request from kwargs (FastAPI dependency injection)
                request = kwargs.get('request')
                if not request:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Request context not available"
                    )
                
                # Authenticate request
                user_context = await self.authenticate_request(request)
                if not user_context:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Authentication required"
                    )
                
                # Check scope
                user_scopes = user_context.get('scopes', [])
                if required_scope.value not in user_scopes:
                    await self.auth_service.log_security_event(
                        "insufficient_scope",
                        user_id=user_context.get('key_id'),
                        ip_address=self._get_client_ip(request),
                        details={'required_scope': required_scope.value, 'user_scopes': user_scopes},
                        severity="warning"
                    )
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Required scope: {required_scope.value}"
                    )
                
                # Add user context to kwargs
                kwargs['user_context'] = user_context
                
                return await func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers (proxy/load balancer)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"

# FastAPI dependency for authentication
async def get_current_user(request: Request, 
                          security_middleware: SecurityMiddleware) -> dict:
    """FastAPI dependency to get current authenticated user."""
    return await security_middleware.authenticate_request(request)

# FastAPI dependency for specific scopes
def require_scope(scope: AuthScope):
    """FastAPI dependency factory for scope requirements."""
    async def scope_dependency(user_context: dict = Depends(get_current_user)) -> dict:
        user_scopes = user_context.get('scopes', [])
        if scope.value not in user_scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required scope: {scope.value}"
            )
        return user_context
    
    return scope_dependency
```

### Step 4: Secure API Endpoints

#### 4.1 Update Server with Security
**File**: `packages/morag/src/morag/server.py` (additions)

```python
# Add to existing imports
from morag.services.auth_service import AuthenticationService
from morag.middleware.security import SecurityMiddleware, require_scope
from morag_core.models.auth import AuthScope, AuthRole

# Initialize authentication service
auth_service = AuthenticationService(redis_client, os.getenv('JWT_SECRET', 'change-this-secret'))
security_middleware = SecurityMiddleware(auth_service)

# Add authentication endpoints
@app.post("/api/v1/auth/api-key")
async def create_api_key(request: Request, name: str, role: AuthRole, 
                        scopes: List[AuthScope], expires_days: Optional[int] = None):
    """Create a new API key (admin only)."""
    # This would require admin authentication
    raw_key, api_key = await auth_service.create_api_key(name, role, scopes, expires_days)
    
    return {
        "api_key": raw_key,
        "key_id": api_key.key_id,
        "expires_at": api_key.expires_at
    }

@app.post("/api/v1/auth/token")
async def create_token(api_key: str, worker_id: Optional[str] = None):
    """Create JWT token from API key."""
    api_key_record = await auth_service.validate_api_key(api_key)
    if not api_key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    token = await auth_service.create_jwt_token(api_key_record, worker_id)
    return token

# Secure existing endpoints
@app.post("/api/v1/workers/register")
@require_scope(AuthScope.WORKER_REGISTER)
async def register_worker(request: Request, worker_request: WorkerRegistrationRequest,
                         user_context: dict = Depends(get_current_user)):
    """Register a new worker (secured)."""
    # Implementation with security context
    pass

@app.post("/api/v1/transfers/{transfer_id}/upload")
@require_scope(AuthScope.FILE_TRANSFER)
async def upload_file(transfer_id: str, file: UploadFile,
                     user_context: dict = Depends(get_current_user)):
    """Upload file (secured)."""
    # Implementation with security context
    pass
```

## Testing Requirements

### Unit Tests
1. **Authentication Service Tests**
   - Test API key creation and validation
   - Test JWT token creation and validation
   - Test rate limiting
   - Test security event logging

2. **Security Middleware Tests**
   - Test request authentication
   - Test scope-based authorization
   - Test rate limiting middleware
   - Test security headers

### Integration Tests
1. **End-to-End Security Tests**
   - Test complete authentication flow
   - Test worker authentication
   - Test file transfer security
   - Test rate limiting behavior

### Test Files to Create
- `tests/test_auth_service.py`
- `tests/test_security_middleware.py`
- `tests/integration/test_security_e2e.py`

## Dependencies
- **New**: `PyJWT` for JWT token handling
- **New**: `bcrypt` for password hashing
- **Existing**: Redis for session storage

## Success Criteria
1. All API endpoints are properly authenticated
2. Worker connections use secure JWT tokens
3. File transfers are encrypted and authenticated
4. Rate limiting prevents abuse
5. Security events are logged and monitored
6. System handles authentication failures gracefully

## Next Steps
After completing this task:
1. Proceed to Task 10: Configuration Management
2. Test security with penetration testing
3. Validate authentication and authorization flows

---

**Dependencies**: Task 8 (Remote Worker Package)
**Estimated Time**: 4-5 days
**Risk Level**: High (security is critical)
