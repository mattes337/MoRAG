# Task 02: User Management System

## 📋 Task Overview

**Objective**: Implement comprehensive user management system including user registration, authentication, role-based access control, and user settings management.

**Priority**: Critical - Required for multi-user support
**Estimated Time**: 1-2 weeks
**Dependencies**: Task 01 (Database Configuration and Setup)

## 🎯 Goals

1. Implement user registration and authentication
2. Create JWT-based session management
3. Implement role-based access control (RBAC)
4. Add user settings and preferences management
5. Create user management API endpoints
6. Integrate authentication middleware with FastAPI
7. Add password security and validation

## 📊 Current State Analysis

### Existing User Models (from database/DATABASE.md)
- **User**: ID, name, email, avatar, role, timestamps
- **UserSettings**: Theme, language, notifications, auto_save, default_database
- **UserRole**: ADMIN, USER, VIEWER enums
- **Theme**: LIGHT, DARK, SYSTEM enums
- **Complete Schema**: See `database/DATABASE.md` for full entity definitions and relationships

### Current MoRAG Authentication
- **Status**: No authentication system
- **API**: Completely open, no user context
- **Sessions**: Stateless API without user tracking

## 🔧 Implementation Plan

### Step 1: Create User Service Layer

**Files to Create**:
```
packages/morag-core/src/morag_core/
├── auth/
│   ├── __init__.py
│   ├── models.py          # Pydantic models for auth
│   ├── service.py         # User service logic
│   ├── security.py        # Password hashing, JWT
│   ├── middleware.py      # FastAPI middleware
│   └── dependencies.py    # FastAPI dependencies
```

**Implementation Details**:

1. **Authentication Models**:
```python
# packages/morag-core/src/morag_core/auth/models.py
"""Authentication and user management models."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "ADMIN"
    USER = "USER"
    VIEWER = "VIEWER"

class Theme(str, Enum):
    LIGHT = "LIGHT"
    DARK = "DARK"
    SYSTEM = "SYSTEM"

class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    avatar: Optional[str] = None

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    name: str
    email: str
    avatar: Optional[str]
    role: UserRole
    created_at: datetime
    updated_at: datetime

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=255)
    email: Optional[EmailStr] = None
    avatar: Optional[str] = None

class UserSettingsUpdate(BaseModel):
    theme: Optional[Theme] = None
    language: Optional[str] = Field(None, max_length=10)
    notifications: Optional[bool] = None
    auto_save: Optional[bool] = None
    default_database: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
```

2. **Security Utilities**:
```python
# packages/morag-core/src/morag_core/auth/security.py
"""Security utilities for authentication."""

import bcrypt
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import structlog

from morag_core.config import get_settings

logger = structlog.get_logger(__name__)

class PasswordManager:
    """Password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify a password against its hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

class JWTManager:
    """JWT token management."""
    
    def __init__(self):
        self.settings = get_settings()
        self.secret_key = self.settings.jwt_secret_key
        self.algorithm = self.settings.jwt_algorithm
        self.expiration_hours = self.settings.jwt_expiration_hours
    
    def create_access_token(self, user_id: str, email: str, role: str) -> Dict[str, Any]:
        """Create a JWT access token."""
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.expiration_hours)
        
        payload = {
            "sub": user_id,
            "email": email,
            "role": role,
            "iat": now,
            "exp": expires_at
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": self.expiration_hours * 3600,
            "expires_at": expires_at.isoformat()
        }
    
    def decode_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            return None
    
    def refresh_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Refresh a JWT token if it's still valid."""
        payload = self.decode_token(token)
        if payload:
            return self.create_access_token(
                payload["sub"], 
                payload["email"], 
                payload["role"]
            )
        return None
```

3. **User Service**:
```python
# packages/morag-core/src/morag_core/auth/service.py
"""User management service."""

from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import structlog

from morag_core.database import User, UserSettings, get_database_manager
from morag_core.database.models import UserRole as DBUserRole, Theme as DBTheme
from .models import UserCreate, UserUpdate, UserSettingsUpdate, UserResponse
from .security import PasswordManager, JWTManager
from morag_core.exceptions import (
    AuthenticationError, ValidationError, NotFoundError, ConflictError
)

logger = structlog.get_logger(__name__)

class UserService:
    """User management service."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.password_manager = PasswordManager()
        self.jwt_manager = JWTManager()
    
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user."""
        with self.db_manager.get_session() as session:
            # Check if user already exists
            existing_user = session.query(User).filter_by(email=user_data.email).first()
            if existing_user:
                raise ConflictError(f"User with email {user_data.email} already exists")
            
            # Hash password
            hashed_password = self.password_manager.hash_password(user_data.password)
            
            # Create user
            user = User(
                name=user_data.name,
                email=user_data.email,
                password_hash=hashed_password,  # Add this field to User model
                avatar=user_data.avatar,
                role=DBUserRole.USER
            )
            
            session.add(user)
            session.flush()  # Get user ID
            
            # Create default user settings
            user_settings = UserSettings(
                user_id=user.id,
                theme=DBTheme.LIGHT,
                language="en",
                notifications=True,
                auto_save=True
            )
            session.add(user_settings)
            
            logger.info("User created", user_id=user.id, email=user.email)
            return self._user_to_response(user)
    
    def authenticate_user(self, email: str, password: str) -> Optional[UserResponse]:
        """Authenticate a user by email and password."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(email=email).first()
            if not user:
                return None
            
            if not self.password_manager.verify_password(password, user.password_hash):
                return None
            
            logger.info("User authenticated", user_id=user.id, email=email)
            return self._user_to_response(user)
    
    def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if user:
                return self._user_to_response(user)
            return None
    
    def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get user by email."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(email=email).first()
            if user:
                return self._user_to_response(user)
            return None
    
    def update_user(self, user_id: str, user_data: UserUpdate) -> UserResponse:
        """Update user information."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                raise NotFoundError(f"User {user_id} not found")
            
            # Update fields
            if user_data.name is not None:
                user.name = user_data.name
            if user_data.email is not None:
                # Check email uniqueness
                existing = session.query(User).filter_by(email=user_data.email).first()
                if existing and existing.id != user_id:
                    raise ConflictError(f"Email {user_data.email} already in use")
                user.email = user_data.email
            if user_data.avatar is not None:
                user.avatar = user_data.avatar
            
            logger.info("User updated", user_id=user_id)
            return self._user_to_response(user)
    
    def update_user_settings(self, user_id: str, settings_data: UserSettingsUpdate) -> dict:
        """Update user settings."""
        with self.db_manager.get_session() as session:
            settings = session.query(UserSettings).filter_by(user_id=user_id).first()
            if not settings:
                raise NotFoundError(f"User settings for {user_id} not found")
            
            # Update settings
            if settings_data.theme is not None:
                settings.theme = DBTheme(settings_data.theme.value)
            if settings_data.language is not None:
                settings.language = settings_data.language
            if settings_data.notifications is not None:
                settings.notifications = settings_data.notifications
            if settings_data.auto_save is not None:
                settings.auto_save = settings_data.auto_save
            if settings_data.default_database is not None:
                settings.default_database = settings_data.default_database
            
            logger.info("User settings updated", user_id=user_id)
            return self._settings_to_dict(settings)
    
    def delete_user(self, user_id: str) -> bool:
        """Delete a user (soft delete by setting inactive)."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter_by(id=user_id).first()
            if not user:
                return False
            
            # Soft delete - add is_active field to User model
            user.is_active = False
            logger.info("User deleted", user_id=user_id)
            return True
    
    def _user_to_response(self, user: User) -> UserResponse:
        """Convert User model to UserResponse."""
        return UserResponse(
            id=user.id,
            name=user.name,
            email=user.email,
            avatar=user.avatar,
            role=user.role.value,
            created_at=user.created_at,
            updated_at=user.updated_at
        )
    
    def _settings_to_dict(self, settings: UserSettings) -> dict:
        """Convert UserSettings to dict."""
        return {
            "theme": settings.theme.value,
            "language": settings.language,
            "notifications": settings.notifications,
            "auto_save": settings.auto_save,
            "default_database": settings.default_database
        }
```

### Step 2: Create Authentication Middleware

**File to Create**: `packages/morag-core/src/morag_core/auth/middleware.py`

```python
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
        credentials: HTTPAuthorizationCredentials = await self.security(request)
        
        if not credentials:
            return None
        
        token_data = self.jwt_manager.decode_token(credentials.credentials)
        if not token_data:
            return None
        
        user = self.user_service.get_user_by_id(token_data["sub"])
        return user
    
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
                detail=f"Role {required_role} required",
            )
        
        return user
```

### Step 3: Create FastAPI Dependencies

**File to Create**: `packages/morag-core/src/morag_core/auth/dependencies.py`

```python
"""FastAPI dependencies for authentication."""

from fastapi import Depends, Request
from typing import Optional

from .middleware import AuthenticationMiddleware
from .models import UserResponse

# Global middleware instance
auth_middleware = AuthenticationMiddleware()

async def get_current_user(request: Request) -> Optional[UserResponse]:
    """FastAPI dependency to get current user."""
    return await auth_middleware.get_current_user(request)

async def require_authentication(request: Request) -> UserResponse:
    """FastAPI dependency to require authentication."""
    return await auth_middleware.require_authentication(request)

async def require_admin(request: Request) -> UserResponse:
    """FastAPI dependency to require admin role."""
    return await auth_middleware.require_role(request, "ADMIN")

async def require_user(request: Request) -> UserResponse:
    """FastAPI dependency to require user role or higher."""
    return await auth_middleware.require_role(request, "USER")

# Convenience dependencies
CurrentUser = Depends(get_current_user)
RequireAuth = Depends(require_authentication)
RequireAdmin = Depends(require_admin)
RequireUser = Depends(require_user)
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_user_management.py
import pytest
from morag_core.auth import UserService, PasswordManager, JWTManager
from morag_core.auth.models import UserCreate, UserLogin

def test_password_hashing():
    """Test password hashing and verification."""
    pm = PasswordManager()
    password = "test_password_123"
    hashed = pm.hash_password(password)
    
    assert pm.verify_password(password, hashed)
    assert not pm.verify_password("wrong_password", hashed)

def test_jwt_token_creation():
    """Test JWT token creation and validation."""
    jwt_manager = JWTManager()
    token_data = jwt_manager.create_access_token("user123", "test@example.com", "USER")
    
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    
    # Decode token
    payload = jwt_manager.decode_token(token_data["access_token"])
    assert payload["sub"] == "user123"
    assert payload["email"] == "test@example.com"
    assert payload["role"] == "USER"

def test_user_creation():
    """Test user creation."""
    user_service = UserService()
    user_data = UserCreate(
        name="Test User",
        email="test@example.com",
        password="secure_password_123"
    )
    
    user = user_service.create_user(user_data)
    assert user.name == "Test User"
    assert user.email == "test@example.com"
    assert user.role == "USER"

def test_user_authentication():
    """Test user authentication."""
    user_service = UserService()
    
    # Create user
    user_data = UserCreate(
        name="Test User",
        email="auth@example.com",
        password="secure_password_123"
    )
    created_user = user_service.create_user(user_data)
    
    # Authenticate
    authenticated_user = user_service.authenticate_user("auth@example.com", "secure_password_123")
    assert authenticated_user is not None
    assert authenticated_user.id == created_user.id
    
    # Wrong password
    wrong_auth = user_service.authenticate_user("auth@example.com", "wrong_password")
    assert wrong_auth is None
```

## 📋 Acceptance Criteria

- [ ] User registration and login functionality implemented
- [ ] JWT-based authentication working
- [ ] Role-based access control enforced
- [ ] User settings management functional
- [ ] Password security with bcrypt hashing
- [ ] Authentication middleware integrated with FastAPI
- [ ] Comprehensive unit tests passing
- [ ] API endpoints for user management created
- [ ] Error handling for authentication failures
- [ ] User session management working

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 03: Authentication Middleware](./task-03-authentication-middleware.md)
2. Integrate authentication with existing API endpoints
3. Add user context to all operations
4. Test multi-user scenarios

## 📝 Notes

- Use bcrypt for password hashing (more secure than basic hashing)
- Implement proper JWT token expiration and refresh
- Add rate limiting for authentication endpoints
- Consider implementing password reset functionality
- Add user activity logging for security auditing
