# Task 03: Authentication Middleware

## 📋 Task Overview

**Objective**: Integrate authentication middleware with the MoRAG FastAPI application, add user authentication endpoints, and implement user context throughout the system.

**Priority**: Critical - Required for secure multi-user operations
**Estimated Time**: 1 week
**Dependencies**: Task 02 (User Management System)

## 🎯 Goals

1. Integrate authentication middleware with MoRAG FastAPI server
2. Create user authentication API endpoints
3. Add user context to all existing endpoints
4. Implement optional authentication (backward compatibility)
5. Add user-specific data isolation
6. Create authentication documentation and examples

## 📊 Current State Analysis

### Current MoRAG Server
- **Location**: `packages/morag/src/morag/server.py`
- **Authentication**: None - completely open API
- **User Context**: No user tracking or isolation
- **Endpoints**: Processing and ingestion without user association

### Required Integration Points
- FastAPI application initialization
- All existing API endpoints
- Celery task execution with user context
- Qdrant collection management per user
- File upload and processing with user association

## 🔧 Implementation Plan

### Step 1: Update MoRAG Server with Authentication

**File to Modify**: `packages/morag/src/morag/server.py`

**Add Authentication Imports**:
```python
# Add to imports
from morag_core.auth import (
    UserService, AuthenticationMiddleware,
    get_current_user, require_authentication, require_admin, require_user
)
from morag_core.auth.models import (
    UserCreate, UserLogin, UserResponse, UserUpdate, 
    UserSettingsUpdate, TokenResponse
)
from morag_core.database import get_database_manager
```

**Add Authentication Endpoints**:
```python
# Add after existing imports and before endpoint definitions

# Initialize authentication components
user_service = UserService()
auth_middleware = AuthenticationMiddleware()

# Authentication endpoints
@app.post("/auth/register", response_model=UserResponse, tags=["Authentication"])
async def register_user(user_data: UserCreate):
    """Register a new user."""
    try:
        user = user_service.create_user(user_data)
        logger.info("User registered", user_id=user.id, email=user.email)
        return user
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/login", response_model=TokenResponse, tags=["Authentication"])
async def login_user(login_data: UserLogin):
    """Authenticate user and return access token."""
    user = user_service.authenticate_user(login_data.email, login_data.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    # Create JWT token
    token_data = user_service.jwt_manager.create_access_token(
        user.id, user.email, user.role
    )
    
    logger.info("User logged in", user_id=user.id, email=user.email)
    return TokenResponse(
        access_token=token_data["access_token"],
        token_type=token_data["token_type"],
        expires_in=token_data["expires_in"],
        user=user
    )

@app.get("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def get_current_user_info(current_user: UserResponse = require_authentication):
    """Get current user information."""
    return current_user

@app.put("/auth/me", response_model=UserResponse, tags=["Authentication"])
async def update_current_user(
    user_data: UserUpdate,
    current_user: UserResponse = require_authentication
):
    """Update current user information."""
    try:
        updated_user = user_service.update_user(current_user.id, user_data)
        logger.info("User updated", user_id=current_user.id)
        return updated_user
    except ConflictError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/auth/settings", tags=["Authentication"])
async def get_user_settings(current_user: UserResponse = require_authentication):
    """Get current user settings."""
    with get_database_manager().get_session() as session:
        settings = session.query(UserSettings).filter_by(user_id=current_user.id).first()
        if not settings:
            raise HTTPException(status_code=404, detail="User settings not found")
        return user_service._settings_to_dict(settings)

@app.put("/auth/settings", tags=["Authentication"])
async def update_user_settings(
    settings_data: UserSettingsUpdate,
    current_user: UserResponse = require_authentication
):
    """Update current user settings."""
    try:
        updated_settings = user_service.update_user_settings(current_user.id, settings_data)
        logger.info("User settings updated", user_id=current_user.id)
        return updated_settings
    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/auth/refresh", response_model=TokenResponse, tags=["Authentication"])
async def refresh_token(current_user: UserResponse = require_authentication):
    """Refresh access token."""
    token_data = user_service.jwt_manager.create_access_token(
        current_user.id, current_user.email, current_user.role
    )
    
    return TokenResponse(
        access_token=token_data["access_token"],
        token_type=token_data["token_type"],
        expires_in=token_data["expires_in"],
        user=current_user
    )
```

### Step 2: Add Optional Authentication to Existing Endpoints

**Modify Processing Endpoints**:
```python
# Update existing endpoints to support optional authentication
@app.post("/process/file", response_model=ProcessingResult, tags=["Processing"])
async def process_file(
    source_type: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    use_docling: Optional[bool] = Form(default=False),
    chunk_size: Optional[int] = Form(default=None),
    chunk_overlap: Optional[int] = Form(default=None),
    chunking_strategy: Optional[str] = Form(default=None),
    include_thumbnails: Optional[bool] = Form(default=False),
    current_user: Optional[UserResponse] = get_current_user  # Optional auth
):
    """Process uploaded file with optional user context."""
    
    # Add user context to processing options
    user_context = {
        "user_id": current_user.id if current_user else None,
        "user_email": current_user.email if current_user else None
    }
    
    # ... rest of existing processing logic with user context
```

**Modify Ingestion Endpoints**:
```python
@app.post("/api/v1/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
    source_type: Optional[str] = Form(default=None),
    file: UploadFile = File(...),
    document_id: Optional[str] = Form(default=None),
    replace_existing: bool = Form(default=False),
    webhook_url: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
    use_docling: Optional[bool] = Form(default=False),
    chunk_size: Optional[int] = Form(default=None),
    chunk_overlap: Optional[int] = Form(default=None),
    chunking_strategy: Optional[str] = Form(default=None),
    remote: Optional[bool] = Form(default=False),
    current_user: Optional[UserResponse] = get_current_user  # Optional auth
):
    """Ingest file with optional user context."""
    
    # User context for ingestion
    user_context = {
        "user_id": current_user.id if current_user else None,
        "user_email": current_user.email if current_user else None,
        "collection_name": f"user_{current_user.id}_documents" if current_user else "morag_documents"
    }
    
    # ... rest of existing ingestion logic with user context
```

### Step 3: Update Task Execution with User Context

**File to Modify**: `packages/morag/src/morag/ingest_tasks.py`

**Add User Context to Tasks**:
```python
# Update task signatures to include user context
@celery_app.task(bind=True, name="ingest_file_task")
def ingest_file_task(self, file_path: str, source_type: str, task_options: dict, user_context: dict = None):
    """Ingest file task with user context."""
    
    # Extract user information
    user_id = user_context.get("user_id") if user_context else None
    collection_name = user_context.get("collection_name", "morag_documents")
    
    logger.info("Starting file ingestion task", 
               task_id=self.request.id,
               file_path=file_path,
               source_type=source_type,
               user_id=user_id,
               collection_name=collection_name)
    
    try:
        # Update job status with user context
        if user_id:
            update_job_status_with_user(self.request.id, user_id, JobStatus.PROCESSING, 10)
        
        # ... rest of existing processing logic
        
        # Store in user-specific collection
        if content and content.strip():
            vector_ids = await store_content_in_vector_db(
                content=content,
                metadata=enhanced_metadata,
                collection_name=collection_name,  # User-specific collection
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                document_id=document_id,
                replace_existing=replace_existing
            )
            
            # Update document record with user association
            if user_id:
                update_document_record(document_id, user_id, len(vector_ids))
        
        # ... rest of task logic
        
    except Exception as e:
        logger.error("File ingestion task failed", 
                    task_id=self.request.id, 
                    error=str(e),
                    user_id=user_id)
        
        if user_id:
            update_job_status_with_user(self.request.id, user_id, JobStatus.FAILED, 0, str(e))
        
        raise
```

### Step 4: Add User-Specific Collection Management

**File to Create**: `packages/morag/src/morag/services/user_collection_service.py`

```python
"""User-specific collection management service."""

import structlog
from typing import Optional, List, Dict, Any
from morag_services import QdrantVectorStorage
from morag_core.auth.models import UserResponse
from morag_core.database import get_database_manager, Database, DatabaseServer

logger = structlog.get_logger(__name__)

class UserCollectionService:
    """Manage user-specific Qdrant collections."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def get_user_collection_name(self, user_id: str, database_name: str = "default") -> str:
        """Get collection name for user and database."""
        return f"user_{user_id}_{database_name}"
    
    def get_user_collections(self, user_id: str) -> List[str]:
        """Get all collections for a user."""
        with self.db_manager.get_session() as session:
            databases = session.query(Database).filter_by(user_id=user_id).all()
            return [self.get_user_collection_name(user_id, db.name) for db in databases]
    
    async def create_user_collection(self, user_id: str, database_name: str) -> bool:
        """Create a new collection for user."""
        collection_name = self.get_user_collection_name(user_id, database_name)
        
        try:
            # Get user's database server configuration
            with self.db_manager.get_session() as session:
                database = session.query(Database).filter_by(
                    user_id=user_id, name=database_name
                ).first()
                
                if not database:
                    logger.error("Database not found", user_id=user_id, database_name=database_name)
                    return False
                
                server = database.server
                
                # Create Qdrant storage with user's server config
                storage = QdrantVectorStorage(
                    host=server.host,
                    port=server.port,
                    api_key=server.api_key,
                    collection_name=collection_name
                )
                
                await storage.create_collection(collection_name)
                logger.info("User collection created", 
                           user_id=user_id, 
                           collection_name=collection_name)
                return True
                
        except Exception as e:
            logger.error("Failed to create user collection", 
                        user_id=user_id, 
                        collection_name=collection_name,
                        error=str(e))
            return False
    
    async def delete_user_collection(self, user_id: str, database_name: str) -> bool:
        """Delete a user's collection."""
        collection_name = self.get_user_collection_name(user_id, database_name)
        
        try:
            # Implementation depends on Qdrant client capabilities
            # This is a placeholder for collection deletion
            logger.info("User collection deleted", 
                       user_id=user_id, 
                       collection_name=collection_name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete user collection", 
                        user_id=user_id, 
                        collection_name=collection_name,
                        error=str(e))
            return False
```

### Step 5: Add Database Health Check with Authentication

**Update Health Check Endpoint**:
```python
@app.get("/health", tags=["System"])
async def health_check():
    """System health check including database."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }
    
    # Check database connectivity
    try:
        db_manager = get_database_manager()
        if db_manager.health_check():
            health_status["services"]["database"] = "healthy"
        else:
            health_status["services"]["database"] = "unhealthy"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Check Qdrant connectivity
    try:
        from morag_services import QdrantVectorStorage
        storage = QdrantVectorStorage()
        await storage.connect()
        health_status["services"]["qdrant"] = "healthy"
    except Exception as e:
        health_status["services"]["qdrant"] = f"error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

@app.get("/admin/health", tags=["Administration"])
async def admin_health_check(current_user: UserResponse = require_admin):
    """Detailed health check for administrators."""
    # Extended health check with user statistics, database metrics, etc.
    health_data = await health_check()
    
    # Add admin-specific information
    with get_database_manager().get_session() as session:
        user_count = session.query(User).count()
        document_count = session.query(Document).count()
        job_count = session.query(Job).count()
        
        health_data["statistics"] = {
            "total_users": user_count,
            "total_documents": document_count,
            "total_jobs": job_count
        }
    
    return health_data
```

## 🧪 Testing Requirements

### Integration Tests
```python
# tests/test_authentication_integration.py
import pytest
from fastapi.testclient import TestClient
from morag.server import app

client = TestClient(app)

def test_user_registration():
    """Test user registration endpoint."""
    response = client.post("/auth/register", json={
        "name": "Test User",
        "email": "test@example.com",
        "password": "secure_password_123"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["name"] == "Test User"
    assert "id" in data

def test_user_login():
    """Test user login endpoint."""
    # First register a user
    client.post("/auth/register", json={
        "name": "Login Test",
        "email": "login@example.com",
        "password": "secure_password_123"
    })
    
    # Then login
    response = client.post("/auth/login", json={
        "email": "login@example.com",
        "password": "secure_password_123"
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert "user" in data

def test_authenticated_endpoint():
    """Test accessing authenticated endpoint."""
    # Register and login
    client.post("/auth/register", json={
        "name": "Auth Test",
        "email": "auth@example.com",
        "password": "secure_password_123"
    })
    
    login_response = client.post("/auth/login", json={
        "email": "auth@example.com",
        "password": "secure_password_123"
    })
    token = login_response.json()["access_token"]
    
    # Access authenticated endpoint
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/auth/me", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "auth@example.com"

def test_optional_authentication():
    """Test endpoints with optional authentication."""
    # Test without authentication
    with open("test_file.txt", "w") as f:
        f.write("Test content")
    
    with open("test_file.txt", "rb") as f:
        response = client.post("/process/file", files={"file": f})
    assert response.status_code == 200
    
    # Test with authentication
    # ... (similar test with auth headers)
```

## 📋 Acceptance Criteria

- [ ] Authentication endpoints implemented and working
- [ ] Optional authentication on existing endpoints
- [ ] User context passed to all operations
- [ ] User-specific collection management
- [ ] JWT token validation working
- [ ] Role-based access control enforced
- [ ] Backward compatibility maintained
- [ ] Comprehensive integration tests passing
- [ ] API documentation updated
- [ ] Error handling for authentication failures

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 04: Document Lifecycle Management](./task-04-document-lifecycle-management.md)
2. Test multi-user scenarios thoroughly
3. Add user activity logging
4. Implement rate limiting for authentication endpoints

## 📝 Notes

- Maintain backward compatibility - existing API calls should work without authentication
- Add proper error handling for authentication failures
- Consider implementing API key authentication for programmatic access
- Add comprehensive logging for security auditing
- Test with multiple users to ensure proper data isolation
