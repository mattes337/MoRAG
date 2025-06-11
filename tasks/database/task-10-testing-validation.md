# Task 10: Testing and Validation

## 📋 Task Overview

**Objective**: Implement comprehensive testing and validation framework for the database integration, ensuring reliability, performance, and data integrity across all database operations.

**Priority**: Critical - Required for production readiness
**Estimated Time**: 1-2 weeks
**Dependencies**: Task 09 (User Settings and Preferences)

## 🎯 Goals

1. Create comprehensive unit test suite for database operations
2. Implement integration tests for multi-user scenarios
3. Add performance testing and benchmarking
4. Create data integrity validation tests
5. Implement security testing for authentication and authorization
6. Add load testing for concurrent operations
7. Create automated testing pipeline

## 📊 Testing Strategy

### Test Categories
1. **Unit Tests**: Individual component testing
2. **Integration Tests**: Cross-component functionality
3. **Performance Tests**: Speed and scalability validation
4. **Security Tests**: Authentication and authorization
5. **Data Integrity Tests**: Consistency and reliability
6. **Load Tests**: Concurrent user scenarios
7. **Database Tests**: Schema validation and data integrity

## 🔧 Implementation Plan

### Step 1: Create Testing Infrastructure

**Files to Create**:
```
tests/
├── database/
│   ├── __init__.py
│   ├── conftest.py           # Test configuration and fixtures
│   ├── test_models.py        # Database model tests
│   ├── test_services.py      # Service layer tests
│   ├── test_auth.py          # Authentication tests
│   ├── test_multi_tenancy.py # Multi-tenancy tests
│   ├── test_schema.py        # Schema validation tests
│   └── test_performance.py   # Performance tests
├── integration/
│   ├── __init__.py
│   ├── test_user_workflows.py
│   ├── test_document_lifecycle.py
│   ├── test_job_tracking.py
│   └── test_api_endpoints.py
├── load/
│   ├── __init__.py
│   ├── test_concurrent_users.py
│   ├── test_bulk_operations.py
│   └── test_stress_scenarios.py
└── fixtures/
    ├── __init__.py
    ├── users.py
    ├── documents.py
    └── sample_data.py
```

**Implementation Details**:

1. **Test Configuration**:
```python
# tests/database/conftest.py
"""Test configuration and fixtures for database tests."""

import pytest
import tempfile
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from morag_core.database import Base, get_database_manager, DatabaseManager
from morag_core.database.models import User, UserSettings, Document, Job, ApiKey
from morag_core.auth.service import UserService
from morag_core.documents.service import DocumentService
from morag_core.jobs.service import JobService

@pytest.fixture(scope="session")
def test_database_url():
    """Create a temporary test database."""
    temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
    temp_db.close()
    yield f"sqlite:///{temp_db.name}"
    os.unlink(temp_db.name)

@pytest.fixture(scope="session")
def test_engine(test_database_url):
    """Create test database engine."""
    engine = create_engine(test_database_url, echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture
def test_session(test_engine):
    """Create test database session."""
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()

@pytest.fixture
def test_db_manager(test_database_url):
    """Create test database manager."""
    return DatabaseManager(test_database_url)

@pytest.fixture
def sample_user(test_session):
    """Create a sample user for testing."""
    user = User(
        id="test-user-123",
        name="Test User",
        email="test@example.com",
        password_hash="hashed_password",
        role="USER"
    )
    test_session.add(user)
    test_session.commit()
    return user

@pytest.fixture
def sample_admin_user(test_session):
    """Create a sample admin user for testing."""
    user = User(
        id="admin-user-123",
        name="Admin User",
        email="admin@example.com",
        password_hash="hashed_password",
        role="ADMIN"
    )
    test_session.add(user)
    test_session.commit()
    return user

@pytest.fixture
def sample_document(test_session, sample_user):
    """Create a sample document for testing."""
    document = Document(
        id="test-doc-123",
        name="Test Document",
        type="pdf",
        state="INGESTED",
        user_id=sample_user.id,
        chunks=10,
        quality=0.85
    )
    test_session.add(document)
    test_session.commit()
    return document

@pytest.fixture
def multiple_users(test_session):
    """Create multiple users for multi-tenancy testing."""
    users = []
    for i in range(5):
        user = User(
            id=f"user-{i}",
            name=f"User {i}",
            email=f"user{i}@example.com",
            password_hash="hashed_password",
            role="USER"
        )
        test_session.add(user)
        users.append(user)
    
    test_session.commit()
    return users

@pytest.fixture
def user_service(test_db_manager):
    """Create user service with test database."""
    return UserService()

@pytest.fixture
def document_service(test_db_manager):
    """Create document service with test database."""
    return DocumentService()

@pytest.fixture
def job_service(test_db_manager):
    """Create job service with test database."""
    return JobService()
```

2. **Database Model Tests**:
```python
# tests/database/test_models.py
"""Test database models and relationships."""

import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from morag_core.database.models import (
    User, UserSettings, Document, Job, ApiKey, DatabaseServer, Database,
    UserRole, DocumentState, JobStatus, DatabaseType
)

class TestUserModel:
    """Test User model."""
    
    def test_user_creation(self, test_session):
        """Test user creation with required fields."""
        user = User(
            name="John Doe",
            email="john@example.com",
            role=UserRole.USER
        )
        test_session.add(user)
        test_session.commit()
        
        assert user.id is not None
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.role == UserRole.USER
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_email_uniqueness(self, test_session):
        """Test that user emails must be unique."""
        user1 = User(name="User 1", email="same@example.com", role=UserRole.USER)
        user2 = User(name="User 2", email="same@example.com", role=UserRole.USER)
        
        test_session.add(user1)
        test_session.commit()
        
        test_session.add(user2)
        with pytest.raises(IntegrityError):
            test_session.commit()
    
    def test_user_settings_relationship(self, test_session):
        """Test user-settings relationship."""
        user = User(name="Test User", email="test@example.com", role=UserRole.USER)
        test_session.add(user)
        test_session.flush()
        
        settings = UserSettings(
            user_id=user.id,
            theme="LIGHT",
            language="en",
            notifications=True,
            auto_save=True
        )
        test_session.add(settings)
        test_session.commit()
        
        assert user.user_settings == settings
        assert settings.user == user

class TestDocumentModel:
    """Test Document model."""
    
    def test_document_creation(self, test_session, sample_user):
        """Test document creation."""
        document = Document(
            name="Test Document",
            type="pdf",
            state=DocumentState.PENDING,
            user_id=sample_user.id
        )
        test_session.add(document)
        test_session.commit()
        
        assert document.id is not None
        assert document.name == "Test Document"
        assert document.type == "pdf"
        assert document.state == DocumentState.PENDING
        assert document.user_id == sample_user.id
        assert document.version == 1
        assert document.chunks == 0
        assert document.quality == 0.0
    
    def test_document_user_relationship(self, test_session, sample_user):
        """Test document-user relationship."""
        document = Document(
            name="Test Document",
            type="pdf",
            state=DocumentState.PENDING,
            user_id=sample_user.id
        )
        test_session.add(document)
        test_session.commit()
        
        assert document.user == sample_user
        assert document in sample_user.documents

class TestJobModel:
    """Test Job model."""
    
    def test_job_creation(self, test_session, sample_user, sample_document):
        """Test job creation."""
        job = Job(
            id="job-123",
            document_id=sample_document.id,
            document_name=sample_document.name,
            document_type=sample_document.type,
            user_id=sample_user.id,
            status=JobStatus.PENDING
        )
        test_session.add(job)
        test_session.commit()
        
        assert job.id == "job-123"
        assert job.document_id == sample_document.id
        assert job.user_id == sample_user.id
        assert job.status == JobStatus.PENDING
        assert job.percentage == 0
    
    def test_job_relationships(self, test_session, sample_user, sample_document):
        """Test job relationships."""
        job = Job(
            id="job-123",
            document_id=sample_document.id,
            document_name=sample_document.name,
            document_type=sample_document.type,
            user_id=sample_user.id,
            status=JobStatus.PENDING
        )
        test_session.add(job)
        test_session.commit()
        
        assert job.user == sample_user
        assert job.document == sample_document
        assert job in sample_user.jobs
        assert job in sample_document.jobs
```

3. **Service Layer Tests**:
```python
# tests/database/test_services.py
"""Test service layer functionality."""

import pytest
from datetime import datetime

from morag_core.auth.models import UserCreate, UserLogin
from morag_core.documents.models import DocumentCreate, DocumentType
from morag_core.jobs.models import JobCreate, JobStatus
from morag_core.exceptions import ConflictError, NotFoundError

class TestUserService:
    """Test UserService functionality."""
    
    def test_create_user(self, user_service):
        """Test user creation."""
        user_data = UserCreate(
            name="John Doe",
            email="john@example.com",
            password="secure_password_123"
        )
        
        user = user_service.create_user(user_data)
        
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.role == "USER"
        assert user.id is not None
    
    def test_create_duplicate_user(self, user_service):
        """Test creating user with duplicate email."""
        user_data = UserCreate(
            name="John Doe",
            email="duplicate@example.com",
            password="secure_password_123"
        )
        
        # Create first user
        user_service.create_user(user_data)
        
        # Try to create duplicate
        with pytest.raises(ConflictError):
            user_service.create_user(user_data)
    
    def test_authenticate_user(self, user_service):
        """Test user authentication."""
        user_data = UserCreate(
            name="Auth Test",
            email="auth@example.com",
            password="secure_password_123"
        )
        
        # Create user
        created_user = user_service.create_user(user_data)
        
        # Authenticate with correct credentials
        authenticated_user = user_service.authenticate_user("auth@example.com", "secure_password_123")
        assert authenticated_user is not None
        assert authenticated_user.id == created_user.id
        
        # Authenticate with wrong password
        wrong_auth = user_service.authenticate_user("auth@example.com", "wrong_password")
        assert wrong_auth is None
        
        # Authenticate with non-existent email
        no_user = user_service.authenticate_user("nonexistent@example.com", "password")
        assert no_user is None

class TestDocumentService:
    """Test DocumentService functionality."""
    
    def test_create_document(self, document_service, sample_user):
        """Test document creation."""
        document_data = DocumentCreate(
            name="Test Document",
            type=DocumentType.PDF
        )
        
        document = document_service.create_document(sample_user.id, document_data)
        
        assert document.name == "Test Document"
        assert document.type == DocumentType.PDF
        assert document.user_id == sample_user.id
        assert document.state.value == "PENDING"
    
    def test_search_documents(self, document_service, sample_user):
        """Test document search functionality."""
        # Create multiple documents
        for i in range(5):
            document_data = DocumentCreate(
                name=f"Document {i}",
                type=DocumentType.PDF
            )
            document_service.create_document(sample_user.id, document_data)
        
        # Search all documents
        from morag_core.documents.models import DocumentSearchRequest
        search_request = DocumentSearchRequest(limit=10)
        documents, total = document_service.search_documents(sample_user.id, search_request)
        
        assert len(documents) == 5
        assert total == 5
        
        # Search with query
        search_request = DocumentSearchRequest(query="Document 2")
        documents, total = document_service.search_documents(sample_user.id, search_request)
        
        assert len(documents) == 1
        assert documents[0].name == "Document 2"

class TestJobService:
    """Test JobService functionality."""
    
    def test_create_job(self, job_service, sample_user, sample_document):
        """Test job creation."""
        job_data = JobCreate(
            document_id=sample_document.id,
            document_name=sample_document.name,
            document_type=sample_document.type
        )
        
        job = job_service.create_job(sample_user.id, job_data, "celery-task-123")
        
        assert job.document_id == sample_document.id
        assert job.user_id == sample_user.id
        assert job.status == JobStatus.PENDING
        assert job.id == "celery-task-123"
    
    def test_update_job_progress(self, job_service, sample_user, sample_document):
        """Test job progress updates."""
        job_data = JobCreate(
            document_id=sample_document.id,
            document_name=sample_document.name,
            document_type=sample_document.type
        )
        
        job = job_service.create_job(sample_user.id, job_data, "progress-test-123")
        
        # Update progress
        from morag_core.jobs.models import JobUpdate
        update = JobUpdate(
            status=JobStatus.PROCESSING,
            percentage=50,
            summary="Processing document..."
        )
        
        updated_job = job_service.update_job(job.id, update)
        
        assert updated_job.status == JobStatus.PROCESSING
        assert updated_job.percentage == 50
        assert updated_job.summary == "Processing document..."
```

### Step 2: Create Integration Tests

**File to Create**: `tests/integration/test_user_workflows.py`

```python
# tests/integration/test_user_workflows.py
"""Integration tests for complete user workflows."""

import pytest
from fastapi.testclient import TestClient

from morag.server import app
from morag_core.auth.models import UserCreate, UserLogin

class TestUserWorkflows:
    """Test complete user workflows."""
    
    def test_user_registration_and_login_workflow(self):
        """Test complete user registration and login workflow."""
        client = TestClient(app)
        
        # Register user
        registration_data = {
            "name": "Integration Test User",
            "email": "integration@example.com",
            "password": "secure_password_123"
        }
        
        response = client.post("/auth/register", json=registration_data)
        assert response.status_code == 200
        user_data = response.json()
        assert user_data["email"] == "integration@example.com"
        
        # Login user
        login_data = {
            "email": "integration@example.com",
            "password": "secure_password_123"
        }
        
        response = client.post("/auth/login", json=login_data)
        assert response.status_code == 200
        token_data = response.json()
        assert "access_token" in token_data
        
        # Access protected endpoint
        headers = {"Authorization": f"Bearer {token_data['access_token']}"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        current_user = response.json()
        assert current_user["email"] == "integration@example.com"
    
    def test_document_processing_workflow(self):
        """Test complete document processing workflow."""
        client = TestClient(app)
        
        # Register and login user
        registration_data = {
            "name": "Doc Test User",
            "email": "doctest@example.com",
            "password": "secure_password_123"
        }
        client.post("/auth/register", json=registration_data)
        
        login_response = client.post("/auth/login", json={
            "email": "doctest@example.com",
            "password": "secure_password_123"
        })
        token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        
        # Upload and process document
        test_content = b"This is a test document content for processing."
        files = {"file": ("test.txt", test_content, "text/plain")}
        
        response = client.post("/process/file", files=files, headers=headers)
        assert response.status_code == 200
        result = response.json()
        assert "content" in result
        
        # Ingest document
        files = {"file": ("test_ingest.txt", test_content, "text/plain")}
        data = {"document_id": "test-doc-123"}
        
        response = client.post("/api/v1/ingest/file", files=files, data=data, headers=headers)
        assert response.status_code == 200
        ingest_result = response.json()
        assert "task_id" in ingest_result
```

### Step 3: Create Performance Tests

**File to Create**: `tests/database/test_performance.py`

```python
# tests/database/test_performance.py
"""Performance tests for database operations."""

import pytest
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from morag_core.auth.models import UserCreate
from morag_core.documents.models import DocumentCreate, DocumentType

class TestDatabasePerformance:
    """Test database performance."""
    
    def test_user_creation_performance(self, user_service):
        """Test user creation performance."""
        start_time = time.time()
        
        # Create 100 users
        for i in range(100):
            user_data = UserCreate(
                name=f"Performance User {i}",
                email=f"perf{i}@example.com",
                password="password123"
            )
            user_service.create_user(user_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should create 100 users in less than 10 seconds
        assert duration < 10.0
        print(f"Created 100 users in {duration:.2f} seconds")
    
    def test_document_search_performance(self, document_service, sample_user):
        """Test document search performance."""
        # Create 1000 documents
        for i in range(1000):
            document_data = DocumentCreate(
                name=f"Performance Document {i}",
                type=DocumentType.PDF
            )
            document_service.create_document(sample_user.id, document_data)
        
        # Test search performance
        from morag_core.documents.models import DocumentSearchRequest
        
        start_time = time.time()
        search_request = DocumentSearchRequest(query="Document 500", limit=50)
        documents, total = document_service.search_documents(sample_user.id, search_request)
        end_time = time.time()
        
        duration = end_time - start_time
        
        # Search should complete in less than 1 second
        assert duration < 1.0
        assert len(documents) > 0
        print(f"Searched 1000 documents in {duration:.3f} seconds")
    
    def test_concurrent_operations(self, user_service):
        """Test concurrent database operations."""
        def create_user(index):
            user_data = UserCreate(
                name=f"Concurrent User {index}",
                email=f"concurrent{index}@example.com",
                password="password123"
            )
            return user_service.create_user(user_data)
        
        start_time = time.time()
        
        # Create 50 users concurrently
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_user, i) for i in range(50)]
            results = [future.result() for future in as_completed(futures)]
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert len(results) == 50
        # Concurrent creation should be faster than sequential
        assert duration < 5.0
        print(f"Created 50 users concurrently in {duration:.2f} seconds")
```

## 📋 Acceptance Criteria

- [ ] Comprehensive unit test suite with >95% coverage
- [ ] Integration tests for all major workflows
- [ ] Performance tests with benchmarks established
- [ ] Security tests for authentication and authorization
- [ ] Data integrity validation tests
- [ ] Load tests for concurrent operations
- [ ] Migration tests for schema changes
- [ ] Automated testing pipeline configured
- [ ] Test documentation and guidelines
- [ ] Performance benchmarks documented

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 11: Documentation and Deployment](./task-11-documentation-deployment.md)
2. Set up continuous integration testing
3. Create performance monitoring dashboards
4. Implement automated testing in deployment pipeline

## 📝 Notes

- Maintain high test coverage (>95%) for all database operations
- Use realistic test data that mirrors production scenarios
- Implement proper test isolation to prevent test interference
- Add performance regression testing to catch performance degradation
- Create comprehensive test documentation for future developers
