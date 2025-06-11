"""Comprehensive tests for complete database integration."""

import pytest
import tempfile
import os
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from morag_core.database import (
    DatabaseManager, DatabaseInitializer, get_database_manager, reset_database_manager,
    User, UserSettings, DatabaseServer, Database, Document, ApiKey, Job,
    UserRole, Theme, DocumentState, DatabaseType, JobStatus
)
from morag_core.auth import UserService, UserCreate, UserLogin
from morag_core.api_keys import ApiKeyService, ApiKeyCreate
from morag_core.tenancy import TenantService, ResourceType
from morag_core.jobs import JobService, JobTracker, JobCreate
from morag_core.document import DocumentService, DocumentLifecycleManager


@pytest.fixture
def temp_db_url():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    db_url = f"sqlite:///{db_path}"
    yield db_url
    
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def db_manager(temp_db_url):
    """Create a database manager for testing."""
    manager = DatabaseManager(temp_db_url)
    manager.create_tables()
    yield manager
    reset_database_manager()


@pytest.fixture
def test_user(db_manager):
    """Create a test user."""
    # Use the test database manager
    from morag_core.database import reset_database_manager
    reset_database_manager()

    # Initialize services with test database
    user_service = UserService()
    user_service.db_manager = db_manager

    user_data = UserCreate(
        name="Test User",
        email="test@example.com",
        password="secure_password_123"
    )
    return user_service.create_user(user_data)


class TestCompleteIntegration:
    """Test complete database integration."""

    def test_user_lifecycle(self, db_manager):
        """Test complete user lifecycle."""
        # Use the test database manager
        from morag_core.database import reset_database_manager
        reset_database_manager()

        user_service = UserService()
        user_service.db_manager = db_manager

        # Create user
        user_data = UserCreate(
            name="Integration Test User",
            email="integration@example.com",
            password="secure_password_123"
        )
        user = user_service.create_user(user_data)
        
        assert user.id is not None
        assert user.email == "integration@example.com"
        assert user.user_settings is not None
        
        # Authenticate user
        authenticated_user = user_service.authenticate_user(
            "integration@example.com", 
            "secure_password_123"
        )
        assert authenticated_user is not None
        assert authenticated_user.id == user.id
        
        # Update user
        from morag_core.auth.models import UserUpdate
        update_data = UserUpdate(name="Updated User Name")
        updated_user = user_service.update_user(user.id, update_data)
        assert updated_user.name == "Updated User Name"

    def test_api_key_management(self, db_manager, test_user):
        """Test API key management."""
        api_key_service = ApiKeyService()
        
        # Create API key
        api_key_data = ApiKeyCreate(
            name="Test API Key",
            description="Test key for integration testing"
        )
        api_key_response = api_key_service.create_api_key(api_key_data, test_user.id)
        
        assert api_key_response.api_key.name == "Test API Key"
        assert api_key_response.secret_key.startswith("mk_")
        
        # Authenticate with API key
        user_id = api_key_service.authenticate_api_key(api_key_response.secret_key)
        assert user_id == test_user.id
        
        # List API keys
        from morag_core.api_keys.models import ApiKeySearchRequest
        search_request = ApiKeySearchRequest()
        search_response = api_key_service.list_api_keys(search_request, test_user.id)
        assert len(search_response.api_keys) == 1
        assert search_response.api_keys[0].name == "Test API Key"

    def test_tenant_management(self, db_manager, test_user):
        """Test multi-tenancy features."""
        tenant_service = TenantService()
        
        # Get tenant info
        tenant_info = tenant_service.get_tenant_info(test_user.id)
        assert tenant_info is not None
        assert tenant_info.user_id == test_user.id
        assert tenant_info.user_email == test_user.email
        
        # Check quotas
        can_create_document = tenant_service.check_resource_quota(
            test_user.id, ResourceType.DOCUMENTS, 1
        )
        assert can_create_document is True
        
        # Get user collections
        collections = tenant_service.get_user_collections(test_user.id)
        assert isinstance(collections, list)

    def test_job_tracking(self, db_manager, test_user):
        """Test job tracking integration."""
        job_service = JobService()
        job_tracker = JobTracker()
        
        # Create job
        job_data = JobCreate(
            document_name="test_document.pdf",
            document_type="document",
            document_id="test_doc_123",
            summary="Test job for integration testing"
        )
        job = job_service.create_job(job_data, test_user.id)
        
        assert job.document_name == "test_document.pdf"
        assert job.status == JobStatus.PENDING
        assert job.user_id == test_user.id
        
        # Update job progress
        job_tracker.mark_processing(job.id, test_user.id)
        updated_job = job_service.get_job(job.id, test_user.id)
        assert updated_job.status == JobStatus.PROCESSING
        
        # Complete job
        job_tracker.mark_completed(job.id, "Processing completed successfully", test_user.id)
        completed_job = job_service.get_job(job.id, test_user.id)
        assert completed_job.status == JobStatus.FINISHED

    def test_document_lifecycle(self, db_manager, test_user):
        """Test document lifecycle management."""
        document_service = DocumentService()
        lifecycle_manager = DocumentLifecycleManager()
        
        # Create document
        from morag_core.document.models import DocumentCreate, DocumentType
        doc_data = DocumentCreate(
            title="Test Document",
            source_path="/test/path/document.pdf",
            content_type=DocumentType.DOCUMENT
        )
        document = document_service.create_document(doc_data, test_user.id)
        
        assert document.title == "Test Document"
        assert document.status == DocumentState.PENDING
        assert document.user_id == test_user.id
        
        # Start processing
        lifecycle_manager.mark_processing_started(document.id, test_user.id)
        updated_doc = document_service.get_document(document.id, test_user.id)
        assert updated_doc.status == DocumentState.INGESTING
        
        # Complete processing
        lifecycle_manager.mark_processing_completed(
            document.id, test_user.id, chunks_created=5, quality_score=0.95
        )
        completed_doc = document_service.get_document(document.id, test_user.id)
        assert completed_doc.status == DocumentState.INGESTED

    def test_database_server_management(self, db_manager, test_user):
        """Test database server management."""
        from morag_core.database import create_database_server, create_database
        from morag_core.database.session import get_session_context
        
        with get_session_context(db_manager) as session:
            # Create database server
            server = create_database_server(
                session,
                user_id=test_user.id,
                name="Test Qdrant Server",
                db_type=DatabaseType.QDRANT,
                host="localhost",
                port=6333,
                is_active=True
            )
            
            assert server.name == "Test Qdrant Server"
            assert server.type == DatabaseType.QDRANT
            assert server.user_id == test_user.id
            
            # Create database
            database = create_database(
                session,
                user_id=test_user.id,
                server_id=server.id,
                name="Test Database",
                description="Test database for integration testing"
            )
            
            assert database.name == "Test Database"
            assert database.user_id == test_user.id
            assert database.server_id == server.id

    def test_complete_workflow(self, db_manager):
        """Test complete workflow from user creation to document processing."""
        # Initialize services
        user_service = UserService()
        api_key_service = ApiKeyService()
        tenant_service = TenantService()
        document_service = DocumentService()
        job_service = JobService()
        
        # 1. Create user
        user_data = UserCreate(
            name="Workflow Test User",
            email="workflow@example.com",
            password="secure_password_123"
        )
        user = user_service.create_user(user_data)
        
        # 2. Create API key
        api_key_data = ApiKeyCreate(name="Workflow API Key")
        api_key_response = api_key_service.create_api_key(api_key_data, user.id)
        
        # 3. Check tenant quotas
        can_create_doc = tenant_service.check_resource_quota(user.id, ResourceType.DOCUMENTS)
        assert can_create_doc is True
        
        # 4. Create document
        from morag_core.document.models import DocumentCreate, DocumentType
        doc_data = DocumentCreate(
            title="Workflow Test Document",
            content_type=DocumentType.DOCUMENT
        )
        document = document_service.create_document(doc_data, user.id)
        
        # 5. Create processing job
        from morag_core.jobs.models import JobCreate
        job_data = JobCreate(
            document_name=document.title,
            document_type="document",
            document_id=document.id
        )
        job = job_service.create_job(job_data, user.id)
        
        # 6. Verify all components work together
        assert user.id is not None
        assert api_key_response.secret_key is not None
        assert document.id is not None
        assert job.id is not None
        
        # 7. Verify relationships
        tenant_info = tenant_service.get_tenant_info(user.id)
        assert tenant_info.usage_stats.total_documents >= 1
        assert tenant_info.usage_stats.total_api_keys >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
