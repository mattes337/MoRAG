"""Simple database integration tests."""

import pytest
import tempfile
import os
from datetime import datetime, timezone

from morag_core.database import DatabaseManager
from morag_core.database.models import User, UserSettings, UserRole, Theme


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
    return manager


class TestDatabaseBasics:
    """Test basic database functionality."""

    def test_database_creation(self, db_manager):
        """Test database and table creation."""
        assert db_manager.test_connection()
        
        # Check that tables exist
        tables = db_manager.get_table_names()
        expected_tables = [
            'users', 'user_settings', 'database_servers', 'databases',
            'documents', 'api_keys', 'jobs'
        ]
        
        for table in expected_tables:
            assert table in tables

    def test_user_creation_direct(self, db_manager):
        """Test direct user creation using database models."""
        from morag_core.database.session import get_session_context
        from morag_core.database.models import create_user

        with get_session_context(db_manager) as session:
            # Create user using utility function
            user = create_user(
                session,
                name="Test User",
                email="test@example.com",
                role=UserRole.USER
            )

            # Verify user was created
            assert user.id is not None
            assert user.email == "test@example.com"
            assert user.user_settings is not None
            assert user.user_settings.theme == Theme.LIGHT

    def test_api_key_creation_direct(self, db_manager):
        """Test direct API key creation."""
        from morag_core.database.session import get_session_context
        from morag_core.database.models import create_user, create_api_key

        with get_session_context(db_manager) as session:
            # Create user first
            user = create_user(
                session,
                name="API User",
                email="api@example.com"
            )

            # Create API key
            api_key = create_api_key(
                session,
                user_id=user.id,
                name="Test API Key",
                key="hashed_api_key"
            )

            # Verify API key was created
            assert api_key.id is not None
            assert api_key.name == "Test API Key"
            assert api_key.user_id == user.id

    def test_document_creation_direct(self, db_manager):
        """Test direct document creation."""
        from morag_core.database.session import get_session_context
        from morag_core.database.models import create_user, create_document, DocumentState

        with get_session_context(db_manager) as session:
            # Create user first
            user = create_user(
                session,
                name="Doc User",
                email="doc@example.com"
            )

            # Create document
            document = create_document(
                session,
                user_id=user.id,
                name="Test Document",
                doc_type="document"
            )

            # Verify document was created
            assert document.id is not None
            assert document.name == "Test Document"
            assert document.user_id == user.id
            assert document.state == DocumentState.PENDING

    def test_job_creation_direct(self, db_manager):
        """Test direct job creation."""
        from morag_core.database.session import get_session_context
        from morag_core.database.models import create_user, create_document, create_job, JobStatus

        with get_session_context(db_manager) as session:
            # Create user first
            user = create_user(
                session,
                name="Job User",
                email="job@example.com"
            )

            # Create document first (required for job)
            document = create_document(
                session,
                user_id=user.id,
                name="test_job.pdf",
                doc_type="document"
            )

            # Create job
            job = create_job(
                session,
                user_id=user.id,
                document_id=document.id,
                document_name="test_job.pdf",
                document_type="document"
            )

            # Verify job was created
            assert job.id is not None
            assert job.document_name == "test_job.pdf"
            assert job.user_id == user.id
            assert job.status == JobStatus.PENDING

    def test_database_relationships(self, db_manager):
        """Test database relationships work correctly."""
        from morag_core.database.session import get_session_context
        from morag_core.database.models import (
            create_user, create_document, create_job, create_api_key,
            DocumentState, JobStatus
        )

        with get_session_context(db_manager) as session:
            # Create user
            user = create_user(
                session,
                name="Relationship User",
                email="rel@example.com"
            )

            # Create related entities
            document = create_document(
                session,
                user_id=user.id,
                name="Related Document",
                doc_type="document"
            )

            job = create_job(
                session,
                user_id=user.id,
                document_id=document.id,
                document_name="Related Job",
                document_type="document"
            )

            api_key = create_api_key(
                session,
                user_id=user.id,
                name="Related API Key",
                key="hashed_key"
            )

            # Test relationships
            assert user.user_settings.theme == Theme.LIGHT  # Default theme
            assert len(user.documents) == 1
            assert user.documents[0].name == "Related Document"
            assert len(user.jobs) == 1
            assert user.jobs[0].document_name == "Related Job"
            assert len(user.api_keys) == 1
            assert user.api_keys[0].name == "Related API Key"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
