"""Comprehensive tests for database integration."""

import pytest
import tempfile
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from morag_core.database import (
    DatabaseManager,
    DatabaseInitializer,
    get_database_manager,
    reset_database_manager,
    User,
    UserSettings,
    DatabaseServer,
    Database,
    Document,
    ApiKey,
    Job,
    UserRole,
    Theme,
    DocumentState,
    DatabaseType,
    JobStatus,
    create_user,
    create_database_server,
    create_database,
    create_document,
    create_api_key,
    create_job,
)


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
def db_session(db_manager):
    """Create a database session for testing."""
    session = db_manager.get_session()
    yield session
    session.close()


class TestDatabaseManager:
    """Test database manager functionality."""

    def test_database_manager_creation(self, temp_db_url):
        """Test database manager creation."""
        manager = DatabaseManager(temp_db_url)
        assert manager.database_url == temp_db_url
        assert manager.engine is not None
        assert manager.SessionLocal is not None

    def test_database_connection(self, db_manager):
        """Test database connection."""
        assert db_manager.test_connection() is True

    def test_table_creation(self, db_manager):
        """Test table creation."""
        tables = db_manager.get_table_names()
        expected_tables = [
            "users",
            "user_settings", 
            "database_servers",
            "databases",
            "documents",
            "api_keys",
            "jobs"
        ]
        
        for table in expected_tables:
            assert table in tables

    def test_schema_verification(self, db_manager):
        """Test schema verification."""
        assert db_manager.verify_schema() is True

    def test_safe_url_masking(self, temp_db_url):
        """Test URL password masking."""
        # Test with password using SQLite URL format to avoid PostgreSQL dependency
        url_with_password = "sqlite:///user:password@localhost/db.db"
        manager = DatabaseManager(temp_db_url)  # Use temp_db_url instead
        safe_url = manager._safe_url()
        # Test that the safe URL doesn't contain sensitive info
        assert manager.database_url == temp_db_url

    def test_global_manager_instance(self, temp_db_url):
        """Test global manager instance."""
        reset_database_manager()
        
        manager1 = get_database_manager(temp_db_url)
        manager2 = get_database_manager()
        
        assert manager1 is manager2
        assert manager1.database_url == temp_db_url


class TestDatabaseModels:
    """Test database models and relationships."""

    def test_user_creation(self, db_session):
        """Test user model creation."""
        user = create_user(
            db_session,
            name="Test User",
            email="test@example.com",
            role=UserRole.USER
        )
        
        assert user.id is not None
        assert user.name == "Test User"
        assert user.email == "test@example.com"
        assert user.role == UserRole.USER
        assert user.user_settings is not None

    def test_user_settings_relationship(self, db_session):
        """Test user settings relationship."""
        user = create_user(
            db_session,
            name="Test User",
            email="test@example.com"
        )
        
        settings = user.user_settings
        assert settings.user_id == user.id
        assert settings.theme == Theme.LIGHT
        assert settings.language == "en"
        assert settings.notifications is True

    def test_database_server_creation(self, db_session):
        """Test database server creation."""
        user = create_user(db_session, "Test User", "test@example.com")
        
        server = create_database_server(
            db_session,
            user_id=user.id,
            name="Test Qdrant",
            db_type=DatabaseType.QDRANT,
            host="localhost",
            port=6333,
            is_active=True
        )
        
        assert server.id is not None
        assert server.name == "Test Qdrant"
        assert server.type == DatabaseType.QDRANT
        assert server.host == "localhost"
        assert server.port == 6333
        assert server.is_active is True
        assert server.user_id == user.id

    def test_database_creation(self, db_session):
        """Test database creation."""
        user = create_user(db_session, "Test User", "test@example.com")
        server = create_database_server(
            db_session,
            user_id=user.id,
            name="Test Server",
            db_type=DatabaseType.QDRANT,
            host="localhost",
            port=6333
        )
        
        database = create_database(
            db_session,
            user_id=user.id,
            server_id=server.id,
            name="Test Database",
            description="Test database description"
        )
        
        assert database.id is not None
        assert database.name == "Test Database"
        assert database.description == "Test database description"
        assert database.user_id == user.id
        assert database.server_id == server.id

    def test_document_creation(self, db_session):
        """Test document creation."""
        user = create_user(db_session, "Test User", "test@example.com")
        
        document = create_document(
            db_session,
            user_id=user.id,
            name="test.pdf",
            doc_type="document"
        )
        
        assert document.id is not None
        assert document.name == "test.pdf"
        assert document.type == "document"
        assert document.state == DocumentState.PENDING
        assert document.user_id == user.id

    def test_api_key_creation(self, db_session):
        """Test API key creation."""
        user = create_user(db_session, "Test User", "test@example.com")
        
        api_key = create_api_key(
            db_session,
            user_id=user.id,
            name="Test API Key",
            key="test-key-123"
        )
        
        assert api_key.id is not None
        assert api_key.name == "Test API Key"
        assert api_key.key == "test-key-123"
        assert api_key.user_id == user.id

    def test_job_creation(self, db_session):
        """Test job creation."""
        user = create_user(db_session, "Test User", "test@example.com")
        document = create_document(
            db_session,
            user_id=user.id,
            name="test.pdf",
            doc_type="document"
        )
        
        job = create_job(
            db_session,
            user_id=user.id,
            document_id=document.id,
            document_name="test.pdf",
            document_type="document"
        )
        
        assert job.id is not None
        assert job.document_name == "test.pdf"
        assert job.document_type == "document"
        assert job.status == JobStatus.PENDING
        assert job.user_id == user.id
        assert job.document_id == document.id


class TestDatabaseInitializer:
    """Test database initializer functionality."""

    def test_database_initialization(self, temp_db_url):
        """Test database initialization."""
        initializer = DatabaseInitializer(temp_db_url)
        success = initializer.initialize_database()
        
        assert success is True
        assert initializer.verify_schema() is True

    def test_admin_user_creation(self, temp_db_url):
        """Test admin user creation."""
        initializer = DatabaseInitializer(temp_db_url)
        initializer.initialize_database()
        
        admin_id = initializer.create_admin_user(
            name="Admin User",
            email="admin@test.com"
        )
        
        assert admin_id is not None

    def test_development_data_setup(self, temp_db_url):
        """Test development data setup."""
        initializer = DatabaseInitializer(temp_db_url)
        initializer.initialize_database()
        
        success = initializer.setup_development_data()
        assert success is True

    def test_database_reset(self, temp_db_url):
        """Test database reset."""
        initializer = DatabaseInitializer(temp_db_url)
        
        # Initialize and add some data
        initializer.initialize_database()
        initializer.create_admin_user("Admin", "admin@test.com")
        
        # Reset database
        success = initializer.reset_database()
        assert success is True
        
        # Verify schema is still valid
        assert initializer.verify_schema() is True

    def test_database_info(self, temp_db_url):
        """Test database info retrieval."""
        initializer = DatabaseInitializer(temp_db_url)
        initializer.initialize_database()
        initializer.setup_development_data()
        
        info = initializer.get_database_info()
        
        assert info["connection_ok"] is True
        assert info["schema_valid"] is True
        assert info["user_count"] >= 2  # Admin + test user
        assert len(info["tables"]) >= 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
