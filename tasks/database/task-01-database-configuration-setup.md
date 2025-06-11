# Task 01: Database Configuration and Setup

## 📋 Task Overview

**Objective**: Set up the database infrastructure, configuration management, and connection handling for the MoRAG system using the SQL Alchemy schema from `database/database.py` and entity definitions from `database/DATABASE.md`.

**Priority**: Critical - Foundation for all other database tasks
**Estimated Time**: 1-2 weeks
**Dependencies**: None (foundation task)

## 🎯 Goals

1. Integrate the existing SQL Alchemy models into MoRAG core
2. Set up database configuration management
3. Implement connection pooling and session management
4. Create database initialization system
5. Add environment-based database configuration
6. Ensure compatibility with multiple database backends

## 📊 Current State Analysis

### Existing Database Schema
- **Location**: `database/database.py` (implementation) and `database/DATABASE.md` (specification)
- **Models**: User, UserSettings, DatabaseServer, Database, Document, ApiKey, Job
- **Features**: Complete relationships, utility functions, DatabaseManager class
- **Database Support**: PostgreSQL (primary), MySQL (alternative), SQLite (development)
- **Schema Reference**: See `database/DATABASE.md` for complete entity definitions, relationships, and business logic

### Current MoRAG Configuration
- **Config Location**: `packages/morag-core/src/morag_core/config.py`
- **Current Database**: Qdrant vector database only
- **Storage**: File-based temporary storage
- **Session Management**: None (stateless API)

## 🔧 Implementation Plan

### Step 1: Move Database Models to MoRAG Core

**Files to Create/Modify**:
```
packages/morag-core/src/morag_core/
├── database/
│   ├── __init__.py
│   ├── models.py          # Move from database/database.py
│   ├── manager.py         # Database connection management
│   ├── session.py         # Session handling
│   └── initialization.py  # Database initialization
```

**Implementation Details**:

1. **Create Database Package**:
```python
# packages/morag-core/src/morag_core/database/__init__.py
"""Database package for MoRAG core."""

from .models import (
    Base, User, UserSettings, DatabaseServer, Database, 
    Document, ApiKey, Job, UserRole, Theme, DocumentState, 
    DatabaseType, JobStatus
)
from .manager import DatabaseManager, get_database_manager
from .session import get_session, SessionLocal

__all__ = [
    "Base", "User", "UserSettings", "DatabaseServer", "Database",
    "Document", "ApiKey", "Job", "UserRole", "Theme", "DocumentState",
    "DatabaseType", "JobStatus", "DatabaseManager", "get_database_manager",
    "get_session", "SessionLocal"
]
```

2. **Move and Enhance Models**:
```python
# packages/morag-core/src/morag_core/database/models.py
"""
SQLAlchemy Database Models for MoRAG
Implementation of database schema defined in database/DATABASE.md
"""

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, DateTime,
    ForeignKey, Text, Enum as SQLEnum, UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import enum
import uuid

# Import MoRAG configuration
from morag_core.config import get_settings

# Base class for all models
Base = declarative_base()

# Copy all enums and models from database/database.py
# Implement exactly as specified in database/DATABASE.md
# No legacy compatibility - clean implementation only
```

3. **Create Database Manager**:
```python
# packages/morag-core/src/morag_core/database/manager.py
"""Database connection and session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
import structlog
from typing import Optional

from morag_core.config import get_settings
from .models import Base

logger = structlog.get_logger(__name__)

class DatabaseManager:
    """Enhanced database manager with connection pooling."""
    
    def __init__(self, database_url: Optional[str] = None):
        settings = get_settings()
        self.database_url = database_url or settings.database_url
        
        # Configure engine with connection pooling
        self.engine = create_engine(
            self.database_url,
            poolclass=QueuePool,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,
            echo=settings.database_echo
        )
        
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
        
        logger.info("Database manager initialized", url=self.database_url)
    
    def create_tables(self):
        """Create all tables in the database."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all tables in the database."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("Database tables dropped")
    
    @contextmanager
    def get_session(self):
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

# Global database manager instance
_database_manager: Optional[DatabaseManager] = None

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
    return _database_manager
```

### Step 2: Update MoRAG Core Configuration

**File to Modify**: `packages/morag-core/src/morag_core/config.py`

**Add Database Configuration**:
```python
# Add to Settings class
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./morag.db", 
        alias="MORAG_DATABASE_URL",
        description="Database connection URL"
    )
    database_pool_size: int = Field(
        default=5, 
        alias="MORAG_DATABASE_POOL_SIZE",
        description="Database connection pool size"
    )
    database_max_overflow: int = Field(
        default=10, 
        alias="MORAG_DATABASE_MAX_OVERFLOW",
        description="Database connection pool max overflow"
    )
    database_echo: bool = Field(
        default=False, 
        alias="MORAG_DATABASE_ECHO",
        description="Enable SQL query logging"
    )
    
    # Authentication Configuration
    jwt_secret_key: str = Field(
        default="your-secret-key-change-in-production",
        alias="MORAG_JWT_SECRET_KEY",
        description="JWT token secret key"
    )
    jwt_algorithm: str = Field(
        default="HS256",
        alias="MORAG_JWT_ALGORITHM",
        description="JWT token algorithm"
    )
    jwt_expiration_hours: int = Field(
        default=24,
        alias="MORAG_JWT_EXPIRATION_HOURS",
        description="JWT token expiration in hours"
    )
    
    # User Management
    enable_user_registration: bool = Field(
        default=True,
        alias="MORAG_ENABLE_USER_REGISTRATION",
        description="Allow new user registration"
    )
    default_user_role: str = Field(
        default="USER",
        alias="MORAG_DEFAULT_USER_ROLE",
        description="Default role for new users"
    )
```

### Step 3: Create Database Initialization

**Database Initialization**:
```python
# packages/morag-core/src/morag_core/database/initialization.py
"""Database initialization for fresh MoRAG setup."""

import structlog
from sqlalchemy import create_engine, inspect
from .models import Base
from .manager import DatabaseManager

logger = structlog.get_logger(__name__)

class DatabaseInitializer:
    """Initialize database for fresh MoRAG deployment."""

    def __init__(self, database_url: str = None):
        self.db_manager = DatabaseManager(database_url)

    def initialize_database(self, drop_existing: bool = False):
        """Initialize database with fresh schema."""
        if drop_existing:
            logger.warning("Dropping existing database tables")
            Base.metadata.drop_all(bind=self.db_manager.engine)

        logger.info("Creating database tables")
        Base.metadata.create_all(bind=self.db_manager.engine)

        # Verify tables were created
        inspector = inspect(self.db_manager.engine)
        tables = inspector.get_table_names()

        expected_tables = ['users', 'user_settings', 'database_servers',
                          'databases', 'documents', 'api_keys', 'jobs']

        for table in expected_tables:
            if table not in tables:
                raise RuntimeError(f"Failed to create table: {table}")

        logger.info("Database initialization completed", tables_created=len(tables))

    def verify_schema(self) -> bool:
        """Verify database schema matches expected structure."""
        try:
            inspector = inspect(self.db_manager.engine)
            tables = inspector.get_table_names()

            expected_tables = ['users', 'user_settings', 'database_servers',
                              'databases', 'documents', 'api_keys', 'jobs']

            return all(table in tables for table in expected_tables)
        except Exception as e:
            logger.error("Schema verification failed", error=str(e))
            return False
```

### Step 4: Add Database Dependencies

**Update Requirements**:
```python
# packages/morag-core/requirements.txt
# Add database dependencies
sqlalchemy>=1.4.0,<2.0.0
psycopg2-binary>=2.9.0  # PostgreSQL
pymysql>=1.0.0          # MySQL
cryptography>=3.0.0     # For password hashing
bcrypt>=3.2.0           # Password hashing
PyJWT>=2.4.0            # JWT tokens
```

### Step 4: Create Database CLI Commands

**File to Create**: `packages/morag-core/src/morag_core/database/cli.py`

```python
"""Database CLI commands for MoRAG."""

import click
from .manager import get_database_manager
from .initialization import DatabaseInitializer

@click.group()
def database():
    """Database management commands."""
    pass

@database.command()
def init():
    """Initialize the database with fresh schema."""
    initializer = DatabaseInitializer()
    initializer.initialize_database()
    click.echo("Database initialized successfully.")

@database.command()
def reset():
    """Reset the database (WARNING: Destroys all data)."""
    if click.confirm("This will destroy all data. Are you sure?"):
        initializer = DatabaseInitializer()
        initializer.initialize_database(drop_existing=True)
        click.echo("Database reset successfully.")

@database.command()
def verify():
    """Verify database schema."""
    initializer = DatabaseInitializer()
    if initializer.verify_schema():
        click.echo("Database schema is valid.")
    else:
        click.echo("Database schema verification failed.")
        exit(1)

@database.command()
def health():
    """Check database health."""
    manager = get_database_manager()
    if manager.health_check():
        click.echo("Database is healthy.")
    else:
        click.echo("Database health check failed.")
        exit(1)
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_database_setup.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from morag_core.database import DatabaseManager, Base, User

def test_database_manager_initialization():
    """Test database manager initialization."""
    manager = DatabaseManager("sqlite:///:memory:")
    assert manager.database_url == "sqlite:///:memory:"
    assert manager.engine is not None
    assert manager.SessionLocal is not None

def test_database_initialization():
    """Test database initialization."""
    from morag_core.database.initialization import DatabaseInitializer
    initializer = DatabaseInitializer("sqlite:///:memory:")
    initializer.initialize_database()

    # Verify schema
    assert initializer.verify_schema()

    # Verify tables exist
    from sqlalchemy import inspect
    inspector = inspect(initializer.db_manager.engine)
    tables = inspector.get_table_names()
    expected_tables = ['users', 'user_settings', 'database_servers',
                      'databases', 'documents', 'api_keys', 'jobs']

    for table in expected_tables:
        assert table in tables

def test_session_management():
    """Test session context manager."""
    manager = DatabaseManager("sqlite:///:memory:")
    manager.create_tables()
    
    with manager.get_session() as session:
        user = User(name="Test User", email="test@example.com")
        session.add(user)
        # Session should auto-commit on exit
    
    # Verify user was saved
    with manager.get_session() as session:
        saved_user = session.query(User).filter_by(email="test@example.com").first()
        assert saved_user is not None
        assert saved_user.name == "Test User"
```

### Integration Tests
```python
# tests/test_database_integration.py
def test_database_configuration_loading():
    """Test database configuration from environment."""
    import os
    os.environ["MORAG_DATABASE_URL"] = "sqlite:///test.db"

    from morag_core.config import get_settings
    settings = get_settings()
    assert settings.database_url == "sqlite:///test.db"

def test_fresh_database_setup():
    """Test fresh database setup process."""
    from morag_core.database.initialization import DatabaseInitializer
    initializer = DatabaseInitializer("sqlite:///:memory:")

    # Initialize fresh database
    initializer.initialize_database()

    # Verify schema
    assert initializer.verify_schema()
```

## 📋 Acceptance Criteria

- [ ] Database models successfully moved to morag-core
- [ ] Database manager with connection pooling implemented
- [ ] Configuration system supports database settings
- [ ] Database initialization system implemented
- [ ] CLI commands for database management created
- [ ] Unit tests for all database components pass
- [ ] Integration tests with different database backends pass
- [ ] Documentation for database setup completed

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 02: User Management System](./task-02-user-management-system.md)
2. Test database connectivity with different backends
3. Verify fresh database initialization
4. Update deployment documentation with database requirements

## 📝 Notes

- Use SQLite for development and testing
- PostgreSQL recommended for production
- Ensure proper connection pooling for performance
- Add comprehensive logging for database operations
- Consider database backup and recovery procedures
