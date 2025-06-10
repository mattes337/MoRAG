# Task 2: Database Schema for Remote Job Tracking

## Overview

Implement the database schema and migration system for remote job tracking. This includes creating the necessary tables, indexes, and database utilities to support the remote conversion system.

## Objectives

1. Create database migration scripts for remote job tables
2. Implement database connection and session management
3. Add proper indexing for performance optimization
4. Create database initialization and cleanup utilities
5. Ensure compatibility with existing MoRAG database structure

## Technical Requirements

### 1. Database Migration Script

**File**: `packages/morag/migrations/001_create_remote_jobs_table.sql`

```sql
-- Migration: Create remote_jobs table
-- Version: 001
-- Description: Add support for remote conversion job tracking

BEGIN;

-- Create remote_jobs table
CREATE TABLE IF NOT EXISTS remote_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ingestion_task_id VARCHAR(255) NOT NULL,
    source_file_path TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    task_options JSONB NOT NULL DEFAULT '{}',
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    worker_id VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    result_data JSONB,
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    timeout_at TIMESTAMP WITH TIME ZONE,
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('pending', 'processing', 'completed', 'failed', 'timeout', 'cancelled')),
    CONSTRAINT valid_retry_count CHECK (retry_count >= 0),
    CONSTRAINT valid_max_retries CHECK (max_retries >= 0),
    CONSTRAINT valid_content_type CHECK (content_type IN ('audio', 'video', 'document', 'image', 'web', 'youtube'))
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_remote_jobs_status ON remote_jobs(status);
CREATE INDEX IF NOT EXISTS idx_remote_jobs_content_type ON remote_jobs(content_type);
CREATE INDEX IF NOT EXISTS idx_remote_jobs_created_at ON remote_jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_remote_jobs_ingestion_task ON remote_jobs(ingestion_task_id);
CREATE INDEX IF NOT EXISTS idx_remote_jobs_worker_id ON remote_jobs(worker_id);
CREATE INDEX IF NOT EXISTS idx_remote_jobs_timeout ON remote_jobs(timeout_at) WHERE timeout_at IS NOT NULL;

-- Composite indexes for common queries
CREATE INDEX IF NOT EXISTS idx_remote_jobs_status_content_type ON remote_jobs(status, content_type);
CREATE INDEX IF NOT EXISTS idx_remote_jobs_pending_timeout ON remote_jobs(status, timeout_at) WHERE status IN ('pending', 'processing');

-- Create migration tracking table if it doesn't exist
CREATE TABLE IF NOT EXISTS schema_migrations (
    version VARCHAR(10) PRIMARY KEY,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    description TEXT
);

-- Record this migration
INSERT INTO schema_migrations (version, description) 
VALUES ('001', 'Create remote_jobs table for remote conversion tracking')
ON CONFLICT (version) DO NOTHING;

COMMIT;
```

### 2. Database Configuration and Connection

**File**: `packages/morag/src/morag/database/__init__.py`

```python
from .connection import DatabaseManager, get_db_session
from .migrations import MigrationManager

__all__ = ['DatabaseManager', 'get_db_session', 'MigrationManager']
```

**File**: `packages/morag/src/morag/database/connection.py`

```python
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from contextlib import contextmanager
from typing import Generator
import structlog
import os

logger = structlog.get_logger(__name__)

# Database configuration
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://morag:morag@localhost:5432/morag')

# Create SQLAlchemy engine
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=os.getenv('SQL_DEBUG', 'false').lower() == 'true'
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create declarative base
Base = declarative_base()

class DatabaseManager:
    """Database connection and session management."""
    
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Database session error", error=str(e))
            raise
        finally:
            session.close()
    
    def create_tables(self):
        """Create all tables defined in models."""
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e))
            raise
    
    def drop_tables(self):
        """Drop all tables (use with caution!)."""
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e))
            raise
    
    def test_connection(self) -> bool:
        """Test database connectivity."""
        try:
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            return False

# Global database manager instance
db_manager = DatabaseManager()

def get_db_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
```

### 3. Migration Management System

**File**: `packages/morag/src/morag/database/migrations.py`

```python
import os
import glob
from pathlib import Path
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Tuple
import structlog

from .connection import db_manager

logger = structlog.get_logger(__name__)

class MigrationManager:
    """Database migration management."""
    
    def __init__(self, migrations_dir: str = None):
        if migrations_dir is None:
            # Default to migrations directory in package
            package_dir = Path(__file__).parent.parent.parent
            migrations_dir = package_dir / "migrations"
        
        self.migrations_dir = Path(migrations_dir)
        self.migrations_dir.mkdir(exist_ok=True)
    
    def get_applied_migrations(self, session: Session) -> List[str]:
        """Get list of applied migration versions."""
        try:
            result = session.execute(text("SELECT version FROM schema_migrations ORDER BY version"))
            return [row[0] for row in result]
        except Exception as e:
            logger.warning("Could not read migration history, assuming fresh database", error=str(e))
            return []
    
    def get_available_migrations(self) -> List[Tuple[str, Path]]:
        """Get list of available migration files."""
        migration_files = glob.glob(str(self.migrations_dir / "*.sql"))
        migrations = []
        
        for file_path in sorted(migration_files):
            file_name = Path(file_path).name
            # Extract version from filename (e.g., "001_create_table.sql" -> "001")
            version = file_name.split('_')[0]
            migrations.append((version, Path(file_path)))
        
        return migrations
    
    def apply_migration(self, session: Session, version: str, file_path: Path) -> bool:
        """Apply a single migration."""
        try:
            logger.info("Applying migration", version=version, file=str(file_path))
            
            with open(file_path, 'r') as f:
                migration_sql = f.read()
            
            # Execute migration SQL
            session.execute(text(migration_sql))
            session.commit()
            
            logger.info("Migration applied successfully", version=version)
            return True
            
        except Exception as e:
            logger.error("Failed to apply migration", version=version, error=str(e))
            session.rollback()
            return False
    
    def run_migrations(self) -> bool:
        """Run all pending migrations."""
        try:
            with db_manager.get_session() as session:
                applied = set(self.get_applied_migrations(session))
                available = self.get_available_migrations()
                
                pending = [(v, p) for v, p in available if v not in applied]
                
                if not pending:
                    logger.info("No pending migrations")
                    return True
                
                logger.info("Running migrations", pending_count=len(pending))
                
                for version, file_path in pending:
                    if not self.apply_migration(session, version, file_path):
                        logger.error("Migration failed, stopping", version=version)
                        return False
                
                logger.info("All migrations completed successfully")
                return True
                
        except Exception as e:
            logger.error("Migration process failed", error=str(e))
            return False
    
    def create_migration(self, description: str) -> Path:
        """Create a new migration file template."""
        # Get next version number
        available = self.get_available_migrations()
        if available:
            last_version = max(int(v) for v, _ in available)
            next_version = f"{last_version + 1:03d}"
        else:
            next_version = "001"
        
        # Create filename
        safe_description = description.lower().replace(' ', '_').replace('-', '_')
        filename = f"{next_version}_{safe_description}.sql"
        file_path = self.migrations_dir / filename
        
        # Create template content
        template = f"""-- Migration: {description}
-- Version: {next_version}
-- Description: {description}

BEGIN;

-- Add your migration SQL here
-- Example:
-- CREATE TABLE example (
--     id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
--     name VARCHAR(255) NOT NULL
-- );

-- Record this migration
INSERT INTO schema_migrations (version, description) 
VALUES ('{next_version}', '{description}')
ON CONFLICT (version) DO NOTHING;

COMMIT;
"""
        
        with open(file_path, 'w') as f:
            f.write(template)
        
        logger.info("Migration template created", file=str(file_path))
        return file_path
```

### 4. Database Initialization Script

**File**: `packages/morag/src/morag/database/init.py`

```python
#!/usr/bin/env python3
"""Database initialization script for MoRAG remote conversion system."""

import sys
import argparse
from pathlib import Path
import structlog

from .connection import db_manager
from .migrations import MigrationManager

logger = structlog.get_logger(__name__)

def init_database(run_migrations: bool = True) -> bool:
    """Initialize the database with required tables and data."""
    try:
        logger.info("Initializing MoRAG database")
        
        # Test database connection
        if not db_manager.test_connection():
            logger.error("Database connection failed")
            return False
        
        # Run migrations if requested
        if run_migrations:
            migration_manager = MigrationManager()
            if not migration_manager.run_migrations():
                logger.error("Database migrations failed")
                return False
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error("Database initialization failed", error=str(e))
        return False

def reset_database() -> bool:
    """Reset the database by dropping and recreating all tables."""
    try:
        logger.warning("Resetting database - all data will be lost!")
        
        # Drop all tables
        db_manager.drop_tables()
        
        # Recreate tables
        db_manager.create_tables()
        
        # Run migrations
        migration_manager = MigrationManager()
        if not migration_manager.run_migrations():
            logger.error("Failed to run migrations after reset")
            return False
        
        logger.info("Database reset completed successfully")
        return True
        
    except Exception as e:
        logger.error("Database reset failed", error=str(e))
        return False

def main():
    """Command-line interface for database management."""
    parser = argparse.ArgumentParser(description="MoRAG Database Management")
    parser.add_argument('command', choices=['init', 'migrate', 'reset', 'test'],
                       help='Database command to execute')
    parser.add_argument('--force', action='store_true',
                       help='Force operation without confirmation')
    
    args = parser.parse_args()
    
    if args.command == 'init':
        success = init_database()
    elif args.command == 'migrate':
        migration_manager = MigrationManager()
        success = migration_manager.run_migrations()
    elif args.command == 'reset':
        if not args.force:
            confirm = input("This will delete all data. Are you sure? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Operation cancelled")
                return
        success = reset_database()
    elif args.command == 'test':
        success = db_manager.test_connection()
    else:
        parser.print_help()
        return
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
```

### 5. Database Models Integration

**File**: `packages/morag-core/src/morag_core/models/remote_job.py`

```python
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

from morag.database.connection import Base

class RemoteJob(Base):
    """Remote conversion job model."""
    
    __tablename__ = 'remote_jobs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ingestion_task_id = Column(String(255), nullable=False, index=True)
    source_file_path = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False, index=True)
    task_options = Column(JSONB, nullable=False, default={})
    status = Column(String(20), nullable=False, default='pending', index=True)
    worker_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    result_data = Column(JSONB, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    timeout_at = Column(DateTime, nullable=True)
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            'job_id': str(self.id),
            'ingestion_task_id': self.ingestion_task_id,
            'source_file_path': self.source_file_path,
            'content_type': self.content_type,
            'task_options': self.task_options,
            'status': self.status,
            'worker_id': self.worker_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'timeout_at': self.timeout_at.isoformat() if self.timeout_at else None
        }
    
    @property
    def is_expired(self) -> bool:
        """Check if job has expired."""
        if not self.timeout_at:
            return False
        return datetime.utcnow() > self.timeout_at
    
    @property
    def processing_duration(self) -> float:
        """Get processing duration in seconds."""
        if not self.started_at:
            return 0.0
        
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
    
    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries and self.status in ['failed', 'timeout']
```

## Implementation Steps

1. **Create Migration Directory Structure** (Day 1)
   - Set up migrations directory
   - Create initial migration script
   - Test migration on development database

2. **Implement Database Connection** (Day 1)
   - Create connection management utilities
   - Add session handling for FastAPI
   - Test database connectivity

3. **Create Migration System** (Day 2)
   - Implement migration manager
   - Add migration tracking table
   - Create CLI for database management

4. **Integrate with Models** (Day 2)
   - Update RemoteJob model to use Base
   - Add model relationships if needed
   - Test model operations

5. **Database Initialization** (Day 3)
   - Create initialization scripts
   - Add database reset functionality
   - Create development setup guide

6. **Testing and Validation** (Day 3)
   - Test all database operations
   - Validate migration system
   - Performance testing with indexes

## Testing Requirements

### Database Tests

**File**: `tests/test_database_schema.py`

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta

from morag.database.connection import Base
from morag_core.models.remote_job import RemoteJob
from morag.database.migrations import MigrationManager

class TestDatabaseSchema:
    def test_remote_job_creation(self):
        # Test job creation and constraints
        pass
    
    def test_database_indexes(self):
        # Test index performance
        pass
    
    def test_migration_system(self):
        # Test migration application
        pass
    
    def test_job_lifecycle(self):
        # Test complete job lifecycle
        pass
```

## Success Criteria

1. Database migrations run successfully on fresh database
2. All indexes are created and improve query performance
3. RemoteJob model operations work correctly
4. Migration system tracks applied migrations
5. Database initialization script works for new deployments

## Dependencies

- PostgreSQL database server
- SQLAlchemy ORM
- Existing MoRAG database configuration
- Migration tracking system

## Next Steps

After completing this task:
1. Proceed to [Task 3: Worker Modifications](./03-worker-modifications.md)
2. Test database schema with sample data
3. Set up database in development environment
4. Begin integration with API endpoints
