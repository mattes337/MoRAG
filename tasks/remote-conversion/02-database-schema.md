# Task 2: File-Based Storage for Remote Job Tracking

## Overview

Implement a file-based storage system for remote job tracking using the repository pattern. This approach allows the system to work without a database while providing an abstraction layer that makes it easy to migrate to a database later when one is added to MoRAG.

## Objectives

1. Create file-based storage structure for remote jobs
2. Implement repository pattern for data access abstraction
3. Add proper file organization and indexing for performance
4. Create storage initialization and cleanup utilities
5. Ensure easy migration path to database when available

## Technical Requirements

### 1. File Storage Structure

**Directory Structure**: `packages/morag/src/morag/storage/remote_jobs_storage.py`

```python
import os
import json
import glob
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import structlog
import fcntl  # For file locking on Unix systems
import time

logger = structlog.get_logger(__name__)

class RemoteJobsStorage:
    """File-based storage for remote jobs with atomic operations."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or os.getenv('MORAG_REMOTE_JOBS_DATA_DIR', '/app/data/remote_jobs'))
        self.lock_dir = self.data_dir / '.locks'
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        status_dirs = ['pending', 'processing', 'completed', 'failed', 'timeout', 'cancelled']

        # Create main data directory
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.lock_dir.mkdir(parents=True, exist_ok=True)

        # Create status directories
        for status in status_dirs:
            (self.data_dir / status).mkdir(exist_ok=True)

        # Create index files for fast lookups
        self._ensure_index_files()

    def _ensure_index_files(self):
        """Create index files for fast lookups."""
        index_files = [
            'content_type_index.json',  # Maps content_type -> [job_ids]
            'worker_index.json',        # Maps worker_id -> [job_ids]
            'ingestion_task_index.json' # Maps ingestion_task_id -> job_id
        ]

        for index_file in index_files:
            index_path = self.data_dir / index_file
            if not index_path.exists():
                with open(index_path, 'w') as f:
                    json.dump({}, f)

    def _get_lock_file(self, job_id: str) -> Path:
        """Get lock file path for a job."""
        return self.lock_dir / f"{job_id}.lock"

    def _acquire_lock(self, job_id: str, timeout: float = 5.0) -> Optional[object]:
        """Acquire a file lock for atomic operations."""
        lock_file = self._get_lock_file(job_id)

        try:
            # Create lock file if it doesn't exist
            lock_file.touch()

            # Open and lock the file
            lock_fd = open(lock_file, 'w')

            # Try to acquire lock with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock_fd
                except BlockingIOError:
                    time.sleep(0.1)

            # Timeout reached
            lock_fd.close()
            return None

        except Exception as e:
            logger.error("Failed to acquire lock", job_id=job_id, error=str(e))
            return None

    def _release_lock(self, lock_fd):
        """Release a file lock."""
        try:
            if lock_fd:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
        except Exception as e:
            logger.error("Failed to release lock", error=str(e))
```

### 2. Storage Operations

**File**: `packages/morag/src/morag/storage/remote_jobs_storage.py` (continued)

```python
    def _get_job_file_path(self, job_id: str, status: str) -> Path:
        """Get the file path for a job based on its status."""
        return self.data_dir / status / f"{job_id}.json"

    def _find_job_file(self, job_id: str) -> Optional[Path]:
        """Find the job file across all status directories."""
        for status_dir in self.data_dir.iterdir():
            if status_dir.is_dir() and not status_dir.name.startswith('.'):
                job_file = status_dir / f"{job_id}.json"
                if job_file.exists():
                    return job_file
        return None

    def _update_indexes(self, job_data: Dict[str, Any], operation: str = 'add'):
        """Update index files for fast lookups."""
        try:
            job_id = job_data['id']
            content_type = job_data['content_type']
            worker_id = job_data.get('worker_id')
            ingestion_task_id = job_data['ingestion_task_id']

            # Update content type index
            content_index_path = self.data_dir / 'content_type_index.json'
            with open(content_index_path, 'r+') as f:
                content_index = json.load(f)

                if operation == 'add':
                    if content_type not in content_index:
                        content_index[content_type] = []
                    if job_id not in content_index[content_type]:
                        content_index[content_type].append(job_id)
                elif operation == 'remove':
                    if content_type in content_index and job_id in content_index[content_type]:
                        content_index[content_type].remove(job_id)

                f.seek(0)
                json.dump(content_index, f, indent=2)
                f.truncate()

            # Update worker index
            if worker_id:
                worker_index_path = self.data_dir / 'worker_index.json'
                with open(worker_index_path, 'r+') as f:
                    worker_index = json.load(f)

                    if operation == 'add':
                        if worker_id not in worker_index:
                            worker_index[worker_id] = []
                        if job_id not in worker_index[worker_id]:
                            worker_index[worker_id].append(job_id)
                    elif operation == 'remove':
                        if worker_id in worker_index and job_id in worker_index[worker_id]:
                            worker_index[worker_id].remove(job_id)

                    f.seek(0)
                    json.dump(worker_index, f, indent=2)
                    f.truncate()

            # Update ingestion task index
            ingestion_index_path = self.data_dir / 'ingestion_task_index.json'
            with open(ingestion_index_path, 'r+') as f:
                ingestion_index = json.load(f)

                if operation == 'add':
                    ingestion_index[ingestion_task_id] = job_id
                elif operation == 'remove':
                    ingestion_index.pop(ingestion_task_id, None)

                f.seek(0)
                json.dump(ingestion_index, f, indent=2)
                f.truncate()

        except Exception as e:
            logger.error("Failed to update indexes", job_id=job_data.get('id'), error=str(e))

    def save_job(self, job_data: Dict[str, Any]) -> bool:
        """Save a job to storage with atomic operations."""
        job_id = job_data['id']
        status = job_data['status']

        # Acquire lock for atomic operation
        lock_fd = self._acquire_lock(job_id)
        if not lock_fd:
            logger.error("Failed to acquire lock for job save", job_id=job_id)
            return False

        try:
            # Remove old file if it exists in a different status directory
            old_file = self._find_job_file(job_id)

            # Write new file
            new_file = self._get_job_file_path(job_id, status)
            with open(new_file, 'w') as f:
                json.dump(job_data, f, indent=2)

            # Remove old file if it's in a different directory
            if old_file and old_file != new_file:
                old_file.unlink()

            # Update indexes
            self._update_indexes(job_data, 'add')

            logger.debug("Job saved", job_id=job_id, status=status)
            return True

        except Exception as e:
            logger.error("Failed to save job", job_id=job_id, error=str(e))
            return False
        finally:
            self._release_lock(lock_fd)

    def load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load a job from storage."""
        try:
            job_file = self._find_job_file(job_id)
            if not job_file:
                return None

            with open(job_file, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error("Failed to load job", job_id=job_id, error=str(e))
            return None

    def delete_job(self, job_id: str) -> bool:
        """Delete a job from storage."""
        # Acquire lock for atomic operation
        lock_fd = self._acquire_lock(job_id)
        if not lock_fd:
            logger.error("Failed to acquire lock for job deletion", job_id=job_id)
            return False

        try:
            job_file = self._find_job_file(job_id)
            if not job_file:
                return False

            # Load job data for index cleanup
            with open(job_file, 'r') as f:
                job_data = json.load(f)

            # Delete file
            job_file.unlink()

            # Update indexes
            self._update_indexes(job_data, 'remove')

            logger.info("Job deleted", job_id=job_id)
            return True

        except Exception as e:
            logger.error("Failed to delete job", job_id=job_id, error=str(e))
            return False
        finally:
            self._release_lock(lock_fd)
```

### 3. Query Operations

**File**: `packages/morag/src/morag/storage/remote_jobs_storage.py` (continued)

```python
    def find_jobs_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Find all jobs with a specific status."""
        try:
            jobs = []
            status_dir = self.data_dir / status

            if not status_dir.exists():
                return jobs

            for job_file in status_dir.glob('*.json'):
                try:
                    with open(job_file, 'r') as f:
                        jobs.append(json.load(f))
                except Exception as e:
                    logger.error("Failed to load job file", file=str(job_file), error=str(e))

            # Sort by creation time
            jobs.sort(key=lambda x: x.get('created_at', ''))
            return jobs

        except Exception as e:
            logger.error("Failed to find jobs by status", status=status, error=str(e))
            return []

    def find_jobs_by_content_type(self, content_type: str) -> List[Dict[str, Any]]:
        """Find all jobs with a specific content type using index."""
        try:
            jobs = []

            # Use content type index for fast lookup
            content_index_path = self.data_dir / 'content_type_index.json'
            with open(content_index_path, 'r') as f:
                content_index = json.load(f)

            job_ids = content_index.get(content_type, [])

            for job_id in job_ids:
                job_data = self.load_job(job_id)
                if job_data:
                    jobs.append(job_data)

            return jobs

        except Exception as e:
            logger.error("Failed to find jobs by content type", content_type=content_type, error=str(e))
            return []

    def find_jobs_by_worker(self, worker_id: str) -> List[Dict[str, Any]]:
        """Find all jobs assigned to a specific worker using index."""
        try:
            jobs = []

            # Use worker index for fast lookup
            worker_index_path = self.data_dir / 'worker_index.json'
            with open(worker_index_path, 'r') as f:
                worker_index = json.load(f)

            job_ids = worker_index.get(worker_id, [])

            for job_id in job_ids:
                job_data = self.load_job(job_id)
                if job_data:
                    jobs.append(job_data)

            return jobs

        except Exception as e:
            logger.error("Failed to find jobs by worker", worker_id=worker_id, error=str(e))
            return []

    def find_job_by_ingestion_task(self, ingestion_task_id: str) -> Optional[Dict[str, Any]]:
        """Find job by ingestion task ID using index."""
        try:
            # Use ingestion task index for fast lookup
            ingestion_index_path = self.data_dir / 'ingestion_task_index.json'
            with open(ingestion_index_path, 'r') as f:
                ingestion_index = json.load(f)

            job_id = ingestion_index.get(ingestion_task_id)
            if job_id:
                return self.load_job(job_id)

            return None

        except Exception as e:
            logger.error("Failed to find job by ingestion task",
                        ingestion_task_id=ingestion_task_id, error=str(e))
            return None

    def get_expired_jobs(self) -> List[Dict[str, Any]]:
        """Get all expired jobs from pending and processing directories."""
        try:
            expired_jobs = []
            now = datetime.utcnow().isoformat()

            # Check pending and processing directories
            for status in ['pending', 'processing']:
                status_dir = self.data_dir / status
                if not status_dir.exists():
                    continue

                for job_file in status_dir.glob('*.json'):
                    try:
                        with open(job_file, 'r') as f:
                            job_data = json.load(f)

                        timeout_at = job_data.get('timeout_at')
                        if timeout_at and timeout_at < now:
                            expired_jobs.append(job_data)

                    except Exception as e:
                        logger.error("Failed to check job expiration",
                                   file=str(job_file), error=str(e))

            return expired_jobs

        except Exception as e:
            logger.error("Failed to get expired jobs", error=str(e))
            return []

    def get_old_jobs(self, days_old: int = 7) -> List[Dict[str, Any]]:
        """Get old completed/failed jobs for cleanup."""
        try:
            old_jobs = []
            cutoff_time = datetime.utcnow() - timedelta(days=days_old)

            # Check completed, failed, timeout, and cancelled directories
            for status in ['completed', 'failed', 'timeout', 'cancelled']:
                status_dir = self.data_dir / status
                if not status_dir.exists():
                    continue

                for job_file in status_dir.glob('*.json'):
                    try:
                        # Check file modification time
                        if datetime.fromtimestamp(job_file.stat().st_mtime) < cutoff_time:
                            with open(job_file, 'r') as f:
                                old_jobs.append(json.load(f))

                    except Exception as e:
                        logger.error("Failed to check job age",
                                   file=str(job_file), error=str(e))

            return old_jobs

        except Exception as e:
            logger.error("Failed to get old jobs", error=str(e))
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            stats = {
                'total_jobs': 0,
                'status_counts': {},
                'content_type_counts': {},
                'storage_size_mb': 0
            }

            # Count jobs by status
            for status_dir in self.data_dir.iterdir():
                if status_dir.is_dir() and not status_dir.name.startswith('.'):
                    job_count = len(list(status_dir.glob('*.json')))
                    stats['status_counts'][status_dir.name] = job_count
                    stats['total_jobs'] += job_count

            # Get content type counts from index
            content_index_path = self.data_dir / 'content_type_index.json'
            if content_index_path.exists():
                with open(content_index_path, 'r') as f:
                    content_index = json.load(f)
                    stats['content_type_counts'] = {
                        ct: len(job_ids) for ct, job_ids in content_index.items()
                    }

            # Calculate storage size
            total_size = 0
            for file_path in self.data_dir.rglob('*.json'):
                total_size += file_path.stat().st_size
            stats['storage_size_mb'] = round(total_size / (1024 * 1024), 2)

            return stats

        except Exception as e:
            logger.error("Failed to get storage statistics", error=str(e))
            return {}
```

### 4. Storage Manager

**File**: `packages/morag/src/morag/storage/__init__.py`

```python
from .remote_jobs_storage import RemoteJobsStorage

__all__ = ['RemoteJobsStorage']
```

**File**: `packages/morag/src/morag/storage/storage_manager.py`

```python
import structlog
from typing import Optional
import os

from .remote_jobs_storage import RemoteJobsStorage

logger = structlog.get_logger(__name__)

class StorageManager:
    """Central storage manager for all MoRAG storage needs."""

    def __init__(self):
        self._remote_jobs_storage: Optional[RemoteJobsStorage] = None

    @property
    def remote_jobs(self) -> RemoteJobsStorage:
        """Get remote jobs storage instance."""
        if self._remote_jobs_storage is None:
            self._remote_jobs_storage = RemoteJobsStorage()
        return self._remote_jobs_storage

    def initialize_storage(self) -> bool:
        """Initialize all storage systems."""
        try:
            # Initialize remote jobs storage
            self.remote_jobs._ensure_directories()

            logger.info("Storage systems initialized successfully")
            return True

        except Exception as e:
            logger.error("Failed to initialize storage systems", error=str(e))
            return False

    def cleanup_storage(self, days_old: int = 7) -> Dict[str, int]:
        """Clean up old data across all storage systems."""
        try:
            results = {}

            # Clean up old remote jobs
            old_jobs = self.remote_jobs.get_old_jobs(days_old)
            cleaned_count = 0

            for job_data in old_jobs:
                if self.remote_jobs.delete_job(job_data['id']):
                    cleaned_count += 1

            results['remote_jobs_cleaned'] = cleaned_count

            logger.info("Storage cleanup completed", results=results)
            return results

        except Exception as e:
            logger.error("Failed to cleanup storage", error=str(e))
            return {}

    def get_storage_statistics(self) -> Dict[str, Any]:
        """Get statistics for all storage systems."""
        try:
            return {
                'remote_jobs': self.remote_jobs.get_statistics(),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error("Failed to get storage statistics", error=str(e))
            return {}

# Global storage manager instance
storage_manager = StorageManager()
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

1. **Create Storage Structure** (Day 1)
   - Set up file-based storage directories
   - Create storage classes with atomic operations
   - Test basic file operations

2. **Implement Repository Pattern** (Day 1)
   - Create repository interface
   - Implement file-based repository
   - Add proper error handling

3. **Add Indexing System** (Day 2)
   - Implement index files for fast lookups
   - Add index maintenance operations
   - Test query performance

4. **Integrate with Models** (Day 2)
   - Update RemoteJob model for file storage
   - Add serialization/deserialization
   - Test model operations

5. **Storage Management** (Day 3)
   - Create storage manager
   - Add initialization and cleanup utilities
   - Create development setup guide

6. **Testing and Validation** (Day 3)
   - Test all storage operations
   - Validate atomic operations and locking
   - Performance testing with large datasets

## Testing Requirements

### Storage Tests

**File**: `tests/test_file_storage.py`

```python
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

from morag.storage.remote_jobs_storage import RemoteJobsStorage
from morag_core.models.remote_job import RemoteJob

class TestFileStorage:
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = RemoteJobsStorage(temp_dir)
        yield storage
        shutil.rmtree(temp_dir)

    def test_job_creation_and_retrieval(self, temp_storage):
        # Test job creation and retrieval
        pass

    def test_status_transitions(self, temp_storage):
        # Test job status transitions and file moves
        pass

    def test_indexing_system(self, temp_storage):
        # Test index file maintenance
        pass

    def test_atomic_operations(self, temp_storage):
        # Test file locking and atomic operations
        pass

    def test_query_performance(self, temp_storage):
        # Test query performance with indexes
        pass

    def test_cleanup_operations(self, temp_storage):
        # Test cleanup of old and expired jobs
        pass
```

## Success Criteria

1. File storage operations work reliably with atomic guarantees
2. Index files improve query performance significantly
3. RemoteJob model serialization/deserialization works correctly
4. Storage manager handles initialization and cleanup properly
5. File-based storage provides easy migration path to database

## Dependencies

- File system with proper permissions
- Python file locking capabilities
- JSON serialization support
- Existing MoRAG configuration system

## Next Steps

After completing this task:
1. Proceed to [Task 3: Worker Modifications](./03-worker-modifications.md)
2. Test storage system with sample data
3. Set up storage directories in development environment
4. Begin integration with API endpoints

## Migration Path to Database

When MoRAG adds database support, the migration will be straightforward:

1. **Repository Interface**: The repository pattern provides a clean interface that can be implemented for database storage
2. **Data Migration**: JSON files can be easily imported into database tables
3. **Index Migration**: File-based indexes can be replaced with database indexes
4. **Atomic Operations**: Database transactions replace file locking
5. **Configuration**: Simple environment variable change to switch storage backends
