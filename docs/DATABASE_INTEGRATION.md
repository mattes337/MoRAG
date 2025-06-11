# MoRAG Database Integration

This document describes the SQL Alchemy database integration for MoRAG, providing comprehensive user management, multi-tenancy, and advanced features.

## Overview

The database integration adds relational data management to MoRAG's existing vector database capabilities, enabling:

- **User Management**: Registration, authentication, and role-based access control
- **Multi-tenancy**: Complete data isolation between users
- **Database Management**: Multiple vector database connections per user
- **Document Lifecycle**: Complete document state tracking and version control
- **Job Tracking**: Comprehensive processing job monitoring
- **API Key Management**: Programmatic access with usage tracking

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │     Celery      │    │     Qdrant      │
│   REST API      │────│   Task Queue    │────│  Vector Store   │
│  + Auth Layer   │    │  + Job Tracking │    │  (Per Database) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  SQL Database   │
                    │   (Primary)     │
                    │ ┌─────────────┐ │
                    │ │   Users     │ │
                    │ │ Databases   │ │
                    │ │ Documents   │ │
                    │ │   Jobs      │ │
                    │ │ API Keys    │ │
                    │ └─────────────┘ │
                    └─────────────────┘
```

## Database Schema

### Core Tables

1. **users** - User accounts with roles and authentication
2. **user_settings** - User preferences and configuration
3. **database_servers** - Vector database connection configurations
4. **databases** - Logical databases within servers
5. **documents** - Document metadata and processing state
6. **api_keys** - API key management and usage tracking
7. **jobs** - Processing job status and progress tracking

### Key Relationships

- Users → User Settings (1:1)
- Users → Database Servers (1:N)
- Users → Databases (1:N)
- Users → Documents (1:N)
- Users → API Keys (1:N)
- Users → Jobs (1:N)
- Database Servers → Databases (1:N)
- Databases → Documents (1:N)
- Documents → Jobs (1:N)

## Configuration

### Environment Variables

```bash
# Database Configuration
MORAG_DATABASE_URL="sqlite:///./morag.db"  # Default: SQLite
MORAG_DATABASE_POOL_SIZE=5                 # Connection pool size
MORAG_DATABASE_MAX_OVERFLOW=10             # Max pool overflow
MORAG_DATABASE_ECHO=false                  # Enable SQL logging
```

### Database URLs

**SQLite (Development)**:
```bash
MORAG_DATABASE_URL="sqlite:///./morag.db"
```

**PostgreSQL (Production)**:
```bash
MORAG_DATABASE_URL="postgresql://user:password@localhost:5432/morag"
```

**MySQL/MariaDB**:
```bash
MORAG_DATABASE_URL="mysql+pymysql://user:password@localhost:3306/morag"
```

## Installation

### 1. Install Dependencies

**SQLite (included with Python)**:
```bash
pip install morag-core
```

**PostgreSQL**:
```bash
pip install morag-core[postgresql]
```

**MySQL/MariaDB**:
```bash
pip install morag-core[mysql]
```

**All Database Drivers**:
```bash
pip install morag-core[all-databases]
```

### 2. Initialize Database

```bash
# Initialize database with tables
python -m morag_core.database.cli init

# Initialize with admin user
python -m morag_core.database.cli init --create-admin --admin-email admin@yourcompany.com

# Initialize with development data
python -m morag_core.database.cli init --setup-dev-data
```

### 3. Verify Installation

```bash
# Check database status
python -m morag_core.database.cli check
```

## Usage

### Python API

```python
from morag_core.database import (
    DatabaseManager,
    DatabaseInitializer,
    get_database_manager,
    User,
    create_user,
    get_session_context
)

# Initialize database
initializer = DatabaseInitializer()
initializer.initialize_database()

# Create a user
with get_session_context() as session:
    user = create_user(
        session,
        name="John Doe",
        email="john@example.com",
        role=UserRole.USER
    )
    print(f"Created user: {user.id}")

# Get database manager
db_manager = get_database_manager()
print(f"Connection OK: {db_manager.test_connection()}")
```

### CLI Management

```bash
# Initialize database
python -m morag_core.database.cli init

# Check database status
python -m morag_core.database.cli check

# Create admin user
python -m morag_core.database.cli create-user "Admin User" admin@company.com --admin

# Reset database (WARNING: Deletes all data)
python -m morag_core.database.cli reset --confirm
```

## Database Models

### User Model

```python
class User(Base):
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    avatar = Column(String(500), nullable=True)
    role = Column(Enum(UserRole), default=UserRole.USER)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
```

### Document Model

```python
class Document(Base):
    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False)
    type = Column(String(50), nullable=False)
    state = Column(Enum(DocumentState), default=DocumentState.PENDING)
    version = Column(Integer, default=1)
    chunks = Column(Integer, default=0)
    quality = Column(Float, default=0.0)
    user_id = Column(String(36), ForeignKey('users.id'))
    database_id = Column(String(36), ForeignKey('databases.id'))
```

### Job Model

```python
class Job(Base):
    id = Column(String(36), primary_key=True)
    document_name = Column(String(255), nullable=False)
    document_type = Column(String(50), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    percentage = Column(Integer, default=0)
    summary = Column(Text, default="")
    user_id = Column(String(36), ForeignKey('users.id'))
    document_id = Column(String(36), ForeignKey('documents.id'))
```

## Migration Management

### Create Migration

```python
from morag_core.database.migrations import MigrationManager

migration_manager = MigrationManager()
migration_manager.init_migrations()

# Create auto-generated migration
revision = migration_manager.create_migration("Add new column")

# Create empty migration
revision = migration_manager.create_migration("Custom changes", auto_generate=False)
```

### Apply Migrations

```python
# Upgrade to latest
migration_manager.upgrade_database()

# Upgrade to specific revision
migration_manager.upgrade_database("abc123")

# Downgrade to previous revision
migration_manager.downgrade_database("-1")
```

### Check Migration Status

```python
status = migration_manager.check_migration_status()
print(f"Current: {status['current_revision']}")
print(f"Head: {status['head_revision']}")
print(f"Up to date: {status['up_to_date']}")
```

## Testing

### Run Database Tests

```bash
# Run all database tests
pytest tests/test_database_integration.py -v

# Run specific test class
pytest tests/test_database_integration.py::TestDatabaseManager -v

# Run with coverage
pytest tests/test_database_integration.py --cov=morag_core.database
```

### Test Database Setup

```python
import pytest
from morag_core.database import DatabaseManager

@pytest.fixture
def test_db():
    """Create test database."""
    manager = DatabaseManager("sqlite:///:memory:")
    manager.create_tables()
    yield manager

def test_user_creation(test_db):
    """Test user creation."""
    session = test_db.get_session()
    user = create_user(session, "Test", "test@example.com")
    assert user.id is not None
    session.close()
```

## Production Deployment

### PostgreSQL Setup

1. **Install PostgreSQL**:
```bash
# Ubuntu/Debian
sudo apt-get install postgresql postgresql-contrib

# CentOS/RHEL
sudo yum install postgresql postgresql-server
```

2. **Create Database**:
```sql
CREATE DATABASE morag;
CREATE USER morag_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE morag TO morag_user;
```

3. **Configure Environment**:
```bash
export MORAG_DATABASE_URL="postgresql://morag_user:secure_password@localhost:5432/morag"
```

### Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: morag
      POSTGRES_USER: morag_user
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  morag:
    build: .
    environment:
      MORAG_DATABASE_URL: postgresql://morag_user:secure_password@postgres:5432/morag
    depends_on:
      - postgres

volumes:
  postgres_data:
```

## Security Considerations

### Password Hashing

User passwords are hashed using bcrypt:

```python
import bcrypt

# Hash password
password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

# Verify password
bcrypt.checkpw(password.encode('utf-8'), password_hash)
```

### API Key Management

API keys are securely generated and stored:

```python
import secrets

# Generate secure API key
api_key = secrets.token_urlsafe(32)

# Store with user association
create_api_key(session, user_id, "My API Key", api_key)
```

### SQL Injection Prevention

All database operations use SQLAlchemy ORM with parameterized queries, preventing SQL injection attacks.

## Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Verify database URL format
   - Check database server is running
   - Verify credentials and permissions

2. **Migration Errors**:
   - Check Alembic configuration
   - Verify database schema state
   - Review migration history

3. **Permission Errors**:
   - Verify database user permissions
   - Check file system permissions for SQLite

### Debug Mode

Enable SQL query logging:

```bash
export MORAG_DATABASE_ECHO=true
```

### Health Checks

```python
from morag_core.database import get_database_manager

db_manager = get_database_manager()
print(f"Connection: {db_manager.test_connection()}")
print(f"Schema: {db_manager.verify_schema()}")
print(f"Tables: {db_manager.get_table_names()}")
```
