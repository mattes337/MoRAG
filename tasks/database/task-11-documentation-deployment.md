# Task 11: Documentation and Deployment

## 📋 Task Overview

**Objective**: Create comprehensive documentation for the database integration and prepare deployment guides, configuration templates, and operational procedures for production deployment.

**Priority**: Critical - Required for production deployment
**Estimated Time**: 1 week
**Dependencies**: Task 10 (Testing and Validation)

## 🎯 Goals

1. Create comprehensive API documentation
2. Write deployment and configuration guides
3. Create database administration documentation
4. Add troubleshooting and monitoring guides
5. Create migration and upgrade procedures
6. Write security and backup documentation
7. Prepare production deployment templates

## 📊 Documentation Structure

### Documentation Categories
1. **API Documentation**: Complete API reference with examples
2. **Deployment Guides**: Step-by-step deployment instructions
3. **Configuration Reference**: All configuration options explained
4. **Administration Guide**: Database management procedures
5. **Security Guide**: Security best practices and procedures
6. **Troubleshooting**: Common issues and solutions
7. **Deployment Guide**: Fresh deployment procedures

## 🔧 Implementation Plan

### Step 1: Create API Documentation

**Files to Create**:
```
docs/
├── api/
│   ├── authentication.md
│   ├── users.md
│   ├── documents.md
│   ├── jobs.md
│   ├── database_servers.md
│   ├── api_keys.md
│   └── examples/
│       ├── curl_examples.md
│       ├── python_examples.md
│       └── postman_collection.json
├── deployment/
│   ├── docker_deployment.md
│   ├── kubernetes_deployment.md
│   ├── production_setup.md
│   └── environment_variables.md
├── administration/
│   ├── database_management.md
│   ├── user_management.md
│   ├── monitoring.md
│   └── backup_restore.md
├── security/
│   ├── authentication_security.md
│   ├── api_key_management.md
│   ├── data_encryption.md
│   └── access_control.md
└── troubleshooting/
    ├── common_issues.md
    ├── performance_tuning.md
    └── debugging.md
```

**Implementation Details**:

1. **API Authentication Documentation**:
```markdown
# docs/api/authentication.md

# Authentication API

MoRAG supports two authentication methods:
1. **JWT Token Authentication** - For user sessions
2. **API Key Authentication** - For programmatic access

## JWT Authentication

### Register User

**Endpoint:** `POST /auth/register`

**Request Body:**
```json
{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "secure_password_123"
}
```

**Response:**
```json
{
  "id": "user-123",
  "name": "John Doe",
  "email": "john@example.com",
  "role": "USER",
  "created_at": "2025-01-15T10:30:00Z",
  "updated_at": "2025-01-15T10:30:00Z"
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe",
    "email": "john@example.com",
    "password": "secure_password_123"
  }'
```

### Login User

**Endpoint:** `POST /auth/login`

**Request Body:**
```json
{
  "email": "john@example.com",
  "password": "secure_password_123"
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400,
  "user": {
    "id": "user-123",
    "name": "John Doe",
    "email": "john@example.com",
    "role": "USER"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "john@example.com",
    "password": "secure_password_123"
  }'
```

### Using JWT Token

Include the JWT token in the Authorization header:

```bash
curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
```

## API Key Authentication

### Create API Key

**Endpoint:** `POST /api/v1/api-keys`

**Headers:**
```
Authorization: Bearer <jwt_token>
```

**Request Body:**
```json
{
  "name": "My API Key",
  "description": "API key for automation scripts",
  "permissions": ["READ", "WRITE"],
  "expires_at": "2025-12-31T23:59:59Z",
  "rate_limit_per_hour": 1000
}
```

**Response:**
```json
{
  "api_key": {
    "id": "key-123",
    "name": "My API Key",
    "key_prefix": "morag_abc123",
    "permissions": ["READ", "WRITE"],
    "status": "ACTIVE",
    "expires_at": "2025-12-31T23:59:59Z",
    "rate_limit_per_hour": 1000,
    "created_at": "2025-01-15T10:30:00Z"
  },
  "secret_key": "morag_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
}
```

### Using API Key

Include the API key in the X-API-Key header:

```bash
curl -X GET "http://localhost:8000/api/v1/documents" \
  -H "X-API-Key: morag_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
```

Or in the Authorization header:

```bash
curl -X GET "http://localhost:8000/api/v1/documents" \
  -H "Authorization: Bearer morag_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
```

## Error Responses

### Authentication Errors

**401 Unauthorized:**
```json
{
  "detail": "Authentication required"
}
```

**403 Forbidden:**
```json
{
  "detail": "Insufficient permissions"
}
```

**422 Validation Error:**
```json
{
  "detail": [
    {
      "loc": ["body", "email"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
```

2. **Deployment Documentation**:
```markdown
# docs/deployment/docker_deployment.md

# Docker Deployment Guide

This guide covers deploying MoRAG with the database integration using Docker and Docker Compose.

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- PostgreSQL 13+ (or MySQL 8.0+)
- Redis 6.0+
- Qdrant 1.0+

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/morag.git
cd morag
```

### 2. Environment Configuration

Create `.env` file:

```bash
# Database Configuration
MORAG_DATABASE_URL=postgresql://morag:password@postgres:5432/morag
MORAG_DATABASE_POOL_SIZE=10
MORAG_DATABASE_MAX_OVERFLOW=20

# Authentication
MORAG_JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
MORAG_JWT_ALGORITHM=HS256
MORAG_JWT_EXPIRATION_HOURS=24

# Vector Database
MORAG_QDRANT_HOST=qdrant
MORAG_QDRANT_PORT=6333
MORAG_QDRANT_API_KEY=your-qdrant-api-key

# Redis
MORAG_REDIS_URL=redis://redis:6379/0

# File Storage
MORAG_TEMP_DIR=/app/temp
MORAG_MAX_UPLOAD_SIZE_BYTES=104857600

# Processing
MORAG_ENABLE_GPU=false
MORAG_CELERY_SOFT_TIME_LIMIT=3600
MORAG_CELERY_TIME_LIMIT=3900

# API Configuration
MORAG_ENABLE_USER_REGISTRATION=true
MORAG_DEFAULT_USER_ROLE=USER
```

### 3. Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: morag
      POSTGRES_USER: morag
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U morag"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  qdrant:
    image: qdrant/qdrant:v1.7.0
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__HTTP_PORT: 6333
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  morag-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MORAG_DATABASE_URL=postgresql://morag:password@postgres:5432/morag
      - MORAG_REDIS_URL=redis://redis:6379/0
      - MORAG_QDRANT_HOST=qdrant
      - MORAG_QDRANT_PORT=6333
    volumes:
      - ./temp:/app/temp
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    command: >
      sh -c "
        python -m morag_core.migrations.cli upgrade &&
        uvicorn morag.server:app --host 0.0.0.0 --port 8000
      "

  morag-worker:
    build:
      context: .
      dockerfile: Dockerfile
    environment:
      - MORAG_DATABASE_URL=postgresql://morag:password@postgres:5432/morag
      - MORAG_REDIS_URL=redis://redis:6379/0
      - MORAG_QDRANT_HOST=qdrant
      - MORAG_QDRANT_PORT=6333
    volumes:
      - ./temp:/app/temp
    depends_on:
      - postgres
      - redis
      - qdrant
    command: >
      celery -A morag.celery_app worker --loglevel=info --concurrency=4

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
```

### 4. Deploy

```bash
# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f morag-api
```

### 5. Initialize Database

```bash
# Initialize fresh database
docker-compose exec morag-api python -m morag_core.database.cli init

# Create admin user (optional)
docker-compose exec morag-api python -c "
from morag_core.auth.service import UserService
from morag_core.auth.models import UserCreate
service = UserService()
admin = UserCreate(name='Admin', email='admin@example.com', password='admin123')
user = service.create_user(admin)
print(f'Admin user created: {user.email}')
"
```

## Production Considerations

### Security

1. **Change default passwords** in production
2. **Use strong JWT secret keys**
3. **Enable HTTPS** with proper SSL certificates
4. **Configure firewall rules** to restrict access
5. **Use secrets management** for sensitive data

### Performance

1. **Scale workers** based on load:
   ```bash
   docker-compose up --scale morag-worker=4
   ```

2. **Configure database connection pooling**:
   ```env
   MORAG_DATABASE_POOL_SIZE=20
   MORAG_DATABASE_MAX_OVERFLOW=40
   ```

3. **Enable Redis persistence**:
   ```yaml
   redis:
     command: redis-server --appendonly yes
   ```

### Monitoring

1. **Add health checks** to all services
2. **Configure log aggregation** (ELK stack, Fluentd)
3. **Set up monitoring** (Prometheus, Grafana)
4. **Configure alerting** for critical issues

### Backup

1. **Database backups**:
   ```bash
   docker-compose exec postgres pg_dump -U morag morag > backup.sql
   ```

2. **Vector database backups**:
   ```bash
   docker-compose exec qdrant tar -czf /tmp/qdrant-backup.tar.gz /qdrant/storage
   ```

3. **Automated backup scripts** with retention policies
```

3. **Configuration Reference**:
```markdown
# docs/deployment/environment_variables.md

# Environment Variables Reference

This document lists all environment variables used by MoRAG with database integration.

## Database Configuration

### MORAG_DATABASE_URL
- **Description**: Database connection URL
- **Required**: Yes
- **Default**: `sqlite:///./morag.db`
- **Examples**:
  - PostgreSQL: `postgresql://user:password@host:5432/database`
  - MySQL: `mysql+pymysql://user:password@host:3306/database`
  - SQLite: `sqlite:///./morag.db`

### MORAG_DATABASE_POOL_SIZE
- **Description**: Database connection pool size
- **Required**: No
- **Default**: `5`
- **Range**: 1-50
- **Production**: 10-20

### MORAG_DATABASE_MAX_OVERFLOW
- **Description**: Maximum overflow connections
- **Required**: No
- **Default**: `10`
- **Range**: 0-100
- **Production**: 20-40

### MORAG_DATABASE_ECHO
- **Description**: Enable SQL query logging
- **Required**: No
- **Default**: `false`
- **Values**: `true`, `false`
- **Production**: `false`

## Authentication Configuration

### MORAG_JWT_SECRET_KEY
- **Description**: JWT token signing key
- **Required**: Yes
- **Default**: `your-secret-key-change-in-production`
- **Security**: Use strong, random key in production
- **Length**: Minimum 32 characters

### MORAG_JWT_ALGORITHM
- **Description**: JWT signing algorithm
- **Required**: No
- **Default**: `HS256`
- **Values**: `HS256`, `HS384`, `HS512`

### MORAG_JWT_EXPIRATION_HOURS
- **Description**: JWT token expiration time
- **Required**: No
- **Default**: `24`
- **Range**: 1-168 (1 week)
- **Security**: Shorter is more secure

### MORAG_ENABLE_USER_REGISTRATION
- **Description**: Allow new user registration
- **Required**: No
- **Default**: `true`
- **Values**: `true`, `false`
- **Production**: Consider `false` for private deployments

### MORAG_DEFAULT_USER_ROLE
- **Description**: Default role for new users
- **Required**: No
- **Default**: `USER`
- **Values**: `ADMIN`, `USER`, `VIEWER`

## Vector Database Configuration

### MORAG_QDRANT_HOST
- **Description**: Qdrant server hostname
- **Required**: Yes
- **Default**: `localhost`

### MORAG_QDRANT_PORT
- **Description**: Qdrant server port
- **Required**: No
- **Default**: `6333`

### MORAG_QDRANT_API_KEY
- **Description**: Qdrant API key for authentication
- **Required**: No
- **Default**: None
- **Security**: Required for production Qdrant instances

### MORAG_QDRANT_COLLECTION_NAME
- **Description**: Default collection name
- **Required**: No
- **Default**: `morag_documents`
- **Note**: User-specific collections will be created automatically

## Processing Configuration

### MORAG_TEMP_DIR
- **Description**: Temporary file storage directory
- **Required**: No
- **Default**: `./temp`
- **Production**: Use dedicated volume

### MORAG_MAX_UPLOAD_SIZE_BYTES
- **Description**: Maximum file upload size
- **Required**: No
- **Default**: `104857600` (100MB)
- **Range**: 1MB - 1GB

### MORAG_ENABLE_GPU
- **Description**: Enable GPU processing
- **Required**: No
- **Default**: `false`
- **Values**: `true`, `false`
- **Requirements**: CUDA-compatible GPU

### MORAG_CELERY_SOFT_TIME_LIMIT
- **Description**: Celery task soft time limit (seconds)
- **Required**: No
- **Default**: `3600` (1 hour)
- **Range**: 300-7200

### MORAG_CELERY_TIME_LIMIT
- **Description**: Celery task hard time limit (seconds)
- **Required**: No
- **Default**: `3900` (65 minutes)
- **Note**: Should be higher than soft limit

## Redis Configuration

### MORAG_REDIS_URL
- **Description**: Redis connection URL
- **Required**: Yes
- **Default**: `redis://localhost:6379/0`
- **Examples**:
  - Local: `redis://localhost:6379/0`
  - With auth: `redis://:password@host:6379/0`
  - Cluster: `redis://host1:6379,host2:6379/0`

## API Configuration

### MORAG_API_HOST
- **Description**: API server bind address
- **Required**: No
- **Default**: `0.0.0.0`
- **Production**: `0.0.0.0` for containers

### MORAG_API_PORT
- **Description**: API server port
- **Required**: No
- **Default**: `8000`

### MORAG_API_WORKERS
- **Description**: Number of API worker processes
- **Required**: No
- **Default**: `1`
- **Production**: 2-4 per CPU core

## Logging Configuration

### MORAG_LOG_LEVEL
- **Description**: Logging level
- **Required**: No
- **Default**: `INFO`
- **Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

### MORAG_LOG_FORMAT
- **Description**: Log format
- **Required**: No
- **Default**: `json`
- **Values**: `json`, `text`

## Example Production Configuration

```bash
# Database
MORAG_DATABASE_URL=postgresql://morag:secure_password@db.example.com:5432/morag_prod
MORAG_DATABASE_POOL_SIZE=15
MORAG_DATABASE_MAX_OVERFLOW=30
MORAG_DATABASE_ECHO=false

# Authentication
MORAG_JWT_SECRET_KEY=super-secure-random-key-for-production-use-only
MORAG_JWT_ALGORITHM=HS256
MORAG_JWT_EXPIRATION_HOURS=8
MORAG_ENABLE_USER_REGISTRATION=false
MORAG_DEFAULT_USER_ROLE=USER

# Vector Database
MORAG_QDRANT_HOST=qdrant.example.com
MORAG_QDRANT_PORT=6333
MORAG_QDRANT_API_KEY=qdrant-production-api-key

# Processing
MORAG_TEMP_DIR=/app/temp
MORAG_MAX_UPLOAD_SIZE_BYTES=524288000
MORAG_ENABLE_GPU=true
MORAG_CELERY_SOFT_TIME_LIMIT=7200
MORAG_CELERY_TIME_LIMIT=7500

# Redis
MORAG_REDIS_URL=redis://:redis_password@redis.example.com:6379/0

# API
MORAG_API_HOST=0.0.0.0
MORAG_API_PORT=8000
MORAG_API_WORKERS=4

# Logging
MORAG_LOG_LEVEL=INFO
MORAG_LOG_FORMAT=json
```
```

## 📋 Acceptance Criteria

- [ ] Complete API documentation with examples
- [ ] Deployment guides for Docker and Kubernetes
- [ ] Configuration reference documentation
- [ ] Database administration guide
- [ ] Security best practices documented
- [ ] Troubleshooting guide with common issues
- [ ] Migration and upgrade procedures
- [ ] Production deployment templates
- [ ] Monitoring and backup procedures
- [ ] Performance tuning guidelines

## 🔄 Next Steps

After completing this task:
1. Review all documentation with stakeholders
2. Test deployment procedures in staging environment
3. Create training materials for operations team
4. Set up documentation website and maintenance procedures

## 📝 Notes

- Keep documentation up-to-date with code changes
- Include real-world examples and use cases
- Provide troubleshooting steps for common issues
- Create separate documentation for different audiences (developers, operators, users)
- Use clear, concise language and proper formatting
