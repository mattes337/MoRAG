# Task 22: Deployment Configuration

## Overview
Set up production-ready deployment configuration with Docker, environment management, and orchestration for the MoRAG pipeline.

## Prerequisites
- All core tasks completed (01-21)
- Docker installed
- Basic understanding of container orchestration

## Dependencies
- Task 01: Project Setup
- Task 02: API Framework
- Task 03: Database Setup
- Task 04: Task Queue Setup

## Implementation Steps

### 1. Production Dockerfile
Create `Dockerfile`:
```dockerfile
# Multi-stage build for production
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p uploads temp logs && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/ || exit 1

# Default command
CMD ["python", "-m", "uvicorn", "src.morag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Worker Dockerfile
Create `Dockerfile.worker`:
```dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv

# Copy requirements and install dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Create app user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create app directory
WORKDIR /app

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p uploads temp logs && \
    chown -R appuser:appuser /app

# Switch to app user
USER appuser

# Default command for worker
CMD ["celery", "-A", "src.morag.core.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
```

### 3. Production Docker Compose
Create `docker-compose.prod.yml`:
```yaml
version: '3.8'

services:
  # Redis for task queue
  redis:
    image: redis:7.2-alpine
    container_name: morag-redis-prod
    restart: unless-stopped
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - morag-network

  # Qdrant vector database
  qdrant:
    image: qdrant/qdrant:v1.7.4
    container_name: morag-qdrant-prod
    restart: unless-stopped
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - morag-network

  # Main API service
  api:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: morag-api-prod
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - API_WORKERS=4
      - LOG_LEVEL=INFO
    env_file:
      - .env.prod
    volumes:
      - ./uploads:/app/uploads
      - ./logs:/app/logs
    depends_on:
      redis:
        condition: service_healthy
      qdrant:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health/"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - morag-network

  # Celery workers
  worker-documents:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: morag-worker-docs-prod
    restart: unless-stopped
    environment:
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    env_file:
      - .env.prod
    volumes:
      - ./uploads:/app/uploads
      - ./temp:/app/temp
      - ./logs:/app/logs
    depends_on:
      - redis
      - qdrant
    command: ["celery", "-A", "src.morag.core.celery_app", "worker", "--loglevel=info", "--queues=document_processing", "--concurrency=2"]
    networks:
      - morag-network

  worker-media:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: morag-worker-media-prod
    restart: unless-stopped
    environment:
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    env_file:
      - .env.prod
    volumes:
      - ./uploads:/app/uploads
      - ./temp:/app/temp
      - ./logs:/app/logs
    depends_on:
      - redis
      - qdrant
    command: ["celery", "-A", "src.morag.core.celery_app", "worker", "--loglevel=info", "--queues=audio_processing,video_processing", "--concurrency=1"]
    networks:
      - morag-network

  worker-web:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: morag-worker-web-prod
    restart: unless-stopped
    environment:
      - REDIS_URL=redis://redis:6379/0
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
    env_file:
      - .env.prod
    volumes:
      - ./uploads:/app/uploads
      - ./temp:/app/temp
      - ./logs:/app/logs
    depends_on:
      - redis
      - qdrant
    command: ["celery", "-A", "src.morag.core.celery_app", "worker", "--loglevel=info", "--queues=web_processing", "--concurrency=2"]
    networks:
      - morag-network

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: morag-nginx-prod
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api
    networks:
      - morag-network

  # Monitoring with Flower (Celery monitoring)
  flower:
    build:
      context: .
      dockerfile: Dockerfile.worker
    container_name: morag-flower-prod
    restart: unless-stopped
    ports:
      - "5555:5555"
    environment:
      - REDIS_URL=redis://redis:6379/0
    env_file:
      - .env.prod
    depends_on:
      - redis
    command: ["celery", "-A", "src.morag.core.celery_app", "flower", "--port=5555"]
    networks:
      - morag-network

volumes:
  redis_data:
    driver: local
  qdrant_data:
    driver: local

networks:
  morag-network:
    driver: bridge
```

### 4. Nginx Configuration
Create `nginx/nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;

    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                    '$status $body_bytes_sent "$http_referer" '
                    '"$http_user_agent" "$http_x_forwarded_for"';

    access_log /var/log/nginx/access.log main;
    error_log /var/log/nginx/error.log warn;

    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript text/xml application/xml application/xml+rss text/javascript;

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # File upload size limit
        client_max_body_size 2G;
        client_body_timeout 300s;
        client_header_timeout 300s;

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts for long-running requests
            proxy_connect_timeout 300s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
        }

        # File upload endpoints (stricter rate limiting)
        location /api/v1/ingest/file {
            limit_req zone=upload_limit burst=5 nodelay;
            
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Extended timeouts for file uploads
            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
        }

        # Health checks (no rate limiting)
        location /health/ {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Documentation
        location /docs {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Default location
        location / {
            return 404;
        }
    }
}
```

### 5. Environment Configuration
Create `.env.prod.example`:
```env
# Production Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Gemini API (REQUIRED)
GEMINI_API_KEY=your_production_gemini_api_key
GEMINI_MODEL=gemini-pro
GEMINI_EMBEDDING_MODEL=text-embedding-004

# Database Configuration
QDRANT_HOST=qdrant
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents
QDRANT_API_KEY=your_qdrant_api_key_if_needed

# Redis Configuration
REDIS_URL=redis://redis:6379/0

# Task Queue
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# File Storage
UPLOAD_DIR=/app/uploads
TEMP_DIR=/app/temp
MAX_FILE_SIZE=2GB

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Security
API_KEY_HEADER=X-API-Key
ALLOWED_ORIGINS=["https://yourdomain.com"]

# Processing Limits
MAX_CHUNK_SIZE=1000
MAX_CONCURRENT_TASKS=20
WEBHOOK_TIMEOUT=30

# Production Settings
ENVIRONMENT=production
DEBUG=false
```

### 6. Deployment Scripts
Create `scripts/deploy.sh`:
```bash
#!/bin/bash
set -e

echo "🚀 Starting MoRAG deployment..."

# Check if .env.prod exists
if [ ! -f .env.prod ]; then
    echo "❌ .env.prod file not found. Please copy .env.prod.example and configure it."
    exit 1
fi

# Check if required environment variables are set
source .env.prod
if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ GEMINI_API_KEY not set in .env.prod"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads temp logs logs/nginx

# Build and start services
echo "🔨 Building and starting services..."
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose -f docker-compose.prod.yml ps

# Initialize database
echo "🗄️ Initializing database..."
docker-compose -f docker-compose.prod.yml exec api python scripts/init_db.py

# Run health checks
echo "🏥 Running health checks..."
curl -f http://localhost/health/ || echo "❌ API health check failed"
curl -f http://localhost/health/ready || echo "❌ API readiness check failed"

echo "✅ Deployment completed!"
echo "📊 Monitor services:"
echo "  - API: http://localhost"
echo "  - Flower (Celery): http://localhost:5555"
echo "  - Logs: docker-compose -f docker-compose.prod.yml logs -f"
```

Create `scripts/backup.sh`:
```bash
#!/bin/bash
set -e

BACKUP_DIR="./backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup in $BACKUP_DIR..."

# Backup Qdrant data
echo "🗄️ Backing up Qdrant data..."
docker-compose -f docker-compose.prod.yml exec qdrant tar czf - /qdrant/storage > "$BACKUP_DIR/qdrant_data.tar.gz"

# Backup Redis data
echo "💾 Backing up Redis data..."
docker-compose -f docker-compose.prod.yml exec redis redis-cli BGSAVE
sleep 5
docker-compose -f docker-compose.prod.yml exec redis tar czf - /data > "$BACKUP_DIR/redis_data.tar.gz"

# Backup uploaded files
echo "📁 Backing up uploaded files..."
tar czf "$BACKUP_DIR/uploads.tar.gz" uploads/

# Backup configuration
echo "⚙️ Backing up configuration..."
cp .env.prod "$BACKUP_DIR/"
cp docker-compose.prod.yml "$BACKUP_DIR/"

echo "✅ Backup completed: $BACKUP_DIR"
```

### 7. Monitoring and Logging
Create `scripts/monitor.sh`:
```bash
#!/bin/bash

echo "📊 MoRAG System Status"
echo "====================="

# Service status
echo "🔧 Service Status:"
docker-compose -f docker-compose.prod.yml ps

echo ""
echo "💾 Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

echo ""
echo "📈 Queue Status:"
curl -s http://localhost/api/v1/status/stats/queues | jq '.'

echo ""
echo "🏥 Health Status:"
curl -s http://localhost/health/ready | jq '.'

echo ""
echo "📝 Recent Logs (last 50 lines):"
docker-compose -f docker-compose.prod.yml logs --tail=50
```

### 8. Kubernetes Configuration (Optional)
Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: morag-api
  labels:
    app: morag-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: morag-api
  template:
    metadata:
      labels:
        app: morag-api
    spec:
      containers:
      - name: api
        image: morag:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379/0"
        - name: QDRANT_HOST
          value: "qdrant-service"
        - name: GEMINI_API_KEY
          valueFrom:
            secretKeyRef:
              name: morag-secrets
              key: gemini-api-key
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: morag-api-service
spec:
  selector:
    app: morag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

## Testing Instructions

### 1. Local Production Test
```bash
# Copy and configure environment
cp .env.prod.example .env.prod
# Edit .env.prod with your actual values

# Deploy locally
chmod +x scripts/deploy.sh
./scripts/deploy.sh

# Test API
curl http://localhost/health/
curl http://localhost/api/v1/status/stats/queues

# Monitor
chmod +x scripts/monitor.sh
./scripts/monitor.sh
```

### 2. Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test API endpoints
ab -n 100 -c 10 http://localhost/health/
ab -n 50 -c 5 -H "Authorization: Bearer test-key" http://localhost/api/v1/status/stats/queues
```

### 3. Backup and Restore
```bash
# Create backup
chmod +x scripts/backup.sh
./scripts/backup.sh

# Restore from backup (example)
# docker-compose -f docker-compose.prod.yml down
# tar xzf backups/20231201_120000/uploads.tar.gz
# docker-compose -f docker-compose.prod.yml up -d
```

## Success Criteria
- [ ] Docker images build successfully
- [ ] All services start and pass health checks
- [ ] API is accessible through Nginx
- [ ] Workers process tasks correctly
- [ ] Database data persists across restarts
- [ ] Monitoring tools work (Flower)
- [ ] Backup and restore procedures work
- [ ] Resource usage is reasonable
- [ ] Logs are properly collected
- [ ] Security headers are configured
- [ ] Rate limiting works
- [ ] SSL/TLS can be configured (if needed)

## Next Steps
- Configure SSL certificates for HTTPS
- Set up log aggregation (ELK stack, etc.)
- Configure monitoring and alerting
- Set up CI/CD pipeline for automated deployments
- Scale horizontally based on load
- Implement blue-green deployment strategy
