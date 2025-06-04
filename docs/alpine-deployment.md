# Alpine Linux Docker Deployment Guide

This guide covers deploying MoRAG using the optimized Alpine Linux Docker container, designed for production environments with external Qdrant servers.

## Overview

The Alpine Linux container provides:
- **Lightweight**: ~200MB smaller than Ubuntu-based containers
- **CPU-Optimized**: Configured for CPU-only processing with automatic GPU fallback
- **External Qdrant**: Designed to work with existing Qdrant servers
- **Production Ready**: Includes health checks, logging, and monitoring
- **Conservative Resources**: Optimized for minimal resource usage

## Prerequisites

### Required Services
- **External Qdrant Server**: Running on a separate machine/container
- **Docker & Docker Compose**: For container orchestration
- **Gemini API Key**: For embeddings and LLM operations

### System Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 10GB for container + space for uploads/temp files
- **Network**: Access to external Qdrant server

## Quick Start

### 1. Clone Repository
```bash
git clone <repository-url>
cd morag
```

### 2. Configure Environment
```bash
# Copy Alpine environment template
cp .env.alpine .env

# Edit with your configuration
nano .env
```

### 3. Update Required Settings
```bash
# Required settings in .env
GEMINI_API_KEY=your_gemini_api_key_here
QDRANT_HOST=your_qdrant_server_ip
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=morag_documents

# Optional if Qdrant requires authentication
QDRANT_API_KEY=your_qdrant_api_key_if_needed
```

### 4. Start Services
```bash
# Build and start all services
docker-compose -f docker-compose.alpine.yml up --build

# Or run in background
docker-compose -f docker-compose.alpine.yml up --build -d
```

### 5. Validate Deployment
```bash
# Test container functionality
docker exec -it morag-api-alpine bash scripts/validate_alpine_deployment.sh

# Test API endpoints
curl http://localhost:8000/health/
curl http://localhost:8000/docs
```

## Configuration Details

### Environment Variables

#### Required Configuration
```bash
# Gemini API (required for embeddings)
GEMINI_API_KEY=your_gemini_api_key_here

# External Qdrant server (required)
QDRANT_HOST=192.168.1.100  # Your Qdrant server IP
QDRANT_PORT=6333           # Qdrant port (usually 6333)
QDRANT_COLLECTION_NAME=morag_documents
```

#### Optional Configuration
```bash
# Qdrant authentication (if required)
QDRANT_API_KEY=your_qdrant_api_key

# Redis (handled by Docker Compose)
REDIS_URL=redis://redis:6379/0

# Alpine-specific optimizations
PREFERRED_DEVICE=cpu
FORCE_CPU=true
ENABLE_DYNAMIC_WEB_SCRAPING=false
WEB_SCRAPING_FALLBACK_ONLY=true

# Resource limits
MAX_CONCURRENT_TASKS=2
CELERY_WORKER_CONCURRENCY=1
MAX_FILE_SIZE=50MB
WHISPER_MODEL_SIZE=base
```

### Docker Compose Services

The Alpine deployment includes:

1. **morag-api**: Main API service
2. **morag-worker**: Celery worker for background tasks
3. **redis**: Task queue (local to container stack)

## Service Management

### Starting Services
```bash
# Start all services
docker-compose -f docker-compose.alpine.yml up

# Start in background
docker-compose -f docker-compose.alpine.yml up -d

# Start specific service
docker-compose -f docker-compose.alpine.yml up morag-api
```

### Stopping Services
```bash
# Stop all services
docker-compose -f docker-compose.alpine.yml down

# Stop and remove volumes
docker-compose -f docker-compose.alpine.yml down -v
```

### Viewing Logs
```bash
# View all logs
docker-compose -f docker-compose.alpine.yml logs

# View specific service logs
docker-compose -f docker-compose.alpine.yml logs morag-api
docker-compose -f docker-compose.alpine.yml logs morag-worker

# Follow logs in real-time
docker-compose -f docker-compose.alpine.yml logs -f
```

## Monitoring and Health Checks

### Health Endpoints
- **API Health**: http://localhost:8000/health/
- **API Documentation**: http://localhost:8000/docs
- **Redis Health**: Automatic health checks in Docker Compose

### Container Health Checks
```bash
# Check container status
docker-compose -f docker-compose.alpine.yml ps

# Check individual container health
docker inspect morag-api-alpine --format='{{.State.Health.Status}}'
```

### Log Monitoring
```bash
# Monitor API logs
docker exec -it morag-api-alpine tail -f logs/morag.log

# Monitor worker logs
docker-compose -f docker-compose.alpine.yml logs -f morag-worker
```

## Troubleshooting

### Common Issues

#### 1. Qdrant Connection Failed
```bash
# Check if Qdrant server is accessible
curl http://your_qdrant_server_ip:6333/health

# Test from container
docker exec -it morag-api-alpine curl http://your_qdrant_server_ip:6333/health
```

#### 2. API Not Starting
```bash
# Check container logs
docker-compose -f docker-compose.alpine.yml logs morag-api

# Run validation script
docker exec -it morag-api-alpine bash scripts/validate_alpine_deployment.sh
```

#### 3. Worker Not Processing Tasks
```bash
# Check worker logs
docker-compose -f docker-compose.alpine.yml logs morag-worker

# Check Redis connection
docker exec -it morag-api-alpine redis-cli -h redis ping
```

#### 4. Import Errors
```bash
# Test MoRAG import
docker exec -it morag-api-alpine python3 -c "import morag; print('Success')"

# Run comprehensive tests
docker exec -it morag-api-alpine python3 scripts/test_alpine_container.py
```

### Performance Optimization

#### Resource Limits
```bash
# Adjust in .env file
MAX_CONCURRENT_TASKS=4      # Increase for more CPU cores
CELERY_WORKER_CONCURRENCY=2 # Increase for more workers
MAX_FILE_SIZE=100MB         # Increase for larger files
WHISPER_MODEL_SIZE=large    # Use larger model for better quality
```

#### Memory Usage
```bash
# Monitor container memory usage
docker stats morag-api-alpine morag-worker-alpine

# Adjust Docker memory limits in docker-compose.alpine.yml
```

## Production Considerations

### Security
- Use environment files with restricted permissions (600)
- Consider using Docker secrets for sensitive data
- Ensure Qdrant server has proper authentication
- Use HTTPS in production with reverse proxy

### Backup
- Backup your `.env` configuration
- Ensure Qdrant data is backed up separately
- Consider backing up uploaded files volume

### Scaling
- Scale worker containers: `docker-compose -f docker-compose.alpine.yml up --scale morag-worker=3`
- Use load balancer for multiple API instances
- Monitor resource usage and adjust limits

### Updates
```bash
# Update to latest version
git pull
docker-compose -f docker-compose.alpine.yml build --no-cache
docker-compose -f docker-compose.alpine.yml up -d
```

## Support

For issues specific to Alpine deployment:
1. Run the validation script: `scripts/validate_alpine_deployment.sh`
2. Check container logs for error messages
3. Verify external Qdrant server connectivity
4. Ensure all required environment variables are set

For general MoRAG issues, see the main [README.md](../README.md) troubleshooting section.
