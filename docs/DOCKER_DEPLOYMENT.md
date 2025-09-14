# MoRAG Docker Deployment Guide

This guide covers deploying MoRAG using Docker and Docker Compose in various configurations.

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- At least 4GB RAM
- 10GB free disk space

### Basic Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/morag.git
   cd morag
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start the services**:
   ```bash
   docker-compose up -d
   ```

4. **Check health**:
   ```bash
   curl http://localhost:8000/health
   ```

## Deployment Configurations

### 1. Monolithic Deployment (Recommended for Development)

Uses the main `docker-compose.yml` file with all services in a single container.

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f morag-api

# Stop services
docker-compose down
```

**Services included**:
- MoRAG API Server (port 8000)
- Celery Worker
- Celery Beat (scheduler)
- Redis (cache & message broker)
- Qdrant (vector database)
- Flower (task monitoring, port 5555)

### 2. Development Deployment

Uses `docker-compose.dev.yml` with hot-reload and development tools.

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up -d

# Access Jupyter notebook
open http://localhost:8888

# Run tests
docker-compose -f docker-compose.dev.yml exec morag-api-dev pytest
```

**Additional features**:
- Code hot-reload
- Jupyter notebook (port 8888)
- Development dependencies
- Debug logging

### 3. Microservices Deployment (Recommended for Production)

Uses `docker-compose.microservices.yml` with separate containers for each service.

```bash
# Start microservices
docker-compose -f docker-compose.microservices.yml up -d

# Scale specific services
docker-compose -f docker-compose.microservices.yml up -d --scale morag-audio=3
```

**Services**:
- API Gateway (port 8000)
- Audio Processing Service
- Video Processing Service
- Document Processing Service
- Web Processing Service
- Load Balancer (Nginx)

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for complete list):

```bash
# Required API Keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key  # Optional
ANTHROPIC_API_KEY=your_anthropic_api_key  # Optional

# Database URLs
REDIS_URL=redis://redis:6379/0
QDRANT_URL=http://qdrant:6333

# Application Settings
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false
```

### Volume Mounts

```yaml
volumes:
  - ./data:/app/data          # Persistent data
  - ./logs:/app/logs          # Application logs
  - ./temp:/app/temp          # Temporary files (shared between API and workers)
```

**Important**: The `/app/temp` volume is critical for proper operation as it allows file sharing between the API server and worker containers. All uploaded files are stored here temporarily during processing.

### Resource Limits

Add resource limits to `docker-compose.yml`:

```yaml
services:
  morag-api:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

## Monitoring and Observability

### Health Checks

All services include health checks:

```bash
# Check all service health
docker-compose ps

# Check specific service
docker-compose exec morag-api curl http://localhost:8000/health
```

### Logging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f morag-api

# View logs with timestamps
docker-compose logs -f -t morag-worker
```

### Monitoring Stack (Optional)

Enable monitoring with profiles:

```bash
# Start with monitoring
docker-compose --profile monitoring up -d

# Access Grafana
open http://localhost:3000  # admin/admin

# Access Prometheus
open http://localhost:9090
```

## Production Deployment

### SSL/TLS Configuration

1. **Generate certificates**:
   ```bash
   mkdir -p nginx/ssl
   # Add your SSL certificates to nginx/ssl/
   ```

2. **Configure Nginx**:
   ```bash
   cp nginx/nginx.conf.example nginx/nginx.conf
   # Edit nginx.conf with your domain and SSL settings
   ```

3. **Start with SSL**:
   ```bash
   docker-compose --profile production up -d
   ```

### Scaling

```bash
# Scale workers
docker-compose up -d --scale morag-worker=3

# Scale specific microservices
docker-compose -f docker-compose.microservices.yml up -d --scale morag-audio=2 --scale morag-video=2
```

### Backup and Recovery

```bash
# Backup volumes
docker run --rm -v morag_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup.tar.gz -C /data .
docker run --rm -v morag_qdrant_data:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v morag_redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis_backup.tar.gz -C /data
```

## Troubleshooting

### Common Issues

1. **Port conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Change ports in docker-compose.yml
   ports:
     - "8001:8000"  # Use different host port
   ```

2. **Memory issues**:
   ```bash
   # Increase Docker memory limit
   # Docker Desktop: Settings > Resources > Memory
   
   # Check container memory usage
   docker stats
   ```

3. **Permission issues**:
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./data ./logs ./temp
   ```

### Debug Mode

```bash
# Start in debug mode
ENVIRONMENT=development DEBUG=true docker-compose up

# Access container shell
docker-compose exec morag-api bash

# Run specific commands
docker-compose exec morag-api python -c "import morag; print('OK')"
```

### Performance Tuning

1. **Optimize worker concurrency**:
   ```bash
   # Set in .env
   CELERY_WORKER_CONCURRENCY=4
   ```

2. **Tune Redis memory**:
   ```bash
   # Add to docker-compose.yml
   command: redis-server --maxmemory 1gb --maxmemory-policy allkeys-lru
   ```

3. **Configure Qdrant**:
   ```bash
   # Add to docker-compose.yml
   environment:
     - QDRANT__SERVICE__MAX_REQUEST_SIZE_MB=32
   ```

## Security Considerations

1. **Use secrets for API keys**:
   ```yaml
   secrets:
     google_api_key:
       file: ./secrets/google_api_key.txt
   ```

2. **Network isolation**:
   ```yaml
   networks:
     frontend:
       driver: bridge
     backend:
       driver: bridge
       internal: true
   ```

3. **Regular updates**:
   ```bash
   # Update images
   docker-compose pull
   docker-compose up -d
   ```

## Next Steps

- [API Documentation](API_REFERENCE.md)
- [Development Guide](DEVELOPMENT_GUIDE.md)
- [Architecture Overview](ARCHITECTURE.md)
