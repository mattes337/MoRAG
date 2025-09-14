# MoRAG Deployment Guide

This guide covers deploying MoRAG in production environments using Docker, Docker Compose, and Kubernetes.

## Quick Start (Docker Compose)

### Prerequisites
- Docker and Docker Compose installed
- At least 4GB RAM and 2 CPU cores
- Gemini API key

### 1. Configuration
```bash
# Copy and configure environment
cp .env.prod.example .env.prod
# Edit .env.prod with your actual values, especially GEMINI_API_KEY
```

### 2. Deploy
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Deploy the stack
./scripts/deploy.sh
```

### 3. Verify
```bash
# Check service status
./scripts/monitor.sh

# Test API
curl http://localhost/health/
curl http://localhost/api/v1/status/stats/queues
```

## Architecture

The production deployment includes:

- **API Server**: FastAPI application with monitoring middleware
- **Workers**: Celery workers for different task types
  - Document processing workers (2 concurrent)
  - Media processing workers (1 concurrent) 
  - Web processing workers (2 concurrent)
- **Redis**: Task queue and caching
- **Qdrant**: Vector database for embeddings
- **Nginx**: Reverse proxy with rate limiting and security headers
- **Flower**: Celery monitoring dashboard

## Services

### API (Port 8000)
- Main FastAPI application
- Health checks at `/health/`
- API documentation at `/docs`
- Metrics at `/health/metrics`

### Workers
- **Document Workers**: Process PDFs, text files, documents
- **Media Workers**: Process audio, video, images
- **Web Workers**: Process URLs, web scraping

### Monitoring
- **Flower**: Celery monitoring at `http://localhost:5555`
- **Metrics**: Prometheus-compatible metrics at `/health/metrics`
- **Logs**: Structured JSON logs in `./logs/`

## Configuration

### Environment Variables

Key production settings in `.env.prod`:

```env
# Required
GEMINI_API_KEY=your_production_gemini_api_key

# Database
QDRANT_HOST=qdrant
QDRANT_PORT=6333
REDIS_URL=redis://redis:6379/0

# Performance
API_WORKERS=4
MAX_CONCURRENT_TASKS=20
SLOW_QUERY_THRESHOLD=1.0

# Security
ALLOWED_ORIGINS=["https://yourdomain.com"]
LOG_LEVEL=INFO

# Monitoring
METRICS_ENABLED=true
MEMORY_THRESHOLD=80
CPU_THRESHOLD=80
```

### Resource Requirements

**Minimum:**
- 4GB RAM
- 2 CPU cores
- 20GB disk space

**Recommended:**
- 8GB RAM
- 4 CPU cores
- 100GB disk space

## Security

### Network Security
- Nginx reverse proxy with rate limiting
- Security headers (X-Frame-Options, X-XSS-Protection, etc.)
- File upload size limits (2GB)
- Request timeouts

### Rate Limiting
- API endpoints: 10 requests/second
- File uploads: 2 requests/second
- Burst allowance with nodelay

### Data Security
- Non-root container users
- Read-only configuration mounts
- Isolated container networks

## Monitoring

### Health Checks
```bash
# API health
curl http://localhost/health/

# Detailed health with dependencies
curl http://localhost/health/ready

# System metrics
curl http://localhost/health/metrics

# Queue statistics
curl http://localhost/api/v1/status/stats/queues
```

### Logs
```bash
# View all logs
docker-compose -f docker-compose.prod.yml logs -f

# View specific service
docker-compose -f docker-compose.prod.yml logs -f api

# Monitor system resources
./scripts/monitor.sh
```

### Flower Dashboard
Access Celery monitoring at `http://localhost:5555`

## Backup and Recovery

### Create Backup
```bash
./scripts/backup.sh
```

This creates a timestamped backup in `./backups/` containing:
- Qdrant vector database
- Redis data
- Uploaded files
- Configuration files

### Restore from Backup
```bash
# Stop services
docker-compose -f docker-compose.prod.yml down

# Restore data (example for backup from 2023-12-01 12:00:00)
BACKUP_DIR="./backups/20231201_120000"

# Restore uploads
tar xzf "$BACKUP_DIR/uploads.tar.gz"

# Restore configuration
cp "$BACKUP_DIR/.env.prod" .

# Start services
docker-compose -f docker-compose.prod.yml up -d

# Restore database data (requires manual steps)
# See backup.sh for detailed restore procedures
```

## Scaling

### Horizontal Scaling
```bash
# Scale API servers
docker-compose -f docker-compose.prod.yml up -d --scale api=3

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale worker-documents=4
```

### Vertical Scaling
Edit `docker-compose.prod.yml` to adjust:
- Memory limits
- CPU limits
- Worker concurrency

## Kubernetes Deployment

### Prerequisites
- Kubernetes cluster
- kubectl configured
- Persistent volume support

### Deploy
```bash
# Create namespace
kubectl create namespace morag

# Apply configurations
kubectl apply -f k8s/ -n morag

# Check status
kubectl get pods -n morag
kubectl get services -n morag
```

### Configuration
Update `k8s/deployment.yaml`:
- Set correct image tags
- Configure resource limits
- Update secrets with actual API keys

## Troubleshooting

### Common Issues

**Services won't start:**
```bash
# Check logs
docker-compose -f docker-compose.prod.yml logs

# Check service health
docker-compose -f docker-compose.prod.yml ps
```

**High memory usage:**
```bash
# Check resource usage
docker stats

# Adjust worker concurrency in docker-compose.prod.yml
```

**Database connection issues:**
```bash
# Test Qdrant connection
curl http://localhost:6333/health

# Test Redis connection
docker-compose -f docker-compose.prod.yml exec redis redis-cli ping
```

**Worker tasks failing:**
```bash
# Check worker logs
docker-compose -f docker-compose.prod.yml logs worker-documents

# Check Flower dashboard
open http://localhost:5555
```

### Performance Tuning

**API Performance:**
- Increase `API_WORKERS` in .env.prod
- Scale API containers horizontally
- Tune `SLOW_QUERY_THRESHOLD`

**Worker Performance:**
- Adjust worker concurrency per queue type
- Scale worker containers
- Monitor queue lengths

**Database Performance:**
- Monitor Qdrant memory usage
- Consider Qdrant clustering for large datasets
- Tune Redis memory policies

## Maintenance

### Updates
```bash
# Pull latest images
docker-compose -f docker-compose.prod.yml pull

# Restart with new images
docker-compose -f docker-compose.prod.yml up -d
```

### Log Rotation
Logs are automatically rotated based on:
- Size: 100MB max per file
- Count: 5 backup files
- Time: Daily rotation

### Database Maintenance
```bash
# Qdrant collection info
curl http://localhost:6333/collections/morag_documents

# Redis memory usage
docker-compose -f docker-compose.prod.yml exec redis redis-cli info memory
```

## Production Checklist

- [ ] Environment variables configured
- [ ] SSL certificates installed (if using HTTPS)
- [ ] Firewall rules configured
- [ ] Monitoring alerts set up
- [ ] Backup procedures tested
- [ ] Log aggregation configured
- [ ] Resource limits appropriate
- [ ] Security headers verified
- [ ] Rate limiting tested
- [ ] Health checks working
- [ ] Documentation updated

## Support

For deployment issues:
1. Check logs: `./scripts/monitor.sh`
2. Verify configuration: `.env.prod`
3. Test connectivity: health check endpoints
4. Review resource usage: `docker stats`
5. Check task queue status: Flower dashboard
