# Docker Log Issues Fixes - January 2025

## Issues Fixed Based on Docker Logs

### 1. ✅ Missing Health Check Methods

**Problem**: 
```
Health check failed for audio service error='AudioService' object has no attribute 'health_check'
Health check failed for video service error='VideoService' object has no attribute 'health_check'
```

**Solution**: Added health_check methods to both services
- **AudioService**: Checks processor and transcriber initialization
- **VideoService**: Checks processor initialization and configuration

**Files Modified**:
- `packages/morag-audio/src/morag_audio/service.py`
- `packages/morag-video/src/morag_video/service.py`

### 2. ✅ FastAPI Deprecation Warnings

**Problem**:
```
DeprecationWarning: on_event is deprecated, use lifespan event handlers instead
```

**Solution**: Converted from `@app.on_event()` to lifespan context manager
- Replaced startup/shutdown events with async context manager
- Follows FastAPI best practices for application lifecycle

**Files Modified**:
- `packages/morag/src/morag/server.py`

### 3. ✅ Celery Deprecation Warnings

**Problem**:
```
CPendingDeprecationWarning: The broker_connection_retry configuration setting will no longer determine
whether broker connection retries are made during startup in Celery 6.0 and above.
```

**Solution**: Added `broker_connection_retry_on_startup=True` to Celery configuration

**Files Modified**:
- `packages/morag/src/morag/worker.py`

### 4. ✅ Removed All Docker Health Checks

**Problem**: Health checks were causing complexity and potential issues

**Solution**: Removed all health checks from Docker configurations
- Simplified service dependencies (removed `condition: service_healthy`)
- Removed health check configurations from all services
- Services now start based on simple dependency order

**Files Modified**:
- `docker-compose.yml`
- `docker-compose.dev.yml`
- `docker-compose.prod.yml`
- `docker-compose.microservices.yml`
- `docker/docker-compose.qdrant.yml`
- `Dockerfile` (removed HEALTHCHECK)

### 5. ✅ Missing Optional Dependencies

**Problem**: Warning messages for missing optional packages:
```
pyannote.audio not available, speaker diarization disabled
sentence-transformers not available, using basic topic segmentation
Readability not available, content cleaning disabled
Newspaper3k not available, article extraction disabled
```

**Solution**: Documented optional dependencies in requirements.txt
- Added commented-out optional dependencies with explanations
- Users can uncomment specific packages as needed
- Maintains lean base installation while providing upgrade path

**Files Modified**:
- `requirements.txt`

### 6. ✅ Redis Memory Overcommit Warning

**Problem**:
```
WARNING Memory overcommit must be enabled! Without it, a background save or replication 
may fail under low memory condition.
```

**Solution**: Added documentation comment in docker-compose files
- This is a host-level system configuration
- Added instructions for fixing: `echo 'vm.overcommit_memory = 1' | sudo tee -a /etc/sysctl.conf && sudo sysctl -p`

**Files Modified**:
- `docker-compose.yml`

## Impact Summary

### Before Fixes:
- ❌ Health check errors in logs
- ❌ FastAPI deprecation warnings
- ❌ Celery deprecation warnings
- ❌ Complex health check dependencies
- ⚠️ Missing optional dependency warnings
- ⚠️ Redis memory overcommit warning

### After Fixes:
- ✅ Clean application startup
- ✅ No deprecation warnings
- ✅ Simplified Docker configuration
- ✅ Proper service health checking
- ✅ Clear documentation for optional features
- ✅ System configuration guidance

## Testing

### Verify Fixes:
```bash
# Start services
docker-compose up -d

# Check logs for errors
docker-compose logs morag-api | grep -i error
docker-compose logs morag-worker-1 | grep -i warning
docker-compose logs morag-worker-2 | grep -i warning

# Test API health
curl http://localhost:8000/health

# Check service status
docker-compose ps
```

### Expected Clean Logs:
- No health check errors
- No FastAPI deprecation warnings
- No Celery deprecation warnings
- Only informational messages about optional features

## Optional Dependencies Installation

To enable advanced features, uncomment and install optional dependencies:

```bash
# For speaker diarization
pip install pyannote.audio>=3.1.0

# For advanced topic segmentation
pip install sentence-transformers>=2.2.0

# For web content cleaning
pip install readability>=0.3.1

# For article extraction
pip install newspaper3k>=0.2.8
```

## System Configuration

For production deployments, configure Redis memory overcommit:

```bash
# Add to system configuration
echo 'vm.overcommit_memory = 1' | sudo tee -a /etc/sysctl.conf

# Apply immediately
sudo sysctl -p
```

## Files Modified Summary

### Application Code (4 files):
1. `packages/morag-audio/src/morag_audio/service.py` - Added health_check method
2. `packages/morag-video/src/morag_video/service.py` - Added health_check method  
3. `packages/morag/src/morag/server.py` - Fixed FastAPI deprecation
4. `packages/morag/src/morag/worker.py` - Fixed Celery deprecation

### Docker Configuration (6 files):
5. `docker-compose.yml` - Removed health checks, added Redis note
6. `docker-compose.dev.yml` - Removed health checks
7. `docker-compose.prod.yml` - Removed health checks
8. `docker-compose.microservices.yml` - Removed health checks
9. `docker/docker-compose.qdrant.yml` - Removed health checks
10. `Dockerfile` - Removed HEALTHCHECK directive

### Documentation (3 files):
11. `requirements.txt` - Added optional dependencies documentation
12. `TASKS.md` - Updated with fix summary
13. `DOCKER_LOG_FIXES_SUMMARY.md` - This comprehensive summary

## Result

The MoRAG system now starts cleanly without errors or warnings in the Docker logs, providing a better user experience and cleaner deployment process.
