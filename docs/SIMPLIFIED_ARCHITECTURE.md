# Simplified MoRAG Architecture - No Redis Required

MoRAG has been simplified to eliminate Redis dependency and complex infrastructure requirements.

## What Was Removed

### Redis/Celery Infrastructure
- ❌ **Redis server requirement** - No longer needed for task queues
- ❌ **Celery workers** - Replaced with HTTP-based workers
- ❌ **Celery beat scheduler** - Server handles scheduling directly
- ❌ **Complex queue management** - No more user-specific Redis queues
- ❌ **Redis-based API key storage** - Now uses in-memory storage

### Removed Files
- `scripts/start-remote-worker.sh` - Old Redis-based worker script
- `scripts/start-remote-worker.bat` - Windows Redis worker script
- `scripts/start_worker.py` - Celery worker startup script
- `scripts/start_worker_safe.sh` - Safe Celery worker startup
- `configs/remote-worker.env` - Redis worker configuration
- `configs/gpu-worker.env.example` - Redis worker example config
- `Dockerfile.worker` (old) - Redis-based worker container
- `docker-compose.microservices.yml` - Redis-based microservices setup
- `docs/remote-workers-setup.md` - Redis worker setup guide
- `docs/REDIS_ALTERNATIVES.md` - Migration documentation
- `packages/morag/src/morag/services/user_task_router.py` - Redis task routing

### Simplified Services
- `packages/morag/src/morag/services/auth_service.py` - Now uses in-memory storage
- `packages/morag/src/morag/worker.py` - Now imports HTTP worker functions

## What's Available Now

### HTTP Remote Workers
- ✅ **Direct HTTP communication** - Workers poll server via HTTP API
- ✅ **No infrastructure dependencies** - Just run Python script or Docker
- ✅ **Simple deployment** - Single command to start workers
- ✅ **Easy debugging** - Direct HTTP requests are easy to troubleshoot
- ✅ **Flexible scaling** - Add/remove workers without configuration

### New Files
- `scripts/start_http_remote_worker.py` - Main HTTP worker implementation
- `scripts/start-http-worker.sh` - Linux/macOS startup script
- `scripts/start-http-worker.bat` - Windows startup script
- `configs/http-worker.env.example` - HTTP worker configuration template
- `Dockerfile.worker` - HTTP worker container
- `docker-compose.workers.yml` - HTTP worker deployment
- `docs/HTTP_REMOTE_WORKERS.md` - Complete HTTP worker guide
- `examples/http_remote_worker_demo.py` - Usage examples
- `packages/morag/src/morag/http_worker.py` - HTTP worker core functions

## Architecture Comparison

### Before (Redis/Celery)
```
[Main Server] <-> [Redis] <-> [Celery Workers]
     |                            |
[Qdrant DB]                  [GPU Processing]
```

**Complexity:**
- Redis server deployment and maintenance
- Celery worker configuration and monitoring
- Queue management and routing
- Network configuration for Redis access
- Complex debugging across multiple services

### After (HTTP Workers)
```
[Main Server] <-> [HTTP Workers]
     |                 |
[Qdrant DB]       [GPU Processing]
```

**Simplicity:**
- Direct HTTP communication
- Single Python script or Docker container
- No queue management needed
- Simple network configuration (HTTP only)
- Easy debugging with standard HTTP tools

## Quick Start

### 1. Start Main Server
```bash
# No Redis needed!
uvicorn morag.api.main:app --reload
```

### 2. Start HTTP Workers
```bash
# Python script
python scripts/start_http_remote_worker.py \
    --server-url http://localhost:8000 \
    --api-key your-api-key

# Docker
docker-compose -f docker-compose.workers.yml up

# Script with environment file
./scripts/start-http-worker.sh
```

### 3. Process Content
```bash
# Same API, simpler infrastructure
curl -X POST "http://localhost:8000/process/file" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@audio.mp3" \
  -F "gpu=true"
```

## Benefits

### For Developers
- **Faster setup** - No Redis installation or configuration
- **Easier debugging** - HTTP requests are visible and testable
- **Simpler testing** - No infrastructure dependencies in tests
- **Better isolation** - Workers are completely independent

### For Operations
- **Reduced infrastructure** - One less service to deploy and maintain
- **Simpler monitoring** - Standard HTTP health checks
- **Easier scaling** - Add workers without queue configuration
- **Lower complexity** - Fewer moving parts to manage

### For Users
- **Same functionality** - All processing capabilities remain
- **Better reliability** - Fewer points of failure
- **Easier deployment** - Single script or container
- **Flexible configuration** - Environment-based setup

## Migration Notes

If you were using the old Redis-based workers:

1. **Stop old workers** - No longer needed
2. **Remove Redis** - Can be uninstalled if only used for MoRAG
3. **Update scripts** - Use new HTTP worker scripts
4. **Update configs** - Use new environment file format
5. **Test functionality** - Same API, different backend

The HTTP workers provide the same processing capabilities with significantly reduced complexity.

## Performance

HTTP workers have minimal performance impact:

- **Polling latency**: 3-10 seconds (configurable)
- **Processing speed**: Same as Redis workers (GPU acceleration unchanged)
- **Throughput**: Comparable for most workloads
- **Resource usage**: Lower overall (no Redis overhead)

For most use cases, the slight polling latency is negligible compared to the processing time for audio/video content.

## Future Enhancements

Possible improvements while maintaining simplicity:

- **WebSocket support** for real-time task assignment
- **Built-in load balancing** across multiple workers
- **Worker pools** for different task types
- **Automatic scaling** based on queue length

These would be additive features that maintain the core simplicity of the HTTP-based approach.
