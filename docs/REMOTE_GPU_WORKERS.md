# Remote GPU Workers

## Overview

The Remote GPU Workers feature enables MoRAG to route processing tasks to dedicated GPU workers based on API key authentication. This provides significant performance improvements for GPU-intensive tasks like audio/video processing while maintaining complete user isolation.

## Key Features

- **API Key Authentication**: Secure user identification and task routing
- **User-Specific Queues**: Each user gets dedicated GPU and CPU queues
- **Simple Integration**: Just add `gpu=true` parameter to existing API calls
- **Automatic Fallback**: Falls back to local processing if GPU workers unavailable
- **Backward Compatibility**: Existing API calls work unchanged
- **HTTP File Transfer**: Remote workers download files via authenticated HTTP endpoints

## Architecture

- **API Key Service**: Redis-based authentication with secure key generation
- **Authentication Middleware**: Optional API key validation for user identification
- **Queue Architecture**: User-specific queue naming (`gpu-tasks-{user_id}`)
- **API Integration**: GPU parameter added to processing endpoints
- **Remote Worker Tasks**: HTTP file transfer variants for remote processing
- **Management Endpoints**: API key creation, validation, and queue information

## API Changes

### New Parameters

All processing endpoints now accept an optional `gpu` parameter:

```bash
# File processing with GPU
curl -X POST http://localhost:8000/process/file \
  -H 'Authorization: Bearer <api_key>' \
  -F 'file=@document.pdf' \
  -F 'gpu=true'

# URL processing with GPU
curl -X POST http://localhost:8000/process/url \
  -H 'Authorization: Bearer <api_key>' \
  -H 'Content-Type: application/json' \
  -d '{"url": "https://example.com", "gpu": true}'
```

### New Endpoints

#### Create API Key
```bash
POST /api/v1/auth/create-key
Parameters:
  - user_id: unique user identifier
  - description: key description
  - expires_days: expiration in days
```

#### Validate API Key
```bash
POST /api/v1/auth/validate-key
Parameters:
  - api_key: key to validate
```

#### Queue Information
```bash
GET /api/v1/auth/queue-info
Headers:
  - Authorization: Bearer <api_key> (optional)
```

## Worker Deployment

### Local Workers (Unchanged)
```bash
celery -A morag.worker worker --loglevel=info --queues=celery
```

### User-Specific GPU Worker
```bash
celery -A morag.worker worker --loglevel=info --queues=gpu-tasks-user123
```

### Multi-User GPU Worker
```bash
celery -A morag.worker worker --loglevel=info --queues=gpu-tasks-user1,gpu-tasks-user2
```

## Task Routing Logic

| User Authentication | GPU Parameter | Result |
|---------------------|---------------|--------|
| None | false | Default queue (celery) |
| None | true | Default queue (celery) - fallback |
| Valid API key | false | Default queue (celery) |
| Valid API key | true | User GPU queue (gpu-tasks-{user_id}) |

## Queue Naming Convention

- **GPU Queue**: `gpu-tasks-{user_id}`
- **CPU Queue**: `cpu-tasks-{user_id}`
- **Default Queue**: `celery`

## Security Features

- **API Key Hashing**: Keys stored as SHA256 hashes in Redis
- **Expiration Support**: Configurable key expiration
- **User Isolation**: Tasks only processed by authenticated user's workers
- **Revocation**: API keys can be deactivated
- **Fallback Security**: Invalid authentication falls back to anonymous processing

## Performance Benefits

- **5-10x Faster**: GPU acceleration for audio/video processing
- **Dedicated Resources**: User-specific workers prevent resource contention
- **Scalable**: Add GPU workers on-demand
- **Efficient**: HTTP file transfer eliminates shared storage requirements

## Testing

### Standalone Tests
```bash
# Test API key service
python tests/test_auth_service_standalone.py

# Test API integration (requires running server)
python tests/test_api_integration.py

# View implementation demo
python tests/demo_simple_remote_gpu_workers.py
```

### Manual Testing
1. Start Redis: `docker-compose -f docker/docker-compose.redis.yml up -d`
2. Start MoRAG server: `python -m morag.server`
3. Create API key via `/api/v1/auth/create-key`
4. Test GPU processing with `gpu=true` parameter

## Configuration

### Environment Variables
- `REDIS_URL`: Redis connection string (default: `redis://localhost:6379/0`)

### Redis Storage
- API keys stored with prefix: `morag:api_keys:`
- Automatic expiration based on `expires_days`
- JSON metadata including user_id, description, timestamps

## Error Handling

- **Invalid API Key**: Falls back to anonymous processing
- **Missing GPU Worker**: Falls back to local processing
- **Network Issues**: Comprehensive retry mechanisms
- **File Transfer Errors**: Detailed error logging and recovery

## Production Deployment

For production deployment considerations:
- **Security Hardening**: Use strong API keys and secure Redis configuration
- **Monitoring**: Monitor queue lengths and worker performance
- **Scaling**: Add GPU workers based on demand
- **Network Security**: Use VPN or private networks for worker communication

## Support

For issues or questions about the Remote GPU Workers feature:
1. Check the [Remote Workers Setup Guide](remote-workers-setup.md) for detailed setup instructions
2. Review the API documentation at `/docs`
3. Examine the queue information endpoint for debugging
4. Check Redis for API key storage and validation
