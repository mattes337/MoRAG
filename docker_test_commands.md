# Docker Environment Testing Commands

This document provides commands to test that the Docker startup fix is working correctly.

## Quick Verification

### 1. Test Environment Variables in Container
```bash
# Test API container
docker-compose exec morag-api python -c "import os; print('QDRANT_COLLECTION_NAME:', os.getenv('QDRANT_COLLECTION_NAME'))"

# Test Worker container
docker-compose exec morag-worker-1 python -c "import os; print('QDRANT_COLLECTION_NAME:', os.getenv('QDRANT_COLLECTION_NAME'))"
```

### 2. Test Settings Import
```bash
# Test API container settings
docker-compose exec morag-api python -c "from morag_core.config import settings; print('Collection:', settings.qdrant_collection_name)"

# Test Worker container settings
docker-compose exec morag-worker-1 python -c "from morag_core.config import settings; print('Collection:', settings.qdrant_collection_name)"
```

### 3. Test Module Imports
```bash
# Test API server import
docker-compose exec morag-api python -c "import morag.server; print('Server imported successfully')"

# Test Worker import
docker-compose exec morag-worker-1 python -c "import morag.worker; print('Worker imported successfully')"
```

## Comprehensive Verification

### Run Full Verification Script
```bash
# Test API container
docker-compose exec morag-api python docker_verification_test.py

# Test Worker container
docker-compose exec morag-worker-1 python docker_verification_test.py
```

## Container Startup Testing

### 1. Check Container Logs
```bash
# Check API container logs
docker-compose logs morag-api

# Check Worker container logs
docker-compose logs morag-worker-1
docker-compose logs morag-worker-2
```

### 2. Restart Containers to Test Startup
```bash
# Restart all containers
docker-compose restart

# Or restart specific containers
docker-compose restart morag-api
docker-compose restart morag-worker-1 morag-worker-2
```

### 3. Build and Start Fresh
```bash
# Build and start all containers
docker-compose up --build

# Or build and start in detached mode
docker-compose up --build -d
```

## Expected Results

### ✅ Success Indicators
- Containers start without `ValidationError: QDRANT_COLLECTION_NAME environment variable is required`
- Environment variables are loaded correctly from .env file
- Settings can be imported and accessed
- Both API and worker services are ready to handle requests

### ❌ Failure Indicators
- Containers exit with validation errors
- Environment variables are not available
- Settings import fails
- Services cannot start properly

## Troubleshooting

### If containers still fail to start:

1. **Check .env file exists and is readable**
   ```bash
   ls -la .env
   cat .env | grep QDRANT_COLLECTION_NAME
   ```

2. **Verify docker-compose.yml has env_file configuration**
   ```bash
   grep -A 5 -B 5 "env_file" docker-compose.yml
   ```

3. **Check container environment**
   ```bash
   docker-compose exec morag-api env | grep QDRANT
   ```

4. **Test with explicit environment variable**
   ```bash
   docker-compose exec -e QDRANT_COLLECTION_NAME=test_collection morag-api python -c "from morag_core.config import settings; print(settings.qdrant_collection_name)"
   ```

## Manual Testing

### Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Test with actual request (requires running containers)
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 5}'
```

### Test Worker Tasks
```bash
# Check Celery worker status
docker-compose exec morag-worker-1 celery -A morag.worker inspect active

# Check worker stats
docker-compose exec morag-worker-1 celery -A morag.worker inspect stats
```

## Fix Verification Checklist

- [ ] API container starts without validation errors
- [ ] Worker containers start without validation errors  
- [ ] Environment variables are loaded from .env file
- [ ] Settings can be imported and accessed
- [ ] API endpoints respond correctly
- [ ] Worker tasks can be executed
- [ ] All services use the same collection name
- [ ] No module-level validation errors during import

If all items are checked, the Docker startup fix is working correctly!
