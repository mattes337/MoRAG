# Task 1: Queue Architecture Setup

## Objective
Configure Celery to support user-specific task queues with API key-based routing for remote workers.

## Background
Currently, MoRAG uses a single Celery queue ("celery") for all tasks. We need to add user-specific queues that route tasks based on API keys, allowing remote workers to process only their user's tasks while maintaining backward compatibility.

## Implementation Steps

### 1.1 Add API Key Authentication Service

**File**: `packages/morag/src/morag/services/auth_service.py`

Create API key authentication service:

```python
"""API key authentication service for remote workers."""

import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import redis
import json
import structlog

logger = structlog.get_logger(__name__)

class APIKeyService:
    """Service for managing API keys and user authentication."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "morag:api_keys:"
        self.user_prefix = "morag:users:"

    async def create_api_key(self, user_id: str, description: str = "",
                           expires_days: Optional[int] = None) -> str:
        """Create a new API key for a user."""
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        key_data = {
            "user_id": user_id,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(days=expires_days)).isoformat() if expires_days else None,
            "active": True
        }

        # Store API key data
        self.redis.setex(
            f"{self.key_prefix}{key_hash}",
            timedelta(days=expires_days or 365).total_seconds(),
            json.dumps(key_data)
        )

        logger.info("API key created", user_id=user_id, description=description)
        return api_key

    async def validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return user information."""
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_data_json = self.redis.get(f"{self.key_prefix}{key_hash}")

        if not key_data_json:
            return None

        key_data = json.loads(key_data_json)

        # Check if key is active
        if not key_data.get("active", False):
            return None

        # Check expiration
        if key_data.get("expires_at"):
            expires_at = datetime.fromisoformat(key_data["expires_at"])
            if datetime.utcnow() > expires_at:
                return None

        return key_data

    def get_user_queue_name(self, user_id: str, worker_type: str = "gpu") -> str:
        """Get queue name for user and worker type."""
        return f"{worker_type}-tasks-{user_id}"
```

### 1.2 Update Celery Configuration

**File**: `packages/morag/src/morag/worker.py`

Add dynamic queue routing configuration:

```python
# Add imports
from morag.services.auth_service import APIKeyService
import redis
import os

# Initialize API key service
redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
api_key_service = APIKeyService(redis_client)

# Add after existing celery_app.conf.update() call
celery_app.conf.update(
    # ... existing configuration ...

    # Dynamic queue routing - queues created on demand
    task_default_queue='celery',
    task_default_exchange='celery',
    task_default_exchange_type='direct',
    task_default_routing_key='celery',

    # Enable dynamic queue creation
    task_create_missing_queues=True,
    worker_direct=True,
)
```

### 1.3 Create Remote Worker Task Variants

**File**: `packages/morag/src/morag/worker.py`

Add remote worker task variants that handle HTTP file transfer:

```python
@celery_app.task(bind=True)
def process_file_task_remote(self, file_url: str, user_id: str, content_type: Optional[str] = None,
                           task_options: Optional[Dict[str, Any]] = None):
    """Remote worker variant - downloads file via HTTP and processes."""
    import tempfile
    import requests
    from pathlib import Path

    async def _process():
        api = get_morag_api()
        temp_path = None

        try:
            # Download file from server
            self.update_state(state='DOWNLOADING', meta={'stage': 'downloading_file'})

            response = requests.get(file_url, stream=True)
            response.raise_for_status()

            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_url).suffix) as temp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_path = temp_file.name

            self.update_state(state='PROCESSING', meta={'stage': 'processing'})

            # Process the file (only heavy lifting - no external services)
            result = await api.process_file(temp_path, content_type, task_options)

            # Return only markdown content and metadata - no vector storage
            return {
                'success': result.success,
                'content': result.text_content or result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'user_id': user_id
            }

        except Exception as e:
            logger.error("Remote file processing failed", file_url=file_url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
        finally:
            # Clean up temporary file
            if temp_path and Path(temp_path).exists():
                Path(temp_path).unlink()

    return asyncio.run(_process())

@celery_app.task(bind=True)
def process_url_task_remote(self, url: str, user_id: str, content_type: Optional[str] = None,
                          task_options: Optional[Dict[str, Any]] = None):
    """Remote worker variant - processes URL directly."""
    async def _process():
        api = get_morag_api()
        try:
            self.update_state(state='PROCESSING', meta={'stage': 'processing_url'})

            # Process URL directly (no external service calls)
            result = await api.process_url(url, content_type, task_options)

            # Return only markdown content and metadata
            return {
                'success': result.success,
                'content': result.text_content or result.content,
                'metadata': result.metadata,
                'processing_time': result.processing_time,
                'error_message': result.error_message,
                'user_id': user_id
            }

        except Exception as e:
            logger.error("Remote URL processing failed", url=url, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise

    return asyncio.run(_process())
```

### 1.4 Add Queue Selection Helper

**File**: `packages/morag/src/morag/worker.py`

Add helper function to select appropriate task and queue based on user:

```python
def get_task_for_user(base_task_name: str, user_id: Optional[str] = None,
                     use_remote: bool = False):
    """Get the appropriate task function and queue based on user and remote flag."""
    if use_remote and user_id:
        remote_task_name = f"{base_task_name}_remote"
        queue_name = api_key_service.get_user_queue_name(user_id, "gpu")
        return globals().get(remote_task_name, globals()[base_task_name]), queue_name

    # Default to local processing
    return globals()[base_task_name], 'celery'

def submit_task_for_user(task_func, args, kwargs, user_id: Optional[str] = None,
                        use_remote: bool = False):
    """Submit task to appropriate queue based on user."""
    task, queue = get_task_for_user(task_func.__name__, user_id, use_remote)

    return task.apply_async(
        args=args,
        kwargs=kwargs,
        queue=queue
    )
```

## Testing

### 1.1 Test API Key Service
```bash
# Test API key creation and validation
python -c "
import asyncio
import redis
from morag.services.auth_service import APIKeyService

async def test_api_keys():
    redis_client = redis.from_url('redis://localhost:6379/0')
    service = APIKeyService(redis_client)

    # Create API key
    api_key = await service.create_api_key('user123', 'Test key')
    print(f'Created API key: {api_key}')

    # Validate API key
    user_data = await service.validate_api_key(api_key)
    print(f'User data: {user_data}')

    # Get queue name
    queue_name = service.get_user_queue_name('user123', 'gpu')
    print(f'Queue name: {queue_name}')

asyncio.run(test_api_keys())
"
```

### 1.2 Test Queue Configuration
```bash
# Start local worker (default queue)
celery -A morag.worker worker --loglevel=info --queues=celery

# Start remote worker for specific user
celery -A morag.worker worker --loglevel=info --queues=gpu-tasks-user123

# Test dynamic queue creation
python -c "
from morag.worker import submit_task_for_user, process_file_task
import tempfile

# Test local task submission
result = submit_task_for_user(
    process_file_task,
    args=['/tmp/test.txt'],
    kwargs={},
    user_id=None,
    use_remote=False
)
print(f'Local task: {result.id}')

# Test remote task submission
result = submit_task_for_user(
    process_file_task,
    args=['/tmp/test.txt'],
    kwargs={},
    user_id='user123',
    use_remote=True
)
print(f'Remote task: {result.id}')
"
```

## Acceptance Criteria

- [ ] API key authentication service implemented with Redis storage
- [ ] User-specific queue naming convention established
- [ ] Remote worker task variants created for file and URL processing
- [ ] HTTP file download functionality implemented in remote tasks
- [ ] Queue selection helper functions work correctly
- [ ] Workers can be started for user-specific queues
- [ ] API key validation and user identification working
- [ ] No breaking changes to existing functionality

## Files Modified

- `packages/morag/src/morag/worker.py`
- `packages/morag/src/morag/services/auth_service.py` (new file)

## Files Created

- `packages/morag/src/morag/services/__init__.py`
- `packages/morag/src/morag/services/auth_service.py`

## Next Steps

After completing this task:
1. Proceed to Task 2: API Key Integration in API endpoints
2. Test API key authentication and queue routing
3. Verify user isolation and task routing
4. Test HTTP file download in remote workers

## Notes

- This approach provides complete user isolation
- Remote workers only process tasks for their authenticated user
- HTTP file transfer eliminates need for shared storage
- Workers return only processed content, no external service calls
- Maintains backward compatibility with anonymous processing
