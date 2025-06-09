# Task 2: API Key Integration

## Objective
Integrate API key authentication into all relevant API endpoints to enable user-specific task routing to remote workers.

## Background
Currently, API endpoints route tasks to the default Celery queue without user identification. We need to add API key authentication that allows clients to route tasks to their dedicated remote workers.

## Implementation Steps

### 2.1 Add Authentication Middleware

**File**: `packages/morag/src/morag/middleware/auth.py`

Create authentication middleware for API key validation:

```python
"""Authentication middleware for API key validation."""

from fastapi import Request, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import structlog

from morag.services.auth_service import APIKeyService

logger = structlog.get_logger(__name__)

class APIKeyAuth:
    """API key authentication handler."""

    def __init__(self, api_key_service: APIKeyService):
        self.api_key_service = api_key_service
        self.bearer_scheme = HTTPBearer(auto_error=False)

    async def get_current_user(self, request: Request) -> Optional[dict]:
        """Extract and validate API key from request."""
        # Try Authorization header first
        auth_header = request.headers.get("Authorization")
        api_key = None

        if auth_header and auth_header.startswith("Bearer "):
            api_key = auth_header[7:]
        elif auth_header and auth_header.startswith("ApiKey "):
            api_key = auth_header[7:]
        else:
            # Try X-API-Key header
            api_key = request.headers.get("X-API-Key")

        if not api_key:
            return None  # Anonymous access allowed

        # Validate API key
        user_data = await self.api_key_service.validate_api_key(api_key)
        if not user_data:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key"
            )

        return user_data

    async def get_user_id(self, request: Request) -> Optional[str]:
        """Get user ID from API key, return None for anonymous."""
        user_data = await self.get_current_user(request)
        return user_data.get("user_id") if user_data else None
```

### 2.2 Update Processing Endpoints

**File**: `packages/morag/src/morag/server.py`

Add API key authentication to processing endpoints:

```python
# Add imports
from morag.middleware.auth import APIKeyAuth
from morag.services.auth_service import APIKeyService

# Initialize authentication
redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
api_key_service = APIKeyService(redis_client)
auth = APIKeyAuth(api_key_service)

# Update process_file endpoint
@app.post("/process/file", response_model=ProcessingResult, tags=["Processing"])
async def process_file(
    request: Request,
    source_type: Optional[str] = Form(None),
    file: UploadFile = File(...),
    use_remote: Optional[bool] = Form(False),  # NEW PARAMETER
    metadata: Optional[str] = Form(None)
):
    """Process uploaded file and return results immediately."""
    # Get user ID from API key
    user_id = await auth.get_user_id(request)

    # ... existing file handling code ...

    # Select task based on remote flag and user
    if use_remote and user_id:
        # Create temporary download URL for remote worker
        download_url = await create_temp_download_url(temp_path, user_id)
        task = submit_task_for_user(
            process_file_task_remote,
            args=[download_url, user_id, source_type],
            kwargs=options,
            user_id=user_id,
            use_remote=True
        )
    else:
        # Local processing
        task = process_file_task.delay(temp_path, source_type, options)

    # ... rest of existing code ...
```

### 2.3 Add File Download Service

**File**: `packages/morag/src/morag/services/file_transfer.py`

Create temporary file download service for remote workers:

```python
"""File transfer service for remote workers."""

import secrets
import tempfile
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta
import redis
import json
import structlog

logger = structlog.get_logger(__name__)

class FileTransferService:
    """Service for managing temporary file downloads for remote workers."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.transfer_prefix = "morag:transfers:"
        self.default_expiry = 3600  # 1 hour

    async def create_download_url(self, file_path: str, user_id: str,
                                expires_seconds: int = None) -> str:
        """Create temporary download URL for file."""
        transfer_id = secrets.token_urlsafe(32)
        expires_seconds = expires_seconds or self.default_expiry

        transfer_data = {
            "file_path": file_path,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": (datetime.utcnow() + timedelta(seconds=expires_seconds)).isoformat()
        }

        # Store transfer data
        self.redis.setex(
            f"{self.transfer_prefix}{transfer_id}",
            expires_seconds,
            json.dumps(transfer_data)
        )

        # Return download URL
        return f"/api/v1/download/{transfer_id}"

    async def get_transfer_data(self, transfer_id: str) -> Optional[Dict]:
        """Get transfer data for download."""
        transfer_data_json = self.redis.get(f"{self.transfer_prefix}{transfer_id}")

        if not transfer_data_json:
            return None

        transfer_data = json.loads(transfer_data_json)

        # Check expiration
        expires_at = datetime.fromisoformat(transfer_data["expires_at"])
        if datetime.utcnow() > expires_at:
            return None

        return transfer_data
```

### 2.4 Update Ingestion Endpoints

**File**: `packages/morag/src/morag/server.py`

Add API key authentication to ingestion endpoints:

```python
# Add file transfer service
file_transfer_service = FileTransferService(redis_client)

async def create_temp_download_url(file_path: str, user_id: str) -> str:
    """Create temporary download URL for remote worker."""
    return await file_transfer_service.create_download_url(file_path, user_id)

# Update ingest_file endpoint
@app.post("/api/v1/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
    request: Request,
    source_type: Optional[str] = Form(None),
    file: UploadFile = File(...),
    webhook_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    use_docling: Optional[bool] = Form(False),
    use_remote: Optional[bool] = Form(False)  # NEW PARAMETER
):
    """Ingest and process a file, storing results in vector database."""
    # Get user ID from API key
    user_id = await auth.get_user_id(request)

    # ... existing file handling code ...

    # Select task based on remote flag and user
    if use_remote and user_id:
        # Create temporary download URL for remote worker
        download_url = await create_temp_download_url(temp_path, user_id)

        # Submit to remote worker (processing only)
        processing_task = submit_task_for_user(
            process_file_task_remote,
            args=[download_url, user_id, source_type],
            kwargs=options,
            user_id=user_id,
            use_remote=True
        )

        # Wait for processing result and handle vector storage on server
        processing_result = processing_task.get()

        if processing_result['success']:
            # Store in vector database on server side
            point_ids = await store_content_in_vector_db(
                processing_result['content'],
                {**processing_result['metadata'], 'user_id': user_id}
            )
            processing_result['vector_point_ids'] = point_ids

        return IngestResponse(
            task_id=processing_task.id,
            status="completed",
            result=processing_result
        )
    else:
        # Local processing with vector storage
        task = ingest_file_task.delay(temp_path, source_type, options)
        return IngestResponse(task_id=task.id, status="pending")
```

### 2.5 Add File Download Endpoint

**File**: `packages/morag/src/morag/server.py`

Add secure file download endpoint for remote workers:

```python
from fastapi.responses import FileResponse

@app.get("/api/v1/download/{transfer_id}")
async def download_file(transfer_id: str, request: Request):
    """Download file for remote worker processing."""
    # Validate API key
    user_data = await auth.get_current_user(request)
    if not user_data:
        raise HTTPException(status_code=401, detail="API key required")

    # Get transfer data
    transfer_data = await file_transfer_service.get_transfer_data(transfer_id)
    if not transfer_data:
        raise HTTPException(status_code=404, detail="Transfer not found or expired")

    # Verify user owns this transfer
    if transfer_data["user_id"] != user_data["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    file_path = transfer_data["file_path"]
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=Path(file_path).name,
        headers={"Cache-Control": "no-cache"}
    )
```

### 2.6 Update Request Models

**File**: `packages/morag_core/src/morag_core/models.py`

Add `use_remote` parameter to request models:

```python
class IngestURLRequest(BaseModel):
    url: str
    source_type: Optional[str] = None
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    use_remote: Optional[bool] = False  # NEW FIELD

class IngestBatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    webhook_url: Optional[str] = None
    use_remote: Optional[bool] = False  # NEW FIELD
```

### 2.7 Update URL Processing Logic

**File**: `packages/morag/src/morag/server.py`

Update URL endpoints to handle remote processing:

```python
@app.post("/api/v1/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(request: IngestURLRequest, http_request: Request):
    """Ingest and process content from URL, storing results in vector database."""
    # Get user ID from API key
    user_id = await auth.get_user_id(http_request)

    # ... existing validation code ...

    # Select task based on remote flag and user
    if request.use_remote and user_id:
        # Submit URL directly to remote worker
        processing_task = submit_task_for_user(
            process_url_task_remote,
            args=[request.url, user_id, request.source_type],
            kwargs={"task_options": options},
            user_id=user_id,
            use_remote=True
        )

        # Wait for processing result and handle vector storage on server
        processing_result = processing_task.get()

        if processing_result['success']:
            # Store in vector database on server side
            point_ids = await store_content_in_vector_db(
                processing_result['content'],
                {**processing_result['metadata'], 'user_id': user_id}
            )
            processing_result['vector_point_ids'] = point_ids

        return IngestResponse(
            task_id=processing_task.id,
            status="completed",
            result=processing_result
        )
    else:
        # Local processing with vector storage
        task = ingest_url_task.delay(request.url, request.source_type, options)
        return IngestResponse(task_id=task.id, status="pending")

@app.post("/api/v1/ingest/batch", response_model=BatchIngestResponse, tags=["Ingestion"])
async def ingest_batch(request: IngestBatchRequest, http_request: Request):
    """Ingest and process multiple items in batch, storing results in vector database."""
    # Get user ID from API key
    user_id = await auth.get_user_id(http_request)

    # ... existing validation code ...

    # Select task based on remote flag and user
    if request.use_remote and user_id:
        # Process each item with remote worker
        results = []
        for item in request.items:
            if 'url' in item:
                processing_task = submit_task_for_user(
                    process_url_task_remote,
                    args=[item['url'], user_id, item.get('source_type')],
                    kwargs={"task_options": options},
                    user_id=user_id,
                    use_remote=True
                )
            elif 'file_path' in item:
                download_url = await create_temp_download_url(item['file_path'], user_id)
                processing_task = submit_task_for_user(
                    process_file_task_remote,
                    args=[download_url, user_id, item.get('source_type')],
                    kwargs={"task_options": options},
                    user_id=user_id,
                    use_remote=True
                )

            # Wait for result and store in vector database
            processing_result = processing_task.get()
            if processing_result['success']:
                point_ids = await store_content_in_vector_db(
                    processing_result['content'],
                    {**processing_result['metadata'], 'user_id': user_id}
                )
                processing_result['vector_point_ids'] = point_ids

            results.append(processing_result)

        return BatchIngestResponse(
            task_id=f"batch-{secrets.token_hex(8)}",
            status="completed",
            results=results
        )
    else:
        # Local batch processing
        task = ingest_batch_task.delay(request.items, options)
        return BatchIngestResponse(task_id=task.id, status="pending")
```

### 2.5 Add Import Statements

**File**: `packages/morag/src/morag/server.py`

Add imports for GPU task variants:

```python
from morag.worker import (
    process_file_task, process_url_task, process_web_page_task,
    process_youtube_video_task, process_batch_task, celery_app,
    # Add GPU variants
    process_file_task_gpu, process_url_task_gpu
)
from morag.ingest_tasks import (
    ingest_file_task, ingest_url_task, ingest_batch_task,
    # Add GPU variants  
    ingest_file_task_gpu, ingest_url_task_gpu, ingest_batch_task_gpu
)
```

### 2.6 Update API Documentation

**File**: `packages/morag/src/morag/server.py`

Update FastAPI app description to document GPU parameter:

```python
app = FastAPI(
    title="MoRAG API",
    description="""
    Modular Retrieval Augmented Generation System

    ## Features
    - **Processing Endpoints**: Process content and return results immediately
    - **Ingestion Endpoints**: Process content and store in vector database for retrieval
    - **GPU Processing**: Add `gpu=true` parameter to use GPU workers for faster processing
    - **Task Management**: Track processing status and manage background tasks
    - **Search**: Query stored content using vector similarity

    ## GPU Processing
    Add `gpu=true` parameter to any processing or ingestion endpoint to route tasks to GPU workers:
    - Faster audio transcription with GPU-accelerated Whisper
    - Faster video processing with GPU-accelerated FFmpeg
    - Automatic fallback to CPU workers if GPU workers unavailable

    ## Endpoint Categories
    - `/process/*` - Immediate processing (no storage)
    - `/api/v1/ingest/*` - Background processing with vector storage
    - `/api/v1/status/*` - Task status and management
    - `/search` - Vector similarity search
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)
```

## Testing

### 2.1 Test API Key Authentication
```bash
# Create API key for testing
python -c "
import asyncio
import redis
from morag.services.auth_service import APIKeyService

async def create_test_key():
    redis_client = redis.from_url('redis://localhost:6379/0')
    service = APIKeyService(redis_client)
    api_key = await service.create_api_key('test_user', 'Test key')
    print(f'API Key: {api_key}')

asyncio.run(create_test_key())
"

# Test processing endpoints with API key
curl -X POST "http://localhost:8000/process/file" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@test.mp3" \
  -F "use_remote=true"

curl -X POST "http://localhost:8000/process/url" \
  -H "X-API-Key: YOUR_API_KEY" \
  -F "url=https://example.com/video.mp4" \
  -F "use_remote=true"
```

### 2.2 Test File Download Service
```bash
# Test file download endpoint
curl -X GET "http://localhost:8000/api/v1/download/TRANSFER_ID" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -o downloaded_file.ext

# Test unauthorized access
curl -X GET "http://localhost:8000/api/v1/download/TRANSFER_ID"
# Should return 401 Unauthorized
```

### 2.3 Test Remote Worker Integration
```bash
# Test ingestion endpoints with remote processing
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -F "file=@test.mp3" \
  -F "use_remote=true"

curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"url": "https://example.com/video.mp4", "use_remote": true}'
```

### 2.4 Test Backward Compatibility
```bash
# Test that existing API calls still work without API key
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@test.mp3"

curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/video.mp4"}'
```

## Acceptance Criteria

- [ ] API key authentication middleware implemented
- [ ] All processing endpoints accept API key authentication
- [ ] All ingestion endpoints accept API key authentication
- [ ] File transfer service with temporary download URLs implemented
- [ ] Secure file download endpoint for remote workers
- [ ] Request models include `use_remote` field with default `False`
- [ ] Remote processing routes tasks to user-specific queues
- [ ] Server-side vector storage for remote worker results
- [ ] Backward compatibility maintained (existing calls work unchanged)
- [ ] API documentation updated to describe authentication
- [ ] No breaking changes to existing API contracts

## Files Modified

- `packages/morag/src/morag/server.py`
- `packages/morag_core/src/morag_core/models.py`

## Files Created

- `packages/morag/src/morag/middleware/__init__.py`
- `packages/morag/src/morag/middleware/auth.py`
- `packages/morag/src/morag/services/file_transfer.py`

## Next Steps

After completing this task:
1. Proceed to Task 3: GPU Worker Configuration
2. Test API key authentication and file download
3. Verify user-specific task routing works correctly
4. Test remote worker integration

## Notes

- API key authentication is optional - anonymous access still supported
- Remote workers only do processing, server handles vector storage
- File downloads are secured with user-specific API keys
- Temporary download URLs expire automatically
- User isolation ensures workers only process their user's tasks
