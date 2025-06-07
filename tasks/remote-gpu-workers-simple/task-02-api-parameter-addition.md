# Task 2: API Parameter Addition

## Objective
Add optional `gpu` parameter to all relevant API endpoints to enable GPU task routing while maintaining backward compatibility.

## Background
Currently, API endpoints automatically route tasks to the default Celery queue. We need to add a `gpu` parameter that allows clients to request GPU processing when available.

## Implementation Steps

### 2.1 Update Processing Endpoints

**File**: `packages/morag/src/morag/server.py`

Add `gpu` parameter to immediate processing endpoints:

```python
# Update process_file endpoint
@app.post("/process/file", response_model=ProcessingResult, tags=["Processing"])
async def process_file(
    source_type: Optional[str] = Form(None),
    file: UploadFile = File(...),
    gpu: Optional[bool] = Form(False),  # NEW PARAMETER
    metadata: Optional[str] = Form(None)
):
    """Process uploaded file and return results immediately."""
    # ... existing code ...
    
    # Select task based on GPU requirement
    if gpu:
        task = process_file_task_gpu.delay(temp_path, source_type, options)
    else:
        task = process_file_task.delay(temp_path, source_type, options)
    
    # ... rest of existing code ...

# Update process_url endpoint  
@app.post("/process/url", response_model=ProcessingResult, tags=["Processing"])
async def process_url(
    url: str = Form(...),
    source_type: Optional[str] = Form(None),
    gpu: Optional[bool] = Form(False),  # NEW PARAMETER
    metadata: Optional[str] = Form(None)
):
    """Process content from URL and return results immediately."""
    # ... existing code ...
    
    # Select task based on GPU requirement
    if gpu:
        task = process_url_task_gpu.delay(url, source_type, options)
    else:
        task = process_url_task.delay(url, source_type, options)
    
    # ... rest of existing code ...
```

### 2.2 Update Ingestion Endpoints

**File**: `packages/morag/src/morag/server.py`

Add `gpu` parameter to ingestion endpoints:

```python
# Update ingest_file endpoint
@app.post("/api/v1/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
    source_type: Optional[str] = Form(None),
    file: UploadFile = File(...),
    webhook_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    use_docling: Optional[bool] = Form(False),
    gpu: Optional[bool] = Form(False)  # NEW PARAMETER
):
    """Ingest and process a file, storing results in vector database."""
    # ... existing code ...
    
    # Select task based on GPU requirement
    if gpu:
        task = ingest_file_task_gpu.delay(temp_path, source_type, options)
    else:
        task = ingest_file_task.delay(temp_path, source_type, options)
    
    # ... rest of existing code ...
```

### 2.3 Update Request Models

**File**: `packages/morag_core/src/morag_core/models.py`

Add `gpu` parameter to request models:

```python
class IngestURLRequest(BaseModel):
    url: str
    source_type: Optional[str] = None
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    gpu: Optional[bool] = False  # NEW FIELD

class IngestBatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    webhook_url: Optional[str] = None
    gpu: Optional[bool] = False  # NEW FIELD
```

### 2.4 Update Batch Processing Logic

**File**: `packages/morag/src/morag/server.py`

Update batch endpoints to handle GPU parameter:

```python
@app.post("/api/v1/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_url(request: IngestURLRequest):
    """Ingest and process content from URL, storing results in vector database."""
    # ... existing code ...
    
    # Select task based on GPU requirement
    if request.gpu:
        task = ingest_url_task_gpu.delay(request.url, source_type, options)
    else:
        task = ingest_url_task.delay(request.url, source_type, options)
    
    # ... rest of existing code ...

@app.post("/api/v1/ingest/batch", response_model=BatchIngestResponse, tags=["Ingestion"])
async def ingest_batch(request: IngestBatchRequest):
    """Ingest and process multiple items in batch, storing results in vector database."""
    # ... existing code ...
    
    # Select task based on GPU requirement
    if request.gpu:
        task = ingest_batch_task_gpu.delay(request.items, options)
    else:
        task = ingest_batch_task.delay(request.items, options)
    
    # ... rest of existing code ...
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

### 2.1 Test API Parameter Acceptance
```bash
# Test processing endpoints with GPU parameter
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@test.mp3" \
  -F "gpu=true"

curl -X POST "http://localhost:8000/process/url" \
  -F "url=https://example.com/video.mp4" \
  -F "gpu=true"

# Test ingestion endpoints with GPU parameter
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -F "file=@test.mp3" \
  -F "gpu=true"

curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/video.mp4", "gpu": true}'
```

### 2.2 Test Backward Compatibility
```bash
# Test that existing API calls still work without gpu parameter
curl -X POST "http://localhost:8000/process/file" \
  -F "file=@test.mp3"

curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/video.mp4"}'
```

### 2.3 Test API Documentation
```bash
# Check that API docs include GPU parameter
curl http://localhost:8000/docs
# Verify GPU parameter appears in endpoint documentation
```

## Acceptance Criteria

- [ ] All processing endpoints accept optional `gpu` parameter
- [ ] All ingestion endpoints accept optional `gpu` parameter  
- [ ] Request models include `gpu` field with default `False`
- [ ] GPU parameter routes tasks to appropriate GPU task variants
- [ ] Backward compatibility maintained (existing calls work unchanged)
- [ ] API documentation updated to describe GPU parameter
- [ ] Import statements include GPU task variants
- [ ] No breaking changes to existing API contracts

## Files Modified

- `packages/morag/src/morag/server.py`
- `packages/morag_core/src/morag_core/models.py`

## Next Steps

After completing this task:
1. Proceed to Task 3: GPU Worker Configuration
2. Test API parameter routing with queue monitoring
3. Verify task selection logic works correctly

## Notes

- Default value `False` ensures backward compatibility
- GPU parameter is optional on all endpoints
- Task selection happens at API level, not worker level
- No changes to existing processing logic required
- API documentation clearly explains GPU functionality
