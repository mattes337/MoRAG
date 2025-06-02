# Task 17: Ingestion API Implementation

## Overview
Implement the main ingestion API endpoints that accept various content types and trigger appropriate processing tasks asynchronously.

## Prerequisites
- Task 01: Project Setup completed
- Task 02: API Framework completed
- Task 04: Task Queue Setup completed
- Task 05: Document Parser completed
- Task 14: Gemini Integration completed

## Dependencies
- Task 01: Project Setup
- Task 02: API Framework
- Task 04: Task Queue Setup
- Task 05: Document Parser
- Task 14: Gemini Integration

## Implementation Steps

### 1. Request/Response Models
Create `src/morag/api/models.py`:
```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import mimetypes

class SourceType(str, Enum):
    """Supported source types for ingestion."""
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    WEB = "web"
    YOUTUBE = "youtube"

class IngestionRequest(BaseModel):
    """Base request for content ingestion."""
    source_type: SourceType
    webhook_url: Optional[str] = Field(None, description="URL to notify when processing completes")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True

class FileIngestionRequest(IngestionRequest):
    """Request for file-based ingestion (documents, audio, video)."""
    # File will be uploaded via multipart form data
    use_docling: Optional[bool] = Field(False, description="Use docling for PDF parsing")
    
class URLIngestionRequest(IngestionRequest):
    """Request for URL-based ingestion (web, youtube)."""
    url: str = Field(..., description="URL to process")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class IngestionResponse(BaseModel):
    """Response for ingestion request."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Human-readable message")
    estimated_time: Optional[int] = Field(None, description="Estimated processing time in seconds")

class TaskStatusResponse(BaseModel):
    """Response for task status check."""
    task_id: str
    status: str
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_time_remaining: Optional[int] = None

class BatchIngestionRequest(BaseModel):
    """Request for batch ingestion of multiple items."""
    items: List[Union[FileIngestionRequest, URLIngestionRequest]]
    webhook_url: Optional[str] = None
    
    @validator('items')
    def validate_items(cls, v):
        if len(v) == 0:
            raise ValueError('At least one item is required')
        if len(v) > 50:  # Reasonable batch limit
            raise ValueError('Maximum 50 items per batch')
        return v

class BatchIngestionResponse(BaseModel):
    """Response for batch ingestion."""
    batch_id: str
    task_ids: List[str]
    total_items: int
    message: str
```

### 2. File Upload Utilities
Create `src/morag/utils/file_handling.py`:
```python
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import tempfile
import shutil
import mimetypes
import hashlib
import structlog
from fastapi import UploadFile

from morag.core.config import settings
from morag.core.exceptions import ValidationError

logger = structlog.get_logger()

class FileHandler:
    """Handles file upload and validation."""
    
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.temp_dir = Path(settings.temp_dir)
        
        # Create directories if they don't exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported MIME types
        self.supported_mimes = {
            # Documents
            'application/pdf': 'pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'text/markdown': 'md',
            'text/plain': 'txt',
            
            # Audio
            'audio/mpeg': 'mp3',
            'audio/wav': 'wav',
            'audio/x-wav': 'wav',
            'audio/mp4': 'm4a',
            'audio/x-m4a': 'm4a',
            
            # Video
            'video/mp4': 'mp4',
            'video/quicktime': 'mov',
            'video/x-msvideo': 'avi',
            'video/webm': 'webm',
        }
    
    def validate_file(self, file: UploadFile, source_type: str) -> Tuple[str, Dict[str, Any]]:
        """Validate uploaded file and return file info."""
        
        # Check file size
        if hasattr(file, 'size') and file.size:
            max_size = self._get_max_size_for_type(source_type)
            if file.size > max_size:
                raise ValidationError(f"File too large: {file.size} bytes (max: {max_size})")
        
        # Detect MIME type
        mime_type = file.content_type
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file.filename)
        
        if mime_type not in self.supported_mimes:
            raise ValidationError(f"Unsupported file type: {mime_type}")
        
        # Validate source type matches file type
        file_extension = self.supported_mimes[mime_type]
        if not self._is_valid_for_source_type(file_extension, source_type):
            raise ValidationError(f"File type {file_extension} not valid for source type {source_type}")
        
        # Generate file info
        file_info = {
            'original_filename': file.filename,
            'mime_type': mime_type,
            'file_extension': file_extension,
            'size': getattr(file, 'size', 0)
        }
        
        return file_extension, file_info
    
    async def save_uploaded_file(
        self,
        file: UploadFile,
        source_type: str
    ) -> Tuple[Path, Dict[str, Any]]:
        """Save uploaded file and return path and metadata."""
        
        # Validate file
        file_extension, file_info = self.validate_file(file, source_type)
        
        # Generate unique filename
        file_hash = hashlib.md5(f"{file.filename}{file.size}".encode()).hexdigest()[:8]
        safe_filename = f"{file_hash}_{file.filename}"
        file_path = self.upload_dir / safe_filename
        
        try:
            # Save file
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Update file info with actual size
            file_info['size'] = len(content)
            file_info['file_path'] = str(file_path)
            file_info['file_hash'] = file_hash
            
            logger.info(
                "File uploaded successfully",
                filename=safe_filename,
                size=file_info['size'],
                mime_type=file_info['mime_type']
            )
            
            return file_path, file_info
            
        except Exception as e:
            # Clean up on error
            if file_path.exists():
                file_path.unlink()
            logger.error("Failed to save uploaded file", error=str(e))
            raise ValidationError(f"Failed to save file: {str(e)}")
    
    def _get_max_size_for_type(self, source_type: str) -> int:
        """Get maximum file size for source type."""
        max_sizes = {
            'document': 100 * 1024 * 1024,  # 100MB
            'audio': 500 * 1024 * 1024,     # 500MB
            'video': 2 * 1024 * 1024 * 1024, # 2GB
        }
        return max_sizes.get(source_type, 100 * 1024 * 1024)
    
    def _is_valid_for_source_type(self, file_extension: str, source_type: str) -> bool:
        """Check if file extension is valid for source type."""
        valid_extensions = {
            'document': ['pdf', 'docx', 'doc', 'md', 'txt'],
            'audio': ['mp3', 'wav', 'm4a'],
            'video': ['mp4', 'mov', 'avi', 'webm'],
        }
        return file_extension in valid_extensions.get(source_type, [])
    
    def cleanup_file(self, file_path: Path) -> None:
        """Clean up uploaded file."""
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("File cleaned up", file_path=str(file_path))
        except Exception as e:
            logger.warning("Failed to cleanup file", file_path=str(file_path), error=str(e))

# Global instance
file_handler = FileHandler()
```

### 3. Ingestion Router Implementation
Update `src/morag/api/routes/ingestion.py`:
```python
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import structlog
import uuid
import json

from morag.api.models import (
    IngestionRequest, IngestionResponse, URLIngestionRequest,
    BatchIngestionRequest, BatchIngestionResponse, SourceType
)
from morag.utils.file_handling import file_handler
from morag.tasks.document_tasks import process_document_task
from morag.tasks.base import process_audio_task, process_video_task, process_web_task
from morag.core.exceptions import ValidationError, AuthenticationError

logger = structlog.get_logger()
router = APIRouter()

# Security
security = HTTPBearer(auto_error=False)

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key authentication."""
    if not credentials:
        raise AuthenticationError("API key required")
    
    # In production, verify against actual API keys
    # For now, accept any non-empty token
    if not credentials.credentials:
        raise AuthenticationError("Invalid API key")
    
    return credentials.credentials

@router.post("/file", response_model=IngestionResponse)
async def ingest_file(
    source_type: SourceType = Form(...),
    file: UploadFile = File(...),
    webhook_url: Optional[str] = Form(None),
    metadata: Optional[str] = Form("{}"),
    use_docling: Optional[bool] = Form(False),
    api_key: str = Depends(verify_api_key)
):
    """Ingest a file (document, audio, or video)."""
    
    try:
        # Parse metadata
        try:
            metadata_dict = json.loads(metadata) if metadata else {}
        except json.JSONDecodeError:
            raise ValidationError("Invalid JSON in metadata field")
        
        # Save uploaded file
        file_path, file_info = await file_handler.save_uploaded_file(file, source_type.value)
        
        # Add file info to metadata
        metadata_dict.update(file_info)
        metadata_dict['webhook_url'] = webhook_url
        
        # Route to appropriate task based on source type
        if source_type == SourceType.DOCUMENT:
            task = process_document_task.delay(
                file_path=str(file_path),
                source_type=source_type.value,
                metadata=metadata_dict,
                use_docling=use_docling
            )
            estimated_time = 60  # 1 minute for documents
            
        elif source_type == SourceType.AUDIO:
            task = process_audio_task.delay(
                file_path=str(file_path),
                metadata=metadata_dict
            )
            estimated_time = 300  # 5 minutes for audio
            
        elif source_type == SourceType.VIDEO:
            task = process_video_task.delay(
                file_path=str(file_path),
                metadata=metadata_dict
            )
            estimated_time = 600  # 10 minutes for video
            
        else:
            raise ValidationError(f"Source type {source_type} not supported for file upload")
        
        logger.info(
            "File ingestion task created",
            task_id=task.id,
            source_type=source_type.value,
            filename=file.filename,
            file_size=file_info.get('size', 0)
        )
        
        return IngestionResponse(
            task_id=task.id,
            status="pending",
            message=f"File ingestion started for {file.filename}",
            estimated_time=estimated_time
        )
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error("File ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/url", response_model=IngestionResponse)
async def ingest_url(
    request: URLIngestionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Ingest content from a URL (web page or YouTube)."""
    
    try:
        # Prepare metadata
        metadata = request.metadata or {}
        metadata['webhook_url'] = request.webhook_url
        metadata['url'] = request.url
        
        # Route based on source type
        if request.source_type == SourceType.WEB:
            task = process_web_task.delay(
                url=request.url,
                metadata=metadata
            )
            estimated_time = 120  # 2 minutes for web pages
            
        elif request.source_type == SourceType.YOUTUBE:
            # YouTube processing goes through video task
            task = process_video_task.delay(
                file_path=request.url,  # URL instead of file path
                metadata={**metadata, 'is_youtube': True}
            )
            estimated_time = 900  # 15 minutes for YouTube videos
            
        else:
            raise ValidationError(f"Source type {request.source_type} not supported for URL ingestion")
        
        logger.info(
            "URL ingestion task created",
            task_id=task.id,
            source_type=request.source_type.value,
            url=request.url
        )
        
        return IngestionResponse(
            task_id=task.id,
            status="pending",
            message=f"URL ingestion started for {request.url}",
            estimated_time=estimated_time
        )
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error("URL ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

@router.post("/batch", response_model=BatchIngestionResponse)
async def ingest_batch(
    request: BatchIngestionRequest,
    api_key: str = Depends(verify_api_key)
):
    """Ingest multiple items in batch."""
    
    try:
        batch_id = str(uuid.uuid4())
        task_ids = []
        
        logger.info(
            "Starting batch ingestion",
            batch_id=batch_id,
            item_count=len(request.items)
        )
        
        for i, item in enumerate(request.items):
            try:
                # Add batch info to metadata
                item_metadata = item.metadata or {}
                item_metadata.update({
                    'batch_id': batch_id,
                    'batch_index': i,
                    'batch_total': len(request.items),
                    'webhook_url': request.webhook_url
                })
                
                # Create individual ingestion task
                # Note: This is a simplified implementation
                # In practice, you'd need to handle file uploads differently for batch
                if item.source_type in [SourceType.WEB, SourceType.YOUTUBE]:
                    if item.source_type == SourceType.WEB:
                        task = process_web_task.delay(
                            url=item.url,
                            metadata=item_metadata
                        )
                    else:  # YouTube
                        task = process_video_task.delay(
                            file_path=item.url,
                            metadata={**item_metadata, 'is_youtube': True}
                        )
                    
                    task_ids.append(task.id)
                    
                else:
                    # File-based ingestion in batch would need special handling
                    logger.warning(
                        "File-based batch ingestion not fully implemented",
                        source_type=item.source_type
                    )
                    
            except Exception as e:
                logger.error(
                    "Failed to create task for batch item",
                    batch_id=batch_id,
                    item_index=i,
                    error=str(e)
                )
                # Continue with other items
        
        return BatchIngestionResponse(
            batch_id=batch_id,
            task_ids=task_ids,
            total_items=len(task_ids),
            message=f"Batch ingestion started with {len(task_ids)} tasks"
        )
        
    except Exception as e:
        logger.error("Batch ingestion failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Batch ingestion failed: {str(e)}")

@router.delete("/{task_id}")
async def cancel_ingestion(
    task_id: str,
    api_key: str = Depends(verify_api_key)
):
    """Cancel an ingestion task."""
    
    try:
        from morag.services.task_manager import task_manager
        
        success = task_manager.cancel_task(task_id)
        
        if success:
            return {"message": f"Task {task_id} cancelled successfully"}
        else:
            raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")
            
    except Exception as e:
        logger.error("Failed to cancel task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")
```

### 4. Enhanced Status Router
Update `src/morag/api/routes/status.py`:
```python
from fastapi import APIRouter, Path, HTTPException, Depends
from typing import List
import structlog

from morag.api.models import TaskStatusResponse
from morag.services.task_manager import task_manager, TaskStatus
from morag.api.routes.ingestion import verify_api_key

logger = structlog.get_logger()
router = APIRouter()

@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str = Path(..., description="Task ID to check"),
    api_key: str = Depends(verify_api_key)
):
    """Get the status of an ingestion task."""
    
    try:
        task_info = task_manager.get_task_status(task_id)
        
        # Calculate estimated time remaining
        estimated_remaining = None
        if task_info.progress and task_info.progress > 0:
            # Simple estimation based on progress
            if task_info.status == TaskStatus.PROGRESS:
                estimated_remaining = int((1 - task_info.progress) * 300)  # Rough estimate
        
        return TaskStatusResponse(
            task_id=task_info.task_id,
            status=task_info.status.value,
            progress=task_info.progress,
            result=task_info.result,
            error=task_info.error,
            created_at=task_info.created_at.isoformat() if task_info.created_at else None,
            started_at=task_info.started_at.isoformat() if task_info.started_at else None,
            completed_at=task_info.completed_at.isoformat() if task_info.completed_at else None,
            estimated_time_remaining=estimated_remaining
        )
        
    except Exception as e:
        logger.error("Failed to get task status", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

@router.get("/")
async def list_active_tasks(api_key: str = Depends(verify_api_key)):
    """List all active tasks."""
    
    try:
        active_tasks = task_manager.get_active_tasks()
        return {
            "active_tasks": active_tasks,
            "count": len(active_tasks)
        }
        
    except Exception as e:
        logger.error("Failed to list active tasks", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

@router.get("/stats/queues")
async def get_queue_stats(api_key: str = Depends(verify_api_key)):
    """Get queue statistics."""
    
    try:
        stats = task_manager.get_queue_stats()
        return stats
        
    except Exception as e:
        logger.error("Failed to get queue stats", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get queue stats: {str(e)}")
```

## Testing Instructions

### 1. Test File Upload
```bash
# Test document upload
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -H "Authorization: Bearer test-api-key" \
  -F "source_type=document" \
  -F "file=@test.pdf" \
  -F "metadata={\"test\": true}"

# Test with webhook
curl -X POST "http://localhost:8000/api/v1/ingest/file" \
  -H "Authorization: Bearer test-api-key" \
  -F "source_type=document" \
  -F "file=@test.pdf" \
  -F "webhook_url=https://example.com/webhook"
```

### 2. Test URL Ingestion
```bash
# Test web page
curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Authorization: Bearer test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "web",
    "url": "https://example.com",
    "metadata": {"test": true}
  }'

# Test YouTube
curl -X POST "http://localhost:8000/api/v1/ingest/url" \
  -H "Authorization: Bearer test-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "youtube",
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
  }'
```

### 3. Test Status Checking
```bash
# Get task status
curl -X GET "http://localhost:8000/api/v1/status/{task_id}" \
  -H "Authorization: Bearer test-api-key"

# List active tasks
curl -X GET "http://localhost:8000/api/v1/status/" \
  -H "Authorization: Bearer test-api-key"

# Get queue stats
curl -X GET "http://localhost:8000/api/v1/status/stats/queues" \
  -H "Authorization: Bearer test-api-key"
```

## Success Criteria
- [ ] File upload endpoint accepts and validates files
- [ ] URL ingestion endpoint processes web and YouTube URLs
- [ ] Tasks are created and queued correctly
- [ ] Status endpoints return accurate information
- [ ] API key authentication works
- [ ] Error handling provides meaningful messages
- [ ] File validation prevents invalid uploads
- [ ] Batch ingestion works for URL-based content
- [ ] Task cancellation works
- [ ] Integration with existing task system works

## Next Steps
- Task 18: Status Tracking (enhanced webhook notifications)
- Task 08: Audio Processing (implements audio_tasks)
- Task 09: Video Processing (implements video_tasks)
- Task 12: Web Scraping (implements web_tasks)
