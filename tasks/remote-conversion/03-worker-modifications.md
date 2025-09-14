# Task 3: Worker Modifications for Remote Processing

## Overview

Modify existing MoRAG workers to handle the `remote=true` parameter in ingestion requests. When remote processing is requested, workers will create remote jobs instead of processing content locally, while maintaining backward compatibility for local processing.

## Objectives

1. Modify ingestion tasks to check for `remote` parameter
2. Implement remote job creation logic in workers
3. Add fallback mechanisms when remote workers are unavailable
4. Maintain backward compatibility with existing API
5. Add proper logging and monitoring for remote job creation

## Technical Requirements

### 1. Enhanced Ingestion Task Parameters

**File**: `packages/morag/src/morag/models/ingestion_request.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class EnhancedIngestionRequest(BaseModel):
    """Enhanced ingestion request with remote processing support."""
    
    source_type: Optional[str] = Field(None, description="Content type (auto-detected if not provided)")
    webhook_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    use_docling: Optional[bool] = Field(False, description="Use docling for PDF processing")
    chunk_size: Optional[int] = Field(None, description="Chunk size for document processing")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap for document processing")
    chunking_strategy: Optional[str] = Field(None, description="Chunking strategy")
    document_id: Optional[str] = Field(None, description="Document identifier for replacement")
    replace_existing: Optional[bool] = Field(False, description="Replace existing document")
    
    # New remote processing parameters
    remote: Optional[bool] = Field(False, description="Use remote processing workers")
    remote_timeout: Optional[int] = Field(None, description="Remote processing timeout in seconds")
    fallback_to_local: Optional[bool] = Field(True, description="Fallback to local processing if remote fails")

class RemoteJobCreationRequest(BaseModel):
    """Request for creating a remote conversion job."""
    
    source_file_path: str = Field(..., description="Path to source file")
    content_type: str = Field(..., description="Content type")
    task_options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    ingestion_task_id: str = Field(..., description="Associated ingestion task ID")
    timeout_seconds: Optional[int] = Field(None, description="Job timeout in seconds")
```

### 2. Remote Job Creation Service

**File**: `packages/morag/src/morag/services/remote_job_creator.py`

```python
import structlog
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import requests
import os

from morag.models.ingestion_request import RemoteJobCreationRequest
from morag_core.config import settings

logger = structlog.get_logger(__name__)

class RemoteJobCreator:
    """Service for creating and managing remote conversion jobs."""
    
    def __init__(self):
        self.api_base_url = os.getenv('MORAG_API_BASE_URL', 'http://localhost:8000')
        self.timeout_config = {
            'audio': int(os.getenv('MORAG_REMOTE_TIMEOUT_AUDIO', '3600')),  # 60 minutes
            'video': int(os.getenv('MORAG_REMOTE_TIMEOUT_VIDEO', '3600')),  # 60 minutes
            'document': int(os.getenv('MORAG_REMOTE_TIMEOUT_DOCUMENT', '600')),  # 10 minutes
            'image': int(os.getenv('MORAG_REMOTE_TIMEOUT_IMAGE', '300')),  # 5 minutes
            'web': int(os.getenv('MORAG_REMOTE_TIMEOUT_WEB', '600')),  # 10 minutes
            'youtube': int(os.getenv('MORAG_REMOTE_TIMEOUT_YOUTUBE', '1800'))  # 30 minutes
        }
    
    def should_use_remote_processing(self, content_type: str, remote_requested: bool) -> bool:
        """Determine if remote processing should be used."""
        if not remote_requested:
            return False
        
        # Check if remote processing is enabled globally
        if not os.getenv('MORAG_REMOTE_CONVERSION_ENABLED', 'false').lower() == 'true':
            logger.warning("Remote conversion requested but not enabled globally")
            return False
        
        # Check if content type supports remote processing
        supported_types = ['audio', 'video', 'youtube']  # Start with media types
        if content_type not in supported_types:
            logger.info("Content type not supported for remote processing",
                       content_type=content_type,
                       supported_types=supported_types)
            return False
        
        return True
    
    def create_remote_job(self, request: RemoteJobCreationRequest) -> Optional[str]:
        """Create a remote conversion job and return job ID."""
        try:
            # Determine timeout based on content type
            timeout_seconds = request.timeout_seconds
            if not timeout_seconds:
                timeout_seconds = self.timeout_config.get(request.content_type, 1800)
            
            # Prepare job creation payload
            payload = {
                'source_file_path': request.source_file_path,
                'content_type': request.content_type,
                'task_options': request.task_options,
                'ingestion_task_id': request.ingestion_task_id
            }
            
            # Make API call to create remote job
            response = requests.post(
                f"{self.api_base_url}/api/v1/remote-jobs/",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data.get('job_id')
                
                logger.info("Remote job created successfully",
                           job_id=job_id,
                           content_type=request.content_type,
                           ingestion_task_id=request.ingestion_task_id)
                
                return job_id
            else:
                logger.error("Failed to create remote job",
                           status_code=response.status_code,
                           response=response.text)
                return None
                
        except Exception as e:
            logger.error("Exception creating remote job", error=str(e))
            return None
    
    def wait_for_remote_job_completion(self, job_id: str, timeout_seconds: int = 1800) -> Optional[Dict[str, Any]]:
        """Wait for remote job completion and return result."""
        import time
        
        start_time = time.time()
        poll_interval = 10  # Poll every 10 seconds
        
        while time.time() - start_time < timeout_seconds:
            try:
                # Check job status
                response = requests.get(
                    f"{self.api_base_url}/api/v1/remote-jobs/{job_id}/status",
                    timeout=10
                )
                
                if response.status_code == 200:
                    job_status = response.json()
                    status = job_status.get('status')
                    
                    if status == 'completed':
                        logger.info("Remote job completed successfully", job_id=job_id)
                        return job_status
                    elif status in ['failed', 'timeout', 'cancelled']:
                        logger.error("Remote job failed",
                                   job_id=job_id,
                                   status=status,
                                   error=job_status.get('error_message'))
                        return None
                    else:
                        # Job still processing, continue waiting
                        logger.debug("Remote job still processing",
                                   job_id=job_id,
                                   status=status)
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error("Error checking remote job status",
                           job_id=job_id,
                           error=str(e))
                time.sleep(poll_interval)
        
        logger.error("Remote job timed out", job_id=job_id, timeout=timeout_seconds)
        return None
    
    def get_remote_job_result(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get the result data from a completed remote job."""
        try:
            response = requests.get(
                f"{self.api_base_url}/api/v1/remote-jobs/{job_id}/status",
                timeout=10
            )
            
            if response.status_code == 200:
                job_data = response.json()
                if job_data.get('status') == 'completed':
                    # Extract result data from the job
                    # This would need to be implemented based on how results are stored
                    return job_data
            
            return None
            
        except Exception as e:
            logger.error("Error getting remote job result", job_id=job_id, error=str(e))
            return None
```

### 3. Modified Ingestion Tasks

**File**: `packages/morag/src/morag/ingest_tasks_enhanced.py`

```python
import structlog
from typing import Optional, Dict, Any
from celery import current_task
import asyncio

from morag.worker import celery_app, get_morag_api
from morag.services.remote_job_creator import RemoteJobCreator
from morag.models.ingestion_request import RemoteJobCreationRequest
from morag_core.models import ProcessingResult
from morag.storage.storage_manager import storage_manager

logger = structlog.get_logger(__name__)

@celery_app.task(bind=True)
def enhanced_ingest_file_task(self, file_path: str, content_type: Optional[str] = None, task_options: Optional[Dict[str, Any]] = None):
    """Enhanced file ingestion task with remote processing support."""
    
    async def _ingest():
        api = get_morag_api()
        options = task_options or {}
        
        try:
            self.update_state(state='PROGRESS', meta={'stage': 'initializing', 'progress': 0.1})
            
            # Check if remote processing is requested
            remote_requested = options.get('remote', False)
            fallback_to_local = options.get('fallback_to_local', True)
            
            logger.info("Starting file ingestion",
                       file_path=file_path,
                       content_type=content_type,
                       remote_requested=remote_requested,
                       task_id=self.request.id)
            
            # Determine if we should use remote processing
            remote_creator = RemoteJobCreator()
            use_remote = remote_creator.should_use_remote_processing(content_type, remote_requested)
            
            if use_remote:
                logger.info("Using remote processing", file_path=file_path, content_type=content_type)
                result = await _process_remotely(self, file_path, content_type, options, remote_creator)
                
                # If remote processing failed and fallback is enabled, try local processing
                if result is None and fallback_to_local:
                    logger.warning("Remote processing failed, falling back to local processing",
                                 file_path=file_path)
                    result = await _process_locally(self, api, file_path, content_type, options)
            else:
                logger.info("Using local processing", file_path=file_path, content_type=content_type)
                result = await _process_locally(self, api, file_path, content_type, options)
            
            if result is None:
                raise Exception("Both remote and local processing failed")
            
            # Continue with vector storage (same as before)
            return await _complete_ingestion(self, result, options)
            
        except Exception as e:
            logger.error("File ingestion task failed", file_path=file_path, error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise
    
    return asyncio.run(_ingest())

async def _process_remotely(task, file_path: str, content_type: str, options: Dict[str, Any], remote_creator: RemoteJobCreator) -> Optional[ProcessingResult]:
    """Process file using remote workers."""
    try:
        task.update_state(state='PROGRESS', meta={'stage': 'creating_remote_job', 'progress': 0.2})
        
        # Create remote job using repository
        from morag_core.models.remote_job import RemoteJob

        job = RemoteJob.create_new(
            ingestion_task_id=task.request.id,
            source_file_path=file_path,
            content_type=content_type,
            task_options=options
        )

        # Set timeout based on content type
        timeout_hours = {'audio': 0.5, 'video': 1, 'document': 0.25}.get(content_type, 1)
        job.timeout_at = datetime.utcnow() + timedelta(hours=timeout_hours)

        # Save job to storage
        if not storage_manager.remote_jobs.save_job(job.to_dict()):
            logger.error("Failed to create remote job")
            return None

        job_id = job.id
        if not job_id:
            logger.error("Failed to create remote job")
            return None
        
        task.update_state(state='PROGRESS', meta={
            'stage': 'waiting_for_remote_completion',
            'progress': 0.3,
            'remote_job_id': job_id
        })
        
        # Wait for remote job completion by polling storage
        timeout_seconds = options.get('remote_timeout', 1800)
        start_time = time.time()
        poll_interval = 10  # Poll every 10 seconds

        while time.time() - start_time < timeout_seconds:
            # Check job status from storage
            job_data = storage_manager.remote_jobs.load_job(job_id)
            if not job_data:
                logger.error("Remote job not found", job_id=job_id)
                return None

            status = job_data.get('status')

            if status == 'completed':
                # Job completed successfully
                result_data = job_data.get('result_data', {})
                return ProcessingResult(
                    success=True,
                    text_content=result_data.get('content', ''),
                    metadata=result_data.get('metadata', {}),
                    processing_time=result_data.get('processing_time', 0.0),
                    error_message=None
                )
            elif status in ['failed', 'timeout', 'cancelled']:
                # Job failed
                logger.error("Remote job failed",
                           job_id=job_id,
                           status=status,
                           error=job_data.get('error_message'))
                return None

            # Job still processing, wait and check again
            await asyncio.sleep(poll_interval)

        # Timeout reached
        logger.error("Remote job timed out", job_id=job_id, timeout=timeout_seconds)
        return None
        
    except Exception as e:
        logger.error("Remote processing failed", error=str(e))
        return None

async def _process_locally(task, api, file_path: str, content_type: str, options: Dict[str, Any]) -> Optional[ProcessingResult]:
    """Process file using local workers (existing logic)."""
    try:
        task.update_state(state='PROGRESS', meta={'stage': 'local_processing', 'progress': 0.4})
        
        # Use existing local processing logic
        result = await api.process_file(file_path, content_type, options)
        
        if not result.success:
            raise Exception(f"Local processing failed: {result.error_message}")
        
        return result
        
    except Exception as e:
        logger.error("Local processing failed", error=str(e))
        return None

async def _complete_ingestion(task, result: ProcessingResult, options: Dict[str, Any]) -> Dict[str, Any]:
    """Complete the ingestion process with vector storage."""
    try:
        task.update_state(state='PROGRESS', meta={'stage': 'storing', 'progress': 0.7})
        
        # Ensure metadata is always a dictionary
        if result.metadata is None:
            result.metadata = {}
        
        # Store in vector database (existing logic)
        from morag_services import MoRAGServices
        from morag_services.storage import QdrantVectorStorage
        from morag_services.embedding import GeminiEmbeddingService
        
        services = MoRAGServices()
        storage = QdrantVectorStorage()
        embedding_service = GeminiEmbeddingService()
        
        # Prepare content for vector storage
        content_for_storage = result.text_content or ""
        
        # Generate document chunks
        from morag_core.utils.chunking import create_document_chunks
        chunks = create_document_chunks(
            content_for_storage,
            chunk_size=options.get('chunk_size', 4000),
            chunk_overlap=options.get('chunk_overlap', 200),
            strategy=options.get('chunking_strategy', 'sentence')
        )
        
        # Store chunks in vector database
        stored_count = 0
        for chunk in chunks:
            try:
                embedding = await embedding_service.generate_embedding(chunk.content)
                
                vector_metadata = {
                    **result.metadata,
                    'chunk_index': chunk.index,
                    'chunk_size': len(chunk.content),
                    'document_id': options.get('document_id'),
                    'source_type': options.get('source_type', 'unknown'),
                    'processing_method': 'remote' if options.get('remote') else 'local'
                }
                
                await storage.store_vector(
                    vector=embedding,
                    metadata=vector_metadata,
                    content=chunk.content
                )
                stored_count += 1
                
            except Exception as e:
                logger.error("Failed to store chunk", chunk_index=chunk.index, error=str(e))
        
        task.update_state(state='PROGRESS', meta={'stage': 'completed', 'progress': 1.0})
        
        logger.info("Ingestion completed successfully",
                   chunks_stored=stored_count,
                   processing_time=result.processing_time)
        
        return {
            'success': True,
            'chunks_stored': stored_count,
            'processing_time': result.processing_time,
            'metadata': result.metadata,
            'processing_method': 'remote' if options.get('remote') else 'local'
        }
        
    except Exception as e:
        logger.error("Failed to complete ingestion", error=str(e))
        raise
```

### 4. API Endpoint Updates

**File**: `packages/morag/src/morag/server_enhanced.py`

```python
# Add to existing server.py file

from morag.models.ingestion_request import EnhancedIngestionRequest

@app.post("/api/v1/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def enhanced_ingest_file(
    file: UploadFile = File(...),
    source_type: Optional[str] = Form(default=None),
    document_id: Optional[str] = Form(default=None),
    replace_existing: Optional[bool] = Form(default=False),
    webhook_url: Optional[str] = Form(default=None),
    metadata: Optional[str] = Form(default=None),
    use_docling: Optional[bool] = Form(default=False),
    chunk_size: Optional[int] = Form(default=None),
    chunk_overlap: Optional[int] = Form(default=None),
    chunking_strategy: Optional[str] = Form(default=None),
    # New remote processing parameters
    remote: Optional[bool] = Form(default=False),
    remote_timeout: Optional[int] = Form(default=None),
    fallback_to_local: Optional[bool] = Form(default=True)
):
    """Enhanced file ingestion with remote processing support."""
    try:
        # Validate file upload
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file
        upload_handler = get_upload_handler()
        temp_path = await upload_handler.save_upload(file)
        
        # Parse metadata if provided
        parsed_metadata = {}
        if metadata:
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid metadata JSON")
        
        # Auto-detect source type if not provided
        if not source_type:
            source_type = get_morag_api()._detect_content_type(str(temp_path))
            logger.info("Auto-detected content type for file",
                       filename=file.filename,
                       detected_type=source_type)
        
        # Create enhanced task options
        options = {
            "document_id": document_id,
            "replace_existing": replace_existing,
            "webhook_url": webhook_url or "",
            "metadata": parsed_metadata or {},
            "use_docling": use_docling,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunking_strategy": chunking_strategy,
            "store_in_vector_db": True,
            # Remote processing options
            "remote": remote,
            "remote_timeout": remote_timeout,
            "fallback_to_local": fallback_to_local
        }
        
        # Submit to enhanced ingestion task
        from morag.ingest_tasks_enhanced import enhanced_ingest_file_task
        task = enhanced_ingest_file_task.delay(
            str(temp_path),
            source_type,
            options
        )
        
        logger.info("Enhanced file ingestion task created",
                   task_id=task.id,
                   filename=file.filename,
                   source_type=source_type,
                   remote_requested=remote)
        
        return IngestResponse(
            task_id=task.id,
            status="pending",
            message=f"File '{file.filename}' queued for {'remote' if remote else 'local'} processing and ingestion"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File ingestion failed", filename=file.filename, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File ingestion failed: {str(e)}"
        )
```

## Implementation Steps

1. **Create Enhanced Models** (Day 1)
   - Add remote processing parameters to ingestion requests
   - Create remote job creation request model
   - Update API documentation

2. **Implement Remote Job Creator** (Day 1-2)
   - Create service for remote job management
   - Add timeout configuration
   - Implement job status polling

3. **Modify Ingestion Tasks** (Day 2-3)
   - Update existing ingestion tasks
   - Add remote processing logic
   - Implement fallback mechanisms

4. **Update API Endpoints** (Day 3)
   - Add remote parameters to ingestion endpoints
   - Update request validation
   - Add proper error handling

5. **Testing and Integration** (Day 4)
   - Test remote job creation
   - Test fallback mechanisms
   - Integration testing with existing system

## Testing Requirements

### Unit Tests

**File**: `tests/test_worker_modifications.py`

```python
import pytest
from unittest.mock import Mock, patch, AsyncMock

from morag.services.remote_job_creator import RemoteJobCreator
from morag.ingest_tasks_enhanced import enhanced_ingest_file_task

class TestWorkerModifications:
    def test_should_use_remote_processing(self):
        # Test remote processing decision logic
        pass
    
    def test_create_remote_job_success(self):
        # Test successful remote job creation
        pass
    
    def test_create_remote_job_failure(self):
        # Test remote job creation failure
        pass
    
    def test_fallback_to_local_processing(self):
        # Test fallback when remote processing fails
        pass
    
    def test_enhanced_ingestion_task(self):
        # Test enhanced ingestion task with remote processing
        pass
```

## Success Criteria

1. Workers correctly identify when to use remote processing
2. Remote job creation works reliably
3. Fallback to local processing works when remote fails
4. All existing functionality remains unchanged when remote=false
5. Proper logging and monitoring for remote operations

## Dependencies

- Remote job API endpoints (Task 1)
- Database schema for remote jobs (Task 2)
- Existing MoRAG ingestion system
- Celery task queue system

## Next Steps

After completing this task:
1. Proceed to [Task 4: Remote Converter Tool](./04-remote-converter-tool.md)
2. Test worker modifications with mock remote jobs
3. Validate fallback mechanisms
4. Begin development of remote converter application
