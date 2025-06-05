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
from morag.utils.file_handling import FileHandler
from morag.core.exceptions import ValidationError, AuthenticationError

# Mock task implementations
class MockTask:
    def __init__(self, task_id):
        self.id = task_id

    @classmethod
    def delay(cls, **kwargs):
        return cls(str(uuid.uuid4()))

process_document_task = MockTask
process_audio_file = MockTask
process_video_file = MockTask
process_image_file = MockTask
process_web_url = MockTask

# Mock file handler
file_handler = FileHandler()

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
    """Ingest a file (document, audio, video, or image)."""

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
            task = process_audio_file.delay(
                file_path=str(file_path),
                task_id=str(uuid.uuid4()),
                config=metadata_dict
            )
            estimated_time = 300  # 5 minutes for audio

        elif source_type == SourceType.VIDEO:
            task = process_video_file.delay(
                file_path=str(file_path),
                task_id=str(uuid.uuid4()),
                config=metadata_dict
            )
            estimated_time = 600  # 10 minutes for video

        elif source_type == SourceType.IMAGE:
            task = process_image_file.delay(
                file_path=str(file_path),
                task_id=str(uuid.uuid4()),
                config=metadata_dict
            )
            estimated_time = 30  # 30 seconds for images

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
            task = process_web_url.delay(
                url=request.url,
                metadata=metadata,
                task_id=str(uuid.uuid4())
            )
            estimated_time = 120  # 2 minutes for web pages

        elif request.source_type == SourceType.YOUTUBE:
            # YouTube processing goes through video task
            task = process_video_file.delay(
                file_path=request.url,  # URL instead of file path
                task_id=str(uuid.uuid4()),
                config={**metadata, 'is_youtube': True}
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
                        task = process_web_url.delay(
                            url=item.url,
                            metadata=item_metadata,
                            task_id=str(uuid.uuid4())
                        )
                    else:  # YouTube
                        task = process_video_file.delay(
                            file_path=item.url,
                            task_id=str(uuid.uuid4()),
                            config={**item_metadata, 'is_youtube': True}
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
