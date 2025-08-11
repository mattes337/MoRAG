"""Ingestion endpoints for MoRAG API."""

from typing import Optional
from pathlib import Path
import structlog

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks

from morag.api_models.models import (
    IngestFileRequest, IngestURLRequest, IngestBatchRequest, 
    IngestRemoteFileRequest, IngestResponse, BatchIngestResponse
)
from morag.api_models.utils import download_remote_file, normalize_content_type
from morag.utils.file_upload import get_upload_handler, FileUploadError
from morag.ingest_tasks import ingest_file_task, ingest_url_task, ingest_batch_task

logger = structlog.get_logger(__name__)

ingestion_router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


def setup_ingestion_endpoints(morag_api_getter):
    """Setup ingestion endpoints with MoRAG API getter function."""
    
    @ingestion_router.post("/file", response_model=IngestResponse)
    async def ingest_file(
        file: UploadFile = File(...),
        request_data: str = Form(...)  # JSON string containing IngestFileRequest data
    ):
        """Ingest and process a file, storing results in vector database."""
        temp_path = None
        try:
            # Parse request data
            import json
            try:
                request_dict = json.loads(request_data)
                request = IngestFileRequest(**request_dict)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Invalid request data", request_data=request_data, error=str(e))
                raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")

            # Get upload handler
            upload_handler = get_upload_handler()
            
            # Save uploaded file to temp directory
            temp_path = await upload_handler.save_upload(file)
            
            # Normalize content type
            source_type = request.source_type
            if not source_type:
                source_type = normalize_content_type(file.content_type)
            
            # Create task options
            options = {
                "source_type": source_type,
                "document_id": request.document_id,
                "replace_existing": request.replace_existing,
                "webhook_url": request.webhook_url or "",
                "metadata": request.metadata or {},
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "chunking_strategy": request.chunking_strategy,
                "remote": request.remote or False,
                "databases": [db.dict() if hasattr(db, 'dict') else db for db in (request.databases or [])],
                "language": request.language
            }
            
            # Submit task to Celery
            task = ingest_file_task.delay(str(temp_path), options)
            
            logger.info("File ingestion task submitted",
                       task_id=task.id,
                       filename=file.filename,
                       source_type=source_type)
            
            return IngestResponse(
                task_id=task.id,
                status="pending",
                message=f"File '{file.filename}' queued for ingestion",
                estimated_time=300  # 5 minutes estimate
            )
            
        except FileUploadError as e:
            logger.error("File upload failed", filename=file.filename, error=str(e))
            raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")
        except Exception as e:
            logger.error("File ingestion failed", filename=file.filename, error=str(e))
            # Clean up temp file on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 cleanup_error=str(cleanup_error))
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        finally:
            # Note: Don't clean up temp file here - the background task needs it
            pass

    @ingestion_router.post("/url", response_model=IngestResponse)
    async def ingest_url(request: IngestURLRequest):
        """Ingest and process content from URL, storing results in vector database."""
        try:
            # Auto-detect source type if not provided
            source_type = request.source_type
            if not source_type:
                if 'youtube.com' in request.url or 'youtu.be' in request.url:
                    source_type = 'youtube'
                else:
                    source_type = 'web'
            
            # Create task options
            options = {
                "source_type": source_type,
                "webhook_url": request.webhook_url or "",
                "metadata": request.metadata or {},
                "remote": request.remote or False,
                "databases": [db.dict() if hasattr(db, 'dict') else db for db in (request.databases or [])],
                "language": request.language
            }
            
            # Submit task to Celery
            task = ingest_url_task.delay(request.url, options)
            
            logger.info("URL ingestion task submitted",
                       task_id=task.id,
                       url=request.url,
                       source_type=source_type)
            
            return IngestResponse(
                task_id=task.id,
                status="pending",
                message=f"URL '{request.url}' queued for ingestion",
                estimated_time=180  # 3 minutes estimate
            )

        except Exception as e:
            logger.error("URL ingestion failed", url=request.url, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @ingestion_router.post("/batch", response_model=BatchIngestResponse)
    async def ingest_batch(request: IngestBatchRequest):
        """Ingest and process multiple items in batch, storing results in vector database."""
        try:
            # Create batch options with sanitized inputs
            options = {
                "webhook_url": request.webhook_url or "",  # Ensure string, not None
                "remote": request.remote or False,
                "databases": [db.dict() if hasattr(db, 'dict') else db for db in (request.databases or [])],
                "language": request.language
            }
            
            # Submit batch task to Celery
            task = ingest_batch_task.delay(request.items, options)
            
            logger.info("Batch ingestion task submitted",
                       task_id=task.id,
                       item_count=len(request.items))
            
            return BatchIngestResponse(
                batch_id=task.id,
                task_ids=[task.id],  # Single task for now, could be split later
                total_items=len(request.items),
                message=f"Batch of {len(request.items)} items queued for ingestion"
            )

        except Exception as e:
            logger.error("Batch ingestion failed", item_count=len(request.items), error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @ingestion_router.post("/remote-file", response_model=IngestResponse)
    async def ingest_remote_file(request: IngestRemoteFileRequest):
        """Ingest and process a remote file (UNC path or HTTP/HTTPS URL), storing results in vector database."""
        temp_path = None
        try:
            # Get upload handler for temp directory
            upload_handler = get_upload_handler()
            temp_dir = upload_handler.temp_dir
            
            # Download remote file to temp directory
            temp_path = await download_remote_file(request.file_path, temp_dir)
            
            # Normalize content type
            source_type = request.source_type
            if not source_type:
                # Try to detect from file extension
                file_ext = temp_path.suffix.lower()
                if file_ext in ['.pdf', '.docx', '.doc', '.txt', '.md']:
                    source_type = 'document'
                elif file_ext in ['.mp3', '.wav', '.m4a', '.flac']:
                    source_type = 'audio'
                elif file_ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    source_type = 'video'
                elif file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    source_type = 'image'
                else:
                    source_type = 'document'  # Default fallback
            
            # Create task options
            options = {
                "source_type": source_type,
                "document_id": request.document_id,
                "replace_existing": request.replace_existing,
                "webhook_url": request.webhook_url or "",
                "metadata": request.metadata or {},
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "chunking_strategy": request.chunking_strategy,
                "remote": request.remote or False,
                "databases": [db.dict() if hasattr(db, 'dict') else db for db in (request.databases or [])],
                "language": request.language
            }
            
            # Submit task to Celery
            task = ingest_file_task.delay(str(temp_path), options)
            
            logger.info("Remote file ingestion task submitted",
                       task_id=task.id,
                       remote_path=request.file_path,
                       local_path=str(temp_path),
                       source_type=source_type)
            
            return IngestResponse(
                task_id=task.id,
                status="pending",
                message=f"Remote file '{request.file_path}' queued for ingestion",
                estimated_time=300  # 5 minutes estimate
            )
            
        except Exception as e:
            # Clean up temp file on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 cleanup_error=str(cleanup_error))
            
            logger.error("Remote file ingestion failed",
                        remote_path=request.file_path,
                        error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    return ingestion_router
