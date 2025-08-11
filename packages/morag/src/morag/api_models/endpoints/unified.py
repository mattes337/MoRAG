"""Unified processing endpoint for MoRAG API."""

import time
import json
from typing import Optional, Dict, Any
from pathlib import Path
import structlog

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from morag.api import MoRAGAPI
from morag.api_models.models import (
    UnifiedProcessRequest, UnifiedProcessResponse,
    ProcessingOptions, DatabaseConfig, WebhookConfig
)
from morag.api_models.utils import (
    download_remote_file, normalize_content_type, 
    normalize_processing_result, encode_thumbnails_to_base64
)
from morag.utils.file_upload import get_upload_handler, FileUploadError
from morag.services.document_deduplication_service import get_deduplication_service
from morag_document.services.markitdown_service import MarkitdownService
from morag.tasks.enhanced_processing_task import enhanced_process_ingest_task
from morag.ingest_tasks import ingest_file_task, ingest_url_task, ingest_batch_task

logger = structlog.get_logger(__name__)

unified_router = APIRouter(prefix="/api/v1", tags=["Unified Processing"])


def setup_unified_endpoints(morag_api_getter):
    """Setup unified processing endpoint with MoRAG API getter function."""
    
    @unified_router.post("/process", response_model=UnifiedProcessResponse)
    async def unified_process(
        # File upload (for source_type='file')
        file: Optional[UploadFile] = File(default=None),
        
        # Request data as JSON string
        request_data: str = Form(..., description="JSON string containing UnifiedProcessRequest data")
    ):
        """Unified processing endpoint that handles convert, process, and ingest modes.
        
        This single endpoint replaces all previous processing/ingestion endpoints:
        - mode='convert': Fast markdown conversion for UI preview
        - mode='process': Full processing with immediate results
        - mode='ingest': Full processing + vector storage with background tasks
        
        Source types:
        - source_type='file': Upload file via multipart form
        - source_type='url': Process content from URL
        - source_type='batch': Process multiple items
        """
        start_time = time.time()
        temp_path = None
        
        try:
            # Parse request data
            try:
                request_dict = json.loads(request_data)
                request = UnifiedProcessRequest(**request_dict)
            except (json.JSONDecodeError, ValueError) as e:
                logger.error("Invalid request data", request_data=request_data, error=str(e))
                raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")
            
            # Validate request based on source type
            if request.source_type == "file" and not file:
                raise HTTPException(status_code=400, detail="File upload required for source_type='file'")
            elif request.source_type == "url" and not request.url:
                raise HTTPException(status_code=400, detail="URL required for source_type='url'")
            elif request.source_type == "batch" and not request.items:
                raise HTTPException(status_code=400, detail="Items array required for source_type='batch'")
            
            # Validate webhook config for async modes
            if request.mode == "ingest" and not request.webhook_config:
                raise HTTPException(status_code=400, detail="Webhook configuration required for mode='ingest'")
            
            logger.info("Processing unified request",
                       mode=request.mode,
                       source_type=request.source_type,
                       document_id=request.document_id)
            
            # Route to appropriate handler based on mode and source type
            if request.mode == "convert":
                return await handle_convert_mode(request, file, start_time)
            elif request.mode == "process":
                return await handle_process_mode(request, file, morag_api_getter, start_time)
            elif request.mode == "ingest":
                return await handle_ingest_mode(request, file, start_time)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
                
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            logger.error("Unified processing failed", 
                        error=str(e),
                        processing_time_ms=processing_time_ms)
            
            # Clean up temp file on error
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 cleanup_error=str(cleanup_error))
            
            return UnifiedProcessResponse(
                success=False,
                mode=request.mode if 'request' in locals() else "unknown",
                error_message=str(e),
                processing_time_ms=processing_time_ms
            )
    
    return unified_router


async def handle_convert_mode(request: UnifiedProcessRequest, file: Optional[UploadFile], start_time: float) -> UnifiedProcessResponse:
    """Handle convert mode - fast markdown conversion."""
    temp_path = None
    
    try:
        if request.source_type == "file":
            # Handle file upload
            upload_handler = get_upload_handler()
            temp_path = await upload_handler.save_upload(file)
            
            # Get file info
            file_size = temp_path.stat().st_size
            file_extension = temp_path.suffix.lower().lstrip('.')
            
            # Check for duplicates if document_id provided
            document_id = request.document_id
            if document_id:
                dedup_service = await get_deduplication_service()
                if await dedup_service.document_exists(document_id) and not request.replace_existing:
                    processing_time_ms = (time.time() - start_time) * 1000
                    return UnifiedProcessResponse(
                        success=True,
                        mode="convert",
                        content="",
                        metadata={"duplicate": True, "document_id": document_id},
                        processing_time_ms=processing_time_ms,
                        document_id=document_id,
                        message="Document already exists (use replace_existing=true to override)"
                    )
            
            # Convert to markdown based on file type
            conversion_metadata = {
                "original_format": file_extension,
                "file_size_bytes": file_size,
                "original_filename": file.filename
            }
            
            if file_extension in ['pdf', 'doc', 'docx', 'txt', 'md', 'html', 'rtf']:
                # Use markitdown service for documents
                markitdown_service = MarkitdownService()
                markdown_content = await markitdown_service.convert_file(temp_path)
                
                if file_extension == 'pdf':
                    # Estimate page count
                    estimated_pages = max(1, file_size // 50000)
                    conversion_metadata["estimated_page_count"] = estimated_pages
                    
            else:
                # For other file types, we'd need to implement conversion logic
                # For now, return an error for unsupported types in convert mode
                raise HTTPException(status_code=400, detail=f"Convert mode not supported for file type: {file_extension}")
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            return UnifiedProcessResponse(
                success=True,
                mode="convert",
                content=markdown_content,
                metadata=conversion_metadata,
                processing_time_ms=processing_time_ms,
                document_id=document_id
            )
            
        else:
            # URL and batch modes not supported for convert mode
            raise HTTPException(status_code=400, detail=f"Convert mode only supports source_type='file'")
            
    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as cleanup_error:
                logger.warning("Failed to clean up temp file",
                             temp_path=str(temp_path),
                             cleanup_error=str(cleanup_error))


async def handle_process_mode(request: UnifiedProcessRequest, file: Optional[UploadFile], morag_api_getter, start_time: float) -> UnifiedProcessResponse:
    """Handle process mode - full processing with immediate results."""
    temp_path = None
    
    try:
        # Convert processing options to dict
        options = {}
        if request.processing_options:
            options.update(request.processing_options.dict(exclude_none=True))
        if request.metadata:
            options["metadata"] = request.metadata
        
        if request.source_type == "file":
            # Handle file upload
            upload_handler = get_upload_handler()
            temp_path = await upload_handler.save_upload(file)
            
            # Normalize content type
            normalized_content_type = normalize_content_type(request.content_type)
            
            # Process the file
            result = await morag_api_getter().process_file(
                str(temp_path),
                normalized_content_type,
                options
            )
            
        elif request.source_type == "url":
            # Process URL
            result = await morag_api_getter().process_url(
                request.url,
                request.content_type,
                options
            )
            
        elif request.source_type == "batch":
            # Process batch
            result = await morag_api_getter().process_batch(request.items, options)
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {request.source_type}")
        
        # Normalize result
        result = normalize_processing_result(result)
        processing_time_ms = (time.time() - start_time) * 1000
        
        return UnifiedProcessResponse(
            success=result.success,
            mode="process",
            content=result.content,
            metadata=result.metadata,
            processing_time_ms=processing_time_ms,
            error_message=result.error_message,
            warnings=result.warnings,
            thumbnails=result.thumbnails,
            document_id=request.document_id
        )
        
    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as cleanup_error:
                logger.warning("Failed to clean up temp file",
                             temp_path=str(temp_path),
                             cleanup_error=str(cleanup_error))


async def handle_ingest_mode(request: UnifiedProcessRequest, file: Optional[UploadFile], start_time: float) -> UnifiedProcessResponse:
    """Handle ingest mode - full processing + vector storage with background tasks."""
    temp_path = None
    
    try:
        if request.source_type == "file":
            # Handle file upload
            upload_handler = get_upload_handler()
            temp_path = await upload_handler.save_upload(file)
            
            # Create task options
            options = {
                "source_type": normalize_content_type(request.content_type) or "document",
                "document_id": request.document_id,
                "replace_existing": request.replace_existing,
                "webhook_url": request.webhook_config.url if request.webhook_config else "",
                "webhook_auth_token": request.webhook_config.auth_token if request.webhook_config else None,
                "metadata": request.metadata or {},
                "remote": False,
                "databases": [request.database_config.dict(exclude_none=True)] if request.database_config else [],
                "language": request.processing_options.language if request.processing_options else None,
                "chunking_strategy": request.processing_options.chunking_strategy if request.processing_options else None,
                "chunk_size": request.processing_options.chunk_size if request.processing_options else None,
                "chunk_overlap": request.processing_options.chunk_overlap if request.processing_options else None,
            }
            
            # Submit background task
            task = ingest_file_task.delay(str(temp_path), options)
            
            # Estimate processing time based on file size
            file_size_mb = temp_path.stat().st_size / (1024 * 1024)
            estimated_time = max(60, int(file_size_mb * 30))  # 30 seconds per MB, minimum 1 minute
            
            return UnifiedProcessResponse(
                success=True,
                mode="ingest",
                task_id=task.id,
                document_id=request.document_id,
                estimated_time_seconds=estimated_time,
                status_url=f"/api/v1/status/{task.id}",
                message=f"Processing started for '{file.filename}'. Webhook notifications will be sent to {request.webhook_config.url}"
            )
            
        elif request.source_type == "url":
            # Create task options for URL
            options = {
                "source_type": request.content_type or "web",
                "webhook_url": request.webhook_config.url if request.webhook_config else "",
                "metadata": request.metadata or {},
                "remote": False,
                "databases": [request.database_config.dict(exclude_none=True)] if request.database_config else [],
                "language": request.processing_options.language if request.processing_options else None
            }
            
            # Submit URL task
            task = ingest_url_task.delay(request.url, options)
            
            return UnifiedProcessResponse(
                success=True,
                mode="ingest",
                task_id=task.id,
                document_id=request.document_id,
                estimated_time_seconds=180,  # 3 minutes estimate
                status_url=f"/api/v1/status/{task.id}",
                message=f"URL '{request.url}' queued for ingestion"
            )
            
        elif request.source_type == "batch":
            # Create batch options
            options = {
                "webhook_url": request.webhook_config.url if request.webhook_config else "",
                "remote": False,
                "databases": [request.database_config.dict(exclude_none=True)] if request.database_config else [],
                "language": request.processing_options.language if request.processing_options else None
            }
            
            # Submit batch task
            task = ingest_batch_task.delay(request.items, options)
            
            return UnifiedProcessResponse(
                success=True,
                mode="ingest",
                task_id=task.id,
                estimated_time_seconds=len(request.items) * 60,  # 1 minute per item estimate
                status_url=f"/api/v1/status/{task.id}",
                message=f"Batch of {len(request.items)} items queued for ingestion"
            )
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid source_type: {request.source_type}")
            
    except Exception as e:
        # Clean up temp file on error
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except Exception as cleanup_error:
                logger.warning("Failed to clean up temp file after error",
                             temp_path=str(temp_path),
                             cleanup_error=str(cleanup_error))
        raise
