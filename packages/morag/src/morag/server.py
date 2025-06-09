"""FastAPI server for MoRAG system."""

import asyncio
import base64
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import structlog
import os

from morag.api import MoRAGAPI
from morag_services import ServiceConfig
from morag_core.models import ProcessingResult, IngestionResponse, BatchIngestionResponse, TaskStatusResponse
from morag.utils.file_upload import get_upload_handler, FileUploadError, validate_temp_directory_access
from morag.services.cleanup_service import start_cleanup_service, stop_cleanup_service, force_cleanup
from morag.worker import submit_task_for_user
from morag.services.auth_service import APIKeyService
from morag.middleware.auth import APIKeyAuth
from morag.task_queue import task_queue, TaskStatus as QueueTaskStatus

logger = structlog.get_logger(__name__)


def normalize_content_type(content_type: Optional[str]) -> Optional[str]:
    """Normalize content type from MIME type to MoRAG content type.

    Args:
        content_type: MIME type or MoRAG content type

    Returns:
        Normalized MoRAG content type or None
    """
    if not content_type:
        return None

    content_type = content_type.lower().strip()

    # If it's already a MoRAG content type, return as-is
    morag_types = {'document', 'audio', 'video', 'image', 'web', 'youtube', 'text', 'unknown'}
    if content_type in morag_types:
        return content_type

    # Convert MIME types to MoRAG content types
    mime_to_morag = {
        # Document types
        'application/pdf': 'document',
        'application/msword': 'document',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'document',
        'application/vnd.ms-powerpoint': 'document',
        'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'document',
        'application/vnd.ms-excel': 'document',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'document',
        'text/plain': 'document',
        'text/markdown': 'document',
        'text/html': 'document',
        'application/rtf': 'document',

        # Audio types
        'audio/mpeg': 'audio',
        'audio/mp3': 'audio',
        'audio/wav': 'audio',
        'audio/x-wav': 'audio',
        'audio/mp4': 'audio',
        'audio/m4a': 'audio',
        'audio/x-m4a': 'audio',
        'audio/flac': 'audio',
        'audio/ogg': 'audio',
        'audio/aac': 'audio',

        # Video types
        'video/mp4': 'video',
        'video/avi': 'video',
        'video/x-msvideo': 'video',
        'video/quicktime': 'video',
        'video/x-matroska': 'video',
        'video/webm': 'video',
        'video/x-ms-wmv': 'video',
        'video/x-flv': 'video',

        # Image types
        'image/jpeg': 'image',
        'image/jpg': 'image',
        'image/png': 'image',
        'image/gif': 'image',
        'image/bmp': 'image',
        'image/webp': 'image',
        'image/tiff': 'image',
        'image/svg+xml': 'image',
    }

    return mime_to_morag.get(content_type, 'document')


def normalize_processing_result(result: ProcessingResult) -> ProcessingResult:
    """Normalize ProcessingResult to ensure it has a content attribute.

    Args:
        result: ProcessingResult from any processor

    Returns:
        ProcessingResult with normalized content attribute
    """
    # Import here to avoid circular imports
    from morag_core.models.config import ProcessingResult as CoreProcessingResult

    # If result already has content attribute and it's a proper string, return as-is
    if (hasattr(result, 'content') and
        result.content is not None and
        isinstance(result.content, str) and
        isinstance(result, CoreProcessingResult)):
        return result

    # Get content for API response (prefer JSON from raw_result, fallback to text_content)
    content = ""

    # First try to get JSON content from raw_result for API response
    if hasattr(result, 'raw_result') and result.raw_result is not None:
        raw_result = result.raw_result
        if isinstance(raw_result, dict):
            # Convert JSON to string for API response
            import json
            content = json.dumps(raw_result, indent=2)
        elif hasattr(raw_result, 'content'):
            content = str(raw_result.content)

    # Fallback to text_content if no raw_result available
    if not content and hasattr(result, 'text_content') and result.text_content is not None:
        content = result.text_content

    # Create a new ProcessingResult with content field
    # This handles both Pydantic models and dataclasses
    return CoreProcessingResult(
        success=result.success,
        task_id=getattr(result, 'task_id', 'unknown'),
        source_type=getattr(result, 'content_type', 'unknown'),
        content=content,
        metadata=getattr(result, 'metadata', {}),
        processing_time=getattr(result, 'processing_time', 0.0),
        error_message=getattr(result, 'error_message', None)
    )


def encode_thumbnails_to_base64(thumbnail_paths: List[str]) -> List[str]:
    """Encode thumbnail images to base64 strings.

    Args:
        thumbnail_paths: List of paths to thumbnail files

    Returns:
        List of base64 encoded thumbnail strings
    """
    encoded_thumbnails = []
    for path in thumbnail_paths:
        try:
            if Path(path).exists():
                with open(path, 'rb') as f:
                    image_data = f.read()
                    encoded = base64.b64encode(image_data).decode('utf-8')
                    # Add data URL prefix for direct use in HTML
                    mime_type = 'image/jpeg'  # Default to JPEG
                    if path.lower().endswith('.png'):
                        mime_type = 'image/png'
                    elif path.lower().endswith('.gif'):
                        mime_type = 'image/gif'
                    elif path.lower().endswith('.webp'):
                        mime_type = 'image/webp'

                    encoded_thumbnails.append(f"data:{mime_type};base64,{encoded}")
        except Exception as e:
            logger.warning("Failed to encode thumbnail", path=path, error=str(e))

    return encoded_thumbnails


# Pydantic models for API
class ProcessURLRequest(BaseModel):
    url: str
    content_type: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    gpu: Optional[bool] = False  # NEW: GPU worker flag


class ProcessBatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None


class ProcessingResultResponse(BaseModel):
    success: bool
    content: str
    metadata: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None
    thumbnails: Optional[List[str]] = None  # Base64 encoded thumbnails


# Ingest API models
class IngestFileRequest(BaseModel):
    source_type: Optional[str] = None  # Auto-detect if not provided
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    use_docling: Optional[bool] = False


class IngestURLRequest(BaseModel):
    source_type: Optional[str] = None  # Auto-detect if not provided
    url: str
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class IngestBatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    webhook_url: Optional[str] = None


class IngestResponse(BaseModel):
    task_id: str
    status: str = "pending"
    message: str
    estimated_time: Optional[int] = None


class BatchIngestResponse(BaseModel):
    batch_id: str
    task_ids: List[str]
    total_items: int
    message: str


class TaskStatus(BaseModel):
    task_id: str
    status: str
    progress: float = 0.0
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_time_remaining: Optional[int] = None


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Create FastAPI application."""

    # Initialize MoRAG API lazily to avoid settings validation at import time
    morag_api = None

    # Initialize authentication services (HTTP workers use in-memory storage)
    api_key_service = APIKeyService()
    auth = APIKeyAuth(api_key_service)

    def get_morag_api() -> MoRAGAPI:
        """Get or create MoRAG API instance."""
        nonlocal morag_api
        if morag_api is None:
            morag_api = MoRAGAPI(config)
        return morag_api

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown."""
        # Startup
        logger.info("MoRAG API server starting up")

        # Validate temp directory access early - fail fast if not accessible
        try:
            validate_temp_directory_access()
            logger.info("Temp directory validation passed")
        except RuntimeError as e:
            logger.error("STARTUP FAILURE: Temp directory validation failed", error=str(e))
            raise RuntimeError(f"Cannot start server: {str(e)}")

        # Start periodic cleanup service
        start_cleanup_service(
            cleanup_interval_hours=1,    # Run cleanup every hour
            max_file_age_hours=24,       # Files older than 24 hours are eligible for cleanup
            max_disk_usage_mb=10000       # Aggressive cleanup if temp files exceed 10GB
        )
        logger.info("Periodic cleanup service started")

        yield

        # Shutdown
        logger.info("MoRAG API server shutting down")
        stop_cleanup_service()
        logger.info("Periodic cleanup service stopped")
        await get_morag_api().cleanup()
        logger.info("MoRAG API server shut down")

    app = FastAPI(
        title="MoRAG API",
        description="""
        Modular Retrieval Augmented Generation System

        ## Features
        - **Processing Endpoints**: Process content and return results immediately
        - **Ingestion Endpoints**: Process content and store in vector database for retrieval
        - **Task Management**: Track processing status and manage background tasks
        - **Search**: Query stored content using vector similarity

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

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"message": "MoRAG API", "version": "0.1.0"}
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            status = await get_morag_api().health_check()
            return JSONResponse(content=status)
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/process/url", response_model=ProcessingResultResponse, tags=["Processing"])
    async def process_url(http_request: Request, request: ProcessURLRequest):
        """Process content from a URL."""
        try:
            # Check for API key authentication and GPU routing
            user_id = await auth.get_user_id(http_request)

            if request.gpu and user_id:
                # Route to GPU worker for authenticated user
                logger.info("Routing URL to GPU worker",
                           url=request.url,
                           user_id=user_id,
                           content_type=request.content_type)

                # For now, process locally but log the routing decision
                # TODO: Implement remote worker task submission
                result = await get_morag_api().process_url(
                    request.url,
                    request.content_type,
                    request.options
                )

                logger.info("GPU worker URL processing completed (local fallback)",
                           url=request.url,
                           user_id=user_id,
                           success=result.success)
            elif request.gpu and not user_id:
                # GPU requested but no authentication - fallback to local
                logger.warning("GPU processing requested without authentication, falling back to local",
                             url=request.url)
                result = await get_morag_api().process_url(
                    request.url,
                    request.content_type,
                    request.options
                )
            else:
                # Standard local processing
                result = await get_morag_api().process_url(
                    request.url,
                    request.content_type,
                    request.options
                )

            result = normalize_processing_result(result)
            return ProcessingResultResponse(
                success=result.success,
                content=result.content,
                metadata=result.metadata,
                processing_time=result.processing_time,
                error_message=result.error_message
            )
        except Exception as e:
            logger.error("URL processing failed", url=request.url, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/process/file", response_model=ProcessingResultResponse, tags=["Processing"])
    async def process_file(
        request: Request,
        file: UploadFile = File(...),
        content_type: Optional[str] = Form(default=None),
        options: Optional[str] = Form(default=None),  # JSON string
        gpu: Optional[bool] = Form(default=False)  # NEW: GPU worker flag
    ):
        """Process content from an uploaded file."""
        temp_path = None
        try:
            # Parse options if provided
            parsed_options = None
            if options:
                import json
                try:
                    parsed_options = json.loads(options)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in options", options=options, error=str(e))
                    raise HTTPException(status_code=400, detail=f"Invalid JSON in options: {str(e)}")

            # Save uploaded file using secure file upload handler
            upload_handler = get_upload_handler()
            try:
                temp_path = await upload_handler.save_upload(file)
                logger.info("File uploaded successfully",
                           filename=file.filename,
                           temp_path=str(temp_path),
                           content_type=content_type)
            except FileUploadError as e:
                logger.error("File upload validation failed",
                           filename=file.filename,
                           error=str(e))
                raise HTTPException(status_code=400, detail=str(e))

            # Process the file
            try:
                # Normalize content type from MIME type to MoRAG content type
                normalized_content_type = normalize_content_type(content_type)

                # Check for API key authentication and GPU routing
                user_id = await auth.get_user_id(request)

                if gpu and user_id:
                    # Route to GPU worker for authenticated user
                    logger.info("Routing to GPU worker",
                               filename=file.filename,
                               user_id=user_id,
                               content_type=normalized_content_type)

                    # For GPU workers, we need to provide a download URL instead of file path
                    # For now, process locally but log the routing decision
                    # TODO: Implement HTTP file transfer for remote workers
                    result = await get_morag_api().process_file(
                        temp_path,
                        normalized_content_type,
                        parsed_options
                    )

                    logger.info("GPU worker processing completed (local fallback)",
                               filename=file.filename,
                               user_id=user_id,
                               success=result.success,
                               processing_time=result.processing_time)
                elif gpu and not user_id:
                    # GPU requested but no authentication - fallback to local
                    logger.warning("GPU processing requested without authentication, falling back to local",
                                 filename=file.filename)
                    result = await get_morag_api().process_file(
                        temp_path,
                        normalized_content_type,
                        parsed_options
                    )
                else:
                    # Standard local processing
                    result = await get_morag_api().process_file(
                        temp_path,
                        normalized_content_type,
                        parsed_options
                    )

                logger.info("File processing completed",
                           filename=file.filename,
                           success=result.success,
                           processing_time=result.processing_time)

                result = normalize_processing_result(result)

                # Handle thumbnails if requested and available
                thumbnails = None
                if parsed_options and parsed_options.get('include_thumbnails', False):
                    # Get thumbnails from the original result before normalization
                    original_result = getattr(result, 'raw_result', None)
                    if original_result:
                        # Check if it's a dictionary (from services)
                        if isinstance(original_result, dict):
                            thumbnail_paths = original_result.get('thumbnails', [])
                            if thumbnail_paths:
                                thumbnails = encode_thumbnails_to_base64(thumbnail_paths)
                        # Check if it has thumbnails attribute (from processors)
                        elif hasattr(original_result, 'thumbnails'):
                            thumbnail_paths = [str(p) for p in original_result.thumbnails]
                            if thumbnail_paths:
                                thumbnails = encode_thumbnails_to_base64(thumbnail_paths)

                    # Also check extracted_files for thumbnails
                    if not thumbnails and hasattr(result, 'extracted_files'):
                        thumbnail_paths = [f for f in result.extracted_files if 'thumb' in f.lower()]
                        if thumbnail_paths:
                            thumbnails = encode_thumbnails_to_base64(thumbnail_paths)

                return ProcessingResultResponse(
                    success=result.success,
                    content=result.content,
                    metadata=result.metadata,
                    processing_time=result.processing_time,
                    error_message=result.error_message,
                    thumbnails=thumbnails
                )
            except Exception as e:
                logger.error("File processing failed",
                           filename=file.filename,
                           temp_path=str(temp_path),
                           error=str(e))
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error("Unexpected error in file upload endpoint",
                        filename=getattr(file, 'filename', 'unknown'),
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        finally:
            # Clean up temporary file immediately after processing
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug("Cleaned up temporary file", temp_path=str(temp_path))
                except Exception as e:
                    logger.warning("Failed to clean up temporary file",
                                 temp_path=str(temp_path),
                                 error=str(e))
    
    @app.post("/process/web", response_model=ProcessingResultResponse, tags=["Processing"])
    async def process_web_page(request: ProcessURLRequest):
        """Process a web page."""
        try:
            result = await get_morag_api().process_web_page(request.url, request.options)
            result = normalize_processing_result(result)
            return ProcessingResultResponse(
                success=result.success,
                content=result.content,
                metadata=result.metadata,
                processing_time=result.processing_time,
                error_message=result.error_message
            )
        except Exception as e:
            logger.error("Web page processing failed", url=request.url, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/process/youtube", response_model=ProcessingResultResponse, tags=["Processing"])
    async def process_youtube_video(request: ProcessURLRequest):
        """Process a YouTube video."""
        try:
            result = await get_morag_api().process_youtube_video(request.url, request.options)
            result = normalize_processing_result(result)
            return ProcessingResultResponse(
                success=result.success,
                content=result.content,
                metadata=result.metadata,
                processing_time=result.processing_time,
                error_message=result.error_message
            )
        except Exception as e:
            logger.error("YouTube processing failed", url=request.url, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/process/batch", tags=["Processing"])
    async def process_batch(request: ProcessBatchRequest):
        """Process multiple items in batch."""
        try:
            results = await get_morag_api().process_batch(request.items, request.options)
            return [
                ProcessingResultResponse(
                    success=result.success,
                    content=normalize_processing_result(result).content,
                    metadata=result.metadata,
                    processing_time=result.processing_time,
                    error_message=result.error_message
                ) for result in results
            ]
        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/search", tags=["Search"])
    async def search_similar(request: SearchRequest):
        """Search for similar content."""
        try:
            results = await get_morag_api().search(
                request.query,
                request.limit,
                request.filters
            )
            return {"results": results}
        except Exception as e:
            logger.error("Search failed", query=request.query, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Ingest API endpoints
    @app.post("/api/v1/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
    async def ingest_file(
        http_request: Request,
        source_type: Optional[str] = Form(default=None),  # Auto-detect if not provided
        file: UploadFile = File(...),
        document_id: Optional[str] = Form(default=None),  # Custom document identifier
        replace_existing: bool = Form(default=False),  # Whether to replace existing document
        webhook_url: Optional[str] = Form(default=None),
        metadata: Optional[str] = Form(default=None),  # JSON string
        use_docling: Optional[bool] = Form(default=False),
        chunk_size: Optional[int] = Form(default=None),  # Use default from settings if not provided
        chunk_overlap: Optional[int] = Form(default=None),  # Use default from settings if not provided
        chunking_strategy: Optional[str] = Form(default=None)  # paragraph, sentence, word, character, etc.
    ):
        """Ingest and process a file, storing results in vector database."""
        temp_path = None
        try:
            # Check for authentication to determine worker routing
            user_id = await auth.get_user_id(http_request)

            # Parse metadata if provided
            parsed_metadata = None
            if metadata:
                try:
                    parsed_metadata = json.loads(metadata)
                except json.JSONDecodeError as e:
                    logger.error("Invalid JSON in metadata", metadata=metadata, error=str(e))
                    raise HTTPException(status_code=400, detail=f"Invalid JSON in metadata: {str(e)}")

            # Save uploaded file using secure file upload handler
            upload_handler = get_upload_handler()
            try:
                temp_path = await upload_handler.save_upload(file)

                # Auto-detect source type if not provided
                if not source_type:
                    source_type = get_morag_api()._detect_content_type_from_file(temp_path)
                    logger.info("Auto-detected content type",
                               filename=file.filename,
                               detected_type=source_type)

                logger.info("File uploaded for ingestion",
                           filename=file.filename,
                           temp_path=str(temp_path),
                           source_type=source_type)
            except FileUploadError as e:
                logger.error("File upload validation failed",
                           filename=file.filename,
                           error=str(e))
                raise HTTPException(status_code=400, detail=str(e))

            # Validate chunk size parameters
            if chunk_size is not None and (chunk_size < 500 or chunk_size > 16000):
                raise HTTPException(
                    status_code=400,
                    detail="chunk_size must be between 500 and 16000 characters"
                )

            if chunk_overlap is not None and (chunk_overlap < 0 or chunk_overlap > 1000):
                raise HTTPException(
                    status_code=400,
                    detail="chunk_overlap must be between 0 and 1000 characters"
                )

            # Validate document ID if provided
            if document_id:
                import re
                if not re.match(r'^[a-zA-Z0-9_-]+$', document_id):
                    raise HTTPException(
                        status_code=400,
                        detail="Document ID must contain only alphanumeric characters, hyphens, and underscores"
                    )

            # Create task parameters
            parameters = {
                "file_path": str(temp_path),
                "content_type": source_type,
                "options": {
                    "document_id": document_id,
                    "replace_existing": replace_existing,
                    "webhook_url": webhook_url or "",
                    "metadata": parsed_metadata or {},
                    "use_docling": use_docling,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "chunking_strategy": chunking_strategy,
                    "store_in_vector_db": True
                }
            }

            # Submit task to queue for async processing
            task_id = await task_queue.submit_task(
                task_type="process_file",
                parameters=parameters,
                user_id=user_id
            )

            logger.info("File ingestion task submitted",
                       task_id=task_id,
                       filename=file.filename,
                       source_type=source_type,
                       user_id=user_id)

            # Note: temp file will be cleaned up by the worker after processing
            return IngestResponse(
                task_id=task_id,
                status="pending",
                message=f"File ingestion task submitted for {file.filename}",
                estimated_time=120  # Estimated processing time in seconds
            )

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error("Unexpected error in file ingestion endpoint",
                        filename=getattr(file, 'filename', 'unknown'),
                        error=str(e))
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        finally:
            # Cleanup is handled in the main processing block
            pass

    @app.post("/api/v1/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
    async def ingest_url(http_request: Request, request: IngestURLRequest):
        """Ingest and process content from URL, storing results in vector database."""
        try:
            # Check for authentication to determine worker routing
            user_id = await auth.get_user_id(http_request)

            # Auto-detect source type if not provided
            source_type = request.source_type
            if not source_type:
                source_type = get_morag_api()._detect_content_type(request.url)
                logger.info("Auto-detected content type for URL",
                           url=request.url,
                           detected_type=source_type)

            # Create task parameters
            parameters = {
                "url": request.url,
                "content_type": source_type,
                "options": {
                    "webhook_url": request.webhook_url or "",
                    "metadata": request.metadata or {},
                    "store_in_vector_db": True
                }
            }

            # Submit task to queue for async processing
            task_id = await task_queue.submit_task(
                task_type="process_url",
                parameters=parameters,
                user_id=user_id
            )

            logger.info("URL ingestion task submitted",
                       task_id=task_id,
                       url=request.url,
                       source_type=source_type,
                       user_id=user_id)

            return IngestResponse(
                task_id=task_id,
                status="pending",
                message=f"URL ingestion task submitted for {request.url}",
                estimated_time=60  # Estimated processing time in seconds
            )

        except Exception as e:
            logger.error("URL ingestion failed", url=request.url, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/ingest/batch", response_model=BatchIngestResponse, tags=["Ingestion"])
    async def ingest_batch(request: IngestBatchRequest):
        """Ingest and process multiple items in batch, storing results in vector database."""
        try:
            # Create batch options with sanitized inputs
            options = {
                "webhook_url": request.webhook_url or "",  # Ensure string, not None
                "store_in_vector_db": True  # Key difference from process endpoints
            }

            # Process directly (no background tasks)
            import uuid
            batch_id = f"batch-{uuid.uuid4().hex[:8]}"
            task_id = str(uuid.uuid4())

            logger.info("Starting direct batch ingestion",
                       task_id=task_id,
                       batch_id=batch_id,
                       item_count=len(request.items))

            # Process the batch directly
            results = await get_morag_api().process_batch(request.items, options)

            successful_items = sum(1 for r in results if r.success)

            if successful_items > 0:
                return BatchIngestResponse(
                    batch_id=batch_id,
                    task_ids=[task_id],
                    total_items=len(request.items),
                    message=f"Batch ingestion completed: {successful_items}/{len(request.items)} items successful"
                )
            else:
                raise HTTPException(
                    status_code=500,
                    detail="All batch items failed to process"
                )

        except Exception as e:
            logger.error("Batch ingestion failed", item_count=len(request.items), error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Task status endpoints
    @app.get("/api/v1/status/{task_id}", response_model=TaskStatus, tags=["Task Management"])
    async def get_task_status(task_id: str):
        """Get the status of a processing task."""
        try:
            logger.info("Task status requested", task_id=task_id)

            # Get task status from queue
            task_status = await task_queue.get_task_status(task_id)

            if not task_status:
                raise HTTPException(status_code=404, detail="Task not found")

            return TaskStatus(
                task_id=task_status["task_id"],
                status=task_status["status"],
                progress=task_status["progress"],
                message=task_status["message"],
                result=task_status["result"],
                error=task_status["error"],
                created_at=task_status["created_at"],
                started_at=task_status["started_at"],
                completed_at=task_status["completed_at"]
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get task status", task_id=task_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/status/", tags=["Task Management"])
    async def list_active_tasks():
        """Get all currently active tasks (HTTP-only mode)."""
        try:
            # In HTTP-only mode, tasks are processed immediately
            logger.info("Active tasks requested (HTTP-only mode)")

            return {
                "active_tasks": [],
                "count": 0,
                "message": "HTTP-only mode: tasks are processed immediately"
            }

        except Exception as e:
            logger.error("Failed to list active tasks", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/status/stats/queues", tags=["Task Management"])
    async def get_queue_stats():
        """Get processing queue statistics (HTTP-only mode)."""
        try:
            # In HTTP-only mode, no queues exist
            logger.info("Queue stats requested (HTTP-only mode)")

            return {
                "pending": 0,
                "active": 0,
                "completed": 0,
                "failed": 0,
                "message": "HTTP-only mode: no queues, tasks processed immediately"
            }

        except Exception as e:
            logger.error("Failed to get queue stats", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v1/ingest/{task_id}", tags=["Task Management"])
    async def cancel_task(task_id: str):
        """Cancel a running or pending task (HTTP-only mode)."""
        try:
            # In HTTP-only mode, tasks are processed immediately and cannot be cancelled
            logger.info("Task cancellation requested (HTTP-only mode)", task_id=task_id)

            return {
                "message": f"Task {task_id} cannot be cancelled (HTTP-only mode: tasks complete immediately)"
            }

        except Exception as e:
            logger.error("Failed to cancel task", task_id=task_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/admin/cleanup", tags=["Administration"])
    async def force_temp_cleanup():
        """Force immediate cleanup of old temporary files."""
        try:
            deleted_count = force_cleanup()

            logger.info("Manual cleanup completed", files_deleted=deleted_count)

            return {
                "message": f"Cleanup completed successfully",
                "files_deleted": deleted_count
            }

        except Exception as e:
            logger.error("Failed to perform manual cleanup", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # API Key Management endpoints
    @app.post("/api/v1/auth/create-key", tags=["Authentication"])
    async def create_api_key(
        user_id: str = Form(...),
        description: str = Form(default=""),
        expires_days: Optional[int] = Form(default=None)
    ):
        """Create a new API key for a user."""
        try:
            api_key = await api_key_service.create_api_key(user_id, description, expires_days)

            logger.info("API key created via endpoint", user_id=user_id, description=description)

            return {
                "api_key": api_key,
                "user_id": user_id,
                "description": description,
                "expires_days": expires_days,
                "message": "API key created successfully"
            }
        except Exception as e:
            logger.error("Failed to create API key", user_id=user_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/auth/validate-key", tags=["Authentication"])
    async def validate_api_key_endpoint(api_key: str = Form(...)):
        """Validate an API key and return user information."""
        try:
            user_data = await api_key_service.validate_api_key(api_key)

            if not user_data:
                raise HTTPException(status_code=401, detail="Invalid API key")

            return {
                "valid": True,
                "user_data": user_data,
                "message": "API key is valid"
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to validate API key", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/auth/queue-info", tags=["Authentication"])
    async def get_queue_info(request: Request):
        """Get queue information for the authenticated user."""
        try:
            user_id = await auth.get_user_id(request)

            if not user_id:
                return {
                    "authenticated": False,
                    "default_queue": api_key_service.get_default_queue_name(),
                    "message": "No authentication provided - using default queue"
                }

            gpu_queue = api_key_service.get_user_queue_name(user_id, "gpu")
            cpu_queue = api_key_service.get_cpu_queue_name(user_id)

            return {
                "authenticated": True,
                "user_id": user_id,
                "gpu_queue": gpu_queue,
                "cpu_queue": cpu_queue,
                "default_queue": api_key_service.get_default_queue_name(),
                "message": "User-specific queues available"
            }
        except Exception as e:
            logger.error("Failed to get queue info", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # File Transfer endpoints for remote workers
    @app.post("/api/v1/files/download", tags=["File Transfer"])
    async def download_file(request: Request, file_request: Dict[str, str]):
        """Download file for remote worker processing."""
        try:
            # Require authentication for file downloads
            user_data = await auth.require_authentication(request)
            user_id = user_data.get("user_id")

            file_path = file_request.get('file_path')
            if not file_path:
                raise HTTPException(status_code=400, detail="file_path required")

            # Validate file exists and is accessible
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="File not found")

            # Security check - ensure file is in allowed directories
            allowed_dirs = ['/app/temp', '/app/uploads', '/tmp']
            if not any(file_path.startswith(d) for d in allowed_dirs):
                raise HTTPException(status_code=403, detail="File access denied")

            logger.info("File download requested",
                       file_path=file_path,
                       user_id=user_id)

            return FileResponse(
                path=file_path,
                filename=os.path.basename(file_path),
                media_type='application/octet-stream'
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error("File download failed", file_path=file_request.get('file_path'), error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/files/upload-result", tags=["File Transfer"])
    async def upload_result(
        request: Request,
        file: Optional[UploadFile] = File(None),
        result_data: str = Form(...)
    ):
        """Upload processing result from remote worker."""
        try:
            # Require authentication for result uploads
            user_data = await auth.require_authentication(request)
            user_id = user_data.get("user_id")

            import json
            result = json.loads(result_data)

            # Handle file upload if provided
            if file:
                # Save uploaded file to temp directory
                temp_path = f"/tmp/result_{uuid.uuid4().hex}_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    import shutil
                    shutil.copyfileobj(file.file, buffer)
                result['uploaded_file'] = temp_path

            # Process result (store in database, trigger webhooks, etc.)
            logger.info("Result uploaded from remote worker",
                       result=result,
                       user_id=user_id)

            return {"status": "success", "message": "Result uploaded successfully"}

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Result upload failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Worker status monitoring endpoint (HTTP-only mode)
    @app.get("/api/v1/status/workers", tags=["Task Management"])
    async def get_worker_status():
        """Get current worker and queue status (HTTP-only mode)."""
        try:
            # In HTTP-only mode, there are no background workers or queues
            logger.info("Worker status requested (HTTP-only mode)")

            return {
                "workers": {
                    "total": 1,
                    "gpu": 0,
                    "cpu": 1,
                    "details": {
                        "http-server": {
                            "type": "http",
                            "active_tasks": 0,
                            "max_tasks": 1,
                            "queues": []
                        }
                    }
                },
                "queues": {
                    "http": {
                        "length": 0,
                        "workers": 1
                    }
                },
                "gpu_available": False,
                "message": "HTTP-only mode: tasks processed immediately by server"
            }

        except Exception as e:
            logger.error("Failed to get worker status", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # HTTP Worker Management endpoints
    @app.post("/api/v1/workers/register", tags=["Worker Management"])
    async def register_worker(
        request: Request,
        worker_id: str = Form(...),
        worker_type: str = Form(...),
        max_concurrent_tasks: int = Form(default=1)
    ):
        """Register a new HTTP worker."""
        try:
            # Require authentication for worker registration
            user_data = await auth.require_authentication(request)
            user_id = user_data.get("user_id")
            api_key = request.headers.get("Authorization", "").replace("Bearer ", "")

            success = await task_queue.register_worker(
                worker_id=worker_id,
                worker_type=worker_type,
                api_key=api_key,
                user_id=user_id,
                max_concurrent_tasks=max_concurrent_tasks
            )

            if success:
                return {
                    "status": "success",
                    "message": f"Worker {worker_id} registered successfully",
                    "worker_id": worker_id,
                    "user_id": user_id
                }
            else:
                raise HTTPException(status_code=400, detail="Failed to register worker")

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Worker registration failed", worker_id=worker_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v1/workers/{worker_id}", tags=["Worker Management"])
    async def unregister_worker(worker_id: str, request: Request):
        """Unregister an HTTP worker."""
        try:
            # Require authentication for worker unregistration
            await auth.require_authentication(request)

            success = await task_queue.unregister_worker(worker_id)

            if success:
                return {
                    "status": "success",
                    "message": f"Worker {worker_id} unregistered successfully"
                }
            else:
                raise HTTPException(status_code=404, detail="Worker not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Worker unregistration failed", worker_id=worker_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/workers/{worker_id}/tasks/next", tags=["Worker Management"])
    async def get_next_task(worker_id: str, request: Request):
        """Get next available task for a worker."""
        try:
            # Require authentication for task retrieval
            await auth.require_authentication(request)

            task = await task_queue.get_next_task(worker_id)

            if task:
                return task
            else:
                # No tasks available - return 204 No Content
                from fastapi import Response
                return Response(status_code=204)

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get next task", worker_id=worker_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/workers/{worker_id}/tasks/{task_id}/status", tags=["Worker Management"])
    async def update_task_status(
        worker_id: str,
        task_id: str,
        request: Request,
        status: str = Form(...),
        result: Optional[str] = Form(default=None),  # JSON string
        error_message: Optional[str] = Form(default=None)
    ):
        """Update task status from worker."""
        try:
            # Require authentication for status updates
            await auth.require_authentication(request)

            # Parse result if provided
            parsed_result = None
            if result:
                import json
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON in result: {str(e)}")

            # Convert string status to enum
            try:
                task_status = QueueTaskStatus(status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

            success = await task_queue.update_task_status(
                task_id=task_id,
                status=task_status,
                result=parsed_result,
                error_message=error_message
            )

            if success:
                return {
                    "status": "success",
                    "message": f"Task {task_id} status updated to {status}"
                }
            else:
                raise HTTPException(status_code=404, detail="Task not found")

        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to update task status",
                        worker_id=worker_id,
                        task_id=task_id,
                        error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    """Main entry point for the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MoRAG API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--config", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            import json
            with open(config_path) as f:
                config_data = json.load(f)
                config = ServiceConfig(**config_data)
    
    # Create app
    app = create_app(config)
    
    # Run server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
