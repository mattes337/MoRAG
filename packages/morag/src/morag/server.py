"""FastAPI server for MoRAG system."""

import asyncio
import base64
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import json
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import structlog

from morag.api import MoRAGAPI
from morag_services import ServiceConfig
from morag_core.models import ProcessingResult, IngestionResponse, BatchIngestionResponse, TaskStatusResponse
from morag_graph.models.database_config import DatabaseType, DatabaseConfig
from morag.utils.file_upload import get_upload_handler, FileUploadError, validate_temp_directory_access
from morag.services.cleanup_service import start_cleanup_service, stop_cleanup_service, force_cleanup
from morag.worker import (
    process_file_task, process_url_task, process_web_page_task,
    process_youtube_video_task, process_batch_task, celery_app
)
from morag.ingest_tasks import ingest_file_task, ingest_url_task, ingest_batch_task
from morag.endpoints import remote_jobs_router

logger = structlog.get_logger(__name__)





async def download_remote_file(file_path: str, temp_dir: Path) -> Path:
    """Download a remote file (HTTP/HTTPS URL or UNC path) to local temp directory.

    Args:
        file_path: Remote file path (URL or UNC path)
        temp_dir: Local temporary directory

    Returns:
        Path to downloaded local file

    Raises:
        HTTPException: If download fails
    """
    import aiohttp
    import aiofiles
    from urllib.parse import urlparse
    import os

    # Determine if it's a URL or UNC path
    parsed = urlparse(file_path)
    is_url = parsed.scheme in ('http', 'https')
    is_unc = file_path.startswith('\\\\') or (len(file_path) > 1 and file_path[1] == ':' and '\\' in file_path)

    if not is_url and not is_unc:
        raise HTTPException(status_code=400, detail=f"Invalid remote file path: {file_path}. Must be HTTP/HTTPS URL or UNC path.")

    # Generate local filename
    if is_url:
        filename = Path(parsed.path).name or "downloaded_file"
    else:
        filename = Path(file_path).name

    if not filename or filename == '.':
        filename = f"remote_file_{uuid.uuid4().hex[:8]}"

    local_path = temp_dir / filename

    try:
        if is_url:
            # Download from HTTP/HTTPS URL
            async with aiohttp.ClientSession() as session:
                async with session.get(file_path) as response:
                    if response.status != 200:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Failed to download file from {file_path}: HTTP {response.status}"
                        )

                    # Check content length if available
                    content_length = response.headers.get('content-length')
                    if content_length:
                        try:
                            from morag_core.config import get_settings
                            settings = get_settings()
                            max_size = settings.get_max_upload_size_bytes()
                            if int(content_length) > max_size:
                                raise HTTPException(
                                    status_code=413,
                                    detail=f"Remote file too large: {content_length} bytes exceeds limit of {max_size} bytes"
                                )
                        except Exception as e:
                            logger.warning("Could not check file size limit", error=str(e))

                    # Download file
                    async with aiofiles.open(local_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)

        else:
            # Copy from UNC path
            import shutil
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"UNC file not found: {file_path}")

            # Check file size
            try:
                file_size = os.path.getsize(file_path)
                from morag_core.config import get_settings
                settings = get_settings()
                max_size = settings.get_max_upload_size_bytes()
                if file_size > max_size:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Remote file too large: {file_size} bytes exceeds limit of {max_size} bytes"
                    )
            except Exception as e:
                logger.warning("Could not check UNC file size limit", error=str(e))

            # Copy file
            await asyncio.get_event_loop().run_in_executor(
                None, shutil.copy2, file_path, local_path
            )

        logger.info("Remote file downloaded successfully",
                   remote_path=file_path,
                   local_path=str(local_path),
                   file_size=local_path.stat().st_size)

        return local_path

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error("Failed to download remote file",
                    remote_path=file_path,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to download remote file: {str(e)}")


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
    language: str  # Language code for processing (e.g., en, de, fr) - MANDATORY


class ProcessBatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None
    language: str  # Language code for processing (e.g., en, de, fr) - MANDATORY


class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    filters: Optional[Dict[str, Any]] = None
    database_servers: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional array of database server configurations. If not provided, uses environment defaults."
    )


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
    document_id: Optional[str] = None  # Custom document identifier
    replace_existing: bool = False  # Whether to replace existing document
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    use_docling: Optional[bool] = False
    chunk_size: Optional[int] = None  # Use default from settings if not provided
    chunk_overlap: Optional[int] = None  # Use default from settings if not provided
    chunking_strategy: Optional[str] = None  # paragraph, sentence, word, character, etc.
    remote: Optional[bool] = False  # Use remote processing for audio/video
    databases: Optional[List[DatabaseConfig]] = None  # List of databases to process with
    language: str  # Language code for processing (e.g., en, de, fr) - MANDATORY


class IngestURLRequest(BaseModel):
    source_type: Optional[str] = None  # Auto-detect if not provided
    url: str
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    remote: Optional[bool] = False  # Use remote processing for audio/video
    databases: Optional[List[DatabaseConfig]] = None  # List of databases to process with
    language: str  # Language code for processing (e.g., en, de, fr) - MANDATORY


class IngestBatchRequest(BaseModel):
    items: List[Dict[str, Any]]
    webhook_url: Optional[str] = None
    remote: Optional[bool] = False  # Use remote processing for audio/video
    databases: Optional[List[DatabaseConfig]] = None  # List of databases to process with
    language: str  # Language code for processing (e.g., en, de, fr) - MANDATORY


class IngestRemoteFileRequest(BaseModel):
    file_path: str  # UNC path or HTTP/HTTPS URL
    source_type: Optional[str] = None  # Auto-detect if not provided
    document_id: Optional[str] = None  # Custom document identifier
    replace_existing: bool = False  # Whether to replace existing document
    webhook_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    use_docling: Optional[bool] = False
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunking_strategy: Optional[str] = None
    remote: Optional[bool] = False  # Use remote processing for audio/video
    databases: Optional[List[DatabaseConfig]] = None  # List of databases to process with
    language: str  # Language code for processing (e.g., en, de, fr) - MANDATORY


class ProcessRemoteFileRequest(BaseModel):
    file_path: str  # UNC path or HTTP/HTTPS URL
    content_type: Optional[str] = None  # Auto-detect if not provided
    options: Optional[Dict[str, Any]] = None
    language: str  # Language code for processing (e.g., en, de, fr) - MANDATORY


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
    async def process_url(request: ProcessURLRequest):
        """Process content from a URL."""
        try:
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
        file: UploadFile = File(...),
        content_type: Optional[str] = Form(default=None),
        options: Optional[str] = Form(default=None)  # JSON string
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
            # Handle custom database servers if provided
            if request.database_servers:
                from morag.database_factory import get_qdrant_storages
                qdrant_storages = get_qdrant_storages(request.database_servers)

                if qdrant_storages:
                    # Use first available custom Qdrant storage for search
                    # TODO: In future, could aggregate results from multiple storages
                    custom_storage = qdrant_storages[0]

                    # Perform search using custom storage
                    results = await custom_storage.search_similar(
                        query_vector=None,  # Will need to generate embedding
                        limit=request.limit,
                        score_threshold=0.5,
                        filters=request.filters
                    )
                    return {"results": results}

            # Fall back to default search
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
        file: UploadFile = File(...),
        request_data: str = Form(...)  # JSON string containing IngestFileRequest data
    ):
        """Ingest and process a file, storing results in vector database."""
        temp_path = None
        try:
            # Parse request data JSON
            try:
                request_dict = json.loads(request_data)
                request = IngestFileRequest(**request_dict)
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON in request_data", request_data=request_data, error=str(e))
                raise HTTPException(status_code=400, detail=f"Invalid JSON in request_data: {str(e)}")
            except Exception as e:
                logger.error("Invalid request data", request_data=request_data, error=str(e))
                raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")

            # Save uploaded file using secure file upload handler
            upload_handler = get_upload_handler()
            try:
                temp_path = await upload_handler.save_upload(file)

                # Auto-detect source type if not provided
                source_type = request.source_type
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
            if request.chunk_size is not None and (request.chunk_size < 500 or request.chunk_size > 16000):
                raise HTTPException(
                    status_code=400,
                    detail="chunk_size must be between 500 and 16000 characters"
                )

            if request.chunk_overlap is not None and (request.chunk_overlap < 0 or request.chunk_overlap > 1000):
                raise HTTPException(
                    status_code=400,
                    detail="chunk_overlap must be between 0 and 1000 characters"
                )

            # Validate document ID if provided
            if request.document_id:
                import re
                if not re.match(r'^[a-zA-Z0-9_-]+$', request.document_id):
                    raise HTTPException(
                        status_code=400,
                        detail="Document ID must contain only alphanumeric characters, hyphens, and underscores"
                    )

            # Create task options with sanitized inputs
            options = {
                "document_id": request.document_id,
                "replace_existing": request.replace_existing,
                "webhook_url": request.webhook_url or "",  # Ensure string, not None
                "metadata": request.metadata or {},  # Ensure dict, not None
                "use_docling": request.use_docling,
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "chunking_strategy": request.chunking_strategy,
                "remote": request.remote,  # Remote processing flag
                "databases": request.databases,  # Database configurations for graph processing
                "store_in_vector_db": True  # Key difference from process endpoints
            }

            # Submit to background task queue for processing and storage
            task = ingest_file_task.delay(
                str(temp_path),
                source_type,
                options
            )

            logger.info("File ingestion task created",
                       task_id=task.id,
                       filename=file.filename,
                       source_type=source_type)

            return IngestResponse(
                task_id=task.id,
                status="pending",
                message=f"File ingestion started for {file.filename}",
                estimated_time=60  # Estimate based on content type
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
            # Note: Don't clean up temp file here - the background task needs it
            pass

    @app.post("/api/v1/ingest/url", response_model=IngestResponse, tags=["Ingestion"])
    async def ingest_url(request: IngestURLRequest):
        """Ingest and process content from URL, storing results in vector database."""
        try:
            # Auto-detect source type if not provided
            source_type = request.source_type
            if not source_type:
                source_type = get_morag_api()._detect_content_type(request.url)
                logger.info("Auto-detected content type for URL",
                           url=request.url,
                           detected_type=source_type)

            # Create task options with sanitized inputs
            options = {
                "webhook_url": request.webhook_url or "",  # Ensure string, not None
                "metadata": request.metadata or {},  # Ensure dict, not None
                "remote": request.remote,  # Remote processing flag
                "databases": request.databases,  # Database configurations for graph processing
                "store_in_vector_db": True  # Key difference from process endpoints
            }

            # Submit to ingest URL task for processing and storage
            task = ingest_url_task.delay(request.url, source_type, options)

            logger.info("URL ingestion task created",
                       task_id=task.id,
                       url=request.url,
                       source_type=request.source_type)

            return IngestResponse(
                task_id=task.id,
                status="pending",
                message=f"URL ingestion started for {request.url}",
                estimated_time=120  # URLs typically take longer
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
                "remote": request.remote,  # Remote processing flag
                "databases": request.databases,  # Database configurations for graph processing
                "store_in_vector_db": True  # Key difference from process endpoints
            }

            # Submit batch ingest task
            task = ingest_batch_task.delay(request.items, options)

            batch_id = f"batch-{uuid.uuid4().hex[:8]}"

            logger.info("Batch ingestion task created",
                       task_id=task.id,
                       batch_id=batch_id,
                       item_count=len(request.items))

            return BatchIngestResponse(
                batch_id=batch_id,
                task_ids=[task.id],  # Single task for the batch
                total_items=len(request.items),
                message=f"Batch ingestion started with {len(request.items)} items"
            )

        except Exception as e:
            logger.error("Batch ingestion failed", item_count=len(request.items), error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/ingest/remote-file", response_model=IngestResponse, tags=["Ingestion"])
    async def ingest_remote_file(request: IngestRemoteFileRequest):
        """Ingest and process a remote file (UNC path or HTTP/HTTPS URL), storing results in vector database."""
        temp_path = None
        try:
            # Get upload handler for temp directory
            upload_handler = get_upload_handler()

            # Download remote file to temp directory
            temp_path = await download_remote_file(request.file_path, upload_handler.temp_dir)

            # Auto-detect source type if not provided
            source_type = request.source_type
            if not source_type:
                source_type = get_morag_api()._detect_content_type_from_file(temp_path)
                logger.info("Auto-detected content type for remote file",
                           remote_path=request.file_path,
                           detected_type=source_type)

            logger.info("Remote file downloaded for ingestion",
                       remote_path=request.file_path,
                       temp_path=str(temp_path),
                       source_type=source_type)

            # Create task options with sanitized inputs
            options = {
                "document_id": request.document_id,
                "replace_existing": request.replace_existing,
                "webhook_url": request.webhook_url or "",  # Ensure string, not None
                "metadata": request.metadata or {},  # Ensure dict, not None
                "use_docling": request.use_docling,
                "chunk_size": request.chunk_size,
                "chunk_overlap": request.chunk_overlap,
                "chunking_strategy": request.chunking_strategy,
                "remote": request.remote,  # Remote processing flag
                "databases": request.databases,  # Database configurations for graph processing
                "store_in_vector_db": True  # Key difference from process endpoints
            }

            # Submit to background task queue for processing and storage
            task = ingest_file_task.delay(
                str(temp_path),
                source_type,
                options
            )

            logger.info("Remote file ingestion task created",
                       task_id=task.id,
                       remote_path=request.file_path,
                       source_type=source_type)

            return IngestResponse(
                task_id=task.id,
                status="pending",
                message=f"Remote file ingestion started for {request.file_path}",
                estimated_time=60  # Remote files may take longer due to download
            )

        except HTTPException:
            # Clean up temp file on HTTP errors
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 error=str(cleanup_error))
            raise
        except Exception as e:
            # Clean up temp file on other errors
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 error=str(cleanup_error))

            logger.error("Remote file ingestion failed",
                        remote_path=request.file_path,
                        error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/v1/process/remote-file", response_model=ProcessingResultResponse, tags=["Processing"])
    async def process_remote_file(request: ProcessRemoteFileRequest):
        """Process content from a remote file (UNC path or HTTP/HTTPS URL) without storing in vector database."""
        temp_path = None
        try:
            # Get upload handler for temp directory
            upload_handler = get_upload_handler()

            # Download remote file to temp directory
            temp_path = await download_remote_file(request.file_path, upload_handler.temp_dir)

            # Auto-detect content type if not provided
            content_type = request.content_type
            if not content_type:
                content_type = get_morag_api()._detect_content_type_from_file(temp_path)
                logger.info("Auto-detected content type for remote file",
                           remote_path=request.file_path,
                           detected_type=content_type)

            logger.info("Remote file downloaded for processing",
                       remote_path=request.file_path,
                       temp_path=str(temp_path),
                       content_type=content_type)

            # Submit to background task queue for processing (no storage)
            task = process_file_task.delay(
                str(temp_path),
                content_type,
                request.options
            )

            logger.info("Remote file processing task created",
                       task_id=task.id,
                       remote_path=request.file_path,
                       content_type=content_type)

            return ProcessingResultResponse(
                task_id=task.id,
                status="pending",
                message=f"Remote file processing started for {request.file_path}",
                estimated_time=60  # Remote files may take longer due to download
            )

        except HTTPException:
            # Clean up temp file on HTTP errors
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 error=str(cleanup_error))
            raise
        except Exception as e:
            # Clean up temp file on other errors
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.warning("Failed to clean up temp file after error",
                                 temp_path=str(temp_path),
                                 error=str(cleanup_error))

            logger.error("Remote file processing failed",
                        remote_path=request.file_path,
                        error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    # Task status endpoints
    @app.get("/api/v1/status/{task_id}", response_model=TaskStatus, tags=["Task Management"])
    async def get_task_status(task_id: str):
        """Get the status of a processing task."""
        try:
            # Get task result from Celery
            task_result = celery_app.AsyncResult(task_id)

            # Map Celery states to our API states
            status_mapping = {
                'PENDING': 'PENDING',
                'STARTED': 'PROGRESS',
                'PROGRESS': 'PROGRESS',
                'SUCCESS': 'SUCCESS',
                'FAILURE': 'FAILURE',
                'REVOKED': 'REVOKED'
            }

            status = status_mapping.get(task_result.state, task_result.state)

            # Get task info
            task_info = task_result.info or {}

            # Calculate progress
            progress = 0.0
            if status == 'SUCCESS':
                progress = 1.0
            elif status == 'PROGRESS' and isinstance(task_info, dict):
                progress = task_info.get('progress', 0.0)

            # Get result if completed successfully
            result = None
            error = None
            if status == 'SUCCESS':
                result = task_result.result
            elif status == 'FAILURE':
                error = str(task_result.info) if task_result.info else "Task failed"

            return TaskStatus(
                task_id=task_id,
                status=status,
                progress=progress,
                message=task_info.get('message') if isinstance(task_info, dict) else None,
                result=result,
                error=error
            )

        except Exception as e:
            logger.error("Failed to get task status", task_id=task_id, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/status/", tags=["Task Management"])
    async def list_active_tasks():
        """Get all currently active tasks."""
        try:
            # Get active tasks from Celery
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()

            if not active_tasks:
                return {"active_tasks": [], "count": 0}

            # Extract task IDs from all workers
            task_ids = []
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task_ids.append(task['id'])

            return {
                "active_tasks": task_ids,
                "count": len(task_ids)
            }

        except Exception as e:
            logger.error("Failed to list active tasks", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/v1/status/stats/queues", tags=["Task Management"])
    async def get_queue_stats():
        """Get processing queue statistics."""
        try:
            # Get queue stats from Celery
            inspect = celery_app.control.inspect()

            # Get active tasks
            active_tasks = inspect.active() or {}
            active_count = sum(len(tasks) for tasks in active_tasks.values())

            # Get scheduled tasks
            scheduled_tasks = inspect.scheduled() or {}
            pending_count = sum(len(tasks) for tasks in scheduled_tasks.values())

            # Note: Celery doesn't provide completed/failed counts directly
            # These would need to be tracked separately in a database

            return {
                "pending": pending_count,
                "active": active_count,
                "completed": 0,  # Would need separate tracking
                "failed": 0      # Would need separate tracking
            }

        except Exception as e:
            logger.error("Failed to get queue stats", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/v1/ingest/{task_id}", tags=["Task Management"])
    async def cancel_task(task_id: str):
        """Cancel a running or pending task."""
        try:
            # Revoke the task
            celery_app.control.revoke(task_id, terminate=True)

            logger.info("Task cancelled", task_id=task_id)

            return {
                "message": f"Task {task_id} cancelled successfully"
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

    # Include routers
    app.include_router(remote_jobs_router)

    # Include enhanced query router (v2 API)
    try:
        from morag.endpoints import enhanced_query_router
        app.include_router(enhanced_query_router)
        logger.info("Enhanced query API endpoints loaded")
    except ImportError as e:
        logger.warning("Enhanced query endpoints not available", error=str(e))

    # Include intelligent retrieval router (v2 API)
    try:
        from morag.endpoints.intelligent_retrieval import router as intelligent_retrieval_router
        app.include_router(intelligent_retrieval_router)
        logger.info("Intelligent retrieval API endpoints loaded")
    except ImportError as e:
        logger.warning("Intelligent retrieval endpoints not available", error=str(e))

    # Legacy router temporarily disabled
    # try:
    #     from morag.endpoints import legacy_router
    #     app.include_router(legacy_router)
    #     logger.info("Legacy API endpoints loaded")
    # except ImportError as e:
    #     logger.warning("Legacy endpoints not available", error=str(e))

    # Include reasoning router (multi-hop reasoning API)
    try:
        from morag.endpoints.reasoning import router as reasoning_router
        app.include_router(reasoning_router, prefix="/api/v2")
        logger.info("Multi-hop reasoning API endpoints loaded")
    except ImportError as e:
        logger.warning("Multi-hop reasoning endpoints not available", error=str(e))

    # Include recursive fact retrieval router (recursive fact retrieval API)
    try:
        from morag.endpoints.recursive_fact_retrieval import router as recursive_fact_retrieval_router
        app.include_router(recursive_fact_retrieval_router)
        logger.info("Recursive fact retrieval API endpoints loaded")
    except ImportError as e:
        logger.warning("Recursive fact retrieval endpoints not available", error=str(e))

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
