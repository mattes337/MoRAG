"""FastAPI server for MoRAG system."""

import asyncio
import base64
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import structlog

from morag.api import MoRAGAPI
from morag_services import ServiceConfig
from morag_core.models import ProcessingResult
from morag.utils.file_upload import get_upload_handler, FileUploadError

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

    # If result already has content attribute, return as-is
    if hasattr(result, 'content') and result.content is not None:
        return result

    # Get content from text_content or set empty string
    content = ""
    if hasattr(result, 'text_content') and result.text_content is not None:
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


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Create FastAPI application."""

    # Initialize MoRAG API outside of lifespan for access in routes
    morag_api = MoRAGAPI(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup and shutdown."""
        # Startup
        logger.info("MoRAG API server starting up")
        yield
        # Shutdown
        await morag_api.cleanup()
        logger.info("MoRAG API server shut down")

    app = FastAPI(
        title="MoRAG API",
        description="Modular Retrieval Augmented Generation System",
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
            status = await morag_api.health_check()
            return JSONResponse(content=status)
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/process/url", response_model=ProcessingResultResponse)
    async def process_url(request: ProcessURLRequest):
        """Process content from a URL."""
        try:
            result = await morag_api.process_url(
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
    
    @app.post("/process/file", response_model=ProcessingResultResponse)
    async def process_file(
        file: UploadFile = File(...),
        content_type: Optional[str] = Form(None),
        options: Optional[str] = Form(None)  # JSON string
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

                result = await morag_api.process_file(
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
    
    @app.post("/process/web", response_model=ProcessingResultResponse)
    async def process_web_page(request: ProcessURLRequest):
        """Process a web page."""
        try:
            result = await morag_api.process_web_page(request.url, request.options)
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
    
    @app.post("/process/youtube", response_model=ProcessingResultResponse)
    async def process_youtube_video(request: ProcessURLRequest):
        """Process a YouTube video."""
        try:
            result = await morag_api.process_youtube_video(request.url, request.options)
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
    
    @app.post("/process/batch")
    async def process_batch(request: ProcessBatchRequest):
        """Process multiple items in batch."""
        try:
            results = await morag_api.process_batch(request.items, request.options)
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
    
    @app.post("/search")
    async def search_similar(request: SearchRequest):
        """Search for similar content."""
        try:
            results = await morag_api.search(
                request.query,
                request.limit,
                request.filters
            )
            return {"results": results}
        except Exception as e:
            logger.error("Search failed", query=request.query, error=str(e))
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
