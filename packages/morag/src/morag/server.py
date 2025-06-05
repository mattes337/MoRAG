"""FastAPI server for MoRAG system."""

import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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


def create_app(config: Optional[ServiceConfig] = None) -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="MoRAG API",
        description="Modular Retrieval Augmented Generation System",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize MoRAG API
    morag_api = MoRAGAPI(config)
    
    @app.on_event("startup")
    async def startup_event():
        """Startup event handler."""
        logger.info("MoRAG API server starting up")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Shutdown event handler."""
        await morag_api.cleanup()
        logger.info("MoRAG API server shut down")
    
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

                return ProcessingResultResponse(
                    success=result.success,
                    content=result.content,
                    metadata=result.metadata,
                    processing_time=result.processing_time,
                    error_message=result.error_message
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
                    content=result.content,
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
