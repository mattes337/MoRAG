"""Processing endpoints for MoRAG API."""

from typing import Optional
from pathlib import Path
import structlog

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse

from morag.api import MoRAGAPI
from morag.api.models import ProcessURLRequest, ProcessBatchRequest, ProcessingResultResponse, ProcessRemoteFileRequest
from morag.api.utils import download_remote_file, normalize_content_type, normalize_processing_result, encode_thumbnails_to_base64
from morag.utils.file_upload import get_upload_handler, FileUploadError

logger = structlog.get_logger(__name__)

processing_router = APIRouter(prefix="/process", tags=["Processing"])


def setup_processing_endpoints(morag_api_getter):
    """Setup processing endpoints with MoRAG API getter function."""
    
    @processing_router.post("/url", response_model=ProcessingResultResponse)
    async def process_url(request: ProcessURLRequest):
        """Process content from a URL."""
        try:
            result = await morag_api_getter().process_url(
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
    
    @processing_router.post("/file", response_model=ProcessingResultResponse)
    async def process_file(
        file: UploadFile = File(...),
        content_type: Optional[str] = Form(default=None),
        options: Optional[str] = Form(default=None)  # JSON string
    ):
        """Process content from an uploaded file."""
        temp_path = None
        try:
            # Get upload handler
            upload_handler = get_upload_handler()
            
            # Save uploaded file to temp directory
            temp_path = await upload_handler.save_upload(file)
            
            # Parse options if provided
            parsed_options = None
            if options:
                import json
                try:
                    parsed_options = json.loads(options)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON in options parameter", options=options, error=str(e))
                    parsed_options = {}
            
            # Normalize content type
            normalized_content_type = normalize_content_type(content_type or file.content_type)
            
            # Process the file
            result = await morag_api_getter().process_file(
                str(temp_path),
                normalized_content_type,
                parsed_options
            )
            
            # Normalize result for API response
            result = normalize_processing_result(result)
            
            # Handle thumbnails if present
            thumbnails = []
            if hasattr(result, 'thumbnails') and result.thumbnails:
                thumbnails = encode_thumbnails_to_base64(result.thumbnails)
            
            return ProcessingResultResponse(
                success=result.success,
                content=result.content,
                metadata=result.metadata,
                processing_time=result.processing_time,
                error_message=result.error_message,
                thumbnails=thumbnails
            )
            
        except FileUploadError as e:
            logger.error("File upload failed", filename=file.filename, error=str(e))
            raise HTTPException(status_code=400, detail=f"File upload failed: {str(e)}")
        except Exception as e:
            logger.error("File processing failed", filename=file.filename, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug("Cleaned up temporary file", temp_path=str(temp_path))
                except Exception as e:
                    logger.warning("Failed to clean up temporary file",
                                 temp_path=str(temp_path),
                                 error=str(e))
    
    @processing_router.post("/web", response_model=ProcessingResultResponse)
    async def process_web_page(request: ProcessURLRequest):
        """Process a web page."""
        try:
            result = await morag_api_getter().process_web_page(request.url, request.options)
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
    
    @processing_router.post("/youtube", response_model=ProcessingResultResponse)
    async def process_youtube_video(request: ProcessURLRequest):
        """Process a YouTube video."""
        try:
            result = await morag_api_getter().process_youtube_video(request.url, request.options)
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
    
    @processing_router.post("/batch")
    async def process_batch(request: ProcessBatchRequest):
        """Process multiple items in batch."""
        try:
            results = await morag_api_getter().process_batch(request.items, request.options)
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
    
    @processing_router.post("/remote-file", response_model=ProcessingResultResponse)
    async def process_remote_file(request: ProcessRemoteFileRequest):
        """Process content from a remote file (UNC path or HTTP/HTTPS URL) without storing in vector database."""
        temp_path = None
        try:
            # Get upload handler for temp directory
            upload_handler = get_upload_handler()
            temp_dir = upload_handler.temp_dir
            
            # Download remote file to temp directory
            temp_path = await download_remote_file(request.file_path, temp_dir)
            
            # Normalize content type
            normalized_content_type = normalize_content_type(request.content_type)
            
            # Process the file
            result = await morag_api_getter().process_file(
                str(temp_path),
                normalized_content_type,
                request.options
            )
            
            # Normalize result for API response
            result = normalize_processing_result(result)
            
            # Handle thumbnails if present
            thumbnails = []
            if hasattr(result, 'thumbnails') and result.thumbnails:
                thumbnails = encode_thumbnails_to_base64(result.thumbnails)
            
            return ProcessingResultResponse(
                success=result.success,
                content=result.content,
                metadata=result.metadata,
                processing_time=result.processing_time,
                error_message=result.error_message,
                thumbnails=thumbnails
            )
            
        except Exception as e:
            logger.error("Remote file processing failed",
                        remote_path=request.file_path,
                        error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Clean up temporary file
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                    logger.debug("Cleaned up temporary file", temp_path=str(temp_path))
                except Exception as e:
                    logger.warning("Failed to clean up temporary file",
                                 temp_path=str(temp_path),
                                 error=str(e))

    return processing_router
